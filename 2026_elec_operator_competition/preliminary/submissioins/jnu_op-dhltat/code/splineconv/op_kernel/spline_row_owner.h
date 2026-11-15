/*
 * Copyright (c) 2026 dhltat.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "spline_device.h"

// General degree/dimension implementation: each core owns disjoint output
// rows. Optional private edge lists are built and consumed by that same core.
template <class T, uint32_t MODE, uint32_t DIM, uint32_t DEG> class KernelSplineConv {
    LocalTensor<float> weights, acc, slot, xlocal, product, spread, rows;
    LocalTensor<int32_t> counts;
    LocalTensor<int64_t> graphControls;
    LocalTensor<T> raw;
    GlobalTensor<T> x, a, w, out;
    GlobalTensor<int64_t> edge;
    GlobalTensor<uint16_t> xb, ab;
    GlobalTensor<int64_t> ownedEdges;
    SplineConvTilingData t;
    __aicore__ inline void LoadWeight(uint32_t k, uint32_t dest) {
        Fence<HardEvent::V_MTE2>();
        if constexpr (sizeof(T) == 4) {
            DataCopyPad(weights[dest], w[uint64_t(k) * t.inputChannels * t.outputChannels],
                        {uint16_t(t.inputChannels), uint32_t(t.outputChannels * 4), 0, 0, 0},
                        {false, 0, 0, 0});
        } else {
            DataCopyPad(raw, w[uint64_t(k) * t.inputChannels * t.outputChannels],
                        {uint16_t(t.inputChannels), uint32_t(t.outputChannels * 2), 0, 0, 0},
                        {false, 0, 0, 0});
            Fence<HardEvent::MTE2_V>();
            Cast(weights[dest], raw, RoundMode::CAST_NONE, t.inputChannels * t.alignedOutputChannels);
        }
        Fence<HardEvent::MTE2_V>();
    }
    // Cached buffers already fit the Host UB budget, hence a contiguous
    // byte count fits DataCopyPad's21-bit blockLen. A2 blockCount has12bits:
    // 4096rows cannot be encoded as one multi-row transfer.
    __aicore__ inline void CopyCachedWeightRows(uint32_t knotCount) {
        const uint32_t rowCount = knotCount * t.inputChannels;
        if (t.alignedOutputChannels == t.outputChannels) {
            if constexpr (sizeof(T) == 4)
                DataCopyPad(weights, w, {1, uint32_t(rowCount * t.outputChannels * 4), 0, 0, 0},
                            {false, 0, 0, 0});
            else
                DataCopyPad(raw, w, {1, uint32_t(rowCount * t.outputChannels * 2), 0, 0, 0},
                            {false, 0, 0, 0});
            return;
        }
        constexpr uint32_t maxRowsPerTransfer = 4095;
        for (uint32_t begin = 0; begin < rowCount; begin += maxRowsPerTransfer) {
            const uint32_t count = rowCount - begin < maxRowsPerTransfer
                                       ? rowCount - begin : maxRowsPerTransfer;
            if constexpr (sizeof(T) == 4)
                DataCopyPad(weights[begin * t.alignedOutputChannels], w[uint64_t(begin) * t.outputChannels],
                            {uint16_t(count), uint32_t(t.outputChannels * 4), 0, 0, 0}, {false, 0, 0, 0});
            else
                DataCopyPad(raw[begin * t.alignedOutputChannels], w[uint64_t(begin) * t.outputChannels],
                            {uint16_t(count), uint32_t(t.outputChannels * 2), 0, 0, 0}, {false, 0, 0, 0});
        }
    }
    __aicore__ inline void LoadAll(uint32_t ktotal) {
        if constexpr (sizeof(T) == 4) {
            if (t.contractKnots)
                Fence<HardEvent::V_MTE2>();
            CopyCachedWeightRows(ktotal);
            Fence<HardEvent::MTE2_V>();
        } else {
            Fence<HardEvent::V_MTE2>();
            CopyCachedWeightRows(ktotal);
            Fence<HardEvent::MTE2_V>();
            Cast(weights, raw, RoundMode::CAST_NONE, ktotal * t.inputChannels * t.alignedOutputChannels);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline float Basis(float v, uint32_t j) {
        if constexpr (DEG == 1)
            return j == 0 ? 1 - v : v;
        if constexpr (DEG == 2)
            return j == 0 ? 0.5f * v * v - v + 0.5f : (j == 1 ? -v * v + v + 0.5f : 0.5f * v * v);
        float v2 = v * v, v3 = v2 * v;
        if (j == 0) {
            float z = 1 - v;
            return z * z * z * (1.0f / 6);
        }
        if (j == 1)
            return (3 * v3 - 6 * v2 + 4) * (1.0f / 6);
        if (j == 2)
            return (-3 * v3 + 3 * v2 + 3 * v + 1) * (1.0f / 6);
        return v3 * (1.0f / 6);
    }
    __aicore__ inline float Scalar(GlobalTensor<T> tensor, GlobalTensor<uint16_t> bits, uint64_t i) {
        if constexpr (sizeof(T) == 4)
            return tensor.GetValue(i);
        else {
            union {
                uint32_t u;
                float f;
            } v;
            v.u = uint32_t(bits.GetValue(i)) << 16;
            return v.f;
        }
    }
    __aicore__ inline void LoadX(uint32_t n) {
        Fence<HardEvent::V_MTE2>();
        if constexpr (sizeof(T) == 4) {
            DataCopyPad(xlocal, x[uint64_t(n) * t.inputChannels], {1, uint32_t(t.inputChannels * 4), 0, 0, 0},
                        {false, 0, 0, 0});
            Fence<HardEvent::MTE2_V>();
        } else {
            DataCopyPad(raw, x[uint64_t(n) * t.inputChannels], {1, uint32_t(t.inputChannels * 2), 0, 0, 0},
                        {false, 0, 0, 0});
            Fence<HardEvent::MTE2_V>();
            Cast(xlocal, raw, RoundMode::CAST_NONE, t.inputChannels);
            PipeBarrier<PIPE_V>();
        }
    }
    // A full FP32 vector is 64 elements. Set both mask words explicitly;
    // only FP32 DMA or already-cached weights may preserve this mask across
    // supports. Streaming BF16 Cast changes it and uses the count fallback.
    // Both versions visit all supports in the original supportCoefficient/FMA order.
    template <bool reuseFullVectorMask>
    __aicore__ inline void AccumulateKnotSupport(const uint32_t *axisBaseIndices, const float *axisBasisValues,
                                                uint32_t matrixElementCount, uint8_t fullVectorRepeatCount) {
        if constexpr (reuseFullVectorMask) {
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(uint64_t(0), ~uint64_t(0));
        }
#pragma unroll 1
        for (uint32_t supportIndex = 0; supportIndex < ((DEG + 1) * (DEG + 1) * (DIM == 3 ? DEG + 1 : 1)); ++supportIndex) {
            uint32_t remainingSupportDigits = supportIndex, supportKnotIndex = 0, axisKnotStride = 1;
            float supportCoefficient = 1;
            for (uint32_t axis = 0; axis < DIM; ++axis) {
                uint32_t axisBasisIndex = remainingSupportDigits % (DEG + 1);
                remainingSupportDigits /= DEG + 1;
                supportKnotIndex += ((axisBaseIndices[axis] + axisBasisIndex) % t.knotSizes[axis]) * axisKnotStride;
                axisKnotStride *= t.knotSizes[axis];
                supportCoefficient *= axisBasisValues[axis * 4 + axisBasisIndex];
            }
            uint32_t weightOffsetElements = t.cacheWeights ? supportKnotIndex * matrixElementCount : 0;
            if (!t.cacheWeights)
                LoadWeight(supportKnotIndex, 0);
            if constexpr (reuseFullVectorMask)
                Axpy<float, float, false>(product, weights[weightOffsetElements], supportCoefficient,
                                          MASK_PLACEHOLDER, fullVectorRepeatCount, {1, 1, 8, 8});
            else
                Axpy(product, weights[weightOffsetElements], supportCoefficient, matrixElementCount);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Gemv(LocalTensor<float> matrix) {
        Brcb(spread, xlocal, (t.inputChannels + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        for (uint32_t c = 0; c < t.alignedOutputChannels; c += 64) {
            uint32_t count = t.alignedOutputChannels - c < 64 ? t.alignedOutputChannels - c : 64;
            Mul(product[c], matrix[c], spread, uint64_t(count), uint8_t(t.inputChannels),
                {1, 1, 0, uint8_t(t.alignedOutputChannels / 8), uint8_t(t.alignedOutputChannels / 8), 1});
        }
        PipeBarrier<PIPE_V>();
        uint32_t rows = t.inputChannels;
        while (rows > 1) {
            uint32_t half = (rows + 1) / 2;
            Add(product, product, product[half * t.alignedOutputChannels],
                (rows - half) * t.alignedOutputChannels);
            PipeBarrier<PIPE_V>();
            rows = half;
        }
        if constexpr (MODE == kGlobalAccumulationMode) {
            Adds(slot, product, 0.0f, t.outputChannels);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void WriteNode(uint32_t n, float count, LocalTensor<float> result) {
        if (t.applyMean && count > 1) {
            Muls(result, result, 1 / count, t.outputChannels);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (sizeof(T) == 4) {
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(out[uint64_t(n) * t.outputChannels], result,
                        {1, uint32_t(t.outputChannels * 4), 0, 0, 0});
        } else {
            Cast(raw, result, RoundMode::CAST_RINT, t.outputChannels);
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(out[uint64_t(n) * t.outputChannels], raw,
                        {1, uint32_t(t.outputChannels * 2), 0, 0, 0});
        }
        Fence<HardEvent::MTE3_V>();
    }
    __aicore__ inline void LoadMessageInput(uint32_t src, uint32_t ktotal, bool &ready) {
        if constexpr (sizeof(T) == 4) {
            Fence<HardEvent::V_MTE2>();
            if (t.cacheWeights && !ready) {
                CopyCachedWeightRows(ktotal);
                ready = true;
            }
            DataCopyPad(xlocal, x[uint64_t(src) * t.inputChannels],
                        {1, uint32_t(t.inputChannels * 4), 0, 0, 0}, {false, 0, 0, 0});
        } else {
            if (t.cacheWeights && !ready) {
                LoadAll(ktotal);
                ready = true;
            }
            LoadX(src);
        }
    }

    __aicore__ inline void Message(uint32_t e, uint32_t ktotal, bool &ready) {
        uint32_t src = edge.GetValue(e), base[3];
        float b[12];
        LoadMessageInput(src, ktotal, ready);
        for (uint32_t d = 0; d < DIM; ++d) {
            float v = Scalar(a, ab, e * DIM + d) * t.coordinateScales[d];
            base[d] = int32_t(v);
            v -= int32_t(base[d]);
            for (uint32_t j = 0; j <= DEG; ++j)
                b[d * 4 + j] = Basis(v, j);
        }
        if constexpr (sizeof(T) == 4) {
            Fence<HardEvent::MTE2_V>();
        }
        // This direct-owner helper uses slot as its support accumulator.
        // Multiply x by each real knot matrix before applying its basis value.
        // Interpolating W first can overflow at max_float even when x@W and
        // the required output are finite. The original global path uses this
        // same stable product-before-basis order.
        Duplicate(slot, 0.0f, t.outputChannels);
        PipeBarrier<PIPE_V>();
#pragma unroll 1
        for (uint32_t supportIndex = 0; supportIndex < ((DEG + 1) * (DEG + 1) * (DIM == 3 ? DEG + 1 : 1)); ++supportIndex) {
            uint32_t remainingSupportDigits = supportIndex, supportKnotIndex = 0, axisKnotStride = 1;
            float supportCoefficient = 1;
            for (uint32_t axis = 0; axis < DIM; ++axis) {
                uint32_t axisBasisIndex = remainingSupportDigits % (DEG + 1);
                remainingSupportDigits /= DEG + 1;
                supportKnotIndex += ((base[axis] + axisBasisIndex) % t.knotSizes[axis]) * axisKnotStride;
                axisKnotStride *= t.knotSizes[axis];
                supportCoefficient *= b[axis * 4 + axisBasisIndex];
            }
            uint32_t weightOffsetElements =
                t.cacheWeights ? supportKnotIndex * t.inputChannels * t.alignedOutputChannels : 0;
            if (!t.cacheWeights)
                LoadWeight(supportKnotIndex, 0);
            // MODE1 Gemv returns its reduced row in product and leaves slot
            // untouched. Cached weights are never overwritten, so ready may
            // retain them across later messages owned by this worker.
            Gemv(weights[weightOffsetElements]);
            Axpy(slot, product, supportCoefficient, t.outputChannels);
            PipeBarrier<PIPE_V>();
        }
        Adds(product, slot, 0.0f, t.outputChannels);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Direct(uint32_t ktotal) {
        bool ready = false;
        for (uint32_t n = GetBlockIdx(); n < t.nodeCount; n += t.workerCount) {
            uint32_t count = 0;
            for (uint32_t e = 0; e < t.edgeCount; ++e) {
                if (uint32_t(edge.GetValue(t.edgeCount + e)) != n)
                    continue;
                if (count == 1) {
                    Adds(acc, product, 0.0f, t.outputChannels);
                    PipeBarrier<PIPE_V>();
                }
                ++count;
                Message(e, ktotal, ready);
                if (count > 1) {
                    Add(acc, acc, product, t.outputChannels);
                    PipeBarrier<PIPE_V>();
                }
            }
            if (count == 0) {
                Duplicate(acc, 0.0f, t.outputChannels);
                PipeBarrier<PIPE_V>();
            }
            WriteNode(n, int32_t(count), count == 1 ? product : acc);
        }
    }
    __aicore__ inline uint32_t BuildOwnedEdgeList(uint32_t ownerBeginRow, uint32_t ownerEndRow) {
        constexpr uint32_t batchEntries = kOwnedDestinationBatchEntries;
        LocalTensor<int64_t> destinations = graphControls;
        LocalTensor<int64_t> stagedMatches = graphControls[batchEntries];
        auto destinationValues = (__ubuf__ int64_t *)destinations.GetPhyAddr();
        auto matchValues = (__ubuf__ int64_t *)stagedMatches.GetPhyAddr();
        uint32_t storedEntries = 0, bufferedEntries = 0;
        for (uint32_t edgeBegin = 0; edgeBegin < t.edgeCount; edgeBegin += batchEntries) {
            const uint32_t actualEntries = t.edgeCount - edgeBegin < batchEntries
                                               ? t.edgeCount - edgeBegin : batchEntries;
            Fence<HardEvent::S_MTE2>();
            DataCopyPad(destinations, edge[t.edgeCount + edgeBegin],
                        {1, uint32_t(actualEntries * sizeof(int64_t)), 0, 0, 0}, {false, 0, 0, 0});
            Fence<HardEvent::MTE2_S>();
            for (uint32_t position = 0; position < actualEntries; ++position) {
                const uint32_t destinationRow = uint32_t(destinationValues[position]);
                if (destinationRow < ownerBeginRow || destinationRow >= ownerEndRow)
                    continue;
                matchValues[bufferedEntries++] = int64_t((uint64_t(destinationRow) << 32) | (edgeBegin + position));
                if (bufferedEntries == batchEntries) {
                    Fence<HardEvent::S_MTE3>();
                    DataCopy(ownedEdges[storedEntries], stagedMatches, batchEntries);
                    Fence<HardEvent::MTE3_S>();
                    storedEntries += bufferedEntries;
                    bufferedEntries = 0;
                }
            }
        }
        if (bufferedEntries != 0) {
            Fence<HardEvent::S_MTE3>();
            DataCopyPad(ownedEdges[storedEntries], stagedMatches,
                        {1, uint32_t(bufferedEntries * sizeof(int64_t)), 0, 0, 0});
            Fence<HardEvent::MTE3_S>();
            storedEntries += bufferedEntries;
        }
        Fence<HardEvent::MTE3_MTE2>();
        return storedEntries;
    }
    __aicore__ inline void AccumulateOwnedEdge(uint32_t edgeIndex, uint32_t destinationRow,
                                              uint32_t tileBeginRow, uint32_t knotCount, bool &weightsReady) {
        const uint32_t localRow = destinationRow - tileBeginRow;
        Message(edgeIndex, knotCount, weightsReady);
        Add(rows[localRow * t.alignedOutputChannels], rows[localRow * t.alignedOutputChannels],
            product, t.outputChannels);
        PipeBarrier<PIPE_V>();
        ((__ubuf__ int32_t *)counts.GetPhyAddr())[localRow] += 1;
    }
    __aicore__ inline void StoreOwnedRows(uint32_t tileBeginRow, uint32_t rowCount) {
        auto incomingCounts = (__ubuf__ int32_t *)counts.GetPhyAddr();
        if constexpr (sizeof(T) == 4) {
            if (t.applyMean) {
                for (uint32_t localRow = 0; localRow < rowCount; ++localRow) {
                    const uint32_t incomingCount = uint32_t(incomingCounts[localRow]);
                    if (incomingCount > 1)
                        Muls(rows[localRow * t.alignedOutputChannels], rows[localRow * t.alignedOutputChannels],
                             1.0f / incomingCount, t.outputChannels);
                }
            }
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(out[uint64_t(tileBeginRow) * t.outputChannels], rows,
                        {uint16_t(rowCount), uint32_t(t.outputChannels * sizeof(float)), 0, 0, 0});
            Fence<HardEvent::MTE3_V>();
        } else {
            for (uint32_t localRow = 0; localRow < rowCount; ++localRow)
                WriteNode(tileBeginRow + localRow, incomingCounts[localRow],
                          rows[localRow * t.alignedOutputChannels]);
        }
    }
    __aicore__ inline void AccumulateBucketedTile(uint32_t tileBeginRow, uint32_t tileEndRow,
                                                 uint32_t ownedEdgeCount, uint32_t knotCount, bool &weightsReady) {
        constexpr uint32_t batchEntries = kOwnedControlBufferBytes / sizeof(int64_t);
        auto packedEntries = (__ubuf__ int64_t *)graphControls.GetPhyAddr();
        for (uint32_t edgeBegin = 0; edgeBegin < ownedEdgeCount; edgeBegin += batchEntries) {
            const uint32_t remaining = ownedEdgeCount - edgeBegin;
            const uint32_t actualEntries = remaining < batchEntries ? remaining : batchEntries;
            Fence<HardEvent::S_MTE2>();
            DataCopyPad(graphControls, ownedEdges[edgeBegin],
                        {1, uint32_t(actualEntries * sizeof(int64_t)), 0, 0, 0}, {false, 0, 0, 0});
            Fence<HardEvent::MTE2_S>();
            for (uint32_t position = 0; position < actualEntries; ++position) {
                const uint64_t entry = uint64_t(packedEntries[position]);
                const uint32_t destinationRow = entry >> 32;
                if (destinationRow >= tileBeginRow && destinationRow < tileEndRow) {
                    AccumulateOwnedEdge(uint32_t(entry), destinationRow, tileBeginRow, knotCount, weightsReady);
                }
            }
        }
    }

    __aicore__ inline void RowDirect(uint32_t knotCount, GM_ADDR scratch) {
        const uint32_t rowsPerOwner = (t.nodeCount + t.workerCount - 1) / t.workerCount;
        const uint32_t ownerBeginRow = GetBlockIdx() * rowsPerOwner;
        const uint32_t ownerEndRow = ownerBeginRow + rowsPerOwner < t.nodeCount
                                          ? ownerBeginRow + rowsPerOwner : t.nodeCount;
        if (ownerBeginRow >= ownerEndRow)
            return;
        uint32_t ownedEdgeCount = 0;
        if (t.usePrivateEdgeBuckets) {
            const uint64_t entriesPerOwner =
                (uint64_t(t.edgeCount) + kOwnedEdgeWorkspaceAlignmentEntries - 1) /
                kOwnedEdgeWorkspaceAlignmentEntries * kOwnedEdgeWorkspaceAlignmentEntries;
            ownedEdges.SetGlobalBuffer((__gm__ int64_t *)scratch + GetBlockIdx() * entriesPerOwner);
            ownedEdgeCount = BuildOwnedEdgeList(ownerBeginRow, ownerEndRow);
        }
        bool weightsReady = false;
        for (uint32_t tileBeginRow = ownerBeginRow; tileBeginRow < ownerEndRow; tileBeginRow += t.rowCapacity) {
            const uint32_t rowCount = ownerEndRow - tileBeginRow < t.rowCapacity
                                           ? ownerEndRow - tileBeginRow : t.rowCapacity;
            const uint32_t tileEndRow = tileBeginRow + rowCount;
            Fence<HardEvent::S_V>();
            Duplicate(rows, 0.0f, rowCount * t.alignedOutputChannels);
            Duplicate(counts, int32_t(0), rowCount);
            Fence<HardEvent::V_S>();
            if (t.usePrivateEdgeBuckets) {
                AccumulateBucketedTile(tileBeginRow, tileEndRow, ownedEdgeCount, knotCount, weightsReady);
            } else {
                for (uint32_t edgeIndex = 0; edgeIndex < t.edgeCount; ++edgeIndex) {
                    const uint32_t destinationRow = uint32_t(edge.GetValue(t.edgeCount + edgeIndex));
                    if (destinationRow >= tileBeginRow && destinationRow < tileEndRow)
                        AccumulateOwnedEdge(edgeIndex, destinationRow, tileBeginRow, knotCount, weightsReady);
                }
            }
            StoreOwnedRows(tileBeginRow, rowCount);
        }
    }

    // Assigns each direct-owner buffer once. Counts are elements; the allocator
    // advances an aligned byte offset. Host UB checks include this same capacity.
    __aicore__ inline void AllocateDirectBuffers(uint32_t knotCount) {
        uint32_t offsetBytes = 0;
        xlocal = DirectUbAlloc<float>(offsetBytes, 128);
        weights = DirectUbAlloc<float>(offsetBytes, (t.cacheWeights ? knotCount : 1) * t.inputChannels *
                                                        t.alignedOutputChannels);
        raw = DirectUbAlloc<T>(offsetBytes, sizeof(T) == 4 ? 8
                                                           : (t.cacheWeights ? knotCount : 1) *
                                                                 t.inputChannels * t.alignedOutputChannels);
        product = DirectUbAlloc<float>(offsetBytes, (t.contractKnots ? knotCount : 1) * t.inputChannels *
                                                        t.alignedOutputChannels);
        spread = DirectUbAlloc<float>(offsetBytes, 1024);
        acc = DirectUbAlloc<float>(offsetBytes, t.scratchRowElements);
        slot = DirectUbAlloc<float>(offsetBytes, t.alignedOutputChannels);
        if (t.directRowIntervals) {
            rows = DirectUbAlloc<float>(offsetBytes, t.rowCapacity * t.alignedOutputChannels);
            uint32_t metadataCapacityRows = (t.rowCapacity + 7) / 8 * 8;
            counts = DirectUbAlloc<int32_t>(offsetBytes, metadataCapacityRows);
            graphControls = DirectUbAlloc<int64_t>(offsetBytes, kOwnedControlBufferBytes / sizeof(int64_t));
        }
    }

  public:
    __aicore__ inline void Run(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp, GM_ADDR scratch,
                               const SplineConvTilingData &data) {
        // Padded launch lanes own no rows or workspace. Return before any
        // graph transfer, UB allocation, or output initialization.
        if (GetBlockIdx() >= data.workerCount)
            return;
        t = data;
        xb.SetGlobalBuffer((__gm__ uint16_t *)xp);
        ab.SetGlobalBuffer((__gm__ uint16_t *)ap);
        x.SetGlobalBuffer((__gm__ T *)xp);
        a.SetGlobalBuffer((__gm__ T *)ap);
        w.SetGlobalBuffer((__gm__ T *)wp);
        out.SetGlobalBuffer((__gm__ T *)yp);
        edge.SetGlobalBuffer((__gm__ int64_t *)ep);
        uint32_t ktotal = 1;
        for (uint32_t d = 0; d < DIM; ++d)
            ktotal *= t.knotSizes[d];
        AllocateDirectBuffers(ktotal);
        if (t.directRowIntervals)
            RowDirect(ktotal, scratch);
        else
            Direct(ktotal);
    }

};
