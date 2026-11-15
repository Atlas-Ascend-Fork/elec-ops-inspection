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
#include "spline_sparse_math.h"

template <class T, uint32_t CI, uint32_t CO, bool Coowned, bool CacheFeatures = false,
          uint32_t StaticEdgeCount = 0, bool Paired = false>
struct SparseGraphWorker {
    static constexpr uint32_t inputChannels = CI, matrixElements = CI * CO;
    // At least16 FP32 vector repeats per matrix amortize four packet tokens.
    // The existing complete owner group and UB budget remain unchanged.
    static constexpr bool streamWeightStages = sizeof(T) != 4 && Coowned && StaticEdgeCount == 8 && !CacheFeatures &&
        matrixElements >= 16 * kFloatVectorElements && matrixElements <= 2048 && CO * sizeof(T) >= 32;
    uint32_t worker, offsetBytes, nodeCount, edgeCount, tileRows, target, incomingCount, begin, end;
    uint64_t packed, occupiedRows;
    bool leader, mean, weightPrefetched, messagePrepared;
    LocalTensor<int64_t> indices;
    LocalTensor<uint64_t> metadataLocal;
    LocalTensor<T> coords, rawFeature, rawWeight, rawOutput, zero, cachedInput;
    LocalTensor<float> feature, weights, result, broadcast, accumulator, messageSum;
    GlobalTensor<T> x, weightsGm, coordsGm, outputGm;
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<uint64_t> metadata;
    __aicore__ inline void Allocate() {
        if constexpr (Paired) {
            static_assert(sizeof(T) == 4 && CO >= 16 && CO <= 32 &&
                kFloatVectorElements % CO == 0 && (CI & (CI - 1)) == 0 &&
                uint64_t(4) * matrixElements * sizeof(float) >= kPairedProductMinimumBytes &&
                uint64_t(4) * matrixElements * sizeof(float) <= kPairedProductMaximumBytes,
                "Host-selected complete paired-product resource family");
        }
        worker = GetBlockIdx();
        offsetBytes = 2048;
        indices = LocalTensor<int64_t>(TPosition::VECCALC, 0, 128);
        coords = LocalTensor<T>(TPosition::VECCALC, 1024, 128);
        metadataLocal = LocalTensor<uint64_t>(TPosition::VECCALC, 1536, 4);
        constexpr uint32_t productGapBytes = 4 * kUbAlignmentBytes;
        constexpr uint32_t trailingGapBytes = Paired ? kUbAlignmentBytes : 0;
        constexpr uint32_t broadcastGapBytes = 5 * kUbAlignmentBytes - trailingGapBytes;
        feature = SparseRuntimeAlloc<float>(offsetBytes, inputChannels);
        weights = SparseRuntimeAlloc<float>(offsetBytes, 4 * matrixElements);
        offsetBytes += productGapBytes;
        result = SparseRuntimeAlloc<float>(offsetBytes, BilinearProductElementCount(CI, CO));
        offsetBytes += broadcastGapBytes;
        broadcast = SparseRuntimeAlloc<float>(offsetBytes, inputChannels * (CO <= 32 ? CO : 8));
        offsetBytes += trailingGapBytes;
        accumulator = SparseRuntimeAlloc<float>(offsetBytes, CO);
        messageSum = SparseRuntimeAlloc<float>(offsetBytes, CO);
        rawFeature = LocalTensor<T>(TPosition::VECCALC, 0, 1);
        rawWeight = LocalTensor<T>(TPosition::VECCALC, 0, 1);
        rawOutput = LocalTensor<T>(TPosition::VECCALC, 0, 1);
        if constexpr (sizeof(T) != 4) {
            offsetBytes = (offsetBytes + 65535) / 65536 * 65536;
            rawFeature = SparseRuntimeAlloc<T>(offsetBytes, inputChannels);
            rawWeight = SparseRuntimeAlloc<T>(offsetBytes, 4 * matrixElements);
            rawOutput = SparseRuntimeAlloc<T>(offsetBytes, CO);
        }
        zero = SparseRuntimeAlloc<T>(offsetBytes, 4096);
    }

    __aicore__ inline void Bind(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling) {
        x.SetGlobalBuffer((__gm__ T *)xp);
        weightsGm.SetGlobalBuffer((__gm__ T *)wp);
        coordsGm.SetGlobalBuffer((__gm__ T *)ap);
        outputGm.SetGlobalBuffer((__gm__ T *)yp);
        edgesGm.SetGlobalBuffer((__gm__ int64_t *)ep);
        metadata.SetGlobalBuffer((__gm__ uint64_t *)tiling);
    }

    __aicore__ inline bool ReadMetadata() {
        // Runtime owner counts arrive in metadata. A static complete owner group
        // already knows every active worker and control-array length before DMA.
        weightPrefetched = Coowned || worker < 8;
        packed = 0;
        if constexpr (Coowned) {
            if constexpr (StaticEdgeCount != 0) {
                if (worker >= StaticEdgeCount)
                    return false;
            }
            if constexpr (StaticEdgeCount != 0) {
                DataCachePreload(metadata, int64_t(0));
            } else {
                SplineCCopy(metadataLocal, metadata, 4);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
            }
            if constexpr (StaticEdgeCount != 0) {
                // A complete static owner group knows the control lengths before
                // N arrives. Queue endpoints, coordinates and weights now so the
                // metadata wait does not serialize the remaining input DMA.
                SplineCCopyPad(indices, edgesGm, {1, uint32_t(2 * StaticEdgeCount * 8), 0, 0, 0}, {false, 0, 0, 0});
                SplineCCopyPad(coords, coordsGm, {1, uint32_t(2 * StaticEdgeCount * sizeof(T)), 0, 0, 0},
                            {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                if constexpr (streamWeightStages)
                    StageSupportWeightPackets<T, CI, CO>(weightsGm, weights, rawWeight);
                else if constexpr (sizeof(T) == 4)
                    SplineCCopy(weights, weightsGm, 4 * matrixElements);
                else
                    SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
            }
            if constexpr (StaticEdgeCount != 0) {
                // Controls have compile-time lengths, so the packed metadata
                // fetch can overlap endpoint, coordinate and weight transfers.
                packed = metadata.GetValue(0);
            } else {
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
                packed = SplineCGet(metadataLocal, 0);
            }
        } else {
            DataCachePreload(metadata, int64_t(0));
            if (weightPrefetched) {
                if constexpr (sizeof(T) == 4)
                    SplineCCopy(weights, weightsGm, 4 * matrixElements);
                else
                    SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
            }
            packed = metadata.GetValue(0);
        }
        return true;
    }

    __aicore__ inline bool PrepareOwner() {
        edgeCount = 0;
        nodeCount = packed & kNodeCountMask;
        tileRows = (packed >> kZeroTileRowsShift) & kZeroTileRowsMask;
        mean = (packed >> kMeanShift) & 1;
        cachedInput = LocalTensor<T>(TPosition::VECCALC, 0, 1);
        if constexpr (Coowned && StaticEdgeCount != 0) {
            const uint32_t zeroBeginRow = worker * tileRows;
            const uint32_t zeroEndRow = zeroBeginRow + tileRows < nodeCount ? zeroBeginRow + tileRows : nodeCount;
            if (zeroBeginRow < zeroEndRow) {
                if constexpr (sizeof(T) == 4)
                    SplineCDuplicate(zero, 0.0f, (zeroEndRow - zeroBeginRow) * CO);
                else
                    SplineCDuplicate(zero.template ReinterpretCast<uint16_t>(), uint16_t(0),
                                     (zeroEndRow - zeroBeginRow) * CO);
                // Event 1 publishes only this zero buffer. Later interpolation
                // instructions can proceed while MTE3 consumes the early buffer.
                SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            }
        }
        if constexpr (Coowned) {
            if constexpr (StaticEdgeCount != 0) {
                edgeCount = StaticEdgeCount;
            } else {
                edgeCount = (packed >> kSparseEdgeShift) & kSparseEdgeMask;
                // Inactive workers drain metadata loading and perform no
                // graph input DMA or output store. Active owners complete the full graph.
                if (worker >= edgeCount) {
                    PipeBarrier<PIPE_ALL>();
                    return false;
                }
                SplineCCopyPad(indices, edgesGm, {1, uint32_t(2 * edgeCount * 8), 0, 0, 0}, {false, 0, 0, 0});
                SplineCCopyPad(coords, coordsGm, {1, uint32_t(2 * edgeCount * sizeof(T)), 0, 0, 0},
                            {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                if constexpr (sizeof(T) == 4)
                    SplineCCopy(weights, weightsGm, 4 * matrixElements);
                else
                    SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
            }
        }
        return true;
    }

    __aicore__ inline void QueueInputs() {
        if constexpr (!Coowned) {
            edgeCount = (packed >> kSparseEdgeShift) & kSparseEdgeMask;
            SplineCCopyPad(indices, edgesGm, {1, uint32_t(2 * edgeCount * 8), 0, 0, 0}, {false, 0, 0, 0});
            if (worker < edgeCount)
                SplineCCopyPad(coords, coordsGm, {1, uint32_t(2 * edgeCount * sizeof(T)), 0, 0, 0},
                            {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
        if (worker < edgeCount && !weightPrefetched) {
            if constexpr (sizeof(T) == 4)
                SplineCCopy(weights, weightsGm, 4 * matrixElements);
            else
                SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
        }
        if (CacheFeatures && worker < edgeCount) {
            const uint32_t featureElements = nodeCount * inputChannels;
            cachedInput = SparseRuntimeAlloc<T>(offsetBytes, featureElements);
            // Queue the complete input before waiting for endpoints. Ownership
            // scanning can overlap this DMA; FinishRuntimeMessage drains MTE2
            // before reading the chosen row. Idle/zero owners never load X.
            SplineCCopy(cachedInput, x, featureElements);
        }
    }

    __aicore__ inline void WriteZeroTiles() {
        if constexpr (sizeof(T) == 4)
            SplineCDuplicate(zero, 0.0f, tileRows * CO);
        else
            SplineCDuplicate(zero.template ReinterpretCast<uint16_t>(), uint16_t(0), tileRows * CO);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t edge = 1; edge < edgeCount; ++edge) {
            int64_t key = SplineCGet(indices, edgeCount + edge);
            uint32_t position = edge;
            while (position && SplineCGet(indices, edgeCount + position - 1) > key) {
                SplineCSet(indices, edgeCount + position, SplineCGet(indices, edgeCount + position - 1));
                --position;
            }
            SplineCSet(indices, edgeCount + position, key);
        }
        Fence<HardEvent::V_MTE3>();
        uint32_t tileCount = (nodeCount + tileRows - 1) / tileRows,
                 zeroWorkerCount = GetBlockNum() - edgeCount;
        for (uint32_t tile = worker - edgeCount; tile < tileCount; tile += zeroWorkerCount) {
            uint32_t begin = tile * tileRows,
                     end = begin + tileRows < nodeCount ? begin + tileRows : nodeCount, cursor = begin;
            for (uint32_t edge = 0; edge < edgeCount; ++edge) {
                uint32_t destination = SplineCGet(indices, edgeCount + edge);
                if (destination < cursor)
                    continue;
                if (destination >= end)
                    break;
                if (destination > cursor)
                    SplineCCopyPad(outputGm[uint64_t(cursor) * CO], zero,
                                {1, uint32_t((destination - cursor) * CO * sizeof(T)), 0, 0, 0});
                cursor = destination + 1;
            }
            if (cursor < end)
                SplineCCopyPad(outputGm[uint64_t(cursor) * CO], zero,
                            {1, uint32_t((end - cursor) * CO * sizeof(T)), 0, 0, 0});
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StageMessage() {
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        target = SplineCGet(indices, edgeCount + worker);
        incomingCount = 0;
        // Every message worker owns a real edge. Stage its actual source row as
        // soon as endpoints are ready; ownership scans do not consume this row.
        // Duplicate-target nonleaders drain this issued DMA before returning.
        if constexpr (!CacheFeatures) {
            StageRuntimeFeature(x, SplineCGet(indices, worker), inputChannels, feature, rawFeature);
            if constexpr (StaticEdgeCount != 0)
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        // Queue current-edge weight arithmetic before scalar ownership scanning.
        // Every active worker consumes its four packet tokens here. A nonleader
        // later drains the queued work; leaders reuse this current-call kernel.
        if constexpr (streamWeightStages)
            BlendSupportWeightPackets<T, CI, CO>(LocalPseudo(coords, 2 * worker),
                LocalPseudo(coords, 2 * worker + 1), weights, rawWeight, result);
    }

    __aicore__ inline void FindLeader() {
        leader = true;
        begin = worker * tileRows;
        end = begin + tileRows < nodeCount ? begin + tileRows : nodeCount;
        occupiedRows = 0;
        if constexpr (Coowned) {
#pragma unroll
            for (uint32_t edge = 0; edge < 8; ++edge) {
                if (edge >= edgeCount)
                    continue;
                uint32_t destination = SplineCGet(indices, edgeCount + edge);
                if (destination == target) {
                    ++incomingCount;
                    if (edge < worker)
                        leader = false;
                }
                if (destination >= begin && destination < end)
                    occupiedRows |= uint64_t(1) << (destination - begin);
            }
        } else {
            for (uint32_t edge = 0; edge < edgeCount; ++edge) {
                uint32_t destination = SplineCGet(indices, edgeCount + edge);
                if (destination == target) {
                    ++incomingCount;
                    if (edge < worker)
                        leader = false;
                }
            }
        }
    }

    __aicore__ inline void PrepareMessage() {
        // The complete static owner group can overlap its first real feature row
        // with zero-complement stores, including repeated-target accumulation.
        if (leader && CacheFeatures && (StaticEdgeCount != 0 || incomingCount == 1))
            SelectCachedFeatureRow(cachedInput, SplineCGet(indices, worker), inputChannels, feature, rawFeature);
        if constexpr (streamWeightStages)
            if (Coowned && begin < end)
                StoreSparseZeroComplement<T, CO, true>(outputGm, zero, begin, end, occupiedRows);
        messagePrepared = false;
        if constexpr (streamWeightStages) {
            if (leader) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if constexpr (sizeof(T) != 4) {
                    SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, inputChannels);
                    PipeBarrier<PIPE_V>();
                }
                FinishStreamedGemv<CI, CO>(feature, result, broadcast);
                messagePrepared = true;
            }
        } else if constexpr (Coowned && StaticEdgeCount != 0 && !CacheFeatures) {
            if (leader && incomingCount == 1) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if constexpr (sizeof(T) != 4) {
                    SplineCCast(weights, rawWeight, RoundMode::CAST_NONE, 4 * matrixElements);
                    SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, inputChannels);
                    PipeBarrier<PIPE_V>();
                }
                RuntimeBilinear<CI, CO, true, (sizeof(T) != 4)>(inputChannels, LocalPseudo(coords, 2 * worker),
                                             LocalPseudo(coords, 2 * worker + 1), feature, weights, result,
                                             broadcast, messageSum);
                messagePrepared = true;
            }
        }
        if constexpr (!streamWeightStages)
            if (Coowned && begin < end)
                StoreSparseZeroComplement<T, CO, StaticEdgeCount != 0>(outputGm, zero, begin, end, occupiedRows);
    }

    __aicore__ inline bool FinishEarly() {
        if (!leader) {
            if constexpr (StaticEdgeCount != 0 && !CacheFeatures)
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            PipeBarrier<PIPE_ALL>();
            return true;
        }
        if (messagePrepared && incomingCount == 1) {
            // The complete support GEMVs finished before independent zero DMA.
            FastStore(outputGm, uint64_t(target) * CO, result, rawOutput, CO);
            return true;
        }
        return false;
    }

    __aicore__ inline void AccumulatePair(uint32_t currentEdge, uint32_t secondEdge) {
        constexpr uint32_t secondSlot = 4 * kFloatVectorElements;
        static_assert(CI <= secondSlot, "Second real row fits one untouched message slot");
        // Slot1 is untouched by slot0 products. The first input's
        // vector work overlaps this complete second-row transfer.
        SplineCCopy(result[secondSlot], x[uint64_t(SplineCGet(indices, secondEdge)) * CI], CI);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        ProduceIncomingPairSlot<CI, CO>(feature, weights, result, broadcast, 0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        ProduceIncomingPairSlot<CI, CO>(result[secondSlot], weights, result, broadcast, 1);
        ReducePairedMessageChunks<matrixElements / (2 * kFloatVectorElements)>(result);
        ReducePairedRowsInsideChunk<kFloatVectorElements / CO, CO>(result);
        FinishIncomingPairSlot<CO>(LocalPseudo(coords, 2 * currentEdge),
                                   LocalPseudo(coords, 2 * currentEdge + 1), result);
        SplineCAdd(accumulator, accumulator, result, CO);
        PipeBarrier<PIPE_V>();
        FinishIncomingPairSlot<CO>(LocalPseudo(coords, 2 * secondEdge),
                                   LocalPseudo(coords, 2 * secondEdge + 1), result[secondSlot]);
        SplineCAdd(accumulator, accumulator, result[secondSlot], CO);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreAccumulator() {
        if (mean) {
            SplineCMuls(accumulator, accumulator, 1.0f / incomingCount, CO);
            PipeBarrier<PIPE_V>();
        }
        FastStore(outputGm, uint64_t(target) * CO, accumulator, rawOutput, CO);
    }

    __aicore__ inline void AccumulatePairedEdges() {
        SplineCDuplicate(accumulator, 0.0f, CO);
        PipeBarrier<PIPE_V>();
        uint32_t currentEdge = worker;
        bool firstStagedRow = true;
        while (currentEdge < edgeCount) {
            if (!firstStagedRow) {
                Fence<HardEvent::V_MTE2>();
                StageRuntimeFeature(x, SplineCGet(indices, currentEdge), inputChannels, feature, rawFeature);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }
            uint32_t secondEdge = currentEdge + 1;
            while (secondEdge < edgeCount && SplineCGet(indices, edgeCount + secondEdge) != target)
                ++secondEdge;
            if (secondEdge < edgeCount) {
                AccumulatePair(currentEdge, secondEdge);
                currentEdge = secondEdge + 1;
            } else {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                RuntimeBilinear<CI, CO, false, false>(inputChannels,
                    LocalPseudo(coords, 2 * currentEdge), LocalPseudo(coords, 2 * currentEdge + 1),
                    feature, weights, result, broadcast, messageSum);
                SplineCAdd(accumulator, accumulator, result, CO);
                PipeBarrier<PIPE_V>();
                currentEdge = edgeCount;
            }
            firstStagedRow = false;
            while (currentEdge < edgeCount && SplineCGet(indices, edgeCount + currentEdge) != target)
                ++currentEdge;
        }
        StoreAccumulator();
    }

    __aicore__ inline void FinishMessages() {
        const bool firstMessageReady = streamWeightStages && messagePrepared;
        bool ready = firstMessageReady;
        if (incomingCount == 1) {
            if (CacheFeatures)
                FinishRuntimeMessage<T, CI, CO, true>(worker, inputChannels, coords, feature, weights, result,
                                                  broadcast, messageSum, rawFeature, rawWeight, ready, StaticEdgeCount != 0 && !CacheFeatures);
            else
                FinishRuntimeMessage<T, CI, CO>(worker, inputChannels, coords, feature, weights, result, broadcast, messageSum,
                                            rawFeature, rawWeight, ready, StaticEdgeCount != 0 && !CacheFeatures);
            FastStore(outputGm, uint64_t(target) * CO, result, rawOutput, CO);
            return;
        }
        if (firstMessageReady)
            SplineCAdds(accumulator, result, 0.0f, CO);
        else
            SplineCDuplicate(accumulator, 0.0f, CO);
        PipeBarrier<PIPE_V>();
        const uint32_t firstUnprocessedEdge = firstMessageReady ? worker + 1 : worker;
        // Dispatch once per target. The uncached loop retains its original
        // row-DMA schedule and pays no cache selection branch per message.
        if (CacheFeatures) {
            for (uint32_t edge = firstUnprocessedEdge; edge < edgeCount; ++edge) {
                if (SplineCGet(indices, edgeCount + edge) != target)
                    continue;
                SelectCachedFeatureRow(cachedInput, SplineCGet(indices, edge), inputChannels, feature, rawFeature);
                FinishRuntimeMessage<T, CI, CO, true>(edge, inputChannels, coords, feature, weights, result,
                                                  broadcast, messageSum, rawFeature, rawWeight, ready, StaticEdgeCount != 0 && !CacheFeatures);
                SplineCAdd(accumulator, accumulator, result, CO);
                PipeBarrier<PIPE_V>();
            }
        } else {
            for (uint32_t edge = firstUnprocessedEdge; edge < edgeCount; ++edge) {
                if (SplineCGet(indices, edgeCount + edge) != target)
                    continue;
                // The first matching edge is the leader's own staged row. Only
                // later messages replace it, after the previous V consumer.
                if (ready) {
                    Fence<HardEvent::V_MTE2>();
                    StageRuntimeFeature(x, SplineCGet(indices, edge), inputChannels, feature, rawFeature);
                }
                FinishRuntimeMessage<T, CI, CO>(edge, inputChannels, coords, feature, weights, result, broadcast, messageSum,
                                            rawFeature, rawWeight, ready, StaticEdgeCount != 0 && !CacheFeatures);
                SplineCAdd(accumulator, accumulator, result, CO);
                PipeBarrier<PIPE_V>();
            }
        }
        StoreAccumulator();
    }

    __aicore__ inline void Run(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling) {
        Allocate();
        Bind(xp, ep, ap, wp, yp, tiling);
        if (!ReadMetadata() || !PrepareOwner()) return;
        QueueInputs();
        if constexpr (!Coowned) {
            if (worker >= edgeCount) {
                WriteZeroTiles();
                return;
            }
        }
        StageMessage();
        FindLeader();
        PrepareMessage();
        if (FinishEarly()) return;
        if constexpr (Paired) {
            if (incomingCount > 1) {
                AccumulatePairedEdges();
                return;
            }
        }
        FinishMessages();
    }

};

template <class T, uint32_t CI, uint32_t CO, bool Coowned, bool CacheFeatures = false,
          uint32_t StaticEdgeCount = 0, bool Paired = false>
__aicore__ inline void RuntimeSparseGraph(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp,
                                          GM_ADDR tiling) {
    SparseGraphWorker<T, CI, CO, Coowned, CacheFeatures, StaticEdgeCount, Paired> worker;
    worker.Run(xp, ep, ap, wp, yp, tiling);
}
