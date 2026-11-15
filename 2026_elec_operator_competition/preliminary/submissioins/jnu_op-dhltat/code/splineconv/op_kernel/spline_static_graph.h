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

// A complete small-edge bucket has one leader per connected destination and
// one separate zero-complement owner. Padding reserves whole 8-worker groups;
// padding workers return before DMA. N and mean remain runtime metadata.
template <class U> __aicore__ inline LocalTensor<U> StaticGraphAlloc(uint32_t &offset, uint32_t count) {
    LocalTensor<U> tensor(TPosition::VECCALC, offset, count);
    offset += (count * sizeof(U) + 31) / 32 * 32;
    return tensor;
}
// Separate the packed result from weights by 128 bytes and retain the
// existing total 288-byte padding before the broadcast buffer. Both offsets
// are included in the host's conservative UB budget. This avoids the aliases
// created by contiguous power-of-two matrix buffers.
template <class U, uint32_t paddingBytes>
__aicore__ inline LocalTensor<U> StaticGraphPaddedAlloc(uint32_t &offset, uint32_t count) {
    offset += paddingBytes;
    return StaticGraphAlloc<U>(offset, count);
}
template <class T, uint32_t CI, uint32_t CO, bool PREFETCH = false>
__aicore__ inline void StaticGraphMessage(uint32_t e, uint32_t source, GlobalTensor<T> x, GlobalTensor<T> gmW,
                                          GlobalTensor<T> pseudo, GlobalTensor<uint16_t> pseudoBits,
                                          LocalTensor<float> feature, LocalTensor<float> weight,
                                          LocalTensor<float> result, LocalTensor<float> broadcast,
                                          LocalTensor<float> messageSum, LocalTensor<T> rawFeature, LocalTensor<T> rawWeight, bool &ready) {
    constexpr uint32_t M = CI * CO;
    if (ready)
        Fence<HardEvent::V_MTE2>();
    if constexpr (sizeof(T) == 4) {
        if (!ready && !PREFETCH)
            SplineCCopy(weight, gmW, 4 * M);
        SplineCCopy(feature, x[uint64_t(source) * CI], CI);
    } else {
        if (!ready && !PREFETCH)
            SplineCCopy(rawWeight, gmW, 4 * M);
        if constexpr (CI % 16 == 0)
            SplineCCopy(rawFeature, x[uint64_t(source) * CI], CI);
        else
            SplineCCopyPad(rawFeature, x[uint64_t(source) * CI], {1, CI * 2, 0, 0, 0}, {false, 0, 0, 0});
    }
    float u = PseudoValue(pseudo, pseudoBits, 2 * e), v = PseudoValue(pseudo, pseudoBits, 2 * e + 1);
    Fence<HardEvent::MTE2_V>();
    if constexpr (sizeof(T) != 4) {
        if (!ready)
            SplineCCast(weight, rawWeight, RoundMode::CAST_NONE, 4 * M);
        SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, CI);
        PipeBarrier<PIPE_V>();
    }
    ready = true;
    FastBilinear<CI, CO, (sizeof(T) != 4)>(u, v, feature, weight, result, broadcast, messageSum);
}
// The caller reads the actual coordinates before consuming the input-ready
// token. Arithmetic and first weight conversion keep the original order.
template <class T, uint32_t CI, uint32_t CO>
__aicore__ inline void StaticGraphStagedMessage(float u, float v,
                                              LocalTensor<float> feature, LocalTensor<float> weight,
                                              LocalTensor<float> result, LocalTensor<float> broadcast,
                                              LocalTensor<float> messageSum, LocalTensor<T> rawFeature, LocalTensor<T> rawWeight,
                                              bool &ready) {
    constexpr uint32_t matrixElements = CI * CO;
    if constexpr (sizeof(T) != 4) {
        if (!ready)
            SplineCCast(weight, rawWeight, RoundMode::CAST_NONE, 4 * matrixElements);
        SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, CI);
        PipeBarrier<PIPE_V>();
    }
    ready = true;
    FastBilinear<CI, CO, (sizeof(T) != 4)>(u, v, feature, weight, result, broadcast, messageSum);
}
__aicore__ inline void OrderDestinationPair(uint32_t &a, uint32_t &b) {
    const uint32_t lo = a < b ? a : b, hi = a > b ? a : b;
    a = lo;
    b = hi;
}

template <uint32_t ES>
__aicore__ inline void SortDestinations(uint32_t (&rows)[8]) {
    if constexpr (ES > 0 && ES <= 2) {
        OrderDestinationPair(rows[0], rows[1]);
    } else if constexpr (ES > 0 && ES <= 4) {
        OrderDestinationPair(rows[0], rows[1]);
        OrderDestinationPair(rows[2], rows[3]);
        OrderDestinationPair(rows[0], rows[2]);
        OrderDestinationPair(rows[1], rows[3]);
        OrderDestinationPair(rows[1], rows[2]);
    } else {
        OrderDestinationPair(rows[0], rows[1]);
        OrderDestinationPair(rows[2], rows[3]);
        OrderDestinationPair(rows[4], rows[5]);
        OrderDestinationPair(rows[6], rows[7]);
        OrderDestinationPair(rows[0], rows[2]);
        OrderDestinationPair(rows[1], rows[3]);
        OrderDestinationPair(rows[4], rows[6]);
        OrderDestinationPair(rows[5], rows[7]);
        OrderDestinationPair(rows[1], rows[2]);
        OrderDestinationPair(rows[5], rows[6]);
        OrderDestinationPair(rows[0], rows[4]);
        OrderDestinationPair(rows[3], rows[7]);
        OrderDestinationPair(rows[1], rows[5]);
        OrderDestinationPair(rows[2], rows[6]);
        OrderDestinationPair(rows[1], rows[4]);
        OrderDestinationPair(rows[3], rows[6]);
        OrderDestinationPair(rows[2], rows[4]);
        OrderDestinationPair(rows[3], rows[5]);
        OrderDestinationPair(rows[3], rows[4]);
    }
}

template <class T, uint32_t CO>
__aicore__ inline void StoreUnoccupiedMask(GlobalTensor<T> y, LocalTensor<T> zero, LocalTensor<int64_t> indices, uint32_t e, uint32_t n) {
    uint64_t occupiedRows = 0;
    for (uint32_t edgeIndex = 0; edgeIndex < e; ++edgeIndex) {
        uint32_t destinationRow = uint32_t(SplineCGet(indices, e + edgeIndex));
        if (destinationRow < n)
            occupiedRows |= uint64_t(1) << destinationRow;
    }
    Fence<HardEvent::V_MTE3>();
    // Enumerate occupied rows in descending order. Every
    // emitted interval excludes message rows; duplicate
    // destinations share one bit and require no extra store.
    uint32_t intervalEndRow = n;
    while (occupiedRows != 0) {
        uint32_t destinationRow = 63 - uint32_t(CountLeadingZero(occupiedRows));
        uint32_t intervalBeginRow = destinationRow + 1;
        if (intervalBeginRow < intervalEndRow)
            SplineCCopyPad(y[uint64_t(intervalBeginRow) * CO], zero,
                        {1, uint32_t((intervalEndRow - intervalBeginRow) * CO * sizeof(T)), 0, 0, 0});
        intervalEndRow = destinationRow;
        occupiedRows &= ~(uint64_t(1) << destinationRow);
    }
    if (intervalEndRow != 0)
        SplineCCopyPad(y, zero, {1, uint32_t(intervalEndRow * CO * sizeof(T)), 0, 0, 0});
    Fence<HardEvent::MTE3_V>();
}

template <class T, uint32_t CO, uint32_t ES>
__aicore__ inline void StoreSortedComplement(GlobalTensor<T> y, LocalTensor<T> zero, LocalTensor<int64_t> indices, uint32_t e, uint32_t n) {
    uint32_t rows[8];
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j)
        rows[j] = j < e ? uint32_t(SplineCGet(indices, e + j)) : n;
    SortDestinations<ES>(rows);
    Fence<HardEvent::V_MTE3>();
    uint32_t cursor = 0;
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
        const uint32_t destination = rows[j];
        if (destination < cursor || destination >= n) continue;
        if (cursor < destination)
            SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                           {1, uint32_t((destination - cursor) * CO * sizeof(T)), 0, 0, 0});
        cursor = destination + 1;
    }
    if (cursor < n)
        SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                    {1, uint32_t((n - cursor) * CO * sizeof(T)), 0, 0, 0});
    Fence<HardEvent::MTE3_V>();
}

template <class T, uint32_t CI, uint32_t CO, bool MEAN, uint32_t ES = 0, bool RUNTIME_MEAN = false>
struct StaticSplineOwner {
    static constexpr uint32_t matrixElements = CI * CO;
    static constexpr bool streamWeightStages = sizeof(T) != 4 && matrixElements >= 512 && matrixElements <= 2048 &&
        matrixElements % 128 == 0 && CO % 16 == 0;
    uint32_t e, offset, block, target, count;
    bool stageFirstFeature, ready;
    GlobalTensor<T> x, w, a, y;
    GlobalTensor<int64_t> edge;
    GlobalTensor<uint16_t> bits;
    LocalTensor<int64_t> indices;
    LocalTensor<T> stagedCoordinates, rawFeature, rawWeight;
    LocalTensor<float> feature, weight, result, broadcast, output, messageSum;
    __aicore__ inline void LoadControls(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp) {
        x.SetGlobalBuffer((__gm__ T *)xp);
        w.SetGlobalBuffer((__gm__ T *)wp);
        a.SetGlobalBuffer((__gm__ T *)ap);
        y.SetGlobalBuffer((__gm__ T *)yp);
        edge.SetGlobalBuffer((__gm__ int64_t *)ep);
        bits.SetGlobalBuffer((__gm__ uint16_t *)ap);
        offset = 0;
        block = GetBlockIdx();
        indices = StaticGraphAlloc<int64_t>(offset, 2 * e);
        stagedCoordinates = StaticGraphAlloc<T>(offset, 2 * e);
        SplineCCopyPad(indices, edge, {1, uint32_t(2 * e * 8), 0, 0, 0}, {false, 0, 0, 0});
        if (block < e)
            SplineCCopyPad(stagedCoordinates, a, {1, uint32_t(2 * e * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    __aicore__ inline void StageMessageOperands() {
        feature = StaticGraphAlloc<float>(offset, CI);
        weight = StaticGraphAlloc<float>(offset, 4 * CI * CO);
        result = StaticGraphPaddedAlloc<float, 128>(offset, BilinearProductElementCount(CI, CO));
        broadcast = StaticGraphPaddedAlloc<float, 160>(offset, CI * (CO <= 32 ? CO : 8));
        output = StaticGraphAlloc<float>(offset, CO);
        messageSum = StaticGraphAlloc<float>(offset, CO);
        rawFeature = LocalTensor<T>(TPosition::VECCALC, 0, 1);
        rawWeight = LocalTensor<T>(TPosition::VECCALC, 0, 1);
        if constexpr (sizeof(T) != 4) {
            rawFeature = StaticGraphAlloc<T>(offset, CI > CO ? CI : CO);
            rawWeight = StaticGraphAlloc<T>(offset, 4 * CI * CO);
        }
        if constexpr (streamWeightStages)
            StageSupportWeightPackets<T, CI, CO>(w, weight, rawWeight);
        else if constexpr (sizeof(T) == 4)
            SplineCCopy(weight, w, 4 * CI * CO);
        else
            SplineCCopy(rawWeight, w, 4 * CI * CO);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        stageFirstFeature = e > 0 && e < 8;
        if (stageFirstFeature) {
            // The actual own-edge row is independent of target ownership.
            // Publish it before any later mean or alternate-slot DMA.
            StageFeature<T, CI>(x, SplineCGet(indices, block), feature, rawFeature);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
    }

    __aicore__ inline bool IsDestinationLeader() {
        target = SplineCGet(indices, e + block);
        count = 0;
        for (uint32_t j = 0; j < e; ++j) {
            if (SplineCGet(indices, e + j) == target) {
                if (j < block) {
                    if constexpr (streamWeightStages)
                        DrainSupportWeightPackets();
                    if (stageFirstFeature)
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    // Retain the original MTE2-to-scalar drain on nonleaders.
                    Fence<HardEvent::MTE2_S>();
                    return false;
                }
                ++count;
            }
        }
        return true;
    }

    __aicore__ inline void StoreSingleMessage() {
        if (stageFirstFeature) {
            float coordinateU = LocalPseudo(stagedCoordinates, 2 * block),
                  coordinateV = LocalPseudo(stagedCoordinates, 2 * block + 1);
            if constexpr (streamWeightStages) {
                BlendSupportWeightPackets<T, CI, CO>(coordinateU, coordinateV, weight, rawWeight, result);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if constexpr (sizeof(T) != 4) {
                    SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, CI);
                    PipeBarrier<PIPE_V>();
                }
                FinishStreamedGemv<CI, CO>(feature, result, broadcast);
                ready = true;
            } else {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                StaticGraphStagedMessage<T, CI, CO>(coordinateU, coordinateV, feature, weight, result,
                                                    broadcast, messageSum, rawFeature, rawWeight, ready);
            }
        } else {
            if constexpr (streamWeightStages)
                DrainSupportWeightPackets();
            StaticGraphMessage<T, CI, CO, true>(block, SplineCGet(indices, block), x, w, a, bits, feature,
                                                weight, result, broadcast, messageSum, rawFeature, rawWeight, ready);
        }
        FastStore(y, uint64_t(target) * CO, result, rawFeature, CO);
    }

    __aicore__ inline void AccumulateOnlyEdge() {
        // Preserve the original empty invalid-endpoint fallback while
        // consuming the early row's already published token.
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SplineCDuplicate(output, 0.0f, CO);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = block; j < e; ++j)
            if (SplineCGet(indices, e + j) == target) {
                StaticGraphMessage<T, CI, CO, true>(j, SplineCGet(indices, j), x, w, a, bits, feature,
                                                    weight, result, broadcast, messageSum, rawFeature, rawWeight,
                                                    ready);
                SplineCAdd(output, output, result, CO);
                PipeBarrier<PIPE_V>();
            }
    }

    __aicore__ inline void AccumulateRowPair(uint32_t currentEdge, uint32_t nextEdge, bool alternateInput,
                                             LocalTensor<float> alternateFeature, LocalTensor<T> alternateRawFeature) {
        if (nextEdge < e) {
            if (alternateInput) {
                StageFeature<T, CI>(x, SplineCGet(indices, nextEdge), feature, rawFeature);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            } else {
                StageFeature<T, CI>(x, SplineCGet(indices, nextEdge), alternateFeature,
                                    alternateRawFeature);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
        }
        float coordinateU = LocalPseudo(stagedCoordinates, 2 * currentEdge),
              coordinateV = LocalPseudo(stagedCoordinates, 2 * currentEdge + 1);
        if (alternateInput)
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        else
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        auto currentFeature = alternateInput ? alternateFeature : feature;
        auto currentRawFeature = alternateInput ? alternateRawFeature : rawFeature;
        StaticGraphStagedMessage<T, CI, CO>(coordinateU, coordinateV, currentFeature, weight,
                                            result, broadcast, messageSum, currentRawFeature, rawWeight,
                                            ready);
        SplineCAdd(output, output, result, CO);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateIncomingRows() {
        // A future input row occupies the other staging slot. BF16 keeps
        // one converted FP32 feature and alternates only its raw input.
        LocalTensor<float> alternateFeature = feature;
        LocalTensor<T> alternateRawFeature = rawFeature;
        if constexpr (sizeof(T) == 4)
            alternateFeature = StaticGraphAlloc<float>(offset, CI);
        else
            alternateRawFeature = StaticGraphAlloc<T>(offset, CI);
        SplineCDuplicate(output, 0.0f, CO);
        PipeBarrier<PIPE_V>();
        uint32_t currentEdge = block;
        while (currentEdge < e && SplineCGet(indices, e + currentEdge) != target)
            ++currentEdge;
        if (currentEdge < e) {
            bool alternateInput = false;
            if (!stageFirstFeature) {
                StageFeature<T, CI>(x, SplineCGet(indices, currentEdge), feature, rawFeature);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }
            // Every active small-edge owner has already published its
            // first row; the slot0 wait consumes that token.
            while (currentEdge < e) {
                uint32_t nextEdge = currentEdge + 1;
                while (nextEdge < e && SplineCGet(indices, e + nextEdge) != target)
                    ++nextEdge;
                AccumulateRowPair(currentEdge, nextEdge, alternateInput, alternateFeature, alternateRawFeature);
                if (nextEdge == e)
                    break;
                // Only a consumed slot can be refilled. This retains one
                // V_MTE2 dependency per transition, with no unused final token.
                Fence<HardEvent::V_MTE2>();
                currentEdge = nextEdge;
                alternateInput = !alternateInput;
            }
        }
    }

    __aicore__ inline void StoreMultipleMessages(GM_ADDR tiling) {
        // Repeated destinations retain the established double-buffered
        // row loop. Drain each weight token once before its first cast.
        if constexpr (streamWeightStages)
            DrainSupportWeightPackets();
        LocalTensor<uint64_t> meanData(TPosition::VECCALC, 0, 1);
        if constexpr (RUNTIME_MEAN) {
            meanData = StaticGraphAlloc<uint64_t>(offset, 4);
            GlobalTensor<uint64_t> metadata;
            metadata.SetGlobalBuffer((__gm__ uint64_t *)tiling);
            SplineCCopyPad(meanData, metadata, {1, 8, 0, 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        }
        // One edge has no future message to overlap. Keep the original
        // ES1 accumulator, including its empty invalid-endpoint fallback.
        if (e == 1)
            AccumulateOnlyEdge();
        else
            AccumulateIncomingRows();
        bool applyMean = MEAN;
        if constexpr (RUNTIME_MEAN) {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            applyMean = (SplineCGet(meanData, 0) >> kMeanShift) & 1;
        }
        if (applyMean) {
            // This branch follows count==1. In a legal two-edge graph,
            // the current edge always matches its own destination, so
            // count is exactly2 here and its reciprocal is exactly0.5.
            // Other complete edge-count families keep runtime division.
            const float meanScale = e == 2 ? 0.5f : 1.0f / count;
            SplineCMuls(output, output, meanScale, CO);
            PipeBarrier<PIPE_V>();
        }
        FastStore(y, uint64_t(target) * CO, output, rawFeature, CO);
    }

    __aicore__ inline void StoreSmallComplement(GM_ADDR tiling) {
        GlobalTensor<uint64_t> metadata;
        metadata.SetGlobalBuffer((__gm__ uint64_t *)tiling);
        auto md = StaticGraphAlloc<uint64_t>(offset, 4);
        SplineCCopyPad(md, metadata, {1, 8, 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        uint32_t n = SplineCGet(md, 0) & kNodeCountMask;
        auto zero = StaticGraphAlloc<T>(offset, 4096);
        if constexpr (sizeof(T) == 4)
            SplineCDuplicate(zero, 0.0f, n * CO);
        else
            SplineCDuplicate(zero.template ReinterpretCast<uint16_t>(), uint16_t(0), n * CO);
        if (e > 0 && e < 8 && n <= 64)
            StoreUnoccupiedMask<T, CO>(y, zero, indices, e, n);
        else
            StoreSortedComplement<T, CO, ES>(y, zero, indices, e, n);
    }

    __aicore__ inline void StoreLargeComplement(GM_ADDR tiling) {
        GlobalTensor<uint64_t> metadata;
        metadata.SetGlobalBuffer((__gm__ uint64_t *)tiling);
        uint32_t n = metadata.GetValue(0) & kNodeCountMask, begin = 0, rows = n, end = n;
        auto zero = StaticGraphAlloc<T>(offset, rows * CO);
        if constexpr (sizeof(T) == 4)
            SplineCDuplicate(zero, 0.0f, rows * CO);
        else
            SplineCDuplicate(zero.template ReinterpretCast<uint16_t>(), uint16_t(0), rows * CO);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t j = 1; j < e; ++j) {
            int64_t key = SplineCGet(indices, e + j);
            uint32_t q = j;
            while (q > 0 && SplineCGet(indices, e + q - 1) > key) {
                SplineCSet(indices, e + q, SplineCGet(indices, e + q - 1));
                --q;
            }
            SplineCSet(indices, e + q, key);
        }
        Fence<HardEvent::V_MTE3>();
        uint32_t cursor = begin;
        for (uint32_t j = 0; j < e; ++j) {
            uint32_t dst = SplineCGet(indices, e + j);
            if (dst < cursor)
                continue;
            if (dst >= end)
                break;
            if (dst > cursor)
                SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                            {1, uint32_t((dst - cursor) * CO * sizeof(T)), 0, 0, 0});
            cursor = dst + 1;
        }
        if (cursor < end)
            SplineCCopyPad(y[uint64_t(cursor) * CO], zero,
                        {1, uint32_t((end - cursor) * CO * sizeof(T)), 0, 0, 0});
        Fence<HardEvent::MTE3_V>();
    }

    __aicore__ inline void Run(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling, uint32_t runtimeEdgeCount) {
        e = ES ? ES : runtimeEdgeCount;
        if (GetBlockIdx() > e)
            return;
        LoadControls(xp, ep, ap, wp, yp);
        if (block == e) {
            if (e <= 8) StoreSmallComplement(tiling);
            else StoreLargeComplement(tiling);
            return;
        }
        StageMessageOperands();
        if (!IsDestinationLeader()) return;
        ready = false;
        if (count == 1) StoreSingleMessage();
        else StoreMultipleMessages(tiling);
    }

};

template <class T, uint32_t CI, uint32_t CO, bool MEAN, uint32_t ES = 0, bool RUNTIME_MEAN = false>
__aicore__ inline void StaticEdgeGraph(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp,
                                       GM_ADDR tiling, uint32_t runtimeEdgeCount = 0) {
    StaticSplineOwner<T, CI, CO, MEAN, ES, RUNTIME_MEAN> owner;
    owner.Run(xp, ep, ap, wp, yp, tiling, runtimeEdgeCount);
}
