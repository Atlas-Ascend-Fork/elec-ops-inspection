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
#include "spline_single_gemv.h"

template <class T, uint32_t CI, uint32_t CO, uint32_t HintWordCount = 0>
struct RuntimeSingleEdgeWorker {
    static constexpr bool alignedRows = (CO * sizeof(T)) % kUbAlignmentBytes == 0;
    static constexpr bool overlapFeature = CI <= 24 && CO <= 32 && CI * CO <= 256;
    // This matrix resource family regressed under packed feature DMA in
    // stable ABBA. Keep its original paired scalar support arithmetic.
    static constexpr bool scalarSupportFamily = sizeof(T) == 4 && overlapFeature && CO / 8 < 4 && (4 * CO) % kFloatVectorElements == 0;
    static constexpr bool packedSmallFeature = sizeof(T) == 4 && overlapFeature && CO / 8 < 4 && !scalarSupportFamily;
    static constexpr uint32_t hintMinRows = HintWordCount > 0
        ? ((HintWordCount - 1) * 64 / (CI * sizeof(T))) + 1 : 0;
    static constexpr uint32_t hintMaxRows = HintWordCount > 0
        ? HintWordCount * 64 / (CI * sizeof(T)) : 0;
    static constexpr bool exactHintRows = HintWordCount > 0 && hintMinRows == hintMaxRows;
    GlobalTensor<T> x, weightsGm, coordsGm, outputGm;
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<uint64_t> metadataGm;
    GlobalTensor<uint64_t> edgeCacheLine, coordinateCacheLine, nodeCacheLine;
    GlobalTensor<uint16_t> coordinateBits;
    LocalTensor<float> feature, weights, result, broadcast, messageSum, output;
    LocalTensor<T> rawFeature, rawWeights, rawOutput;
    uint32_t source, target, nodeCount;

    __aicore__ inline void Bind(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling) {
        x.SetGlobalBuffer((__gm__ T*)xp);
        weightsGm.SetGlobalBuffer((__gm__ T*)wp);
        coordsGm.SetGlobalBuffer((__gm__ T*)ap);
        outputGm.SetGlobalBuffer((__gm__ T*)yp);
        edgesGm.SetGlobalBuffer((__gm__ int64_t*)ep);
        metadataGm.SetGlobalBuffer((__gm__ uint64_t*)tiling);
        // Preload current-call controls while independent weight DMA and buffer
        // setup proceed. Values are still read from the actual input tensors.
        edgeCacheLine.SetGlobalBuffer((__gm__ uint64_t*)ep);
        coordinateCacheLine.SetGlobalBuffer((__gm__ uint64_t*)ap);
        nodeCacheLine.SetGlobalBuffer((__gm__ uint64_t*)tiling);
        coordinateBits.SetGlobalBuffer((__gm__ uint16_t*)ap);
        DataCachePreload(edgeCacheLine, int64_t(0));
        DataCachePreload(coordinateCacheLine, int64_t(0));
        if constexpr (!alignedRows && !exactHintRows)
            DataCachePreload(nodeCacheLine, int64_t(0));
        // NN counts in-bounds early hint words; actual N remains runtime metadata.
        if constexpr (sizeof(T) == 2 && !alignedRows && HintWordCount > 0 &&
                      CI <= 24 && CO <= 32 && CI * CO <= 256)
            HintBoundedInputSpan<T, HintWordCount>(xp);
    }

    __aicore__ inline void StageWeights() {
        uint32_t offsetBytes = 2048;
        feature = SmallSingleAlloc<float>(offsetBytes, CI);
        weights = SmallSingleAlloc<float>(offsetBytes, 4 * CI * CO);
        offsetBytes += 128;
        result = SmallSingleAlloc<float>(offsetBytes, BilinearProductElementCount(CI, CO));
        offsetBytes += 160;
        broadcast = SmallSingleAlloc<float>(offsetBytes, CI * CO);
        messageSum = SmallSingleAlloc<float>(offsetBytes, CO);
        output = SmallSingleAlloc<float>(offsetBytes, 1024);
        rawFeature = LocalTensor<T>(TPosition::VECCALC, 0, 1);
        rawWeights = LocalTensor<T>(TPosition::VECCALC, 0, 1);
        rawOutput = LocalTensor<T>(TPosition::VECCALC, 0, 1);
        if constexpr (sizeof(T) == 4) {
            if constexpr (overlapFeature && !packedSmallFeature) {
                // Four aligned strided copies transpose support and CI in UB.
                // Each descriptor copies all actual CI rows of one support.
                const DataCopyParams rows {uint16_t(CI), uint16_t(CO / 8), 0, uint16_t(3 * CO / 8)};
#pragma unroll
                for (uint32_t support = 0; support < 4; ++support)
                    DataCopy(weights[support * CO], weightsGm[support * CI * CO], rows);
            } else {
                SplineCCopy(weights, weightsGm, 4 * CI * CO);
            }
        } else {
            offsetBytes = 65536;
            rawFeature = FastAlloc<T>(offsetBytes, CI);
            rawWeights = FastAlloc<T>(offsetBytes, 4 * CI * CO);
            rawOutput = FastAlloc<T>(offsetBytes, 1024);
            SplineCCopy(rawWeights, weightsGm, 4 * CI * CO);
        }
        if constexpr (!alignedRows) {
            // NN covers the entire actual input span. Its upper row bound also
            // covers every logical output row, without specializing exact N.
            constexpr uint32_t zeroElements = HintWordCount > 0
                ? (HintWordCount * 64 / (CI * sizeof(T))) * CO : 1024;
            static_assert(zeroElements > 0 && zeroElements <= 1024, "Whole logical output must fit");
            SplineCDuplicate(output, 0.0f, zeroElements);
        }
    }

    __aicore__ inline void ReadOutputExtent() {
        if constexpr (!alignedRows) {
            if constexpr (exactHintRows)
                nodeCount = hintMaxRows;
            else
                nodeCount = metadataGm.GetValue(0) & kNodeCountMask;
        }
    }

    __aicore__ inline void ComputeSmallFloat(GM_ADDR xp) {
        // The actual source index determines the feature row. All four
        // support products are independent of the later coordinate reads.
        static_assert(alignedRows, "FP32 small outputs contain complete DMA blocks");
        if constexpr (packedSmallFeature) {
            SplineCCopy(feature, x[uint64_t(source) * CI], CI);
            Fence<HardEvent::MTE2_V>();
            PrepareSmallPackedSupportDmaGemv<CI, CO>(feature, weights, result, broadcast);
        } else {
            PrefetchActualFeatureRow<T>(xp, source, CI);
            float features[CI];
            ReadFloatScalarRow<CI>(xp, source, features);
            Fence<HardEvent::MTE2_V>();
            PrepareSmallSupportFirstGemv<CI, CO>(features, weights);
        }
        const float u = PseudoValue(coordsGm, coordinateBits, 0);
        const float v = PseudoValue(coordsGm, coordinateBits, 1);
        if constexpr (packedSmallFeature)
            FinishSmallSupportGemv<CO>(u, v, result, result);
        else
            FinishSmallSupportGemv<CO>(u, v, weights, result);
        target = uint32_t(edgesGm.GetValue(1));
    }

    __aicore__ inline void ComputeOverlapped(GM_ADDR xp) {
        if constexpr (packedSmallFeature)
            SplineCCopy(feature, x[uint64_t(source) * CI], CI);
        else
            PrefetchActualFeatureRow<T>(xp, source, CI);
        if constexpr (CI > 16 && CI * sizeof(T) < 64) {
            // A48-byte BF16 row can cross a64-byte cache line. Hint its
            // actual final word as well; both addresses stay inside the row.
            GlobalTensor<uint64_t> lastFeatureWord;
            lastFeatureWord.SetGlobalBuffer((__gm__ uint64_t*)((__gm__ uint8_t*)xp +
                uint64_t(source) * CI * sizeof(T) + CI * sizeof(T) - sizeof(uint64_t)));
            DataCachePreload(lastFeatureWord, int64_t(0));
        }
        target = uint32_t(edgesGm.GetValue(1));
        const float u = PseudoValue(coordsGm, coordinateBits, 0);
        const float v = PseudoValue(coordsGm, coordinateBits, 1);
        // Unaligned BF16 output keeps its original sole whole-output owner.
        // It still needs the actual runtime N for exact-length final storage.
        ReadOutputExtent();
        if constexpr (sizeof(T) == 4) {
            if constexpr (packedSmallFeature) {
                Fence<HardEvent::MTE2_V>();
                SmallPackedSupportDmaGemv<CI, CO>(u, v, feature, weights, result, broadcast);
            } else {
                float features[CI];
                ReadFloatScalarRow<CI>(xp, source, features);
                Fence<HardEvent::MTE2_V>();
                SmallSupportFirstGemv<CI, CO>(features, u, v, weights, result);
            }
        } else {
            Fence<HardEvent::MTE2_V>();
            SplineCCast(weights, rawWeights, RoundMode::CAST_NONE, 4 * CI * CO);
            PipeBarrier<PIPE_V>();
            BlendSmallEffectiveKernel<CI, CO>(u, v, weights, result);
            FinishScalarRowGemv<T, CI, CO>(xp, source, result);
        }
    }

    __aicore__ inline void ComputeGeneral(GM_ADDR xp) {
        if constexpr (CI <= 16) {
            GlobalTensor<uint16_t> featureBits;
            featureBits.SetGlobalBuffer((__gm__ uint16_t*)xp);
            // No data-derived lookup is retained between calls. Every channel
            // of the actual source row is read and converted on this invocation.
            for (uint32_t channel = 0; channel < CI; ++channel)
                SplineCSet(feature, channel, PseudoValue(x, featureBits, source * CI + channel));
            Fence<HardEvent::S_V>();
        } else {
            StageFeature<T, CI>(x, source, feature, rawFeature);
        }
        target = uint32_t(edgesGm.GetValue(1));
        const float u = PseudoValue(coordsGm, coordinateBits, 0);
        const float v = PseudoValue(coordsGm, coordinateBits, 1);
        ReadOutputExtent();
        Fence<HardEvent::MTE2_V>();
        if constexpr (sizeof(T) != 4) {
            SplineCCast(weights, rawWeights, RoundMode::CAST_NONE, 4 * CI * CO);
            if constexpr (CI > 16)
                SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, CI);
            PipeBarrier<PIPE_V>();
        }
        FastBilinear<CI, CO, (sizeof(T) != 4)>(u, v, feature, weights, result, broadcast, messageSum);
    }

    __aicore__ inline void StoreMessage() {
        if constexpr (alignedRows) {
            FastStore(outputGm, uint64_t(target) * CO, result, rawOutput, CO);
        } else {
            // Unaligned BF16 rows share DMA blocks. One owner emits the complete
            // tensor so a zero-complement writer cannot overwrite a message row.
            SplineCAdds(output[target * CO], result, 0.0f, CO);
            PipeBarrier<PIPE_V>();
            FastStore(outputGm, 0, output, rawOutput, nodeCount * CO);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Run(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling) {
        const uint32_t worker = GetBlockIdx();
        if (worker != 0) {
            if constexpr (alignedRows)
                if (worker == 1)
                    ZeroSingleEdgeComplement<T, CO, 0, true>(ep, yp, tiling);
            return;
        }
        Bind(xp, ep, ap, wp, yp, tiling);
        StageWeights();
        source = uint32_t(edgesGm.GetValue(0));
        target = 0;
        nodeCount = 0;
        if constexpr (sizeof(T) == 4 && overlapFeature && !scalarSupportFamily)
            ComputeSmallFloat(xp);
        else if constexpr (overlapFeature)
            ComputeOverlapped(xp);
        else
            ComputeGeneral(xp);
        StoreMessage();
    }

};

template <class T, uint32_t CI, uint32_t CO, uint32_t HintWordCount = 0>
__aicore__ inline void RuntimeSingleEdge(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp,
                                       GM_ADDR yp, GM_ADDR tiling) {
    RuntimeSingleEdgeWorker<T, CI, CO, HintWordCount> worker;
    worker.Run(xp, ep, ap, wp, yp, tiling);
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void EvaluateCachedFloatRow(float u, float v, LocalTensor<float> row,
                                              LocalTensor<float> weights, LocalTensor<float> products,
                                              LocalTensor<float> broadcast, LocalTensor<float> destination) {
    static_assert(CO <= 32 && CI * CO >= 128 && (CI & (CI - 1)) == 0,
                  "Existing complete-input cache family uses packed support reductions.");
    float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
#pragma unroll
    for (uint32_t column = 0; column < CO; column += 8)
        SplineCBrcb(broadcast[column], row, CI / 8, {CO / 8, CO});
    PipeBarrier<PIPE_V>();
    EvaluatePackedMessageToOutput<CI, CO>(coefficients, weights, products, broadcast,
                                         destination);
}

// Host admits only the complete actual input in full32-byte DMA blocks. Every
// legal source row is initialized before its aligned vector input view.
template <uint32_t CI, uint32_t CO, uint32_t InputDmaBlocks>
__aicore__ inline void CachedFloatSingleEdge(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap,
                                              GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling) {
    static_assert(InputDmaBlocks > 0 && InputDmaBlocks <= 16, "Reserved512byte actual-input slab");
    constexpr uint32_t prefixElements = InputDmaBlocks * kUbAlignmentBytes / sizeof(float);
    constexpr uint32_t matrixElements = CI * CO;
    static_assert(prefixElements % CI == 0 && CO % 16 == 0 && matrixElements <= 256,
                  "Complete actual rows and full support-vector resource family");
    constexpr uint32_t outputElements = (prefixElements / CI) * CO;
    static_assert(outputElements <= 1024, "Complete small-output Host capacity");
    if (GetBlockIdx() != 0)
        return;
    GlobalTensor<float> x, weightsGm, coordsGm, outputGm;
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<uint64_t> edgeCacheLine, coordinateCacheLine;
    GlobalTensor<uint16_t> coordinateBits;
    x.SetGlobalBuffer((__gm__ float*)xp);
    weightsGm.SetGlobalBuffer((__gm__ float*)wp);
    coordsGm.SetGlobalBuffer((__gm__ float*)ap);
    outputGm.SetGlobalBuffer((__gm__ float*)yp);
    edgesGm.SetGlobalBuffer((__gm__ int64_t*)ep);
    edgeCacheLine.SetGlobalBuffer((__gm__ uint64_t*)ep);
    coordinateCacheLine.SetGlobalBuffer((__gm__ uint64_t*)ap);
    coordinateBits.SetGlobalBuffer((__gm__ uint16_t*)ap);
    DataCachePreload(edgeCacheLine, int64_t(0));
    DataCachePreload(coordinateCacheLine, int64_t(0));
    LocalTensor<float> inputPrefix(TPosition::VECCALC, 0, prefixElements);
    SplineCCopy(inputPrefix, x, prefixElements);
    uint32_t offsetBytes = kSparseControlBytes;
    auto weights = SmallSingleAlloc<float>(offsetBytes, 4 * matrixElements);
    offsetBytes += 4 * kUbAlignmentBytes;
    auto products = SmallSingleAlloc<float>(offsetBytes, 4 * matrixElements);
    offsetBytes += 5 * kUbAlignmentBytes;
    auto broadcast = SmallSingleAlloc<float>(offsetBytes, matrixElements);
    auto messageSum = SmallSingleAlloc<float>(offsetBytes, CO);
    auto wholeOutput = SmallSingleAlloc<float>(offsetBytes, outputElements);
    LocalTensor<float> unusedRawOutput(TPosition::VECCALC, 0, 1);
    SplineCDuplicate(wholeOutput, 0.0f, outputElements);
    // One real contiguous four-support DMA replaces four transposed packets.
    // The complete cached input makes the real source view independent of
    // another source-row DMA or a scalar-facing input-ready token.
    SplineCCopy(weights, weightsGm, 4 * matrixElements);
    const uint32_t source = uint32_t(edgesGm.GetValue(0));
    const uint32_t target = uint32_t(edgesGm.GetValue(1));
    const float u = PseudoValue(coordsGm, coordinateBits, 0);
    const float v = PseudoValue(coordsGm, coordinateBits, 1);
    Fence<HardEvent::MTE2_V>();
    EvaluateCachedFloatRow<CI, CO>(u, v, inputPrefix[source * CI], weights, products, broadcast,
                                  wholeOutput[target * CO]);
    FastStore(outputGm, 0, wholeOutput, unusedRawOutput, outputElements);
    PipeBarrier<PIPE_ALL>();
}

// Host admits complete input DMA blocks in a partial64-byte cache line. Every
// legal source row is initialized before its scalar feature read.
template <uint32_t CI, uint32_t CO, uint32_t InputDmaBlocks>
struct PartialOutputWorker {
    static constexpr uint32_t blockElements = kUbAlignmentBytes / sizeof(float);
    static constexpr uint32_t partElements = kGmCacheLineBytes / sizeof(float);
    static constexpr uint32_t messageWorkers = CO / partElements;
    // Match the existing Host contract: this path launches eight AIV workers.
    static constexpr uint32_t zeroWorkers = 1;
    static constexpr uint32_t inputElements = InputDmaBlocks * blockElements;
    static constexpr uint32_t nodeCount = inputElements / CI;
    static constexpr uint32_t matrixElements = CI * CO;
    static constexpr uint32_t supportPitchElements = matrixElements + blockElements;
    static_assert(InputDmaBlocks > 0 && InputDmaBlocks <= 16 && inputElements % CI == 0,
                  "Complete actual rows fit the reserved input slab.");
    static_assert(CO % 16 == 0 && CO <= 32 && CI % 8 == 0 && matrixElements <= 256,
                  "Existing full-cache aligned small-matrix resource family.");
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<uint64_t> edgeCacheLine;
    GlobalTensor<float> outputGm;
    GlobalTensor<float> x, weightsGm, coordsGm;
    GlobalTensor<uint64_t> coordinateCacheLine;
    LocalTensor<float> input, coordinates, weights, products, broadcast;
    uint32_t worker, target, channelOffset;
    float u, v;

    __aicore__ inline void ZeroComplement() {
        const uint32_t target = uint32_t(edgesGm.GetValue(1));
        constexpr uint32_t rowsPerZeroWorker = (nodeCount + zeroWorkers - 1) / zeroWorkers;
        const uint32_t beginRow = (worker - messageWorkers) * rowsPerZeroWorker;
        const uint32_t limitRow = beginRow + rowsPerZeroWorker;
        const uint32_t endRow = limitRow < nodeCount ? limitRow : nodeCount;
        if (beginRow >= endRow)
            return;
        LocalTensor<float> zero(TPosition::VECCALC, 0, rowsPerZeroWorker * CO);
        SplineCDuplicate(zero, 0.0f, rowsPerZeroWorker * CO);
        Fence<HardEvent::V_MTE3>();
        if (target < beginRow || target >= endRow) {
            SplineCCopy(outputGm[beginRow * CO], zero, (endRow - beginRow) * CO);
        } else {
            if (beginRow < target)
                SplineCCopy(outputGm[beginRow * CO], zero, (target - beginRow) * CO);
            if (target + 1 < endRow)
                SplineCCopy(outputGm[(target + 1) * CO], zero, (endRow - target - 1) * CO);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StageInputs(GM_ADDR xp, GM_ADDR ap, GM_ADDR wp) {
        x.SetGlobalBuffer((__gm__ float*)xp);
        weightsGm.SetGlobalBuffer((__gm__ float*)wp);
        coordsGm.SetGlobalBuffer((__gm__ float*)ap);
        coordinateCacheLine.SetGlobalBuffer((__gm__ uint64_t*)ap);
        if constexpr (InputDmaBlocks % 2 == 0)
            DataCachePreload(coordinateCacheLine, int64_t(0));
        input = LocalTensor<float>(TPosition::VECCALC, 0, inputElements);
        SplineCCopy(input, x, inputElements);
        coordinates = LocalTensor<float>(TPosition::VECCALC, kCachedFloatInputCapacityBytes, 2);
        if constexpr (InputDmaBlocks % 2 != 0) {
            // Preserve the qualified partial-cache coordinate DMA and scalar
            // readiness event. It never aliases the at-most-512-byte input slab.
            SplineCCopyPad(coordinates, coordsGm, {1, 2 * sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
        uint32_t offsetBytes = kSparseControlBytes;
        weights = SmallSingleAlloc<float>(offsetBytes, 4 * supportPitchElements);
        products = SmallSingleAlloc<float>(offsetBytes, (CI / 2) * 4 * partElements);
        offsetBytes += kUbAlignmentBytes;
        broadcast = SmallSingleAlloc<float>(offsetBytes, CI * blockElements);
        // The public non-padding DMA API counts burst length and gaps in 32B
        // blocks. Each actual support matrix is contiguous; its 32B UB gap is
        // never read. The odd support pitch avoids coincident operand banks.
        asc_copy_gm2ub((__ubuf__ void*)weights.GetPhyAddr(), (__gm__ void*)weightsGm.GetPhyAddr(),
                      uint16_t(4), uint16_t(matrixElements / blockElements), uint16_t(0), uint16_t(1));
    }

    __aicore__ inline void BroadcastFeature() {
        // Submit the complete input and weights before scalar endpoint reads.
        const uint32_t sourceRow = uint32_t(edgesGm.GetValue(0));
        target = uint32_t(edgesGm.GetValue(1));
        if constexpr (InputDmaBlocks % 2 != 0) {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            u = SplineCGet(coordinates, 0);
            v = SplineCGet(coordinates, 1);
        } else {
            u = coordsGm.GetValue(0);
            v = coordsGm.GetValue(1);
        }
        Fence<HardEvent::MTE2_V>();
        SplineCBrcb(broadcast, input[sourceRow * CI], CI / 8, {1, 8});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeProducts() {
        channelOffset = worker * partElements;
        // Each output owner writes complete 64B cache lines. Two masked
        // eight-column slices fill separate halves of every support subrow;
        // they reuse one narrow feature broadcast and all actual weight bursts.
        SetMaskNorm();
        asc_set_vector_mask(0, (uint64_t(1) << 32) - 1);
#pragma unroll
        for (uint32_t column = 0; column < partElements; column += blockElements)
            asc_mul(SplinePtr(products[column]), SplinePtr(weights[channelOffset + column]), SplinePtr(broadcast),
                    uint8_t(CI / 2), uint8_t(partElements / blockElements),
                    uint8_t(supportPitchElements / blockElements), 0,
                    uint8_t(4 * partElements / blockElements), uint8_t(CO / blockElements), 1);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t column = 0; column < partElements; column += blockElements)
            asc_fma(SplinePtr(products[column]),
                    SplinePtr(weights[(CI / 2) * CO + channelOffset + column]),
                    SplinePtr(broadcast[(CI / 2) * blockElements]), uint8_t(CI / 2),
                    uint8_t(partElements / blockElements), uint8_t(supportPitchElements / blockElements), 0,
                    uint8_t(4 * partElements / blockElements), uint8_t(CO / blockElements), 1);
        PipeBarrier<PIPE_V>();
        // Keep the qualified lower/upper CI FMA, then the same binary CI tree.
        FastReduce<CI / 2, 4 * partElements>(products);
    }

    __aicore__ inline void StoreSlice() {
        const float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
        // FastReduce explicitly ends in normal mode in this invocation.
        // Narrow that mask once to this owner's complete 64-byte output slice;
        // the same ordered coefficient operations then reuse it unchanged.
        static_assert(CI >= 4 && partElements == 16,
                      "A nonempty CI reduction establishes normal mode for a full output slice");
        asc_set_vector_mask(0, (uint64_t(1) << partElements) - 1);
        asc_mul_scalar(SplinePtr(products), SplinePtr(products), coefficients[0],
                       uint8_t(1), 1, 1, 8, 8);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t support = 1; support < 4; ++support) {
            asc_axpy(SplinePtr(products), SplinePtr(products[support * partElements]),
                     coefficients[support], uint8_t(1), 1, 1, 8, 8);
            PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::V_MTE3>();
        SplineCCopy(outputGm[target * CO + channelOffset], products, partElements);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Run(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling) {
        worker = GetBlockIdx();
        if (worker >= messageWorkers + zeroWorkers)
            return;
        edgesGm.SetGlobalBuffer((__gm__ int64_t*)ep);
        edgeCacheLine.SetGlobalBuffer((__gm__ uint64_t*)ep);
        outputGm.SetGlobalBuffer((__gm__ float*)yp);
        DataCachePreload(edgeCacheLine, int64_t(0));
        if (worker >= messageWorkers) {
            ZeroComplement();
            return;
        }
        StageInputs(xp, ap, wp);
        BroadcastFeature();
        ComputeProducts();
        StoreSlice();
    }

};

template <uint32_t CI, uint32_t CO, uint32_t InputDmaBlocks>
__aicore__ inline void PartitionPartialCachedOutput(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap,
                                              GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling) {
    PartialOutputWorker<CI, CO, InputDmaBlocks> worker;
    worker.Run(xp, ep, ap, wp, yp, tiling);
}

template <uint32_t CI, uint32_t CO, uint32_t InputDmaBlocks>
__aicore__ inline void CachedCoordinateSingleEdge(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap,
                                              GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling) {
    // The input extent selects only the coordinate transport; ownership and
    // exact support-first arithmetic are shared with the full-cache family.
    PartitionPartialCachedOutput<CI, CO, InputDmaBlocks>(xp, ep, ap, wp, yp, tiling);
}
