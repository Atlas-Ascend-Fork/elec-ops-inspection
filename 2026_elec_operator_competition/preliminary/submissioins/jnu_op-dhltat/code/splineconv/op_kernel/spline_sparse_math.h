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

// CI is restricted to powers of two by tiling. CO remains a small width family;
// the CI thresholds select reduction layers without creating CI instantiations.
template <uint32_t CO, uint32_t MaximumRows = 128>
__aicore__ inline void RuntimeFeatureReduce(uint32_t inputChannels, LocalTensor<float> values) {
    if constexpr (MaximumRows > 8) {
        if (inputChannels >= MaximumRows) {
            constexpr uint32_t count = MaximumRows / 2 * CO;
            if constexpr (count <= 64)
                SplineCAdd(values, values, values[count], uint64_t(count), uint8_t(1), {1, 1, 1, 8, 8, 8});
            else
                SplineCAdd(values, values, values[count], uint64_t(64), uint8_t(count / 64), {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
        }
        RuntimeFeatureReduce<CO, MaximumRows / 2>(inputChannels, values);
    } else
        FastReduce<8, CO>(values);
}
template <uint32_t CI, uint32_t CO, bool PrepareOnly = false, bool AllowWeightContraction = true>
__aicore__ inline void RuntimeBilinear(uint32_t inputChannels, float u, float v, LocalTensor<float> feature,
                                      LocalTensor<float> weights, LocalTensor<float> result,
                                      LocalTensor<float> broadcast, LocalTensor<float> messageSum) {
    // On an open 2x2 grid, canonical bilinear weighting extends to both
    // endpoints. Boundary wrapping only permutes zero-coefficient terms.
    uint32_t knotIndices[4] = {0, 1, 2, 3};
    float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    EvaluateSupportMessages<CI, CO, 4, true, AllowWeightContraction>(knotIndices, coefficients, feature, weights, result, broadcast, messageSum);
}
template <class T>
__aicore__ inline void StageRuntimeFeature(GlobalTensor<T> x, uint32_t source, uint32_t inputChannels,
                                           LocalTensor<float> feature, LocalTensor<T> rawFeature) {
    if constexpr (sizeof(T) == 4)
        SplineCCopy(feature, x[uint64_t(source) * inputChannels], inputChannels);
    else if (inputChannels % 16 == 0)
        SplineCCopy(rawFeature, x[uint64_t(source) * inputChannels], inputChannels);
    else
        SplineCCopyPad(rawFeature, x[uint64_t(source) * inputChannels],
                    {1, uint32_t(inputChannels * 2), 0, 0, 0}, {false, 0, 0, 0});
}
// Take the actual source row from the complete immutable input cache.
// FP32 uses a local view; BF16 keeps the original input dtype until the
// message helper casts just this row. Every valid source is in the cache.
template <class T>
__aicore__ inline void SelectCachedFeatureRow(LocalTensor<T> cachedInput, uint32_t source,
                                              uint32_t inputChannels, LocalTensor<float> &feature,
                                              LocalTensor<T> &rawFeature) {
    if constexpr (sizeof(T) == 4)
        feature = cachedInput[source * inputChannels];
    else
        rawFeature = cachedInput[source * inputChannels];
}

template <class T, uint32_t CI, uint32_t CO, bool FEATURE_CACHED = false>
__aicore__ inline void
FinishRuntimeMessage(uint32_t edge, uint32_t inputChannels, LocalTensor<T> coords, LocalTensor<float> feature,
                     LocalTensor<float> weights, LocalTensor<float> result, LocalTensor<float> broadcast,
                     LocalTensor<float> messageSum, LocalTensor<T> rawFeature, LocalTensor<T> rawWeight, bool &ready, bool firstRowPublished = false) {
    float u = LocalPseudo(coords, 2 * edge), v = LocalPseudo(coords, 2 * edge + 1);
    // The complete input/weight DMA is drained once for a cached sequence.
    // Later cached rows issue no MTE2 stores; vector instructions keep their
    // existing PIPE_V barriers and never modify the immutable cache.
    if (firstRowPublished && !ready)
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    else if (!FEATURE_CACHED || !ready)
        Fence<HardEvent::MTE2_V>();
    if constexpr (sizeof(T) != 4) {
        if (!ready)
            SplineCCast(weights, rawWeight, RoundMode::CAST_NONE, 4 * inputChannels * CO);
        SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, inputChannels);
        PipeBarrier<PIPE_V>();
    }
    ready = true;
    RuntimeBilinear<CI, CO, false, (sizeof(T) != 4)>(inputChannels, u, v, feature, weights, result, broadcast, messageSum);
}
// Each token publishes one complete support matrix. Event0 remains the
// first feature-row token; the four weight tokens are consumed exactly once.
template <class T, uint32_t CI, uint32_t CO>
__aicore__ inline void StageSupportWeightPackets(GlobalTensor<T> source, LocalTensor<float> weights,
                                               LocalTensor<T> rawWeights) {
    constexpr uint32_t matrixElements = CI * CO;
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support) {
        if constexpr (sizeof(T) == 4)
            SplineCCopy(weights[support * matrixElements], source[support * matrixElements], matrixElements);
        else
            SplineCCopy(rawWeights[support * matrixElements], source[support * matrixElements], matrixElements);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1 + support);
    }
}
__aicore__ inline void DrainSupportWeightPackets() {
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1 + support);
}
template <class T, uint32_t CI, uint32_t CO>
__aicore__ inline void BlendSupportWeightPackets(float u, float v, LocalTensor<float> weights,
                                               LocalTensor<T> rawWeights, LocalTensor<float> effectiveKernel) {
    constexpr uint32_t matrixElements = CI * CO;
    const float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1 + support);
        if constexpr (sizeof(T) != 4) {
            SplineCCast(weights[support * matrixElements], rawWeights[support * matrixElements],
                        RoundMode::CAST_NONE, matrixElements);
            PipeBarrier<PIPE_V>();
        }
        if (support == 0)
            SplineCMuls(effectiveKernel, weights, coefficients[0], matrixElements);
        else
            SplineCAxpy(effectiveKernel, weights[support * matrixElements], coefficients[support], matrixElements);
        PipeBarrier<PIPE_V>();
    }
}
// Effective-kernel storage can be overwritten by the consumed lower half.
// Wide CO uses compact scalar broadcast blocks; small CO uses full rows.
template <uint32_t CI, uint32_t CO>
__aicore__ inline void FinishStreamedGemv(LocalTensor<float> feature, LocalTensor<float> result,
                                         LocalTensor<float> broadcast) {
    constexpr uint32_t matrixElements = CI * CO;
    BroadcastFeatureRow<CI, CO>(feature, broadcast);
    SetMaskNorm();
    asc_set_vector_mask(0, ~uint64_t(0));
    if constexpr (CO <= 32) {
        asc_mul(SplinePtr(result), SplinePtr(result), SplinePtr(broadcast),
                uint8_t(matrixElements / 128), 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        asc_fma(SplinePtr(result), SplinePtr(result[matrixElements / 2]),
                SplinePtr(broadcast[matrixElements / 2]), uint8_t(matrixElements / 128),
                1, 1, 1, 8, 8, 8);
    } else {
#pragma unroll
        for (uint32_t column = 0; column < CO; column += 64)
            asc_mul(SplinePtr(result[column]), SplinePtr(result[column]), SplinePtr(broadcast),
                    uint8_t(CI / 2), 1, 1, 0, CO / 8, CO / 8, 1);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t column = 0; column < CO; column += 64)
            asc_fma(SplinePtr(result[column]), SplinePtr(result[matrixElements / 2 + column]),
                    SplinePtr(broadcast[(CI / 2) * 8]), uint8_t(CI / 2),
                    1, 1, 0, CO / 8, CO / 8, 1);
    }
    PipeBarrier<PIPE_V>();
    FastReduce<CI / 2, CO>(result);
}
template <class T, uint32_t CO, bool Published>
__aicore__ inline void StoreSparseZeroComplement(GlobalTensor<T> output, LocalTensor<T> zero,
                                                 uint32_t begin, uint32_t end, uint64_t occupiedRows) {
    if constexpr (Published) {
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    } else {
        if constexpr (sizeof(T) == 4)
            SplineCDuplicate(zero, 0.0f, (end - begin) * CO);
        else
            SplineCDuplicate(zero.template ReinterpretCast<uint16_t>(), uint16_t(0), (end - begin) * CO);
        Fence<HardEvent::V_MTE3>();
    }
    uint32_t cursor = end - begin;
    while (occupiedRows) {
        const uint32_t row = 63 - uint32_t(CountLeadingZero(occupiedRows));
        if (cursor > row + 1)
            SplineCCopyPad(output[uint64_t(begin + row + 1) * CO], zero,
                          {1, uint32_t((cursor - row - 1) * CO * sizeof(T)), 0, 0, 0});
        cursor = row;
        occupiedRows &= ~(uint64_t(1) << row);
    }
    if (cursor)
        SplineCCopyPad(output[uint64_t(begin) * CO], zero,
                      {1, uint32_t(cursor * CO * sizeof(T)), 0, 0, 0});
}
template <class U>
__aicore__ inline LocalTensor<U> SparseRuntimeAlloc(uint32_t &offsetBytes, uint32_t elementCount) {
    LocalTensor<U> buffer(TPosition::VECCALC, offsetBytes, elementCount);
    const uint32_t allocationBytes =
        (elementCount * sizeof(U) + kUbAlignmentBytes - 1) / kUbAlignmentBytes * kUbAlignmentBytes;
    offsetBytes += allocationBytes;
    return buffer;
}
// Two message slots share a channel-reduction instruction. Each slot keeps
// the original lower/upper CI pair and support-coefficient arithmetic.
template <uint32_t ChunkCount>
__aicore__ inline void ReducePairedMessageChunks(LocalTensor<float> products) {
    if constexpr (ChunkCount > 1) {
        constexpr uint32_t elements = (ChunkCount / 2) * 8 * kFloatVectorElements;
        asc_add(SplinePtr(products), SplinePtr(products), SplinePtr(products[elements]),
                uint8_t(elements / kFloatVectorElements), 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        ReducePairedMessageChunks<ChunkCount / 2>(products);
    }
}
template <uint32_t RemainingRows, uint32_t CO>
__aicore__ inline void ReducePairedRowsInsideChunk(LocalTensor<float> products) {
    if constexpr (RemainingRows > 1) {
        constexpr uint32_t halfRows = RemainingRows / 2;
        constexpr uint32_t count = halfRows * CO;
        SetMaskNorm();
        asc_set_vector_mask(0, (uint64_t(1) << count) - 1);
        asc_add(SplinePtr(products), SplinePtr(products), SplinePtr(products[count]),
                uint8_t(8), 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        ReducePairedRowsInsideChunk<halfRows, CO>(products);
    }
}
template <uint32_t CI, uint32_t CO>
__aicore__ inline void ProduceIncomingPairSlot(LocalTensor<float> feature,
                                               LocalTensor<float> weights,
                                               LocalTensor<float> products,
                                               LocalTensor<float> broadcast,
                                               uint32_t slot) {
    constexpr uint32_t matrixElements = CI * CO;
    constexpr uint32_t halfMatrixElements = matrixElements / 2;
    constexpr uint32_t chunkCount = halfMatrixElements / kFloatVectorElements;
#pragma unroll
    for (uint32_t column = 0; column < CO; column += 8)
        SplineCBrcb(broadcast[column], feature, CI / 8, {CO / 8, CO});
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    asc_set_vector_mask(0, ~uint64_t(0));
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        asc_mul(SplinePtr(products[(slot * 4 + support) * kFloatVectorElements]),
                SplinePtr(weights[support * matrixElements]), SplinePtr(broadcast),
                uint8_t(chunkCount), 1, 1, 1, 64, 8, 8);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        asc_fma(SplinePtr(products[(slot * 4 + support) * kFloatVectorElements]),
                SplinePtr(weights[support * matrixElements + halfMatrixElements]),
                SplinePtr(broadcast[halfMatrixElements]),
                uint8_t(chunkCount), 1, 1, 1, 64, 8, 8);
    PipeBarrier<PIPE_V>();
}
template <uint32_t CO>
__aicore__ inline void FinishIncomingPairSlot(float u, float v, LocalTensor<float> products) {
    const float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    BlendPackedRow<CO>(coefficients, products, products);
}
