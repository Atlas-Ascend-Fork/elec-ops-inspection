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
#include "spline_vector_ops.h"

template <uint32_t ROWS, uint32_t CO> __aicore__ inline void FastReduce(LocalTensor<float> result) {
    if constexpr (ROWS > 1) {
        constexpr uint32_t half = (ROWS + 1) / 2, count = (ROWS - half) * CO;
        if constexpr (count <= 64)
            SplineCAdd(result, result, result[half * CO], uint64_t(count), uint8_t(1), {1, 1, 1, 8, 8, 8});
        else if constexpr (count % 64 == 0 && count / 64 <= 255)
            SplineCAdd(result, result, result[half * CO], uint64_t(64), uint8_t(count / 64), {1, 1, 1, 8, 8, 8});
        else
            SplineCAdd(result, result, result[half * CO], count);
        PipeBarrier<PIPE_V>();
        FastReduce<half, CO>(result);
    }
}
// Distribute the four exact basis coefficients into a complete CI-by-CO
// kernel, then evaluate one GEMV. The original weights stay immutable for
// repeated destinations; each message rebuilds all coefficients and elements.
// This reassociates FP32 operations and requires separate precision testing.
template <uint32_t CI, uint32_t CO>
__aicore__ inline void ContractCanonicalBilinear(const float *coefficients,
                                               LocalTensor<float> feature, LocalTensor<float> weights,
                                               LocalTensor<float> result, LocalTensor<float> broadcast) {
    constexpr uint32_t matrixElements = CI * CO;
    static_assert(matrixElements % (2 * kFloatVectorElements) == 0,
                  "Both CI halves contain complete vectors.");
#pragma unroll
    for (uint32_t column = 0; column < CO; column += 8)
        SplineCBrcb(broadcast[column], feature, CI / 8, {CO / 8, CO});
    SplineCMuls(result, weights, coefficients[0], matrixElements);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 1; support < 4; ++support) {
        SplineCAxpy(result, weights[support * matrixElements], coefficients[support], matrixElements);
        PipeBarrier<PIPE_V>();
    }
    // Lower-half products overwrite only their consumed kernel elements.
    // Upper-half elements remain intact until the paired FMA completes.
    SetMaskNorm();
    asc_set_vector_mask(0, ~uint64_t(0));
    asc_mul(SplinePtr(result), SplinePtr(result), SplinePtr(broadcast),
            uint8_t(matrixElements / (2 * kFloatVectorElements)), 1, 1, 1, 8, 8, 8);
    PipeBarrier<PIPE_V>();
    asc_fma(SplinePtr(result), SplinePtr(result[matrixElements / 2]),
            SplinePtr(broadcast[matrixElements / 2]),
            uint8_t(matrixElements / (2 * kFloatVectorElements)), 1, 1, 1, 8, 8, 8);
    PipeBarrier<PIPE_V>();
    FastReduce<CI / 2, CO>(result);
}

// Advance one CI tree layer in each independent contiguous support matrix.
// All four stores are disjoint; one PIPE_V publishes the complete layer.
template <uint32_t RemainingRows, uint32_t CI, uint32_t CO>
__aicore__ inline void ReduceContiguousSupportLayer(LocalTensor<float> products) {
    constexpr uint32_t matrixElements = CI * CO;
    constexpr uint32_t halfRows = RemainingRows / 2;
    constexpr uint32_t count = halfRows * CO;
    if constexpr (count <= 64 || (count % 64 == 0 && count / 64 <= 255)) {
        constexpr uint32_t maskElements = count <= 64 ? count : 64;
        SetMaskNorm();
        asc_set_vector_mask(0, maskElements == 64 ? ~uint64_t(0) : (uint64_t(1) << maskElements) - 1);
#pragma unroll
        for (uint32_t support = 0; support < 4; ++support)
            asc_add(SplinePtr(products[support * matrixElements]),
                    SplinePtr(products[support * matrixElements]),
                    SplinePtr(products[support * matrixElements + halfRows * CO]),
                    uint8_t(count <= 64 ? 1 : count / 64), 1, 1, 1, 8, 8, 8);
    } else {
#pragma unroll
        for (uint32_t support = 0; support < 4; ++support)
            SplineCAdd(products[support * matrixElements], products[support * matrixElements],
                        products[support * matrixElements + halfRows * CO], count);
    }
    PipeBarrier<PIPE_V>();
}

// Four full-vector products use contiguous support storage first. After two
// CI layers, the next layer fits in half of the existing feature broadcast
// buffer. That layer simultaneously changes the UB layout, with no extra
// allocation. The remaining tree and basis sum preserve the original order.
template <uint32_t CI, uint32_t CO>
__aicore__ inline void EvaluateContiguousFourSupports(const float *coefficients,
                                                      LocalTensor<float> weights,
                                                      LocalTensor<float> products,
                                                      LocalTensor<float> broadcast) {
    static_assert(CI % 8 == 0, "The resource family has complete groups of eight CI rows.");
    constexpr uint32_t matrixElements = CI * CO;
    SetMaskNorm();
    asc_set_vector_mask(0, ~uint64_t(0));
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        asc_mul(SplinePtr(products[support * matrixElements]), SplinePtr(weights[support * matrixElements]),
                SplinePtr(broadcast), uint8_t(matrixElements / 64), 1, 1, 1, 8, 8, 8);
    PipeBarrier<PIPE_V>();
    ReduceContiguousSupportLayer<CI, CI, CO>(products);
    ReduceContiguousSupportLayer<CI / 2, CI, CO>(products);
    // The feature broadcast is dead after all products have completed.
    // Each support now stores its own CO subrow in this reused buffer.
    SetMaskNorm();
    asc_set_vector_mask(0, (uint64_t(1) << CO) - 1);
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        asc_add(SplinePtr(broadcast[support * CO]), SplinePtr(products[support * matrixElements]),
                SplinePtr(products[support * matrixElements + (CI / 8) * CO]),
                uint8_t(CI / 8), 1, 1, 1, uint8_t(4 * CO / 8), uint8_t(CO / 8), uint8_t(CO / 8));
    PipeBarrier<PIPE_V>();
    FastReduce<CI / 8, 4 * CO>(broadcast);
    SplineCMuls(products, broadcast, coefficients[0], CO);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 1; support < 4; ++support) {
        SplineCAxpy(products, broadcast[support * CO], coefficients[support], CO);
        PipeBarrier<PIPE_V>();
    }
}

// Full-vector chunks keep each support's nearby CI rows contiguous. The
// first lower/upper CI pair is produced directly in [chunk,4supports,64]
// order, using half of the existing product scratch and no UB transpose.
template <uint32_t ChunkCount>
__aicore__ inline void ReducePackedSupportChunks(LocalTensor<float> products) {
    if constexpr (ChunkCount > 1) {
        constexpr uint32_t count = (ChunkCount / 2) * 4 * kFloatVectorElements;
        // All chunk rows contain256 FP32 values, so every layer retains the
        // full64-lane mask established by the product instructions.
        asc_add(SplinePtr(products), SplinePtr(products), SplinePtr(products[count]),
                uint8_t(count / kFloatVectorElements), 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        ReducePackedSupportChunks<ChunkCount / 2>(products);
    }
}

template <uint32_t RemainingRows, uint32_t CO>
__aicore__ inline void ReduceRowsInsideSupportChunk(LocalTensor<float> products) {
    if constexpr (RemainingRows > 1) {
        constexpr uint32_t halfRows = RemainingRows / 2;
        constexpr uint32_t count = halfRows * CO;
        SetMaskNorm();
        asc_set_vector_mask(0, (uint64_t(1) << count) - 1);
        // Four independent64-float support slots remain in the first chunk.
        asc_add(SplinePtr(products), SplinePtr(products), SplinePtr(products[count]),
                uint8_t(4), 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        ReduceRowsInsideSupportChunk<halfRows, CO>(products);
    }
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void MultiplyPackedSupports(LocalTensor<float> weights, LocalTensor<float> products,
                                            LocalTensor<float> broadcast) {
    constexpr uint32_t matrixElements = CI * CO;
    constexpr uint32_t halfMatrixElements = matrixElements / 2;
    constexpr uint32_t rowsPerSupportChunk = kFloatVectorElements / CO;
    constexpr uint32_t chunkCount = halfMatrixElements / kFloatVectorElements;
    static_assert(chunkCount > 0 && (chunkCount & (chunkCount - 1)) == 0 && chunkCount <= 255,
                  "The binary CI tree needs a positive power-of-two repeat count.");
    static_assert(matrixElements % (2 * kFloatVectorElements) == 0,
                  "Each CI half must contain complete FP32 vectors.");
    static_assert(kFloatVectorElements % CO == 0,
                  "A support chunk contains complete CO rows.");
    SetMaskNorm();
    asc_set_vector_mask(0, ~uint64_t(0));
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        asc_mul(SplinePtr(products[support * kFloatVectorElements]),
                SplinePtr(weights[support * matrixElements]), SplinePtr(broadcast),
                uint8_t(chunkCount), 1, 1, 1, 32, 8, 8);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 0; support < 4; ++support)
        // Both CI halves contribute their real products to the same slots.
        // Device cancellation probes qualify multiply-add rounding on the target.
        asc_fma(SplinePtr(products[support * kFloatVectorElements]),
                SplinePtr(weights[support * matrixElements + halfMatrixElements]),
                SplinePtr(broadcast[halfMatrixElements]),
                uint8_t(chunkCount), 1, 1, 1, 32, 8, 8);
    PipeBarrier<PIPE_V>();
    ReducePackedSupportChunks<chunkCount>(products);
    ReduceRowsInsideSupportChunk<rowsPerSupportChunk, CO>(products);
}

template <uint32_t CO>
__aicore__ inline void BlendPackedRow(const float *coefficients, LocalTensor<float> products,
                                     LocalTensor<float> destination) {
    SplineCMuls(destination, products, coefficients[0], CO);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 1; support < 4; ++support) {
        SplineCAxpy(destination, products[support * kFloatVectorElements], coefficients[support], CO);
        PipeBarrier<PIPE_V>();
    }
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void EvaluatePackedFirstPair(const float *coefficients,
                                               LocalTensor<float> weights,
                                               LocalTensor<float> products,
                                               LocalTensor<float> broadcast) {
    MultiplyPackedSupports<CI, CO>(weights, products, broadcast);
    BlendPackedRow<CO>(coefficients, products, products);
}

// The complete-output owner can blend directly into its actual target row.
// Product scratch remains separate; no support is read after being overwritten.
template <uint32_t CI, uint32_t CO>
__aicore__ inline void EvaluatePackedMessageToOutput(const float *coefficients,
                                               LocalTensor<float> weights,
                                               LocalTensor<float> products,
                                               LocalTensor<float> broadcast,
                                               LocalTensor<float> outputRow) {
    constexpr uint32_t halfMatrixElements = CI * CO / 2;
    constexpr uint32_t rowsPerSupportChunk = kFloatVectorElements / CO;
    MultiplyPackedSupports<CI, CO>(weights, products, broadcast);
    // Host already emits separate CI/CO layout keys. Restrict this measured
    // instruction schedule to a CI half that occupies exactly one vector;
    // larger matrix layouts retain their original counted blend operations.
    if constexpr (halfMatrixElements == kFloatVectorElements) {
        // The final CI layer above explicitly establishes normal mode with
        // exactly CO active lanes in this invocation. The repeat overloads keep
        // that mask while preserving all four ordered, dependent operations.
        static_assert(rowsPerSupportChunk > 1 && CO <= 32,
                      "The final CI layer establishes the complete output-row mask");
        asc_mul_scalar(SplinePtr(outputRow), SplinePtr(products), coefficients[0],
                       uint8_t(1), 1, 1, 8, 8);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t support = 1; support < 4; ++support) {
            asc_axpy(SplinePtr(outputRow), SplinePtr(products[support * kFloatVectorElements]),
                     coefficients[support], uint8_t(1), 1, 1, 8, 8);
            PipeBarrier<PIPE_V>();
        }
    } else {
        BlendPackedRow<CO>(coefficients, products, outputRow);
    }
}

// Keep each knot's complete channel reduction before its basis coefficient.
// Interleaving independent supports in the output-channel dimension lets one
// vector reduction advance all of them without interpolating large weights.
// The existing product scratch budget bounds the batch width.
template <uint32_t CI, uint32_t CO, uint32_t supportBatchCount, bool CanonicalGrid>
__aicore__ inline void MultiplySupportBatch(uint32_t firstSupport, const uint32_t *knotIndices,
    LocalTensor<float> weights, LocalTensor<float> result, LocalTensor<float> broadcast) {
    constexpr uint32_t matrixElements = CI * CO, productRowElements = supportBatchCount * CO;
    if constexpr (CanonicalGrid && matrixElements / 8 <= 255 && CO / 8 < supportBatchCount) {
        // One active eight-lane block per support gathers a common
        // output-column slice from consecutive real knot matrices.
        // All strides count 32-byte UB blocks, and the support masks
        // store disjoint CO subrows inside each input-channel row.
#pragma unroll
        for (uint32_t column = 0; column < CO; column += 8) {
            SplineCMul(result[column], weights[firstSupport * matrixElements + column], broadcast[column],
                        uint64_t(supportBatchCount * 8), uint8_t(CI),
                        {uint8_t(CO / 8), uint8_t(matrixElements / 8), 0,
                         uint8_t(productRowElements / 8), uint8_t(CO / 8), uint8_t(CO / 8)});
        }
    } else {
#pragma unroll
        for (uint32_t support = 0; support < supportBatchCount; ++support) {
            auto knotWeight = weights[knotIndices[firstSupport + support] * matrixElements];
            if constexpr (CO <= 32) {
                SplineCMul(result[support * CO], knotWeight, broadcast, uint64_t(CO), uint8_t(CI),
                            {1, 1, 1, uint8_t(productRowElements / 8), uint8_t(CO / 8), uint8_t(CO / 8)});
            } else {
#pragma unroll
                for (uint32_t column = 0; column < CO; column += 64)
                    SplineCMul(result[support * CO + column], knotWeight[column], broadcast,
                                uint64_t(64), uint8_t(CI),
                                {1, 1, 0, uint8_t(productRowElements / 8), uint8_t(CO / 8), 1});
            }
        }
    }
    PipeBarrier<PIPE_V>();
}

template <uint32_t CO, uint32_t supportBatchCount, uint32_t SupportCount>
__aicore__ inline void AccumulateSupportBatch(uint32_t firstSupport, const float *coefficients,
    LocalTensor<float> result, LocalTensor<float> messageSum) {
    if constexpr (supportBatchCount == SupportCount) {
        // The first reduced support row can own the final message. Its
        // neighboring support rows remain intact until each is consumed.
        SplineCMuls(result, result, coefficients[0], CO);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t support = 1; support < supportBatchCount; ++support) {
            SplineCAxpy(result, result[support * CO], coefficients[support], CO);
            PipeBarrier<PIPE_V>();
        }
    } else {
#pragma unroll
        for (uint32_t support = 0; support < supportBatchCount; ++support) {
            SplineCAxpy(messageSum, result[support * CO], coefficients[firstSupport + support], CO);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void EvaluateCanonicalSupports(const float *coefficients, LocalTensor<float> weights,
                                                 LocalTensor<float> result, LocalTensor<float> broadcast) {
    if constexpr ((CI * CO) % (2 * kFloatVectorElements) == 0 && kFloatVectorElements % CO == 0 &&
                  (CI & (CI - 1)) == 0)
        EvaluatePackedFirstPair<CI, CO>(coefficients, weights, result, broadcast);
    else
        EvaluateContiguousFourSupports<CI, CO>(coefficients, weights, result, broadcast);
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void BroadcastFeatureRow(LocalTensor<float> feature, LocalTensor<float> broadcast) {
    if constexpr (CO <= 32) {
#pragma unroll
        for (uint32_t column = 0; column < CO; column += 8)
            SplineCBrcb(broadcast[column], feature, CI / 8, {CO / 8, CO});
    } else {
        SplineCBrcb(broadcast, feature, CI / 8, {1, 8});
    }
    PipeBarrier<PIPE_V>();
}

template <uint32_t CI, uint32_t CO, uint32_t SupportCount, bool CanonicalGrid = false, bool AllowWeightContraction = true>
__aicore__ inline void EvaluateSupportMessages(const uint32_t *knotIndices, const float *coefficients,
                                               LocalTensor<float> feature, LocalTensor<float> weights,
                                               LocalTensor<float> result, LocalTensor<float> broadcast,
                                               LocalTensor<float> messageSum) {
    constexpr uint32_t matrixElements = CI * CO;
    constexpr bool packedBilinear = CanonicalGrid && SupportCount == 4 && UsePackedBilinearWeights(CI, CO);
    if constexpr (AllowWeightContraction && packedBilinear &&
                  CO % 16 == 0 && CI >= 8 && (CI & (CI - 1)) == 0) {
        ContractCanonicalBilinear<CI, CO>(coefficients, feature, weights, result, broadcast);
        return;
    }
    // Canonical bilinear callers already reserve the full packed product
    // when its byte footprint fits the shared 32-KiB capacity contract.
    constexpr uint32_t supportBatchCount =
        packedBilinear ? 4 : SupportBatchWidth(matrixElements, SupportCount);
    constexpr uint32_t productRowElements = supportBatchCount * CO;
    if constexpr (supportBatchCount != SupportCount)
        SplineCDuplicate(messageSum, 0.0f, CO);
    BroadcastFeatureRow<CI, CO>(feature, broadcast);
    if constexpr (CanonicalGrid && SupportCount == 4 && supportBatchCount == 4 && CO <= 32 &&
                  matrixElements >= 2 * kFloatVectorElements) {
        EvaluateCanonicalSupports<CI, CO>(coefficients, weights, result, broadcast);
        return;
    }
#pragma unroll
    for (uint32_t firstSupport = 0; firstSupport < SupportCount; firstSupport += supportBatchCount) {
        MultiplySupportBatch<CI, CO, supportBatchCount, CanonicalGrid>(firstSupport, knotIndices, weights, result, broadcast);
        FastReduce<CI, productRowElements>(result);
        AccumulateSupportBatch<CO, supportBatchCount, SupportCount>(firstSupport, coefficients, result, messageSum);
    }
    if constexpr (supportBatchCount != SupportCount) {
        SplineCAdds(result, messageSum, 0.0f, CO);
        PipeBarrier<PIPE_V>();
    }
}

template <uint32_t CI, uint32_t CO, bool AllowWeightContraction = true>
__aicore__ inline void BuildBilinearProducts(float u, float v, LocalTensor<float> feature, LocalTensor<float> weight,
                                           LocalTensor<float> result, LocalTensor<float> broadcast,
                                           LocalTensor<float> messageSum) {
    // On an open 2x2 grid, canonical bilinear weighting extends to both
    // endpoints. Boundary wrapping only permutes zero-coefficient terms.
    uint32_t knotIndices[4] = {0, 1, 2, 3};
    float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    EvaluateSupportMessages<CI, CO, 4, true, AllowWeightContraction>(knotIndices, coefficients, feature, weight, result, broadcast, messageSum);
}
// The final channel reduction can store directly into an already cleared
// FP32 output row. Earlier tree levels keep the exact original buffer/order.
// No later code reads the old final row in the intermediate matrix.
template <uint32_t remainingRowCount, uint32_t CO>
__aicore__ inline void ReduceToOutputRow(LocalTensor<float> channelProducts, LocalTensor<float> outputRow) {
    static_assert(remainingRowCount > 1, "The aligned input-channel family has at least 8 rows.");
    constexpr uint32_t leadingRowCount = (remainingRowCount + 1) / 2;
    constexpr uint32_t mergedElementCount = (remainingRowCount - leadingRowCount) * CO;
    if constexpr (remainingRowCount == 2) {
        SplineCAdd(outputRow, channelProducts, channelProducts[CO], uint64_t(CO), uint8_t(1), {1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
    } else {
        if constexpr (mergedElementCount <= 64)
            SplineCAdd(channelProducts, channelProducts, channelProducts[leadingRowCount * CO],
                uint64_t(mergedElementCount), uint8_t(1), {1, 1, 1, 8, 8, 8});
        else if constexpr (mergedElementCount % 64 == 0 && mergedElementCount / 64 <= 255)
            SplineCAdd(channelProducts, channelProducts, channelProducts[leadingRowCount * CO],
                uint64_t(64), uint8_t(mergedElementCount / 64), {1, 1, 1, 8, 8, 8});
        else
            SplineCAdd(channelProducts, channelProducts, channelProducts[leadingRowCount * CO],
                mergedElementCount);
        PipeBarrier<PIPE_V>();
        ReduceToOutputRow<leadingRowCount, CO>(channelProducts, outputRow);
    }
}
template <uint32_t CI, uint32_t CO, bool AllowWeightContraction = true>
__aicore__ inline void FastBilinear(float u, float v, LocalTensor<float> feature, LocalTensor<float> weight,
                                  LocalTensor<float> result, LocalTensor<float> broadcast,
                                  LocalTensor<float> messageSum) {
    BuildBilinearProducts<CI, CO, AllowWeightContraction>(u, v, feature, weight, result, broadcast, messageSum);
}
template <uint32_t CI, uint32_t CO>
__aicore__ inline void BilinearToOutputRow(float u, float v, LocalTensor<float> feature,
                                          LocalTensor<float> weight, LocalTensor<float> result,
                                          LocalTensor<float> broadcast, LocalTensor<float> outputRow,
                                          LocalTensor<float> messageSum) {
    BuildBilinearProducts<CI, CO>(u, v, feature, weight, result, broadcast, messageSum);
    SplineCAdds(outputRow, result, 0.0f, CO);
    PipeBarrier<PIPE_V>();
}
template <uint32_t CI, uint32_t CO>
__aicore__ inline void IndexedBilinear(float u, float v, uint32_t k00, uint32_t k10, uint32_t k01,
                                      uint32_t k11, LocalTensor<float> feature, LocalTensor<float> weight,
                                      LocalTensor<float> result, LocalTensor<float> broadcast,
                                      LocalTensor<float> messageSum) {
    uint32_t knotIndices[4] = {k00, k10, k01, k11};
    float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    EvaluateSupportMessages<CI, CO, 4>(knotIndices, coefficients, feature, weight, result, broadcast, messageSum);
}
template <uint32_t CI, uint32_t CO>
__aicore__ inline void IndexedTrilinear(float u, float v, float z, uint32_t k00, uint32_t k10, uint32_t k01,
                                       uint32_t k11, uint32_t k00Upper, uint32_t k10Upper, uint32_t k01Upper,
                                       uint32_t k11Upper, LocalTensor<float> feature,
                                       LocalTensor<float> weight, LocalTensor<float> result,
                                       LocalTensor<float> broadcast, LocalTensor<float> messageSum) {
    uint32_t knotIndices[8] = {k00, k10, k01, k11, k00Upper, k10Upper, k01Upper, k11Upper};
    float coefficients[8] = {(1 - u) * (1 - v) * (1 - z), u * (1 - v) * (1 - z),
                             (1 - u) * v * (1 - z), u * v * (1 - z),
                             (1 - u) * (1 - v) * z, u * (1 - v) * z, (1 - u) * v * z, u * v * z};
    EvaluateSupportMessages<CI, CO, 8>(knotIndices, coefficients, feature, weight, result, broadcast, messageSum);
}
