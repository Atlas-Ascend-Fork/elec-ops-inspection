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
#include "spline_edge_io.h"

// Allocate complete 32-byte blocks without adding another bank-phase shift.
// The single-edge family has already reserved these buffers in its UB budget.
template <class U>
__aicore__ inline LocalTensor<U> SmallSingleAlloc(uint32_t &offsetBytes, uint32_t elementCount) {
    LocalTensor<U> tensor(TPosition::VECCALC, offsetBytes, elementCount);
    offsetBytes += (elementCount * sizeof(U) + kUbAlignmentBytes - 1) /
                   kUbAlignmentBytes * kUbAlignmentBytes;
    return tensor;
}

// Hints and reads stay within the actual source row. Each row is an
// integral number of 64-bit words in the admitted channel families.
template <class T>
__aicore__ inline void PrefetchActualFeatureRow(GM_ADDR inputAddress, uint32_t sourceRow,
                                               uint32_t inputChannels) {
    const uint32_t rowBytes = inputChannels * sizeof(T);
    const uint64_t startByte = uint64_t(sourceRow) * rowBytes;
    for (uint32_t byteOffset = 0; byteOffset < rowBytes; byteOffset += 64) {
        GlobalTensor<uint64_t> cacheLine;
        cacheLine.SetGlobalBuffer((__gm__ uint64_t*)((__gm__ uint8_t*)inputAddress + startByte + byteOffset));
        DataCachePreload(cacheLine, int64_t(0));
    }
}

template <class T>
__aicore__ inline void ReadActualFeatureWords(GM_ADDR inputAddress, uint32_t sourceRow,
                                              uint32_t inputChannels, LocalTensor<float> feature,
                                              LocalTensor<T> rawFeature) {
    const uint32_t rowWords = inputChannels * sizeof(T) / sizeof(uint64_t);
    volatile __gm__ const uint64_t* source =
        ((__gm__ const uint64_t*)inputAddress) + uint64_t(sourceRow) * rowWords;
    LocalTensor<uint64_t> words = feature.template ReinterpretCast<uint64_t>();
    if constexpr (sizeof(T) != 4)
        words = rawFeature.template ReinterpretCast<uint64_t>();
    volatile __ubuf__ uint64_t* destination = (volatile __ubuf__ uint64_t*)words.GetPhyAddr();
    for (uint32_t begin = 0; begin < rowWords; begin += 8) {
        const uint32_t remaining = rowWords - begin;
        const uint64_t word0 = source[begin];
        const uint64_t word1 = source[begin + 1];
        uint64_t word2 = 0, word3 = 0, word4 = 0, word5 = 0, word6 = 0, word7 = 0;
        if (remaining >= 4) {
            word2 = source[begin + 2];
            word3 = source[begin + 3];
        }
        if (remaining >= 8) {
            word4 = source[begin + 4];
            word5 = source[begin + 5];
            word6 = source[begin + 6];
            word7 = source[begin + 7];
        }
        destination[begin] = word0;
        destination[begin + 1] = word1;
        if (remaining >= 4) {
            destination[begin + 2] = word2;
            destination[begin + 3] = word3;
        }
        if (remaining >= 8) {
            destination[begin + 4] = word4;
            destination[begin + 5] = word5;
            destination[begin + 6] = word6;
            destination[begin + 7] = word7;
        }
    }
    Fence<HardEvent::S_V>();
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void BlendSmallEffectiveKernel(float u, float v, LocalTensor<float> weights,
                                                 LocalTensor<float> effectiveKernel) {
    constexpr uint32_t matrixElements = CI * CO;
    const float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    SplineCMuls(effectiveKernel, weights, coefficients[0], matrixElements);
    PipeBarrier<PIPE_V>();
    for (uint32_t support = 1; support < 4; ++support) {
        SplineCAxpy(effectiveKernel, weights[support * matrixElements], coefficients[support], matrixElements);
        PipeBarrier<PIPE_V>();
    }
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void FinishStreamedGemv(LocalTensor<float> feature, LocalTensor<float> result,
                                         LocalTensor<float> broadcast);

// Read every channel of the actual row on this invocation. The complete
// raw row fits in scalar registers; no cross-call state or UB feature exists.
template <class T, uint32_t CI, uint32_t CO>
__aicore__ inline void FinishScalarRowGemv(GM_ADDR inputAddress, uint32_t sourceRow,
                                          LocalTensor<float> effectiveKernel) {
    constexpr uint32_t wordCount = CI * sizeof(T) / sizeof(uint64_t);
    volatile __gm__ const uint64_t* input =
        ((__gm__ const uint64_t*)inputAddress) + uint64_t(sourceRow) * wordCount;
    uint64_t words[wordCount];
#pragma unroll
    for (uint32_t word = 0; word < wordCount; ++word)
        words[word] = input[word];
    float features[CI];
#pragma unroll
    for (uint32_t channel = 0; channel < CI; ++channel) {
        union { uint32_t bits; float value; } converted;
        if constexpr (sizeof(T) == 4)
            converted.bits = uint32_t(words[channel / 2] >> (32 * (channel % 2)));
        else
            converted.bits = uint32_t((words[channel / 4] >> (16 * (channel % 4))) & 65535) << 16;
        features[channel] = converted.value;
    }
    // Independent lower-half rows can issue together. Upper rows remain
    // intact until their corresponding scalar multiply-add consumes them.
#pragma unroll
    for (uint32_t row = 0; row < CI / 2; ++row)
        SplineCMuls(effectiveKernel[row * CO], effectiveKernel[row * CO], features[row], CO);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t row = 0; row < CI / 2; ++row)
        SplineCAxpy(effectiveKernel[row * CO], effectiveKernel[(row + CI / 2) * CO],
                    features[row + CI / 2], CO);
    PipeBarrier<PIPE_V>();
    FastReduce<CI / 2, CO>(effectiveKernel);
}

// Keep each support's matrix-vector product separate until CI is reduced.
// The DMA layout is [CI, support, CO], so all four products share each
// scalar feature load and each reduction instruction. Basis blending last
// avoids overflowing a rounded convex weight sum before a small X scales it.
template <uint32_t CI>
__aicore__ inline void ReadFloatScalarRow(GM_ADDR inputAddress, uint32_t sourceRow,
                                         float* features) {
    volatile __gm__ const uint64_t* input =
        ((__gm__ const uint64_t*)inputAddress) + uint64_t(sourceRow) * (CI / 2);
#pragma unroll
    for (uint32_t word = 0; word < CI / 2; ++word) {
        const uint64_t bits = input[word];
        union { uint32_t bits; float value; } low, high;
        low.bits = uint32_t(bits);
        high.bits = uint32_t(bits >> 32);
        features[2 * word] = low.value;
        features[2 * word + 1] = high.value;
    }
}



// Small source rows use actual scalar values when they fit the port budget;
// wider families keep their row DMA. The function does not specialize N.
template <class T, uint32_t HintWordCount>
__aicore__ inline void HintBoundedInputSpan(GM_ADDR inputAddress) {
    static_assert(HintWordCount > 0 && HintWordCount <= 8, "Hint count exceeds its resource class.");
#pragma unroll
    for (uint32_t byteOffset = 0; byteOffset < HintWordCount * 64; byteOffset += 64) {
        GlobalTensor<uint64_t> hintWord;
        hintWord.SetGlobalBuffer((__gm__ uint64_t*)((__gm__ uint8_t*)inputAddress + byteOffset));
        DataCachePreload(hintWord, int64_t(0));
    }
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void PrepareSmallSupportFirstGemv(const float* features,
                                                   LocalTensor<float> products) {
    constexpr uint32_t supportRowElements = 4 * CO;
#pragma unroll
    for (uint32_t row = 0; row < CI / 2; ++row)
        SplineCMuls(products[row * supportRowElements], products[row * supportRowElements],
                    features[row], supportRowElements);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t row = 0; row < CI / 2; ++row)
        SplineCAxpy(products[row * supportRowElements], products[(row + CI / 2) * supportRowElements],
                    features[row + CI / 2], supportRowElements);
    PipeBarrier<PIPE_V>();
    FastReduce<CI / 2, supportRowElements>(products);
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void PrepareSmallPackedSupportDmaGemv(LocalTensor<float> feature,
                                                       LocalTensor<float> weights,
                                                       LocalTensor<float> products,
                                                       LocalTensor<float> broadcast) {
    constexpr uint32_t matrixElements = CI * CO;
    constexpr uint32_t supportRowElements = 4 * CO;
    static_assert(CI % 8 == 0 && CO % 8 == 0 && CO / 8 < 4,
                  "A product slice contains four distinct support blocks.");
#pragma unroll
    for (uint32_t column = 0; column < CO; column += 8)
        SplineCBrcb(broadcast[column], feature, CI / 8, {CO / 8, CO});
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t column = 0; column < CO; column += 8)
        SplineCMul(products[column], weights[column], broadcast[column], uint64_t(32), uint8_t(CI),
            {uint8_t(CO / 8), uint8_t(matrixElements / 8), 0,
             uint8_t(supportRowElements / 8), uint8_t(CO / 8), uint8_t(CO / 8)});
    PipeBarrier<PIPE_V>();
    FastReduce<CI, supportRowElements>(products);
}

template <uint32_t CO>
__aicore__ inline void FinishSmallSupportGemv(float u, float v,
                                             LocalTensor<float> products,
                                             LocalTensor<float> result) {
    const float coefficients[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v};
    SplineCMuls(result, products, coefficients[0], CO);
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (uint32_t support = 1; support < 4; ++support) {
        SplineCAxpy(result, products[support * CO], coefficients[support], CO);
        PipeBarrier<PIPE_V>();
    }
}

template <uint32_t CI, uint32_t CO>
__aicore__ inline void SmallSupportFirstGemv(const float* features, float u, float v,
                                            LocalTensor<float> products, LocalTensor<float> result) {
    PrepareSmallSupportFirstGemv<CI, CO>(features, products);
    FinishSmallSupportGemv<CO>(u, v, products, result);
}

// One contiguous four-support weight copy and one actual feature-row copy
// feed a shared masked product. Four eight-lane blocks gather the support
// matrices into [CI,4,CO] products; immutable weights never need a transpose.
template <uint32_t CI, uint32_t CO>
__aicore__ inline void SmallPackedSupportDmaGemv(float u, float v, LocalTensor<float> feature,
                                               LocalTensor<float> weights, LocalTensor<float> products,
                                               LocalTensor<float> broadcast) {
    PrepareSmallPackedSupportDmaGemv<CI, CO>(feature, weights, products, broadcast);
    FinishSmallSupportGemv<CO>(u, v, products, products);
}
