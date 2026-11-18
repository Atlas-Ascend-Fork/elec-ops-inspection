/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "spline_conv_tiling.h"

namespace AscendC {

template <bool Closed0, bool Closed1>
__aicore__ inline void SplineLinearFactors(float q0, float q1, float *f)
{
    if constexpr (Closed0) {
        q0 *= 2.0f;
        q0 = q0 > 1.0f ? 2.0f - q0 : q0;
    }
    if constexpr (Closed1) {
        q1 *= 2.0f;
        q1 = q1 > 1.0f ? 2.0f - q1 : q1;
    }
    f[0] = (1.0f - q0) * (1.0f - q1);
    f[1] = q0 * (1.0f - q1);
    f[2] = (1.0f - q0) * q1;
    f[3] = q0 * q1;
}

template <bool Closed0, bool Closed1, bool ExplicitBarrier = true>
__aicore__ inline void SplineSingleInterpolate(const LocalTensor<float> &dot, const LocalTensor<float> &weights,
                                               float q0, float q1)
{
    if constexpr (Closed0) {
        q0 *= 2.0f;
        q0 = q0 > 1.0f ? 2.0f - q0 : q0;
    }
    if constexpr (Closed1) {
        q1 *= 2.0f;
        q1 = q1 > 1.0f ? 2.0f - q1 : q1;
    }
    Muls(dot, weights, (1 - q0) * (1 - q1), 128);
    if constexpr (ExplicitBarrier) {
        PipeBarrier<PIPE_V>();
    }
    Axpy(dot, weights[128], q0 * (1 - q1), 128);
    if constexpr (ExplicitBarrier) {
        PipeBarrier<PIPE_V>();
    }
    Axpy(dot, weights[256], (1 - q0) * q1, 128);
    if constexpr (ExplicitBarrier) {
        PipeBarrier<PIPE_V>();
    }
    Axpy(dot, weights[384], q0 * q1, 128);
    if constexpr (ExplicitBarrier) {
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline float SplineBf16ToFloat(uint16_t value)
{
    union {
        uint32_t bits;
        float value;
    } converted;
    converted.bits = uint32_t(value) << 16;
    return converted.value;
}

__aicore__ inline float SplitSplineCoordinate(float v, uint32_t size, bool wrap, uint32_t &base)
{
    int32_t integral = int32_t(v);
    if (wrap) {
        if (v < float(integral)) {
            --integral;
        }
        int32_t wrapped = integral % int32_t(size);
        base = wrapped < 0 ? wrapped + int32_t(size) : wrapped;
    } else {
        base = integral;
    }
    return v - float(integral);
}

__aicore__ inline void SplineBasisFactors(float v, uint32_t degree, float factors[4])
{
    if (degree == 1) {
        factors[0] = 1 - v;
        factors[1] = v;
    } else if (degree == 2) {
        factors[0] = 0.5f * v * v - v + 0.5f;
        factors[1] = -v * v + v + 0.5f;
        factors[2] = 0.5f * v * v;
    } else if (degree == 0) {
        factors[0] = 1.0f;
    } else {
        float v2 = v * v;
        float v3 = v2 * v;
        float u = 1 - v;
        factors[0] = u * u * u / 6;
        factors[1] = (3 * v3 - 6 * v2 + 4) / 6;
        factors[2] = (-3 * v3 + 3 * v2 + 3 * v + 1) / 6;
        factors[3] = v3 / 6;
    }
}

__aicore__ inline uint32_t SplineWeightIndex(uint32_t slot, uint32_t dim, uint32_t degree, const uint32_t sizes[8],
                                             const float factors[8][4], const uint32_t base[8], float &factor)
{
    uint32_t weight = 0;
    uint32_t stride = 1;
    factor = 1;
    for (uint32_t d = 0; d < dim; ++d) {
        uint32_t component = slot % (degree + 1);
        slot /= degree + 1;
        weight += ((base[d] + component) % sizes[d]) * stride;
        stride *= sizes[d];
        factor *= factors[d][component];
    }
    return weight;
}

__aicore__ inline void SplineCompensatedAccumulate(const LocalTensor<float> &accum, const LocalTensor<float> &source,
                                                   const LocalTensor<float> &correction, const LocalTensor<float> &prod,
                                                   const LocalTensor<float> &delta, const LocalTensor<float> &sum,
                                                   float factor, uint32_t count)
{
    Muls(prod, source, factor, count);
    PipeBarrier<PIPE_V>();
    Sub(delta, prod, correction, count);
    PipeBarrier<PIPE_V>();
    Add(sum, accum, delta, count);
    PipeBarrier<PIPE_V>();
    Sub(correction, sum, accum, count);
    PipeBarrier<PIPE_V>();
    Sub(correction, correction, delta, count);
    PipeBarrier<PIPE_V>();
    DataCopy(accum, sum, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void SplineReduceDotRows(const LocalTensor<float> &y, const LocalTensor<float> &dot, uint32_t rows,
                                           uint32_t width)
{
    while (rows > 1) {
        uint32_t half = (rows + 1) / 2;
        Add(dot, dot, dot[half * width], (rows - half) * width);
        PipeBarrier<PIPE_V>();
        rows = half;
    }
    Add(y, y, dot, width);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void SplineDotMatrix(const LocalTensor<float> &y, const LocalTensor<float> &dot,
                                       const LocalTensor<float> &broadcast, const LocalTensor<float> &weights,
                                       const LocalTensor<float> &values, const SplineConvTilingData &t)
{
    Brcb(broadcast, values, uint8_t(t.ciPad / 8), BrcbRepeatParams{uint16_t(t.coPad / 8), uint16_t(t.coPad)});
    PipeBarrier<PIPE_V>();
    uint32_t first = t.coPad < 64 ? t.coPad : 64;
    BinaryRepeatParams params{1, 1, 0, uint8_t(t.coPad / 8), uint8_t(t.coPad / 8), uint8_t(t.coPad / 8)};
    Mul(dot, weights, broadcast, uint64_t(first), uint8_t(t.ci), params);
    if (t.coPad > 64) {
        Mul(dot[64], weights[64], broadcast, uint64_t(t.coPad - 64), uint8_t(t.ci), params);
    }
    PipeBarrier<PIPE_V>();
    SplineReduceDotRows(y, dot, t.ci, t.coPad);
}
__aicore__ inline void SplineFusedDotMatrix(const LocalTensor<float> &y, const LocalTensor<float> &dot,
                                            const LocalTensor<float> &broadcast, const LocalTensor<float> &weights,
                                            const LocalTensor<float> &values, uint32_t rows, uint32_t coPad)
{
    Brcb(broadcast, values, uint8_t((rows + 7) / 8), BrcbRepeatParams{1, 8});
    PipeBarrier<PIPE_V>();
    BinaryRepeatParams params{1, 1, 0, uint8_t(coPad / 8), uint8_t(coPad / 8), 1};
    uint32_t first = coPad < 64 ? coPad : 64;
    for (uint32_t r = 0; r < rows; r += 255) {
        uint32_t count = rows - r;
        if (count > 255) {
            count = 255;
        }
        Mul(dot[r * coPad], weights[r * coPad], broadcast[r * 8], uint64_t(first), uint8_t(count), params);
        if (coPad > 64) {
            Mul(dot[r * coPad + 64], weights[r * coPad + 64], broadcast[r * 8], uint64_t(coPad - 64), uint8_t(count),
                params);
        }
    }
    PipeBarrier<PIPE_V>();
    SplineReduceDotRows(y, dot, rows, coPad);
}
} // namespace AscendC
