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
#include "spline_conv_sparse_buffers.h"
#include "spline_conv_basis.h"

namespace AscendC {

template <uint32_t C = 128>
struct SplineConvWideTile {
    template <uint32_t Slot>
    __aicore__ static inline void Load(const LocalTensor<float> &local, const GlobalTensor<float> &global)
    {
        DataCopy(local[Slot * C * 16], global[Slot * C * 16], C * 16);
        SetFlag<HardEvent::MTE2_V>(TEventID(Slot));
    }
    __aicore__ static inline void Prefetch(const LocalTensor<float> &local, const GlobalTensor<float> &global)
    {
        Load<0>(local, global);
        Load<1>(local, global);
        Load<2>(local, global);
        Load<3>(local, global);
    }
    __aicore__ static inline void Interpolate(const LocalTensor<float> &dot, const LocalTensor<float> &weights,
                                              float f0, float f1, float f2, float f3)
    {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Muls(dot, weights, f0, C * 16);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Axpy(dot, weights[C * 16], f1, C * 16);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        Axpy(dot, weights[2 * C * 16], f2, C * 16);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
        Axpy(dot, weights[3 * C * 16], f3, C * 16);
    }
    template <uint32_t Rows, bool ExplicitBarrier = false>
    __aicore__ static inline void Reduce(const LocalTensor<float> &dot)
    {
        for (uint32_t r = Rows; r > 1; r /= 2) {
            Add(dot, dot, dot[r / 2 * 16], r / 2 * 16);
            if constexpr (ExplicitBarrier) {
                PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ static inline void FeatureDot(const LocalTensor<float> &dot, const LocalTensor<float> &features,
                                             const LocalTensor<float> &spread)
    {
        Brcb(spread, features, uint8_t(C / 8), BrcbRepeatParams{2, 16});
        Brcb(spread[8], features, uint8_t(C / 8), BrcbRepeatParams{2, 16});
        Mul(dot, dot, spread, C * 16);
        Reduce<C>(dot);
    }
    template <bool ExplicitBarrier = false>
    __aicore__ static inline void GroupDot(const LocalTensor<float> &dot, const LocalTensor<float> &weights,
                                           const LocalTensor<float> &agg, const LocalTensor<float> &spread)
    {
        Brcb(spread, agg, uint8_t(4 * C / 8), BrcbRepeatParams{1, 8});
        if constexpr (ExplicitBarrier) {
            PipeBarrier<PIPE_V>();
        }
        for (uint32_t row = 0; row < 4 * C; row += 255) {
            uint32_t count = 4 * C - row;
            if (count > 255) {
                count = 255;
            }
            Mul(dot[row * 16], weights[row * 16], spread[row * 8], uint64_t(16), uint8_t(count),
                BinaryRepeatParams{1, 1, 0, 2, 2, 1});
        }
        if constexpr (ExplicitBarrier) {
            PipeBarrier<PIPE_V>();
        }
        Reduce<4 * C, ExplicitBarrier>(dot);
    }
};

template <uint32_t Channels, bool Closed0, bool Closed1, bool PerEdge, bool ExplicitBarrier = false,
          typename EdgeTensor, typename PseudoTensor>
__aicore__ inline uint32_t SplineAccumulateTarget(const LocalTensor<float> &agg, const LocalTensor<float> &features,
                                                  const EdgeTensor &edges, const PseudoTensor &pseudo, uint32_t n,
                                                  uint32_t edgeCount, int64_t target)
{
    uint32_t valid = 0;
    for (uint32_t e = 0; e < edgeCount; ++e) {
        if (edges.GetValue(edgeCount + e) != target) {
            continue;
        }
        int64_t src = edges.GetValue(e);
        if (src < 0 || src >= n) {
            continue;
        }
        float factors[4];
        SplineLinearFactors<Closed0, Closed1>(pseudo.GetValue(e * 2), pseudo.GetValue(e * 2 + 1), factors);
        const auto values = features[(PerEdge ? e : uint32_t(src)) * Channels];
        for (uint32_t slot = 0; slot < 4; ++slot) {
            Axpy(agg[slot * Channels], values, factors[slot], Channels);
            if constexpr (ExplicitBarrier) {
                PipeBarrier<PIPE_V>();
            }
        }
        ++valid;
    }
    return valid;
}

} // namespace AscendC
