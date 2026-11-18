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
#include "spline_conv_basis.h"
#include "spline_conv_wide_tile.h"
#include "spline_conv_sparse_buffers.h"

namespace AscendC {

template <uint32_t Nodes, bool Closed0, bool Closed1>
__aicore__ __attribute__((noinline)) void
SplineFourGroupDot(const GlobalTensor<int64_t> &edges, const GlobalTensor<float> &pseudo,
                   const GlobalTensor<float> &out, GM_ADDR tiling, uint32_t edge, const LocalTensor<float> &weights,
                   const LocalTensor<float> &features)
{
    constexpr uint32_t edgeCount = 4;
    if (edge >= edgeCount) {
        return;
    }
    const int64_t target = edges.GetValue(edgeCount + edge);
    if (target < 0 || target >= Nodes) {
        return;
    }
    for (uint32_t e = 0; e < edge; ++e) {
        if (edges.GetValue(edgeCount + e) == target) {
            return;
        }
    }
    LocalTensor<float> aggregate(TPosition::VECCALC, 40960, 128);
    LocalTensor<float> spread(TPosition::VECCALC, 41472, 1024);
    LocalTensor<float> dot(TPosition::VECCALC, 45568, 2048);
    Duplicate(aggregate, 0.0f, 128);
    PipeBarrier<PIPE_V>();
    const uint32_t valid = SplineAccumulateTarget<32, Closed0, Closed1, false, true>(aggregate, features, edges, pseudo,
                                                                                     Nodes, edgeCount, target);
    if (valid != 0) {
        SplineConvWideTile<32>::GroupDot<true>(dot, weights, aggregate, spread);
        if (valid > 1 && reinterpret_cast<const volatile __gm__ SplineConvTilingData *>(tiling)->mean) {
            Muls(dot, dot, 1.0f / valid, 16);
            PipeBarrier<PIPE_V>();
        }
    } else {
        Duplicate(dot, 0.0f, 16);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    spline_output::CopyRows(out, dot, uint32_t(target), 1);
    PipeBarrier<PIPE_ALL>();
}

} // namespace AscendC
