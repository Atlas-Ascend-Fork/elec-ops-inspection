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
#include "spline_conv_edge_owner.h"
#include "spline_conv_four_group.h"

namespace AscendC {

template <uint32_t Nodes, bool Closed0, bool Closed1>
class SplineConvFourOwner {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out)
    {
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
        pg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pseudo));
        wg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight));
        yg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        eg.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(edges));
    }

    __aicore__ inline void Process(GM_ADDR edges, GM_ADDR pseudo, GM_ADDR tiling)
    {
        uint32_t block = GetBlockIdx();
        block = (block >> 1) | ((block & 1) << 2);
        if (block >= 4) {
            LocalTensor<int64_t> graph(TPosition::VECCALC, 0, 8);
            LocalTensor<float> zeros(TPosition::VECCALC, 128, 512);
            SplineClearOwner<4, 8>(eg, yg, Nodes, block, graph, zeros);
            return;
        }
        Prefetch(edges, pseudo);
        const int64_t source = eg.GetValue(block);
        const int64_t target = eg.GetValue(4 + block);
        const uint32_t bits = SplineTargetBits4(eg, target);
        if ((bits & (bits - 1)) != 0) {
            WaitInputs();
            PipeBarrier<PIPE_ALL>();
            SplineFourGroupDot<Nodes, Closed0, Closed1>(eg, pg, yg, tiling, block, weights, features);
            return;
        }
        if (target < 0 || target >= Nodes) {
            WaitInputs();
            return;
        }
        ComputeEdge(block, source);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        spline_output::CopyRows(yg, dot, uint32_t(target), 1);
    }

private:
    __aicore__ inline void Prefetch(GM_ADDR edges, GM_ADDR pseudo)
    {
        GlobalTensor<uint64_t> edgeCache;
        GlobalTensor<uint64_t> pseudoCache;
        edgeCache.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(edges));
        pseudoCache.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(pseudo));
        DataCachePreload(edgeCache, int64_t(0));
        DataCachePreload(pseudoCache, int64_t(0));
        SplineConvWideTile<32>::Prefetch(weights, wg);
        DataCopy(features, xg, Nodes * 32);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
    }

    __aicore__ inline void WaitInputs()
    {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
    }

    __aicore__ inline void ComputeEdge(uint32_t edge, int64_t source)
    {
        const bool validSource = source >= 0 && source < Nodes;
        float q0 = pg.GetValue(edge * 2);
        float q1 = pg.GetValue(edge * 2 + 1);
        if constexpr (Closed0) {
            q0 *= 2.0f;
            q0 = q0 > 1.0f ? 2.0f - q0 : q0;
        }
        if constexpr (Closed1) {
            q1 *= 2.0f;
            q1 = q1 > 1.0f ? 2.0f - q1 : q1;
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (validSource) {
            SetMaskNorm();
            SetVectorMask<float, MaskMode::NORMAL>(64);
            Muls<float, false>(dot, weights, (1.0f - q0) * (1.0f - q1), uint64_t(64), uint8_t(8),
                               UnaryRepeatParams{1, 1, 8, 8});
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Axpy<float, float, false>(dot, weights[512], q0 * (1.0f - q1), uint64_t(64), uint8_t(8),
                                      UnaryRepeatParams{1, 1, 8, 8});
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            Axpy<float, float, false>(dot, weights[1024], (1.0f - q0) * q1, uint64_t(64), uint8_t(8),
                                      UnaryRepeatParams{1, 1, 8, 8});
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
            Axpy<float, float, false>(dot, weights[1536], q0 * q1, uint64_t(64), uint8_t(8),
                                      UnaryRepeatParams{1, 1, 8, 8});
        } else {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        if (validSource) {
            uint32_t offset = uint32_t(source) * 32;
            Brcb(spread, features[offset], 4, BrcbRepeatParams{2, 16});
            Brcb(spread[8], features[offset], 4, BrcbRepeatParams{2, 16});
            Mul<float, false>(dot, dot, spread, uint64_t(64), uint8_t(8), BinaryRepeatParams{1, 1, 1, 8, 8, 8});
            Add<float, false>(dot, dot[64], dot, uint64_t(64), uint8_t(7), BinaryRepeatParams{1, 1, 1, 0, 8, 0});
            Add<float, false>(reduced, dot, dot[32], uint64_t(64), uint8_t(1), BinaryRepeatParams{1, 1, 1, 8, 8, 8});
            Add<float, false>(dot, reduced, reduced[16], uint64_t(64), uint8_t(1),
                              BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        } else {
            Duplicate(dot, 0.0f, 16);
        }
    }

    GlobalTensor<float> xg;
    GlobalTensor<float> pg;
    GlobalTensor<float> wg;
    GlobalTensor<float> yg;
    GlobalTensor<int64_t> eg;
    LocalTensor<float> weights{TPosition::VECCALC, 2304, 2048};
    LocalTensor<float> features{TPosition::VECCALC, 10496, 2048};
    LocalTensor<float> spread{TPosition::VECCALC, 24832, 512};
    LocalTensor<float> reduced{TPosition::VECCALC, 32768, 512};
    LocalTensor<float> dot{TPosition::VECCALC, 19712, 512};
};

template <uint32_t Pad, bool Closed0, bool Closed1>
__aicore__ inline void RunPartialEdgeOwner(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                           GM_ADDR tiling)
{
    SplineConvFourOwner<64 - Pad, Closed0, Closed1> op;
    op.Init(x, edges, pseudo, weight, out);
    op.Process(edges, pseudo, tiling);
}

} // namespace AscendC
