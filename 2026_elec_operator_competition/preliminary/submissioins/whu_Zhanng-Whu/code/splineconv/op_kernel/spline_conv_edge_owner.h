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
#include "spline_conv_clear_owner.h"
#include "spline_conv_basis.h"
#include "spline_conv_wide_tile.h"
#include "spline_owner_buffers.h"

namespace AscendC {

template <bool Closed0, bool Closed1, bool DedicatedClear = false>
class SplineConvEdgeOwner : private SplineOwnerBuffers<32> {
    static constexpr uint32_t C = 32;

public:
    using SplineOwnerBuffers<32>::Init;
    __aicore__ inline void Process(GM_ADDR attributes = nullptr)
    {
        uint32_t block = GetBlockIdx();
        const uint32_t blocks = DedicatedClear ? 8 : GetBlockNum();
        if (blocks == 0) {
            return;
        }
        if constexpr (DedicatedClear) {
            block = (block >> 1) | ((block & 1) << 2);
            if (block >= 4) {
                SplineClearOwner<4, 8>(eg, yg, nodes, block, el, zeros);
                return;
            }
        }
        uint32_t bound = (nodes + blocks - 1) / blocks;
        uint32_t begin = block * bound;
        const uint32_t remaining = begin < nodes ? nodes - begin : 0;
        const uint32_t rows = remaining < bound ? remaining : bound;
        Prefetch(rows);
        int64_t target = block < edgeCount ? el.GetValue(edgeCount + block) : -1;
        const TargetGroup group = FindTargetGroup(block, target, begin, rows);
        if constexpr (!DedicatedClear) {
            if (rows) {
                spline_output::ClearInactiveRows(yg, zeros, begin, rows, group.occupied);
            }
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        if constexpr (!DedicatedClear) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        }
        if (!group.owner) {
            WaitData();
            return;
        }
        if (group.degree == 1) {
            Single(block);
        } else {
            ReduceGroup(target, attributes);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        spline_output::CopyRows(yg, dot, uint32_t(target), 1);
    }

private:
    struct TargetGroup {
        uint32_t degree = 0;
        uint32_t occupied = 0;
        bool owner = false;
    };

    __aicore__ inline void Prefetch(uint32_t rows)
    {
        DataCopyPad(el, eg, DataCopyExtParams{1, edgeCount * 16, 0, 0, 0},
                    DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        DataCopyPad(pr, pg, DataCopyExtParams{1, edgeCount * 8, 0, 0, 0},
                    DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        DataCopy(weights, wg, 4 * C * 16);
        if constexpr (DedicatedClear) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        DataCopy(features, xg, nodes * C);
        if constexpr (!DedicatedClear) {
            Duplicate(zeros, 0.0f, rows * 16);
        }
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    __aicore__ inline TargetGroup FindTargetGroup(uint32_t block, int64_t target, uint32_t begin, uint32_t rows)
    {
        TargetGroup group{};
        group.owner = target >= 0 && target < nodes;
        for (uint32_t edge = 0; edge < edgeCount; ++edge) {
            int64_t dst = el.GetValue(edgeCount + edge);
            if constexpr (!DedicatedClear) {
                if (dst >= begin && dst < int64_t(begin + rows)) {
                    group.occupied |= 1u << (uint32_t(dst) - begin);
                }
            }
            if (dst == target) {
                group.owner = group.owner && edge >= block;
                ++group.degree;
            }
        }
        return group;
    }

    __aicore__ inline void ReduceGroup(int64_t target, GM_ADDR attributes)
    {
        WaitData();
        Duplicate(agg, 0.0f, 4 * C);
        PipeBarrier<PIPE_ALL>();
        const uint32_t valid =
            SplineAccumulateTarget<C, Closed0, Closed1, false>(agg, features, el, pr, nodes, edgeCount, target);
        if (valid == 0) {
            Duplicate(dot, 0.0f, 16);
            return;
        }
        SplineConvWideTile<C>::GroupDot(dot, weights, agg, spread);
        uint32_t mean = average;
        if (attributes && valid != 1) {
            mean = reinterpret_cast<const volatile __gm__ SplineConvTilingData *>(attributes)->mean;
        }
        if (mean && valid != 1) {
            Muls(dot, dot, 1.0f / valid, 16);
        }
    }

    __aicore__ inline void WaitData()
    {
        if constexpr (DedicatedClear) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        }
    }
    __aicore__ inline void Factors(uint32_t e, float *f, uint32_t &wi)
    {
        SplineLinearFactors<Closed0, Closed1>(pr.GetValue(e * 2), pr.GetValue(e * 2 + 1), f);
        wi = 0;
    }
    __aicore__ inline void Single(uint32_t e)
    {
        int64_t src = el.GetValue(e);
        if (src < 0 || src >= nodes) {
            WaitData();
            Duplicate(dot, 0.0f, 16);
            return;
        }
        float f[4];
        uint32_t wi;
        Factors(e, f, wi);
        if constexpr (DedicatedClear) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        Muls(dot, weights[wi * C * 16], f[0], C * 16);
        for (uint32_t slot = 1; slot < 4; ++slot) {
            Axpy(dot, weights[(wi ^ slot) * C * 16], f[slot], C * 16);
        }
        if constexpr (DedicatedClear) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        }
        Brcb(spread, features[uint32_t(src) * C], uint8_t(C / 8), BrcbRepeatParams{1, 8});
        Mul(dot, dot, spread, uint64_t(16), uint8_t(C), BinaryRepeatParams{1, 1, 0, 2, 2, 1});
        Add(dot, dot[64], dot, uint64_t(64), uint8_t(7), BinaryRepeatParams{1, 1, 1, 0, 8, 0});
        Add(dot, dot, dot[32], uint64_t(32), uint8_t(1), BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        Add(dot, dot, dot[16], uint64_t(16), uint8_t(1), BinaryRepeatParams{1, 1, 1, 8, 8, 8});
    }
};

template <uint32_t Pad, bool Closed0, bool Closed1>
__aicore__ inline void RunFixedEdgeOwner(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                         GM_ADDR tiling)
{
    static_assert(Pad < 16);
    uint32_t blocks = GetBlockNum();
    if (blocks < 5 || blocks > 8) {
        return;
    }
    if (blocks == 8) {
        SplineConvEdgeOwner<Closed0, Closed1, true> op;
        op.Init(x, edges, pseudo, weight, out, 64 - Pad, 4, 0);
        op.Process(tiling);
        return;
    }
    uint32_t nodes = 16 * (blocks - 4) - Pad;
    SplineConvEdgeOwner<Closed0, Closed1> op;
    op.Init(x, edges, pseudo, weight, out, nodes, 4, 0);
    op.Process(tiling);
}

} // namespace AscendC
