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
#include "spline_conv_sparse_buffers.h"
#include "spline_conv_wide_tile.h"
#include "spline_owner_buffers.h"

namespace AscendC {

template <bool Closed0, bool Closed1>
class SplineConvWideOwner : private SplineOwnerBuffers<128> {
    static constexpr uint32_t C = 128;

public:
    using SplineOwnerBuffers<128>::Init;
    __aicore__ inline void PrefetchGraph()
    {
        DataCopyPadExtParams<int64_t> epad{false, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(el, eg, DataCopyExtParams{1, edgeCount * 16, 0, 0, 0}, epad);
        DataCopyPad(pr, pg, DataCopyExtParams{1, edgeCount * 8, 0, 0, 0}, pad);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        SplineConvWideTile<>::Prefetch(weights, wg);
    }
    __aicore__ inline void SetShape(uint32_t n, uint32_t mean)
    {
        nodes = n;
        average = mean;
    }
    __aicore__ inline void Process()
    {
        uint32_t block = GetBlockIdx();
        uint32_t blocks = GetBlockNum();
        if (blocks == 0) {
            return;
        }
        // The host selects this path with one block per edge and at most 32 rows per block.
        uint32_t bound = (nodes + blocks - 1) / blocks;
        uint32_t zeroBegin = block * bound;
        uint32_t zeroRows = zeroBegin < nodes ? nodes - zeroBegin : 0;
        if (zeroRows > bound) {
            zeroRows = bound;
        }
        PrepareZeros(zeroRows);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        int64_t src = el.GetValue(block);
        if (src >= 0 && src < nodes) {
            DataCopy(features[block * C], xg[uint32_t(src) * C], C);
        }
        int64_t target = block < edgeCount ? el.GetValue(edgeCount + block) : -1;
        bool owner = target >= 0 && target < nodes;
        bool prepared = false;
        if (owner && src >= 0 && src < nodes) {
            Interpolate(block);
            prepared = true;
        }
        EdgeGroup group = CollectEdges(block, target, zeroBegin, zeroRows, owner);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
        if (zeroRows) {
            spline_output::ClearInactiveRows<float, true>(yg, zeros, zeroBegin, zeroRows, group.occupied);
        }
        if (!group.owner) {
            WaitForDotInputs(prepared);
            return;
        }
        if (group.degree == 1) {
            Single(block, prepared);
        } else {
            WaitForDotInputs(prepared);
            ReduceGroup(target);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        spline_output::CopyRows(yg, dot, uint32_t(target), 1);
    }

private:
    __aicore__ inline void WaitForDotInputs(bool weightsReady)
    {
        if (!weightsReady) {
            WaitWeights();
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
    }

    __aicore__ inline void PrepareZeros(uint32_t rows)
    {
        if (rows) {
            Duplicate(zeros, 0.0f, rows * 16);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        }
    }

    __aicore__ inline void WaitWeights()
    {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    }
    struct EdgeGroup {
        uint32_t degree = 0;
        uint32_t occupied = 0;
        bool owner = false;
    };

    __aicore__ inline EdgeGroup CollectEdges(uint32_t block, int64_t target, uint32_t begin, uint32_t rows, bool owner)
    {
        EdgeGroup group{};
        group.owner = owner;
        for (uint32_t e = 0; e < edgeCount; ++e) {
            int64_t dst = el.GetValue(edgeCount + e);
            if (dst >= begin && dst < int64_t(begin + rows)) {
                group.occupied |= 1u << (uint32_t(dst) - begin);
            }
            if (dst != target) {
                continue;
            }
            if (e < block) {
                group.owner = false;
            }
            ++group.degree;
            if (e != block) {
                int64_t src = el.GetValue(e);
                if (src >= 0 && src < nodes) {
                    DataCopy(features[e * C], xg[uint32_t(src) * C], C);
                }
            }
        }
        return group;
    }

    __aicore__ inline void ReduceGroup(int64_t target)
    {
        Duplicate(agg, 0.0f, 4 * C);
        PipeBarrier<PIPE_ALL>();
        const uint32_t valid =
            SplineAccumulateTarget<C, Closed0, Closed1, true>(agg, features, el, pr, nodes, edgeCount, target);
        if (valid == 0) {
            Duplicate(dot, 0.0f, 16);
            return;
        }
        SplineConvWideTile<>::GroupDot(dot, weights, agg, spread);
        if (average && valid != 1) {
            Muls(dot, dot, 1.0f / valid, 16);
        }
    }

    __aicore__ inline void Interpolate(uint32_t edge)
    {
        float factors[4];
        SplineLinearFactors<Closed0, Closed1>(pr.GetValue(edge * 2), pr.GetValue(edge * 2 + 1), factors);
        SplineConvWideTile<>::Interpolate(dot, weights, factors[0], factors[1], factors[2], factors[3]);
    }
    __aicore__ inline void Single(uint32_t e, bool prepared = false)
    {
        int64_t src = el.GetValue(e);
        if (src < 0 || src >= nodes) {
            WaitWeights();
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
            Duplicate(dot, 0.0f, 16);
            return;
        }
        if (!prepared) {
            Interpolate(e);
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        SplineConvWideTile<>::FeatureDot(dot, features[e * C], spread);
    }
};

template <bool Closed0, bool Closed1>
__aicore__ inline void RunWideEarlyOwner(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                         GM_ADDR tiling)
{
    const __gm__ SplineConvTilingData *td = (const __gm__ SplineConvTilingData *)tiling;
    SplineConvWideOwner<Closed0, Closed1> op;
    op.Init(x, edges, pseudo, weight, out, 0, GetBlockNum(), 0);
    op.PrefetchGraph();
    op.SetShape(td->n, td->mean);
    op.Process();
}

} // namespace AscendC
