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

namespace AscendC {

template <bool Closed0, bool Closed1, bool Bulk>
class SplineConvSingleStage {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out)
    {
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
        pg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pseudo));
        wg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight));
        eg.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(edges));
        yg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
    }

    template <bool Tail>
    __aicore__ inline void ProcessChunk(uint32_t n, GM_ADDR edges, GM_ADDR pseudo)
    {
        const uint32_t block = GetBlockIdx();
        const uint32_t aligned = n / 2 * 2;
        const uint32_t active = Tail ? (aligned + 15) / 16 + 1 : (n + 15) / 16;
        if (block >= active) {
            return;
        }
        if constexpr (Bulk) {
            Preload(edges);
            Preload(pseudo);
        }
        LoadWeight();
        if constexpr (Bulk) {
            DataCopy(features, xg, n * 8);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        }
        const bool tail = Tail && block == (aligned + 15) / 16;
        const uint32_t start = tail ? aligned : block * 16;
        const uint32_t limit = Tail && !tail ? aligned : n;
        const uint32_t count = limit - start;
        const uint32_t rows = count > 16 ? 16 : count;
        LocalTensor<float> result(TPosition::VECCALC, Output, 256);
        Duplicate(result, 0.0f, rows * 16);
        const int64_t source = eg.GetValue(0);
        const int64_t target = eg.GetValue(1);
        const bool valid = IsEdgeInRows(source, target, n, start, rows);
        if (Compute(source, valid)) {
            DataCopy(result[(uint32_t(target) - start) * 16], dot, 16);
        }
        StoreRows(result, start, rows);
    }

    template <bool ClearAhead, bool LinearReduce = false, bool ExplicitBarrier = !LinearReduce>
    __aicore__ inline void ProcessBulk(uint32_t n)
    {
        LoadWeight();
        DataCopy(features, xg, n * 8);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        if constexpr (ClearAhead) {
            LocalTensor<float> zeros(TPosition::VECCALC, Output, 512);
            Duplicate(zeros, 0.0f, n * 16);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            spline_output::CopyRows(yg, zeros, 0, n);
        }
        const int64_t source = eg.GetValue(0);
        const int64_t target = eg.GetValue(1);
        const bool valid = IsEdgeInRows(source, target, n, 0, n);
        if (Compute<LinearReduce, ExplicitBarrier>(source, valid)) {
            StoreRows<ClearAhead>(dot, uint32_t(target), 1);
        }
    }

    __aicore__ inline void ClearTail(uint32_t n)
    {
        LocalTensor<float> zeros(TPosition::VECCALC, 0, 64 * 16);
        Duplicate(zeros, 0.0f, n * 16);
        const int64_t source = eg.GetValue(0);
        const int64_t target = eg.GetValue(1);
        const bool valid = IsEdgeInRows(source, target, n, 0, n);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if (valid) {
            const uint32_t row = uint32_t(target);
            spline_output::CopyRows(yg, zeros, 0, row);
            spline_output::CopyRows(yg, zeros, row + 1, n - row - 1);
        } else {
            spline_output::CopyRows(yg, zeros, 0, n);
        }
    }

    __aicore__ static inline void Preload(GM_ADDR address)
    {
        GlobalTensor<uint64_t> cache;
        cache.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(address));
        DataCachePreload(cache, int64_t(0));
    }

private:
    static constexpr uint32_t FeatureOffset = 2048;
    static constexpr uint32_t FeatureCount = Bulk ? 512 : 8;
    static constexpr uint32_t Spread = FeatureOffset + FeatureCount * 4;
    static constexpr uint32_t Dot = Spread + 256;
    static constexpr uint32_t Output = Dot + 512;

    __aicore__ static inline bool IsEdgeInRows(int64_t src, int64_t dst, uint32_t n, uint32_t start, uint32_t rows)
    {
        return src >= 0 && src < n && dst >= start && dst < int64_t(start + rows);
    }

    __aicore__ inline void LoadWeight()
    {
        DataCopyPad(weights, wg, DataCopyExtParams{1, 2048, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    template <bool LinearReduce = false, bool ExplicitBarrier = true>
    __aicore__ inline bool Compute(int64_t source, bool valid)
    {
        if constexpr (!Bulk) {
            if (valid) {
                DataCopyPad(features, xg[uint32_t(source) * 8], DataCopyExtParams{1, 32, 0, 0, 0},
                            DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
        }
        float q0 = 0.0f;
        float q1 = 0.0f;
        if (valid) {
            q0 = pg.GetValue(0);
            q1 = pg.GetValue(1);
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if (valid) {
            SplineSingleInterpolate<Closed0, Closed1, ExplicitBarrier>(dot, weights, q0, q1);
        }
        if (Bulk || valid) {
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        }
        if (valid) {
            MultiplyFeatures<LinearReduce>(Bulk ? uint32_t(source) * 8 : 0);
        }
        return valid;
    }

    template <bool LinearReduce>
    __aicore__ inline void MultiplyFeatures(uint32_t offset)
    {
        Brcb(spread, features[offset], 1, BrcbRepeatParams{1, 8});
        Mul(dot, dot, spread, uint64_t(16), uint8_t(8), BinaryRepeatParams{1, 1, 0, 2, 2, 1});
        if constexpr (LinearReduce) {
            Add(dot, dot[16], dot, uint64_t(16), uint8_t(7), BinaryRepeatParams{1, 1, 1, 0, 2, 0});
        } else {
            Add(dot, dot, dot[64], 64);
            Add(dot, dot, dot[32], 32);
            Add(dot, dot, dot[16], 16);
        }
    }

    template <bool ClearAhead = false>
    __aicore__ inline void StoreRows(const LocalTensor<float> &result, uint32_t start, uint32_t rows)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        if constexpr (ClearAhead) {
            PipeBarrier<PIPE_MTE3>();
        }
        spline_output::CopyRows(yg, result, start, rows);
    }

    LocalTensor<float> weights{TPosition::VECCALC, 0, 512};
    LocalTensor<float> features{TPosition::VECCALC, FeatureOffset, FeatureCount};
    LocalTensor<float> spread{TPosition::VECCALC, Spread, 64};
    LocalTensor<float> dot{TPosition::VECCALC, Dot, 128};
    GlobalTensor<float> xg;
    GlobalTensor<float> pg;
    GlobalTensor<float> wg;
    GlobalTensor<float> yg;
    GlobalTensor<int64_t> eg;
};

template <bool Closed0, bool Closed1, bool Tail>
__aicore__ inline void RunSingleStaged(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out)
{
    const uint32_t blocks = GetBlockNum();
    if (blocks == 0) {
        return;
    }
    SplineConvSingleStage<Closed0, Closed1, false> op;
    op.Init(x, edges, pseudo, weight, out);
    op.template ProcessChunk<Tail>(2 * blocks - (Tail ? 1 : 0), edges, pseudo);
}
} // namespace AscendC
