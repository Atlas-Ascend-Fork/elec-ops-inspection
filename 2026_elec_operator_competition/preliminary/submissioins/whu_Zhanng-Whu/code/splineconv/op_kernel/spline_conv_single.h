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

namespace AscendC {

template <typename T, bool One, bool Closed0, bool Closed1, bool Chunk, bool Tail>
class SplineConvSingle {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out)
    {
        xg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        eg.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(edges));
        pg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(pseudo));
        wg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight));
        yg.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));
    }

    __aicore__ inline void Process(GM_ADDR tiling)
    {
        if constexpr (Chunk) {
            uint32_t active = (GetBlockNum() + 15) / 16;
            if constexpr (Tail) {
                const uint32_t aligned = (GetBlockNum() - 1) / Alignment * Alignment;
                active = (aligned + 15) / 16 + 1;
            }
            if (GetBlockIdx() >= active) {
                return;
            }
        }
        Prefetch();
        if (!SetRows(tiling)) {
            return;
        }
        Duplicate(yr.template ReinterpretCast<int32_t>(), int32_t(0), rows * 16 * sizeof(T) / 4);
        if constexpr (Tail) {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        } else {
            PipeBarrier<PIPE_ALL>();
        }
        int64_t src = el.GetValue(0);
        int64_t target = el.GetValue(1);
        const bool owns = oneRow ? target == int64_t(start)
                                 : (target >= start && target < end && (uint32_t(target) - start) % step == 0);
        if (owns && src >= 0 && src < nodes) {
            Compute(uint32_t(src), uint32_t(target));
        }
        Store();
    }

private:
    static constexpr uint32_t Alignment = 128 / (16 * sizeof(T));
    static constexpr uint32_t WeightRaw = 64;
    static constexpr uint32_t WeightFloat = sizeof(T) == 2 ? WeightRaw + 512 * sizeof(T) : WeightRaw;
    static constexpr uint32_t FeatureRaw = WeightFloat + 512 * 4;
    static constexpr uint32_t FeatureFloat = sizeof(T) == 2 ? FeatureRaw + 32 : FeatureRaw;
    static constexpr uint32_t Broadcast = FeatureFloat + 32 + 128;
    static constexpr uint32_t Dot = Broadcast + 1024;
    static constexpr uint32_t Output = Dot + 2048;

    __aicore__ inline bool SetRows(GM_ADDR tiling)
    {
        if constexpr (One || Chunk) {
            nodes = GetBlockNum() - (Tail ? 1 : 0);
            blocks = nodes;
        } else {
            const auto *raw = reinterpret_cast<const volatile __gm__ SplineConvTilingData *>(tiling);
            nodes = raw->n;
            blocks = raw->blocks;
            scale0 = raw->scale[0];
            scale1 = raw->scale[1];
        }
        start = Chunk ? GetBlockIdx() * 16 : GetBlockIdx();
        step = Chunk ? 1 : blocks;
        if (step == 0) {
            return false;
        }
        oneRow = One || (!Chunk && nodes == step);
        if constexpr (Chunk) {
            uint32_t limit = nodes;
            if constexpr (Tail) {
                const uint32_t aligned = nodes / Alignment * Alignment;
                if (GetBlockIdx() == (aligned + 15) / 16) {
                    start = aligned;
                } else {
                    limit = aligned;
                }
            }
            rows = limit - start;
            rows = rows > 16 ? 16 : rows;
            end = start + rows;
        } else {
            if (start >= nodes) {
                return false;
            }
            rows = oneRow ? 1 : (nodes - start + step - 1) / step;
            end = nodes;
        }
        return rows > 0 && rows <= (Chunk ? 16 : 8);
    }

    __aicore__ inline void Prefetch()
    {
        DataCopyPad(el, eg, DataCopyExtParams{1, 16, 0, 0, 0}, DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        if constexpr (Tail) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            LoadWeight();
            LoadPseudo();
        }
    }

    __aicore__ inline void LoadWeight()
    {
        DataCopyPad(wr, wg, DataCopyExtParams{1, 512 * uint32_t(sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<T>{false, 0, 0, {}});
    }

    __aicore__ inline void LoadPseudo()
    {
        DataCopyPad(pr, pg, DataCopyExtParams{1, 2 * uint32_t(sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<T>{false, 0, 0, {}});
    }

    __aicore__ inline void Compute(uint32_t src, uint32_t target)
    {
        if constexpr (!Tail) {
            LoadWeight();
        }
        DataCopyPad(xr, xg[src * 8], DataCopyExtParams{1, 8 * uint32_t(sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<T>{false, 0, 0, {}});
        if constexpr (!Tail) {
            LoadPseudo();
        }
        PipeBarrier<PIPE_ALL>();
        float p0;
        float p1;
        if constexpr (sizeof(T) == 2) {
            Cast(weights, wr, RoundMode::CAST_NONE, 512);
            Cast(xv, xr, RoundMode::CAST_NONE, 8);
            const auto bits = pr.template ReinterpretCast<uint16_t>();
            p0 = SplineBf16ToFloat(bits.GetValue(0));
            p1 = SplineBf16ToFloat(bits.GetValue(1));
        } else {
            p0 = pr.GetValue(0);
            p1 = pr.GetValue(1);
        }
        if constexpr (One || Chunk) {
            SplineSingleInterpolate<Closed0, Closed1>(dot, weights, p0, p1);
        } else {
            Interpolate(p0, p1);
        }
        Brcb(broadcast, xv, 1, BrcbRepeatParams{1, 8});
        PipeBarrier<PIPE_V>();
        Mul(dot, dot, broadcast, uint64_t(16), uint8_t(8), BinaryRepeatParams{1, 1, 0, 2, 2, 1});
        PipeBarrier<PIPE_V>();
        for (uint32_t channels = 8; channels > 1; channels /= 2) {
            Add(dot, dot, dot[channels / 2 * 16], channels / 2 * 16);
            PipeBarrier<PIPE_V>();
        }
        const uint32_t row = oneRow ? 0 : (target - start) / step;
        if constexpr (sizeof(T) == 2) {
            Cast(yr[row * 16], dot, RoundMode::CAST_RINT, 16);
        } else {
            DataCopy(yr[row * 16], dot, 16);
        }
    }

    __aicore__ inline void Interpolate(float p0, float p1)
    {
        p0 *= int32_t(scale0);
        p1 *= int32_t(scale1);
        const int32_t b0 = int32_t(p0);
        const int32_t b1 = int32_t(p1);
        p0 -= float(b0);
        p1 -= float(b1);
        float factors[4] = {(1 - p0) * (1 - p1), p0 * (1 - p1), (1 - p0) * p1, p0 * p1};
        PipeBarrier<PIPE_V>();
        uint32_t wi = (b0 & 1) + 2 * (b1 & 1);
        Muls(dot, weights[wi * 128], factors[0], 128);
        PipeBarrier<PIPE_V>();
        for (uint32_t slot = 1; slot < 4; ++slot) {
            wi = ((b0 + (slot & 1)) & 1) + 2 * ((b1 + (slot >> 1)) & 1);
            Axpy(dot, weights[wi * 128], factors[slot], 128);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Store()
    {
        PipeBarrier<PIPE_ALL>();
        if ((oneRow || Chunk) && (uint64_t(yg.GetPhyAddr()) & 31) == 0) {
            DataCopy(yg[start * 16], yr, rows * 16);
        } else {
            DataCopyExtParams params{uint16_t(rows), 16 * uint32_t(sizeof(T)), 0, (step - 1) * 16 * uint32_t(sizeof(T)),
                                     0};
            DataCopyPad(yg[start * 16], yr, params);
        }
        PipeBarrier<PIPE_ALL>();
    }

    LocalTensor<int64_t> el{TPosition::VECCALC, 0, 4};
    LocalTensor<T> pr{TPosition::VECCALC, 32, 32 / sizeof(T)};
    LocalTensor<T> wr{TPosition::VECCALC, WeightRaw, 512};
    LocalTensor<float> weights{TPosition::VECCALC, WeightFloat, 512};
    LocalTensor<T> xr{TPosition::VECCALC, FeatureRaw, 32 / sizeof(T)};
    LocalTensor<float> xv{TPosition::VECCALC, FeatureFloat, 8};
    LocalTensor<float> broadcast{TPosition::VECCALC, Broadcast, 256};
    LocalTensor<float> dot{TPosition::VECCALC, Dot, 512};
    LocalTensor<T> yr{TPosition::VECCALC, Output, Chunk ? 256 : 128};
    GlobalTensor<int64_t> eg;
    GlobalTensor<T> xg;
    GlobalTensor<T> pg;
    GlobalTensor<T> wg;
    GlobalTensor<T> yg;
    uint32_t nodes;
    uint32_t blocks;
    uint32_t start;
    uint32_t end;
    uint32_t step;
    uint32_t rows;
    uint32_t scale0;
    uint32_t scale1;
    bool oneRow;
};

template <typename T, bool One = false, bool Closed0 = false, bool Closed1 = false, bool Chunk = false,
          bool Tail = false>
__aicore__ inline void RunSingleStatic(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                       GM_ADDR tiling)
{
    SplineConvSingle<T, One, Closed0, Closed1, Chunk, Tail> op;
    op.Init(x, edges, pseudo, weight, out);
    op.Process(tiling);
}

} // namespace AscendC
