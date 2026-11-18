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

namespace AscendC {

template <typename T, uint32_t C>
class SplineConvLocalWeights {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                const SplineConvTilingData &td, TPipe *pipe)
    {
        t = td;
        buffers.SetGlobal(x, edges, pseudo, weight, out);
        buffers.InitDynamic(t, pipe);
    }
    __aicore__ inline bool SingleDot(uint32_t e)
    {
        int64_t src = buffers.el.GetValue(e);
        if (src < 0 || src >= t.n) {
            return false;
        }
        float p0 = buffers.Pseudo(e * 2) * int32_t(t.scale[0]);
        float p1 = buffers.Pseudo(e * 2 + 1) * int32_t(t.scale[1]);
        int32_t b0 = int32_t(p0);
        int32_t b1 = int32_t(p1);
        p0 -= float(b0);
        p1 -= float(b1);
        float f[4] = {(1 - p0) * (1 - p1), p0 * (1 - p1), (1 - p0) * p1, p0 * p1};
        DataCopyPadExtParams<T> pad{false, 0, 0, {}};
        DataCopyExtParams xp{1, C * uint32_t(sizeof(T)), 0, 0, 0};
        DataCopyPad(buffers.xr, buffers.xg[uint32_t(src) * C], xp, pad);
        DataCopyExtParams wp{1, C * 16 * uint32_t(sizeof(T)), 0, 0, 0};
        for (uint32_t slot = 0; slot < 4; ++slot) {
            uint32_t wi = (uint32_t(b0) + (slot & 1)) % t.ks[0] + t.ks[0] * ((uint32_t(b1) + (slot >> 1)) % t.ks[1]);
            DataCopyPad(buffers.wr[slot * C * 16], buffers.wg[wi * C * 16], wp, pad);
        }
        PipeBarrier<PIPE_ALL>();
        if constexpr (sizeof(T) == 2) {
            Cast(buffers.xv, buffers.xr, RoundMode::CAST_NONE, C);
            Cast(buffers.weights, buffers.wr, RoundMode::CAST_NONE, 4 * C * 16);
            PipeBarrier<PIPE_V>();
        }
        Muls(buffers.dot, buffers.weights, f[0], C * 16);
        PipeBarrier<PIPE_V>();
        for (uint32_t slot = 1; slot < 4; ++slot) {
            Axpy(buffers.dot, buffers.weights[slot * C * 16], f[slot], C * 16);
            PipeBarrier<PIPE_V>();
        }
        Brcb(buffers.broadcast, buffers.xv, uint8_t(C / 8), BrcbRepeatParams{1, 8});
        PipeBarrier<PIPE_V>();
        Mul(buffers.dot, buffers.dot, buffers.broadcast, uint64_t(16), uint8_t(C),
            BinaryRepeatParams{1, 1, 0, 2, 2, 1});
        PipeBarrier<PIPE_V>();
        for (uint32_t rows = C; rows > 1; rows /= 2) {
            Add(buffers.dot, buffers.dot, buffers.dot[rows / 2 * 16], rows / 2 * 16);
            PipeBarrier<PIPE_V>();
        }
        return true;
    }
    __aicore__ inline void Process()
    {
        const uint32_t start = GetBlockIdx();
        const uint32_t step = t.blocks;
        const uint32_t rows = (t.n - start + step - 1) / step;
        buffers.LoadGraph(t.e);
        Duplicate(buffers.outLocal.template ReinterpretCast<int32_t>(), int32_t(0), rows * 16 * sizeof(T) / 4);
        PipeBarrier<PIPE_ALL>();
        const uint32_t active = buffers.BuildIncoming(t, start, step, rows);
        for (uint32_t j = 0; j < active; ++j) {
            ProcessRow(j);
        }
        DataCopyExtParams yp{uint16_t(rows), 16 * uint32_t(sizeof(T)), 0, (step - 1) * 16 * uint32_t(sizeof(T)), 0};
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(buffers.yg[start * 16], buffers.outLocal, yp);
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void ProcessRow(uint32_t j)
    {
        uint32_t count = 0;
        Duplicate(buffers.agg, 0.0f, 16);
        PipeBarrier<PIPE_V>();
        for (int32_t e = buffers.head.GetValue(j); e >= 0; e = buffers.links.GetValue(e)) {
            if (SingleDot(uint32_t(e))) {
                Add(buffers.agg, buffers.agg, buffers.dot, 16);
                PipeBarrier<PIPE_V>();
                ++count;
            }
        }
        if (count == 0) {
            return;
        }
        if (t.mean && count > 1) {
            Muls(buffers.agg, buffers.agg, 1.0f / count, 16);
            PipeBarrier<PIPE_V>();
        }
        buffers.SetOutputRow(buffers.occupied.GetValue(j), buffers.agg);
        PipeBarrier<PIPE_ALL>();
    }

    SplineConvTilingData t;
    SplineConvSparseBuffers<T, C> buffers;
};
template <typename T>
__aicore__ inline bool RunSparseLocalWeights(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                             const SplineConvTilingData &t)
{
    if (t.ci != 128 || t.co != 16 || t.kernels <= 4 || t.dim != 2 || t.degree != 1 || t.ks[0] < 2 || t.ks[0] > 4 ||
        t.ks[1] < 2 || t.ks[1] > 4 || t.e == 0 || t.e > 16 || t.e >= t.n || t.nodesPerBlock > 128 || t.hasRoot ||
        t.hasBias || t.streaming || t.edgeTrans || t.pseudoTrans || t.weightTrans) {
        return false;
    }
    TPipe pipe;
    SplineConvLocalWeights<T, 128> op;
    op.Init(x, edges, pseudo, weight, out, t, &pipe);
    op.Process();
    return true;
}

} // namespace AscendC
