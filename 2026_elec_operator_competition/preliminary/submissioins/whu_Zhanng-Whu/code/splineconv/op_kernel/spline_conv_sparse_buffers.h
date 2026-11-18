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
#include "kernel_operator.h"
#include "spline_conv_basis.h"
namespace spline_output {
using namespace AscendC;
__aicore__ inline uint32_t FirstBit(uint64_t mask)
{
    return uint32_t(63 - CountLeadingZero(mask & (uint64_t(0) - mask)));
}
template <typename T>
__aicore__ inline void CopyRows(const GlobalTensor<T> &global, const LocalTensor<T> &local, uint32_t node,
                                uint32_t rows)
{
    if (rows == 0) {
        return;
    }
    if ((uint64_t(global.GetPhyAddr()) & 31) == 0) {
        DataCopy(global[node * 16], local, rows * 16);
    } else {
        DataCopyExtParams p{1, rows * 16 * uint32_t(sizeof(T)), 0, 0, 0};
        DataCopyPad(global[node * 16], local, p);
    }
}
template <typename T, bool Ready = false>
__aicore__ inline void ClearInactiveRows(const GlobalTensor<T> &global, const LocalTensor<T> &zeros, uint32_t begin,
                                         uint32_t rows, uint32_t occupied)
{
    // Each caller partitions the output into spans of at most 32 rows.
    if (rows == 0) {
        return;
    }
    if constexpr (!Ready) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    }
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if (occupied == 0) {
        CopyRows(global, zeros, begin, rows);
        return;
    }
    uint64_t mask = (~uint64_t(occupied)) & ((uint64_t(1) << rows) - 1);
    while (mask) {
        uint32_t first = FirstBit(mask);
        uint64_t ended = ~(mask >> first);
        uint32_t count = FirstBit(ended);
        CopyRows(global, zeros, begin + first, count);
        mask &= ~(((uint64_t(1) << count) - 1) << first);
    }
}
} // namespace spline_output

namespace AscendC {
template <typename T, uint32_t C>
struct SplineConvSparseBuffers {
    __aicore__ inline void SetGlobal(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out)
    {
        xg.SetGlobalBuffer((__gm__ T *)x);
        eg.SetGlobalBuffer((__gm__ int64_t *)edges);
        pg.SetGlobalBuffer((__gm__ T *)pseudo);
        wg.SetGlobalBuffer((__gm__ T *)weight);
        yg.SetGlobalBuffer((__gm__ T *)out);
    }

    __aicore__ inline void InitDynamic(const SplineConvTilingData &t, TPipe *pipe)
    {
        uint32_t off = 0;
        uint32_t eo = Reserve(off, t.e * 16);
        uint32_t po = Reserve(off, t.e * 2 * sizeof(T));
        uint32_t ho = Reserve(off, t.nodesPerBlock * 4);
        uint32_t no = Reserve(off, t.e * 4);
        uint32_t ro = C == 128 ? Reserve(off, t.e * 4) : 0;
        uint32_t wo = Reserve(off, 4 * C * 16 * sizeof(T));
        uint32_t wf = sizeof(T) == 2 ? Reserve(off, 4 * C * 16 * 4) : wo;
        uint32_t xo = Reserve(off, C * sizeof(T));
        uint32_t xf = sizeof(T) == 2 ? Reserve(off, C * 4) : xo;
        uint32_t ao = Reserve(off, 4 * C * 4);
        uint32_t bo = Reserve(off, 4 * C * 8 * 4);
        uint32_t doff = Reserve(off, 4 * C * 16 * 4);
        uint32_t yo = Reserve(off, t.nodesPerBlock * 16 * sizeof(T));
        pipe->InitBuffer(buf, off);
        el = buf.Get<int64_t>()[eo / 8];
        pr = buf.Get<T>()[po / sizeof(T)];
        pb = pr.template ReinterpretCast<uint16_t>();
        head = buf.Get<int32_t>()[ho / 4];
        links = buf.Get<int32_t>()[no / 4];
        if constexpr (C == 128) {
            occupied = buf.Get<uint32_t>()[ro / 4];
        }
        wr = buf.Get<T>()[wo / sizeof(T)];
        weights = buf.Get<float>()[wf / 4];
        xr = buf.Get<T>()[xo / sizeof(T)];
        xv = buf.Get<float>()[xf / 4];
        agg = buf.Get<float>()[ao / 4];
        broadcast = buf.Get<float>()[bo / 4];
        dot = buf.Get<float>()[doff / 4];
        outLocal = buf.Get<T>()[yo / sizeof(T)];
    }
    __aicore__ inline uint32_t Reserve(uint32_t &off, uint32_t bytes)
    {
        uint32_t result = off;
        off += (bytes + 31) / 32 * 32;
        return result;
    }

    __aicore__ inline float Pseudo(uint32_t index) const
    {
        if constexpr (sizeof(T) == 2) {
            return SplineBf16ToFloat(pb.GetValue(index));
        } else {
            return pr.GetValue(index);
        }
    }

    __aicore__ inline void LoadGraph(uint32_t edges)
    {
        DataCopyPad(el, eg, DataCopyExtParams{1, edges * 16, 0, 0, 0}, DataCopyPadExtParams<int64_t>{false, 0, 0, 0});
        DataCopyPad(pr, pg, DataCopyExtParams{1, edges * 2 * uint32_t(sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<T>{false, 0, 0, {}});
    }

    __aicore__ inline uint32_t BuildIncoming(const SplineConvTilingData &t, uint32_t start, uint32_t step,
                                             uint32_t rows)
    {
        if (step == 0) {
            return 0;
        }
        if constexpr (C != 128) {
            for (uint32_t i = 0; i < rows; ++i) {
                head.SetValue(i, -1);
            }
        }
        uint32_t active = 0;
        for (uint32_t edge = 0; edge < t.e; ++edge) {
            int64_t dst = el.GetValue(t.e + edge);
            if (dst < start || dst >= t.n || (uint32_t(dst) - start) % step != 0) {
                continue;
            }
            uint32_t row = (uint32_t(dst) - start) / step;
            if constexpr (C == 128) {
                row = FindOrAddRow(row, active);
            } else {
                active = rows;
            }
            links.SetValue(edge, head.GetValue(row));
            head.SetValue(row, edge);
        }
        return active;
    }

    __aicore__ inline uint32_t FindOrAddRow(uint32_t row, uint32_t &count)
    {
        uint32_t index = 0;
        while (index < count && occupied.GetValue(index) != row) {
            ++index;
        }
        if (index == count) {
            occupied.SetValue(index, row);
            head.SetValue(index, -1);
            ++count;
        }
        return index;
    }

    __aicore__ inline void SetOutputRow(uint32_t row, const LocalTensor<float> &result)
    {
        if constexpr (sizeof(T) == 2) {
            Cast(outLocal[row * 16], result, RoundMode::CAST_RINT, 16);
        } else {
            DataCopy(outLocal[row * 16], result, 16);
        }
    }

    TBuf<TPosition::VECCALC> buf;
    LocalTensor<int64_t> el;
    LocalTensor<int32_t> head;
    LocalTensor<int32_t> links;
    LocalTensor<uint32_t> occupied;
    LocalTensor<T> pr;
    LocalTensor<T> wr;
    LocalTensor<T> xr;
    LocalTensor<T> outLocal;
    LocalTensor<uint16_t> pb;
    LocalTensor<float> weights;
    LocalTensor<float> xv;
    LocalTensor<float> agg;
    LocalTensor<float> broadcast;
    LocalTensor<float> dot;
    GlobalTensor<T> xg;
    GlobalTensor<T> pg;
    GlobalTensor<T> wg;
    GlobalTensor<T> yg;
    GlobalTensor<int64_t> eg;
};
} // namespace AscendC
