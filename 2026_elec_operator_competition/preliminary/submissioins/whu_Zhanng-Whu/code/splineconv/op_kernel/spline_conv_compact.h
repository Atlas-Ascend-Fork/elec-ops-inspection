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
class SplineConvCompact {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                const SplineConvTilingData &td, TPipe *pipe)
    {
        t = td;
        buffers.SetGlobal(x, edges, pseudo, weight, out);
        if constexpr (C == 8) {
            buffers.InitDynamic(t, pipe);
        } else {
            InitStatic();
        }
    }

    template <uint32_t Step = 0>
    __aicore__ inline void Process()
    {
        const uint32_t start = GetBlockIdx();
        const uint32_t step = Step ? Step : t.blocks;
        if (step == 0) {
            return;
        }
        const uint32_t rows = (t.n - start + step - 1) / step;
        const uint32_t zeroBound = (t.n + step - 1) / step;
        splitZero = C == 128 && zeroBound <= 32;
        Prepare(rows, zeroBound);
        if constexpr (C != 8) {
            const uint32_t zeroBegin = start * zeroBound;
            const uint32_t remaining = zeroBegin < t.n ? t.n - zeroBegin : 0;
            const uint32_t zeroRows = remaining < zeroBound ? remaining : zeroBound;
            const OwnedEdges own = PrefetchOwned(start, step, zeroBegin, zeroRows);
            if constexpr (C == 128) {
                ClearUnownedRows(zeroBegin, zeroRows, own);
            }
            if (own.count <= 1) {
                ProcessSingleOwner(own, start, step);
                StoreOutput(start, step, rows);
                return;
            }
        }
        const uint32_t active = buffers.BuildIncoming(t, start, step, rows);
        ProcessRows(active);
        StoreOutput(start, step, rows);
    }

private:
    struct OwnedEdges {
        uint32_t count = 0;
        uint32_t edge = 0;
        uint32_t occupied = 0;
    };

    __aicore__ inline void InitStatic()
    {
        constexpr uint32_t WO = 2560;
        constexpr uint32_t WF = sizeof(T) == 2 ? WO + 4 * C * 16 * sizeof(T) : WO;
        constexpr uint32_t XO = WF + 4 * C * 16 * 4;
        constexpr uint32_t XF = sizeof(T) == 2 ? XO + C * sizeof(T) : XO;
        constexpr uint32_t AO = XF + C * 4;
        constexpr uint32_t BO = AO + 4 * C * 4;
        constexpr uint32_t DO = BO + 4 * C * 8 * 4;
        constexpr uint32_t YO = DO + 4 * C * 16 * 4;
        buffers.el = LocalTensor<int64_t>(TPosition::VECCALC, 0, 128);
        buffers.pr = LocalTensor<T>(TPosition::VECCALC, 1024, 128);
        buffers.pb = buffers.pr.template ReinterpretCast<uint16_t>();
        buffers.head = LocalTensor<int32_t>(TPosition::VECCALC, 1536, 128);
        buffers.links = LocalTensor<int32_t>(TPosition::VECCALC, 2048, 64);
        buffers.occupied = LocalTensor<uint32_t>(TPosition::VECCALC, 2304, 64);
        buffers.wr = LocalTensor<T>(TPosition::VECCALC, WO, 4 * C * 16);
        buffers.weights = LocalTensor<float>(TPosition::VECCALC, WF, 4 * C * 16);
        buffers.xr = LocalTensor<T>(TPosition::VECCALC, XO, C);
        buffers.xv = LocalTensor<float>(TPosition::VECCALC, XF, C);
        buffers.agg = LocalTensor<float>(TPosition::VECCALC, AO, 4 * C);
        buffers.broadcast = LocalTensor<float>(TPosition::VECCALC, BO, 4 * C * 8);
        buffers.dot = LocalTensor<float>(TPosition::VECCALC, DO, 4 * C * 16);
        buffers.outLocal = LocalTensor<T>(TPosition::VECCALC, YO, 128 * 16);
        prefetched = LocalTensor<T>(TPosition::VECCALC, YO + 128 * 16 * sizeof(T), 64 * C);
    }

    __aicore__ inline void Prepare(uint32_t rows, uint32_t zeroBound)
    {
        buffers.LoadGraph(t.e);
        if constexpr (C != 8) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            if ((uint64_t(buffers.wg.GetPhyAddr()) & 31) == 0) {
                DataCopy(buffers.wr, buffers.wg, 4 * C * 16);
            } else {
                DataCopyPad(buffers.wr, buffers.wg, DataCopyExtParams{1, 4 * C * 16 * uint32_t(sizeof(T)), 0, 0, 0},
                            DataCopyPadExtParams<T>{false, 0, 0, {}});
            }
        }
        Duplicate(buffers.outLocal.template ReinterpretCast<int32_t>(), int32_t(0),
                  (C == 128 ? zeroBound : rows) * 16 * sizeof(T) / 4);
        if constexpr (C == 8) {
            PipeBarrier<PIPE_ALL>();
        } else {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
    }

    __aicore__ inline OwnedEdges PrefetchOwned(uint32_t start, uint32_t step, uint32_t begin, uint32_t rows)
    {
        OwnedEdges own{};
        if (step == 0) {
            return own;
        }
        nodeMask = 0;
        for (uint32_t edge = 0; edge < t.e; ++edge) {
            int64_t dst = buffers.el.GetValue(t.e + edge);
            if constexpr (C == 128) {
                if (splitZero && dst >= begin && dst < int64_t(begin + rows)) {
                    own.occupied |= 1u << (uint32_t(dst) - begin);
                }
            }
            if (dst < start || dst >= t.n || (uint32_t(dst) - start) % step != 0) {
                continue;
            }
            if constexpr (C == 128) {
                if (splitZero) {
                    nodeMask |= 1u << ((uint32_t(dst) - start) / step);
                }
            }
            int64_t src = buffers.el.GetValue(edge);
            if (src >= 0 && src < t.n) {
                DataCopyPad(prefetched[edge * C], buffers.xg[uint32_t(src) * C],
                            DataCopyExtParams{1, C * uint32_t(sizeof(T)), 0, 0, 0},
                            DataCopyPadExtParams<T>{false, 0, 0, {}});
            }
            ++own.count;
            own.edge = edge;
        }
        return own;
    }

    __aicore__ inline void ClearUnownedRows(uint32_t begin, uint32_t rows, const OwnedEdges &own)
    {
        if (!splitZero || rows == 0) {
            return;
        }
        spline_output::ClearInactiveRows(buffers.yg, buffers.outLocal, begin, rows, own.occupied);
        if (own.count > 1 || (sizeof(T) == 2 && own.count == 1)) {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void ProcessSingleOwner(const OwnedEdges &own, uint32_t start, uint32_t step)
    {
        if constexpr (C == 128) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        } else if (own.count == 1) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        }
        if (own.count == 0 || step == 0) {
            return;
        }
        ConvertWeights();
        if (!SingleDot(own.edge)) {
            return;
        }
        uint32_t row = (uint32_t(buffers.el.GetValue(t.e + own.edge)) - start) / step;
        if constexpr (C == 128 && sizeof(T) == 4) {
            if (splitZero) {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
                spline_output::CopyRows(buffers.yg, buffers.dot, start + row * step, 1);
                directMask |= 1u << row;
                return;
            }
        }
        buffers.SetOutputRow(row, buffers.dot);
    }

    __aicore__ inline void ConvertWeights()
    {
        if constexpr (sizeof(T) == 2) {
            Cast(buffers.weights, buffers.wr, RoundMode::CAST_NONE, 4 * C * 16);
        }
    }

    __aicore__ inline void ProcessRows(uint32_t active)
    {
        if (active == 0) {
            return;
        }
        if constexpr (C == 8) {
            DataCopyPad(buffers.wr, buffers.wg, DataCopyExtParams{1, 4 * C * 16 * uint32_t(sizeof(T)), 0, 0, 0},
                        DataCopyPadExtParams<T>{false, 0, 0, {}});
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        ConvertWeights();
        for (uint32_t j = 0; j < active; ++j) {
            if (buffers.head.GetValue(j) >= 0) {
                ProcessRow(buffers.head.GetValue(j), C == 128 ? buffers.occupied.GetValue(j) : j);
            }
        }
    }

    __aicore__ inline void ProcessRow(int32_t firstEdge, uint32_t outputRow)
    {
        if constexpr (C != 8) {
            if (buffers.links.GetValue(firstEdge) < 0) {
                if (SingleDot(firstEdge)) {
                    buffers.SetOutputRow(outputRow, buffers.dot);
                    PipeBarrier<PIPE_ALL>();
                }
                return;
            }
        }
        Duplicate(buffers.agg, 0.0f, 4 * C);
        PipeBarrier<PIPE_ALL>();
        uint32_t count = 0;
        for (int32_t edge = firstEdge; edge >= 0; edge = buffers.links.GetValue(edge)) {
            if (AccumulateEdge(edge)) {
                ++count;
            }
        }
        if (count == 0) {
            return;
        }
        Dot();
        if (t.mean && count != 1) {
            Muls(buffers.dot, buffers.dot, 1.0f / count, 16);
        }
        buffers.SetOutputRow(outputRow, buffers.dot);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline bool AccumulateEdge(uint32_t edge)
    {
        int64_t src = buffers.el.GetValue(edge);
        if (src < 0 || src >= t.n) {
            return false;
        }
        if constexpr (C != 8) {
            buffers.xr = prefetched[edge * C];
            if constexpr (sizeof(T) == 4) {
                buffers.xv = buffers.xr.template ReinterpretCast<float>();
            }
        } else {
            DataCopyPad(buffers.xr, buffers.xg[uint32_t(src) * C],
                        DataCopyExtParams{1, C * uint32_t(sizeof(T)), 0, 0, 0},
                        DataCopyPadExtParams<T>{false, 0, 0, {}});
        }
        PipeBarrier<PIPE_ALL>();
        if constexpr (sizeof(T) == 2) {
            Cast(buffers.xv, buffers.xr, RoundMode::CAST_NONE, C);
        }
        uint32_t base[2];
        float f[4];
        Factors(edge, base, f);
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(C);
        for (uint32_t slot = 0; slot < 4; ++slot) {
            uint32_t wi = ((base[0] + (slot & 1)) & 1) + 2 * ((base[1] + (slot >> 1)) & 1);
            Axpy<float, float, false>(buffers.agg[wi * C], buffers.xv, f[slot], MASK_PLACEHOLDER, 1,
                                      UnaryRepeatParams{1, 1, 8, 8});
        }
        SetMaskNorm();
        ResetMask();
        return true;
    }

    __aicore__ inline void Factors(uint32_t edge, uint32_t base[2], float f[4])
    {
        float p0 = buffers.Pseudo(edge * 2) * int32_t(t.scale[0]);
        float p1 = buffers.Pseudo(edge * 2 + 1) * int32_t(t.scale[1]);
        int32_t b0 = int32_t(p0);
        int32_t b1 = int32_t(p1);
        base[0] = uint32_t(b0);
        base[1] = uint32_t(b1);
        p0 -= float(b0);
        p1 -= float(b1);
        f[0] = (1 - p0) * (1 - p1);
        f[1] = p0 * (1 - p1);
        f[2] = (1 - p0) * p1;
        f[3] = p0 * p1;
    }

    __aicore__ inline bool SingleDot(uint32_t e)
    {
        int64_t src = buffers.el.GetValue(e);
        if (src < 0 || src >= t.n) {
            return false;
        }
        DataCopyExtParams xp{1, C * uint32_t(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, {}};
        if constexpr (C == 128 || C == 32) {
            buffers.xr = prefetched[e * C];
            if constexpr (sizeof(T) == 4) {
                buffers.xv = buffers.xr.template ReinterpretCast<float>();
            }
        } else {
            DataCopyPad(buffers.xr, buffers.xg[uint32_t(src) * C], xp, pad);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        if constexpr (sizeof(T) == 2) {
            Cast(buffers.xv, buffers.xr, RoundMode::CAST_NONE, C);
        }
        float p0 = buffers.Pseudo(e * 2) * int32_t(t.scale[0]);
        float p1 = buffers.Pseudo(e * 2 + 1) * int32_t(t.scale[1]);
        int32_t b0 = int32_t(p0);
        int32_t b1 = int32_t(p1);
        p0 -= float(b0);
        p1 -= float(b1);
        float f[4] = {(1 - p0) * (1 - p1), p0 * (1 - p1), (1 - p0) * p1, p0 * p1};
        uint32_t wi = (b0 & 1) + 2 * (b1 & 1);
        Muls(buffers.dot, buffers.weights[wi * C * 16], f[0], C * 16);
        for (uint32_t slot = 1; slot < 4; ++slot) {
            wi = ((b0 + (slot & 1)) & 1) + 2 * ((b1 + (slot >> 1)) & 1);
            Axpy(buffers.dot, buffers.weights[wi * C * 16], f[slot], C * 16);
        }
        Brcb(buffers.broadcast, buffers.xv, uint8_t(C / 8), BrcbRepeatParams{1, 8});
        Mul(buffers.dot, buffers.dot, buffers.broadcast, uint64_t(16), uint8_t(C),
            BinaryRepeatParams{1, 1, 0, 2, 2, 1});
        for (uint32_t rows = C; rows > 1; rows /= 2) {
            Add(buffers.dot, buffers.dot, buffers.dot[rows / 2 * 16], rows / 2 * 16);
        }
        return true;
    }
    __aicore__ inline void Dot()
    {
        Brcb(buffers.broadcast, buffers.agg, uint8_t(4 * C / 8), BrcbRepeatParams{1, 8});
        BinaryRepeatParams params{1, 1, 0, 2, 2, 1};
        for (uint32_t row = 0; row < 4 * C; row += 255) {
            uint32_t count = 4 * C - row;
            if (count > 255) {
                count = 255;
            }
            Mul(buffers.dot[row * 16], buffers.weights[row * 16], buffers.broadcast[row * 8], uint64_t(16),
                uint8_t(count), params);
        }
        for (uint32_t rows = 4 * C; rows > 1; rows /= 2) {
            Add(buffers.dot, buffers.dot, buffers.dot[rows / 2 * 16], rows / 2 * 16);
        }
    }
    __aicore__ inline void StoreOutput(uint32_t start, uint32_t step, uint32_t rows)
    {
        if constexpr (C == 128) {
            if (splitZero) {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
                uint32_t mask = nodeMask & ~directMask;
                while (mask) {
                    uint32_t row = spline_output::FirstBit(mask);
                    spline_output::CopyRows(buffers.yg, buffers.outLocal[row * 16], start + row * step, 1);
                    mask &= mask - 1;
                }
                return;
            }
        }
        DataCopyExtParams yp{uint16_t(rows), 16 * uint32_t(sizeof(T)), 0, (step - 1) * 16 * uint32_t(sizeof(T)), 0};
        SetFlag<HardEvent::V_MTE3>(EVENT_ID3);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        if constexpr (C == 128) {
            if ((uint64_t(buffers.yg.GetPhyAddr()) & 31) == 0) {
                DataCopyParams basic{uint16_t(rows), uint16_t(16 * sizeof(T) / 32), 0,
                                     uint16_t((step - 1) * 16 * sizeof(T) / 32)};
                DataCopy(buffers.yg[start * 16], buffers.outLocal, basic);
            } else {
                DataCopyPad(buffers.yg[start * 16], buffers.outLocal, yp);
            }
        } else {
            DataCopyPad(buffers.yg[start * 16], buffers.outLocal, yp);
        }
    }

    SplineConvTilingData t;
    SplineConvSparseBuffers<T, C> buffers;
    LocalTensor<T> prefetched;
    bool splitZero = false;
    uint32_t nodeMask = 0;
    uint32_t directMask = 0;
};

__aicore__ inline bool CanUseCompactSparse(const SplineConvTilingData &t)
{
    return !t.streaming && !t.hasRoot && !t.hasBias && t.cacheGraph && t.cacheWeights && t.fusedDot && t.e > 0 &&
           t.e < t.n && t.e <= 64 && uint64_t(t.nodesPerBlock) * t.e <= 512 && t.nodesPerBlock <= t.ci &&
           t.ci == t.ciPad;
}

template <typename T>
__aicore__ inline bool RunCompactSparse(GM_ADDR x, GM_ADDR edges, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR out,
                                        const SplineConvTilingData &t)
{
    if (!CanUseCompactSparse(t) || t.co != 16 || t.dim != 2 || t.degree != 1 || t.ks[0] != 2 || t.ks[1] != 2) {
        return false;
    }
    if (t.ci == 8) {
        TPipe pipe;
        SplineConvCompact<T, 8> op;
        op.Init(x, edges, pseudo, weight, out, t, &pipe);
        op.Process();
    } else if (t.ci == 32) {
        SplineConvCompact<T, 32> op;
        op.Init(x, edges, pseudo, weight, out, t, nullptr);
        op.Process();
    } else if (t.ci == 128) {
        SplineConvCompact<T, 128> op;
        op.Init(x, edges, pseudo, weight, out, t, nullptr);
        if (t.blocks == 8) {
            op.template Process<8>();
        } else {
            op.Process();
        }
    } else {
        return false;
    }
    return true;
}

} // namespace AscendC
