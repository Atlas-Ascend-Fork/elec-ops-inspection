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

template <typename T>
class SplineConvStreaming {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edge, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR root, GM_ADDR bias,
                                GM_ADDR out, GM_ADDR work, const SplineConvTilingData &td)
    {
        t = td;
        xg.SetGlobalBuffer((__gm__ T *)x);
        xb.SetGlobalBuffer((__gm__ uint16_t *)x);
        pg.SetGlobalBuffer((__gm__ T *)pseudo);
        pb.SetGlobalBuffer((__gm__ uint16_t *)pseudo);
        eg.SetGlobalBuffer((__gm__ int64_t *)edge);
        wg.SetGlobalBuffer((__gm__ T *)weight);
        wb.SetGlobalBuffer((__gm__ uint16_t *)weight);
        rg.SetGlobalBuffer((__gm__ T *)root);
        bg.SetGlobalBuffer((__gm__ T *)bias);
        yg.SetGlobalBuffer((__gm__ T *)out);
        next.SetGlobalBuffer((__gm__ int32_t *)work + GetBlockIdx() * ((t.e + 15) / 16 * 16));
        pipe.InitBuffer(headBuf, ((t.nodesPerBlock + 7) / 8 * 8) * 4);
        pipe.InitBuffer(floatBuf, 128 * 24);
        pipe.InitBuffer(rawBuf, 128 * sizeof(T));
        pipe.InitBuffer(outBuf, 128 * sizeof(T));
        h = headBuf.Get<int32_t>();
        w = floatBuf.Get<float>();
        y = w[128];
        comp = w[256];
        prod = w[384];
        z = w[512];
        sum = w[640];
        raw = rawBuf.Get<T>();
        yr = outBuf.Get<T>();
    }
    __aicore__ inline float XValue(uint64_t offset)
    {
        if constexpr (sizeof(T) == 2) {
            union {
                uint32_t u;
                float f;
            } bits;
            bits.u = uint32_t(xb.GetValue(offset)) << 16;
            return bits.f;
        } else {
            return xg.GetValue(offset);
        }
    }
    __aicore__ inline void Basis(uint32_t edge, float factors[8][4], uint32_t base[8])
    {
        for (uint32_t d = 0; d < t.dim; ++d) {
            uint32_t offset = t.pseudoTrans ? d * t.e + edge : edge * t.dim + d;
            float pseudo;
            if constexpr (sizeof(T) == 2) {
                pseudo = SplineBf16ToFloat(pb.GetValue(offset));
            } else {
                pseudo = pg.GetValue(offset);
            }
            float value = pseudo * int32_t(t.scale[d]);
            float fraction = SplitSplineCoordinate(value, t.ks[d], true, base[d]);
            SplineBasisFactors(fraction, t.degree, factors[d]);
        }
    }
    __aicore__ inline void LoadWeight(const GlobalTensor<T> &g, uint64_t offset, uint32_t count, uint32_t padded)
    {
        DataCopyExtParams p{1, count * uint32_t(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, uint8_t(padded - count), {}};
        DataCopyPad(raw, g[offset], p, pad);
        PipeBarrier<PIPE_ALL>();
        if constexpr (sizeof(T) == 2) {
            Cast(w, raw, RoundMode::CAST_NONE, padded);
        } else {
            DataCopy(w, raw, padded);
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Accumulate(float factor, uint32_t count)
    {
        SplineCompensatedAccumulate(y, w, comp, prod, z, sum, factor, count);
    }
    __aicore__ inline void LoadKernelWeight(uint32_t kernel, uint32_t channel, uint32_t col, uint32_t count,
                                            uint32_t padded)
    {
        if (!t.weightTrans) {
            LoadWeight(wg, (uint64_t(kernel) * t.ci + channel) * t.co + col, count, padded);
            return;
        }
        for (uint32_t j = 0; j < padded; ++j) {
            float value = 0;
            if (j < count) {
                uint64_t offset = (uint64_t(kernel) * t.co + col + j) * t.ci + channel;
                if constexpr (sizeof(T) == 2) {
                    union {
                        uint32_t u;
                        float f;
                    } bits;
                    bits.u = uint32_t(wb.GetValue(offset)) << 16;
                    value = bits.f;
                } else {
                    value = wg.GetValue(offset);
                }
            }
            w.SetValue(j, value);
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Process()
    {
        const uint32_t start = GetBlockIdx() * t.nodesPerBlock;
        const uint32_t limit = start + t.nodesPerBlock;
        const uint32_t end = limit > t.n ? t.n : limit;
        BuildHeads(start, end);
        for (uint32_t node = start; node < end; ++node) {
            const int32_t firstEdge = h.GetValue(node - start);
            const uint32_t degree = CountValidEdges(firstEdge);
            for (uint32_t col = 0; col < t.co; col += 128) {
                ProcessTile(node, col, firstEdge, degree);
            }
        }
    }

private:
    __aicore__ inline void BuildHeads(uint32_t start, uint32_t end)
    {
        for (uint32_t j = 0; j < t.nodesPerBlock; ++j) {
            h.SetValue(j, -1);
        }
        for (uint32_t edge = 0; edge < t.e; ++edge) {
            int64_t dst = eg.GetValue(t.edgeTrans ? 2 * edge + 1 : t.e + edge);
            if (dst >= start && dst < end) {
                uint32_t j = uint32_t(dst) - start;
                next.SetValue(edge, h.GetValue(j));
                h.SetValue(j, edge);
            }
        }
    }

    __aicore__ inline uint32_t CountValidEdges(int32_t firstEdge)
    {
        uint32_t degree = 0;
        for (int32_t edge = firstEdge; edge >= 0; edge = next.GetValue(edge)) {
            int64_t src = eg.GetValue(t.edgeTrans ? 2 * edge : edge);
            if (src >= 0 && src < t.n) {
                ++degree;
            }
        }
        return degree;
    }

    __aicore__ inline void ProcessTile(uint32_t node, uint32_t col, int32_t firstEdge, uint32_t degree)
    {
        const uint32_t remaining = t.co - col;
        const uint32_t count = remaining > 128 ? 128 : remaining;
        constexpr uint32_t align = sizeof(T) == 2 ? 16 : 8;
        const uint32_t padded = (count + align - 1) / align * align;
        Duplicate(y, 0.0f, padded);
        Duplicate(comp, 0.0f, padded);
        PipeBarrier<PIPE_ALL>();
        for (int32_t edge = firstEdge; edge >= 0; edge = next.GetValue(edge)) {
            int64_t src = eg.GetValue(t.edgeTrans ? 2 * edge : edge);
            if (src >= 0 && src < t.n) {
                AccumulateEdge(edge, uint64_t(src) * t.ci, col, count, padded);
            }
        }
        if (t.mean && degree > 0) {
            Muls(y, y, 1.0f / degree, padded);
            PipeBarrier<PIPE_V>();
        }
        AddRootAndBias(node, col, count, padded);
        StoreTile(node, col, count, padded);
    }

    __aicore__ inline void AccumulateEdge(uint32_t edge, uint64_t srcOffset, uint32_t col, uint32_t count,
                                          uint32_t padded)
    {
        float factors[8][4];
        uint32_t base[8];
        Basis(edge, factors, base);
        for (uint32_t slot = 0; slot < t.slots; ++slot) {
            float factor;
            uint32_t weight = SplineWeightIndex(slot, t.dim, t.degree, t.ks, factors, base, factor);
            if (factor == 0) {
                continue;
            }
            for (uint32_t channel = 0; channel < t.ci; ++channel) {
                float input = factor * XValue(srcOffset + channel);
                if (input != 0) {
                    LoadKernelWeight(weight, channel, col, count, padded);
                    Accumulate(input, padded);
                }
            }
        }
    }

    __aicore__ inline void AddRootAndBias(uint32_t node, uint32_t col, uint32_t count, uint32_t padded)
    {
        if (t.hasRoot) {
            Duplicate(comp, 0.0f, padded);
            PipeBarrier<PIPE_V>();
            for (uint32_t c = 0; c < t.ci; ++c) {
                float a = XValue(uint64_t(node) * t.ci + c);
                if (a != 0) {
                    LoadWeight(rg, uint64_t(c) * t.co + col, count, padded);
                    Accumulate(a, padded);
                }
            }
        }
        if (t.hasBias) {
            LoadWeight(bg, col, count, padded);
            Add(y, y, w, padded);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreTile(uint32_t node, uint32_t col, uint32_t count, uint32_t padded)
    {
        if constexpr (sizeof(T) == 2) {
            Cast(yr, y, RoundMode::CAST_RINT, padded);
        } else {
            DataCopy(yr, y, padded);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams p{1, count * uint32_t(sizeof(T)), 0, 0, 0};
        DataCopyPad(yg[node * t.co + col], yr, p);
        PipeBarrier<PIPE_ALL>();
    }

    SplineConvTilingData t;
    TPipe pipe;
    TBuf<TPosition::VECCALC> headBuf;
    TBuf<TPosition::VECCALC> floatBuf;
    TBuf<TPosition::VECCALC> rawBuf;
    TBuf<TPosition::VECCALC> outBuf;
    LocalTensor<int32_t> h;
    LocalTensor<float> w;
    LocalTensor<float> y;
    LocalTensor<float> comp;
    LocalTensor<float> prod;
    LocalTensor<float> z;
    LocalTensor<float> sum;
    LocalTensor<T> raw;
    LocalTensor<T> yr;
    GlobalTensor<T> xg;
    GlobalTensor<T> pg;
    GlobalTensor<T> wg;
    GlobalTensor<T> rg;
    GlobalTensor<T> bg;
    GlobalTensor<T> yg;
    GlobalTensor<uint16_t> xb;
    GlobalTensor<uint16_t> pb;
    GlobalTensor<uint16_t> wb;
    GlobalTensor<int64_t> eg;
    GlobalTensor<int32_t> next;
};

} // namespace AscendC
