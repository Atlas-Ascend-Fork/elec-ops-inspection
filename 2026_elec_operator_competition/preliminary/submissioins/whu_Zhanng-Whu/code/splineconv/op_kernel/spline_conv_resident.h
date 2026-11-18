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

template <typename T, bool LargeKernelCount = false, uint32_t FixedCi = 0>
class SplineConvResident {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR edge, GM_ADDR pseudo, GM_ADDR weight, GM_ADDR root, GM_ADDR bias,
                                GM_ADDR out, GM_ADDR work, const SplineConvTilingData &td)
    {
        t = td;
        if constexpr (FixedCi != 0) {
            t.ci = FixedCi;
            t.ciPad = FixedCi;
            t.co = 16;
            t.coPad = 16;
            t.dim = 2;
            t.degree = 1;
            t.slots = 4;
            t.kernels = 4;
            t.hasRoot = 0;
            t.hasBias = 0;
            t.cacheWeights = 1;
            t.cacheGraph = 1;
            t.fusedDot = 1;
        }
        xg.SetGlobalBuffer((__gm__ T *)x);
        eg.SetGlobalBuffer((__gm__ int64_t *)edge);
        pg.SetGlobalBuffer((__gm__ T *)pseudo);
        pbits.SetGlobalBuffer((__gm__ uint16_t *)pseudo);
        wg.SetGlobalBuffer((__gm__ T *)weight);
        rg.SetGlobalBuffer((__gm__ T *)root);
        bg.SetGlobalBuffer((__gm__ T *)bias);
        yg.SetGlobalBuffer((__gm__ T *)out);
        next.SetGlobalBuffer((__gm__ int32_t *)work + GetBlockIdx() * ((t.e + 15) / 16 * 16));
        InitCacheBuffers();
        InitComputeBuffers();
    }
    __aicore__ inline int64_t EdgeValue(uint32_t i)
    {
        return t.cacheGraph ? edgeLocal.GetValue(i) : eg.GetValue(i);
    }
    __aicore__ inline int32_t NextValue(uint32_t i)
    {
        return t.cacheGraph ? nextLocal.GetValue(i) : next.GetValue(i);
    }
    __aicore__ inline void NextAssign(uint32_t i, int32_t v)
    {
        if (t.cacheGraph) {
            nextLocal.SetValue(i, v);
        } else {
            next.SetValue(i, v);
        }
    }
    __aicore__ inline void LoadRow(const GlobalTensor<T> &g, uint32_t offset)
    {
        DataCopyExtParams p{1, t.ci * uint32_t(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, uint8_t(t.ciPad - t.ci), {}};
        DataCopyPad(xr, g[offset], p, pad);
        PipeBarrier<PIPE_ALL>();
        if constexpr (sizeof(T) == 2) {
            Cast(xv, xr, RoundMode::CAST_NONE, t.ciPad);
        } else {
            DataCopy(xv, xr, t.ciPad);
        }
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void LoadMatrix(const GlobalTensor<T> &g, uint32_t offset)
    {
        uint32_t rows = t.co == t.coPad ? 1 : t.ci;
        uint32_t width = t.co == t.coPad ? t.ci * t.co : t.co;
        DataCopyExtParams p{uint16_t(rows), width * uint32_t(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, uint8_t(t.coPad - t.co), {}};
        DataCopyPad(raw, g[offset], p, pad);
        PipeBarrier<PIPE_ALL>();
        if constexpr (sizeof(T) == 2) {
            Cast(mat, raw, RoundMode::CAST_NONE, t.ci * t.coPad);
        }

        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline float PseudoValue(uint32_t offset)
    {
        if constexpr (sizeof(T) == 2) {
            return SplineBf16ToFloat(t.cacheGraph ? pseudoBits.GetValue(offset) : pbits.GetValue(offset));
        } else {
            return t.cacheGraph ? pseudoLocal.GetValue(offset) : pg.GetValue(offset);
        }
    }

    template <uint32_t D, uint32_t Degree>
    __aicore__ inline void AccumulateEdge(uint32_t edge, bool stable, bool active[64])
    {
        const uint32_t dim = D == 0 ? t.dim : D;
        const uint32_t degree = D == 0 ? t.degree : Degree;
        constexpr uint32_t oneDim = Degree + 1;
        const uint32_t slots = D == 0 ? t.slots : (D == 2 ? oneDim * oneDim : oneDim * oneDim * oneDim);
        float factors[8][4];
        uint32_t base[8];
        for (uint32_t d = 0; d < dim; ++d) {
            int32_t scale = int32_t(t.scale[d]);
            float value = PseudoValue(edge * dim + d) * scale;
            float fraction = SplitSplineCoordinate(value, t.ks[d], scale < 0, base[d]);
            SplineBasisFactors(fraction, degree, factors[d]);
        }
        if (!stable) {
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(t.ciPad);
        }
        for (uint32_t slot = 0; slot < slots; ++slot) {
            float factor;
            uint32_t weight = SplineWeightIndex(slot, dim, degree, t.ks, factors, base, factor);
            if (factor != 0) {
                AddFeature(weight, factor, stable, active);
            }
        }
        if (!stable) {
            SetMaskNorm();
            ResetMask();
        }
    }

    __aicore__ inline void AddFeature(uint32_t weight, float factor, bool stable, bool active[64])
    {
        if constexpr (LargeKernelCount) {
            activeFlags.SetValue(weight, 1);
        } else {
            active[weight] = true;
        }
        uint32_t offset = weight * t.ciPad;
        if (stable) {
            SplineCompensatedAccumulate(agg[offset], xv, comp[offset], prod, z, sum, factor, t.ciPad);
        } else {
            Axpy<float, float, false>(agg[offset], xv, factor, MASK_PLACEHOLDER, 1, UnaryRepeatParams{1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Process()
    {
        LoadCaches();
        const bool cyclic = t.e < t.n && !t.hasRoot && t.nodesPerBlock > 2;
        const uint32_t step = cyclic ? t.blocks : 1;
        if (step == 0) {
            return;
        }
        const uint32_t start = cyclic ? GetBlockIdx() : GetBlockIdx() * t.nodesPerBlock;
        const uint32_t limit = start + t.nodesPerBlock;
        const uint32_t end = cyclic || limit > t.n ? t.n : limit;
        BuildHeads(start, end, step, cyclic);
        const bool sparseZero = t.e < t.n && !t.hasRoot && !t.hasBias && t.co == t.coPad && t.nodesPerBlock <= t.ci;
        if (sparseZero && start < end) {
            ClearSparseOutput(start, end, step);
        }
        for (uint32_t node = start; node < end; node += step) {
            const int32_t firstEdge = h.GetValue((node - start) / step);
            if (!sparseZero || firstEdge >= 0) {
                ProcessNode(node, firstEdge);
            }
        }
    }

private:
    __aicore__ inline void InitCacheBuffers()
    {
        if (t.cacheGraph) {
            pipe.InitBuffer(edgeBuf, ((2 * t.e + 3) / 4 * 4) * 8);
            pipe.InitBuffer(pseudoBuf, ((t.e * t.dim + 15) / 16 * 16) * sizeof(T));
            pipe.InitBuffer(nextBuf, ((t.e + 7) / 8 * 8) * 4);
            edgeLocal = edgeBuf.Get<int64_t>();
            pseudoLocal = pseudoBuf.Get<T>();
            pseudoBits = pseudoLocal.template ReinterpretCast<uint16_t>();
            nextLocal = nextBuf.Get<int32_t>();
        }
        if (t.vectorDot || t.fusedDot) {
            uint32_t dotRows = t.fusedDot ? t.kernels * t.ciPad : t.ciPad;
            pipe.InitBuffer(dotBuf, dotRows * t.coPad * 4);
            pipe.InitBuffer(broadcastBuf, t.fusedDot ? dotRows * 32 : t.ciPad * t.coPad * 4);
            dot = dotBuf.Get<float>();
            broadcast = broadcastBuf.Get<float>();
        }
        cacheSize = t.kernels * t.ci * t.coPad;
        useCache = t.cacheWeights != 0;
        if (useCache) {
            pipe.InitBuffer(cacheRawBuf, cacheSize * sizeof(T));
            cacheRaw = cacheRawBuf.Get<T>();
            if constexpr (sizeof(T) == 2) {
                pipe.InitBuffer(cacheBuf, cacheSize * 4);
                cached = cacheBuf.Get<float>();
            } else {
                cached = cacheRaw.template ReinterpretCast<float>();
            }
        }
    }

    __aicore__ inline void InitComputeBuffers()
    {
        pipe.InitBuffer(headBuf, ((t.nodesPerBlock + 7) / 8 * 8) * 4);
        if constexpr (LargeKernelCount) {
            pipe.InitBuffer(activeBuf, ((t.kernels + 7) / 8 * 8) * 4);
            activeFlags = activeBuf.Get<uint32_t>();
        }
        pipe.InitBuffer(aggBuf, t.kernels * t.ciPad * 4);
        if constexpr (sizeof(T) == 2) {
            pipe.InitBuffer(wBuf, t.ci * t.coPad * 4);
        }
        pipe.InitBuffer(compBuf, t.kernels * t.ciPad * 4);
        pipe.InitBuffer(tempBuf, t.ciPad * 12);
        pipe.InitBuffer(rawBuf, t.ci * t.coPad * sizeof(T));
        pipe.InitBuffer(xBuf, t.ciPad * 4);
        pipe.InitBuffer(xRawBuf, t.ciPad * sizeof(T));
        pipe.InitBuffer(yBuf, t.coPad * 4);
        pipe.InitBuffer(yRawBuf, t.coPad * sizeof(T));
        h = headBuf.Get<int32_t>();
        agg = aggBuf.Get<float>();
        raw = rawBuf.Get<T>();
        if constexpr (sizeof(T) == 2) {
            mat = wBuf.Get<float>();
        } else {
            mat = raw.template ReinterpretCast<float>();
        }
        comp = compBuf.Get<float>();
        prod = tempBuf.Get<float>();
        z = prod[t.ciPad];
        sum = prod[2 * t.ciPad];
        xv = xBuf.Get<float>();
        xr = xRawBuf.Get<T>();
        y = yBuf.Get<float>();
        yr = yRawBuf.Get<T>();
    }

    __aicore__ inline void LoadCaches()
    {
        if (t.cacheGraph) {
            DataCopyExtParams ep{1, t.e * 16, 0, 0, 0};
            DataCopyPadExtParams<int64_t> ez{false, 0, 0, 0};
            DataCopyPad(edgeLocal, eg, ep, ez);
            DataCopyExtParams pp{1, t.e * t.dim * uint32_t(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pz{false, 0, 0, {}};
            DataCopyPad(pseudoLocal, pg, pp, pz);
            PipeBarrier<PIPE_ALL>();
        }
        if (useCache) {
            uint32_t rows = t.co == t.coPad ? 1 : t.kernels * t.ci;
            uint32_t width = t.co == t.coPad ? cacheSize : t.co;
            DataCopyExtParams p{uint16_t(rows), width * uint32_t(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pad{true, 0, uint8_t(t.coPad - t.co), {}};
            DataCopyPad(cacheRaw, wg, p, pad);
            PipeBarrier<PIPE_ALL>();
            if constexpr (sizeof(T) == 2) {
                Cast(cached, cacheRaw, RoundMode::CAST_NONE, cacheSize);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void BuildHeads(uint32_t start, uint32_t end, uint32_t step, bool cyclic)
    {
        if (step == 0) {
            return;
        }
        for (uint32_t j = 0; j < t.nodesPerBlock; ++j) {
            h.SetValue(j, -1);
        }
        for (uint32_t e = 0; e < t.e; ++e) {
            int64_t dst = EdgeValue(t.e + e);
            if (dst >= start && dst < end && (!cyclic || (uint32_t(dst) - start) % step == 0)) {
                uint32_t j = (uint32_t(dst) - start) / step;
                NextAssign(e, h.GetValue(j));
                h.SetValue(j, e);
            }
        }
    }

    __aicore__ inline void ClearSparseOutput(uint32_t start, uint32_t end, uint32_t step)
    {
        if (step == 0) {
            return;
        }
        uint32_t rows = (end - start + step - 1) / step;
        Duplicate(raw.template ReinterpretCast<int32_t>(), int32_t(0), rows * t.co * sizeof(T) / 4);
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams zeros{uint16_t(rows), t.co * uint32_t(sizeof(T)), 0, (step - 1) * t.co * uint32_t(sizeof(T)),
                                0};
        DataCopyPad(yg[start * t.co], raw, zeros);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessNode(uint32_t node, int32_t firstEdge)
    {
        Duplicate(agg, 0.0f, t.kernels * t.ciPad);
        Duplicate(y, 0.0f, t.coPad);
        PipeBarrier<PIPE_ALL>();
        uint32_t count = 0;
        bool active[64] = {};
        if constexpr (LargeKernelCount) {
            for (uint32_t k = 0; k < t.kernels; ++k) {
                activeFlags.SetValue(k, 0);
            }
        }
        const bool stable = NeedsCompensation(firstEdge);
        if (stable) {
            Duplicate(comp, 0.0f, t.kernels * t.ciPad);
            PipeBarrier<PIPE_V>();
        }
        for (int32_t edge = firstEdge; edge >= 0; edge = NextValue(edge)) {
            const int64_t src = EdgeValue(edge);
            if (src < 0 || src >= t.n) {
                continue;
            }
            LoadRow(xg, uint32_t(src) * t.ci);
            ++count;
            AccumulateByShape(edge, stable, active);
        }
        ApplyWeights(count, active);
        if (t.mean && count > 0) {
            Muls(y, y, 1.0f / count, t.coPad);
            PipeBarrier<PIPE_V>();
        }
        AddRoot(node);
        AddBias();
        StoreNode(node);
    }

    __aicore__ inline bool NeedsCompensation(int32_t firstEdge)
    {
        uint32_t indegree = 0;
        for (int32_t e = firstEdge; e >= 0; e = NextValue(e)) {
            ++indegree;
        }
        return indegree > 128;
    }

    __aicore__ inline void AccumulateByShape(uint32_t e, bool stable, bool active[64])
    {
        if (t.dim == 2 && t.degree == 1) {
            AccumulateEdge<2, 1>(e, stable, active);
        } else if (t.dim == 2 && t.degree == 2) {
            AccumulateEdge<2, 2>(e, stable, active);
        } else if (t.dim == 2 && t.degree == 3) {
            AccumulateEdge<2, 3>(e, stable, active);
        } else if (t.dim == 3 && t.degree == 1) {
            AccumulateEdge<3, 1>(e, stable, active);
        } else if (t.dim == 3 && t.degree == 2) {
            AccumulateEdge<3, 2>(e, stable, active);
        } else if (t.dim == 3 && t.degree == 3) {
            AccumulateEdge<3, 3>(e, stable, active);
        } else {
            AccumulateEdge<0, 0>(e, stable, active);
        }
    }

    __aicore__ inline void ApplyWeights(uint32_t count, const bool active[64])
    {
        PipeBarrier<PIPE_ALL>();
        // Tiling enables fusedDot only with cached weights and contiguous channel rows.
        if (t.fusedDot) {
            if (count > 0) {
                SplineFusedDotMatrix(y, dot, broadcast, cached, agg, t.kernels * t.ci, t.coPad);
            }
            return;
        }
        for (uint32_t k = 0; k < t.kernels; ++k) {
            const bool isActive = LargeKernelCount ? activeFlags.GetValue(k) != 0 : active[k];
            if (isActive) {
                ApplyWeight(k);
            }
        }
    }

    __aicore__ inline void ApplyWeight(uint32_t k)
    {
        if (!useCache) {
            LoadMatrix(wg, k * t.ci * t.co);
        }
        LocalTensor<float> weight = useCache ? cached[k * t.ci * t.coPad] : mat;
        const auto values = agg[k * t.ciPad];
        if (t.vectorDot) {
            SplineDotMatrix(y, dot, broadcast, weight, values, t);
            return;
        }
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(t.coPad);
        for (uint32_t c = 0; c < t.ci; ++c) {
            Axpy<float, float, false>(y, weight[c * t.coPad], values.GetValue(c), MASK_PLACEHOLDER, 1,
                                      UnaryRepeatParams{1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
        }
        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline void AddRoot(uint32_t node)
    {
        if (t.hasRoot) {
            LoadRow(xg, node * t.ci);
            LoadMatrix(rg, 0);
            if (t.fusedDot) {
                SplineFusedDotMatrix(y, dot, broadcast, mat, xv, t.ci, t.coPad);
            } else if (t.vectorDot) {
                SplineDotMatrix(y, dot, broadcast, mat, xv, t);
            } else {
                for (uint32_t c = 0; c < t.ci; ++c) {
                    Axpy(y, mat[c * t.coPad], xv.GetValue(c), t.coPad);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
    }

    __aicore__ inline void AddBias()
    {
        if (t.hasBias) {
            DataCopyExtParams p{1, t.co * uint32_t(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pad{true, 0, uint8_t(t.coPad - t.co), {}};
            DataCopyPad(raw, bg, p, pad);
            PipeBarrier<PIPE_ALL>();
            if constexpr (sizeof(T) == 2) {
                Cast(mat, raw, RoundMode::CAST_NONE, t.coPad);
            }
            PipeBarrier<PIPE_ALL>();
            Add(y, y, mat, t.coPad);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreNode(uint32_t node)
    {
        PipeBarrier<PIPE_ALL>();
        if constexpr (sizeof(T) == 2) {
            Cast(yr, y, RoundMode::CAST_RINT, t.coPad);
        } else {
            DataCopy(yr, y, t.coPad);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams p{1, t.co * uint32_t(sizeof(T)), 0, 0, 0};
        DataCopyPad(yg[node * t.co], yr, p);
        PipeBarrier<PIPE_ALL>();
    }

    SplineConvTilingData t;
    bool useCache;
    uint32_t cacheSize;
    TBuf<TPosition::VECCALC> dotBuf;
    TBuf<TPosition::VECCALC> broadcastBuf;
    LocalTensor<float> dot;
    LocalTensor<float> broadcast;
    TPipe pipe;
    TBuf<TPosition::VECCALC> edgeBuf;
    TBuf<TPosition::VECCALC> pseudoBuf;
    TBuf<TPosition::VECCALC> nextBuf;
    LocalTensor<int64_t> edgeLocal;
    LocalTensor<int32_t> nextLocal;
    LocalTensor<T> pseudoLocal;
    LocalTensor<uint16_t> pseudoBits;
    TBuf<TPosition::VECCALC> headBuf;
    TBuf<TPosition::VECCALC> aggBuf;
    TBuf<TPosition::VECCALC> wBuf;
    TBuf<TPosition::VECCALC> rawBuf;
    TBuf<TPosition::VECCALC> xBuf;
    TBuf<TPosition::VECCALC> xRawBuf;
    TBuf<TPosition::VECCALC> yBuf;
    TBuf<TPosition::VECCALC> yRawBuf;
    TBuf<TPosition::VECCALC> compBuf;
    TBuf<TPosition::VECCALC> tempBuf;
    TBuf<TPosition::VECCALC> cacheBuf;
    TBuf<TPosition::VECCALC> cacheRawBuf;
    TBuf<TPosition::VECCALC> activeBuf;
    LocalTensor<uint32_t> activeFlags;
    LocalTensor<int32_t> h;
    LocalTensor<float> agg;
    LocalTensor<float> mat;
    LocalTensor<float> xv;
    LocalTensor<float> y;
    LocalTensor<float> comp;
    LocalTensor<float> prod;
    LocalTensor<float> z;
    LocalTensor<float> sum;
    LocalTensor<float> cached;
    LocalTensor<T> raw;
    LocalTensor<T> xr;
    LocalTensor<T> yr;
    LocalTensor<T> cacheRaw;
    GlobalTensor<T> xg;
    GlobalTensor<T> pg;
    GlobalTensor<T> wg;
    GlobalTensor<T> rg;
    GlobalTensor<T> bg;
    GlobalTensor<T> yg;
    GlobalTensor<uint16_t> pbits;
    GlobalTensor<int64_t> eg;
    GlobalTensor<int32_t> next;
};

} // namespace AscendC
