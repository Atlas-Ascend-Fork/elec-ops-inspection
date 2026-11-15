/*
 * Copyright (c) 2026 Beizhi contributors.
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

template <class T, uint32_t N> class TinyPinv {
    GlobalTensor<T> input, output;
    LocalTensor<float> mem;
    LocalTensor<uint32_t> offsets;
    LocalTensor<T> raw;
    LinalgPinvTilingData t;
    uint32_t base;

    __aicore__ inline void Fence()
    {
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Sum(LocalTensor<float> dst, LocalTensor<float> src, uint32_t len)
    {
        PinvTinySum(mem, dst, src, len);
    }

    __aicore__ inline void WriteOutput(LocalTensor<float> src, uint32_t at, uint32_t len)
    {
        PinvWriteOutput(output, raw, src, at, len);
    }

    template <typename ExtractType108_0, typename ExtractType108_1, typename ExtractType108_2,
              typename ExtractType108_3, typename ExtractType108_4, typename ExtractType108_5,
              typename ExtractType108_6, typename ExtractType108_7, typename ExtractType108_8,
              typename ExtractType108_9>
    __aicore__ inline bool FastInverseBlock108(ExtractType108_0 &n, ExtractType108_1 &span, ExtractType108_2 &width,
                                               ExtractType108_3 &mat, ExtractType108_4 &u, ExtractType108_5 &v,
                                               ExtractType108_6 &h, ExtractType108_7 &tmp, ExtractType108_8 &red,
                                               ExtractType108_9 &anorm)
    {
        return PinvTinyEliminate<T>(n, span, width, mat, u, v, h, tmp, red, anorm, offsets);
    }

    template <typename ExtractType112_0, typename ExtractType112_1, typename ExtractType112_2,
              typename ExtractType112_3>
    __aicore__ inline void FastInverseBlock112(ExtractType112_0 &n, ExtractType112_1 &np, ExtractType112_2 &width,
                                               ExtractType112_3 &mat)
    {
        PinvLoadTiny(input, raw, base, n, np, width, mat);
    }

    __aicore__ inline bool FastInverse()
    {
        uint32_t n = N;
        uint32_t np = (N + 7) / 8 * 8;
        uint32_t span = np * 2;
        uint32_t width = span + 8;
        auto mat = mem;
        auto u = mem[38408];
        auto v = mem[38928];
        auto h = mem[39448];
        auto tmp = mem[41496];
        auto red = mem[42528];
        Duplicate(mat, 0.0f, n * width);
        Fence();
        FastInverseBlock112(n, np, width, mat);
        Fence();
        Mul(mem[16384], mat, mat, np, n, {1, 1, 1, (uint8_t)(np / 8), (uint8_t)(width / 8), (uint8_t)(width / 8)});
        PipeBarrier<PIPE_V>();
        Sum(red, mem[16384], n * np);
        LocalSync<HardEvent::V_S>();
        float anorm = red.GetValue(0);
        if (!(anorm > 0))
            return false;
        for (uint32_t i = 0; i < n; i++) {
            mat.SetValue(i * width + np + i, 1.0f);
            offsets.SetValue(i, i * width * 4);
        }
        Fence();
        if (!FastInverseBlock108(n, span, width, mat, u, v, h, tmp, red, anorm))
            return false;
        Fence(); // Augmented Frobenius norm is a conservative inverse-norm bound.
        Mul(mem[16384], mat[np], mat[np], np, n,
            {1, 1, 1, (uint8_t)(np / 8), (uint8_t)(width / 8), (uint8_t)(width / 8)});
        PipeBarrier<PIPE_V>();
        Sum(red, mem[16384], n * np);
        LocalSync<HardEvent::V_S>();
        float ix = red.GetValue(0);
        float threshold = t.atol * t.atol > anorm * t.rtol * t.rtol ? t.atol * t.atol : anorm * t.rtol * t.rtol;
        if (!(threshold * ix < 25.0f))
            return false;
        PinvStoreTiny(output, raw, mat, base, n, np, width);
        return true;
    }

public:
    __aicore__ inline void Init(GM_ADDR ai, GM_ADDR ao, LinalgPinvTilingData tt)
    {
        t = tt;
        input.SetGlobalBuffer((__gm__ T *)ai);
        output.SetGlobalBuffer((__gm__ T *)ao);
        InitSocState();
        mem = LocalTensor<float>(TPosition::VECCALC, 0, 48640);
        offsets = mem.template ReinterpretCast<uint32_t>()[45568];
        raw = mem.template ReinterpretCast<T>()[32768 * 4 / sizeof(T)];
    }

    __aicore__ inline bool Run()
    {
        for (uint32_t batch = GetBlockIdx(); batch < t.batch; batch += GetBlockNum()) {
            base = batch * N * N;
            if (!FastInverse())
                return false;
        }
        return true;
    }
};

template <class T, uint32_t N, uint32_t G> class ColumnTinyPinv {
    GlobalTensor<T> input, output;
    GlobalTensor<float> state;
    LocalTensor<float> mem;
    LocalTensor<uint32_t> offsets, permutation;
    LinalgPinvTilingData t;
    uint32_t part, base;
    float anorm;
    static constexpr uint32_t n = N, np = (N + 7) / 8 * 8, cols = (N + G - 1) / G, span = n + cols,
                              ld = (N > 16 && G > 1) ? 136 : np + (N == 50 ? 16 : 8);

    __aicore__ inline void Fence()
    {
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Sum(LocalTensor<float> dst, LocalTensor<float> src, uint32_t len)
    {
        PinvTinySum(mem, dst, src, len);
    }

    template <typename ExtractType107_0, typename ExtractType107_1, typename ExtractType107_2,
              typename ExtractType107_3, typename ExtractType107_4, typename ExtractType107_5,
              typename ExtractType107_6, typename ExtractType107_7>
    __aicore__ inline bool ComputeBlock107(ExtractType107_0 &mat, ExtractType107_1 &piv, ExtractType107_2 &h,
                                           ExtractType107_3 &values, ExtractType107_4 &tmp, ExtractType107_5 &red,
                                           ExtractType107_6 &active, ExtractType107_7 &minPivot2)
    {
        for (uint32_t j = 0; j < n; j++) {
            auto column = mat[j * ld];
            Abs(tmp, column, n);
            PipeBarrier<PIPE_V>();
            uint64_t mask[2] = {active, 0};
            WholeReduceMax(red, tmp, mask, 1, 1, 1, 8);
            LocalSync<HardEvent::V_S>();
            float mx;
            float indexBits;
            GetReduceMaxMinCount(mx, indexBits);

            union IndexBits {
                float value;
                uint32_t bits;
            };

            IndexBits chosen;
            chosen.value = indexBits;
            uint32_t pi = chosen.bits;
            if (!(mx * mx > minPivot2))
                return false;
            float pval = column.GetValue(pi);
            if (pval == 0.0f)
                return false;
            uint32_t remaining = span - j - 1;
            Gather(values, mat[(j + 1) * ld], offsets, pi * 4, remaining);
            uint64_t pivotMask[2] = {1ULL << (pi % 8), 0};
            Adds(column[pi / 8 * 8], column[pi / 8 * 8], -1.0f, pivotMask, 1, {1, 1, 8, 8});
            float inversePivot = 1.0f / pval;
            active &= ~(1ULL << pi);
            permutation.SetValue(j, (n * ld + pi) * 4);
            PipeBarrier<PIPE_V>();
            Muls(piv, column, -inversePivot, np);
            if constexpr (N > 16 && G > 1)
                h = mem[39424 + ((j + 9) & 15) * 8];
            Brcb(h, values, (remaining + 7) / 8, {1, 8});
            PipeBarrier<PIPE_V>();
            MulAddDst(mat[(j + 1) * ld], piv, h, np, remaining, {1, 1, 0, (uint8_t)(ld / 8), 0, 1});
            PipeBarrier<PIPE_V>();
        }
        return true;
    }

    template <typename ExtractType109_0> __aicore__ inline void ComputeBlock109(ExtractType109_0 &mat)
    {
        if constexpr (N == 15) {
            auto load = mem[16384];
            auto packed = mem[24096];
            auto loadIndices = mem[42048].template ReinterpretCast<uint32_t>();
            for (uint32_t c = 0; c < 16; c++)
                loadIndices.SetValue(c, c * 4);
            LocalSync<HardEvent::S_V>();
            for (uint32_t step = 16; step < n * np; step *= 2) {
                Adds(loadIndices[step].template ReinterpretCast<int32_t>(),
                     loadIndices.template ReinterpretCast<int32_t>(), int32_t(step / 16 * n * 4), step);
                PipeBarrier<PIPE_V>();
            }
            LocalSync<HardEvent::MTE2_V>();
            Gather(packed, load, loadIndices, 0, n * np);
            PipeBarrier<PIPE_V>();
            Adds(mat, packed, 0.0f, np, n, {1, 1, (uint8_t)(ld / 8), (uint8_t)(np / 8)});
            PipeBarrier<PIPE_V>();
            uint64_t padding[2] = {128, 0};
            Duplicate(mat[8], 0.0f, padding, n, 1, ld / 8);
            PipeBarrier<PIPE_V>();
        } else
            LocalSync<HardEvent::MTE2_V>();
    }

    template <typename ExtractType110_0, typename ExtractType110_1, typename ExtractType110_2>
    __aicore__ inline bool ComputeBlock110(ExtractType110_0 &mat, ExtractType110_1 &red, ExtractType110_2 &factor)
    {
        if constexpr (N <= 16) {
            if (anorm < 0.000000059604644775390625f || anorm > 16777216.0f)
                return false;
        } else {
            union ScaleBits {
                uint32_t bits;
                float value;
            };

            ScaleBits scale;
            int exponent = int((red.template ReinterpretCast<uint32_t>().GetValue(0) >> 23) & 255) - 127;
            int halfExponent = exponent >= 0 ? (exponent + 1) / 2 : exponent / 2;
            scale.bits = uint32_t(127 - halfExponent) << 23;
            factor = scale.value;
            Muls(mat, mat, factor, np, n, {1, 1, (uint8_t)(ld / 8), (uint8_t)(ld / 8)});
        }
        return true;
    }

    template <typename ExtractType113_0> __aicore__ inline void ComputeBlock113(ExtractType113_0 &mat)
    {
        if constexpr (N == 15) {
            auto load = mem[16384];
            Duplicate(load, 0.0f, 232);
            LocalSync<HardEvent::V_MTE2>();
            DataCopyPad(load, input[base], {1, n * n * 4, 0, 0, 0}, {false, 0, 0, 0});
        } else
            DataCopyPad(mat, input[base], {(uint16_t)n, n * 4, 0, (ld - np) / 8, 0}, {true, 0, (uint8_t)(np - n), 0});
    }

    __aicore__ inline void ComputeBlock114()
    {
        for (uint32_t step = 8; step < span; step *= 2) {
            Adds(offsets[step].template ReinterpretCast<int32_t>(), offsets.template ReinterpretCast<int32_t>(),
                 int32_t(step * ld * 4), step);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline float Compute()
    {
        auto mat = mem;
        auto piv = mem[38928];
        auto h = mem[39448];
        auto values = mem[41024];
        auto tmp = mem[41496];
        auto red = mem[42528];
        ComputeBlock113(mat);
        Duplicate(mat[n * ld], 0.0f, cols * ld);
        for (uint32_t c = 0; c < 8; c++)
            offsets.SetValue(c, c * ld * 4);
        LocalSync<HardEvent::S_V>();
        ComputeBlock114();
        for (uint32_t r = n; r < np; r++)
            permutation.SetValue(r, n * ld * 4);
        auto pattern = mem.template ReinterpretCast<uint32_t>()[45760];
        uint64_t bits = (~0ULL) >> (64 - N);
        pattern.SetValue(0, uint32_t(bits));
        pattern.SetValue(1, uint32_t(bits >> 32));
        ComputeBlock109(mat);
        Mul(mem[16384], mat, mat, np, n, {1, 1, 1, (uint8_t)(np / 8), (uint8_t)(ld / 8), (uint8_t)(ld / 8)});
        PipeBarrier<PIPE_V>();
        Sum(red, mem[16384], n * np);
        LocalSync<HardEvent::V_S>();
        anorm = red.GetValue(0);
        if (!(anorm > 0))
            return -1;
        float factor = 1.0f;
        if (!ComputeBlock110(mat, red, factor))
            return -1;
        for (uint32_t c = 0; c < cols; c++)
            if (part * cols + c < n)
                mat.SetValue((n + c) * ld + part * cols + c, factor);
        LocalSync<HardEvent::S_V>();
        PipeBarrier<PIPE_V>();
        uint64_t active = bits;
        float minPivot2 = anorm * factor * factor * 1e-10f;
        if (!ComputeBlock107(mat, piv, h, values, tmp, red, active, minPivot2))
            return -1;
        Mul(mem[16384], mat[n * ld], mat[n * ld], np, cols,
            {1, 1, 1, (uint8_t)(np / 8), (uint8_t)(ld / 8), (uint8_t)(ld / 8)});
        PipeBarrier<PIPE_V>();
        Sum(red, mem[16384], np * cols);
        LocalSync<HardEvent::V_S>();
        return red.GetValue(0);
    }

    __aicore__ inline void PrepareOutput()
    {
        uint32_t first = part * cols;
        uint32_t ac = first < n ? (n - first < cols ? n - first : cols) : 0;
        if (!ac)
            return;
        auto h = mem[39448];
        auto packed = mem[(N > 16 && G > 1) ? 12000 : 10000];
        auto compact = mem[24096];
        auto indices = mem.template ReinterpretCast<int32_t>()[20000];
        LocalSync<HardEvent::S_V>();
        Brcb(h, offsets.template ReinterpretCast<float>(), (ac + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        Add(indices, permutation.template ReinterpretCast<int32_t>(), h.template ReinterpretCast<int32_t>(), np, ac,
            {1, 1, 0, (uint8_t)(np / 8), 0, 1});
        PipeBarrier<PIPE_V>();
        Gather(packed, mem, indices.template ReinterpretCast<uint32_t>(), 0, ac * np);
        if constexpr (N % 8 != 0) {
            PipeBarrier<PIPE_V>();
            uint64_t count = 0;
            GatherMask(compact, packed, mem.template ReinterpretCast<uint32_t>()[45760], false, 0,
                       {1, (uint16_t)ac, (uint16_t)(np / 8), 0}, count);
        }
    }

    __aicore__ inline void WriteOutput()
    {
        uint32_t first = part * cols;
        uint32_t ac = first < n ? (n - first < cols ? n - first : cols) : 0;
        if (!ac)
            return;
        auto src = N % 8 == 0 ? mem[(N > 16 && G > 1) ? 12000 : 10000] : mem[24096];
        LocalSync<HardEvent::V_MTE3>();
        DataCopyPad(output[base + first * n], src, {1, ac * n * 4, 0, 0, 0});
        LocalSync<HardEvent::MTE3_MTE2>();
    }

public:
    __aicore__ inline void Init(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws, LinalgPinvTilingData tt)
    {
        t = tt;
        part = GetBlockIdx() % G;
        input.SetGlobalBuffer((__gm__ T *)ai);
        output.SetGlobalBuffer((__gm__ T *)ao);
        if constexpr (G > 1)
            state.SetGlobalBuffer((__gm__ float *)ws);
        InitSocState();
        mem = LocalTensor<float>(TPosition::VECCALC, 0, 48640);
        offsets = mem.template ReinterpretCast<uint32_t>()[45568];
        permutation = mem.template ReinterpretCast<uint32_t>()[45824];
    }

    __aicore__ inline void Init(GM_ADDR ai, GM_ADDR ao, LinalgPinvTilingData tt)
    {
        Init(ai, ao, nullptr, tt);
    }

    __aicore__ inline bool RunBlock111()
    {
        for (uint32_t batch = GetBlockIdx(); batch < t.batch; batch += GetBlockNum()) {
            base = batch * n * n;
            float norm = Compute();
            if (!(norm >= 0))
                return false;
            PrepareOutput();
            float threshold = t.atol * t.atol > anorm * t.rtol * t.rtol ? t.atol * t.atol : anorm * t.rtol * t.rtol;
            if (!(threshold * norm < 25.0f)) {
                Fence();
                return false;
            }
            WriteOutput();
        }
        return true;
    }

    __aicore__ inline bool Run()
    {
        if constexpr (G == 1) {
            if (!RunBlock111())
                return false;
            return true;
        } else {
            base = GetBlockIdx() / G * n * n;
            float norm = Compute();
            auto red = mem[42528];
            auto status = mem[42600];
            red.SetValue(0, norm);
            LocalSync<HardEvent::S_MTE3>();
            DataCopy(state[GetBlockIdx() * 8], red, 8);
            LocalSync<HardEvent::MTE3_S>();
            SyncAll();
            DataCopy(status, state[GetBlockIdx() / G * G * 8], G * 8);
            PrepareOutput();
            LocalSync<HardEvent::MTE2_S>();
            float total = 0;
            for (uint32_t k = 0; k < G; k++) {
                float x = status.GetValue(k * 8);
                if (!(x >= 0)) {
                    Fence();
                    return false;
                }
                total += x;
            }
            float threshold = t.atol * t.atol > anorm * t.rtol * t.rtol ? t.atol * t.atol : anorm * t.rtol * t.rtol;
            if (!(threshold * total < 25.0f)) {
                Fence();
                return false;
            }
            WriteOutput();
            return true;
        }
    }
};

template <class T, uint32_t N, uint32_t G>
__aicore__ __attribute__((noinline, aligned(128))) bool AlignedTinyInverse(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws,
                                                                           LinalgPinvTilingData t)
{
    ColumnTinyPinv<T, N, G> fast;
    fast.Init(ai, ao, ws, t);
    return fast.Run();
}

template <class T, uint32_t N, uint32_t G>
__aicore__ __attribute__((noinline)) bool SmallInverse(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws, LinalgPinvTilingData t)
{
    ColumnTinyPinv<T, N, G> fast;
    fast.Init(ai, ao, ws, t);
    return fast.Run();
}
