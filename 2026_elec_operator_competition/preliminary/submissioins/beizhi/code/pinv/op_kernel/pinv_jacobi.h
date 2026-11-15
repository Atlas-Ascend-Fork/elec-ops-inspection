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

template <class T, uint32_t N>
__aicore__ __attribute__((noinline, aligned(128))) bool SpecializedInverse(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws,
                                                                           LinalgPinvTilingData t)
{
    if constexpr (sizeof(T) == 4 && N == 199) {
        PinvInverse199<T, N, true> fast;
        fast.Init(ai, ao, ws, t);
        return fast.Run();
    } else {
        ParallelInverse<T, N> fast;
        fast.Init(ai, ao, ws, t);
        return fast.Run();
    }
}

template <class T>
__aicore__ __attribute__((noinline, aligned(128))) bool WideInPlaceInverse(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws,
                                                                           LinalgPinvTilingData t)
{
    PinvInverse199<T, 199, false> fast;
    fast.Init(ai, ao, ws, t);
    return fast.Run();
}

template <class T> class PinvJacobi {
    TPipe pipe;
    TBuf<QuePosition::VECCALC> buf;
    LinalgPinvTilingData t;
    GlobalTensor<T> input, output;
    GlobalTensor<float> all, work, vg, ns, state, yall, y;
    LocalTensor<float> mem, u, v, a, b, tmp, tmp2, red, vj, vk, cj, ck, out, converted;
    LocalTensor<T> raw;
    LocalTensor<uint32_t> offsets, bfOffsets;
    uint32_t part, groupid, batch, base;
    bool inputCached;
    __aicore__ inline void Fence();
    __aicore__ inline void Sync();
    __aicore__ inline float AbsScalar(float z);
    __aicore__ inline float Root(float z);
    __aicore__ inline float Dot(LocalTensor<float> x, LocalTensor<float> y, uint32_t count);
    __aicore__ inline void Store(GlobalTensor<float> dst, float value);
    __aicore__ inline float Load(GlobalTensor<float> src);
    __aicore__ inline void ReadInput(uint32_t col, LocalTensor<float> dst);
    __aicore__ inline void Read(GlobalTensor<float> src, LocalTensor<float> dst, uint32_t len);
    __aicore__ inline void Write(GlobalTensor<float> dst, LocalTensor<float> src, uint32_t len);
    __aicore__ inline bool Pair(uint32_t j, uint32_t k, float tiny, bool precise = false);
    template <typename ExtractType63_0, typename ExtractType63_1, typename ExtractType63_2, typename ExtractType63_3,
              typename ExtractType63_4>
    __aicore__ inline void PreciseDotBlock63(ExtractType63_0 &sums, ExtractType63_1 &zz, ExtractType63_2 &errors,
                                             ExtractType63_3 &low, ExtractType63_4 &count);
    template <typename ExtractType65_0, typename ExtractType65_1, typename ExtractType65_2>
    __aicore__ inline void PreciseDotBlock65(ExtractType65_0 &low, ExtractType65_1 &sum, ExtractType65_2 &err);
    __aicore__ inline float PreciseDot(LocalTensor<float> av, LocalTensor<float> xv, uint32_t len);
    template <typename ExtractType62_0> __aicore__ inline void RebuildBlock62(ExtractType62_0 &compensation);
    __aicore__ inline void Rebuild();
    __aicore__ inline void OutputRow(uint32_t row, float threshold2);
    template <typename ExtractType61_0, typename ExtractType61_1>
    __aicore__ inline void RefineBlock61(ExtractType61_0 &rr, ExtractType61_1 &correction);
    __aicore__ inline void Refine();

public:
    __aicore__ inline void Init(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws, LinalgPinvTilingData tt);
    __aicore__ inline void FormGram(GM_ADDR gram);
    __aicore__ inline void GramOutput(GM_ADDR inverse);
    template <typename ExtractType64_0> __aicore__ inline void RunBlock60Block64(ExtractType64_0 &blocks);
    template <typename ExtractType60_0, typename ExtractType60_1, typename ExtractType60_2>
    __aicore__ inline void RunBlock60(ExtractType60_0 &blocks, ExtractType60_1 &circle, ExtractType60_2 &pairs);
    __aicore__ inline void Run();
};

#include "pinv_jacobi_impl1.h"

#include "pinv_jacobi_impl2.h"
