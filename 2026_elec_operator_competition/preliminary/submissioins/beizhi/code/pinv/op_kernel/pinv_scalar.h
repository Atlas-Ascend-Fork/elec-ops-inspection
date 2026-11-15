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

template <class T> class PinvScalar {
    TPipe pipe;
    TBuf<QuePosition::VECCALC> buf;
    GlobalTensor<T> input, output;
    GlobalTensor<float> work, vg;
    LocalTensor<float> mem, wc, vc, u, v, h, hv, a, b, tmp, tmp2, red, d, e, norms, out;
    LocalTensor<T> raw;
    LocalTensor<uint32_t> offsets, bfOffsets;
    LinalgPinvTilingData t;
    uint32_t batch, base;
    __aicore__ inline float Fabs(float x);
    __aicore__ inline float Root(float x);
    __aicore__ inline float Hyp(float x, float y);
    __aicore__ inline float Signed(float x, float y);
    __aicore__ inline void Fence();
    __aicore__ inline float Dot(LocalTensor<float> x, LocalTensor<float> y, uint32_t len);
    __aicore__ inline float MaxAbs(LocalTensor<float> x, uint32_t len);
    __aicore__ inline void ReadInput(LocalTensor<float> dst, uint32_t col);
    __aicore__ inline void ReadCol(bool right, uint32_t col, LocalTensor<float> dst);
    __aicore__ inline void WriteCol(bool right, uint32_t col, LocalTensor<float> src);
    __aicore__ inline void Rotate(bool right, uint32_t j, uint32_t k, float c, float s);
    __aicore__ inline void WriteOutput(LocalTensor<float> src, uint32_t at, uint32_t len);
    template <typename ExtractType102_0, typename ExtractType102_1, typename ExtractType102_2,
              typename ExtractType102_3, typename ExtractType102_4, typename ExtractType102_5,
              typename ExtractType102_6, typename ExtractType102_7, typename ExtractType102_8,
              typename ExtractType102_9>
    __aicore__ inline bool FastInverseBlock102(ExtractType102_0 &n, ExtractType102_1 &span, ExtractType102_2 &width,
                                               ExtractType102_3 &mat, ExtractType102_4 &u, ExtractType102_5 &v,
                                               ExtractType102_6 &h, ExtractType102_7 &tmp, ExtractType102_8 &red,
                                               ExtractType102_9 &anorm);
    template <typename ExtractType106_0, typename ExtractType106_1, typename ExtractType106_2,
              typename ExtractType106_3>
    __aicore__ inline void FastInverseBlock106(ExtractType106_0 &n, ExtractType106_1 &np, ExtractType106_2 &width,
                                               ExtractType106_3 &mat);
    template <uint32_t N = 0> __aicore__ inline bool FastInverse();
    template <typename ExtractType100_0, typename ExtractType100_1>
    __aicore__ inline void BidiagonalBlock100(ExtractType100_0 &k, ExtractType100_1 &scale);
    __aicore__ inline void Bidiagonal();
    __aicore__ inline float QRShift(float x, float y, float z, float g, float hh);
    template <typename ExtractType103_0, typename ExtractType103_1, typename ExtractType103_2,
              typename ExtractType103_3, typename ExtractType103_4, typename ExtractType103_5,
              typename ExtractType103_6, typename ExtractType103_7, typename ExtractType103_8,
              typename ExtractType103_9>
    __aicore__ inline void DiagonalQRBlock103(ExtractType103_0 &k, ExtractType103_1 &l, ExtractType103_2 &z,
                                              ExtractType103_3 &x, ExtractType103_4 &y, ExtractType103_5 &g,
                                              ExtractType103_6 &hh, ExtractType103_7 &f, ExtractType103_8 &c,
                                              ExtractType103_9 &s);
    template <typename ExtractType105_0, typename ExtractType105_1, typename ExtractType105_2,
              typename ExtractType105_3>
    __aicore__ inline void DiagonalQRBlock104Block105(ExtractType105_0 &tol, ExtractType105_1 &k,
                                                      ExtractType105_2 &flag, ExtractType105_3 &l);
    template <typename ExtractType104_0> __aicore__ inline void DiagonalQRBlock104(ExtractType104_0 &tol);
    __aicore__ inline void DiagonalQR();
    __aicore__ inline void RebuildAndPolish();
    __aicore__ inline void AssembleBlock101();
    __aicore__ inline void Assemble();

public:
    __aicore__ inline void Init(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws, LinalgPinvTilingData tt);
    __aicore__ inline void Run(bool tryFast = true, uint32_t start = ~0U);
};

#include "pinv_scalar_impl1.h"

#include "pinv_scalar_impl2.h"
