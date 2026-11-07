/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 黎何.
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

template <typename T, uint32_t FixedD, bool OneRow> class TinyColumnsSwiGlu {
public:
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24,
              class A25>
    __aicore__ inline __attribute__((always_inline)) void
    RunStage1a2222bc(A0& d, A1& row, A2& col, A3& gx, A4& gw1, A5& gw2, A6& gb1, A7& gb2, A8& gy, A9& hx,
                     A10& hw1, A11& hw2, A12& hb1, A13& hb2, A14& fx, A15& fw1, A16& fw2, A17& pg, A18& pu,
                     A19& ag, A20& au, A21& g, A22& u, A23& sig, A24& ones, A25& out);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               const SwiGluTinyTilingData* td);
};

template <typename T, uint32_t FixedD, bool OneRow>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18,
          class A19, class A20, class A21, class A22, class A23, class A24, class A25>
__aicore__ inline __attribute__((always_inline)) void TinyColumnsSwiGlu<T, FixedD, OneRow>::RunStage1a2222bc(
    A0& d, A1& row, A2& col, A3& gx, A4& gw1, A5& gw2, A6& gb1, A7& gb2, A8& gy, A9& hx, A10& hw1, A11& hw2,
    A12& hb1, A13& hb2, A14& fx, A15& fw1, A16& fw2, A17& pg, A18& pu, A19& ag, A20& au, A21& g, A22& u,
    A23& sig, A24& ones, A25& out)
{
    SetAtomicNone();
    SetMaskNorm();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    DataCopyExtParams cp{1, d * 2U, 0, 0, 0};
    DataCopyExtParams bp{1, 2, 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(16U - d), static_cast<T>(0)};
    DataCopyPadExtParams<T> one{true, 0, 15, static_cast<T>(0)};
    DataCopyPad(hx, gx[row * d], cp, pad);
    DataCopyPad(hw1, gw1[col * d], cp, pad);
    DataCopyPad(hw2, gw2[col * d], cp, pad);
    DataCopyPad(hb1, gb1[col], bp, one);
    DataCopyPad(hb2, gb2[col], bp, one);
    ::SharedSynccc9d9b6d<HardEvent::MTE2_V>();
    Cast(fx, hx, RoundMode::CAST_NONE, 16);
    Cast(fw1, hw1, RoundMode::CAST_NONE, 16);
    Cast(fw2, hw2, RoundMode::CAST_NONE, 16);
    Duplicate(ones, static_cast<T>(1), 16);
    PipeBarrier<PIPE_V>();
    Mul(pg, fx, fw1, 16);
    Mul(pu, fx, fw2, 16);
    PipeBarrier<PIPE_V>();
    WholeReduceSum(ag, pg, 16, 1, 1, 1, 2);
    WholeReduceSum(au, pu, 16, 1, 1, 1, 2);
    PipeBarrier<PIPE_V>();
    Cast(g, ag, RoundMode::CAST_RINT, 1);
    Cast(u, au, RoundMode::CAST_RINT, 1);
    PipeBarrier<PIPE_V>();
    Add(g, g, hb1, 1);
    Add(u, u, hb2, 1);
    PipeBarrier<PIPE_V>();
    GateCounter<T>(g, u, sig, out, ones, 1);
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopyPad(gy[row * d + col], out, bp);
}

template <typename T, uint32_t FixedD, bool OneRow>
__aicore__ inline void TinyColumnsSwiGlu<T, FixedD, OneRow>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1,
                                                                 GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                                                                 const SwiGluTinyTilingData* td)
{
    const uint32_t d = FixedD != 0 ? FixedD : (OneRow ? GetBlockNum() : static_cast<uint32_t>(td->d));
    const uint32_t task = GetBlockIdx();
    const uint64_t row = OneRow ? 0U : task / d;
    const uint32_t col = OneRow ? task : task % d;
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
    ::SharedPrimitive9aae833572<T>(b1, b2, gb1, gb2, gw1, gw2, gx, gy, w1, w2, x, y);
    LocalTensor<T> hx(TPosition::VECIN, 0, 16);
    LocalTensor<T> hw1(TPosition::VECIN, 32, 16);
    LocalTensor<T> hw2(TPosition::VECIN, 64, 16);
    LocalTensor<T> hb1(TPosition::VECIN, 96, 16);
    LocalTensor<T> hb2(TPosition::VECIN, 128, 16);
    LocalTensor<float> fx(TPosition::VECCALC, 192, 16);
    LocalTensor<float> fw1(TPosition::VECCALC, 256, 16);
    LocalTensor<float> fw2(TPosition::VECCALC, 320, 16);
    LocalTensor<float> pg(TPosition::VECCALC, 384, 16);
    LocalTensor<float> pu(TPosition::VECCALC, 448, 16);
    LocalTensor<float> ag(TPosition::VECCALC, 512, 8);
    LocalTensor<float> au(TPosition::VECCALC, 544, 8);
    LocalTensor<T> g(TPosition::VECCALC, 576, 16);
    LocalTensor<T> u(TPosition::VECCALC, 608, 16);
    LocalTensor<T> sig(TPosition::VECCALC, 640, 16);
    LocalTensor<T> ones(TPosition::VECCALC, 672, 16);
    LocalTensor<T> out(TPosition::VECOUT, 704, 16);
    RunStage1a2222bc(d, row, col, gx, gw1, gw2, gb1, gb2, gy, hx, hw1, hw2, hb1, hb2, fx, fw1, fw2, pg, pu,
                     ag, au, g, u, sig, ones, out);
    PipeBarrier<PIPE_ALL>();
}
