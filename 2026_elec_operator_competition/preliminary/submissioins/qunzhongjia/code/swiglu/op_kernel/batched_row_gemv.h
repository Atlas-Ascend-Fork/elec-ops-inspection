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

template <typename T, uint32_t D, uint32_t Rows> class BatchedRowGemv {
public:
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24,
              class A25, class A26>
    __aicore__ inline __attribute__((always_inline)) void
    RunStagebf7e6c70(A0& row, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& gy, A7& hw1, A8& hw2, A9& hx,
                     A10& hb1, A11& hb2, A12& fw1, A13& fw2, A14& fx, A15& pg, A16& pu, A17& ag, A18& au,
                     A19& g, A20& u, A21& sig, A22& ones, A23& out, A24& wc, A25& bc, A26& xc);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y);

private:
    static constexpr uint32_t K = (D + 15U) / 16U * 16U;
    static constexpr uint32_t W = D * K;
    static constexpr uint32_t HB = 4U * W + 2U * Rows * K;
    static constexpr uint32_t FB = HB + 4U * K;
    static constexpr uint32_t XB = FB + 8U * W;
    static constexpr uint32_t PB = XB + 4U * Rows * K;
    static constexpr uint32_t AB = PB + 8U * W;
    static constexpr uint32_t GB = AB + 8U * Rows * K;
};

template <typename T, uint32_t D, uint32_t Rows>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18,
          class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26>
__aicore__ inline __attribute__((always_inline)) void
BatchedRowGemv<T, D, Rows>::RunStagebf7e6c70(A0& row, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& gy,
                                             A7& hw1, A8& hw2, A9& hx, A10& hb1, A11& hb2, A12& fw1, A13& fw2,
                                             A14& fx, A15& pg, A16& pu, A17& ag, A18& au, A19& g, A20& u,
                                             A21& sig, A22& ones, A23& out, A24& wc, A25& bc, A26& xc)
{
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(K - D), static_cast<T>(0)};
    DataCopyPad(hw1, gw1, wc, pad);
    DataCopyPad(hw2, gw2, wc, pad);
    DataCopyPad(hb1, gb1, bc, pad);
    DataCopyPad(hb2, gb2, bc, pad);
    DataCopyPad(hx, gx[row * D], xc, pad);
    ::SharedSynccc9d9b6d<HardEvent::MTE2_V>();
    Cast(fw1, hw1, RoundMode::CAST_NONE, W);
    Cast(fw2, hw2, RoundMode::CAST_NONE, W);
    Cast(fx, hx, RoundMode::CAST_NONE, Rows * K);
    Duplicate(ag, 0.0f, Rows * K);
    Duplicate(au, 0.0f, Rows * K);
    Duplicate(ones, static_cast<T>(1), Rows * K);
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams rp{1, 1, 1, K / 8U, 0, K / 8U};
#pragma unroll

    for (uint32_t i = 0; i < Rows; ++i) {
        Mul(pg, fx[i * K], fw1, static_cast<uint64_t>(K), static_cast<uint8_t>(D), rp);
        Mul(pu, fx[i * K], fw2, static_cast<uint64_t>(K), static_cast<uint8_t>(D), rp);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(ag[i * K], pg, K, D, 1, 1, K / 8U);
        WholeReduceSum(au[i * K], pu, K, D, 1, 1, K / 8U);
        PipeBarrier<PIPE_V>();
    }
    Cast(g, ag, RoundMode::CAST_RINT, Rows * K);
    Cast(u, au, RoundMode::CAST_RINT, Rows * K);
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams bp{1, 1, 1, K / 16U, K / 16U, 0};
    Add(g, g, hb1, static_cast<uint64_t>(D), static_cast<uint8_t>(Rows), bp);
    Add(u, u, hb2, static_cast<uint64_t>(D), static_cast<uint8_t>(Rows), bp);
    PipeBarrier<PIPE_V>();
    GateCounter<T>(g, u, sig, out, ones, Rows * K);
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopyPad(gy[row * D], out, xc);
}

template <typename T, uint32_t D, uint32_t Rows>
__aicore__ inline void BatchedRowGemv<T, D, Rows>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
                                                       GM_ADDR b2, GM_ADDR y)
{
    const uint64_t row = static_cast<uint64_t>(GetBlockIdx()) * Rows;
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
    ::SharedPrimitive9aae833572<T>(b1, b2, gb1, gb2, gw1, gw2, gx, gy, w1, w2, x, y);
    LocalTensor<T> hw1(TPosition::VECIN, 0, W);
    LocalTensor<T> hw2(TPosition::VECIN, 2U * W, W);
    LocalTensor<T> hx(TPosition::VECIN, 4U * W, Rows * K);
    LocalTensor<T> hb1(TPosition::VECIN, HB, K);
    LocalTensor<T> hb2(TPosition::VECIN, HB + 2U * K, K);
    LocalTensor<float> fw1(TPosition::VECCALC, FB, W);
    LocalTensor<float> fw2(TPosition::VECCALC, FB + 4U * W, W);
    LocalTensor<float> fx(TPosition::VECCALC, XB, Rows * K);
    LocalTensor<float> pg(TPosition::VECCALC, PB, W);
    LocalTensor<float> pu(TPosition::VECCALC, PB + 4U * W, W);
    LocalTensor<float> ag(TPosition::VECCALC, AB, Rows * K);
    LocalTensor<float> au(TPosition::VECCALC, AB + 4U * Rows * K, Rows * K);
    LocalTensor<T> g(TPosition::VECCALC, GB, Rows * K);
    LocalTensor<T> u(TPosition::VECCALC, GB + 2U * Rows * K, Rows * K);
    LocalTensor<T> sig(TPosition::VECCALC, GB + 4U * Rows * K, Rows * K);
    LocalTensor<T> ones(TPosition::VECCALC, GB + 6U * Rows * K, Rows * K);
    LocalTensor<T> out(TPosition::VECOUT, GB + 8U * Rows * K, Rows * K);
    SetAtomicNone();
    SetMaskNorm();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    DataCopyExtParams wc{D, D * 2U, 0, 0, 0};
    DataCopyExtParams bc{1, D * 2U, 0, 0, 0};
    DataCopyExtParams xc{Rows, D * 2U, 0, 0, 0};
    RunStagebf7e6c70(row, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hx, hb1, hb2, fw1, fw2, fx, pg, pu, ag, au, g,
                     u, sig, ones, out, wc, bc, xc);
    PipeBarrier<PIPE_ALL>();
}
