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

template <typename T, uint32_t FixedD, uint32_t Rows = 1, bool SingleBatch = false>
class BatchedTinyVectorSwiGlu {
public:
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24>
    __aicore__ inline __attribute__((always_inline)) void
    RunStagebf47799f(A0& td, A1& d, A2& gx, A3& gy, A4& hx, A5& hb1, A6& hb2, A7& fw1, A8& fw2, A9& fx,
                     A10& pg, A11& pu, A12& ag, A13& au, A14& g, A15& u, A16& sig, A17& ones, A18& out,
                     A19& pad, A20& rp, A21& bp, A22& first, A23& stride, A24& group);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20, class A21, class A22>
    __aicore__ inline __attribute__((always_inline)) void
    RunStage91fa1677(A0& td, A1& d, A2& gx, A3& gw1, A4& gw2, A5& gb1, A6& gb2, A7& gy, A8& hw1, A9& hw2,
                     A10& hx, A11& hb1, A12& hb2, A13& fw1, A14& fw2, A15& fx, A16& pg, A17& pu, A18& ag,
                     A19& au, A20& g, A21& u, A22& sig);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               const SwiGluTinyTilingData* td);
};

template <typename T, uint32_t FixedD, uint32_t Rows, bool SingleBatch>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18,
          class A19, class A20, class A21, class A22, class A23, class A24>
__aicore__ inline __attribute__((always_inline)) void
BatchedTinyVectorSwiGlu<T, FixedD, Rows, SingleBatch>::RunStagebf47799f(
    A0& td, A1& d, A2& gx, A3& gy, A4& hx, A5& hb1, A6& hb2, A7& fw1, A8& fw2, A9& fx, A10& pg, A11& pu,
    A12& ag, A13& au, A14& g, A15& u, A16& sig, A17& ones, A18& out, A19& pad, A20& rp, A21& bp, A22& first,
    A23& stride, A24& group)
{
    const uint64_t row = first + group * stride;
    const uint32_t nr = SingleBatch ? Rows : (td->m - row < Rows ? static_cast<uint32_t>(td->m - row) : Rows);
    DataCopyExtParams rc{static_cast<uint16_t>(nr), d * 2U, 0, 0, 0};
    if constexpr (!SingleBatch) {
        DataCopyPad(hx, gx[row * d], rc, pad);
        ::SharedSynccc9d9b6d<HardEvent::MTE2_V>();
    }
    Cast(fx, hx, RoundMode::CAST_NONE, nr * 16U);
    PipeBarrier<PIPE_V>();
#pragma unroll

    for (uint32_t i = 0; i < nr; ++i) {
        Mul(pg, fx[i * 16U], fw1, static_cast<uint64_t>(16), static_cast<uint8_t>(d), rp);
        Mul(pu, fx[i * 16U], fw2, static_cast<uint64_t>(16), static_cast<uint8_t>(d), rp);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(ag[i * 16U], pg, 16, d, 1, 1, 2);
        WholeReduceSum(au[i * 16U], pu, 16, d, 1, 1, 2);
        PipeBarrier<PIPE_V>();
    }
    Cast(g, ag, RoundMode::CAST_RINT, nr * 16U);
    Cast(u, au, RoundMode::CAST_RINT, nr * 16U);
    PipeBarrier<PIPE_V>();
    Add(g, g, hb1, static_cast<uint64_t>(d), static_cast<uint8_t>(nr), bp);
    Add(u, u, hb2, static_cast<uint64_t>(d), static_cast<uint8_t>(nr), bp);
    PipeBarrier<PIPE_V>();
    GateCounter<T>(g, u, sig, out, ones, nr * 16U);
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopyPad(gy[row * d], out, rc);
    if constexpr (!SingleBatch) {
        ::SharedSynccc9d9b6d<HardEvent::MTE3_MTE2>();
    }
}

template <typename T, uint32_t FixedD, uint32_t Rows, bool SingleBatch>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18,
          class A19, class A20, class A21, class A22>
__aicore__ inline __attribute__((always_inline)) void
BatchedTinyVectorSwiGlu<T, FixedD, Rows, SingleBatch>::RunStage91fa1677(
    A0& td, A1& d, A2& gx, A3& gw1, A4& gw2, A5& gb1, A6& gb2, A7& gy, A8& hw1, A9& hw2, A10& hx, A11& hb1,
    A12& hb2, A13& fw1, A14& fw2, A15& fx, A16& pg, A17& pu, A18& ag, A19& au, A20& g, A21& u, A22& sig)
{
    constexpr uint32_t fb = 1088 + Rows * 32;
    constexpr uint32_t pb = fb + 2048 + Rows * 64;
    constexpr uint32_t ab = pb + 2048;
    constexpr uint32_t hb = ab + Rows * 128;
    LocalTensor<T> ones(TPosition::VECCALC, hb + Rows * 96, Rows * 16);
    LocalTensor<T> out(TPosition::VECOUT, hb + Rows * 128, Rows * 16);
    SetAtomicNone();
    SetMaskNorm();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    DataCopyExtParams wc{static_cast<uint16_t>(d), d * 2U, 0, 0, 0};
    DataCopyExtParams bc{1, d * 2U, 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(16U - d), static_cast<T>(0)};
    DataCopyPad(hw1, gw1, wc, pad);
    DataCopyPad(hw2, gw2, wc, pad);
    DataCopyPad(hb1, gb1, bc, pad);
    DataCopyPad(hb2, gb2, bc, pad);
    if constexpr (SingleBatch) {
        DataCopyExtParams rc{static_cast<uint16_t>(Rows), d * 2U, 0, 0, 0};
        DataCopyPad(hx, gx[static_cast<uint64_t>(GetBlockIdx()) * Rows * d], rc, pad);
    }
    ::SharedSynccc9d9b6d<HardEvent::MTE2_V>();
    Cast(fw1, hw1, RoundMode::CAST_NONE, d * 16U);
    Cast(fw2, hw2, RoundMode::CAST_NONE, d * 16U);
    Duplicate(ones, static_cast<T>(1), Rows * 16U);
    Duplicate(ag, 0.0f, Rows * 16U);
    Duplicate(au, 0.0f, Rows * 16U);
    const BinaryRepeatParams rp{1, 1, 1, 2, 0, 2};
    const BinaryRepeatParams bp{1, 1, 1, 1, 1, 0};
    const uint64_t first = static_cast<uint64_t>(GetBlockIdx()) * Rows;
    const uint64_t stride = SingleBatch ? 0U : static_cast<uint64_t>(GetBlockNum()) * Rows;
    uint64_t groups = 1U;
    if constexpr (!SingleBatch) {
        groups = (td->m - first + stride - 1U) / stride;
    }
    for (uint64_t group = 0; group < groups; ++group) {
        RunStagebf47799f(td, d, gx, gy, hx, hb1, hb2, fw1, fw2, fx, pg, pu, ag, au, g, u, sig, ones, out, pad,
                         rp, bp, first, stride, group);
    }
}

template <typename T, uint32_t FixedD, uint32_t Rows, bool SingleBatch>
__aicore__ inline void BatchedTinyVectorSwiGlu<T, FixedD, Rows, SingleBatch>::Run(
    GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y, const SwiGluTinyTilingData* td)
{
    const uint32_t d = FixedD != 0 ? FixedD : static_cast<uint32_t>(td->d);
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
    ::SharedPrimitivefa859e08f7<T>(b1, b2, d, gb1, gb2, gw1, gw2, gx, gy, w1, w2, x, y);
    LocalTensor<T> hw1(TPosition::VECIN, 0, 256);
    LocalTensor<T> hw2(TPosition::VECIN, 512, 256);
    LocalTensor<T> hx(TPosition::VECIN, 1024, Rows * 16);
    LocalTensor<T> hb1(TPosition::VECIN, 1024 + Rows * 32, 16);
    LocalTensor<T> hb2(TPosition::VECIN, 1056 + Rows * 32, 16);
    constexpr uint32_t fb = 1088 + Rows * 32;
    LocalTensor<float> fw1(TPosition::VECCALC, fb, 256);
    LocalTensor<float> fw2(TPosition::VECCALC, fb + 1024, 256);
    LocalTensor<float> fx(TPosition::VECCALC, fb + 2048, Rows * 16);
    constexpr uint32_t pb = fb + 2048 + Rows * 64;
    LocalTensor<float> pg(TPosition::VECCALC, pb, 256);
    LocalTensor<float> pu(TPosition::VECCALC, pb + 1024, 256);
    constexpr uint32_t ab = pb + 2048;
    LocalTensor<float> ag(TPosition::VECCALC, ab, Rows * 16);
    LocalTensor<float> au(TPosition::VECCALC, ab + Rows * 64, Rows * 16);
    constexpr uint32_t hb = ab + Rows * 128;
    LocalTensor<T> g(TPosition::VECCALC, hb, Rows * 16);
    LocalTensor<T> u(TPosition::VECCALC, hb + Rows * 32, Rows * 16);
    LocalTensor<T> sig(TPosition::VECCALC, hb + Rows * 64, Rows * 16);
    RunStage91fa1677(td, d, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hx, hb1, hb2, fw1, fw2, fx, pg, pu, ag, au,
                     g, u, sig);
    if constexpr (SingleBatch) {
        PipeBarrier<PIPE_ALL>();
    }
}
