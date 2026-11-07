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

template <typename T, uint32_t FixedD, bool SingleRow = false> class TinyVectorSwiGlu {
public:
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8,
              class B9, class B10>
    __aicore__ inline __attribute__((always_inline)) void
    LoadWeightsAndBias(B0& gb1, B1& gb2, B2& gw1, B3& gw2, B4& hb1, B5& hb2, B6& hw1, B7& hw2, B8& pad,
                       B9& rc, B10& wc);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24,
              class A25>
    __aicore__ inline __attribute__((always_inline)) void
    RunStage5d091889(A0& d, A1& row, A2& gx, A3& gw1, A4& gw2, A5& gb1, A6& gb2, A7& gy, A8& hw1, A9& hw2,
                     A10& hx, A11& hb1, A12& hb2, A13& fw1, A14& fw2, A15& fx, A16& pg, A17& pu, A18& ag,
                     A19& au, A20& g, A21& u, A22& sig, A23& ones, A24& out, A25& wc);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24,
              class A25>
    __aicore__ inline __attribute__((always_inline)) void
    RunStage607f2b6e(A0& td, A1& d, A2& gx, A3& gw1, A4& gw2, A5& gb1, A6& gb2, A7& gy, A8& hw1, A9& hw2,
                     A10& hx, A11& hb1, A12& hb2, A13& fw1, A14& fw2, A15& fx, A16& pg, A17& pu, A18& ag,
                     A19& au, A20& g, A21& u, A22& sig, A23& ones, A24& out, A25& wc);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               const SwiGluTinyTilingData* td);
};

template <typename T, uint32_t FixedD, bool SingleRow>
template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8, class B9,
          class B10>
__aicore__ inline __attribute__((always_inline)) void
TinyVectorSwiGlu<T, FixedD, SingleRow>::LoadWeightsAndBias(B0& gb1, B1& gb2, B2& gw1, B3& gw2, B4& hb1,
                                                           B5& hb2, B6& hw1, B7& hw2, B8& pad, B9& rc,
                                                           B10& wc)
{
    DataCopyPad(hw1, gw1, wc, pad);
    DataCopyPad(hw2, gw2, wc, pad);
    DataCopyPad(hb1, gb1, rc, pad);
    DataCopyPad(hb2, gb2, rc, pad);
}

template <typename T, uint32_t FixedD, bool SingleRow>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18,
          class A19, class A20, class A21, class A22, class A23, class A24, class A25>
__aicore__ inline __attribute__((always_inline)) void
TinyVectorSwiGlu<T, FixedD, SingleRow>::RunStage5d091889(A0& d, A1& row, A2& gx, A3& gw1, A4& gw2, A5& gb1,
                                                         A6& gb2, A7& gy, A8& hw1, A9& hw2, A10& hx, A11& hb1,
                                                         A12& hb2, A13& fw1, A14& fw2, A15& fx, A16& pg,
                                                         A17& pu, A18& ag, A19& au, A20& g, A21& u, A22& sig,
                                                         A23& ones, A24& out, A25& wc)
{
    DataCopyExtParams rc{1, d * 2U, 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(16U - d), static_cast<T>(0)};
    LoadWeightsAndBias(gb1, gb2, gw1, gw2, hb1, hb2, hw1, hw2, pad, rc, wc);
    DataCopyPad(hx, gx[row * d], rc, pad);
    ::SharedSynccc9d9b6d<HardEvent::MTE2_V>();
    Duplicate(ones, static_cast<T>(1), 16);
    if constexpr (FixedD == 1) {
        Mul(g, hx, hw1, 1);
        Mul(u, hx, hw2, 1);
    } else {
        Cast(fw1, hw1, RoundMode::CAST_NONE, d * 16U);
        Cast(fw2, hw2, RoundMode::CAST_NONE, d * 16U);
        Cast(fx, hx, RoundMode::CAST_NONE, 16);
        const BinaryRepeatParams rp{1, 1, 1, 2, 0, 2};
        Mul(pg, fx, fw1, static_cast<uint64_t>(16), static_cast<uint8_t>(d), rp);
        Mul(pu, fx, fw2, static_cast<uint64_t>(16), static_cast<uint8_t>(d), rp);
        WholeReduceSum(ag, pg, 16, d, 1, 1, 2);
        WholeReduceSum(au, pu, 16, d, 1, 1, 2);
        Cast(g, ag, RoundMode::CAST_RINT, d);
        Cast(u, au, RoundMode::CAST_RINT, d);
    }
    Add(g, g, hb1, d);
    Add(u, u, hb2, d);
    if constexpr (FixedD != 0) {
        FastGate<T, FixedD>(g, u, sig, out, ones);
    } else {
        GateCounter<T>(g, u, sig, out, ones, d);
    }
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopyPad(gy[row * d], out, rc);
}

template <typename T, uint32_t FixedD, bool SingleRow>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18,
          class A19, class A20, class A21, class A22, class A23, class A24, class A25>
__aicore__ inline __attribute__((always_inline)) void
TinyVectorSwiGlu<T, FixedD, SingleRow>::RunStage607f2b6e(A0& td, A1& d, A2& gx, A3& gw1, A4& gw2, A5& gb1,
                                                         A6& gb2, A7& gy, A8& hw1, A9& hw2, A10& hx, A11& hb1,
                                                         A12& hb2, A13& fw1, A14& fw2, A15& fx, A16& pg,
                                                         A17& pu, A18& ag, A19& au, A20& g, A21& u, A22& sig,
                                                         A23& ones, A24& out, A25& wc)
{
    DataCopyExtParams rc{1, d * 2U, 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(16U - d), static_cast<T>(0)};
    LoadWeightsAndBias(gb1, gb2, gw1, gw2, hb1, hb2, hw1, hw2, pad, rc, wc);
    ::SharedSynccc9d9b6d<HardEvent::MTE2_V>();
    Cast(fw1, hw1, RoundMode::CAST_NONE, d * 16U);
    Cast(fw2, hw2, RoundMode::CAST_NONE, d * 16U);
    Duplicate(ones, static_cast<T>(1), 16);
    const BinaryRepeatParams rp{1, 1, 1, 2, 0, 2};
    for (uint64_t row = GetBlockIdx(); row < td->m; row += GetBlockNum()) {
        DataCopyPad(hx, gx[row * d], rc, pad);
        ::SharedSynccc9d9b6d<HardEvent::MTE2_V>();
        Cast(fx, hx, RoundMode::CAST_NONE, 16);
        PipeBarrier<PIPE_V>();
        Mul(pg, fx, fw1, static_cast<uint64_t>(16), static_cast<uint8_t>(d), rp);
        Mul(pu, fx, fw2, static_cast<uint64_t>(16), static_cast<uint8_t>(d), rp);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(ag, pg, 16, d, 1, 1, 2);
        WholeReduceSum(au, pu, 16, d, 1, 1, 2);
        PipeBarrier<PIPE_V>();
        Cast(g, ag, RoundMode::CAST_RINT, d);
        Cast(u, au, RoundMode::CAST_RINT, d);
        PipeBarrier<PIPE_V>();
        Add(g, g, hb1, d);
        Add(u, u, hb2, d);
        PipeBarrier<PIPE_V>();
        GateCounter<T>(g, u, sig, out, ones, d);
        ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
        DataCopyPad(gy[row * d], out, rc);
        ::SharedSynccc9d9b6d<HardEvent::MTE3_MTE2>();
    }
}

template <typename T, uint32_t FixedD, bool SingleRow>
__aicore__ inline void TinyVectorSwiGlu<T, FixedD, SingleRow>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1,
                                                                   GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                                                                   const SwiGluTinyTilingData* td)
{
    const uint32_t d = FixedD != 0 ? FixedD : static_cast<uint32_t>(td->d);
    const uint64_t row = SingleRow ? GetBlockIdx() : 0U;
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
    gx.SetGlobalBuffer((__gm__ T*)x, SingleRow ? d : td->m * d);
    gy.SetGlobalBuffer((__gm__ T*)y, SingleRow ? d : td->m * d);
    gw1.SetGlobalBuffer((__gm__ T*)w1, d * d);
    gw2.SetGlobalBuffer((__gm__ T*)w2, d * d);
    gb1.SetGlobalBuffer((__gm__ T*)b1, d);
    gb2.SetGlobalBuffer((__gm__ T*)b2, d);
    LocalTensor<T> hw1(TPosition::VECIN, 0, 256);
    LocalTensor<T> hw2(TPosition::VECIN, 512, 256);
    LocalTensor<T> hx(TPosition::VECIN, 1024, 16);
    LocalTensor<T> hb1(TPosition::VECIN, 1056, 16);
    LocalTensor<T> hb2(TPosition::VECIN, 1088, 16);
    LocalTensor<float> fw1(TPosition::VECCALC, 1152, 256);
    LocalTensor<float> fw2(TPosition::VECCALC, 2176, 256);
    LocalTensor<float> fx(TPosition::VECCALC, 3200, 16);
    LocalTensor<float> pg(TPosition::VECCALC, 3264, 256);
    LocalTensor<float> pu(TPosition::VECCALC, 4288, 256);
    LocalTensor<float> ag(TPosition::VECCALC, 5312, 16);
    LocalTensor<float> au(TPosition::VECCALC, 5376, 16);
    LocalTensor<T> g(TPosition::VECCALC, 5440, 16);
    LocalTensor<T> u(TPosition::VECCALC, 5472, 16);
    LocalTensor<T> sig(TPosition::VECCALC, 5504, 16);
    LocalTensor<T> ones(TPosition::VECCALC, 5536, 16);
    LocalTensor<T> out(TPosition::VECOUT, 5568, 16);
    SetAtomicNone();
    SetMaskNorm();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    DataCopyExtParams wc{static_cast<uint16_t>(d), d * 2U, 0, 0, 0};
    if constexpr (SingleRow) {
        RunStage5d091889(d, row, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hx, hb1, hb2, fw1, fw2, fx, pg, pu, ag,
                         au, g, u, sig, ones, out, wc);
    } else {
        RunStage607f2b6e(td, d, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hx, hb1, hb2, fw1, fw2, fx, pg, pu, ag,
                         au, g, u, sig, ones, out, wc);
    }
}
