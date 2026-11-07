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

template <typename T> class SubBlockGemv4 {
public:
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8,
              class B9, class B10, class B11>
    __aicore__ inline __attribute__((always_inline)) void
    SharedStage9a0a31e17c(B0& acc, B1& bias, B2& gb1, B3& gb2, B4& gw1, B5& gw2, B6& hx, B7& pairs, B8& prod,
                          B9& w, B10& x, B11& xc);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
    __aicore__ inline __attribute__((always_inline)) void
    RunStage5182e543(A0& x, A1& gw1, A2& gw2, A3& gb1, A4& gb2, A5& gy, A6& hx, A7& w, A8& bias, A9& prod,
                     A10& pairs, A11& acc, A12& g, A13& sig, A14& ones, A15& out, A16& xc);
#else
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15>
    __aicore__ inline __attribute__((always_inline)) void
    RunStage7cc3bb46(A0& x, A1& gw1, A2& gw2, A3& gb1, A4& gb2, A5& gy, A6& hx, A7& w, A8& bias, A9& prod,
                     A10& pairs, A11& acc, A12& g, A13& sig, A14& ones, A15& out);
#endif
    template <class B0, class B1, class B2, class B3, class B4, class B5>
    __aicore__ inline __attribute__((always_inline)) void RunActive(B0& b1, B1& b2, B2& w1, B3& w2, B4& x,
                                                                    B5& y);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y);
};

template <typename T>
template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8, class B9,
          class B10, class B11>
__aicore__ inline __attribute__((always_inline)) void
SubBlockGemv4<T>::SharedStage9a0a31e17c(B0& acc, B1& bias, B2& gb1, B3& gb2, B4& gw1, B5& gw2, B6& hx,
                                        B7& pairs, B8& prod, B9& w, B10& x, B11& xc)
{
    const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
    GlobalTensor<uint64_t> xword;
    xword.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(x), 1);
    DataCachePreload(xword, static_cast<int16_t>(0));
    DataCopy(w, gw1, 16);
    DataCopy(w[16], gw2, 16);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopyPad(bias, gb1, xc, pad);
    DataCopyPad(bias[16], gb2, xc, pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Duplicate(prod, 0.0f, 32);
    const uint64_t xv = xword.GetValue(0);
    auto xx = hx.template ReinterpretCast<uint64_t>();
    xx.SetValue(0, xv);
    xx.SetValue(1, xv);
    xx.SetValue(2, xv);
    xx.SetValue(3, xv);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    MulAddDst<float, T>(prod, hx, w, static_cast<uint64_t>(16), 2, BinaryRepeatParams{1, 1, 1, 2, 0, 1});
    PairReduceSum(pairs, prod, 1, 32, 1, 1, 8);
    PairReduceSum(acc, pairs, 2, 8, 1, 1, 1);
}

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16>
__aicore__ inline __attribute__((always_inline)) void
SubBlockGemv4<T>::RunStage5182e543(A0& x, A1& gw1, A2& gw2, A3& gb1, A4& gb2, A5& gy, A6& hx, A7& w, A8& bias,
                                   A9& prod, A10& pairs, A11& acc, A12& g, A13& sig, A14& ones, A15& out,
                                   A16& xc)
{
    SharedStage9a0a31e17c(acc, bias, gb1, gb2, gw1, gw2, hx, pairs, prod, w, x, xc);
    constexpr uint64_t mask = 0;
    SetVectorMask<float, MaskMode::NORMAL>(0, 15);
    Cast<T, float, false>(g, acc, RoundMode::CAST_RINT, mask, 2, UnaryRepeatParams{1, 1, 1, 4});
    SetVectorMask<T, MaskMode::NORMAL>(0, 15);
    Duplicate<T, false>(ones, static_cast<T>(1), mask, 1, 1, 8);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Add<T, false>(g, g, bias, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 1, 1});
    FastGate<T, 4U, false>(g, g[16], sig, out, ones);
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopyPad(gy, out, xc);
}

#endif

template <typename T>
template <class B0, class B1, class B2, class B3, class B4, class B5>
__aicore__ inline __attribute__((always_inline)) void SubBlockGemv4<T>::RunActive(B0& b1, B1& b2, B2& w1,
                                                                                  B3& w2, B4& x, B5& y)
{
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
    gx.SetGlobalBuffer((__gm__ T*)x, 4);
    gw1.SetGlobalBuffer((__gm__ T*)w1, 16);
    gw2.SetGlobalBuffer((__gm__ T*)w2, 16);
    gb1.SetGlobalBuffer((__gm__ T*)b1, 4);
    gb2.SetGlobalBuffer((__gm__ T*)b2, 4);
    gy.SetGlobalBuffer((__gm__ T*)y, 4);
    LocalTensor<T> hx(TPosition::VECIN, 0, 16);
    LocalTensor<T> w(TPosition::VECIN, 32, 32);
    LocalTensor<T> bias(TPosition::VECIN, 96, 32);
    LocalTensor<float> prod(TPosition::VECCALC, 352, 32);
    LocalTensor<float> pairs(TPosition::VECCALC, 480, 16);
    LocalTensor<float> acc(TPosition::VECCALC, 544, 40);
    LocalTensor<T> g(TPosition::VECCALC, 704, 32);
    LocalTensor<T> sig(TPosition::VECCALC, 768, 16);
    LocalTensor<T> ones(TPosition::VECCALC, 800, 16);
    LocalTensor<T> out(TPosition::VECOUT, 832, 16);
    SetAtomicNone();
    SetMaskNorm();
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    const DataCopyExtParams xc{1, 8, 0, 0, 0};
    RunStage5182e543(x, gw1, gw2, gb1, gb2, gy, hx, w, bias, prod, pairs, acc, g, sig, ones, out, xc);
#else
    RunStage7cc3bb46(x, gw1, gw2, gb1, gb2, gy, hx, w, bias, prod, pairs, acc, g, sig, ones, out);
#endif
}

template <typename T>
__aicore__ inline void SubBlockGemv4<T>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
                                             GM_ADDR y)
{
    if (GetBlockIdx() != 0U) {
        return;
    }
    RunActive(b1, b2, w1, w2, x, y);
}

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15>
__aicore__ inline __attribute__((always_inline)) void
SubBlockGemv4<T>::RunStage7cc3bb46(A0& x, A1& gw1, A2& gw2, A3& gb1, A4& gb2, A5& gy, A6& hx, A7& w, A8& bias,
                                   A9& prod, A10& pairs, A11& acc, A12& g, A13& sig, A14& ones, A15& out)
{
    const DataCopyExtParams xc{1, 8, 0, 0, 0};
    SharedStage9a0a31e17c(acc, bias, gb1, gb2, gw1, gw2, hx, pairs, prod, w, x, xc);
    constexpr uint64_t mask = 0;
    SetVectorMask<float, MaskMode::NORMAL>(0, 15);
    Cast<T, float, false>(g, acc, RoundMode::CAST_RINT, mask, 2, UnaryRepeatParams{1, 1, 1, 4});
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    SetVectorMask<T, MaskMode::NORMAL>(0, 15);
    Add<T, false>(g, g, bias, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 1, 1});
    FastGate<T, 4U, false>(g, g[16], sig, out, ones);
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopyPad(gy, out, xc);
}

#endif
