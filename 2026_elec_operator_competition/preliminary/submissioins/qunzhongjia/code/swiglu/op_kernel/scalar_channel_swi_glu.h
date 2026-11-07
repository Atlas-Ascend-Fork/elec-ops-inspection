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

template <typename T> class ScalarChannelSwiGlu {
public:
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
    __aicore__ inline __attribute__((always_inline)) void
    RunStagedf927ab1(A0& b1, A1& w2, A2& b2, A3& gx, A4& gw1, A5& gw2, A6& gb1, A7& gb2, A8& gy);
    template <class B0, class B1, class B2, class B3, class B4, class B5>
    __aicore__ inline __attribute__((always_inline)) void RunActive(B0& b1, B1& b2, B2& w1, B3& w2, B4& x,
                                                                    B5& y);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y);
};

template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
__aicore__ inline __attribute__((always_inline)) void
ScalarChannelSwiGlu<T>::RunStagedf927ab1(A0& b1, A1& w2, A2& b2, A3& gx, A4& gw1, A5& gw2, A6& gb1, A7& gb2,
                                         A8& gy)
{
    gw2.SetGlobalBuffer((__gm__ T*)w2, 1);
    gb1.SetGlobalBuffer((__gm__ T*)b1, 1);
    gb2.SetGlobalBuffer((__gm__ T*)b2, 1);
    LocalTensor<T> hx(TPosition::VECIN, 0, 16);
    LocalTensor<T> w(TPosition::VECIN, 32, 32);
    LocalTensor<T> bias(TPosition::VECIN, 96, 32);
    LocalTensor<T> g(TPosition::VECCALC, 160, 32);
    LocalTensor<T> sig(TPosition::VECCALC, 224, 16);
    LocalTensor<T> ones(TPosition::VECCALC, 256, 16);
    LocalTensor<T> out(TPosition::VECOUT, 288, 16);
    SetAtomicNone();
    SetMaskNorm();
    SetVectorMask<T, MaskMode::NORMAL>(0, 1);
    const DataCopyExtParams rc{1, 2, 0, 0, 0};
    const DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
    const T xv = gx.GetValue(0);
    const T wv1 = gw1.GetValue(0);
    const T wv2 = gw2.GetValue(0);
    const T bv1 = gb1.GetValue(0);
    const T bv2 = gb2.GetValue(0);
    hx.SetValue(0, xv);
    w.SetValue(0, wv1);
    w.SetValue(16, wv2);
    bias.SetValue(0, bv1);
    bias.SetValue(16, bv2);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    constexpr uint64_t mask = 0;
    Mul<T, false>(g, hx, w, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 0, 1});
    Add<T, false>(g, g, bias, mask, 2, BinaryRepeatParams{1, 1, 1, 1, 1, 1});
    FastGate<T, 1U, false>(g, g[16], sig, out, ones);
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopyPad(gy, out, rc);
}

template <typename T>
template <class B0, class B1, class B2, class B3, class B4, class B5>
__aicore__ inline __attribute__((always_inline)) void
ScalarChannelSwiGlu<T>::RunActive(B0& b1, B1& b2, B2& w1, B3& w2, B4& x, B5& y)
{
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
    gx.SetGlobalBuffer((__gm__ T*)x, 1);
    gy.SetGlobalBuffer((__gm__ T*)y, 1);
    gw1.SetGlobalBuffer((__gm__ T*)w1, 1);
    RunStagedf927ab1(b1, w2, b2, gx, gw1, gw2, gb1, gb2, gy);
}

template <typename T>
__aicore__ inline void ScalarChannelSwiGlu<T>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
                                                   GM_ADDR y)
{
    if (GetBlockIdx() != 0U) {
        return;
    }
    RunActive(b1, b2, w1, w2, x, y);
}
