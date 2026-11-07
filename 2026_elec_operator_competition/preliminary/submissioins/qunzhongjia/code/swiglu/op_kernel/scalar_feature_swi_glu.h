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

template <typename T> class ScalarFeatureSwiGlu {
public:
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18>
    __aicore__ inline __attribute__((always_inline)) void
    RunStaged9e73839(A0& td, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& gy, A7& hx, A8& params, A9& fx,
                     A10& fg, A11& fu, A12& g, A13& u, A14& sig, A15& ones, A16& out, A17& pc, A18& pp);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               const SwiGluTinyTilingData* td);
};

template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18>
__aicore__ inline __attribute__((always_inline)) void
ScalarFeatureSwiGlu<T>::RunStaged9e73839(A0& td, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& gy, A7& hx,
                                         A8& params, A9& fx, A10& fg, A11& fu, A12& g, A13& u, A14& sig,
                                         A15& ones, A16& out, A17& pc, A18& pp)
{
    DataCopyPad(params, gw1, pc, pp);
    DataCopyPad(params[16], gw2, pc, pp);
    DataCopyPad(params[32], gb1, pc, pp);
    DataCopyPad(params[48], gb2, pc, pp);
    ::SharedSynccc9d9b6d<HardEvent::MTE2_S>();
    const float a = static_cast<float>(params.GetValue(0));
    const float b = static_cast<float>(params.GetValue(16));
    const T c = params.GetValue(32);
    const T e = params.GetValue(48);
    Duplicate(ones, static_cast<T>(1), 256);
    for (uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * 256U; start < td->m;
         start += static_cast<uint64_t>(GetBlockNum()) * 256U) {
        const uint32_t n = td->m - start < 256U ? static_cast<uint32_t>(td->m - start) : 256U;
        DataCopyExtParams cp{1, n * 2U, 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>((16U - n % 16U) % 16U), static_cast<T>(0)};
        DataCopyPad(hx, gx[start], cp, pad);
        ::SharedSynccc9d9b6d<HardEvent::MTE2_V>();
        Cast(fx, hx, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        Muls(fg, fx, a, n);
        Muls(fu, fx, b, n);
        PipeBarrier<PIPE_V>();
        Cast(g, fg, RoundMode::CAST_RINT, n);
        Cast(u, fu, RoundMode::CAST_RINT, n);
        PipeBarrier<PIPE_V>();
        Adds(g, g, c, n);
        Adds(u, u, e, n);
        PipeBarrier<PIPE_V>();
        GateCounter<T>(g, u, sig, out, ones, n);
        ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
        DataCopyPad(gy[start], out, cp);
        ::SharedSynccc9d9b6d<HardEvent::MTE3_MTE2>();
    }
}

template <typename T>
__aicore__ inline void ScalarFeatureSwiGlu<T>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
                                                   GM_ADDR y, const SwiGluTinyTilingData* td)
{
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
    gx.SetGlobalBuffer((__gm__ T*)x, td->m);
    gy.SetGlobalBuffer((__gm__ T*)y, td->m);
    gw1.SetGlobalBuffer((__gm__ T*)w1, 1);
    gw2.SetGlobalBuffer((__gm__ T*)w2, 1);
    gb1.SetGlobalBuffer((__gm__ T*)b1, 1);
    gb2.SetGlobalBuffer((__gm__ T*)b2, 1);
    LocalTensor<T> hx(TPosition::VECIN, 0, 256);
    LocalTensor<T> params(TPosition::VECIN, 512, 64);
    LocalTensor<float> fx(TPosition::VECCALC, 640, 256);
    LocalTensor<float> fg(TPosition::VECCALC, 1664, 256);
    LocalTensor<float> fu(TPosition::VECCALC, 2688, 256);
    LocalTensor<T> g(TPosition::VECCALC, 3712, 256);
    LocalTensor<T> u(TPosition::VECCALC, 4224, 256);
    LocalTensor<T> sig(TPosition::VECCALC, 4736, 256);
    LocalTensor<T> ones(TPosition::VECCALC, 5248, 256);
    LocalTensor<T> out(TPosition::VECOUT, 5760, 256);
    SetAtomicNone();
    SetMaskNorm();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    DataCopyExtParams pc{1, 2, 0, 0, 0};
    DataCopyPadExtParams<T> pp{true, 0, 15, static_cast<T>(0)};
    RunStaged9e73839(td, gx, gw1, gw2, gb1, gb2, gy, hx, params, fx, fg, fu, g, u, sig, ones, out, pc, pp);
}
