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

template <typename T, uint32_t FixedD, uint32_t DynamicMaxK = 256U> class SmallMVectorSwiGlu {
public:
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24,
              class A25, class A26, class A27, class A28>
    __aicore__ inline __attribute__((always_inline)) void
    RunStage44bca580(A0& d, A1& ka, A2& columns, A3& gx, A4& gw1, A5& gw2, A6& gb1, A7& gb2, A8& gy, A9& hw1,
                     A10& hw2, A11& hx, A12& hb1, A13& hb2, A14& fw1, A15& fw2, A16& pg, A17& pu, A18& fx,
                     A19& ag, A20& au, A21& sg, A22& su, A23& g, A24& u, A25& sig, A26& ones, A27& out,
                     A28& task);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13>
    __aicore__ inline __attribute__((always_inline)) void
    RunStagede391c17(A0& w1, A1& b1, A2& w2, A3& b2, A4& td, A5& d, A6& ka, A7& columns, A8& gx, A9& gw1,
                     A10& gw2, A11& gb1, A12& gb2, A13& gy);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               const SwiGluTinyTilingData* td);

private:
    template <bool First>
    __aicore__ inline void Dot(const LocalTensor<float>& ag, const LocalTensor<float>& au,
                               const LocalTensor<float>& sg, const LocalTensor<float>& su,
                               const LocalTensor<float>& pg, const LocalTensor<float>& pu,
                               const LocalTensor<float>& fx, const LocalTensor<float>& fw1,
                               const LocalTensor<float>& fw2, uint32_t ka, uint32_t k, uint32_t cols);
};

template <typename T, uint32_t FixedD, uint32_t DynamicMaxK>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18,
          class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27,
          class A28>
__aicore__ inline __attribute__((always_inline)) void
SmallMVectorSwiGlu<T, FixedD, DynamicMaxK>::RunStage44bca580(A0& d, A1& ka, A2& columns, A3& gx, A4& gw1,
                                                             A5& gw2, A6& gb1, A7& gb2, A8& gy, A9& hw1,
                                                             A10& hw2, A11& hx, A12& hb1, A13& hb2, A14& fw1,
                                                             A15& fw2, A16& pg, A17& pu, A18& fx, A19& ag,
                                                             A20& au, A21& sg, A22& su, A23& g, A24& u,
                                                             A25& sig, A26& ones, A27& out, A28& task)
{
    const uint64_t row = task / columns;
    const uint32_t col = task % columns * 16U;
    const uint32_t cols = d - col < 16U ? d - col : 16U;
    DataCopyExtParams wc{static_cast<uint16_t>(cols), d * 2U, 0, 0, 0};
    DataCopyExtParams xc{1, d * 2U, 0, 0, 0};
    DataCopyExtParams bc{1, cols * 2U, 0, 0, 0};
    DataCopyPadExtParams<T> wp{true, 0, static_cast<uint8_t>(ka - d), static_cast<T>(0)};
    DataCopyPadExtParams<T> bp{true, 0, static_cast<uint8_t>(16U - cols), static_cast<T>(0)};
    DataCopyPad(hw1, gw1[static_cast<uint64_t>(col) * d], wc, wp);
    DataCopyPad(hw2, gw2[static_cast<uint64_t>(col) * d], wc, wp);
    DataCopyPad(hx, gx[row * d], xc, wp);
    DataCopyPad(hb1, gb1[col], bc, bp);
    DataCopyPad(hb2, gb2[col], bc, bp);
    ::SharedSynccc9d9b6d<HardEvent::MTE2_V>();
    Cast(fw1, hw1, RoundMode::CAST_NONE, cols * ka);
    Cast(fw2, hw2, RoundMode::CAST_NONE, cols * ka);
    Cast(fx, hx, RoundMode::CAST_NONE, ka);
    PipeBarrier<PIPE_V>();
    Dot<true>(ag, au, sg, su, pg, pu, fx, fw1, fw2, ka, 0, cols);
    for (uint32_t k = 64U; k < ka; k += 64U) {
        Dot<false>(ag, au, sg, su, pg, pu, fx, fw1, fw2, ka, k, cols);
    }
    Cast(g, ag, RoundMode::CAST_RINT, cols);
    Cast(u, au, RoundMode::CAST_RINT, cols);
    PipeBarrier<PIPE_V>();
    Add(g, g, hb1, cols);
    Add(u, u, hb2, cols);
    PipeBarrier<PIPE_V>();
    GateCounter<T>(g, u, sig, out, ones, cols);
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopyPad(gy[row * d + col], out, bc);
    ::SharedSynccc9d9b6d<HardEvent::MTE3_MTE2>();
}

template <typename T, uint32_t FixedD, uint32_t DynamicMaxK>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13>
__aicore__ inline __attribute__((always_inline)) void
SmallMVectorSwiGlu<T, FixedD, DynamicMaxK>::RunStagede391c17(A0& w1, A1& b1, A2& w2, A3& b2, A4& td, A5& d,
                                                             A6& ka, A7& columns, A8& gx, A9& gw1, A10& gw2,
                                                             A11& gb1, A12& gb2, A13& gy)
{
    constexpr uint32_t maxK = FixedD == 0 ? DynamicMaxK : FixedD;
    constexpr uint32_t wElems = 16U * maxK;
    constexpr uint32_t fBase = 4U * wElems + 2U * maxK + 64U;
    gw1.SetGlobalBuffer((__gm__ T*)w1, d * d);
    gw2.SetGlobalBuffer((__gm__ T*)w2, d * d);
    gb1.SetGlobalBuffer((__gm__ T*)b1, d);
    gb2.SetGlobalBuffer((__gm__ T*)b2, d);
    LocalTensor<T> hw1(TPosition::VECIN, 0, wElems);
    LocalTensor<T> hw2(TPosition::VECIN, 2U * wElems, wElems);
    LocalTensor<T> hx(TPosition::VECIN, 4U * wElems, maxK);
    LocalTensor<T> hb1(TPosition::VECIN, 4U * wElems + 2U * maxK, 16);
    LocalTensor<T> hb2(TPosition::VECIN, 4U * wElems + 2U * maxK + 32U, 16);
    LocalTensor<float> fw1(TPosition::VECCALC, fBase, wElems);
    LocalTensor<float> fw2(TPosition::VECCALC, fBase + 4U * wElems, wElems);
    LocalTensor<float> pg(TPosition::VECCALC, fBase + 8U * wElems, wElems);
    LocalTensor<float> pu(TPosition::VECCALC, fBase + 12U * wElems, wElems);
    constexpr uint32_t xBase = fBase + 16U * wElems;
    LocalTensor<float> fx(TPosition::VECCALC, xBase, maxK);
    constexpr uint32_t aBase = xBase + 4U * maxK;
    LocalTensor<float> ag(TPosition::VECCALC, aBase, 16);
    LocalTensor<float> au(TPosition::VECCALC, aBase + 64U, 16);
    LocalTensor<float> sg(TPosition::VECCALC, aBase + 128U, 16);
    LocalTensor<float> su(TPosition::VECCALC, aBase + 192U, 16);
    constexpr uint32_t hBase = aBase + 256U;
    LocalTensor<T> g(TPosition::VECCALC, hBase, 16);
    LocalTensor<T> u(TPosition::VECCALC, hBase + 32U, 16);
    LocalTensor<T> sig(TPosition::VECCALC, hBase + 64U, 16);
    LocalTensor<T> ones(TPosition::VECCALC, hBase + 96U, 16);
    LocalTensor<T> out(TPosition::VECOUT, hBase + 128U, 16);
    SetAtomicNone();
    SetMaskNorm();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    Duplicate(ones, static_cast<T>(1), 16);
    for (uint64_t task = GetBlockIdx(); task < td->m * columns; task += GetBlockNum()) {
        RunStage44bca580(d, ka, columns, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hx, hb1, hb2, fw1, fw2, pg, pu,
                         fx, ag, au, sg, su, g, u, sig, ones, out, task);
    }
}

template <typename T, uint32_t FixedD, uint32_t DynamicMaxK>
__aicore__ inline void SmallMVectorSwiGlu<T, FixedD, DynamicMaxK>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1,
                                                                       GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                                                                       const SwiGluTinyTilingData* td)
{
    constexpr uint32_t maxK = FixedD == 0 ? DynamicMaxK : FixedD;
    constexpr uint32_t wElems = 16U * maxK;
    constexpr uint32_t fBase = 4U * wElems + 2U * maxK + 64U;
    const uint32_t d = FixedD != 0 ? FixedD : static_cast<uint32_t>(td->d);
    const uint32_t ka = (d + 15U) / 16U * 16U;
    const uint32_t columns = (d + 15U) / 16U;
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
    gx.SetGlobalBuffer((__gm__ T*)x, td->m * d);
    gy.SetGlobalBuffer((__gm__ T*)y, td->m * d);
    RunStagede391c17(w1, b1, w2, b2, td, d, ka, columns, gx, gw1, gw2, gb1, gb2, gy);
}

template <typename T, uint32_t FixedD, uint32_t DynamicMaxK>
template <bool First>
__aicore__ inline void SmallMVectorSwiGlu<T, FixedD, DynamicMaxK>::Dot(
    const LocalTensor<float>& ag, const LocalTensor<float>& au, const LocalTensor<float>& sg,
    const LocalTensor<float>& su, const LocalTensor<float>& pg, const LocalTensor<float>& pu,
    const LocalTensor<float>& fx, const LocalTensor<float>& fw1, const LocalTensor<float>& fw2, uint32_t ka,
    uint32_t k, uint32_t cols)
{
    SharedKernel0c144b8d11<First>(ag, au, sg, su, pg, pu, fx, fw1, fw2, ka, k, cols);
}
