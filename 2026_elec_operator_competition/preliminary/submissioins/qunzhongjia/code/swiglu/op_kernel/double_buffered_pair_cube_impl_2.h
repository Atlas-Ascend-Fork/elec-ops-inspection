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

namespace optiling {
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
#if !((defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3, uint32_t TM4, class B0, class B1, class B2,
          class B3, class B4, class B5, class B6, class B7, class B8, class B9, class B10, class B11,
          class B12, class B13, class B14, class B15, class B16, class B17>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::FinishPrepareAll(B0& a1_0, B1& a1_1, B2& a1_2, B3& a2_0,
                                                                  B4& ap0, B5& ap1, B6& ap2, B7& b1, B8& b2_0,
                                                                  B9& b2_1, B10& bp, B11& w1, B12& w2,
                                                                  B13& x0, B14& x1, B15& x2, B16& x3, B17& x4)
#endif
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
    template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3, class B0, class B1, class B2, class B3,
              class B4, class B5, class B6, class B7, class B8, class B9, class B10, class B11, class B12,
              class B13, class B14, class B15, class B16>
    __aicore__ inline
    __attribute__((always_inline)) void DoubleBufferedPairCube<T, TM, D, N, Iterations>::FinishPrepareAll(
        B0& a1_0, B1& a1_1, B2& a1_2, B3& a2_0, B4& ap0, B5& ap1, B6& ap2, B7& b1, B8& b2_0, B9& b2_1,
        B10& bp, B11& w1, B12& w2, B13& x0, B14& x1, B15& x2, B16& x3)
#endif
{
    DataCopy(a1_0, x0, ap0);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    DataCopy(b1, w1, bp);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    DataCopy(b1[N * D], w2, bp);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    PrepareAllStagecfd973b0<TM0, TM1, TM2, TM3>(x1, x2, x3, a1_0, a1_1, a1_2, b1, a2_0, b2_0, b2_1, ap1, ap2);
#else
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    DataCopy(a1_1, x1, ap1);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    if constexpr (Iterations >= 3U) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }
    PrepareAllStagee11ca2e9<TM0, TM1, TM2, TM3, TM4>(x2, x3, a1_0, a1_2, b1, a2_0, b2_0, b2_1, ap2);
#endif
    SetFlag<HardEvent::MTE1_M>(EVENT_ID3);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
#else
    if constexpr (Iterations >= 5U) {
        LocalTensor<T> a1_4(TPosition::A1, 4U * Common::A * sizeof(T), TM4 * D);
        const Nd2NzParams ap4{1, TM4, D, 0, D, TM4, 1, 0};
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        DataCopy(a1_4, x4, ap4);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    }
#endif
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
#if !((defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3, uint32_t TM4>
__aicore__ inline void DoubleBufferedPairCube<T, TM, D, N, Iterations>::PrepareAll(
    const GlobalTensor<T>& w1, const GlobalTensor<T>& w2, const GlobalTensor<T>& x0,
    const GlobalTensor<T>& x1, const GlobalTensor<T>& x2, const GlobalTensor<T>& x3,
    const GlobalTensor<T>& x4)
#endif
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
    template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3>
    __aicore__ inline void DoubleBufferedPairCube<T, TM, D, N, Iterations>::PrepareAll(
        const GlobalTensor<T>& w1, const GlobalTensor<T>& w2, const GlobalTensor<T>& x0,
        const GlobalTensor<T>& x1, const GlobalTensor<T>& x2, const GlobalTensor<T>& x3)
#endif
{
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    SetMMRowMajor();
    LocalTensor<uint64_t> unitState(TPosition::CO1, 0U, 16U);
    SetFixPipeConfig(unitState, true);
    SetFlag<HardEvent::FIX_M>(EVENT_ID7);
#else
#endif
    LocalTensor<T> a1_0(TPosition::A1, 0U, TM0 * D);
    LocalTensor<T> a1_1(TPosition::A1, A * sizeof(T), TM1 * D);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    LocalTensor<T> a1_2(TPosition::A1, 2U * A * sizeof(T), TM2 * D);
    LocalTensor<T> b1(TPosition::B1, Iterations * A * sizeof(T), B);
#else
    LocalTensor<T> a1_2(TPosition::A1, 2U * Common::A * sizeof(T), TM2 * D);
    LocalTensor<T> b1(TPosition::B1, Iterations * Common::A * sizeof(T), Common::B);
#endif
    LocalTensor<T> a2_0(TPosition::A2, 0U, TM0 * D);
    LocalTensor<T> b2_0(TPosition::B2, 0U, N * D);
    LocalTensor<T> b2_1(TPosition::B2, N * D * sizeof(T), N * D);
    const Nd2NzParams ap0{1, TM0, D, 0, D, TM0, 1, 0};
    const Nd2NzParams ap1{1, TM1, D, 0, D, TM1, 1, 0};
    const Nd2NzParams ap2{1, TM2, D, 0, D, TM2, 1, 0};
    const Nd2NzParams bp{1, N, D, 0, D, N, 1, 0};
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    FinishPrepareAll<TM0, TM1, TM2, TM3>(a1_0, a1_1, a1_2, a2_0, ap0, ap1, ap2, b1, b2_0, b2_1, bp, w1, w2,
                                         x0, x1, x2, x3);
#else
    FinishPrepareAll<TM0, TM1, TM2, TM3, TM4>(a1_0, a1_1, a1_2, a2_0, ap0, ap1, ap2, b1, b2_0, b2_1, bp, w1,
                                              w2, x0, x1, x2, x3, x4);
#endif
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, class B0, class B1, class B2, class B3>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::SharedStageab5685b342(B0& b1, B1& b2_0, B2& b2_1, B3& bl)
{
    SharedKerneldc2a2bdd73<D, N, I, B0, B1, B2, B3>(b1, b2_0, b2_1, bl);
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5, class A6,
          class A7, class A8>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::ComputeStage90cd7c22(A0& b1, A1& b2_0, A2& b2_1, A3& bl,
                                                                      A4& a2_0, A5& a2_1, A6& c0, A7& c1,
                                                                      A8& mad)
{
    constexpr uint32_t HM = TM / 2U;
    constexpr uint32_t firstOff = (I % 2U == 1U) ? (HM * N) : 0U;
    constexpr uint32_t secondOff = (I % 2U == 1U) ? 0U : (HM * N);
    constexpr uint32_t nextB1Off = (I % 2U == 1U) ? 0U : (N * D);
    mad.m = HM;
    mad.n = N / 2U;
    mad.k = D;
    mad.cmatrixInitVal = true;
    mad.unitFlag = 3;
    ::SharedPrimitivea423544562<TM, N, I>(a2_0, b2_0, b2_1, c0, mad);
    if constexpr (I > 0U) {
    }
    Mmad(c1[firstOff], a2_1, b2_0, mad);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
    Mmad(c1[firstOff + HM * N / 2U], a2_1, b2_1, mad);
    SharedStageab5685b342<I>(b1, b2_0, b2_1, bl);
    Mmad(c0[secondOff], a2_0, b2_0, mad);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
    Mmad(c0[secondOff + HM * N / 2U], a2_0, b2_1, mad);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    Mmad(c1[secondOff], a2_1, b2_0, mad);
    Mmad(c1[secondOff + HM * N / 2U], a2_1, b2_1, mad);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class B0, class B1>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::SharedStaged3e4cc9684(B0& a1, B1& a2)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    if constexpr (I > 0U) {
        constexpr event_t mte2Ev = (I == 1U) ? EVENT_ID1 : ((I == 2U) ? EVENT_ID0 : EVENT_ID2);
        WaitFlag<HardEvent::MTE2_MTE1>(mte2Ev);
        ::SharedLoadTileA<TM, D, I, CurTM>(a1, a2);
        SetFlag<HardEvent::MTE1_M>(slot);
    }
#else
    if constexpr (I > 0U) {
        constexpr event_t mte2Ev = (I == 1U) ? EVENT_ID1 : ((I == 2U) ? EVENT_ID0 : EVENT_ID2);
        WaitFlag<HardEvent::MTE2_MTE1>(mte2Ev);
        ::SharedLoadTileA<TM, D, I, CurTM, true>(a1, a2);
    }
#endif
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::ComputeStagef0f64ea7(A0& a1, A1& a2, A2& b2_0, A3& b2_1,
                                                                      A4& c)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    SharedStaged3e4cc9684<I, CurTM>(a1, a2);
    WaitFlag<HardEvent::MTE1_M>(slot);
    if constexpr (I == 0U) {
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
    }
    if constexpr (I >= 2U) {
    }
    MmadParams mad;
    mad.m = CurTM;
    mad.n = N;
    mad.k = D;
    mad.cmatrixInitVal = true;
    mad.unitFlag = 3;
    Mmad(c, a2, b2_0, mad);
    if constexpr (I == 0U) {
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
    }
    Mmad(c[CurTM * N], a2, b2_1, mad);
    SetFlag<HardEvent::M_MTE1>(slot);
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::ComputeStage56696d27(A0& b1, A1& b2_0, A2& b2_1, A3& bl,
                                                                      A4& a2_0, A5& c0, A6& mad)
{
    constexpr uint32_t nextB1Off = (I % 2U == 1U) ? 0U : (N * D);
    constexpr uint32_t cFirstOff = (I % 2U == 1U) ? (CurTM * N) : 0U;
    constexpr uint32_t cSecondOff = (I % 2U == 1U) ? 0U : (CurTM * N);
    mad.m = CurTM;
    mad.n = N / 2U;
    mad.k = D;
    mad.cmatrixInitVal = true;
    mad.unitFlag = 3;
    ::SharedPrimitiveb68d84b691<TM, N, I, CurTM>(a2_0, b2_0, b2_1, c0, mad);
    SharedStageab5685b342<I>(b1, b2_0, b2_1, bl);
    Mmad(c0[cSecondOff], a2_0, b2_0, mad);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
    Mmad(c0[cSecondOff + CurTM * N / 2U], a2_0, b2_1, mad);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5, class A6,
          class A7, class A8>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::ComputeStagecf6a2fb3(A0& a1, A1& b1, A2& b2_0, A3& b2_1,
                                                                      A4& bl, A5& a2_0, A6& a2_1, A7& c0,
                                                                      A8& c1)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr uint32_t HM = TM / 2U;
    ::SharedPrimitive02eeb40300<TM, D, I, CurTM>(a1, a2_0, a2_1);
    if constexpr (I > 0U) {
    }
    MmadParams mad;
    ComputeStage90cd7c22<I, CurTM>(b1, b2_0, b2_1, bl, a2_0, a2_1, c0, c1, mad);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class B0, class B1>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::SharedStagec7aaa4f442(B0& a1, B1& a2_0)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    if constexpr (I > 0U) {
        ::SharedPrimitive338d40bc0d<TM, D, I, CurTM>(a1, a2_0);
    }
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::ComputeStageea253d2a(A0& a1, A1& b1, A2& b2_0, A3& b2_1,
                                                                      A4& bl, A5& a2_0, A6& c0)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    SharedStagec7aaa4f442<I, CurTM>(a1, a2_0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    if constexpr (I == 0U) {
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
    }
    if constexpr (I > 0U) {
    }
    MmadParams mad;
    ComputeStage56696d27<I, CurTM>(b1, b2_0, b2_1, bl, a2_0, c0, mad);
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::ComputeStage5944224d(A0& dst)
{
    LocalTensor<T> a1(TPosition::A1, I * A * sizeof(T), CurTM * D);
    LocalTensor<T> a2(TPosition::A2, (I % 2U) * A * sizeof(T), CurTM * D);
    LocalTensor<T> b2_0(TPosition::B2, 0U, N * D);
    LocalTensor<T> b2_1(TPosition::B2, N * D * sizeof(T), N * D);
    LocalTensor<float> c(TPosition::CO1, (I % 2U) * 2U * TM * N * sizeof(float), 2U * CurTM * N);
    ComputeStagef0f64ea7<I, CurTM>(a1, a2, b2_0, b2_1, c);
    FixpipeParamsV220 fix;
    fix.nSize = 2U * N;
    fix.mSize = CurTM;
    fix.srcStride = CurTM;
    fix.dstStride = 2U * N;
    fix.quantPre = QuantMode_t::F322F16;
    fix.unitFlag = 3;
    Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c, fix);
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::ComputeStage69b25c34(A0& dst, A1& a1, A2& b1, A3& b2_0,
                                                                      A4& b2_1, A5& bl)
{
    constexpr uint32_t HM = TM / 2U;
    LocalTensor<T> a2_0(TPosition::A2, 0U, HM * D);
    LocalTensor<T> a2_1(TPosition::A2, HM * D * sizeof(T), HM * D);
    LocalTensor<float> c0(TPosition::CO1, 0U, 2U * HM * N);
    LocalTensor<float> c1(TPosition::CO1, 2U * HM * N * sizeof(float), 2U * HM * N);
    ComputeStagecf6a2fb3<I, CurTM>(a1, b1, b2_0, b2_1, bl, a2_0, a2_1, c0, c1);
    FixpipeParamsV220 fix;
    fix.nSize = 2U * N;
    fix.mSize = HM;
    fix.srcStride = HM;
    ::SharedPrimitive2f7c32ba02<T, N, I>(c0, dst, fix);
    Fixpipe<T, float, CFG_ROW_MAJOR>(dst[HM * 2U * N], c1, fix);
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class B0, class B1, class B2, class B3, class B4, class B5, class B6,
          class B7>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::ComputeSplitSmall(B0& a1, B1& a2_0, B2& b1, B3& b2_0,
                                                                   B4& b2_1, B5& bl, B6& c0, B7& dst)
{
    ComputeStageea253d2a<I, CurTM>(a1, b1, b2_0, b2_1, bl, a2_0, c0);
    FixpipeParamsV220 fix;
    fix.nSize = 2U * N;
    fix.mSize = CurTM;
    fix.srcStride = CurTM;
    ::SharedPrimitive2f7c32ba02<T, N, I>(c0, dst, fix);
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class B0, class B1, class B2, class B3, class B4, class B5>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::ComputeSplitLarge(B0& a1, B1& b1, B2& b2_0, B3& b2_1, B4& bl,
                                                                   B5& dst)
{
    ComputeStage69b25c34<I, CurTM>(dst, a1, b1, b2_0, b2_1, bl);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3, uint32_t TM4, class A0, class A1, class A2,
          class A3, class A4, class A5, class A6, class A7, class A8>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::PrepareAllStagee11ca2e9(A0& x2, A1& x3, A2& a1_0, A3& a1_2,
                                                                         A4& b1, A5& a2_0, A6& b2_0, A7& b2_1,
                                                                         A8& ap2)
{
    ::SharedPrimitive13dcc0f596<D, TM0>(a1_0, a2_0);
    if constexpr (Iterations >= 3U) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        DataCopy(a1_2, x2, ap2);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    }
    ::SharedWaitWeights<Iterations>();
    LoadData2dParams bl;
    bl.repeatTimes = (N * D) / 256U;
    bl.srcStride = 1;
    LoadData(b2_0, b1, bl);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
    if constexpr (Iterations >= 4U) {
        LocalTensor<T> a1_3(TPosition::A1, 3U * Common::A * sizeof(T), TM3 * D);
        const Nd2NzParams ap3{1, TM3, D, 0, D, TM3, 1, 0};
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        DataCopy(a1_3, x3, ap3);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    }
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    if constexpr (Iterations >= 5U) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
    }
    LoadData(b2_1, b1[N * D], bl);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::ComputeStageb52739cb(A0& dst)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    LocalTensor<T> a1(TPosition::A1, I * Common::A * sizeof(T), CurTM * D);
    LocalTensor<T> a2(TPosition::A2, I * Common::A * sizeof(T), CurTM * D);
    LocalTensor<T> b2_0(TPosition::B2, 0U, N * D);
    LocalTensor<T> b2_1(TPosition::B2, N * D * sizeof(T), N * D);
    LocalTensor<float> c(TPosition::CO1, (I % 2U) * 2U * TM * N * sizeof(float), 2U * TM * N);
    SharedStaged3e4cc9684<I, CurTM>(a1, a2);
    Common::template ComputeStage06a53e30<I, CurTM>(dst, a2, b2_0, b2_1, c);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::ComputeStage21e20e1c(A0& dst, A1& a1, A2& b2_0, A3& b2_1)
{
    LocalTensor<T> a2_2(TPosition::A2, 80U * D * sizeof(T), 48U * D);
    LocalTensor<float> c2(TPosition::CO1, 2U * 80U * N * sizeof(float), 2U * 48U * N);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    ::SharedPrimitivedc71102ef9<D>(a1, a2_2);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    ::SharedPrimitived55642c3a1<D, N>(a2_2, b2_0, b2_1, c2);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    ::SharedPrimitive5ed408cbed<T, N>(c2, dst);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::ComputeStageeeb2fd57(A0& dst, A1& a1, A2& b2_0, A3& b2_1)
{
    LocalTensor<T> a2_1(TPosition::A2, 32U * D * sizeof(T), 48U * D);
    LocalTensor<float> c1(TPosition::CO1, 2U * 32U * N * sizeof(float), 2U * 48U * N);
    ::SharedPrimitive754371e923<D, N>(a1, a2_1, b2_0, b2_1, c1);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    ::SharedPrimitive0689c680de<T, N>(c1, dst);
}

#endif

}
