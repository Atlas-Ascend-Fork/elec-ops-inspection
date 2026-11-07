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
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations> class PairCubeCommon {
public:
    template <uint32_t I, uint32_t CurTM, class B0, class B1, class B2, class B3, class B4, class B5,
              class B6, class B7>
    __aicore__ inline __attribute__((always_inline)) void
    ComputeSplitSmall(B0& a1, B1& a2_0, B2& b1, B3& b2_0, B4& b2_1, B5& bl, B6& c0, B7& dst);
    template <uint32_t I, uint32_t CurTM, class B0, class B1, class B2, class B3, class B4, class B5>
    __aicore__ inline __attribute__((always_inline)) void ComputeSplitLarge(B0& a1, B1& b1, B2& b2_0,
                                                                            B3& b2_1, B4& bl, B5& dst);
    template <uint32_t I, uint32_t CurTM>
    __aicore__ inline __attribute__((always_inline)) void ComputeSplit(const GlobalTensor<T>& dst);
    template <uint32_t FirstTM = TM, class A0, class A1, class A2, class A3, class A4, class A5, class A6,
              class A7, class A8>
    __aicore__ inline __attribute__((always_inline)) void
    PrepareWeightsStagec2e3b721(A0& w1, A1& w2, A2& x0, A3& b1, A4& a1, A5& b2_0, A6& b2_1, A7& hap, A8& bp);
    template <uint32_t FirstTM = TM>
    __aicore__ inline void PrepareWeights(const GlobalTensor<T>& w1, const GlobalTensor<T>& w2,
                                          const GlobalTensor<T>& x0);
    template <uint32_t I, uint32_t CurTM = TM> __aicore__ inline void Prefetch(const GlobalTensor<T>& x);
    template <uint32_t I, class B0, class B1, class B2, class B3>
    __aicore__ inline __attribute__((always_inline)) void SharedStageab5685b342(B0& b1, B1& b2_0, B2& b2_1,
                                                                                B3& bl);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5,
              class A6, class A7, class A8>
    __aicore__ inline __attribute__((always_inline)) void
    ComputeStage1c2e9f71(A0& b1, A1& b2_0, A2& b2_1, A3& bl, A4& a2_0, A5& a2_1, A6& c0, A7& c1, A8& mad);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4>
    __aicore__ inline __attribute__((always_inline)) void ComputeStage06a53e30(A0& dst, A1& a2, A2& b2_0,
                                                                               A3& b2_1, A4& c);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5,
              class A6>
    __aicore__ inline __attribute__((always_inline)) void
    ComputeStage93def300(A0& b1, A1& b2_0, A2& b2_1, A3& bl, A4& a2_0, A5& c0, A6& mad);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5,
              class A6, class A7, class A8>
    __aicore__ inline __attribute__((always_inline)) void
    ComputeStage436fda8a(A0& a1, A1& b1, A2& b2_0, A3& b2_1, A4& bl, A5& a2_0, A6& a2_1, A7& c0, A8& c1);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5>
    __aicore__ inline __attribute__((always_inline)) void ComputeStagead0df8f8(A0& dst, A1& a1, A2& b1,
                                                                               A3& b2_0, A4& b2_1, A5& bl);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5,
              class A6>
    __aicore__ inline __attribute__((always_inline)) void
    ComputeStagebf1cadda(A0& dst, A1& b1, A2& b2_0, A3& b2_1, A4& bl, A5& a2_0, A6& c0);

protected:
    using Shape = PairCubeShape<T, TM, D, N, Iterations>;
    static constexpr uint32_t A = Shape::A;
    static constexpr uint32_t B = Shape::B;
    static constexpr bool SplitL0B = Shape::SplitL0B;
    static constexpr bool DoubleA2 = Shape::DoubleA2;
    static constexpr bool DoubleC = Shape::DoubleC;
    static constexpr uint32_t L1Slots = Shape::L1Slots;
};
#endif
}

namespace optiling {
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class B0, class B1, class B2, class B3, class B4, class B5, class B6,
          class B7>
__aicore__ inline __attribute__((always_inline)) void
PairCubeCommon<T, TM, D, N, Iterations>::ComputeSplitSmall(B0& a1, B1& a2_0, B2& b1, B3& b2_0, B4& b2_1,
                                                           B5& bl, B6& c0, B7& dst)
{
    if constexpr (I > 0U) {
        ::SharedPrimitive338d40bc0d<TM, D, I, CurTM>(a1, a2_0);
    }
    ComputeStagebf1cadda<I, CurTM>(dst, b1, b2_0, b2_1, bl, a2_0, c0);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class B0, class B1, class B2, class B3, class B4, class B5>
__aicore__ inline __attribute__((always_inline)) void
PairCubeCommon<T, TM, D, N, Iterations>::ComputeSplitLarge(B0& a1, B1& b1, B2& b2_0, B3& b2_1, B4& bl,
                                                           B5& dst)
{
    ComputeStagead0df8f8<I, CurTM>(dst, a1, b1, b2_0, b2_1, bl);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM>
__aicore__ inline __attribute__((always_inline)) void
PairCubeCommon<T, TM, D, N, Iterations>::ComputeSplit(const GlobalTensor<T>& dst)
{
    ::SharedComputeSplit<T, TM, D, N, A, B, I, CurTM>(*this, dst);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t FirstTM, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7,
          class A8>
__aicore__ inline __attribute__((always_inline)) void
PairCubeCommon<T, TM, D, N, Iterations>::PrepareWeightsStagec2e3b721(A0& w1, A1& w2, A2& x0, A3& b1, A4& a1,
                                                                     A5& b2_0, A6& b2_1, A7& hap, A8& bp)
{
    SharedKerneld5184a4a53<T, TM, D, N, A0, A1, A2, A3, A4, A5, A6, A7, A8>(w1, w2, x0, b1, a1, b2_0, b2_1,
                                                                            hap, bp);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t FirstTM>
__aicore__ inline void PairCubeCommon<T, TM, D, N, Iterations>::PrepareWeights(const GlobalTensor<T>& w1,
                                                                               const GlobalTensor<T>& w2,
                                                                               const GlobalTensor<T>& x0)
{
    ::SharedPrepareWeights<T, TM, D, N, A, B, FirstTM>(w1, w2, x0);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM>
__aicore__ inline void PairCubeCommon<T, TM, D, N, Iterations>::Prefetch(const GlobalTensor<T>& x)
{
    SharedKernel23bd47df61<T, D, I, CurTM, A, SplitL0B>(x);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, class B0, class B1, class B2, class B3>
__aicore__ inline __attribute__((always_inline)) void
PairCubeCommon<T, TM, D, N, Iterations>::SharedStageab5685b342(B0& b1, B1& b2_0, B2& b2_1, B3& bl)
{
    SharedKerneldc2a2bdd73<D, N, I, B0, B1, B2, B3>(b1, b2_0, b2_1, bl);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5, class A6,
          class A7, class A8>
__aicore__ inline __attribute__((always_inline)) void
PairCubeCommon<T, TM, D, N, Iterations>::ComputeStage1c2e9f71(A0& b1, A1& b2_0, A2& b2_1, A3& bl, A4& a2_0,
                                                              A5& a2_1, A6& c0, A7& c1, A8& mad)
{
    constexpr uint32_t HM = TM / 2U;
    constexpr uint32_t firstOff = (I % 2U == 1U) ? (HM * N) : 0U;
    constexpr uint32_t secondOff = (I % 2U == 1U) ? 0U : (HM * N);
    constexpr uint32_t nextB1Off = (I % 2U == 1U) ? 0U : (N * D);
    mad.m = HM;
    mad.n = N / 2U;
    mad.k = D;
    mad.cmatrixInitVal = true;
    ::SharedPrimitivea423544562<TM, N, I>(a2_0, b2_0, b2_1, c0, mad);
    if constexpr (I > 0U) {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
    }
    Mmad(c1[firstOff], a2_1, b2_0, mad);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
    Mmad(c1[firstOff + HM * N / 2U], a2_1, b2_1, mad);
    SharedStageab5685b342<I>(b1, b2_0, b2_1, bl);
    Mmad(c0[secondOff], a2_0, b2_0, mad);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
    Mmad(c0[secondOff + HM * N / 2U], a2_0, b2_1, mad);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    Mmad(c1[secondOff], a2_1, b2_0, mad);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4>
__aicore__ inline __attribute__((always_inline)) void
PairCubeCommon<T, TM, D, N, Iterations>::ComputeStage06a53e30(A0& dst, A1& a2, A2& b2_0, A3& b2_1, A4& c)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr event_t cSlot = DoubleC ? slot : EVENT_ID0;
    WaitFlag<HardEvent::MTE1_M>(slot);
    if constexpr (I < 2U) {
        SetFlag<HardEvent::M_MTE1>(slot);
    }
    if constexpr (I == 0U) {
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
    }
    if constexpr (I >= 2U) {
        WaitFlag<HardEvent::FIX_M>(cSlot);
    }
    MmadParams mad;
    mad.m = CurTM;
    mad.n = N;
    mad.k = D;
    mad.cmatrixInitVal = true;
    Mmad(c, a2, b2_0, mad);
    if constexpr (I == 0U) {
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
    }
    Mmad(c[CurTM * N], a2, b2_1, mad);
    if constexpr (I >= 2U) {
        SetFlag<HardEvent::M_MTE1>(slot);
    }
    SetFlag<HardEvent::M_FIX>(cSlot);
    ::SharedPrimitive99895aa8e1<T, TM, N, DoubleC, I, CurTM>(c, dst);
    SetFlag<HardEvent::FIX_M>(cSlot);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
__aicore__ inline __attribute__((always_inline)) void
PairCubeCommon<T, TM, D, N, Iterations>::ComputeStage93def300(A0& b1, A1& b2_0, A2& b2_1, A3& bl, A4& a2_0,
                                                              A5& c0, A6& mad)
{
    constexpr uint32_t nextB1Off = (I % 2U == 1U) ? 0U : (N * D);
    constexpr uint32_t cFirstOff = (I % 2U == 1U) ? (CurTM * N) : 0U;
    constexpr uint32_t cSecondOff = (I % 2U == 1U) ? 0U : (CurTM * N);
    mad.m = CurTM;
    mad.n = N / 2U;
    mad.k = D;
    mad.cmatrixInitVal = true;
    ::SharedPrimitiveb68d84b691<TM, N, I, CurTM>(a2_0, b2_0, b2_1, c0, mad);
    SharedStageab5685b342<I>(b1, b2_0, b2_1, bl);
    Mmad(c0[cSecondOff], a2_0, b2_0, mad);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
    Mmad(c0[cSecondOff + CurTM * N / 2U], a2_0, b2_1, mad);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5, class A6,
          class A7, class A8>
__aicore__ inline __attribute__((always_inline)) void
PairCubeCommon<T, TM, D, N, Iterations>::ComputeStage436fda8a(A0& a1, A1& b1, A2& b2_0, A3& b2_1, A4& bl,
                                                              A5& a2_0, A6& a2_1, A7& c0, A8& c1)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr uint32_t HM = TM / 2U;
    constexpr uint32_t secondOff = (I % 2U == 1U) ? 0U : (HM * N);
    ::SharedPrimitive02eeb40300<TM, D, I, CurTM>(a1, a2_0, a2_1);
    if constexpr (I > 0U) {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    }
    MmadParams mad;
    ComputeStage1c2e9f71<I, CurTM>(b1, b2_0, b2_1, bl, a2_0, a2_1, c0, c1, mad);
    Mmad(c1[secondOff + HM * N / 2U], a2_1, b2_1, mad);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    SetFlag<HardEvent::M_FIX>(EVENT_ID1);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5>
__aicore__ inline __attribute__((always_inline)) void
PairCubeCommon<T, TM, D, N, Iterations>::ComputeStagead0df8f8(A0& dst, A1& a1, A2& b1, A3& b2_0, A4& b2_1,
                                                              A5& bl)
{
    constexpr uint32_t HM = TM / 2U;
    LocalTensor<T> a2_0(TPosition::A2, 0U, HM * D);
    LocalTensor<T> a2_1(TPosition::A2, HM * D * sizeof(T), HM * D);
    LocalTensor<float> c0(TPosition::CO1, 0U, 2U * HM * N);
    LocalTensor<float> c1(TPosition::CO1, 2U * HM * N * sizeof(float), 2U * HM * N);
    ComputeStage436fda8a<I, CurTM>(a1, b1, b2_0, b2_1, bl, a2_0, a2_1, c0, c1);
    FixpipeParamsV220 fix;
    fix.nSize = 2U * N;
    fix.mSize = HM;
    fix.srcStride = HM;
    ::SharedPrimitive47ced2592b<T, N, I>(c0, dst, fix);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
    Fixpipe<T, float, CFG_ROW_MAJOR>(dst[HM * 2U * N], c1, fix);
    SetFlag<HardEvent::FIX_M>(EVENT_ID1);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
__aicore__ inline __attribute__((always_inline)) void
PairCubeCommon<T, TM, D, N, Iterations>::ComputeStagebf1cadda(A0& dst, A1& b1, A2& b2_0, A3& b2_1, A4& bl,
                                                              A5& a2_0, A6& c0)
{
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    if constexpr (I == 0U) {
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
    }
    if constexpr (I > 0U) {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    }
    MmadParams mad;
    ComputeStage93def300<I, CurTM>(b1, b2_0, b2_1, bl, a2_0, c0, mad);
    FixpipeParamsV220 fix;
    fix.nSize = 2U * N;
    fix.mSize = CurTM;
    fix.srcStride = CurTM;
    ::SharedPrimitive47ced2592b<T, N, I>(c0, dst, fix);
}

#endif

}
