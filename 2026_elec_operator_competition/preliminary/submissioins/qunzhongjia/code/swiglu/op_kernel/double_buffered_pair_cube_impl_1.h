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
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t TM, uint32_t D, uint32_t N>
template <class B0, class B1, class B2>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N>::SharedStageebbd2e0f1a(B0& w1, B1& w2, B2& x0)
{
    LocalTensor<T> b1(TPosition::B1, 2U * A * sizeof(T), B);
    LocalTensor<T> b2(TPosition::B2, 0, B);
    LocalTensor<T> a1(TPosition::A1, 0, A);
    const Nd2NzParams bp{1, N, D, 0, D, 2U * N, 1, 0};
    const Nd2NzParams ap{1, TM, D, 0, D, TM, 1, 0};
    DataCopy(b1, w1, bp);
    DataCopy(b1[N * 16U], w2, bp);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    DataCopy(a1, x0, ap);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    LoadData2dParams bl;
    bl.repeatTimes = B / 256U;
    bl.srcStride = 1;
    LoadData(b2, b1, bl);
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
    (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t TM, uint32_t D, uint32_t N>
__aicore__ inline void DoubleBufferedPairCube<T, TM, D, N>::PrepareWeights(const GlobalTensor<T>& w1,
                                                                           const GlobalTensor<T>& w2,
                                                                           const GlobalTensor<T>& x0)
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
    template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
    template <uint32_t FirstTM>
    __aicore__
    inline void DoubleBufferedPairCube<T, TM, D, N, Iterations>::PrepareWeights(const GlobalTensor<T>& w1,
                                                                                const GlobalTensor<T>& w2,
                                                                                const GlobalTensor<T>& x0)
#endif
{
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
    SetMMRowMajor();
    LocalTensor<uint64_t> unitState(TPosition::CO1, 0U, 16U);
    SetFixPipeConfig(unitState, true);
    SetFlag<HardEvent::FIX_M>(EVENT_ID7);
    ::SharedPrepareWeights<T, TM, D, N, A, B, FirstTM>(w1, w2, x0);
#else
    SharedStageebbd2e0f1a(w1, w2, x0);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
#else
// The first A load and its MTE1_M fence also order the preceding resident B load.
#endif
#endif
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
    (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t TM, uint32_t D, uint32_t N>
template <uint32_t I, uint32_t CurTM>
__aicore__ inline void DoubleBufferedPairCube<T, TM, D, N>::Prefetch(const GlobalTensor<T>& x)
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
    template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
    template <uint32_t I, uint32_t CurTM>
    __aicore__ inline void DoubleBufferedPairCube<T, TM, D, N, Iterations>::Prefetch(const GlobalTensor<T>& x)
#endif
{
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    if constexpr (I >= 2U) {
        WaitFlag<HardEvent::MTE1_MTE2>(slot);
    }
    LocalTensor<T> a1(TPosition::A1, (I % 2U) * A * sizeof(T), CurTM * D);
    const Nd2NzParams ap{1, CurTM, D, 0, D, CurTM, 1, 0};
    DataCopy(a1, x, ap);
    SetFlag<HardEvent::MTE2_MTE1>(slot);
#else
    SharedKernel23bd47df61<T, D, I, CurTM, A, SplitL0B>(x);
#endif
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t TM, uint32_t D, uint32_t N>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N>::ComputeStage6e2aa2f7(A0& a1, A1& b2, A2& a2_0, A3& a2_1, A4& c0, A5& c1)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr event_t cSlot = DoubleC ? slot : EVENT_ID0;
    constexpr uint32_t HM = CurTM / 2U;
    WaitFlag<HardEvent::MTE2_MTE1>(slot);
    if constexpr (I > 0U) {
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    }
    LoadData2dParams al;
    al.repeatTimes = D / 16U;
    al.srcStride = CurTM / 16U;
#pragma unroll

    for (uint32_t mb = 0; mb < HM / 16U; ++mb) {
        LoadData(a2_0[mb * D * 16U], a1[mb * 256U], al);
    }
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    if constexpr (I > 0U) {
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    }
#pragma unroll

    for (uint32_t mb = 0; mb < HM / 16U; ++mb) {
        LoadData(a2_1[mb * D * 16U], a1[(mb + HM / 16U) * 256U], al);
    }
    SetFlag<HardEvent::MTE1_MTE2>(slot);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    if constexpr (I >= 2U) {
        WaitFlag<HardEvent::FIX_M>(cSlot);
    }
    MmadParams mad;
    mad.m = HM;
    mad.n = 2U * N;
    mad.k = D;
    mad.cmatrixInitVal = true;
    Mmad(c0, a2_0, b2, mad);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
    Mmad(c1, a2_1, b2, mad);
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t TM, uint32_t D, uint32_t N>
template <uint32_t I, uint32_t CurTM, class B0, class B1, class B2>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N>::SharedStage69c5fc7c97(B0& c0, B1& c1, B2& dst)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr event_t cSlot = DoubleC ? slot : EVENT_ID0;
    constexpr uint32_t HM = CurTM / 2U;
    FixpipeParamsV220 fix;
    fix.nSize = 2U * N;
    fix.mSize = HM;
    fix.srcStride = HM;
    fix.dstStride = 2U * N;
    fix.quantPre = QuantMode_t::F322F16;
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c0, fix);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
    Fixpipe<T, float, CFG_ROW_MAJOR>(dst[HM * 2U * N], c1, fix);
    SetFlag<HardEvent::FIX_M>(cSlot);
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T, uint32_t TM, uint32_t D, uint32_t N>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N>::ComputeStageede6278d(A0& dst, A1& a1, A2& b2)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr event_t cSlot = DoubleC ? slot : EVENT_ID0;
    constexpr uint32_t HM = CurTM / 2U;
    LocalTensor<T> a2_0(TPosition::A2, 0U, HM * D);
    LocalTensor<T> a2_1(TPosition::A2, HM * D * sizeof(T), HM * D);
    LocalTensor<float> c0(TPosition::CO1, (I % 2U) * 2U * TM * N * sizeof(float), 2U * HM * N);
    LocalTensor<float> c1(TPosition::CO1, ((I % 2U) * 2U * TM * N + 2U * HM * N) * sizeof(float),
                          2U * HM * N);
    ComputeStage6e2aa2f7<I, CurTM>(a1, b2, a2_0, a2_1, c0, c1);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    SetFlag<HardEvent::M_FIX>(EVENT_ID1);
    if constexpr (I >= 4U) {
        constexpr uint32_t cid = (I % 4U) == 0 ? 0x8 : ((I % 4U) == 1 ? 0x9 : ((I % 4U) == 2 ? 0xC : 0xF));
        CrossCoreWaitFlag(cid);
    }
    SharedStage69c5fc7c97<I, CurTM>(c0, c1, dst);
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t TM, uint32_t D, uint32_t N>
template <uint32_t I, uint32_t CurTM, class B0, class B1, class B2, class B3>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N>::SharedStage448bfd918a(B0& a1, B1& a2, B2& b2, B3& c)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr event_t cSlot = DoubleC ? slot : EVENT_ID0;
    WaitFlag<HardEvent::MTE2_MTE1>(slot);
    ::SharedLoadTileA<TM, D, I, CurTM>(a1, a2);
    SetFlag<HardEvent::MTE1_MTE2>(slot);
    SetFlag<HardEvent::MTE1_M>(slot);
    WaitFlag<HardEvent::MTE1_M>(slot);
    if constexpr (DoubleC ? (I >= 2U) : (I > 0U)) {
        WaitFlag<HardEvent::FIX_M>(cSlot);
    }
    MmadParams mad;
    mad.m = CurTM;
    mad.n = 2U * N;
    mad.k = D;
    mad.cmatrixInitVal = true;
    Mmad(c, a2, b2, mad);
    SetFlag<HardEvent::M_MTE1>(slot);
    SetFlag<HardEvent::M_FIX>(cSlot);
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T, uint32_t TM, uint32_t D, uint32_t N>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N>::ComputeStage5d29390b(A0& dst, A1& a1, A2& b2, A3& a2)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr event_t cSlot = DoubleC ? slot : EVENT_ID0;
    LocalTensor<float> c(TPosition::CO1, DoubleC ? (I % 2U) * 2U * TM * N * sizeof(float) : 0U,
                         2U * CurTM * N);
    SharedStage448bfd918a<I, CurTM>(a1, a2, b2, c);
    ::SharedPrimitive99895aa8e1<T, TM, N, DoubleC, I, CurTM>(c, dst);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM>
__aicore__ inline void DoubleBufferedPairCube<T, TM, D, N, Iterations>::Compute(const GlobalTensor<T>& dst)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr event_t cSlot = Common::DoubleC ? slot : EVENT_ID0;
    if constexpr (Common::SplitL0B) {
        Common::template ComputeSplit<I, CurTM>(dst);
    } else if constexpr (TM == 48U) {
        ComputeStage584a2d8d<I, CurTM>(dst);
    } else {
        ComputeStageb52739cb<I, CurTM>(dst);
    }
}
#endif
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
    (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t TM, uint32_t D, uint32_t N>
template <uint32_t I, uint32_t CurTM>
__aicore__ inline void DoubleBufferedPairCube<T, TM, D, N>::Compute(const GlobalTensor<T>& dst)
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
    template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
    template <uint32_t I, uint32_t CurTM>
    __aicore__
    inline void DoubleBufferedPairCube<T, TM, D, N, Iterations>::Compute(const GlobalTensor<T>& dst)
#endif
{
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
#else
    if constexpr (I == 0U) {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID7);
    }
#endif
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr event_t cSlot = DoubleC ? slot : EVENT_ID0;
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
    if constexpr (SplitL0B) {
        ::SharedComputeSplit<T, TM, D, N, A, B, I, CurTM>(*this, dst);
    } else {
        ComputeStage5944224d<I, CurTM>(dst);
    }
#else
    LocalTensor<T> a1(TPosition::A1, (I % 2U) * A * sizeof(T), CurTM * D);
    LocalTensor<T> b2(TPosition::B2, 0, B);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    if constexpr (DoubleA2) {
        LocalTensor<T> a2(TPosition::A2, (I % 2U) * A * sizeof(T), CurTM * D);
        ComputeStage5d29390b<I, CurTM>(dst, a1, b2, a2);
        SetFlag<HardEvent::FIX_M>(cSlot);
    } else {
        ComputeStageede6278d<I, CurTM>(dst, a1, b2);
    }
#else
    if constexpr (DoubleA2) {
        ComputeStage71fca89b<I, CurTM>(a1, b2);
    } else {
        constexpr uint32_t HM = CurTM / 2U;
        LocalTensor<T> a2_0(TPosition::A2, 0U, HM * D);
        LocalTensor<T> a2_1(TPosition::A2, HM * D * sizeof(T), HM * D);
        LocalTensor<float> c0(TPosition::CO1, (I % 2U) * 2U * TM * N * sizeof(float), 2U * HM * N);
        LocalTensor<float> c1(TPosition::CO1, ((I % 2U) * 2U * TM * N + 2U * HM * N) * sizeof(float),
                              2U * HM * N);
        ComputeStage6e2aa2f7<I, CurTM>(a1, b2, a2_0, a2_1, c0, c1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::M_FIX>(EVENT_ID1);
        SharedStage69c5fc7c97<I, CurTM>(c0, c1, dst);
    }
#endif
#endif
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t TM, uint32_t D, uint32_t N>
template <uint32_t Iterations>
__aicore__ inline void DoubleBufferedPairCube<T, TM, D, N>::Drain()
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
    template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
    __aicore__ inline void DoubleBufferedPairCube<T, TM, D, N, Iterations>::Drain()
#endif
{
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    if constexpr (!DoubleA2 || Iterations >= 2U) {
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    }
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    if constexpr (DoubleC && Iterations >= 2U) {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
    }
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    if constexpr (Iterations >= 2U) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    }
#else
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    if constexpr (SplitL0B) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    }
#else
    ::SharedPrimitivebd2d8d032a<TM, Common, Common::SplitL0B>();
#endif
#endif
}

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t TM, uint32_t D, uint32_t N>
template <uint32_t I, uint32_t CurTM, class A0, class A1>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N>::ComputeStage71fca89b(A0& a1, A1& b2)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr event_t cSlot = DoubleC ? slot : EVENT_ID0;
    LocalTensor<T> a2(TPosition::A2, (I % 2U) * A * sizeof(T), CurTM * D);
    LocalTensor<float> c(TPosition::CO1, DoubleC ? (I % 2U) * 2U * TM * N * sizeof(float) : 0U,
                         2U * CurTM * N);
    SharedStage448bfd918a<I, CurTM>(a1, a2, b2, c);
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t TM, uint32_t D, uint32_t N>
template <uint32_t I>
__aicore__ inline void DoubleBufferedPairCube<T, TM, D, N>::Store(const GlobalTensor<T>& dst)
{
    static_assert(DoubleA2 && DoubleC, "Two independent producer buffers");
    constexpr event_t slot = (I % 2U == 0U) ? EVENT_ID0 : EVENT_ID1;
    LocalTensor<float> c(TPosition::CO1, (I % 2U) * 2U * TM * N * sizeof(float), 2U * TM * N);
    FixpipeParamsV220 fix;
    fix.nSize = 2U * N;
    fix.mSize = TM;
    fix.srcStride = TM;
    fix.dstStride = 2U * N;
    fix.quantPre = QuantMode_t::F322F16;
    WaitFlag<HardEvent::M_FIX>(slot);
    Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c, fix);
    SetFlag<HardEvent::FIX_M>(slot);
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t FirstTM, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7,
          class A8>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::PrepareWeightsStagec2e3b721(A0& w1, A1& w2, A2& x0, A3& b1,
                                                                             A4& a1, A5& b2_0, A6& b2_1,
                                                                             A7& hap, A8& bp)
{
    SharedKerneld5184a4a53<T, TM, D, N, A0, A1, A2, A3, A4, A5, A6, A7, A8>(w1, w2, x0, b1, a1, b2_0, b2_1,
                                                                            hap, bp);
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t TM0, uint32_t TM1, uint32_t TM2, uint32_t TM3, class A0, class A1, class A2, class A3,
          class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::PrepareAllStagecfd973b0(A0& x1, A1& x2, A2& x3, A3& a1_0,
                                                                         A4& a1_1, A5& a1_2, A6& b1, A7& a2_0,
                                                                         A8& b2_0, A9& b2_1, A10& ap1,
                                                                         A11& ap2)
{
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    DataCopy(a1_1, x1, ap1);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    ::SharedPrimitive13dcc0f596<D, TM0>(a1_0, a2_0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    DataCopy(a1_2, x2, ap2);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    ::SharedWaitWeights<Iterations>();
    LoadData2dParams bl;
    bl.repeatTimes = (N * D) / 256U;
    bl.srcStride = 1;
    LoadData(b2_0, b1, bl);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
    if constexpr (Iterations >= 4U) {
        LocalTensor<T> a1_3(TPosition::A1, 3U * A * sizeof(T), TM3 * D);
        const Nd2NzParams ap3{1, TM3, D, 0, D, TM3, 1, 0};
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        DataCopy(a1_3, x3, ap3);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    }
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    LoadData(b2_1, b1[N * D], bl);
}

#endif

}
