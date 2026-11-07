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
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t N, class B0, class B1>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitive0689c680de(B0& c1, B1& dst)
{
    SetFlag<HardEvent::M_FIX>(EVENT_ID1);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
    FixpipeParamsV220 fix;
    fix.nSize = 2U * N;
    fix.mSize = 48U;
    fix.srcStride = 48U;
    fix.dstStride = 2U * N;
    fix.quantPre = QuantMode_t::F322F16;
    Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c1, fix);
    SetFlag<HardEvent::FIX_M>(EVENT_ID1);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <uint32_t TM, typename Common, auto Common_SplitL0B>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitivebd2d8d032a()
{
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
    if constexpr (TM == 48U) {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID2);
    }
    if constexpr (Common_SplitL0B) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    }
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t D, uint32_t TM0, class B0, class B1>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitive13dcc0f596(B0& a1_0, B1& a2_0)
{
    LoadData2dParams al0;
    al0.repeatTimes = D / 16U;
    al0.srcStride = TM0 / 16U;
#pragma unroll
    for (uint32_t mb = 0; mb < TM0 / 16U; ++mb) {
        LoadData(a2_0[mb * D * 16U], a1_0[mb * 256U], al0);
    }
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <uint32_t D, uint32_t N, class B0, class B1, class B2, class B3>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitived55642c3a1(B0& a2_2, B1& b2_0, B2& b2_1,
                                                                                B3& c2)
{
    MmadParams mad;
    mad.m = 48U;
    mad.n = N;
    mad.k = D;
    mad.cmatrixInitVal = true;
    Mmad(c2, a2_2, b2_0, mad);
    Mmad(c2[48U * N], a2_2, b2_1, mad);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <uint32_t D, class B0, class B1>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitivedc71102ef9(B0& a1, B1& a2_2)
{
    LoadData2dParams al;
    al.repeatTimes = D / 16U;
    al.srcStride = 48U / 16U;
#pragma unroll
    for (uint32_t mb = 0; mb < 48U / 16U; ++mb) {
        LoadData(a2_2[mb * D * 16U], a1[mb * 256U], al);
    }
}
#endif
template <HardEvent E> __aicore__ inline __attribute__((always_inline)) void SharedSynccc9d9b6d()
{
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}

template <HardEvent E> __aicore__ inline __attribute__((always_inline)) void SharedSyncb55bd560()
{
    constexpr event_t event = EVENT_ID0;
    SetFlag<E>(event);
    WaitFlag<E>(event);
}

template <HardEvent E> __aicore__ inline __attribute__((always_inline)) void SharedSync64ce5475()
{
    constexpr event_t e = EVENT_ID0;
    SetFlag<E>(e);
    WaitFlag<E>(e);
}
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t Slot> __aicore__ inline __attribute__((always_inline)) void SharedReady94b23fa6()
{
    constexpr uint32_t id = Slot == 0U ? 0x6 : (Slot == 1U ? 0x7 : (Slot == 2U ? 0x8 : 0x9));
    CrossCoreSetFlag<0x2, PIPE_FIX>(id);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t Slot> __aicore__ inline __attribute__((always_inline)) void SharedWaitReady3eda4349()
{
    constexpr uint32_t id = Slot == 0U ? 0x6 : (Slot == 1U ? 0x7 : (Slot == 2U ? 0x8 : 0x9));
    CrossCoreWaitFlag(id);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t Slot> __aicore__ inline __attribute__((always_inline)) void SharedConsumed245076b5()
{
    constexpr uint32_t id = Slot == 0U ? 0xA : (Slot == 1U ? 0xB : (Slot == 2U ? 0xC : 0xE));
    CrossCoreSetFlag<0x2, PIPE_MTE2>(id);
}
#endif
template <uint32_t FixedM, class B0, class B1, class B2, class B3, class B4>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitive8310a39caa(B0& col, B1& d, B2& gy,
                                                                                B3& out, B4& row)
{
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    if constexpr (FixedM == 1U) {
        DataCopy(gy[col], out, 16U);
    } else {
        DataCopy(gy[row * d + col], out, 16U);
    }
}
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t I, uint32_t InSlots, uint32_t Plane, class Core>
__aicore__ inline __attribute__((always_inline)) uint64_t SharedWaitNextTile(Core& core_)
{
    constexpr uint32_t nextSlot = (I + 1U) % 4U;
    constexpr uint32_t nextUbSlot = (I + 1U) % InSlots;
    constexpr event_t nextEv =
        nextUbSlot == 0U ? EVENT_ID0
                         : (nextUbSlot == 1U ? EVENT_ID1 : (nextUbSlot == 2U ? EVENT_ID2 : EVENT_ID3));
    ::SharedWaitReady3eda4349<nextSlot>();
    if constexpr (I + 1U >= InSlots) {
        WaitFlag<HardEvent::V_MTE2>(nextEv);
    }
    return (core_ * 4U + nextSlot) * 2U * Plane;
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <uint32_t I, uint32_t Iterations, uint32_t InSlots, uint32_t Plane, uint32_t Cap, uint32_t N,
          uint32_t NextNR, bool Single, bool Release, class Input, class Slots, class Core>
__aicore__ inline __attribute__((always_inline)) void SharedPrefetchTile(Input& input, Slots& slots_,
                                                                         Core& core_)
{
    if constexpr (I + 1U < Iterations) {
        constexpr uint32_t nextSlot = (I + 1U) % 4U;
        constexpr uint32_t nextUbSlot = (I + 1U) % InSlots;
        constexpr event_t nextEv =
            nextUbSlot == 0U ? EVENT_ID0
                             : (nextUbSlot == 1U ? EVENT_ID1 : (nextUbSlot == 2U ? EVENT_ID2 : EVENT_ID3));
        const uint64_t nextOff = ::SharedWaitNextTile<I, InSlots, Plane>(core_);
        constexpr uint32_t nextNR = NextNR;
        if constexpr (Single) {
            DataCopy(input[nextUbSlot * 2U * Cap], slots_[nextOff + 0U * nextNR * 2U * N], 2U * nextNR * N);
        } else {
            DataCopy(input[nextUbSlot * 2U * Cap], slots_[nextOff + GetSubBlockIdx() * nextNR * 2U * N],
                     2U * nextNR * N);
        }
        if constexpr (Release && I + 5U < Iterations) {
            ::SharedConsumed245076b5<nextSlot>();
        }
        SetFlag<HardEvent::MTE2_V>(nextEv);
    }
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t A, uint32_t B, uint32_t I, uint32_t CurTM,
          class Self>
__aicore__ inline __attribute__((always_inline)) void SharedComputeSplit(Self& self,
                                                                         const GlobalTensor<T>& dst)
{
    LocalTensor<T> a1(TPosition::A1, (I % 2U) * A * sizeof(T), CurTM * D);
    LocalTensor<T> b1(TPosition::B1, 2U * A * sizeof(T), B);
    LocalTensor<T> b2_0(TPosition::B2, 0, N * D / 2U);
    LocalTensor<T> b2_1(TPosition::B2, N * D / 2U * sizeof(T), N * D / 2U);
    LoadData2dParams bl;
    bl.repeatTimes = (N * D / 2U) / 256U;
    bl.srcStride = 1;
    constexpr uint32_t HM = TM / 2U;
    constexpr uint32_t firstOff = (I % 2U == 1U) ? (HM * N) : 0U;
    constexpr uint32_t secondOff = (I % 2U == 1U) ? 0U : (HM * N);
    constexpr uint32_t nextB1Off = (I % 2U == 1U) ? 0U : (N * D);
    if constexpr (CurTM <= HM) {
        constexpr uint32_t cFirstOff = (I % 2U == 1U) ? (CurTM * N) : 0U;
        constexpr uint32_t cSecondOff = (I % 2U == 1U) ? 0U : (CurTM * N);
        LocalTensor<T> a2_0(TPosition::A2, 0U, CurTM * D);
        LocalTensor<float> c0(TPosition::CO1, 0U, 2U * HM * N);
        self.template ComputeSplitSmall<I, CurTM>(a1, a2_0, b1, b2_0, b2_1, bl, c0, dst);
    } else {
        self.template ComputeSplitLarge<I, CurTM>(a1, b1, b2_0, b2_1, bl, dst);
    }
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t A, uint32_t B, uint32_t FirstTM>
__aicore__ inline __attribute__((always_inline)) void
SharedPrepareWeights(const GlobalTensor<T>& w1, const GlobalTensor<T>& w2, const GlobalTensor<T>& x0)
{
    LocalTensor<T> b1(TPosition::B1, 2U * A * sizeof(T), B);
    LocalTensor<T> a1(TPosition::A1, 0, FirstTM * D);
    LocalTensor<T> b2_0(TPosition::B2, 0, N * D / 2U);
    LocalTensor<T> b2_1(TPosition::B2, N * D / 2U * sizeof(T), N * D / 2U);
    constexpr uint32_t HM = TM / 2U;
    const Nd2NzParams hap{1, HM, D, 0, D, HM, 1, 0};
    const Nd2NzParams bp{1, N / 2U, D, 0, D, N / 2U, 1, 0};
    ::SharedKerneld5184a4a53<T, TM, D, N>(w1, w2, x0, b1, a1, b2_0, b2_1, hap, bp);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t I, uint32_t CurTM, bool UseDefault = false, class Piped, class Slots, class Offset>
__aicore__ inline __attribute__((always_inline)) void SharedComputeReady(Piped& piped_, Slots& slots_,
                                                                         Offset& off)
{
    constexpr uint32_t Slot = I % 4U;
    if ASCEND_IS_AIC {
        if constexpr (UseDefault) {
            piped_.template Compute<I>(slots_[off]);
        } else {
            piped_.template Compute<I, CurTM>(slots_[off]);
        }
        ::SharedReady94b23fa6<Slot>();
    }
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t Iterations> __aicore__ inline __attribute__((always_inline)) void SharedWaitWeights()
{
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    if constexpr (Iterations >= 4U) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    }
}
#endif
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <uint32_t D, auto N, auto NR, auto Cap, uint32_t I, uint32_t CurIter, class B0, class B1, class B2,
          class C0, class C1>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitive9cb48942a1(C0& y_, C1& col_, B0& begin,
                                                                                B1& out, B2& row)
{
    constexpr uint32_t slot = I % 4U;
    constexpr event_t vEv =
        slot == 0U ? EVENT_ID0 : (slot == 1U ? EVENT_ID1 : (slot == 2U ? EVENT_ID2 : EVENT_ID3));
    DataCopyParams op{NR, N / 16U, 0, (D - N) / 16U};
    if constexpr (D == N) {
        DataCopy(y_[(row + begin) * D], out, Cap);
    } else {
        DataCopy(y_[(row + begin) * D + col_], out, op);
    }
    if constexpr (I + 4U < CurIter) {
        SetFlag<HardEvent::MTE3_V>(vEv);
    }
}
#endif
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <uint32_t Slot> __aicore__ inline __attribute__((always_inline)) void SharedReadyecfb6a7c()
{
    constexpr uint32_t id = Slot == 0 ? 0x6 : (Slot == 1 ? 0x7 : (Slot == 2 ? 0xB : 0xE));
    CrossCoreSetFlag<0x2, PIPE_FIX>(id);
}
#endif
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <uint32_t Slot> __aicore__ inline __attribute__((always_inline)) void SharedWaitReadyfe305ae3()
{
    constexpr uint32_t id = Slot == 0 ? 0x6 : (Slot == 1 ? 0x7 : (Slot == 2 ? 0xB : 0xE));
    CrossCoreWaitFlag(id);
}
#endif
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <uint32_t Slot> __aicore__ inline __attribute__((always_inline)) void SharedConsumed144332b4()
{
    constexpr uint32_t id = Slot == 0 ? 0x8 : (Slot == 1 ? 0x9 : (Slot == 2 ? 0xC : 0xF));
    CrossCoreSetFlag<0x2, PIPE_MTE2>(id);
}
#endif
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <uint32_t Slot> __aicore__ inline __attribute__((always_inline)) void SharedReady2a68fd75()
{
    constexpr uint32_t id = Slot == 0U ? 0x6 : (Slot == 1U ? 0x7 : (Slot == 2U ? 0xB : 0xE));
    CrossCoreSetFlag<0x2, PIPE_FIX>(id);
}
#endif
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <uint32_t Slot> __aicore__ inline __attribute__((always_inline)) void SharedWaitReady8b083f09()
{
    constexpr uint32_t id = Slot == 0U ? 0x6 : (Slot == 1U ? 0x7 : (Slot == 2U ? 0xB : 0xE));
    CrossCoreWaitFlag(id);
}
#endif
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t N, uint32_t I, class B0, class B1, class B2>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitive2f7c32ba02(B0& c0, B1& dst, B2& fix)
{
    fix.dstStride = 2U * N;
    fix.quantPre = QuantMode_t::F322F16;
    fix.unitFlag = 3;

    if constexpr (I >= 4U) {
        constexpr uint32_t cid = (I % 4U) == 0U ? 0xA : ((I % 4U) == 1U ? 0xB : ((I % 4U) == 2U ? 0xC : 0xE));
        CrossCoreWaitFlag(cid);
    }
    Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c0, fix);
}
#endif
