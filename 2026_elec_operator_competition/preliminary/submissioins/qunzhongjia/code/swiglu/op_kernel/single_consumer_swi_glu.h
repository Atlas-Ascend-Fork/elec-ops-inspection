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
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN = 64U>
class SingleConsumerSwiGlu {
public:
    template <class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void RunStage9d5a5f63(A0& b1, A1& w2, A2& b2,
                                                                           A3& workspace);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               GM_ADDR workspace);

private:
    template <uint32_t I, class B0>
    __aicore__ inline __attribute__((always_inline)) void StepGroupFour(B0& off);
    template <uint32_t I, class B0>
    __aicore__ inline __attribute__((always_inline)) void StepDefault(B0& off);
    template <uint32_t I> __aicore__ inline void Step();
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3, class A4, class A5,
              class A6>
    __aicore__ inline __attribute__((always_inline)) void
    EpilogueStage9e7e3712(A0& curInput, A1& b1, A2& b2, A3& g, A4& u, A5& sig, A6& out);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void EpilogueStage1de823e0(A0& row, A1& begin, A2& input,
                                                                                A3& out);
    template <uint32_t I, uint32_t CurTM = TM, class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void EpilogueStage368944e2(A0& row, A1& begin, A2& input,
                                                                                A3& out);
    template <uint32_t I, uint32_t CurTM = TM> __aicore__ inline void Epilogue(uint64_t off, uint64_t row);
    static constexpr uint32_t N = TileN;
    static constexpr uint32_t TN = D / N;
    static constexpr uint32_t Groups = Cores / TN;
    static constexpr uint32_t Plane = TM * N;
    static constexpr uint32_t M = (Groups == 4U) ? 512U : ((Groups == 10U) ? 8192U : 4096U);
    static constexpr uint32_t FirstTM = (Groups == 4U) ? 32U : TM;
    static constexpr bool FastAiv = true;
    static constexpr bool SplitL0B = (2U * N * D * sizeof(T) > 64U * 1024U);
    static constexpr uint32_t NR = TM;
    static constexpr uint32_t Cap = NR * N;
    static constexpr uint32_t InSlots = 3U;
    static constexpr uint32_t OutSlots = 3U;
    static_assert(D % N == 0U && TM % 16U == 0U, "Exact tile shape");
    static_assert(TM * D * sizeof(T) <= 64U * 1024U, "L0A capacity");
    static_assert(N * D * sizeof(T) <= 64U * 1024U, "L0B half capacity");
    static_assert(2U * TM * N * sizeof(float) <= 128U * 1024U, "L0C capacity");
    static_assert(((2U * InSlots + 3U + OutSlots) * Cap + 2U * N) * sizeof(T) <= 192U * 1024U, "UB capacity");
    SingleConsumerPairCube<T, TM, D, N, Iterations> piped_;
    GlobalTensor<T> x_;
    GlobalTensor<T> w1_;
    GlobalTensor<T> w2_;
    GlobalTensor<T> b1_;
    GlobalTensor<T> b2_;
    GlobalTensor<T> y_;
    GlobalTensor<T> slots_;
    uint32_t core_ = 0;
    uint32_t col_ = 0;
    uint32_t grp_ = 0;
    uint32_t baseRow_ = 0;
};
#endif
}

namespace optiling {
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <class A0, class A1, class A2, class A3>
__aicore__ inline __attribute__((always_inline)) void
SingleConsumerSwiGlu<T, D, TM, Cores, Iterations, TileN>::RunStage9d5a5f63(A0& b1, A1& w2, A2& b2,
                                                                           A3& workspace)
{
    w2_.SetGlobalBuffer((__gm__ T*)w2, D * D);
    b1_.SetGlobalBuffer((__gm__ T*)b1, D);
    b2_.SetGlobalBuffer((__gm__ T*)b2, D);
    slots_.SetGlobalBuffer((__gm__ T*)workspace, 2U * Plane * 4U * Cores);
    SetAtomicNone();
    ::SharedPrimitivef685356fbf<D, Groups, FirstTM>(piped_, x_, w1_, w2_, col_, grp_, baseRow_);
    if ASCEND_IS_AIV {
        SetMaskNorm();
        SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
        LocalTensor<T> bias(TPosition::VECIN, 2U * InSlots * Cap * sizeof(T), 2U * N);
        DataCopy(bias, b1_[col_], N);
        DataCopy(bias[N], b2_[col_], N);
        if constexpr (FastAiv) {
            constexpr uint32_t FirstNR = FirstTM;
            constexpr uint32_t FirstCap = FirstNR * N;
            ::SharedWaitReady3eda4349<0U>();
            LocalTensor<T> input0(TPosition::VECIN, 0, 2U * Cap);
            DataCopy(input0, slots_[core_ * 8U * Plane + 0U * FirstNR * 2U * N], 2U * FirstCap);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
    }
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
__aicore__ inline void
SingleConsumerSwiGlu<T, D, TM, Cores, Iterations, TileN>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
                                                              GM_ADDR b2, GM_ADDR y, GM_ADDR workspace)
{
    core_ = GetBlockIdx();
    // AIC:AIV 1:1: both processors have the same logical block index.
    ::SharedPrimitivecbf5c24993<N, TN, Groups>(core_, col_, grp_, baseRow_);
    x_.SetGlobalBuffer((__gm__ T*)x, M * D);
    y_.SetGlobalBuffer((__gm__ T*)y, M * D);
    w1_.SetGlobalBuffer((__gm__ T*)w1, D * D);
    RunStage9d5a5f63(b1, w2, b2, workspace);
    Step<0>();
    if ASCEND_IS_AIC {
        piped_.Drain();
    }
    PipeBarrier<PIPE_ALL>();
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, class B0>
__aicore__ inline __attribute__((always_inline)) void
SingleConsumerSwiGlu<T, D, TM, Cores, Iterations, TileN>::StepGroupFour(B0& off)
{
    constexpr uint32_t curTM = (I == 0U) ? 32U : 48U;
    constexpr uint64_t rowOff = (I == 0U) ? 0U : ((I == 1U) ? 32U : 80U);
    const uint64_t row = baseRow_ + rowOff;
    ::SharedComputeReady<I, curTM>(piped_, slots_, off);
    if ASCEND_IS_AIV {
        Epilogue<I, curTM>(off, row);
    }
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, class B0>
__aicore__ inline __attribute__((always_inline)) void
SingleConsumerSwiGlu<T, D, TM, Cores, Iterations, TileN>::StepDefault(B0& off)
{
    const uint64_t row = (I * 1024U) + grp_ * 64U;
    ::SharedComputeReady<I, 64U>(piped_, slots_, off);
    if ASCEND_IS_AIV {
        Epilogue<I, 64U>(off, row);
    }
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I>
__aicore__ inline void SingleConsumerSwiGlu<T, D, TM, Cores, Iterations, TileN>::Step()
{
    constexpr uint32_t Slot = I % 4U;
    const uint64_t off = (core_ * 4U + Slot) * 2U * Plane;
    if constexpr (Groups == 4U) {
        StepGroupFour<I>(off);
    } else if constexpr (Groups == 10U) {
        constexpr uint32_t CurTM = (I < 6U) ? 128U : 64U;
        const uint64_t row = baseRow_ + I * 128U;
        if ASCEND_IS_AIC {
            ::SharedPrimitivec49e28b728<D, I>(piped_, x_, slots_, grp_, baseRow_, off, row);
            ::SharedReady94b23fa6<Slot>();
        }
        if ASCEND_IS_AIV {
            Epilogue<I, CurTM>(off, row);
        }
    } else {
        StepDefault<I>(off);
    }
    if constexpr (I + 1U < Iterations) {
        Step<I + 1U>();
    }
}


template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
__aicore__ inline __attribute__((always_inline)) void
SingleConsumerSwiGlu<T, D, TM, Cores, Iterations, TileN>::EpilogueStage9e7e3712(A0& curInput, A1& b1, A2& b2,
                                                                                A3& g, A4& u, A5& sig,
                                                                                A6& out)
{
    constexpr uint32_t ubSlot = I % InSlots;
    constexpr event_t vEv =
        ubSlot == 0U ? EVENT_ID0 : (ubSlot == 1U ? EVENT_ID1 : (ubSlot == 2U ? EVENT_ID2 : EVENT_ID3));
    constexpr uint32_t CurNR = CurTM;
    constexpr uint32_t CurCap = CurNR * N;
    WaitFlag<HardEvent::MTE2_V>(vEv);
    if constexpr (I == 0U && Iterations > 4U) {
        ::SharedConsumed245076b5<0U>();
    }
    const BinaryRepeatParams rp{1, 1, 1, N / 16U, 2U * N / 16U, 0};
    constexpr uint64_t mask = 0;
    ::SharedPrimitivee3102f6fcd<T, TM, Iterations, N, InSlots, I, CurTM, true>(b1, b2, curInput, g, rp, u);
    constexpr uint8_t repeats = CurCap / 128U;
    const UnaryRepeatParams ur{1, 1, 8, 8};
    const BinaryRepeatParams br{1, 1, 1, 8, 8, 8};
    if constexpr (N != 128U) {
        SetVectorMask<T, MaskMode::NORMAL>(~0ULL, ~0ULL);
    }
    Muls<T, false>(sig, g, static_cast<T>(-1), mask, repeats, ur);
    ::SharedPrimitive0547bf83d4<T, TM, N, InSlots, OutSlots, I, CurTM, true>(br, g, out, sig, u, ur);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3>
__aicore__ inline __attribute__((always_inline)) void
SingleConsumerSwiGlu<T, D, TM, Cores, Iterations, TileN>::EpilogueStage1de823e0(A0& row, A1& begin, A2& input,
                                                                                A3& out)
{
    constexpr uint32_t ubSlot = I % InSlots;
    constexpr event_t vEv =
        ubSlot == 0U ? EVENT_ID0 : (ubSlot == 1U ? EVENT_ID1 : (ubSlot == 2U ? EVENT_ID2 : EVENT_ID3));
    constexpr uint32_t CurNR = CurTM;
    constexpr uint32_t CurCap = CurNR * N;
    SetFlag<HardEvent::V_MTE3>(vEv);
    ::SharedPrefetchTile<I, Iterations, InSlots, Plane, Cap, N, (Groups == 4U) ? 48U : 64U, true, false>(
        input, slots_, core_);
    WaitFlag<HardEvent::V_MTE3>(vEv);
    DataCopyParams op{CurNR, N / 16U, 0, (D - N) / 16U};
    ::SharedPrimitiveac2daf19f8<D, TM, Iterations, N, InSlots, OutSlots, I, CurTM, true>(y_, col_, begin, op,
                                                                                         out, row);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3>
__aicore__ inline __attribute__((always_inline)) void
SingleConsumerSwiGlu<T, D, TM, Cores, Iterations, TileN>::EpilogueStage368944e2(A0& row, A1& begin, A2& input,
                                                                                A3& out)
{
    constexpr uint32_t ubSlot = I % InSlots;
    constexpr event_t vEv =
        ubSlot == 0U ? EVENT_ID0 : (ubSlot == 1U ? EVENT_ID1 : (ubSlot == 2U ? EVENT_ID2 : EVENT_ID3));
    constexpr uint32_t CurNR = CurTM;
    SetFlag<HardEvent::V_MTE3>(vEv);
    WaitFlag<HardEvent::V_MTE3>(vEv);
    DataCopyParams op{CurNR, N / 16U, 0, (D - N) / 16U};
    ::SharedPrimitive1908ccb961<D, Iterations, Groups, InSlots, OutSlots, I>(y_, col_, grp_, begin, op, out,
                                                                             row);
    ::SharedPrefetchTile<I, Iterations, InSlots, Plane, Cap, N, (I < 5U) ? 64U : 32U, true, true>(
        input, slots_, core_);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, uint32_t CurTM>
__aicore__ inline void SingleConsumerSwiGlu<T, D, TM, Cores, Iterations, TileN>::Epilogue(uint64_t off,
                                                                                          uint64_t row)
{
    constexpr uint32_t ubSlot = I % InSlots;
    constexpr event_t vEv =
        ubSlot == 0U ? EVENT_ID0 : (ubSlot == 1U ? EVENT_ID1 : (ubSlot == 2U ? EVENT_ID2 : EVENT_ID3));
    constexpr uint32_t CurNR = CurTM;
    constexpr uint32_t CurCap = CurNR * N;
    const uint32_t begin = 0U * CurNR;
    LocalTensor<T> input(TPosition::VECIN, 0, 2U * InSlots * Cap + 2U * N);
    auto curInput = input[ubSlot * 2U * Cap];
    auto b1 = input[2U * InSlots * Cap];
    auto b2 = input[2U * InSlots * Cap + N];
    LocalTensor<T> work(TPosition::VECOUT, (2U * InSlots * Cap + 2U * N) * sizeof(T), (3U + OutSlots) * Cap);
    LocalTensor<T> g(TPosition::VECCALC, 66048U, Cap);
    LocalTensor<T> u(TPosition::VECCALC, 66048U + Cap * sizeof(T), Cap);
    LocalTensor<T> sig(TPosition::VECCALC, 131328U, Cap);
    LocalTensor<T> out(TPosition::VECOUT, 131328U + (1U + ubSlot) * Cap * sizeof(T), Cap);
    EpilogueStage9e7e3712<I, CurTM>(curInput, b1, b2, g, u, sig, out);
    if constexpr (SplitL0B) {
        EpilogueStage368944e2<I, CurTM>(row, begin, input, out);
    } else {
        EpilogueStage1de823e0<I, CurTM>(row, begin, input, out);
    }
}

#endif

}
