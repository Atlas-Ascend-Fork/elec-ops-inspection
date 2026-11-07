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
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN,
          uint32_t TailIterations, uint32_t FullCores>
template <uint32_t I, uint32_t CurIter, class B0, class B1, class B2>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN, TailIterations, FullCores>::ApplyEpilogue(B0& begin,
                                                                                                  B1& off,
                                                                                                  B2& row)
{
    constexpr uint32_t slot = I % 4U;
    constexpr event_t vEv =
        slot == 0U ? EVENT_ID0 : (slot == 1U ? EVENT_ID1 : (slot == 2U ? EVENT_ID2 : EVENT_ID3));
    LocalTensor<T> input(TPosition::VECIN, 0, 2U * InSlots * Cap + 2U * N);
    auto curInput = input[slot * 2U * Cap];
    auto b1 = input[2U * InSlots * Cap];
    auto b2 = input[2U * InSlots * Cap + N];
    LocalTensor<T> work(TPosition::VECOUT, (2U * InSlots * Cap + 2U * N) * sizeof(T), (3U + OutSlots) * Cap);
    auto g = work;
    auto u = work[Cap];
    auto sig = work[2U * Cap];
    auto out = work[(3U + slot) * Cap];
    const uint64_t start = off + begin * 2U * N;
    if constexpr (!FastAiv) {
        if constexpr (I >= 4U) {
            WaitFlag<HardEvent::V_MTE2>(vEv);
        }
        DataCopy(curInput, slots_[start], 2U * Cap);
        if constexpr (I + 4U < CurIter) {
            ::SharedConsumed144332b4<slot>();
        }
        ::SharedSynccc9d9b6d<HardEvent::MTE2_V>();
    } else {
        WaitFlag<HardEvent::MTE2_V>(vEv);
    }
    const BinaryRepeatParams rp{1, 1, 1, N / 16U, 2U * N / 16U, 0};
    constexpr uint64_t mask = 0;
    if constexpr (N != 128U) {
        SetVectorMask<T, MaskMode::NORMAL>(0, ~0ULL);
    }
    Add<T, false>(g, curInput, b1, mask, static_cast<uint8_t>(NR), rp);
    Add<T, false>(u, curInput[N], b2, mask, static_cast<uint8_t>(NR), rp);
    if constexpr (I + 4U < CurIter) {
        SetFlag<HardEvent::V_MTE2>(vEv);
    }
    static_assert(Cap % 128U == 0U && Cap / 128U <= 255U);
    EpilogueStage23fd9fc7<I, CurIter>(row, begin, input, g, u, sig, out);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
#if !((defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, uint32_t CurTM, class B0, class B1>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::ApplyEpilogue(B0& begin, B1& row)
#endif
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
    template <uint32_t I, uint32_t CurTM, class B0, class B1, class B2>
    __aicore__ inline __attribute__((
        always_inline)) void PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::ApplyEpilogue(B0& begin,
                                                                                                    B1& off,
                                                                                                    B2& row)
#endif
{
    constexpr uint32_t ubSlot = I % InSlots;
    constexpr event_t vEv =
        ubSlot == 0U ? EVENT_ID0 : (ubSlot == 1U ? EVENT_ID1 : (ubSlot == 2U ? EVENT_ID2 : EVENT_ID3));
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    constexpr uint32_t CurNR = CurTM / 2U;
    constexpr uint32_t CurCap = CurNR * N;
#else
#endif
    LocalTensor<T> input(TPosition::VECIN, 0, 2U * InSlots * Cap + 2U * N);
    auto curInput = input[ubSlot * 2U * Cap];
    auto b1 = input[2U * InSlots * Cap];
    auto b2 = input[2U * InSlots * Cap + N];
    LocalTensor<T> work(TPosition::VECOUT, (2U * InSlots * Cap + 2U * N) * sizeof(T), (3U + OutSlots) * Cap);
    auto g = work;
    auto u = work[Cap];
    auto sig = work[2U * Cap];
    auto out = work[(3U + ubSlot) * Cap];
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    const uint64_t start = off + begin * 2U * N;
    if constexpr (!FastAiv) {
        if constexpr (I >= InSlots) {
            WaitFlag<HardEvent::V_MTE2>(vEv);
        }
        DataCopy(curInput, slots_[start], 2U * CurCap);
        if constexpr (I + 4U < Iterations) {
            ::SharedConsumed245076b5<I % 4U>();
        }
        ::SharedSynccc9d9b6d<HardEvent::MTE2_V>();
    } else {
        WaitFlag<HardEvent::MTE2_V>(vEv);
        if constexpr (I == 0U && Iterations > 4U) {
            ::SharedConsumed245076b5<0U>();
        }
    }
#else
    WaitFlag<HardEvent::MTE2_V>(vEv);
#endif
    EpilogueSharedStage1<I, CurTM>(row, begin, input, curInput, b1, b2, g, u, sig, out);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN,
          uint32_t TailIterations, uint32_t FullCores>
template <uint32_t I, uint32_t CurIter>
__aicore__ inline void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN, TailIterations, FullCores>::Epilogue(uint64_t off,
                                                                                             uint64_t row)
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL))
    template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
    template <uint32_t I, uint32_t CurTM>
    __aicore__
    inline void PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::Epilogue(uint64_t off, uint64_t row)
#endif
{
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    constexpr uint32_t slot = I % 4U;
    constexpr event_t vEv =
        slot == 0U ? EVENT_ID0 : (slot == 1U ? EVENT_ID1 : (slot == 2U ? EVENT_ID2 : EVENT_ID3));
    const uint32_t begin = GetSubBlockIdx() * NR;
    ApplyEpilogue<I, CurIter>(begin, off, row);
#else
    constexpr uint32_t ubSlot = I % InSlots;
    constexpr event_t vEv =
        ubSlot == 0U ? EVENT_ID0 : (ubSlot == 1U ? EVENT_ID1 : (ubSlot == 2U ? EVENT_ID2 : EVENT_ID3));
    constexpr uint32_t CurNR = CurTM / 2U;
    constexpr uint32_t CurCap = CurNR * N;
    const uint32_t begin = GetSubBlockIdx() * CurNR;
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    ApplyEpilogue<I, CurTM>(begin, off, row);
#else
    ApplyEpilogue<I, CurTM>(begin, row);
#endif
#endif
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t First, uint32_t Count>
__aicore__ inline void PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::PrefetchBatch()
{
    static_assert((First == 0U && Count == 1U) || (First == 1U && Count == 2U) ||
                  (First == 3U && Count == 1U));
    constexpr event_t ev = First == 0U ? EVENT_ID0 : (First == 1U ? EVENT_ID1 : EVENT_ID3);
    ::SharedWaitReady8b083f09<First>();
    if constexpr (Count == 2U) {
        ::SharedWaitReady8b083f09<First + 1U>();
    }
    LocalTensor<T> input(TPosition::VECIN, First * 2U * Cap * sizeof(T), Count * 2U * Cap);
    const uint64_t off = (core_ * 4U + First) * 2U * Plane + GetSubBlockIdx() * NR * 2U * N;
    if constexpr (Count == 1U) {
        DataCopy(input, slots_[off], 2U * Cap);
    } else {
        const DataCopyParams dp{2U, static_cast<uint16_t>(2U * Cap * sizeof(T) / 32U),
                                static_cast<uint16_t>(2U * Cap * sizeof(T) / 32U), 0U};
        DataCopy(input, slots_[off], dp);
    }
    SetFlag<HardEvent::MTE2_V>(ev);
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t First, uint32_t Count>
__aicore__ inline void PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::EpilogueBatch()
{
    constexpr event_t ev = First == 0U ? EVENT_ID0 : (First == 1U ? EVENT_ID1 : EVENT_ID3);
    constexpr uint32_t BatchCap = Count * Cap;
    constexpr uint32_t BatchRows = Count * NR;
    LocalTensor<T> input(TPosition::VECIN, First * 2U * Cap * sizeof(T), 2U * BatchCap);
    LocalTensor<T> bias(TPosition::VECIN, 65536U, 2U * N);
    LocalTensor<T> g(TPosition::VECCALC, 66048U, BatchCap);
    LocalTensor<T> u(TPosition::VECCALC, 82432U, BatchCap);
    LocalTensor<T> sig(TPosition::VECCALC, 131328U, BatchCap);
    LocalTensor<T> out(TPosition::VECOUT, 147712U + First * Cap * sizeof(T), BatchCap);
    LocalTensor<T> ones(TPosition::VECCALC, 123392U, 128U);
    WaitFlag<HardEvent::MTE2_V>(ev);
    constexpr uint64_t mask = 0;
    const BinaryRepeatParams rp{1, 1, 1, N / 16U, 2U * N / 16U, 0};
    Add<T, false>(g, input, bias, mask, static_cast<uint8_t>(BatchRows), rp);
    Add<T, false>(u, input[N], bias[N], mask, static_cast<uint8_t>(BatchRows), rp);
    constexpr uint8_t repeats = BatchCap / 128U;
    const UnaryRepeatParams ur{1, 1, 8, 8};
    const BinaryRepeatParams br{1, 1, 1, 8, 8, 8};
    const BinaryRepeatParams broadcast{1, 1, 1, 8, 0, 8};
    Muls<T, false>(sig, g, static_cast<T>(-1), mask, repeats, ur);
    Exp<T, false>(sig, sig, mask, repeats, ur);
    Adds<T, false>(sig, sig, static_cast<T>(1), mask, repeats, ur);
    Div<T, false>(sig, ones, sig, mask, repeats, broadcast);
    Mul<T, false>(out, g, sig, mask, repeats, br);
    Mul<T, false>(out, out, u, mask, repeats, br);
    // A2 CrossCoreWaitFlag blocks later dispatch. Queue this output before waiting for future input.
    SetFlag<HardEvent::V_MTE3>(ev);
    WaitFlag<HardEvent::V_MTE3>(ev);
    const uint64_t row = (core_ + First * Cores) * TM + GetSubBlockIdx() * NR;
    if constexpr (Count == 1U) {
        DataCopy(y_[row * D], out, Cap);
    } else {
        const DataCopyParams dp{2U, static_cast<uint16_t>(Cap * sizeof(T) / 32U), 0U,
                                static_cast<uint16_t>((Cores * TM * D - Cap) * sizeof(T) / 32U)};
        DataCopy(y_[row * D], out, dp);
    }
    if constexpr (First == 0U) {
        PrefetchBatch<1U, 2U>();
    } else if constexpr (First == 1U) {
        PrefetchBatch<3U, 1U>();
    }
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::RunStagee3ff0211(A0& x, A1& w1, A2& b1, A3& w2,
                                                                          A4& b2, A5& y, A6& workspace)
{
    if ASCEND_IS_AIV {
        core_ /= 2U;
    }
    col_ = core_ % TN * N;
    grp_ = core_ / TN;
    if constexpr (Groups == 4U) {
        baseRow_ = grp_ * 128U;
    } else if constexpr (Groups == 10U) {
        baseRow_ = (grp_ < 8U) ? (grp_ * 832U) : (6656U + (grp_ - 8U) * 768U);
    } else {
        baseRow_ = grp_ * TM;
    }
    ::SharedPrimitive88fe1d65b1<T, D, Cores, Plane, M>(x_, w1_, w2_, b1_, b2_, y_, slots_, b1, b2, w1, w2,
                                                       workspace, x, y);
    if ASCEND_IS_AIC {
        if constexpr (Groups == 4U) {
            piped_.template PrepareAll<32U, 64U, 32U>(w1_[col_ * D], w2_[col_ * D], x_[baseRow_ * D],
                                                      x_[(baseRow_ + 32U) * D], x_[(baseRow_ + 96U) * D],
                                                      x_[baseRow_ * D]);
        } else if constexpr (Groups == 16U) {
            piped_.template PrepareAll<64U, 64U, 64U, 64U>(
                w1_[col_ * D], w2_[col_ * D], x_[(grp_ + 0U * 16U) * 64U * D],
                x_[(grp_ + 1U * 16U) * 64U * D], x_[(grp_ + 2U * 16U) * 64U * D],
                x_[(grp_ + 3U * 16U) * 64U * D]);
        } else {
            piped_.template PrepareWeights<FirstTM>(w1_[col_ * D], w2_[col_ * D], x_[baseRow_ * D]);
        }
    }
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::SharedStageccbb7d440e()
{
    if ASCEND_IS_AIV {
        SetMaskNorm();
        SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
        LocalTensor<T> bias(TPosition::VECIN, 2U * InSlots * Cap * sizeof(T), 2U * N);
        DataCopy(bias, b1_[col_], N);
        DataCopy(bias[N], b2_[col_], N);
        if constexpr (FastAiv) {
            constexpr uint32_t FirstNR = FirstTM / 2U;
            constexpr uint32_t FirstCap = FirstNR * N;
            ::SharedWaitReady3eda4349<0U>();
            LocalTensor<T> input0(TPosition::VECIN, 0, 2U * Cap);
            DataCopy(input0, slots_[core_ * 8U * Plane + GetSubBlockIdx() * FirstNR * 2U * N], 2U * FirstCap);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
    }
    Step<0>();
    if ASCEND_IS_AIC {
        piped_.Drain();
    }
    PipeBarrier<PIPE_ALL>();
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, class B0, class B1>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::SharedStage00bbcac747(B0& off, B1& row)
{
    constexpr uint32_t Slot = I % 4U;
    constexpr uint32_t CurTM = (I < 6U) ? 128U : 64U;
    if ASCEND_IS_AIC {
        ::SharedPrimitivec49e28b728<D, I>(piped_, x_, slots_, grp_, baseRow_, off, row);
        ::SharedReady94b23fa6<Slot>();
    }
    if ASCEND_IS_AIV {
        Epilogue<I, CurTM>(off, row);
    }
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, class B0>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::StepGroupFour(B0& off)
{
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    constexpr uint64_t localRow = (I == 0U) ? 0U : ((I == 1U) ? 32U : 96U);
    constexpr uint32_t curTM = (I == 1U) ? 64U : 32U;
    const uint64_t row = baseRow_ + localRow;
#else
    constexpr uint32_t curTM = (I == 0U) ? 32U : 48U;
    constexpr uint64_t rowOff = (I == 0U) ? 0U : ((I == 1U) ? 32U : 80U);
    const uint64_t row = baseRow_ + rowOff;
#endif
    ::SharedComputeReady<I, curTM>(piped_, slots_, off);
    if ASCEND_IS_AIV {
        Epilogue<I, curTM>(off, row);
    }
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, class B0>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::StepDefault(B0& off)
{
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    const uint64_t row = (grp_ + I * Groups) * TM;
    ::SharedComputeReady<I, TM, true>(piped_, slots_, off);
    if ASCEND_IS_AIV {
        Epilogue<I>(off, row);
    }
#else
    const uint64_t row = (I * 1024U) + grp_ * 64U;
    ::SharedComputeReady<I, 64U>(piped_, slots_, off);
    if ASCEND_IS_AIV {
        Epilogue<I, 64U>(off, row);
    }
#endif
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5, class A6,
          class A7, class A8>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::EpilogueStage1f90073b(A0& row, A1& begin, A2& input,
                                                                               A3& g, A4& u, A5& sig, A6& out,
                                                                               A7& ur, A8& br)
{
    constexpr uint32_t ubSlot = I % InSlots;
    constexpr event_t vEv =
        ubSlot == 0U ? EVENT_ID0 : (ubSlot == 1U ? EVENT_ID1 : (ubSlot == 2U ? EVENT_ID2 : EVENT_ID3));
    constexpr uint32_t CurNR = CurTM / 2U;
    constexpr uint32_t CurCap = CurNR * N;
    constexpr uint64_t mask = 0;
    constexpr uint8_t repeats = CurCap / 128U;
    ::SharedPrimitive0547bf83d4<T, TM, N, InSlots, OutSlots, I, CurTM>(br, g, out, sig, u, ur);
    if constexpr (!FastAiv) {
        ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
        DataCopyParams op{CurNR, N / 16U, 0, (D - N) / 16U};
        ::SharedPrimitiveac2daf19f8<D, TM, Iterations, N, InSlots, OutSlots, I, CurTM>(y_, col_, begin, op,
                                                                                       out, row);
    } else {
        SetFlag<HardEvent::V_MTE3>(vEv);
        WaitFlag<HardEvent::V_MTE3>(vEv);
        DataCopyParams op{CurNR, N / 16U, 0, (D - N) / 16U};
        if constexpr (Groups == 10U && I == 6U) {
            if (grp_ < 8U) {
                DataCopy(y_[(row + begin) * D + col_], out, op);
            }
        } else if constexpr (D == N) {
            DataCopy(y_[(row + begin) * D], out, CurCap);
        } else {
            DataCopy(y_[(row + begin) * D + col_], out, op);
        }
        EpilogueSharedStage0<I, CurTM>(input);
    }
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3, class A4, class A5, class A6,
          class A7, class A8, class A9>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::EpilogueSharedStage1(A0& row, A1& begin, A2& input,
                                                                              A3& curInput, A4& b1, A5& b2,
                                                                              A6& g, A7& u, A8& sig, A9& out)
{
    constexpr uint32_t ubSlot = I % InSlots;
    constexpr event_t vEv =
        ubSlot == 0U ? EVENT_ID0 : (ubSlot == 1U ? EVENT_ID1 : (ubSlot == 2U ? EVENT_ID2 : EVENT_ID3));
    constexpr uint32_t CurNR = CurTM / 2U;
    constexpr uint32_t CurCap = CurNR * N;
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
#else
    if constexpr (I == 0U && Iterations > 4U) {
        ::SharedConsumed245076b5<0U>();
    }
#endif
    const BinaryRepeatParams rp{1, 1, 1, N / 16U, 2U * N / 16U, 0};
    constexpr uint64_t mask = 0;
    ::SharedPrimitivee3102f6fcd<T, TM, Iterations, N, InSlots, I, CurTM>(b1, b2, curInput, g, rp, u);
    constexpr uint8_t repeats = CurCap / 128U;
    const UnaryRepeatParams ur{1, 1, 8, 8};
    const BinaryRepeatParams br{1, 1, 1, 8, 8, 8};
    if constexpr (N != 128U) {
        SetVectorMask<T, MaskMode::NORMAL>(~0ULL, ~0ULL);
    }
    Muls<T, false>(sig, g, static_cast<T>(-1), mask, repeats, ur);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    EpilogueStage1f90073b<I, CurTM>(row, begin, input, g, u, sig, out, ur, br);
#else
    ::SharedPrimitive0547bf83d4<T, TM, N, InSlots, OutSlots, I, CurTM>(br, g, out, sig, u, ur);
    if constexpr (SplitL0B) {
        EpilogueSharedStage0<I, CurTM>(row, begin, input, out);
    } else {
        EpilogueStage224a0a0c<I, CurTM>(row, begin, input, out);
    }
#endif
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::RunStageec337e10(A0& x, A1& w1, A2& b1, A3& w2,
                                                                          A4& b2, A5& y, A6& workspace)
{
    if ASCEND_IS_AIV {
        core_ /= 2U;
    }
    ::SharedPrimitivecbf5c24993<N, TN, Groups>(core_, col_, grp_, baseRow_);
    ::SharedPrimitive88fe1d65b1<T, D, Cores, Plane, M>(x_, w1_, w2_, b1_, b2_, y_, slots_, b1, b2, w1, w2,
                                                       workspace, x, y);
    ::SharedPrimitivef685356fbf<D, Groups, FirstTM>(piped_, x_, w1_, w2_, col_, grp_, baseRow_);
}

#endif

}
