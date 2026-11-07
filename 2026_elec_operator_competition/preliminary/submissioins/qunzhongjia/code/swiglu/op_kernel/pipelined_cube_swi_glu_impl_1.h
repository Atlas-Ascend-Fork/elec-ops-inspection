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
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN, TailIterations, FullCores>::RunStage46925810(
    A0& x, A1& w1, A2& b1, A3& w2, A4& b2, A5& y, A6& workspace)
{
    if ASCEND_IS_AIV {
        core_ /= 2U;
    }
    col_ = core_ % TN * N;
    ::SharedPrimitive88fe1d65b1<T, D, Cores, Plane, M>(x_, w1_, w2_, b1_, b2_, y_, slots_, b1, b2, w1, w2,
                                                       workspace, x, y);
    if ASCEND_IS_AIC {
        piped_.PrepareWeights(w1_[col_ * D], w2_[col_ * D], x_[(core_ / TN) * TM * D]);
    }
    if ASCEND_IS_AIV {
        SetMaskNorm();
        SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
        LocalTensor<T> bias(TPosition::VECIN, 2U * InSlots * Cap * sizeof(T), 2U * N);
        DataCopy(bias, b1_[col_], N);
        DataCopy(bias[N], b2_[col_], N);
        LocalTensor<T> ones(TPosition::VECCALC, ((2U * InSlots + 3U + OutSlots) * Cap + 2U * N) * sizeof(T),
                            128U);
        Duplicate(ones, static_cast<T>(1), 128U);
        if constexpr (FastAiv) {
            ::SharedWaitReadyfe305ae3<0U>();
            LocalTensor<T> input0(TPosition::VECIN, 0, 2U * Cap);
            DataCopy(input0, slots_[core_ * 8U * Plane + GetSubBlockIdx() * NR * 2U * N], 2U * Cap);
            if constexpr (Iterations > 4U) {
                ::SharedConsumed144332b4<0U>();
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
    }
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN,
          uint32_t TailIterations, uint32_t FullCores>
__aicore__ inline void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN, TailIterations, FullCores>::Run(
    GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y, GM_ADDR workspace)
{
    core_ = GetBlockIdx();
    RunStage46925810(x, w1, b1, w2, b2, y, workspace);
    if constexpr (TailIterations == Iterations) {
        Step<0, Iterations>();
        if ASCEND_IS_AIC {
            piped_.template Drain<Iterations>();
        }
    } else {
        if (core_ < FullCores) {
            Step<0, Iterations>();
            if ASCEND_IS_AIC {
                piped_.template Drain<Iterations>();
            }
        } else {
            Step<0, TailIterations>();
            if ASCEND_IS_AIC {
                piped_.template Drain<TailIterations>();
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
__aicore__ inline void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
                                                             GM_ADDR b2, GM_ADDR y, GM_ADDR workspace)
{
    core_ = GetBlockIdx();
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    if ASCEND_IS_AIV {
        core_ /= 2U;
    }
    col_ = 0U;
    ::SharedPrimitive88fe1d65b1<T, D, Cores, Plane, M>(x_, w1_, w2_, b1_, b2_, y_, slots_, b1, b2, w1, w2,
                                                       workspace, x, y);
    if ASCEND_IS_AIC {
        piped_.PrepareWeights(w1_, w2_, x_[core_ * TM * D]);
    }
    if ASCEND_IS_AIV {
        SetMaskNorm();
        SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
        LocalTensor<T> bias(TPosition::VECIN, 65536U, 2U * N);
        DataCopy(bias, b1_, N);
        DataCopy(bias[N], b2_, N);
        LocalTensor<T> ones(TPosition::VECCALC, 123392U, 128U);
        Duplicate(ones, static_cast<T>(1), 128U);
        PrefetchBatch<0U, 1U>();
    }
    Step<0U>();
    if ASCEND_IS_AIC {
        piped_.template Drain<Iterations>();
    }
    PipeBarrier<PIPE_ALL>();
#else
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    RunStagee3ff0211(x, w1, b1, w2, b2, y, workspace);
#else
    RunStageec337e10(x, w1, b1, w2, b2, y, workspace);
#endif
    SharedStageccbb7d440e();
#endif
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN,
          uint32_t TailIterations, uint32_t FullCores>
template <uint32_t Slot>
__aicore__ inline void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN, TailIterations, FullCores>::WaitConsumed()
{
    constexpr uint32_t id = Slot == 0 ? 0x8 : (Slot == 1 ? 0x9 : (Slot == 2 ? 0xC : 0xF));
    CrossCoreWaitFlag(id);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I>
__aicore__ inline void PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::Step()
{
    constexpr uint32_t Slot = I % 4U;
    const uint64_t off = (core_ * 4U + Slot) * 2U * Plane;
    if constexpr (Groups == 4U) {
        StepGroupFour<I>(off);
    } else if constexpr (Groups == 10U) {
        constexpr uint32_t CurTM = (I < 6U) ? 128U : 64U;
        const uint64_t row = baseRow_ + I * 128U;
        SharedStage00bbcac747<I>(off, row);
    } else {
        StepDefault<I>(off);
    }
    if constexpr (I + 1U < Iterations) {
        Step<I + 1U>();
    }
}
#endif
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN,
          uint32_t TailIterations, uint32_t FullCores>
template <uint32_t I, uint32_t CurIter>
__aicore__ inline void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN, TailIterations, FullCores>::Step()
{
    constexpr uint32_t Slot = I % 4U;
    const uint64_t row = (core_ / TN + I * (Cores / TN)) * TM;
    const uint64_t off = (core_ * 4U + Slot) * 2U * Plane;
    if ASCEND_IS_AIC {
        if constexpr (!FastAiv && I >= 4U) {
            WaitConsumed<Slot>();
        }
        if constexpr (I + 1U < CurIter) {
            piped_.template Prefetch<I + 1U>(x_[(row + (Cores / TN) * TM) * D]);
        }
        piped_.template Compute<I>(slots_[off]);
        ::SharedReadyecfb6a7c<Slot>();
    }
    if ASCEND_IS_AIV {
        if constexpr (!FastAiv) {
            ::SharedWaitReadyfe305ae3<Slot>();
        }
        Epilogue<I, CurIter>(off, row);
    }
    if constexpr (I + 1U < CurIter) {
        Step<I + 1U, CurIter>();
    }
}
#endif
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I>
__aicore__ inline void PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::Step()
{
    constexpr uint32_t Slot = I % 4U;
    const uint64_t row = (core_ / TN + I * (Cores / TN)) * TM;
    const uint64_t off = (core_ * 4U + Slot) * 2U * Plane;
    if ASCEND_IS_AIC {
        static_assert(Iterations == 4U, "One-ahead schedule covers four tiles");
        if constexpr (I == 0U) {
            piped_.template Prefetch<1U>(x_[(row + (Cores / TN) * TM) * D]);
            piped_.template Compute<0U>(slots_[off]);
        }
        if constexpr (I + 1U < Iterations) {
            if constexpr (I + 2U < Iterations) {
                piped_.template Prefetch<I + 2U>(x_[(row + 2U * (Cores / TN) * TM) * D]);
            }
            const uint64_t nextOff = (core_ * 4U + (I + 1U)) * 2U * Plane;
            piped_.template Compute<I + 1U>(slots_[nextOff]);
        }
        piped_.template Store<I>(slots_[off]);
        ::SharedReady2a68fd75<Slot>();
    }
    if ASCEND_IS_AIV {
        if constexpr (I == 0U) {
            EpilogueBatch<0U, 1U>();
        } else if constexpr (I == 1U) {
            EpilogueBatch<1U, 2U>();
        } else if constexpr (I == 3U) {
            EpilogueBatch<3U, 1U>();
        }
    }
    if constexpr (I + 1U < Iterations) {
        Step<I + 1U>();
    }
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::EpilogueSharedStage0(A0& row, A1& begin, A2& input,
                                                                              A3& out)
{
    constexpr uint32_t ubSlot = I % InSlots;
    constexpr event_t vEv =
        ubSlot == 0U ? EVENT_ID0 : (ubSlot == 1U ? EVENT_ID1 : (ubSlot == 2U ? EVENT_ID2 : EVENT_ID3));
    constexpr uint32_t CurNR = CurTM / 2U;
    SetFlag<HardEvent::V_MTE3>(vEv);
    WaitFlag<HardEvent::V_MTE3>(vEv);
    DataCopyParams op{CurNR, N / 16U, 0, (D - N) / 16U};
    ::SharedPrimitive1908ccb961<D, Iterations, Groups, InSlots, OutSlots, I>(y_, col_, grp_, begin, op, out,
                                                                             row);
    ::SharedPrefetchTile<I, Iterations, InSlots, Plane, Cap, N, (I < 5U) ? 64U : 32U, false, true>(
        input, slots_, core_);
}
#endif
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN,
          uint32_t TailIterations, uint32_t FullCores>
template <uint32_t I, uint32_t CurIter, class A0, class A1, class A2, class A3>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN, TailIterations, FullCores>::EpilogueSharedStage0(
    A0& row, A1& begin, A2& input, A3& out)
{
    constexpr uint32_t slot = I % 4U;
    constexpr event_t vEv =
        slot == 0U ? EVENT_ID0 : (slot == 1U ? EVENT_ID1 : (slot == 2U ? EVENT_ID2 : EVENT_ID3));

    SetFlag<HardEvent::V_MTE3>(vEv);
    if constexpr (I + 1U < CurIter) {
        constexpr uint32_t nextSlot = (I + 1U) % 4U;
        constexpr event_t nextEv =
            nextSlot == 0U ? EVENT_ID0
                           : (nextSlot == 1U ? EVENT_ID1 : (nextSlot == 2U ? EVENT_ID2 : EVENT_ID3));
        ::SharedWaitReadyfe305ae3<nextSlot>();
        if constexpr (I + 1U >= 4U) {
            WaitFlag<HardEvent::V_MTE2>(nextEv);
        }
        const uint64_t nextOff = (core_ * 4U + nextSlot) * 2U * Plane;
        DataCopy(input[nextSlot * 2U * Cap], slots_[nextOff + begin * 2U * N], 2U * Cap);
        if constexpr (I + 5U < CurIter) {
            ::SharedConsumed144332b4<nextSlot>();
        }
        SetFlag<HardEvent::MTE2_V>(nextEv);
    }
    WaitFlag<HardEvent::V_MTE3>(vEv);
    ::SharedPrimitive9cb48942a1<D, N, NR, Cap, I, CurIter>(y_, col_, begin, out, row);
}
#endif
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, uint32_t CurTM, class A0>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::EpilogueSharedStage0(A0& input)
{
    constexpr uint32_t ubSlot = I % InSlots;
    constexpr event_t vEv =
        ubSlot == 0U ? EVENT_ID0 : (ubSlot == 1U ? EVENT_ID1 : (ubSlot == 2U ? EVENT_ID2 : EVENT_ID3));
    if constexpr (I + OutSlots < Iterations) {
        SetFlag<HardEvent::MTE3_V>(vEv);
    }
    if constexpr (I + 1U < Iterations) {
        constexpr uint32_t nextSlot = (I + 1U) % 4U;
        constexpr uint32_t nextUbSlot = (I + 1U) % InSlots;
        constexpr event_t nextEv =
            nextUbSlot == 0U ? EVENT_ID0
                             : (nextUbSlot == 1U ? EVENT_ID1 : (nextUbSlot == 2U ? EVENT_ID2 : EVENT_ID3));
        const uint64_t nextOff = ::SharedWaitNextTile<I, InSlots, Plane>(core_);
        if constexpr (Groups == 4U) {
            constexpr uint32_t nextNR = (I == 0U) ? 32U : 16U;
            DataCopy(input[nextUbSlot * 2U * Cap], slots_[nextOff + GetSubBlockIdx() * nextNR * 2U * N],
                     2U * nextNR * N);
        } else if constexpr (Groups == 10U) {
            constexpr uint32_t nextNR = (I < 5U) ? 64U : 32U;
            DataCopy(input[nextUbSlot * 2U * Cap], slots_[nextOff + GetSubBlockIdx() * nextNR * 2U * N],
                     2U * nextNR * N);
        } else {
            DataCopy(input[nextUbSlot * 2U * Cap], slots_[nextOff + GetSubBlockIdx() * NR * 2U * N],
                     2U * Cap);
        }
        if constexpr (I + 5U < Iterations) {
            ::SharedConsumed245076b5<nextSlot>();
        }
        SetFlag<HardEvent::MTE2_V>(nextEv);
    }
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN,
          uint32_t TailIterations, uint32_t FullCores>
template <uint32_t I, uint32_t CurIter, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN, TailIterations, FullCores>::EpilogueStage23fd9fc7(
    A0& row, A1& begin, A2& input, A3& g, A4& u, A5& sig, A6& out)
{
    constexpr uint32_t slot = I % 4U;
    constexpr event_t vEv =
        slot == 0U ? EVENT_ID0 : (slot == 1U ? EVENT_ID1 : (slot == 2U ? EVENT_ID2 : EVENT_ID3));
    constexpr uint64_t mask = 0;
    LocalTensor<T> ones(TPosition::VECCALC, ((2U * InSlots + 3U + OutSlots) * Cap + 2U * N) * sizeof(T),
                        128U);
    constexpr uint8_t repeats = Cap / 128U;
    const UnaryRepeatParams ur{1, 1, 8, 8};
    const BinaryRepeatParams br{1, 1, 1, 8, 8, 8};
    const BinaryRepeatParams broadcast{1, 1, 1, 8, 0, 8};
    if constexpr (N != 128U) {
        SetVectorMask<T, MaskMode::NORMAL>(~0ULL, ~0ULL);
    }
    Muls<T, false>(sig, g, static_cast<T>(-1), mask, repeats, ur);
    Exp<T, false>(sig, sig, mask, repeats, ur);
    Adds<T, false>(sig, sig, static_cast<T>(1), mask, repeats, ur);
    Div<T, false>(sig, ones, sig, mask, repeats, broadcast);
    if constexpr (I >= 4U) {
        WaitFlag<HardEvent::MTE3_V>(vEv);
    }
    Mul<T, false>(out, g, sig, mask, repeats, br);
    Mul<T, false>(out, out, u, mask, repeats, br);
    if constexpr (!FastAiv) {
        ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
        ::SharedPrimitive9cb48942a1<D, N, NR, Cap, I, CurIter>(y_, col_, begin, out, row);
    } else {
        EpilogueSharedStage0<I, CurIter>(row, begin, input, out);
    }
}

#endif

}
