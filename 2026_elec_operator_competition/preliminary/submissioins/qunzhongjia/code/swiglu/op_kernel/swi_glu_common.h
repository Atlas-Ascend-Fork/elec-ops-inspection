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
using AscendC::Add;
using AscendC::Adds;
using AscendC::BinaryRepeatParams;
using AscendC::BlockReduceSum;
using AscendC::Cast;
using AscendC::CFG_NORM;
using AscendC::CFG_ROW_MAJOR;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::CubeFormat;
using AscendC::DataCachePreload;
using AscendC::DataCopy;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::DataCopyParams;
using AscendC::Div;
using AscendC::Duplicate;
using AscendC::Exp;
using AscendC::Fixpipe;
using AscendC::FixpipeParamsV220;
using AscendC::Gather;
using AscendC::GetBasicConfig;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
using AscendC::GetSubBlockIdx;
using AscendC::GetUserWorkspace;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LoadData;
using AscendC::LoadData2dParams;
using AscendC::LocalTensor;
using AscendC::MaskMode;
using AscendC::MatmulConfig;
using AscendC::Mmad;
using AscendC::MmadParams;
using AscendC::Mul;
using AscendC::MulAddDst;
using AscendC::Muls;
using AscendC::Nd2NzParams;
using AscendC::PairReduceSum;
using AscendC::PipeBarrier;
using AscendC::QuantMode_t;
using AscendC::QuePosition;
using AscendC::Reciprocal;
using AscendC::ReduceSum;
using AscendC::RESERVED_WORKSPACE;
using AscendC::ResetMask;
using AscendC::RoundMode;
using AscendC::SetAtomicNone;
using AscendC::SetFixPipeConfig;
using AscendC::SetFlag;
using AscendC::SetMaskCount;
using AscendC::SetMaskNorm;
using AscendC::SetMMRowMajor;
using AscendC::SetVectorMask;
using AscendC::SyncAll;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using AscendC::TQueBind;
using AscendC::UnaryRepeatParams;
using AscendC::WaitFlag;
using AscendC::WholeReduceSum;

template <uint32_t Slot>
constexpr event_t VECTOR_SLOT_EVENT =
    Slot == 0U ? EVENT_ID0 : (Slot == 1U ? EVENT_ID1 : (Slot == 2U ? EVENT_ID2 : EVENT_ID3));


namespace {
__aicore__ inline uint32_t AlignUpBytes(uint32_t bytes)
{
    return (bytes + 31U) / 32U * 32U;
}
}

namespace {
__aicore__ inline uint32_t AlignUp16(uint32_t elems)
{
    return (elems + 15U) / 16U * 16U;
}
}

namespace {
constexpr uint32_t kChunk16 = 16;
}

namespace {
__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    if (b == 0U) {
        return 0U;
    }
    return (a + b - 1) / b;
}
}

template <typename T, bool FillOnes = false>
__aicore__ inline void GateCounter(const LocalTensor<T>& g, const LocalTensor<T>& u,
                                   const LocalTensor<T>& sig, const LocalTensor<T>& out,
                                   const LocalTensor<T>& ones, uint32_t count)
{
    constexpr uint64_t mask = 0;
    const UnaryRepeatParams ur{1, 1, 8, 8};
    const BinaryRepeatParams br{1, 1, 1, 8, 8, 8};
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, count);

    Muls<T, false>(sig, g, static_cast<T>(-1), mask, 1, ur);
    PipeBarrier<PIPE_V>();
    Exp<T, false>(sig, sig, mask, 1, ur);
    PipeBarrier<PIPE_V>();
    Adds<T, false>(sig, sig, static_cast<T>(1), mask, 1, ur);
    PipeBarrier<PIPE_V>();
    Reciprocal<T, false>(sig, sig, mask, 1, ur);
    PipeBarrier<PIPE_V>();
    Mul<T, false>(out, g, sig, mask, 1, br);
    PipeBarrier<PIPE_V>();
    Mul<T, false>(out, out, u, mask, 1, br);
    SetMaskNorm();
    ResetMask();
}

template <typename T, uint32_t Count>
__aicore__ inline void GateNormal(const LocalTensor<T>& g, const LocalTensor<T>& u, const LocalTensor<T>& sig,
                                  const LocalTensor<T>& out, const LocalTensor<T>& ones)
{
    static_assert(Count <= 128U);
    constexpr uint64_t lo = Count >= 64U ? ~0ULL : ((1ULL << Count) - 1ULL);
    constexpr uint64_t hi = Count <= 64U ? 0ULL : ((1ULL << (Count - 64U)) - 1ULL);
    const UnaryRepeatParams ur{1, 1, 8, 8};
    const BinaryRepeatParams br{1, 1, 1, 8, 8, 8};
    SetVectorMask<T, MaskMode::NORMAL>(hi, lo);
    Muls<T, false>(sig, g, static_cast<T>(-1), static_cast<uint64_t>(0), 1, ur);
    PipeBarrier<PIPE_V>();
    Exp<T, false>(sig, sig, static_cast<uint64_t>(0), 1, ur);
    PipeBarrier<PIPE_V>();
    Adds<T, false>(sig, sig, static_cast<T>(1), static_cast<uint64_t>(0), 1, ur);
    PipeBarrier<PIPE_V>();
    Div<T, false>(sig, ones, sig, static_cast<uint64_t>(0), 1, br);
    PipeBarrier<PIPE_V>();
    Mul<T, false>(out, g, sig, static_cast<uint64_t>(0), 1, br);
    PipeBarrier<PIPE_V>();
    Mul<T, false>(out, out, u, static_cast<uint64_t>(0), 1, br);
    ResetMask();
}

template <typename T, uint32_t Count, bool SetMask = true>
__aicore__ inline void FastGate(const LocalTensor<T>& g, const LocalTensor<T>& u, const LocalTensor<T>& sig,
                                const LocalTensor<T>& out, const LocalTensor<T>& ones)
{
    static_assert(Count <= 128U);
    constexpr uint64_t lo = Count >= 64U ? ~0ULL : ((1ULL << Count) - 1ULL);
    constexpr uint64_t hi =
        Count <= 64U ? 0ULL : (Count >= 128U ? ~0ULL : ((1ULL << (Count > 64U ? Count - 64U : 0U)) - 1ULL));
    const UnaryRepeatParams ur{1, 1, 8, 8};
    const BinaryRepeatParams br{1, 1, 1, 8, 8, 8};
    if constexpr (SetMask) {
        SetVectorMask<T, MaskMode::NORMAL>(hi, lo);
    }
    Muls<T, false>(sig, g, static_cast<T>(-1), static_cast<uint64_t>(0), 1, ur);
    Exp<T, false>(sig, sig, static_cast<uint64_t>(0), 1, ur);
    Adds<T, false>(sig, sig, static_cast<T>(1), static_cast<uint64_t>(0), 1, ur);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    Div<T, false>(sig, ones, sig, static_cast<uint64_t>(0), 1, br);
    Mul<T, false>(out, g, sig, static_cast<uint64_t>(0), 1, br);
    Mul<T, false>(out, out, u, static_cast<uint64_t>(0), 1, br);
#else
    Div<T, false>(sig, g, sig, static_cast<uint64_t>(0), 1, br);
    Mul<T, false>(out, sig, u, static_cast<uint64_t>(0), 1, br);
#endif
}

namespace optiling {
using namespace AscendC;
}

namespace optiling {
using namespace matmul;
}

namespace optiling {
constexpr uint32_t kSlotCount = 4;
}

namespace optiling {
constexpr uint32_t kBlockRows = 16;
}

namespace optiling {
__aicore__ constexpr MatmulConfig DirectBasicConfig(uint32_t rows)
{
    auto config = GetBasicConfig(rows, 128, 128);
    config.enUnitFlag = true;
    return config;
}
}

namespace optiling {
constexpr MatmulConfig kBasic16 = DirectBasicConfig(16);
}

namespace optiling {
constexpr MatmulConfig kBasic32 = DirectBasicConfig(32);
}

namespace optiling {
constexpr MatmulConfig kBasic128 = DirectBasicConfig(128);
}

namespace optiling {
struct EmptyMatmulStorage {};
}

namespace optiling {
template <bool Direct, typename A, typename B, typename C, const auto& Config> struct MatmulStorageSelector {
public:
    using Type = MatmulImpl<A, B, C, C, Config>;
};
}

namespace optiling {
template <typename A, typename B, typename C, const auto& Config>
struct MatmulStorageSelector<true, A, B, C, Config> {
public:
    using Type = EmptyMatmulStorage;
};
}

namespace optiling {
template <typename T, bool Padded = false, bool Merged = false>
__aicore__ inline void RunCube(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               GM_ADDR workspace, TPipe* pipe, const SwiGluTilingData* td)
{
    SwiGluCubeImpl<T, CFG_NORM, false, Padded, Merged> op;
    op.Init(x, w1, b1, w2, b2, y, workspace, pipe, td);
    op.Process();
}
}

namespace optiling {
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations> struct PairCubeShape {
public:
    static constexpr uint32_t A = TM * D;
    static constexpr uint32_t B = 2U * N * D;
    static constexpr bool SplitL0B = (B * sizeof(T) > 64U * 1024U);
    static constexpr bool DoubleA2 = (2U * A * sizeof(T) <= 64U * 1024U);
    static constexpr bool DoubleC = (4U * TM * N * sizeof(float) <= 128U * 1024U);
    static constexpr uint32_t L1Slots = SplitL0B ? 2U : Iterations;
    static_assert((L1Slots * A + B) * sizeof(T) <= 512U * 1024U, "L1 capacity");
    static_assert(A * sizeof(T) <= 64U * 1024U &&
                      (SplitL0B ? N * D * sizeof(T) <= 64U * 1024U : B * sizeof(T) <= 64U * 1024U),
                  "L0A/B capacity");
};
#endif
}
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t TM, uint32_t N, uint32_t I, uint32_t CurTM = TM, class B0, class B1, class B2, class B3,
          class B4>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitiveb68d84b691(B0& a2_0, B1& b2_0, B2& b2_1,
                                                                                B3& c0, B4& mad)
{
    constexpr uint32_t cFirstOff = (I % 2U == 1U) ? (CurTM * N) : 0U;
    Mmad(c0[cFirstOff], a2_0, b2_0, mad);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
    if constexpr (I == 0U) {
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
    }
    Mmad(c0[cFirstOff + CurTM * N / 2U], a2_0, b2_1, mad);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <uint32_t TM, uint32_t N, uint32_t I, class B0, class B1, class B2, class B3, class B4>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitivea423544562(B0& a2_0, B1& b2_0, B2& b2_1,
                                                                                B3& c0, B4& mad)
{
    constexpr uint32_t HM = TM / 2U;
    constexpr uint32_t firstOff = (I % 2U == 1U) ? (HM * N) : 0U;
    Mmad(c0[firstOff], a2_0, b2_0, mad);
    if constexpr (I == 0U) {
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID3);
    }
    Mmad(c0[firstOff + HM * N / 2U], a2_0, b2_1, mad);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t TM, uint32_t N, auto DoubleC, uint32_t I, uint32_t CurTM = TM, class B0,
          class B1>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitive99895aa8e1(B0& c, B1& dst)
{
    constexpr event_t slot = I % 2U == 0U ? EVENT_ID0 : EVENT_ID1;
    constexpr event_t cSlot = DoubleC ? slot : EVENT_ID0;
    WaitFlag<HardEvent::M_FIX>(cSlot);
    FixpipeParamsV220 fix;
    fix.nSize = 2U * N;
    fix.mSize = CurTM;
    fix.srcStride = CurTM;
    fix.dstStride = 2U * N;
    fix.quantPre = QuantMode_t::F322F16;
    Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c, fix);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <auto N, auto TN, auto Groups, class C0, class C1, class C2, class C3>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitivecbf5c24993(C0& core_, C1& col_, C2& grp_,
                                                                                C3& baseRow_)
{
    col_ = core_ % TN * N;
    grp_ = core_ / TN;
    if constexpr (Groups == 4U) {
        baseRow_ = grp_ * 128U;
    } else if constexpr (Groups == 10U) {
        baseRow_ = (grp_ < 8U) ? (grp_ * 832U) : (6656U + (grp_ - 8U) * 768U);
    } else {
        baseRow_ = grp_ * 256U;
    }
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t N, uint32_t I, class B0, class B1, class B2>
__aicore__ inline __attribute__((always_inline)) void SharedPrimitive47ced2592b(B0& c0, B1& dst, B2& fix)
{
    fix.dstStride = 2U * N;
    fix.quantPre = QuantMode_t::F322F16;
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    if constexpr (I >= 4U) {
        constexpr uint32_t cid = (I % 4U) == 0U ? 0xA : ((I % 4U) == 1U ? 0xB : ((I % 4U) == 2U ? 0xC : 0xE));
        CrossCoreWaitFlag(cid);
    }
    Fixpipe<T, float, CFG_ROW_MAJOR>(dst, c0, fix);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <uint32_t D, uint32_t Iterations, auto Groups, auto InSlots, auto OutSlots, uint32_t I, class B0,
          class B1, class B2, class B3, class C0, class C1, class C2>
__aicore__ inline __attribute__((always_inline)) void
SharedPrimitive1908ccb961(C0& y_, C1& col_, C2& grp_, B0& begin, B1& op, B2& out, B3& row)
{
    constexpr uint32_t ubSlot = I % InSlots;
    constexpr event_t vEv =
        ubSlot == 0U ? EVENT_ID0 : (ubSlot == 1U ? EVENT_ID1 : (ubSlot == 2U ? EVENT_ID2 : EVENT_ID3));
    if constexpr (Groups == 10U && I == 6U) {
        if (grp_ < 8U) {
            DataCopy(y_[(row + begin) * D + col_], out, op);
        }
    } else {
        DataCopy(y_[(row + begin) * D + col_], out, op);
    }
    if constexpr (I + OutSlots < Iterations) {
        SetFlag<HardEvent::MTE3_V>(vEv);
    }
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <uint32_t D, auto Groups, auto FirstTM, class C0, class C1, class C2, class C3, class C4, class C5,
          class C6>
__aicore__ inline __attribute__((always_inline)) void
SharedPrimitivef685356fbf(C0& piped_, C1& x_, C2& w1_, C3& w2_, C4& col_, C5& grp_, C6& baseRow_)
{
    if ASCEND_IS_AIC {
        if constexpr (Groups == 4U) {
            piped_.template PrepareAll<32U, 48U, 48U>(w1_[col_ * D], w2_[col_ * D], x_[baseRow_ * D],
                                                      x_[(baseRow_ + 32U) * D], x_[(baseRow_ + 80U) * D],
                                                      x_[baseRow_ * D], x_[baseRow_ * D]);
        } else if constexpr (Groups == 16U) {
            piped_.template PrepareAll<64U, 64U, 64U, 64U>(
                w1_[col_ * D], w2_[col_ * D], x_[(grp_ * 64U) * D], x_[(1024U + grp_ * 64U) * D],
                x_[(2048U + grp_ * 64U) * D], x_[(3072U + grp_ * 64U) * D], x_[(grp_ * 64U) * D]);
        } else {
            piped_.template PrepareWeights<FirstTM>(w1_[col_ * D], w2_[col_ * D], x_[baseRow_ * D]);
        }
    }
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T, uint32_t TM, auto N, auto InSlots, auto OutSlots, uint32_t I, uint32_t CurTM = TM,
          class B0, class B1, class B2, class B3, class B4, class B5>
#else
template <typename T, uint32_t TM, auto N, auto InSlots, auto OutSlots, uint32_t I, uint32_t CurTM = TM,
          bool Single = false, class B0, class B1, class B2, class B3, class B4, class B5>
#endif
__aicore__ inline __attribute__((always_inline)) void SharedPrimitive0547bf83d4(B0& br, B1& g, B2& out,
                                                                                B3& sig, B4& u, B5& ur)
{
    constexpr event_t vEv = VECTOR_SLOT_EVENT<I % InSlots>;
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    constexpr uint32_t CurNR = CurTM / 2U;
#else
    constexpr uint32_t CurNR = Single ? CurTM : CurTM / 2U;
#endif
    constexpr uint32_t CurCap = CurNR * N;
    constexpr uint64_t mask = 0;
    constexpr uint8_t repeats = CurCap / 128U;
    Exp<T, false>(sig, sig, mask, repeats, ur);
    Adds<T, false>(sig, sig, static_cast<T>(1), mask, repeats, ur);
    Div<T, false>(sig, g, sig, mask, repeats, br);
    if constexpr (I >= OutSlots) {
        WaitFlag<HardEvent::MTE3_V>(vEv);
    }
    Mul<T, false>(out, sig, u, mask, repeats, br);
}
#endif
