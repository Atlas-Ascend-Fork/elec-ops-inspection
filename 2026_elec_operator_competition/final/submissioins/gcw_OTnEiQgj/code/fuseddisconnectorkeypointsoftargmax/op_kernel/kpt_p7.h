/*
 * Copyright (c) 2026 gcw_OTnEiQgj.
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

#include "kpt_p1.h"
#include "kpt_p2.h"
#include "kpt_p3.h"
#include "kpt_p5.h"

namespace elec::impl {

struct TapGroup9 {
    float w0, w1, w2, w3, w4, w5, w6, w7, w8;
};

struct PlaneSet9 {
    __ubuf__ float *pY0, *pY1, *pY2, *pY3, *pY4, *pY5, *pY6, *pY7, *pY8;
    uint8_t reps0, reps1, reps2;
};

KPT_INL TapGroup9 ReadTaps9(const __ubuf__ float *p) {
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]};
}

template <bool FIRST_P>
KPT_INL void ApplyTap1(__ubuf__ float *dst, __ubuf__ float *src, float w, uint8_t reps) {
    if constexpr (FIRST_P) {
        OpVmuls(dst, src, w, reps);
    } else {
        OpVaxpy(dst, src, w, reps);
    }
}

template <bool FIRST_P, bool ALT_ORD>
KPT_INL void ApplyTap9(
    const PlaneSet9 &yp, __ubuf__ float *s0, __ubuf__ float *s1, __ubuf__ float *s2, const TapGroup9 &kw) {
    if constexpr (ALT_ORD) {
        ApplyTap1<FIRST_P>(yp.pY0, s0, kw.w0, yp.reps0);
        ApplyTap1<FIRST_P>(yp.pY5, s1, kw.w5, yp.reps1);
        ApplyTap1<FIRST_P>(yp.pY1, s0, kw.w1, yp.reps0);
        ApplyTap1<FIRST_P>(yp.pY6, s2, kw.w6, yp.reps2);
        ApplyTap1<FIRST_P>(yp.pY2, s0, kw.w2, yp.reps0);
        ApplyTap1<FIRST_P>(yp.pY7, s2, kw.w7, yp.reps2);
        ApplyTap1<FIRST_P>(yp.pY3, s1, kw.w3, yp.reps1);
        ApplyTap1<FIRST_P>(yp.pY8, s2, kw.w8, yp.reps2);
        ApplyTap1<FIRST_P>(yp.pY4, s1, kw.w4, yp.reps1);
    } else {
        ApplyTap1<FIRST_P>(yp.pY0, s0, kw.w0, yp.reps0);
        ApplyTap1<FIRST_P>(yp.pY1, s0, kw.w1, yp.reps0);
        ApplyTap1<FIRST_P>(yp.pY2, s0, kw.w2, yp.reps0);
        ApplyTap1<FIRST_P>(yp.pY3, s1, kw.w3, yp.reps1);
        ApplyTap1<FIRST_P>(yp.pY4, s1, kw.w4, yp.reps1);
        ApplyTap1<FIRST_P>(yp.pY5, s1, kw.w5, yp.reps1);
        ApplyTap1<FIRST_P>(yp.pY6, s2, kw.w6, yp.reps2);
        ApplyTap1<FIRST_P>(yp.pY7, s2, kw.w7, yp.reps2);
        ApplyTap1<FIRST_P>(yp.pY8, s2, kw.w8, yp.reps2);
    }
}

template <typename CfgT, uint32_t W, int32_t FIX_TOP, uint32_t FIX_R>
KPT_INL void ZeroTailPad(const AscendC::LocalTensor<float> &ubBase, uint32_t isTop, uint32_t M) {
    constexpr uint32_t PAD_A0 = (FIX_TOP == 1) ? CfgT::SlotY(3U) : CfgT::SlotY(0U);
    constexpr uint32_t PAD_C0 = (FIX_TOP == 1) ? CfgT::SlotY(5U) : CfgT::SlotY(2U);
    constexpr uint32_t PAD_M = FIX_R * W;
    __ubuf__ float *pTailA = nullptr, *pTailC = nullptr;
    if constexpr (FIX_TOP >= 0 && FIX_R > 0U) {
        pTailA = reinterpret_cast<__ubuf__ float *>(ubBase[PAD_A0 + PAD_M].GetPhyAddr());
        pTailC = reinterpret_cast<__ubuf__ float *>(ubBase[PAD_C0 + PAD_M].GetPhyAddr());
    } else if constexpr (FIX_TOP >= 0) {
        pTailA = reinterpret_cast<__ubuf__ float *>(ubBase[PAD_A0].GetPhyAddr()) + M;
        pTailC = reinterpret_cast<__ubuf__ float *>(ubBase[PAD_C0].GetPhyAddr()) + M;
    } else {
        pTailA = reinterpret_cast<__ubuf__ float *>(ubBase[isTop ? CfgT::SlotY(3U) : CfgT::SlotY(0U)].GetPhyAddr()) + M;
        pTailC = reinterpret_cast<__ubuf__ float *>(ubBase[isTop ? CfgT::SlotY(5U) : CfgT::SlotY(2U)].GetPhyAddr()) + M;
    }
    set_vector_mask(0ULL, 0xFFULL);
    OpVdup(pTailA, 0.0f, 1U);
    OpVdup(pTailC, 0.0f, 1U);
    set_vector_mask(~0ULL, ~0ULL);
}

template <typename CfgT>
KPT_INL void ZeroInTail(
    const AscendC::LocalTensor<float> &ubFeat0, const AscendC::LocalTensor<float> &ubFeat1,
    const AscendC::LocalTensor<float> &ubFeat2, uint32_t loadFloats, uint32_t vecFloats) {
    if (vecFloats <= loadFloats) {
        return;
    }
    constexpr uint32_t SPAN_F = CfgT::SPAN_F;
    const uint32_t padFloats = vecFloats - loadFloats;
    const uint8_t fullPadReps = static_cast<uint8_t>(padFloats >> 6U);
    __ubuf__ float *pPad0 = reinterpret_cast<__ubuf__ float *>(ubFeat0.GetPhyAddr()) + loadFloats;
    __ubuf__ float *pPad1 = reinterpret_cast<__ubuf__ float *>(ubFeat1.GetPhyAddr()) + loadFloats;
    if (fullPadReps > 0U) {
        OpVdup(pPad0, 0.0f, fullPadReps);
        OpVdup(pPad0 + SPAN_F, 0.0f, fullPadReps);
        OpVdup(pPad1, 0.0f, fullPadReps);
        OpVdup(pPad1 + SPAN_F, 0.0f, fullPadReps);
        if constexpr (CfgT::OPT_F) {
            __ubuf__ float *pPad2 = reinterpret_cast<__ubuf__ float *>(ubFeat2.GetPhyAddr()) + loadFloats;
            OpVdup(pPad2, 0.0f, fullPadReps);
            OpVdup(pPad2 + SPAN_F, 0.0f, fullPadReps);
        }
    }
    if ((padFloats & 32U) != 0U) {
        const uint32_t tailOff = padFloats & ~63U;
        set_vector_mask(0ULL, 0xFFFFFFFFULL);
        OpVdup(pPad0 + tailOff, 0.0f, 1U);
        OpVdup(pPad0 + SPAN_F + tailOff, 0.0f, 1U);
        OpVdup(pPad1 + tailOff, 0.0f, 1U);
        OpVdup(pPad1 + SPAN_F + tailOff, 0.0f, 1U);
        if constexpr (CfgT::OPT_F) {
            __ubuf__ float *pPad2 = reinterpret_cast<__ubuf__ float *>(ubFeat2.GetPhyAddr()) + loadFloats;
            OpVdup(pPad2 + tailOff, 0.0f, 1U);
            OpVdup(pPad2 + SPAN_F + tailOff, 0.0f, 1U);
        }
        set_vector_mask(~0ULL, ~0ULL);
    }
}

template <typename CfgT, uint32_t HW, uint32_t FIX_LD>
KPT_INL AscendC::DataCopyParams MakeGrpCopy(uint32_t loadFloats) {
    constexpr uint32_t G = CfgT::GRP_W;
    const uint32_t effLoad = (FIX_LD > 0U) ? FIX_LD : loadFloats;
    return AscendC::DataCopyParams{
        static_cast<uint16_t>(G), static_cast<uint16_t>(effLoad >> 3U),
        (G > 1U) ? static_cast<uint16_t>((HW - effLoad) >> 3U) : static_cast<uint16_t>(0U),
        (G > 1U) ? static_cast<uint16_t>((CfgT::SPAN_F - effLoad) >> 3U) : static_cast<uint16_t>(0U)};
}

template <typename CfgT, bool CST_TOP, uint32_t SHF_1, uint32_t SHF_2>
KPT_INL void RunGrpStep(
    __ubuf__ float *fBase, const TapGroup9 *kw, const PlaneSet9 &yp,
    uint32_t p, uint32_t shift1, uint32_t shift2) {
    constexpr uint32_t G = CfgT::GRP_W;
    constexpr bool ALT_ORD = CfgT::OPT_J;
    #pragma unroll
    for (uint32_t c = 0U; c < G; ++c) {
        __ubuf__ float * const f = fBase + c * CfgT::SPAN_F;
        __ubuf__ float * const f1 = CST_TOP ? (f + SHF_1) : (f + shift1);
        __ubuf__ float * const f2 = CST_TOP ? (f + SHF_2) : (f + shift2);
        if (p == 0U && c == 0U) {
            ApplyTap9<true, ALT_ORD>(yp, f, f1, f2, kw[c]);
        } else {
            ApplyTap9<false, ALT_ORD>(yp, f, f1, f2, kw[c]);
        }
        if constexpr (CfgT::OPT_H) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

template <typename CfgT, uint32_t HW, uint32_t FIX_LD, bool CST_TOP, uint32_t SHF_1, uint32_t SHF_2>
KPT_INL void ExecGrpPipe(
    const AscendC::GlobalTensor<float> &gmFeat, const AscendC::LocalTensor<float> &ubBase,
    const __ubuf__ float *pW, __ubuf__ float *pF0_0, const PlaneSet9 &yp,
    uint32_t inBaseOff, uint32_t loadFloats, uint32_t shift1, uint32_t shift2) {
    constexpr uint32_t G = CfgT::GRP_W;
    constexpr uint32_t N = CfgT::GRP_D;
    static_assert(G > 0U && N >= 2U && N <= 4U && (DIM_C0 % G) == 0U, "grp_cfg");
    constexpr uint32_t GROUPS = DIM_C0 / G;
    const AscendC::DataCopyParams grpCopy = MakeGrpCopy<CfgT, HW, FIX_LD>(loadFloats);
    #pragma unroll
    for (uint32_t p = 0U; p < GROUPS; ++p) {
        const uint32_t slot = p % N;
        const event_t curEv = static_cast<event_t>(slot);
        TapGroup9 kw[G];
        #pragma unroll
        for (uint32_t c = 0U; c < G; ++c) {
            kw[c] = ReadTaps9(pW + (p * G + c) * DIM_K9);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(curEv);
        RunGrpStep<CfgT, CST_TOP, SHF_1, SHF_2>(
            pF0_0 + slot * (G * CfgT::SPAN_F), kw, yp, p, shift1, shift2);
        if (p + N < GROUPS) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(curEv);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(curEv);
            AscendC::DataCopy(
                ubBase[CfgT::OFF_X0 + slot * G * CfgT::SPAN_F],
                gmFeat[inBaseOff + (p + N) * G * HW], grpCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(curEv);
        }
    }
    if constexpr (!CfgT::OPT_H) {
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename CfgT, bool FIXED_REP>
KPT_INL void CondPipeSync(uint8_t reps) {
    if constexpr (CfgT::OPT_H) {
        AscendC::PipeBarrier<PIPE_V>();
    } else if constexpr (!FIXED_REP) {
        if (reps < 4U) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

template <typename CfgT, bool FIXED_REP, bool CST_TOP, uint32_t SHF_1, uint32_t SHF_2>
KPT_INL void RunPairStep(
    const PlaneSet9 &yp, __ubuf__ float *fA, const TapGroup9 &kw0, const TapGroup9 &kw1,
    uint32_t p, uint32_t shift1, uint32_t shift2, uint8_t reps) {
    constexpr bool ALT_ORD = CfgT::OPT_J;
    __ubuf__ float * const fB = fA + CfgT::SPAN_F;
    const uint32_t s1 = CST_TOP ? SHF_1 : (FIXED_REP ? shift1 : 0U);
    const uint32_t s2 = CST_TOP ? SHF_2 : (FIXED_REP ? shift2 : 0U);
    if (p == 0U) {
        ApplyTap9<true, ALT_ORD>(yp, fA, fA + s1, fA + s2, kw0);
    } else {
        ApplyTap9<false, ALT_ORD>(yp, fA, fA + s1, fA + s2, kw0);
    }
    CondPipeSync<CfgT, FIXED_REP>(reps);
    ApplyTap9<false, ALT_ORD>(yp, fB, fB + s1, fB + s2, kw1);
    CondPipeSync<CfgT, FIXED_REP>(reps);
}

template <typename CfgT, bool BUF3_M, uint32_t HW, bool FIXED_REP, bool CST_TOP, uint32_t SHF_1, uint32_t SHF_2>
KPT_INL void ExecPairPipe(
    const AscendC::GlobalTensor<float> &gmFeat, const AscendC::LocalTensor<float> &ubFeat0,
    const AscendC::LocalTensor<float> &ubFeat1, const AscendC::LocalTensor<float> &ubFeat2,
    const AscendC::DataCopyParams &pairCopy, const __ubuf__ float *pW, const PlaneSet9 &yp,
    uint32_t inBaseOff, uint32_t shift1, uint32_t shift2, uint8_t reps) {
    constexpr uint32_t SLOT_N = BUF3_M ? 3U : 2U;
    __ubuf__ float * const pF0_0 = reinterpret_cast<__ubuf__ float *>(ubFeat0.GetPhyAddr());
    __ubuf__ float * const pF1_0 = reinterpret_cast<__ubuf__ float *>(ubFeat1.GetPhyAddr());
    __ubuf__ float * const pF2_0 = BUF3_M ? reinterpret_cast<__ubuf__ float *>(ubFeat2.GetPhyAddr()) : nullptr;
    #pragma unroll
    for (uint32_t p = 0U; p < 16U; ++p) {
        const uint32_t bIdx = BUF3_M ? (p % 3U) : (p & 1U);
        const event_t curEv = (bIdx == 0U) ? EVENT_ID0 : ((bIdx == 1U) ? EVENT_ID1 : EVENT_ID2);
        __ubuf__ float * const fA = (bIdx == 0U) ? pF0_0 : ((bIdx == 1U) ? pF1_0 : pF2_0);
        const uint32_t wOff = (p * 2U) * DIM_K9;
        const TapGroup9 kw0 = ReadTaps9(pW + wOff);
        const TapGroup9 kw1 = ReadTaps9(pW + wOff + 9U);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(curEv);
        RunPairStep<CfgT, FIXED_REP, CST_TOP, SHF_1, SHF_2>(
            yp, fA, kw0, kw1, p, shift1, shift2, reps);
        if (p + SLOT_N < 16U) {
            const uint32_t nextOff = inBaseOff + ((p + SLOT_N) * 2U) * HW;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(curEv);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(curEv);
            if (bIdx == 0U) {
                AscendC::DataCopy(ubFeat0, gmFeat[nextOff], pairCopy);
            } else if (!BUF3_M || bIdx == 1U) {
                AscendC::DataCopy(ubFeat1, gmFeat[nextOff], pairCopy);
            } else {
                AscendC::DataCopy(ubFeat2, gmFeat[nextOff], pairCopy);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(curEv);
        }
    }
    if constexpr (!CfgT::OPT_H) {
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <uint32_t W, bool FIXED_REP, int32_t FIX_TOP, int32_t FIX_BOT, uint32_t FIX_R>
KPT_INL void CalcPlaneReps(
    uint32_t isTop, uint32_t isBot, uint32_t M, uint32_t shift2, uint8_t reps,
    uint8_t &outReps0, uint8_t &outReps1, uint8_t &outReps2) {
    if constexpr (!FIXED_REP) {
        outReps0 = reps;
        outReps1 = reps;
        outReps2 = reps;
        return;
    }
    const uint32_t effTop = (FIX_TOP >= 0) ? ((FIX_TOP > 0) ? 1U : 0U) : isTop;
    const uint32_t effBot = (FIX_BOT >= 0) ? ((FIX_BOT > 0) ? 1U : 0U) : isBot;
    const uint32_t effM = (FIX_R > 0U) ? (FIX_R * W) : M;
    const uint8_t mReps = static_cast<uint8_t>(effM >> 6U);
    outReps0 = (effTop != 0U) ? static_cast<uint8_t>((effM - W + 63U) >> 6U) : mReps;
    outReps1 = (effTop != 0U) ? mReps : static_cast<uint8_t>(((W - 64U) + effM + 63U) >> 6U);
    const uint32_t row2Len = (effBot != 0U) ? (effM - W) : effM;
    outReps2 = static_cast<uint8_t>((((2U - effTop) * W - shift2) + row2Len + 63U) >> 6U);
}

template <typename CfgT, uint32_t W, bool FIXED_REP, int32_t FIX_TOP, int32_t FIX_BOT, uint32_t FIX_R>
KPT_INL PlaneSet9 BuildPlaneSet(
    const AscendC::LocalTensor<float> &ubBase, uint32_t isTop, uint32_t isBot, uint32_t M, uint8_t reps,
    uint32_t &outShift1, uint32_t &outShift2) {
    constexpr bool CST_TOP = (FIX_TOP >= 0) && FIXED_REP;
    constexpr uint32_t C_TOP = (FIX_TOP > 0) ? 1U : 0U;
    constexpr uint32_t SHF_1 = CST_TOP ? ((C_TOP != 0U) ? 0U : 64U) : 0U;
    constexpr uint32_t SHF_2 = CST_TOP ? (((2U - C_TOP) * W) & ~63U) : 0U;
    outShift1 = CST_TOP ? SHF_1 : (FIXED_REP ? (isTop ? 0U : 64U) : 0U);
    outShift2 = CST_TOP ? SHF_2 : (FIXED_REP ? (((2U - isTop) * W) & ~63U) : 0U);
    uint8_t reps0 = 0U, reps1 = 0U, reps2 = 0U;
    CalcPlaneReps<W, FIXED_REP, FIX_TOP, FIX_BOT, FIX_R>(
        isTop, isBot, M, outShift2, reps, reps0, reps1, reps2);
    __ubuf__ float *pY3 = nullptr, *pY4 = nullptr, *pY5 = nullptr, *pY6 = nullptr, *pY7 = nullptr, *pY8 = nullptr;
    if constexpr (CST_TOP) {
        pY3 = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(3U) + SHF_1].GetPhyAddr());
        pY4 = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(4U) + SHF_1].GetPhyAddr());
        pY5 = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(5U) + SHF_1].GetPhyAddr());
        pY6 = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(6U) + SHF_2].GetPhyAddr());
        pY7 = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(7U) + SHF_2].GetPhyAddr());
        pY8 = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(8U) + SHF_2].GetPhyAddr());
    } else {
        pY3 = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(3U)].GetPhyAddr()) + outShift1;
        pY4 = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(4U)].GetPhyAddr()) + outShift1;
        pY5 = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(5U)].GetPhyAddr()) + outShift1;
        pY6 = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(6U)].GetPhyAddr()) + outShift2;
        pY7 = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(7U)].GetPhyAddr()) + outShift2;
        pY8 = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(8U)].GetPhyAddr()) + outShift2;
    }
    return PlaneSet9{
        reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(0U)].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(1U)].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::SlotY(2U)].GetPhyAddr()),
        pY3, pY4, pY5, pY6, pY7, pY8, reps0, reps1, reps2};
}

template <typename CfgT, uint32_t HW, uint32_t FIX_LD>
KPT_INL void FetchInitSlot(
    const AscendC::GlobalTensor<float> &gmFeat, const AscendC::LocalTensor<float> &ubBase,
    const AscendC::LocalTensor<float> &ubFeat0, const AscendC::DataCopyParams &pairCopy,
    uint32_t inBaseOff, uint32_t loadFloats) {
    if constexpr (CfgT::OPT_K) {
        constexpr uint32_t G = CfgT::GRP_W;
        constexpr uint32_t INIT_N = (CfgT::GRP_D <= 2U) ? CfgT::GRP_D : 2U;
        const AscendC::DataCopyParams grpInit = MakeGrpCopy<CfgT, HW, FIX_LD>(loadFloats);
        #pragma unroll
        for (uint32_t slot = 0U; slot < INIT_N; ++slot) {
            AscendC::DataCopy(
                ubBase[CfgT::OFF_X0 + slot * G * CfgT::SPAN_F],
                gmFeat[inBaseOff + slot * G * HW], grpInit);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(slot));
        }
    } else {
        AscendC::DataCopy(ubFeat0, gmFeat[inBaseOff], pairCopy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }
}

template <typename CfgT, uint32_t HW, uint32_t FIX_LD>
KPT_INL void FetchNextSlots(
    const AscendC::GlobalTensor<float> &gmFeat, const AscendC::LocalTensor<float> &ubBase,
    const AscendC::LocalTensor<float> &ubFeat1, const AscendC::LocalTensor<float> &ubFeat2,
    const AscendC::DataCopyParams &pairCopy, uint32_t inBaseOff, uint32_t loadFloats) {
    if constexpr (CfgT::OPT_K) {
        constexpr uint32_t G = CfgT::GRP_W;
        constexpr uint32_t N = CfgT::GRP_D;
        if constexpr (N > 2U) {
            const AscendC::DataCopyParams grpNext = MakeGrpCopy<CfgT, HW, FIX_LD>(loadFloats);
            #pragma unroll
            for (uint32_t slot = 2U; slot < N; ++slot) {
                AscendC::DataCopy(
                    ubBase[CfgT::OFF_X0 + slot * G * CfgT::SPAN_F],
                    gmFeat[inBaseOff + slot * G * HW], grpNext);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(slot));
            }
        }
    } else {
        AscendC::DataCopy(ubFeat1, gmFeat[inBaseOff + 2U * HW], pairCopy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        if constexpr (CfgT::OPT_F) {
            AscendC::DataCopy(ubFeat2, gmFeat[inBaseOff + 4U * HW], pairCopy);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        }
    }
}

template <typename CfgT, uint32_t W, bool FIXED_REP, int32_t FIX_TOP, uint32_t FIX_R>
KPT_INL void SetupStageState(
    const AscendC::LocalTensor<float> &ubBase, const AscendC::LocalTensor<float> &ubFeat0,
    const AscendC::LocalTensor<float> &ubFeat1, const AscendC::LocalTensor<float> &ubFeat2,
    bool doInitSoc, uint32_t v0, uint32_t isTop, uint32_t loadFloats, uint32_t vecFloats, uint32_t M, uint32_t R) {
    if (doInitSoc) {
        AscendC::InitSocState();
    } else {
        AscendC::SetMaskNorm();
        set_vector_mask(~0ULL, ~0ULL);
    }
    if constexpr (CfgT::OPT_G && FIXED_REP) {
        ZeroTailPad<CfgT, W, FIX_TOP, FIX_R>(ubBase, isTop, M);
    }
    if constexpr (!FIXED_REP) {
        ZeroInTail<CfgT>(ubFeat0, ubFeat1, ubFeat2, loadFloats, vecFloats);
    }
    AscendC::LocalTensor<float> ubSeqF = ubBase[368U];
    AscendC::LocalTensor<int32_t> ubSeqI = ubBase[376U].ReinterpretCast<int32_t>();
    InitBaseSeq(ubSeqF, ubSeqI);
    if constexpr (CfgT::OPT_G) {
        PrepAxisFull<W, FIXED_REP, (CfgT::M_LIM > 2560U)>(
            ubBase[CfgT::OFF_CU], ubBase[CfgT::OFF_CV],
            ubBase[CfgT::OFF_IL], ubBase[CfgT::OFF_IR], ubSeqF, ubSeqI, v0, M, R);
    } else {
        PrepAxisCompact<W, CfgT::LEN_SD>(
            ubBase[CfgT::OFF_CU], ubBase[CfgT::OFF_CV],
            ubBase[CfgT::OFF_SD].template ReinterpretCast<int32_t>(), ubSeqF, ubSeqI, v0);
    }
}

template <
    typename CfgT, uint32_t W, uint32_t HW, bool FIXED_REP,
    int32_t FIX_TOP, int32_t FIX_BOT, uint32_t FIX_R, uint32_t FIX_LD, uint32_t SYNC_TAG>
KPT_INL void ExecMainPipe(
    const AscendC::GlobalTensor<float> &gmFeat, const AscendC::LocalTensor<float> &ubBase,
    const AscendC::LocalTensor<float> &ubFeat0, const AscendC::LocalTensor<float> &ubFeat1,
    const AscendC::LocalTensor<float> &ubFeat2, const AscendC::DataCopyParams &pairCopy,
    const __ubuf__ float *pW, uint32_t inBaseOff, uint32_t loadFloats,
    uint32_t isTop, uint32_t isBot, uint32_t M, uint8_t reps) {
    FetchNextSlots<CfgT, HW, FIX_LD>(gmFeat, ubBase, ubFeat1, ubFeat2, pairCopy, inBaseOff, loadFloats);
    if constexpr (SYNC_TAG == 20U) {
        InitSyncGates<2U, false>(0U);
    } else if constexpr (SYNC_TAG == 21U) {
        InitSyncGates<2U, false>(1U);
    }
    uint32_t shift1 = 0U, shift2 = 0U;
    const PlaneSet9 yp = BuildPlaneSet<CfgT, W, FIXED_REP, FIX_TOP, FIX_BOT, FIX_R>(
        ubBase, isTop, isBot, M, reps, shift1, shift2);
    constexpr bool CST_TOP = (FIX_TOP >= 0) && FIXED_REP;
    constexpr uint32_t C_TOP = (FIX_TOP > 0) ? 1U : 0U;
    constexpr uint32_t SHF_1 = CST_TOP ? ((C_TOP != 0U) ? 0U : 64U) : 0U;
    constexpr uint32_t SHF_2 = CST_TOP ? (((2U - C_TOP) * W) & ~63U) : 0U;
    if constexpr (CfgT::OPT_K) {
        ExecGrpPipe<CfgT, HW, FIX_LD, CST_TOP, SHF_1, SHF_2>(
            gmFeat, ubBase, pW, reinterpret_cast<__ubuf__ float *>(ubFeat0.GetPhyAddr()),
            yp, inBaseOff, loadFloats, shift1, shift2);
    } else {
        ExecPairPipe<CfgT, CfgT::OPT_F, HW, FIXED_REP, CST_TOP, SHF_1, SHF_2>(
            gmFeat, ubFeat0, ubFeat1, ubFeat2, pairCopy, pW, yp, inBaseOff, shift1, shift2, reps);
    }
}

template <
    typename CfgT, uint32_t W, uint32_t HW, bool FIXED_REP,
    int32_t FIX_TOP = -1, int32_t FIX_BOT = -1,
    uint32_t FIX_R = 0U, uint32_t FIX_LD = 0U, uint32_t SYNC_TAG = 0U>
KPT_INL bool RunStagePipe(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, bool doInitSoc,
    uint32_t b, uint32_t k, uint32_t v0, uint32_t isTop, uint32_t isBot,
    uint32_t rStart, uint32_t loadFloats, uint32_t vecFloats, uint8_t reps, uint32_t M, uint32_t R) {
    static_assert(
        CfgT::SlotY(7U) == CfgT::SlotY(6U) + CfgT::SPAN_Y &&
        CfgT::SlotY(8U) == CfgT::SlotY(7U) + CfgT::SPAN_Y &&
        3U * CfgT::SPAN_Y >= (2U * DIM_WT + 64U) && (CfgT::SlotY(6U) & 63U) == 0U, "buf_cfg");
    AscendC::GlobalTensor<float> gmFeat, gmWeight;
    gmFeat.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(feat));
    gmWeight.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight));
    AscendC::LocalTensor<float> ubBase(AscendC::TPosition::VECCALC, 0U, 49152U);
    AscendC::LocalTensor<float> ubWeight = ubBase[64U];
    AscendC::LocalTensor<float> ubFeat0 = ubBase[CfgT::OFF_X0];
    AscendC::LocalTensor<float> ubFeat1 = ubBase[CfgT::OFF_X1];
    AscendC::LocalTensor<float> ubFeat2;
    if constexpr (CfgT::OPT_F) {
        ubFeat2 = ubBase[CfgT::OFF_X2];
    }
    const uint32_t inBaseOff = b * DIM_C0 * HW + rStart * W;
    const uint32_t effLoad = (FIX_LD > 0U) ? FIX_LD : loadFloats;
    const AscendC::DataCopyParams pairCopy{
        2U, static_cast<uint16_t>(effLoad >> 3U),
        static_cast<uint16_t>((HW - effLoad) >> 3U), static_cast<uint16_t>((CfgT::SPAN_F - effLoad) >> 3U)};
    AscendC::DataCopy(ubWeight, gmWeight[k * DIM_C0 * DIM_K9], DIM_C0 * DIM_K9);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
    FetchInitSlot<CfgT, HW, FIX_LD>(gmFeat, ubBase, ubFeat0, pairCopy, inBaseOff, loadFloats);
    SetupStageState<CfgT, W, FIXED_REP, FIX_TOP, FIX_R>(
        ubBase, ubFeat0, ubFeat1, ubFeat2, doInitSoc, v0, isTop, loadFloats, vecFloats, M, R);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
    const __ubuf__ float * const pW = reinterpret_cast<const __ubuf__ float *>(ubWeight.GetPhyAddr());
    if (__builtin_expect((reinterpret_cast<const __ubuf__ uint64_t *>(pW)[0] & 0x7FFFFFFF7FFFFFFFULL) == 0ULL, 0)) {
        if (TestParamBuf(weight, bias, CfgT::SlotY(6U))) {
            if constexpr (CfgT::OPT_K && CfgT::GRP_D >= 2U) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
            return true;
        }
    }
    ExecMainPipe<CfgT, W, HW, FIXED_REP, FIX_TOP, FIX_BOT, FIX_R, FIX_LD, SYNC_TAG>(
        gmFeat, ubBase, ubFeat0, ubFeat1, ubFeat2, pairCopy, pW, inBaseOff, loadFloats, isTop, isBot, M, reps);
    return false;
}

}
