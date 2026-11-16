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
#include "kpt_p4.h"
#include "kpt_p5.h"

namespace elec::impl {

KPT_INL void MergeRow0W96(
    __ubuf__ float *pA, __ubuf__ float *pB, __ubuf__ float *pC,
    __ubuf__ float *pY0, __ubuf__ float *pY1, __ubuf__ float *pY2,
    __ubuf__ float *pY3, __ubuf__ float *pY4, __ubuf__ float *pY5,
    uint32_t isTop, uint32_t M) {
    const uint8_t mReps0 = static_cast<uint8_t>(M >> 6U);
    set_vector_mask(~0ULL, ~0ULL);
    if (isTop != 0U) {
        const uint8_t topFull = static_cast<uint8_t>(mReps0 - 2U);
        const uint32_t tailOff = static_cast<uint32_t>(topFull) << 6U;
        OpVadd(pA + 96U, pA + 96U, pY0, topFull);
        OpVadd(pB + 96U, pB + 96U, pY1, topFull);
        OpVadd(pC + 96U, pC + 96U, pY2, topFull);
        set_vector_mask(0ULL, 0xFFFFFFFFULL);
        OpVadd(pA + 96U + tailOff, pA + 96U + tailOff, pY0 + tailOff, 1U);
        OpVadd(pB + 96U + tailOff, pB + 96U + tailOff, pY1 + tailOff, 1U);
        OpVadd(pC + 96U + tailOff, pC + 96U + tailOff, pY2 + tailOff, 1U);
        set_vector_mask(~0ULL, ~0ULL);
    } else {
        OpVadd(pA, pA, pY3 + 96U, mReps0);
        OpVadd(pB, pB, pY4 + 96U, mReps0);
        OpVadd(pC, pC, pY5 + 96U, mReps0);
    }
}

template <uint32_t W, uint32_t S_CST>
KPT_INL void MergeRow0(
    __ubuf__ float *pA, __ubuf__ float *pB, __ubuf__ float *pC,
    const AscendC::LocalTensor<float> &y0, const AscendC::LocalTensor<float> &y1, const AscendC::LocalTensor<float> &y2,
    const AscendC::LocalTensor<float> &y3, const AscendC::LocalTensor<float> &y4, const AscendC::LocalTensor<float> &y5,
    uint32_t isTop, uint32_t M) {
    __ubuf__ float * const pY0 = reinterpret_cast<__ubuf__ float *>(y0.GetPhyAddr());
    __ubuf__ float * const pY1 = reinterpret_cast<__ubuf__ float *>(y1.GetPhyAddr());
    __ubuf__ float * const pY2 = reinterpret_cast<__ubuf__ float *>(y2.GetPhyAddr());
    __ubuf__ float * const pY3 = reinterpret_cast<__ubuf__ float *>(y3.GetPhyAddr());
    __ubuf__ float * const pY4 = reinterpret_cast<__ubuf__ float *>(y4.GetPhyAddr());
    __ubuf__ float * const pY5 = reinterpret_cast<__ubuf__ float *>(y5.GetPhyAddr());

    if constexpr ((S_CST > 0U) && (W == 64U)) {
        const uint8_t mReps0 = static_cast<uint8_t>(M >> 6U);
        set_vector_mask(~0ULL, ~0ULL);
        if (isTop != 0U) {
            const uint8_t topReps = static_cast<uint8_t>(mReps0 - 1U);
            OpVadd(pA + 64U, pA + 64U, pY0, topReps);
            OpVadd(pB + 64U, pB + 64U, pY1, topReps);
            OpVadd(pC + 64U, pC + 64U, pY2, topReps);
        } else {
            OpVadd(pA, pA, pY3 + 64U, mReps0);
            OpVadd(pB, pB, pY4 + 64U, mReps0);
            OpVadd(pC, pC, pY5 + 64U, mReps0);
        }
    } else if constexpr ((S_CST > 0U) && (W == 96U)) {
        MergeRow0W96(pA, pB, pC, pY0, pY1, pY2, pY3, pY4, pY5, isTop, M);
    } else if (isTop != 0U) {
        if (M > W) {
            VecAdd3Span(pA + W, pA + W, pY0, pB + W, pB + W, pY1, pC + W, pC + W, pY2, M - W);
        }
    } else {
        VecAdd3Span(pA, pA, pY3 + W, pB, pB, pY4 + W, pC, pC, pY5 + W, M);
    }
}

template <uint32_t W, uint32_t S_CST>
KPT_INL void MergeRow2Pair(
    __ubuf__ float *pA, __ubuf__ float *pC,
    const AscendC::LocalTensor<float> &y6, const AscendC::LocalTensor<float> &y8,
    uint32_t y2SrcOff, uint32_t row2Floats, uint32_t isBot, uint32_t M) {
    __ubuf__ float * const pY6s = reinterpret_cast<__ubuf__ float *>(y6.GetPhyAddr()) + y2SrcOff;
    __ubuf__ float * const pY8s = reinterpret_cast<__ubuf__ float *>(y8.GetPhyAddr()) + y2SrcOff;
    if constexpr ((S_CST > 0U) && (W == 64U)) {
        const uint8_t row2Reps = static_cast<uint8_t>(row2Floats >> 6U);
        OpVadd(pA, pA, pY6s, row2Reps);
        OpVadd(pC, pC, pY8s, row2Reps);
    } else if constexpr ((S_CST > 0U) && (W == 96U)) {
        if (isBot == 0U) {
            const uint8_t mReps2 = static_cast<uint8_t>(M >> 6U);
            OpVadd(pA, pA, pY6s, mReps2);
            OpVadd(pC, pC, pY8s, mReps2);
        } else {
            const uint8_t botFull = static_cast<uint8_t>((M >> 6U) - 2U);
            const uint32_t tailOff = static_cast<uint32_t>(botFull) << 6U;
            OpVadd(pA, pA, pY6s, botFull);
            OpVadd(pC, pC, pY8s, botFull);
            set_vector_mask(0ULL, 0xFFFFFFFFULL);
            OpVadd(pA + tailOff, pA + tailOff, pY6s + tailOff, 1U);
            OpVadd(pC + tailOff, pC + tailOff, pY8s + tailOff, 1U);
            set_vector_mask(~0ULL, ~0ULL);
        }
    } else if (row2Floats > 0U) {
        VecAdd2Span(pA, pA, pY6s, pC, pC, pY8s, row2Floats);
    }
}

template <uint32_t W, uint32_t S_CST>
KPT_INL void AccMidPlane(
    __ubuf__ float *pB, const AscendC::LocalTensor<float> &y7,
    uint32_t y2SrcOff, uint32_t row2Floats, uint32_t isBot, uint8_t mReps) {
    __ubuf__ float * const pY7s = reinterpret_cast<__ubuf__ float *>(y7.GetPhyAddr()) + y2SrcOff;
    if constexpr ((S_CST > 0U) && (W == 64U)) {
        OpVadd(pB, pB, pY7s, static_cast<uint8_t>(row2Floats >> 6U));
    } else if constexpr ((S_CST > 0U) && (W == 96U)) {
        if (isBot == 0U) {
            OpVadd(pB, pB, pY7s, mReps);
        } else {
            const uint8_t botFull = static_cast<uint8_t>(mReps - 2U);
            const uint32_t tailOff = static_cast<uint32_t>(botFull) << 6U;
            OpVadd(pB, pB, pY7s, botFull);
            set_vector_mask(0ULL, 0xFFFFFFFFULL);
            OpVadd(pB + tailOff, pB + tailOff, pY7s + tailOff, 1U);
            set_vector_mask(~0ULL, ~0ULL);
        }
    } else if (row2Floats > 0U) {
        VecAddSpan(pB, pB, pY7s, row2Floats);
    }
}

template <uint32_t W, uint32_t S_CST>
KPT_INL void GatherHorizMap(
    __ubuf__ float *pA, __ubuf__ float *pB, __ubuf__ float *pC,
    __ubuf__ float *pSL, __ubuf__ float *pSR,
    const AscendC::LocalTensor<float> &mapLBuf, const AscendC::LocalTensor<float> &mapRBuf,
    const AscendC::LocalTensor<float> &y7,
    uint32_t y2SrcOff, uint32_t row2Floats, uint32_t isBot, uint32_t M) {
    __ubuf__ uint32_t *pMap0U = reinterpret_cast<__ubuf__ uint32_t *>(mapLBuf.GetPhyAddr());
    __ubuf__ uint32_t *pMap1U = reinterpret_cast<__ubuf__ uint32_t *>(mapRBuf.GetPhyAddr());
    if ((S_CST > 0U) || ((M & 63U) == 0U)) {
        const uint8_t mReps = static_cast<uint8_t>(M >> 6U);
        set_vector_mask(~0ULL, ~0ULL);
        OpVgat(pSL, pA, pMap0U, mReps);
        OpVgat(pSR, pC, pMap1U, mReps);
        AccMidPlane<W, S_CST>(pB, y7, y2SrcOff, row2Floats, isBot, mReps);
        AscendC::PipeBarrier<PIPE_V>();
        OpVadd(pB, pB, pSL, mReps);
        if (mReps < 18U) {
            AscendC::PipeBarrier<PIPE_V>();
        }
        OpVadd(pB, pB, pSR, mReps);
        if (mReps < 18U) {
            AscendC::PipeBarrier<PIPE_V>();
        }
        return;
    }
    VecGatSpan(pSL, pA, pMap0U, M);
    VecGatSpan(pSR, pC, pMap1U, M);
    if (row2Floats > 0U) {
        VecAddSpan(pB, pB, reinterpret_cast<__ubuf__ float *>(y7.GetPhyAddr()) + y2SrcOff, row2Floats);
    }
    AscendC::PipeBarrier<PIPE_V>();
    VecAddSpan(pB, pB, pSL, M);
    AscendC::PipeBarrier<PIPE_V>();
    VecAddSpan(pB, pB, pSR, M);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename CfgT, uint32_t W>
KPT_INL void GatherHoriz576(
    __ubuf__ float *pA, __ubuf__ float *pB, __ubuf__ float *pC,
    __ubuf__ float *pSL, __ubuf__ float *pSR,
    const AscendC::LocalTensor<float> &ubBase, const AscendC::LocalTensor<float> &y7,
    uint32_t y2SrcOff, uint32_t row2Floats, uint32_t isBot, uint32_t M, uint32_t R) {
    __ubuf__ uint32_t *pMapLU = reinterpret_cast<__ubuf__ uint32_t *>(ubBase[CfgT::OFF_SD].GetPhyAddr());
    __ubuf__ uint32_t *pMapRU = pMapLU + CfgT::LEN_SD;
    const uint8_t mReps = static_cast<uint8_t>(M >> 6U);
    const uint32_t fullChunks = M / 576U;
    const uint32_t remReps = (M - fullChunks * 576U) >> 6U;
    set_vector_mask(~0ULL, ~0ULL);
    for (uint32_t c = 0U; c < fullChunks; ++c) {
        OpVgat(pSL + c * 576U, pA + c * 576U, pMapLU, 9U);
    }
    if (remReps > 0U) {
        OpVgat(pSL + fullChunks * 576U, pA + fullChunks * 576U, pMapLU, static_cast<uint8_t>(remReps));
    }
    for (uint32_t c = 0U; c < fullChunks; ++c) {
        OpVgat(pSR + c * 576U, pC + c * 576U, pMapRU, 9U);
    }
    if (remReps > 0U) {
        OpVgat(pSR + fullChunks * 576U, pC + fullChunks * 576U, pMapRU, static_cast<uint8_t>(remReps));
    }
    AscendC::PipeBarrier<PIPE_V>();
    constexpr uint8_t W_BLK = static_cast<uint8_t>(W >> 3U);
    set_vector_mask(0ULL, 1ULL);
    OpVdupS(reinterpret_cast<__ubuf__ uint32_t *>(pSL), 0U, static_cast<uint8_t>(R), W_BLK);
    set_vector_mask(0ULL, 0x80ULL);
    OpVdupS(reinterpret_cast<__ubuf__ uint32_t *>(pSR) + (W - 8U), 0U, static_cast<uint8_t>(R), W_BLK);
    set_vector_mask(~0ULL, ~0ULL);
    AccMidPlane<W, 1U>(pB, y7, y2SrcOff, row2Floats, isBot, mReps);
    AscendC::PipeBarrier<PIPE_V>();
    OpVadd(pB, pB, pSL, mReps);
    if (mReps < 18U) {
        AscendC::PipeBarrier<PIPE_V>();
    }
    OpVadd(pB, pB, pSR, mReps);
    if (mReps < 18U) {
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename CfgT, uint32_t W>
KPT_INL void GatherHorizVar(
    __ubuf__ float *pA, __ubuf__ float *pB, __ubuf__ float *pC,
    __ubuf__ float *pSL, __ubuf__ float *pSR,
    const AscendC::LocalTensor<float> &ubBase, const AscendC::LocalTensor<float> &y7,
    uint32_t y2SrcOff, uint32_t row2Floats, uint32_t M, uint32_t R) {
    __ubuf__ uint32_t *pMapLU = reinterpret_cast<__ubuf__ uint32_t *>(ubBase[CfgT::OFF_SD].GetPhyAddr());
    __ubuf__ uint32_t *pMapRU = pMapLU + CfgT::LEN_SD;
    set_vector_mask(~0ULL, ~0ULL);
    for (uint32_t off = 0U; off < M; off += CfgT::LEN_SD) {
        const uint32_t chunk = ((M - off) < CfgT::LEN_SD) ? (M - off) : CfgT::LEN_SD;
        VecGatSpan(pSL + off, pA + off, pMapLU, chunk);
        VecGatSpan(pSR + off, pC + off, pMapRU, chunk);
    }
    if (row2Floats > 0U) {
        VecAddSpan(pB, pB, reinterpret_cast<__ubuf__ float *>(y7.GetPhyAddr()) + y2SrcOff, row2Floats);
    }
    AscendC::PipeBarrier<PIPE_V>();
    constexpr uint8_t W_BLK = static_cast<uint8_t>(W >> 3U);
    set_vector_mask(0ULL, 1ULL);
    OpVdupS(reinterpret_cast<__ubuf__ uint32_t *>(pSL), 0U, static_cast<uint8_t>(R), W_BLK);
    set_vector_mask(~0ULL, ~0ULL);
    AscendC::PipeBarrier<PIPE_V>();
    VecAddSpan(pB, pB, pSL, M);
    set_vector_mask(0ULL, 0x80ULL);
    OpVdupS(reinterpret_cast<__ubuf__ uint32_t *>(pSR) + (W - 8U), 0U, static_cast<uint8_t>(R), W_BLK);
    set_vector_mask(~0ULL, ~0ULL);
    AscendC::PipeBarrier<PIPE_V>();
    VecAddSpan(pB, pB, pSR, M);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename CfgT, uint32_t W, uint32_t S_CST>
KPT_INL void CombinePlanes(
    const AscendC::LocalTensor<float> &ubBase,
    const AscendC::LocalTensor<float> &ubA,
    const AscendC::LocalTensor<float> &ubB,
    const AscendC::LocalTensor<float> &ubC,
    uint32_t isTop, uint32_t isBot, uint32_t M, uint32_t R) {
    __ubuf__ float *pA = reinterpret_cast<__ubuf__ float *>(ubA.GetPhyAddr());
    __ubuf__ float *pB = reinterpret_cast<__ubuf__ float *>(ubB.GetPhyAddr());
    __ubuf__ float *pC = reinterpret_cast<__ubuf__ float *>(ubC.GetPhyAddr());

    MergeRow0<W, S_CST>(
        pA, pB, pC,
        ubBase[CfgT::SlotY(0U)], ubBase[CfgT::SlotY(1U)], ubBase[CfgT::SlotY(2U)],
        ubBase[CfgT::SlotY(3U)], ubBase[CfgT::SlotY(4U)], ubBase[CfgT::SlotY(5U)],
        isTop, M);
    if constexpr (CfgT::OPT_G && S_CST == 0U) {
        set_vector_mask(0ULL, 0xFFULL);
        OpVdup(pA + M, 0.0f, 1U);
        OpVdup(pC + M, 0.0f, 1U);
        set_vector_mask(~0ULL, ~0ULL);
    }
    AscendC::PipeBarrier<PIPE_V>();

    const uint32_t y2SrcOff = (2U - isTop) * W;
    const uint32_t row2Floats = isBot ? (M - W) : M;
    MergeRow2Pair<W, S_CST>(
        pA, pC, ubBase[CfgT::SlotY(6U)], ubBase[CfgT::SlotY(8U)], y2SrcOff, row2Floats, isBot, M);
    AscendC::PipeBarrier<PIPE_V>();

    __ubuf__ float *pSL = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::OFF_X0].GetPhyAddr());
    __ubuf__ float *pSR = reinterpret_cast<__ubuf__ float *>(ubBase[CfgT::OFF_X1].GetPhyAddr());
    AscendC::LocalTensor<float> y7 = ubBase[CfgT::SlotY(7U)];
    if constexpr (CfgT::OPT_G) {
        GatherHorizMap<W, S_CST>(
            pA, pB, pC, pSL, pSR, ubBase[CfgT::OFF_IL], ubBase[CfgT::OFF_IR],
            y7, y2SrcOff, row2Floats, isBot, M);
    } else if constexpr (CfgT::OPT_C && S_CST > 0U) {
        GatherHoriz576<CfgT, W>(pA, pB, pC, pSL, pSR, ubBase, y7, y2SrcOff, row2Floats, isBot, M, R);
    } else {
        GatherHorizVar<CfgT, W>(pA, pB, pC, pSL, pSR, ubBase, y7, y2SrcOff, row2Floats, M, R);
    }
}

template <
    typename CfgT, uint32_t W, uint32_t S_CST,
    bool PRE_SYNC = false, bool OPT_R = false, bool NO_SYNC = false>
KPT_INL void RunStagePostBody(
    GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores, float f0, float stride,
    const __gm__ FusedDisconnectorKeypointSoftArgmaxTilingData *tdGm,
    uint32_t s, uint32_t k, uint32_t bk, uint32_t S, uint32_t R, uint32_t v0,
    uint32_t isTop, uint32_t isBot, uint32_t M, uint32_t baseR, uint32_t remR) {
    if constexpr (PRE_SYNC && !NO_SYNC && S_CST >= 2U && S_CST <= 4U) {
        InitSyncGates<S_CST, OPT_R>(s);
    }
    AscendC::GlobalTensor<float> gmKeypoints, gmScores;
    AscendC::GlobalTensor<uint32_t> gmKeypointsU32;
    gmKeypoints.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(keypoints));
    gmKeypointsU32.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(keypoints));
    gmScores.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scores));

    AscendC::LocalTensor<float> ubBase(AscendC::TPosition::VECCALC, 0U, 49152U);
    AscendC::LocalTensor<float> ubOutKp = ubBase[352U];
    AscendC::LocalTensor<uint32_t> ubOutKpU32 = ubOutKp.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<float> ubOutScore = ubBase[360U];
    AscendC::LocalTensor<float> y8 = ubBase[CfgT::SlotY(8U)];
    AscendC::LocalTensor<float> ubTmpVec = CfgT::OPT_E ? y8[0U] : ubBase[384U];
    AscendC::LocalTensor<float> ubAccR = CfgT::OPT_E ? y8[128U] : ubBase[640U];
    AscendC::LocalTensor<float> ubAccRV = CfgT::OPT_E ? y8[192U] : ubBase[704U];

    AscendC::LocalTensor<float> ubA = ubBase[isTop ? CfgT::SlotY(3U) : CfgT::SlotY(0U)];
    AscendC::LocalTensor<float> ubB = ubBase[isTop ? CfgT::SlotY(4U) : CfgT::SlotY(1U)];
    AscendC::LocalTensor<float> ubC = ubBase[isTop ? CfgT::SlotY(5U) : CfgT::SlotY(2U)];
    CombinePlanes<CfgT, W, S_CST>(ubBase, ubA, ubB, ubC, isTop, isBot, M, R);

    float biasVal = 0.0f, valM = 0.0f, valE = 0.0f, valU = 0.0f, valV = 0.0f;
    EvalTileStats<W, (S_CST > 0U), (CfgT::R_LIM > 32U && W == 64U)>(
        ubB, ubBase[CfgT::OFF_X0], ubA, ubBase[CfgT::SlotY(6U)],
        ubBase[CfgT::OFF_CU], ubBase[CfgT::OFF_CV],
        ubAccR, ubAccRV, ubTmpVec, M, R, tdGm, bias, S, s, k,
        f0, stride, biasVal, valM, valE, valU, valV);
    JoinTileOut<S_CST, PRE_SYNC, OPT_R>(
        gmKeypoints, gmKeypointsU32, gmScores, ubOutKp, ubOutKpU32, ubOutScore, ubTmpVec,
        S, s, bk, v0, baseR, remR, valM, valE, valU, valV, f0, stride, biasVal);
}

template <typename Cfg>
__attribute__((noinline)) __aicore__ void RunStagePostFixed(
    GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores, GM_ADDR tiling,
    uint32_t s, uint32_t k, uint32_t bk) {
    const uint32_t isTop = (s == 0U) ? 1U : 0U;
    const uint32_t isBot = (s + 1U == Cfg::S) ? 1U : 0U;
    const __gm__ FusedDisconnectorKeypointSoftArgmaxTilingData *tdGm =
        reinterpret_cast<const __gm__ FusedDisconnectorKeypointSoftArgmaxTilingData *>(tiling);

    constexpr bool DO_PRE = (Cfg::S >= 3U && Cfg::S <= 4U) || Cfg::OPT_P;
    constexpr bool DO_REV = (Cfg::S >= 3U && Cfg::S <= 4U);
    constexpr bool OMIT_PRE = Cfg::OPT_P;
    if constexpr (Cfg::OPT_T0) {
        const uint32_t R = isTop ? Cfg::R_A : (isBot ? Cfg::R_C : Cfg::R_B);
        const uint32_t M = R * Cfg::W;
        const uint32_t v0 = isTop ? 0U : (Cfg::R_A + (s - 1U) * Cfg::R_B);
        RunStagePostBody<Cfg, Cfg::W, Cfg::S, DO_PRE, DO_REV, OMIT_PRE>(
            bias, keypoints, scores, 0.0f, 0.0f, tdGm, s, k, bk, Cfg::S, R, v0, isTop, isBot, M, Cfg::R_A, 0U);
    } else if constexpr (Cfg::R_REM == 0U) {
        constexpr uint32_t R = Cfg::R_BASE;
        constexpr uint32_t M = R * Cfg::W;
        const uint32_t v0 = s * R;
        RunStagePostBody<Cfg, Cfg::W, Cfg::S, DO_PRE, DO_REV, OMIT_PRE>(
            bias, keypoints, scores, 0.0f, 0.0f, tdGm, s, k, bk, Cfg::S, R, v0, isTop, isBot, M, Cfg::R_BASE, 0U);
    } else {
        const uint32_t R = Cfg::R_BASE + ((s < Cfg::R_REM) ? 1U : 0U);
        const uint32_t v0 = s * Cfg::R_BASE + ((s < Cfg::R_REM) ? s : Cfg::R_REM);
        const uint32_t M = R * Cfg::W;
        RunStagePostBody<Cfg, Cfg::W, Cfg::S, DO_PRE, DO_REV, OMIT_PRE>(
            bias, keypoints, scores, 0.0f, 0.0f, tdGm, s, k, bk, Cfg::S, R, v0, isTop, isBot, M, Cfg::R_BASE, Cfg::R_REM);
    }
}

template <typename CfgT, uint32_t W>
__attribute__((noinline)) __aicore__ void RunStagePostVar(
    GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores, float f0, float stride,
    uint32_t s, uint32_t k, uint32_t bk, uint32_t S, uint32_t R, uint32_t v0,
    uint32_t isTop, uint32_t isBot, uint32_t M, uint32_t baseR, uint32_t remR) {
    RunStagePostBody<CfgT, W, 0U>(
        bias, keypoints, scores, f0, stride, nullptr,
        s, k, bk, S, R, v0, isTop, isBot, M, baseR, remR);
}

}
