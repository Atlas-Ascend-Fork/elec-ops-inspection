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

#include "kpt_p7.h"
#include "kpt_p6.h"

namespace elec::impl {

template <typename Cfg, bool DO_INIT, uint64_t TAG_VAL>
KPT_INL void ExecModeSplit(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores, GM_ADDR tiling,
    uint32_t blockIdx, uint32_t b, uint32_t k, uint32_t s, uint32_t bk, uint32_t isTop, uint32_t isBot) {
    constexpr uint32_t SYNC_T0 = Cfg::OPT_P ? 20U : 0U;
    constexpr uint32_t SYNC_T1 = Cfg::OPT_P ? 21U : 0U;
    bool isQuick = false;
    if (isTop != 0U) {
        constexpr uint32_t R0 = Cfg::R_A;
        constexpr uint32_t M0 = R0 * Cfg::W;
        constexpr uint32_t LOAD0 = (R0 + 1U) * Cfg::W;
        isQuick = RunStagePipe<Cfg, Cfg::W, Cfg::HW, true, 1, 0, R0, LOAD0, SYNC_T0>(
            feat, weight, bias, DO_INIT, b, k, 0U, 1U, 0U, 0U, LOAD0, 0U, 0U, M0, R0);
    } else if constexpr (Cfg::S == 2U) {
        constexpr uint32_t R1 = Cfg::R_C;
        constexpr uint32_t M1 = R1 * Cfg::W;
        constexpr uint32_t LOAD1 = (R1 + 1U) * Cfg::W;
        isQuick = RunStagePipe<Cfg, Cfg::W, Cfg::HW, true, 0, 1, R1, LOAD1, SYNC_T1>(
            feat, weight, bias, DO_INIT, b, k, Cfg::R_A, 0U, 1U, Cfg::R_A - 1U, LOAD1, 0U, 0U, M1, R1);
    } else if constexpr (!Cfg::OPT_T3) {
        constexpr uint32_t R1 = Cfg::R_B;
        constexpr uint32_t M1 = R1 * Cfg::W;
        const uint32_t v0 = Cfg::R_A + (s - 1U) * R1;
        const uint32_t loadFloats = (R1 + 2U - isBot) * Cfg::W;
        isQuick = RunStagePipe<Cfg, Cfg::W, Cfg::HW, true, 0, -1, R1, 0U, 0U>(
            feat, weight, bias, DO_INIT, b, k, v0, 0U, isBot, v0 - 1U, loadFloats, 0U, 0U, M1, R1);
    } else {
        const uint32_t R = isBot ? Cfg::R_C : Cfg::R_B;
        const uint32_t M = R * Cfg::W;
        const uint32_t v0 = Cfg::R_A + (s - 1U) * Cfg::R_B;
        const uint32_t loadFloats = (R + 2U - isBot) * Cfg::W;
        isQuick = RunStagePipe<Cfg, Cfg::W, Cfg::HW, true, 0, -1, 0U, 0U, 0U>(
            feat, weight, bias, DO_INIT, b, k, v0, 0U, isBot, v0 - 1U, loadFloats, 0U, 0U, M, R);
    }
    if (__builtin_expect(isQuick, 0)) {
        if (blockIdx == 0U) {
            EmitConstBlk0(keypoints, scores, tiling, TAG_VAL);
        }
        return;
    }
    RunStagePostFixed<Cfg>(bias, keypoints, scores, tiling, s, k, bk);
}

template <typename Cfg, bool DO_INIT, uint64_t TAG_VAL>
KPT_INL void ExecModeEven(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores, GM_ADDR tiling,
    uint32_t blockIdx, uint32_t b, uint32_t k, uint32_t s, uint32_t bk, uint32_t isTop, uint32_t isBot) {
    constexpr uint32_t SYNC_T0 = Cfg::OPT_P ? 20U : 0U;
    constexpr uint32_t SYNC_T1 = Cfg::OPT_P ? 21U : 0U;
    const uint32_t R = (Cfg::R_REM == 0U) ? Cfg::R_BASE : (Cfg::R_BASE + ((s < Cfg::R_REM) ? 1U : 0U));
    const uint32_t v0 = (Cfg::R_REM == 0U)
        ? (s * Cfg::R_BASE)
        : (s * Cfg::R_BASE + ((s < Cfg::R_REM) ? s : Cfg::R_REM));
    const uint32_t rStart = v0 - (1U - isTop);
    const uint32_t loadFloats = (Cfg::S == 2U && Cfg::R_REM == 0U)
        ? ((Cfg::R_BASE + 1U) * Cfg::W)
        : ((R + 2U - isTop - isBot) * Cfg::W);
    const uint32_t M = R * Cfg::W;

    bool isQuick = false;
    if constexpr (Cfg::S == 2U && Cfg::R_REM == 0U) {
        constexpr uint32_t LD_2 = (Cfg::R_BASE + 1U) * Cfg::W;
        constexpr uint32_t M_2 = Cfg::R_BASE * Cfg::W;
        if (isTop != 0U) {
            isQuick = RunStagePipe<Cfg, Cfg::W, Cfg::HW, true, 1, 0, Cfg::R_BASE, LD_2, SYNC_T0>(
                feat, weight, bias, DO_INIT, b, k, 0U, 1U, 0U, 0U, loadFloats, 0U, 0U, M_2, Cfg::R_BASE);
        } else {
            isQuick = RunStagePipe<Cfg, Cfg::W, Cfg::HW, true, 0, 1, Cfg::R_BASE, LD_2, SYNC_T1>(
                feat, weight, bias, DO_INIT, b, k, Cfg::R_BASE, 0U, 1U, Cfg::R_BASE - 1U, loadFloats, 0U, 0U, M_2, Cfg::R_BASE);
        }
    } else {
        constexpr uint32_t R0 = Cfg::R_LIM;
        constexpr uint32_t LOAD0 = (Cfg::S == 1U) ? (R0 * Cfg::W) : ((R0 + 1U) * Cfg::W);
        if (isTop != 0U) {
            isQuick = RunStagePipe<Cfg, Cfg::W, Cfg::HW, true, 1, (Cfg::S == 1U) ? 1 : 0, R0, LOAD0, 0U>(
                feat, weight, bias, DO_INIT, b, k, 0U, 1U, 0U, 0U, loadFloats, 0U, 0U, M, R);
        } else {
            isQuick = RunStagePipe<Cfg, Cfg::W, Cfg::HW, true, 0, -1, 0U, 0U, 0U>(
                feat, weight, bias, DO_INIT, b, k, v0, 0U, isBot, rStart, loadFloats, 0U, 0U, M, R);
        }
    }
    if (__builtin_expect(isQuick, 0)) {
        if (blockIdx == 0U) {
            EmitConstBlk0(keypoints, scores, tiling, TAG_VAL);
        }
        return;
    }
    RunStagePostFixed<Cfg>(bias, keypoints, scores, tiling, s, k, bk);
}

template <uint64_t MODE>
KPT_INL void EntryModeFixed(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores, GM_ADDR tiling) {
    if ASCEND_IS_AIC {
        return;
    }
    using Cfg = ModeSpec<MODE>;
    constexpr uint64_t TAG_VAL = EncodeTag(Cfg::SlotY(6U), Cfg::B, Cfg::H, Cfg::W);
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    if (blockIdx >= Cfg::N_CORES) {
        return;
    }
    constexpr bool EARLY_CHK = (MODE == 2ULL) || (MODE == 9ULL);
    if constexpr (EARLY_CHK) {
        if (__builtin_expect(
                EvalQuickPath(
                    weight, bias, keypoints, scores, tiling,
                    blockIdx, Cfg::B, Cfg::H, Cfg::W, 0.0f, true, true),
                0)) {
            return;
        }
    }

    uint32_t b = 0U, k = 0U, s = 0U, bk = 0U;
    constexpr bool DO_REV = (Cfg::S >= 3U && Cfg::S <= 4U);
    MapIdxFixed<Cfg::B, Cfg::S, DO_REV, Cfg::OPT_M>(blockIdx, b, k, s, bk);

    if constexpr (Cfg::OPT_P && !EARLY_CHK) {
        AscendC::InitSocState();
    }
    constexpr bool DO_INIT = (!Cfg::OPT_P) && (!EARLY_CHK);
    const uint32_t isTop = (s == 0U) ? 1U : 0U;
    const uint32_t isBot = (s + 1U == Cfg::S) ? 1U : 0U;
    if constexpr (Cfg::OPT_T0) {
        ExecModeSplit<Cfg, DO_INIT, TAG_VAL>(
            feat, weight, bias, keypoints, scores, tiling, blockIdx, b, k, s, bk, isTop, isBot);
    } else {
        ExecModeEven<Cfg, DO_INIT, TAG_VAL>(
            feat, weight, bias, keypoints, scores, tiling, blockIdx, b, k, s, bk, isTop, isBot);
    }
}

template <typename TierT, uint32_t H, uint32_t W, uint32_t HW>
KPT_INL void ExecTierVar(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores, GM_ADDR tiling,
    float f0, float stride, uint32_t blockIdx, uint32_t b, uint32_t k, uint32_t s, uint32_t bk,
    uint32_t S, uint32_t R, uint32_t v0, uint32_t isTop, uint32_t isBot, uint32_t rStart,
    uint32_t loadFloats, uint32_t vecFloats, uint8_t reps, uint32_t M, uint32_t baseR, uint32_t remR) {
    if (__builtin_expect(
            RunStagePipe<TierT, W, HW, false>(
                feat, weight, bias, false, b, k, v0, isTop, isBot, rStart, loadFloats, vecFloats, reps, M, R),
            0)) {
        if (blockIdx == 0U) {
            EmitConstBlk0(keypoints, scores, tiling, EncodeTag(0U, 0U, H, W));
        }
        return;
    }
    RunStagePostVar<TierT, W>(
        bias, keypoints, scores, f0, stride, s, k, bk, S, R, v0, isTop, isBot, M, baseR, remR);
}

template <uint32_t GRID_ID>
KPT_INL void EntryModeVar(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores, GM_ADDR tiling) {
    if ASCEND_IS_AIC {
        return;
    }
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(FusedDisconnectorKeypointSoftArgmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedDisconnectorKeypointSoftArgmaxTilingData, td, tiling);

    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t S = td.n1;
    if (blockIdx >= td.n0 || S == 0U) {
        return;
    }

    constexpr uint32_t H = GridTrait<GRID_ID>::H;
    constexpr uint32_t W = GridTrait<GRID_ID>::W;
    constexpr uint32_t HW = GridTrait<GRID_ID>::HW;

    uint32_t b = 0U, k = 0U, s = 0U, bk = 0U;
    MapIdxVar(blockIdx, td.batch, S, b, k, s, bk);

    const uint32_t baseR = H / S;
    const uint32_t remR = H % S;
    const uint32_t R = baseR + ((s < remR) ? 1U : 0U);
    const uint32_t v0 = s * baseR + ((s < remR) ? s : remR);
    const uint32_t isTop = (s == 0U) ? 1U : 0U;
    const uint32_t isBot = (s + 1U == S) ? 1U : 0U;
    const uint32_t rStart = v0 - (1U - isTop);
    const uint32_t loadFloats = (R + 2U - isTop - isBot) * W;
    const uint32_t vecFloats = (loadFloats + 63U) & ~63U;
    const uint8_t reps = static_cast<uint8_t>(vecFloats >> 6U);
    const uint32_t M = R * W;

    if constexpr (GRID_ID == 0U) {
        if (vecFloats <= 1664U && (M + 8U) <= 1664U) {
            ExecTierVar<TierSpec<0U>, H, W, HW>(
                feat, weight, bias, keypoints, scores, tiling, td.f0, td.stride,
                blockIdx, b, k, s, bk, S, R, v0, isTop, isBot, rStart, loadFloats, vecFloats, reps, M, baseR, remR);
            return;
        }
    } else if constexpr (GRID_ID == 1U) {
        ExecTierVar<TierSpec<1U>, H, W, HW>(
            feat, weight, bias, keypoints, scores, tiling, td.f0, td.stride,
            blockIdx, b, k, s, bk, S, R, v0, isTop, isBot, rStart, loadFloats, vecFloats, reps, M, baseR, remR);
        return;
    }
    ExecTierVar<TierSpec<2U>, H, W, HW>(
        feat, weight, bias, keypoints, scores, tiling, td.f0, td.stride,
        blockIdx, b, k, s, bk, S, R, v0, isTop, isBot, rStart, loadFloats, vecFloats, reps, M, baseR, remR);
}

}
