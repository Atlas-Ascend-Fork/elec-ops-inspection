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

#include "kpt_p0.h"

namespace elec::impl {

template <uint32_t GRID_ID>
struct GridTrait {
    static constexpr bool FIXED_F = false;
    static constexpr uint32_t H = 72U;
    static constexpr uint32_t W = 96U;
    static constexpr uint32_t HW = 6912U;
};

template <>
struct GridTrait<0U> {
    static constexpr bool FIXED_F = true;
    static constexpr uint32_t H = 48U;
    static constexpr uint32_t W = 64U;
    static constexpr uint32_t HW = 3072U;
};

template <>
struct GridTrait<1U> {
    static constexpr bool FIXED_F = true;
    static constexpr uint32_t H = 64U;
    static constexpr uint32_t W = 64U;
    static constexpr uint32_t HW = 4096U;
};

template <>
struct GridTrait<2U> {
    static constexpr bool FIXED_F = true;
    static constexpr uint32_t H = 72U;
    static constexpr uint32_t W = 96U;
    static constexpr uint32_t HW = 6912U;
};

__aicore__ constexpr inline uint32_t CalcSpan(uint32_t rawStride, uint32_t baseSh, uint32_t s) {
    if (baseSh == 0U && s == 3U) {
        return 13U * 128U;
    }
    if (baseSh == 2U && s <= 2U) {
        return (((rawStride / 128U) % 3U) == 0U) ? (rawStride + 128U) : rawStride;
    }
    uint32_t blk = rawStride / 128U;
    while ((blk % 2U) == 0U || (blk % 3U) == 0U) {
        ++blk;
    }
    return blk * 128U;
}

template <
    uint64_t M_ID,
    uint32_t B_VAL,
    uint32_t S_VAL,
    uint32_t GRID_TAG,
    uint32_t R0_CFG,
    uint32_t R1_CFG,
    bool MAP_CFG>
struct ModeSpecBase {
    static constexpr uint32_t B = B_VAL;
    static constexpr uint32_t S = S_VAL;
    static constexpr bool OPT_P = (S == 2U);
    static constexpr uint32_t H = GridTrait<GRID_TAG>::H;
    static constexpr uint32_t W = GridTrait<GRID_TAG>::W;
    static constexpr uint32_t HW = GridTrait<GRID_TAG>::HW;
    static constexpr uint32_t N_CORES = B * DIM_C1 * S;
    static constexpr bool OPT_T0 = (R0_CFG > 0U);
    static constexpr bool OPT_T3 = OPT_T0 && (R1_CFG > 0U) && (S >= 3U);
    static constexpr bool OPT_M = MAP_CFG;
    static constexpr uint32_t R_BASE = H / S;
    static constexpr uint32_t R_REM = H % S;
    static constexpr uint32_t R_A = OPT_T0 ? R0_CFG : (R_BASE + ((R_REM > 0U) ? 1U : 0U));
    static constexpr uint32_t R_B =
        OPT_T3 ? R1_CFG : (OPT_T0 ? ((H - R0_CFG) / (S - 1U)) : R_BASE);
    static constexpr uint32_t R_C =
        OPT_T3 ? (H - R0_CFG - (S - 2U) * R1_CFG) : R_B;
    static constexpr uint32_t R_SUB = (R_B > R_C) ? R_B : R_C;
    static constexpr uint32_t R_LIM = (R_A > R_SUB) ? R_A : R_SUB;
    static constexpr uint32_t M_LIM = R_LIM * W;
    static constexpr uint32_t LD_R0 = (S == 1U) ? H : (R_A + 1U);
    static constexpr uint32_t LD_R1 =
        (S <= 2U) ? (R_C + 1U)
                  : ((OPT_T0 ? R_B : ((R_REM >= 2U) ? R_LIM : R_BASE)) + 2U);
    static constexpr uint32_t LD_R2 = (S == 1U) ? H : (R_C + 1U);
    static constexpr uint32_t LD_SUB =
        (LD_R1 > LD_R2) ? LD_R1 : LD_R2;
    static constexpr uint32_t LD_LIM =
        (LD_R0 > LD_SUB) ? LD_R0 : LD_SUB;
    static constexpr uint32_t VEC_LIM = ((LD_LIM * W) + 63U) & ~63U;
    static constexpr uint32_t SPAN_MIN =
        (VEC_LIM > (M_LIM + 8U)) ? VEC_LIM : (M_LIM + 8U);
    static constexpr uint32_t SPAN_RAW = (SPAN_MIN + 127U) & ~127U;
    static constexpr bool OPT_A = (SPAN_RAW <= 2688U);
    static constexpr bool OPT_B = OPT_A;
    static constexpr bool OPT_C = (!OPT_A) && (GRID_TAG == 2U && S == 2U && !OPT_T0);
    static constexpr bool OPT_D = (!OPT_A) && (M_ID == 11ULL) && !OPT_C;
    static constexpr bool OPT_E =
        OPT_C || OPT_D ||
        ((!OPT_A) && ((GRID_TAG == 1U && S == 2U) || (GRID_TAG == 2U && S == 4U)));
    static constexpr uint32_t SPAN_Y =
        OPT_A
            ? (((M_ID >= 9ULL) && (((SPAN_RAW / 128U) % 3U) == 0U) && (SPAN_RAW + 128U <= 2560U))
                   ? (SPAN_RAW + 128U)
                   : SPAN_RAW)
            : (OPT_C
                   ? SPAN_RAW
                   : (OPT_D
                          ? ((((SPAN_RAW / 128U) % 3U) == 0U) ? (SPAN_RAW + 128U) : SPAN_RAW)
                          : CalcSpan(SPAN_RAW, GRID_TAG, S)));
    static constexpr uint32_t OFF_Y0 =
        OPT_B ? (15872U - 4U * SPAN_Y) : (OPT_C ? 512U : (OPT_E ? 384U : 768U));
    static constexpr uint32_t OFF_CV = OPT_C ? 384U : (OPT_E ? 48960U : 448U);
    static constexpr uint32_t OFF_CU = OPT_E ? 49024U : 512U;
    static constexpr uint32_t LIM_UB = OPT_E ? 48960U : 48640U;
    static constexpr uint32_t SPAN_F = SPAN_RAW;
    static constexpr uint32_t OFF_X0 =
        OPT_A ? (32768U + 64U) : (OFF_Y0 + 9U * SPAN_Y + 64U);
    static constexpr uint32_t OFF_X1 = OFF_X0 + 2U * SPAN_F;
    static constexpr uint32_t OFF_X2 = OFF_X1 + 2U * SPAN_F;
    static constexpr uint32_t SPAN_IR = (M_LIM + 127U) & ~127U;
    static constexpr uint32_t SPAN_IX = (SPAN_IR < 1024U) ? 1024U : SPAN_IR;
    static constexpr uint32_t BASE_I3 = (OFF_X2 + 2U * SPAN_F + 127U) & ~127U;
    static constexpr bool OPT_F =
        OPT_A ? ((OFF_X2 + 2U * SPAN_F) <= 49152U)
                    : (BASE_I3 + 2U * SPAN_IX <= LIM_UB);
    static constexpr uint32_t BASE_IL =
        OPT_F ? BASE_I3 : ((OFF_X1 + 2U * SPAN_F + 127U) & ~127U);
    static constexpr bool OPT_G =
        OPT_A ? true : (BASE_IL + 2U * SPAN_IX <= LIM_UB);
    static constexpr uint32_t OFF_IL =
        OPT_B ? 768U : (OPT_A ? (768U + 5U * SPAN_Y) : BASE_IL);
    static constexpr uint32_t OFF_IR =
        OPT_B ? (OFF_Y0 + 9U * SPAN_Y)
                    : (OPT_A ? (16384U + 4U * SPAN_Y) : (BASE_IL + SPAN_IX));
    static constexpr uint32_t OFF_SD = (OFF_X1 + 2U * SPAN_F + 127U) & ~127U;
    static constexpr uint32_t LEN_SD = OPT_C ? 576U : 384U;
    static constexpr bool OPT_H =
        OPT_A ? false : ((GRID_TAG == 1U) || (GRID_TAG == 2U && S == 4U));
    static constexpr bool OPT_J = (OPT_A || OPT_C || OPT_D);
    static constexpr bool OPT_K = (M_ID == 0ULL) || (M_ID == 8ULL) || (M_ID == 11ULL);
    static constexpr uint32_t GRP_W = (M_ID == 0ULL) ? 2U : 1U;
    static constexpr uint32_t GRP_D = 4U;

    __aicore__ static constexpr inline uint32_t SlotY(uint32_t idx) {
        if (OPT_B) {
            return OFF_Y0 + idx * SPAN_Y;
        }
        if (OPT_A) {
            return (idx < 5U) ? (768U + idx * SPAN_Y) : (16384U + (idx - 5U) * SPAN_Y);
        }
        return OFF_Y0 + idx * SPAN_Y;
    }
};

template <uint64_t MODE>
struct ModeSpec;

template <> struct ModeSpec<0ULL> : ModeSpecBase<0ULL, 1U, 3U, 0U, 20U, 15U, false> {};
template <> struct ModeSpec<1ULL> : ModeSpecBase<1ULL, 1U, 3U, 1U, 26U, 20U, false> {};
template <> struct ModeSpec<2ULL> : ModeSpecBase<2ULL, 1U, 4U, 2U, 24U, 0U, false> {};
template <> struct ModeSpec<3ULL> : ModeSpecBase<3ULL, 2U, 3U, 0U, 20U, 16U, true> {};
template <> struct ModeSpec<4ULL> : ModeSpecBase<4ULL, 2U, 3U, 1U, 27U, 21U, true> {};
template <> struct ModeSpec<5ULL> : ModeSpecBase<5ULL, 2U, 3U, 2U, 26U, 24U, true> {};
template <> struct ModeSpec<6ULL> : ModeSpecBase<6ULL, 3U, 2U, 0U, 25U, 0U, true> {};
template <> struct ModeSpec<7ULL> : ModeSpecBase<7ULL, 3U, 2U, 1U, 35U, 0U, true> {};
template <> struct ModeSpec<8ULL> : ModeSpecBase<8ULL, 3U, 2U, 2U, 0U, 0U, true> {};
template <> struct ModeSpec<9ULL> : ModeSpecBase<9ULL, 4U, 2U, 0U, 27U, 0U, true> {};
template <> struct ModeSpec<10ULL> : ModeSpecBase<10ULL, 4U, 2U, 1U, 35U, 0U, true> {};
template <> struct ModeSpec<11ULL> : ModeSpecBase<11ULL, 4U, 2U, 2U, 0U, 0U, true> {};

template <uint32_t TIER>
struct TierSpec {
    static constexpr uint32_t SPAN_RAW = (TIER == 0U) ? 1664U : ((TIER == 1U) ? 2176U : 3584U);
    static constexpr uint32_t R_LIM = 32U;
    static constexpr uint32_t M_LIM = 2560U;
    static constexpr bool OPT_A = (TIER < 2U);
    static constexpr bool OPT_B = OPT_A;
    static constexpr bool OPT_C = false;
    static constexpr bool OPT_E = false;
    static constexpr uint32_t SPAN_Y = SPAN_RAW;
    static constexpr uint32_t SPAN_F = SPAN_RAW;
    static constexpr uint32_t OFF_Y0 = OPT_B ? (15872U - 4U * SPAN_Y) : 768U;
    static constexpr uint32_t OFF_CV = 448U;
    static constexpr uint32_t OFF_CU = 512U;
    static constexpr uint32_t LIM_UB = 48640U;
    static constexpr uint32_t OFF_X0 = OPT_A ? (32768U + 64U) : (OFF_Y0 + 9U * SPAN_Y + 64U);
    static constexpr uint32_t OFF_X1 = OFF_X0 + 2U * SPAN_F;
    static constexpr uint32_t OFF_X2 = OFF_X1 + 2U * SPAN_F;
    static constexpr uint32_t SPAN_IX = SPAN_RAW;
    static constexpr uint32_t BASE_I3 = (OFF_X2 + 2U * SPAN_F + 127U) & ~127U;
    static constexpr bool OPT_F =
        OPT_A ? ((OFF_X2 + 2U * SPAN_F) <= 49152U)
                    : (BASE_I3 + 2U * SPAN_IX <= LIM_UB);
    static constexpr uint32_t BASE_IL =
        OPT_F ? BASE_I3 : ((OFF_X1 + 2U * SPAN_F + 127U) & ~127U);
    static constexpr bool OPT_G =
        OPT_A ? true : (BASE_IL + 2U * SPAN_IX <= LIM_UB);
    static constexpr uint32_t OFF_IL = OPT_B ? 768U : BASE_IL;
    static constexpr uint32_t OFF_IR = OPT_B ? (OFF_Y0 + 9U * SPAN_Y) : (BASE_IL + SPAN_IX);
    static constexpr uint32_t OFF_SD = (OFF_X1 + 2U * SPAN_F + 127U) & ~127U;
    static constexpr uint32_t LEN_SD = 384U;
    static constexpr bool OPT_H = false;
    static constexpr bool OPT_J = true;
    static constexpr bool OPT_K = false;
    static constexpr uint32_t GRP_W = 2U;
    static constexpr uint32_t GRP_D = 2U;

    __aicore__ static constexpr inline uint32_t SlotY(uint32_t idx) {
        return OFF_Y0 + idx * SPAN_Y;
    }
};

template <uint32_t B_FIX, uint32_t S_FIX>
__aicore__ constexpr inline void MapIdxSub(uint32_t blockIdx, uint32_t &s, uint32_t &bk) {
    constexpr uint32_t N_TOT = B_FIX * DIM_C1 * S_FIX;
    constexpr uint32_t N_OVF = N_TOT - 20U;
    constexpr uint32_t N_GRP = B_FIX * 2U;
    uint32_t grpIdx = 0U;
    uint32_t sub = 0U;
    if (blockIdx < N_OVF) {
        grpIdx = blockIdx;
    } else if (blockIdx >= 20U) {
        grpIdx = blockIdx - 20U;
        sub = 1U;
    } else {
        const uint32_t t = blockIdx - N_OVF;
        grpIdx = N_OVF + (t >> 1U);
        sub = t & 1U;
    }
    uint32_t stepIdx = 0U;
    uint32_t grpSub = 0U;
    if constexpr (N_GRP == 2U) {
        stepIdx = grpIdx >> 1U;
        grpSub = grpIdx & 1U;
    } else if constexpr (N_GRP == 4U) {
        stepIdx = grpIdx >> 2U;
        grpSub = grpIdx & 3U;
    } else if constexpr (N_GRP == 8U) {
        stepIdx = grpIdx >> 3U;
        grpSub = grpIdx & 7U;
    } else if constexpr (N_GRP == 6U) {
        stepIdx = (grpIdx * 43U) >> 8U;
        grpSub = grpIdx - stepIdx * 6U;
    } else if constexpr (N_GRP > 0U) {
        stepIdx = grpIdx / N_GRP;
        grpSub = grpIdx % N_GRP;
    }
    s = (S_FIX - 1U) - stepIdx;
    bk = (grpSub << 1U) | sub;
}

template <uint32_t B_FIX, uint32_t S_FIX, bool OPT_R = false, bool OPT_M = false>
__aicore__ constexpr inline void MapIdxFixed(
    uint32_t blockIdx, uint32_t &b, uint32_t &k, uint32_t &s, uint32_t &bk) {
    constexpr uint32_t N_TOT = B_FIX * DIM_C1 * S_FIX;
    if constexpr (OPT_M && (N_TOT > 20U)) {
        MapIdxSub<B_FIX, S_FIX>(blockIdx, s, bk);
    } else if constexpr (S_FIX == 2U) {
        s = OPT_R ? ((blockIdx & 1U) ^ 1U) : (blockIdx & 1U);
        bk = blockIdx >> 1U;
    } else if constexpr (S_FIX == 4U) {
        s = OPT_R ? ((blockIdx & 3U) ^ 3U) : (blockIdx & 3U);
        bk = blockIdx >> 2U;
    } else if constexpr (S_FIX == 3U) {
        constexpr uint32_t N_DBL = 2U * B_FIX * DIM_C1;
        if (blockIdx < N_DBL) {
            bk = blockIdx >> 1U;
            const uint32_t sub = blockIdx & 1U;
            s = OPT_R ? (2U - sub) : (1U + sub);
        } else {
            bk = blockIdx - N_DBL;
            s = 0U;
        }
    } else if constexpr (S_FIX > 0U) {
        bk = blockIdx / S_FIX;
        const uint32_t rem = blockIdx % S_FIX;
        s = OPT_R ? ((S_FIX - 1U) - rem) : rem;
    }
    k = bk & 3U;
    b = (B_FIX == 1U) ? 0U : (bk >> 2U);
}

KPT_INL void MapIdxVar(
    uint32_t blockIdx, uint32_t batch, uint32_t S, uint32_t &b, uint32_t &k, uint32_t &s, uint32_t &bk) {
    if (S == 2U) {
        s = blockIdx & 1U;
        bk = blockIdx >> 1U;
    } else if (S == 4U) {
        s = blockIdx & 3U;
        bk = blockIdx >> 2U;
    } else if (S == 3U) {
        const uint32_t twoBk = (batch * DIM_C1) << 1U;
        if (blockIdx < twoBk) {
            bk = blockIdx >> 1U;
            s = 1U + (blockIdx & 1U);
        } else {
            bk = blockIdx - twoBk;
            s = 0U;
        }
    } else if (S == 1U) {
        s = 0U;
        bk = blockIdx;
    } else if (S > 0U) {
        bk = blockIdx / S;
        s = blockIdx % S;
    }
    k = bk & 3U;
    b = bk >> 2U;
}

}
