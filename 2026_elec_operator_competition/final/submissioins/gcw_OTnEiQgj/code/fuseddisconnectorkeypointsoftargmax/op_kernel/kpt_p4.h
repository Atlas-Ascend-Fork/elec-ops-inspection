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

struct StatPtrs {
    __ubuf__ float *pWork0, *pWork1, *pWork2, *pAxisU, *pAxisV, *pAccR, *pAccRV, *pTmpVec;
};

KPT_INL void ScanPeakVar(
    __ubuf__ float *pB, __ubuf__ float *pWork0, __ubuf__ float *pWork1, __ubuf__ float *pTmpVec, uint32_t M) {
    const uint32_t full64 = M >> 6U;
    const uint32_t tail64 = M & 63U;
    set_vector_mask(~0ULL, ~0ULL);
    OpVcgmax(pWork0, pB, static_cast<int32_t>(full64), 1, 1, 8);
    uint32_t stage1Len = full64 << 3U;
    if (tail64 > 0U) {
        ApplyVecMask(tail64);
        OpVcgmax(pWork0 + stage1Len, pB + (full64 << 6U), 1, 1, 1, 8);
        set_vector_mask(~0ULL, ~0ULL);
        stage1Len += ((tail64 + 7U) >> 3U);
    }
    AscendC::PipeBarrier<PIPE_V>();
    if (stage1Len <= 64U) {
        ApplyVecMask(stage1Len);
        OpVcmax(pTmpVec, pWork0, 1, 1, 1, 8);
        set_vector_mask(~0ULL, ~0ULL);
        return;
    }
    const uint32_t full2 = stage1Len >> 6U;
    const uint32_t tail2 = stage1Len & 63U;
    OpVcgmax(pWork1, pWork0, static_cast<int32_t>(full2), 1, 1, 8);
    uint32_t stage2Len = full2 << 3U;
    if (tail2 > 0U) {
        ApplyVecMask(tail2);
        OpVcgmax(pWork1 + stage2Len, pWork0 + (full2 << 6U), 1, 1, 1, 8);
        set_vector_mask(~0ULL, ~0ULL);
        stage2Len += ((tail2 + 7U) >> 3U);
    }
    AscendC::PipeBarrier<PIPE_V>();
    ApplyVecMask(stage2Len);
    OpVcmax(pTmpVec, pWork1, 1, 1, 1, 8);
    set_vector_mask(~0ULL, ~0ULL);
}

template <uint32_t W, bool FAST_M>
KPT_INL void ScanPeak(
    __ubuf__ float *pB, __ubuf__ float *pWork0, __ubuf__ float *pWork1, __ubuf__ float *pTmpVec, uint32_t M) {
    if ((FAST_M && (W == 64U)) || (FAST_M && ((M & 63U) == 0U))) {
        const uint32_t mRepsU = M >> 6U;
        set_vector_mask(~0ULL, ~0ULL);
        OpVcmax(pWork0, pB, static_cast<int32_t>(mRepsU), 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        ApplyVecMask(mRepsU);
        OpVcmax(pTmpVec, pWork0, 1, 1, 1, 8);
        set_vector_mask(~0ULL, ~0ULL);
    } else if (M <= 64U) {
        ApplyVecMask(M);
        OpVcmax(pTmpVec, pB, 1, 1, 1, 8);
        set_vector_mask(~0ULL, ~0ULL);
    } else if ((M & 63U) == 0U && M <= 4096U) {
        const int32_t mReps = static_cast<int32_t>(M >> 6U);
        set_vector_mask(~0ULL, ~0ULL);
        OpVcmax(pWork0, pB, mReps, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        ApplyVecMask(static_cast<uint32_t>(mReps));
        OpVcmax(pTmpVec, pWork0, 1, 1, 1, 8);
        set_vector_mask(~0ULL, ~0ULL);
    } else {
        ScanPeakVar(pB, pWork0, pWork1, pTmpVec, M);
    }
}

template <uint32_t W, bool FAST_M>
KPT_INL void EvalScaledExp(
    __ubuf__ float *pWork0, __ubuf__ float *pB, __ubuf__ float *pWork1, uint32_t M, float valM, float f0) {
    if ((FAST_M && (W == 64U)) || ((M & 63U) == 0U)) {
        const uint8_t mReps = static_cast<uint8_t>(M >> 6U);
        set_vector_mask(~0ULL, ~0ULL);
        OpVadds(pWork0, pB, -valM, mReps);
        if (mReps < 18U) { AscendC::PipeBarrier<PIPE_V>(); }
        OpVmuls(pWork1, pWork0, f0, mReps);
        if (mReps < 18U) { AscendC::PipeBarrier<PIPE_V>(); }
        OpVexp(pWork0, pWork1, mReps);
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }
    const uint32_t full = M >> 6U;
    const uint32_t tail = M & 63U;
    set_vector_mask(~0ULL, ~0ULL);
    if (full > 0U) { OpVadds(pWork0, pB, -valM, static_cast<uint8_t>(full)); }
    ApplyVecMask(tail);
    OpVadds(pWork0 + (full << 6U), pB + (full << 6U), -valM, 1U);
    AscendC::PipeBarrier<PIPE_V>();
    set_vector_mask(~0ULL, ~0ULL);
    if (full > 0U) { OpVmuls(pWork0, pWork0, f0, static_cast<uint8_t>(full)); }
    ApplyVecMask(tail);
    OpVmuls(pWork0 + (full << 6U), pWork0 + (full << 6U), f0, 1U);
    AscendC::PipeBarrier<PIPE_V>();
    set_vector_mask(~0ULL, ~0ULL);
    if (full > 0U) { OpVexp(pWork0, pWork0, static_cast<uint8_t>(full)); }
    ApplyVecMask(tail);
    OpVexp(pWork0 + (full << 6U), pWork0 + (full << 6U), 1U);
    set_vector_mask(~0ULL, ~0ULL);
    AscendC::PipeBarrier<PIPE_V>();
}

template <bool EXT_36>
KPT_INL float AccCols64(const AscendC::LocalTensor<float> &ubTmpVec, uint32_t r2) {
    const float sPU0 = ubTmpVec.GetValue(8U) + ubTmpVec.GetValue(9U);
    float sPU1 = (r2 > 2U) ? ubTmpVec.GetValue(10U) : 0.0f;
    if (r2 > 3U) { sPU1 += ubTmpVec.GetValue(11U); }
    float sPU2 = (r2 > 4U) ? ubTmpVec.GetValue(12U) : 0.0f;
    if (r2 > 5U) { sPU2 += ubTmpVec.GetValue(13U); }
    float sPU3 = (r2 > 6U) ? ubTmpVec.GetValue(14U) : 0.0f;
    if (r2 > 7U) { sPU3 += ubTmpVec.GetValue(15U); }
    float sPU4 = (r2 > 8U) ? ubTmpVec.GetValue(16U) : 0.0f;
    if constexpr (EXT_36) {
        if (r2 > 9U) { sPU4 += ubTmpVec.GetValue(17U); }
        const float sPU5 = (r2 > 10U) ? ubTmpVec.GetValue(18U) : 0.0f;
        return ((sPU0 + sPU1) + (sPU2 + sPU3)) + (sPU4 + sPU5);
    }
    return ((sPU0 + sPU1) + (sPU2 + sPU3)) + sPU4;
}

template <bool EXT_36>
KPT_INL void FillAxis64(__ubuf__ float *pU, __ubuf__ float *pAxisU, uint32_t r2) {
    OpVadds(pU + 0U, pAxisU, 0.0f, 1U);
    OpVadds(pU + 64U, pAxisU, 0.0f, 1U);
    if (r2 > 2U) {
        OpVadds(pU + 128U, pAxisU, 0.0f, 1U);
        OpVadds(pU + 192U, pAxisU, 0.0f, 1U);
    }
    if (r2 > 8U) {
        OpVadds(pU + 512U, pAxisU, 0.0f, 1U);
        if constexpr (EXT_36) {
            if (r2 > 9U) { OpVadds(pU + 576U, pAxisU, 0.0f, 1U); }
            if (r2 > 10U) { OpVadds(pU + 640U, pAxisU, 0.0f, 1U); }
        }
    }
}

template <bool EXT_36>
KPT_INL void EvalStat64(
    const StatPtrs &sb, const AscendC::LocalTensor<float> &ubTmpVec, uint32_t R, float valM,
    float &outMax, float &resE, float &resU, float &resV) {
    __ubuf__ float *pU = sb.pWork2 + 64U;
    uint32_t r1 = R - (R >> 1U);
    if (R >= 6U && ((r1 & 1U) == 0U)) { ++r1; }
    const uint32_t h1 = R - r1;
    uint32_t r2 = (r1 <= 3U) ? r1 : (r1 - (r1 >> 1U));
    if (r1 >= 4U && ((r2 & 1U) == 0U)) { ++r2; }
    const uint32_t h2 = r1 - r2;

    set_vector_mask(~0ULL, ~0ULL);
    OpVcadd(sb.pAccR, sb.pWork0, static_cast<int32_t>(R), 1, 1, 8);
    FillAxis64<EXT_36>(pU, sb.pAxisU, r2);
    OpVadd(sb.pWork1, sb.pWork0, sb.pWork0 + (r1 << 6U), static_cast<uint8_t>(h1));
    if (r1 > h1) {
        OpVadds(sb.pWork1 + (h1 << 6U), sb.pWork0 + (h1 << 6U), 0.0f, static_cast<uint8_t>(r1 - h1));
    }
    AscendC::PipeBarrier<PIPE_V>();

    ApplyVecMask(R);
    OpVcadd(sb.pTmpVec + 0U, sb.pAccR, 1, 1, 1, 8);
    OpVmul(sb.pAccRV, sb.pAccR, sb.pAxisV, 1U);
    set_vector_mask(~0ULL, ~0ULL);
    if (r2 > 4U) {
        OpVadds(pU + 256U, pU, 0.0f, static_cast<uint8_t>((r2 > 8U) ? 4U : (r2 - 4U)));
    }
    if (h2 > 0U) {
        OpVadd(sb.pWork1, sb.pWork1, sb.pWork1 + (r2 << 6U), static_cast<uint8_t>(h2));
    }
    AscendC::PipeBarrier<PIPE_V>();

    ApplyVecMask(R);
    OpVcadd(sb.pTmpVec + 24U, sb.pAccRV, 1, 1, 1, 8);
    set_vector_mask(~0ULL, ~0ULL);
    OpVmul(sb.pWork0, sb.pWork1, pU, static_cast<uint8_t>(r2));
    AscendC::PipeBarrier<PIPE_V>();

    OpVcadd(sb.pTmpVec + 8U, sb.pWork0, static_cast<int32_t>(r2), 1, 1, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    outMax = valM;
    resE = ubTmpVec.GetValue(0U);
    resV = ubTmpVec.GetValue(24U);
    resU = AccCols64<EXT_36>(ubTmpVec, r2);
}

KPT_INL float AccCols96(const AscendC::LocalTensor<float> &ubTmpVec, uint32_t uReps) {
    const float g0 = (ubTmpVec.GetValue(8U) + ubTmpVec.GetValue(9U)) + ubTmpVec.GetValue(10U);
    const float g1 = (ubTmpVec.GetValue(11U) + ubTmpVec.GetValue(12U)) + ubTmpVec.GetValue(13U);
    const float g2 = (uReps >= 9U) ? ((ubTmpVec.GetValue(14U) + ubTmpVec.GetValue(15U)) + ubTmpVec.GetValue(16U)) : 0.0f;
    const float g3 = (uReps >= 12U) ? ((ubTmpVec.GetValue(17U) + ubTmpVec.GetValue(18U)) + ubTmpVec.GetValue(19U)) : 0.0f;
    const float g4 = (uReps >= 15U) ? ((ubTmpVec.GetValue(20U) + ubTmpVec.GetValue(21U)) + ubTmpVec.GetValue(22U)) : 0.0f;
    return ((g0 + g1) + (g2 + g3)) + g4;
}

KPT_INL void RunStage1W96(const StatPtrs &sb, __ubuf__ float *pU, uint32_t R, uint32_t r1, uint32_t h1) {
    const uint32_t h1Reps = (h1 * 3U) >> 1U;
    set_vector_mask(~0ULL, ~0ULL);
    OpVcadd(sb.pAccR, sb.pWork0, static_cast<int32_t>(R), 1, 1, 12);
    OpVadds(pU + 0U, sb.pAxisU + 0U, 0.0f, 1U);
    OpVadds(pU + 96U, sb.pAxisU + 0U, 0.0f, 1U);
    OpVadd(sb.pWork1, sb.pWork0, sb.pWork0 + r1 * 96U, static_cast<uint8_t>(h1Reps));
    if (r1 > h1) {
        const uint32_t c1Reps = ((r1 - h1) * 3U) >> 1U;
        OpVadds(sb.pWork1 + h1 * 96U, sb.pWork0 + h1 * 96U, 0.0f, static_cast<uint8_t>(c1Reps));
    }
    set_vector_mask(0ULL, 0xFFFFFFFFULL);
    OpVcadd(sb.pTmpVec + 24U, sb.pWork0 + 64U, static_cast<int32_t>(R), 1, 1, 12);
    OpVadds(pU + 64U, sb.pAxisU + 64U, 0.0f, 1U);
    OpVadds(pU + 160U, sb.pAxisU + 64U, 0.0f, 1U);
    set_vector_mask(~0ULL, ~0ULL);
    AscendC::PipeBarrier<PIPE_V>();
}

KPT_INL void EvalStat96(
    const StatPtrs &sb, const AscendC::LocalTensor<float> &ubTmpVec, uint32_t R, float valM,
    float &outMax, float &resE, float &resU, float &resV) {
    __ubuf__ float *pU = sb.pWork2 + 64U;
    uint32_t r1 = ((R >> 1U) + 1U) & ~1U;
    if (R >= 8U && R <= 36U && ((r1 & 3U) == 0U)) { r1 += 2U; }
    const uint32_t r2 = (r1 <= 10U) ? r1 : 10U;
    const uint32_t uReps = (r2 * 3U) >> 1U;
    RunStage1W96(sb, pU, R, r1, R - r1);

    ApplyVecMask(R);
    OpVadd(sb.pAccR, sb.pAccR, sb.pTmpVec + 24U, 1U);
    set_vector_mask(~0ULL, ~0ULL);
    OpVadds(pU + 192U, pU, 0.0f, 3U);
    if (r2 >= 6U) { OpVadds(pU + 384U, pU, 0.0f, 3U); }
    if (r2 >= 8U) { OpVadds(pU + 576U, pU, 0.0f, 3U); }
    if (r2 >= 10U) { OpVadds(pU + 768U, pU, 0.0f, 3U); }
    if (r1 > 10U) {
        const uint32_t h2Reps = ((r1 - r2) * 3U) >> 1U;
        OpVadd(sb.pWork1, sb.pWork1, sb.pWork1 + r2 * 96U, static_cast<uint8_t>(h2Reps));
    }
    AscendC::PipeBarrier<PIPE_V>();

    ApplyVecMask(R);
    OpVcadd(sb.pTmpVec + 0U, sb.pAccR, 1, 1, 1, 8);
    OpVmul(sb.pAccRV, sb.pAccR, sb.pAxisV, 1U);
    set_vector_mask(~0ULL, ~0ULL);
    OpVmul(sb.pWork0, sb.pWork1, pU, static_cast<uint8_t>(uReps));
    AscendC::PipeBarrier<PIPE_V>();

    OpVcadd(sb.pTmpVec + 8U, sb.pWork0, static_cast<int32_t>(uReps), 1, 1, 8);
    ApplyVecMask(R);
    OpVcadd(sb.pTmpVec + 24U, sb.pAccRV, 1, 1, 1, 8);
    set_vector_mask(~0ULL, ~0ULL);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    outMax = valM;
    resE = ubTmpVec.GetValue(0U);
    resV = ubTmpVec.GetValue(24U);
    resU = AccCols96(ubTmpVec, uReps);
}

template <uint32_t W>
KPT_INL uint32_t StepFoldRows(__ubuf__ float *pAccC, uint32_t curR) {
    uint32_t halfR = curR >> 1U;
    uint32_t remRows = curR - halfR;
    if constexpr (W == 64U) {
        if (curR >= 6U && (remRows % 6U) == 0U) {
            remRows += 1U;
            halfR = curR - remRows;
        }
        OpVadd(pAccC, pAccC, pAccC + remRows * W, static_cast<uint8_t>(halfR));
    } else {
        if (curR > 2U && (remRows & 1U) != 0U) {
            remRows += 1U;
            halfR = curR - remRows;
        }
        VecAddW96(pAccC, pAccC, pAccC + remRows * W, halfR);
    }
    return remRows;
}

KPT_INL void InitFold64(
    const StatPtrs &sb, uint32_t R, __ubuf__ float *&pAccC, uint32_t &curR, bool &hasRV) {
    OpVcadd(sb.pAccR, sb.pWork0, static_cast<int32_t>(R), 1, 1, 8);
    if (curR > 1U) {
        uint32_t halfR = curR >> 1U;
        uint32_t remRows = curR - halfR;
        if (curR >= 6U && (remRows % 6U) == 0U) {
            remRows += 1U;
            halfR = curR - remRows;
        }
        OpVadd(sb.pWork1, sb.pWork0, sb.pWork0 + remRows * 64U, static_cast<uint8_t>(halfR));
        if (remRows > halfR) {
            OpVadds(sb.pWork1 + halfR * 64U, sb.pWork0 + halfR * 64U, 0.0f, static_cast<uint8_t>(remRows - halfR));
        }
        pAccC = sb.pWork1;
        curR = remRows;
    }
    AscendC::PipeBarrier<PIPE_V>();
    if (curR > 1U) {
        ApplyVecMask(R);
        OpVmul(sb.pAccRV, sb.pAccR, sb.pAxisV, 1U);
        set_vector_mask(~0ULL, ~0ULL);
        hasRV = true;
        curR = StepFoldRows<64U>(pAccC, curR);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

KPT_INL void CopySpan96(__ubuf__ float *pDst, __ubuf__ float *pSrc, uint32_t extraRows) {
    const uint32_t copyElems = extraRows * 96U;
    const uint32_t copyFull = copyElems >> 6U;
    if (copyFull > 0U) {
        OpVadds(pDst, pSrc, 0.0f, static_cast<uint8_t>(copyFull));
    }
    if ((copyElems & 32U) != 0U) {
        set_vector_mask(0ULL, 0xFFFFFFFFULL);
        OpVadds(pDst + (copyFull << 6U), pSrc + (copyFull << 6U), 0.0f, 1U);
        set_vector_mask(~0ULL, ~0ULL);
    }
}

KPT_INL void InitFold96(
    const StatPtrs &sb, uint32_t R, __ubuf__ float *&pAccC, uint32_t &curR, bool &hasRV) {
    OpVcadd(sb.pAccR, sb.pWork0, static_cast<int32_t>(R), 1, 1, 12);
    set_vector_mask(0ULL, 0xFFFFFFFFULL);
    OpVcadd(sb.pTmpVec, sb.pWork0 + 64U, static_cast<int32_t>(R), 1, 1, 12);
    set_vector_mask(~0ULL, ~0ULL);
    if (curR > 1U) {
        uint32_t halfR = curR >> 1U;
        uint32_t remRows = curR - halfR;
        if (curR > 2U && (remRows & 1U) != 0U) {
            remRows += 1U;
            halfR = curR - remRows;
        }
        VecAddW96(sb.pWork1, sb.pWork0, sb.pWork0 + remRows * 96U, halfR);
        if (remRows > halfR) {
            CopySpan96(sb.pWork1 + halfR * 96U, sb.pWork0 + halfR * 96U, remRows - halfR);
        }
        pAccC = sb.pWork1;
        curR = remRows;
    }
    AscendC::PipeBarrier<PIPE_V>();
    ApplyVecMask(R);
    OpVadd(sb.pAccR, sb.pAccR, sb.pTmpVec, 1U);
    set_vector_mask(~0ULL, ~0ULL);
    if (curR > 1U) { curR = StepFoldRows<96U>(pAccC, curR); }
    AscendC::PipeBarrier<PIPE_V>();
    if (curR > 1U) {
        ApplyVecMask(R);
        OpVmul(sb.pAccRV, sb.pAccR, sb.pAxisV, 1U);
        set_vector_mask(~0ULL, ~0ULL);
        hasRV = true;
        curR = StepFoldRows<96U>(pAccC, curR);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <uint32_t W, bool FAST_M>
KPT_INL void FinishStatVar(
    const StatPtrs &sb, __ubuf__ float *pAccC, uint32_t R, bool hasRV) {
    __ubuf__ float *pAccCU = sb.pWork2 + 64U;
    OpVcadd(sb.pTmpVec + 0U, pAccC, 1, 1, 1, 8);
    OpVmul(pAccCU, pAccC, sb.pAxisU, 1U);
    if constexpr (W == 96U) {
        set_vector_mask(0ULL, 0xFFFFFFFFULL);
        OpVcadd(sb.pTmpVec + 24U, pAccC + 64U, 1, 1, 1, 8);
        OpVmul(pAccCU + 64U, pAccC + 64U, sb.pAxisU + 64U, 1U);
    }
    ApplyVecMask(R);
    if (FAST_M || hasRV) {
        OpVcadd(sb.pTmpVec + 16U, sb.pAccRV, 1, 1, 1, 8);
    } else {
        OpVmul(sb.pAccRV, sb.pAccR, sb.pAxisV, 1U);
    }
    set_vector_mask(~0ULL, ~0ULL);
    AscendC::PipeBarrier<PIPE_V>();

    OpVcadd(sb.pTmpVec + 8U, pAccCU, 1, 1, 1, 8);
    if constexpr (W == 96U) {
        set_vector_mask(0ULL, 0xFFFFFFFFULL);
        OpVcadd(sb.pTmpVec + 32U, pAccCU + 64U, 1, 1, 1, 8);
    }
    if constexpr (!FAST_M) {
        if (!hasRV) {
            ApplyVecMask(R);
            OpVcadd(sb.pTmpVec + 16U, sb.pAccRV, 1, 1, 1, 8);
        }
    }
    set_vector_mask(~0ULL, ~0ULL);
}

template <uint32_t W, bool FAST_M>
KPT_INL void EvalStatVar(
    const StatPtrs &sb, const AscendC::LocalTensor<float> &ubTmpVec, uint32_t R, float valM,
    float &outMax, float &resE, float &resU, float &resV) {
    __ubuf__ float *pAccC = sb.pWork0;
    uint32_t curR = R;
    bool hasRV = false;
    if constexpr (W == 64U) {
        InitFold64(sb, R, pAccC, curR, hasRV);
    } else {
        InitFold96(sb, R, pAccC, curR, hasRV);
    }
    while (curR > 1U) {
        curR = StepFoldRows<W>(pAccC, curR);
        AscendC::PipeBarrier<PIPE_V>();
    }
    FinishStatVar<W, FAST_M>(sb, pAccC, R, hasRV);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    outMax = valM;
    resE = (W == 64U) ? ubTmpVec.GetValue(0U) : (ubTmpVec.GetValue(0U) + ubTmpVec.GetValue(24U));
    resU = (W == 64U) ? ubTmpVec.GetValue(8U) : (ubTmpVec.GetValue(8U) + ubTmpVec.GetValue(32U));
    resV = ubTmpVec.GetValue(16U);
}

template <uint32_t W, bool FAST_M = false, bool EXT_36 = false>
KPT_INL void EvalTileStats(
    const AscendC::LocalTensor<float> &ubB, const AscendC::LocalTensor<float> &ubWork0,
    const AscendC::LocalTensor<float> &ubWork1, const AscendC::LocalTensor<float> &ubWork2,
    const AscendC::LocalTensor<float> &ubAxisU, const AscendC::LocalTensor<float> &ubAxisV,
    const AscendC::LocalTensor<float> &ubAccR, const AscendC::LocalTensor<float> &ubAccRV,
    const AscendC::LocalTensor<float> &ubTmpVec, uint32_t M, uint32_t R,
    const __gm__ FusedDisconnectorKeypointSoftArgmaxTilingData *tdGm,
    GM_ADDR bias, uint32_t S, uint32_t s, uint32_t k,
    float &ioInvTau, float &ioStride, float &outBiasVal,
    float &outMax, float &resE, float &resU, float &resV) {
    __ubuf__ float *pB = reinterpret_cast<__ubuf__ float *>(ubB.GetPhyAddr());
    const StatPtrs sb{
        reinterpret_cast<__ubuf__ float *>(ubWork0.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ubWork1.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ubWork2.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ubAxisU.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ubAxisV.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ubAccR.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ubAccRV.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float *>(ubTmpVec.GetPhyAddr())};
    ScanPeak<W, FAST_M>(pB, sb.pWork0, sb.pWork1, sb.pTmpVec, M);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    if (tdGm != nullptr) {
        ioInvTau = tdGm->f0;
        ioStride = tdGm->stride;
    }
    outBiasVal = (S == 1U || s == 0U) ? reinterpret_cast<const __gm__ float *>(bias)[k] : 0.0f;
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    const float valM = ubTmpVec.GetValue(0U);
    EvalScaledExp<W, FAST_M>(sb.pWork0, pB, sb.pWork1, M, valM, ioInvTau);
    if constexpr (FAST_M && W == 64U) {
        EvalStat64<EXT_36>(sb, ubTmpVec, R, valM, outMax, resE, resU, resV);
    } else if constexpr (FAST_M && W == 96U) {
        EvalStat96(sb, ubTmpVec, R, valM, outMax, resE, resU, resV);
    } else {
        EvalStatVar<W, FAST_M>(sb, ubTmpVec, R, valM, outMax, resE, resU, resV);
    }
}

}
