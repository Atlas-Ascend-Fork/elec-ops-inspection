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
#include "kpt_p3.h"
#include "kpt_p5.h"

namespace elec::impl {

KPT_INL void InitGmViews(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores,
    AscendC::GlobalTensor<float> &gmFeat, AscendC::GlobalTensor<float> &gmWeight,
    AscendC::GlobalTensor<float> &gmBias, AscendC::GlobalTensor<float> &gmKeypoints,
    AscendC::GlobalTensor<float> &gmScores) {
    gmFeat.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(feat));
    gmWeight.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight));
    gmBias.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias));
    gmKeypoints.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(keypoints));
    gmScores.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scores));
}

KPT_INL float DotTap3(
    const AscendC::LocalTensor<float> &rowBuf, uint32_t idxC, bool validL, bool validR,
    float w0, float w1, float w2) {
    const float vL = validL ? rowBuf.GetValue(idxC - 1U) : 0.0f;
    const float vM = rowBuf.GetValue(idxC);
    const float vR = validR ? rowBuf.GetValue(idxC + 1U) : 0.0f;
    return w0 * vL + w1 * vM + w2 * vR;
}

KPT_INL void StepScanCh(
    const AscendC::GlobalTensor<float> &gmFeat,
    const AscendC::LocalTensor<float> &ubPrev, const AscendC::LocalTensor<float> &ubCur,
    const AscendC::LocalTensor<float> &ubNext, const AscendC::LocalTensor<float> &ubAcc,
    const AscendC::LocalTensor<float> &ubWeight, const AscendC::DataCopyExtParams &rowParams,
    uint32_t chBase, uint32_t wOff, uint32_t H, uint32_t W,
    uint32_t v, uint32_t u0, uint32_t uLen, uint32_t hasL, uint32_t uLoadStart) {
    const AscendC::DataCopyPadExtParams<float> padZeroF{false, 0U, 0U, 0.0f};
    if (v > 0U) {
        AscendC::DataCopyPad(ubPrev, gmFeat[chBase + (v - 1U) * W + uLoadStart], rowParams, padZeroF);
    }
    AscendC::DataCopyPad(ubCur, gmFeat[chBase + v * W + uLoadStart], rowParams, padZeroF);
    if (v + 1U < H) {
        AscendC::DataCopyPad(ubNext, gmFeat[chBase + (v + 1U) * W + uLoadStart], rowParams, padZeroF);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    const float w00 = ubWeight.GetValue(wOff + 0U), w01 = ubWeight.GetValue(wOff + 1U), w02 = ubWeight.GetValue(wOff + 2U);
    const float w10 = ubWeight.GetValue(wOff + 3U), w11 = ubWeight.GetValue(wOff + 4U), w12 = ubWeight.GetValue(wOff + 5U);
    const float w20 = ubWeight.GetValue(wOff + 6U), w21 = ubWeight.GetValue(wOff + 7U), w22 = ubWeight.GetValue(wOff + 8U);

    for (uint32_t u = 0U; u < uLen; ++u) {
        const uint32_t idxC = u + hasL;
        const bool validL = (u0 + u) > 0U;
        const bool validR = (u0 + u + 1U) < W;
        float acc = ubAcc.GetValue(u) + DotTap3(ubCur, idxC, validL, validR, w10, w11, w12);
        if (v > 0U) {
            acc += DotTap3(ubPrev, idxC, validL, validR, w00, w01, w02);
        }
        if (v + 1U < H) {
            acc += DotTap3(ubNext, idxC, validL, validR, w20, w21, w22);
        }
        ubAcc.SetValue(u, acc);
    }
}

KPT_INL void StepScanStat(
    const AscendC::LocalTensor<float> &ubAcc, const AscendC::LocalTensor<float> &ubExp,
    uint32_t v, uint32_t u0, uint32_t uLen, float f0,
    bool &initTile, float &gMax, float &totE, float &totU, float &totV) {
    float tileMax = ubAcc.GetValue(0U);
    for (uint32_t u = 1U; u < uLen; ++u) {
        const float val = ubAcc.GetValue(u);
        if (val > tileMax) {
            tileMax = val;
        }
    }
    const float newMax = (initTile || tileMax > gMax) ? tileMax : gMax;
    for (uint32_t u = 0U; u < uLen; ++u) {
        const float z = (ubAcc.GetValue(u) - newMax) * f0;
        ubExp.SetValue(u, (z < -104.0f) ? -104.0f : z);
    }
    const float zScale = initTile ? 0.0f : ((gMax - newMax) * f0);
    ubExp.SetValue(uLen, (zScale < -104.0f) ? -104.0f : zScale);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::SetMaskNorm();
    set_vector_mask(~0ULL, ~0ULL);
    AscendC::Exp(ubExp, ubExp, uLen + 1U);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

    const float scale = initTile ? 0.0f : ubExp.GetValue(uLen);
    float tileSum = 0.0f;
    float tileSumPU = 0.0f;
    for (uint32_t u = 0U; u < uLen; ++u) {
        const float e = ubExp.GetValue(u);
        tileSum += e;
        tileSumPU += e * static_cast<float>(static_cast<int32_t>(u0 + u));
    }
    totE = totE * scale + tileSum;
    totU = totU * scale + tileSumPU;
    totV = totV * scale + tileSum * static_cast<float>(static_cast<int32_t>(v));
    gMax = newMax;
    initTile = false;
}

KPT_INL void ExecScanTask(
    const AscendC::GlobalTensor<float> &gmFeat,
    const AscendC::LocalTensor<float> &ubPrev, const AscendC::LocalTensor<float> &ubCur,
    const AscendC::LocalTensor<float> &ubNext, const AscendC::LocalTensor<float> &ubAcc,
    const AscendC::LocalTensor<float> &ubExp, const AscendC::LocalTensor<float> &ubWeight,
    uint32_t inBatchOff, uint32_t wtChOff, uint32_t H, uint32_t W, uint32_t HW, float f0,
    float &outMax, float &resE, float &resU, float &resV) {
    constexpr uint32_t TW = 256U;
    bool initTile = true;
    outMax = 0.0f;
    resE = 0.0f;
    resU = 0.0f;
    resV = 0.0f;
    for (uint32_t v = 0U; v < H; ++v) {
        for (uint32_t u0 = 0U; u0 < W; u0 += TW) {
            const uint32_t uLen = ((W - u0) < TW) ? (W - u0) : TW;
            const uint32_t hasL = (u0 > 0U) ? 1U : 0U;
            const uint32_t hasR = ((u0 + uLen) < W) ? 1U : 0U;
            const uint32_t uLoadStart = u0 - hasL;
            const AscendC::DataCopyExtParams rowParams{
                1U, (uLen + hasL + hasR) * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            for (uint32_t u = 0U; u < uLen; ++u) {
                ubAcc.SetValue(u, 0.0f);
            }
            for (uint32_t c = 0U; c < DIM_C0; ++c) {
                StepScanCh(
                    gmFeat, ubPrev, ubCur, ubNext, ubAcc, ubWeight, rowParams,
                    inBatchOff + c * HW, wtChOff + c * DIM_K9,
                    H, W, v, u0, uLen, hasL, uLoadStart);
            }
            StepScanStat(
                ubAcc, ubExp, v, u0, uLen, f0, initTile, outMax, resE, resU, resV);
        }
    }
}

KPT_INL void EntryScanFallback(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &td) {
    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t H = td.height, W = td.width, HW = H * W;
    const uint32_t n0 = td.n0, totalTasks = td.batch * DIM_C1;
    if (n0 == 0U || blockIdx >= n0 || blockIdx >= totalTasks || H == 0U || W == 0U) {
        return;
    }

    AscendC::GlobalTensor<float> gmFeat, gmWeight, gmBias, gmKeypoints, gmScores;
    InitGmViews(feat, weight, bias, keypoints, scores, gmFeat, gmWeight, gmBias, gmKeypoints, gmScores);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufAll;
    constexpr uint32_t LINE_SPAN = 264U;
    constexpr uint32_t BUF_LEN = 5U * LINE_SPAN + DIM_WT + 32U;
    pipe.InitBuffer(bufAll, BUF_LEN * static_cast<uint32_t>(sizeof(float)));

    AscendC::LocalTensor<float> ubBase = bufAll.Get<float>();
    AscendC::LocalTensor<float> ubWeight = ubBase[5U * LINE_SPAN];
    AscendC::LocalTensor<float> ubBias = ubBase[5U * LINE_SPAN + DIM_WT];

    AscendC::DataCopy(ubWeight, gmWeight, DIM_WT);
    const AscendC::DataCopyExtParams biasCopyParams{1U, DIM_C1 * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> padZeroF{false, 0U, 0U, 0.0f};
    AscendC::DataCopyPad(ubBias, gmBias, biasCopyParams, padZeroF);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    for (uint32_t taskIdx = blockIdx; taskIdx < totalTasks; taskIdx += n0) {
        const uint32_t k = taskIdx & 3U;
        float gMax = 0.0f, totE = 0.0f, totU = 0.0f, totV = 0.0f;
        ExecScanTask(
            gmFeat, ubBase[0U], ubBase[LINE_SPAN], ubBase[2U * LINE_SPAN],
            ubBase[3U * LINE_SPAN], ubBase[4U * LINE_SPAN], ubWeight,
            (taskIdx >> 2U) * DIM_C0 * HW, k * DIM_C0 * DIM_K9,
            H, W, HW, td.f0, gMax, totE, totU, totV);
        StoreTaskOut(
            gmKeypoints, gmScores, ubBias[8U], ubBias[16U], taskIdx,
            totE, totU, totV, gMax, td.stride, ubBias.GetValue(k));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }
}

KPT_INL void StepFullCh(
    const AscendC::LocalTensor<float> &ubA, const AscendC::LocalTensor<float> &ubB, const AscendC::LocalTensor<float> &ubC,
    const AscendC::LocalTensor<float> &f, const AscendC::LocalTensor<float> &ubWeight,
    uint32_t wOff, uint32_t W, uint32_t HW, uint32_t hwMinusW, event_t curEv) {
    const float w00 = ubWeight.GetValue(wOff + 0U), w01 = ubWeight.GetValue(wOff + 1U), w02 = ubWeight.GetValue(wOff + 2U);
    const float w10 = ubWeight.GetValue(wOff + 3U), w11 = ubWeight.GetValue(wOff + 4U), w12 = ubWeight.GetValue(wOff + 5U);
    const float w20 = ubWeight.GetValue(wOff + 6U), w21 = ubWeight.GetValue(wOff + 7U), w22 = ubWeight.GetValue(wOff + 8U);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(curEv);
    AscendC::Axpy(ubA[W], f, w00, hwMinusW);
    AscendC::Axpy(ubB[W], f, w01, hwMinusW);
    AscendC::Axpy(ubC[W], f, w02, hwMinusW);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Axpy(ubA, f, w10, HW);
    AscendC::Axpy(ubB, f, w11, HW);
    AscendC::Axpy(ubC, f, w12, HW);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Axpy(ubA, f[W], w20, hwMinusW);
    AscendC::Axpy(ubB, f[W], w21, hwMinusW);
    AscendC::Axpy(ubC, f[W], w22, hwMinusW);
    AscendC::PipeBarrier<PIPE_V>();
}

KPT_INL void ExecFullPipe(
    const AscendC::GlobalTensor<float> &gmFeat,
    const AscendC::LocalTensor<float> &ubA, const AscendC::LocalTensor<float> &ubB, const AscendC::LocalTensor<float> &ubC,
    const AscendC::LocalTensor<float> &ubFeat0, const AscendC::LocalTensor<float> &ubFeat1,
    const AscendC::LocalTensor<float> &ubWeight,
    uint32_t inBatchOff, uint32_t wtChOff, uint32_t W, uint32_t HW) {
    const uint32_t hwMinusW = HW - W;
    AscendC::SetMaskNorm();
    set_vector_mask(~0ULL, ~0ULL);
    AscendC::DataCopy(ubFeat0, gmFeat[inBatchOff], HW);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Duplicate(ubA, 0.0f, HW + 8U);
    AscendC::Duplicate(ubB, 0.0f, HW);
    AscendC::Duplicate(ubC, 0.0f, HW + 8U);
    AscendC::PipeBarrier<PIPE_V>();

    for (uint32_t c = 0U; c < DIM_C0; ++c) {
        const uint32_t curBuf = c & 1U;
        const uint32_t nxtBuf = curBuf ^ 1U;
        const event_t curEv = curBuf ? EVENT_ID1 : EVENT_ID0;
        const event_t nxtEv = nxtBuf ? EVENT_ID1 : EVENT_ID0;
        const AscendC::LocalTensor<float> &f = curBuf ? ubFeat1 : ubFeat0;
        const AscendC::LocalTensor<float> &fNxt = nxtBuf ? ubFeat1 : ubFeat0;

        if (c + 1U < DIM_C0) {
            if (c >= 1U) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(nxtEv);
            }
            AscendC::DataCopy(fNxt, gmFeat[inBatchOff + (c + 1U) * HW], HW);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(nxtEv);
        }
        StepFullCh(ubA, ubB, ubC, f, ubWeight, wtChOff + c * DIM_K9, W, HW, hwMinusW, curEv);
        if (c + 2U < DIM_C0) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(curEv);
        }
    }
}

KPT_INL void GatherFullHoriz(
    const AscendC::LocalTensor<float> &ubAPrev8, const AscendC::LocalTensor<float> &ubB,
    const AscendC::LocalTensor<float> &ubC, const AscendC::LocalTensor<float> &ubFeat0,
    const AscendC::LocalTensor<float> &ubFeat1, uint32_t H, uint32_t W, uint32_t HW) {
    uint64_t maskFirst[2] = {1ULL, 0ULL};
    uint64_t maskLast[2] = {0x80ULL, 0ULL};
    const uint8_t wBlocks = static_cast<uint8_t>(W >> 3U);
    AscendC::LocalTensor<uint32_t> ubIdx = ubFeat0.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<int32_t> ubIdxI32 = ubFeat0.ReinterpretCast<int32_t>();
    AscendC::LocalTensor<uint32_t> ubShiftU32 = ubFeat1.ReinterpretCast<uint32_t>();

    AscendC::SetMaskNorm();
    set_vector_mask(~0ULL, ~0ULL);
    AscendC::ArithProgression(ubIdxI32, 0, 4, static_cast<int32_t>(HW));
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Gather(ubFeat1, ubAPrev8, ubIdx, 28U, HW);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate<uint32_t>(ubShiftU32, 0U, maskFirst, static_cast<uint8_t>(H), 1U, wBlocks);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    set_vector_mask(~0ULL, ~0ULL);
    AscendC::Add(ubB, ubB, ubFeat1, HW);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Gather(ubFeat1, ubC, ubIdx, 4U, HW);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate<uint32_t>(ubShiftU32[W - 8U], 0U, maskLast, static_cast<uint8_t>(H), 1U, wBlocks);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskNorm();
    set_vector_mask(~0ULL, ~0ULL);
    AscendC::Add(ubB, ubB, ubFeat1, HW);
    AscendC::PipeBarrier<PIPE_V>();
}

KPT_INL uint32_t HalvePeak(const AscendC::LocalTensor<float> &buf, uint32_t len) {
    while (len > 64U) {
        const uint32_t rem = (((len + 1U) >> 1U) + 7U) & ~7U;
        AscendC::Max(buf, buf, buf[rem], len - rem);
        AscendC::PipeBarrier<PIPE_V>();
        len = rem;
    }
    return len;
}

KPT_INL uint32_t HalveSum(const AscendC::LocalTensor<float> &buf, uint32_t len) {
    while (len > 64U) {
        const uint32_t rem = (((len + 1U) >> 1U) + 7U) & ~7U;
        AscendC::Add(buf, buf, buf[rem], len - rem);
        AscendC::PipeBarrier<PIPE_V>();
        len = rem;
    }
    return len;
}

KPT_INL void CollapseGrid2D(
    const AscendC::LocalTensor<float> &ubFeat0, const AscendC::LocalTensor<float> &ubAccR,
    const AscendC::LocalTensor<float> &ubTmpVec, uint32_t H, uint32_t W) {
    const uint32_t rowBlkStride = W >> 3U;
    if (W <= 64U) {
        AscendC::WholeReduceSum<float>(
            ubAccR, ubFeat0, static_cast<int32_t>(W), static_cast<int32_t>(H), 1, 1, rowBlkStride);
    } else {
        AscendC::WholeReduceSum<float>(ubAccR, ubFeat0, 64U, static_cast<int32_t>(H), 1, 1, rowBlkStride);
        AscendC::WholeReduceSum<float>(ubTmpVec[16], ubFeat0[64], W - 64U, static_cast<int32_t>(H), 1, 1, rowBlkStride);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(ubAccR, ubAccR, ubTmpVec[16], H);
    }
    AscendC::PipeBarrier<PIPE_V>();

    for (uint32_t curH = H; curH > 1U;) {
        const uint32_t half = curH >> 1U;
        const uint32_t rem = curH - half;
        AscendC::Add(ubFeat0, ubFeat0, ubFeat0[rem * W], half * W);
        AscendC::PipeBarrier<PIPE_V>();
        curH = rem;
    }
}

KPT_INL void EvalFullStat(
    const AscendC::LocalTensor<float> &ubA, const AscendC::LocalTensor<float> &ubB,
    const AscendC::LocalTensor<float> &ubFeat0, const AscendC::LocalTensor<float> &ubMisc,
    uint32_t H, uint32_t W, uint32_t HW, float f0,
    float &outMax, float &resE, float &resU, float &resV) {
    AscendC::LocalTensor<float> ubAccR = ubMisc[256];
    AscendC::LocalTensor<float> ubAccCU = ubMisc[384];
    AscendC::LocalTensor<float> ubAccRV = ubMisc[512];
    AscendC::LocalTensor<float> ubTmpVec = ubMisc[640];

    AscendC::Adds(ubA, ubB, 0.0f, HW);
    AscendC::PipeBarrier<PIPE_V>();
    const uint32_t curLen = HalvePeak(ubA, HW);
    AscendC::WholeReduceMax<float>(
        ubTmpVec, ubA, static_cast<int32_t>(curLen), 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_VALUE_INDEX);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    outMax = ubTmpVec.GetValue(0U);

    AscendC::Adds(ubFeat0, ubB, -outMax, HW);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(ubFeat0, ubFeat0, f0, HW);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(ubFeat0, ubFeat0, HW);
    AscendC::PipeBarrier<PIPE_V>();

    CollapseGrid2D(ubFeat0, ubAccR, ubTmpVec, H, W);
    AscendC::Mul(ubAccCU, ubFeat0, ubMisc[0], W);
    AscendC::Mul(ubAccRV, ubAccR, ubMisc[128], H);
    AscendC::PipeBarrier<PIPE_V>();

    const uint32_t lenW = HalveSum(ubFeat0, W);
    (void)HalveSum(ubAccCU, W);
    const uint32_t lenH = HalveSum(ubAccRV, H);
    AscendC::WholeReduceSum<float>(ubTmpVec[0], ubFeat0, static_cast<int32_t>(lenW), 1, 1, 1, 8);
    AscendC::WholeReduceSum<float>(ubTmpVec[8], ubAccCU, static_cast<int32_t>(lenW), 1, 1, 1, 8);
    AscendC::WholeReduceSum<float>(ubTmpVec[16], ubAccRV, static_cast<int32_t>(lenH), 1, 1, 1, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

    resE = ubTmpVec.GetValue(0U);
    resU = ubTmpVec.GetValue(8U);
    resV = ubTmpVec.GetValue(16U);
}

KPT_INL void ExecFullTask(
    const AscendC::GlobalTensor<float> &gmFeat,
    const AscendC::GlobalTensor<float> &gmKeypoints,
    const AscendC::GlobalTensor<float> &gmScores,
    const AscendC::LocalTensor<float> &ubBase,
    const AscendC::LocalTensor<float> &ubMisc,
    uint32_t offA, uint32_t offB, uint32_t offC, uint32_t offF0, uint32_t offF1,
    uint32_t taskIdx, uint32_t H, uint32_t W, uint32_t HW, float f0, float stride) {
    const uint32_t k = taskIdx & 3U;
    ExecFullPipe(
        gmFeat, ubBase[offA >> 2U], ubBase[offB >> 2U], ubBase[offC >> 2U],
        ubBase[offF0 >> 2U], ubBase[offF1 >> 2U], ubMisc[1024],
        (taskIdx >> 2U) * DIM_C0 * HW, k * DIM_C0 * DIM_K9, W, HW);
    GatherFullHoriz(
        ubBase[(offA - 32U) >> 2U], ubBase[offB >> 2U], ubBase[offC >> 2U],
        ubBase[offF0 >> 2U], ubBase[offF1 >> 2U], H, W, HW);
    float rawPeak = 0.0f, totE = 0.0f, totU = 0.0f, totV = 0.0f;
    EvalFullStat(
        ubBase[offA >> 2U], ubBase[offB >> 2U], ubBase[offF0 >> 2U], ubMisc,
        H, W, HW, f0, rawPeak, totE, totU, totV);
    StoreTaskOut(
        gmKeypoints, gmScores, ubMisc[2184], ubMisc[2192], taskIdx,
        totE, totU, totV, rawPeak, stride, ubMisc[2176].GetValue(k));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

KPT_INL void LoadFullMeta(
    const AscendC::GlobalTensor<float> &gmWeight,
    const AscendC::GlobalTensor<float> &gmBias,
    const AscendC::LocalTensor<float> &ubMisc) {
    AscendC::DataCopy(ubMisc[1024], gmWeight, DIM_WT);
    const AscendC::DataCopyExtParams biasCopyParams{1U, DIM_C1 * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> biasPadParams{false, 0U, 0U, 0.0f};
    AscendC::DataCopyPad(ubMisc[2176], gmBias, biasCopyParams, biasPadParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
    AscendC::SetMaskNorm();
    set_vector_mask(~0ULL, ~0ULL);
    AscendC::ArithProgression(ubMisc[0], 0.0f, 1.0f, static_cast<int32_t>(DIM_CR));
    AscendC::ArithProgression(ubMisc[128], 0.0f, 1.0f, static_cast<int32_t>(DIM_CR));
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
}

KPT_INL void EntryModeFull(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores, GM_ADDR tiling) {
    if ASCEND_IS_AIC {
        return;
    }
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(FusedDisconnectorKeypointSoftArgmaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedDisconnectorKeypointSoftArgmaxTilingData, td, tiling);

    const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t n0 = td.n0, totalTasks = td.batch * DIM_C1;
    const uint32_t H = td.height, W = td.width;
    if (n0 == 0U || blockIdx >= n0 || blockIdx >= totalTasks || H == 0U || W == 0U) {
        return;
    }
    if (EvalQuickPath(weight, bias, keypoints, scores, tiling, blockIdx, td.batch, H, W, td.stride, false, false)) {
        return;
    }
    const uint32_t HW = H * W;
    if (H <= 1U || H > 128U || W > 128U || (W & 7U) != 0U || HW > 6912U) {
        EntryScanFallback(feat, weight, bias, keypoints, scores, td);
        return;
    }

    AscendC::GlobalTensor<float> gmFeat, gmWeight, gmBias, gmKeypoints, gmScores;
    InitGmViews(feat, weight, bias, keypoints, scores, gmFeat, gmWeight, gmBias, gmKeypoints, gmScores);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufAll;
    constexpr uint32_t PLANE_CAP = (6912U + 8U) * static_cast<uint32_t>(sizeof(float));
    constexpr uint32_t OFF_A = 512U;
    constexpr uint32_t OFF_B = AlignStep(OFF_A + PLANE_CAP, 0U);
    constexpr uint32_t OFF_C = AlignStep(OFF_B + PLANE_CAP, 0U);
    constexpr uint32_t OFF_F0 = AlignStep(OFF_C + PLANE_CAP, 64U);
    constexpr uint32_t OFF_F1 = AlignStep(OFF_F0 + PLANE_CAP, 64U);
    constexpr uint32_t OFF_MISC = AlignStep(OFF_F1 + PLANE_CAP, 0U);
    constexpr uint32_t BUF_BYTES = OFF_MISC + 2560U * static_cast<uint32_t>(sizeof(float));
    static_assert(OFF_A >= 32U && BUF_BYTES <= 196608U, "ub_cfg");

    pipe.InitBuffer(bufAll, BUF_BYTES);
    AscendC::LocalTensor<float> ubBase = bufAll.Get<float>();
    AscendC::LocalTensor<float> ubMisc = ubBase[OFF_MISC >> 2U];
    LoadFullMeta(gmWeight, gmBias, ubMisc);

    for (uint32_t taskIdx = blockIdx; taskIdx < totalTasks; taskIdx += n0) {
        ExecFullTask(
            gmFeat, gmKeypoints, gmScores, ubBase, ubMisc,
            OFF_A, OFF_B, OFF_C, OFF_F0, OFF_F1, taskIdx, H, W, HW, td.f0, td.stride);
    }
}

}
