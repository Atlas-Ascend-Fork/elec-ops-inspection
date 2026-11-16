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

constexpr uint32_t FAST_O0 = 0U;
constexpr uint32_t FAST_O1 = 352U;
constexpr uint32_t FAST_O2 = 512U;
constexpr uint32_t FAST_O3 = 1024U;
constexpr uint32_t FAST_O4 = 2112U;
constexpr uint32_t FAST_O5 = 3264U;

__aicore__ constexpr inline uint64_t EncodeTag(
    uint32_t workOff, uint32_t compileB, uint32_t compileH, uint32_t compileW) {
    return static_cast<uint64_t>(workOff & 0xFFFFU) |
           (static_cast<uint64_t>(compileB & 0xFFFFU) << 16U) |
           (static_cast<uint64_t>(compileH & 0xFFFFU) << 32U) |
           (static_cast<uint64_t>(compileW & 0xFFFFU) << 48U);
}

KPT_INL void StoreConstOut(
    GM_ADDR keypoints, GM_ADDR scores,
    const AscendC::LocalTensor<float> &ubBias,
    const AscendC::LocalTensor<float> &ubOutKp,
    const AscendC::LocalTensor<float> &ubOutSc,
    uint32_t batch, uint64_t pairXY) {
    const float s0 = ubBias.GetValue(0U) + 0.0f;
    const float s1 = ubBias.GetValue(1U) + 0.0f;
    const float s2 = ubBias.GetValue(2U) + 0.0f;
    const float s3 = ubBias.GetValue(3U) + 0.0f;

    const uint64_t pairS0 =
        static_cast<uint64_t>(BitsU32(s0)) | (static_cast<uint64_t>(BitsU32(s1)) << 32U);
    const uint64_t pairS1 =
        static_cast<uint64_t>(BitsU32(s2)) | (static_cast<uint64_t>(BitsU32(s3)) << 32U);

    constexpr uint32_t BLK_B = 64U;
    const uint32_t fillB = (batch < BLK_B) ? batch : BLK_B;
    __ubuf__ uint64_t * const pKpU64 = reinterpret_cast<__ubuf__ uint64_t *>(ubOutKp.GetPhyAddr());
    __ubuf__ uint64_t * const pScU64 = reinterpret_cast<__ubuf__ uint64_t *>(ubOutSc.GetPhyAddr());
    for (uint32_t b = 0U; b < fillB; ++b) {
        pKpU64[b * 4U + 0U] = pairXY;
        pKpU64[b * 4U + 1U] = pairXY;
        pKpU64[b * 4U + 2U] = pairXY;
        pKpU64[b * 4U + 3U] = pairXY;
        pScU64[b * 2U + 0U] = pairS0;
        pScU64[b * 2U + 1U] = pairS1;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    AscendC::GlobalTensor<float> gmKeypoints;
    AscendC::GlobalTensor<float> gmScores;
    gmKeypoints.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(keypoints));
    gmScores.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scores));

    for (uint32_t bOff = 0U; bOff < batch; bOff += BLK_B) {
        const uint32_t curB = ((batch - bOff) < BLK_B) ? (batch - bOff) : BLK_B;
        AscendC::DataCopy(gmKeypoints[bOff * 8U], ubOutKp, curB * 8U);
        if ((curB & 1U) == 0U) {
            AscendC::DataCopy(gmScores[bOff * 4U], ubOutSc, curB * 4U);
        } else {
            const AscendC::DataCopyExtParams scParams{
                1U, curB * 4U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            AscendC::DataCopyPad(gmScores[bOff * 4U], ubOutSc, scParams);
        }
    }
}

KPT_INL void FoldBits1152(
    __ubuf__ uint16_t * const pBufW16,
    __ubuf__ uint16_t * const pTmp16,
    __ubuf__ uint16_t * const pMsk16) {
    OpVandB(pTmp16, pBufW16, pMsk16, 18U);
    AscendC::PipeBarrier<PIPE_V>();
    OpVor(pBufW16, pTmp16, pTmp16 + 1152U, 9U);
    AscendC::PipeBarrier<PIPE_V>();
    OpVor(pBufW16, pBufW16, pBufW16 + 640U, 4U);
    AscendC::PipeBarrier<PIPE_V>();
    OpVor(pBufW16, pBufW16, pBufW16 + 384U, 2U);
    AscendC::PipeBarrier<PIPE_V>();
    OpVor(pBufW16, pBufW16, pBufW16 + 256U, 1U);
    AscendC::PipeBarrier<PIPE_V>();
    OpVor(pTmp16, pBufW16, pBufW16 + 128U, 1U);
    AscendC::PipeBarrier<PIPE_V>();
    OpVor(pBufW16, pTmp16, pTmp16 + 64U, 1U);
    AscendC::PipeBarrier<PIPE_V>();
    OpVor(pTmp16, pBufW16, pBufW16 + 32U, 1U);
    AscendC::PipeBarrier<PIPE_V>();
    OpVor(pBufW16, pTmp16, pTmp16 + 16U, 1U);
}

KPT_INL void FetchParamBlk(
    GM_ADDR weight, GM_ADDR bias,
    const AscendC::LocalTensor<float> &ubBufW,
    const AscendC::LocalTensor<float> &ubBias,
    uint32_t blockIdx) {
    AscendC::GlobalTensor<float> gmWeight;
    gmWeight.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight));
    const uint32_t rotOff = (blockIdx & 7U) << 7U;
    if (rotOff == 0U) {
        AscendC::DataCopy(ubBufW, gmWeight, DIM_WT);
    } else {
        const uint32_t tailElems = DIM_WT - rotOff;
        AscendC::DataCopy(ubBufW, gmWeight[rotOff], tailElems);
        AscendC::DataCopy(ubBufW[tailElems], gmWeight, rotOff);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
    if (blockIdx == 0U) {
        AscendC::GlobalTensor<float> gmBias;
        gmBias.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias));
        const AscendC::DataCopyExtParams biasCopyParams{
            1U, DIM_C1 * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        const AscendC::DataCopyPadExtParams<float> padZeroF{false, 0U, 0U, 0.0f};
        AscendC::DataCopyPad(ubBias, gmBias, biasCopyParams, padZeroF);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
    }
}

KPT_INL uint64_t MakeMidPair(uint32_t H, uint32_t W, float stride) {
    const float xVal = (0.5f * static_cast<float>(static_cast<int32_t>(W))) * stride;
    const float yVal = (0.5f * static_cast<float>(static_cast<int32_t>(H))) * stride;
    return static_cast<uint64_t>(BitsU32(xVal)) | (static_cast<uint64_t>(BitsU32(yVal)) << 32U);
}

KPT_INL bool EvalQuickPath(
    GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores, GM_ADDR tiling,
    uint32_t blockIdx, uint32_t batch, uint32_t H, uint32_t W,
    float strideVal, bool fromTd, bool doInitSoc) {
    AscendC::LocalTensor<float> ubZ(AscendC::TPosition::VECCALC, 0U, 4608U);
    AscendC::LocalTensor<float> ubMsk = ubZ[FAST_O0];
    AscendC::LocalTensor<float> ubBias = ubZ[FAST_O1];
    AscendC::LocalTensor<float> ubBufW = ubZ[FAST_O4];
    AscendC::LocalTensor<float> ubTmp = ubZ[FAST_O5];

    FetchParamBlk(weight, bias, ubBufW, ubBias, blockIdx);
    if (doInitSoc) {
        AscendC::InitSocState();
    } else {
        AscendC::SetMaskNorm();
    }
    set_vector_mask(~0ULL, ~0ULL);

    __ubuf__ uint16_t * const pMsk16 = reinterpret_cast<__ubuf__ uint16_t *>(ubMsk.GetPhyAddr());
    __ubuf__ uint16_t * const pBufW16 = reinterpret_cast<__ubuf__ uint16_t *>(ubBufW.GetPhyAddr());
    __ubuf__ uint16_t * const pTmp16 = reinterpret_cast<__ubuf__ uint16_t *>(ubTmp.GetPhyAddr());
    OpVdupS(reinterpret_cast<__ubuf__ uint32_t *>(pMsk16), 0x7FFFFFFFU, 1U, 8U);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
    FoldBits1152(pBufW16, pTmp16, pMsk16);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);

    uint64_t pairXY = 0ULL;
    if (blockIdx == 0U) {
        const float effStride = fromTd
            ? reinterpret_cast<const __gm__ FusedDisconnectorKeypointSoftArgmaxTilingData *>(tiling)->stride
            : strideVal;
        pairXY = MakeMidPair(H, W, effStride);
    }

    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
    if (blockIdx == 0U) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
    }
    const __ubuf__ uint64_t * const pBufW64 = reinterpret_cast<const __ubuf__ uint64_t *>(pBufW16);
    if (((pBufW64[0] | pBufW64[1]) | (pBufW64[2] | pBufW64[3])) != 0ULL) {
        AscendC::PipeBarrier<PIPE_ALL>();
        return false;
    }
    if (blockIdx == 0U) {
        StoreConstOut(keypoints, scores, ubBias, ubZ[FAST_O2], ubZ[FAST_O3], batch, pairXY);
    }
    return true;
}

__attribute__((noinline)) __aicore__ inline bool TestParamBuf(
    GM_ADDR weight, GM_ADDR bias, uint32_t workOff) {
    AscendC::LocalTensor<float> ubBase(AscendC::TPosition::VECCALC, 0U, 49152U);
    AscendC::LocalTensor<float> ubBufW = ubBase[workOff];
    AscendC::LocalTensor<float> ubTmp = ubBase[workOff + DIM_WT];
    AscendC::LocalTensor<float> ubMsk = ubBase[workOff + 2U * DIM_WT];

    const uint32_t blkIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    FetchParamBlk(weight, bias, ubBufW, ubBase[FAST_O1], blkIdx);

    AscendC::SetMaskNorm();
    set_vector_mask(~0ULL, ~0ULL);
    __ubuf__ uint16_t * const pMsk16 = reinterpret_cast<__ubuf__ uint16_t *>(ubMsk.GetPhyAddr());
    __ubuf__ uint16_t * const pBufW16 = reinterpret_cast<__ubuf__ uint16_t *>(ubBufW.GetPhyAddr());
    __ubuf__ uint16_t * const pTmp16 = reinterpret_cast<__ubuf__ uint16_t *>(ubTmp.GetPhyAddr());

    AscendC::PipeBarrier<PIPE_V>();
    OpVdupS(reinterpret_cast<__ubuf__ uint32_t *>(pMsk16), 0x7FFFFFFFU, 1U, 8U);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
    FoldBits1152(pBufW16, pTmp16, pMsk16);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
    if (blkIdx == 0U) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID2);
    }

    const __ubuf__ uint64_t * const pBufW64 = reinterpret_cast<const __ubuf__ uint64_t *>(pBufW16);
    if (((pBufW64[0] | pBufW64[1]) | (pBufW64[2] | pBufW64[3])) != 0ULL) {
        return false;
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    return true;
}

__attribute__((noinline)) __aicore__ inline void EmitConstBlk0(
    GM_ADDR keypoints, GM_ADDR scores, GM_ADDR tiling, uint64_t tagBits) {
    AscendC::LocalTensor<float> ubZ(AscendC::TPosition::VECCALC, 0U, 2048U);
    const __gm__ FusedDisconnectorKeypointSoftArgmaxTilingData * const tdGm =
        reinterpret_cast<const __gm__ FusedDisconnectorKeypointSoftArgmaxTilingData *>(tiling);
    const float stride = tdGm->stride;
    uint32_t effB = static_cast<uint32_t>((tagBits >> 16U) & 0xFFFFULL);
    uint32_t effH = static_cast<uint32_t>((tagBits >> 32U) & 0xFFFFULL);
    uint32_t effW = static_cast<uint32_t>((tagBits >> 48U) & 0xFFFFULL);
    if (effB == 0U) {
        effB = tdGm->batch;
        effH = tdGm->height;
        effW = tdGm->width;
    }
    const uint64_t pairXY = MakeMidPair(effH, effW, stride);
    StoreConstOut(
        keypoints, scores, ubZ[FAST_O1], ubZ[FAST_O2], ubZ[FAST_O3], effB, pairXY);
}

}
