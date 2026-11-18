/*
 * Copyright (c) 2026 hit_op-samerberry.
 * SPDX-License-Identifier: Apache-2.0
 *
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

#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "tiling_key_fused_disconnector_keypoint_soft_argmax.h"

using namespace AscendC;

namespace {

constexpr uint32_t kUbTotalFloats = 192u * 1024u / sizeof(float);
constexpr uint32_t kWeightFloats = 1152u;
constexpr uint32_t kHalfWeightFloats = 576u;
constexpr uint32_t kFilterFloats = 288u;
constexpr uint32_t kPacketFloats = 24u;
constexpr uint32_t kCoordSeedOff = 1280u;
constexpr uint32_t kSlot0BaseOff = 1536u;
constexpr uint32_t kScratchRelOff = 4096u;
constexpr uint32_t kMaxTreeRelOff = 2112u;
constexpr uint32_t kAlphaTreeRelOff = 4224u;
constexpr float kNegMaxSentinel = -3.402823466e+38f;

struct KernelAddrs {
    GM_ADDR feat;
    GM_ADDR weight;
    GM_ADDR bias;
    GM_ADDR keypoints;
    GM_ADDR scores;
    GM_ADDR ws;
};

struct SlotLoadPlan {
    uint32_t groupStride;
    uint32_t topOffset;
    uint32_t padBlockLenBytes;
    uint32_t padSrcStrideBytes;
    uint16_t copyBlkLen;
    uint16_t copySrcStride;
    uint16_t copyDstStride;
};

template <HardEvent E>
__aicore__ __attribute__((always_inline)) inline void SyncFence()
{
    SetFlag<E>(0);
    WaitFlag<E>(0);
}

__aicore__ __attribute__((always_inline)) inline void RawAxpy64(
    __ubuf__ float *dst, __ubuf__ float *src, float scalarVal, uint8_t repeatTime)
{
    if constexpr (g_coreType == AscendC::AIV) {
        vaxpy(dst, src, scalarVal, repeatTime, 1, 1, 8, 8);
    }
}

template <uint32_t KC>
__aicore__ __attribute__((always_inline)) inline void RawAxpyGroup(
    __ubuf__ float *const *dst, __ubuf__ float *src, const __ubuf__ float *wTap, uint8_t oRep)
{
    RawAxpy64(dst[0], src, wTap[0u], oRep);
    RawAxpy64(dst[1], src, wTap[kFilterFloats], oRep);
    if constexpr (KC == 4u) {
        RawAxpy64(dst[2], src, wTap[2u * kFilterFloats], oRep);
        RawAxpy64(dst[3], src, wTap[3u * kFilterFloats], oRep);
    }
}

__aicore__ __attribute__((always_inline)) inline void RawAdd64(
    __ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeatTime)
{
    if constexpr (g_coreType == AscendC::AIV) {
        vadd(dst, src0, src1, repeatTime, 1, 1, 1, 8, 8, 8);
    }
}

__aicore__ __attribute__((always_inline)) inline void RawDup0(
    __ubuf__ float *dst, uint8_t repeatTime, uint8_t dstRepStride)
{
    if constexpr (g_coreType == AscendC::AIV) {
        vector_dup(dst, 0.0f, repeatTime, 1, 1, dstRepStride, 0);
    }
}

__aicore__ __attribute__((always_inline)) inline void RawDupVal64(
    __ubuf__ float *dst, float val, uint8_t repeatTime)
{
    if constexpr (g_coreType == AscendC::AIV) {
        vector_dup(dst, val, repeatTime, 1, 1, 8, 0);
    }
}

__aicore__ __attribute__((always_inline)) inline void Stage1Max(
    const LocalTensor<float> &dst, const LocalTensor<float> &tmp,
    const LocalTensor<float> &src, uint32_t count)
{
    if (count <= 64u) {
        WholeReduceMax(dst, src, static_cast<int32_t>(count), 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    } else {
        const uint32_t full = count >> 6;
        const uint32_t tail = count & 63u;
        WholeReduceMax(tmp, src, 64, static_cast<int32_t>(full), 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        if (tail != 0u) {
            WholeReduceMax(tmp[full], src[full << 6], static_cast<int32_t>(tail),
                           1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        }
    }
}

__aicore__ __attribute__((always_inline)) inline void Stage2Max(
    const LocalTensor<float> &dst, const LocalTensor<float> &tmp, uint32_t count)
{
    if (count > 64u) {
        const uint32_t full = count >> 6;
        const uint32_t tail = count & 63u;
        const int32_t totalBlocks = static_cast<int32_t>(full + (tail != 0u ? 1u : 0u));
        WholeReduceMax(dst, tmp, totalBlocks, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
    }
}

__aicore__ __attribute__((always_inline)) inline void Stage1Sum(
    const LocalTensor<float> &dst, const LocalTensor<float> &tmp,
    const LocalTensor<float> &src, uint32_t count)
{
    if (count <= 64u) {
        WholeReduceSum(dst, src, static_cast<int32_t>(count), 1, 1, 1, 8);
    } else {
        const uint32_t full = count >> 6;
        const uint32_t tail = count & 63u;
        WholeReduceSum(tmp, src, 64, static_cast<int32_t>(full), 1, 1, 8);
        if (tail != 0u) {
            WholeReduceSum(tmp[full], src[full << 6], static_cast<int32_t>(tail), 1, 1, 1, 8);
        }
    }
}

__aicore__ __attribute__((always_inline)) inline void Stage2Sum(
    const LocalTensor<float> &dst, const LocalTensor<float> &tmp, uint32_t count)
{
    if (count > 64u) {
        const uint32_t full = count >> 6;
        const uint32_t tail = count & 63u;
        const int32_t totalBlocks = static_cast<int32_t>(full + (tail != 0u ? 1u : 0u));
        WholeReduceSum(dst, tmp, totalBlocks, 1, 1, 1, 8);
    }
}

__aicore__ __attribute__((always_inline)) inline void IssueSlotLoad(
    const LocalTensor<float> &ub, const GlobalTensor<float> &featGm,
    uint32_t slotOff, uint32_t gBase, const SlotLoadPlan &plan)
{
    DataCopyExtParams padCopy{4, plan.padBlockLenBytes, plan.padSrcStrideBytes, plan.copyDstStride, 0u};
    DataCopyPad(ub[slotOff + plan.topOffset], featGm[gBase],
                padCopy, DataCopyPadExtParams<float>{true, 1, 0, 0.0f});
    DataCopy(ub[slotOff + plan.groupStride + plan.topOffset], featGm[gBase],
             DataCopyParams{4, plan.copyBlkLen, plan.copySrcStride, plan.copyDstStride});
    DataCopyPad(ub[slotOff + 2u * plan.groupStride + plan.topOffset], featGm[gBase + 1u],
                padCopy, DataCopyPadExtParams<float>{true, 0, 1, 0.0f});
}

__aicore__ __attribute__((always_inline)) inline bool IsAllFilterBitsZero(const LocalTensor<float> &ub)
{
    LocalTensor<uint16_t> bits = ub[1536].ReinterpretCast<uint16_t>();
    Or<uint16_t>(bits, ub.ReinterpretCast<uint16_t>(), ub[576].ReinterpretCast<uint16_t>(),
                 128, 9, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Or<uint16_t>(bits, bits, ub[1856].ReinterpretCast<uint16_t>(), 128, 4, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Or<uint16_t>(bits, bits, ub[1728].ReinterpretCast<uint16_t>(), 128, 2, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Or<uint16_t>(bits, bits, ub[1664].ReinterpretCast<uint16_t>(), 128, 1, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Or<uint16_t>(bits, bits, ub[1600].ReinterpretCast<uint16_t>(), 128, 1, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    And<uint16_t>(bits, bits, ub[3072].ReinterpretCast<uint16_t>(), 128, 1, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Cast<float, int32_t>(ub[1536], ub[1536].ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, 64);
    PipeBarrier<PIPE_V>();
    Stage1Max(ub[2752], ub[2688], ub[1536], 64);
    SyncFence<HardEvent::V_S>();
    return ub.GetValue(2752) == 0.0f;
}

__aicore__ __attribute__((always_inline)) inline void WriteUniformCenterOutputs(
    const LocalTensor<float> &ub, const GlobalTensor<float> &keypoints,
    const GlobalTensor<float> &scores, const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    const float strideF = static_cast<float>(t.stride);
    const float centerX = static_cast<float>(static_cast<int32_t>(t.w)) * 0.5f * strideF;
    const float centerY = static_cast<float>(static_cast<int32_t>(t.h)) * 0.5f * strideF;
    uint64_t even[2] = {0x5555555555555555ULL, 0ULL};
    uint64_t odd[2] = {0xaaaaaaaaaaaaaaaaULL, 0ULL};
    const uint8_t kpRepeats = static_cast<uint8_t>((t.batch * 8u + 63u) / 64u);
    Duplicate<float>(ub[1536], centerX, even, kpRepeats, 1, 8);
    Duplicate<float>(ub[1536], centerY, odd, kpRepeats, 1, 8);
    const uint8_t scoreRepeats = static_cast<uint8_t>((t.batch * 4u + 63u) / 64u);
    uint64_t channelMask[2] = {0x1111111111111111ULL, 0ULL};
    for (uint32_t k = 0u; k < 4u; ++k) {
        Duplicate<float>(ub[2048], ub.GetValue(kWeightFloats + k), channelMask, scoreRepeats, 1, 8);
        channelMask[0] <<= 1;
    }
    SyncFence<HardEvent::V_MTE3>();
    DataCopy(keypoints, ub[1536], t.batch * 8u);
    DataCopyPad(scores, ub[2048], {1, t.batch * 16u, 0, 0, 0});
}

__aicore__ __attribute__((noinline)) bool PrepareWeightsAndTryUniform(
    const KernelAddrs &gm, const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
{
    if (t.stripes == 0u) {
        return false;
    }
    GlobalTensor<float> weights;
    GlobalTensor<float> bias;
    GlobalTensor<float> keypoints;
    GlobalTensor<float> scores;
    weights.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm.weight));
    bias.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm.bias));
    keypoints.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm.keypoints));
    scores.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm.scores));
    LocalTensor<float> ub(TPosition::VECCALC, 0, kUbTotalFloats);
    DataCopy(ub, weights, kWeightFloats);
    DataCopyPad(ub[kWeightFloats], bias, {1, 16, 0, 0, 0}, {false, 0, 0, 0.0f});
    const uint32_t maxTileH = (t.h + t.stripes - 1u) / t.stripes;
    if (maxTileH * t.w <= 256u) {
        return false;
    }
    Duplicate<uint32_t>(ub[3072].ReinterpretCast<uint32_t>(), 0x7fffffffu, 64);
    SyncFence<HardEvent::MTE2_V>();
    if (!IsAllFilterBitsZero(ub)) {
        return false;
    }
    if (GetBlockIdx() == 0u) {
        WriteUniformCenterOutputs(ub, keypoints, scores, t);
    }
    return true;
}

__aicore__ __attribute__((always_inline)) inline void ExtractBatchScores(
    __ubuf__ float *pSc, const __ubuf__ float *pSrc, uint32_t batch, uint32_t elemStride)
{
    for (uint32_t bIdx = 0u; bIdx < batch; ++bIdx) {
        const uint32_t base = bIdx * elemStride;
        for (uint32_t k = 0u; k < 4u; ++k) {
            pSc[bIdx * 4u + k] = pSrc[base + k];
        }
    }
}

__aicore__ __attribute__((always_inline)) inline void ReduceBatchScoreTree(
    const LocalTensor<float> &ub, const GlobalTensor<float> &wsGm,
    uint32_t batch, uint32_t reduceStripes, int32_t p2Elems)
{
    LocalTensor<float> allPkts = ub[0];
    LocalTensor<float> sZero = ub[1664];
    LocalTensor<float> sWork = ub[12288];
    const uint32_t p2ElemsU = static_cast<uint32_t>(p2Elems);
    const uint32_t totalCores = batch * reduceStripes;
    Duplicate<float>(sZero, 0.0f, p2Elems);
    Duplicate<float>(sWork, kNegMaxSentinel, static_cast<int32_t>(batch * p2ElemsU));
    PipeBarrier<PIPE_V>();
    SyncFence<HardEvent::V_MTE2>();
    DataCopy(allPkts, wsGm, totalCores * kPacketFloats);
    SyncFence<HardEvent::MTE2_V>();
    const int32_t sCount = static_cast<int32_t>(reduceStripes * kPacketFloats);
    #pragma clang loop unroll(disable)
    for (uint32_t bIdx = 0u; bIdx < batch; ++bIdx) {
        Add<float>(sWork[bIdx * p2ElemsU], allPkts[bIdx * reduceStripes * kPacketFloats], sZero, sCount);
    }
    PipeBarrier<PIPE_V>();
    #pragma clang loop unroll(disable)
    for (int32_t half = p2Elems >> 1; half >= 24; half >>= 1) {
        const uint32_t halfU = static_cast<uint32_t>(half);
        #pragma clang loop unroll(disable)
        for (uint32_t bIdx = 0u; bIdx < batch; ++bIdx) {
            const uint32_t bOff = bIdx * p2ElemsU;
            Max<float>(sWork[bOff], sWork[bOff], sWork[bOff + halfU], half);
        }
        PipeBarrier<PIPE_V>();
    }
    SyncFence<HardEvent::V_S>();
}

__aicore__ __attribute__((noinline)) void RunBatchScoreEpilogue(
    GM_ADDR scoreAddr, GM_ADDR wsAddr, uint32_t batch, uint32_t reduceStripes, int32_t p2Elems)
{
    GlobalTensor<float> scoreGm;
    GlobalTensor<float> wsGm;
    scoreGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scoreAddr));
    wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wsAddr));
    LocalTensor<float> ub(TPosition::VECCALC, 0, kUbTotalFloats);
    LocalTensor<float> scOutUb = ub[3840];
    __ubuf__ float *pSc = reinterpret_cast<__ubuf__ float *>(scOutUb.GetPhyAddr());

    if (reduceStripes == 1u) {
        LocalTensor<float> allPkts = ub[0];
        DataCopy(allPkts, wsGm, batch * kPacketFloats);
        SyncFence<HardEvent::MTE2_S>();
        ExtractBatchScores(pSc, reinterpret_cast<const __ubuf__ float *>(allPkts.GetPhyAddr()), batch, kPacketFloats);
    } else {
        ReduceBatchScoreTree(ub, wsGm, batch, reduceStripes, p2Elems);
        ExtractBatchScores(pSc, reinterpret_cast<const __ubuf__ float *>(ub[12288].GetPhyAddr()),
                           batch, static_cast<uint32_t>(p2Elems));
    }
    SyncFence<HardEvent::S_MTE3>();
    DataCopyPad(scoreGm, scOutUb, {1, batch * 16u, 0, 0, 0});
}

template <uint32_t KC>
class CooperativeSoftArgmax {
public:
    __aicore__ inline void Run(const KernelAddrs &gm, const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        if (t.stripes == 0u || t.w == 0u || !(t.tau > 0.0f)) {
            return;
        }
        if constexpr (KC == 4u) {
            if (!PrepareWeightsAndTryUniform(gm, t)) {
                RunDense(gm, t);
            }
        } else {
            PreloadSplitGroupWeights(gm);
            RunDense(gm, t);
        }
    }

private:
    uint32_t b_ = 0u;
    uint32_t s_ = 0u;
    uint32_t keyBase_ = 0u;
    uint32_t reduceStripes_ = 0u;
    uint32_t v0_ = 0u;
    uint32_t hTile_ = 0u;
    uint32_t hIn_ = 0u;
    uint32_t topPad_ = 0u;
    uint32_t yStart_ = 0u;
    uint32_t validH_ = 0u;
    uint32_t outElems_ = 0u;
    uint32_t outRepeats_ = 0u;
    uint32_t maxReadSpan_ = 0u;
    uint32_t chanStride_ = 0u;
    uint32_t groupStride_ = 0u;
    uint32_t outStride_ = 0u;
    uint32_t slot0Off_ = 0u;
    uint32_t slot1Off_ = 0u;
    uint32_t outR0Off_ = 0u;
    uint32_t outR1Off_ = 0u;
    uint32_t outR2Off_ = 0u;

    __aicore__ __attribute__((always_inline)) inline void PreloadSplitGroupWeights(const KernelAddrs &gm)
    {
        GlobalTensor<float> weights;
        GlobalTensor<float> bias;
        weights.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm.weight));
        bias.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm.bias));
        LocalTensor<float> ub(TPosition::VECCALC, 0, kUbTotalFloats);
        const uint32_t weightOffset = (GetBlockIdx() & 1u) * kHalfWeightFloats;
        DataCopy(ub[weightOffset], weights[weightOffset], kHalfWeightFloats);
        DataCopyPad(ub[kWeightFloats], bias, {1, 16u, 0, 0, 0}, {false, 0, 0, 0.0f});
    }

    __aicore__ __attribute__((always_inline)) inline void InitGeometry(
        uint32_t coreIdx, const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        if constexpr (KC == 4u) {
            b_ = coreIdx / t.stripes;
            s_ = coreIdx % t.stripes;
            keyBase_ = 0u;
            reduceStripes_ = t.stripes;
        } else {
            b_ = coreIdx / (2u * t.stripes);
            keyBase_ = (coreIdx & 1u) * 2u;
            s_ = (coreIdx / 2u) % t.stripes;
            reduceStripes_ = 2u * t.stripes;
        }
        const uint32_t baseH = t.h / t.stripes;
        const uint32_t remH = t.h % t.stripes;
        v0_ = s_ * baseH + (s_ < remH ? s_ : remH);
        hTile_ = baseH + (s_ < remH ? 1u : 0u);
        hIn_ = hTile_ + 2u;
        topPad_ = (v0_ == 0u) ? 1u : 0u;
        yStart_ = (v0_ == 0u) ? 0u : (v0_ - 1u);
        const uint32_t yEnd = (v0_ + hTile_ < t.h) ? (v0_ + hTile_ + 1u) : t.h;
        validH_ = yEnd - yStart_;
        outElems_ = hTile_ * t.w;
        outRepeats_ = (outElems_ + 63u) >> 6;
        maxReadSpan_ = 2u * t.w + (outRepeats_ << 6);
    }

    __aicore__ __attribute__((always_inline)) inline void InitUbLayout(uint32_t w)
    {
        chanStride_ = (maxReadSpan_ + 127u) & ~127u;
        if ((chanStride_ % 384u) == 0u) {
            chanStride_ += 128u;
        }
        groupStride_ = 4u * chanStride_;
        outStride_ = ((outRepeats_ << 6) + 127u) & ~127u;
        if (outRepeats_ == 2u) {
            outStride_ = 256u;
        } else {
            while ((outStride_ % 384u) != 128u) {
                outStride_ += 128u;
            }
        }
        slot0Off_ = kSlot0BaseOff;
        slot1Off_ = ((slot0Off_ + 3u * groupStride_ + 383u) / 384u) * 384u;
        const uint32_t accArea0 = ((slot1Off_ + 3u * groupStride_ + 383u) / 384u) * 384u;
        const uint32_t dst0Bg = 8u;
        const uint32_t dst1Bg = (((w >> 3) + 8u) & 15u);
        const uint32_t dst2Bg = ((((2u * w) >> 3) + 8u) & 15u);
        outR0Off_ = accArea0 + dst0Bg * 8u;
        const uint32_t accArea1 = ((outR0Off_ + 4u * outStride_ + 383u) / 384u) * 384u;
        outR1Off_ = accArea1 + dst1Bg * 8u;
        const uint32_t accArea2 = ((outR1Off_ + 4u * outStride_ + 383u) / 384u) * 384u;
        outR2Off_ = accArea2 + dst2Bg * 8u;
    }

    __aicore__ __attribute__((always_inline)) inline void ClearPaddingHalos(__ubuf__ float *baseUbPtr, uint32_t w)
    {
        ResetMask();
        const uint32_t writtenEnd = (topPad_ + validH_) * w;
        const uint8_t cBlkStride = static_cast<uint8_t>(chanStride_ >> 3);
        if (topPad_ != 0u) {
            RawDup0(baseUbPtr + slot0Off_, 12u, cBlkStride);
            RawDup0(baseUbPtr + slot1Off_, 12u, cBlkStride);
            if (w > 64u) {
                RawDup0(baseUbPtr + slot0Off_ + 64u, 12u, cBlkStride);
                RawDup0(baseUbPtr + slot1Off_ + 64u, 12u, cBlkStride);
            }
        }
        if (maxReadSpan_ > writtenEnd) {
            const uint32_t tailLen = maxReadSpan_ - writtenEnd;
            for (uint32_t off = 0u; off < tailLen; off += 64u) {
                RawDup0(baseUbPtr + slot0Off_ + writtenEnd + off, 12u, cBlkStride);
                RawDup0(baseUbPtr + slot1Off_ + writtenEnd + off, 12u, cBlkStride);
            }
        }
    }

    __aicore__ __attribute__((always_inline)) inline void InitPrologueAndBias(
        __ubuf__ float *baseUbPtr, const LocalTensor<float> &bUb)
    {
        #pragma unroll
        for (int32_t i = 0; i < 8; ++i) {
            baseUbPtr[kCoordSeedOff + static_cast<uint32_t>(i)] = static_cast<float>(i);
        }
        const uint8_t s1Rep = static_cast<uint8_t>(outStride_ >> 6);
        const uint8_t sAllRep = static_cast<uint8_t>((KC * outStride_) >> 6);
        RawDupVal64(baseUbPtr + outR1Off_, 0.0f, sAllRep);
        RawDupVal64(baseUbPtr + outR2Off_, 0.0f, sAllRep);
        SyncFence<HardEvent::MTE2_S>();
        const __ubuf__ float *bPtr = reinterpret_cast<const __ubuf__ float *>(bUb.GetPhyAddr());
        float biasVals[KC];
        #pragma unroll
        for (uint32_t k = 0u; k < KC; ++k) {
            biasVals[k] = bPtr[keyBase_ + k];
        }
        SyncFence<HardEvent::S_V>();
        #pragma unroll
        for (uint32_t k = 0u; k < KC; ++k) {
            RawDupVal64(baseUbPtr + outR0Off_ + k * outStride_, biasVals[k], s1Rep);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ __attribute__((always_inline)) inline void MaskSlotHorizontalBorders(
        __ubuf__ float *slotPtr, uint32_t w)
    {
        const uint8_t hInRep = static_cast<uint8_t>(hIn_);
        const uint8_t wBlk8 = static_cast<uint8_t>(w >> 3);
        const uint32_t rLastOff = w - 8u;
        SetVectorMask<float, MaskMode::NORMAL>(0, 1ULL);
        #pragma unroll
        for (uint32_t cSub = 0u; cSub < 4u; ++cSub) {
            RawDup0(slotPtr + cSub * chanStride_, hInRep, wBlk8);
        }
        SetVectorMask<float, MaskMode::NORMAL>(0, 0x80ULL);
        #pragma unroll
        for (uint32_t cSub = 0u; cSub < 4u; ++cSub) {
            RawDup0(slotPtr + 2u * groupStride_ + cSub * chanStride_ + rLastOff, hInRep, wBlk8);
        }
        ResetMask();
        PipeBarrier<PIPE_V>();
    }

    __aicore__ __attribute__((always_inline)) inline void AccumulateChannelGroup(
        __ubuf__ float *baseUbPtr, __ubuf__ float *slotPtr,
        const __ubuf__ float *wPtr, uint32_t grp, uint32_t w)
    {
        __ubuf__ float *pR0[KC];
        __ubuf__ float *pR1[KC];
        __ubuf__ float *pR2[KC];
        #pragma unroll
        for (uint32_t k = 0u; k < KC; ++k) {
            pR0[k] = baseUbPtr + outR0Off_ + k * outStride_;
            pR1[k] = baseUbPtr + outR1Off_ + k * outStride_;
            pR2[k] = baseUbPtr + outR2Off_ + k * outStride_;
        }
        const uint8_t oRep = static_cast<uint8_t>(outRepeats_);
        const uint32_t w2 = 2u * w;
        #pragma clang loop unroll(disable)
        for (uint32_t cSub = 0u; cSub < 4u; ++cSub) {
            const uint32_t c = grp * 4u + cSub;
            const uint32_t cOff = cSub * chanStride_;
            const __ubuf__ float *wc = wPtr + c * 9u;
            #pragma unroll
            for (uint32_t dc = 0u; dc < 3u; ++dc) {
                __ubuf__ float *pS = slotPtr + dc * groupStride_ + cOff;
                const __ubuf__ float *wTap = wc + dc;
                RawAxpyGroup<KC>(pR0, pS, wTap, oRep);
                RawAxpyGroup<KC>(pR1, pS + w, wTap + 3u, oRep);
                RawAxpyGroup<KC>(pR2, pS + w2, wTap + 6u, oRep);
            }
        }
    }

    __aicore__ __attribute__((always_inline)) inline void MergeRowPhaseAccumulators(__ubuf__ float *baseUbPtr)
    {
        const uint8_t oRep = static_cast<uint8_t>(outRepeats_);
        #pragma unroll
        for (uint32_t k = 0u; k < KC; ++k) {
            __ubuf__ float *dst = baseUbPtr + outR0Off_ + k * outStride_;
            RawAdd64(dst, dst, baseUbPtr + outR1Off_ + k * outStride_, oRep);
        }
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t k = 0u; k < KC; ++k) {
            __ubuf__ float *dst = baseUbPtr + outR0Off_ + k * outStride_;
            RawAdd64(dst, dst, baseUbPtr + outR2Off_ + k * outStride_, oRep);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ __attribute__((always_inline)) inline void RunConvPipeline(
        const LocalTensor<float> &ub, const GlobalTensor<float> &featGm,
        __ubuf__ float *baseUbPtr, const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        const uint32_t wBlocks = t.w >> 3;
        SlotLoadPlan loadPlan{};
        loadPlan.groupStride = groupStride_;
        loadPlan.topOffset = topPad_ * t.w;
        loadPlan.padBlockLenBytes = (validH_ * t.w - 1u) * 4u;
        loadPlan.padSrcStrideBytes = ((t.h - validH_) * t.w + 1u) * 4u;
        loadPlan.copyBlkLen = static_cast<uint16_t>(validH_ * wBlocks);
        loadPlan.copySrcStride = static_cast<uint16_t>((t.h - validH_) * wBlocks);
        loadPlan.copyDstStride = static_cast<uint16_t>((chanStride_ - validH_ * t.w) >> 3);
        const uint32_t hw = t.h * t.w;
        const uint32_t bFeatBase = b_ * 32u * hw + yStart_ * t.w;

        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0);
        SetFlag<HardEvent::V_MTE2>(1);
        WaitFlag<HardEvent::V_MTE2>(0);
        IssueSlotLoad(ub, featGm, slot0Off_, bFeatBase, loadPlan);
        SetFlag<HardEvent::MTE2_V>(0);

        InitPrologueAndBias(baseUbPtr, ub[kWeightFloats]);
        const __ubuf__ float *wPtr =
            reinterpret_cast<const __ubuf__ float *>(ub[0].GetPhyAddr()) + keyBase_ * kFilterFloats;
        ResetMask();
        #pragma clang loop unroll(disable)
        for (uint32_t grp = 0u; grp < 8u; ++grp) {
            const uint32_t slot = grp & 1u;
            const uint32_t nextSlot = slot ^ 1u;
            if (grp + 1u < 8u) {
                WaitFlag<HardEvent::V_MTE2>(nextSlot);
                IssueSlotLoad(ub, featGm, (nextSlot == 0u) ? slot0Off_ : slot1Off_,
                              bFeatBase + (grp + 1u) * 4u * hw, loadPlan);
                SetFlag<HardEvent::MTE2_V>(nextSlot);
            }
            WaitFlag<HardEvent::MTE2_V>(slot);
            __ubuf__ float *slotPtr = baseUbPtr + ((slot == 0u) ? slot0Off_ : slot1Off_);
            MaskSlotHorizontalBorders(slotPtr, t.w);
            AccumulateChannelGroup(baseUbPtr, slotPtr, wPtr, grp, t.w);
            SetFlag<HardEvent::V_MTE2>(slot);
        }
        WaitFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(1);
        PipeBarrier<PIPE_V>();
        MergeRowPhaseAccumulators(baseUbPtr);
    }

    __aicore__ __attribute__((always_inline)) inline void ComputeStripeMaxAndExp(
        const LocalTensor<float> &ub, const FusedDisconnectorKeypointSoftArgmaxTilingData &t,
        float (&rawMax)[KC])
    {
        LocalTensor<float> outR0 = ub[outR0Off_];
        LocalTensor<float> scaledZ = ub[slot0Off_];
        LocalTensor<float> scratch = ub[slot0Off_ + kScratchRelOff];
        LocalTensor<float> uCoord = ub[kCoordSeedOff];
        LocalTensor<float> redRes = scratch[1280];
        LocalTensor<float> redTmp = scratch[1536];

        Duplicate<float>(scratch[256], 0.0f, static_cast<int32_t>(KC * 256u));
        #pragma clang loop unroll(disable)
        for (uint32_t k = 0u; k < KC; ++k) {
            Stage1Max(redRes[k * 8u], redTmp[k * 64u], outR0[k * outStride_], outElems_);
        }
        const float invTau = 1.0f / t.tau;
        const int32_t totalZ = static_cast<int32_t>(KC * outStride_);
        Muls<float>(scaledZ, outR0, invTau, totalZ);
        PipeBarrier<PIPE_V>();
        for (uint32_t span = 8u; span < t.w; span <<= 1) {
            const uint32_t ucnt = (t.w - span < span) ? (t.w - span) : span;
            Adds<float>(uCoord[span], uCoord, static_cast<float>(static_cast<int32_t>(span)),
                        static_cast<int32_t>(ucnt));
            PipeBarrier<PIPE_V>();
        }
        #pragma clang loop unroll(disable)
        for (uint32_t k = 0u; k < KC; ++k) {
            Stage2Max(redRes[k * 8u], redTmp[k * 64u], outElems_);
        }
        PipeBarrier<PIPE_V>();
        SyncFence<HardEvent::V_S>();

        const __ubuf__ float *pRed = reinterpret_cast<const __ubuf__ float *>(redRes.GetPhyAddr());
        const int32_t outElemsI32 = static_cast<int32_t>(outElems_);
        #pragma unroll
        for (uint32_t k = 0u; k < KC; ++k) {
            rawMax[k] = pRed[k * 8u];
            Adds<float>(scaledZ[k * outStride_], scaledZ[k * outStride_], -(rawMax[k] * invTau), outElemsI32);
        }
        PipeBarrier<PIPE_V>();
        Exp<float>(scaledZ, scaledZ, totalZ);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ __attribute__((always_inline)) inline void ComputeRowAndColProjections(
        const LocalTensor<float> &ub, __ubuf__ float *baseUbPtr, uint32_t w)
    {
        LocalTensor<float> scaledZ = ub[slot0Off_];
        LocalTensor<float> scratch = ub[slot0Off_ + kScratchRelOff];
        LocalTensor<float> rowSums = scratch[2048];
        LocalTensor<float> rowSums2 = scratch[2304];
        const int32_t rowStrideBlk = static_cast<int32_t>(w >> 3);
        const int32_t hTileI32 = static_cast<int32_t>(hTile_);
        if (w <= 64u) {
            #pragma clang loop unroll(disable)
            for (uint32_t k = 0u; k < KC; ++k) {
                WholeReduceSum(rowSums[k * 64u], scaledZ[k * outStride_],
                               static_cast<int32_t>(w), hTileI32, 1, 1, rowStrideBlk);
            }
        } else {
            const int32_t remW = static_cast<int32_t>(w - 64u);
            #pragma clang loop unroll(disable)
            for (uint32_t k = 0u; k < KC; ++k) {
                WholeReduceSum(rowSums[k * 64u], scaledZ[k * outStride_], 64, hTileI32, 1, 1, rowStrideBlk);
                WholeReduceSum(rowSums2[k * 64u], scaledZ[k * outStride_ + 64u], remW, hTileI32, 1, 1, rowStrideBlk);
            }
            PipeBarrier<PIPE_V>();
            Add<float>(rowSums, rowSums, rowSums2, static_cast<int32_t>(KC * 64u));
        }
        ResetMask();
        __ubuf__ float *pColBase = reinterpret_cast<__ubuf__ float *>(scratch[256].GetPhyAddr());
        const uint8_t wRep = static_cast<uint8_t>((w + 63u) >> 6);
        #pragma clang loop unroll(disable)
        for (uint32_t r = 0u; r < hTile_; ++r) {
            const uint32_t rOff = r * w;
            #pragma unroll
            for (uint32_t k = 0u; k < KC; ++k) {
                RawAdd64(pColBase + k * 256u, pColBase + k * 256u,
                         baseUbPtr + slot0Off_ + k * outStride_ + rOff, wRep);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ __attribute__((always_inline)) inline void ComputeStripeMoments(
        const LocalTensor<float> &ub, uint32_t w, float (&evOut)[KC])
    {
        LocalTensor<float> scratch = ub[slot0Off_ + kScratchRelOff];
        LocalTensor<float> uCoord = ub[kCoordSeedOff];
        LocalTensor<float> colSum0 = scratch[256];
        LocalTensor<float> redRes = scratch[1280];
        LocalTensor<float> redTmp = scratch[1536];
        LocalTensor<float> rowSums = scratch[2048];
        const int32_t wI32 = static_cast<int32_t>(w);

        #pragma clang loop unroll(disable)
        for (uint32_t k = 0u; k < KC; ++k) {
            Stage1Sum(redRes[64u + k * 8u], redTmp[k * 64u], colSum0[k * 256u], w);
            Mul<float>(colSum0[k * 256u], colSum0[k * 256u], uCoord, wI32);
        }
        PipeBarrier<PIPE_V>();
        #pragma clang loop unroll(disable)
        for (uint32_t k = 0u; k < KC; ++k) {
            Stage2Sum(redRes[64u + k * 8u], redTmp[k * 64u], w);
            Stage1Sum(redRes[96u + k * 8u], redTmp[256u + k * 64u], colSum0[k * 256u], w);
        }
        PipeBarrier<PIPE_V>();
        #pragma clang loop unroll(disable)
        for (uint32_t k = 0u; k < KC; ++k) {
            Stage2Sum(redRes[96u + k * 8u], redTmp[256u + k * 64u], w);
            evOut[k] = 0.0f;
        }
        PipeBarrier<PIPE_V>();
        SyncFence<HardEvent::V_S>();

        const __ubuf__ float *pRowS = reinterpret_cast<const __ubuf__ float *>(rowSums.GetPhyAddr());
        for (uint32_t r = 0u; r < hTile_; ++r) {
            const float vf = static_cast<float>(static_cast<int32_t>(v0_ + r));
            #pragma unroll
            for (uint32_t k = 0u; k < KC; ++k) {
                evOut[k] += pRowS[k * 64u + r] * vf;
            }
        }
    }

    __aicore__ __attribute__((always_inline)) inline void PublishStripePacket(
        const LocalTensor<float> &ub, const GlobalTensor<float> &wsGm,
        uint32_t coreIdx, const float (&rawMax)[KC], const float (&ev)[KC])
    {
        LocalTensor<float> scratch = ub[slot0Off_ + kScratchRelOff];
        LocalTensor<float> pktOut = scratch[2560];
        const __ubuf__ float *pRed = reinterpret_cast<const __ubuf__ float *>(scratch[1280].GetPhyAddr());
        __ubuf__ float *pPkt = reinterpret_cast<__ubuf__ float *>(pktOut.GetPhyAddr());
        if constexpr (KC == 2u) {
            #pragma unroll
            for (uint32_t i = 0u; i < 8u; ++i) {
                pPkt[i] = kNegMaxSentinel;
            }
            #pragma unroll
            for (uint32_t i = 8u; i < kPacketFloats; ++i) {
                pPkt[i] = 0.0f;
            }
        } else {
            #pragma unroll
            for (uint32_t i = 20u; i < kPacketFloats; ++i) {
                pPkt[i] = 0.0f;
            }
        }
        #pragma unroll
        for (uint32_t k = 0u; k < KC; ++k) {
            pPkt[keyBase_ + k] = rawMax[k];
            pPkt[4u + keyBase_ + k] = rawMax[k];
            pPkt[8u + keyBase_ + k] = pRed[64u + k * 8u];
            pPkt[12u + keyBase_ + k] = pRed[96u + k * 8u];
            pPkt[16u + keyBase_ + k] = ev[k];
        }
        SyncFence<HardEvent::S_MTE3>();
        DataCopy(wsGm[coreIdx * kPacketFloats], pktOut, kPacketFloats);
        SyncAll<true>();
    }

    __aicore__ __attribute__((always_inline)) inline void WriteSingleStripeBatch(
        const LocalTensor<float> &ub, const GlobalTensor<float> &kpGm,
        const GlobalTensor<float> &scoreGm, const FusedDisconnectorKeypointSoftArgmaxTilingData &t,
        const float (&ev)[KC])
    {
        LocalTensor<float> scratch = ub[slot0Off_ + kScratchRelOff];
        LocalTensor<float> kpOutUb = scratch[512];
        const __ubuf__ float *pRed = reinterpret_cast<const __ubuf__ float *>(scratch[1280].GetPhyAddr());
        __ubuf__ float *pKp = reinterpret_cast<__ubuf__ float *>(kpOutUb.GetPhyAddr());
        const float strideF = static_cast<float>(t.stride);
        #pragma unroll
        for (uint32_t k = 0u; k < 4u; ++k) {
            const float denom = pRed[64u + k * 8u];
            const float invS = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
            pKp[k * 2u + 0u] = (pRed[96u + k * 8u] * invS + 0.5f) * strideF;
            pKp[k * 2u + 1u] = (ev[k < KC ? k : 0u] * invS + 0.5f) * strideF;
        }
        SyncFence<HardEvent::S_MTE3>();
        DataCopy(kpGm[GetBlockIdx() * 8u], kpOutUb, 8);
        if (t.batch == 1u) {
            DataCopyPad(scoreGm, scratch[2560], {1, 16u, 0, 0, 0});
        }
    }

    __aicore__ __attribute__((always_inline)) inline void RescaleBatchPackets(
        const LocalTensor<float> &ub, uint8_t p2Rep, int32_t p2Elems, float invTau)
    {
        LocalTensor<float> bPkts = ub[slot0Off_];
        LocalTensor<float> maxTree = ub[slot0Off_ + kMaxTreeRelOff];
        LocalTensor<float> alphaTree = ub[slot0Off_ + kAlphaTreeRelOff];
        LocalTensor<float> scaledMax = ub[slot0Off_ + kScratchRelOff];

        const int32_t firstHalf = p2Elems >> 1;
        Max<float>(maxTree, bPkts, bPkts[firstHalf], firstHalf);
        PipeBarrier<PIPE_V>();
        #pragma clang loop unroll(disable)
        for (int32_t half = firstHalf >> 1; half >= 24; half >>= 1) {
            Max<float>(maxTree, maxTree, maxTree[half], half);
            PipeBarrier<PIPE_V>();
        }
        Muls<float>(scaledMax, maxTree, invTau, 8);
        Muls<float>(alphaTree, bPkts, invTau, 8ULL, p2Rep, {1, 1, 3, 3});
        PipeBarrier<PIPE_V>();
        Sub<float>(alphaTree, alphaTree, scaledMax, 8ULL, p2Rep, {1, 1, 1, 3, 3, 0});
        PipeBarrier<PIPE_V>();
        Exp<float>(alphaTree, alphaTree, p2Elems);
        PipeBarrier<PIPE_V>();
        Mul<float>(bPkts[8], bPkts[8], alphaTree, 8ULL, p2Rep, {1, 1, 1, 3, 3, 3});
        Mul<float>(bPkts[16], bPkts[16], alphaTree, 8ULL, p2Rep, {1, 1, 1, 3, 3, 3});
        PipeBarrier<PIPE_V>();
        #pragma clang loop unroll(disable)
        for (int32_t half = firstHalf; half >= 24; half >>= 1) {
            Add<float>(bPkts, bPkts, bPkts[half], half);
            PipeBarrier<PIPE_V>();
        }
        SyncFence<HardEvent::V_S>();
    }

    __aicore__ __attribute__((always_inline)) inline void ReduceMultiStripeBatch(
        const LocalTensor<float> &ub, const GlobalTensor<float> &wsGm,
        const GlobalTensor<float> &kpGm, const GlobalTensor<float> &scoreGm,
        const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        uint32_t p2S = 2u;
        while (p2S < reduceStripes_) {
            p2S <<= 1;
        }
        const int32_t p2Elems = static_cast<int32_t>(p2S * kPacketFloats);
        LocalTensor<float> bPkts = ub[slot0Off_];
        LocalTensor<float> alphaTree = ub[slot0Off_ + kAlphaTreeRelOff];
        LocalTensor<float> kpOutUb = ub[slot0Off_ + kScratchRelOff + 512u];
        const uint32_t bi = GetBlockIdx();

        Duplicate<float>(bPkts, 0.0f, p2Elems);
        Duplicate<float>(alphaTree, 0.0f, p2Elems);
        PipeBarrier<PIPE_V>();
        if (reduceStripes_ < p2S) {
            const uint8_t remS = static_cast<uint8_t>(p2S - reduceStripes_);
            Duplicate<float>(bPkts[reduceStripes_ * kPacketFloats], kNegMaxSentinel, 8ULL, remS, 1, 3);
            PipeBarrier<PIPE_V>();
        }
        SyncFence<HardEvent::V_MTE2>();
        DataCopy(bPkts, wsGm[bi * reduceStripes_ * kPacketFloats], reduceStripes_ * kPacketFloats);
        SyncFence<HardEvent::MTE2_V>();

        RescaleBatchPackets(ub, static_cast<uint8_t>(p2S), p2Elems, 1.0f / t.tau);
        const __ubuf__ float *pB = reinterpret_cast<const __ubuf__ float *>(bPkts.GetPhyAddr());
        __ubuf__ float *pKp = reinterpret_cast<__ubuf__ float *>(kpOutUb.GetPhyAddr());
        const float strideF = static_cast<float>(t.stride);
        #pragma unroll
        for (uint32_t k = 0u; k < 4u; ++k) {
            const float denom = pB[8u + k];
            const float invS = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
            pKp[k * 2u + 0u] = (pB[12u + k] * invS + 0.5f) * strideF;
            pKp[k * 2u + 1u] = (pB[16u + k] * invS + 0.5f) * strideF;
        }
        SyncFence<HardEvent::S_MTE3>();
        DataCopy(kpGm[bi * 8u], kpOutUb, 8);
        if (t.batch == 1u) {
            DataCopyPad(scoreGm, ub[slot0Off_ + kMaxTreeRelOff], {1, 16u, 0, 0, 0});
        }
    }

    __aicore__ __attribute__((always_inline)) inline void FinalizeBatchOutputs(
        const LocalTensor<float> &ub, const KernelAddrs &gm,
        const GlobalTensor<float> &wsGm, const FusedDisconnectorKeypointSoftArgmaxTilingData &t,
        const float (&ev)[KC])
    {
        const uint32_t coreIdx = GetBlockIdx();
        GlobalTensor<float> kpGm;
        GlobalTensor<float> scoreGm;
        kpGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm.keypoints));
        scoreGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm.scores));
        if (coreIdx < t.batch) {
            if (reduceStripes_ == 1u) {
                WriteSingleStripeBatch(ub, kpGm, scoreGm, t, ev);
            } else {
                ReduceMultiStripeBatch(ub, wsGm, kpGm, scoreGm, t);
            }
        }
        if (t.batch > 1u) {
            uint32_t p2S = 2u;
            while (p2S < reduceStripes_) {
                p2S <<= 1;
            }
            const uint32_t scoreCore = (t.batch < t.cores) ? t.batch : 0u;
            if (coreIdx == scoreCore) {
                RunBatchScoreEpilogue(gm.scores, gm.ws, t.batch, reduceStripes_,
                                      static_cast<int32_t>(p2S * kPacketFloats));
            }
        }
    }

    __aicore__ __attribute__((noinline)) void RunDense(
        const KernelAddrs &gm, const FusedDisconnectorKeypointSoftArgmaxTilingData &t)
    {
        ICachePreLoad(4);
        const uint32_t coreIdx = GetBlockIdx();
        if (coreIdx >= t.cores) {
            return;
        }
        InitGeometry(coreIdx, t);
        InitUbLayout(t.w);

        GlobalTensor<float> featGm;
        GlobalTensor<float> wsGm;
        featGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm.feat));
        wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm.ws));

        LocalTensor<float> ub(TPosition::VECCALC, 0, kUbTotalFloats);
        __ubuf__ float *baseUbPtr = reinterpret_cast<__ubuf__ float *>(ub.GetPhyAddr());
        ClearPaddingHalos(baseUbPtr, t.w);
        RunConvPipeline(ub, featGm, baseUbPtr, t);

        float rawMax[KC];
        float ev[KC];
        ComputeStripeMaxAndExp(ub, t, rawMax);
        ComputeRowAndColProjections(ub, baseUbPtr, t.w);
        ComputeStripeMoments(ub, t.w, ev);
        PublishStripePacket(ub, wsGm, coreIdx, rawMax, ev);
        FinalizeBatchOutputs(ub, gm, wsGm, t, ev);
    }
};

__aicore__ __attribute__((always_inline)) inline bool HasValidKernelAddrs(
    const KernelAddrs &addrs, GM_ADDR tiling)
{
    const GM_ADDR ptrs[7] = {
        addrs.feat, addrs.weight, addrs.bias, addrs.keypoints, addrs.scores, addrs.ws, tiling
    };
    #pragma unroll
    for (int32_t i = 0; i < 7; ++i) {
        if (ptrs[i] == nullptr) {
            return false;
        }
    }
    return true;
}

}  // namespace

template <uint32_t MODE>
__global__ __aicore__ void fused_disconnector_keypoint_soft_argmax(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR keypoints, GM_ADDR scores,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    const KernelAddrs addrs{feat, weight, bias, keypoints, scores, ::GetSysWorkSpacePtr()};
    if (!HasValidKernelAddrs(addrs, tiling)) {
        return;
    }
    REGISTER_TILING_DEFAULT(FusedDisconnectorKeypointSoftArgmaxTilingData);
    COPY_TILING_WITH_STRUCT(FusedDisconnectorKeypointSoftArgmaxTilingData, tiling, tp);
    const FusedDisconnectorKeypointSoftArgmaxTilingData &tilingData = *tp;
    InitSocState();
    if constexpr (MODE == 2u) {
        CooperativeSoftArgmax<2u> op;
        op.Run(addrs, tilingData);
    } else if constexpr (MODE == 1u) {
        CooperativeSoftArgmax<4u> op;
        op.Run(addrs, tilingData);
    }
}
