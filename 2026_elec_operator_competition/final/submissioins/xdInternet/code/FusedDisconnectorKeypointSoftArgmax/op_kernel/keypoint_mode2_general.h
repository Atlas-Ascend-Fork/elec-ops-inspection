/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef KEYPOINT_MODE2_GENERAL_H
#define KEYPOINT_MODE2_GENERAL_H

#include "keypoint_common.h"

namespace keypoint_impl {

static constexpr uint32_t kMode2TileW = 1024U;
static constexpr uint32_t kMode2MaxWCache = 4096U;

static constexpr uint32_t kM2BiasOff = 1024U;
static constexpr uint32_t kM2InTileOff = 1536U;
static constexpr uint32_t kM2AccTileOff = 5888U;
static constexpr uint32_t kM2UTileOff = 10496U;
static constexpr uint32_t kM2WorkOff = 14592U;
static constexpr uint32_t kM2ScalarOff = 16640U;
static constexpr uint32_t kM2ExpTinyOff = 16896U;
static constexpr uint32_t kM2OutOff = 16960U;
static constexpr uint32_t kM2AccStatsOff = 17088U;
static constexpr uint32_t kM2TileStatsOff = 17152U;
static constexpr uint32_t kM2WCacheOff = 17408U;

class KeypointMode2Kernel {
public:
    __aicore__ inline void Process(
        GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores,
        GM_ADDR workspace, const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        (void)workspace;
        if (t.outChannels == 0U) {
            return;
        }
        uint32_t coreIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t stepCores = static_cast<uint32_t>(AscendC::GetBlockNum());
        if (stepCores == 0U) {
            return;
        }
        gm_.Bind(feat, weight, bias, keypoints, scores);
        wCacheStart_ = 0ULL;
        wCacheLen_ = 0U;
        uint32_t totalPlanes = ClampTotalPlanes(t.batch, t.outChannels);
        for (uint32_t p = coreIdx; p < totalPlanes; p += stepCores) {
            ProcessSinglePlane(p, t);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    KeypointGmTensors gm_;
    uint64_t wCacheStart_ = 0ULL;
    uint32_t wCacheLen_ = 0U;

    __aicore__ inline float ReadWeightCached(uint64_t kPlaneBase, uint64_t relIdx, uint64_t planeElems)
    {
        auto wUb = UbTensor<float>(kM2WCacheOff, kMode2MaxWCache);
        if (wCacheLen_ == 0U || relIdx < wCacheStart_ || relIdx >= wCacheStart_ + wCacheLen_) {
            uint64_t rem = (planeElems > relIdx) ? (planeElems - relIdx) : 0ULL;
            uint32_t loadLen = (rem > kMode2MaxWCache) ? kMode2MaxWCache : static_cast<uint32_t>(rem);
            if (loadLen == 0U) {
                return 0.0f;
            }
            AscendC::DataCopyPad(
                wUb, gm_.weight[kPlaneBase + relIdx],
                AscendC::DataCopyExtParams{1U, loadLen * 4U, 0U, 0U, 0U},
                AscendC::DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            wCacheStart_ = relIdx;
            wCacheLen_ = loadLen;
        }
        return wUb.GetValue(static_cast<uint32_t>(relIdx - wCacheStart_));
    }

    __aicore__ inline void FillTileCoords(uint32_t u0, uint32_t alignedLen)
    {
        auto uUb = UbTensor<float>(kM2UTileOff, alignedLen);
        for (uint32_t i = 0U; i < 8U; ++i) {
            uUb.SetValue(i, ScalarToFloat(u0 + i));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        ExpandDoublingFloatUb(uUb, 8U, alignedLen);
    }

    __aicore__ inline void AccumulateSingleTap(
        uint64_t featRowBase, uint32_t u0, uint32_t tileLen, uint32_t alignedLen,
        uint32_t kw, float wVal, const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        if (wVal == 0.0f) {
            return;
        }
        int32_t padW = static_cast<int32_t>((t.kernelW > 0U) ? ((t.kernelW - 1U) >> 1U) : 0U);
        int32_t inCol0 = static_cast<int32_t>(u0) + static_cast<int32_t>(kw) - padW;
        int32_t leftPadS = (inCol0 < 0) ? (0 - inCol0) : 0;
        int32_t srcCol = (inCol0 > 0) ? inCol0 : 0;
        int32_t inColEndRaw = inCol0 + static_cast<int32_t>(tileLen);
        int32_t wS = static_cast<int32_t>(t.width);
        int32_t inColEnd = (inColEndRaw < wS) ? inColEndRaw : wS;
        if (inColEnd <= srcCol || leftPadS >= static_cast<int32_t>(tileLen)) {
            return;
        }
        uint32_t leftPad = static_cast<uint32_t>(leftPadS);
        uint32_t validLen = static_cast<uint32_t>(inColEnd - srcCol);
        uint32_t baseBlkIdx = leftPad & ~7U;
        uint8_t subLeft = static_cast<uint8_t>(leftPad & 7U);
        uint8_t tailRpad = static_cast<uint8_t>((8U - ((static_cast<uint32_t>(subLeft) + validLen) & 7U)) & 7U);

        auto inUb = UbTensor<float>(kM2InTileOff, alignedLen);
        if (leftPad > 0U || validLen < alignedLen) {
            AscendC::Duplicate(inUb, 0.0f, alignedLen);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::DataCopyPad(
            inUb[baseBlkIdx], gm_.feat[featRowBase + static_cast<uint64_t>(srcCol)],
            AscendC::DataCopyExtParams{1U, validLen * 4U, 0U, 0U, 0U},
            AscendC::DataCopyPadExtParams<float>{true, subLeft, tailRpad, 0.0f});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Axpy(UbTensor<float>(kM2AccTileOff, alignedLen), inUb, wVal, alignedLen);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateRowTaps(
        uint64_t featRowBase, uint64_t kPlaneBase, uint64_t planeElems, uint64_t rowWBase,
        uint32_t u0, uint32_t tileLen, uint32_t alignedLen,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        for (uint32_t kw = 0U; kw < t.kernelW; ++kw) {
            float wVal = ReadWeightCached(kPlaneBase, rowWBase + kw, planeElems);
            AccumulateSingleTap(featRowBase, u0, tileLen, alignedLen, kw, wVal, t);
        }
    }

    __aicore__ inline void ComputeTileConv(
        uint32_t b, uint32_t k, uint32_t v, uint32_t u0, uint32_t tileLen, uint32_t alignedLen,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        auto accUb = UbTensor<float>(kM2AccTileOff, alignedLen);
        AscendC::Duplicate(accUb, 0.0f, alignedLen);
        AscendC::PipeBarrier<PIPE_V>();
        uint64_t planeElems = static_cast<uint64_t>(t.inChannels) * t.kernelH * t.kernelW;
        uint64_t kPlaneBase = static_cast<uint64_t>(k) * planeElems;
        uint64_t featPlaneSize = static_cast<uint64_t>(t.height) * t.width;
        uint64_t featBatchBase = static_cast<uint64_t>(b) * t.inChannels * featPlaneSize;
        int32_t padH = static_cast<int32_t>((t.kernelH > 0U) ? ((t.kernelH - 1U) >> 1U) : 0U);
        int32_t hS = static_cast<int32_t>(t.height);

        for (uint32_t c = 0U; c < t.inChannels; ++c) {
            uint64_t cFeatBase = featBatchBase + static_cast<uint64_t>(c) * featPlaneSize;
            for (uint32_t kh = 0U; kh < t.kernelH; ++kh) {
                int32_t inY = static_cast<int32_t>(v) + static_cast<int32_t>(kh) - padH;
                if (inY >= 0 && inY < hS) {
                    uint64_t featRowBase = cFeatBase + static_cast<uint64_t>(inY) * t.width;
                    uint64_t rowWBase = (static_cast<uint64_t>(c) * t.kernelH + kh) * t.kernelW;
                    AccumulateRowTaps(featRowBase, kPlaneBase, planeElems, rowWBase, u0, tileLen, alignedLen, t);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceTileSoftmax(
        uint32_t v, uint32_t tileLen, uint32_t alignedLen,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t,
        const AscendC::LocalTensor<float>& tileStatsUb)
    {
        auto accUb = UbTensor<float>(kM2AccTileOff, alignedLen);
        auto inUb = UbTensor<float>(kM2InTileOff, alignedLen);
        auto uUb = UbTensor<float>(kM2UTileOff, alignedLen);
        auto workUb = UbTensor<float>(kM2WorkOff, 512U);
        auto sBox = UbTensor<float>(kM2ScalarOff, 64U);

        float sTile = ReduceMaxAndExpUb(inUb, accUb, workUb, sBox, tileLen, t.tempMode, t.invTau);
        VecReduceSum(sBox[8U], inUb, workUb, tileLen);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(inUb, inUb, uUb, tileLen);
        AscendC::PipeBarrier<PIPE_V>();
        VecReduceSum(sBox[16U], inUb, workUb, tileLen);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);

        float dTile = sBox.GetValue(8U);
        tileStatsUb.SetValue(0U, sTile);
        tileStatsUb.SetValue(4U, dTile);
        tileStatsUb.SetValue(8U, sBox.GetValue(16U));
        tileStatsUb.SetValue(12U, ScalarToFloat(v) * dTile);
    }

    __aicore__ inline void ProcessSinglePlane(
        uint32_t planeIdx, const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        if (t.outChannels == 0U) {
            return;
        }
        uint32_t b = SafeDivU(planeIdx, t.outChannels);
        uint32_t k = SafeModU(planeIdx, t.outChannels);
        auto bOneUb = UbTensor<float>(kM2BiasOff, 8U);
        AscendC::DataCopyPad(
            bOneUb, gm_.bias[k], AscendC::DataCopyExtParams{1U, 4U, 0U, 0U, 0U},
            AscendC::DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        float biasVal = bOneUb.GetValue(0U);

        wCacheLen_ = 0U;
        auto accStatsUb = UbTensor<float>(kM2AccStatsOff, 16U);
        auto tileStatsUb = UbTensor<float>(kM2TileStatsOff, 16U);
        auto expUb = UbTensor<float>(kM2ExpTinyOff, 16U);
        ZeroStatsUb(accStatsUb, 1U);
        for (uint32_t u0 = 0U; u0 < t.width; u0 += kMode2TileW) {
            uint32_t rem = t.width - u0;
            uint32_t tileLen = (rem < kMode2TileW) ? rem : kMode2TileW;
            uint32_t alignedLen = (tileLen + 7U) & ~7U;
            FillTileCoords(u0, alignedLen);
            for (uint32_t v = 0U; v < t.height; ++v) {
                ComputeTileConv(b, k, v, u0, tileLen, alignedLen, t);
                ReduceTileSoftmax(v, tileLen, alignedLen, t, tileStatsUb);
                MergeStatsUb(accStatsUb, tileStatsUb, 1U, t.tempMode, t.invTau, expUb);
            }
        }
        StoreStatsPlaneUb(gm_.keypoints, gm_.scores, planeIdx, kM2OutOff, accStatsUb, biasVal, t);
    }
};

}

#endif
