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

#ifndef KEYPOINT_MODE4_STRIP_EXEC_H
#define KEYPOINT_MODE4_STRIP_EXEC_H

#include "keypoint_mode4_strip_init.h"

namespace keypoint_impl {

struct StripHaloView {
    uint32_t curR;
    uint32_t wPad;
    uint32_t gatherCnt;
    uint32_t topRows;
    uint32_t topOutOff;
    uint32_t botRows;
    uint32_t cInOff;
    uint32_t bInOff;
};

class KeypointMode4BKFast : public Mode4StripBase {
public:
    __aicore__ inline void Process(
        GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores,
        GM_ADDR workspace, const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        (void)workspace;
        uint32_t coreIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t stepCores = static_cast<uint32_t>(AscendC::GetBlockNum());
        uint32_t totalPlanes = ClampTotalPlanes(t.batch, kM4OutChannels);
        if (stepCores == 0U || coreIdx >= totalPlanes) {
            return;
        }
        this->gm_.Bind(feat, weight, bias, keypoints, scores);

        uint32_t nzPlaneMask = this->InitBuffersAndMask(t);
        auto bUb = UbTensor<float>(kM4BiasOff, 8U);
        auto accStatsUb = UbTensor<float>(kM4AccStatsOff, 16U);
        auto kpOne = UbTensor<float>(kM4OutOff, 8U);
        auto scOne = UbTensor<float>(kM4OutOff + 64U, 8U);
        const float zeroX = ScalarToFloat(t.width) * t.halfStrideF;
        const float zeroY = ScalarToFloat(t.height) * t.halfStrideF;
        for (uint32_t planeIdx = coreIdx; planeIdx < totalPlanes; planeIdx += stepCores) {
            uint32_t b = planeIdx >> 2U;
            uint32_t k = planeIdx & 3U;
            const float bVal = bUb.GetValue(k);
            if ((nzPlaneMask & (1U << k)) == 0U) {
                StoreSinglePlaneUb(this->gm_.keypoints, this->gm_.scores, planeIdx, kpOne, scOne, zeroX, zeroY, bVal);
            } else {
                ZeroStatsUb(accStatsUb, 1U);
                ProcessStripLoop(b, k, t, accStatsUb);
                StoreStatsPlaneUb(this->gm_.keypoints, this->gm_.scores, planeIdx, kM4OutOff, accStatsUb, bVal, t);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    static __aicore__ inline void AccumRowTriple(
        const AscendC::LocalTensor<float>& acc0,
        const AscendC::LocalTensor<float>& acc1,
        const AscendC::LocalTensor<float>& acc2,
        const AscendC::LocalTensor<float>& srcL,
        const AscendC::LocalTensor<float>& srcC,
        const AscendC::LocalTensor<float>& srcR,
        float wL, float wC, float wR, uint32_t elemCount)
    {
        bool any = false;
        if (wL != 0.0f) {
            AscendC::Axpy(acc0, srcL, wL, elemCount);
            any = true;
        }
        if (wC != 0.0f) {
            AscendC::Axpy(acc1, srcC, wC, elemCount);
            any = true;
        }
        if (wR != 0.0f) {
            AscendC::Axpy(acc2, srcR, wR, elemCount);
            any = true;
        }
        if (any) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    static __aicore__ inline void GatherChanEdgesM4(
        const AscendC::LocalTensor<float>& chUb, const AscendC::LocalTensor<float>& leftUb,
        const AscendC::LocalTensor<float>& rightUb, uint32_t gatherCnt, bool needL, bool needR)
    {
        if (needL) {
            AscendC::Gather(leftUb, chUb, UbTensor<uint32_t>(kM4OffLeftOff, gatherCnt), 0U, gatherCnt);
        }
        if (needR) {
            AscendC::Gather(rightUb, chUb, UbTensor<uint32_t>(kM4OffRightOff, gatherCnt), 0U, gatherCnt);
        }
        if (needL || needR) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    static __aicore__ inline void AccumSingleChanM4(
        const AscendC::LocalTensor<float>& acc0, const AscendC::LocalTensor<float>& acc1,
        const AscendC::LocalTensor<float>& acc2, uint32_t chanOff,
        const TapWeights3x3& tw, const StripHaloView& hv)
    {
        bool needL = (tw.w0 != 0.0f) || (tw.w3 != 0.0f) || (tw.w6 != 0.0f);
        bool needC = (tw.w1 != 0.0f) || (tw.w4 != 0.0f) || (tw.w7 != 0.0f);
        bool needR = (tw.w2 != 0.0f) || (tw.w5 != 0.0f) || (tw.w8 != 0.0f);
        if (!needL && !needC && !needR) {
            return;
        }
        auto chUb = UbTensor<float>(chanOff, 8U + hv.gatherCnt);
        auto leftUb = UbTensor<float>(kM4LeftOff, hv.gatherCnt);
        auto rightUb = UbTensor<float>(kM4RightOff, hv.gatherCnt);
        GatherChanEdgesM4(chUb, leftUb, rightUb, hv.gatherCnt, needL, needR);
        if (hv.topRows > 0U) {
            AccumRowTriple(
                acc0[hv.topOutOff], acc1[hv.topOutOff], acc2[hv.topOutOff],
                leftUb, chUb[8U], rightUb, tw.w0, tw.w1, tw.w2, hv.topRows * hv.wPad);
        }
        AccumRowTriple(
            acc0, acc1, acc2, leftUb[hv.cInOff], chUb[8U + hv.cInOff], rightUb[hv.cInOff],
            tw.w3, tw.w4, tw.w5, hv.curR * hv.wPad);
        if (hv.botRows > 0U) {
            AccumRowTriple(
                acc0, acc1, acc2, leftUb[hv.bInOff], chUb[8U + hv.bInOff], rightUb[hv.bInOff],
                tw.w6, tw.w7, tw.w8, hv.botRows * hv.wPad);
        }
    }

    static __aicore__ inline void AccumChannelPair1(
        const AscendC::LocalTensor<float>& acc0,
        const AscendC::LocalTensor<float>& acc1,
        const AscendC::LocalTensor<float>& acc2,
        uint32_t k, uint32_t pairIdx, uint32_t slotIdx,
        uint32_t curR, bool topExist, bool botExist, uint32_t inRows, uint32_t wPad)
    {
        uint32_t slotBase = (slotIdx == 0U) ? kM4Slot0Off : kM4Slot1Off;
        uint32_t centerRowOff = topExist ? 1U : 0U;
        StripHaloView hv{
            curR, wPad, inRows * wPad,
            topExist ? curR : ((curR > 1U) ? (curR - 1U) : 0U),
            (topExist ? 0U : 1U) * wPad,
            botExist ? curR : ((curR > 1U) ? (curR - 1U) : 0U),
            centerRowOff * wPad,
            (centerRowOff + 1U) * wPad};
        auto wUb = UbTensor<float>(kM4WeightOff, 1152U);
        uint32_t kBase = k * kM4PlaneWeights;
        for (uint32_t cc = 0U; cc < 2U; ++cc) {
            const TapWeights3x3 tw = LoadTapWeights3x3(wUb, kBase + (pairIdx * 2U + cc) * 9U);
            AccumSingleChanM4(acc0, acc1, acc2, slotBase + cc * kM4ChanBytes, tw, hv);
        }
    }

    static __aicore__ inline void ApplyStripMask1(
        const AscendC::LocalTensor<float>& acc0, uint32_t curR, uint32_t wPad, bool addTailFloor)
    {
        auto validUb = UbTensor<float>(kM4ValidMaskOff, wPad);
        for (uint32_t r = 0U; r < curR; ++r) {
            uint32_t off = r * wPad;
            AscendC::Mul(acc0[off], acc0[off], validUb, wPad);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (addTailFloor) {
            auto floorUb = UbTensor<float>(kM4TailFloorOff, wPad);
            for (uint32_t r = 0U; r < curR; ++r) {
                uint32_t off = r * wPad;
                AscendC::Add(acc0[off], acc0[off], floorUb, wPad);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    static __aicore__ inline void ReduceRowsM4(
        const AscendC::LocalTensor<float>& acc0,
        const AscendC::LocalTensor<float>& rowSumA,
        const AscendC::LocalTensor<float>& rowSumB,
        uint32_t curR, uint32_t wPad,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        if (t.height > t.maxStripRows) {
            AscendC::Duplicate(rowSumA, 0.0f, 64U);
            AscendC::Duplicate(rowSumB, 0.0f, 64U);
            AscendC::PipeBarrier<PIPE_V>();
        }
        uint32_t curRAligned = (curR + 7U) & ~7U;
        int32_t strideBlks = static_cast<int32_t>(wPad >> 3U);
        if (wPad <= 64U) {
            AscendC::WholeReduceSum(
                rowSumA, acc0, static_cast<int32_t>(wPad), static_cast<int32_t>(curR), 1, 1, strideBlks);
            AscendC::ResetMask();
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        AscendC::WholeReduceSum(rowSumA, acc0, 64, static_cast<int32_t>(curR), 1, 1, strideBlks);
        uint32_t nChunks = (wPad + 63U) >> 6U;
        for (uint32_t c = 1U; c < nChunks; ++c) {
            uint32_t off = c * 64U;
            uint32_t rem = wPad - off;
            uint32_t cnt = (rem < 64U) ? rem : 64U;
            AscendC::WholeReduceSum(
                rowSumB, acc0[off], static_cast<int32_t>(cnt), static_cast<int32_t>(curR), 1, 1, strideBlks);
            AscendC::ResetMask();
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(rowSumA, rowSumA, rowSumB, curRAligned);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    static __aicore__ inline void ExportStripStatsM4(
        const AscendC::LocalTensor<float>& sBox, float sLoc,
        const AscendC::LocalTensor<float>& locStatsUb)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        locStatsUb.SetValue(0U, sLoc);
        locStatsUb.SetValue(4U, sBox.GetValue(8U));
        locStatsUb.SetValue(8U, sBox.GetValue(24U));
        locStatsUb.SetValue(12U, sBox.GetValue(16U));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
    }

    static __aicore__ inline void ProjectStripMomentsM4(
        const AscendC::LocalTensor<float>& acc0,
        const AscendC::LocalTensor<float>& rowSumA,
        const AscendC::LocalTensor<float>& rowSumB,
        const AscendC::LocalTensor<float>& workUb,
        const AscendC::LocalTensor<float>& sBox,
        uint32_t r0, uint32_t curR, uint32_t wPad)
    {
        uint32_t curRAligned = (curR + 7U) & ~7U;
        VecReduceSum(sBox[8U], rowSumA, workUb, curR);
        if (r0 == 0U) {
            AscendC::Mul(rowSumB, rowSumA, UbTensor<float>(kM4VBaseOff, curRAligned), curRAligned);
        } else {
            auto vStrip = UbTensor<float>(kM4VStripOff, curRAligned);
            AscendC::Adds(vStrip, UbTensor<float>(kM4VBaseOff, curRAligned), ScalarToFloat(r0), curRAligned);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(rowSumB, rowSumA, vStrip, curRAligned);
        }
        AscendC::PipeBarrier<PIPE_V>();
        VecReduceSum(sBox[16U], rowSumB, workUb, curR);

        for (uint32_t remRows = curR; remRows > 1U; ) {
            uint32_t half = remRows >> 1U;
            uint32_t odd = remRows & 1U;
            AscendC::Add(acc0, acc0, acc0[(half + odd) * wPad], half * wPad);
            AscendC::PipeBarrier<PIPE_V>();
            remRows = half + odd;
        }
        AscendC::Mul(acc0, acc0, UbTensor<float>(kM4UCoordOff, wPad), wPad);
        AscendC::PipeBarrier<PIPE_V>();
        VecReduceSum(sBox[24U], acc0, workUb, wPad);
    }

    static __aicore__ inline void ReduceStrip1Channel(
        const AscendC::LocalTensor<float>& acc0,
        const AscendC::LocalTensor<float>& acc1,
        const AscendC::LocalTensor<float>& acc2,
        uint32_t r0, uint32_t curR, uint32_t pElems,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t,
        const AscendC::LocalTensor<float>& locStatsUb)
    {
        uint32_t wPad = t.wPad;
        AscendC::Add(acc0, acc0, acc1, pElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(acc0, acc0, acc2, pElems);
        AscendC::PipeBarrier<PIPE_V>();

        bool unaligned = (t.flags & 1U) == 0U;
        if (unaligned) {
            ApplyStripMask1(acc0, curR, wPad, true);
        }

        auto workUb = UbTensor<float>(kM4WorkOff, 256U);
        auto sBox = UbTensor<float>(kM4ScalarBoxOff, 64U);
        float sLoc = ReduceMaxAndExpUb(acc0, acc0, workUb, sBox, pElems, t.tempMode, t.invTau);
        if (unaligned) {
            ApplyStripMask1(acc0, curR, wPad, false);
        }

        auto rowSumA = UbTensor<float>(kM4RowSumAOff, 64U);
        auto rowSumB = UbTensor<float>(kM4RowSumBOff, 64U);
        ReduceRowsM4(acc0, rowSumA, rowSumB, curR, wPad, t);
        ProjectStripMomentsM4(acc0, rowSumA, rowSumB, workUb, sBox, r0, curR, wPad);
        ExportStripStatsM4(sBox, sLoc, locStatsUb);
    }

    __aicore__ inline void ProcessSingleStrip(
        uint32_t bIdx, uint32_t kIdx, uint32_t r0, uint32_t curR,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t,
        const AscendC::LocalTensor<float>& accStatsUb)
    {
        bool topExist = (r0 > 0U);
        bool botExist = (r0 + curR < t.height);
        uint32_t inStartRow = topExist ? (r0 - 1U) : r0;
        uint32_t inRows = curR + (topExist ? 1U : 0U) + (botExist ? 1U : 0U);
        uint64_t batchBase = static_cast<uint64_t>(bIdx) * kM4InChannels * t.height * t.width;

        this->LoadChannelPair(batchBase, 0U, inStartRow, inRows, 0U, t);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        this->LoadChannelPair(batchBase, 1U, inStartRow, inRows, 1U, t);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

        uint32_t pElems = curR * t.wPad;
        auto acc0 = UbTensor<float>(kM4Acc0Off, pElems);
        auto acc1 = UbTensor<float>(kM4Acc1Off, pElems);
        auto acc2 = UbTensor<float>(kM4Acc2Off, pElems);
        AscendC::Duplicate(acc0, 0.0f, pElems);
        AscendC::Duplicate(acc1, 0.0f, pElems);
        AscendC::Duplicate(acc2, 0.0f, pElems);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t p = 0U; p < 16U; ++p) {
            uint32_t slot = p & 1U;
            event_t ev = SlotEventId(slot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ev);
            AccumChannelPair1(acc0, acc1, acc2, kIdx, p, slot, curR, topExist, botExist, inRows, t.wPad);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ev);
            if (p + 2U < 16U) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ev);
                this->LoadChannelPair(batchBase, p + 2U, inStartRow, inRows, slot, t);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ev);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::PipeBarrier<PIPE_V>();

        auto locStatsUb = UbTensor<float>(kM4LocStatsOff, 16U);
        ReduceStrip1Channel(acc0, acc1, acc2, r0, curR, pElems, t, locStatsUb);
        auto expTinyUb = UbTensor<float>(kM4ExpTinyOff, 16U);
        MergeStatsUb(accStatsUb, locStatsUb, 1U, t.tempMode, t.invTau, expTinyUb);
    }

    __aicore__ inline void ProcessStripLoop(
        uint32_t bIdx, uint32_t kIdx,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t,
        const AscendC::LocalTensor<float>& accStatsUb)
    {
        uint32_t maxR = (t.maxStripRows > 0U) ? t.maxStripRows : 1U;
        uint32_t processed = 0U;
        while (processed < t.height) {
            uint32_t rem = t.height - processed;
            uint32_t curR = (rem < maxR) ? rem : maxR;
            ProcessSingleStrip(bIdx, kIdx, processed, curR, t, accStatsUb);
            processed += curR;
        }
    }
};

}

#endif
