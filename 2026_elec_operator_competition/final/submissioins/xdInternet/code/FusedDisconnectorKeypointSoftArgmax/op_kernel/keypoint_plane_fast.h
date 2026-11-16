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

#ifndef KEYPOINT_PLANE_FAST_H
#define KEYPOINT_PLANE_FAST_H

#include "keypoint_common.h"

namespace keypoint_impl {

template <uint32_t K_ACC_PITCH, uint32_t K_CH_DMA>
class PlaneFastAivKernel {
public:
    static constexpr uint32_t kPlaneInCh = 32U;
    static constexpr uint32_t kPlaneWeights = 288U;
    static constexpr uint32_t kChPerDma = K_CH_DMA;
    static constexpr uint32_t kSubPer8 = (K_CH_DMA > 0U) ? (8U / K_CH_DMA) : 1U;
    static constexpr uint32_t kAccOff = 1024U;
    static constexpr uint32_t kAccHalf1Off = kAccOff + 3U * K_ACC_PITCH * 4U + 256U;
    static constexpr uint32_t kAccTotalFloats = (9U * K_ACC_PITCH + 191U) & ~127U;
    static constexpr uint32_t kInOff = kAccOff + kAccTotalFloats * 4U;

    __aicore__ inline void Process(
        GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        const uint32_t rawBlock = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const bool tileExcl = ((t.flags & 2U) != 0U);
        if (tileExcl && ((rawBlock & 1U) != 0U)) {
            return;
        }
        const uint32_t mapIdx = tileExcl ? (rawBlock >> 1U) : rawBlock;
        if (mapIdx >= t.batch * 4U) {
            return;
        }
        const uint32_t bIdx = mapIdx >> 2U;
        const uint32_t kIdx = mapIdx & 3U;
        const uint32_t h = t.height;
        const uint32_t w = t.width;
        plane_ = h * w;

        gm_.Bind(feat, weight, bias, keypoints, scores);
        InitLayout();

        uint32_t activePack = 0U;
        uint32_t numActive = 0U;
        if (ClassifySingleKWeights(kIdx, activePack, numActive)) {
            StoreZeroPlane(mapIdx, kIdx, h, w, t);
            return;
        }
        AscendC::ICachePreLoad(4);
        ConvolveActivePlane(activePack, numActive, bIdx, h, w);
        ComputeAndStorePlane(mapIdx, kIdx, h, w, t);
    }

private:
    uint32_t plane_ = 0U;
    uint32_t zeroIdx_ = 0U;
    uint32_t weightOff_ = 0U;
    uint32_t resOff_ = 0U;
    DmaSlotLayout dma_{};
    KeypointGmTensors gm_;

    __aicore__ inline void InitLayout()
    {
        dma_ = ComputeDmaSlotLayout(plane_, kChPerDma, 2048U);
        zeroIdx_ = (plane_ + 63U) & ~63U;
        weightOff_ = kInOff + dma_.inFloats * 4U;
        resOff_ = weightOff_ + 320U * 4U;
    }

    static __aicore__ inline constexpr uint32_t AccByteOff(uint32_t tap)
    {
        return (tap >= 6U)
            ? (kAccOff + ((tap - 6U) * K_ACC_PITCH) * 4U)
            : (kAccHalf1Off + (tap * K_ACC_PITCH) * 4U);
    }

    template <uint32_t TAP>
    static __aicore__ inline AscendC::LocalTensor<float> TapAcc()
    {
        constexpr uint32_t kByteOff = AccByteOff(TAP);
        return UbTensor<float>(kByteOff, K_ACC_PITCH);
    }

    static __aicore__ inline AscendC::LocalTensor<float> AccRt(uint32_t tap)
    {
        return UbTensor<float>(AccByteOff(tap), K_ACC_PITCH);
    }

    __aicore__ inline bool ClassifySingleKWeights(uint32_t kIdx, uint32_t& activePack, uint32_t& numActive)
    {
        auto coeff = UbTensor<float>(weightOff_, 320U);
        auto scratch = UbTensor<float>(kInOff, 1024U);
        auto res = UbTensor<float>(resOff_, 64U);
        LoadWeightAndBiasUb(gm_, coeff, kIdx * kPlaneWeights, kPlaneWeights);
        ReduceSuperGroups72MaxUb(res, scratch, coeff, kPlaneWeights, 4);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        for (uint32_t g = 0U; g < 4U; ++g) {
            if (res.GetValue(g) != 0.0f) {
                activePack |= (g << (numActive * 4U));
                ++numActive;
            }
        }
        return numActive == 0U;
    }

    __aicore__ inline void StoreZeroPlane(
        uint32_t mapIdx, uint32_t kIdx, uint32_t h, uint32_t w,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        auto coeff = UbTensor<float>(weightOff_, 320U);
        auto outBuf = UbTensor<float>(resOff_ + 256U, 16U);
        const float kpX = ScalarToFloat(w) * t.halfStrideF;
        const float kpY = ScalarToFloat(h) * t.halfStrideF;
        StoreSinglePlaneUb(
            gm_.keypoints, gm_.scores, mapIdx, outBuf, outBuf[8U],
            kpX, kpY, coeff.GetValue(kPlaneWeights + kIdx));
    }

    __aicore__ inline void LoadChannelGroup(
        const AscendC::LocalTensor<float>& dst, uint64_t batchBase, uint32_t dmaG)
    {
        const uint64_t srcOff = batchBase + static_cast<uint64_t>(dmaG * kChPerDma) * plane_;
        if (gm_.featAligned) {
            const AscendC::DataCopyParams params{
                static_cast<uint16_t>(kChPerDma),
                static_cast<uint16_t>(plane_ >> 3U),
                0U,
                static_cast<uint16_t>(dma_.dstGapBlks)};
            AscendC::DataCopy(dst, gm_.feat[srcOff], params);
            return;
        }
        const AscendC::DataCopyExtParams extParams{
            static_cast<uint16_t>(kChPerDma), plane_ * 4U, 0U, dma_.dstGapBlks, 0U};
        AscendC::DataCopyPad(
            dst, gm_.feat[srcOff], extParams, AscendC::DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
    }

    __aicore__ inline void ConvolveGroupDense(
        const AscendC::LocalTensor<float>& groupSlot, const AscendC::LocalTensor<float>& coeff,
        uint32_t dmaG, uint32_t w, uint8_t rep02, uint8_t rep1) const
    {
        for (uint32_t lane = 0U; lane < kChPerDma; ++lane) {
            const uint32_t c = dmaG * kChPerDma + lane;
            auto c0 = groupSlot[lane * dma_.centerStride];
            auto c2 = c0[w];
            const TapWeights3x3 tw = LoadTapWeights3x3(coeff, c * 9U);
            AxpyRow3Taps(TapAcc<0U>(), TapAcc<1U>(), TapAcc<2U>(), c0, tw.w0, tw.w1, tw.w2, rep02);
            AxpyRow3Taps(TapAcc<3U>(), TapAcc<4U>(), TapAcc<5U>(), c0, tw.w3, tw.w4, tw.w5, rep1);
            AxpyRow3Taps(TapAcc<6U>(), TapAcc<7U>(), TapAcc<8U>(), c2, tw.w6, tw.w7, tw.w8, rep02);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrefetchPlaneNext(
        const AscendC::LocalTensor<float>& inStorage, uint64_t batchBase, uint32_t activePack, uint32_t step)
    {
        const uint32_t nextSlot = (step + 1U) & 1U;
        const event_t nextEv = SlotEventId(nextSlot);
        if (step >= 1U) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(nextEv);
        }
        const uint32_t nextG = UnpackActiveDmaGroup(activePack, kSubPer8, step + 1U);
        LoadChannelGroup(inStorage[nextSlot * dma_.inSlotFloats], batchBase, nextG);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(nextEv);
    }

    __aicore__ inline void CombinePlaneTaps(uint32_t h, uint32_t w, uint32_t cnt02)
    {
        AscendC::ResetMask();
        AscendC::Add(AccRt(3U)[w], AccRt(3U)[w], AccRt(0U), cnt02);
        AscendC::Add(AccRt(4U)[w], AccRt(4U)[w], AccRt(1U), cnt02);
        AscendC::Add(AccRt(5U)[w], AccRt(5U)[w], AccRt(2U), cnt02);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(AccRt(3U), AccRt(3U), AccRt(6U), cnt02);
        AscendC::Add(AccRt(4U), AccRt(4U), AccRt(7U), cnt02);
        AscendC::Add(AccRt(5U), AccRt(5U), AccRt(8U), cnt02);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        auto offLi = UbTensor<int32_t>(AccByteOff(6U), plane_);
        auto offRi = UbTensor<int32_t>(AccByteOff(7U), plane_);
        BuildShiftOffsetsUb(offLi, offRi, h, w, plane_, static_cast<int32_t>(zeroIdx_ * sizeof(float)));
        auto offL = UbTensor<uint32_t>(AccByteOff(6U), plane_);
        auto offR = UbTensor<uint32_t>(AccByteOff(7U), plane_);
        AscendC::Gather(AccRt(0U), AccRt(3U), offL, 0U, plane_);
        AscendC::Gather(AccRt(2U), AccRt(5U), offR, 0U, plane_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(AccRt(0U), AccRt(0U), AccRt(2U), plane_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(AccRt(4U), AccRt(4U), AccRt(0U), plane_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ConvolveActivePlane(
        uint32_t activePack, uint32_t numActive, uint32_t bIdx, uint32_t h, uint32_t w)
    {
        const uint64_t batchBase = static_cast<uint64_t>(bIdx) * kPlaneInCh * plane_;
        auto inStorage = UbTensor<float>(kInOff, dma_.inFloats);
        auto coeff = UbTensor<float>(weightOff_, 320U);
        AscendC::Duplicate(inStorage, 0.0f, dma_.inFloats);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        LoadChannelGroup(inStorage, batchBase, UnpackActiveDmaGroup(activePack, kSubPer8, 0U));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Duplicate(UbTensor<float>(kAccOff, kAccTotalFloats), 0.0f, kAccTotalFloats);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ResetMask();

        const uint32_t cnt02 = (h - 1U) * w;
        const uint8_t rep02 = static_cast<uint8_t>((cnt02 + 63U) >> 6U);
        const uint8_t rep1 = static_cast<uint8_t>((plane_ + 63U) >> 6U);
        const uint32_t totalPlaneSteps = numActive * kSubPer8;
        for (uint32_t step = 0U; step < totalPlaneSteps; ++step) {
            const uint32_t slot = WaitStepSlotUb(step);
            if (step + 1U < totalPlaneSteps) {
                PrefetchPlaneNext(inStorage, batchBase, activePack, step);
            }
            const uint32_t curG = UnpackActiveDmaGroup(activePack, kSubPer8, step);
            ConvolveGroupDense(inStorage[slot * dma_.inSlotFloats], coeff, curG, w, rep02, rep1);
            ReleaseStepSlotUb(step, totalPlaneSteps);
        }
        CombinePlaneTaps(h, w, cnt02);
    }

    __aicore__ inline void FoldAndReducePlane(
        const AscendC::LocalTensor<float>& xVec, const AscendC::LocalTensor<float>& yVec,
        const AscendC::LocalTensor<float>& outSlot, uint32_t h, uint32_t w)
    {
        for (uint32_t active = h; active > 1U; active = (active + 1U) >> 1U) {
            const uint32_t half = (active + 1U) >> 1U;
            AscendC::Add(AccRt(4U), AccRt(4U), AccRt(4U)[half * w], (active >> 1U) * w);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Mul(AccRt(2U), AccRt(4U), xVec, w);
        AscendC::Mul(AccRt(3U), AccRt(0U), yVec, h);
        AscendC::PipeBarrier<PIPE_V>();
        if (w > 64U) {
            AscendC::Add(AccRt(2U), AccRt(2U), AccRt(2U)[64U], w - 64U);
            AscendC::PipeBarrier<PIPE_V>();
        }
        const int32_t xRedCnt = (w > 64U) ? 64 : static_cast<int32_t>(w);
        AscendC::WholeReduceSum(outSlot[8U], AccRt(0U), static_cast<int32_t>(h), 1, 1, 1, 8);
        AscendC::WholeReduceSum(outSlot[16U], AccRt(2U), xRedCnt, 1, 1, 1, 8);
        AscendC::WholeReduceSum(outSlot[24U], AccRt(3U), static_cast<int32_t>(h), 1, 1, 1, 8);
        AscendC::ResetMask();
    }

    __aicore__ inline void ComputeAndStorePlane(
        uint32_t mapIdx, uint32_t kIdx, uint32_t h, uint32_t w,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        auto inStorage = UbTensor<float>(kInOff, dma_.inFloats);
        auto xVec = inStorage;
        auto yVec = inStorage[128U];
        auto scratch = inStorage[256U];
        auto coeff = UbTensor<float>(weightOff_, 320U);
        auto res = UbTensor<float>(resOff_, 64U);
        auto outSlot = UbTensor<float>(resOff_ + 256U, 32U);
        auto outBuf = UbTensor<float>(resOff_ + 384U, 16U);

        const float mx = ReduceMaxAndExpUb(AccRt(4U), AccRt(4U), scratch, res[0U], plane_, t.tempMode, t.invTau);
        FillCoordVectorsUb(xVec, yVec, 0U, h, w);

        const int32_t wBlocks = static_cast<int32_t>(w >> 3U);
        const int32_t firstRepCnt = (w > 64U) ? 64 : static_cast<int32_t>(w);
        AscendC::WholeReduceSum(AccRt(0U), AccRt(4U), firstRepCnt, static_cast<int32_t>(h), 1, 1, wBlocks);
        if (w > 64U) {
            AscendC::WholeReduceSum(
                AccRt(1U), AccRt(4U)[64U], static_cast<int32_t>(w - 64U), static_cast<int32_t>(h), 1, 1, wBlocks);
        }
        AscendC::ResetMask();
        AscendC::PipeBarrier<PIPE_V>();
        if (w > 64U) {
            AscendC::Add(AccRt(0U), AccRt(0U), AccRt(1U), h);
            AscendC::PipeBarrier<PIPE_V>();
        }
        FoldAndReducePlane(xVec, yVec, outSlot, h, w);

        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        const float sumTot = outSlot.GetValue(8U);
        const float kpX = SafeDiv(outSlot.GetValue(16U), sumTot) * t.strideF + t.halfStrideF;
        const float kpY = SafeDiv(outSlot.GetValue(24U), sumTot) * t.strideF + t.halfStrideF;
        StoreSinglePlaneUb(
            gm_.keypoints, gm_.scores, mapIdx, outBuf, outBuf[8U],
            kpX, kpY, mx + coeff.GetValue(kPlaneWeights + kIdx));
    }
};

}

#endif
