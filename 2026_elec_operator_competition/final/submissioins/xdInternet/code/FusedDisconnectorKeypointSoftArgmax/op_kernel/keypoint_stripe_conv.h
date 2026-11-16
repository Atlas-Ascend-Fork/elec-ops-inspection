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

#ifndef KEYPOINT_STRIPE_CONV_H
#define KEYPOINT_STRIPE_CONV_H

#include "keypoint_common.h"

namespace keypoint_impl {

struct StripeGeometry {
    uint32_t block, groupIdx, stripe, firstMap;
    uint32_t bIdx, firstK, firstRow, rows;
    uint32_t w, topHalo, botHalo, viewFirst, viewRows;
};

template <uint32_t M, uint32_t S, uint32_t K_ACC_PITCH, uint32_t K_CH_DMA, bool K_C2_HALF1>
class StripeConvStage {
protected:
    static constexpr uint32_t kInChannels = 32U;
    static constexpr uint32_t kWeightCount = 288U;
    static constexpr uint32_t kAllWeights = 1152U;
    static constexpr uint32_t kSuperGroups = 4U;
    static constexpr uint32_t kChPerDma = K_CH_DMA;
    static constexpr uint32_t kSubPer8 = (K_CH_DMA > 0U) ? (8U / K_CH_DMA) : 1U;
    static constexpr uint32_t kSlotFloats = 32U;
    static constexpr uint32_t kAccOff = 1024U;
    static constexpr uint32_t kHalf0Taps = K_C2_HALF1 ? 6U : 3U;
    static constexpr uint32_t kAccHalf1Off = kAccOff + kHalf0Taps * M * K_ACC_PITCH * 4U + 256U;
    static constexpr uint32_t kAccTotalFloats = (9U * M * K_ACC_PITCH + 191U) & ~127U;
    static constexpr uint32_t kInOff = kAccOff + kAccTotalFloats * 4U;

    uint32_t plane_ = 0U;
    uint32_t outElems_ = 0U;
    uint32_t offStride_ = 0U;
    uint32_t viewElems_ = 0U;
    uint32_t zeroIdx_ = 0U;
    uint32_t offOff_ = 0U;
    uint32_t coordOff_ = 0U;
    uint32_t weightOff_ = 0U;
    uint32_t resOff_ = 0U;
    DmaSlotLayout dma_{};
    KeypointGmTensors gm_;
    AscendC::GlobalTensor<float> partials_;

    static __aicore__ inline StripeGeometry BuildStripeGeometry(uint32_t block, uint32_t h, uint32_t w)
    {
        StripeGeometry g{};
        g.block = block;
        g.groupIdx = SafeDivU(block, S);
        g.stripe = SafeModU(block, S);
        g.firstMap = g.groupIdx * M;
        g.bIdx = g.firstMap >> 2U;
        g.firstK = g.firstMap & 3U;
        g.firstRow = SafeDivU(g.stripe * h, S);
        const uint32_t endRow = SafeDivU((g.stripe + 1U) * h, S);
        g.rows = endRow - g.firstRow;
        g.w = w;
        g.topHalo = (g.firstRow > 0U) ? 1U : 0U;
        g.botHalo = (endRow < h) ? 1U : 0U;
        g.viewFirst = g.firstRow - g.topHalo;
        g.viewRows = g.rows + g.topHalo + g.botHalo;
        return g;
    }

    __aicore__ inline void BindStripeTensors(
        GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores)
    {
        gm_.Bind(feat, weight, bias, keypoints, scores);
        partials_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetSysWorkSpacePtr()));
    }

    __aicore__ inline void InitLayout(uint32_t rows, uint32_t viewRows, uint32_t w)
    {
        outElems_ = rows * w;
        offStride_ = (outElems_ + 15U) & ~15U;
        viewElems_ = viewRows * w;
        dma_ = ComputeDmaSlotLayout(viewElems_, kChPerDma, 4096U);
        zeroIdx_ = (outElems_ + 63U) & ~63U;
        offOff_ = kInOff + dma_.inFloats * 4U;
        coordOff_ = offOff_ + 2U * offStride_ * 4U;
        weightOff_ = coordOff_ + 256U * 4U;
        resOff_ = weightOff_ + 1168U * 4U;
    }

    static __aicore__ inline constexpr uint32_t AccByteOff(uint32_t m, uint32_t tap)
    {
        if constexpr (K_C2_HALF1) {
            return (tap >= 3U)
                ? (kAccOff + (((tap - 3U) * M + m) * K_ACC_PITCH) * 4U)
                : (kAccHalf1Off + ((tap * M + m) * K_ACC_PITCH) * 4U);
        } else {
            return (tap >= 3U && tap <= 5U)
                ? (kAccOff + (((tap - 3U) * M + m) * K_ACC_PITCH) * 4U)
                : (kAccHalf1Off + ((((tap < 3U) ? tap : (tap - 3U)) * M + m) * K_ACC_PITCH) * 4U);
        }
    }

    template <uint32_t M_IDX, uint32_t TAP>
    static __aicore__ inline AscendC::LocalTensor<float> TapAcc()
    {
        constexpr uint32_t kByteOff = AccByteOff(M_IDX, TAP);
        return UbTensor<float>(kByteOff, K_ACC_PITCH);
    }

    static __aicore__ inline AscendC::LocalTensor<float> AccRt(uint32_t m, uint32_t tap)
    {
        return UbTensor<float>(AccByteOff(m, tap), K_ACC_PITCH);
    }

    __aicore__ inline void StoreAllZeroBatches(
        uint32_t h, uint32_t w, const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        auto coeff = UbTensor<float>(weightOff_, 1168U);
        auto outBuf = UbTensor<float>(resOff_ + 384U, 64U);
        const float kpX = ScalarToFloat(w) * t.halfStrideF;
        const float kpY = ScalarToFloat(h) * t.halfStrideF;
        const uint32_t maps = t.batch * 4U;
        for (uint32_t idx = 0U; idx < maps; ++idx) {
            outBuf.SetValue(idx * 2U + 0U, kpX);
            outBuf.SetValue(idx * 2U + 1U, kpY);
            outBuf.SetValue(32U + idx, coeff.GetValue(kAllWeights + (idx & 3U)));
        }
        const uint32_t kpFloats = maps * 2U;
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        if (gm_.kpAligned) {
            AscendC::DataCopy(gm_.keypoints, outBuf, kpFloats);
        } else {
            AscendC::DataCopyPad(
                gm_.keypoints, outBuf, AscendC::DataCopyExtParams{1U, kpFloats * 4U, 0U, 0U, 0U});
        }
        if (gm_.scAligned && ((maps & 7U) == 0U)) {
            AscendC::DataCopy(gm_.scores, outBuf[32U], maps);
        } else {
            AscendC::DataCopyPad(
                gm_.scores, outBuf[32U], AscendC::DataCopyExtParams{1U, maps * 4U, 0U, 0U, 0U});
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void ReduceWeightGroups(
        const AscendC::LocalTensor<float>& coeff,
        const AscendC::LocalTensor<float>& scratch, const AscendC::LocalTensor<float>& res)
    {
        ReduceSuperGroups72MaxUb(res, scratch, coeff, kAllWeights, 16);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceMax(
            res[32U], res[0U], 16, 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::ResetMask();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }

    static __aicore__ inline bool HasActiveMapGroup(
        const AscendC::LocalTensor<float>& res, uint32_t firstK, uint32_t g)
    {
        for (uint32_t m = 0U; m < M; ++m) {
            if (res.GetValue((firstK + m) * kSuperGroups + g) != 0.0f) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline bool ClassifyWeightsVec(uint32_t firstK, uint32_t& activePack, uint32_t& numActive)
    {
        auto coeff = UbTensor<float>(weightOff_, 1168U);
        auto scratch = UbTensor<float>(kInOff, 2048U);
        auto res = UbTensor<float>(resOff_, 64U);
        LoadWeightAndBiasUb(gm_, coeff, 0U, kAllWeights);
        ReduceWeightGroups(coeff, scratch, res);
        if (res.GetValue(32U) == 0.0f) {
            return true;
        }
        for (uint32_t g = 0U; g < kSuperGroups; ++g) {
            if (HasActiveMapGroup(res, firstK, g)) {
                activePack |= (g << (numActive * 4U));
                ++numActive;
            }
        }
        return false;
    }

    __aicore__ inline void LoadChannelGroup(
        const AscendC::LocalTensor<float>& dst, uint64_t batchBase,
        uint32_t dmaG, uint32_t viewFirst, uint32_t w)
    {
        const uint64_t srcOff = batchBase + static_cast<uint64_t>(dmaG * kChPerDma) * plane_ +
                                static_cast<uint64_t>(viewFirst) * w;
        if (gm_.featAligned) {
            const AscendC::DataCopyParams params{
                static_cast<uint16_t>(kChPerDma),
                static_cast<uint16_t>(viewElems_ >> 3U),
                static_cast<uint16_t>((plane_ - viewElems_) >> 3U),
                static_cast<uint16_t>(dma_.dstGapBlks)};
            AscendC::DataCopy(dst, gm_.feat[srcOff], params);
            return;
        }
        const AscendC::DataCopyExtParams extParams{
            static_cast<uint16_t>(kChPerDma), viewElems_ * 4U, (plane_ - viewElems_) * 4U, dma_.dstGapBlks, 0U};
        AscendC::DataCopyPad(
            dst, gm_.feat[srcOff], extParams, AscendC::DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
    }

    template <uint32_t M_IDX>
    __aicore__ inline void AccumMapDense(
        const AscendC::LocalTensor<float>& coeffK, const AscendC::LocalTensor<float>& c0,
        const AscendC::LocalTensor<float>& c1, const AscendC::LocalTensor<float>& c2,
        uint32_t c, uint8_t rep0, uint8_t rep1, uint8_t rep2) const
    {
        const TapWeights3x3 tw = LoadTapWeights3x3(coeffK, (M_IDX * kInChannels + c) * 9U);
        AxpyRow3Taps(TapAcc<M_IDX, 0U>(), TapAcc<M_IDX, 1U>(), TapAcc<M_IDX, 2U>(), c0, tw.w0, tw.w1, tw.w2, rep0);
        AxpyRow3Taps(TapAcc<M_IDX, 3U>(), TapAcc<M_IDX, 4U>(), TapAcc<M_IDX, 5U>(), c1, tw.w3, tw.w4, tw.w5, rep1);
        AxpyRow3Taps(TapAcc<M_IDX, 6U>(), TapAcc<M_IDX, 7U>(), TapAcc<M_IDX, 8U>(), c2, tw.w6, tw.w7, tw.w8, rep2);
    }

    __aicore__ inline void ConvolveGroupDense(
        const AscendC::LocalTensor<float>& groupSlot, const AscendC::LocalTensor<float>& coeffK,
        uint32_t dmaG, uint32_t srcOff1, uint32_t srcOff2,
        uint8_t rep0, uint8_t rep1, uint8_t rep2) const
    {
        for (uint32_t lane = 0U; lane < kChPerDma; ++lane) {
            const uint32_t c = dmaG * kChPerDma + lane;
            auto c0 = groupSlot[lane * dma_.centerStride];
            auto c1 = c0[srcOff1];
            auto c2 = c0[srcOff2];
            AccumMapDense<0U>(coeffK, c0, c1, c2, c, rep0, rep1, rep2);
            if constexpr (M >= 2U) {
                AccumMapDense<1U>(coeffK, c0, c1, c2, c, rep0, rep1, rep2);
            }
            if constexpr (M >= 4U) {
                AccumMapDense<2U>(coeffK, c0, c1, c2, c, rep0, rep1, rep2);
                AccumMapDense<3U>(coeffK, c0, c1, c2, c, rep0, rep1, rep2);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void InitCoordAndShiftTables(
        uint32_t firstRow, uint32_t rows, uint32_t w, int32_t zeroByteOff)
    {
        auto offL = UbTensor<int32_t>(offOff_, offStride_);
        auto offR = UbTensor<int32_t>(offOff_ + offStride_ * 4U, offStride_);
        auto xVec = UbTensor<float>(coordOff_, 128U);
        auto yVec = xVec[128U];
        const float yStart = ScalarToFloat(firstRow);
        for (uint32_t i = 0U; i < 16U; ++i) {
            const float fi = ScalarToFloat(i);
            offL.SetValue(i, static_cast<int32_t>(i * 4U) - 4);
            offR.SetValue(i, static_cast<int32_t>((i + 1U) * 4U));
            xVec.SetValue(i, fi);
            yVec.SetValue(i, yStart + fi);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        for (uint32_t step = 16U; step < outElems_; step <<= 1U) {
            const uint32_t rem = outElems_ - step;
            const uint32_t cnt = (rem < step) ? rem : step;
            const int32_t delta = static_cast<int32_t>(step * 4U);
            AscendC::Adds(offL[step], offL, delta, cnt);
            AscendC::Adds(offR[step], offR, delta, cnt);
            AscendC::PipeBarrier<PIPE_V>();
        }
        ExpandDoublingFloatUb(xVec, 16U, w);
        ExpandDoublingFloatUb(yVec, 16U, rows);
        PatchShiftRowBoundsUb(offL, offR, rows, w, zeroByteOff);
    }

    __aicore__ inline void CombineShiftedTaps(uint32_t outOff0, uint32_t cnt0, uint32_t cnt2)
    {
        AscendC::ResetMask();
        for (uint32_t m = 0U; m < M; ++m) {
            AscendC::Add(AccRt(m, 3U)[outOff0], AccRt(m, 3U)[outOff0], AccRt(m, 0U), cnt0);
            AscendC::Add(AccRt(m, 4U)[outOff0], AccRt(m, 4U)[outOff0], AccRt(m, 1U), cnt0);
            AscendC::Add(AccRt(m, 5U)[outOff0], AccRt(m, 5U)[outOff0], AccRt(m, 2U), cnt0);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t m = 0U; m < M; ++m) {
            AscendC::Add(AccRt(m, 3U), AccRt(m, 3U), AccRt(m, 6U), cnt2);
            AscendC::Add(AccRt(m, 4U), AccRt(m, 4U), AccRt(m, 7U), cnt2);
            AscendC::Add(AccRt(m, 5U), AccRt(m, 5U), AccRt(m, 8U), cnt2);
        }
        AscendC::PipeBarrier<PIPE_V>();
        auto offL = UbTensor<uint32_t>(offOff_, outElems_);
        auto offR = UbTensor<uint32_t>(offOff_ + offStride_ * 4U, outElems_);
        for (uint32_t m = 0U; m < M; ++m) {
            AscendC::Gather(AccRt(m, 0U), AccRt(m, 3U), offL, 0U, outElems_);
            AscendC::Gather(AccRt(m, 2U), AccRt(m, 5U), offR, 0U, outElems_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t m = 0U; m < M; ++m) {
            AscendC::Add(AccRt(m, 0U), AccRt(m, 0U), AccRt(m, 2U), outElems_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t m = 0U; m < M; ++m) {
            AscendC::Add(AccRt(m, 4U), AccRt(m, 4U), AccRt(m, 0U), outElems_);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrefetchStripeNext(
        const AscendC::LocalTensor<float>& inStorage, uint64_t batchBase,
        uint32_t activePack, uint32_t step, uint32_t viewFirst, uint32_t w)
    {
        const uint32_t nextSlot = (step + 1U) & 1U;
        const event_t nextEv = SlotEventId(nextSlot);
        if (step >= 1U) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(nextEv);
        }
        const uint32_t nextG = UnpackActiveDmaGroup(activePack, kSubPer8, step + 1U);
        LoadChannelGroup(inStorage[nextSlot * dma_.inSlotFloats], batchBase, nextG, viewFirst, w);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(nextEv);
    }

    __aicore__ inline void ConvolveActiveGroups(
        const StripeGeometry& g, uint32_t activePack, uint32_t numActive)
    {
        const uint64_t batchBase = static_cast<uint64_t>(g.bIdx) * kInChannels * plane_;
        auto inStorage = UbTensor<float>(kInOff, dma_.inFloats);
        auto coeffK = UbTensor<float>(weightOff_ + g.firstK * kWeightCount * 4U, M * kWeightCount);
        AscendC::Duplicate(inStorage, 0.0f, dma_.inFloats);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        LoadChannelGroup(
            inStorage, batchBase, UnpackActiveDmaGroup(activePack, kSubPer8, 0U), g.viewFirst, g.w);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Duplicate(UbTensor<float>(kAccOff, kAccTotalFloats), 0.0f, kAccTotalFloats);
        InitCoordAndShiftTables(g.firstRow, g.rows, g.w, static_cast<int32_t>(zeroIdx_ * sizeof(float)));

        const uint32_t cnt0 = (g.rows - (1U - g.topHalo)) * g.w;
        const uint32_t cnt2 = (g.rows - (1U - g.botHalo)) * g.w;
        const uint32_t stripeSteps = numActive * kSubPer8;
        for (uint32_t idxStep = 0U; idxStep < stripeSteps; ++idxStep) {
            const uint32_t slot = WaitStepSlotUb(idxStep);
            if (idxStep + 1U < stripeSteps) {
                PrefetchStripeNext(inStorage, batchBase, activePack, idxStep, g.viewFirst, g.w);
            }
            ConvolveGroupDense(
                inStorage[slot * dma_.inSlotFloats], coeffK,
                UnpackActiveDmaGroup(activePack, kSubPer8, idxStep),
                g.topHalo * g.w, (g.topHalo + 1U) * g.w,
                static_cast<uint8_t>((cnt0 + 63U) >> 6U),
                static_cast<uint8_t>((outElems_ + 63U) >> 6U),
                static_cast<uint8_t>((cnt2 + 63U) >> 6U));
            ReleaseStepSlotUb(idxStep, stripeSteps);
        }
        CombineShiftedTaps((1U - g.topHalo) * g.w, cnt0, cnt2);
    }
};

}

#endif
