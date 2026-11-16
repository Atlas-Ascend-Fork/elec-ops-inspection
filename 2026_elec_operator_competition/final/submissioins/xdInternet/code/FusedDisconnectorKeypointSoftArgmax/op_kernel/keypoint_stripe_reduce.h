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

#ifndef KEYPOINT_STRIPE_REDUCE_H
#define KEYPOINT_STRIPE_REDUCE_H

#include "keypoint_stripe_conv.h"

namespace keypoint_impl {

template <uint32_t M, uint32_t S, uint32_t K_ACC_PITCH, uint32_t K_CH_DMA, bool K_C2_HALF1>
class StripeFastKernel : public StripeConvStage<M, S, K_ACC_PITCH, K_CH_DMA, K_C2_HALF1> {
    using Base = StripeConvStage<M, S, K_ACC_PITCH, K_CH_DMA, K_C2_HALF1>;

public:
    __aicore__ inline void Process(
        GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t h = t.height;
        const uint32_t w = t.width;
        this->plane_ = h * w;
        const StripeGeometry g = Base::BuildStripeGeometry(block, h, w);
        this->BindStripeTensors(feat, weight, bias, keypoints, scores);
        this->InitLayout(g.rows, g.viewRows, w);

        uint32_t activePack = 0U;
        uint32_t numActive = 0U;
        if (this->ClassifyWeightsVec(g.firstK, activePack, numActive)) {
            if (block == 0U) {
                this->StoreAllZeroBatches(h, w, t);
            }
            return;
        }
        AscendC::ICachePreLoad(4);
        RunStripeAndSync(g, activePack, numActive, t);
    }

private:
    __aicore__ inline void FillZeroStripeSlot(
        const AscendC::LocalTensor<float>& outSlot, uint32_t firstRow, uint32_t rows, uint32_t w) const
    {
        const float sumVal = ScalarToFloat(this->outElems_);
        const float xVal = ScalarToFloat(rows * ((w * (w - 1U)) >> 1U));
        const float yVal = ScalarToFloat(w * (rows * firstRow + ((rows * (rows - 1U)) >> 1U)));
        for (uint32_t m = 0U; m < M; ++m) {
            outSlot.SetValue(0U + m, 0.0f);
            outSlot.SetValue(8U + m, sumVal);
            outSlot.SetValue(16U + m, xVal);
            outSlot.SetValue(24U + m, yVal);
        }
    }

    __aicore__ inline void PreScaleHeatmapDiffs(
        const AscendC::LocalTensor<float>& res, uint32_t tempMode)
    {
        if (tempMode == 2U) {
            for (uint32_t m = 0U; m < M; ++m) {
                AscendC::Muls(Base::AccRt(m, 4U), Base::AccRt(m, 4U), kHugeTauPreScale, this->outElems_);
            }
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t m = 0U; m < M; ++m) {
                const float mx = res.GetValue(m * 8U);
                AscendC::Adds(Base::AccRt(m, 4U), Base::AccRt(m, 4U), 0.0f - (mx * kHugeTauPreScale), this->outElems_);
            }
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        for (uint32_t m = 0U; m < M; ++m) {
            const float mx = res.GetValue(m * 8U);
            AscendC::Adds(Base::AccRt(m, 4U), Base::AccRt(m, 4U), 0.0f - mx, this->outElems_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (tempMode == 1U) {
            for (uint32_t m = 0U; m < M; ++m) {
                AscendC::Muls(Base::AccRt(m, 4U), Base::AccRt(m, 4U), kTinyTauPreScale, this->outElems_);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ScaleAndExpHeatmap(
        const AscendC::LocalTensor<float>& res,
        const AscendC::LocalTensor<float>& outSlot,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        for (uint32_t m = 0U; m < M; ++m) {
            outSlot.SetValue(m, res.GetValue(m * 8U));
        }
        PreScaleHeatmapDiffs(res, t.tempMode);
        for (uint32_t m = 0U; m < M; ++m) {
            AscendC::Muls(Base::AccRt(m, 4U), Base::AccRt(m, 4U), t.invTau, this->outElems_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t m = 0U; m < M; ++m) {
            AscendC::Maxs(Base::AccRt(m, 4U), Base::AccRt(m, 4U), kNegExpFloor, this->outElems_);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t m = 0U; m < M; ++m) {
            AscendC::Exp(Base::AccRt(m, 4U), Base::AccRt(m, 4U), this->outElems_);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FoldAndProjectMoments(
        const AscendC::LocalTensor<float>& xVec,
        const AscendC::LocalTensor<float>& yVec,
        const AscendC::LocalTensor<float>& outSlot,
        uint32_t rows, uint32_t w)
    {
        for (uint32_t active = rows; active > 1U; active = (active + 1U) >> 1U) {
            const uint32_t half = (active + 1U) >> 1U;
            const uint32_t foldCnt = (active >> 1U) * w;
            for (uint32_t m = 0U; m < M; ++m) {
                AscendC::Add(Base::AccRt(m, 4U), Base::AccRt(m, 4U), Base::AccRt(m, 4U)[half * w], foldCnt);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t m = 0U; m < M; ++m) {
            AscendC::Mul(Base::AccRt(m, 2U), Base::AccRt(m, 4U), xVec, w);
            AscendC::Mul(Base::AccRt(m, 3U), Base::AccRt(m, 0U), yVec, rows);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if (w > 64U) {
            for (uint32_t m = 0U; m < M; ++m) {
                AscendC::Add(Base::AccRt(m, 2U), Base::AccRt(m, 2U), Base::AccRt(m, 2U)[64U], w - 64U);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        const int32_t xRedCnt = (w > 64U) ? 64 : static_cast<int32_t>(w);
        constexpr int32_t kAccBlkStride = (M > 1U) ? static_cast<int32_t>(K_ACC_PITCH >> 3U) : 8;
        AscendC::WholeReduceSum(
            outSlot[8U], Base::AccRt(0U, 0U), static_cast<int32_t>(rows),
            static_cast<int32_t>(M), 1, 1, kAccBlkStride);
        AscendC::WholeReduceSum(
            outSlot[16U], Base::AccRt(0U, 2U), xRedCnt,
            static_cast<int32_t>(M), 1, 1, kAccBlkStride);
        AscendC::WholeReduceSum(
            outSlot[24U], Base::AccRt(0U, 3U), static_cast<int32_t>(rows),
            static_cast<int32_t>(M), 1, 1, kAccBlkStride);
        AscendC::ResetMask();
    }

    __aicore__ inline void ComputeMomentsBatched(
        const AscendC::LocalTensor<float>& outSlot,
        uint32_t rows, uint32_t w,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        auto scratch = UbTensor<float>(Base::kInOff, this->dma_.inFloats);
        auto xVec = UbTensor<float>(this->coordOff_, 128U);
        auto yVec = xVec[128U];
        auto res = UbTensor<float>(this->resOff_, 64U);
        for (uint32_t m = 0U; m < M; ++m) {
            VecReduceMax(res[m * 8U], Base::AccRt(m, 4U), scratch[m * 256U], this->outElems_);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        ScaleAndExpHeatmap(res, outSlot, t);

        const int32_t wBlocks = static_cast<int32_t>(w >> 3U);
        const int32_t firstRepCnt = (w > 64U) ? 64 : static_cast<int32_t>(w);
        for (uint32_t m = 0U; m < M; ++m) {
            AscendC::WholeReduceSum(
                Base::AccRt(m, 0U), Base::AccRt(m, 4U), firstRepCnt, static_cast<int32_t>(rows), 1, 1, wBlocks);
            if (w > 64U) {
                AscendC::WholeReduceSum(
                    Base::AccRt(m, 1U), Base::AccRt(m, 4U)[64U], static_cast<int32_t>(w - 64U),
                    static_cast<int32_t>(rows), 1, 1, wBlocks);
            }
        }
        AscendC::ResetMask();
        AscendC::PipeBarrier<PIPE_V>();
        if (w > 64U) {
            for (uint32_t m = 0U; m < M; ++m) {
                AscendC::Add(Base::AccRt(m, 0U), Base::AccRt(m, 0U), Base::AccRt(m, 1U), rows);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        FoldAndProjectMoments(xVec, yVec, outSlot, rows, w);
    }

    __aicore__ inline void ScaleMergeScores(
        const AscendC::LocalTensor<float>& sArr,
        const AscendC::LocalTensor<float>& mxBcast,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        constexpr uint32_t kMergeElems = S * 8U;
        if (t.tempMode == 2U) {
            AscendC::Muls(sArr, sArr, kHugeTauPreScale, kMergeElems);
            AscendC::Muls(mxBcast, mxBcast, kHugeTauPreScale, kMergeElems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(sArr, sArr, mxBcast, kMergeElems);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::Sub(sArr, sArr, mxBcast, kMergeElems);
            AscendC::PipeBarrier<PIPE_V>();
            if (t.tempMode == 1U) {
                AscendC::Muls(sArr, sArr, kTinyTauPreScale, kMergeElems);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        AscendC::Muls(sArr, sArr, t.invTau, kMergeElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(sArr, sArr, 0.0f, static_cast<int32_t>(kMergeElems));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(sArr, sArr, kNegExpFloor, kMergeElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(sArr, sArr, kMergeElems);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreMergedResult(
        uint32_t firstMap, uint32_t firstK,
        const AscendC::LocalTensor<float>& mergeBuf,
        const AscendC::LocalTensor<float>& outBuf,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        auto coeff = UbTensor<float>(this->weightOff_, 1168U);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        for (uint32_t m = 0U; m < M; ++m) {
            const float mx = mergeBuf.GetValue(128U + m);
            const float sumTot = mergeBuf.GetValue(384U + m);
            const float xTot = mergeBuf.GetValue(512U + m);
            const float yTot = mergeBuf.GetValue(640U + m);
            const float bVal = coeff.GetValue(Base::kAllWeights + firstK + m);
            outBuf.SetValue(m * 2U + 0U, SafeDiv(xTot, sumTot) * t.strideF + t.halfStrideF);
            outBuf.SetValue(m * 2U + 1U, SafeDiv(yTot, sumTot) * t.strideF + t.halfStrideF);
            outBuf.SetValue(8U + m, mx + bVal);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(
            this->gm_.keypoints[firstMap * 2U], outBuf,
            AscendC::DataCopyExtParams{1U, static_cast<uint32_t>(2U * M * sizeof(float)), 0U, 0U, 0U});
        AscendC::DataCopyPad(
            this->gm_.scores[firstMap], outBuf[8U],
            AscendC::DataCopyExtParams{1U, static_cast<uint32_t>(M * sizeof(float)), 0U, 0U, 0U});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void MergeAndStore(
        uint32_t groupIdx, uint32_t firstMap, uint32_t firstK,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        auto records = UbTensor<float>(Base::kInOff, S * Base::kSlotFloats);
        auto mergeBuf = UbTensor<float>(Base::kAccOff, 768U);
        auto outBuf = UbTensor<float>(this->resOff_ + 384U, 16U);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::DataCopy(records, this->partials_[groupIdx * S * Base::kSlotFloats], S * Base::kSlotFloats);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        AscendC::Adds(mergeBuf[0U], records, 0.0f, 8ULL, S, AscendC::UnaryRepeatParams{1U, 1U, 1U, 4U});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(mergeBuf[128U], mergeBuf[0U], 0.0f, S * 8U);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t act = S; act > 1U; act = (act + 1U) >> 1U) {
            const uint32_t half = (act + 1U) >> 1U;
            AscendC::Max(mergeBuf[128U], mergeBuf[128U], mergeBuf[128U + half * 8U], (act >> 1U) * 8U);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Adds(mergeBuf[256U], mergeBuf[128U], 0.0f, 8ULL, S, AscendC::UnaryRepeatParams{1U, 1U, 1U, 0U});
        AscendC::PipeBarrier<PIPE_V>();
        ScaleMergeScores(mergeBuf[0U], mergeBuf[256U], t);

        const AscendC::BinaryRepeatParams mulP{1U, 1U, 1U, 1U, 4U, 1U};
        AscendC::Mul(mergeBuf[384U], records[8U], mergeBuf[0U], 8ULL, S, mulP);
        AscendC::Mul(mergeBuf[512U], records[16U], mergeBuf[0U], 8ULL, S, mulP);
        AscendC::Mul(mergeBuf[640U], records[24U], mergeBuf[0U], 8ULL, S, mulP);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t act = S; act > 1U; act = (act + 1U) >> 1U) {
            const uint32_t half = (act + 1U) >> 1U;
            const uint32_t fold = (act >> 1U) * 8U;
            AscendC::Add(mergeBuf[384U], mergeBuf[384U], mergeBuf[384U + half * 8U], fold);
            AscendC::Add(mergeBuf[512U], mergeBuf[512U], mergeBuf[512U + half * 8U], fold);
            AscendC::Add(mergeBuf[640U], mergeBuf[640U], mergeBuf[640U + half * 8U], fold);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::ResetMask();
        StoreMergedResult(firstMap, firstK, mergeBuf, outBuf, t);
    }

    __aicore__ inline void RunStripeAndSync(
        const StripeGeometry& g, uint32_t activePack, uint32_t numActive,
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        auto outSlot = UbTensor<float>(this->resOff_ + 256U, Base::kSlotFloats);
        if (numActive == 0U) {
            FillZeroStripeSlot(outSlot, g.firstRow, g.rows, g.w);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        } else {
            this->ConvolveActiveGroups(g, activePack, numActive);
            ComputeMomentsBatched(outSlot, g.rows, g.w, t);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        }
        AscendC::DataCopy(this->partials_[g.block * Base::kSlotFloats], outSlot, Base::kSlotFloats);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SyncAll<true>();
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
        if (g.stripe == 0U) {
            MergeAndStore(g.groupIdx, g.firstMap, g.firstK, t);
        }
    }
};

}

#endif
