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

#ifndef KEYPOINT_COMMON_H
#define KEYPOINT_COMMON_H

#include "kernel_operator.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"

namespace keypoint_impl {

static constexpr float kTinyTauPreScale = 0x1.0p126f;
static constexpr float kHugeTauPreScale = 0x1.0p-64f;
static constexpr float kNegExpFloor = -104.0f;

static __aicore__ inline uint32_t SafeDivU(uint32_t num, uint32_t den)
{
    return (den > 0U) ? (num / den) : 0U;
}
static __aicore__ inline uint32_t SafeModU(uint32_t num, uint32_t den)
{
    return (den > 0U) ? (num % den) : 0U;
}
static __aicore__ inline float SafeDiv(float num, float den)
{
    return (den > 0.0f) ? (num / den) : 0.0f;
}
static __aicore__ inline float ScalarToFloat(uint32_t v)
{
    return static_cast<float>(static_cast<int32_t>(v));
}
static __aicore__ inline uint32_t ClampTotalPlanes(uint32_t batch, uint32_t channels)
{
    uint64_t prod = static_cast<uint64_t>(batch) * channels;
    return (prod > 0xFFFFFFFFULL) ? 0xFFFFFFFFU : static_cast<uint32_t>(prod);
}
template <typename T>
static __aicore__ inline AscendC::LocalTensor<T> UbTensor(uint32_t byteOffset, uint32_t elemCount)
{
    return AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC, byteOffset, elemCount);
}
static __aicore__ inline event_t SlotEventId(uint32_t slot)
{
    return (slot == 0U) ? EVENT_ID0 : EVENT_ID1;
}

static __aicore__ inline uint32_t WaitStepSlotUb(uint32_t step)
{
    const uint32_t slot = step & 1U;
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(SlotEventId(slot));
    return slot;
}
static __aicore__ inline void ReleaseStepSlotUb(uint32_t step, uint32_t totalSteps)
{
    if (step + 2U < totalSteps) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(SlotEventId(step & 1U));
    }
}

static __aicore__ inline uint32_t UnpackActiveDmaGroup(uint32_t activePack, uint32_t subPer8, uint32_t step)
{
    const uint32_t grpIdx = SafeDivU(step, subPer8);
    const uint32_t subIdx = SafeModU(step, subPer8);
    const uint32_t g8 = (activePack >> (grpIdx * 4U)) & 0xFU;
    return g8 * subPer8 + subIdx;
}

struct DmaSlotLayout {
    uint32_t dstGapBlks = 1U;
    uint32_t centerStride = 0U;
    uint32_t inSlotFloats = 0U;
    uint32_t inFloats = 0U;
};

static __aicore__ inline DmaSlotLayout ComputeDmaSlotLayout(uint32_t elems, uint32_t chPerDma, uint32_t minFloats)
{
    DmaSlotLayout lay{};
    const uint32_t rem128 = elems & 127U;
    lay.dstGapBlks = (rem128 == 0U) ? 16U : ((128U - rem128) >> 3U);
    lay.centerStride = elems + lay.dstGapBlks * 8U;
    lay.inSlotFloats = (chPerDma * lay.centerStride + 64U + 127U) & ~127U;
    const uint32_t twoSlots = 2U * lay.inSlotFloats;
    lay.inFloats = (twoSlots < minFloats) ? minFloats : twoSlots;
    return lay;
}

struct TapWeights3x3 {
    float w0, w1, w2;
    float w3, w4, w5;
    float w6, w7, w8;
};

static __aicore__ inline TapWeights3x3 LoadTapWeights3x3(const AscendC::LocalTensor<float>& coeffUb, uint32_t wBase)
{
    return TapWeights3x3{
        coeffUb.GetValue(wBase + 0U), coeffUb.GetValue(wBase + 1U), coeffUb.GetValue(wBase + 2U),
        coeffUb.GetValue(wBase + 3U), coeffUb.GetValue(wBase + 4U), coeffUb.GetValue(wBase + 5U),
        coeffUb.GetValue(wBase + 6U), coeffUb.GetValue(wBase + 7U), coeffUb.GetValue(wBase + 8U)};
}

static __aicore__ inline void AxpyRow3Taps(
    const AscendC::LocalTensor<float>& dst0, const AscendC::LocalTensor<float>& dst1,
    const AscendC::LocalTensor<float>& dst2, const AscendC::LocalTensor<float>& src,
    float wA, float wB, float wC, uint8_t reps)
{
    const AscendC::UnaryRepeatParams uRep{1U, 1U, 8U, 8U};
    AscendC::Axpy<float, float, false>(dst0, src, wA, AscendC::MASK_PLACEHOLDER, reps, uRep);
    AscendC::Axpy<float, float, false>(dst1, src, wB, AscendC::MASK_PLACEHOLDER, reps, uRep);
    AscendC::Axpy<float, float, false>(dst2, src, wC, AscendC::MASK_PLACEHOLDER, reps, uRep);
}

struct KeypointGmTensors {
    AscendC::GlobalTensor<float> feat, weight, bias, keypoints, scores;
    bool featAligned = true;
    bool weightAligned = true;
    bool kpAligned = true;
    bool scAligned = true;

    __aicore__ inline void Bind(
        GM_ADDR featAddr, GM_ADDR weightAddr, GM_ADDR biasAddr, GM_ADDR kpAddr, GM_ADDR scAddr)
    {
        featAligned = ((reinterpret_cast<uintptr_t>(featAddr) & 31ULL) == 0ULL);
        weightAligned = ((reinterpret_cast<uintptr_t>(weightAddr) & 31ULL) == 0ULL);
        kpAligned = ((reinterpret_cast<uintptr_t>(kpAddr) & 31ULL) == 0ULL);
        scAligned = ((reinterpret_cast<uintptr_t>(scAddr) & 31ULL) == 0ULL);
        feat.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(featAddr));
        weight.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weightAddr));
        bias.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(biasAddr));
        keypoints.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(kpAddr));
        scores.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scAddr));
    }
};

static __aicore__ inline void LoadWeightAndBiasUb(
    const KeypointGmTensors& gm, const AscendC::LocalTensor<float>& coeff,
    uint32_t weightElemOff, uint32_t weightCount)
{
    if (gm.weightAligned) {
        AscendC::DataCopy(coeff, gm.weight[weightElemOff], weightCount);
    } else {
        AscendC::DataCopyPad(
            coeff, gm.weight[weightElemOff],
            AscendC::DataCopyExtParams{1U, weightCount * 4U, 0U, 0U, 0U},
            AscendC::DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
    }
    AscendC::DataCopyPad(
        coeff[weightCount], gm.bias, AscendC::DataCopyExtParams{1U, 16U, 0U, 0U, 0U},
        AscendC::DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
}

static __aicore__ inline void ReduceSuperGroups72MaxUb(
    const AscendC::LocalTensor<float>& res, const AscendC::LocalTensor<float>& scratch,
    const AscendC::LocalTensor<float>& coeff, uint32_t weightCount, int32_t groups)
{
    const uint32_t alignedG = (static_cast<uint32_t>(groups) + 7U) & ~7U;
    AscendC::Abs(scratch, coeff, static_cast<int32_t>(weightCount));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceMax(res[0U], scratch[0U], 64, groups, 1, 1, 9, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    AscendC::WholeReduceMax(res[alignedG], scratch[64U], 8, groups, 1, 1, 9, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(res[0U], res[0U], res[alignedG], alignedG);
}

static __aicore__ inline void ZeroStatsUb(const AscendC::LocalTensor<float>& statsUb, uint32_t count)
{
    for (uint32_t k = 0U; k < count; ++k) {
        statsUb.SetValue(k, 0.0f);
        statsUb.SetValue(4U + k, 0.0f);
        statsUb.SetValue(8U + k, 0.0f);
        statsUb.SetValue(12U + k, 0.0f);
    }
}

static __aicore__ inline float ClampNegExpArg(float x)
{
    if (x < kNegExpFloor) {
        return kNegExpFloor;
    }
    return (x > 0.0f) ? 0.0f : x;
}

static __aicore__ inline float ScaleDiffScalar(float val, float refMax, uint32_t tempMode, float invTau)
{
    if (tempMode == 0U) {
        return ClampNegExpArg((val - refMax) * invTau);
    }
    if (tempMode == 1U) {
        return ClampNegExpArg(((val - refMax) * kTinyTauPreScale) * invTau);
    }
    return ClampNegExpArg((val * kHugeTauPreScale - refMax * kHugeTauPreScale) * invTau);
}

static __aicore__ inline uint32_t PrepMergeExpsUb(
    const AscendC::LocalTensor<float>& accUb, const AscendC::LocalTensor<float>& curUb,
    uint32_t count, uint32_t tempMode, float invTau, const AscendC::LocalTensor<float>& expUb)
{
    uint32_t mergeMask = 0U;
    for (uint32_t k = 0U; k < count; ++k) {
        float accD = accUb.GetValue(4U + k);
        float curD = curUb.GetValue(4U + k);
        if (accD <= 0.0f) {
            accUb.SetValue(k, curUb.GetValue(k));
            accUb.SetValue(4U + k, curD);
            accUb.SetValue(8U + k, curUb.GetValue(8U + k));
            accUb.SetValue(12U + k, curUb.GetValue(12U + k));
            expUb.SetValue(2U * k, 0.0f);
            expUb.SetValue(2U * k + 1U, 0.0f);
        } else if (curD <= 0.0f) {
            expUb.SetValue(2U * k, 0.0f);
            expUb.SetValue(2U * k + 1U, 0.0f);
        } else {
            mergeMask |= (1U << k);
            float accS = accUb.GetValue(k);
            float curS = curUb.GetValue(k);
            float mNew = (accS > curS) ? accS : curS;
            expUb.SetValue(2U * k, ScaleDiffScalar(accS, mNew, tempMode, invTau));
            expUb.SetValue(2U * k + 1U, ScaleDiffScalar(curS, mNew, tempMode, invTau));
        }
    }
    return mergeMask;
}

static __aicore__ inline void ApplyMergedScalesUb(
    const AscendC::LocalTensor<float>& accUb, const AscendC::LocalTensor<float>& curUb,
    const AscendC::LocalTensor<float>& expUb, uint32_t count, uint32_t mergeMask)
{
    for (uint32_t k = 0U; k < count; ++k) {
        if ((mergeMask & (1U << k)) != 0U) {
            float scA = expUb.GetValue(2U * k);
            float scB = expUb.GetValue(2U * k + 1U);
            float accS = accUb.GetValue(k);
            float curS = curUb.GetValue(k);
            accUb.SetValue(k, (accS > curS) ? accS : curS);
            accUb.SetValue(4U + k, scA * accUb.GetValue(4U + k) + scB * curUb.GetValue(4U + k));
            accUb.SetValue(8U + k, scA * accUb.GetValue(8U + k) + scB * curUb.GetValue(8U + k));
            accUb.SetValue(12U + k, scA * accUb.GetValue(12U + k) + scB * curUb.GetValue(12U + k));
        }
    }
}

static __aicore__ inline void MergeStatsUb(
    const AscendC::LocalTensor<float>& accUb, const AscendC::LocalTensor<float>& curUb,
    uint32_t count, uint32_t tempMode, float invTau, const AscendC::LocalTensor<float>& expUb)
{
    uint32_t mergeMask = PrepMergeExpsUb(accUb, curUb, count, tempMode, invTau, expUb);
    if (mergeMask == 0U) {
        return;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::Exp(expUb, expUb, 8U);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    ApplyMergedScalesUb(accUb, curUb, expUb, count, mergeMask);
}

template <bool IS_MAX>
static __aicore__ inline void WholeRedOp(
    const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
    int32_t maskCnt, int32_t reps)
{
    if constexpr (IS_MAX) {
        AscendC::WholeReduceMax(
            dst, src, maskCnt, reps, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
    } else {
        AscendC::WholeReduceSum(dst, src, maskCnt, reps, 1, 1, 8);
    }
}

template <bool IS_MAX>
static __aicore__ inline void BinaryFoldOp(
    const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src0,
    const AscendC::LocalTensor<float>& src1, uint32_t cnt)
{
    if constexpr (IS_MAX) {
        AscendC::Max(dst, src0, src1, cnt);
    } else {
        AscendC::Add(dst, src0, src1, cnt);
    }
}

template <bool IS_MAX>
static __aicore__ inline uint32_t FoldExtraChunksUb(
    const AscendC::LocalTensor<float>& workUb, uint32_t fullReps)
{
    if (fullReps <= 64U) {
        return fullReps;
    }
    for (uint32_t off = 64U; off < fullReps; off += 64U) {
        uint32_t left = fullReps - off;
        BinaryFoldOp<IS_MAX>(workUb, workUb, workUb[off], (left < 64U) ? left : 64U);
        AscendC::PipeBarrier<PIPE_V>();
    }
    return 64U;
}

template <bool IS_MAX>
static __aicore__ inline void VecReduceImpl(
    const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& workUb, uint32_t count)
{
    if (count <= 64U) {
        WholeRedOp<IS_MAX>(dst, src, static_cast<int32_t>(count), 1);
        AscendC::ResetMask();
        return;
    }
    uint32_t fullReps = count >> 6U;
    uint32_t rem = count & 63U;
    WholeRedOp<IS_MAX>(workUb, src, 64, static_cast<int32_t>(fullReps));
    if (rem > 0U) {
        WholeRedOp<IS_MAX>(workUb[192U], src[fullReps * 64U], static_cast<int32_t>(rem), 1);
    }
    AscendC::PipeBarrier<PIPE_V>();
    uint32_t stage2Cnt = FoldExtraChunksUb<IS_MAX>(workUb, fullReps);
    if (rem == 0U) {
        WholeRedOp<IS_MAX>(dst, workUb, static_cast<int32_t>(stage2Cnt), 1);
    } else {
        WholeRedOp<IS_MAX>(workUb[200U], workUb, static_cast<int32_t>(stage2Cnt), 1);
        AscendC::PipeBarrier<PIPE_V>();
        BinaryFoldOp<IS_MAX>(dst, workUb[200U], workUb[192U], 8U);
    }
    AscendC::ResetMask();
}

static __aicore__ inline void VecReduceMax(
    const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& workUb, uint32_t count)
{
    VecReduceImpl<true>(dst, src, workUb, count);
}

static __aicore__ inline void VecReduceSum(
    const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
    const AscendC::LocalTensor<float>& workUb, uint32_t count)
{
    VecReduceImpl<false>(dst, src, workUb, count);
}

static __aicore__ inline void ExpandDoublingFloatUb(
    const AscendC::LocalTensor<float>& vec, uint32_t seedCnt, uint32_t totalCnt)
{
    for (uint32_t step = seedCnt; step < totalCnt; step <<= 1U) {
        const uint32_t rem = totalCnt - step;
        AscendC::Adds(vec[step], vec, ScalarToFloat(step), (rem < step) ? rem : step);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

static __aicore__ inline void PatchShiftRowBoundsUb(
    const AscendC::LocalTensor<int32_t>& offL, const AscendC::LocalTensor<int32_t>& offR,
    uint32_t rows, uint32_t w, int32_t zeroByteOff)
{
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
    for (uint32_t r = 0U; r < rows; ++r) {
        offL.SetValue(r * w, zeroByteOff);
        offR.SetValue((r + 1U) * w - 1U, zeroByteOff);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
    AscendC::ResetMask();
}

static __aicore__ inline void BuildShiftOffsetsUb(
    const AscendC::LocalTensor<int32_t>& offL, const AscendC::LocalTensor<int32_t>& offR,
    uint32_t rows, uint32_t w, uint32_t outElems, int32_t zeroByteOff)
{
    for (uint32_t i = 0U; i < 8U; ++i) {
        offR.SetValue(i, static_cast<int32_t>(i * 4U));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
    for (uint32_t step = 8U; step < outElems; step <<= 1U) {
        const uint32_t rem = outElems - step;
        const uint32_t cnt = (rem < step) ? rem : step;
        AscendC::Adds(offR[step], offR, static_cast<int32_t>(step * 4U), cnt);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Adds(offL, offR, -4, outElems);
    AscendC::Adds(offR, offR, 4, outElems);
    PatchShiftRowBoundsUb(offL, offR, rows, w, zeroByteOff);
}

static __aicore__ inline void FillCoordVectorsUb(
    const AscendC::LocalTensor<float>& xVec, const AscendC::LocalTensor<float>& yVec,
    uint32_t firstRow, uint32_t rows, uint32_t w)
{
    const float yStart = ScalarToFloat(firstRow);
    for (uint32_t i = 0U; i < 8U; ++i) {
        const float fi = ScalarToFloat(i);
        xVec.SetValue(i, fi);
        yVec.SetValue(i, yStart + fi);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
    ExpandDoublingFloatUb(xVec, 8U, w);
    ExpandDoublingFloatUb(yVec, 8U, rows);
    AscendC::ResetMask();
}

static __aicore__ inline void ScaleAndExpSingleUb(
    const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
    float mx, uint32_t elems, uint32_t tempMode, float invTau)
{
    if (tempMode == 2U) {
        AscendC::Muls(dst, src, kHugeTauPreScale, elems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(dst, dst, 0.0f - (mx * kHugeTauPreScale), elems);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        AscendC::Adds(dst, src, 0.0f - mx, elems);
        AscendC::PipeBarrier<PIPE_V>();
        if (tempMode == 1U) {
            AscendC::Muls(dst, dst, kTinyTauPreScale, elems);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    AscendC::Muls(dst, dst, invTau, elems);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Maxs(dst, dst, kNegExpFloor, elems);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(dst, dst, elems);
    AscendC::PipeBarrier<PIPE_V>();
}

static __aicore__ inline float ReduceMaxAndExpUb(
    const AscendC::LocalTensor<float>& dstExp, const AscendC::LocalTensor<float>& srcAcc,
    const AscendC::LocalTensor<float>& workUb, const AscendC::LocalTensor<float>& sBox,
    uint32_t elems, uint32_t tempMode, float invTau)
{
    VecReduceMax(sBox, srcAcc, workUb, elems);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    const float mx = sBox.GetValue(0U);
    ScaleAndExpSingleUb(dstExp, srcAcc, mx, elems, tempMode, invTau);
    return mx;
}

static __aicore__ inline void StoreSinglePlaneUb(
    const AscendC::GlobalTensor<float>& kpGm, const AscendC::GlobalTensor<float>& scGm,
    uint32_t planeIdx, const AscendC::LocalTensor<float>& kpUb,
    const AscendC::LocalTensor<float>& scUb, float kpX, float kpY, float scVal)
{
    kpUb.SetValue(0U, kpX);
    kpUb.SetValue(1U, kpY);
    scUb.SetValue(0U, scVal);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyPad(
        kpGm[static_cast<uint64_t>(planeIdx) * 2U], kpUb,
        AscendC::DataCopyExtParams{1U, 8U, 0U, 0U, 0U});
    AscendC::DataCopyPad(
        scGm[planeIdx], scUb,
        AscendC::DataCopyExtParams{1U, 4U, 0U, 0U, 0U});
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

static __aicore__ inline void StoreStatsPlaneUb(
    const AscendC::GlobalTensor<float>& kpGm, const AscendC::GlobalTensor<float>& scGm,
    uint32_t planeIdx, uint32_t outOff, const AscendC::LocalTensor<float>& accStatsUb,
    float biasVal, const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
{
    auto kpOne = UbTensor<float>(outOff, 8U);
    auto scOne = UbTensor<float>(outOff + 64U, 8U);
    float dVal = accStatsUb.GetValue(4U);
    float kpX = SafeDiv(accStatsUb.GetValue(8U), dVal) * t.strideF + t.halfStrideF;
    float kpY = SafeDiv(accStatsUb.GetValue(12U), dVal) * t.strideF + t.halfStrideF;
    StoreSinglePlaneUb(kpGm, scGm, planeIdx, kpOne, scOne, kpX, kpY, accStatsUb.GetValue(0U) + biasVal);
}

}

#endif
