// Copyright (c) 2026 JLU_op-wuliuqi.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

template <typename T>
struct SbgRangeViews {
    AscendC::LocalTensor<uint8_t> compare;
    AscendC::LocalTensor<T> one;
    AscendC::LocalTensor<T> zero;
    AscendC::LocalTensor<float> pairs;
    AscendC::LocalTensor<float> y1;
    AscendC::LocalTensor<float> y2;
    AscendC::LocalTensor<float> difference;
    AscendC::LocalTensor<float> boundary;
    AscendC::LocalTensor<float> cy;
};


template <typename T>
class SbgRangeFields {
protected:
    uint32_t rowStart = 0U, coreRows = 0U, movedRows = 0U, pad = 0U;
    uint32_t hasThreshold = 0U;
    GM_ADDR threshold = nullptr;
    GM_ADDR tiling = nullptr;
    float waistRatio = 0.0f, thresholdValue = 0.0f;
    __gm__ T *boxesBase = nullptr, *anchorsBase = nullptr, *outBase = nullptr;
    AscendC::LocalTensor<T> boxIn, ancIn, outUb, cst;
    AscendC::LocalTensor<float> sv;
};

template <typename T>
class SbgRangeStorage : protected SbgRangeFields<T> {};

template <>
class SbgRangeStorage<half> : protected SbgRangeFields<half> {
protected:
    AscendC::LocalTensor<float> boxF32, ancF32;
};
template <typename T>
class SbgRangeState : protected SbgRangeStorage<T> {
protected:
    static constexpr uint32_t tSize = static_cast<uint32_t>(sizeof(T));
    static constexpr uint32_t bankGap = 0U;
    using Base = SbgRangeStorage<T>;
    using Base::rowStart;
    using Base::coreRows;
    using Base::movedRows;
    using Base::pad;
    using Base::hasThreshold;
    using Base::threshold;
    using Base::tiling;
    using Base::waistRatio;
    using Base::thresholdValue;
    using Base::boxesBase;
    using Base::anchorsBase;
    using Base::outBase;
    using Base::boxIn;
    using Base::ancIn;
    using Base::outUb;
    using Base::cst;
    using Base::sv;

    template <uint32_t GAP = 0U, bool FIXED_SCRATCH = false>
    __aicore__ inline void InitRangeStorage() {
        uint32_t offset = 0U;
        boxIn = AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, offset, pad * 4U);
        offset += SbgAlignBytes(pad * 4U * tSize);
        ancIn = AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, offset, pad * 2U);
        offset += SbgAlignBytes(pad * 2U * tSize);
        outUb = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC, offset, pad);
        offset += SbgAlignBytes(pad * tSize);
        if constexpr (sizeof(T) == 2U) {
            this->boxF32 = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, offset, pad * 4U);
            offset += SbgAlignBytes(pad * 4U * 4U);
            this->ancF32 = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, offset, pad * 2U);
            offset += SbgAlignBytes(pad * 2U * 4U);
        }
        cst = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC, offset, pad * 2U);
        offset += SbgAlignBytes(pad * tSize * 2U);
        const uint32_t scratchOffset = FIXED_SCRATCH && GAP ? 65536U : offset;
        sv = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, scratchOffset,
            7U * pad + 4U * GAP + SbgAlignBytes(pad) / 4U + 8U);
    }

    __aicore__ inline void PreloadRangeThreshold() {
        if constexpr (sizeof(T) == 2U) {
            if (hasThreshold != 0U && threshold != nullptr) {
                AscendC::GlobalTensor<uint64_t> thresholdCacheLine;
                thresholdCacheLine.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(threshold), 1);
                AscendC::DataCachePreload(thresholdCacheLine, static_cast<int16_t>(0));
            }
        }
    }

    template <uint32_t MIN_ROWS, uint32_t MAX_ROWS, bool SMALL>
    __aicore__ inline bool InitRange(GM_ADDR person_boxes, GM_ADDR anchor_points,
        GM_ADDR thresholdArg, GM_ADDR violation_score, uint32_t length,
        uint32_t rowsPerCore, uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        threshold = thresholdArg;
        hasThreshold = hasThresholdArg;
        tiling = tilingArg;
        if (length < MIN_ROWS || length > MAX_ROWS) return false;
        rowStart = static_cast<uint32_t>(AscendC::GetBlockIdx()) * rowsPerCore;
        if (rowStart >= length) return false;
        coreRows = length - rowStart;
        if (coreRows > rowsPerCore) coreRows = rowsPerCore;
        PreloadRangeThreshold();
        movedRows = SbgAlignUp(coreRows, static_cast<uint32_t>(kSbgRowAlign));
        pad = SMALL ? 64U : SbgAlignUp(coreRows, 64U);
        InitRangeStorage();
        boxesBase = reinterpret_cast<__gm__ T*>(person_boxes);
        anchorsBase = reinterpret_cast<__gm__ T*>(anchor_points);
        outBase = reinterpret_cast<__gm__ T*>(violation_score);
        return true;
    }

    template <bool UNIT_RATIO = false, uint32_t RATIO_KIND = 0U>
    __aicore__ inline void CopyInRange() {
        AscendC::GlobalTensor<T> boxGm;
        boxGm.SetGlobalBuffer(boxesBase + static_cast<int64_t>(rowStart) * 4,
            SbgAlignBytes(coreRows * 4U * sizeof(T)) / sizeof(T));
        const uint32_t boxBytes = SbgAlignBytes(coreRows * 4U * tSize);
        AscendC::DataCopyParams boxDp{1, static_cast<uint16_t>(boxBytes / 32U), 0, 0};
        AscendC::DataCopy(boxIn, boxGm, boxDp);
        AscendC::GlobalTensor<T> ancGm;
        ancGm.SetGlobalBuffer(anchorsBase + static_cast<int64_t>(rowStart) * 2,
            SbgAlignBytes(coreRows * 2U * sizeof(T)) / sizeof(T));
        const uint32_t anchorBytes = SbgAlignBytes(coreRows * 2U * tSize);
        AscendC::DataCopyParams ancDp{1, static_cast<uint16_t>(anchorBytes / 32U), 0, 0};
        AscendC::DataCopy(ancIn, ancGm, ancDp);
        waistRatio = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
            RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
            reinterpret_cast<__gm__ volatile float*>(tiling)[3];
        AscendC::InitSocState();
        thresholdValue = 0.0f;
        if constexpr (sizeof(T) == 4U) {
            if (hasThreshold != 0U && threshold != nullptr) {
                thresholdValue = static_cast<float>(reinterpret_cast<__gm__ T*>(threshold)[0]);
            }
        }
    }

    template <uint32_t GAP = 0U>
    __aicore__ inline SbgRangeViews<T> MakeRangeViews() {
        return {
            sv.template ReinterpretCast<uint8_t>()[(7U * pad + 4U * GAP) * sizeof(float)],
            cst[0], cst[pad], sv[0], sv[2U * pad], sv[3U * pad + GAP],
            sv[4U * pad + GAP], sv[5U * pad + 2U * GAP], sv[6U * pad + 3U * GAP]
        };
    }

    __aicore__ inline void ExtractRangeColumns(const SbgRangeViews<T> &views) {
        if constexpr (sizeof(T) == 4U) {
            const uint64_t maskOdd[1] = {kSbgMaskOdd};
            const uint64_t maskEven[1] = {kSbgMaskEven};
            const int32_t rep4 = static_cast<int32_t>(4U * pad / kSbgPerRepeat);
            const int32_t rep2 = static_cast<int32_t>(2U * pad / kSbgPerRepeat);
            AscendC::PairReduceSum<float>(views.pairs, boxIn.template ReinterpretCast<float>(), rep4, maskOdd, 1, 1, 8);
            AscendC::PairReduceSum<float>(views.y1, views.pairs, rep2, maskEven, 1, 1, 8);
            AscendC::PairReduceSum<float>(views.y2, views.pairs, rep2, maskOdd, 1, 1, 8);
            AscendC::PairReduceSum<float>(views.cy, ancIn.template ReinterpretCast<float>(), rep2, maskOdd, 1, 1, 8);
        } else {
            AscendC::SetMaskNorm();
            AscendC::ResetMask();
            const uint16_t boxReps = static_cast<uint16_t>(4U * pad * sizeof(T) / 256U);
            const uint16_t ancReps = static_cast<uint16_t>(2U * pad * sizeof(T) / 256U);
            AscendC::LocalTensor<T> y1Half = this->boxF32.template ReinterpretCast<T>();
            AscendC::LocalTensor<T> y2Half = y1Half[pad];
            AscendC::LocalTensor<T> cyHalf = y1Half[2U * pad];
            AscendC::GatherMask(y1Half, boxIn, static_cast<uint8_t>(4), {1, boxReps, 8, 0});
            AscendC::GatherMask(y2Half, boxIn, static_cast<uint8_t>(6), {1, boxReps, 8, 0});
            AscendC::GatherMask(cyHalf, ancIn, static_cast<uint8_t>(2), {1, ancReps, 8, 0});
            AscendC::Cast(views.y1, y1Half, AscendC::RoundMode::CAST_NONE, pad);
            AscendC::Cast(views.y2, y2Half, AscendC::RoundMode::CAST_NONE, pad);
            AscendC::Cast(views.cy, cyHalf, AscendC::RoundMode::CAST_NONE, pad);
        }
    }

    template <bool UNIT_RATIO = false>
    __aicore__ inline void ComputeRangeBoundary(const SbgRangeViews<T> &views) {
        const int32_t count = static_cast<int32_t>(pad);
        AscendC::Sub(views.difference, views.y2, views.y1, count);
        if constexpr (!UNIT_RATIO) AscendC::Muls(views.difference, views.difference, waistRatio, count);
        AscendC::Add(views.boundary, views.y1, views.difference, count);
        if constexpr (sizeof(T) == 2U) {
            if (hasThreshold != 0U && threshold != nullptr) {
                thresholdValue = static_cast<float>(reinterpret_cast<__gm__ volatile T*>(threshold)[0]);
            }
        }
        AscendC::Adds(views.boundary, views.boundary, thresholdValue, count);
    }

    __aicore__ inline void ComputeRange() {
        const SbgRangeViews<T> views = MakeRangeViews();
        AscendC::Duplicate(views.one, static_cast<T>(1.0f), static_cast<int32_t>(pad));
        AscendC::Duplicate(views.zero, static_cast<T>(0.0f), static_cast<int32_t>(pad));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        ExtractRangeColumns(views);
        ComputeRangeBoundary(views);
        AscendC::Compare(views.compare, views.cy, views.boundary, AscendC::CMPMODE::GT, static_cast<int32_t>(pad));
        AscendC::Select(outUb, views.compare, views.one, views.zero,
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<int32_t>(pad));
    }

    template <uint32_t CONST_B = 0U, uint32_t CONST_PER = 1U>
    __aicore__ inline void CopyOutRange() {
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::GlobalTensor<T> outGm;
    outGm.SetGlobalBuffer(outBase + rowStart, SbgAlignBytes(coreRows * sizeof(T)) / sizeof(T));
    const uint32_t outBytes = coreRows * tSize;
    if constexpr (CONST_B > 0 && CONST_B % CONST_PER != 0) {
        AscendC::DataCopyParams cp{1, static_cast<uint16_t>(SbgAlignBytes(outBytes) / 32U), 0, 0};
        AscendC::DataCopy(outGm, outUb, cp);
    } else if ((outBytes & 31U) == 0U) {
        AscendC::DataCopyParams outDp;
        outDp.blockCount = 1;
        outDp.blockLen = static_cast<uint16_t>(outBytes / 32U);
        outDp.srcStride = 0;
        outDp.dstStride = 0;
        AscendC::DataCopy(outGm, outUb, outDp);
    } else {
        AscendC::DataCopyParams outCp;
        outCp.blockCount = 1;
        outCp.blockLen = static_cast<uint16_t>(SbgAlignBytes(outBytes) / 32U);
        outCp.srcStride = 0;
        outCp.dstStride = 0;
        AscendC::DataCopy(outGm, outUb, outCp);
    }
    }
};


using SbgHalfRangeState = SbgRangeState<half>;
template <typename T>
class SbgSmallRange final : public SbgStages<SbgSmallRange<T>>, protected SbgRangeState<T> {
    friend class SbgStages<SbgSmallRange<T>>;
    using Base = SbgRangeState<T>;

    __aicore__ inline bool Init(GM_ADDR boxes, GM_ADDR anchors, GM_ADDR threshold, GM_ADDR output,
        uint32_t rows, uint32_t per, uint32_t present, GM_ADDR tiling) {
        return Base::template InitRange<1U, 31U, true>(boxes, anchors, threshold, output,
            rows, per, present, tiling);
    }
    __aicore__ inline void CopyIn() { Base::CopyInRange(); }
    __aicore__ inline void Compute() { Base::ComputeRange(); }
    __aicore__ inline void CopyOut() { Base::CopyOutRange(); }
};
