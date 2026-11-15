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

template <typename T, uint32_t CONST_B, uint32_t CONST_PER, bool UNIT_RATIO, uint32_t RATIO_KIND>
class SbgFallback final :
    public SbgStages<SbgFallback<T, CONST_B, CONST_PER, UNIT_RATIO, RATIO_KIND>>,
    protected SbgRangeState<T> {
    friend class SbgStages<SbgFallback<T, CONST_B, CONST_PER, UNIT_RATIO, RATIO_KIND>>;
    using Base = SbgRangeState<T>;
    using Base::rowStart;
    using Base::coreRows;
    using Base::movedRows;
    using Base::pad;
    using Base::threshold;
    using Base::hasThreshold;
    using Base::tiling;
    using Base::waistRatio;
    using Base::thresholdValue;
    using Base::boxesBase;
    using Base::anchorsBase;
    using Base::outBase;
    using Base::boxIn;
    using Base::ancIn;
    using Base::outUb;
    using Base::sv;
    static constexpr uint32_t bankGap =
        CONST_B > 0U && CONST_PER > 64U && (((CONST_PER + 63U) / 64U * 64U) % 128U) == 0U ? 64U : 0U;
    SbgRangeViews<T> views_;

    __aicore__ inline bool Init(GM_ADDR person_boxes, GM_ADDR anchor_points, GM_ADDR thresholdArg,
        GM_ADDR violation_score, uint32_t length, uint32_t rowsPerCore,
        uint32_t hasThresholdArg, GM_ADDR tilingArg) {
        threshold = thresholdArg;
        hasThreshold = hasThresholdArg;
        tiling = tilingArg;
        const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
        constexpr bool single33 = sizeof(T) == 2U && CONST_B == 33U && CONST_PER >= CONST_B;
        rowStart = single33 ? 0U : block * rowsPerCore;
        if constexpr (single33) {
            if (block != 0U) return false;
        } else {
            if (rowStart >= length) return false;
        }
        if constexpr (single33) {
            coreRows = CONST_B;
        } else if constexpr (CONST_B > 0U && CONST_B % CONST_PER == 0U) {
            coreRows = CONST_PER;
        } else {
            coreRows = length - rowStart;
            if (coreRows > rowsPerCore) coreRows = rowsPerCore;
        }
        Base::PreloadRangeThreshold();
        movedRows = SbgAlignUp(coreRows, static_cast<uint32_t>(kSbgRowAlign));
        pad = CONST_B > 0U ? (CONST_PER + 63U) / 64U * 64U : SbgAlignUp(coreRows, kSbgPadMin);
        Base::template InitRangeStorage<bankGap, true>();
        boxesBase = reinterpret_cast<__gm__ T*>(person_boxes);
        anchorsBase = reinterpret_cast<__gm__ T*>(anchor_points);
        outBase = reinterpret_cast<__gm__ T*>(violation_score);
        return true;
    }

    __aicore__ inline void CopyIn() {
        Base::template CopyInRange<UNIT_RATIO, RATIO_KIND>();
    }

    __aicore__ inline void PrepareCompareAddress() {
        if constexpr (CONST_B > 0U && CONST_PER > 64U) {
            if ASCEND_IS_AIV {
                AscendC::LocalTensor<uint32_t> payload = sv.template ReinterpretCast<uint32_t>()[
                    7U * pad + 4U * bankGap + SbgAlignBytes(pad) / sizeof(uint32_t)];
                auto address = reinterpret_cast<__ubuf__ uint32_t*>(payload.GetPhyAddr());
                const uint32_t compareAddress = static_cast<uint32_t>(
                    reinterpret_cast<uint64_t>(reinterpret_cast<__ubuf__ uint8_t*>(views_.compare.GetPhyAddr())));
                set_mask_norm();
                set_vector_mask(0ULL, 0xffULL);
                vector_dup(address, compareAddress, 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                set_cmpmask(address);
                pipe_barrier(PIPE_V);
                set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            }
        }
    }

    __aicore__ inline void ExtractNativeColumns() {
        if ASCEND_IS_AIV {
            auto pairs = reinterpret_cast<__ubuf__ float*>(views_.pairs.GetPhyAddr());
            auto boxes = reinterpret_cast<__ubuf__ float*>(boxIn.GetPhyAddr());
            auto anchors = reinterpret_cast<__ubuf__ float*>(ancIn.GetPhyAddr());
            auto y1 = reinterpret_cast<__ubuf__ float*>(views_.y1.GetPhyAddr());
            auto y2 = reinterpret_cast<__ubuf__ float*>(views_.y2.GetPhyAddr());
            auto cy = reinterpret_cast<__ubuf__ float*>(views_.cy.GetPhyAddr());
            const int32_t rep4 = static_cast<int32_t>(4U * pad / kSbgPerRepeat);
            const int32_t rep2 = static_cast<int32_t>(2U * pad / kSbgPerRepeat);
            set_mask_norm();
            set_vector_mask(0ULL, kSbgMaskOdd);
            vcpadd(pairs, boxes, rep4, 1, 1, 8);
            vcpadd(cy, anchors, rep2, 1, 1, 8);
            pipe_barrier(PIPE_V);
            vcpadd(y2, pairs, rep2, 1, 1, 8);
            set_vector_mask(0ULL, kSbgMaskEven);
            vcpadd(y1, pairs, rep2, 1, 1, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
        }
    }

    __aicore__ inline void ComputeNativeBoundary() {
        if ASCEND_IS_AIV {
            auto difference = reinterpret_cast<__ubuf__ float*>(views_.difference.GetPhyAddr());
            auto y1 = reinterpret_cast<__ubuf__ float*>(views_.y1.GetPhyAddr());
            auto y2 = reinterpret_cast<__ubuf__ float*>(views_.y2.GetPhyAddr());
            auto boundary = reinterpret_cast<__ubuf__ float*>(views_.boundary.GetPhyAddr());
            const uint8_t repeats = static_cast<uint8_t>(pad / 64U);
            vsub(difference, y2, y1, repeats, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if constexpr (!UNIT_RATIO) {
                vmuls(difference, difference, static_cast<float>(waistRatio), repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            vadd(boundary, y1, difference, repeats, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vadds(boundary, boundary, thresholdValue, repeats, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void SelectNative() {
        if ASCEND_IS_AIV {
            auto mask = reinterpret_cast<__ubuf__ uint8_t*>(views_.compare.GetPhyAddr());
            auto cy = reinterpret_cast<__ubuf__ float*>(views_.cy.GetPhyAddr());
            auto boundary = reinterpret_cast<__ubuf__ float*>(views_.boundary.GetPhyAddr());
            auto output = reinterpret_cast<__ubuf__ T*>(outUb.GetPhyAddr());
            auto one = reinterpret_cast<__ubuf__ T*>(views_.one.GetPhyAddr());
            auto zero = reinterpret_cast<__ubuf__ T*>(views_.zero.GetPhyAddr());
            constexpr uint32_t padded = (CONST_PER + 63U) / 64U * 64U;
            constexpr uint8_t repeats = static_cast<uint8_t>(padded / 64U);
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            if constexpr (sizeof(T) == 2U) pipe_barrier(PIPE_V);
            vcmpv_gt(mask, cy, boundary, repeats, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if constexpr (sizeof(T) == 4U) {
                vsel(output, one, zero, repeats, 1, 1, 1, 8, 8, 8,
                    static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE));
            } else {
                set_mask_count();
                set_vector_mask(0ULL, static_cast<uint64_t>(padded));
                vsel(output, one, zero, 1, 1, 1, 1, 8, 8, 8,
                    static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE));
            }
            pipe_barrier(PIPE_V);
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
    }

    __aicore__ inline void SelectResult() {
        if constexpr (CONST_B > 0U && CONST_PER <= 64U) {
            AscendC::SetMaskNorm();
            AscendC::Compare(views_.cy, views_.boundary, AscendC::CMPMODE::GT, uint64_t(64), {1, 1, 1, 8, 8, 8});
            AscendC::Select<T, AscendC::SELMODE::VSEL_CMPMASK_SPR>(outUb, views_.one, views_.zero, 1, {1, 1, 1, 8, 8, 8});
        } else if constexpr (CONST_B > 0U) {
            SelectNative();
        } else {
            AscendC::Compare(views_.compare, views_.cy, views_.boundary, AscendC::CMPMODE::GT, static_cast<int32_t>(pad));
            AscendC::Select(outUb, views_.compare, views_.one, views_.zero,
                AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<int32_t>(pad));
        }
    }

    __aicore__ inline void Compute() {
        views_ = Base::template MakeRangeViews<bankGap>();
        AscendC::Duplicate(views_.one, static_cast<T>(1.0f), static_cast<int32_t>(pad));
        AscendC::Duplicate(views_.zero, static_cast<T>(0.0f), static_cast<int32_t>(pad));
        PrepareCompareAddress();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if constexpr (sizeof(T) == 4U && CONST_B > 0U) {
            ExtractNativeColumns();
            ComputeNativeBoundary();
        } else {
            Base::ExtractRangeColumns(views_);
            Base::template ComputeRangeBoundary<UNIT_RATIO>(views_);
        }
        SelectResult();
    }

    __aicore__ inline void CopyOut() {
        Base::template CopyOutRange<CONST_B, CONST_PER>();
    }
};
