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

template <uint32_t BATCH, bool FP16, uint32_t WORKING, uint32_t PER, uint32_t UB_BYTES>
__aicore__ constexpr bool SbgValidatePointPolicy() {
    static_assert(SbgShapePolicy(BATCH, FP16).launch == 8U, "Host launch policy changed");
    static_assert(SbgShapePolicy(BATCH, FP16).working == WORKING, "Host working-core policy changed");
    static_assert(UB_BYTES <= 192U * 1024U, "Point UB allocation exceeds 192 KiB");
    if constexpr (FP16 && BATCH == 4096U) {
        static_assert(BATCH == 5U * PER + 256U, "Exact 512B-aligned partition");
    } else {
        static_assert(((BATCH + WORKING - 1U) / WORKING + 15U) / 16U * 16U == PER,
            "Host/device slice mismatch");
    }
    return true;
}

template <uint32_t FIXED, uint32_t FIRST, uint32_t LAST, uint32_t UB_BYTES>
__aicore__ constexpr bool SbgValidateIntervalPolicy() {
    static_assert(FIXED == 0U || FIXED == FIRST || FIXED == LAST, "Unsupported constant interval plan");
    static_assert(UB_BYTES <= 192U * 1024U, "UB budget exceeds 192 KiB");
    return true;
}

// 统一四阶段调度，具体类继续提供 Init、CopyIn、Compute、CopyOut。
template <typename Kernel>
class SbgStages {
public:
    template <typename... Args>
    __aicore__ inline void Run(Args... args) {
        Kernel &kernel = static_cast<Kernel &>(*this);
        if (!kernel.Init(args...)) return;
        kernel.CopyIn();
        kernel.Compute();
        kernel.CopyOut();
    }
};



struct SbgPointColumns {
    AscendC::LocalTensor<float> y1;
    AscendC::LocalTensor<float> y2;
    AscendC::LocalTensor<float> difference;
    AscendC::LocalTensor<float> boundary;
    AscendC::LocalTensor<float> cy;
};

template <typename T>
struct SbgHalfColumns {
    AscendC::LocalTensor<T> y1;
    AscendC::LocalTensor<T> y2;
    AscendC::LocalTensor<T> cy;
};

// 共用成员保持原来的声明顺序；地址与长度仍由各类的 Init 显式设置。
template <typename T>
class SbgPointState {
protected:
    uint32_t rowStart_ = 0U;
    uint32_t coreRows_ = 0U;
    uint32_t hasThreshold_ = 0U;
    GM_ADDR threshold_ = nullptr;
    GM_ADDR tiling_ = nullptr;
    float waistRatio_ = 0.0f;
    float thresholdValue_ = 0.0f;
    AscendC::GlobalTensor<T> boxesGm_;
    AscendC::GlobalTensor<T> anchorsGm_;
    AscendC::GlobalTensor<T> outputGm_;
    AscendC::LocalTensor<T> boxIn_;
    AscendC::LocalTensor<T> anchorIn_;
    AscendC::LocalTensor<T> output_;
    AscendC::LocalTensor<T> constants_;
    AscendC::LocalTensor<T> one_;
    AscendC::LocalTensor<T> zero_;
    AscendC::LocalTensor<float> scratch_;
    AscendC::LocalTensor<uint8_t> compare_;

    __aicore__ inline void LoadCompareAddressPayload(AscendC::LocalTensor<uint32_t> payload) {
        auto address = reinterpret_cast<__ubuf__ uint32_t*>(payload.GetPhyAddr());
        const uint32_t compareAddress = static_cast<uint32_t>(
            reinterpret_cast<uint64_t>(reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr())));
        set_mask_norm();
        set_vector_mask(0ULL, 0xffULL);
        vector_dup(address, compareAddress, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        set_cmpmask(address);
        pipe_barrier(PIPE_V);
        set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
    }

    template <bool LEADING_BARRIER>
    __aicore__ inline void CompareNativeColumns(AscendC::LocalTensor<float> cy,
        AscendC::LocalTensor<float> boundary, uint32_t repeats) {
        auto mask = reinterpret_cast<__ubuf__ uint8_t*>(compare_.GetPhyAddr());
        auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
        auto boundaryPtr = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
        set_mask_norm();
        set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        if constexpr (LEADING_BARRIER) pipe_barrier(PIPE_V);
        vcmpv_gt(mask, cyPtr, boundaryPtr, repeats, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }

    template <uint32_t SELECT_REPEATS>
    __aicore__ inline void CompareAndSelectCountedImpl(AscendC::LocalTensor<float> cy,
        AscendC::LocalTensor<float> boundary, uint32_t rows, uint32_t compareRepeats) {
        if ASCEND_IS_AIV {
            auto output = reinterpret_cast<__ubuf__ T*>(output_.GetPhyAddr());
            auto one = reinterpret_cast<__ubuf__ T*>(one_.GetPhyAddr());
            auto zero = reinterpret_cast<__ubuf__ T*>(zero_.GetPhyAddr());
            CompareNativeColumns<true>(cy, boundary, compareRepeats);
            set_mask_count();
            set_vector_mask(0ULL, static_cast<uint64_t>(rows));
            vsel(output, one, zero, SELECT_REPEATS, 1, 1, 1, 8, 8, 8,
                static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE));
            pipe_barrier(PIPE_V);
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
    }

    template <uint32_t BOX_REPEATS, uint32_t COLUMN_REPEATS, bool INCLUDE_ANCHOR = true>
    __aicore__ inline void ExtractFp32AllColumns(AscendC::LocalTensor<float> pairs, AscendC::LocalTensor<float> y1, AscendC::LocalTensor<float> y2, AscendC::LocalTensor<float> cy) {
        if ASCEND_IS_AIV {
            auto pairsPtr = reinterpret_cast<__ubuf__ float*>(pairs.GetPhyAddr());
            auto boxesPtr = reinterpret_cast<__ubuf__ float*>(boxIn_.GetPhyAddr());
            auto anchorsPtr = reinterpret_cast<__ubuf__ float*>(anchorIn_.GetPhyAddr());
            auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
            auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
            auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0ULL, 0xAAAAAAAAAAAAAAAAULL);
            vcpadd(pairsPtr, boxesPtr, BOX_REPEATS, 1, 1, 8);
            if constexpr (INCLUDE_ANCHOR) vcpadd(cyPtr, anchorsPtr, COLUMN_REPEATS, 1, 1, 8);
            pipe_barrier(PIPE_V);
            vcpadd(y2Ptr, pairsPtr, COLUMN_REPEATS, 1, 1, 8);
            set_vector_mask(0ULL, 0x5555555555555555ULL);
            vcpadd(y1Ptr, pairsPtr, COLUMN_REPEATS, 1, 1, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
        }
    }

    template <uint32_t CALC_REPEATS, bool UNIT_RATIO, bool READ_THRESHOLD = false>
    __aicore__ inline void ComputeNativeBoundary(AscendC::LocalTensor<float> difference, AscendC::LocalTensor<float> y1, AscendC::LocalTensor<float> y2, AscendC::LocalTensor<float> boundary) {
        if ASCEND_IS_AIV {
            auto d = reinterpret_cast<__ubuf__ float*>(difference.GetPhyAddr());
            auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
            auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
            auto bound = reinterpret_cast<__ubuf__ float*>(boundary.GetPhyAddr());
        // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
        // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
            vsub(d, y2Ptr, y1Ptr, CALC_REPEATS, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if constexpr (!UNIT_RATIO) {
                vmuls(d, d, waistRatio_, CALC_REPEATS, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
            vadd(bound, y1Ptr, d, CALC_REPEATS, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            if constexpr (READ_THRESHOLD) {
                if (hasThreshold_ != 0U && threshold_ != nullptr) {
                    thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ volatile T*>(threshold_)[0]);
                }
            }
            vadds(bound, bound, thresholdValue_, CALC_REPEATS, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
    }

    template <uint32_t CALC_REPEATS, bool UNIT_RATIO>
    __aicore__ inline void ComputeNativeBoundaryMasked(AscendC::LocalTensor<float> difference, AscendC::LocalTensor<float> y1, AscendC::LocalTensor<float> y2, AscendC::LocalTensor<float> boundary) {
        if ASCEND_IS_AIV {
            set_mask_norm();
            set_vector_mask(0ULL, (1ULL << coreRows_) - 1ULL);
            ComputeNativeBoundary<CALC_REPEATS, UNIT_RATIO>(difference, y1, y2, boundary);
        }
    }

    template <uint32_t ADDRESS_OFFSET>
    __aicore__ inline void PrepareCompareAddress() {
        if ASCEND_IS_AIV {
            LoadCompareAddressPayload(scratch_.template ReinterpretCast<uint32_t>()[ADDRESS_OFFSET]);
        }
    }

    template <uint32_t COMPARE_REPEATS, uint32_t SELECT_REPEATS, bool TRAILING_BARRIER = false, bool CONSTANT_SOURCES = false>
    __aicore__ inline void CompareAndSelectNative(AscendC::LocalTensor<float> cy, AscendC::LocalTensor<float> boundary) {
        if ASCEND_IS_AIV {
            auto output = reinterpret_cast<__ubuf__ T*>(output_.GetPhyAddr());
            auto one = reinterpret_cast<__ubuf__ T*>(one_.GetPhyAddr());
            auto zero = reinterpret_cast<__ubuf__ T*>(zero_.GetPhyAddr());
            CompareNativeColumns<false>(cy, boundary, COMPARE_REPEATS);
            if constexpr (CONSTANT_SOURCES) {
                vsel(output, one, zero, SELECT_REPEATS, 1, 0, 0, 8, 0, 0,
                    static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE));
            } else {
                vsel(output, one, zero, SELECT_REPEATS, 1, 1, 1, 8, 8, 8,
                    static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE));
            }
            if constexpr (TRAILING_BARRIER) pipe_barrier(PIPE_V);
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
    }

    template <uint32_t COMPARE_REPEATS, uint32_t SELECT_REPEATS>
    __aicore__ inline void CompareAndSelectNativeFp32Range257To4095(AscendC::LocalTensor<float> cy, AscendC::LocalTensor<float> boundary) { CompareAndSelectNative<COMPARE_REPEATS, SELECT_REPEATS, true>(cy, boundary); }

    template <uint32_t PAD, uint32_t BANK_GAP>
    __aicore__ inline void PrepareCompareAddressFp32Range257To4095() { PrepareCompareAddress<7U * PAD + 3U * BANK_GAP + 48U>(); }

    template <uint32_t PAD>
    __aicore__ inline void ExtractFp32BoxColumns(AscendC::LocalTensor<float> pairs, AscendC::LocalTensor<float> y1, AscendC::LocalTensor<float> y2, AscendC::LocalTensor<float> cy) { ExtractFp32AllColumns<PAD / 16U, PAD / 32U, false>(pairs, y1, y2, cy); }

    template <uint32_t PAD, bool UNIT_RATIO>
    __aicore__ inline void ComputeNativeBoundaryFp32Range257To4095(AscendC::LocalTensor<float> difference,
        AscendC::LocalTensor<float> y1, AscendC::LocalTensor<float> y2, AscendC::LocalTensor<float> boundary) {
        ComputeNativeBoundary<static_cast<uint8_t>(PAD / 64U), UNIT_RATIO>(difference, y1, y2, boundary);
    }

    template <uint32_t PAD>
    __aicore__ inline void ExtractFp32AnchorColumn(AscendC::LocalTensor<float> cy) {
        if ASCEND_IS_AIV {
            auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
            auto anchorsPtr = reinterpret_cast<__ubuf__ float*>(anchorIn_.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0ULL, 0xAAAAAAAAAAAAAAAAULL);
            vcpadd(cyPtr, anchorsPtr, static_cast<uint8_t>(PAD / 32U), 1, 1, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
    }

    template <uint32_t PAD>
    __aicore__ inline void CompareAndSelectNativeFp32Range257To4095Part2(AscendC::LocalTensor<float> cy, AscendC::LocalTensor<float> boundary) { CompareAndSelectNative<PAD / 64U, PAD / 64U, true, true>(cy, boundary); }

    template <uint32_t BOX_REPEATS, uint32_t COLUMN_REPEATS>
    __aicore__ inline void ExtractFp32BoxColumnsFp32B4096(AscendC::LocalTensor<float> pairs, AscendC::LocalTensor<float> y1, AscendC::LocalTensor<float> y2, AscendC::LocalTensor<float> cy) { ExtractFp32AllColumns<BOX_REPEATS, COLUMN_REPEATS, false>(pairs, y1, y2, cy); }

    template <uint32_t COLUMN_REPEATS>
    __aicore__ inline void ExtractFp32AnchorColumnFp32B4096(AscendC::LocalTensor<float> cy) {
        if ASCEND_IS_AIV {
            auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
            auto anchorsPtr = reinterpret_cast<__ubuf__ float*>(anchorIn_.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0ULL, 0xAAAAAAAAAAAAAAAAULL);
            vcpadd(cyPtr, anchorsPtr, COLUMN_REPEATS, 1, 1, 8);
            pipe_barrier(PIPE_V);
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
        }
    }

    template <uint32_t CONVERT_REPEATS>
    __aicore__ inline void ConvertHalfColumns(AscendC::LocalTensor<float> y1, AscendC::LocalTensor<float> y2, AscendC::LocalTensor<float> cy, AscendC::LocalTensor<T> y1Half, AscendC::LocalTensor<T> y2Half, AscendC::LocalTensor<T> cyHalf) {
        if ASCEND_IS_AIV {
            auto y1Ptr = reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr());
            auto y2Ptr = reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr());
            auto cyPtr = reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr());
            auto y1HalfPtr = reinterpret_cast<__ubuf__ T*>(y1Half.GetPhyAddr());
            auto y2HalfPtr = reinterpret_cast<__ubuf__ T*>(y2Half.GetPhyAddr());
            auto cyHalfPtr = reinterpret_cast<__ubuf__ T*>(cyHalf.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
            pipe_barrier(PIPE_V);  // all GatherMask writes must be complete
            // 三次原生half→FP32转换；后面的几何运算和比较不使用half精度。
            vconv_f162f32(y1Ptr, y1HalfPtr, CONVERT_REPEATS, 1, 1, 8, 4);
            vconv_f162f32(y2Ptr, y2HalfPtr, CONVERT_REPEATS, 1, 1, 8, 4);
            vconv_f162f32(cyPtr, cyHalfPtr, CONVERT_REPEATS, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
        }
    }

    template <uint32_t CALC_REPEATS, bool UNIT_RATIO>
    __aicore__ inline void ComputeNativeBoundaryWithThresholdLoadMasked(AscendC::LocalTensor<float> difference, AscendC::LocalTensor<float> y1, AscendC::LocalTensor<float> y2, AscendC::LocalTensor<float> boundary) {
        if ASCEND_IS_AIV {
            set_mask_norm();
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
            ComputeNativeBoundary<CALC_REPEATS, UNIT_RATIO, true>(difference, y1, y2, boundary);
        }
    }

    template <uint32_t CONVERT_REPEATS>
    __aicore__ inline void ConvertHalfY1Column(AscendC::LocalTensor<float> y1, AscendC::LocalTensor<T> y1Half) {
        if ASCEND_IS_AIV {
            set_mask_norm();
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
            // 原生转换不会代替高层Cast建立Gather→VCONV的数据依赖。
            // 先确保boxes两次Gather完成，再提升两列；末尾保留算术消费者屏障。
            pipe_barrier(PIPE_V);
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr()),
                reinterpret_cast<__ubuf__ T*>(y1Half.GetPhyAddr()), CONVERT_REPEATS, 1, 1, 8, 4);

            pipe_barrier(PIPE_V);
        }
    }

    template <uint32_t CONVERT_REPEATS>
    __aicore__ inline void ConvertHalfAnchorColumn(AscendC::LocalTensor<float> cy, AscendC::LocalTensor<T> cyHalf) {
        if ASCEND_IS_AIV {
            set_mask_norm();
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
            pipe_barrier(PIPE_V);
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr()),
                reinterpret_cast<__ubuf__ T*>(cyHalf.GetPhyAddr()), CONVERT_REPEATS, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
        }
    }

    template <uint32_t CONVERT_REPEATS>
    __aicore__ inline void ConvertHalfColumnsFp16B256(AscendC::LocalTensor<float> y1, AscendC::LocalTensor<T> y1Half, AscendC::LocalTensor<float> y2, AscendC::LocalTensor<T> y2Half, AscendC::LocalTensor<float> cy, AscendC::LocalTensor<T> cyHalf) {
        if ASCEND_IS_AIV {
            set_mask_norm();
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
            pipe_barrier(PIPE_V);
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr()),
                reinterpret_cast<__ubuf__ T*>(y1Half.GetPhyAddr()), CONVERT_REPEATS, 1, 1, 8, 4);
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr()),
                reinterpret_cast<__ubuf__ T*>(y2Half.GetPhyAddr()), CONVERT_REPEATS, 1, 1, 8, 4);
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr()),
                reinterpret_cast<__ubuf__ T*>(cyHalf.GetPhyAddr()), CONVERT_REPEATS, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
        }
    }

    template <uint32_t COMPARE_REPEATS, uint32_t SELECT_REPEATS, uint32_t PAD>
    __aicore__ inline void CompareAndSelectCounted(AscendC::LocalTensor<float> cy, AscendC::LocalTensor<float> boundary) { CompareAndSelectCountedImpl<SELECT_REPEATS>(cy, boundary, PAD, COMPARE_REPEATS); }

    template <uint32_t PAD>
    __aicore__ inline void PrepareCompareAddressFp16Range257To4095() { PrepareCompareAddress<7U * PAD + 48U>(); }

    template <uint32_t SELECT_REPEATS, uint32_t PAD>
    __aicore__ inline void CompareAndSelectCountedFp16Range257To4095(AscendC::LocalTensor<float> cy, AscendC::LocalTensor<float> boundary) { CompareAndSelectCountedImpl<SELECT_REPEATS>(cy, boundary, PAD, PAD / 64U); }

    __aicore__ inline void ExtractHalfBoxes(AscendC::LocalTensor<T> y1Half, AscendC::LocalTensor<T> y2Half, AscendC::LocalTensor<T> cyHalf, uint32_t calcRows, AscendC::LocalTensor<float> y1, AscendC::LocalTensor<float> y2) {
        if ASCEND_IS_AIV {
            auto boxesPtr = reinterpret_cast<__ubuf__ uint16_t*>(boxIn_.GetPhyAddr());
            auto anchorsPtr = reinterpret_cast<__ubuf__ uint16_t*>(anchorIn_.GetPhyAddr());
            auto y1HalfPtr = reinterpret_cast<__ubuf__ uint16_t*>(y1Half.GetPhyAddr());
            auto y2HalfPtr = reinterpret_cast<__ubuf__ uint16_t*>(y2Half.GetPhyAddr());
            auto cyHalfPtr = reinterpret_cast<__ubuf__ uint16_t*>(cyHalf.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            // 原生固定模式抽列；src1的+512保留已核验的CANN9调用形式。
            // 它不是“第512行”，也不要当作本算子的自定义索引表。
            vreducev2(y1HalfPtr, boxesPtr, boxesPtr + 512, calcRows / 32U, 1, 4, 8, 0);
            vreducev2(y2HalfPtr, boxesPtr, boxesPtr + 512, calcRows / 32U, 1, 6, 8, 0);

            pipe_barrier(PIPE_V);
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
            // 三次原生half→FP32转换；后面的几何运算和比较不使用half精度。
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y1.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ T*>(y1Half.GetPhyAddr()), calcRows / 64U, 1, 1, 8, 4);
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(y2.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ T*>(y2Half.GetPhyAddr()), calcRows / 64U, 1, 1, 8, 4);

            pipe_barrier(PIPE_V);
        }
    }

    __aicore__ inline void ExtractHalfAnchors(AscendC::LocalTensor<T> cyHalf, uint32_t calcRows, AscendC::LocalTensor<float> cy) {
        if ASCEND_IS_AIV {
            auto anchorsPtr = reinterpret_cast<__ubuf__ uint16_t*>(anchorIn_.GetPhyAddr());
            auto cyHalfPtr = reinterpret_cast<__ubuf__ uint16_t*>(cyHalf.GetPhyAddr());
            set_mask_norm();
            set_vector_mask(0xffffffffffffffffULL, 0xffffffffffffffffULL);
            vreducev2(cyHalfPtr, anchorsPtr, anchorsPtr + 512, calcRows / 64U, 1, 2, 8, 0);
            pipe_barrier(PIPE_V);
            set_vector_mask(0ULL, 0xffffffffffffffffULL);
            vconv_f162f32(reinterpret_cast<__ubuf__ float*>(cy.GetPhyAddr()),
                          reinterpret_cast<__ubuf__ T*>(cyHalf.GetPhyAddr()), calcRows / 64U, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
        }
    }

    template <uint32_t SELECT_REPEATS>
    __aicore__ inline void CompareAndSelectCountedFp16Range257To4095Part2(AscendC::LocalTensor<float> cy, AscendC::LocalTensor<float> boundary, uint32_t calcRows) { CompareAndSelectCountedImpl<SELECT_REPEATS>(cy, boundary, calcRows, calcRows / 64U); }

    __aicore__ inline void BindPointGlobals(GM_ADDR boxes, GM_ADDR anchors,
        GM_ADDR output, uint32_t boxElements, uint32_t anchorElements) {
        boxesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(boxes) + rowStart_ * 4U, boxElements);
        anchorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchors) + rowStart_ * 2U, anchorElements);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output) + rowStart_,
            SbgAlignBytes(coreRows_ * sizeof(T)) / sizeof(T));
    }

    template <uint32_t PAD, uint32_t SCRATCH_BASE, uint32_t SCRATCH_COUNT>
    __aicore__ inline void BindPointUb(uint32_t anchorOffset, uint32_t outputOffset,
        uint32_t constantsOffset, uint32_t compareOffset) {
        boxIn_ = AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, 0U, PAD * 4U);
        anchorIn_ = AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, anchorOffset, PAD * 2U);
        output_ = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC, outputOffset, PAD);
        constants_ = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC, constantsOffset, PAD * 2U);
        scratch_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, SCRATCH_BASE, SCRATCH_COUNT);
        compare_ = scratch_.template ReinterpretCast<uint8_t>()[compareOffset];
        one_ = constants_[0];
        zero_ = constants_[PAD];
    }

    template <bool UNIT_RATIO, uint32_t RATIO_KIND, bool READ_THRESHOLD>
    __aicore__ inline void LoadPointParameters() {
        waistRatio_ = UNIT_RATIO ? 1.0f : RATIO_KIND == 1U ? 0.8f :
            RATIO_KIND == 2U ? 0.9f : RATIO_KIND == 3U ? 0.5f :
            reinterpret_cast<__gm__ volatile float*>(tiling_)[3];
        AscendC::InitSocState();
        if constexpr (READ_THRESHOLD) {
            if (hasThreshold_ != 0U && threshold_ != nullptr) {
                thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ T*>(threshold_)[0]);
            }
        }
    }

    template <uint32_t PAD, uint32_t BANK_GAP>
    __aicore__ inline SbgPointColumns MakePointColumns() {
        return {scratch_[2U * PAD], scratch_[3U * PAD + BANK_GAP],
            scratch_[4U * PAD + BANK_GAP], scratch_[5U * PAD + 2U * BANK_GAP],
            scratch_[6U * PAD + 3U * BANK_GAP]};
    }

    template <uint32_t WORKING, uint32_t PER>
    __aicore__ inline bool InitPointSlice(uint32_t &block) {
        block = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (block >= WORKING) return false;
        rowStart_ = block * PER;
        return true;
    }

    __aicore__ inline void SetPointParameters(GM_ADDR threshold, GM_ADDR tiling, uint32_t present) {
        threshold_ = threshold;
        tiling_ = tiling;
        hasThreshold_ = present;
        thresholdValue_ = 0.0f;
    }

    template <uint32_t WORKING, uint32_t ROWS>
    __aicore__ inline bool InitSinglePoint(GM_ADDR boxes, GM_ADDR anchors, GM_ADDR threshold,
        GM_ADDR output, uint32_t present, GM_ADDR tiling) {
        const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (block >= WORKING) return false;
        rowStart_ = 0U;
        coreRows_ = ROWS;
        SetPointParameters(threshold, tiling, present);
        BindPointGlobals(boxes, anchors, output,
            SbgAlignBytes(coreRows_ * 4U * sizeof(T)) / sizeof(T),
            SbgAlignBytes(coreRows_ * 2U * sizeof(T)) / sizeof(T));
        return true;
    }

    __aicore__ inline void CopyOutPoint(uint16_t blocks) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyParams copy{1, blocks, 0, 0};
        AscendC::DataCopy(outputGm_, output_, copy);
    }

    template <bool LOW_MASK, bool RESET_AFTER>
    __aicore__ inline void SelectPointMask(const SbgPointColumns &columns, uint64_t count) {
        AscendC::SetMaskNorm();
        AscendC::Compare(columns.cy, columns.boundary, AscendC::CMPMODE::GT, count, {1, 1, 1, 8, 8, 8});
        if constexpr (LOW_MASK) {
            if ASCEND_IS_AIV {
                set_mask_norm();
                set_vector_mask(0ULL, (1ULL << coreRows_) - 1ULL);
            }
        }
        AscendC::Select<T, AscendC::SELMODE::VSEL_CMPMASK_SPR>(output_, one_, zero_, 1, {1, 1, 1, 8, 8, 8});
        if constexpr (RESET_AFTER) AscendC::ResetMask();
    }
};

class SbgHalfPointState : protected SbgPointState<half> {
protected:
    AscendC::LocalTensor<float> halfStage_;


    template <uint32_t PAD, uint32_t SCRATCH_BASE, uint32_t SCRATCH_COUNT>
    __aicore__ inline void BindPointUb(uint32_t anchorOffset, uint32_t outputOffset,
        uint32_t constantsOffset, uint32_t compareOffset, uint32_t halfOffset) {
        SbgPointState<half>::BindPointUb<PAD, SCRATCH_BASE, SCRATCH_COUNT>(
            anchorOffset, outputOffset, constantsOffset, compareOffset);
        halfStage_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, halfOffset, PAD * 4U);
    }
    template <uint32_t PAD>
    __aicore__ inline SbgHalfColumns<half> MakeHalfColumns() {
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
        AscendC::LocalTensor<half> y1 = this->halfStage_.template ReinterpretCast<half>();
        return {y1, y1[PAD], y1[2U * PAD]};
    }

    template <uint32_t PAD, uint32_t BANK_GAP, bool UNIT_RATIO>
    __aicore__ inline void ComputeLargeHalf(uint32_t calcRows) {
        AscendC::Duplicate(one_, static_cast<half>(1.0f), static_cast<int32_t>(calcRows));
        AscendC::Duplicate(zero_, static_cast<half>(0.0f), static_cast<int32_t>(calcRows));
        PrepareCompareAddress<7680U>();
        // MTE2→Vector事件：必须等输入DMA完成才能读UB；不是全卡核间屏障。
        // CopyIn已分别记录两路就绪；这里先仅等待boxes。
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        // 下面为同一个scratch中的视图，各偏移单位是float元素。
        // y1/y2/cy是按样本连续的列；difference和boundary分别保存差值与判定边界。
        const SbgPointColumns columns = MakePointColumns<PAD, BANK_GAP>();
        const SbgHalfColumns<half> halfColumns = MakeHalfColumns<PAD>();
        // Independent extraction operations share one dependency barrier.
        // Fixed-pattern source-1 addresses match the CANN 9.0 dav_c220 wrapper.
        ExtractHalfBoxes(halfColumns.y1, halfColumns.y2, halfColumns.cy, calcRows, columns.y1, columns.y2);
        // 严格保持FP32分步舍入：d=y2-y1；d*=ratio；bound=y1+d；再加threshold。
        // 未经边界验证不能改为FMA、重新结合公式，或把ratio=1直接替换成y2。
        AscendC::Sub(columns.difference, columns.y2, columns.y1, static_cast<int32_t>(calcRows));
        if constexpr (!UNIT_RATIO) AscendC::Muls(columns.difference, columns.difference, waistRatio_, static_cast<int32_t>(calcRows));
        AscendC::Add(columns.boundary, columns.y1, columns.difference, static_cast<int32_t>(calcRows));
        if (hasThreshold_ != 0U && threshold_ != nullptr) {
            // 读取当前调用的真实阈值；不存在时保留0，绝不解引用空GM指针。
            thresholdValue_ = static_cast<float>(reinterpret_cast<__gm__ volatile half*>(threshold_)[0]);
        }
        // An actual zero threshold leaves every finite comparison unchanged,
        // including signed-zero boundaries. Positive thresholds retain Adds.
        // 仅在真实读取的阈值为±0时省略Adds；正阈值必须走原来的加法。
        if (thresholdValue_ != 0.0f) {
            AscendC::Adds(columns.boundary, columns.boundary, thresholdValue_, static_cast<int32_t>(calcRows));
        }
        // boxes的half抽列、FP32转换及参考线先算；比较前汇合anchors。
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        ExtractHalfAnchors(halfColumns.cy, calcRows, columns.cy);
        CompareAndSelectCountedFp16Range257To4095Part2<1>(columns.cy, columns.boundary, calcRows);
    }
};
