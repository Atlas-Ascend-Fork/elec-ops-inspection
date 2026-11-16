/*
 * Copyright (c) 2026 gcw_OTnEiQgj.
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

#pragma once

#include "kernel_operator.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"

#define KPT_INL __aicore__ __attribute__((always_inline)) inline

namespace elec::impl {

constexpr uint32_t DIM_C0 = 32U;
constexpr uint32_t DIM_C1 = 4U;
constexpr uint32_t DIM_K9 = 9U;
constexpr uint32_t DIM_WT = DIM_C1 * DIM_C0 * DIM_K9;
constexpr uint32_t DIM_CR = 128U;

KPT_INL uint32_t BitsU32(float x) {
    union {
        float f;
        uint32_t u;
    } cvt = {x};
    return cvt.u;
}

KPT_INL float BitsF32(uint32_t u) {
    union {
        uint32_t u;
        float f;
    } cvt = {u};
    return cvt.f;
}

KPT_INL float RecipPos(float x) {
    return (x > 0.0f) ? (1.0f / x) : 1.0f;
}

KPT_INL float EvalPolyExp(float z) {
    if (z >= 0.0f) {
        return 1.0f;
    }
    if (z <= -104.0f) {
        return 0.0f;
    }
    const int32_t n = static_cast<int32_t>(z * 1.44269504f - 0.5f);
    const float fn = static_cast<float>(n);
    const float r = (z - fn * 0.693145751953125f) - fn * 1.428606765330187e-06f;
    float p = 0.0013888889f;
    p = p * r + 0.0083333338f;
    p = p * r + 0.0416666679f;
    p = p * r + 0.1666666716f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    if (n >= -126) {
        return p * BitsF32(static_cast<uint32_t>(n + 127) << 23U);
    }
    return (p * BitsF32(static_cast<uint32_t>(64) << 23U)) *
           BitsF32(static_cast<uint32_t>(n + 190) << 23U);
}

__aicore__ constexpr inline uint32_t AlignStep(uint32_t prevEnd, uint32_t phase) {
    const uint32_t aligned = (prevEnd + 511U) & ~511U;
    const uint32_t cand = aligned + phase;
    return (cand >= prevEnd) ? cand : (cand + 512U);
}

KPT_INL void ApplyVecMask(uint32_t count) {
    if (count >= 64U) {
        set_vector_mask(~0ULL, ~0ULL);
    } else {
        set_vector_mask(0ULL, (~0ULL) >> (64U - count));
    }
}

KPT_INL void OpVmuls(__ubuf__ float *dst, __ubuf__ float *src, float s, uint8_t reps) {
    if ASCEND_IS_AIV {
        vmuls(dst, src, s, reps, uint16_t(1U), uint16_t(1U), uint16_t(8U), uint16_t(8U));
    }
}

KPT_INL void OpVaxpy(__ubuf__ float *dst, __ubuf__ float *src, float s, uint8_t reps) {
    if ASCEND_IS_AIV {
        vaxpy(dst, src, s, reps, uint16_t(1U), uint16_t(1U), uint16_t(8U), uint16_t(8U));
    }
}

KPT_INL void OpVadds(__ubuf__ float *dst, __ubuf__ float *src, float s, uint8_t reps) {
    if ASCEND_IS_AIV {
        vadds(dst, src, s, reps, uint16_t(1U), uint16_t(1U), uint16_t(8U), uint16_t(8U));
    }
}

KPT_INL void OpVaddsI(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, int32_t s, uint8_t reps) {
    if ASCEND_IS_AIV {
        vadds(dst, src, s, reps, uint16_t(1U), uint16_t(1U), uint16_t(8U), uint16_t(8U));
    }
}

KPT_INL void OpVadd(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t reps) {
    if ASCEND_IS_AIV {
        vadd(dst, src0, src1, reps, uint16_t(1U), uint16_t(1U), uint16_t(1U), uint16_t(8U), uint16_t(8U), uint16_t(8U));
    }
}

KPT_INL void OpVmul(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t reps) {
    if ASCEND_IS_AIV {
        vmul(dst, src0, src1, reps, uint16_t(1U), uint16_t(1U), uint16_t(1U), uint16_t(8U), uint16_t(8U), uint16_t(8U));
    }
}

KPT_INL void OpVexp(__ubuf__ float *dst, __ubuf__ float *src, uint8_t reps) {
    if ASCEND_IS_AIV {
        vexp(dst, src, reps, uint16_t(1U), uint16_t(1U), uint16_t(8U), uint16_t(8U));
    }
}

KPT_INL void OpVdup(__ubuf__ float *dst, float s, uint8_t reps) {
    if ASCEND_IS_AIV {
        vector_dup(dst, s, reps, uint16_t(1U), uint16_t(1U), uint16_t(8U), uint16_t(0U));
    }
}

KPT_INL void OpVdupS(__ubuf__ uint32_t *dst, uint32_t s, uint8_t reps, uint8_t dstRepStride) {
    if ASCEND_IS_AIV {
        vector_dup(dst, s, reps, uint16_t(1U), uint16_t(1U), dstRepStride, uint16_t(0U));
    }
}

KPT_INL void OpVandB(
    __ubuf__ uint16_t *dst, __ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1Mask256B, uint8_t reps) {
    if ASCEND_IS_AIV {
        vand(dst, src0, src1Mask256B, reps, uint16_t(1U), uint16_t(1U), uint16_t(1U), uint16_t(8U), uint16_t(8U), uint16_t(0U));
    }
}

KPT_INL void OpVor(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t reps) {
    if ASCEND_IS_AIV {
        vor(dst, src0, src1, reps, uint16_t(1U), uint16_t(1U), uint16_t(1U), uint16_t(8U), uint16_t(8U), uint16_t(8U));
    }
}

KPT_INL void OpVgat(__ubuf__ float *dst, __ubuf__ float *src, __ubuf__ uint32_t *offset, uint8_t reps) {
    if ASCEND_IS_AIV {
        const uint32_t addr = static_cast<uint32_t>(reinterpret_cast<uint64_t>(src));
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(dst), offset, addr, 8U, reps);
    }
}

KPT_INL void OpVcmax(
    __ubuf__ float *dst, __ubuf__ float *src, int32_t reps, int32_t dstRepStride, int32_t srcBlkStride, int32_t srcRepStride) {
    if ASCEND_IS_AIV {
        vcmax(dst, src, reps, dstRepStride, srcBlkStride, srcRepStride, Order_t::ONLY_VALUE);
    }
}

KPT_INL void OpVcgmax(
    __ubuf__ float *dst, __ubuf__ float *src, int32_t reps, int32_t dstRepStride, int32_t srcBlkStride, int32_t srcRepStride) {
    if ASCEND_IS_AIV {
        vcgmax(dst, src, reps, dstRepStride, srcBlkStride, srcRepStride);
    }
}

KPT_INL void OpVcadd(
    __ubuf__ float *dst, __ubuf__ float *src, int32_t reps, int32_t dstRepStride, int32_t srcBlkStride, int32_t srcRepStride) {
    if ASCEND_IS_AIV {
        vcadd(dst, src, reps, dstRepStride, srcBlkStride, srcRepStride, 0);
    }
}

KPT_INL void VecAddSpan(__ubuf__ float *pD, __ubuf__ float *p0, __ubuf__ float *p1, uint32_t count) {
    const uint32_t full = count >> 6U;
    const uint32_t tail = count & 63U;
    if (full > 0U) {
        OpVadd(pD, p0, p1, static_cast<uint8_t>(full));
    }
    if (tail > 0U) {
        const uint32_t off = full << 6U;
        ApplyVecMask(tail);
        OpVadd(pD + off, p0 + off, p1 + off, 1U);
        set_vector_mask(~0ULL, ~0ULL);
    }
}

KPT_INL void VecAdd2Span(
    __ubuf__ float *d0, __ubuf__ float *a0, __ubuf__ float *b0,
    __ubuf__ float *d1, __ubuf__ float *a1, __ubuf__ float *b1,
    uint32_t count) {
    const uint32_t full = count >> 6U;
    const uint32_t tail = count & 63U;
    if (full > 0U) {
        const uint8_t r = static_cast<uint8_t>(full);
        OpVadd(d0, a0, b0, r);
        OpVadd(d1, a1, b1, r);
    }
    if (tail > 0U) {
        const uint32_t off = full << 6U;
        ApplyVecMask(tail);
        OpVadd(d0 + off, a0 + off, b0 + off, 1U);
        OpVadd(d1 + off, a1 + off, b1 + off, 1U);
        set_vector_mask(~0ULL, ~0ULL);
    }
}

KPT_INL void VecAdd3Span(
    __ubuf__ float *d0, __ubuf__ float *a0, __ubuf__ float *b0,
    __ubuf__ float *d1, __ubuf__ float *a1, __ubuf__ float *b1,
    __ubuf__ float *d2, __ubuf__ float *a2, __ubuf__ float *b2,
    uint32_t count) {
    const uint32_t full = count >> 6U;
    const uint32_t tail = count & 63U;
    if (full > 0U) {
        const uint8_t r = static_cast<uint8_t>(full);
        OpVadd(d0, a0, b0, r);
        OpVadd(d1, a1, b1, r);
        OpVadd(d2, a2, b2, r);
    }
    if (tail > 0U) {
        const uint32_t off = full << 6U;
        ApplyVecMask(tail);
        OpVadd(d0 + off, a0 + off, b0 + off, 1U);
        OpVadd(d1 + off, a1 + off, b1 + off, 1U);
        OpVadd(d2 + off, a2 + off, b2 + off, 1U);
        set_vector_mask(~0ULL, ~0ULL);
    }
}

KPT_INL void VecAddsISpan(__ubuf__ int32_t *pD, __ubuf__ int32_t *pS, int32_t s, uint32_t count) {
    const uint32_t full = count >> 6U;
    const uint32_t tail = count & 63U;
    if (full > 0U) {
        OpVaddsI(pD, pS, s, static_cast<uint8_t>(full));
    }
    if (tail > 0U) {
        const uint32_t off = full << 6U;
        ApplyVecMask(tail);
        OpVaddsI(pD + off, pS + off, s, 1U);
        set_vector_mask(~0ULL, ~0ULL);
    }
}

KPT_INL void VecGatSpan(__ubuf__ float *pDst, __ubuf__ float *pSrc, __ubuf__ uint32_t *pOff, uint32_t count) {
    const uint32_t full = count >> 6U;
    const uint32_t tail = count & 63U;
    if (full > 0U) {
        OpVgat(pDst, pSrc, pOff, static_cast<uint8_t>(full));
    }
    if (tail > 0U) {
        const uint32_t off = full << 6U;
        ApplyVecMask(tail);
        OpVgat(pDst + off, pSrc, pOff + off, 1U);
        set_vector_mask(~0ULL, ~0ULL);
    }
}

KPT_INL void VecAddW96(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint32_t rows) {
    const uint32_t full = (rows * 96U) >> 6U;
    if (full > 0U) {
        OpVadd(dst, src0, src1, static_cast<uint8_t>(full));
    }
    if ((rows & 1U) != 0U) {
        set_vector_mask(0ULL, 0xFFFFFFFFULL);
        OpVadd(dst + (full << 6U), src0 + (full << 6U), src1 + (full << 6U), 1U);
        set_vector_mask(~0ULL, ~0ULL);
    }
}

template <uint16_t FLAG_ID>
KPT_INL void SyncAllBlk() {
    (void)FLAG_ID;
    AscendC::SyncAll<true>();
}

template <uint16_t FLAG_ID>
KPT_INL void SyncStepBlk() {
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(FLAG_ID);
    AscendC::CrossCoreWaitFlag(FLAG_ID);
}

}
