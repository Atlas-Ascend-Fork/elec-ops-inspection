/*
 * Copyright (c) 2026 dhltat.
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
#include "spline_device.h"

// Fast-family UB layout and shared interpolation/reduction primitives.
template <class U> __aicore__ inline LocalTensor<U> FastAlloc(uint32_t &offset, uint32_t count) {
    LocalTensor<U> tensor(TPosition::VECCALC, offset, count);
    uint32_t phase = ((offset & 511) + 256) & 511;
    offset = (offset + (count * sizeof(U) + 31) / 32 * 32 + 511) / 512 * 512 + phase;
    return tensor;
}
template <class U> __aicore__ inline LocalTensor<U> OperandAlloc(uint32_t &offset, uint32_t count) {
    offset += 32;
    return FastAlloc<U>(offset, count);
}
template <class U> __aicore__ inline LocalTensor<U> FastOutAlloc(uint32_t &offset, uint32_t count) {
    offset += 256;
    return FastAlloc<U>(offset, count);
}
template <class T>
__aicore__ inline float PseudoValue(GlobalTensor<T> input, GlobalTensor<uint16_t> bits, uint32_t index) {
    if constexpr (sizeof(T) == 4)
        return input.GetValue(index);
    else {
        union {
            uint32_t u;
            float f;
        } v;
        v.u = uint32_t(bits.GetValue(index)) << 16;
        return v.f;
    }
}


// Public pointer DMA API. Byte counts and 32-byte strides match DataCopy.
template<class T> __aicore__ inline void SplineCCopy(LocalTensor<T> d, GlobalTensor<T> s,uint32_t n) {
    asc_copy_gm2ub((__ubuf__ void *)d.GetPhyAddr(),(__gm__ void *)s.GetPhyAddr(),n*sizeof(T));
}
template<class T> __aicore__ inline void SplineCCopy(GlobalTensor<T> d, LocalTensor<T> s,uint32_t n) {
    asc_copy_ub2gm((__gm__ void *)d.GetPhyAddr(),(__ubuf__ void *)s.GetPhyAddr(),n*sizeof(T));
}
template<class T> __aicore__ inline void SplineCCopyPad(LocalTensor<T> d, GlobalTensor<T> s,
    DataCopyExtParams p, DataCopyPadExtParams<T> padding) {
    // All call sites in this experiment use false/zero explicit padding.
    asc_copy_gm2ub_align((__ubuf__ uint8_t *)d.GetPhyAddr(),(__gm__ uint8_t *)s.GetPhyAddr(),
                         p.blockCount,p.blockLen,0,0,p.srcStride,p.dstStride);
}
template<class T> __aicore__ inline void SplineCCopyPad(GlobalTensor<T> d, LocalTensor<T> s,DataCopyExtParams p) {
    asc_copy_ub2gm_align((__gm__ uint8_t *)d.GetPhyAddr(),(__ubuf__ uint8_t *)s.GetPhyAddr(),
                         p.blockCount,p.blockLen,0,0,p.srcStride,p.dstStride);
}

// Official pointer C API math; synchronization is explicit at every dependency.
__aicore__ inline __ubuf__ float *SplinePtr(LocalTensor<float> t) {
    return (__ubuf__ float *)t.GetPhyAddr();
}
__aicore__ inline void SplineCAdd(LocalTensor<float> d, LocalTensor<float> a,
                                 LocalTensor<float> b, uint32_t count) {
    asc_add(SplinePtr(d), SplinePtr(a), SplinePtr(b), count);
}
__aicore__ inline void SplineCAdd(LocalTensor<float> d, LocalTensor<float> a,
                                 LocalTensor<float> b, uint64_t mask, uint8_t repeat,
                                 BinaryRepeatParams p) {
    SetMaskNorm();
    asc_set_vector_mask(0, mask == 64 ? ~uint64_t(0) : (uint64_t(1) << mask) - 1);
    asc_add(SplinePtr(d), SplinePtr(a), SplinePtr(b), repeat,
            p.dstBlkStride,p.src0BlkStride,p.src1BlkStride,
            p.dstRepStride,p.src0RepStride,p.src1RepStride);
}
__aicore__ inline void SplineCMul(LocalTensor<float> d, LocalTensor<float> a,
                                 LocalTensor<float> b, uint32_t count) {
    asc_mul(SplinePtr(d), SplinePtr(a), SplinePtr(b), count);
}
__aicore__ inline void SplineCMul(LocalTensor<float> d, LocalTensor<float> a,
                                 LocalTensor<float> b, uint64_t mask, uint8_t repeat,
                                 BinaryRepeatParams p) {
    SetMaskNorm();
    asc_set_vector_mask(0, mask == 64 ? ~uint64_t(0) : (uint64_t(1) << mask) - 1);
    asc_mul(SplinePtr(d), SplinePtr(a), SplinePtr(b), repeat,
            p.dstBlkStride,p.src0BlkStride,p.src1BlkStride,
            p.dstRepStride,p.src0RepStride,p.src1RepStride);
}
__aicore__ inline void SplineCMuls(LocalTensor<float> d, LocalTensor<float> a,float v,uint32_t n) {
    asc_mul_scalar(SplinePtr(d),SplinePtr(a),v,n);
}
__aicore__ inline void SplineCAxpy(LocalTensor<float> d, LocalTensor<float> a,float v,uint32_t n) {
    asc_axpy(SplinePtr(d),SplinePtr(a),v,n);
}
__aicore__ inline void SplineCBrcb(LocalTensor<float> d, LocalTensor<float> a,uint8_t repeat,BrcbRepeatParams p) {
    asc_brcb((__ubuf__ uint32_t *)SplinePtr(d),(__ubuf__ uint32_t *)SplinePtr(a),p.dstBlkStride,p.dstRepStride,repeat);
}


template<class T> __aicore__ inline T SplineCGet(LocalTensor<T> s,uint32_t i) {
    return ((__ubuf__ T *)s.GetPhyAddr())[i];
}
template<class T> __aicore__ inline void SplineCSet(LocalTensor<T> tensor, uint32_t elementIndex, T value) {
    ((__ubuf__ T *)tensor.GetPhyAddr())[elementIndex] = value;
}
template<class T> __aicore__ inline void SplineCDuplicate(LocalTensor<T> d,T value,uint32_t n) {
    asc_duplicate((__ubuf__ T *)d.GetPhyAddr(),value,n);
}
__aicore__ inline void SplineCCast(LocalTensor<float> d,LocalTensor<bfloat16_t> s,RoundMode,uint32_t n) {
    asc_bfloat162float(SplinePtr(d),(__ubuf__ bfloat16_t *)s.GetPhyAddr(),n);
}
__aicore__ inline void SplineCCast(LocalTensor<bfloat16_t> d,LocalTensor<float> s,RoundMode,uint32_t n) {
    asc_float2bfloat16_rn((__ubuf__ bfloat16_t *)d.GetPhyAddr(),SplinePtr(s),n);
}

__aicore__ inline void SplineCAdds(LocalTensor<float> d, LocalTensor<float> s, float v, uint32_t n) {
    asc_add_scalar(SplinePtr(d), SplinePtr(s), v, n);
}
__aicore__ inline void SplineCAdds(LocalTensor<float> d, LocalTensor<float> s, float v,
                                  uint64_t mask, uint8_t repeat, UnaryRepeatParams p) {
    SetMaskNorm();
    asc_set_vector_mask(0, mask == 64 ? ~uint64_t(0) : (uint64_t(1) << mask) - 1);
    asc_add_scalar(SplinePtr(d), SplinePtr(s), v, repeat,
                   p.dstBlkStride, p.srcBlkStride, p.dstRepStride, p.srcRepStride);
}
template <class T>
__aicore__ inline void SplineCDuplicate(LocalTensor<T> d, T value, uint64_t mask,
                                       uint8_t repeat, uint16_t blockStride, uint16_t repeatStride) {
    const uint64_t lo = mask >= 64 ? ~uint64_t(0) : (uint64_t(1) << mask) - 1;
    const uint64_t hi = mask <= 64 ? 0 : mask == 128 ? ~uint64_t(0) : (uint64_t(1) << (mask - 64)) - 1;
    SetMaskNorm();
    asc_set_vector_mask(hi, lo);
    asc_duplicate((__ubuf__ T *)d.GetPhyAddr(), value, repeat, blockStride, repeatStride);
}
