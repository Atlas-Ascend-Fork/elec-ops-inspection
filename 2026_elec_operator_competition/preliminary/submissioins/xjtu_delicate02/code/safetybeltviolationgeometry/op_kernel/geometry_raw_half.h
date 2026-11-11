// Copyright 2026 Delicate02
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

#if SBG_PARSE_HALF_SHARED_ONE_PAD_MERGED || SBG_PARSE_HALF_SCALAR_BITS_DUPLICATE
static __aicore__ __attribute__((always_inline)) inline void GeometryHalfFullBound09() {
AscendC::PipeBarrier<PIPE_V>();
 vsub((__ubuf__ float*)1024,(__ubuf__ float*)1024,(__ubuf__ float*)0,1,1,1,1,0,0,0);
 vbrcb((__ubuf__ uint32_t*)16384,(__ubuf__ uint32_t*)3072,1,0,1);
 AscendC::PipeBarrier<PIPE_V>();
 vmuls((__ubuf__ float*)1024,(__ubuf__ float*)1024,0.9f,1,1,1,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vadd((__ubuf__ float*)1024,(__ubuf__ float*)1024,(__ubuf__ float*)0,1,1,1,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vadd((__ubuf__ float*)1024,(__ubuf__ float*)16384,(__ubuf__ float*)1024,1,1,0,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
}
#endif

#if SBG_PARSE_HALF_SHARED_ONE_PAD_MERGED
static __aicore__ __attribute__((always_inline)) inline void GeometryHalfSharedOnePadMerged(
    GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o){
 const uint64_t rawOwner=static_cast<uint64_t>(get_block_idx());
 const uint64_t owner=rawOwner*static_cast<uint64_t>(AscendC::GetTaskRation())+static_cast<uint64_t>(AscendC::GetSubBlockIdx());
 const uint64_t origin=owner*512ULL;
 set_mov_pad_val(static_cast<uint64_t>(0x3c00U));
#if SBG_PARSE_HALF_ANCHOR_PAD_ONES
 // Anchor padding already supplies the live score ones.
 copy_gm_to_ubuf((__ubuf__ void*)0,(__gm__ void*)(b+origin),0,1,16,0,0);
#else
 copy_gm_to_ubuf_align_b16((__ubuf__ uint16_t*)0,(__gm__ uint16_t*)(b+origin),0,1,512,0,16,0,0);
#endif
 GeometryWideSocState<true,false,false>();
 const uint64_t outputByteOffset=origin/4ULL;
 uint64_t anchorAddress=reinterpret_cast<uint64_t>(a)+outputByteOffset;
 anchorAddress+=outputByteOffset;
 // All input reads use the same typed Half1 padding value.
 copy_gm_to_ubuf_align_b16((__ubuf__ uint16_t*)2048,(__gm__ uint16_t*)anchorAddress,0,1,256,0,16,0,0);
#if SBG_PARSE_HALF_SPR_SCORE_REUSE
 // Converted padding supplies a defined zero block for mode0 scoring.
 set_mov_pad_val(static_cast<uint64_t>(0U));
#endif
 copy_gm_to_ubuf_align_b16((__ubuf__ uint16_t*)18432,(__gm__ uint16_t*)t,0,1,2,0,15,0,0);
 AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 vreducev2((__ubuf__ uint16_t*)15360,(__ubuf__ uint16_t*)0,(__ubuf__ uint16_t*)0,2,1,4,8,0);
 vreducev2((__ubuf__ uint16_t*)16384,(__ubuf__ uint16_t*)0,(__ubuf__ uint16_t*)0,2,1,6,8,0);
 vreducev2((__ubuf__ uint16_t*)17408,(__ubuf__ uint16_t*)2048,(__ubuf__ uint16_t*)2048,1,1,2,8,0);
 AscendC::PipeBarrier<PIPE_V>();
 vconv_f162f32((__ubuf__ float*)0,(__ubuf__ half*)15360,4,1,1,32,32);
  GeometryHalfFullBound09();
#if SBG_PARSE_HALF_SPR_SCORE_REUSE
 // Only the first64 Half scores are stored; each uses its real Float GT bit.
 vcmp_gt((__ubuf__ float*)2048,(__ubuf__ float*)1024,1,1,1,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vsel((__ubuf__ half*)1024,(__ubuf__ half*)2304,(__ubuf__ half*)16416,
      1,1,0,0,0,0,0);
#else
 vcmpv_gt((__ubuf__ uint8_t*)20480,(__ubuf__ float*)2048,(__ubuf__ float*)1024,1,1,1,1,0,0,0);
 // Half1 > Half1 is false in every lane; mode1 reads that zero Scalar.
 constexpr uint32_t onesAddress=SBG_PARSE_HALF_ANCHOR_PAD_ONES?2304U:512U;
 vcmp_gt((__ubuf__ half*)onesAddress,(__ubuf__ half*)onesAddress,1,1,0,0,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vsel((__ubuf__ half*)1024,(__ubuf__ half*)onesAddress,(__ubuf__ uint8_t*)20480,
      1,1,0,1,0,0,0,static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 copy_ubuf_to_gm((__gm__ void*)(o+outputByteOffset),(__ubuf__ void*)1024,0,1,4,0,0);
}
#endif


#if SBG_PARSE_FLOAT_FULL_SCALAR_BITMAP_SCORE
#if SBG_PARSE_FLOAT_FULL_SCALAR_BITMAP_SCORE
static __aicore__ __attribute__((always_inline)) inline void GeometryFloatFullBoundAndSelect(__ubuf__ float *cy,__ubuf__ float *y1,__ubuf__ float *y2,__ubuf__ float *ones,__ubuf__ uint8_t *bitmap,float actualThreshold) {
AscendC::PipeBarrier<PIPE_V>();
 vmuls(y2,y2,0.9f,1,1,1,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vadd(y2,y2,y1,1,1,1,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vadds(y2,y2,actualThreshold,1,1,1,0,0);
 AscendC::PipeBarrier<PIPE_V>();
#if SBG_PARSE_FLOAT_FULL_DIRECT_SPR_ZERO
 // The real ordered comparison supplies the selection mask directly.
 vcmp_gt(cy,y2,1,1,1,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vsel(cy,ones,(__ubuf__ float*)21504,1,1,0,1,0,0,0,0);
#else
 // y1 has no remaining arithmetic consumer; only its first bitmap block is reused.
 vcmpv_gt(bitmap,cy,y2,1,1,1,1,0,0,0);
#if SBG_PARSE_FLOAT_BOUND_ZERO_DESCRIPTOR
 // Each defined bound lane compares with itself, including NaN/Inf/signed zero.
 vcmp_gt(y2,y2,1,1,1,1,0,0,0);
#else
 vcmp_gt(ones,ones,1,1,0,0,0,0,0);
#endif
 AscendC::PipeBarrier<PIPE_V>();
 vsel(cy,ones,bitmap,1,1,0,1,0,0,0,static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
}
#endif

static __aicore__ __attribute__((always_inline)) inline void GeometryFloatFullScalarBitmapScore(
    GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o){
 const uint64_t rawOwner=static_cast<uint64_t>(get_block_idx());
 const uint64_t owner=rawOwner*static_cast<uint64_t>(AscendC::GetTaskRation())+static_cast<uint64_t>(AscendC::GetSubBlockIdx());
 const uint64_t origin=owner*1024ULL;
 auto raw=(__ubuf__ float*)0;
 auto anchor=(__ubuf__ float*)4096;
 auto y1=(__ubuf__ float*)14336;
 auto y2=(__ubuf__ float*)15360;
 auto cy=(__ubuf__ float*)16384;
 auto ones=(__ubuf__ float*)1024;
 auto bitmap=(__ubuf__ uint8_t*)14336;
#if SBG_PARSE_FLOAT_FULL_PLAIN_BOX_SCORES
 // Real coordinates occupy exactly1024 bytes; score constants are private.
 copy_gm_to_ubuf((__ubuf__ void*)raw,(__gm__ void*)(b+origin),0,1,32,0,0);
#else
 set_mov_pad_val(static_cast<uint64_t>(0x3f800000U));
 copy_gm_to_ubuf_align_b32(raw,(__gm__ float*)(b+origin),0,1,1024,0,8,0,0);
#endif
 GeometryWideSocState<true,false,false>();
#if SBG_PARSE_FLOAT_ANCHOR_FOUR_BLOCKS
 // Four adjacent128B bursts retain the same512B coordinate interval.
 copy_gm_to_ubuf((__ubuf__ void*)anchor,(__gm__ void*)(a+origin/2ULL),0,4,4,0,0);
#else
 copy_gm_to_ubuf((__ubuf__ void*)anchor,(__gm__ void*)(a+origin/2ULL),0,1,16,0,0);
#endif
#if SBG_PARSE_FLOAT_FULL_DIRECT_SPR_ZERO
#if SBG_PARSE_FLOAT_FULL_PLAIN_BOX_SCORES
 // This disjoint ones region begins after the real box input. Both constant
 // producers complete at the original gather barrier before score selection.
 vector_dup(ones,1.0f,1,1,1,0,0);
#endif
 // Private zero scores are independent of both pending coordinate copies.
 vector_dup((__ubuf__ float*)21504,0.0f,1,1,1,0,0);
#endif
 AscendC::DcciGMImpl<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>((__gm__ float*)t);
 float actualThreshold=*reinterpret_cast<const volatile __gm__ float*>(t);
 AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 vreducev2((__ubuf__ uint32_t*)y1,(__ubuf__ uint32_t*)raw,(__ubuf__ uint32_t*)raw,4,1,4,8,0);
 vreducev2((__ubuf__ uint32_t*)y2,(__ubuf__ uint32_t*)raw,(__ubuf__ uint32_t*)raw,4,1,6,8,0);
 AscendC::PipeBarrier<PIPE_V>();
 vsub(y2,y2,y1,1,1,1,1,0,0,0);
 vreducev2((__ubuf__ uint32_t*)cy,(__ubuf__ uint32_t*)anchor,(__ubuf__ uint32_t*)anchor,2,1,2,8,0);
 GeometryFloatFullBoundAndSelect(cy,y1,y2,ones,bitmap,actualThreshold);
 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 copy_ubuf_to_gm((__gm__ void*)(o+origin/4ULL),(__ubuf__ void*)cy,0,1,8,0,0);
}
#endif


#if SBG_PARSE_HALF_SCALAR_BITS_DUPLICATE
#if SBG_PARSE_HALF_SCALAR_BITS_DUPLICATE
static __aicore__ __attribute__((always_inline)) inline void GeometryHalfConvertThresholdFields() {
AscendC::PipeBarrier<PIPE_V>();
#if SBG_PARSE_HALF_THRESHOLD_DMA_TRANSPORT
 vconv_f162f32((__ubuf__ float*)0,(__ubuf__ half*)15360,3,1,1,32,32);
#if SBG_PARSE_HALF_THRESHOLD_BLOCK_BROADCAST
 // Each source block is the same initialized32B region. The first eight
 // Float values retain the original threshold and seven positive zeros.
 vconv_f162f32((__ubuf__ float*)3072,(__ubuf__ half*)18432,1,1,0,0,0);
#else
 set_vector_mask(0,0xffULL);
 vconv_f162f32((__ubuf__ float*)3072,(__ubuf__ half*)18432,1,1,1,0,0);
 set_vector_mask(0xffffffffffffffffULL,0xffffffffffffffffULL);
#endif
#else
 vconv_f162f32((__ubuf__ float*)0,(__ubuf__ half*)15360,4,1,1,32,32);
#endif
}
#endif

static __aicore__ __attribute__((always_inline)) inline void GeometryHalfScalarBitsDuplicate(
    GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o){
 const uint64_t rawOwner=static_cast<uint64_t>(get_block_idx());
 const uint64_t owner=rawOwner*static_cast<uint64_t>(AscendC::GetTaskRation())+static_cast<uint64_t>(AscendC::GetSubBlockIdx());
 const uint64_t origin=owner*512ULL;
 set_mov_pad_val(static_cast<uint64_t>(0x3c00U));
#if SBG_PARSE_HALF_BOX_FOUR_BLOCKS
 // Four adjacent128B bursts cover exactly the original512B interval.
 copy_gm_to_ubuf((__ubuf__ void*)0,(__gm__ void*)(b+origin),0,4,4,0,0);
#else
 copy_gm_to_ubuf((__ubuf__ void*)0,(__gm__ void*)(b+origin),0,1,16,0,0);
#endif
 GeometryWideSocState<true,false,false>();
 const uint64_t outputByteOffset=origin/4ULL;
 uint64_t anchorAddress=reinterpret_cast<uint64_t>(a)+outputByteOffset;
 anchorAddress+=outputByteOffset;
 copy_gm_to_ubuf_align_b16((__ubuf__ uint16_t*)2048,(__gm__ uint16_t*)anchorAddress,0,1,256,0,16,0,0);
#if SBG_PARSE_HALF_THRESHOLD_DMA_TRANSPORT
 // Eight defined Half inputs supply the separate threshold conversion.
 set_mov_pad_val(static_cast<uint64_t>(0));
#if SBG_PARSE_HALF_THRESHOLD_BLOCK_BROADCAST
 // Define every Half in the source block used by the repeated-block Cast.
 copy_gm_to_ubuf_align_b16((__ubuf__ uint16_t*)18432,(__gm__ uint16_t*)t,0,1,2,0,15,0,0);
#else
 copy_gm_to_ubuf_align_b16((__ubuf__ uint16_t*)18432,(__gm__ uint16_t*)t,0,1,2,0,7,0,0);
#endif
#else
 AscendC::DcciGMImpl<half,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>((__gm__ half*)t);
 const uint64_t rawHalf=*reinterpret_cast<const volatile __gm__ uint16_t*>(t);
#endif
 AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 vreducev2((__ubuf__ uint16_t*)15360,(__ubuf__ uint16_t*)0,(__ubuf__ uint16_t*)0,2,1,4,8,0);
 vreducev2((__ubuf__ uint16_t*)16384,(__ubuf__ uint16_t*)0,(__ubuf__ uint16_t*)0,2,1,6,8,0);
 vreducev2((__ubuf__ uint16_t*)17408,(__ubuf__ uint16_t*)2048,(__ubuf__ uint16_t*)2048,1,1,2,8,0);
#if !SBG_PARSE_HALF_THRESHOLD_DMA_TRANSPORT
 // Each little-endian B32 word is [actual Half bits, Half +0].
 vector_dup((__ubuf__ uint32_t*)18432,static_cast<uint32_t>(rawHalf),1,1,1,0,0);
#endif
 GeometryHalfConvertThresholdFields();
 GeometryHalfFullBound09();
 vcmp_gt((__ubuf__ float*)2048,(__ubuf__ float*)1024,1,1,1,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vsel((__ubuf__ half*)1024,(__ubuf__ half*)2304,(__ubuf__ half*)16416,1,1,0,0,0,0,0);
 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 copy_ubuf_to_gm((__gm__ void*)(o+outputByteOffset),(__ubuf__ void*)1024,0,1,4,0,0);
}
#endif
