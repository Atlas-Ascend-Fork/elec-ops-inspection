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

#if SBG_PARSE_HALF_SHORT_LITERAL_BITS
template<uint32_t ROWS>
static __aicore__ __attribute__((always_inline)) inline void GeometryHalfShortLiteralBits(
    GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o){
 static_assert(ROWS>=33U && ROWS<=63U,"Short tile occupancy");
 constexpr uint32_t ones=((ROWS*8U+31U)/32U)*32U;
 copy_gm_to_ubuf_align_b16((__ubuf__ uint16_t*)2048,(__gm__ uint16_t*)a,0,1,ROWS*4U,0,0,0,0);
 GeometryWideSocState<true,false,false>();
 set_mov_pad_val(static_cast<uint64_t>(0x3c00U));
 copy_gm_to_ubuf_align_b16((__ubuf__ uint16_t*)0,(__gm__ uint16_t*)b,0,1,ROWS*8U,0,16,0,0);
 AscendC::DcciGMImpl<half,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>((__gm__ half*)t);
 const uint64_t rawHalf=*reinterpret_cast<const volatile __gm__ uint16_t*>(t);
#if SBG_PARSE_HALF_SHORT_LOCAL_PIPE_JOIN
 // Drain preceding local-pipeline work before the dependent consumer.
 AscendC::PipeBarrier<PIPE_ALL>();
#else
 AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
#endif
 vreducev2((__ubuf__ uint16_t*)15360,(__ubuf__ uint16_t*)0,(__ubuf__ uint16_t*)0,2,1,4,8,0);
 vreducev2((__ubuf__ uint16_t*)16384,(__ubuf__ uint16_t*)0,(__ubuf__ uint16_t*)0,2,1,6,8,0);
 vreducev2((__ubuf__ uint16_t*)17408,(__ubuf__ uint16_t*)2048,(__ubuf__ uint16_t*)2048,1,1,2,8,0);
 // Each little-endian word retains [actual Half encoding, Half +0].
 vector_dup((__ubuf__ uint32_t*)18432,static_cast<uint32_t>(rawHalf),1,1,1,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vconv_f162f32((__ubuf__ float*)0,(__ubuf__ half*)15360,4,1,1,32,32);
 AscendC::PipeBarrier<PIPE_V>();
 vsub((__ubuf__ float*)1024,(__ubuf__ float*)1024,(__ubuf__ float*)0,1,1,1,1,0,0,0);
 vbrcb((__ubuf__ uint32_t*)16384,(__ubuf__ uint32_t*)3072,1,0,1);
 AscendC::PipeBarrier<PIPE_V>();
 vmuls((__ubuf__ float*)1024,(__ubuf__ float*)1024,0.8f,1,1,1,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vadd((__ubuf__ float*)1024,(__ubuf__ float*)1024,(__ubuf__ float*)0,1,1,1,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vadd((__ubuf__ float*)1024,(__ubuf__ float*)1024,(__ubuf__ float*)16384,1,1,1,0,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vcmp_gt((__ubuf__ float*)2048,(__ubuf__ float*)1024,1,1,1,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vsel((__ubuf__ half*)1024,(__ubuf__ half*)ones,(__ubuf__ half*)16416,1,1,0,0,0,0,0);
#if SBG_PARSE_HALF_SHORT_LOCAL_PIPE_JOIN
 // Drain preceding local-pipeline work before the dependent consumer.
 AscendC::PipeBarrier<PIPE_ALL>();
#else
 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
#endif
 copy_ubuf_to_gm_align_b16((__gm__ half*)o,(__ubuf__ half*)1024,0,1,ROWS*2U,0,0,0,0);
}
#endif

// The launch can exceed the output-owner count. Idle cores enter no compute path.

#if SBG_PARSE_HALF_BATCH_MERGE
template<uint32_t ROWS>
static __aicore__ __attribute__((always_inline)) inline void GeometryHalfBatchMerge(
    GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o){
 static_assert(ROWS>=128U && ROWS<=512U && ROWS%64U==0U,"Complete small batch");
 constexpr uint32_t repeats=ROWS/64U;
 constexpr uint32_t onesAddress=4096U+ROWS*4U;
 auto raw=(__ubuf__ uint16_t*)0;
 auto anchor=(__ubuf__ uint16_t*)4096;
 auto y1Half=(__ubuf__ uint16_t*)16384;
 auto y2Half=(__ubuf__ uint16_t*)(16384U+ROWS*2U);
 auto cyHalf=(__ubuf__ uint16_t*)(16384U+ROWS*4U);
 auto y1=(__ubuf__ float*)8192;
 auto y2=(__ubuf__ float*)(8192U+ROWS*4U);
 auto cy=(__ubuf__ float*)(8192U+ROWS*8U);
 auto bitmap=(__ubuf__ uint8_t*)33024;
 copy_gm_to_ubuf((__ubuf__ void*)raw,(__gm__ void*)b,0,1,ROWS/4U,0,0);
 GeometryWideSocState<true,false,false>();
 set_mov_pad_val(static_cast<uint64_t>(0x3c00U));
 copy_gm_to_ubuf_align_b16(anchor,(__gm__ uint16_t*)a,0,1,ROWS*4U,0,16,0,0);
 set_mov_pad_val(static_cast<uint64_t>(0U));
 copy_gm_to_ubuf_align_b16((__ubuf__ uint16_t*)30720,(__gm__ uint16_t*)t,0,1,2,0,15,0,0);
 // Define bitmap padding for the last private128-lane selection repeat.
 vector_dup((__ubuf__ uint32_t*)bitmap,static_cast<uint32_t>(0),1,1,1,0,0);
 AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 vreducev2(y1Half,raw,raw,2U*repeats,1,4,8,0);
 vreducev2(y2Half,raw,raw,2U*repeats,1,6,8,0);
 vreducev2(cyHalf,anchor,anchor,repeats,1,2,8,0);
 AscendC::PipeBarrier<PIPE_V>();
 // Three adjacent gathered fields become three adjacent Float fields.
 vconv_f162f32(y1,(__ubuf__ half*)y1Half,3U*repeats,1,1,8,4);
 vconv_f162f32((__ubuf__ float*)30752,(__ubuf__ half*)30720,1,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vsub(y2,y2,y1,repeats,1,1,1,8,8,8);
 vbrcb((__ubuf__ uint32_t*)32000,(__ubuf__ uint32_t*)30752,1,0,1);
 AscendC::PipeBarrier<PIPE_V>();
 vmuls(y2,y2,0.9f,repeats,1,1,8,8);
 AscendC::PipeBarrier<PIPE_V>();
 vadd(y2,y2,y1,repeats,1,1,1,8,8,8);
 AscendC::PipeBarrier<PIPE_V>();
 vadd(y2,(__ubuf__ float*)32000,y2,repeats,1,0,1,8,0,8);
 AscendC::PipeBarrier<PIPE_V>();
 vcmpv_gt(bitmap,cy,y2,repeats,1,1,1,8,8,8);
 // Every source is the defined finite Half1 padding block.
 vcmp_gt((__ubuf__ half*)onesAddress,(__ubuf__ half*)onesAddress,1,1,0,0,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 // Only ROWS real outputs are written; extra selected lanes stay in private UB.
 vsel((__ubuf__ half*)0,(__ubuf__ half*)onesAddress,bitmap,(ROWS+127U)/128U,
      1,0,1,8,0,8,static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE));
 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 copy_ubuf_to_gm((__gm__ void*)o,(__ubuf__ void*)0,0,1,ROWS/16U,0,0);
}
#endif


#if SBG_PARSE_FLOAT_BATCH_MERGE
#if SBG_PARSE_FLOAT_BATCH_MERGE
static __aicore__ __attribute__((always_inline)) inline void GeometryFloatBatchBoundAndSelect(__ubuf__ float *bound,__ubuf__ float *cy,__ubuf__ float *y1,__ubuf__ float *ones,__ubuf__ uint8_t *bitmap,uint32_t repeats,float actualThreshold) {
AscendC::PipeBarrier<PIPE_V>();
 vmuls(bound,bound,0.9f,repeats,1,1,8,8);
 AscendC::PipeBarrier<PIPE_V>();
 vadd(bound,bound,y1,repeats,1,1,1,8,8,8);
 AscendC::PipeBarrier<PIPE_V>();
 vadds(bound,bound,actualThreshold,repeats,1,1,8,8);
 AscendC::PipeBarrier<PIPE_V>();
 vcmpv_gt(bitmap,cy,bound,repeats,1,1,1,8,8,8);
#if SBG_PARSE_FLOAT_BATCH_SHARED_SCORE_DESCRIPTOR
 // All64 Float ones are already defined by the completed original producer.
 // Reading the broadcast block against them contiguously still gives allfalse.
 vcmp_gt(ones,ones,1,1,0,1,0,0,0);
#else
 vcmp_gt(ones,ones,1,1,0,0,0,0,0);
#endif
 AscendC::PipeBarrier<PIPE_V>();
 vsel(bound,ones,bitmap,repeats,1,0,1,8,0,0,static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE));
}
#endif

template<uint32_t ROWS>
static __aicore__ __attribute__((always_inline)) inline void GeometryFloatBatchMerge(
    GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o){
 static_assert(ROWS>=128U && ROWS<=512U && ROWS%64U==0U,"Complete small batch");
 constexpr uint32_t repeats=ROWS/64U;
 auto raw=(__ubuf__ float*)0;
 auto anchor=(__ubuf__ float*)10240;
 auto y1=(__ubuf__ float*)16384;
 auto y2=(__ubuf__ float*)(16384U+ROWS*4U);
 auto cy=(__ubuf__ float*)(16384U+ROWS*8U);
 auto bound=(__ubuf__ float*)0;
 auto ones=(__ubuf__ float*)(ROWS*16U);
#if SBG_PARSE_FLOAT_BATCH_SHARED_SCORE_DESCRIPTOR
 // The original y1 values are dead before the bitmap is produced.
 auto bitmap=(__ubuf__ uint8_t*)16384;
#else
 auto bitmap=(__ubuf__ uint8_t*)24576;
#endif
 copy_gm_to_ubuf((__ubuf__ void*)raw,(__gm__ void*)b,0,1,ROWS/2U,0,0);
 GeometryWideSocState<true,false,false>();
 copy_gm_to_ubuf((__ubuf__ void*)anchor,(__gm__ void*)a,0,1,ROWS/4U,0,0);
 // Independent private ones are initialized while both real inputs arrive.
 vector_dup(ones,1.0f,1,1,1,0,0);
 AscendC::DcciGMImpl<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>((__gm__ float*)t);
 float actualThreshold=*reinterpret_cast<const volatile __gm__ float*>(t);
#if SBG_PARSE_FLOAT_BATCH_LOCAL_PIPE_JOIN
 AscendC::PipeBarrier<PIPE_ALL>();
#else
 AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
#endif
 vreducev2((__ubuf__ uint32_t*)y1,(__ubuf__ uint32_t*)raw,(__ubuf__ uint32_t*)raw,4U*repeats,1,4,8,0);
 vreducev2((__ubuf__ uint32_t*)y2,(__ubuf__ uint32_t*)raw,(__ubuf__ uint32_t*)raw,4U*repeats,1,6,8,0);
 AscendC::PipeBarrier<PIPE_V>();
 // Box input is dead; the independent anchor extraction shares this barrier.
 vsub(bound,y2,y1,repeats,1,1,1,8,8,8);
 vreducev2((__ubuf__ uint32_t*)cy,(__ubuf__ uint32_t*)anchor,(__ubuf__ uint32_t*)anchor,2U*repeats,1,2,8,0);
 GeometryFloatBatchBoundAndSelect(bound,cy,y1,ones,bitmap,repeats,actualThreshold);
#if SBG_PARSE_FLOAT_BATCH_LOCAL_PIPE_JOIN
 AscendC::PipeBarrier<PIPE_ALL>();
#else
 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
#endif
 copy_ubuf_to_gm((__gm__ void*)o,(__ubuf__ void*)bound,0,1,ROWS/8U,0,0);
}
#endif


#if SBG_PARSE_HALF_UNIT_NATIVE_DMA
static __aicore__ __attribute__((always_inline)) inline void GeometryHalfUnitNativeDma(
    GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o){
 copy_gm_to_ubuf((__ubuf__ void*)0,(__gm__ void*)b,0,2,4,0,0);
 copy_gm_to_ubuf((__ubuf__ void*)2048,(__gm__ void*)a,0,1,4,0,0);
 set_mov_pad_val(static_cast<uint64_t>(0));
 copy_gm_to_ubuf_align_b16((__ubuf__ uint16_t*)18432,(__gm__ uint16_t*)t,0,1,2,0,15,0,0);
 GeometryWideSocState<true,false,false>();
 vector_dup((__ubuf__ uint16_t*)512,static_cast<uint16_t>(0x3c00U),1,1,1,0,0);
 AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 vreducev2((__ubuf__ uint16_t*)15360,(__ubuf__ uint16_t*)0,(__ubuf__ uint16_t*)0,1,1,4,8,0);
 vreducev2((__ubuf__ uint16_t*)16384,(__ubuf__ uint16_t*)0,(__ubuf__ uint16_t*)0,1,1,6,8,0);
 vreducev2((__ubuf__ uint16_t*)17408,(__ubuf__ uint16_t*)2048,(__ubuf__ uint16_t*)2048,1,1,2,8,0);
 AscendC::PipeBarrier<PIPE_V>();
 set_vector_mask(0,0xffffffffULL);
 vconv_f162f32((__ubuf__ float*)0,(__ubuf__ half*)15360,4,1,1,32,32);
 AscendC::PipeBarrier<PIPE_V>();
 vsub((__ubuf__ float*)1024,(__ubuf__ float*)1024,(__ubuf__ float*)0,1,1,1,1,0,0,0);
 vbrcb((__ubuf__ uint32_t*)16384,(__ubuf__ uint32_t*)3072,1,0,1);
 AscendC::PipeBarrier<PIPE_V>();
 vadd((__ubuf__ float*)1024,(__ubuf__ float*)1024,(__ubuf__ float*)0,1,1,1,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vadd((__ubuf__ float*)1024,(__ubuf__ float*)1024,(__ubuf__ float*)16384,1,1,1,0,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vcmp_gt((__ubuf__ float*)2048,(__ubuf__ float*)1024,1,1,1,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vsel((__ubuf__ half*)1024,(__ubuf__ half*)512,(__ubuf__ half*)16416,1,1,0,0,0,0,0);
 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 copy_ubuf_to_gm((__gm__ void*)o,(__ubuf__ void*)1024,0,1,2,0,0);
}
#endif
