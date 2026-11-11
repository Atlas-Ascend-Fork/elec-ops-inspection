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

#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
__aicore__ inline void GeometryCompleteRawEntryState(GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o);
#if SBG_PARSE_COMPLETE_RAW
__aicore__ inline void GeometryCompleteRawEntryState(GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o){
 const uint64_t origin=static_cast<uint64_t>(AscendC::GetBlockIdx())*(512ULL*16ULL);
 auto raw=(__ubuf__ float*)0;
 auto anchor=(__ubuf__ float*)8192;
 auto y1=(__ubuf__ float*)28672;
 auto y2=(__ubuf__ float*)30720;
 auto cy=(__ubuf__ float*)32768;
 auto result=(__ubuf__ float*)34816;
 auto ones=(__ubuf__ float*)36864;
 auto bitmap=(__ubuf__ uint8_t*)43008;
 copy_gm_to_ubuf((__ubuf__ void*)raw,(__gm__ void*)(b+origin),0,1,256,0,0);
 GeometryWideSocState<true,false,false>();
 copy_gm_to_ubuf((__ubuf__ void*)anchor,(__gm__ void*)(a+origin/2ULL),0,1,128,0,0);
 AscendC::DcciGMImpl<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>((__gm__ float*)t);
 float actualThreshold=*reinterpret_cast<const volatile __gm__ float*>(t);
 AscendC::PipeBarrier<PIPE_V>();
 vector_dup(ones,1.0f,1,1,1,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vcmp_gt(ones,ones,1,1,0,1,0,0,0);
 AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 vreducev2((__ubuf__ uint32_t*)y1,(__ubuf__ uint32_t*)raw,(__ubuf__ uint32_t*)raw,32,1,4,8,0);
 vreducev2((__ubuf__ uint32_t*)y2,(__ubuf__ uint32_t*)raw,(__ubuf__ uint32_t*)raw,32,1,6,8,0);
 AscendC::PipeBarrier<PIPE_V>();
 vsub(y2,y2,y1,8,1,1,1,8,8,8);
 vreducev2((__ubuf__ uint32_t*)cy,(__ubuf__ uint32_t*)anchor,(__ubuf__ uint32_t*)anchor,16,1,2,8,0);
 AscendC::PipeBarrier<PIPE_V>();
 vmuls(y2,y2,0.8f,8,1,1,8,8);
 AscendC::PipeBarrier<PIPE_V>();
 vadd(y2,y2,y1,8,1,1,1,8,8,8);
 AscendC::PipeBarrier<PIPE_V>();
 vadds(y2,y2,actualThreshold,8,1,1,8,8);
 AscendC::PipeBarrier<PIPE_V>();
 vcmpv_gt(bitmap,cy,y2,8,1,1,1,8,8,8);
 AscendC::PipeBarrier<PIPE_V>();
 vsel(result,ones,bitmap,8,1,0,1,8,0,8,static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE));
 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 copy_ubuf_to_gm((__gm__ void*)(o+origin/4ULL),(__ubuf__ void*)result,0,1,64,0,0);
}
#endif
#endif


#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
// Both original owner roles have the same rounded512-lane Vector descriptors.
// A real call tests whether the compiler can retain one physical math body.
static __aicore__ __attribute__((always_inline)) inline void GeometryCompactEntryFloatMath(GM_ADDR t);
#if SBG_PARSE_COMPACT_RAW
static __aicore__ __attribute__((always_inline)) inline void GeometryCompactEntryFloatMath(GM_ADDR t){
 auto raw=(__ubuf__ float*)0;
 auto anchor=(__ubuf__ float*)8192;
 auto y1=(__ubuf__ float*)28672;
 auto y2=(__ubuf__ float*)30720;
 auto cy=(__ubuf__ float*)32768;
 auto result=(__ubuf__ float*)34816;
 auto bitmap=(__ubuf__ uint8_t*)43008;
 AscendC::DcciGMImpl<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>((__gm__ float*)t);
 const int64_t thresholdBits=*reinterpret_cast<const volatile __gm__ uint32_t*>(t);
 AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
 vreducev2((__ubuf__ uint32_t*)y1,(__ubuf__ uint32_t*)raw,(__ubuf__ uint32_t*)raw,32,1,4,8,0);
 vreducev2((__ubuf__ uint32_t*)y2,(__ubuf__ uint32_t*)raw,(__ubuf__ uint32_t*)raw,32,1,6,8,0);
 AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 vreducev2((__ubuf__ uint32_t*)cy,(__ubuf__ uint32_t*)anchor,(__ubuf__ uint32_t*)anchor,16,1,2,8,0);
 AscendC::PipeBarrier<PIPE_V>();
 vsub(y2,y2,y1,8,1,1,1,8,8,8);
 AscendC::PipeBarrier<PIPE_V>();
 vmuls(y2,y2,0.8f,8,1,1,8,8);
 AscendC::PipeBarrier<PIPE_V>();
 vadd(y2,y2,y1,8,1,1,1,8,8,8);
 if(__builtin_expect(thresholdBits!=0,0)){
  float actualThreshold=*reinterpret_cast<const volatile __gm__ float*>(t);
  AscendC::PipeBarrier<PIPE_V>();
  vadds(y2,y2,actualThreshold,8,1,1,8,8);
 }
 AscendC::PipeBarrier<PIPE_V>();
 vector_dup(y1,1.0f,1,1,1,0,0);
 vcmpv_gt(bitmap,cy,y2,8,1,1,1,8,8,8);
 AscendC::PipeBarrier<PIPE_V>();
 vcmp_gt(y1,y1,1,1,0,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vsel(result,y1,bitmap,8,1,0,1,8,0,8,static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE));
 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
}
#endif

template<uint32_t ROWS,bool IS_TAIL>
__aicore__ inline void GeometryCompactEntryFloatPair(GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o,int64_t owner);
#if SBG_PARSE_COMPACT_RAW
template<uint32_t ROWS,bool IS_TAIL>
__aicore__ inline void GeometryCompactEntryFloatPair(GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o,int64_t owner){
 static_assert(ROWS>=505U && ROWS<=512U,"Native rounded512 tile");
 const uint64_t origin=IS_TAIL?
     (static_cast<uint64_t>(kGeometryFloatWideOwners)-1ULL)*(512ULL*16ULL):
     (static_cast<uint64_t>(owner)-1ULL)*(512ULL*16ULL);
 if constexpr(IS_TAIL)
  copy_gm_to_ubuf_align_b32((__ubuf__ float*)0,(__gm__ float*)(b+origin),0,1,ROWS*16U,0,0,0,0);
 else
  copy_gm_to_ubuf((__ubuf__ void*)0,(__gm__ void*)(b+origin),0,2,128,0,0);
 AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
 GeometryWideSocState<true,false,false>();
 if constexpr(IS_TAIL)
  copy_gm_to_ubuf_align_b32((__ubuf__ float*)8192,(__gm__ float*)(a+origin/2ULL),0,1,ROWS*8U,0,0,0,0);
 else
  copy_gm_to_ubuf((__ubuf__ void*)8192,(__gm__ void*)(a+origin/2ULL),0,1,128,0,0);
 GeometryCompactEntryFloatMath(t);
 if constexpr(IS_TAIL)
  copy_ubuf_to_gm_align_b32((__gm__ float*)(o+origin/4ULL),(__ubuf__ float*)34816,0,1,ROWS*4U,0,0,0,0);
 else
  copy_ubuf_to_gm((__gm__ void*)(o+origin/4ULL),(__ubuf__ void*)34816,0,2,32,0,0);
}
#endif
#endif
