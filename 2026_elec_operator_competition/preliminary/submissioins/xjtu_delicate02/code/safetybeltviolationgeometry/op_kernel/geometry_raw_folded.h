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
#if SBG_PARSE_FOLD_DYNAMIC_RATIO
#define SBG_FOLD_RATIO_PARAMETER ,float dynamicRatio
#else
#define SBG_FOLD_RATIO_PARAMETER
#endif
template<uint32_t EXTRA,bool TAIL,uint32_t FIXED_WORKERS=0U,bool CANONICAL_SINGLE=false>
__aicore__ inline void GeometryFoldedRawFloatEntry(GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o,int64_t owner SBG_FOLD_RATIO_PARAMETER);
#if SBG_PARSE_FOLDED_RAW
template<uint32_t EXTRA,bool TAIL,uint32_t FIXED_WORKERS,bool CANONICAL_SINGLE>
struct GeometryFoldedRawFloatState {
static_assert(EXTRA>=1U && EXTRA<=8U,"Folded64 remainder");
 static constexpr uint32_t rows=TAIL?64U+EXTRA:64U;
 static constexpr uint8_t repeats=TAIL?2U:1U;
#if defined(TILING_KEY_VAR)
 static constexpr bool BOTH_INPUTS_BEFORE_CTRL=!TAIL && (SBG_PARSE_FOLD_MINIMUM ||
     ((TILING_KEY_VAR & 65535ULL)==0ULL &&
      ((TILING_KEY_VAR>>16U)&3ULL)==1ULL && ((TILING_KEY_VAR>>26U)&15ULL)==15ULL &&
      ((TILING_KEY_VAR>>18U)&255ULL)>=125ULL && ((TILING_KEY_VAR>>18U)&255ULL)<=132ULL));
#else
 static constexpr bool BOTH_INPUTS_BEFORE_CTRL=false;
#endif
 // A one-repeat source/destination never advances its repeat base.
 static constexpr uint8_t unaryRepeatStride=CANONICAL_SINGLE && !TAIL?0U:8U;
 // Only complete prefixes have repeat0 alone. Two-repeat tails retain8,
 // including the packed comparison's destination repeat advance.
 static constexpr uint8_t binaryRepeatStride=SBG_PARSE_FOLD_BINARY_REPEAT && !TAIL?0U:8U;
 // SDK block counts are positive. In this original fully occupied
 // family4..7 owners imply byte offsets<=6144; the unsigned narrowing is exact.

#if SBG_PARSE_FOLD_PAD_ONES
 __ubuf__ float *raw=(__ubuf__ float*)(TAIL?0U:32U);
#else
 __ubuf__ float *raw=(__ubuf__ float*)0;
#endif
 __ubuf__ float *anchor=(__ubuf__ float*)4096;
 __ubuf__ float *y1=(__ubuf__ float*)14336;
 __ubuf__ float *y2=(__ubuf__ float*)15360;
#if SBG_PARSE_FOLD_PREFIX_BRANCH_HINT
 __ubuf__ float *cy=(__ubuf__ float*)(TAIL?14336U:16384U);
#else
 __ubuf__ float *cy=(__ubuf__ float*)16384;
#endif
#if SBG_PARSE_FOLD_PREFIX_BRANCH_HINT
 // The tail's input at UB0 is dead after the gather barrier.
 __ubuf__ float *bound=(__ubuf__ float*)(TAIL?0U:32U);
#else
 __ubuf__ float *bound=(__ubuf__ float*)32;
#endif
#if SBG_PARSE_FOLD_PREFIX_BRANCH_HINT
 // Both fields are dead before the packed comparison writes its bitmap.
 __ubuf__ uint8_t *bitmap=(__ubuf__ uint8_t*)(TAIL?15360U:14336U);
#else
 __ubuf__ uint8_t *bitmap=(__ubuf__ uint8_t*)21504;
#endif
#if SBG_PARSE_FOLD_PAD_ONES
 __ubuf__ float *scoreOnes=(__ubuf__ float*)(TAIL?((rows*16U+31U)&~31U):0U);
#endif
 __aicore__ __attribute__((always_inline)) inline void CopyInputs(GM_ADDR b,GM_ADDR a,uint64_t origin) {
#if SBG_PARSE_FOLD_PAD_ONES
 set_mov_pad_val(static_cast<uint64_t>(0x3f800000U));
 copy_gm_to_ubuf_align_b32((__ubuf__ float*)0,(__gm__ float*)(b+origin),0,1,rows*16U,TAIL?0U:8U,TAIL?8U:0U,0,0);
#else
 if constexpr(TAIL)
  copy_gm_to_ubuf_align_b32(raw,(__gm__ float*)(b+origin),0,1,rows*16U,0,0,0,0);
 else copy_gm_to_ubuf((__ubuf__ void*)raw,(__gm__ void*)(b+origin),0,1,32,0,0);
#endif
 if constexpr(!BOTH_INPUTS_BEFORE_CTRL)GeometryWideSocState<true,false,false>();
 if constexpr(TAIL)
  copy_gm_to_ubuf_align_b32(anchor,(__gm__ float*)(a+origin/2ULL),0,1,rows*8U,0,0,0,0);
 else {
#if SBG_PARSE_FOLD_PREFIX_ANCHOR_FOUR_BLOCKS
  // Adjacent bursts cover the same complete-prefix coordinate interval.
  copy_gm_to_ubuf((__ubuf__ void*)anchor,(__gm__ void*)(a+origin/2ULL),0,4,4,0,0);
#else
  copy_gm_to_ubuf((__ubuf__ void*)anchor,(__gm__ void*)(a+origin/2ULL),0,1,16,0,0);
#endif
 }
 if constexpr(BOTH_INPUTS_BEFORE_CTRL)GeometryWideSocState<true,false,false>();

 }
 __aicore__ __attribute__((always_inline)) inline void ComputeBound(float actualThreshold SBG_FOLD_RATIO_PARAMETER) {
#if SBG_PARSE_FOLD_LOCAL_PIPE_JOIN
 AscendC::PipeBarrier<PIPE_ALL>();
#else
 AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
#endif
 vreducev2((__ubuf__ uint32_t*)y1,(__ubuf__ uint32_t*)raw,(__ubuf__ uint32_t*)raw,TAIL?5:4,1,4,8,0);
 vreducev2((__ubuf__ uint32_t*)y2,(__ubuf__ uint32_t*)raw,(__ubuf__ uint32_t*)raw,TAIL?5:4,1,6,8,0);
#if !SBG_PARSE_FOLD_SUB_CY_OVERLAP
 vreducev2((__ubuf__ uint32_t*)cy,(__ubuf__ uint32_t*)anchor,(__ubuf__ uint32_t*)anchor,TAIL?3:2,1,2,8,0);
#endif
 AscendC::PipeBarrier<PIPE_V>();
 vsub(bound,y2,y1,repeats,1,1,1,binaryRepeatStride,binaryRepeatStride,binaryRepeatStride);
#if SBG_PARSE_FOLD_SUB_CY_OVERLAP
 // Box extraction has completed before Sub reuses the dead raw region.
 // This anchor read and cy write are disjoint from every Sub operand.
#if SBG_PARSE_FOLD_PREFIX_BRANCH_HINT
 if constexpr(!TAIL)
#endif
 vreducev2((__ubuf__ uint32_t*)cy,(__ubuf__ uint32_t*)anchor,(__ubuf__ uint32_t*)anchor,TAIL?3:2,1,2,8,0);
 // When issued here, the retained Sub completion barrier also completes cy.
#endif
 AscendC::PipeBarrier<PIPE_V>();
#if SBG_PARSE_FOLD_PREFIX_BRANCH_HINT && !SBG_PARSE_FOLD_DYNAMIC_RATIO && SBG_PARSE_FOLD_PAD_ONES
 if constexpr(!TAIL)
  vsub(y2,scoreOnes,scoreOnes,1,1,0,0,0,0,0);
#endif
#if SBG_PARSE_FOLD_DYNAMIC_RATIO
 vmuls(bound,bound,dynamicRatio,repeats,1,1,unaryRepeatStride,unaryRepeatStride);
#else
 vmuls(bound,bound,0.5f,repeats,1,1,unaryRepeatStride,unaryRepeatStride);
#endif
 AscendC::PipeBarrier<PIPE_V>();
 vadd(bound,bound,y1,repeats,1,1,1,binaryRepeatStride,binaryRepeatStride,binaryRepeatStride);
 AscendC::PipeBarrier<PIPE_V>();
 vadds(bound,bound,actualThreshold,repeats,1,1,unaryRepeatStride,unaryRepeatStride);
#if SBG_PARSE_FOLD_PREFIX_BRANCH_HINT && SBG_PARSE_FOLD_SUB_CY_OVERLAP
 // Add has released y1; its storage is disjoint from the in-flight Adds.
 if constexpr(TAIL)
  vreducev2((__ubuf__ uint32_t*)cy,(__ubuf__ uint32_t*)anchor,(__ubuf__ uint32_t*)anchor,3,1,2,8,0);
#endif
 AscendC::PipeBarrier<PIPE_V>();

 }
 __aicore__ __attribute__((always_inline)) inline void SelectScores() {
#if SBG_PARSE_FOLD_PREFIX_BRANCH_HINT && !SBG_PARSE_FOLD_DYNAMIC_RATIO && SBG_PARSE_FOLD_PAD_ONES
 if constexpr(!TAIL){
  vcmp_gt(cy,bound,1,1,1,1,0,0,0);
  AscendC::PipeBarrier<PIPE_V>();
  vsel(bound,scoreOnes,y2,1,1,0,0,0,0,0,0);
 }else{
#endif
 vcmpv_gt(bitmap,cy,bound,repeats,1,1,1,binaryRepeatStride,binaryRepeatStride,binaryRepeatStride);
#if SBG_PARSE_FOLD_PAD_ONES
 // Every Float x, including NaN and infinity, has false ordered x>x.
 // Only the resulting zero Scalar is consumed by mode1 Select.
 if constexpr(TAIL)
  // Both sources broadcast the same eight defined, finite Float ones.
  vcmp_gt(scoreOnes,scoreOnes,1,1,0,0,0,0,0);
 else vcmp_gt(bound,bound,1,1,1,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
#if SBG_PARSE_FOLD_MODE1_SCALAR_PITCH
 // mode1 consumes consecutive selection bits and one fixed Scalar.
 vsel(bound,scoreOnes,bitmap,repeats,1,0,1,8,0,0,static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE));
#else
 vsel(bound,scoreOnes,bitmap,repeats,1,0,1,8,0,8,static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
#else
 AscendC::PipeBarrier<PIPE_V>();
 vector_dup(y1,1.0f,1,1,1,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vcmp_gt(y1,y1,1,1,0,1,0,0,0);
 AscendC::PipeBarrier<PIPE_V>();
 vsel(bound,y1,bitmap,repeats,1,0,1,8,0,8,static_cast<uint8_t>(AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
#if SBG_PARSE_FOLD_PREFIX_BRANCH_HINT && !SBG_PARSE_FOLD_DYNAMIC_RATIO && SBG_PARSE_FOLD_PAD_ONES
 }
#endif

 }
 __aicore__ __attribute__((always_inline)) inline void CopyOutput(GM_ADDR o,uint64_t origin) {
#if SBG_PARSE_FOLD_LOCAL_PIPE_JOIN
 AscendC::PipeBarrier<PIPE_ALL>();
#else
 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
#endif
 if constexpr(TAIL)
  copy_ubuf_to_gm_align_b32((__gm__ float*)(o+origin/4ULL),bound,0,1,rows*4U,0,0,0,0);
 else copy_ubuf_to_gm((__gm__ void*)(o+origin/4ULL),(__ubuf__ void*)bound,0,1,8,0,0);

 }
};
template<uint32_t EXTRA,bool TAIL,uint32_t FIXED_WORKERS,bool CANONICAL_SINGLE>
__aicore__ inline void GeometryFoldedRawFloatEntry(GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o,int64_t owner SBG_FOLD_RATIO_PARAMETER){ const uint64_t origin=TAIL?
     (FIXED_WORKERS?static_cast<uint64_t>(FIXED_WORKERS-1U)*1024ULL:
      static_cast<uint64_t>((static_cast<uint32_t>(AscendC::GetBlockNum())-1U)*1024U)):
     (static_cast<uint64_t>(owner)-1ULL)*1024ULL;
 GeometryFoldedRawFloatState<EXTRA,TAIL,FIXED_WORKERS,CANONICAL_SINGLE> state;
 state.CopyInputs(b,a,origin);
 AscendC::DcciGMImpl<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>((__gm__ float*)t);
 float actualThreshold=*reinterpret_cast<const volatile __gm__ float*>(t);
#if SBG_PARSE_FOLD_DYNAMIC_RATIO
 state.ComputeBound(actualThreshold,dynamicRatio);
#else
 state.ComputeBound(actualThreshold);
#endif
 state.SelectScores();
 state.CopyOutput(o,origin);
}
#endif
#endif
