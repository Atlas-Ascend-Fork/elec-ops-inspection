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
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::RawCompact(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
#if SBG_PARSE_FLOAT_NEARFULL_RAW_OWNER
  const uint64_t rawOwner=static_cast<uint64_t>(get_block_idx());
#else
  const int64_t rawOwner=get_block_idx();
#endif
  const int64_t subOwner=AscendC::GetSubBlockIdx();
  if(__builtin_expect((rawOwner|subOwner)==0,1)){
   GeometryCompactEntryFloatPair<TILE_CLASS+413U,true>(person_boxes,anchor_points,threshold,violation_score,0);
  }else{
   const int64_t logicalOwner=rawOwner*AscendC::GetTaskRation()+subOwner;
   GeometryCompactEntryFloatPair<512U,false>(person_boxes,anchor_points,threshold,violation_score,logicalOwner);
  }
}
#endif

#if SBG_PARSE_FOLD_MINIMUM
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::MinimumFold(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
constexpr uint32_t extra=TILE_CLASS-251U+1U+(THRESHOLD_KIND==3U?4U:0U);
#if SBG_PARSE_FOLD_DYNAMIC_RATIO
 constexpr uint32_t activeWorkers=REPEAT_ADDRESS_PLAN-68U;
 const uint64_t rawOwner=static_cast<uint64_t>(get_block_idx());
 const uint64_t subOwner=static_cast<uint64_t>(AscendC::GetSubBlockIdx());
 const uint64_t owner=rawOwner*static_cast<uint64_t>(AscendC::GetTaskRation())+subOwner;
 if(owner<activeWorkers){
  auto ratioAddress=(__gm__ float*)(tiling+__builtin_offsetof(SafetyBeltViolationGeometryTilingData,waistRatio));
  AscendC::DcciGMImpl<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(ratioAddress);
  const float dynamicRatio=*reinterpret_cast<const volatile __gm__ float*>(ratioAddress);
  if(owner==0U)
   GeometryFoldedRawFloatEntry<extra,true,activeWorkers>(person_boxes,anchor_points,threshold,violation_score,0,dynamicRatio);
  else
   GeometryFoldedRawFloatEntry<extra,false,0U,true>(person_boxes,anchor_points,threshold,violation_score,static_cast<int64_t>(owner),dynamicRatio);
 }
#else
#if SBG_PARSE_FOLD_MIN_LAZY_ZERO_OWNER
 const uint64_t rawOwner=static_cast<uint64_t>(get_block_idx());
 const uint64_t subOwner=static_cast<uint64_t>(AscendC::GetSubBlockIdx());
 // A positive task ratio and nonnegative fields make logical zero exact.
 if(__builtin_expect((rawOwner|subOwner)==0ULL,SBG_PARSE_FOLD_PREFIX_BRANCH_HINT?0:1)){
  GeometryFoldedRawFloatEntry<extra,true,kGeometryFoldMinLaunch>(person_boxes,anchor_points,threshold,violation_score,0);
 }else{
  const uint64_t taskRatio=static_cast<uint64_t>(AscendC::GetTaskRation());
  const int64_t owner=static_cast<int64_t>(rawOwner*taskRatio+subOwner);
  if(owner<static_cast<int64_t>(kGeometryFoldMinLaunch))
   GeometryFoldedRawFloatEntry<extra,false,0U,true>(person_boxes,anchor_points,threshold,violation_score,owner);
 }
#else
 const int32_t rawOwner=static_cast<int32_t>(get_block_idx());
 const int32_t taskRatio=static_cast<int32_t>(AscendC::GetTaskRation());
 const int32_t subOwner=static_cast<int32_t>(AscendC::GetSubBlockIdx());
 const int64_t owner=static_cast<int64_t>(rawOwner*taskRatio+subOwner);
 if(__builtin_expect(owner==0,1))
  GeometryFoldedRawFloatEntry<extra,true,kGeometryFoldMinLaunch>(person_boxes,anchor_points,threshold,violation_score,owner);
 else if(owner<static_cast<int64_t>(kGeometryFoldMinLaunch))
  GeometryFoldedRawFloatEntry<extra,false,0U,true>(person_boxes,anchor_points,threshold,violation_score,owner);
#endif
#endif
}
#endif
