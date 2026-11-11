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

SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::FloatFoldedUpper(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
constexpr uint32_t extra=TILE_CLASS-125U+1U;
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
  // This host-selected family has4..7 real logical owners. The original
  // nonnegative raw*ratio+sub value and all intermediates fit signed32.
  const int32_t rawOwner=static_cast<int32_t>(get_block_idx());
  const int32_t taskRatio=static_cast<int32_t>(AscendC::GetTaskRation());
  const int32_t subOwner=static_cast<int32_t>(AscendC::GetSubBlockIdx());
  const int32_t logicalOwner=rawOwner*taskRatio+subOwner;
  const int64_t owner=logicalOwner;
#else
  const int64_t owner=AscendC::GetBlockIdx();
#endif
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
  if(__builtin_expect(owner==0,1)){
   const uint32_t blocks=static_cast<uint32_t>(AscendC::GetBlockNum());
   if(__builtin_expect(blocks==kGeometryFoldMinLaunch,1))
    GeometryFoldedRawFloatEntry<extra,true,kGeometryFoldMinLaunch>(person_boxes,anchor_points,threshold,violation_score,owner);
   else GeometryFoldedRawFloatEntry<extra,true>(person_boxes,anchor_points,threshold,violation_score,owner);
  }else GeometryFoldedRawFloatEntry<extra,false,0U,true>(person_boxes,anchor_points,threshold,violation_score,owner);
#else
  FloatFoldedFallback(person_boxes,anchor_points,threshold,violation_score,workspace,tiling,extra,owner);
#endif
}

SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::FloatFoldedLower(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
constexpr uint32_t extra=TILE_CLASS-99U+1U;
  const int64_t owner=AscendC::GetBlockIdx();
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
  if(__builtin_expect(owner==0,1)){
   const uint32_t blocks=static_cast<uint32_t>(AscendC::GetBlockNum());
   if(__builtin_expect(blocks==kGeometryFoldMinLaunch,1))
    GeometryFoldedRawFloatEntry<extra,true,kGeometryFoldMinLaunch>(person_boxes,anchor_points,threshold,violation_score,owner);
   else GeometryFoldedRawFloatEntry<extra,true>(person_boxes,anchor_points,threshold,violation_score,owner);
  }else GeometryFoldedRawFloatEntry<extra,false>(person_boxes,anchor_points,threshold,violation_score,owner);
#else
  FloatFoldedFallback(person_boxes,anchor_points,threshold,violation_score,workspace,tiling,extra,owner);
#endif
}

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::HalfFolded(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
if constexpr(ADDRESS_PLAN==1 && TILE_CLASS<=162 && THRESHOLD_KIND==3 &&
       AscendC::IsSameType<DT_PERSON_BOXES,half>::value && AscendC::IsSameType<DT_OUTPUT,half>::value){
   if(AscendC::GetBlockIdx()==0){
    if constexpr(!GeometryHalfFoldInputsFirstPolicy<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS,ADDRESS_PLAN>::enabled)
     AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
    SafetyBeltViolationGeometryTilingData d;
    constexpr uint32_t rows=TILE_CLASS-155U+1U;
    GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,85,true,false,false,
             1,false,0,rows,true,false,256>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
   }
  }else{

  constexpr uint32_t workers=(TILE_CLASS-155)/8+1;
  constexpr uint32_t rows=(TILE_CLASS-155)%8+1;
  if(AscendC::GetBlockIdx()<workers){
   AscendC::InitSocState();
   REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
   SafetyBeltViolationGeometryTilingData d;
   if(AscendC::GetBlockIdx()==0){
    GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,85,true,false,false,
             workers,false,0,rows,(THRESHOLD_KIND==3),false,128>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
   }else{
    GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,86,true,false,false,workers,false,0,0,(THRESHOLD_KIND==3)>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
   }
  }

  }
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::Folded(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
if constexpr((AscendC::IsSameType<DT_PERSON_BOXES,float>::value && AscendC::IsSameType<DT_OUTPUT,float>::value)){
 const int64_t cachedOwner=AscendC::GetBlockIdx();

  constexpr uint32_t workers=(TILE_CLASS-99)/8+1;
  constexpr uint32_t rows=(TILE_CLASS-99)%8+1;
  constexpr bool midpoint=AscendC::IsSameType<DT_PERSON_BOXES,float>::value;
  constexpr bool lateNarrow=false;
  if(cachedOwner<workers){
   if constexpr(!GeometryCopyFirstPolicy<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS,ADDRESS_PLAN>::enabled)
    AscendC::InitSocState();
   REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
   SafetyBeltViolationGeometryTilingData d;
   if(cachedOwner==0){
    GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,85,true,false,false,workers,false,0,rows,midpoint,lateNarrow>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,cachedOwner);
   }else{
    GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,68,true,
             AscendC::IsSameType<DT_PERSON_BOXES,half>::value,false,workers,false,0,0,midpoint,lateNarrow>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,cachedOwner);
   }
  }

 }else{
  constexpr uint32_t workers=(TILE_CLASS-99)/8+1;
  constexpr uint32_t rows=(TILE_CLASS-99)%8+1;
  constexpr bool midpoint=AscendC::IsSameType<DT_PERSON_BOXES,float>::value;
  constexpr bool lateNarrow=false;
  if(AscendC::GetBlockIdx()<workers){
   AscendC::InitSocState();
   REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
   SafetyBeltViolationGeometryTilingData d;
   if(AscendC::GetBlockIdx()==0){
    GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,85,true,false,false,workers,false,0,rows,midpoint,lateNarrow>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
   }else{
    GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,68,true,
             AscendC::IsSameType<DT_PERSON_BOXES,half>::value,false,workers,false,0,0,midpoint,lateNarrow>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
   }
  }

 }
}
#endif
