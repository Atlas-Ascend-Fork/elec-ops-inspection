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

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::HalfSplit(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
HalfCachedWide<TILE_CLASS+361U>(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::HalfWide(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
HalfCachedWide<TILE_CLASS+390U>(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::NearFull(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 if constexpr(THRESHOLD_KIND==3U && TILE_CLASS>=92U && TILE_CLASS<=98U &&
              ADDRESS_PLAN==1U && BROADCAST_PLAN==1U && FLOAT_METADATA_PLAN==3U &&
              AscendC::IsSameType<DT_PERSON_BOXES,float>::value && AscendC::IsSameType<DT_OUTPUT,float>::value){
RawCompact(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else
#endif
 if constexpr(THRESHOLD_KIND==3 && AscendC::IsSameType<DT_PERSON_BOXES,float>::value &&
              AscendC::IsSameType<DT_OUTPUT,float>::value){
  FloatNearFull(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else if constexpr(GeometryHalfWideOwnerPolicy<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS>::enabled){
  AscendC::InitSocState();
  REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
  SafetyBeltViolationGeometryTilingData d;
  // Cache the full SDK mapping, including subblock dimensions and index.
  const int64_t cachedOwner=AscendC::GetBlockIdx();
  if(cachedOwner==0){
   GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,71,true,false,true,0,false,TILE_CLASS+413>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,cachedOwner);
  }else{
   GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,70,true,false,true>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,cachedOwner);
  }
 }else{
  if constexpr(!GeometryCopyFirstPolicy<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS,ADDRESS_PLAN>::enabled)
   AscendC::InitSocState();
  REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
  SafetyBeltViolationGeometryTilingData d;
  if(AscendC::GetBlockIdx()==0){
   GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,71,true,false,true,0,false,TILE_CLASS+413>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
  }else{
   GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,70,true,false,true>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
  }
 }
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::Wide(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
if constexpr(!GeometryCopyFirstPolicy<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS,ADDRESS_PLAN>::enabled)
   AscendC::InitSocState();
  REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
  SafetyBeltViolationGeometryTilingData d;
  if constexpr(TILE_CLASS==20){
   // Start the metadata-dependent tail owner first; all GM intervals stay disjoint.
   if(AscendC::GetBlockIdx()!=0){
    GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,70,true,false,true>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
   }else{
    ReadGeometryHeader<71>(tiling,d);
    GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,71,true,false,true>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
   }
  }else{
   GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,70,true,false,false,0,
            AscendC::IsSameType<DT_PERSON_BOXES,half>::value>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
  }
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::HalfSplitBytes(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
GeometryWideSocState<true,false,false>();
  REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
  SafetyBeltViolationGeometryTilingData d;
  const uint64_t rawOwner=static_cast<uint64_t>(get_block_idx());
  const uint64_t subOwner=static_cast<uint64_t>(AscendC::GetSubBlockIdx());
  if((rawOwner|subOwner)==0U){
   GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,71,true,false,true,0,false,TILE_CLASS+361U>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,0);
  }else{
   const int64_t owner=static_cast<int64_t>(rawOwner*static_cast<uint64_t>(AscendC::GetTaskRation())+subOwner);
   GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,70,true,false,true>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,owner);
  }
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::FloatNearFull(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
if constexpr(GeometryCopyFirstPolicy<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS,ADDRESS_PLAN>::enabled){
   REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
   SafetyBeltViolationGeometryTilingData d;
   const int64_t rawOwner=get_block_idx();
   const int64_t subOwner=AscendC::GetSubBlockIdx();
   // For positive SDK task ratios, logical owner0 is exactly raw0/subid0.
   // The constant tail origin needs no numerical owner calculation.
   if(__builtin_expect((rawOwner|subOwner)==0,1)){
    GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,71,true,false,true,0,false,TILE_CLASS+413>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,0);
   }else{
    const int64_t cachedOwner=rawOwner*AscendC::GetTaskRation()+subOwner;
    GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,70,true,false,true>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,cachedOwner);
   }
  }else{
  const int64_t cachedOwner=AscendC::GetBlockIdx();
  if constexpr(!GeometryCopyFirstPolicy<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS,ADDRESS_PLAN>::enabled)
   AscendC::InitSocState();
  REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
  SafetyBeltViolationGeometryTilingData d;
  if(__builtin_expect(cachedOwner==0,1)){
   GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,71,true,false,true,0,false,TILE_CLASS+413>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,cachedOwner);
  }else{
   GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,70,true,false,true>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,cachedOwner);
  }
  }
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
template<uint32_t TailRows>
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::HalfCachedWide(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
GeometryWideSocState<true,false,false>();
  REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
  SafetyBeltViolationGeometryTilingData d;
  const int64_t owner=AscendC::GetBlockIdx();
  if(owner==0){
   GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,71,true,false,true,0,false,TailRows>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,owner);
  }else{
   GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,70,true,false,true>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,owner);
  }
}
#endif
