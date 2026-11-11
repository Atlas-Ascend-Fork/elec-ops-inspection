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
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::Generic(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
AscendC::InitSocState();
  REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
  SafetyBeltViolationGeometryTilingData d;
  if constexpr(TILE_CLASS==2){
   if(AscendC::GetBlockIdx()+1<AscendC::GetBlockNum()){
    GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,3>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
   }else{
    ReadGeometryHeader<2>(tiling,d);
    GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,2>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
   }
  }else{
   constexpr uint32_t impl=TILE_CLASS==1?3:0;
   ReadGeometryHeader<impl>(tiling,d);
   GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,impl>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
  }
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::MinimumFoldFallback(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
constexpr uint32_t extra=TILE_CLASS-251U+1U+(THRESHOLD_KIND==3U?4U:0U);
  const int64_t owner=AscendC::GetBlockIdx();
  if(owner<static_cast<int64_t>(kGeometryFoldMinLaunch)){
  const uint32_t rows=owner==0?64U+extra:64U;
  const uint64_t origin=owner==0?(static_cast<uint64_t>(kGeometryFoldMinLaunch)-1ULL)*64ULL:
      (static_cast<uint64_t>(owner)-1ULL)*64ULL;
  AscendC::InitSocState();
  REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
  SafetyBeltViolationGeometryTilingData d;d.batch=rows;d.perCore=rows;
  GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,0,true,false,false,0,false,0,0,true>>(
      person_boxes+origin*16ULL,anchor_points+origin*8ULL,threshold,
      violation_score+origin*4ULL,tiling,d,0);
  }
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
template<uint32_t Kind>
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::RuntimeFallback(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
AscendC::InitSocState();
  REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
  SafetyBeltViolationGeometryTilingData d;
  ReadGeometryHeader<0>(tiling,d);
  if(static_cast<uint64_t>(AscendC::GetBlockIdx())*d.perCore<d.batch){
   GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,Kind,0U>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
  }
}
#endif

#if !(__NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM))
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::FloatFoldedFallback(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling,uint32_t extra,int64_t owner) {
// Preserve the same valid owner intervals with the original typed loop.
  const uint32_t rows=owner==0?64U+extra:64U;
  const uint64_t origin=owner==0?(static_cast<uint64_t>(AscendC::GetBlockNum())-1ULL)*64ULL:
      (static_cast<uint64_t>(owner)-1ULL)*64ULL;
  AscendC::InitSocState();
  REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
  SafetyBeltViolationGeometryTilingData d;d.batch=rows;d.perCore=rows;
  GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,0,true,false,false,0,false,0,0,true>>(
      person_boxes+origin*16ULL,anchor_points+origin*8ULL,threshold,
      violation_score+origin*4ULL,tiling,d,0);
}
#endif
