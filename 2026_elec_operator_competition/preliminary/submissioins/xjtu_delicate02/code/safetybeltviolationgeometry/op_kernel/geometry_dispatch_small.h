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
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::UnitPad(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
#if SBG_UNIT_PAD_COHORT
  constexpr uint32_t workers=(TILE_CLASS>=248U && TILE_CLASS<=249U)?TILE_CLASS-248U+1U:
      (TILE_CLASS<=160U?TILE_CLASS-159U+1U:TILE_CLASS-243U+1U);
#else
  constexpr uint32_t workers=TILE_CLASS<=160U?TILE_CLASS-159U+1U:TILE_CLASS-243U+1U;
#endif
#if SBG_PARSE_UNIT_RAW_ZERO_OWNER
  static_assert(workers==1U,"Single logical output owner");
  const int64_t owner=0;
  if(GeometryShortActualZeroOwner()){
#else
  const int64_t owner=AscendC::GetBlockIdx();
  if(owner<workers){
#endif
#if !SBG_PARSE_UNIT_COPY_FIRST
   GeometryWideSocState<true,false,false>();
#endif
   REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
   SafetyBeltViolationGeometryTilingData d;
   GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,84,true,false>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,workers==1U?0:owner);
  }
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::FloatPad(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
// Only low64 bits are consumed by these Float operations. The retained SDK
  // Brcb later restores its own full64 mask before dependent waist math.
  GeometryWideSocState<GeometryFloatFullNorm64Policy<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS,FLOAT_METADATA_PLAN>::enabled,true,false>();
  REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
  SafetyBeltViolationGeometryTilingData d;
  GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,68,true,false,false,0,false,0,0,false,false,64,0,false,false,false>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::HalfPad(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
if constexpr(!GeometryHalfFullInitPolicy<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS,BROADCAST_PLAN>::enabled)
   AscendC::InitSocState();
  REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
  SafetyBeltViolationGeometryTilingData d;
  GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,68,true,false,false,0,false,0,0,false,false,64,0,false,true,false>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::Short(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
constexpr uint32_t shortRows=TILE_CLASS-179+1;
  if constexpr(GeometryFloatShortNorm64Policy<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS>::enabled && shortRows>32U){
#if SBG_PARSE_FLOAT_SHORT_POSITIVE_ZERO
   const uint64_t rawOwner=static_cast<uint64_t>(get_block_idx());
   // Positive real task ratio makes every raw block above one inactive.
   if(__builtin_expect(rawOwner<2ULL,1)){
    uint64_t logicalOwner=rawOwner;
    const uint64_t dimension=static_cast<uint64_t>(get_subblockdim());
    if(__builtin_expect(dimension!=1ULL,0)){
     const uint64_t subOwner=static_cast<uint64_t>(AscendC::GetSubBlockIdx());
     logicalOwner=rawOwner*dimension+subOwner;
     if(logicalOwner>=2ULL)goto sbg_short_checked_idle;
    }
    ShortParts(person_boxes,anchor_points,threshold,violation_score,workspace,tiling,logicalOwner);
   }
sbg_short_checked_idle:;

#else
   const int64_t logicalOwner=AscendC::GetBlockIdx();
   if(logicalOwner<2){
    ShortParts(person_boxes,anchor_points,threshold,violation_score,workspace,tiling,logicalOwner);
   }
#endif
  }else{
#if SBG_PARSE_HALF_SHORT_RAW_ZERO_OWNER
  if(GeometryShortActualZeroOwner()){
#else
  if(AscendC::GetBlockIdx()==0){
#endif
   if constexpr(!SBG_PARSE_HALF_SHORT_FOUR_FIELD) GeometryWideSocState<GeometryFloatShortNorm64Policy<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS>::enabled,true>();
   REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
   SafetyBeltViolationGeometryTilingData d;
   constexpr uint32_t rows=TILE_CLASS-179+1;
   GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,69,true,true,false,1,false,0,0,false,false,64,rows>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
  }

  }
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::Unit(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
constexpr uint32_t workers=TILE_CLASS<35?TILE_CLASS-26:0;
  if constexpr(ADDRESS_PLAN==1){
  const int64_t cachedOwner=AscendC::GetBlockIdx();
  if(workers==0 || cachedOwner<workers){
   AscendC::InitSocState();
   REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
   SafetyBeltViolationGeometryTilingData d;
   GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,84,true,
            AscendC::IsSameType<DT_PERSON_BOXES,half>::value>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,workers==1?0:cachedOwner);
  }
  }else if constexpr(GeometryHalfUnitOwnerPolicy<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS>::enabled){
  // Only the zero owner writes the existing single-owner tile.
#if SBG_PARSE_HALF_UNIT_RAW_ZERO_OWNER
  static_assert(workers==1U,"Single logical output owner");
  const int64_t cachedOwner=0;
  if(GeometryShortActualZeroOwner()){
#else
  const int64_t cachedOwner=AscendC::GetBlockIdx();
  if(workers==0 || cachedOwner<workers){
#endif
   if constexpr(!SBG_PARSE_HALF_UNIT_SINGLE_CTRL)AscendC::InitSocState();
   REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
   SafetyBeltViolationGeometryTilingData d;
   GeometryExecute<CachedOwnerGeometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,84,true,
            AscendC::IsSameType<DT_PERSON_BOXES,half>::value>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d,workers==1?0:cachedOwner);
  }
  }else{
  if(workers==0 || AscendC::GetBlockIdx()<workers){
   AscendC::InitSocState();
   REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
   SafetyBeltViolationGeometryTilingData d;
   GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,84,true,
            AscendC::IsSameType<DT_PERSON_BOXES,half>::value>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
  }
  }
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::Small(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
constexpr uint32_t workers=(TILE_CLASS-23)/2+1;
  constexpr bool hasTail=(TILE_CLASS-23)%2!=0;
  if(AscendC::GetBlockIdx()<workers){
   AscendC::InitSocState();
   REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
   SafetyBeltViolationGeometryTilingData d;
   if constexpr(hasTail){
    if(AscendC::GetBlockIdx()==0){
     ReadGeometryHeader<82>(tiling,d);
     GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,82,true,AscendC::IsSameType<DT_PERSON_BOXES,half>::value,false,(workers>1?workers:0)>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
    }else{
     GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,81,true,AscendC::IsSameType<DT_PERSON_BOXES,half>::value,false,(workers>1?workers:0)>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
    }
   }else{
    GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,81>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
   }
  }
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::SixtyFour(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
constexpr uint32_t workers=(TILE_CLASS-3)/2+1;
  constexpr bool hasTail=(TILE_CLASS-3)%2!=0;
  if constexpr(!hasTail && workers>1){
   constexpr bool floatState=GeometryFloatFullNorm64Policy<DT_PERSON_BOXES,DT_OUTPUT,
       THRESHOLD_KIND,TILE_CLASS,FLOAT_METADATA_PLAN>::enabled;
   constexpr bool halfState=GeometryHalfFullInitPolicy<DT_PERSON_BOXES,DT_OUTPUT,
       THRESHOLD_KIND,TILE_CLASS,BROADCAST_PLAN>::enabled;
   // Read-only input DMA can start before complete Half state setup.
   if constexpr(!halfState)GeometryWideSocState<floatState || halfState,floatState>();
   REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
   SafetyBeltViolationGeometryTilingData d;
   GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,68,true,false,false,0,false,0,0,false,false,64,0,false,(BROADCAST_PLAN!=0),(FLOAT_METADATA_PLAN!=0)>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
  }else if constexpr(hasTail && workers>1 && AscendC::IsSameType<DT_PERSON_BOXES,float>::value){
   if constexpr(workers<8){if(AscendC::GetBlockIdx()>=workers)return;}
   AscendC::InitSocState();
   REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
   SafetyBeltViolationGeometryTilingData d;
   if(AscendC::GetBlockIdx()+1<workers){
    GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,68,true,false,false,0,false,0,0,false,false,64,0,false,(BROADCAST_PLAN!=0),(FLOAT_METADATA_PLAN!=0)>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
   }else{
    ReadGeometryHeader<69,true>(tiling,d);
    GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,69,false>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
   }
  }else if(AscendC::GetBlockIdx()<workers){
   AscendC::InitSocState();
   REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
   SafetyBeltViolationGeometryTilingData d;
   if constexpr(hasTail){
    if(AscendC::GetBlockIdx()==0){
     ReadGeometryHeader<69>(tiling,d);
     GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,69,true,AscendC::IsSameType<DT_PERSON_BOXES,half>::value,false,(workers>1?workers:0)>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
    }else{
     GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,68,true,AscendC::IsSameType<DT_PERSON_BOXES,half>::value,false,(workers>1?workers:0)>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
    }
   }else{
    GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,68,true,false,false,0,false,0,0,false,false,64,0,false,(BROADCAST_PLAN!=0),(FLOAT_METADATA_PLAN!=0)>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
   }
  }
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
template<typename Owner>
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::ShortParts(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling,Owner logicalOwner) {
constexpr uint32_t shortRows=TILE_CLASS-179+1;
GeometryWideSocState<true,false,false>();
    REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
    SafetyBeltViolationGeometryTilingData d;
    if(logicalOwner==0ULL){
     GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,69,true,true,false,1,false,0,0,false,false,64,32>>(
      person_boxes,anchor_points,threshold,violation_score,tiling,d);
    }else{
     GeometryExecute<Geometry<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,69,true,true,false,1,false,0,0,false,false,64,shortRows-32>>(
      person_boxes+32U*4U*sizeof(DT_PERSON_BOXES),
         anchor_points+32U*2U*sizeof(DT_PERSON_BOXES),threshold,
         violation_score+32U*sizeof(DT_OUTPUT),tiling,d);
    }
}
#endif
