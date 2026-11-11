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
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::GenericRoutes(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
if constexpr(IS_FLOAT_UNIT_PAD){
  UnitPad(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else if constexpr(IS_FLOAT_PAD){
  FloatPad(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else
 if constexpr(IS_HALF_PAD){
  HalfPad(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else if constexpr(IS_FLOAT_UPPER_FOLD){
  FloatFoldedUpper(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else if constexpr(IS_FLOAT_LOWER_FOLD){
  FloatFoldedLower(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else if constexpr(IS_HALF_SPLIT){
  HalfSplit(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else if constexpr(IS_HALF_WIDE){
  HalfWide(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 } else { GenericTileRoutes(person_boxes,anchor_points,threshold,violation_score,workspace,tiling); }
}
#endif

#if !SBG_RAW_ONLY
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::GenericTileRoutes(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
if constexpr(IS_SHORT){
  Short(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else
 if constexpr(IS_HALF_FOLDED){
  HalfFolded(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else if constexpr(IS_FOLDED){
 Folded(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
}else if constexpr(IS_NEAR_FULL){
NearFull(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else if constexpr(IS_UNIT){
  Unit(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else if constexpr(IS_SMALL){
  Small(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 if constexpr(IS_RAW_COMPLETE){
  GeometryCompleteRawEntryState(person_boxes,anchor_points,threshold,violation_score);
 }else
#endif
 if constexpr(IS_WIDE){
  Wide(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else if constexpr(IS_SIXTY_FOUR){
  SixtyFour(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }else{
  Generic(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 }
}
#endif
