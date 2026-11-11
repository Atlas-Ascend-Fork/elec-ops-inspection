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
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::ParsedRoutes(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
// Select only the already-instantiated release entry before C++ parsing.
// Other resources, future unknown families and unsupported modes retain it all.
#if SBG_RAW_ONLY || SBG_COMPILE_NATIVE_ONLY
#if SBG_RAW_ONLY
#if SBG_PARSE_COMPLETE_RAW
 GeometryCompleteRawEntryState(person_boxes,anchor_points,threshold,violation_score);
#elif SBG_PARSE_COMPACT_RAW

RawCompact(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#elif SBG_PARSE_FOLDED_RAW
#if SBG_COMPILE_TILE >= 125ULL && SBG_COMPILE_TILE <= 132ULL

  FloatFoldedUpper(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#else

  FloatFoldedLower(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#endif
#endif
#elif SBG_COMPILE_NATIVE_ONLY
#if SBG_COMPILE_TILE == 19ULL || SBG_COMPILE_TILE == 20ULL

  Wide(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#elif SBG_COMPILE_TILE >= 36ULL && SBG_COMPILE_TILE <= 98ULL

NearFull(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#elif SBG_COMPILE_TILE >= 107ULL && SBG_COMPILE_TILE <= 121ULL

  HalfWide(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#elif SBG_COMPILE_TILE >= 136ULL && SBG_COMPILE_TILE <= 150ULL

  HalfSplitBytes(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#elif SBG_COMPILE_TILE >= 155ULL && SBG_COMPILE_TILE <= 178ULL

  HalfFolded(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#else

 GenericRoutes(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#endif

#endif
#elif defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM) && ((TILING_KEY_VAR & 65535ULL)==0ULL && ((TILING_KEY_VAR>>16U)&3ULL)==1ULL && ((SBG_COMPILE_TILE>=159ULL && SBG_COMPILE_TILE<=160ULL) || (SBG_COMPILE_TILE>=243ULL && SBG_COMPILE_TILE<=244ULL) SBG_UNIT_PAD_EXTRA_TILE(SBG_COMPILE_TILE)) && ((TILING_KEY_VAR>>26U)&15ULL)==15ULL)

  UnitPad(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#elif defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM) && ((TILING_KEY_VAR & 65535ULL)==0ULL && ((TILING_KEY_VAR>>16U)&3ULL)==3ULL && (SBG_COMPILE_TILE==135ULL SBG_FLOAT_FULL_EXTRA_TILE(SBG_COMPILE_TILE)) && ((TILING_KEY_VAR>>26U)&15ULL)==15ULL)

  FloatPad(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#elif defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM) && (SBG_COMPILE_HALF_PAIR && ((TILING_KEY_VAR>>16U)&3ULL)==3ULL && (SBG_COMPILE_TILE==124ULL || SBG_COMPILE_TILE==242ULL SBG_HALF_PAD_EXTRA_TILE(SBG_COMPILE_TILE)) && ((TILING_KEY_VAR>>26U)&15ULL)==15ULL)

  HalfPad(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#elif defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM) && (SBG_COMPILE_TILE>=179ULL && SBG_COMPILE_TILE<=241ULL)

  Short(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#elif defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM) && (SBG_COMPILE_TILE>=155ULL && SBG_COMPILE_TILE<=178ULL)

  HalfFolded(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#elif defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM) && (SBG_COMPILE_TILE>=99ULL && SBG_COMPILE_TILE<=154ULL)

 Folded(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#elif defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM) && (SBG_COMPILE_TILE>=27ULL && SBG_COMPILE_TILE<=35ULL)

  Unit(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#else

 GenericRoutes(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);

#endif
}
