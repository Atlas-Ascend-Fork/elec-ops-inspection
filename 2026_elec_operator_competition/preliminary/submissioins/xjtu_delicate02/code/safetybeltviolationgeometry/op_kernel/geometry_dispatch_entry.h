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
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::Run(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
#if SBG_PARSE_HALF_UNIT_NESTED_ZERO_GUARD
 const uint64_t rawOwner=static_cast<uint64_t>(get_block_idx());
 if(rawOwner==0ULL){
  const uint64_t subOwner=static_cast<uint64_t>(AscendC::GetSubBlockIdx());
  if(subOwner==0ULL)
   GeometryHalfUnitNativeDma(person_boxes,anchor_points,threshold,violation_score);
 }
#elif SBG_PARSE_HALF_UNIT_NATIVE_DMA
 if(GeometryShortActualZeroOwner())
  GeometryHalfUnitNativeDma(person_boxes,anchor_points,threshold,violation_score);
#else
 FloatBatchRoutes(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
#endif
}

#if !SBG_PARSE_FOLD_MINIMUM
SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::MinimumRoutes(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
#if SBG_FOLD_MINIMUM_COHORT
 if constexpr(IS_DYNAMIC_FOLD) {
  RuntimeFallback<1U>(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 } else if constexpr(IS_MINIMUM_FOLD) {
  MinimumFoldFallback(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
 } else
#endif
 { ParsedRoutes(person_boxes,anchor_points,threshold,violation_score,workspace,tiling); }
}
#endif

SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::SmallRawRoutes(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
#if SBG_PARSE_HALF_SHORT_LITERAL_BITS
 if(GeometryShortActualZeroOwner())
  GeometryHalfShortLiteralBits<TILE_CLASS-178U>(person_boxes,anchor_points,threshold,violation_score);
#elif SBG_PARSE_HALF_SCALAR_BITS_DUPLICATE
 GeometryHalfScalarBitsDuplicate(person_boxes,anchor_points,threshold,violation_score);
#elif SBG_PARSE_FLOAT_FULL_SCALAR_BITMAP_SCORE
 GeometryFloatFullScalarBitmapScore(person_boxes,anchor_points,threshold,violation_score);
#elif SBG_PARSE_HALF_SHARED_ONE_PAD_MERGED
 GeometryHalfSharedOnePadMerged(person_boxes,anchor_points,threshold,violation_score);
#elif SBG_PARSE_FOLD_MINIMUM
 MinimumFold(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
#else
 MinimumRoutes(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
#endif
}

SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::HalfBatchRoutes(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
#if SBG_PARSE_HALF_BATCH_MERGE
 const uint64_t rawOwner=static_cast<uint64_t>(get_block_idx());
 const uint64_t subOwner=static_cast<uint64_t>(AscendC::GetSubBlockIdx());
 if((rawOwner|subOwner)==0ULL)
  GeometryHalfBatchMerge<(REPEAT_ADDRESS_PLAN-50U)*64U>(person_boxes,anchor_points,threshold,violation_score);
#else
 if constexpr(IS_HALF_BATCH) { RuntimeFallback<2U>(person_boxes,anchor_points,threshold,violation_score,workspace,tiling); }
 else { SmallRawRoutes(person_boxes,anchor_points,threshold,violation_score,workspace,tiling); }
#endif
}

SBG_DISPATCH_TEMPLATE
__aicore__ __attribute__((always_inline)) inline void SBG_DISPATCH_TYPE::FloatBatchRoutes(
    GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
#if SBG_PARSE_FLOAT_BATCH_MERGE
 const uint64_t rawOwner=static_cast<uint64_t>(get_block_idx());
 const uint64_t subOwner=static_cast<uint64_t>(AscendC::GetSubBlockIdx());
 if((rawOwner|subOwner)==0ULL)
  GeometryFloatBatchMerge<(TILE_CLASS-62U)*64U>(person_boxes,anchor_points,threshold,violation_score);
#else
 if constexpr(IS_FLOAT_BATCH) { RuntimeFallback<1U>(person_boxes,anchor_points,threshold,violation_score,workspace,tiling); }
 else { HalfBatchRoutes(person_boxes,anchor_points,threshold,violation_score,workspace,tiling); }
#endif
}
