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

#define SBG_DISPATCH_TEMPLATE template<typename DT_PERSON_BOXES,typename DT_OUTPUT,uint32_t THRESHOLD_KIND,uint32_t TILE_CLASS,uint32_t ADDRESS_PLAN,uint32_t BROADCAST_PLAN,uint32_t FLOAT_METADATA_PLAN,uint32_t REPEAT_ADDRESS_PLAN>
#define SBG_DISPATCH_TYPE GeometryDispatch<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS,ADDRESS_PLAN,BROADCAST_PLAN,FLOAT_METADATA_PLAN,REPEAT_ADDRESS_PLAN>

SBG_DISPATCH_TEMPLATE
struct GeometryDispatch {
 static constexpr bool IS_FLOAT_BATCH = AscendC::IsSameType<DT_PERSON_BOXES,float>::value && AscendC::IsSameType<DT_OUTPUT,float>::value &&
              THRESHOLD_KIND==3U && TILE_CLASS>=64U && TILE_CLASS<=70U && ADDRESS_PLAN==1U &&
              BROADCAST_PLAN==1U && FLOAT_METADATA_PLAN==3U && (REPEAT_ADDRESS_PLAN==59U || REPEAT_ADDRESS_PLAN==63U || REPEAT_ADDRESS_PLAN==66U);
 static constexpr bool IS_HALF_BATCH = AscendC::IsSameType<DT_PERSON_BOXES,half>::value && AscendC::IsSameType<DT_OUTPUT,half>::value &&
              THRESHOLD_KIND==3U && TILE_CLASS==255U && ADDRESS_PLAN==1U && BROADCAST_PLAN==1U &&
              FLOAT_METADATA_PLAN==3U && REPEAT_ADDRESS_PLAN>=52U && REPEAT_ADDRESS_PLAN<=58U;
 static constexpr bool IS_DYNAMIC_FOLD = AscendC::IsSameType<DT_PERSON_BOXES,float>::value && AscendC::IsSameType<DT_OUTPUT,float>::value &&
              REPEAT_ADDRESS_PLAN>=69U && REPEAT_ADDRESS_PLAN<=75U &&
              (THRESHOLD_KIND==1U || THRESHOLD_KIND==3U) && FLOAT_METADATA_PLAN==3U &&
              ADDRESS_PLAN==1U && BROADCAST_PLAN==1U && TILE_CLASS>=251U && TILE_CLASS<=254U;
 static constexpr bool IS_MINIMUM_FOLD = AscendC::IsSameType<DT_PERSON_BOXES,float>::value && AscendC::IsSameType<DT_OUTPUT,float>::value &&
              (THRESHOLD_KIND==1U || THRESHOLD_KIND==3U) && FLOAT_METADATA_PLAN==3U &&
              ADDRESS_PLAN==1U && BROADCAST_PLAN==1U && TILE_CLASS>=251U && TILE_CLASS<=254U;
 static constexpr bool IS_FLOAT_UNIT_PAD = AscendC::IsSameType<DT_PERSON_BOXES,float>::value && AscendC::IsSameType<DT_OUTPUT,float>::value &&
              THRESHOLD_KIND==1U && FLOAT_METADATA_PLAN==3U && ADDRESS_PLAN==1U && BROADCAST_PLAN==1U &&
              ((TILE_CLASS>=159U && TILE_CLASS<=160U) || (TILE_CLASS>=243U && TILE_CLASS<=244U) SBG_UNIT_PAD_EXTRA_TILE(TILE_CLASS));
 static constexpr bool IS_FLOAT_PAD = AscendC::IsSameType<DT_PERSON_BOXES,float>::value && AscendC::IsSameType<DT_OUTPUT,float>::value &&
              THRESHOLD_KIND==3U && (TILE_CLASS==135U SBG_FLOAT_FULL_EXTRA_TILE(TILE_CLASS)) && ADDRESS_PLAN==1U && BROADCAST_PLAN==1U && FLOAT_METADATA_PLAN==3U;
 static constexpr bool IS_HALF_PAD = AscendC::IsSameType<DT_PERSON_BOXES,half>::value && AscendC::IsSameType<DT_OUTPUT,half>::value &&
              THRESHOLD_KIND==3U && (TILE_CLASS==124U || TILE_CLASS==242U SBG_HALF_PAD_EXTRA_TILE(TILE_CLASS)) && ADDRESS_PLAN==1U && BROADCAST_PLAN==1U && FLOAT_METADATA_PLAN==3U;
 static constexpr bool IS_FLOAT_UPPER_FOLD = AscendC::IsSameType<DT_PERSON_BOXES,float>::value && AscendC::IsSameType<DT_OUTPUT,float>::value &&
              THRESHOLD_KIND==1U && FLOAT_METADATA_PLAN==3U && ADDRESS_PLAN==1U && BROADCAST_PLAN==1U &&
              TILE_CLASS>=125U && TILE_CLASS<=132U;
 static constexpr bool IS_FLOAT_LOWER_FOLD = AscendC::IsSameType<DT_PERSON_BOXES,float>::value && AscendC::IsSameType<DT_OUTPUT,float>::value &&
              THRESHOLD_KIND==1U && FLOAT_METADATA_PLAN==3U && ADDRESS_PLAN==1U && BROADCAST_PLAN==1U &&
              TILE_CLASS>=99U && TILE_CLASS<=106U;
 static constexpr bool IS_HALF_SPLIT = AscendC::IsSameType<DT_PERSON_BOXES,half>::value && AscendC::IsSameType<DT_OUTPUT,half>::value &&
              THRESHOLD_KIND==3U && FLOAT_METADATA_PLAN==3U && ADDRESS_PLAN==1U && BROADCAST_PLAN==1U &&
              TILE_CLASS>=136U && TILE_CLASS<=150U;
 static constexpr bool IS_HALF_WIDE = AscendC::IsSameType<DT_PERSON_BOXES,half>::value && AscendC::IsSameType<DT_OUTPUT,half>::value &&
              THRESHOLD_KIND==3U && FLOAT_METADATA_PLAN==3U && ADDRESS_PLAN==1U && BROADCAST_PLAN==1U &&
              TILE_CLASS>=107U && TILE_CLASS<=121U;
 static constexpr bool IS_SHORT = TILE_CLASS>=179 && TILE_CLASS<=241;
 static constexpr bool IS_HALF_FOLDED = TILE_CLASS>=155 && TILE_CLASS<=178;
 static constexpr bool IS_FOLDED = TILE_CLASS>=99 && TILE_CLASS<=154;
 static constexpr bool IS_NEAR_FULL = TILE_CLASS>=36 && TILE_CLASS<=98;
 static constexpr bool IS_UNIT = TILE_CLASS>=27 && TILE_CLASS<=35;
 static constexpr bool IS_SMALL = TILE_CLASS>=23 && TILE_CLASS<=26;
 static constexpr bool IS_RAW_COMPLETE = TILE_CLASS==19U && THRESHOLD_KIND==3U && ADDRESS_PLAN==0U && BROADCAST_PLAN==1U &&
              FLOAT_METADATA_PLAN==2U && AscendC::IsSameType<DT_PERSON_BOXES,float>::value && AscendC::IsSameType<DT_OUTPUT,float>::value;
 static constexpr bool IS_WIDE = TILE_CLASS==19 || TILE_CLASS==20;
 static constexpr bool IS_SIXTY_FOUR = TILE_CLASS>=3;
 static __aicore__ __attribute__((always_inline)) inline void UnitPad(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void FloatPad(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void HalfPad(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void FloatFoldedUpper(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void FloatFoldedLower(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void HalfSplit(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void HalfWide(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void Short(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void HalfFolded(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void Folded(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void NearFull(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void Unit(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void Small(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void Wide(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void SixtyFour(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void Generic(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void RawCompact(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void HalfSplitBytes(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void GenericRoutes(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void Run(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void FloatNearFull(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 template<typename Owner>
 static __aicore__ __attribute__((always_inline)) inline void ShortParts(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling,Owner logicalOwner);
 static __aicore__ __attribute__((always_inline)) inline void ParsedRoutes(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void MinimumFold(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void MinimumFoldFallback(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 template<uint32_t Kind>
 static __aicore__ __attribute__((always_inline)) inline void RuntimeFallback(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void MinimumRoutes(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void SmallRawRoutes(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void HalfBatchRoutes(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void FloatBatchRoutes(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void GenericTileRoutes(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
 static __aicore__ __attribute__((always_inline)) inline void FloatFoldedFallback(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling,uint32_t extra,int64_t owner);
 template<uint32_t TailRows>
 static __aicore__ __attribute__((always_inline)) inline void HalfCachedWide(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling);
};
