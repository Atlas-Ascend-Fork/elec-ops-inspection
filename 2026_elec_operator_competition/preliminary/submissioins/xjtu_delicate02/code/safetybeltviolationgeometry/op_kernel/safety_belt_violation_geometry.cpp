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

#include "kernel_operator.h"
#include "safety_belt_violation_geometry_tiling.h"
#include "tiling_key_safety_belt_violation_geometry.h"
#include "geometry_compile_policy.h"

template<class T,class O,uint32_t TH,uint32_t TILE,uint32_t PLAN>
struct GeometryCopyFirstPolicy {
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool enabled=(AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value &&
     ((TH==1U && TILE>=99U && TILE<=154U) || (TH==3U && (TILE==19U || (TILE>=36U && TILE<=98U))))) ||
     (AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value && TH==3U && TILE==19U) ||
     (SBG_PARSE_HALF_SHORT_FOUR_FIELD && AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value && TH==2U && TILE>=211U && TILE<=241U);
#else
 static constexpr bool enabled=false;
#endif
};
template<class T,class O,uint32_t TH,uint32_t TILE,uint32_t ADDR>
struct GeometryHalfFoldInputsFirstPolicy {
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool enabled=AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value &&
     TH==3U && TILE>=155U && TILE<=162U && ADDR==1U;
#else
 static constexpr bool enabled=false;
#endif
};
template<class T,class O,uint32_t TH,uint32_t TILE,uint32_t META>
struct GeometryFloatFullNorm64Policy {
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool enabled=AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value &&
     ((TILE==5U && ((TH==1U && META==1U) || (TH==3U && META==0U))) ||
      ((TILE==135U SBG_FLOAT_FULL_EXTRA_TILE(TILE)) && TH==3U && META==3U));
#else
 static constexpr bool enabled=false;
#endif
};
template<class T,class O,uint32_t TH,uint32_t TILE,uint32_t BROADCAST>
struct GeometryHalfFullInitPolicy {
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool enabled=AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value &&
     TH==3U && (TILE==5U || TILE==124U || TILE==242U SBG_HALF_PAD_EXTRA_TILE(TILE)) && BROADCAST==1U;
#else
 static constexpr bool enabled=false;
#endif
};
template<class T,class O,uint32_t TH,uint32_t TILE>
struct GeometryHalfUnitOwnerPolicy {
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool enabled=AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value &&
     TH==2U && TILE>=27U && TILE<=35U;
#else
 static constexpr bool enabled=false;
#endif
};
template<class T,class O,uint32_t TH,uint32_t TILE>
struct GeometryFloatShortNorm64Policy {
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool enabled=AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value &&
     TH==3U && TILE>=179U && TILE<=241U;
#else
 static constexpr bool enabled=false;
#endif
};
template<class T,class O,uint32_t TH,uint32_t TILE>
struct GeometryHalfWideOwnerPolicy {
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool enabled=AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value &&
     TH==3U && TILE>=36U && TILE<=98U;
#else
 static constexpr bool enabled=false;
#endif
};
template<bool ENABLED,bool LOW64=false,bool INIT_MASK=true>
__aicore__ inline void GeometryWideSocState(){
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 if constexpr(ENABLED){
  if ASCEND_IS_AIV {
   // Preserve C220 atomic-none/normal mode; this Float family consumes only low64 mask bits.
#if SBG_PARSE_FLOAT_CTRL_SINGLE_BITS
   uint64_t value=get_ctrl();
   value&=~(1ULL<<6U);
   value&=~(1ULL<<7U);
   value&=~(1ULL<<56U);
   set_ctrl(value);
#else
   constexpr uint64_t resetFields=(3ULL<<6U)|(1ULL<<56U);
   set_ctrl(get_ctrl() & ~resetFields);
#endif
   if constexpr(INIT_MASK)set_vector_mask(LOW64?0ULL:static_cast<uint64_t>(-1),static_cast<uint64_t>(-1));
  }else AscendC::InitSocState();
 }else AscendC::InitSocState();
#else
 AscendC::InitSocState();
#endif
}
// Fixed-pattern extraction uses the SDK vreducev2 primitive. The selected
// prefix length is known from the AoS layout, so no scalar count read is needed.
__aicore__ inline void ExtractFields(AscendC::LocalTensor<float> dst,AscendC::LocalTensor<float> src,uint8_t pattern,uint8_t repeats){
 vreducev2((__ubuf__ uint32_t*)dst.GetPhyAddr(),(__ubuf__ uint32_t*)src.GetPhyAddr(),
           (__ubuf__ uint32_t*)src.GetPhyAddr(),repeats,1,pattern,8,0);
}
template<bool Wide> struct GeometryRowIndex {using type=uint32_t;};
template<> struct GeometryRowIndex<true> {using type=uint64_t;};
template<class T,class O,uint32_t THRESHOLD_KIND,uint32_t TILE_CLASS,bool USE_VECTOR_RATIO=(TILE_CLASS!=0 && TILE_CLASS!=2),bool EARLY_SMALL=false,bool ROTATE_WIDE=false,uint32_t ROTATE_WORKERS=0,bool NATIVE_WIDE=false,uint32_t CONSTANT_WIDE_TAIL=0,uint32_t FOLDED_ROWS=0,bool MIDPOINT_RATIO=false,bool LATE_NARROW_RATIO=false,uint32_t FOLDED_WIDTH=64,uint32_t CONSTANT_SMALL_ROWS=0,bool CACHED_OWNER=false,bool FUSE_BROADCAST=false,bool FLOAT_METADATA=false> class Geometry
#if SBG_RAW_ONLY
;
#else
{
 static constexpr bool FULL64=TILE_CLASS==68;
 static constexpr bool NATIVE_FOLD_MIDPOINT=(TILE_CLASS==86 || (FOLDED_ROWS!=0 && (FOLDED_WIDTH==128 || FOLDED_WIDTH==256))) &&
     MIDPOINT_RATIO && THRESHOLD_KIND==3 && AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value;
 static constexpr bool LITERAL_FLOAT_FULL09=FULL64 && ROTATE_WORKERS==0 &&
     THRESHOLD_KIND==3 && AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value;
 static constexpr bool FLOAT_CONSTANTS=FLOAT_METADATA && FULL64 && ROTATE_WORKERS==0 &&
     AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value && THRESHOLD_KIND==1;
 static constexpr bool LITERAL_FULL09=FUSE_BROADCAST && FULL64 && ROTATE_WORKERS==0 &&
     AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value && THRESHOLD_KIND==3;
 static constexpr bool FUSED_META=(FUSE_BROADCAST && FULL64 && ROTATE_WORKERS==0 && AscendC::IsSameType<T,O>::value &&
     THRESHOLD_KIND==(AscendC::IsSameType<T,half>::value?2U:1U)) || FLOAT_CONSTANTS || LITERAL_FULL09;
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool UNIT32_GROUP=TILE_CLASS==84 && THRESHOLD_KIND==2U &&
     AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value;
#else
 static constexpr bool UNIT32_GROUP=false;
#endif
 static constexpr bool FOUR_FIELD_CAST=LITERAL_FULL09 || UNIT32_GROUP || SBG_PARSE_HALF_SHORT_FOUR_FIELD;
 static constexpr bool LITERAL_SHORT08=CONSTANT_SMALL_ROWS!=0 && THRESHOLD_KIND==3 && AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value;
 // Rotated prefixes retain three field repeats. The matched full64 metadata
 // family appends the real threshold as a fourth field at the same pitch.
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 // More than half of a Float64 conversion repeat contains valid short rows.
 static constexpr bool HALF_SHORT_GROUP_CAST=AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value &&
     THRESHOLD_KIND==2U && TILE_CLASS==69U && CONSTANT_SMALL_ROWS>32U && CONSTANT_SMALL_ROWS<=64U;
#else
 static constexpr bool HALF_SHORT_GROUP_CAST=false;
#endif
 static constexpr bool FULL_STRIDE_CAST=AscendC::IsSameType<T,half>::value && (HALF_SHORT_GROUP_CAST || UNIT32_GROUP || (FULL64 && (ROTATE_WORKERS!=0 || FOUR_FIELD_CAST)));
 static constexpr bool RATIO_ONE=TILE_CLASS==84;
 static constexpr bool FULL32=TILE_CLASS==81 || RATIO_ONE;
#if defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool FLOAT_UNIT_ENTRY_STATE=RATIO_ONE && AscendC::IsSameType<T,float>::value &&
     AscendC::IsSameType<O,float>::value && THRESHOLD_KIND==1U &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>28U)&3U)==3U &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>27U)&1U)!=0U &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>26U)&1U)!=0U &&
     ((((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)>=159U &&
       ((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)<=160U) ||
      (((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)>=243U &&
       ((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)<=244U)
       SBG_UNIT_PAD_EXTRA_TILE(((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)));
#if SBG_PARSE_UNIT_THRESHOLD_GUARD
 static constexpr bool FLOAT_UNIT_SCALAR_POSITIVE_ZERO=FLOAT_UNIT_ENTRY_STATE &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)>=243U;
#else
 static constexpr bool FLOAT_UNIT_SCALAR_POSITIVE_ZERO=false;
#endif
#else
 static constexpr bool FLOAT_UNIT_ENTRY_STATE=false;
 static constexpr bool FLOAT_UNIT_SCALAR_POSITIVE_ZERO=false;
#endif

#if defined(TILING_KEY_VAR) && !ASCENDC_CPU_DEBUG
 static constexpr bool FLOAT_SHORT_NORMAL64=GeometryFloatShortNorm64Policy<T,O,THRESHOLD_KIND,
     (static_cast<uint64_t>(TILING_KEY_VAR)>>18)&255U>::enabled;
 static constexpr bool FLOAT_FULL_NORMAL64=GeometryFloatFullNorm64Policy<T,O,THRESHOLD_KIND,
     (static_cast<uint64_t>(TILING_KEY_VAR)>>18)&255U,
     (static_cast<uint64_t>(TILING_KEY_VAR)>>28)&3U>::enabled;
 static constexpr bool HALF_FULL_FUSED_STATE=GeometryHalfFullInitPolicy<T,O,THRESHOLD_KIND,
     (static_cast<uint64_t>(TILING_KEY_VAR)>>18)&255U,
     (static_cast<uint64_t>(TILING_KEY_VAR)>>27)&1U>::enabled;
 static constexpr bool HALF_FULL_EARLY_ENTRY_ONES=HALF_FULL_FUSED_STATE &&
     (((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)==124U ||
      ((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)==242U SBG_HALF_PAD_EXTRA_TILE(((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U))) &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>28U)&3U)==3U &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>26U)&1U)!=0U;
 static constexpr bool HALF_FULL_REUSE_CONVERSION_MASK=HALF_FULL_EARLY_ENTRY_ONES &&
     (((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)==242U SBG_HALF_PAD_EXTRA_TILE(((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)));

 static constexpr bool COPY_FOLD_FIRST=GeometryHalfFoldInputsFirstPolicy<T,O,THRESHOLD_KIND,
     (static_cast<uint64_t>(TILING_KEY_VAR)>>18)&255U,
     (static_cast<uint64_t>(TILING_KEY_VAR)>>26)&1U>::enabled;
 static constexpr bool COPY_FIRST=!FLOAT_UNIT_ENTRY_STATE && GeometryCopyFirstPolicy<T,O,THRESHOLD_KIND,
     (static_cast<uint64_t>(TILING_KEY_VAR)>>18)&255U,
     (static_cast<uint64_t>(TILING_KEY_VAR)>>26)&1U>::enabled;
 static constexpr bool FLOAT_FULL_ENTRY_MASK_REUSE=FLOAT_FULL_NORMAL64 &&
     (((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)==135U
      SBG_FLOAT_FULL_EXTRA_TILE(((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)));
 static constexpr bool PREBIASED_CORE=!FLOAT_FULL_ENTRY_MASK_REUSE && !HALF_FULL_EARLY_ENTRY_ONES && ((FULL64 && ROTATE_WORKERS==0) || RATIO_ONE) &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>26)&1U)!=0;
#else
 static constexpr bool FLOAT_FULL_NORMAL64=false;
 static constexpr bool FLOAT_SHORT_NORMAL64=false;
 static constexpr bool HALF_FULL_FUSED_STATE=false;
 static constexpr bool HALF_FULL_EARLY_ENTRY_ONES=false;
 static constexpr bool HALF_FULL_REUSE_CONVERSION_MASK=false;
 static constexpr bool FLOAT_FULL_ENTRY_MASK_REUSE=false;
 static constexpr bool PREBIASED_CORE=false;
 static constexpr bool COPY_FIRST=false;
 static constexpr bool COPY_FOLD_FIRST=false;
#endif
 static constexpr bool WIDE=TILE_CLASS==70 || TILE_CLASS==71;
 static constexpr bool LITERAL_WIDE08=WIDE && THRESHOLD_KIND==3 &&
     ((AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value) ||
      (AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value));
 static constexpr bool FUSED_FLOAT_WIDE_STATE=COPY_FIRST && WIDE && ROTATE_WIDE &&
     LITERAL_WIDE08 && AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value;
 // This single-tile prefix shares one byte origin across its three GM tensors.
 static constexpr bool HALF_FULL_BYTE_ORIGIN=LITERAL_FULL09 && HALF_FULL_FUSED_STATE && !PREBIASED_CORE;
 static constexpr bool FLOAT_FULL_BYTE_ORIGIN=FLOAT_FULL_ENTRY_MASK_REUSE;
 static constexpr bool BYTE_PREFIX_ORIGIN=(FUSED_FLOAT_WIDE_STATE &&
     TILE_CLASS==70U && !NATIVE_WIDE) || HALF_FULL_BYTE_ORIGIN || FLOAT_FULL_BYTE_ORIGIN;
 static constexpr uint32_t SCALE=(WIDE || FOLDED_WIDTH==256)?2:1;
 static constexpr uint32_t FOLD_PACK_WIDTH=((FOLDED_WIDTH+FOLDED_ROWS+63U)/64U)*64U;
 static constexpr uint32_t FOLD_GATHER_WIDTH=((FOLDED_WIDTH+FOLDED_ROWS+31U)/32U)*32U;
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 // Complete and near-full native wide tiles share the padded512-row layout.
 static constexpr bool HALF_COMPLETE_DIRECT=WIDE &&
     (TILE_CLASS==70U || (ROTATE_WIDE && CONSTANT_WIDE_TAIL>=449U)) &&
     LITERAL_WIDE08 && AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value;
 static constexpr bool FOLD256_PACKED=!WIDE && FOLDED_WIDTH==256U && FOLDED_ROWS!=0U &&
     MIDPOINT_RATIO && THRESHOLD_KIND==3U && AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value &&
     FOLD_GATHER_WIDTH<FOLD_PACK_WIDTH && 3U*FOLD_PACK_WIDTH<=(4352U-3584U)*SCALE;
#else
 static constexpr bool FOLD256_PACKED=false;
 static constexpr bool HALF_COMPLETE_DIRECT=false;
#endif
 static constexpr bool NATIVE_HALF_SCORE=HALF_COMPLETE_DIRECT || FOLD256_PACKED;
 // Input MTE2 reads can precede atomic-store and Vector state initialization.
 static constexpr bool HALF_COMPLETE_FIRST_COPY=COPY_FIRST && WIDE && NATIVE_WIDE &&
     !ROTATE_WIDE && HALF_COMPLETE_DIRECT;

 // Private Half ones can use the full128 mask already established at entry.
 static constexpr bool HALF_COMPLETE_FULL_INIT_MASK=HALF_COMPLETE_DIRECT && !ROTATE_WIDE;

 // CMPMASK captures zero before this private UB block is reused by Mul.
 static constexpr bool HALF_COMPLETE_REUSE_ZERO_COEFFICIENT=HALF_COMPLETE_DIRECT && !ROTATE_WIDE;
#if defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool HALF_COMPLETE_SHARED_ZERO_PLAN=HALF_COMPLETE_REUSE_ZERO_COEFFICIENT &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>27U)&1U)!=0U;
#else
 static constexpr bool HALF_COMPLETE_SHARED_ZERO_PLAN=false;
#endif


 // These constants have exactly one repeat; their inter-repeat pitch is unused.
 static constexpr bool HALF_COMPLETE_CONSTANT_PITCH=HALF_COMPLETE_DIRECT && !ROTATE_WIDE;
 // Near-full tiles measure their single-repeat constant pitches independently.
 static constexpr bool HALF_PARTIAL_CONSTANT_PITCH=HALF_COMPLETE_DIRECT && ROTATE_WIDE;
#if defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool HALF_PARTIAL_SHARED_ZERO_PLAN=HALF_PARTIAL_CONSTANT_PITCH &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>27U)&1U)!=0U;
#else
 static constexpr bool HALF_PARTIAL_SHARED_ZERO_PLAN=false;
#endif
 static constexpr bool HALF_PARTIAL_SPLIT_ANCHOR_CAST=HALF_PARTIAL_SHARED_ZERO_PLAN;



 // Full wide batches use the overlapped scalar ratio for both dtypes.
 // Partial Float retains its existing path; partial Half retains late loading.
 static constexpr bool LATE_SCALAR_RATIO=((WIDE && (!ROTATE_WIDE || AscendC::IsSameType<T,half>::value)) || LATE_NARROW_RATIO) && !LITERAL_WIDE08;
 static constexpr bool VECTOR_RATIO=USE_VECTOR_RATIO && !MIDPOINT_RATIO && !LATE_SCALAR_RATIO && !LITERAL_WIDE08 && !LITERAL_SHORT08 && !LITERAL_FULL09 && !LITERAL_FLOAT_FULL09;
 static constexpr bool SINGLE_TILE=TILE_CLASS!=0;
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool FOLD_PREFIX_ZERO=FULL64 && ROTATE_WORKERS!=0 && MIDPOINT_RATIO &&
     THRESHOLD_KIND==1U && AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value;
#else
 static constexpr bool FOLD_PREFIX_ZERO=false;
#endif
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool SHORT_THRESHOLD_ZERO=FLOAT_SHORT_NORMAL64 && LITERAL_SHORT08;
#else
 static constexpr bool SHORT_THRESHOLD_ZERO=false;
#endif
 static constexpr bool SINGLE_ROW_FIELDS=SHORT_THRESHOLD_ZERO && CONSTANT_SMALL_ROWS==1U;
 static constexpr uint8_t FLOAT_SHORT_CURRENT_BLOCK=(SBG_PARSE_FLOAT_SHORT_SINGLE_BLOCK_ZERO && SINGLE_ROW_FIELDS)?0U:1U;
 static constexpr bool FLOAT_SHORT_SCALAR_POSITIVE_ZERO=SBG_PARSE_FLOAT_SHORT_POSITIVE_ZERO && SHORT_THRESHOLD_ZERO && !SINGLE_ROW_FIELDS;
 // Complete native Half64 math has exactly one repeat.
 static constexpr uint8_t HALF_CURRENT_REPEAT=(HALF_FULL_FUSED_STATE || SBG_PARSE_FLOAT_REPEAT_SHARE || SBG_PARSE_FLOAT_SHORT_POSITIVE_ZERO)?0U:8U;
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool HALF_BOUND_INPUT_REUSE=LITERAL_FULL09;
 static constexpr bool FLOAT_FOLD_BOUND_INPUT_REUSE=AscendC::IsSameType<T,float>::value &&
     AscendC::IsSameType<O,float>::value && MIDPOINT_RATIO &&
     ((FOLDED_ROWS!=0U && FOLDED_WIDTH==64U) || (FULL64 && ROTATE_WORKERS!=0U));
#else
 static constexpr bool HALF_BOUND_INPUT_REUSE=false;
 static constexpr bool FLOAT_FOLD_BOUND_INPUT_REUSE=false;
#endif
#if __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool FLOAT_FOLD_RAW_GM_THRESHOLD=FOLDED_ROWS!=0U && FOLDED_WIDTH==64U &&
     MIDPOINT_RATIO && THRESHOLD_KIND==1U && AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value;
#else
 static constexpr bool FLOAT_FOLD_RAW_GM_THRESHOLD=false;
#endif
 // Complete Float512 fields are independent until the final comparison.
 static constexpr bool FLOAT_COMPLETE_LATE_ANCHOR=COPY_FIRST && TILE_CLASS==70U &&
     !ROTATE_WIDE && !NATIVE_WIDE && LITERAL_WIDE08 &&
     AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value;
 // The complete Float coefficient family also selects Scalar threshold input.
 static constexpr bool FLOAT_COMPLETE_VECTOR_COEFFICIENT=COPY_FIRST && TILE_CLASS==70U &&
     !ROTATE_WIDE && !NATIVE_WIDE && LITERAL_WIDE08 &&
     AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value;
 // A real Scalar threshold replaces its metadata copy and broadcast.
 static constexpr bool FLOAT_COMPLETE_SCALAR_THRESHOLD_SELF_ZERO=FLOAT_COMPLETE_VECTOR_COEFFICIENT;
 // Plan1 provides immutable score state before the complete Float pipeline.
#if defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool FLOAT_COMPLETE_EARLY_SCORE_PLAN=FLOAT_COMPLETE_SCALAR_THRESHOLD_SELF_ZERO &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>27U)&1U)!=0U;
#else
 static constexpr bool FLOAT_COMPLETE_EARLY_SCORE_PLAN=false;
#endif
 static constexpr bool FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO=FUSED_FLOAT_WIDE_STATE;
#if defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool FLOAT_PARTIAL_SHARED_SCORE_DESCRIPTOR=FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>27U)&1U)!=0U &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>26U)&1U)!=0U;
#else
 static constexpr bool FLOAT_PARTIAL_SHARED_SCORE_DESCRIPTOR=false;
#endif

#if defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool FLOAT_PARTIAL_ONES_BEFORE_PACKED=FLOAT_PARTIAL_SHARED_SCORE_DESCRIPTOR &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>28U)&1U)!=0U;
#else
 static constexpr bool FLOAT_PARTIAL_ONES_BEFORE_PACKED=false;
#endif

#if defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool FLOAT_PARTIAL_POSITIVE_ZERO_PLAN=FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>27U)&1U)!=0U;
#else
 static constexpr bool FLOAT_PARTIAL_POSITIVE_ZERO_PLAN=false;
#endif

 static constexpr bool NATIVE_PARTIAL_WIDE=WIDE && ROTATE_WIDE &&
                            AscendC::IsSameType<T,half>::value && AscendC::IsSameType<O,half>::value;
#if defined(TILING_KEY_VAR) && __NPU_ARCH__ == 2201 && !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_OOM)
 static constexpr bool HALF_SPLIT_ENTRY_V2=HALF_PARTIAL_SPLIT_ANCHOR_CAST &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>28U)&3U)==3U &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>26U)&1U)!=0U &&
     ((((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)>=107U &&
       ((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)<=121U) ||
      (((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)>=136U &&
       ((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)<=150U));
 static constexpr bool HALF_SPLIT_BYTE_ORIGIN=HALF_SPLIT_ENTRY_V2 &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)>=136U &&
     ((static_cast<uint64_t>(TILING_KEY_VAR)>>18U)&255U)<=150U;
#else
 static constexpr bool HALF_SPLIT_ENTRY_V2=false;
 static constexpr bool HALF_SPLIT_BYTE_ORIGIN=false;
#endif

 AscendC::GlobalTensor<T> boxes,anchors;
 AscendC::GlobalTensor<float> threshold;
 AscendC::GlobalTensor<O> out;
 AscendC::GlobalTensor<half> threshold16;
 static constexpr bool hasThreshold=THRESHOLD_KIND!=0;
 static constexpr bool thresholdIsHalf=THRESHOLD_KIND==2 || (THRESHOLD_KIND==3 && AscendC::IsSameType<T,half>::value);
 // Complete native Half row offsets remain wide through GM address scaling.
 using RowIndex=typename GeometryRowIndex<(WIDE && LITERAL_WIDE08 && AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value) || HALF_FULL_BYTE_ORIGIN || FLOAT_FULL_BYTE_ORIGIN || HALF_SPLIT_BYTE_ORIGIN || HALF_COMPLETE_FIRST_COPY>::type;
 RowIndex start; uint32_t count; float ratio; AscendC::GlobalTensor<float> ratioGm;
 __aicore__ inline int64_t OwnerIndex(int64_t logicalOwner){
 if constexpr(CACHED_OWNER)return logicalOwner;
 else {
#if SBG_PARSE_HALF_RAW_OWNER_SPR || SBG_PARSE_FLOAT_RAW_OWNER_SPR || SBG_PARSE_HALF_COMPLETE_RAW_OWNER
 // Preserve the real block, task ratio and subblock ID in owner arithmetic.
 const uint64_t rawOwner=static_cast<uint64_t>(get_block_idx());
 const uint64_t taskRatio=static_cast<uint64_t>(AscendC::GetTaskRation());
 const uint64_t subOwner=static_cast<uint64_t>(AscendC::GetSubBlockIdx());
 return static_cast<int64_t>(rawOwner*taskRatio+subOwner);
#else
 return AscendC::GetBlockIdx();
#endif
 }
 }
 __aicore__ __attribute__((always_inline)) inline void InitNarrowOrigin(const SafetyBeltViolationGeometryTilingData &d,int64_t logicalOwner) {
 if constexpr(CONSTANT_SMALL_ROWS){start=0;count=CONSTANT_SMALL_ROWS;}
 else if constexpr(FOLDED_ROWS){start=(ROTATE_WORKERS-1U)*FOLDED_WIDTH;count=FOLDED_WIDTH+FOLDED_ROWS;}
 else if constexpr(FULL32){start=(OwnerIndex(logicalOwner)-(ROTATE_WORKERS?1U:0U))*32;count=32;}
 else if constexpr(TILE_CLASS==86){start=(OwnerIndex(logicalOwner)-1U)*128;count=128;}
 else if constexpr(TILE_CLASS==82){start=(ROTATE_WORKERS?ROTATE_WORKERS-1U:OwnerIndex(logicalOwner))*32;count=(32U<d.batch-start?32U:d.batch-start);}
 else if constexpr(FULL64){
 if constexpr(HALF_FULL_BYTE_ORIGIN || FLOAT_FULL_BYTE_ORIGIN)
  start=static_cast<uint64_t>(OwnerIndex(logicalOwner))*(64ULL*4ULL*sizeof(T));
 else start=(OwnerIndex(logicalOwner)-(ROTATE_WORKERS?1U:0U))*64;
 count=64;
 }
 else if constexpr(TILE_CLASS==69){start=(ROTATE_WORKERS?ROTATE_WORKERS-1U:OwnerIndex(logicalOwner))*64;count=(64U<d.batch-start?64U:d.batch-start);}
 else InitWideOrigin(d,logicalOwner);
 }
 __aicore__ __attribute__((always_inline)) inline void InitWideOrigin(const SafetyBeltViolationGeometryTilingData &d,int64_t logicalOwner) {
 if constexpr(TILE_CLASS==70){
 if constexpr(HALF_SPLIT_BYTE_ORIGIN)
  start=(static_cast<uint64_t>(OwnerIndex(logicalOwner))-1ULL)*(512ULL*4ULL*sizeof(T));
 else if constexpr(BYTE_PREFIX_ORIGIN)
  start=(static_cast<uint64_t>(OwnerIndex(logicalOwner))-1ULL)*(512ULL*4ULL*sizeof(T));
 else start=(OwnerIndex(logicalOwner)-(ROTATE_WIDE?1U:0U))*512;
 count=512;
 }
 else if constexpr(TILE_CLASS==71){
 // Tiling's Float wide lower bound equals the legal owner capacity,
 // so this existing static family has exactly that many owners.
 if constexpr(HALF_SPLIT_BYTE_ORIGIN)
  start=(static_cast<uint64_t>(AscendC::GetBlockNum())-1ULL)*(512ULL*4ULL*sizeof(T));
 else if constexpr(FUSED_FLOAT_WIDE_STATE && CONSTANT_WIDE_TAIL)
  start=(kGeometryFloatWideOwners-1U)*512U;
 else start=(ROTATE_WIDE?AscendC::GetBlockNum()-1U:OwnerIndex(logicalOwner))*512;
 if constexpr(CONSTANT_WIDE_TAIL)count=CONSTANT_WIDE_TAIL;
 else count=(512U<d.batch-start?512U:d.batch-start);
 }
 else if constexpr(TILE_CLASS==3){start=OwnerIndex(logicalOwner)*256;count=256;}
 else if constexpr(SINGLE_TILE){start=OwnerIndex(logicalOwner)*256;count=(256U<d.batch-start?256U:d.batch-start);}
 else {start=OwnerIndex(logicalOwner)*d.perCore;count=(d.perCore<d.batch-start?d.perCore:d.batch-start);}
 }
 __aicore__ __attribute__((always_inline)) inline void InitRatio(GM_ADDR tiling,const SafetyBeltViolationGeometryTilingData &d) {
 if constexpr(!RATIO_ONE){
 if constexpr(LITERAL_FULL09 || LITERAL_FLOAT_FULL09)ratio=0.9f;
 else if constexpr(LITERAL_SHORT08)ratio=0.8f;
 else if constexpr(LITERAL_WIDE08)ratio=0.8f;
 else if constexpr(VECTOR_RATIO || LATE_SCALAR_RATIO)ratioGm.SetGlobalBuffer((__gm__ float*)tiling+(FLOAT_CONSTANTS?0U:5U));
 else if constexpr(MIDPOINT_RATIO)ratio=0.5f;
 else ratio=d.waistRatio;
 }
 }
 __aicore__ __attribute__((always_inline)) inline void BindGlobalBuffers(GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o) {
 if constexpr(PREBIASED_CORE){
  boxes.SetGlobalBuffer((__gm__ T*)b+static_cast<uint64_t>(start)*4U);
  anchors.SetGlobalBuffer((__gm__ T*)a+static_cast<uint64_t>(start)*2U);
  out.SetGlobalBuffer((__gm__ O*)o+start);
  start=0;
 }else{
 boxes.SetGlobalBuffer((__gm__ T*)b); anchors.SetGlobalBuffer((__gm__ T*)a); out.SetGlobalBuffer((__gm__ O*)o);
 }


 if(hasThreshold){
 if(thresholdIsHalf)threshold16.SetGlobalBuffer((__gm__ half*)t);
 else threshold.SetGlobalBuffer((__gm__ float*)t);
 }
 }
public:
 __aicore__ inline void Init(GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o,GM_ADDR tiling,const SafetyBeltViolationGeometryTilingData &d,int64_t logicalOwner=0) {
 InitNarrowOrigin(d,logicalOwner);
 InitRatio(tiling,d);
 BindGlobalBuffers(b,a,t,o);
 }
 __aicore__ inline void Process() {
 if constexpr(CONSTANT_SMALL_ROWS){Tile(0,CONSTANT_SMALL_ROWS);}
 else if constexpr(FOLDED_ROWS){Tile(0,FOLDED_WIDTH+FOLDED_ROWS);}
 else if constexpr(CONSTANT_WIDE_TAIL){Tile(0,CONSTANT_WIDE_TAIL);}
 else if constexpr(FULL32){Tile(0,32);}
 else if constexpr(FULL64){Tile(0,64);}
 else if constexpr(SINGLE_TILE){Tile(0,count);}
 else {for(uint32_t pos=0;pos<count;pos+=256)Tile(pos,(256U<count-pos?256U:count-pos));}
 }
#if !SBG_PARSE_NATIVE_BODY
 __aicore__ inline void NativeWideTile(uint32_t pos,uint32_t n);
#else
 __aicore__ inline void NativeWideTile(uint32_t pos,uint32_t n) {
 constexpr bool PACK_WIDE_METADATA=(WIDE && AscendC::IsSameType<T,half>::value &&
     AscendC::IsSameType<O,half>::value && hasThreshold && thresholdIsHalf &&
     (TILE_CLASS==70 || CONSTANT_WIDE_TAIL>=449)) || FOLD256_PACKED;
 AscendC::LocalTensor<float> mem(AscendC::TPosition::VECCALC,0U,8192U*SCALE);
 auto raw=mem.ReinterpretCast<T>();
 auto thloc=mem[5504*SCALE];
 auto y1=mem[3584*SCALE];auto y2=mem[3840*SCALE];auto cy=mem[4096*SCALE];
 auto result=mem[4352*SCALE];auto mask=mem[5376*SCALE].ReinterpretCast<uint8_t>();
 auto scalarZero=mem[5696*SCALE];
 const uint8_t repeats=static_cast<uint8_t>((n+63)/64);
 if constexpr(FOLD256_PACKED){
  // Fixed-pattern y1 extraction stops at its32-row boundary. The next
  // aligned block remains inside the original64-row conversion padding.
  thloc=mem[2048*SCALE+FOLD_GATHER_WIDTH/2U];
  y2=y1[FOLD_PACK_WIDTH];cy=y1[2U*FOLD_PACK_WIDTH];
 }else if constexpr(PACK_WIDE_METADATA){
  // The real Half threshold follows the1536 packedHalf coordinates.
  // Its converted64-lane field precedes the score lifetime without overlap.
  thloc=mem[2048*SCALE+768];
  if constexpr(HALF_PARTIAL_SPLIT_ANCHOR_CAST){
   // Pack1024 box coordinates and64 threshold lanes before anchors.
   thloc=mem[2048*SCALE+512];
   cy=y1[1088];
  }
  result=mem[4352*SCALE+64];
#if SBG_PARSE_HALF_COMPLETE_SKEW_FIELDS
  // The three 512-row fields have one private 32-byte gap each. A single
  // conversion expands their 528-Half pitch into a 528-Float pitch.
  thloc=mem[2048*SCALE+792];
  y2=y1[528];cy=y1[1056];
  // The original 25-repeat expansion ends before these initialized ones.
#if SBG_PARSE_HALF_COMPLETE_RAW_ZERO_JOIN
  // The full anchor copy fills one private 32-byte block with Half ones.
  result=mem[1536];
#else
  result=y1[1664];
#endif
#endif
 }

 if constexpr(LATE_SCALAR_RATIO)
 AscendC::DataCacheCleanAndInvalid<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(ratioGm);
 if constexpr(HALF_SPLIT_BYTE_ORIGIN){
  // Process calls this full owner or constant near-full tail once at pos0.
  // start is a byte origin shared by boxes, anchors and output.
  if(n%8U==0U){
   AscendC::DataCopy(raw,boxes[start/sizeof(T)],n*4U);
   AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
   AscendC::DataCopy(raw[1024*SCALE],anchors[start/(2ULL*sizeof(T))],n*2U);
  }else{
   AscendC::DataCopyPad(raw,boxes[start/sizeof(T)],AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*4U*sizeof(T)),0,0,0},AscendC::DataCopyPadExtParams<T>{false,0,0,0});
   AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
   AscendC::DataCopyPad(raw[1024*SCALE],anchors[start/(2ULL*sizeof(T))],AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*2U*sizeof(T)),0,0,0},AscendC::DataCopyPadExtParams<T>{false,0,0,0});
  }
 }else{
 if(n%8==0){
 AscendC::DataCopy(raw,boxes[(start+pos)*4],n*4);
 if constexpr(HALF_COMPLETE_FIRST_COPY)GeometryWideSocState<true>();
 if constexpr(FOLD256_PACKED || (NATIVE_PARTIAL_WIDE && HALF_COMPLETE_DIRECT))AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
#if SBG_PARSE_HALF_COMPLETE_RAW_ZERO_JOIN
 AscendC::DataCopyPad(raw[1024*SCALE],anchors[(start+pos)*2],
     AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*2U*sizeof(T)),0,0,0},
     AscendC::DataCopyPadExtParams<T>{true,0,16,static_cast<T>(1.0f)});
#else
 AscendC::DataCopy(raw[1024*SCALE],anchors[(start+pos)*2],n*2);
#endif
 }else{
 AscendC::DataCopyPad(raw,boxes[(start+pos)*4],AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*4*sizeof(T)),0,0,0},AscendC::DataCopyPadExtParams<T>{false,0,0,0});
 if constexpr(HALF_COMPLETE_FIRST_COPY)GeometryWideSocState<true>();
 if constexpr(FOLD256_PACKED || (NATIVE_PARTIAL_WIDE && HALF_COMPLETE_DIRECT))AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
 AscendC::DataCopyPad(raw[1024*SCALE],anchors[(start+pos)*2],AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*2*sizeof(T)),0,0,0},AscendC::DataCopyPadExtParams<T>{false,0,0,0});
 }
 }
 if constexpr(VECTOR_RATIO)AscendC::DataCopyPad(mem[6016*SCALE],ratioGm,AscendC::DataCopyExtParams{1,4,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
 if constexpr(hasThreshold){
 if constexpr(thresholdIsHalf)AscendC::DataCopyPad(thloc.ReinterpretCast<half>(),threshold16,AscendC::DataCopyExtParams{1,2,0,0,0},AscendC::DataCopyPadExtParams<half>{false,0,0,0});
 else AscendC::DataCopyPad(thloc,threshold,AscendC::DataCopyExtParams{1,4,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
 }
 if constexpr(COPY_FOLD_FIRST)GeometryWideSocState<FOLD256_PACKED,true,false>();
 // These writes are independent of every in-flight input DMA. CMPMASK stores
 // the scalar zero needed by mode-1 Select, not the later packed compare bits.
 if constexpr(!FOLD256_PACKED && !HALF_COMPLETE_FIRST_COPY && !HALF_SPLIT_ENTRY_V2)AscendC::SetMaskNorm();
 if constexpr(HALF_PARTIAL_SHARED_ZERO_PLAN && !HALF_SPLIT_ENTRY_V2)
  set_vector_mask(0xffffffffffffffffULL,0xffffffffffffffffULL);
 else if constexpr(!HALF_COMPLETE_FULL_INIT_MASK && !HALF_SPLIT_ENTRY_V2)set_vector_mask(0,0xffffffffffffffffULL);
 if constexpr(NATIVE_HALF_SCORE){
#if !SBG_PARSE_HALF_COMPLETE_RAW_ZERO_JOIN
  // One private Half ones block suffices for the broadcast Select source.
  if constexpr(HALF_COMPLETE_CONSTANT_PITCH || HALF_PARTIAL_CONSTANT_PITCH)
   AscendC::Duplicate<half,false>(result.ReinterpretCast<half>(),half(1.0f),(HALF_COMPLETE_FULL_INIT_MASK || HALF_PARTIAL_SHARED_ZERO_PLAN)?128:64,1,1,0);
  else AscendC::Duplicate<half,false>(result.ReinterpretCast<half>(),half(1.0f),64,1,1,8);
#endif
 }else AscendC::Duplicate<float,false>(result,1.0f,64,repeats,1,8);
 if constexpr(HALF_COMPLETE_SHARED_ZERO_PLAN || HALF_PARTIAL_SHARED_ZERO_PLAN){
  // Both operands contain initialized Half ones. Strict self-GT fills
  // CMPMASK with zero; the scalar source of the later Select uses Half +0.
  // One source broadcasts16 lanes and the other reads all128 initialized
  // lanes. This single-repeat descriptor also serves both Duplicates.
#if !SBG_PARSE_HALF_COMPLETE_RAW_ZERO_JOIN
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Compare<half,false>(result.ReinterpretCast<half>(),result.ReinterpretCast<half>(),
      AscendC::CMPMODE::GT,128,AscendC::BinaryRepeatParams{1,0,1,0,0,0});
  AscendC::PipeBarrier<PIPE_V>();
#endif
 }else{
  if constexpr(HALF_COMPLETE_CONSTANT_PITCH || HALF_PARTIAL_CONSTANT_PITCH)
   AscendC::Duplicate<float,false>(scalarZero,0.0f,64,1,1,0);
  else AscendC::Duplicate<float,false>(scalarZero,0.0f,64,1,1,8);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::SetCmpMask(scalarZero);
  AscendC::PipeBarrier<PIPE_V>();
 }
 if constexpr(HALF_COMPLETE_REUSE_ZERO_COEFFICIENT){
  // The previous barrier completes the128-bit UB->CMPMASK read. Its scalar
  // zero remains in SPR through packed Compare and the original Half Select.
#if SBG_PARSE_HALF_COMPLETE_RAW_ZERO_JOIN
  vector_dup((__ubuf__ float*)scalarZero.GetPhyAddr(),0.8f,1,1,1,0,0);
#else
  AscendC::Duplicate<float,false>(scalarZero,0.8f,64,1,1,0);
#endif
 }
 if constexpr(LATE_SCALAR_RATIO){
 ratio=ratioGm.GetValue(0);
 }
 AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 if constexpr(FOLD256_PACKED || (NATIVE_PARTIAL_WIDE && HALF_COMPLETE_DIRECT))AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
 else AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 auto thresholdBroadcast=mem[5952*SCALE];
 if constexpr(VECTOR_RATIO)AscendC::Brcb(mem[6080*SCALE],mem[6016*SCALE],1,{1,8});
 if constexpr(!PACK_WIDE_METADATA){
 if constexpr(hasThreshold){
 auto source=thloc;
 if constexpr(thresholdIsHalf){
 source=mem[5888*SCALE];
 AscendC::Cast<float,half,false>(source,thloc.ReinterpretCast<half>(),AscendC::RoundMode::CAST_NONE,64,1,AscendC::UnaryRepeatParams{1,1,8,4});
 AscendC::PipeBarrier<PIPE_V>();
 }
 AscendC::Brcb(thresholdBroadcast,source,1,{1,8});
 }else AscendC::Duplicate<float,false>(thresholdBroadcast,0.0f,64,1,1,8);
 }
 if constexpr(AscendC::IsSameType<T,half>::value){
 auto yh1=mem[2048*SCALE].ReinterpretCast<half>();
 auto yh2=mem[2304*SCALE].ReinterpretCast<half>();
 auto ah=mem[2560*SCALE].ReinterpretCast<half>();
 constexpr bool PACK_WIDE_FIELDS=(WIDE && AscendC::IsSameType<O,half>::value &&
     (TILE_CLASS==70 || CONSTANT_WIDE_TAIL>=449)) || FOLD256_PACKED;
 if constexpr(PACK_WIDE_FIELDS){
  // Three512-lane Half fields occupy1536 contiguous lanes; Float fields
  // already occupy three contiguous512-lane regions. Rounded width is512.
  if constexpr(FOLD256_PACKED){
   yh2=mem[2048*SCALE+FOLD_PACK_WIDTH/2U].ReinterpretCast<half>();
   ah=mem[2048*SCALE+FOLD_PACK_WIDTH].ReinterpretCast<half>();
  }else{
#if SBG_PARSE_HALF_COMPLETE_SKEW_FIELDS
   yh2=yh1[528];ah=yh1[1056];
#else
   yh2=mem[2048*SCALE+256].ReinterpretCast<half>();
   ah=mem[2048*SCALE+512].ReinterpretCast<half>();
#endif
   if constexpr(HALF_PARTIAL_SPLIT_ANCHOR_CAST)
    ah=mem[2048*SCALE+544].ReinterpretCast<half>();
  }
 }

 if constexpr(!FOLD256_PACKED && !HALF_COMPLETE_FIRST_COPY && !HALF_SPLIT_ENTRY_V2)AscendC::SetMaskNorm();
 if constexpr(!HALF_COMPLETE_FULL_INIT_MASK && !HALF_SPLIT_ENTRY_V2)set_vector_mask(0xffffffffffffffffULL,0xffffffffffffffffULL);
 vreducev2((__ubuf__ uint16_t*)yh1.GetPhyAddr(),(__ubuf__ uint16_t*)raw.GetPhyAddr(),(__ubuf__ uint16_t*)raw.GetPhyAddr(),static_cast<uint8_t>((n*4+127)/128),1,4,8,0);
 vreducev2((__ubuf__ uint16_t*)yh2.GetPhyAddr(),(__ubuf__ uint16_t*)raw.GetPhyAddr(),(__ubuf__ uint16_t*)raw.GetPhyAddr(),static_cast<uint8_t>((n*4+127)/128),1,6,8,0);
 if constexpr(FOLD256_PACKED || (NATIVE_PARTIAL_WIDE && HALF_COMPLETE_DIRECT))AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 vreducev2((__ubuf__ uint16_t*)ah.GetPhyAddr(),(__ubuf__ uint16_t*)raw[1024*SCALE].GetPhyAddr(),(__ubuf__ uint16_t*)raw[1024*SCALE].GetPhyAddr(),static_cast<uint8_t>((n*2+127)/128),1,2,8,0);
 AscendC::PipeBarrier<PIPE_V>();
 if constexpr(!FOLD256_PACKED && !HALF_COMPLETE_FIRST_COPY && !HALF_SPLIT_ENTRY_V2)AscendC::SetMaskNorm();
#if !SBG_PARSE_HALF_COMPLETE_FULL_MASK && !SBG_PARSE_HALF_SPLIT_CAST_FULL_MASK
 set_vector_mask(0,0xffffffffffffffffULL);
#endif
 if constexpr(PACK_WIDE_FIELDS){
  if constexpr(FOLD256_PACKED)
   // Three packed padded fields; the actual threshold reuses y1 padding.
   AscendC::Cast<float,half,false>(y1,yh1,AscendC::RoundMode::CAST_NONE,64,static_cast<uint8_t>(3U*FOLD_PACK_WIDTH/64U),AscendC::UnaryRepeatParams{1,1,8,4});
  else if constexpr(HALF_PARTIAL_SPLIT_ANCHOR_CAST)
   AscendC::Cast<float,half,false>(y1,yh1,AscendC::RoundMode::CAST_NONE,64,17,AscendC::UnaryRepeatParams{1,1,8,4});
#if SBG_PARSE_HALF_COMPLETE_SKEW_FIELDS
  else AscendC::Cast<float,half,false>(y1,yh1,AscendC::RoundMode::CAST_NONE,64,25,AscendC::UnaryRepeatParams{1,1,8,4});
#else
  else AscendC::Cast<float,half,false>(y1,yh1,AscendC::RoundMode::CAST_NONE,64,PACK_WIDE_METADATA?25:24,AscendC::UnaryRepeatParams{1,1,8,4});
#endif
 }else{
 AscendC::Cast<float,half,false>(y1,yh1,AscendC::RoundMode::CAST_NONE,64,repeats,AscendC::UnaryRepeatParams{1,1,8,4});
 AscendC::Cast<float,half,false>(y2,yh2,AscendC::RoundMode::CAST_NONE,64,repeats,AscendC::UnaryRepeatParams{1,1,8,4});
 AscendC::Cast<float,half,false>(cy,ah,AscendC::RoundMode::CAST_NONE,64,repeats,AscendC::UnaryRepeatParams{1,1,8,4});
 }
 }else{
 ExtractFields(y1,raw,4,static_cast<uint8_t>((n*4+63)/64));
 ExtractFields(y2,raw,6,static_cast<uint8_t>((n*4+63)/64));
 ExtractFields(cy,raw[1024*SCALE],2,static_cast<uint8_t>((n*2+63)/64));
 }
 AscendC::PipeBarrier<PIPE_V>();
 if constexpr(PACK_WIDE_METADATA){
  if constexpr(HALF_COMPLETE_FIRST_COPY){
   // Keep the same full128 state as SDK Brcb. Float arithmetic consumes
   // low64; the final Half Select still consumes all128 lanes.
#if !SBG_PARSE_HALF_COMPLETE_FULL_MASK
   set_vector_mask(0xffffffffffffffffULL,0xffffffffffffffffULL);
#endif
  }else if constexpr(FOLD256_PACKED)AscendC::Brcb(thresholdBroadcast,y1[FOLD_GATHER_WIDTH],1,{1,8});
  else if constexpr(HALF_COMPLETE_CONSTANT_PITCH || HALF_PARTIAL_CONSTANT_PITCH){
   if constexpr(!(SBG_PARSE_HALF_SPLIT_RAW_BRCB_OVERLAP && HALF_PARTIAL_SPLIT_ANCHOR_CAST))
    AscendC::Brcb(thresholdBroadcast,HALF_PARTIAL_SPLIT_ANCHOR_CAST?y1[1024]:mem[4352*SCALE],1,{1,0});
  }
  else AscendC::Brcb(thresholdBroadcast,mem[4352*SCALE],1,{1,8});
 }
 // All expansion stays in UB. Only n valid rows are read from or written to GM.
 AscendC::Sub<float,false>(y2,y2,y1,64,repeats,AscendC::BinaryRepeatParams{1,1,1,8,8,8});
#if SBG_PARSE_HALF_SPLIT_RAW_BRCB_OVERLAP
 if constexpr(HALF_PARTIAL_SPLIT_ANCHOR_CAST){
  // Cast completion releases both the box fields and actual threshold.
  // This disjoint broadcast shares the original Sub completion barrier.
  vbrcb((__ubuf__ uint32_t*)thresholdBroadcast.GetPhyAddr(),
        (__ubuf__ uint32_t*)y1[1024].GetPhyAddr(),1,0,1);
 }
#endif
 if constexpr(HALF_COMPLETE_FIRST_COPY){
  // The converted threshold and its broadcast are disjoint from this Sub.
  // Their first consumer is the final threshold Add. Preserve the original
  // single-repeat Brcb descriptor and the original Sub completion barrier.
  vbrcb((__ubuf__ uint32_t*)thresholdBroadcast.GetPhyAddr(),
#if SBG_PARSE_HALF_COMPLETE_SKEW_FIELDS
        (__ubuf__ uint32_t*)y1[1584].GetPhyAddr(),1,0,1);
#else
        (__ubuf__ uint32_t*)mem[4352*SCALE].GetPhyAddr(),1,0,1);
#endif
 }
 AscendC::PipeBarrier<PIPE_V>();
#if SBG_PARSE_HALF_FOLD_EXACT_AXPY
 // The retained FP32 difference of Half inputs scales exactly by0.5.
 // Sub and the prior broadcast have finished reading this original y1.
 AscendC::Axpy<float,float,false>(y1,y2,0.5f,64,repeats,AscendC::UnaryRepeatParams{1,1,8,8});
 AscendC::PipeBarrier<PIPE_V>();
 y2=y1;
#else
 if constexpr(VECTOR_RATIO)AscendC::Mul<float,false>(y2,y2,mem[6080*SCALE],64,repeats,AscendC::BinaryRepeatParams{1,1,0,8,8,0});
 else if constexpr(HALF_COMPLETE_REUSE_ZERO_COEFFICIENT)
  AscendC::Mul<float,false>(y2,y2,scalarZero,64,repeats,AscendC::BinaryRepeatParams{1,1,0,8,8,0});
 else AscendC::Muls<float,false>(y2,y2,ratio,64,repeats,AscendC::UnaryRepeatParams{1,1,8,8});
 AscendC::PipeBarrier<PIPE_V>();
 AscendC::Add<float,false>(y2,y2,y1,64,repeats,AscendC::BinaryRepeatParams{1,1,1,8,8,8});
 AscendC::PipeBarrier<PIPE_V>();
#endif
 AscendC::Add<float,false>(y2,y2,thresholdBroadcast,64,repeats,AscendC::BinaryRepeatParams{1,1,0,8,8,0});
 if constexpr(HALF_PARTIAL_SPLIT_ANCHOR_CAST){
  // This conversion is independent of the last waist/threshold Add.
  // The retained completion barrier protects both inputs of Compare.
  // Exact C220 CAST_NONE expansion, retaining the original low64
  // Float consumers and the following real completion barrier.
  vconv_f162f32((__ubuf__ float*)cy.GetPhyAddr(),
      (__ubuf__ half*)mem[2048*SCALE+544].GetPhyAddr(),8,1,1,8,4);
 }
#if SBG_PARSE_HALF_COMPLETE_RAW_ZERO_JOIN
 // Ones have been complete since the gather join and stay live for Select.
 // This independent SPR write shares the last Float Add completion join.
 vcmp_gt((__ubuf__ half*)result.GetPhyAddr(),(__ubuf__ half*)result.GetPhyAddr(),
     1,1,0,0,0,0,0);
#endif
 AscendC::PipeBarrier<PIPE_V>();
 AscendC::Compare<float,uint8_t,false>(mask,cy,y2,AscendC::CMPMODE::GT,64,repeats,AscendC::BinaryRepeatParams{1,1,1,8,8,8});
 AscendC::PipeBarrier<PIPE_V>();
 auto outputLocal=mem[5632*SCALE].ReinterpretCast<O>();
 if constexpr(NATIVE_HALF_SCORE){
  // Original FP32 Compare bits map directly to exact Half +0/+1.
  // This selected pipeline preserves both full entry masks throughout.
  if constexpr(!(SBG_PARSE_HALF_SPLIT_RAW_BRCB_OVERLAP && HALF_PARTIAL_SPLIT_ANCHOR_CAST))
   set_vector_mask(0xffffffffffffffffULL,0xffffffffffffffffULL);
  AscendC::Select<half,uint8_t>(outputLocal,mask,result.ReinterpretCast<half>(),
      static_cast<uint8_t>((n+127U)/128U),AscendC::BinaryRepeatParams{1,0,1,8,0,8});
 }else{
 AscendC::Select<float,uint8_t>(result,mask,result,repeats,AscendC::BinaryRepeatParams{1,1,1,8,8,8});

 if constexpr(AscendC::IsSameType<O,half>::value){
 AscendC::PipeBarrier<PIPE_V>();
 AscendC::Cast<half,float,false>(outputLocal,result,AscendC::RoundMode::CAST_RINT,64,repeats,AscendC::UnaryRepeatParams{1,1,4,8});
 }else outputLocal=result;
 }
 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 if constexpr(HALF_SPLIT_BYTE_ORIGIN){
  if(n%(32/sizeof(O))==0)AscendC::DataCopy(out[start/(4ULL*sizeof(O))],outputLocal,n);
  else AscendC::DataCopyPad(out[start/(4ULL*sizeof(O))],outputLocal,AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*sizeof(O)),0,0,0});
 }else{
 if(n%(32/sizeof(O))==0)AscendC::DataCopy(out[start+pos],outputLocal,n);
 else AscendC::DataCopyPad(out[start+pos],outputLocal,AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*sizeof(O)),0,0,0});
 }
 }
#endif
#if SBG_COMPILE_NATIVE_ONLY
 __aicore__ inline void Tile(uint32_t pos,uint32_t n){NativeWideTile(pos,n);}
#else
 __aicore__ inline void Tile(uint32_t pos,uint32_t n) {
 if constexpr(WIDE && NATIVE_WIDE || NATIVE_PARTIAL_WIDE || NATIVE_FOLD_MIDPOINT){NativeWideTile(pos,n);}else{

 AscendC::LocalTensor<float> mem(AscendC::TPosition::VECCALC,0U,8192U*SCALE);
 auto raw=mem.ReinterpretCast<T>();
 auto thloc=mem[(FOUR_FIELD_CAST?2816U:5504U)*SCALE];
#if SBG_PARSE_HALF_PAD_ONE_FIELD_BASE
 thloc=mem[4608];
#endif
 auto bf=mem[2048*SCALE]; auto af=mem[3072*SCALE];
 if constexpr(AscendC::IsSameType<T,float>::value){bf=mem;af=mem[1024*SCALE];}
 auto y1=mem[3584*SCALE]; auto y2=mem[3840*SCALE]; auto cy=mem[4096*SCALE]; auto result=mem[4352*SCALE];
#if SBG_PARSE_FLOAT_THRESHOLD_REUSE_CY
 thloc=cy;
#endif
#if SBG_PARSE_FLOAT_SCORE_REUSE_CY
 // Compare is the last reader of anchor-y. Its existing completion barrier
 // precedes the final score Select and releases these same64 Float lanes.
 result=cy;
#endif
#if SBG_PARSE_HALF_GROUPED_INPUT_REUSE
 // The three gathers finish before this four-field conversion. Its sources
 // remain in their original disjoint Half fields; input-coordinate UB is dead.
 y1=mem; y2=mem[256]; cy=mem[512]; result=mem[768];
#endif
 auto mask=mem[5376*SCALE].ReinterpretCast<uint8_t>();
 constexpr bool NORMAL_WIDE_SELECT=((WIDE && (!ROTATE_WIDE || LITERAL_WIDE08)) ||
     FLOAT_FOLD_RAW_GM_THRESHOLD) && AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value;


 float scalarThreshold=0.0f;
 int64_t scalarThresholdIdentityBits=0;

 if constexpr(FLOAT_FOLD_RAW_GM_THRESHOLD)
  AscendC::DataCacheCleanAndInvalid<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(threshold);
 if constexpr(LATE_SCALAR_RATIO)
 AscendC::DataCacheCleanAndInvalid<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(ratioGm);
 if constexpr(SINGLE_ROW_FIELDS){
  // A one-row tile uses only y1,y2 and anchor-y. Keep each field32B aligned.
  // Keep the complete box read within its real sixteen-byte row.
  static_assert(!COPY_FIRST,"The short-row entry initializes state before Tile");
  // One complete12B span contains y1,the unusedx2,and y2.
  // All bytes lie within this real16B box row.
  AscendC::DataCopyPad(y1,boxes[1],AscendC::DataCopyExtParams{1,12,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
  AscendC::DataCopyPad(cy,anchors[1],AscendC::DataCopyExtParams{1,4,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
 }else if constexpr(BYTE_PREFIX_ORIGIN){
  // Process calls this complete prefix once at pos0. The SDK
  // element indices preserve its original byte addresses and GM intervals.
  if constexpr(FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO)
   AscendC::DataCopy(raw,boxes[start/sizeof(T)],AscendC::DataCopyParams{2,static_cast<uint16_t>(n*4U*sizeof(T)/(2U*32U)),0,0});
  else
#if SBG_PARSE_HALF_PAD_SCORE
   AscendC::DataCopyPad(raw,boxes[start/sizeof(T)],
       AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*4U*sizeof(T)),0,0,0},
       AscendC::DataCopyPadExtParams<T>{true,0,16,static_cast<T>(1.0f)});
#else
#if SBG_PARSE_FLOAT_PAD_SCORE
   AscendC::DataCopyPad(raw,boxes[start/sizeof(T)],
       AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*4U*sizeof(T)),0,0,0},
       AscendC::DataCopyPadExtParams<T>{true,0,8,static_cast<T>(1.0f)});
#else
   AscendC::DataCopy(raw,boxes[start/sizeof(T)],n*4);
#endif
#endif
#if SBG_PARSE_HALF_BOX_MATH_PIPELINE
  // Only the first box copy feeds the early y1/y2 conversion and waist math.
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
#endif
 if constexpr(FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO)AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
  if constexpr(COPY_FIRST)GeometryWideSocState<FUSED_FLOAT_WIDE_STATE>();
 if constexpr(HALF_FULL_FUSED_STATE && !SBG_PARSE_HALF_ALL_INPUTS_BEFORE_CTRL)GeometryWideSocState<true,false,!HALF_FULL_EARLY_ENTRY_ONES>();
#if SBG_PARSE_HALF_PAD_SCORE
  // Original output byte displacement also derives the anchor displacement.
#if SBG_PARSE_HALF_ALL_INPUTS_BEFORE_CTRL
  // The complete64 Half tile's byte origin is a multiple of512, as before.
  __builtin_assume((start & 511ULL)==0ULL);
#endif
  const uint64_t outputByteOffset=start/4ULL;
  uint64_t anchorAddress=reinterpret_cast<uint64_t>(anchors.GetPhyAddr(0));
  anchorAddress+=outputByteOffset;
  anchorAddress+=outputByteOffset;
  AscendC::GlobalTensor<T> anchorTile;
  anchorTile.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(anchorAddress));
#if SBG_PARSE_HALF_ANCHOR_TWO_BLOCK
  // Two adjacent equal blocks cover the original complete anchor interval.
  // Both zero strides retain a contiguous GM source and UB destination.
  AscendC::DataCopy(raw[1024*SCALE],anchorTile,
      AscendC::DataCopyParams{2,static_cast<uint16_t>(n*2U*sizeof(T)/(2U*32U)),0,0});
#else
  AscendC::DataCopy(raw[1024*SCALE],anchorTile,n*2);
#endif
#else
  AscendC::DataCopy(raw[1024*SCALE],anchors[start/(2ULL*sizeof(T))],n*2);
#endif
 }else if(n%8==0){
 #if SBG_PARSE_UNIT_PAD_SCORE
 AscendC::DataCopyPad(raw,boxes[(start+pos)*4],
     AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*4U*sizeof(T)),0,0,0},
     AscendC::DataCopyPadExtParams<T>{true,0,8,static_cast<T>(1.0f)});
#elif SBG_PARSE_HALF_UNIT_BOX_TWO_BURSTS
 // Two adjacent128B bursts retain the complete input and zero strides.
 AscendC::DataCopy(raw,boxes[(start+pos)*4],AscendC::DataCopyParams{2,4,0,0});
#else
 AscendC::DataCopy(raw,boxes[(start+pos)*4],n*4);
#endif
#if SBG_PARSE_UNIT_COPY_FIRST
 // Read-only input transfer can overlap the original complete CTRL reset.
 GeometryWideSocState<true,false,false>();
#endif
 if constexpr(FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO)AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
 if constexpr(COPY_FIRST)GeometryWideSocState<FUSED_FLOAT_WIDE_STATE>();
 if constexpr(HALF_FULL_FUSED_STATE && !SBG_PARSE_HALF_ALL_INPUTS_BEFORE_CTRL)GeometryWideSocState<true,false,!HALF_FULL_EARLY_ENTRY_ONES>();
 #if SBG_PARSE_UNIT_PAD_SCORE
 AscendC::DataCopyPad(raw[1024*SCALE],anchors[(start+pos)*2],
     AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*2U*sizeof(T)),0,0,0},
#if SBG_PARSE_UNIT_COPY_FIRST
     AscendC::DataCopyPadExtParams<T>{false,0,0,0});
#else
     AscendC::DataCopyPadExtParams<T>{true,0,8,static_cast<T>(0.0f)});
#endif
#else
#if SBG_PARSE_FLOAT_SHORT_POSITIVE_ZERO
 // The original real anchor interval gains one initialized private zero block.
 AscendC::DataCopyPad(raw[1024*SCALE],anchors[(start+pos)*2],
     AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*2U*sizeof(T)),0,0,0},
     AscendC::DataCopyPadExtParams<T>{true,0,8,static_cast<T>(0.0f)});
#else
 AscendC::DataCopy(raw[1024*SCALE],anchors[(start+pos)*2],n*2);
#endif
#endif
 }else{
 AscendC::DataCopyPad(raw,boxes[(start+pos)*4],AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*4*sizeof(T)),0,0,0},AscendC::DataCopyPadExtParams<T>{false,0,0,0});
 if constexpr(FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO)AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
 if constexpr(COPY_FIRST)GeometryWideSocState<FUSED_FLOAT_WIDE_STATE>();
 if constexpr(HALF_FULL_FUSED_STATE && !SBG_PARSE_HALF_ALL_INPUTS_BEFORE_CTRL)GeometryWideSocState<true,false,!HALF_FULL_EARLY_ENTRY_ONES>();
#if SBG_PARSE_FLOAT_SHORT_POSITIVE_ZERO
 AscendC::DataCopyPad(raw[1024*SCALE],anchors[(start+pos)*2],
     AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*2U*sizeof(T)),0,0,0},
     AscendC::DataCopyPadExtParams<T>{true,0,8,static_cast<T>(0.0f)});
#else
 AscendC::DataCopyPad(raw[1024*SCALE],anchors[(start+pos)*2],AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*2*sizeof(T)),0,0,0},AscendC::DataCopyPadExtParams<T>{false,0,0,0});
#endif
 }
 if constexpr(FLOAT_FOLD_RAW_GM_THRESHOLD || (FLOAT_COMPLETE_SCALAR_THRESHOLD_SELF_ZERO || FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO SBG_UNIT_THRESHOLD_OR SBG_SHORT_THRESHOLD_OR)){
  if constexpr((FLOAT_COMPLETE_SCALAR_THRESHOLD_SELF_ZERO || FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO SBG_UNIT_THRESHOLD_OR SBG_SHORT_THRESHOLD_OR))
   AscendC::DataCacheCleanAndInvalid<float,AscendC::CacheLine::SINGLE_CACHE_LINE,AscendC::DcciDst::CACHELINE_OUT>(threshold);
  // Invalidate this real input once per invocation; issue both independent
  // coordinate copies before the exact four-byte Scalar GM read.
  if constexpr(FLOAT_PARTIAL_POSITIVE_ZERO_PLAN SBG_UNIT_THRESHOLD_OR SBG_SHORT_THRESHOLD_OR)
   // Read exactly four real input bytes. Widening the unsigned bits to S64
   // is exact and permits a direct scalar zero predicate without Float punning.
   scalarThresholdIdentityBits=*reinterpret_cast<const volatile __gm__ uint32_t*>(threshold.GetPhyAddr());
  else scalarThreshold=*reinterpret_cast<const volatile __gm__ float*>(threshold.GetPhyAddr());
 }
 if constexpr(VECTOR_RATIO && !RATIO_ONE){if(pos==0){
  if constexpr(FLOAT_CONSTANTS)AscendC::DataCopy(mem[5896*SCALE],ratioGm,8);
  else AscendC::DataCopyPad(mem[((FUSED_META && !FOUR_FIELD_CAST)?5896:6016)*SCALE],ratioGm,AscendC::DataCopyExtParams{1,4,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
 }}
 if(hasThreshold && pos==0 && !FLOAT_FOLD_RAW_GM_THRESHOLD && !(FLOAT_COMPLETE_SCALAR_THRESHOLD_SELF_ZERO || FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO SBG_UNIT_THRESHOLD_OR SBG_SHORT_THRESHOLD_OR)){
 if(thresholdIsHalf)AscendC::DataCopyPad(thloc.ReinterpretCast<half>(),threshold16,AscendC::DataCopyExtParams{1,2,0,0,0},AscendC::DataCopyPadExtParams<half>{FOUR_FIELD_CAST,0,static_cast<uint8_t>(FOUR_FIELD_CAST?15U:0U),0});
 else {
 if constexpr(SINGLE_ROW_FIELDS)
  AscendC::DataCopyPad(thloc,threshold,AscendC::DataCopyExtParams{1,4,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
 else if constexpr(NORMAL_WIDE_SELECT || LITERAL_FLOAT_FULL09 || SHORT_THRESHOLD_ZERO || FOLD_PREFIX_ZERO)
  // The real threshold occupies lane0; its private32B block supplies zeros.
  AscendC::DataCopyPad(thloc,threshold,AscendC::DataCopyExtParams{1,4,0,0,0},AscendC::DataCopyPadExtParams<float>{true,0,7,0});
 else AscendC::DataCopyPad(FUSED_META?mem[5888*SCALE]:thloc,threshold,AscendC::DataCopyExtParams{1,4,0,0,0},AscendC::DataCopyPadExtParams<float>{false,0,0,0});
 }
 }
#if SBG_PARSE_HALF_ALL_INPUTS_BEFORE_CTRL
 // These three operations are input reads. Establish the same atomic-none,
 // normal mode before the original Vector wait/compute and every GM store.
 GeometryWideSocState<true,false,false>();
#endif
 // This AIV family has read-only MTE2 copies before any Vector operation or
 // GM store. Retain the SDK initialization once, during those pending reads.
 // This result region is disjoint from in-flight input and metadata DMA.
 if constexpr(WIDE && AscendC::IsSameType<T,half>::value){
 const uint8_t earlyRepeats=TILE_CLASS==70?8:static_cast<uint8_t>((n+63)/64);
 AscendC::Duplicate(result,1.0f,static_cast<uint32_t>(earlyRepeats)*64);
 }
 if constexpr(HALF_FULL_EARLY_ENTRY_ONES){
  // All128 private ones are disjoint from pending input and threshold copies.
  // The entry full128 mask remains active until the first mixed-width Cast.
#if !SBG_PARSE_HALF_PAD_SCORE
  AscendC::Duplicate<half,false>(mem[5760*SCALE].ReinterpretCast<half>(),half(1.0f),128,1,1,0);
#endif
 }
 // Constant buffers are disjoint from raw input and metadata DMA.
 if constexpr(SINGLE_ROW_FIELDS){
#if SBG_PARSE_FLOAT_SINGLE_ROW_RAW_CONSTANTS
  // No earlier Vector data producer precedes these two disjoint writes.
  // The original post-gather barrier completes both before score selection.
#if SBG_PARSE_FLOAT_SINGLE_ROW_RAW_REPEAT_ZERO
  // A single repeat never advances the destination repeat address.
  vector_dup((__ubuf__ float*)mem[5696*SCALE].GetPhyAddr(),0.0f,1,1,1,0,0);
  vector_dup((__ubuf__ float*)mem[5760*SCALE].GetPhyAddr(),1.0f,1,1,1,0,0);
#else
  vector_dup((__ubuf__ float*)mem[5696*SCALE].GetPhyAddr(),0.0f,1,1,1,8,0);
  vector_dup((__ubuf__ float*)mem[5760*SCALE].GetPhyAddr(),1.0f,1,1,1,8,0);
#endif
#else
  // Only lane0 is consumed after the original one-row mask. Its zero score
  // is independent of pending inputs and needs no threshold broadcast.
  AscendC::Duplicate<float,false>(mem[5696*SCALE],0.0f,64,1,1,8);
  // The private ones block is independent of all pending input reads.
  // Later one-row consumers read lane0 only.
  AscendC::Duplicate<float,false>(mem[5760*SCALE],1.0f,64,1,1,8);
#endif
 }
#if !SBG_PARSE_FLOAT_PAD_SCORE
 if constexpr(LITERAL_FLOAT_FULL09){
  // The private ones block is independent of all pending input reads.
  AscendC::Duplicate<float,false>(mem[5760*SCALE],1.0f,64,1,1,8);
 }
#endif
 if constexpr(EARLY_SMALL && AscendC::IsSameType<T,half>::value){
#if SBG_PARSE_HALF_SHORT_FOUR_FIELD
 // Keep the original mode reset once, then retain the full entry masks.
 // This Float producer uses low64 and remains before the input-ready wait.
 AscendC::SetMaskNorm();
 AscendC::Duplicate<float,false>(mem[5760*SCALE],1.0f,64,1,1,8);
#else
 if constexpr(SBG_PARSE_HALF_UNIT_SINGLE_CTRL)GeometryWideSocState<true,false,false>();
 else AscendC::SetMaskNorm();
 set_vector_mask(0,0xffffffffffffffffULL);
 if constexpr(UNIT32_GROUP)
  AscendC::Duplicate<half,false>(mem[5760*SCALE].ReinterpretCast<half>(),static_cast<half>(1.0f),128,1,1,8);
 else AscendC::Duplicate<float,false>(mem[5760*SCALE],1.0f,64,1,1,8);
 // The grouped metadata conversion supplies an initialized zero block.
 if constexpr(!UNIT32_GROUP)AscendC::Duplicate<float,false>(mem[5696*SCALE],0.0f,64,1,1,8);
#endif
 }
 if constexpr(LATE_SCALAR_RATIO){
 // Issue independent input transfers before reading the actual ratio.
 ratio=ratioGm.GetValue(0);
 }
 if constexpr(FLOAT_COMPLETE_VECTOR_COEFFICIENT && !FLOAT_COMPLETE_SCALAR_THRESHOLD_SELF_ZERO){
  // Input DMA does not touch the result region. Only its first eight Floats
  // are broadcast by Mul;the final Select overwrites this dead coefficient.
  AscendC::Duplicate<float,false>(result,0.8f,64,1,1,0);
 }
 if constexpr(FLOAT_COMPLETE_EARLY_SCORE_PLAN){
  // This private block starts after the complete Float score interval.
  // Only packed comparisons follow, so they preserve this zero CMPMASK.
  auto scoreOnes=mem[4608*SCALE];
  AscendC::Duplicate<float,false>(scoreOnes,1.0f,64,1,1,0);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Compare<float,false>(scoreOnes,scoreOnes,AscendC::CMPMODE::GT,64,AscendC::BinaryRepeatParams{1,0,1,0,0,0});
 }
 AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
#if SBG_PARSE_HALF_BOX_MATH_PIPELINE
 AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
#else
 if constexpr(FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO)AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
 else AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
#endif
#if SBG_PARSE_UNIT_COPY_FIRST
 // The original post-gather barrier also completes this disjoint zero block.
 auto unitScoreOnes=mem[4U*n];
 auto unitScoreZeros=mem[1024U*SCALE+2U*n];
 vsub((__ubuf__ float*)unitScoreZeros.GetPhyAddr(),
      (__ubuf__ float*)unitScoreOnes.GetPhyAddr(),
      (__ubuf__ float*)unitScoreOnes.GetPhyAddr(),1,1,0,0,0,0,0);
#endif
#if SBG_PARSE_HALF_REUSE_ANCHOR_BRCB
 // The anchor gather has completed before this broadcast executes. Reuse
 // its dead256B input tile, retaining the existing Cast/Sub completion fences.
#if SBG_PARSE_HALF_GROUPED_INPUT_REUSE
 // Grouped H2F has consumed this Half field before the retained Cast barrier.
 // This broadcast is disjoint from all converted fields and the bound.
 auto thresholdBroadcast=mem[SBG_PARSE_HALF_PAD_ONE_FIELD_BASE?4096:2304];
#else
 auto thresholdBroadcast=mem[512];
#endif
#else
 auto thresholdBroadcast=mem[5952*SCALE];
#endif
 if constexpr(SINGLE_ROW_FIELDS)thresholdBroadcast=thloc;
 if(pos==0){
 if constexpr(FOUR_FIELD_CAST){
  // The original coefficient is independent of the later grouped conversion.
  if constexpr(VECTOR_RATIO && !RATIO_ONE){
#if SBG_PARSE_HALF_SHORT_FOUR_FIELD
   AscendC::PipeBarrier<PIPE_V>();
   vbrcb((__ubuf__ uint32_t*)mem[6080*SCALE].GetPhyAddr(),
         (__ubuf__ uint32_t*)mem[6016*SCALE].GetPhyAddr(),1,8,1);
#else
   AscendC::Brcb(mem[6080*SCALE],mem[6016*SCALE],1,{1,8});
#endif
  }
 }else if constexpr(FUSED_META){
  if constexpr(thresholdIsHalf)
   AscendC::Cast(mem[5888*SCALE],thloc.ReinterpretCast<half>(),AscendC::RoundMode::CAST_NONE,1);
  AscendC::Brcb(mem[5952*SCALE],mem[5888*SCALE],2,{1,8});
 }else{
 if constexpr(VECTOR_RATIO && !RATIO_ONE)AscendC::Brcb(mem[6080*SCALE],mem[6016*SCALE],1,{1,8});
 if(hasThreshold && !SINGLE_ROW_FIELDS && !FLOAT_FOLD_RAW_GM_THRESHOLD && !(FLOAT_COMPLETE_SCALAR_THRESHOLD_SELF_ZERO || FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO SBG_UNIT_THRESHOLD_OR SBG_SHORT_THRESHOLD_OR)){
 auto source=thloc;
 if(thresholdIsHalf){source=mem[5888*SCALE];AscendC::Cast(source,thloc.ReinterpretCast<half>(),AscendC::RoundMode::CAST_NONE,1);}
 if constexpr(NORMAL_WIDE_SELECT && !ROTATE_WIDE)
  AscendC::Brcb(thresholdBroadcast,source,1,{1,0});
#if SBG_PARSE_FLOAT_FULL_RAW_BRCB
 else if constexpr(FLOAT_FULL_ENTRY_MASK_REUSE){
  // Both entry masks remain full before this independent scalar broadcast.
  // Preserve its original Vector fence and exact one-repeat descriptor.
  // The input-ready wait already completes the real threshold DMA.
  // Only entry masks precede this first Vector data producer.
  if constexpr(!SBG_PARSE_FLOAT_FIRST_BRCB_BARRIERLESS)AscendC::PipeBarrier<PIPE_V>();
  vbrcb((__ubuf__ uint32_t*)thresholdBroadcast.GetPhyAddr(),
        (__ubuf__ uint32_t*)source.GetPhyAddr(),1,SBG_PARSE_FLOAT_REPEAT_SHARE?0:8,1);
 }
#endif
 else if constexpr(FLOAT_UNIT_ENTRY_STATE){
  // C220 broadcasts the same eight scalar words independently of lane mask.
  // Keep the actual entry low64 mask and the original pre-broadcast fence.
  AscendC::PipeBarrier<PIPE_V>();
  vbrcb((__ubuf__ uint32_t*)thresholdBroadcast.GetPhyAddr(),
        (__ubuf__ uint32_t*)source.GetPhyAddr(),1,8,1);
 }else AscendC::Brcb(thresholdBroadcast,source,1,{1,8});
 }else if constexpr(!SINGLE_ROW_FIELDS && !FLOAT_FOLD_RAW_GM_THRESHOLD && !(FLOAT_COMPLETE_SCALAR_THRESHOLD_SELF_ZERO || FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO SBG_UNIT_THRESHOLD_OR SBG_SHORT_THRESHOLD_OR))AscendC::Duplicate(thresholdBroadcast,0.0f,8);
 }
 }
 if constexpr(AscendC::IsSameType<T,half>::value){
 auto yh1=mem[(SBG_PARSE_HALF_PAD_ONE_FIELD_BASE?3840:2048)*SCALE].ReinterpretCast<half>();auto yh2=mem[(SBG_PARSE_HALF_PAD_ONE_FIELD_BASE?4096:2304)*SCALE].ReinterpretCast<half>();auto ah=mem[(SBG_PARSE_HALF_PAD_ONE_FIELD_BASE?4352:2560)*SCALE].ReinterpretCast<half>();
 // No earlier data Vector instruction in this complete Half resource:
 // retain the entry normal/full128 mask for the three input extractions.
 if constexpr(!HALF_FULL_FUSED_STATE && !SBG_PARSE_HALF_SHORT_FOUR_FIELD){if constexpr(!SBG_PARSE_HALF_UNIT_SINGLE_CTRL)AscendC::SetMaskNorm();set_vector_mask(0xffffffffffffffffULL,0xffffffffffffffffULL);}
 vreducev2((__ubuf__ uint16_t*)yh1.GetPhyAddr(),(__ubuf__ uint16_t*)raw.GetPhyAddr(),(__ubuf__ uint16_t*)raw.GetPhyAddr(),static_cast<uint8_t>((n*4+127)/128),1,4,8,0);
 vreducev2((__ubuf__ uint16_t*)yh2.GetPhyAddr(),(__ubuf__ uint16_t*)raw.GetPhyAddr(),(__ubuf__ uint16_t*)raw.GetPhyAddr(),static_cast<uint8_t>((n*4+127)/128),1,6,8,0);
#if !SBG_PARSE_HALF_BOX_MATH_PIPELINE
 vreducev2((__ubuf__ uint16_t*)ah.GetPhyAddr(),(__ubuf__ uint16_t*)raw[1024*SCALE].GetPhyAddr(),(__ubuf__ uint16_t*)raw[1024*SCALE].GetPhyAddr(),static_cast<uint8_t>((n*2+127)/128),1,2,8,0);
#endif
 AscendC::PipeBarrier<PIPE_V>();
 if constexpr(FULL_STRIDE_CAST){
 // Each field is 1024 bytes apart in both source and destination.
 // Complete Half extraction has retained the normal mode established
 // after its first input copy. Only the128-to64 lane mask must change here.
 if constexpr(!HALF_FULL_FUSED_STATE && !SBG_PARSE_HALF_SHORT_FOUR_FIELD && !SBG_PARSE_HALF_UNIT_SINGLE_CTRL)AscendC::SetMaskNorm();
 if constexpr(!HALF_FULL_REUSE_CONVERSION_MASK && !SBG_PARSE_HALF_SHORT_FOUR_FIELD)set_vector_mask(0,UNIT32_GROUP?0xffffffffULL:0xffffffffffffffffULL);
 // The fourth field is the actual Half threshold. Unit32 uses its typed
 // padding zeros for Select; other converted padding lanes stay private.
 // Each source/destination field retains its original1024B pitch.
#if SBG_PARSE_HALF_BOX_MATH_PIPELINE
 // Original field pitch; exactly the two box fields are converted now.
 AscendC::Cast<float,half,false>(y1,yh1,AscendC::RoundMode::CAST_NONE,64,2,AscendC::UnaryRepeatParams{1,1,32,32});
#else
 AscendC::Cast<float,half,false>(y1,yh1,AscendC::RoundMode::CAST_NONE,UNIT32_GROUP?32:64,FOUR_FIELD_CAST?4:3,AscendC::UnaryRepeatParams{1,1,32,32});
#endif
 }else{
 AscendC::Cast(y1,yh1,AscendC::RoundMode::CAST_NONE,n);AscendC::Cast(y2,yh2,AscendC::RoundMode::CAST_NONE,n);AscendC::Cast(cy,ah,AscendC::RoundMode::CAST_NONE,n);
 }
 }else{
 if constexpr(SINGLE_ROW_FIELDS){
  // y1 already occupies lane0. Pattern5 places source lane2 in y2 lane0.
  // The existing full64 mask precedes this extraction;only lane0 is later used.
  ExtractFields(y2,y1,5,1);
  set_vector_mask(0,1ULL);
 }else{
 // This static Float scope never enters count-mask mode after initialization.
 if constexpr(!FUSED_FLOAT_WIDE_STATE && !FLOAT_COMPLETE_EARLY_SCORE_PLAN && !FLOAT_FULL_NORMAL64 && !FLOAT_SHORT_NORMAL64 && !FLOAT_UNIT_ENTRY_STATE)AscendC::SetMaskNorm();
 if constexpr(!FLOAT_COMPLETE_EARLY_SCORE_PLAN && !FLOAT_FULL_NORMAL64 && !FLOAT_SHORT_NORMAL64 && !FLOAT_UNIT_ENTRY_STATE)set_vector_mask(0,0xffffffffffffffffULL);
 ExtractFields(y1,bf,4,static_cast<uint8_t>((n*4+63)/64));
 ExtractFields(y2,bf,6,static_cast<uint8_t>((n*4+63)/64));
 // The box fields are independent of the pending anchor copy.
 if constexpr(FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO)AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 if constexpr(!FLOAT_COMPLETE_LATE_ANCHOR && !SBG_PARSE_FLOAT_THRESHOLD_REUSE_CY)ExtractFields(cy,af,2,static_cast<uint8_t>((n*2+63)/64));
 }
 }
 AscendC::PipeBarrier<PIPE_V>();
 if constexpr(FOUR_FIELD_CAST){
  if constexpr(!HALF_FULL_REUSE_CONVERSION_MASK){
#if SBG_PARSE_HALF_SHORT_FOUR_FIELD
   AscendC::PipeBarrier<PIPE_V>();
   vbrcb((__ubuf__ uint32_t*)thresholdBroadcast.GetPhyAddr(),
         (__ubuf__ uint32_t*)result.GetPhyAddr(),1,HALF_CURRENT_REPEAT,1);
#else
   if constexpr(!SBG_PARSE_HALF_UNIT_BRCB_SUB_OVERLAP)AscendC::Brcb(thresholdBroadcast,result,1,{1,HALF_CURRENT_REPEAT});
#endif
  }
  // The SDK broadcast restores a full64 mask; Unit32 retains its valid rows.
  if constexpr(UNIT32_GROUP)set_vector_mask(0,0xffffffffULL);
 }
 const uint8_t repeats=TILE_CLASS==70?8:TILE_CLASS==3?4:((FULL64 || FULL32 || TILE_CLASS==69 || TILE_CLASS==82)?1:static_cast<uint8_t>((n+63)/64));
 if constexpr(FULL64 || FULL32 || TILE_CLASS==69 || TILE_CLASS==82){
 if constexpr(!FULL_STRIDE_CAST && !FLOAT_FULL_NORMAL64 && !FLOAT_SHORT_NORMAL64 && !FLOAT_UNIT_ENTRY_STATE){AscendC::SetMaskNorm();set_vector_mask(0,0xffffffffffffffffULL);}
 auto originalY2=y2;
 if constexpr(HALF_BOUND_INPUT_REUSE || FLOAT_FOLD_BOUND_INPUT_REUSE){
  // Extraction and grouped conversion have passed their existing V barriers.
  // The input region is dead; this disjoint32B-aligned slot holds64 Float bounds.
#if SBG_PARSE_HALF_GROUPED_INPUT_REUSE
  // Grouped H2F has consumed the first Half field. The existing completion
  // barrier precedes this new, disjoint Float64 bound write.
#if !SBG_PARSE_HALF_BOUND_IN_PLACE
  y2=mem[2056];
#endif
  // The complete original y2 field permits full overlap for each Float
  // operation. The final Compare barrier releases it for Half output.
#else
  y2=mem[8];
#endif
 }
 AscendC::Sub<float,false>(y2,originalY2,y1,64,1,AscendC::BinaryRepeatParams{FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT});
#if SBG_PARSE_HALF_UNIT_BRCB_SUB_OVERLAP
 // The grouped Cast completion also releases the threshold field.
 // This broadcast and Sub access disjoint UB intervals. The original
 // Sub-to-waist-Add barrier completes both before either later consumer.
 vbrcb((__ubuf__ uint32_t*)thresholdBroadcast.GetPhyAddr(),
       (__ubuf__ uint32_t*)result.GetPhyAddr(),1,HALF_CURRENT_REPEAT,1);
#endif
#if SBG_PARSE_FLOAT_THRESHOLD_REUSE_CY
 // The prior field barrier releases the threshold read before cy overwrites it.
 // This gather is independent of Sub; the existing next barrier completes both.
 ExtractFields(cy,af,2,static_cast<uint8_t>((n*2+63)/64));
#endif
 if constexpr(HALF_FULL_REUSE_CONVERSION_MASK){
  // The completed threshold field and this private broadcast do not alias Sub.
  // The following original Sub-to-Muls barrier completes both operations.
#if !SBG_PARSE_HALF_BOX_MATH_PIPELINE
  vbrcb((__ubuf__ uint32_t*)thresholdBroadcast.GetPhyAddr(),
        (__ubuf__ uint32_t*)result.GetPhyAddr(),1,HALF_CURRENT_REPEAT,1);
#endif
 }
 if constexpr(!RATIO_ONE){
 if constexpr(VECTOR_RATIO)AscendC::Mul<float,false>(y2,y2,mem[((FUSED_META && !FOUR_FIELD_CAST)?6016:6080)*SCALE],64,1,AscendC::BinaryRepeatParams{FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,0,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT,0});
 else AscendC::Muls<float,false>(y2,y2,ratio,64,1,AscendC::UnaryRepeatParams{FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT});
 }
 AscendC::Add<float,false>(y2,y2,y1,64,1,AscendC::BinaryRepeatParams{FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT});
#if SBG_PARSE_HALF_BOX_MATH_PIPELINE
 // The early waist result is disjoint from pending anchor/threshold input.
 // Event0 completes both late inputs before their first actual readers.
 AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
 auto lateHalfAnchor=mem[(SBG_PARSE_HALF_PAD_ONE_FIELD_BASE?4352:2560)*SCALE].ReinterpretCast<half>();
 vreducev2((__ubuf__ uint16_t*)lateHalfAnchor.GetPhyAddr(),
     (__ubuf__ uint16_t*)raw[1024*SCALE].GetPhyAddr(),
     (__ubuf__ uint16_t*)raw[1024*SCALE].GetPhyAddr(),1,1,2,8,0);
 AscendC::PipeBarrier<PIPE_V>();
 // The retained gather completion also releases the now-dead anchor input.
 // cy and the real threshold retain their original1024B field pitch.
 AscendC::Cast<float,half,false>(cy,lateHalfAnchor,AscendC::RoundMode::CAST_NONE,64,2,AscendC::UnaryRepeatParams{1,1,32,32});
 AscendC::PipeBarrier<PIPE_V>();
 vbrcb((__ubuf__ uint32_t*)thresholdBroadcast.GetPhyAddr(),
       (__ubuf__ uint32_t*)result.GetPhyAddr(),1,HALF_CURRENT_REPEAT,1);
 // The final threshold Add consumes the newly converted real threshold.
 AscendC::PipeBarrier<PIPE_V>();
#endif
#if SBG_PARSE_UNIT_THRESHOLD_GUARD || SBG_PARSE_FLOAT_SHORT_POSITIVE_ZERO
 if constexpr(FLOAT_UNIT_SCALAR_POSITIVE_ZERO SBG_SHORT_THRESHOLD_OR){
  // Only actual positive-zero bits omit the final threshold addition.
  // All other patterns retain the real Float32 input and addition.
  if(__builtin_expect(scalarThresholdIdentityBits!=0,0)){
   float fallbackThreshold=*reinterpret_cast<const volatile __gm__ float*>(threshold.GetPhyAddr());
   AscendC::Adds<float,false>(y2,y2,fallbackThreshold,64,1,
       AscendC::UnaryRepeatParams{FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT});
  }
 }else
#endif
#if SBG_PARSE_HALF_ADD_SHARE
 // Only exchange the two sources of this final FP32 addition. Broadcast
 // stays on the threshold source; the bound retains contiguous64 addressing.
 AscendC::Add<float,false>(y2,thresholdBroadcast,y2,64,1,AscendC::BinaryRepeatParams{1,0,1,0,0,0});
#else
#if SBG_PARSE_FLOAT_REPEAT_SHARE
 // Same final FP32 addition with the threshold on the broadcast source.
 AscendC::Add<float,false>(y2,thresholdBroadcast,y2,64,1,AscendC::BinaryRepeatParams{1,0,1,0,0,0});
#else
 AscendC::Add<float,false>(y2,y2,thresholdBroadcast,64,1,AscendC::BinaryRepeatParams{FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,0,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT,0});
#endif
#endif
 }else if constexpr(((WIDE && (!ROTATE_WIDE || LITERAL_WIDE08)) ||
                            (FOLDED_ROWS!=0 && MIDPOINT_RATIO)) &&
                     AscendC::IsSameType<T,float>::value && AscendC::IsSameType<O,float>::value){
 // Scalar threshold and coefficient share the unary repeat descriptor.
 // Field extraction leaves the normal64 mask active. Extra rounded lanes
 // stay private in UB; only the original valid output interval is stored.
 auto originalY2=y2;
 if constexpr(FLOAT_FOLD_BOUND_INPUT_REUSE)y2=mem[8];
 AscendC::Sub<float,false>(y2,originalY2,y1,64,repeats,AscendC::BinaryRepeatParams{1,1,1,8,8,8});
 if constexpr(FLOAT_COMPLETE_LATE_ANCHOR){
  // The original gather barrier releases y1/y2. Anchor extraction reads a
  // separate input and writes cy,disjoint from this subtraction's operands.
  ExtractFields(cy,af,2,static_cast<uint8_t>((n*2+63)/64));
 }
 if constexpr(FLOAT_COMPLETE_VECTOR_COEFFICIENT && !FLOAT_COMPLETE_SCALAR_THRESHOLD_SELF_ZERO)
  AscendC::Mul<float,false>(y2,y2,result,64,repeats,AscendC::BinaryRepeatParams{1,1,0,8,8,0});
 else
 AscendC::Muls<float,false>(y2,y2,ratio,64,repeats,AscendC::UnaryRepeatParams{1,1,8,8});
 AscendC::Add<float,false>(y2,y2,y1,64,repeats,AscendC::BinaryRepeatParams{1,1,1,8,8,8});
 if constexpr(FLOAT_FOLD_RAW_GM_THRESHOLD || (FLOAT_COMPLETE_SCALAR_THRESHOLD_SELF_ZERO || FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO SBG_UNIT_THRESHOLD_OR SBG_SHORT_THRESHOLD_OR)){
  if constexpr(FLOAT_PARTIAL_POSITIVE_ZERO_PLAN){
   if(__builtin_expect(scalarThresholdIdentityBits!=0,0)){
    // All other bit patterns retain the actual Float32 threshold Add.
    // Inputs are stable for the invocation; no representation cast is needed.
    float fallbackThreshold=*reinterpret_cast<const volatile __gm__ float*>(threshold.GetPhyAddr());
    AscendC::Adds<float,false>(y2,y2,fallbackThreshold,64,repeats,AscendC::UnaryRepeatParams{1,1,8,8});
   }
  }else AscendC::Adds<float,false>(y2,y2,scalarThreshold,64,repeats,AscendC::UnaryRepeatParams{1,1,8,8});
 }else
 AscendC::Add<float,false>(y2,y2,thresholdBroadcast,64,repeats,AscendC::BinaryRepeatParams{1,1,0,8,8,0});
 }else{
 AscendC::Sub(y2,y2,y1,n);
 if constexpr(!RATIO_ONE){
 if constexpr(VECTOR_RATIO)AscendC::Mul(y2,y2,mem[6080*SCALE],64,repeats,AscendC::BinaryRepeatParams{1,1,0,8,8,0});
 else AscendC::Muls(y2,y2,ratio,n);
 }
 AscendC::Add(y2,y2,y1,n);
 AscendC::Add(y2,y2,thresholdBroadcast,64,repeats,AscendC::BinaryRepeatParams{1,1,0,8,8,0});
 }
 if(n<=64){
 auto ones=mem[(FLOAT_CONSTANTS?6024U:5760U)*SCALE];
 auto zeros=mem[((FOUR_FIELD_CAST || LITERAL_FLOAT_FULL09)?5960U:(FLOAT_CONSTANTS?6032U:5696U))*SCALE];
#if SBG_PARSE_HALF_REUSE_ANCHOR_BRCB
 // Original threshold-padding lane1 expands to eight Float zeros here.
 zeros=thresholdBroadcast[8];
#endif
#if SBG_PARSE_HALF_PAD_SCORE
 // The original input-ready event completes the sixteen padded Half ones.
 ones=mem[n*4U*sizeof(T)/sizeof(float)];
#endif
#if SBG_PARSE_UNIT_PAD_SCORE
 // Original coordinate-ready synchronization completes these private pads.
 ones=mem[4U*n];
 zeros=mem[1024U*SCALE+2U*n];
#endif
#if SBG_PARSE_FLOAT_PAD_SCORE
 // The original MTE2-ready event completes these eight private Float ones.
 ones=mem[n*4U];
#endif
 // The first zero input lane expands into eight initialized Float zeros.
 if constexpr((SHORT_THRESHOLD_ZERO && !SINGLE_ROW_FIELDS) || FOLD_PREFIX_ZERO)zeros=thresholdBroadcast[8];
#if SBG_PARSE_FLOAT_SHORT_POSITIVE_ZERO
 // Right8 Float padding defines [alignUp(realBytes,32),+32) as zeros.
 // The retained input-ready event completes this disjoint private block.
 if constexpr(!SINGLE_ROW_FIELDS)zeros=mem[1024U*SCALE+((n*2U+7U)&~7U)];
#endif
 if constexpr(FOUR_FIELD_CAST || LITERAL_FLOAT_FULL09){
  // The literal Float owner already initialized its independent ones block.
  if constexpr(HALF_FULL_FUSED_STATE){
   // Both Half blocks are private; initialize all128 ones under the
   // retained full mask. Only the first64 selected scores are stored to GM.
   if constexpr(!HALF_FULL_EARLY_ENTRY_ONES)
   AscendC::Duplicate<half,false>(ones.ReinterpretCast<half>(),half(1.0f),128,1,1,HALF_CURRENT_REPEAT);
  }else if constexpr(!LITERAL_FLOAT_FULL09 && !UNIT32_GROUP && !SBG_PARSE_HALF_SHORT_FOUR_FIELD)AscendC::Duplicate<float,false>(ones,1.0f,64,1,1,8);
 }else if constexpr(!FLOAT_CONSTANTS && !(EARLY_SMALL && AscendC::IsSameType<T,half>::value) && !SINGLE_ROW_FIELDS){
#if !SBG_PARSE_UNIT_PAD_SCORE
 AscendC::Duplicate<float,false>(ones,1.0f,64,1,1,8);
 if constexpr(!SHORT_THRESHOLD_ZERO && !FOLD_PREFIX_ZERO)AscendC::Duplicate<float,false>(zeros,0.0f,64,1,1,8);
#endif
 }
 AscendC::Compare<float,false>(cy,y2,AscendC::CMPMODE::GT,64,AscendC::BinaryRepeatParams{FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT});
#if SBG_PARSE_UNIT_PAD_SCORE
 // Compare is the last cy reader; Select's existing fence releases its buffer.
 if constexpr(FLOAT_UNIT_SCALAR_POSITIVE_ZERO)result=cy;
#endif
 if constexpr(UNIT32_GROUP || HALF_FULL_FUSED_STATE)
  // Exact Boolean0/1 values need no intermediate Float score conversion.
#if SBG_PARSE_HALF_PAD_SCORE
#if SBG_PARSE_HALF_SCORE_REUSE_Y2
  // Sub has consumed the original Float y2. All later bounds occupy a
  // separate region, so the unchanged Select can reuse these256 bytes.
  AscendC::Select<half,AscendC::SELMODE::VSEL_CMPMASK_SPR>(mem[256].ReinterpretCast<half>(),
#else
  AscendC::Select<half,AscendC::SELMODE::VSEL_CMPMASK_SPR>(mem[5632*SCALE].ReinterpretCast<half>(),
#endif
      ones.ReinterpretCast<half>(),zeros.ReinterpretCast<half>(),1,AscendC::BinaryRepeatParams{1,0,0,0,0,0});
#else
  AscendC::Select<half,AscendC::SELMODE::VSEL_CMPMASK_SPR>(mem[5632*SCALE].ReinterpretCast<half>(),
      ones.ReinterpretCast<half>(),zeros.ReinterpretCast<half>(),1,AscendC::BinaryRepeatParams{FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,0,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT,0});
#endif
 else
 if constexpr(FLOAT_CONSTANTS)
  AscendC::Select<float,AscendC::SELMODE::VSEL_CMPMASK_SPR>(result,ones,zeros,1,AscendC::BinaryRepeatParams{1,0,0,8,0,0});
 else
 if constexpr(FOUR_FIELD_CAST || LITERAL_FLOAT_FULL09 || SHORT_THRESHOLD_ZERO || FOLD_PREFIX_ZERO)
 #if SBG_PARSE_FLOAT_PAD_SCORE
 AscendC::Select<float,AscendC::SELMODE::VSEL_CMPMASK_SPR>(result,ones,zeros,1,AscendC::BinaryRepeatParams{1,0,0,0,0,0});
#else
 AscendC::Select<float,AscendC::SELMODE::VSEL_CMPMASK_SPR>(result,ones,zeros,1,AscendC::BinaryRepeatParams{FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,0,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT,0});
#endif
 else
#if SBG_PARSE_UNIT_PAD_SCORE
 AscendC::Select<float,AscendC::SELMODE::VSEL_CMPMASK_SPR>(result,ones,zeros,1,
     AscendC::BinaryRepeatParams{1,0,0,0,0,0});
#else
 AscendC::Select<float,AscendC::SELMODE::VSEL_CMPMASK_SPR>(result,ones,zeros,1,AscendC::BinaryRepeatParams{FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,FLOAT_SHORT_CURRENT_BLOCK,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT,HALF_CURRENT_REPEAT});
#endif
 }else{
 if constexpr(FLOAT_PARTIAL_ONES_BEFORE_PACKED){
  // Complete the last waist read of y1 before its original ones overwrite.
  // The following packed comparison reads cy/y2 and writes a disjoint bitmap.
  AscendC::PipeBarrier<PIPE_V>();
  // Exact C220 Duplicate intrinsic: same64 distinct writes and descriptor.
  vector_dup((__ubuf__ float*)y1.GetPhyAddr(),1.0f,1,1,1,0,0);
 }
 if constexpr(FLOAT_PARTIAL_ONES_BEFORE_PACKED){
  // Exact C220 CompareCompute expansion for this original8-repeat domain.
  vcmpv_gt((__ubuf__ uint8_t*)mask.GetPhyAddr(),
           (__ubuf__ float*)cy.GetPhyAddr(),(__ubuf__ float*)y2.GetPhyAddr(),
           repeats,1,1,1,8,8,8);
 }else{
 AscendC::Compare(mask,cy,y2,AscendC::CMPMODE::GT,static_cast<uint32_t>(repeats)*64);
 }
 if constexpr(NORMAL_WIDE_SELECT){
  // Packed Compare preserves the normal64 mask from extraction and math.
  // y1 is dead after waist math. Its independent ones block stays unchanged
  // while all output repeats consume it through zero source strides.
  AscendC::PipeBarrier<PIPE_V>();
  // The constant tile has one repeat; no second repeat uses its pitch.
  if constexpr(!FLOAT_COMPLETE_EARLY_SCORE_PLAN){
   if constexpr(!FLOAT_PARTIAL_ONES_BEFORE_PACKED){
   if constexpr(!ROTATE_WIDE)AscendC::Duplicate<float,false>(y1,1.0f,64,1,1,0);
   else if constexpr(FLOAT_PARTIAL_SHARED_SCORE_DESCRIPTOR)
    AscendC::Duplicate<float,false>(y1,1.0f,64,1,1,0);
   else AscendC::Duplicate<float,false>(y1,1.0f,64,1,1,8);
   AscendC::PipeBarrier<PIPE_V>();
   }
   if constexpr((FLOAT_COMPLETE_SCALAR_THRESHOLD_SELF_ZERO || FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO SBG_UNIT_THRESHOLD_OR SBG_SHORT_THRESHOLD_OR || FLOAT_FOLD_RAW_GM_THRESHOLD)){
    // Both sources read initialized finite ones, so every strict-GT bit is zero.
    // One source broadcasts8 Floats; the other reads all64 initialized Floats.
    if constexpr(FLOAT_PARTIAL_SHARED_SCORE_DESCRIPTOR)
     AscendC::Compare<float,false>(y1,y1,AscendC::CMPMODE::GT,64,AscendC::BinaryRepeatParams{1,0,1,0,0,0});
    else AscendC::Compare<float,false>(y1,y1,AscendC::CMPMODE::GT,64,AscendC::BinaryRepeatParams{0,0,0,0,0,0});
   }else AscendC::SetCmpMask(thloc[4]);
   AscendC::PipeBarrier<PIPE_V>();
  }
  auto selectOnes=y1;
  if constexpr(FLOAT_COMPLETE_EARLY_SCORE_PLAN)selectOnes=mem[4608*SCALE];
  // The retained packed-Compare barrier completes the last bound read.
  // Reuse its private aligned region for the final Boolean scores.
  if constexpr(FLOAT_FOLD_RAW_GM_THRESHOLD)result=mem[8];
  AscendC::Select<float,uint8_t>(result,mask,selectOnes,repeats,AscendC::BinaryRepeatParams{1,0,1,8,0,8});
 }else{
 if constexpr(!(WIDE && AscendC::IsSameType<T,half>::value))AscendC::Duplicate(result,1.0f,static_cast<uint32_t>(repeats)*64);
 AscendC::Select(result,mask,result,0.0f,AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,static_cast<uint32_t>(repeats)*64);
 }
 }
#if SBG_PARSE_HALF_SCORE_REUSE_Y2
 auto outputLocal=mem[256].ReinterpretCast<O>();
#else
 auto outputLocal=mem[5632*SCALE].ReinterpretCast<O>();
#endif
 if constexpr(AscendC::IsSameType<O,half>::value){
  if constexpr(!UNIT32_GROUP && !HALF_FULL_FUSED_STATE)
#if SBG_PARSE_HALF_SHORT_FOUR_FIELD
   // Float scores are already initialized0/1 in all64 lanes. Only the
   // original n Half results are copied to GM; the remainder stays private.
   AscendC::Cast<half,float,false>(outputLocal,result,AscendC::RoundMode::CAST_RINT,64,1,AscendC::UnaryRepeatParams{1,1,4,8});
#else
   AscendC::Cast(outputLocal,result,AscendC::RoundMode::CAST_RINT,n);
#endif
 }
 else {outputLocal=result;}
 AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
 if constexpr(BYTE_PREFIX_ORIGIN){
  // Two adjacent blocks retain the original complete prefix interval.
  if constexpr(FLOAT_PARTIAL_SCALAR_THRESHOLD_SELF_ZERO)
   AscendC::DataCopy(out[start/(4ULL*sizeof(O))],outputLocal,AscendC::DataCopyParams{2,static_cast<uint16_t>(n*sizeof(O)/(2U*32U)),0,0});
  else AscendC::DataCopy(out[start/(4ULL*sizeof(O))],outputLocal,n);
 }
 else if(n%(32/sizeof(O))==0)AscendC::DataCopy(out[start+pos],outputLocal,n);
 else AscendC::DataCopyPad(out[start+pos],outputLocal,AscendC::DataCopyExtParams{1,static_cast<uint32_t>(n*sizeof(O)),0,0,0});
 if constexpr(!SINGLE_TILE){if(pos+256<count){AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);}}
 }
 }
#endif
};
#endif
template<class T,class O,uint32_t THRESHOLD_KIND,uint32_t TILE_CLASS,bool USE_VECTOR_RATIO=(TILE_CLASS!=0 && TILE_CLASS!=2),bool EARLY_SMALL=false,bool ROTATE_WIDE=false,uint32_t ROTATE_WORKERS=0,bool NATIVE_WIDE=false,uint32_t CONSTANT_WIDE_TAIL=0,uint32_t FOLDED_ROWS=0,bool MIDPOINT_RATIO=false,bool LATE_NARROW_RATIO=false,uint32_t FOLDED_WIDTH=64,uint32_t CONSTANT_SMALL_ROWS=0>
using CachedOwnerGeometry=Geometry<T,O,THRESHOLD_KIND,TILE_CLASS,USE_VECTOR_RATIO,
 EARLY_SMALL,ROTATE_WIDE,ROTATE_WORKERS,NATIVE_WIDE,CONSTANT_WIDE_TAIL,FOLDED_ROWS,
 MIDPOINT_RATIO,LATE_NARROW_RATIO,FOLDED_WIDTH,CONSTANT_SMALL_ROWS,true>;
template<uint32_t TILE_CLASS,bool SCALAR_RATIO=(TILE_CLASS==0 || TILE_CLASS==2)>
__aicore__ inline void ReadGeometryHeader(GM_ADDR tiling,SafetyBeltViolationGeometryTilingData &d){
 if constexpr(TILE_CLASS==0 || TILE_CLASS==2 || TILE_CLASS==69 || TILE_CLASS==82 || TILE_CLASS==71){
 AscendC::GlobalTensor<uint32_t> tilingWords;tilingWords.SetGlobalBuffer((__gm__ uint32_t*)tiling);
 AscendC::LocalTensor<uint32_t> header(AscendC::TPosition::VECCALC,24576U,8U);
 AscendC::DataCopyPad(header,tilingWords,AscendC::DataCopyExtParams{1,SCALAR_RATIO?24U:4U,0,0,0},AscendC::DataCopyPadExtParams<uint32_t>{false,0,0,0});
 AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
 d.batch=header.GetValue(0);

 if constexpr(TILE_CLASS==0){
  d.perCore=header.GetValue(1);
  }
 if constexpr(SCALAR_RATIO)d.waistRatio=header.ReinterpretCast<float>().GetValue(5);
 }
}

#include "geometry_raw_wide.h"
#include "geometry_raw_folded.h"
#include "geometry_raw_half.h"
#include "geometry_raw_small.h"

template<class Op>
__aicore__ __attribute__((always_inline)) inline void GeometryExecute(
    GM_ADDR b,GM_ADDR a,GM_ADDR t,GM_ADDR o,GM_ADDR tiling,
    const SafetyBeltViolationGeometryTilingData &d,int64_t logicalOwner=0) {
 Op op;
 op.Init(b,a,t,o,tiling,d,logicalOwner);
 op.Process();
}

#include "geometry_dispatch.h"
#include "geometry_dispatch_small.h"
#include "geometry_dispatch_folded.h"
#include "geometry_dispatch_wide.h"
#include "geometry_dispatch_generic.h"
#include "geometry_dispatch_raw.h"
#include "geometry_dispatch_routes.h"
#include "geometry_dispatch_entry.h"
#include "geometry_dispatch_parsed.h"
#undef SBG_DISPATCH_TEMPLATE
#undef SBG_DISPATCH_TYPE

template<typename DT_PERSON_BOXES,typename DT_OUTPUT,uint32_t THRESHOLD_KIND,uint32_t TILE_CLASS,uint32_t ADDRESS_PLAN,uint32_t BROADCAST_PLAN,uint32_t FLOAT_METADATA_PLAN,uint32_t REPEAT_ADDRESS_PLAN>
__global__ __aicore__ void safety_belt_violation_geometry(GM_ADDR person_boxes,GM_ADDR anchor_points,GM_ADDR threshold,GM_ADDR violation_score,GM_ADDR workspace,GM_ADDR tiling) {
 GeometryDispatch<DT_PERSON_BOXES,DT_OUTPUT,THRESHOLD_KIND,TILE_CLASS,ADDRESS_PLAN,BROADCAST_PLAN,FLOAT_METADATA_PLAN,REPEAT_ADDRESS_PLAN>::Run(person_boxes,anchor_points,threshold,violation_score,workspace,tiling);
}

// CANN8.5 TILING_KEY_VAR uses the declaration's two 8-bit datatype fields,
// a 2-bit threshold index and an 8-bit tile-class index. TILE_CLASS keeps its
// original contiguous indices, extended through242,, so its index equals its value.
// The selected static structural families never consumes system/user workspace.
#if defined(TILING_KEY_VAR) && defined(DTYPE_PERSON_BOXES) && defined(DTYPE_VIOLATION_SCORE)
namespace AscendC {
template<uint64_t Key> struct GeometryWorkspacePolicy {
 #if defined(ASCENDC_DUMP) || defined(ASCENDC_TIME_STAMP_ON) || defined(ASCENDC_OOM) || ASCENDC_CPU_DEBUG
 static constexpr bool unused=false;
 #else
 static constexpr uint32_t tile=static_cast<uint32_t>((Key>>18)&0xffU);
 static constexpr bool identity32=(tile>=27U && tile<=35U);
 static constexpr bool single_repeat=(tile>=179U && tile<=241U);
 static constexpr bool folded128=(tile>=155U && tile<=178U);
 static constexpr bool inherited_unused=AscendC::IsSameType<DTYPE_PERSON_BOXES,half>::value &&
     AscendC::IsSameType<DTYPE_VIOLATION_SCORE,half>::value &&
     (Key&0xffU)==1U && ((Key>>8)&0xffU)==1U && ((Key>>16)&3U)==2U &&
     (identity32 || single_repeat || folded128);
 static constexpr bool address_inherited_unused=inherited_unused || (((Key>>16)&3U)==3U && ((((tile>=36U && tile<=98U)) && (Key&0xffffU)==0U && AscendC::IsSameType<DTYPE_PERSON_BOXES,float>::value && AscendC::IsSameType<DTYPE_VIOLATION_SCORE,float>::value) || (tile>=155U && tile<=178U && (Key&0xffffU)==0x101U && AscendC::IsSameType<DTYPE_PERSON_BOXES,half>::value && AscendC::IsSameType<DTYPE_VIOLATION_SCORE,half>::value) || (tile>=179U && tile<=241U && (Key&0xffffU)==0U && AscendC::IsSameType<DTYPE_PERSON_BOXES,float>::value && AscendC::IsSameType<DTYPE_VIOLATION_SCORE,float>::value)));
 static constexpr bool optimizedAddress=((Key>>26)&1U)!=0;
 static constexpr bool floatRoot=AscendC::IsSameType<DTYPE_PERSON_BOXES,float>::value && AscendC::IsSameType<DTYPE_VIOLATION_SCORE,float>::value && (Key&0xffffU)==0U;
 static constexpr bool before_half_wide_unused=address_inherited_unused || (optimizedAddress && floatRoot &&
     ((Key>>16)&3U)==1U && tile>=27U && tile<=35U);
 static constexpr bool unitEntry=floatRoot && ((Key>>16U)&3U)==1U &&
     ((Key>>28U)&3U)==3U && ((Key>>26U)&3U)==3U &&
     ((tile>=159U && tile<=160U) || (tile>=243U && tile<=244U) SBG_UNIT_PAD_EXTRA_TILE(tile));

 // The complete Half numerical body has no system/user workspace consumer.
 #if __NPU_ARCH__ == 2201 && (((TILING_KEY_VAR>>18U)&255ULL)==247ULL || SBG_PARSE_HALF_ADD_SHARE)
 static constexpr bool halfPadUnused=(Key&0xffffU)==0x101U && (tile==247U SBG_HALF_ADD_SHARE_EXTRA_TILE(tile)) &&
     ((Key>>16U)&3U)==3U && ((Key>>26U)&15U)==15U &&
     GeometryHalfFullInitPolicy<DTYPE_PERSON_BOXES,DTYPE_VIOLATION_SCORE,
       static_cast<uint32_t>((Key>>16U)&3U),tile,
       static_cast<uint32_t>((Key>>27U)&1U)>::enabled;
 #else
 static constexpr bool halfPadUnused=false;
 #endif
 #if SBG_PARSE_FOLD_MINIMUM
 static constexpr bool minimumFoldUnused=floatRoot && ((Key>>26U)&15U)==15U &&
     (tile>=251U && tile<=254U) && (((Key>>16U)&3U)==1U || ((Key>>16U)&3U)==3U);
 #else
 static constexpr bool minimumFoldUnused=false;
 #endif
 static constexpr bool unused=SBG_PARSE_FLOAT_BATCH_MERGE || minimumFoldUnused || halfPadUnused || unitEntry || before_half_wide_unused ||
     (floatRoot && (tile==135U SBG_FLOAT_FULL_EXTRA_TILE(tile)) && ((Key>>16U)&3U)==3U && ((Key>>26U)&15U)==15U) ||
     // The literal complete-Float family also has no workspace consumer.
     (floatRoot && tile==5U && ((Key>>16)&3U)==3U) ||
     // These static full-repeat owners consume no workspace.
     (floatRoot && ((Key>>16)&3U)==1U && tile>=3U && tile<=17U && (tile&1U)!=0U) ||
     // Static Float midpoint cohorts use no system or user workspace.
     (floatRoot && ((Key>>16)&3U)==1U && tile>=99U && tile<=154U) ||
     (floatRoot && tile==19U && (((Key>>16)&3U)==1U || ((Key>>16)&3U)==3U)) ||
     (((Key>>16)&3U)==3U && tile==19U && (Key&0xffffU)==0x101U &&
      AscendC::IsSameType<DTYPE_PERSON_BOXES,half>::value && AscendC::IsSameType<DTYPE_VIOLATION_SCORE,half>::value);
 #endif
};
template<uint64_t Key>
__aicore__ inline void GeometryCompilerWorkspace(GM_ADDR workspace){
 if constexpr(!GeometryWorkspacePolicy<Key>::unused)
  (SetSysWorkspaceForce)(workspace);
}
template<uint64_t Key>
__aicore__ inline GM_ADDR GeometryCompilerUserWorkspace(GM_ADDR workspace){
 if constexpr(GeometryWorkspacePolicy<Key>::unused)
  return nullptr;
 else
  return (GetUserWorkspace)(workspace);
}
}
// Both calls remain at their original SDK wrapper positions. Nonselected
// instantiations call the real SDK functions with the original system base.
// The selected body never reads its workspace argument; its getter is omitted
// together with the unused save, without reading an uninitialized global.
#define SetSysWorkspaceForce(...) GeometryCompilerWorkspace<static_cast<uint64_t>(TILING_KEY_VAR)>(__VA_ARGS__)
#define GetUserWorkspace(...) GeometryCompilerUserWorkspace<static_cast<uint64_t>(TILING_KEY_VAR)>(__VA_ARGS__)
#endif
