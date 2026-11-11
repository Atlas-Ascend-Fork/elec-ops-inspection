// Copyright 2026 Qianqiuer
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
// Inline implementation of KernelTreeBatch.

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, class StageCertHigh_Type, class StageCert_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageRecord_Type, class StageSecond_Type, class StageTileIdx_Type, class StageSameType, class StageUnaryType, class StageJType, class StageChunkMinimumType, class StageChunkUpperType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixStateMergePrefixMinimumStage2(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageRecord_Type& record_, StageSecond_Type& second_, StageTileIdx_Type& tileIdx_, StageSameType& same, StageUnaryType& unary, StageJType& j, StageChunkMinimumType& chunkMinimum, StageChunkUpperType& chunkUpper) {
        if constexpr(COMPACT_TAIL) {MergePrefixMinimum<HALF_RESOURCE>(j);} else {

            Compare<float,false>(chunkMinimum,lo_,CMPMODE::LT,uint64_t(64),same);
            Max<float,false>(record_,chunkMinimum,lo_,uint64_t(64),1,same);
            Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
            Min<float,false>(lo_,lo_,chunkMinimum,uint64_t(64),1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,chunkUpper,hi_,1,same);
            if constexpr(!HALF_RESOURCE) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,PrefixW<HALF_RESOURCE>(0),cert_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,PrefixW<HALF_RESOURCE>(1),certHigh_,1,same);
            }

            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
#else
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixState(uint32_t liveRows_,uint32_t segmentCount) {
        // Fold the conservative prefix interval into the norm before the projection FMA.
        // Keep W4 immutable so only the winning norm needs an error-scale operation.
        constexpr float intervalFactor=1.621246337890625e-5F;
        constexpr uint32_t lowerPlane=5U;
        constexpr uint32_t errorPlane=4U;
        SharedPrefixArena20<HALF_RESOURCE>();
        TreePrefixSearchState<HALF_RESOURCE> arena;
        constexpr uint32_t pp_=1056;
        InitializeGroupedSearchState(arena.lo_,arena.idx_,3);

        PipeBarrier<PIPE_V>();
        const uint64_t rowMask[2]={liveRows_==64?~uint64_t(0):((uint64_t(1)<<liveRows_)-1),uint64_t(0)};

        SearchPrefixStateSavePrefixPredicateStage<HALF_RESOURCE, lowerPlane, errorPlane, pp_>(liveRows_, segmentCount, intervalFactor, arena.certHigh_, arena.cert_, arena.chunkLower_, arena.hi_, arena.idx_, arena.lo_, arena.mask_, arena.offset_, arena.oneHot_, arena.record_, arena.refined_, arena.rowBase_, arena.second_, arena.segmentIndices_, arena.tileHi_, arena.tileIdx_, arena.tileLo_, arena.valid_, arena.xyz_, arena.same, arena.segment, arena.pointSegment, arena.unary, rowMask);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
#else
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, auto lowerPlane, auto errorPlane, auto pp_, class StageLiveRows_Type, class StageSegmentCountType, class StageIntervalFactorType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageOneHot_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageSegmentIndices_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageValid_Type, class StageXyz_Type, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType, class StageRowMaskType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixStateSavePrefixPredicateStage(StageLiveRows_Type& liveRows_, StageSegmentCountType& segmentCount, StageIntervalFactorType& intervalFactor, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageSegmentIndices_Type& segmentIndices_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageValid_Type& valid_, StageXyz_Type& xyz_, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary, StageRowMaskType& rowMask) {
        SearchPrefixStateSavePrefixPredicateStage2<HALF_RESOURCE, lowerPlane, errorPlane, pp_>(liveRows_, segmentCount, intervalFactor, certHigh_, cert_, chunkLower_, hi_, idx_, lo_, mask_, offset_, record_, refined_, rowBase_, second_, segmentIndices_, tileHi_, tileIdx_, tileLo_, valid_, xyz_, same, segment, pointSegment, unary, rowMask);
        // Refinement gathers a full register, including inactive point lanes.
        if(liveRows_<64) {
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),~((uint64_t(1)<<liveRows_)-1));
            Duplicate<int32_t,false>(idx_,int32_t(0),uint64_t(64),1,1,8);
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
        PipeBarrier<PIPE_V>();

        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        SharedPrefixPredicate1<HALF_RESOURCE>(offset_, idx_, oneHot_, cert_, certHigh_, valid_);


    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
#else
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, auto lowerPlane, auto errorPlane, auto pp_, class StageLiveRows_Type, class StageSegmentCountType, class StageIntervalFactorType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageSegmentIndices_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageValid_Type, class StageXyz_Type, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType, class StageRowMaskType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixStateSavePrefixPredicateStage2(StageLiveRows_Type& liveRows_, StageSegmentCountType& segmentCount, StageIntervalFactorType& intervalFactor, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageSegmentIndices_Type& segmentIndices_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageValid_Type& valid_, StageXyz_Type& xyz_, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary, StageRowMaskType& rowMask) {
        for(uint32_t j=0;j<segmentCount;j+=64) {
        if constexpr(COMPACT_TAIL) {
            if(segmentCount-j<=8) {
                SavePrefixPredicate<HALF_RESOURCE>(j,segmentCount-j);
                CompactPrefixTail<HALF_RESOURCE>(j,segmentCount-j);MergePrefixMinimum<HALF_RESOURCE>(j);continue;
            }
        }
        // Initialization and the previous merge leave the 64-lane normal mask.
    #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(PrefixW<HALF_RESOURCE>(axis),PrefixP<HALF_RESOURCE>(axis),PrefixC<HALF_RESOURCE>(axis)[j],uint64_t(64),liveRows_,pointSegment);
        if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
        Mul<float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(0),PrefixC<HALF_RESOURCE>(6)[j],uint64_t(64),liveRows_,segment);
        Mul<float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(0),uint64_t(64),liveRows_,same);

        if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(1),PrefixC<HALF_RESOURCE>(7)[j],uint64_t(64),liveRows_,segment);
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(1),PrefixW<HALF_RESOURCE>(1),uint64_t(64),liveRows_,same);

        if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(2),PrefixC<HALF_RESOURCE>(8)[j],uint64_t(64),liveRows_,segment);
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(2),uint64_t(64),liveRows_,same);

        if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
        // APz is dead; its bank phase differs from the dot accumulator.
        constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
        const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
        Muls<float,false>(PrefixW<HALF_RESOURCE>(5),PrefixW<HALF_RESOURCE>(4),1.0F-intervalFactor,uint64_t(64),liveRows_,unary);
        Maxs<float,false>(PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(3),0.0F,uint64_t(64),liveRows_,unary);if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
        Min<float,false>(PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(2),PrefixC<HALF_RESOURCE>(9)[j],uint64_t(64),liveRows_,segment);if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
        FusedMulAdd<float,false>(PrefixW<HALF_RESOURCE>(3),minusTwo,PrefixW<HALF_RESOURCE>(2),uint64_t(64),liveRows_,BinaryRepeatParams{1,0,1,8,0,8});if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(5),PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(3),uint64_t(64),liveRows_,same);if(liveRows_<ROWS/2)PipeBarrier<PIPE_V>();
        if(j+64>segmentCount) {
            CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(segmentCount-j)),CMPMODE::LT,uint64_t(64),liveRows_,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
            Select(PrefixW<HALF_RESOURCE>(lowerPlane),valid_,PrefixW<HALF_RESOURCE>(lowerPlane),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(liveRows_*64));PipeBarrier<PIPE_V>();
            // The count-based tail select restores a full mask; narrow it here.
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            WholeReduceMin<float,false>(record_,PrefixW<HALF_RESOURCE>(lowerPlane),uint64_t(64),liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        } else WholeReduceMin(record_,PrefixW<HALF_RESOURCE>(lowerPlane),64,liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        PipeBarrier<PIPE_V>();
        GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
        GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));SearchPrefixStateCopyStage<HALF_RESOURCE, lowerPlane, errorPlane, pp_>(liveRows_, intervalFactor, certHigh_, cert_, chunkLower_, hi_, idx_, lo_, mask_, offset_, record_, refined_, rowBase_, second_, tileHi_, tileIdx_, tileLo_, xyz_, same, unary, rowMask, j);}
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
#else
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, auto lowerPlane, auto errorPlane, auto pp_, class StageLiveRows_Type, class StageIntervalFactorType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageXyz_Type, class StageSameType, class StageUnaryType, class StageRowMaskType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixStateCopyStage(StageLiveRows_Type& liveRows_, StageIntervalFactorType& intervalFactor, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageXyz_Type& xyz_, StageSameType& same, StageUnaryType& unary, StageRowMaskType& rowMask, StageJType& j) {
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            // WholeReduceMin and the fixed-pattern GatherMask preserve the mask.
            Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
        } else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
        // Only live rows have valid reduction indices; preserve their bit mask
        // throughout offset construction without switching to counter mode.
        SetVectorMask<int32_t,MaskMode::NORMAL>(rowMask[1],rowMask[0]);
        Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Gather(tileHi_,PrefixW<HALF_RESOURCE>(errorPlane),offset_.template ReinterpretCast<uint32_t>(),0U,rowMask,uint8_t(1),uint16_t(8));

        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Muls<float,false>(tileHi_,tileHi_,intervalFactor,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        auto chunkMinimum=tileLo_;
        auto chunkUpper=tileHi_;
        // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t upperConstantBase=14*(CAPACITY+32)+pointPitch-16;
        const LocalTensor<float> upperScale(TPosition::VECCALC,upperConstantBase*4,8);
        FusedMulAdd<float,false>(tileHi_,upperScale,tileLo_,uint64_t(64),1,BinaryRepeatParams{1,0,1,8,0,8});
        PipeBarrier<PIPE_V>();
        Brcb(PrefixW<HALF_RESOURCE>(1),chunkUpper,(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Compare<float,uint8_t,false>(mask_,PrefixW<HALF_RESOURCE>(lowerPlane),PrefixW<HALF_RESOURCE>(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
        const uint32_t group=j/256,slot=(j/64)%4;
        auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)
        Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
        else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
        GatherMask(PrefixW<HALF_RESOURCE>(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
        GatherMask(PrefixW<HALF_RESOURCE>(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));

        PipeBarrier<PIPE_V>();
        SearchPrefixStateMergePrefixMinimumStage<HALF_RESOURCE>(certHigh_, cert_, hi_, idx_, lo_, record_, second_, tileIdx_, same, unary, j, chunkMinimum, chunkUpper);

    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
#else
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, class StageCertHigh_Type, class StageCert_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageRecord_Type, class StageSecond_Type, class StageTileIdx_Type, class StageSameType, class StageUnaryType, class StageJType, class StageChunkMinimumType, class StageChunkUpperType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixStateMergePrefixMinimumStage(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageRecord_Type& record_, StageSecond_Type& second_, StageTileIdx_Type& tileIdx_, StageSameType& same, StageUnaryType& unary, StageJType& j, StageChunkMinimumType& chunkMinimum, StageChunkUpperType& chunkUpper) {
        if constexpr(COMPACT_TAIL) {MergePrefixMinimum<HALF_RESOURCE,true>(j);} else {

            Compare<float,false>(chunkMinimum,lo_,CMPMODE::LT,uint64_t(64),same);
            Max<float,false>(record_,chunkMinimum,lo_,uint64_t(64),1,same);
            Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
            Min<float,false>(lo_,lo_,chunkMinimum,uint64_t(64),1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,chunkUpper,hi_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,PrefixW<HALF_RESOURCE>(0),cert_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,PrefixW<HALF_RESOURCE>(1),certHigh_,1,same);

            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
        }
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixOutlined(uint32_t liveRows_,uint32_t segmentCount) {
        AscendC::ICachePreLoad(int64_t(2));
        SearchPrefixState<false>(liveRows_,segmentCount);
        return liveRows_*64;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchHalfPrefixOutlined(uint32_t liveRows,uint32_t segmentCount) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        if(liveRows==64U)SearchPrefixState<true,true>(64U,segmentCount);
        else SearchPrefixState<true>(liveRows,segmentCount);
#else
        SearchPrefixState<true>(liveRows,segmentCount);
#endif
        return liveRows*64;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::Search() {
        constexpr bool fixedSearchMask=FULL_FIXED_MASK || NATIVE_FLOAT_PREFIX_SPARSE;
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            if constexpr(COEFFICIENTS==192 && SINGLE_TAIL) {
                if(liveRows_==16U)SearchStaticPolicy<true,16>();
                else SearchStaticPolicy<true>();
            } else SearchStaticPolicy<true>();
        } else {
        if constexpr((sizeof(T)==4 && (CAPACITY==1024 || CAPACITY==256)) || fixedSearchMask) {
            InitializeGroupedSearchState(lo_,idx_,DEFER_CHUNK_CERTIFICATE?1:3);
        } else {
            SharedSearchCoefficients21();
        }
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)Duplicate(refined_[520],-2.0F,8);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        count_=(liveRows_*64);
        if constexpr(CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        SearchSavePrefixTailPredicateStage(same, segment, pointSegment, unary);
        // Refinement gathers a full register, including inactive point lanes.
        if(liveRows_<64) {
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),~((uint64_t(1)<<liveRows_)-1));
            Duplicate<int32_t,false>(idx_,int32_t(0),uint64_t(64),1,1,8);
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
        PipeBarrier<PIPE_V>();
        SearchArenaStage<fixedSearchMask>();
        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        SearchDuplicateStage<fixedSearchMask>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchSavePrefixTailPredicateStage(StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary) {
        for(uint32_t j=0;j<t_.segments;j+=64) {
        if constexpr(COMPACT_TAIL) {
            if(t_.segments-j<=8) {
                if constexpr(SAVE_PREFIX_CHUNK_PREDICATES)SavePrefixTailPredicate(j,t_.segments-j);
                CompactSearchTail<false>(j,t_.segments-j);MergeSearchMinimum<false>(j);continue;
            }
        }
        if constexpr(!CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(SearchW<false>(axis),SearchP<false>(axis),SearchC<false>(axis)[j],uint64_t(64),liveRows_,pointSegment);
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        Mul<float,false>(SearchW<false>(4),SearchW<false>(0),SearchW<false>(0),uint64_t(64),liveRows_,same);
        Mul<float,false>(SearchW<false>(3),SearchW<false>(0),SearchC<false>(6)[j],uint64_t(64),liveRows_,segment);
        } else {
        Mul<float,false>(SearchW<false>(3),SearchW<false>(0),SearchC<false>(6)[j],uint64_t(64),liveRows_,segment);
        Mul<float,false>(SearchW<false>(4),SearchW<false>(0),SearchW<false>(0),uint64_t(64),liveRows_,same);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        MulAddDst<float,float,false>(SearchW<false>(4),SearchW<false>(1),SearchW<false>(1),uint64_t(64),liveRows_,same);
        MulAddDst<float,float,false>(SearchW<false>(3),SearchW<false>(1),SearchC<false>(7)[j],uint64_t(64),liveRows_,segment);
        } else {
        MulAddDst<float,float,false>(SearchW<false>(3),SearchW<false>(1),SearchC<false>(7)[j],uint64_t(64),liveRows_,segment);
        MulAddDst<float,float,false>(SearchW<false>(4),SearchW<false>(1),SearchW<false>(1),uint64_t(64),liveRows_,same);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        MulAddDst<float,float,false>(SearchW<false>(4),SearchW<false>(2),SearchW<false>(2),uint64_t(64),liveRows_,same);
        MulAddDst<float,float,false>(SearchW<false>(3),SearchW<false>(2),SearchC<false>(8)[j],uint64_t(64),liveRows_,segment);
        } else {
        MulAddDst<float,float,false>(SearchW<false>(3),SearchW<false>(2),SearchC<false>(8)[j],uint64_t(64),liveRows_,segment);
        MulAddDst<float,float,false>(SearchW<false>(4),SearchW<false>(2),SearchW<false>(2),uint64_t(64),liveRows_,same);
        }
        PipeBarrier<PIPE_V>();
        SearchMulsStage(same, segment, unary, j);
        SearchSubStage(same, unary, j);}
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchSubStage(StageSameType& same, StageUnaryType& unary, StageJType& j) {
        constexpr bool fixedSearchMask=FULL_FIXED_MASK || NATIVE_FLOAT_PREFIX_SPARSE;

        Sub<float,false>(SearchW<false>(0),SearchW<false>(4),SearchW<false>(5),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
        if(j+64>t_.segments) {
            CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(t_.segments-j)),CMPMODE::LT,uint64_t(64),liveRows_,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
            Select(SearchW<false>(0),valid_,SearchW<false>(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(liveRows_*64));PipeBarrier<PIPE_V>();
        }
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            WholeReduceMin<float,false>(record_,SearchW<false>(0),uint64_t(64),liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        } else WholeReduceMin(record_,SearchW<false>(0),64,liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        PipeBarrier<PIPE_V>();
        GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
        GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
        } else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
        SearchSetMaskNormStage2<fixedSearchMask>(same, unary);
        PipeBarrier<PIPE_V>();
        auto chunkMinimum=(CLEAN_PRIMARY64 && sizeof(T)==2)?chunkLower_[j]:tileLo_;
        auto chunkUpper=(CLEAN_PRIMARY64 && sizeof(T)==2)?tileLo_:tileHi_;
        // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
        SearchSetMaskNormStage<fixedSearchMask>(same, unary);
        Brcb(SearchW<false>(1),chunkUpper,(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Compare<float,uint8_t,false>(mask_,SearchW<false>(0),SearchW<false>(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
        if constexpr(SAVE_PREFIX_CHUNK_PREDICATES) {
            const uint32_t group=j/256,slot=(j/64)%4;
            auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)
                Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
            else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
            PipeBarrier<PIPE_V>();
        }
        SearchCopyStage<fixedSearchMask>(j);
        PipeBarrier<PIPE_V>();
        SearchMergeSearchMinimumStage<fixedSearchMask>(same, unary, j, chunkMinimum, chunkUpper);

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto fixedSearchMask, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchCopyStage(StageJType& j) {
        if constexpr(DEFER_CHUNK_CERTIFICATE) {
            const uint32_t group=j/256,slot=(j/64)%4;
            auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
            if constexpr(fixedSearchMask)
                Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
            else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
        } else {
            GatherMask(SearchW<false>(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(SearchW<false>(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto fixedSearchMask, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchSetMaskNormStage2(StageSameType& same, StageUnaryType& unary) {
        if constexpr(fixedSearchMask) {
            if(liveRows_==64) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Gather(tileHi_,SearchW<false>(5),offset_.template ReinterpretCast<uint32_t>(),0U,64);
            } else {
                Muls(offset_,tileIdx_,int32_t(4),liveRows_);PipeBarrier<PIPE_V>();
                Add(offset_,offset_,rowBase_,liveRows_);PipeBarrier<PIPE_V>();
                Gather(tileHi_,SearchW<false>(5),offset_.template ReinterpretCast<uint32_t>(),0U,liveRows_);
            }
        } else {
            Muls(offset_,tileIdx_,int32_t(4),liveRows_);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,rowBase_,liveRows_);PipeBarrier<PIPE_V>();
            Gather(tileHi_,SearchW<false>(5),offset_.template ReinterpretCast<uint32_t>(),0U,liveRows_);
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto fixedSearchMask, class StageSameType, class StageUnaryType, class StageJType, class StageChunkMinimumType, class StageChunkUpperType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchMergeSearchMinimumStage(StageSameType& same, StageUnaryType& unary, StageJType& j, StageChunkMinimumType& chunkMinimum, StageChunkUpperType& chunkUpper) {
        if constexpr(COMPACT_TAIL) {MergeSearchMinimum<false>(j);} else {
        if constexpr(!(fixedSearchMask)) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        Compare<float,false>(chunkMinimum,lo_,CMPMODE::LT,uint64_t(64),same);
        Max<float,false>(record_,chunkMinimum,lo_,uint64_t(64),1,same);
        Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
        Min<float,false>(lo_,lo_,chunkMinimum,uint64_t(64),1,same);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,chunkUpper,hi_,1,same);
        if constexpr(!DEFER_CHUNK_CERTIFICATE) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,SearchW<false>(0),cert_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,SearchW<false>(1),certHigh_,1,same);
        }
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSameType, class StageSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchMulsStage(StageSameType& same, StageSegmentType& segment, StageUnaryType& unary, StageJType& j) {
        if constexpr(CLEAN_PRIMARY64) {
            // APz is dead here; keep the clamp in a bank phase distinct from dot.
            if constexpr(sizeof(T)==2)
                Muls<float,false>(SearchW<false>(5),SearchW<false>(4),1.52587890625e-5F,uint64_t(64),liveRows_,unary);
            Maxs<float,false>(SearchW<false>(2),SearchW<false>(3),0.0F,uint64_t(64),liveRows_,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(SearchW<false>(2),SearchW<false>(2),SearchC<false>(9)[j],uint64_t(64),liveRows_,segment);PipeBarrier<PIPE_V>();
            const BinaryRepeatParams fused{1,0,1,8,0,8};
            if constexpr(sizeof(T)==4)
                FusedMulAdd<float,false>(SearchW<false>(3),refined_[520],SearchW<false>(2),uint64_t(64),liveRows_,fused);
            else {
                constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
                const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                FusedMulAdd<float,false>(SearchW<false>(3),minusTwo,SearchW<false>(2),uint64_t(64),liveRows_,fused);
            }
            if constexpr(sizeof(T)==4)
                Muls<float,false>(SearchW<false>(5),SearchW<false>(4),1.52587890625e-5F,uint64_t(64),liveRows_,unary);
            PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(SearchW<false>(4),SearchW<false>(2),SearchW<false>(3),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
        } else {
        Maxs<float,false>(SearchW<false>(0),SearchW<false>(3),0.0F,uint64_t(64),liveRows_,unary);PipeBarrier<PIPE_V>();
        Min<float,false>(SearchW<false>(0),SearchW<false>(0),SearchC<false>(9)[j],uint64_t(64),liveRows_,segment);PipeBarrier<PIPE_V>();
        Muls<float,false>(SearchW<false>(2),SearchW<false>(3),-2.0F,uint64_t(64),liveRows_,unary);
        Muls<float,false>(SearchW<false>(5),SearchW<false>(4),1.52587890625e-5F,uint64_t(64),liveRows_,unary);PipeBarrier<PIPE_V>();
        Add<float,false>(SearchW<false>(1),SearchW<false>(0),SearchW<false>(2),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(SearchW<false>(4),SearchW<false>(0),SearchW<false>(1),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
        }
    }
