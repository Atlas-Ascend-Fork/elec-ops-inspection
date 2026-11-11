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

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageGuardedRowsType, class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfCachedRowsCompareScalarStage(StageGuardedRowsType& guardedRows, StageSameType& same, StageUnaryType& unary, StageJType& j) {
        constexpr bool DEFER_SECOND=CLEAN_PRIMARY64 && sizeof(T)==2 && ROWS==64;

        PipeBarrier<PIPE_V>();
        if(j+64>t_.segments) {
            CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(t_.segments-j)),CMPMODE::LT,uint64_t(64),guardedRows,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
            Select(W(0),valid_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(guardedRows*64));PipeBarrier<PIPE_V>();
        }
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4) {
            if(j+64>t_.segments) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            WholeReduceMin<float,false>(record_,W(0),uint64_t(64),guardedRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        } else WholeReduceMin(record_,W(0),64,guardedRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        PipeBarrier<PIPE_V>();
        GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
        GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)
            Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
        else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
        SearchGuardedHalfCachedRowsSetMaskNormStage2(guardedRows, same, unary);
        PipeBarrier<PIPE_V>();
        auto chunkMinimum=(CLEAN_PRIMARY64 && sizeof(T)==2)?chunkLower_[j]:tileLo_;
        auto chunkUpper=(CLEAN_PRIMARY64 && sizeof(T)==2)?tileLo_:tileHi_;
        // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
        SearchGuardedHalfCachedRowsSetMaskNormStage(same, unary);
        Brcb(W(1),chunkUpper,(guardedRows+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        SearchGuardedHalfCachedRowsCompareStage(guardedRows, j);
        PipeBarrier<PIPE_V>();
        SearchGuardedHalfCachedRowsMergeSearchMinimumStage<DEFER_SECOND>(same, unary, j, chunkMinimum, chunkUpper);

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageGuardedRowsType, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfCachedRowsSetMaskNormStage2(StageGuardedRowsType& guardedRows, StageSameType& same, StageUnaryType& unary) {
        if constexpr(FULL_FIXED_MASK) {
            if(guardedRows==64) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,64);
            } else {
                Muls(offset_,tileIdx_,int32_t(4),guardedRows);PipeBarrier<PIPE_V>();
                Add(offset_,offset_,rowBase_,guardedRows);PipeBarrier<PIPE_V>();
                Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,guardedRows);
            }
        } else {
            Muls(offset_,tileIdx_,int32_t(4),guardedRows);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,rowBase_,guardedRows);PipeBarrier<PIPE_V>();
            Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,guardedRows);
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto DEFER_SECOND, class StageSameType, class StageUnaryType, class StageJType, class StageChunkMinimumType, class StageChunkUpperType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfCachedRowsMergeSearchMinimumStage(StageSameType& same, StageUnaryType& unary, StageJType& j, StageChunkMinimumType& chunkMinimum, StageChunkUpperType& chunkUpper) {
        if constexpr(COMPACT_TAIL) {MergeSearchMinimum<false>(j);} else {
        MergeGuardedChunkMinimum<FULL_FIXED_MASK,DEFER_SECOND>(lo_,hi_,second_,record_,tileIdx_,chunkMinimum,chunkUpper,j,same,unary);
        if constexpr(!DEFER_CHUNK_CERTIFICATE) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,W(0),cert_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,W(1),certHigh_,1,same);
        }
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfCachedRowsSetMaskNormStage(StageSameType& same, StageUnaryType& unary) {
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Axpy<float,float,false>(tileLo_,tileHi_,2.125F,uint64_t(64),1,unary);
            PipeBarrier<PIPE_V>();
        } else if constexpr(CLEAN_PRIMARY64) {
            SharedPrefetchFence17();
        } else if constexpr(FULL_FIXED_MASK) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Muls<float,false>(tileHi_,tileHi_,2.125F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Add<float,false>(tileHi_,tileLo_,tileHi_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        } else {
            Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
            Add(tileHi_,tileLo_,tileHi_,64);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageGuardedRowsType, class StageCoarseErrorFactorType, class StageSameType, class StageSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfCachedRowsMulsStage(StageGuardedRowsType& guardedRows, StageCoarseErrorFactorType& coarseErrorFactor, StageSameType& same, StageSegmentType& segment, StageUnaryType& unary, StageJType& j) {
        if constexpr(CLEAN_PRIMARY64) {
            // APz is dead here; keep the clamp in a bank phase distinct from dot.
            if constexpr(sizeof(T)==2)
                Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),guardedRows,unary);
            Maxs<float,false>(W(2),W(3),0.0F,uint64_t(64),guardedRows,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(W(2),W(2),C(9)[j],uint64_t(64),guardedRows,segment);PipeBarrier<PIPE_V>();
            const BinaryRepeatParams fused{1,0,1,8,0,8};
            if constexpr(sizeof(T)==4)
                FusedMulAdd<float,false>(W(3),refined_[520],W(2),uint64_t(64),guardedRows,fused);
            else {
                constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
                const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                FusedMulAdd<float,false>(W(3),minusTwo,W(2),uint64_t(64),guardedRows,fused);
            }
            if constexpr(sizeof(T)==4)
                Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),guardedRows,unary);
            PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(W(4),W(2),W(3),uint64_t(64),guardedRows,same);PipeBarrier<PIPE_V>();
        } else {
        Maxs<float,false>(W(0),W(3),0.0F,uint64_t(64),guardedRows,unary);PipeBarrier<PIPE_V>();
        Min<float,false>(W(0),W(0),C(9)[j],uint64_t(64),guardedRows,segment);PipeBarrier<PIPE_V>();
        Muls<float,false>(W(2),W(3),-2.0F,uint64_t(64),guardedRows,unary);
        Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),guardedRows,unary);PipeBarrier<PIPE_V>();
        Add<float,false>(W(1),W(0),W(2),uint64_t(64),guardedRows,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(W(4),W(0),W(1),uint64_t(64),guardedRows,same);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageGuardedRowsType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfCachedRowsCompareStage(StageGuardedRowsType& guardedRows, StageJType& j) {
        if constexpr(CLEAN_PRIMARY64) {
            const uint32_t group=j/256,slot=(j/64)%4;
            auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
            if(guardedRows==64) {
                Compare<float,uint8_t,false>(predicates[slot*128].template ReinterpretCast<uint8_t>(),W(0),W(1),CMPMODE::LE,uint64_t(64),guardedRows,BinaryRepeatParams(1,1,0,8,8,1));
            } else {
                Compare<float,uint8_t,false>(mask_,W(0),W(1),CMPMODE::LE,uint64_t(64),guardedRows,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
            }
        } else {
            Compare<float,uint8_t,false>(mask_,W(0),W(1),CMPMODE::LE,uint64_t(64),guardedRows,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
            SharedGuardedMinimum13(j);
        }
    }

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::HalfFullC(uint32_t k) {
        constexpr uint32_t pitch=1056U;
        const uint32_t offset=k>=6U && k<=8U?6U*pitch+(k-6U)*1024U:k*pitch;
        return LocalTensor<float>(TPosition::VECCALC,offset*4U,14U*pitch-offset);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::HalfFullW(uint32_t k) {
        constexpr uint32_t banks[13]={32,0,0,64,96,32,96,0,0,0,0,0,0};
        const uint32_t offset=k*4224U+banks[k];
        return LocalTensor<float>(TPosition::VECCALC,71808U+offset*4U,6U*4224U-offset);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::HalfFullP(uint32_t k) {
        return LocalTensor<float>(TPosition::VECCALC,59136U+k*4224U,3U*1056U-k*1056U);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfTypeCertifiedRows(uint32_t liveRows_,uint32_t segmentCount,bool tightCoefficientSafe_) {
        static_assert(sizeof(T)==2 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Full native Half resource required");
        constexpr uint32_t pp_=1056U;
        const uint32_t guardedRows=liveRows_;
        LocalTensor<float> certHigh_(TPosition::VECCALC,184640U,64U);
        LocalTensor<float> cert_(TPosition::VECCALC,184384U,64U);
        LocalTensor<float> chunkLower_(TPosition::VECCALC,174592U,1024U);
        LocalTensor<float> hi_(TPosition::VECCALC,183872U,64U);
        LocalTensor<int32_t> idx_(TPosition::VECCALC,184128U,64U);
        LocalTensor<float> lo_(TPosition::VECCALC,183360U,64U);
        LocalTensor<uint8_t> mask_(TPosition::VECCALC,187328U,512U);
        LocalTensor<int32_t> offset_(TPosition::VECCALC,186176U,64U);
        LocalTensor<int32_t> oneHot_(TPosition::VECCALC,184896U,64U);
        LocalTensor<int32_t> outputOffset_(TPosition::VECCALC,186944U,64U);
        LocalTensor<float> record_(TPosition::VECCALC,182784U,128U);
        LocalTensor<float> refined_(TPosition::VECCALC,179456U,832U);
        LocalTensor<int32_t> rowBase_(TPosition::VECCALC,186432U,64U);
        LocalTensor<float> second_(TPosition::VECCALC,183616U,64U);
        LocalTensor<float> segmentIndices_(TPosition::VECCALC,173184U,64U);
        LocalTensor<uint8_t> smallMask_(TPosition::VECCALC,188352U,32U);
        LocalTensor<float> tileHi_(TPosition::VECCALC,185664U,64U);
        LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,185920U,64U);
        LocalTensor<float> tileLo_(TPosition::VECCALC,185152U,64U);
        LocalTensor<uint8_t> valid_(TPosition::VECCALC,187840U,512U);
        LocalTensor<float> xyz_(TPosition::VECCALC,59136U,3168U);
        constexpr bool DEFER_SECOND=CLEAN_PRIMARY64 && sizeof(T)==2 && ROWS==64;
        const float coarseErrorFactor=tightCoefficientSafe_?
            7.62939453125e-6F:1.52587890625e-5F;
        const float intervalFactor=coarseErrorFactor*1.0625F;
        const float lowerNormFactor=1.0F-intervalFactor;
        const float upperNormFactor=intervalFactor*2.125F*1.000003814697265625F;
        return SearchGuardedHalfTypeCertifiedRowsInitializeGroupedSearchStateStage(segmentCount, guardedRows, certHigh_, cert_, chunkLower_, hi_, idx_, lo_, mask_, offset_, oneHot_, outputOffset_, record_, refined_, rowBase_, second_, segmentIndices_, smallMask_, tileHi_, tileIdx_, tileLo_, valid_, xyz_, coarseErrorFactor, lowerNormFactor, upperNormFactor);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSegmentCountType, class StageGuardedRowsType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageOneHot_Type, class StageOutputOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageSegmentIndices_Type, class StageSmallMask_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageValid_Type, class StageXyz_Type, class StageCoarseErrorFactorType, class StageLowerNormFactorType, class StageUpperNormFactorType>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfTypeCertifiedRowsInitializeGroupedSearchStateStage(StageSegmentCountType& segmentCount, StageGuardedRowsType& guardedRows, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageOutputOffset_Type& outputOffset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageSegmentIndices_Type& segmentIndices_, StageSmallMask_Type& smallMask_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageValid_Type& valid_, StageXyz_Type& xyz_, StageCoarseErrorFactorType& coarseErrorFactor, StageLowerNormFactorType& lowerNormFactor, StageUpperNormFactorType& upperNormFactor) {
        constexpr bool DEFER_SECOND=CLEAN_PRIMARY64 && sizeof(T)==2 && ROWS==64;

        if constexpr(UsesCapacityStage3()) {
            InitializeGroupedSearchState(lo_,idx_,DEFER_CHUNK_CERTIFICATE?1:3);
        } else {
            Duplicate(lo_,__builtin_inff(),64);Duplicate(second_,__builtin_inff(),64);
            Duplicate(hi_,__builtin_inff(),64);Duplicate(idx_,int32_t(0),64);
            if constexpr(!DEFER_CHUNK_CERTIFICATE){Duplicate(cert_,0.0F,64);Duplicate(certHigh_,0.0F,64);}
        }
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)Duplicate(refined_[520],-2.0F,8);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        const uint32_t count_=(guardedRows*64);
        if constexpr(CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage5(segmentCount, guardedRows, certHigh_, cert_, chunkLower_, hi_, idx_, lo_, mask_, offset_, record_, refined_, rowBase_, second_, segmentIndices_, tileHi_, tileIdx_, tileLo_, valid_, xyz_, coarseErrorFactor, lowerNormFactor, upperNormFactor, same, segment, pointSegment, unary);
        SearchGuardedHalfTypeCertifiedRowsDuplicateStage2<DEFER_SECOND>(segmentCount, chunkLower_, hi_, mask_, second_, same);
        // Refinement gathers a full register, including inactive point lanes.
        if(guardedRows<64) {
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),~((uint64_t(1)<<guardedRows)-1));
            Duplicate<int32_t,false>(idx_,int32_t(0),uint64_t(64),1,1,8);
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
        PipeBarrier<PIPE_V>();
        SearchGuardedHalfTypeCertifiedRowsArenaStage(certHigh_, cert_, idx_, offset_, oneHot_, outputOffset_);
        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        SearchGuardedHalfTypeCertifiedRowsDuplicateStage(certHigh_, cert_, idx_, offset_, oneHot_, smallMask_, valid_);
        return count_;

    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSegmentCountType, class StageGuardedRowsType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageSegmentIndices_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageValid_Type, class StageXyz_Type, class StageCoarseErrorFactorType, class StageLowerNormFactorType, class StageUpperNormFactorType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage5(StageSegmentCountType& segmentCount, StageGuardedRowsType& guardedRows, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageSegmentIndices_Type& segmentIndices_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageValid_Type& valid_, StageXyz_Type& xyz_, StageCoarseErrorFactorType& coarseErrorFactor, StageLowerNormFactorType& lowerNormFactor, StageUpperNormFactorType& upperNormFactor, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary) {
        for(uint32_t j=0;j<segmentCount;j+=64) {
        if constexpr(!CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(HalfFullW(axis),HalfFullP(axis),HalfFullC(axis)[j],uint64_t(64),guardedRows,pointSegment);
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        Mul<float,false>(HalfFullW(4),HalfFullW(0),HalfFullW(0),uint64_t(64),guardedRows,same);
        Mul<float,false>(HalfFullW(3),HalfFullW(0),HalfFullC(6)[j],uint64_t(64),guardedRows,segment);
        } else {
        Mul<float,false>(HalfFullW(3),HalfFullW(0),HalfFullC(6)[j],uint64_t(64),guardedRows,segment);
        Mul<float,false>(HalfFullW(4),HalfFullW(0),HalfFullW(0),uint64_t(64),guardedRows,same);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        MulAddDst<float,float,false>(HalfFullW(4),HalfFullW(1),HalfFullW(1),uint64_t(64),guardedRows,same);
        MulAddDst<float,float,false>(HalfFullW(3),HalfFullW(1),HalfFullC(7)[j],uint64_t(64),guardedRows,segment);
        } else {
        MulAddDst<float,float,false>(HalfFullW(3),HalfFullW(1),HalfFullC(7)[j],uint64_t(64),guardedRows,segment);
        MulAddDst<float,float,false>(HalfFullW(4),HalfFullW(1),HalfFullW(1),uint64_t(64),guardedRows,same);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        MulAddDst<float,float,false>(HalfFullW(4),HalfFullW(2),HalfFullW(2),uint64_t(64),guardedRows,same);
        MulAddDst<float,float,false>(HalfFullW(3),HalfFullW(2),HalfFullC(8)[j],uint64_t(64),guardedRows,segment);
        } else {
        MulAddDst<float,float,false>(HalfFullW(3),HalfFullW(2),HalfFullC(8)[j],uint64_t(64),guardedRows,segment);
        MulAddDst<float,float,false>(HalfFullW(4),HalfFullW(2),HalfFullW(2),uint64_t(64),guardedRows,same);
        }
        PipeBarrier<PIPE_V>();
        SearchGuardedHalfTypeCertifiedRowsMulsStage(guardedRows, refined_, coarseErrorFactor, lowerNormFactor, same, segment, unary, j);
        // W5 already holds an outward-biased lower score; normW4 stays live.
        if(j+64>segmentCount) {
            CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(segmentCount-j)),CMPMODE::LT,uint64_t(64),guardedRows,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
            Select(HalfFullW(5),valid_,HalfFullW(5),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(guardedRows*64));PipeBarrier<PIPE_V>();
        }
        SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage4(segmentCount, guardedRows, certHigh_, cert_, chunkLower_, hi_, idx_, lo_, mask_, offset_, record_, refined_, rowBase_, second_, tileHi_, tileIdx_, tileLo_, xyz_, upperNormFactor, same, unary, j);}
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSegmentCountType, class StageGuardedRowsType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageXyz_Type, class StageUpperNormFactorType, class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage4(StageSegmentCountType& segmentCount, StageGuardedRowsType& guardedRows, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageXyz_Type& xyz_, StageUpperNormFactorType& upperNormFactor, StageSameType& same, StageUnaryType& unary, StageJType& j) {
        constexpr bool DEFER_SECOND=CLEAN_PRIMARY64 && sizeof(T)==2 && ROWS==64;
        constexpr uint32_t pp_=1056U;

        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4) {
            if(j+64>segmentCount) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            WholeReduceMin<float,false>(record_,HalfFullW(5),uint64_t(64),guardedRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        } else WholeReduceMin(record_,HalfFullW(5),64,guardedRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        PipeBarrier<PIPE_V>();
        GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
        GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)
            Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
        else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
        SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage3(guardedRows, offset_, rowBase_, tileHi_, tileIdx_, same, unary);
        PipeBarrier<PIPE_V>();
        auto chunkMinimum=(CLEAN_PRIMARY64 && sizeof(T)==2)?chunkLower_[j]:tileLo_;
        auto chunkUpper=(CLEAN_PRIMARY64 && sizeof(T)==2)?tileLo_:tileHi_;
        // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
        SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage(tileHi_, tileLo_, upperNormFactor, same, unary);
        Brcb(HalfFullW(1),chunkUpper,(guardedRows+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        SearchGuardedHalfTypeCertifiedRowsCompareStage<pp_>(guardedRows, mask_, refined_, xyz_, j);
        PipeBarrier<PIPE_V>();
          SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage2<DEFER_SECOND>(certHigh_, cert_, hi_, idx_, lo_, record_, second_, tileIdx_, same, unary, j, chunkMinimum, chunkUpper);

    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageGuardedRowsType, class StageOffset_Type, class StageRowBase_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage3(StageGuardedRowsType& guardedRows, StageOffset_Type& offset_, StageRowBase_Type& rowBase_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageSameType& same, StageUnaryType& unary) {
        if constexpr(FULL_FIXED_MASK) {
            if(guardedRows==64) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Gather(tileHi_,HalfFullW(4),offset_.template ReinterpretCast<uint32_t>(),0U,64);
            } else {
                Muls(offset_,tileIdx_,int32_t(4),guardedRows);PipeBarrier<PIPE_V>();
                Add(offset_,offset_,rowBase_,guardedRows);PipeBarrier<PIPE_V>();
                Gather(tileHi_,HalfFullW(4),offset_.template ReinterpretCast<uint32_t>(),0U,guardedRows);
            }
        } else {
            Muls(offset_,tileIdx_,int32_t(4),guardedRows);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,rowBase_,guardedRows);PipeBarrier<PIPE_V>();
            Gather(tileHi_,HalfFullW(4),offset_.template ReinterpretCast<uint32_t>(),0U,guardedRows);
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto DEFER_SECOND, class StageCertHigh_Type, class StageCert_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageRecord_Type, class StageSecond_Type, class StageTileIdx_Type, class StageSameType, class StageUnaryType, class StageJType, class StageChunkMinimumType, class StageChunkUpperType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage2(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageRecord_Type& record_, StageSecond_Type& second_, StageTileIdx_Type& tileIdx_, StageSameType& same, StageUnaryType& unary, StageJType& j, StageChunkMinimumType& chunkMinimum, StageChunkUpperType& chunkUpper) {
        {
        MergeGuardedChunkMinimum<FULL_FIXED_MASK,DEFER_SECOND>(lo_,hi_,second_,record_,tileIdx_,chunkMinimum,chunkUpper,j,same,unary);
        if constexpr(!DEFER_CHUNK_CERTIFICATE) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,HalfFullW(0),cert_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,HalfFullW(1),certHigh_,1,same);
        }
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageTileHi_Type, class StageTileLo_Type, class StageUpperNormFactorType, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage(StageTileHi_Type& tileHi_, StageTileLo_Type& tileLo_, StageUpperNormFactorType& upperNormFactor, StageSameType& same, StageUnaryType& unary) {
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Axpy<float,float,false>(tileLo_,tileHi_,upperNormFactor,uint64_t(64),1,unary);
            PipeBarrier<PIPE_V>();
        } else if constexpr(CLEAN_PRIMARY64) {
            constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
            constexpr uint32_t constantBase=14*(CAPACITY+32)+pointPitch-16;
            const LocalTensor<float> upperScale(TPosition::VECCALC,constantBase*4,8);
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            FusedMulAdd<float,false>(tileHi_,upperScale,tileLo_,uint64_t(64),1,BinaryRepeatParams{1,0,1,8,0,8});
            PipeBarrier<PIPE_V>();
        } else if constexpr(FULL_FIXED_MASK) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Muls<float,false>(tileHi_,tileHi_,2.125F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Add<float,false>(tileHi_,tileLo_,tileHi_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        } else {
            Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
            Add(tileHi_,tileLo_,tileHi_,64);PipeBarrier<PIPE_V>();
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageGuardedRowsType, class StageRefined_Type, class StageCoarseErrorFactorType, class StageLowerNormFactorType, class StageSameType, class StageSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfTypeCertifiedRowsMulsStage(StageGuardedRowsType& guardedRows, StageRefined_Type& refined_, StageCoarseErrorFactorType& coarseErrorFactor, StageLowerNormFactorType& lowerNormFactor, StageSameType& same, StageSegmentType& segment, StageUnaryType& unary, StageJType& j) {
        if constexpr(CLEAN_PRIMARY64) {
            // APz is dead here; keep the clamp in a bank phase distinct from dot.
            if constexpr(sizeof(T)==2)
                Muls<float,false>(HalfFullW(5),HalfFullW(4),lowerNormFactor,uint64_t(64),guardedRows,unary);
            Maxs<float,false>(HalfFullW(2),HalfFullW(3),0.0F,uint64_t(64),guardedRows,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(HalfFullW(2),HalfFullW(2),HalfFullC(9)[j],uint64_t(64),guardedRows,segment);PipeBarrier<PIPE_V>();
            const BinaryRepeatParams fused{1,0,1,8,0,8};
            if constexpr(sizeof(T)==4)
                FusedMulAdd<float,false>(HalfFullW(3),refined_[520],HalfFullW(2),uint64_t(64),guardedRows,fused);
            else {
                constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
                const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                FusedMulAdd<float,false>(HalfFullW(3),minusTwo,HalfFullW(2),uint64_t(64),guardedRows,fused);
            }
            if constexpr(sizeof(T)==4)
                Muls<float,false>(HalfFullW(5),HalfFullW(4),coarseErrorFactor,uint64_t(64),guardedRows,unary);
            PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(HalfFullW(5),HalfFullW(2),HalfFullW(3),uint64_t(64),guardedRows,same);PipeBarrier<PIPE_V>();
        } else {
        Maxs<float,false>(HalfFullW(0),HalfFullW(3),0.0F,uint64_t(64),guardedRows,unary);PipeBarrier<PIPE_V>();
        Min<float,false>(HalfFullW(0),HalfFullW(0),HalfFullC(9)[j],uint64_t(64),guardedRows,segment);PipeBarrier<PIPE_V>();
        Muls<float,false>(HalfFullW(2),HalfFullW(3),-2.0F,uint64_t(64),guardedRows,unary);
        Muls<float,false>(HalfFullW(5),HalfFullW(4),coarseErrorFactor,uint64_t(64),guardedRows,unary);PipeBarrier<PIPE_V>();
        Add<float,false>(HalfFullW(1),HalfFullW(0),HalfFullW(2),uint64_t(64),guardedRows,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(HalfFullW(4),HalfFullW(0),HalfFullW(1),uint64_t(64),guardedRows,same);PipeBarrier<PIPE_V>();
        }
    }
#endif
