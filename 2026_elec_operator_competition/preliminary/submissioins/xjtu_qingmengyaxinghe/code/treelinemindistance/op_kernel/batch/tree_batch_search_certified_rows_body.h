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

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool SHARED_BOUND, bool COMPLETE_COEFFICIENTS>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchCertifiedRowsBody(bool corePointSafe) {
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            SearchStaticPolicy<false>();
        } else {
        const uint32_t searchRows=SHARED_BOUND?64U:liveRows_;
        constexpr bool DEFER_SECOND=CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64;
        const float coarseErrorFactor=(tightCoefficientSafe_ &&
            (corePointSafe || CoarseCoordinatesSafe<false>()))?
            7.62939453125e-6F:1.52587890625e-5F;
        constexpr bool sharedBound=SHARED_BOUND;
        auto pointBound=refined_[12*64];
        const BinaryRepeatParams sharedSame{1,1,1,8,8,8};
        const UnaryRepeatParams sharedUnary{1,1,8,8};
        SearchCertifiedRowsBodySetMaskNormStage2<SHARED_BOUND>(coarseErrorFactor, pointBound, sharedSame, sharedUnary);
        auto lowerScores=sharedBound?W(4):W(0);
        SharedQueryDifference18();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        count_=(searchRows*64);
        if constexpr(CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        SearchCertifiedRowsBodyCompactSearchTailStage<SHARED_BOUND, COMPLETE_COEFFICIENTS>(searchRows, coarseErrorFactor, pointBound, sharedSame, lowerScores, same, segment, pointSegment, unary);}
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool SHARED_BOUND, bool COMPLETE_COEFFICIENTS, class StageSearchRowsType, class StageCoarseErrorFactorType, class StagePointBoundType, class StageSharedSameType, class StageLowerScoresType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchCertifiedRowsBodyCompactSearchTailStage(StageSearchRowsType& searchRows, StageCoarseErrorFactorType& coarseErrorFactor, StagePointBoundType& pointBound, StageSharedSameType& sharedSame, StageLowerScoresType& lowerScores, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary) {
        constexpr bool DEFER_SECOND=CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64;

        for(uint32_t j=0;j<(COMPLETE_COEFFICIENTS?COEFFICIENTS:t_.segments);j+=64) {
            if constexpr(COMPACT_TAIL) {
                if((COMPLETE_COEFFICIENTS?COEFFICIENTS:t_.segments)-j<=8) {
                    CompactSearchTail<false>(j,(COMPLETE_COEFFICIENTS?COEFFICIENTS:t_.segments)-j);MergeSearchMinimum<false>(j);continue;
                }
            }
            if constexpr(!CLEAN_PRIMARY64) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            SearchCertifiedRowsBodyDefinedStage<SHARED_BOUND, COMPLETE_COEFFICIENTS>(searchRows, coarseErrorFactor, lowerScores, same, segment, pointSegment, unary, j);
            SearchCertifiedRowsBodyCompactSearchTailStageSetMaskNormStage<COMPLETE_COEFFICIENTS>(searchRows, lowerScores, j);
            PipeBarrier<PIPE_V>();
            GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
            if constexpr(!COMPLETE_COEFFICIENTS && !SHARED_BOUND) {
                if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)
                    Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
                else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            }
            SearchCertifiedRowsBodyCopyStage<SHARED_BOUND>(searchRows, pointBound, sharedSame, same, unary, j);
            PipeBarrier<PIPE_V>();
            auto chunkMinimum=(CLEAN_PRIMARY64 && sizeof(T)==2)?chunkLower_[j]:tileLo_;
            auto chunkUpper=(CLEAN_PRIMARY64 && sizeof(T)==2)?tileLo_:tileHi_;
            // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
            SearchGuardedHalfCachedRowsSetMaskNormStage(same, unary);
            if constexpr(SHARED_BOUND) {
                Add<float,false>(tileSecond_,chunkUpper,pointBound,uint64_t(64),1,sharedSame);PipeBarrier<PIPE_V>();
                Brcb(W(1),tileSecond_,(searchRows+7)/8,BrcbRepeatParams(1,8));
            } else Brcb(W(1),chunkUpper,(searchRows+7)/8,BrcbRepeatParams(1,8));
            PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            SearchCertifiedRowsBodyCompareStage(searchRows, lowerScores, j);
            PipeBarrier<PIPE_V>();
            SearchGuardedHalfCachedRowsMergeSearchMinimumStage<DEFER_SECOND>(same, unary, j, chunkMinimum, chunkUpper);
        }
        SearchCertifiedRowsBodyDuplicateStage<COMPLETE_COEFFICIENTS, DEFER_SECOND>(same, unary);
        FinalizeGuardedCertificate(searchRows);

    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool COMPLETE_COEFFICIENTS, class StageSearchRowsType, class StageLowerScoresType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchCertifiedRowsBodyCompactSearchTailStageSetMaskNormStage(StageSearchRowsType& searchRows, StageLowerScoresType& lowerScores, StageJType& j) {
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4) {
            if constexpr(!COMPLETE_COEFFICIENTS) {
            if(j+64>t_.segments) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            }
            WholeReduceMin<float,false>(record_,lowerScores,uint64_t(64),searchRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        } else WholeReduceMin(record_,lowerScores,64,searchRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool COMPLETE_COEFFICIENTS, auto DEFER_SECOND, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchCertifiedRowsBodyDuplicateStage(StageSameType& same, StageUnaryType& unary) {
        if constexpr(DEFER_SECOND) {
            const uint32_t chunks=((COMPLETE_COEFFICIENTS?COEFFICIENTS:t_.segments)+63)/64;
            CommonVectorStage51(chunks, same, unary);
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool SHARED_BOUND, bool COMPLETE_COEFFICIENTS, class StageSearchRowsType, class StageCoarseErrorFactorType, class StageLowerScoresType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchCertifiedRowsBodyDefinedStage(StageSearchRowsType& searchRows, StageCoarseErrorFactorType& coarseErrorFactor, StageLowerScoresType& lowerScores, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary, StageJType& j) {
        if constexpr(COMPLETE_COEFFICIENTS) {
        if constexpr(UsesRowsStage14()) { SearchCertifiedRowsBodyWStage(searchRows, same, segment, pointSegment, unary, j); } else {
        Sub<float,false>(W(0),P(0),C(0)[j],uint64_t(64),searchRows,pointSegment);
        Sub<float,false>(W(1),P(1),C(1)[j],uint64_t(64),searchRows,pointSegment);

        if constexpr(NORM_BEFORE_DOT) {
        Mul<float,false>(W(4),W(0),W(0),uint64_t(64),searchRows,same);
        Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),searchRows,segment);
        } else {
        Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),searchRows,segment);
        Mul<float,false>(W(4),W(0),W(0),uint64_t(64),searchRows,same);
        }
        Sub<float,false>(W(2),P(2),C(2)[j],uint64_t(64),searchRows,pointSegment);

        if constexpr(NORM_BEFORE_DOT) {
        MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),searchRows,same);
        MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),searchRows,segment);
        } else {
        MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),searchRows,segment);
        MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),searchRows,same);
        }

        if constexpr(NORM_BEFORE_DOT) {
        MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),searchRows,same);
        MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),searchRows,segment);
        } else {
        MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),searchRows,segment);
        MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),searchRows,same);
        }

        SearchCertifiedRowsBodyMulsStage<SHARED_BOUND>(searchRows, coarseErrorFactor, same, segment, unary, j);
        }
        {}

        } else { SearchCertifiedRowsBodySubStage<SHARED_BOUND>(searchRows, coarseErrorFactor, lowerScores, same, segment, pointSegment, unary, j); }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ constexpr bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::UsesRowsStage14() { return (sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && CLEAN_PRIMARY64);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSearchRowsType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchCertifiedRowsBodyWStage(StageSearchRowsType& searchRows, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary, StageJType& j) {
        {
            auto dotZ=W(2);auto projectedClamp=W(3);
            Sub<float,false>(W(0),P(0),C(0)[j],uint64_t(64),searchRows,pointSegment);
            Sub<float,false>(W(1),P(1),C(1)[j],uint64_t(64),searchRows,pointSegment);
            Mul<float,false>(W(4),W(0),W(0),uint64_t(64),searchRows,same);
            Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),searchRows,segment);
            Sub<float,false>(dotZ,P(2),C(2)[j],uint64_t(64),searchRows,pointSegment);
            MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),searchRows,same);
            MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),searchRows,segment);
            MulAddDst<float,float,false>(W(4),dotZ,dotZ,uint64_t(64),searchRows,same);
            FusedMulAdd<float,false>(dotZ,C(8)[j],W(3),uint64_t(64),searchRows,BinaryRepeatParams{1,1,1,8,0,8});
            Maxs<float,false>(projectedClamp,dotZ,0.0F,uint64_t(64),searchRows,unary);
            Min<float,false>(projectedClamp,projectedClamp,C(9)[j],uint64_t(64),searchRows,segment);
            constexpr uint32_t constantBase=4*(CAPACITY+32)+CAPACITY+8;
            const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
            FusedMulAdd<float,false>(dotZ,minusTwo,projectedClamp,uint64_t(64),searchRows,BinaryRepeatParams{1,0,1,8,0,8});
            MulAddDst<float,float,false>(W(4),projectedClamp,dotZ,uint64_t(64),searchRows,same);
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool SHARED_BOUND, class StageCoarseErrorFactorType, class StagePointBoundType, class StageSharedSameType, class StageSharedUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchCertifiedRowsBodySetMaskNormStage2(StageCoarseErrorFactorType& coarseErrorFactor, StagePointBoundType& pointBound, StageSharedSameType& sharedSame, StageSharedUnaryType& sharedUnary) {
        if constexpr(SHARED_BOUND) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            // RN inflation bounds every rounded AP norm by the positive coordinate envelope.
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Abs<float,false>(tileSecond_,compactP_[axis*64],uint64_t(64),1,sharedUnary);
                Brcb(W(0),C(axis)[CAPACITY+16],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Add<float,false>(tileSecond_,tileSecond_,W(0),uint64_t(64),1,BinaryRepeatParams{1,1,0,8,8,0});PipeBarrier<PIPE_V>();
                if(axis==0)Mul<float,false>(pointBound,tileSecond_,tileSecond_,uint64_t(64),1,sharedSame);
                else MulAddDst<float,float,false>(pointBound,tileSecond_,tileSecond_,uint64_t(64),1,sharedSame);
                PipeBarrier<PIPE_V>();
            }
            Muls<float,false>(pointBound,pointBound,coarseErrorFactor*1.000003814697265625F,uint64_t(64),1,sharedUnary);PipeBarrier<PIPE_V>();
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool SHARED_BOUND, class StageSearchRowsType, class StagePointBoundType, class StageSharedSameType, class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchCertifiedRowsBodyCopyStage(StageSearchRowsType& searchRows, StagePointBoundType& pointBound, StageSharedSameType& sharedSame, StageSameType& same, StageUnaryType& unary, StageJType& j) {
        if constexpr(SHARED_BOUND) {
            Copy<float,false>(tileHi_,pointBound,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
            Sub<float,false>(tileLo_,tileLo_,pointBound,uint64_t(64),1,sharedSame);PipeBarrier<PIPE_V>();
            Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
        } else if constexpr(FULL_FIXED_MASK) {
            if(searchRows==64) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,64);
            } else {
                Muls(offset_,tileIdx_,int32_t(4),searchRows);PipeBarrier<PIPE_V>();
                Add(offset_,offset_,rowBase_,searchRows);PipeBarrier<PIPE_V>();
                Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,searchRows);
            }
        } else {
            Muls(offset_,tileIdx_,int32_t(4),searchRows);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,rowBase_,searchRows);PipeBarrier<PIPE_V>();
            Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,searchRows);
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool SHARED_BOUND, class StageSearchRowsType, class StageCoarseErrorFactorType, class StageLowerScoresType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchCertifiedRowsBodySubStage(StageSearchRowsType& searchRows, StageCoarseErrorFactorType& coarseErrorFactor, StageLowerScoresType& lowerScores, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary, StageJType& j) {
        {
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(W(axis),P(axis),C(axis)[j],uint64_t(64),searchRows,pointSegment);
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        Mul<float,false>(W(4),W(0),W(0),uint64_t(64),searchRows,same);
        Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),searchRows,segment);
        } else {
        Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),searchRows,segment);
        Mul<float,false>(W(4),W(0),W(0),uint64_t(64),searchRows,same);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),searchRows,same);
        MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),searchRows,segment);
        } else {
        MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),searchRows,segment);
        MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),searchRows,same);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),searchRows,same);
        MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),searchRows,segment);
        } else {
        MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),searchRows,segment);
        MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),searchRows,same);
        }
        PipeBarrier<PIPE_V>();
        SearchCertifiedRowsBodyMulsStage2<SHARED_BOUND>(searchRows, coarseErrorFactor, same, segment, unary, j);
        if constexpr(!SHARED_BOUND) {Sub<float,false>(W(0),W(4),W(5),uint64_t(64),searchRows,same);PipeBarrier<PIPE_V>();}
        if(j+64>t_.segments) {
            CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(t_.segments-j)),CMPMODE::LT,uint64_t(64),searchRows,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
            Select(lowerScores,valid_,lowerScores,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(searchRows*64));PipeBarrier<PIPE_V>();
        }
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool SHARED_BOUND, class StageSearchRowsType, class StageCoarseErrorFactorType, class StageSameType, class StageSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchCertifiedRowsBodyMulsStage2(StageSearchRowsType& searchRows, StageCoarseErrorFactorType& coarseErrorFactor, StageSameType& same, StageSegmentType& segment, StageUnaryType& unary, StageJType& j) {
        if constexpr(CLEAN_PRIMARY64) {
            // APz is dead here; keep the clamp in a bank phase distinct from dot.
            if constexpr(sizeof(T)==2)
                Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),searchRows,unary);
            Maxs<float,false>(W(2),W(3),0.0F,uint64_t(64),searchRows,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(W(2),W(2),C(9)[j],uint64_t(64),searchRows,segment);PipeBarrier<PIPE_V>();
            const BinaryRepeatParams fused{1,0,1,8,0,8};
            {
                constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
                const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                FusedMulAdd<float,false>(W(3),minusTwo,W(2),uint64_t(64),searchRows,fused);
            }
            if constexpr(sizeof(T)==4) {
                if constexpr(!SHARED_BOUND)Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),searchRows,unary);
            }
            PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(W(4),W(2),W(3),uint64_t(64),searchRows,same);PipeBarrier<PIPE_V>();
        } else {
        SharedResourceBlock14(searchRows, unary, j, segment, coarseErrorFactor, same);
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSearchRowsType, class StageLowerScoresType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchCertifiedRowsBodyCompareStage(StageSearchRowsType& searchRows, StageLowerScoresType& lowerScores, StageJType& j) {
        if constexpr(CLEAN_PRIMARY64) {
            const uint32_t group=j/256,slot=(j/64)%4;
            auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
            if(searchRows==64) {
                Compare<float,uint8_t,false>(predicates[slot*128].template ReinterpretCast<uint8_t>(),lowerScores,W(1),CMPMODE::LE,uint64_t(64),searchRows,BinaryRepeatParams(1,1,0,8,8,1));
            } else {
                Compare<float,uint8_t,false>(mask_,lowerScores,W(1),CMPMODE::LE,uint64_t(64),searchRows,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
            }
        } else {
            Compare<float,uint8_t,false>(mask_,lowerScores,W(1),CMPMODE::LE,uint64_t(64),searchRows,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
            SharedGuardedMinimum13(j);
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool SHARED_BOUND, class StageSearchRowsType, class StageCoarseErrorFactorType, class StageSameType, class StageSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchCertifiedRowsBodyMulsStage(StageSearchRowsType& searchRows, StageCoarseErrorFactorType& coarseErrorFactor, StageSameType& same, StageSegmentType& segment, StageUnaryType& unary, StageJType& j) {
        if constexpr(CLEAN_PRIMARY64) {
            // APz is dead here; keep the clamp in a bank phase distinct from dot.
            if constexpr(sizeof(T)==2)
                Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),searchRows,unary);
            Maxs<float,false>(W(2),W(3),0.0F,uint64_t(64),searchRows,unary);
            Min<float,false>(W(2),W(2),C(9)[j],uint64_t(64),searchRows,segment);
            const BinaryRepeatParams fused{1,0,1,8,0,8};
            if constexpr(sizeof(T)==4) {
                constexpr uint32_t constantBase=4*(CAPACITY+32)+CAPACITY+8;
                const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                FusedMulAdd<float,false>(W(3),minusTwo,W(2),uint64_t(64),searchRows,fused);
            } else {
                constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
                const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                FusedMulAdd<float,false>(W(3),minusTwo,W(2),uint64_t(64),searchRows,fused);
            }
            if constexpr(sizeof(T)==4) {
                if constexpr(!SHARED_BOUND)Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),searchRows,unary);
            }

            MulAddDst<float,float,false>(W(4),W(2),W(3),uint64_t(64),searchRows,same);
        } else {
        SharedResourceBlock14(searchRows, unary, j, segment, coarseErrorFactor, same);
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool SHARED_BOUND>
    __aicore__ __attribute__((noinline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedCoreCertifiedRowsImpl(bool corePointSafe) {
        SearchCertifiedRowsBody<SHARED_BOUND,false>(corePointSafe);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedCoreCertifiedRowsAlignedImpl(bool corePointSafe) {
        SearchCertifiedRowsBody<true,true>(corePointSafe);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SpatialStateC(uint32_t k) {
        static_assert(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Native Float full spatial resource required");
        constexpr uint32_t pitch=1056;
        const uint32_t offset=k>=6 && k<=8?6*pitch+(k-6)*1024:k*pitch;
        return LocalTensor<float>(TPosition::VECCALC,offset*4,14*pitch-offset);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SpatialStateW(uint32_t k) {
        static_assert(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Native Float full spatial resource required");
        constexpr uint32_t pitch=4224,base=17952;
        constexpr uint32_t phases[13]={32,0,0,64,96,32,96,0,0,0,0,0,0};
        const uint32_t offset=k*pitch+phases[k];
        return LocalTensor<float>(TPosition::VECCALC,(base+offset)*4,6*pitch-offset);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SpatialStateP(uint32_t k) {
        static_assert(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Native Float full spatial resource required");
        constexpr uint32_t pitch=1056,base=14784;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*pitch)*4,3*pitch-k*pitch);
    }
#endif
