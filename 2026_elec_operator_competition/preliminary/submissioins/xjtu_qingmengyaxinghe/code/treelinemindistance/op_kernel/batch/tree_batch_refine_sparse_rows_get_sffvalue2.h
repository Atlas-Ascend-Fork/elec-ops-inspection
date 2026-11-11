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
    template<class StageRowsType, class StageSegmentsType, class StageUniqueRowsType, class StageChunkLower_Type, class StageOutputOffset_Type, class StageOffset_Type, class StageIdx_Type, class StageHi_Type, class StageSegmentIndices_Type, class StageValid_Type, class StageSmallMask_Type, class StageCountType, class StageValidLanesType, class StageGroupRowsType, class StageAcceptedType, class StageActiveType>
    __aicore__ __attribute__((always_inline)) inline uint64_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineSparseRowsGetSFFValueStage2(StageRowsType& rows, StageSegmentsType& segments, StageUniqueRowsType& uniqueRows, StageChunkLower_Type& chunkLower_, StageOutputOffset_Type& outputOffset_, StageOffset_Type& offset_, StageIdx_Type& idx_, StageHi_Type& hi_, StageSegmentIndices_Type& segmentIndices_, StageValid_Type& valid_, StageSmallMask_Type& smallMask_, StageCountType& count, StageValidLanesType& validLanes, StageGroupRowsType& groupRows, StageAcceptedType& accepted, StageActiveType& active) {
        for(uint64_t pendingRows=rows&active;pendingRows!=0;pendingRows&=pendingRows-1) {
            const uint32_t row=uint32_t(GetSFFValue<1>(pendingRows));
            Muls(offset_,outputOffset_,int32_t(32),16);PipeBarrier<PIPE_V>();
            Adds(offset_,offset_,int32_t(row*4),16);
            Duplicate(FewWork(0),__builtin_inff(),64);
            Brcb(FewWork(1),hi_[(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Copy(FewWork(2),FewWork(1)[(row%8)*8],64,1,CopyRepeatParams(1,0,8,0));
            Gather(FewWork(0),chunkLower_,offset_.template ReinterpretCast<uint32_t>(),0U,(segments+63)/64);PipeBarrier<PIPE_V>();
            Compare(smallMask_,FewWork(0),FewWork(2),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t chunks=smallMask_.template ReinterpretCast<uint32_t>().GetValue(0)&((1U<<((segments+63)/64))-1);
            const uint32_t chosenChunk=uint32_t(idx_.GetValue(row))/64;
            if(chunks==0 || !(chunks&(uint32_t(1)<<chosenChunk)))continue;
            uint32_t candidates=0;
            for(uint32_t pendingChunks=chunks;pendingChunks!=0;pendingChunks&=pendingChunks-1) {
                const uint32_t chunk=uint32_t(GetSFFValue<1>(uint64_t(pendingChunks)));
                uint64_t bits=SavedChunkPredicate(row,chunk);
                const uint32_t remaining=segments-chunk*64;
                if(remaining<64)bits&=(uint64_t(1)<<remaining)-1;
                candidates+=PredicatePopulation(bits);
            }
            if(candidates==0 || candidates>8)continue;
            Duplicate(FewWork(6),0.0F,8);
            Duplicate(valid_.template ReinterpretCast<uint32_t>(),uint32_t(0),32);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
            uint32_t packed=0;
            RefineSparseRowsGetSFFValueStage(segments, valid_, row, chunks, packed);
            SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
            packed=0;
            for(uint32_t pendingChunks=chunks;pendingChunks!=0;pendingChunks&=pendingChunks-1) {
                const uint32_t chunk=uint32_t(GetSFFValue<1>(uint64_t(pendingChunks)));
                Adds(FewWork(7)[packed*64],segmentIndices_,float(int32_t(chunk*64)),64);
                ++packed;
            }
            PipeBarrier<PIPE_V>();
            uint64_t kept=0;
            if (RefineSparseRowsGetSFFValueStage2GatherMaskStage(uniqueRows, valid_, count, validLanes, groupRows, accepted, row, packed, kept)) continue;}
        if(count!=0 && RefineCandidateBatch(validLanes,groupRows,uniqueRows))accepted|=groupRows;
        return accepted;

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageUniqueRowsType, class StageValid_Type, class StageCountType, class StageValidLanesType, class StageGroupRowsType, class StageAcceptedType, class StageRowType, class StagePackedType, class StageKeptType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineSparseRowsGetSFFValueStage2GatherMaskStage(StageUniqueRowsType& uniqueRows, StageValid_Type& valid_, StageCountType& count, StageValidLanesType& validLanes, StageGroupRowsType& groupRows, StageAcceptedType& accepted, StageRowType& row, StagePackedType& packed, StageKeptType& kept) {
        GatherMask(FewWork(6),FewWork(7),valid_.template ReinterpretCast<uint32_t>(),true,packed*64,
                GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
            if(kept==0 || kept>8)return true;
            Copy(BatchIds()[count*8],FewWork(6),8,1,CopyRepeatParams(1,1,1,1));PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
            BatchPointIds().SetValue(count,row*4);
            BatchPointMap().SetValue(row,count*4);
            validLanes|=((uint64_t(1)<<kept)-1)<<(count*8);
            groupRows|=uint64_t(1)<<row;
            ++count;
            SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
            if(count==8) {
                if(RefineCandidateBatch(validLanes,groupRows,uniqueRows))accepted|=groupRows;
                ClearCandidateBatch();count=0;validLanes=0;groupRows=0;
            }

        return false;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSegmentsType, class StageValid_Type, class StageRowType, class StageChunksType, class StagePackedType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineSparseRowsGetSFFValueStage(StageSegmentsType& segments, StageValid_Type& valid_, StageRowType& row, StageChunksType& chunks, StagePackedType& packed) {
        for(uint32_t pendingChunks=chunks;pendingChunks!=0;pendingChunks&=pendingChunks-1) {
            const uint32_t chunk=uint32_t(GetSFFValue<1>(uint64_t(pendingChunks)));
            uint64_t bits=SavedChunkPredicate(row,chunk);
            const uint32_t remaining=segments-chunk*64;
            if(remaining<64)bits&=(uint64_t(1)<<remaining)-1;
            valid_.template ReinterpretCast<uint32_t>().SetValue(packed*2,uint32_t(bits));
            valid_.template ReinterpretCast<uint32_t>().SetValue(packed*2+1,uint32_t(bits>>32));
            ++packed;
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool MERGED_CERTIFICATE>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::FixedFewInlineBody() {
        constexpr uint32_t COUNT=8;
        constexpr bool REJECT_ZERO=true;
        TreeFewRefinementState<T,ROWS,CAPACITY> arena;
        // Reconstruct exact-path geometry in scratch, keeping the normalized search cache intact.
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub(FixedFewW(axis+9),FixedFewC(axis+3),FixedFewC(axis),COUNT);
        PipeBarrier<PIPE_V>();
        constexpr bool directNorm=sizeof(T)==4 && CAPACITY==256
#ifdef DTYPE_TREE_POINTS
            && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ;
        const LocalTensor<float> norm=directNorm?FixedFewC(13):FixedFewW(12);
        if constexpr(!directNorm) {
            Adds(FixedFewW(12),FixedFewC(13),0.0F,COUNT);PipeBarrier<PIPE_V>();
        }
        const uint8_t repeats = (COUNT+63)/64;
        constexpr uint32_t lanes=COUNT<64?COUNT:64;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(uint64_t(lanes));
        FixedFewInlineBodySubStage(repeats, binary);
        Mul<float, false>(FixedFewW(8), FixedFewW(0), FixedFewW(9), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(8), FixedFewW(1), FixedFewW(10), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(8), FixedFewW(2), FixedFewW(11), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Div<float, false>(FixedFewW(8), FixedFewW(8), norm, uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Maxs<float, false>(FixedFewW(8), FixedFewW(8), 0.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        Mins<float, false>(FixedFewW(8), FixedFewW(8), 1.0F, uint64_t(64), repeats, unary);
        return FixedFewInlineBodyCompareScalarStage<MERGED_CERTIFICATE, COUNT, lanes>(arena.record, arena.mask, arena.valid, arena.smallMask, norm, repeats, unary, binary);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool MERGED_CERTIFICATE, auto COUNT, auto lanes, class StageRecordType, class StageMaskType, class StageValidType, class StageSmallMaskType, class StageNormType, class StageRepeatsType, class StageUnaryType, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::FixedFewInlineBodyCompareScalarStage(StageRecordType& record, StageMaskType& mask, StageValidType& valid, StageSmallMaskType& smallMask, StageNormType& norm, StageRepeatsType& repeats, StageUnaryType& unary, StageBinaryType& binary) {
        PipeBarrier<PIPE_V>();
        CompareScalar<float, uint8_t, false>(mask, FixedFewW(8), 1.0F, CMPMODE::LT, uint64_t(64), repeats, unary);
        Muls<float, false>(FixedFewW(8), FixedFewW(8), -1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            MulAddDst<float, float, false>(FixedFewW(axis), FixedFewW(8), FixedFewW(axis+9), uint64_t(64), repeats, binary);
            MulAddDst<float, float, false>(FixedFewW(axis+3), FixedFewW(8), FixedFewC(axis+10), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(FixedFewW(axis), FixedFewW(axis), FixedFewW(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(FixedFewW(axis+3), FixedFewP(axis), FixedFewC(axis+3), uint64_t(64), repeats, binary);
        SharedFixedResidualNorm5<COUNT>(mask, binary, repeats, valid);
        bool fixedFewInlineBodyWholeReduceMaxStageResult{};
        if (FixedFewInlineBodyWholeReduceMaxStage<MERGED_CERTIFICATE, lanes>(record, smallMask, norm, repeats, fixedFewInlineBodyWholeReduceMaxStageResult)) return fixedFewInlineBodyWholeReduceMaxStageResult;
        if constexpr(COUNT<64) {SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));}
        return true;

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool MERGED_CERTIFICATE, auto lanes, class StageRecordType, class StageSmallMaskType, class StageNormType, class StageRepeatsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::FixedFewInlineBodyWholeReduceMaxStage(StageRecordType& record, StageSmallMaskType& smallMask, StageNormType& norm, StageRepeatsType& repeats, bool& stageResult) {
        if constexpr(!MERGED_CERTIFICATE) {
        WholeReduceMax(FixedFewW(7),norm,lanes,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(record,FixedFewW(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Muls(record,record,1.8189894035458565e-12F,1);
        WholeReduceMin(FixedFewW(7),FixedFewW(0),lanes,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(record[64],FixedFewW(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        if(HasAmbiguousMinimum(record,smallMask)) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            { stageResult = false; return true; }
        }
        }
        return false;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageRepeatsType, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::FixedFewInlineBodySubStage(StageRepeatsType& repeats, StageBinaryType& binary) { if constexpr(DEFER_DIRECTION_LOW)CompensatedDeferred<true>(repeats,binary);else CompensatedImmediate<true>(repeats,binary);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::FixedFewCompensated() {
        return FixedFewInlineBody();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<uint32_t COUNT, bool REJECT_ZERO>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SparseCompensated() {
        // Reconstruct exact-path geometry in scratch, keeping the normalized search cache intact.
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub(SparseW(axis+9),SparseC(axis+3),SparseC(axis),COUNT);
        PipeBarrier<PIPE_V>();
        Adds(SparseW(12),SparseC(13),0.0F,COUNT);PipeBarrier<PIPE_V>();
        const uint8_t repeats = (COUNT+63)/64;
        constexpr uint32_t lanes=COUNT<64?COUNT:64;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(uint64_t(lanes));
        // Error-free subtraction preserves short distances at large coordinates.
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(SparseW(axis), SearchP<true>(axis), SparseC(axis), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(SparseW(axis+3), SearchP<true>(axis), SparseW(axis), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            Add<float, false>(SparseW(axis+6), SparseW(axis), SparseW(axis+3), uint64_t(64), repeats, binary);
            Sub<float, false>(SparseW(axis+3), SparseW(axis+3), SparseC(axis), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(SparseW(axis+6), SearchP<true>(axis), SparseW(axis+6), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(SparseW(axis+3), SparseW(axis+3), SparseW(axis+6), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(SparseW(8), SparseW(0), SparseW(9), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(SparseW(8), SparseW(1), SparseW(10), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(SparseW(8), SparseW(2), SparseW(11), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Div<float, false>(SparseW(8), SparseW(8), SparseW(12), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Maxs<float, false>(SparseW(8), SparseW(8), 0.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        Mins<float, false>(SparseW(8), SparseW(8), 1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        CompareScalar<float, uint8_t, false>(mask_, SparseW(8), 1.0F, CMPMODE::LT, uint64_t(64), repeats, unary);
        Muls<float, false>(SparseW(8), SparseW(8), -1.0F, uint64_t(64), repeats, unary);
        return SparseCompensatedMulAddDstStage<COUNT, REJECT_ZERO, lanes>(repeats, binary);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<uint32_t COUNT, bool REJECT_ZERO, auto lanes, class StageRepeatsType, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SparseCompensatedMulAddDstStage(StageRepeatsType& repeats, StageBinaryType& binary) {
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            MulAddDst<float, float, false>(SparseW(axis), SparseW(8), SparseW(axis+9), uint64_t(64), repeats, binary);
            MulAddDst<float, float, false>(SparseW(axis+3), SparseW(8), SparseC(axis+10), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(SparseW(axis), SparseW(axis), SparseW(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(SparseW(axis+3), SearchP<true>(axis), SparseC(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        if constexpr(COUNT<=64) {
            SetCmpMask(mask_);
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(SparseW(axis),SparseW(axis),SparseW(axis+3),1,binary);
        } else {
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,uint8_t,false>(SparseW(axis),mask_,SparseW(axis),SparseW(axis+3),SELMODE::VSEL_TENSOR_TENSOR_MODE,uint64_t(64),repeats,binary);
        }
        PipeBarrier<PIPE_V>();
        Mul<float, false>(SparseW(0), SparseW(0), SparseW(0), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(SparseW(0), SparseW(1), SparseW(1), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(SparseW(0), SparseW(2), SparseW(2), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Select<float, uint8_t, false>(SparseW(0), valid_, SparseW(0), __builtin_inff(), SELMODE::VSEL_TENSOR_SCALAR_MODE, uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        bool sparseCompensatedWholeReduceMaxStageResult{};
        if (SparseCompensatedWholeReduceMaxStage<COUNT, REJECT_ZERO, lanes>(repeats, sparseCompensatedWholeReduceMaxStageResult)) return sparseCompensatedWholeReduceMaxStageResult;
        if constexpr(COUNT<64) {SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));}
        return true;

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<uint32_t COUNT, bool REJECT_ZERO, auto lanes, class StageRepeatsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SparseCompensatedWholeReduceMaxStage(StageRepeatsType& repeats, bool& stageResult) {
        if constexpr(true) {
        WholeReduceMax(SparseW(7),SparseW(12),lanes,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(record_,SparseW(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Muls(record_,record_,1.8189894035458565e-12F,1);
        WholeReduceMin(SparseW(7),SparseW(0),lanes,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(record_[64],SparseW(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Compare(smallMask_,record_[64],record_,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if(smallMask_.GetValue(0)&1U) {
            if constexpr(REJECT_ZERO) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                { stageResult = false; return true; }
            } else {
            CorrectAxialZeroScores(uint32_t(SparseW(0).GetPhyAddr()),uint32_t(14*(CAPACITY+32)*4),uint32_t((CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32),
                uint32_t((14*(CAPACITY+32)+3*((CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32)+16*1056)*4),uint32_t(128),uint32_t(SparseW(1).GetPhyAddr()),uint32_t(160),uint32_t(mask_.GetPhyAddr()),
                (COUNT+255)/256*32,COUNT);
            CorrectCollinearZeroScores<sizeof(T)==2>(uint32_t(SparseW(0).GetPhyAddr()),uint32_t(14*(CAPACITY+32)*4),uint32_t((CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32),
                uint32_t((14*(CAPACITY+32)+3*((CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32)+16*1056)*4),uint32_t(128),uint32_t(work_[20*1056].GetPhyAddr()),uint32_t(record_.GetPhyAddr()),
                uint32_t(mask_.GetPhyAddr()),COUNT);
            Select(SparseW(0),valid_,SparseW(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,COUNT);PipeBarrier<PIPE_V>();
            }
        }
        }
        return false;
    }
