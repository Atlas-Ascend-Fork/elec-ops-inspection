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
    template<class StageValidLanesType, class StageFiniteBitsType, class StagePossibleBitsType>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CertifyRefinedBodyReturnStage(StageValidLanesType& validLanes, StageFiniteBitsType& finiteBits, StagePossibleBitsType& possibleBits) {
        if constexpr(CAPACITY==256) {
            if((validLanes>>8)==0) {
                const uint64_t selected=possibleBits&validLanes;
                return (finiteBits&validLanes)==validLanes && selected && !(selected&(selected-1)) ? 1U : 0U;
            }
        }
        uint32_t result=0;
        for(uint32_t group=0;group<8;++group) {
            const uint32_t active=uint32_t((validLanes>>(8*group))&255U);
            const uint32_t finiteGroup=uint32_t((finiteBits>>(8*group))&255U);
            const uint32_t candidates=uint32_t((possibleBits>>(8*group))&255U);
            if(active && (finiteGroup&active)==active && candidates && !(candidates&(candidates-1U)))
                result|=uint32_t(1)<<group;
        }
        return result;

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageValidLanesType, class StageFiniteType, class StageNormFiniteType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CertifyRefinedBodyMulsStage(StageValidLanesType& validLanes, StageFiniteType& finite, StageNormFiniteType& normFinite, uint32_t& stageResult) {
        if constexpr(DEFER_DIRECTION_LOW) {
            Muls(FewWork(5),FewWork(2),1.8189894035458565e-12F,64);PipeBarrier<PIPE_V>();
            Compare(normFinite,FewWork(1),FewWork(5),CMPMODE::GT,64);PipeBarrier<PIPE_V>();
            And(finite.template ReinterpretCast<uint16_t>(),finite.template ReinterpretCast<uint16_t>(),normFinite.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            if(validLanes==1) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                { stageResult = uint32_t(finite.template ReinterpretCast<uint64_t>().GetValue(0)&1U); return true; }
            }
            // sqrt(s*n) <= (s+n)/2. Both constants exceed the old
            // coefficients after this substitution, including float rounding slack.
            Muls(FewWork(3),FewWork(1),1.25e-6F,64);
            Muls(FewWork(5),FewWork(2),2.5e-7F,64);PipeBarrier<PIPE_V>();
            Add(FewWork(3),FewWork(3),FewWork(5),64);PipeBarrier<PIPE_V>();
        } else {
        Mul(FewWork(3),FewWork(1),FewWork(2),64);PipeBarrier<PIPE_V>();
        Sqrt(FewWork(3),FewWork(3),64);PipeBarrier<PIPE_V>();
        Muls(FewWork(3),FewWork(3),4.76837158203125e-7F,64);
        Muls(FewWork(4),FewWork(1),9.5367431640625e-7F,64);
        Muls(FewWork(5),FewWork(2),9.094947017729282e-13F,64);PipeBarrier<PIPE_V>();
        Add(FewWork(3),FewWork(3),FewWork(4),64);PipeBarrier<PIPE_V>();
        Add(FewWork(3),FewWork(3),FewWork(5),64);PipeBarrier<PIPE_V>();
        }
        return false;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CertifyRefined(
        const LocalTensor<float>& minima,uint64_t validLanes) {
        return CertifyRefinedBody(minima,validLanes);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineFewStatic(uint32_t row,uint32_t chunks,uint64_t& uniqueRows) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+1024+192+13*64;
        const LocalTensor<float> chunkLower_(TPosition::VECCALC,(workBase+6*(ROWS*64+128)+64+192)*4,1024);
        const LocalTensor<float> compactP_(TPosition::VECCALC,(stateBase-13*64-192)*4,192);
        const LocalTensor<float> segmentIndices_(TPosition::VECCALC,(workBase+6*(ROWS*64+128))*4,64);
        const LocalTensor<int32_t> outputOffset_(TPosition::VECCALC,(stateBase+128+16+14*64)*4,64);
        const LocalTensor<int32_t> offset_(TPosition::VECCALC,(stateBase+128+16+11*64)*4,64);
        const LocalTensor<float> record_(TPosition::VECCALC,stateBase*4,128);
        const LocalTensor<float> record2_(TPosition::VECCALC,(stateBase+128)*4,16);
        const LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,(stateBase+128+16+10*64)*4,64);
        const LocalTensor<int32_t> idx_(TPosition::VECCALC,(stateBase+128+16+3*64)*4,64);
        const LocalTensor<float> tileSecond_(TPosition::VECCALC,(stateBase+128+16+8*64)*4,64);
        const LocalTensor<uint8_t> valid_(TPosition::VECCALC,(stateBase+128+16+16*64+ROWS*2)*4,ROWS*8);
        const LocalTensor<uint8_t> smallMask_(TPosition::VECCALC,(stateBase+128+16+16*64+ROWS*4)*4,32);
        // Each cached set contains its chunk's true minimizer; chunk bounds
        // exclude only chunks that cannot improve the incumbent.
        Muls(offset_,outputOffset_,int32_t(32),8);PipeBarrier<PIPE_V>();
        Adds(offset_,offset_,int32_t(2048+row*4),8);PipeBarrier<PIPE_V>();
        Gather(valid_.template ReinterpretCast<uint32_t>(),chunkLower_.template ReinterpretCast<uint32_t>(),
            offset_.template ReinterpretCast<uint32_t>(),0U,8);
        Duplicate(FewWork(6),0.0F,8);PipeBarrier<PIPE_V>();
        uint32_t wordBits=chunks&15U;
        wordBits=(wordBits|(wordBits<<2))&0x33U;
        wordBits=(wordBits|(wordBits<<1))&0x55U;
        wordBits|=wordBits<<1;
        SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t((~wordBits)&255U));
        Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(0),uint64_t(64),1,1,8);
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        #pragma unroll
        for(uint32_t chunk=0;chunk<COEFFICIENTS/64;++chunk)
            Adds(FewWork(7)[chunk*64],segmentIndices_,float(int32_t(chunk*64)),64);
        PipeBarrier<PIPE_V>();
        uint64_t kept=0;
        GatherMask(FewWork(6),FewWork(7),valid_.template ReinterpretCast<uint32_t>(),true,COEFFICIENTS,
            GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
        if(kept==0 || kept>8)return false;
        Copy(record2_,FewWork(6),8,1,CopyRepeatParams(1,1,1,1));
        Cast(offset_,FewWork(6),RoundMode::CAST_RINT,8);return RefineFewStaticMulsStage(row, uniqueRows, compactP_, segmentIndices_, outputOffset_, offset_, record_, record2_, tileIdx_, idx_, tileSecond_, valid_, smallMask_, kept);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageRowType, class StageUniqueRowsType, class StageCompactP_Type, class StageSegmentIndices_Type, class StageOutputOffset_Type, class StageOffset_Type, class StageRecord_Type, class StageRecord2_Type, class StageTileIdx_Type, class StageIdx_Type, class StageTileSecond_Type, class StageValid_Type, class StageSmallMask_Type, class StageKeptType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineFewStaticMulsStage(StageRowType& row, StageUniqueRowsType& uniqueRows, StageCompactP_Type& compactP_, StageSegmentIndices_Type& segmentIndices_, StageOutputOffset_Type& outputOffset_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRecord2_Type& record2_, StageTileIdx_Type& tileIdx_, StageIdx_Type& idx_, StageTileSecond_Type& tileSecond_, StageValid_Type& valid_, StageSmallMask_Type& smallMask_, StageKeptType& kept) {
        PipeBarrier<PIPE_V>();
        Muls(offset_,offset_,int32_t(4),8);PipeBarrier<PIPE_V>();
        auto selectedColumns=FewWork(0)[16*1056];
#pragma unroll
        for(uint32_t col=0;col<14;++col)if(col<6||col==13||(col>9&&!DEFER_DIRECTION_LOW))
            Gather(selectedColumns[col*128],FewCoefficients(col),offset_.template ReinterpretCast<uint32_t>(),0U,8);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Brcb(FewWork(0),compactP_[axis*64+(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Copy(FixedFewP(axis),FewWork(0)[(row%8)*8],8,1,CopyRepeatParams(1,0,1,0));PipeBarrier<PIPE_V>();
        }
        Duplicate(valid_.template ReinterpretCast<uint32_t>(),uint32_t(0),8);PipeBarrier<PIPE_V>();
        Duplicate(valid_.template ReinterpretCast<uint32_t>(),uint32_t((uint32_t(1)<<uint32_t(kept))-1),1);PipeBarrier<PIPE_V>();
        bool accepted;
        if constexpr(sizeof(T)==4 && CAPACITY==256
#ifdef DTYPE_TREE_POINTS
            && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ) accepted=FixedFewInlineBody<true>();
        else accepted=FixedFewCompensated();
        if(accepted) {
            // Interval certification below resolves ambiguous candidates exactly.
            WholeReduceMin(record_,FixedFewW(0),8,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
            Gather(offset_,record_.template ReinterpretCast<int32_t>(),outputOffset_.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(4),1);PipeBarrier<PIPE_V>();
            Gather(tileIdx_.template ReinterpretCast<float>(),record2_,offset_.template ReinterpretCast<uint32_t>(),0U,1);PipeBarrier<PIPE_V>();
            Cast(tileIdx_,tileIdx_.template ReinterpretCast<float>(),RoundMode::CAST_RINT,1);PipeBarrier<PIPE_V>();
        }
        if(!accepted)return false;
        bool unique;
        if constexpr(DEFER_DIRECTION_LOW)unique=(CertifyRefinedBody(record_,(uint64_t(1)<<kept)-1)&1U)!=0;
        else unique=kept==1 || (CertifyRefined(record_,(uint64_t(1)<<kept)-1)&1U);
#if defined(DTYPE_TREE_POINTS)
        if(!unique)unique=ResolveExactSingle(uint32_t(kept),record2_,record_,tileIdx_,offset_,outputOffset_);
#endif
        SharedResourceBlock10(unique, uniqueRows, row, record_, tileIdx_, smallMask_, segmentIndices_, tileSecond_, idx_);
        return true;

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SparseW(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*160)*4,160);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SparseC(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch+16*1056;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*128)*4,128);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PredicatePopulation(uint64_t bits) {
        return uint32_t(__builtin_popcountll(bits));
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline uint64_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SavedChunkPredicate(uint32_t row,uint32_t chunk) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        // Post-search snapshot lies above generic W0..12 and below selected columns.
        // Primary/few/zero-correction helpers and later fallback rows do not write it.
        constexpr uint32_t snapshot=workBase+14*1056;
        const uint32_t group=chunk/4,slot=chunk%4;
        const LocalTensor<uint64_t> saved(TPosition::VECCALC,(snapshot+group*512+slot*128)*4+row*8,1);
        return saved.GetValue(0);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OVERLAPPING_CHUNKS>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineFewCandidates(uint32_t row,uint32_t chunks,uint64_t& uniqueRows) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<float> compactP_(TPosition::VECCALC,(stateBase-13*64-192)*4,192);
        const LocalTensor<float> segmentIndices_(TPosition::VECCALC,(workBase+6*(ROWS*64+128))*4,64);
        const LocalTensor<int32_t> outputOffset_(TPosition::VECCALC,(stateBase+128+16+14*64)*4,64);
        const LocalTensor<int32_t> offset_(TPosition::VECCALC,(stateBase+128+16+11*64)*4,64);
        const LocalTensor<float> record_(TPosition::VECCALC,stateBase*4,128);
        const LocalTensor<float> record2_(TPosition::VECCALC,(stateBase+128)*4,16);
        const LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,(stateBase+128+16+10*64)*4,64);
        const LocalTensor<int32_t> idx_(TPosition::VECCALC,(stateBase+128+16+3*64)*4,64);
        const LocalTensor<uint32_t> cert(TPosition::VECCALC,(stateBase+128+16+4*64)*4,64);
        const LocalTensor<uint32_t> certHigh(TPosition::VECCALC,(stateBase+128+16+5*64)*4,64);
        const LocalTensor<float> tileSecond_(TPosition::VECCALC,(stateBase+128+16+8*64)*4,64);
        const LocalTensor<uint8_t> valid_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        const LocalTensor<uint8_t> smallMask_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*4)*4,32);
        uint64_t kept;
        bool refineFewCandidatesRefineFewCandidatesGetValueStageStageResult{};
        if (RefineFewCandidatesRefineFewCandidatesGetValueStageStage<OVERLAPPING_CHUNKS>(row, chunks, segmentIndices_, idx_, cert, certHigh, valid_, kept, refineFewCandidatesRefineFewCandidatesGetValueStageStageResult)) return refineFewCandidatesRefineFewCandidatesGetValueStageStageResult;
        Copy(record2_,FewWork(6),8,1,CopyRepeatParams(1,1,1,1));
        Cast(offset_,FewWork(6),RoundMode::CAST_RINT,8);PipeBarrier<PIPE_V>();
        Muls(offset_,offset_,int32_t(4),8);PipeBarrier<PIPE_V>();
        auto selectedColumns=FewWork(0)[16*1056];
        #pragma unroll
        for(uint32_t col=0;col<14;++col)if(col<6||col>9)
            Gather(selectedColumns[col*128],FewCoefficients(col),offset_.template ReinterpretCast<uint32_t>(),0U,8);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Brcb(FewWork(axis),compactP_[axis*64+(row/8)*8],1,BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Copy(FixedFewP(axis),FewWork(axis)[(row%8)*8],8,1,CopyRepeatParams(1,0,1,0));
        PipeBarrier<PIPE_V>();
        CompareScalar(valid_,segmentIndices_,float(int32_t(kept)),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
        const bool accepted=FixedFewCompensated();
        return RefineFewCandidatesWholeReduceMinStage(row, uniqueRows, segmentIndices_, outputOffset_, offset_, record_, record2_, tileIdx_, idx_, tileSecond_, smallMask_, kept, accepted);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageRowType, class StageUniqueRowsType, class StageSegmentIndices_Type, class StageOutputOffset_Type, class StageOffset_Type, class StageRecord_Type, class StageRecord2_Type, class StageTileIdx_Type, class StageIdx_Type, class StageTileSecond_Type, class StageSmallMask_Type, class StageKeptType, class StageAcceptedType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineFewCandidatesWholeReduceMinStage(StageRowType& row, StageUniqueRowsType& uniqueRows, StageSegmentIndices_Type& segmentIndices_, StageOutputOffset_Type& outputOffset_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRecord2_Type& record2_, StageTileIdx_Type& tileIdx_, StageIdx_Type& idx_, StageTileSecond_Type& tileSecond_, StageSmallMask_Type& smallMask_, StageKeptType& kept, StageAcceptedType& accepted) {
        if(accepted) {
            WholeReduceMin(record_,FixedFewW(0),8,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
            Gather(offset_,record_.template ReinterpretCast<int32_t>(),outputOffset_.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(4),1);PipeBarrier<PIPE_V>();
            Gather(tileIdx_.template ReinterpretCast<float>(),record2_,offset_.template ReinterpretCast<uint32_t>(),0U,1);PipeBarrier<PIPE_V>();
            Cast(tileIdx_,tileIdx_.template ReinterpretCast<float>(),RoundMode::CAST_RINT,1);PipeBarrier<PIPE_V>();
        }
        if(!accepted)return false;
        bool unique=kept==1 || (CertifyRefined(record_,(uint64_t(1)<<kept)-1)&1U);
#if defined(DTYPE_TREE_POINTS)
        if(!unique)unique=ResolveExactSingle(uint32_t(kept),record2_,record_,tileIdx_,offset_,outputOffset_);
#endif
        SharedResourceBlock10(unique, uniqueRows, row, record_, tileIdx_, smallMask_, segmentIndices_, tileSecond_, idx_);
        return true;

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OVERLAPPING_CHUNKS, class StageRowType, class StageChunksType, class StageSegmentIndices_Type, class StageIdx_Type, class StageCertType, class StageCertHighType, class StageValid_Type, class StageKeptType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineFewCandidatesRefineFewCandidatesGetValueStageStage(StageRowType& row, StageChunksType& chunks, StageSegmentIndices_Type& segmentIndices_, StageIdx_Type& idx_, StageCertType& cert, StageCertHighType& certHigh, StageValid_Type& valid_, StageKeptType& kept, bool& stageResult) {
        if constexpr(OVERLAPPING_CHUNKS) { bool refineFewCandidatesGetValueStageResult{};
        if (RefineFewCandidatesGetValueStage(row, chunks, segmentIndices_, idx_, valid_, kept, refineFewCandidatesGetValueStageResult)) { stageResult = refineFewCandidatesGetValueStageResult; return true; } } else {
        // The saved winning predicate contains every possible minimizer in
        // that chunk. A single overlapping chunk permits the same precise
        // refinement on only its compacted candidates.
        if(chunks==0 || (chunks & (chunks-1))!=0){ stageResult = false; return true; }
        const uint32_t chosenChunk=uint32_t(idx_.GetValue(row))/64;
        if(chunks!=(uint32_t(1)<<chosenChunk)){ stageResult = false; return true; }
        const uint32_t low=cert.GetValue(row),high=certHigh.GetValue(row);
        Duplicate(valid_.template ReinterpretCast<uint32_t>(),low,2);
        Duplicate(FewWork(6),0.0F,8);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(2));
        Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),high,uint64_t(64),1,1,8);
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));PipeBarrier<PIPE_V>();
        kept=0;
        GatherMask(FewWork(6),segmentIndices_,valid_.template ReinterpretCast<uint32_t>(),true,64,
            GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
        if(kept==0 || kept>8){ stageResult = false; return true; }
        Adds(FewWork(6),FewWork(6),float(int32_t(chosenChunk*64)),8);PipeBarrier<PIPE_V>();
        }
        return false;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageRowType, class StageChunksType, class StageSegmentIndices_Type, class StageIdx_Type, class StageValid_Type, class StageKeptType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineFewCandidatesGetValueStage(StageRowType& row, StageChunksType& chunks, StageSegmentIndices_Type& segmentIndices_, StageIdx_Type& idx_, StageValid_Type& valid_, StageKeptType& kept, bool& stageResult) {
        {
        // Each saved chunk predicate includes every possible chunk minimizer.
        // Chunk bounds exclude only chunks incapable of improving the winner.
        if(chunks==0 || (chunks & (chunks-1))==0){ stageResult = false; return true; }
        const uint32_t chosenChunk=uint32_t(idx_.GetValue(row))/64;
        if(!(chunks&(uint32_t(1)<<chosenChunk))){ stageResult = false; return true; }
        uint32_t candidates=0;
        for(uint32_t chunk=0;chunk<COEFFICIENTS/64;++chunk)if(chunks&(uint32_t(1)<<chunk)) {
            candidates+=PredicatePopulation(SavedChunkPredicate(row,chunk));
            if(candidates>8){ stageResult = false; return true; }
        }
        if(candidates==0){ stageResult = false; return true; }
        Duplicate(FewWork(6),0.0F,8);
        Duplicate(valid_.template ReinterpretCast<uint32_t>(),uint32_t(0),32);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        for(uint32_t chunk=0;chunk<COEFFICIENTS/64;++chunk)if(chunks&(uint32_t(1)<<chunk)) {
            const uint64_t bits=SavedChunkPredicate(row,chunk);
            valid_.template ReinterpretCast<uint32_t>().SetValue(chunk*2,uint32_t(bits));
            valid_.template ReinterpretCast<uint32_t>().SetValue(chunk*2+1,uint32_t(bits>>32));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
#pragma unroll
        for(uint32_t chunk=0;chunk<COEFFICIENTS/64;++chunk)
            Adds(FewWork(7)[chunk*64],segmentIndices_,float(int32_t(chunk*64)),64);
        PipeBarrier<PIPE_V>();
        kept=0;
        GatherMask(FewWork(6),FewWork(7),valid_.template ReinterpretCast<uint32_t>(),true,COEFFICIENTS,
            GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
        if(kept==0 || kept>8){ stageResult = false; return true; }
        }
        return false;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineOverlappingFew(uint32_t row,uint32_t chunks,uint64_t& uniqueRows) {
        return RefineFewCandidates<true>(row,chunks,uniqueRows);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefineWinningFew(uint32_t row,uint32_t chunks,uint64_t& uniqueRows) {
        return RefineFewCandidates<false>(row,chunks,uniqueRows);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::FixedFewW(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*160)*4,160);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::FixedFewC(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch+16*1056;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*128)*4,128);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::FixedFewP(uint32_t k) {
        constexpr uint32_t pitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32);
        return LocalTensor<float>(TPosition::VECCALC,(base+k*pitch)*4,pitch);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::BatchFewCompensated() {
        constexpr uint32_t COUNT=64;
        constexpr bool REJECT_ZERO=true;
        TreeFewRefinementState<T,ROWS,CAPACITY> arena;
        constexpr uint32_t pointPitch=TreeFewRefinementState<T,ROWS,CAPACITY>::pointPitch;
        // Reconstruct exact-path geometry in scratch, keeping the normalized search cache intact.
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub(FixedFewW(axis+9),FixedFewC(axis+3),FixedFewC(axis),COUNT);
        PipeBarrier<PIPE_V>();
        Adds(FixedFewW(12),FixedFewC(13),0.0F,COUNT);PipeBarrier<PIPE_V>();
        const uint8_t repeats = (COUNT+63)/64;
        constexpr uint32_t lanes=COUNT<64?COUNT:64;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        auto axesW0=FixedFewW(0);axesW0.SetSize(3*160);
        auto axesW3=FixedFewW(3);axesW3.SetSize(3*160);
        auto axesW6=FixedFewW(6);axesW6.SetSize(3*160);
        auto axesW9=FixedFewW(9);axesW9.SetSize(3*160);
        auto axesC0=FixedFewC(0);axesC0.SetSize(3*128);
        auto axesC3=FixedFewC(3);axesC3.SetSize(3*128);
        auto axesC10=FixedFewC(10);axesC10.SetSize(3*128);
        auto axesP0=FixedFewP(0);axesP0.SetSize(3*pointPitch);
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(uint64_t(lanes));
        // Error-free subtraction preserves short distances at large coordinates.
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Sub<float,false>(axesW0,axesP0,axesC0,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,pointPitch/8,16));
        } else {
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(FixedFewW(axis), FixedFewP(axis), FixedFewC(axis), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        return BatchFewCompensatedSubStage<COUNT, pointPitch, lanes>(arena.record, arena.mask, arena.valid, arena.smallMask, repeats, unary, binary, axesW0, axesW3, axesW6, axesW9, axesC0, axesC3, axesC10, axesP0);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto COUNT, auto pointPitch, auto lanes, class StageRecordType, class StageMaskType, class StageValidType, class StageSmallMaskType, class StageRepeatsType, class StageUnaryType, class StageBinaryType, class StageAxesW0Type, class StageAxesW3Type, class StageAxesW6Type, class StageAxesW9Type, class StageAxesC0Type, class StageAxesC3Type, class StageAxesC10Type, class StageAxesP0Type>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::BatchFewCompensatedSubStage(StageRecordType& record, StageMaskType& mask, StageValidType& valid, StageSmallMaskType& smallMask, StageRepeatsType& repeats, StageUnaryType& unary, StageBinaryType& binary, StageAxesW0Type& axesW0, StageAxesW3Type& axesW3, StageAxesW6Type& axesW6, StageAxesW9Type& axesW9, StageAxesC0Type& axesC0, StageAxesC3Type& axesC3, StageAxesC10Type& axesC10, StageAxesP0Type& axesP0) {
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Sub<float,false>(axesW3,axesP0,axesW0,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,pointPitch/8,20));
        } else {
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(FixedFewW(axis+3), FixedFewP(axis), FixedFewW(axis), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        BatchFewCompensatedAddStage(repeats, binary, axesW0, axesW3, axesW6, axesC0);
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Sub<float,false>(axesW6,axesP0,axesW6,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,pointPitch/8,20));
        } else {
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(FixedFewW(axis+6), FixedFewP(axis), FixedFewW(axis+6), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Add<float,false>(axesW3,axesW3,axesW6,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,20,20));
        } else {
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(FixedFewW(axis+3), FixedFewW(axis+3), FixedFewW(axis+6), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        Mul<float, false>(FixedFewW(8), FixedFewW(0), FixedFewW(9), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(8), FixedFewW(1), FixedFewW(10), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(8), FixedFewW(2), FixedFewW(11), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Div<float, false>(FixedFewW(8), FixedFewW(8), FixedFewW(12), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Maxs<float, false>(FixedFewW(8), FixedFewW(8), 0.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        Mins<float, false>(FixedFewW(8), FixedFewW(8), 1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        CompareScalar<float, uint8_t, false>(mask, FixedFewW(8), 1.0F, CMPMODE::LT, uint64_t(64), repeats, unary);
        Muls<float, false>(FixedFewW(8), FixedFewW(8), -1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        BatchFewCompensatedMulAddDstStage(repeats, binary, axesW0, axesW3, axesW9, axesC10);
        return BatchFewCompensatedAddStage2<COUNT, pointPitch, lanes>(record, mask, valid, smallMask, repeats, binary, axesW0, axesW3, axesC3, axesP0);
    }
