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
// Inline implementation of KernelTreeBatchPacked.

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::Process() {
        const uint32_t total=t_.batches*t_.points;
        const uint32_t begin=total*GetBlockIdx()/t_.blocks,end=total*(GetBlockIdx()+1)/t_.blocks;
        uint32_t cached=0xffffffffU;
        bool coefficientSafe=false;
        for(uint32_t position=begin;position<end;) {
            const uint32_t batch=position/t_.points,start=position%t_.points;
            const uint32_t validRows=Smallest(Smallest(end-position,t_.points-start),ROWS);
            liveRows_=validRows;
            count_=COEFFICIENTS;
            ProcessLoadPrefixResidentSegmentsStage(cached, coefficientSafe, batch);
            if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) {
                // The prior iteration ended at MTE3_S; coefficient setup also drains.
                DataCopyPad(rawP_,pg_[(batch*t_.points+start)*3],DataCopyExtParams{1,validRows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
                SetFlag<HardEvent::MTE2_V>(EVENT_ID3);WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
            } else {
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(rawP_,pg_[(batch*t_.points+start)*3],DataCopyExtParams{1,validRows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
            PipeBarrier<PIPE_ALL>();
            }
            if constexpr(sizeof(T)==2)Cast(pf_,rawP_,RoundMode::CAST_NONE,192);
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Gather(compactP_[axis*64],pf_,pointOffset_.template ReinterpretCast<uint32_t>(),axis*4,liveRows_);
            }
            PipeBarrier<PIPE_V>();
            #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                Brcb(P(axis),compactP_[axis*64],(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            }
            if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) count_=this->SearchPrefixOutlined(liveRows_,t_.segments);
            else Search();
            Compare(smallMask_,second_,hi_,CMPMODE::GT,64);
            PipeBarrier<PIPE_V>();
            And(smallMask_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            uint64_t certified=smallMask_.template ReinterpretCast<uint64_t>().GetValue(0);
            const uint64_t active=(~uint64_t(0))>>(64-validRows);
            uint64_t exactRows=active&~certified,uniqueRows=0;
            ProcessCopyStage3(end, cached, coefficientSafe, position, batch, start, validRows, certified, active, exactRows, uniqueRows);}
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageEndType, class StageCachedType, class StageCoefficientSafeType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageCertifiedType, class StageActiveType, class StageExactRowsType, class StageUniqueRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessCopyStage3(StageEndType& end, StageCachedType& cached, StageCoefficientSafeType& coefficientSafe, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageCertifiedType& certified, StageActiveType& active, StageExactRowsType& exactRows, StageUniqueRowsType& uniqueRows) {
        if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) {
            if((certified&active)!=active) {
                // Search predicates survive subsequent point broadcasts and exact scratch reuse.
        #pragma unroll
                for(uint32_t group=0;group<3;++group)
                    Copy(work_[14*1056+group*512],xyz_[group*pp_+ROWS*8],64,8,CopyRepeatParams(1,1,8,8));
                PipeBarrier<PIPE_V>();
            }
        }
        constexpr uint32_t refinementRows=64;
        Muls(offset_,idx_,int32_t(4),64);PipeBarrier<PIPE_V>();
#pragma unroll
        for(uint32_t col=0;col<13;++col)
            if(col<6 || col>9)Gather(refined_[col*64],C(col),offset_.template ReinterpretCast<uint32_t>(),0U,refinementRows);
        Gather(refined_[9*64],C(13),offset_.template ReinterpretCast<uint32_t>(),0U,refinementRows);
        PipeBarrier<PIPE_V>();
        auto originalColumns=columns_,originalPoints=xyz_;
        const uint32_t originalCp=cp_,originalPp=pp_;
        columns_=refined_;cp_=64;xyz_=compactP_;pp_=64;count_=refinementRows;wp_=96;
        if constexpr(!Base::NATIVE_FLOAT_PREFIX_SPARSE) {
            Duplicate(valid_.template ReinterpretCast<uint16_t>(),uint16_t(65535),128);PipeBarrier<PIPE_V>();
        }
        Compensated();
        Copy(tileSecond_,W(0),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        columns_=originalColumns;cp_=originalCp;xyz_=originalPoints;pp_=originalPp;count_=m_;wp_=m_+32;
        if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) {
            if((certified&active)!=active)
                certified|=this->RefineSparseRows(active&~certified,validRows,t_.segments,uniqueRows);
        }
        bool packed=false;
        if constexpr(CAPACITY==1024 && COEFFICIENTS<=576 && sizeof(T)==4) {
            if((certified & active) != active) {
                if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE)packed=RefinePrefixPacked(certified,active);
                else packed=RefinePacked(certified,active);
            }
        }
        ProcessRefinePrefixUnpackedRowsStage(validRows, certified, active, packed);
        Sqrt(tileSecond_,tileSecond_,64);ProcessDefinedStage(end, cached, coefficientSafe, position, batch, start, validRows, exactRows, uniqueRows);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageEndType, class StageCachedType, class StageCoefficientSafeType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageExactRowsType, class StageUniqueRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDefinedStage(StageEndType& end, StageCachedType& cached, StageCoefficientSafeType& coefficientSafe, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageExactRowsType& exactRows, StageUniqueRowsType& uniqueRows) {
        PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_PACKED_EXACT
        if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) {
            RefineWideFloatRows((GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),t_.segments,
                uint32_t(work_.GetPhyAddr()),uint32_t(compactP_.GetPhyAddr()),
                uint32_t(idx_.GetPhyAddr()),uint32_t(tileSecond_.GetPhyAddr()),liveRows_,coefficientSafe);
        }
#endif // TREE_PACKED_EXACT
        auto selectedOutput=outT_;
        if constexpr(sizeof(T)==2)Cast(selectedOutput,tileSecond_,RoundMode::CAST_RINT,64);
        else selectedOutput=tileSecond_.template ReinterpretCast<T>();
        if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID3);WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        } else PipeBarrier<PIPE_ALL>();
        DataCopyPad(dg_[batch*t_.points+start],selectedOutput,DataCopyExtParams{1,validRows*uint32_t(sizeof(T)),0,0,0});
        DataCopyPad(ig_[batch*t_.points+start],idx_,DataCopyExtParams{1,validRows*4,0,0,0});
        if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE) {
            SetFlag<HardEvent::MTE3_S>(EVENT_ID3);WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
        } else PipeBarrier<PIPE_ALL>();
#if defined(DTYPE_TREE_POINTS)
        exactRows&=~uniqueRows;
        ProcessTreeGeneralExactFallbackStage(end, cached, position, batch, validRows, exactRows);
#endif
        wp_=ROWS*64+128;
        position+=validRows;


    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class ControlArgument0, class ControlArgument1>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessRefinePrefixUnpackedRowsStageControlStage6(ControlArgument0& row, ControlArgument1& candidates) {
        #pragma unroll
                for(uint32_t axis=0;axis<3;++axis) {
                    Brcb(W(0),compactP_[axis*64+(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                    Copy(P(axis),W(0)[(row%8)*8],64,candidates/64,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
                }
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class ControlArgument0, class ControlArgument1>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessRefinePrefixUnpackedRowsStageControlStage7(ControlArgument0& certified, ControlArgument1& row) {
if(!(certified&(uint64_t(1)<<row))) {
                // Only chunks whose conservative lower bound overlaps the incumbent can win.
                CommonVectorStage55(row);
                const uint32_t chunks=smallMask_.template ReinterpretCast<uint32_t>().GetValue(0)&((1U<<((t_.segments+63)/64))-1);
                uint32_t first=0,second=0,hits=0;
                for(uint32_t chunk=0;chunk<(t_.segments+63)/64;++chunk)if(chunks&(1U<<chunk)) {
                    if(hits==0)first=chunk;else if(hits==1)second=chunk;++hits;
                }
                const bool sparse=hits>0&&hits<=2;
                const uint32_t candidates=sparse?hits*64:COEFFICIENTS;
                auto searchColumns=columns_;const uint32_t searchCp=cp_;
                ProcessCopyStage2(first, second, hits, sparse);
                count_=candidates;wp_=sparse?160:m_+32;
ProcessRefinePrefixUnpackedRowsStageControlStage6(row, candidates);
                SharedResourceBlock4(sparse, first, hits, second, candidates, row, searchColumns, searchCp);}
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageValidRowsType, class StageCertifiedType, class StageActiveType, class StagePackedType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessRefinePrefixUnpackedRowsStage(StageValidRowsType& validRows, StageCertifiedType& certified, StageActiveType& active, StagePackedType& packed) {
        if((certified & active) != active && !packed) {
            if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE)RefinePrefixUnpackedRows(certified,validRows);
            else {
            for(uint32_t row=0;row<validRows;++row) ProcessRefinePrefixUnpackedRowsStageControlStage7(certified, row);

            }
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageRowType, class StageSparseType, class StageSearchColumnsType, class StageSearchCpType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessMulsStage(StageRowType& row, StageSparseType& sparse, StageSearchColumnsType& searchColumns, StageSearchCpType& searchCp) {
        PipeBarrier<PIPE_V>();
        if(sparse) {
            Muls(tileIdx_,offset_,int32_t(32),1);PipeBarrier<PIPE_V>();
            Gather(tileIdx_,record2_.template ReinterpretCast<int32_t>(),tileIdx_.template ReinterpretCast<uint32_t>(),0U,1);
        } else Muls(tileIdx_,offset_,int32_t(64),1);
        Muls(offset_,offset_,int32_t(8),1);PipeBarrier<PIPE_V>();
        Gather(offset_,W(6).template ReinterpretCast<int32_t>(),offset_.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
        Add(tileIdx_,tileIdx_,offset_,1);PipeBarrier<PIPE_V>();
        columns_=searchColumns;cp_=searchCp;count_=m_;wp_=m_+32;
        this->CommitRefinedCandidate(row);

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageCachedType, class StageCoefficientSafeType, class StageBatchType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessLoadPrefixResidentSegmentsStage(StageCachedType& cached, StageCoefficientSafeType& coefficientSafe, StageBatchType& batch) {
        if(batch!=cached){
            wp_=m_+32;
            if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE)
                this->LoadPrefixResidentSegments((GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),t_.segments);
            else LoadSegments(batch);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_PACKED_EXACT
            if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE)
                coefficientSafe=this->template CoarseCoordinatesSafe<true>();
#endif // TREE_PACKED_EXACT
            wp_=ROWS*64+128;cached=batch;
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageFirstType, class StageSecondType, class StageHitsType, class StageSparseType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessCopyStage2(StageFirstType& first, StageSecondType& second, StageHitsType& hits, StageSparseType& sparse) { Base::RefineFullFloatFallbackBodyCopyStage(first,second,hits,sparse);
    }

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageEndType, class StageCachedType, class StagePositionType, class StageBatchType, class StageValidRowsType, class StageExactRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessTreeGeneralExactFallbackStage(StageEndType& end, StageCachedType& cached, StagePositionType& position, StageBatchType& batch, StageValidRowsType& validRows, StageExactRowsType& exactRows) {
        if(exactRows) {
            PipeBarrier<PIPE_ALL>();
            for(uint32_t row=0;row<validRows;++row)if(exactRows&(uint64_t(1)<<row)) {
                const uint32_t pos=position+row;
                TreeGeneralExactFallback<T>((GM_ADDR)pg_[pos*3].GetPhyAddr(),
                    (GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),
                    (GM_ADDR)dg_[pos].GetPhyAddr(),(GM_ADDR)ig_[pos].GetPhyAddr(),t_.segments);
            }
            if(position+validRows<end) {
                if constexpr(Base::NATIVE_FLOAT_PREFIX_SPARSE)this->RestartBoundedAfterExact();
                else {
                    const TreeLineMinDistanceTilingData t=t_;
                    this->Init((GM_ADDR)pg_[0].GetPhyAddr(),(GM_ADDR)sg_[0].GetPhyAddr(),
                               (GM_ADDR)dg_[0].GetPhyAddr(),(GM_ADDR)ig_[0].GetPhyAddr(),t);
                }
                cached=0xffffffffU;
            }
        }
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class ControlArgument0, class ControlArgument1>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefinePrefixUnpackedRowsControlStage8(ControlArgument0& row, ControlArgument1& candidates) {
                    #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis) {
                        Brcb(W(0),compactP_[axis*64+(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                        Copy(P(axis),W(0)[(row%8)*8],64,candidates/64,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
                    }
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class ControlArgument0, class ControlArgument1>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefinePrefixUnpackedRowsControlStage9(ControlArgument0& certified, ControlArgument1& row) { ProcessRefinePrefixUnpackedRowsStageControlStage7(certified,row);
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefinePrefixUnpackedRows(uint64_t certified,uint32_t validRows) {
        static_assert(Base::NATIVE_FLOAT_PREFIX_SPARSE,"Native Float prefix cache required");

                for(uint32_t row=0;row<validRows;++row) RefinePrefixUnpackedRowsControlStage9(certified, row);

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline bool KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefinePrefixPacked(uint64_t certified,uint64_t active) {
        static_assert(Base::NATIVE_FLOAT_PREFIX_SPARSE,"Native Float prefix cache required");
        return RefinePacked(certified,active);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline bool KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefinePacked(uint64_t certified,uint64_t active) {
        // Non-finite search bounds keep the complete per-point fallback.
        CompareScalar(smallMask_,hi_,__builtin_inff(),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if((smallMask_.template ReinterpretCast<uint64_t>().GetValue(0)&active)!=active)return false;
        const uint64_t uncertain=active&~certified;
        const uint32_t chunks=(t_.segments+63)/64;
        const BinaryRepeatParams same{1,1,1,8,8,8};
        SetMaskNorm();
        SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x5555555555555555ULL));
        Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(uncertain),uint64_t(64),1,1,8);
        SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaaaaaaaaaaaaaaaaULL));
        Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(uncertain>>32),uint64_t(64),1,1,8);
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Duplicate(W(0),__builtin_inff(),64);PipeBarrier<PIPE_V>();
        SetCmpMask(valid_);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileSecond_,W(0),tileSecond_,1,same);PipeBarrier<PIPE_V>();
        auto originalColumns=columns_,originalPoints=xyz_;
        const uint32_t originalCp=cp_,originalPp=pp_;
        RefinePackedDuplicateStage(uncertain, chunks, same, originalColumns, originalPoints, originalCp, originalPp);
        return true;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageUncertainType, class StageChunksType, class StageSameType, class StageOriginalColumnsType, class StageOriginalPointsType, class StageOriginalCpType, class StageOriginalPpType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefinePackedDuplicateStage(StageUncertainType& uncertain, StageChunksType& chunks, StageSameType& same, StageOriginalColumnsType& originalColumns, StageOriginalPointsType& originalPoints, StageOriginalCpType& originalCp, StageOriginalPpType& originalPp) {
        for(uint32_t chunkBegin=0;chunkBegin<chunks;chunkBegin+=12) {
            Duplicate(mask_.template ReinterpretCast<uint16_t>(),uint16_t(0),64);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Compare<float,uint8_t,false>(mask_,chunkLower_,hi_,CMPMODE::LE,uint64_t(64),chunks,BinaryRepeatParams(1,1,1,8,8,0));PipeBarrier<PIPE_V>();
            Or(valid_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),mask_[64].template ReinterpretCast<uint16_t>(),32);PipeBarrier<PIPE_V>();
            Or(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),valid_[32].template ReinterpretCast<uint16_t>(),16);PipeBarrier<PIPE_V>();
            Duplicate(W(0).template ReinterpretCast<int32_t>(),int32_t(31),8);
            ShiftRight(offset_,outputOffset_,int32_t(1),8);PipeBarrier<PIPE_V>();
            Adds(offset_,offset_,int32_t(16),8);PipeBarrier<PIPE_V>();
            And(offset_.template ReinterpretCast<uint16_t>(),offset_.template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),16);PipeBarrier<PIPE_V>();
            Gather(W(1).template ReinterpretCast<uint32_t>(),valid_.template ReinterpretCast<uint32_t>(),offset_.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            Or(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),W(1).template ReinterpretCast<uint16_t>(),16);PipeBarrier<PIPE_V>();
            ShiftRight(offset_,outputOffset_,int32_t(1),8);PipeBarrier<PIPE_V>();
            Adds(offset_,offset_,int32_t(8),8);PipeBarrier<PIPE_V>();
            And(offset_.template ReinterpretCast<uint16_t>(),offset_.template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),16);PipeBarrier<PIPE_V>();
            Gather(W(1).template ReinterpretCast<uint32_t>(),valid_.template ReinterpretCast<uint32_t>(),offset_.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            Or(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),W(1).template ReinterpretCast<uint16_t>(),16);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint64_t covered=valid_.template ReinterpretCast<uint64_t>().GetValue(0);
            const uint64_t missing=uncertain&~covered;
            SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x5555555555555555ULL));
            Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(missing),uint64_t(64),1,1,8);
            SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaaaaaaaaaaaaaaaaULL));
            Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(missing>>32),uint64_t(64),1,1,8);
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));PipeBarrier<PIPE_V>();
            Or(mask_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),chunks*4);PipeBarrier<PIPE_V>();
            SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x5555555555555555ULL));
            Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(uncertain),uint64_t(64),1,1,8);
            SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaaaaaaaaaaaaaaaaULL));
            Duplicate<uint32_t,false>(valid_.template ReinterpretCast<uint32_t>(),uint32_t(uncertain>>32),uint64_t(64),1,1,8);
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));PipeBarrier<PIPE_V>();
            And(mask_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),chunks*4);PipeBarrier<PIPE_V>();
            const uint32_t groupChunks=Smallest(chunks-chunkBegin,12);
            MakeIndices(work_.template ReinterpretCast<int32_t>(),int32_t(chunkBegin*64),groupChunks*64);PipeBarrier<PIPE_V>();
            uint64_t kept=0;
            GatherMask(refined_,work_,mask_[chunkBegin*8].template ReinterpretCast<uint32_t>(),true,groupChunks*64,GatherMaskParams(1,1,8,1),kept);PipeBarrier<PIPE_V>();
            RefinePackedSmallestStage(same, originalColumns, originalPoints, originalCp, originalPp, kept);
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSameType, class StageOriginalColumnsType, class StageOriginalPointsType, class StageOriginalCpType, class StageOriginalPpType, class StageKeptType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefinePackedSmallestStage(StageSameType& same, StageOriginalColumnsType& originalColumns, StageOriginalPointsType& originalPoints, StageOriginalCpType& originalCp, StageOriginalPpType& originalPp, StageKeptType& kept) {
        for(uint32_t first=0;first<uint32_t(kept);first+=8) {
            const uint32_t queries=Smallest(uint32_t(kept)-first,8),pairs=queries*64;
            wp_=544;
            Copy(W(0),refined_[first],8,1,CopyRepeatParams(1,1,1,1));
            Duplicate(W(2).template ReinterpretCast<uint32_t>(),uint32_t(63),8);PipeBarrier<PIPE_V>();
            ShiftRight(W(1).template ReinterpretCast<uint32_t>(),W(0).template ReinterpretCast<uint32_t>(),uint32_t(6),8);
            And(W(3).template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),W(2).template ReinterpretCast<uint16_t>(),16);PipeBarrier<PIPE_V>();
            ShiftLeft(W(1).template ReinterpretCast<uint32_t>(),W(1).template ReinterpretCast<uint32_t>(),uint32_t(8),8);
            ShiftLeft(W(3).template ReinterpretCast<uint32_t>(),W(3).template ReinterpretCast<uint32_t>(),uint32_t(2),8);PipeBarrier<PIPE_V>();
            Brcb(W(2),W(1),1,BrcbRepeatParams(1,8));
            Brcb(W(4),W(3),1,BrcbRepeatParams(1,8));
            ShiftRight(W(5).template ReinterpretCast<int32_t>(),outputOffset_,int32_t(1),64);PipeBarrier<PIPE_V>();
            Copy(W(6),W(2),64,queries,CopyRepeatParams(1,0,8,1));
            Copy(W(7),W(4),64,queries,CopyRepeatParams(1,0,8,1));PipeBarrier<PIPE_V>();
            Add<int32_t,false>(W(6).template ReinterpretCast<int32_t>(),W(5).template ReinterpretCast<int32_t>(),W(6).template ReinterpretCast<int32_t>(),uint64_t(64),queries,BinaryRepeatParams(1,1,1,8,0,8));PipeBarrier<PIPE_V>();
            auto packedColumns=work_[8*1056];
    #pragma unroll
            for(uint32_t col=0;col<14;++col)if(col<6||col>9)
                Gather(packedColumns[col*512],C(col),W(6).template ReinterpretCast<uint32_t>(),0U,pairs);
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Gather(P(axis),compactP_[axis*64],W(7).template ReinterpretCast<uint32_t>(),0U,pairs);
            PipeBarrier<PIPE_V>();
            ShiftRight(W(6).template ReinterpretCast<int32_t>(),W(6).template ReinterpretCast<int32_t>(),int32_t(2),pairs);PipeBarrier<PIPE_V>();
            Cast(W(5),W(6).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,pairs);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,W(5),float(int32_t(t_.segments)),CMPMODE::LT,pairs);PipeBarrier<PIPE_V>();
            columns_=packedColumns;cp_=512;count_=pairs;
            this->template Compensated<true>();
            WholeReduceMin(record_,W(0),64,queries,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
            GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,1,8,0));
            GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,1,8,0));PipeBarrier<PIPE_V>();
            columns_=originalColumns;cp_=originalCp;xyz_=originalPoints;pp_=originalPp;count_=m_;wp_=m_+32;
            Brcb(W(0),tileLo_,1,BrcbRepeatParams(1,8));
            Brcb(W(1),tileIdx_.template ReinterpretCast<float>(),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            RefinePackedReinterpretCastStage(same, first, queries);
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSameType, class StageFirstType, class StageQueriesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatchPacked<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::RefinePackedReinterpretCastStage(StageSameType& same, StageFirstType& first, StageQueriesType& queries) {
        for(uint32_t query=0;query<queries;++query) {
            const uint32_t gid=refined_.template ReinterpretCast<uint32_t>().GetValue(first+query);
            Copy(W(2),W(0)[query*8],64,1,CopyRepeatParams(1,0,8,0));
            Copy(W(3),W(1)[query*8],64,1,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
            Adds(W(3).template ReinterpretCast<int32_t>(),W(3).template ReinterpretCast<int32_t>(),int32_t(gid&~uint32_t(63)),64);
            CompareScalar(smallMask_,segmentIndices_,float(int32_t(gid&63)),CMPMODE::EQ,64);
            Compare(mask_,W(2),tileSecond_,CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            And(smallMask_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            SetCmpMask(smallMask_);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileSecond_,W(2),tileSecond_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),W(3),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
        }
    }
