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
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageSpatialPointCoordinatesSafeType, class StageBeginType, class StageEndType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageSpatialCoefficientsType, class StageCorePointSafeType, class StageSharedCoefficientCoordinatesSafeType, class StageCachedType, class StagePrefetchedPointType, class StageOutputInFlightType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageLogicalSegmentsType, class StageExactPointSafeType, class StageInitialInputsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDefinedStage3(StageSpatialWorkspaceType& spatialWorkspace, StageSpatialPointCoordinatesSafeType& spatialPointCoordinatesSafe, StageBeginType& begin, StageEndType& end, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageSpatialCoefficientsType& spatialCoefficients, StageCorePointSafeType& corePointSafe, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageCachedType& cached, StagePrefetchedPointType& prefetchedPoint, StageOutputInFlightType& outputInFlight, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe, StageInitialInputsType& initialInputs) {
        ProcessDefinedStage3DefinedStage<OUTLINE_FULL_FLOAT_FALLBACK>(end, sharedCoefficientCoordinatesSafe, cached, outputInFlight, position, batch, start, initialInputs);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
        ProcessDataCopyStage(spatialWorkspace, begin, spatialCoefficientDirty, spatialPruned, spatialCoefficients, position);
#endif
        ProcessDataCopyPadStage(prefetchedPoint, position, batch, start, validRows, initialInputs);
        if constexpr(sizeof(T)==2)Cast(pf_,rawP_,RoundMode::CAST_NONE,192);
        PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(UsesOutlineFullFloatFallbackStage4<OUTLINE_FULL_FLOAT_FALLBACK>()) { ProcessMinsStage(spatialWorkspace); } else
#endif
        {
    #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Gather(compactP_[axis*64],pf_,pointOffset_.template ReinterpretCast<uint32_t>(),axis*4,liveRows_);
        }
        }
        PipeBarrier<PIPE_V>();
        ProcessSmallestStage(end, prefetchedPoint, position, validRows);
    #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            if constexpr(UsesOutlineFullFloatFallbackStage4<OUTLINE_FULL_FLOAT_FALLBACK>()) {
                Brcb(P(axis),compactP_[axis*64],spatialWorkspace?8U:(liveRows_+7)/8,BrcbRepeatParams(1,8));
            } else
#endif
            Brcb(P(axis),compactP_[axis*64],(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        }
        if constexpr(OVERLAP_POINT_INPUT) {
            // Search is the next writer of the prior output index buffer.
            if(outputInFlight){WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);outputInFlight=false;}
        }
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
        if constexpr(UsesRowsStage3()) {
            if(spatialWorkspace && spatialCoefficientDirty) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
                spatialCoefficientDirty=false;
            }
        }
#endif
        ProcessDefinedStage2<OUTLINE_FULL_FLOAT_FALLBACK>(spatialWorkspace, spatialPointCoordinatesSafe, spatialCoefficientDirty, spatialPruned, spatialDirtyCount, corePointSafe, sharedCoefficientCoordinatesSafe, logicalSegments, exactPointSafe);
        ProcessDefinedStage3UsesRowsStage7Stage<OUTLINE_FULL_FLOAT_FALLBACK>(spatialWorkspace, end, spatialCoefficientDirty, spatialPruned, spatialDirtyCount, sharedCoefficientCoordinatesSafe, cached, prefetchedPoint, outputInFlight, position, batch, start, validRows, logicalSegments, exactPointSafe);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageEndType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageSharedCoefficientCoordinatesSafeType, class StageCachedType, class StagePrefetchedPointType, class StageOutputInFlightType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageLogicalSegmentsType, class StageExactPointSafeType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDefinedStage3UsesRowsStage7Stage(StageSpatialWorkspaceType& spatialWorkspace, StageEndType& end, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageCachedType& cached, StagePrefetchedPointType& prefetchedPoint, StageOutputInFlightType& outputInFlight, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe) {
        if constexpr(UsesRowsStage2()) {
            And(smallMask_.template ReinterpretCast<uint16_t>(),second_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),4);
            PipeBarrier<PIPE_V>();
        } else {
            Compare(smallMask_,second_,hi_,CMPMODE::GT,64);
            PipeBarrier<PIPE_V>();
            And(smallMask_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint64_t certified=smallMask_.template ReinterpretCast<uint64_t>().GetValue(0);
        const uint64_t active=(~uint64_t(0))>>(64-validRows);
        uint64_t exactRows=active&~certified,uniqueRows=0;
        ProcessCopyStage3(certified, active);
        if constexpr(DEFER_DIRECTION_LOW) {
            // The genuine score certificate identifies pending refinement.
            // Prefetch its bounded helper while primary gathers/computation run.
            if(exactRows)AscendC::PreLoadImpl(reinterpret_cast<void*>(&KernelTreeBatch::RefineFewStatic),int64_t(2));
        }
        uint32_t refinementRows=64;
        if constexpr(CAPACITY==256 && sizeof(T)==4)refinementRows=validRows<=16?16:validRows<=32?32:64;
        Muls(offset_,idx_,int32_t(4),64);ProcessDefinedStage3GatherStage<OUTLINE_FULL_FLOAT_FALLBACK>(spatialWorkspace, end, spatialCoefficientDirty, spatialPruned, spatialDirtyCount, sharedCoefficientCoordinatesSafe, cached, prefetchedPoint, outputInFlight, position, batch, start, validRows, logicalSegments, exactPointSafe, certified, active, exactRows, uniqueRows, refinementRows);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageEndType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageSharedCoefficientCoordinatesSafeType, class StageCachedType, class StagePrefetchedPointType, class StageOutputInFlightType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageLogicalSegmentsType, class StageExactPointSafeType, class StageCertifiedType, class StageActiveType, class StageExactRowsType, class StageUniqueRowsType, class StageRefinementRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDefinedStage3GatherStage(StageSpatialWorkspaceType& spatialWorkspace, StageEndType& end, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageCachedType& cached, StagePrefetchedPointType& prefetchedPoint, StageOutputInFlightType& outputInFlight, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe, StageCertifiedType& certified, StageActiveType& active, StageExactRowsType& exactRows, StageUniqueRowsType& uniqueRows, StageRefinementRowsType& refinementRows) {
        PipeBarrier<PIPE_V>();
    #pragma unroll
        for(uint32_t col=0;col<13;++col) {
            if constexpr((CLEAN_PRIMARY64 && sizeof(T)==2) || NATIVE_HALF_PREFIX_PRIMARY) {
                if(col<6 || (col>9 && !exactDirectionLow_))
                    Gather(refined_[col*64],C(col),offset_.template ReinterpretCast<uint32_t>(),0U,refinementRows);
            } else {
                if(col<6 || (col>9 && !DEFER_DIRECTION_LOW))Gather(refined_[col*64],C(col),offset_.template ReinterpretCast<uint32_t>(),0U,refinementRows);
            }
        }
        Gather(refined_[9*64],C(13),offset_.template ReinterpretCast<uint32_t>(),0U,refinementRows);
        PipeBarrier<PIPE_V>();
        ProcessCompareScalarStage(validRows, refinementRows);
        ProcessDefinedStage3__builtin_expectStage<OUTLINE_FULL_FLOAT_FALLBACK>(spatialWorkspace, end, spatialCoefficientDirty, spatialPruned, spatialDirtyCount, sharedCoefficientCoordinatesSafe, cached, prefetchedPoint, outputInFlight, position, batch, start, validRows, logicalSegments, exactPointSafe, certified, active, exactRows, uniqueRows);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageEndType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageSharedCoefficientCoordinatesSafeType, class StageCachedType, class StagePrefetchedPointType, class StageOutputInFlightType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageLogicalSegmentsType, class StageExactPointSafeType, class StageCertifiedType, class StageActiveType, class StageExactRowsType, class StageUniqueRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDefinedStage3__builtin_expectStage(StageSpatialWorkspaceType& spatialWorkspace, StageEndType& end, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageCachedType& cached, StagePrefetchedPointType& prefetchedPoint, StageOutputInFlightType& outputInFlight, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe, StageCertifiedType& certified, StageActiveType& active, StageExactRowsType& exactRows, StageUniqueRowsType& uniqueRows) {
        ProcessDefinedStage3__builtin_expectStage__builtin_expectStage<OUTLINE_FULL_FLOAT_FALLBACK>(spatialPruned, spatialDirtyCount, validRows, logicalSegments, certified, active, uniqueRows);
        Sqrt(tileSecond_,tileSecond_,64);PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
        ProcessDefinedStage4(spatialWorkspace, spatialPruned);
#endif
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_EXACT
        if constexpr(UsesRowsStage8()) {
            if(!exactPointSafe || !sharedCoefficientCoordinatesSafe)
                RefineWideFloatRows((GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),t_.segments,
                    uint32_t(work_.GetPhyAddr()),uint32_t(compactP_.GetPhyAddr()),
                    uint32_t(idx_.GetPhyAddr()),uint32_t(tileSecond_.GetPhyAddr()),liveRows_,sharedCoefficientCoordinatesSafe);
        }
#endif // TREE_STREAM_EXACT
        auto selectedOutput=outT_;
        if constexpr(sizeof(T)==2)Cast(selectedOutput,tileSecond_,RoundMode::CAST_RINT,64);
        else selectedOutput=tileSecond_.template ReinterpretCast<T>();
        if constexpr(OVERLAP_POINT_INPUT) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID3);WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        } else PipeBarrier<PIPE_ALL>();
        DataCopyPad(dg_[batch*t_.points+start],selectedOutput,DataCopyExtParams{1,validRows*uint32_t(sizeof(T)),0,0,0});
        DataCopyPad(ig_[batch*t_.points+start],idx_,DataCopyExtParams{1,validRows*4,0,0,0});
        if constexpr(OVERLAP_POINT_INPUT) {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID3);outputInFlight=true;
        } else PipeBarrier<PIPE_ALL>();
#if defined(DTYPE_TREE_POINTS)
        exactRows&=~uniqueRows;
        ProcessTreeGeneralExactFallbackStage(end, spatialCoefficientDirty, cached, prefetchedPoint, outputInFlight, position, batch, validRows, exactRows);
#endif
        wp_=ROWS*64+128;
        position+=validRows;


    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageValidRowsType, class StageLogicalSegmentsType, class StageCertifiedType, class StageActiveType, class StageUniqueRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDefinedStage3__builtin_expectStage__builtin_expectStage(StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageCertifiedType& certified, StageActiveType& active, StageUniqueRowsType& uniqueRows) {
        if(__builtin_expect((certified & active) != active,0)) {
            bool cachedChunks=false;
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2) {
                cachedChunks=PredicatePopulation(active&~certified)>=8;
                if(cachedChunks)BuildOverlapChunks(t_.segments);
            }
            uint64_t acceptedSparseRows=0;
            if constexpr(SAVE_PREFIX_CHUNK_PREDICATES || (CLEAN_PRIMARY64 && sizeof(T)==4))
                acceptedSparseRows=RefineSparseRows(active&~certified,validRows,logicalSegments,uniqueRows);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
            ProcessSetMaskNormStage(spatialPruned, spatialDirtyCount, certified, active, acceptedSparseRows);
#endif
            constexpr bool sparseRowDispatch=DEFER_DIRECTION_LOW || NATIVE_HALF_PREFIX_PRIMARY || (CLEAN_PRIMARY64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && (sizeof(T)==4 || sizeof(T)==2)
#ifdef DTYPE_TREE_POINTS
                && (sizeof(T)==4 || sizeof(DTYPE_TREE_POINTS)==sizeof(T))
#endif
                );
            ProcessDefinedStage3__builtin_expectStageGetSFFValueStage<OUTLINE_FULL_FLOAT_FALLBACK, sparseRowDispatch>(validRows, logicalSegments, certified, active, uniqueRows, cachedChunks, acceptedSparseRows);
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, auto sparseRowDispatch, class StageValidRowsType, class StageLogicalSegmentsType, class StageCertifiedType, class StageActiveType, class StageUniqueRowsType, class StageCachedChunksType, class StageAcceptedSparseRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDefinedStage3__builtin_expectStageGetSFFValueStage(StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageCertifiedType& certified, StageActiveType& active, StageUniqueRowsType& uniqueRows, StageCachedChunksType& cachedChunks, StageAcceptedSparseRowsType& acceptedSparseRows) {
        if constexpr(sparseRowDispatch) {
            for(uint64_t pendingRows=active&~(certified|acceptedSparseRows);pendingRows!=0;pendingRows&=pendingRows-1) {
                uint32_t row;
                if constexpr(sizeof(T)==4) row=uint32_t(GetSFFValue<1>(pendingRows));
                else row=uint32_t(__builtin_ctzll(pendingRows));
            // Only chunks whose conservative lower bound overlaps the incumbent can win.
            uint32_t chunks;
            ProcessSavedOverlapChunksStage2(logicalSegments, cachedChunks, row, chunks);
            if constexpr(CAPACITY==256 && sizeof(T)==4) {
                if(RefineFewStatic(row,chunks,uniqueRows))continue;
            }
            if constexpr(DEFER_DIRECTION_LOW)MaterializeDirectionLow();
            if (ProcessDefinedStage3__builtin_expectStageRefineOverlappingFewStage(uniqueRows, row, chunks)) continue;
            if constexpr(DEFER_DIRECTION_LOW) {
                if(RefineBoundedFullRow(row,chunks,logicalSegments))uniqueRows|=uint64_t(1)<<row;
                count_=m_;wp_=m_+32;
            } else
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            if constexpr(UsesOutlineFullFloatFallbackStage4<OUTLINE_FULL_FLOAT_FALLBACK>()) {
                if(RefineFullFloatFallback(row,chunks,logicalSegments))uniqueRows|=uint64_t(1)<<row;
            } else
#endif
            { ProcessProcessCopyStageStage(logicalSegments, uniqueRows, row, chunks); }
            }
        } else { ProcessProcessSavedOverlapChunksStage2Stage(validRows, logicalSegments, certified, uniqueRows, cachedChunks, acceptedSparseRows); }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageUniqueRowsType, class StageRowType, class StageChunksType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDefinedStage3__builtin_expectStageRefineOverlappingFewStage(StageUniqueRowsType& uniqueRows, StageRowType& row, StageChunksType& chunks) {
        if constexpr(CLEAN_PRIMARY64) {
                    if(chunks!=0 && (chunks&(chunks-1))!=0) {
                        if(RefineOverlappingFew(row,chunks,uniqueRows))return true;
                    } else if(RefineWinningFew(row,chunks,uniqueRows))return true;
                } else if constexpr(UsesCapacityStage()) {
                    if(RefineWinningFew(row,chunks,uniqueRows))return true;
                }
        return false;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageEndType, class StageSharedCoefficientCoordinatesSafeType, class StageCachedType, class StageOutputInFlightType, class StagePositionType, class StageBatchType, class StageStartType, class StageInitialInputsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDefinedStage3DefinedStage(StageEndType& end, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageCachedType& cached, StageOutputInFlightType& outputInFlight, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageInitialInputsType& initialInputs) {
        if(batch!=cached) {
            if constexpr(OVERLAP_POINT_INPUT) {
                if(outputInFlight){WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);outputInFlight=false;}
            }
            wp_=m_+32;
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            if constexpr(UsesOutlineFullFloatFallbackStage4<OUTLINE_FULL_FLOAT_FALLBACK>()) {
                LoadFullFloatSegmentsStatic((GM_ADDR)sg_.GetPhyAddr(),batch,t_.segments);
            } else
#endif
            {
                if constexpr(DEFER_DIRECTION_LOW && ROWS==64) {
                    LoadBoundedResidentSegments((GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),t_.segments,initialInputs);
                    directionLowReady_=false;
                } else if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)
                    LoadResidentSegments(batch,initialInputs);
                else LoadSegments(batch,initialInputs);
            }
            wp_=ROWS*64+128;cached=batch;
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            if constexpr(UsesRowsStage3()) {
                sharedCoefficientCoordinatesSafe=CoarseCoordinatesSafe<true>();
                tightCoefficientSafe_=Smallest(end-position,t_.points-start)>=ROWS*8 &&
                    sharedCoefficientCoordinatesSafe && CoarseDirectionSafe();
            } else
#endif
            if constexpr(GUARDED_ERROR)tightCoefficientSafe_=Smallest(end-position,t_.points-start)>=ROWS*8 &&
                CoarseCoordinatesSafe<true>() && CoarseDirectionSafe();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_EXACT_TAIL
            if constexpr(UsesRowsStage5())
                sharedCoefficientCoordinatesSafe=CoarseCoordinatesSafe<true>();
#endif // TREE_EXACT_TAIL
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            ProcessAbsStage(sharedCoefficientCoordinatesSafe);
#endif
        }
    }

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_EXACT
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ constexpr bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::UsesRowsStage8() { return (sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OUTLINE_FULL_FLOAT_FALLBACK>
    __aicore__ constexpr bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::UsesOutlineFullFloatFallbackStage4() { return (OUTLINE_FULL_FLOAT_FALLBACK && CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL);
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ constexpr bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::UsesCapacityStage() { return (CAPACITY==1024 && COEFFICIENTS==1024 && (sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
                    || sizeof(DTYPE_TREE_POINTS)==2
#endif
                ));
    }

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_EXACT_TAIL
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ constexpr bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::UsesRowsStage5() { return (sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && COMPACT_TAIL);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSpatialWorkspaceType, class StageSpatialPrunedType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDefinedStage4(StageSpatialWorkspaceType& spatialWorkspace, StageSpatialPrunedType& spatialPruned) {
        if constexpr(UsesRowsStage3()) {
            if(spatialPruned) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                const LocalTensor<int32_t> originalIndices(TPosition::VECCALC,192512U,1024U);
#else
                auto originalIndices=work_.template ReinterpretCast<int32_t>();
                SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
                GlobalTensor<int32_t> originalMap;originalMap.SetGlobalBuffer((__gm__ int32_t*)spatialWorkspace+6*4096+14*1056);
                DataCopy(originalIndices,originalMap,COEFFICIENTS);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
#endif
                Muls(tileIdx_,idx_,int32_t(4),64);PipeBarrier<PIPE_V>();
                Gather(idx_,originalIndices,tileIdx_.template ReinterpretCast<uint32_t>(),0U,liveRows_);PipeBarrier<PIPE_V>();
            }
        }
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class ControlArgument0, class ControlArgument1, class ControlArgument2>
__aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessProcessSavedOverlapChunksStage2StageControlStage40(ControlArgument0& chunks, ControlArgument1& row, ControlArgument2& uniqueRows) { return ProcessDefinedStage3__builtin_expectStageRefineOverlappingFewStage(uniqueRows,row,chunks);
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageValidRowsType, class StageLogicalSegmentsType, class StageCertifiedType, class StageUniqueRowsType, class StageCachedChunksType, class StageAcceptedSparseRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessProcessSavedOverlapChunksStage2Stage(StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageCertifiedType& certified, StageUniqueRowsType& uniqueRows, StageCachedChunksType& cachedChunks, StageAcceptedSparseRowsType& acceptedSparseRows) {

        for(uint32_t row=0;row<validRows;++row) if(!((certified|acceptedSparseRows)&(uint64_t(1)<<row))) {
            // Only chunks whose conservative lower bound overlaps the incumbent can win.
            uint32_t chunks;
            ProcessSavedOverlapChunksStage2(logicalSegments, cachedChunks, row, chunks);
            if constexpr(CAPACITY==256 && sizeof(T)==4) {
                if(RefineFewStatic(row,chunks,uniqueRows))continue;
            }
            if constexpr(DEFER_DIRECTION_LOW)MaterializeDirectionLow();
            if (ProcessProcessSavedOverlapChunksStage2StageControlStage40(chunks, row, uniqueRows)) continue;
            uint32_t first=0,second=0,hits=0;
            ProcessProcessCopyStage2Stage(logicalSegments, uniqueRows, row, chunks, first, second, hits);}

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageLogicalSegmentsType, class StageUniqueRowsType, class StageRowType, class StageChunksType, class StageFirstType, class StageSecondType, class StageHitsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessProcessCopyStage2Stage(StageLogicalSegmentsType& logicalSegments, StageUniqueRowsType& uniqueRows, StageRowType& row, StageChunksType& chunks, StageFirstType& first, StageSecondType& second, StageHitsType& hits) {
        for(uint32_t chunk=0;chunk<(logicalSegments+63)/64;++chunk)if(chunks&(1U<<chunk)) {
            if(hits==0)first=chunk;else if(hits==1)second=chunk;++hits;
        }
        const bool sparse=hits>0&&hits<=2;
        const uint32_t candidates=sparse?hits*64:COEFFICIENTS;
        auto searchColumns=columns_;const uint32_t searchCp=cp_;
        CommonVectorStage54(first, second, hits, sparse, candidates, row, logicalSegments);
        if(CertifyWideScore(candidates))uniqueRows|=uint64_t(1)<<row;
        columns_=searchColumns;cp_=searchCp;count_=m_;wp_=m_+32;
        CommitRefinedCandidate(row);

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageLogicalSegmentsType, class StageCachedChunksType, class StageRowType, class StageChunksType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessSavedOverlapChunksStage2(StageLogicalSegmentsType& logicalSegments, StageCachedChunksType& cachedChunks, StageRowType& row, StageChunksType& chunks) {
        if(cachedChunks)chunks=SavedOverlapChunks().GetValue(row);
        else {
        Muls(offset_,outputOffset_,int32_t(32),16);PipeBarrier<PIPE_V>();
        Adds(offset_,offset_,int32_t(row*4),16);
        Duplicate(W(0),__builtin_inff(),64);
        Brcb(W(1),hi_[(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Copy(W(2),W(1)[(row%8)*8],64,1,CopyRepeatParams(1,0,8,0));
        Gather(W(0),chunkLower_,offset_.template ReinterpretCast<uint32_t>(),0U,(logicalSegments+63)/64);PipeBarrier<PIPE_V>();
        Compare(smallMask_,W(0),W(2),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        chunks=smallMask_.template ReinterpretCast<uint32_t>().GetValue(0)&((1U<<((logicalSegments+63)/64))-1);
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageEndType, class StagePrefetchedPointType, class StagePositionType, class StageValidRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessSmallestStage(StageEndType& end, StagePrefetchedPointType& prefetchedPoint, StagePositionType& position, StageValidRowsType& validRows) {
        if constexpr(OVERLAP_POINT_INPUT) {
            // The packed input is dead after all coordinate gathers.
            // Refill its storage while current coordinates remain in compactP/xyz.
            const uint32_t nextPosition=position+validRows;
            if(nextPosition<end) {
                const uint32_t nextPoint=nextPosition%t_.points;
                const uint32_t nextRows=Smallest(Smallest(end-nextPosition,t_.points-nextPoint),ROWS);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID3);WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
                DataCopyPad(rawP_,pg_[nextPosition*3],DataCopyExtParams{1,nextRows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
                SetFlag<HardEvent::MTE2_V>(EVENT_ID3);prefetchedPoint=true;
            }
        }
    }

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSpatialWorkspaceType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessMinsStage(StageSpatialWorkspaceType& spatialWorkspace) {
        {
            uint32_t spatialQueryRows=liveRows_;
            auto queryOffsets=pointOffset_;
            if(spatialWorkspace && liveRows_<64U) {
                // Repeated genuine queries complete the UB tile without a GM overread.
                Mins(offset_,pointOffset_,int32_t((liveRows_-1U)*12U),64);
                PipeBarrier<PIPE_V>();
                queryOffsets=offset_;
                spatialQueryRows=64U;
            }
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Gather(compactP_[axis*64],pf_,queryOffsets.template ReinterpretCast<uint32_t>(),axis*4,spatialQueryRows);
        }
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageCertifiedType, class StageActiveType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessCopyStage3(StageCertifiedType& certified, StageActiveType& active) {
        if constexpr(SAVE_PREFIX_CHUNK_PREDICATES || CLEAN_PRIMARY64) {
            if((certified&active)!=active) {
                // Search is finished. Preserve all four predicate groups before
                // primary gathers or fallback point broadcasts reuse their storage.
        #pragma unroll
                for(uint32_t group=0;group<4;++group) {
                    auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                    Copy(work_[14*1056+group*512],predicates,64,8,CopyRepeatParams(1,1,8,8));
                }
                PipeBarrier<PIPE_V>();
            }
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StagePrefetchedPointType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageInitialInputsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDataCopyPadStage(StagePrefetchedPointType& prefetchedPoint, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageInitialInputsType& initialInputs) {
        if constexpr(OVERLAP_POINT_INPUT) {
            if(!prefetchedPoint) {
                DataCopyPad(rawP_,pg_[position*3],DataCopyExtParams{1,validRows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
                SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
            }
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);prefetchedPoint=false;
        } else {
            PipeBarrier<PIPE_ALL>();
            if(!initialInputs) {
                DataCopyPad(rawP_,pg_[(batch*t_.points+start)*3],DataCopyExtParams{1,validRows*3*uint32_t(sizeof(T)),0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageLogicalSegmentsType, class StageUniqueRowsType, class StageRowType, class StageChunksType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessProcessCopyStageStage(StageLogicalSegmentsType& logicalSegments, StageUniqueRowsType& uniqueRows, StageRowType& row, StageChunksType& chunks) {
        {
        uint32_t first=0,second=0,hits=0;
        for(uint32_t chunk=0;chunk<(logicalSegments+63)/64;++chunk)if(chunks&(1U<<chunk)) {
            if(hits==0)first=chunk;else if(hits==1)second=chunk;++hits;
        }
        const bool sparse=hits>0&&hits<=2;
        const uint32_t candidates=sparse?hits*64:COEFFICIENTS;
        auto searchColumns=columns_;const uint32_t searchCp=cp_;
        RefineFullFloatFallbackBodyCopyStage(first, second, hits, sparse);
        SharedCandidateProjection11(candidates, sparse, row, first, hits, second);
        CompareScalar(valid_,W(7),float(int32_t(logicalSegments)),CMPMODE::LT,candidates);PipeBarrier<PIPE_V>();
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            if(sparse) {
                if(hits==1)SparseCompensated<64>();
                else SparseCompensated<128>();
            } else Compensated<true>();
        } else Compensated<true>();
        SharedSparseProjection7(candidates, sparse);
        if(CertifyWideScore(candidates))uniqueRows|=uint64_t(1)<<row;
        columns_=searchColumns;cp_=searchCp;count_=m_;wp_=m_+32;
        CommitRefinedCandidate(row);
        }
    }
