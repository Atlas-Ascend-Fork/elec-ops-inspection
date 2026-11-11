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
    template<class StageSharedCoefficientCoordinatesSafeType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessAbsStage(StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe) {
        if constexpr(UsesRowsStage3()) {
            if(sharedCoefficientCoordinatesSafe) {
                // Unused anchor-column padding holds one reduced absolute maximum per axis.
        #pragma unroll
                for(uint32_t axis=0;axis<3;++axis) {
                    Abs(W(0),C(axis),COEFFICIENTS);PipeBarrier<PIPE_V>();
                    WholeReduceMax(record_,W(0),64,COEFFICIENTS/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
                    WholeReduceMax(C(axis)[CAPACITY+16],record_,COEFFICIENTS/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
                }
            }
        }
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageSpatialPointCoordinatesSafeType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageCorePointSafeType, class StageSharedCoefficientCoordinatesSafeType, class StageLogicalSegmentsType, class StageExactPointSafeType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDefinedStage2(StageSpatialWorkspaceType& spatialWorkspace, StageSpatialPointCoordinatesSafeType& spatialPointCoordinatesSafe, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageCorePointSafeType& corePointSafe, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe) {
        if constexpr(GUARDED_ERROR) { ProcessDefinedStage5<OUTLINE_FULL_FLOAT_FALLBACK>(spatialWorkspace, spatialPointCoordinatesSafe, spatialCoefficientDirty, spatialPruned, spatialDirtyCount, corePointSafe, sharedCoefficientCoordinatesSafe, logicalSegments, exactPointSafe); }
        else if constexpr(NATIVE_HALF_PREFIX_PRIMARY && ROWS==64)count_=SearchHalfPrefixOutlined(liveRows_,t_.segments);
        else Search();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageSpatialPointCoordinatesSafeType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageCorePointSafeType, class StageSharedCoefficientCoordinatesSafeType, class StageLogicalSegmentsType, class StageExactPointSafeType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDefinedStage5(StageSpatialWorkspaceType& spatialWorkspace, StageSpatialPointCoordinatesSafeType& spatialPointCoordinatesSafe, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageCorePointSafeType& corePointSafe, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe) {
        {
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2
#ifdef DTYPE_TREE_POINTS
                && sizeof(DTYPE_TREE_POINTS)==2
#endif
            ) {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
                if constexpr(UsesRowsStage())
                    count_=SearchGuardedHalfTypeCertifiedRows(liveRows_,t_.segments,tightCoefficientSafe_);
                else
#endif
                    SearchGuardedHalfCachedRows();
            }
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
            else if constexpr(UsesRowsStage3()) { ProcessDefinedStage<OUTLINE_FULL_FLOAT_FALLBACK>(spatialWorkspace, spatialPointCoordinatesSafe, spatialCoefficientDirty, spatialPruned, spatialDirtyCount, corePointSafe, sharedCoefficientCoordinatesSafe, logicalSegments, exactPointSafe); }
#endif
            else SearchGuardedError();
        }
    }

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageSpatialPointCoordinatesSafeType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageCorePointSafeType, class StageSharedCoefficientCoordinatesSafeType, class StageLogicalSegmentsType, class StageExactPointSafeType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDefinedStage(StageSpatialWorkspaceType& spatialWorkspace, StageSpatialPointCoordinatesSafeType& spatialPointCoordinatesSafe, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageCorePointSafeType& corePointSafe, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe) {
        {
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)) // TREE_HALF_COORD_CERT_3
            bool spatialPointSafe=false;
            if(spatialWorkspace && sharedCoefficientCoordinatesSafe && (liveRows_==64 || (OUTLINE_FULL_FLOAT_FALLBACK && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)))) {
                if(spatialPointCoordinatesSafe) {
                    // The launch's producer certificate already proves every input coordinate.
                    // PruneSpatialStaticState overwrites its scratch and SearchSpatialStaticState
                    // produces the actual row predicates before their first consumer.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
                    Abs(W(1),compactP_,192);PipeBarrier<PIPE_V>();
                    Duplicate(record_.template ReinterpretCast<uint16_t>(),uint16_t(65535),12);
                    CompareScalar(mask_,W(1),0.0F,CMPMODE::EQ,192);PipeBarrier<PIPE_V>();
                    Duplicate(valid_.template ReinterpretCast<uint16_t>(),uint16_t(65535),12);PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
#endif
                    spatialPointSafe=true;
                } else spatialPointSafe=CoarseCoordinatesSafe<false>();
            }
            if(spatialPointSafe) {
#else // TREE_HALF_COORD_CERT_3
            if(spatialWorkspace && sharedCoefficientCoordinatesSafe &&
               (liveRows_==64 || OUTLINE_FULL_FLOAT_FALLBACK) && CoarseCoordinatesSafe<false>()) {
#endif // TREE_HALF_COORD_CERT_3
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_EXACT
                exactPointSafe=true;
#endif // TREE_STREAM_EXACT
                const uint32_t limit=PruneSpatialStaticState(spatialWorkspace,t_.segments,spatialPruned,logicalSegments);
                spatialCoefficientDirty=spatialPruned;
        #if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
                spatialDirtyCount=spatialPruned?limit:COEFFICIENTS;
#endif
                // The helper consumes the map event before its first
                // overlapping W5 write (or before an empty resume returns).
                count_=SearchSpatialStaticState(0,limit,false,tightCoefficientSafe_);
            } else {
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_EXACT
                exactPointSafe=
#endif // TREE_STREAM_EXACT
                SearchGuardedCoreCertifiedRows(corePointSafe,sharedCoefficientCoordinatesSafe);
            }
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSpatialWorkspaceType, class StageBeginType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialCoefficientsType, class StagePositionType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessDataCopyStage(StageSpatialWorkspaceType& spatialWorkspace, StageBeginType& begin, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialCoefficientsType& spatialCoefficients, StagePositionType& position) {
        if constexpr(UsesRowsStage3()) {
            if(spatialWorkspace) {
                if(position==begin) {
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                    DataCopy(spatialCoefficients,columns_,14*cp_);
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID2);WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
                } else if(spatialCoefficientDirty) {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
                    DataCopy(columns_,spatialCoefficients,14*cp_);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
                }
                spatialPruned=false;
            }
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageCertifiedType, class StageActiveType, class StageAcceptedSparseRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessSetMaskNormStage(StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageCertifiedType& certified, StageActiveType& active, StageAcceptedSparseRowsType& acceptedSparseRows) {
        if constexpr(UsesRowsStage3()) {
            if(spatialPruned && spatialDirtyCount<COEFFICIENTS && ((certified|acceptedSparseRows)&active)!=active) {
                // Selected sparse refiners read only searched packets. Any remaining
                // fallback may scan the whole cache, so finish its original duplicate tail.
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                for(uint32_t col=0;col<14;++col)
                    Brcb(work_[col*64U],C(col)[spatialDirtyCount-8U],1,BrcbRepeatParams(1,8));
                PipeBarrier<PIPE_V>();
                for(uint32_t col=0;col<14;++col)
                    Copy(C(col)[spatialDirtyCount],work_[col*64U+56U],64,(COEFFICIENTS-spatialDirtyCount)/64U,CopyRepeatParams(1,0,8,0));
                PipeBarrier<PIPE_V>();
                spatialDirtyCount=COEFFICIENTS;
            }
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageEndType, class StageSpatialCoefficientDirtyType, class StageCachedType, class StagePrefetchedPointType, class StageOutputInFlightType, class StagePositionType, class StageBatchType, class StageValidRowsType, class StageExactRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessTreeGeneralExactFallbackStage(StageEndType& end, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageCachedType& cached, StagePrefetchedPointType& prefetchedPoint, StageOutputInFlightType& outputInFlight, StagePositionType& position, StageBatchType& batch, StageValidRowsType& validRows, StageExactRowsType& exactRows) {
        if(exactRows) {
            if constexpr(OVERLAP_POINT_INPUT) {
                if(prefetchedPoint){WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);prefetchedPoint=false;}
                if(outputInFlight){WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);outputInFlight=false;}
            }
            PipeBarrier<PIPE_ALL>();
            for(uint32_t row=0;row<validRows;++row)if(exactRows&(uint64_t(1)<<row)) {
                const uint32_t pos=position+row;
                TreeGeneralExactFallback<T>((GM_ADDR)pg_[pos*3].GetPhyAddr(),
                    (GM_ADDR)sg_[batch*t_.segments*6].GetPhyAddr(),
                    (GM_ADDR)dg_[pos].GetPhyAddr(),(GM_ADDR)ig_[pos].GetPhyAddr(),t_.segments);
            }
            if(position+validRows<end) {
                if constexpr(DEFER_DIRECTION_LOW)RestartBoundedInlineAfterExact();
                else {
                    const TreeLineMinDistanceTilingData t=t_;
                    Init((GM_ADDR)pg_[0].GetPhyAddr(),(GM_ADDR)sg_[0].GetPhyAddr(),
                         (GM_ADDR)dg_[0].GetPhyAddr(),(GM_ADDR)ig_[0].GetPhyAddr(),t);
                }
                cached=0xffffffffU;
                spatialCoefficientDirty=false;
            }
        }
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageValidRowsType, class StageRefinementRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ProcessCompareScalarStage(StageValidRowsType& validRows, StageRefinementRowsType& refinementRows) {
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            CompareScalar(valid_,segmentIndices_,float(int32_t(validRows)),CMPMODE::LT,64);PipeBarrier<PIPE_V>();
            if(validRows<=16)PrimaryCompensated16();
            else if(validRows<=32)PrimaryCompensated32();
            else PrimaryCompensatedFull();
            Copy(tileSecond_,work_,64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            count_=m_;wp_=m_+32;
        } else if constexpr(DEFER_CHUNK_CERTIFICATE && (sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
            || sizeof(DTYPE_TREE_POINTS)==2
#endif
        )) {
            if constexpr(!CLEAN_PRIMARY64) {
                Duplicate(valid_.template ReinterpretCast<uint16_t>(),uint16_t(65535),128);PipeBarrier<PIPE_V>();
            }
            if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2) {
                if(exactDirectionLow_)PrimaryCompensatedExactDirection();
                else PrimaryCompensatedFull();
            } else PrimaryCompensatedFull();
            Copy(tileSecond_,work_,64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            count_=m_;wp_=m_+32;
        } else if constexpr(NATIVE_HALF_PREFIX_PRIMARY) {
            Duplicate(valid_.template ReinterpretCast<uint16_t>(),uint16_t(65535),128);PipeBarrier<PIPE_V>();
            if(exactDirectionLow_)PrimaryCompensatedExactDirection();
            else PrimaryCompensatedFull();
            Copy(tileSecond_,work_,64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            count_=m_;wp_=m_+32;
        } else {
        auto originalColumns=columns_,originalPoints=xyz_;
        const uint32_t originalCp=cp_,originalPp=pp_;
        columns_=refined_;cp_=64;xyz_=compactP_;pp_=64;count_=refinementRows;wp_=96;
        if constexpr(!NATIVE_FLOAT_PREFIX_SPARSE) {
            Duplicate(valid_.template ReinterpretCast<uint16_t>(),uint16_t(65535),128);PipeBarrier<PIPE_V>();
        }
        Compensated();
        Copy(tileSecond_,W(0),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        columns_=originalColumns;cp_=originalCp;xyz_=originalPoints;pp_=originalPp;count_=m_;wp_=m_+32;
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::Smallest(uint32_t a,uint32_t b){return a<b?a:b;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::C(uint32_t k){
        const uint32_t offset=(cp_==m_+32 && k>=6 && k<=8)?6*cp_+(k-6)*m_:k*cp_;
        return columns_[offset];
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::W(uint32_t k){
        constexpr uint32_t bankOffset[13]={64,0,0,64,96,32,96,0,0,0,0,0,0};
        constexpr bool fullFloatCache=sizeof(T)==4 && CAPACITY==1024 && COEFFICIENTS==1024;
        constexpr bool fullHalfCache=sizeof(T)==2 && CAPACITY==1024 && COEFFICIENTS==1024
#ifdef DTYPE_TREE_POINTS
            && sizeof(DTYPE_TREE_POINTS)==2
#endif
            ;
        // Native full caches use the same APx phase candidate within existing padding.
        const uint32_t phase=(fullFloatCache || fullHalfCache) && k==0?32U:bankOffset[k];
        return work_[k*wp_+(wp_==ROWS*64+128?phase:0)];
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::P(uint32_t k){return xyz_[k*pp_];
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool FIXED_ARENA>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchC(uint32_t k) {
        if constexpr(FIXED_ARENA ? (CAPACITY==256 && sizeof(T)==4) : NATIVE_FLOAT_PREFIX_SPARSE) {
            constexpr uint32_t pitch=CAPACITY+32;
            const uint32_t offset=k>=6 && k<=8?6*pitch+(k-6)*CAPACITY:k*pitch;
            return LocalTensor<float>(TPosition::VECCALC,offset*4,k>=6 && k<=8?CAPACITY:pitch);
        } else return C(k);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool FIXED_ARENA>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchW(uint32_t k) {
        if constexpr(FIXED_ARENA ? (CAPACITY==256 && sizeof(T)==4) : NATIVE_FLOAT_PREFIX_SPARSE) {
            constexpr uint32_t pitch=ROWS*64+128;
            constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
            constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch;
            constexpr uint32_t banks[13]={64,0,0,64,96,32,96,0,0,0,0,0,0};
            const uint32_t offset=k*pitch+banks[k];
            return LocalTensor<float>(TPosition::VECCALC,(base+offset)*4,6*pitch-offset);
        } else return W(k);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool FIXED_ARENA>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchP(uint32_t k) {
        if constexpr(FIXED_ARENA ? (CAPACITY==256 && sizeof(T)==4) : NATIVE_FLOAT_PREFIX_SPARSE) {
            constexpr uint32_t pitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
            constexpr uint32_t base=14*(CAPACITY+32);
            return LocalTensor<float>(TPosition::VECCALC,(base+k*pitch)*4,pitch);
        } else return P(k);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::InitializeGroupedSearchState(
        LocalTensor<float> initialDistances,const LocalTensor<int32_t>& indices,uint32_t indexRepeats) {
        auto initialIndices=indices.template ReinterpretCast<float>();
        initialDistances.SetSize(3*64);initialIndices.SetSize(3*64);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Duplicate<float,false>(initialDistances,__builtin_inff(),uint64_t(64),3,1,8);
        Duplicate<float,false>(initialIndices,0.0F,uint64_t(64),indexRepeats,1,8);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool FIXED_ARENA, bool DIRECT_PREDICATES>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::MergeSearchMinimum(uint32_t j) {
        const BinaryRepeatParams same{1,1,1,8,8,8};
        const UnaryRepeatParams unary{1,1,8,8};
        if constexpr(FIXED_ARENA && !DIRECT_PREDICATES) {
            Copy(chunkLower_[512+j*2],SearchW<true>(0),64,1,CopyRepeatParams(1,1,8,8));
            Copy(chunkLower_[512+j*2+64],SearchW<true>(1),64,1,CopyRepeatParams(1,1,8,8));
        }

            SharedMergeMinimum15(same, j, unary);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,(DIRECT_PREDICATES?chunkLower_[512+j*2]:SearchW<FIXED_ARENA>(0)),cert_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,(DIRECT_PREDICATES?chunkLower_[512+j*2+64]:SearchW<FIXED_ARENA>(1)),certHigh_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool FIXED_ARENA>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CompactSearchTail(uint32_t j,uint32_t tail) {
        // Tail segments use complete point vectors instead of padded segment rows.
        #pragma unroll
        for(uint32_t col=0;col<10;++col)if(col<3||col>=6)
            Brcb(refined_[col*64],SearchC<FIXED_ARENA>(col)[j],1,BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        const BinaryRepeatParams same{1,1,1,8,8,8},coefficient{1,1,0,8,8,1},pointCoefficient{1,1,0,8,0,1};
        const UnaryRepeatParams unary{1,1,8,8};
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Sub<float,false>(SearchW<FIXED_ARENA>(axis),compactP_[axis*64],refined_[axis*64],uint64_t(64),tail,pointCoefficient);
        PipeBarrier<PIPE_V>();
        Mul<float,false>(SearchW<FIXED_ARENA>(3),SearchW<FIXED_ARENA>(0),refined_[6*64],uint64_t(64),tail,coefficient);
        Mul<float,false>(SearchW<FIXED_ARENA>(4),SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(0),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(SearchW<FIXED_ARENA>(3),SearchW<FIXED_ARENA>(1),refined_[7*64],uint64_t(64),tail,coefficient);
        MulAddDst<float,float,false>(SearchW<FIXED_ARENA>(4),SearchW<FIXED_ARENA>(1),SearchW<FIXED_ARENA>(1),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(SearchW<FIXED_ARENA>(3),SearchW<FIXED_ARENA>(2),refined_[8*64],uint64_t(64),tail,coefficient);
        MulAddDst<float,float,false>(SearchW<FIXED_ARENA>(4),SearchW<FIXED_ARENA>(2),SearchW<FIXED_ARENA>(2),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        Maxs<float,false>(SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(3),0.0F,uint64_t(64),tail,unary);PipeBarrier<PIPE_V>();
        Min<float,false>(SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(0),refined_[9*64],uint64_t(64),tail,coefficient);PipeBarrier<PIPE_V>();
        Muls<float,false>(SearchW<FIXED_ARENA>(2),SearchW<FIXED_ARENA>(3),-2.0F,uint64_t(64),tail,unary);
        Muls<float,false>(SearchW<FIXED_ARENA>(5),SearchW<FIXED_ARENA>(4),1.52587890625e-5F,uint64_t(64),tail,unary);PipeBarrier<PIPE_V>();
        Add<float,false>(SearchW<FIXED_ARENA>(1),SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(2),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(SearchW<FIXED_ARENA>(4),SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(1),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        Sub<float,false>(SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(4),SearchW<FIXED_ARENA>(5),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        CompactSearchTailCopyStage<FIXED_ARENA>(j, tail, same);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool FIXED_ARENA, class StageJType, class StageTailType, class StageSameType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CompactSearchTailCopyStage(StageJType& j, StageTailType& tail, StageSameType& same) {
        if constexpr(SINGLE_TAIL) {
            Copy(tileLo_,SearchW<FIXED_ARENA>(0),64,1,CopyRepeatParams(1,1,8,8));
            Muls(tileHi_,SearchW<FIXED_ARENA>(5),2.125F,64);Duplicate(tileIdx_,int32_t(0),64);PipeBarrier<PIPE_V>();
            Add(tileHi_,tileLo_,tileHi_,64);
            Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            Duplicate(SearchW<FIXED_ARENA>(0).template ReinterpretCast<int32_t>(),int32_t(1),64);
            Duplicate(SearchW<FIXED_ARENA>(1),0.0F,64);PipeBarrier<PIPE_V>();
        } else {
            Duplicate(tileLo_,__builtin_inff(),64);Duplicate(tileHi_,__builtin_inff(),64);
            Duplicate(tileIdx_,int32_t(0),64);PipeBarrier<PIPE_V>();
            for(uint32_t r=0;r<tail;++r) {
                Compare<float>(SearchW<FIXED_ARENA>(0)[r*64],tileLo_,CMPMODE::LT,uint64_t(64),same);
                Duplicate(offset_,int32_t(r),64);PipeBarrier<PIPE_V>();
                Min(tileLo_,SearchW<FIXED_ARENA>(0)[r*64],tileLo_,64);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileHi_,SearchW<FIXED_ARENA>(5)[r*64],tileHi_,1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileIdx_.template ReinterpretCast<float>(),offset_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            }
            Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
            Add(tileHi_,tileLo_,tileHi_,64);
            Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            Duplicate(SearchW<FIXED_ARENA>(2),0.0F,64);PipeBarrier<PIPE_V>();
            for(uint32_t r=0;r<tail;++r) {
                Compare(mask_,SearchW<FIXED_ARENA>(0)[r*64],tileHi_,CMPMODE::LE,64);
                Duplicate(SearchW<FIXED_ARENA>(3).template ReinterpretCast<int32_t>(),int32_t(1U<<r),64);PipeBarrier<PIPE_V>();
                Select(SearchW<FIXED_ARENA>(3),mask_,SearchW<FIXED_ARENA>(3),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
                Or(SearchW<FIXED_ARENA>(2).template ReinterpretCast<uint16_t>(),SearchW<FIXED_ARENA>(2).template ReinterpretCast<uint16_t>(),SearchW<FIXED_ARENA>(3).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            }
            Copy(SearchW<FIXED_ARENA>(0),SearchW<FIXED_ARENA>(2),64,1,CopyRepeatParams(1,1,8,8));
            Duplicate(SearchW<FIXED_ARENA>(1),0.0F,64);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool DIRECT_PREDICATES, uint32_t FIXED_SEARCH_ROWS>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchStaticPolicy() {
        const uint32_t liveRows_=FIXED_SEARCH_ROWS?FIXED_SEARCH_ROWS:this->liveRows_;
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        constexpr bool explicitStageBarrier = !DIRECT_PREDICATES || SINGLE_TAIL;
#else
        constexpr bool explicitStageBarrier = true;
#endif
        if constexpr(sizeof(T)==4 && (CAPACITY==1024 || CAPACITY==256)) {
            InitializeGroupedSearchState(lo_,idx_,3);
        } else {
            Duplicate(lo_,__builtin_inff(),64);Duplicate(second_,__builtin_inff(),64);
            Duplicate(hi_,__builtin_inff(),64);Duplicate(idx_,int32_t(0),64);
            Duplicate(cert_,0.0F,64);Duplicate(certHigh_,0.0F,64);
        }
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        count_=(liveRows_*64);
        // The one-tail 192-coefficient resource has two complete packets and one tail.
        const uint32_t segmentCount=(CAPACITY==256 && COEFFICIENTS==192 && SINGLE_TAIL)?129U:t_.segments;
                SearchStaticPolicyCompactSearchTailStage<DIRECT_PREDICATES, explicitStageBarrier>(liveRows_, same, segment, pointSegment, unary, segmentCount);
        // Refinement gathers a full register, including inactive point lanes.
        if(liveRows_<64) {
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),~((uint64_t(1)<<liveRows_)-1));
            Duplicate<int32_t,false>(idx_,int32_t(0),uint64_t(64),1,1,8);
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
        PipeBarrier<PIPE_V>();
        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        SearchStaticPolicyDuplicateStage();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool DIRECT_PREDICATES, auto explicitStageBarrier, class StageLiveRows_Type, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType, class StageSegmentCountType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchStaticPolicyCompactSearchTailStage(StageLiveRows_Type& liveRows_, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary, StageSegmentCountType& segmentCount) {
#pragma unroll

        for(uint32_t j=0;j<segmentCount;j+=64) {
        if constexpr(COMPACT_TAIL) {
            if(segmentCount-j<=8) {
                CompactSearchTail<true>(j,segmentCount-j);MergeSearchMinimum<true>(j);continue;
            }
        }
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(SearchW<true>(axis),SearchP<true>(axis),SearchC<true>(axis)[j],uint64_t(64),liveRows_,pointSegment);
        if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
        Mul<float,false>(SearchW<true>(3),SearchW<true>(0),SearchC<true>(6)[j],uint64_t(64),liveRows_,segment);
        Mul<float,false>(SearchW<true>(4),SearchW<true>(0),SearchW<true>(0),uint64_t(64),liveRows_,same);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(SearchW<true>(3),SearchW<true>(1),SearchC<true>(7)[j],uint64_t(64),liveRows_,segment);
        MulAddDst<float,float,false>(SearchW<true>(4),SearchW<true>(1),SearchW<true>(1),uint64_t(64),liveRows_,same);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(SearchW<true>(3),SearchW<true>(2),SearchC<true>(8)[j],uint64_t(64),liveRows_,segment);
        MulAddDst<float,float,false>(SearchW<true>(4),SearchW<true>(2),SearchW<true>(2),uint64_t(64),liveRows_,same);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
        Maxs<float,false>(SearchW<true>(0),SearchW<true>(3),0.0F,uint64_t(64),liveRows_,unary);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
        Min<float,false>(SearchW<true>(0),SearchW<true>(0),SearchC<true>(9)[j],uint64_t(64),liveRows_,segment);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
        Muls<float,false>(SearchW<true>(2),SearchW<true>(3),-2.0F,uint64_t(64),liveRows_,unary);
        Muls<float,false>(SearchW<true>(5),SearchW<true>(4),1.52587890625e-5F,uint64_t(64),liveRows_,unary);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
        Add<float,false>(SearchW<true>(1),SearchW<true>(0),SearchW<true>(2),uint64_t(64),liveRows_,same);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(SearchW<true>(4),SearchW<true>(0),SearchW<true>(1),uint64_t(64),liveRows_,same);if constexpr(explicitStageBarrier)PipeBarrier<PIPE_V>();
        Sub<float,false>(SearchW<true>(0),SearchW<true>(4),SearchW<true>(5),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
        if(j+64>segmentCount) {
            CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(segmentCount-j)),CMPMODE::LT,uint64_t(64),liveRows_,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
            Select(SearchW<true>(0),valid_,SearchW<true>(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(liveRows_*64));PipeBarrier<PIPE_V>();
        }
        WholeReduceMin(record_,SearchW<true>(0),64,liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
        GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
        GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
        Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
        Muls(offset_,tileIdx_,int32_t(4),liveRows_);PipeBarrier<PIPE_V>();
        Add(offset_,offset_,rowBase_,liveRows_);PipeBarrier<PIPE_V>();
        Gather(tileHi_,SearchW<true>(5),offset_.template ReinterpretCast<uint32_t>(),0U,liveRows_);PipeBarrier<PIPE_V>();
        // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
        Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
        Add(tileHi_,tileLo_,tileHi_,64);PipeBarrier<PIPE_V>();
        Brcb(SearchW<true>(1),tileHi_,(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        SetMaskNorm();SearchStaticPolicyCompareStage<DIRECT_PREDICATES>(liveRows_, same, unary, j);}
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool DIRECT_PREDICATES, class StageLiveRows_Type, class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchStaticPolicyCompareStage(StageLiveRows_Type& liveRows_, StageSameType& same, StageUnaryType& unary, StageJType& j) {
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Compare<float,uint8_t,false>(mask_,SearchW<true>(0),SearchW<true>(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
        if constexpr(DIRECT_PREDICATES) {
            GatherMask(chunkLower_[512+j*2],mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(chunkLower_[512+j*2+64],mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
        } else {
            GatherMask(SearchW<true>(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(SearchW<true>(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
        }
        SearchStaticPolicyMergeSearchMinimumStage<DIRECT_PREDICATES>(same, unary, j);

    }
