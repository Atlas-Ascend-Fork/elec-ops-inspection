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
// Member declarations included by KernelTreeBatch.


    static constexpr bool DEFER_DIRECTION_LOW=sizeof(T)==4 && CAPACITY==256
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ;
    static constexpr bool NATIVE_HALF_PREFIX_PRIMARY=sizeof(T)==2 && CAPACITY==1024 && COEFFICIENTS==576
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==2
#else
        && false
#endif
        ;
protected:
    static constexpr bool NATIVE_FLOAT_PREFIX_SPARSE=sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==576
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        ;
#else
        && false;
#endif
private:
    static constexpr bool SAVE_PREFIX_CHUNK_PREDICATES=NATIVE_HALF_PREFIX_PRIMARY || NATIVE_FLOAT_PREFIX_SPARSE;
    static constexpr bool PREFIX_NORMALIZATION=sizeof(T)==4 && ((CAPACITY==1024 && COEFFICIENTS==576) || CAPACITY==256);
    static constexpr bool OVERLAP_POINT_INPUT=NATIVE_HALF_PREFIX_PRIMARY || (CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL);
    static constexpr bool DEFER_CHUNK_CERTIFICATE=CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL;
    static constexpr bool NORM_BEFORE_DOT=DEFER_CHUNK_CERTIFICATE && sizeof(T)==4;
    static constexpr bool CLEAN_PRIMARY64=DEFER_CHUNK_CERTIFICATE
#ifdef DTYPE_TREE_POINTS
        && (sizeof(T)==4 || sizeof(T)==sizeof(DTYPE_TREE_POINTS))
#else
        && sizeof(T)==4
#endif
        ;
    static constexpr bool GUARDED_ERROR=CLEAN_PRIMARY64;
    static constexpr bool FULL_FIXED_MASK=NATIVE_HALF_PREFIX_PRIMARY || (DEFER_CHUNK_CERTIFICATE && (sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
        || sizeof(DTYPE_TREE_POINTS)==2
#endif
        ));
public:
    __aicore__ inline void Init(GM_ADDR p,GM_ADDR s,GM_ADDR d,GM_ADDR i,const TreeLineMinDistanceTilingData& t);
    __aicore__ __attribute__((always_inline)) inline void InitCastStage();
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    __aicore__ static constexpr bool UsesRowsStage();
#endif

    __aicore__ static constexpr bool UsesNativeHalfPrefixPrimaryStage();


    __aicore__ __attribute__((always_inline)) inline void InitDuplicateStage();

    __aicore__ __attribute__((always_inline)) inline void InitShiftLeftStage();

    __aicore__ __attribute__((always_inline)) inline void InitCStage();
    __aicore__ static constexpr bool UsesRowsStage2();


#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    __aicore__ inline bool SpatialSortPointCoordinatesSafe(uint32_t begin,uint32_t end);
#endif
    // Commit one exact fallback result while preserving all other vector rows.
    __aicore__ __attribute__((always_inline)) inline void CommitRefinedCandidate(uint32_t row);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    __aicore__ __attribute__((always_inline)) inline bool RefineFullFloatFallbackBody(uint32_t row,uint32_t chunks);
    template<class StageFirstType, class StageSecondType, class StageHitsType, class StageSparseType>
    __aicore__ __attribute__((always_inline)) inline void RefineFullFloatFallbackBodyCopyStage(StageFirstType& first, StageSecondType& second, StageHitsType& hits, StageSparseType& sparse);

    // Shared fixed UB layout; inlined into each existing static entry.
    __aicore__ __attribute__((always_inline)) static inline void InitFullFloatFrame(
        KernelTreeBatch& frame,uint32_t segmentCount);
    __aicore__ __attribute__((noinline)) static bool RefineFullFloatFallback(uint32_t row,uint32_t chunks,uint32_t segmentCount);
#endif
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    __aicore__ __attribute__((noinline)) static void LoadFullFloatSegmentsStatic(GM_ADDR segments,uint32_t batch,uint32_t segmentCount);
#endif
    __aicore__ __attribute__((noinline,aligned(128))) static bool RefineBoundedFullRow(
        uint32_t row,uint32_t chunks,uint32_t logicalSegments);
    __aicore__ __attribute__((always_inline)) inline bool RefineBoundedFullRowBody(
        uint32_t row,uint32_t chunks,uint32_t logicalSegments);


    __aicore__ __attribute__((noinline,aligned(128))) void RestartBoundedAfterExact();
    __aicore__ __attribute__((always_inline)) inline void RestartBoundedInlineAfterExact();
    template<bool OUTLINE_FULL_FLOAT_FALLBACK=false>
    __aicore__ inline void Process(uint32_t rangeBegin=0,uint32_t rangeEnd=0,GM_ADDR spatialWorkspace=nullptr,bool spatialPointCoordinatesSafe=false);
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))

    __aicore__ static constexpr bool UsesRowsStage3();
#endif

    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageSpatialPointCoordinatesSafeType, class StageBeginType, class StageEndType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageSpatialCoefficientsType, class StageCorePointSafeType, class StageSharedCoefficientCoordinatesSafeType, class StageCachedType, class StagePrefetchedPointType, class StageOutputInFlightType>
    __aicore__ __attribute__((always_inline)) inline void ProcessSmallestStage2(StageSpatialWorkspaceType& spatialWorkspace, StageSpatialPointCoordinatesSafeType& spatialPointCoordinatesSafe, StageBeginType& begin, StageEndType& end, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageSpatialCoefficientsType& spatialCoefficients, StageCorePointSafeType& corePointSafe, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageCachedType& cached, StagePrefetchedPointType& prefetchedPoint, StageOutputInFlightType& outputInFlight);
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageSpatialPointCoordinatesSafeType, class StageBeginType, class StageEndType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageSpatialCoefficientsType, class StageCorePointSafeType, class StageSharedCoefficientCoordinatesSafeType, class StageCachedType, class StagePrefetchedPointType, class StageOutputInFlightType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageLogicalSegmentsType, class StageExactPointSafeType, class StageInitialInputsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDefinedStage3(StageSpatialWorkspaceType& spatialWorkspace, StageSpatialPointCoordinatesSafeType& spatialPointCoordinatesSafe, StageBeginType& begin, StageEndType& end, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageSpatialCoefficientsType& spatialCoefficients, StageCorePointSafeType& corePointSafe, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageCachedType& cached, StagePrefetchedPointType& prefetchedPoint, StageOutputInFlightType& outputInFlight, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe, StageInitialInputsType& initialInputs);
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageEndType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageSharedCoefficientCoordinatesSafeType, class StageCachedType, class StagePrefetchedPointType, class StageOutputInFlightType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageLogicalSegmentsType, class StageExactPointSafeType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDefinedStage3UsesRowsStage7Stage(StageSpatialWorkspaceType& spatialWorkspace, StageEndType& end, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageCachedType& cached, StagePrefetchedPointType& prefetchedPoint, StageOutputInFlightType& outputInFlight, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe);
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageEndType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageSharedCoefficientCoordinatesSafeType, class StageCachedType, class StagePrefetchedPointType, class StageOutputInFlightType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageLogicalSegmentsType, class StageExactPointSafeType, class StageCertifiedType, class StageActiveType, class StageExactRowsType, class StageUniqueRowsType, class StageRefinementRowsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDefinedStage3GatherStage(StageSpatialWorkspaceType& spatialWorkspace, StageEndType& end, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageCachedType& cached, StagePrefetchedPointType& prefetchedPoint, StageOutputInFlightType& outputInFlight, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe, StageCertifiedType& certified, StageActiveType& active, StageExactRowsType& exactRows, StageUniqueRowsType& uniqueRows, StageRefinementRowsType& refinementRows);
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageEndType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageSharedCoefficientCoordinatesSafeType, class StageCachedType, class StagePrefetchedPointType, class StageOutputInFlightType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageLogicalSegmentsType, class StageExactPointSafeType, class StageCertifiedType, class StageActiveType, class StageExactRowsType, class StageUniqueRowsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDefinedStage3__builtin_expectStage(StageSpatialWorkspaceType& spatialWorkspace, StageEndType& end, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageCachedType& cached, StagePrefetchedPointType& prefetchedPoint, StageOutputInFlightType& outputInFlight, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe, StageCertifiedType& certified, StageActiveType& active, StageExactRowsType& exactRows, StageUniqueRowsType& uniqueRows);
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageValidRowsType, class StageLogicalSegmentsType, class StageCertifiedType, class StageActiveType, class StageUniqueRowsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDefinedStage3__builtin_expectStage__builtin_expectStage(StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageCertifiedType& certified, StageActiveType& active, StageUniqueRowsType& uniqueRows);

    template<bool OUTLINE_FULL_FLOAT_FALLBACK, auto sparseRowDispatch, class StageValidRowsType, class StageLogicalSegmentsType, class StageCertifiedType, class StageActiveType, class StageUniqueRowsType, class StageCachedChunksType, class StageAcceptedSparseRowsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDefinedStage3__builtin_expectStageGetSFFValueStage(StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageCertifiedType& certified, StageActiveType& active, StageUniqueRowsType& uniqueRows, StageCachedChunksType& cachedChunks, StageAcceptedSparseRowsType& acceptedSparseRows);

    template<class StageUniqueRowsType, class StageRowType, class StageChunksType>
    __aicore__ __attribute__((always_inline)) inline bool ProcessDefinedStage3__builtin_expectStageRefineOverlappingFewStage(StageUniqueRowsType& uniqueRows, StageRowType& row, StageChunksType& chunks);


    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageEndType, class StageSharedCoefficientCoordinatesSafeType, class StageCachedType, class StageOutputInFlightType, class StagePositionType, class StageBatchType, class StageStartType, class StageInitialInputsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDefinedStage3DefinedStage(StageEndType& end, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageCachedType& cached, StageOutputInFlightType& outputInFlight, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageInitialInputsType& initialInputs);

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_EXACT

    __aicore__ static constexpr bool UsesRowsStage8();
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    template<bool OUTLINE_FULL_FLOAT_FALLBACK>
    __aicore__ static constexpr bool UsesOutlineFullFloatFallbackStage4();
#endif


    __aicore__ static constexpr bool UsesCapacityStage();


#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))


#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)

#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)

#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_EXACT_TAIL

    __aicore__ static constexpr bool UsesRowsStage5();
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))


#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))

#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    template<class StageSpatialWorkspaceType, class StageSpatialPrunedType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDefinedStage4(StageSpatialWorkspaceType& spatialWorkspace, StageSpatialPrunedType& spatialPruned);


#endif


    template<class StageValidRowsType, class StageLogicalSegmentsType, class StageCertifiedType, class StageUniqueRowsType, class StageCachedChunksType, class StageAcceptedSparseRowsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessProcessSavedOverlapChunksStage2Stage(StageValidRowsType& validRows, StageLogicalSegmentsType& logicalSegments, StageCertifiedType& certified, StageUniqueRowsType& uniqueRows, StageCachedChunksType& cachedChunks, StageAcceptedSparseRowsType& acceptedSparseRows);


    template<class StageLogicalSegmentsType, class StageUniqueRowsType, class StageRowType, class StageChunksType, class StageFirstType, class StageSecondType, class StageHitsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessProcessCopyStage2Stage(StageLogicalSegmentsType& logicalSegments, StageUniqueRowsType& uniqueRows, StageRowType& row, StageChunksType& chunks, StageFirstType& first, StageSecondType& second, StageHitsType& hits);

    template<class StageLogicalSegmentsType, class StageCachedChunksType, class StageRowType, class StageChunksType>
    __aicore__ __attribute__((always_inline)) inline void ProcessSavedOverlapChunksStage2(StageLogicalSegmentsType& logicalSegments, StageCachedChunksType& cachedChunks, StageRowType& row, StageChunksType& chunks);


    template<class StageEndType, class StagePrefetchedPointType, class StagePositionType, class StageValidRowsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessSmallestStage(StageEndType& end, StagePrefetchedPointType& prefetchedPoint, StagePositionType& position, StageValidRowsType& validRows);

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
    template<class StageSpatialWorkspaceType>
    __aicore__ __attribute__((always_inline)) inline void ProcessMinsStage(StageSpatialWorkspaceType& spatialWorkspace);
#endif

    template<class StageCertifiedType, class StageActiveType>
    __aicore__ __attribute__((always_inline)) inline void ProcessCopyStage3(StageCertifiedType& certified, StageActiveType& active);

    template<class StagePrefetchedPointType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageInitialInputsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDataCopyPadStage(StagePrefetchedPointType& prefetchedPoint, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageInitialInputsType& initialInputs);


    template<class StageLogicalSegmentsType, class StageUniqueRowsType, class StageRowType, class StageChunksType>
    __aicore__ __attribute__((always_inline)) inline void ProcessProcessCopyStageStage(StageLogicalSegmentsType& logicalSegments, StageUniqueRowsType& uniqueRows, StageRowType& row, StageChunksType& chunks);


#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    template<class StageSharedCoefficientCoordinatesSafeType>
    __aicore__ __attribute__((always_inline)) inline void ProcessAbsStage(StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe);


#endif

    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageSpatialPointCoordinatesSafeType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageCorePointSafeType, class StageSharedCoefficientCoordinatesSafeType, class StageLogicalSegmentsType, class StageExactPointSafeType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDefinedStage2(StageSpatialWorkspaceType& spatialWorkspace, StageSpatialPointCoordinatesSafeType& spatialPointCoordinatesSafe, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageCorePointSafeType& corePointSafe, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe);
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageSpatialPointCoordinatesSafeType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageCorePointSafeType, class StageSharedCoefficientCoordinatesSafeType, class StageLogicalSegmentsType, class StageExactPointSafeType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDefinedStage5(StageSpatialWorkspaceType& spatialWorkspace, StageSpatialPointCoordinatesSafeType& spatialPointCoordinatesSafe, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageCorePointSafeType& corePointSafe, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)


#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))


#endif


#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    template<bool OUTLINE_FULL_FLOAT_FALLBACK, class StageSpatialWorkspaceType, class StageSpatialPointCoordinatesSafeType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageCorePointSafeType, class StageSharedCoefficientCoordinatesSafeType, class StageLogicalSegmentsType, class StageExactPointSafeType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDefinedStage(StageSpatialWorkspaceType& spatialWorkspace, StageSpatialPointCoordinatesSafeType& spatialPointCoordinatesSafe, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageCorePointSafeType& corePointSafe, StageSharedCoefficientCoordinatesSafeType& sharedCoefficientCoordinatesSafe, StageLogicalSegmentsType& logicalSegments, StageExactPointSafeType& exactPointSafe);
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    template<class StageSpatialWorkspaceType, class StageBeginType, class StageSpatialCoefficientDirtyType, class StageSpatialPrunedType, class StageSpatialCoefficientsType, class StagePositionType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDataCopyStage(StageSpatialWorkspaceType& spatialWorkspace, StageBeginType& begin, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageSpatialPrunedType& spatialPruned, StageSpatialCoefficientsType& spatialCoefficients, StagePositionType& position);


#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
    template<class StageSpatialPrunedType, class StageSpatialDirtyCountType, class StageCertifiedType, class StageActiveType, class StageAcceptedSparseRowsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessSetMaskNormStage(StageSpatialPrunedType& spatialPruned, StageSpatialDirtyCountType& spatialDirtyCount, StageCertifiedType& certified, StageActiveType& active, StageAcceptedSparseRowsType& acceptedSparseRows);


#endif

#if defined(DTYPE_TREE_POINTS)
    template<class StageEndType, class StageSpatialCoefficientDirtyType, class StageCachedType, class StagePrefetchedPointType, class StageOutputInFlightType, class StagePositionType, class StageBatchType, class StageValidRowsType, class StageExactRowsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessTreeGeneralExactFallbackStage(StageEndType& end, StageSpatialCoefficientDirtyType& spatialCoefficientDirty, StageCachedType& cached, StagePrefetchedPointType& prefetchedPoint, StageOutputInFlightType& outputInFlight, StagePositionType& position, StageBatchType& batch, StageValidRowsType& validRows, StageExactRowsType& exactRows);
#endif

    template<class StageValidRowsType, class StageRefinementRowsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessCompareScalarStage(StageValidRowsType& validRows, StageRefinementRowsType& refinementRows);

protected:
    __aicore__ inline uint32_t Smallest(uint32_t a,uint32_t b);
    __aicore__ inline LocalTensor<float> C(uint32_t k);
    __aicore__ inline LocalTensor<float> W(uint32_t k);
    __aicore__ inline LocalTensor<float> P(uint32_t k);
    template<bool FIXED_ARENA>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> SearchC(uint32_t k);
    template<bool FIXED_ARENA>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> SearchW(uint32_t k);
    template<bool FIXED_ARENA>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> SearchP(uint32_t k);

    // Copy handles so grouped initialization never changes the caller's tensor view.
    __aicore__ __attribute__((always_inline)) static inline void InitializeGroupedSearchState(
        LocalTensor<float> initialDistances,const LocalTensor<int32_t>& indices,uint32_t indexRepeats);
    template<bool FIXED_ARENA,bool DIRECT_PREDICATES=false>
    __aicore__ __attribute__((always_inline)) inline void MergeSearchMinimum(uint32_t j);
    template<bool FIXED_ARENA>
    __aicore__ __attribute__((always_inline)) inline void CompactSearchTail(uint32_t j,uint32_t tail);
    template<bool FIXED_ARENA, class StageJType, class StageTailType, class StageSameType>
    __aicore__ __attribute__((always_inline)) inline void CompactSearchTailCopyStage(StageJType& j, StageTailType& tail, StageSameType& same);

    template<bool DIRECT_PREDICATES,uint32_t FIXED_SEARCH_ROWS=0>
    __aicore__ __attribute__((always_inline)) inline void SearchStaticPolicy();
    template<bool DIRECT_PREDICATES, auto explicitStageBarrier, class StageLiveRows_Type, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType, class StageSegmentCountType>
    __aicore__ __attribute__((always_inline)) inline void SearchStaticPolicyCompactSearchTailStage(StageLiveRows_Type& liveRows_, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary, StageSegmentCountType& segmentCount);

    template<bool DIRECT_PREDICATES, class StageLiveRows_Type, class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchStaticPolicyCompareStage(StageLiveRows_Type& liveRows_, StageSameType& same, StageUnaryType& unary, StageJType& j);


    __aicore__ __attribute__((always_inline)) inline void SearchStaticPolicyDuplicateStage();

    template<bool DIRECT_PREDICATES, class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchStaticPolicyMergeSearchMinimumStage(StageSameType& same, StageUnaryType& unary, StageJType& j);


    __aicore__ __attribute__((always_inline)) inline void SavePrefixTailPredicate(uint32_t start,uint32_t tail);

    // Float and Half prefix searches share operations; arena offsets are compile-time policies.
    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> PrefixC(uint32_t k);

    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> PrefixW(uint32_t k);

    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> PrefixP(uint32_t k);

    template<bool HALF_RESOURCE,bool MASK_READY=false>
    __aicore__ __attribute__((always_inline)) static inline void MergePrefixMinimum(uint32_t j);

    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) static inline void CompactPrefixTail(uint32_t j,uint32_t tail);
    template<bool HALF_RESOURCE, class StageJType, class StageTailType, class StageChunkLower_Type, class StageMask_Type, class StageOffset_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageSameType>
    __aicore__ __attribute__((always_inline)) static inline void CompactPrefixTailCopyStage(StageJType& j, StageTailType& tail, StageChunkLower_Type& chunkLower_, StageMask_Type& mask_, StageOffset_Type& offset_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageSameType& same);


    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) static inline void SavePrefixPredicate(uint32_t start,uint32_t tail);

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
    template<bool HALF_RESOURCE,bool FULL_ROWS=false>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixState(uint32_t inputRows,uint32_t segmentCount);
    template<bool HALF_RESOURCE, auto pp_, class StageLiveRows_Type, class StageCertHigh_Type, class StageCert_Type, class StageIdx_Type, class StageOffset_Type, class StageOneHot_Type, class StageRowBase_Type, class StageValid_Type, class StageXyz_Type>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixStateSetMaskNormStage4(StageLiveRows_Type& liveRows_, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageIdx_Type& idx_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageRowBase_Type& rowBase_, StageValid_Type& valid_, StageXyz_Type& xyz_);

    template<bool HALF_RESOURCE, bool FULL_ROWS, auto lowerPlane, auto pp_, auto errorPlane, class StageSegmentCountType, class StageLiveRows_Type, class StageIntervalFactorType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageSegmentIndices_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageValid_Type, class StageXyz_Type, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixStateSavePrefixPredicateStage3(StageSegmentCountType& segmentCount, StageLiveRows_Type& liveRows_, StageIntervalFactorType& intervalFactor, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageSegmentIndices_Type& segmentIndices_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageValid_Type& valid_, StageXyz_Type& xyz_, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary);

    template<bool HALF_RESOURCE, auto lowerPlane, auto pp_, auto errorPlane, class StageLiveRows_Type, class StageIntervalFactorType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageXyz_Type, class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixStateSetMaskNormStage3(StageLiveRows_Type& liveRows_, StageIntervalFactorType& intervalFactor, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageXyz_Type& xyz_, StageSameType& same, StageUnaryType& unary, StageJType& j);

    template<bool HALF_RESOURCE, auto lowerPlane, class StageLiveRows_Type, class StageMask_Type, class StageSlotType, class StagePredicatesType>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixStateCompareStage(StageLiveRows_Type& liveRows_, StageMask_Type& mask_, StageSlotType& slot, StagePredicatesType& predicates);

    template<bool HALF_RESOURCE, auto errorPlane, class StageLiveRows_Type, class StageOffset_Type, class StageRowBase_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixStateSetMaskNormStage2(StageLiveRows_Type& liveRows_, StageOffset_Type& offset_, StageRowBase_Type& rowBase_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageSameType& same, StageUnaryType& unary);

    template<bool HALF_RESOURCE, auto pp_, class StageCertHigh_Type, class StageCert_Type, class StageIdx_Type, class StageOffset_Type, class StageRowBase_Type, class StageXyz_Type>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixStateSetMaskNormStage(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageIdx_Type& idx_, StageOffset_Type& offset_, StageRowBase_Type& rowBase_, StageXyz_Type& xyz_);

    template<bool HALF_RESOURCE, class StageCertHigh_Type, class StageCert_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageRecord_Type, class StageSecond_Type, class StageTileIdx_Type, class StageSameType, class StageUnaryType, class StageJType, class StageChunkMinimumType, class StageChunkUpperType>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixStateMergePrefixMinimumStage2(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageRecord_Type& record_, StageSecond_Type& second_, StageTileIdx_Type& tileIdx_, StageSameType& same, StageUnaryType& unary, StageJType& j, StageChunkMinimumType& chunkMinimum, StageChunkUpperType& chunkUpper);


#else
    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixState(uint32_t liveRows_,uint32_t segmentCount);
    template<bool HALF_RESOURCE, auto lowerPlane, auto errorPlane, auto pp_, class StageLiveRows_Type, class StageSegmentCountType, class StageIntervalFactorType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageOneHot_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageSegmentIndices_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageValid_Type, class StageXyz_Type, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType, class StageRowMaskType>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixStateSavePrefixPredicateStage(StageLiveRows_Type& liveRows_, StageSegmentCountType& segmentCount, StageIntervalFactorType& intervalFactor, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageSegmentIndices_Type& segmentIndices_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageValid_Type& valid_, StageXyz_Type& xyz_, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary, StageRowMaskType& rowMask);
    template<bool HALF_RESOURCE, auto lowerPlane, auto errorPlane, auto pp_, class StageLiveRows_Type, class StageSegmentCountType, class StageIntervalFactorType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageSegmentIndices_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageValid_Type, class StageXyz_Type, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType, class StageRowMaskType>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixStateSavePrefixPredicateStage2(StageLiveRows_Type& liveRows_, StageSegmentCountType& segmentCount, StageIntervalFactorType& intervalFactor, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageSegmentIndices_Type& segmentIndices_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageValid_Type& valid_, StageXyz_Type& xyz_, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary, StageRowMaskType& rowMask);

    template<bool HALF_RESOURCE, auto lowerPlane, auto errorPlane, auto pp_, class StageLiveRows_Type, class StageIntervalFactorType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageXyz_Type, class StageSameType, class StageUnaryType, class StageRowMaskType, class StageJType>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixStateCopyStage(StageLiveRows_Type& liveRows_, StageIntervalFactorType& intervalFactor, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageXyz_Type& xyz_, StageSameType& same, StageUnaryType& unary, StageRowMaskType& rowMask, StageJType& j);


    template<bool HALF_RESOURCE, class StageCertHigh_Type, class StageCert_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageRecord_Type, class StageSecond_Type, class StageTileIdx_Type, class StageSameType, class StageUnaryType, class StageJType, class StageChunkMinimumType, class StageChunkUpperType>
    __aicore__ __attribute__((always_inline)) static inline void SearchPrefixStateMergePrefixMinimumStage(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageRecord_Type& record_, StageSecond_Type& second_, StageTileIdx_Type& tileIdx_, StageSameType& same, StageUnaryType& unary, StageJType& j, StageChunkMinimumType& chunkMinimum, StageChunkUpperType& chunkUpper);


#endif

    __aicore__ __attribute__((noinline,aligned(128))) static uint32_t SearchPrefixOutlined(uint32_t liveRows_,uint32_t segmentCount);

    __aicore__ __attribute__((noinline)) static uint32_t SearchHalfPrefixOutlined(uint32_t liveRows,uint32_t segmentCount);

    __aicore__ inline void Search();
    template<class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void SearchSavePrefixTailPredicateStage(StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary);

    template<class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchSubStage(StageSameType& same, StageUnaryType& unary, StageJType& j);

    template<auto fixedSearchMask, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchCopyStage(StageJType& j);

    template<auto fixedSearchMask, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void SearchSetMaskNormStage2(StageSameType& same, StageUnaryType& unary);

    template<auto fixedSearchMask, class StageSameType, class StageUnaryType, class StageJType, class StageChunkMinimumType, class StageChunkUpperType>
    __aicore__ __attribute__((always_inline)) inline void SearchMergeSearchMinimumStage(StageSameType& same, StageUnaryType& unary, StageJType& j, StageChunkMinimumType& chunkMinimum, StageChunkUpperType& chunkUpper);

    template<class StageSameType, class StageSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchMulsStage(StageSameType& same, StageSegmentType& segment, StageUnaryType& unary, StageJType& j);

    template<auto fixedSearchMask>
    __aicore__ __attribute__((always_inline)) inline void SearchArenaStage();

    template<auto fixedSearchMask, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void SearchSetMaskNormStage(StageSameType& same, StageUnaryType& unary);

    template<auto fixedSearchMask>
    __aicore__ __attribute__((always_inline)) inline void SearchDuplicateStage();


    template<uint32_t extent>
    __aicore__ __attribute__((always_inline)) inline bool ReduceCoordinateValidity();
    // Coordinate storage changes the admissible range, not certificate reduction.
    template<bool COEFFICIENT_INPUT>
    __aicore__ __attribute__((always_inline)) inline bool CoarseCoordinatesSafe();
    __aicore__ __attribute__((always_inline)) inline bool CoarseDirectionSafe();
    // Shared certificate resolution, fully inlined into each original search.
    __aicore__ __attribute__((always_inline)) inline void FinalizeGuardedCertificate(uint32_t activeRows);
    __aicore__ __attribute__((always_inline)) inline void FinalizeGuardedCertificateDuplicateStage();

    // Keep the cross-chunk certificate packed until the caller consumes it.
    // Both callers share this exact mask and barrier sequence.
    __aicore__ __attribute__((always_inline)) inline void BuildCrossChunkCertificate(
        uint32_t chunks,const BinaryRepeatParams& same,const UnaryRepeatParams& unary);
    template<class StageSameType, class StageUnionsType, class StageCollisionsType, class StageNextUnionsType, class StageNextCollisionsType, class StageMergedCollisionsType>
    __aicore__ __attribute__((always_inline)) inline void BuildCrossChunkCertificateCStage(StageSameType& same, StageUnionsType& unions, StageCollisionsType& collisions, StageNextUnionsType& nextUnions, StageNextCollisionsType& nextCollisions, StageMergedCollisionsType& mergedCollisions);

    __aicore__ inline void SearchGuardedHalfCachedRows();
    __aicore__ static constexpr bool UsesCapacityStage3();

    template<class StageGuardedRowsType, class StageCoarseErrorFactorType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedHalfCachedRowsCompactSearchTailStage(StageGuardedRowsType& guardedRows, StageCoarseErrorFactorType& coarseErrorFactor, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary);

    template<class StageGuardedRowsType, class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedHalfCachedRowsCompareScalarStage(StageGuardedRowsType& guardedRows, StageSameType& same, StageUnaryType& unary, StageJType& j);

    template<class StageGuardedRowsType, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedHalfCachedRowsSetMaskNormStage2(StageGuardedRowsType& guardedRows, StageSameType& same, StageUnaryType& unary);

    template<auto DEFER_SECOND, class StageSameType, class StageUnaryType, class StageJType, class StageChunkMinimumType, class StageChunkUpperType>
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedHalfCachedRowsMergeSearchMinimumStage(StageSameType& same, StageUnaryType& unary, StageJType& j, StageChunkMinimumType& chunkMinimum, StageChunkUpperType& chunkUpper);

    template<class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedHalfCachedRowsSetMaskNormStage(StageSameType& same, StageUnaryType& unary);

    template<class StageGuardedRowsType, class StageCoarseErrorFactorType, class StageSameType, class StageSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedHalfCachedRowsMulsStage(StageGuardedRowsType& guardedRows, StageCoarseErrorFactorType& coarseErrorFactor, StageSameType& same, StageSegmentType& segment, StageUnaryType& unary, StageJType& j);

    template<class StageGuardedRowsType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedHalfCachedRowsCompareStage(StageGuardedRowsType& guardedRows, StageJType& j);

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> HalfFullC(uint32_t k);
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> HalfFullW(uint32_t k);
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> HalfFullP(uint32_t k);
    __aicore__ __attribute__((noinline)) static uint32_t SearchGuardedHalfTypeCertifiedRows(uint32_t liveRows_,uint32_t segmentCount,bool tightCoefficientSafe_);
    template<class StageSegmentCountType, class StageGuardedRowsType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageOneHot_Type, class StageOutputOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageSegmentIndices_Type, class StageSmallMask_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageValid_Type, class StageXyz_Type, class StageCoarseErrorFactorType, class StageLowerNormFactorType, class StageUpperNormFactorType>
    __aicore__ __attribute__((always_inline)) static inline uint32_t SearchGuardedHalfTypeCertifiedRowsInitializeGroupedSearchStateStage(StageSegmentCountType& segmentCount, StageGuardedRowsType& guardedRows, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageOutputOffset_Type& outputOffset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageSegmentIndices_Type& segmentIndices_, StageSmallMask_Type& smallMask_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageValid_Type& valid_, StageXyz_Type& xyz_, StageCoarseErrorFactorType& coarseErrorFactor, StageLowerNormFactorType& lowerNormFactor, StageUpperNormFactorType& upperNormFactor);


    template<class StageSegmentCountType, class StageGuardedRowsType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageSegmentIndices_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageValid_Type, class StageXyz_Type, class StageCoarseErrorFactorType, class StageLowerNormFactorType, class StageUpperNormFactorType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) static inline void SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage5(StageSegmentCountType& segmentCount, StageGuardedRowsType& guardedRows, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageSegmentIndices_Type& segmentIndices_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageValid_Type& valid_, StageXyz_Type& xyz_, StageCoarseErrorFactorType& coarseErrorFactor, StageLowerNormFactorType& lowerNormFactor, StageUpperNormFactorType& upperNormFactor, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary);

    template<class StageSegmentCountType, class StageGuardedRowsType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageXyz_Type, class StageUpperNormFactorType, class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) static inline void SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage4(StageSegmentCountType& segmentCount, StageGuardedRowsType& guardedRows, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageXyz_Type& xyz_, StageUpperNormFactorType& upperNormFactor, StageSameType& same, StageUnaryType& unary, StageJType& j);

    template<class StageGuardedRowsType, class StageOffset_Type, class StageRowBase_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) static inline void SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage3(StageGuardedRowsType& guardedRows, StageOffset_Type& offset_, StageRowBase_Type& rowBase_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageSameType& same, StageUnaryType& unary);

    template<auto DEFER_SECOND, class StageCertHigh_Type, class StageCert_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageRecord_Type, class StageSecond_Type, class StageTileIdx_Type, class StageSameType, class StageUnaryType, class StageJType, class StageChunkMinimumType, class StageChunkUpperType>
    __aicore__ __attribute__((always_inline)) static inline void SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage2(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageRecord_Type& record_, StageSecond_Type& second_, StageTileIdx_Type& tileIdx_, StageSameType& same, StageUnaryType& unary, StageJType& j, StageChunkMinimumType& chunkMinimum, StageChunkUpperType& chunkUpper);

    template<class StageTileHi_Type, class StageTileLo_Type, class StageUpperNormFactorType, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) static inline void SearchGuardedHalfTypeCertifiedRowsSetMaskNormStage(StageTileHi_Type& tileHi_, StageTileLo_Type& tileLo_, StageUpperNormFactorType& upperNormFactor, StageSameType& same, StageUnaryType& unary);

    template<class StageGuardedRowsType, class StageRefined_Type, class StageCoarseErrorFactorType, class StageLowerNormFactorType, class StageSameType, class StageSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) static inline void SearchGuardedHalfTypeCertifiedRowsMulsStage(StageGuardedRowsType& guardedRows, StageRefined_Type& refined_, StageCoarseErrorFactorType& coarseErrorFactor, StageLowerNormFactorType& lowerNormFactor, StageSameType& same, StageSegmentType& segment, StageUnaryType& unary, StageJType& j);

    template<auto pp_, class StageGuardedRowsType, class StageMask_Type, class StageRefined_Type, class StageXyz_Type, class StageJType>
    __aicore__ __attribute__((always_inline)) static inline void SearchGuardedHalfTypeCertifiedRowsCompareStage(StageGuardedRowsType& guardedRows, StageMask_Type& mask_, StageRefined_Type& refined_, StageXyz_Type& xyz_, StageJType& j);

    template<class StageCertHigh_Type, class StageCert_Type, class StageIdx_Type, class StageOffset_Type, class StageOneHot_Type, class StageOutputOffset_Type>
    __aicore__ __attribute__((always_inline)) static inline void SearchGuardedHalfTypeCertifiedRowsArenaStage(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageIdx_Type& idx_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageOutputOffset_Type& outputOffset_);

    template<auto DEFER_SECOND, class StageSegmentCountType, class StageChunkLower_Type, class StageHi_Type, class StageMask_Type, class StageSecond_Type, class StageSameType>
    __aicore__ __attribute__((always_inline)) static inline void SearchGuardedHalfTypeCertifiedRowsDuplicateStage2(StageSegmentCountType& segmentCount, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageMask_Type& mask_, StageSecond_Type& second_, StageSameType& same);

    template<class StageCertHigh_Type, class StageCert_Type, class StageIdx_Type, class StageOffset_Type, class StageOneHot_Type, class StageSmallMask_Type, class StageValid_Type>
    __aicore__ __attribute__((always_inline)) static inline void SearchGuardedHalfTypeCertifiedRowsDuplicateStage(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageIdx_Type& idx_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageSmallMask_Type& smallMask_, StageValid_Type& valid_);

#endif
    __aicore__ inline void SearchGuardedError();


    template<class StageCoarseErrorFactorType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedErrorCompactSearchTailStage(StageCoarseErrorFactorType& coarseErrorFactor, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary);

    template<class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedErrorCompareScalarStage(StageSameType& same, StageUnaryType& unary, StageJType& j);

    template<class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedErrorSetMaskNormStage2(StageSameType& same, StageUnaryType& unary);


    template<class StageCoarseErrorFactorType, class StageSameType, class StageSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedErrorMulsStage(StageCoarseErrorFactorType& coarseErrorFactor, StageSameType& same, StageSegmentType& segment, StageUnaryType& unary, StageJType& j);

    template<class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchGuardedErrorCompareStage(StageJType& j);

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    __aicore__ __attribute__((always_inline)) inline bool CertifyCorePointCoordinates(uint32_t begin,uint32_t end);
    __aicore__ __attribute__((always_inline)) inline
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_RETURN
bool
#else
void
#endif
    SearchGuardedCoreCertifiedRows(bool corePointSafe,bool coefficientCoordinatesSafe);
    template<bool SHARED_BOUND,bool COMPLETE_COEFFICIENTS>
    __aicore__ __attribute__((always_inline)) inline void SearchCertifiedRowsBody(bool corePointSafe);


    template<bool SHARED_BOUND, bool COMPLETE_COEFFICIENTS, class StageSearchRowsType, class StageCoarseErrorFactorType, class StagePointBoundType, class StageSharedSameType, class StageLowerScoresType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void SearchCertifiedRowsBodyCompactSearchTailStage(StageSearchRowsType& searchRows, StageCoarseErrorFactorType& coarseErrorFactor, StagePointBoundType& pointBound, StageSharedSameType& sharedSame, StageLowerScoresType& lowerScores, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary);
    template<bool COMPLETE_COEFFICIENTS, class StageSearchRowsType, class StageLowerScoresType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchCertifiedRowsBodyCompactSearchTailStageSetMaskNormStage(StageSearchRowsType& searchRows, StageLowerScoresType& lowerScores, StageJType& j);


    template<bool COMPLETE_COEFFICIENTS, auto DEFER_SECOND, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void SearchCertifiedRowsBodyDuplicateStage(StageSameType& same, StageUnaryType& unary);


    template<bool SHARED_BOUND, bool COMPLETE_COEFFICIENTS, class StageSearchRowsType, class StageCoarseErrorFactorType, class StageLowerScoresType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchCertifiedRowsBodyDefinedStage(StageSearchRowsType& searchRows, StageCoarseErrorFactorType& coarseErrorFactor, StageLowerScoresType& lowerScores, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary, StageJType& j);

    __aicore__ static constexpr bool UsesRowsStage14();


    template<class StageSearchRowsType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchCertifiedRowsBodyWStage(StageSearchRowsType& searchRows, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary, StageJType& j);

    template<bool SHARED_BOUND, class StageCoarseErrorFactorType, class StagePointBoundType, class StageSharedSameType, class StageSharedUnaryType>
    __aicore__ __attribute__((always_inline)) inline void SearchCertifiedRowsBodySetMaskNormStage2(StageCoarseErrorFactorType& coarseErrorFactor, StagePointBoundType& pointBound, StageSharedSameType& sharedSame, StageSharedUnaryType& sharedUnary);

    template<bool SHARED_BOUND, class StageSearchRowsType, class StagePointBoundType, class StageSharedSameType, class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchCertifiedRowsBodyCopyStage(StageSearchRowsType& searchRows, StagePointBoundType& pointBound, StageSharedSameType& sharedSame, StageSameType& same, StageUnaryType& unary, StageJType& j);


    template<bool SHARED_BOUND, class StageSearchRowsType, class StageCoarseErrorFactorType, class StageLowerScoresType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchCertifiedRowsBodySubStage(StageSearchRowsType& searchRows, StageCoarseErrorFactorType& coarseErrorFactor, StageLowerScoresType& lowerScores, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary, StageJType& j);

    template<bool SHARED_BOUND, class StageSearchRowsType, class StageCoarseErrorFactorType, class StageSameType, class StageSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchCertifiedRowsBodyMulsStage2(StageSearchRowsType& searchRows, StageCoarseErrorFactorType& coarseErrorFactor, StageSameType& same, StageSegmentType& segment, StageUnaryType& unary, StageJType& j);

    template<class StageSearchRowsType, class StageLowerScoresType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchCertifiedRowsBodyCompareStage(StageSearchRowsType& searchRows, StageLowerScoresType& lowerScores, StageJType& j);

    template<bool SHARED_BOUND, class StageSearchRowsType, class StageCoarseErrorFactorType, class StageSameType, class StageSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void SearchCertifiedRowsBodyMulsStage(StageSearchRowsType& searchRows, StageCoarseErrorFactorType& coarseErrorFactor, StageSameType& same, StageSegmentType& segment, StageUnaryType& unary, StageJType& j);


    template<bool SHARED_BOUND>
    __aicore__ __attribute__((noinline)) void SearchGuardedCoreCertifiedRowsImpl(bool corePointSafe);
    // The complete coefficient tile has no segment padding; all original row and range certificates still apply.
    __aicore__ __attribute__((noinline)) void SearchGuardedCoreCertifiedRowsAlignedImpl(bool corePointSafe);

    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> SpatialStateC(uint32_t k);
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> SpatialStateW(uint32_t k);
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> SpatialStateP(uint32_t k);

    __aicore__ __attribute__((noinline)) static uint32_t SearchSpatialStaticState(uint32_t first,uint32_t limit,bool resume,bool tightCoefficientSafe_);

#endif
