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

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
    template<auto pp_, auto searchRows, class StageFirstType, class StageLimitType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageOneHot_Type, class StageOutputOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageValid_Type, class StageXyz_Type, class StagePendingSpatialMapType, class StageLowerNormFactorType, class StageUpperNormFactorType, class StageLowerScoresType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) static inline uint32_t SearchSpatialStaticStateSetMaskNormStage(StageFirstType& first, StageLimitType& limit, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageOutputOffset_Type& outputOffset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageValid_Type& valid_, StageXyz_Type& xyz_, StagePendingSpatialMapType& pendingSpatialMap, StageLowerNormFactorType& lowerNormFactor, StageUpperNormFactorType& upperNormFactor, StageLowerScoresType& lowerScores, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary);
    template<class StageCertHigh_Type, class StageCert_Type, class StageIdx_Type, class StageOffset_Type, class StageOneHot_Type, class StageOutputOffset_Type, class StageSecond_Type, class StageValid_Type, class StagePendingSpatialMapType, class StageCount_Type, class StageU0Type, class StageU1Type, class StageU2Type, class StageU3Type, class StageC0Type, class StageC1Type, class StageC2Type, class StageC3Type>
    __aicore__ __attribute__((always_inline)) static inline uint32_t SearchSpatialStaticStateReinterpretCastStage(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageIdx_Type& idx_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageOutputOffset_Type& outputOffset_, StageSecond_Type& second_, StageValid_Type& valid_, StagePendingSpatialMapType& pendingSpatialMap, StageCount_Type& count_, StageU0Type& u0, StageU1Type& u1, StageU2Type& u2, StageU3Type& u3, StageC0Type& c0, StageC1Type& c1, StageC2Type& c2, StageC3Type& c3);
    template<class StageCertHigh_Type, class StageCert_Type, class StageValid_Type, class StagePendingSpatialMapType, class StageCount_Type, class StageCertBinaryType>
    __aicore__ __attribute__((always_inline)) static inline uint32_t SearchSpatialStaticStateOrStage(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageValid_Type& valid_, StagePendingSpatialMapType& pendingSpatialMap, StageCount_Type& count_, StageCertBinaryType& certBinary);


    template<auto pp_, auto searchRows, class StageFirstType, class StageLimitType, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageXyz_Type, class StagePendingSpatialMapType, class StageLowerNormFactorType, class StageUpperNormFactorType, class StageLowerScoresType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) static inline void SearchSpatialStaticStateDefinedStage(StageFirstType& first, StageLimitType& limit, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageXyz_Type& xyz_, StagePendingSpatialMapType& pendingSpatialMap, StageLowerNormFactorType& lowerNormFactor, StageUpperNormFactorType& upperNormFactor, StageLowerScoresType& lowerScores, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary);

    template<auto pp_, auto searchRows, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageRefined_Type, class StageTileIdx_Type, class StageXyz_Type, class StageLowerScoresType, class StageSameType, class StageUnaryType, class StageJType, class StageChunkMinimumType, class StageChunkUpperType>
    __aicore__ __attribute__((always_inline)) static inline void SearchSpatialStaticStateBrcbStage(StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageRefined_Type& refined_, StageTileIdx_Type& tileIdx_, StageXyz_Type& xyz_, StageLowerScoresType& lowerScores, StageSameType& same, StageUnaryType& unary, StageJType& j, StageChunkMinimumType& chunkMinimum, StageChunkUpperType& chunkUpper);


    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> SpatialPruneW(uint32_t k);
    __aicore__ __attribute__((always_inline)) static inline void SpatialPruneIndices(const LocalTensor<int32_t>& values,int32_t first,uint32_t count);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
    __attribute__((noinline,minsize,aligned(128)))
#else
    __attribute__((noinline,aligned(128)))
#endif
    __aicore__ static uint32_t PruneSpatialStaticState(GM_ADDR workspace,uint32_t segments,bool& pruned,uint32_t& logicalSegments);
    template<class StageSegmentsType, class StagePrunedType, class StageLogicalSegmentsType, class StageWork_Type, class StageCompactP_Type, class StageRecord_Type, class StageTileHi_Type, class StageCenterType, class StageThresholdType, class StageBroadcastType>
    __aicore__ __attribute__((always_inline)) static inline uint32_t PruneSpatialStaticStateDefinedStage(GlobalTensor<int32_t>& spatialIndices, StageSegmentsType& segments, StagePrunedType& pruned, StageLogicalSegmentsType& logicalSegments, StageWork_Type& work_, StageCompactP_Type& compactP_, StageRecord_Type& record_, StageTileHi_Type& tileHi_, StageCenterType& center, StageThresholdType& threshold, StageBroadcastType& broadcast);
    template<class StageSegmentsType, class StagePrunedType, class StageLogicalSegmentsType, class StageWork_Type, class StageCompactP_Type, class StageRecord_Type, class StageTileHi_Type, class StageThresholdType, class StageBroadcastType, class StageQueryRadiusType, class StageSavedCenterScoresType, class StageOtherCenterScoresType, class StageCenterBroadcastType, class StageCenterErrorBroadcastType>
    __aicore__ __attribute__((always_inline)) static inline uint32_t PruneSpatialStaticStateDefinedStage2(GlobalTensor<int32_t>& spatialIndices, StageSegmentsType& segments, StagePrunedType& pruned, StageLogicalSegmentsType& logicalSegments, StageWork_Type& work_, StageCompactP_Type& compactP_, StageRecord_Type& record_, StageTileHi_Type& tileHi_, StageThresholdType& threshold, StageBroadcastType& broadcast, StageQueryRadiusType& queryRadius, StageSavedCenterScoresType& savedCenterScores, StageOtherCenterScoresType& otherCenterScores, StageCenterBroadcastType& centerBroadcast, StageCenterErrorBroadcastType& centerErrorBroadcast);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
    template<class StageWork_Type, class StageCenterBroadcastType, class StageCenterSameType, class StageCenterUnaryType>
    __aicore__ __attribute__((always_inline)) static inline void PruneSpatialStaticStateDefinedStage2SetMaskNormStage(StageWork_Type& work_, StageCenterBroadcastType& centerBroadcast, StageCenterSameType& centerSame, StageCenterUnaryType& centerUnary);
#endif

    template<class StageWitnessOffsetsType>
    __aicore__ __attribute__((always_inline)) static inline void PruneSpatialStaticStateDefinedStage2SpatialStateCStage(StageWitnessOffsetsType& witnessOffsets);

    template<class StageSegmentsType, class StagePredicateType, class StageRealCenterScoresType>
    __aicore__ __attribute__((always_inline)) static inline void PruneSpatialStaticStateDefinedStage2SpatialPruneIndicesStage(StageSegmentsType& segments, StagePredicateType& predicate, StageRealCenterScoresType& realCenterScores);

    template<class StageSegmentsType, class StagePrunedType, class StageLogicalSegmentsType, class StageWork_Type, class StageCompactP_Type, class StageRecord_Type, class StageTileHi_Type, class StageThresholdType, class StageBroadcastType, class StageQueryRadiusType, class StageSavedCenterScoresType, class StageOtherCenterScoresType, class StagePredicateType, class StageWitnessType>
    __aicore__ __attribute__((always_inline)) static inline uint32_t PruneSpatialStaticStateSetMaskNormStage2(GlobalTensor<int32_t>& spatialIndices, StageSegmentsType& segments, StagePrunedType& pruned, StageLogicalSegmentsType& logicalSegments, StageWork_Type& work_, StageCompactP_Type& compactP_, StageRecord_Type& record_, StageTileHi_Type& tileHi_, StageThresholdType& threshold, StageBroadcastType& broadcast, StageQueryRadiusType& queryRadius, StageSavedCenterScoresType& savedCenterScores, StageOtherCenterScoresType& otherCenterScores, StagePredicateType& predicate, StageWitnessType& witness);
    template<class StageSegmentsType, class StagePrunedType, class StageLogicalSegmentsType, class StageWork_Type, class StageBroadcastType, class StageSavedCenterScoresType, class StageOtherCenterScoresType, class StagePredicateType>
    __aicore__ __attribute__((always_inline)) static inline uint32_t PruneSpatialStaticStateReinterpretCastStage(GlobalTensor<int32_t>& spatialIndices, StageSegmentsType& segments, StagePrunedType& pruned, StageLogicalSegmentsType& logicalSegments, StageWork_Type& work_, StageBroadcastType& broadcast, StageSavedCenterScoresType& savedCenterScores, StageOtherCenterScoresType& otherCenterScores, StagePredicateType& predicate);
    template<class StagePrunedType, class StageWork_Type, class StageKeptType, class StagePendingIndicesType, class StageCompactedType>
    __aicore__ __attribute__((always_inline)) static inline uint32_t PruneSpatialStaticStateDefinedStage3(GlobalTensor<int32_t>& spatialIndices, StagePrunedType& pruned, StageWork_Type& work_, StageKeptType& kept, StagePendingIndicesType& pendingIndices, StageCompactedType& compacted);


#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
#else
    template<class StageWork_Type, class StageCenterBroadcastType, class StageCenterErrorBroadcastType>
    __aicore__ __attribute__((always_inline)) static inline void PruneSpatialStaticStateSetMaskNormStage(StageWork_Type& work_, StageCenterBroadcastType& centerBroadcast, StageCenterErrorBroadcastType& centerErrorBroadcast);
#endif


    __aicore__ inline void MakeIndices(const LocalTensor<int32_t>& values, int32_t first, uint32_t count);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
    __aicore__ static constexpr bool UsesNativeFloatPrefixSparseStage();
#endif

    template <class U>
    __aicore__ inline LocalTensor<U> Allocate(uint32_t &address, uint32_t elements);
    __aicore__ __attribute__((always_inline)) inline void MaterializeDirectionLow();

    __aicore__ __attribute__((noinline,aligned(128))) static void LoadPrefixResidentSegments(
        GM_ADDR source,uint32_t segmentCount);

    __aicore__ __attribute__((noinline,aligned(128))) static void LoadBoundedResidentSegments(
        GM_ADDR source,uint32_t segmentCount,bool initialInputs);

    __aicore__ __attribute__((noinline,aligned(128))) void LoadResidentSegments(uint32_t batch,bool initialInputs);
    __aicore__ inline void LoadSegments(uint32_t batch,bool initialInputs=false);
    __aicore__ __attribute__((always_inline)) inline void LoadSegmentsWStage();

    __aicore__ __attribute__((always_inline)) inline void LoadSegmentsSetMaskNormStage();

    __aicore__ inline void NormalizeDirections();
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> PrimaryW(uint32_t k);
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> PrimaryC(uint32_t k);
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> PrimaryP(uint32_t k);
    template<uint32_t WIDTH=64,bool OMIT_DIRECTION_LOW=false>
    __aicore__ __attribute__((always_inline)) static inline void PrimaryCompensatedBody();
    template<uint32_t WIDTH, auto DIRECT_LENGTH, auto repeats, class StageMask_Type, class StageValid_Type, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) static inline void PrimaryCompensatedBodyAddStage(StageMask_Type& mask_, StageValid_Type& valid_, StageBinaryType& binary);

    template<uint32_t WIDTH, auto DIRECT_LENGTH, auto repeats, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) static inline void PrimaryCompensatedBodySetMaskNormStage(StageBinaryType& binary);

    template<auto repeats, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) static inline void PrimaryCompensatedBodySubStage(StageBinaryType& binary);


    __aicore__ __attribute__((noinline,aligned(128))) static void PrimaryCompensated16();
    __aicore__ __attribute__((noinline,aligned(128))) static void PrimaryCompensated32();
    __aicore__ __attribute__((noinline,aligned(128))) static void PrimaryCompensatedFull();
    __aicore__ __attribute__((noinline,aligned(128))) static void PrimaryCompensatedExactDirection();
    template<uint32_t WIDTH=64>
    __aicore__ __attribute__((always_inline)) static inline void PrimaryCompensated();

    __aicore__ __attribute__((noinline,aligned(128))) static void PrefixPrimaryCompensatedAxes();
    template<class StageMaskType, class StageWorkCoefficientType, class StageWorkAxesType, class StageProjectAxesType, class StageProjectLowAxesType, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) static inline void PrefixPrimaryCompensatedAxesCompareScalarStage(StageMaskType& mask, StageWorkCoefficientType& workCoefficient, StageWorkAxesType& workAxes, StageProjectAxesType& projectAxes, StageProjectLowAxesType& projectLowAxes, StageSameType& same, StageUnaryType& unary);


    template<bool CORRECT_ZERO=false>
    __aicore__ __attribute__((always_inline)) inline void Compensated();
    template<bool CORRECT_ZERO, class StageRepeatsType, class StageUnaryType, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) inline void CompensatedMaxsStage(StageRepeatsType& repeats, StageUnaryType& unary, StageBinaryType& binary);

    template<bool CORRECT_ZERO, class StageRepeatsType>
    __aicore__ __attribute__((always_inline)) inline void CompensatedWholeReduceMaxStage(StageRepeatsType& repeats);


    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> FewWork(uint32_t k);
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> FewCoefficients(uint32_t k);
    __aicore__ __attribute__((noinline,aligned(128))) inline bool CertifyWideScore(uint32_t candidates);
    __aicore__ __attribute__((always_inline)) static inline uint32_t CertifyRefinedBody(
        const LocalTensor<float>& minima,uint64_t validLanes);
    template<class StageValidLanesType, class StageFiniteBitsType, class StagePossibleBitsType>
    __aicore__ __attribute__((always_inline)) static inline uint32_t CertifyRefinedBodyReturnStage(StageValidLanesType& validLanes, StageFiniteBitsType& finiteBits, StagePossibleBitsType& possibleBits);

    template<class StageValidLanesType, class StageFiniteType, class StageNormFiniteType>
    __aicore__ __attribute__((always_inline)) static inline bool CertifyRefinedBodyMulsStage(StageValidLanesType& validLanes, StageFiniteType& finite, StageNormFiniteType& normFinite, uint32_t& stageResult);

    __aicore__ __attribute__((noinline,aligned(128))) static uint32_t CertifyRefined(
        const LocalTensor<float>& minima,uint64_t validLanes);
    __aicore__ __attribute__((noinline,aligned(128))) static bool RefineFewStatic(uint32_t row,uint32_t chunks,uint64_t& uniqueRows);
    template<class StageRowType, class StageUniqueRowsType, class StageCompactP_Type, class StageSegmentIndices_Type, class StageOutputOffset_Type, class StageOffset_Type, class StageRecord_Type, class StageRecord2_Type, class StageTileIdx_Type, class StageIdx_Type, class StageTileSecond_Type, class StageValid_Type, class StageSmallMask_Type, class StageKeptType>
    __aicore__ __attribute__((always_inline)) static inline bool RefineFewStaticMulsStage(StageRowType& row, StageUniqueRowsType& uniqueRows, StageCompactP_Type& compactP_, StageSegmentIndices_Type& segmentIndices_, StageOutputOffset_Type& outputOffset_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRecord2_Type& record2_, StageTileIdx_Type& tileIdx_, StageIdx_Type& idx_, StageTileSecond_Type& tileSecond_, StageValid_Type& valid_, StageSmallMask_Type& smallMask_, StageKeptType& kept);


    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> SparseW(uint32_t k);
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> SparseC(uint32_t k);
    __aicore__ __attribute__((always_inline)) static inline uint32_t PredicatePopulation(uint64_t bits);
    __aicore__ __attribute__((always_inline)) static inline uint64_t SavedChunkPredicate(uint32_t row,uint32_t chunk);
    template<bool OVERLAPPING_CHUNKS>
    __aicore__ __attribute__((always_inline)) static inline bool RefineFewCandidates(uint32_t row,uint32_t chunks,uint64_t& uniqueRows);
    template<class StageRowType, class StageUniqueRowsType, class StageSegmentIndices_Type, class StageOutputOffset_Type, class StageOffset_Type, class StageRecord_Type, class StageRecord2_Type, class StageTileIdx_Type, class StageIdx_Type, class StageTileSecond_Type, class StageSmallMask_Type, class StageKeptType, class StageAcceptedType>
    __aicore__ __attribute__((always_inline)) static inline bool RefineFewCandidatesWholeReduceMinStage(StageRowType& row, StageUniqueRowsType& uniqueRows, StageSegmentIndices_Type& segmentIndices_, StageOutputOffset_Type& outputOffset_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRecord2_Type& record2_, StageTileIdx_Type& tileIdx_, StageIdx_Type& idx_, StageTileSecond_Type& tileSecond_, StageSmallMask_Type& smallMask_, StageKeptType& kept, StageAcceptedType& accepted);

    template<bool OVERLAPPING_CHUNKS, class StageRowType, class StageChunksType, class StageSegmentIndices_Type, class StageIdx_Type, class StageCertType, class StageCertHighType, class StageValid_Type, class StageKeptType>
    __aicore__ __attribute__((always_inline)) static inline bool RefineFewCandidatesRefineFewCandidatesGetValueStageStage(StageRowType& row, StageChunksType& chunks, StageSegmentIndices_Type& segmentIndices_, StageIdx_Type& idx_, StageCertType& cert, StageCertHighType& certHigh, StageValid_Type& valid_, StageKeptType& kept, bool& stageResult);

    template<class StageRowType, class StageChunksType, class StageSegmentIndices_Type, class StageIdx_Type, class StageValid_Type, class StageKeptType>
    __aicore__ __attribute__((always_inline)) static inline bool RefineFewCandidatesGetValueStage(StageRowType& row, StageChunksType& chunks, StageSegmentIndices_Type& segmentIndices_, StageIdx_Type& idx_, StageValid_Type& valid_, StageKeptType& kept, bool& stageResult);

    __aicore__ __attribute__((noinline,aligned(128))) static bool RefineOverlappingFew(uint32_t row,uint32_t chunks,uint64_t& uniqueRows);

    __aicore__ __attribute__((noinline,aligned(128))) static bool RefineWinningFew(uint32_t row,uint32_t chunks,uint64_t& uniqueRows);

    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> FixedFewW(uint32_t k);
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> FixedFewC(uint32_t k);
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> FixedFewP(uint32_t k);

    __aicore__ __attribute__((noinline,aligned(128))) static bool BatchFewCompensated();
    template<auto COUNT, auto pointPitch, auto lanes, class StageRecordType, class StageMaskType, class StageValidType, class StageSmallMaskType, class StageRepeatsType, class StageUnaryType, class StageBinaryType, class StageAxesW0Type, class StageAxesW3Type, class StageAxesW6Type, class StageAxesW9Type, class StageAxesC0Type, class StageAxesC3Type, class StageAxesC10Type, class StageAxesP0Type>
    __aicore__ __attribute__((always_inline)) static inline bool BatchFewCompensatedSubStage(StageRecordType& record, StageMaskType& mask, StageValidType& valid, StageSmallMaskType& smallMask, StageRepeatsType& repeats, StageUnaryType& unary, StageBinaryType& binary, StageAxesW0Type& axesW0, StageAxesW3Type& axesW3, StageAxesW6Type& axesW6, StageAxesW9Type& axesW9, StageAxesC0Type& axesC0, StageAxesC3Type& axesC3, StageAxesC10Type& axesC10, StageAxesP0Type& axesP0);
    template<auto COUNT, auto pointPitch, auto lanes, class StageRecordType, class StageMaskType, class StageValidType, class StageSmallMaskType, class StageRepeatsType, class StageBinaryType, class StageAxesW0Type, class StageAxesW3Type, class StageAxesC3Type, class StageAxesP0Type>
    __aicore__ __attribute__((always_inline)) static inline bool BatchFewCompensatedAddStage2(StageRecordType& record, StageMaskType& mask, StageValidType& valid, StageSmallMaskType& smallMask, StageRepeatsType& repeats, StageBinaryType& binary, StageAxesW0Type& axesW0, StageAxesW3Type& axesW3, StageAxesC3Type& axesC3, StageAxesP0Type& axesP0);


    template<class StageRepeatsType, class StageBinaryType, class StageAxesW0Type, class StageAxesW3Type, class StageAxesW9Type, class StageAxesC10Type>
    __aicore__ __attribute__((always_inline)) static inline void BatchFewCompensatedMulAddDstStage(StageRepeatsType& repeats, StageBinaryType& binary, StageAxesW0Type& axesW0, StageAxesW3Type& axesW3, StageAxesW9Type& axesW9, StageAxesC10Type& axesC10);

    template<class StageRepeatsType, class StageBinaryType, class StageAxesW0Type, class StageAxesW3Type, class StageAxesW6Type, class StageAxesC0Type>
    __aicore__ __attribute__((always_inline)) static inline void BatchFewCompensatedAddStage(StageRepeatsType& repeats, StageBinaryType& binary, StageAxesW0Type& axesW0, StageAxesW3Type& axesW3, StageAxesW6Type& axesW6, StageAxesC0Type& axesC0);

    template<auto lanes, auto COUNT, class StageRecordType, class StageSmallMaskType, class StageRepeatsType>
    __aicore__ __attribute__((always_inline)) static inline bool BatchFewCompensatedDefinedStage(StageRecordType& record, StageSmallMaskType& smallMask, StageRepeatsType& repeats, bool& stageResult);

    __aicore__ static constexpr bool UsesNativeHalfPrefixPrimaryStage2();


    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> BatchIds();
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<uint32_t> BatchPointIds();
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<uint32_t> BatchPointMap();
    // Dead candidate-refinement scratch. No search coefficients, points, row state,
    // candidate IDs or output records overlap these 65 vector planes.
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> EndpointF(uint32_t k);
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<int32_t> EndpointI(uint32_t k);
    // Normalize one signed base-2^15 digit; carry is an arithmetic right shift.
    __aicore__ __attribute__((always_inline)) static inline void EndpointSplit(uint32_t digit);
    // Every finite binary16 coordinate times 2^24 is an exact signed integer.
    // A high part and a signed low part avoid any out-of-range int32 cast.
    __aicore__ __attribute__((always_inline)) static inline void EndpointEncode(LocalTensor<float> x,uint32_t dst);
    __aicore__ __attribute__((noinline,aligned(128))) static bool CorrectHalfLatticePacket(
        uint64_t validLanes,uint64_t near,LocalTensor<int32_t> outputOffset,uint32_t maskAddress);
    template<class StageOutputOffsetType, class StageMasksType, class StageActiveType, class StageFactorType, class StageUnitSquaredType>
    __aicore__ __attribute__((always_inline)) static inline bool CorrectHalfLatticePacketEndpointFStage(StageOutputOffsetType& outputOffset, StageMasksType& masks, StageActiveType& active, StageFactorType& factor, StageUnitSquaredType& unitSquared);
    template<class StageOutputOffsetType, class StageMasksType, class StageActiveType, class StageUnitSquaredType, class StageIntegralType, class StageAdmittedType>
    __aicore__ __attribute__((always_inline)) static inline bool CorrectHalfLatticePacketReinterpretCastStage(StageOutputOffsetType& outputOffset, StageMasksType& masks, StageActiveType& active, StageUnitSquaredType& unitSquared, StageIntegralType& integral, StageAdmittedType& admitted);
    template<class StageOutputOffsetType, class StageMasksType, class StageUnitSquaredType>
    __aicore__ __attribute__((always_inline)) static inline bool CorrectHalfLatticePacketSubStage(StageOutputOffsetType& outputOffset, StageMasksType& masks, StageUnitSquaredType& unitSquared);


    template<class StageOutputOffsetType, class StageMasksType>
    __aicore__ __attribute__((always_inline)) static inline void CorrectHalfLatticePacketMulsStage(StageOutputOffsetType& outputOffset, StageMasksType& masks);


    __aicore__ __attribute__((always_inline)) static inline bool CorrectHalfEndpointPacketTies(
        uint64_t validLanes,uint64_t equal,LocalTensor<int32_t> outputOffset,uint32_t maskAddress);
    template<class StageValidLanesType, class StageOutputOffsetType, class StageMasksType, class StageAmbiguousType>
    __aicore__ __attribute__((always_inline)) static inline bool CorrectHalfEndpointPacketTiesMulsStage(StageValidLanesType& validLanes, StageOutputOffsetType& outputOffset, StageMasksType& masks, StageAmbiguousType& ambiguous);

    template<class StageOutputOffsetType, class StageMasksType, class StageRecordType>
    __aicore__ __attribute__((always_inline)) static inline void CorrectHalfEndpointPacketTiesCastStage(StageOutputOffsetType& outputOffset, StageMasksType& masks, StageRecordType& record);

    template<class StageMasksType>
    __aicore__ __attribute__((always_inline)) static inline void CorrectHalfEndpointPacketTiesEndpointEncodeStage(StageMasksType& masks);


#if defined(DTYPE_TREE_POINTS)
    __aicore__ __attribute__((noinline,aligned(128))) static uint32_t ResolveExactPacketBody(
        uint64_t validLanes,uint32_t pending);
    template<class StageLaneIdsType, class StagePacketStateType, class StageBackupType, class StageShortBitsType, class StageResolvedType, class StageExactLanesType, class StageGroupMaskType, class StageActiveMaskType>
    __aicore__ __attribute__((always_inline)) static inline uint32_t ResolveExactPacketBodyFixedFewPStage2(StageLaneIdsType& laneIds, StagePacketStateType& packetState, StageBackupType& backup, StageShortBitsType& shortBits, StageResolvedType& resolved, StageExactLanesType& exactLanes, StageGroupMaskType& groupMask, StageActiveMaskType& activeMask);

    template<auto pointPitch, class StageLaneIdsType, class StageBackupType, class StageShortBitsType, class StageExactLanesType, class StageArenaType>
    __aicore__ __attribute__((always_inline)) static inline void ResolveExactPacketBodyExactStage(StageLaneIdsType& laneIds, StageBackupType& backup, StageShortBitsType& shortBits, StageExactLanesType& exactLanes, StageArenaType& arena);

    template<class StageBackupType, class StageCertificateType, class StageShortCertificateType>
    __aicore__ __attribute__((always_inline)) static inline void ResolveExactPacketBodyFixedFewPStage(StageBackupType& backup, StageCertificateType& certificate, StageShortCertificateType& shortCertificate);


    __aicore__ __attribute__((always_inline)) static inline uint32_t ResolveExactPacket(
        uint64_t validLanes,uint32_t safeGroups);

    __aicore__ __attribute__((noinline,aligned(128))) static bool ResolveExactSingle(
        uint32_t kept,LocalTensor<float> candidates,LocalTensor<float> record,
        LocalTensor<int32_t> index,LocalTensor<int32_t> offset,LocalTensor<int32_t> outputOffset);
#endif
    __aicore__ __attribute__((noinline,aligned(128))) static bool CorrectPendingCollinearGroups(
        uint64_t validLanes,uint32_t safeGroups,uint32_t groups);
    __aicore__ __attribute__((noinline,aligned(128))) static bool RefineCandidateBatch(uint64_t validLanes,uint64_t rows,uint64_t& uniqueRows);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
    template<class StageValidLanesType, class StageRowsType, class StageUniqueRowsType, class StageOutputOffset_Type, class StageRecord_Type, class StageIdx_Type, class StageTileSecond_Type, class StageSmallMask_Type, class StageDistancesType, class StageIndicesType>
    __aicore__ __attribute__((always_inline)) static inline bool RefineCandidateBatchEndpointIStage(StageValidLanesType& validLanes, StageRowsType& rows, StageUniqueRowsType& uniqueRows, StageOutputOffset_Type& outputOffset_, StageRecord_Type& record_, StageIdx_Type& idx_, StageTileSecond_Type& tileSecond_, StageSmallMask_Type& smallMask_, StageDistancesType& distances, StageIndicesType& indices, bool& stageResult);
#endif

    template<class StageRowsType, class StageIdx_Type, class StageTileSecond_Type, class StageSmallMask_Type, class StageDistancesType, class StageIndicesType>
    __aicore__ __attribute__((always_inline)) static inline bool RefineCandidateBatchCastStage(StageRowsType& rows, StageIdx_Type& idx_, StageTileSecond_Type& tileSecond_, StageSmallMask_Type& smallMask_, StageDistancesType& distances, StageIndicesType& indices);

#if defined(DTYPE_TREE_POINTS)
    template<class StageValidLanesType, class StageRowsType, class StageUniqueRowsType, class StageOutputOffset_Type, class StageRecord_Type, class StageDistancesType>
    __aicore__ __attribute__((always_inline)) static inline void RefineCandidateBatchGatherStage(StageValidLanesType& validLanes, StageRowsType& rows, StageUniqueRowsType& uniqueRows, StageOutputOffset_Type& outputOffset_, StageRecord_Type& record_, StageDistancesType& distances);
#endif

    __aicore__ __attribute__((always_inline)) static inline void ClearCandidateBatch();
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<int32_t> SavedOverlapChunks();
    __aicore__ __attribute__((noinline,aligned(128))) static void BuildOverlapChunks(uint32_t segments);
    __aicore__ __attribute__((noinline,aligned(128))) static uint64_t RefineSparseRows(uint64_t rows,uint32_t validRows,uint32_t segments,uint64_t& uniqueRows);
    template<class StageRowsType, class StageSegmentsType, class StageUniqueRowsType, class StageChunkLower_Type, class StageOutputOffset_Type, class StageOffset_Type, class StageIdx_Type, class StageHi_Type, class StageSegmentIndices_Type, class StageValid_Type, class StageSmallMask_Type, class StageCountType, class StageValidLanesType, class StageGroupRowsType, class StageAcceptedType, class StageActiveType>
    __aicore__ __attribute__((always_inline)) static inline uint64_t RefineSparseRowsGetSFFValueStage2(StageRowsType& rows, StageSegmentsType& segments, StageUniqueRowsType& uniqueRows, StageChunkLower_Type& chunkLower_, StageOutputOffset_Type& outputOffset_, StageOffset_Type& offset_, StageIdx_Type& idx_, StageHi_Type& hi_, StageSegmentIndices_Type& segmentIndices_, StageValid_Type& valid_, StageSmallMask_Type& smallMask_, StageCountType& count, StageValidLanesType& validLanes, StageGroupRowsType& groupRows, StageAcceptedType& accepted, StageActiveType& active);
    template<class StageUniqueRowsType, class StageValid_Type, class StageCountType, class StageValidLanesType, class StageGroupRowsType, class StageAcceptedType, class StageRowType, class StagePackedType, class StageKeptType>
    __aicore__ __attribute__((always_inline)) static inline bool RefineSparseRowsGetSFFValueStage2GatherMaskStage(StageUniqueRowsType& uniqueRows, StageValid_Type& valid_, StageCountType& count, StageValidLanesType& validLanes, StageGroupRowsType& groupRows, StageAcceptedType& accepted, StageRowType& row, StagePackedType& packed, StageKeptType& kept);


    template<class StageSegmentsType, class StageValid_Type, class StageRowType, class StageChunksType, class StagePackedType>
    __aicore__ __attribute__((always_inline)) static inline void RefineSparseRowsGetSFFValueStage(StageSegmentsType& segments, StageValid_Type& valid_, StageRowType& row, StageChunksType& chunks, StagePackedType& packed);

    template<bool MERGED_CERTIFICATE=false>
    __aicore__ __attribute__((always_inline)) static inline bool FixedFewInlineBody();
    template<bool MERGED_CERTIFICATE, auto COUNT, auto lanes, class StageRecordType, class StageMaskType, class StageValidType, class StageSmallMaskType, class StageNormType, class StageRepeatsType, class StageUnaryType, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) static inline bool FixedFewInlineBodyCompareScalarStage(StageRecordType& record, StageMaskType& mask, StageValidType& valid, StageSmallMaskType& smallMask, StageNormType& norm, StageRepeatsType& repeats, StageUnaryType& unary, StageBinaryType& binary);

    template<bool MERGED_CERTIFICATE, auto lanes, class StageRecordType, class StageSmallMaskType, class StageNormType, class StageRepeatsType>
    __aicore__ __attribute__((always_inline)) static inline bool FixedFewInlineBodyWholeReduceMaxStage(StageRecordType& record, StageSmallMaskType& smallMask, StageNormType& norm, StageRepeatsType& repeats, bool& stageResult);

    template<class StageRepeatsType, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) static inline void FixedFewInlineBodySubStage(StageRepeatsType& repeats, StageBinaryType& binary);


    __aicore__ __attribute__((noinline,aligned(128))) static bool FixedFewCompensated();

    template<uint32_t COUNT,bool REJECT_ZERO=false>
    __aicore__ __attribute__((always_inline)) inline bool SparseCompensated();
    template<uint32_t COUNT, bool REJECT_ZERO, auto lanes, class StageRepeatsType, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) inline bool SparseCompensatedMulAddDstStage(StageRepeatsType& repeats, StageBinaryType& binary);

    template<uint32_t COUNT, bool REJECT_ZERO, auto lanes, class StageRepeatsType>
    __aicore__ __attribute__((always_inline)) inline bool SparseCompensatedWholeReduceMaxStage(StageRepeatsType& repeats, bool& stageResult);


    bool exactDirectionLow_=false;
    bool tightCoefficientSafe_=false;
    bool directionLowReady_;
    TreeLineMinDistanceTilingData t_;uint32_t m_,cp_,wp_,pp_,count_,liveRows_;
    GlobalTensor<T> pg_,sg_,dg_;GlobalTensor<int32_t> ig_;
    LocalTensor<T> rawS_,rawP_,outT_;
    LocalTensor<float> columns_,xyz_,work_,sf_,pf_,ones_,segmentIndices_,compactP_,refined_,chunkLower_;
    LocalTensor<float> record_,record2_,lo_,second_,hi_,tileLo_,tileSecond_,tileHi_,cert_,certHigh_;
    LocalTensor<int32_t> so_,idx_,tileIdx_,offset_,rowBase_,pointOffset_,outputOffset_,oneHot_;
    LocalTensor<uint8_t> mask_,valid_,smallMask_;

#endif
