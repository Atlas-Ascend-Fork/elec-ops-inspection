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
// Resource declaration and inline implementation modules.

template <class T, uint32_t MODE, bool SINGLE_TILE=false,uint32_t FIXED_ROWS=0,uint32_t FIXED_STRIDE=0>
class KernelTreePairs {
    static constexpr bool NATIVE_HALF_ENDPOINT_ORDER=SINGLE_TILE && sizeof(T)==2 &&
        ((MODE==4 && (FIXED_ROWS==0 || FIXED_ROWS==16 || FIXED_ROWS==32)) ||
         (MODE==1 && FIXED_STRIDE==32 && FIXED_ROWS>0 && FIXED_ROWS<=8));
    static constexpr bool TILE_COEFFICIENTS=MODE==1 && SINGLE_TILE;
    static constexpr bool PACKED_HALF_ROW_OUTPUT=sizeof(T)==2 && TILE_COEFFICIENTS && FIXED_STRIDE==32 && FIXED_ROWS>0 && FIXED_ROWS<=8;
    static constexpr bool DIRECT_HALF_ROW_POINTS=sizeof(T)==2 && MODE==4 && SINGLE_TILE && FIXED_ROWS==32 && FIXED_STRIDE==64;
    static constexpr bool DIRECT_SINGLE_QUERY_POINTS=sizeof(T)==2 && MODE==8 && FIXED_ROWS==1 && FIXED_STRIDE==1024;
    static constexpr bool SHORT_FP32_METADATA=sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ;
    static constexpr bool MAPPED_STARTS=(MODE==1 && SINGLE_TILE && FIXED_ROWS==4 && FIXED_STRIDE==32 && SHORT_FP32_METADATA) ||
        (MODE==4 && SINGLE_TILE && sizeof(T)==2
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==2
#endif
        );
    static constexpr bool OWNED_AXES8=MODE==2 && SINGLE_TILE && SHORT_FP32_METADATA;
    static constexpr bool NATIVE_SINGLE_SEGMENT_GEOMETRY=MODE==3 && sizeof(T)==4
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==4
#endif
        ;
    static constexpr bool BATCH_COORDINATE_AXES=(MAPPED_STARTS && SHORT_FP32_METADATA) || OWNED_AXES8;
    static constexpr bool PACKED_HALF_GEOMETRY_AXES=MODE==4 && SINGLE_TILE && sizeof(T)==2
#ifdef DTYPE_TREE_POINTS
        && sizeof(DTYPE_TREE_POINTS)==2
#endif
        ;
public:
    __aicore__ inline void Init(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                               const TreeLineMinDistanceTilingData &t);
    template<class StageAddressType>
    __aicore__ __attribute__((always_inline)) inline void InitAllocateStage(StageAddressType& address);

    __aicore__ __attribute__((always_inline)) inline void InitMakeIndicesStage2();

    __aicore__ __attribute__((always_inline)) inline void InitWStage();

    template<class StageIndexType, class StageTmpType>
    __aicore__ __attribute__((always_inline)) inline void InitMakeIndicesStage(StageIndexType& index, StageTmpType& tmp);

    __aicore__ inline void Process();
    __aicore__ __attribute__((always_inline)) inline void ProcessGetBlockIdxStage2();

    template<class StageGroupsType, class StageValidRowsType, class StageDistanceStrideType, class StageIndexStrideType>
    __aicore__ __attribute__((always_inline)) inline void ProcessGatherMaskStage(StageGroupsType& groups, StageValidRowsType& validRows, StageDistanceStrideType& distanceStride, StageIndexStrideType& indexStride);

    __aicore__ __attribute__((always_inline)) inline void ProcessGetBlockIdxStage();

private:
    __aicore__ inline void RepairRange(uint32_t batch,uint32_t start,uint32_t rows);
    template<bool FAST_FLOAT_DIFFERENCES=false>
    __aicore__ inline bool PackedRowsAreUnique(LocalTensor<uint8_t> possible);
    template<bool FAST_FLOAT_DIFFERENCES, class StagePossibleType>
    __aicore__ __attribute__((always_inline)) inline bool PackedRowsAreUniqueReinterpretCastStage(StagePossibleType& possible, bool& stageResult);

    template<bool FAST_FLOAT_DIFFERENCES=false>
    __aicore__ inline void CheckNumericalRisk();
#if defined(DTYPE_TREE_POINTS)

    __aicore__ __attribute__((always_inline)) inline bool CheckNumericalRiskGetValueStage3();
#endif

#if defined(DTYPE_TREE_POINTS)

    __aicore__ __attribute__((always_inline)) inline void CheckNumericalRiskCompareScalarStage();
#endif

#if defined(DTYPE_TREE_POINTS)
    template<bool FAST_FLOAT_DIFFERENCES, class StagePossibleType>
    __aicore__ __attribute__((always_inline)) inline bool CheckNumericalRiskMulsStage(StagePossibleType& possible);
#endif

#if defined(DTYPE_TREE_POINTS)
    template<class StageBitsType>
    __aicore__ __attribute__((always_inline)) inline void CheckNumericalRiskGetValueStage2(StageBitsType& bits);
#endif

#if defined(DTYPE_TREE_POINTS)
    template<class StagePossibleType, class StageBitsType>
    __aicore__ __attribute__((always_inline)) inline void CheckNumericalRiskGetValueStage(StagePossibleType& possible, StageBitsType& bits);
#endif

    __aicore__ inline void PrepareMetadata();
    __aicore__ __attribute__((always_inline)) inline void PrepareMetadataWStage();
    __aicore__ static constexpr bool UsesTileCoefficientsStage();


    __aicore__ inline void MakeIndices(const LocalTensor<int32_t>& values, int32_t first, uint32_t count);
    __aicore__ static constexpr bool UsesModeStage();

    template <class U>
    __aicore__ inline LocalTensor<U> Allocate(uint32_t &address, uint32_t elements);
    __aicore__ inline LocalTensor<float> RawC(uint32_t col);
    __aicore__ inline LocalTensor<float> C(uint32_t col);
    __aicore__ inline LocalTensor<float> W(uint32_t col);
    __aicore__ inline LocalTensor<float> P(uint32_t col);
    __aicore__ __attribute__((always_inline)) inline uint32_t LoadSegmentInputs(
        uint32_t batch,uint32_t pointStart,uint32_t validRows);
    __aicore__ inline void LoadSegments(uint32_t batch,uint32_t pointStart,uint32_t validRows,bool prepareMetadata);
    template<class StageExpandedCountType>
    __aicore__ __attribute__((always_inline)) inline void LoadSegmentsCopyStage(StageExpandedCountType& expandedCount);

    __aicore__ __attribute__((always_inline)) inline void LoadSegmentsBrcbStage();

    template<class StagePrepareMetadataType, class StageExpandedCountType>
    __aicore__ __attribute__((always_inline)) inline void LoadSegmentsReinterpretCastStage(StagePrepareMetadataType& prepareMetadata, StageExpandedCountType& expandedCount);

    __aicore__ inline void CheckExactTileDifferences(uint32_t pointCount);
    template<class StagePointCountType>
    __aicore__ __attribute__((always_inline)) inline void CheckExactTileDifferencesWStage(StagePointCountType& pointCount);

    __aicore__ inline void CheckExactDifferences(uint32_t pointCount);
    template<class StagePointCountType>
    __aicore__ __attribute__((always_inline)) inline void CheckExactDifferencesWStage(StagePointCountType& pointCount);

    __aicore__ __attribute__((always_inline)) inline void PrepareTileGeometry();
    template<auto lanes, class StageUnaryType, class StageSameType, class StageAxisPitchType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void PrepareTileGeometryWStage(StageUnaryType& unary, StageSameType& same, StageAxisPitchType& axisPitch, StageAxesType& axes);

    __aicore__ static constexpr bool UsesTileCoefficientsStage2();


    template<class StageSameType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void PrepareTileGeometrySelectStage2(StageSameType& same, StageAxesType& axes);

    template<class StageSameType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void PrepareTileGeometrySelectStage(StageSameType& same, StageAxesType& axes);

    template<auto lanes, class StageSameType, class StageKeepType, class StageEqualType, class StageLessType, class StageEqualYType, class StageHasXTiesType>
    __aicore__ __attribute__((always_inline)) inline void PrepareTileGeometryCompareStage(StageSameType& same, StageKeepType& keep, StageEqualType& equal, StageLessType& less, StageEqualYType& equalY, StageHasXTiesType& hasXTies);

    template<auto lanes, class StageSameType, class StageEqualType, class StageHasXTiesType>
    __aicore__ __attribute__((always_inline)) inline void PrepareTileGeometryReinterpretCastStage(StageSameType& same, StageEqualType& equal, StageHasXTiesType& hasXTies);

    __aicore__ static constexpr bool UsesModeStage2();


    template<auto lanes, class StageSameType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void PrepareTileGeometrySubStage(StageSameType& same, StageAxesType& axes);

    __aicore__ __attribute__((always_inline)) inline void PrepareSingleSegmentGeometryNative();
    template<bool EXACT> __aicore__ inline void PrepareGeometry(bool canonicalize=true);
    template<bool EXACT, class StageCanonicalizeType>
    __aicore__ __attribute__((always_inline)) inline void PrepareGeometryWStage(StageCanonicalizeType& canonicalize);

    template<class StageKeepType>
    __aicore__ __attribute__((always_inline)) inline void PrepareGeometrySetMaskNormStage(StageKeepType& keep);

    template<bool EXACT>
    __aicore__ __attribute__((always_inline)) inline void PrepareGeometrySubStage();

    __aicore__ inline void LoadPoints(uint32_t batch, uint32_t start, uint32_t validRows);
    template<bool MAP_ORIGINAL=false> __aicore__ inline void ReducePairs();

    __aicore__ static constexpr bool UsesSingleTileStage();


    __aicore__ __attribute__((always_inline)) inline void ReducePairsWStage();
    __aicore__ static constexpr bool UsesSingleTileStage2();


    __aicore__ inline void SelectCandidates();
    template<class StageSelectedType>
    __aicore__ __attribute__((always_inline)) inline void SelectCandidatesCopyStage(StageSelectedType& selected);

    template<bool EXACT_DIFFERENCES=false,bool HALF_SELECTED_PACKET=false>
    __aicore__ __attribute__((always_inline)) inline void Compute();
    template<bool EXACT_DIFFERENCES, bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageUnaryType, class StageBinaryType, class StageCoefficientType, class StagePointCoefficientType, class StagePointBinaryType, class StageAxisPitchType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void ComputeUint8_tStage(StageRepeatsType& repeats, StageUnaryType& unary, StageBinaryType& binary, StageCoefficientType& coefficient, StagePointCoefficientType& pointCoefficient, StagePointBinaryType& pointBinary, StageAxisPitchType& axisPitch, StageAxesType& axes);
    template<bool EXACT_DIFFERENCES, bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageUnaryType, class StageBinaryType, class StagePointCoefficientType, class StageAxesType, class StagePointAxesType>
    __aicore__ __attribute__((always_inline)) inline void ComputeAddStage2(StageRepeatsType& repeats, StageUnaryType& unary, StageBinaryType& binary, StagePointCoefficientType& pointCoefficient, StageAxesType& axes, StagePointAxesType& pointAxes);
    template<bool EXACT_DIFFERENCES, bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageBinaryType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void ComputeAddStage2AddStage(StageRepeatsType& repeats, StageBinaryType& binary, StageAxesType& axes);


    __aicore__ __attribute__((always_inline)) inline void ComputeWholeReduceMinStage();

    template<bool EXACT_DIFFERENCES, bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageBinaryType, class StageCoefficientType, class StagePointBinaryType, class StageAxesType, class StagePointAxesType>
    __aicore__ __attribute__((always_inline)) inline void ComputeSubStage2(StageRepeatsType& repeats, StageBinaryType& binary, StageCoefficientType& coefficient, StagePointBinaryType& pointBinary, StageAxesType& axes, StagePointAxesType& pointAxes);

    template<bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageBinaryType, class StageCoefficientType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void ComputeAddStage(StageRepeatsType& repeats, StageBinaryType& binary, StageCoefficientType& coefficient, StageAxesType& axes);

    template<bool EXACT_DIFFERENCES, bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageCoefficientType, class StageProjectionAxesType>
    __aicore__ __attribute__((always_inline)) inline void ComputeMulAddDstStage(StageRepeatsType& repeats, StageCoefficientType& coefficient, StageProjectionAxesType& projectionAxes);

    template<bool EXACT_DIFFERENCES, bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageBinaryType, class StagePointCoefficientType, class StageAxesType, class StagePointAxesType>
    __aicore__ __attribute__((always_inline)) inline void ComputeSubStage(StageRepeatsType& repeats, StageBinaryType& binary, StagePointCoefficientType& pointCoefficient, StageAxesType& axes, StagePointAxesType& pointAxes);


    __aicore__ __attribute__((always_inline)) inline void ComputeCompareStage();

    TreeLineMinDistanceTilingData t_;
    uint32_t stride_, rows_, count_, allocCount_, group_, selectedCount_, validPointRows_, validComputeRows_;
    bool exactDifferences_, nativeHalfXTies_, needsExact_;
    GlobalTensor<T> pg_, sg_, dg_;
    GlobalTensor<int32_t> ig_;
    LocalTensor<T> rawS_, rawP_, outT_, stageOut_;
    LocalTensor<float> sf_, pf_, pointSlots_, columns_, xyz_, work_, records_, reduceWork_, out_, zeroBound_, zeroExact_;
    LocalTensor<int32_t> po_, so_, outputOffsets_, outIdx_, reduceOffsets_, reduceBases_, stageIdx_;
    LocalTensor<uint8_t> mask_, valid_;

template<class SharedArgument0, class SharedArgument1>
__aicore__ __attribute__((always_inline)) inline void SharedSegmentPadding9(SharedArgument0& index, SharedArgument1& tmp);

__aicore__ __attribute__((always_inline)) inline void SharedSegmentMetadata16();

template<auto maskWords, class ControlArgument0, class ControlArgument1, class ControlArgument2, class ControlArgument4>
__aicore__ __attribute__((always_inline)) inline void LoadSegmentsReinterpretCastStageControlStage2(ControlArgument0& lanes, ControlArgument1& bitmap, ControlArgument2& axis, ControlArgument4& words);

__aicore__ __attribute__((always_inline)) inline void ComputeCompareStageControlStage3();

__aicore__ __attribute__((always_inline)) inline void ComputeCompareStageControlStage4();

template<class ControlArgument0>
__aicore__ __attribute__((always_inline)) inline void ComputeCompareStageControlStage5(ControlArgument0& nearZero);

template<class SharedArgument0>
__aicore__ __attribute__((always_inline)) inline void CommonVectorStage52(SharedArgument0& index);

__aicore__ __attribute__((always_inline)) inline void CommonVectorStage56();

template<auto lanes, class ControlArgument0, class ControlArgument1>
__aicore__ __attribute__((always_inline)) inline void PrepareTileGeometrySubStageControlStage61(ControlArgument0& axes, ControlArgument1& same);
};
#include "pairs/tree_pairs_remaining_2.h"

#include "pairs/tree_pairs_remaining_1.h"

#include "pairs/tree_pairs_common_1.h"



#include "pairs/tree_pairs_init.h"
#include "pairs/tree_pairs_check_numerical_risk.h"
#include "pairs/tree_pairs_load_segments_reinterpret_cast.h"
#include "pairs/tree_pairs_prepare_geometry.h"
#include "pairs/tree_pairs_compute_add.h"
