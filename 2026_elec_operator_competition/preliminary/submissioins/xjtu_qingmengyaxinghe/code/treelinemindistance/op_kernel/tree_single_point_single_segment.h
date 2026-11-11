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

template <class T,uint32_t POINT_ROWS=1,uint32_t PAIR_STRIDE=8,bool CONTIG_SEGMENTS=false,bool BLOCK_INDICES=false,bool SINGLE_SEGMENT=false,bool OPTIMIZE_FOUR_ROW=false,uint32_t FIXED_SEGMENTS=0,bool PAIR_SINGLE_POINT=false,bool DEFER_SOC_INIT=false>
class KernelTreeSinglePointSingleSegment {
    static_assert(sizeof(T)==4 && POINT_ROWS==1 && PAIR_STRIDE==8 &&
                  !CONTIG_SEGMENTS && !BLOCK_INDICES && SINGLE_SEGMENT &&
                  !OPTIMIZE_FOUR_ROW && FIXED_SEGMENTS==0 && PAIR_SINGLE_POINT &&
                  !DEFER_SOC_INIT, "Native FP32 single-pair resource required");

    // Zero encodes one point per batch; physical row capacity remains one.
    // Positive values retain the original flattened/partial-row ownership.
    static constexpr uint32_t ROWS=POINT_ROWS?POINT_ROWS:1;
    static constexpr bool STATIC_ONE_POINT=POINT_ROWS==0;
    static constexpr bool THIN_SINGLE_FP32=ROWS==1 && SINGLE_SEGMENT && sizeof(T)==4
        && sizeof(T)==4
        ;
    static constexpr bool NATIVE_SINGLE_POINT=PAIR_SINGLE_POINT && THIN_SINGLE_FP32;
    static constexpr bool NATIVE_ONE_ROW=ROWS==1 && !SINGLE_SEGMENT &&
        ((sizeof(T)==4 && sizeof(T)==4) ||
         (STATIC_ONE_POINT && sizeof(T)==2 && sizeof(T)==2))
        ;
    static constexpr bool NATIVE_FOUR_ROW_SINGLE=OPTIMIZE_FOUR_ROW && ROWS==4 && PAIR_STRIDE==1 && SINGLE_SEGMENT && sizeof(T)==4
        && sizeof(T)==4
        ;
    static constexpr bool NATIVE_HALF_PACKET=FIXED_SEGMENTS==2 && ROWS==1 && NATIVE_ONE_ROW && sizeof(T)==2;
    static constexpr bool COMPLETE_ENDPOINT_LOAD=NATIVE_FOUR_ROW_SINGLE ||
        (FIXED_SEGMENTS==2 && ROWS==4 && !CONTIG_SEGMENTS && sizeof(T)==4
         && sizeof(T)==4
        );
    static constexpr bool MAPPED_TINY_STARTS=NATIVE_ONE_ROW;
public:
    __aicore__ inline void Init(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                               const TreeLineMinDistanceTilingData& t);
    __aicore__ inline void Process();
    __aicore__ __attribute__((always_inline)) inline void ProcessSetMaskNormStage();

private:
    __aicore__ inline void DyadicLatticeFlags(const LocalTensor<float>& left,
            const LocalTensor<float>& right,uint32_t coordinates,bool packedInputs=false);

    __aicore__ inline bool TryExactDyadicGeometry(uint32_t validRows,uint32_t segments,bool inputReady=false);
    template<class StageInputReadyType, class StagePairsType, class StageCoordinatesType, class StageLiveCoordinatesType, class StageLivePairsType, class StageLeftType, class StageRightType, class StageDeltaType, class StageRelAType, class StageRelCType, class StageTempType, class StageProductType, class StageNormType, class StageDotType, class StageEndpointType, class StageBoxType, class StageUpperType, class StageFlagsType, class StageWordsType, class StageAdmittedType>
    __aicore__ __attribute__((always_inline)) inline bool TryExactDyadicGeometryGetValueStage(StageInputReadyType& inputReady, StagePairsType& pairs, StageCoordinatesType& coordinates, StageLiveCoordinatesType& liveCoordinates, StageLivePairsType& livePairs, StageLeftType& left, StageRightType& right, StageDeltaType& delta, StageRelAType& relA, StageRelCType& relC, StageTempType& temp, StageProductType& product, StageNormType& norm, StageDotType& dot, StageEndpointType& endpoint, StageBoxType& box, StageUpperType& upper, StageFlagsType& flags, StageWordsType& words, StageAdmittedType& admitted);
    template<class StagePairsType, class StageCoordinatesType, class StageLivePairsType, class StageLeftType, class StageRightType, class StageDeltaType, class StageRelAType, class StageRelCType, class StageTempType, class StageProductType, class StageNormType, class StageDotType, class StageEndpointType, class StageUpperType, class StageFlagsType, class StageWordsType, class StageAxisPairsType, class StageEndpointsType>
    __aicore__ __attribute__((always_inline)) inline bool TryExactDyadicGeometryReinterpretCastStage(StagePairsType& pairs, StageCoordinatesType& coordinates, StageLivePairsType& livePairs, StageLeftType& left, StageRightType& right, StageDeltaType& delta, StageRelAType& relA, StageRelCType& relC, StageTempType& temp, StageProductType& product, StageNormType& norm, StageDotType& dot, StageEndpointType& endpoint, StageUpperType& upper, StageFlagsType& flags, StageWordsType& words, StageAxisPairsType& axisPairs, StageEndpointsType& endpoints, bool& stageResult);

    template<class StagePairsType, class StageCoordinatesType, class StageLivePairsType, class StageLeftType, class StageRightType, class StageRelCType, class StageTempType, class StageNormType, class StageEndpointType, class StageUpperType, class StageFlagsType, class StageWordsType, class StageEndpointsType, class StageUType, class StageVType, class StageCrossNormType, class StageZeroPairsType, class StageFirstPairsType>
    __aicore__ __attribute__((always_inline)) inline bool TryExactDyadicGeometryDuplicateStage(StagePairsType& pairs, StageCoordinatesType& coordinates, StageLivePairsType& livePairs, StageLeftType& left, StageRightType& right, StageRelCType& relC, StageTempType& temp, StageNormType& norm, StageEndpointType& endpoint, StageUpperType& upper, StageFlagsType& flags, StageWordsType& words, StageEndpointsType& endpoints, StageUType& u, StageVType& v, StageCrossNormType& crossNorm, StageZeroPairsType& zeroPairs, StageFirstPairsType& firstPairs, bool& stageResult);


    template<class StagePairsType, class StageCoordinatesType, class StageLivePairsType, class StageLeftType, class StageRightType, class StageProductType, class StageBoxType, class StageFlagsType, class StageAxisPairsType>
    __aicore__ __attribute__((always_inline)) inline bool TryExactDyadicGeometryMinStage(StagePairsType& pairs, StageCoordinatesType& coordinates, StageLivePairsType& livePairs, StageLeftType& left, StageRightType& right, StageProductType& product, StageBoxType& box, StageFlagsType& flags, StageAxisPairsType& axisPairs, bool& stageResult);

    __aicore__ inline void ExactFallback(uint32_t position,uint32_t batch,uint32_t rows,uint32_t segments);
    __aicore__ inline bool CertifiedTiny(uint32_t validRows,uint32_t segments);
    __aicore__ inline void ReduceTiny();
    template<class U> __aicore__ inline LocalTensor<U> Alloc(uint32_t& address,uint32_t n);
    __aicore__ inline LocalTensor<float> W(uint32_t k);
    template<int32_t STEP=1> __aicore__ inline void MakeIndices(const LocalTensor<int32_t>& v,uint32_t n);


    static constexpr uint32_t slot_=32/sizeof(T);
    TreeLineMinDistanceTilingData t_;uint32_t rows_,pairs_,count_;
    GlobalTensor<T> pg_,sg_,dg_;GlobalTensor<int32_t> ig_;
    LocalTensor<T> rawP_,rawS_,outT_;
    LocalTensor<float> fp_,fs_,p_,a_,c_,work_,lengths_,projection_,scores_,records_,out_,zeroGeometry_,zeroExact_;
    LocalTensor<float> originalA_;
    LocalTensor<int32_t> pointOffsets_,segmentOffsets_,indices_;
    LocalTensor<uint8_t> masks_;
};

#include "tiny/tree_single_point_single_segment_init.h"
