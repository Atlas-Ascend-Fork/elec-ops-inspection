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

template<class T,uint32_t ROWS,uint32_t CAPACITY,uint32_t COEFFICIENTS,bool COMPACT_TAIL,bool SINGLE_TAIL>
class KernelTreeBatchPacked : public KernelTreeBatch<T,ROWS,CAPACITY,COEFFICIENTS,COMPACT_TAIL,SINGLE_TAIL> {
    using Base=KernelTreeBatch<T,ROWS,CAPACITY,COEFFICIENTS,COMPACT_TAIL,SINGLE_TAIL>;
    using Base::t_;
    using Base::m_;
    using Base::cp_;
    using Base::wp_;
    using Base::pp_;
    using Base::count_;
    using Base::liveRows_;
    using Base::pg_;
    using Base::sg_;
    using Base::dg_;
    using Base::ig_;
    using Base::rawP_;
    using Base::pf_;
    using Base::compactP_;
    using Base::refined_;
    using Base::chunkLower_;
    using Base::record_;
    using Base::record2_;
    using Base::hi_;
    using Base::second_;
    using Base::idx_;
    using Base::valid_;
    using Base::smallMask_;
    using Base::mask_;
    using Base::tileSecond_;
    using Base::offset_;
    using Base::pointOffset_;
    using Base::outputOffset_;
    using Base::tileLo_;
    using Base::tileIdx_;
    using Base::segmentIndices_;
    using Base::work_;
    using Base::columns_;
    using Base::xyz_;
    using Base::outT_;
    using Base::Smallest;
    using Base::C;
    using Base::W;
    using Base::P;
    using Base::LoadSegments;
    using Base::Search;
    using Base::Compensated;
    using Base::MakeIndices;
public:
    __aicore__ inline void Process();
    template<class StageEndType, class StageCachedType, class StageCoefficientSafeType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageCertifiedType, class StageActiveType, class StageExactRowsType, class StageUniqueRowsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessCopyStage3(StageEndType& end, StageCachedType& cached, StageCoefficientSafeType& coefficientSafe, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageCertifiedType& certified, StageActiveType& active, StageExactRowsType& exactRows, StageUniqueRowsType& uniqueRows);
    template<class StageEndType, class StageCachedType, class StageCoefficientSafeType, class StagePositionType, class StageBatchType, class StageStartType, class StageValidRowsType, class StageExactRowsType, class StageUniqueRowsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessDefinedStage(StageEndType& end, StageCachedType& cached, StageCoefficientSafeType& coefficientSafe, StagePositionType& position, StageBatchType& batch, StageStartType& start, StageValidRowsType& validRows, StageExactRowsType& exactRows, StageUniqueRowsType& uniqueRows);


    template<class StageValidRowsType, class StageCertifiedType, class StageActiveType, class StagePackedType>
    __aicore__ __attribute__((always_inline)) inline void ProcessRefinePrefixUnpackedRowsStage(StageValidRowsType& validRows, StageCertifiedType& certified, StageActiveType& active, StagePackedType& packed);

    template<class StageRowType, class StageSparseType, class StageSearchColumnsType, class StageSearchCpType>
    __aicore__ __attribute__((always_inline)) inline void ProcessMulsStage(StageRowType& row, StageSparseType& sparse, StageSearchColumnsType& searchColumns, StageSearchCpType& searchCp);

    template<class StageCachedType, class StageCoefficientSafeType, class StageBatchType>
    __aicore__ __attribute__((always_inline)) inline void ProcessLoadPrefixResidentSegmentsStage(StageCachedType& cached, StageCoefficientSafeType& coefficientSafe, StageBatchType& batch);

    template<class StageFirstType, class StageSecondType, class StageHitsType, class StageSparseType>
    __aicore__ __attribute__((always_inline)) inline void ProcessCopyStage2(StageFirstType& first, StageSecondType& second, StageHitsType& hits, StageSparseType& sparse);

#if defined(DTYPE_TREE_POINTS)
    template<class StageEndType, class StageCachedType, class StagePositionType, class StageBatchType, class StageValidRowsType, class StageExactRowsType>
    __aicore__ __attribute__((always_inline)) inline void ProcessTreeGeneralExactFallbackStage(StageEndType& end, StageCachedType& cached, StagePositionType& position, StageBatchType& batch, StageValidRowsType& validRows, StageExactRowsType& exactRows);
#endif

private:
    __aicore__ __attribute__((noinline,aligned(128))) void RefinePrefixUnpackedRows(uint64_t certified,uint32_t validRows);


    __aicore__ __attribute__((noinline,aligned(128))) bool RefinePrefixPacked(uint64_t certified,uint64_t active);
    __aicore__ inline bool RefinePacked(uint64_t certified,uint64_t active);
    template<class StageUncertainType, class StageChunksType, class StageSameType, class StageOriginalColumnsType, class StageOriginalPointsType, class StageOriginalCpType, class StageOriginalPpType>
    __aicore__ __attribute__((always_inline)) inline void RefinePackedDuplicateStage(StageUncertainType& uncertain, StageChunksType& chunks, StageSameType& same, StageOriginalColumnsType& originalColumns, StageOriginalPointsType& originalPoints, StageOriginalCpType& originalCp, StageOriginalPpType& originalPp);

    template<class StageSameType, class StageOriginalColumnsType, class StageOriginalPointsType, class StageOriginalCpType, class StageOriginalPpType, class StageKeptType>
    __aicore__ __attribute__((always_inline)) inline void RefinePackedSmallestStage(StageSameType& same, StageOriginalColumnsType& originalColumns, StageOriginalPointsType& originalPoints, StageOriginalCpType& originalCp, StageOriginalPpType& originalPp, StageKeptType& kept);

    template<class StageSameType, class StageFirstType, class StageQueriesType>
    __aicore__ __attribute__((always_inline)) inline void RefinePackedReinterpretCastStage(StageSameType& same, StageFirstType& first, StageQueriesType& queries);


template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6, class SharedArgument7>
__aicore__ __attribute__((always_inline)) inline void SharedResourceBlock4(SharedArgument0& sparse, SharedArgument1& first, SharedArgument2& hits, SharedArgument3& second, SharedArgument4& candidates, SharedArgument5& row, SharedArgument6& searchColumns, SharedArgument7& searchCp);

template<class ControlArgument0, class ControlArgument1>
__aicore__ __attribute__((always_inline)) inline void ProcessRefinePrefixUnpackedRowsStageControlStage6(ControlArgument0& row, ControlArgument1& candidates);

template<class ControlArgument0, class ControlArgument1>
__aicore__ __attribute__((always_inline)) inline void ProcessRefinePrefixUnpackedRowsStageControlStage7(ControlArgument0& certified, ControlArgument1& row);

template<class ControlArgument0, class ControlArgument1>
__aicore__ __attribute__((always_inline)) inline void RefinePrefixUnpackedRowsControlStage8(ControlArgument0& row, ControlArgument1& candidates);

template<class ControlArgument0, class ControlArgument1>
__aicore__ __attribute__((always_inline)) inline void RefinePrefixUnpackedRowsControlStage9(ControlArgument0& certified, ControlArgument1& row);

template<class SharedArgument0>
__aicore__ __attribute__((always_inline)) inline void CommonVectorStage55(SharedArgument0& row);
};
#include "batch/tree_batch_packed_remaining_2.h"

#include "batch/tree_batch_packed_common_1.h"



#include "batch/tree_batch_packed_process.h"
