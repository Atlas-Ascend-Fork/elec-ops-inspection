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

template<class T, uint32_t ROWS=64, uint32_t CAPACITY=1024, uint32_t COEFFICIENTS=CAPACITY,bool COMPACT_TAIL=false,bool SINGLE_TAIL=false>
// Power-of-two segment capacity bounds the shared coefficient cache.
class KernelTreeBatch {
#include "tree_batch_api_1.h"
#include "tree_batch_api_2.h"
template<bool HALF_RESOURCE, class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5>
__aicore__ __attribute__((always_inline)) static inline void SharedPrefixPredicate1(SharedArgument0& offset_, SharedArgument1& idx_, SharedArgument2& oneHot_, SharedArgument3& cert_, SharedArgument4& certHigh_, SharedArgument5& valid_);

template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5>
__aicore__ __attribute__((always_inline)) inline void SharedSparseWinner2(SharedArgument0& sparse, SharedArgument1& first, SharedArgument2& hits, SharedArgument3& second, SharedArgument4& logicalSegments, SharedArgument5& candidates);

template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6, class SharedArgument7>
__aicore__ __attribute__((always_inline)) inline void SharedChunkUnion3(SharedArgument0& collisions, SharedArgument1& sourceBits, SharedArgument2& same, SharedArgument3& nextUnions, SharedArgument4& unions, SharedArgument5& nextCollisions, SharedArgument6& mergedCollisions, SharedArgument7& unary);

template<auto COUNT, class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3>
__aicore__ __attribute__((always_inline)) static inline void SharedFixedResidualNorm5(SharedArgument0& mask, SharedArgument1& binary, SharedArgument2& repeats, SharedArgument3& valid);
template<class SharedArgument0, class SharedArgument1>
__aicore__ __attribute__((always_inline)) inline void SharedSparseProjection7(SharedArgument0& candidates, SharedArgument1& sparse);

template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6, class SharedArgument7, class SharedArgument8>
__aicore__ __attribute__((always_inline)) static inline void SharedResourceBlock10(SharedArgument0& unique, SharedArgument1& uniqueRows, SharedArgument2& row, SharedArgument3& record_, SharedArgument4& tileIdx_, SharedArgument5& smallMask_, SharedArgument6& segmentIndices_, SharedArgument7& tileSecond_, SharedArgument8& idx_);
template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5>
__aicore__ __attribute__((always_inline)) inline void SharedCandidateProjection11(SharedArgument0& candidates, SharedArgument1& sparse, SharedArgument2& row, SharedArgument3& first, SharedArgument4& hits, SharedArgument5& second);
template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6>
__aicore__ __attribute__((always_inline)) static inline void SharedChunkFold12(SharedArgument0& unions, SharedArgument1& sourceBits, SharedArgument2& same, SharedArgument3& collisions, SharedArgument4& nextUnions, SharedArgument5& nextCollisions, SharedArgument6& mergedCollisions);
template<class SharedArgument0>
__aicore__ __attribute__((always_inline)) inline void SharedGuardedMinimum13(SharedArgument0& j);
template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5>
__aicore__ __attribute__((always_inline)) inline void SharedResourceBlock14(SharedArgument0& searchRows, SharedArgument1& unary, SharedArgument2& j, SharedArgument3& segment, SharedArgument4& coarseErrorFactor, SharedArgument5& same);

template<class SharedArgument0, class SharedArgument1, class SharedArgument2>
__aicore__ __attribute__((always_inline)) inline void SharedMergeMinimum15(SharedArgument0& same, SharedArgument1& j, SharedArgument2& unary);

__aicore__ __attribute__((always_inline)) inline void SharedPrefetchFence17();
__aicore__ __attribute__((always_inline)) inline void SharedQueryDifference18();

template<auto pitch, auto pointPitch, auto workBase, class SharedArgument0, class SharedArgument3, class SharedArgument4>
__aicore__ __attribute__((always_inline)) static inline void SharedResidentCoefficients19(SharedArgument0& op, SharedArgument3& segmentCount, SharedArgument4& source);
template<bool HALF_RESOURCE>
__aicore__ __attribute__((always_inline)) static inline void SharedPrefixArena20();

__aicore__ __attribute__((always_inline)) inline void SharedSearchCoefficients21();

template<class ControlArgument0, class ControlArgument1, class ControlArgument2>
__aicore__ __attribute__((always_inline)) inline bool ProcessProcessSavedOverlapChunksStage2StageControlStage40(ControlArgument0& chunks, ControlArgument1& row, ControlArgument2& uniqueRows);

template<class SharedArgument0, class SharedArgument1, class SharedArgument2>
__aicore__ __attribute__((always_inline)) inline void CommonVectorStage51(SharedArgument0& chunks, SharedArgument1& same, SharedArgument2& unary);

template<class SharedArgument0>
__aicore__ __attribute__((always_inline)) inline void CommonVectorStage53(SharedArgument0& j);

template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6>
__aicore__ __attribute__((always_inline)) inline void CommonVectorStage54(SharedArgument0& first, SharedArgument1& second, SharedArgument2& hits, SharedArgument3& sparse, SharedArgument4& candidates, SharedArgument5& row, SharedArgument6& logicalSegments);

__aicore__ __attribute__((always_inline)) inline bool TryCompactSearchTail(uint32_t j);

template<bool FIXED>
__aicore__ __attribute__((always_inline)) inline static LocalTensor<float> CompensatedW(uint32_t axis);
template<bool FIXED>
__aicore__ __attribute__((always_inline)) inline static LocalTensor<float> CompensatedP(uint32_t axis);
template<bool FIXED>
__aicore__ __attribute__((always_inline)) inline static LocalTensor<float> CompensatedC(uint32_t axis);
template<bool FIXED,class Repeats>
__aicore__ __attribute__((always_inline)) inline static void CompensatedDeferred(Repeats repeats,const BinaryRepeatParams& binary);
template<bool FIXED,class Repeats>
__aicore__ __attribute__((always_inline)) inline static void CompensatedImmediate(Repeats repeats,const BinaryRepeatParams& binary);
};
#include "batch/tree_batch_remaining_1.h"

#include "batch/tree_batch_common_1.h"







#include "batch/tree_batch_init.h"
#include "batch/tree_batch_process_defined3.h"
#include "batch/tree_batch_process_abs.h"
#include "batch/tree_batch_search_static_policy_duplicate.h"
#include "batch/tree_batch_search_prefix_state_merge_prefix_minimum2.h"
#include "batch/tree_batch_search_arena.h"
#include "batch/tree_batch_search_guarded_half_cached_rows_compare_scalar.h"
#include "batch/tree_batch_search_guarded_half_type_certified_rows_compare.h"
#include "batch/tree_batch_search_certified_rows_body.h"
#include "batch/tree_batch_search_spatial_static_state.h"
#include "batch/tree_batch_prune_spatial_static_state_set_mask_norm2.h"
#include "batch/tree_batch_primary_compensated_body.h"
#include "batch/tree_batch_certify_refined_body_return.h"
#include "batch/tree_batch_batch_few_compensated_add2.h"
#include "batch/tree_batch_correct_half_endpoint_packet_ties_endpoint_encode.h"
#include "batch/tree_batch_refine_sparse_rows_get_sffvalue2.h"
