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

class TreeM1FixedArena {
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> C(uint32_t col);
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> W(uint32_t col);
    __aicore__ __attribute__((always_inline)) static inline LocalTensor<float> P(uint32_t col);
public:
    __aicore__ __attribute__((always_inline)) static inline void Process(
        GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices,uint32_t pointCount);
    template<class StagePointCountType, class StageBatchType, class StageBeginType, class StageValidRowsType, class StagePgType, class StageSgType, class StageDgType, class StageIgType, class StagePfType, class StageBitmapType, class StageMaskType, class StageOutIdxType, class StageOutType, class StageProjectionType, class StageSameType, class StageCoefficientType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) static inline void ProcessCopyStage4(StagePointCountType& pointCount, StageBatchType& batch, StageBeginType& begin, StageValidRowsType& validRows, StagePgType& pg, StageSgType& sg, StageDgType& dg, StageIgType& ig, StagePfType& pf, StageBitmapType& bitmap, StageMaskType& mask, StageOutIdxType& outIdx, StageOutType& out, StageProjectionType& projection, StageSameType& same, StageCoefficientType& coefficient, StageUnaryType& unary);
    template<class StagePointCountType, class StageBatchType, class StageBeginType, class StageValidRowsType, class StagePgType, class StageSgType, class StageDgType, class StageIgType, class StageMaskType, class StageOutIdxType, class StageOutType, class StageProjectionType, class StageSameType, class StageCoefficientType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) static inline void ProcessAddStage(StagePointCountType& pointCount, StageBatchType& batch, StageBeginType& begin, StageValidRowsType& validRows, StagePgType& pg, StageSgType& sg, StageDgType& dg, StageIgType& ig, StageMaskType& mask, StageOutIdxType& outIdx, StageOutType& out, StageProjectionType& projection, StageSameType& same, StageCoefficientType& coefficient, StageUnaryType& unary);
    template<class StagePointCountType, class StageBatchType, class StageBeginType, class StageValidRowsType, class StagePgType, class StageSgType, class StageDgType, class StageIgType, class StageOutIdxType, class StageOutType, class StageNeedsExactType, class StageSafeType>
    __aicore__ __attribute__((always_inline)) static inline void ProcessGetValueStage3(StagePointCountType& pointCount, StageBatchType& batch, StageBeginType& begin, StageValidRowsType& validRows, StagePgType& pg, StageSgType& sg, StageDgType& dg, StageIgType& ig, StageOutIdxType& outIdx, StageOutType& out, StageNeedsExactType& needsExact, StageSafeType& safe);


    template<class StagePointCountType, class StageBatchType, class StageBeginType, class StageValidRowsType, class StagePgType, class StageSgType, class StageDgType, class StageIgType, class StageNeedsExactType>
    __aicore__ __attribute__((always_inline)) static inline void ProcessTreeGeneralExactFallbackStage2(StagePointCountType& pointCount, StageBatchType& batch, StageBeginType& begin, StageValidRowsType& validRows, StagePgType& pg, StageSgType& sg, StageDgType& dg, StageIgType& ig, StageNeedsExactType& needsExact);

};

#include "tree_m1_fixed_arena_c.h"
