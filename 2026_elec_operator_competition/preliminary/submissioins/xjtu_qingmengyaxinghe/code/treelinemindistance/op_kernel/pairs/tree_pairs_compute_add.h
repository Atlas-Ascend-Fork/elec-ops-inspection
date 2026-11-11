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
// Inline implementation of KernelTreePairs.

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageBinaryType, class StageCoefficientType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ComputeAddStage(StageRepeatsType& repeats, StageBinaryType& binary, StageCoefficientType& coefficient, StageAxesType& axes) {
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Add<float,false>(W(6),W(0),W(3),uint64_t(64),3,axes);
            if constexpr(MAPPED_STARTS) Add<float,false>(W(6)[64],W(0)[64],W(3)[64],uint64_t(64),3,axes);
            Sub<float,false>(W(3),W(3),C(0),uint64_t(64),3,axes);
            if constexpr(MAPPED_STARTS) Sub<float,false>(W(3)[64],W(3)[64],C(0),uint64_t(64),3,axes);
        } else {
    #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) {
                Add<float, false>(W(axis+6), W(axis), W(axis+3), uint64_t(64), repeats, binary);
                Sub<float, false>(W(axis+3), W(axis+3), C(axis), uint64_t(64), repeats, coefficient);
            }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool EXACT_DIFFERENCES, bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageCoefficientType, class StageProjectionAxesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ComputeMulAddDstStage(StageRepeatsType& repeats, StageCoefficientType& coefficient, StageProjectionAxesType& projectionAxes) {
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            MulAddDst<float,float,false>(W(0),W(8),C(6),uint64_t(64),3,projectionAxes);
            if constexpr(MAPPED_STARTS) MulAddDst<float,float,false>(W(0)[64],W(8)[64],C(6),uint64_t(64),3,projectionAxes);
            if constexpr(!EXACT_DIFFERENCES) {
                MulAddDst<float,float,false>(W(3),W(8),C(10),uint64_t(64),3,projectionAxes);
                if constexpr(MAPPED_STARTS) MulAddDst<float,float,false>(W(3)[64],W(8)[64],C(10),uint64_t(64),3,projectionAxes);
            }
        } else {
    #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) {
                MulAddDst<float, float, false>(W(axis), W(8), C(axis+6), uint64_t(64), repeats, coefficient);
                if constexpr(!EXACT_DIFFERENCES)
                    MulAddDst<float,float,false>(W(axis+3),W(8),C(axis+10),uint64_t(64),repeats,coefficient);
            }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool EXACT_DIFFERENCES, bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageBinaryType, class StagePointCoefficientType, class StageAxesType, class StagePointAxesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ComputeSubStage(StageRepeatsType& repeats, StageBinaryType& binary, StagePointCoefficientType& pointCoefficient, StageAxesType& axes, StagePointAxesType& pointAxes) {
        if constexpr (MODE != 3 && !EXACT_DIFFERENCES) {
        // When all coordinate differences are exact, t=1 gives P-C in one FMA.
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Sub<float,false>(W(3),P(0),C(3),uint64_t(64),3,pointAxes);
            if constexpr(MAPPED_STARTS) Sub<float,false>(W(3)[64],P(0)[64],C(3),uint64_t(64),3,axes);
        } else {
    #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis+3), P(axis), C(axis+3), uint64_t(64), repeats, pointCoefficient);
        }
        PipeBarrier<PIPE_V>();
        SelectResidualPlanes(W(0), W(1), W(2), W(3), W(4), W(5), mask_, count_==64, repeats, binary);
        PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
__aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ComputeCompareStageControlStage3() {
if constexpr(MAPPED_STARTS) {
                    // All ordinary pair scores have been computed. Direction
                    // columns are dead; restore the raw-endpoint ABI only for
                    // the exact geometric-zero helpers, before the final store.
            #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis)
                        Copy(RawC(axis),C(axis),64,1,CopyRepeatParams(1,1,8,8));
                    PipeBarrier<PIPE_V>();
                }
}

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
__aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ComputeCompareStageControlStage4() {
if constexpr(DIRECT_SINGLE_QUERY_POINTS) {
                    // Geometric-zero helpers retain their expanded-point ABI.
                    // Materialize that view only on entry to those helpers.
            #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis)
                        Copy(xyz_[axis*allocCount_],pointSlots_[axis*8],64,count_/64,CopyRepeatParams(1,0,8,0));
                    PipeBarrier<PIPE_V>();
                }
}

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
template<class ControlArgument0>
__aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ComputeCompareStageControlStage5(ControlArgument0& nearZero) {
if(nearZero&((uint64_t(1)<<rows_)-1)) {
                ComputeCompareStageControlStage3();
                ComputeCompareStageControlStage4();
                const uint32_t predicatePitch=(count_+255)/256*32;
                CorrectAxialZeroScores<MODE==4 || TILE_COEFFICIENTS>(uint32_t(W(0).GetPhyAddr()),uint32_t(xyz_.GetPhyAddr()),allocCount_,
                    uint32_t(columns_.GetPhyAddr()),MODE==4?96:allocCount_,uint32_t(W(1).GetPhyAddr()),allocCount_,
                    uint32_t(po_.GetPhyAddr()),predicatePitch,count_);
                CorrectCollinearZeroScores<sizeof(T)==2>(uint32_t(W(0).GetPhyAddr()),uint32_t(xyz_.GetPhyAddr()),allocCount_,
                    uint32_t(columns_.GetPhyAddr()),MODE==4?96:allocCount_,uint32_t(zeroExact_.GetPhyAddr()),
                    uint32_t(zeroBound_[32].GetPhyAddr()),uint32_t(po_.GetPhyAddr()),count_,MODE==4 || TILE_COEFFICIENTS);
                Select(W(0),valid_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,count_);PipeBarrier<PIPE_V>();
                if constexpr(MODE==8) {
                    if(selectedCount_==1)out_=W(0);else ReducePairs<true>();
                } else ReducePairs<false>();
            }
}

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ComputeCompareStage() {
        if constexpr(MODE==8) {
            if(!(MODE==8 && selectedCount_==1)) {
            Compare(mask_,out_,zeroBound_[32],CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint64_t nearZero=mask_.template ReinterpretCast<uint64_t>().GetValue(0);
            ComputeCompareStageControlStage5(nearZero);
            }
        }
    }
