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
    template<bool EXACT> __aicore__ inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareGeometry(bool canonicalize) {
        if constexpr(EXACT && NATIVE_SINGLE_SEGMENT_GEOMETRY) {
            if(count_==64 && allocCount_/8<=255) {
                PrepareSingleSegmentGeometryNative();
                return;
            }
        }
        PrepareGeometryWStage<EXACT>(canonicalize);
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub(C(axis+6), C(axis+3), C(axis), count_);
        PipeBarrier<PIPE_V>();
        PrepareGeometrySubStage<EXACT>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(EXACT && MODE==8 && sizeof(T)==2) {
            Adds(C(9),W(7),0.0F,count_);
            PipeBarrier<PIPE_V>();
        } else {
#endif
        Mul(C(9), C(6), C(6), count_);
        PipeBarrier<PIPE_V>();
        MulAddDst(C(9), C(7), C(7), count_);
        PipeBarrier<PIPE_V>();
        MulAddDst(C(9), C(8), C(8), count_);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask_, C(9), 0.0F, CMPMODE::GT, count_);
        PipeBarrier<PIPE_V>();
        Select(C(9), mask_, C(9), 1.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, count_);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        }
#endif
        if constexpr(EXACT && (MODE!=3 || sizeof(T)==2)) {
            if(!(MODE==8 && selectedCount_==1)) {
            PipeBarrier<PIPE_V>();
            WholeReduceMax(W(6),C(9),64,count_/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMax(zeroBound_,W(6),count_/64,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            Muls(zeroBound_,zeroBound_,1.8189894035458565e-12F,1);PipeBarrier<PIPE_V>();
            Brcb(zeroBound_[8],zeroBound_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Copy(zeroBound_[32],zeroBound_[8],64,1,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
            }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool EXACT, class StageCanonicalizeType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareGeometryWStage(StageCanonicalizeType& canonicalize) {
        if constexpr(EXACT) {
        if constexpr (MODE != 3) {
        if(canonicalize) {
            // A segment and its reversed representation must follow identical
            // floating-point operations, including the exact-index tie rule.
            auto keep = W(8).template ReinterpretCast<uint8_t>();
            auto equal = W(0).template ReinterpretCast<uint8_t>();
            auto less = W(1).template ReinterpretCast<uint8_t>();
            auto equalY = W(2).template ReinterpretCast<uint8_t>();
            CompareEndpointOrder(keep, equal, less, equalY, C(0), C(1), C(2), C(3), C(4), C(5), count_);
            PrepareGeometrySetMaskNormStage(keep);
    #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Adds(C(axis), W(axis), 0.0F, count_);
            PipeBarrier<PIPE_V>();
        }
        }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<class StageKeepType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareGeometrySetMaskNormStage(StageKeepType& keep) {
        if(count_==64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));SetCmpMask(keep);
            const BinaryRepeatParams same{1,1,1,8,8,8};
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(axis),C(axis),C(axis+3),1,same);
            PipeBarrier<PIPE_V>();
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(C(axis+3),C(axis+3),C(axis),1,same);
            PipeBarrier<PIPE_V>();
        } else {
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select(W(axis),keep,C(axis),C(axis+3),SELMODE::VSEL_TENSOR_TENSOR_MODE,count_);
            PipeBarrier<PIPE_V>();
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Select(C(axis+3),keep,C(axis+3),C(axis),SELMODE::VSEL_TENSOR_TENSOR_MODE,count_);
            PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool EXACT>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::PrepareGeometrySubStage() {
        if constexpr(EXACT) {
        if(!((MODE==4 || TILE_COEFFICIENTS) && sizeof(T)==2 && exactDifferences_)) {
CompensateDirectionPlanes(W(0), W(1), W(2), W(3), W(4), W(5), C(0), C(1), C(2), C(3), C(4), C(5), C(6), C(7), C(8), C(10), C(11), C(12), count_);
        } else {
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Duplicate(C(axis+10),0.0F,64);
        }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::LoadPoints(uint32_t batch, uint32_t start, uint32_t validRows) {
        if constexpr(MODE==3 && sizeof(T)==2) validPointRows_=validRows;
        const uint32_t transferCount=(validRows*3+63)/64*64;
        Duplicate(rawP_, T(0), transferCount);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(rawP_, pg_[(batch * t_.points + start) * 3],
                    DataCopyExtParams{1, validRows * 3 * uint32_t(sizeof(T)), 0, 0, 0},
                    DataCopyPadExtParams<T>{false, 0, 0, T(0)});
        PipeBarrier<PIPE_ALL>();
        if constexpr (sizeof(T) == 2) Cast(pf_, rawP_, RoundMode::CAST_NONE, transferCount);
        PipeBarrier<PIPE_V>();
        if constexpr(MODE==4 && sizeof(T)==2)CheckExactDifferences(transferCount);
        if constexpr (MODE != 3) Brcb(pointSlots_, pf_, (validRows * 3 + 7) / 8, BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool MAP_ORIGINAL> __aicore__ inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ReducePairs() {
        ReducePairsWStage();
        if constexpr (MODE != 3) {
            if constexpr(UsesSingleTileStage()) Gather(out_,records_,outputOffsets_.template ReinterpretCast<uint32_t>(),0U,rows_);
            else if(stride_>64) Gather(out_,records_,outputOffsets_.template ReinterpretCast<uint32_t>(),0U,rows_);
        }
        if constexpr(!(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024) || MAP_ORIGINAL)
            PipeBarrier<PIPE_V>();
        if constexpr(MAP_ORIGINAL) {
            Muls(W(6).template ReinterpretCast<int32_t>(),outIdx_,int32_t(4),1);PipeBarrier<PIPE_V>();
            Gather(outIdx_,so_,W(6).template ReinterpretCast<uint32_t>(),0U,1);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ constexpr bool KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::UsesSingleTileStage() { return (!SINGLE_TILE || (sizeof(T)==2 && MODE!=4 && !PACKED_HALF_ROW_OUTPUT));
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ReducePairsWStage() {
        if constexpr (MODE == 3) {
            out_=W(0);
            if constexpr(!(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024))
                Duplicate(outIdx_, int32_t(0), rows_);
        } else if (stride_ <= 64) {
            WholeReduceMin(records_, W(0), static_cast<int32_t>(stride_), rows_, 1, 1, stride_ / 8,
                           ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            if constexpr(UsesSingleTileStage2()) {
                // Reduction scratch is dead until the single GM index write.
                outIdx_=reduceWork_.template ReinterpretCast<int32_t>();
                GatherMask(outIdx_.template ReinterpretCast<float>(),records_,uint8_t(2),GatherMaskParams(1,1,8,0));
                GatherMask(out_,records_,uint8_t(1),GatherMaskParams(1,1,8,0));
            } else Gather(outIdx_, records_.template ReinterpretCast<int32_t>(),
                          outputOffsets_.template ReinterpretCast<uint32_t>(), 4U, rows_);
        } else {
            const uint32_t parts = stride_ / 64;
            const uint32_t partStride = parts < 8 ? 8 : parts;
            WholeReduceMin(reduceWork_, W(0), 64, count_ / 64, 1, 1, 8,
                           ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            Gather(W(6), reduceWork_, reduceOffsets_.template ReinterpretCast<uint32_t>(), 0U,
                   rows_ * partStride);
            PipeBarrier<PIPE_V>();
            WholeReduceMin(records_, W(6), static_cast<int32_t>(parts), rows_, 1, 1, partStride / 8,
                           ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            auto block = W(7).template ReinterpretCast<int32_t>();
            Gather(block, records_.template ReinterpretCast<int32_t>(),
                   outputOffsets_.template ReinterpretCast<uint32_t>(), 4U, rows_);
            PipeBarrier<PIPE_V>();
            Muls(outIdx_, block, int32_t(8), rows_);
            Muls(block, block, int32_t(64), rows_);
            PipeBarrier<PIPE_V>();
            auto chosenOffsets = W(8).template ReinterpretCast<int32_t>();
            Add(chosenOffsets, outIdx_, reduceBases_, rows_);
            PipeBarrier<PIPE_V>();
            Gather(outIdx_, reduceWork_.template ReinterpretCast<int32_t>(),
                   chosenOffsets.template ReinterpretCast<uint32_t>(), 0U, rows_);
            PipeBarrier<PIPE_V>();
            Add(outIdx_, outIdx_, block, rows_);
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ constexpr bool KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::UsesSingleTileStage2() { return (SINGLE_TILE && (sizeof(T)==4 || MODE==4 || PACKED_HALF_ROW_OUTPUT));
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::SelectCandidates() {
        const uint8_t repeats=count_/64;
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        const UnaryRepeatParams unary{1,1,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(W(axis),P(axis),C(axis),uint64_t(64),repeats,BinaryRepeatParams(1,DIRECT_SINGLE_QUERY_POINTS?0:1,1,8,DIRECT_SINGLE_QUERY_POINTS?0:8,8));
        PipeBarrier<PIPE_V>();
        Mul<float,false>(W(3),W(0),C(6),uint64_t(64),repeats,binary);
        Mul<float,false>(W(4),W(0),W(0),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(W(3),W(1),C(7),uint64_t(64),repeats,binary);
        MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(W(3),W(2),C(8),uint64_t(64),repeats,binary);
        MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        Div<float,false>(W(6),W(3),C(9),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        Maxs<float,false>(W(6),W(6),0.0F,uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        Mins<float,false>(W(6),W(6),1.0F,uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        Mul<float,false>(W(7),W(6),C(9),uint64_t(64),repeats,binary);
        Muls<float,false>(W(3),W(3),-2.0F,uint64_t(64),repeats,unary);
        Muls<float,false>(W(5),W(4),1.52587890625e-5F,uint64_t(64),repeats,unary);PipeBarrier<PIPE_V>();
        Add<float,false>(W(7),W(7),W(3),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(W(4),W(6),W(7),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        Sub<float,false>(W(0),W(4),W(5),uint64_t(64),repeats,binary);
        Add<float,false>(W(5),W(4),W(5),uint64_t(64),repeats,binary);PipeBarrier<PIPE_V>();
        Select(W(0),valid_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,count_);PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        // The minimum of conservative upper bounds is itself an incumbent.
        // Every possible winner has a lower bound no greater than this value.
        // Padding must be excluded from both sides of the interval reduction.
        Select(W(5),valid_,W(5),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,count_);PipeBarrier<PIPE_V>();
        WholeReduceMin(records_,W(5),64,count_/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(W(6),records_,count_/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
#else
        ReducePairs<false>();
        Muls(W(6).template ReinterpretCast<int32_t>(),outIdx_,int32_t(4),1);PipeBarrier<PIPE_V>();
        Gather(W(6),W(5),W(6).template ReinterpretCast<uint32_t>(),0U,1);PipeBarrier<PIPE_V>();
#endif
        Brcb(W(7),W(6),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Copy(W(8),W(7),64,count_/64,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
        Compare(mask_,W(0),W(8),CMPMODE::LE,count_);
        Duplicate(so_,int32_t(0),count_);PipeBarrier<PIPE_V>();
        uint64_t kept=0;
        GatherMask(so_.template ReinterpretCast<float>(),po_.template ReinterpretCast<float>(),
                   mask_.template ReinterpretCast<uint32_t>(),true,stride_,GatherMaskParams(1,1,8,1),kept);
        PipeBarrier<PIPE_V>();
        uint32_t selected=uint32_t(kept);
        SelectCandidatesCopyStage(selected);
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<class StageSelectedType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::SelectCandidatesCopyStage(StageSelectedType& selected) {
        if(selected==0 || selected>t_.segments) {
            Copy(so_,po_,64,count_/64,CopyRepeatParams(1,1,8,8));selected=t_.segments;PipeBarrier<PIPE_V>();
        }
        selectedCount_=selected;
        stride_=8;while(stride_<selected)stride_*=2;
        count_=stride_<64?64:stride_;
        Muls(W(6).template ReinterpretCast<int32_t>(),so_,int32_t(24),count_);PipeBarrier<PIPE_V>();
#pragma unroll
        for(uint32_t axis=0;axis<6;++axis)
            Gather(C(axis),sf_,W(6).template ReinterpretCast<uint32_t>(),axis*4,count_);
        PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
        if constexpr(MODE==8 && sizeof(T)==2) {
            Muls(W(6).template ReinterpretCast<int32_t>(),so_,int32_t(4),count_);PipeBarrier<PIPE_V>();
            Gather(W(7),C(9),W(6).template ReinterpretCast<uint32_t>(),0U,count_);PipeBarrier<PIPE_V>();
        }
#endif
        PrepareGeometry<true>(selected>1);
        Cast(W(8),po_,RoundMode::CAST_NONE,count_);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,W(8),float(int32_t(selected)),CMPMODE::LT,count_);PipeBarrier<PIPE_V>();

    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool EXACT_DIFFERENCES, bool HALF_SELECTED_PACKET>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::Compute() {
        const uint8_t repeats = HALF_SELECTED_PACKET ? 1 : count_ / 64;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        const BinaryRepeatParams coefficient{1,1,1,8,8,(MODE == 3 || MODE == 4 || TILE_COEFFICIENTS) ? 0 : 8};
        // One 64-lane segment packet corresponds to one query. Broadcast
        // its existing eight-value slot directly, with three slots per query.
        const BinaryRepeatParams pointCoefficient{1,(DIRECT_HALF_ROW_POINTS || DIRECT_SINGLE_QUERY_POINTS)?0:1,1,8,DIRECT_SINGLE_QUERY_POINTS?0:DIRECT_HALF_ROW_POINTS?3:8,(MODE==3 || MODE==4 || TILE_COEFFICIENTS)?0:8};
        const BinaryRepeatParams pointBinary{1,(DIRECT_HALF_ROW_POINTS || DIRECT_SINGLE_QUERY_POINTS)?0:1,1,8,DIRECT_SINGLE_QUERY_POINTS?0:DIRECT_HALF_ROW_POINTS?3:8,8};
        const uint8_t axisPitch=uint8_t(allocCount_/8);
        const BinaryRepeatParams axes{1,1,1,axisPitch,axisPitch,axisPitch};
        ComputeUint8_tStage<EXACT_DIFFERENCES, HALF_SELECTED_PACKET>(repeats, unary, binary, coefficient, pointCoefficient, pointBinary, axisPitch, axes);
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool EXACT_DIFFERENCES, bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageUnaryType, class StageBinaryType, class StageCoefficientType, class StagePointCoefficientType, class StagePointBinaryType, class StageAxisPitchType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ComputeUint8_tStage(StageRepeatsType& repeats, StageUnaryType& unary, StageBinaryType& binary, StageCoefficientType& coefficient, StagePointCoefficientType& pointCoefficient, StagePointBinaryType& pointBinary, StageAxisPitchType& axisPitch, StageAxesType& axes) {
        const BinaryRepeatParams pointAxes{1,DIRECT_SINGLE_QUERY_POINTS?0:1,1,axisPitch,uint8_t(DIRECT_SINGLE_QUERY_POINTS?1:axisPitch),axisPitch};
        const BinaryRepeatParams projectionAxes{1,1,1,axisPitch,0,axisPitch};
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(uint64_t(64));
        // Error-free subtraction preserves short distances at large coordinates.
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Sub<float,false>(W(0),P(0),C(0),uint64_t(64),3,pointAxes);
            if constexpr(MAPPED_STARTS) Sub<float,false>(W(0)[64],P(0)[64],C(0),uint64_t(64),3,axes);
        } else {
    #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis), P(axis), C(axis), uint64_t(64), repeats, pointCoefficient);
        }
        PipeBarrier<PIPE_V>();
        ComputeSubStage2<EXACT_DIFFERENCES, HALF_SELECTED_PACKET>(repeats, binary, coefficient, pointBinary, axes, pointAxes);
        Mul<float, false>(W(8), W(0), C(6), uint64_t(64), repeats, coefficient);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(8), W(1), C(7), uint64_t(64), repeats, coefficient);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(8), W(2), C(8), uint64_t(64), repeats, coefficient);
        PipeBarrier<PIPE_V>();
        Div<float, false>(W(8), W(8), C(9), uint64_t(64), repeats, coefficient);
        PipeBarrier<PIPE_V>();
        Maxs<float, false>(W(8), W(8), 0.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        Mins<float, false>(W(8), W(8), 1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        if constexpr (MODE != 3 && !EXACT_DIFFERENCES) CompareScalar<float, uint8_t, false>(mask_, W(8), 1.0F, CMPMODE::LT, uint64_t(64), repeats, unary);
        Muls<float, false>(W(8), W(8), -1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        ComputeMulAddDstStage<EXACT_DIFFERENCES, HALF_SELECTED_PACKET>(repeats, coefficient, projectionAxes);
        PipeBarrier<PIPE_V>();
        ComputeAddStage2<EXACT_DIFFERENCES, HALF_SELECTED_PACKET>(repeats, unary, binary, pointCoefficient, axes, pointAxes);
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool EXACT_DIFFERENCES, bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageUnaryType, class StageBinaryType, class StagePointCoefficientType, class StageAxesType, class StagePointAxesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ComputeAddStage2(StageRepeatsType& repeats, StageUnaryType& unary, StageBinaryType& binary, StagePointCoefficientType& pointCoefficient, StageAxesType& axes, StagePointAxesType& pointAxes) {
        ComputeAddStage2AddStage<EXACT_DIFFERENCES, HALF_SELECTED_PACKET>(repeats, binary, axes);
        PipeBarrier<PIPE_V>();
        ComputeSubStage<EXACT_DIFFERENCES, HALF_SELECTED_PACKET>(repeats, binary, pointCoefficient, axes, pointAxes);
        Mul<float, false>(W(0), W(0), W(0), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(0), W(1), W(1), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(0), W(2), W(2), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        if constexpr (MODE != 3) Select<float, uint8_t, false>(W(0), valid_, W(0), __builtin_inff(), SELMODE::VSEL_TENSOR_SCALAR_MODE, uint64_t(64), repeats, binary);
        if constexpr(MODE==3 && sizeof(T)==2) {
            // Cached gather offsets identify real point lanes. Padded lanes
            // must not contribute NaNs to the near-distance reduction.
            Cast(W(6),po_,RoundMode::CAST_NONE,count_);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,W(6),float(int32_t(validPointRows_*12U)),CMPMODE::LT,count_);PipeBarrier<PIPE_V>();
            Select(W(0),valid_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,count_);
        }
        // The native M1 final score FMA already has its V barrier, and its
        // alias-only reduction adds no vector instruction before the square root.
        if constexpr(!(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024))
            PipeBarrier<PIPE_V>();
        if constexpr(MODE==8) {
            if(selectedCount_==1) {
                out_=W(0);
                Copy(outIdx_,so_,8,1,CopyRepeatParams(1,1,1,1));PipeBarrier<PIPE_V>();
            } else ReducePairs<true>();
        } else ReducePairs<false>();
        ComputeWholeReduceMinStage();
        ComputeCompareStage();
        CheckNumericalRisk<EXACT_DIFFERENCES && SHORT_FP32_METADATA && SINGLE_TILE && MODE==2>();
        if constexpr(sizeof(T)==2) {
            Sqrt(out_,out_,rows_);PipeBarrier<PIPE_V>();
            Cast(outT_,out_,RoundMode::CAST_RINT,rows_);
        } else if constexpr(NATIVE_SINGLE_SEGMENT_GEOMETRY && FIXED_ROWS==1024) {
            // The same1024 values use16 complete repeats with the existing64 mask.
            // Avoid counter-mask setup/restoration for this fixed-capacity drain.
            Sqrt<float,false>(outT_,out_,uint64_t(64),repeats,unary);
        } else Sqrt(outT_,out_,rows_);
        if constexpr(SINGLE_TILE) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID3);WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        } else PipeBarrier<PIPE_ALL>();


    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool EXACT_DIFFERENCES, bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageBinaryType, class StageAxesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ComputeAddStage2AddStage(StageRepeatsType& repeats, StageBinaryType& binary, StageAxesType& axes) {
        if constexpr(!EXACT_DIFFERENCES) {
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Add<float,false>(W(0),W(0),W(3),uint64_t(64),3,axes);
            if constexpr(MAPPED_STARTS) Add<float,false>(W(0)[64],W(0)[64],W(3)[64],uint64_t(64),3,axes);
        } else {
    #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(W(axis), W(axis), W(axis+3), uint64_t(64), repeats, binary);
        }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ComputeWholeReduceMinStage() {
        if constexpr(MODE==3 && sizeof(T)==2) {
            WholeReduceMin(W(6),W(0),64,count_/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            WholeReduceMin(zeroBound_[8],W(6),count_/64,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            Compare(mask_,zeroBound_[8],zeroBound_[32],CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if(mask_.GetValue(0)&1U) {
                CorrectAxialZeroScores<true>(uint32_t(W(0).GetPhyAddr()),uint32_t(xyz_.GetPhyAddr()),allocCount_,
                    uint32_t(columns_.GetPhyAddr()),104,uint32_t(W(1).GetPhyAddr()),allocCount_,
                    uint32_t(outputOffsets_.GetPhyAddr()),(count_+255)/256*32,count_);
                CorrectCollinearZeroScores<true>(uint32_t(W(0).GetPhyAddr()),uint32_t(xyz_.GetPhyAddr()),allocCount_,
                    uint32_t(columns_.GetPhyAddr()),104,uint32_t(zeroExact_.GetPhyAddr()),
                    uint32_t(zeroBound_[32].GetPhyAddr()),uint32_t(outputOffsets_.GetPhyAddr()),count_,true);
            }
        }
    }

template<class T, uint32_t MODE, bool SINGLE_TILE, uint32_t FIXED_ROWS, uint32_t FIXED_STRIDE>
    template<bool EXACT_DIFFERENCES, bool HALF_SELECTED_PACKET, class StageRepeatsType, class StageBinaryType, class StageCoefficientType, class StagePointBinaryType, class StageAxesType, class StagePointAxesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreePairs<T, MODE, SINGLE_TILE, FIXED_ROWS, FIXED_STRIDE>::ComputeSubStage2(StageRepeatsType& repeats, StageBinaryType& binary, StageCoefficientType& coefficient, StagePointBinaryType& pointBinary, StageAxesType& axes, StagePointAxesType& pointAxes) {
        if constexpr(!EXACT_DIFFERENCES) {
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Sub<float,false>(W(3),P(0),W(0),uint64_t(64),3,pointAxes);
            if constexpr(MAPPED_STARTS) Sub<float,false>(W(3)[64],P(0)[64],W(0)[64],uint64_t(64),3,axes);
        } else {
    #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis+3), P(axis), W(axis), uint64_t(64), repeats, pointBinary);
        }
        PipeBarrier<PIPE_V>();
        ComputeAddStage<HALF_SELECTED_PACKET>(repeats, binary, coefficient, axes);
        PipeBarrier<PIPE_V>();
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Sub<float,false>(W(6),P(0),W(6),uint64_t(64),3,pointAxes);
            if constexpr(MAPPED_STARTS) Sub<float,false>(W(6)[64],P(0)[64],W(6)[64],uint64_t(64),3,axes);
        } else {
    #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis+6), P(axis), W(axis+6), uint64_t(64), repeats, pointBinary);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(BATCH_COORDINATE_AXES || HALF_SELECTED_PACKET) {
            Add<float,false>(W(3),W(3),W(6),uint64_t(64),3,axes);
            if constexpr(MAPPED_STARTS) Add<float,false>(W(3)[64],W(3)[64],W(6)[64],uint64_t(64),3,axes);
        } else {
    #pragma unroll
            for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(W(axis+3), W(axis+3), W(axis+6), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        }
    }
