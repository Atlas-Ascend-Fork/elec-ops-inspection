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

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<uint32_t WIDTH, bool OMIT_DIRECTION_LOW>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrimaryCompensatedBody() {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<uint8_t> mask_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4)*4,ROWS*8);
        const LocalTensor<uint8_t> valid_(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        // Reconstruct exact-path geometry in scratch, keeping the normalized search cache intact.
        constexpr bool DIRECT_LENGTH=WIDTH==64 && (CLEAN_PRIMARY64 || NATIVE_HALF_PREFIX_PRIMARY);
        constexpr uint8_t repeats=1;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        PrimaryCompensatedBodySetMaskNormStage<WIDTH, DIRECT_LENGTH, repeats>(binary);
        PrimaryCompensatedBodySubStage<repeats>(binary);
        Mul<float, false>(PrimaryW(8), PrimaryW(0), PrimaryW(9), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(PrimaryW(8), PrimaryW(1), PrimaryW(10), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(PrimaryW(8), PrimaryW(2), PrimaryW(11), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        if constexpr(DIRECT_LENGTH)
            Div<float, false>(PrimaryW(8), PrimaryW(8), PrimaryC(9), uint64_t(64), repeats, binary);
        else
            Div<float, false>(PrimaryW(8), PrimaryW(8), PrimaryW(12), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Maxs<float, false>(PrimaryW(8), PrimaryW(8), 0.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        Mins<float, false>(PrimaryW(8), PrimaryW(8), 1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        if constexpr(WIDTH<64)SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        CompareScalar<float, uint8_t, false>(mask_, PrimaryW(8), 1.0F, CMPMODE::LT, uint64_t(64), repeats, unary);
        if constexpr(WIDTH<64)SetVectorMask<float,MaskMode::NORMAL>(uint64_t(WIDTH));
        Muls<float, false>(PrimaryW(8), PrimaryW(8), -1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            MulAddDst<float, float, false>(PrimaryW(axis), PrimaryW(8), PrimaryW(axis+9), uint64_t(64), repeats, binary);
            if constexpr(!OMIT_DIRECTION_LOW)
                MulAddDst<float, float, false>(PrimaryW(axis+3), PrimaryW(8), PrimaryC(axis+10), uint64_t(64), repeats, binary);
        }
        PrimaryCompensatedBodyAddStage<WIDTH, DIRECT_LENGTH, repeats>(mask_, valid_, binary);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<uint32_t WIDTH, auto DIRECT_LENGTH, auto repeats, class StageMask_Type, class StageValid_Type, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrimaryCompensatedBodyAddStage(StageMask_Type& mask_, StageValid_Type& valid_, StageBinaryType& binary) {
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(PrimaryW(axis), PrimaryW(axis), PrimaryW(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(PrimaryW(axis+3), PrimaryP(axis), PrimaryC(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        SetCmpMask(mask_);
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(PrimaryW(axis),PrimaryW(axis),PrimaryW(axis+3),1,binary);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(PrimaryW(0), PrimaryW(0), PrimaryW(0), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(PrimaryW(0), PrimaryW(1), PrimaryW(1), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(PrimaryW(0), PrimaryW(2), PrimaryW(2), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        if constexpr(WIDTH<64)SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        if constexpr(!DIRECT_LENGTH) {
            Select<float, uint8_t, false>(PrimaryW(0), valid_, PrimaryW(0), __builtin_inff(), SELMODE::VSEL_TENSOR_SCALAR_MODE, uint64_t(64), repeats, binary);
            PipeBarrier<PIPE_V>();
        }

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<uint32_t WIDTH, auto DIRECT_LENGTH, auto repeats, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrimaryCompensatedBodySetMaskNormStage(StageBinaryType& binary) {
        if constexpr(DIRECT_LENGTH) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                Sub<float,false>(PrimaryW(axis+9),PrimaryC(axis+3),PrimaryC(axis),uint64_t(64),repeats,binary);
        } else {
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)Sub(PrimaryW(axis+9),PrimaryC(axis+3),PrimaryC(axis),WIDTH);
            PipeBarrier<PIPE_V>();
            Adds(PrimaryW(12),PrimaryC(9),0.0F,WIDTH);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(WIDTH));
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto repeats, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrimaryCompensatedBodySubStage(StageBinaryType& binary) { if constexpr(DEFER_DIRECTION_LOW)CompensatedDeferred<false>(repeats,binary);else CompensatedImmediate<false>(repeats,binary);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrimaryCompensated16() {
        PrimaryCompensatedBody<16>();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrimaryCompensated32() {
        if constexpr(DEFER_DIRECTION_LOW)AscendC::ICachePreLoad(int64_t(1));
        PrimaryCompensatedBody<32>();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrimaryCompensatedFull() {
        if constexpr(DEFER_DIRECTION_LOW)AscendC::ICachePreLoad(int64_t(1));
        PrimaryCompensatedBody<64>();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrimaryCompensatedExactDirection() {
        PrimaryCompensatedBody<64,true>();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<uint32_t WIDTH>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrimaryCompensated() {
        PrimaryCompensatedBody<WIDTH>();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrefixPrimaryCompensatedAxes() {
        static_assert(NATIVE_FLOAT_PREFIX_SPARSE && ROWS==64,"Native prefix primary resource required");
        AscendC::ICachePreLoad(int64_t(1));
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+1024+192+13*64;
        const LocalTensor<uint8_t> mask(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4)*4,ROWS*8);
        // This arena's primary has cp=pp=64, wp=96 and one64-lane packet.
        // Twelve-block work pitch keeps each axis'32-word gap untouched.
        const BinaryRepeatParams workCoefficient{1,1,1,12,8,8};
        const BinaryRepeatParams workPointWork{1,1,1,12,8,12};
        const BinaryRepeatParams workWorkCoefficient{1,1,1,12,12,8};
        const BinaryRepeatParams workAxes{1,1,1,12,12,12};
        const BinaryRepeatParams projectAxes{1,1,1,12,0,12};
        const BinaryRepeatParams projectLowAxes{1,1,1,12,0,8};
        const BinaryRepeatParams same{1,1,1,8,8,8};
        const UnaryRepeatParams unary{1,1,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Sub<float,false>(PrimaryW(9),PrimaryC(3),PrimaryC(0),uint64_t(64),3,workCoefficient);
        PipeBarrier<PIPE_V>();
        Adds(PrimaryW(12),PrimaryC(9),0.0F,64);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Sub<float,false>(PrimaryW(0),PrimaryP(0),PrimaryC(0),uint64_t(64),3,workCoefficient);
        PipeBarrier<PIPE_V>();
        Sub<float,false>(PrimaryW(3),PrimaryP(0),PrimaryW(0),uint64_t(64),3,workPointWork);
        PipeBarrier<PIPE_V>();
        Add<float,false>(PrimaryW(6),PrimaryW(0),PrimaryW(3),uint64_t(64),3,workAxes);
        Sub<float,false>(PrimaryW(3),PrimaryW(3),PrimaryC(0),uint64_t(64),3,workWorkCoefficient);
        PipeBarrier<PIPE_V>();
        Sub<float,false>(PrimaryW(6),PrimaryP(0),PrimaryW(6),uint64_t(64),3,workPointWork);
        PipeBarrier<PIPE_V>();
        Add<float,false>(PrimaryW(3),PrimaryW(3),PrimaryW(6),uint64_t(64),3,workAxes);
        PipeBarrier<PIPE_V>();
        Mul<float,false>(PrimaryW(8),PrimaryW(0),PrimaryW(9),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrimaryW(8),PrimaryW(1),PrimaryW(10),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrimaryW(8),PrimaryW(2),PrimaryW(11),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Div<float,false>(PrimaryW(8),PrimaryW(8),PrimaryW(12),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Maxs<float,false>(PrimaryW(8),PrimaryW(8),0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        Mins<float,false>(PrimaryW(8),PrimaryW(8),1.0F,uint64_t(64),1,unary);PrefixPrimaryCompensatedAxesCompareScalarStage(mask, workCoefficient, workAxes, projectAxes, projectLowAxes, same, unary);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageMaskType, class StageWorkCoefficientType, class StageWorkAxesType, class StageProjectAxesType, class StageProjectLowAxesType, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrefixPrimaryCompensatedAxesCompareScalarStage(StageMaskType& mask, StageWorkCoefficientType& workCoefficient, StageWorkAxesType& workAxes, StageProjectAxesType& projectAxes, StageProjectLowAxesType& projectLowAxes, StageSameType& same, StageUnaryType& unary) {
        PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,false>(mask,PrimaryW(8),1.0F,CMPMODE::LT,uint64_t(64),1,unary);
        Muls<float,false>(PrimaryW(8),PrimaryW(8),-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrimaryW(0),PrimaryW(8),PrimaryW(9),uint64_t(64),3,projectAxes);
        MulAddDst<float,float,false>(PrimaryW(3),PrimaryW(8),PrimaryC(10),uint64_t(64),3,projectLowAxes);
        PipeBarrier<PIPE_V>();
        Add<float,false>(PrimaryW(0),PrimaryW(0),PrimaryW(3),uint64_t(64),3,workAxes);PipeBarrier<PIPE_V>();
        Sub<float,false>(PrimaryW(3),PrimaryP(0),PrimaryC(3),uint64_t(64),3,workCoefficient);PipeBarrier<PIPE_V>();
        SetCmpMask(mask);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(PrimaryW(0),PrimaryW(0),PrimaryW(3),3,workAxes);PipeBarrier<PIPE_V>();
        Mul<float,false>(PrimaryW(0),PrimaryW(0),PrimaryW(0),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrimaryW(0),PrimaryW(1),PrimaryW(1),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrimaryW(0),PrimaryW(2),PrimaryW(2),uint64_t(64),1,same);PipeBarrier<PIPE_V>();

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool CORRECT_ZERO>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::Compensated() {
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE && !CORRECT_ZERO) {
            PrefixPrimaryCompensatedAxes();return;
        }
        // Reconstruct exact-path geometry in scratch, keeping the normalized search cache intact.
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub(W(axis+9),C(axis+3),C(axis),count_);
        PipeBarrier<PIPE_V>();
        if constexpr(CORRECT_ZERO) {
            Adds(W(12),C(13),0.0F,count_);PipeBarrier<PIPE_V>();
        } else {
            Adds(W(12),C(9),0.0F,count_);PipeBarrier<PIPE_V>();
        }
        const uint8_t repeats = count_/64;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(uint64_t(64));
        // Error-free subtraction preserves short distances at large coordinates.
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis), P(axis), C(axis), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis+3), P(axis), W(axis), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            Add<float, false>(W(axis+6), W(axis), W(axis+3), uint64_t(64), repeats, binary);
            Sub<float, false>(W(axis+3), W(axis+3), C(axis), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis+6), P(axis), W(axis+6), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        #pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(W(axis+3), W(axis+3), W(axis+6), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(W(8), W(0), W(9), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(8), W(1), W(10), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(8), W(2), W(11), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Div<float, false>(W(8), W(8), W(12), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        CompensatedMaxsStage<CORRECT_ZERO>(repeats, unary, binary);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool CORRECT_ZERO, class StageRepeatsType, class StageUnaryType, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CompensatedMaxsStage(StageRepeatsType& repeats, StageUnaryType& unary, StageBinaryType& binary) {
        Maxs<float, false>(W(8), W(8), 0.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        Mins<float, false>(W(8), W(8), 1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
        CompareScalar<float, uint8_t, false>(mask_, W(8), 1.0F, CMPMODE::LT, uint64_t(64), repeats, unary);
        Muls<float, false>(W(8), W(8), -1.0F, uint64_t(64), repeats, unary);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            MulAddDst<float, float, false>(W(axis), W(8), W(axis+9), uint64_t(64), repeats, binary);
            MulAddDst<float, float, false>(W(axis+3), W(8), C(axis+10), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(W(axis), W(axis), W(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(W(axis+3), P(axis), C(axis+3), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        SelectResidualPlanes(W(0), W(1), W(2), W(3), W(4), W(5), mask_, count_==64, repeats, binary);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(W(0), W(0), W(0), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(0), W(1), W(1), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(W(0), W(2), W(2), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Select<float, uint8_t, false>(W(0), valid_, W(0), __builtin_inff(), SELMODE::VSEL_TENSOR_SCALAR_MODE, uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        CompensatedWholeReduceMaxStage<CORRECT_ZERO>(repeats);

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool CORRECT_ZERO, class StageRepeatsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CompensatedWholeReduceMaxStage(StageRepeatsType& repeats) {
        if constexpr(CORRECT_ZERO) {
        WholeReduceMax(W(7),W(12),64,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(record_,W(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Muls(record_,record_,1.8189894035458565e-12F,1);
        WholeReduceMin(W(7),W(0),64,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(record_[64],W(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Compare(smallMask_,record_[64],record_,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if(smallMask_.GetValue(0)&1U) {
            CorrectAxialZeroScores(uint32_t(W(0).GetPhyAddr()),uint32_t(xyz_.GetPhyAddr()),pp_,
                uint32_t(columns_.GetPhyAddr()),cp_,uint32_t(W(1).GetPhyAddr()),wp_,uint32_t(mask_.GetPhyAddr()),
                (count_+255)/256*32,count_);
            CorrectCollinearZeroScores<sizeof(T)==2>(uint32_t(W(0).GetPhyAddr()),uint32_t(xyz_.GetPhyAddr()),pp_,
                uint32_t(columns_.GetPhyAddr()),cp_,uint32_t(work_[20*1056].GetPhyAddr()),uint32_t(record_.GetPhyAddr()),
                uint32_t(mask_.GetPhyAddr()),count_);
            Select(W(0),valid_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,count_);PipeBarrier<PIPE_V>();
        }
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::FewWork(uint32_t k) {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*(CAPACITY+32))*4,6*(ROWS*64+128)-k*(CAPACITY+32));
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::FewCoefficients(uint32_t k) {
        constexpr uint32_t pitch=CAPACITY+32;
        const uint32_t offset=k>=6 && k<=8?6*pitch+(k-6)*CAPACITY:k*pitch;
        return LocalTensor<float>(TPosition::VECCALC,offset*4,k>=6 && k<=8?CAPACITY:pitch);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CertifyWideScore(uint32_t candidates) {
        auto finite=W(8).template ReinterpretCast<uint8_t>();
        auto other=finite[128],possible=finite[256];
        CompareScalar(finite,W(0),1152921504606846976.0F,CMPMODE::LE,candidates);
        CompareScalar(other,C(13),1152921504606846976.0F,CMPMODE::LE,candidates);PipeBarrier<PIPE_V>();
        And(finite.template ReinterpretCast<uint16_t>(),finite.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),candidates/16);PipeBarrier<PIPE_V>();
        CompareScalar(other,W(0),0.0F,CMPMODE::GE,candidates);PipeBarrier<PIPE_V>();
        And(finite.template ReinterpretCast<uint16_t>(),finite.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),candidates/16);PipeBarrier<PIPE_V>();
        CompareScalar(other,C(13),0.0F,CMPMODE::GE,candidates);PipeBarrier<PIPE_V>();
        And(finite.template ReinterpretCast<uint16_t>(),finite.template ReinterpretCast<uint16_t>(),other.template ReinterpretCast<uint16_t>(),candidates/16);PipeBarrier<PIPE_V>();
        Not(possible.template ReinterpretCast<uint16_t>(),finite.template ReinterpretCast<uint16_t>(),candidates/16);PipeBarrier<PIPE_V>();
        And(possible.template ReinterpretCast<uint16_t>(),possible.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),candidates/16);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for(uint32_t k=0;k<candidates/64;++k)
            if(possible.template ReinterpretCast<uint64_t>().GetValue(k)!=0)return false;
        const uint32_t group=uint32_t(record_.template ReinterpretCast<int32_t>().GetValue(1));
        const uint32_t lane=uint32_t(offset_.GetValue(0));
        if(group>=candidates/64 || lane>=64)return false;
        Select(W(1),valid_,W(0),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,candidates);
        Select(W(2),valid_,C(13),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,candidates);PipeBarrier<PIPE_V>();
        Mul(W(3),W(1),W(2),candidates);PipeBarrier<PIPE_V>();
        Sqrt(W(3),W(3),candidates);PipeBarrier<PIPE_V>();
        Muls(W(3),W(3),4.76837158203125e-7F,candidates);
        Muls(W(4),W(1),9.5367431640625e-7F,candidates);
        Muls(W(5),W(2),9.094947017729282e-13F,candidates);PipeBarrier<PIPE_V>();
        Add(W(3),W(3),W(4),candidates);PipeBarrier<PIPE_V>();
        Add(W(3),W(3),W(5),candidates);PipeBarrier<PIPE_V>();
        Muls(W(3),W(3),1.0001220703125F,candidates);PipeBarrier<PIPE_V>();
        Adds(W(3),W(3),1.0e-30F,candidates);PipeBarrier<PIPE_V>();
        offset_.SetValue(0,int32_t((group*64+lane)*4));
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Gather(record_[8],W(3),offset_.template ReinterpretCast<uint32_t>(),0U,1);PipeBarrier<PIPE_V>();
        Add(record_[16],record_,record_[8],1);PipeBarrier<PIPE_V>();
        Brcb(W(4),record_[16],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Brcb(W(5),W(4),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Copy(W(4),W(5),64,candidates/64,CopyRepeatParams(1,0,8,0));
        Sub(W(3),W(1),W(3),candidates);PipeBarrier<PIPE_V>();
        Compare(possible,W(3),W(4),CMPMODE::LE,candidates);PipeBarrier<PIPE_V>();
        And(possible.template ReinterpretCast<uint16_t>(),possible.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),candidates/16);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        uint32_t count=0;
        for(uint32_t k=0;k<candidates/64;++k)count+=uint32_t(__builtin_popcountll(possible.template ReinterpretCast<uint64_t>().GetValue(k)));
        return count==1;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CertifyRefinedBody(
        const LocalTensor<float>& minima,uint64_t validLanes) {
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)AscendC::ICachePreLoad(int64_t(1));
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t workBase=14*(CAPACITY+32)+3*pointPitch;
        constexpr uint32_t stateBase=workBase+6*(ROWS*64+128)+64+192+(sizeof(T)==2?96:0)+1024+192+13*64;
        const LocalTensor<uint8_t> valid(TPosition::VECCALC,(stateBase+128+16+15*64+64*sizeof(T)/4+ROWS*2)*4,ROWS*8);
        auto finite=FewWork(8).template ReinterpretCast<uint8_t>();
        auto normFinite=finite[32];auto possible=finite[64];
        Select(FewWork(1),valid,FixedFewW(0),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
        Select(FewWork(2),valid,FixedFewC(13),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        CompareScalar(finite,FewWork(1),1152921504606846976.0F,CMPMODE::LE,64);
        CompareScalar(normFinite,FewWork(2),1152921504606846976.0F,CMPMODE::LE,64);
        PipeBarrier<PIPE_V>();
        And(finite.template ReinterpretCast<uint16_t>(),finite.template ReinterpretCast<uint16_t>(),normFinite.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        uint32_t certifyRefinedBodyMulsStageResult{};
        if (CertifyRefinedBodyMulsStage(validLanes, finite, normFinite, certifyRefinedBodyMulsStageResult)) return certifyRefinedBodyMulsStageResult;
        WholeReduceMax(FewWork(5),FewWork(3),64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Brcb(FewWork(6),FewWork(5),1,BrcbRepeatParams(1,8));
        Brcb(FewWork(4),minima,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        if constexpr(CAPACITY==256) {
            Add<float,true>(FewWork(4),FewWork(4),FewWork(6),uint64_t(64),uint8_t(1),BinaryRepeatParams(1,1,0,8,8,0));
        } else {
            Brcb(FewWork(7),FewWork(6),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Add(FewWork(4),FewWork(4),FewWork(7),64);
        }
        Sub(FewWork(3),FewWork(1),FewWork(3),64);PipeBarrier<PIPE_V>();
        Compare(possible,FewWork(3),FewWork(4),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        And(possible.template ReinterpretCast<uint16_t>(),possible.template ReinterpretCast<uint16_t>(),valid.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint64_t finiteBits=finite.template ReinterpretCast<uint64_t>().GetValue(0);
        const uint64_t possibleBits=possible.template ReinterpretCast<uint64_t>().GetValue(0);
        return CertifyRefinedBodyReturnStage(validLanes, finiteBits, possibleBits);
    }
