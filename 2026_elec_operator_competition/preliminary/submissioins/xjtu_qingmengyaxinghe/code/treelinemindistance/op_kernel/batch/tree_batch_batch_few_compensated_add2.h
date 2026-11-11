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
    template<auto COUNT, auto pointPitch, auto lanes, class StageRecordType, class StageMaskType, class StageValidType, class StageSmallMaskType, class StageRepeatsType, class StageBinaryType, class StageAxesW0Type, class StageAxesW3Type, class StageAxesC3Type, class StageAxesP0Type>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::BatchFewCompensatedAddStage2(StageRecordType& record, StageMaskType& mask, StageValidType& valid, StageSmallMaskType& smallMask, StageRepeatsType& repeats, StageBinaryType& binary, StageAxesW0Type& axesW0, StageAxesW3Type& axesW3, StageAxesC3Type& axesC3, StageAxesP0Type& axesP0) {
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Add<float,false>(axesW0,axesW0,axesW3,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,20,20));
        } else {
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Add<float, false>(FixedFewW(axis), FixedFewW(axis), FixedFewW(axis+3), uint64_t(64), repeats, binary);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Sub<float,false>(axesW3,axesP0,axesC3,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,pointPitch/8,16));
        } else {
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) Sub<float, false>(FixedFewW(axis+3), FixedFewP(axis), FixedFewC(axis+3), uint64_t(64), repeats, binary);
        }
        SharedFixedResidualNorm5<COUNT>(mask, binary, repeats, valid);
        bool batchFewCompensatedDefinedStageResult{};
        if (BatchFewCompensatedDefinedStage<lanes, COUNT>(record, smallMask, repeats, batchFewCompensatedDefinedStageResult)) return batchFewCompensatedDefinedStageResult;
        if constexpr(COUNT<64) {SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));}
        return true;


    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageRepeatsType, class StageBinaryType, class StageAxesW0Type, class StageAxesW3Type, class StageAxesW9Type, class StageAxesC10Type>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::BatchFewCompensatedMulAddDstStage(StageRepeatsType& repeats, StageBinaryType& binary, StageAxesW0Type& axesW0, StageAxesW3Type& axesW3, StageAxesW9Type& axesW9, StageAxesC10Type& axesC10) {
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            MulAddDst<float,float,false>(axesW0,FixedFewW(8),axesW9,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,0,20));
            MulAddDst<float,float,false>(axesW3,FixedFewW(8),axesC10,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,0,16));
        } else {
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            MulAddDst<float, float, false>(FixedFewW(axis), FixedFewW(8), FixedFewW(axis+9), uint64_t(64), repeats, binary);
            MulAddDst<float, float, false>(FixedFewW(axis+3), FixedFewW(8), FixedFewC(axis+10), uint64_t(64), repeats, binary);
        }
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageRepeatsType, class StageBinaryType, class StageAxesW0Type, class StageAxesW3Type, class StageAxesW6Type, class StageAxesC0Type>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::BatchFewCompensatedAddStage(StageRepeatsType& repeats, StageBinaryType& binary, StageAxesW0Type& axesW0, StageAxesW3Type& axesW3, StageAxesW6Type& axesW6, StageAxesC0Type& axesC0) {
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            Add<float,false>(axesW6,axesW0,axesW3,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,20,20));
            Sub<float,false>(axesW3,axesW3,axesC0,uint64_t(64),uint8_t(3),BinaryRepeatParams(1,1,1,20,20,16));
        } else {
#pragma unroll
        for (uint32_t axis = 0; axis < 3; ++axis) {
            Add<float, false>(FixedFewW(axis+6), FixedFewW(axis), FixedFewW(axis+3), uint64_t(64), repeats, binary);
            Sub<float, false>(FixedFewW(axis+3), FixedFewW(axis+3), FixedFewC(axis), uint64_t(64), repeats, binary);
        }
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto lanes, auto COUNT, class StageRecordType, class StageSmallMaskType, class StageRepeatsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::BatchFewCompensatedDefinedStage(StageRecordType& record, StageSmallMaskType& smallMask, StageRepeatsType& repeats, bool& stageResult) {
        if constexpr(true) {
        if constexpr(UsesNativeHalfPrefixPrimaryStage2()) {
            // COUNT is64, hence repeats is1. A second one-element reduction
            // preserves the first result and needs no separate instruction.
            WholeReduceMax(record,FixedFewW(12),lanes,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
            Muls(record,record,1.8189894035458565e-12F,1);
            WholeReduceMin(record[64],FixedFewW(0),lanes,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        } else {
        WholeReduceMax(FixedFewW(7),FixedFewW(12),lanes,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(record,FixedFewW(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Muls(record,record,1.8189894035458565e-12F,1);
        WholeReduceMin(FixedFewW(7),FixedFewW(0),lanes,repeats,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMin(record[64],FixedFewW(7),repeats,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        }
        if(HasAmbiguousMinimum(record,smallMask)) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            { stageResult = false; return true; }
        }
        }
        return false;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ constexpr bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::UsesNativeHalfPrefixPrimaryStage2() { return (NATIVE_HALF_PREFIX_PRIMARY
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
                     || (sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024)
#endif
                     );
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::BatchIds() {
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch+13*1056;
        return LocalTensor<float>(TPosition::VECCALC,base*4,64);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<uint32_t> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::BatchPointIds() {
        return LocalTensor<uint32_t>(TPosition::VECCALC,uint32_t(BatchIds().GetPhyAddr())+64*4,8);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<uint32_t> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::BatchPointMap() {
        return LocalTensor<uint32_t>(TPosition::VECCALC,uint32_t(BatchIds().GetPhyAddr())+96*4,64);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::EndpointF(uint32_t k) {
        return FewWork(0)[20*1056+k*64];
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<int32_t> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::EndpointI(uint32_t k) {
        return EndpointF(k).template ReinterpretCast<int32_t>();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::EndpointSplit(uint32_t digit) {
        ShiftRight(EndpointI(30),EndpointI(31),int32_t(15),64);PipeBarrier<PIPE_V>();
        Muls(EndpointI(32),EndpointI(30),int32_t(32768),64);PipeBarrier<PIPE_V>();
        Sub(EndpointI(digit),EndpointI(31),EndpointI(32),64);PipeBarrier<PIPE_V>();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::EndpointEncode(LocalTensor<float> x,uint32_t dst) {
        Muls(EndpointF(34),x,0.0078125F,64);PipeBarrier<PIPE_V>();
        Cast(EndpointI(33),EndpointF(34),RoundMode::CAST_TRUNC,64);PipeBarrier<PIPE_V>();
        Cast(EndpointF(34),EndpointI(33),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Muls(EndpointF(34),EndpointF(34),128.0F,64);PipeBarrier<PIPE_V>();
        Sub(EndpointF(35),x,EndpointF(34),64);PipeBarrier<PIPE_V>();
        Muls(EndpointF(35),EndpointF(35),16777216.0F,64);PipeBarrier<PIPE_V>();
        Cast(EndpointI(31),EndpointF(35),RoundMode::CAST_TRUNC,64);PipeBarrier<PIPE_V>();
        ShiftRight(EndpointI(36),EndpointI(31),int32_t(30),64);PipeBarrier<PIPE_V>();
        EndpointSplit(dst);
        Copy(EndpointI(31),EndpointI(30),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        EndpointSplit(dst+1);
        Muls(EndpointI(dst+2),EndpointI(33),int32_t(2),64);PipeBarrier<PIPE_V>();
        Add(EndpointI(dst+2),EndpointI(dst+2),EndpointI(36),64);PipeBarrier<PIPE_V>();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline,aligned(128))) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CorrectHalfLatticePacket(
        uint64_t validLanes,uint64_t near,LocalTensor<int32_t> outputOffset,uint32_t maskAddress) {
        uint64_t count=near-((near>>1)&uint64_t(0x5555555555555555ULL));
        count=(count&uint64_t(0x3333333333333333ULL))+((count>>2)&uint64_t(0x3333333333333333ULL));
        count=(count+(count>>4))&uint64_t(0x0f0f0f0f0f0f0f0fULL);
        if(!(count&uint64_t(0x0e0e0e0e0e0e0e0eULL)))return false;
        uint64_t packets=0;
        for(uint32_t group=0;group<8;++group) {
            const uint32_t bits=uint32_t(near>>(group*8))&255U;
            if(bits && (bits&(bits-1U)))packets|=uint64_t(255)<<(group*8);
        }
        const LocalTensor<uint8_t> masks(TPosition::VECCALC,maskAddress,256);
        const uint64_t active=validLanes&packets;
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        masks.template ReinterpretCast<uint64_t>().SetValue(0,active);
        masks[128].template ReinterpretCast<uint64_t>().SetValue(0,active);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Sub(EndpointF(axis),FixedFewP(axis),FixedFewC(axis),64);
            Sub(EndpointF(axis+3),FixedFewC(axis+3),FixedFewC(axis),64);
        }
        PipeBarrier<PIPE_V>();
        Abs(EndpointF(6),EndpointF(0),64);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t k=1;k<6;++k) {
            Abs(EndpointF(7),EndpointF(k),64);PipeBarrier<PIPE_V>();
            Max(EndpointF(6),EndpointF(6),EndpointF(7),64);PipeBarrier<PIPE_V>();
        }
        Select(EndpointF(6),masks,EndpointF(6),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        WholeReduceMax(EndpointF(8),EndpointF(6),64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        const uint32_t maximumBits=EndpointI(8).template ReinterpretCast<uint32_t>().GetValue(0);
        if(maximumBits==0 || maximumBits>=0x7f800000U)return false;
        const int32_t exponent=int32_t((maximumBits>>23)&255U)-127;
        if(exponent< -24 || exponent>17)return false;
        union FloatPower {uint32_t bits;float value;};
        FloatPower factor,unitSquared;factor.bits=uint32_t(131-exponent)<<23;
        unitSquared.bits=uint32_t(127+2*(exponent-4))<<23;
        SetFlag<HardEvent::S_V>(EVENT_ID1);return CorrectHalfLatticePacketEndpointFStage(outputOffset, masks, active, factor, unitSquared);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageOutputOffsetType, class StageMasksType, class StageActiveType, class StageFactorType, class StageUnitSquaredType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CorrectHalfLatticePacketEndpointFStage(StageOutputOffsetType& outputOffset, StageMasksType& masks, StageActiveType& active, StageFactorType& factor, StageUnitSquaredType& unitSquared) {
        WaitFlag<HardEvent::S_V>(EVENT_ID1);
        // Validate the same nine raw coordinate planes. Predicates keep
        // independent aligned destinations; only element-wise arithmetic is batched.
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        auto scaled=EndpointF(9),selected=EndpointF(18),rounded=EndpointF(27);
        auto integers=EndpointI(0);
        auto points=FixedFewP(0),endpoints=FixedFewC(0);
        scaled.SetSize(9*64);selected.SetSize(9*64);rounded.SetSize(9*64);
        integers.SetSize(9*64);points.SetSize(3*pointPitch);endpoints.SetSize(6*128);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Muls<float,false>(scaled,points,factor.value,uint64_t(64),uint8_t(3),UnaryRepeatParams{1,1,8,pointPitch/8});
        Muls<float,false>(scaled[3*64],endpoints,factor.value,uint64_t(64),uint8_t(6),UnaryRepeatParams{1,1,8,16});
        PipeBarrier<PIPE_V>();
        // The later round-trip comparison rejects every value outside this range.
        // Limit only the cast input; accepted integer coordinates are unchanged.
        Mins(selected,scaled,1048576.0F,9*64);PipeBarrier<PIPE_V>();
        Maxs(selected,selected,-1048576.0F,9*64);PipeBarrier<PIPE_V>();
        Cast(integers,selected,RoundMode::CAST_TRUNC,9*64);PipeBarrier<PIPE_V>();
        Cast(rounded,integers,RoundMode::CAST_NONE,9*64);PipeBarrier<PIPE_V>();
#pragma unroll
        for(uint32_t axis=0;axis<9;++axis) {
            Compare(masks[64],rounded[axis*64],scaled[axis*64],CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(masks[128].template ReinterpretCast<uint16_t>(),masks[128].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Sub(EndpointI(axis),EndpointI(axis),EndpointI(axis+3),64);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Sub(EndpointI(axis+3),EndpointI(axis+6),EndpointI(axis+3),64);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for(uint32_t axis=0;axis<6;++axis) {
            Cast(EndpointF(9),EndpointI(axis),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(masks[32],EndpointF(9),-32.0F,CMPMODE::GE,64);
            CompareScalar(masks[64],EndpointF(9),32.0F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            And(masks[32].template ReinterpretCast<uint16_t>(),masks[32].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(masks[128].template ReinterpretCast<uint16_t>(),masks[128].template ReinterpretCast<uint16_t>(),masks[32].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        const uint64_t integral=masks[128].template ReinterpretCast<uint64_t>().GetValue(0);
        uint64_t admitted=0;
        return CorrectHalfLatticePacketReinterpretCastStage(outputOffset, masks, active, unitSquared, integral, admitted);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageOutputOffsetType, class StageMasksType, class StageActiveType, class StageUnitSquaredType, class StageIntegralType, class StageAdmittedType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CorrectHalfLatticePacketReinterpretCastStage(StageOutputOffsetType& outputOffset, StageMasksType& masks, StageActiveType& active, StageUnitSquaredType& unitSquared, StageIntegralType& integral, StageAdmittedType& admitted) {
        for(uint32_t group=0;group<8;++group) {
            const uint64_t packet=uint64_t(255)<<(group*8);
            if((active&packet) && !(active&packet&~integral))admitted|=active&packet;
        }
        if(!admitted)return false;
        masks.template ReinterpretCast<uint64_t>().SetValue(0,admitted);
        masks[128].template ReinterpretCast<uint64_t>().SetValue(0,admitted);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
#pragma unroll
        for(uint32_t axis=0;axis<6;++axis)
            Select(EndpointF(axis),masks,EndpointF(axis),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
        PipeBarrier<PIPE_V>();
        Mul(EndpointI(9),EndpointI(0),EndpointI(3),64);
        Mul(EndpointI(10),EndpointI(3),EndpointI(3),64);
        Mul(EndpointI(11),EndpointI(0),EndpointI(0),64);PipeBarrier<PIPE_V>();
#pragma unroll
        for(uint32_t axis=1;axis<3;++axis) {
            Mul(EndpointI(6),EndpointI(axis),EndpointI(axis+3),64);
            Mul(EndpointI(7),EndpointI(axis+3),EndpointI(axis+3),64);
            Mul(EndpointI(8),EndpointI(axis),EndpointI(axis),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(9),EndpointI(9),EndpointI(6),64);
            Add(EndpointI(10),EndpointI(10),EndpointI(7),64);
            Add(EndpointI(11),EndpointI(11),EndpointI(8),64);PipeBarrier<PIPE_V>();
        }
        Mul(EndpointI(12),EndpointI(11),EndpointI(10),64);
        Mul(EndpointI(13),EndpointI(9),EndpointI(9),64);PipeBarrier<PIPE_V>();
        Sub(EndpointI(12),EndpointI(12),EndpointI(13),64);PipeBarrier<PIPE_V>();
        Copy(EndpointI(13),EndpointI(10),64,1,CopyRepeatParams(1,1,8,8));
        Duplicate(EndpointI(16),int32_t(1),64);PipeBarrier<PIPE_V>();
        Cast(EndpointF(14),EndpointI(9),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(masks[32],EndpointF(14),0.0F,CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        Select(EndpointF(12),masks[32],EndpointF(11),EndpointF(12),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
        Select(EndpointF(13),masks[32],EndpointF(16),EndpointF(13),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);return CorrectHalfLatticePacketSubStage(outputOffset, masks, unitSquared);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageOutputOffsetType, class StageMasksType, class StageUnitSquaredType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CorrectHalfLatticePacketSubStage(StageOutputOffsetType& outputOffset, StageMasksType& masks, StageUnitSquaredType& unitSquared) {
        PipeBarrier<PIPE_V>();
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Sub(EndpointI(15),EndpointI(axis),EndpointI(axis+3),64);PipeBarrier<PIPE_V>();
            Mul(EndpointI(6),EndpointI(15),EndpointI(15),64);PipeBarrier<PIPE_V>();
            if(axis==0)Copy(EndpointI(14),EndpointI(6),64,1,CopyRepeatParams(1,1,8,8));
            else Add(EndpointI(14),EndpointI(14),EndpointI(6),64);
            PipeBarrier<PIPE_V>();
        }
        Cast(EndpointF(15),EndpointI(9),RoundMode::CAST_NONE,64);
        Cast(EndpointF(6),EndpointI(10),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Compare(masks[32],EndpointF(15),EndpointF(6),CMPMODE::GE,64);PipeBarrier<PIPE_V>();
        Select(EndpointF(12),masks[32],EndpointF(14),EndpointF(12),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
        Select(EndpointF(13),masks[32],EndpointF(16),EndpointF(13),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        Select(EndpointF(12),masks,EndpointF(12),EndpointF(16),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
        Duplicate(EndpointI(17),int32_t(0),64);PipeBarrier<PIPE_V>();
        Select(EndpointF(13),masks,EndpointF(13),EndpointF(17),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        ShiftRight(EndpointI(18),EndpointI(12),int32_t(15),64);PipeBarrier<PIPE_V>();
        Muls(EndpointI(19),EndpointI(18),int32_t(32768),64);PipeBarrier<PIPE_V>();
        Sub(EndpointI(19),EndpointI(12),EndpointI(19),64);PipeBarrier<PIPE_V>();
        // Compare N/D exactly. Base-2^15 cross-products fit two small int32 limbs.
        CorrectHalfLatticePacketMulsStage(outputOffset, masks);
        Cast(EndpointF(33),EndpointI(12),RoundMode::CAST_NONE,64);
        Cast(EndpointF(34),EndpointI(13),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        Div(EndpointF(35),EndpointF(33),EndpointF(34),64);PipeBarrier<PIPE_V>();
        Muls(EndpointF(35),EndpointF(35),unitSquared.value,64);PipeBarrier<PIPE_V>();
        Select(EndpointF(35),masks[128],EndpointF(35),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        Select(FixedFewW(0),masks,EndpointF(35),FixedFewW(0),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        return true;


    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageOutputOffsetType, class StageMasksType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CorrectHalfLatticePacketMulsStage(StageOutputOffsetType& outputOffset, StageMasksType& masks) {
        for(uint32_t lane=0;lane<8;++lane) {
            Muls(EndpointI(20),outputOffset,int32_t(4),8);PipeBarrier<PIPE_V>();
            Adds(EndpointI(20),EndpointI(20),int32_t(lane*4),8);PipeBarrier<PIPE_V>();
            Gather(EndpointF(21),EndpointF(18),EndpointI(20).template ReinterpretCast<uint32_t>(),0U,8);
            Gather(EndpointF(22),EndpointF(19),EndpointI(20).template ReinterpretCast<uint32_t>(),0U,8);
            Gather(EndpointF(23),EndpointF(13),EndpointI(20).template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            Brcb(EndpointF(24),EndpointF(21),1,BrcbRepeatParams(1,8));
            Brcb(EndpointF(25),EndpointF(22),1,BrcbRepeatParams(1,8));
            Brcb(EndpointF(26),EndpointF(23),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Mul(EndpointI(27),EndpointI(19),EndpointI(26),64);
            Mul(EndpointI(28),EndpointI(25),EndpointI(13),64);PipeBarrier<PIPE_V>();
            ShiftRight(EndpointI(29),EndpointI(27),int32_t(15),64);
            ShiftRight(EndpointI(30),EndpointI(28),int32_t(15),64);
            Mul(EndpointI(31),EndpointI(18),EndpointI(26),64);
            Mul(EndpointI(32),EndpointI(24),EndpointI(13),64);PipeBarrier<PIPE_V>();
            Add(EndpointI(31),EndpointI(31),EndpointI(29),64);
            Add(EndpointI(32),EndpointI(32),EndpointI(30),64);PipeBarrier<PIPE_V>();
            Muls(EndpointI(29),EndpointI(29),int32_t(32768),64);
            Muls(EndpointI(30),EndpointI(30),int32_t(32768),64);PipeBarrier<PIPE_V>();
            Sub(EndpointI(27),EndpointI(27),EndpointI(29),64);
            Sub(EndpointI(28),EndpointI(28),EndpointI(30),64);PipeBarrier<PIPE_V>();
            Cast(EndpointF(6),EndpointI(31),RoundMode::CAST_NONE,64);
            Cast(EndpointF(7),EndpointI(32),RoundMode::CAST_NONE,64);
            Cast(EndpointF(8),EndpointI(27),RoundMode::CAST_NONE,64);
            Cast(EndpointF(9),EndpointI(28),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Compare(masks[32],EndpointF(6),EndpointF(7),CMPMODE::LT,64);
            Compare(masks[64],EndpointF(6),EndpointF(7),CMPMODE::EQ,64);
            Compare(masks[96],EndpointF(8),EndpointF(9),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
            And(masks[64].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),masks[96].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            Or(masks[32].template ReinterpretCast<uint16_t>(),masks[32].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
            And(masks[128].template ReinterpretCast<uint16_t>(),masks[128].template ReinterpretCast<uint16_t>(),masks[32].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CorrectHalfEndpointPacketTies(
        uint64_t validLanes,uint64_t equal,LocalTensor<int32_t> outputOffset,uint32_t maskAddress) {
        const LocalTensor<uint8_t> masks(TPosition::VECCALC,maskAddress,256);
        uint64_t counts=equal-((equal>>1)&uint64_t(0x5555555555555555ULL));
        counts=(counts&uint64_t(0x3333333333333333ULL))+((counts>>2)&uint64_t(0x3333333333333333ULL));
        counts=(counts+(counts>>4))&uint64_t(0x0f0f0f0f0f0f0f0fULL);
        if(!(counts&uint64_t(0x0e0e0e0e0e0e0e0eULL)))return false;
        uint64_t ambiguous=0;
        #pragma unroll 1
        for(uint32_t group=0;group<8;++group) {
            const uint32_t bits=uint32_t(equal>>(group*8))&255U;
            if(bits && (bits&(bits-1U)))ambiguous|=uint64_t(255)<<(group*8);
        }
        if(!ambiguous)return false;
        // A conservative projection margin admits only strict endpoint regions.
        // If any nearby candidate has an uncertain/interior projection, abstain
        // for that complete packet and preserve its original computation.
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            Sub(EndpointF(axis),FixedFewP(axis),FixedFewC(axis),64);
            Sub(EndpointF(axis+3),FixedFewC(axis+3),FixedFewC(axis),64);
        }
        PipeBarrier<PIPE_V>();
        Mul(EndpointF(9),EndpointF(0),EndpointF(3),64);
        Mul(EndpointF(10),EndpointF(3),EndpointF(3),64);
        Mul(EndpointF(11),EndpointF(0),EndpointF(0),64);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=1;axis<3;++axis) {
            Mul(EndpointF(12),EndpointF(axis),EndpointF(axis+3),64);
            Mul(EndpointF(13),EndpointF(axis+3),EndpointF(axis+3),64);
            Mul(EndpointF(15),EndpointF(axis),EndpointF(axis),64);PipeBarrier<PIPE_V>();
            Add(EndpointF(9),EndpointF(9),EndpointF(12),64);
            Add(EndpointF(10),EndpointF(10),EndpointF(13),64);
            Add(EndpointF(11),EndpointF(11),EndpointF(15),64);PipeBarrier<PIPE_V>();
        }
        Add(EndpointF(12),EndpointF(10),EndpointF(11),64);
        Sub(EndpointF(13),EndpointF(9),EndpointF(10),64);PipeBarrier<PIPE_V>();
        Muls(EndpointF(12),EndpointF(12),0.0009765625F,64);PipeBarrier<PIPE_V>();
        Compare(masks[64],EndpointF(13),EndpointF(12),CMPMODE::GT,64);
        Muls(EndpointF(13),EndpointF(12),-1.0F,64);PipeBarrier<PIPE_V>();
        Compare(masks[32],EndpointF(9),EndpointF(13),CMPMODE::LT,64);
        CompareScalar(masks[96],EndpointF(10),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        Or(masks[96].template ReinterpretCast<uint16_t>(),masks[96].template ReinterpretCast<uint16_t>(),masks[32].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        Or(masks[96].template ReinterpretCast<uint16_t>(),masks[96].template ReinterpretCast<uint16_t>(),masks[64].template ReinterpretCast<uint16_t>(),4);
        return CorrectHalfEndpointPacketTiesMulsStage(validLanes, outputOffset, masks, ambiguous);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageValidLanesType, class StageOutputOffsetType, class StageMasksType, class StageAmbiguousType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CorrectHalfEndpointPacketTiesMulsStage(StageValidLanesType& validLanes, StageOutputOffsetType& outputOffset, StageMasksType& masks, StageAmbiguousType& ambiguous) {
        Muls(EndpointF(15),EndpointF(14),0.0009765625F,64);PipeBarrier<PIPE_V>();
        Add(EndpointF(15),EndpointF(15),EndpointF(14),64);PipeBarrier<PIPE_V>();
        Compare(masks,FixedFewW(0),EndpointF(15),CMPMODE::LE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID1);WaitFlag<HardEvent::V_S>(EVENT_ID1);
        const uint64_t near=masks.template ReinterpretCast<uint64_t>().GetValue(0)&validLanes&ambiguous;
        const uint64_t endpoint=masks[96].template ReinterpretCast<uint64_t>().GetValue(0);
        uint64_t admitted=0;
#pragma unroll 1
        for(uint32_t group=0;group<8;++group) {
            const uint64_t packet=uint64_t(255)<<(group*8);
            if((near&packet) && !(near&packet&~endpoint))admitted|=near&packet;
        }
        if(!admitted)return false;
        masks.template ReinterpretCast<uint64_t>().SetValue(0,admitted);
        masks[128].template ReinterpretCast<uint64_t>().SetValue(0,admitted);
        SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Select(EndpointF(6+axis),masks[64],FixedFewC(axis+3),FixedFewC(axis),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
#pragma unroll
        for(uint32_t k=0;k<6;++k)Duplicate(EndpointI(k),int32_t(0),64);
        PipeBarrier<PIPE_V>();
        CorrectHalfEndpointPacketTiesEndpointEncodeStage(masks);
        // Lexicographic comparison of exact squared-norm digits; each digit
        // converts exactly to float and each eight-lane packet is independent.
        auto record=EndpointF(41);record.SetSize(128);
        CorrectHalfEndpointPacketTiesCastStage(outputOffset, masks, record);
        Cast(EndpointF(45),EndpointI(5),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
#pragma unroll 1
        for(int32_t k=4;k>=0;--k) {
            Muls(EndpointF(45),EndpointF(45),32768.0F,64);
            Cast(EndpointF(46),EndpointI(uint32_t(k)),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Add(EndpointF(45),EndpointF(45),EndpointF(46),64);PipeBarrier<PIPE_V>();
        }
        Muls(EndpointF(45),EndpointF(45),3.5527136788005009e-15F,64);PipeBarrier<PIPE_V>();
        Select(EndpointF(45),masks[128],EndpointF(45),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        Select(FixedFewW(0),masks,EndpointF(45),FixedFewW(0),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);PipeBarrier<PIPE_V>();
        return true;

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageOutputOffsetType, class StageMasksType, class StageRecordType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CorrectHalfEndpointPacketTiesCastStage(StageOutputOffsetType& outputOffset, StageMasksType& masks, StageRecordType& record) {
#pragma unroll

        for(int32_t k=5;k>=0;--k) {
            Cast(EndpointF(39),EndpointI(uint32_t(k)),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Select(EndpointF(39),masks[128],EndpointF(39),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            WholeReduceMin(record,EndpointF(39),8,8,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
            Gather(EndpointF(43),record,outputOffset.template ReinterpretCast<uint32_t>(),0U,8);PipeBarrier<PIPE_V>();
            Brcb(EndpointF(44),EndpointF(43),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            Compare(masks[160],EndpointF(39),EndpointF(44),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
            And(masks[128].template ReinterpretCast<uint16_t>(),masks[128].template ReinterpretCast<uint16_t>(),masks[160].template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        }
    }
