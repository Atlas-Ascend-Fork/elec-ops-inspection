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
    template<auto fixedSearchMask>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchArenaStage() {
        if constexpr(DEFER_CHUNK_CERTIFICATE) {
            // Resolve only the winning chunk's two predicate words per point.
            // Indices are byte offsets into the existing usable UB arena.
            const LocalTensor<float> arena(TPosition::VECCALC,0,188416/4);
            if constexpr(fixedSearchMask) {
                const UnaryRepeatParams resolveUnary{1,1,8,8};
                const BinaryRepeatParams resolveBinary{1,1,1,8,8,8};
                SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
                ShiftRight<uint32_t,false>(offset_.template ReinterpretCast<uint32_t>(),idx_.template ReinterpretCast<uint32_t>(),uint32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Muls<int32_t,false>(offset_,offset_,int32_t(32),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Gather(SearchW<false>(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
                SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Duplicate<int32_t,false>(SearchW<false>(4).template ReinterpretCast<int32_t>(),int32_t(192),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
                SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
                And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SearchW<false>(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,resolveBinary);PipeBarrier<PIPE_V>();
                SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Muls<int32_t,false>(offset_,offset_,int32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,SearchW<false>(2).template ReinterpretCast<int32_t>(),uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,outputOffset_,uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
                Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
                SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Adds<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            } else {
            ShiftRight(offset_.template ReinterpretCast<uint32_t>(),idx_.template ReinterpretCast<uint32_t>(),uint32_t(8),64);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(32),64);PipeBarrier<PIPE_V>();
            Gather(SearchW<false>(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Duplicate(SearchW<false>(4).template ReinterpretCast<int32_t>(),int32_t(192),64);PipeBarrier<PIPE_V>();
            And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SearchW<false>(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(8),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,SearchW<false>(2).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,outputOffset_,64);PipeBarrier<PIPE_V>();
            Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Adds(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
            Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            }
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto fixedSearchMask, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchSetMaskNormStage(StageSameType& same, StageUnaryType& unary) {
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==2) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Axpy<float,float,false>(tileLo_,tileHi_,2.125F,uint64_t(64),1,unary);
            PipeBarrier<PIPE_V>();
        } else if constexpr(CLEAN_PRIMARY64) {
            SharedPrefetchFence17();
        } else if constexpr(NATIVE_HALF_PREFIX_PRIMARY) {
            constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
            constexpr uint32_t constantBase=14*(CAPACITY+32)+pointPitch-16;
            const LocalTensor<float> upperScale(TPosition::VECCALC,constantBase*4,8);
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            FusedMulAdd<float,false>(tileHi_,upperScale,tileLo_,uint64_t(64),1,BinaryRepeatParams{1,0,1,8,0,8});
            PipeBarrier<PIPE_V>();
        } else if constexpr(fixedSearchMask) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Muls<float,false>(tileHi_,tileHi_,2.125F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Add<float,false>(tileHi_,tileLo_,tileHi_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        } else {
            Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
            Add(tileHi_,tileLo_,tileHi_,64);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto fixedSearchMask>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchDuplicateStage() {
        if constexpr(CAPACITY<1024) {
            // A unique bit across both words requires a nonzero one-bit union
            // and no bit occupied in both words.
            Duplicate(SearchW<false>(4).template ReinterpretCast<int32_t>(),int32_t(1),64);
            Or(SearchW<false>(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
            And(SearchW<false>(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Sub(SearchW<false>(0).template ReinterpretCast<int32_t>(),SearchW<false>(2).template ReinterpretCast<int32_t>(),SearchW<false>(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            And(SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(2).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Or(SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(3).template ReinterpretCast<uint16_t>(),128);
            Cast(SearchW<false>(1),SearchW<false>(2).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Cast(SearchW<false>(0),SearchW<false>(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,SearchW<false>(0),0.0F,CMPMODE::EQ,64);
            CompareScalar(smallMask_,SearchW<false>(1),0.0F,CMPMODE::NE,64);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        } else {
        if constexpr(fixedSearchMask) {
            const UnaryRepeatParams certUnary{1,1,8,8};
            const BinaryRepeatParams certBinary{1,1,1,8,8,8};
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Duplicate<int32_t,false>(SearchW<false>(4).template ReinterpretCast<int32_t>(),int32_t(31),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SearchW<false>(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Muls<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,certUnary);PipeBarrier<PIPE_V>();
            Gather(SearchW<false>(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            Or<uint16_t,false>(SearchW<false>(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);
            And<uint16_t,false>(SearchW<false>(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Sub<int32_t,false>(SearchW<false>(0).template ReinterpretCast<int32_t>(),SearchW<false>(2).template ReinterpretCast<int32_t>(),SearchW<false>(4).template ReinterpretCast<int32_t>(),uint64_t(64),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            Or<uint16_t,false>(SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(3).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        } else {
        Duplicate(SearchW<false>(4).template ReinterpretCast<int32_t>(),int32_t(31),64);PipeBarrier<PIPE_V>();
        And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SearchW<false>(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Muls(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
        Gather(SearchW<false>(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        Or(SearchW<false>(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
        And(SearchW<false>(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Sub(SearchW<false>(0).template ReinterpretCast<int32_t>(),SearchW<false>(2).template ReinterpretCast<int32_t>(),SearchW<false>(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
        Or(SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(0).template ReinterpretCast<uint16_t>(),SearchW<false>(3).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        }
        Cast(SearchW<false>(0),SearchW<false>(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,SearchW<false>(0),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<uint32_t extent>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::ReduceCoordinateValidity() {
        Duplicate(W(0),0.0F,extent);PipeBarrier<PIPE_V>();
        Select(W(0),valid_,W(0),1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,extent);PipeBarrier<PIPE_V>();
        WholeReduceMax(record_,W(0),64,extent/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        WholeReduceMax(record_,record_,extent/64,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        CompareScalar(smallMask_,record_,0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return (smallMask_.GetValue(0)&1U)!=0;
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool COEFFICIENT_INPUT>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CoarseCoordinatesSafe() {
        constexpr uint32_t extent=COEFFICIENT_INPUT?COEFFICIENTS:192;
        constexpr uint32_t words=extent/16;
        auto temporaryMask=record_.template ReinterpretCast<uint8_t>();
        Duplicate(valid_.template ReinterpretCast<uint16_t>(),uint16_t(65535),words);PipeBarrier<PIPE_V>();
        #pragma unroll
        for(uint32_t axis=0;axis<(COEFFICIENT_INPUT?6U:1U);++axis) {
            auto x=COEFFICIENT_INPUT?C(axis):compactP_;
            Abs(W(1),x,extent);PipeBarrier<PIPE_V>();
            if constexpr(sizeof(T)==2) {
            // Finite Half coordinates have spacing at least 2^-24, including subnormals.
            CompareScalar(mask_,W(1),65504.0F,CMPMODE::LE,extent);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),words);PipeBarrier<PIPE_V>();
            } else {
            CompareScalar(mask_,W(1),1099511627776.0F,CMPMODE::LE,extent);
            CompareScalar(temporaryMask,W(1),9.5367431640625e-7F,CMPMODE::GE,extent);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),words);
            CompareScalar(mask_,W(1),0.0F,CMPMODE::EQ,extent);PipeBarrier<PIPE_V>();
            Or(temporaryMask.template ReinterpretCast<uint16_t>(),temporaryMask.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),words);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),temporaryMask.template ReinterpretCast<uint16_t>(),words);PipeBarrier<PIPE_V>();
            }
        }
        if constexpr(COEFFICIENT_INPUT) {
            CompareScalar(mask_,C(13),8.673617379884035e-19F,CMPMODE::GE,extent);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),words);PipeBarrier<PIPE_V>();
        }
        if constexpr(!COEFFICIENT_INPUT) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            auto packed=valid_.template ReinterpretCast<uint64_t>();
            return (packed.GetValue(0)&packed.GetValue(1)&packed.GetValue(2))==~uint64_t(0);
        } else {
        return ReduceCoordinateValidity<extent>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CoarseDirectionSafe() {
        constexpr uint32_t extent=COEFFICIENTS,words=extent/16;
        Mul(W(0),C(6),C(6),extent);PipeBarrier<PIPE_V>();
        MulAddDst(W(0),C(7),C(7),extent);PipeBarrier<PIPE_V>();
        MulAddDst(W(0),C(8),C(8),extent);PipeBarrier<PIPE_V>();
        MakeIndices(W(2).template ReinterpretCast<int32_t>(),0,extent);PipeBarrier<PIPE_V>();
        // C220 ordered comparison uses exactly represented FP32 coefficient indices.
        Cast(W(2),W(2).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,extent);PipeBarrier<PIPE_V>();
        CompareScalar(mask_,W(2),float(int32_t(t_.segments)),CMPMODE::LT,extent);PipeBarrier<PIPE_V>();
        // Padded coefficients do not represent legal candidate segments.
        Select(W(0),mask_,W(0),1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,extent);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,W(0),0.999999523162841796875F,CMPMODE::GE,extent);
        CompareScalar(mask_,W(0),1.000000476837158203125F,CMPMODE::LE,extent);PipeBarrier<PIPE_V>();
        And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),words);
        Mul(W(1),C(9),C(9),extent);PipeBarrier<PIPE_V>();
        Sub(W(1),W(1),C(13),extent);PipeBarrier<PIPE_V>();
        Abs(W(1),W(1),extent);
        Muls(W(2),C(13),2.384185791015625e-7F,extent);PipeBarrier<PIPE_V>();
        Compare(mask_,W(1),W(2),CMPMODE::LE,extent);PipeBarrier<PIPE_V>();
        And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),words);PipeBarrier<PIPE_V>();
        return ReduceCoordinateValidity<extent>();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::FinalizeGuardedCertificate(uint32_t activeRows) {
        // Refinement gathers a full register, including inactive point lanes.
        if(activeRows<64) {
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),~((uint64_t(1)<<activeRows)-1));
            Duplicate<int32_t,false>(idx_,int32_t(0),uint64_t(64),1,1,8);
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
        PipeBarrier<PIPE_V>();
        if constexpr(DEFER_CHUNK_CERTIFICATE) {
            // Resolve only the winning chunk's two predicate words per point.
            // Indices are byte offsets into the existing usable UB arena.
            const LocalTensor<float> arena(TPosition::VECCALC,0,188416/4);
            if constexpr(FULL_FIXED_MASK) {
                const UnaryRepeatParams resolveUnary{1,1,8,8};
                const BinaryRepeatParams resolveBinary{1,1,1,8,8,8};
                SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
                ShiftRight<uint32_t,false>(offset_.template ReinterpretCast<uint32_t>(),idx_.template ReinterpretCast<uint32_t>(),uint32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Muls<int32_t,false>(offset_,offset_,int32_t(32),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Gather(W(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
                SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Duplicate<int32_t,false>(W(4).template ReinterpretCast<int32_t>(),int32_t(192),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
                SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
                And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),W(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,resolveBinary);PipeBarrier<PIPE_V>();
                SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Muls<int32_t,false>(offset_,offset_,int32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,W(2).template ReinterpretCast<int32_t>(),uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,outputOffset_,uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
                Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
                SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Adds<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            } else {
            ShiftRight(offset_.template ReinterpretCast<uint32_t>(),idx_.template ReinterpretCast<uint32_t>(),uint32_t(8),64);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(32),64);PipeBarrier<PIPE_V>();
            Gather(W(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Duplicate(W(4).template ReinterpretCast<int32_t>(),int32_t(192),64);PipeBarrier<PIPE_V>();
            And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),W(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(8),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,W(2).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,outputOffset_,64);PipeBarrier<PIPE_V>();
            Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Adds(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
            Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            }
        }
        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        FinalizeGuardedCertificateDuplicateStage();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::FinalizeGuardedCertificateDuplicateStage() {
        if constexpr(CAPACITY<1024) {
            // A unique bit across both words requires a nonzero one-bit union
            // and no bit occupied in both words.
            Duplicate(W(4).template ReinterpretCast<int32_t>(),int32_t(1),64);
            Or(W(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
            And(W(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Sub(W(0).template ReinterpretCast<int32_t>(),W(2).template ReinterpretCast<int32_t>(),W(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            And(W(0).template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),W(2).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Or(W(0).template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),W(3).template ReinterpretCast<uint16_t>(),128);
            Cast(W(1),W(2).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Cast(W(0),W(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,W(0),0.0F,CMPMODE::EQ,64);
            CompareScalar(smallMask_,W(1),0.0F,CMPMODE::NE,64);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        } else {
        if constexpr(FULL_FIXED_MASK) {
            const UnaryRepeatParams certUnary{1,1,8,8};
            const BinaryRepeatParams certBinary{1,1,1,8,8,8};
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Duplicate<int32_t,false>(W(4).template ReinterpretCast<int32_t>(),int32_t(31),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),W(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Muls<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,certUnary);PipeBarrier<PIPE_V>();
            Gather(W(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            Or<uint16_t,false>(W(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);
            And<uint16_t,false>(W(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Sub<int32_t,false>(W(0).template ReinterpretCast<int32_t>(),W(2).template ReinterpretCast<int32_t>(),W(4).template ReinterpretCast<int32_t>(),uint64_t(64),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            Or<uint16_t,false>(W(0).template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),W(3).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        } else {
        Duplicate(W(4).template ReinterpretCast<int32_t>(),int32_t(31),64);PipeBarrier<PIPE_V>();
        And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),W(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Muls(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
        Gather(W(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        Or(W(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
        And(W(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Sub(W(0).template ReinterpretCast<int32_t>(),W(2).template ReinterpretCast<int32_t>(),W(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
        Or(W(0).template ReinterpretCast<uint16_t>(),W(0).template ReinterpretCast<uint16_t>(),W(3).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        }
        Cast(W(0),W(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,W(0),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::BuildCrossChunkCertificate(
        uint32_t chunks,const BinaryRepeatParams& same,const UnaryRepeatParams& unary) {
        CommonVectorStage51(chunks, same, unary);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSameType, class StageUnionsType, class StageCollisionsType, class StageNextUnionsType, class StageNextCollisionsType, class StageMergedCollisionsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::BuildCrossChunkCertificateCStage(StageSameType& same, StageUnionsType& unions, StageCollisionsType& collisions, StageNextUnionsType& nextUnions, StageNextCollisionsType& nextCollisions, StageMergedCollisionsType& mergedCollisions) {
#pragma unroll

        for(uint32_t stage=0;stage<2;++stage) {
            auto rotation=C(3)[CAPACITY+16+stage*8].template ReinterpretCast<uint32_t>();
            Gather(nextUnions.template ReinterpretCast<float>(),unions.template ReinterpretCast<float>(),rotation,0U,8);
            Gather(nextCollisions.template ReinterpretCast<float>(),collisions.template ReinterpretCast<float>(),rotation,0U,8);
            PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(16));
            And<uint16_t,false>(mergedCollisions,unions,nextUnions,uint64_t(128),1,same);
            Or<uint16_t,false>(nextUnions,unions,nextUnions,uint64_t(128),1,same);
            Or<uint16_t,false>(nextCollisions,collisions,nextCollisions,uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
            Or<uint16_t,false>(nextCollisions,nextCollisions,mergedCollisions,uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
            auto oldU=unions;unions=nextUnions;nextUnions=oldU;
            auto oldC=collisions;collisions=nextCollisions;nextCollisions=oldC;
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfCachedRows() {
        const uint32_t guardedRows=liveRows_;
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            SearchStaticPolicy<false>();
        } else {
        constexpr bool DEFER_SECOND=CLEAN_PRIMARY64 && sizeof(T)==2 && ROWS==64;
        const float coarseErrorFactor=(tightCoefficientSafe_ && CoarseCoordinatesSafe<false>())?
            7.62939453125e-6F:1.52587890625e-5F;
        if constexpr(UsesCapacityStage3()) {
            InitializeGroupedSearchState(lo_,idx_,DEFER_CHUNK_CERTIFICATE?1:3);
        } else {
            SharedSearchCoefficients21();
        }
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)Duplicate(refined_[520],-2.0F,8);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        count_=(guardedRows*64);
        if constexpr(CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        SearchGuardedHalfCachedRowsCompactSearchTailStage(guardedRows, coarseErrorFactor, same, segment, pointSegment, unary);
        if constexpr(DEFER_SECOND) {
            const uint32_t chunks=(t_.segments+63)/64;
            BuildCrossChunkCertificate(chunks,same,unary);
        }
        FinalizeGuardedCertificate(guardedRows);
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ constexpr bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::UsesCapacityStage3() { return ((sizeof(T)==4 && (CAPACITY==1024 || CAPACITY==256)) || FULL_FIXED_MASK);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageGuardedRowsType, class StageCoarseErrorFactorType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfCachedRowsCompactSearchTailStage(StageGuardedRowsType& guardedRows, StageCoarseErrorFactorType& coarseErrorFactor, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary) {
        for(uint32_t j=0;j<t_.segments;j+=64) {
        if(TryCompactSearchTail(j))continue;
        if constexpr(!CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(W(axis),P(axis),C(axis)[j],uint64_t(64),guardedRows,pointSegment);
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        Mul<float,false>(W(4),W(0),W(0),uint64_t(64),guardedRows,same);
        Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),guardedRows,segment);
        } else {
        Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),guardedRows,segment);
        Mul<float,false>(W(4),W(0),W(0),uint64_t(64),guardedRows,same);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),guardedRows,same);
        MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),guardedRows,segment);
        } else {
        MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),guardedRows,segment);
        MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),guardedRows,same);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),guardedRows,same);
        MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),guardedRows,segment);
        } else {
        MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),guardedRows,segment);
        MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),guardedRows,same);
        }
        PipeBarrier<PIPE_V>();
        SearchGuardedHalfCachedRowsMulsStage(guardedRows, coarseErrorFactor, same, segment, unary, j);
        Sub<float,false>(W(0),W(4),W(5),uint64_t(64),guardedRows,same);SearchGuardedHalfCachedRowsCompareScalarStage(guardedRows, same, unary, j);}
    }
