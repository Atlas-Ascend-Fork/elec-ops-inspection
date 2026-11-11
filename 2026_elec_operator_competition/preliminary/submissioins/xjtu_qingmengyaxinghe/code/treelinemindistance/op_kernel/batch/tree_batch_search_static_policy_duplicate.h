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
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchStaticPolicyDuplicateStage() {
        if constexpr(CAPACITY<1024) {
            // A unique bit across both words requires a nonzero one-bit union
            // and no bit occupied in both words.
            Duplicate(SearchW<true>(4).template ReinterpretCast<int32_t>(),int32_t(1),64);
            Or(SearchW<true>(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
            And(SearchW<true>(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Sub(SearchW<true>(0).template ReinterpretCast<int32_t>(),SearchW<true>(2).template ReinterpretCast<int32_t>(),SearchW<true>(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            And(SearchW<true>(0).template ReinterpretCast<uint16_t>(),SearchW<true>(0).template ReinterpretCast<uint16_t>(),SearchW<true>(2).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Or(SearchW<true>(0).template ReinterpretCast<uint16_t>(),SearchW<true>(0).template ReinterpretCast<uint16_t>(),SearchW<true>(3).template ReinterpretCast<uint16_t>(),128);
            Cast(SearchW<true>(1),SearchW<true>(2).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Cast(SearchW<true>(0),SearchW<true>(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,SearchW<true>(0),0.0F,CMPMODE::EQ,64);
            CompareScalar(smallMask_,SearchW<true>(1),0.0F,CMPMODE::NE,64);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        } else {
        Duplicate(SearchW<true>(4).template ReinterpretCast<int32_t>(),int32_t(31),64);PipeBarrier<PIPE_V>();
        And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SearchW<true>(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Muls(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
        Gather(SearchW<true>(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        Or(SearchW<true>(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
        And(SearchW<true>(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Sub(SearchW<true>(0).template ReinterpretCast<int32_t>(),SearchW<true>(2).template ReinterpretCast<int32_t>(),SearchW<true>(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
        Or(SearchW<true>(0).template ReinterpretCast<uint16_t>(),SearchW<true>(0).template ReinterpretCast<uint16_t>(),SearchW<true>(3).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Cast(SearchW<true>(0),SearchW<true>(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,SearchW<true>(0),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool DIRECT_PREDICATES, class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchStaticPolicyMergeSearchMinimumStage(StageSameType& same, StageUnaryType& unary, StageJType& j) {
        if constexpr(COMPACT_TAIL) {
            if constexpr(DIRECT_PREDICATES) MergeSearchMinimum<true,true>(j);
            else MergeSearchMinimum<true>(j);
        } else {
        if constexpr(!DIRECT_PREDICATES) {
            Copy(chunkLower_[512+j*2],SearchW<true>(0),64,1,CopyRepeatParams(1,1,8,8));
            Copy(chunkLower_[512+j*2+64],SearchW<true>(1),64,1,CopyRepeatParams(1,1,8,8));
        }

        SharedMergeMinimum15(same, j, unary);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,DIRECT_PREDICATES?chunkLower_[512+j*2]:SearchW<true>(0),cert_,1,same);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,DIRECT_PREDICATES?chunkLower_[512+j*2+64]:SearchW<true>(1),certHigh_,1,same);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SavePrefixTailPredicate(uint32_t start,uint32_t tail) {
        const uint32_t group=start/256,slot=(start/64)%4;
        auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
        auto words=predicates[slot*128].template ReinterpretCast<int32_t>();
        Duplicate(words,int32_t(0),128);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x5555555555555555ULL));
        Duplicate<int32_t,false>(words,int32_t((uint32_t(1)<<tail)-1),uint64_t(64),2,1,8);
        PipeBarrier<PIPE_V>();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrefixC(uint32_t k) {
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
                      "Native prefix search requires its matching UB resource");
            constexpr uint32_t pitch=CAPACITY+32;
            const uint32_t offset=k>=6 && k<=8?6*pitch+(k-6)*CAPACITY:k*pitch;
            return LocalTensor<float>(TPosition::VECCALC,offset*4,k>=6 && k<=8?CAPACITY:pitch);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrefixW(uint32_t k) {
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
                      "Native prefix search requires its matching UB resource");
            constexpr uint32_t pitch=ROWS*64+128;
            constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
            constexpr uint32_t base=14*(CAPACITY+32)+3*pointPitch;
            constexpr uint32_t banks[13]={64,0,0,64,96,32,96,0,0,0,0,0,0};
            const uint32_t offset=k*pitch+banks[k];
            return LocalTensor<float>(TPosition::VECCALC,(base+offset)*4,6*pitch-offset);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PrefixP(uint32_t k) {
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
                      "Native prefix search requires its matching UB resource");
            constexpr uint32_t pitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
            constexpr uint32_t base=14*(CAPACITY+32);
            return LocalTensor<float>(TPosition::VECCALC,(base+k*pitch)*4,pitch);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, bool MASK_READY>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::MergePrefixMinimum(uint32_t j) {
        constexpr uint32_t stateShift=HALF_RESOURCE?384U:0U;
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
                      "Native prefix search requires its matching UB resource");
        LocalTensor<float> certHigh_(TPosition::VECCALC,184256U+stateShift,64U);
        LocalTensor<float> cert_(TPosition::VECCALC,184000U+stateShift,64U);
        LocalTensor<float> hi_(TPosition::VECCALC,183488U+stateShift,64U);
        LocalTensor<int32_t> idx_(TPosition::VECCALC,183744U+stateShift,64U);
        LocalTensor<float> lo_(TPosition::VECCALC,182976U+stateShift,64U);
        LocalTensor<float> record_(TPosition::VECCALC,182400U+stateShift,128U);
        LocalTensor<float> second_(TPosition::VECCALC,183232U+stateShift,64U);
        LocalTensor<float> tileHi_(TPosition::VECCALC,185280U+stateShift,64U);
        LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,185536U+stateShift,64U);
        LocalTensor<float> tileLo_(TPosition::VECCALC,184768U+stateShift,64U);

        const BinaryRepeatParams same{1,1,1,8,8,8};
        const UnaryRepeatParams unary{1,1,8,8};
            if constexpr(!MASK_READY) {SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));}
            Compare<float,false>(tileLo_,lo_,CMPMODE::LT,uint64_t(64),same);
            Max<float,false>(record_,tileLo_,lo_,uint64_t(64),1,same);
            Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
            Min<float,false>(lo_,lo_,tileLo_,uint64_t(64),1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,tileHi_,hi_,1,same);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
            if constexpr(!HALF_RESOURCE) {
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,PrefixW<HALF_RESOURCE>(0),cert_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,PrefixW<HALF_RESOURCE>(1),certHigh_,1,same);
            }
#else
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(cert_,PrefixW<HALF_RESOURCE>(0),cert_,1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(certHigh_,PrefixW<HALF_RESOURCE>(1),certHigh_,1,same);
#endif
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CompactPrefixTail(uint32_t j,uint32_t tail) {
        constexpr uint32_t stateShift=HALF_RESOURCE?384U:0U;
        constexpr uint32_t predicateShift=HALF_RESOURCE?256U:0U;
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
                      "Native prefix search requires its matching UB resource");
        LocalTensor<float> chunkLower_(TPosition::VECCALC,174208U+stateShift,1024U);
        LocalTensor<float> compactP_(TPosition::VECCALC,178304U+stateShift,192U);
        LocalTensor<uint8_t> mask_(TPosition::VECCALC,187072U+predicateShift,512U);
        LocalTensor<int32_t> offset_(TPosition::VECCALC,185792U+stateShift,64U);
        LocalTensor<float> refined_(TPosition::VECCALC,179072U+stateShift,832U);
        LocalTensor<float> tileHi_(TPosition::VECCALC,185280U+stateShift,64U);
        LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,185536U+stateShift,64U);
        LocalTensor<float> tileLo_(TPosition::VECCALC,184768U+stateShift,64U);

        // Tail segments use complete point vectors instead of padded segment rows.
        #pragma unroll
        for(uint32_t col=0;col<10;++col)if(col<3||col>=6)
            Brcb(refined_[col*64],PrefixC<HALF_RESOURCE>(col)[j],1,BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        const BinaryRepeatParams same{1,1,1,8,8,8},coefficient{1,1,0,8,8,1},pointCoefficient{1,1,0,8,0,1};
        const UnaryRepeatParams unary{1,1,8,8};
        #pragma unroll
        for(uint32_t axis=0;axis<3;++axis)
            Sub<float,false>(PrefixW<HALF_RESOURCE>(axis),compactP_[axis*64],refined_[axis*64],uint64_t(64),tail,pointCoefficient);
        PipeBarrier<PIPE_V>();
        Mul<float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(0),refined_[6*64],uint64_t(64),tail,coefficient);
        Mul<float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(0),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(1),refined_[7*64],uint64_t(64),tail,coefficient);
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(1),PrefixW<HALF_RESOURCE>(1),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(2),refined_[8*64],uint64_t(64),tail,coefficient);
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(2),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        Maxs<float,false>(PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(3),0.0F,uint64_t(64),tail,unary);PipeBarrier<PIPE_V>();
        Min<float,false>(PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(0),refined_[9*64],uint64_t(64),tail,coefficient);PipeBarrier<PIPE_V>();
        Muls<float,false>(PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(3),-2.0F,uint64_t(64),tail,unary);
        Muls<float,false>(PrefixW<HALF_RESOURCE>(5),PrefixW<HALF_RESOURCE>(4),1.52587890625e-5F,uint64_t(64),tail,unary);PipeBarrier<PIPE_V>();
        Add<float,false>(PrefixW<HALF_RESOURCE>(1),PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(2),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(1),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        Sub<float,false>(PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(5),uint64_t(64),tail,same);PipeBarrier<PIPE_V>();
        CompactPrefixTailCopyStage<HALF_RESOURCE>(j, tail, chunkLower_, mask_, offset_, tileHi_, tileIdx_, tileLo_, same);
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, class StageJType, class StageTailType, class StageChunkLower_Type, class StageMask_Type, class StageOffset_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageSameType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CompactPrefixTailCopyStage(StageJType& j, StageTailType& tail, StageChunkLower_Type& chunkLower_, StageMask_Type& mask_, StageOffset_Type& offset_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageSameType& same) {
        if constexpr(SINGLE_TAIL) {
            Copy(tileLo_,PrefixW<HALF_RESOURCE>(0),64,1,CopyRepeatParams(1,1,8,8));
            Muls(tileHi_,PrefixW<HALF_RESOURCE>(5),2.125F,64);Duplicate(tileIdx_,int32_t(0),64);PipeBarrier<PIPE_V>();
            Add(tileHi_,tileLo_,tileHi_,64);
            Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            Duplicate(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),int32_t(1),64);
            Duplicate(PrefixW<HALF_RESOURCE>(1),0.0F,64);PipeBarrier<PIPE_V>();
        } else {
            Duplicate(tileLo_,__builtin_inff(),64);Duplicate(tileHi_,__builtin_inff(),64);
            Duplicate(tileIdx_,int32_t(0),64);PipeBarrier<PIPE_V>();
            for(uint32_t r=0;r<tail;++r) {
                Compare<float>(PrefixW<HALF_RESOURCE>(0)[r*64],tileLo_,CMPMODE::LT,uint64_t(64),same);
                Duplicate(offset_,int32_t(r),64);PipeBarrier<PIPE_V>();
                Min(tileLo_,PrefixW<HALF_RESOURCE>(0)[r*64],tileLo_,64);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileHi_,PrefixW<HALF_RESOURCE>(5)[r*64],tileHi_,1,same);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileIdx_.template ReinterpretCast<float>(),offset_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
            }
            Muls(tileHi_,tileHi_,2.125F,64);PipeBarrier<PIPE_V>();
            Add(tileHi_,tileLo_,tileHi_,64);
            Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
            Duplicate(PrefixW<HALF_RESOURCE>(2),0.0F,64);PipeBarrier<PIPE_V>();
            for(uint32_t r=0;r<tail;++r) {
                Compare(mask_,PrefixW<HALF_RESOURCE>(0)[r*64],tileHi_,CMPMODE::LE,64);
                Duplicate(PrefixW<HALF_RESOURCE>(3).template ReinterpretCast<int32_t>(),int32_t(1U<<r),64);PipeBarrier<PIPE_V>();
                Select(PrefixW<HALF_RESOURCE>(3),mask_,PrefixW<HALF_RESOURCE>(3),0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
                Or(PrefixW<HALF_RESOURCE>(2).template ReinterpretCast<uint16_t>(),PrefixW<HALF_RESOURCE>(2).template ReinterpretCast<uint16_t>(),PrefixW<HALF_RESOURCE>(3).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            }
            Copy(PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(2),64,1,CopyRepeatParams(1,1,8,8));
            Duplicate(PrefixW<HALF_RESOURCE>(1),0.0F,64);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SavePrefixPredicate(uint32_t start,uint32_t tail) {
        constexpr uint32_t stateShift=HALF_RESOURCE?384U:0U;
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
                      "Native prefix search requires its matching UB resource");
        LocalTensor<float> refined_(TPosition::VECCALC,179072U+stateShift,832U);
        LocalTensor<float> xyz_(TPosition::VECCALC,59136U,3168U);
        constexpr uint32_t pp_=1056;

        const uint32_t group=start/256,slot=(start/64)%4;
        auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
        auto words=predicates[slot*128].template ReinterpretCast<int32_t>();
        Duplicate(words,int32_t(0),128);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x5555555555555555ULL));
        Duplicate<int32_t,false>(words,int32_t((uint32_t(1)<<tail)-1),uint64_t(64),2,1,8);
        PipeBarrier<PIPE_V>();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
    }

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, bool FULL_ROWS>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixState(uint32_t inputRows,uint32_t segmentCount) {
        const uint32_t liveRows_=FULL_ROWS?64U:inputRows;
        // Fold the conservative prefix interval into the norm before the projection FMA.
        // Keep W4 immutable so only the winning norm needs an error-scale operation.
        constexpr float intervalFactor=1.621246337890625e-5F;
        constexpr uint32_t lowerPlane=5U;
        constexpr uint32_t errorPlane=4U;
        SharedPrefixArena20<HALF_RESOURCE>();
        TreePrefixSearchState<HALF_RESOURCE> arena;
        constexpr uint32_t pp_=1056;
        InitializeGroupedSearchState(arena.lo_,arena.idx_,3);

        PipeBarrier<PIPE_V>();

        SearchPrefixStateSavePrefixPredicateStage3<HALF_RESOURCE, FULL_ROWS, lowerPlane, pp_, errorPlane>(segmentCount, liveRows_, intervalFactor, arena.certHigh_, arena.cert_, arena.chunkLower_, arena.hi_, arena.idx_, arena.lo_, arena.mask_, arena.offset_, arena.record_, arena.refined_, arena.rowBase_, arena.second_, arena.segmentIndices_, arena.tileHi_, arena.tileIdx_, arena.tileLo_, arena.valid_, arena.xyz_, arena.same, arena.segment, arena.pointSegment, arena.unary);
        // Refinement gathers a full register, including inactive point lanes.
        SearchPrefixStateSetMaskNormStage4<HALF_RESOURCE, pp_>(liveRows_, arena.certHigh_, arena.cert_, arena.idx_, arena.offset_, arena.oneHot_, arena.rowBase_, arena.valid_, arena.xyz_);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, auto pp_, class StageLiveRows_Type, class StageCertHigh_Type, class StageCert_Type, class StageIdx_Type, class StageOffset_Type, class StageOneHot_Type, class StageRowBase_Type, class StageValid_Type, class StageXyz_Type>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixStateSetMaskNormStage4(StageLiveRows_Type& liveRows_, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageIdx_Type& idx_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageRowBase_Type& rowBase_, StageValid_Type& valid_, StageXyz_Type& xyz_) {
        if(liveRows_<64) {
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),~((uint64_t(1)<<liveRows_)-1));
            Duplicate<int32_t,false>(idx_,int32_t(0),uint64_t(64),1,1,8);
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        }
        PipeBarrier<PIPE_V>();

        SearchPrefixStateSetMaskNormStage<HALF_RESOURCE, pp_>(certHigh_, cert_, idx_, offset_, rowBase_, xyz_);
        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        SharedPrefixPredicate1<HALF_RESOURCE>(offset_, idx_, oneHot_, cert_, certHigh_, valid_);


    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, bool FULL_ROWS, auto lowerPlane, auto pp_, auto errorPlane, class StageSegmentCountType, class StageLiveRows_Type, class StageIntervalFactorType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageSegmentIndices_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageValid_Type, class StageXyz_Type, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixStateSavePrefixPredicateStage3(StageSegmentCountType& segmentCount, StageLiveRows_Type& liveRows_, StageIntervalFactorType& intervalFactor, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageSegmentIndices_Type& segmentIndices_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageValid_Type& valid_, StageXyz_Type& xyz_, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary) {
        for(uint32_t j=0;j<segmentCount;j+=64) {
        if constexpr(COMPACT_TAIL) {
            if(segmentCount-j<=8) {
                SavePrefixPredicate<HALF_RESOURCE>(j,segmentCount-j);
                CompactPrefixTail<HALF_RESOURCE>(j,segmentCount-j);MergePrefixMinimum<HALF_RESOURCE>(j);continue;
            }
        }
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(PrefixW<HALF_RESOURCE>(axis),PrefixP<HALF_RESOURCE>(axis),PrefixC<HALF_RESOURCE>(axis)[j],uint64_t(64),liveRows_,pointSegment);
        if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
        Mul<float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(0),PrefixC<HALF_RESOURCE>(6)[j],uint64_t(64),liveRows_,segment);
        Mul<float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(0),uint64_t(64),liveRows_,same);

        if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(1),PrefixC<HALF_RESOURCE>(7)[j],uint64_t(64),liveRows_,segment);
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(1),PrefixW<HALF_RESOURCE>(1),uint64_t(64),liveRows_,same);

        if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(3),PrefixW<HALF_RESOURCE>(2),PrefixC<HALF_RESOURCE>(8)[j],uint64_t(64),liveRows_,segment);
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(4),PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(2),uint64_t(64),liveRows_,same);

        if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
        // APz is dead; its bank phase differs from the dot accumulator.
        constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
        const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
        Muls<float,false>(PrefixW<HALF_RESOURCE>(5),PrefixW<HALF_RESOURCE>(4),1.0F-intervalFactor,uint64_t(64),liveRows_,unary);
        Maxs<float,false>(PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(3),0.0F,uint64_t(64),liveRows_,unary);if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
        Min<float,false>(PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(2),PrefixC<HALF_RESOURCE>(9)[j],uint64_t(64),liveRows_,segment);if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
        FusedMulAdd<float,false>(PrefixW<HALF_RESOURCE>(3),minusTwo,PrefixW<HALF_RESOURCE>(2),uint64_t(64),liveRows_,BinaryRepeatParams{1,0,1,8,0,8});if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(PrefixW<HALF_RESOURCE>(5),PrefixW<HALF_RESOURCE>(2),PrefixW<HALF_RESOURCE>(3),uint64_t(64),liveRows_,same);if constexpr(!FULL_ROWS)PipeBarrier<PIPE_V>();
        if(j+64>segmentCount) {
            CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(segmentCount-j)),CMPMODE::LT,uint64_t(64),liveRows_,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
            Select(PrefixW<HALF_RESOURCE>(lowerPlane),valid_,PrefixW<HALF_RESOURCE>(lowerPlane),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(liveRows_*64));PipeBarrier<PIPE_V>();
        }
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            WholeReduceMin<float,false>(record_,PrefixW<HALF_RESOURCE>(lowerPlane),uint64_t(64),liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        } else WholeReduceMin(record_,PrefixW<HALF_RESOURCE>(lowerPlane),64,liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        PipeBarrier<PIPE_V>();
        GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
        GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));SearchPrefixStateSetMaskNormStage3<HALF_RESOURCE, lowerPlane, pp_, errorPlane>(liveRows_, intervalFactor, certHigh_, cert_, chunkLower_, hi_, idx_, lo_, mask_, offset_, record_, refined_, rowBase_, second_, tileHi_, tileIdx_, tileLo_, xyz_, same, unary, j);}
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, auto lowerPlane, auto pp_, auto errorPlane, class StageLiveRows_Type, class StageIntervalFactorType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageXyz_Type, class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixStateSetMaskNormStage3(StageLiveRows_Type& liveRows_, StageIntervalFactorType& intervalFactor, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageXyz_Type& xyz_, StageSameType& same, StageUnaryType& unary, StageJType& j) {
        PipeBarrier<PIPE_V>();
        if constexpr(NATIVE_FLOAT_PREFIX_SPARSE) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
        } else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
        SearchPrefixStateSetMaskNormStage2<HALF_RESOURCE, errorPlane>(liveRows_, offset_, rowBase_, tileHi_, tileIdx_, same, unary);

        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Muls<float,false>(tileHi_,tileHi_,intervalFactor,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        auto chunkMinimum=tileLo_;
        auto chunkUpper=tileHi_;
        // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
        constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
        constexpr uint32_t upperConstantBase=14*(CAPACITY+32)+pointPitch-16;
        const LocalTensor<float> upperScale(TPosition::VECCALC,upperConstantBase*4,8);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        FusedMulAdd<float,false>(tileHi_,upperScale,tileLo_,uint64_t(64),1,BinaryRepeatParams{1,0,1,8,0,8});
        PipeBarrier<PIPE_V>();
        Brcb(PrefixW<HALF_RESOURCE>(1),chunkUpper,(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        const uint32_t group=j/256,slot=(j/64)%4;
        auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
        if constexpr(HALF_RESOURCE) {
            if(liveRows_==64) {
                Compare<float,uint8_t,false>(predicates[slot*128].template ReinterpretCast<uint8_t>(),PrefixW<HALF_RESOURCE>(lowerPlane),PrefixW<HALF_RESOURCE>(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));
            } else {
                Compare<float,uint8_t,false>(mask_,PrefixW<HALF_RESOURCE>(lowerPlane),PrefixW<HALF_RESOURCE>(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
            }
        } else
        { SearchPrefixStateCompareStage<HALF_RESOURCE, lowerPlane>(liveRows_, mask_, slot, predicates); }

        PipeBarrier<PIPE_V>();
        SearchPrefixStateMergePrefixMinimumStage2<HALF_RESOURCE>(certHigh_, cert_, hi_, idx_, lo_, record_, second_, tileIdx_, same, unary, j, chunkMinimum, chunkUpper);

    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, auto lowerPlane, class StageLiveRows_Type, class StageMask_Type, class StageSlotType, class StagePredicatesType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixStateCompareStage(StageLiveRows_Type& liveRows_, StageMask_Type& mask_, StageSlotType& slot, StagePredicatesType& predicates) {
        {
            Compare<float,uint8_t,false>(mask_,PrefixW<HALF_RESOURCE>(lowerPlane),PrefixW<HALF_RESOURCE>(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
            if constexpr(NATIVE_FLOAT_PREFIX_SPARSE)
                Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
            else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
            PipeBarrier<PIPE_V>();
            GatherMask(PrefixW<HALF_RESOURCE>(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
            GatherMask(PrefixW<HALF_RESOURCE>(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, auto errorPlane, class StageLiveRows_Type, class StageOffset_Type, class StageRowBase_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixStateSetMaskNormStage2(StageLiveRows_Type& liveRows_, StageOffset_Type& offset_, StageRowBase_Type& rowBase_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageSameType& same, StageUnaryType& unary) {
        if(liveRows_==64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Gather(tileHi_,PrefixW<HALF_RESOURCE>(errorPlane),offset_.template ReinterpretCast<uint32_t>(),0U,64);
        } else {
            Muls(offset_,tileIdx_,int32_t(4),liveRows_);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,rowBase_,liveRows_);PipeBarrier<PIPE_V>();
            Gather(tileHi_,PrefixW<HALF_RESOURCE>(errorPlane),offset_.template ReinterpretCast<uint32_t>(),0U,liveRows_);
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<bool HALF_RESOURCE, auto pp_, class StageCertHigh_Type, class StageCert_Type, class StageIdx_Type, class StageOffset_Type, class StageRowBase_Type, class StageXyz_Type>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchPrefixStateSetMaskNormStage(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageIdx_Type& idx_, StageOffset_Type& offset_, StageRowBase_Type& rowBase_, StageXyz_Type& xyz_) {
        if constexpr(HALF_RESOURCE) {
            // Layout is [group][four chunks][64 rows][two uint32 words].
            // A compact tail stores a conservative superset of its predicates;
            // extra ambiguous rows retain the complete numerical fallback.
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            ShiftRight(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),idx_,int32_t(8),64);
            ShiftRight(PrefixW<HALF_RESOURCE>(1).template ReinterpretCast<int32_t>(),idx_,int32_t(6),64);PipeBarrier<PIPE_V>();
            Muls(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),int32_t(pp_*4-4*512),64);
            Muls(offset_,PrefixW<HALF_RESOURCE>(1).template ReinterpretCast<int32_t>(),int32_t(512),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),64);
            ShiftRight(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),rowBase_,int32_t(5),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            Adds(offset_,offset_,int32_t(ROWS*8*4),64);PipeBarrier<PIPE_V>();
            Gather(cert_,xyz_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Gather(certHigh_,xyz_,offset_.template ReinterpretCast<uint32_t>(),4U,64);PipeBarrier<PIPE_V>();
        }
    }
#endif
