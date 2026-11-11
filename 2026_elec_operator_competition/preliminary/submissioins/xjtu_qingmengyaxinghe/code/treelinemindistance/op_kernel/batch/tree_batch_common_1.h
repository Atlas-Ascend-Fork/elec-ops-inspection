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
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<bool HALF_RESOURCE>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedPrefixArena20() {
static_assert(CAPACITY==1024 && COEFFICIENTS==576 && ROWS==64, "Bounded prefix resource required");
        static_assert(!CLEAN_PRIMARY64 && !NORM_BEFORE_DOT && !DEFER_CHUNK_CERTIFICATE, "Full-resource paths are separate");
        static_assert(HALF_RESOURCE ? (NATIVE_HALF_PREFIX_PRIMARY && ROWS==64) : NATIVE_FLOAT_PREFIX_SPARSE,
        "Native prefix search requires its matching UB resource");
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedSearchCoefficients21() {
Duplicate(lo_,__builtin_inff(),64);Duplicate(second_,__builtin_inff(),64);
            Duplicate(hi_,__builtin_inff(),64);Duplicate(idx_,int32_t(0),64);
            if constexpr(!DEFER_CHUNK_CERTIFICATE){Duplicate(cert_,0.0F,64);Duplicate(certHigh_,0.0F,64);}
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class SharedArgument0, class SharedArgument1, class SharedArgument2>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedMergeMinimum15(SharedArgument0& same, SharedArgument1& j, SharedArgument2& unary) {
SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Compare<float,false>(tileLo_,lo_,CMPMODE::LT,uint64_t(64),same);
            Max<float,false>(record_,tileLo_,lo_,uint64_t(64),1,same);
            Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(second_,record_,second_,uint64_t(64),1,same);
            Min<float,false>(lo_,lo_,tileLo_,uint64_t(64),1,same);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,tileHi_,hi_,1,same);
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedPrefetchFence17() {
constexpr uint32_t pointPitch=(CAPACITY>ROWS*8?CAPACITY:ROWS*8)+32;
            constexpr uint32_t constantBase=14*(CAPACITY+32)+pointPitch-16;
            const LocalTensor<float> upperScale(TPosition::VECCALC,constantBase*4,8);
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            FusedMulAdd<float,false>(tileHi_,upperScale,tileLo_,uint64_t(64),1,BinaryRepeatParams{1,0,1,8,0,8});
            PipeBarrier<PIPE_V>();
}
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedQueryDifference18() {
if constexpr(UsesCapacityStage3()) {
            InitializeGroupedSearchState(lo_,idx_,DEFER_CHUNK_CERTIFICATE?1:3);
        } else {
            Duplicate(lo_,__builtin_inff(),64);Duplicate(second_,__builtin_inff(),64);
            Duplicate(hi_,__builtin_inff(),64);Duplicate(idx_,int32_t(0),64);
            if constexpr(!DEFER_CHUNK_CERTIFICATE){Duplicate(cert_,0.0F,64);Duplicate(certHigh_,0.0F,64);}
        }
        PipeBarrier<PIPE_V>();
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<auto pitch, auto pointPitch, auto workBase, class SharedArgument0, class SharedArgument3, class SharedArgument4>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedResidentCoefficients19(SharedArgument0& op, SharedArgument3& segmentCount, SharedArgument4& source) {
op.m_=CAPACITY;op.cp_=pitch;op.wp_=pitch;op.pp_=pointPitch;op.count_=COEFFICIENTS;
        op.t_.segments=segmentCount;
        op.sg_.SetGlobalBuffer((__gm__ T*)source);
        op.columns_=LocalTensor<float>(TPosition::VECCALC,0,14*pitch);
        op.work_=LocalTensor<float>(TPosition::VECCALC,workBase*4,6*(ROWS*64+128));
        op.sf_=op.work_[6*CAPACITY];op.rawS_=op.sf_.template ReinterpretCast<T>();
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6, class SharedArgument7, class SharedArgument8>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedResourceBlock10(SharedArgument0& unique, SharedArgument1& uniqueRows, SharedArgument2& row, SharedArgument3& record_, SharedArgument4& tileIdx_, SharedArgument5& smallMask_, SharedArgument6& segmentIndices_, SharedArgument7& tileSecond_, SharedArgument8& idx_) {
if(unique)uniqueRows|=uint64_t(1)<<row;
        Brcb(FewWork(0),record_,1,BrcbRepeatParams(1,8));
        Brcb(FewWork(1),tileIdx_.template ReinterpretCast<float>(),1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        Copy(FewWork(2),FewWork(0),64,1,CopyRepeatParams(1,0,8,0));
        Copy(FewWork(3),FewWork(1),64,1,CopyRepeatParams(1,0,8,0));
        CompareScalar(smallMask_,segmentIndices_,float(int32_t(row)),CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        SetCmpMask(smallMask_);
        const BinaryRepeatParams same{1,1,1,8,8,8};
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(tileSecond_,FewWork(2),tileSecond_,1,same);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),FewWork(3),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();
}
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedCandidateProjection11(SharedArgument0& candidates, SharedArgument1& sparse, SharedArgument2& row, SharedArgument3& first, SharedArgument4& hits, SharedArgument5& second) {
count_=candidates;wp_=sparse?160:m_+32;
                    #pragma unroll
                    for(uint32_t axis=0;axis<3;++axis) {
                        Brcb(W(0),compactP_[axis*64+(row/8)*8],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                        Copy(P(axis),W(0)[(row%8)*8],64,candidates/64,CopyRepeatParams(1,0,8,0));PipeBarrier<PIPE_V>();
                    }
                    if(sparse) {
                        Adds(W(7),segmentIndices_,float(int32_t(first*64)),64);
                        if(hits==2)Adds(W(7)[64],segmentIndices_,float(int32_t(second*64)),64);
                    } else {
                        MakeIndices(W(6).template ReinterpretCast<int32_t>(),0,COEFFICIENTS);PipeBarrier<PIPE_V>();
                        Cast(W(7),W(6).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,COEFFICIENTS);
                    }
                    PipeBarrier<PIPE_V>();
}
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class SharedArgument0>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedGuardedMinimum13(SharedArgument0& j) {
CommonVectorStage53(j);
}
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedResourceBlock14(SharedArgument0& searchRows, SharedArgument1& unary, SharedArgument2& j, SharedArgument3& segment, SharedArgument4& coarseErrorFactor, SharedArgument5& same) {
Maxs<float,false>(W(0),W(3),0.0F,uint64_t(64),searchRows,unary);PipeBarrier<PIPE_V>();
        Min<float,false>(W(0),W(0),C(9)[j],uint64_t(64),searchRows,segment);PipeBarrier<PIPE_V>();
        Muls<float,false>(W(2),W(3),-2.0F,uint64_t(64),searchRows,unary);
        Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),searchRows,unary);PipeBarrier<PIPE_V>();
        Add<float,false>(W(1),W(0),W(2),uint64_t(64),searchRows,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(W(4),W(0),W(1),uint64_t(64),searchRows,same);PipeBarrier<PIPE_V>();
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<auto COUNT, class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedFixedResidualNorm5(SharedArgument0& mask, SharedArgument1& binary, SharedArgument2& repeats, SharedArgument3& valid) {
PipeBarrier<PIPE_V>();
        SelectResidualPlanes(FixedFewW(0), FixedFewW(1), FixedFewW(2), FixedFewW(3), FixedFewW(4), FixedFewW(5), mask, COUNT<=64, repeats, binary);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(FixedFewW(0), FixedFewW(0), FixedFewW(0), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(0), FixedFewW(1), FixedFewW(1), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float, float, false>(FixedFewW(0), FixedFewW(2), FixedFewW(2), uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
        Select<float, uint8_t, false>(FixedFewW(0), valid, FixedFewW(0), __builtin_inff(), SELMODE::VSEL_TENSOR_SCALAR_MODE, uint64_t(64), repeats, binary);
        PipeBarrier<PIPE_V>();
}
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class SharedArgument0, class SharedArgument1>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedSparseProjection7(SharedArgument0& candidates, SharedArgument1& sparse) {
WholeReduceMin(W(6),W(0),64,candidates/64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                    Gather(W(7),W(6),outputOffset_.template ReinterpretCast<uint32_t>(),0U,candidates/64);PipeBarrier<PIPE_V>();
                    WholeReduceMin(record_,W(7),candidates/64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                    Gather(offset_,record_.template ReinterpretCast<int32_t>(),outputOffset_.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
                    if(sparse) {
                        Muls(tileIdx_,offset_,int32_t(32),1);PipeBarrier<PIPE_V>();
                        Gather(tileIdx_,record2_.template ReinterpretCast<int32_t>(),tileIdx_.template ReinterpretCast<uint32_t>(),0U,1);
                    } else Muls(tileIdx_,offset_,int32_t(64),1);
                    Muls(offset_,offset_,int32_t(8),1);PipeBarrier<PIPE_V>();
                    Gather(offset_,W(6).template ReinterpretCast<int32_t>(),offset_.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
                    Add(tileIdx_,tileIdx_,offset_,1);PipeBarrier<PIPE_V>();
}
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<bool HALF_RESOURCE, class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedPrefixPredicate1(SharedArgument0& offset_, SharedArgument1& idx_, SharedArgument2& oneHot_, SharedArgument3& cert_, SharedArgument4& certHigh_, SharedArgument5& valid_) {
const UnaryRepeatParams certUnary{1,1,8,8};
        const BinaryRepeatParams certBinary{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Duplicate<int32_t,false>(PrefixW<HALF_RESOURCE>(4).template ReinterpretCast<int32_t>(),int32_t(31),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),PrefixW<HALF_RESOURCE>(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Muls<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,certUnary);PipeBarrier<PIPE_V>();
        Gather(PrefixW<HALF_RESOURCE>(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        Or<uint16_t,false>(PrefixW<HALF_RESOURCE>(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);
        And<uint16_t,false>(PrefixW<HALF_RESOURCE>(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Sub<int32_t,false>(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),PrefixW<HALF_RESOURCE>(2).template ReinterpretCast<int32_t>(),PrefixW<HALF_RESOURCE>(4).template ReinterpretCast<int32_t>(),uint64_t(64),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        Or<uint16_t,false>(PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<uint16_t>(),PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<uint16_t>(),PrefixW<HALF_RESOURCE>(3).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

        Cast(PrefixW<HALF_RESOURCE>(0),PrefixW<HALF_RESOURCE>(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,PrefixW<HALF_RESOURCE>(0),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedSparseWinner2(SharedArgument0& sparse, SharedArgument1& first, SharedArgument2& hits, SharedArgument3& second, SharedArgument4& logicalSegments, SharedArgument5& candidates) {
if(sparse) {
                        Adds(W(7),segmentIndices_,float(int32_t(first*64)),64);
                        if(hits==2)Adds(W(7)[64],segmentIndices_,float(int32_t(second*64)),64);
                    } else {
                        MakeIndices(W(6).template ReinterpretCast<int32_t>(),0,COEFFICIENTS);PipeBarrier<PIPE_V>();
                        Cast(W(7),W(6).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,COEFFICIENTS);
                    }
                    PipeBarrier<PIPE_V>();
                    CompareScalar(valid_,W(7),float(int32_t(logicalSegments)),CMPMODE::LT,candidates);PipeBarrier<PIPE_V>();
                    if constexpr(CAPACITY==256 && sizeof(T)==4) {
                        if(sparse) {
                            if(hits==1)SparseCompensated<64>();
                            else SparseCompensated<128>();
                        } else Compensated<true>();
                    } else Compensated<true>();
                    WholeReduceMin(W(6),W(0),64,candidates/64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                    Gather(W(7),W(6),outputOffset_.template ReinterpretCast<uint32_t>(),0U,candidates/64);PipeBarrier<PIPE_V>();
                    WholeReduceMin(record_,W(7),candidates/64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
                    Gather(offset_,record_.template ReinterpretCast<int32_t>(),outputOffset_.template ReinterpretCast<uint32_t>(),4U,1);PipeBarrier<PIPE_V>();
                    if(sparse) {
                        Muls(tileIdx_,offset_,int32_t(32),1);PipeBarrier<PIPE_V>();
                        Gather(tileIdx_,record2_.template ReinterpretCast<int32_t>(),tileIdx_.template ReinterpretCast<uint32_t>(),0U,1);
                    } else Muls(tileIdx_,offset_,int32_t(64),1);
                    Muls(offset_,offset_,int32_t(8),1);PipeBarrier<PIPE_V>();
}

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6, class SharedArgument7>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedChunkUnion3(SharedArgument0& collisions, SharedArgument1& sourceBits, SharedArgument2& same, SharedArgument3& nextUnions, SharedArgument4& unions, SharedArgument5& nextCollisions, SharedArgument6& mergedCollisions, SharedArgument7& unary) {
And<uint16_t,false>(collisions,sourceBits,sourceBits[32],uint64_t(128),1,same);
        PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(16));
        Or<uint16_t,false>(nextUnions,unions,unions[16],uint64_t(128),1,same);
        And<uint16_t,false>(nextCollisions,unions,unions[16],uint64_t(128),1,same);
        Or<uint16_t,false>(mergedCollisions,collisions,collisions[16],uint64_t(128),1,same);
        PipeBarrier<PIPE_V>();
        Or<uint16_t,false>(nextCollisions,nextCollisions,mergedCollisions,uint64_t(128),1,same);
        PipeBarrier<PIPE_V>();
        auto oldUnions=unions;unions=nextUnions;nextUnions=oldUnions;
        auto oldCollisions=collisions;collisions=nextCollisions;nextCollisions=oldCollisions;
                BuildCrossChunkCertificateCStage(same, unions, collisions, nextUnions, nextCollisions, mergedCollisions);
        Not<uint16_t,false>(second_.template ReinterpretCast<uint16_t>(),collisions,uint64_t(128),1,unary);
        PipeBarrier<PIPE_V>();
        And<uint16_t,false>(second_.template ReinterpretCast<uint16_t>(),
            second_.template ReinterpretCast<uint16_t>(),unions,uint64_t(128),1,same);
        PipeBarrier<PIPE_V>();
        // Keep the cross-chunk certificate as a packed bitset until Process.
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
}
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
template<class SharedArgument0, class SharedArgument1, class SharedArgument2, class SharedArgument3, class SharedArgument4, class SharedArgument5, class SharedArgument6>
__aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SharedChunkFold12(SharedArgument0& unions, SharedArgument1& sourceBits, SharedArgument2& same, SharedArgument3& collisions, SharedArgument4& nextUnions, SharedArgument5& nextCollisions, SharedArgument6& mergedCollisions) {
SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(32));
            Or<uint16_t,false>(unions,sourceBits,sourceBits[32],uint64_t(128),1,same);
            And<uint16_t,false>(collisions,sourceBits,sourceBits[32],uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(16));
            Or<uint16_t,false>(nextUnions,unions,unions[16],uint64_t(128),1,same);
            And<uint16_t,false>(nextCollisions,unions,unions[16],uint64_t(128),1,same);
            Or<uint16_t,false>(mergedCollisions,collisions,collisions[16],uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
            Or<uint16_t,false>(nextCollisions,nextCollisions,mergedCollisions,uint64_t(128),1,same);
            PipeBarrier<PIPE_V>();
}
