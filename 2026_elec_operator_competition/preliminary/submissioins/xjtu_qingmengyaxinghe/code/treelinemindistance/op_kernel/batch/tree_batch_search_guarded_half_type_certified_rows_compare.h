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

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto pp_, class StageGuardedRowsType, class StageMask_Type, class StageRefined_Type, class StageXyz_Type, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfTypeCertifiedRowsCompareStage(StageGuardedRowsType& guardedRows, StageMask_Type& mask_, StageRefined_Type& refined_, StageXyz_Type& xyz_, StageJType& j) {
        if constexpr(CLEAN_PRIMARY64) {
            const uint32_t group=j/256,slot=(j/64)%4;
            auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
            if(guardedRows==64) {
                Compare<float,uint8_t,false>(predicates[slot*128].template ReinterpretCast<uint8_t>(),HalfFullW(5),HalfFullW(1),CMPMODE::LE,uint64_t(64),guardedRows,BinaryRepeatParams(1,1,0,8,8,1));
            } else {
                Compare<float,uint8_t,false>(mask_,HalfFullW(5),HalfFullW(1),CMPMODE::LE,uint64_t(64),guardedRows,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
            }
        } else {
            Compare<float,uint8_t,false>(mask_,HalfFullW(5),HalfFullW(1),CMPMODE::LE,uint64_t(64),guardedRows,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
            if constexpr(DEFER_CHUNK_CERTIFICATE) {
                const uint32_t group=j/256,slot=(j/64)%4;
                auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
                if constexpr(FULL_FIXED_MASK)
                    Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
                else Copy(predicates[slot*128],mask_.template ReinterpretCast<float>(),64,2,CopyRepeatParams(1,1,8,8));
            } else {
                GatherMask(HalfFullW(0),mask_.template ReinterpretCast<float>(),uint8_t(1),GatherMaskParams(1,2,8,0));
                GatherMask(HalfFullW(1),mask_.template ReinterpretCast<float>(),uint8_t(2),GatherMaskParams(1,2,8,0));
            }
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageCertHigh_Type, class StageCert_Type, class StageIdx_Type, class StageOffset_Type, class StageOneHot_Type, class StageOutputOffset_Type>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfTypeCertifiedRowsArenaStage(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageIdx_Type& idx_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageOutputOffset_Type& outputOffset_) {
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
                Gather(HalfFullW(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
                SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Duplicate<int32_t,false>(HalfFullW(4).template ReinterpretCast<int32_t>(),int32_t(192),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
                SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
                And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),HalfFullW(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,resolveBinary);PipeBarrier<PIPE_V>();
                SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Muls<int32_t,false>(offset_,offset_,int32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,HalfFullW(2).template ReinterpretCast<int32_t>(),uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,outputOffset_,uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
                Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
                SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
                Adds<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
                Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            } else {
            ShiftRight(offset_.template ReinterpretCast<uint32_t>(),idx_.template ReinterpretCast<uint32_t>(),uint32_t(8),64);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(32),64);PipeBarrier<PIPE_V>();
            Gather(HalfFullW(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Duplicate(HalfFullW(4).template ReinterpretCast<int32_t>(),int32_t(192),64);PipeBarrier<PIPE_V>();
            And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),HalfFullW(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Muls(offset_,offset_,int32_t(8),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,HalfFullW(2).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,outputOffset_,64);PipeBarrier<PIPE_V>();
            Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            Adds(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
            Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
            }
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto DEFER_SECOND, class StageSegmentCountType, class StageChunkLower_Type, class StageHi_Type, class StageMask_Type, class StageSecond_Type, class StageSameType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfTypeCertifiedRowsDuplicateStage2(StageSegmentCountType& segmentCount, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageMask_Type& mask_, StageSecond_Type& second_, StageSameType& same) {
        if constexpr(DEFER_SECOND) {
            const uint32_t chunks=(segmentCount+63)/64;
            if(chunks<16) {
                Duplicate(chunkLower_[chunks*64],__builtin_inff(),(16-chunks)*64);
                PipeBarrier<PIPE_V>();
            }
            // Multi-repeat Compare packs sixteen64-bit results into128 bytes.
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Compare<float,uint8_t,false>(mask_,chunkLower_,hi_,CMPMODE::LE,
                uint64_t(64),16,BinaryRepeatParams{1,1,1,8,8,0});
            PipeBarrier<PIPE_V>();
            auto sourceBits=mask_.template ReinterpretCast<uint16_t>();
            auto unions=HalfFullW(0).template ReinterpretCast<uint16_t>();
            auto collisions=HalfFullW(1).template ReinterpretCast<uint16_t>();
            auto nextUnions=HalfFullW(2).template ReinterpretCast<uint16_t>();
            auto nextCollisions=HalfFullW(3).template ReinterpretCast<uint16_t>();
            auto mergedCollisions=HalfFullW(4).template ReinterpretCast<uint16_t>();
            SharedChunkFold12(unions, sourceBits, same, collisions, nextUnions, nextCollisions, mergedCollisions);
            // Four independent packed groups remain after the first two levels.
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const auto packedUnions=nextUnions.template ReinterpretCast<uint64_t>();
            const auto packedCollisions=nextCollisions.template ReinterpretCast<uint64_t>();
            const uint64_t u0=packedUnions.GetValue(0),u1=packedUnions.GetValue(1);
            const uint64_t u2=packedUnions.GetValue(2),u3=packedUnions.GetValue(3);
            const uint64_t c0=packedCollisions.GetValue(0),c1=packedCollisions.GetValue(1);
            const uint64_t c2=packedCollisions.GetValue(2),c3=packedCollisions.GetValue(3);
            const uint64_t u01=u0|u1,u23=u2|u3;
            const uint64_t c01=c0|c1|(u0&u1),c23=c2|c3|(u2&u3);
            const uint64_t unique=(u01|u23)&~(c01|c23|(u01&u23));
            second_.template ReinterpretCast<uint64_t>().SetValue(0,unique);
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            // Keep the cross-chunk certificate as a packed bitset until Process.
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageCertHigh_Type, class StageCert_Type, class StageIdx_Type, class StageOffset_Type, class StageOneHot_Type, class StageSmallMask_Type, class StageValid_Type>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedHalfTypeCertifiedRowsDuplicateStage(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageIdx_Type& idx_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageSmallMask_Type& smallMask_, StageValid_Type& valid_) {
        if constexpr(CAPACITY<1024) {
            // A unique bit across both words requires a nonzero one-bit union
            // and no bit occupied in both words.
            Duplicate(HalfFullW(4).template ReinterpretCast<int32_t>(),int32_t(1),64);
            Or(HalfFullW(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
            And(HalfFullW(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Sub(HalfFullW(0).template ReinterpretCast<int32_t>(),HalfFullW(2).template ReinterpretCast<int32_t>(),HalfFullW(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
            And(HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(2).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
            Or(HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(3).template ReinterpretCast<uint16_t>(),128);
            Cast(HalfFullW(1),HalfFullW(2).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            Cast(HalfFullW(0),HalfFullW(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
            CompareScalar(valid_,HalfFullW(0),0.0F,CMPMODE::EQ,64);
            CompareScalar(smallMask_,HalfFullW(1),0.0F,CMPMODE::NE,64);PipeBarrier<PIPE_V>();
            And(valid_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),4);PipeBarrier<PIPE_V>();
        } else {
        if constexpr(FULL_FIXED_MASK) {
            const UnaryRepeatParams certUnary{1,1,8,8};
            const BinaryRepeatParams certBinary{1,1,1,8,8,8};
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Duplicate<int32_t,false>(HalfFullW(4).template ReinterpretCast<int32_t>(),int32_t(31),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),HalfFullW(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Muls<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,certUnary);PipeBarrier<PIPE_V>();
            Gather(HalfFullW(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
            SetMaskNorm();SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            Or<uint16_t,false>(HalfFullW(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);
            And<uint16_t,false>(HalfFullW(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
            Sub<int32_t,false>(HalfFullW(0).template ReinterpretCast<int32_t>(),HalfFullW(2).template ReinterpretCast<int32_t>(),HalfFullW(4).template ReinterpretCast<int32_t>(),uint64_t(64),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
            Or<uint16_t,false>(HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(3).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        } else {
        Duplicate(HalfFullW(4).template ReinterpretCast<int32_t>(),int32_t(31),64);PipeBarrier<PIPE_V>();
        And(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),HalfFullW(4).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Muls(offset_,offset_,int32_t(4),64);PipeBarrier<PIPE_V>();
        Gather(HalfFullW(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        Or(HalfFullW(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);
        And(HalfFullW(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        Sub(HalfFullW(0).template ReinterpretCast<int32_t>(),HalfFullW(2).template ReinterpretCast<int32_t>(),HalfFullW(4).template ReinterpretCast<int32_t>(),64);PipeBarrier<PIPE_V>();
        Or(HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(0).template ReinterpretCast<uint16_t>(),HalfFullW(3).template ReinterpretCast<uint16_t>(),128);PipeBarrier<PIPE_V>();
        }
        Cast(HalfFullW(0),HalfFullW(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,HalfFullW(0),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();
        }
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedError() {
        if constexpr(CAPACITY==256 && sizeof(T)==4) {
            SearchStaticPolicy<false>();
        } else {
        constexpr bool DEFER_SECOND=CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64;
        const float coarseErrorFactor=(tightCoefficientSafe_ && CoarseCoordinatesSafe<false>())?
            7.62939453125e-6F:1.52587890625e-5F;
        SharedQueryDifference18();
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        count_=(liveRows_*64);
        if constexpr(CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
        SearchGuardedErrorCompactSearchTailStage(coarseErrorFactor, same, segment, pointSegment, unary);
        if constexpr(DEFER_SECOND) {
            const uint32_t chunks=(t_.segments+63)/64;
            BuildCrossChunkCertificate(chunks,same,unary);
        }
        FinalizeGuardedCertificate(liveRows_);
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageCoarseErrorFactorType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedErrorCompactSearchTailStage(StageCoarseErrorFactorType& coarseErrorFactor, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary) {
        for(uint32_t j=0;j<t_.segments;j+=64) {
        if(TryCompactSearchTail(j))continue;
        if constexpr(!CLEAN_PRIMARY64) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        }
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Sub<float,false>(W(axis),P(axis),C(axis)[j],uint64_t(64),liveRows_,pointSegment);
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        Mul<float,false>(W(4),W(0),W(0),uint64_t(64),liveRows_,same);
        Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),liveRows_,segment);
        } else {
        Mul<float,false>(W(3),W(0),C(6)[j],uint64_t(64),liveRows_,segment);
        Mul<float,false>(W(4),W(0),W(0),uint64_t(64),liveRows_,same);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),liveRows_,same);
        MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),liveRows_,segment);
        } else {
        MulAddDst<float,float,false>(W(3),W(1),C(7)[j],uint64_t(64),liveRows_,segment);
        MulAddDst<float,float,false>(W(4),W(1),W(1),uint64_t(64),liveRows_,same);
        }
        PipeBarrier<PIPE_V>();
        if constexpr(NORM_BEFORE_DOT) {
        MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),liveRows_,same);
        MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),liveRows_,segment);
        } else {
        MulAddDst<float,float,false>(W(3),W(2),C(8)[j],uint64_t(64),liveRows_,segment);
        MulAddDst<float,float,false>(W(4),W(2),W(2),uint64_t(64),liveRows_,same);
        }
        PipeBarrier<PIPE_V>();
        SearchGuardedErrorMulsStage(coarseErrorFactor, same, segment, unary, j);
        Sub<float,false>(W(0),W(4),W(5),uint64_t(64),liveRows_,same);SearchGuardedErrorCompareScalarStage(same, unary, j);}
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSameType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedErrorCompareScalarStage(StageSameType& same, StageUnaryType& unary, StageJType& j) {
        constexpr bool DEFER_SECOND=CLEAN_PRIMARY64 && sizeof(T)==4 && ROWS==64;

        PipeBarrier<PIPE_V>();
        if(j+64>t_.segments) {
            CompareScalar<float,uint8_t,false>(valid_,segmentIndices_,float(int32_t(t_.segments-j)),CMPMODE::LT,uint64_t(64),liveRows_,UnaryRepeatParams(1,1,8,0));PipeBarrier<PIPE_V>();
            Select(W(0),valid_,W(0),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,(liveRows_*64));PipeBarrier<PIPE_V>();
        }
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4) {
            if(j+64>t_.segments) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            }
            WholeReduceMin<float,false>(record_,W(0),uint64_t(64),liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        } else WholeReduceMin(record_,W(0),64,liveRows_,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
        PipeBarrier<PIPE_V>();
        GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
        GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
        if constexpr(CLEAN_PRIMARY64 && sizeof(T)==4)
            Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));
        else Copy(chunkLower_[j],tileLo_,64,1,CopyRepeatParams(1,1,8,8));
        SearchGuardedErrorSetMaskNormStage2(same, unary);
        PipeBarrier<PIPE_V>();
        auto chunkMinimum=(CLEAN_PRIMARY64 && sizeof(T)==2)?chunkLower_[j]:tileLo_;
        auto chunkUpper=(CLEAN_PRIMARY64 && sizeof(T)==2)?tileLo_:tileHi_;
        // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
        SearchGuardedHalfCachedRowsSetMaskNormStage(same, unary);
        Brcb(W(1),chunkUpper,(liveRows_+7)/8,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        SearchGuardedErrorCompareStage(j);
        PipeBarrier<PIPE_V>();
        SearchGuardedHalfCachedRowsMergeSearchMinimumStage<DEFER_SECOND>(same, unary, j, chunkMinimum, chunkUpper);

    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedErrorSetMaskNormStage2(StageSameType& same, StageUnaryType& unary) {
        if constexpr(FULL_FIXED_MASK) {
            if(liveRows_==64) {
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,64);
            } else {
                Muls(offset_,tileIdx_,int32_t(4),liveRows_);PipeBarrier<PIPE_V>();
                Add(offset_,offset_,rowBase_,liveRows_);PipeBarrier<PIPE_V>();
                Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,liveRows_);
            }
        } else {
            Muls(offset_,tileIdx_,int32_t(4),liveRows_);PipeBarrier<PIPE_V>();
            Add(offset_,offset_,rowBase_,liveRows_);PipeBarrier<PIPE_V>();
            Gather(tileHi_,W(5),offset_.template ReinterpretCast<uint32_t>(),0U,liveRows_);
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageCoarseErrorFactorType, class StageSameType, class StageSegmentType, class StageUnaryType, class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedErrorMulsStage(StageCoarseErrorFactorType& coarseErrorFactor, StageSameType& same, StageSegmentType& segment, StageUnaryType& unary, StageJType& j) {
        if constexpr(CLEAN_PRIMARY64) {
            // APz is dead here; keep the clamp in a bank phase distinct from dot.
            if constexpr(sizeof(T)==2)
                Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),liveRows_,unary);
            Maxs<float,false>(W(2),W(3),0.0F,uint64_t(64),liveRows_,unary);PipeBarrier<PIPE_V>();
            Min<float,false>(W(2),W(2),C(9)[j],uint64_t(64),liveRows_,segment);PipeBarrier<PIPE_V>();
            const BinaryRepeatParams fused{1,0,1,8,0,8};
            {
                constexpr uint32_t constantBase=3*(CAPACITY+32)+CAPACITY+8;
                const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
                FusedMulAdd<float,false>(W(3),minusTwo,W(2),uint64_t(64),liveRows_,fused);
            }
            if constexpr(sizeof(T)==4)
                Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),liveRows_,unary);
            PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(W(4),W(2),W(3),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
        } else {
        Maxs<float,false>(W(0),W(3),0.0F,uint64_t(64),liveRows_,unary);PipeBarrier<PIPE_V>();
        Min<float,false>(W(0),W(0),C(9)[j],uint64_t(64),liveRows_,segment);PipeBarrier<PIPE_V>();
        Muls<float,false>(W(2),W(3),-2.0F,uint64_t(64),liveRows_,unary);
        Muls<float,false>(W(5),W(4),coarseErrorFactor,uint64_t(64),liveRows_,unary);PipeBarrier<PIPE_V>();
        Add<float,false>(W(1),W(0),W(2),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(W(4),W(0),W(1),uint64_t(64),liveRows_,same);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageJType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedErrorCompareStage(StageJType& j) {
        if constexpr(CLEAN_PRIMARY64) {
            const uint32_t group=j/256,slot=(j/64)%4;
            auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
            if(liveRows_==64) {
                Compare<float,uint8_t,false>(predicates[slot*128].template ReinterpretCast<uint8_t>(),W(0),W(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));
            } else {
                Compare<float,uint8_t,false>(mask_,W(0),W(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
                Copy<float,false>(predicates[slot*128],mask_.template ReinterpretCast<float>(),uint64_t(64),2,CopyRepeatParams(1,1,8,8));
            }
        } else {
            Compare<float,uint8_t,false>(mask_,W(0),W(1),CMPMODE::LE,uint64_t(64),liveRows_,BinaryRepeatParams(1,1,0,8,8,1));PipeBarrier<PIPE_V>();
            CommonVectorStage53(j);
        }
    }

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::CertifyCorePointCoordinates(uint32_t begin,uint32_t end) {
        // This is a data-range certificate, not a shape or test identity check.
        if(end-begin<ROWS*8)return false;
        auto raw=work_,absolute=work_[4224],nonzero=work_[8448],minimumRows=work_[12672];
        for(uint32_t coordinate=begin*3;coordinate<end*3;coordinate+=4096) {
            const uint32_t available=Smallest(end*3-coordinate,4096);
            const uint32_t aligned=(available+63)/64*64;
            if(aligned!=available)Duplicate(raw,0.0F,aligned);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
            DataCopyPad(raw,pg_[coordinate],DataCopyExtParams{1,available*4,0,0,0},DataCopyPadExtParams<T>{false,0,0,T(0)});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            Abs(absolute,raw,aligned);PipeBarrier<PIPE_V>();
            CompareScalar(mask_,absolute,0.0F,CMPMODE::NE,aligned);PipeBarrier<PIPE_V>();
            Select(nonzero,mask_,absolute,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,aligned);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(record_,absolute,64,aligned/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            WholeReduceMin(minimumRows,nonzero,64,aligned/64,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(tileLo_,record_,aligned/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            WholeReduceMin(tileHi_,minimumRows,aligned/64,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            CompareScalar(smallMask_,tileLo_,1099511627776.0F,CMPMODE::LE,64);
            CompareScalar(valid_,tileHi_,9.5367431640625e-7F,CMPMODE::GE,64);
            PipeBarrier<PIPE_V>();
            And(smallMask_.template ReinterpretCast<uint16_t>(),smallMask_.template ReinterpretCast<uint16_t>(),valid_.template ReinterpretCast<uint16_t>(),4);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if((smallMask_.GetValue(0)&1U)==0) {PipeBarrier<PIPE_ALL>();return false;}
        }
        PipeBarrier<PIPE_ALL>();
        return true;
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_RETURN
bool
#else
void
#endif
    KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchGuardedCoreCertifiedRows(bool corePointSafe,bool coefficientCoordinatesSafe) {
        // Cache endpoint range checks per batch; use the original point certificate per tile when needed.
        const bool pointSafe=corePointSafe ||
            (coefficientCoordinatesSafe && liveRows_==64 && CoarseCoordinatesSafe<false>());
        if(coefficientCoordinatesSafe && pointSafe && liveRows_==64) {
            if(t_.segments==COEFFICIENTS)SearchGuardedCoreCertifiedRowsAlignedImpl(pointSafe);
            else SearchGuardedCoreCertifiedRowsImpl<true>(pointSafe);
        } else SearchGuardedCoreCertifiedRowsImpl<false>(corePointSafe);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) // TREE_STREAM_EXACT
        return pointSafe;
#endif // TREE_STREAM_EXACT
    }
#endif
