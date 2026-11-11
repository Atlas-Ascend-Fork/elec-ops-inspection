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

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((noinline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchSpatialStaticState(uint32_t first,uint32_t limit,bool resume,bool tightCoefficientSafe_) {
        static_assert(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Full native Float spatial state required");
        static_assert(CLEAN_PRIMARY64 && NORM_BEFORE_DOT && DEFER_CHUNK_CERTIFICATE && FULL_FIXED_MASK, "Full spatial search uses fixed-mask deferred certificates");
        LocalTensor<float> certHigh_(TPosition::VECCALC,184256U,64U);
        LocalTensor<float> cert_(TPosition::VECCALC,184000U,64U);
        LocalTensor<float> chunkLower_(TPosition::VECCALC,174208U,1024U);
        LocalTensor<float> hi_(TPosition::VECCALC,183488U,64U);
        LocalTensor<int32_t> idx_(TPosition::VECCALC,183744U,64U);
        LocalTensor<float> lo_(TPosition::VECCALC,182976U,64U);
        LocalTensor<uint8_t> mask_(TPosition::VECCALC,187072U,512U);
        LocalTensor<int32_t> offset_(TPosition::VECCALC,185792U,64U);
        LocalTensor<int32_t> oneHot_(TPosition::VECCALC,184512U,64U);
        LocalTensor<int32_t> outputOffset_(TPosition::VECCALC,186560U,64U);
        LocalTensor<float> record_(TPosition::VECCALC,182400U,128U);
        LocalTensor<float> refined_(TPosition::VECCALC,179072U,832U);
        LocalTensor<int32_t> rowBase_(TPosition::VECCALC,186048U,64U);
        LocalTensor<float> second_(TPosition::VECCALC,183232U,64U);
        LocalTensor<float> tileHi_(TPosition::VECCALC,185280U,64U);
        LocalTensor<int32_t> tileIdx_(TPosition::VECCALC,185536U,64U);
        LocalTensor<float> tileLo_(TPosition::VECCALC,184768U,64U);
        LocalTensor<uint8_t> valid_(TPosition::VECCALC,187584U,512U);
        LocalTensor<float> xyz_(TPosition::VECCALC,59136U,3168U);
        constexpr uint32_t pp_=1056;

        AscendC::ICachePreLoad(int64_t(2));
        // Only successful compaction starts below the full coefficient extent.
        // Complete its map transfer before the first overlapping W5 write.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        bool pendingSpatialMap=false;
#else
        bool pendingSpatialMap=!resume && first==0U && limit<COEFFICIENTS;
#endif
        constexpr uint32_t searchRows=64U;
        const float coarseErrorFactor=tightCoefficientSafe_?7.62939453125e-6F:1.52587890625e-5F;
        const float intervalFactor=coarseErrorFactor*1.0625F;
        const float lowerNormFactor=1.0F-intervalFactor;
        const float upperNormFactor=intervalFactor*2.125F*1.000003814697265625F;

        auto lowerScores=SpatialStateW(5);
        if(!resume) {
            InitializeGroupedSearchState(lo_,idx_,1);

            PipeBarrier<PIPE_V>();
        }
        const BinaryRepeatParams same{1,1,1,8,8,8},segment{1,1,1,8,8,0},pointSegment{1,0,1,8,1,0};
        const UnaryRepeatParams unary{1,1,8,8};
        return SearchSpatialStaticStateSetMaskNormStage<pp_, searchRows>(first, limit, certHigh_, cert_, chunkLower_, hi_, idx_, lo_, mask_, offset_, oneHot_, outputOffset_, record_, refined_, rowBase_, second_, tileHi_, tileIdx_, tileLo_, valid_, xyz_, pendingSpatialMap, lowerNormFactor, upperNormFactor, lowerScores, same, segment, pointSegment, unary);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto pp_, auto searchRows, class StageFirstType, class StageLimitType, class StageCertHigh_Type, class StageCert_Type, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageMask_Type, class StageOffset_Type, class StageOneHot_Type, class StageOutputOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageSecond_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageValid_Type, class StageXyz_Type, class StagePendingSpatialMapType, class StageLowerNormFactorType, class StageUpperNormFactorType, class StageLowerScoresType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchSpatialStaticStateSetMaskNormStage(StageFirstType& first, StageLimitType& limit, StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageMask_Type& mask_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageOutputOffset_Type& outputOffset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageSecond_Type& second_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageValid_Type& valid_, StageXyz_Type& xyz_, StagePendingSpatialMapType& pendingSpatialMap, StageLowerNormFactorType& lowerNormFactor, StageUpperNormFactorType& upperNormFactor, StageLowerScoresType& lowerScores, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary) {
        const uint32_t count_=(searchRows*64);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

        SearchSpatialStaticStateDefinedStage<pp_, searchRows>(first, limit, chunkLower_, hi_, idx_, lo_, offset_, record_, refined_, rowBase_, tileHi_, tileIdx_, tileLo_, xyz_, pendingSpatialMap, lowerNormFactor, upperNormFactor, lowerScores, same, segment, pointSegment, unary);
        // A complete single-pass search also finalizes every chunk certificate.
        const uint32_t chunks=(limit+63)/64;
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
        auto unions=SpatialStateW(0).template ReinterpretCast<uint16_t>();
        auto collisions=SpatialStateW(1).template ReinterpretCast<uint16_t>();
        auto nextUnions=SpatialStateW(2).template ReinterpretCast<uint16_t>();
        auto nextCollisions=SpatialStateW(3).template ReinterpretCast<uint16_t>();
        auto mergedCollisions=SpatialStateW(4).template ReinterpretCast<uint16_t>();
        SharedChunkFold12(unions, sourceBits, same, collisions, nextUnions, nextCollisions, mergedCollisions);
        // Fold the remaining four spatial union and collision groups.
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const auto packedUnions=nextUnions.template ReinterpretCast<uint64_t>();
        const auto packedCollisions=nextCollisions.template ReinterpretCast<uint64_t>();
        const uint64_t u0=packedUnions.GetValue(0),u1=packedUnions.GetValue(1);
        const uint64_t u2=packedUnions.GetValue(2),u3=packedUnions.GetValue(3);
        const uint64_t c0=packedCollisions.GetValue(0),c1=packedCollisions.GetValue(1);
        const uint64_t c2=packedCollisions.GetValue(2),c3=packedCollisions.GetValue(3);
        return SearchSpatialStaticStateReinterpretCastStage(certHigh_, cert_, idx_, offset_, oneHot_, outputOffset_, second_, valid_, pendingSpatialMap, count_, u0, u1, u2, u3, c0, c1, c2, c3);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageCertHigh_Type, class StageCert_Type, class StageIdx_Type, class StageOffset_Type, class StageOneHot_Type, class StageOutputOffset_Type, class StageSecond_Type, class StageValid_Type, class StagePendingSpatialMapType, class StageCount_Type, class StageU0Type, class StageU1Type, class StageU2Type, class StageU3Type, class StageC0Type, class StageC1Type, class StageC2Type, class StageC3Type>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchSpatialStaticStateReinterpretCastStage(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageIdx_Type& idx_, StageOffset_Type& offset_, StageOneHot_Type& oneHot_, StageOutputOffset_Type& outputOffset_, StageSecond_Type& second_, StageValid_Type& valid_, StagePendingSpatialMapType& pendingSpatialMap, StageCount_Type& count_, StageU0Type& u0, StageU1Type& u1, StageU2Type& u2, StageU3Type& u3, StageC0Type& c0, StageC1Type& c1, StageC2Type& c2, StageC3Type& c3) {
        const uint64_t u01=u0|u1,u23=u2|u3;
        const uint64_t c01=c0|c1|(u0&u1),c23=c2|c3|(u2&u3);
        const uint64_t unique=(u01|u23)&~(c01|c23|(u01&u23));
        second_.template ReinterpretCast<uint64_t>().SetValue(0,unique);
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        // Keep the cross-chunk certificate as a packed bitset until Process.
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

        // Refinement gathers a full register, including inactive point lanes.

        PipeBarrier<PIPE_V>();
        // Resolve only the winning chunk's two predicate words per point.
        // Indices are byte offsets into the existing usable UB arena.
        const LocalTensor<float> arena(TPosition::VECCALC,0,188416/4);
        const UnaryRepeatParams resolveUnary{1,1,8,8};
        const BinaryRepeatParams resolveBinary{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
        ShiftRight<uint32_t,false>(offset_.template ReinterpretCast<uint32_t>(),idx_.template ReinterpretCast<uint32_t>(),uint32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
        Muls<int32_t,false>(offset_,offset_,int32_t(32),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
        Gather(SpatialStateW(2).template ReinterpretCast<int32_t>(),oneHot_[32],offset_.template ReinterpretCast<uint32_t>(),0U,64);
        SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Duplicate<int32_t,false>(SpatialStateW(4).template ReinterpretCast<int32_t>(),int32_t(192),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SpatialStateW(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,resolveBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Muls<int32_t,false>(offset_,offset_,int32_t(8),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
        Add<int32_t,false>(offset_,offset_,SpatialStateW(2).template ReinterpretCast<int32_t>(),uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
        Add<int32_t,false>(offset_,offset_,outputOffset_,uint64_t(64),1,resolveBinary);PipeBarrier<PIPE_V>();
        Gather(cert_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Adds<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,resolveUnary);PipeBarrier<PIPE_V>();
        Gather(certHigh_,arena,offset_.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();

        // Check the winning chunk's packed predicate once, after all chunk minima are merged.
        const UnaryRepeatParams certUnary{1,1,8,8};
        const BinaryRepeatParams certBinary{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Duplicate<int32_t,false>(SpatialStateW(4).template ReinterpretCast<int32_t>(),int32_t(31),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        And<uint16_t,false>(offset_.template ReinterpretCast<uint16_t>(),idx_.template ReinterpretCast<uint16_t>(),SpatialStateW(4).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Muls<int32_t,false>(offset_,offset_,int32_t(4),uint64_t(64),1,certUnary);PipeBarrier<PIPE_V>();
        Gather(SpatialStateW(4).template ReinterpretCast<int32_t>(),oneHot_,offset_.template ReinterpretCast<uint32_t>(),0U,64);
        SetMaskNorm();return SearchSpatialStaticStateOrStage(certHigh_, cert_, valid_, pendingSpatialMap, count_, certBinary);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageCertHigh_Type, class StageCert_Type, class StageValid_Type, class StagePendingSpatialMapType, class StageCount_Type, class StageCertBinaryType>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchSpatialStaticStateOrStage(StageCertHigh_Type& certHigh_, StageCert_Type& cert_, StageValid_Type& valid_, StagePendingSpatialMapType& pendingSpatialMap, StageCount_Type& count_, StageCertBinaryType& certBinary) {
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        Or<uint16_t,false>(SpatialStateW(2).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);
        And<uint16_t,false>(SpatialStateW(3).template ReinterpretCast<uint16_t>(),cert_.template ReinterpretCast<uint16_t>(),certHigh_.template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        Sub<int32_t,false>(SpatialStateW(0).template ReinterpretCast<int32_t>(),SpatialStateW(2).template ReinterpretCast<int32_t>(),SpatialStateW(4).template ReinterpretCast<int32_t>(),uint64_t(64),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(128));
        Or<uint16_t,false>(SpatialStateW(0).template ReinterpretCast<uint16_t>(),SpatialStateW(0).template ReinterpretCast<uint16_t>(),SpatialStateW(3).template ReinterpretCast<uint16_t>(),uint64_t(128),1,certBinary);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

        Cast(SpatialStateW(0),SpatialStateW(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64);PipeBarrier<PIPE_V>();
        CompareScalar(valid_,SpatialStateW(0),0.0F,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();

        // A compacted seed-only resume performs no W5 write, but its map
        // transfer must still finish before caller scratch or EVENT_ID2 reuse.
        if(pendingSpatialMap)WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        return count_;


    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto pp_, auto searchRows, class StageFirstType, class StageLimitType, class StageChunkLower_Type, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageOffset_Type, class StageRecord_Type, class StageRefined_Type, class StageRowBase_Type, class StageTileHi_Type, class StageTileIdx_Type, class StageTileLo_Type, class StageXyz_Type, class StagePendingSpatialMapType, class StageLowerNormFactorType, class StageUpperNormFactorType, class StageLowerScoresType, class StageSameType, class StageSegmentType, class StagePointSegmentType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchSpatialStaticStateDefinedStage(StageFirstType& first, StageLimitType& limit, StageChunkLower_Type& chunkLower_, StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageOffset_Type& offset_, StageRecord_Type& record_, StageRefined_Type& refined_, StageRowBase_Type& rowBase_, StageTileHi_Type& tileHi_, StageTileIdx_Type& tileIdx_, StageTileLo_Type& tileLo_, StageXyz_Type& xyz_, StagePendingSpatialMapType& pendingSpatialMap, StageLowerNormFactorType& lowerNormFactor, StageUpperNormFactorType& upperNormFactor, StageLowerScoresType& lowerScores, StageSameType& same, StageSegmentType& segment, StagePointSegmentType& pointSegment, StageUnaryType& unary) {
        for(uint32_t j=first;j<limit;j+=64) {

        auto dotZ=SpatialStateW(2);auto projectedClamp=SpatialStateW(3);
        Sub<float,false>(SpatialStateW(0),SpatialStateP(0),SpatialStateC(0)[j],uint64_t(64),searchRows,pointSegment);
        Sub<float,false>(SpatialStateW(1),SpatialStateP(1),SpatialStateC(1)[j],uint64_t(64),searchRows,pointSegment);
        Mul<float,false>(SpatialStateW(4),SpatialStateW(0),SpatialStateW(0),uint64_t(64),searchRows,same);
        Mul<float,false>(SpatialStateW(3),SpatialStateW(0),SpatialStateC(6)[j],uint64_t(64),searchRows,segment);
        Sub<float,false>(dotZ,SpatialStateP(2),SpatialStateC(2)[j],uint64_t(64),searchRows,pointSegment);
        MulAddDst<float,float,false>(SpatialStateW(4),SpatialStateW(1),SpatialStateW(1),uint64_t(64),searchRows,same);
        MulAddDst<float,float,false>(SpatialStateW(3),SpatialStateW(1),SpatialStateC(7)[j],uint64_t(64),searchRows,segment);
        MulAddDst<float,float,false>(SpatialStateW(4),dotZ,dotZ,uint64_t(64),searchRows,same);
        FusedMulAdd<float,false>(dotZ,SpatialStateC(8)[j],SpatialStateW(3),uint64_t(64),searchRows,BinaryRepeatParams{1,1,1,8,0,8});
        // Complete the pending map before reusing its W5 source.
        if(pendingSpatialMap) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
            pendingSpatialMap=false;
        }
        // Preserve the AP norm in W4; bias its independent W5 copy.
        Muls<float,false>(SpatialStateW(5),SpatialStateW(4),lowerNormFactor,uint64_t(64),searchRows,unary);
        Maxs<float,false>(projectedClamp,dotZ,0.0F,uint64_t(64),searchRows,unary);
        Min<float,false>(projectedClamp,projectedClamp,SpatialStateC(9)[j],uint64_t(64),searchRows,segment);
        constexpr uint32_t constantBase=4*(CAPACITY+32)+CAPACITY+8;
        const LocalTensor<float> minusTwo(TPosition::VECCALC,constantBase*4,8);
        FusedMulAdd<float,false>(dotZ,minusTwo,projectedClamp,uint64_t(64),searchRows,BinaryRepeatParams{1,0,1,8,0,8});
        MulAddDst<float,float,false>(SpatialStateW(5),projectedClamp,dotZ,uint64_t(64),searchRows,same);
        // W5 already contains the outward-biased lower score.
        PipeBarrier<PIPE_V>();

        WholeReduceMin<float,false>(record_,lowerScores,uint64_t(64),searchRows,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);

        PipeBarrier<PIPE_V>();
        GatherMask(tileLo_,record_,uint8_t(1),GatherMaskParams(1,2,8,0));
        GatherMask(tileIdx_.template ReinterpretCast<float>(),record_,uint8_t(2),GatherMaskParams(1,2,8,0));PipeBarrier<PIPE_V>();
        Copy<float,false>(chunkLower_[j],tileLo_,uint64_t(64),1,CopyRepeatParams(1,1,8,8));

        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Muls<int32_t,false>(offset_,tileIdx_,int32_t(4),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        Add<int32_t,false>(offset_,offset_,rowBase_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Gather(tileHi_,SpatialStateW(4),offset_.template ReinterpretCast<uint32_t>(),0U,64);

        PipeBarrier<PIPE_V>();
        auto chunkMinimum=chunkLower_[j];
        auto chunkUpper=tileLo_;
        // Outward slack exceeds the rounding error introduced by reconstructing the upper bound.
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        // The unchanged chunkLower copy retains the minimum for merging.
        Axpy<float,float,false>(tileLo_,tileHi_,upperNormFactor,uint64_t(64),1,unary);
        SearchSpatialStaticStateBrcbStage<pp_, searchRows>(hi_, idx_, lo_, refined_, tileIdx_, xyz_, lowerScores, same, unary, j, chunkMinimum, chunkUpper);}
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<auto pp_, auto searchRows, class StageHi_Type, class StageIdx_Type, class StageLo_Type, class StageRefined_Type, class StageTileIdx_Type, class StageXyz_Type, class StageLowerScoresType, class StageSameType, class StageUnaryType, class StageJType, class StageChunkMinimumType, class StageChunkUpperType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SearchSpatialStaticStateBrcbStage(StageHi_Type& hi_, StageIdx_Type& idx_, StageLo_Type& lo_, StageRefined_Type& refined_, StageTileIdx_Type& tileIdx_, StageXyz_Type& xyz_, StageLowerScoresType& lowerScores, StageSameType& same, StageUnaryType& unary, StageJType& j, StageChunkMinimumType& chunkMinimum, StageChunkUpperType& chunkUpper) {
        PipeBarrier<PIPE_V>();

        Brcb(SpatialStateW(1),chunkUpper,(searchRows+7)/8,BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        const uint32_t group=j/256,slot=(j/64)%4;
        auto predicates=group<3?xyz_[group*pp_+ROWS*8]:refined_;
        Compare<float,uint8_t,false>(predicates[slot*128].template ReinterpretCast<uint8_t>(),lowerScores,SpatialStateW(1),CMPMODE::LE,uint64_t(64),searchRows,BinaryRepeatParams(1,1,0,8,8,1));

        PipeBarrier<PIPE_V>();

        Compare<float,false>(chunkMinimum,lo_,CMPMODE::LT,uint64_t(64),same);

        Adds<int32_t,false>(tileIdx_,tileIdx_,int32_t(j),uint64_t(64),1,unary);PipeBarrier<PIPE_V>();

        Min<float,false>(lo_,lo_,chunkMinimum,uint64_t(64),1,same);
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(hi_,chunkUpper,hi_,1,same);

        Select<float,SELMODE::VSEL_CMPMASK_SPR>(idx_.template ReinterpretCast<float>(),tileIdx_.template ReinterpretCast<float>(),idx_.template ReinterpretCast<float>(),1,same);PipeBarrier<PIPE_V>();


    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SpatialPruneW(uint32_t k) {
        constexpr uint32_t base=17952,pitch=1056;
        return LocalTensor<float>(TPosition::VECCALC,(base+k*pitch)*4,6*4224-k*pitch);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::SpatialPruneIndices(const LocalTensor<int32_t>& values,int32_t first,uint32_t count) {
        static_assert(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Full native Float pruning resource required");
        const LocalTensor<int32_t> rowBase(TPosition::VECCALC,186048U,64U);
        const uint32_t seed=count<64U?count:64U;
        ShiftRight(values,rowBase,int32_t(8),seed);PipeBarrier<PIPE_V>();
        if(first!=0) {Adds(values,values,first,seed);PipeBarrier<PIPE_V>();}
        for(uint32_t filled=64U;filled<count;filled*=2U) {
            const uint32_t n=count-filled<filled?count-filled:filled;
            Adds(values[filled],values,static_cast<int32_t>(filled),n);PipeBarrier<PIPE_V>();
        }
    }
#endif

template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
    __attribute__((noinline,minsize,aligned(128)))
#else
    __attribute__((noinline,aligned(128)))
#endif
    __aicore__ inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PruneSpatialStaticState(GM_ADDR workspace,uint32_t segments,bool& pruned,uint32_t& logicalSegments) {
        static_assert(sizeof(T)==4 && ROWS==64 && CAPACITY==1024 && COEFFICIENTS==1024 && !COMPACT_TAIL && !SINGLE_TAIL,"Full native Float spatial pruning state required");
        LocalTensor<float> work_(TPosition::VECCALC,71808U,25344U);
        LocalTensor<float> compactP_(TPosition::VECCALC,178304U,192U);
        LocalTensor<float> record_(TPosition::VECCALC,182400U,128U);
        LocalTensor<float> tileHi_(TPosition::VECCALC,185280U,64U);
        LocalTensor<uint8_t> mask_(TPosition::VECCALC,187072U,512U);
        LocalTensor<uint8_t> smallMask_(TPosition::VECCALC,188096U,32U);

        GlobalTensor<int32_t> spatialIndices;spatialIndices.SetGlobalBuffer((__gm__ int32_t*)workspace+6*4096+14*1056);
        // A data-selected genuine witness packet supplies the initial upper bound.
        // Search initializes and finalizes once after stable compaction.
        // C220 CompareScalar count form requires complete 64-float repeats.
        // Both temporary input arenas have 64 words; duplicate the last center
        // index in unused lanes so the first packed byte covers all six centers.
        auto center=work_[10112],threshold=work_[10160];
        auto broadcast=work_[11008],centerValues=work_[11072];
        auto centerOffsets=work_[11136].template ReinterpretCast<int32_t>();
        WholeReduceMin(record_,compactP_,32,6,8,1,4,ReduceOrder::ORDER_ONLY_VALUE);
        WholeReduceMax(record_[64],compactP_,32,6,8,1,4,ReduceOrder::ORDER_ONLY_VALUE);PipeBarrier<PIPE_V>();
        Add(center,record_,record_[64],48);PipeBarrier<PIPE_V>();
        Muls(center,center,0.5F,48);PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        // Use a valid geometric center without a scalar safety branch.
        // Recompute all radii and bounds from these same snapped coordinates.
        Duplicate(center[48],0.0F,16);PipeBarrier<PIPE_V>();
        Abs(SpatialPruneW(0),center,64);PipeBarrier<PIPE_V>();
        CompareScalar(mask_,SpatialPruneW(0),9.5367431640625e-7F,CMPMODE::GE,64);PipeBarrier<PIPE_V>();
        Select(center,mask_,center,0.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
#else
        SpatialPruneIndices(centerOffsets,0,64);PipeBarrier<PIPE_V>();
        Mins(centerOffsets,centerOffsets,int32_t(5),64);PipeBarrier<PIPE_V>();
        Muls(centerOffsets,centerOffsets,int32_t(32),64);PipeBarrier<PIPE_V>();
        Gather(centerValues,center,centerOffsets.template ReinterpretCast<uint32_t>(),0U,64);PipeBarrier<PIPE_V>();
        Abs(centerValues,centerValues,64);PipeBarrier<PIPE_V>();
        CompareScalar(mask_,centerValues,9.5367431640625e-7F,CMPMODE::LT,64);
        CompareScalar(smallMask_,centerValues,0.0F,CMPMODE::NE,64);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if((mask_.GetValue(0)&smallMask_.GetValue(0))!=0) {return COEFFICIENTS;}
#endif
        Duplicate(SpatialPruneW(4),0.0F,64);Duplicate(SpatialPruneW(3),0.0F,64);PipeBarrier<PIPE_V>();
        // Axis-major pairs describe disjoint halves of the same real query tile.
        // Broadcast slots are temporary until the shared witness gather begins.
        Brcb(broadcast,center,6,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(32));
        Sub<float,false>(SpatialPruneW(0),compactP_,broadcast,uint64_t(64),6,BinaryRepeatParams{1,1,0,8,4,8});
        Abs<float,false>(SpatialPruneW(1),compactP_,uint64_t(64),6,UnaryRepeatParams{1,1,8,4});PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
        Abs<float,false>(broadcast,broadcast,uint64_t(64),6,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(32));
        Add<float,false>(SpatialPruneW(1),SpatialPruneW(1),broadcast,uint64_t(64),6,BinaryRepeatParams{1,1,0,8,8,8});PipeBarrier<PIPE_V>();
        for(uint32_t axis=0;axis<3;++axis) {
            MulAddDst<float,float,false>(SpatialPruneW(4),SpatialPruneW(0)[axis*128],SpatialPruneW(0)[axis*128],uint64_t(64),2,BinaryRepeatParams{1,1,1,4,8,8});
            MulAddDst<float,float,false>(SpatialPruneW(3),SpatialPruneW(1)[axis*128],SpatialPruneW(1)[axis*128],uint64_t(64),2,BinaryRepeatParams{1,1,1,4,8,8});PipeBarrier<PIPE_V>();
        }
        // Radius covers the exact input coordinates around the rounded center.
        Muls(SpatialPruneW(3),SpatialPruneW(3),1.5258847270160913e-5F,64);PipeBarrier<PIPE_V>();
        Add(SpatialPruneW(4),SpatialPruneW(4),SpatialPruneW(3),64);PipeBarrier<PIPE_V>();
        Maxs(SpatialPruneW(4),SpatialPruneW(4),0.0F,64);PipeBarrier<PIPE_V>();
        Muls(SpatialPruneW(4),SpatialPruneW(4),1.0000152587890625F,64);return PruneSpatialStaticStateDefinedStage(spatialIndices, segments, pruned, logicalSegments, work_, compactP_, record_, tileHi_, center, threshold, broadcast);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSegmentsType, class StagePrunedType, class StageLogicalSegmentsType, class StageWork_Type, class StageCompactP_Type, class StageRecord_Type, class StageTileHi_Type, class StageCenterType, class StageThresholdType, class StageBroadcastType>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PruneSpatialStaticStateDefinedStage(GlobalTensor<int32_t>& spatialIndices, StageSegmentsType& segments, StagePrunedType& pruned, StageLogicalSegmentsType& logicalSegments, StageWork_Type& work_, StageCompactP_Type& compactP_, StageRecord_Type& record_, StageTileHi_Type& tileHi_, StageCenterType& center, StageThresholdType& threshold, StageBroadcastType& broadcast) {
        PipeBarrier<PIPE_V>();
        // This 64-word slot is dead until coefficient compaction begins.
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        auto queryRadius=work_[18432];
#else
        auto queryRadius=work_[10176];
#endif
        Sqrt(queryRadius,SpatialPruneW(4),64);PipeBarrier<PIPE_V>();
        auto savedCenterScores=work_[12288],otherCenterScores=work_[13312];
        // The radius phase has finished reading the temporary broadcasts.
        // Both center coordinates and error envelopes are shared across all segments.
        auto centerBroadcast=work_[11008],centerError=work_[15360];
        auto centerMagnitude=work_[15424],axisMaximum=work_[15552];
        auto axisEnvelope=work_[15616],centerErrorBroadcast=work_[15744];
        Brcb(centerBroadcast,center,6,BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
        Abs<float,false>(centerMagnitude,centerBroadcast,uint64_t(64),6,UnaryRepeatParams{1,1,1,8});
        Duplicate(centerError,0.0F,16);PipeBarrier<PIPE_V>();
        for(uint32_t axis=0;axis<3;++axis) {
            Brcb(axisMaximum,SpatialStateC(axis)[CAPACITY+16],1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
            Add<float,false>(axisEnvelope,axisMaximum,centerMagnitude[axis*16],uint64_t(64),2,BinaryRepeatParams{1,0,1,1,0,1});PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(centerError,axisEnvelope,axisEnvelope,uint64_t(64),2,BinaryRepeatParams{1,1,1,1,1,1});PipeBarrier<PIPE_V>();
        }
        Muls(centerError,centerError,1.5258847270160913e-5F,16);PipeBarrier<PIPE_V>();
        Brcb(centerErrorBroadcast,centerError,2,BrcbRepeatParams(1,8));return PruneSpatialStaticStateDefinedStage2(spatialIndices, segments, pruned, logicalSegments, work_, compactP_, record_, tileHi_, threshold, broadcast, queryRadius, savedCenterScores, otherCenterScores, centerBroadcast, centerErrorBroadcast);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSegmentsType, class StagePrunedType, class StageLogicalSegmentsType, class StageWork_Type, class StageCompactP_Type, class StageRecord_Type, class StageTileHi_Type, class StageThresholdType, class StageBroadcastType, class StageQueryRadiusType, class StageSavedCenterScoresType, class StageOtherCenterScoresType, class StageCenterBroadcastType, class StageCenterErrorBroadcastType>
    __aicore__ __attribute__((always_inline)) inline uint32_t KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PruneSpatialStaticStateDefinedStage2(GlobalTensor<int32_t>& spatialIndices, StageSegmentsType& segments, StagePrunedType& pruned, StageLogicalSegmentsType& logicalSegments, StageWork_Type& work_, StageCompactP_Type& compactP_, StageRecord_Type& record_, StageTileHi_Type& tileHi_, StageThresholdType& threshold, StageBroadcastType& broadcast, StageQueryRadiusType& queryRadius, StageSavedCenterScoresType& savedCenterScores, StageOtherCenterScoresType& otherCenterScores, StageCenterBroadcastType& centerBroadcast, StageCenterErrorBroadcastType& centerErrorBroadcast) {
        PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        // Each plane stores two independent 1024-lane centers plus 32 guard words.
        // Odd half-bank plane padding matches the retained scratch bank phase.
        // The old center scalars are dead; the live query radius uses a separate slot.
        const BinaryRepeatParams centerSame{1,1,1,8,8,8};
        const UnaryRepeatParams centerUnary{1,1,8,8};
        PruneSpatialStaticStateDefinedStage2SetMaskNormStage(work_, centerBroadcast, centerSame, centerUnary);
        PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(work_[8320],work_[6240],work_[4160],uint64_t(64),COEFFICIENTS/64,centerSame);
        MulAddDst<float,float,false>(work_[9344],work_[7264],work_[5184],uint64_t(64),COEFFICIENTS/64,centerSame);
        PipeBarrier<PIPE_V>();
        Sub<float,false>(work_[8320],work_[8320],centerErrorBroadcast[0],uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,1,0,8,8,0});
        Sub<float,false>(work_[9344],work_[9344],centerErrorBroadcast[64],uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,1,0,8,8,0});
        PipeBarrier<PIPE_V>();
        Copy<float,false>(work_[12288],work_[8320],uint64_t(64),COEFFICIENTS/64,CopyRepeatParams(1,1,8,8));
        Copy<float,false>(work_[13312],work_[9344],uint64_t(64),COEFFICIENTS/64,CopyRepeatParams(1,1,8,8));
        PipeBarrier<PIPE_V>();
#else
        // Both centers retain the original conservative score and arithmetic order.
        PruneSpatialStaticStateSetMaskNormStage(work_, centerBroadcast, centerErrorBroadcast);
#endif
        // Genuine witnesses are selected once for both groups.
        Min(SpatialPruneW(4),savedCenterScores,otherCenterScores,COEFFICIENTS);PipeBarrier<PIPE_V>();
        auto predicate=work_[10048].template ReinterpretCast<uint8_t>();
        // Distance to any real segment is 1-Lipschitz in the query point.
        // Padded coefficient records cannot seed the genuine witness packet.
        auto realCenterScores=SpatialPruneW(4);
        PruneSpatialStaticStateDefinedStage2SpatialPruneIndicesStage(segments, predicate, realCenterScores);
        // One value/index reduction selects the16 genuine witnesses.
        WholeReduceMin(record_,realCenterScores,64,COEFFICIENTS/64,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);PipeBarrier<PIPE_V>();
        // Reuse the original16 genuine value/index pairs at record_[0..31].
        // The witness pairs remain live until their indices are gathered.
        auto witnessPairs=record_;
        SpatialPruneIndices(SpatialPruneW(0).template ReinterpretCast<int32_t>(),0,16);PipeBarrier<PIPE_V>();
        Muls(SpatialPruneW(0).template ReinterpretCast<int32_t>(),SpatialPruneW(0).template ReinterpretCast<int32_t>(),int32_t(8),16);PipeBarrier<PIPE_V>();
        Adds(SpatialPruneW(0).template ReinterpretCast<int32_t>(),SpatialPruneW(0).template ReinterpretCast<int32_t>(),int32_t(4),16);PipeBarrier<PIPE_V>();
        Gather(SpatialPruneW(1).template ReinterpretCast<int32_t>(),witnessPairs.template ReinterpretCast<int32_t>(),SpatialPruneW(0).template ReinterpretCast<uint32_t>(),0U,16);PipeBarrier<PIPE_V>();
        Muls(SpatialPruneW(0).template ReinterpretCast<int32_t>(),SpatialPruneW(0).template ReinterpretCast<int32_t>(),int32_t(8),16);PipeBarrier<PIPE_V>();
        Adds(SpatialPruneW(0).template ReinterpretCast<int32_t>(),SpatialPruneW(0).template ReinterpretCast<int32_t>(),int32_t(-32),16);PipeBarrier<PIPE_V>();
        Add(SpatialPruneW(1).template ReinterpretCast<int32_t>(),SpatialPruneW(1).template ReinterpretCast<int32_t>(),SpatialPruneW(0).template ReinterpretCast<int32_t>(),16);PipeBarrier<PIPE_V>();
        Mins(SpatialPruneW(1).template ReinterpretCast<int32_t>(),SpatialPruneW(1).template ReinterpretCast<int32_t>(),int32_t(segments-1U),16);PipeBarrier<PIPE_V>();
        Muls(SpatialPruneW(1).template ReinterpretCast<int32_t>(),SpatialPruneW(1).template ReinterpretCast<int32_t>(),int32_t(4),16);PipeBarrier<PIPE_V>();
        auto rawWitness=work_[16384],witness=work_[11264];
        // Seven genuine coefficient planes; each scalar is expanded to8 lanes.
        auto witnessOffsets=work_[16512].template ReinterpretCast<int32_t>();
        PruneSpatialStaticStateDefinedStage2SpatialStateCStage(witnessOffsets);
        PipeBarrier<PIPE_V>();
        Gather(rawWitness,SpatialStateC(0),witnessOffsets.template ReinterpretCast<uint32_t>(),0U,112);
        PipeBarrier<PIPE_V>();
        for(uint32_t col=0;col<7;++col)
            Brcb(witness[col*128],rawWitness[col*16],2,BrcbRepeatParams(1,8));
        PipeBarrier<PIPE_V>();
        return PruneSpatialStaticStateSetMaskNormStage2(spatialIndices, segments, pruned, logicalSegments, work_, compactP_, record_, tileHi_, threshold, broadcast, queryRadius, savedCenterScores, otherCenterScores, predicate, witness);
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageWork_Type, class StageCenterBroadcastType, class StageCenterSameType, class StageCenterUnaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PruneSpatialStaticStateDefinedStage2SetMaskNormStage(StageWork_Type& work_, StageCenterBroadcastType& centerBroadcast, StageCenterSameType& centerSame, StageCenterUnaryType& centerUnary) {
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Sub<float,false>(work_[0],centerBroadcast[0],SpatialStateC(0),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        Sub<float,false>(work_[1024],centerBroadcast[64],SpatialStateC(0),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        Sub<float,false>(work_[2080],centerBroadcast[128],SpatialStateC(1),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        Sub<float,false>(work_[3104],centerBroadcast[192],SpatialStateC(1),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        Sub<float,false>(work_[4160],centerBroadcast[256],SpatialStateC(2),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        Sub<float,false>(work_[5184],centerBroadcast[320],SpatialStateC(2),uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        PipeBarrier<PIPE_V>();
        Mul<float,false>(work_[8320],work_[0],work_[0],uint64_t(64),COEFFICIENTS/64,centerSame);
        Mul<float,false>(work_[6240],work_[0],SpatialStateC(6),uint64_t(64),COEFFICIENTS/64,centerSame);
        Mul<float,false>(work_[9344],work_[1024],work_[1024],uint64_t(64),COEFFICIENTS/64,centerSame);
        Mul<float,false>(work_[7264],work_[1024],SpatialStateC(6),uint64_t(64),COEFFICIENTS/64,centerSame);
        PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(work_[8320],work_[2080],work_[2080],uint64_t(64),COEFFICIENTS/64,centerSame);
        MulAddDst<float,float,false>(work_[6240],work_[2080],SpatialStateC(7),uint64_t(64),COEFFICIENTS/64,centerSame);
        MulAddDst<float,float,false>(work_[9344],work_[3104],work_[3104],uint64_t(64),COEFFICIENTS/64,centerSame);
        MulAddDst<float,float,false>(work_[7264],work_[3104],SpatialStateC(7),uint64_t(64),COEFFICIENTS/64,centerSame);
        PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(work_[8320],work_[4160],work_[4160],uint64_t(64),COEFFICIENTS/64,centerSame);
        MulAddDst<float,float,false>(work_[9344],work_[5184],work_[5184],uint64_t(64),COEFFICIENTS/64,centerSame);
        PipeBarrier<PIPE_V>();
        FusedMulAdd<float,false>(work_[4160],SpatialStateC(8),work_[6240],uint64_t(64),COEFFICIENTS/64,centerSame);
        FusedMulAdd<float,false>(work_[5184],SpatialStateC(8),work_[7264],uint64_t(64),COEFFICIENTS/64,centerSame);
        PipeBarrier<PIPE_V>();
        Maxs<float,false>(work_[6240],work_[4160],0.0F,uint64_t(64),COEFFICIENTS/64,centerUnary);
        Maxs<float,false>(work_[7264],work_[5184],0.0F,uint64_t(64),COEFFICIENTS/64,centerUnary);
        PipeBarrier<PIPE_V>();
        Min<float,false>(work_[6240],work_[6240],SpatialStateC(9),uint64_t(64),COEFFICIENTS/64,centerSame);
        Min<float,false>(work_[7264],work_[7264],SpatialStateC(9),uint64_t(64),COEFFICIENTS/64,centerSame);
        PipeBarrier<PIPE_V>();
        FusedMulAdd<float,false>(work_[4160],SpatialStateC(4)[CAPACITY+8],work_[6240],uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
        FusedMulAdd<float,false>(work_[5184],SpatialStateC(4)[CAPACITY+8],work_[7264],uint64_t(64),COEFFICIENTS/64,BinaryRepeatParams{1,0,1,8,0,8});
    }
#endif
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageWitnessOffsetsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PruneSpatialStaticStateDefinedStage2SpatialStateCStage(StageWitnessOffsetsType& witnessOffsets) {
        for(uint32_t col=0;col<7;++col) {
            const uint32_t plane=col<3?col:col+3;
            const int32_t planeBytes=int32_t(uint32_t(SpatialStateC(plane).GetPhyAddr())-uint32_t(SpatialStateC(0).GetPhyAddr()));
            Adds(witnessOffsets[col*16],SpatialPruneW(1).template ReinterpretCast<int32_t>(),planeBytes,16);
        }
    }
#endif

#if defined(DTYPE_TREE_POINTS) && (TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS) || TREE_NATIVE_HALF_TOKEN(DTYPE_TREE_POINTS))
template<class T, uint32_t ROWS, uint32_t CAPACITY, uint32_t COEFFICIENTS, bool COMPACT_TAIL, bool SINGLE_TAIL>
    template<class StageSegmentsType, class StagePredicateType, class StageRealCenterScoresType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeBatch<T, ROWS, CAPACITY, COEFFICIENTS, COMPACT_TAIL, SINGLE_TAIL>::PruneSpatialStaticStateDefinedStage2SpatialPruneIndicesStage(StageSegmentsType& segments, StagePredicateType& predicate, StageRealCenterScoresType& realCenterScores) {
        if(segments<COEFFICIENTS) {
            SpatialPruneIndices(SpatialPruneW(0).template ReinterpretCast<int32_t>(),0,COEFFICIENTS);PipeBarrier<PIPE_V>();
            Cast(SpatialPruneW(1),SpatialPruneW(0).template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,COEFFICIENTS);PipeBarrier<PIPE_V>();
            CompareScalar(predicate,SpatialPruneW(1),float(int32_t(segments)),CMPMODE::LT,COEFFICIENTS);PipeBarrier<PIPE_V>();
            Select(SpatialPruneW(1),predicate,SpatialPruneW(4),__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,COEFFICIENTS);PipeBarrier<PIPE_V>();
            realCenterScores=SpatialPruneW(1);
        }
    }
#endif
