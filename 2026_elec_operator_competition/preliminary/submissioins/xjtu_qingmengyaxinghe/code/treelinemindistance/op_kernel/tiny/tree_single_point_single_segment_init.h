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
// Inline implementation of KernelTreeSinglePointSingleSegment.

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline void KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::Init(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                               const TreeLineMinDistanceTilingData& t) {
        t_ = t; rows_ = ROWS; pairs_ = ROWS==1?8:ROWS*PAIR_STRIDE; count_ = ROWS==4 && PAIR_STRIDE==1?32:64;
        pg_.SetGlobalBuffer((__gm__ T*)p); sg_.SetGlobalBuffer((__gm__ T*)s);
        dg_.SetGlobalBuffer((__gm__ T*)d); ig_.SetGlobalBuffer((__gm__ int32_t*)i);
        uint32_t address = 0;
        rawP_ = Alloc<T>(address, 256 / sizeof(T));
        rawS_ = Alloc<T>(address, 512 / sizeof(T));
        { fp_ = rawP_.template ReinterpretCast<float>(); fs_ = rawS_.template ReinterpretCast<float>(); }
        p_ = Alloc<float>(address, count_); a_ = Alloc<float>(address, count_); c_ = Alloc<float>(address, count_);
        work_ = Alloc<float>(address, count_ * 8);

        pointOffsets_ = Alloc<int32_t>(address, count_);
        segmentOffsets_ = Alloc<int32_t>(address, count_);
        lengths_ = Alloc<float>(address, 64); projection_ = Alloc<float>(address, 64);
        scores_ = Alloc<float>(address, 64); records_ = Alloc<float>(address, 64);
        out_ = Alloc<float>(address, 64); outT_ = Alloc<T>(address, 64);
        indices_ = Alloc<int32_t>(address, 64);
        masks_ = Alloc<uint8_t>(address, 512);
        zeroGeometry_=Alloc<float>(address,9*64);
        zeroExact_=Alloc<float>(address,45*64);


    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline void KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::Process() {
        const uint32_t pointCount=STATIC_ONE_POINT || (ROWS==1 && SINGLE_SEGMENT)?1U:t_.points;
        const uint32_t segmentCount=SINGLE_SEGMENT?1U:FIXED_SEGMENTS?FIXED_SEGMENTS:t_.segments;
        uint32_t batch,pointPosition,validRows;
        {
            // One-row tiles own consecutive flattened point/output positions.
            batch=pointCount==1?GetBlockIdx():GetBlockIdx()/pointCount;
            pointPosition=GetBlockIdx();
            validRows=1;
        }
        // The native single-segment load defines both padded endpoint
        // records; no preceding vector write is needed for this domain.


        DataCopyPad(rawP_,pg_[pointPosition*3],DataCopyExtParams{static_cast<uint16_t>(validRows),3U*sizeof(T),0,0,0},
                    DataCopyPadExtParams<T>{false,0,0,T(0)});

            DataCopyPad(rawS_,sg_[batch*segmentCount*6],DataCopyExtParams{static_cast<uint16_t>(2*segmentCount),3U*sizeof(T),0,0,0},
                        DataCopyPadExtParams<T>{false,0,0,T(0)});


        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);


        {
            // Padded native DMA records already form complete coordinate vectors.
            p_=fp_;a_=fs_;c_=fs_[slot_];
        }
        PipeBarrier<PIPE_V>();


        // The bounded certificate below permits ordinary FP32 differences;
        // difficult exponents, cancellation and ties use exact vector packets.
        count_=8;
        Sub(W(2),p_,a_,count_);
        PipeBarrier<PIPE_V>();
          ProcessSetMaskNormStage();
        ReduceTiny();
        if(!CertifiedTiny(validRows,segmentCount) && !TryExactDyadicGeometry(validRows,segmentCount)) {
            ExactFallback(pointPosition,batch,validRows,segmentCount);return;
        }
        Sqrt(out_,out_,rows_);
        PipeBarrier<PIPE_V>();
        outT_=out_;
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(dg_[pointPosition],outT_,DataCopyExtParams{1,validRows*uint32_t(sizeof(T)),0,0,0});
        DataCopyPad(ig_[pointPosition],indices_,DataCopyExtParams{1,validRows*4,0,0,0});
        // No subsequent UB reuse or GM overwrite follows this terminal write.
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessSetMaskNormStage() {
        {
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
            {
                const LocalTensor<float> products(TPosition::VECCALC,uint32_t(W(4).GetPhyAddr()),16);
                const LocalTensor<float> operands(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),24);
                Mul<float,false>(products,operands,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,1,2,0));
            }
            PipeBarrier<PIPE_V>();
            WholeReduceSum(lengths_,W(NATIVE_SINGLE_POINT?5:4),3,1,1,1,1);
            WholeReduceSum(projection_,W(NATIVE_SINGLE_POINT?4:5),3,1,1,1,1);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(1));
            // The certificate tests the original norm; clamping only the
            // divisor cannot admit an underflowed or degenerate segment.
            Maxs<float,false>(W(4),lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Div<float,false>(projection_,projection_,W(4),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Maxs<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Mins<float,false>(projection_,projection_,1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            ProjectSingleResidual(W(6),projection_,W(2),W(0),scores_,unary,same);
        }
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline void KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::DyadicLatticeFlags(const LocalTensor<float>& left,
            const LocalTensor<float>& right,uint32_t coordinates,bool packedInputs) {
        auto scaled=zeroExact_[1280],rounded=zeroExact_[1472],restored=zeroExact_[1664];
        auto flags=zeroExact_[880].template ReinterpretCast<uint8_t>();
        if(packedInputs && FIXED_SEGMENTS==2) {
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Muls<float,false>(scaled,p_,16.0F,uint64_t(64),3,UnaryRepeatParams(1,1,8,8));
        } else {
            Muls(scaled,p_,16.0F,coordinates);
            Muls(scaled[64],left,16.0F,coordinates);
            Muls(scaled[128],right,16.0F,coordinates);
        }
        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        const UnaryRepeatParams blocks{1,1,8,8};
        Cast<int32_t,float,false>(rounded.template ReinterpretCast<int32_t>(),scaled,RoundMode::CAST_RINT,uint64_t(64),3,blocks);PipeBarrier<PIPE_V>();
        Cast<float,int32_t,false>(restored,rounded.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,uint64_t(64),3,blocks);PipeBarrier<PIPE_V>();
        Muls<float,false>(restored,restored,0.0625F,uint64_t(64),3,blocks);PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8};
        Compare<float,uint8_t,false>(flags,p_,restored,CMPMODE::EQ,uint64_t(64),1,same);
        Compare<float,uint8_t,false>(flags[64],left,restored[64],CMPMODE::EQ,uint64_t(64),1,same);
        Compare<float,uint8_t,false>(flags[128],right,restored[128],CMPMODE::EQ,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline bool KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::TryExactDyadicGeometry(uint32_t validRows,uint32_t segments,bool inputReady) {
        // Matched integer reconstruction certifies a 2^-4 coordinate lattice.
        // A difference bounded by 64 is exact: small operands fit binary32,
        // and larger nearby same-sign operands satisfy Sterbenz's lemma.
        // Relative/direction bounds make dots and squared norms exact (22 bits).
        // Endpoint projections and axis-aligned segments therefore admit an
        // exact argmin, including zero length, zero distance and true ties.
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        const uint32_t pairs=ROWS==1?segments:validRows*PAIR_STRIDE;
        const uint32_t coordinates=pairs*8;
        const uint64_t prefix=coordinates==64?~uint64_t(0):((uint64_t(1)<<coordinates)-1);
        const uint64_t liveCoordinates=prefix&((ROWS>1 && PAIR_STRIDE==2 && segments==1)?
            0x0007000700070007ULL:0x0707070707070707ULL);
        const uint32_t livePairs=((1U<<pairs)-1U)&
            ((ROWS>1 && PAIR_STRIDE==2 && segments==1)?0x55U:0xffU);
        TreeDyadicScratch arena(zeroExact_);
        if(inputReady) {arena.left=a_;arena.right=c_;}
        else {
        {
            Copy(arena.left,fs_,PAIR_STRIDE*8,validRows,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
            Copy(arena.right,fs_[slot_],PAIR_STRIDE*8,validRows,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
        }
        PipeBarrier<PIPE_V>();
        }
        DyadicLatticeFlags(arena.left,arena.right,coordinates,inputReady && count_==64);
        Sub(arena.delta,arena.right,arena.left,coordinates);Sub(arena.relA,p_,arena.left,coordinates);Sub(arena.relC,p_,arena.right,coordinates);PipeBarrier<PIPE_V>();
        {
            Abs(arena.temp,arena.delta,coordinates);Abs(arena.rounded,arena.relA,coordinates);Abs(arena.product,arena.relC,coordinates);
        }
        PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        CompareScalar<float,uint8_t,false>(arena.flags[96],arena.temp,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        CompareScalar<float,uint8_t,false>(arena.flags[128],arena.rounded,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        CompareScalar<float,uint8_t,false>(arena.flags[160],arena.product,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        Compare<float,uint8_t,false>(arena.flags,arena.left,arena.right,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const auto words=arena.flags.template ReinterpretCast<uint64_t>();
        const uint64_t admitted=words.GetValue(24)&words.GetValue(32)&words.GetValue(40);
        return TryExactDyadicGeometryGetValueStage(inputReady, pairs, coordinates, liveCoordinates, livePairs, arena.left, arena.right, arena.delta, arena.relA, arena.relC, arena.temp, arena.product, arena.norm, arena.dot, arena.endpoint, arena.box, arena.upper, arena.flags, words, admitted);
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageInputReadyType, class StagePairsType, class StageCoordinatesType, class StageLiveCoordinatesType, class StageLivePairsType, class StageLeftType, class StageRightType, class StageDeltaType, class StageRelAType, class StageRelCType, class StageTempType, class StageProductType, class StageNormType, class StageDotType, class StageEndpointType, class StageBoxType, class StageUpperType, class StageFlagsType, class StageWordsType, class StageAdmittedType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::TryExactDyadicGeometryGetValueStage(StageInputReadyType& inputReady, StagePairsType& pairs, StageCoordinatesType& coordinates, StageLiveCoordinatesType& liveCoordinates, StageLivePairsType& livePairs, StageLeftType& left, StageRightType& right, StageDeltaType& delta, StageRelAType& relA, StageRelCType& relC, StageTempType& temp, StageProductType& product, StageNormType& norm, StageDotType& dot, StageEndpointType& endpoint, StageBoxType& box, StageUpperType& upper, StageFlagsType& flags, StageWordsType& words, StageAdmittedType& admitted) {
        const uint64_t bounded=words.GetValue(12)&words.GetValue(16)&words.GetValue(20);
        if(((admitted&bounded)&liveCoordinates)!=liveCoordinates)return false;
        const uint64_t fixed=words.GetValue(0);
        const uint64_t axis=((fixed&(fixed>>1))|(fixed&(fixed>>2))|((fixed>>1)&(fixed>>2)))&0x0101010101010101ULL;
        const uint32_t axisPairs=inputReady?livePairs:uint32_t((axis*0x0102040810204080ULL)>>56);
        bool tryExactDyadicGeometryMinStageResult{};
        if (TryExactDyadicGeometryMinStage(pairs, coordinates, livePairs, left, right, product, box, flags, axisPairs, tryExactDyadicGeometryMinStageResult)) return tryExactDyadicGeometryMinStageResult;
        Mul(product,delta,delta,coordinates);Mul(temp,relA,delta,coordinates);PipeBarrier<PIPE_V>();
        WholeReduceSum(norm,product,3,pairs,1,1,1);
        WholeReduceSum(dot,temp,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
        CompareScalar<float,uint8_t,false>(flags[32],dot,0.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        Compare<float,uint8_t,false>(flags[64],dot,norm,CMPMODE::GE,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint32_t endpoints=uint32_t(words.GetValue(4)|words.GetValue(8));
        bool tryExactDyadicGeometryReinterpretCastStageResult{};
        if (TryExactDyadicGeometryReinterpretCastStage(pairs, coordinates, livePairs, left, right, delta, relA, relC, temp, product, norm, dot, endpoint, upper, flags, words, axisPairs, endpoints, tryExactDyadicGeometryReinterpretCastStageResult)) return tryExactDyadicGeometryReinterpretCastStageResult;
        CommonDyadicGeometry54(product, relA, coordinates, temp, relC, endpoint, pairs, upper, box, left, right);
        Max(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
        Min(box,box,product,coordinates);Min(endpoint,endpoint,upper,pairs);PipeBarrier<PIPE_V>();
        Sub(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
        Mul(box,box,box,coordinates);PipeBarrier<PIPE_V>();
        WholeReduceSum(upper,box,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
        Duplicate(flags.template ReinterpretCast<uint32_t>(),axisPairs,2);PipeBarrier<PIPE_V>();
        SetCmpMask(flags);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(endpoint,upper,endpoint,1,BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
        Duplicate(flags.template ReinterpretCast<uint32_t>(),livePairs,2);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Select(scores_,flags,endpoint,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        ReduceTiny();
        return true;

    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StagePairsType, class StageCoordinatesType, class StageLivePairsType, class StageLeftType, class StageRightType, class StageDeltaType, class StageRelAType, class StageRelCType, class StageTempType, class StageProductType, class StageNormType, class StageDotType, class StageEndpointType, class StageUpperType, class StageFlagsType, class StageWordsType, class StageAxisPairsType, class StageEndpointsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::TryExactDyadicGeometryReinterpretCastStage(StagePairsType& pairs, StageCoordinatesType& coordinates, StageLivePairsType& livePairs, StageLeftType& left, StageRightType& right, StageDeltaType& delta, StageRelAType& relA, StageRelCType& relC, StageTempType& temp, StageProductType& product, StageNormType& norm, StageDotType& dot, StageEndpointType& endpoint, StageUpperType& upper, StageFlagsType& flags, StageWordsType& words, StageAxisPairsType& axisPairs, StageEndpointsType& endpoints, bool& stageResult) {
        if(((axisPairs|endpoints)&livePairs)!=livePairs) {
            // On this lattice every cross-product subtraction is exact. A
            // certified zero beats every nonzero distance without any division.
            auto u=zeroExact_[896],v=zeroExact_[960],cross=zeroExact_[1024],crossNorm=zeroExact_[1088];
            auto rotate1=zeroExact_[1152].template ReinterpretCast<int32_t>();
            auto rotate2=zeroExact_[1216].template ReinterpretCast<int32_t>();
            MakeIndices(rotate1,64);PipeBarrier<PIPE_V>();
            ShiftRight(rotate1,rotate1,int32_t(3),64);PipeBarrier<PIPE_V>();
            Muls(rotate1,rotate1,int32_t(32),64);PipeBarrier<PIPE_V>();
            Adds(rotate2,rotate1,int32_t(8),64);Adds(rotate1,rotate1,int32_t(4),64);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x0202020202020202ULL));
            CommonDyadicGeometry51(rotate1, rotate2, coordinates, u, relA, v, delta, cross, crossNorm, pairs, product, endpoint);
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
            CompareScalar<float,uint8_t,false>(flags,crossNorm,0.0F,CMPMODE::EQ,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            CompareScalar<float,uint8_t,false>(flags[32],dot,0.0F,CMPMODE::GE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            Compare<float,uint8_t,false>(flags[64],dot,norm,CMPMODE::LE,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            CompareScalar<float,uint8_t,false>(flags[96],norm,0.0F,CMPMODE::GT,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            CompareScalar<float,uint8_t,false>(flags[128],endpoint,0.0F,CMPMODE::EQ,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t zeroPairs=uint32_t(words.GetValue(0)&words.GetValue(4)&words.GetValue(8)&
                (words.GetValue(12)|words.GetValue(16)))&livePairs;
            const uint32_t firstPairs=ROWS==1?1U:(PAIR_STRIDE==2?(livePairs&0x55U):livePairs);
            bool tryExactDyadicGeometryDuplicateStageResult{};
        if (TryExactDyadicGeometryDuplicateStage(pairs, coordinates, livePairs, left, right, relC, temp, norm, endpoint, upper, flags, words, endpoints, u, v, crossNorm, zeroPairs, firstPairs, tryExactDyadicGeometryDuplicateStageResult)) { stageResult = tryExactDyadicGeometryDuplicateStageResult; return true; }}
        return false;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StagePairsType, class StageCoordinatesType, class StageLivePairsType, class StageLeftType, class StageRightType, class StageRelCType, class StageTempType, class StageNormType, class StageEndpointType, class StageUpperType, class StageFlagsType, class StageWordsType, class StageEndpointsType, class StageUType, class StageVType, class StageCrossNormType, class StageZeroPairsType, class StageFirstPairsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::TryExactDyadicGeometryDuplicateStage(StagePairsType& pairs, StageCoordinatesType& coordinates, StageLivePairsType& livePairs, StageLeftType& left, StageRightType& right, StageRelCType& relC, StageTempType& temp, StageNormType& norm, StageEndpointType& endpoint, StageUpperType& upper, StageFlagsType& flags, StageWordsType& words, StageEndpointsType& endpoints, StageUType& u, StageVType& v, StageCrossNormType& crossNorm, StageZeroPairsType& zeroPairs, StageFirstPairsType& firstPairs, bool& stageResult) {
        const bool allRowsZero=ROWS==1?zeroPairs!=0U:
                (PAIR_STRIDE==2?(((zeroPairs|(zeroPairs>>1))&firstPairs)==firstPairs):zeroPairs==livePairs);
            if(allRowsZero) {
                Duplicate(endpoint,0.0F,64);
                Duplicate(flags.template ReinterpretCast<uint32_t>(),zeroPairs,2);PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Select(scores_,flags,endpoint,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
                ReduceTiny();{ stageResult = true; return true; }
            }
            // Equal undirected segments have equal mathematical distances for
            // every point. Compare original endpoints before forcing index zero.
            Copy(u,left,uint64_t(8),pairs,CopyRepeatParams(1,1,1,0));
            Copy(v,right,uint64_t(8),pairs,CopyRepeatParams(1,1,1,0));PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Compare<float,uint8_t,false>(flags,left,u,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            Compare<float,uint8_t,false>(flags[32],right,v,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            Compare<float,uint8_t,false>(flags[64],left,v,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            Compare<float,uint8_t,false>(flags[96],right,u,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
            if(!HasIdenticalDyadicPairs(words,livePairs)){ stageResult = false; return true; }
            // Positive cross norms avoid projection cancellation; float
            // multiply/sum/divide contributes far less than the declared tolerance.
            CommonDyadicGeometry53(temp, relC, coordinates, upper, pairs, endpoint, norm, crossNorm, flags, endpoints);
            SetCmpMask(flags);SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(endpoint,endpoint,upper,1,BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
            Duplicate(flags.template ReinterpretCast<uint32_t>(),firstPairs,2);PipeBarrier<PIPE_V>();
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Select(scores_,flags,endpoint,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            ReduceTiny();{ stageResult = true; return true; }

        return false;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StagePairsType, class StageCoordinatesType, class StageLivePairsType, class StageLeftType, class StageRightType, class StageProductType, class StageBoxType, class StageFlagsType, class StageAxisPairsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::TryExactDyadicGeometryMinStage(StagePairsType& pairs, StageCoordinatesType& coordinates, StageLivePairsType& livePairs, StageLeftType& left, StageRightType& right, StageProductType& product, StageBoxType& box, StageFlagsType& flags, StageAxisPairsType& axisPairs, bool& stageResult) {
        if((axisPairs&livePairs)==livePairs) {
            // An axis-aligned segment equals its AABB. The verified 2^-4
            // lattice and bounded differences make these squared norms exact.
            Min(box,left,right,coordinates);Max(product,left,right,coordinates);PipeBarrier<PIPE_V>();
            Max(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
            Min(box,box,product,coordinates);PipeBarrier<PIPE_V>();
            Sub(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
            Mul(box,box,box,coordinates);Duplicate(scores_,__builtin_inff(),64);PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,box,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
            if(livePairs!=((1U<<pairs)-1U)) {
                Duplicate(flags.template ReinterpretCast<uint32_t>(),livePairs,2);PipeBarrier<PIPE_V>();
                SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Select(scores_,flags,scores_,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            }
            SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            ReduceTiny();{ stageResult = true; return true; }
        }
        return false;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline void KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ExactFallback(uint32_t position,uint32_t batch,uint32_t rows,uint32_t segments) {
        TreeTinyExactFallback<T>((GM_ADDR)pg_[position*3].GetPhyAddr(),
            (GM_ADDR)sg_[batch*segments*6].GetPhyAddr(),(GM_ADDR)dg_[position].GetPhyAddr(),
            (GM_ADDR)ig_[position].GetPhyAddr(),rows,segments);
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline bool KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::CertifiedTiny(uint32_t validRows,uint32_t segments) {
        // In this bounded domain, the norm error is below 2^-13 and the
        // squared-score error below 2^-6. A gap > 2^-5 proves the winner;
        // the admitted residual range also leaves the specified FP32 tolerance margin.
        {
            return IsCertifiedResidualRange(lengths_,scores_);
        }
        auto flags=zeroGeometry_.template ReinterpretCast<uint8_t>();
        const uint32_t n=SINGLE_SEGMENT?validRows:ROWS==1?segments:validRows*PAIR_STRIDE;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams same{1,1,1,8,8,8};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(n));
        CompareScalar<float,uint8_t,false>(flags,lengths_,0.000244140625F,CMPMODE::GE,uint64_t(64),1,unary);
        CompareScalar<float,uint8_t,false>(flags[32],lengths_,1024.0F,CMPMODE::LE,uint64_t(64),1,unary);
        CompareScalar<float,uint8_t,false>(flags[64],scores_,ROWS>1 && !SINGLE_SEGMENT?0.25F:1.0F,CMPMODE::GE,uint64_t(64),1,unary);
        CompareScalar<float,uint8_t,false>(flags[96],scores_,1024.0F,CMPMODE::LE,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto words=flags.template ReinterpretCast<uint32_t>();
        const uint32_t active=(1U<<n)-1U;
        const uint32_t safe=words.GetValue(0)&words.GetValue(8)&words.GetValue(16)&words.GetValue(24);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        if((safe&active)!=active)return false;

        return true;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline void KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ReduceTiny() {
        {
            out_=scores_;
            Duplicate(indices_,int32_t(0),ROWS);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class U> __aicore__ inline LocalTensor<U> KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::Alloc(uint32_t& address,uint32_t n) {
        uint32_t bytes=(n*sizeof(U)+31)/32*32;
        LocalTensor<U> v(TPosition::VECCALC,address,bytes/sizeof(U));address+=bytes;return v;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline LocalTensor<float> KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::W(uint32_t k) {
        {
            if(k<4)return work_[(k^2U)*count_];
        }

        return work_[k*(NATIVE_HALF_PACKET?64U:count_)];
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<int32_t STEP> __aicore__ inline void KernelTreeSinglePointSingleSegment<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::MakeIndices(const LocalTensor<int32_t>& v,uint32_t n) {
        Duplicate(v,int32_t(0),8);PipeBarrier<PIPE_V>();SetMaskNorm();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaa));
        Adds<int32_t,false>(v,v,int32_t(STEP),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xcc));
        Adds<int32_t,false>(v,v,int32_t(2*STEP),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xf0));
        Adds<int32_t,false>(v,v,int32_t(4*STEP),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        for(uint32_t filled=8;filled<n;filled*=2){Adds(v[filled],v,int32_t(filled*STEP),filled);PipeBarrier<PIPE_V>();}
    }
