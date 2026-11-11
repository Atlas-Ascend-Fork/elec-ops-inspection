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
// Inline implementation of KernelTreeTiny.

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StagePairsType, class StageCoordinatesType, class StageLivePairsType, class StageLeftType, class StageRightType, class StageRelCType, class StageTempType, class StageNormType, class StageEndpointType, class StageUpperType, class StageFlagsType, class StageWordsType, class StageEndpointsType, class StageUType, class StageVType, class StageCrossNormType, class StageZeroPairsType, class StageFirstPairsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ComputeExactDyadicGeometryDuplicateStage(StagePairsType& pairs, StageCoordinatesType& coordinates, StageLivePairsType& livePairs, StageLeftType& left, StageRightType& right, StageRelCType& relC, StageTempType& temp, StageNormType& norm, StageEndpointType& endpoint, StageUpperType& upper, StageFlagsType& flags, StageWordsType& words, StageEndpointsType& endpoints, StageUType& u, StageVType& v, StageCrossNormType& crossNorm, StageZeroPairsType& zeroPairs, StageFirstPairsType& firstPairs, bool& stageResult) {
        const bool allRowsZero=ROWS==1?zeroPairs!=0U:
                (PAIR_STRIDE==2?(((zeroPairs|(zeroPairs>>1))&firstPairs)==firstPairs):zeroPairs==livePairs);
            if(allRowsZero) {
                Duplicate(endpoint,0.0F,64);
                Duplicate(flags.template ReinterpretCast<uint32_t>(),zeroPairs,2);PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Select(scores_,flags,endpoint,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
                ReduceTiny();{ stageResult = true; return true; }
            }
            // Equal undirected segments have equal mathematical distances for
            // every point. Compare original endpoints before forcing index zero.
            Copy(u,left,uint64_t(8),pairs,CopyRepeatParams(1,1,1,0));
            Copy(v,right,uint64_t(8),pairs,CopyRepeatParams(1,1,1,0));PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Compare<float,uint8_t,false>(flags,left,u,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            Compare<float,uint8_t,false>(flags[32],right,v,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            Compare<float,uint8_t,false>(flags[64],left,v,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            Compare<float,uint8_t,false>(flags[96],right,u,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            if(!HasIdenticalDyadicPairs(words,livePairs)){ stageResult = false; return true; }
            // Positive cross norms avoid projection cancellation; float
            // multiply/sum/divide contributes far less than the declared tolerance.
            CommonDyadicGeometry53(temp, relC, coordinates, upper, pairs, endpoint, norm, crossNorm, flags, endpoints);
            SetCmpMask(flags);SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(endpoint,endpoint,upper,1,BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
            Duplicate(flags.template ReinterpretCast<uint32_t>(),firstPairs,2);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Select(scores_,flags,endpoint,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            ReduceTiny();{ stageResult = true; return true; }

        return false;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageValidRowsType, class StageInputReadyType, class StageCoordinatesType, class StageLeftType, class StageRightType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ComputeExactDyadicGeometryGatherStage(StageValidRowsType& validRows, StageInputReadyType& inputReady, StageCoordinatesType& coordinates, StageLeftType& left, StageRightType& right) {
        if(inputReady) {left=a_;right=c_;}
        else {
        if constexpr(CONTIG_SEGMENTS) {
            auto sourceOffsets=BLOCK_INDICES?pointOffsets_:segmentOffsets_;
            Gather(left,fs_,sourceOffsets.template ReinterpretCast<uint32_t>(),0U,coordinates);
            Gather(right,fs_,sourceOffsets.template ReinterpretCast<uint32_t>(),12U,coordinates);
        } else {
            Copy(left,fs_,PAIR_STRIDE*8,validRows,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
            Copy(right,fs_[slot_],PAIR_STRIDE*8,validRows,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
        }
        PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StagePairsType, class StageCoordinatesType, class StageLivePairsType, class StageLeftType, class StageRightType, class StageProductType, class StageBoxType, class StageFlagsType, class StageAxisPairsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ComputeExactDyadicGeometryMinStage(StagePairsType& pairs, StageCoordinatesType& coordinates, StageLivePairsType& livePairs, StageLeftType& left, StageRightType& right, StageProductType& product, StageBoxType& box, StageFlagsType& flags, StageAxisPairsType& axisPairs, bool& stageResult) {
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
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                Select(scores_,flags,scores_,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            }
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            ReduceTiny();{ stageResult = true; return true; }
        }
        return false;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ExactFallback(uint32_t position,uint32_t batch,uint32_t rows,uint32_t segments) {
        TreeTinyExactFallback<T>((GM_ADDR)pg_[position*3].GetPhyAddr(),
            (GM_ADDR)sg_[batch*segments*6].GetPhyAddr(),(GM_ADDR)dg_[position].GetPhyAddr(),
            (GM_ADDR)ig_[position].GetPhyAddr(),rows,segments);
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::CertifiedTiny(uint32_t validRows,uint32_t segments) {
        // In this bounded domain, the norm error is below 2^-13 and the
        // squared-score error below 2^-6. A gap > 2^-5 proves the winner;
        // the admitted residual range also leaves the specified FP32 tolerance margin.
        if constexpr(ROWS==1 && SINGLE_SEGMENT) {
            return IsCertifiedResidualRange(lengths_,scores_);
        }
        auto flags=zeroGeometry_.template ReinterpretCast<uint8_t>();
        const uint32_t n=SINGLE_SEGMENT?validRows:ROWS==1?segments:validRows*PAIR_STRIDE;
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams same{1,1,1,8,8,8};
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(n));
        CompareScalar<float,uint8_t,false>(flags,lengths_,0.000244140625F,CMPMODE::GE,uint64_t(64),1,unary);
        CompareScalar<float,uint8_t,false>(flags[32],lengths_,1024.0F,CMPMODE::LE,uint64_t(64),1,unary);
        CompareScalar<float,uint8_t,false>(flags[64],scores_,ROWS>1 && !SINGLE_SEGMENT?0.25F:1.0F,CMPMODE::GE,uint64_t(64),1,unary);
        CompareScalar<float,uint8_t,false>(flags[96],scores_,1024.0F,CMPMODE::LE,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        if constexpr(!SINGLE_SEGMENT) {
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            if constexpr(ROWS==1) {
                Brcb(W(6),out_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
                Adds<float,false>(W(6),W(6),0.03125F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Compare<float,uint8_t,false>(flags[128],scores_,W(6),CMPMODE::LE,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            } else {
                // The pair reduction already supplies each row's larger score.
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(validRows));
                Adds<float,false>(W(7),out_,0.03125F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Compare<float,uint8_t,false>(flags[128],W(6),W(7),CMPMODE::GT,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            }
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto words=flags.template ReinterpretCast<uint32_t>();
        const uint32_t active=(1U<<n)-1U;
        const uint32_t safe=words.GetValue(0)&words.GetValue(8)&words.GetValue(16)&words.GetValue(24);
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        if((safe&active)!=active)return false;
        if constexpr(!SINGLE_SEGMENT) {
            const uint32_t close=words.GetValue(32)&active;
            if constexpr(ROWS==1)return close && !(close&(close-1U));
            else return (close&((1U<<validRows)-1U))==((1U<<validRows)-1U);
        }
        return true;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ReduceTiny() {
        if constexpr(SINGLE_SEGMENT) {
            if constexpr(ROWS==1 || PAIR_STRIDE==1)out_=scores_;
            else GatherMask(out_,scores_,uint8_t(1),GatherMaskParams(1,1,8,0));
            Duplicate(indices_,int32_t(0),ROWS);PipeBarrier<PIPE_V>();
        } else if constexpr(ROWS==1) {
            WholeReduceMin(records_,scores_,FIXED_SEGMENTS?FIXED_SEGMENTS:8,rows_,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);
            PipeBarrier<PIPE_V>();
            out_=records_;
            GatherMask(indices_.template ReinterpretCast<float>(),records_,uint8_t(2),GatherMaskParams(1,1,8,0));
            PipeBarrier<PIPE_V>();
        } else if constexpr(UsesRowsStage()) {
            // Gather overwrites every live result lane; four rows fit a single packet.
            GatherMask(out_,scores_,uint8_t(1),GatherMaskParams(1,1,8,0));
            GatherMask(W(4),scores_,uint8_t(2),GatherMaskParams(1,1,8,0));
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(8));
            Duplicate<int32_t,false>(indices_,int32_t(0),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
            const BinaryRepeatParams same{1,1,1,8,8,8};
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(4));
            Compare<float>(out_,W(4),CMPMODE::LE,uint64_t(4),same);PipeBarrier<PIPE_V>();
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(indices_.template ReinterpretCast<float>(),indices_.template ReinterpretCast<float>(),segmentOffsets_.template ReinterpretCast<float>(),1,same);
            Max<float,false>(W(6),out_,W(4),uint64_t(64),1,same);
            Min<float,false>(out_,out_,W(4),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        } else {
            Duplicate(out_,__builtin_inff(),64);
            Duplicate(indices_,int32_t(0),64);PipeBarrier<PIPE_V>();
            GatherMask(out_,scores_,uint8_t(1),GatherMaskParams(1,1,8,0));
            GatherMask(W(4),scores_,uint8_t(2),GatherMaskParams(1,1,8,0));PipeBarrier<PIPE_V>();
            const BinaryRepeatParams same{1,1,1,8,8,8};

            Compare<float>(out_,W(4),CMPMODE::LE,uint64_t(64),same);PipeBarrier<PIPE_V>();
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(indices_.template ReinterpretCast<float>(),indices_.template ReinterpretCast<float>(),segmentOffsets_.template ReinterpretCast<float>(),1,same);
            Max(W(6),out_,W(4),32);
            Min(out_,out_,W(4),32);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class U> __aicore__ inline LocalTensor<U> KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::Alloc(uint32_t& address,uint32_t n) {
        uint32_t bytes=(n*sizeof(U)+31)/32*32;
        LocalTensor<U> v(TPosition::VECCALC,address,bytes/sizeof(U));address+=bytes;return v;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline LocalTensor<float> KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::W(uint32_t k) {
        if constexpr(NATIVE_HALF_PACKET) {
            if(k<4)return work_[(k^2U)*64U];
        } else if constexpr(NATIVE_ONE_ROW || NATIVE_FOUR_ROW_SINGLE || NATIVE_SINGLE_POINT) {
            if(k<4)return work_[(k^2U)*count_];
        }
        if constexpr(MAPPED_TINY_STARTS) { if(k==4)return originalA_; }
        return work_[k*(NATIVE_HALF_PACKET?64U:count_)];
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<int32_t STEP> __aicore__ inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::MakeIndices(const LocalTensor<int32_t>& v,uint32_t n) {
        Duplicate(v,int32_t(0),8);PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xaa));
        Adds<int32_t,false>(v,v,int32_t(STEP),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xcc));
        Adds<int32_t,false>(v,v,int32_t(2*STEP),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0xf0));
        Adds<int32_t,false>(v,v,int32_t(4*STEP),uint64_t(64),1,UnaryRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(64));
        for(uint32_t filled=8;filled<n;filled*=2){Adds(v[filled],v,int32_t(filled*STEP),filled);PipeBarrier<PIPE_V>();}
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::Difference(const LocalTensor<float>& x,const LocalTensor<float>& y,
                                      const LocalTensor<float>& hi,const LocalTensor<float>& lo) {
        Sub(hi,x,y,count_);PipeBarrier<PIPE_V>();
        Sub(W(6),x,hi,count_);PipeBarrier<PIPE_V>();
        Add(W(7),hi,W(6),count_);Sub(lo,W(6),y,count_);PipeBarrier<PIPE_V>();
        Sub(W(7),x,W(7),count_);PipeBarrier<PIPE_V>();
        Add(lo,lo,W(7),count_);PipeBarrier<PIPE_V>();
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::Canonicalize() {
        auto less=masks_.template ReinterpretCast<uint16_t>();
        auto eq=masks_[64].template ReinterpretCast<uint16_t>();
        auto keep=masks_[128].template ReinterpretCast<uint16_t>();
        auto tmp=masks_[192].template ReinterpretCast<uint16_t>();
        auto ones=masks_[256].template ReinterpretCast<uint16_t>();
        const UnaryRepeatParams unary{1,1,8,8};
        const BinaryRepeatParams binary{1,1,1,8,8,8};
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(NATIVE_HALF_PACKET?16:64));
        Compare<float,uint8_t,false>(masks_,a_,c_,CMPMODE::LT,uint64_t(64),1,binary);
        Compare<float,uint8_t,false>(masks_[64],a_,c_,CMPMODE::EQ,uint64_t(64),1,binary);PipeBarrier<PIPE_V>();
        CanonicalizeGetValueStage(less, eq, keep, tmp, ones, unary, binary);
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(NATIVE_HALF_PACKET?16:64));SetCmpMask(masks_[128]);
        if constexpr(MAPPED_TINY_STARTS)
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(work_[4*count_],a_,c_,1,binary);
        else Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(4),a_,c_,1,binary);
        PipeBarrier<PIPE_V>();
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(c_,c_,a_,1,binary);PipeBarrier<PIPE_V>();
        if constexpr(MAPPED_TINY_STARTS) {
            // Original A is dead after canonical B consumes it. Keep canonical
            // A in its selected plane and reuse old A for ordinary W4 scratch.
            // CorrectTinyZeroScores receives the explicit A address and only
            // uses work planes 0, 1 and 5, so it cannot overwrite canonical A.
            a_=work_[4*count_];
        } else {Copy(a_,W(4),64,1,CopyRepeatParams(1,1,8,8));PipeBarrier<PIPE_V>();}
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageLessType, class StageEqType, class StageKeepType, class StageTmpType, class StageOnesType, class StageUnaryType, class StageBinaryType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::CanonicalizeGetValueStage(StageLessType& less, StageEqType& eq, StageKeepType& keep, StageTmpType& tmp, StageOnesType& ones, StageUnaryType& unary, StageBinaryType& binary) {
        if constexpr(NATIVE_HALF_PACKET) {
            // Float metadata bounds reuse the original canonical comparison barrier.
            // Two packed eight-lane records use only the first predicate word.
            // Merge predicate metadata after the vector comparisons complete.
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t lt=less.GetValue(0),equal=eq.GetValue(0);
            const uint16_t packet=uint16_t((lt|((lt>>1)&equal)|((lt>>2)&equal&(equal>>1)))&0x0101U);
            keep.SetValue(0,uint16_t(packet*255U));
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        } else if constexpr(NATIVE_ONE_ROW && sizeof(T)==4) {
            // Eight coordinate records occupy one 64-bit predicate word.
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if constexpr(FIXED_SEGMENTS==2) {
                // Only the two real coordinate records feed this bounded resource.
                const uint32_t lt=less.template ReinterpretCast<uint32_t>().GetValue(0);
                const uint32_t equal=eq.template ReinterpretCast<uint32_t>().GetValue(0);
                const uint32_t packet=(lt|((lt>>1)&equal)|((lt>>2)&equal&(equal>>1)))&0x0101U;
                keep.template ReinterpretCast<uint64_t>().SetValue(0,uint64_t(packet*255U));
            } else {
                const uint64_t lt=less.template ReinterpretCast<uint64_t>().GetValue(0);
                const uint64_t equal=eq.template ReinterpretCast<uint64_t>().GetValue(0);
                const uint64_t packet=(lt|((lt>>1)&equal)|((lt>>2)&equal&(equal>>1)))&0x0101010101010101ULL;
                keep.template ReinterpretCast<uint64_t>().SetValue(0,packet*255ULL);
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        } else {
        // Packed predicate words share one mask throughout the lexicographic merge.
        SetVectorMask<uint16_t,MaskMode::NORMAL>(uint64_t(NATIVE_HALF_PACKET?1:4));
        Duplicate<uint16_t,false>(ones,uint16_t(0x0101),uint64_t(128),1,1,8);PipeBarrier<PIPE_V>();
        ShiftRight<uint16_t,false>(tmp,less,uint16_t(1),uint64_t(128),1,unary,false);PipeBarrier<PIPE_V>();
        And<uint16_t,false>(tmp,tmp,eq,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        Or<uint16_t,false>(keep,less,tmp,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        ShiftRight<uint16_t,false>(tmp,keep,uint16_t(1),uint64_t(128),1,unary,false);PipeBarrier<PIPE_V>();
        And<uint16_t,false>(tmp,tmp,eq,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        Or<uint16_t,false>(keep,less,tmp,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        And<uint16_t,false>(keep,keep,ones,uint64_t(128),1,binary);PipeBarrier<PIPE_V>();
        Muls<int16_t,false>(keep.template ReinterpretCast<int16_t>(),keep.template ReinterpretCast<int16_t>(),int16_t(255),uint64_t(128),1,unary);PipeBarrier<PIPE_V>();
        }
    }
