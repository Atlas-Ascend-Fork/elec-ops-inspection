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
    template<class StageSegmentCountType, class StageBatchType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessDataCopyPadStage(StageSegmentCountType& segmentCount, StageBatchType& batch) {
        if constexpr(CONTIG_SEGMENTS) {
            DataCopyPad(rawS_,sg_[batch*segmentCount*6],DataCopyExtParams{1,segmentCount*6U*uint32_t(sizeof(T)),0,0,0},
                        DataCopyPadExtParams<T>{false,0,0,T(0)});
            if constexpr(DEFER_SOC_INIT) { InitSocState(); InitializeVectorState(); }
            // i = 8*segment+coordinate; its packed source offset is 4*i-8*segment.
            if constexpr(BLOCK_INDICES) {
                if constexpr(UsesFixedSegmentsStage()) {
                    // Construct byte offsets directly with the original vector seed.
                    MakeIndices<4>(pointOffsets_,8);PipeBarrier<PIPE_V>();
                } else {
                MakeIndices(pointOffsets_,8);PipeBarrier<PIPE_V>();
                Muls(pointOffsets_,pointOffsets_,int32_t(4),8);PipeBarrier<PIPE_V>();
                }
                Adds(pointOffsets_[8],pointOffsets_,int32_t(24),8);PipeBarrier<PIPE_V>();
                if(segmentCount>2) {Adds(pointOffsets_[16],pointOffsets_,int32_t(48),16);PipeBarrier<PIPE_V>();}
                if(segmentCount>4) {Adds(pointOffsets_[32],pointOffsets_,int32_t(96),32);PipeBarrier<PIPE_V>();}
            } else {
                MakeIndices(pointOffsets_,64);PipeBarrier<PIPE_V>();
                ShiftRight(segmentOffsets_,pointOffsets_,int32_t(3),64);
                Muls(pointOffsets_,pointOffsets_,int32_t(4),64);PipeBarrier<PIPE_V>();
                Muls(segmentOffsets_,segmentOffsets_,int32_t(-8),64);PipeBarrier<PIPE_V>();
                Add(segmentOffsets_,segmentOffsets_,pointOffsets_,64);PipeBarrier<PIPE_V>();
            }
        } else {
            DataCopyPad(rawS_,sg_[batch*segmentCount*6],DataCopyExtParams{static_cast<uint16_t>(2*segmentCount),3U*sizeof(T),0,0,0},
                        DataCopyPadExtParams<T>{false,0,0,T(0)});
            if constexpr(DEFER_SOC_INIT) { InitSocState(); InitializeVectorState(); }
        }
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ constexpr bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::UsesFixedSegmentsStage() { return (FIXED_SEGMENTS==2 && ROWS==1 && NATIVE_ONE_ROW && sizeof(T)==4);
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessProductPairsStage(StageSegmentCountType& segmentCount) {
        {
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(count_));
            const LocalTensor<float> productPairs(TPosition::VECCALC,uint32_t(originalA_.GetPhyAddr()),512);
            const LocalTensor<float> pointDirectionPairs(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),192);
            Mul<float,false>(productPairs,pointDirectionPairs,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,56,16,0));
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Duplicate<float,false>(lengths_,1.0F,uint64_t(64),1,1,8);
            Duplicate<float,false>(projection_,0.0F,uint64_t(64),1,1,8);
            if constexpr(!SINGLE_SEGMENT)Duplicate<float,false>(scores_,__builtin_inff(),uint64_t(64),1,1,8);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(lengths_,W(5),3,SINGLE_SEGMENT?1:segmentCount,1,1,1);
            WholeReduceSum(projection_,W(4),3,SINGLE_SEGMENT?1:segmentCount,1,1,1);PipeBarrier<PIPE_V>();
            // Keep the original norm for the numerical certificate. Only the
            // divisor is clamped; small/degenerate norms retain the exact fallback.
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(count_));
            Maxs<float,false>(W(4),lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Div<float,false>(projection_,projection_,W(4),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Maxs<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Mins<float,false>(projection_,projection_,1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(count_));
            if constexpr(!SINGLE_SEGMENT)CompareScalar(masks_,W(6),1.0F,CMPMODE::LT,64);
            Muls<float,false>(W(6),W(6),-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            const LocalTensor<float> residualPairs(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),128);
            const LocalTensor<float> directionPairs(TPosition::VECCALC,uint32_t(W(0).GetPhyAddr()),128);
            MulAddDst<float,float,false>(residualPairs,W(6),directionPairs,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,0,8));PipeBarrier<PIPE_V>();
            if constexpr(!SINGLE_SEGMENT)Sub<float,false>(W(4),p_,c_,uint64_t(64),1,same);
            PipeBarrier<PIPE_V>();
            if constexpr(!SINGLE_SEGMENT) {
                SetCmpMask(masks_);
                Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(2),W(2),W(4),1,same);PipeBarrier<PIPE_V>();
            }
            Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,W(2),3,SINGLE_SEGMENT?1:segmentCount,1,1,1);PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType, class StageBatchType, class StagePointPositionType, class StageValidRowsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessReinterpretCastStage(StageSegmentCountType& segmentCount, StageBatchType& batch, StagePointPositionType& pointPosition, StageValidRowsType& validRows) {
        if constexpr(NATIVE_HALF_PACKET) {
            auto rangeFlags=zeroGeometry_.template ReinterpretCast<uint32_t>();
            const uint32_t accepted=rangeFlags.GetValue(72);
            if((accepted&0x0fff0007U)==0x0fff0007U) {
                // Bounded Half coordinates keep endpoint dot and norm errors below the guard.
                const BinaryRepeatParams same{1,1,1,8,8,8};
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                Sub<float,false>(W(0),c_,a_,uint64_t(64),1,same);
                Sub<float,false>(W(2),p_,a_,uint64_t(64),1,same);
                Sub<float,false>(W(3),p_,c_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                const LocalTensor<float> endpointPairs(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),128);
                const LocalTensor<float> products(TPosition::VECCALC,uint32_t(zeroGeometry_[128].GetPhyAddr()),128);
                Mul<float,false>(products,endpointPairs,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,8,8,0));PipeBarrier<PIPE_V>();
                WholeReduceSum(zeroGeometry_[256],zeroGeometry_[128],3,2,1,1,1);
                WholeReduceSum(zeroGeometry_[320],zeroGeometry_[192],3,2,1,1,1);PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(2));
                CompareScalar<float,uint8_t,false>(masks_,zeroGeometry_[256],-0.001953125F,CMPMODE::LT,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
                CompareScalar<float,uint8_t,false>(masks_[64],zeroGeometry_[320],0.001953125F,CMPMODE::GT,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                const uint32_t atStart=masks_.template ReinterpretCast<uint32_t>().GetValue(0);
                const uint32_t atEnd=masks_[64].template ReinterpretCast<uint32_t>().GetValue(0);
                if(((atStart|atEnd)&3U)==3U) {
                    // Only reduced predicates choose addresses; every residual is vector-computed.
                    const auto residual0=(atStart&1U)?W(2):W(3);
                    const auto residual1=(atStart&2U)?W(2)[8]:W(3)[8];
                    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                    Duplicate<float,false>(scores_,__builtin_inff(),uint64_t(64),1,1,8);
                    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
                    Mul<float,false>(zeroGeometry_[128],residual0,residual0,uint64_t(64),1,same);
                    Mul<float,false>(zeroGeometry_[136],residual1,residual1,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                    WholeReduceSum(scores_,zeroGeometry_[128],3,2,1,1,1);PipeBarrier<PIPE_V>();
                    SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
                    ReduceTiny();
                    // Bounded endpoint arithmetic still needs an index certificate.
                    SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    const auto scoreBits=scores_.template ReinterpretCast<uint32_t>();
                    const uint32_t score0=scoreBits.GetValue(0),score1=scoreBits.GetValue(1);
                    // Nonnegative finite endpoint norms differ by far more than
                    // the bounded binary32 accumulation error at 128 ULPs.
                    const uint32_t separation=score0>score1?score0-score1:score1-score0;
                    if(separation<=128U && !TryExactDyadicGeometry(validRows,segmentCount)) {
                        ExactFallback(pointPosition,batch,validRows,segmentCount);return true;
                    }
                    WriteTinyResult(pointPosition,validRows);
                    // No subsequent UB reuse or GM overwrite follows this terminal write.
                    return true;
                }
            }
        }
        return false;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessMulStage(StageSegmentCountType& segmentCount) {
        {
        Mul(W(4),W(0),W(0),count_); Mul(W(5),W(2),W(0),count_);
            Duplicate(lengths_,1.0F,64); Duplicate(projection_,0.0F,64);
            if constexpr(!NATIVE_FOUR_ROW_SINGLE)Duplicate(scores_,__builtin_inff(),64);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(lengths_,W(4),3,ROWS==1?(SINGLE_SEGMENT?1:segmentCount):pairs_,1,1,1);
            WholeReduceSum(projection_,W(5),3,ROWS==1?(SINGLE_SEGMENT?1:segmentCount):pairs_,1,1,1);
            PipeBarrier<PIPE_V>();
            CompareScalar(masks_,lengths_,0.0F,CMPMODE::GT,64);
            PipeBarrier<PIPE_V>();
            Select(lengths_,masks_,lengths_,1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
            PipeBarrier<PIPE_V>();
            Div(projection_,projection_,lengths_,64);
            PipeBarrier<PIPE_V>();
            Maxs(projection_,projection_,0.0F,64);
            PipeBarrier<PIPE_V>();
            Mins(projection_,projection_,1.0F,64);
            PipeBarrier<PIPE_V>();
            Brcb(W(6),projection_,(pairs_+7)/8,BrcbRepeatParams(1,8));
            PipeBarrier<PIPE_V>();
            if(!SINGLE_SEGMENT && segmentCount>1) CompareScalar(masks_,W(6),1.0F,CMPMODE::LT,64);
            Muls(W(6),W(6),-1.0F,count_);
            PipeBarrier<PIPE_V>();
            if constexpr(NATIVE_FOUR_ROW_SINGLE) {
                const LocalTensor<float> residualPairs(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),64);
                const LocalTensor<float> directionPairs(TPosition::VECCALC,uint32_t(W(0).GetPhyAddr()),64);
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(32));
                MulAddDst<float,float,false>(residualPairs,W(6),directionPairs,uint64_t(64),1,BinaryRepeatParams(1,1,1,4,0,4));
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            } else {
                MulAddDst(W(2),W(6),W(0),count_);
            }
            PipeBarrier<PIPE_V>();
            if(!SINGLE_SEGMENT && segmentCount>1) Sub(W(4),p_,c_,count_);
            PipeBarrier<PIPE_V>();
            if(!SINGLE_SEGMENT && segmentCount>1) Select(W(2),masks_,W(2),W(4),SELMODE::VSEL_CMPMASK_SPR,count_);
            PipeBarrier<PIPE_V>();
            Mul(W(2),W(2),W(2),count_);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,W(2),3,ROWS==1?(SINGLE_SEGMENT?1:segmentCount):pairs_,1,1,1);
            PipeBarrier<PIPE_V>();
            if constexpr(ROWS!=1 && !SINGLE_SEGMENT)Select(scores_,masks_[384],scores_,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);
            PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::WriteTinyResult(uint32_t pointPosition,uint32_t validRows) {
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(ROWS));
        Sqrt<float,false>(out_,out_,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        if constexpr(sizeof(T)==2) {
            PipeBarrier<PIPE_V>();
            Cast<T,float,false>(outT_,out_,RoundMode::CAST_RINT,uint64_t(64),1,UnaryRepeatParams(1,1,4,8));
        } else outT_=out_;
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(dg_[pointPosition],outT_,DataCopyExtParams{1,validRows*uint32_t(sizeof(T)),0,0,0});
        DataCopyPad(ig_[pointPosition],indices_,DataCopyExtParams{1,validRows*4,0,0,0});
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::InitializeVectorState() {
        // Count-form APIs restore normal mode; native masked operations preserve it.
        SetMaskNorm();
        if constexpr(!SINGLE_SEGMENT && !NATIVE_ONE_ROW) {
            const uint32_t segmentCount=FIXED_SEGMENTS?FIXED_SEGMENTS:t_.segments;
            const uint32_t bits=(SINGLE_SEGMENT?1U:((1U<<segmentCount)-1U))*(ROWS==1?0x01010101U:0x55555555U);
            Duplicate(masks_[384].template ReinterpretCast<uint32_t>(),bits,2);
        }
        if constexpr(SINGLE_SEGMENT && sizeof(T)==2) {
            Duplicate(masks_[384].template ReinterpretCast<uint32_t>(),uint32_t((1U<<ROWS)-1U),2);
        }
        if constexpr(ROWS!=1 && !SINGLE_SEGMENT)Duplicate(segmentOffsets_,int32_t(1),64);
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::DyadicLatticeFlags(const LocalTensor<float>& left,
            const LocalTensor<float>& right,uint32_t coordinates,bool packedInputs) {
        auto scaled=zeroExact_[1280],rounded=zeroExact_[1472],restored=zeroExact_[1664];
        auto flags=zeroExact_[880].template ReinterpretCast<uint8_t>();
        if(packedInputs && FIXED_SEGMENTS==2) {
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Muls<float,false>(scaled,p_,16.0F,uint64_t(64),3,UnaryRepeatParams(1,1,8,8));
        } else {
            Muls(scaled,p_,16.0F,coordinates);
            Muls(scaled[64],left,16.0F,coordinates);
            Muls(scaled[128],right,16.0F,coordinates);
        }
        PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        const UnaryRepeatParams blocks{1,1,8,8};
        // Round in FP32: scaled coordinates on the dyadic lattice are already integers.
        // Large FP32 values are also grid-aligned; downstream finite residual bounds remain required.
        Cast<float,float,false>(restored,scaled,RoundMode::CAST_RINT,uint64_t(64),3,blocks);PipeBarrier<PIPE_V>();
        const BinaryRepeatParams same{1,1,1,8,8,8};
        Compare<float,uint8_t,false>(flags,scaled,restored,CMPMODE::EQ,uint64_t(64),1,same);
        Compare<float,uint8_t,false>(flags[64],scaled[64],restored[64],CMPMODE::EQ,uint64_t(64),1,same);
        Compare<float,uint8_t,false>(flags[128],scaled[128],restored[128],CMPMODE::EQ,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::TryAxisDyadicGeometry(uint32_t validRows,uint32_t segments) {
        // Axis alignment was established from original endpoints. On the 2^-4
        // lattice, bounded residuals and their three-term squared norm are exact.
        const uint32_t pairs=ROWS==1?segments:validRows*PAIR_STRIDE;
        const uint32_t coordinates=pairs*8U;
        const uint64_t prefix=coordinates==64?~uint64_t(0):((uint64_t(1)<<coordinates)-1U);
        const uint64_t live=prefix&0x0707070707070707ULL;
        auto lower=zeroExact_[704],upper=zeroExact_[768],residual=zeroExact_[192];
        auto magnitude=zeroExact_[320];
        auto flags=zeroExact_[832].template ReinterpretCast<uint8_t>();
        DyadicLatticeFlags(a_,c_,coordinates,count_==64 && FIXED_SEGMENTS==2);
        const BinaryRepeatParams same{1,1,1,8,8,8};
        const UnaryRepeatParams unary{1,1,8,8};
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        Min<float,false>(lower,a_,c_,uint64_t(64),1,same);Max<float,false>(upper,a_,c_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Max<float,false>(lower,p_,lower,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Min<float,false>(lower,lower,upper,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Sub<float,false>(residual,p_,lower,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Abs<float,false>(magnitude,residual,uint64_t(64),1,unary);
        Mul<float,false>(residual,residual,residual,uint64_t(64),1,same);
        // Fixed reductions read exactly the written segment scores; inactive rows are independent.
        if constexpr(FIXED_SEGMENTS==0) Duplicate(scores_,__builtin_inff(),64);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        CompareScalar<float,uint8_t,false>(flags,magnitude,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        WholeReduceSum(scores_,residual,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        ReduceTiny();
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        auto words=flags.template ReinterpretCast<uint64_t>();
        const uint64_t admitted=words.GetValue(24)&words.GetValue(32)&words.GetValue(40)&words.GetValue(0);
        return (admitted&live)==live;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::TryAxisSingle(uint32_t validRows) {
        // A single axis-aligned segment has no argmin ambiguity. AABB clipping
        // avoids projection cancellation, including zero-length/zero-distance.
        // Finite residuals bounded by 2^30 keep squares and sums far from overflow;
        // subnormal residual loss is far below the contract's absolute tolerance.
        const uint32_t coordinates=validRows*8U;
        auto clipped=zeroExact_[704],upper=zeroExact_[768],residual=zeroExact_[192];
        auto flags=zeroExact_[832].template ReinterpretCast<uint8_t>();
        const BinaryRepeatParams same{1,1,1,8,8,8};
        const UnaryRepeatParams unary{1,1,8,8};
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        Min<float,false>(clipped,a_,c_,uint64_t(64),1,same);Max<float,false>(upper,a_,c_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Max<float,false>(clipped,p_,clipped,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Min<float,false>(clipped,clipped,upper,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Sub<float,false>(residual,p_,clipped,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        Abs<float,false>(upper,residual,uint64_t(64),1,unary);
        Mul<float,false>(residual,residual,residual,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,false>(flags,upper,1073741824.0F,CMPMODE::LE,uint64_t(64),1,unary);
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
        WholeReduceSum<float,false>(out_,residual,3,validRows,1,1,1);
        SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(8));
        Duplicate<int32_t,false>(indices_,int32_t(0),uint64_t(64),1,1,8);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint64_t live=((uint64_t(1)<<coordinates)-1)&0x0707070707070707ULL;
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        return (flags.template ReinterpretCast<uint64_t>().GetValue(0)&live)==live;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<bool INPUT_READY>
    __aicore__ __attribute__((noinline)) inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ExactDyadicPrepared(uint32_t validRows,uint32_t segments) {
        // Reconstruct compile-time UB views locally instead of spilling the caller object.
        // Input DMA and its completion event remain owned by the caller.
        KernelTreeTiny local;
        const TreeLineMinDistanceTilingData resource{0,ROWS,segments,8,ROWS,0,0,3};
        local.Init(nullptr,nullptr,nullptr,nullptr,resource);
        if constexpr(NATIVE_SINGLE_POINT) {
            local.p_=local.fp_;local.a_=local.fs_;local.c_=local.fs_[slot_];
        }
        return local.ComputeExactDyadicGeometry(validRows,segments,INPUT_READY);
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::TryExactDyadicGeometry(uint32_t validRows,uint32_t segments,bool inputReady) {
        const bool accepted=inputReady?ExactDyadicPrepared<true>(validRows,segments):
                                         ExactDyadicPrepared<false>(validRows,segments);
        if constexpr(SINGLE_SEGMENT && (ROWS==1 || PAIR_STRIDE==1))out_=scores_;
        else if constexpr(ROWS==1)out_=records_;
        return accepted;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ComputeExactDyadicGeometry(uint32_t validRows,uint32_t segments,bool inputReady) {
        // Matched integer reconstruction certifies a 2^-4 coordinate lattice.
        // A difference bounded by 64 is exact: small operands fit binary32,
        // and larger nearby same-sign operands satisfy Sterbenz's lemma.
        // Relative/direction bounds make dots and squared norms exact (22 bits).
        // Endpoint projections and axis-aligned segments therefore admit an
        // exact argmin, including zero length, zero distance and true ties.
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        const uint32_t pairs=ROWS==1?segments:validRows*PAIR_STRIDE;
        const uint32_t coordinates=pairs*8;
        const uint64_t prefix=coordinates==64?~uint64_t(0):((uint64_t(1)<<coordinates)-1);
        const uint64_t liveCoordinates=prefix&((ROWS>1 && PAIR_STRIDE==2 && segments==1)?
            0x0007000700070007ULL:0x0707070707070707ULL);
        const uint32_t livePairs=((1U<<pairs)-1U)&
            ((ROWS>1 && PAIR_STRIDE==2 && segments==1)?0x55U:0xffU);
        TreeDyadicScratch arena(zeroExact_);
        ComputeExactDyadicGeometryGatherStage(validRows, inputReady, coordinates, arena.left, arena.right);
        DyadicLatticeFlags(arena.left,arena.right,coordinates,inputReady && count_==64);
        Sub(arena.delta,arena.right,arena.left,coordinates);Sub(arena.relA,p_,arena.left,coordinates);Sub(arena.relC,p_,arena.right,coordinates);PipeBarrier<PIPE_V>();
        if constexpr(FIXED_SEGMENTS==2) {
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
            Abs<float,false>(arena.temp,arena.delta,uint64_t(64),3,UnaryRepeatParams(1,1,8,8));
        } else {
            Abs(arena.temp,arena.delta,coordinates);Abs(arena.rounded,arena.relA,coordinates);Abs(arena.product,arena.relC,coordinates);
        }
        PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(coordinates));
        CompareScalar<float,uint8_t,false>(arena.flags[96],arena.temp,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        CompareScalar<float,uint8_t,false>(arena.flags[128],arena.rounded,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        CompareScalar<float,uint8_t,false>(arena.flags[160],arena.product,64.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        Compare<float,uint8_t,false>(arena.flags,arena.left,arena.right,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const auto words=arena.flags.template ReinterpretCast<uint64_t>();
        const uint64_t admitted=words.GetValue(24)&words.GetValue(32)&words.GetValue(40);
        const uint64_t bounded=words.GetValue(12)&words.GetValue(16)&words.GetValue(20);
        if(((admitted&bounded)&liveCoordinates)!=liveCoordinates)return false;
        const uint64_t fixed=words.GetValue(0);
        const uint64_t axis=((fixed&(fixed>>1))|(fixed&(fixed>>2))|((fixed>>1)&(fixed>>2)))&0x0101010101010101ULL;
        return ComputeExactDyadicGeometryComputeExactDyadicGeometryMinStageStage(inputReady, pairs, coordinates, livePairs, arena.left, arena.right, arena.delta, arena.relA, arena.relC, arena.temp, arena.product, arena.norm, arena.dot, arena.endpoint, arena.box, arena.upper, arena.flags, words, axis);
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageInputReadyType, class StagePairsType, class StageCoordinatesType, class StageLivePairsType, class StageLeftType, class StageRightType, class StageDeltaType, class StageRelAType, class StageRelCType, class StageTempType, class StageProductType, class StageNormType, class StageDotType, class StageEndpointType, class StageBoxType, class StageUpperType, class StageFlagsType, class StageWordsType, class StageAxisType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ComputeExactDyadicGeometryComputeExactDyadicGeometryMinStageStage(StageInputReadyType& inputReady, StagePairsType& pairs, StageCoordinatesType& coordinates, StageLivePairsType& livePairs, StageLeftType& left, StageRightType& right, StageDeltaType& delta, StageRelAType& relA, StageRelCType& relC, StageTempType& temp, StageProductType& product, StageNormType& norm, StageDotType& dot, StageEndpointType& endpoint, StageBoxType& box, StageUpperType& upper, StageFlagsType& flags, StageWordsType& words, StageAxisType& axis) {
        const uint32_t axisPairs=inputReady?livePairs:uint32_t((axis*0x0102040810204080ULL)>>56);
        bool computeExactDyadicGeometryMinStageResult{};
        if (ComputeExactDyadicGeometryMinStage(pairs, coordinates, livePairs, left, right, product, box, flags, axisPairs, computeExactDyadicGeometryMinStageResult)) return computeExactDyadicGeometryMinStageResult;
        Mul(product,delta,delta,coordinates);Mul(temp,relA,delta,coordinates);PipeBarrier<PIPE_V>();
        WholeReduceSum(norm,product,3,pairs,1,1,1);
        WholeReduceSum(dot,temp,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
        CompareScalar<float,uint8_t,false>(flags[32],dot,0.0F,CMPMODE::LE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
        Compare<float,uint8_t,false>(flags[64],dot,norm,CMPMODE::GE,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint32_t endpoints=uint32_t(words.GetValue(4)|words.GetValue(8));
        bool computeExactDyadicGeometryReinterpretCastStageResult{};
        if (ComputeExactDyadicGeometryReinterpretCastStage(pairs, coordinates, livePairs, left, right, delta, relA, relC, temp, product, norm, dot, endpoint, upper, flags, words, axisPairs, endpoints, computeExactDyadicGeometryReinterpretCastStageResult)) return computeExactDyadicGeometryReinterpretCastStageResult;
        CommonDyadicGeometry54(product, relA, coordinates, temp, relC, endpoint, pairs, upper, box, left, right);
        Max(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
        Min(box,box,product,coordinates);Min(endpoint,endpoint,upper,pairs);PipeBarrier<PIPE_V>();
        Sub(box,p_,box,coordinates);PipeBarrier<PIPE_V>();
        Mul(box,box,box,coordinates);PipeBarrier<PIPE_V>();
        WholeReduceSum(upper,box,3,pairs,1,1,1);PipeBarrier<PIPE_V>();
        Duplicate(flags.template ReinterpretCast<uint32_t>(),axisPairs,2);PipeBarrier<PIPE_V>();
        SetCmpMask(flags);
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
        Select<float,SELMODE::VSEL_CMPMASK_SPR>(endpoint,upper,endpoint,1,BinaryRepeatParams(1,1,1,8,8,8));PipeBarrier<PIPE_V>();
        Duplicate(flags.template ReinterpretCast<uint32_t>(),livePairs,2);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Select(scores_,flags,endpoint,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        ReduceTiny();
        return true;

    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StagePairsType, class StageCoordinatesType, class StageLivePairsType, class StageLeftType, class StageRightType, class StageDeltaType, class StageRelAType, class StageRelCType, class StageTempType, class StageProductType, class StageNormType, class StageDotType, class StageEndpointType, class StageUpperType, class StageFlagsType, class StageWordsType, class StageAxisPairsType, class StageEndpointsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ComputeExactDyadicGeometryReinterpretCastStage(StagePairsType& pairs, StageCoordinatesType& coordinates, StageLivePairsType& livePairs, StageLeftType& left, StageRightType& right, StageDeltaType& delta, StageRelAType& relA, StageRelCType& relC, StageTempType& temp, StageProductType& product, StageNormType& norm, StageDotType& dot, StageEndpointType& endpoint, StageUpperType& upper, StageFlagsType& flags, StageWordsType& words, StageAxisPairsType& axisPairs, StageEndpointsType& endpoints, bool& stageResult) {
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
            SetVectorMask<int32_t,MaskMode::NORMAL>(uint64_t(0),uint64_t(0x0202020202020202ULL));
            CommonDyadicGeometry51(rotate1, rotate2, coordinates, u, relA, v, delta, cross, crossNorm, pairs, product, endpoint);
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs));
            CompareScalar<float,uint8_t,false>(flags,crossNorm,0.0F,CMPMODE::EQ,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            CompareScalar<float,uint8_t,false>(flags[32],dot,0.0F,CMPMODE::GE,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            Compare<float,uint8_t,false>(flags[64],dot,norm,CMPMODE::LE,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            CompareScalar<float,uint8_t,false>(flags[96],norm,0.0F,CMPMODE::GT,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            CompareScalar<float,uint8_t,false>(flags[128],endpoint,0.0F,CMPMODE::EQ,uint64_t(64),1,UnaryRepeatParams(1,1,8,8));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint32_t zeroPairs=uint32_t(words.GetValue(0)&words.GetValue(4)&words.GetValue(8)&
                (words.GetValue(12)|words.GetValue(16)))&livePairs;
            const uint32_t firstPairs=ROWS==1?1U:(PAIR_STRIDE==2?(livePairs&0x55U):livePairs);
            bool computeExactDyadicGeometryDuplicateStageResult{};
        if (ComputeExactDyadicGeometryDuplicateStage(pairs, coordinates, livePairs, left, right, relC, temp, norm, endpoint, upper, flags, words, endpoints, u, v, crossNorm, zeroPairs, firstPairs, computeExactDyadicGeometryDuplicateStageResult)) { stageResult = computeExactDyadicGeometryDuplicateStageResult; return true; }}
        return false;
    }
