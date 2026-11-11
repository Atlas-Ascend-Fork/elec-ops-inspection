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
    __aicore__ inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::Init(GM_ADDR p, GM_ADDR s, GM_ADDR d, GM_ADDR i,
                               const TreeLineMinDistanceTilingData& t) {
        t_ = t; rows_ = ROWS; pairs_ = ROWS==1?8:ROWS*PAIR_STRIDE; count_ = ROWS==4 && PAIR_STRIDE==1?32:64;
        pg_.SetGlobalBuffer((__gm__ T*)p); sg_.SetGlobalBuffer((__gm__ T*)s);
        dg_.SetGlobalBuffer((__gm__ T*)d); ig_.SetGlobalBuffer((__gm__ int32_t*)i);
        uint32_t address = 0;
        rawP_ = Alloc<T>(address, 256 / sizeof(T));
        rawS_ = Alloc<T>(address, 512 / sizeof(T));
        if constexpr (sizeof(T) == 2) {
            fp_ = Alloc<float>(address, 128); fs_ = Alloc<float>(address, 256);
        } else { fp_ = rawP_.template ReinterpretCast<float>(); fs_ = rawS_.template ReinterpretCast<float>(); }
        p_ = Alloc<float>(address, count_); a_ = Alloc<float>(address, count_); c_ = Alloc<float>(address, count_);
        work_ = Alloc<float>(address, count_ * 8);
        if constexpr(MAPPED_TINY_STARTS) originalA_=a_;
        pointOffsets_ = Alloc<int32_t>(address, count_);
        segmentOffsets_ = Alloc<int32_t>(address, count_);
        lengths_ = Alloc<float>(address, 64); projection_ = Alloc<float>(address, 64);
        scores_ = Alloc<float>(address, 64); records_ = Alloc<float>(address, 64);
        out_ = Alloc<float>(address, 64); outT_ = Alloc<T>(address, 64);
        indices_ = Alloc<int32_t>(address, 64);
        masks_ = Alloc<uint8_t>(address, 512);
        zeroGeometry_=Alloc<float>(address,9*64);
        zeroExact_=Alloc<float>(address,45*64);
        if constexpr(!DEFER_SOC_INIT) InitializeVectorState();
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::Process(uint32_t logicalOwner) {
        const uint32_t pointCount=STATIC_ONE_POINT || (ROWS==1 && SINGLE_SEGMENT)?1U:t_.points;
        const uint32_t segmentCount=SINGLE_SEGMENT?1U:FIXED_SEGMENTS?FIXED_SEGMENTS:t_.segments;
        uint32_t batch,pointPosition,validRows;
        ProcessStateStage(logicalOwner, pointCount, batch, pointPosition, validRows);
        // The native single-segment load defines both padded endpoint
        // records; no preceding vector write is needed for this domain.
        if constexpr(ROWS!=1 && !COMPLETE_ENDPOINT_LOAD)Duplicate(rawS_,T(0),CONTIG_SEGMENTS?64:512/sizeof(T));
        if constexpr(ROWS!=1 && !COMPLETE_ENDPOINT_LOAD) {SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);}
        DataCopyPad(rawP_,pg_[pointPosition*3],DataCopyExtParams{static_cast<uint16_t>(validRows),3U*sizeof(T),0,0,0},
                    DataCopyPadExtParams<T>{false,0,0,T(0)});
        ProcessDataCopyPadStage(segmentCount, batch);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        if constexpr(NATIVE_HALF_PACKET) {
            auto absoluteInputs=zeroGeometry_.template ReinterpretCast<T>();
            SetVectorMask<T,MaskMode::NORMAL>(uint64_t(16));
            Abs<T,false>(absoluteInputs,rawP_,uint64_t(128),2,UnaryRepeatParams(1,1,1,8));
        }
        ProcessConvertedInputsStage(segmentCount, validRows);
        if constexpr(NATIVE_HALF_PACKET) {
            auto absoluteInputs=zeroGeometry_.template ReinterpretCast<T>();
            auto belowMaximum=zeroGeometry_[72].template ReinterpretCast<uint8_t>();
            SetVectorMask<T,MaskMode::NORMAL>(uint64_t(32));
            CompareScalar<T,uint8_t,false>(belowMaximum,absoluteInputs,T(2.0F),CMPMODE::LT,uint64_t(128),1,UnaryRepeatParams(1,1,8,8));
        }
        ProcessCopyStage(segmentCount);
        PipeBarrier<PIPE_V>();
        if (ProcessReinterpretCastStage2(segmentCount, pointPosition, validRows)) return;
        if constexpr(NATIVE_HALF_PACKET) {
            // Coordinate range flags reuse the original input-copy barrier.
        }
        if (ProcessReinterpretCastStage(segmentCount, batch, pointPosition, validRows)) return;
        if (ProcessReinterpretCastStage3(segmentCount, pointPosition, validRows)) return;
        ProcessProcessReinterpretCastStage4Stage(segmentCount, batch, pointPosition, validRows);
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType, class StageBatchType, class StagePointPositionType, class StageValidRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessProcessReinterpretCastStage4Stage(StageSegmentCountType& segmentCount, StageBatchType& batch, StagePointPositionType& pointPosition, StageValidRowsType& validRows) {
        if (ProcessReinterpretCastStage4(segmentCount, pointPosition, validRows)) return;
        // The bounded certificate below permits ordinary FP32 differences;
        // difficult exponents, cancellation and ties use exact vector packets.
        if constexpr(THIN_SINGLE_FP32)count_=8;
        else if constexpr(ROWS==1 && sizeof(T)==2 && !NATIVE_ONE_ROW)count_=SINGLE_SEGMENT?8:(segmentCount<=2?16:segmentCount<=4?32:64);
        else if constexpr(NATIVE_HALF_PACKET)count_=16;
        else if constexpr(UsesRowsStage())count_=64;
        else if constexpr(ROWS>1 && !SINGLE_SEGMENT)count_=validRows*PAIR_STRIDE*8;
        Sub(W(0),c_,a_,count_);
        Sub(W(2),p_,a_,count_);
        PipeBarrier<PIPE_V>();
        ProcessProcessProductsStageStage(segmentCount);
        ReduceTiny();
        if(!CertifiedTiny(validRows,segmentCount) && !TryExactDyadicGeometry(validRows,segmentCount)) {
            ExactFallback(pointPosition,batch,validRows,segmentCount);return;
        }
        WriteTinyResult(pointPosition,validRows);
        // No subsequent UB reuse or GM overwrite follows this terminal write.

    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ constexpr bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::UsesRowsStage() { return (ROWS==4 && PAIR_STRIDE==2 && FIXED_SEGMENTS==2 && sizeof(T)==4);
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType, class StagePointPositionType, class StageValidRowsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessReinterpretCastStage4(StageSegmentCountType& segmentCount, StagePointPositionType& pointPosition, StageValidRowsType& validRows) {
        if constexpr(UsesNativeOneRowStage()) {
            // Certify bounded differences and residuals after vector geometry.
            auto bounds=zeroExact_.template ReinterpretCast<uint8_t>();
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};

                SharedResourceBlock6(same, bounds, unary);
                CompareScalar<float,uint8_t,false>(bounds[320],lengths_,256.0F,CMPMODE::LE,uint64_t(64),1,unary);
                Maxs<float,false>(lengths_,lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Div<float,false>(projection_,zeroGeometry_[256],lengths_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Muls<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                MulAddDst<float,float,false>(zeroGeometry_[256],projection_,lengths_,uint64_t(64),1,same);
                Muls<float,false>(zeroGeometry_[384],lengths_,0.0000002384185791015625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                SharedResourceBlock8(unary, bounds, same);
                CompareScalar<float,uint8_t,false>(bounds[384],scores_,1024.0F,CMPMODE::LE,uint64_t(64),1,unary);
                WholeReduceMin<float,false>(records_,scores_,2,1,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);
                WholeReduceMax<float,false>(zeroGeometry_[384],scores_,2,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);if (ProcessGatherMaskStage3(segmentCount, pointPosition, validRows, bounds, same, unary)) return true;}
        return false;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ constexpr bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::UsesNativeOneRowStage() { return (NATIVE_ONE_ROW && !STATIC_ONE_POINT && FIXED_SEGMENTS==2 && sizeof(T)==4);
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType, class StagePointPositionType, class StageValidRowsType, class StageBoundsType, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessGatherMaskStage3(StageSegmentCountType& segmentCount, StagePointPositionType& pointPosition, StageValidRowsType& validRows, StageBoundsType& bounds, StageSameType& same, StageUnaryType& unary) {
        PipeBarrier<PIPE_V>();
                out_=records_;
                GatherMask(indices_.template ReinterpretCast<float>(),records_,uint8_t(2),GatherMaskParams(1,1,8,0));
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(1));
                Sub<float,false>(zeroGeometry_[384],zeroGeometry_[384],out_,uint64_t(64),1,same);
                CompareScalar<float,uint8_t,false>(bounds[192],out_,0.25F,CMPMODE::GE,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                CompareScalar<float,uint8_t,false>(bounds[256],zeroGeometry_[384],0.03125F,CMPMODE::GT,uint64_t(64),1,unary);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                auto flags=bounds.template ReinterpretCast<uint32_t>();
                if (ProcessGetValueStage2(segmentCount, pointPosition, validRows, unary, flags)) return true;
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

        return false;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType, class StagePointPositionType, class StageValidRowsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessReinterpretCastStage3(StageSegmentCountType& segmentCount, StagePointPositionType& pointPosition, StageValidRowsType& validRows) {
        if constexpr(UsesNativeOneRowStage2()) {
            // Certify bounded differences and residuals after vector geometry.
            auto bounds=zeroExact_.template ReinterpretCast<uint8_t>();
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};

                SharedResourceBlock6(same, bounds, unary);
                CompareScalar<float,uint8_t,false>(bounds[320],lengths_,64.0F,CMPMODE::LE,uint64_t(64),1,unary);
                Maxs<float,false>(lengths_,lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                Div<float,false>(projection_,zeroGeometry_[256],lengths_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
                Muls<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                MulAddDst<float,float,false>(zeroGeometry_[256],projection_,lengths_,uint64_t(64),1,same);
                Muls<float,false>(zeroGeometry_[384],lengths_,0.00000095367431640625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                SharedResourceBlock8(unary, bounds, same);
                CompareScalar<float,uint8_t,false>(bounds[384],scores_,64.0F,CMPMODE::LE,uint64_t(64),1,unary);
                WholeReduceMin<float,false>(records_,scores_,2,1,1,1,1,ReduceOrder::ORDER_VALUE_INDEX);
                WholeReduceMax<float,false>(zeroGeometry_[384],scores_,2,1,1,1,1,ReduceOrder::ORDER_ONLY_VALUE);if (ProcessGatherMaskStage2(segmentCount, pointPosition, validRows, bounds, same, unary)) return true;}
        return false;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ constexpr bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::UsesNativeOneRowStage2() { return (NATIVE_ONE_ROW && STATIC_ONE_POINT && FIXED_SEGMENTS==2 && sizeof(T)==4);
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType, class StagePointPositionType, class StageValidRowsType, class StageBoundsType, class StageSameType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessGatherMaskStage2(StageSegmentCountType& segmentCount, StagePointPositionType& pointPosition, StageValidRowsType& validRows, StageBoundsType& bounds, StageSameType& same, StageUnaryType& unary) {
        PipeBarrier<PIPE_V>();
                out_=records_;
                GatherMask(indices_.template ReinterpretCast<float>(),records_,uint8_t(2),GatherMaskParams(1,1,8,0));
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(1));
                Sub<float,false>(zeroGeometry_[384],zeroGeometry_[384],out_,uint64_t(64),1,same);
                CompareScalar<float,uint8_t,false>(bounds[192],out_,0.015625F,CMPMODE::GE,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
                CompareScalar<float,uint8_t,false>(bounds[256],zeroGeometry_[384],0.00390625F,CMPMODE::GT,uint64_t(64),1,unary);
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                auto flags=bounds.template ReinterpretCast<uint32_t>();
                if (ProcessGetValueStage2(segmentCount, pointPosition, validRows, unary, flags)) return true;
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));

        return false;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType, class StagePointPositionType, class StageValidRowsType, class StageUnaryType, class StageFlagsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessGetValueStage2(StageSegmentCountType& segmentCount, StagePointPositionType& pointPosition, StageValidRowsType& validRows, StageUnaryType& unary, StageFlagsType& flags) {
        if((((flags.GetValue(16)&flags.GetValue(32)&flags.GetValue(80)&flags.GetValue(96))&3U)==3U &&
                   ((flags.GetValue(48)&flags.GetValue(64))&1U)==1U) ||
                   TryExactDyadicGeometry(validRows,segmentCount)) {
                    Sqrt<float,false>(out_,out_,uint64_t(64),1,unary);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopyPad(dg_[pointPosition],out_,DataCopyExtParams{1,4,0,0,0});
                    DataCopyPad(ig_[pointPosition],indices_,DataCopyExtParams{1,4,0,0,0});
                    // No subsequent UB reuse or GM overwrite follows this terminal write.
                    return true;
                }
        return false;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageLogicalOwnerType, class StagePointCountType, class StageBatchType, class StagePointPositionType, class StageValidRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessStateStage(StageLogicalOwnerType& logicalOwner, StagePointCountType& pointCount, StageBatchType& batch, StagePointPositionType& pointPosition, StageValidRowsType& validRows) {
        if constexpr(ROWS==1) {
            // One-row tiles own consecutive flattened point/output positions.
            batch=pointCount==1?logicalOwner:logicalOwner/pointCount;
            pointPosition=logicalOwner;
            validRows=1;
        } else {
            const uint32_t tiles=(pointCount+rows_-1)/rows_;
            batch=logicalOwner/tiles;
            const uint32_t start=logicalOwner%tiles*rows_;
            pointPosition=batch*pointCount+start;
            validRows=pointCount-start<rows_?pointCount-start:rows_;
        }
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType, class StageValidRowsType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessConvertedInputsStage(StageSegmentCountType& segmentCount, StageValidRowsType& validRows) {
        if constexpr (sizeof(T)==2) {
            if constexpr(NATIVE_HALF_PACKET) {
                // Repeat strides address the two independent padded DMA records.
                const LocalTensor<float> convertedInputs(TPosition::VECCALC,uint32_t(fp_.GetPhyAddr()),256);
                const LocalTensor<T> rawInputs(TPosition::VECCALC,uint32_t(rawP_.GetPhyAddr()),256);
                SetVectorMask<float,MaskMode::NORMAL>(uint64_t(16));
                Cast<float,T,false>(convertedInputs,rawInputs,RoundMode::CAST_NONE,uint64_t(64),2,UnaryRepeatParams(1,1,16,8));
            } else {
                Cast(fp_,rawP_,RoundMode::CAST_NONE,validRows*slot_);
                Cast(fs_,rawS_,RoundMode::CAST_NONE,CONTIG_SEGMENTS?((6*segmentCount+7)/8*8):(2*segmentCount*slot_));
            }
            PipeBarrier<PIPE_V>();
        }
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessProcessProductsStageStage(StageSegmentCountType& segmentCount) {
        if constexpr(THIN_SINGLE_FP32) { ProcessProductsStage(); } else if constexpr(NATIVE_ONE_ROW) { ProcessProductPairsStage(segmentCount); } else if constexpr(NATIVE_FOUR_ROW_SINGLE) {
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            // One repeat forms each independent set of dot/norm products.
            const LocalTensor<float> operands(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),96);
            const LocalTensor<float> products(TPosition::VECCALC,uint32_t(W(4).GetPhyAddr()),64);
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(32));
            Mul<float,false>(products,operands,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,4,8,0));
            Duplicate(projection_,0.0F,8);PipeBarrier<PIPE_V>();
            WholeReduceSum(lengths_,W(5),3,4,1,1,1);
            WholeReduceSum(projection_,W(4),3,4,1,1,1);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(4));
            Maxs<float,false>(W(4),lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Div<float,false>(projection_,projection_,W(4),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Maxs<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Mins<float,false>(projection_,projection_,1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Muls<float,false>(projection_,projection_,-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(32));
            MulAddDst<float,float,false>(W(2),W(6),W(0),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            WholeReduceSum(scores_,W(2),3,4,1,1,1);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        } else ProcessMulStage2(segmentCount);
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessProductsStage() {
        {
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(8));
            if constexpr(NATIVE_SINGLE_POINT) {
                const LocalTensor<float> products(TPosition::VECCALC,uint32_t(W(4).GetPhyAddr()),16);
                const LocalTensor<float> operands(TPosition::VECCALC,uint32_t(W(2).GetPhyAddr()),24);
                Mul<float,false>(products,operands,W(0),uint64_t(64),2,BinaryRepeatParams(1,1,1,1,2,0));
            } else {
                Mul<float,false>(W(4),W(0),W(0),uint64_t(64),1,same);
                Mul<float,false>(W(5),W(2),W(0),uint64_t(64),1,same);
            }
            PipeBarrier<PIPE_V>();
            WholeReduceSum(lengths_,W(NATIVE_SINGLE_POINT?5:4),3,1,1,1,1);
            WholeReduceSum(projection_,W(NATIVE_SINGLE_POINT?4:5),3,1,1,1,1);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(1));
            // The certificate tests the original norm; clamping only the
            // divisor cannot admit an underflowed or degenerate segment.
            Maxs<float,false>(W(4),lengths_,0.000244140625F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Div<float,false>(projection_,projection_,W(4),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Maxs<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Mins<float,false>(projection_,projection_,1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            ProjectSingleResidual(W(6),projection_,W(2),W(0),scores_,unary,same);
        }
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessCopyStage(StageSegmentCountType& segmentCount) {
        if constexpr(NATIVE_SINGLE_POINT) {
            // Padded native DMA records already form complete coordinate vectors.
            p_=fp_;a_=fs_;c_=fs_[slot_];
        } else {
        Copy(p_,fp_,THIN_SINGLE_FP32?8:PAIR_STRIDE*8,rows_,CopyRepeatParams(1,0,PAIR_STRIDE,slot_/8));
        if constexpr(CONTIG_SEGMENTS) {
            auto sourceOffsets=BLOCK_INDICES?pointOffsets_:segmentOffsets_;
            const uint32_t coordinateCount=BLOCK_INDICES?(segmentCount<=2?16:segmentCount<=4?32:64):64;
            Gather(a_,fs_,sourceOffsets.template ReinterpretCast<uint32_t>(),0U,coordinateCount);
            Gather(c_,fs_,sourceOffsets.template ReinterpretCast<uint32_t>(),12U,coordinateCount);
        } else if constexpr(NATIVE_FOUR_ROW_SINGLE) {
            const LocalTensor<float> endpointPairs(TPosition::VECCALC,uint32_t(a_.GetPhyAddr()),64);
            Copy(endpointPairs,fs_,uint64_t(32),2,CopyRepeatParams(1,0,4,1));
        } else {
            Copy(a_,fs_,THIN_SINGLE_FP32?8:PAIR_STRIDE*8,rows_,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
            Copy(c_,fs_[slot_],THIN_SINGLE_FP32?8:PAIR_STRIDE*8,rows_,CopyRepeatParams(1,2*slot_/8,PAIR_STRIDE,0));
        }
        }
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType, class StagePointPositionType, class StageValidRowsType>
    __aicore__ __attribute__((always_inline)) inline bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessReinterpretCastStage2(StageSegmentCountType& segmentCount, StagePointPositionType& pointPosition, StageValidRowsType& validRows) {
        if constexpr(UsesNativeOneRowStage3()) {
            // Axis alignment is a geometric property of the loaded endpoints.
            // The exact lattice/bound checks below guard the direct AABB result.
            const uint32_t pairs=ROWS==1?segmentCount:validRows*PAIR_STRIDE;
            const uint32_t active=(1U<<pairs)-1U;
            auto axisFlags=zeroExact_.template ReinterpretCast<uint8_t>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(pairs*8U));
            Compare<float,uint8_t,false>(axisFlags,a_,c_,CMPMODE::EQ,uint64_t(64),1,BinaryRepeatParams(1,1,1,8,8,8));
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            const uint64_t fixed=axisFlags.template ReinterpretCast<uint64_t>().GetValue(0);
            const uint64_t axis=((fixed&(fixed>>1))|(fixed&(fixed>>2))|((fixed>>1)&(fixed>>2)))&0x0101010101010101ULL;
            const uint32_t aligned=uint32_t((axis*0x0102040810204080ULL)>>56);
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            bool directReady=false;
            if((aligned&active)==active) {
                if constexpr(NATIVE_FOUR_ROW_SINGLE) directReady=TryAxisSingle(validRows);
                else directReady=TryAxisDyadicGeometry(validRows,segmentCount);
            }
            if(directReady) {
                WriteTinyResult(pointPosition,validRows);
                return true;
            }
        }
        return false;
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    __aicore__ constexpr bool KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::UsesNativeOneRowStage3() { return (NATIVE_ONE_ROW || NATIVE_FOUR_ROW_SINGLE ||
                     (COMPLETE_ENDPOINT_LOAD && FIXED_SEGMENTS==2));
    }

template<class T, uint32_t POINT_ROWS, uint32_t PAIR_STRIDE, bool CONTIG_SEGMENTS, bool BLOCK_INDICES, bool SINGLE_SEGMENT, bool OPTIMIZE_FOUR_ROW, uint32_t FIXED_SEGMENTS, bool PAIR_SINGLE_POINT, bool DEFER_SOC_INIT>
    template<class StageSegmentCountType>
    __aicore__ __attribute__((always_inline)) inline void KernelTreeTiny<T, POINT_ROWS, PAIR_STRIDE, CONTIG_SEGMENTS, BLOCK_INDICES, SINGLE_SEGMENT, OPTIMIZE_FOUR_ROW, FIXED_SEGMENTS, PAIR_SINGLE_POINT, DEFER_SOC_INIT>::ProcessMulStage2(StageSegmentCountType& segmentCount) {
        if constexpr(UsesRowsStage()) {
            // Eight independent padded point/segment pairs fit one native vector packet.
            // Only the first validRows outputs are stored; inactive rows cannot join an argmin.
            const BinaryRepeatParams same{1,1,1,8,8,8};
            const UnaryRepeatParams unary{1,1,8,8};
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Mul<float,false>(W(4),W(0),W(0),uint64_t(64),1,same);
            Mul<float,false>(W(5),W(2),W(0),uint64_t(64),1,same);
            Duplicate<float,false>(lengths_,1.0F,uint64_t(64),1,1,8);
            Duplicate<float,false>(projection_,0.0F,uint64_t(64),1,1,8);
            Duplicate<float,false>(scores_,__builtin_inff(),uint64_t(64),1,1,8);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
            WholeReduceSum<float,false>(lengths_,W(4),3,8,1,1,1);
            WholeReduceSum<float,false>(projection_,W(5),3,8,1,1,1);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            CompareScalar<float,uint8_t,false>(masks_,lengths_,0.0F,CMPMODE::GT,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Select(lengths_,masks_,lengths_,1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
            Div<float,false>(projection_,projection_,lengths_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            Maxs<float,false>(projection_,projection_,0.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Mins<float,false>(projection_,projection_,1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            Brcb(W(6),projection_,1,BrcbRepeatParams(1,8));PipeBarrier<PIPE_V>();
            CompareScalar<float,uint8_t,false>(masks_,W(6),1.0F,CMPMODE::LT,uint64_t(64),1,unary);
            Muls<float,false>(W(6),W(6),-1.0F,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
            MulAddDst<float,float,false>(W(2),W(6),W(0),uint64_t(64),1,same);
            Sub<float,false>(W(4),p_,c_,uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            SetCmpMask(masks_);
            Select<float,SELMODE::VSEL_CMPMASK_SPR>(W(2),W(2),W(4),1,same);PipeBarrier<PIPE_V>();
            Mul<float,false>(W(2),W(2),W(2),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(3));
            WholeReduceSum<float,false>(scores_,W(2),3,8,1,1,1);PipeBarrier<PIPE_V>();
            SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
            Select(scores_,masks_[384],scores_,__builtin_inff(),SELMODE::VSEL_TENSOR_SCALAR_MODE,64);PipeBarrier<PIPE_V>();
        } else { ProcessMulStage(segmentCount); }
    }
