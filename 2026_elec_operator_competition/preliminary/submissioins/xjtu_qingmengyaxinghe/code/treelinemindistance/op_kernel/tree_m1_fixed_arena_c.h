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
// Inline implementation of TreeM1FixedArena.

    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> TreeM1FixedArena::C(uint32_t col) {
        const uint32_t offset=col*104U+(col==9?16U:0U);
        return LocalTensor<float>(TPosition::VECCALC,32U+offset*4U,1352U-offset);
    }

    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> TreeM1FixedArena::W(uint32_t col) {
        return LocalTensor<float>(TPosition::VECCALC,30656U+col*4096U,9504U-col*1024U);
    }

    __aicore__ __attribute__((always_inline)) inline LocalTensor<float> TreeM1FixedArena::P(uint32_t col) {
        return LocalTensor<float>(TPosition::VECCALC,17984U+col*4096U,3168U-col*1024U);
    }

    __aicore__ __attribute__((always_inline)) inline void TreeM1FixedArena::Process(
        GM_ADDR points,GM_ADDR segments,GM_ADDR distance,GM_ADDR indices,uint32_t pointCount) {

        const uint32_t tiles=(pointCount+1023U)/1024U;
        const uint32_t owner=GetBlockIdx();
        uint32_t batch,begin;
        if(owner<tiles) {batch=0;begin=owner*1024U;}
        else {batch=owner/tiles;begin=(owner-batch*tiles)*1024U;}
        const uint32_t validRows=pointCount-begin<1024U?pointCount-begin:1024U;
        GlobalTensor<float> pg,sg,dg;GlobalTensor<int32_t> ig;
        pg.SetGlobalBuffer((__gm__ float*)points);sg.SetGlobalBuffer((__gm__ float*)segments);
        dg.SetGlobalBuffer((__gm__ float*)distance);ig.SetGlobalBuffer((__gm__ int32_t*)indices);
        LocalTensor<float> sf(TPosition::VECCALC,0U,8U),pf(TPosition::VECCALC,5440U,3136U);
        LocalTensor<uint32_t> bitmap(TPosition::VECCALC,68672U,1056U);
        LocalTensor<uint8_t> mask(TPosition::VECCALC,77120U,160U);
        LocalTensor<int32_t> outIdx(TPosition::VECCALC,123392U,1024U);
        LocalTensor<float> out(TPosition::VECCALC,127488U,1024U);
        LocalTensor<float> projection(TPosition::VECCALC,127552U,1024U);
        const BinaryRepeatParams same{1,1,1,8,8,8},coefficient{1,1,1,8,8,0};
        const UnaryRepeatParams unary{1,1,8,8};
        // All six segment coordinates are overwritten by the following DMA.
        // Only a partial point tile has a vector initialization to order before MTE2.
        if(validRows!=1024U) {
            Duplicate(pf,0.0F,(validRows*3U+63U)/64U*64U);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        }
        DataCopyPad(sf,sg[batch*6U],DataCopyExtParams{1,24U,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        DataCopyPad(pf,pg[(batch*pointCount+begin)*3U],DataCopyExtParams{1,validRows*12U,0,0,0},DataCopyPadExtParams<float>{false,0,0,0.0F});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        constexpr uint32_t words[3]={0x49249249U,0x92492492U,0x24924924U};
        constexpr uint64_t lanes[3]={0x249249ULL,0x492492ULL,0x924924ULL};
        SetMaskNorm();
        #pragma unroll
        for(uint32_t phase=0;phase<3;++phase) {
            SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(0),lanes[phase]);
            Duplicate<uint32_t,false>(bitmap,words[phase],uint64_t(64),1,1,8);
        }
        PipeBarrier<PIPE_V>();SetVectorMask<uint32_t,MaskMode::NORMAL>(uint64_t(64));
        Copy(bitmap[24],bitmap,24,4,CopyRepeatParams(1,1,3,0));PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);PipeBarrier<PIPE_V>();
        Brcb(C(0),sf,1,BrcbRepeatParams(13,104));ProcessCopyStage4(pointCount, batch, begin, validRows, pg, sg, dg, ig, pf, bitmap, mask, outIdx, out, projection, same, coefficient, unary);
    }

    template<class StagePointCountType, class StageBatchType, class StageBeginType, class StageValidRowsType, class StagePgType, class StageSgType, class StageDgType, class StageIgType, class StagePfType, class StageBitmapType, class StageMaskType, class StageOutIdxType, class StageOutType, class StageProjectionType, class StageSameType, class StageCoefficientType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void TreeM1FixedArena::ProcessCopyStage4(StagePointCountType& pointCount, StageBatchType& batch, StageBeginType& begin, StageValidRowsType& validRows, StagePgType& pg, StageSgType& sg, StageDgType& dg, StageIgType& ig, StagePfType& pf, StageBitmapType& bitmap, StageMaskType& mask, StageOutIdxType& outIdx, StageOutType& out, StageProjectionType& projection, StageSameType& same, StageCoefficientType& coefficient, StageUnaryType& unary) {
        PipeBarrier<PIPE_V>();
        Copy(C(0)[8],C(0),56,6,CopyRepeatParams(1,0,13,13));
        PipeBarrier<PIPE_V>();SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
        Sub<float,false>(C(6),C(3),C(0),uint64_t(64),3,BinaryRepeatParams(1,1,1,13,13,13));PipeBarrier<PIPE_V>();
        Sub<float,false>(W(0),C(3),C(6),uint64_t(64),3,BinaryRepeatParams(1,1,1,132,13,13));PipeBarrier<PIPE_V>();
        Add<float,false>(W(3),C(6),W(0),uint64_t(64),3,BinaryRepeatParams(1,1,1,132,13,132));
        Sub<float,false>(C(10),W(0),C(0),uint64_t(64),3,BinaryRepeatParams(1,1,1,13,132,13));PipeBarrier<PIPE_V>();
        Sub<float,false>(W(3),C(3),W(3),uint64_t(64),3,BinaryRepeatParams(1,1,1,132,13,132));PipeBarrier<PIPE_V>();
        Add<float,false>(C(10),C(10),W(3),uint64_t(64),3,BinaryRepeatParams(1,1,1,13,13,132));
        Mul<float,false>(C(9),C(6),C(6),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(C(9),C(7),C(7),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(C(9),C(8),C(8),uint64_t(64),1,same);PipeBarrier<PIPE_V>();
        CompareScalar<float,uint8_t,false>(mask,C(9),0.0F,CMPMODE::GT,uint64_t(64),1,unary);PipeBarrier<PIPE_V>();
        Select<float,uint8_t,false>(C(9),mask,C(9),1.0F,SELMODE::VSEL_TENSOR_SCALAR_MODE,uint64_t(64),1,same);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        SetFlag<HardEvent::V_S>(EVENT_ID0);
#endif
        PipeBarrier<PIPE_V>();WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);PipeBarrier<PIPE_V>();
        SetMaskCount();
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis) {
            SetVectorMask<float,MaskMode::COUNTER>(uint64_t(3072));
            GatherMask(P(axis),pf,bitmap[axis==0?0U:(axis==1?16U:8U)].template ReinterpretCast<float>(),
                uint8_t(0),GatherMaskParams(1,1,8,1));
        }
        PipeBarrier<PIPE_V>();SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(uint64_t(64));
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Copy<float,false>(pf[axis*1024U],C(axis),uint64_t(64),16,CopyRepeatParams(1,0,8,0));
        PipeBarrier<PIPE_V>();
        Sub<float,false>(W(0),P(0),pf,uint64_t(64),48,same);
        Sub<float,false>(W(3),P(0),W(0),uint64_t(64),48,same);
        Mul<float,false>(projection,W(0),C(6),uint64_t(64),16,coefficient);
        Add<float,false>(W(6),W(0),W(3),uint64_t(64),48,same);
        MulAddDst<float,float,false>(projection,W(1),C(7),uint64_t(64),16,coefficient);
        Sub<float,false>(W(3),W(3),pf,uint64_t(64),48,same);
        MulAddDst<float,float,false>(projection,W(2),C(8),uint64_t(64),16,coefficient);
        Sub<float,false>(W(6),P(0),W(6),uint64_t(64),48,same);
        Div<float,false>(projection,projection,C(9),uint64_t(64),16,coefficient);
        ProcessAddStage(pointCount, batch, begin, validRows, pg, sg, dg, ig, mask, outIdx, out, projection, same, coefficient, unary);
    }

    template<class StagePointCountType, class StageBatchType, class StageBeginType, class StageValidRowsType, class StagePgType, class StageSgType, class StageDgType, class StageIgType, class StageMaskType, class StageOutIdxType, class StageOutType, class StageProjectionType, class StageSameType, class StageCoefficientType, class StageUnaryType>
    __aicore__ __attribute__((always_inline)) inline void TreeM1FixedArena::ProcessAddStage(StagePointCountType& pointCount, StageBatchType& batch, StageBeginType& begin, StageValidRowsType& validRows, StagePgType& pg, StageSgType& sg, StageDgType& dg, StageIgType& ig, StageMaskType& mask, StageOutIdxType& outIdx, StageOutType& out, StageProjectionType& projection, StageSameType& same, StageCoefficientType& coefficient, StageUnaryType& unary) {
        Add<float,false>(W(3),W(3),W(6),uint64_t(64),48,same);
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const uint32_t errorBits=C(10).template ReinterpretCast<uint32_t>().GetValue(0)
            |C(11).template ReinterpretCast<uint32_t>().GetValue(0)
            |C(12).template ReinterpretCast<uint32_t>().GetValue(0);
        const bool directionErrorZero=(errorBits&0x7fffffffU)==0;
#endif
        PipeBarrier<PIPE_V>();
        Maxs<float,false>(projection,projection,0.0F,uint64_t(64),16,unary);PipeBarrier<PIPE_V>();
        Mins<float,false>(projection,projection,1.0F,uint64_t(64),16,unary);PipeBarrier<PIPE_V>();
        // Negate cached directions once instead of every query projection.
        // For the following FMAs, t*(-u) has the same signed product as (-t)*u.
        Muls<float,false>(C(6),C(6),-1.0F,uint64_t(64),3,UnaryRepeatParams(1,1,13,13));
        Muls<float,false>(C(10),C(10),-1.0F,uint64_t(64),3,UnaryRepeatParams(1,1,13,13));
        PipeBarrier<PIPE_V>();
#if defined(DTYPE_TREE_POINTS) && TREE_NATIVE_FLOAT_TOKEN(DTYPE_TREE_POINTS)
        if(directionErrorZero) {
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis)
                MulAddDst<float,float,false>(W(axis),projection,C(axis+6U),uint64_t(64),16,coefficient);
        } else
#endif
        {
    #pragma unroll
            for(uint32_t axis=0;axis<3;++axis) {
                MulAddDst<float,float,false>(W(axis),projection,C(axis+6U),uint64_t(64),16,coefficient);
                MulAddDst<float,float,false>(W(axis+3U),projection,C(axis+10U),uint64_t(64),16,coefficient);
            }
        }
        PipeBarrier<PIPE_V>();
#pragma unroll
        for(uint32_t axis=0;axis<3;++axis)Add<float,false>(W(axis),W(axis),W(axis+3U),uint64_t(64),16,same);
        PipeBarrier<PIPE_V>();
        Mul<float,false>(P(0),W(0),W(0),uint64_t(64),16,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(P(0),W(1),W(1),uint64_t(64),16,same);PipeBarrier<PIPE_V>();
        MulAddDst<float,float,false>(P(0),W(2),W(2),uint64_t(64),16,same);PipeBarrier<PIPE_V>();
        CompareScalar(mask,P(0),1152921504606846976.0F,CMPMODE::LE,1024);
        Sqrt<float,false>(out,P(0),uint64_t(64),16,unary);
        Duplicate<int32_t,false>(outIdx,int32_t(0),uint64_t(64),16,1,8);
        SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float norm=C(9).GetValue(0);
        bool needsExact=!(norm>=0.0F && norm<=1152921504606846976.0F);
        auto safe=mask.template ReinterpretCast<uint64_t>();
        ProcessGetValueStage3(pointCount, batch, begin, validRows, pg, sg, dg, ig, outIdx, out, needsExact, safe);
    }

    template<class StagePointCountType, class StageBatchType, class StageBeginType, class StageValidRowsType, class StagePgType, class StageSgType, class StageDgType, class StageIgType, class StageOutIdxType, class StageOutType, class StageNeedsExactType, class StageSafeType>
    __aicore__ __attribute__((always_inline)) inline void TreeM1FixedArena::ProcessGetValueStage3(StagePointCountType& pointCount, StageBatchType& batch, StageBeginType& begin, StageValidRowsType& validRows, StagePgType& pg, StageSgType& sg, StageDgType& dg, StageIgType& ig, StageOutIdxType& outIdx, StageOutType& out, StageNeedsExactType& needsExact, StageSafeType& safe) {
        for(uint32_t first=0;first<validRows;first+=64U) {
            const uint32_t count=validRows-first<64U?validRows-first:64U;
            const uint64_t active=count==64U?~uint64_t(0):(uint64_t(1)<<count)-1U;
            if((safe.GetValue(first/64U)&active)!=active)needsExact=true;
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID3);WaitFlag<HardEvent::V_MTE3>(EVENT_ID3);
        DataCopyPad(dg[batch*pointCount+begin],out,DataCopyExtParams{1,validRows*4U,0,0,0});
        DataCopyPad(ig[batch*pointCount+begin],outIdx,DataCopyExtParams{1,validRows*4U,0,0,0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID3);WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
        ProcessTreeGeneralExactFallbackStage2(pointCount, batch, begin, validRows, pg, sg, dg, ig, needsExact);


    }

    template<class StagePointCountType, class StageBatchType, class StageBeginType, class StageValidRowsType, class StagePgType, class StageSgType, class StageDgType, class StageIgType, class StageNeedsExactType>
    __aicore__ __attribute__((always_inline)) inline void TreeM1FixedArena::ProcessTreeGeneralExactFallbackStage2(StagePointCountType& pointCount, StageBatchType& batch, StageBeginType& begin, StageValidRowsType& validRows, StagePgType& pg, StageSgType& sg, StageDgType& dg, StageIgType& ig, StageNeedsExactType& needsExact) {
        if(needsExact) {
            PipeBarrier<PIPE_ALL>();
            for(uint32_t row=0;row<validRows;++row) {
                const uint32_t pos=batch*pointCount+begin+row;
                TreeGeneralExactFallback<float>((GM_ADDR)pg[pos*3U].GetPhyAddr(),
                    (GM_ADDR)sg[batch*6U].GetPhyAddr(),(GM_ADDR)dg[pos].GetPhyAddr(),
                    (GM_ADDR)ig[pos].GetPhyAddr(),1U);
            }
        }
    }
