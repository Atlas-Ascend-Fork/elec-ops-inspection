/*
 * Copyright (c) 2026 Delicate02. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "fire_math.h"

__aicore__ inline __attribute__((always_inline)) void LaunchStrideMax(const LocalTensor<float>& source,LocalTensor<float>& scratch,uint32_t count) {
    ResetMask();SetMaskNorm();
    uint32_t full=count/64,tail=count%64;
    if(full) {
        WholeReduceMax(scratch,source,64,full,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        if(full>1) {
            WholeReduceMax(scratch,scratch,full,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
        }
    }
    if(tail) {
        WholeReduceMax(scratch[32],source[full*64],tail,1,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_S>(EVENT_ID1);
}
template<bool ExplicitScalarRead=false>
__aicore__ inline float DirectStrideMax(const LocalTensor<float>& source,
    LocalTensor<float>& scratch,uint32_t count) {
    LaunchStrideMax(source,scratch,count);
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    uint32_t full=count/64,tail=count%64;
    float maximum;
#if __NPU_ARCH__ == 2201 && !defined(ASCENDC_CPU_DEBUG) && !defined(__CCE_KT_TEST__)
    if constexpr(ExplicitScalarRead) {
        // V_S1 completes the reduction; S_V1 releases its Scalar readers.
        auto* reduced=reinterpret_cast<const volatile __ubuf__ float*>(scratch.GetPhyAddr());
        maximum=full?reduced[0]:0.f;
        if(tail){float last=reduced[32];if(last>maximum)maximum=last;}
    } else
#endif
    {
        maximum=full?scratch.GetValue(0):0.f;
        if(tail){float last=scratch.GetValue(32);if(last>maximum)maximum=last;}
    }
    SetFlag<HardEvent::S_V>(EVENT_ID1);WaitFlag<HardEvent::S_V>(EVENT_ID1);
    return maximum;
}
// The one-tile guard owns a separate, immutable reduction result until END.
__aicore__ inline void BeginOneStrideMax(const LocalTensor<float>& source,
    LocalTensor<float>& scratch,uint32_t count) {
    LaunchStrideMax(source,scratch,count);
}
__aicore__ inline float FinishOneStrideMax(LocalTensor<float>& scratch,uint32_t count) {
    WaitFlag<HardEvent::V_S>(EVENT_ID1);
    float maximum=count>=64?scratch.GetValue(0):0.f;
    if(count%64){float last=scratch.GetValue(32);if(last>maximum)maximum=last;}
    return maximum;
}
template<class T,uint32_t K,uint32_t valuesOffset,uint32_t priorOffset,uint32_t scratchOffset>
__aicore__ __attribute__((noinline)) void RepairDenseCold(GM_ADDR rawAddress,
    uint32_t count,uint32_t rawOffset) {
    constexpr uint32_t pitch=K+8;
    LocalTensor<float> values(TPosition::VECCALC,valuesOffset,K*7);
    LocalTensor<float> priors(TPosition::VECCALC,priorOffset,pitch*5);
    LocalTensor<float> scratch(TPosition::VECCALC,scratchOffset,K*8);
    GlobalTensor<T> raw;raw.SetGlobalBuffer((__gm__ T*)rawAddress);
    RepairCenterRows<T,false>(values,priors,raw,scratch,pitch,count,rawOffset);
}


template<class T,uint32_t K,uint32_t valuesOffset,uint32_t priorOffset,uint32_t scratchOffset>
__aicore__ __attribute__((noinline)) void RepairScaledDenseCold(GM_ADDR rawAddress,
    uint32_t count,uint32_t rawOffset) {
    constexpr uint32_t pitch=FireDensePitchPolicy<T,K>::Value;
    LocalTensor<float> values(TPosition::VECCALC,valuesOffset,K*7);
    LocalTensor<float> priors(TPosition::VECCALC,priorOffset,pitch*5);
    LocalTensor<float> scratch(TPosition::VECCALC,scratchOffset,K*8);
    GlobalTensor<T> raw;raw.SetGlobalBuffer((__gm__ T*)rawAddress);
    RepairCenterRows<T,true>(values,priors,raw,scratch,pitch,count,rawOffset);
}


template<unsigned P,bool CombinedScore>
__aicore__ inline __attribute__((always_inline)) void DecodeMaskedGeometry(LocalTensor<float> s,LocalTensor<float> v,LocalTensor<float> p,uint32_t count) {
SetMaskCount();
if(count*2>=P) {
    // The two center planes and two size planes are disjoint.
    // Each consumer follows a barrier after both producers.
    SetVectorMask<float,MaskMode::COUNTER>(P+count);
    Adds<float,false>(v,s,-.5f,MASK_PLACEHOLDER,1,{1,1,8,8});
    Mul<float,false>(v[2*P],s[2*P],s[2*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
    PipeBarrier<PIPE_V>();
    Add<float,false>(s,v,p,MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
    Mul<float,false>(s[2*P],v[2*P],p[3*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
    PipeBarrier<PIPE_V>();
} else {
    SetVectorMask<float,MaskMode::COUNTER>(count);
    for(uint32_t c=0;c<2;++c)Adds<float,false>(v[c*P],s[c*P],-.5f,MASK_PLACEHOLDER,1,{1,1,8,8});
    for(uint32_t c=2;c<4;++c)Mul<float,false>(v[c*P],s[c*P],s[c*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
    PipeBarrier<PIPE_V>();
    for(uint32_t c=0;c<2;++c)Add<float,false>(s[c*P],v[c*P],p[c*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
    for(uint32_t c=2;c<4;++c)Mul<float,false>(s[c*P],v[c*P],p[(c+1)*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
    PipeBarrier<PIPE_V>();
}
SetVectorMask<float,MaskMode::COUNTER>(count);
for(uint32_t c=0;c<4;++c)Mul<float,false>(s[c*P],s[c*P],p[2*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
PipeBarrier<PIPE_V>();
if constexpr(CombinedScore) {
    Muls<float,false>(s[4*P],s[4*P],.25f,MASK_PLACEHOLDER,1,{1,1,8,8});
} else {
    Mul<float,false>(v[4*P],s[4*P],s[5*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
    PipeBarrier<PIPE_V>();
    Muls<float,false>(s[4*P],v[4*P],.25f,MASK_PLACEHOLDER,1,{1,1,8,8});
}
PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
uint8_t repeats=static_cast<uint8_t>((count+15)/16);
scatter_vnchwconv_b32(VA0,VA2,repeats,repeats==1?0:16,repeats==1?0:2);
PipeBarrier<PIPE_V>();
}
template<unsigned P>
__aicore__ inline __attribute__((always_inline)) void InitDenseOutputContext(LocalTensor<float> values,LocalTensor<float> v,uint64_t* outSrc,uint64_t* outDst) {
#pragma unroll
        for(uint32_t c=0;c<8;++c) {
            uint32_t channel=c<5?c:0;
            outSrc[c]=reinterpret_cast<uint64_t>(values[channel*P].GetPhyAddr());
            outSrc[c+8]=reinterpret_cast<uint64_t>(values[channel*P+8].GetPhyAddr());
            outDst[2*c]=reinterpret_cast<uint64_t>(v[c*8].GetPhyAddr());
            outDst[2*c+1]=reinterpret_cast<uint64_t>(v[(c+8)*8].GetPhyAddr());
        }
        // The dense path has one fixed transpose layout. No other operation
        // in this path writes VA0..VA3, including the numerical repair.
        set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
        set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
}

template<class T,unsigned K,bool Full>
__aicore__ inline __attribute__((always_inline)) void StageDenseRaw(LocalTensor<T> target,LocalTensor<T> suffix,GlobalTensor<T> raw,uint32_t row,uint32_t count) {
    const uint32_t elements=count*6;
    if constexpr(sizeof(T)==2 || Full) CopyDenseInput(target,raw,row*6,elements);
    else {
        const uint32_t first=elements<5*K?elements:5*K;
        CopyDenseInput(target,raw,row*6,first);
        if(elements>first)CopyDenseInput(suffix,raw,row*6+first,elements-first);
    }
}
template<unsigned K>
__aicore__ inline __attribute__((always_inline)) void CopySplitRaw(LocalTensor<float> next,LocalTensor<int32_t> packed,LocalTensor<int32_t> suffix,uint32_t count) {
    const uint32_t elements=count*6,first=elements<5*K?elements:5*K;
    auto target=next.template ReinterpretCast<int32_t>();
    Adds(target,packed,int32_t(0),first);PipeBarrier<PIPE_V>();
    if(elements>first) {Adds(target[first],suffix,int32_t(0),elements-first);PipeBarrier<PIPE_V>();}
}
template<unsigned Words>
__aicore__ inline __attribute__((always_inline)) void SetDensePriorHeads(LocalTensor<uint64_t> masks,uint64_t keep) {
    constexpr uint32_t bits[5]={0x42108421U,0x10842108U,0x84210842U,0x21084210U,0x08421084U};
#pragma unroll
    for(uint32_t c=0;c<5;++c) {
        const uint32_t offset=c==0?0:8*(5-c);
        const uint64_t head=uint64_t(bits[offset%5])|(uint64_t(bits[(offset+1)%5])<<32);
        masks.SetValue(c*Words/2,head&keep);
    }
}
__aicore__ inline __attribute__((always_inline)) void SplitPackedEvenOdd(LocalTensor<float> values,LocalTensor<float> even,LocalTensor<float> odd,uint32_t count) {
    SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(6*count);
    GatherMask(even,values,uint8_t(1),GatherMaskParams{1,1,8,0});
    GatherMask(odd,values,uint8_t(2),GatherMaskParams{1,1,8,0});PipeBarrier<PIPE_V>();
    SetVectorMask<float,MaskMode::COUNTER>(3*count);
}
template<bool Early,class Pipeline,class T>
__aicore__ inline __attribute__((always_inline)) void CompleteDenseInput(Pipeline& pipeline,LocalTensor<T> r,GlobalTensor<T> raw,uint32_t row,uint32_t count) {
    if(count) {
        if constexpr(Early) {
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            pipeline.PrepareNextInput(count);
        } else {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            CopyDenseInput(r,raw,row*6,count*6);
        }
    }
}
template<class T>
__aicore__ inline __attribute__((always_inline)) void EmitDenseOutput(LocalTensor<float> o,LocalTensor<float> v,LocalTensor<uint32_t> mask,LocalTensor<T> castBuffer,GlobalTensor<T> out,uint32_t row,uint32_t count,uint64_t& reserved) {
    GatherMask(o,v,mask,true,count*8,{1,1,8,0},reserved);PipeBarrier<PIPE_V>();
    if constexpr(sizeof(T)==2) {Cast(castBuffer,o,RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();}
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    CopyDenseOutput(out,castBuffer,row*5,count*5);
}
