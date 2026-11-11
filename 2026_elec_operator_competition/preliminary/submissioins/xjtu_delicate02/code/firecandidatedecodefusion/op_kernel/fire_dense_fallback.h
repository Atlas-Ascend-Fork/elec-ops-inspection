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

template<class T, uint32_t K> class KernelFireCandidateDecodeFusion {
    static constexpr uint32_t P=K+8;
    static constexpr uint32_t P0=0,S0=P0+P*5*4,V0=S0+K*8*4,O0=V0+K*8*4;
    static constexpr uint32_t I0=O0+K*5*4,G0=I0+K*2*4,M0=G0+64*4;
    StaticUBBuffer<P0,P*5*4> priorBuf;
    StaticUBBuffer<S0,K*8*4> sigBuf;
    StaticUBBuffer<V0,K*8*4> planeBuf;
    StaticUBBuffer<O0,K*5*4> outBuf;
    StaticUBBuffer<I0,K*2*4> indexBuf;
    StaticUBBuffer<G0,64*4> genBuf;
    StaticUBBuffer<M0,K> maskBuf;
    GlobalTensor<T> raw, out;
    GlobalTensor<float> prior;
    FireCandidateDecodeFusionTilingData d;
    __aicore__ inline __attribute__((always_inline)) void InitDoubleOutputContext(LocalTensor<float> buf0,LocalTensor<float> buf1,uint64_t (&srcList)[2][16],uint64_t (&dstList)[2][16],uint32_t tileCount) {
        for(uint32_t parity=0;parity<(tileCount>1?2u:1u);++parity) {
            auto src=parity==0?buf1:buf0;auto dst=parity==0?buf0:buf1;
            for(uint32_t c=0;c<8;++c) {
                uint32_t channel=c<5?c:0;
                srcList[parity][c]=reinterpret_cast<uint64_t>(src[channel*P].GetPhyAddr());
                srcList[parity][c+8]=reinterpret_cast<uint64_t>(src[channel*P+8].GetPhyAddr());
                dstList[parity][2*c]=reinterpret_cast<uint64_t>(dst[c*8].GetPhyAddr());
                dstList[parity][2*c+1]=reinterpret_cast<uint64_t>(dst[(c+8)*8].GetPhyAddr());
            }
        }
    }
    __aicore__ inline __attribute__((always_inline)) void DecodeBaseGeometry(LocalTensor<float> v,LocalTensor<float> p,uint32_t count) {
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(count);
        for(uint32_t c=0;c<4;++c) {Muls<float,false>(v[c*P],v[c*P],2.f,MASK_PLACEHOLDER,1,{1,1,8,8});}PipeBarrier<PIPE_V>();
        for(uint32_t c=0;c<2;++c) {Adds<float,false>(v[c*P],v[c*P],-.5f,MASK_PLACEHOLDER,1,{1,1,8,8});}PipeBarrier<PIPE_V>();
        for(uint32_t c=0;c<2;++c) {Add<float,false>(v[c*P],v[c*P],p[c*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});}PipeBarrier<PIPE_V>();
        for(uint32_t c=2;c<4;++c) {Mul<float,false>(v[c*P],v[c*P],v[c*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});}PipeBarrier<PIPE_V>();
        for(uint32_t c=2;c<4;++c) {Mul<float,false>(v[c*P],v[c*P],p[(c+1)*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});}PipeBarrier<PIPE_V>();
        for(uint32_t c=0;c<4;++c) {Mul<float,false>(v[c*P],v[c*P],p[2*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});}PipeBarrier<PIPE_V>();
        Mul<float,false>(v[4*P],v[4*P],v[5*P],MASK_PLACEHOLDER,1,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
        ResetMask();SetMaskNorm();
    }
    __aicore__ inline __attribute__((always_inline)) void PrefetchBaseSuccessor(LocalTensor<float> v,uint32_t tile,uint32_t end) {
        uint32_t nextB=(tile+1)%d.batch,nextN=((tile+1)/d.batch)*K;
        uint32_t nextCount=d.n-nextN<K?d.n-nextN:K;
        auto nextRaw=v.template ReinterpretCast<T>();if constexpr(sizeof(T)==2) nextRaw=nextRaw[8*K];
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyValidInput(nextRaw,raw,(nextB*d.n+nextN)*6,nextCount*6);
    }
public:
    __aicore__ inline void Init(GM_ADDR r, GM_ADDR p, GM_ADDR o, FireCandidateDecodeFusionTilingData t) {
        d=t; raw.SetGlobalBuffer((__gm__ T*)r); prior.SetGlobalBuffer((__gm__ float*)p); out.SetGlobalBuffer((__gm__ T*)o);
        InitSocState();ResetMask();SetMaskNorm();
        auto idx=indexBuf.template Get<int32_t>();
        for(uint32_t j=0;j<8;++j) idx.SetValue(j,static_cast<int32_t>(j));
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        for(uint32_t span=8;span<K;span*=2) {
            uint32_t length=K-span<span?K-span:span;
            Adds(idx[span],idx,static_cast<int32_t>(span),length);PipeBarrier<PIPE_V>();
        }
        Muls(idx[K],idx,static_cast<int32_t>(20),K);PipeBarrier<PIPE_V>();
        Muls(idx,idx,static_cast<int32_t>(24),K);PipeBarrier<PIPE_V>();
        Duplicate(genBuf.template Get<float>(),1.0f,64); PipeBarrier<PIPE_V>();
        Duplicate(maskBuf.template Get<int32_t>(),static_cast<int32_t>(0x1f1f1f1f),K/4); PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    __aicore__ inline void Process() {
        auto p=priorBuf.template Get<float>();auto o=outBuf.template Get<float>();auto idx=indexBuf.template Get<uint32_t>();
        auto buf0=sigBuf.template Get<float>();auto buf1=planeBuf.template Get<float>();
        uint32_t tiles=(d.n+K-1)/K,total=tiles*d.batch;
        uint32_t begin=total*GetBlockIdx()/d.blocks,end=total*(GetBlockIdx()+1)/d.blocks;
        uint64_t srcList[2][16],dstList[2][16];
        InitDoubleOutputContext(buf0,buf1,srcList,dstList,end-begin);
        Duplicate(buf0,0.f,K*8);Duplicate(buf1,0.f,K*8);Duplicate(p,0.f,P*5);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        uint64_t reserved=0;
        uint32_t cachedN=d.n;
        for(uint32_t tile=begin;tile<end;++tile) {
            uint32_t parity=(tile-begin)&1;
            auto s=parity==0?buf0:buf1;auto v=parity==0?buf1:buf0;auto packed=s;
            auto r=s.template ReinterpretCast<T>();if constexpr(sizeof(T)==2) r=r[8*K];
            uint32_t b=tile%d.batch, n=(tile/d.batch)*K;
            uint32_t count=(d.n-n<K?d.n-n:K);
            if(n!=cachedN) {
                CopyValidInput(o,prior,n*5,count*5);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                GatherChannelPlanes<5>(p,o,idx[K],P,count);
                cachedN=n;
            }
            {
                if(tile==begin) CopyValidInput(r,raw,(b*d.n+n)*6,count*6);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if constexpr(sizeof(T)==2) { Cast(s,r,RoundMode::CAST_NONE,count*6); PipeBarrier<PIPE_V>(); }
                Muls(s,s,-1.0f,count*6); PipeBarrier<PIPE_V>(); Exp(s,s,count*6); PipeBarrier<PIPE_V>(); Adds(s,s,1.0f,count*6); PipeBarrier<PIPE_V>(); SetMaskCount(); SetVectorMask<float,MaskMode::COUNTER>(count*6); Div<float,false>(s,genBuf.template Get<float>(),s,MASK_PLACEHOLDER,1,{1,1,1,8,0,8}); ResetMask(); SetMaskNorm(); PipeBarrier<PIPE_V>();
                GatherChannelPlanes<6>(v,s,idx,P,count);
                RepairSensitiveCenters<T,K>(v,p,idx,raw,s,P,count,(b*d.n+n)*6);
                DecodeBaseGeometry(v,p,count);
                SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
                ResetMask();
                uint8_t repeats=static_cast<uint8_t>((count+15)/16);
                TransDataTo5HD<float>(dstList[parity],srcList[parity],TransDataTo5HDParams(false,false,repeats,repeats==1?0:16,repeats==1?0:2)); PipeBarrier<PIPE_V>();
                if(tile+1<end) PrefetchBaseSuccessor(v,tile,end);
                GatherMask(o,packed,maskBuf.template Get<uint32_t>(),true,count*8,{1,1,8,0},reserved); PipeBarrier<PIPE_V>();
                if constexpr(sizeof(T)==2) Cast(outBuf.template Get<T>(),o,RoundMode::CAST_RINT,count*5); PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyPad(out[(b*d.n+n)*5],outBuf.template Get<T>(),DataCopyExtParams{1,count*5*uint32_t(sizeof(T)),0,0,0});
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
};
// Split packed six-channel activations with sequential mask compaction.
