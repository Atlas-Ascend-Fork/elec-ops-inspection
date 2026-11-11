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
#include "fire_dense_helpers.h"

template<class T,uint32_t K,bool Early=true> class FlatMaskDecode {
    static constexpr uint32_t P=K+(sizeof(T)==4?24:8);
    static_assert(6*P<=7*K,"dense planes overlap raw prefetch suffix");
    static constexpr uint32_t P0=0,S0=P0+P*5*4,V0=S0+K*8*4+128,O0=V0+K*8*4+128;
    static constexpr uint32_t I0=O0+K*5*4+160,G0=I0+K*2*4,M0=G0+128*4;
    static constexpr uint32_t N0=S0+256+((M0+K-S0-256+511)/512)*512;
    static_assert(N0+K*4<=196352,"division numerator exceeds usable UB");
    static constexpr uint32_t MASK_WORDS=((3*K+255)/256)*8;
    static constexpr uint32_t CMP0=I0+K*4+3*MASK_WORDS*4;
    static_assert(CMP0+K/8<=I0+K*2*4,"wrap mask exceeds metadata storage");
    StaticUBBuffer<CMP0,K/8> wrapMaskBuf;
    static constexpr uint32_t PATTERN_WORDS=((MASK_WORDS+16+23)/24)*24;
    static constexpr uint32_t SEED_WORDS=24;
    static constexpr uint32_t PRIOR_MASK_WORDS=((5*(K+7)+255)/256)*8;
    static constexpr uint32_t PRIOR_PATTERN_WORDS=((PRIOR_MASK_WORDS+32+39)/40)*40;
    static constexpr uint32_t PRIOR_SEED_WORDS=40;
    static constexpr uint32_t PRIOR_STREAMS=PATTERN_WORDS+PRIOR_PATTERN_WORDS;
    static_assert(PRIOR_STREAMS+5*PRIOR_MASK_WORDS<=2*K,
        "independent prior compaction masks exceed index storage");
    static_assert(K*4+3*MASK_WORDS*4<=K*8,"mask metadata exceeds index storage");
    static_assert(PRIOR_STREAMS+5*PRIOR_MASK_WORDS<=(CMP0-I0)/4,
        "periodic patterns overlap wrap scratch");
    StaticUBBuffer<P0,P*5*4> priorBuf;
    StaticUBBuffer<S0,K*7*4> sigBuf;
    StaticUBBuffer<S0+K*7*4,K*4> rawSuffixBuf;
    StaticUBBuffer<V0,K*8*4> workBuf;
    StaticUBBuffer<O0,K*5*4+160> outBuf;
    StaticUBBuffer<I0,K*2*4> indexBuf;
    StaticUBBuffer<G0,64*4> onesBuf;
    StaticUBBuffer<M0,K> outputMaskBuf;
    GlobalTensor<T> raw,out;
    GlobalTensor<float> prior;
    FireCandidateDecodeFusionTilingData d;
    uint32_t beginRow,endRow;
    uint64_t outSrc[16],outDst[16];

    __aicore__ inline LocalTensor<T> RawInput() {
        auto result=sigBuf.template Get<T>();
        if constexpr(sizeof(T)==2)result=result[8*K];
        return result;
    }
    __aicore__ inline void StageNextRaw(uint32_t row,uint32_t count) {
        StageDenseRaw<T,K,false>(outBuf.template Get<T>(),rawSuffixBuf.template Get<T>(),raw,row,count);
    }
public:
    __aicore__ inline void PrepareNextInput(uint32_t count) {
        auto next=sigBuf.template Get<float>();
        if constexpr(sizeof(T)==2) {
            auto bits=outBuf.template Get<int16_t>();
            Adds(bits,bits,static_cast<int16_t>(-32768),count*6);PipeBarrier<PIPE_V>();
            Cast(next,outBuf.template Get<T>(),RoundMode::CAST_NONE,count*6);
            PipeBarrier<PIPE_V>();
        } else {
            CopySplitRaw<K>(next,outBuf.template Get<int32_t>(),rawSuffixBuf.template Get<int32_t>(),count);
        }
    }
private:
    __aicore__ inline void SetPriorHeads(uint64_t keep) {
        SetDensePriorHeads<PRIOR_MASK_WORDS>(indexBuf.template Get<uint64_t>()[PRIOR_STREAMS/2],keep);
    }
    template<bool Restore=true,bool PreserveCounter=false>
    __aicore__ inline void UnpackContiguousPrior(LocalTensor<float>& destination,
        LocalTensor<float>& packed,uint32_t count,uint32_t prefix=0) {
        if(prefix) {
            // Each channel owns its mask head, so clearing one head cannot
            // remove valid middle rows from another channel's mask stream.
            SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
            SetPriorHeads(~uint64_t(0)<<(5*prefix));
            SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }
        static_assert(!(Restore && PreserveCounter),"counter preservation needs immutable one-tile heads");
        if constexpr(!PreserveCounter) SetMaskCount();
        SetVectorMask<float,MaskMode::COUNTER>(5*(count+prefix));
        auto patterns=indexBuf.template Get<float>()[PRIOR_STREAMS];
#pragma unroll
        for(uint32_t c=0;c<5;++c) {
            GatherMask(destination[c*P],packed,patterns[c*PRIOR_MASK_WORDS],uint8_t(0),
                GatherMaskParams{1,1,8,0});
        }
        PipeBarrier<PIPE_V>();
        if constexpr(!PreserveCounter) {ResetMask();SetMaskNorm();}
        if constexpr(Restore) {
            if(prefix) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
                SetPriorHeads(~uint64_t(0));
                SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
            }
        }
    }
    __aicore__ inline void SplitPlanes(LocalTensor<float>& values,LocalTensor<float>& even,
        LocalTensor<float>& odd,uint32_t count) {
        SplitPackedEvenOdd(values,even,odd,count);
        auto patterns=indexBuf.template Get<float>();
#pragma unroll
        for(uint32_t phase=0;phase<2;++phase) {
            auto mask=patterns[phase==0?0:(phase==1?16:8)];
            GatherMask(values[(2*phase)*P],even,mask,uint8_t(0),GatherMaskParams{1,1,8,0});
            GatherMask(values[(2*phase+1)*P],odd,mask,uint8_t(0),GatherMaskParams{1,1,8,0});
        }
        PipeBarrier<PIPE_V>();
        // Pair objectness and class confidence in the even/odd streams.
        // Only phase two of this product is consumed by the output.
        Mul<float,false>(even,even,odd,MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
        PipeBarrier<PIPE_V>();
        GatherMask(values[4*P],even,patterns[8],uint8_t(0),GatherMaskParams{1,1,8,0});
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline uint32_t PriorPrefix(uint32_t n,uint32_t count) {
        uint32_t first=d.n-n;
        return first<count?((first+7)/8*8-first):0;
    }
    __aicore__ inline void StagePrior(uint32_t n,uint32_t count) {
        auto packed=outBuf.template Get<float>();
        uint32_t first=d.n-n;
        if(first>=count)CopyDenseInput(packed,prior,n*5,count*5);
        else {
            uint32_t prefix=PriorPrefix(n,count),length=first+prefix;
            CopyDenseInput(packed,prior,(n-prefix)*5,length*5);
            CopyDenseInput(packed[length*5],prior,0,(count-first)*5);
        }
    }
    template<bool Restore=true,bool PreserveCounter=false>
    __aicore__ inline void UnpackPrior(LocalTensor<float>& p,LocalTensor<float>& packed,
        LocalTensor<uint32_t>& indices,uint32_t n,uint32_t count) {
        uint32_t prefix=PriorPrefix(n,count);
        UnpackContiguousPrior<Restore,PreserveCounter>(p,packed,count,prefix);
    }
    template<bool One>
    __aicore__ inline __attribute__((always_inline)) void InitPatterns() {
        constexpr uint32_t bits[3]={0x49249249U,0x92492492U,0x24924924U};
                constexpr uint64_t rawLanes[3]={0x249249ULL,0x492492ULL,0x924924ULL};
                constexpr uint32_t priorBits[5]={0x42108421U,0x10842108U,
                    0x84210842U,0x21084210U,0x08421084U};
                constexpr uint64_t priorLanes[5]={0x0842108421ULL,0x1084210842ULL,
                    0x2108421084ULL,0x4210842108ULL,0x8421084210ULL};
                auto rawSeeds=indexBuf.template Get<int32_t>();
                auto priorSeeds=indexBuf.template Get<int32_t>()[PATTERN_WORDS];
                // A repeat advances by one full period group: 24 or 40 words.
                // The normal masks select only those first 24/40 lanes.
                SetMaskNorm();
        #pragma unroll
                for(uint32_t c=0;c<3;++c) {
                    SetVectorMask<int32_t,MaskMode::NORMAL>(0,rawLanes[c]);
                    Duplicate<int32_t,false>(rawSeeds,static_cast<int32_t>(bits[c]),
                        MASK_PLACEHOLDER,PATTERN_WORDS/24,1,3);
                }
        #pragma unroll
                for(uint32_t c=0;c<5;++c) {
                    SetVectorMask<int32_t,MaskMode::NORMAL>(0,priorLanes[c]);
                    Duplicate<int32_t,false>(priorSeeds,static_cast<int32_t>(priorBits[c]),
                        MASK_PLACEHOLDER,PRIOR_PATTERN_WORDS/40,1,5);
                }
                PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
                SetMaskCount();
                SetVectorMask<int32_t,MaskMode::COUNTER>(PRIOR_MASK_WORDS);
                auto priorWords=indexBuf.template Get<int32_t>()[PATTERN_WORDS];
                auto priorStreams=indexBuf.template Get<int32_t>()[PRIOR_STREAMS];
        #pragma unroll
                for(uint32_t c=0;c<5;++c) {
                    uint32_t wordOffset=c==0?0:8*(5-c);
                    Adds<int32_t,false>(priorStreams[c*PRIOR_MASK_WORDS],priorWords[wordOffset],
                        int32_t(0),MASK_PLACEHOLDER,1,{1,1,8,8});
                }
                PipeBarrier<PIPE_V>();
                SetVectorMask<float,MaskMode::COUNTER>(64);
                Duplicate<float,false>(onesBuf.template Get<float>(),2.f,MASK_PLACEHOLDER,1,1,8);
                SetVectorMask<int32_t,MaskMode::COUNTER>(K/4);
                Duplicate<int32_t,false>(outputMaskBuf.template Get<int32_t>(),static_cast<int32_t>(0x1f1f1f1f),MASK_PLACEHOLDER,1,1,8);
                PipeBarrier<PIPE_V>();
                if constexpr(!One) {ResetMask();SetMaskNorm();}
    }
    __aicore__ inline __attribute__((always_inline)) void RefreshMaskPrior(LocalTensor<float> p,LocalTensor<float> o,LocalTensor<float> v,LocalTensor<uint32_t> idx,uint32_t row,uint32_t n,uint32_t count,uint32_t& cachedN,uint32_t& cachedCount,bool& needsRepair) {
        if(row!=beginRow) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            StagePrior(n,count);
        }
        if(row==beginRow)WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        else {SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);}
        UnpackPrior(p,o,idx,n,count);
        // This predicate depends only on the cached priors and dtype.
        // The existing exact repair remains available for every tile.
        needsRepair=DirectStrideMax(p[2*P],v,count)>(sizeof(T)==2?4096.f:32.f);
        cachedN=n;cachedCount=count;
    }
    __aicore__ inline __attribute__((always_inline)) void PrefetchSuccessor(uint32_t row,uint32_t& nextRow,uint32_t& nextCount) {
        if(nextRow<endRow)nextCount=endRow-nextRow<K?endRow-nextRow:K;
        if(Early && nextCount) {
            // O was consumed by SplitPlanes. Suffix is a distinct buffer
            // beyond the active S planes; current decode never touches it.
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            StageNextRaw(nextRow,nextCount);
        }
    }
    __aicore__ inline __attribute__((always_inline)) void EvaluateTileSigmoid(LocalTensor<float> s,LocalTensor<float> v,uint32_t count) {
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(count*6);
        if constexpr(sizeof(T)==4) {Muls<float,false>(v,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();}
        Exp<float,false>(s,sizeof(T)==2?s:v,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(v,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Div<float,false>(s,onesBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
        PipeBarrier<PIPE_V>();
    }
public:
    template<bool One=false>
    __aicore__ inline void Init(GM_ADDR r,GM_ADDR p,GM_ADDR o,FireCandidateDecodeFusionTilingData t) {
        d=t;InitSocState();ResetMask();SetMaskNorm();
        raw.SetGlobalBuffer((__gm__ T*)r);prior.SetGlobalBuffer((__gm__ float*)p);out.SetGlobalBuffer((__gm__ T*)o);
        auto values=sigBuf.template Get<float>();auto v=workBuf.template Get<float>();auto temp=outBuf.template Get<float>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        constexpr uint32_t rowAlign=sizeof(T)==2?16:8;
        const uint32_t owner=GetBlockIdx();
        const uint32_t firstUnit=owner*d.unitsPerWorker+
            (owner<d.extraUnits?owner:d.extraUnits);
        const uint32_t ownerUnits=d.unitsPerWorker+(owner<d.extraUnits);
        beginRow=firstUnit*rowAlign;
        endRow=beginRow+ownerUnits*rowAlign;
        if(endRow>d.totalRows)endRow=d.totalRows;
        uint32_t firstRow=beginRow,firstN=firstRow%d.n;
        uint32_t firstCount=endRow-firstRow<K?endRow-firstRow:K;
        StagePrior(firstN,firstCount);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        CopyDenseInput(RawInput(),raw,firstRow*6,firstCount*6);
        InitDenseOutputContext<P>(values,v,outSrc,outDst);

        InitPatterns<One>();
}
    __aicore__ inline void ProcessOne() {
        auto s=sigBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto p=priorBuf.template Get<float>();auto o=outBuf.template Get<float>();
        auto idx=indexBuf.template Get<uint32_t>();auto r=RawInput();
        uint32_t row=beginRow,count=endRow-beginRow;
        if(!count) {ResetMask();SetMaskNorm();return;}
        uint32_t n=row%d.n;uint64_t reserved=0;
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // This worker has no subsequent prior compaction.
        // Init and this compaction share counter mode. The following
        // stride guard establishes normal mode before its first reduction.
        UnpackPrior<false,true>(p,o,idx,n,count);
        static_assert(G0+128*4<=M0,"one-tile guard exceeds constant gap");
        LocalTensor<float> guard(TPosition::VECCALC,G0+64*4,64);
        BeginOneStrideMax(p[2*P],guard,count);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if constexpr(sizeof(T)==2) {
            auto bits=r.template ReinterpretCast<int16_t>();
            Adds(bits,bits,static_cast<int16_t>(-32768),count*6);PipeBarrier<PIPE_V>();
            Cast(s,r,RoundMode::CAST_NONE,count*6);PipeBarrier<PIPE_V>();}
        EvaluateTileSigmoid(s,v,count);
        SplitPlanes(s,o,v,count);
        bool needsRepair=FinishOneStrideMax(guard,count)>(sizeof(T)==2?4096.f:32.f);
        if(__builtin_expect(needsRepair,false)) {
            RepairScaledDenseCold<T,K,S0,P0,V0>((GM_ADDR)raw.GetPhyAddr(),count,row*6);
            set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
            set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
        }
        DecodeMaskedGeometry<P,true>(s,v,p,count);
        EmitDenseOutput(o,v,outputMaskBuf.template Get<uint32_t>(),outBuf.template Get<T>(),out,row,count,reserved);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process() {
        auto s=sigBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto p=priorBuf.template Get<float>();auto o=outBuf.template Get<float>();
        auto idx=indexBuf.template Get<uint32_t>();auto r=RawInput();
        uint32_t cachedN=d.n,cachedCount=0;uint64_t reserved=0;
        bool needsRepair=true;
        for(uint32_t row=beginRow;row<endRow;row+=K) {
            uint32_t n=row%d.n;
            uint32_t count=endRow-row<K?endRow-row:K;
            const bool reusePrior=n==cachedN && count==cachedCount;
            if(!reusePrior) RefreshMaskPrior(p,o,v,idx,row,n,count,cachedN,cachedCount,needsRepair);
            if(!Early || row==beginRow) {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if constexpr(sizeof(T)==2) {
            auto bits=r.template ReinterpretCast<int16_t>();
            Adds(bits,bits,static_cast<int16_t>(-32768),count*6);PipeBarrier<PIPE_V>();
            Cast(s,r,RoundMode::CAST_NONE,count*6);PipeBarrier<PIPE_V>();}
            }
            EvaluateTileSigmoid(s,v,count);
            // The next sigmoid uses S only; O remains owned by the previous
            // output DMA until its first new consumer, the even compaction.
            if(row!=beginRow && reusePrior)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SplitPlanes(s,o,v,count);
            if(__builtin_expect(needsRepair,false)) {
                RepairScaledDenseCold<T,K,S0,P0,V0>((GM_ADDR)raw.GetPhyAddr(),count,row*6);
                set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
                set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
            }
            uint32_t nextRow=row+K,nextCount=0;
            PrefetchSuccessor(row,nextRow,nextCount);
        DecodeMaskedGeometry<P,true>(s,v,p,count);
            CompleteDenseInput<Early>(*this,r,raw,nextRow,nextCount);
            EmitDenseOutput(o,v,outputMaskBuf.template Get<uint32_t>(),outBuf.template Get<T>(),out,row,count,reserved);
            if(row+K<endRow) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
};
