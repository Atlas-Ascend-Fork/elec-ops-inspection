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

template<class T,uint32_t K,bool Early=true> class MaskDecode {
    static constexpr uint32_t P=FireDensePitchPolicy<T,K>::Value;
    static_assert(6*P<=7*K,"dense planes overlap raw prefetch suffix");
    static constexpr bool FullRawPrefetch=sizeof(T)==4 && K>1024;
    static constexpr uint32_t OPlanes=FullRawPrefetch?6:5;
    //512-byte arena extents preserve the original source bank offsets. The
    // V arena also contains the padded16-row final transpose for K%16!=0.
    static constexpr uint32_t SSpan=FullRawPrefetch?(K*8*4+511)/512*512:K*8*4;
    static constexpr uint32_t OSpan=FullRawPrefetch?(K*OPlanes*4+511)/512*512:K*OPlanes*4;
    static constexpr uint32_t ISpan=FullRawPrefetch?(K*2*4+511)/512*512:K*2*4;
    static constexpr uint32_t P0=0,S0=P*5*4,V0=S0+SSpan+128,O0=V0+SSpan+128;
    static constexpr uint32_t I0=O0+OSpan+128,G0=I0+ISpan,M0=G0+128*4;
    // This kernel uses only the64-element numerator at G0. No K-element
    // numerator tensor exists; bound the buffers that are actually accessed.
    static_assert(M0+K<=196352,"large raw staging exceeds usable UB");
    static constexpr uint32_t MASK_WORDS=((3*K+255)/256)*8;
    static constexpr uint32_t PATTERN_WORDS=MASK_WORDS+16;
    static constexpr uint32_t SEED_WORDS=24;
    static constexpr uint32_t PRIOR_MASK_WORDS=((5*(K+7)+255)/256)*8;
    static constexpr uint32_t PRIOR_PATTERN_WORDS=PRIOR_MASK_WORDS+32;
    static constexpr uint32_t PRIOR_SEED_WORDS=40;
    static constexpr uint32_t PRIOR_STREAMS=PATTERN_WORDS+PRIOR_PATTERN_WORDS;
    static_assert(PRIOR_STREAMS+5*PRIOR_MASK_WORDS<=2*K,
        "independent prior compaction masks exceed index storage");
    static_assert(K*4+3*MASK_WORDS*4<=K*8,"mask metadata exceeds index storage");
    StaticUBBuffer<P0,P*5*4> priorBuf;
    StaticUBBuffer<S0,K*7*4> sigBuf;
    StaticUBBuffer<S0+K*7*4,K*4> rawSuffixBuf;
    StaticUBBuffer<V0,SSpan> workBuf;
    StaticUBBuffer<O0,OSpan> outBuf;
    StaticUBBuffer<I0,ISpan> indexBuf;
    StaticUBBuffer<G0,64*4> onesBuf;
    StaticUBBuffer<M0,K> outputMaskBuf;
    GlobalTensor<T> raw,out;
    GlobalTensor<float> prior;
    FireCandidateDecodeFusionTilingData d;
    uint32_t beginTile,endTile;
    uint64_t outSrc[16],outDst[16];

    __aicore__ inline LocalTensor<T> RawInput() {
        auto result=sigBuf.template Get<T>();
        if constexpr(sizeof(T)==2)result=result[8*K];
        return result;
    }
    // Only the large FP16 path moves bitwise channel separation before
    // widening. Sigmoid/decode/repair still operate entirely in FP32.
    static constexpr bool HalfPrepack=sizeof(T)==2 && K>1024;
    static constexpr uint32_t HQ=(P+15)/16*16;
    __aicore__ inline void UnpackHalfInput(LocalTensor<T> packed,uint32_t count) {
        auto even=sigBuf.template Get<T>();
        auto odd=even[3*K];
        LocalTensor<T> separated(TPosition::VECCALC,S0+16*K,8*K);
        static_assert(16*K+2*(5*HQ+K)<=32*K,"half plane staging exceeds S and suffix");
        SetMaskCount();SetVectorMask<T,MaskMode::COUNTER>(6*count);
        GatherMask(even,packed,uint8_t(1),GatherMaskParams{1,1,8,0});
        GatherMask(odd,packed,uint8_t(2),GatherMaskParams{1,1,8,0});
        PipeBarrier<PIPE_V>();
        SetVectorMask<T,MaskMode::COUNTER>(3*count);
        auto patterns=indexBuf.template Get<T>();
#pragma unroll
        for(uint32_t phase=0;phase<3;++phase) {
            auto mask=patterns[phase==0?0:(phase==1?32:16)];
            GatherMask(separated[(2*phase)*HQ],even,mask,uint8_t(0),GatherMaskParams{1,1,8,0});
            GatherMask(separated[(2*phase+1)*HQ],odd,mask,uint8_t(0),GatherMaskParams{1,1,8,0});
        }
        PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
        auto values=sigBuf.template Get<float>();
        static_assert(!HalfPrepack || HQ==P,"half plane pitch must match widening");
        // Staging starts above the complete widening destination. The same
        // per-element cast now traverses the aligned planes in one command.
        Cast(values,separated,RoundMode::CAST_NONE,5*P+count);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void StageNextRaw(uint32_t row,uint32_t count) {
        StageDenseRaw<T,K,FullRawPrefetch>(outBuf.template Get<T>(),rawSuffixBuf.template Get<T>(),raw,row,count);
    }
public:
    __aicore__ inline void PrepareNextInput(uint32_t count) {
        auto next=sigBuf.template Get<float>();
        if constexpr(sizeof(T)==2) {
            if constexpr(HalfPrepack) {
                UnpackHalfInput(outBuf.template Get<T>(),count);
            } else {
                Cast(next,outBuf.template Get<T>(),RoundMode::CAST_NONE,count*6);
                PipeBarrier<PIPE_V>();
            }
        } else if constexpr(FullRawPrefetch) {
            Adds(next.template ReinterpretCast<int32_t>(),outBuf.template Get<int32_t>(),int32_t(0),count*6);
            PipeBarrier<PIPE_V>();
        } else {
            CopySplitRaw<K>(next,outBuf.template Get<int32_t>(),rawSuffixBuf.template Get<int32_t>(),count);
        }
    }
private:
    __aicore__ inline void SetPriorHeads(uint64_t keep) {
        SetDensePriorHeads<PRIOR_MASK_WORDS>(indexBuf.template Get<uint64_t>()[PRIOR_STREAMS/2],keep);
    }
    __aicore__ inline void UnpackContiguousPrior(LocalTensor<float>& destination,
        LocalTensor<float>& packed,uint32_t count) {
        // Each MaskDecode tile begins at a complete candidate row. Packed
        // prior DMA has no left prefix, so large masks remain immutable.
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(5*count);
        auto patterns=indexBuf.template Get<float>();
#pragma unroll
        for(uint32_t c=0;c<5;++c) {
            const uint32_t offset=K>1024?
                PATTERN_WORDS+(c==0?0:8*(5-c)):PRIOR_STREAMS+c*PRIOR_MASK_WORDS;
            GatherMask(destination[c*P],packed,patterns[offset],uint8_t(0),
                GatherMaskParams{1,1,8,0});
        }
        PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
    }
    __aicore__ inline void SplitPlanes(LocalTensor<float>& values,LocalTensor<float>& even,
        LocalTensor<float>& odd,uint32_t count) {
        SplitPackedEvenOdd(values,even,odd,count);
        auto patterns=indexBuf.template Get<float>();
#pragma unroll
        for(uint32_t phase=0;phase<(FullRawPrefetch?2:3);++phase) {
            auto mask=patterns[phase==0?0:(phase==1?16:8)];
            GatherMask(values[(2*phase)*P],even,mask,uint8_t(0),GatherMaskParams{1,1,8,0});
            GatherMask(values[(2*phase+1)*P],odd,mask,uint8_t(0),GatherMaskParams{1,1,8,0});
        }
        if constexpr(FullRawPrefetch) {
            PipeBarrier<PIPE_V>();
            // Phase2 pairs the two original score lanes. The same FP32
            // multiply commutes with their bitwise permutation.
            Mul<float,false>(even,even,odd,MASK_PLACEHOLDER,1,{1,1,1,8,8,8});
            PipeBarrier<PIPE_V>();
            GatherMask(values[4*P],even,patterns[8],uint8_t(0),GatherMaskParams{1,1,8,0});
        }
        PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
    }
    __aicore__ inline __attribute__((always_inline)) void InitPatterns() {
        constexpr uint32_t bits[3]={0x49249249U,0x92492492U,0x24924924U};
                auto masks=indexBuf.template Get<uint64_t>();
        #pragma unroll
                for(uint32_t j=0;j<SEED_WORDS/2;++j) {
                    uint64_t value=uint64_t(bits[(2*j)%3])|
                        (uint64_t(bits[(2*j+1)%3])<<32);
                    masks.SetValue(j,value);
                }
                constexpr uint32_t priorBits[5]={0x42108421U,0x10842108U,
                    0x84210842U,0x21084210U,0x08421084U};
                auto priorMasks=indexBuf.template Get<uint64_t>()[PATTERN_WORDS/2];
        #pragma unroll
                for(uint32_t j=0;j<PRIOR_SEED_WORDS/2;++j) {
                    uint64_t value=uint64_t(priorBits[(2*j)%5])|
                        (uint64_t(priorBits[(2*j+1)%5])<<32);
                    priorMasks.SetValue(j,value);
                }
                SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);

                auto maskWords=indexBuf.template Get<int32_t>();
                for(uint32_t span=SEED_WORDS;span<PATTERN_WORDS;span*=2) {
                    uint32_t length=PATTERN_WORDS-span<span?PATTERN_WORDS-span:span;
                    Adds(maskWords[span],maskWords,int32_t(0),length);
                    PipeBarrier<PIPE_V>();
                }
                auto priorWords=indexBuf.template Get<int32_t>()[PATTERN_WORDS];
                for(uint32_t span=PRIOR_SEED_WORDS;span<PRIOR_PATTERN_WORDS;span*=2) {
                    uint32_t length=PRIOR_PATTERN_WORDS-span<span?PRIOR_PATTERN_WORDS-span:span;
                    Adds(priorWords[span],priorWords,int32_t(0),length);
                    PipeBarrier<PIPE_V>();
                }
                if constexpr(K<=1024) {
                    auto priorStreams=indexBuf.template Get<int32_t>()[PRIOR_STREAMS];
        #pragma unroll
                    for(uint32_t c=0;c<5;++c) {
                        uint32_t wordOffset=c==0?0:8*(5-c);
                        Adds(priorStreams[c*PRIOR_MASK_WORDS],priorWords[wordOffset],
                            int32_t(0),PRIOR_MASK_WORDS);
                    }
                    PipeBarrier<PIPE_V>();
                }
                Duplicate(onesBuf.template Get<float>(),2.f,64);
                Duplicate(outputMaskBuf.template Get<int32_t>(),static_cast<int32_t>(0x1f1f1f1f),K/4);
                PipeBarrier<PIPE_V>();
    }
    __aicore__ inline __attribute__((always_inline)) void RefreshMaskPrior(LocalTensor<float> p,LocalTensor<float> o,LocalTensor<float> v,uint32_t tile,uint32_t n,uint32_t count,uint32_t& cachedN,bool& needsRepair) {
        if(tile!=beginTile) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            CopyDenseInput(o,prior,n*5,count*5);
        }
        if(tile==beginTile)WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        else {SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);}
        UnpackContiguousPrior(p,o,count);
        // This predicate depends only on the cached priors and dtype.
        // The existing exact repair remains available for every tile.
        needsRepair=DirectStrideMax(p[2*P],v,count)>(sizeof(T)==2?4096.f:32.f);
        cachedN=n;
    }
    __aicore__ inline __attribute__((always_inline)) void PrefetchSuccessor(uint32_t tile,uint32_t b,uint32_t n,uint32_t& nextRow,uint32_t& nextCount) {
        if(tile+1<endTile) {
            uint32_t nextB=b+1,nextN=n;
            if(nextB==d.batch){nextB=0;nextN+=K;}
            nextRow=nextB*d.n+nextN;
            nextCount=d.n-nextN<K?d.n-nextN:K;
        }
        if(Early && nextCount) {
            // O was consumed by SplitPlanes. Suffix is a distinct buffer
            // beyond the active S planes; current decode never touches it.
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            StageNextRaw(nextRow,nextCount);
        }
    }
    __aicore__ inline __attribute__((always_inline)) void EvaluateTileSigmoid(LocalTensor<float> s,LocalTensor<float> v,uint32_t count) {
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(HalfPrepack?5*P+count:count*6);
        Muls<float,false>(v,s,-1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Exp<float,false>(s,v,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Adds<float,false>(v,s,1.f,MASK_PLACEHOLDER,1,{1,1,8,8});PipeBarrier<PIPE_V>();
        Div<float,false>(s,onesBuf.template Get<float>(),v,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});
        PipeBarrier<PIPE_V>();ResetMask();SetMaskNorm();
    }
public:
    __aicore__ inline void Init(GM_ADDR r,GM_ADDR p,GM_ADDR o,FireCandidateDecodeFusionTilingData t) {
        d=t;InitSocState();ResetMask();SetMaskNorm();
        raw.SetGlobalBuffer((__gm__ T*)r);prior.SetGlobalBuffer((__gm__ float*)p);out.SetGlobalBuffer((__gm__ T*)o);
        auto values=sigBuf.template Get<float>();auto v=workBuf.template Get<float>();auto temp=outBuf.template Get<float>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        uint32_t tiles=(d.n+K-1)/K,total=tiles*d.batch;
        beginTile=total*GetBlockIdx()/d.blocks;endTile=total*(GetBlockIdx()+1)/d.blocks;
        uint32_t firstB=beginTile%d.batch,firstN=(beginTile/d.batch)*K;
        uint32_t firstCount=d.n-firstN<K?d.n-firstN:K;
        CopyDenseInput(temp,prior,firstN*5,firstCount*5);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        CopyDenseInput(RawInput(),raw,(firstB*d.n+firstN)*6,firstCount*6);
        InitDenseOutputContext<P>(values,v,outSrc,outDst);

        InitPatterns();
}
    __aicore__ inline void Process() {
        auto s=sigBuf.template Get<float>();auto v=workBuf.template Get<float>();
        auto p=priorBuf.template Get<float>();auto o=outBuf.template Get<float>();
        auto idx=indexBuf.template Get<uint32_t>();auto r=RawInput();
        uint32_t tiles=(d.n+K-1)/K,cachedN=d.n;uint64_t reserved=0;
        bool needsRepair=true;
        uint32_t b=beginTile%d.batch,n=(beginTile/d.batch)*K;
        for(uint32_t tile=beginTile;tile<endTile;++tile) {
            uint32_t count=d.n-n<K?d.n-n:K;
            const bool reusePrior=n==cachedN;
            if(!reusePrior) RefreshMaskPrior(p,o,v,tile,n,count,cachedN,needsRepair);
            if(!Early || tile==beginTile) {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if constexpr(sizeof(T)==2) {
                    if constexpr(HalfPrepack)UnpackHalfInput(r,count);
                    else {Cast(s,r,RoundMode::CAST_NONE,count*6);PipeBarrier<PIPE_V>();}
                }
            }
            EvaluateTileSigmoid(s,v,count);
            // The next sigmoid uses S only; O remains owned by the previous
            // output DMA until its first new consumer, the even compaction.
            if(tile!=beginTile && reusePrior)WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            if constexpr(!HalfPrepack)SplitPlanes(s,o,v,count);
            if(__builtin_expect(needsRepair,false)) {
                RepairScaledDenseCold<T,K,S0,P0,V0>((GM_ADDR)raw.GetPhyAddr(),count,(b*d.n+n)*6);
                set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
                set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
            }
            uint32_t nextRow=0,nextCount=0;
            PrefetchSuccessor(tile,b,n,nextRow,nextCount);
        DecodeMaskedGeometry<P,FullRawPrefetch>(s,v,p,count);
            CompleteDenseInput<Early>(*this,r,raw,nextRow,nextCount);
            EmitDenseOutput(o,v,outputMaskBuf.template Get<uint32_t>(),outBuf.template Get<T>(),out,(b*d.n+n),count,reserved);
            if(tile+1<endTile) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            if(++b==d.batch){b=0;n+=K;}
        }
        PipeBarrier<PIPE_ALL>();
    }
};
