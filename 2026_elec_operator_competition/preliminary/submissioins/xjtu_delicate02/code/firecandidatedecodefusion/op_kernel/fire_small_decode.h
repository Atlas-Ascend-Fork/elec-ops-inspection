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

template<class T, uint32_t K> class PadSmallDecode {
    static constexpr uint32_t P=K+8,Q=K+16;
    StaticUBBuffer<0,8*P*4> priorBuf;
    StaticUBBuffer<8*P*4,8*P*4> planeBuf;
    StaticUBBuffer<16*P*4,16*K*4> rawBuf;
    StaticUBBuffer<16*P*4+16*K*4,16*K*2> halfBuf;
    StaticUBBuffer<16*P*4+24*K*4,8*K*4> priorPackedBuf;
    StaticUBBuffer<16*P*4+32*K*4,8*K*4> outBuf;
    StaticUBBuffer<16*P*4+40*K*4,64*4> onesBuf;
    StaticUBBuffer<16*P*4+40*K*4+64*4,K> maskBuf;
    StaticUBBuffer<16*P*4+40*K*4+64*4+K,K*2*4> indexBuf;
    GlobalTensor<T> raw,out;
    GlobalTensor<float> prior;
    FireCandidateDecodeFusionTilingData d;
    uint64_t rawSrc[16],rawDst[16],priorSrc[16],priorDst[16],outSrc[16],outDst[16];
    template<bool EntryInitialized>
    __aicore__ inline __attribute__((always_inline)) void InitState(GM_ADDR r,GM_ADDR p,GM_ADDR o,FireCandidateDecodeFusionTilingData t) {
        d=t;
        #if __NPU_ARCH__ == 2201
                if constexpr(!EntryInitialized) {InitSocState();ResetMask();SetMaskNorm();}
        #else
                InitSocState();ResetMask();SetMaskNorm();
        #endif
                raw.SetGlobalBuffer((__gm__ T*)r);prior.SetGlobalBuffer((__gm__ float*)p);out.SetGlobalBuffer((__gm__ T*)o);
    }
    __aicore__ inline __attribute__((always_inline)) void SetSmallOutputContext() {
        set_va_reg_sb(VA0,outDst);set_va_reg_sb(VA1,outDst+8);
        set_va_reg_sb(VA2,outSrc);set_va_reg_sb(VA3,outSrc+8);
    }
public:
    template<bool IndexPrefetched>
    __aicore__ inline __attribute__((always_inline)) void InitTables() {
        auto s=rawBuf.template Get<float>();auto v=planeBuf.template Get<float>();auto pp=priorPackedBuf.template Get<float>();auto pr=priorBuf.template Get<float>();
        constexpr uint32_t row=sizeof(T)==2?16:8;
        for(uint32_t c=0;c<8;++c) {
            if constexpr(sizeof(T)==2) {
                auto hs=halfBuf.template Get<T>();auto hd=rawBuf.template Get<T>();
                rawSrc[c]=reinterpret_cast<uint64_t>(hs[c*16].GetPhyAddr());rawSrc[c+8]=reinterpret_cast<uint64_t>(hs[(c+8)*16].GetPhyAddr());
                rawDst[c]=reinterpret_cast<uint64_t>(hd[c*Q].GetPhyAddr());rawDst[c+8]=reinterpret_cast<uint64_t>(hd[(c+8)*Q].GetPhyAddr());
            } else {
                rawSrc[c]=reinterpret_cast<uint64_t>(s[c*row].GetPhyAddr());rawSrc[c+8]=reinterpret_cast<uint64_t>(s[(c+8)*row].GetPhyAddr());
                rawDst[2*c]=reinterpret_cast<uint64_t>(v[c*P].GetPhyAddr());rawDst[2*c+1]=reinterpret_cast<uint64_t>(v[c*P+8].GetPhyAddr());
            }
            priorSrc[c]=reinterpret_cast<uint64_t>(pp[c*8].GetPhyAddr());priorSrc[c+8]=reinterpret_cast<uint64_t>(pp[(c+8)*8].GetPhyAddr());
            priorDst[2*c]=reinterpret_cast<uint64_t>(pr[c*P].GetPhyAddr());priorDst[2*c+1]=reinterpret_cast<uint64_t>(pr[c*P+8].GetPhyAddr());
            uint32_t channel=c<5?c:0;
            outSrc[c]=reinterpret_cast<uint64_t>(v[channel*P].GetPhyAddr());outSrc[c+8]=reinterpret_cast<uint64_t>(v[channel*P+8].GetPhyAddr());
            outDst[2*c]=reinterpret_cast<uint64_t>(s[c*8].GetPhyAddr());outDst[2*c+1]=reinterpret_cast<uint64_t>(s[(c+8)*8].GetPhyAddr());
        }
        if constexpr(!IndexPrefetched) {
        auto idx=indexBuf.template Get<int32_t>();
        for(uint32_t j=0;j<8;++j) {idx.SetValue(j,static_cast<int32_t>(j));}
        SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
        for(uint32_t span=8;span<K;span*=2) {Adds(idx[span],idx,static_cast<int32_t>(span),span);PipeBarrier<PIPE_V>();}
        Muls(idx[K],idx,static_cast<int32_t>(20),K);PipeBarrier<PIPE_V>();
        Muls(idx,idx,static_cast<int32_t>(24),K);PipeBarrier<PIPE_V>();
        } else static_assert(K<=128,"prefetched index capacity");
        Duplicate(planeBuf.template Get<float>(),0.f,8*P);PipeBarrier<PIPE_V>();
        Duplicate(onesBuf.template Get<float>(),1.f,64);PipeBarrier<PIPE_V>();
        Duplicate(maskBuf.template Get<int32_t>(),static_cast<int32_t>(0x1f1f1f1f),K/4);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline __attribute__((always_inline)) void EvaluateSingleSigmoid(LocalTensor<float> s,uint32_t count) {
        if constexpr(sizeof(T)==2) {
            auto bits=halfBuf.template Get<int16_t>();
            Adds(bits,bits,static_cast<int16_t>(-32768),count*6);PipeBarrier<PIPE_V>();
            Cast(s,halfBuf.template Get<T>(),RoundMode::CAST_NONE,count*6);PipeBarrier<PIPE_V>();
        } else {Muls(s,s,-1.f,count*6);PipeBarrier<PIPE_V>();}
        Exp(s,s,count*6);PipeBarrier<PIPE_V>();Adds(s,s,1.f,count*6);PipeBarrier<PIPE_V>();
        SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(count*6);
        Div<float,false>(s,onesBuf.template Get<float>(),s,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
    }
    template<bool Compact>
    __aicore__ inline __attribute__((always_inline)) void DecodeSmallGeometry(LocalTensor<float> v,LocalTensor<float> p,uint32_t count) {
        for(uint32_t off=0;off<count;off+=64) {
            uint64_t m=count-off<64?count-off:64;
            auto x=v[off];auto q=p[off];
            uint64_t rowMask = m == 64 ? ~uint64_t(0) : ((uint64_t(1) << m) - 1);
            SetVectorMask<float,MaskMode::NORMAL>(0,rowMask);
            if constexpr(Compact) {
                auto* xp=(__ubuf__ float*)x.GetPhyAddr();
                auto* qp=(__ubuf__ float*)q.GetPhyAddr();
                vmuls(xp,xp,2.f,4,1,1,P/8,P/8);pipe_barrier(PIPE_V);
                vadds(xp,xp,-.5f,2,1,1,P/8,P/8);
                vmul(xp+2*P,xp+2*P,xp+2*P,2,1,1,1,P/8,P/8,P/8);
                pipe_barrier(PIPE_V);
                vadd(xp,xp,qp,2,1,1,1,P/8,P/8,P/8);
                vmul(xp+2*P,xp+2*P,qp+3*P,2,1,1,1,P/8,P/8,P/8);
                pipe_barrier(PIPE_V);
                vmul(xp,xp,qp+2*P,4,1,1,1,P/8,P/8,0);
                vmul(xp+4*P,xp+4*P,xp+5*P,1,1,1,1,0,0,0);
                pipe_barrier(PIPE_V);
            } else {
            Muls<float,false>(x,x,2.f,m,4,{1,1,P/8,P/8});PipeBarrier<PIPE_V>();
            Adds<float,false>(x,x,-.5f,m,2,{1,1,P/8,P/8});PipeBarrier<PIPE_V>();
            Add<float,false>(x,x,q,m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
            Mul<float,false>(x[2*P],x[2*P],x[2*P],m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
            Mul<float,false>(x[2*P],x[2*P],q[3*P],m,2,{1,1,1,P/8,P/8,P/8});PipeBarrier<PIPE_V>();
            Mul<float,false>(x,x,q[2*P],m,4,{1,1,1,P/8,P/8,0});PipeBarrier<PIPE_V>();
            Mul<float,false>(x[4*P],x[4*P],x[5*P],m,1,{1,1,1,0,0,0});PipeBarrier<PIPE_V>();
            }
        }
    }
    __aicore__ inline __attribute__((always_inline)) void StoreSmallOutput(LocalTensor<float> s,LocalTensor<float> o,uint32_t b,uint32_t n,uint32_t count,uint64_t& reserved) {
        ResetMask(); SetMaskNorm();
        uint8_t repeats=static_cast<uint8_t>((count+15)/16);
        scatter_vnchwconv_b32(VA0,VA2,repeats,repeats==1?0:16,repeats==1?0:2);
        PipeBarrier<PIPE_V>();
        GatherMask(o,s,maskBuf.template Get<uint32_t>(),true,count*8,{1,1,8,0},reserved);PipeBarrier<PIPE_V>();
        if constexpr(sizeof(T)==2) {Cast(halfBuf.template Get<T>(),o,RoundMode::CAST_RINT,count*5);PipeBarrier<PIPE_V>();}
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        if constexpr(sizeof(T)==2) CopyDenseOutput(out,halfBuf.template Get<T>(),(b*d.n+n)*5,count*5);
        else CopyDenseOutput(out,outBuf.template Get<T>(),(b*d.n+n)*5,count*5);
    }
    template<bool StaticGuard,bool RawReady=false>
    __aicore__ inline __attribute__((always_inline)) bool CheckSingleGuard(LocalTensor<float> p,LocalTensor<float> s,LocalTensor<float> v,LocalTensor<float> o,LocalTensor<uint32_t> idx,uint32_t count) {
        bool needsRepair;
                    if constexpr(K<=64) {
                        ResetMask();SetMaskNorm();
                        WholeReduceMax(o,p[2*P],count,1,1,1,8,
                            ReduceOrder::ORDER_ONLY_VALUE);
                        PipeBarrier<PIPE_V>();
                        SetFlag<HardEvent::V_S>(EVENT_ID1);
                        if constexpr(!RawReady) GatherChannelPlanes<6>(v,s,idx,P,count);
                        WaitFlag<HardEvent::V_S>(EVENT_ID1);
                        needsRepair=!(*((__ubuf__ float*)o.GetPhyAddr())<=(sizeof(T)==2?4096.f:32.f));
                    } else {
                        GatherChannelPlanes<6>(v,s,idx,P,count);
                        needsRepair=SmallNeedsRepair<T,K,StaticGuard>(p,s,P,count);
                    }

        return needsRepair;
    }
    __aicore__ inline __attribute__((always_inline)) void RepairSmall(uint32_t b,uint32_t n,uint32_t count) {
        // Only primitive arguments cross the cold-call boundary.
        // The existing unscaled repair reconstructs these same views.
        RepairDenseCold<T,K,8*P*4,0,16*P*4>(
            (GM_ADDR)raw.GetPhyAddr(),count,(b*d.n+n)*6);
        SetSmallOutputContext();
    }

#if FIRE_COMPILED_SMALL_SINGLE
    __aicore__ inline static void PrefetchCompleteSingleInputs(GM_ADDR r,GM_ADDR p) {
        static_assert(sizeof(T)==4 && K==32,"complete Float compact input");
        GlobalTensor<T> input;input.SetGlobalBuffer((__gm__ T*)r);
        GlobalTensor<float> priors;priors.SetGlobalBuffer((__gm__ float*)p);
        LocalTensor<T> inputLocal(TPosition::VECCALC,16*P*4,16*K);
        LocalTensor<float> priorLocal(TPosition::VECCALC,16*P*4+24*K*4,8*K);
        CopyDenseInput(priorLocal,priors,0,K*5);
        CopyDenseInput(inputLocal,input,0,K*6);
    }
#endif
    __aicore__ inline static void PrefetchIndexMetadata(GM_ADDR addressMetadata) {
        static_assert(K<=128,"index metadata is a compact-capacity resource");
        GlobalTensor<uint32_t> words;words.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(fire_metadata::Table<0,K>::Address()));
        LocalTensor<uint32_t> destination(TPosition::VECCALC,
            16*P*4+40*K*4+64*4+K,2*K);
        DataCopy(destination,words,2*K);
    }
    template<bool IndexPrefetched=false,bool EntryInitialized=false>
    __aicore__ inline void Init(GM_ADDR r,GM_ADDR p,GM_ADDR o,FireCandidateDecodeFusionTilingData t) {
        InitState<EntryInitialized>(r,p,o,t);
        InitTables<IndexPrefetched>();
    }
    template<bool StaticGuard=false,bool CompactGeometry=false>
    __aicore__ inline void ProcessSingle() {
        auto s=rawBuf.template Get<float>();auto v=planeBuf.template Get<float>();auto p=priorBuf.template Get<float>();auto pp=priorPackedBuf.template Get<float>();auto o=outBuf.template Get<float>();
#if FIRE_COMPILED_SMALL_SINGLE
        const uint32_t b=0,n=0,count=K;
#else
        const uint32_t b=GetBlockIdx(),n=0,count=d.n<K?d.n:K;
#endif
        uint64_t reserved=0;
#if FIRE_COMPILED_SMALL_SINGLE
        if constexpr (!(sizeof(T)==4 && K==32)) {
#endif
            CopyDenseInput(pp,prior,n*5,count*5);
            if constexpr(sizeof(T)==2) {CopyDenseInput(halfBuf.template Get<T>(),raw,(b*d.n+n)*6,count*6);}
            else {CopyDenseInput(rawBuf.template Get<T>(),raw,(b*d.n+n)*6,count*6);}
#if FIRE_COMPILED_SMALL_SINGLE
        }
#endif
            {
                // Only scalar initialization writes these immutable tables.
                // This path has no other VA-register consumer or producer.
                SetSmallOutputContext();
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            EvaluateSingleSigmoid(s,count);
            auto idx=indexBuf.template Get<uint32_t>();
            GatherChannelPlanes<5>(p,pp,idx[K],P,count);
            const bool needsRepair=CheckSingleGuard<StaticGuard>(p,s,v,o,idx,count);
            if(__builtin_expect(needsRepair,false)) RepairSmall(b,n,count);
            DecodeSmallGeometry<(K<=64 || CompactGeometry)>(v,p,count);
            StoreSmallOutput(s,o,b,n,count,reserved);
        PipeBarrier<PIPE_ALL>();
    }
    template<bool IndexPrefetched=false,bool OwnerKnown=false,bool EntryInitialized=false>
    __aicore__ inline void InitWithInputPrefetch(GM_ADDR r,GM_ADDR p,GM_ADDR o,FireCandidateDecodeFusionTilingData t,int64_t owner=0) {
        InitState<EntryInitialized>(r,p,o,t);
        const uint32_t count=t.n<K?t.n:K;
        static_assert(!OwnerKnown || K==64,"known-owner input resource required");
        const int64_t inputOwner=OwnerKnown?owner:GetBlockIdx();
        CopyDenseInput(priorPackedBuf.template Get<float>(),prior,0,count*5);
        if constexpr(sizeof(T)==2) CopyDenseInput(halfBuf.template Get<T>(),raw,inputOwner*t.n*6,count*6);
        else CopyDenseInput(rawBuf.template Get<T>(),raw,inputOwner*t.n*6,count*6);
        InitTables<IndexPrefetched>();
    }
    template<bool PriorMaskForward=false,bool OwnerKnown=false>
    __aicore__ inline void ProcessSingleAfterInputPrefetch(int64_t owner=0) {
        auto s=rawBuf.template Get<float>();auto v=planeBuf.template Get<float>();auto p=priorBuf.template Get<float>();auto pp=priorPackedBuf.template Get<float>();auto o=outBuf.template Get<float>();
        static_assert(!OwnerKnown || K==64,"known-owner output resource required");
        const uint32_t b=static_cast<uint32_t>(OwnerKnown?owner:GetBlockIdx()),n=0,count=d.n<K?d.n:K;
        uint64_t reserved=0;
            {
                // Only scalar initialization writes these immutable tables.
                // This path has no other VA-register consumer or producer.
                SetSmallOutputContext();
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            EvaluateSingleSigmoid(s,count);
            auto idx=indexBuf.template Get<uint32_t>();
            GatherChannelPlanes<5,!(PriorMaskForward && K==64)>(p,pp,idx[K],P,count);
            const bool needsRepair=CheckSingleGuard<false>(p,s,v,o,idx,count);
            if(__builtin_expect(needsRepair,false)) RepairSmall(b,n,count);
            DecodeSmallGeometry<(K<=64)>(v,p,count);
            StoreSmallOutput(s,o,b,n,count,reserved);
        PipeBarrier<PIPE_ALL>();
    }
    template<bool IndexPrefetched=false,bool EntryInitialized=false>
    __aicore__ inline void InitWithHalfRawReady(GM_ADDR r,GM_ADDR p,GM_ADDR o,FireCandidateDecodeFusionTilingData t,int64_t owner) {
        InitState<EntryInitialized>(r,p,o,t);
        const uint32_t count=t.n<K?t.n:K;
        static_assert(sizeof(T)==2 && K==32,"compact Half resource only");
        CopyDenseInput(halfBuf.template Get<T>(),raw,owner*t.n*6,count*6);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        CopyDenseInput(priorPackedBuf.template Get<float>(),prior,0,count*5);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        InitTables<IndexPrefetched>();
    }
    __aicore__ inline void ProcessSingleWithHalfRawReady(int64_t owner) {
        auto s=rawBuf.template Get<float>();auto v=planeBuf.template Get<float>();auto p=priorBuf.template Get<float>();auto pp=priorPackedBuf.template Get<float>();auto o=outBuf.template Get<float>();
        const uint32_t b=static_cast<uint32_t>(owner),n=0,count=d.n<K?d.n:K;
        uint64_t reserved=0;
            {
                // Only scalar initialization writes these immutable tables.
                // This path has no other VA-register consumer or producer.
                SetSmallOutputContext();
            }
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            EvaluateSingleSigmoid(s,count);
            auto idx=indexBuf.template Get<uint32_t>();
            // Div has produced all raw planes. These gathers only write V;
            // the still-pending prior DMA writes the disjoint packed slot.
            GatherChannelPlanes<6>(v,s,idx,P,count);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
            GatherChannelPlanes<5>(p,pp,idx[K],P,count);
            const bool needsRepair=CheckSingleGuard<false,true>(p,s,v,o,idx,count);
            if(__builtin_expect(needsRepair,false)) RepairSmall(b,n,count);
            DecodeSmallGeometry<(K<=64)>(v,p,count);
            StoreSmallOutput(s,o,b,n,count,reserved);
        PipeBarrier<PIPE_ALL>();
    }
    template<bool StaticGuard=false,bool CompactGeometry=false>
    __aicore__ inline void Process() {
        auto s=rawBuf.template Get<float>();auto v=planeBuf.template Get<float>();auto p=priorBuf.template Get<float>();auto pp=priorPackedBuf.template Get<float>();auto o=outBuf.template Get<float>();
        uint32_t tiles=(d.n+K-1)/K,total=tiles*d.batch,begin=total*GetBlockIdx()/d.blocks,end=total*(GetBlockIdx()+1)/d.blocks;
        uint64_t reserved=0;
        for(uint32_t tile=begin;tile<end;++tile) {
            uint32_t b=tile%d.batch,n=tile/d.batch*K,count=d.n-n<K?d.n-n:K;
            CopyDenseInput(pp,prior,n*5,count*5);
            if constexpr(sizeof(T)==2) {CopyDenseInput(halfBuf.template Get<T>(),raw,(b*d.n+n)*6,count*6);}
            else {CopyDenseInput(rawBuf.template Get<T>(),raw,(b*d.n+n)*6,count*6);}
            if(tile==begin) {
                // Only scalar initialization writes these immutable tables.
                // This path has no other VA-register consumer or producer.
                SetSmallOutputContext();
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            if constexpr(sizeof(T)==2) {Cast(s,halfBuf.template Get<T>(),RoundMode::CAST_NONE,count*6);PipeBarrier<PIPE_V>();}
            Muls(s,s,-1.f,count*6);PipeBarrier<PIPE_V>();
            Exp(s,s,count*6);PipeBarrier<PIPE_V>();Adds(s,s,1.f,count*6);PipeBarrier<PIPE_V>();
            SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(count*6);
            Div<float,false>(s,onesBuf.template Get<float>(),s,MASK_PLACEHOLDER,1,{1,1,1,8,0,8});ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
            auto idx=indexBuf.template Get<uint32_t>();
            GatherChannelPlanes<5>(p,pp,idx[K],P,count);
            GatherChannelPlanes<6>(v,s,idx,P,count);
            if(__builtin_expect(SmallNeedsRepair<T,K,StaticGuard>(p,s,P,count),false)) RepairSmall(b,n,count);
            DecodeSmallGeometry<CompactGeometry>(v,p,count);
            StoreSmallOutput(s,o,b,n,count,reserved);
            if(tile+1<end) {SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);}
            }
        PipeBarrier<PIPE_ALL>();
    }
};
template<class T,uint32_t K,bool EntryInitialized=false>
__aicore__ __attribute__((noinline)) void RunSmallManyIndexed(GM_ADDR r,GM_ADDR p,
    GM_ADDR o,uint32_t batch,uint32_t n,uint32_t blocks,uint32_t tile) {
    static_assert(K==128,"indexed many-tile resource required");
    FireCandidateDecodeFusionTilingData t{batch,n,blocks,tile};
    PadSmallDecode<T,K> op;op.template Init<true,EntryInitialized>(r,p,o,t);op.template Process<true,true>();
}
template<class T,uint32_t K>
__aicore__ __attribute__((noinline)) void RunSmallMany(GM_ADDR r,GM_ADDR p,
    GM_ADDR o,uint32_t batch,uint32_t n,uint32_t blocks,uint32_t tile) {
    FireCandidateDecodeFusionTilingData t{batch,n,blocks,tile};
    if constexpr(K==128) {
        PadSmallDecode<T,K> op;op.Init(r,p,o,t);op.template Process<true>();
    } else if constexpr(K<=64) {
        PadSmallDecode<T,K> op;op.Init(r,p,o,t);op.Process();
    } else {
        TPipe pipe;PadSmallDecode<T,K> op;op.Init(r,p,o,t);op.Process();
    }
}
