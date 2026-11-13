/*
 * Copyright (c) 2026 NCEPU_jiejiejie.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Apache License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "det_k64_base.h"
#include "det_cube_lu.h"
namespace DetK64 {
constexpr uint32_t N=256,RP=104;

template<bool HELPER,bool STREAMED=false> class Vector : public DetK64Base {
    static_assert(!STREAMED||HELPER,"streamed main uses DirectDiag");
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR det,GM_ADDR workspace,const LinalgDetConfig &cfg){
        InitK64Context(workspace,cfg);
        static_assert(N*RP*4+N*64*4+5*N*4+64*64*4+N*4==194560,"main UB bound");
        static_assert((12800+320+4096)*4==68864,"helper UB bound");
        // Same physical UB ranges as269, expressed directly for this fixedshape.
        if constexpr(HELPER){InitK64Helper(a);}else{
            input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)det);
            own=allShared[group*t.robustOffset+2*(N*128+128)+q*(N*64+8192)];prodGm=own[4096];
            panel=LocalTensor<float>(TPosition::VECCALC,0,N*RP);
            raw=LocalTensor<float>(TPosition::VECCALC,N*RP*4,N*64);
            col=LocalTensor<float>(TPosition::VECCALC,(N*RP+N*64)*4,5*N);row=col[N];meta=row[N];
            top=LocalTensor<float>(TPosition::VECCALC,(N*RP+N*64+5*N)*4,4096);
            idx=LocalTensor<uint32_t>(TPosition::VECCALC,(N*RP+N*64+5*N+4096)*4,N);
        }
    }

    __aicore__ inline void Factor(uint32_t start){
        SetMaskNorm();float m=1.0f;int32_t e=0;uint32_t risk=0;
        const bool saveLower=start+64<N;
        auto p=(__ubuf__ float*)panel.GetPhyAddr();auto c0=(__ubuf__ float*)col.GetPhyAddr();
        auto c1=(__ubuf__ float*)row.GetPhyAddr();auto lower=(__ubuf__ float*)raw.GetPhyAddr();
        #pragma unroll 1
        for(uint32_t i=0;i<64;i+=2){
            const uint32_t k=start+i,rowBase=(k+2)/8*8,rows=start+64-rowBase;
            if(i==0)Fence<HardEvent::V_S>();else WaitFlag<HardEvent::V_S>(0);
            const bool needLower=i+2<64;
            if(needLower){GatherPair(c0,c1,uint32_t(uint64_t(p)+(rowBase*104+i)*4),rows);PipeBarrier<PIPE_V>();}
            const float a=p[k*104+i],b=p[k*104+i+1],c=p[(k+1)*104+i],d=p[(k+1)*104+i+1];
            const float pivot=a*d-b*c;union{float f;uint32_t u;}bits;bits.f=pivot;const uint32_t mag=bits.u&0x7fffffffU;
            risk|=uint32_t(mag<0x3d800000U)|uint32_t(mag>=0x7149f2caU);
            if(needLower){
                auto v0=lower+i*N+rowBase;auto v1=v0+N;const float inv=1.0f/pivot;
                SetMaskCount();set_vector_mask(0,uint64_t(rows));
                vmuls(v0,c0,float(-d*inv),1,1,1,8,8);vmuls(v1,c0,float(b*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
                vaxpy(v0,c1,float(c*inv),1,1,1,8,8);vaxpy(v1,c1,float(-a*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();SetMaskNorm();
                if(i+2<64){
                    auto ex=(__ubuf__ uint32_t*)(p+rowBase*104+64);set_vector_mask(~uint64_t(0),~uint64_t(0));
                    vbrcb(ex,(__ubuf__ uint32_t*)v0,13,104,(rows+7)/8);vbrcb(ex+8,(__ubuf__ uint32_t*)v1,13,104,(rows+7)/8);PipeBarrier<PIPE_V>();
                    const uint32_t off=(i+2)/8*8;Mask(64-off);
                    vmla(p+(k+2)*104+off,p+k*104+off,(__ubuf__ float*)ex+(k+2-rowBase)*104,start+64-k-2,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                    vmla(p+(k+2)*104+off,p+(k+1)*104+off,(__ubuf__ float*)ex+8+(k+2-rowBase)*104,start+64-k-2,1,1,0,13,0,13);
                }
            }
            m*=pivot;
            if((i&7)==6){
                union{uint32_t u;float f;}x;x.f=m;const uint32_t am=x.u&0x7fffffffU;
                risk|=uint32_t(am<=0x0da24260U)|uint32_t(am>=0x7149f2caU);
                e+=int32_t((x.u>>23)&255)-127;x.u=(x.u&0x807fffffU)|(127U<<23);m=x.f;
            }
            SetFlag<HardEvent::V_S>(0);
        }
        WaitFlag<HardEvent::V_S>(0);SetMaskNorm();ResetMask();
        if(saveLower){
            Fence<HardEvent::V_MTE3>();
            DataCopyExtParams lowerStore{64,256,(N-64)/8,(N-64)*4,0};
            DataCopyPad(shared[start],raw[start],lowerStore);
            DataCopyExtParams packed{uint16_t(start==0&&t.batch==4&&t.groups==4?64:N-start),256,(RP-64)/8,0,0};
            DataCopyPad(shared[N*64+start*64],panel[start*RP],packed);
        }
        meta.SetValue(0,m);meta.ReinterpretCast<int32_t>().SetValue(1,e);meta.SetValue(2,risk?1.0f:0.0f);meta.SetValue(3,risk?1.0f:0.0f);
        Fence<HardEvent::S_MTE3>();DataCopy(allShared[group*t.robustOffset+N*128+64+q*8],meta,8);
        if(saveLower){
            auto signal=SignalBuffer();Duplicate(signal,0,32);Fence<HardEvent::V_MTE3>();
            DataCopy(LowerSignalGM(0),signal,32);
        }
    }
    __aicore__ inline void Solve(uint32_t start){
        DataCopyExtParams cp{64,256,(N-64)*4,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(top,shared[start],cp,pad);Fence<HardEvent::MTE2_V>();
        auto p=(__ubuf__ float*)panel.GetPhyAddr();auto lower=(__ubuf__ uint32_t*)top.GetPhyAddr();
        // RAW is dead until Update and holds eight expanded immutable L columns.
        auto ex=(__ubuf__ uint32_t*)raw.GetPhyAddr();SetMaskNorm();
        #pragma unroll 1
        for(uint32_t i=0;i+2<64;i+=2){
            if((i&7)==0){
                set_vector_mask(~uint64_t(0),~uint64_t(0));
                vbrcb(ex,lower+i*64,1,8,i==56?48:64);PipeBarrier<PIPE_V>();
                Mask(64);
            }
            vmla(p+(start+i+2)*104,p+(start+i)*104,(__ubuf__ float*)ex+((i&7)*64+i+2)*8,62-i,1,1,0,13,0,1);PipeBarrier<PIPE_V>();
            vmla(p+(start+i+2)*104,p+(start+i+1)*104,(__ubuf__ float*)ex+(((i&7)+1)*64+i+2)*8,62-i,1,1,0,13,0,1);PipeBarrier<PIPE_V>();
        }
        SetMaskNorm();ResetMask();Fence<HardEvent::V_MTE3>();
        DataCopyExtParams store{64,256,5,0,0};DataCopyPad(own,panel[start*104],store);Fence<HardEvent::MTE3_MTE2>();
    }
    __aicore__ inline void Update(uint32_t start){
        const uint32_t rows=N-start-64;DataCopy(raw,prodGm,rows*64);Fence<HardEvent::MTE2_V>();
        Add(panel[(start+64)*104],panel[(start+64)*104],raw,64,rows,{1,1,1,13,13,8});Fence<HardEvent::V_MTE2>();
    }
    __aicore__ inline void FactorTail(uint32_t start,uint32_t batch){
        FactorTailCommon<false,STREAMED>(start,batch);
    }
    __aicore__ inline LocalTensor<int32_t> SignalBuffer(){
        if constexpr(HELPER)return col[192].template ReinterpretCast<int32_t>();
        else return col[4*N].template ReinterpretCast<int32_t>();
    }

    __aicore__ inline void PublishLowerReady(){
        PublishK64LowerReady(*this);
    }
    __aicore__ inline bool ProcessOne(uint32_t batch){
        if(batch>=t.groups){CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
        SetMaskNorm();ResetMask();
        if constexpr(!HELPER){
            MakeIndex<N,RP*4>(idx.ReinterpretCast<int32_t>());
            if(batch>=t.batch){
                Duplicate(panel,0.0f,N*RP);Fence<HardEvent::V_S>();
                for(uint32_t j=first;j<first+64;++j)panel.SetValue(j*RP+j-first,1.0f);
                Fence<HardEvent::S_V>();
            }else{
                Fence<HardEvent::V_MTE2>();DataCopyExtParams cp{uint16_t(q==0&&t.batch==4&&t.groups==4?64:N),256,(N-64)*4,(RP-64)/8,0};DataCopyPadExtParams<float> pad{false,0,0,0};
                DataCopyPad(panel,input[uint64_t(batch)*N*N+first],cp,pad);Fence<HardEvent::MTE2_V>();
            }
        }
        for(uint32_t start=0;start<N;start+=64){
            const uint32_t owner=start/64;
            shared=allShared[group*t.robustOffset+(start/64%2)*(N*128+128)];
            if constexpr(STREAMED){
                if(start+64<N){
                    CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);
                    FactorTail(start,batch);PublishLowerReady();
                    if(q>owner){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);}
                }else{CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
            }else{
                if constexpr(!HELPER){if(q==owner)Factor(start);}
                CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);
                if(start+64<N){
                    if constexpr(HELPER){FactorTail(start,batch);PublishLowerReady();}
                    else {if(q>owner)Solve(start);}
                    if(q>owner){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);if constexpr(!HELPER)Update(start);}
                }
            }
        }
        return FinishK64<HELPER>(batch);
    }
private:

};
__aicore__ inline void RunCubePlan(GM_ADDR ws,uint32_t robustOffset,uint32_t batches,uint32_t groups){
    if ASCEND_IS_AIC {
        // Basic API matches the qualified SDK9 Matmul Mmad/Fixpipe control fields.
        TPipe pipe;
        DetCubePipeline<64> cube;
        GlobalTensor<float> shared,own,prod;
        const uint32_t q=GetBlockIdx()%4,group=GetBlockIdx()/4;
        if(q==0)return;
        own.SetGlobalBuffer((__gm__ float*)ws+group*robustOffset+2*(256*128+128)+q*(256*64+8192));prod=own[4096];
        for(uint32_t batch=0;batch<(batches+groups-1)/groups;++batch)
          for(uint32_t start=0;start<q*64;start+=64){
            const uint32_t rows=256-start-64;
            shared.SetGlobalBuffer((__gm__ float*)ws+group*robustOffset+(start/64%2)*(256*128+128));
            auto ap=DetNd2NzParams(64,rows,256);
            auto bp=DetNd2NzParams(64,64,64);
            // Paired8 commits this core's B and helper.  Other helpers may
            // still be producing lower A, so keep all four volatile reads.
            CrossCoreWaitFlag(8);DataCopy(cube.b1,own,bp);
            DetWaitLowerFlags(shared,256*128+96);
            DataCopy(cube.a1,shared[start+64],ap);
            cube.Compute(rows,rows,prod);
          }
        PipeBarrier<PIPE_ALL>();
    }
}
__aicore__ inline void RunCube(const LinalgDetTilingData &t,GM_ADDR ws){RunCubePlan(ws,t.robustOffset,t.batch,t.groups);}

template<bool HELPER> __aicore__ __attribute__((noinline)) static bool VectorRun(GM_ADDR a,GM_ADDR out,GM_ADDR ws,const LinalgDetConfig&t,uint32_t batch){
    Vector<HELPER> op;op.Init(a,out,ws,t);return op.ProcessOne(batch);
}
__aicore__ inline bool RunVector(GM_ADDR a,GM_ADDR out,GM_ADDR ws,const LinalgDetConfig&t,uint32_t batch){
    if(GetSubBlockIdx()==1)return VectorRun<true>(a,out,ws,t,batch);
    return VectorRun<false>(a,out,ws,t,batch);
}
__aicore__ inline void RunCubeFixed4(GM_ADDR ws){
    RunCubePlan(ws,164096U,4U,4U);
}
template<bool HELPER,bool STREAMED=false> __aicore__ __attribute__((noinline)) static bool VectorRunFixed4(GM_ADDR a,GM_ADDR out,GM_ADDR ws,uint32_t batch){
    // Only the nine-word fixed geometry changes; Vector arithmetic is shared.
    const LinalgDetConfig cfg{256,4,256,4,4,64,17024,0,164096};
    Vector<HELPER,STREAMED> op;op.Init(a,out,ws,cfg);return op.ProcessOne(batch);
}
__aicore__ inline bool RunVectorFixed4(GM_ADDR a,GM_ADDR out,GM_ADDR ws,uint32_t batch){
    if(GetSubBlockIdx()==1)return VectorRunFixed4<true>(a,out,ws,batch);
    return VectorRunFixed4<false>(a,out,ws,batch);
}
} // namespace DetK64
