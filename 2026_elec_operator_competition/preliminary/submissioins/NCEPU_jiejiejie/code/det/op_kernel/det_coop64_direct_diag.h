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
namespace DetK64DirectDiag {
constexpr uint32_t N=256,RP=104;
template<bool HELPER,bool STREAMED=false> class Vector : public DetK64Base {
    static_assert(!STREAMED||!HELPER,"streamed helper uses DetK64");
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR det,GM_ADDR workspace,const LinalgDetConfig &cfg){
        InitK64Context(workspace,cfg);
        static_assert((192*64+64*RP+192*64+352)*4==126336,"compact main UB bound");
        static_assert((12800+320+4096)*4==68864,"helper UB bound");
        // Same physical UB ranges as269, expressed directly for this fixedshape.
        if constexpr(HELPER){InitK64Helper(a);}else{
            input.SetGlobalBuffer((__gm__ float*)a);output.SetGlobalBuffer((__gm__ float*)det);
            own=allShared[group*t.robustOffset+2*(N*128+128)+q*(N*64+8192)];prodGm=own[4096];
            panel=LocalTensor<float>(TPosition::VECCALC,0,192*64);
            diag=LocalTensor<float>(TPosition::VECCALC,192*64*4,64*RP);
            raw=LocalTensor<float>(TPosition::VECCALC,(192*64+64*RP)*4,192*64);
            col=LocalTensor<float>(TPosition::VECCALC,(192*64+192*64+64*RP)*4,288);row=col[64];meta=col[128];
            // Solve topL and expanded weights occupy disjoint RAW slices. Update reuses both only after the original fences.
            top=raw[4096];
            idx=LocalTensor<uint32_t>(TPosition::VECCALC,(192*64+192*64+64*RP+288)*4,64);
        }
    }

    __aicore__ inline void PrepareStream(uint32_t start){
        // Lower A is immutable during owner Factor. Helpers need it before U.
        Fence<HardEvent::V_MTE3>();
        if(start!=0)DataCopy(shared[N*64+(start+64)*64],panel[start*64],(N-start-64)*64);
        auto signal=SignalBuffer();Fence<HardEvent::MTE3_V>();
        Duplicate(signal,0,32);Fence<HardEvent::V_MTE3>();
        DataCopy(LowerSignalGM(0),signal,32);
        GlobalTensor<int32_t> progress;progress.SetGlobalBuffer((__gm__ int32_t*)shared[N*128].GetPhyAddr());
        DataCopy(progress,signal,8);
    }
    __aicore__ inline void PublishUProgress(uint32_t rows){
        Fence<HardEvent::MTE3_S>();auto signal=SignalBuffer();signal.SetValue(0,int32_t(rows));
        Fence<HardEvent::S_MTE3>();
        GlobalTensor<int32_t> progress;progress.SetGlobalBuffer((__gm__ int32_t*)shared[N*128].GetPhyAddr());
        DataCopy(progress,signal,8);
    }
    __aicore__ inline void PublishUBlock(uint32_t start,uint32_t rows){
        // Called after nextpair V_S: both completed L columns and U rows are immutable.
        DataCopyExtParams lowerPart{16,256,0,(N-64)*4,0};
        DataCopyPad(shared[start+(rows-16)*N],raw[(rows-16)*64],lowerPart);
        DataCopyExtParams upperPart{16,256,(RP-64)/8,0,0};
        DataCopyPad(shared[N*64+(start+rows-16)*64],diag[(rows-16)*RP],upperPart);
    }
    __aicore__ inline void UpdateDirectColumns(uint32_t i,uint32_t k,uint32_t rows,uint32_t rowBase,__ubuf__ float *p,__ubuf__ float *v0,__ubuf__ float *v1){

                    auto ex=(__ubuf__ uint32_t*)(p+rowBase*104+64);set_vector_mask(~uint64_t(0),~uint64_t(0));
                    vbrcb(ex,(__ubuf__ uint32_t*)v0,13,104,(rows+7)/8);vbrcb(ex+8,(__ubuf__ uint32_t*)v1,13,104,(rows+7)/8);PipeBarrier<PIPE_V>();
                    const uint32_t off=(i+2)/8*8;Mask(64-off);
                    // Publish the next pivot's two rows before updating the rest.
                    // Scalar pivot arithmetic can overlap that remaining Vector work.
                    vmla(p+(k+2)*104+off,p+k*104+off,(__ubuf__ float*)ex+(k+2-rowBase)*104,2,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                    vmla(p+(k+2)*104+off,p+(k+1)*104+off,(__ubuf__ float*)ex+8+(k+2-rowBase)*104,2,1,1,0,13,0,13);
                    SetFlag<HardEvent::V_S>(0);
                    if(k+4<64){
                        vmla(p+(k+4)*104+off,p+k*104+off,(__ubuf__ float*)ex+(k+4-rowBase)*104,64-k-4,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(p+(k+4)*104+off,p+(k+1)*104+off,(__ubuf__ float*)ex+8+(k+4-rowBase)*104,64-k-4,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                    }

    }
    __aicore__ inline void PublishDirectUpper(uint32_t start){

            Fence<HardEvent::V_MTE3>();
            if constexpr(!STREAMED){
                DataCopyExtParams lowerStore{64,256,0,(N-64)*4,0};
                DataCopyPad(shared[start],raw,lowerStore);
            }
            if constexpr(STREAMED){PublishUBlock(start,64);}
            else{
                DataCopyExtParams packed{64,256,(RP-64)/8,0,0};
                DataCopyPad(shared[N*64+start*64],diag,packed);
                if(start!=0)DataCopy(shared[N*64+(start+64)*64],panel[start*64],(N-start-64)*64);
            }

    }
    __aicore__ inline void Factor(uint32_t start){
        // The owner block is already current in DIAG; no local copy.
        SetMaskNorm();float m=1.0f;int32_t e=0;uint32_t risk=0;
        const bool saveLower=start+64<N;
        auto p=(__ubuf__ float*)diag.GetPhyAddr();auto c0=(__ubuf__ float*)col.GetPhyAddr();
        auto c1=(__ubuf__ float*)row.GetPhyAddr();auto lower=(__ubuf__ float*)raw.GetPhyAddr();
        #pragma unroll 1
        for(uint32_t i=0;i<64;i+=2){
            const uint32_t k=i,rowBase=(k+2)/8*8,rows=64-rowBase;
            if(i==0)Fence<HardEvent::V_S>();else WaitFlag<HardEvent::V_S>(0);
            if constexpr(STREAMED){if(saveLower&&i&&(i&15)==0){
                PublishUBlock(start,i);PublishUProgress(i);
            }}
            const bool needLower=i+2<64;
            if(needLower){GatherPair(c0,c1,uint32_t(uint64_t(p)+(rowBase*104+i)*4),rows);PipeBarrier<PIPE_V>();}
            const float a=p[k*104+i],b=p[k*104+i+1],c=p[(k+1)*104+i],d=p[(k+1)*104+i+1];
            const float pivot=a*d-b*c;union{float f;uint32_t u;}bits;bits.f=pivot;const uint32_t mag=bits.u&0x7fffffffU;
            risk|=uint32_t(mag<0x3d800000U)|uint32_t(mag>=0x7149f2caU);
            if(needLower){
                auto v0=lower+i*64+rowBase;auto v1=v0+64;const float inv=1.0f/pivot;
                SetMaskCount();set_vector_mask(0,uint64_t(rows));
                vmuls(v0,c0,float(-d*inv),1,1,1,8,8);vmuls(v1,c0,float(b*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
                vaxpy(v0,c1,float(c*inv),1,1,1,8,8);vaxpy(v1,c1,float(-a*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();SetMaskNorm();
                if(i+2<64){        UpdateDirectColumns(i,k,rows,rowBase,p,v0,v1);}
            }
            m*=pivot;
            if((i&7)==6){
                union{uint32_t u;float f;}x;x.f=m;const uint32_t am=x.u&0x7fffffffU;
                risk|=uint32_t(am<=0x0da24260U)|uint32_t(am>=0x7149f2caU);
                e+=int32_t((x.u>>23)&255)-127;x.u=(x.u&0x807fffffU)|(127U<<23);m=x.f;
            }
            if(!needLower)SetFlag<HardEvent::V_S>(0);
        }
        WaitFlag<HardEvent::V_S>(0);SetMaskNorm();ResetMask();
        if(saveLower){        PublishDirectUpper(start);}
        meta.SetValue(0,m);meta.ReinterpretCast<int32_t>().SetValue(1,e);meta.SetValue(2,risk?1.0f:0.0f);meta.SetValue(3,risk?1.0f:0.0f);
        Fence<HardEvent::S_MTE3>();DataCopy(allShared[group*t.robustOffset+N*128+64+q*8],meta,8);
        if(saveLower){
            if constexpr(STREAMED)PublishUProgress(64);
            else{
                auto signal=SignalBuffer();Duplicate(signal,0,32);Fence<HardEvent::V_MTE3>();
                DataCopy(LowerSignalGM(0),signal,32);
            }
        }
    }
    __aicore__ inline void Solve(uint32_t start){
        DataCopyExtParams cp{64,256,(N-64)*4,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        if constexpr(!STREAMED){DataCopyPad(top,shared[start],cp,pad);Fence<HardEvent::MTE2_V>();}
        auto p=(__ubuf__ float*)panel.GetPhyAddr();auto lower=(__ubuf__ uint32_t*)top.GetPhyAddr();
        // RAW is dead until Update and holds eight expanded immutable L columns.
        auto ex=(__ubuf__ uint32_t*)raw.GetPhyAddr();SetMaskNorm();
        #pragma unroll 1
        for(uint32_t i=0;i+2<64;i+=2){
            if constexpr(STREAMED){if((i&15)==0){
                WaitUChunk((__gm__ int32_t*)shared[N*128].GetPhyAddr(),i+16);
                DataCopyExtParams part{16,256,(N-64)*4,0,0};
                DataCopyPad(top[i*64],shared[start+i*N],part,pad);Fence<HardEvent::MTE2_V>();
            }}
            if((i&7)==0){
                set_vector_mask(~uint64_t(0),~uint64_t(0));
                vbrcb(ex,lower+i*64,1,8,i==56?48:64);PipeBarrier<PIPE_V>();
                Mask(64);
            }
            vmla(p+(start+i+2)*64,p+(start+i)*64,(__ubuf__ float*)ex+((i&7)*64+i+2)*8,62-i,1,1,0,8,0,1);PipeBarrier<PIPE_V>();
            vmla(p+(start+i+2)*64,p+(start+i+1)*64,(__ubuf__ float*)ex+(((i&7)+1)*64+i+2)*8,62-i,1,1,0,8,0,1);PipeBarrier<PIPE_V>();
        }
        SetMaskNorm();ResetMask();Fence<HardEvent::V_MTE3>();
        DataCopy(own,panel[start*64],64*64);Fence<HardEvent::MTE3_MTE2>();
    }
    __aicore__ inline void Update(uint32_t start){
        const uint32_t begin=start+64,rows=N-begin,before=first-begin,after=N-first-64;
        DataCopy(raw,prodGm,rows*64);Fence<HardEvent::MTE2_V>();
        // The three disjoint row ranges cover the exact original Schur update.
        if(before)Add(panel[begin*64],panel[begin*64],raw,64,before,{1,1,1,8,8,8});
        Add(diag,diag,raw[before*64],64,64,{1,1,1,13,13,8});
        if(after)Add(panel[first*64],panel[first*64],raw[(before+64)*64],64,after,{1,1,1,8,8,8});
        Fence<HardEvent::V_MTE2>();
    }
    __aicore__ inline void FactorTail(uint32_t start,uint32_t batch){
        FactorTailCommon<true,STREAMED>(start,batch);
    }
    __aicore__ inline LocalTensor<int32_t> SignalBuffer(){
        if constexpr(HELPER)return col[192].template ReinterpretCast<int32_t>();
        else return col[256].template ReinterpretCast<int32_t>();
    }

    __aicore__ inline void PublishLowerReady(){
        PublishK64LowerReady(*this);
    }
    __aicore__ inline bool ProcessOne(uint32_t batch){
        if(batch>=t.groups){CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
        SetMaskNorm();ResetMask();
        if constexpr(!HELPER){
            // Only key5 calls this class: batch=groups=4, all launched groups valid.
            MakeIndex<64,RP*4>(idx.ReinterpretCast<int32_t>());
            Fence<HardEvent::V_MTE2>();DataCopyPadExtParams<float> pad{false,0,0,0};
            const uint64_t base=uint64_t(batch)*N*N+first;
            if(first){
                DataCopyExtParams cp{uint16_t(first),256,(N-64)*4,0,0};
                DataCopyPad(panel,input[base],cp,pad);
            }
            DataCopyExtParams diagonal{64,256,(N-64)*4,(RP-64)/8,0};
            DataCopyPad(diag,input[base+uint64_t(first)*N],diagonal,pad);
            // q0 lower data is read directly from original A by qualified286 helpers.
            if(q&&first+64<N){
                DataCopyExtParams cp{uint16_t(N-first-64),256,(N-64)*4,0,0};
                DataCopyPad(panel[first*64],input[base+uint64_t(first+64)*N],cp,pad);
            }
            Fence<HardEvent::MTE2_V>();
        }
        for(uint32_t start=0;start<N;start+=64){
            const uint32_t owner=start/64;
            shared=allShared[group*t.robustOffset+(start/64%2)*(N*128+128)];
            if constexpr(STREAMED){
                if(start+64<N){
                    if(q==owner)PrepareStream(start);
                    CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);
                    if(q==owner)Factor(start);
                }else{
                    if(q==owner)Factor(start);
                    CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);
                }
            }else{
                if constexpr(!HELPER){if(q==owner)Factor(start);}
                CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);
            }
            if(start+64<N){
                if constexpr(HELPER){FactorTail(start,batch);PublishLowerReady();}
                else {if(q>owner){
                    Solve(start);
                }}
                // Cube waits on all helpers itself; no lower-ready grid barrier.
                if(q>owner){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);if constexpr(!HELPER)Update(start);}
            }
        }
        return FinishK64<HELPER>(batch);
    }
private:
    LocalTensor<float> diag;
};
__aicore__ __attribute__((noinline)) static bool MainRunFixed4(GM_ADDR a,GM_ADDR out,GM_ADDR ws,uint32_t batch){
    const LinalgDetConfig cfg{256,4,256,4,4,64,17024,0,164096};
    Vector<false,true> op;op.Init(a,out,ws,cfg);return op.ProcessOne(batch);
}
__aicore__ inline bool RunVectorFixed4(GM_ADDR a,GM_ADDR out,GM_ADDR ws,uint32_t batch){
    if(GetSubBlockIdx()==1)return DetK64::VectorRunFixed4<true,true>(a,out,ws,batch);
    return MainRunFixed4(a,out,ws,batch);
}
} // namespace DetK64DirectDiag
