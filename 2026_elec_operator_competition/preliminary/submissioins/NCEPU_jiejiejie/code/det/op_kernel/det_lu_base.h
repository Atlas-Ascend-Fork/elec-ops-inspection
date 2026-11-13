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
#include "kernel_operator.h"
#include "linalg_det_tiling.h"
using namespace AscendC;

class DetScalarOps {
public:
    __aicore__ inline uint32_t Min(uint32_t a,uint32_t b){return a<b?a:b;}
    __aicore__ inline float Pow2(int32_t e){
        union{uint32_t u;float f;}x; x.u=uint32_t(e+127)<<23; return x.f;
    }
    __aicore__ inline void Acc(float p,float &m,int32_t &e){
        union{uint32_t u;float f;}x; x.f=p;
        e+=int32_t((x.u>>23)&255)-127; x.u=(x.u&0x807fffffU)|(127U<<23); m*=x.f;
        x.f=m; e+=int32_t((x.u>>23)&255)-127; x.u=(x.u&0x807fffffU)|(127U<<23); m=x.f;
    }
};

// Shared storage and vector primitives for the dynamic and fixed LU plans.
class DetLUBase : public DetScalarOps {
public:
    template<HardEvent E> __aicore__ inline void Fence(){
        SetFlag<E>(0); WaitFlag<E>(0);
    }

    __aicore__ inline void NativeMask(uint32_t n){set_vector_mask(0,n>=64?~uint64_t(0):(uint64_t(1)<<n)-1);}

    __aicore__ inline void NativeGather(__ubuf__ float *dst,uint32_t base,uint32_t rows){
        auto ip=(__ubuf__ uint32_t*)idx.GetPhyAddr();const uint32_t full=rows/64,tail=rows%64;
        if(full){NativeMask(64);vgather((__ubuf__ uint32_t*)dst,ip,base,8,uint8_t(full));}
        if(tail){NativeMask(tail);vgather((__ubuf__ uint32_t*)(dst+full*64),ip+full*64,base,8,1);}
    }
    __aicore__ inline void NativeMuls(__ubuf__ float *dst,__ubuf__ float *src,float value,uint32_t rows){
        const uint32_t full=rows/64,tail=rows%64;
        if(full){NativeMask(64);vmuls(dst,src,float(value),uint8_t(full),1,1,8,8);}
        if(tail){NativeMask(tail);vmuls(dst+full*64,src+full*64,float(value),1,1,1,8,8);}
    }
    __aicore__ inline void NativeAxpy(__ubuf__ float *dst,__ubuf__ float *src,float value,uint32_t rows){
        const uint32_t full=rows/64,tail=rows%64;
        if(full){NativeMask(64);vaxpy(dst,src,float(value),uint8_t(full),1,1,8,8);}
        if(tail){NativeMask(tail);vaxpy(dst+full*64,src+full*64,float(value),1,1,1,8,8);}
    }
    __aicore__ inline void NativeCopy(__ubuf__ float *dst,__ubuf__ float *src,uint32_t rows){
        const uint32_t full=rows/64,tail=rows%64;
        if(full){NativeMask(64);vadds(dst,src,0.0f,uint8_t(full),1,1,8,8);}
        if(tail){NativeMask(tail);vadds(dst+full*64,src+full*64,0.0f,1,1,1,8,8);}
    }
    __aicore__ inline void GatherPair(__ubuf__ float *d0,__ubuf__ float *d1,uint32_t base,uint32_t rows){
        auto ip=(__ubuf__ uint32_t*)idx.GetPhyAddr();NativeMask(64);
        vgather((__ubuf__ uint32_t*)d0,ip,base,8,uint8_t((rows+63)/64));
        vgather((__ubuf__ uint32_t*)d1,ip,base+4,8,uint8_t((rows+63)/64));
    }
    __aicore__ inline void MulsPair(__ubuf__ float *d0,__ubuf__ float *d1,__ubuf__ float *src,float v0,float v1,uint32_t rows){
        const uint32_t full=rows/64,tail=rows%64;
        if(full){NativeMask(64);vmuls(d0,src,v0,uint8_t(full),1,1,8,8);vmuls(d1,src,v1,uint8_t(full),1,1,8,8);}
        if(tail){NativeMask(tail);vmuls(d0+full*64,src+full*64,v0,1,1,1,8,8);vmuls(d1+full*64,src+full*64,v1,1,1,1,8,8);}
    }
    __aicore__ inline void AxpyPair(__ubuf__ float *d0,__ubuf__ float *d1,__ubuf__ float *src,float v0,float v1,uint32_t rows){
        const uint32_t full=rows/64,tail=rows%64;
        if(full){NativeMask(64);vaxpy(d0,src,v0,uint8_t(full),1,1,8,8);vaxpy(d1,src,v1,uint8_t(full),1,1,8,8);}
        if(tail){NativeMask(tail);vaxpy(d0+full*64,src+full*64,v0,1,1,1,8,8);vaxpy(d1+full*64,src+full*64,v1,1,1,1,8,8);}
    }
    __aicore__ inline void SolvePanel(uint32_t start,uint32_t pn){
        DataCopyExtParams cp{32,128,(pn-32)*4,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(top,shared[start],cp,pad);Fence<HardEvent::MTE2_V>();
        auto pp=(__ubuf__ float*)panel.GetPhyAddr();auto lower=(__ubuf__ uint32_t*)top.GetPhyAddr();auto expanded=(__ubuf__ uint32_t*)(pp+start*104+64);
        SetMaskNorm();
        for(uint32_t i=0;i+2<32;i+=2){
            set_vector_mask(~uint64_t(0),~uint64_t(0));
            vbrcb(expanded,lower+i*32,13,104,4);vbrcb(expanded+8,lower+(i+1)*32,13,104,4);PipeBarrier<PIPE_V>();NativeMask(64);
            vmla(pp+(start+i+2)*104,pp+(start+i)*104,(__ubuf__ float*)expanded+(i+2)*104,30-i,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
            vmla(pp+(start+i+2)*104,pp+(start+i+1)*104,(__ubuf__ float*)expanded+8+(i+2)*104,30-i,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
        }
        SetMaskNorm();ResetMask();Fence<HardEvent::V_MTE3>();DataCopyExtParams store{32,256,5,0,0};DataCopyPad(own,panel[start*104],store);Fence<HardEvent::MTE3_MTE2>();
    }
    __aicore__ inline void Solve(uint32_t start){SolvePanel(start,t.paddedN);}

    __aicore__ inline void UpdatePanel(uint32_t start,uint32_t n){
        const uint32_t off=q==start/64?start%64+32:0;
        if(off>=64||start+32>=n)return;
        const uint32_t rows=n-start-32;
        DataCopy(raw,prodGm,rows*64);Fence<HardEvent::MTE2_V>();
        Add(panel[(start+32)*104+off],panel[(start+32)*104+off],raw[off],64-off,rows,{1,1,1,13,13,8});Fence<HardEvent::V_MTE2>();
    }
    __aicore__ inline void Update(uint32_t start){UpdatePanel(start,t.n);}

    __aicore__ inline void InitBuffers(TPipe &pipe, GM_ADDR A, GM_ADDR det, GM_ADDR ws) {
        input.SetGlobalBuffer((__gm__ float*)A);
        output.SetGlobalBuffer((__gm__ float*)det);
        allShared.SetGlobalBuffer((__gm__ float*)ws);
        shared=allShared[group*t.robustOffset];
        comm=shared[t.paddedN*64];
        own=allShared[group*t.robustOffset+2*(t.paddedN*64+128)+q*(t.paddedN*64+8192)];
        prodGm=own[4096];
        pipe.InitBuffer(panelBuf,(t.paddedN+32)*104*4); panel=panelBuf.Get<float>();
        pipe.InitBuffer(rawBuf,t.paddedN*64*4); raw=rawBuf.Get<float>();
        const uint32_t vectorPitch=(t.paddedN+63)/64*64;
        pipe.InitBuffer(vecBuf,5*vectorPitch*4); col=vecBuf.Get<float>();
        row=col[vectorPitch]; meta=row[vectorPitch]; temp=meta[vectorPitch];
        pipe.InitBuffer(topBuf,32*32*4); top=topBuf.Get<float>();
        pipe.InitBuffer(idxBuf,vectorPitch*4); idx=idxBuf.Get<uint32_t>();
        }
    template<bool COOPERATIVE, bool ACCUMULATE=false,bool SEQUENTIAL=false> __aicore__ inline void FactorCooperative(uint32_t start){
        SetMaskNorm();
        float m=1.0f;int32_t e=0;uint32_t riskMask=0;
        const uint32_t rowEnd=COOPERATIVE?start+32:t.n;
        const uint32_t rows=rowEnd-start,steps=COOPERATIVE?32U:Min((!SEQUENTIAL&&t.n==256&&start==192)?64U:32U,t.n-start),local=start%64;
        const bool saveLower=COOPERATIVE||start+steps<t.n;
        const bool splitColumns=local==0&&saveLower;
        auto pp=(__ubuf__ float*)panel.GetPhyAddr();auto c0=(__ubuf__ float*)col.GetPhyAddr();
        auto c1=(__ubuf__ float*)row.GetPhyAddr();
        auto lower=(__ubuf__ float*)raw.GetPhyAddr();auto expanded=(__ubuf__ uint32_t*)(pp+start*104+64);
        for(uint32_t i=0;i<steps;i+=2){
            const uint32_t k=start+i;
            auto v0=lower+i*t.paddedN+start;auto second=lower+(i+1)*t.paddedN+start;
            if(i==0)Fence<HardEvent::V_S>();else WaitFlag<HardEvent::V_S>(0);
            float p;bool pivotReady=false;
            if(i+1==steps){
                p=pp[k*104+local+i];union{float f;uint32_t u;}bits;bits.f=p;const uint32_t ap=bits.u&0x7fffffffU;
                riskMask|=uint32_t(ap<0x38d1b717U)|uint32_t(ap>=0x60ad78ecU);
            }
            else{
                const bool needLower=saveLower||i+2<steps;
                if(needLower){GatherPair(c0,c1,uint32_t(uint64_t(pp)+(start*104+local+i)*4),rows);PipeBarrier<PIPE_V>();}
                const float a=pp[k*104+local+i],b=pp[k*104+local+i+1],c=pp[(k+1)*104+local+i],d=pp[(k+1)*104+local+i+1];
                p=a*d-b*c;union{float f;uint32_t u;}bits;bits.f=p;const uint32_t ap=bits.u&0x7fffffffU;
                riskMask|=uint32_t(ap<0x3d800000U)|uint32_t(ap>=0x7149f2caU);
                if(needLower) PreparePairLower(p,a,b,c,d,rows,v0,second,c0,c1);
                if(i+2<steps) pivotReady=UpdatePairColumns(start,i,k,rows,steps,local,rowEnd,splitColumns,pp,v0,second,expanded);
            }
            AccumulateRisk(p,i,steps,m,e,riskMask);
            if(pivotReady){if(!splitColumns)PipeBarrier<PIPE_V>();}
            else SetFlag<HardEvent::V_S>(0);
        }
        WaitFlag<HardEvent::V_S>(0);
        PublishFactor<COOPERATIVE,ACCUMULATE,SEQUENTIAL>(start,steps,local,saveLower,m,e,riskMask);
    }
    __aicore__ inline void PreparePairLower(float p,float a,float b,float c,float d,uint32_t rows,__ubuf__ float *v0,__ubuf__ float *second,__ubuf__ float *c0,__ubuf__ float *c1){
                    const float inv=1.0f/p;
                    if(rows>64&&rows%64!=0){
                        SetMaskCount();set_vector_mask(0,uint64_t(rows));
                        vmuls(v0,c0,float(-d*inv),1,1,1,8,8);vmuls(second,c0,float(b*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
                        vaxpy(v0,c1,float(c*inv),1,1,1,8,8);vaxpy(second,c1,float(-a*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
                        SetMaskNorm();
                    }else{
                        MulsPair(v0,second,c0,-d*inv,b*inv,rows);PipeBarrier<PIPE_V>();
                        AxpyPair(v0,second,c1,c*inv,-a*inv,rows);PipeBarrier<PIPE_V>();
                    }

    }
    __aicore__ inline bool UpdatePairColumns(uint32_t start,uint32_t i,uint32_t k,uint32_t rows,uint32_t steps,uint32_t local,uint32_t rowEnd,bool splitColumns,__ubuf__ float *pp,__ubuf__ float *v0,__ubuf__ float *second,__ubuf__ uint32_t *expanded){
        bool pivotReady=false;
                    set_vector_mask(~uint64_t(0),~uint64_t(0));vbrcb(expanded,(__ubuf__ uint32_t*)v0,13,104,(rows+7)/8);
                    vbrcb(expanded+8,(__ubuf__ uint32_t*)second,13,104,(rows+7)/8);PipeBarrier<PIPE_V>();
                    const uint32_t off=(local+i+2)/8*8,boundary=Min(start+steps,t.n),upper=boundary-k-2;
                    if(splitColumns){
                        NativeMask(32-off);
                        vmla(pp+(k+2)*104+off,pp+k*104+off,(__ubuf__ float*)expanded+(i+2)*104,rowEnd-k-2,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(pp+(k+2)*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+(i+2)*104,rowEnd-k-2,1,1,0,13,0,13);
                        SetFlag<HardEvent::V_S>(0);pivotReady=true;
                        NativeMask(32);
                        vmla(pp+(k+2)*104+32,pp+k*104+32,(__ubuf__ float*)expanded+(i+2)*104,upper,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(pp+(k+2)*104+32,pp+(k+1)*104+32,(__ubuf__ float*)expanded+8+(i+2)*104,upper,1,1,0,13,0,13);
                    }else if(local==32){
                        NativeMask(64-off);
                        vmla(pp+(k+2)*104+off,pp+k*104+off,(__ubuf__ float*)expanded+(i+2)*104,rowEnd-k-2,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                        vmla(pp+(k+2)*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+(i+2)*104,rowEnd-k-2,1,1,0,13,0,13);
                    }else{
                        if(upper){
                            NativeMask(64-off);
                            vmla(pp+(k+2)*104+off,pp+k*104+off,(__ubuf__ float*)expanded+(i+2)*104,upper,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                            vmla(pp+(k+2)*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+(i+2)*104,upper,1,1,0,13,0,13);
                            SetFlag<HardEvent::V_S>(0);pivotReady=true;
                        }
                        if(boundary<rowEnd&&off<local+32){
                            NativeMask(local+32-off);
                            vmla(pp+boundary*104+off,pp+k*104+off,(__ubuf__ float*)expanded+32*104,rowEnd-boundary,1,1,0,13,0,13);PipeBarrier<PIPE_V>();
                            vmla(pp+boundary*104+off,pp+(k+1)*104+off,(__ubuf__ float*)expanded+8+32*104,rowEnd-boundary,1,1,0,13,0,13);
                        }
                    }

        return pivotReady;
    }
    template<bool COOPERATIVE,bool ACCUMULATE,bool SEQUENTIAL> __aicore__ inline void PublishFactor(uint32_t start,uint32_t steps,uint32_t local,bool saveLower,float m,int32_t e,uint32_t riskMask){
        const bool risky=riskMask!=0;
        if constexpr(ACCUMULATE) { SetMaskNorm();ResetMask();ownRisk=ownRisk||risky;ownedM*=m;ownedE+=e;Fence<HardEvent::V_S>();meta.SetValue(0,ownedM);meta.ReinterpretCast<int32_t>().SetValue(1,ownedE);meta.SetValue(2,ownRisk?1.0f:0.0f);meta.SetValue(3,ownRisk?1.0f:0.0f); }
        else { SetMaskNorm();ResetMask();Fence<HardEvent::V_S>();meta.SetValue(0,m);meta.ReinterpretCast<int32_t>().SetValue(1,e);meta.SetValue(2,risky?1.0f:0.0f);ownRisk=ownRisk||risky;meta.SetValue(3,ownRisk?1.0f:0.0f); }
        if(saveLower){
            if constexpr(SEQUENTIAL){
            Fence<HardEvent::V_MTE3>();DataCopy(shared,raw,32*t.paddedN);
            DataCopyExtParams cp{32,256,5,0,0};DataCopyPad(own,panel[start*104],cp);
            }else{
            if(COOPERATIVE){
                DataCopyParams pack{uint16_t(t.n-start),4,9,0};
                DataCopy(raw[32*t.paddedN],panel[start*104+local],pack);
            }
            Fence<HardEvent::V_MTE3>();
            if(COOPERATIVE){
                DataCopyExtParams upper{32,128,(t.paddedN-32)/8,(t.paddedN-32)*4,0};
                DataCopyPad(shared[start],raw[start],upper);
                DataCopy(shared[32*t.paddedN+start*32],raw[32*t.paddedN],(t.n-start)*32);
            }else {const uint32_t rows=t.n-start;DataCopyExtParams copy{32,rows*4,(t.paddedN-(rows+7)/8*8)/8,(t.paddedN-rows)*4,0};DataCopyPad(shared[start],raw[start],copy);}
            if(local==0){DataCopyExtParams cp{32,256,5,0,0};DataCopyPad(own,panel[start*104],cp);}

            }
        }
        Fence<HardEvent::S_MTE3>();DataCopy(comm,meta,8);
        if(start+steps>=Min(first+64,t.n))DataCopy(allShared[group*t.robustOffset+t.paddedN*64+64+q*8],meta,8);

    }

    __aicore__ inline void AccumulateRisk(float p,uint32_t i,uint32_t steps,float &m,int32_t &e,uint32_t &riskMask){
            m*=p;
            if((i&7)==6||i+2>=steps){
                union{uint32_t u;float f;}x;x.f=m;const uint32_t am=x.u&0x7fffffffU;
                riskMask|=uint32_t(am<=0x0da24260U)|uint32_t(am>=0x7149f2caU);
                e+=int32_t((x.u>>23)&255)-127;x.u=(x.u&0x807fffffU)|(127U<<23);m=x.f;
            }
    }

    template<bool MAKE_TEMP> __aicore__ inline void LoadDynamicPanel(uint32_t batch){
        const uint32_t vectorPitch=(t.paddedN+63)/64*64;
        CreateVecIndex(idx.ReinterpretCast<int32_t>(),0,vectorPitch);PipeBarrier<PIPE_V>();
        Muls(idx.ReinterpretCast<int32_t>(),idx.ReinterpretCast<int32_t>(),int32_t(416),vectorPitch);PipeBarrier<PIPE_V>();
        if constexpr(MAKE_TEMP){
            CreateVecIndex(temp.ReinterpretCast<int32_t>(),0,64);PipeBarrier<PIPE_V>();
            Muls(temp.ReinterpretCast<int32_t>(),temp.ReinterpretCast<int32_t>(),int32_t(128),64);PipeBarrier<PIPE_V>();
        }
        LoadPanelMemory(batch,t.n,t.paddedN);
    }

    __aicore__ inline bool ReadRiskFlags(uint32_t cores,uint32_t offset){
        DataCopyExtParams flags{uint16_t(t.groups),cores*32,t.robustOffset*4-cores*32,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
        DataCopyPad(meta,allShared[offset],flags,pad);Fence<HardEvent::MTE2_S>();
        bool safe=true;for(uint32_t c=0;c<t.groups*cores;++c)safe=safe&&meta.GetValue(c*8+3)==0.0f;
        return safe;
    }

    __aicore__ inline void WriteProduct(uint32_t batch,float m,int32_t e){
            while(e>120){m*=Pow2(120);e-=120;} while(e< -120){m*=Pow2(-120);e+=120;}
            Fence<HardEvent::V_S>(); meta.SetValue(0,m*Pow2(e)); Fence<HardEvent::S_MTE3>();
            DataCopyExtParams cp{1,4,0,0,0}; DataCopyPad(output[batch],meta,cp);
            Fence<HardEvent::MTE3_S>();

    }

    template<bool HELPERS,typename Op> __aicore__ inline void UpdateDynamicTrailing(Op &op,uint32_t start,uint32_t owner,bool cooperative){
        if constexpr(HELPERS){
                // Only noncooperative n256 stages have complete lower A at global10.
                const bool cubeNeeded=q>owner||(q==owner&&start%64==0);
                if(t.n==256&&!cooperative&&cubeNeeded)CrossCoreSetFlag<2,PIPE_MTE3>(11);
                // Keep the original helper/Solve overlap in cooperative stages.
                if(q>owner){
                    if(cooperative)Fence<HardEvent::V_MTE2>();
                    op.Solve(start);
                }

                if(cubeNeeded){if(cooperative&&!(t.n==256&&t.coresPerMatrix==4))op.WaitLowerReady();CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);Update(start);}

        }else{
                if(cooperative){
                    op.FactorTail(start);
                    CrossCoreSetFlag<0,PIPE_MTE3>(12);
                }
                if(q>owner){
                    if(cooperative)Fence<HardEvent::V_MTE2>();
                    op.Solve(start);
                }
                if(cooperative)CrossCoreWaitFlag(12);
                const bool cubeNeeded=q>owner||(q==owner&&start%64==0);
                if(cubeNeeded){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);Update(start);}

        }
    }

    template<bool HELPERS,typename Op> __aicore__ inline void ProcessDynamicPanels(Op &op,float &m,int32_t &e){
for(uint32_t start=0;start<t.n;start+=(t.n==256&&start==192)?64:32){
            const uint32_t owner=start/64;
            shared=allShared[group*t.robustOffset+(start/32%2)*(t.paddedN*64+128)];comm=shared[t.paddedN*64];
            const bool cooperative=start+32<t.n&&t.n-start-32>=t.coresPerMatrix*32;
            if(q==owner){
                if(cooperative)op.template Factor<true>(start);else op.template Factor<false>(start);
            }
            {CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
            if(q==0){
                if(q!=owner){DataCopy(meta,comm,8);Fence<HardEvent::MTE2_S>();}
                e+=meta.ReinterpretCast<int32_t>().GetValue(1);m*=meta.GetValue(0);
            }
            if(start+32<t.n&&!(t.n==256&&start==192))UpdateDynamicTrailing<HELPERS>(op,start,owner,cooperative);
        }
    }

    template<typename Op>
    __aicore__ inline void WaitSharedLower(Op &op){
        auto signal=op.SignalBuffer();Fence<HardEvent::MTE3_MTE2>();
        do{DataCopy(signal,op.LowerSignalGM(0),32);Fence<HardEvent::MTE2_S>();}
        while(signal.GetValue(0)!=1||signal.GetValue(8)!=1||signal.GetValue(16)!=1||signal.GetValue(24)!=1);
    }

    __aicore__ inline void LoadPanelMemory(uint32_t batch,uint32_t n,uint32_t pn){
        Duplicate(panel[pn*104],0.0f,32*104);
        if(batch>=t.batch||first+64>n)Duplicate(panel,0.0f,pn*104);
        else if(pn>n)Duplicate(panel[n*104],0.0f,(pn-n)*104);
        Fence<HardEvent::V_S>();
        for(uint32_t r=batch<t.batch?n:0;r<pn;++r)
            if(r>=first&&r<first+64)panel.SetValue(r*104+r-first,1.0f);
        Fence<HardEvent::S_MTE2>();
        if(first<n&&batch<t.batch){
            const uint32_t width=Min(64,n-first),aligned=(width+7)/8*8;
            DataCopyExtParams cp{uint16_t(n),width*4,(n-width)*4,(104-aligned)/8,0};
            DataCopyPadExtParams<float> pad{true,0,uint8_t(aligned-width),0};
            DataCopyPad(panel,input[uint64_t(batch)*n*n+first],cp,pad);
            Fence<HardEvent::MTE2_V>();
        }
        }

    struct TailLayout {
        uint32_t pitch,begin,rows,columnPitch,packedOffset;
        LocalTensor<float> scratch;
        __ubuf__ float *lower,*matrix,*packed,*u;
        __ubuf__ uint32_t *expanded,*offsets;
    };
    template<uint32_t WIDTH,bool REUSE_PACKED>
    __aicore__ inline bool SetupTail(uint32_t start,TailLayout &s){
        const uint32_t remaining=t.n-start-WIDTH;
        s.pitch=((remaining+t.coresPerMatrix-1)/t.coresPerMatrix+31)/32*32;
        s.begin=start+WIDTH+q*s.pitch;
        if(s.begin>=t.n)return false;
        s.rows=Min(s.pitch,t.n-s.begin);s.columnPitch=s.pitch+8;
        s.scratch=raw;s.lower=(__ubuf__ float*)s.scratch.GetPhyAddr();s.matrix=s.lower+WIDTH*s.pitch;
        s.packedOffset=WIDTH*s.pitch+WIDTH*s.columnPitch;s.packed=s.lower+s.packedOffset;
        s.expanded=(__ubuf__ uint32_t*)(s.packed+(REUSE_PACKED?0:32*s.pitch));
        s.u=(__ubuf__ float*)top.GetPhyAddr();
        if constexpr(WIDTH==32)s.offsets=(__ubuf__ uint32_t*)temp.GetPhyAddr();
        return true;
    }

    template<uint32_t WIDTH> __aicore__ inline void TransposePacked(uint32_t rows,uint32_t columnPitch,__ubuf__ float *packed,__ubuf__ float *matrix){
            // Repeat over four8-column groups; each call transposes16 rows.
            TransDataTo5HDParams cv{};cv.repeatTimes=WIDTH/8;
            cv.srcRepStride=1;cv.dstRepStride=columnPitch;
            for(uint32_t rb=0;rb<rows;rb+=16){
                uint64_t srcList[16],dstList[16];
                #pragma unroll
                for(uint32_t rr=0;rr<16;++rr)srcList[rr]=uint64_t(packed+(rb+rr)*WIDTH);
                #pragma unroll
                for(uint32_t j=0;j<8;++j){dstList[2*j]=uint64_t(matrix+j*columnPitch+rb);dstList[2*j+1]=uint64_t(matrix+j*columnPitch+rb+8);}
                TransDataTo5HD<float>(dstList,srcList,cv);
            }

    }

    __aicore__ inline GlobalTensor<int32_t> LowerSignalGM(uint32_t qoffset){
        GlobalTensor<int32_t> gm;gm.SetGlobalBuffer((__gm__ int32_t*)comm[96+qoffset*8].GetPhyAddr());return gm;
    }

    template<typename Op>
    __aicore__ inline void PublishSharedLower(Op &op){
        Fence<HardEvent::MTE3_S>();auto signal=op.SignalBuffer();signal.SetValue(0,1);
        Fence<HardEvent::S_MTE3>();DataCopy(LowerSignalGM(q),signal,8);
    }

protected:
    TBuf<TPosition::VECCALC> panelBuf,rawBuf,vecBuf,topBuf,idxBuf;
    LocalTensor<float> panel,raw,col,row,meta,temp,top;
    LocalTensor<uint32_t> idx;
    GlobalTensor<float> input,output,shared,comm,own,prodGm,allShared;
    LinalgDetConfig t;
    uint32_t q,first,group;
    float magnitude;
    bool ownRisk;
    float ownedM;
    int32_t ownedE;
};
