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
#include "det_lu_base.h"
class DetK64Base : public DetLUBase {
public:
    static constexpr uint32_t N=256,RP=104;
    __aicore__ inline void Mask(uint32_t n){NativeMask(n);}
__aicore__ inline void WaitUChunk(__gm__ int32_t *address,uint32_t rows){
    volatile __gm__ int32_t *progress=address;
    do{dcci((__gm__ uint64_t*)address,cache_line_t::SINGLE_CACHE_LINE,dcci_dst_t::CACHELINE_OUT);}
    while(progress[0]<int32_t(rows));
}
    template<uint32_t COUNT,uint32_t STRIDE>
    __aicore__ inline void MakeIndex(LocalTensor<int32_t> dst){
        auto p=(__ubuf__ int32_t*)dst.GetPhyAddr();
        #pragma unroll
        for(uint32_t j=0;j<8;++j)p[j]=int32_t(j*STRIDE);
        Fence<HardEvent::S_V>();
        #pragma unroll
        for(uint32_t step=8;step<COUNT;step*=2){
            Mask(step<64?step:64);
            vadds(p+step,p,int32_t(step*STRIDE),uint8_t((step+63)/64),1,1,8,8);
            PipeBarrier<PIPE_V>();
        }
        ResetMask();
    }
    __aicore__ inline void GatherPair(__ubuf__ float*d0,__ubuf__ float*d1,uint32_t base,uint32_t count){
        auto offsets=(__ubuf__ uint32_t*)idx.GetPhyAddr();const uint32_t full=count/64,rest=count%64;
        if(full){Mask(64);vgather((__ubuf__ uint32_t*)d0,offsets,base,8,uint8_t(full));vgather((__ubuf__ uint32_t*)d1,offsets,base+4,8,uint8_t(full));}
        if(rest){Mask(rest);vgather((__ubuf__ uint32_t*)(d0+full*64),offsets+full*64,base,8,1);vgather((__ubuf__ uint32_t*)(d1+full*64),offsets+full*64,base+4,8,1);}
    }
    template<bool HELPER> __aicore__ inline bool FinishK64(uint32_t batch){
        const bool safe=ReadRiskFlags(4,N*128+64);
        if(!safe)return false;
        if constexpr(!HELPER){if(q==0&&batch<t.batch){
            float m=1.0f;int32_t e=0;
            for(uint32_t c=0;c<4;++c){uint32_t at=(group*4+c)*8;m*=meta.GetValue(at);e+=meta.ReinterpretCast<int32_t>().GetValue(at+1);}
            WriteProduct(batch,m,e);
        }}
        return true;

    }

    __aicore__ inline void PublishTail(uint32_t rows,uint32_t pitch,uint32_t begin,LocalTensor<float> scratch){
        ResetMask();Fence<HardEvent::V_MTE3>();
        const uint32_t alignedRows=(rows+7)/8*8;
        DataCopyExtParams write{64,rows*4,(pitch-alignedRows)/8,(t.paddedN-rows)*4,0};
        DataCopyPad(shared[begin],scratch,write);

    }
    template<bool DIRECT,bool STREAMED>
    __aicore__ inline void FactorTailCommon(uint32_t start,uint32_t batch){
        TailLayout s;
        if(!SetupTail<64,true>(start,s))return;
        if(start==0&&t.batch==4&&t.groups==4){
            // Owner0 updates only its top64 s.rows. Lower input remains original A.
            DataCopyExtParams original{uint16_t(s.rows),256,(N-64)*4,0,0};
            DataCopyPadExtParams<float> noPad{false,0,0,0};
            DataCopyPad(s.scratch[s.packedOffset],input[uint64_t(batch)*N*N+s.begin*N],original,noPad);
        }else DataCopy(s.scratch[s.packedOffset],shared[64*t.paddedN+s.begin*64],s.rows*64);
        if constexpr(DIRECT||!STREAMED)DataCopy(top,shared[64*t.paddedN+start*64],64*64);
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::V_S>();SetMaskNorm();Mask(s.rows);
        // Every active row slice is 32 or 64 s.rows; FP32 16-row transposes.
        TransposePacked<64>(s.rows,s.columnPitch,s.packed,s.matrix);
        PipeBarrier<PIPE_V>();
        if constexpr(!DIRECT)Mask(s.rows);
        for(uint32_t i=0;i<64;i+=2){
            if constexpr(DIRECT)Mask(s.rows);
            if constexpr(STREAMED&&!DIRECT){if((i&15)==0){
                WaitUChunk((__gm__ int32_t*)shared[N*128].GetPhyAddr(),i+16);
                DataCopy(top[i*64],shared[64*t.paddedN+(start+i)*64],16*64);
                Fence<HardEvent::MTE2_S>();
            }}
            auto c0=s.matrix+i*s.columnPitch;auto c1=c0+s.columnPitch;
            const float a=s.u[i*64+i],b=s.u[i*64+i+1],c=s.u[(i+1)*64+i],d=s.u[(i+1)*64+i+1];
            const float inv=1.0f/(a*d-b*c);
            auto v0=s.lower+i*s.pitch;auto v1=v0+s.pitch;
            vmuls(v0,c0,float(-d*inv),1,1,1,8,8);vmuls(v1,c0,float(b*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
            vaxpy(v0,c1,float(c*inv),1,1,1,8,8);vaxpy(v1,c1,float(-a*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
            if(i+2<64){
                if((i&7)==0){
                    set_vector_mask(~uint64_t(0),~uint64_t(0));
                    vbrcb(s.expanded,(__ubuf__ uint32_t*)(s.u+i*64),1,8,i==56?48:64);
                    PipeBarrier<PIPE_V>();
                    if constexpr(!DIRECT)Mask(s.rows);
                }
                if constexpr(DIRECT)Mask(s.rows);
                vmla(s.matrix+(i+2)*s.columnPitch,v0,(__ubuf__ float*)s.expanded+((i&7)*64+i+2)*8,62-i,1,1,0,s.columnPitch/8,0,1);PipeBarrier<PIPE_V>();
                vmla(s.matrix+(i+2)*s.columnPitch,v1,(__ubuf__ float*)s.expanded+(((i&7)+1)*64+i+2)*8,62-i,1,1,0,s.columnPitch/8,0,1);PipeBarrier<PIPE_V>();
            }
        }
        PublishTail(s.rows,s.pitch,s.begin,s.scratch);}
    __aicore__ inline void InitK64Context(GM_ADDR workspace,const LinalgDetConfig &cfg){
        t=cfg;const uint32_t logical=uint32_t(GetBlockIdx())/2;
        q=logical&3;group=logical>>2;first=q*64;
        allShared.SetGlobalBuffer((__gm__ float*)workspace);
    }

    __aicore__ inline void InitK64Helper(GM_ADDR a){
            input.SetGlobalBuffer((__gm__ float*)a);
            raw=LocalTensor<float>(TPosition::VECCALC,0,12800);
            col=LocalTensor<float>(TPosition::VECCALC,12800*4,320);meta=col;
            top=LocalTensor<float>(TPosition::VECCALC,(12800+320)*4,4096);

    }

    __aicore__ inline GlobalTensor<int32_t> LowerSignalGM(uint32_t core){
        GlobalTensor<int32_t> signal;signal.SetGlobalBuffer((__gm__ int32_t*)shared[N*128+96+core*8].GetPhyAddr());return signal;
    }

    template<typename Op>
    __aicore__ inline void PublishK64LowerReady(Op &op){
        Fence<HardEvent::MTE3_S>();auto signal=op.SignalBuffer();signal.SetValue(0,1);
        Fence<HardEvent::S_MTE3>();DataCopy(LowerSignalGM(q),signal,8);
    }

};
