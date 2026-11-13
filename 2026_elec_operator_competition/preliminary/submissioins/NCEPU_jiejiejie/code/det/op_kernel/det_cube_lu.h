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
#include "det_lu_base.h"
#include "det_cube_pipeline.h"
#include "lib/matmul_intf.h"
#include "linalg_det_tiling.h"
#include "tiling_key_linalg_det.h"
using namespace AscendC;
using namespace matmul;
using AB = MatmulType<TPosition::GM, CubeFormat::ND, float>;
__aicore__ constexpr MatmulConfig DetFixedCubeConfig(){
    auto cfg=GetNormalConfig();
    cfg.singleCoreM=256;cfg.basicM=256;cfg.singleCoreN=64;cfg.basicN=64;cfg.singleCoreK=32;cfg.basicK=32;
    cfg.enableSetBias=false;cfg.enableQuantVector=false;cfg.enableSetDefineData=false;
    cfg.iterateMode=IterateMode::ITERATE_MODE_ALL;return cfg;
}
constexpr MatmulConfig DET_FIXED_CUBE_CFG=DetFixedCubeConfig();

class CubeLUVector : public DetLUBase {
public:
    __aicore__ inline CubeLUVector(TPipe &external):pipe(external){}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR det, GM_ADDR ws,
                                const LinalgDetConfig &cfg) {
        t=cfg;
        const uint32_t logical=GetBlockIdx()%t.coresPerMatrix;
        q=logical;
        group=GetBlockIdx()/t.coresPerMatrix; first=q*64;
        InitBuffers(pipe,A,det,ws);
}

    template<bool COOPERATIVE> __aicore__ inline void Factor(uint32_t start){
        FactorCooperative<COOPERATIVE>(start);
    }

    __aicore__ inline void FactorTail(uint32_t start){
        TailLayout s;
        if(!SetupTail<32,false>(start,s))return;
        DataCopy(s.scratch[s.packedOffset],shared[32*t.paddedN+s.begin*32],s.rows*32);
        DataCopy(top,shared[32*t.paddedN+start*32],32*32);
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::V_S>();SetMaskNorm();NativeMask(s.rows);
        for(uint32_t c=0;c<32;++c)
            vgather((__ubuf__ uint32_t*)(s.matrix+c*s.columnPitch),s.offsets,uint32_t(uint64_t(s.packed)+c*4),8,1);
        PipeBarrier<PIPE_V>();
        for(uint32_t i=0;i<32;i+=2){
            NativeMask(s.rows);
            auto c0=s.matrix+i*s.columnPitch;auto c1=c0+s.columnPitch;
            const float a=s.u[i*32+i],b=s.u[i*32+i+1],c=s.u[(i+1)*32+i],d=s.u[(i+1)*32+i+1];
            const float inv=1.0f/(a*d-b*c);
            auto v0=s.lower+i*s.pitch;auto v1=v0+s.pitch;
            vmuls(v0,c0,float(-d*inv),1,1,1,8,8);vmuls(v1,c0,float(b*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
            vaxpy(v0,c1,float(c*inv),1,1,1,8,8);vaxpy(v1,c1,float(-a*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
            if(i+2<32){
                set_vector_mask(~uint64_t(0),~uint64_t(0));
                vbrcb(s.expanded,(__ubuf__ uint32_t*)(s.u+i*32),1,8,4);
                vbrcb(s.expanded+256,(__ubuf__ uint32_t*)(s.u+(i+1)*32),1,8,4);PipeBarrier<PIPE_V>();
                NativeMask(s.rows);
                vmla(s.matrix+(i+2)*s.columnPitch,v0,(__ubuf__ float*)s.expanded+(i+2)*8,30-i,1,1,0,s.columnPitch/8,0,1);PipeBarrier<PIPE_V>();
                vmla(s.matrix+(i+2)*s.columnPitch,v1,(__ubuf__ float*)s.expanded+256+(i+2)*8,30-i,1,1,0,s.columnPitch/8,0,1);PipeBarrier<PIPE_V>();
            }
        }
        ResetMask();Fence<HardEvent::V_MTE3>();
        const uint32_t alignedRows=(s.rows+7)/8*8;
        DataCopyExtParams write{32,s.rows*4,(s.pitch-alignedRows)/8,(t.paddedN-s.rows)*4,0};
        DataCopyPad(shared[s.begin],s.scratch,write);
    }

    __aicore__ inline bool ProcessOne(uint32_t batch){
        if(batch>=t.groups){CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
        LoadDynamicPanel<true>(batch);
        magnitude=1.0f;
        float m=1.0f;int32_t e=0;ownRisk=false;
        // Finish the final 64 columns locally when the matrix ends on a 64-column boundary.
        ProcessDynamicPanels<false>(*this,m,e);
        if(!ReadRiskFlags(t.coresPerMatrix,t.paddedN*64+64))return false;
        if(q==0&&batch<t.batch)WriteProduct(batch,m,e);
        return true;
    }
private:
    TPipe &pipe;

};

class CubeLUVectorHelpers : public DetLUBase {
public:
    __aicore__ inline CubeLUVectorHelpers(TPipe &external):pipe(external){}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR det, GM_ADDR ws,
                                const LinalgDetConfig &cfg) {
        t=cfg;
        const uint32_t logical=(GetBlockIdx()/2)%t.coresPerMatrix;
        q=logical;
        group=GetBlockIdx()/(2*t.coresPerMatrix); first=q*64;
        if(GetBlockIdx()&1){
            // Helpers only transpose/factor the lower rows and inspect risk flags.
            allShared.SetGlobalBuffer((__gm__ float*)ws);
            shared=allShared[group*t.robustOffset];comm=shared[t.paddedN*64];
            pipe.InitBuffer(rawBuf,6912*4);raw=rawBuf.Get<float>();
            pipe.InitBuffer(vecBuf,(256+64)*4);col=vecBuf.Get<float>();meta=col;temp=col[256];
            pipe.InitBuffer(topBuf,32*32*4);top=topBuf.Get<float>();
            return;
        }

        InitBuffers(pipe,A,det,ws);
}

    template<bool COOPERATIVE> __aicore__ inline void Factor(uint32_t start){
        FactorCooperative<COOPERATIVE>(start);
        if constexpr(COOPERATIVE){
            auto signal=SignalBuffer();Duplicate(signal,0,32);Fence<HardEvent::V_MTE3>();
            DataCopy(LowerSignalGM(0),signal,32);
        }
    }

    __aicore__ inline void TransposeHelperTail(uint32_t rows,uint32_t columnPitch,__ubuf__ float *packed,__ubuf__ float *matrix,__ubuf__ uint32_t *offsets){
if((t.n==256||t.n==200)&&rows%16==0){TransposePacked<32>(rows,columnPitch,packed,matrix);}else{
            for(uint32_t c=0;c<32;++c)
                vgather((__ubuf__ uint32_t*)(matrix+c*columnPitch),offsets,uint32_t(uint64_t(packed)+c*4),8,1);
        }
    }
    __aicore__ inline void FactorTail(uint32_t start){
        TailLayout s;
        if(!SetupTail<32,true>(start,s))return;
        DataCopy(s.scratch[s.packedOffset],shared[32*t.paddedN+s.begin*32],s.rows*32);
        DataCopy(top,shared[32*t.paddedN+start*32],32*32);
        Fence<HardEvent::MTE2_V>();Fence<HardEvent::V_S>();SetMaskNorm();NativeMask(s.rows);
                TransposeHelperTail(s.rows,s.columnPitch,s.packed,s.matrix,s.offsets);

        PipeBarrier<PIPE_V>();
        NativeMask(s.rows);
        for(uint32_t i=0;i<32;i+=2){
            auto c0=s.matrix+i*s.columnPitch;auto c1=c0+s.columnPitch;
            const float a=s.u[i*32+i],b=s.u[i*32+i+1],c=s.u[(i+1)*32+i],d=s.u[(i+1)*32+i+1];
            const float inv=1.0f/(a*d-b*c);
            auto v0=s.lower+i*s.pitch;auto v1=v0+s.pitch;
            vmuls(v0,c0,float(-d*inv),1,1,1,8,8);vmuls(v1,c0,float(b*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
            vaxpy(v0,c1,float(c*inv),1,1,1,8,8);vaxpy(v1,c1,float(-a*inv),1,1,1,8,8);PipeBarrier<PIPE_V>();
            if(i+2<32){
                if((i&7)==0){
                    set_vector_mask(~uint64_t(0),~uint64_t(0));
                    // Eight s.rows for0/8/16; only six still-used s.rows for24.
                    vbrcb(s.expanded,(__ubuf__ uint32_t*)(s.u+i*32),1,8,i==24?24:32);
                    PipeBarrier<PIPE_V>();
                    NativeMask(s.rows);
                }
                vmla(s.matrix+(i+2)*s.columnPitch,v0,(__ubuf__ float*)s.expanded+((i&7)*32+i+2)*8,30-i,1,1,0,s.columnPitch/8,0,1);PipeBarrier<PIPE_V>();
                vmla(s.matrix+(i+2)*s.columnPitch,v1,(__ubuf__ float*)s.expanded+(((i&7)+1)*32+i+2)*8,30-i,1,1,0,s.columnPitch/8,0,1);PipeBarrier<PIPE_V>();
            }
        }
        ResetMask();Fence<HardEvent::V_MTE3>();
        const uint32_t alignedRows=(s.rows+7)/8*8;
        DataCopyExtParams write{32,s.rows*4,(s.pitch-alignedRows)/8,(t.paddedN-s.rows)*4,0};
        DataCopyPad(shared[s.begin],s.scratch,write);
    }

    __aicore__ inline LocalTensor<int32_t> SignalBuffer(){
        // Helper meta reduction uses at most160 floats;192..223 are disjoint.
        if(GetBlockIdx()&1)return col[192].ReinterpretCast<int32_t>();
        return col[4*((t.paddedN+63)/64*64)].ReinterpretCast<int32_t>();
    }

    __aicore__ inline void PublishLowerReady(){
        PublishSharedLower(*this);
    }
    __aicore__ inline void WaitLowerReady(){
        WaitSharedLower(*this);
    }
    __aicore__ inline bool ProcessOne(uint32_t batch){
        if(batch>=t.groups){CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
        LoadDynamicPanel<true>(batch);
        magnitude=1.0f;
        float m=1.0f;int32_t e=0;ownRisk=false;
        // Finish the final 64 columns locally when the matrix ends on a 64-column boundary.
        ProcessDynamicPanels<true>(*this,m,e);
        if(!ReadRiskFlags(t.coresPerMatrix,t.paddedN*64+64))return false;
        if(q==0&&batch<t.batch)WriteProduct(batch,m,e);
        return true;
    }
    // Paired helper for n200: all four K32 prefix stages cooperate.
    __aicore__ inline bool ProcessHelper200(uint32_t batch){
        if(batch>=t.groups){CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
        CreateVecIndex(temp.ReinterpretCast<int32_t>(),0,64);PipeBarrier<PIPE_V>();
        Muls(temp.ReinterpretCast<int32_t>(),temp.ReinterpretCast<int32_t>(),int32_t(128),64);PipeBarrier<PIPE_V>();
        for(uint32_t start=0;start<=128;start+=32){
            const uint32_t owner=start/64;
            shared=allShared[group*t.robustOffset+(start/32%2)*(t.paddedN*64+128)];comm=shared[t.paddedN*64];
            if(start==128){CrossCoreSetFlag<0,PIPE_MTE3>(12);CrossCoreWaitFlag(12);}
            CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);
            if(start==128)break;
            if(start<128){FactorTail(start);PublishLowerReady();}
            const bool cubeNeeded=q>owner||(q==owner&&start%64==0);
            if(cubeNeeded){
                CrossCoreSetFlag<2,PIPE_MTE3>(11);
                CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);
            }
        }
        return ReadRiskFlags(t.coresPerMatrix,t.paddedN*64+64);
    }
    __aicore__ inline bool ProcessHelper(uint32_t batch){
        if(batch>=t.groups){CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
        CreateVecIndex(temp.ReinterpretCast<int32_t>(),0,64);PipeBarrier<PIPE_V>();
        Muls(temp.ReinterpretCast<int32_t>(),temp.ReinterpretCast<int32_t>(),int32_t(128),64);PipeBarrier<PIPE_V>();
        for(uint32_t start=0;start<t.n;start+=(t.n==256&&start==192)?64:32){
            const uint32_t owner=start/64;
            shared=allShared[group*t.robustOffset+(start/32%2)*(t.paddedN*64+128)];comm=shared[t.paddedN*64];
            const bool cooperative=start+32<t.n&&t.n-start-32>=t.coresPerMatrix*32;
            CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);
            if(start+32<t.n&&!(t.n==256&&start==192)){
                if(cooperative){
                    FactorTail(start);
                    PublishLowerReady();
                }
                const bool cubeNeeded=q>owner||(q==owner&&start%64==0);
                // Both AIV subblocks must publish11 before the paired Cube proceeds.
                if(t.n==256&&!cooperative&&cubeNeeded)CrossCoreSetFlag<2,PIPE_MTE3>(11);
                if(cubeNeeded){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);}
            }
        }
        return ReadRiskFlags(t.coresPerMatrix,t.paddedN*64+64);
    }
private:
    TPipe &pipe;

};

class CubeLUSequentialVector : public DetLUBase {
public:
    __aicore__ inline CubeLUSequentialVector(TPipe &external):pipe(external){}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR det, GM_ADDR ws,
                                const LinalgDetConfig &cfg) {
        t=cfg;
        const uint32_t logical=GetBlockIdx()%t.coresPerMatrix;
        q=logical;
        group=GetBlockIdx()/t.coresPerMatrix; first=q*64;
        InitBuffers(pipe,A,det,ws);
}

    __aicore__ inline void Factor(uint32_t start){
        FactorCooperative<false,false,true>(start);
    }

    __aicore__ inline bool ProcessOne(uint32_t batch){
        if(batch>=t.groups){CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
        LoadDynamicPanel<false>(batch);
        magnitude=1.0f;
        float m=1.0f;int32_t e=0;ownRisk=false;
        for(uint32_t start=0;start<t.n;start+=32){
            const uint32_t owner=start/64;
            shared=allShared[group*t.robustOffset+(start/32%2)*(t.paddedN*64+128)];comm=shared[t.paddedN*64];
            if(q==owner)Factor(start);
            {CrossCoreSetFlag<0,PIPE_MTE3>(10);CrossCoreWaitFlag(10);}
            if(q==0){
                if(q!=owner){DataCopy(meta,comm,8);Fence<HardEvent::MTE2_S>();}
                e+=meta.ReinterpretCast<int32_t>().GetValue(1);m*=meta.GetValue(0);
            }
            if(start+32<t.n){
                if(q>owner)Solve(start);
                const bool cubeNeeded=q>owner||(q==owner&&start%64==0);
                if(cubeNeeded){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);Update(start);}
            }
        }
        if(!ReadRiskFlags(t.coresPerMatrix,t.paddedN*64+64))return false;
        if(q==0&&batch<t.batch)WriteProduct(batch,m,e);
        return true;
    }
private:
    TPipe &pipe;

};

template<const MatmulConfig &CFG> __aicore__ inline void RunLUCubeImpl(const LinalgDetTilingData &t, GM_ADDR ws){
    if ASCEND_IS_AIC {
        TPipe pipe;MatmulImpl<AB,AB,AB,AB,CFG> mm;mm.SetSubBlockIdx(0);mm.Init(&t.cube,&pipe);mm.SetHF32(false);mm.SetOrgShape(t.paddedN,64,32);
        GlobalTensor<float> shared,own,prod;
        const uint32_t q=GetBlockIdx()%t.coresPerMatrix,group=GetBlockIdx()/t.coresPerMatrix;
        own.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset+2*(t.paddedN*64+128)+q*(t.paddedN*64+8192));prod=own[4096];
        for(uint32_t batch=0;batch<(t.batch+t.groups-1)/t.groups;++batch)
          for(uint32_t start=0;start+32<t.n&&!(t.n==256&&start==192)&&!(t.n==200&&start>=128)&&start<=q*64;start+=32){
            shared.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset+(start/32%2)*(t.paddedN*64+128));
            CrossCoreWaitFlag(8);
            mm.SetTail(t.n-start-32,64,32);mm.SetTensorA(shared[start+32],true);mm.SetTensorB(own);mm.IterateAll(prod);mm.End();
            CrossCoreSetFlag<2,PIPE_FIX>(9);
          }
    }
}

// CANN9/arch2201 implementation of early GM->L1 A loading.
// One buffer is reused only after the prior Fixpipe/Vector update/global barrier.
__aicore__ inline void RunLUCubePrefetch200(const LinalgDetTilingData &t,GM_ADDR ws){
    if ASCEND_IS_AIC {
        // Basic API matches the qualified SDK9 Matmul Mmad/Fixpipe control fields.
        TPipe pipe;
        DetCubePipeline<32> cube;
        GlobalTensor<float> shared,own,prod;
        const uint32_t q=GetBlockIdx()%4,group=GetBlockIdx()/4;
        own.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset+2*(t.paddedN*64+128)+q*(t.paddedN*64+8192));prod=own[4096];
        for(uint32_t batch=0;batch<(t.batch+t.groups-1)/t.groups;++batch)
          for(uint32_t start=0;start<128&&start<=q*64;start+=32){
            const uint32_t rows=200-start-32,ma=(rows+15)/16*16;
            shared.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset+(start/32%2)*(t.paddedN*64+128));
            CrossCoreWaitFlag(11);
            // Keep256's paired11 and allfour volatile helper-ready reads.
            DetWaitLowerFlags(shared,t.paddedN*64+96);
            auto ap=DetNd2NzParams(32,rows,t.paddedN);
            InitConstValue(cube.a1[rows*32],{1,32,0,0U});
            DataCopy(cube.a1,shared[start+32],ap);
            CrossCoreWaitFlag(8);
            auto bp=DetNd2NzParams(32,64,64);
            DataCopy(cube.b1,own,bp);
            cube.Compute(rows,ma,prod);
          }
        PipeBarrier<PIPE_ALL>();
    }
}

// Early A is legal only at starts128/160; earlier A still depends on all helpers.
__aicore__ inline void RunLUCubePrefetch256(const LinalgDetTilingData &t,GM_ADDR ws){
    if ASCEND_IS_AIC {
        // Basic API matches the qualified SDK9 Matmul Mmad/Fixpipe control fields.
        TPipe pipe;
        DetCubePipeline<32> cube;
        GlobalTensor<float> shared,own,prod;
        const uint32_t q=GetBlockIdx()%4,group=GetBlockIdx()/4;
        own.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset+2*(256*64+128)+q*(256*64+8192));prod=own[4096];
        for(uint32_t batch=0;batch<(t.batch+t.groups-1)/t.groups;++batch)
          for(uint32_t start=0;start<192&&start<=q*64;start+=32){
            const uint32_t rows=256-start-32;
            shared.SetGlobalBuffer((__gm__ float*)ws+group*t.robustOffset+(start/32%2)*(256*64+128));
            auto ap=DetNd2NzParams(32,rows,256);
            auto bp=DetNd2NzParams(32,64,64);
            if(start>=128){
                CrossCoreWaitFlag(11);DataCopy(cube.a1,shared[start+32],ap);
                CrossCoreWaitFlag(8);DataCopy(cube.b1,own,bp);
            }else{
                // Paired8 orders this core's complete upperB and local helper.
                CrossCoreWaitFlag(8);DataCopy(cube.b1,own,bp);
                // Volatile reads must remain inside the poll: ordinary GetValue
                // loads were hoisted before the DCCI loop by the compiler.
                DetWaitLowerFlags(shared,256*64+96);
                DataCopy(cube.a1,shared[start+32],ap);
            }
            cube.Compute(rows,rows,prod);
          }
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void RunLUCube(const LinalgDetTilingData &t,GM_ADDR ws){
    if(t.n==256&&t.coresPerMatrix==4){RunLUCubePrefetch256(t,ws);return;}
    if(t.n==200&&t.coresPerMatrix==4){RunLUCubePrefetch200(t,ws);return;}
    if(t.cubeMode==2)RunLUCubeImpl<DET_FIXED_CUBE_CFG>(t,ws);
    else RunLUCubeImpl<CFG_NORM>(t,ws);
}
