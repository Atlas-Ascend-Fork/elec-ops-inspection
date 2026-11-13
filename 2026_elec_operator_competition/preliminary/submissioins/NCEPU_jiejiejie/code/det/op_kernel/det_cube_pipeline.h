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
using namespace AscendC;

template<uint32_t K> class DetCubePipeline {
public:
    LocalTensor<float> a1,b1,a2,b2,c;
    __aicore__ inline DetCubePipeline():a1(TPosition::A1,0,256*K),b1(TPosition::B1,256*K*4,64*K),a2(TPosition::A2,0,256*K),b2(TPosition::B2,0,64*K),c(TPosition::CO1,0,256*64){
        SetHF32Mode(0);SetHF32TransMode(0);
        const uint8_t pad[4]={0,0,0,0};
        SetFmatrix(1,K,pad,FmatrixMode::FMATRIX_LEFT);
        SetFmatrix(1,K,pad,FmatrixMode::FMATRIX_RIGHT);
    }
    __aicore__ inline void Compute(uint32_t rows,uint32_t ma,GlobalTensor<float> prod){
            SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
            LoadData3DParamsV2Pro la{};la.channelSize=ma;
            la.extConfig=(uint64_t(K)<<16)|ma;la.enTranspose=true;
            LoadData(a2,a1,la);
            LoadData3DParamsV2Pro lb{};lb.channelSize=64;
            lb.extConfig=(uint64_t(K)<<16)|64;lb.fMatrixCtrl=true;
            LoadData(b2,b1,lb);
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
            MmadParams mp{};mp.m=rows;mp.n=64;mp.k=K;
            mp.cmatrixInitVal=true;mp.cmatrixSource=false;mp.unitFlag=3;mp.kDirectionAlign=1;
            Mmad(c,a2,b2,mp);
            // Unit flag3 waits on Cube completion; original9 protects all later reuse.
            FixpipeParamsV220 fp{};fp.mSize=rows;fp.nSize=64;fp.srcStride=ma;fp.dstStride=64;
            fp.quantPre=QuantMode_t::NoQuant;fp.unitFlag=3;
            Fixpipe<float,float,CFG_ROW_MAJOR>(prod,c,fp);
            CrossCoreSetFlag<2,PIPE_FIX>(9);
    }
};

__aicore__ inline void DetWaitLowerFlags(GlobalTensor<float> shared,uint32_t offset){
            volatile __gm__ int32_t *ready=(volatile __gm__ int32_t*)shared[offset].GetPhyAddr();
            bool complete=false;
            do{
                dcci((__gm__ uint64_t*)ready,cache_line_t::SINGLE_CACHE_LINE,dcci_dst_t::CACHELINE_OUT);
                dcci((__gm__ uint64_t*)(ready+16),cache_line_t::SINGLE_CACHE_LINE,dcci_dst_t::CACHELINE_OUT);
                complete=ready[0]==1&&ready[8]==1&&ready[16]==1&&ready[24]==1;
            }while(!complete);
}

__aicore__ inline Nd2NzParams DetNd2NzParams(uint32_t n,uint32_t d,uint32_t stride){
    Nd2NzParams cp{};cp.ndNum=1;cp.nValue=n;cp.dValue=d;
    cp.srcDValue=stride;cp.dstNzC0Stride=n;cp.dstNzNStride=1;return cp;
}
