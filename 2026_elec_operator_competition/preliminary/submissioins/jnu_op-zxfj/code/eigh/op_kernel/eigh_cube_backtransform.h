/*
 * Copyright (c) 2026 zjyzjy2002.
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
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "linalg_eigh_tiling.h"
using namespace AscendC;
using CubeBackA=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
using CubeBackBT=MatmulType<TPosition::GM,CubeFormat::ND,float,true>;
using CubeBackB=MatmulType<TPosition::GM,CubeFormat::ND,float,false>;
using CubeBackC=MatmulType<TPosition::GM,CubeFormat::ND,float>;
__aicore__ constexpr MatmulConfig BackConfig(uint32_t kind){
    auto c=GetNormalConfig(false,false,false,BatchMode::BATCH_LESS_THAN_L1,true,IterateOrder::ORDER_M,ScheduleType::INNER_PRODUCT,false,true);
    c.singleCoreM=kind==0?64:128;c.singleCoreN=kind==2?512:64;c.singleCoreK=kind==2?64:512;
    c.basicM=32;c.basicN=kind==2?64:32;c.basicK=64;c.enableSetBias=false;
    return c;
}
constexpr auto BACK_GRAM_CFG=BackConfig(0),BACK_PROJ_CFG=BackConfig(1),BACK_UPDATE_CFG=BackConfig(2);
constexpr auto BACK_GRAM_TILING=GetMatmulApiTiling<CubeBackA,CubeBackBT,CubeBackC,CubeBackC>(BACK_GRAM_CFG,131072);
constexpr auto BACK_PROJ_TILING=GetMatmulApiTiling<CubeBackA,CubeBackBT,CubeBackC,CubeBackC>(BACK_PROJ_CFG,131072);
constexpr auto BACK_UPDATE_TILING=GetMatmulApiTiling<CubeBackA,CubeBackB,CubeBackC,CubeBackC>(BACK_UPDATE_CFG,131072);
__aicore__ constexpr MatmulConfig NeumannConfig(){
    auto c=GetNormalConfig(false,false,false,BatchMode::BATCH_LESS_THAN_L1,true,IterateOrder::ORDER_M,ScheduleType::INNER_PRODUCT,false,true);
    c.singleCoreM=64;c.singleCoreN=64;c.singleCoreK=64;
    c.basicM=64;c.basicN=64;c.basicK=64;c.enableSetBias=false;return c;
}
constexpr auto NEUMANN_CFG=NeumannConfig();
constexpr auto NEUMANN_TILING=GetMatmulApiTiling<CubeBackA,CubeBackB,CubeBackC,CubeBackC>(NEUMANN_CFG,65536);
static_assert(NEUMANN_TILING.baseM==64 && NEUMANN_TILING.baseN==64 && NEUMANN_TILING.baseK==64);
static_assert(NEUMANN_TILING.singleCoreM==64 && NEUMANN_TILING.singleCoreN==64 && NEUMANN_TILING.singleCoreK==64);
template <typename Polynomial, typename Update>
__aicore__ inline void BuildPanelTransform(Polynomial &polynomial, Update &update, GlobalTensor<float> hist,
    GlobalTensor<float> gamma, uint32_t panel, const LinalgEighTilingData &data) {
    auto tensors=gamma[panel*6*64*64];
    auto pc=tensors[64*64];
    auto rc=tensors[3*64*64],rn=tensors[4*64*64];
    polynomial.SetOrgShape(64,64,64,64,64);polynomial.SetTail(64,64,64);
    // Five exact finite-product factors, no convergence shortcut.
    for(uint32_t level=0;level<5;++level){
        polynomial.SetTensorA(rc);polynomial.SetTensorB(rc,false);
        polynomial.IterateAll(rn);polynomial.End();PipeBarrier<PIPE_ALL>();
        // All old P elements are in the complete input tile
        // before Fixpipe atomically adds this tile's product.
        polynomial.SetTensorA(pc);polynomial.SetTensorB(rn,false);
        polynomial.IterateAll(pc,1);polynomial.End();PipeBarrier<PIPE_ALL>();
        auto rt=rc;rc=rn;rn=rt;
    }
    uint32_t low=panel*64,number=data.n-2-low;number=number<64?number:64;
    update.SetOrgShape(64,data.stride,64,64,data.stride);update.SetTail(number,data.stride,number);
    update.SetTensorA(pc);update.SetTensorB(hist[low*data.stride],false);
    update.IterateAll(hist[uint64_t(2*data.n+5)*data.stride+low*data.stride]);update.End();PipeBarrier<PIPE_ALL>();
}
template <typename Gram>
__aicore__ inline void ComputeGramPanels(Gram &gram, GlobalTensor<float> hist,
    GlobalTensor<float> gamma, const LinalgEighTilingData &data, uint32_t pair,
    uint32_t pairCount, uint32_t panels) {
    for (uint32_t panel = pair; panel < panels; panel += pairCount) {
        uint32_t low = panel * 64;
        uint32_t number = data.n - 2 - low;
        number = number < 64 ? number : 64;
        gram.SetOrgShape(64, 64, data.stride, data.stride, 64);
        gram.SetTail(number, number, data.stride);
        gram.SetTensorA(hist[low * data.stride]);
        gram.SetTensorB(hist[low * data.stride], true);
        gram.IterateAll(gamma[panel * 6 * 64 * 64]);
        gram.End();
        PipeBarrier<PIPE_ALL>();
    }
}
__aicore__ inline void CubeBackProcess(GM_ADDR workspace,LinalgEighTilingData data){
    uint32_t padded=(data.rows+7)/8*8,physicalTeam=(data.team+1)/2*2;
    if(!(data.n>25 && data.stride>=64 && data.stride%64==0 && padded>=16))return;
    TPipe cubePipe;
    MatmulImpl<CubeBackA,CubeBackBT,CubeBackC,CubeBackC,BACK_GRAM_TILING> gram;
    MatmulImpl<CubeBackA,CubeBackBT,CubeBackC,CubeBackC,BACK_PROJ_TILING> projection;
    MatmulImpl<CubeBackA,CubeBackB,CubeBackC,CubeBackC,BACK_UPDATE_TILING> update;
    MatmulImpl<CubeBackA,CubeBackB,CubeBackC,CubeBackC,NEUMANN_TILING> polynomial;
    gram.Init(static_cast<const TCubeTiling*>(nullptr),&cubePipe);
    projection.Init(static_cast<const TCubeTiling*>(nullptr),&cubePipe);
    update.Init(static_cast<const TCubeTiling*>(nullptr),&cubePipe);
    polynomial.Init(static_cast<const TCubeTiling*>(nullptr),&cubePipe);
    gram.SetHF32(false);projection.SetHF32(false);update.SetHF32(false);polynomial.SetHF32(false);
    uint32_t pairCount=physicalTeam/2,group=GetBlockIdx()/pairCount,pair=GetBlockIdx()%pairCount;
    uint32_t height=2*padded,first=(data.n-3)/64*64,panels=(data.n-2+63)/64;
    uint64_t perMatrix=uint64_t(3*data.n+5)*data.stride+uint64_t(physicalTeam)*padded*(data.stride+64)+uint64_t(panels)*6*64*64;
    GlobalTensor<float> work;work.SetGlobalBuffer((__gm__ float*)workspace);
    for(uint32_t step=0;step<(data.batch+data.groups-1)/data.groups;++step){
        uint32_t b=step*data.groups+group;
        if(b<data.batch){
            auto hist=work[uint64_t(b)*perMatrix];
            auto gamma=hist[uint64_t(3*data.n+5)*data.stride+uint64_t(physicalTeam)*padded*(data.stride+64)];
            CrossCoreWaitFlag(8);
            ComputeGramPanels(gram, hist, gamma, data, pair, pairCount, panels);
            CrossCoreSetFlag<2,PIPE_FIX>(9);
            CrossCoreWaitFlag(8);
            for (uint32_t panel = pair; panel < panels; panel += pairCount) {
                BuildPanelTransform(polynomial, update, hist, gamma, panel, data);
            }
        }
        // Native AIC-only barrier: every launched AIC, including an empty
        // final group, finishes panel construction before any Q reads Z.
        CrossCoreSetFlag<0,PIPE_FIX>(7);CrossCoreWaitFlag(7);
        if(b>=data.batch)continue;
        auto hist=work[uint64_t(b)*perMatrix],z=hist[uint64_t(2*data.n+5)*data.stride];
        auto q=hist[uint64_t(3*data.n+5)*data.stride+uint64_t(pair)*height*data.stride];
        auto y=hist[uint64_t(3*data.n+5)*data.stride+uint64_t(physicalTeam)*padded*data.stride+uint64_t(pair)*height*64];
        CrossCoreWaitFlag(10);
        for(int low=int(first);low>=0;low-=64){
            uint32_t number=data.n-2-uint32_t(low);number=number<64?number:64;
            projection.SetOrgShape(height,64,data.stride,data.stride,64);projection.SetTail(height,number,data.stride);
            projection.SetTensorA(q);projection.SetTensorB(z[uint32_t(low)*data.stride],true);
            projection.IterateAll(y);projection.End();PipeBarrier<PIPE_ALL>();
            update.SetOrgShape(height,data.stride,64,64,data.stride);update.SetTail(height,data.stride,number);
            update.SetTensorA(y);update.SetTensorB(hist[uint32_t(low)*data.stride],false);
            update.IterateAll(q,1);update.End();PipeBarrier<PIPE_ALL>();
        }
        CrossCoreSetFlag<2,PIPE_FIX>(9);
        CrossCoreWaitFlag(6);
    }
}
__aicore__ constexpr MatmulConfig BackConfig16(bool projection){
    auto c=GetNormalConfig(false,false,false,BatchMode::BATCH_LESS_THAN_L1,true,IterateOrder::ORDER_M,ScheduleType::INNER_PRODUCT,false,true);
    c.singleCoreM=128;c.singleCoreN=projection?16:512;c.singleCoreK=projection?512:16;
    c.basicM=32;c.basicN=projection?16:64;c.basicK=projection?64:16;c.enableSetBias=false;
    return c;
}
constexpr auto BACK16_PROJ_CFG=BackConfig16(true),BACK16_UPDATE_CFG=BackConfig16(false);
constexpr auto BACK16_PROJ_TILING=GetMatmulApiTiling<CubeBackA,CubeBackBT,CubeBackC,CubeBackC>(BACK16_PROJ_CFG);
constexpr auto BACK16_UPDATE_TILING=GetMatmulApiTiling<CubeBackA,CubeBackB,CubeBackC,CubeBackC>(BACK16_UPDATE_CFG);
__aicore__ inline void CubeBackProcess16(GM_ADDR workspace,LinalgEighTilingData data){
    uint32_t padded=(data.rows+7)/8*8,physicalTeam=(data.team+1)/2*2;
    if(!(data.n>25 && data.stride>=64 && data.stride%64==0 && padded>=16))return;
    TPipe cubePipe;
    MatmulImpl<CubeBackA,CubeBackBT,CubeBackC,CubeBackC,BACK16_PROJ_TILING> projection;
    MatmulImpl<CubeBackA,CubeBackB,CubeBackC,CubeBackC,BACK16_UPDATE_TILING> update;
    projection.Init(static_cast<const TCubeTiling*>(nullptr),&cubePipe);
    update.Init(static_cast<const TCubeTiling*>(nullptr),&cubePipe);
    projection.SetHF32(false);update.SetHF32(false);
    uint32_t pairCount=physicalTeam/2,group=GetBlockIdx()/pairCount,pair=GetBlockIdx()%pairCount;
    uint32_t height=2*padded,first=(data.n-3)/16*16;
    uint64_t perMatrix=uint64_t(3*data.n+5)*data.stride+uint64_t(physicalTeam)*padded*(data.stride+16);
    GlobalTensor<float> work;work.SetGlobalBuffer((__gm__ float*)workspace);
    for(uint32_t step=0;step<(data.batch+data.groups-1)/data.groups;++step){
        uint32_t b=step*data.groups+group;if(b>=data.batch)continue;
        auto hist=work[uint64_t(b)*perMatrix],z=hist[uint64_t(2*data.n+5)*data.stride];
        auto q=hist[uint64_t(3*data.n+5)*data.stride+uint64_t(pair)*height*data.stride];
        auto y=hist[uint64_t(3*data.n+5)*data.stride+uint64_t(physicalTeam)*padded*data.stride+uint64_t(pair)*height*16];
        CrossCoreWaitFlag(10);
        for(int low=int(first);low>=0;low-=16){
            uint32_t number=data.n-2-uint32_t(low);number=number<16?number:16;
            projection.SetOrgShape(height,16,data.stride,data.stride,16);projection.SetTail(height,number,data.stride);
            projection.SetTensorA(q);projection.SetTensorB(z[uint32_t(low)*data.stride],true);
            projection.IterateAll(y);projection.End();PipeBarrier<PIPE_ALL>();
            update.SetOrgShape(height,data.stride,16,16,data.stride);update.SetTail(height,data.stride,number);
            update.SetTensorA(y);update.SetTensorB(hist[uint32_t(low)*data.stride],false);
            update.IterateAll(q,1);update.End();PipeBarrier<PIPE_ALL>();
        }
        CrossCoreSetFlag<2,PIPE_FIX>(9);
        CrossCoreWaitFlag(6);
    }
}
