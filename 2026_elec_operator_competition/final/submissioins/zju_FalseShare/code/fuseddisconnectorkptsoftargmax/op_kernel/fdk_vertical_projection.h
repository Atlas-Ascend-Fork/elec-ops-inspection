/*
 * Copyright (c) 2026 FalseShare Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Apache License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "kernel_operator.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "fdk_cube_statistics.h"
using namespace AscendC;

// Exact reassociation: Q[k,dx,y,x] = sum(dy,c) W[k,c,dy,dx]*F[c,y+dy-1,x].
// One K96 FP32 MMAD emits twelve Q columns. AIV shifts each dx and sums.
class FdkVerticalProjection {
public:
 __aicore__ inline void Init(GM_ADDR feat,GM_ADDR weight,GM_ADDR bias,GM_ADDR kp,GM_ADDR score,
                           const FusedDisconnectorKeypointSoftArgmaxTilingData& t,GM_ADDR tiling) {
  h=t.height;w=t.width;full=h*w;batch=t.batch;parts=t.parts;blocks=t.blocks;
  group=GetBlockIdx()/GetTaskRation();sub=GetSubBlockIdx();part=group%parts;b=group/parts;
  R=t.tileRows;first=(part*(h/R)/parts)*R;rows=(((part+1)*(h/R)/parts)*R)-first;P=R*w;N=rows*w;halo=(rows+2)*w;ringN=12*P;tau=t.tau;stride=t.stride;
  auto base=GetSysWorkSpacePtr();
  fg.SetGlobalBuffer((__gm__ float*)feat);wg.SetGlobalBuffer((__gm__ float*)weight);
  bg.SetGlobalBuffer((__gm__ float*)bias);kg.SetGlobalBuffer((__gm__ float*)kp);sg.SetGlobalBuffer((__gm__ float*)score);
  ring.SetGlobalBuffer((__gm__ float*)base+group*ringN);
  stats.SetGlobalBuffer((__gm__ float*)base+blocks*ringN);
  map.SetGlobalBuffer((__gm__ uint32_t*)tiling+8);
  if ASCEND_IS_AIV {
   pipe.InitBuffer(rawBuf,(6*P+16)*4);pipe.InitBuffer(accBuf,2*N*4);
   pipe.InitBuffer(tmpBuf,(N>6*P?N:6*P)*4);pipe.InitBuffer(scratchBuf,N*4);
   pipe.InitBuffer(coordBuf,2*N*4);pipe.InitBuffer(indexBuf,(6*P>576?6*P:576)*4);
   pipe.InitBuffer(smallBuf,64*4);pipe.InitBuffer(statsBuf,parts*32*4);
  } else {
   pipe.InitBuffer(inputBuf,32*halo*4);pipe.InitBuffer(weightBuf,16*96*4);
   pipe.InitBuffer(aBuf,16*96*4);pipe.InitBuffer(bBuf,96*P*4);pipe.InitBuffer(cBuf,2*16*P*4);
  }
 }
 __aicore__ inline void Process() {
  if ASCEND_IS_AIC { Cube(); }
  else {
   PackWeights();CrossCoreSetFlag<2,PIPE_MTE3>(0);
   if(batch==1)DataCopyPad(smallBuf.Get<float>()[32],bg,DataCopyExtParams{1,16,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
   auto ix=indexBuf.Get<uint32_t>();
   for(uint32_t k=0;k<2;++k)for(uint32_t dx=0;dx<3;++dx) {
    uint32_t off=(k*3+dx)*P;
    CreateVecIndex(ix[off].ReinterpretCast<int32_t>(),static_cast<int32_t>(8+off+dx-1),P);
    PipeBarrier<PIPE_V>();Muls(ix[off].ReinterpretCast<int32_t>(),ix[off].ReinterpretCast<int32_t>(),int32_t(4),P);
   }
   PipeBarrier<PIPE_ALL>();
   // Route all horizontal zero-padding entries to the known-zero prefix.
   for(uint32_t k=0;k<2;++k) {
    for(uint32_t y=0;y<R;++y) {
     ix.SetValue(k*3*P+y*w,0);ix.SetValue((k*3+2)*P+(y+1)*w-1,0);
    }
   }
   PipeBarrier<PIPE_ALL>();
   auto coord=coordBuf.Get<float>();
   CreateVecIndex(coord,0.0f,w);PipeBarrier<PIPE_ALL>();
   for(uint32_t y=1;y<rows;++y)DataCopy(coord[y*w],coord,w);
   for(uint32_t y=0;y<rows;++y)Duplicate(coord[N+y*w],static_cast<float>(static_cast<int32_t>(first+y)),w);
   PipeBarrier<PIPE_ALL>();
   auto raw=rawBuf.Get<float>();auto tmp=tmpBuf.Get<float>();auto acc=accBuf.Get<float>();
   Duplicate(raw,0.0f,8);PipeBarrier<PIPE_ALL>();
   for(uint32_t r=0;r<rows;r+=R) {
    CrossCoreWaitFlag<2,PIPE_MTE2>(1);
    DataCopy(raw[8],ring[sub*6*P],6*P);
    CrossCoreSetFlag<2,PIPE_MTE2>(2);
    PipeBarrier<PIPE_ALL>();
    Gather(tmp,raw,ix,0,6*P);PipeBarrier<PIPE_V>();
    for(uint32_t k=0;k<2;++k) {
     auto dst=acc[k*N+r*w];
     Add(dst,tmp[k*3*P],tmp[(k*3+1)*P],P);PipeBarrier<PIPE_V>();
     Add(dst,dst,tmp[(k*3+2)*P],P);PipeBarrier<PIPE_V>();
    }
   }
   Statistics();SyncAll();
   if(part==0)Merge();
  }
 }
private:
 __aicore__ inline void PackWeights() {
  auto raw=rawBuf.Get<float>();auto tmp=tmpBuf.Get<float>();auto ix=indexBuf.Get<uint32_t>();
  DataCopy(raw,wg[sub*576],576);DataCopy(ix,map,576);PipeBarrier<PIPE_ALL>();
  Gather(tmp,raw,ix,0,576);PipeBarrier<PIPE_ALL>();DataCopy(ring[sub*576],tmp,576);PipeBarrier<PIPE_ALL>();
 }
 __aicore__ inline void Cube() {
  SetHF32Mode(false);SetHF32TransMode(false);
  auto x=inputBuf.Get<float>();auto wt=weightBuf.Get<float>();
  auto a=aBuf.Get<float>();auto xb=bBuf.Get<float>();auto c=cBuf.Get<float>();
  uint32_t start=first?first-1:0,top=first?0:w;
  uint32_t end=first+rows<h?first+rows+1:h,valid=(end-start)*w;
  if(top)InitConstValue(x,InitConstValueParams<float>{1,static_cast<uint16_t>(32*w/8),0,0.0f});
  if(first+rows==h)InitConstValue(x[(halo-w)*32],InitConstValueParams<float>{1,static_cast<uint16_t>(32*w/8),0,0.0f});
  if(top || first+rows==h)PipeBarrier<PIPE_ALL>();
  Nd2NzParams nd{};nd.ndNum=1;nd.nValue=32;nd.dValue=valid;nd.srcDValue=full;nd.dstNzC0Stride=32;nd.dstNzNStride=1;
  DataCopy(x[top*32],fg[b*32*full+start*w],nd);
  PipeBarrier<PIPE_ALL>();
  CrossCoreWaitFlag<2,PIPE_MTE2>(0);
  Nd2NzParams nw{};nw.ndNum=1;nw.nValue=12;nw.dValue=96;nw.srcDValue=96;nw.dstNzC0Stride=16;nw.dstNzNStride=1;
  DataCopy(wt,ring,nw);PipeBarrier<PIPE_ALL>();
  LoadData2DParams aw{};aw.repeatTimes=12;aw.srcStride=1;LoadData(a,wt,aw);PipeBarrier<PIPE_ALL>();
  for(uint32_t r=0;r<rows;r+=R) {
   if(r)WaitFlag<HardEvent::M_MTE1>(5);
   for(uint32_t dy=0;dy<3;++dy) {
    LoadData3DParamsV2<float> pb{};pb.l1H=1;pb.l1W=32;pb.channelSize=halo;
    pb.kExtension=P;pb.mExtension=32;pb.kStartPt=(r+dy)*w;pb.enTranspose=true;
    pb.filterW=1;pb.filterH=1;pb.strideW=1;pb.strideH=1;pb.dilationFilterW=1;pb.dilationFilterH=1;
    LoadData(xb[dy*32*P],x,pb);
   }
   SetFlag<HardEvent::MTE1_M>(5);WaitFlag<HardEvent::MTE1_M>(5);
   const uint32_t slot=(r/R)%2;
   if(r>=2*R) {if(slot==0)WaitFlag<HardEvent::FIX_M>(5);else WaitFlag<HardEvent::FIX_M>(6);}
   auto tileC=c[slot*16*P];
   MmadParams mm{};mm.m=16;mm.n=P;mm.k=96;mm.cmatrixInitVal=true;Mmad(tileC,a,xb,mm);
   SetFlag<HardEvent::M_MTE1>(5);
   SetFlag<HardEvent::M_FIX>(5);WaitFlag<HardEvent::M_FIX>(5);
   if(r)CrossCoreWaitFlag<2,PIPE_FIX>(2);
   FixpipeParamsV220 fp{};fp.mSize=12;fp.nSize=P;fp.srcStride=16;fp.dstStride=P;
   Fixpipe<float,float,CFG_ROW_MAJOR>(ring,tileC,fp);
   if(slot==0)SetFlag<HardEvent::FIX_M>(5);else SetFlag<HardEvent::FIX_M>(6);
   CrossCoreSetFlag<2,PIPE_FIX>(1);
  }
  WaitFlag<HardEvent::M_MTE1>(5);
  WaitFlag<HardEvent::FIX_M>(5);WaitFlag<HardEvent::FIX_M>(6);
  CrossCoreWaitFlag<2,PIPE_FIX>(2);PipeBarrier<PIPE_ALL>();
 }
 __aicore__ inline void Statistics() {
  auto acc=accBuf.Get<float>();auto tmp=tmpBuf.Get<float>();auto scratch=scratchBuf.Get<float>();
  auto coord=coordBuf.Get<float>();auto red=smallBuf.Get<float>();
  if(batch!=1) {DataCopyPad(red[32],bg,DataCopyExtParams{1,16,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();}
  FdkCubeStatistics::WriteStatistics({acc,tmp,scratch,coord,red},stats,N,group,sub,tau);
 }
 __aicore__ inline void Merge() {
  auto st=statsBuf.Get<float>();auto red=smallBuf.Get<float>();auto factors=accBuf.Get<float>();
  DataCopy(st,stats[b*parts*32],parts*32);PipeBarrier<PIPE_ALL>();
  for(uint32_t k=sub*2;k<sub*2+2;++k) {
   FdkCubeStatistics::MergeChannel(st,red,factors,{kg,sg,b,stride},k,parts);
  }
 }
 TPipe pipe;
 TBuf<QuePosition::VECCALC> rawBuf,accBuf,tmpBuf,scratchBuf,coordBuf,indexBuf,smallBuf,statsBuf;
 TBuf<QuePosition::A1> inputBuf;TBuf<QuePosition::B1> weightBuf;
 TBuf<QuePosition::A2> aBuf;TBuf<QuePosition::B2> bBuf;TBuf<QuePosition::CO1> cBuf;
 GlobalTensor<float> fg,wg,bg,kg,sg,ring,stats;GlobalTensor<uint32_t> map;
 uint32_t h,w,full,batch,parts,blocks,group,sub,part,b,rows,first,R,P,N,halo,ringN,stride;float tau;
};
