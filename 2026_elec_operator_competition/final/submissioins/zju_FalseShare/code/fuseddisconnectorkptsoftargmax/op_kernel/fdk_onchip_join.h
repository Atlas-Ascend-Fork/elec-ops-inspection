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

// Vertical FP32 projection stays on Cube. Decompose each projection into two
// FP16 components, then apply the sparse horizontal sum to both in FP32.
class FdkOnchipJoin {
public:
 __aicore__ inline void Init(GM_ADDR feat,GM_ADDR weight,GM_ADDR bias,GM_ADDR kp,GM_ADDR score,
                           const FusedDisconnectorKeypointSoftArgmaxTilingData& t,GM_ADDR tilingAddr) {
  h=t.height;w=t.width;full=h*w;parts=t.parts;blocks=t.blocks;R=t.tileRows;P=R*w;T=(R+2)*w;
  fastPack=t.batch==2&&h==64&&w==64&&t.tau==0.5f&&t.stride==4;
  packOffsets.SetGlobalBuffer((__gm__ uint32_t*)tilingAddr+8);
  group=GetBlockIdx()/GetTaskRation();sub=GetSubBlockIdx();part=group%parts;batch=group/parts;
  uint32_t tileFirst=part*(h/R)/parts,tileLast=(part+1)*(h/R)/parts;first=tileFirst*R;rows=(tileLast-tileFirst)*R;N=rows*w;tau=t.tau;stride=t.stride;
  auto base=GetSysWorkSpacePtr();
  fg.SetGlobalBuffer((__gm__ float*)feat);wg.SetGlobalBuffer((__gm__ float*)weight);
  bg.SetGlobalBuffer((__gm__ float*)bias);kg.SetGlobalBuffer((__gm__ float*)kp);sg.SetGlobalBuffer((__gm__ float*)score);
  ringN=4*P>1152?4*P:1152;
  packed.SetGlobalBuffer((__gm__ float*)base+group*ringN);coeff.SetGlobalBuffer((__gm__ half*)tilingAddr+(16+576*2));
  ring.SetGlobalBuffer((__gm__ float*)base+group*ringN);
  stats.SetGlobalBuffer((__gm__ float*)base+blocks*ringN);
  if ASCEND_IS_AIV {
   pipe.InitBuffer(rawBuf,576*4);pipe.InitBuffer(packBuf,(fastPack?576:96)*4);pipe.InitBuffer(ixBuf,(P>576?P:576)*4);
   pipe.InitBuffer(heatBuf,4*P*4);pipe.InitBuffer(accBuf,2*N*4);pipe.InitBuffer(tmpBuf,N*4);
   pipe.InitBuffer(scratchBuf,N*4);pipe.InitBuffer(coordBuf,2*N*4);pipe.InitBuffer(redBuf,64*4);pipe.InitBuffer(statsBuf,parts*128);
  } else {
   pipe.InitBuffer(inputL1,(w==96?2:1)*T*32*4);pipe.InitBuffer(weightL1,16*96*4);pipe.InitBuffer(halfL1,2*P*16*2);pipe.InitBuffer(constL1,4096);
   pipe.InitBuffer(aBuf,65536);pipe.InitBuffer(bBuf,10240);pipe.InitBuffer(cBuf,2*16*P*4);
  }
 }
 __aicore__ inline uint32_t Process() {
  if ASCEND_IS_AIV {
   Pack();CrossCoreSetFlag<2,PIPE_MTE3>(0);
   auto coord=coordBuf.Get<float>();
   CreateVecIndex(coord,0.0f,w);PipeBarrier<PIPE_ALL>();
   for(uint32_t y=1;y<rows;++y)DataCopy(coord[y*w],coord,w);
   PipeBarrier<PIPE_ALL>();
   for(uint32_t y=0;y<rows;++y)Duplicate(coord[N+y*w],static_cast<float>(static_cast<int32_t>(first+y)),w);
   PipeBarrier<PIPE_V>();
  }
  if ASCEND_IS_AIC {Cube();return 0;}
  else {
   auto heat=heatBuf.Get<float>();auto acc=accBuf.Get<float>();auto ix=ixBuf.Get<int32_t>();
   CreateVecIndex(ix,0,P);PipeBarrier<PIPE_V>();Muls(ix,ix,int32_t(16),P);PipeBarrier<PIPE_ALL>();
   for(uint32_t r=0;r<rows;r+=R) {
    CrossCoreWaitFlag<2>(1);
    DataCopy(heat,ring,4*P);CrossCoreSetFlag<2,PIPE_MTE2>(2);PipeBarrier<PIPE_ALL>();
    for(uint32_t k=0;k<2;++k) {Gather(acc[k*N+r*w],heat,ix.ReinterpretCast<uint32_t>(),uint32_t((sub*2+k)*4),P);PipeBarrier<PIPE_V>();}
    SetFlag<HardEvent::V_MTE2>(6);WaitFlag<HardEvent::V_MTE2>(6);
   }
   Statistics();SyncAll();return part==0?Merge():0;
  }
 }
private:
 __aicore__ inline void Pack() {
  auto raw=rawBuf.Get<float>();auto tmp=packBuf.Get<float>();auto ix=ixBuf.Get<uint32_t>();
  if(fastPack){
   DataCopy(raw,wg[sub*576],576);DataCopy(ix,packOffsets,576);PipeBarrier<PIPE_ALL>();
   Gather(tmp,raw,ix,0,576);PipeBarrier<PIPE_ALL>();
   for(uint32_t dx=0;dx<3;++dx)DataCopy(packed[(dx*4+sub*2)*96],tmp[dx*192],192);
   PipeBarrier<PIPE_ALL>();return;
  }
  DataCopy(raw,wg[sub*576],576);PipeBarrier<PIPE_ALL>();
  if(w==96) {
   auto index=ix.ReinterpretCast<int32_t>();
   CreateVecIndex(index,0,32);PipeBarrier<PIPE_V>();
   Muls(index,index,int32_t(36),32);PipeBarrier<PIPE_V>();
   Adds(index[32],index,int32_t(12),32);Adds(index[64],index,int32_t(24),32);
  } else for(uint32_t dy=0;dy<3;++dy)for(uint32_t c=0;c<32;++c)ix.SetValue(dy*32+c,(c*9+dy*3)*4);
  PipeBarrier<PIPE_ALL>();
  for(uint32_t dx=0;dx<3;++dx)for(uint32_t k=0;k<2;++k) {
   Gather(tmp,raw[k*288],ix,dx*4,96);PipeBarrier<PIPE_ALL>();
   DataCopy(packed[(dx*4+sub*2+k)*96],tmp,96);PipeBarrier<PIPE_ALL>();
  }

 }
 __aicore__ inline void LoadInput(const LocalTensor<float>& x,uint32_t y) {
  uint32_t start=y?y-1:0,end=y+R<h?y+R+1:h,top=y?0:w;
  if(y==0||y+R==h){InitConstValue(x,InitConstValueParams<float>{1,static_cast<uint16_t>(T*4),0,0.0f});PipeBarrier<PIPE_MTE2>();}
  Nd2NzParams nd{};nd.ndNum=1;nd.nValue=32;nd.dValue=(end-start)*w;nd.srcDValue=full;nd.dstNzC0Stride=32;nd.dstNzNStride=1;
  DataCopy(x[top*32],fg[batch*32*full+start*w],nd);
 }
 __aicore__ inline void PrepareCube() {
  auto wt=weightL1.Get<float>();auto constants=constL1.Get<half>();auto b=bBuf.Get<float>();
  auto id=b.ReinterpretCast<half>()[3072];auto join=id[256];
  Nd2NzParams nw{};nw.ndNum=1;nw.nValue=12;nw.dValue=96;nw.srcDValue=96;nw.dstNzC0Stride=16;nw.dstNzNStride=1;
  CrossCoreWaitFlag<2,PIPE_MTE2>(0);
  // Initialize ND2NZ N padding before it enters residual and join MMADs.
  InitConstValue(wt,InitConstValueParams<float>{1,192,0,0.0f});PipeBarrier<PIPE_MTE2>();
  DataCopy(wt,packed,nw);DataCopy(constants,coeff,256);
  Nd2NzParams nj{};nj.ndNum=1;nj.nValue=4;nj.dValue=96;nj.srcDValue=96;nj.dstNzC0Stride=16;nj.dstNzNStride=1;
  DataCopy(constants[256],coeff[256],nj);PipeBarrier<PIPE_ALL>();
  LoadData2DParams lw{};lw.repeatTimes=12;lw.srcStride=1;LoadData(b,wt,lw);
  lw.repeatTimes=1;LoadData(id,constants,lw);lw.repeatTimes=6;LoadData(join,constants[256],lw);PipeBarrier<PIPE_ALL>();
  SetHF32Mode(false);SetHF32TransMode(false);
 }
 __aicore__ inline void LoadTileInput(const LocalTensor<float>& tileInput,uint32_t r) {
  if(w!=96||r==0) {
   if(r){SetFlag<HardEvent::MTE1_MTE2>(6);WaitFlag<HardEvent::MTE1_MTE2>(6);}
   LoadInput(tileInput,first+r);PipeBarrier<PIPE_ALL>();
  } else WaitFlag<HardEvent::MTE2_MTE1>(6);
 }
 __aicore__ inline void PrefetchNextInput(uint32_t r) {
  auto x=inputL1.Get<float>();
  SetFlag<HardEvent::MTE1_MTE2>(6);WaitFlag<HardEvent::MTE1_MTE2>(6);
  LoadInput(x[(((r/R)+1)&1)*T*32],first+r+R);
  SetFlag<HardEvent::MTE2_MTE1>(6);SetFlag<HardEvent::MTE2_FIX>(6);
 }
 __aicore__ inline void ProjectTile(uint32_t r) {
  auto x=inputL1.Get<float>();auto a=aBuf.Get<float>();auto b=bBuf.Get<float>();auto c=cBuf.Get<float>();
  auto tileInput=x[(w==96?((r/R)&1):0)*T*32];
  LoadTileInput(tileInput,r);
  LoadData3DParamsV2<float> ld{};ld.l1H=1;ld.l1W=32;ld.channelSize=T;ld.mExtension=32;ld.kExtension=P;ld.enTranspose=true;
  for(uint32_t dy=0;dy<3;++dy) {
   if(r||dy){SetFlag<HardEvent::M_MTE1>(7);WaitFlag<HardEvent::M_MTE1>(7);}
   ld.kStartPt=dy*w;LoadData(a,tileInput,ld);SetFlag<HardEvent::MTE1_M>(7);WaitFlag<HardEvent::MTE1_M>(7);
   MmadParams mm{};mm.m=P;mm.n=16;mm.k=32;mm.cmatrixInitVal=dy==0;Mmad(c,a,b[dy*512],mm);
   if(w==96&&dy==0&&r+R<rows)PrefetchNextInput(r);
  }
  if(w==96&&r+R<rows)WaitFlag<HardEvent::MTE2_FIX>(6);
 }
 __aicore__ inline void DecomposeProjection() {
  auto halves=halfL1.Get<half>();auto hi=halves;auto lo=halves[P*16];
  auto ah=aBuf.Get<float>().ReinterpretCast<half>();auto id=bBuf.Get<float>().ReinterpretCast<half>()[3072];auto c=cBuf.Get<float>();
  SetFlag<HardEvent::M_FIX>(5);WaitFlag<HardEvent::M_FIX>(5);
  FixpipeParamsV220 fp{};fp.nSize=16;fp.mSize=P;fp.srcStride=P;fp.dstStride=P;fp.quantPre=QuantMode_t::F322F16;
  Fixpipe<half,float,CFG_NZ>(hi,c,fp);SetFlag<HardEvent::FIX_MTE1>(5);WaitFlag<HardEvent::FIX_MTE1>(5);
  LoadData3DParamsV2<half> lh{};lh.l1H=R;lh.l1W=w;lh.channelSize=16;lh.mExtension=P;lh.kExtension=16;
  LoadData(ah,hi,lh);SetFlag<HardEvent::MTE1_M>(7);WaitFlag<HardEvent::MTE1_M>(7);
  MmadParams mm{};mm.m=P;mm.n=16;mm.k=16;mm.cmatrixInitVal=false;Mmad(c,ah,id,mm);
  SetFlag<HardEvent::M_FIX>(5);WaitFlag<HardEvent::M_FIX>(5);Fixpipe<half,float,CFG_NZ>(lo,c,fp);
  SetFlag<HardEvent::FIX_MTE1>(5);WaitFlag<HardEvent::FIX_MTE1>(5);
 }
 __aicore__ inline void JoinTile(uint32_t r) {
  auto hi=halfL1.Get<half>();auto ah=aBuf.Get<float>().ReinterpretCast<half>();
  auto join=bBuf.Get<float>().ReinterpretCast<half>()[3072+256];auto result=cBuf.Get<float>()[16*P];
  LoadData3DParamsV2<half> lh{};lh.l1H=R;lh.l1W=w;
  lh.padList[0]=1;lh.padList[1]=1;lh.filterW=3;lh.channelSize=32;lh.kExtension=96;lh.mExtension=256;
  MmadParams mm{};
  const bool pipelinedJoin=w==96&&parts==5;
  const uint32_t joinStep=pipelinedJoin?128:256;
  for(uint32_t mp=0;mp<P;mp+=joinStep) {
   uint32_t joinPixels=P-mp<joinStep?P-mp:joinStep;lh.mExtension=joinPixels;
   if(!pipelinedJoin||mp==0){SetFlag<HardEvent::M_MTE1>(7);WaitFlag<HardEvent::M_MTE1>(7);}
   else if(mp>=2*joinStep)WaitFlag<HardEvent::M_MTE1>(6);
   auto joinA=ah[pipelinedJoin?((mp/joinStep)&1)*joinStep*96:0];
   lh.mStartPt=mp;LoadData(joinA,hi,lh);
   SetFlag<HardEvent::MTE1_M>(7);WaitFlag<HardEvent::MTE1_M>(7);
   if(r&&mp==0)WaitFlag<HardEvent::FIX_M>(5);
   mm.m=joinPixels;mm.n=4;mm.k=96;mm.cmatrixInitVal=true;Mmad(result[mp*16],joinA,join,mm);
   if(pipelinedJoin&&mp+2*joinStep<P)SetFlag<HardEvent::M_MTE1>(6);
  }
 }
 __aicore__ inline void StoreJoinedTile(uint32_t r) {
  auto result=cBuf.Get<float>()[16*P];
  SetFlag<HardEvent::M_FIX>(5);WaitFlag<HardEvent::M_FIX>(5);
  if(r)CrossCoreWaitFlag<2>(2);
  FixpipeParamsV220 out{};out.nSize=4;out.mSize=P;out.srcStride=P;out.dstStride=4;Fixpipe(ring,result,out);
  CrossCoreSetFlag<2,PIPE_FIX>(1);if(r+R<rows)SetFlag<HardEvent::FIX_M>(5);
 }
 __aicore__ inline void Cube() {
  PrepareCube();
  for(uint32_t r=0;r<rows;r+=R) {
   ProjectTile(r);DecomposeProjection();JoinTile(r);StoreJoinedTile(r);
  }
  CrossCoreWaitFlag<2>(2);
 }
 __aicore__ inline void Statistics() {
  auto acc=accBuf.Get<float>();auto tmp=tmpBuf.Get<float>();auto scratch=scratchBuf.Get<float>();
  auto coord=coordBuf.Get<float>();auto red=redBuf.Get<float>();
  DataCopyPad(red[32],bg,DataCopyExtParams{1,16,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});PipeBarrier<PIPE_ALL>();
  FdkCubeStatistics::WriteStatistics({acc,tmp,scratch,coord,red},stats,N,group,sub,tau);
 }
 __aicore__ inline uint32_t Merge() {
  auto st=statsBuf.Get<float>();auto red=redBuf.Get<float>();auto factors=accBuf.Get<float>();
  DataCopy(st,stats[batch*parts*32],parts*32);PipeBarrier<PIPE_ALL>();
  uint32_t invalid=0;
  for(uint32_t k=sub*2;k<sub*2+2;++k) {
   bool bad=false;
   for(uint32_t p=0;p<parts;++p) {float den=st.GetValue(p*32+k*8+2),peak=st.GetValue(p*32+k*8);if(!(den>0.0f)||!(peak<=3.402823466e38f&&peak>=-3.402823466e38f))bad=true;}
   if(bad){invalid|=1U<<(k-sub*2);continue;}
   FdkCubeStatistics::MergeChannel(st,red,factors,{kg,sg,batch,stride},k,parts);
  }
  return invalid;
 }
 TPipe pipe;
 TBuf<QuePosition::VECCALC> rawBuf,packBuf,ixBuf,heatBuf,accBuf,tmpBuf,scratchBuf,coordBuf,redBuf,statsBuf;
 TBuf<QuePosition::A1> inputL1,weightL1,halfL1,constL1;
 TBuf<QuePosition::A2> aBuf;TBuf<QuePosition::B2> bBuf;TBuf<QuePosition::CO1> cBuf;
 GlobalTensor<float> fg,wg,bg,kg,sg,packed,ring,stats;GlobalTensor<half> coeff;GlobalTensor<uint32_t> packOffsets;bool fastPack;
 uint32_t h,w,full,parts,blocks,R,P,T,group,sub,part,batch,rows,first,N,stride,ringN;float tau;
};
