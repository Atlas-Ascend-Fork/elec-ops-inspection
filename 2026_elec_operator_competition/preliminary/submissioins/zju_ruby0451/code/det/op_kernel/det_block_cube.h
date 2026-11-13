/*
 * Copyright (c) 2026 ruby0451 contributors.
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

#ifndef RUBY0451_DET_BLOCK_CUBE_H
#define RUBY0451_DET_BLOCK_CUBE_H
#include "det_block_layout.h"
class DetLeftCube {
 TPipe pipe;TBuf<TPosition::A1>ab;TBuf<TPosition::B1>bb;
 TBuf<TPosition::A2>ac;TBuf<TPosition::B2>bc;TBuf<TPosition::CO1>cc;
public:
 __aicore__ inline void Init(){
  pipe.InitBuffer(ab,65536);pipe.InitBuffer(bb,65536);
  pipe.InitBuffer(ac,65536);pipe.InitBuffer(bc,65536);pipe.InitBuffer(cc,65536);
  SetHF32Mode(HF32Mode::DISABLE);
 }
 __aicore__ inline void OwnedProduct(GlobalTensor<float> w,uint32_t base,uint32_t pitch,
     uint32_t start,uint32_t first,uint32_t width,uint32_t k,uint32_t count,bool raw,bool init){
  auto a1=ab.Get<float>(),b1=bb.Get<float>(),a2=ac.Get<float>(),b2=bc.Get<float>();
  uint32_t rows=raw?32:32+width,cols=raw?width:32;
  if(raw){
   DataCopy(a1,w[base+start*pitch+k],Nd2NzParams{1,32,uint16_t(count),0,uint16_t(pitch),32,1,0});
   DataCopy(b1,w[base+k*pitch+start+32+first],Nd2NzParams{1,uint16_t(count),uint16_t(width),0,uint16_t(pitch),uint16_t(count),1,0});
  }else{
   // FP32 NZ uses eight elements per C0. The second copy starts at row32
   // within each C0 plane; both copies share the full stacked row stride.
   DataCopy(a1,w[base+start*pitch+k],Nd2NzParams{1,32,uint16_t(count),0,uint16_t(pitch),uint16_t(rows),1,0});
   DataCopy(a1[32*8],w[base+(start+32+first)*pitch+k],Nd2NzParams{1,uint16_t(width),uint16_t(count),0,uint16_t(pitch),uint16_t(rows),1,0});
   DataCopy(b1,w[base+k*pitch+start],Nd2NzParams{1,uint16_t(count),32,0,uint16_t(pitch),uint16_t(count),1,0});
  }
  FastFence<HardEvent::MTE2_MTE1>();
  FastLoadLocal(a1,a2,rows,count);FastLoadLocal(b1,b2,count,cols);FastFence<HardEvent::MTE1_M>();
  Mmad(cc.Get<float>()[raw?4096:0],a2,b2,MmadParams{uint16_t(rows),uint16_t(cols),uint16_t(count),0,false,init});
  FastFence<HardEvent::M_MTE1>();FastFence<HardEvent::MTE1_MTE2>();
 }
 __aicore__ inline void RebuildOwned(GM_ADDR W,uint32_t size,uint32_t slots,uint32_t start){
  uint32_t core=GetBlockIdx(),batch=core/4,rhs=size-start-32,chunk=DetOwnerRhsChunk(size,start),first=(core%4)*(2*chunk);
  if(batch>=slots)return;
  if(first>=rhs){CrossCoreWaitFlag(11);return;}
  uint32_t live=rhs-first<2*chunk?rhs-first:2*chunk,width=(live+15)/16*16;
  uint32_t pitch=(size+127)/128*128,base=batch*pitch*pitch,temp=slots*(pitch*pitch+64*pitch);
  uint32_t history=start-32;GlobalTensor<float>w;w.SetGlobalBuffer((__gm__ float*)W);
  // Idle stages consumed11 too: old history is already published for every owner.
  for(uint32_t k=0;k<history;k+=128){
   uint32_t count=history-k;if(count>128)count=128;
   OwnedProduct(w,base,pitch,start,first,width,k,count,false,k==0);
   OwnedProduct(w,base,pitch,start,first,width,k,count,true,k==0);
  }
  CrossCoreWaitFlag(11);
  OwnedProduct(w,base,pitch,start,first,width,history,32,false,history==0);
  FastFence<HardEvent::M_FIX>();
  Fixpipe(w[temp+core*8192],cc.Get<float>(),FixpipeParamsV220{32,uint16_t(32+width),uint16_t(32+width),32,false,QuantMode_t::NoQuant,0,1,0,0,0});
  CrossCoreSetFlag<2,PIPE_FIX>(10);
  // Separate scratch retains both products until the following publication
  // barrier. Raw reconstruction overlaps the sibling AIV factorization.
  OwnedProduct(w,base,pitch,start,first,width,history,32,true,history==0);
  FastFence<HardEvent::M_FIX>();
  Fixpipe(w[temp+core*8192+4096],cc.Get<float>()[4096],FixpipeParamsV220{uint16_t(width),32,32,uint16_t(width),false,QuantMode_t::NoQuant,0,1,0,0,0});
  PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_FIX>(12);
 }

 __aicore__ inline void Rebuild(GM_ADDR W,uint32_t size,uint32_t slots,uint32_t cores,uint32_t start){
 // All participants receive the same positive scheduler core count.
 if(cores==0)return;
  uint32_t pitch=(size+127)/128*128,core=GetBlockIdx();
  uint32_t jobs=slots*DetLeftJobs(size,start,slots,cores),temp=slots*(pitch*pitch+64*pitch);
  if(core>=jobs)return;
  GlobalTensor<float>w;w.SetGlobalBuffer((__gm__ float*)W);
  auto a1=ab.Get<float>(),b1=bb.Get<float>(),a2=ac.Get<float>(),b2=bc.Get<float>(),c=cc.Get<float>();
  uint32_t history=start-32;
  // Every real owner group consumed the preceding11, including idle groups.
  // That global publication proves old tail history ready for every matrix.
  bool ownerTail=size<=320&&size-start<=64;
  bool deferredHistory=history&&!ownerTail&&core>=slots*DetLeftJobs(size,start-32,slots,cores);
  if(deferredHistory)CrossCoreWaitFlag(11);
  // k<history was published before the preceding reconstruction. Its factors
  // are immutable while AIV consumes that cross and solves the newest block.
  if(history){
   for(uint32_t job=core;job<jobs;job+=cores){
    DetLeftJob tile=DetLeftTile(size,start,job,slots,cores);uint32_t base=tile.batch*pitch*pitch;
    auto partial=c[((job-core)/cores)*4096];
    for(uint32_t k=0;k<history;k+=128){
     uint32_t count=history-k;if(count>128)count=128;
     DataCopy(a1,w[base+tile.col*pitch+k],Nd2NzParams{1,uint16_t(tile.n),uint16_t(count),0,uint16_t(pitch),uint16_t(tile.n),1,0});
     DataCopy(b1,w[base+k*pitch+tile.row],Nd2NzParams{1,uint16_t(count),uint16_t(tile.m),0,uint16_t(pitch),uint16_t(count),1,0});
     FastFence<HardEvent::MTE2_MTE1>();
     FastLoadLocal(a1,a2,tile.n,count);FastLoadLocal(b1,b2,count,tile.m);
     FastFence<HardEvent::MTE1_M>();
     Mmad(partial,a2,b2,MmadParams{uint16_t(tile.n),uint16_t(tile.m),uint16_t(count),0,false,k==0});
     FastFence<HardEvent::M_MTE1>();FastFence<HardEvent::MTE1_MTE2>();
    }
   }
  }
  if(!deferredHistory)CrossCoreWaitFlag(11);
  for(uint32_t job=core;job<jobs;job+=cores){
   DetLeftJob tile=DetLeftTile(size,start,job,slots,cores);uint32_t base=tile.batch*pitch*pitch;
   auto partial=c[((job-core)/cores)*4096];
   DataCopy(a1,w[base+tile.col*pitch+history],Nd2NzParams{1,uint16_t(tile.n),32,0,uint16_t(pitch),uint16_t(tile.n),1,0});
   DataCopy(b1,w[base+history*pitch+tile.row],Nd2NzParams{1,32,uint16_t(tile.m),0,uint16_t(pitch),32,1,0});
   FastFence<HardEvent::MTE2_MTE1>();
   FastLoadLocal(a1,a2,tile.n,32);FastLoadLocal(b1,b2,32,tile.m);
   FastFence<HardEvent::MTE1_M>();
   Mmad(partial,a2,b2,MmadParams{uint16_t(tile.n),uint16_t(tile.m),32,0,false,history==0});
   FastFence<HardEvent::M_FIX>();
   if(job!=core)CrossCoreWaitFlag(11);
   Fixpipe(w[temp+core*8192],partial,FixpipeParamsV220{uint16_t(tile.m),uint16_t(tile.n),uint16_t(tile.n),uint16_t(tile.m),false,QuantMode_t::NoQuant,0,1,0,0,0});
   PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_FIX>(10);
  }
 }

};


#endif
