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

namespace FdkCubeStatistics {
using AscendC::GlobalTensor;
using AscendC::LocalTensor;

struct Buffers {
 const LocalTensor<float>& acc;
 const LocalTensor<float>& tmp;
 const LocalTensor<float>& scratch;
 const LocalTensor<float>& coord;
 const LocalTensor<float>& red;
};

struct Outputs {
 const GlobalTensor<float>& keypoints;
 const GlobalTensor<float>& scores;
 uint32_t batch;
 uint32_t stride;
};

__aicore__ inline void WriteStatistics(const Buffers& buffers,const GlobalTensor<float>& stats,
                                     uint32_t count,uint32_t group,uint32_t sub,float tau) {
 // Host validation guarantees a positive temperature before any kernel launch.
 if (!(tau > 0.0f)) return;
 auto tmp=buffers.tmp;auto scratch=buffers.scratch;auto coord=buffers.coord;auto red=buffers.red;
 for(uint32_t k=0;k<2;++k) {
  auto z=buffers.acc[k*count];AscendC::Adds(z,z,red.GetValue(32+sub*2+k),count);AscendC::PipeBarrier<PIPE_V>();
  AscendC::ReduceMax(red,z,scratch,count,false);AscendC::PipeBarrier<PIPE_ALL>();float score=red.GetValue(0);
  AscendC::Adds(z,z,-score,count);AscendC::PipeBarrier<PIPE_V>();AscendC::Muls(z,z,1.0f/tau,count);AscendC::PipeBarrier<PIPE_V>();AscendC::Exp(z,z,count);AscendC::PipeBarrier<PIPE_V>();
  AscendC::ReduceSum(red,z,scratch,count);AscendC::PipeBarrier<PIPE_ALL>();float den=red.GetValue(0);
  AscendC::Mul(tmp,z,coord,count);AscendC::PipeBarrier<PIPE_V>();AscendC::ReduceSum(red,tmp,scratch,count);AscendC::PipeBarrier<PIPE_ALL>();float nx=red.GetValue(0);
  AscendC::Mul(tmp,z,coord[count],count);AscendC::PipeBarrier<PIPE_V>();AscendC::ReduceSum(red,tmp,scratch,count);AscendC::PipeBarrier<PIPE_ALL>();float ny=red.GetValue(0);
  red.SetValue(0,score);red.SetValue(1,score/tau);red.SetValue(2,den);red.SetValue(3,nx);red.SetValue(4,ny);
  AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopy(stats[(group*4+sub*2+k)*8],red,8);AscendC::PipeBarrier<PIPE_ALL>();
 }
}

__aicore__ inline void MergeChannel(const LocalTensor<float>& st,const LocalTensor<float>& red,
                                  const LocalTensor<float>& factors,const Outputs& outputs,
                                  uint32_t k,uint32_t parts) {
 float mx=st.GetValue(k*8+1),score=st.GetValue(k*8);
 for(uint32_t p=1;p<parts;++p) {float v=st.GetValue(p*32+k*8+1);mx=mx>v?mx:v;v=st.GetValue(p*32+k*8);score=score>v?score:v;}
 for(uint32_t p=0;p<parts;++p)factors.SetValue(p,st.GetValue(p*32+k*8+1)-mx);
 AscendC::PipeBarrier<PIPE_ALL>();AscendC::Exp(factors,factors,32);AscendC::PipeBarrier<PIPE_ALL>();
 float d=0,x=0,y=0;
 for(uint32_t p=0;p<parts;++p) {float f=factors.GetValue(p);d+=f*st.GetValue(p*32+k*8+2);x+=f*st.GetValue(p*32+k*8+3);y+=f*st.GetValue(p*32+k*8+4);}
 red.SetValue(0,(x/d+0.5f)*static_cast<float>(static_cast<int32_t>(outputs.stride)));red.SetValue(1,(y/d+0.5f)*static_cast<float>(static_cast<int32_t>(outputs.stride)));
 AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopyPad(outputs.keypoints[(outputs.batch*4+k)*2],red,AscendC::DataCopyExtParams{1,8,0,0,0});
 AscendC::PipeBarrier<PIPE_ALL>();red.SetValue(0,score);AscendC::PipeBarrier<PIPE_ALL>();AscendC::DataCopyPad(outputs.scores[outputs.batch*4+k],red,AscendC::DataCopyExtParams{1,4,0,0,0});AscendC::PipeBarrier<PIPE_ALL>();
}
}
