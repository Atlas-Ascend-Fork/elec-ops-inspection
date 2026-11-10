/*
 * Copyright (c) 2026 aodebiao.
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

#include "kernel_operator.h"
#include "fire_candidate_decode_fusion_tiling.h"
#include "tiling_key_fire_candidate_decode_fusion.h"
using namespace AscendC;
template<typename T> class FireKernel {
public:
 __aicore__ inline void Init(GM_ADDR x,GM_ADDR p,GM_ADDR y,const FireCandidateDecodeFusionTilingData& t) {
  n=t.n; b=t.batch; blocks=t.blocks;
  input.SetGlobalBuffer((__gm__ T*)x);prior.SetGlobalBuffer((__gm__ float*)p);output.SetGlobalBuffer((__gm__ T*)y);
  pipe.InitBuffer(raw,384*sizeof(T));pipe.InitBuffer(priorA,320*4);pipe.InitBuffer(floatA,384*4);
  pipe.InitBuffer(sigmoid,384*4);pipe.InitBuffer(priorS,320*4);pipe.InitBuffer(ones,384*4);
  pipe.InitBuffer(result,320*4);pipe.InitBuffer(resultT,320*sizeof(T));
  pipe.InitBuffer(indices,64*4);pipe.InitBuffer(offsetFloat,64*4);pipe.InitBuffer(outIndices,320*4);
  auto idx=outIndices.Get<uint32_t>();
  for(uint32_t j=0;j<320;++j)idx.SetValue(j,((j%5)*64+j/5)*4);
  Sync<HardEvent::S_V>();
  CreateVecIndex(offsetFloat.Get<float>(),0.0F,64);Muls(offsetFloat.Get<float>(),offsetFloat.Get<float>(),24.0F,64);
  Cast(indices.Get<int32_t>(),offsetFloat.Get<float>(),RoundMode::CAST_RINT,64);
  Duplicate(ones.Get<float>(),1.0F,384);
 }
 __aicore__ inline void Process() {
  const uint32_t tiles=(n+63)/64;
  for(uint32_t job=GetBlockIdx();job<b*tiles;job+=blocks) {
   uint32_t batch=job/tiles,first=(job%tiles)*64,count=n-first<64?n-first:64;
   auto r=raw.Get<T>();auto pa=priorA.Get<float>();
   Duplicate(r,T(0.0F),384);Duplicate(pa,0.0F,320);Sync<HardEvent::V_MTE2>();
   DataCopyExtParams xc{1,static_cast<uint32_t>(count*6*sizeof(T)),0,0,0},pc{1,count*5*4,0,0,0};
   DataCopyPad(r,input[(uint64_t(batch)*n+first)*6],xc,DataCopyPadExtParams<T>{true,0,0,T(0.0F)});
   DataCopyPad(pa,prior[first*5],pc,DataCopyPadExtParams<float>{true,0,0,0.0F});Sync<HardEvent::MTE2_V>();
   auto a=floatA.Get<float>();
   if constexpr(sizeof(T)==2) Cast(a,r,RoundMode::CAST_NONE,384); else DataCopy(a,r,384);
   auto s=sigmoid.Get<float>();auto ix=indices.Get<int32_t>();
   CreateVecIndex(offsetFloat.Get<float>(),0.0F,64);Muls(offsetFloat.Get<float>(),offsetFloat.Get<float>(),24.0F,64);
   Cast(ix,offsetFloat.Get<float>(),RoundMode::CAST_RINT,64);
   for(uint32_t c=0;c<6;++c){Gather(s[c*64],a,ix.ReinterpretCast<uint32_t>(),uint32_t(0),64);Adds(ix,ix,int32_t(4),64);}
   auto pr=priorS.Get<float>();
   CreateVecIndex(offsetFloat.Get<float>(),0.0F,64);Muls(offsetFloat.Get<float>(),offsetFloat.Get<float>(),20.0F,64);
   Cast(ix,offsetFloat.Get<float>(),RoundMode::CAST_RINT,64);
   for(uint32_t c=0;c<5;++c){Gather(pr[c*64],pa,ix.ReinterpretCast<uint32_t>(),uint32_t(0),64);Adds(ix,ix,int32_t(4),64);}
   Muls(s,s,-1.0F,384);Maxs(s,s,-80.0F,384);Mins(s,s,80.0F,384);Exp(s,s,384);Adds(s,s,1.0F,384);
   Div(s,ones.Get<float>(),s,384);
   for(uint32_t c=0;c<2;++c){Muls(s[c*64],s[c*64],2.0F,64);Adds(s[c*64],s[c*64],-0.5F,64);Add(s[c*64],s[c*64],pr[c*64],64);Mul(s[c*64],s[c*64],pr[128],64);}
   for(uint32_t c=2;c<4;++c){Muls(s[c*64],s[c*64],2.0F,64);Mul(s[c*64],s[c*64],s[c*64],64);Mul(s[c*64],s[c*64],pr[(c+1)*64],64);Mul(s[c*64],s[c*64],pr[128],64);}
   Mul(s[256],s[256],s[320],64);
   auto res=result.Get<float>();Gather(res,s,outIndices.Get<uint32_t>(),uint32_t(0),count*5);
   auto out=resultT.Get<T>();
   if constexpr(sizeof(T)==2)Cast(out,res,RoundMode::CAST_RINT,count*5);else DataCopy(out,res,(count*5+7)/8*8);
   Sync<HardEvent::V_MTE3>();DataCopyPad(output[(uint64_t(batch)*n+first)*5],out,DataCopyExtParams{1,static_cast<uint32_t>(count*5*sizeof(T)),0,0,0});Sync<HardEvent::MTE3_V>();
  }
 }
private:
 template<HardEvent E> __aicore__ inline void Sync(){auto id=pipe.FetchEventID(E);SetFlag<E>(id);WaitFlag<E>(id);}
 TPipe pipe;TBuf<TPosition::VECCALC> raw,priorA,floatA,sigmoid,priorS,ones,result,resultT,indices,offsetFloat,outIndices;
 GlobalTensor<T> input,output;GlobalTensor<float> prior;uint32_t n,b,blocks;
};
template<typename DT_RAW_PRED> __global__ __aicore__ void fire_candidate_decode_fusion(GM_ADDR raw_pred,GM_ADDR priors,GM_ADDR decoded,GM_ADDR workspace,GM_ADDR tiling){
 REGISTER_TILING_DEFAULT(FireCandidateDecodeFusionTilingData);GET_TILING_DATA_WITH_STRUCT(FireCandidateDecodeFusionTilingData,t,tiling);
 FireKernel<DT_RAW_PRED> op;op.Init(raw_pred,priors,decoded,t);op.Process();
}
