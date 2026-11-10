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
#include "linalg_det_tiling.h"
#include "tiling_key_linalg_det.h"
using namespace AscendC;
class LuKernel {
public:
 __aicore__ inline void Init(GM_ADDR aa,GM_ADDR bb,GM_ADDR oo,GM_ADDR ww,const LinalgDetTilingData& tt){
  t=tt; a.SetGlobalBuffer((__gm__ float*)aa); if(t.k) b.SetGlobalBuffer((__gm__ float*)bb);
  out.SetGlobalBuffer((__gm__ float*)oo); w.SetGlobalBuffer((__gm__ float*)ww);
  pipe.InitBuffer(buf,4*(4*t.p+3*t.q+8*t.n+8));
  ar=buf.Get<float>();pr=ar[t.p];tmp=pr[t.p];orig=tmp[t.p];br=orig[t.p];bp=br[t.q];bt=bp[t.q];col=bt[t.q];
  result=col[8*t.n];
 }
 __aicore__ inline void Process(){
  if(!t.k){
   for(uint32_t first=GetBlockIdx()*8;first<t.batch;first+=t.blocks*8){
    uint32_t count=t.batch-first<8?t.batch-first:8;
    for(uint32_t q=0;q<count;++q){Factor(first+q);result.SetValue(q,detValue);}
    Write(out,first,result,count);
   }
  }else{
   for(uint32_t batch=GetBlockIdx();batch<t.batch;batch+=t.blocks){
    Factor(batch);Back();
    for(uint32_t i=0;i<t.n;++i){Read(w,bo+uint64_t(i)*t.q,br,t.q);Write(out,uint64_t(batch)*t.n*t.k+uint64_t(i)*t.k,br,t.k);}
   }
  }
 }
private:
 template<HardEvent E> __aicore__ inline void Sync(){auto id=pipe.AllocEventID<E>();SetFlag<E>(id);WaitFlag<E>(id);pipe.ReleaseEventID<E>(id);}
 __aicore__ inline void Read(GlobalTensor<float> src,uint64_t off,LocalTensor<float> dst,uint32_t cnt){
  Sync<HardEvent::S_MTE2>();Sync<HardEvent::V_MTE2>();
  DataCopyPad(dst,src[off],DataCopyExtParams{1,cnt*4,0,0,0},DataCopyPadExtParams<float>{true,0,0,0.0F});
  Sync<HardEvent::MTE2_S>();Sync<HardEvent::MTE2_V>();
 }
 __aicore__ inline void Write(GlobalTensor<float> dst,uint64_t off,LocalTensor<float> src,uint32_t cnt){
  Sync<HardEvent::S_MTE3>();Sync<HardEvent::V_MTE3>();
  DataCopyPad(dst[off],src,DataCopyExtParams{1,cnt*4,0,0,0});
  Sync<HardEvent::MTE3_S>();Sync<HardEvent::MTE3_V>();Sync<HardEvent::MTE3_MTE2>();
 }
 __aicore__ inline float Abs(float x){return x<0?-x:x;}
 __aicore__ inline void Factor(uint32_t batch){
  ao=uint64_t(batch)*t.n*(t.p+t.q);bo=ao+uint64_t(t.n)*t.p;
  for(uint32_t i=0;i<t.n;++i){
   Duplicate(ar,0.0F,t.p);Read(a,uint64_t(batch)*t.n*t.n+uint64_t(i)*t.n,ar,t.n);Write(w,ao+uint64_t(i)*t.p,ar,t.p);
   if(t.k){Duplicate(br,0.0F,t.q);Read(b,(t.broadcastB?0:uint64_t(batch)*t.n*t.k)+uint64_t(i)*t.k,br,t.k);Write(w,bo+uint64_t(i)*t.q,br,t.q);}
  }
  float mant=1.0F;int exponent=0;float sign=1.0F;
  for(uint32_t j=0;j<t.n;++j){
   Sync<HardEvent::S_MTE2>();Sync<HardEvent::V_MTE2>();
   DataCopyPad(col,w[ao+uint64_t(j)*t.p+j],DataCopyExtParams{static_cast<uint16_t>(t.n-j),4,(t.p-1)*4,0,0},DataCopyPadExtParams<float>{true,0,0,0.0F});
   Sync<HardEvent::MTE2_S>();
   uint32_t pivot=j;float maxv=Abs(col.GetValue(0));
   for(uint32_t i=j+1;i<t.n;++i){float z=Abs(col.GetValue((i-j)*8));if(z>maxv){maxv=z;pivot=i;}}
   Read(w,ao+uint64_t(j)*t.p,pr,t.p);
   if(pivot!=j){
    Read(w,ao+uint64_t(pivot)*t.p,ar,t.p);Write(w,ao+uint64_t(pivot)*t.p,pr,t.p);Write(w,ao+uint64_t(j)*t.p,ar,t.p);Read(w,ao+uint64_t(j)*t.p,pr,t.p);sign=-sign;
    if(t.k){Read(w,bo+uint64_t(j)*t.q,bp,t.q);Read(w,bo+uint64_t(pivot)*t.q,br,t.q);Write(w,bo+uint64_t(pivot)*t.q,bp,t.q);Write(w,bo+uint64_t(j)*t.q,br,t.q);}
   }
   const float diagonal=pr.GetValue(j);
   if(!t.k){
    if(diagonal==0.0F){detValue=0.0F;return;}
    float z=diagonal;if(z<0){sign=-sign;z=-z;}
    // Finite FP32 normalization needs at most 127 downward or 149 upward shifts.
    for(uint32_t shift=0;shift<128U && z>=2.0F;++shift){z*=0.5F;++exponent;}
    for(uint32_t shift=0;shift<149U && z<1.0F;++shift){z*=2.0F;--exponent;}
    mant*=z;if(mant>=2.0F){mant*=0.5F;++exponent;}
   }
   for(uint32_t h=0;h<=j;++h)pr.SetValue(h,0.0F);
   if(t.k)Read(w,bo+uint64_t(j)*t.q,bp,t.q);
   for(uint32_t i=j+1;i<t.n;++i){
    Read(w,ao+uint64_t(i)*t.p,ar,t.p);float factor=ar.GetValue(j)/diagonal;
    Sync<HardEvent::S_V>();Muls(tmp,pr,factor,t.p);Sub(ar,ar,tmp,t.p);Sync<HardEvent::V_S>();ar.SetValue(j,factor);Write(w,ao+uint64_t(i)*t.p,ar,t.p);
    if(t.k){Read(w,bo+uint64_t(i)*t.q,br,t.q);Sync<HardEvent::S_V>();Muls(bt,bp,factor,t.q);Sub(br,br,bt,t.q);Write(w,bo+uint64_t(i)*t.q,br,t.q);}
   }
  }
  if(!t.k){while(exponent>0){mant*=2.0F;--exponent;}while(exponent<0){mant*=0.5F;++exponent;}detValue=sign*mant;}
 }
 __aicore__ inline void Back(){
  for(int32_t row=static_cast<int32_t>(t.n)-1;row>=0;--row){
   Read(w,ao+uint64_t(row)*t.p,ar,t.p);float d=ar.GetValue(row);if(d==0.0F){continue;}Read(w,bo+uint64_t(row)*t.q,bp,t.q);
   Sync<HardEvent::S_V>();Muls(bp,bp,1.0F/d,t.q);Write(w,bo+uint64_t(row)*t.q,bp,t.q);
   for(uint32_t i=0;i<static_cast<uint32_t>(row);++i){
    Read(w,ao+uint64_t(i)*t.p,ar,t.p);float c=ar.GetValue(row);Read(w,bo+uint64_t(i)*t.q,br,t.q);
    Sync<HardEvent::S_V>();Muls(bt,bp,c,t.q);Sub(br,br,bt,t.q);Write(w,bo+uint64_t(i)*t.q,br,t.q);
   }
  }
 }
 TPipe pipe;TBuf<TPosition::VECCALC> buf;GlobalTensor<float>a,b,out,w;
 LocalTensor<float>ar,pr,tmp,orig,br,bp,bt,col,result;
 LinalgDetTilingData t;uint64_t ao,bo;float detValue;
};
template<typename DT_A> __global__ __aicore__ void linalg_det(GM_ADDR A,GM_ADDR det, GM_ADDR workspace,GM_ADDR tiling){
 REGISTER_TILING_DEFAULT(LinalgDetTilingData);GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData,t,tiling);
 SetSysWorkspace(workspace);LuKernel op;op.Init(A,nullptr,det,GetUserWorkspace(workspace),t);op.Process();
}
