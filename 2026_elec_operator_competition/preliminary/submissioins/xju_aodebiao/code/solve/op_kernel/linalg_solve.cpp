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
#include "linalg_solve_tiling.h"
#include "tiling_key_linalg_solve.h"
using namespace AscendC;
class LuKernel {
public:
 __aicore__ inline void Init(GM_ADDR aa,GM_ADDR bb,GM_ADDR oo,GM_ADDR ww,const LinalgSolveTilingData& tt){
  t=tt; a.SetGlobalBuffer((__gm__ float*)aa); if(t.k) b.SetGlobalBuffer((__gm__ float*)bb);
  out.SetGlobalBuffer((__gm__ float*)oo); w.SetGlobalBuffer((__gm__ float*)ww);
  pipe.InitBuffer(buf,4*(5*t.p+3*t.q+8*t.n+8));
  ar=buf.Get<float>();pr=ar[t.p];tmp=pr[t.p];orig=tmp[t.p];br=orig[t.p];bp=br[t.q];bt=bp[t.q];col=bt[t.q];piv=col[8*t.n].ReinterpretCast<uint32_t>();
  result=piv[t.p].ReinterpretCast<float>();
 }
 __aicore__ inline void Process(){
  if(t.n==0U || t.k==0U) return;
   for(uint64_t batch=GetBlockIdx();batch<t.batch;batch+=t.blocks){
    Factor(batch);Back();Refine();
    StoreOutput(batch);
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
 __aicore__ inline uint64_t InputOffset(uint64_t batch,const uint64_t *strides) const{
  uint64_t offset=0U;
  for(int32_t d=static_cast<int32_t>(t.batchRank)-1;d>=0;--d){
   const uint32_t extent=t.batchDims[d];
   if(extent==0U) return 0U;
   offset+=(batch%extent)*strides[d];batch/=extent;
  }
  return offset;
 }
 __aicore__ inline void LoadMatrixRow(uint32_t row,LocalTensor<float> dst){
  if(t.rightSide==0U){Read(a,aInputOffset+uint64_t(row)*t.n,dst,t.n);return;}
  Sync<HardEvent::V_S>();
  for(uint32_t j=0U;j<t.n;++j)dst.SetValue(j,a.GetValue(aInputOffset+uint64_t(j)*t.n+row));
 }
 __aicore__ inline void LoadRhsRow(uint32_t row,LocalTensor<float> dst){
  if(t.rightSide==0U){Read(b,bInputOffset+uint64_t(row)*t.k,dst,t.k);return;}
  Sync<HardEvent::V_S>();
  for(uint32_t j=0U;j<t.k;++j)dst.SetValue(j,b.GetValue(bInputOffset+uint64_t(j)*t.n+row));
 }
 __aicore__ inline void StoreOutput(uint64_t batch){
  const uint64_t outputBase=batch*t.n*t.k;
  for(uint32_t row=0U;row<t.n;++row){
   Read(w,bo+uint64_t(row)*t.q,br,t.q);
   if(t.rightSide==0U){Write(out,outputBase+uint64_t(row)*t.k,br,t.k);continue;}
   for(uint32_t j=0U;j<t.k;++j){
    result.SetValue(0U,br.GetValue(j));
    Write(out,outputBase+uint64_t(j)*t.n+row,result,1U);
   }
  }
 }
 __aicore__ inline void Factor(uint64_t batch){
  ao=uint64_t(batch)*t.n*(t.p+2*t.q);bo=ao+uint64_t(t.n)*t.p;
  aInputOffset=InputOffset(batch,t.matrixBatchStrides);
  bInputOffset=InputOffset(batch,t.rhsBatchStrides);
  for(uint32_t i=0;i<t.n;++i){
   Duplicate(ar,0.0F,t.p);LoadMatrixRow(i,ar);Write(w,ao+uint64_t(i)*t.p,ar,t.p);
   if(t.k){Duplicate(br,0.0F,t.q);LoadRhsRow(i,br);Write(w,bo+uint64_t(i)*t.q,br,t.q);}
  }
  for(uint32_t j=0;j<t.n;++j){
   Sync<HardEvent::S_MTE2>();Sync<HardEvent::V_MTE2>();
   DataCopyPad(col,w[ao+uint64_t(j)*t.p+j],DataCopyExtParams{static_cast<uint16_t>(t.n-j),4,(t.p-1)*4,0,0},DataCopyPadExtParams<float>{true,0,0,0.0F});
   Sync<HardEvent::MTE2_S>();
   uint32_t pivot=j;float maxv=Abs(col.GetValue(0));
   for(uint32_t i=j+1;i<t.n;++i){float z=Abs(col.GetValue((i-j)*8));if(z>maxv){maxv=z;pivot=i;}}
   piv.SetValue(j,pivot);
   Read(w,ao+uint64_t(j)*t.p,pr,t.p);
   if(pivot!=j){
    Read(w,ao+uint64_t(pivot)*t.p,ar,t.p);Write(w,ao+uint64_t(pivot)*t.p,pr,t.p);Write(w,ao+uint64_t(j)*t.p,ar,t.p);Read(w,ao+uint64_t(j)*t.p,pr,t.p);
    if(t.k){Read(w,bo+uint64_t(j)*t.q,bp,t.q);Read(w,bo+uint64_t(pivot)*t.q,br,t.q);Write(w,bo+uint64_t(pivot)*t.q,bp,t.q);Write(w,bo+uint64_t(j)*t.q,br,t.q);}
   }
   const float diagonal=pr.GetValue(j);
   if(diagonal==0.0F){
    AscendC::Trap();return;
   }
   for(uint32_t h=0;h<=j;++h)pr.SetValue(h,0.0F);
   if(t.k)Read(w,bo+uint64_t(j)*t.q,bp,t.q);
   for(uint32_t i=j+1;i<t.n;++i){
    Read(w,ao+uint64_t(i)*t.p,ar,t.p);float factor=ar.GetValue(j)/diagonal;
    Sync<HardEvent::S_V>();Muls(tmp,pr,factor,t.p);Sub(ar,ar,tmp,t.p);Sync<HardEvent::V_S>();ar.SetValue(j,factor);Write(w,ao+uint64_t(i)*t.p,ar,t.p);
    if(t.k){Read(w,bo+uint64_t(i)*t.q,br,t.q);Sync<HardEvent::S_V>();Muls(bt,bp,factor,t.q);Sub(br,br,bt,t.q);Write(w,bo+uint64_t(i)*t.q,br,t.q);}
   }
  }
 }

 __aicore__ inline void Accumulate(float z,float& sum,float& error){float next=sum+z;error+=Abs(sum)>=Abs(z)?(sum-next)+z:(z-next)+sum;sum=next;}
 __aicore__ inline float ProductError(float a,float b,float product){
  union Bits{float f;uint32_t u;}aa,bb;aa.f=a;bb.f=b;aa.u&=0xfffff000U;bb.u&=0xfffff000U;
  float ah=aa.f,bh=bb.f,al=a-ah,bl=b-bh;return ((ah*bh-product)+ah*bl+al*bh)+al*bl;
 }
 __aicore__ inline void Refine(){
  const uint64_t xo=bo+uint64_t(t.n)*t.q;
  for(uint32_t i=0;i<t.n;++i){Read(w,bo+uint64_t(i)*t.q,br,t.q);Write(w,xo+uint64_t(i)*t.q,br,t.q);}
  for(uint32_t iter=0;iter<2;++iter){
   for(uint32_t i=0;i<t.n;++i){
    LoadMatrixRow(i,orig);
    LoadRhsRow(i,br);
    for(uint32_t h=0;h<t.k;++h)bt.SetValue(h,0.0F);
    for(uint32_t j=0;j<t.n;++j){
     const float av=orig.GetValue(j);Read(w,xo+uint64_t(j)*t.q,bp,t.q);
     for(uint32_t h=0;h<t.k;++h){
      float xv=bp.GetValue(h),prod=av*xv,sum=br.GetValue(h),err=bt.GetValue(h);
      Accumulate(-prod,sum,err);Accumulate(-ProductError(av,xv,prod),sum,err);br.SetValue(h,sum);bt.SetValue(h,err);
     }
    }
    for(uint32_t h=0;h<t.k;++h)br.SetValue(h,br.GetValue(h)+bt.GetValue(h));
    Write(w,bo+uint64_t(i)*t.q,br,t.q);
   }
   // Apply the saved row permutations to the fresh residual, then L and U.
   for(uint32_t j=0;j<t.n;++j){uint32_t p=piv.GetValue(j);if(p!=j){Read(w,bo+uint64_t(j)*t.q,bp,t.q);Read(w,bo+uint64_t(p)*t.q,br,t.q);Write(w,bo+uint64_t(j)*t.q,br,t.q);Write(w,bo+uint64_t(p)*t.q,bp,t.q);}}
   for(uint32_t j=0;j<t.n;++j){
    Read(w,bo+uint64_t(j)*t.q,bp,t.q);
    for(uint32_t i=j+1;i<t.n;++i){
     Read(w,ao+uint64_t(i)*t.p,ar,t.p);float v=ar.GetValue(j);Read(w,bo+uint64_t(i)*t.q,br,t.q);
     Sync<HardEvent::S_V>();Muls(bt,bp,v,t.q);Sub(br,br,bt,t.q);Write(w,bo+uint64_t(i)*t.q,br,t.q);
    }
   }
   Back();
   for(uint32_t i=0;i<t.n;++i){Read(w,bo+uint64_t(i)*t.q,br,t.q);Read(w,xo+uint64_t(i)*t.q,bp,t.q);Add(br,br,bp,t.q);Write(w,xo+uint64_t(i)*t.q,br,t.q);}
  }
  for(uint32_t i=0;i<t.n;++i){Read(w,xo+uint64_t(i)*t.q,br,t.q);Write(w,bo+uint64_t(i)*t.q,br,t.q);}
 }
 __aicore__ inline void Back(){
  for(int32_t row=static_cast<int32_t>(t.n)-1;row>=0;--row){
   Read(w,ao+uint64_t(row)*t.p,ar,t.p);float d=ar.GetValue(row);Read(w,bo+uint64_t(row)*t.q,bp,t.q);
   // A singular system has no unique solution; do not divide or fabricate one.
   if(d==0.0F){AscendC::Trap();return;}
   Sync<HardEvent::S_V>();Muls(bp,bp,1.0F/d,t.q);Write(w,bo+uint64_t(row)*t.q,bp,t.q);
   for(uint32_t i=0;i<static_cast<uint32_t>(row);++i){
    Read(w,ao+uint64_t(i)*t.p,ar,t.p);float c=ar.GetValue(row);Read(w,bo+uint64_t(i)*t.q,br,t.q);
    Sync<HardEvent::S_V>();Muls(bt,bp,c,t.q);Sub(br,br,bt,t.q);Write(w,bo+uint64_t(i)*t.q,br,t.q);
   }
  }
 }
 TPipe pipe;TBuf<TPosition::VECCALC> buf;GlobalTensor<float>a,b,out,w;
 LocalTensor<float>ar,pr,tmp,orig,br,bp,bt,col,result;LocalTensor<uint32_t>piv;
 LinalgSolveTilingData t;uint64_t ao,bo,aInputOffset,bInputOffset;
};
template<typename DT_A> __global__ __aicore__ void linalg_solve(GM_ADDR A,GM_ADDR B,GM_ADDR X, GM_ADDR workspace,GM_ADDR tiling){
 REGISTER_TILING_DEFAULT(LinalgSolveTilingData);GET_TILING_DATA_WITH_STRUCT(LinalgSolveTilingData,t,tiling);
 SetSysWorkspace(workspace);LuKernel op;op.Init(A,B,X,GetUserWorkspace(workspace),t);op.Process();
}
