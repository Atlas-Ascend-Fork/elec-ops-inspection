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
#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"
using namespace AscendC;
template<typename T> class TreeKernel {
public:
 __aicore__ inline void Init(GM_ADDR x,GM_ADDR s,GM_ADDR d,GM_ADDR i,const TreeLineMinDistanceTilingData& t){
  n=t.n;m=t.m;batch=t.batch;blocks=t.blocks;ma=(m+7)/8*8;
  points.SetGlobalBuffer((__gm__ T*)x);segments.SetGlobalBuffer((__gm__ T*)s);distOut.SetGlobalBuffer((__gm__ T*)d);idxOut.SetGlobalBuffer((__gm__ int32_t*)i);
  pipe.InitBuffer(arena,ma*17*4);
  pipe.InitBuffer(bitmap,(ma+255)/256*32);pipe.InitBuffer(minResult,32);pipe.InitBuffer(outFloat,64);pipe.InitBuffer(outNative,16*sizeof(T));pipe.InitBuffer(outIdx,64);
  CreateVecIndex(F(13),0.0F,ma);Muls(F(13),F(13),24.0F,ma);Cast(F(14).template ReinterpretCast<int32_t>(),F(13),RoundMode::CAST_RINT,ma);
  Duplicate(F(16),1.0F,ma);
 }
 __aicore__ inline void Process(){
  uint32_t cached=0xffffffffU;uint32_t total=batch*n;
  for(uint64_t first=static_cast<uint64_t>(GetBlockIdx())*16U;first<total;first+=static_cast<uint64_t>(blocks)*16U){
   uint32_t count=total-first<16?total-first:16;
   for(uint32_t q=0;q<count;++q){
    uint32_t point=first+q,b=point/n;
    if(b!=cached){Load(b);cached=b;}
    float px=static_cast<float>(points.GetValue(uint64_t(point)*3)),py=static_cast<float>(points.GetValue(uint64_t(point)*3+1)),pz=static_cast<float>(points.GetValue(uint64_t(point)*3+2));
    float best=3.402823466e38F;uint32_t index=0;
    for(uint32_t j=0;j<m;++j){
     float ax=F(0).GetValue(j),ay=F(1).GetValue(j),az=F(2).GetValue(j);
     float dx=F(3).GetValue(j),dy=F(4).GetValue(j),dz=F(5).GetValue(j);
     float l=dx*dx+dy*dy+dz*dz,t=0.0F;
     if(l>0.0F){t=((px-ax)*dx+(py-ay)*dy+(pz-az)*dz)/l;if(t<0)t=0;if(t>1)t=1;}
     float ex=px-(ax+t*dx),ey=py-(ay+t*dy),ez=pz-(az+t*dz);
     float d=ex*ex+ey*ey+ez*ez;
     if(d<best){best=d;index=j;}
    }
    auto red=minResult.Get<float>();red.SetValue(0,best);Sync<HardEvent::S_V>();Sqrt(red,red,1);Sync<HardEvent::V_S>();
    outFloat.Get<float>().SetValue(q,red.GetValue(0));outIdx.Get<int32_t>().SetValue(q,static_cast<int32_t>(index));Sync<HardEvent::S_V>();
   }
   auto output=outNative.Get<T>();
   if constexpr(sizeof(T)==2)Cast(output,outFloat.Get<float>(),RoundMode::CAST_RINT,count);else DataCopy(output,outFloat.Get<float>(),(count+7)/8*8);
   Sync<HardEvent::V_MTE3>();Sync<HardEvent::S_MTE3>();
   DataCopyPad(distOut[first],output,DataCopyExtParams{1,static_cast<uint32_t>(count*sizeof(T)),0,0,0});
   DataCopyPad(idxOut[first],outIdx.Get<int32_t>(),DataCopyExtParams{1,count*4,0,0,0});Sync<HardEvent::MTE3_V>();
   Sync<HardEvent::MTE3_S>();Sync<HardEvent::V_S>();
  }
 }
private:
 __aicore__ inline LocalTensor<float> F(uint32_t i){return arena.Get<float>()[i*ma];}
 __aicore__ inline void Load(uint32_t b){
  for(uint32_t j=0;j<m;++j){
   uint64_t off=(uint64_t(b)*m+j)*6;
   float ax=static_cast<float>(segments.GetValue(off)),ay=static_cast<float>(segments.GetValue(off+1)),az=static_cast<float>(segments.GetValue(off+2));
   float cx=static_cast<float>(segments.GetValue(off+3)),cy=static_cast<float>(segments.GetValue(off+4)),cz=static_cast<float>(segments.GetValue(off+5));
   F(0).SetValue(j,ax);F(1).SetValue(j,ay);F(2).SetValue(j,az);F(3).SetValue(j,cx-ax);F(4).SetValue(j,cy-ay);F(5).SetValue(j,cz-az);
  }
 }
 template<HardEvent E> __aicore__ inline void Sync(){auto id=pipe.FetchEventID(E);SetFlag<E>(id);WaitFlag<E>(id);}
 TPipe pipe;TBuf<TPosition::VECCALC> arena,bitmap,minResult,outFloat,outNative,outIdx;
 GlobalTensor<T> points,segments,distOut;GlobalTensor<int32_t> idxOut;uint32_t n,m,batch,blocks,ma;
};
template<typename DT_TREE_POINTS> __global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points,GM_ADDR line_segments,GM_ADDR min_distance,GM_ADDR nearest_segment_idx,GM_ADDR workspace,GM_ADDR tiling){
 REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData,t,tiling);
 TreeKernel<DT_TREE_POINTS> op;op.Init(tree_points,line_segments,min_distance,nearest_segment_idx,t);op.Process();
}
