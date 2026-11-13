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

#ifndef RUBY0451_DET_DET_TINY_H
#define RUBY0451_DET_DET_TINY_H
#include "det_common.h"
class KernelTinyDet {
public:

 __aicore__ inline void UpdateTrailingColumns(uint32_t p,uint32_t remaining,float diagonal,float below,float cross,float future,float& cachedDiagonal,float& cachedInverse) {
     if((p&1)==0)Gather(paired,mat[(p+1)*pitch],bases.ReinterpretCast<uint32_t>(),p*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(p&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((p&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(p+1)*pitch],factor,pairSpread[(p&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((p&1)==0){
    DetPredictTinyDiagonal(pairSpread,below,inverse,cross,future,remaining,cachedDiagonal,cachedInverse);
     }

}

 __aicore__ inline ~KernelTinyDet(){PipeBarrier<PIPE_ALL>();}

 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;batches=t.batches;cores=t.cores;
    DetInitTinyState(A,Y,input,output,n,pitch,mat,loaded,indices,pivot,spread,bases,paired,pairSpread,absolute,reduced,result);
 }
 __aicore__ inline void Load(uint64_t offset){
    DetLoadTiny(input,mat,loaded,indices,offset,n,pitch,cols);
 }
 union Bits {float f;uint32_t u;};



 __aicore__ inline void Process(){
    DetProcessTiny(*this,n,pitch,cores,batches,factor,mat,result,output);
 }
private:
 LocalTensor<float> mat,loaded,factor,absolute,reduced,pivot,spread,result,paired,pairSpread;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 static constexpr uint32_t pitch=16,cols=16;uint32_t n,cores;uint64_t batches;
};


class KernelTinyFullwidthDet {
public:

 __aicore__ inline void UpdateTrailingColumns(uint32_t p,uint32_t remaining,float diagonal,float below,float cross,float future,float& cachedDiagonal,float& cachedInverse) {
     if((p&1)==0)Gather(paired,mat,bases.ReinterpretCast<uint32_t>(),p*4,64);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(p&1)?cachedInverse:-1.0f/divisor;
     Muls(factorWide,factor,inverse,uint64_t(16),4,UnaryRepeatParams{1,1,2,0});PipeBarrier<PIPE_V>();
     if((p&1)==0){
      Brcb(pairSpread,paired,8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     // Retired rows may change; only live diagonal and trailing rows are consumed.
     MulAddDst(mat,factorWide,pairSpread[(p&1)?256:0],uint64_t(64),4,
       BinaryRepeatParams{1,1,1,8,0,8});
     if((p&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[256],pairSpread,float(normalizedBelow),256);
     }

}

 __aicore__ inline ~KernelTinyFullwidthDet(){PipeBarrier<PIPE_ALL>();}

 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
    DetInitWideTinyBuffers(mat,loaded,indices,pivot,spread,bases,paired,pairSpread,factorWide,absolute,reduced,result);
  if(n==pitch){
  #pragma unroll
  for(uint32_t i=0;i<8;++i)bases.ReinterpretCast<uint64_t>().SetValue(i,uint64_t(i*64)|(uint64_t(i*64)<<32));
  DetFence<HardEvent::S_V>();Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  }else{
  auto offsets=indices.ReinterpretCast<int32_t>();
  #pragma unroll
  for(uint32_t i=0;i<8;++i){offsets.SetValue(i,int32_t(i*n*4));bases.ReinterpretCast<uint64_t>().SetValue(i,uint64_t(i*64)|(uint64_t(i*64)<<32));}
    DetExtendTinyIndices(offsets,bases,n);
  }
  Adds(bases[32],bases,int32_t(4),32);PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Load(uint64_t offset){
    DetLoadTiny(input,mat,loaded,indices,offset,n,pitch,cols);
 }
 union Bits {float f;uint32_t u;};



 __aicore__ inline void Process(){
    DetProcessTiny(*this,n,pitch,cores,batches,factor,mat,result,output);
 }
private:
 LocalTensor<float> mat,loaded,factor,absolute,reduced,pivot,spread,result,paired,pairSpread,factorWide;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 static constexpr uint32_t pitch=16,cols=16;uint32_t n,cores;uint64_t batches;
};


class KernelTinyExplicit16 {
public:
 template<uint32_t P>
 __aicore__ inline void EliminatePivot(float& product,float& cachedDiagonal,float& cachedInverse,bool& singular){


    factor=mat[P*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-P-1;
    if((P&1)==0){
     DetFence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(P/2);
      uint64_t second=mat[(P+1)*pitch].ReinterpretCast<uint64_t>().GetValue(P/2);
      Bits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(P);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;return;}
    if(remaining){
     if((P&1)==0)Gather(paired,mat[(P+1)*pitch],bases.ReinterpretCast<uint32_t>(),P*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(P&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,pitch);PipeBarrier<PIPE_V>();
     if((P&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(P+1)*pitch],factor,pairSpread[(P&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((P&1)==0){
    DetPredictTinyDiagonal(pairSpread,below,inverse,cross,future,remaining,cachedDiagonal,cachedInverse);
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();

 }

 __aicore__ inline ~KernelTinyExplicit16(){PipeBarrier<PIPE_ALL>();}

 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  batches=t.batches;cores=t.cores;
    DetInitTinyState(A,Y,input,output,n,pitch,mat,loaded,indices,pivot,spread,bases,paired,pairSpread,absolute,reduced,result);
 }
 __aicore__ inline void Load(uint64_t offset){
    DetLoadTiny(input,mat,loaded,indices,offset,n,pitch,cols);
 }
 union Bits {float f;uint32_t u;};



 __aicore__ inline void Process(){
  if(GetBlockIdx()>=cores)return;
  uint32_t owners=batches<cores?uint32_t(batches):cores;
  if(GetBlockIdx()>=owners)return;
  for(uint64_t batch=GetBlockIdx()%owners;batch<batches;batch+=owners){
   Load(batch*n*n);
   float product=1.0f,cachedDiagonal=0.0f,cachedInverse=0.0f;bool singular=false;
   // Explicit code generation, not a requested compiler loop unroll.
   do {
    EliminatePivot<0>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<1>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<2>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<3>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<4>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<5>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<6>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<7>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<8>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<9>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<10>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<11>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<12>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<13>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<14>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
    EliminatePivot<15>(product,cachedDiagonal,cachedInverse,singular);if(singular)break;
   } while(false);
   DetFence<HardEvent::V_S>();result.SetValue(0,singular?0.0f:product);DetFence<HardEvent::S_MTE3>();
   if(GetBlockIdx()<owners)DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0});DetFence<HardEvent::MTE3_S>();
  }
 }
private:
 LocalTensor<float> mat,loaded,factor,absolute,reduced,pivot,spread,result,paired,pairSpread;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 static constexpr uint32_t n=16,pitch=16,cols=16;uint32_t cores;uint64_t batches;
};


#endif
