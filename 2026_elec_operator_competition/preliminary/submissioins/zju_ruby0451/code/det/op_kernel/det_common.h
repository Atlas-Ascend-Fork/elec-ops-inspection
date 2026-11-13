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

#ifndef RUBY0451_DET_DET_COMMON_H
#define RUBY0451_DET_DET_COMMON_H

#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "linalg_det_tiling.h"
#include "tiling_key_linalg_det.h"
using namespace AscendC;
union DetFloatBits {float f;uint32_t u;};
template<HardEvent E> __aicore__ inline void DetFence(){
  event_t id=static_cast<event_t>(0);SetFlag<E>(id);WaitFlag<E>(id);
 }
__aicore__ inline void DetMakeStrides(LocalTensor<int32_t> target,uint32_t count,uint32_t step){
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ int32_t*>(target.GetPhyAddr())[i]=int32_t(i*step);
  DetFence<HardEvent::S_V>();
  for(uint32_t filled=8;filled<count;filled*=2){
   uint32_t len=count-filled<filled?count-filled:filled;
   Adds(target[filled],target,int32_t(filled*step),len);PipeBarrier<PIPE_V>();
  }
 }
__aicore__ inline uint32_t DetParity(uint64_t x){
  x^=x>>8;x^=x>>4;return (0x6996u>>(uint32_t(x)&15))&1;
 }
__aicore__ inline float DetFinish(float product,int32_t exponent){return product;}
__aicore__ inline void DetMultiply(float value,float& product,int32_t& exponent){product*=value;}
__aicore__ inline uint32_t DetParity1(uint64_t x){x^=x>>32;x^=x>>16;x^=x>>8;x^=x>>4;return(0x6996u>>(uint32_t(x)&15))&1;}
__aicore__ inline float DetFinish1(float product,int32_t exponent){
  DetFloatBits b;if(exponent>127){b.u=product<0?0xff800000u:0x7f800000u;return b.f;}
  if(exponent>=-126){b.u=uint32_t(exponent+127)<<23;return product*b.f;}if(exponent < -150)return 0;
  b.u=uint32_t(exponent+151)<<23;return product*b.f*5.960464477539063e-8f;
 }
__aicore__ inline void DetMultiply1(float value,float& product,int32_t& exponent){
  DetFloatBits b;b.f=value;int32_t e=int32_t((b.u>>23)&255)-127;
  if(e==-127){b.f=value*16777216.0f;e=int32_t((b.u>>23)&255)-127-24;}
  b.u=(b.u&0x807fffffu)|0x3f800000u;product*=b.f;exponent+=e;if(product>=2.0f||product<=-2.0f){product*=.5f;++exponent;}
 }
__aicore__ inline void DetMakeStrides1(LocalTensor<int32_t> target,uint32_t count,uint32_t step){
  for(uint32_t i=0;i<8;++i)target.SetValue(i,int32_t(i*step));
  DetFence<HardEvent::S_V>();
  for(uint32_t filled=8;filled<count;filled*=2){
   uint32_t len=count-filled<filled?count-filled:filled;
   Adds(target[filled],target,int32_t(filled*step),len);PipeBarrier<PIPE_V>();
  }
 }
__aicore__ inline float DetFinish2(float product,int32_t exponent){
  DetFloatBits b;if(exponent>127){b.u=product<0?0xff800000u:0x7f800000u;return b.f;}
  if(exponent>=-126){b.u=uint32_t(exponent+127)<<23;return product*b.f;}
  if(exponent < -150)return 0.0f;
  b.u=uint32_t(exponent+151)<<23;return (product*b.f)*5.960464477539063e-8f;
 }


// Resident column-major elimination. The row permutation is logical, not copied.
__aicore__ inline void DetInitTinyBuffers(LocalTensor<float>& mat,LocalTensor<float>& loaded,LocalTensor<uint32_t>& indices,LocalTensor<float>& pivot,LocalTensor<float>& spread,LocalTensor<int32_t>& bases,LocalTensor<float>& paired,LocalTensor<float>& pairSpread,LocalTensor<float>& absolute,LocalTensor<float>& reduced,LocalTensor<float>& result){
  mat=LocalTensor<float>(TPosition::VECCALC,0,256);
  loaded=LocalTensor<float>(TPosition::VECCALC,1024,256);
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,2048,256);
  pivot=LocalTensor<float>(TPosition::VECCALC,3072,16);
  spread=LocalTensor<float>(TPosition::VECCALC,3136,128);
  bases=LocalTensor<int32_t>(TPosition::VECCALC,4352,32);
  paired=LocalTensor<float>(TPosition::VECCALC,4480,32);
  pairSpread=LocalTensor<float>(TPosition::VECCALC,4608,256);
  absolute=LocalTensor<float>(TPosition::VECCALC,3712,128);
  reduced=LocalTensor<float>(TPosition::VECCALC,4224,8);
  result=LocalTensor<float>(TPosition::VECCALC,4256,8);
}
__aicore__ inline void DetInitColumnBuffers(uint32_t cols,uint32_t pitch,uint32_t ioCount,uint32_t& offset,LocalTensor<float>& mat,LocalTensor<float>& loaded,LocalTensor<float>& transposed,LocalTensor<uint32_t>& indices,LocalTensor<float>& absolute,LocalTensor<float>& reduced,LocalTensor<float>& pivot,LocalTensor<float>& spread){
  mat=LocalTensor<float>(TPosition::VECCALC,offset,cols*pitch);offset+=cols*pitch*4;
  loaded=LocalTensor<float>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  transposed=LocalTensor<float>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  absolute=LocalTensor<float>(TPosition::VECCALC,offset,128);offset+=512;
  reduced=LocalTensor<float>(TPosition::VECCALC,offset,8);offset+=32;
  pivot=LocalTensor<float>(TPosition::VECCALC,offset,cols);offset+=cols*4;
  spread=LocalTensor<float>(TPosition::VECCALC,offset,cols*8);offset+=cols*32;
}


__aicore__ inline void DetInitWideTinyBuffers(LocalTensor<float>& mat,LocalTensor<float>& loaded,LocalTensor<uint32_t>& indices,LocalTensor<float>& pivot,LocalTensor<float>& spread,LocalTensor<int32_t>& bases,LocalTensor<float>& paired,LocalTensor<float>& pairSpread,LocalTensor<float>& factorWide,LocalTensor<float>& absolute,LocalTensor<float>& reduced,LocalTensor<float>& result){
  mat=LocalTensor<float>(TPosition::VECCALC,0,256);
  loaded=LocalTensor<float>(TPosition::VECCALC,1024,256);
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,2048,256);
  pivot=LocalTensor<float>(TPosition::VECCALC,3072,16);
  spread=LocalTensor<float>(TPosition::VECCALC,3136,128);
  bases=LocalTensor<int32_t>(TPosition::VECCALC,5760,64);
  paired=LocalTensor<float>(TPosition::VECCALC,6016,64);
  pairSpread=LocalTensor<float>(TPosition::VECCALC,6272,512);
  factorWide=LocalTensor<float>(TPosition::VECCALC,8320,64);
  absolute=LocalTensor<float>(TPosition::VECCALC,3712,128);
  reduced=LocalTensor<float>(TPosition::VECCALC,4224,8);
  result=LocalTensor<float>(TPosition::VECCALC,4256,8);
}



__aicore__ inline void DetInitTinyIndices(LocalTensor<int32_t> bases,LocalTensor<uint32_t> indices,uint32_t n,uint32_t pitch){
  if(n==pitch){
  #pragma unroll
  for(uint32_t i=0;i<8;++i)bases.ReinterpretCast<uint64_t>().SetValue(i,uint64_t(i*64)|(uint64_t(i*64+4)<<32));
  DetFence<HardEvent::S_V>();Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  }else{
  auto offsets=indices.ReinterpretCast<int32_t>();
  #pragma unroll
  for(uint32_t i=0;i<8;++i){offsets.SetValue(i,int32_t(i*n*4));bases.ReinterpretCast<uint64_t>().SetValue(i,uint64_t(i*64)|(uint64_t(i*64+4)<<32));}
  DetFence<HardEvent::S_V>();
  Adds(offsets[8],offsets,int32_t(8*n*4),8);Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  #pragma unroll
  for(uint32_t filled=16;filled<256;filled*=2){
   Adds(offsets[filled],offsets,int32_t(filled/16*4),filled);PipeBarrier<PIPE_V>();
  }
  }
}



__aicore__ inline void DetExtendTinyIndices(LocalTensor<int32_t> offsets,LocalTensor<int32_t> bases,uint32_t n){
  DetFence<HardEvent::S_V>();
  Adds(offsets[8],offsets,int32_t(8*n*4),8);Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  #pragma unroll
  for(uint32_t filled=16;filled<256;filled*=2){
   Adds(offsets[filled],offsets,int32_t(filled/16*4),filled);PipeBarrier<PIPE_V>();
  }
}



__aicore__ inline void DetInitPairBuffers(uint32_t& offset,LocalTensor<int32_t>& bases,LocalTensor<float>& paired,LocalTensor<float>& pairSpread,LocalTensor<float>& result){
  bases=LocalTensor<int32_t>(TPosition::VECCALC,offset,128);offset+=512;
  paired=LocalTensor<float>(TPosition::VECCALC,offset,128);offset+=512;
  pairSpread=LocalTensor<float>(TPosition::VECCALC,offset,1024);offset+=4096;
  result=LocalTensor<float>(TPosition::VECCALC,offset,8);
}



__aicore__ inline void DetRemapFlatIndices(LocalTensor<float> transposed,LocalTensor<uint32_t> indices,uint32_t n,bool& flatIndicesReady){
   auto temporary=transposed.ReinterpretCast<int32_t>();
   ShiftRight(temporary,indices.ReinterpretCast<int32_t>(),8,n*64);PipeBarrier<PIPE_V>();
   Muls(temporary,temporary,(int32_t(n)-64)*4,n*64);PipeBarrier<PIPE_V>();
   Add(indices.ReinterpretCast<int32_t>(),indices.ReinterpretCast<int32_t>(),temporary,n*64);PipeBarrier<PIPE_V>();
   flatIndicesReady=true;
}


__aicore__ inline float DetFourByFour(float a00,float a01,float a02,float a03,float a10,float a11,float a12,float a13,float a20,float a21,float a22,float a23,float a30,float a31,float a32,float a33){
    const float s0=a00*a11-a01*a10;
    const float c0=a20*a31-a21*a30;
    const float s1=a00*a12-a02*a10;
    const float c1=a20*a32-a22*a30;
    const float s2=a00*a13-a03*a10;
    const float c2=a20*a33-a23*a30;
    const float s3=a01*a12-a02*a11;
    const float c3=a21*a32-a22*a31;
    const float s4=a01*a13-a03*a11;
    const float c4=a21*a33-a23*a31;
    const float s5=a02*a13-a03*a12;
    const float c5=a22*a33-a23*a32;
 return s0*c5-s1*c4+s2*c3+s3*c2-s4*c1+s5*c0;
}

template<uint32_t ColumnBegin,uint32_t ActiveWidth,bool ODD>
__aicore__ inline void DetUpdatePairColumns(LocalTensor<float> mat,LocalTensor<float> factor,LocalTensor<float> paired,LocalTensor<float> pairSpread,uint32_t p,uint32_t remaining,float diagonal,float below,float cross,float future,float& product,float& cachedDiagonal,float& cachedInverse){
constexpr uint32_t pitch=72;
    if(remaining){
     if(!ODD){PipeBarrier<PIPE_V>();Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});}
     float divisor=diagonal;
     float inverse=ODD?cachedInverse:-1.0f/divisor;
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())+ColumnBegin,
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())+ColumnBegin,
       inverse,uint64_t(ActiveWidth),uint8_t(1),UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
     MulAddDst<float,float,false>(mat[(p+1)*pitch+ColumnBegin],factor[ColumnBegin],pairSpread[ODD?24:0],uint64_t(ActiveWidth),remaining,
       BinaryRepeatParams{1,1,0,9,0,2});
     if(!ODD){
      float normalizedBelow=below*inverse;
      cachedDiagonal=normalizedBelow*cross+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // VMLA reads only even coefficient blocks. This writes disjoint odd blocks.
      // The existing loop-end V barrier covers both before the next dependent update.
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,normalizedBelow,uint8_t((remaining+7)/8),
        uint16_t(2),uint16_t(2),uint16_t(16),uint16_t(16));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
}



__aicore__ inline void DetPredictTinyDiagonal(LocalTensor<float> pairSpread,float below,float inverse,float cross,float future,uint32_t remaining,float& cachedDiagonal,float& cachedInverse){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      DetFloatBits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      Axpy(pairSpread[8],pairSpread,float(normalizedBelow),uint64_t(8),remaining,
        UnaryRepeatParams{1,1,2,2});
}



__aicore__ inline void DetLoadTiny(GlobalTensor<float> input,LocalTensor<float> mat,LocalTensor<float> loaded,LocalTensor<uint32_t> indices,uint64_t offset,uint32_t n,uint32_t pitch,uint32_t cols){
  if(n==pitch){
   DetFence<HardEvent::V_MTE2>();DataCopy(mat,input[offset],n*n);DetFence<HardEvent::MTE2_V>();return;
  }
  DetFence<HardEvent::S_V>();Duplicate(loaded,0.0f,cols*pitch);DetFence<HardEvent::V_MTE2>();
  DataCopyPad(loaded,input[offset],DataCopyExtParams{1,n*n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
  DetFence<HardEvent::MTE2_V>();Gather(mat,loaded,indices,0,cols*pitch);PipeBarrier<PIPE_V>();
}


template<class Kernel>
__aicore__ inline void DetProcessTiny(Kernel& op,uint32_t n,uint32_t pitch,uint32_t cores,uint64_t batches,LocalTensor<float>& factor,LocalTensor<float> mat,LocalTensor<float> result,GlobalTensor<float> output){
  if(GetBlockIdx()>=cores)return;
  uint32_t owners=batches<cores?uint32_t(batches):cores;
  if(GetBlockIdx()>=owners)return;
  for(uint64_t batch=GetBlockIdx()%owners;batch<batches;batch+=owners){
   op.Load(batch*n*n);
   float product=1.0f,cachedDiagonal=0.0f,cachedInverse=0.0f;bool singular=false;
   #pragma unroll
   for(uint32_t p=0;p<16;++p){
    if(p>=n)break;
    factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if((p&1)==0){
     DetFence<HardEvent::V_S>();
     if(remaining){
      uint64_t first=factor.ReinterpretCast<uint64_t>().GetValue(p/2);
      uint64_t second=mat[(p+1)*pitch].ReinterpretCast<uint64_t>().GetValue(p/2);
      DetFloatBits word;word.u=uint32_t(first);diagonal=word.f;
      word.u=uint32_t(first>>32);below=word.f;
      word.u=uint32_t(second);cross=word.f;
      word.u=uint32_t(second>>32);future=word.f;
     }else diagonal=factor.GetValue(p);
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
    op.UpdateTrailingColumns(p,remaining,diagonal,below,cross,future,cachedDiagonal,cachedInverse);
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
   }
   DetFence<HardEvent::V_S>();result.SetValue(0,singular?0.0f:product);DetFence<HardEvent::S_MTE3>();
   if(GetBlockIdx()<owners)DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0});DetFence<HardEvent::MTE3_S>();
  }
}


__aicore__ inline void DetInitTinyState(GM_ADDR A,GM_ADDR Y,GlobalTensor<float>& input,GlobalTensor<float>& output,uint32_t n,uint32_t pitch,LocalTensor<float>& mat,LocalTensor<float>& loaded,LocalTensor<uint32_t>& indices,LocalTensor<float>& pivot,LocalTensor<float>& spread,LocalTensor<int32_t>& bases,LocalTensor<float>& paired,LocalTensor<float>& pairSpread,LocalTensor<float>& absolute,LocalTensor<float>& reduced,LocalTensor<float>& result){
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  DetInitTinyBuffers(mat,loaded,indices,pivot,spread,bases,paired,pairSpread,absolute,reduced,result);
    DetInitTinyIndices(bases,indices,n,pitch);
}


#endif
