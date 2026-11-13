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

#ifndef RUBY0451_DET_DET_TINY_BOUND_H
#define RUBY0451_DET_DET_TINY_BOUND_H
#include "det_common.h"
template<bool TERMINAL>
class KernelTinyBoundPadded16 {
public:
template<bool ODD>
__aicore__ inline void EliminatePairPivot(uint32_t p,float& product,float& cachedDiagonal,float& cachedInverse){
    factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if(!ODD){
     DetFence<HardEvent::V_S>();
     if(remaining){
      diagonal=factor.GetValue(p);
      below=factor.GetValue(p+1);
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p+1];
     }else diagonal=factor.GetValue(p);
    }else diagonal=cachedDiagonal;
    if(remaining){
     if(!ODD)Gather(paired,mat,bases.ReinterpretCast<uint32_t>(),p*4,64);
     float divisor=diagonal;
     float inverse=ODD?cachedInverse:-1.0f/divisor;
     // Gather only reads matrix words;factorWide is separate from both matrix and paired coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factorWide.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(4),UnaryRepeatParams{1,1,2,0}); // Consumer synchronization must retain the required V dependency.
     if(!ODD){
      Brcb(pairSpread,paired,8,BrcbRepeatParams{1,8}); // Consumer synchronization must retain the required V dependency.
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     // Retired rows may change; only live diagonal and trailing rows are consumed.
     MulAddDst(mat,factorWide,pairSpread[ODD?256:0],uint64_t(64),4,
       BinaryRepeatParams{1,1,1,8,0,8});
     if(!ODD){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Preserve SDK count-mask preparation, bypass only the annotated tensor interface.
      // The first256 coefficient words are read-only;only the second256 are written.
if constexpr(TERMINAL){
      SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
      vaxpy(reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr())+256,
        reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr()),float(normalizedBelow),uint8_t(4),
        uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
}else{
      AscendC::AxpyImpl<float,float>(reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr())+256,
        reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr()),float(normalizedBelow),int32_t(256));
}
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
}

 __aicore__ inline ~KernelTinyBoundPadded16(){PipeBarrier<PIPE_ALL>();}

 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
    DetInitWideTinyBuffers(mat,loaded,indices,pivot,spread,bases,paired,pairSpread,factorWide,absolute,reduced,result);
  // DMA overwrites every word contributing to a live matrix entry.
  DataCopyPad(loaded,input,DataCopyExtParams{1,n*n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
  if(n==pitch){
  #pragma unroll
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=uint64_t(i*64)|(uint64_t(i*64)<<32);
  DetFence<HardEvent::S_V>();Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  }else{
  auto offsets=indices.ReinterpretCast<int32_t>();
  #pragma unroll
  for(uint32_t i=0;i<8;++i){reinterpret_cast<__ubuf__ int32_t*>(offsets.GetPhyAddr())[i]=int32_t(i*n*4);reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=uint64_t(i*64)|(uint64_t(i*64)<<32);}
    DetExtendTinyIndices(offsets,bases,n);
  }
  Adds(bases[32],bases,int32_t(4),32);PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Load(uint64_t offset){
  // This class is reachable only for one matrix on owner0; Init issued its DMA.
  DetFence<HardEvent::MTE2_V>();
  Gather(mat,loaded,indices,0,cols*pitch);PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Process(){
  // Bound entry admits only owner0. Keep the pivot loop rolled.
  {
   constexpr uint64_t batch=0;
   Load(0);
   float product=1.0f,cachedDiagonal=0.0f,cachedInverse=0.0f;
   // Keep one compact pair body and precompute the odd inverse once.
   #pragma clang loop unroll(disable)
   for(uint32_t pairBase=0;pairBase+1<(TERMINAL && n==15?12:n);pairBase+=2){
    {const uint32_t p=pairBase;
    EliminatePairPivot<false>(p,product,cachedDiagonal,cachedInverse);
       }
    {const uint32_t p=pairBase+1;
    EliminatePairPivot<true>(p,product,cachedDiagonal,cachedInverse);
       }
   }
   if(TERMINAL && n==15){
    DetFence<HardEvent::V_S>();
    auto* words=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr());
    const float a00=words[12*pitch+12],a01=words[12*pitch+13],a02=words[12*pitch+14];
    const float a10=words[13*pitch+12],a11=words[13*pitch+13],a12=words[13*pitch+14];
    const float a20=words[14*pitch+12],a21=words[14*pitch+13],a22=words[14*pitch+14];
    product*=a00*(a11*a22-a12*a21)-a01*(a10*a22-a12*a20)+a02*(a10*a21-a11*a20);
   }else if(n&1){
    DetFence<HardEvent::V_S>();
    // The preceding explicit V_S covers the final matrix value.
    product*=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(n-1)*pitch+n-1];
   }
   *reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr())=product;DetFence<HardEvent::S_MTE3>();
   DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0}); // Final PIPE_ALL covers output completion.
  }
 }
private:
 LocalTensor<float> mat,loaded,factor,absolute,reduced,pivot,spread,result,paired,pairSpread,factorWide;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 static constexpr uint32_t pitch=16,cols=16;uint32_t n,cores;uint64_t batches;
};
using KernelTinyBoundPrefetchPadded16=KernelTinyBoundPadded16<false>;
using KernelTinyBoundTerminal15=KernelTinyBoundPadded16<true>;




class KernelTinyBoundPrefetch16 {
public:

 __aicore__ inline void FinishSchurComplement(float& product) {
     DetFence<HardEvent::V_S>();
     auto* words=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr());
     const float a00=words[12*pitch+12];
     const float a01=words[12*pitch+13];
     const float a02=words[12*pitch+14];
     const float a03=words[12*pitch+15];
     const float a10=words[13*pitch+12];
     const float a11=words[13*pitch+13];
     const float a12=words[13*pitch+14];
     const float a13=words[13*pitch+15];
     const float a20=words[14*pitch+12];
     const float a21=words[14*pitch+13];
     const float a22=words[14*pitch+14];
     const float a23=words[14*pitch+15];
     const float a30=words[15*pitch+12];
     const float a31=words[15*pitch+13];
     const float a32=words[15*pitch+14];
     const float a33=words[15*pitch+15];
    const float determinant=DetFourByFour(a00,a01,a02,a03,a10,a11,a12,a13,a20,a21,a22,a23,a30,a31,a32,a33);
     product*=determinant;

}

 template<uint32_t P>
 __aicore__ inline void EliminatePivot(float& product,float& cachedDiagonal,float& cachedInverse){


    factor=mat[P*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-P-1;
    if((P&1)==0){
     DetFence<HardEvent::V_S>();
     if(remaining){
      diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[P];
      below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[P+1];
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(P+1)*pitch+P];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(P+1)*pitch+P+1];
     }else diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[P];
    }else diagonal=cachedDiagonal;
    if(remaining){
     if((P&1)==0)AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())+(P+1)*pitch,
       reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),256,
       P*4,uint64_t(2*remaining),uint8_t(1),uint16_t(8));
     float divisor=diagonal;
     float inverse=(P&1)?cachedInverse:-1.0f/divisor;
     // Gather reads trailing rows only;the current factor row is disjoint.
     // Keep the existing following V barrier before consuming gathered coefficients.
     AscendC::MulsImpl<float>(reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),
       reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr()),inverse,uint64_t(16),uint8_t(1),UnaryRepeatParams{1,1,8,8});if((P&1)==0)PipeBarrier<PIPE_V>();
     if((P&1)==0){
      Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
     }
     // Odd pivot skips the first column and selects its updated second coefficient.
     MulAddDst(mat[(P+1)*pitch],factor,pairSpread[(P&1)?24:0],n,remaining,
       BinaryRepeatParams{1,1,0,2,0,2});
     if((P&1)==0){
      volatile float normalizedBelow=below*inverse;
      volatile float roundedProduct=normalizedBelow*cross;
      cachedDiagonal=roundedProduct+future;
      cachedInverse=-1.0f/cachedDiagonal;
      // Even coefficient blocks are read-only; update adjacent odd blocks in place.
      // Even32B blocks are read by VMLA;only the disjoint odd blocks are updated.
      // Retain the following loop-end barrier before the dependent matrix update.
      SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xff));
      auto* coefficientWords=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
      vaxpy(coefficientWords+8,coefficientWords,float(normalizedBelow),uint8_t(remaining),
        uint16_t(1),uint16_t(1),uint16_t(2),uint16_t(2));
     }
    }
    product*=diagonal;PipeBarrier<PIPE_V>();

 }

 __aicore__ inline ~KernelTinyBoundPrefetch16(){PipeBarrier<PIPE_ALL>();}

 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  DetInitTinyBuffers(mat,loaded,indices,pivot,spread,bases,paired,pairSpread,absolute,reduced,result);
  DataCopy(mat,input,256);
  if(n==pitch){
  #pragma unroll
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=uint64_t(i*64)|(uint64_t(i*64+4)<<32);
  DetFence<HardEvent::S_V>();Adds(bases[16],bases,int32_t(512),16);PipeBarrier<PIPE_V>();
  }else{
  auto offsets=indices.ReinterpretCast<int32_t>();
  #pragma unroll
  for(uint32_t i=0;i<8;++i){reinterpret_cast<__ubuf__ int32_t*>(offsets.GetPhyAddr())[i]=int32_t(i*n*4);reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=uint64_t(i*64)|(uint64_t(i*64+4)<<32);}
    DetExtendTinyIndices(offsets,bases,n);
  }
 }
 __aicore__ inline void Load(uint64_t offset){
  // This class is reachable only for one matrix on owner0; Init issued its DMA.
  DetFence<HardEvent::MTE2_V>();
 }
 __aicore__ inline void Process(){
  // TASK_MODE2/3 bind one matrix; the kernel entry admits only logical block0.
  {
   constexpr uint64_t batch=0;
   Load(0);
   float product=1.0f,cachedDiagonal=0.0f,cachedInverse=0.0f;
   // Explicit code generation, not a requested compiler loop unroll.
   do {
    EliminatePivot<0>(product,cachedDiagonal,cachedInverse);
    EliminatePivot<1>(product,cachedDiagonal,cachedInverse);
    EliminatePivot<2>(product,cachedDiagonal,cachedInverse);
    EliminatePivot<3>(product,cachedDiagonal,cachedInverse);
    EliminatePivot<4>(product,cachedDiagonal,cachedInverse);
    EliminatePivot<5>(product,cachedDiagonal,cachedInverse);
    EliminatePivot<6>(product,cachedDiagonal,cachedInverse);
    EliminatePivot<7>(product,cachedDiagonal,cachedInverse);
    EliminatePivot<8>(product,cachedDiagonal,cachedInverse);
    EliminatePivot<9>(product,cachedDiagonal,cachedInverse);
    EliminatePivot<10>(product,cachedDiagonal,cachedInverse);
    EliminatePivot<11>(product,cachedDiagonal,cachedInverse);
    { // Determinant of the live final 4x4 Schur complement.
    FinishSchurComplement(product);
    }
   } while(false);
   *reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr())=product;DetFence<HardEvent::S_MTE3>();
   DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0}); // Final PIPE_ALL covers output completion.
  }
 }
private:
 LocalTensor<float> mat,loaded,factor,absolute,reduced,pivot,spread,result,paired,pairSpread;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 static constexpr uint32_t n=16,pitch=16,cols=16;uint32_t cores;uint64_t batches;
};


#endif
