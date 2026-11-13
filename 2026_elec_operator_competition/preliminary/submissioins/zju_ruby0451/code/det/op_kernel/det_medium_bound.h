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

#ifndef RUBY0451_DET_DET_MEDIUM_BOUND_H
#define RUBY0451_DET_DET_MEDIUM_BOUND_H
#include "det_common.h"
#include "det_block.h"
class KernelMediumBoundCompact56 {
public:
 __aicore__ inline ~KernelMediumBoundCompact56(){PipeBarrier<PIPE_ALL>();}


 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;cols=(n+7)/8*8;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  uint32_t ioCount=cols*72,offset=0;
  mat=LocalTensor<float>(TPosition::VECCALC,offset,cols*72);offset+=cols*72*4;
  loaded=LocalTensor<float>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  transposed=LocalTensor<float>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,offset,ioCount);offset+=ioCount*4;
  absolute=LocalTensor<float>(TPosition::VECCALC,offset,128);offset+=512;
  reduced=LocalTensor<float>(TPosition::VECCALC,offset,8);offset+=32;
  pivot=LocalTensor<float>(TPosition::VECCALC,offset,cols);offset+=cols*4;
  spread=LocalTensor<float>(TPosition::VECCALC,offset,cols*8);offset+=cols*32;
    DetInitPairBuffers(offset,bases,paired,pairSpread,result);
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=
    uint64_t(2*i*pitch*4)|(uint64_t((2*i+1)*pitch*4)<<32);
  DetFence<HardEvent::S_V>();
  for(uint32_t filled=16;filled<64;filled*=2){
   Adds(bases[filled],bases,int32_t(filled*pitch*4),filled);PipeBarrier<PIPE_V>();
  }
  Adds(bases[64],bases,int32_t(4),64);PipeBarrier<PIPE_V>();

 }
 __aicore__ inline void Load(uint64_t offset){
  // First input DMA has no prior writer or reader of loaded.
  DataCopyPad(loaded,input[offset],DataCopyExtParams{1,n*n*4,0,0,0},
    DataCopyPadExtParams<float>{true,0,uint8_t((8-n*n%8)%8),0.0f});

  if(!flatIndicesReady){
   // Only the live matrix cells contribute to the determinant.
   // Gather overwrites every live cell; padded lanes never feed live coefficients.

   // Build one 56-word row, then replicate the row map with GM stride n.
   auto offsets=indices.ReinterpretCast<int32_t>();
   DetMakeStrides(offsets,pitch,4);
   for(uint32_t rows=1;rows<n;rows*=2){
    uint32_t live=n-rows<rows?n-rows:rows;
    Adds(offsets[rows*pitch],offsets,int32_t(rows*n*4),live*pitch);
    PipeBarrier<PIPE_V>();
   }
   flatIndicesReady=true;
  }
  DetFence<HardEvent::MTE2_V>();
  Gather(mat,loaded,indices,0,n*pitch);PipeBarrier<PIPE_V>();
 }
 template<uint32_t ColumnBegin,uint32_t ActiveWidth>
 __aicore__ inline void Eliminate(uint32_t first,uint32_t last,float& product){
  auto* words=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr());
  auto* coefficients=reinterpret_cast<__ubuf__ float*>(paired.GetPhyAddr());
  auto* expanded=reinterpret_cast<__ubuf__ float*>(pairSpread.GetPhyAddr());
  for(uint32_t p=first;p<last;p+=2){
   const uint32_t remaining=n-p-1;
   SetFlag<HardEvent::V_S>(static_cast<event_t>(0));
   // Fixed phase mask; dead extra rows stay within owned matrix/input UB and never enter live coefficients.
   AscendC::GatherImpl<float>(coefficients,words+(p+1)*pitch,
     reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr()),n*pitch,
     p*4,uint64_t(ColumnBegin==0?64:32),uint8_t(2),uint16_t(8));
   WaitFlag<HardEvent::V_S>(static_cast<event_t>(0));
   const float diagonal=words[p*pitch+p];
   const float below=words[p*pitch+p+1];
   const float cross=words[(p+1)*pitch+p];
   const float future=words[(p+1)*pitch+p+1];
   const float inverse=-1.0f/diagonal;
   // Gather must complete before Brcb. Factor scaling has independent storage.
   PipeBarrier<PIPE_V>();
   AscendC::BrcbImpl<float>(expanded,coefficients,uint8_t((remaining+7)/8),BrcbRepeatParams{1,8});
   AscendC::MulsImpl<float>(words+p*pitch+ColumnBegin,words+p*pitch+ColumnBegin,
     inverse,uint64_t(ActiveWidth),uint8_t(1),UnaryRepeatParams{1,1,8,8});
   PipeBarrier<PIPE_V>();
   AscendC::MulAddDstImpl<float,float>(words+(p+1)*pitch+ColumnBegin,
     words+p*pitch+ColumnBegin,expanded,uint64_t(ActiveWidth),uint8_t(remaining),
     BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,1});
   const float normalizedBelow=below*inverse;
   const float nextDiagonal=normalizedBelow*cross+future;
   const float nextInverse=-1.0f/nextDiagonal;
   // Compact odd coefficient writes are disjoint from the first matrix update.
   SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
   vaxpy(coefficients+64,coefficients,float(normalizedBelow),uint8_t(1),
     uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));
   PipeBarrier<PIPE_V>();
   if(remaining>1){
    // Odd broadcast and factor scaling can share their producer join.
    AscendC::BrcbImpl<float>(expanded+512,coefficients+64,uint8_t((remaining+7)/8),BrcbRepeatParams{1,8});
    AscendC::MulsImpl<float>(words+(p+1)*pitch+ColumnBegin,words+(p+1)*pitch+ColumnBegin,
      nextInverse,uint64_t(ActiveWidth),uint8_t(1),UnaryRepeatParams{1,1,8,8});
    PipeBarrier<PIPE_V>();
    AscendC::MulAddDstImpl<float,float>(words+(p+2)*pitch+ColumnBegin,
      words+(p+1)*pitch+ColumnBegin,expanded+520,uint64_t(ActiveWidth),uint8_t(remaining-1),
      BinaryRepeatParams{1,1,0,uint8_t(pitch/8),0,1});
   }
   product*=diagonal;
   product*=nextDiagonal;
   PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void Process(uint64_t batch){
  // The bound entry checked this owner; batches<=cores guarantees one matrix.
  {
   Load(batch*n*n);
   float product=1.0f;
   Eliminate<0,56>(0,24,product);
   Eliminate<24,32>(24,n-4,product);
   { // Direct determinant of the remaining 4x4 Schur complement.
    DetFence<HardEvent::V_S>();
    const uint32_t tail=n-4;
    auto* words=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr());
    const float a00=words[(tail+0)*pitch+tail+0];
    const float a01=words[(tail+0)*pitch+tail+1];
    const float a02=words[(tail+0)*pitch+tail+2];
    const float a03=words[(tail+0)*pitch+tail+3];
    const float a10=words[(tail+1)*pitch+tail+0];
    const float a11=words[(tail+1)*pitch+tail+1];
    const float a12=words[(tail+1)*pitch+tail+2];
    const float a13=words[(tail+1)*pitch+tail+3];
    const float a20=words[(tail+2)*pitch+tail+0];
    const float a21=words[(tail+2)*pitch+tail+1];
    const float a22=words[(tail+2)*pitch+tail+2];
    const float a23=words[(tail+2)*pitch+tail+3];
    const float a30=words[(tail+3)*pitch+tail+0];
    const float a31=words[(tail+3)*pitch+tail+1];
    const float a32=words[(tail+3)*pitch+tail+2];
    const float a33=words[(tail+3)*pitch+tail+3];
    product*=DetFourByFour(a00,a01,a02,a03,a10,a11,a12,a13,a20,a21,a22,a23,a30,a31,a32,a33);
   }
   *reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr())=product;DetFence<HardEvent::S_MTE3>();
   DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0}); // Final PIPE_ALL covers output completion.
  }
 }
private:
 LocalTensor<float> mat,loaded,transposed,factor,absolute,reduced,pivot,spread,result,paired,pairSpread;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 bool flatIndicesReady=false;
 static constexpr uint32_t pitch=56;uint32_t n,cols,cores;uint64_t batches;
};

class KernelMediumBoundStaticSuffix64 {
public:
template<uint32_t ColumnBegin,uint32_t ActiveWidth,bool ODD>
__aicore__ inline void EliminatePairPivot(uint32_t p,float& product,float& cachedDiagonal,float& cachedInverse){
    factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if(!ODD){
     // The event covers prior matrix producers; Gather and scalar reads only read them.
     event_t priorMatrixReady=static_cast<event_t>(0);
     SetFlag<HardEvent::V_S>(priorMatrixReady);
     if(remaining){
      // Matrix plus the initialized input copy are owned by this core.
      // Extra gathered words never enter a live coefficient or matrix update.
      auto readable=LocalTensor<float>(TPosition::VECCALC,0,64*72+64*64);
      Gather(paired,readable[(p+1)*pitch],bases.ReinterpretCast<uint32_t>(),p*4,
        uint64_t(64),ColumnBegin==0?2:1,8);
     }
     WaitFlag<HardEvent::V_S>(priorMatrixReady);
     diagonal=factor.GetValue(p);
     if(remaining){
      below=factor.GetValue(p+1);
      cross=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p];
      future=reinterpret_cast<__ubuf__ float*>(mat.GetPhyAddr())[(p+1)*pitch+p+1];
     }
    }else diagonal=cachedDiagonal;
    // No per-pivot zero guard in this bound performance path.
    DetUpdatePairColumns<ColumnBegin,ActiveWidth,ODD>(mat,factor,paired,pairSpread,p,remaining,diagonal,below,cross,future,product,cachedDiagonal,cachedInverse);
}

 __aicore__ inline ~KernelMediumBoundStaticSuffix64(){PipeBarrier<PIPE_ALL>();}


 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;cols=(n+7)/8*8;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  uint32_t ioCount=n<=64?cols*pitch:8*pitch,offset=0;
  DetInitColumnBuffers(cols,pitch,ioCount,offset,mat,loaded,transposed,indices,absolute,reduced,pivot,spread);
    DetInitPairBuffers(offset,bases,paired,pairSpread,result);
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ uint64_t*>(bases.GetPhyAddr())[i]=uint64_t(i*pitch*4)|(uint64_t(i*pitch*4+4)<<32);
  DetFence<HardEvent::S_V>();
  for(uint32_t filled=16;filled<128;filled*=2){Adds(bases[filled],bases,int32_t(filled/2*pitch*4),filled);PipeBarrier<PIPE_V>();}

 }
 __aicore__ inline void Load(uint64_t offset){
  if((n&7)==0){
   // Live row words are contiguous in GM; inactive row padding is never consumed.
   // First input DMA has no prior writer or reader of loaded.
   DataCopy(loaded,input[offset],n*n);
   DetFence<HardEvent::MTE2_V>();
   DataCopy(mat,loaded,DataCopyParams{uint16_t(n),uint16_t(n/8),0,uint16_t((pitch-n)/8)});
   PipeBarrier<PIPE_V>();
   return;
  }
  // The large contiguous transfer runs while the shape-only address map is built.
  // First input DMA has no prior writer or reader of loaded.
  DataCopyPad(loaded,input[offset],DataCopyExtParams{1,n*n*4,0,0,0},
    DataCopyPadExtParams<float>{true,0,uint8_t((8-n*n%8)%8),0.0f});
  Duplicate(loaded[(n*n+7)/8*8],0.0f,64);
  if(!flatIndicesReady){
   DetMakeStrides(indices.ReinterpretCast<int32_t>(),n*64,4);
    DetRemapFlatIndices(transposed,indices,n,flatIndicesReady);
  }
  DetFence<HardEvent::MTE2_V>();
  Gather(transposed,loaded,indices,0,n*64);PipeBarrier<PIPE_V>();
  DataCopy(mat,transposed,DataCopyParams{uint16_t(n),8,0,1});PipeBarrier<PIPE_V>();
 }
 template<uint32_t ColumnBegin,uint32_t ActiveWidth>
 __aicore__ inline void Eliminate(uint32_t first,uint32_t last,float& product){
  float cachedDiagonal=0.0f,cachedInverse=0.0f;
   for(uint32_t pairBase=first;pairBase<last;pairBase+=2){
    {const uint32_t p=pairBase;
    EliminatePairPivot<ColumnBegin,ActiveWidth,false>(p,product,cachedDiagonal,cachedInverse);
    }
    {const uint32_t p=pairBase+1;
    EliminatePairPivot<ColumnBegin,ActiveWidth,true>(p,product,cachedDiagonal,cachedInverse);
    }
   }
 }
 __aicore__ inline void Process(uint64_t batch){
  // The bound entry admits each owner once, batches<=cores.
  {
   Load(batch*n*n);
   float product=1.0f;
   Eliminate<0,64>(0,32,product);
   Eliminate<32,32>(32,n,product);
   *reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr())=product;DetFence<HardEvent::S_MTE3>();
   DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0}); // Final PIPE_ALL covers output completion.
  }
 }
private:
 LocalTensor<float> mat,loaded,transposed,factor,absolute,reduced,pivot,spread,result,paired,pairSpread;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 bool flatIndicesReady=false;
 static constexpr uint32_t pitch=72;uint32_t n,cols,cores;uint64_t batches;
};


#endif
