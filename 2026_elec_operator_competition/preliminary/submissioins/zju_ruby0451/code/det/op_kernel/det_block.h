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

#ifndef RUBY0451_DET_DET_BLOCK_H
#define RUBY0451_DET_DET_BLOCK_H
#include "det_common.h"
#include "det_block_pair.h"
#include "det_block_layout.h"
#include "det_block_cube.h"
class FastBlockVector {
 TPipe pipe;GlobalTensor<float>a,y,w;LocalTensor<float>mat,loaded,tmpProduct,pivot,spread,q,absolute,reduced,result;
 LocalTensor<int32_t>indices,indexTemp,bases,outputIndices,loadIndices;
#ifdef DET_FAST_VECTOR_FINISH
 LocalTensor<float>inverseDiag,inverseOff,pairDiagonal,pairOff;LocalTensor<int32_t>partnerDelta,pairIndices;LocalTensor<int32_t>permutationDelta;
#endif
 uint32_t n,pitch,slots,batches,lane,slot,within,cores,leftOffset,vOffset,tempOffset,currentBatch;float product;int32_t exponent;bool bad;
 __aicore__ inline float PairAbs(float x){return x<0?-x:x;}
 __aicore__ inline float PairMax(float x,float y){return x>y?x:y;}
 union Bits{float f;uint32_t u;};
 __aicore__ inline bool Finite(float x){Bits b;b.f=x;return(b.u&0x7f800000u)!=0x7f800000u;}
 __aicore__ inline void Multiply(float x){product*=x;}
 __aicore__ inline float Finish(){Bits b;if(bad){b.u=0x7fc00000u;return b.f;}return product;}
public:

 __aicore__ inline void FinishEvenTail(LocalTensor<float> matrix,uint32_t width) {
   auto paired=LocalTensor<float>(TPosition::VECCALC,24576,128);
   auto pairSpread=LocalTensor<float>(TPosition::VECCALC,25088,1024);
   auto pairBases=LocalTensor<int32_t>(TPosition::VECCALC,29184,128);
   for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ uint64_t*>(pairBases.GetPhyAddr())[i]=uint64_t(i*72*4)|(uint64_t(i*72*4+4)<<32);
   FastFence<HardEvent::S_V>();
   for(uint32_t filled=16;filled<128;filled*=2){Adds(pairBases[filled],pairBases,int32_t(filled/2*72*4),filled);PipeBarrier<PIPE_V>();}
   DetTailPairEliminate<0,64>(matrix,paired,pairSpread,pairBases,width,0,width<32?width:32,product);
   if(width>32)DetTailPairEliminate<32,32>(matrix,paired,pairSpread,pairBases,width,32,width,product);

}


 __aicore__ inline void UpdateOddTail(uint32_t col,uint32_t best,uint32_t remaining,bool predicted,float diagonal,LocalTensor<float> factor,LocalTensor<float> matrix,LocalTensor<float> pivotTail,LocalTensor<float> spreadTail,LocalTensor<int32_t> basesTail,uint64_t* activeTail,uint32_t& cachedRow,float& cachedDiagonal,float& cachedInverse) {
    float below=0.0f,cross=0.0f,future=0.0f;
    if((col&1)==0){
     uint64_t nextActive=activeTail[0]&~(uint64_t(1)<<best);
     uint64_t first=nextActive&(~nextActive+1);
     cachedRow=63-static_cast<uint32_t>(CountLeadingZero(first));
     below=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[cachedRow];
     cross=reinterpret_cast<__ubuf__ float*>(matrix[(col+1)*72].GetPhyAddr())[best];
     future=reinterpret_cast<__ubuf__ float*>(matrix[(col+1)*72].GetPhyAddr())[cachedRow];
    }
    // remaining <= 63; the gathered trailing columns and factor are disjoint.
    uint64_t liveMask[2]={(uint64_t(1)<<remaining)-1,0};
    AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(pivotTail.GetPhyAddr()),
      reinterpret_cast<__ubuf__ float*>(matrix[(col+1)*72].GetPhyAddr()),
      reinterpret_cast<__ubuf__ uint32_t*>(basesTail.GetPhyAddr()),4608,best*4,
      liveMask,uint8_t(1),uint16_t(8));
    Bits bits;bits.f=diagonal;float divisor=diagonal;
    if((bits.u&0x7f800000u)==0){
     Muls(factor,factor,16777216.0f,64);PipeBarrier<PIPE_V>();
     divisor*=16777216.0f;below*=16777216.0f;
    }
    float inverse=predicted?cachedInverse:-1.0f/divisor;
    SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
    auto* factorWords=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr());
    vmuls(factorWords,factorWords,inverse,uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));PipeBarrier<PIPE_V>();
    Brcb(spreadTail,pivotTail,(remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
    MulAddDst<float,float,false>(matrix[(col+1)*72],factor,spreadTail,uint64_t(64),remaining,
      BinaryRepeatParams{1,1,0,9,0,1});
    if((col&1)==0){
     float normalizedBelow=below*inverse;
     cachedDiagonal=normalizedBelow*cross+future;
     Bits nextBits;nextBits.f=cachedDiagonal;float nextDivisor=cachedDiagonal;
     if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
     cachedInverse=-1.0f/nextDivisor;
    }

}


 __aicore__ inline void StoreSolveChunk(uint32_t rhs,uint32_t live,bool singular,uint32_t width,uint32_t sbase,uint32_t stop,uint32_t first,uint32_t start) {
  if(rhs){
   const uint32_t packed=live*32;
   if(singular){Duplicate(tmpProduct,0.0f,packed);PipeBarrier<PIPE_V>();}
   else{
    FastFence<HardEvent::S_V>();
    // Both row permutation and row scale repeat for each RHS column.
    // pivot is dead after panel elimination; keep the repeated index source
    // separate from pairIndices, whose first repeat is overwritten by Add.
    auto rowDelta=pivot.ReinterpretCast<int32_t>();
    Gather(rowDelta,permutationDelta,indexTemp.ReinterpretCast<uint32_t>(),0,32);PipeBarrier<PIPE_V>();
    Add(pairIndices,rowDelta,outputIndices,uint64_t(32),uint8_t(live),
      BinaryRepeatParams{1,1,1,4,0,4});PipeBarrier<PIPE_V>();
    Gather(tmpProduct,mat[width*40],pairIndices.ReinterpretCast<uint32_t>(),0,packed);PipeBarrier<PIPE_V>();
    Gather(loaded,inverseDiag,loadIndices.ReinterpretCast<uint32_t>(),0,32);PipeBarrier<PIPE_V>();
    Mul(tmpProduct,tmpProduct,loaded,uint64_t(32),uint8_t(live),
      BinaryRepeatParams{1,1,1,4,4,0});PipeBarrier<PIPE_V>();
   }
   FastFence<HardEvent::V_MTE3>();DataCopyPad(w[sbase+(stop+first)*pitch+start],tmpProduct,DataCopyExtParams{uint16_t(live),128,0,(pitch-32)*4,0});FastFence<HardEvent::MTE3_V>();
  }

}


 __aicore__ inline void UpdateSolvePair(uint32_t col,uint32_t row0,uint32_t row1,uint32_t remaining,LocalTensor<float> col0,LocalTensor<float> col1,float a00,float a11,float rowScale,float weight00,float weight11,float weight01,float weight10) {
    auto q0=loaded,q1=loaded[32],pivot1=loaded[64],spread1=loaded[128];
    // remaining <= 62: one normal-mask gather; disjoint coefficient outputs.
    uint64_t gatherMask[2]={(uint64_t(1)<<remaining)-1,0};
    auto* trailingWords=reinterpret_cast<__ubuf__ float*>(mat[(col+2)*40].GetPhyAddr());
    auto* baseWords=reinterpret_cast<__ubuf__ uint32_t*>(bases.GetPhyAddr());
    AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(pivot.GetPhyAddr()),trailingWords,
      baseWords,2560,row0*4,gatherMask,uint8_t(1),uint16_t(8));
    AscendC::GatherImpl<float>(reinterpret_cast<__ubuf__ float*>(pivot1.GetPhyAddr()),trailingWords,
      baseWords,2560,row1*4,gatherMask,uint8_t(1),uint16_t(8));
    reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr())[row0]=a00-rowScale;reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr())[row1]=a11-rowScale;FastFence<HardEvent::S_V>();
    SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xffffffffu));
    vmuls(reinterpret_cast<__ubuf__ float*>(q0.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr()),-weight00,uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));vmuls(reinterpret_cast<__ubuf__ float*>(q1.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr()),-weight11,uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));PipeBarrier<PIPE_V>();
    AscendC::BrcbImpl<float>(reinterpret_cast<__ubuf__ float*>(spread.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(pivot.GetPhyAddr()),uint8_t((remaining+7)/8),BrcbRepeatParams{1,8});
    AscendC::BrcbImpl<float>(reinterpret_cast<__ubuf__ float*>(spread1.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(pivot1.GetPhyAddr()),uint8_t((remaining+7)/8),BrcbRepeatParams{1,8});
     SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xffffffffu));
    vaxpy(reinterpret_cast<__ubuf__ float*>(q0.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr()),-weight10,uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));vaxpy(reinterpret_cast<__ubuf__ float*>(q1.GetPhyAddr()),reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr()),-weight01,uint8_t(1),uint16_t(1),uint16_t(1),uint16_t(8),uint16_t(8));PipeBarrier<PIPE_V>();
    SetVectorMask<float,MaskMode::NORMAL>(0,uint64_t(0xffffffffu));
    MulAddDst<float,float,false>(mat[(col+2)*40],q0,spread,uint64_t(32),remaining,BinaryRepeatParams{1,1,0,5,0,1});PipeBarrier<PIPE_V>();
    MulAddDst<float,float,false>(mat[(col+2)*40],q1,spread1,uint64_t(32),remaining,BinaryRepeatParams{1,1,0,5,0,1});

}


 __aicore__ inline void LoadSolveChunk(uint32_t start,uint32_t stop,uint32_t first,uint32_t live,bool privateInput,uint32_t width,uint32_t sbase) {
  auto source=(start==0||privateInput)?a[uint64_t(currentBatch)*n*n]:w[sbase];uint32_t inputPitch=(start==0||privateInput)?n:pitch;
  // DMA overwrites all 32 live rows in every live panel/RHS column.
  // Keep the empty-wave behavior; padding and unused RHS columns have no live consumer.
  if(currentBatch>=batches){Duplicate(mat,0.0f,2560);if(privateInput)Duplicate(loaded,0.0f,2560);}
  FastFence<HardEvent::V_MTE2>();
  if(currentBatch<batches){
   DataCopyPad(mat,source[start*inputPitch+start],
     DataCopyExtParams{32,128,(inputPitch-32)*4,1,0},DataCopyPadExtParams<float>{false,0,0,0});
   if(live)DataCopyPad(mat[width*40],source[(stop+first)*inputPitch+start],
     DataCopyExtParams{uint16_t(live),128,(inputPitch-32)*4,1,0},DataCopyPadExtParams<float>{false,0,0,0});
  }
  if(privateInput){
   CrossCoreWaitFlag(10);
   uint32_t address=tempOffset+(lane/2)*8192;
   DataCopyPad(loaded,w[address],DataCopyExtParams{32,128,0,1,0},DataCopyPadExtParams<float>{false,0,0,0});
   DataCopyPad(loaded[width*40],w[address+1024+(lane%2)*DetOwnerRhsChunk(n,start)*32],
     DataCopyExtParams{uint16_t(live),128,0,1,0},DataCopyPadExtParams<float>{false,0,0,0});
  }
  FastFence<HardEvent::MTE2_V>();
  if(privateInput){Sub(mat,mat,loaded,2560);PipeBarrier<PIPE_V>();PrefetchOwnedRaw(start);}

}

 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,GM_ADDR W,uint32_t size,uint32_t count,uint32_t owners){
  n=size;pitch=(n+127)/128*128;slots=count<owners?count:owners;batches=count;lane=GetBlockIdx();slot=lane/8;within=lane%8;cores=owners*4;
  leftOffset=slots*pitch*pitch;vOffset=leftOffset+slots*pitch*32;tempOffset=vOffset+slots*32*pitch;
  a.SetGlobalBuffer((__gm__ float*)A);y.SetGlobalBuffer((__gm__ float*)Y);w.SetGlobalBuffer((__gm__ float*)W);SetMaskNorm();SetAtomicNone();
  mat=LocalTensor<float>(TPosition::VECCALC,65536,2560);loaded=LocalTensor<float>(TPosition::VECCALC,8192,4096);
  tmpProduct=LocalTensor<float>(TPosition::VECCALC,24576,4096);indices=LocalTensor<int32_t>(TPosition::VECCALC,40960,1024);
  indexTemp=LocalTensor<int32_t>(TPosition::VECCALC,45056,1024);pivot=LocalTensor<float>(TPosition::VECCALC,49152,64);
  spread=LocalTensor<float>(TPosition::VECCALC,49408,512);bases=LocalTensor<int32_t>(TPosition::VECCALC,51456,64);
  absolute=LocalTensor<float>(TPosition::VECCALC,51840,32);reduced=LocalTensor<float>(TPosition::VECCALC,51968,8);result=LocalTensor<float>(TPosition::VECCALC,52000,8);
  inverseDiag=LocalTensor<float>(TPosition::VECCALC,52032,32);permutationDelta=LocalTensor<int32_t>(TPosition::VECCALC,52160,32);
  outputIndices=LocalTensor<int32_t>(TPosition::VECCALC,56320,1024);pairIndices=LocalTensor<int32_t>(TPosition::VECCALC,81920,1024);
  loadIndices=LocalTensor<int32_t>(TPosition::VECCALC,86016,1024);
  CreateVecIndex(indices,int32_t(0),1024);PipeBarrier<PIPE_V>();
  ShiftRight(indexTemp,indices,int32_t(5),1024);PipeBarrier<PIPE_V>();
  ShiftLeft(loaded.ReinterpretCast<int32_t>(),indexTemp,int32_t(5),1024);PipeBarrier<PIPE_V>();
  Sub(indices,indices,loaded.ReinterpretCast<int32_t>(),1024);PipeBarrier<PIPE_V>();
  Muls(outputIndices,indexTemp,int32_t(40),1024);PipeBarrier<PIPE_V>();
  Add(outputIndices,outputIndices,indices,1024);PipeBarrier<PIPE_V>();
  ShiftLeft(outputIndices,outputIndices,int32_t(2),1024);
  ShiftLeft(indexTemp,indices,int32_t(2),1024);
  ShiftRight(loadIndices,indices,int32_t(1),1024);PipeBarrier<PIPE_V>();
  ShiftLeft(loadIndices,loadIndices,int32_t(2),1024);PipeBarrier<PIPE_V>();
  CreateVecIndex(bases,int32_t(0),64);PipeBarrier<PIPE_V>();Muls(bases,bases,int32_t(160),64);PipeBarrier<PIPE_V>();
 }

 __aicore__ inline void Initialize(uint32_t wave){
  product=1;exponent=0;bad=false;currentBatch=wave*slots+slot;
 }
 __aicore__ inline void PrefetchOwnedRaw(uint32_t start){
  uint32_t rhs=n-start-32,chunk=DetOwnerRhsChunk(n,start),first=(within/2)*(2*chunk);
  if(slot>=slots||first>=rhs)return;
  uint32_t live=rhs-first<2*chunk?rhs-first:2*chunk,width=(live+15)/16*16;
  uint32_t row=start+(lane%2)*16,col=start+32+first;
  auto rawBase=LocalTensor<float>(TPosition::VECCALC,98304,1024);
  // Previous phase SyncAll covers the previous rawBase reader. This phase has
  // no vector writer; only live words are consumed or published.
  if(currentBatch<batches){
   DataCopyPad(rawBase,a[uint64_t(currentBatch)*n*n+row*n+col],
    DataCopyExtParams{16,live*4,(n-live)*4,(width-(live+7)/8*8)/8,0},
    DataCopyPadExtParams<float>{false,0,0,0});
  }else{Duplicate(rawBase,0.0f,16*width);PipeBarrier<PIPE_V>();}
 }
 __aicore__ inline void SolveChunk(uint32_t start,uint32_t stop,uint32_t first,uint32_t live,bool privateInput=false){
  uint32_t width=stop-start,rhs=n-stop,sbase=slot*pitch*pitch;bool singular=false;
    LoadSolveChunk(start,stop,first,live,privateInput,width,sbase);
  uint64_t active[2]={(uint64_t(1)<<width)-1,0};
  for(uint32_t col=0;col<width;col+=2){
   FastFence<HardEvent::V_S>();
   uint64_t bit0=active[0]&(~active[0]+1),rest=active[0]^bit0,bit1=rest&(~rest+1);
   uint32_t row0=63-static_cast<uint32_t>(CountLeadingZero(bit0));
   uint32_t row1=63-static_cast<uint32_t>(CountLeadingZero(bit1));
   auto col0=mat[col*40],col1=mat[(col+1)*40];
   float a00=reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr())[row0],a10=reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr())[row1],a01=reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr())[row0],a11=reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr())[row1];
   float determinant=a00*a11-a01*a10;
   float maxEntry=PairMax(PairMax(PairAbs(a00),PairAbs(a10)),PairMax(PairAbs(a01),PairAbs(a11)));
   if(!(PairAbs(determinant)>=0.25f*maxEntry)||!Finite(determinant)){
    Abs(absolute,col0,32);PipeBarrier<PIPE_V>();
    WholeReduceMax(reduced,absolute,active,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);FastFence<HardEvent::V_S>();
    row0=reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];a00=reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr())[row0];a01=reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr())[row0];
    Muls(loaded,col1,a00,32);Muls(loaded[32],col0,a01,32);PipeBarrier<PIPE_V>();
    Sub(absolute,loaded,loaded[32],32);PipeBarrier<PIPE_V>();Abs(absolute,absolute,32);PipeBarrier<PIPE_V>();
    uint64_t secondMask[2]={active[0]&~(uint64_t(1)<<row0),0};
    WholeReduceMax(reduced,absolute,secondMask,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);FastFence<HardEvent::V_S>();
    row1=reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];a10=reinterpret_cast<__ubuf__ float*>(col0.GetPhyAddr())[row1];a11=reinterpret_cast<__ubuf__ float*>(col1.GetPhyAddr())[row1];
    determinant=a00*a11-a01*a10;
    maxEntry=PairMax(PairMax(PairAbs(a00),PairAbs(a10)),PairMax(PairAbs(a01),PairAbs(a11)));
   }
   if(determinant==0||!Finite(determinant)){singular=true;break;}
   Bits scaleWord,inverseScaleWord;scaleWord.f=maxEntry;scaleWord.u&=0x7f800000u;
   inverseScaleWord.u=0x7f000000u-scaleWord.u;
   float rowScale=scaleWord.f,inverseScale=inverseScaleWord.f;
   float inverse=1.0f/determinant;
   float weight00=a11*inverse,weight11=a00*inverse,weight01=-a01*inverse,weight10=-a10*inverse;
   uint32_t remaining=width+live-col-2;
   if(remaining){
    UpdateSolvePair(col,row0,row1,remaining,col0,col1,a00,a11,rowScale,weight00,weight11,weight01,weight10);
   }
   reinterpret_cast<__ubuf__ float*>(inverseDiag.GetPhyAddr())[col/2]=inverseScale;
   reinterpret_cast<__ubuf__ int32_t*>(permutationDelta.GetPhyAddr())[col]=int32_t(row0-col)*4;reinterpret_cast<__ubuf__ int32_t*>(permutationDelta.GetPhyAddr())[col+1]=int32_t(row1-col-1)*4;
   if(first==0&&FastParity(active[0]&((uint64_t(1)<<row0)-1)))product=-product;
   active[0]&=~(uint64_t(1)<<row0);
   if(first==0&&FastParity(active[0]&((uint64_t(1)<<row1)-1)))product=-product;
   active[0]&=~(uint64_t(1)<<row1);
   if(first==0)Multiply(determinant);
   PipeBarrier<PIPE_V>();
  }
  if(first==0&&singular)bad=true;
    StoreSolveChunk(rhs,live,singular,width,sbase,stop,first,start);
 }

 __aicore__ inline void Prepare(uint32_t start,uint32_t stop){
  if(slot>=slots)return;
  uint32_t rhs=n-stop,chunk=DetOwnerRhsChunk(n,start);
  for(uint32_t first=within*chunk;first<rhs;first+=8*chunk)
   SolveChunk(start,stop,first,rhs-first<chunk?rhs-first:chunk);
  if(start)return;
  for(uint32_t first=within*chunk;first<rhs;first+=8*chunk){
   uint32_t live=rhs-first<chunk?rhs-first:chunk;
   Duplicate(loaded,0.0f,1024);FastFence<HardEvent::V_MTE2>();
   if(currentBatch<batches){
    DataCopyPad(loaded,a[uint64_t(currentBatch)*n*n+stop+first],
      DataCopyExtParams{32,live*4,(n-live)*4,(32-(live+7)/8*8)/8,0},DataCopyPadExtParams<float>{false,0,0,0});
    FastFence<HardEvent::MTE2_V>();
   }
   FastFence<HardEvent::V_MTE3>();
   DataCopyPad(w[slot*pitch*pitch+stop+first],loaded,
     DataCopyExtParams{32,live*4,(32-(live+7)/8*8)/8,(pitch-live)*4,0});
   FastFence<HardEvent::MTE3_V>();
  }
 }

 __aicore__ inline void PrepareOwned(uint32_t start){
  uint32_t rhs=n-start-32,chunk=DetOwnerRhsChunk(n,start),first=(within/2)*(2*chunk);
  if(slot>=slots)return;
  CrossCoreSetFlag<2,PIPE_MTE3>(11);
  if(first>=rhs)return;
  uint32_t mine=within*chunk;
  if(mine<rhs)SolveChunk(start,start+32,mine,rhs-mine<chunk?rhs-mine:chunk,true);
  else {CrossCoreWaitFlag(10);PrefetchOwnedRaw(start);}
 }
 __aicore__ inline void ConsumeOwnedRaw(uint32_t start){
  uint32_t rhs=n-start-32,chunk=DetOwnerRhsChunk(n,start),first=(within/2)*(2*chunk);
  if(slot>=slots||first>=rhs)return;
  uint32_t live=rhs-first<2*chunk?rhs-first:2*chunk,width=(live+15)/16*16;
  uint32_t core=lane/2,sub=lane%2,row=start+sub*16,col=start+32+first;
  auto rawBase=LocalTensor<float>(TPosition::VECCALC,98304,1024);
  // Independent of SolveChunk packing and its MTE3 reader.
  auto rawProduct=LocalTensor<float>(TPosition::VECCALC,102400,1024);
  CrossCoreWaitFlag(12);
  DataCopy(rawProduct,w[tempOffset+core*8192+4096+sub*16*width],16*width);
  FastFence<HardEvent::MTE2_V>();
  Sub(loaded,rawBase,rawProduct,16*width);PipeBarrier<PIPE_V>();FastFence<HardEvent::V_MTE3>();
  DataCopyPad(w[slot*pitch*pitch+row*pitch+col],loaded,
    DataCopyExtParams{16,live*4,(width-(live+7)/8*8)/8,(pitch-live)*4,0});
  FastFence<HardEvent::MTE3_V>();
 }

 __aicore__ inline void Rebuild(uint32_t start){
  uint32_t core=lane/2,sub=lane%2,jobs=slots*DetLeftJobs(n,start,slots,cores);
  if(core<jobs)CrossCoreSetFlag<2,PIPE_MTE3>(11);
  for(uint32_t job=core;job<jobs;job+=cores){
   DetLeftJob tile=DetLeftTile(n,start,job,slots,cores);
   uint32_t row=tile.col+sub*64,sourceBatch=currentBatch-slot+tile.batch;
   uint32_t rows=(sub*64<tile.n&&row<n)?(n-row<64?n-row:64):0;
   if(rows>tile.n-sub*64)rows=tile.n-sub*64;
   uint32_t cols=n-tile.row<tile.m?n-tile.row:tile.m;
   uint32_t chunkRows=tile.n>sub*64?tile.n-sub*64:0;if(chunkRows>64)chunkRows=64;
   if(rows){
    Duplicate(loaded,0.0f,chunkRows*tile.m);FastFence<HardEvent::V_MTE2>();
    if(sourceBatch<batches){
     DataCopyPad(loaded,a[uint64_t(sourceBatch)*n*n+row*n+tile.row],
       DataCopyExtParams{uint16_t(rows),cols*4,(n-cols)*4,(tile.m-(cols+7)/8*8)/8,0},
       DataCopyPadExtParams<float>{false,0,0,0});
     FastFence<HardEvent::MTE2_V>();
    }
   }
   FastFence<HardEvent::V_MTE3>();
   if(job!=core)CrossCoreSetFlag<2,PIPE_MTE3>(11);
   CrossCoreWaitFlag(10);
   if(rows){
    FastFence<HardEvent::V_MTE2>();
    DataCopy(tmpProduct,w[tempOffset+core*8192+sub*64*tile.m],chunkRows*tile.m);
    FastFence<HardEvent::MTE2_V>();
    Sub(loaded,loaded,tmpProduct,chunkRows*tile.m);PipeBarrier<PIPE_V>();
    FastFence<HardEvent::V_MTE3>();
    DataCopyPad(w[tile.batch*pitch*pitch+row*pitch+tile.row],loaded,
      DataCopyExtParams{uint16_t(rows),cols*4,(tile.m-(cols+7)/8*8)/8,(pitch-cols)*4,0});
    FastFence<HardEvent::MTE3_V>();
   }
  }
 }
 __aicore__ inline void FinishTail(uint32_t start){
  if(within||slot>=slots)return;
  uint32_t width=n-start;
  auto matrix=LocalTensor<float>(TPosition::VECCALC,0,4608);
  auto absoluteTail=LocalTensor<float>(TPosition::VECCALC,18432,64);
  auto pivotTail=LocalTensor<float>(TPosition::VECCALC,18688,64);
  auto spreadTail=LocalTensor<float>(TPosition::VECCALC,18944,512);
  auto basesTail=LocalTensor<int32_t>(TPosition::VECCALC,20992,64);
  FastFence<HardEvent::S_V>();Duplicate(matrix,0.0f,4608);FastFence<HardEvent::V_MTE2>();
  DataCopyPad(matrix,w[slot*pitch*pitch+start*pitch+start],
   DataCopyExtParams{uint16_t(width),width*4,(pitch-width)*4,(72-(width+7)/8*8)/8,0},
   DataCopyPadExtParams<float>{true,0,uint8_t((width+7)/8*8-width),0.0f});
  FastFence<HardEvent::MTE2_V>();
  if((width&1)==0){
    FinishEvenTail(matrix,width);
   return;
  }
  for(uint32_t i=0;i<8;++i)reinterpret_cast<__ubuf__ int32_t*>(basesTail.GetPhyAddr())[i]=int32_t(i*72*4);
  FastFence<HardEvent::S_V>();
  for(uint32_t filled=8;filled<64;filled*=2){
   Adds(basesTail[filled],basesTail,int32_t(filled*72*4),filled);PipeBarrier<PIPE_V>();
  }
  uint64_t activeTail[2]={width==64?~uint64_t(0):(uint64_t(1)<<width)-1,0};
  float cachedDiagonal=0.0f,cachedInverse=0.0f;uint32_t cachedRow=0;
  for(uint32_t col=0;col<width;++col){
   auto factor=matrix[col*72];
   uint32_t best;float diagonal;bool predicted=(col&1)!=0;
   if(predicted){best=cachedRow;diagonal=cachedDiagonal;}
   else{
    FastFence<HardEvent::V_S>();
    uint64_t first=activeTail[0]&(~activeTail[0]+1);
    best=63-static_cast<uint32_t>(CountLeadingZero(first));diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[best];
   }
   if(!(diagonal>=1.0f||diagonal<=-1.0f)||!Finite(diagonal)){
    predicted=false;
    Abs(absoluteTail,factor,64);PipeBarrier<PIPE_V>();
    WholeReduceMax(reduced,absoluteTail,activeTail,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
    FastFence<HardEvent::V_S>();
    float largest=reinterpret_cast<__ubuf__ float*>(reduced.GetPhyAddr())[0];
    if(largest==0){product=0;return;}if(!Finite(largest)){bad=true;return;}
    best=reinterpret_cast<__ubuf__ uint32_t*>(reduced.GetPhyAddr())[1];diagonal=reinterpret_cast<__ubuf__ float*>(factor.GetPhyAddr())[best];
   }
   uint32_t remaining=width-col-1;
   if(remaining){
    UpdateOddTail(col,best,remaining,predicted,diagonal,factor,matrix,pivotTail,spreadTail,basesTail,activeTail,cachedRow,cachedDiagonal,cachedInverse);
   }
   if(FastParity(activeTail[0]&((uint64_t(1)<<best)-1)))product=-product;
   Multiply(diagonal);activeTail[0]&=~(uint64_t(1)<<best);PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void Store(uint32_t wave){
  uint32_t batch=wave*slots+slot;if(within||batch>=batches)return;FastFence<HardEvent::V_S>();reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr())[0]=Finish();FastFence<HardEvent::S_MTE3>();
  DataCopyPad(y[batch],result,DataCopyExtParams{1,4,0,0,0});FastFence<HardEvent::MTE3_V>();
 }
};
__aicore__ inline void FastBlockProcess(GM_ADDR a,GM_ADDR y,GM_ADDR w,uint32_t n,uint32_t batches,uint32_t slots){
 InitSocState();uint32_t activeSlots=batches<slots?batches:slots;
 if ASCEND_IS_AIV {
  FastBlockVector vector;vector.Init(a,y,w,n,batches,slots);
  for(uint32_t wave=0;wave<(batches+activeSlots-1)/activeSlots;++wave){
   vector.Initialize(wave);
   for(uint32_t start=0;start<n;start+=32){
    if(start&&n<=320&&n-start>64){
     vector.PrepareOwned(start);vector.ConsumeOwnedRaw(start);SyncAll();continue;
    }
    if(start){vector.Rebuild(start);SyncAll();}
    if(n-start<=64){vector.FinishTail(start);break;}
    vector.Prepare(start,start+32);SyncAll();
   }
   vector.Store(wave);
   // Later waves reuse shared factors/scratch; after the last output no consumer remains.
   // Store completes its own DMA and each TPipe destructor drains its local engines.
   if(wave+1<(batches+activeSlots-1)/activeSlots)SyncAll();
  }
 }
 if ASCEND_IS_AIC {
  DetLeftCube cube;cube.Init();
  for(uint32_t wave=0;wave<(batches+activeSlots-1)/activeSlots;++wave)
   for(uint32_t start=32;start<n;start+=32){
    if(n<=320&&n-start>64)cube.RebuildOwned(w,n,activeSlots,start);else cube.Rebuild(w,n,activeSlots,slots*4,start);if(n-start<=64)break;
   }
 }
}


#endif
