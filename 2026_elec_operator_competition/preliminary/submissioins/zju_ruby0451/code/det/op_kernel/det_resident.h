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

#ifndef RUBY0451_DET_DET_RESIDENT_H
#define RUBY0451_DET_DET_RESIDENT_H
#include "det_common.h"
class KernelColumnDet {
public:

 __aicore__ inline void UpdateTrailingColumns(uint32_t p,uint32_t best,uint32_t remaining,float diagonal) {
    Gather(pivot,mat[(p+1)*pitch],bases.ReinterpretCast<uint32_t>(),best*4,remaining);
    factor.SetValue(best,0.0f);DetFence<HardEvent::S_V>();
    Bits normalization;normalization.f=diagonal;
    float divisor=diagonal;
    if((normalization.u&0x7f800000u)==0u){
     Muls(factor,factor,16777216.0f,pitch);PipeBarrier<PIPE_V>();
     divisor*=16777216.0f;
    }
    Muls(factor,factor,-1.0f/divisor,pitch);PipeBarrier<PIPE_V>();
    Brcb(spread,pivot,(remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
    for(uint32_t r=0;r<n;r+=64){
     uint32_t count=n-r<64?n-r:64;
     MulAddDst(mat[(p+1)*pitch+r],factor[r],spread,count,remaining,
       BinaryRepeatParams{1,1,0,static_cast<uint8_t>(pitch/8),0,1});
    }

}

 __aicore__ inline ~KernelColumnDet(){PipeBarrier<PIPE_ALL>();}


 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;pitch=n<=8?8:n<=16?16:n<=32?32:n<=64?72:(n+7)/8*8;cols=(n+7)/8*8;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  uint32_t ioCount=n<=64?cols*pitch:8*pitch,offset=0;
  DetInitColumnBuffers(cols,pitch,ioCount,offset,mat,loaded,transposed,indices,absolute,reduced,pivot,spread);
  bases=LocalTensor<int32_t>(TPosition::VECCALC,offset,cols);offset+=cols*4;
  result=LocalTensor<float>(TPosition::VECCALC,offset,8);
  DetMakeStrides1(bases,cols,pitch*4);
  auto offsets=indices.ReinterpretCast<int32_t>();
  uint32_t size=n<=64?cols*pitch:n*8,rows=n<=64?pitch:8,stride=n<=64?n:pitch;
  DetMakeStrides1(offsets,rows,stride*4);
  for(uint32_t filled=rows;filled<size;filled*=2){
   uint32_t count=size-filled<filled?size-filled:filled;
   Adds(offsets[filled],offsets,int32_t(filled/rows*4),count);PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void Load(uint64_t offset){
  if(n<=64){
   DetFence<HardEvent::S_V>();Duplicate(loaded,0.0f,cols*pitch);DetFence<HardEvent::V_MTE2>();
   DataCopyPad(loaded,input[offset],DataCopyExtParams{1,n*n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
   DetFence<HardEvent::MTE2_V>();Gather(mat,loaded,indices,0,cols*pitch);PipeBarrier<PIPE_V>();return;
  }
  DetFence<HardEvent::S_V>();Duplicate(mat,0.0f,cols*pitch);PipeBarrier<PIPE_V>();
  for(uint32_t base=0;base<n;base+=8){
   uint32_t rows=n-base<8?n-base:8;
   Duplicate(loaded,0.0f,8*pitch);DetFence<HardEvent::V_MTE2>();
   DataCopyExtParams cp{static_cast<uint16_t>(rows),n*4,0,0,0};DataCopyPadExtParams<float> pad{false,0,0,0};
   DataCopyPad(loaded,input[offset+uint64_t(base)*n],cp,pad);DetFence<HardEvent::MTE2_V>();
   Gather(transposed,loaded,indices,0,n*8);PipeBarrier<PIPE_V>();
   DataCopyParams dest{static_cast<uint16_t>(n),1,0,static_cast<uint16_t>((pitch-8)/8)};
   DataCopy(mat[base],transposed,dest);PipeBarrier<PIPE_V>();
  }
 }
 union Bits {float f;uint32_t u;};



 __aicore__ inline void Process(){
  if(GetBlockIdx()>=cores)return;
  for(uint64_t batch=GetBlockIdx();batch<batches;batch+=cores){
   Load(batch*n*n);
   uint64_t active[2]={n>=64?~uint64_t(0):(uint64_t(1)<<n)-1,n<=64?0:(n==128?~uint64_t(0):(uint64_t(1)<<(n-64))-1)};
   float product=1.0f;int32_t exponent=0;bool singular=false;
   for(uint32_t p=0;p<n;++p){
    factor=mat[p*pitch];
    Abs(absolute,factor,pitch);PipeBarrier<PIPE_V>();
    uint32_t best=0;float largest=0.0f;
    for(uint32_t block=0;block<(n+63)/64;++block){
     if(!active[block])continue;
     uint64_t mask[2]={active[block],0};
     WholeReduceMax(reduced,absolute[block*64],mask,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);DetFence<HardEvent::V_S>();
     float value=reduced.GetValue(0);
     if(value>largest){largest=value;best=block*64+reduced.ReinterpretCast<uint32_t>().GetValue(1);}
    }
    if(largest==0.0f){singular=true;break;}
    float diagonal=factor.GetValue(best);
    uint32_t remaining=n-p-1;
    if(remaining){
    UpdateTrailingColumns(p,best,remaining,diagonal);
    }
    uint32_t block=best/64,bit=best%64;
    uint32_t odd=DetParity1(active[block]&((uint64_t(1)<<bit)-1));
    if(block)odd^=DetParity1(active[0]);
    if(odd)product=-product;
    active[block]&=~(uint64_t(1)<<bit);
    DetMultiply1(diagonal,product,exponent);
    PipeBarrier<PIPE_V>();
   }
   DetFence<HardEvent::V_S>();result.SetValue(0,singular?0.0f:DetFinish2(product,exponent));DetFence<HardEvent::S_MTE3>();
   DataCopyExtParams cp{1,4,0,0,0};DataCopyPad(output[batch],result,cp);DetFence<HardEvent::MTE3_S>();
  }
 }
private:
 LocalTensor<float> mat,loaded,transposed,factor,absolute,reduced,pivot,spread,result;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 uint32_t n,pitch,cols,cores;uint64_t batches;
};

class KernelMediumDet {
public:

 __aicore__ inline void UpdateTrailingColumns(uint32_t p,uint32_t remaining,float diagonal,float below,float cross,float future,float& cachedDiagonal,float& cachedInverse) {
     if((p&1)==0)Gather(paired,mat[(p+1)*pitch],bases.ReinterpretCast<uint32_t>(),p*4,2*remaining);
     Bits normalization;normalization.f=diagonal;
     float divisor=diagonal;
     if((normalization.u&0x7f800000u)==0u){
      Muls(factor,factor,16777216.0f,uint64_t(64),1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
      divisor*=16777216.0f;below*=16777216.0f;
     }
     float inverse=(p&1)?cachedInverse:-1.0f/divisor;
     Muls(factor,factor,inverse,uint64_t(64),1,UnaryRepeatParams{1,1,8,8});PipeBarrier<PIPE_V>();
     if((p&1)==0){Brcb(pairSpread,paired,(2*remaining+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();}
     MulAddDst<float,float,false>(mat[(p+1)*pitch],factor,pairSpread[(p&1)?24:0],uint64_t(64),remaining,
       BinaryRepeatParams{1,1,0,9,0,2});
     if((p&1)==0){
      float normalizedBelow=below*inverse;
      cachedDiagonal=normalizedBelow*cross+future;
      Bits nextBits;nextBits.f=cachedDiagonal;
      float nextDivisor=cachedDiagonal;
      if((nextBits.u&0x7f800000u)==0u)nextDivisor*=16777216.0f;
      cachedInverse=-1.0f/nextDivisor;
      Axpy(pairSpread[8],pairSpread,normalizedBelow,uint64_t(64),(remaining+7)/8,UnaryRepeatParams{2,2,16,16});
     }

}

 __aicore__ inline ~KernelMediumDet(){PipeBarrier<PIPE_ALL>();}


 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,const LinalgDetTilingData& t){
  n=t.n;cols=(n+7)/8*8;batches=t.batches;cores=t.cores;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  SetMaskNorm();SetAtomicNone();
  uint32_t ioCount=n<=64?cols*pitch:8*pitch,offset=0;
  DetInitColumnBuffers(cols,pitch,ioCount,offset,mat,loaded,transposed,indices,absolute,reduced,pivot,spread);
    DetInitPairBuffers(offset,bases,paired,pairSpread,result);
  for(uint32_t i=0;i<8;++i)bases.ReinterpretCast<uint64_t>().SetValue(i,uint64_t(i*pitch*4)|(uint64_t(i*pitch*4+4)<<32));
  DetFence<HardEvent::S_V>();
  for(uint32_t filled=16;filled<128;filled*=2){Adds(bases[filled],bases,int32_t(filled/2*pitch*4),filled);PipeBarrier<PIPE_V>();}

 }
 __aicore__ inline void Load(uint64_t offset){
  if((n&7)==0){
   // Live row words are contiguous in GM; inactive row padding is never consumed.
   DetFence<HardEvent::V_MTE2>();
   DataCopy(loaded,input[offset],n*n);
   DetFence<HardEvent::MTE2_V>();
   DataCopy(mat,loaded,DataCopyParams{uint16_t(n),uint16_t(n/8),0,uint16_t((pitch-n)/8)});
   PipeBarrier<PIPE_V>();
   return;
  }
  // The large contiguous transfer runs while the shape-only address map is built.
  DetFence<HardEvent::V_MTE2>();
  DataCopyPad(loaded,input[offset],DataCopyExtParams{1,n*n*4,0,0,0},
    DataCopyPadExtParams<float>{true,0,uint8_t((8-n*n%8)%8),0.0f});
  Duplicate(loaded[(n*n+7)/8*8],0.0f,64);
  if(!flatIndicesReady){
   DetMakeStrides1(indices.ReinterpretCast<int32_t>(),n*64,4);
    DetRemapFlatIndices(transposed,indices,n,flatIndicesReady);
  }
  DetFence<HardEvent::MTE2_V>();
  Gather(transposed,loaded,indices,0,n*64);PipeBarrier<PIPE_V>();
  DataCopy(mat,transposed,DataCopyParams{uint16_t(n),8,0,1});PipeBarrier<PIPE_V>();
 }
 union Bits {float f;uint32_t u;};



 __aicore__ inline void Process(){
  if(GetBlockIdx()>=cores)return;
  for(uint64_t batch=GetBlockIdx();batch<batches;batch+=cores){
   Load(batch*n*n);
   float product=1.0f,cachedDiagonal=0.0f,cachedInverse=0.0f;bool singular=false;
   for(uint32_t p=0;p<n;++p){
    factor=mat[p*pitch];
    float diagonal,below=0.0f,cross=0.0f,future=0.0f;
    uint32_t remaining=n-p-1;
    if((p&1)==0){
     DetFence<HardEvent::V_S>();diagonal=factor.GetValue(p);
     if(remaining){
      below=factor.GetValue(p+1);
      cross=mat[(p+1)*pitch].GetValue(p);
      future=mat[(p+1)*pitch].GetValue(p+1);
     }
    }else diagonal=cachedDiagonal;
    if(diagonal==0.0f){singular=true;break;}
    if(remaining){
    UpdateTrailingColumns(p,remaining,diagonal,below,cross,future,cachedDiagonal,cachedInverse);
    }
    product*=diagonal;PipeBarrier<PIPE_V>();
   }
   DetFence<HardEvent::V_S>();result.SetValue(0,singular?0.0f:product);DetFence<HardEvent::S_MTE3>();
   DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0});DetFence<HardEvent::MTE3_S>();
  }
 }
private:
 LocalTensor<float> mat,loaded,transposed,factor,absolute,reduced,pivot,spread,result,paired,pairSpread;
 LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;GlobalTensor<float> input,output;
 bool flatIndicesReady=false;
 static constexpr uint32_t pitch=72;uint32_t n,cols,cores;uint64_t batches;
};

// Distributed resident columns; one packet publishes four consecutive pivots.
class KernelCooperativeDet {
public:

 __aicore__ inline void ConsumePanel(bool owner,uint32_t slot,uint32_t steps,uint32_t panel,uint32_t stop,bool& singular,uint64_t* remainingRows,float& product,int32_t& exponent) {
     if(!owner){DataCopy(packet,work[slot],steps*(pitch+8));DetFence<HardEvent::MTE2_V>();DetFence<HardEvent::MTE2_S>();}
     DetFence<HardEvent::V_S>();
     for(uint32_t p=panel;p<stop;++p){
      auto factor=packet[(p-panel)*(pitch+8)];
      if(factor.GetValue(pitch+2)==0.0f){singular=true;break;}
      uint32_t row=factor.ReinterpretCast<uint32_t>().GetValue(pitch+1),block=row/64,bit=row%64;
      if(lane==sink){
      uint32_t odd=DetParity1(remainingRows[block]&((uint64_t(1)<<bit)-1));for(uint32_t b=0;b<block;++b)odd^=DetParity1(remainingRows[b]);
      if(odd)product=-product;remainingRows[block]&=~(uint64_t(1)<<bit);DetMultiply1(factor.GetValue(pitch),product,exponent);
      }
      active.SetValue(row,0.0f);
      uint32_t first=stop>begin?stop:begin;if(first<end)Update(mat[(first-begin)*pitch],end-first,factor,row);
     }

}


 __aicore__ inline void PublishOwnedPanel(uint32_t panel,uint32_t stop,uint32_t slot,uint32_t steps) {
     DetFence<HardEvent::S_V>();Duplicate(packet,0.0f,8*(pitch+8));PipeBarrier<PIPE_V>();
     for(uint32_t p=panel;p<stop;++p){
      auto factor=packet[(p-panel)*(pitch+8)];
      DetFence<HardEvent::S_V>();Mul(factor,mat[(p-begin)*pitch],active,pitch);PipeBarrier<PIPE_V>();
      float largest;uint32_t row=Pivot(factor,largest);if(largest==0.0f)break;
      float diagonal=factor.GetValue(row);factor.SetValue(pitch,diagonal);factor.ReinterpretCast<uint32_t>().SetValue(pitch+1,row);factor.SetValue(pitch+2,1.0f);
      active.SetValue(row,0.0f);factor.SetValue(row,0.0f);DetFence<HardEvent::S_V>();
      NormalizeOwner(factor,mat[(p+1-begin)*pitch],stop-p-1,row,diagonal);
     }
     DetFence<HardEvent::V_MTE3>();DataCopy(work[slot],packet,steps*(pitch+8));

}


 __aicore__ inline void Init(GM_ADDR A,GM_ADDR Y,GM_ADDR W,const LinalgDetTilingData& t){
  n=t.n;pitch=(n+7)/8*8;cores=t.cores;batches=t.batches;workers=n<=256&&t.resident==4?4:8;
  lanes=workers+(workers==4?1:0);sink=workers==4?workers:0;lane=GetBlockIdx()%lanes;group=GetBlockIdx()/lanes;groups=cores/lanes;
  columns=(n+workers*8-1)/(workers*8)*8;begin=lane*columns;end=begin+columns<n?begin+columns:n;
  count=begin<n?end-begin:0;
  input.SetGlobalBuffer((__gm__ float*)A);output.SetGlobalBuffer((__gm__ float*)Y);
  work.SetGlobalBuffer((__gm__ float*)W+uint64_t(group)*16*(pitch+8));
  uint32_t offset=0;
  mat=LocalTensor<float>(TPosition::VECCALC,offset,columns*pitch);offset+=columns*pitch*4;
  loaded=LocalTensor<float>(TPosition::VECCALC,offset,8*pitch);offset+=8*pitch*4;
  transposed=LocalTensor<float>(TPosition::VECCALC,offset,8*columns);offset+=8*columns*4;
  indices=LocalTensor<uint32_t>(TPosition::VECCALC,offset,8*columns);offset+=8*columns*4;
  packet=LocalTensor<float>(TPosition::VECCALC,offset,8*(pitch+8));offset+=8*(pitch+8)*4;
  absolute=LocalTensor<float>(TPosition::VECCALC,offset,pitch);offset+=pitch*4;
  active=LocalTensor<float>(TPosition::VECCALC,offset,pitch);offset+=pitch*4;
  reduced=LocalTensor<float>(TPosition::VECCALC,offset,64);offset+=256;
  pivot=LocalTensor<float>(TPosition::VECCALC,offset,columns);offset+=columns*4;
  spread=LocalTensor<float>(TPosition::VECCALC,offset,columns*8);offset+=columns*32;
  bases=LocalTensor<int32_t>(TPosition::VECCALC,offset,columns);offset+=columns*4;
  result=LocalTensor<float>(TPosition::VECCALC,offset,8);
  for(uint32_t i=0;i<8;++i)bases.SetValue(i,int32_t(i*pitch*4));
  DetFence<HardEvent::S_V>();
  for(uint32_t filled=8;filled<columns;filled*=2){
   uint32_t len=columns-filled<filled?columns-filled:filled;
   Adds(bases[filled],bases,int32_t(filled*pitch*4),len);PipeBarrier<PIPE_V>();
  }
  if(!count)return;
  auto off=indices.ReinterpretCast<int32_t>();
  for(uint32_t i=0;i<8;++i)off.SetValue(i,int32_t((begin+i*pitch)*4));
  DetFence<HardEvent::S_V>();
  for(uint32_t filled=8;filled<count*8;filled*=2){
   uint32_t len=count*8-filled<filled?count*8-filled:filled;
   Adds(off[filled],off,int32_t(filled/8*4),len);PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void Load(uint64_t batch){
  DetFence<HardEvent::S_V>();Duplicate(mat,0.0f,columns*pitch);PipeBarrier<PIPE_V>();
  for(uint32_t r=0;r<n;r+=8){uint32_t rows=n-r<8?n-r:8;
   Duplicate(loaded,0.0f,8*pitch);DetFence<HardEvent::V_MTE2>();
   DataCopyPad(loaded,input[(batch*n+r)*n],DataCopyExtParams{static_cast<uint16_t>(rows),n*4,0,0,0},DataCopyPadExtParams<float>{false,0,0,0});
   DetFence<HardEvent::MTE2_V>();Gather(transposed,loaded,indices,0,count*8);PipeBarrier<PIPE_V>();
   DataCopy(mat[r],transposed,DataCopyParams{static_cast<uint16_t>(count),1,0,static_cast<uint16_t>((pitch-8)/8)});PipeBarrier<PIPE_V>();
  }
 }
 __aicore__ inline void NormalizeOwner(LocalTensor<float> factor,LocalTensor<float> tile,uint32_t cols,uint32_t row,float diagonal){
  // Pivot() fenced all prior tile updates. Each column coefficient read is disjoint from preceding Axpy writes.
  SetMaskCount();SetVectorMask<float,MaskMode::COUNTER>(0,pitch);
  Bits normalization;normalization.f=diagonal;
  float divisor=diagonal;
  if((normalization.u&0x7f800000u)==0u){
   Muls<float,false>(factor,factor,16777216.0f,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
   PipeBarrier<PIPE_V>();divisor*=16777216.0f;
  }
  Muls<float,false>(factor,factor,-1.0f/divisor,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
  PipeBarrier<PIPE_V>();
  for(uint32_t col=0;col<cols;++col){
   float coefficient=tile[col*pitch].GetValue(row);
   Axpy<float,float,false>(tile[col*pitch],factor,coefficient,uint64_t(0),1,UnaryRepeatParams{1,1,8,8});
  }
  ResetMask();SetMaskNorm();PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Update(LocalTensor<float> dst,uint32_t cols,LocalTensor<float> factor,uint32_t row){
  if(!cols)return;
  Gather(pivot,dst,bases.ReinterpretCast<uint32_t>(),row*4,cols);PipeBarrier<PIPE_V>();
  Brcb(spread,pivot,(cols+7)/8,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
  uint32_t full=n/64*64;
  SetVectorMask<float,MaskMode::NORMAL>(0,~uint64_t(0));
  for(uint32_t r=0;r<full;r+=64){
   MulAddDst<float,float,false>(dst[r],factor[r],spread,uint64_t(64),cols,BinaryRepeatParams{1,1,0,static_cast<uint8_t>(pitch/8),0,1});
  }
  if(full<n)MulAddDst(dst[full],factor[full],spread,n-full,cols,BinaryRepeatParams{1,1,0,static_cast<uint8_t>(pitch/8),0,1});
  PipeBarrier<PIPE_V>();
 }
 __aicore__ inline uint32_t Pivot(LocalTensor<float> factor,float& largest){
  Abs(absolute,factor,pitch);PipeBarrier<PIPE_V>();
  WholeReduceMax(reduced,absolute,64,pitch/64,4,1,8,ReduceOrder::ORDER_VALUE_INDEX);
  if(pitch%64)WholeReduceMax(reduced[(pitch/64)*8],absolute[(pitch/64)*64],pitch%64,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);
  PipeBarrier<PIPE_V>();uint64_t mask[2]={0,0};for(uint32_t b=0;b<(n+63)/64;++b)mask[0]|=uint64_t(1)<<(b*8);
  WholeReduceMax(absolute,reduced,mask,1,1,1,8,ReduceOrder::ORDER_VALUE_INDEX);DetFence<HardEvent::V_S>();
  largest=absolute.GetValue(0);uint32_t slot=absolute.ReinterpretCast<uint32_t>().GetValue(1);
  return slot*8+reduced.ReinterpretCast<uint32_t>().GetValue(slot+1);
 }
 union Bits {float f;uint32_t u;};



 __aicore__ inline void Process(){
  uint64_t waves=(batches+groups-1)/groups;
  for(uint64_t wave=0;wave<waves;++wave){
   uint64_t batch=wave*groups+group;bool valid=batch<batches,singular=false;
   float product=1;int32_t exponent=0;uint64_t remainingRows[8]={};
   for(uint32_t b=0;b<(n+63)/64;++b){uint32_t count=n-b*64;remainingRows[b]=count>=64?~uint64_t(0):(uint64_t(1)<<count)-1;}
   if(valid){if(count)Load(batch);Duplicate(active,0.0f,pitch);PipeBarrier<PIPE_V>();Duplicate(active,1.0f,n);PipeBarrier<PIPE_V>();}
   for(uint32_t panel=0;panel<n;panel+=8){
    uint32_t stop=panel+8<n?panel+8:n,steps=stop-panel,slot=((panel/8)%2)*8*(pitch+8);
    bool owner=panel>=begin&&panel<end;
    if(valid&&!singular&&owner){
    PublishOwnedPanel(panel,stop,slot,steps);
    }
    SyncAll();
    if(valid&&!singular){
    ConsumePanel(owner,slot,steps,panel,stop,singular,remainingRows,product,exponent);
    }
   }
   if(valid&&lane==sink){result.SetValue(0,singular?0:DetFinish1(product,exponent));DetFence<HardEvent::S_MTE3>();
    DataCopyPad(output[batch],result,DataCopyExtParams{1,4,0,0,0});DetFence<HardEvent::MTE3_S>();}
   SyncAll();
  }
 }
private:
 TPipe pipe;
 LocalTensor<float> mat,loaded,transposed,packet,absolute,active,reduced,pivot,spread,result;LocalTensor<uint32_t> indices;LocalTensor<int32_t> bases;
 GlobalTensor<float> input,output,work;uint32_t n,pitch,cores,workers,lanes,sink,lane,group,groups,columns,begin,end,count;uint64_t batches;
};


#endif
