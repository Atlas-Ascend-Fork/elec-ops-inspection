// Copyright 2026 XDU_XDU-BTD Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::CachedProduct(GlobalTensor<float>dst,bool square,bool atomic)
{
if(logicalN_==200u){DenseProduct224(dst,square,atomic);return;}


for(Zq2 nb=0;nb<4u;++nb){LoadData2dTransposeParams tr;tr.startIndex=uint16_t((lane_*64u)/16u+nb);tr.repeatTimes=16u;
tr.srcStride=16u;tr.dstGap=7u;tr.dstFracGap=3u;
if(square)LoadDataWithTranspose(b2_[nb*128u],a1_,tr);else LoadDataWithTranspose(b2_[nb*128u],b1_,tr);}
for(Zq2 tile=lane_;tile<16u;tile+=4u){Zq2 row=tile/4u*64u,col=tile%4u*64u;
LoadData2DParams la;la.repeatTimes=128u;la.srcStride=1u;la.dstGap=0u;la.ifTranspose=false;LoadData(a2_,a1_[row*N],la);

ZqB<Zq5::MTE1_M>();Multiply(64u,64u,256u);Store(dst,row*N+col,64u,64u,atomic);
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::CachedSquareFromB256(GlobalTensor<float>dst)
{


for(Zq2 nb=0;nb<4u;++nb){LoadData2dTransposeParams tr;tr.startIndex=uint16_t((lane_*64u)/16u+nb);tr.repeatTimes=16u;
tr.srcStride=16u;tr.dstGap=7u;tr.dstFracGap=3u;
LoadDataWithTranspose(b2_[nb*128u],b1_,tr);}
for(Zq2 tile=lane_;tile<16u;tile+=4u){Zq2 row=tile/4u*64u,col=tile%4u*64u;
LoadData2DParams la;la.repeatTimes=128u;la.srcStride=1u;la.dstGap=0u;la.ifTranspose=false;LoadData(a2_,b1_[row*N],la);
ZqB<Zq5::MTE1_M>();Multiply(64u,64u,256u);Store(dst,row*N+col,64u,64u,false);
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::InitialGramUpper200()
{
CacheDense224(b1_,qt_);
Zq2 lastCol=N;
const Zq2 tasks=lane_==3u?4u:2u;
for(Zq2 task=0u;task<tasks;++task){
Zq2 row,col,rows,cols;
if(lane_==3u){row=task*64u;col=192u;rows=task==3u?16u:64u;cols=16u;}
else{
rows=64u;cols=64u;
if(lane_==0u){row=0u;col=task*64u;}
else if(lane_==1u){row=task==0u?64u:0u;col=64u+task*64u;}
else{row=64u+task*64u;col=128u;}
}
if(col!=lastCol){
for(Zq2 nb=0u;nb<cols/16u;++nb){LoadData2DParams lb;lb.repeatTimes=26u;
lb.srcStride=1u;lb.dstGap=uint16_t(cols/16u-1u);lb.ifTranspose=false;
LoadData(b2_[nb*128u],b1_[(col/16u+nb)*3328u],lb);}
lastCol=col;
}
LoadData2DParams la;la.repeatTimes=uint16_t(rows*208u/128u);la.srcStride=1u;la.dstGap=0u;la.ifTranspose=false;
LoadData(a2_,b1_[row*208u],la);ZqB<Zq5::MTE1_M>();
Multiply(rows,cols,208u);Store(g_,row*N+col,rows,cols,false);
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::EarlyFirstTail256(Zq2 j,Zq2 end,Zq2 tail)
{


const Zq2 width=tail<64u?tail:64u;
DenseBegin(true);DenseWaitMte2();
CopyNd(a1_,neg_[j*N+end],32u,width,N);
CopyNd(b1_,g_[j*N+end],32u,width,N);ZqB<Zq5::MTE2_MTE1>();
LoadATransposeFactor(width);LoadB(width,32u);
ZqB<Zq5::MTE1_M>();DenseMultiply(width,width,32u);
DenseStore(g_,end*N+end,width,width,N);DenseEnd();
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::EarlyR64Block200(Zq2 pair)
{
const Zq2 j=pair*64u;
GlobalTensor<float>frames,tmp;
frames.SetGlobalBuffer(workspace_+356096u,32768u);
const Zq2 block=(pair/2u)*16384u+(pair%2u)*8256u;
tmp.SetGlobalBuffer(workspace_+213248u+pair*1024u,1024u);
CopyNd(a1_,neg_[j*N+j+32u],32u,32u,N);
CopyNd(b1_,frames[block+4128u],32u,32u,128u);
ZqB<Zq5::MTE2_MTE1>();
LoadData2DParams ld;ld.repeatTimes=8u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;
LoadData(a2_,a1_,ld);LoadB(32u,32u);ZqB<Zq5::MTE1_M>();
Multiply(32u,32u,32u);Store(tmp,0u,32u,32u,false,32u);
CopyNd(a1_,frames[block],32u,32u,128u);
CopyNd(b1_,tmp,32u,32u,32u);ZqB<Zq5::MTE2_MTE1>();
LoadData(a2_,a1_,ld);LoadB(32u,32u);ZqB<Zq5::MTE1_M>();

Multiply(32u,32u,32u);Store(frames,block+32u,32u,32u,false,128u);
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::PrepareR64Direct200()
{
GlobalTensor<float>frames;frames.SetGlobalBuffer(workspace_+356096u,32768u);

if(lane_<2u){
const Zq2 r=lane_*128u;
GlobalTensor<float>scratch;
scratch.SetGlobalBuffer(workspace_+213248u+lane_*4096u,4096u);
CopyNd(a1_,neg_[r*N+r+64u],64u,64u,N);
CopyNd(b1_,frames[lane_*16384u+8256u],64u,64u,128u);ZqB<Zq5::MTE2_MTE1>();
LoadData2DParams wideLd;wideLd.repeatTimes=32u;wideLd.srcStride=1u;wideLd.dstGap=0u;wideLd.ifTranspose=false;
LoadData(a2_,a1_,wideLd);LoadB(64u,64u);ZqB<Zq5::MTE1_M>();
Multiply(64u,64u,64u);Store(scratch,0u,64u,64u,false,64u);
CopyNd(a1_,frames[lane_*16384u],64u,64u,128u);
CopyNd(b1_,scratch,64u,64u,64u);ZqB<Zq5::MTE2_MTE1>();
LoadData(a2_,a1_,wideLd);LoadB(64u,64u);ZqB<Zq5::MTE1_M>();
Multiply(64u,64u,64u);Store(frames,lane_*16384u+64u,64u,64u,false,128u);
}
CrossCoreSetFlag<2,PIPE_FIX>(11);CrossCoreWaitFlag(12);
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::PrepareR64()
{
if(logicalN_==200u){PrepareR64Direct200();return;}
const Zq2 j=lane_*64u;
GlobalTensor<float>frame,tmp;
frame.SetGlobalBuffer(workspace_+213248u+lane_*4096u,4096u);
tmp.SetGlobalBuffer(workspace_+213248u+16384u+lane_*1024u,1024u);
CopyNd(a1_,neg_[j*N+j+32u],32u,32u,N);
CopyNd(b1_,invNeg_[(lane_*2u+1u)*1024u],32u,32u,32u);
ZqB<Zq5::MTE2_MTE1>();
LoadData2DParams ld;ld.repeatTimes=8u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;
LoadData(a2_,a1_,ld);LoadB(32u,32u);ZqB<Zq5::MTE1_M>();
Multiply(32u,32u,32u);Store(tmp,0u,32u,32u,false,32u);
CopyNd(a1_,invNeg_[lane_*2u*1024u],32u,32u,32u);
CopyNd(b1_,tmp,32u,32u,32u);ZqB<Zq5::MTE2_MTE1>();
LoadData(a2_,a1_,ld);LoadB(32u,32u);ZqB<Zq5::MTE1_M>();
Multiply(32u,32u,32u);Store(frame,32u,32u,32u,false,64u);


CrossCoreSetFlag<2,PIPE_FIX>(11);CrossCoreWaitFlag(12);

if(lane_<2u){
const Zq2 r=lane_*128u;
GlobalTensor<float>blocks,wide,scratch;
blocks.SetGlobalBuffer(workspace_+213248u,16384u);
wide.SetGlobalBuffer(workspace_+356096u+lane_*16384u,16384u);
scratch.SetGlobalBuffer(workspace_+213248u+16384u+lane_*4096u,4096u);
CopyNd(a1_,neg_[r*N+r+64u],64u,64u,N);
CopyNd(b1_,blocks[(lane_*2u+1u)*4096u],64u,64u,64u);ZqB<Zq5::MTE2_MTE1>();
LoadData2DParams wideLd;wideLd.repeatTimes=32u;wideLd.srcStride=1u;wideLd.dstGap=0u;wideLd.ifTranspose=false;
LoadData(a2_,a1_,wideLd);LoadB(64u,64u);ZqB<Zq5::MTE1_M>();
Multiply(64u,64u,64u);Store(scratch,0u,64u,64u,false,64u);
CopyNd(a1_,blocks[lane_*2u*4096u],64u,64u,64u);
CopyNd(b1_,scratch,64u,64u,64u);ZqB<Zq5::MTE2_MTE1>();
LoadData(a2_,a1_,wideLd);LoadB(64u,64u);ZqB<Zq5::MTE1_M>();
Multiply(64u,64u,64u);Store(wide,64u,64u,64u,false,128u);
}
CrossCoreSetFlag<2,PIPE_FIX>(11);CrossCoreWaitFlag(12);
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::PrepareDistributedR256200()
{
if(lane_>=2u)return;
GlobalTensor<float>wide;wide.SetGlobalBuffer(workspace_+356096u,32768u);
const Zq2 row=lane_*64u;
LoadData2DParams ld;ld.repeatTimes=64u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;
CopyNd(a1_,wide[row*128u],64u,128u,128u);
CopyNd(b1_,neg_[128u],128u,80u,N);ZqB<Zq5::MTE2_MTE1>();
LoadData(a2_,a1_,ld);LoadB(80u,128u);ZqB<Zq5::MTE1_M>();
Multiply(64u,80u,128u);Store(g_,(128u+row)*N,64u,80u,false);
CopyNd(a1_,g_[(128u+row)*N],64u,80u,N);
CopyNd(b1_,wide[16384u],80u,80u,128u);ZqB<Zq5::MTE2_MTE1>();
ld.repeatTimes=40u;
LoadData(a2_,a1_,ld);LoadB(80u,80u);ZqB<Zq5::MTE1_M>();
Multiply(64u,80u,80u);Store(g_,(128u+row)*N,64u,80u,false);
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
template<bool IsGram>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::SymmetricUpper256(GlobalTensor<float>dst)
{
if constexpr(IsGram)CacheB(qt_);
Zq2 lastRow=N,lastCol=N;
const Zq2 tasks=lane_<2u?3u:2u;
for(Zq2 task=0u;task<tasks;++task){
Zq2 row,col;
if(lane_<2u){
row=lane_*64u;
col=task==0u?row:(task==2u?192u:(lane_==0u?64u:128u));
}else{col=lane_*64u;row=task==0u?col:(lane_==2u?0u:128u);}
if(col!=lastCol){
if constexpr(IsGram){
for(Zq2 nb=0u;nb<4u;++nb){LoadData2DParams lb;lb.repeatTimes=32u;lb.srcStride=1u;lb.dstGap=3u;lb.ifTranspose=false;
LoadData(b2_[nb*128u],b1_[(col/16u+nb)*4096u],lb);}
}else{
for(Zq2 nb=0u;nb<4u;++nb){LoadData2dTransposeParams tr;tr.startIndex=uint16_t(col/16u+nb);tr.repeatTimes=16u;
tr.srcStride=16u;tr.dstGap=7u;tr.dstFracGap=3u;LoadDataWithTranspose(b2_[nb*128u],a1_,tr);}
}
lastCol=col;
}
if(row!=lastRow){LoadData2DParams la;la.repeatTimes=128u;la.srcStride=1u;la.dstGap=0u;la.ifTranspose=false;
if constexpr(IsGram)LoadData(a2_,b1_[row*N],la);else LoadData(a2_,a1_[row*N],la);
lastRow=row;}
ZqB<Zq5::MTE1_M>();Multiply(64u,64u,256u);Store(dst,row*N+col,64u,64u,false);
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::DenseProduct224(GlobalTensor<float>dst,bool square,bool atomic)
{
const Zq2 col=lane_*64u,cols=lane_==3u?16u:64u;
for(Zq2 nb=0u;nb<cols/16u;++nb){LoadData2dTransposeParams tr;
tr.startIndex=uint16_t(col/16u+nb);tr.repeatTimes=13u;tr.srcStride=13u;
tr.dstGap=uint16_t(cols/8u-1u);tr.dstFracGap=uint16_t(cols/16u-1u);
if(square)LoadDataWithTranspose(b2_[nb*128u],a1_,tr);else LoadDataWithTranspose(b2_[nb*128u],b1_,tr);}
for(Zq2 row=0u;row<208u;row+=64u){
const Zq2 rows=row==192u?16u:64u;
LoadData2DParams la;la.repeatTimes=uint16_t(rows*208u/128u);la.srcStride=1u;la.dstGap=0u;la.ifTranspose=false;
LoadData(a2_,a1_[row*208u],la);ZqB<Zq5::MTE1_M>();
Multiply(rows,cols,208u);Store(dst,row*N+col,rows,cols,atomic);
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline typename Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::PoolTask Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::PoolGeometry(Zq2 mask)
{
const Zq2 active=(mask&1u)+((mask>>1u)&1u)+((mask>>2u)&1u)+((mask>>3u)&1u);
const Zq2 worker=GetBlockIdx(),workers=GetBlockNum();
Zq2 group,rank,peers;
if(active==1u){group=worker&3u;rank=worker>>2u;peers=workers>>2u;}
else if(active==2u){group=worker&7u;rank=worker>>3u;peers=1u+Zq2(group<workers-8u);}
else if(active==3u){group=worker>=12u?worker-12u:worker;rank=Zq2(worker>=12u);peers=1u+Zq2(group<workers-12u);}
else{group=worker;rank=0u;peers=1u;}
const Zq2 ordinal=group>>2u;
Zq2 selected=mask;
if(ordinal>0u)selected&=selected-1u;
if(ordinal>1u)selected&=selected-1u;
if(ordinal>2u)selected&=selected-1u;
const Zq2 batch=(selected&1u)?0u:((selected&2u)?1u:((selected&4u)?2u:3u));
return PoolTask{batch,(worker&3u)*64u,rank*64u,peers*64u};
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::ComputePool(const PoolTask&task,Zq2 dstOffset,bool atomic)
{
const Zq2 batch=task.batch,col=task.col,step=task.rowStep;Zq2 row=task.firstRow;
GlobalTensor<float>dst;dst.SetGlobalBuffer(PoolBase()+Zq3(batch)*Group+dstOffset,N*N);
LoadB(64u,256u);
for(Zq2 slot=0u;row<N;row+=step,++slot){
LoadData2DParams la;la.repeatTimes=128u;la.srcStride=1u;la.dstGap=0u;la.ifTranspose=false;
LoadData(a2_,a1_[slot*16384u],la);ZqB<Zq5::MTE1_M>();
Multiply(64u,64u,256u);Store(dst,row*N+col,64u,64u,atomic);
}
}
