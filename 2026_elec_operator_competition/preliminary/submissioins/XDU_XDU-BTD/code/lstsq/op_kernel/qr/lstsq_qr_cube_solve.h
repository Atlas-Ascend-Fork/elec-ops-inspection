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
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::LeftTrsm128()
{
PrepareR64();
GlobalTensor<float>wide;wide.SetGlobalBuffer(workspace_+356096u,32768u);
const Zq2 col=lane_*64u;
for(Zq2 j=0u;j<256u;j+=128u){
CopyNd(a1_,wide[(j/128u)*16384u],128u,128u,128u);
CopyNd(b1_,qt_[j*N+col],128u,64u,N);ZqB<Zq5::MTE2_MTE1>();
LoadATranspose128();LoadB(64u,128u);ZqB<Zq5::MTE1_M>();
Multiply(128u,64u,128u);Store(qt_,j*N+col,128u,64u,false);
if(j==0u){
CopyNd(a1_,neg_[128u],128u,128u,N);CopyNd(b1_,qt_[col],128u,64u,N);
ZqB<Zq5::MTE2_MTE1>();LoadATranspose128();LoadB(64u,128u);
ZqB<Zq5::MTE1_M>();Multiply(128u,64u,128u);Store(qt_,128u*N+col,128u,64u,true);
}
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::LeftTrsm()
{


const Zq2 limit=logicalN_==200u?224u:N;
Zq2 col=lane_*64u,cols=(logicalN_==200u&&lane_==3u)?32u:64u;
for(Zq2 j=0;j<limit;j+=32u){
CopyNd(a1_,inv_[j/32u*1024u],32u,32u,32u);CopyNd(b1_,qt_[j*N+col],32u,cols,N);ZqB<Zq5::MTE2_MTE1>();
LoadData2DParams ld;ld.repeatTimes=8u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;LoadData(a2_,a1_,ld);LoadB(cols,32u);
ZqB<Zq5::MTE1_M>();Multiply(32u,cols,32u);Store(qt_,j*N+col,32u,cols,false);
if(j+32u<limit){

CopyNd(b1_,qt_[j*N+col],32u,cols,N);ZqB<Zq5::MTE2_MTE1>();LoadB(cols,32u);


for(Zq2 row=j+32u;row<limit;row+=128u){
const Zq2 rows=limit-row<128u?limit-row:128u;
CopyNd(a1_,neg_[j*N+row],32u,rows,N);ZqB<Zq5::MTE2_MTE1>();LoadATransposeFactor(rows);
ZqB<Zq5::MTE1_M>();Multiply(rows,cols,32u);Store(qt_,row*N+col,rows,cols,true);
}
}
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::ComposeConsumer()
{
Zq2 col=lane_*64u,cols=64u;
for(int32_t jj=224;jj>=0;jj-=32){
Zq2 j=Zq2(jj);
CopyNd(a1_,invNeg_[j/32u*1024u],32u,32u,32u);
CopyNd(b1_,z_[j*N+col],32u,cols,N);ZqB<Zq5::MTE2_MTE1>();
LoadData2DParams ld;ld.repeatTimes=8u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;
LoadData(a2_,a1_,ld);LoadB(cols,32u);ZqB<Zq5::MTE1_M>();
Multiply(32u,cols,32u);Store(z_,j*N+col,32u,cols,false);
if(j){
CopyNd(b1_,z_[j*N+col],32u,cols,N);ZqB<Zq5::MTE2_MTE1>();LoadB(cols,32u);
for(Zq2 row=0u;row<j;row+=32u){
CopyNd(a1_,neg_[row*N+j],32u,32u,N);ZqB<Zq5::MTE2_MTE1>();
LoadData(a2_,a1_,ld);ZqB<Zq5::MTE1_M>();
Multiply(32u,cols,32u);Store(z_,row*N+col,32u,cols,true);
}
}
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::SolveRhsFrame64(GlobalTensor<float>rhs)
{
for(int32_t jj=128;jj>=0;jj-=128){
Zq2 j=Zq2(jj),jb=j/128u;
CopyNd(b1_,rhs[j*16u],128u,16u,16u);ZqB<Zq5::MTE2_MTE1>();LoadB(16u,128u);
LoadData2DParams ld;ld.repeatTimes=128u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;
LoadData(a2_,a1_[16384u+jb*16384u],ld);ZqB<Zq5::MTE1_M>();
Multiply(128u,16u,128u);Store(rhs,j*16u,128u,16u,false,16u);
if(j){
CopyNd(b1_,rhs[j*16u],128u,16u,16u);ZqB<Zq5::MTE2_MTE1>();LoadB(16u,128u);
LoadData(a2_,a1_[49152u],ld);ZqB<Zq5::MTE1_M>();
Multiply(128u,16u,128u);Store(rhs,0u,128u,16u,true,16u);
}
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::SolveDistributedR256200(GlobalTensor<float>src,GlobalTensor<float>dst)
{
const Zq2 row=lane_*64u,j=lane_<2u?0u:128u,k=lane_<2u?128u:80u,rows=lane_==3u?16u:64u;
LoadData2DParams ld;ld.repeatTimes=uint16_t(rows*k/128u);ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;
CopyNd(b1_,src[j*16u],k,16u,16u);ZqB<Zq5::MTE2_MTE1>();LoadB(16u,k);
LoadData(a2_,a1_[16384u],ld);ZqB<Zq5::MTE1_M>();
Multiply(rows,16u,k);Store(dst,row*16u,rows,16u,false,16u);
if(lane_<2u){
CopyNd(b1_,src[128u*16u],80u,16u,16u);ZqB<Zq5::MTE2_MTE1>();LoadB(16u,80u);
ld.repeatTimes=40u;
LoadData(a2_,a1_[24576u],ld);ZqB<Zq5::MTE1_M>();
Multiply(64u,16u,80u);Store(dst,row*16u,64u,16u,true,16u);
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::PrepareFactoredApply()
{
CrossCoreWaitFlag(14);factoredApply_=!ComposedPath();
if(logicalN_==200u){PrepareDistributedR256200();CrossCoreSetFlag<2,PIPE_FIX>(15);return;}
if(logicalN_==256u||logicalN_==200u){
if(lane_==0u){
GlobalTensor<float>wide;wide.SetGlobalBuffer(workspace_+356096u,32768u);
for(Zq2 b=0u;b<2u;++b)CopyNd(a1_[16384u+b*16384u],wide[b*16384u],128u,128u,128u);
if constexpr(ColumnFactored256){CopyNd(a1_[49152u],g_[128u],128u,128u,N);}else{CopyNd(a1_[49152u],neg_[128u],128u,128u,N);}ZqB<Zq5::MTE2_MTE1>();
}
CrossCoreSetFlag<2,PIPE_FIX>(15);return;}
if(!factoredApply_)ComposeConsumer();else if(lane_==0u){
const Zq2 rBlocks=logicalN_==200u?7u:8u;
for(Zq2 j=0u;j<rBlocks;++j){
CopyNd(a1_[16384u+j*1024u],invNeg_[j*1024u],32u,32u,32u);
for(Zq2 i=0u;i<j;++i)CopyNd(a1_[16384u+8192u+(j*(j-1u)/2u+i)*1024u],neg_[i*32u*N+j*32u],32u,32u,N);
}
ZqB<Zq5::MTE2_MTE1>();
}
CrossCoreSetFlag<2,PIPE_FIX>(15);
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::SolveRhsFrame(GlobalTensor<float>rhs)
{
for(int32_t jj=logicalN_==200u?192:224;jj>=0;jj-=32){
Zq2 j=Zq2(jj),jb=j/32u;
CopyNd(b1_,rhs[j*16u],32u,16u,16u);ZqB<Zq5::MTE2_MTE1>();LoadB(16u,32u);
LoadData2DParams ld;ld.repeatTimes=8u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;
LoadData(a2_,a1_[16384u+jb*1024u],ld);ZqB<Zq5::MTE1_M>();Multiply(32u,16u,32u);StoreRhs(rhs,j,false);
if(j){
CopyNd(b1_,rhs[j*16u],32u,16u,16u);ZqB<Zq5::MTE2_MTE1>();LoadB(16u,32u);


LoadData2DParams prefix=ld;prefix.repeatTimes=uint16_t(j/4u);
LoadData(a2_,a1_[16384u+8192u+(jb*(jb-1u)/2u)*1024u],prefix);
ZqB<Zq5::MTE1_M>();Multiply(j,16u,32u);
FixpipeParamsV220 fp;fp.mSize=uint16_t(j);fp.nSize=16u;fp.srcStride=uint16_t(j);fp.dstStride=16u;fp.quantPre=QuantMode_t::NoQuant;
SetAtomicAdd<float>();Fixpipe(rhs,c_,fp);

if(logicalN_==200u){ZqB<Zq5::FIX_M>();ZqB<Zq5::FIX_S>();ZqB<Zq5::FIX_MTE2>();}
else Zq1<PIPE_ALL>();SetAtomicNone();
}
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::ConsumeRhs()
{
const Zq2 tiles=logicalN_==200u?7u:8u;
auto ptr=WorkspacePointer();GlobalTensor<float>decision,inRhs,outRhs;
decision.SetGlobalBuffer(ptr+344448u,16u);inRhs.SetGlobalBuffer(ptr+344576u,4096u);outRhs.SetGlobalBuffer(ptr+348672u,4096u);
CrossCoreWaitFlag(0);
DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(decision);
const bool accepted=decision.GetValue(0u)!=0.0f;
if(accepted){
if constexpr(ColumnFactored256){


if(lane_!=0u){
GlobalTensor<float>wide;wide.SetGlobalBuffer(ptr+356096u,32768u);
for(Zq2 block=0u;block<2u;++block)CopyNd(a1_[16384u+block*16384u],wide[block*16384u],128u,128u,128u);
CopyNd(a1_[49152u],g_[128u],128u,128u,N);
}
CopyNd(b1_[16384u],qt_[lane_*16384u],64u,256u,256u);
const Zq2 col=lane_*64u;
LoadData2DParams ld;ld.repeatTimes=128u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;

CopyNd(b1_,second_[128u*N+col],128u,64u,N);ZqB<Zq5::MTE2_MTE1>();LoadB(64u,128u);
LoadData(a2_,a1_[32768u],ld);ZqB<Zq5::MTE1_M>();
Multiply(128u,64u,128u);Store(z_,128u*N+col,128u,64u,false);

CopyNd(b1_,z_[128u*N+col],128u,64u,N);ZqB<Zq5::MTE2_MTE1>();LoadB(64u,128u);
LoadData(a2_,a1_[49152u],ld);ZqB<Zq5::MTE1_M>();
Multiply(128u,64u,128u);Store(z_,col,128u,64u,false);
}else{
if constexpr(CropZ200){
GlobalTensor<Zq2>ownDegree;ownDegree.SetGlobalBuffer(reinterpret_cast<__gm__ Zq2*>(ptr)+43u,1u);
cropZ200_=ownDegree.GetValue(0u)==1u;
}
if(CropZMode200()){
for(Zq2 tile=lane_,slot=0u;tile<7u;tile+=4u,++slot)
CopyNd(a1_[slot*6656u],z_[tile*8192u],32u,208u,N);
}else{for(Zq2 tile=lane_,slot=0u;tile<tiles;tile+=4u,++slot)CopyNd(a1_[slot*8192u],z_[tile*8192u],32u,256u,256u);}
}
ZqB<Zq5::MTE2_MTE1>();
if(logicalN_==200u)CacheDistributedR256200();
}

if constexpr(ColumnFactored256){Zq1<PIPE_ALL>();CrossCoreSetFlag<2,PIPE_FIX>(1);}
else{CrossCoreSetFlag<2,PIPE_MTE2>(1);}if(!accepted)return;
GlobalTensor<Zq2>degree;degree.SetGlobalBuffer(reinterpret_cast<__gm__ Zq2*>(ptr)+43u,1u);
const Zq2 count=EnableBoundedH3?(degree.GetValue(0u)==1u?3u:4u):(degree.GetValue(0u)==7u?4u:3u);
for(Zq2 apply=0u;apply<count;++apply){
CrossCoreWaitFlag(0);
if constexpr(ColumnFactored256){
if(apply==0u){


const Zq2 col=lane_*64u;
CopyNd(b1_,z_[col],128u,64u,N);ZqB<Zq5::MTE2_MTE1>();LoadB(64u,128u);
LoadData2DParams ld;ld.repeatTimes=128u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;
LoadData(a2_,a1_[16384u],ld);ZqB<Zq5::MTE1_M>();
Multiply(128u,64u,128u);Store(z_,col,128u,64u,false);
CopyNd(a1_,z_[col],256u,64u,N);ZqB<Zq5::MTE2_MTE1>();
}


GlobalTensor<float>partial;partial.SetGlobalBuffer(ptr+213248u+lane_*1024u,1024u);
CopyNd(b1_,inRhs,256u,16u,16u);ZqB<Zq5::MTE2_MTE1>();LoadB(16u,256u);
LoadData2DParams qld;qld.repeatTimes=128u;qld.srcStride=1u;qld.dstGap=0u;qld.ifTranspose=false;
LoadData(a2_,b1_[16384u],qld);ZqB<Zq5::MTE1_M>();
Multiply(64u,16u,256u);

FixpipeParamsV220 relay;relay.mSize=64u;relay.nSize=16u;relay.srcStride=64u;relay.dstStride=16u;relay.quantPre=QuantMode_t::NoQuant;
Fixpipe(partial,c_,relay);ZqB<Zq5::FIX_MTE2>();
CopyNd(b1_,partial,64u,16u,16u);ZqB<Zq5::MTE2_MTE1>();LoadB(16u,64u);
LoadData(a2_,a1_,qld);ZqB<Zq5::MTE1_M>();
Multiply(256u,16u,64u);Store(outRhs,0u,256u,16u,true,16u);
}else{
if(CropZMode200()){
CopyNd(b1_,inRhs,208u,16u,16u);ZqB<Zq5::MTE2_MTE1>();LoadB(16u,208u);
for(Zq2 tile=lane_,slot=0u;tile<7u;tile+=4u,++slot){
LoadData2DParams ld;ld.repeatTimes=52u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;
LoadData(a2_,a1_[slot*6656u],ld);ZqB<Zq5::MTE1_M>();
Multiply(32u,16u,208u);Store(outRhs,tile*512u,32u,16u,false,16u);
}
}else{
CopyNd(b1_,inRhs,256u,16u,16u);ZqB<Zq5::MTE2_MTE1>();LoadB(16u,256u);
for(Zq2 tile=lane_,slot=0u;tile<tiles;tile+=4u,++slot){
LoadData2DParams ld;ld.repeatTimes=64u;ld.srcStride=1u;ld.dstGap=0u;ld.ifTranspose=false;
LoadData(a2_,a1_[slot*8192u],ld);ZqB<Zq5::MTE1_M>();Multiply(32u,16u,256u);
FixpipeParamsV220 fp;fp.mSize=32u;fp.nSize=16u;fp.srcStride=32u;fp.dstStride=16u;fp.quantPre=QuantMode_t::NoQuant;
Fixpipe(outRhs[tile*512u],c_,fp);Zq1<PIPE_ALL>();
}
}
}
CrossCoreSetFlag<2,PIPE_FIX>(1);
if constexpr(!ColumnFactored256){if(factoredApply_){CrossCoreWaitFlag(0);if(logicalN_==200u){SolveDistributedR256200(outRhs,inRhs);}else if(lane_==0u){SolveRhsFrame64(outRhs);}CrossCoreSetFlag<2,PIPE_FIX>(1);}}
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::FinalConsumerZ()
{
if constexpr(ColumnFactored256){if(logicalN_==256u)return;}
if(logicalN_==200u){FinalConsumerZ224();return;}


Zq2 lastRow=N;

for(Zq2 nb=0;nb<4u;++nb){LoadData2dTransposeParams tr;tr.startIndex=uint16_t((lane_*64u)/16u+nb);tr.repeatTimes=16u;
tr.srcStride=16u;tr.dstGap=7u;tr.dstFracGap=3u;LoadDataWithTranspose(b2_[nb*128u],b1_,tr);}
for(Zq2 tile=lane_;tile<16u;tile+=4u){
Zq2 row=tile/4u*64u,col=tile%4u*64u;
if(row!=lastRow){CopyNd(a1_,second_[row*N],64u,256u,N);ZqB<Zq5::MTE2_MTE1>();
LoadData2DParams la;la.repeatTimes=128u;la.srcStride=1u;la.dstGap=0u;la.ifTranspose=false;LoadData(a2_,a1_,la);lastRow=row;}

ZqB<Zq5::MTE1_M>();Multiply(64u,64u,256u);Store(z_,row*N+col,64u,64u,false);
}
}

template<bool EnableBoundedH3,bool ColumnFactored256,bool CropZ200>
__aicore__ inline void Cube<EnableBoundedH3, ColumnFactored256, CropZ200>::FinalConsumerZ224()
{
const Zq2 col=lane_*64u,cols=lane_==3u?16u:64u;
for(Zq2 nb=0u;nb<cols/16u;++nb){LoadData2dTransposeParams tr;
tr.startIndex=uint16_t(col/16u+nb);tr.repeatTimes=13u;tr.srcStride=13u;
tr.dstGap=uint16_t(cols/8u-1u);tr.dstFracGap=uint16_t(cols/16u-1u);
LoadDataWithTranspose(b2_[nb*128u],b1_,tr);}
for(Zq2 row=0u;row<208u;row+=64u){
const Zq2 rows=row==192u?16u:64u;
CopyNd(a1_,second_[row*N],rows,208u,N);ZqB<Zq5::MTE2_MTE1>();
LoadData2DParams la;la.repeatTimes=uint16_t(rows*208u/128u);la.srcStride=1u;la.dstGap=0u;la.ifTranspose=false;
LoadData(a2_,a1_,la);ZqB<Zq5::MTE1_M>();
Multiply(rows,cols,208u);Store(z_,row*N+col,rows,cols,false);
}
}
