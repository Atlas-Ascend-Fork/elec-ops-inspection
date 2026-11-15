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

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::QrCubeApply(Zq3 shared,bool active,bool residualPublished,bool prepareEmit,bool prepareResidualMap)
{
bool finalFrames=false;
if constexpr(CompactConsumer200)finalFrames=prepareEmit&&dual200_&&qrLogicalN_==200u;
if constexpr(ColumnFactored256)finalFrames=prepareEmit&&dual200_&&qrLogicalN_==256u;
if constexpr(ColumnFactored256){


if(qrLogicalN_==256u&&GetBlockIdx()%8u==0u){
GlobalTensor<float>clear;clear.SetGlobalBuffer(w_+shared+348672u,4096u);
ZqB<Zq5::S_V>();Duplicate(factors_,0.0f,4096u);
ZqB<Zq5::V_MTE3>();DataCopy(clear,factors_,4096u);ZqB<Zq5::MTE3_V>();
}
}
if(dual200_&&qrLogicalN_==200u){


GlobalTensor<float>xs;xs.SetGlobalBuffer(w_+shared+(residualPublished?354816u:353792u),1024u);
if(!residualPublished){
if(active){
auto scaled=factors_[4096u];ZqB<Zq5::S_V>();
Muls(scaled,output_[256u],qrInverseScale_,256u);
ZqB<Zq5::V_MTE3>();DataCopy(xs[currentRhs_*256u],scaled,256u);ZqB<Zq5::MTE3_S>();
}


Zq1<PIPE_ALL>();SyncAll<true>();
}
const Zq2 row=(GetBlockIdx()%8u)*32u;
auto packedInput=block_[16384u],frameRows=block_[16640u];
ZqB<Zq5::V_MTE2>();DataCopy(packedInput,xs[row],DataCopyParams{4u,4u,28u,0u});ZqB<Zq5::MTE2_V>();
Gather(frameRows,packedInput,panel_[512u].template Zq6<Zq2>(),0u,512u);
GlobalTensor<float>input;input.SetGlobalBuffer(w_+shared+344576u+row*16u,512u);
ZqB<Zq5::V_MTE3>();DataCopy(input,frameRows,512u);ZqB<Zq5::MTE3_S>();

Zq1<PIPE_ALL>();SyncAll<true>();
}else{
if(active&&!(ColumnFactored256&&qrLogicalN_==256u&&residualPublished)){
auto scaled=factors_[4096u],scatter=block_[16384u];
ZqB<Zq5::S_V>();Muls(scaled,output_[256u],qrInverseScale_,256u);Zq1<PIPE_V>();
Brcb(scatter,scaled,Zq4(32u),Zq9{1u,8u});Zq1<PIPE_V>();
GlobalTensor<float>input;input.SetGlobalBuffer(w_+shared+344576u+currentRhs_,4096u-currentRhs_);
ZqB<Zq5::V_MTE3>();DataCopyPad(input,scatter,Zq8{256u,4u,0u,60u,0u});ZqB<Zq5::MTE3_S>();
}
Zq1<PIPE_ALL>();SyncAll<true>();
}
CrossCoreSetFlag<2,PIPE_MTE3>(0);
if constexpr(ColumnFactored256){if(prepareEmit&&qrLogicalN_==256u)PrepareEmitIndex256();}
if constexpr(CompactConsumer200){
if(prepareResidualMap&&compactQr32_&&dual200_&&qrLogicalN_==200u){
auto ids=ProducerResidualOffsets16().template Zq6<int32_t>();ZqB<Zq5::V_S>();
CreateVecIndex<int32_t>(ids,int32_t(0),Zq2(32));Zq1<PIPE_V>();
Muls<int32_t>(ids,ids,int32_t(832),Zq2(32));Zq1<PIPE_V>();
for(Zq2 f=1u;f<8u;++f)Adds<int32_t>(ids[f*32u],ids,int32_t(f*4u),Zq2(32));
Zq1<PIPE_V>();

if(GetSubBlockIdx()==0u){
auto ids40=panel_[1024u].template Zq6<int32_t>();
CreateVecIndex<int32_t>(ids40,int32_t(0),Zq2(40));Zq1<PIPE_V>();
Muls<int32_t>(ids40,ids40,int32_t(832),Zq2(40));Zq1<PIPE_V>();
for(Zq2 f=1u;f<8u;++f)Adds<int32_t>(ids40[f*40u],ids40,int32_t(f*4u),Zq2(40));
Zq1<PIPE_V>();
}
ZqB<Zq5::S_V>();
}
}
CrossCoreWaitFlag(1);
Zq1<PIPE_ALL>();SyncAll<true>();
{

if constexpr(!ColumnFactored256){CrossCoreSetFlag<2,PIPE_MTE3>(0);CrossCoreWaitFlag(1);Zq1<PIPE_ALL>();
if constexpr(CompactConsumer200){if(!finalFrames)SyncAll<true>();}else{SyncAll<true>();}}
}
if(active&&!finalFrames){
GlobalTensor<float>output;
ZqB<Zq5::V_MTE2>();
if(qrLogicalN_==200u){
output.SetGlobalBuffer(w_+shared+344576u,3200u);DataCopy(block_,output,3200u);
}else{
if constexpr(EnableBoundedH3){
if(qrLogicalN_==256u){
output.SetGlobalBuffer(w_+shared+348672u,4096u);DataCopy(block_,output,4096u);
}else{
output.SetGlobalBuffer(w_+shared+348672u+currentRhs_,4096u-currentRhs_);
DataCopyPad(block_,output,Zq8{uint16_t(qrLogicalN_),4u,60u,0u,0u},DataCopyPadExtParams<float>{false,0u,0u,0.0f});
}
}else{
output.SetGlobalBuffer(w_+shared+348672u+currentRhs_,4096u-currentRhs_);
DataCopyPad(block_,output,Zq8{uint16_t(qrLogicalN_),4u,60u,0u,0u},DataCopyPadExtParams<float>{false,0u,0u,0.0f});
}
}
ZqB<Zq5::MTE2_V>();Gather(solveY_,block_,output_[1792u].template Zq6<Zq2>(),0u,qrLogicalN_);Zq1<PIPE_V>();
if(qrLogicalN_==200u){Duplicate(solveY_[200u],0.0f,56u);Zq1<PIPE_V>();}

}
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline float KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::TriangularDot(Zq3 base,Zq2 row)
{
return TriangularSolveOperations::TriangularDot(*this,base,row,N(),VectorPitch(),FactorPitch(),false);
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::ChainVectorTriangular(Zq3 base)
{
constexpr Zq2 n=256u,pitch=256u;
auto seeds=block_;auto indices=block_[2048u].template Zq6<Zq2>();
auto zeros=block_[2304u].template Zq6<Zq2>();
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<256u;++i)indices.SetValue(i,i<n?i*32u:0u);
for(Zq2 i=0;i<8u;++i)zeros.SetValue(i,0u);
ZqB<Zq5::S_V>();ResetMask();SetMaskNorm();
Gather(solveY_,pathLocalRhs_?output_[256u]:output_[512u],output_[768u].template Zq6<Zq2>(),0u,256u);
Zq1<PIPE_V>();
GlobalTensor<float>columns;columns.SetGlobalBuffer(w_+OffA(chainShared_?chainSharedBase_:base),65536u);
for(Zq2 slot=0;slot<2u;++slot){
ZqB<Zq5::V_MTE2>(slot);DataCopy(block_[4096u+slot*4096u],columns[slot*16u*n],4096u);SetFlag<Zq5::MTE2_V>(slot);
}
for(Zq2 begin=0;begin<n;begin+=16u){
const Zq2 slot=(begin/16u)%2u;auto tile=block_[4096u+slot*4096u];
WaitFlag<Zq5::MTE2_V>(slot);
for(Zq2 row=begin;row<begin+16u;++row){
Gather(seeds[row*8u],solveY_,zeros,row*4u,8u);Zq1<PIPE_V>();
if(row+1u<n){
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,n);
MulAddDst<float,float,false>(solveY_,tile[(row-begin)*n],seeds[row*8u],Zq3(0),Zq4(1),Zq0{1,1,0,8,8,0});
Zq1<PIPE_V>();ResetMask();SetMaskNorm();
}
}
if(begin+32u<n){
ZqB<Zq5::V_MTE2>(slot);DataCopy(tile,columns[(begin+32u)*n],4096u);SetFlag<Zq5::MTE2_V>(slot);
}
}
Gather(solveY_,seeds,indices,0u,n);Zq1<PIPE_V>();
for(Zq2 slot=0;slot<2u;++slot){
ZqB<Zq5::V_MTE2>(slot);DataCopy(block_[4096u+slot*4096u],columns[(240u-slot*16u)*n],4096u);SetFlag<Zq5::MTE2_V>(slot);
}
for(int32_t begin=240;begin>=0;begin-=16){
const Zq2 slot=((240u-Zq2(begin))/16u)%2u;auto tile=block_[4096u+slot*4096u];
WaitFlag<Zq5::MTE2_V>(slot);
for(int32_t r=begin+15;r>=begin;--r){const Zq2 row=Zq2(r);
Gather(seeds[row*8u],solveY_,zeros,row*4u,8u);Zq1<PIPE_V>();
Mul(seeds[row*8u],seeds[row*8u],factors_[4096u+row*8u],8u);Zq1<PIPE_V>();
if(row>0u){
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,n);
MulAddDst<float,float,false>(solveY_,tile[(row-Zq2(begin))*n],seeds[row*8u],Zq3(0),Zq4(1),Zq0{1,1,0,8,8,0});
Zq1<PIPE_V>();ResetMask();SetMaskNorm();
}
}
if(begin>=32){
ZqB<Zq5::V_MTE2>(slot);DataCopy(tile,columns[Zq2(begin-32)*n],4096u);SetFlag<Zq5::MTE2_V>(slot);
}
}
Gather(solveY_,seeds,indices,0u,n);Zq1<PIPE_V>();
Adds(pathLocalRhs_?output_[256u]:output_,solveY_,0.0f,256u);
Zq1<PIPE_V>();pathHasIterate_=true;
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::ColumnGmTriangular256(Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth,Zq2 width)
{
constexpr Zq2 n=256u,p=256u;
GlobalTensor<float>columns;columns.SetGlobalBuffer(w_+OffA(chainShared_?chainSharedBase_:base),n*n);
PrepareChainCaches(base);
{if(width==1u&&chainSafe_){ChainVectorTriangular(base);return;}}
ZqB<Zq5::S_V>();Duplicate(solveY_,0.0f,width*p);ZqB<Zq5::V_S>();
{
if(pathLocalRhs_||chainCachesReady_){
ZqB<Zq5::S_V>();Gather(solveY_,pathLocalRhs_?output_[256u]:output_[512u],output_[768u].template Zq6<Zq2>(),0u,256u);ZqB<Zq5::V_S>();
}else{
for(Zq2 row=0;row<n;++row)solveY_.SetValue(row,in[Zq3(row)*inWidth]);
}
}
if(!chainCachesReady_){
for(Zq2 row=0;row<n;++row){
const Zq2 pick=Zq2(int32_t(w_[OffP(base)+row]));
if(pick!=row)for(Zq2 col=0;col<width;++col){
auto y=solveY_[col*p];const float value=y.GetValue(row);
y.SetValue(row,y.GetValue(pick));y.SetValue(pick,value);
}
}
}


ColumnTriangularSolveOperations::ForwardColumnGm256(*this, columns, width);
ColumnTriangularSolveOperations::BackwardColumnGm256<true>(*this, columns, base, width);
{
ZqB<Zq5::S_V>();
Adds(pathLocalRhs_?output_[256u]:output_,solveY_,0.0f,256u);
Zq1<PIPE_V>();pathHasIterate_=true;
}
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::SolveInPlaceWithFactors(Zq3 base,__gm__ float*rhs,Zq2 width)
{

if(UseVectorSolve()&&width<=4u){
VectorTriangularBatch(base,rhs,width,rhs,width,width);
return;
}
{
for(Zq2 col=0;col<width;++col)
VectorTriangularColumn(base,rhs+col,width,rhs+col,width);
return;
}


}
