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
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t,TPipe*externalPipe,bool compactQr32,bool dual200)
{
    compactQr32_ = compactQr32;
    dual200_ = dual200;
    pipe_ = externalPipe;
    FixedWorkspace::InitSquare<StaticN, StaticK, CompactConsumer200>(*this, *pipe_, t, compactQr32);
    FixedWorkspace::InitResidualPlanes(vr_, vectorPitch_, ResidualMaskPlane(), ResidualZeroPlane());
    FixedWorkspace::BindAddresses(*this, a, b, x, res, workspace);
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::QrCoopCacheB(Zq3 shared)
{
GlobalTensor<float>src;src.SetGlobalBuffer(w_+shared+352768u,768u);
ZqB<Zq5::V_MTE2>();
if constexpr(ColumnFactored256){
const Zq2 row=(GetBlockIdx()%8u)*32u;
for(Zq2 rhs=0u;rhs<3u;++rhs)DataCopy(output_[768u+rhs*256u+row],src[rhs*256u+row],32u);
}else{for(Zq2 rhs=0u;rhs<3u;++rhs)DataCopy(output_[768u+rhs*256u],src[rhs*256u],256u);}
ZqB<Zq5::MTE2_V>();
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::PrepareChainCaches(Zq3 base)
{
if(chainCachesReady_)return;
const Zq3 factorBase=chainShared_?chainSharedBase_:base;
auto map=output_[768u].template Zq6<Zq2>();
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<256u;++i){map.SetValue(i,i*4u);output_[512u].SetValue(i,i<N()?w_[OffB(base)+i]:0.0f);output_[1024u].SetValue(i,i<N()?w_[OffInv(factorBase)+i]:0.0f);}
for(Zq2 i=0;i<N();++i){const Zq2 pick=Zq2(int32_t(w_[OffP(factorBase)+i]));const Zq2 x=map.GetValue(i);map.SetValue(i,map.GetValue(pick));map.SetValue(pick,x);}
ZqB<Zq5::S_V>();
chainSafe_=true;
for(Zq2 i=0;i<256u;++i)chainSafe_=chainSafe_&&output_[1024u].GetValue(i)!=0.0f;


if(chainSafe_){ZqB<Zq5::S_V>();Brcb(factors_[4096u],output_[1024u],Zq4(32),Zq9{1,8});Zq1<PIPE_V>();}
chainCachesReady_=true;
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::Load(Zq3 base,Zq2 batch)
{
const Zq2 m=M(),n=N();
currentBatch_=batch;pathLocalRhs_=false;pathHasIterate_=false;chainCachesReady_=false;chainSafe_=false;chainShared_=false;

const Zq3 aBatch=Zq3(batch)*m*n;
const Zq3 dstL=OffLU(base),dstB=OffB(base);
{
WorkspaceOperations::CopySquareMatrix256(block_,a_+aBatch,w_+dstL);
}
{
for(Zq2 row=0;row<m;++row)
w_[dstB+row]=b_[(Zq3(batch)*m+row)*RhsWidth()+currentRhs_];
}
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::PrepareEmitIndex256()
{
auto ids=panel_[256u].template Zq6<int32_t>();
auto rows=block_[640u].template Zq6<int32_t>();
auto seq=block_[768u].template Zq6<int32_t>();
CreateVecIndex<int32_t>(seq,int32_t(0),Zq2(96));Zq1<PIPE_V>();
Muls<int32_t>(rows,seq,int32_t(43),Zq2(96));Zq1<PIPE_V>();
ShiftRight<int32_t>(rows,rows,int32_t(7),Zq2(96));Zq1<PIPE_V>();
Muls<int32_t>(ids,seq,int32_t(128),Zq2(96));
auto deltaIds=panel_[512u].template Zq6<int32_t>();auto tmp=block_[896u].template Zq6<int32_t>();
Muls<int32_t>(deltaIds,rows,int32_t(52),Zq2(96));
Muls<int32_t>(tmp,seq,int32_t(4),Zq2(96));Zq1<PIPE_V>();
Add<int32_t>(deltaIds,deltaIds,tmp,Zq2(96));
Muls<int32_t>(rows,rows,int32_t(-380),Zq2(96));Zq1<PIPE_V>();
Add<int32_t>(ids,ids,rows,Zq2(96));Zq1<PIPE_V>();
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::EmitQrCoalesced256(Zq2 batch,bool active)
{
GlobalTensor<float>xs;xs.SetGlobalBuffer(w_+Zq3(batch)*397824u+353792u,768u);
if(active){ZqB<Zq5::V_MTE3>();DataCopy(xs[currentRhs_*256u],solveY_,256u);ZqB<Zq5::MTE3_S>();}
Zq1<PIPE_ALL>();SyncAll<true>();
const Zq2 row=(GetBlockIdx()%8u)*32u;
ZqB<Zq5::V_MTE2>();
for(Zq2 rhs=0u;rhs<3u;++rhs)DataCopy(block_[rhs*32u],xs[rhs*256u+row],32u);
Zq7<Zq2>emitIndices;
if constexpr(ColumnFactored256){emitIndices=panel_[256u].template Zq6<Zq2>();}
else{
auto ids=block_[512u].template Zq6<int32_t>();auto rows=block_[640u].template Zq6<int32_t>();
auto seq=block_[768u].template Zq6<int32_t>();auto tmp=block_[896u];
CreateVecIndex<int32_t>(seq,int32_t(0),Zq2(96));Zq1<PIPE_V>();
Cast(tmp,seq,RoundMode::CAST_NONE,96u);Zq1<PIPE_V>();Muls(tmp,tmp,0.3333333432674407958984375f,96u);Zq1<PIPE_V>();
Cast(rows,tmp,RoundMode::CAST_FLOOR,96u);Zq1<PIPE_V>();
Muls<int32_t>(ids,rows,int32_t(3),96u);Zq1<PIPE_V>();Sub<int32_t>(ids,seq,ids,96u);Zq1<PIPE_V>();
Muls<int32_t>(ids,ids,int32_t(32),96u);Zq1<PIPE_V>();Add<int32_t>(ids,ids,rows,96u);Zq1<PIPE_V>();Muls<int32_t>(ids,ids,int32_t(4),96u);Zq1<PIPE_V>();
emitIndices=ids.template Zq6<Zq2>();
}
ZqB<Zq5::MTE2_V>();Gather(block_[256u],block_,emitIndices,0u,96u);
ZqB<Zq5::V_MTE3>();GlobalTensor<float>gx;gx.SetGlobalBuffer(x_+Zq3(batch)*768u+row*3u,96u);DataCopy(gx,block_[256u],96u);
const Zq3 index=Zq3(batch)*3u+currentRhs_;
if(active&&r_!=nullptr&&index<tiling_.residualElements){
ZqB<Zq5::V_S>();block_.SetValue(384u,0.0f);ZqB<Zq5::S_MTE3>();
GlobalTensor<float>gr;gr.SetGlobalBuffer(r_+index,1u);DataCopyPad(gr,block_[384u],Zq8{1u,4u,0u,0u,0u});}
ZqB<Zq5::MTE3_S>();
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::EmitQrFinalFrames256(Zq2 batch,bool active)
{
const Zq3 shared=Zq3(batch)*397824u;
const Zq2 row=(GetBlockIdx()%8u)*32u;
GlobalTensor<float>oldX,delta;oldX.SetGlobalBuffer(w_+shared+353792u,768u);
delta.SetGlobalBuffer(w_+shared+348672u+row*16u,512u);
ZqB<Zq5::V_MTE2>();
for(Zq2 rhs=0u;rhs<3u;++rhs)DataCopy(block_[rhs*32u],oldX[rhs*256u+row],32u);
DataCopy(block_[1024u],delta,512u);ZqB<Zq5::MTE2_V>();
Gather(block_[256u],block_,panel_[256u].template Zq6<Zq2>(),0u,96u);
Gather(block_[384u],block_[1024u],panel_[512u].template Zq6<Zq2>(),0u,96u);Zq1<PIPE_V>();
Add(block_[256u],block_[256u],block_[384u],96u);Zq1<PIPE_V>();
Adds(block_[256u],block_[256u],0.0f,96u);ZqB<Zq5::V_MTE3>();
GlobalTensor<float>gx;gx.SetGlobalBuffer(x_+Zq3(batch)*768u+row*3u,96u);DataCopy(gx,block_[256u],96u);
const Zq3 index=Zq3(batch)*3u+currentRhs_;
if(active&&r_!=nullptr&&index<tiling_.residualElements){
ZqB<Zq5::V_S>();block_.SetValue(512u,0.0f);ZqB<Zq5::S_MTE3>();
GlobalTensor<float>gr;gr.SetGlobalBuffer(r_+index,1u);DataCopyPad(gr,block_[512u],Zq8{1u,4u,0u,0u,0u});
}
ZqB<Zq5::MTE3_S>();
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::EmitQrFinalFrames200(Zq2 batch,bool active)
{
const Zq3 shared=Zq3(batch)*397824u;
const Zq2 row=(GetBlockIdx()%8u)*32u;
if(row<200u){
const Zq2 count=200u-row<32u?200u-row:32u;
GlobalTensor<float>oldX,delta;
oldX.SetGlobalBuffer(w_+shared+353792u,1024u);
delta.SetGlobalBuffer(w_+shared+344576u+row*16u,count*16u);
ZqB<Zq5::V_MTE2>();
DataCopy(block_,oldX[row],DataCopyParams{4u,uint16_t(count/8u),uint16_t(32u-count/8u),uint16_t(4u-count/8u)});
DataCopy(block_[1024u],delta,count*16u);
ZqB<Zq5::MTE2_V>();
Gather(block_[1536u],block_,panel_[512u].template Zq6<Zq2>(),0u,count*16u);Zq1<PIPE_V>();
Add(block_[1536u],block_[1536u],block_[1024u],count*16u);Zq1<PIPE_V>();
Adds(block_[1536u],block_[1536u],0.0f,count*16u);
ZqB<Zq5::V_MTE3>();
GlobalTensor<float>gx;gx.SetGlobalBuffer(x_+Zq3(batch)*800u+row*4u,count*4u);
DataCopyPad(gx,block_[1536u],Zq8{uint16_t(count),16u,1u,0u,0u});
}
WorkspaceOperations::EmitZeroResidual200(block_,r_,batch,currentRhs_,tiling_.residualElements,active);
ZqB<Zq5::MTE3_S>();
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::EmitQrCoalesced200(Zq2 batch,bool active)
{
GlobalTensor<float>xs;xs.SetGlobalBuffer(w_+Zq3(batch)*397824u+353792u,1024u);
if(active){ZqB<Zq5::V_MTE3>();DataCopy(xs[currentRhs_*256u],solveY_,256u);ZqB<Zq5::MTE3_S>();}
Zq1<PIPE_ALL>();SyncAll<true>();
const Zq2 row=(GetBlockIdx()%8u)*32u;
if(row<200u){
const Zq2 count=200u-row<32u?200u-row:32u;
ZqB<Zq5::V_MTE2>();
for(Zq2 rhs=0u;rhs<4u;++rhs)DataCopy(block_[rhs*32u],xs[rhs*256u+row],count);
auto ids=block_[512u].template Zq6<int32_t>();
if constexpr(CompactConsumer200){
auto rows=block_[640u].template Zq6<int32_t>();
auto seq=block_[768u].template Zq6<int32_t>();
CreateVecIndex<int32_t>(seq,int32_t(0),Zq2(128));Zq1<PIPE_V>();
ShiftRight<int32_t>(rows,seq,int32_t(2),Zq2(128));Zq1<PIPE_V>();
Muls<int32_t>(ids,seq,int32_t(128),Zq2(128));
Muls<int32_t>(rows,rows,int32_t(-508),Zq2(128));Zq1<PIPE_V>();
Add<int32_t>(ids,ids,rows,Zq2(128));Zq1<PIPE_V>();
}else{

auto rows=block_[640u].template Zq6<int32_t>();
auto seq=block_[768u].template Zq6<int32_t>();auto tmp=block_[896u];
CreateVecIndex<int32_t>(seq,int32_t(0),Zq2(128));Zq1<PIPE_V>();
ShiftRight<int32_t>(rows,seq,int32_t(2),Zq2(128));Zq1<PIPE_V>();
Muls<int32_t>(ids,rows,int32_t(4),128u);Zq1<PIPE_V>();Sub<int32_t>(ids,seq,ids,128u);Zq1<PIPE_V>();
Muls<int32_t>(ids,ids,int32_t(32),128u);Zq1<PIPE_V>();Add<int32_t>(ids,ids,rows,128u);Zq1<PIPE_V>();
Muls<int32_t>(ids,ids,int32_t(4),128u);Zq1<PIPE_V>();
}
ZqB<Zq5::MTE2_V>();Gather(block_[256u],block_,ids.template Zq6<Zq2>(),0u,count*4u);
ZqB<Zq5::V_MTE3>();GlobalTensor<float>gx;gx.SetGlobalBuffer(x_+Zq3(batch)*800u+row*4u,count*4u);
DataCopyPad(gx,block_[256u],Zq8{1u,count*16u,0u,0u,0u});
}
WorkspaceOperations::EmitZeroResidual200(block_,r_,batch,currentRhs_,tiling_.residualElements,active);
ZqB<Zq5::MTE3_S>();
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::EmitQrVectorDma(Zq2 batch)
{
const Zq2 n=qrLogicalN_,k=qrRhsWidth_,rhs=currentRhs_;
ZqB<Zq5::S_V>();Brcb(output_,solveY_,Zq4(n/8u),Zq9{1u,8u});
output_.SetValue(n*8u,0.0f);ZqB<Zq5::V_MTE3>();ZqB<Zq5::S_MTE3>();
GlobalTensor<float>gx;gx.SetGlobalBuffer(x_+Zq3(batch)*n*k+rhs,n*k-rhs);
DataCopyPad(gx,output_,Zq8{uint16_t(n),4u,0u,(k-1u)*4u,0u});
const Zq3 index=Zq3(batch)*k+rhs;
if(r_!=nullptr&&index<tiling_.residualElements){GlobalTensor<float>gr;gr.SetGlobalBuffer(r_+index,1u);
DataCopyPad(gr,output_[n*8u],Zq8{1u,4u,0u,0u,0u});}
ZqB<Zq5::MTE3_S>();
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::EmitSplitRhsDma(Zq3 base,Zq2 batch)
{
const Zq2 n=N();
{for(Zq2 row=0;row<n;++row)output_.SetValue(row*8u,solveY_.GetValue(row));}
output_.SetValue(n*8u,0.0f);
ZqB<Zq5::S_MTE3>();
WorkspaceOperations::CopySplitRhsOutput(*this, batch, n);
}
