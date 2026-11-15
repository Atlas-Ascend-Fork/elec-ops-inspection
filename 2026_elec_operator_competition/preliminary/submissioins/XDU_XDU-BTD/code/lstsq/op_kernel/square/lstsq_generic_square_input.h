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

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t)
{
static_assert(StaticN==0u||(StaticN==200u&&StaticK==4u)||
(StaticN==256u&&StaticK==3u)||((StaticN==50u||StaticN==64u)&&StaticK==2u),
"Unsupported fixed shape");
if constexpr(StaticN){t.m=StaticN;t.n=StaticN;t.k=StaticK;}
if constexpr(StaticN==50u){pipe_.InitBuffer(producerABuf_,4096u*sizeof(float));producerA_=producerABuf_.template Get<float>();}
rhsWidth_=t.k;
splitRhs_=t.m==t.n&&((t.n==200u&&t.k==4u)||(t.n==256u&&t.k==3u)||((t.n==50u||t.n==64u)&&t.k==2u));
if(splitRhs_)t.k=1u;
tiling_=t;
dmaOutputs_=t.m==t.n&&Zq3(t.n)*t.k<=1024u;
pipe_.InitBuffer(outputBuf_,splitRhs_?(t.n*8u+8u)*sizeof(float):
(dmaOutputs_?((t.n*t.k+7u)/8u*8u+(t.k+7u)/8u*8u)*sizeof(float):32u));
output_=outputBuf_.template Get<float>();
localFactors_=t.m==t.n&&t.n>=32u&&t.n<=200u;
columnLocal_=localFactors_&&(StaticN==50u||StaticN==64u||t.n>=128u||((t.n==50u||t.n==64u)&&t.k==2u));
factorPitch_=(t.n+7u)/8u*8u;
pipe_.InitBuffer(factorBuf_,localFactors_?(StaticN==50u?56u*56u:t.n*(StaticN==64u?72u:factorPitch_))*sizeof(float):
(t.m==t.n&&t.n==256u?32u*256u*sizeof(float):32u));
factors_=factorBuf_.template Get<float>();
pipe_.InitBuffer(rowBuf_,208u*sizeof(float));
rows_=rowBuf_.template Get<float>();
pipe_.InitBuffer(blockBuf_,t.m==t.n&&t.n==256u?(16u*256u+64u+2u*(t.k==1u?48u:32u)*264u)*sizeof(float):(columnLocal_?(8u*factorPitch_+(StaticN==50u?64u:0u))*sizeof(float):32u));
block_=blockBuf_.template Get<float>();
pipe_.InitBuffer(panelBuf_,t.m==t.n&&t.n==256u?256u*8u*sizeof(float):
((StaticN==50u||StaticN==64u||((t.n==50u||t.n==64u)&&t.k==2u))?(StaticN==50u?15u*2048u+512u:17u*512u)*sizeof(float):32u));
panel_=panelBuf_.template Get<float>();
pipe_.InitBuffer(residualTileBuf_,t.m==t.n&&t.n==256u?2u*8u*256u*4u:(columnLocal_?8u*(StaticN==50u?56u:t.n)*4u:32u));
residualTile_=residualTileBuf_.template Get<Zq2>();
vectorResidual_=t.m==t.n;
vectorPitch_=8u;
while(vectorPitch_<t.n)vectorPitch_*=2u;
pipe_.InitBuffer(residualBuf_,vectorResidual_?(t.n==256u?10u:13u)*vectorPitch_*sizeof(float):32u);
vr_=residualBuf_.template Get<float>();
vectorSolve_=t.m==t.n&&((t.n>=32u&&t.n<=200u)||(t.n==256u&&t.k<=4u));
pipe_.InitBuffer(solveBuf_,vectorSolve_?(t.k<=4u?t.k:(columnLocal_?4u:1u))*vectorPitch_*sizeof(float):32u);
solveY_=solveBuf_.template Get<float>();
if constexpr(StaticN==50u||StaticN==64u){
pipe_.InitBuffer(negative64Buf_,(StaticN==50u?4096u+192u:4096u)*sizeof(float));
negative64_=negative64Buf_.template Get<float>();
}
if(vectorResidual_)WorkspaceOperations::InitializeVectorResidual(*this, vectorPitch_);
WorkspaceOperations::InitializeGlobalBuffers(*this,a,b,x,res,workspace);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::Load(Zq3 base,Zq2 batch)
{
const Zq2 m=M(),n=N(),k=K();
augmented50Ready_=false;augmented50RhsReady_=false;currentBatch_=batch;pathLocalRhs_=false;pathHasIterate_=false;if constexpr(StaticN==50u){producerPrivate_=false;producerCorrection_=false;replicaOffsetsReady_=false;}
if constexpr(StaticN==50u){diagonalKnown_=false;diagonalSafe_=false;}

if constexpr(StaticN==50u)negative64Ready_=false;
const Zq3 aBatch=Zq3(batch)*m*n;
const Zq3 bBatch=Zq3(batch)*m*k;
const Zq3 dstA=OffA(base),dstL=OffLU(base),dstB=OffB(base);
if constexpr(StaticN==50u){LoadRows50(batch);}
else {if(UseColumnLocal()&&n==200u){
LoadRows200(base,batch);
}else if(UseColumnLocal()){
WorkspaceOperations::LoadColumnFactors(*this, a_+aBatch, n);
}else if(UseLocalFactors()){

GlobalTensor<float>input;input.SetGlobalBuffer(a_+aBatch,Zq3(n)*n);
DataCopyPad(factors_,input,Zq8{uint16_t(n),n*4u,0u,0u,0u},
DataCopyPadExtParams<float>{true,0u,Zq4(FactorPitch()-n),0.0f});
ZqB<Zq5::MTE2_S>();
}else if(m==n&&n==256u){
WorkspaceOperations::CopySquareMatrix256(block_,a_+aBatch,w_+dstL);
}else if(m==n){
for(Zq3 i=0;i<Zq3(m)*n;++i)w_[dstL+i]=a_[aBatch+i];
}else{
for(Zq3 i=0;i<Zq3(m)*n;++i)w_[dstL+i]=w_[dstA+i]=a_[aBatch+i];
}}
if(UseSplitRhs()){
if constexpr(StaticN==50u&&StaticK==2u){


auto rhsOffsets=rows_.template Zq6<int32_t>();
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
Duplicate(block_,0.0f,Zq2(128));
CreateVecIndex<int32_t>(rhsOffsets,int32_t(0),Zq2(64));Zq1<PIPE_V>();
Muls<int32_t>(rhsOffsets,rhsOffsets,int32_t(8),Zq2(64));Zq1<PIPE_V>();
Adds<int32_t>(rhsOffsets,rhsOffsets,int32_t(currentRhs_*4u),Zq2(64));Zq1<PIPE_V>();
ZqB<Zq5::V_MTE2>();
GlobalTensor<float>packedRhs;packedRhs.SetGlobalBuffer(b_+Zq3(batch)*100u,100u);
DataCopyPad(block_,packedRhs,Zq8{1u,400u,0u,0u,0u},DataCopyPadExtParams<float>{true,0u,4u,0.0f});
ZqB<Zq5::MTE2_V>();
Gather(negative64_[4224u],block_,rhsOffsets.template Zq6<Zq2>(),0u,64u);Zq1<PIPE_V>();


}else{
for(Zq2 row=0;row<m;++row)
w_[dstB+row]=b_[(Zq3(batch)*m+row)*RhsWidth()+currentRhs_];
}
}else for(Zq3 i=0;i<Zq3(m)*k;++i)w_[dstB+i]=b_[bBatch+i];
}


template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::LoadBlockedColumns200(Zq3 base)
{
constexpr Zq2 n=200u;
WorkspaceOperations::LoadColumnFactors(*this, w_+OffLU(base), n);
}
