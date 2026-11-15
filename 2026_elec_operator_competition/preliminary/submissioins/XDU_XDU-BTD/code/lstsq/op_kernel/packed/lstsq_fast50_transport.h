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
__aicore__ inline void KernelFastContext50<StaticN, StaticK>::Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t)
{
static_assert(StaticN==0u||(StaticN==200u&&StaticK==4u)||
(StaticN==256u&&StaticK==3u)||((StaticN==50u||StaticN==64u)&&StaticK==2u),
"Unsupported fixed shape");
{t.m=StaticN;t.n=StaticN;t.k=StaticK;}
{
rhsWidth_=t.k;splitRhs_=true;t.k=1u;tiling_=t;
dmaOutputs_=true;localFactors_=true;columnLocal_=true;factorPitch_=56u;
vectorResidual_=true;vectorPitch_=64u;vectorSolve_=true;
// One owned allocation, exact previous50only ranges and logical lengths.
pipe_.InitBuffer(packed50Buf_,180896u);
auto storage=packed50Buf_.template Get<float>();
producerA_=storage;producerA_.SetSize(4096u);
output_=storage[4096u];output_.SetSize(408u);
factors_=storage[4504u];factors_.SetSize(3136u);
rows_=storage[7640u];rows_.SetSize(208u);
block_=storage[7848u];block_.SetSize(512u);
panel_=storage[8360u];panel_.SetSize(31232u);
residualTile_=storage[39592u].template Zq6<Zq2>();residualTile_.SetSize(448u);
vr_=storage[40040u];vr_.SetSize(832u);
solveY_=storage[40872u];solveY_.SetSize(64u);
negative64_=storage[40936u];negative64_.SetSize(4288u);
}
if(vectorResidual_){
auto mask=vr_[ResidualMaskPlane()*vectorPitch_].template Zq6<Zq2>();
Duplicate(mask,Zq2(0xfffff000u),vectorPitch_);

}
a_=reinterpret_cast<__gm__ float*>(a);
b_=reinterpret_cast<__gm__ float*>(b);
x_=reinterpret_cast<__gm__ float*>(x);
r_=reinterpret_cast<__gm__ float*>(res);
const Zq3 user=reinterpret_cast<Zq3>(AscendC::GetUserWorkspace(workspace));
w_=reinterpret_cast<__gm__ float*>((user+63u)&~Zq3(63u));
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline bool KernelFastContext50<StaticN, StaticK>::TryTask50(Zq2 task)
{
static_assert(StaticN==50u&&StaticK==2u,"Fast50only");
currentRhs_=task%2u;currentBatch_=task/2u;
const Zq2 batch=currentBatch_;
const Zq3 base=Zq3(task)*PerBatch();
directRows50_=false;augmented50Ready_=false;augmented50RhsReady_=false;normalizationScale_=1.0f;inverseNormalizationScale_=1.0f;
pathLocalRhs_=false;pathHasIterate_=false;
producerPrivate_=false;producerCorrection_=false;replicaOffsetsReady_=false;
diagonalKnown_=false;diagonalSafe_=false;negative64Ready_=false;
LoadRows50(batch);


auto rhsOffsets=vr_[640u].template Zq6<int32_t>();
auto originalIndex=rows_.template Zq6<int32_t>();
ZqB<Zq5::V_S>();
SetMaskNorm();ResetMask();
Duplicate(block_,0.0f,Zq2(128));
CreateVecIndex<int32_t>(originalIndex,int32_t(0),Zq2(64));Zq1<PIPE_V>();

Adds<int32_t>(originalIndex[64u],originalIndex,int32_t(64),Zq2(64));Zq1<PIPE_V>();
auto residentEmitMap=panel_[30016u].template Zq6<int32_t>();
ShiftRight<int32_t>(residentEmitMap,originalIndex,int32_t(1),int32_t(128));Zq1<PIPE_V>();
Muls<int32_t>(residentEmitMap,residentEmitMap,int32_t(4),Zq2(128));Zq1<PIPE_V>();
Muls<int32_t>(rhsOffsets,originalIndex,int32_t(8),Zq2(64));Zq1<PIPE_V>();
Adds<int32_t>(rhsOffsets,rhsOffsets,int32_t(currentRhs_*4u),Zq2(64));Zq1<PIPE_V>();
ZqB<Zq5::V_MTE2>();
GlobalTensor<float>packedRhs;packedRhs.SetGlobalBuffer(b_+Zq3(batch)*100u,100u);
DataCopyPad(block_,packedRhs,Zq8{1u,400u,0u,0u,0u},DataCopyPadExtParams<float>{true,0u,4u,0.0f});
ZqB<Zq5::MTE2_V>();
Gather(negative64_[4224u],block_,rhsOffsets.template Zq6<Zq2>(),0u,64u);Zq1<PIPE_V>();


if(!BuildAugmented50()){Zq1<PIPE_ALL>();return false;}
producerPrivate_=true;producerCorrection_=false;
ConsumeAugmented50(w_+OffB(base),1u);
RefineAccepted50(base);
EmitSplitRhsDma(base,batch);
return true;
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelFastContext50<StaticN, StaticK>::CanonicalizeRowMajor50()
{
auto aug=panel_[22528u];
auto indices=panel_.template Zq6<int32_t>();
auto rb=rows_.template Zq6<int32_t>();
auto cb=rows_[64u].template Zq6<int32_t>();

Muls<int32_t>(rb,cb,int32_t(72),Zq2(64));Zq1<PIPE_V>();
Brcb(block_.template Zq6<int32_t>(),cb,Zq4(8),Zq9{1,8});Zq1<PIPE_V>();
Add<int32_t>(indices,rb,block_.template Zq6<int32_t>(),Zq3(64),Zq4(64),Zq0{1,1,0,8,0,1});Zq1<PIPE_V>();
Gather(panel_[8192u],aug,indices.template Zq6<Zq2>(),0u,4096u);Zq1<PIPE_V>();

Adds<int32_t>(aug.template Zq6<int32_t>(),panel_[8192u].template Zq6<int32_t>(),int32_t(0),
Zq3(64),Zq4(64),UnaryRepeatParams{1,1,9,8});Zq1<PIPE_V>();
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelFastContext50<StaticN, StaticK>::EmitSplitRhsAtomic50(Zq3 base,Zq2 batch,Zq2 firstRow)
{
const Zq2 tailRow=firstRow==0u?48u:0u;
WorkspaceOperations::PrepareAtomicRhs50(*this, base, firstRow, tailRow);
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
auto indices=(producerPrivate_?panel_[30016u]:panel_).template Zq6<int32_t>();
auto minimum=panel_[128u].template Zq6<int32_t>();
auto maximum=panel_[256u].template Zq6<int32_t>();
if(!producerPrivate_){
CreateVecIndex<int32_t>(indices,int32_t(0),Zq2(128));Zq1<PIPE_V>();
ShiftRight<int32_t>(indices,indices,int32_t(1),int32_t(128));Zq1<PIPE_V>();
Muls<int32_t>(indices,indices,int32_t(4),Zq2(128));Zq1<PIPE_V>();
}

WorkspaceOperations::CopyAtomicRhs50(*this, batch, firstRow, tailRow, indices, minimum, maximum);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelFastContext50<StaticN, StaticK>::EmitSplitRhsDma(Zq3 base,Zq2 batch)
{
if(WorkspaceOperations::TryEmitSplitRhsAtomic50(*this,base,batch))return;
const Zq2 n=N();
{
WorkspaceOperations::PrepareSplitRhsDmaOutput(*this, base, n);
}
ZqB<Zq5::S_MTE3>();
ZqB<Zq5::V_MTE3>();
WorkspaceOperations::CopySplitRhsOutput(*this, batch, n);
}
