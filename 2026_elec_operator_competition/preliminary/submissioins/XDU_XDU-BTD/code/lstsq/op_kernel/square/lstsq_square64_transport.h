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

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::EmitSplitRhsDma(Zq3 base,Zq2 batch)
{
if constexpr (ColdTask) {
    WorkspaceOperations::EmitSplitRhsDma64(*this, base, batch);
} else {
const Zq2 n=N();
ZqB<Zq5::V_S>();
for(Zq2 row=0;row<n;++row)output_.SetValue(row*8u,(StaticN==64u&&producerPrivate_)?negative64_[4096u].GetValue(row):w_[OffX(base)+row]);
output_.SetValue(n*8u,0.0f);
ZqB<Zq5::S_MTE3>();
WorkspaceOperations::CopySplitRhsOutput(*this, batch, n);
}
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::Process()
{
    WorkspaceOperations::ProcessFixedSquare<false>(*this, GetBlockIdx());
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t)
{
    producerA_ = FixedWorkspace::Allocate<float>(pipe_, producerABuf_, 4096u);
    FixedWorkspace::InitSquare<StaticN, StaticK>(*this, pipe_, t);
    negative64_ = FixedWorkspace::Allocate<float>(pipe_, negative64Buf_, 4096u + 192u);
    FixedWorkspace::InitResidualPlanes(vr_, vectorPitch_, ResidualMaskPlane(), ResidualZeroPlane());
    FixedWorkspace::BindAddresses(*this, a, b, x, res, workspace);
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::Load(Zq3 base,Zq2 batch)
{
augmented64Ready_=false;
if constexpr (ColdTask) rowInverse64Ready_=false;currentBatch_=batch;{producerPrivate_=false;producerCorrection_=false;replicaOffsetsReady_=false;}

{diagonalKnown_=false;diagonalSafe_=false;}

{LoadRows64(batch);}
{
GlobalTensor<float>inputB;inputB.SetGlobalBuffer(b_+Zq3(batch)*128u,128u);
ZqB<Zq5::V_MTE2>();DataCopy(panel_,inputB,128u);ZqB<Zq5::MTE2_V>();
auto bo=vr_[640u].template Zq6<Zq2>();ZqB<Zq5::V_S>();
for(Zq2 i=0;i<64u;++i)bo.SetValue(i,(i*2u+currentRhs_)*4u);
ZqB<Zq5::S_V>();Gather(negative64_[4224u],panel_,bo,0u,64u);ZqB<Zq5::V_S>();
}
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::LoadRows64(Zq2 batch)
{
negative64Ready_=false;
GlobalTensor<float>input;input.SetGlobalBuffer(a_+Zq3(batch)*4096u,4096u);


Zq1<PIPE_V>();ZqB<Zq5::V_MTE2>();
DataCopyPad(producerA_,input,Zq8{64u,256u,0u,0u,0u},DataCopyPadExtParams<float>{true,0u,0u,0.0f});
ZqB<Zq5::MTE2_V>();

ZqB<Zq5::V_S>();
if constexpr (!ColdTask) {
auto offsets=vr_[10u*64u].template Zq6<Zq2>();
for(Zq2 row=0;row<64u;++row)offsets.SetValue(row,row*72u*4u);
}
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::EmitSplitRhsAtomic64(Zq3 base,Zq2 batch)
{
if(!producerPrivate_){


ZqB<Zq5::V_S>();
for(Zq2 row=0u;row<64u;++row)negative64_.SetValue(4096u+row,GetX(base,row,0u));
ZqB<Zq5::S_V>();
}
SetMaskNorm();ResetMask();
auto indices=panel_.template Zq6<int32_t>();
auto minimum=panel_[128u].template Zq6<int32_t>();
auto maximum=panel_[256u].template Zq6<int32_t>();
CreateVecIndex<int32_t>(indices,int32_t(0),Zq2(128));Zq1<PIPE_V>();
ShiftRight<int32_t>(indices,indices,int32_t(1),int32_t(128));Zq1<PIPE_V>();
Muls<int32_t>(indices,indices,int32_t(4),Zq2(128));Zq1<PIPE_V>();
WorkspaceOperations::CopyAtomicRhs64(*this, batch, indices, minimum, maximum);
}
