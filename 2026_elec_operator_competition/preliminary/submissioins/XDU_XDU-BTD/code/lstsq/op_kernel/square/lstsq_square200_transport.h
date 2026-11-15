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
__aicore__ inline void KernelLarge200<StaticN, StaticK>::EmitSplitRhsDma(Zq3 base,Zq2 batch)
{
const Zq2 n=N();
{
ZqB<Zq5::V_S>();
for(Zq2 row=0;row<n;++row)output_.SetValue(row*8u,solveY_.GetValue(row));
}
output_.SetValue(n*8u,0.0f);
ZqB<Zq5::S_MTE3>();

WorkspaceOperations::CopySplitRhsOutput(*this, batch, n);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge200<StaticN, StaticK>::Process(bool dual200)
{
    WorkspaceOperations::ProcessFixedSquare<false>(*this, dual200 ? GetBlockIdx()/2u : GetBlockIdx());
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge200<StaticN, StaticK>::ArchiveRows200(Zq3 base)
{
constexpr Zq2 n=200u,p=256u;


auto rowBytes=vr_.template Zq6<int32_t>();
auto colBytes=vr_[p].template Zq6<int32_t>();
auto offsets=residualTile_.template Zq6<int32_t>();
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<32u;++i)rowBytes.SetValue(i,int32_t(i*n*4u));
for(Zq2 col=0;col<n;++col)colBytes.SetValue(col,int32_t(col*4u));
ZqB<Zq5::S_V>();ResetMask();SetMaskNorm();
for(Zq2 part=0;part<32u;part+=8u)
Brcb(offsets[part],colBytes,Zq4(6),Zq9{4,32});
Zq1<PIPE_V>();
Add<int32_t>(offsets,offsets,rowBytes,Zq3(32),Zq4(48),Zq0{1,1,1,4,4,0});Zq1<PIPE_V>();
GlobalTensor<float>archive;archive.SetGlobalBuffer(w_+OffA(base),40000u);
DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(archive);
for(Zq2 row=0;row+32u<=n;row+=32u){
for(Zq2 col=0;col<n;col+=48u){
const Zq2 cols=n-col<48u?n-col:48u;
ZqB<Zq5::MTE3_V>();
Gather(block_,factors_[row*n+col],offsets.template Zq6<Zq2>(),0u,cols*32u);
ZqB<Zq5::V_MTE3>();
DataCopyPad(archive[col*n+row],block_,Zq8{uint16_t(cols),128u,0u,672u,0u});
}
}


Zq1<PIPE_V>();
Brcb(offsets,colBytes,Zq4(25),Zq9{1,8});Zq1<PIPE_V>();
Add<int32_t>(offsets,offsets,rowBytes,Zq3(64),Zq4(25),Zq0{1,1,0,8,8,0});Zq1<PIPE_V>();
ZqB<Zq5::MTE3_V>();
Gather(block_,factors_[192u*n],offsets.template Zq6<Zq2>(),0u,1600u);
ZqB<Zq5::V_MTE3>();
DataCopyPad(archive[192u],block_,Zq8{uint16_t(n),32u,0u,768u,0u});
ZqB<Zq5::MTE3_MTE2>();ZqB<Zq5::V_MTE2>();
DataCopy(factors_,archive,40000u);ZqB<Zq5::MTE2_V>();ZqB<Zq5::MTE2_S>();

Duplicate(vr_,0.0f,p);Duplicate(vr_[p],0.0f,p);Duplicate(vr_[ResidualZeroPlane()*p],0.0f,p);
ZqB<Zq5::V_S>();
auto cols=vr_[10u*p].template Zq6<Zq2>();
for(Zq2 col=0;col<n;++col)cols.SetValue(col,col*n*4u);
columnArchiveReady_=true;
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge200<StaticN, StaticK>::Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t)
{
    FixedWorkspace::InitSquare<StaticN, StaticK>(*this, pipe_, t);
    FixedWorkspace::InitResidualPlanes(vr_, vectorPitch_, ResidualMaskPlane(), ResidualZeroPlane());
    FixedWorkspace::BindAddresses(*this, a, b, x, res, workspace);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge200<StaticN, StaticK>::PrepareChainCaches(Zq3 base)
{
if(chainCachesReady_)return;
auto map=output_[768u].template Zq6<Zq2>();
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<256u;++i){map.SetValue(i,i*4u);output_[512u].SetValue(i,i<N()?w_[OffB(base)+i]:0.0f);output_[1024u].SetValue(i,i<N()?w_[OffInv(base)+i]:0.0f);}
for(Zq2 i=0;i<N();++i){const Zq2 pick=Zq2(int32_t(w_[OffP(base)+i]));const Zq2 x=map.GetValue(i);map.SetValue(i,map.GetValue(pick));map.SetValue(pick,x);}
ZqB<Zq5::S_V>();
chainSafe_=true;
for(Zq2 i=0;i<200u;++i)chainSafe_=chainSafe_&&output_[1024u].GetValue(i)!=0.0f;
chainCachesReady_=true;
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge200<StaticN, StaticK>::Load(Zq3 base,Zq2 batch)
{
columnArchiveReady_=false;initialForwardReady_=false;
const Zq2 m=M(),n=N(),k=K();
currentBatch_=batch;pathLocalRhs_=false;pathHasIterate_=false;chainCachesReady_=false;chainSafe_=false;


const Zq3 aBatch=Zq3(batch)*m*n;
const Zq3 bBatch=Zq3(batch)*m*k;
const Zq3 dstA=OffA(base),dstL=OffLU(base),dstB=OffB(base);
{
LoadRows200(base,batch);
}
{
for(Zq2 row=0;row<m;++row)
w_[dstB+row]=b_[(Zq3(batch)*m+row)*RhsWidth()+currentRhs_];
}
}
