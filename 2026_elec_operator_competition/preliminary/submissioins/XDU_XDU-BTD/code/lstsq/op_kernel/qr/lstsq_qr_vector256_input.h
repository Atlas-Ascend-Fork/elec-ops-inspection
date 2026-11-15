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

__aicore__ inline void VectorBounded256::StageRetiredInverseTranspose256(Zq2 block)
{
if(lane_!=block/2u)return;
auto seed=input_[4096u];


if((block&1u)==0u){Duplicate(seed,0.0f,4096u);Zq1<PIPE_V>();}
ZqB<Zq5::V_MTE2>();DataCopy(input_,inv_[block*1024u],1024u);ZqB<Zq5::MTE2_V>();
Gather(output_,input_,trans_,0u,1024u);Zq1<PIPE_V>();
Adds<int32_t>(seed[(block&1u)*2080u].Zq6<int32_t>(),output_.Zq6<int32_t>(),int32_t(0),
Zq3(32u),Zq4(32u),UnaryRepeatParams{1u,1u,8u,4u});
ZqB<Zq5::V_MTE3>();DataCopy(invNeg_[block*1024u],output_,1024u);
if(block&1u){GlobalTensor<float>seeds;seeds.SetGlobalBuffer(seedWorkspace256_+213248u+lane_*4096u,4096u);DataCopy(seeds,seed,4096u);}
ZqB<Zq5::MTE3_V>();
}

__aicore__ inline void VectorBounded256::LoadFoldedSymmetric32(Zq7<float>dst,GlobalTensor<float>raw,Zq2 row)
{
const Zq2 first=row/64u*64u;
ZqB<Zq5::S_MTE2>();ZqB<Zq5::V_MTE2>();
DataCopy(dst[first],raw[row*N+first],DataCopyParams{32u,uint16_t((N-first)/8u),uint16_t(first/8u),uint16_t(first/8u)});
for(Zq2 col=0u;col<first;col+=64u){
ZqB<Zq5::V_MTE2>();DataCopy(all_,raw[col*N+row],DataCopyParams{64u,4u,28u,0u});ZqB<Zq5::MTE2_V>();
Gather(all_[2048u],all_,all_[8192u].Zq6<Zq2>(),0u,2048u);Zq1<PIPE_V>();
Adds<int32_t>(dst[col].Zq6<int32_t>(),all_[2048u].Zq6<int32_t>(),int32_t(0),Zq3(64u),Zq4(32u),UnaryRepeatParams{1u,1u,32u,8u});Zq1<PIPE_V>();
}
ZqB<Zq5::MTE2_V>();Zq1<PIPE_V>();
}

template<Zq2 LogicalN,Zq2 RhsK>
__aicore__ inline void VectorBounded256::PrepareStagedB(GM_ADDR b,Zq2 batch,Zq2 lane)
{
    QrVectorOperations::PrepareStagedB<LogicalN, RhsK>(*this, b, batch, lane);
}

template<Zq2 RhsK,Zq2 FactorN>
__aicore__ inline void VectorBounded256::StageBChunk(__gm__ float*ptr,Zq2 lane,Zq2 row,float inverseScale)
{
    QrVectorOperations::StageBChunk<RhsK, FactorN>(*this, ptr, lane, row, inverseScale);
}

__aicore__ inline void VectorBounded256::InitializeDenseExterior(__gm__ float*ptr,Zq2 lane)
{
GlobalTensor<float>targets[4];targets[0].SetGlobalBuffer(ptr+Goff,N*N);
targets[1].SetGlobalBuffer(ptr+213248u,N*N);targets[2].SetGlobalBuffer(ptr+278784u,N*N);targets[3].SetGlobalBuffer(ptr+NegOff,N*N);
for(Zq2 row=lane*32u;row<N;row+=4u*32u){
const bool bottom=row>=224u;Duplicate(input_,0.0f,bottom?8192u:1024u);ZqB<Zq5::V_S>();
if(bottom)for(Zq2 i=0u;i<32u;++i)input_.SetValue(i*N+row+i,1.0f);
ZqB<Zq5::S_MTE3>();
for(Zq2 matrix=0u;matrix<3u;++matrix){
if(bottom)DataCopy(targets[matrix][row*N],input_,8192u);
else DataCopy(targets[matrix][row*N+224u],input_,DataCopyParams{32u,4u,0u,28u});
}
ZqB<Zq5::MTE3_V>();
if(bottom){Muls(input_,input_,-1.0f,8192u);ZqB<Zq5::V_MTE3>();DataCopy(targets[3][row*N],input_,8192u);}
else DataCopy(targets[3][row*N+224u],input_,DataCopyParams{32u,4u,0u,28u});
ZqB<Zq5::MTE3_V>();
}
if(lane==0u){
Duplicate(input_,0.0f,1024u);ZqB<Zq5::V_S>();
for(Zq2 i=0u;i<32u;++i)input_.SetValue(i*32u+i,1.0f);
ZqB<Zq5::S_MTE3>();DataCopy(inv_[7u*1024u],input_,1024u);ZqB<Zq5::MTE3_V>();
Muls(input_,input_,-1.0f,1024u);ZqB<Zq5::V_MTE3>();DataCopy(invNeg_[7u*1024u],input_,1024u);ZqB<Zq5::MTE3_V>();
}
}
