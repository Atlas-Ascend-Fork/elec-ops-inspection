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

__aicore__ inline void VectorNormalize200::PublishPrivateDiagonalSeed200(Zq2 j)
{
const Zq2 pair=j/64u,half=(j/32u)&1u,col=(pair&1u)*64u,pitch=(pair&1u)?128u:64u;


if(!half){Duplicate(input_,0.0f,64u*pitch);Zq1<PIPE_V>();}
Gather(output_,all_[3072u],trans_[3072u],0u,1024u);Zq1<PIPE_V>();
Adds<int32_t>(input_[half*(32u*pitch+32u)+col].Zq6<int32_t>(),output_.Zq6<int32_t>(),int32_t(0),Zq3(32u),Zq4(32u),UnaryRepeatParams{1u,1u,Zq4(pitch/8u),4u});Zq1<PIPE_V>();
if(j==192u){


ZqB<Zq5::V_MTE2>();DataCopy(all_,inv_[7168u],1024u);ZqB<Zq5::MTE2_V>();
Gather(output_,all_,trans_[3072u],0u,1024u);Zq1<PIPE_V>();
Adds<int32_t>(input_[32u*pitch+32u+col].Zq6<int32_t>(),output_.Zq6<int32_t>(),int32_t(0),Zq3(32u),Zq4(32u),UnaryRepeatParams{1u,1u,Zq4(pitch/8u),4u});Zq1<PIPE_V>();
}
if(half||j==192u){
GlobalTensor<float>seeds;seeds.SetGlobalBuffer(seedWorkspace200_+356096u+(pair/2u)*16384u+(pair&1u)*8192u,8192u);
ZqB<Zq5::V_MTE3>();
if(pair&1u)DataCopy(seeds,input_,8192u);
else DataCopy(seeds,input_,DataCopyParams{64u,8u,0u,8u});
ZqB<Zq5::MTE3_V>();
}
}

__aicore__ inline void VectorNormalize200::StageEarlyR64Seed200(GM_ADDR workspace,Zq2 ownBatch,Zq2 j)
{
if(lane_!=3u||!(j==32u||j==96u||j==160u||j==192u))return;
const Zq2 pair=j==192u?3u:(j-32u)/64u,col=(pair%2u)*64u,pitch=(pair%2u)?128u:64u;
GlobalTensor<float>seeds;seeds.SetGlobalBuffer(Workspace(workspace,ownBatch)+356096u+(pair/2u)*16384u+(pair%2u)*8192u,8192u);
Duplicate(input_,0.0f,64u*pitch);Zq1<PIPE_V>();
for(Zq2 half=0u;half<2u;++half){
ZqB<Zq5::V_MTE2>();DataCopy(all_,inv_[(pair*2u+half)*1024u],1024u);ZqB<Zq5::MTE2_V>();
Gather(output_,all_,trans_,0u,1024u);Zq1<PIPE_V>();
Adds<int32_t>(input_[half*(32u*pitch+32u)+col].Zq6<int32_t>(),output_.Zq6<int32_t>(),int32_t(0),Zq3(32u),Zq4(32u),UnaryRepeatParams{1u,1u,Zq4(pitch/8u),4u});Zq1<PIPE_V>();
}
ZqB<Zq5::V_MTE3>();
if(pair%2u)DataCopy(seeds,input_,8192u);
else DataCopy(seeds,input_,DataCopyParams{64u,8u,0u,8u});
ZqB<Zq5::MTE3_V>();
}

template<Zq2 LogicalN,Zq2 RhsK>
__aicore__ inline void VectorNormalize200::PrepareStagedB(GM_ADDR b,Zq2 batch,Zq2 lane)
{
    QrVectorOperations::PrepareStagedB<LogicalN, RhsK>(*this, b, batch, lane);
}

template<Zq2 RhsK,Zq2 FactorN>
__aicore__ inline void VectorNormalize200::StageBChunk(__gm__ float*ptr,Zq2 lane,Zq2 row,float inverseScale)
{
    QrVectorOperations::StageBChunk<RhsK, FactorN>(*this, ptr, lane, row, inverseScale);
}

__aicore__ inline void VectorNormalize200::InitializeDenseExterior(__gm__ float*ptr,Zq2 lane)
{
GlobalTensor<float>targets[4];targets[0].SetGlobalBuffer(ptr+Goff,N*N);
targets[1].SetGlobalBuffer(ptr+213248u,N*N);targets[2].SetGlobalBuffer(ptr+278784u,N*N);targets[3].SetGlobalBuffer(ptr+NegOff,N*N);
for(Zq2 row=lane*32u;row<N;row+=8u*32u){
const bool bottom=row>=192u;Duplicate(input_,0.0f,bottom?8192u:1536u);Zq1<PIPE_V>();
Zq3 identityMask[2]={0x8040201008040201ull,0u};
if(bottom)for(Zq2 group=0u;group<4u;++group)
Duplicate<float>(input_[group*2056u+row],1.0f,identityMask,Zq4(1u),Zq4(32u),Zq4(8u));
ZqB<Zq5::V_MTE3>();
for(Zq2 matrix=0u;matrix<3u;++matrix){
if(bottom)DataCopy(targets[matrix][row*N],input_,8192u);
else DataCopy(targets[matrix][row*N+208u],input_,DataCopyParams{32u,6u,0u,26u});
}
ZqB<Zq5::MTE3_V>();
if(bottom){Muls(input_,input_,-1.0f,8192u);ZqB<Zq5::V_MTE3>();DataCopy(targets[3][row*N],input_,8192u);}
else DataCopy(targets[3][row*N+208u],input_,DataCopyParams{32u,6u,0u,26u});
ZqB<Zq5::MTE3_V>();
}
if(lane==0u){
Duplicate(input_,0.0f,1024u);Zq1<PIPE_V>();
Zq3 identityMask[2]={0x8040201008040201ull,0u};
Duplicate<float>(input_,1.0f,identityMask,Zq4(4u),Zq4(4u),Zq4(33u));
ZqB<Zq5::V_MTE3>();DataCopy(inv_[7u*1024u],input_,1024u);ZqB<Zq5::MTE3_V>();
Muls(input_,input_,-1.0f,1024u);ZqB<Zq5::V_MTE3>();DataCopy(invNeg_[7u*1024u],input_,1024u);ZqB<Zq5::MTE3_V>();
}
}

template<Zq2 LogicalN>
__aicore__ inline QrNormalization VectorNormalize200::Normalize(GM_ADDR a, __gm__ float* ptr, Zq2 batch, Zq2 lane)
{
    return QrVectorOperations::Normalize<LogicalN, false>(*this, a, ptr, batch, lane);
}

template<Zq2 LogicalN,Zq2 RhsK>
__aicore__ inline void VectorNormalize200::StageFactorPanels(__gm__ float* ptr, Zq2 lane, float inverseScale)
{
constexpr Zq2 factorN=224u;
for(Zq2 j=0;j<factorN;j+=32u){
if(lane==0u){if(j==0u)Diagonal<LogicalN>(j);}else StageBChunk<RhsK,factorN>(ptr,lane,j,inverseScale);


if(j>0u){CrossCoreWaitFlag(1);Zq1<PIPE_ALL>();SyncAll<true>();}


if(j==0u){Zq1<PIPE_ALL>();SyncAll<true>();}
CrossCoreSetFlag<2,PIPE_MTE3>(0);CrossCoreWaitFlag(1);Zq1<PIPE_ALL>();SyncAll<true>();
if(j+32u<factorN){CrossCoreSetFlag<2,PIPE_MTE3>(2);if(lane==0u)Diagonal<LogicalN>(j+32u);}
}
}

__aicore__ inline void VectorNormalize200::PrepareSecondGram(__gm__ float* ptr, Zq2 lane)
{
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(4);

CrossCoreWaitFlag(11);Zq1<PIPE_ALL>();SyncAll<true>();
CrossCoreSetFlag<2,PIPE_MTE3>(12);

CrossCoreWaitFlag(5);Zq1<PIPE_ALL>();SyncAll<true>();
if(lane==0u){
GlobalTensor<float>second;second.SetGlobalBuffer(ptr+213248u,N*N);
Duplicate(input_,0.0f,1536u);ZqB<Zq5::V_MTE3>();
DataCopy(second[lane*32u*N+208u],input_,DataCopyParams{32u,6u,0u,26u});
ZqB<Zq5::MTE3_V>();
}
CrossCoreSetFlag<2,PIPE_MTE3>(6);


CrossCoreWaitFlag(7);Zq1<PIPE_ALL>();SyncAll<true>();
}

__aicore__ inline void VectorNormalize200::PreparePolynomial(GM_ADDR workspace, __gm__ float* ptr, Zq2 lane)
{
GlobalTensor<float> h;
h.SetGlobalBuffer(ptr+213248u,N*N);
for(Zq2 row=lane*32u;row<N;row+=4u*32u){
ZqB<Zq5::V_MTE2>();DataCopy(input_,h[row*N],8192u);ZqB<Zq5::MTE2_V>();
Muls(input_,input_,-1.0f,8192u);Zq1<PIPE_V>();
AddSparseDiagonal(input_,row,2.0f);

Muls(output_,input_,-1.0f,8192u);Zq1<PIPE_V>();
AddSparseDiagonal(output_,row,1.0f);
Mul(output_,output_,output_,8192u);Zq1<PIPE_V>();
for(Zq2 half=4096u;half>=8u;half/=2u){Add(output_,output_,output_[half],half);Zq1<PIPE_V>();}
ZqB<Zq5::V_S>();for(Zq2 i=0;i<8u;++i)errorSquared_+=output_.GetValue(i);
ZqB<Zq5::V_MTE3>();DataCopy(h[row*N],input_,8192u);ZqB<Zq5::MTE3_V>();
}
WriteQrError(all_, errorLines_, lane, errorSquared_);

Zq1<PIPE_ALL>();SyncAll<true>();widePolynomial_=false;
for(Zq2 b=0;b<GetBlockNum()/4u;++b){const float value = ReadQrError(workspace, b, all_);
widePolynomial_=widePolynomial_||!(value>=0.0f&&value<0.0009765625f);}
if(lane==0u){auto words=all_.Zq6<Zq2>();for(Zq2 i=0;i<16u;++i)words.SetValue(i,0u);words.SetValue(0u,widePolynomial_?7u:1u);
ZqB<Zq5::S_MTE3>();DataCopy(header_[48u],all_,16u);ZqB<Zq5::MTE3_S>();}
Zq1<PIPE_ALL>();SyncAll<true>();
if(!widePolynomial_){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);Zq1<PIPE_ALL>();SyncAll<true>();return;}
RunWidePolynomial(ptr, h, lane);
}

__aicore__ inline void VectorNormalize200::RunWidePolynomial(__gm__ float* ptr, GlobalTensor<float> h, Zq2 lane)
{
GlobalTensor<float>ep;ep.SetGlobalBuffer(ptr+NegOff,N*N);
for(Zq2 row=lane*32u;row<N;row+=4u*32u){
ZqB<Zq5::V_MTE2>();DataCopy(input_,h[row*N],8192u);ZqB<Zq5::MTE2_V>();Muls(input_,input_,-1.0f,8192u);Zq1<PIPE_V>();
AddSparseDiagonal(input_,row,1.0f);
ZqB<Zq5::V_MTE3>();DataCopy(ep[row*N],input_,8192u);ZqB<Zq5::MTE3_V>();
}
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(8);
for(Zq2 phase=0;phase<4u;++phase){
CrossCoreWaitFlag(11);Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(12);
CrossCoreWaitFlag(13);Zq1<PIPE_ALL>();SyncAll<true>();
if(phase==0u){
errorSquared_=0.0f;
for(Zq2 row=lane*32u;row<N;row+=4u*32u){
ZqB<Zq5::V_MTE2>();DataCopy(input_,ep[row*N],8192u);ZqB<Zq5::MTE2_V>();Mul(output_,input_,input_,8192u);Zq1<PIPE_V>();
for(Zq2 half=4096u;half>=8u;half/=2u){Add(output_,output_,output_[half],half);Zq1<PIPE_V>();}
ZqB<Zq5::V_S>();for(Zq2 i=0;i<8u;++i)errorSquared_+=output_.GetValue(i);
}
WriteQrError(all_, errorLines_, lane, errorSquared_);
}
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(3);
}
CrossCoreWaitFlag(9);Zq1<PIPE_ALL>();SyncAll<true>();
}

__aicore__ inline void VectorNormalize200::PrepareFinalApply(Zq2 lane)
{
const Zq2 row=lane*32u;
ZqB<Zq5::V_MTE2>();DataCopy(input_,g_[row*N+128u],DataCopyParams{32u,16u,16u,0u});ZqB<Zq5::MTE2_V>();
Muls(output_,input_,-1.0f,4096u);ZqB<Zq5::V_MTE3>();
DataCopy(neg_[row*N+128u],output_,DataCopyParams{32u,16u,0u,16u});ZqB<Zq5::MTE3_V>();
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(14);
CrossCoreWaitFlag(15);Zq1<PIPE_ALL>();SyncAll<true>();
}

template<Zq2 LogicalN,Zq2 RhsK>
__aicore__ inline void VectorNormalize200::Run(GM_ADDR a, GM_ADDR b, GM_ADDR workspace, Zq2 batch, Zq2 lane)
{
static_assert(LogicalN == 200u && RhsK == 4u);
auto ptr = Workspace(workspace, batch);
auto normalization = Normalize<LogicalN>(a, ptr, batch, lane);
if (GetSubBlockIdx() != 0u) {
    FollowFactors200<LogicalN>(workspace, GetBlockNum()/4u, batch);
    return;
}
if (lane != 0u) PrepareStagedB<LogicalN,RhsK>(b,batch,lane);
if (lane == 0u) {
    PrepareFusedIndices<LogicalN>();
    PrepareDirectSeedTranspose200();
}
Zq1<PIPE_ALL>();
CrossCoreWaitFlag(1);Zq1<PIPE_ALL>();SyncAll<true>();
StageFactorPanels<LogicalN,RhsK>(ptr, lane, normalization.inverseScale);
PrepareSecondGram(ptr, lane);
PreparePolynomial(workspace, ptr, lane);
PrepareFinalApply(lane);
if (lane == 0u) {
    WriteQrHeader(all_, header_, batch, 7u, normalization, normalScaleDomain_ && pdOkay_,
                  widePolynomial_ ? 7u : 1u, 0u, 0u);
}
}


__aicore__ inline void VectorNormalize200::AddSparseDiagonal(Zq7<float>values,Zq2 row,float bias)
{
    QrVectorOperations::AddSparseDiagonal(values, row, bias);
}

template<Zq2 LogicalN>
__aicore__ inline void VectorNormalize200::ReadOriginal(GlobalTensor<float>matrix,Zq2 row)
{
    QrVectorOperations::ReadOriginal<LogicalN>(*this, matrix, row);
}

template<Zq2 LogicalN>
__aicore__ inline void VectorNormalize200::PrepareTranspose()
{
static_assert(LogicalN == 200u);
auto row=all_.Zq6<int32_t>(),col=all_[64u].Zq6<int32_t>();
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
CreateVecIndex<int32_t>(row,int32_t(0),Zq2(32));CreateVecIndex<int32_t>(col,int32_t(0),Zq2(256));Zq1<PIPE_V>();
Muls<int32_t>(row,row,int32_t(1024),Zq2(32));Muls<int32_t>(col,col,int32_t(4),Zq2(256));Zq1<PIPE_V>();
auto tmp=output_.Zq6<int32_t>();Brcb(tmp,col,Zq4(32u),Zq9{4u,32u});Zq1<PIPE_V>();
for(Zq2 half=0;half<2u;++half)
Add<int32_t>(trans_[half*4096u].Zq6<int32_t>(),row,tmp[half*4096u],Zq3(32u),Zq4(128u),Zq0{1,1,0,4,0,4});
Zq1<PIPE_V>();
}

template<Zq2 LogicalN>
__aicore__ inline void VectorNormalize200::PrepareFusedIndices()
{
static_assert(LogicalN == 200u);
ZqB<Zq5::V_S>();auto zero=all_[7840u].Zq6<Zq2>();

ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
auto ri=all_.Zq6<int32_t>(),ci=all_[64u].Zq6<int32_t>();
auto broadcast=output_.Zq6<int32_t>();auto ti=trans_.Zq6<int32_t>();
CreateVecIndex<int32_t>(ri,int32_t(0),Zq2(32));CreateVecIndex<int32_t>(ci,int32_t(0),Zq2(32));Zq1<PIPE_V>();
Brcb(broadcast,ri,Zq4(4u),Zq9{1u,8u});Zq1<PIPE_V>();
Min<int32_t>(ti,ci,broadcast,Zq3(32u),Zq4(32u),Zq0{1,1,0,4,0,1});
Max<int32_t>(ti[2048u],ci,broadcast,Zq3(32u),Zq4(32u),Zq0{1,1,0,4,0,1});
Adds<int32_t>(ti[1024u],broadcast,int32_t(0),Zq3(32u),Zq4(32u),UnaryRepeatParams{1,0,4,1});Zq1<PIPE_V>();
Muls<int32_t>(ti,ti,int32_t(128),Zq2(1024));Muls<int32_t>(ti[2048u],ti[2048u],int32_t(4),Zq2(1024));Muls<int32_t>(ti[1024u],ti[1024u],int32_t(32),Zq2(1024));Zq1<PIPE_V>();
Add<int32_t>(ti,ti,ti[2048u],Zq2(1024));Zq1<PIPE_V>();
Muls<int32_t>(broadcast,broadcast,int32_t(256),Zq2(256));Muls<int32_t>(ci,ci,int32_t(4),Zq2(32));Zq1<PIPE_V>();
Add<int32_t>(ti[2048u],ci,broadcast,Zq3(32u),Zq4(32u),Zq0{1,1,0,4,0,1});Zq1<PIPE_V>();
Adds<int32_t>(ti[2048u],ti[2048u],int32_t(128),Zq2(1024));Duplicate<Zq2>(zero,Zq2(0),Zq2(32));Zq1<PIPE_V>();
ZqB<Zq5::S_V>();
}

template<Zq2 LogicalN>
__aicore__ inline void VectorNormalize200::Diagonal(Zq2 j)
{
const Zq2 pivots=(LogicalN==200u&&j==192u)?8u:32u;
auto aug=all_,out=all_[2048u],inv=all_[3072u],draw=all_[4096u];
auto row=all_[4352u],negative=all_[4384u],diag=all_[4416u],pivot=all_[4448u],co8=all_[4512u],scales=all_[4768u];
auto gp=all_[5792u],ip=all_[6816u];auto zero=all_[7840u].Zq6<Zq2>();
QrVectorOperations::InitializeDiagonal<LogicalN>(*this,j,pivots);
for(Zq2 k=0;k<pivots;++k){
auto actualRow=out[k*32u];

Gather(draw[k*8u],aug,zero,(k*64u+k)*4u,32u);


if(k+1u<pivots)Muls(scales,aug[k*64u],-1.0f,64u);
Zq1<PIPE_V>();
if(k+1u<pivots){
Div(actualRow,aug[k*64u],draw[k*8u],32u);Zq1<PIPE_V>();
Brcb(co8,actualRow,Zq4(4u),Zq9{1u,8u});Zq1<PIPE_V>();
MulAddDst<float,float,true>(aug[(k+1u)*64u],co8[(k+1u)*8u],scales,
Zq3(64u),Zq4(pivots-1u-k),Zq0{1,0,1,8,1,0});Zq1<PIPE_V>();
}
}

if(pivots==8u){Duplicate(draw[64u],1.0f,192u);Zq1<PIPE_V>();}


ZqB<Zq5::V_S>();for(Zq2 i=0;i<32u;++i){float d=draw.GetValue(i*8u);pdOkay_=pdOkay_&&d>0.0f&&d<3.4028234e38f;}ZqB<Zq5::S_V>();


QrVectorOperations::StoreDiagonalInverse(inv,aug,draw,scales,gp,inv_,invNeg_,j);
if constexpr(LogicalN==200u)PublishPrivateDiagonalSeed200(j);
}
