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

template<Zq2 LogicalN,bool ColumnFactored256>
__aicore__ inline QrNormalization VectorBounded256::Normalize(GM_ADDR a, __gm__ float* ptr, Zq2 batch, Zq2 lane)
{
    return QrVectorOperations::Normalize<LogicalN, ColumnFactored256>(*this, a, ptr, batch, lane);
}

template<Zq2 LogicalN,Zq2 RhsK,bool ColumnFactored256>
__aicore__ inline void VectorBounded256::StageFactorPanels(__gm__ float* ptr, Zq2 lane, float inverseScale)
{
constexpr Zq2 factorN=256u;
for(Zq2 j=0;j<factorN;j+=32u){
if(j>0u)CrossCoreSetFlag<2,PIPE_MTE3>(2);


if(lane==0u)Diagonal<LogicalN,ColumnFactored256>(j);else StageBChunk<RhsK,factorN>(ptr,lane,j,inverseScale);
if(j>0u)CrossCoreWaitFlag(1);
Zq1<PIPE_ALL>();SyncAll<true>();
CrossCoreSetFlag<2,PIPE_MTE3>(0);CrossCoreWaitFlag(1);Zq1<PIPE_ALL>();SyncAll<true>();
}
}

__aicore__ inline void VectorBounded256::PrepareSecondGram(__gm__ float* ptr, Zq2 lane)
{
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(4);
CrossCoreWaitFlag(11);Zq1<PIPE_ALL>();SyncAll<true>();
if(lane<2u){
GlobalTensor<float>blocks,wide;
blocks.SetGlobalBuffer(ptr+213248u+lane*8192u,8192u);
wide.SetGlobalBuffer(ptr+356096u+lane*16384u,16384u);
for(Zq2 half=0u;half<2u;++half){
Duplicate(input_,0.0f,8192u);ZqB<Zq5::V_MTE2>();
DataCopy(input_[half*64u],blocks[half*4096u],DataCopyParams{64u,8u,0u,8u});
ZqB<Zq5::MTE2_MTE3>();DataCopy(wide[half*8192u],input_,8192u);ZqB<Zq5::MTE3_V>();
}
}
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(12);
CrossCoreWaitFlag(11);Zq1<PIPE_ALL>();SyncAll<true>();
CrossCoreSetFlag<2,PIPE_MTE3>(12);
CrossCoreWaitFlag(5);Zq1<PIPE_ALL>();SyncAll<true>();
CrossCoreSetFlag<2,PIPE_MTE3>(6);
PrepareFoldedSymmetricReadIndices256();
CrossCoreWaitFlag(7);Zq1<PIPE_ALL>();SyncAll<true>();
}

template<Zq2 LogicalN,bool ColumnFactored256>
__aicore__ inline void VectorBounded256::PreparePolynomial(GM_ADDR workspace, __gm__ float* ptr, Zq2 batch, Zq2 lane)
{
GlobalTensor<float> h;
h.SetGlobalBuffer(ptr+213248u,N*N);
GlobalTensor<float>rawGram;rawGram.SetGlobalBuffer(ptr+278784u,N*N);
for(Zq2 row=lane*32u;row<N;row+=4u*32u){
auto hwork=trans_.Zq6<float>();
auto retained=row<128u?input_:output_;
LoadFoldedSymmetric32(hwork,rawGram,row);
Muls(hwork,hwork,-1.0f,8192u);Zq1<PIPE_V>();
AddSparseDiagonal(hwork,row,2.0f);
Muls(retained,hwork,-1.0f,8192u);Zq1<PIPE_V>();
AddSparseDiagonal(retained,row,1.0f);
ZqB<Zq5::S_V>();Mul(all_,retained,retained,8192u);Zq1<PIPE_V>();
for(Zq2 half=4096u;half>=8u;half/=2u){Add(all_,all_,all_[half],half);Zq1<PIPE_V>();}
ZqB<Zq5::V_S>();for(Zq2 i=0;i<8u;++i)errorSquared_+=all_.GetValue(i);
ZqB<Zq5::S_MTE3>();DataCopy(h[row*N],hwork,8192u);ZqB<Zq5::MTE3_V>();
}
WriteQrError(all_, errorLines_, lane, errorSquared_);

Zq1<PIPE_ALL>();SyncAll<true>();
widePolynomial_ = SelectInitialDegree(workspace, GetBlockNum()/4u);
{auto words=all_.Zq6<Zq2>();for(Zq2 i=0;i<16u;++i)words.SetValue(i,0u);words.SetValue(1u,pooledLightMask_);words.SetValue(0u,widePolynomial_?(cubicPolynomial_?3u:7u):1u);
ZqB<Zq5::S_MTE3>();DataCopy(header_[64u+lane*16u],all_,16u);ZqB<Zq5::MTE3_S>();}
Zq1<PIPE_ALL>();SyncAll<true>();
if(!widePolynomial_){CrossCoreSetFlag<2,PIPE_MTE3>(8);CrossCoreWaitFlag(9);Zq1<PIPE_ALL>();SyncAll<true>();return;}
RunWidePolynomial(workspace, ptr, h, batch, lane);
}

__aicore__ inline void VectorBounded256::RunWidePolynomial(GM_ADDR workspace, __gm__ float* ptr, GlobalTensor<float> h, Zq2 batch, Zq2 lane)
{
GlobalTensor<float>ep;ep.SetGlobalBuffer(ptr+NegOff,N*N);
for(Zq2 row=lane*32u;row<N;row+=4u*32u){
auto retained=row<128u?input_:output_;
ZqB<Zq5::V_MTE3>();DataCopy(ep[row*N],retained,8192u);ZqB<Zq5::MTE3_V>();
}
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(8);
for(Zq2 phase=0;phase<4u;++phase){
CrossCoreWaitFlag(11);Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(12);
CrossCoreWaitFlag(13);Zq1<PIPE_ALL>();SyncAll<true>();
if (phase == 0u) UpdateSecondPower(workspace, ptr, ep, h, batch, lane);
else { Zq1<PIPE_ALL>();SyncAll<true>(); }
if (phase == 2u && quarticPolynomial_ && !(pooledLightMask_ & (1u << batch))) {
    GlobalTensor<float> e4;
    e4.SetGlobalBuffer(ptr + 278784u, N*N);
    AddPolynomialPower(h, e4, lane);
}
if ((phase == 0u && (quadraticPolynomial_ || reducedH4_ || pooledLightMask_ == 0u)) ||
    (phase == 2u && quarticPolynomial_)) {
    Zq1<PIPE_ALL>();SyncAll<true>();
}
CrossCoreSetFlag<2,PIPE_MTE3>(3);
if((quadraticPolynomial_&&phase==0u)||(cubicPolynomial_&&phase==1u)||(quarticPolynomial_&&phase==(reducedH4_?1u:2u)))break;
}
CrossCoreWaitFlag(9);Zq1<PIPE_ALL>();SyncAll<true>();
}

__aicore__ inline void VectorBounded256::MeasureSecondPower(__gm__ float* ptr, GlobalTensor<float> ep, Zq2 lane)
{
errorSquared_=0.0f;
for(Zq2 row=lane*32u;row<N;row+=4u*32u){


auto retained=row<128u?input_:output_;
if(pooledLightMask_==0u){GlobalTensor<float>rawE2;rawE2.SetGlobalBuffer(ptr+278784u,N*N);LoadFoldedSymmetric32(retained,rawE2,row);}
else{ZqB<Zq5::V_MTE2>();DataCopy(retained,ep[row*N],8192u);ZqB<Zq5::MTE2_V>();}
Mul(all_,retained,retained,8192u);Zq1<PIPE_V>();
for(Zq2 half=4096u;half>=8u;half/=2u){Add(all_,all_,all_[half],half);Zq1<PIPE_V>();}
ZqB<Zq5::V_S>();for(Zq2 i=0;i<8u;++i)errorSquared_+=all_.GetValue(i);
}
WriteQrError(all_, errorLines_, lane, errorSquared_);
}

__aicore__ inline void VectorBounded256::UpdateSecondPower(GM_ADDR workspace, __gm__ float* ptr, GlobalTensor<float> ep, GlobalTensor<float> h, Zq2 batch, Zq2 lane)
{
const bool heavy = !(pooledLightMask_ & (1u << batch));
if (heavy) MeasureSecondPower(ptr, ep, lane);
Zq1<PIPE_ALL>();SyncAll<true>();
if (pooledLightMask_ != 0u && !cubicPolynomial_ && reducedH4Eligible_ && heavy) {
    Muls(input_,input_,1.1311959028244019f,8192u);
    Muls(output_,output_,1.1311959028244019f,8192u);
}
SelectActualE2(workspace,GetBlockNum()/4u,true);
if (pooledLightMask_ == 0u && !quadraticPolynomial_) {
    if(reducedH4_){Muls(input_,input_,1.1311959028244019f,8192u);Muls(output_,output_,1.1311959028244019f,8192u);Zq1<PIPE_V>();}
    StoreSecondPower(ep, lane);
} else if (reducedH4_ && heavy) {
    StoreSecondPower(ep, lane);
}
if (quadraticPolynomial_ && heavy) {
    if (pooledLightMask_ == 0u) {
for(Zq2 row=lane*32u;row<N;row+=4u*32u){
auto retained=row<128u?input_:output_;auto hwork=trans_.Zq6<float>();
ZqB<Zq5::V_MTE2>();DataCopy(hwork,h[row*N],8192u);ZqB<Zq5::MTE2_V>();
Add(hwork,hwork,retained,8192u);ZqB<Zq5::V_MTE3>();DataCopy(h[row*N],hwork,8192u);ZqB<Zq5::MTE3_V>();}
    } else {
        AddPolynomialPower(h, ep, lane);
    }
}
}

__aicore__ inline void VectorBounded256::StoreSecondPower(GlobalTensor<float> ep, Zq2 lane)
{
ZqB<Zq5::V_MTE3>();
DataCopy(ep[lane*32u*N],input_,8192u);
DataCopy(ep[(lane*32u+128u)*N],output_,8192u);
ZqB<Zq5::MTE3_V>();
}

__aicore__ inline void VectorBounded256::AddPolynomialPower(GlobalTensor<float> h, GlobalTensor<float> power, Zq2 lane)
{
for(Zq2 row=lane*32u;row<N;row+=4u*32u){
ZqB<Zq5::V_MTE2>();DataCopy(input_,h[row*N],8192u);DataCopy(output_,power[row*N],8192u);ZqB<Zq5::MTE2_V>();
Add(input_,input_,output_,8192u);ZqB<Zq5::V_MTE3>();DataCopy(h[row*N],input_,8192u);ZqB<Zq5::MTE3_V>();}
}

template<Zq2 LogicalN,Zq2 RhsK,bool ColumnFactored256>
__aicore__ inline void VectorBounded256::Run(GM_ADDR a, GM_ADDR b, GM_ADDR workspace, Zq2 batch, Zq2 lane)
{
static_assert(LogicalN == 256u && RhsK == 3u && ColumnFactored256);
auto ptr = Workspace(workspace, batch);
auto normalization = Normalize<LogicalN,ColumnFactored256>(a, ptr, batch, lane);
if (GetSubBlockIdx() != 0u) {
    seedWorkspace256_ = ptr;
    FollowFactors200<LogicalN,ColumnFactored256>(workspace, GetBlockNum()/4u);
    return;
}
if (lane != 0u) PrepareStagedB<LogicalN,RhsK>(b,batch,lane);
if (lane == 0u) PrepareFusedIndices<LogicalN>();
Zq1<PIPE_ALL>();
CrossCoreWaitFlag(1);Zq1<PIPE_ALL>();SyncAll<true>();
StageFactorPanels<LogicalN,RhsK,ColumnFactored256>(ptr, lane, normalization.inverseScale);
PrepareSecondGram(ptr, lane);
PreparePolynomial<LogicalN,ColumnFactored256>(workspace, ptr, batch, lane);
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(14);
CrossCoreWaitFlag(15);Zq1<PIPE_ALL>();SyncAll<true>();
if (lane == 0u) {
    const Zq2 degree = widePolynomial_ ?
        (quadraticPolynomial_ ? 2u : (cubicPolynomial_ ? 3u : (quarticPolynomial_ ? 4u : 7u))) : 1u;
    WriteQrHeader(all_, header_, batch, 8u, normalization, normalScaleDomain_ && pdOkay_,
                  degree | ((pooledLightMask_ & (1u << batch)) ? 0x100u : 0u),
                  0x43464132u, reducedH4_ ? 0x52483433u : 0u);
}
}


__aicore__ inline void VectorBounded256::AddSparseDiagonal(Zq7<float>values,Zq2 row,float bias)
{
    QrVectorOperations::AddSparseDiagonal(values, row, bias);
}

__aicore__ inline void VectorBounded256::SelectActualE2(GM_ADDR workspace,Zq2 batches,bool publish)
{
bool actualH2=true,actualH3=true,actualH4=true;
for(Zq2 batch=0u;batch<batches;++batch){if(pooledLightMask_&(1u<<batch))continue;GlobalTensor<float>lines;
lines.SetGlobalBuffer(Workspace(workspace,batch)+344320u,64u);
DataCopy(all_,lines,64u);ZqB<Zq5::MTE2_S>();float value=all_.GetValue(0u);
for(Zq2 i=1u;i<4u;++i)value+=all_.GetValue(i*16u);
actualH2=actualH2&&value>=0.0f&&value<0.000244140625f;
actualH3=actualH3&&value>=0.0f&&value<0.0009765625f;
actualH4=actualH4&&value>=0.0f&&value<0.015625f;
}

quadraticPolynomial_=actualH2;
cubicPolynomial_=!quadraticPolynomial_&&(cubicPolynomial_||actualH3);
quarticPolynomial_=!quadraticPolynomial_&&!cubicPolynomial_&&actualH4;
reducedH4_=quarticPolynomial_&&reducedH4Eligible_;
if(publish){auto words=all_.Zq6<Zq2>();for(Zq2 i=0;i<16u;++i)words.SetValue(i,0u);
words.SetValue(0u,quadraticPolynomial_?2u:((cubicPolynomial_||reducedH4_)?3u:(quarticPolynomial_?4u:7u)));words.SetValue(1u,pooledLightMask_);
ZqB<Zq5::S_MTE3>();DataCopy(header_[64u+lane_*16u],all_,16u);ZqB<Zq5::MTE3_S>();}
}

template<Zq2 LogicalN>
__aicore__ inline void VectorBounded256::ReadOriginal(GlobalTensor<float>matrix,Zq2 row)
{
    QrVectorOperations::ReadOriginal<LogicalN>(*this, matrix, row);
}

template<Zq2 LogicalN,bool CompactTranspose>
__aicore__ inline void VectorBounded256::PrepareTranspose()
{
static_assert(LogicalN == 256u && CompactTranspose);
auto row=all_.Zq6<int32_t>(),col=all_[64u].Zq6<int32_t>();
for(Zq2 i=0;i<32u;++i)row.SetValue(i,int32_t(i*1024u));
for(Zq2 i=0;i<32u;++i)col.SetValue(i,int32_t(i*4u));ZqB<Zq5::S_V>();
auto tmp=output_.Zq6<int32_t>();Brcb(tmp,col,Zq4(4u),Zq9{4u,32u});Zq1<PIPE_V>();
Add<int32_t>(trans_.Zq6<int32_t>(),row,tmp,Zq3(32u),Zq4(32u),Zq0{1,1,0,4,0,4});Zq1<PIPE_V>();
}

template<Zq2 LogicalN>
__aicore__ inline void VectorBounded256::PrepareFusedIndices()
{
static_assert(LogicalN == 256u);
ZqB<Zq5::V_S>();auto zero=all_[7840u].Zq6<Zq2>();


ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
auto ri=all_.Zq6<int32_t>(),ci=all_[64u].Zq6<int32_t>();
auto broadcast=output_.Zq6<int32_t>();auto ti=trans_.Zq6<int32_t>();
CreateVecIndex<int32_t>(ri,int32_t(0),Zq2(32));CreateVecIndex<int32_t>(ci,int32_t(0),Zq2(32));Zq1<PIPE_V>();
Brcb(broadcast,ri,Zq4(4u),Zq9{1u,8u});Zq1<PIPE_V>();
Min<int32_t>(ti,ci,broadcast,Zq3(32u),Zq4(32u),Zq0{1,1,0,4,0,1});
Max<int32_t>(ti[2048u],ci,broadcast,Zq3(32u),Zq4(32u),Zq0{1,1,0,4,0,1});Zq1<PIPE_V>();
Muls<int32_t>(ti,ti,int32_t(128),Zq2(1024));Muls<int32_t>(ti[2048u],ti[2048u],int32_t(4),Zq2(1024));Zq1<PIPE_V>();
Add<int32_t>(ti,ti,ti[2048u],Zq2(1024));Duplicate<Zq2>(zero,Zq2(0),Zq2(32));Zq1<PIPE_V>();
}

template<Zq2 LogicalN,bool PackedPivotCheck>
__aicore__ inline void VectorBounded256::Diagonal(Zq2 j)
{
const Zq2 pivots=(LogicalN==200u&&j==192u)?8u:32u;
auto aug=all_,out=all_[2048u],inv=all_[3072u],draw=all_[4096u];
auto row=all_[4352u],negative=all_[4384u],diag=all_[4416u],pivot=all_[4448u],co8=all_[4512u],scales=all_[4768u];
auto gp=all_[5792u],ip=all_[6816u];auto zero=all_[7840u].Zq6<Zq2>();
QrVectorOperations::InitializeDiagonal<LogicalN>(*this,j,pivots);
for(Zq2 k=0;k<pivots;++k){
auto actualRow=out[k*32u];

Gather(draw[k*8u],aug,zero,(k*64u+k)*4u,32u);Zq1<PIPE_V>();
if(k+1u<pivots){
Div(actualRow,aug[k*64u],draw[k*8u],32u);Zq1<PIPE_V>();

Muls(negative,actualRow,-1.0f,32u);Zq1<PIPE_V>();
Brcb(co8,negative,Zq4(4u),Zq9{1u,8u});Zq1<PIPE_V>();


MulAddDst<float,float,true>(aug[(k+1u)*64u],co8[(k+1u)*8u],aug[k*64u],
Zq3(64u),Zq4(pivots-1u-k),Zq0{1,0,1,8,1,0});Zq1<PIPE_V>();
}
}
if constexpr(PackedPivotCheck&&LogicalN==256u){


auto pivotBits=draw.template Zq6<int32_t>();
auto clipped=out.template Zq6<int32_t>();
auto valid=row.template Zq6<Zq4>();
Maxs(clipped,pivotBits,int32_t(1),256u);Zq1<PIPE_V>();
Mins(clipped,clipped,int32_t(0x7f7ffffe),256u);Zq1<PIPE_V>();
Compare(valid,clipped,pivotBits,CMPMODE::EQ,256u);
ZqB<Zq5::V_S>();
auto words=valid.template Zq6<Zq3>();
for(Zq2 i=0u;i<4u;++i)pdOkay_=pdOkay_&&words.GetValue(i)==~Zq3(0);
ZqB<Zq5::S_V>();
}else{
ZqB<Zq5::V_S>();for(Zq2 i=0;i<32u;++i){float d=draw.GetValue(i*8u);pdOkay_=pdOkay_&&d>0.0f&&d<3.4028234e38f;}ZqB<Zq5::S_V>();
}


QrVectorOperations::StoreDiagonalInverse(inv,aug,draw,scales,gp,inv_,invNeg_,j);
}
