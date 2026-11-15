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

struct QrVectorOperations {

__aicore__ static inline void StoreDiagonalInverse(Zq7<float> inverse, Zq7<float> augmented,
    Zq7<float> draw, Zq7<float> scales, Zq7<float> negative,
    GlobalTensor<float> inverseOutput, GlobalTensor<float> negativeOutput, Zq2 block)
{
Adds<int32_t>(inverse.template Zq6<int32_t>(),augmented[32u].template Zq6<int32_t>(),int32_t(0),
Zq3(32u),Zq4(32u),UnaryRepeatParams{1,1,4,8});
Sqrt(draw,draw,256u);Zq1<PIPE_V>();
for(Zq2 group=0u;group<4u;++group){
Adds<int32_t>(scales[group*8u].template Zq6<int32_t>(),draw.template Zq6<int32_t>(),int32_t(0),
Zq3(8u),Zq4(32u),UnaryRepeatParams{1,1,4,1});
}
Zq1<PIPE_V>();Div(inverse,inverse,scales,1024u);Zq1<PIPE_V>();
ZqB<Zq5::V_MTE3>();DataCopy(inverseOutput[block/32u*1024u],inverse,1024u);
Muls(negative,inverse,-1.0f,1024u);ZqB<Zq5::V_MTE3>();DataCopy(negativeOutput[block/32u*1024u],negative,1024u);
ZqB<Zq5::MTE3_V>();
}

template<Zq2 RhsK,Zq2 FactorN, class Kernel>
__aicore__ static inline void StageBChunk(Kernel& kernel, __gm__ float* ptr, Zq2 lane, Zq2 row, float inverseScale)
{
ZqB<Zq5::V_S>();
for(Zq2 slot=0u,rhs=lane-1u;rhs<RhsK;rhs+=3u,++slot){
for(Zq2 i=0u;i<32u;++i){
float value=kernel.output_.GetValue(slot*256u+row+i);
kernel.bDomainOkay_[slot]=kernel.bDomainOkay_[slot]&&value<3.4028234e38f&&value>-3.4028234e38f;
float scaled=value*inverseScale;
kernel.bDomainOkay_[slot]=kernel.bDomainOkay_[slot]&&scaled<1.0e30f&&scaled>-1.0e30f;
}
GlobalTensor<float>staged;staged.SetGlobalBuffer(ptr+352768u+rhs*256u+row,32u);
ZqB<Zq5::S_MTE3>();DataCopy(staged,kernel.output_[slot*256u+row],32u);ZqB<Zq5::MTE3_S>();
if(row+32u==FactorN){
if constexpr(FactorN==224u){

GlobalTensor<float>pad;pad.SetGlobalBuffer(ptr+352768u+rhs*256u+224u,32u);
ZqB<Zq5::S_MTE3>();DataCopy(pad,kernel.output_[slot*256u+224u],32u);ZqB<Zq5::MTE3_S>();
}
for(Zq2 i=0u;i<16u;++i)kernel.all_.SetValue(i,0.0f);kernel.all_.SetValue(0u,kernel.bDomainOkay_[slot]?1.0f:0.0f);
GlobalTensor<float>flag;flag.SetGlobalBuffer(ptr+160u+rhs*16u,16u);
ZqB<Zq5::S_MTE3>();DataCopy(flag,kernel.all_,16u);ZqB<Zq5::MTE3_S>();
}
}
}

template<Zq2 LogicalN,Zq2 RhsK, class Kernel>
__aicore__ static inline void PrepareStagedB(Kernel& kernel, GM_ADDR b, Zq2 batch, Zq2 lane)
{
GlobalTensor<float>source;source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b)+Zq3(batch)*LogicalN*RhsK,LogicalN*RhsK);
ZqB<Zq5::V_MTE2>();DataCopy(kernel.input_,source,LogicalN*RhsK);
ZqB<Zq5::V_S>();auto seed=kernel.all_.Zq6<int32_t>();
for(Zq2 row=0u;row<32u;++row)seed.SetValue(row,int32_t(row*RhsK*4u));
ZqB<Zq5::S_V>();Duplicate(kernel.output_,0.0f,512u);Zq1<PIPE_V>();
for(Zq2 slot=0u,rhs=lane-1u;rhs<RhsK;rhs+=3u,++slot){
for(Zq2 group=0u;group<8u;++group)
Adds(kernel.trans_[slot*256u+group*32u].Zq6<int32_t>(),seed,int32_t((group*32u*RhsK+rhs)*4u),32u);
}
Zq1<PIPE_V>();ZqB<Zq5::MTE2_V>();
for(Zq2 slot=0u,rhs=lane-1u;rhs<RhsK;rhs+=3u,++slot)
Gather(kernel.output_[slot*256u],kernel.input_,kernel.trans_[slot*256u],0u,LogicalN);
Zq1<PIPE_V>();
}

template<Zq2 LogicalN, class Kernel>
__aicore__ static inline void ReadOriginal(Kernel& kernel, GlobalTensor<float> matrix, Zq2 row)
{
ZqB<Zq5::V_MTE2>();
if constexpr(LogicalN==256u){DataCopy(kernel.input_,matrix[row*N],8192u);ZqB<Zq5::MTE2_V>();}
else{
Duplicate(kernel.input_,0.0f,8192u);ZqB<Zq5::V_MTE2>();
if(row<LogicalN){Zq2 count=LogicalN-row<32u?LogicalN-row:32u;
DataCopyPad(kernel.input_,matrix[row*LogicalN],Zq8{uint16_t(count),LogicalN*4u,0u,(N-LogicalN)/8u,0u},DataCopyPadExtParams<float>{false,0u,0u,0.0f});
ZqB<Zq5::MTE2_V>();}else Zq1<PIPE_V>();
}
}

__aicore__ static inline void AddSparseDiagonal(Zq7<float> values, Zq2 row, float bias)
{
Zq3 diagonalMask[2]={0x8040201008040201ull,0u};


SetMaskNorm();ResetMask();
for(Zq2 group=0u;group<4u;++group){
const Zq2 begin=group*2056u+row;
Adds<float>(values[begin],values[begin],bias,diagonalMask,Zq4(1u),UnaryRepeatParams{32u,32u,8u,8u});
}
Zq1<PIPE_V>();ResetMask();SetMaskNorm();
}
template<Zq2 LogicalN, bool ColumnFactored256, class Kernel>
__aicore__ static inline QrNormalization Normalize(Kernel& kernel, GM_ADDR a, __gm__ float* ptr, Zq2 batch, Zq2 lane)
{
kernel.lane_=lane;
kernel.pipe_.InitBuffer(kernel.fb_,45568u);kernel.pipe_.InitBuffer(kernel.ib_,32768u);kernel.pipe_.InitBuffer(kernel.ob_,32768u);kernel.pipe_.InitBuffer(kernel.tb_,32768u);
kernel.all_=kernel.fb_.template Get<float>();kernel.input_=kernel.ib_.template Get<float>();kernel.output_=kernel.ob_.template Get<float>();kernel.trans_=kernel.tb_.template Get<Zq2>();
if constexpr (LogicalN == 200u) kernel.seedWorkspace200_=ptr;
kernel.g_.SetGlobalBuffer(ptr+Goff,N*N);kernel.neg_.SetGlobalBuffer(ptr+NegOff,N*N);
kernel.qt_.SetGlobalBuffer(ptr+QtOff,N*N);kernel.inv_.SetGlobalBuffer(ptr+InvOff,8192u);kernel.invNeg_.SetGlobalBuffer(ptr+InvNegOff,8192u);kernel.header_.SetGlobalBuffer(ptr,256u);kernel.errorLines_.SetGlobalBuffer(ptr+344320u,96u);
GlobalTensor<float>matrix;matrix.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a)+batch*LogicalN*LogicalN,LogicalN*LogicalN);
constexpr Zq2 normalizationWorkers=8u;
const Zq2 normalizationLane=lane*2u+GetSubBlockIdx();
if constexpr (LogicalN == 200u) {
    kernel.template PrepareTranspose<LogicalN>();
} else {
    kernel.template PrepareTranspose<LogicalN,ColumnFactored256>();
}
float localMax=0.0f;
for(Zq2 row=normalizationLane*32u;row<N;row+=normalizationWorkers*32u){
ReadOriginal<LogicalN>(kernel,matrix,row);
Abs(kernel.output_,kernel.input_,8192u);Zq1<PIPE_V>();ReduceMax(kernel.all_,kernel.output_,kernel.all_[1024u],8192u,false);ZqB<Zq5::V_S>();
float value=kernel.all_.GetValue(0u);if(value>localMax)localMax=value;
}
ZqB<Zq5::S_V>();Duplicate(kernel.all_,0.0f,16u);ZqB<Zq5::V_S>();kernel.all_.SetValue(0u,localMax);ZqB<Zq5::S_MTE3>();
DataCopy(kernel.header_[64u+normalizationLane*16u],kernel.all_,16u);ZqB<Zq5::MTE3_S>();

Zq1<PIPE_ALL>();SyncAll<true>();
DataCopy(kernel.all_,kernel.header_[64u],normalizationWorkers*16u);ZqB<Zq5::MTE2_S>();float largest=kernel.all_.GetValue(0u);
for(Zq2 i=1u;i<normalizationWorkers;++i){float value=kernel.all_.GetValue(i*16u);if(value>largest)largest=value;}
union{float f;Zq2 u;}power;power.f=largest;Zq2 exponent=(power.u>>23u)&255u;

kernel.normalScaleDomain_=exponent>=7u&&exponent<=247u;power.u=kernel.normalScaleDomain_?((253u-exponent)<<23u):0x3f800000u;float inverseScale=power.f;
for(Zq2 row=normalizationLane*32u;row<N;row+=normalizationWorkers*32u){

Muls(kernel.input_,kernel.input_,inverseScale,8192u);Zq1<PIPE_V>();
if constexpr(LogicalN==200u){ZqB<Zq5::V_S>();
for(Zq2 i=0;i<32u;++i)if(row+i>=LogicalN)kernel.input_.SetValue(i*N+row+i,1.0f);
ZqB<Zq5::S_V>();}
if constexpr(ColumnFactored256&&LogicalN==256u){
for(Zq2 tile=0u;tile<8u;++tile)Gather(kernel.output_[tile*1024u],kernel.input_,kernel.trans_,tile*128u,1024u);
}else Gather(kernel.output_,kernel.input_,kernel.trans_,0u,8192u);
ZqB<Zq5::V_MTE3>();
DataCopy(kernel.qt_[row],kernel.output_,DataCopyParams{256u,4u,0u,28u});ZqB<Zq5::MTE3_V>();

}
if constexpr(LogicalN==200u) kernel.InitializeDenseExterior(ptr,normalizationLane);
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(10);
return {inverseScale, exponent};
}
template<Zq2 LogicalN, class Kernel>
__aicore__ static inline void InitializeDiagonal(Kernel& kernel, Zq2 j, Zq2 pivots)
{
auto aug=kernel.all_,out=kernel.all_[2048u],inv=kernel.all_[3072u],draw=kernel.all_[4096u];
Duplicate(aug,0.0f,2048u);ZqB<Zq5::V_MTE2>();DataCopy(out,kernel.g_[j*N+j],DataCopyParams{32u,4u,28u,0u});ZqB<Zq5::MTE2_V>();
Gather(inv,out,kernel.trans_,0u,1024u);Zq1<PIPE_V>();
Adds<float>(aug,inv,0.0f,Zq3(32u),Zq4(32u),UnaryRepeatParams{1,1,8,4});Zq1<PIPE_V>();

Zq3 identityMask[2]={0x8040201008040201ull,0u};
Duplicate<float>(aug[32u],1.0f,identityMask,Zq4(4u),Zq4(8u),Zq4(65u));Zq1<PIPE_V>();

if constexpr(LogicalN==200u){if(pivots==8u){Duplicate(draw,1.0f,256u);Zq1<PIPE_V>();}}
}


};
