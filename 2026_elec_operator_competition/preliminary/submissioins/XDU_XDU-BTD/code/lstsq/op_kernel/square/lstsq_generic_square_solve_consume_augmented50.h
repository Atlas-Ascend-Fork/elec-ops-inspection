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
__aicore__ inline void KernelLarge<StaticN, StaticK>::ConsumeAugmented50(const __gm__ float*in,Zq2 stride)
{
ZqB<Zq5::V_S>();
if(!producerCorrection_&&augmented50RhsReady_)return;

if(!producerCorrection_){Adds(solveY_,negative64_[4224u],0.0f,64u);Zq1<PIPE_V>();}
ZqB<Zq5::S_V>();Zq3 rhsPad[2]={~((Zq3(1u)<<50u)-1u),0u};Duplicate<float>(solveY_,0.0f,rhsPad,Zq4(1),1,8);Zq1<PIPE_V>();Brcb(block_,solveY_,Zq4(8),Zq9{1,8});Zq1<PIPE_V>();
auto products=panel_[24576u];
Mul(products,negative64_,block_,Zq3(64),Zq4(64),Zq0{1,1,0,8,8,1});Zq1<PIPE_V>();
for(Zq2 h=2048u;h>=64u;h/=2u){Add(products,products,products[h],h);Zq1<PIPE_V>();}
Adds(solveY_,products,0.0f,64u);ZqB<Zq5::V_S>();
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::BackSubstitute(Zq3 base,Zq2 col)
{
if(UseVectorSolve()){
VectorTriangularColumn(base,w_+OffB(base)+col,K(),w_+OffX(base)+col,K());
return;
}
const Zq2 n=N();
const __gm__ float*lu=w_+OffLU(base);
const __gm__ float*rhsCopy=w_+OffB(base);
__gm__ float*out=w_+OffX(base)+col;
const __gm__ float*piv=w_+OffP(base);
for(Zq2 i=0;i<n;++i)out[Zq3(i)*K()]=rhsCopy[Zq3(i)*K()+col];


for(Zq2 j=0;j<n;++j){
const Zq2 p=static_cast<Zq2>(static_cast<int32_t>(piv[j]));
if(p==j)continue;
const float temp=out[Zq3(j)*K()];
out[Zq3(j)*K()]=out[Zq3(p)*K()];
out[Zq3(p)*K()]=temp;
}
for(Zq2 i=0;i<n;++i){
float value=out[Zq3(i)*K()];
for(Zq2 c=0;c<i;++c)value-=Factor(base,Zq2(i),c)*out[Zq3(c)*K()];
out[Zq3(i)*K()]=value;
}
for(int32_t i=int32_t(n)-1;i>=0;--i){
float value=out[Zq3(i)*K()];
for(Zq2 c=Zq2(i)+1;c<n;++c){
value-=Factor(base,Zq2(i),c)*out[Zq3(c)*K()];
}
const float diagonal=Factor(base,Zq2(i),Zq2(i));
out[Zq3(i)*K()]=diagonal==0.0f?0.0f:
(AbsF(diagonal)>=1.0e-30f&&AbsF(diagonal)<=1.0e30f?
square_math::RoundMul(value,w_[OffInv(base)+Zq2(i)]):square_math::Divide(value,diagonal));
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline float KernelLarge<StaticN, StaticK>::TriangularDot(Zq3 base,Zq2 row)
{
    return TriangularSolveOperations::TriangularDot(*this, base, row, N(), VectorPitch(), FactorPitch(), UseLocalFactors());
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::ColumnGmTriangular256(Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth,Zq2 width)
{
constexpr Zq2 n=256u,p=256u;
GlobalTensor<float>columns;columns.SetGlobalBuffer(w_+OffA(base),n*n);
ZqB<Zq5::S_V>();Duplicate(solveY_,0.0f,width*p);ZqB<Zq5::V_S>();
for(Zq2 col=0;col<width;++col)for(Zq2 row=0;row<n;++row)
solveY_[col*p].SetValue(row,in[Zq3(row)*inWidth+col]);
for(Zq2 row=0;row<n;++row){
const Zq2 pick=Zq2(int32_t(w_[OffP(base)+row]));
if(pick!=row)for(Zq2 col=0;col<width;++col){
auto y=solveY_[col*p];const float value=y.GetValue(row);
y.SetValue(row,y.GetValue(pick));y.SetValue(pick,value);
}
}


ColumnTriangularSolveOperations::ForwardColumnGm256(*this, columns, width);
ColumnTriangularSolveOperations::BackwardColumnGm256<false>(*this, columns, base, width);
for(Zq2 col=0;col<width;++col)for(Zq2 row=0;row<n;++row)
out[Zq3(row)*outWidth+col]=solveY_[col*p].GetValue(row);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::BatchedTriangularDots(Zq3 base,Zq2 row,Zq2 width)
{
const Zq2 p=VectorPitch();
ZqB<Zq5::S_V>();
if(UseLocalFactors())Adds(vr_,factors_[row*FactorPitch()],0.0f,FactorPitch());
else{
GlobalTensor<float>factor;
factor.SetGlobalBuffer(w_+OffLU(base)+Zq3(row)*N(),N());
DataCopy(vr_,factor,N());
ZqB<Zq5::MTE2_V>();
}
TriangularSolveOperations::ReduceBatchedDots(vr_, p, width);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::VectorTriangularBatch(Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth,Zq2 width)
{
const Zq2 n=N(),p=VectorPitch();
if(n==256u&&K()<=4u){
ColumnGmTriangular256(base,in,inWidth,out,outWidth,width);return;
}
TriangularSolveOperations::VectorTriangularBatch<false>(*this, base, in, inWidth, out, outWidth, width, n, p);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::VectorOnlyTriangular50(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride,Zq2 width)
{
constexpr Zq2 n=50u,p=64u;
auto order=residualTile_;
ZqB<Zq5::V_S>();
if(producerCorrection_){
ZqB<Zq5::S_V>();
Gather(negative64_[4160u],solveY_,order,0u,64u);Zq1<PIPE_V>();


}else{
for(Zq2 i=0;i<64u;++i)solveY_.SetValue(i,0.0f);
for(Zq2 i=0;i<N();++i)solveY_.SetValue(i,in[Zq3(order.GetValue(i)/4u)*stride]);
}
auto expanded=panel_;auto state=block_;
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
for(Zq2 c=0;c<n;++c)Brcb(expanded[c*448u],negative64_[c*56u],Zq4(7),Zq9{1,8});
Brcb(state,producerCorrection_?negative64_[4160u]:solveY_,Zq4(7),Zq9{1,8});Zq1<PIPE_V>();
SetMaskCount();
for(Zq2 row=0;row+1u<n;++row){
const Zq2 start=(row+1u)*8u;
ZqA<float,MaskMode::COUNTER>(0u,(n-row-1u)*8u);
MulAddDst<float,float,false>(state[start],expanded[row*448u+start],state[row*8u],Zq3(0),Zq4(1),Zq0{1,1,0,8,8,0});Zq1<PIPE_V>();
}
for(int32_t j=49;j>=0;--j){
const Zq2 row=Zq2(j);const float inverse=w_[OffInv(base)+row];
ZqA<float,MaskMode::COUNTER>(0u,8u);
Muls<float,false>(state[row*8u],state[row*8u],inverse,Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});Zq1<PIPE_V>();
if(!row)continue;
ZqA<float,MaskMode::COUNTER>(0u,row*8u);
MulAddDst<float,float,false>(state,expanded[row*448u],state[row*8u],Zq3(0),Zq4(1),Zq0{1,1,0,8,8,0});Zq1<PIPE_V>();
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
auto offsets=vr_[10u*p].template Zq6<Zq2>();
if(!replicaOffsetsReady_){for(Zq2 i=0;i<64u;++i)offsets.SetValue(i,i<50u?i*32u:1600u);replicaOffsetsReady_=true;}
ZqB<Zq5::S_V>();Gather(solveY_,state,offsets,0u,64u);Zq1<PIPE_V>();
ResetMask();ZqB<Zq5::V_S>();

}


template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::ColumnTriangularBatch(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride,Zq2 width)
{
const Zq2 n=N(),p=VectorPitch(),pitch=FactorPitch();
if constexpr(StaticN==50u){if(augmented50Ready_){producerPrivate_=true;ConsumeAugmented50(in,stride);return;}}
if constexpr(StaticN==50u&&StaticK==2u){
if(!diagonalKnown_){
diagonalSafe_=true;
for(Zq2 i=0;i<50u;++i){const float d=AbsF(factors_.GetValue(i*56u+i));diagonalSafe_=diagonalSafe_&&d>=1.0e-30f&&d<=1.0e30f;}
diagonalKnown_=true;
}
const bool safe50=diagonalSafe_;
if(safe50){
if(!negative64Ready_){ZqB<Zq5::S_V>();Muls(negative64_,factors_,-1.0f,50u*56u);Zq1<PIPE_V>();negative64Ready_=true;}
producerPrivate_=true;
VectorOnlyTriangular50(base,in,out,stride,width);return;
}
}


ZqB<Zq5::S_V>();Duplicate(solveY_,0.0f,width*p);ZqB<Zq5::V_S>();
{
for(Zq2 col=0;col<width;++col)for(Zq2 row=0;row<n;++row)
solveY_[col*p].SetValue(row,in[Zq3(row)*stride+col]);
}
if(!pathLocalRhs_){
ColumnTriangularSolveOperations::ApplyColumnPivots(*this,base,n,p,width);
}
for(Zq2 row=0;row+1u<n;++row){
const Zq2 start=(row+1u)/8u*8u,count=pitch-start;
ZqB<Zq5::S_V>();
if(n==200u||n==50u||n==64u){
ColumnTriangularSolveOperations::UpdateColumnTriangularRow(*this,row,width,p,pitch);
}else{
Adds(vr_[start],factors_[row*pitch+start],0.0f,count);
Zq1<PIPE_V>();
if(start<=row){Duplicate(vr_[start],0.0f,row-start+1u);Zq1<PIPE_V>();}
for(Zq2 col=0;col<width;++col){
const float value=solveY_[col*p].GetValue(row);
Axpy(solveY_[col*p+start],vr_[start],-value,int32_t(count));
}
}
ZqB<Zq5::V_S>();
}
for(int32_t row=int32_t(n)-1;row>=0;--row){
const float diagonal=factors_.GetValue(Zq2(row)*pitch+Zq2(row));
const bool safe=AbsF(diagonal)>=1.0e-30f&&AbsF(diagonal)<=1.0e30f;
const float inverse=pathLocalRhs_?output_[1024u].GetValue(Zq2(row)):w_[OffInv(base)+Zq2(row)];
TriangularSolveOperations::ApplyBackwardColumnRow(solveY_,factors_,row,p,pitch,width,diagonal,inverse,safe);
}
{
for(Zq2 col=0;col<width;++col)for(Zq2 row=0;row<n;++row)
out[Zq3(row)*stride+col]=solveY_[col*p].GetValue(row);
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::TriangularSolveQr(Zq3 base,Zq2 col)
{
    RectangularSolveOperations::SolveRhs(*this, base, col, N(), K());
}
