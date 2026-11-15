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

__aicore__ inline void KernelLegacy::Refine(Zq3 base)
{
const Zq2 n=tiling_.n,k=tiling_.k;
RefinementOperations::RefineSquareRows(*this, base, n, k);
}

__aicore__ inline void KernelLegacy::PrefetchResidualRow(Zq2 row,Zq2 slot)
{
const Zq2 n=tiling_.n,p=vectorPitch_;
Zq7<float>a=slot?vr_[12u*p]:vr_;

ZqB<Zq5::V_MTE2>(slot);
GlobalTensor<float>input;
input.SetGlobalBuffer(a_+(Zq3(currentBatch_)*n+row)*n,n);
DataCopyPad(a,input,Zq8{1u,n*4u,0u,0u,0u},
DataCopyPadExtParams<float>{true,0u,Zq4((8u-n%8u)%8u),0.0f});
SetFlag<Zq5::MTE2_V>(slot);
}

__aicore__ inline void KernelLegacy::PrepareResidualX(Zq3 base,Zq2 col)
{
const Zq2 p=vectorPitch_;
Zq7<float>x=vr_[p],xh=vr_[3u*p],xl=vr_[5u*p];
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<tiling_.n;++i)x.SetValue(i,GetX(base,i,col));
ZqB<Zq5::S_V>();
And(xh.Zq6<uint16_t>(),x.Zq6<uint16_t>(),
vr_[11u*p].Zq6<uint16_t>(),2u*p);
Zq1<PIPE_V>();
Sub(xl,x,xh,p);
Zq1<PIPE_V>();
}

__aicore__ inline float KernelLegacy::VectorResidual(Zq3 base,Zq2 row,Zq2 col,Zq2 slot)
{
    return ResidualOperations::ComputeScalarResidual(vr_, vectorPitch_, slot, 11u, 12u,
        w_ + OffB(base) + Zq3(row)*tiling_.k + col);
}

__aicore__ inline void KernelLegacy::RefineRectangular(Zq3 base,Zq2 batch)
{
    RectangularSolveOperations::Refine(*this, base, batch, tiling_.m, tiling_.n, tiling_.k);
}

__aicore__ inline void KernelLegacy::BackSubstitute(Zq3 base,Zq2 col)
{
if(vectorSolve_){
VectorTriangularColumn(base,w_+OffB(base)+col,tiling_.k,w_+OffX(base)+col,tiling_.k);
return;
}
const Zq2 n=tiling_.n;
const __gm__ float*lu=w_+OffLU(base);
const __gm__ float*rhsCopy=w_+OffB(base);
__gm__ float*out=w_+OffX(base)+col;
const __gm__ float*piv=w_+OffP(base);
for(Zq2 i=0;i<n;++i)out[Zq3(i)*tiling_.k]=rhsCopy[Zq3(i)*tiling_.k+col];


for(Zq2 j=0;j<n;++j){
const Zq2 p=static_cast<Zq2>(static_cast<int32_t>(piv[j]));
if(p==j)continue;
const float temp=out[Zq3(j)*tiling_.k];
out[Zq3(j)*tiling_.k]=out[Zq3(p)*tiling_.k];
out[Zq3(p)*tiling_.k]=temp;
}
for(Zq2 i=0;i<n;++i){
float value=out[Zq3(i)*tiling_.k];
for(Zq2 c=0;c<i;++c)value-=Factor(base,Zq2(i),c)*out[Zq3(c)*tiling_.k];
out[Zq3(i)*tiling_.k]=value;
}
for(int32_t i=int32_t(n)-1;i>=0;--i){
float value=out[Zq3(i)*tiling_.k];
for(Zq2 c=Zq2(i)+1;c<n;++c){
value-=Factor(base,Zq2(i),c)*out[Zq3(c)*tiling_.k];
}
const float diagonal=Factor(base,Zq2(i),Zq2(i));
out[Zq3(i)*tiling_.k]=diagonal==0.0f?0.0f:
(AbsF(diagonal)>=1.0e-30f&&AbsF(diagonal)<=1.0e30f?
square_math::RoundMul(value,w_[OffInv(base)+Zq2(i)]):square_math::Divide(value,diagonal));
}
}

__aicore__ inline float KernelLegacy::TriangularDot(Zq3 base,Zq2 row)
{
    return TriangularSolveOperations::TriangularDot(*this, base, row, tiling_.n, vectorPitch_, factorPitch_, localFactors_);
}

__aicore__ inline void KernelLegacy::VectorTriangularColumn(Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth)
{
const Zq2 n=tiling_.n,p=vectorPitch_;
Zq7<float>x=vr_[p];
ZqB<Zq5::S_V>();
Duplicate(x,0.0f,p);
for(Zq2 i=0;i<n;++i)solveY_.SetValue(i,in[Zq3(i)*inWidth]);
const __gm__ float*piv=w_+OffP(base);
for(Zq2 j=0;j<n;++j){
const Zq2 pick=Zq2(int32_t(piv[j]));
if(pick!=j){
const float v=solveY_.GetValue(j);
solveY_.SetValue(j,solveY_.GetValue(pick));solveY_.SetValue(pick,v);
}
}
for(Zq2 i=0;i<n;++i){
const float value=square_math::RoundSub(solveY_.GetValue(i),TriangularDot(base,i));
solveY_.SetValue(i,value);x.SetValue(i,value);
}
ZqB<Zq5::S_V>();
Duplicate(x,0.0f,p);
for(int32_t i=int32_t(n)-1;i>=0;--i){
const float value=square_math::RoundSub(solveY_.GetValue(i),TriangularDot(base,Zq2(i)));
const float diagonal=Factor(base,Zq2(i),Zq2(i));
const float answer=diagonal==0.0f?0.0f:
(AbsF(diagonal)>=1.0e-30f&&AbsF(diagonal)<=1.0e30f?
square_math::RoundMul(value,w_[OffInv(base)+Zq2(i)]):square_math::Divide(value,diagonal));
x.SetValue(i,answer);
}
for(Zq2 i=0;i<n;++i)out[Zq3(i)*outWidth]=x.GetValue(i);
}

__aicore__ inline void KernelLegacy::BatchedTriangularDots(Zq2 row,Zq2 width)
{
const Zq2 p=vectorPitch_;
ZqB<Zq5::S_V>();
Adds(vr_,factors_[row*factorPitch_],0.0f,factorPitch_);
TriangularSolveOperations::ReduceBatchedDots(vr_, p, width);
}

__aicore__ inline void KernelLegacy::VectorTriangularBatch(Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth,Zq2 width)
{
const Zq2 n=tiling_.n,p=vectorPitch_;
TriangularSolveOperations::VectorTriangularBatch<true>(*this, base, in, inWidth, out, outWidth, width, n, p);
}

__aicore__ inline void KernelLegacy::TriangularSolveQr(Zq3 base,Zq2 col)
{
    RectangularSolveOperations::SolveRhs(*this, base, col, tiling_.n, tiling_.k);
}

__aicore__ inline void KernelLegacy::SolveInPlaceWithFactors(Zq3 base,__gm__ float*rhs,Zq2 width)
{
if(vectorSolve_&&width<=4u){
VectorTriangularBatch(base,rhs,width,rhs,width,width);
return;
}
if(vectorSolve_){
for(Zq2 col=0;col<width;++col)
VectorTriangularColumn(base,rhs+col,width,rhs+col,width);
return;
}
const Zq2 n=tiling_.n;
TriangularSolveOperations::SolveScalarSquare(*this, base, w_+OffLU(base), w_+OffP(base), rhs, n, width);
}
