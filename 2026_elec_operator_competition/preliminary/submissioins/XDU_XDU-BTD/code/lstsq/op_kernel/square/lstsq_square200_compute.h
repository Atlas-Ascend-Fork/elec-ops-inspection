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
__aicore__ inline void KernelLarge200<StaticN, StaticK>::ProducerRefine16(Zq3 base)
{
constexpr Zq2 q=4128u,p=4096u,pitch=256u;
GlobalTensor<float>archived;archived.SetGlobalBuffer(w_+OffA(base),40000u);
if(!columnArchiveReady_){ZqB<Zq5::V_MTE3>();DataCopy(archived,factors_,40000u);ZqB<Zq5::MTE3_S>();columnArchiveReady_=true;}
auto offsets=ProducerResidualOffsets16();
for(Zq2 f=0;f<8u;++f)for(Zq2 row=0;row<16u;++row)
offsets.SetValue(f*16u+row,(row*256u+f)*4u);
ZqB<Zq5::S_V>();
ZqB<Zq5::S_V>();Duplicate(output_[256u],0.0f,256u);Zq1<PIPE_V>();
PrepareChainCaches(base);
for(Zq2 step=0;step<2u;++step){
ResidualOperations::RunResidual16Pass(*this,base);
ZqB<Zq5::V_MTE2>();DataCopy(factors_,archived,40000u);ZqB<Zq5::MTE2_S>();
pathLocalRhs_=true;SolveInPlaceWithFactors(base,w_+OffD(base),1u);
ZqB<Zq5::S_V>();Add(output_,output_,output_[256u],256u);Zq1<PIPE_V>();
}
Adds(solveY_,output_,0.0f,256u);ZqB<Zq5::V_S>();
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge200<StaticN, StaticK>::PrepareResidualX(Zq3 base,Zq2 col)
{
    ResidualOperations::PrepareResidualXFromOutput(*this, base, col);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge200<StaticN, StaticK>::ColumnTriangularBatch(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride,Zq2 width)
{
const Zq2 n=N(),p=VectorPitch(),pitch=FactorPitch();


PrepareChainCaches(base);
{
if(initialForwardReady_&&width==1u&&!pathLocalRhs_){
ZqB<Zq5::S_V>();Adds(solveY_,output_,0.0f,256u);ZqB<Zq5::V_S>();initialForwardReady_=false;
}else{
ZqB<Zq5::S_V>();Duplicate(solveY_,0.0f,width*p);ZqB<Zq5::V_S>();
{
if(pathLocalRhs_||chainCachesReady_){
ZqB<Zq5::S_V>();Gather(solveY_,pathLocalRhs_?output_[256u]:output_[512u],output_[768u].template Zq6<Zq2>(),0u,256u);ZqB<Zq5::V_S>();
}else{
for(Zq2 row=0;row<n;++row)solveY_.SetValue(row,in[Zq3(row)*stride]);
}
}
if(!chainCachesReady_){
ColumnTriangularSolveOperations::ApplyColumnPivots(*this,base,n,p,width);
}
for(Zq2 row=0;row+1u<n;++row){
ZqB<Zq5::S_V>();
{
ColumnTriangularSolveOperations::UpdateColumnTriangularRow(*this,row,width,p,pitch);
}
ZqB<Zq5::V_S>();
}
}
}
for(int32_t row=int32_t(n)-1;row>=0;--row){
{
if(width==1u&&chainSafe_){
const float answer=square_math::RoundMul(solveY_.GetValue(Zq2(row)),output_[1024u].GetValue(Zq2(row)));
solveY_.SetValue(Zq2(row),answer);
if(row>0){ZqB<Zq5::S_V>();Axpy(solveY_,factors_[Zq2(row)*pitch],-answer,row);ZqB<Zq5::V_S>();}
continue;
}
}
const float diagonal=factors_.GetValue(Zq2(row)*pitch+Zq2(row));
const bool safe=AbsF(diagonal)>=1.0e-30f&&AbsF(diagonal)<=1.0e30f;
const float inverse=chainCachesReady_?output_[1024u].GetValue(Zq2(row)):w_[OffInv(base)+Zq2(row)];
TriangularSolveOperations::ApplyBackwardColumnRow(solveY_,factors_,row,p,pitch,width,diagonal,inverse,safe);
}
{
ZqB<Zq5::S_V>();
Adds(pathLocalRhs_?output_[256u]:output_,solveY_,0.0f,256u);
Zq1<PIPE_V>();pathHasIterate_=true;
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline Zq2 KernelLarge200<StaticN, StaticK>::PivotIndex(Zq7<float>column,Zq2 first)
{
const Zq2 n=N(),p=VectorPitch();
float best=AbsF(column.GetValue(first));
if(n-first<=((n==50u||n==64u)?8u:32u)||best!=best){
Zq2 pick=first;
for(Zq2 i=first+1u;i<n;++i){
const float value=AbsF(column.GetValue(i));
if(value>best){best=value;pick=i;}
}
return pick;
}
{
const Zq2 start=first/8u*8u,count=n-start;
ZqB<Zq5::S_V>();Abs(vr_,column[start],count);Zq1<PIPE_V>();
if(first>start){Duplicate(vr_,-1.0f,first-start);Zq1<PIPE_V>();}
ReduceMax(vr_[p],vr_,vr_[2u*p],int32_t(count),true);ZqB<Zq5::V_S>();
const float maximum=vr_[p].GetValue(0u);
const Zq2 pick=start+vr_[p].template Zq6<Zq2>().GetValue(1u);
if(pick>=first&&pick<n&&maximum==maximum)return pick;
Zq2 fallback=first;
for(Zq2 i=first+1u;i<n;++i){const float value=AbsF(column.GetValue(i));if(value>best){best=value;fallback=i;}}
return fallback;
}


}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge200<StaticN, StaticK>::DecomposeRows200(Zq3 base)
{
    PanelLu200::Factor<true>(*this, base);
}
