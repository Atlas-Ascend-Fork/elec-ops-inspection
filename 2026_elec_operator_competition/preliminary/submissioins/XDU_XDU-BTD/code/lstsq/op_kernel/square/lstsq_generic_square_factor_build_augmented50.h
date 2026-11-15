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
__aicore__ inline bool KernelLarge<StaticN, StaticK>::BuildAugmented50()
{
auto aug=panel_[22528u];auto order=residualTile_;
auto physicalOffsets=vr_[640u].template Zq6<Zq2>();
auto diagonal=negative64_[4160u];
ZqB<Zq5::V_S>();

ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
auto physicalIndex=physicalOffsets.template Zq6<int32_t>();
CreateVecIndex<int32_t>(physicalIndex,int32_t(0),Zq2(64));Zq1<PIPE_V>();
Muls<int32_t>(physicalIndex,physicalIndex,int32_t(544),Zq2(64));
Duplicate(diagonal,0.0f,64u);Zq1<PIPE_V>();
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
Duplicate<float>(aug[64u],0.0f,Zq3(64),Zq4(64),1,17);
Adds<int32_t>(aug.template Zq6<int32_t>(),producerA_.template Zq6<int32_t>(),int32_t(0),Zq3(64),Zq4(64),UnaryRepeatParams{1,1,17,8});
Zq1<PIPE_V>();


for(Zq2 stripe=0;stripe<8u;++stripe){Zq3 bit[2]={Zq3(1u)<<stripe,0u};
Duplicate<float>(aug[stripe*136u+64u],1.0f,bit,Zq4((50u-stripe+7u)/8u),1,137);}
Zq1<PIPE_V>();


Brcb(block_,negative64_[4224u],Zq4(7),Zq9{1,8});Zq1<PIPE_V>();
Adds<int32_t>(aug[56u].template Zq6<int32_t>(),block_.template Zq6<int32_t>(),int32_t(0),Zq3(1),Zq4(50),UnaryRepeatParams{1,1,17,1});
Zq1<PIPE_V>();
for(Zq2 j=0;j<50u;++j){
ZqB<Zq5::S_V>();Gather(vr_[128u],aug,physicalOffsets,j*4u,64u);Zq1<PIPE_V>();
Abs(vr_,vr_[128u],64u);Zq1<PIPE_V>();
Zq3 active[2]={((Zq3(1u)<<50u)-1u)&(~Zq3(0u)<<j),0u};
WholeReduceMax(vr_[64u],vr_,active,int32_t(1),int32_t(1),int32_t(1),int32_t(8),ReduceOrder::ORDER_VALUE_INDEX);ZqB<Zq5::V_S>();
Zq2 pick=j;float best=AbsF(vr_[128u].GetValue(j));
const Zq2 candidate=vr_[64u].template Zq6<Zq2>().GetValue(1u);
const float maximum=vr_[64u].GetValue(0u);
if(candidate>=j&&candidate<50u&&maximum==maximum)pick=candidate;
else for(Zq2 i=j+1u;i<50u;++i){float v=AbsF(vr_[128u].GetValue(i));if(v>best){best=v;pick=i;}}
float d=vr_[128u].GetValue(pick);
if(pick!=j){
const float t=vr_[128u].GetValue(j);vr_[128u].SetValue(j,d);vr_[128u].SetValue(pick,t);
Adds<int32_t>(rows_.template Zq6<int32_t>(),aug[j*136u].template Zq6<int32_t>(),int32_t(0),Zq3(136));
Zq1<PIPE_V>();
Adds<int32_t>(aug[j*136u].template Zq6<int32_t>(),aug[pick*136u].template Zq6<int32_t>(),int32_t(0),Zq3(136));
Zq1<PIPE_V>();
Adds<int32_t>(aug[pick*136u].template Zq6<int32_t>(),rows_.template Zq6<int32_t>(),int32_t(0),Zq3(136));
Zq1<PIPE_V>();
}
const float mag=AbsF(d);
if(!(mag>=1.0e-30f&&mag<=1.0e30f))return false;
const float inv=SeedInverse(d);diagonal.SetValue(j,inv);
ZqB<Zq5::S_V>();
Zq3 zero[2]={Zq3(1u)<<j,0u};Duplicate<float>(vr_[128u],0.0f,zero,Zq4(1),1,8);
Brcb(block_,vr_[128u],Zq4(8),Zq9{1,8});
Muls(vr_[256u],aug[j*136u],-inv,128u);
const Zq2 aStart=(j/8u)*8u;
const Zq2 aLanes=64u-aStart;
MulAddDst<float,float>(aug[aStart],vr_[256u+aStart],block_,Zq3(aLanes),Zq4(50),Zq0{1,1,0,17,0,1});
MulAddDst<float,float>(aug[64u],vr_[320u],block_,Zq3(64),Zq4(50),Zq0{1,1,0,17,0,1});
}
auto offsets=panel_.template Zq6<int32_t>();
ZqB<Zq5::S_V>();
for(Zq2 c=0;c<64u;++c)Adds<int32_t>(offsets[c*64u],physicalOffsets.template Zq6<int32_t>(),int32_t(256u+c*4u),64u);
Zq1<PIPE_V>();Gather(panel_[8192u],aug,offsets.template Zq6<Zq2>(),0u,4096u);Zq1<PIPE_V>();
Mul(negative64_,panel_[8192u],diagonal,Zq3(64),Zq4(64),Zq0{1,1,1,8,8,0});Zq1<PIPE_V>();
Abs(panel_[8192u],negative64_,4096u);Zq1<PIPE_V>();
for(Zq2 h=2048u;h>=64u;h/=2u){Add(panel_[8192u],panel_[8192u],panel_[8192u+h],h);Zq1<PIPE_V>();}

auto inverseLow=rows_.template Zq6<Zq4>();
auto inverseHigh=rows_[8u].template Zq6<Zq4>();
Compares(inverseLow,panel_[8192u],0.0f,CMPMODE::GE,Zq2(64));
Compares(inverseHigh,panel_[8192u],1.0e30f,CMPMODE::LE,Zq2(64));Zq1<PIPE_V>();
And(inverseLow.template Zq6<uint16_t>(),inverseLow.template Zq6<uint16_t>(),inverseHigh.template Zq6<uint16_t>(),Zq2(4));
ZqB<Zq5::V_S>();
if(inverseLow.template Zq6<Zq3>().GetValue(0u)!=~Zq3(0u))return false;

ZqB<Zq5::S_V>();Gather(solveY_,aug,physicalOffsets,224u,64u);Zq1<PIPE_V>();
Mul(solveY_,solveY_,diagonal,64u);Zq1<PIPE_V>();
Abs(vr_,solveY_,64u);Zq1<PIPE_V>();
auto finite=rows_.template Zq6<Zq4>();
Compares(finite,vr_,3.4e38f,CMPMODE::LE,Zq2(64));ZqB<Zq5::V_S>();


augmented50RhsReady_=finite.template Zq6<Zq3>().GetValue(0u)==~Zq3(0u);
augmented50Ready_=true;return true;
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline Zq2 KernelLarge<StaticN, StaticK>::PivotIndex(Zq7<float>column,Zq2 first)
{
return TriangularSolveOperations::SelectColumnPivot(column,vr_,first,N(),VectorPitch());
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::DecomposeSquare(Zq3 base)
{
if(UseLocalFactors()){DecomposeLocal(base);return;}
if(N()==256u){DecomposeBlocked(base);return;}
const Zq2 n=N();
__gm__ float*lu=w_+OffLU(base);
__gm__ float*piv=w_+OffP(base);
LuFactorizationOperations::FactorScalarSquare(lu, piv, w_+OffInv(base), n);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::DecomposeLocal(Zq3 base,bool skipTried50)
{
if constexpr(StaticN==50u){if(!skipTried50&&BuildAugmented50())return;

ZqB<Zq5::V_S>();
for(Zq2 row=0;row<50u;++row)w_[OffB(base)+row]=negative64_.GetValue(4224u+row);
PrepareFallbackRows50();Rows50ToColumns();ProducerLazySmallLu(base);return;}

if(N()==200u){
DecomposeRows200(base);ArchiveRows200(base);return;
}
if(UseColumnLocal()){DecomposeColumns(base);return;}
const Zq2 n=N();
__gm__ float*piv=w_+OffP(base);
for(Zq2 j=0;j<n;++j){
const Zq2 pick=TriangularSolveOperations::SelectLocalPivot(*this,base,j,n);
piv[j]=static_cast<float>(static_cast<int32_t>(pick));
if(pick!=j){
ZqB<Zq5::S_V>();
Adds(rows_,factors_[j*FactorPitch()],0.0f,FactorPitch());
Zq1<PIPE_V>();
Adds(factors_[j*FactorPitch()],factors_[pick*FactorPitch()],0.0f,FactorPitch());
Zq1<PIPE_V>();
Adds(factors_[pick*FactorPitch()],rows_,0.0f,FactorPitch());
ZqB<Zq5::V_S>();
}
const float diagonal=Factor(base,j,j);
if(diagonal==0.0f){w_[OffInv(base)+j]=0.0f;continue;}
const bool safe=AbsF(diagonal)>=1.0e-30f&&AbsF(diagonal)<=1.0e30f;
const float inverse=safe?square_math::Divide(1.0f,diagonal):0.0f;
w_[OffInv(base)+j]=inverse;
if(j+1u==n)continue;
for(Zq2 c=0;c<FactorPitch();++c)
rows_.SetValue(c,c>j&&c<n?Factor(base,j,c):0.0f);

for(Zq2 i=j+1;i<n;++i)
PutFactor(base,i,j,safe?square_math::RoundMul(Factor(base,i,j),inverse):square_math::Divide(Factor(base,i,j),diagonal));
ZqB<Zq5::S_V>();


SetMaskCount();
ZqA<float,MaskMode::COUNTER>(0u,FactorPitch());
for(Zq2 i=j+1;i<n;++i){
const float mult=Factor(base,i,j);
Axpy<float,float,false>(factors_[i*FactorPitch()],rows_,-mult,
Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});
}
ResetMask();SetMaskNorm();

ZqB<Zq5::V_S>();
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline bool KernelLarge<StaticN, StaticK>::SelectColumnPivot(
    Zq3 base, Zq2 j, Zq2 n, Zq2 pitch, float& diagonal, float& inverse, bool& safe)
{
const Zq2 pick=PivotIndex(factors_[j*pitch],j);
w_[OffP(base)+j]=float(int32_t(pick));
if(pick!=j){
const Zq2 p=VectorPitch();
auto offsets=vr_[10u*p].template Zq6<Zq2>();
ZqB<Zq5::S_V>();
Gather(vr_,factors_,offsets,j*4u,n);
Gather(vr_[p],factors_,offsets,pick*4u,n);
Zq1<PIPE_V>();
Brcb(vr_[2u*p],vr_,Zq4(pitch/8u),Zq9{1,8});
Brcb(block_,vr_[p],Zq4(pitch/8u),Zq9{1,8});
Zq1<PIPE_V>();
Zq3 maskJ[2]={Zq3(1)<<(j%8u),0u};
Zq3 maskP[2]={Zq3(1)<<(pick%8u),0u};
Adds(factors_[(j/8u)*8u],block_,0.0f,maskJ,Zq4(n),
UnaryRepeatParams{1,1,Zq4(pitch/8u),1});
Zq1<PIPE_V>();
Adds(factors_[(pick/8u)*8u],vr_[2u*p],0.0f,maskP,Zq4(n),
UnaryRepeatParams{1,1,Zq4(pitch/8u),1});
ResetMask();ZqB<Zq5::V_S>();
}
diagonal=factors_.GetValue(j*pitch+j);
if(diagonal==0.0f){w_[OffInv(base)+j]=0.0f;return false;}
safe=AbsF(diagonal)>=1.0e-30f&&AbsF(diagonal)<=1.0e30f;
inverse=safe?SeedInverse(diagonal):0.0f;
w_[OffInv(base)+j]=inverse;
return j+1u!=n;
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::DecomposeSmallMasked(Zq3 base)
{
const Zq2 n=N(),pitch=FactorPitch();
for(Zq2 j=0;j<n;++j){
float diagonal, inverse;
bool safe;
if (!SelectColumnPivot(base, j, n, pitch, diagonal, inverse, safe)) continue;
const Zq2 start=(j+1u)/8u*8u,count=pitch-start;
const Zq2 head=j+1u-start;
const Zq3 low=count==64u?~Zq3(0u):(Zq3(1u)<<count)-1u;
Zq3 valid[2]={low&(~Zq3(0u)<<head),0u};
if(safe){
ZqB<Zq5::S_V>();
Muls<float>(factors_[j*pitch+start],factors_[j*pitch+start],inverse,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});
ResetMask();ZqB<Zq5::V_S>();
}else for(Zq2 i=j+1u;i<n;++i)
factors_.SetValue(j*pitch+i,square_math::Divide(factors_.GetValue(j*pitch+i),diagonal));
ZqB<Zq5::S_V>();


const Zq2 p=VectorPitch();
auto offsets=vr_[10u*p].template Zq6<Zq2>();
Gather(vr_[8u*p],factors_,offsets,j*4u,n);
Zq1<PIPE_V>();
Muls(vr_[8u*p],vr_[8u*p],-1.0f,n);
Zq1<PIPE_V>();
Brcb(block_,vr_[8u*p],Zq4(pitch/8u),Zq9{1,8});
Zq1<PIPE_V>();


MulAddDst<float,float>(factors_[(j+1u)*pitch+start],factors_[j*pitch+start],block_[(j+1u)*8u],
valid,Zq4(n-j-1u),Zq0{1,1,0,Zq4(pitch/8u),0,1});
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::DecomposeColumns(Zq3 base)
{
if(N()==50u||N()==64u){DecomposeSmallMasked(base);return;}
const Zq2 n=N(),pitch=FactorPitch();
for(Zq2 j=0;j<n;++j){
float diagonal, inverse;
bool safe;
if (!SelectColumnPivot(base, j, n, pitch, diagonal, inverse, safe)) continue;
const Zq2 start=(j+1u)/8u*8u,count=pitch-start;
for(Zq2 i=start;i<=j;++i)rows_.SetValue(i-start,factors_.GetValue(j*pitch+i));
if(safe){
ZqB<Zq5::S_V>();
Muls(factors_[j*pitch+start],factors_[j*pitch+start],inverse,count);
ZqB<Zq5::V_S>();
}else for(Zq2 i=j+1u;i<n;++i)
factors_.SetValue(j*pitch+i,square_math::Divide(factors_.GetValue(j*pitch+i),diagonal));
for(Zq2 i=start;i<=j;++i)factors_.SetValue(j*pitch+i,0.0f);
ZqB<Zq5::S_V>();


if(n==200u||n==50u||n==64u){


const Zq2 p=VectorPitch();
auto offsets=vr_[10u*p].template Zq6<Zq2>();
Gather(vr_[8u*p],factors_,offsets,j*4u,n);
Zq1<PIPE_V>();
Muls(vr_[8u*p],vr_[8u*p],-1.0f,n);
Zq1<PIPE_V>();
Brcb(block_,vr_[8u*p],Zq4(pitch/8u),Zq9{1,8});
Zq1<PIPE_V>();


for(Zq2 col=j+1u;col<n;col+=200u){
const Zq2 columns=n-col<200u?n-col:200u;
for(Zq2 row=0;row<count;row+=64u){
const Zq2 width=count-row<64u?count-row:64u;
MulAddDst<float,float>(factors_[col*pitch+start+row],factors_[j*pitch+start+row],block_[col*8u],
Zq3(width),Zq4(columns),Zq0{1,1,0,Zq4(pitch/8u),0,1});
}
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
}else{
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,count);
for(Zq2 col=j+1u;col<n;++col){
const float u=factors_.GetValue(col*pitch+j);
Axpy<float,float,false>(factors_[col*pitch+start],factors_[j*pitch+start],-u,
Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
}
for(Zq2 i=start;i<=j;++i)factors_.SetValue(j*pitch+i,rows_.GetValue(i-start));
}
}
