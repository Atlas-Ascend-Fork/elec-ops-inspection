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
__aicore__ inline void KernelFastContext50<StaticN, StaticK>::ConsumeAugmented50(const __gm__ float*in,Zq2 stride)
{
if(!producerCorrection_&&augmented50RhsReady_)return;
if(!producerCorrection_){Adds(solveY_,negative64_[4224u],0.0f,64u);Zq1<PIPE_V>();}
ResetMask();SetMaskNorm();
Zq3 rhsPad[2]={~((Zq3(1u)<<50u)-1u),0u};
Duplicate<float>(solveY_,0.0f,rhsPad,Zq4(1),1,8);Zq1<PIPE_V>();
auto mapBytes=vr_[640u].template Zq6<Zq2>();
Gather(vr_[128u],solveY_,mapBytes,0u,64u);Zq1<PIPE_V>();
Muls(vr_[128u],vr_[128u],normalizationScale_,64u);Zq1<PIPE_V>();
if(directRows50_){


Mul(negative64_,panel_[22528u],vr_[128u],Zq3(64),Zq4(50),Zq0{1,1,1,8,9,0});
Duplicate(vr_,0.0f,64u);Zq1<PIPE_V>();
WholeReduceSum(vr_,negative64_,int32_t(64),int32_t(50),1,1,8);Zq1<PIPE_V>();
Gather(solveY_,vr_,mapBytes,0u,64u);Zq1<PIPE_V>();
return;
}
Brcb(block_,vr_[128u],Zq4(8),Zq9{1,8});Zq1<PIPE_V>();

auto products=negative64_;
Mul(products,panel_[22528u],block_,Zq3(64),Zq4(50),Zq0{1,1,0,8,9,1});
Duplicate(products[3200u],0.0f,896u);Zq1<PIPE_V>();
for(Zq2 h=2048u;h>=64u;h/=2u){Add(products,products,products[h],h);Zq1<PIPE_V>();}
Gather(solveY_,products,mapBytes,0u,64u);Zq1<PIPE_V>();
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelFastContext50<StaticN, StaticK>::RowMajorResidual50All64(Zq3 base)
{
constexpr Zq2 p=3200u;
auto a=producerA_,ah=panel_,al=panel_[p],hi=panel_[2u*p],lo=panel_[3u*p],t=panel_[4u*p],u=t;
ResidualOperations::RowMajorResidual50Products(*this, a, ah, al, hi, lo, t);
ResidualOperations::RowMajorProducts<50u>(a,ah,al,hi,lo,t,vr_);


Add<float>(panel_,hi,hi[32u],Zq3(32),Zq4(50),Zq0{50,1,1,1,8,8});
Add<float>(panel_[1600u],lo,lo[32u],Zq3(32),Zq4(50),Zq0{50,1,1,1,8,8});Zq1<PIPE_V>();
Sub<float>(u[32u],panel_,hi,Zq3(32),Zq4(50),Zq0{1,50,1,8,1,8});Zq1<PIPE_V>();
Sub<float>(u,panel_,u[32u],Zq3(32),Zq4(50),Zq0{1,50,1,8,1,8});Zq1<PIPE_V>();
Sub<float>(u,hi,u,Zq3(64),Zq4(50),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float>(u,u,u[32u],Zq3(32),Zq4(50),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float>(panel_[1600u],panel_[1600u],u,Zq3(32),Zq4(50),Zq0{50,50,1,1,1,8});Zq1<PIPE_V>();

Add(panel_[6400u],panel_,panel_[800u],Zq2(800));
Add(panel_[7200u],panel_[1600u],panel_[2400u],Zq2(800));Zq1<PIPE_V>();
Sub(u[800u],panel_[6400u],panel_,Zq2(800));Zq1<PIPE_V>();
Sub(u,panel_[6400u],u[800u],Zq2(800));Zq1<PIPE_V>();
Sub(u,panel_,u,Zq2(1600));Zq1<PIPE_V>();
Add(u,u,u[800u],Zq2(800));Zq1<PIPE_V>();
Add(panel_[7200u],panel_[7200u],u,Zq2(800));Zq1<PIPE_V>();

Add(panel_,panel_[6400u],panel_[6800u],Zq2(400));
Add(panel_[400u],panel_[7200u],panel_[7600u],Zq2(400));Zq1<PIPE_V>();
Sub(u[400u],panel_,panel_[6400u],Zq2(400));Zq1<PIPE_V>();
Sub(u,panel_,u[400u],Zq2(400));Zq1<PIPE_V>();
Sub(u,panel_[6400u],u,Zq2(800));Zq1<PIPE_V>();
Add(u,u,u[400u],Zq2(400));Zq1<PIPE_V>();
Add(panel_[400u],panel_[400u],u,Zq2(400));Zq1<PIPE_V>();
ah=panel_;lo=panel_[400u];

Duplicate(panel_[2u*p],0.0f,Zq2(1024));Zq1<PIPE_V>();
auto offsets=panel_[14u*2048u].template Zq6<Zq2>();
hi=panel_[2u*p];
Gather(hi,panel_,offsets,0u,Zq3(50),Zq4(16),uint16_t(8));Zq1<PIPE_V>();
al=panel_[p];

Add<float>(panel_,panel_[6400u],panel_[6912u],Zq3(64),Zq4(8),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(u[512u],panel_,panel_[6400u],Zq3(64),Zq4(4),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float>(u,panel_,u[512u],Zq3(64),Zq4(4),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float>(u,panel_[6400u],u,Zq3(64),Zq4(8),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(u,u,u[512u],Zq3(64),Zq4(4),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(panel_[64u],panel_[64u],u,Zq3(64),Zq4(4),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();

Add<float>(panel_[6400u],panel_,panel_[256u],Zq3(64),Zq4(4),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(u[256u],panel_[6400u],panel_,Zq3(64),Zq4(2),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float>(u,panel_[6400u],u[256u],Zq3(64),Zq4(2),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float>(u,panel_,u,Zq3(64),Zq4(4),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(u,u,u[256u],Zq3(64),Zq4(2),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(panel_[6464u],panel_[6464u],u,Zq3(64),Zq4(2),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();

Add<float>(panel_,panel_[6400u],panel_[6528u],Zq3(64),Zq4(2),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(u[128u],panel_,panel_[6400u],Zq3(64),Zq4(1),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float>(u,panel_,u[128u],Zq3(64),Zq4(1),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float>(u,panel_[6400u],u,Zq3(64),Zq4(2),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(u,u,u[128u],Zq3(64),Zq4(1),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(panel_[64u],panel_[64u],u,Zq3(64),Zq4(1),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
lo=panel_[64u];

ResetMask();SetMaskNorm();


auto bb=negative64_[4224u],neg=panel_[2u*p],ss=panel_[p],vv=panel_,tt=panel_[4u*p];
WriteCompensatedResidual(solveY_, bb, ah, lo, neg, ss, vv, tt, 50u);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelFastContext50<StaticN, StaticK>::PrepareResidualX(Zq3 base,Zq2 col)
{
const Zq2 p=VectorPitch();
Zq7<float>x=vr_[p],xh=vr_[3u*p],xl=vr_[5u*p];
const bool privateVectorInput=(StaticN==50u&&StaticK==2u)&&producerPrivate_;
if(!privateVectorInput)ZqB<Zq5::V_S>();
{

{if(producerPrivate_){Adds(x,negative64_[4096u],0.0f,64u);Zq1<PIPE_V>();}
else for(Zq2 i=0;i<N();++i)x.SetValue(i,GetX(base,i,col));}
}
if(!privateVectorInput)ZqB<Zq5::S_V>();
And(xh.template Zq6<uint16_t>(),x.template Zq6<uint16_t>(),
vr_[ResidualMaskPlane()*p].template Zq6<uint16_t>(),2u*p);
Zq1<PIPE_V>();
Sub(xl,x,xh,p);
Zq1<PIPE_V>();
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline bool KernelFastContext50<StaticN, StaticK>::BuildAugmented50()
{
auto aug=panel_[22528u];
auto map=residualTile_;
auto columnOffsets=vr_[640u].template Zq6<int32_t>();
auto indices=panel_.template Zq6<int32_t>();
auto rowBytes=rows_.template Zq6<int32_t>();
auto colBytes=rows_[64u].template Zq6<int32_t>();
ResetMask();SetMaskNorm();

Muls<int32_t>(colBytes,rowBytes,int32_t(4),Zq2(64));Zq1<PIPE_V>();
Muls<int32_t>(rowBytes,rowBytes,int32_t(256),Zq2(64));Zq1<PIPE_V>();
Brcb(block_.template Zq6<int32_t>(),colBytes,Zq4(8),Zq9{1,8});Zq1<PIPE_V>();


auto denseRowOffsets=rows_[128u].template Zq6<int32_t>();
ShiftRight<int32_t>(denseRowOffsets,rowBytes,int32_t(3),int32_t(64));Zq1<PIPE_V>();
Add<int32_t>(panel_[14u*2048u].template Zq6<int32_t>(),denseRowOffsets,block_.template Zq6<int32_t>(),
Zq3(64),Zq4(8),Zq0{1,1,0,16,0,1});Zq1<PIPE_V>();
Adds<int32_t>(panel_[14u*2048u+64u].template Zq6<int32_t>(),panel_[14u*2048u].template Zq6<int32_t>(),int32_t(400u*4u),Zq3(64),Zq4(8),UnaryRepeatParams{1,1,16,16});
Adds<int32_t>(producerA_.template Zq6<int32_t>(),aug.template Zq6<int32_t>(),int32_t(0),
Zq3(56),Zq4(50),UnaryRepeatParams{1,1,8,9});Zq1<PIPE_V>();

Abs(panel_,producerA_,3200u);Zq1<PIPE_V>();
WholeReduceMax(panel_[4096u],panel_,int32_t(64),int32_t(50),int32_t(1),int32_t(1),int32_t(8),ReduceOrder::ORDER_ONLY_VALUE);Zq1<PIPE_V>();
WholeReduceMax(vr_[64u],panel_[4096u],int32_t(50),int32_t(1),int32_t(1),int32_t(1),int32_t(8),ReduceOrder::ORDER_ONLY_VALUE);ZqB<Zq5::V_S>();
const float inputMaximum=vr_[64u].GetValue(0u);
if(!(inputMaximum>=1.0e-30f&&inputMaximum<=1.0e30f))return false;
const Zq2 maximumBits=vr_[64u].template Zq6<Zq2>().GetValue(0u);
const int32_t exponent=int32_t((maximumBits>>23u)&255u)-127;
const int32_t dominantShift=-80-exponent;
const bool dominantMode=dominantShift>=-120&&dominantShift<=120;
const int32_t shift=dominantMode?dominantShift:-8-exponent;
if(shift<-120||shift>120)return false;
union{Zq2 u;float f;}scaleWord,unscaleWord;
scaleWord.u=Zq2(127+shift)<<23u;unscaleWord.u=Zq2(127-shift)<<23u;
normalizationScale_=scaleWord.f;inverseNormalizationScale_=unscaleWord.f;
ZqB<Zq5::S_V>();


Muls<float>(aug,aug,normalizationScale_,Zq3(56),Zq4(50),UnaryRepeatParams{1,1,9,9});

ShiftRight<int32_t>(map.template Zq6<int32_t>(),colBytes,int32_t(2),Zq2(64));
Muls<int32_t>(columnOffsets,colBytes,int32_t(72),Zq2(64));Zq1<PIPE_V>();


Brcb(block_,negative64_[4224u],Zq4(8),Zq9{1,8});Zq1<PIPE_V>();
Zq3 rhsLane[2]={Zq3(1u)<<2u,0u};
Muls<float>(aug[48u],block_,normalizationScale_,rhsLane,Zq4(64),UnaryRepeatParams{1,1,9,1});Zq1<PIPE_V>();
if(!dominantMode){
CanonicalizeRowMajor50();
ZqB<Zq5::S_V>();
Adds<int32_t>(indices,columnOffsets,int32_t(0),Zq3(64),Zq4(50),UnaryRepeatParams{1,1,8,0});
Duplicate<float>(aug[51u*72u],1.0f,Zq2(64));Zq1<PIPE_V>();
Zq3 descriptorDiagonal[2]={0x8040201008040201ull,0u};
Duplicate<int32_t>(indices,int32_t(51u*72u*4u),descriptorDiagonal,Zq4(6),Zq4(8),Zq4(65));
Zq3 descriptorTail[2]={0x201ull,0u};
Duplicate<int32_t>(indices[48u*65u],int32_t(51u*72u*4u),descriptorTail,Zq4(1),Zq4(8),Zq4(65));Zq1<PIPE_V>();
}
if(dominantMode){


auto chosenHistory=factors_[1536u];
auto chosenIndexOffsets=factors_[1088u].template Zq6<int32_t>();
Muls<int32_t>(chosenIndexOffsets,colBytes,int32_t(8),Zq2(64));Zq1<PIPE_V>();
Adds<int32_t>(chosenIndexOffsets,chosenIndexOffsets,int32_t(4),Zq2(64));Zq1<PIPE_V>();

SetMaskNorm();


const int64_t originalExponentOffset=-int64_t(shift)*int64_t(8388608u);
const int64_t rawLower=int64_t(0x0da24260u)-originalExponentOffset;
const int64_t rawUpper=int64_t(0x7149f2cau)-originalExponentOffset;
const int64_t clippedLower=rawLower<int64_t(0x00800000u)?int64_t(0x00800000u):rawLower;
const int64_t clippedUpper=rawUpper>int64_t(0x7f7fffffu)?int64_t(0x7f7fffffu):rawUpper;
const bool normalInterval=clippedLower<=clippedUpper;
const Zq2 normalLower=normalInterval?Zq2(clippedLower):Zq2(0xffffffffu);
const Zq2 normalWidth=normalInterval?Zq2(clippedUpper-clippedLower):Zq2(0u);


const int64_t interiorLower=clippedLower<int64_t(0x03800000u)?int64_t(0x03800000u):clippedLower;
const int64_t interiorUpper=clippedUpper>int64_t(0x25800000u)?int64_t(0x25800000u):clippedUpper;
const bool interiorInterval=interiorLower<=interiorUpper;
const Zq2 interiorLow=interiorInterval?Zq2(interiorLower):Zq2(0xffffffffu);
const Zq2 interiorWidth=interiorInterval?Zq2(interiorUpper-interiorLower):Zq2(0u);

Zq3 available=(Zq3(1u)<<50u)-1u;
for(Zq2 j=0u;j<50u;++j){


Gather(vr_[128u],aug,columnOffsets.template Zq6<Zq2>(),j*4u,Zq3(64),Zq4(1),uint16_t(8));Zq1<PIPE_V>();

Abs<float,false>(vr_,vr_[128u],Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});Zq1<PIPE_V>();
Zq3 active[2]={available,0u};
WholeReduceMax(chosenHistory[j*8u],vr_,active,int32_t(1),int32_t(1),int32_t(1),int32_t(8),ReduceOrder::ORDER_VALUE_INDEX);ZqB<Zq5::V_S>();

const Zq3 selection=chosenHistory[j*8u].template Zq6<Zq3>().GetValue(0u);
const Zq2 physical=Zq2(selection>>32u);
const Zq2 maximumBits=Zq2(selection);


if(physical>=64u||((maximumBits&0x7fffffffu)>0x7f800000u))return false;


available&=~(Zq3(1u)<<physical);


const float actualPivot=vr_[128u].GetValue(physical);
aug.SetValue(physical*72u+j,1.0f);
vr_[128u].SetValue(physical,-1.0f);
ZqB<Zq5::S_V>();
const float inverse=SeedInverse(actualPivot);


if(Zq2(maximumBits-interiorLow)>interiorWidth){
union{float f;Zq2 u;}pivotWord,inverseWord;
pivotWord.f=actualPivot;inverseWord.f=inverse;
const Zq2 pivotBits=pivotWord.u&Zq2(0x7fffffffu);
const Zq2 inverseBits=inverseWord.u&Zq2(0x7fffffffu);
bool pivotCertified=false;
if(pivotBits>=Zq2(0x00800000u)&&pivotBits<Zq2(0x7f800000u)){


pivotCertified=Zq2(pivotBits-normalLower)<=normalWidth;
}else{

const float pivotMagnitude=AbsF(actualPivot)*inverseNormalizationScale_;
pivotCertified=pivotMagnitude>=1.0e-30f&&pivotMagnitude<=1.0e30f;
}


if(!(pivotCertified&&Zq2(inverseBits-Zq2(0x58800000u))<=
Zq2(0x7f7fc99eu-0x58800000u)))return false;
}


Muls(vr_[272u],aug[physical*72u],-inverse,Zq3(56));
Brcb(block_[40u],vr_[128u],Zq4(7),Zq9{1,8});Zq1<PIPE_V>();


MulAddDst<float,float>(aug,vr_[272u],block_[40u],Zq3(56),Zq4(50),Zq0{1,1,0,9,0,1});Zq1<PIPE_V>();
}
if(available!=0u)return false;


Gather(map,chosenHistory.template Zq6<Zq2>(),
chosenIndexOffsets.template Zq6<Zq2>(),0u,50u);Zq1<PIPE_V>();
}else{
Zq3 available=(Zq3(1u)<<50u)-1u;
for(Zq2 j=0u;j<50u;++j){
ZqB<Zq5::S_V>();
Adds<int32_t>(vr_[128u].template Zq6<int32_t>(),aug[j*72u].template Zq6<int32_t>(),int32_t(0),Zq2(64));Zq1<PIPE_V>();
Abs(vr_,vr_[128u],64u);Zq1<PIPE_V>();
Zq3 active[2]={available,0u};
WholeReduceMax(vr_[64u],vr_,active,int32_t(1),int32_t(1),int32_t(1),int32_t(8),ReduceOrder::ORDER_VALUE_INDEX);ZqB<Zq5::V_S>();
const Zq2 physical=vr_[64u].template Zq6<Zq2>().GetValue(1u);
const float maximum=vr_[64u].GetValue(0u);


if(physical>=50u||!((available>>physical)&1u)||!(maximum==maximum))return false;
const float d=vr_[128u].GetValue(physical);const float magnitude=AbsF(d)*inverseNormalizationScale_;
if(!(magnitude>=1.0e-30f&&magnitude<=1.0e30f))return false;


map.SetValue(j,physical);
available&=~(Zq3(1u)<<physical);
const float inv=SeedInverse(d);const bool exactRow=AbsF(inv)<1.0f;
vr_[128u].SetValue(physical,exactRow?-1.0f:(d-1.0f));
ZqB<Zq5::S_V>();
Gather(vr_[256u],aug,indices[j*64u].template Zq6<Zq2>(),physical*4u,64u);Zq1<PIPE_V>();
Muls(vr_[256u],vr_[256u],-inv,56u);Zq1<PIPE_V>();
Zq3 rowLane[2]={Zq3(1u)<<physical,0u};
Brcb(block_,vr_[256u],Zq4(7),Zq9{1,8});Zq1<PIPE_V>();
if(exactRow){Duplicate<float>(aug,0.0f,rowLane,Zq4(51),1,9);ResetMask();SetMaskNorm();}
Duplicate(aug[j*72u],0.0f,56u);Zq1<PIPE_V>();
MulAddDst<float,float>(aug,vr_[128u],block_,Zq3(56),Zq4(51),Zq0{1,1,0,9,0,1});Zq1<PIPE_V>();
Duplicate<float>(aug[j*72u],inv,rowLane,Zq4(1),1,8);Zq1<PIPE_V>();
}
}

directRows50_=dominantMode;


ResetMask();SetMaskNorm();
Muls<int32_t>(columnOffsets,map.template Zq6<int32_t>(),int32_t(4),Zq2(64));Zq1<PIPE_V>();
if(directRows50_){


auto physicalRows=factors_[1088u].template Zq6<int32_t>();
Muls<int32_t>(physicalRows,map.template Zq6<int32_t>(),int32_t(288),Zq2(64));Zq1<PIPE_V>();
Gather(solveY_,aug,physicalRows.template Zq6<Zq2>(),200u,64u);Zq1<PIPE_V>();

Zq3 bLane[2]={Zq3(1u)<<2u,0u};
Duplicate<float>(aug[48u],0.0f,bLane,Zq4(64),Zq4(1),Zq4(9));Zq1<PIPE_V>();
Abs(panel_[8192u],aug,Zq3(64),Zq4(50),UnaryRepeatParams{1,1,8,9});
Duplicate(vr_,0.0f,64u);Zq1<PIPE_V>();
WholeReduceSum(vr_,panel_[8192u],int32_t(64),int32_t(50),1,1,8);Zq1<PIPE_V>();
Adds(panel_[8192u],vr_,0.0f,64u);Zq1<PIPE_V>();
}else{
Gather(solveY_,aug,columnOffsets.template Zq6<Zq2>(),50u*72u*4u,64u);Zq1<PIPE_V>();

Duplicate(aug[50u*72u],0.0f,64u);Zq1<PIPE_V>();
Abs(panel_[8192u],aug,Zq3(64),Zq4(50),UnaryRepeatParams{1,1,8,9});
Duplicate(panel_[8192u+3200u],0.0f,896u);Zq1<PIPE_V>();
for(Zq2 h=2048u;h>=64u;h/=2u){Add(panel_[8192u],panel_[8192u],panel_[8192u+h],h);Zq1<PIPE_V>();}
}

Muls(panel_[8192u],panel_[8192u],normalizationScale_*1.000003814697265625f,64u);Zq1<PIPE_V>();

auto inverseLow=rows_.template Zq6<Zq4>();
auto inverseHigh=rows_[8u].template Zq6<Zq4>();
Compares(inverseLow,panel_[8192u],0.0f,CMPMODE::GE,Zq2(64));
Compares(inverseHigh,panel_[8192u],1.0e30f,CMPMODE::LE,Zq2(64));Zq1<PIPE_V>();
And(inverseLow.template Zq6<uint16_t>(),inverseLow.template Zq6<uint16_t>(),inverseHigh.template Zq6<uint16_t>(),Zq2(4));


Abs(vr_,solveY_,64u);Zq1<PIPE_V>();
auto finite=rows_[16u].template Zq6<Zq4>();
Compares(finite,vr_,3.4e38f,CMPMODE::LE,Zq2(64));ZqB<Zq5::V_S>();
if(inverseLow.template Zq6<Zq3>().GetValue(0u)!=~Zq3(0u))return false;
augmented50RhsReady_=finite.template Zq6<Zq3>().GetValue(0u)==~Zq3(0u);
if(!augmented50RhsReady_)return false;
augmented50Ready_=true;return true;
}
