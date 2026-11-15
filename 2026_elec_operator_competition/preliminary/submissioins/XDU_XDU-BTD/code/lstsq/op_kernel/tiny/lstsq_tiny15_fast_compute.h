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

template<Zq2 TinyN>
__aicore__ inline void KernelTinyFastContext15<TinyN>::ConsumeTinyRawInverse15()
{
SetMaskNorm();ResetMask();
auto products=panel_[13u*256u];

Mul<float>(products,factors_[16u],solveY_,Zq3(16),Zq4(16),Zq0{1,1,1,2,4,0});Zq1<PIPE_V>();

Zq3 activeProducts[2]={0x7fff7fff7fff7fffull,0u};
BlockReduceSum<float>(block_,products,int32_t(4),activeProducts,int32_t(1),int32_t(1),int32_t(8));Zq1<PIPE_V>();
PairReduceSum<float>(vr_[64u],block_,int32_t(1),int32_t(32),int32_t(1),int32_t(1),int32_t(8));Zq1<PIPE_V>();
Mul<float>(solveY_,vr_[64u],vr_[128u],Zq3(16),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
}

template<Zq2 TinyN>
__aicore__ inline void KernelTinyFastContext15<TinyN>::ConsumeTinyInverse()
{
{Zq3 pad[2]={0x8000u,0u};Duplicate<float>(solveY_,0.0f,pad,Zq4(1),1,8);Zq1<PIPE_V>();}
SetMaskNorm();ResetMask();
Brcb(block_,solveY_,Zq4(2),Zq9{1,8});Zq1<PIPE_V>();
auto products=panel_[13u*256u];
Mul(products,factors_,block_,Zq3(16),Zq4(16),Zq0{1,1,0,2,2,1});Zq1<PIPE_V>();
Add<float>(products,products,products[128u],Zq3(64),Zq4(2),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(products,products,products[64u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float>(products,products,products[32u],Zq3(32),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float>(products,products,products[16u],Zq3(16),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Adds<float,false>(solveY_,products,0.0f,Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});Zq1<PIPE_V>();
}

__aicore__ static inline float TinyNormElementBound(float inputScale, bool fusedResidual)
{
float bound=3.9614081257132168796771975168e28f;
if(fusedResidual){
union{float f;Zq2 u;}maximum,cap;maximum.f=inputScale;
int32_t power=35-(int32_t((maximum.u>>23u)&255u)-127);if(power>20)power=20;
cap.u=Zq2(127+power-4)<<23u;bound=cap.f;
}
return bound;
}

template<Zq2 TinyN>
__aicore__ inline bool KernelTinyFastContext15<TinyN>::BuildTinyInverseVector15()
{
constexpr Zq2 n=TinyN;

auto pivotDiagonals=vr_[16u];


SetMaskNorm();ResetMask();
auto coefficientMap=panel_[11u*256u].template Zq6<Zq2>();

Adds<int32_t>(coefficientMap.template Zq6<int32_t>(),panel_[16u*256u].template Zq6<int32_t>(),int32_t(-128),Zq2(256));Zq1<PIPE_V>();

Gather(factors_,panel_[14u*256u],panel_.template Zq6<Zq2>(),0u,768u);Zq1<PIPE_V>();
Abs(panel_[13u*256u],factors_[512u],256u);Zq1<PIPE_V>();


WholeReduceMax(panel_[12u*256u],panel_[13u*256u],int32_t(64),int32_t(4),int32_t(1),int32_t(1),int32_t(8),ReduceOrder::ORDER_VALUE_INDEX);Zq1<PIPE_V>();
Zq3 maxValues[2]={0x55u,0u};
WholeReduceMax(vr_[16u],panel_[12u*256u],maxValues,int32_t(1),int32_t(1),int32_t(1),int32_t(8),ReduceOrder::ORDER_VALUE_INDEX);
ZqB<Zq5::V_S>();
const float inputScale=vr_[16u].GetValue(0u);
tinyFusedResidual_=inputScale>0.0f&&inputScale<=0x1p36f;
if(!tinyFusedResidual_)return false;

{
constexpr Zq2 p=256u,active=p;
auto a=factors_[512u],ah=panel_[4u*p],al=panel_[5u*p];
Duplicate(panel_[10u*p].template Zq6<Zq2>(),Zq2(0xfffff000u),p);Zq1<PIPE_V>();
SetMaskCount();ZqA<uint16_t,MaskMode::COUNTER>(0u,2u*active);
And<uint16_t,false>(ah.template Zq6<uint16_t>(),a.template Zq6<uint16_t>(),
panel_[10u*p].template Zq6<uint16_t>(),Zq3(0),Zq4(1),Zq0{1,1,1,8,8,0});
Zq1<PIPE_V>();ZqA<float,MaskMode::COUNTER>(0u,active);
Sub<float,false>(al,a,ah,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
ResetMask();SetMaskNorm();
}

Zq3 paddingPivot[2]={0x8000u,0u};
Duplicate<float>(pivotDiagonals,1.0f,paddingPivot,Zq4(1),1,8);
const float normElementBound=TinyNormElementBound(inputScale,tinyFusedResidual_);

if(!(inputScale>=0.0f&&inputScale<=1.0e30f*8192.0f))return false;
union{float f;Zq2 u;}pivotLower;
pivotLower.f=inputScale*(1.0f/8192.0f);
if(pivotLower.f<1.0e-30f)pivotLower.f=1.0e-30f;
constexpr Zq2 pivotUpperBits=0x7149f2cau;
if(pivotLower.u>pivotUpperBits)return false;


Zq3 diagonal0[2]={0x0010000800040002ull,0u};
Zq3 diagonal1[2]={0x0100008000400020ull,0u};
Zq3 diagonal2[2]={0x1000080004000200ull,0u};
Zq3 diagonal3[2]={0x0000800040002000ull,0u};
Duplicate<Zq2>(coefficientMap,Zq2(480u*4u),diagonal0,Zq4(1),1u,8u);
Duplicate<Zq2>(coefficientMap[64u],Zq2(480u*4u),diagonal1,Zq4(1),1u,8u);
Duplicate<Zq2>(coefficientMap[128u],Zq2(480u*4u),diagonal2,Zq4(1),1u,8u);
Duplicate<Zq2>(coefficientMap[192u],Zq2(480u*4u),diagonal3,Zq4(1),1u,8u);Zq1<PIPE_V>();

Brcb(block_.template Zq6<int32_t>(),tinyNegative_.template Zq6<int32_t>(),Zq4(2),Zq9{1,8});Zq1<PIPE_V>();
Zq3 pivotEntry[2]={0x1u,0u};
Adds<int32_t>(coefficientMap.template Zq6<int32_t>(),block_.template Zq6<int32_t>(),int32_t(0),pivotEntry,Zq4(16),UnaryRepeatParams{1,1,2,1});Zq1<PIPE_V>();

Brcb(panel_[6u*256u].template Zq6<int32_t>(),coefficientMap.template Zq6<int32_t>(),Zq4(16),Zq9{1,8});
Brcb(panel_[12u*256u].template Zq6<int32_t>(),coefficientMap[128u].template Zq6<int32_t>(),Zq4(8),Zq9{1,8});
Brcb(panel_[15u*256u].template Zq6<int32_t>(),coefficientMap[192u].template Zq6<int32_t>(),Zq4(4),Zq9{1,8});
Brcb(panel_[640u].template Zq6<int32_t>(),coefficientMap[224u].template Zq6<int32_t>(),Zq4(2),Zq9{1,8});Zq1<PIPE_V>();

#pragma unroll 15
for(Zq2 j=0;j<n;++j){


auto coefficients=panel_[j*32u];
Zq7<Zq2>expandedOffsets;
if(j<8u)expandedOffsets=panel_[6u*256u+j*128u].template Zq6<Zq2>();
else if(j<12u)expandedOffsets=panel_[12u*256u+(j-8u)*128u].template Zq6<Zq2>();
else if(j<14u)expandedOffsets=panel_[15u*256u+(j-12u)*128u].template Zq6<Zq2>();
else expandedOffsets=panel_[640u].template Zq6<Zq2>();
Gather(coefficients,factors_,expandedOffsets,0u,128u);Zq1<PIPE_V>();
Div<float>(vr_[128u],factors_[j*32u],coefficients,Zq3(32),Zq4(1),Zq0{1,1,0,8,8,0});

Muls(coefficients[8u],coefficients[8u],-1.0f,Zq2(120));
Zq1<PIPE_V>();
Zq3 columns[2]={0xffffffffu,0u};
MulAddDst<float,float>(factors_,vr_[128u],coefficients[8u],columns,Zq4(n),Zq0{1,1,0,4,0,1});
Zq1<PIPE_V>();
}


auto pivotOffsets=tinyNegative_[64u].template Zq6<int32_t>();


Gather(pivotDiagonals,panel_,pivotOffsets.template Zq6<Zq2>(),0u,Zq3(15),Zq4(1),uint16_t(8));Zq1<PIPE_V>();


auto pivotAbs=panel_[13u*256u];
Abs(pivotAbs,pivotDiagonals,Zq2(16));Zq1<PIPE_V>();
auto pivGE=residualTile_.template Zq6<Zq4>();
auto pivLE=residualTile_[8u].template Zq6<Zq4>();
Compares<float,Zq4>(pivGE,pivotAbs,pivotLower.f,CMPMODE::GE,Zq2(64));
Compares<float,Zq4>(pivLE,pivotAbs,1.0e30f,CMPMODE::LE,Zq2(64));
ZqB<Zq5::V_S>();
const uint16_t validPivots=pivGE.template Zq6<uint16_t>().GetValue(0u)&pivLE.template Zq6<uint16_t>().GetValue(0u);
const bool rankSafe=(validPivots&uint16_t(0x7fffu))==uint16_t(0x7fffu);
if(!rankSafe)return false;
ResetMask();SetMaskNorm();
auto offsets=panel_.template Zq6<int32_t>();
auto rowBounds=panel_[12u*256u];
auto rawAbs=panel_[13u*256u];


Muls<float>(vr_[64u],pivotAbs,normElementBound*0.5f,Zq3(16),Zq4(1),UnaryRepeatParams{1,1,8,8});
Duplicate<float>(block_,1.0f,Zq3(16),Zq4(1),1,8);Zq1<PIPE_V>();
Div<float>(vr_[128u],block_,pivotDiagonals,Zq3(16),Zq4(1),Zq0{1,1,1,8,8,8});
Mins<float>(vr_[64u],vr_[64u],0x1p56f,Zq2(16));Zq1<PIPE_V>();
Brcb(rowBounds,vr_[64u],Zq4(2),Zq9{2,16});
Brcb(rowBounds[8u],vr_[64u],Zq4(2),Zq9{2,16});

Adds<int32_t>(offsets,pivotOffsets,int32_t(124),Zq3(16),Zq4(1),UnaryRepeatParams{1,1,8,8});
Abs<float>(rawAbs,factors_[16u],Zq3(16),Zq4(16),UnaryRepeatParams{1,1,2,4});Zq1<PIPE_V>();
auto packed=panel_[11u*256u].template Zq6<int32_t>();
Compare<float,Zq4>(packed.template Zq6<Zq4>(),rawAbs,rowBounds,CMPMODE::LE,Zq2(256));Zq1<PIPE_V>();
auto bounded=residualTile_.template Zq6<Zq4>();
Compares<int32_t,Zq4>(bounded,packed,int32_t(-1),CMPMODE::EQ,Zq2(64));
ZqB<Zq5::V_S>();
return(bounded.template Zq6<uint16_t>().GetValue(0u)&uint16_t(0xffu))==uint16_t(0xffu);
}

template<Zq2 TinyN>
__aicore__ __attribute__((noinline))bool KernelTinyFastContext15<TinyN>::BuildTinyInverseLegacy15()
{
constexpr Zq2 n=TinyN;

auto inverseDiagonal=tinyNegative_[32u];


inverseDiagonal.SetValue(15u,0.0f);
#pragma unroll 1
for(Zq2 attempt=0u;attempt<2u;++attempt){


SetMaskNorm();ResetMask();
Duplicate<float>(factors_[480u],0.0f,Zq2(16));
Duplicate<float>(factors_[16u],0.0f,Zq3(16),Zq4(16),1,4);Zq1<PIPE_V>();
Adds<int32_t>(factors_.template Zq6<int32_t>(),panel_[14u*256u].template Zq6<int32_t>(),int32_t(0),Zq3(16),Zq4(15),UnaryRepeatParams{1,1,4,2});
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<n;++i)factors_.SetValue(i*32u+16u+i,1.0f);
ZqB<Zq5::S_V>();


Abs(panel_[13u*256u],panel_[14u*256u],256u);Zq1<PIPE_V>();
if(attempt==0u)ReduceMax(vr_[16u],panel_[13u*256u],panel_[12u*256u],int32_t(256),true);
else ReduceMax(vr_[16u],panel_[13u*256u],panel_[12u*256u],int32_t(240),true);
ZqB<Zq5::V_S>();
float inputScale=vr_[16u].GetValue(0u);
tinyFusedResidual_=attempt==0u&&inputScale>0.0f&&inputScale<=0x1p36f;
if(attempt==0u&&!tinyFusedResidual_){
ReduceMax(vr_[16u],panel_[13u*256u],panel_[12u*256u],int32_t(240),true);ZqB<Zq5::V_S>();
inputScale=vr_[16u].GetValue(0u);
}
const float normElementBound=TinyNormElementBound(inputScale,tinyFusedResidual_);

if(!(inputScale>=0.0f&&inputScale<=1.0e30f*8192.0f))return false;
union{float f;Zq2 u;}pivotLower;
pivotLower.f=inputScale*(1.0f/8192.0f);
if(pivotLower.f<1.0e-30f)pivotLower.f=1.0e-30f;
constexpr Zq2 pivotUpperBits=0x7149f2cau;
if(pivotLower.u>pivotUpperBits)return false;
const Zq2 pivotSpan=pivotUpperBits-pivotLower.u;

auto coefficientMap=panel_[11u*256u].template Zq6<Zq2>();
Add<int32_t>(coefficientMap.template Zq6<int32_t>(),panel_[16u*256u].template Zq6<int32_t>(),tinyNegative_[288u].template Zq6<int32_t>(),Zq3(16),Zq4(16),Zq0{1,1,1,2,2,0});Zq1<PIPE_V>();
Zq3 diagonal0[2]={0x0008000400020001ull,0u};
Zq3 diagonal1[2]={0x0080004000200010ull,0u};
Zq3 diagonal2[2]={0x0800040002000100ull,0u};
Zq3 diagonal3[2]={0x8000400020001000ull,0u};
Duplicate<Zq2>(coefficientMap,Zq2(480u*4u),diagonal0,Zq4(1),1u,8u);
Duplicate<Zq2>(coefficientMap[64u],Zq2(480u*4u),diagonal1,Zq4(1),1u,8u);
Duplicate<Zq2>(coefficientMap[128u],Zq2(480u*4u),diagonal2,Zq4(1),1u,8u);
Duplicate<Zq2>(coefficientMap[192u],Zq2(480u*4u),diagonal3,Zq4(1),1u,8u);Zq1<PIPE_V>();
bool rankSafe=true;
for(Zq2 j=0;j<n;++j){
if(j==0u){

Brcb(block_.template Zq6<int32_t>(),panel_[14u*256u+240u].template Zq6<int32_t>(),Zq4(2),Zq9{1,8});Zq1<PIPE_V>();
Zq3 bColumn[2]={Zq3(1u)<<31u,0u};
Adds<int32_t>(factors_.template Zq6<int32_t>(),block_.template Zq6<int32_t>(),int32_t(0),bColumn,Zq4(n),UnaryRepeatParams{1,0,4,1});
}


SetFlag<Zq5::V_S>(int32_t(0));
Gather(vr_[64u],factors_,coefficientMap[j*16u],0u,16u);Zq1<PIPE_V>();
Brcb(block_,vr_[64u],Zq4(2),Zq9{1,8});
WaitFlag<Zq5::V_S>(int32_t(0));
union{float f;Zq2 u;}pivotWord;
pivotWord.u=factors_.template Zq6<Zq2>().GetValue(j*32u+j);
if((pivotWord.u&0x7fffffffu)-pivotLower.u>pivotSpan){rankSafe=false;break;}
const float diagonal=pivotWord.f;
const float inverse=SeedInverse(diagonal);
inverseDiagonal.SetValue(j,inverse);

Muls(vr_[128u],factors_[j*32u],-inverse,32u);Zq1<PIPE_V>();
Zq3 columns[2]={0xffffffffu,0u};
MulAddDst<float,float>(factors_,vr_[128u],block_,columns,Zq4(n),Zq0{1,1,0,4,0,1});
Zq1<PIPE_V>();
}
if(!rankSafe){
if(tinyFusedResidual_&&attempt==0u){Zq1<PIPE_ALL>();continue;}
return false;
}
ResetMask();SetMaskNorm();
auto offsets=panel_.template Zq6<int32_t>();
ZqB<Zq5::S_V>();


Add<int32_t>(offsets,panel_[16u*256u].template Zq6<int32_t>(),
tinyNegative_[288u].template Zq6<int32_t>(),Zq3(16),Zq4(16),Zq0{1,1,1,2,2,0});
Zq1<PIPE_V>();Adds<int32_t>(offsets,offsets,int32_t(64),Zq2(256));
Zq1<PIPE_V>();
Gather(panel_[12u*256u],factors_,offsets.template Zq6<Zq2>(),0u,256u);Zq1<PIPE_V>();
Mul(factors_,panel_[12u*256u],inverseDiagonal,Zq3(16),Zq4(16),Zq0{1,1,1,2,2,0});


Zq1<PIPE_V>();Abs(panel_[13u*256u],factors_,256u);Zq1<PIPE_V>();


auto bounded=residualTile_.template Zq6<Zq4>();
Compares<float,Zq4>(bounded,panel_[13u*256u],normElementBound,CMPMODE::LE,Zq2(256));
ZqB<Zq5::V_S>();
Zq2 allBounded=0xffffffffu;
for(Zq2 word=0u;word<8u;++word)allBounded&=residualTile_.GetValue(word);
if(allBounded==0xffffffffu)return true;

tinyFusedResidual_=false;
for(Zq2 half=128u;half>=16u;half/=2u){Add(panel_[13u*256u],panel_[13u*256u],panel_[13u*256u+half],half);Zq1<PIPE_V>();}


auto ge=residualTile_.template Zq6<Zq4>();
auto le=residualTile_[8u].template Zq6<Zq4>();
Compares<float,Zq4>(ge,panel_[13u*256u],0.0f,CMPMODE::GE,Zq2(64));
Compares<float,Zq4>(le,panel_[13u*256u],1.0e30f,CMPMODE::LE,Zq2(64));
ZqB<Zq5::V_S>();
const uint16_t accepted=ge.template Zq6<uint16_t>().GetValue(0u)&le.template Zq6<uint16_t>().GetValue(0u);
if((accepted&uint16_t(0x7fffu))!=uint16_t(0x7fffu))return false;
return true;
}
return false;
}
