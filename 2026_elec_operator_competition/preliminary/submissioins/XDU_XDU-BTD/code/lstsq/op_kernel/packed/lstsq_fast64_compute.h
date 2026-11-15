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
__aicore__ inline void KernelFastContext64<StaticN, StaticK>::ConsumeAugmented64(const __gm__ float*in,Zq2 stride)
{
if(compactPpt64Ready_){


if(!producerCorrection_){Adds(solveY_,negative64_[4224u],0.0f,64u);Zq1<PIPE_V>();}
SetMaskNorm();ResetMask();
Muls(vr_[128u],solveY_,compactNormalization64_,64u);Zq1<PIPE_V>();
Brcb(block_,vr_[128u],Zq4(8),Zq9{1,8});Zq1<PIPE_V>();
Mul(negative64_,panel_[24576u],block_,Zq3(64),Zq4(64),Zq0{1,1,0,8,9,1});Zq1<PIPE_V>();
for(Zq2 h=2048u;h>=64u;h/=2u){Add(negative64_,negative64_,negative64_[h],h);Zq1<PIPE_V>();}
auto result=negative64_[producerCorrection_?4160u:4096u];
Adds(result,negative64_,0.0f,64u);Zq1<PIPE_V>();return;
}

WorkspaceOperations::ConsumeRowInverse64(*this);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline bool KernelFastContext64<StaticN, StaticK>::RefineAccepted64(Zq3 base)
{
bool converged=false;


for(Zq2 step=0;step<3u;++step){


PreparePrivateResidualX64();
RowMajorResidual64All64(base,step>0u&&augmented64Ready_);
producerCorrection_=true;ConsumeAugmented64(w_+OffD(base),1u);
Add(negative64_[4096u],negative64_[4096u],negative64_[4160u],64u);Zq1<PIPE_V>();
auto gateMask=panel_[9088u].template Zq6<Zq4>();auto gateWork=panel_[9200u];

Abs(gateWork,negative64_[4096u],Zq3(128));Zq1<PIPE_V>();
Adds(gateWork,gateWork,1.0f,Zq3(64));Zq1<PIPE_V>();
Div(gateWork[64u],gateWork[64u],gateWork,Zq3(64));Zq1<PIPE_V>();


Compares(gateMask,gateWork[64u],3.0e-5f,CMPMODE::LE,Zq2(64));ZqB<Zq5::V_S>();
converged=gateMask.template Zq6<Zq3>().GetValue(0u)==~Zq3(0u);
if(step>=2u||converged)break;
}
return!compactPpt64Ready_||converged;
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelFastContext64<StaticN, StaticK>::FeatureResidual64(bool cachedHigh)
{
// All layouts are[feature64][row64]. Retain the five split products and
// the exact eight original TwoSum nodes at every64->1 reduction level.
auto a=panel_[16384u],ah=panel_[20480u],al=factors_;
auto hi=panel_,lo=panel_[4096u],t=panel_[8192u];
SetMaskCount();
if(!cachedHigh){
ZqA<uint16_t,MaskMode::COUNTER>(0u,8192u);
And<uint16_t,false>(ah.template Zq6<uint16_t>(),a.template Zq6<uint16_t>(),
vr_[704u].template Zq6<uint16_t>(),Zq3(0),Zq4(1),Zq0{1,1,1,8,8,0});Zq1<PIPE_V>();
ZqA<float,MaskMode::COUNTER>(0u,4096u);
Sub<float,false>(al,a,ah,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
}
ResetMask();SetMaskNorm();
// Consumer products in negative[0,4096) are dead; private X/delta/B start4096.
Brcb(negative64_,vr_[64u],Zq4(8),Zq9{1,8});
Brcb(negative64_[512u],vr_[192u],Zq4(8),Zq9{1,8});
Brcb(negative64_[1024u],vr_[320u],Zq4(8),Zq9{1,8});Zq1<PIPE_V>();
Mul(hi,a,negative64_,Zq3(64),Zq4(64),Zq0{1,1,0,8,8,1});
Mul(lo,ah,negative64_[512u],Zq3(64),Zq4(64),Zq0{1,1,0,8,8,1});Zq1<PIPE_V>();
Sub(lo,lo,hi,4096u);
Mul(t,ah,negative64_[1024u],Zq3(64),Zq4(64),Zq0{1,1,0,8,8,1});Zq1<PIPE_V>();
Add(lo,lo,t,4096u);Zq1<PIPE_V>();
Mul(t,al,negative64_[512u],Zq3(64),Zq4(64),Zq0{1,1,0,8,8,1});Zq1<PIPE_V>();
Add(lo,lo,t,4096u);Zq1<PIPE_V>();
Mul(t,al,negative64_[1024u],Zq3(64),Zq4(64),Zq0{1,1,0,8,8,1});Zq1<PIPE_V>();
Add(lo,lo,t,4096u);Zq1<PIPE_V>();
Zq2 hs=0u,ls=4096u;
for(Zq2 count=2048u;count>=64u;count/=2u){
const Zq2 ho=hs==0u?8192u:0u,ll=hs==0u?10240u:4096u;
auto h0=panel_[hs],h1=panel_[hs+count],l0=panel_[ls],l1=panel_[ls+count];
auto hout=panel_[ho],lout=panel_[ll],z=panel_[12288u],u=negative64_[1536u];
Add(hout,h0,h1,count);Add(lout,l0,l1,count);Zq1<PIPE_V>();
Sub(z,hout,h0,count);Zq1<PIPE_V>();
Sub(u,hout,z,count);Zq1<PIPE_V>();
Sub(u,h0,u,count);Sub(z,h1,z,count);Zq1<PIPE_V>();
Add(u,u,z,count);Zq1<PIPE_V>();
Add(lout,lout,u,count);Zq1<PIPE_V>();
hs=ho;ls=ll;
}
// After exactly6levels high=panel0,low=panel4096. Keep low live until subtraction.
auto bb=negative64_[4224u],neg=panel_[12288u],ss=panel_[8192u],vv=panel_,tt=panel_[10240u];
Muls(neg,panel_,-1.0f,64u);Zq1<PIPE_V>();
Add(ss,bb,neg,64u);Zq1<PIPE_V>();
Sub(vv,ss,bb,64u);Zq1<PIPE_V>();
Sub(tt,ss,vv,64u);Zq1<PIPE_V>();
Sub(tt,bb,tt,64u);Sub(neg,neg,vv,64u);Zq1<PIPE_V>();
Add(tt,tt,neg,64u);Zq1<PIPE_V>();
Sub(tt,tt,panel_[4096u],64u);Zq1<PIPE_V>();
Add(solveY_,ss,tt,64u);Zq1<PIPE_V>();
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelFastContext64<StaticN, StaticK>::RowMajorResidual64All64(Zq3 base,bool cachedHigh)
{
{
if(producerPrivate_&&compactPpt64Ready_){FeatureResidual64(cachedHigh);return;}
}

constexpr Zq2 p=4096u;
auto a=producerA_,ah=panel_,al=panel_[p],hi=panel_[2u*p],lo=panel_[3u*p],t=panel_[4u*p],u=t;


auto splitHigh=augmented64Ready_?panel_[20480u]:ah;


auto splitLow=rowInverse64Ready_?factors_:al;
ResidualOperations::RowMajorResidual64Products(*this, cachedHigh, a, splitHigh, splitLow, hi, lo, t);
ResidualOperations::RowMajorProducts<64u>(a,splitHigh,splitLow,hi,lo,t,vr_);

Add<float>(panel_,hi,hi[32u],Zq3(32),Zq4(64),Zq0{64,1,1,1,8,8});
Zq1<PIPE_V>();
Sub<float>(u[32u],panel_,hi,Zq3(32),Zq4(64),Zq0{1,64,1,8,1,8});Zq1<PIPE_V>();
Sub<float>(u,panel_,u[32u],Zq3(32),Zq4(64),Zq0{1,64,1,8,1,8});Zq1<PIPE_V>();
Sub<float>(u,hi,u,Zq3(64),Zq4(64),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();


Add<float>(panel_[4096u],lo,lo[32u],Zq3(32),Zq4(128),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float>(panel_[2048u],panel_[4096u],panel_[8192u],Zq3(32),Zq4(64),Zq0{64,1,1,1,8,8});Zq1<PIPE_V>();

Add<float>(panel_[8192u],panel_,panel_[1024u],Zq3(64),Zq4(16),Zq0{1,1,1,8,8,8});
Add<float>(panel_[9216u],panel_[2048u],panel_[3072u],Zq3(64),Zq4(16),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(u[1024u],panel_[8192u],panel_,Zq3(64),Zq4(16),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(u,panel_[8192u],u[1024u],Zq3(64),Zq4(16),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(u,panel_,u,Zq3(64),Zq4(32),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float>(u,u,u[1024u],Zq3(64),Zq4(16),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float>(panel_[9216u],panel_[9216u],u,Zq3(64),Zq4(16),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();

Add<float>(panel_,panel_[8192u],panel_[8704u],Zq3(64),Zq4(8),Zq0{1,1,1,8,8,8});
Add<float>(panel_[512u],panel_[9216u],panel_[9728u],Zq3(64),Zq4(8),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(u[512u],panel_,panel_[8192u],Zq3(64),Zq4(8),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(u,panel_,u[512u],Zq3(64),Zq4(8),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(u,panel_[8192u],u,Zq3(64),Zq4(16),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float>(u,u,u[512u],Zq3(64),Zq4(8),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float>(panel_[512u],panel_[512u],u,Zq3(64),Zq4(8),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();


Gather(hi,panel_,panel_[30000u].template Zq6<Zq2>(),0u,1024u);Zq1<PIPE_V>();
SetMaskNorm();ResetMask();

Add<float>(panel_,panel_[8192u],panel_[8704u],Zq3(64),Zq4(8),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(u[512u],panel_,panel_[8192u],Zq3(64),Zq4(4),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float>(u,panel_,u[512u],Zq3(64),Zq4(4),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float>(u,panel_[8192u],u,Zq3(64),Zq4(8),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(u,u,u[512u],Zq3(64),Zq4(4),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(panel_[64u],panel_[64u],u,Zq3(64),Zq4(4),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();

Add<float>(panel_[8192u],panel_,panel_[256u],Zq3(64),Zq4(4),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(u[256u],panel_[8192u],panel_,Zq3(64),Zq4(2),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float>(u,panel_[8192u],u[256u],Zq3(64),Zq4(2),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float>(u,panel_,u,Zq3(64),Zq4(4),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(u,u,u[256u],Zq3(64),Zq4(2),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(panel_[8256u],panel_[8256u],u,Zq3(64),Zq4(2),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();

Add<float>(panel_,panel_[8192u],panel_[8320u],Zq3(64),Zq4(2),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(u[128u],panel_,panel_[8192u],Zq3(64),Zq4(1),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float>(u,panel_,u[128u],Zq3(64),Zq4(1),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float>(u,panel_[8192u],u,Zq3(64),Zq4(2),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(u,u,u[128u],Zq3(64),Zq4(1),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(panel_[64u],panel_[64u],u,Zq3(64),Zq4(1),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
lo=panel_[64u];

ResetMask();SetMaskNorm();


if(!producerPrivate_){
ResidualOperations::WriteScalarResidual64(w_+OffB(base),w_+OffD(base),ah,lo);
return;
}
ResetMask();SetMaskNorm();
auto bb=negative64_[4224u],neg=panel_[2u*p],ss=panel_[p],vv=panel_,tt=panel_[4u*p];
WriteCompensatedResidual(solveY_, bb, ah, lo, neg, ss, vv, tt, 64u);
}
