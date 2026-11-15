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
template<bool Direct>
__aicore__ inline bool KernelTinyFastContext15<TinyN>::RefineTinyPrivate()
{
constexpr Zq2 p=256u;auto iterate=tinyNegative_[272u];
if constexpr(!Direct){
Duplicate(panel_[10u*p].template Zq6<Zq2>(),Zq2(0xfffff000u),p);
Zq1<PIPE_V>();
{constexpr Zq2 active=p;auto input=panel_[14u*p];
auto a=Direct?input:panel_;auto ah=panel_[4u*p],al=panel_[5u*p];
if constexpr(!Direct){
if(Direct||tinyFusedResidual_){
Gather(a,input,panel_[16u*p].template Zq6<Zq2>(),0u,240u);
Adds<int32_t>(a[240u].template Zq6<int32_t>(),input[240u].template Zq6<int32_t>(),int32_t(0),Zq2(16));
}else Gather(a,input,panel_[16u*p].template Zq6<Zq2>(),0u,active);
Zq1<PIPE_V>();
}
SetMaskCount();ZqA<uint16_t,MaskMode::COUNTER>(0u,2u*active);
And<uint16_t,false>(ah.template Zq6<uint16_t>(),a.template Zq6<uint16_t>(),
panel_[10u*p].template Zq6<uint16_t>(),Zq3(0),Zq4(1),Zq0{1,1,1,8,8,0});
Zq1<PIPE_V>();ZqA<float,MaskMode::COUNTER>(0u,active);
Sub<float,false>(al,a,ah,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
}
}
ResetMask();SetMaskNorm();
if constexpr(Direct){

Zq3 nodePaddingMask[2]={0xffffu,0u};
Duplicate<float>(panel_[6u*p+240u],0.0f,nodePaddingMask,Zq4(2),1u,32u);
Gather(block_,factors_,panel_.template Zq6<Zq2>(),0u,16u);Zq1<PIPE_V>();
Mul<float>(iterate,block_,vr_[128u],Zq3(16),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
}else{Adds(iterate,factors_[15u*16u],0.0f,16u);Zq1<PIPE_V>();}
if(Direct||tinyFusedResidual_){Zq3 sentinel[2]={0x8000u,0u};Duplicate<float>(iterate,-1.0f,sentinel,Zq4(1),1,8);Zq1<PIPE_V>();}
for(Zq2 step=0;step<2u;++step){

auto x=panel_[p],xh=panel_[2u*p],xl=panel_[3u*p];
if constexpr(Direct){


auto smallX=vr_,smallH=vr_[16u],smallL=vr_[32u];
Muls<float>(smallX,iterate,-1.0f,Zq3(16),Zq4(1),UnaryRepeatParams{1,1,8,8});Zq1<PIPE_V>();
And<uint16_t>(smallH.template Zq6<uint16_t>(),smallX.template Zq6<uint16_t>(),panel_[10u*p].template Zq6<uint16_t>(),Zq3(32),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(smallL,smallX,smallH,Zq3(16),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();


Brcb(x,smallX,Zq4(6),Zq9{2,16});
Brcb(x[8u],smallX,Zq4(6),Zq9{2,16});Zq1<PIPE_V>();
}else{
Brcb(x,iterate,Zq4(2),Zq9{2,16});
Brcb(x[8u],iterate,Zq4(2),Zq9{2,16});Zq1<PIPE_V>();
if(Direct||tinyFusedResidual_){Muls(x,x,-1.0f,p);Zq1<PIPE_V>();}
And(xh.template Zq6<uint16_t>(),x.template Zq6<uint16_t>(),panel_[10u*p].template Zq6<uint16_t>(),2u*p);Zq1<PIPE_V>();
Sub(xl,x,xh,p);Zq1<PIPE_V>();
}
TinyResidualTile<Direct>();
if constexpr(Direct){ConsumeTinyRawInverse15();}else{ConsumeTinyInverse();}
Add(iterate,iterate,solveY_,16u);Zq1<PIPE_V>();
}


auto deltaAbs=rows_,xAbs=rows_[16u],threshold=rows_[128u];
Abs(deltaAbs,solveY_,16u);Abs(xAbs,iterate,16u);Zq1<PIPE_V>();
Adds(threshold,xAbs,1.0f,16u);Zq1<PIPE_V>();
Muls(threshold,threshold,1.0e-6f,16u);Zq1<PIPE_V>();
Sub(deltaAbs,deltaAbs,threshold,16u);


Adds<float>(xAbs,xAbs,-3.402823466e38f,Zq3(16),Zq4(1),UnaryRepeatParams{1,1,8,8});Zq1<PIPE_V>();
auto packed=residualTile_.template Zq6<Zq4>();
Compares<float,Zq4>(packed,deltaAbs,0.0f,CMPMODE::LE,Zq2(64));ZqB<Zq5::V_S>();
return(packed.template Zq6<Zq2>().GetValue(0u)&0x7fff7fffu)==0x7fff7fffu;
}

template<Zq2 TinyN>
template<bool Direct>
__aicore__ inline void KernelTinyFastContext15<TinyN>::TinyResidualTile()
{
constexpr Zq2 p=256u,active=p,groupWidth=1u;
auto work=panel_,extra=panel_[10u*p];
auto input=Direct?factors_[512u]:panel_[14u*p];
Zq7<float>a=Direct?input:work;
Zq7<float>ah=work[4u*p],al=work[5u*p];
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,active);
if constexpr(Direct){

TinyResidualProducts(work, a, ah, al);
TinyResidualRounding(work);

Add<float,false>(solveY_,work[736u],work[2048u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
return;
}


auto tailB=input[240u],tailNeg=panel_[12u*p],tailSum=panel_[12u*p+16u];
auto tailBB=panel_[12u*p+32u],tailLeft=panel_[12u*p+48u],tailRight=panel_[12u*p+64u];
constexpr Zq4 productStride=Direct?16u:8u;
for(Zq2 rhs=0;rhs<groupWidth;++rhs){
Zq7<float>x=rhs?extra:work[p];
Zq7<float>xh=rhs?extra[p]:work[2u*p];
Zq7<float>xl=rhs?extra[2u*p]:work[3u*p];
Zq7<float>hi=rhs?work[8u*p]:work[6u*p];
Zq7<float>lo=rhs?extra[3u*p]:(Direct?work[6u*p+64u]:work[7u*p]);
Zq7<float>t=rhs?work[9u*p]:work[8u*p];

Mul<float,false>(hi,a,x,Zq3(0),Zq4(1),Zq0{1,1,1,productStride,8,8});
Mul<float,false>(lo,ah,xh,Zq3(0),Zq4(1),Zq0{1,1,1,productStride,8,8});
Zq1<PIPE_V>();
Sub<float,false>(lo,lo,hi,Zq3(0),Zq4(1),Zq0{1,1,1,productStride,productStride,productStride});
if constexpr(Direct){


Zq1<PIPE_V>();
SetMaskNorm();
ZqA<float,MaskMode::NORMAL>(Zq3(0),Zq3(0xffffffffffffffffull));
MulAddDst<float,float,false>(lo,ah,xl,Zq3(0),Zq4(4),Zq0{1,1,1,16,8,8});Zq1<PIPE_V>();
MulAddDst<float,float,false>(lo,al,xh,Zq3(0),Zq4(4),Zq0{1,1,1,16,8,8});Zq1<PIPE_V>();
MulAddDst<float,float,false>(lo,al,xl,Zq3(0),Zq4(4),Zq0{1,1,1,16,8,8});Zq1<PIPE_V>();
}else{


auto cross2=work[9u*p],cross3=work[11u*p];
Mul<float,false>(t,ah,xl,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Mul<float,false>(cross2,al,xh,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Mul<float,false>(cross3,al,xl,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,cross2,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,cross3,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
}
}
auto sumScratch=work[2u*p];
for(Zq2 rhs=0;rhs<groupWidth;++rhs){
Zq7<float>hi=rhs?work[8u*p]:work[6u*p];
Zq7<float>lo=rhs?extra[3u*p]:work[7u*p];


if constexpr(!Direct){
SetMaskNorm();
ZqA<float,MaskMode::NORMAL>(Zq3(0),Zq3(0xffffffffffffffffull));
}
auto pairedTmp=work[8u*p];
if constexpr(Direct){
Add<float,false>(sumScratch,hi,hi[256u],Zq3(0),Zq4(4),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp[128u],sumScratch,hi,Zq3(0),Zq4(2),Zq0{1,1,1,8,16,16});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,sumScratch,pairedTmp[128u],Zq3(0),Zq4(2),Zq0{1,1,1,8,16,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,hi,pairedTmp,Zq3(0),Zq4(4),Zq0{1,1,1,8,16,8});Zq1<PIPE_V>();
Add<float,false>(pairedTmp,pairedTmp,pairedTmp[128u],Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(sumScratch[64u],sumScratch[64u],pairedTmp,Zq3(0),Zq4(2),Zq0{1,1,1,16,16,8});Zq1<PIPE_V>();


Add<float,false>(hi,sumScratch,sumScratch[128u],Zq3(0),Zq4(2),Zq0{1,1,1,32,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp[64u],hi,sumScratch,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,hi,pairedTmp[64u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,sumScratch,pairedTmp,Zq3(0),Zq4(2),Zq0{1,1,1,8,16,8});Zq1<PIPE_V>();
Add<float,false>(pairedTmp,pairedTmp,pairedTmp[64u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,pairedTmp,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
}else{
Add<float,false>(sumScratch,hi,hi[128u],Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});
Add<float,false>(lo,lo,lo[128u],Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp[128u],sumScratch,hi,Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,sumScratch,pairedTmp[128u],Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,hi,pairedTmp,Zq3(0),Zq4(4),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(pairedTmp,pairedTmp,pairedTmp[128u],Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,pairedTmp,Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();


Add<float,false>(hi,sumScratch,sumScratch[64u],Zq3(0),Zq4(2),Zq0{1,1,1,32,160,160});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp[64u],hi,sumScratch,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,hi,pairedTmp[64u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,sumScratch,pairedTmp,Zq3(0),Zq4(2),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(pairedTmp,pairedTmp,pairedTmp[64u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,pairedTmp,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
}

ZqA<float,MaskMode::NORMAL>(Zq3(0),Zq3(0xffffffffull));
Add<float,false>(work[p],hi,hi[32u],Zq3(0),Zq4(2),Zq0{1,1,1,32,32,32});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp[32u],work[p],hi,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,work[p],pairedTmp[32u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,hi,pairedTmp,Zq3(0),Zq4(2),Zq0{1,1,1,4,4,4});Zq1<PIPE_V>();
Add<float,false>(pairedTmp,pairedTmp,pairedTmp[32u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(work[2u*p],work[2u*p],pairedTmp,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();

if(Direct||tinyFusedResidual_){
ZqA<float,MaskMode::NORMAL>(Zq3(0),Zq3(0xffffull));
Add<float,false>(hi,work[p],work[p+16u],Zq3(0),Zq4(2),Zq0{1,1,1,32,32,32});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp[16u],hi,work[p],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,hi,pairedTmp[16u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,work[p],pairedTmp,Zq3(0),Zq4(2),Zq0{1,1,1,2,2,2});Zq1<PIPE_V>();
Add<float,false>(pairedTmp,pairedTmp,pairedTmp[16u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,pairedTmp,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(solveY_,hi,lo,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
return;
}

ZqA<float,MaskMode::NORMAL>(Zq3(0),Zq3(0xffffull));
Add<float,false>(hi,work[p],work[p+16u],Zq3(0),Zq4(2),Zq0{1,1,1,32,32,32});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp[16u],hi,work[p],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Muls<float,false>(tailNeg,hi,-1.0f,Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,hi,pairedTmp[16u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Add<float,false>(tailSum,tailB,tailNeg,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(pairedTmp,work[p],pairedTmp,Zq3(0),Zq4(2),Zq0{1,1,1,2,2,2});
Sub<float,false>(tailBB,tailSum,tailB,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(pairedTmp,pairedTmp,pairedTmp[16u],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Sub<float,false>(tailLeft,tailSum,tailBB,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,pairedTmp,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Sub<float,false>(tailLeft,tailB,tailLeft,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Sub<float,false>(tailRight,tailNeg,tailBB,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
}

auto sum=panel_[12u*p+16u],left=panel_[12u*p+48u],right=panel_[12u*p+64u];
Add<float,false>(left,left,right,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(left,left,work[7u*p],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(solveY_,sum,left,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
}
