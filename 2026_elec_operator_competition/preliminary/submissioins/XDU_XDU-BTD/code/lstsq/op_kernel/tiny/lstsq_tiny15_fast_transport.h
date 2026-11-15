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
__aicore__ inline void KernelTinyFastContext15<TinyN>::EmitTinyPrivate(Zq2 batch)
{

auto iterate=tinyNegative_[272u];


GlobalTensor<float>gx;gx.SetGlobalBuffer(x_+Zq3(batch)*15u,15u);
DataCopyPad(gx,iterate,Zq8{1u,60u,0u,0u,0u});
if(r_!=nullptr&&Zq3(batch)<tiling_.residualElements){
GlobalTensor<float>gr;gr.SetGlobalBuffer(r_+batch,1u);
DataCopyPad(gr,factors_[480u],Zq8{1u,4u,0u,0u,0u});
}
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201 && !defined(__ASCENDC_ENABLE_SUPER_KERNEL__))
ZqB<Zq5::MTE3_S>();
#endif
}

template<Zq2 TinyN>
__aicore__ inline bool KernelTinyFastContext15<TinyN>::ProcessFastSingle(Zq2 batch)
{
static_assert(TinyN==15u,"Fast context is15only");

LoadDirect15(batch);
if(!BuildTinyInverseFast(batch)){Zq1<PIPE_ALL>();return false;}


if(!(vectorPivotUsed_?RefineTinyPrivate<true>():RefineLegacy15())){
Zq1<PIPE_ALL>();
if(!vectorPivotUsed_)return false;
vectorPivotUsed_=false;

RestorePristine15(batch);
if(!BuildTinyInverseLegacy15()||!RefineLegacy15()){Zq1<PIPE_ALL>();return false;}
}
EmitTinyPrivate(batch);
return true;
}

template<Zq2 TinyN>
__aicore__ __attribute__((noinline))void KernelTinyFastContext15<TinyN>::RestorePristine15(Zq2 batch)
{
Load(batch);
auto transpose=panel_[16u*256u].template Zq6<Zq2>();
auto features=tinyNegative_[272u].template Zq6<int32_t>();
auto rowBytes=tinyNegative_[288u].template Zq6<int32_t>();


CreateVecIndex<int32_t>(features,int32_t(0),Zq2(16));Zq1<PIPE_V>();
Muls<int32_t>(features,features,int32_t(4),Zq2(16));Zq1<PIPE_V>();
Muls<int32_t>(rowBytes,features,int32_t(16),Zq2(16));
Brcb(transpose.template Zq6<int32_t>(),features,Zq4(2),Zq9{2,16});
Brcb(transpose[8u].template Zq6<int32_t>(),features,Zq4(2),Zq9{2,16});Zq1<PIPE_V>();
Add<int32_t>(transpose.template Zq6<int32_t>(),transpose.template Zq6<int32_t>(),rowBytes,
Zq3(16),Zq4(16),Zq0{1,1,1,2,2,0});Zq1<PIPE_V>();
Zq3 paddingRowMap[2]={0x8000800080008000ull,0u};
Duplicate<Zq2>(transpose,Zq2(15u*4u),paddingRowMap,Zq4(4),1u,8u);

}

template<Zq2 TinyN>
__aicore__ inline void KernelTinyFastContext15<TinyN>::Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t)
{
static_assert(TinyN==15u||TinyN==16u,"Unsupported tiny shape");
t.m=TinyN;t.n=TinyN;t.k=1u;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201 && !(defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1)
pipe_.InitBuffer(arenaBuf_,(4936u+16u*24u+304u+768u)*sizeof(float));
auto arena=Zq7<float>(TPosition::VECCALC,0u,4936u+16u*24u);
tinyNegative_=Zq7<float>(TPosition::VECCALC,(4936u+16u*24u)*sizeof(float),304u);
#else
pipe_.InitBuffer(arenaBuf_,(4936u+16u*24u)*sizeof(float));
auto arena=arenaBuf_.template Get<float>();
pipe_.InitBuffer(tinyNegativeBuf_,304u*sizeof(float));
tinyNegative_=tinyNegativeBuf_.template Get<float>();
#endif


rhsWidth_=t.k;
splitRhs_=t.m==t.n&&((t.n==200u&&t.k==4u)||(t.n==256u&&t.k==3u));
if(splitRhs_)t.k=1u;
tiling_=t;
dmaOutputs_=t.m==t.n&&Zq3(t.n)*t.k<=1024u;
output_=arena[0u];
localFactors_=t.m==t.n&&((t.n>=32u&&t.n<=200u)||((t.n==15u||t.n==16u)&&t.k==1u));
columnLocal_=localFactors_&&(t.n>=128u||((t.n==50u||t.n==64u)&&t.k==2u)||((t.n==15u||t.n==16u)&&t.k==1u));
factorPitch_=(t.n+7u)/8u*8u;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201 && !(defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1)
factors_=Zq7<float>(TPosition::VECCALC,(4936u+16u*24u+304u)*sizeof(float),768u);
#else
pipe_.InitBuffer(tinyAugmentedBuf_,768u*sizeof(float));
factors_=tinyAugmentedBuf_.template Get<float>();
#endif
rows_=arena[24u+16u*16u];
block_=arena[232u+16u*16u];
panel_=arena[360u+16u*16u];
residualTile_=arena[4712u+16u*16u].template Zq6<Zq2>();
vectorResidual_=t.m==t.n;
vectorPitch_=8u;
while(vectorPitch_<t.n)vectorPitch_*=2u;
vr_=arena[4712u+16u*24u];
vectorSolve_=t.m==t.n&&((t.n>=32u&&t.n<=200u)||(t.n==256u&&t.k<=4u)||((t.n==15u||t.n==16u)&&t.k==1u));
solveY_=arena[4920u+16u*24u];

auto transpose=panel_[16u*256u].template Zq6<Zq2>();
auto features=tinyNegative_[272u].template Zq6<int32_t>();
auto rowBytes=tinyNegative_[64u].template Zq6<int32_t>();
auto combinedMap=panel_.template Zq6<Zq2>();
auto rawRows=block_.template Zq6<int32_t>();
SetMaskNorm();ResetMask();
CreateVecIndex<int32_t>(features,int32_t(0),Zq2(16));Zq1<PIPE_V>();
Muls<int32_t>(features,features,int32_t(4),Zq2(16));Zq1<PIPE_V>();

Muls<int32_t>(rowBytes,features,int32_t(32),Zq3(16),Zq4(4),UnaryRepeatParams{1,1,2,0});
Muls<int32_t>(rawRows,features,int32_t(15),Zq2(16));Zq1<PIPE_V>();

Brcb(transpose.template Zq6<int32_t>(),features,Zq4(2),Zq9{2,16});
Brcb(transpose[8u].template Zq6<int32_t>(),features,Zq4(2),Zq9{2,16});

Brcb(combinedMap.template Zq6<int32_t>(),rawRows,Zq4(2),Zq9{4,32});
Brcb(combinedMap[8u].template Zq6<int32_t>(),rawRows,Zq4(2),Zq9{4,32});Zq1<PIPE_V>();
Add<int32_t>(transpose.template Zq6<int32_t>(),transpose.template Zq6<int32_t>(),rowBytes,
Zq3(64),Zq4(4),Zq0{1,1,1,8,8,0});
Zq3 activeLeft[2]={0x7fffu,0u};
Add<int32_t>(combinedMap.template Zq6<int32_t>(),combinedMap.template Zq6<int32_t>(),features,
activeLeft,Zq4(15),Zq0{1,1,1,4,4,0});
Zq3 zeroRight[2]={0xffff8000ffff8000ull,0u};
Duplicate<Zq2>(combinedMap,Zq2(256u*4u),zeroRight,Zq4(8),1u,8u);

Duplicate<Zq2>(combinedMap[480u],Zq2(256u*4u),activeLeft,Zq4(1),1u,8u);Zq1<PIPE_V>();


Muls<int32_t>(tinyNegative_.template Zq6<int32_t>(),features,int32_t(33),Zq2(16));

Zq3 identity0[2]={0x8040201008040201ull,0u};
Zq3 identity1[2]={0x0040201008040201ull,0u};
Duplicate<Zq2>(combinedMap[16u],Zq2(264u*4u),identity0,Zq4(1),4u,8u);
Duplicate<Zq2>(combinedMap[280u],Zq2(264u*4u),identity1,Zq4(1),4u,8u);
Brcb(block_.template Zq6<int32_t>(),features,Zq4(2),Zq9{1,8});Zq1<PIPE_V>();
Zq3 lastColumn[2]={1ull<<31u,0u};
Adds<int32_t>(combinedMap.template Zq6<int32_t>(),block_.template Zq6<int32_t>(),int32_t(240u*4u),
lastColumn,Zq4(15),UnaryRepeatParams{1,0,4,1});

auto factorOnlyMap=combinedMap;factorOnlyMap.SetSize(512u);
Gather<int32_t>(combinedMap[512u].template Zq6<int32_t>(),factorOnlyMap.template Zq6<int32_t>(),transpose,0u,240u);
Adds<int32_t>(combinedMap[752u].template Zq6<int32_t>(),features,int32_t(240u*4u),Zq3(16),Zq4(1),UnaryRepeatParams{1,1,8,8});

Duplicate<float>(panel_[14u*256u+256u],0.0f,Zq3(8),Zq4(1),1u,8u);
Duplicate<float>(panel_[14u*256u+264u],1.0f,Zq3(8),Zq4(1),1u,8u);


a_=reinterpret_cast<__gm__ float*>(a);
b_=reinterpret_cast<__gm__ float*>(b);
x_=reinterpret_cast<__gm__ float*>(x);
r_=reinterpret_cast<__gm__ float*>(res);
(void)workspace;
}

template<Zq2 TinyN>
__aicore__ inline void KernelTinyFastContext15<TinyN>::LoadDirect15(Zq2 batch)
{
auto input=panel_[14u*256u];
GlobalTensor<float>original;original.SetGlobalBuffer(a_+Zq3(batch)*225u,225u);
GlobalTensor<float>originalB;originalB.SetGlobalBuffer(b_+Zq3(batch)*15u,15u);


if(batch>=GetBlockNum())ZqB<Zq5::V_MTE2>();

DataCopyPad(input,original,Zq8{1u,900u,0u,0u,0u},DataCopyPadExtParams<float>{true,0u,7u,0.0f});
DataCopyPad(input[240u],originalB,Zq8{1u,60u,0u,0u,0u},DataCopyPadExtParams<float>{true,0u,1u,0.0f});
ZqB<Zq5::MTE2_V>();
}

template<Zq2 TinyN>
__aicore__ inline void KernelTinyFastContext15<TinyN>::Load(Zq2 batch)
{
static_assert(TinyN==15u,"Padded fifteen input producer");
auto input=panel_[14u*256u];
GlobalTensor<float>original;original.SetGlobalBuffer(a_+Zq3(batch)*225u,225u);


ZqB<Zq5::V_MTE2>();


DataCopyPad(input,original,Zq8{15u,60u,0u,0u,0u},DataCopyPadExtParams<float>{true,0u,1u,0.0f});
GlobalTensor<float>originalB;originalB.SetGlobalBuffer(b_+Zq3(batch)*TinyN,TinyN);

DataCopyPad(input[240u],originalB,Zq8{1u,TinyN*4u,0u,0u,0u},DataCopyPadExtParams<float>{TinyN==15u,0u,Zq4(16u-TinyN),0.0f});
ZqB<Zq5::MTE2_V>();


}
