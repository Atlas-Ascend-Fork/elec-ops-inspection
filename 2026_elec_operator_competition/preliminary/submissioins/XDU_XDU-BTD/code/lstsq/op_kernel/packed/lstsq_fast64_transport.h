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
__aicore__ inline void KernelFastContext64<StaticN, StaticK>::Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t)
{
static_assert(StaticN==0u||(StaticN==200u&&StaticK==4u)||
(StaticN==256u&&StaticK==3u)||((StaticN==50u||StaticN==64u)&&StaticK==2u),
"Unsupported fixed shape");
{t.m=StaticN;t.n=StaticN;t.k=StaticK;}
{
rhsWidth_=t.k;splitRhs_=true;t.k=1u;tiling_=t;
dmaOutputs_=true;localFactors_=true;columnLocal_=true;factorPitch_=64u;
vectorResidual_=true;vectorPitch_=64u;vectorSolve_=true;
// One allocation; preserve every original32B-aligned range and logical tensor size.
pipe_.InitBuffer(packed64Buf_,195680u);
auto storage=packed64Buf_.template Get<float>();
producerA_=storage;producerA_.SetSize(4096u);
output_=storage[4096u];output_.SetSize(520u);
factors_=storage[4616u];factors_.SetSize(4608u);
rows_=storage[9224u];rows_.SetSize(208u);
block_=storage[9432u];block_.SetSize(512u);
panel_=storage[9944u];panel_.SetSize(33280u);
residualTile_=storage[43224u].template Zq6<Zq2>();residualTile_.SetSize(512u);
vr_=storage[43736u];vr_.SetSize(832u);
solveY_=storage[44568u];solveY_.SetSize(64u);
negative64_=storage[44632u];negative64_.SetSize(4288u);
}
if(vectorResidual_)WorkspaceOperations::InitializeVectorResidual(*this, vectorPitch_);
WorkspaceOperations::InitializeGlobalBuffers(*this,a,b,x,res,workspace);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelFastContext64<StaticN, StaticK>::EmitSplitRhsAtomic64(Zq3 base,Zq2 batch)
{
if(!producerPrivate_){


ZqB<Zq5::V_S>();
for(Zq2 row=0u;row<64u;++row)negative64_.SetValue(4096u+row,GetX(base,row,0u));
ZqB<Zq5::S_V>();
}
SetMaskNorm();ResetMask();
auto indices=producerPrivate_?panel_[31024u].template Zq6<int32_t>():panel_.template Zq6<int32_t>();
auto minimum=panel_[128u].template Zq6<int32_t>();
auto maximum=panel_[256u].template Zq6<int32_t>();

if(!producerPrivate_){

CreateVecIndex<int32_t>(indices,int32_t(0),Zq2(8));
Adds<int32_t>(indices[8u],indices,int32_t(8),Zq2(8));Zq1<PIPE_V>();
Adds<int32_t>(indices[16u],indices,int32_t(16),Zq2(16));Zq1<PIPE_V>();
Adds<int32_t>(indices[32u],indices,int32_t(32),Zq2(32));Zq1<PIPE_V>();
Adds<int32_t>(indices[64u],indices,int32_t(64),Zq2(64));Zq1<PIPE_V>();

ShiftRight<int32_t>(indices,indices,int32_t(1),int32_t(128));Zq1<PIPE_V>();
Muls<int32_t>(indices,indices,int32_t(4),Zq2(128));Zq1<PIPE_V>();
}
WorkspaceOperations::CopyAtomicRhs64(*this, batch, indices, minimum, maximum);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelFastContext64<StaticN, StaticK>::EmitSplitRhsDma(Zq3 base,Zq2 batch)
{
    WorkspaceOperations::EmitSplitRhsDma64(*this, base, batch);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline bool KernelFastContext64<StaticN, StaticK>::TryTask64(Zq2 task)
{
static_assert(StaticN==64u&&StaticK==2u,"Fast64only");
currentRhs_=task%2u;currentBatch_=task/2u;
const Zq3 base=Zq3(task)*PerBatch();
augmented64Ready_=false;rowInverse64Ready_=false;compactPpt64Ready_=false;
producerPrivate_=false;producerCorrection_=false;replicaOffsetsReady_=false;
diagonalKnown_=false;diagonalSafe_=false;
PrefetchInputA64(currentBatch_);
GlobalTensor<float>inputB;inputB.SetGlobalBuffer(b_+Zq3(currentBatch_)*128u,128u);
// Future X/delta slots are dead until the original RHS is selected.
// Initial V_MTE2 in PrefetchInputA64 already retires old Vector readers.
DataCopy(negative64_[4096u],inputB,128u);SetFlag<Zq5::MTE2_V>(1);
auto bo=vr_[640u].template Zq6<int32_t>();
auto originalIndex=rows_.template Zq6<int32_t>();ZqB<Zq5::V_S>();


CreateVecIndex<int32_t>(originalIndex,int32_t(0),Zq2(8));
Adds<int32_t>(originalIndex[8u],originalIndex,int32_t(8),Zq2(8));Zq1<PIPE_V>();
Adds<int32_t>(originalIndex[16u],originalIndex,int32_t(16),Zq2(16));Zq1<PIPE_V>();
Adds<int32_t>(originalIndex[32u],originalIndex,int32_t(32),Zq2(32));Zq1<PIPE_V>();

Adds<int32_t>(originalIndex[64u],originalIndex,int32_t(64),Zq2(64));Zq1<PIPE_V>();
auto emitMap=panel_[31024u].template Zq6<int32_t>();
ShiftRight<int32_t>(emitMap,originalIndex,int32_t(1),int32_t(128));Zq1<PIPE_V>();
Muls<int32_t>(emitMap,emitMap,int32_t(4),Zq2(128));Zq1<PIPE_V>();


Muls<int32_t>(bo,originalIndex,int32_t(8),Zq2(64));Zq1<PIPE_V>();
Adds<int32_t>(bo,bo,int32_t(currentRhs_*4u),Zq2(64));Zq1<PIPE_V>();
const bool factorReady=BuildCompactPpt64();
// Drain B ready1 on acceptance AND every original producer rejection.
WaitFlag<Zq5::MTE2_V>(1);
if(!factorReady){Zq1<PIPE_ALL>();return false;}
SetMaskNorm();ResetMask();
Gather(negative64_[4224u],negative64_[4096u],bo.template Zq6<Zq2>(),0u,64u);ZqB<Zq5::V_S>();
producerPrivate_=true;producerCorrection_=false;
ConsumeAugmented64(w_+OffB(base),1u);
if(!RefineAccepted64(base)){Zq1<PIPE_ALL>();return false;}
EmitSplitRhsDma(base,currentBatch_);
return true;
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelFastContext64<StaticN, StaticK>::TransposeOriginalA64()
{
// b32: source16x8 becomes8x16. Pair destination blocks per output row.
// Four repeats advance16source rows and16destination columns each time.
const uint64_t srcBase=(uint64_t)producerA_.GetPhyAddr();
const uint64_t dstBase=(uint64_t)panel_[16384u].GetPhyAddr();
uint64_t srcList[16],dstList[16];
for(Zq2 col=0u;col<8u;++col){
#pragma unroll 16
for(Zq2 i=0u;i<16u;++i){
srcList[i]=srcBase+Zq3(i*64u+col*8u)*4u;
dstList[i]=dstBase+Zq3((col*8u+i/2u)*64u+(i%2u)*8u)*4u;
}
// Address construction is independent of the pending originalA DMA.
if(col==0u)WaitFlag<Zq5::MTE2_V>(0);
TransDataTo5HD<float>(dstList,srcList,TransDataTo5HDParams{false,false,4u,2u,128u});
Zq1<PIPE_V>();
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ __attribute__((noinline))bool KernelFastContext64<StaticN, StaticK>::BuildCompactPpt64()
{
auto aug=panel_[24576u];
PrepareCompactPivotMap64();
TransposeOriginalA64();
Abs(panel_,panel_[16384u],4096u);Zq1<PIPE_V>();


WholeReduceMax(panel_[4096u],panel_,int32_t(64),int32_t(64),int32_t(1),int32_t(1),int32_t(8),ReduceOrder::ORDER_VALUE_INDEX);Zq1<PIPE_V>();
Zq3 maxPairs[2]={0x5555555555555555ull,0u};
WholeReduceMax(panel_[4224u],panel_[4096u],maxPairs,int32_t(2),int32_t(1),int32_t(1),int32_t(8),ReduceOrder::ORDER_VALUE_INDEX);Zq1<PIPE_V>();
Zq3 maxFinal[2]={0x5ull,0u};
WholeReduceMax(vr_[64u],panel_[4224u],maxFinal,int32_t(1),int32_t(1),int32_t(1),int32_t(8),ReduceOrder::ORDER_VALUE_INDEX);ZqB<Zq5::V_S>();
const float maximum=vr_[64u].GetValue(0u);
if(!(maximum>=1.0e-30f&&maximum<=1.0e30f))return false;
const Zq2 bits=vr_[64u].template Zq6<Zq2>().GetValue(0u);
const int32_t shift=-80-(int32_t((bits>>23u)&255u)-127);
if(shift<-120||shift>120)return false;
union{Zq2 u;float f;}sc,un;
sc.u=Zq2(127+shift)<<23u;un.u=Zq2(127-shift)<<23u;
compactNormalization64_=sc.f;
ZqB<Zq5::S_V>();
Muls<float>(aug,panel_[16384u],sc.f,Zq3(64),Zq4(64),UnaryRepeatParams{1,1,9,8});


EliminateCompactPivots64(aug);


if (!ValidateCompactInverse64(aug, un.f, sc.u >> 23u)) return false;
compactPpt64Ready_=true;rowInverse64Ready_=true;augmented64Ready_=true;return true;
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelFastContext64<StaticN, StaticK>::PrepareCompactPivotMap64()
{
auto rb=rows_.template Zq6<int32_t>();
auto cb=rows_[64u].template Zq6<int32_t>();
ZqB<Zq5::V_S>();ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();


Muls<int32_t>(cb,rb,int32_t(4),Zq2(64));Zq1<PIPE_V>();
// Only64pivot-history offsets remain live; old512/1024residual maps have no private consumer.
auto actualOffsets=rows_[128u].template Zq6<int32_t>();
Muls<int32_t>(actualOffsets,cb,int32_t(64),Zq2(64));Zq1<PIPE_V>();
auto pivotLanes=rows_[192u].template Zq6<int32_t>();
CreateVecIndex<int32_t>(pivotLanes,int32_t(0),Zq2(8));Zq1<PIPE_V>();
Muls<int32_t>(pivotLanes,pivotLanes,int32_t(32),Zq2(8));Zq1<PIPE_V>();
Add<int32_t>(actualOffsets,actualOffsets,pivotLanes,Zq3(8),Zq4(8),Zq0{1,1,0,1,1,0});Zq1<PIPE_V>();
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelFastContext64<StaticN, StaticK>::EliminateCompactPivots64(Zq7<float> aug)
{
auto columnOffsets=panel_[8192u].template Zq6<int32_t>();
auto cb=rows_[64u].template Zq6<int32_t>();
// One64word column map replaces both4096word per-pivot tables.
Muls<int32_t>(columnOffsets,cb,int32_t(72),Zq2(64));Zq1<PIPE_V>();
for(Zq2 j=0u;j<64u;++j){

// Prefill sentinels beside the actual pivot read, before its existing fence.
Zq3 keep[2]={~(Zq3(1u)<<j),0u};
const Zq2 pivotSlot=j*64u+(j%8u)*8u;
Brcb(factors_[j*64u],aug[j*72u+(j/8u)*8u],Zq4(1),Zq9{1,8});
Duplicate<float>(vr_[128u],-1.0f,Zq2(64));
Duplicate<float>(vr_[256u],1.0f,Zq2(64));Zq1<PIPE_V>();
Muls(factors_[pivotSlot],factors_[pivotSlot],-1.0f,8u);
// Integer Adds0 copies all floatbits, including signedzero; skip only lane j.
Adds<int32_t>(vr_[128u].template Zq6<int32_t>(),aug[j*72u].template Zq6<int32_t>(),
int32_t(0),keep,Zq4(1),UnaryRepeatParams{1,1,8,8});
Gather(vr_[256u],aug,columnOffsets.template Zq6<Zq2>(),j*4u,keep,Zq4(1),Zq4(8));Zq1<PIPE_V>();
Div<float>(vr_[256u],vr_[256u],factors_[pivotSlot],Zq3(64),Zq4(1),Zq0{1,1,0,8,8,0});Zq1<PIPE_V>();
Brcb(block_,vr_[256u],Zq4(8),Zq9{1,8});Zq1<PIPE_V>();


MulAddDst<float,float>(aug,vr_[128u],block_,Zq3(64),Zq4(64),Zq0{1,1,0,9,0,1});Zq1<PIPE_V>();
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline bool KernelFastContext64<StaticN, StaticK>::ValidateCompactInverse64(Zq7<float> aug, float inverseScale, Zq2 scaleExponent)
{
auto actualOffsets=rows_[128u].template Zq6<int32_t>();
auto actual=panel_[4288u],original=panel_[5120u];

Gather(actual,factors_,actualOffsets.template Zq6<Zq2>(),0u,64u);Zq1<PIPE_V>();
Abs(actual,actual,64u);Zq1<PIPE_V>();
Muls(original,actual,inverseScale,64u);Zq1<PIPE_V>();
auto p0=rows_.template Zq6<Zq4>();auto p1=rows_[16u].template Zq6<Zq4>();
auto p2=rows_[32u].template Zq6<Zq4>();auto p3=rows_[48u].template Zq6<Zq4>();
Compares(p0,actual,1.1754943508222875e-38f,CMPMODE::GE,Zq2(64));
Compares(p1,actual,8.881784197001252e-16f,CMPMODE::LE,Zq2(64));
Compares(p2,original,1.0e-30f,CMPMODE::GE,Zq2(64));
Compares(p3,original,1.0e30f,CMPMODE::LE,Zq2(64));Zq1<PIPE_V>();
And(p0.template Zq6<uint16_t>(),p0.template Zq6<uint16_t>(),p1.template Zq6<uint16_t>(),Zq2(4));
And(p2.template Zq6<uint16_t>(),p2.template Zq6<uint16_t>(),p3.template Zq6<uint16_t>(),Zq2(4));Zq1<PIPE_V>();
And(p0.template Zq6<uint16_t>(),p0.template Zq6<uint16_t>(),p2.template Zq6<uint16_t>(),Zq2(4));ZqB<Zq5::V_S>();
if(p0.template Zq6<Zq3>().GetValue(0u)!=~Zq3(0u))return false;
ZqB<Zq5::S_V>();

// Power-of-two scaling is exact at the normal1e12 boundary. Clamp an
// overflowing inverse threshold to maxfinite so NaN/Inf still reject.
union{Zq2 u;float f;}inverseLimit;
const Zq2 inverseLimitExponent=293u-scaleExponent;
inverseLimit.u=inverseLimitExponent>=255u?Zq2(0x7f7fffffu):
((inverseLimitExponent<<23u)|Zq2(0x0068d4a5u));
auto normBits=panel_[12288u].template Zq6<Zq4>();
auto lowerBits=panel_[12416u].template Zq6<Zq4>();
UnaryRepeatParams inverseBoundRepeats;
inverseBoundRepeats.srcRepStride=9u;
Compares<float,Zq4>(normBits,aug,inverseLimit.f,CMPMODE::LE,Zq3(64),Zq4(64),inverseBoundRepeats);
Compares<float,Zq4>(lowerBits,aug,-inverseLimit.f,CMPMODE::GE,Zq3(64),Zq4(64),inverseBoundRepeats);Zq1<PIPE_V>();
auto words=normBits.template Zq6<uint16_t>();
And(words,words,lowerBits.template Zq6<uint16_t>(),Zq2(256));Zq1<PIPE_V>();
for(Zq2 half=128u;half>=16u;half/=2u){And(words,words,words[half],half);Zq1<PIPE_V>();}
ZqB<Zq5::V_S>();
for(Zq2 word=0u;word<4u;++word)if(normBits.template Zq6<Zq3>().GetValue(word)!=~Zq3(0u))return false;
return true;
}
