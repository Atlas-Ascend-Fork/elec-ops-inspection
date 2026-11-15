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
__aicore__ inline void KernelLarge<StaticN, StaticK>::RefineSmallSingle64(Zq3 base)
{
if constexpr(StaticN==50u){WideRefineSmallSingle64(base);return;}
constexpr Zq2 p=512u,pitch=64u;
auto offsets=panel_[16u*p].template Zq6<Zq2>();
for(Zq2 feature=0;feature<pitch;++feature)for(Zq2 lane=0;lane<8u;++lane)
offsets.SetValue(feature*8u+lane,(lane*pitch+feature)*4u);
ZqB<Zq5::S_V>();Duplicate(panel_[14u*p],0.0f,2u*p);Zq1<PIPE_V>();
if(StaticN==50u&&producerPrivate_){Adds(negative64_[4096u],solveY_,0.0f,64u);Zq1<PIPE_V>();}
for(Zq2 step=0;step<2u;++step){
ResidualOperations::PrepareSmallResidualColumns(*this,base,p,pitch,1u);
SmallPrefetch(0u,0u);
for(Zq2 row=0;row<N();row+=8u){
const Zq2 slot=(row/8u)&1u;WaitFlag<Zq5::MTE2_V>(slot);
if(row+8u<N())SmallPrefetch(row+8u,slot^1u);
SmallResidualSingle64(base,row,slot);
}
if constexpr(StaticN==50u)producerCorrection_=producerPrivate_;SolveInPlaceWithFactors(base,w_+OffD(base),1u);
if(StaticN==50u&&producerPrivate_){Add(negative64_[4096u],negative64_[4096u],solveY_,64u);Zq1<PIPE_V>();}
else for(Zq2 row=0;row<N();++row)SetX(base,row,0u,GetX(base,row,0u)+w_[OffD(base)+row]);
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::Refine(Zq3 base)
{

if constexpr((StaticN==50u||StaticN==64u)&&StaticK==2u){RefineSmallSingle64(base);return;}
if((N()==50u||N()==64u)&&K()==2u){RefineSmallRows(base);return;}
if(N()==256u||N()==200u){RefineEightRows(base);return;}
const Zq2 n=N(),k=K();
RefinementOperations::RefineSquareRows(*this, base, n, k);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::ResidualEightRows(Zq3 base,Zq2 row,Zq2 col,
Zq2 groupWidth,Zq2 slot)
{
constexpr Zq2 p=2048u;
const Zq2 active=N()*8u;
auto input=ResidualInput(slot),work=ResidualWork(),extra=ResidualPairExtra();
Zq7<float>a=work,ah=work[2u*p],al=work[4u*p],u=work[9u*p];
Gather(a,input,ResidualOffsets(),0u,active);
Zq1<PIPE_V>();
ResidualOperations::SplitTileInput(a,ah,al,vr_[ResidualMaskPlane()*VectorPitch()].template Zq6<uint16_t>(),active);
ResidualOperations::TileRhsProducts(work,extra,a,ah,al,p,groupWidth);
for(Zq2 rhs=0;rhs<groupWidth;++rhs){
Zq7<float>hi=rhs?work[8u*p]:work[6u*p];
Zq7<float>lo=rhs?extra[3u*p]:work[7u*p];


if(N()==200u){
ResetMask();SetMaskNorm();Adds(ah[576u],hi[576u],0.0f,448u);Zq1<PIPE_V>();
SetMaskCount();
}
MergeCounterResidualHalves<1024u>(ah, hi, lo, al, u, N()==200u?576u:1024u);
MergeCounterResidualHalves<512u>(hi, ah, lo, al, u, 512u);
MergeCounterResidualHalves<256u>(ah, hi, lo, al, u, 256u);
MergeCounterResidualHalves<128u>(hi, ah, lo, al, u, 128u);
MergeCounterResidualHalves<64u>(ah, hi, lo, al, u, 64u);
MergeCounterResidualHalves<32u>(hi, ah, lo, al, u, 32u);
MergeCounterResidualHalves<16u>(ah, hi, lo, al, u, 16u);
MergeCounterResidualHalves<8u>(hi, ah, lo, al, u, 8u);
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
__gm__ float*dx=w_+OffD(base);const Zq2 k=K();
for(Zq2 rhs=0;rhs<groupWidth;++rhs){
Zq7<float>hi=rhs?work[8u*p]:work[6u*p];
Zq7<float>lo=rhs?extra[3u*p]:work[7u*p];
for(Zq2 lane=0;lane<8u;++lane){
Pair diff=square_math::Sum(w_[OffB(base)+Zq3(row+lane)*k+col+rhs],-hi.GetValue(lane));
dx[Zq3(row+lane)*k+col+rhs]=square_math::RoundAdd(diff.hi,
square_math::RoundSub(diff.lo,lo.GetValue(lane)));
}
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::ResidualRowMajorEight(Zq3 base,Zq2 row,Zq2 slot)
{
constexpr Zq2 p=2048u;
const Zq2 q=(N()==256u||N()==200u)&&K()==1u?2080u:p;
auto a=ResidualInput(slot),work=ResidualWork();
Zq7<float>ah=work[2u*q],al=work[4u*q],u=work[9u*q];
Zq7<float>hi=work[6u*q],lo=work[7u*q],t=work[8u*q];
SetMaskCount();ZqA<uint16_t,MaskMode::COUNTER>(0u,2u*p);
And<uint16_t,false>(ah.template Zq6<uint16_t>(),a.template Zq6<uint16_t>(),
vr_[ResidualMaskPlane()*VectorPitch()].template Zq6<uint16_t>(),Zq3(0),Zq4(1),Zq0{1,1,1,8,8,0});
Zq1<PIPE_V>();ZqA<float,MaskMode::COUNTER>(0u,p);
Sub<float,false>(al,a,ah,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
CompensatedTileProduct(hi, lo, t, a, ah, al, work[q], work[3u*q], work[5u*q]);
ResetMask();SetMaskNorm();


Add<float>((ah)[0u],(hi)[0u],(hi[128u])[0u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Add<float>((lo)[0u],(lo)[0u],(lo[128u])[0u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Add<float>((ah)[64u],(hi)[64u],(hi[128u])[64u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Add<float>((lo)[64u],(lo)[64u],(lo[128u])[64u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Zq1<PIPE_V>();
Sub<float>((al)[0u],(ah)[0u],(hi)[0u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Sub<float>((al)[64u],(ah)[64u],(hi)[64u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Zq1<PIPE_V>();
Sub<float>((u)[0u],(ah)[0u],(al)[0u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Sub<float>((u)[64u],(ah)[64u],(al)[64u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Zq1<PIPE_V>();
Sub<float>((u)[0u],(hi)[0u],(u)[0u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Sub<float>((al)[0u],(hi[128u])[0u],(al)[0u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Sub<float>((u)[64u],(hi)[64u],(u)[64u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Sub<float>((al)[64u],(hi[128u])[64u],(al)[64u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Zq1<PIPE_V>();
Add<float>((u)[0u],(u)[0u],(al)[0u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Add<float>((u)[64u],(u)[64u],(al)[64u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Zq1<PIPE_V>();
Add<float>((lo)[0u],(lo)[0u],(u)[0u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Add<float>((lo)[64u],(lo)[64u],(u)[64u],Zq3(64),Zq4(8),Zq0{1,1,1,32,32,32});
Zq1<PIPE_V>();
MergeStridedResidualHalves<64u, 64, 8, 32>(hi, ah, lo, al, u);
MergeStridedResidualHalves<32u, 32, 8, 32>(ah, hi, lo, al, u);
MergeStridedResidualHalves<16u, 16, 8, 32>(hi, ah, lo, al, u);
MergeStridedResidualHalves<8u, 8, 8, 32>(ah, hi, lo, al, u);

Gather(work[6u*q],work[2u*q],ResidualOffsets(),0u,64u);
Gather(work,lo,ResidualOffsets(),0u,64u);Zq1<PIPE_V>();
hi=work[6u*q];lo=work;ah=work[2u*q];al=work[4u*q];
SetMaskCount();
MergeCounterResidualHalves<32u>(ah, hi, lo, al, u, 32u);
MergeCounterResidualHalves<16u>(hi, ah, lo, al, u, 16u);
MergeCounterResidualHalves<8u>(ah, hi, lo, al, u, 8u);
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
for(Zq2 lane=0;lane<8u;++lane){
Pair diff=square_math::Sum(w_[OffB(base)+row+lane],-work[2u*q].GetValue(lane));
w_[OffD(base)+row+lane]=square_math::RoundAdd(diff.hi,
square_math::RoundSub(diff.lo,lo.GetValue(lane)));
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::RefineEightRows(Zq3 base)
{
constexpr Zq2 pitch=256u,p=2048u;
const Zq2 q=(N()==256u||N()==200u)&&K()==1u?2080u:p;
const Zq2 n=N(),k=K();
GlobalTensor<float>archived;
if(n==200u){


archived.SetGlobalBuffer(w_+OffA(base),n*FactorPitch());
ZqB<Zq5::V_MTE3>();
DataCopy(archived,factors_,n*FactorPitch());
ZqB<Zq5::MTE3_S>();
}
__gm__ float*dx=w_+OffD(base);

if(n==256u&&k==1u){
auto offsets=ResidualOffsets();
for(Zq2 feature=0;feature<8u;++feature)for(Zq2 row=0;row<8u;++row)
offsets.SetValue(feature*8u+row,(row*pitch+feature)*4u);
ZqB<Zq5::S_V>();
}
for(Zq2 step=0;step<2u;++step){
if(!(n==256u&&k==1u)){
auto offsets=ResidualOffsets();
if(k==1u){
for(Zq2 feature=0;feature<8u;++feature)for(Zq2 row=0;row<8u;++row)
offsets.SetValue(feature*8u+row,(row*pitch+feature)*4u);
}else for(Zq2 feature=0;feature<pitch;++feature)for(Zq2 row=0;row<8u;++row)
offsets.SetValue(feature*8u+row,(row*pitch+feature)*4u);
ZqB<Zq5::S_V>();
}
Duplicate(ResidualInput(0u),0.0f,p);
Duplicate(ResidualInput(1u),0.0f,p);
auto work=ResidualWork();
auto extra=ResidualPairExtra();
for(Zq2 col=0;col<k;col+=2u){
const Zq2 width=k-col<2u?k-col:2u;
for(Zq2 rhs=0;rhs<width;++rhs){
PrepareResidualX(base,col+rhs);
Zq7<float>x=rhs?extra:work[q];
Zq7<float>xh=rhs?extra[p]:work[3u*q];
Zq7<float>xl=rhs?extra[2u*p]:work[5u*q];
if(k==1u){
ResetMask();SetMaskNorm();
for(Zq2 chunk=0;chunk<pitch;chunk+=64u){
Adds(x[chunk],vr_[pitch+chunk],0.0f,Zq3(64),Zq4(8),UnaryRepeatParams{1,1,32,0});
Adds(xh[chunk],vr_[3u*pitch+chunk],0.0f,Zq3(64),Zq4(8),UnaryRepeatParams{1,1,32,0});
Adds(xl[chunk],vr_[5u*pitch+chunk],0.0f,Zq3(64),Zq4(8),UnaryRepeatParams{1,1,32,0});
}
}else{
Brcb(x,vr_[pitch],Zq4(pitch/8u),Zq9{1,8});
Brcb(xh,vr_[3u*pitch],Zq4(pitch/8u),Zq9{1,8});
Brcb(xl,vr_[5u*pitch],Zq4(pitch/8u),Zq9{1,8});
}
Zq1<PIPE_V>();
}
PrefetchEightRows(0u,0u);
for(Zq2 row=0;row<n;row+=8u){
const Zq2 slot=(row/8u)&1u;
WaitFlag<Zq5::MTE2_V>(slot);
if(row+8u<n)PrefetchEightRows(row+8u,slot^1u);
if(k==1u)ResidualRowMajorEight(base,row,slot);
else ResidualEightRows(base,row,col,width,slot);
}
}
if(n==200u){
ZqB<Zq5::V_MTE2>();
DataCopy(factors_,archived,n*FactorPitch());
ZqB<Zq5::MTE2_S>();
}
SolveInPlaceWithFactors(base,dx,k);
for(Zq2 col=0;col<k;++col)for(Zq2 row=0;row<n;++row)
SetX(base,row,col,GetX(base,row,col)+dx[Zq3(row)*k+col]);
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::PrefetchResidualRow(Zq2 row,Zq2 slot)
{
const Zq2 n=N(),p=VectorPitch();
Zq7<float>a=slot?vr_[ResidualZeroPlane()*p]:vr_;

ZqB<Zq5::V_MTE2>(slot);
GlobalTensor<float>input;
input.SetGlobalBuffer(a_+(Zq3(currentBatch_)*n+row)*n,n);
DataCopyPad(a,input,Zq8{1u,n*4u,0u,0u,0u},
DataCopyPadExtParams<float>{true,0u,Zq4((8u-n%8u)%8u),0.0f});
SetFlag<Zq5::MTE2_V>(slot);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::PrepareResidualX(Zq3 base,Zq2 col)
{
const Zq2 p=VectorPitch();
Zq7<float>x=vr_[p],xh=vr_[3u*p],xl=vr_[5u*p];
ZqB<Zq5::V_S>();
{

if constexpr(StaticN==50u&&StaticK==2u){if(producerPrivate_){Adds(x,negative64_[4096u],0.0f,64u);Zq1<PIPE_V>();}
else for(Zq2 i=0;i<N();++i)x.SetValue(i,GetX(base,i,col));}
else for(Zq2 i=0;i<N();++i)x.SetValue(i,GetX(base,i,col));
}
ZqB<Zq5::S_V>();
And(xh.template Zq6<uint16_t>(),x.template Zq6<uint16_t>(),
vr_[ResidualMaskPlane()*p].template Zq6<uint16_t>(),2u*p);
Zq1<PIPE_V>();
Sub(xl,x,xh,p);
Zq1<PIPE_V>();
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline float KernelLarge<StaticN, StaticK>::VectorResidual(Zq3 base,Zq2 row,Zq2 col,Zq2 slot)
{
    return ResidualOperations::ComputeScalarResidual(vr_, VectorPitch(), slot, ResidualMaskPlane(), ResidualZeroPlane(),
        w_ + OffB(base) + Zq3(row)*K() + col);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::RefineRectangular(Zq3 base,Zq2 batch)
{
    RectangularSolveOperations::Refine(*this, base, batch, M(), N(), K());
}
