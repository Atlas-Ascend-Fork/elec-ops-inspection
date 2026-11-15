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
__aicore__ inline void KernelLarge<StaticN, StaticK>::WideSmallResidualSingle64(Zq3 base,Zq2 row,Zq2 slot)
{
constexpr Zq2 p=2048u,active=p,groupWidth=1u;
auto work=panel_,extra=panel_[10u*p],input=producerA_[row*64u];
Zq7<float>a=work,ah=work[2u*p],al=work[4u*p],u=work[9u*p];
Gather(a,input,panel_[14u*p].template Zq6<Zq2>(),0u,active);Zq1<PIPE_V>();
ResidualOperations::ComputeWideSmallResidual(*this, base, row, work, a, ah, al);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::RowMajorResidual50All64(Zq3 base)
{
constexpr Zq2 p=4096u;
auto a=producerA_,ah=panel_,al=panel_[p],hi=panel_[2u*p],lo=panel_[3u*p],t=panel_[4u*p],u=t;
ResidualOperations::RowMajorResidual50Products(*this, a, ah, al, hi, lo, t);
ResidualOperations::RowMajorProducts<50u>(a,ah,al,hi,lo,t,vr_);
MergeStridedResidualHalves<32u, 32u, 50u, 8u>(ah, hi, lo, al, u);
ResetMask();SetMaskNorm();
MergeStridedResidualHalves<16u, 16u, 50u, 8u>(hi, ah, lo, al, u);
ResetMask();SetMaskNorm();
MergeStridedResidualHalves<8u, 8u, 50u, 8u>(ah, hi, lo, al, u);


Duplicate(ah[3200u],0.0f,Zq2(896));
Duplicate(lo[3200u],0.0f,Zq2(896));Zq1<PIPE_V>();

auto offsets=panel_[14u*2048u].template Zq6<Zq2>();
Gather(hi,ah,offsets,0u,512u);
Gather(panel_[p],lo,offsets,0u,512u);Zq1<PIPE_V>();
lo=panel_[p];al=panel_[3u*p];
SetMaskCount();MergeCounterResidualHalves<256u>(ah, hi, lo, al, u, 256u);
SetMaskCount();MergeCounterResidualHalves<128u>(hi, ah, lo, al, u, 128u);
SetMaskCount();MergeCounterResidualHalves<64u>(ah, hi, lo, al, u, 64u);

ResetMask();SetMaskNorm();
auto bb=negative64_[4224u],neg=panel_[2u*p],ss=panel_[3u*p],vv=panel_,tt=panel_[4u*p];
WriteCompensatedResidual(solveY_, bb, ah, lo, neg, ss, vv, tt, 50u);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::RefinePrivate50RowMajor64(Zq3 base)
{
auto offsets=panel_[14u*2048u].template Zq6<int32_t>();
ResidualOperations::PrepareColdResidualOffsets(offsets,rows_);

Adds(negative64_[4096u],solveY_,0.0f,64u);Zq1<PIPE_V>();
PrepareResidualX(base,0u);
RowMajorResidual50All64(base);

producerCorrection_=true;SolveInPlaceWithFactors(base,w_+OffD(base),1u);
Add(negative64_[4096u],negative64_[4096u],solveY_,64u);Zq1<PIPE_V>();
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::WideRefineSmallSingle64(Zq3 base)
{
if constexpr(StaticN==50u&&StaticK==2u){if(producerPrivate_){RefinePrivate50RowMajor64(base);return;}}
constexpr Zq2 p=2048u,pitch=64u;

const Zq2 zeroRoundRounds=1u;
if constexpr(StaticN==50u){
if(producerPrivate_&&zeroRoundRounds==0u){ZqB<Zq5::V_S>();Adds(negative64_[4096u],solveY_,0.0f,64u);Zq1<PIPE_V>();return;}
}
auto offsets=panel_[14u*p].template Zq6<Zq2>();
ResidualOperations::PrepareWideResidualOffsets(offsets,rows_);
ZqB<Zq5::S_V>();
if(StaticN==50u&&producerPrivate_){Adds(negative64_[4096u],solveY_,0.0f,64u);Zq1<PIPE_V>();}


for(Zq2 i=0;i<N();++i)negative64_[4224u].SetValue(i,w_[OffB(base)+i]);
for(Zq2 i=N();i<64u;++i)negative64_[4224u].SetValue(i,0.0f);
ZqB<Zq5::S_V>();
for(Zq2 step=0;step<zeroRoundRounds;++step){
ResidualOperations::PrepareWideResidualColumns(*this,base,p,pitch,1u);

for(Zq2 row=0;row<N();row+=32u){
const Zq2 slot=(row/32u)&1u;

WideSmallResidualSingle64(base,row,slot);
}
if constexpr(StaticN==50u)producerCorrection_=producerPrivate_;SolveInPlaceWithFactors(base,w_+OffD(base),1u);
if(StaticN==50u&&producerPrivate_){Add(negative64_[4096u],negative64_[4096u],solveY_,64u);Zq1<PIPE_V>();}
else for(Zq2 row=0;row<N();++row)SetX(base,row,0u,GetX(base,row,0u)+w_[OffD(base)+row]);
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::SmallResidualTile(Zq3 base,Zq2 row,Zq2 slot)
{
constexpr Zq2 p=512u,active=p,groupWidth=2u;
auto work=panel_,extra=panel_[10u*p],input=panel_[14u*p+slot*p];
Zq7<float>a=work,ah=work[2u*p],al=work[4u*p],u=work[9u*p];
Gather(a,input,panel_[16u*p].template Zq6<Zq2>(),0u,active);Zq1<PIPE_V>();
ResidualOperations::SplitTileInput(a,ah,al,vr_[ResidualMaskPlane()*VectorPitch()].template Zq6<uint16_t>(),active);
ResidualOperations::TileRhsProducts(work,extra,a,ah,al,p,groupWidth);
ResidualOperations::ReduceSmallTile(work,extra,ah,al,u,p,groupWidth);
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
const Zq2 count=N()-row<8u?N()-row:8u;
for(Zq2 rhs=0;rhs<2u;++rhs){
auto hi=rhs?work[8u*p]:work[6u*p];auto lo=rhs?extra[3u*p]:work[7u*p];
for(Zq2 lane=0;lane<count;++lane){
Pair d=square_math::Sum(w_[OffB(base)+Zq3(row+lane)*2u+rhs],-hi.GetValue(lane));
w_[OffD(base)+Zq3(row+lane)*2u+rhs]=square_math::RoundAdd(d.hi,square_math::RoundSub(d.lo,lo.GetValue(lane)));
}
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::RefineSmallRows(Zq3 base)
{
constexpr Zq2 p=512u,pitch=64u;
auto offsets=panel_[16u*p].template Zq6<Zq2>();
for(Zq2 feature=0;feature<pitch;++feature)for(Zq2 lane=0;lane<8u;++lane)
offsets.SetValue(feature*8u+lane,(lane*pitch+feature)*4u);
ZqB<Zq5::S_V>();Duplicate(panel_[14u*p],0.0f,2u*p);Zq1<PIPE_V>();
for(Zq2 step=0;step<2u;++step){
ResidualOperations::PrepareSmallResidualColumns(*this,base,p,pitch,2u);
SmallPrefetch(0u,0u);
for(Zq2 row=0;row<N();row+=8u){
const Zq2 slot=(row/8u)&1u;WaitFlag<Zq5::MTE2_V>(slot);
if(row+8u<N())SmallPrefetch(row+8u,slot^1u);
SmallResidualTile(base,row,slot);
}
SolveInPlaceWithFactors(base,w_+OffD(base),2u);
for(Zq2 row=0;row<N();++row)for(Zq2 col=0;col<2u;++col)
SetX(base,row,col,GetX(base,row,col)+w_[OffD(base)+Zq3(row)*2u+col]);
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::SmallResidualSingle64(Zq3 base,Zq2 row,Zq2 slot)
{
constexpr Zq2 p=512u,active=p,groupWidth=1u;
auto work=panel_,extra=panel_[10u*p],input=panel_[14u*p+slot*p];
Zq7<float>a=work,ah=work[2u*p],al=work[4u*p],u=work[9u*p];
Gather(a,input,panel_[16u*p].template Zq6<Zq2>(),0u,active);Zq1<PIPE_V>();
ResidualOperations::SplitTileInput(a,ah,al,vr_[ResidualMaskPlane()*VectorPitch()].template Zq6<uint16_t>(),active);
ResidualOperations::TileRhsProducts(work,extra,a,ah,al,p,groupWidth);
ResidualOperations::ReduceSmallTile(work,extra,ah,al,u,p,groupWidth);
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
const Zq2 count=N()-row<8u?N()-row:8u;
for(Zq2 rhs=0;rhs<1u;++rhs){
auto hi=rhs?work[8u*p]:work[6u*p];auto lo=rhs?extra[3u*p]:work[7u*p];
for(Zq2 lane=0;lane<count;++lane){
Pair d=square_math::Sum(w_[OffB(base)+Zq3(row+lane)+rhs],-hi.GetValue(lane));
if(StaticN==50u&&producerPrivate_)solveY_.SetValue(row+lane,square_math::RoundAdd(d.hi,square_math::RoundSub(d.lo,lo.GetValue(lane))));
else w_[OffD(base)+Zq3(row+lane)+rhs]=square_math::RoundAdd(d.hi,square_math::RoundSub(d.lo,lo.GetValue(lane)));
}
}
}
