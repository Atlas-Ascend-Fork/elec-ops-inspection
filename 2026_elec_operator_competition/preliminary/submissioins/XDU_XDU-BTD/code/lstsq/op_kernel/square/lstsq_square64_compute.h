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

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline bool KernelLarge64<StaticN, StaticK, ColdTask>::BuildAugmented64Fast()
{
if constexpr (ColdTask) rowInverse64Ready_=false;
constexpr Zq2 n=64u,pitch=136u;
auto aug=panel_[24576u];
auto zeros=panel_[4096u].template Zq6<Zq2>();
auto rowOffsets=panel_[4224u].template Zq6<Zq2>();
auto dvec=panel_[4288u];
auto work=panel_[4544u];
ZqB<Zq5::V_S>();
Duplicate<float>(aug[64u],0.0f,Zq3(64),Zq4(64),1,17);
Adds<int32_t>(aug.template Zq6<int32_t>(),producerA_.template Zq6<int32_t>(),int32_t(0),Zq3(64),Zq4(64),UnaryRepeatParams{1,1,17,8});
Duplicate<Zq2>(zeros,Zq2(0),Zq3(128));
for(Zq2 i=0;i<n;++i){rowOffsets.SetValue(i,i*pitch*4u);aug.SetValue(i*pitch+64u+i,1.0f);}
ZqB<Zq5::S_V>();

EliminateAugmented64Fast(aug, zeros, rowOffsets, dvec);
if constexpr (ColdTask) {
Abs(work,aug[64u],Zq3(64),Zq4(64),UnaryRepeatParams{1,1,8,17});
} else {
auto offsets=panel_.template Zq6<int32_t>();
for(Zq2 c=0;c<n;++c)Adds(offsets[c*64u],rowOffsets.template Zq6<int32_t>(),int32_t(256u+c*4u),Zq3(64));
Zq1<PIPE_V>();
Gather(negative64_,aug,offsets.template Zq6<Zq2>(),0u,4096u);
Abs(work,negative64_,Zq3(4096));
}
Zq1<PIPE_V>();
ReduceMax(dvec,work,work[64u],int32_t(4096),true);ZqB<Zq5::V_S>();
const float maximum=dvec.GetValue(0u);
if(!(maximum>=0.0f&&maximum<=1.0e12f))return false;
if constexpr (ColdTask) PrepareRowInverseMap64();
augmented64Ready_=true;
return true;
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::ConsumeAugmented64(const __gm__ float*in,Zq2 stride)
{
if constexpr (ColdTask) {
    WorkspaceOperations::ConsumeRowInverse64(*this);
} else {
ZqB<Zq5::V_S>();
if(!producerCorrection_){ZqB<Zq5::S_V>();Adds(solveY_,negative64_[4224u],0.0f,64u);Zq1<PIPE_V>();}
ZqB<Zq5::S_V>();Brcb(block_,solveY_,Zq4(8),Zq9{1,8});Zq1<PIPE_V>();
auto products=panel_[24576u];
Mul(products,negative64_,block_,Zq3(64),Zq4(64),Zq0{1,1,0,8,8,1});Zq1<PIPE_V>();
for(Zq2 h=2048u;h>=64u;h/=2u){Add(products,products,products[h],h);Zq1<PIPE_V>();}
Adds(solveY_,products,0.0f,64u);ZqB<Zq5::V_S>();
}
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::ColumnTriangularBatch(Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride,Zq2 width)
{
    ColumnTriangularSolveOperations::ColumnTriangularBatch64(*this, base, in, out, stride, width);
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::WideSmallResidualSingle64(Zq3 base,Zq2 row,Zq2 slot,bool cachedA)
{
constexpr Zq2 p=2048u,active=p,groupWidth=1u;
auto work=panel_,extra=panel_[10u*p],input=producerA_[row*64u];
Zq7<float>a=work,ah=work[2u*p],al=work[4u*p],u=work[9u*p];
auto residualOffsets=producerPrivate_?factors_.template Zq6<Zq2>():panel_[14u*p].template Zq6<Zq2>();
if(cachedA){Adds(a,panel_[28672u][Zq3(row/32u)*2048u],0.0f,Zq3(active));}
else{Gather(a,input,residualOffsets,0u,active);}
Zq1<PIPE_V>();


if(producerPrivate_&&!cachedA){Adds(panel_[28672u][Zq3(row/32u)*2048u],a,0.0f,Zq3(active));}
ResidualOperations::ComputeWideSmallResidual(*this, base, row, work, a, ah, al);
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::WideRefineSmallSingle64(Zq3 base)
{
if constexpr (ColdTask) {
    if(!rowInverse64Ready_) PrepareColdResidualOffsets64();
if(producerPrivate_){Adds(negative64_[4096u],solveY_,0.0f,64u);Zq1<PIPE_V>();}
if(!augmented64Ready_){
ZqB<Zq5::V_S>();
for(Zq2 row=0u;row<64u;++row)negative64_.SetValue(4224u+row,w_[OffB(base)+row]);
ZqB<Zq5::S_V>();
}
} else {
    PrepareRefinementOffsets64();
ZqB<Zq5::S_V>();
if(StaticN==64u&&producerPrivate_){Adds(negative64_[4096u],solveY_,0.0f,64u);Zq1<PIPE_V>();}


if(!augmented64Ready_)for(Zq2 i=0;i<N();++i)negative64_[4224u].SetValue(i,w_[OffB(base)+i]);
ZqB<Zq5::S_V>();
}
for (Zq2 step=0;step<3u;++step) {
    if constexpr (ColdTask) {
        if(step>0u&&!augmented64Ready_) PrepareColdResidualOffsets64();
        PrepareResidualX(base,0u);
        RowMajorResidual64All64(base,step>0u&&augmented64Ready_);
    } else {
        ComputeWideResidual64(base, step);
    }
    if (ApplyRefinementStep64(base, step)) break;
}
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::PrepareResidualX(Zq3 base,Zq2 col)
{
    ResidualOperations::PrepareResidualX64(*this, base, col);
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::EliminateAugmented64Fast(Zq7<float> aug, Zq7<Zq2> zeros, Zq7<Zq2> rowOffsets, Zq7<float> dvec)
{
constexpr Zq2 n=64u,pitch=136u;
auto colvec=solveY_;
for(Zq2 j=0;j<n;++j){
Gather(colvec,aug,rowOffsets,j*4u,64u);
Gather(dvec,aug,zeros,(j*pitch+j)*4u,ColdTask?8u:128u);
Muls(colvec,colvec,-1.0f,64u);
Zq3 lane[2]={Zq3(1u)<<j,0u};
Duplicate<float>(colvec,0.0f,lane,Zq4(1),1,8);
if constexpr (ColdTask) {
    Div<float>(aug[j*pitch],aug[j*pitch],dvec,Zq3(64),Zq4(2),Zq0{1,1,0,8,8,0});
} else {
    Div(aug[j*pitch],aug[j*pitch],dvec,Zq3(128));
}
Brcb(block_,colvec,Zq4(8),Zq9{1,8});
const Zq2 aStart=(j/8u)*8u;
const Zq2 aLanes=64u-aStart;
const Zq2 iLanes=(j+8u)/8u*8u;
MulAddDst<float,float>(aug[aStart],aug[j*pitch+aStart],block_,Zq3(aLanes),Zq4(64),Zq0{1,1,0,17,0,1});
MulAddDst<float,float>(aug[64u],aug[j*pitch+64u],block_,Zq3(iLanes),Zq4(64),Zq0{1,1,0,17,0,1});

}
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::PrepareRowInverseMap64()
{
auto map=residualTile_.template Zq6<int32_t>();
auto rb=block_.template Zq6<int32_t>();
auto cb=block_[64u].template Zq6<int32_t>();
auto broadcast=block_[128u].template Zq6<int32_t>();
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
CreateVecIndex<int32_t>(rb,int32_t(0),Zq2(64));CreateVecIndex<int32_t>(cb,int32_t(0),Zq2(8));Zq1<PIPE_V>();
Muls<int32_t>(rb,rb,int32_t(256),Zq2(64));Muls<int32_t>(cb,cb,int32_t(4),Zq2(8));Zq1<PIPE_V>();
Brcb(broadcast,cb,Zq4(1),Zq9{1,8});Zq1<PIPE_V>();
Add<int32_t>(map,rb,broadcast,Zq3(64),Zq4(8),Zq0{1,1,0,8,0,1});Zq1<PIPE_V>();
rowInverse64Ready_=true;
}


template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::PrepareRefinementOffsets64()
{
constexpr Zq2 p=2048u,pitch=64u;
auto offsets=producerPrivate_?factors_.template Zq6<Zq2>():panel_[14u*p].template Zq6<Zq2>();
ResidualOperations::PrepareWideResidualOffsets(offsets,rows_);
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::PrepareColdResidualOffsets64()
{
auto offsets=panel_[14u*2048u].template Zq6<int32_t>();
ResidualOperations::PrepareColdResidualOffsets(offsets,rows_);
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::ComputeWideResidual64(Zq3 base, Zq2 step)
{
constexpr Zq2 p=2048u,pitch=64u;
ResidualOperations::PrepareWideResidualColumns(*this,base,p,pitch,1u);

for(Zq2 row=0;row<N();row+=32u){
const Zq2 slot=(row/32u)&1u;

WideSmallResidualSingle64(base,row,slot,step>0u);
}
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline bool KernelLarge64<StaticN, StaticK, ColdTask>::ApplyRefinementStep64(Zq3 base, Zq2 step)
{
producerCorrection_=producerPrivate_;SolveInPlaceWithFactors(base,w_+OffD(base),1u);
if(producerPrivate_){Add(negative64_[4096u],negative64_[4096u],solveY_,64u);Zq1<PIPE_V>();}
else for(Zq2 row=0u;row<64u;++row)SetX(base,row,0u,GetX(base,row,0u)+w_[OffD(base)+row]);
auto gateMax=panel_[9100u],gateWork=panel_[9200u];
Abs(gateWork,solveY_,Zq3(64));Zq1<PIPE_V>();


Abs(gateWork[64u],negative64_[4096u],Zq3(64));Zq1<PIPE_V>();
Adds(gateWork[64u],gateWork[64u],1.0f,Zq3(64));Zq1<PIPE_V>();
Div(gateWork,gateWork,gateWork[64u],Zq3(64));Zq1<PIPE_V>();
ReduceMax(gateMax,gateWork,gateWork[128u],int32_t(64),true);ZqB<Zq5::V_S>();
const float relativeCorrection=gateMax.GetValue(0u);
return !producerPrivate_||step>=2u||relativeCorrection<=3.0e-5f;
}

template<Zq2 StaticN,Zq2 StaticK,bool ColdTask>
__aicore__ inline void KernelLarge64<StaticN, StaticK, ColdTask>::RowMajorResidual64All64(Zq3 base,bool cachedHigh)
{
constexpr Zq2 p=4096u;
auto a=producerA_,ah=panel_,al=panel_[p],hi=panel_[2u*p],lo=panel_[3u*p],t=panel_[4u*p],u=t;


auto splitHigh=augmented64Ready_?panel_[20480u]:ah;


auto splitLow=rowInverse64Ready_?factors_:al;
ResidualOperations::RowMajorResidual64Products(*this, cachedHigh, a, splitHigh, splitLow, hi, lo, t);
ResidualOperations::RowMajorProducts<64u>(a,splitHigh,splitLow,hi,lo,t,vr_);
MergeStridedResidualHalves<32u, 32u, 64u, 8u>(ah, hi, lo, al, u);
ResetMask();SetMaskNorm();
MergeStridedResidualHalves<16u, 16u, 64u, 8u>(hi, ah, lo, al, u);
ResetMask();SetMaskNorm();
MergeStridedResidualHalves<8u, 8u, 64u, 8u>(ah, hi, lo, al, u);


auto offsets=rowInverse64Ready_?residualTile_:panel_[14u*2048u].template Zq6<Zq2>();
Gather(hi,ah,offsets,0u,512u);
Gather(panel_[p],lo,offsets,0u,512u);Zq1<PIPE_V>();
lo=panel_[p];al=panel_[3u*p];
SetMaskCount();MergeCounterResidualHalves<256u>(ah, hi, lo, al, u, 256u);
SetMaskCount();MergeCounterResidualHalves<128u>(hi, ah, lo, al, u, 128u);
SetMaskCount();MergeCounterResidualHalves<64u>(ah, hi, lo, al, u, 64u);

ResetMask();SetMaskNorm();


if(!producerPrivate_){
ResidualOperations::WriteScalarResidual64(w_+OffB(base),w_+OffD(base),ah,lo);
return;
}
ResetMask();SetMaskNorm();
auto bb=negative64_[4224u],neg=panel_[2u*p],ss=panel_[3u*p],vv=panel_,tt=panel_[4u*p];
WriteCompensatedResidual(solveY_, bb, ah, lo, neg, ss, vv, tt, 64u);
}
