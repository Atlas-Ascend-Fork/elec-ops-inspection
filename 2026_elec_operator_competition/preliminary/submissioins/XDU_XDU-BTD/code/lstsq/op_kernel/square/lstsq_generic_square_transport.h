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
__aicore__ inline void KernelLarge<StaticN, StaticK>::EmitSplitRhsAtomic50(Zq3 base,Zq2 batch,Zq2 firstRow)
{
const Zq2 tailRow=firstRow==0u?48u:0u;
WorkspaceOperations::PrepareAtomicRhs50(*this, base, firstRow, tailRow);
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
auto indices=panel_.template Zq6<int32_t>();
auto minimum=panel_[128u].template Zq6<int32_t>();
auto maximum=panel_[256u].template Zq6<int32_t>();
CreateVecIndex<int32_t>(indices,int32_t(0),Zq2(128));Zq1<PIPE_V>();
ShiftRight<int32_t>(indices,indices,int32_t(1),int32_t(128));Zq1<PIPE_V>();
Muls<int32_t>(indices,indices,int32_t(4),Zq2(128));Zq1<PIPE_V>();

WorkspaceOperations::CopyAtomicRhs50(*this, batch, firstRow, tailRow, indices, minimum, maximum);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::EmitSplitRhsDma(Zq3 base,Zq2 batch)
{
if constexpr(StaticN==50u&&StaticK==2u){
if(WorkspaceOperations::TryEmitSplitRhsAtomic50(*this,base,batch))return;
}
const Zq2 n=N();
{
WorkspaceOperations::PrepareSplitRhsDmaOutput(*this, base, n);
}
ZqB<Zq5::S_MTE3>();
if constexpr(StaticN==50u)ZqB<Zq5::V_MTE3>();
WorkspaceOperations::CopySplitRhsOutput(*this, batch, n);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::EmitSquareDma(Zq3 base,Zq2 batch)
{
if(UseSplitRhs()){EmitSplitRhsDma(base,batch);return;}
WorkspaceOperations::CopySquareOutput(output_, w_+OffX(base), x_, r_, batch, N(), K(), tiling_.residualElements);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::Emit(Zq3 base,Zq2 batch,Zq2 col,bool withResidual)
{
const Zq2 m=M(),n=N(),k=K();
WorkspaceOperations::EmitColumn(*this,base,batch,col,withResidual,m,n,k);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::Process()
{
const Zq2 m=M(),n=N(),k=K();
if(m==0||n==0||k==0)return;
const Zq3 per=PerBatch();


const auto outputSchedule = WorkspaceOperations::CheckOutputSchedule(x_, r_, m, n, k,
    tiling_.batches, tiling_.residualElements, UseDmaOutputs());
const bool serial = outputSchedule.serial;
const Zq3 rCount = outputSchedule.residualCount;
if(serial&&GetBlockIdx()!=0u)return;


if(!UseDmaOutputs()&&m==n&&r_!=nullptr&&GetBlockIdx()==0u)
for(Zq3 i=0;i<rCount;++i)r_[i]=0.0f;
const Zq2 stride=serial?1u:GetBlockNum();
const Zq2 tasks=tiling_.batches*(UseSplitRhs()?RhsWidth():1u);
for(Zq2 task=serial?0u:GetBlockIdx();task<tasks;task+=stride){
const Zq2 batch=UseSplitRhs()?task/RhsWidth():task;
currentRhs_=UseSplitRhs()?task%RhsWidth():0u;
const Zq3 base=Zq3(task)*per;
Load(base,batch);
if(m==n){
DecomposeSquare(base);
if(n==256u&&k<=4u)PrepareColumnFactors256(base);
if(UseColumnLocal())ColumnTriangularAll(base,w_+OffB(base),w_+OffX(base),k);
else if(UseVectorSolve()&&k<=4u)
VectorTriangularBatch(base,w_+OffB(base),k,w_+OffX(base),k,k);
else for(Zq2 col=0;col<k;++col)BackSubstitute(base,col);
Refine(base);
if(UseDmaOutputs())EmitSquareDma(base,batch);
else for(Zq2 col=0;col<k;++col)Emit(base,batch,col,false);
}else{
RectangularSolveOperations::Process(*this, base, batch, k);
}
}
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::ProducerLazySmallLu(Zq3 base)
{
constexpr Zq2 n=50u,pitch=56u,p=64u;
auto map=residualTile_.template Zq6<Zq2>();
auto logical=solveY_;
for(Zq2 i=0;i<n;++i)map.SetValue(i,i*4u);
diagonalKnown_=true;diagonalSafe_=true;
Zq3 active=(Zq3(1u)<<50u)-1u;
for(Zq2 j=0;j<n;++j){
const Zq2 pick=FusedLogicalPivot(factors_[j*pitch],map,logical,j);
w_[OffP(base)+j]=float(int32_t(pick));
const Zq2 old=map.GetValue(j),selected=map.GetValue(pick);
const float diagonal=logical.GetValue(pick);
map.SetValue(j,selected);map.SetValue(pick,old);
const Zq2 physical=selected/4u;active&=~(Zq3(1u)<<physical);
if(diagonal==0.0f){diagonalSafe_=false;w_[OffInv(base)+j]=0.0f;continue;}
const bool safe=AbsF(diagonal)>=1.0e-30f&&AbsF(diagonal)<=1.0e30f;
diagonalSafe_=diagonalSafe_&&safe;
const float inverse=safe?SeedInverse(diagonal):0.0f;w_[OffInv(base)+j]=inverse;
if(j+1u==n)continue;
Zq3 valid[2]={active,0u};
if(safe){
ZqB<Zq5::S_V>();
Muls(factors_[j*pitch],factors_[j*pitch],inverse,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});ResetMask();
}else{
for(Zq2 i=j+1u;i<n;++i){const Zq2 row=map.GetValue(i)/4u;
factors_.SetValue(j*pitch+row,square_math::Divide(factors_.GetValue(j*pitch+row),diagonal));}
}
auto columnOffsets=vr_[10u*p].template Zq6<Zq2>();
ZqB<Zq5::S_V>();
Gather(vr_[8u*p],factors_,columnOffsets,physical*4u,n);Zq1<PIPE_V>();
Muls(vr_[8u*p],vr_[8u*p],-1.0f,n);Zq1<PIPE_V>();
Brcb(block_,vr_[8u*p],Zq4((n+7u)/8u),Zq9{1,8});Zq1<PIPE_V>();
MulAddDst<float,float>(factors_[(j+1u)*pitch],factors_[j*pitch],block_[(j+1u)*8u],valid,Zq4(n-j-1u),Zq0{1,1,0,Zq4(pitch/8u),0,1});
ResetMask();ZqB<Zq5::V_S>();
}


auto savedMap=rows_.template Zq6<int32_t>();
ZqB<Zq5::S_V>();
Adds<int32_t>(savedMap,map.template Zq6<int32_t>(),int32_t(0),Zq2(50));Zq1<PIPE_V>();
ZqB<Zq5::V_S>();for(Zq2 i=50u;i<56u;++i)savedMap.SetValue(i,int32_t(i*4u));ZqB<Zq5::S_V>();
for(Zq2 c=0;c<8u;++c)
Adds<int32_t>(map[c*56u].template Zq6<int32_t>(),savedMap,int32_t(c*224u),Zq2(56));
Zq1<PIPE_V>();
if(diagonalSafe_){
for(Zq2 col=0;col<56u;col+=8u)
Gather(negative64_[col*56u],factors_,map,col*224u,448u);
Zq1<PIPE_V>();Muls(negative64_,negative64_,-1.0f,2800u);Zq1<PIPE_V>();negative64Ready_=true;
}else{
for(Zq2 col=0;col<56u;col+=8u){
Gather(block_,factors_,map,col*224u,448u);Zq1<PIPE_V>();
Adds(factors_[col*56u],block_,0.0f,448u);Zq1<PIPE_V>();
}
}
Adds<int32_t>(map.template Zq6<int32_t>(),savedMap,int32_t(0),Zq2(50));Zq1<PIPE_V>();
ZqB<Zq5::V_S>();


for(Zq2 i=50u;i<64u;++i)map.SetValue(i,i*4u);

}


template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::Rows50ToColumns()
{
constexpr Zq2 n=50u,rp=56u,p=64u;
auto snapshot=panel_;
auto offsets=panel_[rp*rp].template Zq6<int32_t>();
auto rowBase=rows_.template Zq6<int32_t>();
auto colBase=block_.template Zq6<int32_t>();
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<rp;++i){rowBase.SetValue(i,int32_t(i*rp*4u));colBase.SetValue(i,int32_t(i*4u));}
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
Adds(snapshot.template Zq6<int32_t>(),factors_.template Zq6<int32_t>(),int32_t(0),rp*rp);
Zq1<PIPE_V>();
Brcb(vr_.template Zq6<int32_t>(),colBase,Zq4(7),Zq9{1,8});
Zq1<PIPE_V>();
Add<int32_t>(offsets,rowBase,vr_.template Zq6<int32_t>(),Zq3(rp),Zq4(rp),
Zq0{1,1,0,7,0,1});
Zq1<PIPE_V>();
Gather(factors_,snapshot,offsets.template Zq6<Zq2>(),0u,n*rp);
Zq1<PIPE_V>();ZqB<Zq5::V_S>();
auto columnOffsets=vr_[10u*p].template Zq6<Zq2>();
for(Zq2 col=0;col<n;++col)columnOffsets.SetValue(col,col*rp*4u);


ZqB<Zq5::S_V>();
Duplicate(vr_,0.0f,p);Duplicate(vr_[p],0.0f,p);
Duplicate(vr_[12u*p],0.0f,p);ZqB<Zq5::V_S>();
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::SmallPrefetch(Zq2 baseRow,Zq2 slot)
{
constexpr Zq2 p=512u,pitch=64u;
const Zq2 n=N(),rows=n-baseRow<8u?n-baseRow:8u;
auto input=panel_[14u*p+slot*p];ZqB<Zq5::V_MTE2>(slot);
if(rows<8u){Duplicate(input,0.0f,p);ZqB<Zq5::V_MTE2>(slot);}
GlobalTensor<float>original;original.SetGlobalBuffer(a_+(Zq3(currentBatch_)*n+baseRow)*n,rows*n);
DataCopyPad(input,original,Zq8{uint16_t(rows),n*4u,0u,(pitch-n)/8u,0u},
DataCopyPadExtParams<float>{true,0u,Zq4((8u-n%8u)%8u),0.0f});
SetFlag<Zq5::MTE2_V>(slot);
}

template<Zq2 StaticN,Zq2 StaticK>
__aicore__ inline void KernelLarge<StaticN, StaticK>::PrefetchEightRows(Zq2 row,Zq2 slot)
{
auto input=ResidualInput(slot);
ZqB<Zq5::V_MTE2>(slot);
GlobalTensor<float>source;
const Zq2 n=N();
source.SetGlobalBuffer(a_+(Zq3(currentBatch_)*n+row)*n,8u*n);
DataCopyPad(input,source,Zq8{8u,n*4u,0u,(256u-n)/8u,0u},
DataCopyPadExtParams<float>{false,0u,0u,0.0f});
SetFlag<Zq5::MTE2_V>(slot);
}
