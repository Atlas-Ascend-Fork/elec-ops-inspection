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

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::Process()
{
    WorkspaceOperations::ProcessFixedSquare<true>(*this, dual200_ ? GetBlockIdx()/2u : GetBlockIdx());
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::SolveCooperativeTask(Zq3 base, Zq2 batch)
{
const Zq3 per = PerBatch();
if(tiling_.batches<=4u&&GetBlockNum()==tiling_.batches*RhsWidth()){


const Zq3 sharedBase=Zq3(batch*3u)*per;
const Zq3 stageBase=OffA(sharedBase+per);
Zq1<PIPE_ALL>();SyncAll<true>();
CooperativeLu256(base,sharedBase,stageBase,currentRhs_);
PrepareColumnFactors256(sharedBase,currentRhs_,3u);
Zq1<PIPE_ALL>();SyncAll<true>();
chainShared_=true;chainSharedBase_=sharedBase;
}else DecomposeSquare(base);
if(!chainShared_)PrepareColumnFactors256(base);
VectorTriangularBatch(base,w_+OffB(base),K(),w_+OffX(base),K(),K());
Refine(base);
WorkspaceOperations::EmitSquareDma(*this,base,batch);
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::CooperativeLu256(Zq3 base,Zq3 sharedBase,Zq3 stageBase,Zq2 lane)
{

constexpr Zq2 n=256u,pitch=8u,tilePitch=264u,upperPitch=264u;
constexpr Zq2 tileRows=48u;
auto tailSeed=factors_[4096u];auto tailIndex=factors_[5120u].template Zq6<Zq2>();
for(Zq2 rank=0;rank<8u;++rank)for(Zq2 row=0;row<56u;++row)
tailIndex.SetValue(rank*56u+row,(rank*256u+(row<48u?row:0u))*4u);
ZqB<Zq5::S_V>();
__gm__ float*piv=w_+OffP(sharedBase);
GlobalTensor<float>matrix;matrix.SetGlobalBuffer(w_+OffLU(sharedBase),n*n);
Zq7<float>upper=block_,uRows=block_[8u*n],tile=block_[16u*256u+64u];
auto panelCoefficient=factors_,panelSeed=factors_[512u];
auto panelOffsets=rows_[64u].template Zq6<Zq2>();
for(Zq2 cc=0;cc<pitch;++cc)panelOffsets.SetValue(cc,cc*n*4u);

auto toColumns=residualTile_,toRows=residualTile_[pitch*n];


if(lane==0u){
auto rowBytes=vr_.template Zq6<int32_t>();
auto colBytes=vr_[256u].template Zq6<int32_t>();
for(Zq2 row=0;row<n;++row)rowBytes.SetValue(row,int32_t(row*4u));
for(Zq2 cc=0;cc<pitch;++cc)colBytes.SetValue(cc,int32_t(cc*n*4u));
ZqB<Zq5::S_V>();ResetMask();SetMaskNorm();
Brcb(toRows.template Zq6<int32_t>(),rowBytes,Zq4(32),Zq9{1,8});Zq1<PIPE_V>();
Add<int32_t>(toRows.template Zq6<int32_t>(),toRows.template Zq6<int32_t>(),colBytes,Zq3(64),Zq4(32),Zq0{1,1,0,8,8,0});Zq1<PIPE_V>();
for(Zq2 bit=0;bit<3u;++bit){Add<int32_t>(rowBytes,rowBytes,rowBytes,n);Zq1<PIPE_V>();}
for(Zq2 cc=0;cc<pitch;++cc)Adds<int32_t>(toColumns[cc*n].template Zq6<int32_t>(),rowBytes,int32_t(cc*4u),n);
ZqB<Zq5::V_S>();
}
ZqB<Zq5::S_V>();
GlobalTensor<float>stage;stage.SetGlobalBuffer(w_+stageBase,8192u);
for(Zq2 jb=0;jb<n;jb+=pitch){
const Zq2 je=jb+pitch,tail=n-je;
if(lane==0u){
DataCopyPad(upper[jb*pitch],matrix[jb*n+jb],
Zq8{uint16_t(n-jb),pitch*4u,(n-pitch)*4u,0u,0u},
DataCopyPadExtParams<float>{false,0u,0u,0.0f});
ZqB<Zq5::MTE2_V>();


for(Zq2 cc=0;cc<pitch;++cc)
Gather(panel_[cc*n+jb],upper,toColumns[cc*n+jb],0u,n-jb);
ZqB<Zq5::V_S>();
for(Zq2 j=jb;j<je;++j){
const Zq2 c=j-jb;
const Zq2 pick=PivotIndex(panel_[c*n],j);
piv[j]=float(int32_t(pick));
if(pick!=j){


for(Zq2 cc=0;cc<pitch;++cc){
const float temp=panel_.GetValue(cc*n+j);
panel_.SetValue(cc*n+j,panel_.GetValue(cc*n+pick));
panel_.SetValue(cc*n+pick,temp);
}
DataCopy(upper,matrix[j*n],n);
DataCopy(upper[n],matrix[pick*n],n);
ZqB<Zq5::MTE2_MTE3>();
DataCopy(matrix[j*n],upper[n],n);
DataCopy(matrix[pick*n],upper,n);
ZqB<Zq5::MTE3_MTE2>();
}
const float diagonal=panel_.GetValue(c*n+j);
if(diagonal==0.0f){w_[OffInv(sharedBase)+j]=0.0f;continue;}
const bool safe=AbsF(diagonal)>=1.0e-30f&&AbsF(diagonal)<=1.0e30f;
const float inverse=safe?SeedInverse(diagonal):0.0f;
w_[OffInv(sharedBase)+j]=inverse;
if(j+1u==n)continue;
const Zq2 transferStart=(j+1u)/8u*8u;
const Zq2 transferCount=n-transferStart;
if(safe&&transferCount>64u){


const Zq2 head=j+1u-transferStart;
Zq3 valid[2]={~Zq3(0u)<<head,0u};
ZqB<Zq5::S_V>();ResetMask();SetMaskNorm();
Muls(panel_[c*n+transferStart],panel_[c*n+transferStart],inverse,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});
ResetMask();
Muls(panel_[c*n+transferStart+64u],panel_[c*n+transferStart+64u],inverse,transferCount-64u);
Zq1<PIPE_V>();
if(c<3u){


Gather(panelSeed,panel_,panelOffsets,j*4u,8u);Zq1<PIPE_V>();
Muls(panelSeed,panelSeed,-1.0f,8u);Zq1<PIPE_V>();
Brcb(panelCoefficient,panelSeed,Zq4(1),Zq9{8,64});Zq1<PIPE_V>();
MulAddDst<float,float>(panel_[(c+1u)*n+transferStart],panel_[c*n+transferStart],panelCoefficient[(c+1u)*64u],
valid,Zq4(pitch-c-1u),Zq0{1,1,0,32,0,8});
ResetMask();
for(Zq2 chunk=64u;chunk<transferCount;chunk+=64u){
const Zq2 width=transferCount-chunk<64u?transferCount-chunk:64u;
MulAddDst<float,float>(panel_[(c+1u)*n+transferStart+chunk],panel_[c*n+transferStart+chunk],panelCoefficient[(c+1u)*64u],
Zq3(width),Zq4(pitch-c-1u),Zq0{1,1,0,32,0,8});ResetMask();
}
}else{
for(Zq2 cc=c+1u;cc<pitch;++cc){
const float u=panel_.GetValue(cc*n+j);
Axpy<float,float>(panel_[cc*n+transferStart],panel_[c*n+transferStart],-u,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});
ResetMask();
Axpy(panel_[cc*n+transferStart+64u],panel_[c*n+transferStart+64u],-u,int32_t(transferCount-64u));
}
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();continue;
}
if(safe&&transferCount<=64u){
const Zq2 head=j+1u-transferStart;
const Zq3 low=transferCount==64u?~Zq3(0u):(Zq3(1u)<<transferCount)-1u;
Zq3 valid[2]={low&(~Zq3(0u)<<head),0u};
ZqB<Zq5::S_V>();SetMaskNorm();
Muls(panel_[c*n+transferStart],panel_[c*n+transferStart],inverse,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});
Zq1<PIPE_V>();
for(Zq2 cc=c+1u;cc<pitch;++cc){
const float u=panel_.GetValue(cc*n+j);
Axpy<float,float>(panel_[cc*n+transferStart],panel_[c*n+transferStart],-u,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});
}
ResetMask();ZqB<Zq5::V_S>();continue;
}
const Zq2 start=(j+1u)/8u*8u,count=n-start;


for(Zq2 i=start;i<=j;++i)rows_.SetValue(i-start,panel_.GetValue(c*n+i));
if(safe){
ZqB<Zq5::S_V>();
Muls(panel_[c*n+start],panel_[c*n+start],inverse,count);
ZqB<Zq5::V_S>();
}else for(Zq2 i=j+1u;i<n;++i)
panel_.SetValue(c*n+i,square_math::Divide(panel_.GetValue(c*n+i),diagonal));
for(Zq2 i=start;i<=j;++i)panel_.SetValue(c*n+i,0.0f);
ZqB<Zq5::S_V>();
if(count>=128u&&c<3u){
SetMaskNorm();ResetMask();
if(c+1u<pitch){
Gather(panelSeed,panel_,panelOffsets,j*4u,8u);Zq1<PIPE_V>();
Muls(panelSeed,panelSeed,-1.0f,8u);Zq1<PIPE_V>();
Brcb(panelCoefficient,panelSeed,Zq4(1),Zq9{8,64});Zq1<PIPE_V>();
for(Zq2 chunk=start;chunk<n;chunk+=64u){
const Zq2 width=n-chunk<64u?n-chunk:64u;
MulAddDst<float,float>(panel_[(c+1u)*n+chunk],panel_[c*n+chunk],panelCoefficient[(c+1u)*64u],
Zq3(width),Zq4(pitch-c-1u),Zq0{1,1,0,32,0,8});
}
ResetMask();
}
ZqB<Zq5::V_S>();
}else{
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,count);
for(Zq2 cc=c+1u;cc<pitch;++cc){
const float u=panel_.GetValue(cc*n+j);
Axpy<float,float,false>(panel_[cc*n+start],panel_[c*n+start],-u,
Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
}
for(Zq2 i=start;i<=j;++i)panel_.SetValue(c*n+i,rows_.GetValue(i-start));
}


ZqB<Zq5::MTE3_S>();
ZqB<Zq5::S_V>();
Gather(upper[jb*pitch],panel_,toRows[jb*pitch],0u,(n-jb)*pitch);
ZqB<Zq5::V_S>();
ZqB<Zq5::S_MTE3>();
DataCopyPad(matrix[jb*n+jb],upper[jb*pitch],
Zq8{uint16_t(n-jb),pitch*4u,0u,(n-pitch)*4u,0u});


if(tail){

Duplicate(uRows,0.0f,8u*264u);ZqB<Zq5::V_MTE2>();
DataCopyPad(uRows,matrix[jb*n+je],
Zq8{uint16_t(pitch),tail*4u,je*4u,je/8u+1u,0u},
DataCopyPadExtParams<float>{false,0u,0u,0.0f});
ZqB<Zq5::MTE2_V>();
ZqB<Zq5::S_V>();ResetMask();SetMaskNorm();
for(Zq2 prev=0;prev+1u<pitch;++prev)
Brcb(factors_[prev*64u],panel_[prev*n+jb],Zq4(1),Zq9{1,8});
Zq1<PIPE_V>();Muls(factors_,factors_,-1.0f,448u);Zq1<PIPE_V>();
for(Zq2 prev=0;prev+1u<pitch;++prev){
for(Zq2 chunk=0;chunk<tail;chunk+=64u){
const Zq2 count=tail-chunk<64u?tail-chunk:64u;
MulAddDst<float,float>(uRows[(prev+1u)*upperPitch+chunk],uRows[prev*upperPitch+chunk],factors_[prev*64u+(prev+1u)*8u],
Zq3(count),Zq4(pitch-prev-1u),Zq0{1,1,0,33,0,1});
}
Zq1<PIPE_V>();
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_MTE3>();
DataCopyPad(matrix[jb*n+je],uRows,
Zq8{uint16_t(pitch),tail*4u,je/8u+1u,je*4u,0u});


ZqB<Zq5::S_V>();
Muls(panel_,panel_,-1.0f,pitch*n);Zq1<PIPE_V>();
ZqB<Zq5::V_MTE3>();
DataCopy(stage,panel_,2048u);DataCopy(stage[4096u],uRows,2112u);
}
}


Zq1<PIPE_ALL>();SyncAll<true>();
if(tail){
if(lane!=0u){
ZqB<Zq5::V_MTE2>();DataCopy(panel_,stage,2048u);
DataCopy(uRows,stage[4096u],2112u);ZqB<Zq5::MTE2_V>();
}
const Zq2 tiles=(tail+tileRows-1u)/tileRows;

for(Zq2 look=0;look<2u;++look){
const Zq2 ti=lane+look*3u;
if(ti>=tiles)continue;
const Zq2 ib=je+ti*tileRows,count=n-ib<tileRows?n-ib:tileRows;
ZqB<Zq5::V_MTE2>(look);
DataCopyPad(block_[16u*n+64u+look*tileRows*tilePitch],matrix[ib*n+je],Zq8{uint16_t(count),tail*4u,je*4u,je/8u+1u,0u},DataCopyPadExtParams<float>{false,0u,0u,0.0f});
SetFlag<Zq5::MTE2_V>(look);
}
for(Zq2 ti=lane;ti<tiles;ti+=3u){
const Zq2 ib=je+ti*tileRows,slot=(ti-lane)/3u;
const Zq2 count=n-ib<tileRows?n-ib:tileRows;
Zq7<float>current=block_[16u*n+64u+slot*tileRows*tilePitch];
WaitFlag<Zq5::MTE2_V>(slot);
const Zq2 coefficientPitch=count==48u?56u:count+1u;
if(count==48u){
Gather(tailSeed,panel_,tailIndex,ib*4u,448u);Zq1<PIPE_V>();
Brcb(factors_,tailSeed,Zq4(56),Zq9{1,8});
}else for(Zq2 pp=0;pp<pitch;++pp)
Brcb(factors_[pp*coefficientPitch*8u],panel_[pp*n+ib],Zq4(count/8u),Zq9{1,8});
Zq1<PIPE_V>();
BlockedLu256::MultiplyTrailingTile(*this,current,uRows,count,coefficientPitch,tail);
ResetMask();ZqB<Zq5::V_MTE3>(slot);
DataCopyPad(matrix[ib*n+je],current,
Zq8{uint16_t(count),tail*4u,je/8u+1u,je*4u,0u});

}
}
Zq1<PIPE_ALL>();SyncAll<true>();
}
if(lane==0u)DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(matrix);
Zq1<PIPE_ALL>();SyncAll<true>();


}
