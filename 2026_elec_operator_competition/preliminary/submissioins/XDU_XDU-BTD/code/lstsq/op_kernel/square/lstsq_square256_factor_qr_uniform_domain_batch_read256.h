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

struct QrUniformDomainOperations {
template<bool EnableBoundedH3>
__aicore__ static inline Zq2 NormalizeDegree(Zq2 logicalN, Zq2 degree, Zq2& qrDegree, Zq2& reason)
{
qrDegree=degree;
if constexpr(EnableBoundedH3){
if(logicalN==256u){
const Zq2 stepsDegree=degree&0xffu;
const bool pooledH1=(degree&0x100u)!=0u;
if((degree&~0x1ffu)!=0u||(pooledH1&&stepsDegree==1u))reason|=32u;
qrDegree=stepsDegree;degree=pooledH1?1u:stepsDegree;
}
}
return degree;
}

template<bool EnableBoundedH3>
__aicore__ static inline bool IsSupportedDegree(Zq2 degree, Zq2 qrDegree)
{
if constexpr(EnableBoundedH3)
return qrDegree==1u||qrDegree==2u||qrDegree==3u||qrDegree==4u||qrDegree==7u;
return degree==1u||degree==7u;
}

template<bool EnableBoundedH3>
__aicore__ static inline Zq2 AddDegreeReasons(Zq2 degree, float err, Zq2 qrDegree, Zq2 reason)
{
if(!(err>=0.0f))reason|=2u;
if(degree==1u&&!(err<0.0009765625f))reason|=(EnableBoundedH3?128u:4u);
if(degree==7u&&!(err<0.015625f))reason|=8u;
if constexpr(EnableBoundedH3){
if(degree==3u&&!(err<0.001953125f))reason|=64u;
if(degree==2u&&!(err<0.000244140625f))reason|=128u;
if(degree==4u&&!(err<0.015625f))reason|=128u;
}
if(!IsSupportedDegree<EnableBoundedH3>(degree,qrDegree))reason|=32u;
return reason;
}
template<bool EnableBoundedH3>
__aicore__ static inline Zq2 ValidateDegree(Zq2 logicalN, Zq2 degree, float err, Zq2& qrDegree, Zq2 reason)
{
degree=NormalizeDegree<EnableBoundedH3>(logicalN,degree,qrDegree,reason);
return AddDegreeReasons<EnableBoundedH3>(degree,err,qrDegree,reason);
}
};

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline bool KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::QrUniformDomainBatchRead256(Zq2 batches,Zq2 batch,Zq2 rhs,Zq3 shared,Zq2 logicalN,Zq2 rhsWidth)
{
qrLogicalN_=logicalN;qrRhsWidth_=rhsWidth;


ZqB<Zq5::V_S>();GlobalTensor<float>staged;
staged.SetGlobalBuffer(w_+shared+352768u+rhs*256u,256u);
ZqB<Zq5::V_MTE2>();DataCopy(output_[512u],staged,256u);
GlobalTensor<float>meta,allErrors;
meta.SetGlobalBuffer(w_,Zq3(batches-1u)*397824u+208u);
allErrors.SetGlobalBuffer(w_+344320u,Zq3(batches-1u)*397824u+64u);
DataCopy(factors_,meta,DataCopyParams{uint16_t(batches),26u,49702u,0u});
DataCopy(factors_[1024u],allErrors,DataCopyParams{uint16_t(batches),8u,49720u,0u});
ZqB<Zq5::MTE2_V>();ZqB<Zq5::MTE2_S>();qrInverseScale_=factors_.GetValue(batch*208u+40u);

Zq1<PIPE_ALL>();if constexpr(!ColumnFactored256){SyncAll<true>();}
Zq2 reason=0u;
for(Zq2 b=0;b<batches;++b){
auto batchMeta=factors_[b*208u];
auto words=batchMeta.template Zq6<Zq2>();
if(words.GetValue(42u)!=1u)reason|=1u;
if constexpr(ColumnFactored256){if(words.GetValue(44u)!=0x43464132u)reason|=32u;}
const Zq2 errorBase=1024u+b*64u;
float err=factors_.GetValue(errorBase);for(Zq2 lane=1u;lane<4u;++lane)err+=factors_.GetValue(errorBase+lane*16u);


reason=QrUniformDomainOperations::ValidateDegree<EnableBoundedH3>(qrLogicalN_,words.GetValue(43u),err,qrDegree_,reason);
for(Zq2 lane=0;lane<rhsWidth;++lane)if(batchMeta.GetValue(160u+lane*16u)!=1.0f)reason|=16u;
}

Zq1<PIPE_ALL>();if constexpr(!ColumnFactored256){SyncAll<true>();}
const Zq2 nonE=qrLogicalN_==200u?reason:reason&(EnableBoundedH3?(1u|2u|16u|32u|64u|128u):(1u|2u|16u|32u));
if(nonE!=0u){

return false;
}
return true;
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline bool KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::QrUniformDomainBatchRead200(Zq2 batches,Zq2 batch,Zq2 rhs,Zq3 shared)
{
qrLogicalN_=200u;qrRhsWidth_=4u;
ZqB<Zq5::V_S>();GlobalTensor<float>staged;
staged.SetGlobalBuffer(w_+shared+352768u+rhs*256u,256u);
ZqB<Zq5::V_MTE2>();DataCopy(output_[512u],staged,256u);
GlobalTensor<float>meta,allErrors;
meta.SetGlobalBuffer(w_,Zq3(batches-1u)*397824u+224u);
allErrors.SetGlobalBuffer(w_+344320u,Zq3(batches-1u)*397824u+64u);
DataCopy(factors_,meta,DataCopyParams{uint16_t(batches),28u,49700u,0u});
DataCopy(factors_[1024u],allErrors,DataCopyParams{uint16_t(batches),8u,49720u,0u});
ZqB<Zq5::MTE2_V>();ZqB<Zq5::MTE2_S>();
qrInverseScale_=factors_.GetValue(batch*224u+40u);
Zq1<PIPE_ALL>();
Zq2 reason=0u;
for(Zq2 b=0u;b<batches;++b){
auto header=factors_[b*224u];auto words=header.template Zq6<Zq2>();
if(words.GetValue(42u)!=1u)reason|=1u;
const Zq2 errorBase=1024u+b*64u;
float err=factors_.GetValue(errorBase);for(Zq2 lane=1u;lane<4u;++lane)err+=factors_.GetValue(errorBase+lane*16u);
Zq2 degree=words.GetValue(43u);qrDegree_=degree;
if(!(err>=0.0f))reason|=2u;
if(degree==1u&&!(err<0.0009765625f))reason|=4u;
if(degree==7u&&!(err<0.015625f))reason|=8u;
if(degree!=1u&&degree!=7u)reason|=32u;
for(Zq2 lane=0u;lane<4u;++lane)if(header.GetValue(160u+lane*16u)!=1.0f)reason|=16u;
}
Zq1<PIPE_ALL>();
return reason==0u;
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline bool KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::QrUniformDomain(Zq2 batches,Zq2 batch,Zq2 rhs,Zq3 shared,Zq2 logicalN,Zq2 rhsWidth)
{
if constexpr(CompactConsumer200){
if(logicalN==200u&&rhsWidth==4u&&batches>=1u&&batches<=4u)
return QrUniformDomainBatchRead200(batches,batch,rhs,shared);
}

if constexpr(EnableBoundedH3){
if(logicalN==256u&&rhsWidth==3u&&batches>=1u&&batches<=4u)
return QrUniformDomainBatchRead256(batches,batch,rhs,shared,logicalN,rhsWidth);
}
qrLogicalN_=logicalN;qrRhsWidth_=rhsWidth;


ZqB<Zq5::V_S>();GlobalTensor<float>staged;
staged.SetGlobalBuffer(w_+shared+352768u+rhs*256u,256u);
ZqB<Zq5::V_MTE2>();DataCopy(output_[512u],staged,256u);
GlobalTensor<float>meta;meta.SetGlobalBuffer(w_+shared+40u,8u);DataCopy(factors_,meta,8u);
ZqB<Zq5::MTE2_V>();ZqB<Zq5::MTE2_S>();qrInverseScale_=factors_.GetValue(0u);

Zq1<PIPE_ALL>();if constexpr(!CompactConsumer200){SyncAll<true>();}
Zq2 reason=0u;
for(Zq2 b=0;b<batches;++b){
meta.SetGlobalBuffer(w_+Zq3(b)*397824u,rhsWidth==4u?224u:208u);DataCopy(factors_,meta,rhsWidth==4u?224u:208u);ZqB<Zq5::MTE2_S>();
auto words=factors_.template Zq6<Zq2>();
if(words.GetValue(42u)!=1u)reason|=1u;
GlobalTensor<float>errorLines;errorLines.SetGlobalBuffer(w_+Zq3(b)*397824u+344320u,64u);
DataCopy(factors_[256u],errorLines,64u);ZqB<Zq5::MTE2_S>();
float err=factors_.GetValue(256u);for(Zq2 lane=1u;lane<4u;++lane)err+=factors_.GetValue(256u+lane*16u);


reason=QrUniformDomainOperations::ValidateDegree<EnableBoundedH3>(qrLogicalN_,words.GetValue(43u),err,qrDegree_,reason);
for(Zq2 lane=0;lane<rhsWidth;++lane)if(factors_.GetValue(160u+lane*16u)!=1.0f)reason|=16u;
}

Zq1<PIPE_ALL>();if constexpr(!CompactConsumer200){SyncAll<true>();}
const Zq2 nonE=qrLogicalN_==200u?reason:reason&(EnableBoundedH3?(1u|2u|16u|32u|64u|128u):(1u|2u|16u|32u));
if(nonE!=0u){

return false;
}
return true;
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::QrCubeDecision(bool accepted,Zq3 shared)
{
if((dual200_?GetBlockIdx()/2u:GetBlockIdx())%4u==0u&&(!dual200_||GetSubBlockIdx()==0u)){
GlobalTensor<float>input,decision;input.SetGlobalBuffer(w_+shared+344576u,4096u);decision.SetGlobalBuffer(w_+shared+344448u,16u);
const bool privateCompact=CompactConsumer200&&dual200_&&qrLogicalN_==200u;
if(privateCompact){
ZqB<Zq5::S_V>();Duplicate(factors_,0.0f,16u);ZqB<Zq5::V_S>();
}else{
ZqB<Zq5::S_V>();
Duplicate(factors_,0.0f,4096u);ZqB<Zq5::V_MTE3>();DataCopy(input,factors_,4096u);
if(qrLogicalN_==200u){
GlobalTensor<float>pad;pad.SetGlobalBuffer(w_+shared+348672u+224u*16u,512u);DataCopy(pad,factors_,512u);
}
ZqB<Zq5::MTE3_S>();
}
factors_.SetValue(0u,accepted?1.0f:0.0f);ZqB<Zq5::S_MTE3>();DataCopy(decision,factors_,16u);ZqB<Zq5::MTE3_S>();
}
Zq1<PIPE_ALL>();SyncAll<true>();CrossCoreSetFlag<2,PIPE_MTE3>(0);CrossCoreWaitFlag(1);
Zq1<PIPE_ALL>();SyncAll<true>();
if constexpr(ColumnFactored256){
if(accepted&&qrLogicalN_==256u){


const Zq2 offset=(GetBlockIdx()%8u)*4096u;
GlobalTensor<float>oldH,cross;oldH.SetGlobalBuffer(w_+shared+213248u+offset,4096u);
cross.SetGlobalBuffer(w_+shared+278784u+offset,4096u);
ZqB<Zq5::V_MTE2>();DataCopy(factors_,oldH,4096u);DataCopy(factors_[4096u],cross,4096u);
ZqB<Zq5::MTE2_V>();Sub(factors_,factors_,factors_[4096u],4096u);
ZqB<Zq5::V_MTE3>();DataCopy(cross,factors_,4096u);ZqB<Zq5::MTE3_V>();
}
}
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::ProcessQrCubeRhs(Zq2 batch,Zq2 rhs,Zq3 shared)
{
const bool active=(dual200_?(GetSubBlockIdx()==0u&&(GetBlockIdx()/2u)%4u<qrRhsWidth_):GetBlockIdx()%4u<qrRhsWidth_);
currentBatch_=batch;currentRhs_=rhs;
pathLocalRhs_=false;pathHasIterate_=false;chainCachesReady_=false;chainSafe_=false;chainShared_=false;

if(compactQr32_&&!(CompactConsumer200&&dual200_&&qrLogicalN_==200u)){auto offsets=ProducerResidualOffsets16();ZqB<Zq5::V_S>();
if(qrLogicalN_==200u){
auto ids=offsets.template Zq6<int32_t>();
CreateVecIndex<int32_t>(ids,int32_t(0),Zq2(32));Zq1<PIPE_V>();
Muls<int32_t>(ids,ids,int32_t(832),Zq2(32));Zq1<PIPE_V>();
for(Zq2 f=1u;f<8u;++f)Adds<int32_t>(ids[f*32u],ids,int32_t(f*4u),Zq2(32));
Zq1<PIPE_V>();
if constexpr(CompactConsumer200){
auto ids40=panel_[1024u].template Zq6<int32_t>();
CreateVecIndex<int32_t>(ids40,int32_t(0),Zq2(40));Zq1<PIPE_V>();
Muls<int32_t>(ids40,ids40,int32_t(832),Zq2(40));Zq1<PIPE_V>();
for(Zq2 f=1u;f<8u;++f)Adds<int32_t>(ids40[f*40u],ids40,int32_t(f*4u),Zq2(40));
Zq1<PIPE_V>();
}
}else{
if constexpr(EnableBoundedH3){
if(qrLogicalN_==256u){
auto ids=offsets.template Zq6<int32_t>();
if constexpr(ColumnFactored256){
CreateVecIndex<int32_t>(ids,int32_t(0),32u);Zq1<PIPE_V>();
Muls<int32_t>(ids,ids,int32_t(1024),32u);Zq1<PIPE_V>();
for(Zq2 f=1u;f<8u;++f)Adds<int32_t>(ids[f*32u],ids,int32_t(f*4u),32u);
Zq1<PIPE_V>();
}else{
CreateVecIndex<int32_t>(ids,int32_t(0),Zq2(32));Zq1<PIPE_V>();
Muls<int32_t>(ids,ids,int32_t(1024),Zq2(32));Zq1<PIPE_V>();
for(Zq2 f=1u;f<8u;++f)Adds<int32_t>(ids[f*32u],ids,int32_t(f*4u),Zq2(32));
Zq1<PIPE_V>();
}
}else{for(Zq2 f=0;f<8u;++f)for(Zq2 r=0;r<32u;++r)offsets.SetValue(f*32u+r,(r*(qrLogicalN_==200u?208u:256u)+f)*4u);}
}else{for(Zq2 f=0;f<8u;++f)for(Zq2 r=0;r<32u;++r)offsets.SetValue(f*32u+r,(r*(qrLogicalN_==200u?208u:256u)+f)*4u);}
}
ZqB<Zq5::S_V>();}

if(dual200_&&qrLogicalN_==200u){
auto ids=panel_[512u].template Zq6<int32_t>();
auto rowIds=block_[17408u].template Zq6<int32_t>();
auto quot=block_[17920u].template Zq6<int32_t>();
ZqB<Zq5::V_S>();
auto packed=quot.template Zq6<Zq3>();
packed.SetValue(0u,0x0000008000000000ull);
packed.SetValue(1u,0x0000018000000100ull);
packed.SetValue(2u,0x0000008000000000ull);
packed.SetValue(3u,0x0000018000000100ull);
packed.SetValue(4u,0x0000008000000000ull);
packed.SetValue(5u,0x0000018000000100ull);
packed.SetValue(6u,0x0000008000000000ull);
packed.SetValue(7u,0x0000018000000100ull);
CreateVecIndex<int32_t>(rowIds,int32_t(0),Zq2(32));Zq1<PIPE_V>();
Muls<int32_t>(rowIds,rowIds,int32_t(4),Zq2(32));Zq1<PIPE_V>();
Brcb(ids,rowIds,Zq4(4),Zq9{2u,16u});
Brcb(ids[8u],rowIds,Zq4(4),Zq9{2u,16u});Zq1<PIPE_V>();
Add<int32_t>(ids,ids,quot,Zq3(16),Zq4(32),Zq0{1u,1u,1u,2u,2u,0u});Zq1<PIPE_V>();
SetMaskNorm();ResetMask();
}
if(active){
ZqB<Zq5::V_S>();auto offsets=output_[1792u].template Zq6<Zq2>();
if(qrLogicalN_==200u){
auto ids=offsets.template Zq6<int32_t>();
CreateVecIndex<int32_t>(ids,int32_t(0),Zq2(256));Zq1<PIPE_V>();
Muls<int32_t>(ids,ids,int32_t(64),Zq2(256));Zq1<PIPE_V>();
Adds<int32_t>(ids,ids,int32_t(currentRhs_*4u),Zq2(256));Zq1<PIPE_V>();
}else{
if constexpr(EnableBoundedH3){
if(qrLogicalN_==256u){
auto ids=offsets.template Zq6<int32_t>();
CreateVecIndex<int32_t>(ids,int32_t(0),Zq2(256));Zq1<PIPE_V>();
Muls<int32_t>(ids,ids,int32_t(64),Zq2(256));Zq1<PIPE_V>();
Adds<int32_t>(ids,ids,int32_t(currentRhs_*4u),Zq2(256));Zq1<PIPE_V>();
}else{for(Zq2 row=0;row<256u;++row)offsets.SetValue(row,row*32u);}
}else{for(Zq2 row=0;row<256u;++row)offsets.SetValue(row,row*32u);}
}
ZqB<Zq5::S_V>();Adds(output_[256u],output_[512u],0.0f,256u);Zq1<PIPE_V>();
}
if constexpr(!EnableBoundedH3){
if(qrLogicalN_==256u&&qrRhsWidth_==3u&&active){
GlobalTensor<float>dst;dst.SetGlobalBuffer(w_+shared+352768u+currentRhs_*256u,256u);
ZqB<Zq5::V_MTE3>();DataCopy(dst,output_[512u],256u);ZqB<Zq5::MTE3_S>();
}
}
if constexpr(CompactConsumer200)QrCubeApply(shared,active,false,false,true);else QrCubeApply(shared,active);if(active){Adds(output_,solveY_,0.0f,256u);Zq1<PIPE_V>();}
if constexpr(EnableBoundedH3){
if(qrLogicalN_==256u&&qrRhsWidth_==3u){
QrCoopCacheB(shared);
const Zq2 row=(GetBlockIdx()%8u)*32u;
GlobalTensor<float>original;original.SetGlobalBuffer(a_+(Zq3(currentBatch_)*256u+row)*256u,8192u);
ZqB<Zq5::V_MTE2>();DataCopy(ProducerResidualInput16(),original,8192u);ZqB<Zq5::MTE2_V>();
}
}else{if(!dual200_&&qrLogicalN_==256u&&qrRhsWidth_==3u&&!active)QrCoopCacheB(shared);}
constexpr Zq2 q=4128u,p=4096u;const Zq2 steps=EnableBoundedH3?(qrDegree_==1u?2u:3u):(qrDegree_==7u?3u:2u);
for(Zq2 step=0;step<steps;++step){
if(dual200_)QrDualResidual200(shared,active);
else if(qrLogicalN_==256u&&qrRhsWidth_==3u)QrCoopResidual(shared,active);
else if(compactQr32_){if(active)QrResidualRows32(0u,200u);}
else{
if(active){
auto residualOffsets=ProducerResidualOffsets16();ZqB<Zq5::V_S>();
for(Zq2 f=0;f<8u;++f)for(Zq2 row=0;row<16u;++row)residualOffsets.SetValue(f*16u+row,(row*256u+f)*4u);
ZqB<Zq5::S_V>();PrepareResidualX(0u,0u);auto work=ResidualWork();
ResetMask();SetMaskNorm();
for(Zq2 chunk=0;chunk<256u;chunk+=64u){
Adds(work[q+chunk],vr_[256u+chunk],0.0f,Zq3(64u),Zq4(16u),UnaryRepeatParams{1,1,32,0});
Adds(work[3u*q+chunk],vr_[768u+chunk],0.0f,Zq3(64u),Zq4(16u),UnaryRepeatParams{1,1,32,0});
Adds(work[5u*q+chunk],vr_[1280u+chunk],0.0f,Zq3(64u),Zq4(16u),UnaryRepeatParams{1,1,32,0});
}
Duplicate(ProducerResidualInput16(),0.0f,p);Zq1<PIPE_V>();ProducerPrefetchResidual16(0u);
if(qrLogicalN_==200u){Duplicate(output_[456u],0.0f,56u);Zq1<PIPE_V>();}
for(Zq2 row=0;row<qrLogicalN_;row+=16u){WaitFlag<Zq5::MTE2_V>(0);ProducerResidualRowMajor16(0u,row,0u);}

}
}

QrCubeApply(shared,active,dual200_&&(qrLogicalN_==200u||(ColumnFactored256&&qrLogicalN_==256u)),step+1u==steps);
if(active){
if constexpr(ColumnFactored256){
if(!(dual200_&&qrLogicalN_==256u&&step+1u==steps)){Add(output_,output_,solveY_,256u);Zq1<PIPE_V>();}
}else if constexpr(CompactConsumer200){
if(!(dual200_&&qrLogicalN_==200u&&step+1u==steps)){Add(output_,output_,solveY_,256u);Zq1<PIPE_V>();}
}else{Add(output_,output_,solveY_,256u);Zq1<PIPE_V>();}
}
}
if constexpr(ColumnFactored256){if(dual200_&&qrLogicalN_==256u){EmitQrFinalFrames256(batch,active);return;}}
if constexpr(CompactConsumer200){if(dual200_&&qrLogicalN_==200u){EmitQrFinalFrames200(batch,active);return;}}
if(active){Adds(solveY_,output_,0.0f,256u);ZqB<Zq5::V_S>();}
if constexpr(EnableBoundedH3){if(qrLogicalN_==256u&&qrRhsWidth_==3u)EmitQrCoalesced256(batch,active);else if(active)EmitQrVectorDma(batch);}
else if(qrLogicalN_==200u&&qrRhsWidth_==4u&&dual200_)EmitQrCoalesced200(batch,active);
else if(active)EmitQrVectorDma(batch);
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline Zq2 KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::PivotIndex(Zq7<float>column,Zq2 first)
{
return TriangularSolveOperations::SelectColumnPivot(column,vr_,first,N(),VectorPitch());
}
