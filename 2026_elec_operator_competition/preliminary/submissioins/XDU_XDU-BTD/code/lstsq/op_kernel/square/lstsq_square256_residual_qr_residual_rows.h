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
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::QrResidualRows(Zq2 begin,Zq2 end)
{
if(compactQr32_){QrResidualRows32(begin,end);return;}
qrResidualEnd_=end;
auto residualOffsets=ProducerResidualOffsets16();ZqB<Zq5::V_S>();
for(Zq2 f=0u;f<8u;++f)for(Zq2 row=0u;row<16u;++row)residualOffsets.SetValue(f*16u+row,(row*256u+f)*4u);
ZqB<Zq5::S_V>();PrepareResidualX(0u,0u);auto work=ResidualWork();
constexpr Zq2 q=4128u,p=4096u;ResetMask();SetMaskNorm();
for(Zq2 chunk=0u;chunk<256u;chunk+=64u){
Adds(work[q+chunk],vr_[256u+chunk],0.0f,Zq3(64u),Zq4(16u),UnaryRepeatParams{1,1,32,0});
Adds(work[3u*q+chunk],vr_[768u+chunk],0.0f,Zq3(64u),Zq4(16u),UnaryRepeatParams{1,1,32,0});
Adds(work[5u*q+chunk],vr_[1280u+chunk],0.0f,Zq3(64u),Zq4(16u),UnaryRepeatParams{1,1,32,0});
}
Duplicate(ProducerResidualInput16(),0.0f,p);Zq1<PIPE_V>();ProducerPrefetchResidual16(begin);
for(Zq2 row=begin;row<end;row+=16u){WaitFlag<Zq5::MTE2_V>(0);ProducerResidualRowMajor16(0u,row,0u);}
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::QrCoopResidual(Zq3 shared,bool active)
{
constexpr Zq2 split=192u,tail=64u;
GlobalTensor<float>xs;xs.SetGlobalBuffer(w_+shared+353792u,768u);
if(active){ZqB<Zq5::V_MTE3>();DataCopy(xs[currentRhs_*256u],output_,256u);ZqB<Zq5::MTE3_S>();}

Zq1<PIPE_ALL>();SyncAll<true>();
if(active){qrResidualBOffset_=512u;QrResidualRows(0u,split);}
else{
for(Zq2 rhs=0u;rhs<3u;++rhs){
ZqB<Zq5::V_MTE2>();DataCopy(output_,xs[rhs*256u],256u);ZqB<Zq5::MTE2_V>();
qrResidualBOffset_=768u+rhs*256u;QrResidualRows(split,256u);
GlobalTensor<float>tailDst;tailDst.SetGlobalBuffer(w_+shared+354560u+rhs*64u,tail);
ZqB<Zq5::V_MTE3>();DataCopy(tailDst,output_[256u+split],tail);ZqB<Zq5::MTE3_V>();
}
}

Zq1<PIPE_ALL>();SyncAll<true>();
if(active){
GlobalTensor<float>tailSrc;tailSrc.SetGlobalBuffer(w_+shared+354560u+currentRhs_*64u,tail);
ZqB<Zq5::V_MTE2>();DataCopy(output_[256u+split],tailSrc,tail);ZqB<Zq5::MTE2_V>();
}
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::QrDualResidual256(Zq3 shared,bool owner)
{
if constexpr(EnableBoundedH3){
const Zq2 row=(GetBlockIdx()%8u)*32u;
GlobalTensor<float>xs,rs;xs.SetGlobalBuffer(w_+shared+353792u,768u);rs.SetGlobalBuffer(w_+shared+355328u,768u);
if(owner){ZqB<Zq5::V_MTE3>();DataCopy(xs[currentRhs_*256u],output_,256u);ZqB<Zq5::MTE3_S>();}
Zq1<PIPE_ALL>();SyncAll<true>();
for(Zq2 rhs=0u;rhs<3u;++rhs){
if constexpr(ColumnFactored256){

auto x=vr_[256u],xh=vr_[768u],xl=vr_[1280u];
ZqB<Zq5::V_MTE2>();DataCopy(x,xs[rhs*256u],256u);ZqB<Zq5::MTE2_V>();

Adds(x,x,0.0f,256u);Zq1<PIPE_V>();
And(xh.template Zq6<uint16_t>(),x.template Zq6<uint16_t>(),
vr_[2048u].template Zq6<uint16_t>(),512u);Zq1<PIPE_V>();
Sub(xl,x,xh,256u);Zq1<PIPE_V>();
}else{
ZqB<Zq5::V_MTE2>();DataCopy(output_,xs[rhs*256u],256u);ZqB<Zq5::MTE2_V>();
PrepareResidualX(0u,0u);
}
qrResidualBOffset_=768u+rhs*256u;ProducerResidualRowMajor32(row);
if constexpr(ColumnFactored256){
auto scaled=factors_[4096u],scatter=block_[16384u];
ZqB<Zq5::S_V>();Muls(scaled,output_[256u+row],qrInverseScale_,32u);Zq1<PIPE_V>();
Brcb(scatter,scaled,Zq4(4u),Zq9{1u,8u});Zq1<PIPE_V>();
GlobalTensor<float>input;input.SetGlobalBuffer(w_+shared+344576u+row*16u+rhs,4096u-row*16u-rhs);
ZqB<Zq5::V_MTE3>();DataCopyPad(input,scatter,Zq8{32u,4u,0u,60u,0u});ZqB<Zq5::MTE3_V>();
}else{
ZqB<Zq5::V_MTE3>();DataCopy(rs[rhs*256u+row],output_[256u+row],32u);ZqB<Zq5::MTE3_V>();
}
}


if constexpr(!ColumnFactored256){Zq1<PIPE_ALL>();SyncAll<true>();}
if constexpr(!ColumnFactored256){
if(owner){
ZqB<Zq5::V_MTE2>();DataCopy(output_,xs[currentRhs_*256u],256u);
DataCopy(output_[256u],rs[currentRhs_*256u],256u);ZqB<Zq5::MTE2_V>();
}
}
}else{
const bool participates=(GetBlockIdx()/2u)%4u<3u;
GlobalTensor<float>xs;xs.SetGlobalBuffer(w_+shared+353792u+currentRhs_*256u,256u);
if(owner){ZqB<Zq5::V_MTE3>();DataCopy(xs,output_,256u);ZqB<Zq5::MTE3_S>();}
Zq1<PIPE_ALL>();SyncAll<true>();
GlobalTensor<float>tail;tail.SetGlobalBuffer(w_+shared+354816u+currentRhs_*128u,128u);
if(participates){
if(!owner){ZqB<Zq5::V_MTE2>();DataCopy(output_,xs,256u);ZqB<Zq5::MTE2_V>();}
qrResidualBOffset_=512u;QrResidualRows32(owner?0u:128u,owner?128u:256u);
if(!owner){ZqB<Zq5::V_MTE3>();DataCopy(tail,output_[384u],128u);ZqB<Zq5::MTE3_S>();}
}
Zq1<PIPE_ALL>();SyncAll<true>();
if(owner){ZqB<Zq5::V_MTE2>();DataCopy(output_[384u],tail,128u);ZqB<Zq5::MTE2_V>();}

}
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::QrDualResidual200(Zq3 shared,bool owner)
{
if(qrLogicalN_==256u){QrDualResidual256(shared,owner);return;}
GlobalTensor<float>xs;xs.SetGlobalBuffer(w_+shared+353792u+currentRhs_*256u,256u);
if(owner){ZqB<Zq5::V_MTE3>();DataCopy(xs,output_,256u);ZqB<Zq5::MTE3_S>();}
Zq1<PIPE_ALL>();SyncAll<true>();
if(!owner){ZqB<Zq5::V_MTE2>();DataCopy(output_,xs,256u);ZqB<Zq5::MTE2_V>();}
qrResidualBOffset_=512u;
QrResidualRows32(owner?0u:104u,owner?104u:200u);


const Zq2 begin=owner?0u:104u,count=owner?104u:96u;
auto scaled=factors_[4096u+begin];
ZqB<Zq5::S_V>();Muls(scaled,output_[256u+begin],qrInverseScale_,count);
GlobalTensor<float>next;next.SetGlobalBuffer(w_+shared+354816u+currentRhs_*256u,256u);
ZqB<Zq5::V_MTE3>();DataCopy(next[begin],scaled,count);


if(owner)DataCopy(next[200u],output_[456u],56u);
ZqB<Zq5::MTE3_S>();
Zq1<PIPE_ALL>();SyncAll<true>();

}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::ProducerPrefetchResidual16(Zq2 row)
{
if(qrLogicalN_==200u){
auto input=ProducerResidualInput16();ZqB<Zq5::V_MTE2>();Duplicate(input,0.0f,4096u);ZqB<Zq5::V_MTE2>();
if(row<200u){const Zq2 count=200u-row<16u?200u-row:16u;GlobalTensor<float>original;
original.SetGlobalBuffer(a_+(Zq3(currentBatch_)*200u+row)*200u,Zq3(count)*200u);
DataCopyPad(input,original,Zq8{uint16_t(count),800u,0u,7u,0u},DataCopyPadExtParams<float>{false,0u,0u,0.0f});}
ZqB<Zq5::MTE2_S>();
const float padDiagonal=square_math::Divide(1.0f,qrInverseScale_);
for(Zq2 i=0;i<16u;++i)if(row+i>=200u)input.SetValue(i*256u+row+i,padDiagonal);
ZqB<Zq5::S_V>();SetFlag<Zq5::MTE2_V>(0);return;
}
const Zq2 n=N(),count=n-row<16u?n-row:16u;
auto input=ProducerResidualInput16();


if(count<16u){Duplicate(input[count*256u],0.0f,(16u-count)*256u);Zq1<PIPE_V>();}
ZqB<Zq5::V_MTE2>();
GlobalTensor<float>original;original.SetGlobalBuffer(a_+(Zq3(currentBatch_)*n+row)*n,Zq3(count)*n);
DataCopyPad(input,original,Zq8{uint16_t(count),n*4u,0u,(256u-n)/8u,0u},DataCopyPadExtParams<float>{false,0u,0u,0.0f});
SetFlag<Zq5::MTE2_V>(0);
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::ProducerResidualRowMajor16(Zq3 base,Zq2 row,Zq2 slot)
{
constexpr Zq2 p=4096u;
constexpr Zq2 q=4128u;
auto a=ProducerResidualInput16(),work=ResidualWork();
Zq7<float>ah=work[2u*q],al=work[4u*q],u=work;
Zq7<float>hi=work[6u*q],lo=ProducerResidualLow16(),t=work;
SetMaskCount();ZqA<uint16_t,MaskMode::COUNTER>(0u,2u*p);
And<uint16_t,false>(ah.template Zq6<uint16_t>(),a.template Zq6<uint16_t>(),
vr_[ResidualMaskPlane()*VectorPitch()].template Zq6<uint16_t>(),Zq3(0),Zq4(1),Zq0{1,1,1,8,8,0});
Zq1<PIPE_V>();ZqA<float,MaskMode::COUNTER>(0u,p);
Sub<float,false>(al,a,ah,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Mul<float,false>(hi,a,work[q],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Mul<float,false>(lo,ah,work[3u*q],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
if(row+16u<(qrLogicalN_==200u?200u:N())&&row+16u<qrResidualEnd_){ProducerPrefetchResidual16(row+16u);
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,p);}
Sub<float,false>(lo,lo,hi,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Mul<float,false>(t,ah,work[5u*q],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Mul<float,false>(t,al,work[3u*q],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Mul<float,false>(t,al,work[5u*q],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
ResetMask();SetMaskNorm();


Add<float>((ah)[0u],(hi)[0u],(hi[128u])[0u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Add<float>((lo)[0u],(lo)[0u],(lo[128u])[0u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Add<float>((ah)[64u],(hi)[64u],(hi[128u])[64u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Add<float>((lo)[64u],(lo)[64u],(lo[128u])[64u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Zq1<PIPE_V>();
Sub<float>((al)[0u],(ah)[0u],(hi)[0u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Sub<float>((al)[64u],(ah)[64u],(hi)[64u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Zq1<PIPE_V>();
Sub<float>((u)[0u],(ah)[0u],(al)[0u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Sub<float>((u)[64u],(ah)[64u],(al)[64u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Zq1<PIPE_V>();
Sub<float>((u)[0u],(hi)[0u],(u)[0u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Sub<float>((al)[0u],(hi[128u])[0u],(al)[0u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Sub<float>((u)[64u],(hi)[64u],(u)[64u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Sub<float>((al)[64u],(hi[128u])[64u],(al)[64u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Zq1<PIPE_V>();
Add<float>((u)[0u],(u)[0u],(al)[0u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Add<float>((u)[64u],(u)[64u],(al)[64u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Zq1<PIPE_V>();
Add<float>((lo)[0u],(lo)[0u],(u)[0u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Add<float>((lo)[64u],(lo)[64u],(u)[64u],Zq3(64),Zq4(16),Zq0{1,1,1,32,32,32});
Zq1<PIPE_V>();
MergeStridedResidualHalves<64u, 64, 16, 32>(hi, ah, lo, al, u);
MergeStridedResidualHalves<32u, 32, 16, 32>(ah, hi, lo, al, u);
MergeStridedResidualHalves<16u, 16, 16, 32>(hi, ah, lo, al, u);
MergeStridedResidualHalves<8u, 8, 16, 32>(ah, hi, lo, al, u);

Gather(work[6u*q],work[2u*q],ProducerResidualOffsets16(),0u,128u);
Gather(work,lo,ProducerResidualOffsets16(),0u,128u);Zq1<PIPE_V>();
hi=work[6u*q];lo=work;ah=work[2u*q];al=work[4u*q];u=ProducerResidualLow16();
SetMaskCount();
MergeCounterResidualHalves<64u>(ah, hi, lo, al, u, 64u);
MergeCounterResidualHalves<32u>(hi, ah, lo, al, u, 32u);
MergeCounterResidualHalves<16u>(ah, hi, lo, al, u, 16u);
ResetMask();SetMaskNorm();
const Zq2 count=N()-row<16u?N()-row:16u;
auto hh=work[2u*q],bb=output_[qrResidualBOffset_+row];
auto neg=ProducerResidualLow16(),ss=work[4u*q],vv=work[6u*q],tt=output_[256u+row];
WriteCompensatedResidual(tt, bb, hh, lo, neg, ss, vv, tt, count);
}
