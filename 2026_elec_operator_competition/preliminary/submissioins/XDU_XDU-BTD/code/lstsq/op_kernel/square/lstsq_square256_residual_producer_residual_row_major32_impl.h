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
template<bool Compact200,Zq2 TileRows,bool KeepInput>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::ProducerResidualRowMajor32Impl(Zq2 row)
{
static_assert(TileRows==32u||(Compact200&&(TileRows==8u||(CompactConsumer200&&TileRows==40u))),"Unsupported residual rows");
constexpr Zq2 p=TileRows*(Compact200?208u:256u);
constexpr Zq2 q=Compact200?(CompactConsumer200?8384u:6720u):8256u;
constexpr Zq4 stride=Compact200?26u:32u;
auto a=ProducerResidualInput16(),work=ResidualWork();
Zq7<float>ah=work[q],al=work[2u*q],u=a;
Zq7<float>hi=work,lo=ProducerResidualLow16(),t=((EnableBoundedH3&&!Compact200)||KeepInput)?ah:a;
SetMaskCount();ZqA<uint16_t,MaskMode::COUNTER>(0u,2u*p);
And<uint16_t,false>(ah.template Zq6<uint16_t>(),a.template Zq6<uint16_t>(),
vr_[ResidualMaskPlane()*VectorPitch()].template Zq6<uint16_t>(),Zq3(0),Zq4(1),Zq0{1,1,1,8,8,0});
Zq1<PIPE_V>();ZqA<float,MaskMode::COUNTER>(0u,p);
Sub<float,false>(al,a,ah,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
ResetMask();SetMaskNorm();
for(Zq2 chunk=0;chunk<256u;chunk+=64u)Mul<float>(hi[chunk],a[chunk],vr_[256u+chunk],Zq3(Compact200&&chunk==192u?16u:64u),Zq4(TileRows),Zq0{1,1,1,stride,stride,0});

if constexpr(KeepInput&&Compact200){
if(cachedNextRow_<cachedEnd_)PrefetchCompact200(cachedNextRow_,cachedEnd_,1u);
}
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,p);
ResetMask();SetMaskNorm();
for(Zq2 chunk=0;chunk<256u;chunk+=64u)Mul<float>(lo[chunk],ah[chunk],vr_[768u+chunk],Zq3(Compact200&&chunk==192u?16u:64u),Zq4(TileRows),Zq0{1,1,1,stride,stride,0});
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,p);Zq1<PIPE_V>();

Sub<float,false>(lo,lo,hi,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
ResetMask();SetMaskNorm();
for(Zq2 chunk=0;chunk<256u;chunk+=64u)Mul<float>(t[chunk],ah[chunk],vr_[1280u+chunk],Zq3(Compact200&&chunk==192u?16u:64u),Zq4(TileRows),Zq0{1,1,1,stride,stride,0});
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,p);Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
ResetMask();SetMaskNorm();
for(Zq2 chunk=0;chunk<256u;chunk+=64u)Mul<float>(t[chunk],al[chunk],vr_[768u+chunk],Zq3(Compact200&&chunk==192u?16u:64u),Zq4(TileRows),Zq0{1,1,1,stride,stride,0});
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,p);Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
ResetMask();SetMaskNorm();
for(Zq2 chunk=0;chunk<256u;chunk+=64u)Mul<float>(t[chunk],al[chunk],vr_[1280u+chunk],Zq3(Compact200&&chunk==192u?16u:64u),Zq4(TileRows),Zq0{1,1,1,stride,stride,0});
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,p);Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
ResetMask();SetMaskNorm();


if constexpr((EnableBoundedH3&&!Compact200)||KeepInput)u=ah[128u];
MergeStridedResidualFirstHalves<TileRows, stride, Compact200 ? 16 : 64>(hi, lo, ah, al, u);
if constexpr(Compact200){


Adds<float>(ah[80u],hi[80u],0.0f,Zq3(48u),Zq4(TileRows),UnaryRepeatParams{1,1,26,26});
Adds<float>(lo[80u],lo[80u],0.0f,Zq3(48u),Zq4(TileRows),UnaryRepeatParams{1,1,26,26});
Zq1<PIPE_V>();
}
if constexpr((EnableBoundedH3&&!Compact200)||KeepInput)u=hi[64u];
MergeStridedResidualHalves<64, 64, TileRows, stride>(hi, ah, lo, al, u);
if constexpr((EnableBoundedH3&&!Compact200)||KeepInput)u=ah[32u];
MergeStridedResidualHalves<32, 32, TileRows, stride>(ah, hi, lo, al, u);
if constexpr((EnableBoundedH3&&!Compact200)||KeepInput)u=hi[16u];
MergeStridedResidualHalves<16, 16, TileRows, stride>(hi, ah, lo, al, u);
if constexpr((EnableBoundedH3&&!Compact200)||KeepInput)u=ah[8u];
MergeStridedResidualHalves<8, 8, TileRows, stride>(ah, hi, lo, al, u);

if constexpr (ColumnFactored256 && !Compact200) {
    lo=PackResidualPlanes<q,16384u,8256u>(work,lo);
} else if constexpr (CompactConsumer200 && Compact200 && TileRows != 40u) {
    lo=PackResidualPlanes<q,2u*q,0u>(work,lo);
} else {
    lo=PackResidualRows<Compact200,TileRows,KeepInput>(work,a,lo);
}
ResetMask();SetMaskNorm();
const Zq2 count=N()-row<TileRows?N()-row:TileRows;
auto hh=work[q],bb=output_[qrResidualBOffset_+row];
auto neg=ProducerResidualLow16(),ss=work[2u*q],vv=work,tt=output_[256u+row];
WriteCompensatedResidual(tt, bb, hh, lo, neg, ss, vv, tt, count);
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
template<Zq2 q,Zq2 scratch,Zq2 firstHigh>
__aicore__ inline Zq7<float> KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::PackResidualPlanes(Zq7<float> work, Zq7<float> lo)
{
Gather(work[scratch],work[q],ProducerResidualOffsets16(),0u,Zq3(64u),Zq4(4u),uint16_t(16u));
Gather(work[scratch+64u],lo,ProducerResidualOffsets16(),0u,Zq3(64u),Zq4(4u),uint16_t(16u));Zq1<PIPE_V>();
ResetMask();SetMaskNorm();
Add<float>(work[firstHigh],work[scratch],work[scratch+256u],Zq3(64u),Zq4(4u),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(factors_[256u],work[firstHigh],work[scratch],Zq3(64u),Zq4(2u),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float,false>(factors_,work[firstHigh],factors_[256u],Zq3(64u),Zq4(2u),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float,false>(factors_,work[scratch],factors_,Zq3(64u),Zq4(4u),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float,false>(factors_,factors_,factors_[256u],Zq3(64u),Zq4(2u),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float,false>(work[firstHigh+64u],work[firstHigh+64u],factors_,Zq3(64u),Zq4(2u),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(work[scratch],work[firstHigh],work[firstHigh+128u],Zq3(64u),Zq4(2u),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(factors_[128u],work[scratch],work[firstHigh],Zq3(64u),Zq4(1u),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float,false>(factors_,work[scratch],factors_[128u],Zq3(64u),Zq4(1u),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Sub<float,false>(factors_,work[firstHigh],factors_,Zq3(64u),Zq4(2u),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float,false>(factors_,factors_,factors_[128u],Zq3(64u),Zq4(1u),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float,false>(work[scratch+64u],work[scratch+64u],factors_,Zq3(64u),Zq4(1u),Zq0{1,1,1,16,16,16});Zq1<PIPE_V>();
Add<float>(work[q],work[scratch],work[scratch+32u],Zq3(32u),Zq4(2u),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(factors_[32u],work[q],work[scratch],Zq3(32u),Zq4(1u),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float,false>(factors_,work[q],factors_[32u],Zq3(32u),Zq4(1u),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Sub<float>(factors_,work[scratch],factors_,Zq3(64u),Zq4(1u),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float>(factors_,factors_,factors_[32u],Zq3(32u),Zq4(1u),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(work[q+64u],work[q+64u],factors_,Zq3(32u),Zq4(1u),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
return work[q+64u];
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
template<bool Compact200,Zq2 TileRows,bool KeepInput>
__aicore__ inline Zq7<float> KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::PackResidualRows(Zq7<float> work, Zq7<float> a, Zq7<float> lo)
{
constexpr Zq2 q=Compact200?(CompactConsumer200?8384u:6720u):8256u;
constexpr Zq2 packedRows=TileRows;
auto gatherOffsets=ProducerResidualOffsets16();
if constexpr(CompactConsumer200&&Compact200&&TileRows==40u)gatherOffsets=panel_[1024u].template Zq6<Zq2>();
Gather(work,work[q],gatherOffsets,0u,8u*packedRows);
auto packedLow=(CompactConsumer200&&Compact200&&TileRows==40u)?a:(((EnableBoundedH3&&!Compact200)||KeepInput)?panel_[256u]:a);
Gather(packedLow,lo,gatherOffsets,0u,8u*packedRows);Zq1<PIPE_V>();
auto hi=work,ah=work[q],al=work[2u*q],u=ProducerResidualLow16();
lo=packedLow;
SetMaskCount();
MergeCounterResidualHalves<4u*packedRows>(ah, hi, lo, al, u, 4u*packedRows);
MergeCounterResidualHalves<2u*packedRows>(hi, ah, lo, al, u, 2u*packedRows);
MergeCounterResidualHalves<packedRows>(ah, hi, lo, al, u, packedRows);
ResetMask();SetMaskNorm();
return lo;
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::QrResidualRowsDouble200(Zq2 begin,Zq2 end)
{
PrepareResidualX(0u,0u);Duplicate(output_[456u],0.0f,56u);Zq1<PIPE_V>();
cachedNextRow_=200u;cachedEnd_=200u;compactInputSlot_=0u;
for(Zq2 row=begin;row<end;){
const Zq2 count=end-row==40u?40u:32u;
PrefetchCompact200(row,row+count,0u);WaitFlag<Zq5::MTE2_V>(0);
if(count==40u)ProducerResidualRowMajor32Impl<true,40u,true>(row);
else ProducerResidualRowMajor32Impl<true,32u,true>(row);
row+=count;
}
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::QrResidualRows32(Zq2 begin,Zq2 end)
{
if constexpr(CompactConsumer200){QrResidualRowsDouble200(begin,end);return;}
PrepareResidualX(0u,0u);
if(qrLogicalN_==200u){Duplicate(output_[456u],0.0f,56u);Zq1<PIPE_V>();}
for(Zq2 row=begin;row<end;row+=32u){
auto input=ProducerResidualInput16();ZqB<Zq5::V_MTE2>();
GlobalTensor<float>original;
if(qrLogicalN_==256u){
original.SetGlobalBuffer(a_+(Zq3(currentBatch_)*256u+row)*256u,8192u);DataCopy(input,original,8192u);
}else{
Duplicate(input,0.0f,6656u);ZqB<Zq5::V_MTE2>();
const Zq2 count=end-row<32u?end-row:32u;
original.SetGlobalBuffer(a_+(Zq3(currentBatch_)*200u+row)*200u,Zq3(count)*200u);
DataCopyPad(input,original,Zq8{uint16_t(count),800u,0u,1u,0u},DataCopyPadExtParams<float>{false,0u,0u,0.0f});

}
ZqB<Zq5::MTE2_V>();
if(qrLogicalN_==200u&&end-row==8u)ProducerResidualRowMajor32Impl<true,8u>(row);
else ProducerResidualRowMajor32(row);
}
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::ProducerRefine16(Zq3 base)
{
constexpr Zq2 q=4128u,p=4096u,pitch=256u;

auto offsets=ProducerResidualOffsets16();
for(Zq2 f=0;f<8u;++f)for(Zq2 row=0;row<16u;++row)
offsets.SetValue(f*16u+row,(row*256u+f)*4u);
ZqB<Zq5::S_V>();
PrepareChainCaches(base);
for(Zq2 step=0;step<2u;++step){
ResidualOperations::RunResidual16Pass(*this,base);

pathLocalRhs_=true;SolveInPlaceWithFactors(base,w_+OffD(base),1u);
ZqB<Zq5::S_V>();Add(output_,output_,output_[256u],256u);Zq1<PIPE_V>();
}
Adds(solveY_,output_,0.0f,256u);ZqB<Zq5::V_S>();
}

template<Zq2 StaticN,Zq2 StaticK,bool EnableBoundedH3,bool CompactConsumer200,bool ColumnFactored256>
__aicore__ inline void KernelLargePipe256<StaticN, StaticK, EnableBoundedH3, CompactConsumer200, ColumnFactored256>::PrepareResidualX(Zq3 base,Zq2 col)
{
    ResidualOperations::PrepareResidualXFromOutput(*this, base, col);
}
