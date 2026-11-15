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

// Shared by the shape-specific kernels. Each kernel owns its workspace and synchronization state.
struct ResidualOperations {
template<class Kernel>
__aicore__ static inline void PrepareResidualX(Kernel &kernel, Zq3 base,Zq2 col)
{
const Zq2 p=kernel.VectorPitch();
Zq7<float>x=kernel.vr_[p],xh=kernel.vr_[3u*p],xl=kernel.vr_[5u*p];
ZqB<Zq5::V_S>();
if(kernel.tinyPrivateFlow_){Adds(x,kernel.tinyNegative_[272u],0.0f,16u);Zq1<PIPE_V>();}
else for(Zq2 i=0;i<kernel.N();++i)x.SetValue(i,kernel.GetX(base,i,col));
ZqB<Zq5::S_V>();
And(xh.template Zq6<uint16_t>(),x.template Zq6<uint16_t>(),
kernel.vr_[11u*p].template Zq6<uint16_t>(),2u*p);
Zq1<PIPE_V>();
Sub(xl,x,xh,p);
Zq1<PIPE_V>();
}

template<class Kernel>
__aicore__ static inline void TinyPrefetch(Kernel &kernel, Zq2 baseRow,Zq2 slot)
{
constexpr Zq2 p=256u,pitch=16u;
const Zq2 n=kernel.N(),rows=n-baseRow<16u?n-baseRow:16u;
auto input=kernel.panel_[14u*p+slot*p];ZqB<Zq5::V_MTE2>(slot);
if(rows<16u){Duplicate(input,0.0f,p);ZqB<Zq5::V_MTE2>(slot);}
GlobalTensor<float>original;original.SetGlobalBuffer(kernel.a_+(Zq3(kernel.currentBatch_)*n+baseRow)*n,rows*n);
DataCopyPad(input,original,Zq8{uint16_t(rows),n*4u,0u,(pitch-n)/8u,0u},
DataCopyPadExtParams<float>{true,0u,Zq4((8u-n%8u)%8u),0.0f});
SetFlag<Zq5::MTE2_V>(slot);
}

template<class Kernel>
__aicore__ static inline void PrepareResidualX64(Kernel& kernel, Zq3 base, Zq2 col)
{
const Zq2 p=kernel.VectorPitch();
Zq7<float>x=kernel.vr_[p],xh=kernel.vr_[3u*p],xl=kernel.vr_[5u*p];
ZqB<Zq5::V_S>();
{if(kernel.producerPrivate_){Adds(x,kernel.negative64_[4096u],0.0f,64u);Zq1<PIPE_V>();}
else for(Zq2 i=0;i<kernel.N();++i)x.SetValue(i,kernel.GetX(base,i,col));}
ZqB<Zq5::S_V>();
And(xh.template Zq6<uint16_t>(),x.template Zq6<uint16_t>(),
kernel.vr_[kernel.ResidualMaskPlane()*p].template Zq6<uint16_t>(),2u*p);
Zq1<PIPE_V>();
Sub(xl,x,xh,p);
Zq1<PIPE_V>();
}

template<class Kernel>
__aicore__ static inline void PrepareResidualXFromOutput(Kernel& kernel, Zq3 base, Zq2 col)
{
const Zq2 p=kernel.VectorPitch();
Zq7<float>x=kernel.vr_[p],xh=kernel.vr_[3u*p],xl=kernel.vr_[5u*p];
ZqB<Zq5::V_S>();
{
ZqB<Zq5::S_V>();Adds(x,kernel.output_,0.0f,p);ZqB<Zq5::V_S>();
}
ZqB<Zq5::S_V>();
And(xh.template Zq6<uint16_t>(),x.template Zq6<uint16_t>(),
kernel.vr_[kernel.ResidualMaskPlane()*p].template Zq6<uint16_t>(),2u*p);
Zq1<PIPE_V>();
Sub(xl,x,xh,p);
Zq1<PIPE_V>();
}

template<class Kernel>
__aicore__ static inline void ComputeWideSmallResidual(Kernel& kernel, Zq3 base, Zq2 row,
    Zq7<float> work, Zq7<float> a, Zq7<float> ah, Zq7<float> al)
{
constexpr Zq2 p=2048u,active=p,groupWidth=1u;
auto extra=work[10u*p],u=work[9u*p];
ResidualOperations::SplitTileInput(a,ah,al,kernel.vr_[kernel.ResidualMaskPlane()*kernel.VectorPitch()].template Zq6<uint16_t>(),active);
ResidualOperations::TileRhsProducts(work,extra,a,ah,al,p,groupWidth);
for(Zq2 rhs=0;rhs<groupWidth;++rhs){
Zq7<float>hi=rhs?work[8u*p]:work[6u*p];
Zq7<float>lo=rhs?extra[3u*p]:work[7u*p];
MergeCounterResidualHalves<1024u>(ah, hi, lo, al, u, 1024u);
MergeCounterResidualHalves<512u>(hi, ah, lo, al, u, 512u);
MergeCounterResidualHalves<256u>(ah, hi, lo, al, u, 256u);
MergeCounterResidualHalves<128u>(hi, ah, lo, al, u, 128u);
MergeCounterResidualHalves<64u>(ah, hi, lo, al, u, 64u);
MergeCounterResidualHalves<32u>(hi, ah, lo, al, u, 32u);
}
ResetMask();SetMaskNorm();
if (kernel.producerPrivate_) {
    FinishPrivateWideResidual(kernel, row, work);
    return;
}
FinishWideResidual(kernel, base, row, work);
}

template<class Kernel>
__aicore__ static inline void FinishPrivateWideResidual(Kernel& kernel, Zq2 row, Zq7<float> work)
{
constexpr Zq2 p=2048u;
const Zq2 count=kernel.N()-row<32u?kernel.N()-row:32u;
auto hh=work[6u*p],ll=work[7u*p];
auto bb=kernel.negative64_[4224u+row];
auto neg=work,ss=work[2u*p],vv=work[4u*p],tt=work[9u*p];
WriteCompensatedResidual(kernel.solveY_[row], bb, hh, ll, neg, ss, vv, tt, count);
}

template<class Kernel>
__aicore__ static inline void FinishWideResidual(Kernel& kernel, Zq3 base, Zq2 row, Zq7<float> work)
{
constexpr Zq2 p=2048u;
auto extra=work[10u*p];
ZqB<Zq5::V_S>();
const Zq2 count=kernel.N()-row<32u?kernel.N()-row:32u;
for(Zq2 rhs=0;rhs<1u;++rhs){
auto hi=rhs?work[8u*p]:work[6u*p];auto lo=rhs?extra[3u*p]:work[7u*p];
for(Zq2 lane=0;lane<count;++lane){
Pair d=square_math::Sum(kernel.w_[kernel.OffB(base)+Zq3(row+lane)+rhs],-hi.GetValue(lane));
kernel.w_[kernel.OffD(base)+Zq3(row+lane)+rhs]=square_math::RoundAdd(d.hi,square_math::RoundSub(d.lo,lo.GetValue(lane)));
}
}
}


template<class Kernel>
__aicore__ static inline void RowMajorResidual50Products(Kernel& kernel, Zq7<float> a,
    Zq7<float> ah, Zq7<float> al, Zq7<float> hi, Zq7<float> lo, Zq7<float> t)
{
SetMaskCount();ZqA<uint16_t,MaskMode::COUNTER>(0u,2u*3200u);
And<uint16_t,false>(ah.template Zq6<uint16_t>(),a.template Zq6<uint16_t>(),
kernel.vr_[kernel.ResidualMaskPlane()*64u].template Zq6<uint16_t>(),Zq3(0),Zq4(1),Zq0{1,1,1,8,8,0});
Zq1<PIPE_V>();ZqA<float,MaskMode::COUNTER>(0u,3200u);
Sub<float,false>(al,a,ah,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
ResetMask();SetMaskNorm();
}


template<class Kernel>
__aicore__ static inline void RowMajorResidual64Products(Kernel& kernel, bool cachedHigh,
    Zq7<float> a, Zq7<float> splitHigh, Zq7<float> splitLow, Zq7<float> hi, Zq7<float> lo, Zq7<float> t)
{
SetMaskCount();
if(!kernel.augmented64Ready_||!cachedHigh){
ZqA<uint16_t,MaskMode::COUNTER>(0u,2u*4096u);
And<uint16_t,false>(splitHigh.template Zq6<uint16_t>(),a.template Zq6<uint16_t>(),
kernel.vr_[kernel.ResidualMaskPlane()*64u].template Zq6<uint16_t>(),Zq3(0),Zq4(1),Zq0{1,1,1,8,8,0});
Zq1<PIPE_V>();}
if(!kernel.rowInverse64Ready_||!cachedHigh){
ZqA<float,MaskMode::COUNTER>(0u,4096u);
Sub<float,false>(splitLow,a,splitHigh,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
}
ResetMask();SetMaskNorm();
}


__aicore__ static inline void ScalarResidualProducts(Zq7<float> vr, Zq2 p, Zq2 slot, Zq2 maskPlane, Zq2 zeroPlane)
{
Zq7<float>a=slot?vr[zeroPlane*p]:vr,x=vr[p],ah=vr[2u*p],xh=vr[3u*p];
Zq7<float>al=vr[4u*p],xl=vr[5u*p],hi=vr[6u*p],lo=vr[7u*p];
Zq7<float>t=vr[8u*p];

SetMaskCount();
ZqA<uint16_t,MaskMode::COUNTER>(0u,2u*p);
And<uint16_t,false>(ah.Zq6<uint16_t>(),a.Zq6<uint16_t>(),
vr[maskPlane*p].Zq6<uint16_t>(),Zq3(0),Zq4(1),
Zq0{1,1,1,8,8,8});
ZqA<float,MaskMode::COUNTER>(0u,p);
Zq1<PIPE_V>();
Sub<float,false>(al,a,ah,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Mul<float,false>(hi,a,x,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Mul<float,false>(lo,ah,xh,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Sub<float,false>(lo,lo,hi,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Mul<float,false>(t,ah,xl,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Mul<float,false>(t,al,xh,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Mul<float,false>(t,al,xl,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();


}

__aicore__ static inline float FinishScalarResidual(Zq7<float> vr, Zq2 p, const __gm__ float* rhs)
{
Zq7<float>ah=vr[2u*p],al=vr[4u*p],hi=vr[6u*p],lo=vr[7u*p],u=vr[9u*p];
for (Zq2 width = p/2u; width >= 8u; width /= 2u) {
    MergeLegacyResidualHalves(ah, hi, lo, al, u, width, width);
    Adds<float,false>(hi,ah,0.0f,Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});
    Zq1<PIPE_V>();
}
ResetMask();SetMaskNorm();
ZqB<Zq5::V_S>();
float sh=0.0f,sl=0.0f;
for(Zq2 i=0;i<8u;++i){
Pair next=square_math::Sum(sh,hi.GetValue(i));
sl=square_math::RoundAdd(sl,square_math::RoundAdd(next.lo,lo.GetValue(i)));sh=next.hi;
}
Pair diff=square_math::Sum(*rhs,-sh);
return square_math::RoundAdd(diff.hi,square_math::RoundSub(diff.lo,sl));
}

__aicore__ static inline float ComputeScalarResidual(Zq7<float> vr, Zq2 p, Zq2 slot,
    Zq2 maskPlane, Zq2 zeroPlane, const __gm__ float* rhs)
{
    ScalarResidualProducts(vr, p, slot, maskPlane, zeroPlane);
    return FinishScalarResidual(vr, p, rhs);
}

__aicore__ static inline void TileRhsProducts(Zq7<float> work, Zq7<float> extra,
    Zq7<float> a, Zq7<float> ah, Zq7<float> al, Zq2 p, Zq2 groupWidth)
{
for(Zq2 rhs=0;rhs<groupWidth;++rhs){
Zq7<float>x=rhs?extra:work[p];
Zq7<float>xh=rhs?extra[p]:work[3u*p];
Zq7<float>xl=rhs?extra[2u*p]:work[5u*p];
Zq7<float>hi=rhs?work[8u*p]:work[6u*p];
Zq7<float>lo=rhs?extra[3u*p]:work[7u*p];
Zq7<float>t=rhs?work[9u*p]:work[8u*p];

CompensatedTileProduct(hi, lo, t, a, ah, al, x, xh, xl);
}
}


__aicore__ static inline void ReduceSmallTile(Zq7<float> work, Zq7<float> extra,
    Zq7<float> ah, Zq7<float> al, Zq7<float> u, Zq2 p, Zq2 groupWidth)
{
for(Zq2 rhs=0;rhs<groupWidth;++rhs){
Zq7<float>hi=rhs?work[8u*p]:work[6u*p];
Zq7<float>lo=rhs?extra[3u*p]:work[7u*p];
MergeCounterResidualHalves<256u>(ah, hi, lo, al, u, 256u);
MergeCounterResidualHalves<128u>(hi, ah, lo, al, u, 128u);
MergeCounterResidualHalves<64u>(ah, hi, lo, al, u, 64u);
MergeCounterResidualHalves<32u>(hi, ah, lo, al, u, 32u);
MergeCounterResidualHalves<16u>(ah, hi, lo, al, u, 16u);
MergeCounterResidualHalves<8u>(hi, ah, lo, al, u, 8u);
}
}


__aicore__ static inline void SplitTileInput(Zq7<float> a, Zq7<float> ah,
    Zq7<float> al, Zq7<uint16_t> mask, Zq2 active)
{
SetMaskCount();ZqA<uint16_t,MaskMode::COUNTER>(0u,2u*active);
And<uint16_t,false>(ah.template Zq6<uint16_t>(),a.template Zq6<uint16_t>(),
mask,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,0});
Zq1<PIPE_V>();ZqA<float,MaskMode::COUNTER>(0u,active);
Sub<float,false>(al,a,ah,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
}


template<Zq2 Rows>
__aicore__ static inline void RowMajorProducts(Zq7<float> a, Zq7<float> splitHigh,
    Zq7<float> splitLow, Zq7<float> hi, Zq7<float> lo, Zq7<float> t, Zq7<float> vr)
{
Mul<float>(hi,a,vr[64u],Zq3(64),Zq4(Rows),Zq0{1,1,1,8,8,0});
Mul<float>(lo,splitHigh,vr[192u],Zq3(64),Zq4(Rows),Zq0{1,1,1,8,8,0});Zq1<PIPE_V>();
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,Rows*64u);
Sub<float,false>(lo,lo,hi,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
ResetMask();SetMaskNorm();
Mul<float>(t,splitHigh,vr[320u],Zq3(64),Zq4(Rows),Zq0{1,1,1,8,8,0});Zq1<PIPE_V>();
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,Rows*64u);
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
ResetMask();SetMaskNorm();
Mul<float>(t,splitLow,vr[192u],Zq3(64),Zq4(Rows),Zq0{1,1,1,8,8,0});Zq1<PIPE_V>();
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,Rows*64u);
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
ResetMask();SetMaskNorm();
Mul<float>(t,splitLow,vr[320u],Zq3(64),Zq4(Rows),Zq0{1,1,1,8,8,0});Zq1<PIPE_V>();
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,Rows*64u);
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
ResetMask();SetMaskNorm();
}


template<class Kernel>
__aicore__ static inline void PrepareWideResidualColumns(Kernel& kernel,Zq3 base,Zq2 p,Zq2 pitch,Zq2 width)
{
for(Zq2 rhs=0;rhs<width;++rhs){
kernel.PrepareResidualX(base,rhs);
auto x=rhs?kernel.panel_[10u*p]:kernel.panel_[p];
auto xh=rhs?kernel.panel_[kernel.ResidualMaskPlane()*p]:kernel.panel_[3u*p];
auto xl=rhs?kernel.panel_[kernel.ResidualZeroPlane()*p]:kernel.panel_[5u*p];
Brcb(x,kernel.vr_[pitch],Zq4(8),Zq9{4,32});
Brcb(x[8u],kernel.vr_[pitch],Zq4(8),Zq9{4,32});
Brcb(x[16u],kernel.vr_[pitch],Zq4(8),Zq9{4,32});
Brcb(x[24u],kernel.vr_[pitch],Zq4(8),Zq9{4,32});
Brcb(xh,kernel.vr_[3u*pitch],Zq4(8),Zq9{4,32});
Brcb(xh[8u],kernel.vr_[3u*pitch],Zq4(8),Zq9{4,32});
Brcb(xh[16u],kernel.vr_[3u*pitch],Zq4(8),Zq9{4,32});
Brcb(xh[24u],kernel.vr_[3u*pitch],Zq4(8),Zq9{4,32});
Brcb(xl,kernel.vr_[5u*pitch],Zq4(8),Zq9{4,32});
Brcb(xl[8u],kernel.vr_[5u*pitch],Zq4(8),Zq9{4,32});
Brcb(xl[16u],kernel.vr_[5u*pitch],Zq4(8),Zq9{4,32});
Brcb(xl[24u],kernel.vr_[5u*pitch],Zq4(8),Zq9{4,32});Zq1<PIPE_V>();
}
}


template<class Kernel>
__aicore__ static inline void PrepareSmallResidualColumns(Kernel& kernel,Zq3 base,Zq2 p,Zq2 pitch,Zq2 width)
{
for(Zq2 rhs=0;rhs<width;++rhs){
kernel.PrepareResidualX(base,rhs);
auto x=rhs?kernel.panel_[10u*p]:kernel.panel_[p];
auto xh=rhs?kernel.panel_[kernel.ResidualMaskPlane()*p]:kernel.panel_[3u*p];
auto xl=rhs?kernel.panel_[kernel.ResidualZeroPlane()*p]:kernel.panel_[5u*p];
Brcb(x,kernel.vr_[pitch],Zq4(8),Zq9{1,8});
Brcb(xh,kernel.vr_[3u*pitch],Zq4(8),Zq9{1,8});
Brcb(xl,kernel.vr_[5u*pitch],Zq4(8),Zq9{1,8});Zq1<PIPE_V>();
}
}


__aicore__ static inline void PrepareWideResidualOffsets(Zq7<Zq2> offsets,Zq7<float> rows)
{
auto features=rows.template Zq6<int32_t>();
auto rowBytes=rows[64u].template Zq6<int32_t>();
for(Zq2 f=0;f<64u;++f)features.SetValue(f,int32_t(f*4u));
for(Zq2 r=0;r<32u;++r)rowBytes.SetValue(r,int32_t(r*256u));
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
Brcb(offsets[0u].template Zq6<int32_t>(),features,Zq4(8),Zq9{4,32});
Brcb(offsets[8u].template Zq6<int32_t>(),features,Zq4(8),Zq9{4,32});
Brcb(offsets[16u].template Zq6<int32_t>(),features,Zq4(8),Zq9{4,32});
Brcb(offsets[24u].template Zq6<int32_t>(),features,Zq4(8),Zq9{4,32});
Zq1<PIPE_V>();
Add<int32_t>(offsets.template Zq6<int32_t>(),offsets.template Zq6<int32_t>(),rowBytes,
Zq3(32),Zq4(64),Zq0{1,1,1,4,4,0});Zq1<PIPE_V>();
}


__aicore__ static inline void PrepareColdResidualOffsets(Zq7<int32_t> offsets,Zq7<float> rows)
{
auto features=rows.template Zq6<int32_t>();
auto rowBytes=rows[64u].template Zq6<int32_t>();
ZqB<Zq5::V_S>();
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
CreateVecIndex<int32_t>(features,int32_t(0),Zq2(8));
CreateVecIndex<int32_t>(rowBytes,int32_t(0),Zq2(64));Zq1<PIPE_V>();
Muls<int32_t>(features,features,int32_t(4),Zq2(8));
Muls<int32_t>(rowBytes,rowBytes,int32_t(256),Zq2(64));Zq1<PIPE_V>();
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
for(Zq2 chunk=0;chunk<64u;chunk+=8u)
Brcb(offsets[chunk],features,Zq4(1),Zq9{8,64});
Zq1<PIPE_V>();
Add<int32_t>(offsets,offsets,rowBytes,Zq3(64),Zq4(8),Zq0{1,1,1,8,8,0});Zq1<PIPE_V>();
}


template<class Kernel>
__aicore__ static inline void RunResidual16Pass(Kernel& kernel,Zq3 base)
{
constexpr Zq2 q=4128u,p=4096u,pitch=256u;
kernel.PrepareResidualX(base,0u);auto work=kernel.ResidualWork();
ResetMask();SetMaskNorm();
for(Zq2 chunk=0;chunk<256u;chunk+=64u){
Adds(work[q+chunk],kernel.vr_[pitch+chunk],0.0f,Zq3(64),Zq4(16),UnaryRepeatParams{1,1,32,0});
Adds(work[3u*q+chunk],kernel.vr_[3u*pitch+chunk],0.0f,Zq3(64),Zq4(16),UnaryRepeatParams{1,1,32,0});
Adds(work[5u*q+chunk],kernel.vr_[5u*pitch+chunk],0.0f,Zq3(64),Zq4(16),UnaryRepeatParams{1,1,32,0});
}
Duplicate(kernel.ProducerResidualInput16(),0.0f,p);Zq1<PIPE_V>();
kernel.ProducerPrefetchResidual16(0u);
for(Zq2 row=0;row<kernel.N();row+=16u){
WaitFlag<Zq5::MTE2_V>(0);
kernel.ProducerResidualRowMajor16(base,row,0u);
}
}


__aicore__ static inline void WriteScalarResidual64(const __gm__ float* rhs,
    __gm__ float* correction,Zq7<float> hi,Zq7<float> lo)
{
ZqB<Zq5::V_S>();
for(Zq2 row=0u;row<64u;++row){
Pair d=square_math::Sum(rhs[row],-hi.GetValue(row));
correction[row]=square_math::RoundAdd(d.hi,square_math::RoundSub(d.lo,lo.GetValue(row)));
}
}

};
