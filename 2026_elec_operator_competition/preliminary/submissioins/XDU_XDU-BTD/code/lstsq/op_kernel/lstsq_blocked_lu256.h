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

struct BlockedLu256 {
static constexpr Zq2 n=256u, pitch=8u, tilePitch=264u, upperPitch=264u;

template<bool MaskedTransfer, class Kernel>
__aicore__ static inline void Factor(Kernel& kernel, Zq3 base)
{
const Zq2 tileRows=kernel.K()==1u?48u:32u;
auto tailSeed=kernel.factors_[4096u];auto tailIndex=kernel.factors_[5120u].template Zq6<Zq2>();
if (tileRows == 48u) for(Zq2 rank=0;rank<8u;++rank)for(Zq2 row=0;row<56u;++row)
tailIndex.SetValue(rank*56u+row,(rank*256u+(row<48u?row:0u))*4u);
ZqB<Zq5::S_V>();
__gm__ float*piv=kernel.w_+kernel.OffP(base);
GlobalTensor<float>matrix;matrix.SetGlobalBuffer(kernel.w_+kernel.OffLU(base),n*n);
auto upper=kernel.block_,uRows=kernel.block_[8u*n];
auto panelOffsets=kernel.rows_[64u].template Zq6<Zq2>();
for(Zq2 cc=0;cc<pitch;++cc)panelOffsets.SetValue(cc,cc*n*4u);

auto toColumns=kernel.residualTile_,toRows=kernel.residualTile_[pitch*n];
for(Zq2 row=0;row<n;++row)for(Zq2 col=0;col<pitch;++col){
toColumns.SetValue(col*n+row,(row*pitch+col)*4u);
toRows.SetValue(row*pitch+col,(col*n+row)*4u);
}
ZqB<Zq5::S_V>();

for (Zq2 jb=0;jb<n;jb+=pitch) {
    const Zq2 je=jb+pitch, tail=n-je;
    ReadPanel(kernel, matrix, upper, toColumns, jb);
    FactorPanel<MaskedTransfer>(kernel, base, matrix, upper, piv, panelOffsets, jb, je);
    WritePanel(kernel, matrix, upper, toRows, jb);
    if (tail == 0u) continue;
    SolvePanelTail(kernel, matrix, uRows, jb, je, tail);
    UpdateTrailingTiles(kernel, matrix, uRows, tailSeed, tailIndex, je, tail, tileRows);
}
ZqB<Zq5::MTE3_S>();


DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(matrix);
}

template<class Kernel>
__aicore__ static inline void ReadPanel(Kernel& kernel, GlobalTensor<float> matrix, Zq7<float> upper, Zq7<Zq2> toColumns, Zq2 jb)
{
DataCopyPad(upper[jb*pitch],matrix[jb*n+jb],
Zq8{uint16_t(n-jb),pitch*4u,(n-pitch)*4u,0u,0u},
DataCopyPadExtParams<float>{false,0u,0u,0.0f});
ZqB<Zq5::MTE2_V>();


for(Zq2 cc=0;cc<pitch;++cc)
Gather(kernel.panel_[cc*n+jb],upper,toColumns[cc*n+jb],0u,n-jb);
ZqB<Zq5::V_S>();
}

template<class Kernel>
__aicore__ static inline void WritePanel(Kernel& kernel, GlobalTensor<float> matrix, Zq7<float> upper, Zq7<Zq2> toRows, Zq2 jb)
{
ZqB<Zq5::MTE3_S>();
ZqB<Zq5::S_V>();
Gather(upper[jb*pitch],kernel.panel_,toRows[jb*pitch],0u,(n-jb)*pitch);
ZqB<Zq5::V_S>();
ZqB<Zq5::S_MTE3>();
DataCopyPad(matrix[jb*n+jb],upper[jb*pitch],
Zq8{uint16_t(n-jb),pitch*4u,0u,(n-pitch)*4u,0u});
}

template<class Kernel>
__aicore__ static inline void SwapPivot(Kernel& kernel, GlobalTensor<float> matrix, Zq7<float> upper, Zq2 j, Zq2 pick)
{
for(Zq2 cc=0;cc<pitch;++cc){
const float temp=kernel.panel_.GetValue(cc*n+j);
kernel.panel_.SetValue(cc*n+j,kernel.panel_.GetValue(cc*n+pick));
kernel.panel_.SetValue(cc*n+pick,temp);
}
DataCopy(upper,matrix[j*n],n);
DataCopy(upper[n],matrix[pick*n],n);
ZqB<Zq5::MTE2_MTE3>();
DataCopy(matrix[j*n],upper[n],n);
DataCopy(matrix[pick*n],upper,n);
ZqB<Zq5::MTE3_MTE2>();
}

template<bool MaskedTransfer, class Kernel>
__aicore__ static inline void FactorPanel(Kernel& kernel, Zq3 base, GlobalTensor<float> matrix, Zq7<float> upper, __gm__ float* piv, Zq7<Zq2> offsets, Zq2 jb, Zq2 je)
{
for(Zq2 j=jb;j<je;++j){
const Zq2 c=j-jb;
const Zq2 pick=kernel.PivotIndex(kernel.panel_[c*n],j);
piv[j]=float(int32_t(pick));
if (pick != j) SwapPivot(kernel, matrix, upper, j, pick);
const float diagonal=kernel.panel_.GetValue(c*n+j);
if(diagonal==0.0f){kernel.w_[kernel.OffInv(base)+j]=0.0f;continue;}
const bool safe=kernel.AbsF(diagonal)>=1.0e-30f&&kernel.AbsF(diagonal)<=1.0e30f;
const float inverse=safe?kernel.SeedInverse(diagonal):0.0f;
kernel.w_[kernel.OffInv(base)+j]=inverse;
if(j+1u==n)continue;
if constexpr (MaskedTransfer) {
    const Zq2 transferStart=(j+1u)/8u*8u;
    const Zq2 transferCount=n-transferStart;
    if (safe && transferCount<=64u) {
        UpdateMaskedPivot(kernel, c, j, transferStart, transferCount, inverse);
        continue;
    }
}
UpdatePivotColumn(kernel, offsets, c, j, diagonal, safe, inverse);
}
}

template<class Kernel>
__aicore__ static inline void UpdateMaskedPivot(Kernel& kernel, Zq2 c, Zq2 j, Zq2 transferStart, Zq2 transferCount, float inverse)
{
const Zq2 head=j+1u-transferStart;
const Zq3 low=transferCount==64u?~Zq3(0u):(Zq3(1u)<<transferCount)-1u;
Zq3 valid[2]={low&(~Zq3(0u)<<head),0u};
ZqB<Zq5::S_V>();SetMaskNorm();
Muls(kernel.panel_[c*n+transferStart],kernel.panel_[c*n+transferStart],inverse,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});
Zq1<PIPE_V>();
for(Zq2 cc=c+1u;cc<pitch;++cc){
const float u=kernel.panel_.GetValue(cc*n+j);
Axpy<float,float>(kernel.panel_[cc*n+transferStart],kernel.panel_[c*n+transferStart],-u,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});
}
ResetMask();ZqB<Zq5::V_S>();
}

template<class Kernel>
__aicore__ static inline void UpdatePivotColumn(Kernel& kernel, Zq7<Zq2> panelOffsets, Zq2 c, Zq2 j, float diagonal, bool safe, float inverse)
{
auto panelCoefficient=kernel.factors_,panelSeed=kernel.factors_[512u];
const Zq2 start=(j+1u)/8u*8u,count=n-start;


for(Zq2 i=start;i<=j;++i)kernel.rows_.SetValue(i-start,kernel.panel_.GetValue(c*n+i));
if(safe){
ZqB<Zq5::S_V>();
Muls(kernel.panel_[c*n+start],kernel.panel_[c*n+start],inverse,count);
ZqB<Zq5::V_S>();
}else for(Zq2 i=j+1u;i<n;++i)
kernel.panel_.SetValue(c*n+i,square_math::Divide(kernel.panel_.GetValue(c*n+i),diagonal));
for(Zq2 i=start;i<=j;++i)kernel.panel_.SetValue(c*n+i,0.0f);
ZqB<Zq5::S_V>();
if(count>=128u&&c<3u){
SetMaskNorm();ResetMask();
if(c+1u<pitch){
Gather(panelSeed,kernel.panel_,panelOffsets,j*4u,8u);Zq1<PIPE_V>();
Muls(panelSeed,panelSeed,-1.0f,8u);Zq1<PIPE_V>();
Brcb(panelCoefficient,panelSeed,Zq4(1),Zq9{8,64});Zq1<PIPE_V>();
for(Zq2 chunk=start;chunk<n;chunk+=64u){
const Zq2 width=n-chunk<64u?n-chunk:64u;
MulAddDst<float,float>(kernel.panel_[(c+1u)*n+chunk],kernel.panel_[c*n+chunk],panelCoefficient[(c+1u)*64u],
Zq3(width),Zq4(pitch-c-1u),Zq0{1,1,0,32,0,8});
}
ResetMask();
}
ZqB<Zq5::V_S>();
}else{
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,count);
for(Zq2 cc=c+1u;cc<pitch;++cc){
const float u=kernel.panel_.GetValue(cc*n+j);
Axpy<float,float,false>(kernel.panel_[cc*n+start],kernel.panel_[c*n+start],-u,
Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
}
for(Zq2 i=start;i<=j;++i)kernel.panel_.SetValue(c*n+i,kernel.rows_.GetValue(i-start));
}

template<class Kernel>
__aicore__ static inline void SolvePanelTail(Kernel& kernel, GlobalTensor<float> matrix, Zq7<float> uRows, Zq2 jb, Zq2 je, Zq2 tail)
{
DataCopyPad(uRows,matrix[jb*n+je],
Zq8{uint16_t(pitch),tail*4u,je*4u,je/8u+1u,0u},
DataCopyPadExtParams<float>{false,0u,0u,0.0f});
ZqB<Zq5::MTE2_V>();
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,tail);
for(Zq2 row=0;row<pitch;++row){
for(Zq2 prev=0;prev<row;++prev){
const float mult=kernel.panel_.GetValue(prev*n+jb+row);
Axpy<float,float,false>(uRows[row*upperPitch],uRows[prev*upperPitch],-mult,
Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});
Zq1<PIPE_V>();
}
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_MTE3>();
DataCopyPad(matrix[jb*n+je],uRows,
Zq8{uint16_t(pitch),tail*4u,je/8u+1u,je*4u,0u});
}

template<class Kernel>
__aicore__ static inline void UpdateTrailingTiles(Kernel& kernel, GlobalTensor<float> matrix, Zq7<float> uRows, Zq7<float> tailSeed, Zq7<Zq2> tailIndex, Zq2 je, Zq2 tail, Zq2 tileRows)
{
auto tile=kernel.block_[16u*n+64u];
ZqB<Zq5::S_V>();
Muls(kernel.panel_,kernel.panel_,-1.0f,pitch*n);Zq1<PIPE_V>();
const Zq2 tiles=(tail+tileRows-1u)/tileRows;
const Zq2 first=tail<tileRows?tail:tileRows;
DataCopyPad(tile,matrix[je*n+je],
Zq8{uint16_t(first),tail*4u,je*4u,je/8u+1u,0u},
DataCopyPadExtParams<float>{false,0u,0u,0.0f});
SetFlag<Zq5::MTE2_V>(0);
for(Zq2 ti=0;ti<tiles;++ti){
const Zq2 ib=je+ti*tileRows,slot=ti&1u;
const Zq2 count=n-ib<tileRows?n-ib:tileRows;
Zq7<float>current=kernel.block_[16u*n+64u+slot*tileRows*tilePitch];


ZqB<Zq5::S_V>();


const Zq2 coefficientPitch=count==48u?56u:count+1u;
if(count==48u){
Gather(tailSeed,kernel.panel_,tailIndex,ib*4u,448u);Zq1<PIPE_V>();
Brcb(kernel.factors_,tailSeed,Zq4(56),Zq9{1,8});
}else for(Zq2 pp=0;pp<pitch;++pp)
Brcb(kernel.factors_[pp*coefficientPitch*8u],kernel.panel_[pp*n+ib],Zq4(count/8u),Zq9{1,8});
Zq1<PIPE_V>();
WaitFlag<Zq5::MTE2_V>(slot);
if(ti+1u<tiles){
const Zq2 nextSlot=slot^1u,next=ib+tileRows;
const Zq2 nextCount=n-next<tileRows?n-next:tileRows;
if(ti>=1u)WaitFlag<Zq5::MTE3_MTE2>(nextSlot);
DataCopyPad(kernel.block_[16u*n+64u+nextSlot*tileRows*tilePitch],matrix[next*n+je],
Zq8{uint16_t(nextCount),tail*4u,je*4u,je/8u+1u,0u},
DataCopyPadExtParams<float>{false,0u,0u,0.0f});
SetFlag<Zq5::MTE2_V>(nextSlot);
}
MultiplyTrailingTile(kernel, current, uRows, count, coefficientPitch, tail);
ResetMask();ZqB<Zq5::V_MTE3>(slot);
DataCopyPad(matrix[ib*n+je],current,
Zq8{uint16_t(count),tail*4u,je/8u+1u,je*4u,0u});
SetFlag<Zq5::MTE3_MTE2>(slot);
}
WaitFlag<Zq5::MTE3_MTE2>(0);
if(tiles>1u)WaitFlag<Zq5::MTE3_MTE2>(1);
ZqB<Zq5::MTE3_S>();
}

template<class Kernel>
__aicore__ static inline void MultiplyTrailingTile(Kernel& kernel, Zq7<float> current, Zq7<float> uRows, Zq2 count, Zq2 coefficientPitch, Zq2 tail)
{
SetMaskNorm();
const Zq2 full=tail/64u*64u;
for(Zq2 group=0u;group<2u;++group){
const Zq2 begin=group?full:0u,end=group?tail:full;
if(begin==end)continue;
const Zq2 width=group?tail-full:64u;
const Zq3 low=width==64u?~Zq3(0u):(Zq3(1u)<<width)-1u;
ZqA<float,MaskMode::NORMAL>(0u,low);
for(Zq2 pp=0u;pp<pitch;++pp){
for(Zq2 chunk=begin;chunk<end;chunk+=64u)
MulAddDst<float,float,false>(current[chunk],uRows[pp*upperPitch+chunk],kernel.factors_[pp*coefficientPitch*8u],
Zq3(width),Zq4(count),Zq0{1,1,0,33,0,1});
Zq1<PIPE_V>();
}
}
}


template<class Kernel>
__aicore__ static inline void PrepareColumnFactors(
    Kernel& kernel, Zq3 base, Zq2 lane, Zq2 participants)
{
constexpr Zq2 n=256u,rows=32u;
GlobalTensor<float>source,target;
source.SetGlobalBuffer(kernel.w_+kernel.OffLU(base),n*n);
target.SetGlobalBuffer(kernel.w_+kernel.OffA(base),n*n);
ZqB<Zq5::MTE3_S>();
DataCacheCleanAndInvalid<float,CacheLine::ENTIRE_DATA_CACHE,DcciDst::CACHELINE_OUT>(target);


auto columns=kernel.block_.template Zq6<int32_t>();
auto rowBytes=kernel.block_[n].template Zq6<int32_t>();
auto offsets=kernel.block_[rows*n].template Zq6<int32_t>();
ZqB<Zq5::V_S>();
for(Zq2 col=0;col<n;++col)columns.SetValue(col,int32_t(col*4u));
for(Zq2 row=0;row<rows;++row)rowBytes.SetValue(row,int32_t(row*n*4u));
ZqB<Zq5::S_V>();ResetMask();SetMaskNorm();

for(Zq2 lane=0;lane<rows;lane+=8u)
Brcb(offsets[lane],columns,Zq4(n/8u),Zq9{4,32});
Zq1<PIPE_V>();
for(Zq2 half=0;half<2u;++half)
Add<int32_t>(offsets[half*4096u],offsets[half*4096u],rowBytes,
Zq3(rows),Zq4(128),Zq0{1,1,1,4,4,0});
Zq1<PIPE_V>();
ZqB<Zq5::S_MTE2>();
for(Zq2 row=lane*rows;row<n;row+=participants*rows){

ZqB<Zq5::V_MTE2>();
DataCopy(kernel.block_,source[row*n],rows*n);
ZqB<Zq5::MTE2_V>();
Gather(kernel.factors_,kernel.block_,offsets.template Zq6<Zq2>(),0u,rows*n);
Zq1<PIPE_V>();Muls(kernel.factors_,kernel.factors_,-1.0f,rows*n);
ZqB<Zq5::V_MTE3>();
DataCopyPad(target[row],kernel.factors_,Zq8{uint16_t(n),rows*4u,0u,(n-rows)*4u,0u});

ZqB<Zq5::MTE3_S>();ZqB<Zq5::S_V>();
}
ZqB<Zq5::MTE3_MTE2>();
}

};
