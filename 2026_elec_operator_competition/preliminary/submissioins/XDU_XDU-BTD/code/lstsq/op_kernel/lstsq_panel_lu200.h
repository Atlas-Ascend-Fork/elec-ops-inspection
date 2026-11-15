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

struct PanelLu200 {
static constexpr Zq2 n=200u, p=256u, pitch=8u;

template<bool StageRhs, class Kernel>
__aicore__ static inline void Factor(Kernel& kernel, Zq3 base)
{
auto offsets=kernel.vr_[10u*p].template Zq6<Zq2>();
auto toRows=kernel.residualTile_;
auto coefficient=StageRhs?kernel.vr_:kernel.vr_[8u*p],seed=kernel.rows_[32u];
auto upperOffsets=kernel.vr_[9u*p+224u].template Zq6<Zq2>();

PrepareMaps(kernel, toRows);
if constexpr (StageRhs) {
for(Zq2 cc=0;cc<16u;++cc)upperOffsets.SetValue(cc,cc<9u?cc*200u*4u:0u);
for(Zq2 row=0;row<n;++row)kernel.block_.SetValue(8u*n+row,kernel.w_[kernel.OffB(base)+row]);
} else {
for(Zq2 cc=0;cc<8u;++cc)upperOffsets.SetValue(cc,cc*200u*4u);
}
ZqB<Zq5::S_V>();
for(Zq2 jb=0;jb<n;jb+=pitch){
const Zq2 je=jb+pitch,tail=n-je;
for(Zq2 cc=0;cc<pitch;++cc)
Gather(kernel.block_[cc*n+jb],kernel.factors_,offsets[jb],(jb+cc)*4u,n-jb);
ZqB<Zq5::V_S>();
FactorPanel<StageRhs>(kernel, base, jb, je, coefficient, seed, upperOffsets);
ZqB<Zq5::S_V>();
Gather(kernel.vr_[jb*pitch],kernel.block_,toRows[jb*pitch],0u,(n-jb)*pitch);Zq1<PIPE_V>();
Adds(kernel.factors_[jb*n+jb],kernel.vr_[jb*pitch],0.0f,Zq3(pitch),Zq4(n-jb),UnaryRepeatParams{1,1,25,1});
ResetMask();ZqB<Zq5::V_S>();
if(!tail)continue;
SolveUpper<StageRhs>(kernel, jb, je, tail);
UpdateTrailing(kernel, jb, je, tail);
}
if constexpr (StageRhs) {
ZqB<Zq5::S_V>();Duplicate(kernel.output_,0.0f,256u);Zq1<PIPE_V>();
Adds(kernel.output_,kernel.block_[8u*n],0.0f,n);Zq1<PIPE_V>();kernel.initialForwardReady_=true;
}
}

template<class Kernel>
__aicore__ static inline void PrepareMaps(Kernel& kernel, Zq7<Zq2> toRows)
{
auto rowBytes=kernel.rows_.template Zq6<int32_t>();
auto columnBytes=kernel.vr_.template Zq6<int32_t>();
for(Zq2 row=0;row<n;++row)rowBytes.SetValue(row,int32_t(row*4u));
for(Zq2 cc=0;cc<pitch;++cc)columnBytes.SetValue(cc,int32_t(cc*n*4u));
ZqB<Zq5::S_V>();ResetMask();SetMaskNorm();
Brcb(toRows.template Zq6<int32_t>(),rowBytes,Zq4(n/8u),Zq9{1,8});
Zq1<PIPE_V>();
Add<int32_t>(toRows.template Zq6<int32_t>(),toRows.template Zq6<int32_t>(),columnBytes,
Zq3(64),Zq4(n/8u),Zq0{1,1,0,8,8,0});
Zq1<PIPE_V>();
}

template<bool StageRhs, class Kernel>
__aicore__ static inline void FactorPanel(Kernel& kernel, Zq3 base, Zq2 jb, Zq2 je, Zq7<float> coefficient, Zq7<float> seed, Zq7<Zq2> upperOffsets)
{
for(Zq2 j=jb;j<je;++j){
const Zq2 c=j-jb;
const Zq2 pick=kernel.PivotIndex(kernel.block_[c*n],j);
kernel.w_[kernel.OffP(base)+j]=float(int32_t(pick));
if(pick!=j){
SwapPivot<StageRhs>(kernel, j, pick);
}
const float diagonal=kernel.block_.GetValue(c*n+j);
if(diagonal==0.0f){kernel.w_[kernel.OffInv(base)+j]=0.0f;continue;}
const bool safe=kernel.AbsF(diagonal)>=1.0e-30f&&kernel.AbsF(diagonal)<=1.0e30f;
const float inverse=safe?kernel.SeedInverse(diagonal):0.0f;kernel.w_[kernel.OffInv(base)+j]=inverse;
if(j+1u==n)continue;
const Zq2 transferStart=(j+1u)/8u*8u;
const Zq2 transferCount=n-transferStart;
if constexpr (StageRhs) {
    if (safe && transferCount > 64u) {
        UpdateWidePivot(kernel, c, j, transferStart, transferCount, inverse, coefficient, seed, upperOffsets);
        continue;
    }
}
if (safe && transferCount <= 64u) {
    UpdateShortPivot<StageRhs>(kernel, c, j, transferStart, transferCount, inverse);
    continue;
}
UpdateScalarPivot<StageRhs>(kernel, c, j, diagonal, safe, inverse, coefficient, seed, upperOffsets);
}
}

template<bool StageRhs, class Kernel>
__aicore__ static inline void SwapPivot(Kernel& kernel, Zq2 j, Zq2 pick)
{
for(Zq2 cc=0;cc<pitch+Zq2(StageRhs);++cc){
const float temp=kernel.block_.GetValue(cc*n+j);
kernel.block_.SetValue(cc*n+j,kernel.block_.GetValue(cc*n+pick));
kernel.block_.SetValue(cc*n+pick,temp);
}
ZqB<Zq5::S_V>();
Adds(kernel.rows_,kernel.factors_[j*n],0.0f,n);Zq1<PIPE_V>();
Adds(kernel.factors_[j*n],kernel.factors_[pick*n],0.0f,n);Zq1<PIPE_V>();
Adds(kernel.factors_[pick*n],kernel.rows_,0.0f,n);ZqB<Zq5::V_S>();
}

template<class Kernel>
__aicore__ static inline void UpdateWidePivot(Kernel& kernel, Zq2 c, Zq2 j, Zq2 transferStart, Zq2 transferCount, float inverse, Zq7<float> coefficient, Zq7<float> seed, Zq7<Zq2> upperOffsets)
{
const Zq2 head=j+1u-transferStart;
Zq3 valid[2]={~Zq3(0u)<<head,0u};
ZqB<Zq5::S_V>();ResetMask();SetMaskNorm();
Muls(kernel.block_[c*n+transferStart],kernel.block_[c*n+transferStart],inverse,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});
ResetMask();
Muls(kernel.block_[c*n+transferStart+64u],kernel.block_[c*n+transferStart+64u],inverse,transferCount-64u);
Zq1<PIPE_V>();
if(c<3u){


Gather(seed,kernel.block_,upperOffsets,j*4u,16u);Zq1<PIPE_V>();
Muls(seed,seed,-1.0f,16u);Zq1<PIPE_V>();
Brcb(coefficient,seed,Zq4(2),Zq9{8,64});Zq1<PIPE_V>();
MulAddDst<float,float>(kernel.block_[(c+1u)*n+transferStart],kernel.block_[c*n+transferStart],coefficient[(c+1u)*64u],
valid,Zq4(pitch-c),Zq0{1,1,0,25,0,8});
ResetMask();
for(Zq2 chunk=64u;chunk<transferCount;chunk+=64u){
const Zq2 width=transferCount-chunk<64u?transferCount-chunk:64u;
MulAddDst<float,float>(kernel.block_[(c+1u)*n+transferStart+chunk],kernel.block_[c*n+transferStart+chunk],coefficient[(c+1u)*64u],
Zq3(width),Zq4(pitch-c),Zq0{1,1,0,25,0,8});ResetMask();
}
}else{
for(Zq2 cc=c+1u;cc<=pitch;++cc){
const float u=kernel.block_.GetValue(cc*n+j);
Axpy<float,float>(kernel.block_[cc*n+transferStart],kernel.block_[c*n+transferStart],-u,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});
ResetMask();
Axpy(kernel.block_[cc*n+transferStart+64u],kernel.block_[c*n+transferStart+64u],-u,int32_t(transferCount-64u));
}
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
}

template<bool StageRhs, class Kernel>
__aicore__ static inline void UpdateShortPivot(Kernel& kernel, Zq2 c, Zq2 j, Zq2 transferStart, Zq2 transferCount, float inverse)
{
const Zq2 head=j+1u-transferStart;
const Zq3 low=transferCount==64u?~Zq3(0u):(Zq3(1u)<<transferCount)-1u;
Zq3 valid[2]={low&(~Zq3(0u)<<head),0u};
ZqB<Zq5::S_V>();SetMaskNorm();
Muls(kernel.block_[c*n+transferStart],kernel.block_[c*n+transferStart],inverse,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});
Zq1<PIPE_V>();
for(Zq2 cc=c+1u;cc<pitch+Zq2(StageRhs);++cc){
const float u=kernel.block_.GetValue(cc*n+j);
Axpy<float,float>(kernel.block_[cc*n+transferStart],kernel.block_[c*n+transferStart],-u,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});
}
ResetMask();ZqB<Zq5::V_S>();
}

template<bool StageRhs, class Kernel>
__aicore__ static inline void UpdateScalarPivot(Kernel& kernel, Zq2 c, Zq2 j, float diagonal, bool safe, float inverse, Zq7<float> coefficient, Zq7<float> seed, Zq7<Zq2> upperOffsets)
{
const Zq2 start=(j+1u)/8u*8u,count=n-start;
for(Zq2 i=start;i<=j;++i)kernel.rows_.SetValue(i-start,kernel.block_.GetValue(c*n+i));
if(safe){
ZqB<Zq5::S_V>();Muls(kernel.block_[c*n+start],kernel.block_[c*n+start],inverse,count);
ZqB<Zq5::V_S>();
}else for(Zq2 i=j+1u;i<n;++i)
kernel.block_.SetValue(c*n+i,square_math::Divide(kernel.block_.GetValue(c*n+i),diagonal));
for(Zq2 i=start;i<=j;++i)kernel.block_.SetValue(c*n+i,0.0f);
ZqB<Zq5::S_V>();
if(count>=104u&&c<4u){
SetMaskNorm();ResetMask();
Gather(seed,kernel.block_,upperOffsets,j*4u,StageRhs?16u:8u);Zq1<PIPE_V>();
Muls(seed,seed,-1.0f,StageRhs?16u:8u);Zq1<PIPE_V>();
Brcb(coefficient,seed,Zq4(StageRhs?2u:1u),Zq9{8,64});Zq1<PIPE_V>();
for(Zq2 chunk=0;chunk<count;chunk+=64u){
const Zq2 width=count-chunk<64u?count-chunk:64u;
MulAddDst<float,float>(kernel.block_[(c+1u)*n+start+chunk],kernel.block_[c*n+start+chunk],coefficient[(c+1u)*64u],
Zq3(width),Zq4(pitch-c-1u+Zq2(StageRhs)),Zq0{1,1,0,25,0,8});
ResetMask();
}
ZqB<Zq5::V_S>();
}else{
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,count);
for(Zq2 cc=c+1u;cc<pitch+Zq2(StageRhs);++cc){
const float u=kernel.block_.GetValue(cc*n+j);
Axpy<float,float,false>(kernel.block_[cc*n+start],kernel.block_[c*n+start],-u,
Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
}
for(Zq2 i=start;i<=j;++i)kernel.block_.SetValue(c*n+i,kernel.rows_.GetValue(i-start));
}

template<bool StageRhs, class Kernel>
__aicore__ static inline void SolveUpper(Kernel& kernel, Zq2 jb, Zq2 je, Zq2 tail)
{
if constexpr (StageRhs) {
ZqB<Zq5::S_V>();
ZqB<Zq5::S_V>();ResetMask();SetMaskNorm();
for(Zq2 prev=0;prev+1u<pitch;++prev)
Brcb(kernel.vr_[prev*64u],kernel.block_[prev*n+jb],Zq4(1),Zq9{1,8});
Zq1<PIPE_V>();Muls(kernel.vr_,kernel.vr_,-1.0f,448u);Zq1<PIPE_V>();
for(Zq2 prev=0;prev+1u<pitch;++prev){
for(Zq2 chunk=0;chunk<tail;chunk+=64u){
const Zq2 count=tail-chunk<64u?tail-chunk:64u;
MulAddDst<float,float>(kernel.factors_[(jb+prev+1u)*n+je+chunk],kernel.factors_[(jb+prev)*n+je+chunk],kernel.vr_[prev*64u+(prev+1u)*8u],
Zq3(count),Zq4(pitch-prev-1u),Zq0{1,1,0,25,0,1});
}
Zq1<PIPE_V>();
}
} else {
ZqB<Zq5::S_V>();
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,tail);
for(Zq2 row=0;row<pitch;++row)for(Zq2 prev=0;prev<row;++prev){
const float mult=kernel.block_.GetValue(prev*n+jb+row);
Axpy<float,float,false>(kernel.factors_[(jb+row)*n+je],kernel.factors_[(jb+prev)*n+je],-mult,
Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});Zq1<PIPE_V>();
}
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
}

template<class Kernel>
__aicore__ static inline void UpdateTrailing(Kernel& kernel, Zq2 jb, Zq2 je, Zq2 tail)
{
ZqB<Zq5::S_V>();Muls(kernel.block_,kernel.block_,-1.0f,pitch*n);Zq1<PIPE_V>();


for(Zq2 pp=0;pp<pitch;++pp){
Brcb(kernel.vr_,kernel.block_[pp*n+je],Zq4(tail/8u),Zq9{1,8});
Zq1<PIPE_V>();
for(Zq2 chunk=0;chunk<tail;chunk+=64u){
const Zq2 width=tail-chunk<64u?tail-chunk:64u;
MulAddDst<float,float>(kernel.factors_[je*n+je+chunk],kernel.factors_[(jb+pp)*n+je+chunk],kernel.vr_,
Zq3(width),Zq4(tail),Zq0{1,1,0,25,0,1});
}
ResetMask();Zq1<PIPE_V>();
}
ZqB<Zq5::V_S>();ZqB<Zq5::S_V>();
}

};
