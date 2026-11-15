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

struct TinyFallbackOperations {

template<class Kernel>
__aicore__ static inline void EmitSquareDma(Kernel& kernel, Zq3 base, Zq2 batch)
{
if(kernel.tinyPrivateFlow_){kernel.EmitTinyPrivate(batch);return;}

WorkspaceOperations::CopySquareOutput(kernel.output_, kernel.w_+kernel.OffX(base),
    kernel.x_, kernel.r_, batch, kernel.N(), kernel.K(), kernel.tiling_.residualElements);
}

template<Zq2 TinyN, class Kernel>
__aicore__ static inline void TinyTriangular(Kernel& kernel, Zq3 base, const __gm__ float* in, __gm__ float* out, Zq2 stride)
{
if(kernel.tinyVectorSafe_){kernel.TinyVectorTriangular(base,in,out,stride);return;}
const Zq2 n=kernel.N();auto y=kernel.solveY_;
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<n;++i)y.SetValue(i,in[Zq3(i)*stride]);
for(Zq2 i=0;i<n;++i){
const Zq2 pick=Zq2(int32_t(kernel.w_[kernel.OffP(base)+i]));
if(pick!=i){const float v=y.GetValue(i);y.SetValue(i,y.GetValue(pick));y.SetValue(pick,v);}
}
TinyScalarTriangular<TinyN>::Solve(kernel.factors_,y,kernel.w_+kernel.OffInv(base),out,stride);
}

template<Zq2 TinyN, class Kernel>
__aicore__ static inline bool BuildTinyInverse(Kernel& kernel, Zq3 base)
{
constexpr Zq2 n=TinyN;
auto order=kernel.residualTile_.template Zq6<Zq2>();
auto physicalOffsets=kernel.tinyNegative_[48u].template Zq6<Zq2>();
auto inverseDiagonal=kernel.tinyNegative_[32u];
InitializeTinyInverse<TinyN>(kernel, order, physicalOffsets, inverseDiagonal);
for(Zq2 j=0;j<n;++j){
ZqB<Zq5::S_V>();
Gather(kernel.solveY_,kernel.factors_,order,j*4u,n);Zq1<PIPE_V>();
Abs(kernel.vr_,kernel.solveY_,n);Zq1<PIPE_V>();
Zq3 activePivot[2]={((Zq3(1u)<<n)-1u)&(~Zq3(0u)<<j),0u};
WholeReduceMax(kernel.vr_[16u],kernel.vr_,activePivot,int32_t(1),int32_t(1),int32_t(1),int32_t(8),ReduceOrder::ORDER_VALUE_INDEX);ZqB<Zq5::V_S>();
Zq2 pick=j;float best=kernel.AbsF(kernel.solveY_.GetValue(j));
const float maximum=kernel.vr_[16u].GetValue(0u);
const Zq2 candidate=kernel.vr_[16u].template Zq6<Zq2>().GetValue(1u);
if(candidate>=j&&candidate<n&&maximum==maximum)pick=candidate;
else for(Zq2 i=j+1u;i<n;++i){const float x=kernel.AbsF(kernel.solveY_.GetValue(i));if(x>best){best=x;pick=i;}}
const Zq2 selected=order.GetValue(pick),physical=selected/128u;
const float diagonal=kernel.solveY_.GetValue(pick),magnitude=kernel.AbsF(diagonal);
if(!(magnitude>=1.0e-30f&&magnitude<=1.0e30f))return false;
const float inverse=kernel.SeedInverse(diagonal);
const Zq2 old=order.GetValue(j);order.SetValue(j,selected);order.SetValue(pick,old);inverseDiagonal.SetValue(j,inverse);
ZqB<Zq5::S_V>();
Gather(kernel.vr_[64u],kernel.factors_,physicalOffsets,j*4u,16u);Zq1<PIPE_V>();
Zq3 pivotLane[2]={Zq3(1u)<<physical,0u};
Duplicate<float>(kernel.vr_[64u],0.0f,pivotLane,Zq4(1),1,8);Zq1<PIPE_V>();
Brcb(kernel.block_,kernel.vr_[64u],Zq4(2),Zq9{1,8});
Muls(kernel.vr_[128u],kernel.factors_[physical*32u],-inverse,32u);Zq1<PIPE_V>();

Zq3 columns[2]={0xffffffffu,0u};
MulAddDst<float,float>(kernel.factors_,kernel.vr_[128u],kernel.block_,columns,Zq4(n),Zq0{1,1,0,4,0,1});
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
}
return FinalizeTinyInverse<TinyN>(kernel, order, inverseDiagonal);
}

template<Zq2 TinyN, class Kernel>
__aicore__ static inline bool BuildTinyInverseFast(Kernel& kernel, Zq3 base)
{
constexpr Zq2 n=TinyN;
auto order=kernel.residualTile_.template Zq6<Zq2>();
auto physicalOffsets=kernel.tinyNegative_[48u].template Zq6<Zq2>();
auto inverseDiagonal=kernel.tinyNegative_[32u];
InitializeTinyInverse<TinyN>(kernel, order, physicalOffsets, inverseDiagonal);


Abs(kernel.panel_[13u*256u],kernel.panel_[14u*256u],256u);Zq1<PIPE_V>();
ReduceMax(kernel.vr_[16u],kernel.panel_[13u*256u],kernel.panel_[12u*256u],int32_t(256),true);ZqB<Zq5::V_S>();
const float inputScale=kernel.vr_[16u].GetValue(0u);
for(Zq2 j=0;j<n;++j){
ZqB<Zq5::S_V>();
Gather(kernel.vr_[64u],kernel.factors_,physicalOffsets,j*4u,16u);Zq1<PIPE_V>();
ZqB<Zq5::V_S>();
const float diagonal=kernel.vr_[64u].GetValue(j),magnitude=kernel.AbsF(diagonal);
if(!(magnitude>=1.0e-30f&&magnitude<=1.0e30f&&magnitude*8192.0f>=inputScale))return false;
const float inverse=kernel.SeedInverse(diagonal);
inverseDiagonal.SetValue(j,inverse);
Zq3 pivotLane[2]={Zq3(1u)<<j,0u};
Duplicate<float>(kernel.vr_[64u],0.0f,pivotLane,Zq4(1),1,8);Zq1<PIPE_V>();
Brcb(kernel.block_,kernel.vr_[64u],Zq4(2),Zq9{1,8});
Muls(kernel.vr_[128u],kernel.factors_[j*32u],-inverse,32u);Zq1<PIPE_V>();
Zq3 columns[2]={0xffffffffu,0u};
MulAddDst<float,float>(kernel.factors_,kernel.vr_[128u],kernel.block_,columns,Zq4(n),Zq0{1,1,0,4,0,1});
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
}
return FinalizeTinyInverse<TinyN>(kernel, order, inverseDiagonal);
}

template<Zq2 n, class Kernel>
__aicore__ static inline void InitializeTinyInverse(Kernel& kernel, Zq7<Zq2> order,
    Zq7<Zq2> physicalOffsets, Zq7<float> inverseDiagonal)
{
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<16u;++i){order.SetValue(i,i*128u);physicalOffsets.SetValue(i,i*128u);inverseDiagonal.SetValue(i,0.0f);}
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
Duplicate<float>(kernel.factors_[16u],0.0f,Zq3(16),Zq4(16),1,4);Zq1<PIPE_V>();
Adds<int32_t>(kernel.factors_.template Zq6<int32_t>(),kernel.panel_[14u*256u].template Zq6<int32_t>(),int32_t(0),Zq3(16),Zq4(16),UnaryRepeatParams{1,1,4,2});
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<n;++i)kernel.factors_.SetValue(i*32u+16u+i,1.0f);
}

template<Zq2 n, class Kernel>
__aicore__ static inline bool FinalizeTinyInverse(Kernel& kernel, Zq7<Zq2> order,
    Zq7<float> inverseDiagonal)
{
auto offsets=kernel.panel_.template Zq6<int32_t>();
ZqB<Zq5::S_V>();
for(Zq2 col=0;col<16u;++col)Adds<int32_t>(offsets[col*16u],order.template Zq6<int32_t>(),int32_t(col*4u+64u),16u);
Zq1<PIPE_V>();
Gather(kernel.panel_[12u*256u],kernel.factors_,offsets.template Zq6<Zq2>(),0u,256u);Zq1<PIPE_V>();
Mul(kernel.factors_,kernel.panel_[12u*256u],inverseDiagonal,Zq3(16),Zq4(16),Zq0{1,1,1,2,2,0});
Zq1<PIPE_V>();Abs(kernel.panel_[13u*256u],kernel.factors_,256u);Zq1<PIPE_V>();
for(Zq2 half=128u;half>=16u;half/=2u){Add(kernel.panel_[13u*256u],kernel.panel_[13u*256u],kernel.panel_[13u*256u+half],half);Zq1<PIPE_V>();}
ZqB<Zq5::V_S>();
for(Zq2 i=0;i<n;++i){const float sum=kernel.panel_[13u*256u].GetValue(i);if(!(sum>=0.0f&&sum<=1.0e30f))return false;}

kernel.tinyInverseReady_=true;return true;
}
template<Zq2 n, class Kernel>
__aicore__ static inline void ConsumeTinyInverse(Kernel& kernel, const __gm__ float* in, Zq2 stride)
{
ZqB<Zq5::V_S>();
if(!kernel.tinyLocalRhs_){ZqB<Zq5::S_V>();Adds(kernel.solveY_,kernel.tinyNegative_,0.0f,16u);Zq1<PIPE_V>();}
if constexpr (n == 15u) {Zq3 pad[2]={0x8000u,0u};Duplicate<float>(kernel.solveY_,0.0f,pad,Zq4(1),1,8);Zq1<PIPE_V>();}
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
Brcb(kernel.block_,kernel.solveY_,Zq4(2),Zq9{1,8});Zq1<PIPE_V>();
auto products=kernel.panel_[13u*256u];
Mul(products,kernel.factors_,kernel.block_,Zq3(16),Zq4(16),Zq0{1,1,0,2,2,1});Zq1<PIPE_V>();
for(Zq2 half=128u;half>=16u;half/=2u){Add(products,products,products[half],half);Zq1<PIPE_V>();}
Adds(kernel.solveY_,products,0.0f,16u);ZqB<Zq5::V_S>();
}

template<Zq2 n, class Kernel>
__aicore__ static inline void PrepareTinyVectorSolve(Kernel& kernel, Zq3 base)
{
if(kernel.tinyInverseReady_){kernel.tinyVectorSafe_=true;kernel.tinyPrivateFlow_=true;kernel.tinyLocalRhs_=false;return;}
kernel.tinyVectorSafe_=true;kernel.tinyPrivateFlow_=false;kernel.tinyLocalRhs_=false;
if constexpr (n == 16u) {
    if (kernel.tinyLuDiagonalKnown_) {
        kernel.tinyVectorSafe_=kernel.tinyLuSafe_;
    } else {
        CheckTinyDiagonals<n>(kernel);
    }
} else {
    CheckTinyDiagonals<n>(kernel);
}

if(!kernel.tinyVectorSafe_)return;
kernel.tinyPrivateFlow_=true;
if constexpr (n == 15u) kernel.solveY_.SetValue(15u,0.0f);
ZqB<Zq5::S_V>();
if constexpr (n == 16u) {
    if (!kernel.tinyNegativeReady_) Muls(kernel.tinyNegative_,kernel.factors_,-1.0f,256u);
} else {
    Muls(kernel.tinyNegative_,kernel.factors_,-1.0f,256u);
}
Zq1<PIPE_V>();
}

template<Zq2 n, class Kernel>
__aicore__ static inline void CheckTinyDiagonals(Kernel& kernel)
{
for (Zq2 j=0;j<n;++j) {
const float d=kernel.factors_.GetValue(j*16u+j);
if(!(kernel.AbsF(d)>=1.0e-30f&&kernel.AbsF(d)<=1.0e30f))kernel.tinyVectorSafe_=false;
}
}

template<Zq2 n, class Kernel>
__aicore__ static inline void TinyVectorTriangular(Kernel& kernel, Zq3 base, const __gm__ float* in, __gm__ float* out, Zq2 stride)
{
if(kernel.tinyInverseReady_){kernel.ConsumeTinyInverse(in,stride);return;}
auto y=kernel.solveY_;auto seeds=kernel.tinyNegative_[256u];
auto offsets=kernel.tinyNegative_[264u].template Zq6<Zq2>();
ZqB<Zq5::V_S>();
auto order=kernel.residualTile_.template Zq6<Zq2>();
if(kernel.tinyLocalRhs_){


auto permuted=kernel.tinyNegative_[288u];ZqB<Zq5::S_V>();
Gather(permuted,y,order,0u,16u);Zq1<PIPE_V>();
Adds(y,permuted,0.0f,16u);Zq1<PIPE_V>();
}else{
for(Zq2 i=0;i<n;++i)
y.SetValue(i,in[Zq3(order.GetValue(i)/4u)*stride]);
}
ZqB<Zq5::S_V>();SetMaskNorm();
for(Zq2 j=0;j+1u<n;++j){
Gather(seeds,y,offsets,j*4u,8u);Zq1<PIPE_V>();
Zq3 mask[2]={Zq3((Zq3(1u)<<n)-1u)&(~Zq3(0u)<<(j+1u)),0u};
MulAddDst<float,float>(y,kernel.tinyNegative_[j*16u],seeds,mask,Zq4(1),Zq0{1,1,0,8,8,0});
Zq1<PIPE_V>();
}
for(int32_t j=int32_t(n)-1;j>=0;--j){
const Zq2 row=Zq2(j);Zq3 one[2]={Zq3(1u)<<row,0u};
const float inverse=kernel.w_[kernel.OffInv(base)+row];
Muls<float>(y,y,inverse,one,Zq4(1),UnaryRepeatParams{1,1,8,8});Zq1<PIPE_V>();
if(row){
Gather(seeds,y,offsets,row*4u,8u);Zq1<PIPE_V>();
Zq3 mask[2]={(Zq3(1u)<<row)-1u,0u};
MulAddDst<float,float>(y,kernel.tinyNegative_[row*16u],seeds,mask,Zq4(1),Zq0{1,1,0,8,8,0});
Zq1<PIPE_V>();
}
}
ResetMask();ZqB<Zq5::V_S>();
if(!kernel.tinyPrivateFlow_)for(Zq2 i=0;i<n;++i)out[Zq3(i)*stride]=y.GetValue(i);
}

template<Zq2 n, class Kernel>
__aicore__ static inline void DecomposeSmallMasked(Kernel& kernel, Zq3 base)
{
constexpr Zq2 pitch=16u,p=16u;
if constexpr (n == 16u) { kernel.tinyLuDiagonalKnown_=true;kernel.tinyLuSafe_=true;kernel.tinyNegativeReady_=false; }
auto rowBytes=kernel.residualTile_.template Zq6<int32_t>();
for(Zq2 i=0;i<16u;++i)rowBytes.SetValue(i,int32_t(i*4u));
Zq3 active=(Zq3(1u)<<n)-1u;
for (Zq2 j=0;j<n;++j) FactorTinyPivot<n>(kernel, base, rowBytes, active, j);


auto offsets=kernel.panel_.template Zq6<int32_t>();
ZqB<Zq5::S_V>();
for(Zq2 col=0;col<16u;++col)
Adds<int32_t>(offsets[col*16u],rowBytes,int32_t(col*64u),16u);
Zq1<PIPE_V>();
Gather(kernel.panel_[256u],kernel.factors_,offsets.template Zq6<Zq2>(),0u,256u);Zq1<PIPE_V>();
if constexpr (n == 16u) {
    if (kernel.tinyLuSafe_) {
        Muls(kernel.tinyNegative_,kernel.panel_[256u],-1.0f,256u);Zq1<PIPE_V>();
        kernel.tinyNegativeReady_=true;
    } else {
        Adds(kernel.factors_,kernel.panel_[256u],0.0f,256u);
    }
} else {
    Adds(kernel.factors_,kernel.panel_[256u],0.0f,256u);
}
ZqB<Zq5::V_S>();
}

template<Zq2 n, class Kernel>
__aicore__ static inline Zq2 PickTinyPivot(Kernel& kernel, Zq7<int32_t> rowBytes, Zq2 j)
{
Zq2 pick=j,physical=Zq2(rowBytes.GetValue(j))/4u;
float best=kernel.AbsF(kernel.factors_.GetValue(j*16u+physical));
if(n-j>=8u&&best==best){
ZqB<Zq5::S_V>();
Gather(kernel.solveY_,kernel.factors_[j*16u],rowBytes.template Zq6<Zq2>(),0u,n);Zq1<PIPE_V>();
Abs(kernel.vr_,kernel.solveY_,n);Zq1<PIPE_V>();
if(j){Duplicate(kernel.vr_,-1.0f,j);Zq1<PIPE_V>();}
ReduceMax(kernel.vr_[16u],kernel.vr_,kernel.vr_[32u],int32_t(n),true);ZqB<Zq5::V_S>();
const float maximum=kernel.vr_[16u].GetValue(0u);
const Zq2 candidate=kernel.vr_[16u].template Zq6<Zq2>().GetValue(1u);
if(candidate>=j&&candidate<n&&maximum==maximum){
return candidate;
}
}
for(Zq2 i=j+1u;i<n;++i){
const Zq2 row=Zq2(rowBytes.GetValue(i))/4u;
const float value=kernel.AbsF(kernel.factors_.GetValue(j*16u+row));
if(value>best){best=value;pick=i;}
}
return pick;
}

template<Zq2 n, class Kernel>
__aicore__ static inline void FactorTinyPivot(Kernel& kernel, Zq3 base, Zq7<int32_t> rowBytes, Zq3& active, Zq2 j)
{
constexpr Zq2 pitch=16u,p=16u;
const Zq2 pick = PickTinyPivot<n>(kernel, rowBytes, j);
const Zq2 physical=Zq2(rowBytes.GetValue(pick))/4u;
kernel.w_[kernel.OffP(base)+j]=float(int32_t(pick));
if(pick!=j){
const int32_t old=rowBytes.GetValue(j);
rowBytes.SetValue(j,rowBytes.GetValue(pick));rowBytes.SetValue(pick,old);
}
active&=~(Zq3(1u)<<physical);
const float diagonal=kernel.factors_.GetValue(j*16u+physical);
if(diagonal==0.0f){if constexpr (n == 16u) kernel.tinyLuSafe_=false;kernel.w_[kernel.OffInv(base)+j]=0.0f;return;}
const bool safe=kernel.AbsF(diagonal)>=1.0e-30f&&kernel.AbsF(diagonal)<=1.0e30f;
if constexpr (n == 16u) kernel.tinyLuSafe_=kernel.tinyLuSafe_&&safe;
const float inverse=safe?kernel.SeedInverse(diagonal):0.0f;kernel.w_[kernel.OffInv(base)+j]=inverse;
if(j+1u==n)return;
Zq3 valid[2]={active,0u};
if(safe){
ZqB<Zq5::S_V>();
Muls<float>(kernel.factors_[j*16u],kernel.factors_[j*16u],inverse,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});
ResetMask();
}else{
for(Zq2 i=j+1u;i<n;++i){
const Zq2 row=Zq2(rowBytes.GetValue(i))/4u;
kernel.factors_.SetValue(j*16u+row,square_math::Divide(kernel.factors_.GetValue(j*16u+row),diagonal));
}
}
ZqB<Zq5::S_V>();
auto offsets=kernel.vr_[10u*p].template Zq6<Zq2>();
Gather(kernel.vr_[8u*p],kernel.factors_,offsets,physical*4u,16u);Zq1<PIPE_V>();
Muls(kernel.vr_[8u*p],kernel.vr_[8u*p],-1.0f,16u);Zq1<PIPE_V>();
Brcb(kernel.block_,kernel.vr_[8u*p],Zq4(2),Zq9{1,8});Zq1<PIPE_V>();
MulAddDst<float,float>(kernel.factors_[(j+1u)*16u],kernel.factors_[j*16u],kernel.block_[(j+1u)*8u],valid,Zq4(n-1u-j),Zq0{1,1,0,2,0,1});
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
}

template<Zq2 n, class Kernel>
__aicore__ static inline void TinyResidualTile(Kernel& kernel, Zq3 base, Zq2 row, Zq2 slot)
{
constexpr Zq2 p=256u,active=p,groupWidth=1u;
auto work=kernel.panel_,extra=kernel.panel_[10u*p],input=kernel.panel_[14u*p+slot*p];
Zq7<float>a=work,ah=work[2u*p],al=work[4u*p],u=work[9u*p];
Gather(a,input,kernel.panel_[16u*p].template Zq6<Zq2>(),0u,active);Zq1<PIPE_V>();
SetMaskCount();ZqA<uint16_t,MaskMode::COUNTER>(0u,2u*active);
And<uint16_t,false>(ah.template Zq6<uint16_t>(),a.template Zq6<uint16_t>(),
kernel.panel_[10u*p].template Zq6<uint16_t>(),Zq3(0),Zq4(1),Zq0{1,1,1,8,8,0});
Zq1<PIPE_V>();ZqA<float,MaskMode::COUNTER>(0u,active);
Sub<float,false>(al,a,ah,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
for(Zq2 rhs=0;rhs<groupWidth;++rhs){
Zq7<float>x=rhs?extra:work[p];
Zq7<float>xh=rhs?extra[p]:work[3u*p];
Zq7<float>xl=rhs?extra[2u*p]:work[5u*p];
Zq7<float>hi=rhs?work[8u*p]:work[6u*p];
Zq7<float>lo=rhs?extra[3u*p]:work[7u*p];
Zq7<float>t=rhs?work[9u*p]:work[8u*p];

TinyResidualProduct<n>(work, hi, lo, t, a, ah, al, x, xh, xl);
}
for(Zq2 rhs=0;rhs<groupWidth;++rhs){
Zq7<float>hi=rhs?work[8u*p]:work[6u*p];
Zq7<float>lo=rhs?extra[3u*p]:work[7u*p];
MergeCounterResidualHalves<128u>(ah, hi, lo, al, u, 128u);
MergeCounterResidualHalves<64u>(hi, ah, lo, al, u, 64u);
MergeCounterResidualHalves<32u>(ah, hi, lo, al, u, 32u);
MergeCounterResidualHalves<16u>(hi, ah, lo, al, u, 16u);
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
FinishTinyResidual(kernel, base, row, work, extra);
}

template<class Kernel>
__aicore__ static inline void FinishTinyResidual(Kernel& kernel, Zq3 base, Zq2 row,
    Zq7<float> work, Zq7<float> extra)
{
constexpr Zq2 p=256u;
if(kernel.tinyInverseReady_){

auto b=kernel.tinyNegative_,neg=kernel.panel_[12u*p],sum=kernel.panel_[12u*p+16u];
auto bb=kernel.panel_[12u*p+32u],left=kernel.panel_[12u*p+48u],right=kernel.panel_[12u*p+64u];
ZqB<Zq5::S_V>();
Muls(neg,work[6u*p],-1.0f,16u);Zq1<PIPE_V>();
Add(sum,b,neg,16u);Zq1<PIPE_V>();
Sub(bb,sum,b,16u);Zq1<PIPE_V>();
Sub(left,sum,bb,16u);Zq1<PIPE_V>();
Sub(left,b,left,16u);Sub(right,neg,bb,16u);Zq1<PIPE_V>();
Add(left,left,right,16u);Zq1<PIPE_V>();
Sub(left,left,work[7u*p],16u);Zq1<PIPE_V>();
Add(kernel.solveY_,sum,left,16u);Zq1<PIPE_V>();return;
}
const Zq2 count=kernel.N()-row<16u?kernel.N()-row:16u;
for(Zq2 rhs=0;rhs<1u;++rhs){
auto hi=rhs?work[8u*p]:work[6u*p];auto lo=rhs?extra[3u*p]:work[7u*p];
for(Zq2 lane=0;lane<count;++lane){
Pair d=square_math::Sum(kernel.w_[kernel.OffB(base)+Zq3(row+lane)+rhs],-hi.GetValue(lane));
const float value=square_math::RoundAdd(d.hi,square_math::RoundSub(d.lo,lo.GetValue(lane)));
if(kernel.tinyPrivateFlow_)kernel.solveY_.SetValue(row+lane,value);
else kernel.w_[kernel.OffD(base)+Zq3(row+lane)+rhs]=value;
}
}
}

template<Zq2 n>
__aicore__ static inline void TinyResidualProduct(Zq7<float> work, Zq7<float> hi, Zq7<float> lo,
    Zq7<float> t, Zq7<float> a, Zq7<float> ah, Zq7<float> al,
    Zq7<float> x, Zq7<float> xh, Zq7<float> xl)
{
if constexpr (n == 15u) {
    CompensatedTileProduct(hi, lo, t, a, ah, al, x, xh, xl);
} else {
constexpr Zq2 p=256u;
Mul<float,false>(hi,a,x,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Mul<float,false>(lo,ah,xh,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
Sub<float,false>(lo,lo,hi,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});


auto cross2=work[9u*p],cross3=work[11u*p];
Mul<float,false>(t,ah,xl,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Mul<float,false>(cross2,al,xh,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Mul<float,false>(cross3,al,xl,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,cross2,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,cross3,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
}
}

};
