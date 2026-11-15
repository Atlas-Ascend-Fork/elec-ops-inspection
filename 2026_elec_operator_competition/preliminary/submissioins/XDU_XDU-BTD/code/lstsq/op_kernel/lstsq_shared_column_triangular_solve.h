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
struct ColumnTriangularSolveOperations {
template<class Kernel>
__aicore__ static inline void ColumnTriangularBatch(Kernel &kernel, Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride,Zq2 width)
{
if((kernel.N()==15u||kernel.N()==16u)&&kernel.K()==1u&&width==1u){kernel.TinyTriangular(base,in,out,stride);return;}
const Zq2 n=kernel.N(),p=kernel.VectorPitch(),pitch=kernel.FactorPitch();
LoadColumnRhs(kernel, base, in, stride, width, n, p);
for(Zq2 row=0;row+1u<n;++row){
const Zq2 start=(row+1u)/8u*8u,count=pitch-start;
ZqB<Zq5::S_V>();
{
Adds(kernel.vr_[start],kernel.factors_[row*pitch+start],0.0f,count);
Zq1<PIPE_V>();
if(start<=row){Duplicate(kernel.vr_[start],0.0f,row-start+1u);Zq1<PIPE_V>();}
for(Zq2 col=0;col<width;++col){
const float value=kernel.solveY_[col*p].GetValue(row);
Axpy(kernel.solveY_[col*p+start],kernel.vr_[start],-value,int32_t(count));
}
}
ZqB<Zq5::V_S>();
}
ColumnBackward(kernel, base, out, stride, width);
}

template<class Kernel>
__aicore__ static inline void VectorOnlyTriangular64(Kernel &kernel, Zq3 base,const __gm__ float*in,
__gm__ float*out,Zq2 stride)
{
constexpr Zq2 n=64u,p=64u;
auto order=kernel.residualTile_;
ZqB<Zq5::V_S>();
if(kernel.producerCorrection_){
ZqB<Zq5::S_V>();
Gather(kernel.negative64_[4160u],kernel.solveY_,order,0u,64u);Zq1<PIPE_V>();

}else{
for(Zq2 i=0;i<64u;++i)kernel.solveY_.SetValue(i,0.0f);
for(Zq2 i=0;i<kernel.N();++i)kernel.solveY_.SetValue(i,in[Zq3(order.GetValue(i)/4u)*stride]);
}
auto expanded=kernel.panel_;auto state=kernel.block_;
ZqB<Zq5::S_V>();SetMaskNorm();ResetMask();
for(Zq2 c=0;c<n;++c)Brcb(expanded[c*512u],kernel.negative64_[c*64u],Zq4(8),Zq9{1,8});
Brcb(state,kernel.producerCorrection_?kernel.negative64_[4160u]:kernel.solveY_,Zq4(8),Zq9{1,8});Zq1<PIPE_V>();
SetMaskCount();
for(Zq2 row=0;row+1u<n;++row){
const Zq2 start=(row+1u)*8u;
ZqA<float,MaskMode::COUNTER>(0u,(n-row-1u)*8u);
MulAddDst<float,float,false>(state[start],expanded[row*512u+start],state[row*8u],Zq3(0),Zq4(1),Zq0{1,1,0,8,8,0});Zq1<PIPE_V>();
}
for(int32_t j=63;j>=0;--j){
const Zq2 row=Zq2(j);const float inverse=kernel.w_[kernel.OffInv(base)+row];
ZqA<float,MaskMode::COUNTER>(0u,8u);
Muls<float,false>(state[row*8u],state[row*8u],inverse,Zq3(0),Zq4(1),UnaryRepeatParams{1,1,8,8});Zq1<PIPE_V>();
if(!row)continue;
ZqA<float,MaskMode::COUNTER>(0u,row*8u);
MulAddDst<float,float,false>(state,expanded[row*512u],state[row*8u],Zq3(0),Zq4(1),Zq0{1,1,0,8,8,0});Zq1<PIPE_V>();
}
ResetMask();SetMaskNorm();ZqB<Zq5::V_S>();
auto offsets=kernel.vr_[10u*p].template Zq6<Zq2>();
if(!kernel.replicaOffsetsReady_){for(Zq2 i=0;i<64u;++i)offsets.SetValue(i,i*32u);kernel.replicaOffsetsReady_=true;}
ZqB<Zq5::S_V>();Gather(kernel.solveY_,state,offsets,0u,64u);Zq1<PIPE_V>();
ResetMask();ZqB<Zq5::V_S>();

}

template<class Kernel>
__aicore__ static inline void ColumnTriangularBatch64(Kernel& kernel, Zq3 base, const __gm__ float* in, __gm__ float* out, Zq2 stride, Zq2 width)
{
{if(kernel.augmented64Ready_){kernel.producerPrivate_=true;kernel.ConsumeAugmented64(in,stride);return;}}


{


if(!kernel.diagonalKnown_){
kernel.diagonalSafe_=true;
for(Zq2 i=0;i<64u;++i){
const float d=kernel.AbsF(kernel.factors_.GetValue(i*64u+i));
kernel.diagonalSafe_=kernel.diagonalSafe_&&d>=1.0e-30f&&d<=1.0e30f;
}
kernel.diagonalKnown_=true;
}
const bool safe64=width==1u&&kernel.diagonalSafe_;
if(safe64){
if(!kernel.negative64Ready_){
ZqB<Zq5::S_V>();Muls(kernel.negative64_,kernel.factors_,-1.0f,4096u);Zq1<PIPE_V>();
kernel.negative64Ready_=true;
}
kernel.producerPrivate_=true;
kernel.VectorOnlyTriangular64(base,in,out,stride);return;
}
}
ColumnForward64(kernel, base, in, stride, width);
ColumnBackward(kernel, base, out, stride, width);
}

template<class Kernel>
__aicore__ static inline void ApplyColumnPivots(Kernel& kernel, Zq3 base, Zq2 n, Zq2 p, Zq2 width)
{
TriangularSolveOperations::ApplyColumnPivots(kernel,base,n,p,width);
}

template<class Kernel>
__aicore__ static inline void LoadColumnRhs(Kernel& kernel, Zq3 base, const __gm__ float* in,
    Zq2 stride, Zq2 width, Zq2 n, Zq2 p)
{
ZqB<Zq5::S_V>();Duplicate(kernel.solveY_,0.0f,width*p);ZqB<Zq5::V_S>();
for(Zq2 col=0;col<width;++col)for(Zq2 row=0;row<n;++row)
kernel.solveY_[col*p].SetValue(row,in[Zq3(row)*stride+col]);
ApplyColumnPivots(kernel,base,n,p,width);
}

template<class Kernel>
__aicore__ static inline void UpdateColumnTriangularRow(Kernel& kernel,
    Zq2 row, Zq2 width, Zq2 p, Zq2 pitch)
{
TriangularSolveOperations::UpdateColumnTriangularRow(kernel,row,width,p,pitch);
}

template<class Kernel>
__aicore__ static inline void ColumnForward64(Kernel& kernel, Zq3 base,
    const __gm__ float* in, Zq2 stride, Zq2 width)
{
const Zq2 n=kernel.N(),p=kernel.VectorPitch(),pitch=kernel.FactorPitch();
LoadColumnRhs(kernel, base, in, stride, width, n, p);
for(Zq2 row=0;row+1u<n;++row){
ZqB<Zq5::S_V>();
UpdateColumnTriangularRow(kernel,row,width,p,pitch);
ZqB<Zq5::V_S>();
}
}

template<class Kernel>
__aicore__ static inline void ColumnBackward(Kernel& kernel, Zq3 base,
    __gm__ float* out, Zq2 stride, Zq2 width)
{
const Zq2 n=kernel.N(),p=kernel.VectorPitch(),pitch=kernel.FactorPitch();
for(int32_t row=int32_t(n)-1;row>=0;--row){
const float diagonal=kernel.factors_.GetValue(Zq2(row)*pitch+Zq2(row));
const bool safe=kernel.AbsF(diagonal)>=1.0e-30f&&kernel.AbsF(diagonal)<=1.0e30f;
const float inverse=kernel.w_[kernel.OffInv(base)+Zq2(row)];
TriangularSolveOperations::ApplyBackwardColumnRow(kernel.solveY_,kernel.factors_,row,p,pitch,width,diagonal,inverse,safe);
}
for(Zq2 col=0;col<width;++col)for(Zq2 row=0;row<n;++row)
out[Zq3(row)*stride+col]=kernel.solveY_[col*p].GetValue(row);
}

template<class Kernel>
__aicore__ static inline void ForwardColumnGm256(Kernel& kernel, GlobalTensor<float> columns, Zq2 width)
{
constexpr Zq2 n=256u,p=256u;
for(Zq2 begin=0;begin<n;begin+=32u){
ZqB<Zq5::V_MTE2>();
DataCopy(kernel.factors_,columns[begin*n],32u*n);
ZqB<Zq5::MTE2_V>();
for(Zq2 row=begin;row<begin+32u&&row+1u<n;++row){
auto column=kernel.factors_[(row-begin)*n];
const Zq2 start=(row+1u)/8u*8u,count=n-start;
if(start<=row){Duplicate(column[start],0.0f,row-start+1u);Zq1<PIPE_V>();}
ZqB<Zq5::S_V>();
for(Zq2 col=0;col<width;++col){
const float value=kernel.solveY_[col*p].GetValue(row);
Axpy(kernel.solveY_[col*p+start],column[start],value,int32_t(count));
}
ZqB<Zq5::V_S>();
}
}

}

template<bool UseCachedInverse, class Kernel>
__aicore__ static inline void BackwardColumnGm256(
    Kernel& kernel, GlobalTensor<float> columns, Zq3 base, Zq2 width)
{
constexpr Zq2 n=256u,p=256u;
for(int32_t begin=224;begin>=0;begin-=32){
ZqB<Zq5::V_MTE2>();
DataCopy(kernel.factors_,columns[Zq2(begin)*n],32u*n);
ZqB<Zq5::MTE2_V>();
ZqB<Zq5::MTE2_S>();
for(int32_t row=begin+31;row>=begin;--row){
const float diagonal=-kernel.factors_.GetValue((Zq2(row)-Zq2(begin))*n+Zq2(row));
const bool safe=kernel.AbsF(diagonal)>=1.0e-30f&&kernel.AbsF(diagonal)<=1.0e30f;
float inverse;
if constexpr (UseCachedInverse) {
    inverse=kernel.chainCachesReady_?kernel.output_[1024u].GetValue(Zq2(row)):kernel.w_[kernel.OffInv(base)+Zq2(row)];
} else {
    inverse=kernel.w_[kernel.OffInv(base)+Zq2(row)];
}
for(Zq2 col=0;col<width;++col){
auto y=kernel.solveY_[col*p];const float value=y.GetValue(Zq2(row));
y.SetValue(Zq2(row),diagonal==0.0f?0.0f:
(safe?square_math::RoundMul(value,inverse):square_math::Divide(value,diagonal)));
}
if(row>0){
ZqB<Zq5::S_V>();
for(Zq2 col=0;col<width;++col){
const float answer=kernel.solveY_[col*p].GetValue(Zq2(row));
Axpy(kernel.solveY_[col*p],kernel.factors_[(Zq2(row)-Zq2(begin))*n],answer,row);
}
ZqB<Zq5::V_S>();
}
}
}
}

};
