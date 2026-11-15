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
struct ProducerResidualOperations {
template<class Kernel>
__aicore__ static inline void ProducerLazySmallLu(Kernel &kernel, Zq3 base)
{
constexpr Zq2 n=64u,pitch=72u,p=64u;
auto map=kernel.residualTile_.template Zq6<Zq2>();
auto logical=kernel.solveY_;
for(Zq2 i=0;i<n;++i)map.SetValue(i,i*4u);
Zq3 active=~Zq3(0u);
for(Zq2 j=0;j<n;++j){
const Zq2 pick=kernel.FusedLogicalPivot(kernel.factors_[j*pitch],map,logical,j);
kernel.w_[kernel.OffP(base)+j]=float(int32_t(pick));
const Zq2 old=map.GetValue(j),selected=map.GetValue(pick);
const float diagonal=logical.GetValue(pick);
map.SetValue(j,selected);map.SetValue(pick,old);
const Zq2 physical=selected/4u;active&=~(Zq3(1u)<<physical);
if(diagonal==0.0f){kernel.w_[kernel.OffInv(base)+j]=0.0f;continue;}
const bool safe=kernel.AbsF(diagonal)>=1.0e-30f&&kernel.AbsF(diagonal)<=1.0e30f;
const float inverse=safe?kernel.SeedInverse(diagonal):0.0f;kernel.w_[kernel.OffInv(base)+j]=inverse;
if(j+1u==n)continue;
Zq3 valid[2]={active,0u};
if(safe){
ZqB<Zq5::S_V>();
Muls(kernel.factors_[j*pitch],kernel.factors_[j*pitch],inverse,valid,Zq4(1),UnaryRepeatParams{1,1,8,8});ResetMask();
}else{
for(Zq2 i=j+1u;i<n;++i){const Zq2 row=map.GetValue(i)/4u;
kernel.factors_.SetValue(j*pitch+row,square_math::Divide(kernel.factors_.GetValue(j*pitch+row),diagonal));}
}
auto columnOffsets=kernel.vr_[10u*p].template Zq6<Zq2>();
ZqB<Zq5::S_V>();
Gather(kernel.vr_[8u*p],kernel.factors_,columnOffsets,physical*4u,n);Zq1<PIPE_V>();
Muls(kernel.vr_[8u*p],kernel.vr_[8u*p],-1.0f,n);Zq1<PIPE_V>();
Brcb(kernel.block_,kernel.vr_[8u*p],Zq4((n+7u)/8u),Zq9{1,8});Zq1<PIPE_V>();
MulAddDst<float,float>(kernel.factors_[(j+1u)*pitch],kernel.factors_[j*pitch],kernel.block_[(j+1u)*8u],valid,Zq4(n-j-1u),Zq0{1,1,0,Zq4(pitch/8u),0,1});
ResetMask();ZqB<Zq5::V_S>();
}


auto savedMap=kernel.rows_.template Zq6<int32_t>();
ZqB<Zq5::S_V>();
Adds<int32_t>(savedMap,map.template Zq6<int32_t>(),int32_t(0),Zq2(64));Zq1<PIPE_V>();
for(Zq2 c=0;c<8u;++c)
Adds<int32_t>(map[c*64u].template Zq6<int32_t>(),savedMap,int32_t(c*288u),Zq2(64));
Zq1<PIPE_V>();
for(Zq2 col=0;col<64u;col+=8u){
Gather(kernel.block_,kernel.factors_,map,col*288u,512u);Zq1<PIPE_V>();
Adds(kernel.factors_[col*64u],kernel.block_,0.0f,512u);Zq1<PIPE_V>();
}
Adds<int32_t>(map.template Zq6<int32_t>(),savedMap,int32_t(0),Zq2(64));Zq1<PIPE_V>();
ZqB<Zq5::V_S>();
}

template<class Kernel>
__aicore__ static inline void ProducerPrefetchResidual16(Kernel &kernel, Zq2 row)
{
const Zq2 n=kernel.N(),count=n-row<16u?n-row:16u;
auto input=kernel.ProducerResidualInput16();


if(count<16u){Duplicate(input[count*256u],0.0f,(16u-count)*256u);Zq1<PIPE_V>();}
ZqB<Zq5::V_MTE2>();
GlobalTensor<float>original;original.SetGlobalBuffer(kernel.a_+(Zq3(kernel.currentBatch_)*n+row)*n,Zq3(count)*n);
DataCopyPad(input,original,Zq8{uint16_t(count),n*4u,0u,(256u-n)/8u,0u},DataCopyPadExtParams<float>{false,0u,0u,0.0f});
SetFlag<Zq5::MTE2_V>(0);
}

template<class Kernel>
__aicore__ static inline void ProducerResidualRowMajor16(Kernel &kernel, Zq3 base,Zq2 row,Zq2 slot)
{
constexpr Zq2 p=4096u;
constexpr Zq2 q=4128u;
auto a=kernel.ProducerResidualInput16(),work=kernel.ResidualWork();
Zq7<float>ah=work[2u*q],al=work[4u*q],u=work;
Zq7<float>hi=work[6u*q],lo=kernel.ProducerResidualLow16(),t=work;
SetMaskCount();ZqA<uint16_t,MaskMode::COUNTER>(0u,2u*p);
And<uint16_t,false>(ah.template Zq6<uint16_t>(),a.template Zq6<uint16_t>(),
kernel.vr_[kernel.ResidualMaskPlane()*kernel.VectorPitch()].template Zq6<uint16_t>(),Zq3(0),Zq4(1),Zq0{1,1,1,8,8,0});
Zq1<PIPE_V>();ZqA<float,MaskMode::COUNTER>(0u,p);
Sub<float,false>(al,a,ah,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Mul<float,false>(hi,a,work[q],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Mul<float,false>(lo,ah,work[3u*q],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
if(row+16u<kernel.N()){kernel.ProducerPrefetchResidual16(row+16u);
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,p);}
Sub<float,false>(lo,lo,hi,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Mul<float,false>(t,ah,work[5u*q],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Mul<float,false>(t,al,work[3u*q],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Mul<float,false>(t,al,work[5u*q],Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
Add<float,false>(lo,lo,t,Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});Zq1<PIPE_V>();
ResetMask();SetMaskNorm();


MergeStridedResidualFirstHalves<16, 32, 64>(hi, lo, ah, al, u);
MergeStridedResidualHalves<64u, 64, 16, 32>(hi, ah, lo, al, u);
MergeStridedResidualHalves<32u, 32, 16, 32>(ah, hi, lo, al, u);
MergeStridedResidualHalves<16u, 16, 16, 32>(hi, ah, lo, al, u);
MergeStridedResidualHalves<8u, 8, 16, 32>(ah, hi, lo, al, u);

Gather(work[6u*q],work[2u*q],kernel.ProducerResidualOffsets16(),0u,128u);
Gather(work,lo,kernel.ProducerResidualOffsets16(),0u,128u);Zq1<PIPE_V>();
hi=work[6u*q];lo=work;ah=work[2u*q];al=work[4u*q];u=kernel.ProducerResidualLow16();
SetMaskCount();
MergeCounterResidualHalves<64u>(ah, hi, lo, al, u, 64u);
MergeCounterResidualHalves<32u>(hi, ah, lo, al, u, 32u);
MergeCounterResidualHalves<16u>(ah, hi, lo, al, u, 16u);
ResetMask();SetMaskNorm();
const Zq2 count=kernel.N()-row<16u?kernel.N()-row:16u;
auto hh=work[2u*q],bb=kernel.output_[512u+row];
auto neg=kernel.ProducerResidualLow16(),ss=work[4u*q],vv=work[6u*q],tt=kernel.output_[256u+row];
WriteCompensatedResidual(tt, bb, hh, lo, neg, ss, vv, tt, count);
}
};
