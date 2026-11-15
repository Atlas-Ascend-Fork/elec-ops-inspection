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

__aicore__ inline void KernelLegacy::EmitSquareDma(Zq3 base,Zq2 batch)
{
WorkspaceOperations::CopySquareOutput(output_, w_+OffX(base), x_, r_, batch, tiling_.n, tiling_.k, tiling_.residualElements);
}

__aicore__ inline void KernelLegacy::Emit(Zq3 base,Zq2 batch,Zq2 col,bool withResidual)
{
const Zq2 m=tiling_.m,n=tiling_.n,k=tiling_.k;
WorkspaceOperations::EmitColumn(*this,base,batch,col,withResidual,m,n,k);
}

__aicore__ inline void KernelLegacy::Process()
{
const Zq2 m=tiling_.m,n=tiling_.n,k=tiling_.k;
if(m==0||n==0||k==0)return;
const Zq3 per=PerBatch();


const auto outputSchedule = WorkspaceOperations::CheckOutputSchedule(x_, r_, m, n, k,
    tiling_.batches, tiling_.residualElements, dmaOutputs_);
const bool serial = outputSchedule.serial;
const Zq3 rCount = outputSchedule.residualCount;
if(serial&&GetBlockIdx()!=0u)return;


if(!dmaOutputs_&&m==n&&r_!=nullptr&&GetBlockIdx()==0u)
for(Zq3 i=0;i<rCount;++i)r_[i]=0.0f;
const Zq2 stride=serial?1u:GetBlockNum();
for(Zq2 batch=serial?0u:GetBlockIdx();batch<tiling_.batches;
batch+=stride){
const Zq3 base=Zq3(batch)*per;
Load(base,batch);
if (m == n) ProcessSquare(base, batch);
else ProcessRectangular(base, batch);
}
}

__aicore__ inline void KernelLegacy::ProcessSquare(Zq3 base, Zq2 batch)
{
const Zq2 k=tiling_.k;
DecomposeSquare(base);
if(vectorSolve_&&k<=4u)
VectorTriangularBatch(base,w_+OffB(base),k,w_+OffX(base),k,k);
else for(Zq2 col=0;col<k;++col)BackSubstitute(base,col);
Refine(base);
if(dmaOutputs_)EmitSquareDma(base,batch);
else for(Zq2 col=0;col<k;++col)Emit(base,batch,col,false);
}

__aicore__ inline void KernelLegacy::ProcessRectangular(Zq3 base, Zq2 batch)
{
    RectangularSolveOperations::Process(*this, base, batch, tiling_.k);
}


__aicore__ inline void KernelLegacy::Init(GM_ADDR a,GM_ADDR b,GM_ADDR x,GM_ADDR res,
GM_ADDR workspace,LinalgLstsqTilingData t)
{
tiling_=t;
dmaOutputs_=t.m==t.n&&Zq3(t.n)*t.k<=1024u;
pipe_.InitBuffer(outputBuf_,dmaOutputs_?((t.n*t.k+7u)/8u*8u+(t.k+7u)/8u*8u)*sizeof(float):32u);
output_=outputBuf_.Get<float>();
localFactors_=t.m==t.n&&t.n>=32u&&t.n<=200u;
factorPitch_=(t.n+7u)/8u*8u;
pipe_.InitBuffer(factorBuf_,localFactors_?t.n*factorPitch_*sizeof(float):32u);
factors_=factorBuf_.Get<float>();
pipe_.InitBuffer(rowBuf_,208u*sizeof(float));
rows_=rowBuf_.Get<float>();
pipe_.InitBuffer(blockBuf_,t.m==t.n&&t.n==256u?80u*256u*sizeof(float):32u);
block_=blockBuf_.Get<float>();
vectorResidual_=t.m==t.n;
vectorPitch_=8u;
while(vectorPitch_<t.n)vectorPitch_*=2u;
pipe_.InitBuffer(residualBuf_,vectorResidual_?13u*vectorPitch_*sizeof(float):32u);
vr_=residualBuf_.Get<float>();
vectorSolve_=t.m==t.n&&t.n>=32u&&t.n<=200u;
pipe_.InitBuffer(solveBuf_,vectorSolve_?(t.k<=4u?t.k:1u)*vectorPitch_*sizeof(float):32u);
solveY_=solveBuf_.Get<float>();
if(vectorResidual_){
auto mask=vr_[11u*vectorPitch_].Zq6<Zq2>();
Duplicate(mask,Zq2(0xfffff000u),vectorPitch_);
Duplicate(vr_,0.0f,vectorPitch_);
Duplicate(vr_[12u*vectorPitch_],0.0f,vectorPitch_);
Duplicate(vr_[vectorPitch_],0.0f,vectorPitch_);
}
a_=reinterpret_cast<__gm__ float*>(a);
b_=reinterpret_cast<__gm__ float*>(b);
x_=reinterpret_cast<__gm__ float*>(x);
r_=reinterpret_cast<__gm__ float*>(res);
const Zq3 user=reinterpret_cast<Zq3>(AscendC::GetUserWorkspace(workspace));
w_=reinterpret_cast<__gm__ float*>((user+63u)&~Zq3(63u));
}

__aicore__ inline void KernelLegacy::Load(Zq3 base,Zq2 batch)
{
const Zq2 m=tiling_.m,n=tiling_.n,k=tiling_.k;
currentBatch_=batch;
const Zq3 aBatch=Zq3(batch)*m*n;
const Zq3 bBatch=Zq3(batch)*m*k;
const Zq3 dstA=OffA(base),dstL=OffLU(base),dstB=OffB(base);
if(localFactors_){

GlobalTensor<float>input;input.SetGlobalBuffer(a_+aBatch,Zq3(n)*n);
DataCopyPad(factors_,input,Zq8{uint16_t(n),n*4u,0u,0u,0u},
DataCopyPadExtParams<float>{true,0u,Zq4(factorPitch_-n),0.0f});
ZqB<Zq5::MTE2_S>();
}else if(m==n){
for(Zq3 i=0;i<Zq3(m)*n;++i)w_[dstL+i]=a_[aBatch+i];
}else{
for(Zq3 i=0;i<Zq3(m)*n;++i)w_[dstL+i]=w_[dstA+i]=a_[aBatch+i];
}
for(Zq3 i=0;i<Zq3(m)*k;++i)w_[dstB+i]=b_[bBatch+i];
}
