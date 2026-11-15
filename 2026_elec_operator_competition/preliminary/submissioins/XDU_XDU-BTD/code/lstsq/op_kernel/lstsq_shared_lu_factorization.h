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
struct LuFactorizationOperations {
template<class Kernel>
__aicore__ static inline void DecomposeSquare(Kernel &kernel, Zq3 base)
{
{kernel.DecomposeLocal(base);return;}


}

template<class Kernel>
__aicore__ static inline Zq2 FusedLogicalPivot(Kernel &kernel, Zq7<float>column,
Zq7<Zq2>map,Zq7<float>logical,Zq2 first)
{
const Zq2 n=kernel.N(),p=kernel.VectorPitch();
ZqB<Zq5::S_V>();Gather(logical,column,map,0u,n);
if(n-first<=8u){
ZqB<Zq5::V_S>();
float best=kernel.AbsF(logical.GetValue(first));Zq2 pick=first;
for(Zq2 i=first+1u;i<n;++i){const float value=kernel.AbsF(logical.GetValue(i));if(value>best){best=value;pick=i;}}
return pick;
}


Zq1<PIPE_V>();Abs(kernel.vr_,logical,n);Zq1<PIPE_V>();
if(first){Duplicate(kernel.vr_,-1.0f,first);Zq1<PIPE_V>();}
ReduceMax(kernel.vr_[p],kernel.vr_,kernel.vr_[2u*p],int32_t(n),true);ZqB<Zq5::V_S>();
float best=kernel.AbsF(logical.GetValue(first));
if(best!=best)return first;
const float maximum=kernel.vr_[p].GetValue(0u);const Zq2 pick=kernel.vr_[p].template Zq6<Zq2>().GetValue(1u);
if(pick>=first&&pick<n&&maximum==maximum)return pick;
Zq2 fallback=first;
for(Zq2 i=first+1u;i<n;++i){const float value=kernel.AbsF(logical.GetValue(i));if(value>best){best=value;fallback=i;}}
return fallback;
}

__aicore__ static inline float Magnitude(float value) { return value < 0.0f ? -value : value; }

__aicore__ static inline void FactorScalarSquare(
    __gm__ float* lu, __gm__ float* piv, __gm__ float* inverse, Zq2 n)
{
for(Zq2 j=0;j<n;++j){
Zq2 pick=j;
float best=Magnitude(lu[Zq3(j)*n+j]);
for(Zq2 i=j+1;i<n;++i){
const float value=Magnitude(lu[Zq3(i)*n+j]);
if(value>best){best=value;pick=i;}
}
piv[j]=static_cast<float>(static_cast<int32_t>(pick));
if(pick!=j){
__gm__ float*rowA=lu+Zq3(j)*n;
__gm__ float*rowB=lu+Zq3(pick)*n;
for(Zq2 c=0;c<n;++c){
const float temp=rowA[c];rowA[c]=rowB[c];rowB[c]=temp;
}
}
const float diagonal=lu[Zq3(j)*n+j];
if(diagonal==0.0f){inverse[j]=0.0f;continue;}
const float magnitude=Magnitude(diagonal);
const bool reciprocalSafe=magnitude>=1.0e-30f&&magnitude<=1.0e30f;
const float inverseDiagonal=reciprocalSafe?square_math::Divide(1.0f,diagonal):0.0f;
inverse[j]=inverseDiagonal;
for(Zq2 i=j+1;i<n;++i){
const float multiplier=reciprocalSafe?
square_math::RoundMul(lu[Zq3(i)*n+j],inverseDiagonal):
square_math::Divide(lu[Zq3(i)*n+j],diagonal);
lu[Zq3(i)*n+j]=multiplier;
__gm__ float*rowI=lu+Zq3(i)*n;
const __gm__ float*rowJ=lu+Zq3(j)*n;
for(Zq2 c=j+1;c<n;++c)rowI[c]=rowI[c]-multiplier*rowJ[c];
}
}
}

};
