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
struct TriangularSolveOperations {
template<class Kernel>
__aicore__ static inline void SolveInPlaceWithFactors(Kernel &kernel, Zq3 base,__gm__ float*rhs,Zq2 width)
{
{kernel.ColumnTriangularAll(base,rhs,rhs,width);return;}


}

template<class Kernel>
__aicore__ static inline void VectorTriangularColumn(Kernel &kernel, Zq3 base,const __gm__ float*in,Zq2 inWidth,
__gm__ float*out,Zq2 outWidth)
{
const Zq2 n=kernel.N(),p=kernel.VectorPitch();
Zq7<float>x=kernel.vr_[p];
ZqB<Zq5::S_V>();
Duplicate(x,0.0f,p);
for(Zq2 i=0;i<n;++i)kernel.solveY_.SetValue(i,in[Zq3(i)*inWidth]);
const __gm__ float*piv=kernel.w_+kernel.OffP(base);
for(Zq2 j=0;j<n;++j){
const Zq2 pick=Zq2(int32_t(piv[j]));
if(pick!=j){
const float v=kernel.solveY_.GetValue(j);
kernel.solveY_.SetValue(j,kernel.solveY_.GetValue(pick));kernel.solveY_.SetValue(pick,v);
}
}
for(Zq2 i=0;i<n;++i){
const float value=square_math::RoundSub(kernel.solveY_.GetValue(i),kernel.TriangularDot(base,i));
kernel.solveY_.SetValue(i,value);x.SetValue(i,value);
}
ZqB<Zq5::S_V>();
Duplicate(x,0.0f,p);
for(int32_t i=int32_t(n)-1;i>=0;--i){
const float value=square_math::RoundSub(kernel.solveY_.GetValue(i),kernel.TriangularDot(base,Zq2(i)));
const float diagonal=kernel.Factor(base,Zq2(i),Zq2(i));
const float answer=diagonal==0.0f?0.0f:
(kernel.AbsF(diagonal)>=1.0e-30f&&kernel.AbsF(diagonal)<=1.0e30f?
square_math::RoundMul(value,kernel.w_[kernel.OffInv(base)+Zq2(i)]):square_math::Divide(value,diagonal));
x.SetValue(i,answer);
}
for(Zq2 i=0;i<n;++i)out[Zq3(i)*outWidth]=x.GetValue(i);
}

template<class Kernel>
__aicore__ static inline void SolveScalarSquare(
    Kernel& kernel, Zq3 base, const __gm__ float* lu, const __gm__ float* piv, __gm__ float* rhs, Zq2 n, Zq2 width)
{
for(Zq2 col=0;col<width;++col){
for(Zq2 j=0;j<n;++j){
const Zq2 p=static_cast<Zq2>(static_cast<int32_t>(piv[j]));
if(p==j)continue;
const float temp=rhs[Zq3(j)*width+col];
rhs[Zq3(j)*width+col]=rhs[Zq3(p)*width+col];
rhs[Zq3(p)*width+col]=temp;
}
for(Zq2 i=0;i<n;++i){
float value=rhs[Zq3(i)*width+col];
for(Zq2 c=0;c<i;++c)value-=kernel.Factor(base,Zq2(i),c)*rhs[Zq3(c)*width+col];
rhs[Zq3(i)*width+col]=value;
}
for(int32_t i=int32_t(n)-1;i>=0;--i){
float value=rhs[Zq3(i)*width+col];
for(Zq2 c=Zq2(i)+1;c<n;++c){
value-=kernel.Factor(base,Zq2(i),c)*rhs[Zq3(c)*width+col];
}
const float diagonal=kernel.Factor(base,Zq2(i),Zq2(i));
rhs[Zq3(i)*width+col]=diagonal==0.0f?0.0f:
(kernel.AbsF(diagonal)>=1.0e-30f&&kernel.AbsF(diagonal)<=1.0e30f?
square_math::RoundMul(value,kernel.w_[kernel.OffInv(base)+Zq2(i)]):square_math::Divide(value,diagonal));
}
}
}


template<bool Legacy, class Kernel>
__aicore__ static inline void ComputeBatchedDots(Kernel& kernel, Zq3 base, Zq2 row, Zq2 width)
{
    if constexpr (Legacy) {
        kernel.BatchedTriangularDots(row, width);
    } else {
        kernel.BatchedTriangularDots(base, row, width);
    }
}

template<class Kernel>
__aicore__ static inline void ApplyColumnPivots(Kernel& kernel, Zq3 base, Zq2 n, Zq2 p, Zq2 width)
{
for(Zq2 row=0;row<n;++row){
const Zq2 pick=Zq2(int32_t(kernel.w_[kernel.OffP(base)+row]));
if(pick!=row)for(Zq2 col=0;col<width;++col){
Zq7<float>y=kernel.solveY_[col*p];
const float value=y.GetValue(row);y.SetValue(row,y.GetValue(pick));y.SetValue(pick,value);
}
}
}

template<class Kernel>
__aicore__ static inline void UpdateColumnTriangularRow(Kernel& kernel,
    Zq2 row, Zq2 width, Zq2 p, Zq2 pitch)
{
const Zq2 start=(row+1u)/8u*8u,count=pitch-start;
const Zq2 head=row+1u-start,first=count<64u?count:64u;
for(Zq2 col=0;col<width;++col){
const float value=kernel.solveY_[col*p].GetValue(row);
if(head){
const Zq3 low=first==64u?~Zq3(0u):(Zq3(1u)<<first)-1u;
Zq3 mask[2]={low&(~Zq3(0u)<<head),0u};
Axpy<float,float>(kernel.solveY_[col*p+start],kernel.factors_[row*pitch+start],-value,mask,Zq4(1),UnaryRepeatParams{1,1,8,8});
ResetMask();
if(count>first)Axpy((kernel.solveY_[col*p+start])[first],(kernel.factors_[row*pitch+start])[first],-value,int32_t(count-first));
}else Axpy(kernel.solveY_[col*p+start],kernel.factors_[row*pitch+start],-value,int32_t(count));
}
}

template<bool Legacy, class Kernel>
__aicore__ static inline void VectorTriangularBatch(
    Kernel& kernel, Zq3 base, const __gm__ float* in, Zq2 inWidth,
    __gm__ float* out, Zq2 outWidth, Zq2 width, Zq2 n, Zq2 p)
{
ZqB<Zq5::S_V>();
Duplicate(kernel.vr_[p],0.0f,width*p);
for(Zq2 col=0;col<width;++col){
Zq7<float>y=kernel.solveY_[col*p];
for(Zq2 row=0;row<n;++row)y.SetValue(row,in[Zq3(row)*inWidth+col]);
}
const __gm__ float*piv=kernel.w_+kernel.OffP(base);
for(Zq2 j=0;j<n;++j){
const Zq2 pick=Zq2(int32_t(piv[j]));
if(pick!=j)for(Zq2 col=0;col<width;++col){
Zq7<float>y=kernel.solveY_[col*p];
const float v=y.GetValue(j);y.SetValue(j,y.GetValue(pick));y.SetValue(pick,v);
}
}
for(Zq2 row=0;row<n;++row){
ComputeBatchedDots<Legacy>(kernel, base, row, width);
for(Zq2 col=0;col<width;++col){
Zq7<float>y=kernel.solveY_[col*p];
const float value=square_math::RoundSub(y.GetValue(row),kernel.BatchedDotValue(col));
y.SetValue(row,value);kernel.vr_[(1u+col)*p].SetValue(row,value);
}
}
ZqB<Zq5::S_V>();
Duplicate(kernel.vr_[p],0.0f,width*p);
for(int32_t row=int32_t(n)-1;row>=0;--row){
ComputeBatchedDots<Legacy>(kernel, base, Zq2(row), width);
const float diagonal=kernel.Factor(base,Zq2(row),Zq2(row));
const bool safe=kernel.AbsF(diagonal)>=1.0e-30f&&kernel.AbsF(diagonal)<=1.0e30f;
const float inverse=kernel.w_[kernel.OffInv(base)+Zq2(row)];
for(Zq2 col=0;col<width;++col){
const float value=square_math::RoundSub(kernel.solveY_[col*p].GetValue(row),kernel.BatchedDotValue(col));
const float answer=diagonal==0.0f?0.0f:
(safe?square_math::RoundMul(value,inverse):square_math::Divide(value,diagonal));
kernel.vr_[(1u+col)*p].SetValue(row,answer);
}
}
for(Zq2 col=0;col<width;++col)for(Zq2 row=0;row<n;++row)
out[Zq3(row)*outWidth+col]=kernel.vr_[(1u+col)*p].GetValue(row);
}


template<class Kernel>
__aicore__ static inline float TriangularDot(Kernel& kernel, Zq3 base, Zq2 row,
    Zq2 n, Zq2 p, Zq2 factorPitch, bool local)
{
Zq7<float>a=kernel.vr_,x=kernel.vr_[p],hi=kernel.vr_[6u*p];
if(local){
ZqB<Zq5::S_V>();
Adds(a,kernel.factors_[row*factorPitch],0.0f,factorPitch);
}else{
const __gm__ float*lu=kernel.w_+kernel.OffLU(base)+Zq3(row)*n;
for(Zq2 c=0;c<n;++c)a.SetValue(c,lu[c]);
ZqB<Zq5::S_V>();
}
Zq1<PIPE_V>();
Mul(hi,a,x,p);
Zq1<PIPE_V>();
for(Zq2 width=p/2u;width>=8u;width/=2u){
Add(hi,hi,hi[width],width);
Zq1<PIPE_V>();
}
ZqB<Zq5::V_S>();
float sum=0.0f;
for(Zq2 c=0;c<8u;++c)sum=square_math::RoundAdd(sum,hi.GetValue(c));
return sum;
}


__aicore__ static inline void ReduceBatchedDots(Zq7<float> vr, Zq2 p, Zq2 width)
{
Zq1<PIPE_V>();
SetMaskCount();ZqA<float,MaskMode::COUNTER>(0u,p);
for(Zq2 col=0;col<width;++col)
Mul<float,false>(vr[(6u+col)*p],vr,vr[(1u+col)*p],
Zq3(0),Zq4(1),Zq0{1,1,1,8,8,8});
Zq1<PIPE_V>();
for(Zq2 count=p/2u;count>=8u;count/=2u){
ZqA<float,MaskMode::COUNTER>(0u,count);
for(Zq2 col=0;col<width;++col){
Zq7<float>sum=vr[(6u+col)*p];
Add<float,false>(sum,sum,sum[count],Zq3(0),Zq4(1),
Zq0{1,1,1,8,8,8});
}
Zq1<PIPE_V>();
}
ResetMask();SetMaskNorm();
ZqB<Zq5::V_S>();
}

__aicore__ static inline float PivotAbs(float value){return value<0.0f?-value:value;}
__aicore__ static inline Zq2 SelectColumnPivot(Zq7<float> column,Zq7<float> vr,Zq2 first,Zq2 n,Zq2 p)
{

float best=PivotAbs(column.GetValue(first));
if(n-first<=((n==50u||n==64u)?8u:32u)||best!=best){
Zq2 pick=first;
for(Zq2 i=first+1u;i<n;++i){
const float value=PivotAbs(column.GetValue(i));
if(value>best){best=value;pick=i;}
}
return pick;
}

ZqB<Zq5::S_V>();
Abs(vr,column,n);
Zq1<PIPE_V>();

if(first){Duplicate(vr,-1.0f,first);Zq1<PIPE_V>();}
ReduceMax(vr[p],vr,vr[2u*p],int32_t(n),true);
ZqB<Zq5::V_S>();
const float maximum=vr[p].GetValue(0u);
const Zq2 pick=vr[p].template Zq6<Zq2>().GetValue(1u);
if(pick>=first&&pick<n&&maximum==maximum)return pick;


Zq2 fallback=first;
for(Zq2 i=first+1u;i<n;++i){
const float value=PivotAbs(column.GetValue(i));
if(value>best){best=value;fallback=i;}
}
return fallback;
}


__aicore__ static inline void ApplyBackwardColumnRow(Zq7<float> solveY,Zq7<float> factors,
    int32_t row,Zq2 p,Zq2 pitch,Zq2 width,float diagonal,float inverse,bool safe)
{
for(Zq2 col=0;col<width;++col){
Zq7<float>y=solveY[col*p];const float value=y.GetValue(row);
y.SetValue(row,diagonal==0.0f?0.0f:(safe?square_math::RoundMul(value,inverse):
square_math::Divide(value,diagonal)));
}
if(row>0){
ZqB<Zq5::S_V>();
for(Zq2 col=0;col<width;++col){
const float answer=solveY[col*p].GetValue(row);
Axpy(solveY[col*p],factors[Zq2(row)*pitch],-answer,row);
}
ZqB<Zq5::V_S>();
}
}


template<class Kernel>
__aicore__ static inline Zq2 SelectLocalPivot(Kernel& kernel,Zq3 base,Zq2 j,Zq2 n)
{
Zq2 pick=j;float best=kernel.AbsF(kernel.Factor(base,j,j));
for(Zq2 i=j+1;i<n;++i){
const float value=kernel.AbsF(kernel.Factor(base,i,j));
if(value>best){best=value;pick=i;}
}
return pick;
}


};
