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

struct RectangularSolveOperations {
__aicore__ static inline void ApplyHouseholder(
    const __gm__ float* lu, __gm__ float* dst, Zq2 m, Zq2 n, Zq2 width, Zq2 j, Zq2 col, float tau)
{
    float dot=dst[Zq3(j)*width+col];
    for (Zq2 row=j+1u;row<m;++row) {
        dot+=lu[Zq3(row)*n+j]*dst[Zq3(row)*width+col];
    }
    const float coefficient=tau*dot;
    dst[Zq3(j)*width+col]=dst[Zq3(j)*width+col]-coefficient;
    for (Zq2 row=j+1u;row<m;++row) {
        dst[Zq3(row)*width+col]=dst[Zq3(row)*width+col]-coefficient*lu[Zq3(row)*n+j];
    }
}

__aicore__ static inline void Qr(__gm__ float* lu, __gm__ float* rhs, Zq2 m, Zq2 n, Zq2 k)
{
    for (Zq2 j=0;j<n;++j) {
        float largest=0.0f;
        for (Zq2 row=j;row<m;++row) {
            const float x=lu[Zq3(row)*n+j];
            const float value=x<0.0f?-x:x;
            if (value>largest) largest=value;
        }
        if (largest==0.0f) continue;
        float sum=0.0f;
        for (Zq2 row=j;row<m;++row) {
            const float scaled=Divide(lu[Zq3(row)*n+j],largest);
            sum+=scaled*scaled;
        }
        const float norm=largest*sqrt(sum);
        const float alpha=lu[Zq3(j)*n+j];
        const float beta=alpha>=0.0f?-norm:norm;
        const float denominator=alpha-beta;
        for (Zq2 row=j+1u;row<m;++row) {
            lu[Zq3(row)*n+j]=Divide(lu[Zq3(row)*n+j],denominator);
        }
        const float tau=Divide(beta-alpha,beta);
        for (Zq2 col=j+1u;col<n;++col) ApplyHouseholder(lu,lu,m,n,n,j,col,tau);
        for (Zq2 col=0;col<k;++col) ApplyHouseholder(lu,rhs,m,n,k,j,col,tau);
        lu[Zq3(j)*n+j]=beta;
        for (Zq2 row=j+1u;row<m;++row) lu[Zq3(row)*n+j]=0.0f;
    }
}

template<class Kernel>
__aicore__ static inline void Refine(Kernel& kernel, Zq3 base, Zq2 batch, Zq2 m, Zq2 n, Zq2 k)
{
const __gm__ float*a=kernel.w_+kernel.OffA(base);
const __gm__ float*originalB=kernel.b_+Zq3(batch)*m*k;
const __gm__ float*rr=kernel.w_+kernel.OffLU(base);
__gm__ float*hi=kernel.w_+kernel.OffR(base),*lo=kernel.w_+kernel.OffD(base);
for(Zq2 step=0;step<3u;++step){
for(Zq3 c=0;c<Zq3(n)*k;++c){hi[c]=0.0f;lo[c]=0.0f;}
for(Zq2 row=0;row<m;++row)for(Zq2 col=0;col<k;++col){
float rh=originalB[Zq3(row)*k+col],rl=0.0f;
for(Zq2 j=0;j<n;++j){
Pair p=Product(a[Zq3(row)*n+j],kernel.GetX(base,j,col));
Pair t=Sum(rh,-p.hi);
rl=RoundAdd(rl,RoundSub(t.lo,p.lo));rh=t.hi;
}
for(Zq2 j=0;j<n;++j){
const Zq3 index=Zq3(j)*k+col;
Pair p=Product(a[Zq3(row)*n+j],rh);
float tail=RoundAdd(p.lo,RoundMul(a[Zq3(row)*n+j],rl));
Pair t=Sum(hi[index],p.hi);
lo[index]=RoundAdd(lo[index],RoundAdd(t.lo,tail));hi[index]=t.hi;
}
}
for(Zq3 c=0;c<Zq3(n)*k;++c)hi[c]=RoundAdd(hi[c],lo[c]);
for(Zq2 col=0;col<k;++col){
for(Zq2 i=0;i<n;++i){
float v=hi[Zq3(i)*k+col];
for(Zq2 j=0;j<i;++j)v-=rr[Zq3(j)*n+i]*hi[Zq3(j)*k+col];
hi[Zq3(i)*k+col]=Divide(v,rr[Zq3(i)*n+i]);
}
for(int32_t i=int32_t(n)-1;i>=0;--i){
float v=hi[Zq3(i)*k+col];
for(Zq2 j=Zq2(i)+1;j<n;++j)v-=rr[Zq3(i)*n+j]*hi[Zq3(j)*k+col];
hi[Zq3(i)*k+col]=Divide(v,rr[Zq3(i)*n+i]);
}
for(Zq2 i=0;i<n;++i)kernel.SetX(base,i,col,RoundAdd(kernel.GetX(base,i,col),hi[Zq3(i)*k+col]));
}
}
}

template<class Kernel>
__aicore__ static inline void SolveRhs(Kernel& kernel, Zq3 base, Zq2 col, Zq2 n, Zq2 k)
{
const __gm__ float*lu=kernel.w_+kernel.OffLU(base);
for(int32_t i=int32_t(n)-1;i>=0;--i){
float value=kernel.w_[kernel.OffB(base)+Zq3(i)*k+col];
for(Zq2 c=Zq2(i)+1;c<n;++c){
value-=lu[Zq3(i)*n+c]*kernel.GetX(base,c,col);
}
const float diagonal=lu[Zq3(i)*n+i];
kernel.SetX(base,Zq2(i),col,diagonal==0.0f?0.0f:Divide(value,diagonal));
}
}


template<class Kernel>
__aicore__ static inline void Process(Kernel& kernel, Zq3 base, Zq2 batch, Zq2 k)
{
kernel.Qr(base);
for(Zq2 col=0;col<k;++col)kernel.TriangularSolveQr(base,col);
kernel.RefineRectangular(base,batch);
for(Zq2 col=0;col<k;++col)kernel.Emit(base,batch,col,true);
}

};
