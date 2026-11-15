/*
 * Copyright (c) 2026 zjyzjy2002.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "kernel_operator.h"
using namespace AscendC;
// A two-float expansion keeps tridiagonal rotations above float32 precision.
struct ExtendedFloat {
    float h,m,l;
    __aicore__ inline ExtendedFloat(float x=0):h(x),m(0),l(0){}
    __aicore__ inline ExtendedFloat(float a,float b,float c):h(a),m(b),l(c){}
    __aicore__ static inline void Sum(float a,float b,float &s,float &e){s=a+b;float q=s-a;e=(a-(s-q))+(b-q);}
    __aicore__ static inline void Product(float a,float b,float &p,float &e){
        p=a*b;float t=4097.0f*a,ah=t-(t-a),al=a-ah;t=4097.0f*b;float bh=t-(t-b),bl=b-bh;e=((ah*bh-p)+ah*bl+al*bh)+al*bl;
    }
    __aicore__ static inline ExtendedFloat Norm(float a,float b,float c){float s,e;Sum(a,b+c,s,e);return ExtendedFloat(s,e,0);}
    __aicore__ static inline ExtendedFloat QuickNorm(float a,float b){float s=a+b,e=b-(s-a);return ExtendedFloat(s,e,0);}
    __aicore__ inline ExtendedFloat Scale(float b)const{return QuickNorm(h*b,m*b);}
    __aicore__ inline ExtendedFloat operator-()const{return ExtendedFloat(-h,-m,-l);}
    __aicore__ inline ExtendedFloat operator+(const ExtendedFloat &b)const{float s,e;Sum(h,b.h,s,e);return QuickNorm(s,e+m+b.m);}
    __aicore__ inline ExtendedFloat operator-(const ExtendedFloat &b)const{return *this+(-b);}
    __aicore__ inline ExtendedFloat operator*(const ExtendedFloat &b)const{float p,e;Product(h,b.h,p,e);return QuickNorm(p,e+(h*b.m+m*b.h));}
    __aicore__ inline ExtendedFloat DivideBy(const ExtendedFloat &b) const {
        float high = b.h, low = b.m;
        if (high == 0.0f) {
            high = low;
            low = 0.0f;
        }
        // A zero divisor has no expansion quotient. Preserve the non-finite result.
        if (high == 0.0f) return ExtendedFloat(__builtin_nanf(""));
        float quotient = h / high, productHigh, productError;
        Product(high, quotient, productHigh, productError);
        ExtendedFloat product = QuickNorm(productHigh, productError + low * quotient);
        float residual = (h - product.h) + (m - product.m);
        return QuickNorm(quotient, residual / high);
    }
    __aicore__ inline ExtendedFloat operator/(const ExtendedFloat &b) const { return DivideBy(b); }
    __aicore__ inline ExtendedFloat AddFull(const ExtendedFloat &b)const{return *this+b;}
    __aicore__ inline ExtendedFloat MulFull(const ExtendedFloat &b)const{return *this*b;}
    __aicore__ inline float Float()const{return h+m;}
    __aicore__ inline bool Negative()const{return h<0 || (h==0 && m<0);}
    __aicore__ inline bool Zero()const{return h==0 && m==0;}
};

// The probe tracks only the sign of a tridiagonal eigenvector basis.
class TridiagonalBasis {
    LocalTensor<float> buf;
    uint32_t n;
    __aicore__ inline ExtendedFloat Read(uint32_t i){i*=2;return ExtendedFloat(buf.GetValue(i),buf.GetValue(i+1),0);}
    __aicore__ inline void Write(uint32_t i,ExtendedFloat x){i*=2;buf.SetValue(i,x.h);buf.SetValue(i+1,x.m);}
    __aicore__ inline ExtendedFloat D(uint32_t i){return Read(n*n+3*n+i);}
    __aicore__ inline ExtendedFloat E(uint32_t i){return Read(n*n+4*n+i);}
    __aicore__ inline ExtendedFloat Q(uint32_t i){return Read(n*n+i);}
    __aicore__ inline void DS(uint32_t i,ExtendedFloat x){Write(n*n+3*n+i,x);}
    __aicore__ inline void ES(uint32_t i,ExtendedFloat x){Write(n*n+4*n+i,x);}
    __aicore__ inline void QS(uint32_t i,ExtendedFloat x){Write(n*n+i,x);}
    __aicore__ inline ExtendedFloat AbsW(ExtendedFloat x){return x.Negative()?-x:x;}
    __aicore__ inline bool Less(ExtendedFloat a,ExtendedFloat b){
        if(a.h!=b.h)return a.h<b.h;if(a.m!=b.m)return a.m<b.m;return a.l<b.l;
    }
    __aicore__ inline ExtendedFloat RootW(ExtendedFloat x){
        if(x.Zero() || x.Negative())return ExtendedFloat(0);
        if(x.h==1.0f)return ExtendedFloat(1.0f,0.5f*x.m,0.0f);
        union Bits{float f;unsigned u;}b;b.f=x.h;float a;
        if(x.h>=1e-30f && x.h<=1e30f){
            b.u=0x5f375a86u-(b.u>>1);float reciprocal=b.f;
            for(uint32_t i=0;i<3;++i)reciprocal=reciprocal*(1.5f-((0.5f*x.h)*reciprocal)*reciprocal);
            a=x.h*reciprocal;a=0.5f*(a+x.h/a);
        }else{b.u=(b.u>>1)+0x1fc00000u;a=b.f;for(uint32_t i=0;i<3;++i)a=0.5f*(a+x.h/a);}
        ExtendedFloat y(a);
        for(uint32_t i=0;i<1;++i){ExtendedFloat r=x.AddFull(-y.MulFull(y));y=y+r/(y.Scale(2));}
        return y;
    }
    __aicore__ inline void Rotate(uint32_t i,ExtendedFloat c,ExtendedFloat s){
        // A float32 probe is sufficient to identify the final sign; the
        // tridiagonal and rotations retain expansion precision.
        uint32_t pos=2*(n*n+i);
        float a=buf.GetValue(pos),b=buf.GetValue(pos+2);
        buf.SetValue(pos,c.h*a+s.h*b);buf.SetValue(pos+2,c.h*b-s.h*a);
    }
    __aicore__ inline void RotateCarry(uint32_t i,ExtendedFloat c,ExtendedFloat s,float &carry,bool reverse){
        uint32_t pos=2*(n*n+i);
        float a=reverse?carry:buf.GetValue(pos),b=reverse?buf.GetValue(pos+2):carry;
        float left=c.h*a+s.h*b,right=c.h*b-s.h*a;
        if(reverse){buf.SetValue(pos,left);carry=right;}
        else{buf.SetValue(pos+2,right);carry=left;}
    }
    __aicore__ inline void GivensW(ExtendedFloat f,ExtendedFloat g,ExtendedFloat &c,ExtendedFloat &s,ExtendedFloat &r){
        if(g.Zero()){c=ExtendedFloat(1);s=ExtendedFloat(0);r=f;return;}
        if(f.Zero()){c=ExtendedFloat(0);s=ExtendedFloat(g.Negative()?-1.0f:1.0f);r=AbsW(g);return;}
        r=RootW(f*f+g*g);
        if (r.h == 0.0f) {
            // Squaring tiny nonzero components can underflow. Rotate at unit scale.
            ExtendedFloat scale = Less(AbsW(f), AbsW(g)) ? AbsW(g) : AbsW(f);
            if (scale.h == 0.0f) { c=ExtendedFloat(1); s=ExtendedFloat(0); r=f; return; }
            ExtendedFloat fn = f / scale, gn = g / scale;
            ExtendedFloat unitNorm = RootW(fn*fn + gn*gn);
            if (f.Negative()) unitNorm = -unitNorm;
            c = fn.DivideBy(unitNorm); s = gn.DivideBy(unitNorm); r = scale * unitNorm;
            return;
        }
        if(f.Negative())r=-r;
        c = f.DivideBy(r); s = g.DivideBy(r);
    }
    __aicore__ inline void Eigen2W(ExtendedFloat a,ExtendedFloat b,ExtendedFloat c,ExtendedFloat &co,ExtendedFloat &si){
        ExtendedFloat sm=a+c,df=a-c,tb=b.Scale(2),ad=AbsW(df),ab=AbsW(tb),rt;
        if(ad.h != 0.0f && Less(ab,ad)){ExtendedFloat t=ab.DivideBy(ad);rt=ad*RootW(ExtendedFloat(1)+t*t);}
        else if(ab.h != 0.0f && Less(ad,ab)){ExtendedFloat t=ad.DivideBy(ab);rt=ab*RootW(ExtendedFloat(1)+t*t);}
        else rt=ab*RootW(ExtendedFloat(2));
        bool s1=sm.Negative(),s2=df.Negative();
        ExtendedFloat cs=s2?df-rt:df+rt;
        if(Less(ab,AbsW(cs))){ExtendedFloat t=-tb/cs;si=ExtendedFloat(1)/RootW(ExtendedFloat(1)+t*t);co=t*si;}
        else if(tb.h == 0.0f){co=ExtendedFloat(1);si=ExtendedFloat(0);}
        else{ExtendedFloat t=(-cs).DivideBy(tb);co=ExtendedFloat(1)/RootW(ExtendedFloat(1)+t*t);si=t*co;}
        if(s1==s2){ExtendedFloat t=co;co=-si;si=t;}
    }
    __aicore__ inline bool Deflated(uint32_t i){
        ExtendedFloat a=D(i),b=D(i+1),off=E(i);
        float ah=a.h<0?-a.h:a.h,bh=b.h<0?-b.h:b.h,eh=off.h<0?-off.h:off.h;
        // AM-GM gives a cheap upper bound on the exact split threshold.
        if(eh>2.220446049250313e-16f*(ah+bh))return false;
        ExtendedFloat eps(1.1102230246251565e-16f);
        return !Less((eps*eps*AbsW(a))*AbsW(b),off*off);
    }
    __aicore__ inline void ChaseBulge(int l, uint32_t m, bool reverse) {
        uint32_t j=reverse?l-1:l;ExtendedFloat p=D(l),g=(D(reverse?l-1:l+1)-p)/(E(j).Scale(2));
        ExtendedFloat r=RootW(g*g+ExtendedFloat(1));g=D(m)-p+E(j)/(g+(g.Negative()?-r:r));
        ExtendedFloat s(1),c(1);p=ExtendedFloat(0);
        ExtendedFloat currentDiag=D(m);float probeCarry=buf.GetValue(2*(n*n+m));
        if(!reverse){
            for(int i=int(m)-1;i>=l;--i){ExtendedFloat nextDiag=D(i),off=E(i),f=s*off,b=c*off;GivensW(g,f,c,s,r);if(i!=int(m)-1)ES(i+1,r);g=currentDiag-p;r=(nextDiag-g)*s+c.Scale(2)*b;p=s*r;DS(i+1,g+p);g=c*r-b;RotateCarry(i,c,-s,probeCarry,false);currentDiag=nextDiag;}
            buf.SetValue(2*(n*n+l),probeCarry);DS(l,currentDiag-p);ES(l,g);
        }else{
            for(int i=m;i<l;++i){ExtendedFloat nextDiag=D(i+1),off=E(i),f=s*off,b=c*off;GivensW(g,f,c,s,r);if(i!=int(m))ES(i-1,r);g=currentDiag-p;r=(nextDiag-g)*s+c.Scale(2)*b;p=s*r;DS(i,g+p);g=c*r-b;RotateCarry(i,c,s,probeCarry,true);currentDiag=nextDiag;}
            buf.SetValue(2*(n*n+l),probeCarry);DS(l,currentDiag-p);ES(l-1,g);
        }
    }
    __aicore__ inline uint32_t FindUnreducedEnd(uint32_t start, ExtendedFloat eps) {
        uint32_t m=start;
        for(;m+1<n;++m){
            float a=D(m).h,b=D(m+1).h,e=E(m).h;a=a<0?-a:a;b=b<0?-b:b;e=e<0?-e:e;
            if(e>2.220446049250313e-16f*(a+b))continue;
            if(!Less(eps*RootW(AbsW(D(m)))*RootW(AbsW(D(m+1))),AbsW(E(m))))break;
        }
        return m;
    }
    __aicore__ __attribute__((noinline)) float QR(){
        // A decoupled component containing over 80% of the probe's squared
        // norm stays dominant under all remaining orthogonal rotations.
        float probeNorm=0;
        for(uint32_t i=0;i<n;++i){float x=Q(i).h;probeNorm+=x*x;}
        ExtendedFloat eps(1.1102230246251565e-16f);uint32_t start=0;
        while(start<n){
            uint32_t m = FindUnreducedEnd(start, eps);
            ES(m,ExtendedFloat(0));int l=start,last=m;start=m+1;
            bool reverse=Less(AbsW(D(last)),AbsW(D(l)));if(reverse){int t=l;l=last;last=t;}
            for(uint32_t it=0;it<64*n;++it){
                if((!reverse && l>last)||(reverse && l<last))break;m=l;
                if(!reverse){for(;int(m)<last;++m)if(Deflated(m))break;if(int(m)<last)ES(m,ExtendedFloat(0));}
                else{for(;int(m)>last;--m)if(Deflated(m-1))break;if(int(m)>last)ES(m-1,ExtendedFloat(0));}
                if(int(m)==l){float x=Q(l).h;if(x*x>0.8f*probeNorm)return x<0?-1.0f:1.0f;l+=reverse?-1:1;continue;}
                if((!reverse && int(m)==l+1)||(reverse && int(m)==l-1)){
                    uint32_t i=reverse?m:l;ExtendedFloat c,s;Eigen2W(D(i),E(i),D(i+1),c,s);ES(i,ExtendedFloat(0));Rotate(i,c,s);
                    float x=Q(i).h,y=Q(i+1).h;if(x*x>0.8f*probeNorm)return x<0?-1.0f:1.0f;if(y*y>0.8f*probeNorm)return y<0?-1.0f:1.0f;
                    l+=reverse?-2:2;continue;
                }
                ChaseBulge(l, m, reverse);
            }
        }
        ExtendedFloat x=Q(0);for(uint32_t i=1;i<n;++i)if(Less(AbsW(x),AbsW(Q(i))))x=Q(i);
        return x.Negative()?-1.0f:1.0f;
    }
public:
    __aicore__ inline TridiagonalBasis(LocalTensor<float> data,uint32_t size):buf(data),n(size){}
    __aicore__ inline void Save(uint32_t i,ExtendedFloat diag,ExtendedFloat off){Write(n*n+5*n+i,diag);Write(n*n+6*n+i,off);}
    __aicore__ inline float ProbeSaved(LocalTensor<float> q,uint32_t upper){
        for(uint32_t i=0;i<n;++i){DS(i,Read(n*n+5*n+(upper?n-1-i:i)));ES(i,i+1<n?Read(n*n+6*n+(upper?n-2-i:i)):ExtendedFloat(0));QS(i,ExtendedFloat(q.GetValue(upper?n-1-i:i)));}
        return QR();
    }
    __aicore__ inline float Probe(LocalTensor<float> diag,LocalTensor<float> off,LocalTensor<float> q,uint32_t begin){
        for(uint32_t i=0;i<n;++i){DS(i,ExtendedFloat(diag.GetValue(begin+i)));ES(i,ExtendedFloat(i+1<n?off.GetValue(begin+i):0));QS(i,ExtendedFloat(q.GetValue(begin+i)));}
        return QR();
    }

};
