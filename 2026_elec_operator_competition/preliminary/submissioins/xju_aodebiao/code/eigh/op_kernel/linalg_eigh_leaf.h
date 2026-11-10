/*
 * Copyright (c) 2026 aodebiao.
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
#include "linalg_eigh_core.h"

template<class Kernel> class EighLeaf : public EighCore<Kernel> {
public:
    __aicore__ inline Kernel &Self() {return static_cast<Kernel&>(*this);}
    __aicore__ inline void Rotate(uint32_t i,float c,float s) {
        if (!Self().WantsVectors()) return;
        if (Self().UseResident()) {
            auto qc=Self().qCache_.template Get<float>();auto p=qc[i*Self().pitch_];auto q=qc[(i+1U)*Self().pitch_];
            auto tmp=Self().tmp_.template Get<float>();auto a=tmp;auto b=tmp[CAP];
            AscendC::Muls(a,p,c,Self().n_);AscendC::Muls(b,q,s,Self().n_);AscendC::Sub(a,a,b,Self().n_);
            AscendC::Muls(b,p,s,Self().n_);AscendC::Muls(q,q,c,Self().n_);AscendC::Add(q,q,b,Self().n_);
            AscendC::Muls(p,a,1.0f,Self().n_);
            return;
        }
        auto p=Self().Load(i,true);auto q=Self().Load(i+1U,true);
        auto tmp=Self().tmp_.template Get<float>();auto a=tmp;auto b=tmp[CAP];
        AscendC::Muls(a,p,c,Self().n_);AscendC::Muls(b,q,s,Self().n_);AscendC::Sub(a,a,b,Self().n_);
        AscendC::Muls(b,p,s,Self().n_);AscendC::Muls(q,q,c,Self().n_);AscendC::Add(q,q,b,Self().n_);
        Self().Store(i,a,true);Self().Store(i+1U,q,true);Self().in_.FreeTensor(p);Self().in_.FreeTensor(q);
    }
    __aicore__ inline bool SmallOff(uint32_t i) {
        auto d=Self().meta_.template Get<float>();auto e=d[CAP];
        constexpr float eps=5.9604644775390625e-8f;
        const float off=e.GetValue(i);
        return off*off <= eps*eps*Self().Abs(d.GetValue(i))*Self().Abs(d.GetValue(i+1U))+1.17549435e-38f;
    }
    __aicore__ inline void PairEigen(uint32_t i) {
        auto d=Self().meta_.template Get<float>();auto e=d[CAP];
        const float a=d.GetValue(i),b=e.GetValue(i),cc=d.GetValue(i+1U);
        const float sm=a+cc,df=a-cc,tb=2.0f*b,rt=Self().Hypot(df,tb);
        float first,second;int32_t sign1;
        if (sm<0) {first=0.5f*(sm-rt);sign1=-1;}
        else {first=0.5f*(sm+rt);sign1=1;}
        if (sm==0) second=-0.5f*rt;
        else {
            const float big=Self().Abs(a)>Self().Abs(cc)?a:cc,small=Self().Abs(a)>Self().Abs(cc)?cc:a;
            second=first!=0.0f?(big/first)*small-(b/first)*b:0.0f;
        }
        const float cs=df>=0?df+rt:df-rt;const int32_t sign2=df>=0?1:-1;
        float cosine,sine;
        if (Self().Abs(cs)>Self().Abs(tb)) {
            const float t=cs!=0.0f?-tb/cs:0.0f;sine=1.0f/Self().Root(1.0f+t*t);cosine=t*sine;
        } else if (tb==0) {cosine=1.0f;sine=0.0f;}
        else {const float t=-cs/tb;cosine=1.0f/Self().Root(1.0f+t*t);sine=t*cosine;}
        if (sign1==sign2) {const float old=cosine;cosine=-sine;sine=old;}
        d.SetValue(i,first);d.SetValue(i+1U,second);e.SetValue(i,0.0f);
        Self().Rotate(i,cosine,-sine);
    }
    __aicore__ inline void Givens(float f,float g,float &c,float &s,float &r) {
        // Stable LAPACK 3.x plane-rotation convention. The sign of r is
        // retained when g dominates; zero cases do not introduce a sign flip.
        if (g==0.0f) {c=1.0f;s=0.0f;r=f;return;}
        if (f==0.0f) {c=0.0f;s=1.0f;r=g;return;}
        r=Self().Hypot(f,g);
        if (!(r>0.0f)) {c=1.0f;s=0.0f;return;}
        c=f/r;s=g/r;
        if (Self().Abs(f)>Self().Abs(g) && c<0.0f) {c=-c;s=-s;r=-r;}
    }
    __aicore__ inline void WideSolveQL() {
        auto d=Self().meta_.template Get<float>();auto e=d[CAP];auto original=Self().triOriginal_.template Get<float>();Self().VS();
        WideView wide(Self().orientationWork_.template Get<float>()[2U*CAP]);auto off=wide[CAP];
        for (uint32_t i=0;i<Self().n_;++i) {
            original.SetValue(i,d.GetValue(i));original.SetValue(CAP+i,e.GetValue(i));
            wide.SetValue(i,Wide(d.GetValue(i)));off.SetValue(i,Wide(e.GetValue(i)));
        }
        Self().referenceBlocks_=false;
        for (uint32_t i=0;i+1U<Self().n_;++i) if (Self().ReferenceBoundary(i)) Self().referenceBlocks_=true;
        Self().SV();Self().LeafSolveQL();Self().VS();
        for (uint32_t i=0;i<Self().n_;++i) {const Wide x=wide.GetValue(i);d.SetValue(i,x.h+x.l);}
        Self().SV();
    }
    __aicore__ inline void SolveQL() {
        auto d=Self().meta_.template Get<float>();auto e=d[CAP];
        auto original=Self().triOriginal_.template Get<float>();Self().VS();
        for (uint32_t i=0;i<Self().n_;++i) {original.SetValue(i,d.GetValue(i));original.SetValue(CAP+i,e.GetValue(i));}
        Self().referenceBlocks_=false;
        for (uint32_t i=0;i+1U<Self().n_;++i) if (Self().ReferenceBoundary(i)) Self().referenceBlocks_=true;
        const float center=Self().DiagonalCenter();
        for (uint32_t i=0;i<Self().n_;++i) d.SetValue(i,d.GetValue(i)-center);
        Self().SV();
        const float joining=Self().n_>1U?e.GetValue(Self().n_/2U-1U):0.0f;
        Self().splitSign_=joining<0?-1.0f:(joining>0?1.0f:0.0f);
        uint32_t lo=0U,total=0U;
        while (lo<Self().n_) {
            uint32_t hi=lo;
            while (hi+1U<Self().n_ && !Self().SmallOff(hi)) ++hi;
            if (hi+1U<Self().n_) e.SetValue(hi,0.0f);
            if (hi==lo) {++lo;continue;}
            if (Self().Abs(d.GetValue(lo))<=Self().Abs(d.GetValue(hi))) Self().QlSweepUp(lo,hi,total);
            else Self().QlSweepDown(lo,hi,total);
            lo=hi+1U;
        }
        for (uint32_t i=0;i<Self().n_;++i) d.SetValue(i,d.GetValue(i)+center);
    }
    // Implicit-shift QL sweep of one unreduced block, ascending direction.
    __aicore__ inline void QlSweepUp(uint32_t lo,uint32_t hi,uint32_t &total) {
        auto d=Self().meta_.template Get<float>();auto e=d[CAP];
        uint32_t l=lo;
        while (l<=hi && total<96U*Self().n_) {
            uint32_t m=l;while (m<hi && !Self().SmallOff(m)) ++m;
            if (m<hi) e.SetValue(m,0.0f);
            if (m==l) {++l;continue;}
            if (m==l+1U) {Self().PairEigen(l);l+=2U;continue;}
            ++total;
            const float old=d.GetValue(l);
            const float denominator=2.0f*e.GetValue(l);
            float g=0.0f;
            if (denominator!=0.0f) g=(d.GetValue(l+1U)-old)/denominator;
            const float rad=Self().Hypot(g,1.0f);
            const float shift=g+(g>=0?rad:-rad);
            g=d.GetValue(m)-old+(shift!=0.0f?e.GetValue(l)/shift:0.0f);
            float sine=1.0f,cosine=1.0f,p=0.0f;
            for (uint32_t ii=m;ii>l;--ii) {
                const uint32_t i=ii-1U;
                const float f=sine*e.GetValue(i),b=cosine*e.GetValue(i);float r=0.0f;
                Self().Givens(g,f,cosine,sine,r);
                if (i+1U!=m) e.SetValue(i+1U,r);
                g=d.GetValue(i+1U)-p;
                const float next=(d.GetValue(i)-g)*sine+2.0f*cosine*b;
                p=sine*next;d.SetValue(i+1U,g+p);g=cosine*next-b;
                Self().QueueRotation(i,cosine,sine);
            }
            Self().FinishRotations(l,m-1U,true);
            d.SetValue(l,d.GetValue(l)-p);e.SetValue(l,g);
        }
    }
    // Same sweep walked from the bottom of the block.
    __aicore__ inline void QlSweepDown(uint32_t lo,uint32_t hi,uint32_t &total) {
        auto d=Self().meta_.template Get<float>();auto e=d[CAP];
        int32_t l=static_cast<int32_t>(hi);
        const int32_t first=static_cast<int32_t>(lo);
        while (l>=first && total<96U*Self().n_) {
            int32_t m=l;while (m>first && !Self().SmallOff(static_cast<uint32_t>(m-1))) --m;
            if (m>first) e.SetValue(m-1,0.0f);
            if (m==l) {--l;continue;}
            if (m==l-1) {Self().PairEigen(static_cast<uint32_t>(m));l-=2;continue;}
            ++total;
            const float old=d.GetValue(l);
            const float denominator=2.0f*e.GetValue(l-1);
            float g=0.0f;
            if (denominator!=0.0f) g=(d.GetValue(l-1)-old)/denominator;
            const float rad=Self().Hypot(g,1.0f);
            const float shift=g+(g>=0?rad:-rad);
            g=d.GetValue(m)-old+(shift!=0.0f?e.GetValue(l-1)/shift:0.0f);
            float sine=1.0f,cosine=1.0f,p=0.0f;
            for (int32_t i=m;i<l;++i) {
                const float f=sine*e.GetValue(i),b=cosine*e.GetValue(i);float r=0.0f;
                Self().Givens(g,f,cosine,sine,r);
                if (i!=m) e.SetValue(i-1,r);
                g=d.GetValue(i)-p;
                const float next=(d.GetValue(i+1)-g)*sine+2.0f*cosine*b;
                p=sine*next;d.SetValue(i,g+p);g=cosine*next-b;
                Self().QueueRotation(static_cast<uint32_t>(i),cosine,-sine);
            }
            Self().FinishRotations(static_cast<uint32_t>(m),static_cast<uint32_t>(l-1),false);
            d.SetValue(l,d.GetValue(l)-p);e.SetValue(l-1,g);
        }
    }
    // Obtain accurate boundary components before deciding whether a secular
    // update is deflated. Two inverse iterations suppress roundoff in tiny
    // components of a tridiagonal eigenvector; the factorization stays on NPU.
    __aicore__ inline WideView LeafData() {
        if (!Self().UseSmall()) return WideView(Self().orientationWork_.template Get<float>()[2U*CAP]);
        return WideView(Self().leafMeta_.template Get<float>());
    }
    __aicore__ inline float LeafOrientationSign(uint32_t lo,uint32_t hi,Wide target,float leftShift,float rightShift) {
        const uint32_t width=hi-lo;
        WideView vector(Self().orientationWork_.template Get<float>()),d(Self().orientationWork_.template Get<float>()[2U*CAP]);auto e=d[CAP];
        WideView q(Self().orientationWork_.template Get<float>()[6U*CAP]);auto original=Self().triOriginal_.template Get<float>();
        for (uint32_t i=0;i<width;++i) {
            Wide diagonal=Wide(original.GetValue(lo+i))+Wide(Self().referenceCenter_);
            if (i==0U) diagonal=diagonal-Wide(leftShift);
            if (i+1U==width) diagonal=diagonal-Wide(rightShift);
            d.SetValue(i,diagonal);e.SetValue(i,i+1U<width?Wide(original.GetValue(CAP+lo+i)):Wide(0));
            for (uint32_t j=0;j<width;++j) q.SetValue(i*width+j,Wide(i==j?1.0f:0.0f));
        }
        const float oldSplit=Self().splitSign_;Self().leafOrientation_=true;Self().leafWidth_=width;
        Self().LeafSolveQL();Self().leafOrientation_=false;Self().splitSign_=oldSplit;
        uint32_t best=0U;Wide distance=Self().Abs(d.GetValue(0)-target);
        for (uint32_t i=1;i<width;++i) {const Wide delta=Self().Abs(d.GetValue(i)-target);if (delta<distance) {best=i;distance=delta;}}
        Wide dot(0);for (uint32_t j=0;j<width;++j) dot=dot+q.GetValue(best*width+j)*vector.GetValue(lo+j);
        return dot<Wide(0)?-1.0f:1.0f;
    }
    __aicore__ inline void LeafRotate(uint32_t i,Wide c,Wide s) {
        if (!Self().leafOrientation_) {Self().Rotate(i,c.h+c.l,s.h+s.l);return;}
        WideView q(Self().orientationWork_.template Get<float>()[6U*CAP]);
        for (uint32_t j=0;j<Self().leafWidth_;++j) {
            const uint32_t a=i*Self().leafWidth_+j,b=(i+1U)*Self().leafWidth_+j;
            const Wide p=q.GetValue(a),r=q.GetValue(b);
            q.SetValue(a,c*p-s*r);q.SetValue(b,s*p+c*r);
        }
    }
    __aicore__ inline void LeafExport() {
        WideView d(Self().leafMeta_.template Get<float>());auto ordinary=Self().meta_.template Get<float>();Self().VS();
        for (uint32_t i=0;i<Self().n_;++i) {const Wide x=d.GetValue(i);ordinary.SetValue(i,x.h+x.l);}
        Self().SV();
    }
    __aicore__ inline bool LeafSmallOff(uint32_t i) {
        auto d=Self().LeafData();auto e=d[CAP];
        const Wide eps=1.1102230246251565e-16f;
        const Wide off=e.GetValue(i);
        return off*off <= eps*eps*Self().Abs(d.GetValue(i))*Self().Abs(d.GetValue(i+1U))+1.17549435e-38f;
    }
    __aicore__ inline void LeafPairEigen(uint32_t i) {
        WideView d=Self().LeafData();
        auto e=d[CAP];
        const Wide a=d.GetValue(i),b=e.GetValue(i),cc=d.GetValue(i+1U);
        const Wide sm=a+cc,df=a-cc,tb=2.0f*b,rt=Self().Hypot(df,tb);
        Wide first,second;int32_t sign1;
        if (sm<Wide(0)) {first=0.5f*(sm-rt);sign1=-1;}
        else {first=0.5f*(sm+rt);sign1=1;}
        if (sm==Wide(0)) second=-0.5f*rt;
        else {
            const Wide big=Self().Abs(a)>Self().Abs(cc)?a:cc,small=Self().Abs(a)>Self().Abs(cc)?cc:a;
            second=first!=Wide(0)?(big/first)*small-(b/first)*b:Wide(0);
        }
        const Wide cs=df>=Wide(0)?df+rt:df-rt;const int32_t sign2=df>=Wide(0)?1:-1;
        Wide cosine,sine;
        if (Self().Abs(cs)>Self().Abs(tb)) {
            const Wide t=cs!=Wide(0)?-tb/cs:Wide(0);sine=1.0f/Self().Root(1.0f+t*t);cosine=t*sine;
        } else if (tb==Wide(0)) {cosine=Wide(1.0f);sine=Wide(0.0f);}
        else {const Wide t=-cs/tb;cosine=1.0f/Self().Root(1.0f+t*t);sine=t*cosine;}
        if (sign1==sign2) {const Wide old=cosine;cosine=-sine;sine=old;}
        d.SetValue(i,first);d.SetValue(i+1U,second);e.SetValue(i,Wide(0));
        Self().LeafRotate(i,cosine,-sine);
    }
    __aicore__ inline void LeafGivens(Wide f,Wide g,Wide &c,Wide &s,Wide &r) {
        if (g==0.0f) {c=1.0f;s=0.0f;r=f;return;}
        if (f==0.0f) {c=0.0f;s=g<0.0f?-1.0f:1.0f;r=Self().Abs(g);return;}
        r=Self().Hypot(f,g);if (f<0.0f) r=-r;
        const Wide inverse=r!=Wide(0)?Self().DivideOrZero(Wide(1),r):Wide(0);
        c=f*inverse;s=g*inverse;
    }
    // Block size of the current QL sweep: the leaf width or the full order.
    __aicore__ inline uint32_t LeafDimension() {return Self().leafOrientation_?Self().leafWidth_:Self().n_;}
    __aicore__ inline void LeafSolveQL() {
        const uint32_t dimension=Self().LeafDimension();
        WideView d=Self().LeafData();
        auto e=d[CAP];
        const Wide joining=dimension>1U?e.GetValue(dimension/2U-1U):Wide(0);
        Self().splitSign_=joining<Wide(0)?-1.0f:(joining>Wide(0)?1.0f:0.0f);
        uint32_t lo=0U,total=0U;
        while (lo<dimension) {
            uint32_t hi=lo;
            while (hi+1U<dimension && !Self().LeafSmallOff(hi)) ++hi;
            if (hi+1U<dimension) e.SetValue(hi,Wide(0));
            if (hi==lo) {++lo;continue;}
            if (Self().Abs(d.GetValue(lo))<=Self().Abs(d.GetValue(hi))) Self().LeafSweepUp(lo,hi,total,d);
            else Self().LeafSweepDown(lo,hi,total,d);
            lo=hi+1U;
        }
    }
    // Implicit-shift QL sweep of one leaf block, ascending direction.
    __aicore__ inline void LeafSweepUp(uint32_t lo,uint32_t hi,uint32_t &total,WideView d) {
        auto e=d[CAP];
        const uint32_t dimension=Self().LeafDimension();
        uint32_t l=lo;
        while (l<=hi && total<96U*dimension) {
            uint32_t m=l;while (m<hi && !Self().LeafSmallOff(m)) ++m;
            if (m<hi) e.SetValue(m,Wide(0));
            if (m==l) {++l;continue;}
            if (m==l+1U) {Self().LeafPairEigen(l);l+=2U;continue;}
            ++total;
            const Wide old=d.GetValue(l);
            const Wide denominator=2.0f*e.GetValue(l);
            Wide g(0);
            if (denominator!=Wide(0)) g=(d.GetValue(l+1U)-old)/denominator;
            const Wide rad=Self().Hypot(g,Wide(1));
            const Wide shift=g+(g>=Wide(0)?rad:-rad);
            g=d.GetValue(m)-old+(shift!=Wide(0)?e.GetValue(l)/shift:Wide(0));
            Wide sine(1.0f),cosine(1.0f),p(0.0f);
            for (uint32_t ii=m;ii>l;--ii) {
                const uint32_t i=ii-1U;
                const Wide f=sine*e.GetValue(i),b=cosine*e.GetValue(i);Wide r(0);
                Self().LeafGivens(g,f,cosine,sine,r);
                if (i+1U!=m) e.SetValue(i+1U,r);
                g=d.GetValue(i+1U)-p;
                const Wide next=(d.GetValue(i)-g)*sine+2.0f*cosine*b;
                p=sine*next;d.SetValue(i+1U,g+p);g=cosine*next-b;
                Self().LeafRotate(i,cosine,sine);
            }
            d.SetValue(l,d.GetValue(l)-p);e.SetValue(l,g);
        }
    }
    // Same sweep walked from the bottom of the leaf block.
    __aicore__ inline void LeafSweepDown(uint32_t lo,uint32_t hi,uint32_t &total,WideView d) {
        auto e=d[CAP];
        const uint32_t dimension=Self().LeafDimension();
        int32_t l=static_cast<int32_t>(hi);
        const int32_t first=static_cast<int32_t>(lo);
        while (l>=first && total<96U*dimension) {
            int32_t m=l;while (m>first && !Self().LeafSmallOff(static_cast<uint32_t>(m-1))) --m;
            if (m>first) e.SetValue(m-1,Wide(0));
            if (m==l) {--l;continue;}
            if (m==l-1) {Self().LeafPairEigen(static_cast<uint32_t>(m));l-=2;continue;}
            ++total;
            const Wide old=d.GetValue(l);
            const Wide denominator=2.0f*e.GetValue(l-1);
            Wide g(0);
            if (denominator!=Wide(0)) g=(d.GetValue(l-1)-old)/denominator;
            const Wide rad=Self().Hypot(g,Wide(1));
            const Wide shift=g+(g>=Wide(0)?rad:-rad);
            g=d.GetValue(m)-old+(shift!=Wide(0)?e.GetValue(l-1)/shift:Wide(0));
            Wide sine(1.0f),cosine(1.0f),p(0.0f);
            for (int32_t i=m;i<l;++i) {
                const Wide f=sine*e.GetValue(i),b=cosine*e.GetValue(i);Wide r(0);
                Self().LeafGivens(g,f,cosine,sine,r);
                if (i!=m) e.SetValue(i-1,r);
                g=d.GetValue(i)-p;
                const Wide next=(d.GetValue(i+1)-g)*sine+2.0f*cosine*b;
                p=sine*next;d.SetValue(i,g+p);g=cosine*next-b;
                Self().LeafRotate(static_cast<uint32_t>(i),cosine,-sine);
            }
            d.SetValue(l,d.GetValue(l)-p);e.SetValue(l-1,g);
        }
    }
    // Householder vector of one leaf step; false when the block is already zero.
    __aicore__ inline bool LeafReflectorVector(uint32_t j,uint32_t first,uint32_t end,uint32_t pivot, uint32_t off,Wide &t) {
        WideView matrix(Self().leafMatrix_.template Get<float>()),meta(Self().leafMeta_.template Get<float>());
        auto d=meta;auto e=meta[CAP];
        WideView v(Self().leafWork_.template Get<float>());
        auto ordinary=Self().meta_.template Get<float>();auto tau=ordinary[2U*CAP];
        d.SetValue(j,matrix.GetValue(j*Self().n_+j));Wide scale(0);
        for (uint32_t c=0;c<Self().n_;++c) {
            Wide x=(c>=first && c<end)?matrix.GetValue(j*Self().n_+c):Wide(0);
            v.SetValue(c,x);if (Self().Abs(x)>scale) scale=Self().Abs(x);
        }
        if (!(scale>Wide(0))) {e.SetValue(off,Wide(0));tau.SetValue(j,0.0f);return false;}
        Wide norm(0);const Wide inverseScale=Self().DivideOrZero(Wide(1),scale);
        for (uint32_t c=first;c<end;++c) {Wide x=v.GetValue(c)*inverseScale;v.SetValue(c,x);norm=norm+x*x;}
        const Wide length=Self().Root(norm),x0=v.GetValue(pivot),alpha=x0>=Wide(0)?-length:length;
        if (!(length>Wide(0))) {e.SetValue(off,Wide(0));tau.SetValue(j,0.0f);return false;}
        const Wide divisor=x0-alpha;
        if (divisor==Wide(0)) {e.SetValue(off,Wide(0));tau.SetValue(j,0.0f);return false;}
        t=(alpha-x0)/alpha;
        const Wide inverseDivisor=Wide(1)/divisor;
        for (uint32_t c=first;c<end;++c) v.SetValue(c,v.GetValue(c)*inverseDivisor);
        v.SetValue(pivot,Wide(1));e.SetValue(off,alpha*scale);tau.SetValue(j,t.h+t.l);
        return true;
    }
    // Rank-two update of the leaf matrix with that reflector.
    __aicore__ inline void LeafReflectorApply(uint32_t first,uint32_t end,Wide t) {
        WideView matrix(Self().leafMatrix_.template Get<float>());
        WideView v(Self().leafWork_.template Get<float>()),w=v[CAP];
        for (uint32_t r=first;r<end;++r) {
            Wide dot(0);for (uint32_t c=first;c<end;++c) dot=dot+matrix.GetValue(r*Self().n_+c)*v.GetValue(c);
            w.SetValue(r,t*dot);
        }
        Wide dot(0);for (uint32_t r=first;r<end;++r) dot=dot+v.GetValue(r)*w.GetValue(r);
        const Wide beta=Wide(-0.5f)*t*dot;
        for (uint32_t r=first;r<end;++r) w.SetValue(r,w.GetValue(r)+beta*v.GetValue(r));
        for (uint32_t r=first;r<end;++r) for (uint32_t c=first;c<end;++c)
            matrix.SetValue(r*Self().n_+c,matrix.GetValue(r*Self().n_+c)-v.GetValue(r)*w.GetValue(c)-w.GetValue(r)*v.GetValue(c));
    }
    // Publish the last diagonal pair of the leaf factorization.
    __aicore__ inline void LeafFinalizeDiagonal() {
        WideView matrix(Self().leafMatrix_.template Get<float>()),meta(Self().leafMeta_.template Get<float>());
        auto d=meta;auto e=meta[CAP];
        if (Self().n_>=2U) {
            const uint32_t r=Self().lower_?Self().n_-2U:1U,c=Self().lower_?Self().n_-1U:0U;
            d.SetValue(r,matrix.GetValue(r*Self().n_+r));e.SetValue(Self().lower_?Self().n_-2U:0U,matrix.GetValue(r*Self().n_+c));
        }
        const uint32_t last=Self().lower_?Self().n_-1U:0U;d.SetValue(last,matrix.GetValue(last*Self().n_+last));
        Self().SV();
    }
    __aicore__ inline void LeafTridiagonalize(uint32_t batch) {
        Self().LoadLeafMatrix(batch);
        for (uint32_t step=0;step+2U<Self().n_;++step) {
            const uint32_t j=Self().lower_?step:Self().n_-1U-step,first=Self().lower_?j+1U:0U,end=Self().lower_?Self().n_:j;
            const uint32_t pivot=Self().lower_?j+1U:j-1U,off=Self().lower_?j:j-1U;
            Wide t(0);
            if (Self().LeafReflectorVector(j,first,end,pivot,off,t)) Self().LeafReflectorApply(first,end,t);
            auto row=Self().vBuf_.template Get<float>();WideView v(Self().leafWork_.template Get<float>());
            for (uint32_t c=0;c<Self().n_;++c) {const Wide x=v.GetValue(c);row.SetValue(c,x.h+x.l);}
            Self().SV();Self().Store(j,row);Self().VS();
        }
        Self().LeafFinalizeDiagonal();
    }
};
