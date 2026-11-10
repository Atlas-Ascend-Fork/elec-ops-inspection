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
#include "linalg_eigh_leaf.h"

struct EighClusterMeasure {
    Wide projection,boundary;bool fromLeft;float tolerance;
};

template<class Kernel> class EighSpectral : public EighLeaf<Kernel> {
public:
    __aicore__ inline Kernel &Self() {return static_cast<Kernel&>(*this);}
    // Mean of the tridiagonal diagonal; the QL iteration is centred on it.
    __aicore__ inline float DiagonalCenter() {
        auto d=Self().meta_.template Get<float>();
        Wide sum(0),count(0);
        for (uint32_t i=0;i<Self().n_;++i) {sum=sum+Wide(d.GetValue(i));count=count+Wide(1);}
        const Wide average=count!=Wide(0)?sum/count:Wide(0);
        return average.h+average.l;
    }
    __aicore__ inline void ProductPair(AscendC::LocalTensor<float> a,AscendC::LocalTensor<float> b) {
        auto z=Self().orientationWork_.template Get<float>()[6U*CAP];
        auto h=z,l=z[CAP],ah=z[2U*CAP],al=z[3U*CAP],bh=z[4U*CAP],bl=z[5U*CAP],t=z[6U*CAP];
        // Split inputs before writing h: a may alias h.
        AscendC::Muls(t,a,4097.0f,Self().n_);AscendC::Sub(ah,t,a,Self().n_);AscendC::Sub(ah,t,ah,Self().n_);AscendC::Sub(al,a,ah,Self().n_);
        AscendC::Muls(t,b,4097.0f,Self().n_);AscendC::Sub(bh,t,b,Self().n_);AscendC::Sub(bh,t,bh,Self().n_);AscendC::Sub(bl,b,bh,Self().n_);
        AscendC::Mul(h,a,b,Self().n_);
        AscendC::Mul(l,ah,bh,Self().n_);AscendC::Sub(l,l,h,Self().n_);
        AscendC::Mul(t,ah,bl,Self().n_);AscendC::Add(l,l,t,Self().n_);
        AscendC::Mul(t,al,bh,Self().n_);AscendC::Add(l,l,t,Self().n_);
        AscendC::Mul(t,al,bl,Self().n_);AscendC::Add(l,l,t,Self().n_);
    }
    __aicore__ inline void MultiplyPair(AscendC::LocalTensor<float> b) {
        auto z=Self().orientationWork_.template Get<float>()[6U*CAP];
        auto h=z,l=z[CAP],extra=Self().wBuf_.template Get<float>(),t=z[6U*CAP];
        AscendC::Mul(extra,l,b,Self().n_);Self().ProductPair(h,b);AscendC::Add(l,l,extra,Self().n_);
        AscendC::Add(t,h,l,Self().n_);AscendC::Sub(h,t,h,Self().n_);AscendC::Sub(l,l,h,Self().n_);
        AscendC::Muls(h,t,1.0f,Self().n_);
    }
    __aicore__ inline void AddPair(AscendC::LocalTensor<float> bh,AscendC::LocalTensor<float> bl,uint32_t count) {
        auto z=Self().orientationWork_.template Get<float>()[6U*CAP];
        auto h=z,l=z[CAP],t=z[2U*CAP],v=z[3U*CAP],s=z[4U*CAP],u=z[5U*CAP];
        AscendC::Add(t,h,bh,count);AscendC::Sub(v,t,h,count);
        AscendC::Sub(s,t,v,count);AscendC::Sub(s,h,s,count);
        AscendC::Sub(u,bh,v,count);AscendC::Add(s,s,u,count);
        AscendC::Add(u,l,bl,count);AscendC::Add(u,u,s,count);
        AscendC::Add(h,t,u,count);AscendC::Sub(v,h,t,count);AscendC::Sub(l,u,v,count);
    }
    __aicore__ inline void CompensatedResidual(AscendC::LocalTensor<float> row,Wide target,AscendC::LocalTensor<float> rhs) {
        auto original=Self().triOriginal_.template Get<float>(),z=Self().orientationWork_.template Get<float>()[6U*CAP];
        auto h=z,l=z[CAP],t=z[2U*CAP],v=z[3U*CAP],s=z[4U*CAP],u=z[5U*CAP];
        auto next=Self().vBuf_.template Get<float>(),extra=Self().wBuf_.template Get<float>(),saved=Self().tmp_.template Get<float>();
        AscendC::Adds(h,original,-target.h,Self().n_);AscendC::Sub(v,h,original,Self().n_);
        AscendC::Sub(s,h,v,Self().n_);AscendC::Sub(s,original,s,Self().n_);
        AscendC::Muls(u,v,-1.0f,Self().n_);AscendC::Adds(u,u,-target.h,Self().n_);AscendC::Add(s,s,u,Self().n_);
        AscendC::Adds(l,s,-target.l,Self().n_);AscendC::Add(next,h,l,Self().n_);
        AscendC::Sub(t,next,h,Self().n_);AscendC::Sub(l,l,t,Self().n_);
        AscendC::Mul(extra,l,row,Self().n_);Self().ProductPair(next,row);AscendC::Add(l,l,extra,Self().n_);
        AscendC::Add(t,h,l,Self().n_);AscendC::Sub(h,t,h,Self().n_);AscendC::Sub(l,l,h,Self().n_);
        AscendC::Muls(saved,t,1.0f,Self().n_);AscendC::Muls(saved[CAP],l,1.0f,Self().n_);
        auto backward=Self().meta_.template Get<uint32_t>()[3U*CAP];
        AscendC::Gather(next,row,backward,0U,Self().n_);AscendC::Gather(extra,original,backward,CAP*4U,Self().n_);
        Self().VS();extra.SetValue(0U,0.0f);Self().SV();Self().ProductPair(next,extra);Self().AddPair(saved,saved[CAP],Self().n_);
        AscendC::Muls(saved,h,1.0f,Self().n_);AscendC::Muls(saved[CAP],l,1.0f,Self().n_);
        AscendC::Gather(next,row,Self().offsets_.template Get<uint32_t>(),0U,Self().n_);Self().ProductPair(original[CAP],next);
        Self().VS();h.SetValue(Self().n_-1U,0.0f);l.SetValue(Self().n_-1U,0.0f);Self().SV();Self().AddPair(saved,saved[CAP],Self().n_);
        AscendC::Add(rhs,h,l,Self().n_);Self().VS();
    }
    __aicore__ inline Wide SumWidePairs(uint32_t count) {
        auto z=Self().orientationWork_.template Get<float>()[6U*CAP];
        auto h=z,l=z[CAP];
        while (count>8U) {
            // Vector source addresses must remain aligned to a 32-byte sector.
            const uint32_t half=(((count+1U)/2U)+7U)&~7U;
            Self().AddPair(h[half],l[half],count-half);
            count=half;
        }
        Self().VS();Wide result(0.0f);
        for (uint32_t i=0;i<count;++i) result=result+Wide(h.GetValue(i),l.GetValue(i));
        return result;
    }
    __aicore__ inline Wide Rayleigh(AscendC::LocalTensor<float> row) {
        auto original=Self().triOriginal_.template Get<float>(),z=Self().orientationWork_.template Get<float>()[6U*CAP];
        auto h=z,l=z[CAP],saved=Self().tmp_.template Get<float>(),next=Self().vBuf_.template Get<float>();
        Self().ProductPair(row,row);const Wide norm=Self().SumWidePairs(Self().n_);
        Self().SV();Self().ProductPair(original,row);Self().MultiplyPair(row);
        AscendC::Muls(saved,h,1.0f,Self().n_);AscendC::Muls(saved[CAP],l,1.0f,Self().n_);
        AscendC::Muls(next,original[CAP],2.0f,Self().n_);Self().ProductPair(next,row);
        AscendC::Gather(next,row,Self().offsets_.template Get<uint32_t>(),0U,Self().n_);Self().MultiplyPair(next);
        Self().VS();h.SetValue(Self().n_-1U,0.0f);l.SetValue(Self().n_-1U,0.0f);Self().SV();
        Self().AddPair(saved,saved[CAP],Self().n_);
        const Wide value=Self().SumWidePairs(Self().n_);
        return norm>Wide(0)?Self().DivideOrZero(value,norm):Wide(0);
    }
    // Shifted LDL with adjacent row pivoting; false when a pivot vanishes.
    __aicore__ inline bool FloatFactors(float shift) {
        auto original=Self().triOriginal_.template Get<float>();auto work=Self().orientationWork_.template Get<float>();
        auto sub=work;auto diagonal=work[CAP];auto upper=work[2U*CAP];auto upper2=work[3U*CAP];
        auto swaps=Self().orientationWork_.template Get<uint32_t>()[5U*CAP];
        for (uint32_t i=0;i<Self().n_;++i) {
            diagonal.SetValue(i,original.GetValue(i)-shift);upper2.SetValue(i,0.0f);swaps.SetValue(i,0U);
            if (i+1U<Self().n_) {sub.SetValue(i,original.GetValue(CAP+i));upper.SetValue(i,original.GetValue(CAP+i));}
        }
        for (uint32_t i=0;i+1U<Self().n_;++i) {
            const float d=diagonal.GetValue(i),l=sub.GetValue(i);
            if (Self().Abs(d)>=Self().Abs(l)) {
                if (d!=0.0f) {const float factor=l/d;sub.SetValue(i,factor);diagonal.SetValue(i+1U,diagonal.GetValue(i+1U)-factor*upper.GetValue(i));}
            } else {
                const float factor=l!=0.0f?d/l:0.0f,next=diagonal.GetValue(i+1U),old=upper.GetValue(i);
                diagonal.SetValue(i,l);sub.SetValue(i,factor);upper.SetValue(i,next);
                diagonal.SetValue(i+1U,old-factor*next);swaps.SetValue(i,1U);
                if (i+2U<Self().n_) {const float u=upper.GetValue(i+1U);upper2.SetValue(i,u);upper.SetValue(i+1U,-factor*u);}
            }
        }
        for (uint32_t i=0;i<Self().n_;++i) if (diagonal.GetValue(i)==0.0f) return false;
        return true;
    }
    __aicore__ inline void SolveFloatFactors(AscendC::LocalTensor<float> rhs) {
        auto work=Self().orientationWork_.template Get<float>();
        auto sub=work;auto diagonal=work[CAP];auto upper=work[2U*CAP];auto upper2=work[3U*CAP];
        auto swaps=Self().orientationWork_.template Get<uint32_t>()[5U*CAP];
        for (uint32_t i=0;i+1U<Self().n_;++i) {
            if (swaps.GetValue(i)==0U) rhs.SetValue(i+1U,rhs.GetValue(i+1U)-sub.GetValue(i)*rhs.GetValue(i));
            else {const float old=rhs.GetValue(i),next=rhs.GetValue(i+1U);rhs.SetValue(i,next);rhs.SetValue(i+1U,old-sub.GetValue(i)*next);}
        }
        for (uint32_t ii=Self().n_;ii>0U;--ii) {
            const uint32_t i=ii-1U;float x=rhs.GetValue(i);
            if (i+1U<Self().n_) x=x-upper.GetValue(i)*rhs.GetValue(i+1U);
            if (i+2U<Self().n_) x=x-upper2.GetValue(i)*rhs.GetValue(i+2U);
            const float pivot=diagonal.GetValue(i);
            rhs.SetValue(i,pivot!=0.0f?x/pivot:0.0f);
        }
    }
    // Two inverse iterations on the pivoted factors, then the refreshed value.
    __aicore__ inline void RefineResidual(AscendC::LocalTensor<float> row,Wide &target,uint32_t eigenIndex=0U) {
        auto rhs=Self().orientationWork_.template Get<float>()[4U*CAP];
        for (uint32_t iteration=0;iteration<2U;++iteration) {
            if (iteration>0U) target=Self().Rayleigh(row);
            Self().CompensatedResidual(row,target,rhs);
            Self().SolveFloatFactors(rhs);
            Self().SV();const float denominator=Self().Dot(row,row);
            const float projection=denominator!=0.0f?Self().Dot(row,rhs)/denominator:0.0f;
            auto tmp=Self().tmp_.template Get<float>();AscendC::Muls(tmp,row,projection,Self().n_);
            AscendC::Sub(row,row,rhs,Self().n_);AscendC::Add(row,row,tmp,Self().n_);
            const float norm=Self().Root(Self().Dot(row,row));if (norm>0.0f) AscendC::Muls(row,row,1.0f/norm,Self().n_);
            Self().VS();
            if (iteration==0U && Self().ResidualConverged(row,eigenIndex)) break;
        }
    }
    __aicore__ inline void SecularOrientation() {
        if (Self().UseSmall()) return;
        auto forward=Self().offsets_.template Get<uint32_t>();auto backward=Self().meta_.template Get<uint32_t>()[3U*CAP];Self().VS();
        for (uint32_t i=0;i<Self().n_;++i) {
            forward.SetValue(i,(i+1U<Self().n_?i+1U:i)*4U);backward.SetValue(i,(i>0U?i-1U:0U)*4U);
        }
        Self().SV();
        const uint32_t split=Self().n_/2U;
        if (Self().UseResident()) {
            auto q=Self().qCache_.template Get<float>();
            for (uint32_t r=Self().VectorBegin();r<Self().VectorEnd();++r) {
                auto row=q[r*Self().pitch_];Self().DefectCorrect(row,r);Self().VS();
                const float projection=row.GetValue(split-1U)+Self().splitSign_*row.GetValue(split);
                float orientation=projection>0.0f?-1.0f:1.0f;
                if (Self().referenceBlocks_ || Self().splitSign_==0.0f || Self().Abs(projection)<=3.814697265625e-6f) orientation=Self().FallbackOrientation(row);
                if (orientation<0.0f) AscendC::Muls(row,row,-1.0f,Self().n_);
            }
            return;
        }
        for (uint32_t r=Self().VectorBegin();r<Self().VectorEnd();++r) {
            auto row=Self().Load(r,true);Self().DefectCorrect(row,r);Self().VS();
            const float projection=row.GetValue(split-1U)+Self().splitSign_*row.GetValue(split);
            float orientation=projection>0.0f?-1.0f:1.0f;
            if (Self().referenceBlocks_ || Self().splitSign_==0.0f || Self().Abs(projection)<=3.814697265625e-6f) orientation=Self().FallbackOrientation(row);
            if (orientation<0.0f) AscendC::Muls(row,row,-1.0f,Self().n_);
            Self().Store(r,row,true);Self().in_.FreeTensor(row);
        }
    }
    __aicore__ inline bool ReferenceBoundary(uint32_t i) {
        auto t=Self().triOriginal_.template Get<float>();
        const float scale=Self().Root(Self().Abs(t.GetValue(i)+Self().referenceCenter_))*Self().Root(Self().Abs(t.GetValue(i+1U)+Self().referenceCenter_));
        return Self().Abs(t.GetValue(CAP+i))<=1.1102230246251565e-16f*scale;
    }
    __aicore__ inline float RefinedOrientation(AscendC::LocalTensor<float> row) {
        Wide norm(0),shift(0);
        const Wide rayleigh=Self().RayleighShift(row,norm,shift);
        Self().FactorWideTridiagonal(shift);
        if (!Self().InverseIterateWide()) return 1.0f;
        const Wide quotient=norm!=Wide(0)?rayleigh/norm:Wide(0);
        return Self().ClusterOrientation(quotient+Wide(Self().referenceCenter_));
    }
    // Rayleigh quotient of the candidate vector plus the shift it implies.
    __aicore__ inline Wide RayleighShift(AscendC::LocalTensor<float> row,Wide &norm,Wide &shift) {
        auto original=Self().triOriginal_.template Get<float>();
        WideView vector(Self().orientationWork_.template Get<float>());
        Wide rayleigh(0);norm=Wide(0);float scale=0.0f;Self().VS();
        for (uint32_t i=0;i<Self().n_;++i) {
            Wide x(row.GetValue(i));vector.SetValue(i,x);norm=norm+x*x;
            rayleigh=rayleigh+Wide(original.GetValue(i))*x*x;
            const float mag=Self().Abs(original.GetValue(i));if (mag>scale) scale=mag;
            if (i+1U<Self().n_) rayleigh=rayleigh+Wide(2.0f)*Wide(original.GetValue(CAP+i))*x*Wide(row.GetValue(i+1U));
        }
        if (!(scale>0.0f)) scale=1.0f;
        const Wide quotient=norm!=Wide(0)?Self().DivideOrZero(rayleigh,norm):Wide(0);
        shift=quotient+Wide(scale*2.3283064365386963e-10f);
        return rayleigh;
    }
    // Shifted LDL with adjacent row pivoting, in compensated precision.
    __aicore__ inline void FactorWideTridiagonal(Wide shift) {
        auto original=Self().triOriginal_.template Get<float>();
        WideView sub(Self().orientationWork_.template Get<float>());
        WideView diagonal=sub[CAP],upper=sub[2U*CAP],upper2=sub[3U*CAP];
        auto swaps=Self().orientationWork_.template Get<uint32_t>()[12U*CAP];
        for (uint32_t i=0;i<Self().n_;++i) {
            diagonal.SetValue(i,Wide(original.GetValue(i))-shift);upper2.SetValue(i,Wide(0));swaps.SetValue(i,0U);
            if (i+1U<Self().n_) {sub.SetValue(i,Wide(original.GetValue(CAP+i)));upper.SetValue(i,Wide(original.GetValue(CAP+i)));}
        }
        // Adjacent row pivoting bounds elimination multipliers by one.
        // It avoids the growth and artificial pivot floor of an unpivoted LDL.
        for (uint32_t i=0;i+1U<Self().n_;++i) {
            const Wide d=diagonal.GetValue(i),l=sub.GetValue(i);
            if (Self().Abs(d)>=Self().Abs(l)) {
                if (d!=Wide(0)) {
                    const Wide factor=Self().DivideOrZero(l,d);sub.SetValue(i,factor);
                    diagonal.SetValue(i+1U,diagonal.GetValue(i+1U)-factor*upper.GetValue(i));
                }
            } else {
                const Wide factor=l!=Wide(0)?Self().DivideOrZero(d,l):Wide(0);
                const Wide oldUpper=upper.GetValue(i),next=diagonal.GetValue(i+1U);
                diagonal.SetValue(i,l);sub.SetValue(i,factor);upper.SetValue(i,next);
                diagonal.SetValue(i+1U,oldUpper-factor*next);swaps.SetValue(i,1U);
                if (i+2U<Self().n_) {const Wide u=upper.GetValue(i+1U);upper2.SetValue(i,u);upper.SetValue(i+1U,-factor*u);}
            }
        }
    }
    // Two inverse iterations with the factored tridiagonal.
    __aicore__ inline bool InverseIterateWide() {
        WideView vector(Self().orientationWork_.template Get<float>());
        WideView sub=vector[CAP],diagonal=vector[2U*CAP],upper=vector[3U*CAP];
        WideView upper2=vector[4U*CAP],rhs=vector[5U*CAP];
        auto swaps=Self().orientationWork_.template Get<uint32_t>()[12U*CAP];
        for (uint32_t iteration=0;iteration<2U;++iteration) {
            for (uint32_t i=0;i<Self().n_;++i) rhs.SetValue(i,vector.GetValue(i));
            for (uint32_t i=0;i+1U<Self().n_;++i) {
                if (swaps.GetValue(i)==0U) rhs.SetValue(i+1U,rhs.GetValue(i+1U)-sub.GetValue(i)*rhs.GetValue(i));
                else {
                    const Wide old=rhs.GetValue(i),next=rhs.GetValue(i+1U);
                    rhs.SetValue(i,next);rhs.SetValue(i+1U,old-sub.GetValue(i)*next);
                }
            }
            float max=0.0f;
            for (uint32_t ii=Self().n_;ii>0;--ii) {
                const uint32_t i=ii-1U;Wide y=rhs.GetValue(i);
                if (i+1U<Self().n_) y=y-upper.GetValue(i)*vector.GetValue(i+1U);
                if (i+2U<Self().n_) y=y-upper2.GetValue(i)*vector.GetValue(i+2U);
                const Wide denominator=diagonal.GetValue(i);
                if (denominator==Wide(0)) return false;
                y=Self().DivideOrZero(y,denominator);vector.SetValue(i,y);
                if (Self().Abs(y.h)>max) max=Self().Abs(y.h);
            }
            if (!(max>0.0f)) return false;
            // A common finite scale preserves the eigenvector direction;
            // its rounding cancels from boundary/norm and Rayleigh ratios.
            const Wide inverseScale(1.0f/max);
            for (uint32_t i=0;i<Self().n_;++i) vector.SetValue(i,vector.GetValue(i)*inverseScale);
        }
        return true;
    }
    // Locate the child block that holds the largest component.
    __aicore__ inline void LocateCluster(uint32_t &lo,uint32_t &hi) {
        WideView vector(Self().orientationWork_.template Get<float>());
        uint32_t largest=0U;float magnitude=0.0f;
        for (uint32_t i=0;i<Self().n_;++i) {const float value=Self().Abs(vector.GetValue(i).h);if (value>magnitude) {largest=i;magnitude=value;}}
        lo=largest;hi=largest+1U;
        while (lo>0U && !Self().ReferenceBoundary(lo-1U)) --lo;
        while (hi<Self().n_ && !Self().ReferenceBoundary(hi-1U)) ++hi;
    }
    __aicore__ inline EighClusterMeasure MeasureCluster(WideView vector,uint32_t lo,uint32_t hi,
        float rho,float poleScale) {
        const uint32_t middle=(lo+hi)/2U;
        const float sign=rho<0.0f?-1.0f:1.0f;
        const Wide projection=vector.GetValue(middle-1U)+Wide(sign)*vector.GetValue(middle);
        Wide left(0),right(0);
        for (uint32_t i=lo;i<middle;++i) {const Wide x=vector.GetValue(i);left=left+x*x;}
        for (uint32_t i=middle;i<hi;++i) {const Wide x=vector.GetValue(i);right=right+x*x;}
        const bool fromLeft=left>=right;
        const uint32_t boundaryIndex=fromLeft?middle-1U:middle;
        const Wide normSide=Self().Root(fromLeft?left:right);
        const Wide boundary=normSide>Wide(0)?Self().DivideOrZero(Self().Abs(vector.GetValue(boundaryIndex)),normSide):Wide(0);
        // SLAED2's small-weight test uses the child eigenvector weight,
        // not the projection of the updated eigenvector. z is scaled by
        // 1/sqrt(2), while rho is doubled during the positive rank-one cut.
        const float tolerance=Self().Abs(rho)>0.0f?poleScale*6.280369834735101e-16f/Self().Abs(rho):1.0f;
        return {projection,boundary,fromLeft,tolerance};
    }
    // One rank-one cut. Returns 0 to descend, 1 for a negative projection and
    // 2 for a positive one, which decides the orientation of the vector.
    __aicore__ inline int32_t ClusterCut(uint32_t &lo,uint32_t &hi,float poleScale,float &leftShift,float &rightShift) {
        WideView vector=Self().orientationWork_.template Get<float>();
        auto original=Self().triOriginal_.template Get<float>();
        const uint32_t middle=(lo+hi)/2U;
        const float rho=original.GetValue(CAP+middle-1U);
        const auto cut=Self().MeasureCluster(vector,lo,hi,rho,poleScale);
        const Wide projection=cut.projection,boundary=cut.boundary;
        const bool fromLeft=cut.fromLeft;const float tolerance=cut.tolerance;
        if (boundary>Wide(tolerance)) return projection>Wide(0)?1:2;
        if (fromLeft) {hi=middle;rightShift=Self().Abs(rho);}
        else {lo=middle;leftShift=Self().Abs(rho);}
        return 0;
    }
    // Follow the Cuppen tree only while a joining component stays negligible.
    __aicore__ inline float ClusterOrientation(Wide target) {
        uint32_t lo=0U,hi=0U;
        Self().LocateCluster(lo,hi);
        auto eigenvalues=Self().meta_.template Get<float>();
        float poleScale=1.0f,leftShift=0.0f,rightShift=0.0f;
        for (uint32_t i=0;i<Self().n_;++i) {
            const float x=Self().Abs(eigenvalues.GetValue(i)+Self().referenceCenter_);
            if (x>poleScale) poleScale=x;
        }
        while (hi-lo>25U) {
            const int32_t cut=Self().ClusterCut(lo,hi,poleScale,leftShift,rightShift);
            if (cut!=0) return cut==1?-1.0f:1.0f;
        }
        return Self().LeafOrientationSign(lo,hi,target,leftShift,rightShift);
    }
    __aicore__ inline void BuildEigenvectors() {
        auto original=Self().triOriginal_.template Get<float>();auto work=Self().orientationWork_.template Get<float>();
        auto forward=work;auto reverse=work[CAP];
        float scale=0.0f;Self().VS();
        for (uint32_t i=0;i<Self().n_;++i) {const float x=Self().Abs(original.GetValue(i));if (x>scale) scale=x;}
        if (!(scale>0.0f)) scale=1.0f;
        for (uint32_t r=Self().begin_;r<Self().end_;++r) {
            const Wide target(Self().meta_.template Get<float>().GetValue(r));
            const float eigenvalue=target.h;
            const float pivotFloor=scale*1.0e-12f;
            Self().ForwardPivots(eigenvalue,pivotFloor,forward);
            Self().BackwardPivots(eigenvalue,pivotFloor,reverse);
            const uint32_t twist=Self().MinPivotTwist(forward,reverse,eigenvalue);
            const float shift=target.h+scale*9.5367431640625e-7f;
            bool singular=true;float candidateShift=shift;
            for (uint32_t attempt=0;attempt<4U;++attempt) {
                if (Self().FloatFactors(candidateShift)) {singular=false;break;}
                candidateShift+=scale*9.5367431640625e-7f;
            }
            auto row=Self().in_.template AllocTensor<float>();AscendC::Duplicate(row,0.0f,Self().n_);Self().VS();row.SetValue(twist,1.0f);Self().SV();
            if (!singular) Self().InverseIterateEigenvector(row);
            const float norm=Self().Root(Self().Dot(row,row));if (norm>0.0f) AscendC::Muls(row,row,1.0f/norm,Self().n_);
            Self().SV();Self().in_.template EnQue<float>(row);row=Self().in_.template DeQue<float>();Self().Store(r,row,true);Self().in_.FreeTensor(row);Self().VS();
        }
    }
    // Forward continued-fraction pivots of the shifted tridiagonal.
    __aicore__ inline void ForwardPivots(float eigenvalue,float pivotFloor,AscendC::LocalTensor<float> forward) {
        auto original=Self().triOriginal_.template Get<float>();
        float pivot=original.GetValue(0)-eigenvalue;forward.SetValue(0U,pivot);
        for (uint32_t i=1;i<Self().n_;++i) {
            if (Self().Abs(pivot)<pivotFloor) pivot=pivot<0.0f?-pivotFloor:pivotFloor;
            const float off=original.GetValue(CAP+i-1U);
            const float divisor=pivot;
            pivot=(original.GetValue(i)-eigenvalue)-(divisor!=0.0f?(off*off)/divisor:0.0f);
            forward.SetValue(i,pivot);
        }
    }
    // Backward continued-fraction pivots of the shifted tridiagonal.
    __aicore__ inline void BackwardPivots(float eigenvalue,float pivotFloor,AscendC::LocalTensor<float> reverse) {
        auto original=Self().triOriginal_.template Get<float>();
        float pivot=original.GetValue(Self().n_-1U)-eigenvalue;reverse.SetValue(Self().n_-1U,pivot);
        for (uint32_t ii=Self().n_-1U;ii>0U;--ii) {
            const uint32_t i=ii-1U;
            if (Self().Abs(pivot)<pivotFloor) pivot=pivot<0.0f?-pivotFloor:pivotFloor;
            const float off=original.GetValue(CAP+i);
            const float divisor=pivot;
            pivot=(original.GetValue(i)-eigenvalue)-(divisor!=0.0f?(off*off)/divisor:0.0f);
            reverse.SetValue(i,pivot);
        }
    }
    // The twist index minimises the mismatch between both factorizations.
    __aicore__ inline uint32_t MinPivotTwist(AscendC::LocalTensor<float> forward, AscendC::LocalTensor<float> reverse,float eigenvalue) {
        auto original=Self().triOriginal_.template Get<float>();
        uint32_t twist=0U;
        float best=Self().Abs((forward.GetValue(0)+reverse.GetValue(0))-(original.GetValue(0)-eigenvalue));
        for (uint32_t i=1;i<Self().n_;++i) {
            const float gamma=Self().Abs((forward.GetValue(i)+reverse.GetValue(i))-(original.GetValue(i)-eigenvalue));
            if (gamma<best) {best=gamma;twist=i;}
        }
        return twist;
    }
    // Three inverse iterations on the pivoted factors of the current shift.
    __aicore__ inline void InverseIterateEigenvector(AscendC::LocalTensor<float> row) {
        auto rhs=Self().orientationWork_.template Get<float>()[4U*CAP];
        for (uint32_t iteration=0;iteration<3U;++iteration) {
            Self().VS();for (uint32_t i=0;i<Self().n_;++i) rhs.SetValue(i,row.GetValue(i));
            Self().SolveFloatFactors(rhs);
            Self().SV();const float max=Self().MaxAbs(rhs);if (!(max>0.0f)) break;
            AscendC::Muls(row,rhs,1.0f/max,Self().n_);
        }
    }
    __aicore__ inline void DefectCorrect(AscendC::LocalTensor<float> row,uint32_t eigenIndex) {
        auto original=Self().triOriginal_.template Get<float>();
        Wide target=Self().Rayleigh(row);Self().lastRayleigh_=target;float scale=0.0f;
        for (uint32_t i=0;i<Self().n_;++i) {const float x=Self().Abs(original.GetValue(i));if (x>scale) scale=x;}
        if (!(scale>0.0f)) return;
        const float shift=(target.h+target.l)+scale*7.62939453125e-6f;
        bool factored=false;float candidateShift=shift;
        for (uint32_t attempt=0;attempt<4U;++attempt) {
            if (Self().FloatFactors(candidateShift)) {factored=true;break;}
            candidateShift+=scale*9.5367431640625e-7f;
        }
        if (!factored) return;
        Self().RefineResidual(row,target);
        target=Self().Rayleigh(row);Self().lastRayleigh_=target;Self().meta_.template Get<float>().SetValue(eigenIndex,target.h+target.l);Self().SV();
    }
    // One cut of the twisted descent: 0 descends, 1 defers to the refined
    // path, 2 means a negative projection and 3 a positive one.
    __aicore__ inline int32_t TwistedCut(uint32_t &lo,uint32_t &hi,float poleScale,float &leftShift,float &rightShift) {
        auto original=Self().triOriginal_.template Get<float>();
        WideView vector(Self().orientationWork_.template Get<float>());
        const uint32_t middle=(lo+hi)/2U;const float rho=original.GetValue(CAP+middle-1U);
        const auto cut=Self().MeasureCluster(vector,lo,hi,rho,poleScale);
        const Wide projection=cut.projection,boundary=cut.boundary;
        const bool fromLeft=cut.fromLeft;const float tolerance=cut.tolerance;
        if (boundary>Wide(tolerance)) {
            const Wide projectionScale=Self().Abs(vector.GetValue(middle-1U))+Self().Abs(vector.GetValue(middle));
            // Preserve the compensated inverse path for ambiguous cancellation.
            if (Self().Abs(projection)<=Wide(2.0e-5f)*projectionScale || boundary<Wide(tolerance*1.001f)) return 1;
            return projection>Wide(0)?2:3;
        }
        if (boundary>Wide(tolerance*0.999f)) return 1;
        if (fromLeft) {hi=middle;rightShift=Self().Abs(rho);}
        else {lo=middle;leftShift=Self().Abs(rho);}
        return 0;
    }
    __aicore__ inline float TwistedOrientation(AscendC::LocalTensor<float> row) {
        WideView vector(Self().orientationWork_.template Get<float>());Self().VS();
        for (uint32_t i=0;i<Self().n_;++i) vector.SetValue(i,Wide(row.GetValue(i)));
        // Follow the Cuppen tree only when a joining component is numerically
        // negligible. An active positive rank-one update has negative projection.
        uint32_t lo=0U,hi=0U;
        Self().LocateCluster(lo,hi);
        auto eigenvalues=Self().meta_.template Get<float>();
        float poleScale=1.0f,leftShift=0.0f,rightShift=0.0f;
        for (uint32_t i=0;i<Self().n_;++i) {
            const float x=Self().Abs(eigenvalues.GetValue(i)+Self().referenceCenter_);
            if (x>poleScale) poleScale=x;
        }
        while (hi-lo>25U) {
            const int32_t cut=Self().TwistedCut(lo,hi,poleScale,leftShift,rightShift);
            if (cut==1) return Self().RefinedOrientation(row);
            if (cut==2) return -1.0f;
            if (cut==3) return 1.0f;
        }
        return Self().LeafOrientationSign(lo,hi,Self().lastRayleigh_+Wide(Self().referenceCenter_),leftShift,rightShift);
    }
    // Fill the leaf matrix from the selected triangle and clear the pair.
    __aicore__ inline void LoadLeafMatrix(uint32_t batch) {
        WideView matrix(Self().leafMatrix_.template Get<float>()),meta(Self().leafMeta_.template Get<float>());
        auto d=meta;auto e=meta[CAP];
        Self().VS();
        const uint64_t base=static_cast<uint64_t>(batch)*Self().n_*Self().n_;
        const float scale=Self().matrixScale_!=0.0f?Self().matrixScale_:1.0f;
        for (uint32_t r=0;r<Self().n_;++r) for (uint32_t c=0;c<Self().n_;++c) {
            const uint32_t rr=Self().lower_?(r>=c?r:c):(r<=c?r:c);
            const uint32_t cc=Self().lower_?(r>=c?c:r):(r<=c?c:r);
            matrix.SetValue(r*Self().n_+c,Wide(Self().a_.GetValue(base+rr*Self().n_+cc))/Wide(scale));
        }
        for (uint32_t i=0;i<Self().n_;++i) {d.SetValue(i,Wide(0));e.SetValue(i,Wide(0));}
    }
};
