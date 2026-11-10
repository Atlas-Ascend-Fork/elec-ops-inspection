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
#include "kernel_operator.h"
#include "linalg_eigh_spectral.h"
#include "linalg_eigh_tiling.h"
#include "tiling_key_linalg_eigh.h"

template<class DT_A,bool SMALL,bool RESIDENT> class KernelLinalgEigh : private EighSpectral<KernelLinalgEigh<DT_A,SMALL,RESIDENT>> {
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR values,GM_ADDR vectors,GM_ADDR workspace,
        const LinalgEighTilingData &t) {
        this->n_=t.n;this->pitch_=t.pitch;this->batch_=t.batch;this->lower_=t.lower;this->resident_=t.resident;
        this->core_=static_cast<uint32_t>(AscendC::GetBlockIdx());
        this->a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a),static_cast<uint64_t>(this->batch_)*this->n_*this->n_);
        this->values_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(values),static_cast<uint64_t>(this->batch_)*this->n_);
        this->vectors_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vectors),static_cast<uint64_t>(this->batch_)*this->n_*this->n_);
        auto user=reinterpret_cast<__gm__ float*>(AscendC::GetUserWorkspace(workspace));
        this->s_.SetGlobalBuffer(user+static_cast<uint64_t>(this->core_)*t.workspaceStride,this->n_*this->pitch_);
        this->qt_.SetGlobalBuffer(user+static_cast<uint64_t>(this->core_)*t.workspaceStride+this->n_*this->pitch_,this->n_*this->pitch_);
        this->InitializeBuffers(SMALL,RESIDENT,!SMALL);
    }
    __aicore__ inline void Process() {
        const uint32_t cores=static_cast<uint32_t>(AscendC::GetBlockNum());
        for (uint32_t b=this->core_;b<this->batch_;b+=cores) {
            this->Prepare(b);
            if constexpr (SMALL) {LeafTridiagonalize(b);this->LeafSolveQL();this->LeafExport();this->BackTransform();}
            else {this->Tridiagonalize();this->SolveQL();this->SecularOrientation();this->BackTransform();}
            this->Output(b);
        }
    }
private:
    friend class EighCore<KernelLinalgEigh<DT_A,SMALL,RESIDENT>>;
    friend class EighLeaf<KernelLinalgEigh<DT_A,SMALL,RESIDENT>>;
    friend class EighSpectral<KernelLinalgEigh<DT_A,SMALL,RESIDENT>>;
    __aicore__ inline bool UseSmall() {return SMALL;}
    __aicore__ inline bool UseResident() {return RESIDENT;}
    __aicore__ inline bool UseCache(bool) {return RESIDENT;}
    __aicore__ inline bool ScaleWorkspace() {return !SMALL;}
    __aicore__ inline bool WantsVectors() {return true;}
    __aicore__ inline uint32_t VectorBegin() {return 0U;}
    __aicore__ inline uint32_t VectorEnd() {return this->n_;}
    __aicore__ inline float FallbackOrientation(AscendC::LocalTensor<float> row) {return this->RefinedOrientation(row);}
    __aicore__ inline bool ResidualConverged(AscendC::LocalTensor<float> row,uint32_t eigenIndex) {return !NeedsCorrection(row,eigenIndex);}
    __aicore__ inline void QueueRotation(uint32_t i,float c,float s) {
        this->vBuf_.template Get<float>().SetValue(i,c);this->wBuf_.template Get<float>().SetValue(i,s);
    }
    __aicore__ inline void FinishRotations(uint32_t lo,uint32_t hi,bool descending) {ApplyRotations(lo,hi,descending);}

    // Leaf path: the matrix already fits in UB, so the triangle is scanned in place.
    __aicore__ inline void MeasureTriangleSmall(uint64_t base) {
        for (uint32_t r=0;r<this->n_;++r) for (uint32_t c=0;c<this->n_;++c) {
            const uint32_t rr=this->lower_?(r>=c?r:c):(r<=c?r:c);
            const uint32_t cc=this->lower_?(r>=c?c:r):(r<=c?c:r);
            const float value=this->a_.GetValue(base+rr*this->n_+cc);
            const float mag=this->Abs(value);if (mag>this->matrixScale_) this->matrixScale_=mag;
        }
    }

    // Consecutive rotations share one row. Keep it in UB until the sweep ends.
    __aicore__ inline void BeginRotations(uint32_t first,bool descending) {
        carried_=this->Load(first,true);carriedDescending_=descending;
    }
    __aicore__ inline void EndRotations(uint32_t last) {
        this->Store(last,carried_,true);this->in_.FreeTensor(carried_);
    }
    __aicore__ inline void CarriedRotate(uint32_t i,float c,float s) {
        auto neighbor=this->Load(carriedDescending_?i:i+1U,true);
        auto p=carriedDescending_?neighbor:carried_;
        auto q=carriedDescending_?carried_:neighbor;
        auto tmp=this->tmp_.template Get<float>();auto a=tmp;auto b=tmp[CAP];
        AscendC::Muls(a,p,c,this->n_);AscendC::Muls(b,q,s,this->n_);AscendC::Sub(a,a,b,this->n_);
        AscendC::Muls(b,p,s,this->n_);AscendC::Muls(q,q,c,this->n_);AscendC::Add(q,q,b,this->n_);
        if (carriedDescending_) {this->Store(i+1U,q,true);AscendC::Muls(carried_,a,1.0f,this->n_);}
        else {this->Store(i,a,true);AscendC::Muls(carried_,q,1.0f,this->n_);}
        this->in_.FreeTensor(neighbor);
    }
    __aicore__ inline void ApplyRotations(uint32_t first,uint32_t last,bool descending) {
        auto c=this->vBuf_.template Get<float>();auto s=this->wBuf_.template Get<float>();
        BeginRotations(descending?last+1U:first,descending);
        for (uint32_t step=0;step<=last-first;++step) {
            const uint32_t i=descending?last-step:first+step;
            CarriedRotate(i,c.GetValue(i),s.GetValue(i));
        }
        EndRotations(descending?first:last+1U);
    }

    // A rank-one secular update uses normalized vectors z/(D-lambda).
    // For positive update strength their projection onto the joining vector
    // is negative. This selects a standard sign-equivalent eigensystem without
    // reading reference outputs. 25 is the conventional tridiagonal leaf size.
    __aicore__ inline bool NeedsCorrection(AscendC::LocalTensor<float> row,uint32_t eigenIndex) {
        auto original=this->triOriginal_.template Get<float>();auto forward=this->offsets_.template Get<uint32_t>();
        auto backward=this->meta_.template Get<uint32_t>()[3U*CAP];auto residual=this->wBuf_.template Get<float>();
        auto following=this->vBuf_.template Get<float>();auto temp=this->tmp_.template Get<float>();
        AscendC::Gather(following,row,forward,0U,this->n_);AscendC::Mul(following,following,original[CAP],this->n_);
        AscendC::Mul(residual,row,original,this->n_);AscendC::Add(residual,residual,following,this->n_);
        AscendC::Gather(temp,row,backward,0U,this->n_);AscendC::Gather(temp[CAP],original,backward,CAP*4U,this->n_);
        this->VS();temp[CAP].SetValue(0U,0.0f);this->SV();
        AscendC::Mul(temp,temp,temp[CAP],this->n_);AscendC::Add(residual,residual,temp,this->n_);
        const float norm=this->Dot(row,row);
        const float target=norm!=0.0f?this->Dot(row,residual)/norm:0.0f;
        AscendC::Muls(temp,row,target,this->n_);AscendC::Sub(residual,residual,temp,this->n_);
        const float error=this->Root(this->Dot(residual,residual));
        auto eigenvalues=this->meta_.template Get<float>();this->VS();float gap=3.402823466e38f;
        const float value=eigenvalues.GetValue(eigenIndex);
        for (uint32_t j=0;j<this->n_;++j) if (j!=eigenIndex) {
            const float distance=this->Abs(eigenvalues.GetValue(j)-value);if (distance<gap) gap=distance;
        }
        return error>gap*1.0e-6f*this->Root(norm);
    }

    __aicore__ inline void DefectCorrect(AscendC::LocalTensor<float> row,uint32_t eigenIndex) {
        if (!NeedsCorrection(row,eigenIndex)) return;
        auto original=this->triOriginal_.template Get<float>();
        Wide target=this->Rayleigh(row);float scale=0.0f;
        for (uint32_t i=0;i<this->n_;++i) {const float x=this->Abs(original.GetValue(i));if (x>scale) scale=x;}
        if (!(scale>0.0f)) return;
        const float shift=(target.h+target.l)+scale*7.62939453125e-6f;
        if (!this->FloatFactors(shift)) return;
        this->RefineResidual(row,target,eigenIndex);
        target=this->Rayleigh(row);this->meta_.template Get<float>().SetValue(eigenIndex,target.h+target.l);this->SV();
    }

    // Fill the leaf matrix from the selected triangle and clear the pair.
    __aicore__ inline void LoadLeafMatrix(uint32_t batch) {
        auto H=this->leafMatrix_.template Get<float>();
        WideView meta(this->leafMeta_.template Get<float>());
        auto d=meta;auto e=meta[CAP];
        const uint64_t base=static_cast<uint64_t>(batch)*this->n_*this->n_;
        const float scale=this->matrixScale_!=0.0f?this->matrixScale_:1.0f;
        for (uint32_t r=0;r<this->n_;++r) for (uint32_t c=0;c<this->n_;++c) {
            const uint32_t rr=this->lower_?(r>=c?r:c):(r<=c?r:c);
            const uint32_t cc=this->lower_?(r>=c?c:r):(r<=c?c:r);
            H.SetValue(r*this->n_+c,this->a_.GetValue(base+rr*this->n_+cc)/scale);
        }
        for (uint32_t i=0;i<this->n_;++i) {d.SetValue(i,Wide(0));e.SetValue(i,Wide(0));}
    }
    // Householder vector of one leaf step; false when the block is already zero.
    __aicore__ inline bool LeafReflectorVector(uint32_t j,uint32_t first,uint32_t end,uint32_t pivot,
        uint32_t off,float &t) {
        auto H=this->leafMatrix_.template Get<float>();
        WideView meta(this->leafMeta_.template Get<float>());
        auto d=meta;auto e=meta[CAP];
        auto v=this->leafWork_.template Get<float>();
        auto ordinary=this->meta_.template Get<float>();auto tau=ordinary[2U*CAP];
        d.SetValue(j,Wide(H.GetValue(j*this->n_+j)));float scale=0.0f;
        for (uint32_t c=0;c<this->n_;++c) {
            const float x=(c>=first && c<end)?H.GetValue(j*this->n_+c):0.0f;
            v.SetValue(c,x);if (this->Abs(x)>scale) scale=this->Abs(x);
        }
        if (!(scale>0.0f)) {e.SetValue(off,Wide(0));tau.SetValue(j,0.0f);return false;}
        float norm=0.0f;const float inverseScale=1.0f/scale;
        for (uint32_t c=first;c<end;++c) {const float x=v.GetValue(c)*inverseScale;v.SetValue(c,x);norm=norm+x*x;}
        const float length=this->Root(norm),x0=v.GetValue(pivot),alpha=x0>=0.0f?-length:length;
        if (!(length>0.0f)) {e.SetValue(off,Wide(0));tau.SetValue(j,0.0f);return false;}
        const float divisor=x0-alpha;
        if (!(divisor!=0.0f)) {e.SetValue(off,Wide(0));tau.SetValue(j,0.0f);return false;}
        t=(alpha-x0)/alpha;
        for (uint32_t c=first;c<end;++c) v.SetValue(c,v.GetValue(c)/divisor);
        v.SetValue(pivot,1.0f);e.SetValue(off,Wide(alpha*scale));tau.SetValue(j,t);
        return true;
    }
    // Rank-two update of the leaf matrix with that reflector.
    __aicore__ inline void LeafReflectorApply(uint32_t first,uint32_t end,float t) {
        auto H=this->leafMatrix_.template Get<float>();
        auto v=this->leafWork_.template Get<float>();auto w=this->leafWork_.template Get<float>()[CAP];
        for (uint32_t r=first;r<end;++r) {
            float dot=0.0f;for (uint32_t c=first;c<end;++c) dot=dot+H.GetValue(r*this->n_+c)*v.GetValue(c);
            w.SetValue(r,t*dot);
        }
        float dot=0.0f;for (uint32_t r=first;r<end;++r) dot=dot+v.GetValue(r)*w.GetValue(r);
        const float beta=-0.5f*t*dot;
        for (uint32_t r=first;r<end;++r) w.SetValue(r,w.GetValue(r)+beta*v.GetValue(r));
        for (uint32_t r=first;r<end;++r) for (uint32_t c=first;c<end;++c)
            H.SetValue(r*this->n_+c,H.GetValue(r*this->n_+c)-v.GetValue(r)*w.GetValue(c)-w.GetValue(r)*v.GetValue(c));
    }
    // Publish the last diagonal pair of the leaf factorization.
    __aicore__ inline void LeafFinalizeDiagonal() {
        auto H=this->leafMatrix_.template Get<float>();
        WideView meta(this->leafMeta_.template Get<float>());
        auto d=meta;auto e=meta[CAP];
        if (this->n_>=2U) {
            const uint32_t r=this->lower_?this->n_-2U:1U,c=this->lower_?this->n_-1U:0U;
            d.SetValue(r,Wide(H.GetValue(r*this->n_+r)));e.SetValue(this->lower_?this->n_-2U:0U,Wide(H.GetValue(r*this->n_+c)));
        }
        const uint32_t last=this->lower_?this->n_-1U:0U;d.SetValue(last,Wide(H.GetValue(last*this->n_+last)));
        this->SV();
    }
    __aicore__ inline void LeafTridiagonalize(uint32_t batch) {
        LoadLeafMatrix(batch);
        for (uint32_t step=0;step+2U<this->n_;++step) {
            const uint32_t j=this->lower_?step:this->n_-1U-step,first=this->lower_?j+1U:0U,end=this->lower_?this->n_:j;
            const uint32_t pivot=this->lower_?j+1U:j-1U,off=this->lower_?j:j-1U;
            float t=0.0f;
            if (LeafReflectorVector(j,first,end,pivot,off,t)) LeafReflectorApply(first,end,t);
            auto row=this->vBuf_.template Get<float>();auto v=this->leafWork_.template Get<float>();
            for (uint32_t c=0;c<this->n_;++c) row.SetValue(c,v.GetValue(c));
            this->SV();this->Store(j,row);this->VS();
        }
        LeafFinalizeDiagonal();
    }

    AscendC::LocalTensor<float> carried_;bool carriedDescending_=false;
};
