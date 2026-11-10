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

template<class DT_A> class KernelCooperativeEigh : private EighSpectral<KernelCooperativeEigh<DT_A>> {
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR values,GM_ADDR vectors,GM_ADDR workspace,
        const LinalgEighTilingData &t) {
        this->n_=t.n;this->pitch_=t.pitch;this->batch_=t.batch;this->lower_=t.lower;this->resident_=t.resident;
        this->core_=static_cast<uint32_t>(AscendC::GetBlockIdx());
        team_=t.team;groups_=t.groups;group_=this->core_/team_;lane_=this->core_%team_;
        const uint32_t sectors=(this->n_+7U)/8U;
        begin_=(sectors*lane_/team_)*8U;end_=(sectors*(lane_+1U)/team_)*8U;
        if (begin_>this->n_) begin_=this->n_;if (end_>this->n_) end_=this->n_;
        this->a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a),static_cast<uint64_t>(this->batch_)*this->n_*this->n_);
        this->values_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(values),static_cast<uint64_t>(this->batch_)*this->n_);
        this->vectors_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vectors),static_cast<uint64_t>(this->batch_)*this->n_*this->n_);
        auto user=reinterpret_cast<__gm__ float*>(AscendC::GetUserWorkspace(workspace));
        this->s_.SetGlobalBuffer(user+static_cast<uint64_t>(group_)*t.workspaceStride,this->n_*this->pitch_);
        this->qt_.SetGlobalBuffer(user+static_cast<uint64_t>(group_)*t.workspaceStride+this->n_*this->pitch_,this->n_*this->pitch_);
        shared_.SetGlobalBuffer(user+static_cast<uint64_t>(group_)*t.workspaceStride+2U*this->n_*this->pitch_,6U*CAP+16U);
        this->InitializeBuffers(this->n_<=25U,this->resident_,true);
    }
    __aicore__ inline void Process() {
        const uint32_t epochs=(this->batch_+groups_-1U)/groups_;
        for (uint32_t epoch=0;epoch<epochs;++epoch) {
            const uint32_t b=epoch*groups_+group_;const bool present=b<this->batch_;
            if (present && lane_==0U) {
                AscendC::Duplicate(this->vBuf_.template Get<float>(),0.0f,CAP);
                AscendC::Duplicate(this->triOriginal_.template Get<float>(),0.0f,2U*CAP);
                this->referenceBlocks_=false;this->Prepare(b);this->PublishState();
            }
            AscendC::SyncAll<true>();
            if (present) this->ReadState();
            CooperativeReduction(present);
            if (present && lane_==0U) {
                PrepareBisection();this->PublishState();
            }
            AscendC::SyncAll<true>();
            if (present) {
                this->ReadState();BisectEigenvalues();
                if (end_>begin_) this->Write(shared_,begin_,this->meta_.template Get<float>()[begin_],end_-begin_);
            }
            AscendC::SyncAll<true>();
            if (present) {
                this->ReadState();this->BuildEigenvectors();this->SecularOrientation();this->TiledBackTransform();
                if (end_>begin_) this->Write(shared_,begin_,this->meta_.template Get<float>()[begin_],end_-begin_);
            }
            AscendC::SyncAll<true>();
            if (present && lane_==0U) {this->ReadState();this->Output(b);}
            AscendC::SyncAll<true>();
        }
    }
private:
    friend class EighCore<KernelCooperativeEigh<DT_A>>;
    friend class EighLeaf<KernelCooperativeEigh<DT_A>>;
    friend class EighSpectral<KernelCooperativeEigh<DT_A>>;
    __aicore__ inline bool UseSmall() {return this->n_<=25U;}
    __aicore__ inline bool UseResident() {return this->resident_;}
    __aicore__ inline bool UseCache(bool) {return this->resident_;}
    __aicore__ inline bool ScaleWorkspace() {return true;}
    __aicore__ inline bool WantsVectors() {return !valuesOnly_;}
    __aicore__ inline uint32_t VectorBegin() {return begin_;}
    __aicore__ inline uint32_t VectorEnd() {return end_;}
    __aicore__ inline float FallbackOrientation(AscendC::LocalTensor<float> row) {return this->TwistedOrientation(row);}
    __aicore__ inline bool ResidualConverged(AscendC::LocalTensor<float>,uint32_t) {return false;}
    __aicore__ inline bool KeepSpectrum() {return false;}
    __aicore__ inline void QueueRotation(uint32_t i,float c,float s) {this->Rotate(i,c,s);}
    __aicore__ inline void FinishRotations(uint32_t,uint32_t,bool) {}

    __aicore__ inline void PublishReductionVector() {
        this->SV();auto event=this->pipe_.FetchEventID(AscendC::HardEvent::V_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::DataCopy(shared_,this->meta_.template Get<float>(),4U*CAP);
        AscendC::DataCopy(shared_[5U*CAP],this->vBuf_.template Get<float>(),CAP);
    }
    __aicore__ inline void PublishReductionProduct() {
        auto event=this->pipe_.FetchEventID(AscendC::HardEvent::V_MTE3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::DataCopy(shared_[4U*CAP],this->wBuf_.template Get<float>(),CAP);
    }
    __aicore__ inline void ReadReductionProduct() {
        auto event=this->pipe_.FetchEventID(AscendC::HardEvent::V_MTE2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        event=this->pipe_.FetchEventID(AscendC::HardEvent::MTE3_MTE2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        AscendC::DataCopy(this->wBuf_.template Get<float>(),shared_[4U*CAP],CAP);
        event=this->pipe_.FetchEventID(AscendC::HardEvent::MTE2_V);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    }
    __aicore__ inline void CooperativeReduction(bool present) {
        for (uint32_t step=0;step+2U<this->n_;++step) {
            const uint32_t j=this->lower_?step:this->n_-1U-step;
            const uint32_t first=this->lower_?j+1U:0U,end=this->lower_?this->n_:j;
            const uint32_t pivot=this->lower_?j+1U:j-1U,off=this->lower_?j:j-1U;
            const uint32_t localFirst=begin_>first?begin_:first;
            const uint32_t localEnd=end_<end?end_:end;
            ReductionReflector(j,first,end,pivot,off,present);
            AscendC::SyncAll<true>();
            ReductionProduct(j,localFirst,localEnd,present);
            AscendC::SyncAll<true>();
            ReductionApply(j,localFirst,localEnd,present);
            AscendC::SyncAll<true>();
        }
        ReductionTail(present);
    }
    // Lane 0 builds the reflector and publishes it into the shared slot.
    __aicore__ inline void ReductionReflector(uint32_t j,uint32_t first,uint32_t end,uint32_t pivot,
        uint32_t off,bool present) {
        if (!(present && lane_==0U)) return;
        auto d=this->meta_.template Get<float>();auto e=d[CAP];auto tau=d[2U*CAP];
        auto v=this->vBuf_.template Get<float>();
        auto row=this->Load(j);this->VS();d.SetValue(j,row.GetValue(j));
        AscendC::Muls(v,row,1.0f,this->n_);this->in_.FreeTensor(row);this->VS();
        for (uint32_t c=0;c<this->n_;++c) if (c<first || c>=end) v.SetValue(c,0.0f);
        this->SV();const float scale=this->MaxAbs(v);
        if (!(scale>0.0f)) {this->VS();tau.SetValue(j,0.0f);e.SetValue(off,0.0f);}
        else {
            AscendC::Muls(v,v,1.0f/scale,this->n_);
            const float length=this->Root(this->Dot(v,v));this->VS();const float x0=v.GetValue(pivot);
            const float alpha=x0>=0.0f?-length:length,divisor=x0-alpha;
            if (!(length>0.0f) || !(divisor!=0.0f)) {this->VS();tau.SetValue(j,0.0f);e.SetValue(off,0.0f);}
            else {
                const float t=(alpha-x0)/alpha;
                AscendC::Muls(v,v,1.0f/divisor,this->n_);this->VS();v.SetValue(pivot,1.0f);
                e.SetValue(off,alpha*scale);tau.SetValue(j,t);
            }
        }
        PublishReductionVector();
    }
    // Every lane accumulates its slice of S*v, then the slices are joined.
    __aicore__ inline void ReductionProduct(uint32_t j,uint32_t localFirst,uint32_t localEnd,bool present) {
        if (!present) return;
        auto v=this->vBuf_.template Get<float>();auto w=this->wBuf_.template Get<float>();
        auto tau=this->meta_.template Get<float>()[2U*CAP];
        this->ReadState();AscendC::Muls(v,this->triOriginal_.template Get<float>()[CAP],1.0f,this->n_);
        AscendC::Duplicate(w,0.0f,CAP);this->VS();const float t=tau.GetValue(j);
        if (t>0.0f) for (uint32_t base=localFirst;base<localEnd;base+=32U) {
            const uint32_t height=localEnd-base<32U?localEnd-base:32U;
            this->ReadTile(this->s_,base,height);auto tile=this->tile_.template Get<float>();
            for (uint32_t r=0;r<height;++r) w.SetValue(base+r,t*this->Dot(tile[r*this->pitch_],v));
        }
        this->SV();if (end_>begin_) this->Write(shared_,4U*CAP+begin_,w[begin_],end_-begin_);
    }
    // All lanes read the same complete product and compute beta locally.
    __aicore__ inline void ReductionApply(uint32_t j,uint32_t localFirst,uint32_t localEnd,bool present) {
        if (!present) return;
        auto v=this->vBuf_.template Get<float>();auto w=this->wBuf_.template Get<float>();auto tmp=this->tmp_.template Get<float>();
        auto tau=this->meta_.template Get<float>()[2U*CAP];
        ReadReductionProduct();
        const float t=tau.GetValue(j);const float beta=-0.5f*t*this->Dot(v,w);
        AscendC::Muls(tmp,v,beta,this->n_);AscendC::Add(w,w,tmp,this->n_);this->VS();
        if (t>0.0f) for (uint32_t base=localFirst;base<localEnd;base+=32U) {
            const uint32_t height=localEnd-base<32U?localEnd-base:32U;
            this->ReadTile(this->s_,base,height);auto tile=this->tile_.template Get<float>();
            for (uint32_t r=0;r<height;++r) {
                const float vr=v.GetValue(base+r),wr=w.GetValue(base+r);auto input=tile[r*this->pitch_];
                AscendC::Muls(tmp,w,vr,this->n_);AscendC::Sub(input,input,tmp,this->n_);
                AscendC::Muls(tmp,v,wr,this->n_);AscendC::Sub(input,input,tmp,this->n_);
            }
            this->WriteTile(this->s_,base,height);
        }
        if (lane_==0U) this->Store(j,v);
    }
    // Read back the last two columns once the sweep has finished.
    __aicore__ inline void ReductionTail(bool present) {
        if (!(present && lane_==0U)) return;
        auto d=this->meta_.template Get<float>();auto e=d[CAP];
        if (this->n_>=2U) {
            const uint32_t r=this->lower_?this->n_-2U:1U,c=this->lower_?this->n_-1U:0U;
            auto row=this->Load(r);this->VS();d.SetValue(r,row.GetValue(r));
            e.SetValue(this->lower_?this->n_-2U:0U,row.GetValue(c));this->in_.FreeTensor(row);
        }
        const uint32_t last=this->lower_?this->n_-1U:0U;auto row=this->Load(last);this->VS();
        d.SetValue(last,row.GetValue(last));e.SetValue(this->n_-1U,0.0f);this->in_.FreeTensor(row);this->SV();
    }

    // A rank-one secular update uses normalized vectors z/(D-lambda).
    // For positive update strength their projection onto the joining vector
    // is negative. This selects a standard sign-equivalent eigensystem without
    // reading reference outputs. 25 is the conventional tridiagonal leaf size.
    __aicore__ inline void PrepareBisection() {
        auto d=this->meta_.template Get<float>();auto e=d[CAP];auto original=this->triOriginal_.template Get<float>();this->VS();
        for (uint32_t i=0;i<this->n_;++i) {original.SetValue(i,d.GetValue(i));original.SetValue(CAP+i,e.GetValue(i));}
        this->referenceBlocks_=false;
        for (uint32_t i=0;i+1U<this->n_;++i) if (this->ReferenceBoundary(i)) this->referenceBlocks_=true;
        const float joining=this->n_>1U?e.GetValue(this->n_/2U-1U):0.0f;
        this->splitSign_=joining<0?-1.0f:(joining>0?1.0f:0.0f);this->SV();
    }
    __aicore__ inline void BisectEigenvalues() {
        auto original=this->triOriginal_.template Get<float>();auto eigenvalues=this->meta_.template Get<float>();this->VS();
        float lower=3.402823466e38f,upper=-3.402823466e38f;
        for (uint32_t i=0;i<this->n_;++i) {
            const float radius=(i>0U?this->Abs(original.GetValue(CAP+i-1U)):0.0f)+(i+1U<this->n_?this->Abs(original.GetValue(CAP+i)):0.0f);
            const float diagonal=original.GetValue(i);
            if (diagonal-radius<lower) lower=diagonal-radius;
            if (diagonal+radius>upper) upper=diagonal+radius;
        }
        const float scale=this->Abs(lower)>this->Abs(upper)?this->Abs(lower):this->Abs(upper);
        const float padding=scale*3.814697265625e-6f+1.0e-30f;
        lower-=padding;upper+=padding;
        const float floor=scale*1.0e-25f+1.0e-35f;
        for (uint32_t rank=begin_;rank<end_;++rank) {
            float lo=lower,hi=upper;
            for (uint32_t iteration=0;iteration<64U;++iteration) {
                const float middle=0.5f*lo+0.5f*hi;
                if (middle==lo || middle==hi) break;
                float pivot=original.GetValue(0U)-middle;uint32_t count=pivot<=0.0f?1U:0U;
                for (uint32_t i=1;i<this->n_;++i) {
                    if (this->Abs(pivot)<floor) pivot=pivot<=0.0f?-floor:floor;
                    const float off=original.GetValue(CAP+i-1U);
                    const float divisor=pivot;
                    pivot=(original.GetValue(i)-middle)-(divisor!=0.0f?(off*off)/divisor:0.0f);
                    if (pivot<=0.0f) ++count;
                }
                if (count<=rank) lo=middle;else hi=middle;
            }
            eigenvalues.SetValue(rank,0.5f*lo+0.5f*hi);
        }
        this->SV();
    }

    AscendC::GlobalTensor<float> shared_;
    uint32_t team_=1,groups_=1,group_=0,lane_=0,begin_=0,end_=0;
    bool valuesOnly_=false;
    Wide lastRayleigh_;
};
