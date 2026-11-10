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

template<class DT_A> class KernelPooledEigh : private EighSpectral<KernelPooledEigh<DT_A>> {
public:
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR values,GM_ADDR vectors,GM_ADDR workspace,
        const LinalgEighTilingData &t) {
        this->n_=t.n;this->pitch_=t.pitch;this->batch_=t.batch;this->lower_=t.lower;this->resident_=t.resident;
        this->core_=static_cast<uint32_t>(AscendC::GetBlockIdx());
        groups_=t.groups;group_=this->core_%groups_;workspaceStride_=t.workspaceStride;
        blocks_=static_cast<uint32_t>(AscendC::GetBlockNum());
        this->a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a),static_cast<uint64_t>(this->batch_)*this->n_*this->n_);
        this->values_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(values),static_cast<uint64_t>(this->batch_)*this->n_);
        this->vectors_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vectors),static_cast<uint64_t>(this->batch_)*this->n_*this->n_);
        workspaceBase_=reinterpret_cast<__gm__ float*>(AscendC::GetUserWorkspace(workspace));
        BindMatrix(group_);
        this->InitializeBuffers(this->n_<=25U,this->resident_,true);
        if (!this->resident_ && this->n_<=128U) this->pipe_.InitBuffer(this->sCache_,this->n_*this->pitch_*4U);
    }
    __aicore__ inline void Process() {
        const uint32_t sectors=(this->n_+7U)/8U;
        const uint32_t epochs=(this->batch_+groups_-1U)/groups_;
        for (uint32_t epoch=0;epoch<epochs;++epoch) {
            const uint32_t base=epoch*groups_;
            const uint32_t active=this->batch_-base<groups_?this->batch_-base:groups_;
            if (this->core_<active) {
                BindMatrix(this->core_);reductionCached_=this->n_<=128U;
                if (reductionCached_) AscendC::Duplicate(this->sCache_.template Get<float>(),0.0f,this->n_*this->pitch_);
                this->Prepare(base+this->core_);this->Tridiagonalize();
                if (reductionCached_) {
                    auto ev=this->pipe_.FetchEventID(AscendC::HardEvent::V_MTE3);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ev);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ev);
                    AscendC::DataCopy(this->s_,this->sCache_.template Get<float>(),this->n_*this->pitch_);
                    reductionCached_=false;
                }
                valuesOnly_=true;this->SolveQL();valuesOnly_=false;this->PublishState();
            }
            AscendC::SyncAll<true>();
            const uint32_t jobs=active*sectors;
            uint32_t job=jobs*this->core_/blocks_;
            const uint32_t limit=jobs*(this->core_+1U)/blocks_;
            while (job<limit) {
                const uint32_t slot=job/sectors;
                uint32_t last=(slot+1U)*sectors;if (last>limit) last=limit;
                begin_=(job-slot*sectors)*8U;end_=(last-slot*sectors)*8U;
                if (end_>this->n_) end_=this->n_;
                BindMatrix(slot);this->ReadState(true);
                this->BuildEigenvectors();this->SecularOrientation();this->TiledBackTransform();
                this->Write(shared_,begin_,this->meta_.template Get<float>()[begin_],end_-begin_);
                job=last;
            }
            AscendC::SyncAll<true>();
            if (this->n_%8U==0U) {
                if (this->core_<active) {BindMatrix(this->core_);this->ReadState();this->Output(base+this->core_);}
            } else if (this->core_==0U) {
                // Dense odd-size outputs can share a physical 32B sector.
                for (uint32_t slot=0;slot<active;++slot) {
                    BindMatrix(slot);this->ReadState();this->Output(base+slot);
                }
            }
            AscendC::SyncAll<true>();
        }
    }
private:
    friend class EighCore<KernelPooledEigh<DT_A>>;
    friend class EighLeaf<KernelPooledEigh<DT_A>>;
    friend class EighSpectral<KernelPooledEigh<DT_A>>;
    __aicore__ inline bool UseSmall() {return this->n_<=25U;}
    __aicore__ inline bool UseResident() {return this->resident_;}
    __aicore__ inline bool UseCache(bool q) {return this->resident_ || (!q && reductionCached_);}
    __aicore__ inline bool ScaleWorkspace() {return true;}
    __aicore__ inline bool WantsVectors() {return !valuesOnly_;}
    __aicore__ inline uint32_t VectorBegin() {return begin_;}
    __aicore__ inline uint32_t VectorEnd() {return end_;}
    __aicore__ inline float FallbackOrientation(AscendC::LocalTensor<float> row) {return this->TwistedOrientation(row);}
    __aicore__ inline bool ResidualConverged(AscendC::LocalTensor<float>,uint32_t) {return false;}
    __aicore__ inline bool KeepSpectrum() {return true;}
    __aicore__ inline void QueueRotation(uint32_t i,float c,float s) {this->Rotate(i,c,s);}
    __aicore__ inline void FinishRotations(uint32_t,uint32_t,bool) {}
    __aicore__ inline void BindMatrix(uint32_t slot) {
        group_=slot;const uint64_t offset=static_cast<uint64_t>(slot)*workspaceStride_;
        this->s_.SetGlobalBuffer(workspaceBase_+offset,this->n_*this->pitch_);
        this->qt_.SetGlobalBuffer(workspaceBase_+offset+this->n_*this->pitch_,this->n_*this->pitch_);
        shared_.SetGlobalBuffer(workspaceBase_+offset+2U*this->n_*this->pitch_,6U*CAP+16U);
    }

    AscendC::GlobalTensor<float> shared_;
    uint32_t groups_=1,group_=0,begin_=0,end_=0,blocks_=1,workspaceStride_=0;
    __gm__ float *workspaceBase_=nullptr;
    bool reductionCached_=false,valuesOnly_=false;
    Wide lastRayleigh_;
};
