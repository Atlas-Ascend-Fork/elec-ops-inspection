/*
 * Copyright (c) 2026 Lesem.
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

namespace AscendC::SwiGluKernel {

template <bool DoubleA1 = false, bool OffsetZ = false, bool PadM = false, bool OneShotB = false, bool DualC0 = false, bool StagePrefetch = false>
class KernelDirectCube {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
        GM_ADDR b2, GM_ADDR y, GM_ADDR workspace, const SwiGluWorkTiling &t) {
        t_=t;
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
        area_=t_.tileM*t_.tileN;
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(GetUserWorkspace(workspace))+
            static_cast<uint64_t>(GetBlockIdx())*4U*area_);
        if ASCEND_IS_AIC {
            pipe_.InitBuffer(a1_,t_.tileM*t_.alignedDim*(DoubleA1 ? 4U : 2U));
            pipe_.InitBuffer(b11_,t_.tileN*t_.alignedDim*2U);
            pipe_.InitBuffer(b12_,t_.tileN*t_.alignedDim*2U);
            pipe_.InitBuffer(a0_,t_.tileM*t_.alignedDim*2U);
            pipe_.InitBuffer(b0_,t_.tileN*t_.alignedDim*(OneShotB ? 4U : 2U));
            pipe_.InitBuffer(c0_,area_*(DualC0 ? 8U : 4U));
            copyReady_=pipe_.AllocEventID<HardEvent::MTE2_MTE1>();
            if constexpr (DoubleA1) {
                a1Free_[0]=pipe_.AllocEventID<HardEvent::MTE1_MTE2>();
                a1Free_[1]=pipe_.AllocEventID<HardEvent::MTE1_MTE2>();
            } else if constexpr (!OneShotB) l1Free_=pipe_.AllocEventID<HardEvent::MTE1_MTE2>();
            loadReady_=pipe_.AllocEventID<HardEvent::MTE1_M>();
            if constexpr (!OneShotB) inputFree_=pipe_.AllocEventID<HardEvent::M_MTE1>();
            if constexpr (!PadM) {
                resultReady_=pipe_.AllocEventID<HardEvent::M_FIX>();
                resultFree_=pipe_.AllocEventID<HardEvent::FIX_M>();
            }
        }
        if ASCEND_IS_AIV {
            pipe_.InitBuffer(cu1_,area_*4U); pipe_.InitBuffer(cu2_,area_*2U+(OffsetZ ? 256U : 0U));
            pipe_.InitBuffer(zf_,area_*4U);
            pipe_.InitBuffer(bh1_,t_.tileN*2U); pipe_.InitBuffer(bh2_,t_.tileN*2U);
            m2v_=pipe_.AllocEventID<HardEvent::MTE2_V>();
            vm3_=pipe_.AllocEventID<HardEvent::V_MTE3>();
            m3v_=pipe_.AllocEventID<HardEvent::MTE3_V>();
            if constexpr (StagePrefetch) {
                stageFree_=pipe_.AllocEventID<HardEvent::V_MTE2>();
            } else {
                m3m2_=pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
            }
        }
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC { ProcessCube(); }
        if ASCEND_IS_AIV {
            if constexpr (StagePrefetch) ProcessStageVector();
            else ProcessVector();
        }
    }
private:
    static constexpr uint32_t ReadyFlag = StagePrefetch ? 0U : 8U;
    static constexpr uint32_t AckFlag = ReadyFlag + 4U;
    __aicore__ inline void Compute(LocalTensor<half> right,uint32_t mc,uint32_t branch,uint32_t slot) {
        auto a=a0_.Get<half>(); auto b=b0_.Get<half>(); auto c=c0_.Get<float>();
        if constexpr (OneShotB) b=b[branch*t_.tileN*t_.alignedDim];
        if constexpr (DualC0) c=c[branch*area_];
        LoadCubeB(t_.tileN, t_.alignedDim, b,right);
        SetFlag<HardEvent::MTE1_M>(loadReady_); WaitFlag<HardEvent::MTE1_M>(loadReady_);
        MmadParams p{}; p.m=PadM ? (mc+15U)/16U*16U : (mc==1U ? 16U : mc); p.n=t_.tileN; p.k=t_.alignedDim;
        const bool fineSync=PadM || mc%16U==0U;
        p.cmatrixInitVal=true;
        if(fineSync) { SetMMRowMajor(); p.unitFlag=3; }
        else {
            SetFlag<HardEvent::FIX_M>(resultFree_); WaitFlag<HardEvent::FIX_M>(resultFree_);
        }
        Mmad(c,a,b,p);
        if(!fineSync) {
            SetFlag<HardEvent::M_FIX>(resultReady_); WaitFlag<HardEvent::M_FIX>(resultReady_);
        }
        FixpipeParamsV220 f{}; f.nSize=t_.tileN; f.mSize=PadM ? (mc+15U)/16U*16U : mc;
        f.unitFlag=fineSync ? 3U : 0U;
        f.srcStride=(mc+15U)/16U*16U; f.dstStride=t_.tileN;
        f.ndNum=1; f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float>(scratch_[(slot*2U+branch)*area_],c,f);
        if constexpr (!OneShotB) {
            SetFlag<HardEvent::M_MTE1>(inputFree_); WaitFlag<HardEvent::M_MTE1>(inputFree_);
        }
        CrossCoreSetFlag<2,PIPE_FIX>(ReadyFlag+slot*2U+branch);
        if(!fineSync) {
            SetFlag<HardEvent::FIX_M>(resultFree_); WaitFlag<HardEvent::FIX_M>(resultFree_);
        }
    }
    __aicore__ inline void CopyInput(LocalTensor<half> dst, GlobalTensor<half> src, uint32_t mc) {
        CopyPaddedCubeInput(dst, src, mc, t_.tileM, t_.dim, t_.alignedDim);
    }
    __aicore__ inline void CopyCubeWeights(LocalTensor<half> b1, LocalTensor<half> b2, uint32_t n0) {
        const uint32_t nc=t_.dim-n0<t_.tileN ? t_.dim-n0 : t_.tileN;
        if(nc<t_.tileN) {
            InitConstValueParams<half> zero{1,
                static_cast<uint16_t>(t_.tileN*t_.alignedDim/16U),0,static_cast<half>(0)};
            InitConstValue(b1,zero); InitConstValue(b2,zero);
        }
        CopyCubeTileNz(t_.dim, b1,w1_[static_cast<uint64_t>(n0)*t_.dim],nc,t_.dim);
        CopyCubeTileNz(t_.dim, b2,w2_[static_cast<uint64_t>(n0)*t_.dim],nc,t_.dim);
    }

    __aicore__ inline void LoadAndPrefetchA(LocalTensor<half> a0, LocalTensor<half> a1,
        uint64_t task, uint32_t groups, uint32_t localTask, uint32_t mc) {
        if constexpr (DoubleA1) {
            if (groups == 0U) return;
            const uint32_t aSlot=localTask%2U;
            LoadCubeA(t_.alignedDim, a0,a1[aSlot*t_.tileM*t_.alignedDim],mc);
            SetFlag<HardEvent::MTE1_MTE2>(a1Free_[aSlot]);
            const uint64_t nextTask=task+t_.blocks;
            if(nextTask<t_.tasks) {
                const uint32_t nextSlot=(localTask+1U)%2U;
                if(localTask>=1U) WaitFlag<HardEvent::MTE1_MTE2>(a1Free_[nextSlot]);
                const uint64_t nextM=nextTask/groups*t_.tileM;
                const uint32_t nextMc=t_.rows-nextM<t_.tileM ? static_cast<uint32_t>(t_.rows-nextM) : t_.tileM;
                CopyInput(a1[nextSlot*t_.tileM*t_.alignedDim],x_[nextM*t_.dim],nextMc);
            }
        } else LoadCubeA(t_.alignedDim, a0,a1,mc);
    }

    __aicore__ inline void ProcessCube() {
        auto a1=a1_.Get<half>(); auto a0=a0_.Get<half>();
        auto b1=b11_.Get<half>(); auto b2=b12_.Get<half>();
        const uint32_t groups=(t_.dim+t_.tileN-1U)/t_.tileN;
        uint32_t cachedN=0xffffffffU;
        uint32_t localTask=0;
        for(uint64_t task=GetBlockIdx();task<t_.tasks;task+=t_.blocks,++localTask) {
            const uint32_t slot=localTask%2U;
            if(localTask>=2U) CrossCoreWaitFlag<2,PIPE_MTE2>(AckFlag+slot);
            if constexpr (!DoubleA1 && !OneShotB) {
                if(localTask>0U) WaitFlag<HardEvent::MTE1_MTE2>(l1Free_);
            }
            uint64_t m0=task/groups*t_.tileM;
            uint32_t n0=task%groups*t_.tileN;
            uint32_t mc=t_.rows-m0<t_.tileM ? static_cast<uint32_t>(t_.rows-m0) : t_.tileM;
            if constexpr (DoubleA1) {
                if(localTask==0U) CopyInput(a1,x_[m0*t_.dim],mc);
            } else CopyInput(a1,x_[m0*t_.dim],mc);
            if (n0 != cachedN) {
                CopyCubeWeights(b1, b2, n0);
                cachedN = n0;
            }
            SetFlag<HardEvent::MTE2_MTE1>(copyReady_); WaitFlag<HardEvent::MTE2_MTE1>(copyReady_);
            LoadAndPrefetchA(a0, a1, task, groups, localTask, mc);
            Compute(b1,mc,0,slot); Compute(b2,mc,1,slot);
            if constexpr (!DoubleA1 && !OneShotB) SetFlag<HardEvent::MTE1_MTE2>(l1Free_);
        }
        FinishCube(localTask);
    }

    __aicore__ inline void FinishCube(uint32_t localTask) {
        if constexpr (DoubleA1) {
            const uint32_t aPending=localTask<2U ? localTask : 2U;
            for(uint32_t i=localTask-aPending;i<localTask;++i)
                WaitFlag<HardEvent::MTE1_MTE2>(a1Free_[i%2U]);
        } else if constexpr (!OneShotB) {
            if(localTask>0U) WaitFlag<HardEvent::MTE1_MTE2>(l1Free_);
        }
        const uint32_t pending=localTask<2U ? localTask : 2U;
        for(uint32_t i=localTask-pending;i<localTask;++i)
            CrossCoreWaitFlag<2,PIPE_MTE2>(AckFlag+i%2U);
        if constexpr (DoubleA1) {
            pipe_.ReleaseEventID<HardEvent::MTE1_MTE2>(a1Free_[0]);
            pipe_.ReleaseEventID<HardEvent::MTE1_MTE2>(a1Free_[1]);
        } else if constexpr (!OneShotB) pipe_.ReleaseEventID<HardEvent::MTE1_MTE2>(l1Free_);
        pipe_.ReleaseEventID<HardEvent::MTE2_MTE1>(copyReady_);
        pipe_.ReleaseEventID<HardEvent::MTE1_M>(loadReady_);
        if constexpr (!OneShotB) pipe_.ReleaseEventID<HardEvent::M_MTE1>(inputFree_);
        if constexpr (!PadM) {
            pipe_.ReleaseEventID<HardEvent::M_FIX>(resultReady_);
            pipe_.ReleaseEventID<HardEvent::FIX_M>(resultFree_);
        }
    }
    __aicore__ inline void ProcessVector() {
        auto c1=cu1_.Get<half>(); auto c2=cu2_.Get<half>();
        auto z=zf_.Get<float>();
        auto b1=bh1_.Get<half>(); auto b2=bh2_.Get<half>();
        const uint32_t groups=(t_.dim+t_.tileN-1U)/t_.tileN;
        const bool fixedBias=t_.blocks%groups==0U;
        if(fixedBias) CopyCubeBias(t_, b1,b2,GetBlockIdx()%groups*t_.tileN, b1_, b2_);
        uint32_t localTask=0;
        for(uint64_t task=GetBlockIdx();task<t_.tasks;task+=t_.blocks,++localTask) {
            const uint32_t slot=localTask%2U;
            uint64_t m0=task/groups*t_.tileM; uint32_t n0=task%groups*t_.tileN;
            uint32_t mc=t_.rows-m0<t_.tileM ? static_cast<uint32_t>(t_.rows-m0) : t_.tileM;
            const uint32_t nc=t_.dim-n0<t_.tileN ? t_.dim-n0 : t_.tileN;
            if(!fixedBias) CopyCubeBias(t_, b1,b2,n0, b1_, b2_);
            FinishCubeProjectionPair<ReadyFlag, AckFlag>(
                c1, c2, z, b1, b2, scratch_, area_, slot, mc, t_.tileN, m2v_);
            StoreStridedCubeOutput(y_[m0 * t_.dim + n0], c1, mc, nc, t_.tileN, t_.dim,
                vm3_, m3v_, m3m2_);
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(m2v_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(vm3_);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(m3m2_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(m3v_);
    }
    __aicore__ inline void ProcessStageVector() {
        auto out=cu1_.Get<half>(); auto stage=cu2_.Get<half>();
        auto z=zf_.Get<float>(); auto neg=out.ReinterpretCast<float>();
        auto b1=bh1_.Get<half>(); auto b2=bh2_.Get<half>();
        const uint32_t groups=(t_.dim+t_.tileN-1U)/t_.tileN;
        const bool fixedBias=t_.blocks%groups==0U;
        if(fixedBias) CopyCubeBias(t_, b1,b2,GetBlockIdx()%groups*t_.tileN, b1_, b2_);
        // stage owns C1 until Cast, then C2 until Mul. Output has a separate drain.
        SetFlag<HardEvent::V_MTE2>(stageFree_);
        uint32_t localTask=0;
        for(uint64_t task=GetBlockIdx();task<t_.tasks;task+=t_.blocks,++localTask) {
            const uint32_t slot=localTask%2U;
            uint64_t m0=task/groups*t_.tileM; uint32_t n0=task%groups*t_.tileN;
            uint32_t mc=t_.rows-m0<t_.tileM ? static_cast<uint32_t>(t_.rows-m0) : t_.tileM;
            uint32_t count=mc*t_.tileN;
            const uint32_t nc=t_.dim-n0<t_.tileN ? t_.dim-n0 : t_.tileN;
            if(!fixedBias) CopyCubeBias(t_, b1,b2,n0, b1_, b2_);
            WaitFlag<HardEvent::V_MTE2>(stageFree_);
            CrossCoreWaitFlag<2,PIPE_MTE2>(ReadyFlag+slot*2U);
            DataCopy(stage,scratch_[slot*2U*area_],count);
            SetFlag<HardEvent::MTE2_V>(m2v_); WaitFlag<HardEvent::MTE2_V>(m2v_);
            BinaryRepeatParams r{1,1,1,static_cast<uint8_t>(t_.tileN/16U),
                static_cast<uint8_t>(t_.tileN/16U),0};
            Add(stage,stage,b1,static_cast<uint64_t>(t_.tileN),static_cast<uint8_t>(mc),r);
            PipeBarrier<PIPE_V>();
            Cast(z,stage,RoundMode::CAST_NONE,count); PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(stageFree_);
            WaitFlag<HardEvent::V_MTE2>(stageFree_);
            CrossCoreWaitFlag<2,PIPE_MTE2>(ReadyFlag+1U+slot*2U);
            DataCopy(stage,scratch_[(slot*2U+1U)*area_],count);
            CrossCoreSetFlag<2,PIPE_MTE2>(AckFlag+slot);
            SetFlag<HardEvent::MTE2_V>(m2v_);
            // Prior MTE3 reads only out; the next C1 read and Cast may precede this.
            if(localTask>0U) WaitFlag<HardEvent::MTE3_V>(m3v_);
            ApplySilu(z, neg, count);
            Cast(out,z,RoundMode::CAST_RINT,count); PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE2_V>(m2v_);
            Add(stage,stage,b2,static_cast<uint64_t>(t_.tileN),static_cast<uint8_t>(mc),r);
            PipeBarrier<PIPE_V>();
            Mul(out,out,stage,count);
            SetFlag<HardEvent::V_MTE2>(stageFree_);
            SetFlag<HardEvent::V_MTE3>(vm3_); WaitFlag<HardEvent::V_MTE3>(vm3_);
            DataCopyExtParams copyOut{static_cast<uint16_t>(mc),nc*2U,
                (t_.tileN-nc)*2U/32U,(t_.dim-nc)*2U,0};
            DataCopyPad(y_[m0*t_.dim+n0],out,copyOut);
            SetFlag<HardEvent::MTE3_V>(m3v_);
        }
        FinishStageVector(localTask);
    }

    __aicore__ inline void FinishStageVector(uint32_t localTask) {
        WaitFlag<HardEvent::V_MTE2>(stageFree_);
        if(localTask>0U) WaitFlag<HardEvent::MTE3_V>(m3v_);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(m2v_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(vm3_);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(stageFree_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(m3v_);
    }
    SwiGluWorkTiling t_; uint32_t area_;
    TPipe pipe_;
    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_,scratch_;
    TBuf<TPosition::A1> a1_;
    TBuf<TPosition::B1> b11_,b12_;
    TBuf<TPosition::A2> a0_;
    TBuf<TPosition::B2> b0_;
    TBuf<TPosition::CO1> c0_;
    TBuf<TPosition::VECCALC> cu1_,cu2_,zf_,bh1_,bh2_;
    TEventID a1Free_[2];
    TEventID l1Free_,copyReady_,loadReady_,inputFree_,resultReady_,resultFree_,m2v_,vm3_,m3v_;
    union { TEventID m3m2_; TEventID stageFree_; };
};

// Whole N256 output rows; K128 accumulation fits the single 64KiB L0B.

} // namespace AscendC::SwiGluKernel
