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

constexpr TEventID StaticCubeVectorEvent = 0;

class KernelStaticShortCube {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
        GM_ADDR b2, GM_ADDR y, GM_ADDR workspace, const SwiGluWorkTiling &t) {
        t_=t;
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
        area_=t_.tileM*t_.tileN;
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(GetUserWorkspace(workspace))+
            static_cast<uint64_t>(GetBlockIdx())*4U*area_);
        if ASCEND_IS_AIC {
            InitDoubleInputCubeArena(t_.tileM * t_.alignedDim, t_.tileN * t_.alignedDim, area_,
                a1_, b11_, b12_, a0_, b0_, c0_);
            copyReady_=0;loadReady_=0;inputFree_=0;
        }
        if ASCEND_IS_AIV {
            InitCubeVectorArena(area_, t_.tileN, cu1_, cu2_, zf_, bh1_, bh2_);
        }
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC { ProcessCube(); }
        if ASCEND_IS_AIV { ProcessVector(); }
    }
private:
    __aicore__ inline void Compute(LocalTensor<half> right, uint32_t mc,
        uint32_t branch, uint32_t slot, uint32_t job) {
        ProjectResidentCube(a0_, b0_[branch * t_.tileN * t_.alignedDim], right, c0_[branch * area_],
            scratch_[(slot * 2U + branch) * area_], mc, t_.tileN, t_.alignedDim, branch, job, loadReady_, inputFree_,
            8U + slot * 2U + branch);
    }
    __aicore__ inline void CopyInput(LocalTensor<half> dst, GlobalTensor<half> src, uint32_t mc) {
        CopyPaddedCubeInput(dst, src, mc, t_.tileM, t_.dim, t_.alignedDim);
    }
    __aicore__ inline void ProcessCube() {
        auto a1=a1_;auto a0=a0_;auto b1=b11_;auto b2=b12_;
        const uint32_t groups=t_.dim/t_.tileN;
        const uint32_t plane=t_.tileM*t_.alignedDim;
        const uint32_t n0=GetBlockIdx()%groups*t_.tileN;
        const uint64_t firstM=GetBlockIdx()/groups*t_.tileM;
        const uint32_t firstMc=t_.rows-firstM<t_.tileM ? static_cast<uint32_t>(t_.rows-firstM) : t_.tileM;
        CopyInput(a1,x_[firstM*t_.dim],firstMc);
        CopyCubeTileNz(t_.dim, b1,w1_[static_cast<uint64_t>(n0)*t_.dim],t_.tileN,t_.dim);
        CopyCubeTileNz(t_.dim, b2,w2_[static_cast<uint64_t>(n0)*t_.dim],t_.tileN,t_.dim);
        uint32_t i=0;
        for(uint64_t task=GetBlockIdx();task<t_.tasks;task+=t_.blocks,++i) {
            const uint32_t slot=i%2U;
            if(i>=2U) CrossCoreWaitFlag<2,PIPE_MTE2>(12U+slot);
            const uint64_t m0=task/groups*t_.tileM;
            const uint32_t mc=t_.rows-m0<t_.tileM ? static_cast<uint32_t>(t_.rows-m0) : t_.tileM;
            SetFlag<HardEvent::MTE2_MTE1>(copyReady_);WaitFlag<HardEvent::MTE2_MTE1>(copyReady_);
            LoadCubeA(t_.alignedDim, a0,a1[slot*plane],mc);
            // A0 supplies both projections; A1 can be recycled after this load.
            SetFlag<HardEvent::MTE1_MTE2>(slot);
            const uint64_t next=task+t_.blocks;
            if(next<t_.tasks) {
                const uint32_t nextSlot=(i+1U)%2U;
                if(i>=1U) WaitFlag<HardEvent::MTE1_MTE2>(nextSlot);
                const uint64_t nextM=next/groups*t_.tileM;
                const uint32_t nextMc=t_.rows-nextM<t_.tileM ? static_cast<uint32_t>(t_.rows-nextM) : t_.tileM;
                CopyInput(a1[nextSlot*plane],x_[nextM*t_.dim],nextMc);
            }
            Compute(b1,mc,0U,slot,i);Compute(b2,mc,1U,slot,i);
        }
        const uint32_t pending=i<2U ? i : 2U;
        for(uint32_t j=i-pending;j<i;++j) {
            WaitFlag<HardEvent::MTE1_MTE2>(j%2U);
            CrossCoreWaitFlag<2,PIPE_MTE2>(12U+j%2U);
        }
    }
    __aicore__ inline void ProcessVector() {
        auto c1=cu1_; auto c2=cu2_;
        auto z=zf_;
        auto b1=bh1_; auto b2=bh2_;
        const uint32_t groups=(t_.dim+t_.tileN-1U)/t_.tileN;
        CopyCubeBias(t_, b1,b2,GetBlockIdx()%groups*t_.tileN, b1_, b2_);
        uint32_t localTask=0;
        for(uint64_t task=GetBlockIdx();task<t_.tasks;task+=t_.blocks,++localTask) {
            const uint32_t slot=localTask%2U;
            uint64_t m0=task/groups*t_.tileM; uint32_t n0=task%groups*t_.tileN;
            uint32_t mc=t_.rows-m0<t_.tileM ? static_cast<uint32_t>(t_.rows-m0) : t_.tileM;
            const uint32_t nc=t_.dim-n0<t_.tileN ? t_.dim-n0 : t_.tileN;
            FinishCubeProjectionPair<8U, 12U>(
                c1, c2, z, b1, b2, scratch_, area_, slot, mc, t_.tileN, StaticCubeVectorEvent);
            StoreStridedCubeOutput(y_[m0 * t_.dim + n0], c1, mc, nc, t_.tileN, t_.dim,
                StaticCubeVectorEvent, StaticCubeVectorEvent, StaticCubeVectorEvent);
        }
    }
    SwiGluWorkTiling t_; uint32_t area_;
    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_,scratch_;
    LocalTensor<half> a1_,b11_,b12_,a0_,b0_,cu1_,cu2_,bh1_,bh2_;
    LocalTensor<float> c0_,zf_;
    TEventID copyReady_, loadReady_, inputFree_;
};

class KernelStaticTwoJobK128FullCube {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
        GM_ADDR b2, GM_ADDR y, GM_ADDR workspace, uint32_t blocks) {
        blocks_=blocks;bid_=GetBlockIdx();
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(GetUserWorkspace(workspace))+
            static_cast<uint64_t>(bid_)*4U*Area);
        if ASCEND_IS_AIC {
            InitDoubleInputCubeArena(M * K, N * K, Area, a1_, b11_, b12_, a0_, b0_, c0_);
            copyReady_=0;loadReady_=0;inputFree_=0;
        }
        if ASCEND_IS_AIV {
            InitCubeVectorArena(Area, N, cu1_, cu2_, zf_, bh1_, bh2_);
        }
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC { ProcessCube(); }
        if ASCEND_IS_AIV { ProcessVector(); }
    }
private:
    template <uint32_t Job, uint32_t Branch>
    __aicore__ inline void ProjectJobBranch(LocalTensor<half> right) {
        ProjectResidentCube(a0_, b0_[Branch * N * K], right, c0_[Branch * Area],
            scratch_[(Job * 2U + Branch) * Area], M, N, K, Branch, Job, loadReady_, inputFree_,
            8U + Job * 2U + Branch);
    }

    __aicore__ inline void CopyInput(LocalTensor<half> dst, GlobalTensor<half> src, uint32_t mc) {
        CopyPaddedCubeInput(dst, src, mc, M, K, K);
    }
    template<uint32_t Job> __aicore__ inline void CubeJob() {
        constexpr uint32_t plane=M*K;
        SetFlag<HardEvent::MTE2_MTE1>(copyReady_);WaitFlag<HardEvent::MTE2_MTE1>(copyReady_);
        LoadCubeA(K, a0_,a1_[Job*plane],M);
        SetFlag<HardEvent::MTE1_MTE2>(Job);
        if constexpr(Job==0U) {
            CopyInput(a1_[plane],x_[(static_cast<uint64_t>(bid_)+blocks_)*M*K],M);
        }
        ProjectJobBranch<Job, 0U>(b11_);
        ProjectJobBranch<Job, 1U>(b12_);
    }
    __aicore__ inline void ProcessCube() {
        // Preserve parent all-X/W1/W2-ready ordering. No early-W1 change.
        CopyInput(a1_,x_[static_cast<uint64_t>(bid_)*M*K],M);
        CopyCubeTileNz(K, b11_,w1_,N,K);CopyCubeTileNz(K, b12_,w2_,N,K);
        CubeJob<0>();CubeJob<1>();
        WaitFlag<HardEvent::MTE1_MTE2>(0);WaitFlag<HardEvent::MTE1_MTE2>(1);
        CrossCoreWaitFlag<2,PIPE_MTE2>(12);CrossCoreWaitFlag<2,PIPE_MTE2>(13);
    }
    template <uint32_t Job>
    __aicore__ inline void VectorJob() {
        FinishCubeProjectionPair<8U, 12U>(cu1_, cu2_, zf_, bh1_, bh2_, scratch_, Area, Job, M, N, StaticCubeVectorEvent);
        const uint64_t row = (static_cast<uint64_t>(bid_) + static_cast<uint64_t>(Job) * blocks_) * M;
        StoreStridedCubeOutput(y_[row * K], cu1_, M, N, N, K, StaticCubeVectorEvent, StaticCubeVectorEvent, StaticCubeVectorEvent);
    }

    __aicore__ inline void ProcessVector() {
        DataCopy(bh1_, b1_, N);
        DataCopy(bh2_, b2_, N);
        VectorJob<0>();
        VectorJob<1>();
    }
    static constexpr uint32_t M=128U,N=128U,K=128U,Area=M*N;
    uint32_t blocks_,bid_;
    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_,scratch_;
    LocalTensor<half> a1_,b11_,b12_,a0_,b0_,cu1_,cu2_,bh1_,bh2_;
    LocalTensor<float> c0_,zf_;
    TEventID copyReady_, loadReady_, inputFree_;
};

template <bool CompactOutput, uint32_t Owners = 0U, bool FullRows = false>
class KernelShortCube {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
        GM_ADDR b2, GM_ADDR y, GM_ADDR workspace, const SwiGluWorkTiling &t) {
        t_ = t;
        InitStorage(x, w1, b1, w2, b2, y, workspace);
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
        GM_ADDR b2, GM_ADDR y, GM_ADDR workspace, uint64_t rows, uint32_t dim) {
        const uint64_t row = static_cast<uint64_t>(GetBlockIdx()) * 48U;
        if constexpr (FullRows) {
            rowCount_ = 48U;
        } else {
            rowCount_ = rows - row < 48U ? static_cast<uint32_t>(rows - row) : 48U;
        }
        t_ = {rows, Owners, dim, 48U, 48U, Owners, 13U, 48U};
        InitStorage(x, w1, b1, w2, b2, y, workspace);
    }

    __aicore__ inline void Process() {
        if ASCEND_IS_AIC { ProcessCube(); }
        if ASCEND_IS_AIV { ProcessVector(); }
    }

private:
    static constexpr uint32_t ReadyFlag = CompactOutput ? 0U : 8U;
    static constexpr uint32_t AckFlag = ReadyFlag + 4U;

    __aicore__ inline void InitStorage(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
        GM_ADDR b2, GM_ADDR y, GM_ADDR workspace) {
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
        area_=t_.tileM*t_.tileN;
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(GetUserWorkspace(workspace))+
            static_cast<uint64_t>(GetBlockIdx())*4U*area_);
        if ASCEND_IS_AIC {
            const uint32_t mk=t_.tileM*t_.alignedDim, nk=t_.tileN*t_.alignedDim;
            a1_=LocalTensor<half>(TPosition::A1,0,mk);
            b11_=LocalTensor<half>(TPosition::B1,2U*mk,nk);
            b12_=LocalTensor<half>(TPosition::B1,2U*(mk+nk),nk);
            a0_=LocalTensor<half>(TPosition::A2,0,mk);
            b0_=LocalTensor<half>(TPosition::B2,0,2U*nk);
            c0_=LocalTensor<float>(TPosition::CO1,0,2U*area_);
            copyReady_=0; loadReady_=0;
        }
        if ASCEND_IS_AIV {
            InitCubeVectorArena(area_, t_.tileN, cu1_, cu2_, zf_, bh1_, bh2_);
            m2v_=0; vm3_=0; m3v_=0; m3m2_=0;
        }
    }

    __aicore__ inline uint32_t GetTileRows(uint64_t m0) const {
        if constexpr (Owners != 0U) {
            return rowCount_;
        } else {
            return t_.rows - m0 < t_.tileM ? static_cast<uint32_t>(t_.rows - m0) : t_.tileM;
        }
    }
    __aicore__ inline void Compute(LocalTensor<half> right, uint32_t mc, uint32_t branch, uint32_t slot) {
        auto a = a0_;
        auto b = b0_[branch * t_.tileN * t_.alignedDim];
        auto c = c0_[branch * area_];
        LoadCubeB(t_.tileN, t_.alignedDim, b, right);
        SetFlag<HardEvent::MTE1_M>(loadReady_);
        WaitFlag<HardEvent::MTE1_M>(loadReady_);

        const uint32_t alignedRows = (mc + 15U) / 16U * 16U;
        MmadParams p{};
        p.m = alignedRows;
        p.n = t_.tileN;
        p.k = t_.alignedDim;
        p.cmatrixInitVal = true;
        SetMMRowMajor();
        p.unitFlag = 3;
        Mmad(c, a, b, p);

        FixpipeParamsV220 f{};
        f.nSize = t_.tileN;
        f.mSize = alignedRows;
        f.unitFlag = 3U;
        f.srcStride = alignedRows;
        f.dstStride = t_.tileN;
        f.ndNum = 1;
        f.quantPre = QuantMode_t::F322F16;
        Fixpipe<half, float>(scratch_[(slot * 2U + branch) * area_], c, f);
        CrossCoreSetFlag<2, PIPE_FIX>(ReadyFlag + slot * 2U + branch);
    }

    __aicore__ inline void CopyInput(LocalTensor<half> dst,GlobalTensor<half> src,uint32_t mc) {
        if(mc<t_.tileM) {
            Fill(dst,InitConstValueParams<half>{1,
                static_cast<uint16_t>(t_.tileM*t_.alignedDim/16U),0,static_cast<half>(0)});
        }
        CopyCubeTileNz(t_.dim, dst,src,mc,t_.dim);
    }

    __aicore__ inline void CopyWeights(uint32_t n0) {
        const uint32_t nc = t_.dim - n0 < t_.tileN ? t_.dim - n0 : t_.tileN;
        InitConstValueParams<half> zero{1,
            static_cast<uint16_t>(t_.tileN * t_.alignedDim / 16U), 0, static_cast<half>(0)};
        const uint64_t offset = static_cast<uint64_t>(n0) * t_.dim;
        if constexpr (Owners != 0U) {
            if (nc < t_.tileN) Fill(b11_, zero);
            CopyCubeTileNz(t_.dim, b11_, w1_[offset], nc, t_.dim);
            SetFlag<HardEvent::MTE2_MTE1>(copyReady_);
            if (nc < t_.tileN) Fill(b12_, zero);
            CopyCubeTileNz(t_.dim, b12_, w2_[offset], nc, t_.dim);
            SetFlag<HardEvent::MTE2_MTE1>(1);
        } else {
            if (nc < t_.tileN) {
                Fill(b11_, zero);
                Fill(b12_, zero);
            }
            CopyCubeTileNz(t_.dim, b11_, w1_[offset], nc, t_.dim);
            CopyCubeTileNz(t_.dim, b12_, w2_[offset], nc, t_.dim);
        }
    }

    __aicore__ inline void CubeTile(uint64_t task, uint32_t groups, uint32_t localTask, uint32_t &cachedN) {
        if (groups == 0U) return;
        const uint32_t slot = localTask % 2U;
        if (localTask >= 2U) CrossCoreWaitFlag<2, PIPE_MTE2>(AckFlag + slot);
        const uint64_t m0 = task / groups * t_.tileM;
        const uint32_t n0 = task % groups * t_.tileN;
        const uint32_t mc = GetTileRows(m0);
        CopyInput(a1_, x_[m0 * t_.dim], mc);
        if (n0 != cachedN) {
            CopyWeights(n0);
            cachedN = n0;
        }
        if constexpr (Owners == 0U) SetFlag<HardEvent::MTE2_MTE1>(copyReady_);
        WaitFlag<HardEvent::MTE2_MTE1>(copyReady_);
        LoadCubeA(t_.alignedDim, a0_, a1_, mc);
        Compute(b11_, mc, 0, slot);
        if constexpr (Owners != 0U) WaitFlag<HardEvent::MTE2_MTE1>(1);
        Compute(b12_, mc, 1, slot);
    }

    __aicore__ inline void ProcessCube() {
        const uint32_t groups = (t_.dim + t_.tileN - 1U) / t_.tileN;
        uint32_t cachedN = 0xffffffffU;
        if constexpr (Owners != 0U) {
            CubeTile(GetBlockIdx(), groups, 0U, cachedN);
            CrossCoreWaitFlag<2, PIPE_MTE2>(AckFlag);
        } else {
            uint32_t localTask = 0;
            for (uint64_t task = GetBlockIdx(); task < t_.tasks; task += t_.blocks, ++localTask) {
                CubeTile(task, groups, localTask, cachedN);
            }
            const uint32_t pending = localTask < 2U ? localTask : 2U;
            for (uint32_t i = localTask - pending; i < localTask; ++i) {
                CrossCoreWaitFlag<2, PIPE_MTE2>(AckFlag + i % 2U);
            }
        }
    }


    __aicore__ inline void WriteOutput(LocalTensor<half> c1, LocalTensor<half> c2,
        uint64_t m0, uint32_t n0, uint32_t mc, uint32_t nc) {
        if constexpr (CompactOutput) {
            PipeBarrier<PIPE_V>();
            uint64_t kept = 0;
            GatherMask(c2,c1,static_cast<uint8_t>(7),true,t_.dim,
                GatherMaskParams{1,static_cast<uint16_t>(mc),static_cast<uint16_t>(t_.tileN/16U),0},kept);
            ResetMask();
            SetFlag<HardEvent::V_MTE3>(vm3_); WaitFlag<HardEvent::V_MTE3>(vm3_);
            DataCopyExtParams out{1,mc*t_.dim*2U,0,0,0};
            DataCopyPad(y_[m0*t_.dim],c2,out);
        } else {
            StoreStridedCubeOutput(y_[m0 * t_.dim + n0], c1, mc, nc, t_.tileN, t_.dim,
                vm3_, m3v_, m3m2_);
            return;
        }
        SetFlag<HardEvent::MTE3_V>(m3v_); WaitFlag<HardEvent::MTE3_V>(m3v_);
        SetFlag<HardEvent::MTE3_MTE2>(m3m2_); WaitFlag<HardEvent::MTE3_MTE2>(m3m2_);
    }

    __aicore__ inline void VectorTile(uint64_t task, uint32_t groups, uint32_t localTask, bool fixedBias) {
        if (groups == 0U) return;
        const uint32_t slot = localTask % 2U;
        const uint64_t m0 = task / groups * t_.tileM;
        const uint32_t n0 = task % groups * t_.tileN;
        const uint32_t mc = GetTileRows(m0);
        const uint32_t nc = t_.dim - n0 < t_.tileN ? t_.dim - n0 : t_.tileN;
        if (!fixedBias) CopyCubeBias(t_, bh1_, bh2_, n0, b1_, b2_);
        auto c1 = cu1_;
        auto c2 = cu2_;
        FinishCubeProjectionPair<ReadyFlag, AckFlag>(
            c1, c2, zf_, bh1_, bh2_, scratch_, area_, slot, mc, t_.tileN, m2v_);
        WriteOutput(c1, c2, m0, n0, mc, nc);
    }

    __aicore__ inline void ProcessVector() {
        const uint32_t groups = (t_.dim + t_.tileN - 1U) / t_.tileN;
        const bool fixedBias = t_.blocks % groups == 0U;
        if (fixedBias) CopyCubeBias(t_, bh1_, bh2_, GetBlockIdx() % groups * t_.tileN, b1_, b2_);
        if constexpr (Owners != 0U) {
            VectorTile(GetBlockIdx(), groups, 0U, fixedBias);
        } else {
            uint32_t localTask = 0;
            for (uint64_t task = GetBlockIdx(); task < t_.tasks; task += t_.blocks, ++localTask) {
                VectorTile(task, groups, localTask, fixedBias);
            }
        }
    }

    SwiGluWorkTiling t_;
    uint32_t area_, rowCount_;
    GlobalTensor<half> x_, w1_, w2_, b1_, b2_, y_, scratch_;
    LocalTensor<half> a1_, b11_, b12_, a0_, b0_, cu1_, cu2_, bh1_, bh2_;
    LocalTensor<float> c0_, zf_;
    TEventID copyReady_, loadReady_, m2v_, vm3_, m3v_, m3m2_;
};
// FP32 products use explicitly zeroed Kpad16 lanes; two-level reduction changes the tree.
// Normal Copy broadcasts [32,16] weights and [R,16] X into [R,32,16].

} // namespace AscendC::SwiGluKernel
