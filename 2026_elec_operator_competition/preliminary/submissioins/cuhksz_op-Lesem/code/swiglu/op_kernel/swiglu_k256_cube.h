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

template<bool SharedSecondWeight=false>
class KernelStaticFullRowK256Cube {
public:
    __aicore__ inline bool Init(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,
        GM_ADDR b2,GM_ADDR y,GM_ADDR workspace,uint64_t rows,uint32_t blocks,uint32_t worker) {
        if (blocks == 0U) {
            return false;
        }
        worker_=worker;sub_=GetSubBlockIdx();
        const uint64_t units=rows/16ULL,base=units/blocks,rem=units%blocks;
        const uint64_t owned=base+static_cast<uint32_t>(worker<rem);
        first_=(static_cast<uint64_t>(worker)*base+(worker<rem ? worker : rem))*16ULL;
        tail_=static_cast<uint32_t>(owned*16ULL-3ULL*M);extra_=tail_!=0U;
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(GetUserWorkspace(workspace))+
            static_cast<uint64_t>(worker)*4U*Area);
        if ASCEND_IS_AIC {
            if constexpr(SharedSecondWeight) {
                packRows_=2U*N/blocks;packOwners_=blocks/2U;
                packedWeight_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(GetUserWorkspace(workspace))+
                    static_cast<uint64_t>(blocks)*4U*Area);
            }
            a1_=LocalTensor<half>(TPosition::A1,0,2U*Plane);
            b11_=LocalTensor<half>(TPosition::B1,4U*Plane,NK);
            b12_=LocalTensor<half>(TPosition::B1,4U*Plane+2U*NK,NK);
            a0_=LocalTensor<half>(TPosition::A2,0,Plane);
            b0_=LocalTensor<half>(TPosition::B2,0,2U*N*KH);
            c0_=LocalTensor<float>(TPosition::CO1,0,Area);
        }
        if ASCEND_IS_AIV {
            out_=LocalTensor<half>(TPosition::VECCALC,0,S);
            neg_=LocalTensor<float>(TPosition::VECCALC,0,S);
            stage_=LocalTensor<half>(TPosition::VECCALC,4U*S,S);
            z_=LocalTensor<float>(TPosition::VECCALC,6U*S+256U,S);
            bh1_=LocalTensor<half>(TPosition::VECCALC,10U*S+256U,N);
            bh2_=LocalTensor<half>(TPosition::VECCALC,10U*S+2U*N+256U,N);
        }
        return true;
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC { Cube(); }
        if ASCEND_IS_AIV { Vector(); }
    }
private:
    __aicore__ inline void CopyNz(LocalTensor<half> dst,GlobalTensor<half> src,uint32_t m) {
        Nd2NzParams p{};p.ndNum=1;p.nValue=m;p.dValue=K;p.srcDValue=K;
        p.dstNzC0Stride=m;p.dstNzNStride=1;DataCopy(dst,src,p);
    }
    template<uint32_t Slot,uint32_t Part> __aicore__ inline void LoadA(uint32_t m) {
        LoadData2DParams p{};p.repeatTimes=KH/16U;p.srcStride=m/16U;p.ifTranspose=false;
        for(uint32_t mb=0;mb<m/16U;++mb)
            LoadData(a0_[Part*M*KH+mb*KH*16U],
                a1_[Slot*Plane+Part*KH*m+mb*256U],p);
    }
    template<uint32_t Branch,uint32_t Part> __aicore__ inline void LoadB() {
        auto right=Branch==0U ? b11_ : b12_;
        LoadData2DParams p{};p.repeatTimes=N/16U;p.srcStride=1;p.ifTranspose=false;
        for(uint32_t kb=0;kb<KH/16U;++kb) {
            const uint32_t off=kb*N*16U;
            LoadData(b0_[(Part%2U)*N*KH+off],right[Part*KH*N+off],p);
        }
    }
    template<uint32_t Slot,uint32_t Branch,uint32_t Part,bool FirstJob> __aicore__ inline void MmadPart(uint32_t m) {
        constexpr uint32_t Page=Part%2U;
        // Branch0's first pages were drained before LoadA (or never used).
        if constexpr(!FirstJob || Branch!=0U || Part>=2U) WaitFlag<HardEvent::M_MTE1>(Page);
        if constexpr(Branch==0U) {
            LoadA<Slot,Part>(m);
            // This is the last L1 reader; projection two reuses resident A0.
            if constexpr(Part==3U) SetFlag<HardEvent::MTE1_MTE2>(Slot);
        }
        LoadB<Branch,Part>();
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams p{};p.m=m;p.n=N;p.k=KH;p.cmatrixInitVal=Part==0U;
        p.unitFlag=Part==3U ? 3U : 2U;
        SetMMRowMajor();Mmad(c0_,a0_[Part*M*KH],b0_[Page*N*KH],p);
        // Deferred page release allows the other page's next MTE1 load.
        SetFlag<HardEvent::M_MTE1>(Page);
    }
    template<uint32_t Slot,uint32_t Branch,bool FirstJob> __aicore__ inline void Compute(uint32_t m) {
        MmadPart<Slot,Branch,0,FirstJob>(m);MmadPart<Slot,Branch,1,FirstJob>(m);MmadPart<Slot,Branch,2,FirstJob>(m);MmadPart<Slot,Branch,3,FirstJob>(m);
        FixpipeParamsV220 f{};f.nSize=N;f.mSize=m;f.srcStride=m;f.dstStride=N;
        f.ndNum=1;f.quantPre=QuantMode_t::F322F16;f.unitFlag=3;
        Fixpipe<half,float>(scratch_[(Slot*2U+Branch)*Area],c0_,f);
        CrossCoreSetFlag<2,PIPE_FIX>(8U+Slot*2U+Branch);
    }
    template<uint32_t Job> __aicore__ inline void CubeJob(uint32_t m) {
        constexpr uint32_t Slot=Job%2U,NextSlot=1U-Slot;
        if constexpr(Job>=2U) CrossCoreWaitFlag<2,PIPE_MTE2>(12U+Slot);
        if constexpr(Job!=0U) SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        if constexpr(Job<3U) {
            bool prefetch=Job<2U || extra_;
            if(prefetch) {
                if constexpr(Job!=0U) WaitFlag<HardEvent::MTE1_MTE2>(NextSlot);
                CopyNz(a1_[NextSlot*Plane],x_[(first_+(Job+1ULL)*M)*K],Job==2U ? tail_ : M);
            }
        }
        Compute<Slot,0,(Job==0U)>(m);
        if constexpr(Job==0U) WaitFlag<HardEvent::MTE2_MTE1>(1);
        Compute<Slot,1,(Job==0U)>(m);
    }
    __aicore__ inline void PackSecondWeight() {
        if(worker_<packOwners_) {
            CopyNz(b12_,w2_[static_cast<uint64_t>(worker_)*packRows_*K],packRows_);
            SetFlag<HardEvent::MTE2_MTE3>(5);WaitFlag<HardEvent::MTE2_MTE3>(5);
            const DataCopyParams scatter{K/16U,static_cast<uint16_t>(packRows_),0,
                static_cast<uint16_t>(N-packRows_)};
            DataCopy(packedWeight_[worker_*packRows_*16U],b12_,scatter);
        }
        // Every original compute Cube remains a Mode0 participant.
        CrossCoreSetFlag<0,PIPE_MTE3>(0);
    }
    __aicore__ inline void Cube() {
        if constexpr(SharedSecondWeight) PackSecondWeight();
        CopyNz(a1_,x_[first_*K],M);CopyNz(b11_,w1_,N);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        if constexpr(SharedSecondWeight) {
            CrossCoreWaitFlag<0,PIPE_MTE2>(0);
            DataCopy(b12_,packedWeight_,NK);
        } else CopyNz(b12_,w2_,N);
        SetFlag<HardEvent::MTE2_MTE1>(1);
        CubeJob<0>(M);CubeJob<1>(M);CubeJob<2>(M);
        if(extra_) CubeJob<3>(tail_);
        WaitFlag<HardEvent::M_MTE1>(0);WaitFlag<HardEvent::M_MTE1>(1);
        WaitFlag<HardEvent::MTE1_MTE2>(0);WaitFlag<HardEvent::MTE1_MTE2>(1);
        CrossCoreWaitFlag<2,PIPE_MTE2>(12);CrossCoreWaitFlag<2,PIPE_MTE2>(13);
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void BiasAdd(LocalTensor<half> values,LocalTensor<half> bias,uint32_t rows) {
        const BinaryRepeatParams r{1,1,1,16,16,0};
        Add(values,values,bias,uint64_t(128U),uint8_t(rows),r);
        Add(values[128U],values[128U],bias[128U],uint64_t(128U),uint8_t(rows),r);
        PipeBarrier<PIPE_V>();
    }
    template<uint32_t Job> __aicore__ inline void VectorJob(uint32_t m) {
        constexpr uint32_t Slot=Job%2U;
        const uint32_t firstRows=((m/16U+1U)/2U)*16U;
        const uint32_t rowOffset=sub_==0U ? 0U : firstRows;
        const uint32_t rows=sub_==0U ? firstRows : m-firstRows,area=rows*N;
        WaitFlag<HardEvent::V_MTE2>(0);
        CrossCoreWaitFlag<2,PIPE_MTE2>(8U+Slot*2U);
        if(rows==0U) {
            // Empty tail consumer still participates in both READYs and ACK join.
            CrossCoreWaitFlag<2,PIPE_MTE2>(9U+Slot*2U);
            CrossCoreSetFlag<2,PIPE_MTE2>(12U+Slot);
            SetFlag<HardEvent::V_MTE2>(0);
            return;
        }
        DataCopy(stage_,scratch_[Slot*2U*Area+rowOffset*N],area);
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        BiasAdd(stage_,bh1_,rows);
        Cast(z_,stage_,RoundMode::CAST_NONE,area);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
        CrossCoreWaitFlag<2,PIPE_MTE2>(9U+Slot*2U);
        DataCopy(stage_,scratch_[(Slot*2U+1U)*Area+rowOffset*N],area);
        CrossCoreSetFlag<2,PIPE_MTE2>(12U+Slot);SetFlag<HardEvent::MTE2_V>(0);
        if constexpr(Job!=0U) WaitFlag<HardEvent::MTE3_V>(0);
        ApplySilu(z_, neg_, area);
        Cast(out_,z_,RoundMode::CAST_RINT,area);PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(0);BiasAdd(stage_,bh2_,rows);
        Mul(out_,out_,stage_,area);SetFlag<HardEvent::V_MTE2>(0);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(y_[(first_+Job*uint64_t(M)+rowOffset)*K],out_,
            DataCopyExtParams{static_cast<uint16_t>(rows),N*2U,0,0,0});
        SetFlag<HardEvent::MTE3_V>(0);
    }
    __aicore__ inline void Vector() {
        DataCopy(bh1_,b1_,N);DataCopy(bh2_,b2_,N);SetFlag<HardEvent::V_MTE2>(0);
        VectorJob<0>(M);VectorJob<1>(M);VectorJob<2>(M);
        if(extra_) VectorJob<3>(tail_);
        WaitFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::MTE3_V>(0);
        PipeBarrier<PIPE_ALL>();
    }
    static constexpr uint32_t M=128U,N=256U,K=256U,KH=64U,Area=M*N,Plane=M*K,NK=N*K,S=Area/2U;
    uint32_t worker_,sub_,tail_,packRows_,packOwners_;bool extra_;uint64_t first_;
    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_,scratch_,packedWeight_;
    LocalTensor<half> a1_,b11_,b12_,a0_,b0_,stage_,out_,bh1_,bh2_;
    LocalTensor<float> c0_,neg_,z_;
};


// Full M128/N128/K256, two fixed N owners, six or seven jobs.
// Pair loop stays compact; all user transport/dependency edges match StagePrefetch.
template <bool BalancedTail>
class KernelStaticK256PairCube {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,
        GM_ADDR b2,GM_ADDR y,GM_ADDR workspace,uint64_t rows,uint32_t blocks) {
        blocks_=blocks;bid_=GetBlockIdx();
        if constexpr (BalancedTail) {
            const uint32_t owners=blocks_/2U,owner=bid_/2U;
            const uint64_t units=rows/16ULL,base=units/owners,rem=units%owners;
            const uint64_t owned=base+static_cast<uint32_t>(owner<rem);
            tail_=static_cast<uint32_t>(owned*16ULL-6ULL*M);extra_=tail_!=0U;
            xbase_=(static_cast<uint64_t>(owner)*base+(owner<rem ? owner : rem))*16ULL*K;
            rowstep_=static_cast<uint64_t>(M)*K;
        } else {
            extra_=bid_<((rows/M)*2ULL-6ULL*blocks);
            xbase_=static_cast<uint64_t>(bid_/2U)*M*K;
            rowstep_=static_cast<uint64_t>(blocks_/2U)*M*K;
        }
        ybase_=xbase_+(bid_%2U)*N;
        const uint64_t wc=static_cast<uint64_t>(bid_%2U)*N*K;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x));
        w1_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w1)+wc);
        w2_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(w2)+wc);
        b1_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b1)+(bid_%2U)*N);
        b2_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b2)+(bid_%2U)*N);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y));
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(GetUserWorkspace(workspace))+
            static_cast<uint64_t>(bid_)*4U*Area);
        if ASCEND_IS_AIC {
            a1_=LocalTensor<half>(TPosition::A1,0,2U*Plane);
            b11_=LocalTensor<half>(TPosition::B1,4U*Plane,NK);
            b12_=LocalTensor<half>(TPosition::B1,4U*Plane+2U*NK,NK);
            a0_=LocalTensor<half>(TPosition::A2,0,Plane);
            b0_=LocalTensor<half>(TPosition::B2,0,NK);
            c0_=LocalTensor<float>(TPosition::CO1,0,Area);
        }
        if ASCEND_IS_AIV {
            out_=LocalTensor<half>(TPosition::VECCALC,0,Area);
            neg_=LocalTensor<float>(TPosition::VECCALC,0,Area);
            stage_=LocalTensor<half>(TPosition::VECCALC,4U*Area,Area);
            z_=LocalTensor<float>(TPosition::VECCALC,6U*Area+256U,Area);
            bh1_=LocalTensor<half>(TPosition::VECCALC,10U*Area+256U,N);
            bh2_=LocalTensor<half>(TPosition::VECCALC,10U*Area+2U*N+256U,N);
        }
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC { Cube(); }
        if ASCEND_IS_AIV { Vector(); }
    }
private:
    __aicore__ inline uint32_t GetRows(bool finalJob) const {
        if constexpr (BalancedTail) return finalJob ? tail_ : M;
        return M;
    }

    __aicore__ inline void CopyNz(LocalTensor<half> dst,GlobalTensor<half> src,uint32_t m=M) {
        Nd2NzParams p{};p.ndNum=1;p.nValue=m;p.dValue=K;p.srcDValue=K;
        p.dstNzC0Stride=m;p.dstNzNStride=1;DataCopy(dst,src,p);
    }
    template<uint32_t Slot> __aicore__ inline void LoadA(uint32_t m=M) {
        LoadData2DParams p{};p.repeatTimes=K/16U;p.srcStride=m/16U;p.ifTranspose=false;
        for(uint32_t mb=0;mb<m/16U;++mb)
            LoadData(a0_[mb*K*16U],a1_[Slot*Plane+mb*256U],p);
    }
    template<uint32_t Branch> __aicore__ inline void LoadB() {
        auto right=Branch==0U ? b11_ : b12_;
        LoadData2DParams p{};p.repeatTimes=N/16U;p.srcStride=1;p.ifTranspose=false;
        for(uint32_t kb=0;kb<K/16U;++kb) {
            const uint32_t off=kb*N*16U;LoadData(b0_[off],right[off],p);
        }
    }
    template<uint32_t Slot,uint32_t Branch> __aicore__ inline void Compute(uint32_t m=M) {
        LoadB<Branch>();
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams p{};p.m=m;p.n=N;p.k=K;p.cmatrixInitVal=true;p.unitFlag=3;
        SetMMRowMajor();Mmad(c0_,a0_,b0_,p);
        FixpipeParamsV220 f{};f.nSize=N;f.mSize=m;f.srcStride=m;f.dstStride=N;
        f.ndNum=1;f.quantPre=QuantMode_t::F322F16;f.unitFlag=3;
        Fixpipe<half,float>(scratch_[(Slot*2U+Branch)*Area],c0_,f);
        SetFlag<HardEvent::M_MTE1>(0);WaitFlag<HardEvent::M_MTE1>(0);
        CrossCoreSetFlag<2,PIPE_FIX>(Slot*2U+Branch);
    }
    template<uint32_t Slot> __aicore__ inline void CubeJob(uint32_t pair) {
        constexpr uint32_t NextSlot=1U-Slot;
        if(pair!=0U) CrossCoreWaitFlag<2,PIPE_MTE2>(4U+Slot);
        SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadA<Slot>();SetFlag<HardEvent::MTE1_MTE2>(Slot);
        bool prefetch=true;
        if constexpr(Slot==1U) prefetch=pair<2U || extra_;
        if(prefetch) {
            if(pair!=0U || Slot==1U) WaitFlag<HardEvent::MTE1_MTE2>(NextSlot);
            const uint32_t next=pair*2U+Slot+1U;
            CopyNz(a1_[NextSlot*Plane],x_[xbase_+static_cast<uint64_t>(next)*rowstep_],GetRows(next == 6U));
        }
        Compute<Slot,0>();Compute<Slot,1>();
    }
    __aicore__ inline void CubeLast() {
        CrossCoreWaitFlag<2,PIPE_MTE2>(4);
        SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
        const uint32_t rows = GetRows(true);
        LoadA<0>(rows);SetFlag<HardEvent::MTE1_MTE2>(0);
        Compute<0,0>(rows);Compute<0,1>(rows);
    }
    __aicore__ inline void Cube() {
        CopyNz(a1_,x_[xbase_]);CopyNz(b11_,w1_);CopyNz(b12_,w2_);
        #pragma unroll 1
        for(uint32_t pair=0;pair<3U;++pair) { CubeJob<0>(pair);CubeJob<1>(pair); }
        if(extra_) CubeLast();
        if(extra_) {
            WaitFlag<HardEvent::MTE1_MTE2>(1);WaitFlag<HardEvent::MTE1_MTE2>(0);
            CrossCoreWaitFlag<2,PIPE_MTE2>(5);CrossCoreWaitFlag<2,PIPE_MTE2>(4);
        } else {
            WaitFlag<HardEvent::MTE1_MTE2>(0);WaitFlag<HardEvent::MTE1_MTE2>(1);
            CrossCoreWaitFlag<2,PIPE_MTE2>(4);CrossCoreWaitFlag<2,PIPE_MTE2>(5);
        }
        PipeBarrier<PIPE_ALL>();
    }
    template<uint32_t Slot> __aicore__ inline void VectorJob(uint32_t pair) {
        const uint32_t m=GetRows(pair == 3U),area=m*N;
        WaitFlag<HardEvent::V_MTE2>(0);
        CrossCoreWaitFlag<2,PIPE_MTE2>(Slot*2U);
        DataCopy(stage_,scratch_[Slot*2U*Area],area);
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        const BinaryRepeatParams r{1,1,1,8,8,0};
        Add(stage_,stage_,bh1_,uint64_t(N),uint8_t(m),r);PipeBarrier<PIPE_V>();
        Cast(z_,stage_,RoundMode::CAST_NONE,area);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
        CrossCoreWaitFlag<2,PIPE_MTE2>(Slot*2U+1U);
        DataCopy(stage_,scratch_[(Slot*2U+1U)*Area],area);
        CrossCoreSetFlag<2,PIPE_MTE2>(4U+Slot);SetFlag<HardEvent::MTE2_V>(0);
        // out and neg alias. Previous Y DMA must finish before Muls writes neg.
        if(pair!=0U || Slot!=0U) WaitFlag<HardEvent::MTE3_V>(0);
        ApplySilu(z_, neg_, area);
        Cast(out_,z_,RoundMode::CAST_RINT,area);PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(0);
        Add(stage_,stage_,bh2_,uint64_t(N),uint8_t(m),r);PipeBarrier<PIPE_V>();
        Mul(out_,out_,stage_,area);SetFlag<HardEvent::V_MTE2>(0);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        const uint32_t job=pair*2U+Slot;
        DataCopyPad(y_[ybase_+static_cast<uint64_t>(job)*rowstep_],out_,
            DataCopyExtParams{static_cast<uint16_t>(m),N*2U,0,(K-N)*2U,0});
        SetFlag<HardEvent::MTE3_V>(0);
    }
    __aicore__ inline void Vector() {
        DataCopy(bh1_,b1_,N);DataCopy(bh2_,b2_,N);
        SetFlag<HardEvent::V_MTE2>(0);
        #pragma unroll 1
        for(uint32_t pair=0;pair<3U;++pair) { VectorJob<0>(pair);VectorJob<1>(pair); }
        if(extra_) VectorJob<0>(3U);
        WaitFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::MTE3_V>(0);
        PipeBarrier<PIPE_ALL>();
    }
    static constexpr uint32_t M=128U,N=128U,K=256U,Area=M*N,Plane=M*K,NK=N*K;
    uint32_t blocks_,bid_,tail_;bool extra_;uint64_t xbase_,rowstep_,ybase_;
    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_,scratch_;
    LocalTensor<half> a1_,b11_,b12_,a0_,b0_,stage_,out_,bh1_,bh2_;
    LocalTensor<float> c0_,neg_,z_;
};

// Full-row K256/N64/M64, exactly two jobs per useful worker.
// The sole runtime tiling word is the queried useful-worker count.
class KernelTwoJobK256Cube {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,
        GM_ADDR b2,GM_ADDR y,GM_ADDR workspace,uint32_t blocks) {
        blocks_=blocks;bid_=GetBlockIdx();
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(GetUserWorkspace(workspace))+static_cast<uint64_t>(bid_)*4U*Area);
        if ASCEND_IS_AIC {
            a1_=LocalTensor<half>(TPosition::A1,0,2U*Plane);
            b11_=LocalTensor<half>(TPosition::B1,4U*Plane,NK);
            b12_=LocalTensor<half>(TPosition::B1,4U*Plane+2U*NK,NK);
            a0_=LocalTensor<half>(TPosition::A2,0,Plane);
            b0_=LocalTensor<half>(TPosition::B2,0,2U*NK);
            c0_=LocalTensor<float>(TPosition::CO1,0,2U*Area);
        }
        if ASCEND_IS_AIV {
            neg_=LocalTensor<float>(TPosition::VECCALC,0,Area);
            stage_=LocalTensor<half>(TPosition::VECCALC,4U*Area,Area);
            z_=LocalTensor<float>(TPosition::VECCALC,6U*Area,Area);
            bias_=LocalTensor<half>(TPosition::VECCALC,10U*Area,2U*N);
            out_=LocalTensor<half>(TPosition::VECCALC,10U*Area+4U*N,Area);
        }
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC { Cube(); }
        if ASCEND_IS_AIV { Vector(); }
    }
private:
    static constexpr uint32_t M=64U,N=64U,K=256U,Area=M*N,Plane=M*K,NK=N*K;
    template<uint32_t Job> __aicore__ inline uint64_t Row() {
        return (static_cast<uint64_t>(bid_)+static_cast<uint64_t>(Job)*blocks_)/4U*M;
    }
    __aicore__ inline uint32_t Col() { return bid_%4U*N; }
    __aicore__ inline void CopyNz(LocalTensor<half> dst,GlobalTensor<half> src,uint32_t rows) {
        Nd2NzParams p{};p.ndNum=1;p.nValue=rows;p.dValue=K;p.srcDValue=K;p.dstNzC0Stride=rows;p.dstNzNStride=1;
        DataCopy(dst,src,p);
    }
    template<uint32_t Job> __aicore__ inline void LoadA() {
        LoadData2DParams p{};p.repeatTimes=K/16U;p.srcStride=M/16U;p.ifTranspose=false;
        for(uint32_t mb=0;mb<M/16U;++mb) LoadData(a0_[mb*K*16U],a1_[Job*Plane+mb*256U],p);
    }
    template<uint32_t Branch> __aicore__ inline void LoadB() {
        auto src=Branch ? b12_ : b11_;LoadData2DParams p{};p.repeatTimes=NK/256U;p.srcStride=1;p.ifTranspose=false;
        LoadData(b0_[Branch*NK],src,p);
    }
    template<uint32_t Job,uint32_t Branch> __aicore__ inline void Compute() {
        MmadParams p{};p.m=M;p.n=N;p.k=K;p.cmatrixInitVal=true;p.unitFlag=3;
        SetMMRowMajor();Mmad(c0_[Branch*Area],a0_,b0_[Branch*NK],p);
        FixpipeParamsV220 f{};f.nSize=N;f.mSize=M;f.srcStride=M;f.dstStride=N;f.ndNum=1;f.quantPre=QuantMode_t::F322F16;f.unitFlag=3;
        Fixpipe<half,float>(scratch_[(Job*2U+Branch)*Area],c0_[Branch*Area],f);
        CrossCoreSetFlag<2,PIPE_FIX>(Job*2U+Branch);
    }
    __aicore__ inline void Cube() {
        CopyNz(a1_,x_[Row<0>()*K],M);CopyNz(b11_,w1_[static_cast<uint64_t>(Col())*K],N);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        CopyNz(b12_,w2_[static_cast<uint64_t>(Col())*K],N);SetFlag<HardEvent::MTE2_MTE1>(1);
        WaitFlag<HardEvent::MTE2_MTE1>(0);LoadA<0>();LoadB<0>();
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        CopyNz(a1_[Plane],x_[Row<1>()*K],M);SetFlag<HardEvent::MTE2_MTE1>(0);
        Compute<0,0>();
        WaitFlag<HardEvent::MTE2_MTE1>(1);LoadB<1>();
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);Compute<0,1>();
        SetFlag<HardEvent::M_MTE1>(0);WaitFlag<HardEvent::M_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);LoadA<1>();
        SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        Compute<1,0>();Compute<1,1>();
        SetFlag<HardEvent::M_MTE1>(0);WaitFlag<HardEvent::M_MTE1>(0);
        CrossCoreWaitFlag<2,PIPE_MTE2>(4);CrossCoreWaitFlag<2,PIPE_MTE2>(5);
    }
    template<uint32_t Job> __aicore__ inline void VectorJob() {
        WaitFlag<HardEvent::V_MTE2>(0);CrossCoreWaitFlag<2,PIPE_MTE2>(Job*2U);
        DataCopy(stage_,scratch_[Job*2U*Area],Area);
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        Add(stage_,stage_,bias_,uint64_t(N),M,BinaryRepeatParams{1,1,1,4,4,0});PipeBarrier<PIPE_V>();
        Cast(z_,stage_,RoundMode::CAST_NONE,Area);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
        CrossCoreWaitFlag<2,PIPE_MTE2>(Job*2U+1U);DataCopy(stage_,scratch_[(Job*2U+1U)*Area],Area);
        CrossCoreSetFlag<2,PIPE_MTE2>(4U+Job);SetFlag<HardEvent::MTE2_V>(0);
        ApplySilu(z_, neg_, Area);
        if constexpr(Job!=0U) WaitFlag<HardEvent::MTE3_V>(0);
        Cast(out_,z_,RoundMode::CAST_RINT,Area);PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(0);
        Add(stage_,stage_,bias_[N],uint64_t(N),M,BinaryRepeatParams{1,1,1,4,4,0});PipeBarrier<PIPE_V>();
        Mul(out_,out_,stage_,Area);SetFlag<HardEvent::V_MTE2>(0);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(y_[Row<Job>()*K+Col()],out_,DataCopyExtParams{M,2U*N,0,2U*(K-N),0});
        SetFlag<HardEvent::MTE3_V>(0);
    }
    __aicore__ inline void Vector() {
        DataCopy(bias_,b1_[Col()],N);DataCopy(bias_[N],b2_[Col()],N);
        SetFlag<HardEvent::V_MTE2>(0);VectorJob<0>();VectorJob<1>();
        WaitFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::MTE3_V>(0);
    }
    uint32_t blocks_,bid_;GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_,scratch_;
    LocalTensor<half> a1_,b11_,b12_,a0_,b0_,stage_,bias_,out_;
    LocalTensor<float> c0_,neg_,z_;
};

} // namespace AscendC::SwiGluKernel
