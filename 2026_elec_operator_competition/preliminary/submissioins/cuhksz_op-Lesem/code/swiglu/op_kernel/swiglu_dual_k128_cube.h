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

class KernelStaticDualFourJobK128Cube {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
        GM_ADDR b2, GM_ADDR y, GM_ADDR workspace, uint32_t blocks, uint32_t worker) {
        blocks_=blocks;worker_=worker;
        sub_=GetSubBlockIdx();
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
        scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(GetUserWorkspace(workspace))+
            static_cast<uint64_t>(worker_)*(4U*Area+256U));
        if ASCEND_IS_AIC {
            InitDoubleInputCubeArena(M * K, N * K, Area, a1_, b11_, b12_, a0_, b0_, c0_);
        }
        if ASCEND_IS_AIV {
            cu1_=LocalTensor<half>(TPosition::VECCALC,0,2U*S);
            cu2_=LocalTensor<half>(TPosition::VECCALC,4U*S,S);
            zf_=LocalTensor<float>(TPosition::VECCALC,6U*S+256U,S);
            bh1_=LocalTensor<half>(TPosition::VECCALC,12U*S+256U,N);
            bh2_=LocalTensor<half>(TPosition::VECCALC,12U*S+2U*N+256U,N);
        }
    }
    __aicore__ inline void Process() {
        if ASCEND_IS_AIC { ProcessCube(); }
        if ASCEND_IS_AIV { ProcessVector(); }
    }
private:
    __aicore__ inline void LoadB(LocalTensor<half> dst,LocalTensor<half> src) {
        // Physical W is (output channel,K); its NZ rows are N, not K.
        LoadData2DParams p{};p.repeatTimes=N*K/256U;
        p.srcStride=1;p.ifTranspose=false;
        LoadData(dst,src,p);
    }
    template<uint32_t Job,uint32_t Branch> __aicore__ inline void Compute() {
        constexpr uint32_t Slot=Job%2U;
        auto b=b0_[Branch*N*K];auto c=c0_[Branch*Area];
        if constexpr(Job==0U) LoadB(b,Branch==0U ? b11_ : b12_);
        if constexpr(Branch==0U || Job==0U) {
            SetFlag<HardEvent::MTE1_M>(0);WaitFlag<HardEvent::MTE1_M>(0);
        }
        MmadParams p{};p.m=M;p.n=N;p.k=K;p.cmatrixInitVal=true;p.unitFlag=3;
        SetMMRowMajor();Mmad(c,a0_,b,p);
        FixpipeParamsV220 f{};f.nSize=N;f.mSize=M;f.unitFlag=3;
        f.srcStride=M;f.dstStride=N;f.ndNum=1;f.quantPre=QuantMode_t::F322F16;
        Fixpipe<half,float>(scratch_[(Slot*2U+Branch)*Area],c,f);
        if constexpr(Branch==1U) {
            SetFlag<HardEvent::M_MTE1>(0);WaitFlag<HardEvent::M_MTE1>(0);
        }
        CrossCoreSetFlag<2,PIPE_FIX>(8U+Slot*2U+Branch);
    }
    template<uint32_t Job> __aicore__ inline void CubeJob() {
        constexpr uint32_t Slot=Job%2U,NextSlot=(Job+1U)%2U;
        if constexpr(Job>=2U) CrossCoreWaitFlag<2,PIPE_MTE2>(12U+Slot);
        SetFlag<HardEvent::MTE2_MTE1>(0);WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadCubeA(K, a0_,a1_[Slot*M*K],M);
        SetFlag<HardEvent::MTE1_MTE2>(Slot);
        if constexpr(Job<3U) {
            if constexpr(Job>=1U) WaitFlag<HardEvent::MTE1_MTE2>(NextSlot);
            const uint64_t row=(static_cast<uint64_t>(worker_)+static_cast<uint64_t>(Job+1U)*blocks_)*M;
            CopyCubeTileNz(K, a1_[NextSlot*M*K],x_[row*K],M,K);
        }
        Compute<Job,0>();Compute<Job,1>();
    }
    __aicore__ inline void ProcessCube() {
        CopyCubeTileNz(K, a1_,x_[static_cast<uint64_t>(worker_)*M*K],M,K);
        CopyCubeTileNz(K, b11_,w1_,N,K);CopyCubeTileNz(K, b12_,w2_,N,K);
        CubeJob<0>();CubeJob<1>();CubeJob<2>();CubeJob<3>();
        WaitFlag<HardEvent::MTE1_MTE2>(0);CrossCoreWaitFlag<2,PIPE_MTE2>(12);
        WaitFlag<HardEvent::MTE1_MTE2>(1);CrossCoreWaitFlag<2,PIPE_MTE2>(13);
    }
    __aicore__ inline void LoadBias() {
        DataCopy(bh1_,b1_,N);DataCopy(bh2_,b2_,N);
    }
    template<uint32_t Job> __aicore__ inline void VectorJob() {
        constexpr uint32_t Slot=Job%2U;
        auto c1=cu1_;auto c2=cu2_;auto z=zf_;auto neg=c1.ReinterpretCast<float>();
        LocalTensor<half> out(TPosition::VECCALC,10U*S+256U,S);
        if constexpr(Job!=0U) WaitFlag<HardEvent::V_MTE2>(0);
        CrossCoreWaitFlag<2,PIPE_MTE2>(8U+Slot*2U);
        DataCopy(c1,scratch_[Slot*2U*Area+sub_*S],S);
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        const BinaryRepeatParams r{1,1,1,8,8,0};
        Add(c1,c1,bh1_,uint64_t(N),uint8_t(HalfRows),r);PipeBarrier<PIPE_V>();
        Cast(z,c1,RoundMode::CAST_NONE,S);PipeBarrier<PIPE_V>();
        ApplySilu(z, neg, S);
        SetFlag<HardEvent::V_MTE2>(0);
        if constexpr(Job!=0U) WaitFlag<HardEvent::V_MTE2>(1);
        CrossCoreWaitFlag<2,PIPE_MTE2>(9U+Slot*2U);
        DataCopy(c2,scratch_[(Slot*2U+1U)*Area+sub_*S],S);
        CrossCoreSetFlag<2,PIPE_MTE2>(12U+Slot);
        SetFlag<HardEvent::MTE2_V>(0);
        if constexpr(Job!=0U) WaitFlag<HardEvent::MTE3_V>(0);
        Cast(out,z,RoundMode::CAST_RINT,S);PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(0);
        Add(c2,c2,bh2_,uint64_t(N),uint8_t(HalfRows),r);PipeBarrier<PIPE_V>();
        Mul(out,out,c2,S);SetFlag<HardEvent::V_MTE2>(1);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        const uint64_t row=(static_cast<uint64_t>(worker_)+static_cast<uint64_t>(Job)*blocks_)*M+sub_*HalfRows;
        DataCopyPad(y_[row*K],out,DataCopyExtParams{HalfRows,N*2U,0,0,0});
        SetFlag<HardEvent::MTE3_V>(0);
    }
    __aicore__ inline void ProcessVector() {
        LoadBias();VectorJob<0>();VectorJob<1>();VectorJob<2>();VectorJob<3>();
        WaitFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(1);
        WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
    static constexpr uint32_t M=128U,N=128U,K=128U,Area=M*N,HalfRows=M/2U,S=Area/2U;
    uint32_t blocks_,worker_,sub_;
    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_,scratch_;
    LocalTensor<half> a1_,b11_,b12_,a0_,b0_,cu1_,cu2_,bh1_,bh2_;
    LocalTensor<float> c0_,zf_;
};

} // namespace AscendC::SwiGluKernel
