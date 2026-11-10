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

class KernelPairedAligned32 {
    static constexpr uint32_t D=32U,Weight=D*D;
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
    }
    __aicore__ inline void Process() {
        LocalTensor<half> raw(TPosition::VECCALC,131072U,2U*Weight+D);
        LocalTensor<half> bias(TPosition::VECCALC,32768U,2U*D);
        LocalTensor<half> output(TPosition::VECCALC,65536U,D);
        auto rh=reinterpret_cast<__ubuf__ half*>(131072U);
        auto wf=reinterpret_cast<__ubuf__ float*>(0U);
        auto xf=reinterpret_cast<__ubuf__ float*>(16640U);
        auto product=reinterpret_cast<__ubuf__ float*>(65536U);
        auto dot=reinterpret_cast<__ubuf__ float*>(131072U);
        auto hp=reinterpret_cast<__ubuf__ half*>(0U);
        auto bp=reinterpret_cast<__ubuf__ half*>(32768U);
        auto neg=reinterpret_cast<__ubuf__ float*>(131328U);
        auto tmp=reinterpret_cast<__ubuf__ float*>(4096U);
        auto silu=reinterpret_cast<__ubuf__ half*>(131328U);
        auto yp=reinterpret_cast<__ubuf__ half*>(65536U);
        const uint64_t row=GetBlockIdx();
        DataCopyExtParams weight{1,2U*Weight,0,0,0},vec{1,2U*D,0,0,0};
        DataCopyPadExtParams<half> pad{false,0,0,static_cast<half>(0)};
        DataCopyPad(raw,w1_,weight,pad);DataCopyPad(raw[Weight],w2_,weight,pad);
        DataCopyPad(raw[2U*Weight],x_[row*D],vec,pad);
        DataCopyPad(bias,b1_,vec,pad);DataCopyPad(bias[D],b2_,vec,pad);
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(int32_t(64));
        asc_half2float(wf,rh,uint8_t(D),1U,1U,16U,4U);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(D));
        asc_half2float(xf,rh+2U*Weight,uint8_t(1),1U,1U,1U,1U);PipeBarrier<PIPE_V>();
        asc_copy(reinterpret_cast<__ubuf__ uint32_t*>(xf+D),reinterpret_cast<__ubuf__ uint32_t*>(xf),uint8_t(1),1U,1U,4U,4U);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(64));
        asc_mul(product,wf,xf,uint8_t(D),1U,1U,1U,8U,16U,0U);PipeBarrier<PIPE_V>();
        // Each channel: four 8-term partials, then two pairwise levels.
        asc_datablock_reduce_sum(wf,product,uint8_t(D),1U,1U,8U);PipeBarrier<PIPE_V>();
        asc_pair_reduce_sum(product,wf,uint8_t(4),1U,1U,8U);PipeBarrier<PIPE_V>();
        asc_pair_reduce_sum(dot,product,uint8_t(2),1U,1U,8U);PipeBarrier<PIPE_V>();
        asc_float2half_rn(hp,dot,uint32_t(2U*D));PipeBarrier<PIPE_V>();
        asc_add(hp,hp,bp,uint32_t(2U*D));PipeBarrier<PIPE_V>();
        asc_half2float(product,hp,uint32_t(D));PipeBarrier<PIPE_V>();
        asc_mul_scalar(neg,product,-1.0f,uint32_t(D));PipeBarrier<PIPE_V>();
        asc_exp(tmp,neg,uint32_t(D));PipeBarrier<PIPE_V>();
        asc_add_scalar(neg,tmp,1.0f,uint32_t(D));PipeBarrier<PIPE_V>();
        asc_div(tmp,product,neg,uint32_t(D));PipeBarrier<PIPE_V>();
        // Negative is dead after Div; its first Half plane now stores SiLU.
        asc_float2half_rn(silu,tmp,uint32_t(D));PipeBarrier<PIPE_V>();
        asc_mul(yp,silu,hp+D,uint32_t(D));
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(y_[row*D],output,vec);
        SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);ResetMask();
        // All input reads completed; no next MTE2 epoch exists in this row owner.
    }
private:
    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_;
};

template <uint32_t D>
class KernelStripedAligned {
    static_assert(D == 32U || D == 64U);
    static constexpr uint32_t N=D/2U,W=N*D,G=D/N;
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,
        GM_ADDR b2,GM_ADDR y,const SwiGluWorkTiling &) {
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
    }
    __aicore__ inline void Process() {
        // Each source plane starts after all merged Cast destinations.
        LocalTensor<float> f(TPosition::VECCALC,0,4U*W+D+21U*N);
        auto wf=f;auto xf=f[2U*W];auto prod=f[2U*W+D];
        auto wh=prod.ReinterpretCast<half>();auto xh=wh[2U*W];
        auto bh=f[4U*W+D].ReinterpretCast<half>();
        auto z=f[4U*W+D+N];auto neg=z[2U*N];
        auto h=neg[N].ReinterpretCast<half>();auto out=h[N];
        auto sums=f[4U*W+D+5U*N];
        const uint32_t block=GetBlockIdx(),row=block/G,col=(block%G)*N;
        DataCopyPadExtParams<half> pad{false,0,0,static_cast<half>(0)};
        DataCopyExtParams vecD{1,2U*D,0,0,0},weight{1,2U*W,0,0,0},vecN{1,2U*N,0,0,0};
        DataCopyPad(xh,x_[row*D],vecD,pad);
        DataCopyPad(wh,w1_[col*D],weight,pad);
        DataCopyPad(wh[W],w2_[col*D],weight,pad);
        DataCopyPad(bh,b1_[col],vecN,pad);
        DataCopyPad(bh[N],b2_[col],vecN,pad);
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        Cast(wf,wh,RoundMode::CAST_NONE,2U*W+D);PipeBarrier<PIPE_V>();
        BinaryRepeatParams rep{1,1,1,D/8U,D/8U,0};
        Mul(prod,wf,xf,static_cast<uint64_t>(D),static_cast<uint8_t>(2U*N),rep);PipeBarrier<PIPE_V>();
        if constexpr (D == 64U) {
            BlockReduceSum<float>(sums,prod,2U*N,64,1,1,8);PipeBarrier<PIPE_V>();
            BlockReduceSum<float>(z,sums,2U*N/8U,64,1,1,8);PipeBarrier<PIPE_V>();
        } else {
            BlockReduceSum<float>(sums,prod,N,64,1,1,8);PipeBarrier<PIPE_V>();
            PairReduceSum<float>(sums[8U*N],sums,2U*N/16U,64,1,1,8);PipeBarrier<PIPE_V>();
            PairReduceSum<float>(z,sums[8U*N],2U*N/32U,64,1,1,8);PipeBarrier<PIPE_V>();
        }
        Cast(h,z,RoundMode::CAST_RINT,2U*N);PipeBarrier<PIPE_V>();
        Add(h,h,bh,2U*N);PipeBarrier<PIPE_V>();
        Cast(z,h,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();
        ApplySilu(z, neg, N);
        Cast(h,z,RoundMode::CAST_RINT,N);PipeBarrier<PIPE_V>();
        Mul(out,h,out,N);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(y_[row*D+col],out,vecN);
        SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
    }
private:
    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_;
};

class KernelTinyD1 {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
    }
    __aicore__ inline void Process() {
        constexpr uint32_t pitch=16U;
        LocalTensor<half> ub(TPosition::VECCALC,0,272U);
        auto x=ub; auto w1=ub[pitch]; auto w2=ub[2U*pitch];
        auto b1=ub[3U*pitch]; auto b2=ub[4U*pitch];
        auto f=ub[5U*pitch].ReinterpretCast<float>();
        auto product=f[3U*pitch]; auto neg=product[pitch];
        auto h=f[5U*pitch].ReinterpretCast<half>();
        DataCopyExtParams cp{1,2U,0,0,0};
        DataCopyPadExtParams<half> pad{true,0,15,static_cast<half>(0)};
        DataCopyPad(x,x_[GetBlockIdx()],cp,pad);
        DataCopyPad(w1,w1_,cp,pad); DataCopyPad(w2,w2_,cp,pad);
        DataCopyPad(b1,b1_,cp,pad); DataCopyPad(b2,b2_,cp,pad);
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        Cast(f,x,RoundMode::CAST_NONE,3U*pitch); PipeBarrier<PIPE_V>();
        BinaryRepeatParams r{1,1,1,2,2,0};
        Mul(product,f[pitch],f,static_cast<uint64_t>(pitch),2,r); PipeBarrier<PIPE_V>();
        Cast(h,product,RoundMode::CAST_RINT,2U*pitch); PipeBarrier<PIPE_V>();
        Add(h,h,b1,2U*pitch); PipeBarrier<PIPE_V>();
        Cast(product,h,RoundMode::CAST_NONE,1U); PipeBarrier<PIPE_V>();
        ApplySilu(product, neg, 1U);
        Cast(h,product,RoundMode::CAST_RINT,1U); PipeBarrier<PIPE_V>();
        Mul(h,h,h[pitch],1U);
        SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(y_[GetBlockIdx()],h,cp);
        SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
    }
private:
    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_;
};

template <uint32_t StaticD>
class KernelTinyGEMVCompact {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
    }
    __aicore__ inline void Process() {
        constexpr uint32_t p=16U;
        const uint32_t w=d_*p;
        LocalTensor<half> ub(TPosition::VECCALC,0,848U);
        auto x=ub; auto w1=ub[p]; auto w2=ub[p+w];
        auto b1=ub[p+2U*w]; auto b2=b1[p];
        auto f=ub[p+2U*w+2U*p].ReinterpretCast<float>();
        auto products=f[p+2U*w]; auto z=products[2U*w];
        auto neg=z[2U*p]; auto h=neg[p].ReinterpretCast<half>();
        DataCopyExtParams cp{1,d_*2U,0,0,0};
        DataCopyExtParams wm{static_cast<uint16_t>(d_),d_*2U,0,0,0};
        DataCopyPadExtParams<half> pad{true,0,static_cast<uint8_t>(p-d_),static_cast<half>(0)};
        DataCopyPad(x,x_[static_cast<uint64_t>(GetBlockIdx())*d_],cp,pad);
        DataCopyPad(w1,w1_,wm,pad); DataCopyPad(w2,w2_,wm,pad);
        DataCopyPad(b1,b1_,cp,pad); DataCopyPad(b2,b2_,cp,pad);
        SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
        Duplicate(z,0.0f,2U*p);
        Cast(f,x,RoundMode::CAST_NONE,p+2U*w); PipeBarrier<PIPE_V>();
        BinaryRepeatParams r{1,1,1,2,2,0};
        Mul(products,f[p],f,static_cast<uint64_t>(d_),static_cast<uint8_t>(2U*d_),r);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(z,products,d_,d_,1,1,2);
        WholeReduceSum(z[p],products[w],d_,d_,1,1,2); PipeBarrier<PIPE_V>();
        Cast(h,z,RoundMode::CAST_RINT,2U*p); PipeBarrier<PIPE_V>();
        Add(h,h,b1,2U*p); PipeBarrier<PIPE_V>();
        Cast(z,h,RoundMode::CAST_NONE,d_); PipeBarrier<PIPE_V>();
        ApplySilu(z, neg, d_);
        Cast(h,z,RoundMode::CAST_RINT,d_); PipeBarrier<PIPE_V>();
        Mul(h,h,h[p],d_);
        SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(y_[static_cast<uint64_t>(GetBlockIdx())*d_],h,cp);
        SetFlag<HardEvent::MTE3_V>(0); WaitFlag<HardEvent::MTE3_V>(0);
    }
private:
    static constexpr uint32_t d_=StaticD; GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_;
};

template <uint32_t StaticD>
class KernelCachedDotVectorSuffix {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
    }
    __aicore__ inline void Process() {
        Prefetch(x_);Prefetch(w1_);Prefetch(w2_);Prefetch(b1_);Prefetch(b2_);
        LocalTensor<half> h(TPosition::VECCALC,0,32U);
        LocalTensor<float> z(TPosition::VECCALC,256U,16U);
        LocalTensor<float> neg(TPosition::VECCALC,320U,16U);
        __ubuf__ volatile half* raw=reinterpret_cast<__ubuf__ volatile half*>(h.GetPhyAddr());
        float xv[d_];
        #pragma unroll
        for(uint32_t k=0;k<d_;++k) xv[k]=static_cast<float>(x_.GetValue(k));
        // Keep one output's live dot registers across the Scalar prefix.
        #pragma clang loop unroll(disable)
        for(uint32_t n=0;n<d_;++n) {
            float p1[4]={0.0f,0.0f,0.0f,0.0f};
            float p2[4]={0.0f,0.0f,0.0f,0.0f};
            #pragma unroll
            for(uint32_t k=0;k<d_;++k) {
                p1[k]=xv[k]*static_cast<float>(w1_.GetValue(n*d_+k));
                p2[k]=xv[k]*static_cast<float>(w2_.GetValue(n*d_+k));
            }
            const float sum1=(p1[0]+p1[1])+(p1[2]+p1[3]);
            const float sum2=(p2[0]+p2[1])+(p2[2]+p2[3]);
            const half dot1=static_cast<half>(sum1),dot2=static_cast<half>(sum2);
            raw[n]=static_cast<half>(static_cast<float>(dot1)+static_cast<float>(b1_.GetValue(n)));
            raw[16U+n]=static_cast<half>(static_cast<float>(dot2)+static_cast<float>(b2_.GetValue(n)));
        }
        SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
        Cast(z,h,RoundMode::CAST_NONE,d_);PipeBarrier<PIPE_V>();
        ApplySilu(z, neg, d_);
        Cast(h,z,RoundMode::CAST_RINT,d_);PipeBarrier<PIPE_V>();
        Mul(h,h,h[16U],d_);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyExtParams cp{1,d_*2U,0,0,0};
        DataCopyPad(y_,h,cp);
        SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
    }
private:
    __aicore__ inline void Prefetch(const GlobalTensor<half>& value) {
        GlobalTensor<uint64_t> address;
        address.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(const_cast<__gm__ half*>(value.GetPhyAddr())));
        DataCachePreload(address,int16_t(0));
    }
    static constexpr uint32_t d_=StaticD;GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_;
};

// Resource-gated full-K dual MMAD. One A panel serves both projections.
// ND->NZ and transposed-right LoadData mapping are documented in SOURCE_MANIFEST.
// The full-K path is selected only when queried L0/L1/UB capacities permit it.

} // namespace AscendC::SwiGluKernel
