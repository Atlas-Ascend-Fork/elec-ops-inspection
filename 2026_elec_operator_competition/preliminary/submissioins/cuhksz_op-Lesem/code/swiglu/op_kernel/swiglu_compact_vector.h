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

__aicore__ inline uint32_t GetCompactTileRows(GM_ADDR tiling, uint32_t bid, uint64_t m0, uint32_t tileRows) {
    if (bid + 1U != GetBlockNum()) return tileRows;
    const uint64_t rows = *reinterpret_cast<__gm__ uint64_t*>(tiling);
    return static_cast<uint32_t>(rows - m0);
}

template <uint32_t D, uint32_t RawPlane>
__aicore__ inline void PrefetchCompactWeights(LocalTensor<half> raw, LocalTensor<half> bh1,
    LocalTensor<half> bh2, GlobalTensor<half> w1, GlobalTensor<half> w2,
    GlobalTensor<half> b1, GlobalTensor<half> b2) {
    const DataCopyExtParams weights{1, D * D * 2U, 0, 0, 0};
    const DataCopyPadExtParams<half> wp{true, 0, static_cast<uint8_t>(RawPlane - D * D), static_cast<half>(0)};
    DataCopyPad(raw, w1, weights, wp);
    DataCopyPad(raw[RawPlane], w2, weights, wp);
    const DataCopyExtParams bias{1, D * 2U, 0, 0, 0};
    const DataCopyPadExtParams<half> bp{true, 0, static_cast<uint8_t>(48U - D), static_cast<half>(0)};
    DataCopyPad(bh1, b1, bias, bp);
    DataCopyPad(bh2, b2, bias, bp);
}

template <uint32_t D, uint32_t R>
__aicore__ inline void StoreCompactVector(LocalTensor<half> output, LocalTensor<half> product,
    GlobalTensor<half> dst, uint32_t rc) {
    uint64_t kept = 0;
    GatherMask(output, product, static_cast<uint8_t>(7), true, D, GatherMaskParams{1,R,3,0}, kept);
    ResetMask();
    SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
    const DataCopyExtParams params{1, rc * D * 2U, 0, 0, 0};
    DataCopyPad(dst, output, params);
    SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
}

template<uint32_t D>
class KernelCompactBlockDotShortVector {
    static_assert(D>=33U && D<=40U);
    static constexpr uint32_t R=8U,N=48U,P=40U,K=32U,Tail=D-K;
    static constexpr uint32_t RawPlane=(D*D+15U)/16U*16U;
    static constexpr uint32_t RawX=(R*D+15U)/16U*16U;
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,
        GM_ADDR b2,GM_ADDR y,GM_ADDR tiling) {
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);tiling_=tiling;
    }
    __aicore__ inline void Process() {
        LocalTensor<half> raw(TPosition::VECCALC,180224U,2U*RawPlane+RawX);
        LocalTensor<half> bias(TPosition::VECCALC,46080U,2U*N);
        PrefetchCompactWeights<D, RawPlane>(raw, bias, bias[N], w1_, w2_, b1_, b2_);
        BuildIndices();
        const uint32_t bid = GetBlockIdx();
        const uint64_t m0 = static_cast<uint64_t>(bid) * R;
        const uint32_t rc = GetCompactTileRows(tiling_, bid, m0, R);
        GatherOperands(raw, m0, rc);
        ComputePrefix();
        AccumulateTail();
        FinishResult(m0, rc);
    }

private:
    __aicore__ inline void BuildIndices() {
        LocalTensor<int32_t> ix(TPosition::VECCALC,32768U,64U);
        auto wfp=reinterpret_cast<__ubuf__ float*>(0U);
        auto xfp=reinterpret_cast<__ubuf__ float*>(41216U);
        auto wtp=reinterpret_cast<__ubuf__ float*>(24576U);
        auto xtp=reinterpret_cast<__ubuf__ float*>(27648U);
        auto ixp=reinterpret_cast<__ubuf__ int32_t*>(32768U);
        auto rop=reinterpret_cast<__ubuf__ int32_t*>(33024U);
        auto rbp=reinterpret_cast<__ubuf__ int32_t*>(33280U);
        auto gp=reinterpret_cast<__ubuf__ int32_t*>(34304U);
        for(uint32_t j=0;j<8U;++j) ix.SetValue(j,static_cast<int32_t>(j));
        SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
        asc_add_scalar(ixp+8U,ixp,int32_t(8),uint32_t(8));PipeBarrier<PIPE_V>();
        asc_add_scalar(ixp+16U,ixp,int32_t(16),uint32_t(16));PipeBarrier<PIPE_V>();
        asc_add_scalar(ixp+32U,ixp,int32_t(32),uint32_t(32));PipeBarrier<PIPE_V>();
        asc_mul_scalar(rop,ixp,int32_t(8U*D),uint32_t(64));PipeBarrier<PIPE_V>();
        asc_brcb(reinterpret_cast<__ubuf__ uint32_t*>(rbp),reinterpret_cast<__ubuf__ uint32_t*>(rop),1U,8U,uint8_t(3));PipeBarrier<PIPE_V>();
        asc_mul_scalar(ixp,ixp,int32_t(4),uint32_t(64));PipeBarrier<PIPE_V>();
        // Save the true K-tail offsets before making a joint two-channel index.
        // This transient lives in XT until tail indices have consumed it.
        asc_brcb(reinterpret_cast<__ubuf__ uint32_t*>(xtp),
            reinterpret_cast<__ubuf__ uint32_t*>(ixp+32U),1U,8U,uint8_t(1));PipeBarrier<PIPE_V>();
        asc_add_scalar(ixp+32U,ixp+32U,int32_t(4U*(D-32U)),uint32_t(32));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(int32_t(64));asc_add(gp,rbp,ixp,uint8_t(P/2U),1U,0U,1U,8U,1U,0U);
        asc_duplicate(wfp,0.0f,uint32_t(P*128U));asc_duplicate(xfp,0.0f,uint32_t(R*128U));
        asc_duplicate(wtp,0.0f,uint32_t(Tail*2U*P));PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GatherOperands(LocalTensor<half> raw, uint64_t m0, uint32_t rc) {
        auto wfp=reinterpret_cast<__ubuf__ float*>(0U);
        auto xfp=reinterpret_cast<__ubuf__ float*>(41216U);
        auto wtp=reinterpret_cast<__ubuf__ float*>(24576U);
        auto xtp=reinterpret_cast<__ubuf__ float*>(27648U);
        auto xbp=reinterpret_cast<__ubuf__ float*>(28928U);
        auto rop=reinterpret_cast<__ubuf__ int32_t*>(33024U);
        auto gp=reinterpret_cast<__ubuf__ int32_t*>(34304U);
        auto rfp=reinterpret_cast<__ubuf__ float*>(163840U);
        auto rhp=reinterpret_cast<__ubuf__ half*>(180224U);
        const uint32_t xc=rc*D,xp=(xc+15U)/16U*16U;
        DataCopyExtParams in{1,xc*2U,0,0,0};
        DataCopyPadExtParams<half> ip{true,0,static_cast<uint8_t>(xp-xc),static_cast<half>(0)};
        DataCopyPad(raw[2U*RawPlane],x_[m0*D],in,ip);
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        asc_half2float(rfp,rhp,uint32_t(2U*RawPlane));
        asc_half2float(rfp+2U*RawPlane,rhp+2U*RawPlane,uint32_t(xp));PipeBarrier<PIPE_V>();
        // Even/odd channel gathers make every coefficient vector start at BG0.
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(64));asc_gather(wfp,rfp,reinterpret_cast<__ubuf__ uint32_t*>(gp),16U,uint8_t((D+1U)/2U));
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(64));asc_gather(wfp+P/2U*128U,rfp+RawPlane,reinterpret_cast<__ubuf__ uint32_t*>(gp),16U,uint8_t((D+1U)/2U));
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(32));asc_gather(xfp,rfp+2U*RawPlane,reinterpret_cast<__ubuf__ uint32_t*>(gp),32U,uint8_t((rc+1U)/2U));PipeBarrier<PIPE_V>();
        asc_add_scalar(gp,gp,int32_t(D*4U),uint32_t(P/2U*64U));PipeBarrier<PIPE_V>();
        if(rc>1U) {SetVectorMask<float,MaskMode::NORMAL>(int32_t(32));asc_gather(xfp+128U,rfp+2U*RawPlane,reinterpret_cast<__ubuf__ uint32_t*>(gp),32U,uint8_t(rc/2U));};
        PipeBarrier<PIPE_V>();
        // Repeat X32 twice inside each256B vector. All X rows start at BG8.
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(32));asc_copy(reinterpret_cast<__ubuf__ uint32_t*>(xfp+32U),reinterpret_cast<__ubuf__ uint32_t*>(xfp),uint8_t(R),1U,1U,16U,16U);PipeBarrier<PIPE_V>();
        asc_shiftright(rop,rop,int32_t(1),uint32_t(64));PipeBarrier<PIPE_V>();
        SetVectorMask<int32_t,MaskMode::NORMAL>(int32_t(64));asc_add(gp,reinterpret_cast<__ubuf__ int32_t*>(xtp),rop,uint8_t(Tail),1U,0U,1U,8U,1U,0U);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(D));asc_gather(wtp,rfp,reinterpret_cast<__ubuf__ uint32_t*>(gp),uint16_t(2U*P/8U),uint8_t(Tail));
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(D));asc_gather(wtp+P,rfp+RawPlane,reinterpret_cast<__ubuf__ uint32_t*>(gp),uint16_t(2U*P/8U),uint8_t(Tail));
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(rc));asc_gather(xtp,rfp+2U*RawPlane,reinterpret_cast<__ubuf__ uint32_t*>(gp),1U,uint8_t(Tail));PipeBarrier<PIPE_V>();
        asc_brcb(reinterpret_cast<__ubuf__ uint32_t*>(xbp),reinterpret_cast<__ubuf__ uint32_t*>(xtp),1U,8U,uint8_t(Tail));PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputePrefix() {
        auto wfp=reinterpret_cast<__ubuf__ float*>(0U);
        auto pp=reinterpret_cast<__ubuf__ float*>(65536U);
        auto prefix=reinterpret_cast<__ubuf__ float*>(131072U);
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(int32_t(64));
        for(uint32_t row=0;row<R;++row)
            asc_mul(reinterpret_cast<__ubuf__ float*>(65536U+row*P*64U*4U),
                reinterpret_cast<__ubuf__ float*>(0U),
                reinterpret_cast<__ubuf__ float*>(41216U+row*512U),
                static_cast<uint8_t>(P),1U,1U,1U,8U,16U,0U);
        PipeBarrier<PIPE_V>();
        constexpr uint32_t repeats=R*P;
        for(uint32_t first=0;first<repeats;first+=240U) {
            const uint32_t n=repeats-first<240U ? repeats-first : 240U;
            SetVectorMask<float,MaskMode::NORMAL>(int32_t(64));asc_datablock_reduce_sum(wfp+first*8U,pp+first*64U,uint8_t(n),1U,1U,8U);
        }
        PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(64));asc_pair_reduce_sum(pp,wfp,uint8_t(R*P/8U),1U,1U,8U);PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(64));asc_pair_reduce_sum(prefix,pp,uint8_t(R*P/16U),1U,1U,8U);PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateTail() {
        auto wtp=reinterpret_cast<__ubuf__ float*>(24576U);
        auto xbp=reinterpret_cast<__ubuf__ float*>(28928U);
        auto prefix=reinterpret_cast<__ubuf__ float*>(131072U);
        // Prefix remains FP32 until every real K-tail term has been added.
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(int32_t(64));
        for(uint32_t k=0;k<Tail;++k) {
            asc_fma(prefix,wtp+k*2U*P,xbp+k*R*8U,uint8_t(R),1U,1U,0U,uint8_t(2U*P/8U),0U,1U);
            PipeBarrier<PIPE_V>();
        }
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(2U*P-64U));
        for(uint32_t k=0;k<Tail;++k) {
            asc_fma(prefix+64U,wtp+k*2U*P+64U,xbp+k*R*8U,uint8_t(R),1U,1U,0U,uint8_t(2U*P/8U),0U,1U);
            PipeBarrier<PIPE_V>();
        }
        ResetMask();
    }

    __aicore__ inline void FinishResult(uint64_t m0, uint32_t rc) {
        auto bpv=reinterpret_cast<__ubuf__ half*>(46080U);
        auto pp=reinterpret_cast<__ubuf__ float*>(65536U);
        auto prefix=reinterpret_cast<__ubuf__ float*>(131072U);
        LocalTensor<half> h(TPosition::VECCALC,0U,2U*R*N);auto h2=h[R*N];
        auto hp=reinterpret_cast<__ubuf__ half*>(0U);
        auto np=reinterpret_cast<__ubuf__ float*>(131328U);
        auto epTemp=reinterpret_cast<__ubuf__ float*>(4096U);
        auto siluHalf=reinterpret_cast<__ubuf__ half*>(163840U);
        auto productHalf=reinterpret_cast<__ubuf__ half*>(65536U);
        asc_duplicate(hp,static_cast<half>(0),uint32_t(2U*R*N));PipeBarrier<PIPE_V>();
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(D));asc_float2half_rn(hp,prefix,uint8_t(R),1U,1U,3U,uint16_t(2U*P/8U));
        SetVectorMask<float,MaskMode::NORMAL>(int32_t(D));asc_float2half_rn(hp+R*N,prefix+P,uint8_t(R),1U,1U,3U,uint16_t(2U*P/8U));PipeBarrier<PIPE_V>();
        SetVectorMask<half,MaskMode::NORMAL>(int32_t(D));asc_add(hp,hp,bpv,uint8_t(R),1U,1U,1U,3U,3U,0U);SetVectorMask<half,MaskMode::NORMAL>(int32_t(D));asc_add(hp+R*N,hp+R*N,bpv+N,uint8_t(R),1U,1U,1U,3U,3U,0U);PipeBarrier<PIPE_V>();
        asc_half2float(pp,hp,uint32_t(R*N));PipeBarrier<PIPE_V>();
        asc_mul_scalar(np,pp,-1.0f,uint32_t(R*N));PipeBarrier<PIPE_V>();
        asc_exp(epTemp,np,uint32_t(R*N));PipeBarrier<PIPE_V>();
        asc_add_scalar(np,epTemp,1.0f,uint32_t(R*N));PipeBarrier<PIPE_V>();
        asc_div(epTemp,pp,np,uint32_t(R*N));PipeBarrier<PIPE_V>();
        asc_float2half_rn(siluHalf,epTemp,uint32_t(R*N));PipeBarrier<PIPE_V>();
        asc_mul(productHalf,siluHalf,hp+R*N,uint32_t(R*N));PipeBarrier<PIPE_V>();
        LocalTensor<half> productView(TPosition::VECCALC,65536U,R*N);
        StoreCompactVector<D, R>(h2, productView, y_[m0*D], rc);
    }

    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_;GM_ADDR tiling_;
};


template<uint32_t D>
class KernelCompactOuterShortVector {
    static_assert(D>=33U && D<=47U);
    static constexpr uint32_t R=8U,K=48U,N=64U;
    static constexpr uint32_t P=(D+7U)/8U*8U,WRow=2U*P;
    static constexpr uint32_t RawPlane=(D*D+15U)/16U*16U;
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,
        GM_ADDR b2,GM_ADDR y,GM_ADDR tiling) {
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
        tiling_=tiling;
    }
    __aicore__ inline void Process() {
        LocalTensor<float> wf(TPosition::VECCALC,0,K*WRow+R*K);
        LocalTensor<half> bh1(TPosition::VECCALC,123008U,48U);
        LocalTensor<half> bh2(TPosition::VECCALC,122880U,48U);
        LocalTensor<int32_t> grid(TPosition::VECCALC,28672U,3072U);
        LocalTensor<int32_t> index(TPosition::VECCALC,40960U,512U);
        LocalTensor<float> accum(TPosition::VECCALC,98304U,4U*R*128U+512U);
        LocalTensor<half> raw(TPosition::VECCALC,131072U,2U*RawPlane+R*K);
        // Keep constant-size prefetch ahead of the last-owner row read.
        PrefetchCompactWeights<D, RawPlane>(raw, bh1, bh2, w1_, w2_, b1_, b2_);
        BuildIndices(grid, index);
        Duplicate(wf,0.0f,K*WRow+R*K);
        Duplicate(accum,0.0f,4U*(P/8U)*128U+512U);
        const uint32_t bid = GetBlockIdx();
        const uint64_t m0 = static_cast<uint64_t>(bid) * R;
        const uint32_t rc = GetCompactTileRows(tiling_, bid, m0, R);
        PackOperands(wf, raw, grid, index, m0, rc);
        auto even = Accumulate(wf, accum);
        FinishResult(even, bh1, bh2, m0, rc);
    }

private:
    __aicore__ inline void BuildIndices(LocalTensor<int32_t> grid, LocalTensor<int32_t> index) {
        LocalTensor<int32_t> base(TPosition::VECCALC,26112U,64U);
        LocalTensor<int32_t> ko(TPosition::VECCALC,26368U,64U);
        LocalTensor<int32_t> kb(TPosition::VECCALC,26624U,384U);
        LocalTensor<int32_t> ik(TPosition::VECCALC,43008U,512U);
        LocalTensor<int32_t> ir(TPosition::VECCALC,45056U,512U);
        for(uint32_t j=0;j<8U;++j) index.SetValue(j,static_cast<int32_t>(j));
        SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
        Adds(index[8],index,int32_t(8),8U);PipeBarrier<PIPE_V>();
        Adds(index[16],index,int32_t(16),16U);PipeBarrier<PIPE_V>();
        Adds(index[32],index,int32_t(32),32U);PipeBarrier<PIPE_V>();
        Adds(index[64],index,int32_t(64),64U);PipeBarrier<PIPE_V>();
        Adds(index[128],index,int32_t(128),128U);PipeBarrier<PIPE_V>();
        Adds(index[256],index,int32_t(256),256U);PipeBarrier<PIPE_V>();
        // Inactive N columns alias the last valid channel, never an invalid read.
        Mins(base,index,static_cast<int32_t>(D-1U),64U);PipeBarrier<PIPE_V>();
        Muls(base,base,static_cast<int32_t>(4U*D),64U);
        Muls(ko,index,int32_t(4),48U);PipeBarrier<PIPE_V>();
        Brcb(kb,ko,6U,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
        Copy(grid,kb,uint64_t(64),48U,CopyRepeatParams{1,0,8,1});PipeBarrier<PIPE_V>();
        Add(grid,grid,base,uint64_t(64),48U,BinaryRepeatParams{1,1,1,8,8,0});
        ShiftRight(ik,index,int32_t(3),384U);PipeBarrier<PIPE_V>();
        Muls(ir,ik,int32_t(8),384U);PipeBarrier<PIPE_V>();
        Sub(ir,index,ir,384U);PipeBarrier<PIPE_V>();
        Muls(ir,ir,int32_t(48),384U);PipeBarrier<PIPE_V>();
        Add(index,ir,ik,384U);PipeBarrier<PIPE_V>();
        ShiftLeft(index,index,int32_t(2),384U);
    }

    __aicore__ inline void PackOperands(LocalTensor<float> wf, LocalTensor<half> raw, LocalTensor<int32_t> grid,
        LocalTensor<int32_t> index, uint64_t m0, uint32_t rc) {
        LocalTensor<float> rawFloat(TPosition::VECCALC,65536U,2U*RawPlane+R*K);
        auto xh=raw[2U*RawPlane];
        DataCopyPadExtParams<half> bp{true,0,static_cast<uint8_t>(48U-D),static_cast<half>(0)};
        DataCopyExtParams input{static_cast<uint16_t>(rc),D*2U,0,0,0};
        DataCopyPad(xh,x_[m0*D],input,bp);
        // Missing rows occupy disjoint, aligned 96-byte ranges from MTE2 writes.
        if(rc<R) Duplicate(xh[rc*K],static_cast<half>(0),(R-rc)*K);
        SetFlag<HardEvent::MTE2_V>(0);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(0);
        const auto offsets=grid.ReinterpretCast<uint32_t>();
        // Widen each original compact value once. Masked gathers skip N padding.
        Cast(rawFloat,raw,RoundMode::CAST_NONE,2U*RawPlane+R*K);PipeBarrier<PIPE_V>();
        // Pack the two projection panels into adjacent channel groups per K.
        Gather(wf,rawFloat,offsets,0U,static_cast<uint64_t>(D),
            static_cast<uint8_t>(D),static_cast<uint16_t>(WRow/8U));
        Gather(wf[P],rawFloat[RawPlane],offsets,0U,static_cast<uint64_t>(D),
            static_cast<uint8_t>(D),static_cast<uint16_t>(WRow/8U));
        Gather(wf[K*WRow],rawFloat[2U*RawPlane],index.ReinterpretCast<uint32_t>(),0U,R*K);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline LocalTensor<float> Accumulate(LocalTensor<float> wf, LocalTensor<float> accum) {
        // Raw GM staging is dead. Reuse bank 2 for k-major scalar broadcasts.
        LocalTensor<float> xp(TPosition::VECCALC,131072U,R*K*8U);
        Brcb(xp,wf[K*WRow],R*K/8U,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
        constexpr uint8_t groups=P/8U;
        auto even=accum[64U];auto odd=accum[2U*groups*128U+256U];
        const BinaryRepeatParams fma{1,0,1,16,1,0};
        SetMaskNorm();SetVectorMask<float,MaskMode::NORMAL>(int32_t(64));
        // Projection is part of the repeat axis, so each opcode serves both W panels.
        for(uint32_t k=0;k+1U<D;k+=2U) {
            MulAddDst<float,float,false>(even,wf[k*WRow],xp[k*R*8U],
                MASK_PLACEHOLDER,2U*groups,fma);
            MulAddDst<float,float,false>(odd,wf[(k+1U)*WRow],xp[(k+1U)*R*8U],
                MASK_PLACEHOLDER,2U*groups,fma);
            PipeBarrier<PIPE_V>();
        }
        if constexpr(D%2U) {
            MulAddDst<float,float,false>(even,wf[(D-1U)*WRow],xp[(D-1U)*R*8U],
                MASK_PLACEHOLDER,2U*groups,fma);PipeBarrier<PIPE_V>();
        }
        ResetMask();
        Add(even,even,odd,uint64_t(64),2U*groups,BinaryRepeatParams{1,1,1,16,16,16});
        PipeBarrier<PIPE_V>();
        return even;
    }

    __aicore__ inline void FinishResult(LocalTensor<float> even, LocalTensor<half> bh1, LocalTensor<half> bh2,
        uint64_t m0, uint32_t rc) {
        constexpr uint8_t groups=P/8U;
        auto a1=even;auto a2=even[groups*128U];

        // Permute whole eight-Float blocks, retaining a48-channel row pitch.
        LocalTensor<float> dot(TPosition::VECCALC,0,2U*R*48U);
        Copy(dot,a1,uint64_t(64),groups,CopyRepeatParams{6,1,1,16});
        Copy(dot[R*48U],a2,uint64_t(64),groups,CopyRepeatParams{6,1,1,16});
        PipeBarrier<PIPE_V>();
        // Preserve dot->Half, Half+bias, FP32 SiLU->Half, Half product.
        LocalTensor<half> h(TPosition::VECCALC,65536U,2U*R*48U);
        auto h2=h[R*48U];
        Duplicate(h,static_cast<half>(0),2U*R*48U);PipeBarrier<PIPE_V>();
        Cast(h,dot,RoundMode::CAST_RINT,uint64_t(D),R,UnaryRepeatParams{1,1,3,6});
        Cast(h2,dot[R*48U],RoundMode::CAST_RINT,uint64_t(D),R,UnaryRepeatParams{1,1,3,6});
        PipeBarrier<PIPE_V>();
        Add(h,h,bh1,uint64_t(D),R,BinaryRepeatParams{1,1,1,3,3,0});
        Add(h2,h2,bh2,uint64_t(D),R,BinaryRepeatParams{1,1,1,3,3,0});
        PipeBarrier<PIPE_V>();
        auto z=dot;LocalTensor<float> neg(TPosition::VECCALC,131328U,R*48U);
        Cast(z,h,RoundMode::CAST_NONE,R*48U);PipeBarrier<PIPE_V>();
        ApplySilu(z, neg, R*48U);
        Cast(h,z,RoundMode::CAST_RINT,R*48U);PipeBarrier<PIPE_V>();
        Mul(h,h,h2,R*48U);PipeBarrier<PIPE_V>();
        StoreCompactVector<D, R>(h2, h, y_[m0*D], rc);
    }

    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_;
    GM_ADDR tiling_;
};

} // namespace AscendC::SwiGluKernel
