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

__aicore__ inline void ReduceShortRowProducts(LocalTensor<float> wp, LocalTensor<float> bs,
    LocalTensor<float> z, uint32_t rows) {
    // Each full-mask Block repeat consumes four Kpad16 dots.
    const uint32_t blockRepeats=8U*rows;
    if(blockRepeats<=255U) {
        BlockReduceSum<float>(bs,wp,blockRepeats,64,1,1,8);
    } else {
        BlockReduceSum<float>(bs,wp,128,64,1,1,8);
        BlockReduceSum<float>(bs[1024],wp[8192],blockRepeats-128U,64,1,1,8);
    }
    PipeBarrier<PIPE_V>();
    // Adjacent block sums belong to the same dot; no cross-dot addition.
    PairReduceSum<float>(z,bs,rows,64,1,1,8);
    PipeBarrier<PIPE_V>();
}

template <bool PackOutput, bool StridedOutput = false>
__aicore__ inline void FinishShortVector(LocalTensor<float> wp, LocalTensor<float> bs,
    LocalTensor<float> z, LocalTensor<float> neg, LocalTensor<half> joined,
    LocalTensor<half> out, LocalTensor<half> bias, GlobalTensor<half> dst, uint32_t rows, uint32_t dim) {
    const uint32_t dots=rows*32U;
    ReduceShortRowProducts(wp, bs, z, rows);
    Cast(joined,z,RoundMode::CAST_RINT,dots);PipeBarrier<PIPE_V>();
    BinaryRepeatParams add{1,1,1,2,2,0};
    Add(joined,joined,bias,static_cast<uint64_t>(32),static_cast<uint8_t>(rows),add);
    PipeBarrier<PIPE_V>();
    // Only projection1 enters SiLU. Projection2 remains biased Half.
    const uint32_t activeElements=rows*16U;
    UnaryRepeatParams firstProjection{1,1,2,2};
    Cast(z,joined,RoundMode::CAST_NONE,static_cast<uint64_t>(16),
         static_cast<uint8_t>(rows),firstProjection);
    PipeBarrier<PIPE_V>();
    ApplySilu(z, neg, activeElements);
    if constexpr (StridedOutput) {
        Cast(out,z,RoundMode::CAST_RINT,uint64_t(16),uint8_t(rows),UnaryRepeatParams{1,1,2,2});
    } else {
        Cast(out,z,RoundMode::CAST_RINT,activeElements);
    }
    PipeBarrier<PIPE_V>();
    constexpr uint8_t stride = StridedOutput ? 2 : 1;
    BinaryRepeatParams product{1,1,1,stride,stride,2};
    Mul(out,out,joined[16],static_cast<uint64_t>(dim),static_cast<uint8_t>(rows),product);
    if constexpr (PackOutput) {
        PipeBarrier<PIPE_V>();
        // Products in wp died at BlockReduceSum; packing does not touch out.
        auto packed=wp.ReinterpretCast<half>();
        uint64_t kept=0;
        GatherMask(packed,out,static_cast<uint8_t>(7),true,13U,
            GatherMaskParams{1,static_cast<uint16_t>(rows),stride,0},kept);
        ResetMask();
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyExtParams output{1,rows*13U*2U,0,0,0};
        DataCopyPad(dst,packed,output);
    } else {
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyExtParams output{static_cast<uint16_t>(rows),dim*2U,0,0,0};
        DataCopyPad(dst,out,output);
    }
    SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
}

__aicore__ inline void MultiplyShortRows(LocalTensor<float> wp, LocalTensor<float> wf,
    LocalTensor<float> xf, uint32_t rc) {
    // Direct products: original product arena and rounding tree are retained.
    const BinaryRepeatParams directProduct{2,2,0,64,0,2};
    Mul(wp[0U],wf[0U],xf[0U],uint64_t(64),uint8_t(rc),directProduct);
    Mul(wp[8U],wf[8U],xf[8U],uint64_t(64),uint8_t(rc),directProduct);
    Mul(wp[128U],wf[128U],xf[0U],uint64_t(64),uint8_t(rc),directProduct);
    Mul(wp[136U],wf[136U],xf[8U],uint64_t(64),uint8_t(rc),directProduct);
    PipeBarrier<PIPE_V>();
    Mul(wp[256U],wf[256U],xf[0U],uint64_t(64),uint8_t(rc),directProduct);
    Mul(wp[264U],wf[264U],xf[8U],uint64_t(64),uint8_t(rc),directProduct);
    Mul(wp[384U],wf[384U],xf[0U],uint64_t(64),uint8_t(rc),directProduct);
    Mul(wp[392U],wf[392U],xf[8U],uint64_t(64),uint8_t(rc),directProduct);
    PipeBarrier<PIPE_V>();
}

template <uint32_t BatchR, bool PackOutput, bool DirectProduct = false, bool FullRows = false>
class KernelShortVector {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,
        GM_ADDR b2,GM_ADDR y,const SwiGluWorkTiling &t) {
        dim_=t.dim;row_=static_cast<uint64_t>(GetBlockIdx())*BatchR;
        rc_=t.rows-row_<BatchR ? static_cast<uint32_t>(t.rows-row_) : BatchR;
        InitInputs(x,w1,b1,w2,b2,y);
    }

    __aicore__ inline void Init(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,
        GM_ADDR b2,GM_ADDR y,uint64_t rows) {
        dim_=13U;row_=static_cast<uint64_t>(GetBlockIdx())*BatchR;
        if constexpr (FullRows) rc_=BatchR;
        else rc_=static_cast<uint32_t>(rows-row_);
        InitInputs(x,w1,b1,w2,b2,y);
    }

    __aicore__ inline void Process() {
        constexpr uint32_t RawElements=512U+BatchR*16U+32U;
        constexpr uint32_t RawBytes=RawElements*2U;
        constexpr uint32_t WBase=RawBytes+(512U+BatchR*16U)*4U;
        constexpr uint32_t XBase=WBase+BatchR*512U*4U;
        LocalTensor<half> raw(TPosition::VECCALC,0,RawElements);
        LocalTensor<float> wf(TPosition::VECCALC,RawBytes,512U+BatchR*16U);
        auto xf=wf[512];
        LocalTensor<float> wp(TPosition::VECCALC,WBase,BatchR*512U);
        LocalTensor<float> xp(TPosition::VECCALC,XBase,BatchR*512U);
        // Block sums die after PairReduce; their storage then holds neg.
        auto bs=xp;auto z=xp[BatchR*64U];auto neg=xp;
        auto joined=xp[BatchR*96U].ReinterpretCast<half>();
        auto out=xp[BatchR*128U].ReinterpretCast<half>();
        auto bias=raw[512U+BatchR*16U];
        LoadOperands(raw, wf, bias, RawElements);
        if constexpr (DirectProduct) MultiplyShortRows(wp, wf, xf, rc_);
        else ExpandProducts(wp, wf, xp, xf);
        FinishShortVector<PackOutput>(wp, bs, z, neg, joined, out, bias, y_[row_*dim_], rc_, dim_);
    }

private:
    __aicore__ inline void InitInputs(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,GM_ADDR b2,GM_ADDR y) {
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
    }

    __aicore__ inline void LoadOperands(LocalTensor<half> raw, LocalTensor<float> wf,
        LocalTensor<half> bias, uint32_t rawElements) {
        // This defines N padding as well as K padding before any Vector read.
        Duplicate(raw,static_cast<half>(0),rawElements);
        SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyExtParams weights{static_cast<uint16_t>(dim_),dim_*2U,0,0,0};
        DataCopyExtParams inputs{static_cast<uint16_t>(rc_),dim_*2U,0,0,0};
        DataCopyExtParams bi{1,dim_*2U,0,0,0};
        DataCopyPadExtParams<half> pad{true,0,static_cast<uint8_t>(16U-dim_),static_cast<half>(0)};
        DataCopyPad(raw,w1_,weights,pad);DataCopyPad(raw[256],w2_,weights,pad);
        DataCopyPad(raw[512],x_[row_*dim_],inputs,pad);
        DataCopyPad(bias,b1_,bi,pad);DataCopyPad(bias[16],b2_,bi,pad);
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        Cast(wf,raw,RoundMode::CAST_NONE,512U+rc_*16U);PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ExpandProducts(LocalTensor<float> wp, LocalTensor<float> wf,
        LocalTensor<float> xp, LocalTensor<float> xf) {
        // Copy Counter is a high-dimensional exception: mask is per repeat,
        // and repeat still controls rows (official CANN9.0.0 Copy contract).
        SetMaskCount();
        SetVectorMask<float,MaskMode::COUNTER>(0,512);
        const CopyRepeatParams weightCopy{1,1,64,0};
        Copy<float,false>(wp,wf,MASK_PLACEHOLDER,rc_,weightCopy);
        SetVectorMask<float,MaskMode::COUNTER>(0,256);
        const CopyRepeatParams inputCopy{2,0,64,2};
        Copy<float,false>(xp,xf,MASK_PLACEHOLDER,rc_,inputCopy);
        Copy<float,false>(xp[8],xf[8],MASK_PLACEHOLDER,rc_,inputCopy);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();ResetMask();
        Mul(wp,wp,xp,rc_*512U);PipeBarrier<PIPE_V>();
    }

    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_;
    uint64_t row_;uint32_t dim_,rc_;
};

template <uint32_t Rows>
class KernelConstRcShortVector {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR w1,GM_ADDR b1,GM_ADDR w2,
        GM_ADDR b2,GM_ADDR y,uint64_t rows) {
        static_assert(Rows>=13U && Rows<=32U);
        dim_=13U;row_=0U;
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
    }
    __aicore__ inline void Process() {
        constexpr uint32_t RawElements=512U+32U*16U+32U;
        constexpr uint32_t RawBytes=RawElements*2U;
        constexpr uint32_t WBase=65536U;
        constexpr uint32_t XBase=131072U;
        LocalTensor<half> raw(TPosition::VECCALC,WBase,RawElements);
        LocalTensor<float> wf(TPosition::VECCALC,RawBytes,512U+32U*16U);
        auto xf=wf[512];
        LocalTensor<float> wp(TPosition::VECCALC,WBase,32U*512U);
        LocalTensor<float> xp(TPosition::VECCALC,XBase,4680U);
        // Block sums die after PairReduce; their storage then holds neg.
        auto bs=xp;auto z=wf;auto neg=xp[80U];
        auto joined=wp.ReinterpretCast<half>();
        auto out=xp[4168U].ReinterpretCast<half>();
        LocalTensor<half> bias(TPosition::VECCALC,2048U,32U);
        LoadCompactOperands<RawBytes>(raw, wf, bias);
        MultiplyShortRows(wp, wf, xf, rc_);
        FinishShortVector<true, true>(wp, bs, z, neg, joined, out, bias, y_[row_*dim_], rc_, dim_);
    }

private:
    template <uint32_t RawBytes>
    __aicore__ inline void LoadCompactOperands(LocalTensor<half> raw, LocalTensor<float> wf,
        LocalTensor<half> bias) {
        // Whole compact matrices replace short unaligned row transactions.
        constexpr uint32_t CompactW=176U;
        constexpr uint32_t CompactX=(Rows*13U+15U)/16U*16U;
        LocalTensor<half> compact(TPosition::VECCALC,RawBytes,2U*CompactW+CompactX+16U);
        LocalTensor<int32_t> offsets(TPosition::VECCALC,131072U,256U);
        LocalTensor<int32_t> rowOffset(TPosition::VECCALC,64U,16U);
        LocalTensor<int32_t> seed(TPosition::VECCALC,0U,16U);
        LocalTensor<int32_t> rowBroadcast(TPosition::VECCALC,65536U,128U);
        DataCopyExtParams weights{1,169U*2U,0,0,0};
        DataCopyPadExtParams<half> wpPad{true,0,7,static_cast<half>(0)};
        DataCopyPad(compact,w1_,weights,wpPad);
        DataCopyPad(compact[CompactW],w2_,weights,wpPad);
        DataCopyExtParams inputs{1,Rows*13U*2U,0,0,0};
        DataCopyPadExtParams<half> xpPad{true,0,
            static_cast<uint8_t>(CompactX-Rows*13U),static_cast<half>(0)};
        DataCopyPad(compact[2U*CompactW],x_,inputs,xpPad);
        DataCopyExtParams bi{1,26U,0,0,0};
        DataCopyPadExtParams<half> pad{true,0,3,static_cast<half>(0)};
        DataCopyPad(bias,b1_,bi,pad);DataCopyPad(bias[16],b2_,bi,pad);
        SetFlag<HardEvent::MTE2_V>(0);
        // This aligned tail is disjoint from every MTE2 destination. It
        // defines the final three masked input reads when CompactX is exact.
        Duplicate(compact[2U*CompactW+CompactX],static_cast<half>(0),16U);
        for(uint32_t i=0;i<8U;++i) seed.SetValue(i,static_cast<int32_t>(2U*i));
        SetFlag<HardEvent::S_V>(0);WaitFlag<HardEvent::S_V>(0);
        Adds(seed[8],seed,int32_t(16),8U);PipeBarrier<PIPE_V>();
        Muls(rowOffset,seed,int32_t(13),16U);PipeBarrier<PIPE_V>();
        Brcb(rowBroadcast,rowOffset,2U,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
        Add(offsets,rowBroadcast,seed,uint64_t(16),16U,
            BinaryRepeatParams{1,0,1,2,1,0});PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE2_V>(0);
        const auto index=offsets.ReinterpretCast<uint32_t>();
        Gather(raw,compact,index,0U,256U);
        Gather(raw[256],compact[CompactW],index,0U,256U);
        if constexpr(Rows<=16U) {
            Gather(raw[512],compact[2U*CompactW],index,0U,Rows*16U);
        } else {
            Gather(raw[512],compact[2U*CompactW],index,0U,256U);
            Gather(raw[768],compact[2U*CompactW+208U],index,0U,(Rows-16U)*16U);
        }
        PipeBarrier<PIPE_V>();
        // N padding is initialized by Gather and is never stored. K padding
        // participates in real dots and must be exactly zero in both W/X.
        uint64_t zeroK[2]={0xe000e000e000e000ULL,0xe000e000e000e000ULL};
        Duplicate(raw,static_cast<half>(0),zeroK,
            static_cast<uint8_t>((512U+Rows*16U+127U)/128U),1U,8U);
        PipeBarrier<PIPE_V>();
        Cast(wf,raw,RoundMode::CAST_NONE,512U+rc_*16U);PipeBarrier<PIPE_V>();
    }

    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_;
    uint64_t row_;
    uint32_t dim_;
    static constexpr uint32_t rc_ = Rows;
};

} // namespace AscendC::SwiGluKernel
