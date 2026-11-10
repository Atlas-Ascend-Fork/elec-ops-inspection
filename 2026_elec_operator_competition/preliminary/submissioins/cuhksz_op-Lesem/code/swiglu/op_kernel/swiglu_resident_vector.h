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

class KernelSwiGlu {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
        GM_ADDR b2, GM_ADDR y, const SwiGluTilingData &t) {
        t_ = t;
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
        const uint32_t k = t.alignedDim;
        pipe_.InitBuffer(xh_, k * 2); pipe_.InitBuffer(xf_, k * 4);
        pipe_.InitBuffer(wh1_, k * 2); pipe_.InitBuffer(wh2_, k * 2);
        pipe_.InitBuffer(wf1_, k * 4); pipe_.InitBuffer(wf2_, k * 4);
        pipe_.InitBuffer(tmp_, k * 4); pipe_.InitBuffer(sum_, 64);
        pipe_.InitBuffer(z1_, 128); pipe_.InitBuffer(z2_, 128);
        pipe_.InitBuffer(neg_, 128); pipe_.InitBuffer(round_, 64);
        pipe_.InitBuffer(out_, 64);
        m2v_ = pipe_.AllocEventID<HardEvent::MTE2_V>();
        vm2_ = pipe_.AllocEventID<HardEvent::V_MTE2>();
        vs_ = pipe_.AllocEventID<HardEvent::V_S>();
        sv_ = pipe_.AllocEventID<HardEvent::S_V>();
        vm3_ = pipe_.AllocEventID<HardEvent::V_MTE3>();
        m3v_ = pipe_.AllocEventID<HardEvent::MTE3_V>();
    }
    __aicore__ inline void Process() {
        const uint32_t groups = (t_.dim + t_.tileN - 1) / t_.tileN;
        auto xh = xh_.Get<half>(); auto xf = xf_.Get<float>();
        auto wh1 = wh1_.Get<half>(); auto wh2 = wh2_.Get<half>();
        auto wf1 = wf1_.Get<float>(); auto wf2 = wf2_.Get<float>();
        auto tmp = tmp_.Get<float>(); auto sums = sum_.Get<float>();
        auto z1 = z1_.Get<float>(); auto z2 = z2_.Get<float>();
        auto neg = neg_.Get<float>(); auto rounded = round_.Get<half>();
        auto out = out_.Get<half>();
        const DataCopyExtParams in{1, t_.dim * 2U, 0, 0, 0};
        const DataCopyPadExtParams<half> pad{false, 0, 0, static_cast<half>(0)};
        for (uint64_t task = GetBlockIdx(); task < t_.tasks; task += t_.blocks) {
            const uint64_t row = task / groups;
            const uint32_t n0 = (task % groups) * t_.tileN;
            const uint32_t count = t_.dim - n0 < t_.tileN ? t_.dim - n0 : t_.tileN;
            DataCopyPad(xh, x_[row * t_.dim], in, pad);
            SetFlag<HardEvent::MTE2_V>(m2v_); WaitFlag<HardEvent::MTE2_V>(m2v_);
            Cast(xf, xh, RoundMode::CAST_NONE, t_.dim);
            PipeBarrier<PIPE_V>();
            for (uint32_t n = 0; n < count; ++n) {
                const uint64_t wi = static_cast<uint64_t>(n0 + n) * t_.dim;
                DataCopyPad(wh1, w1_[wi], in, pad);
                DataCopyPad(wh2, w2_[wi], in, pad);
                SetFlag<HardEvent::MTE2_V>(m2v_); WaitFlag<HardEvent::MTE2_V>(m2v_);
                Cast(wf1, wh1, RoundMode::CAST_NONE, t_.dim);
                Cast(wf2, wh2, RoundMode::CAST_NONE, t_.dim);
                PipeBarrier<PIPE_V>();
                Mul(wf1, wf1, xf, t_.dim); Mul(wf2, wf2, xf, t_.dim);
                PipeBarrier<PIPE_V>();
                ReduceSum(sums, wf1, tmp, t_.dim);
                PipeBarrier<PIPE_V>();
                ReduceSum(sums[8], wf2, tmp, t_.dim);
                SetFlag<HardEvent::V_S>(vs_); WaitFlag<HardEvent::V_S>(vs_);
                // Candidate rounding: matmul -> half, bias add -> half, SiLU -> half, Mul -> half.
                z1.SetValue(n, static_cast<float>(static_cast<half>(static_cast<float>(static_cast<half>(sums.GetValue(0))) +
                    static_cast<float>(b1_.GetValue(n0 + n)))));
                z2.SetValue(n, static_cast<float>(static_cast<half>(static_cast<float>(static_cast<half>(sums.GetValue(8))) +
                    static_cast<float>(b2_.GetValue(n0 + n)))));
                SetFlag<HardEvent::V_MTE2>(vm2_); WaitFlag<HardEvent::V_MTE2>(vm2_);
            }
            StoreActivated(z1, z2, neg, rounded, out, row*t_.dim+n0, count);
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(m2v_);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(vm2_);
        pipe_.ReleaseEventID<HardEvent::V_S>(vs_);
        pipe_.ReleaseEventID<HardEvent::S_V>(sv_);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(vm3_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(m3v_);
    }
private:
    __aicore__ inline void StoreActivated(LocalTensor<float> z1, LocalTensor<float> z2, LocalTensor<float> neg,
        LocalTensor<half> rounded, LocalTensor<half> out, uint64_t offset, uint32_t count) {
        SetFlag<HardEvent::S_V>(sv_); WaitFlag<HardEvent::S_V>(sv_);
        ApplySilu(z1, neg, count);
        Cast(rounded, z1, RoundMode::CAST_RINT, count); PipeBarrier<PIPE_V>();
        Cast(z1, rounded, RoundMode::CAST_NONE, count); PipeBarrier<PIPE_V>();
        Mul(z1, z1, z2, count); PipeBarrier<PIPE_V>();
        Cast(out, z1, RoundMode::CAST_RINT, count);
        SetFlag<HardEvent::V_MTE3>(vm3_); WaitFlag<HardEvent::V_MTE3>(vm3_);
        DataCopyExtParams store{1, count * 2U, 0, 0, 0};
        DataCopyPad(y_[offset], out, store);
        SetFlag<HardEvent::MTE3_V>(m3v_); WaitFlag<HardEvent::MTE3_V>(m3v_);
    }

    SwiGluTilingData t_;
    TPipe pipe_;
    GlobalTensor<half> x_, w1_, w2_, b1_, b2_, y_;
    TBuf<TPosition::VECCALC> xh_, xf_, wh1_, wh2_, wf1_, wf2_, tmp_, sum_;
    TBuf<TPosition::VECCALC> z1_, z2_, neg_, round_, out_;
    TEventID m2v_, vm2_, vs_, sv_, vm3_, m3v_;
};

class KernelCube {
public:
    TPipe pipe;
    matmul::Matmul<
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, half>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, half, true>,
        matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, half>,
        matmul::MatmulType<TPosition::GM, CubeFormat::ND, half>> mm;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
        GM_ADDR b2, GM_ADDR y, const SwiGluTilingData &t) {
        t_ = t;
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
        uint32_t size = t.tileM * t.tileN;
        pipe.InitBuffer(cq1_, 1, size * 4);
        pipe.InitBuffer(z1_, size * 4);
        pipe.InitBuffer(cq2_, 1, size * 2);
        pipe.InitBuffer(bh1_, t.tileN * 2);
        pipe.InitBuffer(bh2_, t.tileN * 2);
        m2v_ = pipe.AllocEventID<HardEvent::MTE2_V>();
        vm2_ = pipe.AllocEventID<HardEvent::V_MTE2>();
        vm3_ = pipe.AllocEventID<HardEvent::V_MTE3>();
        m3v_ = pipe.AllocEventID<HardEvent::MTE3_V>();
    }
    __aicore__ inline void Process() {
        uint32_t groups = (t_.dim + t_.tileN - 1) / t_.tileN;
        for (uint64_t task = GetBlockIdx(); task < t_.tasks; task += t_.blocks) {
            uint64_t m0 = task / groups * t_.tileM;
            uint32_t n0 = task % groups * t_.tileN;
            uint32_t mc = t_.rows - m0 < t_.tileM ? t_.rows - m0 : t_.tileM;
            uint32_t nc = t_.dim - n0 < t_.tileN ? t_.dim - n0 : t_.tileN;
            auto bh1 = bh1_.Get<half>(); auto bh2 = bh2_.Get<half>();
            DataCopyExtParams biasIn{1,nc*2U,0,0,0};
            DataCopyPadExtParams<half> pad{false,0,0,static_cast<half>(0)};
            DataCopyPad(bh1,b1_[n0],biasIn,pad);
            DataCopyPad(bh2,b2_[n0],biasIn,pad);
            auto c1 = cq1_.AllocTensor<half>();
            Project(x_[m0*t_.dim], w1_[static_cast<uint64_t>(n0)*t_.dim], c1, mc, nc);
            cq1_.EnQue(c1); c1 = cq1_.DeQue<half>();
            auto c2 = cq2_.AllocTensor<half>();
            Project(x_[m0*t_.dim], w2_[static_cast<uint64_t>(n0)*t_.dim], c2, mc, nc);
            cq2_.EnQue(c2); c2 = cq2_.DeQue<half>();
            SetFlag<HardEvent::MTE2_V>(m2v_); WaitFlag<HardEvent::MTE2_V>(m2v_);
            const uint32_t count = mc * t_.tileN;
            auto h=c1; auto out=h;
            auto f1=z1_.Get<float>();
            for (uint32_t col = 0; col < nc; col += 128) {
                uint64_t lanes = nc-col < 128 ? nc-col : 128;
                uint8_t pitch = static_cast<uint8_t>(t_.tileN / 16);
                BinaryRepeatParams repeat{1,1,1,pitch,pitch,0};
                Add(c1[col],c1[col],bh1[col],lanes,static_cast<uint8_t>(mc),repeat); PipeBarrier<PIPE_V>();
                Add(c2[col],c2[col],bh2[col],lanes,static_cast<uint8_t>(mc),repeat);
            }
            PipeBarrier<PIPE_V>();
            Cast(f1,c1,RoundMode::CAST_NONE,count); PipeBarrier<PIPE_V>();
            auto neg=c1.ReinterpretCast<float>();
            ApplySilu(f1, neg, count);
            Cast(h,f1,RoundMode::CAST_RINT,count); PipeBarrier<PIPE_V>();
            Mul(out,h,c2,count);
            SetFlag<HardEvent::V_MTE3>(vm3_); WaitFlag<HardEvent::V_MTE3>(vm3_);
            // UB stride is in 32-byte blocks; GM stride is in bytes.
            DataCopyExtParams store{static_cast<uint16_t>(mc),nc*2U,
                (t_.tileN-nc)*2U/32U,(t_.dim-nc)*2U,0};
            DataCopyPad(y_[m0*t_.dim+n0],out,store);
            SetFlag<HardEvent::MTE3_V>(m3v_); WaitFlag<HardEvent::MTE3_V>(m3v_);
            SetFlag<HardEvent::V_MTE2>(vm2_); WaitFlag<HardEvent::V_MTE2>(vm2_);
            cq1_.FreeTensor(c1); cq2_.FreeTensor(c2);
        }
        pipe.ReleaseEventID<HardEvent::MTE2_V>(m2v_);
        pipe.ReleaseEventID<HardEvent::V_MTE2>(vm2_);
        pipe.ReleaseEventID<HardEvent::V_MTE3>(vm3_);
        pipe.ReleaseEventID<HardEvent::MTE3_V>(m3v_);
    }
private:
    __aicore__ inline void Project(GlobalTensor<half> input, GlobalTensor<half> weight,
        LocalTensor<half> output, uint32_t mc, uint32_t nc) {
        mm.SetTensorA(input);
        mm.SetTensorB(weight, true);
        mm.SetTail(mc, nc, t_.dim);
        mm.Iterate<true>();
        mm.GetTensorC<true>(output, 0, true);
        mm.End();
    }

    SwiGluTilingData t_;
    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_;
    TQue<TPosition::VECIN,1> cq1_,cq2_;
    TBuf<TPosition::VECCALC> z1_,bh1_,bh2_;
    TEventID m2v_,vm2_,vm3_,m3v_;
};

template <bool StaticArena>
class ResidentStorage;

template <>
class ResidentStorage<false> {
public:
    __aicore__ inline void Init(uint32_t bytes, TEventID &m2v, TEventID &vm2, TEventID &vm3, TEventID &m3v) {
        pipe_.InitBuffer(arena_, bytes);
        m2v = pipe_.AllocEventID<HardEvent::MTE2_V>();
        vm2 = pipe_.AllocEventID<HardEvent::V_MTE2>();
        vm3 = pipe_.AllocEventID<HardEvent::V_MTE3>();
        m3v = pipe_.AllocEventID<HardEvent::MTE3_V>();
    }

    __aicore__ inline LocalTensor<float> Get() {
        return arena_.Get<float>();
    }

    __aicore__ inline void Release(TEventID m2v, TEventID vm2, TEventID vm3, TEventID m3v) {
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(m2v);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(vm2);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(vm3);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(m3v);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> arena_;
};

template <>
class ResidentStorage<true> {
public:
    __aicore__ inline void Init(uint32_t bytes, TEventID &m2v, TEventID &vm2, TEventID &vm3, TEventID &m3v) {
        bytes_ = bytes;
        m2v = 0;
        vm2 = 0;
        vm3 = 0;
        m3v = 0;
    }

    __aicore__ inline LocalTensor<float> Get() {
        return LocalTensor<float>(TPosition::VECCALC, 0, bytes_ / 4U);
    }

    __aicore__ inline uint32_t Bytes() const {
        return bytes_;
    }

    __aicore__ inline void Release(TEventID, TEventID, TEventID, TEventID) {}

private:
    uint32_t bytes_;
};

template <bool PackedEP = false, bool MergedInput = false, bool StaticArena = false>
class KernelResident {
public:
    template <typename Tiling>
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
        GM_ADDR b2, GM_ADDR y, const Tiling &t) {
        rows_=t.rows; dim_=t.dim; pitch_=t.alignedDim; blocks_=t.blocks; batch_=t.tileM;
        BindSwiGluInputs(x, w1, b1, w2, b2, y, x_, w1_, b1_, w2_, b2_, y_);
        uint32_t weights=dim_*pitch_;
        // FP32 weights live for all rows. Product zones hold half weights only
        // during initialization; the two bias regions retain half data for all rows.
        // All zone starts are multiples of 32 bytes because pitch is /16 aligned.
        storage_.Init(16U*weights+(22U*batch_+8U)*pitch_, m2v_, vm2_, vm3_, m3v_);
    }
    __aicore__ inline void Process() {
        LocalTensor<float> f=storage_.Get();
        const uint32_t w=dim_*pitch_;
        auto wf1=f; auto wf2=f[w];
        const uint32_t bp=batch_*pitch_;
        auto p1=f[2*w]; auto p2=f[3*w];
        auto wh1=p1.ReinterpretCast<half>();
        auto wh2=p2.ReinterpretCast<half>();
        uint32_t o=4*w;
        auto xh=f[o].ReinterpretCast<half>(); o+=bp/2;
        auto xf=f[o]; o+=bp;
        if constexpr (MergedInput) {
            // Destination floats precede all initialization sources.
            xf=f[2*w]; p1=f[2*w+bp]; p2=f[3*w+bp];
            wh1=p1.ReinterpretCast<half>(); wh2=wh1[w];
            xh=wh1[2*w]; o=4*w+bp;
        }
        auto bf1=f[o]; o+=pitch_;
        auto bf2=f[o]; o+=pitch_;
        auto z1=f[o]; o+=batch_*pitch_;
        auto z2=f[o]; o+=batch_*pitch_;
        auto neg=f[o]; o+=batch_*pitch_;
        auto h=f[o].ReinterpretCast<half>(); o+=batch_*pitch_/2;
        auto out=f[o].ReinterpretCast<half>();
        auto bh1=bf1.ReinterpretCast<half>(); auto bh2=bf2.ReinterpretCast<half>();
        if constexpr (PackedEP) bh2=bh1[pitch_];
        DataCopyPadExtParams<half> pad{true,0,static_cast<uint8_t>(pitch_-dim_),static_cast<half>(0)};
        const uint64_t firstRow=static_cast<uint64_t>(GetBlockIdx())*batch_;
        const uint32_t firstRc=rows_-firstRow<batch_ ? static_cast<uint32_t>(rows_-firstRow) : batch_;
        LoadFirstTile(xh, wh1, wh2, bh1, bh2, wf1, wf2, z1, z2, pad, firstRow, firstRc, w);
        uint8_t stride=static_cast<uint8_t>(pitch_/8);
        BinaryRepeatParams repeat{1,1,1,stride,stride,0};
        for(uint64_t row=firstRow;row<rows_;row+=static_cast<uint64_t>(blocks_)*batch_) {
            const uint32_t rc=rows_-row<batch_ ? static_cast<uint32_t>(rows_-row) : batch_;
            const uint32_t count=batch_==1 ? dim_ : rc*pitch_;
            DataCopyExtParams batchCopy{static_cast<uint16_t>(rc),dim_*2U,0,0,0};
            LoadBatchInput(xh, xf, row, firstRow, batchCopy, pad, count);
            for(uint32_t r=0;r<rc;++r) {
                Mul(p1,wf1,xf[r*pitch_],static_cast<uint64_t>(dim_),static_cast<uint8_t>(dim_),repeat);
                Mul(p2,wf2,xf[r*pitch_],static_cast<uint64_t>(dim_),static_cast<uint8_t>(dim_),repeat);
                PipeBarrier<PIPE_V>();
                ReduceDotPair(z1[r*pitch_], z2[r*pitch_], p1, p2, stride);
            }
            FinishTile(h, out, bh1, bh2, z1, z2, neg, rc, count);
            SetFlag<HardEvent::V_MTE3>(vm3_); WaitFlag<HardEvent::V_MTE3>(vm3_);
            DataCopyPad(y_[row*dim_],out,batchCopy);
            SetFlag<HardEvent::MTE3_V>(m3v_); WaitFlag<HardEvent::MTE3_V>(m3v_);
            SetFlag<HardEvent::V_MTE2>(vm2_); WaitFlag<HardEvent::V_MTE2>(vm2_);
        }
        storage_.Release(m2v_, vm2_, vm3_, m3v_);
    }
private:
    __aicore__ inline void LoadBatchInput(LocalTensor<half> xh, LocalTensor<float> xf,
        uint64_t row, uint64_t firstRow, const DataCopyExtParams &batchCopy,
        const DataCopyPadExtParams<half> &pad, uint32_t count) {
        if(row!=firstRow) {
            DataCopyPad(xh,x_[row*dim_],batchCopy,pad);
            SetFlag<HardEvent::MTE2_V>(m2v_); WaitFlag<HardEvent::MTE2_V>(m2v_);
        }
        if (!MergedInput || row!=firstRow) {
            Cast(xf,xh,RoundMode::CAST_NONE,count); PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void LoadFirstTile(LocalTensor<half> xh, LocalTensor<half> wh1, LocalTensor<half> wh2,
        LocalTensor<half> bh1, LocalTensor<half> bh2, LocalTensor<float> wf1, LocalTensor<float> wf2,
        LocalTensor<float> z1, LocalTensor<float> z2, const DataCopyPadExtParams<half> &pad,
        uint64_t firstRow, uint32_t firstRc, uint32_t w) {
        DataCopyExtParams weights{static_cast<uint16_t>(dim_),dim_*2U,0,0,0};
        DataCopyExtParams firstIn{static_cast<uint16_t>(firstRc),dim_*2U,0,0,0};
        DataCopyPad(xh,x_[firstRow*dim_],firstIn,pad);
        DataCopyPad(wh1,w1_,weights,pad); DataCopyPad(wh2,w2_,weights,pad);
        DataCopyExtParams vec{1,dim_*2U,0,0,0};
        DataCopyPad(bh1,b1_,vec,pad); DataCopyPad(bh2,b2_,vec,pad);
        SetFlag<HardEvent::MTE2_V>(m2v_); WaitFlag<HardEvent::MTE2_V>(m2v_);
        if constexpr (MergedInput) {
            Cast(wf1,wh1,RoundMode::CAST_NONE,2U*w+firstRc*pitch_);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(wf1,wh1,RoundMode::CAST_NONE,w); PipeBarrier<PIPE_V>();
            Cast(wf2,wh2,RoundMode::CAST_NONE,w); PipeBarrier<PIPE_V>();
        }
        if constexpr (PackedEP) {
            if(pitch_!=dim_) Duplicate(z1,0.0f,2U*pitch_);
        } else if(batch_>1 && pitch_!=dim_) {
            Duplicate(z1,0.0f,batch_*pitch_); Duplicate(z2,0.0f,batch_*pitch_);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceDotPair(LocalTensor<float> z1, LocalTensor<float> z2,
        LocalTensor<float> p1, LocalTensor<float> p2, uint8_t stride) {
        if constexpr (StaticArena && PackedEP && MergedInput) {
            if (dim_ == 64U && pitch_ == 64U) {
                // Reduce each complete K64 dot in two stages using adjacent block sums.
                LocalTensor<float> sums(TPosition::VECCALC, storage_.Bytes(), 1024U);
                BlockReduceSum<float>(sums,p1,64,64,1,1,8);
                BlockReduceSum<float>(sums[512],p2,64,64,1,1,8);
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float>(z1,sums,8,64,1,1,8);
                BlockReduceSum<float>(z2,sums[512],8,64,1,1,8);
                PipeBarrier<PIPE_V>();
                return;
            }
        }
        WholeReduceSum(z1,p1,dim_,dim_,1,1,stride);
        WholeReduceSum(z2,p2,dim_,dim_,1,1,stride);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FinishTile(LocalTensor<half> h, LocalTensor<half> out,
        LocalTensor<half> bh1, LocalTensor<half> bh2, LocalTensor<float> z1,
        LocalTensor<float> z2, LocalTensor<float> neg, uint32_t rc, uint32_t count) {
        if constexpr (PackedEP) {
            Cast(h,z1,RoundMode::CAST_RINT,2U*pitch_); PipeBarrier<PIPE_V>();
            Add(h,h,bh1,2U*pitch_); PipeBarrier<PIPE_V>();
        } else {
        Cast(h,z1,RoundMode::CAST_RINT,count); PipeBarrier<PIPE_V>();
        Cast(out,z2,RoundMode::CAST_RINT,count); PipeBarrier<PIPE_V>();
        uint8_t halfStride=static_cast<uint8_t>(pitch_/16);
        BinaryRepeatParams biasRepeat{1,1,1,halfStride,halfStride,0};
        Add(h,h,bh1,static_cast<uint64_t>(dim_),static_cast<uint8_t>(rc),biasRepeat); PipeBarrier<PIPE_V>();
        Add(out,out,bh2,static_cast<uint64_t>(dim_),static_cast<uint8_t>(rc),biasRepeat);
        PipeBarrier<PIPE_V>();
        }
        Cast(z1,h,RoundMode::CAST_NONE,count); PipeBarrier<PIPE_V>();
        ApplySilu(z1, neg, count);
        Cast(h,z1,RoundMode::CAST_RINT,count); PipeBarrier<PIPE_V>();
        Mul(out,h,out,count);
    }

    ResidentStorage<StaticArena> storage_;
    GlobalTensor<half> x_,w1_,w2_,b1_,b2_,y_;
    uint64_t rows_; uint32_t dim_,pitch_,blocks_,batch_;
    TEventID m2v_,vm2_,vm3_,m3v_;
};

// Complete K64 dots partitioned over aligned output-column owners.
// Aligned K32 packs two complete channels into each full64-lane vector.
// One output-row owner retains both projections through the original Half stages.

} // namespace AscendC::SwiGluKernel
