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
// Kernel侧核函数实现
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "swi_glu_tiling.h"
#include "tiling_key_swi_glu.h"

// SiLU(x) * y 的公共实现：denom = 1 + exp(min(-x, 80))，结果写回 out。
// out 允许与 x 别名（原地计算）；denom 允许与 tmp 别名。
// 保持原实现的 PipeBarrier 顺序不变。
__aicore__ inline void SwiGluSiluMul(AscendC::LocalTensor<float> out,
                                     AscendC::LocalTensor<float> x,
                                     AscendC::LocalTensor<float> y,
                                     AscendC::LocalTensor<float> tmp,
                                     AscendC::LocalTensor<float> denom,
                                     uint32_t count)
{
    AscendC::Muls(tmp, x, -1.0f, static_cast<int32_t>(count));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins(tmp, tmp, 80.0f, static_cast<int32_t>(count));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(denom, tmp, static_cast<int32_t>(count));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(denom, denom, 1.0f, static_cast<int32_t>(count));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(out, x, denom, static_cast<int32_t>(count));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(out, out, y, static_cast<int32_t>(count));
    AscendC::PipeBarrier<PIPE_V>();
}

// 保持原实现的 FP16 舍入行为：a/b 先写回 half，再转回 float。
__aicore__ inline void RoundTripFp16(AscendC::LocalTensor<float> a,
                                     AscendC::LocalTensor<float> b,
                                     AscendC::LocalTensor<half> h,
                                     uint32_t offset, uint32_t count)
{
    AscendC::Cast(h, a, AscendC::RoundMode::CAST_NONE, count);
    AscendC::Cast(h[offset], b, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(a, h, AscendC::RoundMode::CAST_NONE, count);
    AscendC::Cast(b, h[offset], AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
}

template <class T>
class KernelSwiGlu {
public:
    __aicore__ inline KernelSwiGlu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                const SwiGluTilingData &tiling)
    {
        m_ = tiling.m;
        k_ = tiling.k;
        n_ = tiling.n;
        weight1Transposed_ = tiling.weight1Transposed;
        weight2Transposed_ = tiling.weight2Transposed;
        blockDim_ = tiling.blockDim;

        inputXGm_.SetGlobalBuffer((__gm__ T *)input_x, static_cast<uint64_t>(m_) * k_);
        weight1Gm_.SetGlobalBuffer((__gm__ T *)weight_1, static_cast<uint64_t>(k_) * n_);
        bias1Gm_.SetGlobalBuffer((__gm__ T *)bias_1, n_);
        weight2Gm_.SetGlobalBuffer((__gm__ T *)weight_2, static_cast<uint64_t>(k_) * n_);
        bias2Gm_.SetGlobalBuffer((__gm__ T *)bias_2, n_);
        outputGm_.SetGlobalBuffer((__gm__ T *)output, static_cast<uint64_t>(m_) * n_);

        pipe_.InitBuffer(branch1Buf_, SWIGLU_OUT_TILE * sizeof(float));
        pipe_.InitBuffer(branch2Buf_, SWIGLU_OUT_TILE * sizeof(float));
        pipe_.InitBuffer(tmpBuf_, SWIGLU_OUT_TILE * sizeof(float));
        pipe_.InitBuffer(denomBuf_, SWIGLU_OUT_TILE * sizeof(float));
        pipe_.InitBuffer(outBuf_, SWIGLU_OUT_TILE * sizeof(float));
        pipe_.InitBuffer(outQueue_, 1, SWIGLU_OUT_TILE * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t coreNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        if (blockDim_ != 0U && blockDim_ < coreNum) {
            coreNum = blockDim_;
        }
        if (coreNum == 0U || coreIdx >= coreNum) {
            return;
        }

        const uint32_t tilesPerRow = (static_cast<uint64_t>(n_) + SWIGLU_OUT_TILE - 1U) / SWIGLU_OUT_TILE;
        if (tilesPerRow == 0U) {
            return;
        }
        const uint64_t totalTasks = static_cast<uint64_t>(m_) * tilesPerRow;

        // Small workload fast path: avoid too many idle AIVs fighting for tiny tasks.
        if (totalTasks < coreNum) {
            coreNum = totalTasks;
            if (coreNum == 0U || coreIdx >= coreNum) {
                return;
            }
        }

        // Key optimization over v9:
        // parallelize by (row, N-tile), not row only.  This keeps all AIVs busy
        // when M is small but N is large, while each task owns a disjoint output
        // range so there is no cross-core write conflict.
        for (uint64_t task = coreIdx; task < totalTasks; task += coreNum) {
            const uint32_t row = task / tilesPerRow;
            const uint32_t tileId = task - row * tilesPerRow;
            const uint32_t tileBase = tileId * SWIGLU_OUT_TILE;
            uint32_t count = n_ - tileBase;
            if (count > SWIGLU_OUT_TILE) {
                count = SWIGLU_OUT_TILE;
            }
            ComputeTile(row, tileBase, count);
        }
    }

private:
    __aicore__ inline uint64_t WeightIndex(uint32_t k, uint32_t n, uint32_t transposed) const
    {
        return transposed == 0U ? (static_cast<uint64_t>(k) * n_ + n) :
                                  (static_cast<uint64_t>(n) * k_ + k);
    }

    // 逐列累加两个投影的 FP16 结果，保持 v9 的算术顺序与舍入行为。
    __aicore__ inline void AccumulateTile(uint64_t xBase, uint32_t tileBase, uint32_t count,
                                          AscendC::LocalTensor<float> branch1Local,
                                          AscendC::LocalTensor<float> branch2Local)
    {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t col = tileBase + i;
            float acc1 = 0.0f;
            float acc2 = 0.0f;

            for (uint32_t k = 0; k < k_; ++k) {
                const float xVal = static_cast<float>(inputXGm_.GetValue(xBase + k));
                const float w1 = static_cast<float>(
                    weight1Gm_.GetValue(WeightIndex(k, col, weight1Transposed_)));
                const float w2 = static_cast<float>(
                    weight2Gm_.GetValue(WeightIndex(k, col, weight2Transposed_)));
                acc1 += xVal * w1;
                acc2 += xVal * w2;
            }

            const T mm1 = static_cast<T>(acc1);
            const T mm2 = static_cast<T>(acc2);
            const T biased1 = static_cast<T>(static_cast<float>(mm1) +
                                             static_cast<float>(bias1Gm_.GetValue(col)));
            const T biased2 = static_cast<T>(static_cast<float>(mm2) +
                                             static_cast<float>(bias2Gm_.GetValue(col)));

            branch1Local.SetValue(i, static_cast<float>(biased1));
            branch2Local.SetValue(i, static_cast<float>(biased2));
        }
    }

    __aicore__ inline void StoreTile(uint64_t outBase, uint32_t tileBase, uint32_t count,
                                     AscendC::LocalTensor<float> outLocal)
    {
        AscendC::LocalTensor<T> outCopyLocal = outQueue_.AllocTensor<T>();
        AscendC::Cast(outCopyLocal, outLocal, AscendC::RoundMode::CAST_NONE,
                      static_cast<int32_t>(count));
        outQueue_.EnQue<T>(outCopyLocal);

        AscendC::LocalTensor<T> copyLocal = outQueue_.DeQue<T>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(count * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(outputGm_[outBase + tileBase], copyLocal, copyParams);
        outQueue_.FreeTensor(copyLocal);
    }

    __aicore__ inline void ComputeTile(uint32_t row, uint32_t tileBase, uint32_t count)
    {
        const uint64_t xBase = static_cast<uint64_t>(row) * k_;
        const uint64_t outBase = static_cast<uint64_t>(row) * n_;

        AscendC::LocalTensor<float> branch1Local = branch1Buf_.Get<float>();
        AscendC::LocalTensor<float> branch2Local = branch2Buf_.Get<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
        AscendC::LocalTensor<float> denomLocal = denomBuf_.Get<float>();
        AscendC::LocalTensor<float> outLocal = outBuf_.Get<float>();

        AccumulateTile(xBase, tileBase, count, branch1Local, branch2Local);
        SwiGluSiluMul(outLocal, branch1Local, branch2Local, tmpLocal, denomLocal, count);
        StoreTile(outBase, tileBase, count, outLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> branch1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> branch2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> denomBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue_;

    AscendC::GlobalTensor<T> inputXGm_;
    AscendC::GlobalTensor<T> weight1Gm_;
    AscendC::GlobalTensor<T> bias1Gm_;
    AscendC::GlobalTensor<T> weight2Gm_;
    AscendC::GlobalTensor<T> bias2Gm_;
    AscendC::GlobalTensor<T> outputGm_;

    uint32_t m_ = 0U;
    uint32_t k_ = 0U;
    uint32_t n_ = 0U;
    uint32_t weight1Transposed_ = 0U;
    uint32_t weight2Transposed_ = 0U;
    uint32_t blockDim_ = 1U;
};

// Two projections share the tile schedule. Each AIV consumes only its own
// synchronous Cube outputs; no grid barrier or second kernel is required.
// Small GEMV: one vector instruction evaluates up to 16 dot products.
class KernelSwiGluSmall {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR b1,
        GM_ADDR w2, GM_ADDR b2, GM_ADDR out, const SwiGluTilingData &t) {
        t_ = t;
        x_.SetGlobalBuffer((__gm__ half *)x);
        w1_.SetGlobalBuffer((__gm__ half *)w1);
        w2_.SetGlobalBuffer((__gm__ half *)w2);
        b1_.SetGlobalBuffer((__gm__ half *)b1);
        b2_.SetGlobalBuffer((__gm__ half *)b2);
        out_.SetGlobalBuffer((__gm__ half *)out);
        pitch_ = (t.k + 63) / 64 * 64;
        pipe_.InitBuffer(in_, 1, (33 * pitch_ + 32) * sizeof(half));
        pipe_.InitBuffer(outQ_, 1, 32);
        pipe_.InitBuffer(floats_, (33 * pitch_ + 96) * sizeof(float));
    }
    __aicore__ inline void Process() {
        const uint32_t nt = (static_cast<uint64_t>(t_.n) + 15U) / 16U;
        if (nt == 0U) {
            return;
        }
        const uint64_t tasks = static_cast<uint64_t>(t_.m) * nt;
        for (uint64_t task = AscendC::GetBlockIdx(); task < tasks; task += t_.blockDim) {
            const uint32_t row = task / nt;
            const uint32_t col = task % nt * 16U;
            const uint32_t count = t_.n - col < 16U ? t_.n - col : 16U;
            AscendC::LocalTensor<half> h = in_.AllocTensor<half>();
            LoadTile(row, col, count, h);
            ComputeTile(row, col, count, h);
            in_.FreeTensor(h);
        }
    }
private:
    // 把一个 (row, N-tile) 的 x/w1/w2/b1/b2 搬到 UB。
    __aicore__ inline void LoadTile(uint32_t row, uint32_t col, uint32_t count,
                                    AscendC::LocalTensor<half> &h) {
        AscendC::DataCopyPadExtParams<half> pad{true, 0, 0, 0};
        AscendC::DataCopyExtParams xp{1, t_.k * 2, 0, 0, 0};
        AscendC::DataCopyPad(h, x_[(uint64_t)row * t_.k], xp, pad);
        const uint32_t stride = (pitch_ - (t_.k + 15U) / 16U * 16U) / 16U;
        AscendC::DataCopyExtParams wp{static_cast<uint16_t>(count), t_.k * 2, 0, stride, 0};
        AscendC::DataCopyPad(h[pitch_], w1_[(uint64_t)col * t_.k], wp, pad);
        AscendC::DataCopyPad(h[17U * pitch_], w2_[(uint64_t)col * t_.k], wp, pad);
        AscendC::DataCopyExtParams bp{1, count * 2, 0, 0, 0};
        AscendC::DataCopyPad(h[33U * pitch_], b1_[col], bp, pad);
        AscendC::DataCopyPad(h[33U * pitch_ + 16U], b2_[col], bp, pad);
        in_.EnQue(h);
        h = in_.DeQue<half>();
    }

    // 按 64 列一段做乘加与 WholeReduceSum，累加两个投影。
    __aicore__ inline void AccumulateChunks(AscendC::LocalTensor<float> a,
                                            AscendC::LocalTensor<float> b,
                                            AscendC::LocalTensor<float> wf1,
                                            AscendC::LocalTensor<float> wf2,
                                            AscendC::LocalTensor<float> xf,
                                            AscendC::LocalTensor<float> tmp,
                                            uint32_t count) {
        for (uint32_t chunk = 0; chunk < t_.k; chunk += 64U) {
            const uint32_t valid = t_.k - chunk < 64U ? t_.k - chunk : 64U;
            const uint8_t repeatStride = static_cast<uint8_t>(pitch_ / 8U);
            AscendC::Mul(wf1[chunk], wf1[chunk], xf[chunk], valid,
                         static_cast<uint8_t>(count), {1, 1, 1, repeatStride, repeatStride, 0});
            AscendC::Mul(wf2[chunk], wf2[chunk], xf[chunk], valid,
                         static_cast<uint8_t>(count), {1, 1, 1, repeatStride, repeatStride, 0});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum(tmp, wf1[chunk], valid, static_cast<uint8_t>(count),
                                    1, 1, repeatStride);
            AscendC::WholeReduceSum(tmp[16], wf2[chunk], valid, static_cast<uint8_t>(count),
                                    1, 1, repeatStride);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(a, a, tmp, count);
            AscendC::Add(b, b, tmp[16], count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreTile(uint32_t row, uint32_t col, uint32_t count,
                                     AscendC::LocalTensor<float> a) {
        AscendC::LocalTensor<half> y = outQ_.AllocTensor<half>();
        AscendC::Cast(y, a, AscendC::RoundMode::CAST_NONE, count);
        outQ_.EnQue(y);
        y = outQ_.DeQue<half>();
        AscendC::DataCopyExtParams bp{1, count * 2, 0, 0, 0};
        AscendC::DataCopyPad(out_[(uint64_t)row * t_.n + col], y, bp);
        outQ_.FreeTensor(y);
    }

    // bias 之后做两次 FP16 往返，再算激活并写回。
    __aicore__ inline void FinishTile(uint32_t row, uint32_t col, uint32_t count,
                                      AscendC::LocalTensor<half> &h) {
        AscendC::LocalTensor<float> f = floats_.Get<float>();
        AscendC::LocalTensor<float> a = f[33U * pitch_];
        AscendC::LocalTensor<float> b = f[33U * pitch_ + 16U];
        AscendC::LocalTensor<float> ba = f[33U * pitch_ + 32U];
        AscendC::LocalTensor<float> bb = f[33U * pitch_ + 48U];
        AscendC::LocalTensor<float> tmp = f[33U * pitch_ + 64U];
        RoundTripFp16(a, b, h, 16U, count);
        AscendC::Add(a, a, ba, count);
        AscendC::Add(b, b, bb, count);
        AscendC::PipeBarrier<PIPE_V>();
        RoundTripFp16(a, b, h, 16U, count);
        SwiGluSiluMul(a, a, b, tmp, tmp, count);
        StoreTile(row, col, count, a);
    }

    __aicore__ inline void ComputeTile(uint32_t row, uint32_t col, uint32_t count,
                                       AscendC::LocalTensor<half> &h) {
        AscendC::LocalTensor<float> f = floats_.Get<float>();
        AscendC::LocalTensor<float> wf1 = f;
        AscendC::LocalTensor<float> wf2 = f[16U * pitch_];
        AscendC::LocalTensor<float> xf = f[32U * pitch_];
        AscendC::LocalTensor<float> a = f[33U * pitch_];
        AscendC::LocalTensor<float> b = f[33U * pitch_ + 16U];
        AscendC::LocalTensor<float> ba = f[33U * pitch_ + 32U];
        AscendC::LocalTensor<float> bb = f[33U * pitch_ + 48U];
        AscendC::LocalTensor<float> tmp = f[33U * pitch_ + 64U];
        AscendC::Cast(wf1, h[pitch_], AscendC::RoundMode::CAST_NONE, pitch_ * count);
        AscendC::Cast(wf2, h[17U * pitch_], AscendC::RoundMode::CAST_NONE, pitch_ * count);
        AscendC::Cast(xf, h, AscendC::RoundMode::CAST_NONE, pitch_);
        AscendC::Cast(ba, h[33U * pitch_], AscendC::RoundMode::CAST_NONE, count);
        AscendC::Cast(bb, h[33U * pitch_ + 16U], AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(a, 0.0f, count);
        AscendC::Duplicate(b, 0.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AccumulateChunks(a, b, wf1, wf2, xf, tmp, count);
        FinishTile(row, col, count, h);
    }
private:
    SwiGluTilingData t_;
    uint32_t pitch_;
    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<half> x_, w1_, w2_, b1_, b2_, out_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> in_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQ_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floats_;
};

class KernelSwiGluCube {
public:
    using H = half;
    static constexpr uint32_t TM = 32, TN = 128, TILE = TM * TN;
    matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, H>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, H, true>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, H>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, H>> mm;
    __aicore__ inline void Init(AscendC::TPipe *pipe, GM_ADDR x, GM_ADDR w1,
        GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR out, GM_ADDR workspace,
        const SwiGluTilingData &t) {
        t_ = t;
        x_.SetGlobalBuffer((__gm__ H *)x);
        w1_.SetGlobalBuffer((__gm__ H *)w1);
        w2_.SetGlobalBuffer((__gm__ H *)w2);
        b1_.SetGlobalBuffer((__gm__ H *)b1);
        b2_.SetGlobalBuffer((__gm__ H *)b2);
        out_.SetGlobalBuffer((__gm__ H *)out);
        c1_.SetGlobalBuffer((__gm__ H *)workspace);
        c2_.SetGlobalBuffer((__gm__ H *)workspace + (uint64_t)t.m * t.n);
        pipe->InitBuffer(input_, 1, (2 * TILE + 2 * TN) * sizeof(H));
        pipe->InitBuffer(output_, 1, TILE * sizeof(H));
        pipe->InitBuffer(floatBuf_, (3 * TILE + 2 * TN) * sizeof(float));
    }
    __aicore__ inline void Process() {
        const uint32_t nt = (static_cast<uint64_t>(t_.n) + TN - 1) / TN;
        if (nt == 0U) {
            return;
        }
        const uint64_t tasks = ((static_cast<uint64_t>(t_.m) + TM - 1) / TM) * nt;
        for (uint64_t task = AscendC::GetBlockIdx(); task < tasks; task += t_.blockDim) {
            uint32_t row = task / nt * TM;
            uint32_t col = task % nt * TN;
            uint32_t rows = t_.m - row < TM ? t_.m - row : TM;
            uint32_t cols = t_.n - col < TN ? t_.n - col : TN;
            uint64_t offset = (uint64_t)row * t_.n + col;
            mm.SetTensorA(x_[(uint64_t)row * t_.k]);
            mm.SetTensorB(w1_[(uint64_t)col * t_.k], true);
            mm.SetSingleShape(rows, cols, t_.k);
            mm.template IterateAll<true>(c1_[offset]);
            mm.End();
            mm.SetTensorA(x_[(uint64_t)row * t_.k]);
            mm.SetTensorB(w2_[(uint64_t)col * t_.k], true);
            mm.SetSingleShape(rows, cols, t_.k);
            mm.template IterateAll<true>(c2_[offset]);
            mm.End();
            Epilogue(offset, col, rows, cols);
        }
    }
private:
    // 把 C1/C2 和 bias 从 GM 搬到 UB。
    __aicore__ inline void LoadTile(uint64_t offset, uint32_t col, uint32_t rows,
                                    uint32_t cols, uint32_t alignedCols,
                                    AscendC::LocalTensor<H> &h) {
        AscendC::Duplicate(h, (H)0, 2 * TILE + 2 * TN);
        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams cp{static_cast<uint16_t>(rows), cols * 2,
                                    (t_.n - cols) * 2, (TN - alignedCols) / 16, 0};
        AscendC::DataCopyPadExtParams<H> pad{true, 0, 0, 0};
        AscendC::DataCopyPad(h, c1_[offset], cp, pad);
        AscendC::DataCopyPad(h[TILE], c2_[offset], cp, pad);
        AscendC::DataCopyExtParams bp{1, cols * 2, 0, 0, 0};
        AscendC::DataCopyPad(h[2 * TILE], b1_[col], bp, pad);
        AscendC::DataCopyPad(h[2 * TILE + TN], b2_[col], bp, pad);
        input_.EnQue(h);
        h = input_.DeQue<H>();
    }

    // bias 按行广播加到两个投影上。
    __aicore__ inline void AddBias(AscendC::LocalTensor<float> a,
                                   AscendC::LocalTensor<float> b,
                                   AscendC::LocalTensor<float> ba,
                                   AscendC::LocalTensor<float> bb,
                                   uint32_t rows) {
        for (uint32_t row = 0; row < rows; ++row) {
            AscendC::Add(a[row * TN], a[row * TN], ba, TN);
            AscendC::Add(b[row * TN], b[row * TN], bb, TN);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreTile(uint64_t offset, uint32_t col, uint32_t rows,
                                     uint32_t cols, uint32_t alignedCols,
                                     AscendC::LocalTensor<float> a) {
        AscendC::LocalTensor<H> out = output_.AllocTensor<H>();
        AscendC::Cast(out, a, AscendC::RoundMode::CAST_NONE, rows * TN);
        output_.EnQue(out);
        out = output_.DeQue<H>();
        AscendC::DataCopyExtParams ocp{static_cast<uint16_t>(rows), cols * 2,
                                     (TN - cols) * 2, (t_.n - cols) * 2, 0};
        // For UB -> GM, source stride uses 32-byte blocks. The payload length
        // is rounded up to a block in UB while the GM payload stays byte-exact.
        ocp.srcStride = (TN - alignedCols) / 16;
        AscendC::DataCopyPad(out_[offset], out, ocp);
        output_.FreeTensor(out);
    }

    __aicore__ inline void ComputeTile(uint64_t offset, uint32_t col, uint32_t rows,
                                       uint32_t cols, uint32_t alignedCols,
                                       AscendC::LocalTensor<H> &h) {
        AscendC::LocalTensor<float> f = floatBuf_.Get<float>();
        AscendC::LocalTensor<float> a = f;
        AscendC::LocalTensor<float> b = f[TILE];
        AscendC::LocalTensor<float> tmp = f[2 * TILE];
        AscendC::LocalTensor<float> ba = f[3 * TILE];
        AscendC::LocalTensor<float> bb = f[3 * TILE + TN];
        const uint32_t count = rows * TN;
        AscendC::Cast(a, h, AscendC::RoundMode::CAST_NONE, count);
        AscendC::Cast(b, h[TILE], AscendC::RoundMode::CAST_NONE, count);
        AscendC::Cast(ba, h[2 * TILE], AscendC::RoundMode::CAST_NONE, TN);
        AscendC::Cast(bb, h[2 * TILE + TN], AscendC::RoundMode::CAST_NONE, TN);
        AscendC::PipeBarrier<PIPE_V>();
        AddBias(a, b, ba, bb, rows);
        RoundTripFp16(a, b, h, TILE, count);
        SwiGluSiluMul(a, a, b, tmp, tmp, count);
        StoreTile(offset, col, rows, cols, alignedCols, a);
    }

    __aicore__ inline void Epilogue(uint64_t offset, uint32_t col,
                                   uint32_t rows, uint32_t cols) {
        const uint32_t alignedCols = (cols + 15U) / 16U * 16U;
        AscendC::LocalTensor<H> h = input_.AllocTensor<H>();
        LoadTile(offset, col, rows, cols, alignedCols, h);
        ComputeTile(offset, col, rows, cols, alignedCols, h);
        input_.FreeTensor(h);
    }
    SwiGluTilingData t_;
    AscendC::GlobalTensor<H> x_, w1_, w2_, b1_, b2_, out_, c1_, c2_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> input_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatBuf_;
};

template <typename DT_INPUT_X, uint64_t KERNEL_MODE, bool USE_CUBE>
__global__ __aicore__ void swi_glu(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
    GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
    if constexpr (USE_CUBE) {
        AscendC::TPipe pipe;
        KernelSwiGluCube op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm, &tiling_data.cube);
        op.Init(&pipe, input_x, weight_1, bias_1, weight_2, bias_2, output,
                AscendC::GetUserWorkspace(workspace), tiling_data);
        op.Process();
    } else {
        if ASCEND_IS_AIV {
            if (tiling_data.k > 0 && tiling_data.k <= 512 &&
                tiling_data.weight1Transposed && tiling_data.weight2Transposed) {
                KernelSwiGluSmall op;
                op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, tiling_data);
                op.Process();
            } else {
                KernelSwiGlu<DT_INPUT_X> op;
                op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, tiling_data);
                op.Process();
            }
        }
    }
}
