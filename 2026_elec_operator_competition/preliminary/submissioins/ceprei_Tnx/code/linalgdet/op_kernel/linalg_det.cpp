/*
 * Copyright (c) 2026 ceprei_Tnx.
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
#include "kernel_operator.h"
#ifndef DET_USE_VBARRIER
#define DET_USE_VBARRIER 0
#endif
#ifndef DET_LEAN_RESIDENT
#define DET_LEAN_RESIDENT 1
#endif
#ifndef DET_LEAN_CUBE
#define DET_LEAN_CUBE 1
#endif
#ifndef DET_FUSE_CUBE
#define DET_FUSE_CUBE 1
#endif
#ifndef DET_LEAN_QUOT
#define DET_LEAN_QUOT 1
#endif
#ifndef DET_LEAN_UPDATE
#define DET_LEAN_UPDATE 1
#endif
#ifndef DET_SKIP_ACTIVE_SCAN
#define DET_SKIP_ACTIVE_SCAN 1
#define DET_BIG_CORR 1   // 纠正阶段用满 scratch_（仅 LEAN_CUBE+FUSE_CUBE 分支有效）
#define DET_K32FULL 1    // cube 的 K=32 槽全部用满，省掉 lh_ 的整块清零
#define DET_LEAN_PANEL 1 // panel 内更新改为单精度乘减（有精度风险，可改 0 回退）
#define DET_LEAN_AVAIL 1 // resident: 复用常驻「可用行」掩码 + 去掉恒零低半部分的搬运
#define DET_ATOMIC_C 1   // cube: 直接原子累加到 W（A 取负），省掉 c0_ 回写/回读与纠正 pass
#define DET_WIDE_BCAST 1 // PanelGroup: 广播乘法改用 64 lane（原来每个 repeat 只用 8 lane）
#define DET_DEADLOW 1    // cube: 删掉对恒零的列块低半部分做 GatherMask
#define DET_SKIPHSPLIT 1 // LoadHistory: 省掉只喂补偿的 pivot 行 hi/lo 拆分
#define DET_WIDE_GROUP 1 // resident: 放大分组宽度，整个矩阵只走一趟分组循环
#define DET_LEAN_CHAIN 1 // resident: 缩短每个 pivot 的关键链（省一次清零 + 标量行列式累加）
#ifndef DET_PANEL_GATHER
#define DET_PANEL_GATHER 1 // cube: 单次 Gather 取列，取代两次 GatherMask（后者要扫约 10 倍元素）
#endif
#endif
#define DET_FAST_PANEL (DET_LEAN_CUBE && DET_LEAN_PANEL && DET_LEAN_QUOT && DET_LEAN_UPDATE && DET_ATOMIC_C)
// resident: 用「整行清零已消耗行」替代常驻「可用行掩码」。
// 每 pivot 省掉 2 条掩码 Mul + 1 次标量掩码写 + 1 次 V->S 排空，换入 1 条 Duplicate(stride)。
// 语义等价：被消耗的行数据恒为 0，Gather 取到 0 -> 主元搜索忽略它、更新乘子为 0。
#define DET_ROW_CLEAR 1
#include "lib/matmul_intf.h"
#include "linalg_det_tiling.h"
#include "tiling_key_linalg_det.h"

// Scalar Dekker/TwoSum steps require separate binary32 operations. Do not
// reassociate or contract them. Vector cross terms use composite instructions.
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
namespace det_detail {
struct Pair {
    float hi;
    float lo;
};
union FloatWord {
    float f;
    uint32_t u;
};
__aicore__ inline float AbsScalar(float x) {
    return x < 0.0f ? -x : x;
}
__aicore__ inline Pair Sum(float a, float b) {
    float s = a + b;
    float v = s - a;
    float e0 = a - (s - v);
    float e1 = b - v;
    return {s, e0 + e1};
}
__aicore__ inline Pair Split(float a) {
    float c = 4097.0f * a;
    float hi = c - (c - a);
    return {hi, a - hi};
}
__aicore__ inline Pair Product(float a, float b) {
    Pair x = Split(a), y = Split(b);
    float p = a * b;
    float e = x.hi * y.hi - p;
    e = e + x.hi * y.lo;
    e = e + x.lo * y.hi;
    e = e + x.lo * y.lo;
    return {p, e};
}
__aicore__ inline Pair AddPair(Pair a, Pair b) {
    Pair s = Sum(a.hi, b.hi);
    float e = s.lo + a.lo;
    e = e + b.lo;
    return Sum(s.hi, e);
}
__aicore__ inline Pair Multiply(Pair a, Pair b) {
    Pair p = Product(a.hi, b.hi);
    float e = p.lo + a.hi * b.lo;
    e = e + a.lo * b.hi;
    return Sum(p.hi, e);
}
__aicore__ inline Pair Divide(Pair a, Pair b) {
    // Scale both operands together before Dekker's split (tiny pivots).
    float mag = AbsScalar(b.hi);
    while (mag != 0.0f && mag < 0.5f) {
        a.hi *= 2.0f;
        a.lo *= 2.0f;
        b.hi *= 2.0f;
        b.lo *= 2.0f;
        mag *= 2.0f;
    }
    while (mag >= 2.0f) {
        a.hi *= 0.5f;
        a.lo *= 0.5f;
        b.hi *= 0.5f;
        b.lo *= 0.5f;
        mag *= 0.5f;
    }
    float q = a.hi / b.hi;
    Pair p = Multiply({q, 0.0f}, b);
    Pair r = AddPair(a, {-p.hi, -p.lo});
    return Sum(q, (r.hi + r.lo) / b.hi);
}
__aicore__ inline void Normalize(Pair &p, int32_t &exponent) {
    float a = AbsScalar(p.hi);
    if (a == 0.0f)
        return;
    FloatWord word;
    word.f = a;
    int32_t shift = static_cast<int32_t>((word.u >> 23U) & 255U) - 127;
    if (shift >= -126 && shift <= 126) {
        FloatWord scale;
        scale.u = static_cast<uint32_t>(127 - shift) << 23U;
        p.hi *= scale.f;
        p.lo *= scale.f;
        exponent += shift;
        return;
    }
    while (a >= 2.0f) {
        p.hi *= 0.5f;
        p.lo *= 0.5f;
        a *= 0.5f;
        ++exponent;
    }
    while (a < 1.0f) {
        p.hi *= 2.0f;
        p.lo *= 2.0f;
        a *= 2.0f;
        --exponent;
    }
}
__aicore__ inline void NormalizePivot(Pair &p, int32_t &exponent) {
    FloatWord word;
    word.f = p.hi;
    uint32_t encoded = (word.u >> 23U) & 255U;
    // For a normal binary32 value, exact power-of-two normalization only
    // changes the exponent bits. Keep the original path for compensated
    // values, subnormals, and the two extreme exponent cases.
    if (p.lo == 0.0f && encoded >= 1U && encoded <= 253U) {
        exponent += static_cast<int32_t>(encoded) - 127;
        word.u = (word.u & 0x807fffffU) | 0x3f800000U;
        p.hi = word.f;
    } else
        Normalize(p, exponent);
}
__aicore__ inline void ConsumeRow(uint64_t &available, uint32_t row, float &sign) {
    // Stable extraction of this row crosses exactly the lower available
    // rows. Its parity equals the resulting row-permutation parity.
    uint64_t below = available & ((1ULL << row) - 1ULL);
    below ^= below >> 32U;
    below ^= below >> 16U;
    below ^= below >> 8U;
    below ^= below >> 4U;
    below ^= below >> 2U;
    below ^= below >> 1U;
    if (below & 1ULL)
        sign = -sign;
    available &= ~(1ULL << row);
}
} // namespace det_detail

using namespace AscendC;
using namespace matmul;
using namespace det_detail;

// Batch-parallel pivoted LU: small matrices retain physical UB row positions
// and track permutation parity in a bit set. Cube matrices physically compact
// pivot rows and update only the active trailing square in per-core workspace.
// Keep the MDL algorithm, HF32 and atomic-output behavior unchanged.
// Only IterateAll is called; compile out unused interface dispatch paths.
constexpr MatmulConfig MakeDetMatmulConfig() {
    auto config = CFG_MDL;
    config.iterateMode = IterateMode::ITERATE_MODE_ALL;
    config.enableSetBias = false;
    config.enableQuantVector = false;
    // MDL defaults to coarse MMAD/FIXPIPE synchronization. UnitFlag allows
    // completed 512-byte L0C regions to drain while MMAD computes the rest.
    config.enUnitFlag = true;
    config.enableEnd = true;
    return config;
}
constexpr MatmulConfig DET_MATMUL_CONFIG = MakeDetMatmulConfig();
template <bool USE_CUBE> struct DetMatmulObject {
    using Operand = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using Type = Matmul<Operand, Operand, Operand, Operand, DET_MATMUL_CONFIG>;
};
template <> struct DetMatmulObject<false> {
    struct Type {};
};
// Matmul registration consumes TCubeTiling separately. Keep only live shape
// fields in the LU kernel instead of copying its large Cube tiling payload.
struct DetKernelShape {
    uint32_t n = 0, batch = 0, stride = 0, tileRows = 0, resident = 0;
    uint32_t cubeEnabled = 0, scratchStride = 0, matPerCore = 0;
    uint64_t coreWsElems = 0;
};
template <bool USE_CUBE> class KernelLinalgDet {
  public:
    using MmOperand = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    typename DetMatmulObject<USE_CUBE>::Type mm;
    __aicore__ inline uint32_t Core() const {
        return t_.cubeEnabled ? static_cast<uint32_t>(GetBlockIdx() / GetTaskRation())
                              : static_cast<uint32_t>(GetBlockIdx());
    }
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR d, GM_ADDR workspace, const LinalgDetTilingData &t, TPipe *pipe) {
        t_ = t;
        scratchStride_ = t_.scratchStride;
        originalTileRows_ = t_.tileRows;
        matrixElems_ = 2U * t_.tileRows * t_.stride;
        runCube_ = false;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d));
        auto ws = reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace));
        ws += static_cast<uint64_t>(Core()) * t_.coreWsElems;
        wh_.SetGlobalBuffer(ws);
        wl_.SetGlobalBuffer(ws + static_cast<uint64_t>(t_.n) * t_.stride);
        uint64_t off = static_cast<uint64_t>(2U) * t_.n * t_.stride;
        lh_.SetGlobalBuffer(ws + off);
        off += 32U * t_.n;
        bh_.SetGlobalBuffer(ws + off);
        off += 32U * t_.stride;
        bl_.SetGlobalBuffer(ws + off);
        off += 32U * t_.stride;
        c0_.SetGlobalBuffer(ws + off);
        off += static_cast<uint64_t>(t_.n) * t_.stride;
        c1_.SetGlobalBuffer(ws + off);
        pipe->InitBuffer(matrix_, matrixElems_ * sizeof(float));
        pipe->InitBuffer(pivot_, 4U * t_.stride * sizeof(float));
        pipe->InitBuffer(scratch_, 6U * scratchStride_ * sizeof(float));
        pipe->InitBuffer(column_, 16U * t_.n * sizeof(float));
        const uint32_t an = (t_.n + 7U) / 8U * 8U;
        pipe->InitBuffer(orderBuf_, 3U * an * sizeof(uint32_t));
        pipe->InitBuffer(outputBuf_, 256U * sizeof(float));
        pipe->InitBuffer(factors_, 16U * t_.n * sizeof(float));
        pipe->InitBuffer(history_, 16U * t_.stride * sizeof(float));
        pipe->InitBuffer(coeff_, 4U * (scratchStride_ / 8U) * sizeof(float));
        pipe->InitBuffer(broadcast_, 4U * scratchStride_ * sizeof(float));
    }
    __aicore__ inline void Process() {
        uint32_t start = Core() * t_.matPerCore;
        if (start >= t_.batch)
            return;
        uint32_t count = t_.batch - start;
        if (count > t_.matPerCore)
            count = t_.matPerCore;
        uint32_t end = start + count;
        auto result = outputBuf_.Get<float>();
        for (uint32_t begin = start; begin < end;) {
            uint32_t chunk = end - begin;
            if (chunk > 256U)
                chunk = 256U;
            DataCopyExtParams p{1, chunk * 4U, 0, 0, 0};
            for (uint32_t m = begin; m < begin + chunk; ++m) {
                float value = One<false>(m);
                DrainRowMove();
                result.SetValue(m - begin, value);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(out_[begin], result, p);
            Stored(true);
            begin += chunk;
        }
    }

  private:
    template <bool CUBE_PATH> __aicore__ inline LocalTensor<float> FactorLowFor() {
#if DET_UB_FACTOR_ALIAS
        static_assert(DET_FAST_PANEL, "Factor alias requires the lean fast-panel configuration");
        if (t_.cubeEnabled) {
            // Cube split follows the final panel read; retry lows are all zero.
            return column_.Get<float>()[CUBE_PATH ? 0U : 8U * t_.n];
        }
#endif
        return factors_.Get<float>()[8U * t_.n];
    }
    __aicore__ inline LocalTensor<float> FactorLow() {
        return runCube_ ? FactorLowFor<true>() : FactorLowFor<false>();
    }
    __aicore__ inline void VS() {
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    __aicore__ inline void SV() {
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }
    template <bool CUBE_PATH> __aicore__ inline void LoadedFor() {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        // Cube-path DMA destinations are consumed by Vector before Scalar.
        // Gather/Reduce + VS provides the later transitive Scalar dependency.
        if (!CUBE_PATH) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
    }
    __aicore__ inline void Loaded() {
        if (runCube_)
            LoadedFor<true>();
        else
            LoadedFor<false>();
    }
    __aicore__ inline void DrainRowMove() {
        if (!rowMovePending_)
            return;
        // GM visibility before another MTE2/Cube read, and source lifetime before UB reuse.
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        rowMovePending_ = false;
    }
    __aicore__ inline void LoadReady() {
        DrainRowMove();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    }
    __aicore__ inline void CubeRowLoadReady() {
        // Only the matrix row planes are overwritten. Scalar reads panel/
        // reduction buffers, not these row planes; Vector owns their lifetime.
        DrainRowMove();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    __aicore__ inline void Stored(bool scalarReuse = false) {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        // Final output is written by Scalar and explicitly requests this wait.
        if (!runCube_ || scalarReuse) {
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }
    __aicore__ inline void StoreReady() {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // Working Cube buffers are produced by Vector, never Scalar.
        if (!runCube_) {
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        }
    }
    __aicore__ inline void VBarrier() {
#if DET_USE_VBARRIER
        PipeBarrier<PIPE_V>();
#endif
    }

    __aicore__ inline void LoadTile(uint32_t row, uint32_t rows) {
        LoadReady();
        DataCopyExtParams p{1, rows * t_.stride * 4U, 0, 0, 0};
        DataCopyPad(matrix_.Get<float>(), wh_[static_cast<uint64_t>(row) * t_.stride], p,
                    DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        DataCopyPad(matrix_.Get<float>()[t_.tileRows * t_.stride], wl_[static_cast<uint64_t>(row) * t_.stride], p,
                    DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        Loaded();
    }
    __aicore__ inline void StoreTile(uint32_t row, uint32_t rows) {
        StoreReady();
        DataCopyExtParams p{1, rows * t_.stride * 4U, 0, 0, 0};
        DataCopyPad(wh_[static_cast<uint64_t>(row) * t_.stride], matrix_.Get<float>(), p);
#if DET_FAST_PANEL
        if (!runCube_)
#endif
            DataCopyPad(wl_[static_cast<uint64_t>(row) * t_.stride], matrix_.Get<float>()[t_.tileRows * t_.stride], p);
        Stored();
    }
    __aicore__ inline void Update(LocalTensor<float> h, LocalTensor<float> l, Pair q, uint32_t k, uint32_t width = 0U) {
        // Aligned suffix; first <=7 already-eliminated entries are harmless.
        uint32_t offset = k / 8U * 8U;
        uint32_t len = (width == 0U ? t_.stride : width) - offset;
        h = h[offset];
        l = l[offset];
        auto ph = pivot_.Get<float>()[offset];
        auto pl = pivot_.Get<float>()[t_.stride + offset];
        auto sh = pivot_.Get<float>()[2U * t_.stride + offset];
        auto sl = pivot_.Get<float>()[3U * t_.stride + offset];
        auto p = scratch_.Get<float>();
        auto e = p[scratchStride_];
        auto z = p[2U * scratchStride_];
        auto s = p[3U * scratchStride_];
        auto v = p[4U * scratchStride_];
        auto r = p[5U * scratchStride_];

        // 与 resident 同源的精简：只做 h -= pivot_hi * q_hi，低半部分清 0
        Muls(p, ph, q.hi, len);
        VBarrier();
        Sub(h, h, p, len);
        VBarrier();
        Duplicate(l, 0.0f, len);
        VBarrier();

        // No per-row V_S/S_V round trip. Entries at/before k are no longer
        // consumed by LU: only later columns are needed. Synchronize once
        // before the next column's pivot search instead.
    }
    __aicore__ inline bool NormalizeResidentRows(LocalTensor<float> matrix, int32_t &exponent) {
        return false;
    }
    __aicore__ inline void ScaleExtremeRows(LocalTensor<float> matrix, LocalTensor<float> q, uint32_t base,
                                            uint32_t rows, uint32_t stride, int32_t &exponent) {
        for (uint32_t r = 0; r < rows; ++r) {
            float mag = q.GetValue(r * 8U);
            auto row = matrix[(base + r) * stride];
            while (mag >= 2.0f) {
                float scale = 1.0f;
                uint32_t steps = 0;
                while (mag >= 2.0f && steps < 16U) {
                    mag *= 0.5f;
                    scale *= 0.5f;
                    ++steps;
                }
                SV();
                Muls(row, row, scale, stride);
                VBarrier();
                VS();
                exponent += steps;
            }
            while (mag < 1.0f) {
                float scale = 1.0f;
                uint32_t steps = 0;
                while (mag < 1.0f && steps < 16U) {
                    mag *= 2.0f;
                    scale *= 2.0f;
                    ++steps;
                }
                SV();
                Muls(row, row, scale, stride);
                VBarrier();
                VS();
                exponent -= steps;
            }
        }
    }

    __aicore__ inline void ApplyPowerScale(LocalTensor<float> matrix, LocalTensor<float> q, uint32_t base,
                                           uint32_t rows, uint32_t stride, int32_t &exponent) {
        for (uint32_t r = 0; r < (rows + 7U) / 8U * 8U; ++r) {
            int32_t shift =
                r < rows ? static_cast<int32_t>((coeff_.Get<uint32_t>().GetValue(r * 8U) >> 23U) & 255U) - 127 : 0;
            coeff_.Get<uint32_t>().SetValue(r, static_cast<uint32_t>(127 - shift) << 23U);
            exponent += shift;
        }
        SV();
        auto bc = broadcast_.Get<float>();
        uint8_t rb = static_cast<uint8_t>(stride / 8U);
        Brcb(bc, q, static_cast<uint8_t>((rows + 7U) / 8U), {rb, static_cast<uint16_t>(rb * 8U)});
        VBarrier();
        for (uint32_t col = 0; col < stride; col += 64U) {
            auto dst = matrix[base * stride + col];
            uint32_t lanes = stride - col;
            if (lanes > 64U)
                lanes = 64U;
            Mul(dst, dst, bc, lanes, static_cast<uint8_t>(rows), {1, 1, 0, rb, rb, rb});
            VBarrier();
        }
    }

    __aicore__ inline void ReduceTileMaxima(LocalTensor<float> matrix, LocalTensor<float> tmp,
                                            LocalTensor<float> maxima, LocalTensor<float> q, uint32_t base,
                                            uint32_t rows, uint32_t stride, uint32_t chunks, uint64_t *mask) {
        Abs(tmp, matrix[base * stride], rows * stride);
        VBarrier();
        if ((stride & 63U) == 0U) {
            WholeReduceMax(maxima, tmp, 64, static_cast<int32_t>(rows * chunks), 4, 1, 8,
                           ReduceOrder::ORDER_VALUE_INDEX);
            VBarrier();
        } else {
            for (uint32_t c = 0; c < chunks; ++c) {
                uint32_t lanes = stride - c * 64U;
                if (lanes > 64U)
                    lanes = 64U;
                WholeReduceMax(maxima[c * 8U], tmp[c * 64U], lanes, static_cast<int32_t>(rows),
                               static_cast<int32_t>(4U * chunks), 1, static_cast<int32_t>(stride / 8U),
                               ReduceOrder::ORDER_VALUE_INDEX);
                VBarrier();
            }
        }
        WholeReduceMax(q, maxima, mask, static_cast<int32_t>(rows), 4, 1, static_cast<int32_t>(chunks),
                       ReduceOrder::ORDER_VALUE_INDEX);
        VBarrier();
    }

    __aicore__ inline bool NormalizeTileRows(LocalTensor<float> matrix, uint32_t total, int32_t &exponent) {
        uint32_t stride = t_.stride, cap = (4U * scratchStride_) / stride;
        const uint32_t reductionRows = t_.n / ((stride + 63U) / 64U);
        if (cap > reductionRows)
            cap = reductionRows;
        cap = cap / 8U * 8U;
        if (cap == 0U)
            cap = 1U;
        auto tmp = scratch_.Get<float>();
        auto maxima = column_.Get<float>();
        auto q = coeff_.Get<float>();
        uint32_t chunks = (stride + 63U) / 64U;
        uint64_t mask[2] = {0ULL, 0ULL};
        for (uint32_t j = 0; j < chunks; ++j)
            mask[0] |= 1ULL << (8U * j);
        for (uint32_t base = 0; base < total; base += cap) {
            uint32_t rows = total - base;
            if (rows > cap)
                rows = cap;
            ReduceTileMaxima(matrix, tmp, maxima, q, base, rows, stride, chunks, mask);
            VS();
            bool extreme = false;
            for (uint32_t r = 0; r < rows; ++r) {
                float mag = q.GetValue(r * 8U);
                if (mag == 0.0f)
                    return false;
                if (mag < 7.52316384526264e-37f || mag >= 1.329227995784916e36f)
                    extreme = true;
            }
            if (extreme) {
                ScaleExtremeRows(matrix, q, base, rows, stride, exponent);
                continue;
            }
            ApplyPowerScale(matrix, q, base, rows, stride, exponent);
        }
        return true;
    }
    template <bool CUBE_PATH> __aicore__ inline float One(uint32_t m) {
        const uint32_t n = t_.n, stride = t_.stride;
        auto mh = matrix_.Get<float>();
        auto ml = mh[t_.tileRows * stride];
        auto order = orderBuf_.Get<uint32_t>();
        auto used = order[(n + 7U) / 8U * 8U];
        int32_t exponent = 0;
        for (uint32_t base = 0; base < n; base += t_.tileRows) {
            uint32_t rows = n - base;
            if (rows > t_.tileRows)
                rows = t_.tileRows;
            DataCopyExtParams p{static_cast<uint16_t>(rows), n * 4U, 0, 0, 0};
            LoadReady();
            DataCopyPad(mh, a_[(static_cast<uint64_t>(m) * n + base) * n], p,
                        DataCopyPadExtParams<float>{true, 0, 0, 0.0f});
            LoadedFor<CUBE_PATH>();
            Duplicate(ml, 0.0f, rows * stride);
            VBarrier();
            if (!NormalizeTileRows(mh, rows, exponent))
                return 0.0f;
            StoreTile(base, rows);
        }
        VS();
        const uint32_t an = (n + 7U) / 8U * 8U;
        for (uint32_t r = 0; r < n; ++r) {
            order.SetValue(r, r);
            used.SetValue(r, 0U);
        }
#if DET_FAST_PANEL
        Duplicate(scratch_.Get<float>()[4U * scratchStride_], 0.0f, scratchStride_);
        VBarrier();
#endif
        auto inverse = order[2U * an];
        for (uint32_t r = 0; r < n; ++r)
            inverse.SetValue(r, r);
#if !DET_OFFS_INIT
        auto offsets = order[3U * an];
        for (uint32_t r = 0; r < n; ++r)
            offsets.SetValue(r, r * stride * 4U);
#endif
        return Blocked<false>(exponent);
    }
    __aicore__ inline void SplitPivot(uint32_t width) {
        auto ph = pivot_.Get<float>();
        auto sh = ph[2U * t_.stride];
        auto sl = ph[3U * t_.stride];
        Muls(sh, ph, 4097.0f, width);
        VBarrier();
        Sub(sl, sh, ph, width);
        VBarrier();
        Sub(sh, sh, sl, width);
        VBarrier();
        Sub(sl, ph, sh, width);
        VBarrier();
    }
    __aicore__ inline void LoadHistory(uint32_t s) {
        auto ph = pivot_.Get<float>();
        auto pl = ph[t_.stride];
        auto hist = history_.Get<float>();
        Muls(ph, hist[2U * s * t_.stride], 1.0f, t_.stride);
        Muls(pl, hist[(2U * s + 1U) * t_.stride], 1.0f, t_.stride);
#if DET_SKIPHSPLIT
        // LEAN_PANEL + LEAN_UPDATE 下 sh/sl 无人读取，这次拆分是纯空转。
        (void)pl;
        VBarrier();
#else
        VBarrier();
        SplitPivot(t_.stride);
#endif
    }
    __aicore__ inline void QuotientGroup(LocalTensor<float> h, LocalTensor<float> l, Pair reciprocal, uint32_t count) {
        auto a = scratch_.Get<float>();
        auto p = a;
        auto e = a[scratchStride_];
        auto z = a[2U * scratchStride_];
        auto sum = a[3U * scratchStride_];
        auto v = a[4U * scratchStride_];
        auto err = a[5U * scratchStride_];
        Pair rs = Split(reciprocal.hi);
        Muls(p, h, reciprocal.hi, count);
        VBarrier();
        Muls(sum, h, 4097.0f, count);
        VBarrier();
        Sub(v, sum, h, count);
        VBarrier();
        Sub(sum, sum, v, count);
        VBarrier();
        Sub(v, h, sum, count);
        VBarrier();
        Muls(e, sum, rs.hi, count);
        VBarrier();
        Sub(e, e, p, count);
        VBarrier();
        Axpy(e, sum, rs.lo, count);
        VBarrier();
        Axpy(e, v, rs.hi, count);
        VBarrier();
        Axpy(e, v, rs.lo, count);
        VBarrier();
        Axpy(e, h, reciprocal.lo, count);
        VBarrier();
        Axpy(e, l, reciprocal.hi, count);
        VBarrier();
        // TwoSum: retain the low component even when h == 0 and l != 0.
        Add(sum, p, e, count);
        VBarrier();
        Sub(v, sum, p, count);
        VBarrier();
        Sub(z, sum, v, count);
        VBarrier();
        Sub(err, p, z, count);
        VBarrier();
        Sub(z, e, v, count);
        VBarrier();
        Add(e, err, z, count);
        VBarrier();
        Muls(p, sum, 1.0f, count);
        VBarrier();
    }
    // Up to 128 physical rows x eight panel columns. Coefficients are expanded by
    // Brcb, pivot rows by repeat-stride zero; panel updates remain compensated.
    template <bool CUBE_PATH>
    __aicore__ inline void ComputePanelQuotient(LocalTensor<float> h, LocalTensor<float> l, uint32_t base,
                                                uint32_t rows, uint32_t s, Pair pv, LocalTensor<uint32_t> used) {
        const uint32_t qp = scratchStride_ / 8U;
        auto q = coeff_.Get<float>();
        const uint32_t paddedRows = (rows + 7U) / 8U * 8U;
        VS();
        for (uint32_t r = 0; r < paddedRows; ++r) {
            Pair v{0.0f, 0.0f};
            if (r < rows && used.GetValue(base + r) == 0U) {
                Pair c{h.GetValue(r * 8U + s), l.GetValue(r * 8U + s)};
                if (c.hi != 0.0f || c.lo != 0.0f) {
                    if (c.lo == 0.0f && pv.lo == 0.0f)
                        v = {c.hi / pv.hi, 0.0f};
                    else
                        v = Divide(c, pv);
                }
            }
            q.SetValue(r, v.hi);
            q.SetValue(qp + r, v.lo);
        }
        SV();
    }

    __aicore__ inline void ApplyPanelUpdate(LocalTensor<float> h, uint32_t base, uint32_t rows, uint32_t s) {
        auto q = coeff_.Get<float>();
        auto factors = factors_.Get<float>();
        auto b = broadcast_.Get<float>();
        auto bh = b;
        const uint8_t repeats = static_cast<uint8_t>((rows + 7U) / 8U);
        const uint32_t count = rows * 8U;
        auto ph = pivot_.Get<float>();
        auto p = scratch_.Get<float>();
        Brcb(bh, q, repeats, {1, 8});
        VBarrier();
        const uint32_t fullRows = rows / 8U * 8U;
        if (fullRows != 0U) {
            uint64_t mask[2] = {0x0101010101010101ULL << s, 0ULL};
            Muls(factors[base * 8U], bh, 1.0f, mask, static_cast<uint8_t>(fullRows / 8U), {1, 1, 8, 8});
            VBarrier();
        }
        if (fullRows < rows) {
            VS();
            for (uint32_t r = fullRows; r < rows; ++r)
                factors.SetValue((base + r) * 8U + s, q.GetValue(r));
            SV();
        }
        if ((rows & 7U) == 0U && rows != 0U) {
            auto phx = ph[2U * t_.stride];
            Muls(phx, ph, 1.0f, 8U, 8U, {1, 0, 1, 0});
            Mul(p, bh, phx, 64U, static_cast<uint8_t>(rows / 8U), BinaryRepeatParams{1, 1, 1, 8, 8, 0});
        } else {
            Mul(p, bh, ph, 8U, static_cast<uint8_t>(rows), BinaryRepeatParams{1, 1, 1, 1, 1, 0});
        }
        VBarrier();
        Sub(h, h, p, count);
        VBarrier();
    }

    __aicore__ inline void PanelGroup(LocalTensor<float> h, LocalTensor<float> l, uint32_t base, uint32_t rows,
                                      uint32_t s, Pair pv, Pair reciprocal, LocalTensor<uint32_t> used) {
        (void)reciprocal;
        ComputePanelQuotient(h, l, base, rows, s, pv, used);
        ApplyPanelUpdate(h, base, rows, s);
    }
    __aicore__ inline uint32_t ResidentPivot(uint32_t k, float &best) {
        return 0U;
    }
    __aicore__ inline void ResidentUpdate(uint32_t k, Pair pv) {
    }
    __aicore__ inline uint32_t SelectPivot(LocalTensor<float> panel, LocalTensor<float> low, uint32_t column,
                                           float &best, uint32_t first = 0U) {
        return 0U;
    }
    __aicore__ inline void LazyUpdatePanel(LocalTensor<float> h, uint32_t row, uint32_t previous, uint32_t kk) {
        if (previous == 0U)
            return;
        uint32_t stride = t_.stride;
        auto q = coeff_.Get<float>();
        auto f = factors_.Get<float>();
        auto hist = history_.Get<float>();
        auto b = broadcast_.Get<float>();
        auto products = scratch_.Get<float>();
        // High U rows are compact: each row contains only [kk,stride).
        // The low plane is separate and is filled only before Cube packing.
        const uint32_t width = stride - kk;
        // A few pivot products over a long suffix need fewer issues as Muls.
        // Keep the original per-pivot subtract order; no FMA or dot reduction.
        if (width >= 256U && previous <= 3U) {
            for (uint32_t j = 0; j < previous; ++j) {
                float coefficient = j == 0U ? lazy0_ : (j == 1U ? lazy1_ : lazy2_);
                Muls(products, hist[j * width], coefficient, width);
                VBarrier();
                Sub(h[kk], h[kk], products, width);
                VBarrier();
            }
            return;
        }
        Muls(q, f[row * 8U], 1.0f, 8U);
        VBarrier();
        uint8_t blocks = static_cast<uint8_t>(width / 8U);
        Brcb(b, q, 1U, {blocks, static_cast<uint16_t>(blocks * 8U)});
        VBarrier();
        for (uint32_t c = 0; c < width; c += 64U) {
            uint32_t lanes = width - c;
            if (lanes > 64U)
                lanes = 64U;
            Mul(products[c], hist[c], b, lanes, static_cast<uint8_t>(previous), {1, 1, 0, blocks, blocks, blocks});
            VBarrier();
        }
        for (uint32_t j = 0; j < previous; ++j) {
            // Keep the original sequential binary32 subtraction order.
            Sub(h[kk], h[kk], products[j * width], width);
            VBarrier();
        }
    }
    // Keep finished pivot rows at the front of W. This permits an actual
    // shrinking Cube M dimension instead of multiplying zero coefficients.
    __aicore__ inline void CompactPivot(uint32_t k, uint32_t r, LocalTensor<float> panel, LocalTensor<float> factor) {
        if (k == r)
            return;
        uint32_t stride = t_.stride;
        auto h = matrix_.Get<float>();
        auto other = h[matrixElems_ - stride];
        auto tmp = pivot_.Get<float>();
        // SelectPivot only reads UB from Scalar; panel/factor remain Vector-owned.
        Muls(tmp, panel[k * 8U], 1.0f, 8U);
        Muls(panel[k * 8U], panel[r * 8U], 1.0f, 8U);
        Muls(panel[r * 8U], tmp, 1.0f, 8U);
        Muls(tmp, factor[k * 8U], 1.0f, 8U);
        Muls(factor[k * 8U], factor[r * 8U], 1.0f, 8U);
        Muls(factor[r * 8U], tmp, 1.0f, 8U);
        VBarrier();
        CubeRowLoadReady();
        uint32_t col = k / 8U * 8U;
        DataCopyExtParams rp{1, (stride - col) * 4U, 0, 0, 0};
        DataCopyPad(h[col], wh_[static_cast<uint64_t>(r) * stride + col], rp,
                    DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        DataCopyPad(other[col], wh_[static_cast<uint64_t>(k) * stride + col], rp,
                    DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        // Row exchange has no Vector arithmetic on these two DMA buffers.
        // Route the dependency directly from MTE2 to MTE3 instead of passing
        // through both the Vector and Scalar pipelines.
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        // h retains the selected pivot for LazyUpdatePanel. Finished W[k] is dead.
        DataCopyPad(wh_[static_cast<uint64_t>(r) * stride + col], other[col], rp);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        rowMovePending_ = true;
        // h and other are separate tile planes. Vector may consume selected h while other is written out.
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
#if DET_NO_PIVOT_VEC
    // v41: diagonal-pivot vector-only LU. The main loop has no V_S/S_V stalls:
    // the pivot is replicated with a zero-offset Gather, coefficients use vector
    // Div, and the determinant is accumulated in a vector lane. Periodic scalar
    // normalization happens only every 16 pivots. A small/zero pivot falls back
    // to the pivoted CompactResident after reloading the original tile.
    __aicore__ inline bool CompactResidentNoPivot(uint32_t first, int32_t &exponent, float &detScalar, float &sign) {
        return false;
    }
#endif
    __aicore__ inline bool CompactResident(uint32_t first, int32_t &exponent, float &detScalar, float &sign,
                                           bool retryGuard, bool noPivot = false) {
        return false;
    }
    __aicore__ inline bool ResidentTail(uint32_t first, int32_t &exponent, float &detScalar, float &sign) {
        return false;
    }
    __aicore__ inline void RetryLoadTile(uint32_t base, uint32_t rows) {
        LoadReady();
        DataCopyExtParams p{1, rows * t_.stride * 4U, 0, 0, 0};
        auto matrix = matrix_.Get<float>();
        DataCopyPad(matrix, wh_[static_cast<uint64_t>(base) * t_.stride], p,
                    DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
        // Lean row updates keep low entries identically zero. Generate them
        // locally rather than round-trip the zero matrix through GM.
        Duplicate(matrix[t_.tileRows * t_.stride], 0.0f, rows * t_.stride);
        VBarrier();
        Loaded();
    }
    __aicore__ inline void RetryStoreTile(uint32_t base, uint32_t rows) {
        StoreReady();
        DataCopyExtParams p{1, rows * t_.stride * 4U, 0, 0, 0};
        DataCopyPad(wh_[static_cast<uint64_t>(base) * t_.stride], matrix_.Get<float>(), p);
        Stored();
    }
    __aicore__ inline void RetryUpdateSparse(uint32_t base, uint32_t rows, uint32_t col, uint32_t s) {
        const uint32_t stride = t_.stride, width = stride - col;
        auto h = matrix_.Get<float>();
        auto f = factors_.Get<float>();
        auto used = orderBuf_.Get<uint32_t>()[(t_.n + 7U) / 8U * 8U];
        auto pivot = pivot_.Get<float>()[col];
        auto product = scratch_.Get<float>();
        for (uint32_t r = 0; r < rows; ++r) {
            if (used.GetValue(base + r) != 0U)
                continue;
            float q = f.GetValue((base + r) * 8U + s);
            if (q == 0.0f)
                continue;
            Muls(product, pivot, q, width);
            VBarrier();
            auto dst = h[r * stride + col];
            Sub(dst, dst, product, width);
            VBarrier();
        }
    }

    __aicore__ inline void ApplyDenseBlock(LocalTensor<float> matrix, LocalTensor<float> product, LocalTensor<float> bc,
                                           LocalTensor<float> pivot, uint32_t begin, uint32_t col, uint32_t width,
                                           uint32_t count, uint32_t stride, uint8_t rowPitch, uint8_t productPitch) {
        for (uint32_t c = 0; c < width; c += 64U) {
            uint32_t lanes = width - c;
            if (lanes > 64U)
                lanes = 64U;
            Mul(product[c], bc, pivot[col + c], lanes, static_cast<uint8_t>(count),
                {1, 0, 1, productPitch, rowPitch, 0});
            VBarrier();
            auto dst = matrix[begin * stride + col + c];
            Sub(dst, dst, product[c], lanes, static_cast<uint8_t>(count), {1, 1, 1, rowPitch, rowPitch, productPitch});
            VBarrier();
        }
    }

    __aicore__ inline void RetryUpdateDense(uint32_t base, uint32_t rows, uint32_t col, uint32_t s) {
        const uint32_t stride = t_.stride, width = stride - col;
        uint32_t capacity = (6U * scratchStride_) / width;
        const uint32_t broadcastRows = (4U * scratchStride_) / stride;
        if (capacity > broadcastRows)
            capacity = broadcastRows;
        capacity = capacity / 8U * 8U;
        if (capacity == 0U)
            capacity = 1U;
        auto matrix = matrix_.Get<float>();
        auto factors = factors_.Get<float>();
        auto q = coeff_.Get<float>();
        auto bc = broadcast_.Get<float>();
        auto product = scratch_.Get<float>();
        auto stage = product[3U * scratchStride_];
        auto pivot = pivot_.Get<float>();
        const uint32_t an = (t_.n + 7U) / 8U * 8U;
        auto used = orderBuf_.Get<float>()[an];
        const uint8_t rowPitch = static_cast<uint8_t>(stride / 8U);
        const uint8_t productPitch = static_cast<uint8_t>(width / 8U);
        for (uint32_t begin = 0; begin < rows; begin += capacity) {
            uint32_t count = rows - begin;
            if (count > capacity)
                count = capacity;
            uint64_t retained = 0;
            GatherMask(stage, factors[(base + begin) * 8U], static_cast<uint8_t>(3U + s % 4U), true, count * 8U,
                       {1, 1, 8, 0}, retained);
            VBarrier();
            GatherMask(q, stage, static_cast<uint8_t>(s < 4U ? 1U : 2U), true, count * 2U, {1, 1, 8, 0}, retained);
            VBarrier();
            auto offsets = scratch_.Get<uint32_t>()[5U * scratchStride_];
            Gather(stage, used, offsets, (base + begin) * 4U, count);
            VBarrier();
            auto ones = q[scratchStride_ / 8U];
            Duplicate(ones, 1.0f, count);
            VBarrier();
            Sub(stage, ones, stage, count);
            VBarrier();
            Mul(q, q, stage, count);
            VBarrier();
            Brcb(bc, q, static_cast<uint8_t>((count + 7U) / 8U), {rowPitch, static_cast<uint16_t>(rowPitch * 8U)});
            VBarrier();
            ApplyDenseBlock(matrix, product, bc, pivot, begin, col, width, count, stride, rowPitch, productPitch);
        }
    }

    __aicore__ inline void RetryUpdateTile(uint32_t base, uint32_t rows, uint32_t col, uint32_t s,
                                           uint32_t activeRows) {
        const uint32_t width = t_.stride - col;
        bool sparse = width > 256U || activeRows * 2U < rows;
        if (!sparse) {
            auto flags = orderBuf_.Get<uint32_t>()[(t_.n + 7U) / 8U * 8U];
            auto coefficients = factors_.Get<float>();
            uint32_t nonzero = 0;
            for (uint32_t r = 0; r < rows; ++r)
                if (flags.GetValue(base + r) == 0U && coefficients.GetValue((base + r) * 8U + s) != 0.0f)
                    ++nonzero;
            if (nonzero == 0U)
                return;
            sparse = nonzero * 2U < activeRows;
        }
        if (sparse) {
            RetryUpdateSparse(base, rows, col, s);
            return;
        }
        RetryUpdateDense(base, rows, col, s);
    }
    __aicore__ inline uint32_t RetryPivot(LocalTensor<float> panel, LocalTensor<uint32_t> offsets, uint32_t first,
                                          uint32_t col, float &best) {
        const uint32_t alignedFirst = first / 8U * 8U;
        const uint32_t remaining = t_.n - alignedFirst;
        auto values = scratch_.Get<float>();
        auto result = coeff_.Get<float>();
        // Scalar updates the two permutation offsets after every row exchange.
        SV();
        Gather(values, panel, offsets[alignedFirst], col * 4U, remaining);
        VBarrier();
        Abs(values, values, remaining);
        VBarrier();
        if (first != alignedFirst) {
            Duplicate(values, 0.0f, first - alignedFirst);
            VBarrier();
        }
        ReduceMax(result, values, values[3U * scratchStride_], remaining, true);
        VS();
        best = result.GetValue(0);
        return alignedFirst + coeff_.Get<uint32_t>().GetValue(1);
    }

#if DET_CUBE_B16
    __aicore__ inline void PanelGroup16(LocalTensor<float> h, LocalTensor<float> factorsBase, uint32_t base,
                                        uint32_t rows, uint32_t s, float reciprocal, LocalTensor<float> u,
                                        LocalTensor<float> h2, LocalTensor<float> u2, bool doCross) {
        auto q = coeff_.Get<float>();
        {
            auto offsP = column_.Get<uint32_t>()[16U * t_.n];
            Gather(q, h, offsP, s * 4U, rows);
        }
        Muls(q, q, reciprocal, rows);
        auto b = broadcast_.Get<float>();
        Brcb(b, q, static_cast<uint8_t>((rows + 7U) / 8U), {static_cast<uint8_t>(1U), static_cast<uint16_t>(8U)});
        uint32_t full = rows / 8U * 8U;
        if (full) {
            uint64_t mask[2] = {0x0101010101010101ULL << s, 0ULL};
            Muls(factorsBase[base * 8U], b, 1.0f, mask, static_cast<uint8_t>(full / 8U), {1, 1, 8, 8});
        }
        if (full < rows) {
            uint64_t mask[2] = {(0x0101010101010101ULL << s) & ((1ULL << ((rows - full) * 8U)) - 1ULL), 0ULL};
            Muls(factorsBase[(base + full) * 8U], b[full * 8U], 1.0f, mask, 1U, {1, 1, 8, 8});
        }
        auto product = scratch_.Get<float>();
        if (full)
            Mul(product, b, u, 64U, static_cast<uint8_t>(full / 8U), {1, 1, 1, 8, 8, 0});
        if (full < rows)
            Mul(product[full * 8U], b[full * 8U], u, 8U, static_cast<uint8_t>(rows - full), {1, 1, 1, 1, 1, 0});
        VBarrier();
        Sub(h, h, product, rows * 8U);
        VBarrier();
        if (doCross) {
            if (full)
                Mul(product, b, u2, 64U, static_cast<uint8_t>(full / 8U), {1, 1, 1, 8, 8, 0});
            if (full < rows)
                Mul(product[full * 8U], b[full * 8U], u2, 8U, static_cast<uint8_t>(rows - full), {1, 1, 1, 1, 1, 0});
            VBarrier();
            Sub(h2, h2, product, rows * 8U);
            VBarrier();
        }
    }
    __aicore__ inline void LazyUpdatePanel16(LocalTensor<float> h, uint32_t row, uint32_t previous, uint32_t kk,
                                             LocalTensor<float> histBase, LocalTensor<float> factorBase) {
        if (previous == 0U)
            return;
        uint32_t width = t_.stride - kk;
        auto q = coeff_.Get<float>();
        auto b = broadcast_.Get<float>();
        auto products = scratch_.Get<float>();
        Muls(q, factorBase[row * 8U], 1.0f, 8U);
        uint8_t blocks = static_cast<uint8_t>(width / 8U);
        Brcb(b, q, 1U, {blocks, static_cast<uint16_t>(blocks * 8U)});
        VBarrier();
        for (uint32_t c = 0; c < width; c += 64U) {
            uint32_t lanes = width - c;
            if (lanes > 64U)
                lanes = 64U;
            Mul(products[c], histBase[c], b, lanes, static_cast<uint8_t>(previous), {1, 1, 0, blocks, blocks, blocks});
        }
        VBarrier();
        for (uint32_t j = 0; j < previous; ++j) {
            Sub(h[kk], h[kk], products[j * width], width);
            VBarrier();
        }
    }
    __aicore__ inline void SplitFactor16(LocalTensor<float> hi, LocalTensor<float> lo, uint32_t count) {
        auto tmp = scratch_.Get<float>();
        auto split = tmp[scratchStride_];
        for (uint32_t off = 0; off < count;) {
            uint32_t c = count - off;
            if (c > scratchStride_)
                c = scratchStride_;
            Muls(tmp, hi[off], 65537.0f, c);
            Sub(split, tmp, hi[off], c);
            Sub(split, tmp, split, c);
            Sub(lo[off], split, hi[off], c);
            Muls(hi[off], split, -1.0f, c);
            VBarrier();
            off += c;
        }
    }
    __aicore__ inline void SplitHist16(LocalTensor<float> histHigh, LocalTensor<float> histLow, uint32_t count) {
        auto tmp = scratch_.Get<float>();
        auto upper = tmp[scratchStride_];
        for (uint32_t off = 0; off < count;) {
            uint32_t c = count - off;
            if (c > scratchStride_)
                c = scratchStride_;
            Muls(tmp, histHigh[off], 65537.0f, c);
            Sub(upper, tmp, histHigh[off], c);
            Sub(upper, tmp, upper, c);
            Sub(histLow[off], histHigh[off], upper, c);
            Muls(histHigh[off], upper, 1.0f, c);
            VBarrier();
            off += c;
        }
    }
    __aicore__ inline void CubeUpdate16(uint32_t block0, uint32_t bd0, uint32_t bd1) {
        const uint32_t n = t_.n, stride = t_.stride;
        const uint32_t col = block0 + bd0;
        const uint32_t width = stride - col;
        auto f0 = factors_.Get<float>();
        auto f1 = f0[8U * n];
        auto panel0 = column_.Get<float>();
        auto panel1 = panel0[8U * n];
        auto hist = history_.Get<float>();
        // f0/f1 become -Fh, panel0/panel1 receive -Fl for rows block0..n-1.
        SplitFactor16(f0[block0 * 8U], panel0[block0 * 8U], (n - block0) * 8U);
        SplitFactor16(f1[block0 * 8U], panel1[block0 * 8U], (n - block0) * 8U);
        // U rows: 16 high rows at s*width, 16 low rows at (16+s)*width.
        SplitHist16(hist, hist[16U * width], 16U * width);
        StoreReady();
        {
            DataCopyExtParams ap{static_cast<uint16_t>(n), 32U, 0, 32U * 7U, 0};
            DataCopyPad(lh_[0], f0, ap);
            DataCopyPad(lh_[8U], f0, ap);
            DataCopyPad(lh_[16U], panel0, ap);
            DataCopyPad(lh_[24U], panel0, ap);
            DataCopyPad(lh_[32U], f1, ap);
            DataCopyPad(lh_[40U], f1, ap);
            DataCopyPad(lh_[48U], panel1, ap);
            DataCopyPad(lh_[56U], panel1, ap);
        }
        {
            DataCopyExtParams up{static_cast<uint16_t>(8U), width * 4U, 0, col * 4U, 0};
            DataCopyPad(bh_[0U * stride + col], hist, up);
            DataCopyPad(bh_[8U * stride + col], hist[16U * width], up);
            DataCopyPad(bh_[16U * stride + col], hist, up);
            DataCopyPad(bh_[24U * stride + col], hist[16U * width], up);
            DataCopyPad(bh_[32U * stride + col], hist[8U * width], up);
            DataCopyPad(bh_[40U * stride + col], hist[24U * width], up);
            DataCopyPad(bh_[48U * stride + col], hist[8U * width], up);
            DataCopyPad(bh_[56U * stride + col], hist[24U * width], up);
        }
        Stored();
        MatProduct(lh_, bh_, wh_, col, 1U);
    }
    __aicore__ inline float Blocked16(int32_t exponent) {
        return 0.0f;
    }
#endif
    template <bool CUBE_PATH> __aicore__ inline float Blocked(int32_t exponent) {
        const uint32_t n = t_.n, stride = t_.stride;
        auto panel = column_.Get<float>();
        auto low = panel[n * 8U];
        auto factor = factors_.Get<float>();
        auto fl = FactorLowFor<CUBE_PATH>();
        auto hist = history_.Get<float>();
        auto order = orderBuf_.Get<uint32_t>();
        auto used = order[(n + 7U) / 8U * 8U];
        auto mh = matrix_.Get<float>();
        auto ml = mh[t_.tileRows * stride];
        Pair determinant{1.0f, 0.0f};
        float sign = 1.0f;
#if DET_LEAN_CHAIN
        float detScalar = 1.0f;
#endif
#if DET_LEAN_CUBE
#if !DET_FAST_PANEL
        if (CUBE_PATH)
            ZeroGM(wl_, static_cast<uint32_t>(n) * stride);
#endif
#endif
        for (uint32_t kk = 0; kk < n; kk += 8U) {
#if DET_FAST_PANEL && DET_LEAN_CHAIN
            if (CUBE_PATH && n >= 128U && n - kk <= 64U) {
                if (!ResidentTail(kk, exponent, detScalar, sign)) {
                    cubeAbort_ = true;
                    return 0.0f;
                }
                break;
            }
#endif
            uint32_t bd = n - kk;
            if (bd > 8U)
                bd = 8U;
            LoadReady();
            uint32_t firstRow = CUBE_PATH ? kk : 0U;
            DataCopyExtParams pp{static_cast<uint16_t>(n - firstRow), bd * 4U, (stride - bd) * 4U, 0, 0};
            DataCopyPad(panel[firstRow * 8U], wh_[static_cast<uint64_t>(firstRow) * stride + kk], pp,
                        DataCopyPadExtParams<float>{true, 0, 0, 0.0f});
#if DET_LEAN_CUBE
#if DET_FAST_PANEL
            if (!CUBE_PATH)
                Duplicate(low, 0.0f, 8U * n);
#else
            Duplicate(low, 0.0f, 8U * n);
#endif
#else
            DataCopyPad(low, wl_[kk], pp, DataCopyPadExtParams<float>{true, 0, 0, 0.0f});
#endif
            LoadedFor<CUBE_PATH>();
#if DET_CUBE_NO_PIVOT
            // Diagonal-pivot path reads the panel from Scalar at s=0.
            if (CUBE_PATH && !forcePivotCube_) {
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            }
#endif
#if DET_FAST_PANEL
            if (CUBE_PATH)
                Duplicate(factor[kk * 8U], 0.0f, 8U * (n - kk));
            else {
                Duplicate(factor, 0.0f, 8U * n);
                Duplicate(fl, 0.0f, 8U * n);
            }
            VBarrier();
#else
            Duplicate(factor, 0.0f, 8U * n);
            Duplicate(fl, 0.0f, 8U * n);
            VBarrier();
#endif
            for (uint32_t s = 0; s < bd; ++s) {
                uint32_t k = kk + s;
                float best = -1.0f;
                uint32_t chosen = k;
                uint32_t an = (n + 7U) / 8U * 8U;
                auto inverse = order[2U * an];
                if (CUBE_PATH) {
#if DET_CUBE_NO_PIVOT
                    if (!forcePivotCube_) {
                        // Diagonal pivot: no column-max reduction, no row move.
                        VS();
                        float diag = panel.GetValue(k * 8U + s);
                        best = AbsScalar(diag);
                        if (best < 1.0e-3f) {
                            cubeNoPivotAbort_ = true;
                            return 0.0f;
                        }
                        chosen = k;
                    } else
#endif
                    {
                        chosen = SelectPivot(panel, low, s, best, k);
                    }
                } else {
#if DET_LEAN_UPDATE && DET_LEAN_PANEL && DET_LEAN_CUBE
                    chosen = RetryPivot(panel, inverse, k, s, best);
#else
                    VS();
                    for (uint32_t j = k; j < n; ++j) {
                        uint32_t row = order.GetValue(j);
                        float magnitude = AbsScalar(panel.GetValue(row * 8U + s) + low.GetValue(row * 8U + s));
                        if (magnitude > best) {
                            best = magnitude;
                            chosen = j;
                        }
                    }
#endif
                }
                if (best == 0.0f) {
                    if (CUBE_PATH)
                        cubeAbort_ = true;
                    return 0.0f;
                }
                Pair pv{panel.GetValue(chosen * 8U + s), 0.0f};
                bool pivotReady = false;
                uint32_t r = CUBE_PATH ? chosen : order.GetValue(chosen);
                if (CUBE_PATH && s > 0U && s <= 3U && stride - kk >= 256U) {
                    // SelectPivot's V_S has already completed coefficient writes.
                    // Capture before CompactPivot swaps the selected factor row.
                    lazy0_ = factor.GetValue(r * 8U);
                    lazy1_ = s >= 2U ? factor.GetValue(r * 8U + 1U) : 0.0f;
                    lazy2_ = s >= 3U ? factor.GetValue(r * 8U + 2U) : 0.0f;
                }
#if DET_FAST_PANEL
                if (CUBE_PATH) {
                    if (r != k) {
                        CompactPivot(k, r, panel, factor);
                        pivotReady = true;
                        sign = -sign;
                    }
                    r = k;
                    chosen = k;
                }
#endif
                if (chosen != k) {
                    uint32_t previous = order.GetValue(k);
                    order.SetValue(chosen, previous);
                    order.SetValue(k, r);
                    sign = -sign;
#if DET_LEAN_UPDATE && DET_LEAN_PANEL && DET_LEAN_CUBE
                    inverse.SetValue(chosen, previous * 32U);
                    inverse.SetValue(k, r * 32U);
#else
                    inverse.SetValue(previous, chosen);
                    inverse.SetValue(r, k);
#endif
                }
                if (!CUBE_PATH) {
                    used.SetValue(r, 0x3f800000U);
                    pv = {panel.GetValue(r * 8U + s), low.GetValue(r * 8U + s)};
                }
#if DET_CUBE_NO_PIVOT
                if (CUBE_PATH && forcePivotCube_ && (best < 0.05f || best > 4096.0f)) {
                    cubeAbort_ = true;
                    return 0.0f;
                }
#else
                if (CUBE_PATH && (best < 0.05f || best > 4096.0f)) {
                    cubeAbort_ = true;
                    return 0.0f;
                }
#endif
                Pair norm = pv;
                NormalizePivot(norm, exponent);
#if DET_LEAN_CHAIN
                detScalar *= norm.hi;
                // Each normalized pivot has magnitude in [1,2). Keep the
                // mantissa product bounded without changing its binary32
                // significand; the determinant exponent carries the scale.
                if ((k & 15U) == 15U) {
                    Pair bounded{detScalar, 0.0f};
                    NormalizePivot(bounded, exponent);
                    detScalar = bounded.hi;
                }
#else
                determinant = Multiply(determinant, norm);
                Normalize(determinant, exponent);
#endif
                // Pivot trailing row is updated lazily by previous panel pivots.
                if (!pivotReady) {
                    if (CUBE_PATH)
                        CubeRowLoadReady();
                    else
                        LoadReady();
                    uint32_t startCol = CUBE_PATH ? kk : 0U;
                    DataCopyExtParams rp{1, (stride - startCol) * 4U, 0, 0, 0};
                    DataCopyPad(mh[startCol], wh_[static_cast<uint64_t>(r) * stride + startCol], rp,
                                DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
#if DET_LEAN_CUBE
#if DET_FAST_PANEL
                    if (!CUBE_PATH) {
                        Duplicate(ml, 0.0f, stride);
                        VBarrier();
                    }
                    LoadedFor<CUBE_PATH>();
#else
                    Duplicate(ml, 0.0f, stride);
                    VBarrier();
                    LoadedFor<CUBE_PATH>();
#endif
#else
                    DataCopyPad(ml, wl_[static_cast<uint64_t>(r) * stride], rp,
                                DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
                    LoadedFor<CUBE_PATH>();
#endif
                }
#if DET_FAST_PANEL
                if (CUBE_PATH)
                    LazyUpdatePanel(mh, r, s, kk);
                else {
                    for (uint32_t j = 0; j < s; ++j) {
                        Pair q{factor.GetValue(r * 8U + j), fl.GetValue(r * 8U + j)};
                        if (q.hi == 0.0f && q.lo == 0.0f)
                            continue;
                        LoadHistory(j);
                        Update(mh, ml, q, kk + j);
                    }
                }
#else
                for (uint32_t j = 0; j < s; ++j) {
                    Pair q{factor.GetValue(r * 8U + j), fl.GetValue(r * 8U + j)};
                    if (q.hi == 0.0f && q.lo == 0.0f)
                        continue;
                    LoadHistory(j);
                    Update(mh, ml, q, kk + j);
                }
#endif
                if (CUBE_PATH)
                    Muls(hist[s * (stride - kk)], mh[kk], 1.0f, stride - kk);
                else
                    Muls(hist[2U * s * stride], mh, 1.0f, stride);
#if DET_FAST_PANEL
                if (!CUBE_PATH)
#endif
                    Muls(hist[(2U * s + 1U) * stride], ml, 1.0f, stride);
                VBarrier();
                if (k + 1U == n)
                    continue;
                auto ph = pivot_.Get<float>();
#if DET_FAST_PANEL
                if (CUBE_PATH) {
                    // Build all eight copies directly from the panel; no intermediate pivot copy.
                    Muls(ph[2U * stride], panel[r * 8U], 1.0f, 8U, 8U, {1, 0, 1, 0});
                    VBarrier();
                } else {
                    Muls(ph, panel[r * 8U], 1.0f, 8U);
                    Muls(ph[stride], low[r * 8U], 1.0f, 8U);
                    VBarrier();
                    SplitPivot(8U);
                }
#else
                Muls(ph, panel[r * 8U], 1.0f, 8U);
                Muls(ph[stride], low[r * 8U], 1.0f, 8U);
                VBarrier();
                SplitPivot(8U);
#endif
#if DET_FAST_PANEL
                Pair reciprocal = CUBE_PATH ? Pair{1.0f / pv.hi, 0.0f} : Pair{0.0f, 0.0f};
#else
                Pair reciprocal = CUBE_PATH ? Divide({1.0f, 0.0f}, pv) : Pair{0.0f, 0.0f};
#endif
                // Fast panel needs one product plane, a coefficient vector and one broadcast plane.
                // Four scratch planes and the whole broadcast buffer can handle up to 512 rows.
                uint32_t groupMax = scratchStride_ / 8U;
#if DET_FAST_PANEL
                if (CUBE_PATH)
                    groupMax = 4U * scratchStride_ / 8U;
                else
#endif
                    if (groupMax > 128U)
                    groupMax = 128U;
                for (uint32_t base = CUBE_PATH ? k + 1U : 0U; base < n; base += groupMax) {
                    uint32_t rows = n - base;
                    if (rows > groupMax)
                        rows = groupMax;
                    PanelGroup<CUBE_PATH>(panel[base * 8U], low[base * 8U], base, rows, s, pv, reciprocal, used);
                }
            }
            if constexpr (USE_CUBE) {
                if (CUBE_PATH) {
                    // The determinant already owns every final-panel pivot.
                    // No trailing matrix exists after the final panel.
                    if (kk + bd < n)
                        CubeUpdate(kk, bd, used);
                    continue;
                }
            }
            // Remaining physical rows: one GM round trip per eight LU columns.
            for (uint32_t base = 0; base < n; base += t_.tileRows) {
                uint32_t rows = n - base;
                if (rows > t_.tileRows)
                    rows = t_.tileRows;
                uint32_t activeRows = 0;
                for (uint32_t r = base; r < base + rows; ++r)
                    if (used.GetValue(r) == 0U)
                        ++activeRows;
                if (activeRows == 0U)
                    continue;
#if DET_LEAN_UPDATE && DET_LEAN_PANEL && DET_LEAN_CUBE
                RetryLoadTile(base, rows);
                for (uint32_t s = 0; s < bd; ++s) {
                    LoadHistory(s);
                    RetryUpdateTile(base, rows, kk, s, activeRows);
                }
                RetryStoreTile(base, rows);
#else
                LoadTile(base, rows);
                for (uint32_t s = 0; s < bd; ++s) {
                    LoadHistory(s);
                    for (uint32_t r = 0; r < rows; ++r) {
                        if (used.GetValue(base + r) != 0U)
                            continue;
                        Pair q{factor.GetValue((base + r) * 8U + s), fl.GetValue((base + r) * 8U + s)};
                        if (q.hi == 0.0f && q.lo == 0.0f)
                            continue;
                        Update(mh[r * stride], ml[r * stride], q, kk + s);
                    }
                }
                StoreTile(base, rows);
#endif
            }
        }
#if DET_LEAN_CHAIN
        float result = detScalar;
#else
        float result = determinant.hi + determinant.lo;
#endif
        while (exponent > 0) {
            result *= 2.0f;
            --exponent;
        }
        while (exponent < 0) {
            result *= 0.5f;
            ++exponent;
        }
        return sign * result;
    }
    __aicore__ inline void ZeroGM(GlobalTensor<float> dst, uint32_t count) {
        auto stage = matrix_.Get<float>();
        uint32_t capacity = t_.tileRows * t_.stride;
        Duplicate(stage, 0.0f, capacity);
        VBarrier();
        for (uint32_t off = 0; off < count;) {
            uint32_t size = count - off;
            if (size > capacity)
                size = capacity;
            StoreReady();
            DataCopyExtParams p{1, size * 4U, 0, 0, 0};
            DataCopyPad(dst[off], stage, p);
            Stored();
            off += size;
        }
    }
    __aicore__ inline void MatProduct(GlobalTensor<float> a, GlobalTensor<float> b, GlobalTensor<float> c, uint32_t col,
                                      uint8_t atomic = 0U) {
        if constexpr (USE_CUBE) {
            // SetOrgShape retains physical ND strides; offsets select the
            // active square, while SetTail controls the effective Cube workload.
            uint32_t first = (atomic != 0U && runCube_) ? col : 0U;
            mm.SetTail(t_.n - first, t_.n - col, DET_CUBE_K);
            mm.SetTensorA(a[first * DET_CUBE_K]);
            mm.SetTensorB(b[col], false);
            // atomic=1 时以 C = A*B 原子累加到 c[col] 起的位置（N 方向 stride 由 OrgShape 给出）。
            if (atomic != 0U) {
                mm.IterateAll(c[static_cast<uint64_t>(first) * t_.stride + col], 1U);
            } else {
                mm.IterateAll(c, 0U);
            }
            // Synchronous real IterateAll already completes server-side cleanup.
        }
    }
    __aicore__ inline void PackCubeA(uint32_t first) {
        uint32_t n = t_.n, cap = matrixElems_ / 32U;
        if (cap > 512U)
            cap = 512U;
        cap = cap / 8U * 8U;
        if (cap == 0U)
            cap = 1U;
        auto packed = matrix_.Get<float>();
        auto f = factors_.Get<float>();
        for (uint32_t base = first; base < n; base += cap) {
            uint32_t rows = n - base;
            if (rows > cap)
                rows = cap;
            uint32_t full = rows / 8U * 8U;
            auto hi = f[base * 8U];
            auto lo = FactorLow()[base * 8U];
            if (full) {
                uint8_t rr = static_cast<uint8_t>(full / 8U);
                Muls(packed, hi, 1.0f, 64U, rr, {4, 1, 32, 8});
                Muls(packed[8U], hi, 1.0f, 64U, rr, {4, 1, 32, 8});
                Muls(packed[16U], lo, 1.0f, 64U, rr, {4, 1, 32, 8});
                Muls(packed[24U], lo, 1.0f, 64U, rr, {4, 1, 32, 8});
            }
            if (full < rows) {
                uint8_t rr = static_cast<uint8_t>(rows - full);
                auto dst = packed[full * 32U];
                hi = hi[full * 8U];
                lo = lo[full * 8U];
                Muls(dst, hi, 1.0f, 8U, rr, {1, 1, 4, 1});
                Muls(dst[8U], hi, 1.0f, 8U, rr, {1, 1, 4, 1});
                Muls(dst[16U], lo, 1.0f, 8U, rr, {1, 1, 4, 1});
                Muls(dst[24U], lo, 1.0f, 8U, rr, {1, 1, 4, 1});
            }
            VBarrier();
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams cp{1, rows * 128U, 0, 0, 0};
            DataCopyPad(lh_[base * 32U], packed, cp);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }
    __aicore__ inline void CubeUpdateCompact(uint32_t col, uint32_t bd) {
    }
    __aicore__ inline void CubeUpdate(uint32_t kk, uint32_t bd, LocalTensor<uint32_t> used) {
    }
    __aicore__ inline void ApplyCubeCorrection(uint32_t col, LocalTensor<uint32_t> used) {
    }
    float lazy0_ = 0.0f, lazy1_ = 0.0f, lazy2_ = 0.0f;
    uint32_t scratchStride_ = 0, originalTileRows_ = 0, matrixElems_ = 0;
    bool runCube_ = false, cubeAbort_ = false, rowMovePending_ = false;
    bool cubeNoPivotAbort_ = false, forcePivotCube_ = false;
    DetKernelShape t_{};
    GlobalTensor<float> a_, out_, wh_, wl_;
    GlobalTensor<float> lh_, bh_, bl_, c0_, c1_;
    TBuf<TPosition::VECCALC> matrix_, pivot_, scratch_, column_, orderBuf_, outputBuf_;
    TBuf<TPosition::VECCALC> factors_, history_, coeff_, broadcast_;
};

#if DET_AIV_KEY
template <typename DT_A, bool DET_CUBE>
#else
template <typename DT_A>
#endif
__global__ __aicore__ void linalg_det(GM_ADDR A, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LinalgDetTilingData);
    TPipe pipe;
    const __gm__ LinalgDetTilingData *td = reinterpret_cast<const __gm__ LinalgDetTilingData *>(tiling);
    LinalgDetTilingData lean{};
    lean.coreWsElems = td->coreWsElems;
    lean.n = td->n;
    lean.batch = td->batch;
    lean.stride = td->stride;
    lean.tileRows = td->tileRows;
    lean.matPerCore = td->matPerCore;
    lean.resident = td->resident;
    lean.cubeEnabled = td->cubeEnabled;
    lean.scratchStride = td->scratchStride;
    KernelLinalgDet<false> op;
    op.Init(A, det, workspace, lean, &pipe);
    op.Process();
}
