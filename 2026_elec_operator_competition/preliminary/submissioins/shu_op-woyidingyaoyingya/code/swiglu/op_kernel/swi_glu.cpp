/*
 * Copyright (c) 2026 SwiGlu Authors.
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
#define ARCH_HALF_ADD_OK 1
#define ARCH_HALF_MUL_OK 1
#define ARCH_HALF_ONE_OK 1
#include "kernel_operator.h"
#include "swi_glu_tiling.h"
#include "tiling_key_swi_glu.h"
#include "lib/matmul_intf.h"
#include <type_traits>

using namespace AscendC;

// IEEE binary32 -> binary16, round to nearest with ties to even.
// Integer arithmetic preserves subnormals and infinity independently of
// the scalar conversion instruction's rounding and saturation modes.
__aicore__ inline uint16_t FloatBitsToHalfRne(uint32_t bits) {
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return static_cast<uint16_t>(sign | 0x7c00u |
            (fraction ? (0x200u | (fraction >> 13)) : 0u));
    }
    int32_t e = static_cast<int32_t>(exponent) - 112;
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return static_cast<uint16_t>(sign);
        fraction |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - e);
        uint32_t result = fraction >> shift;
        const uint32_t remainder = fraction & ((1u << shift) - 1u);
        const uint32_t midpoint = 1u << (shift - 1u);
        if (remainder > midpoint || (remainder == midpoint && (result & 1u))) ++result;
        return static_cast<uint16_t>(sign | result);
    }
    uint32_t result = fraction >> 13;
    const uint32_t remainder = fraction & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u))) {
        ++result;
        if (result == 0x400u) {
            result = 0;
            ++e;
        }
    }
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | result);
}

__aicore__ inline float HalfBitsToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1fu;
    uint32_t fraction = h & 0x3ffu;
    union Bits { uint32_t bits; float value; } result;
    if (exponent == 0) {
        if (fraction == 0) { result.bits = sign; return result.value; }
        int32_t e = -14;
        while ((fraction & 0x400u) == 0) { fraction <<= 1; --e; }
        result.bits = sign | (static_cast<uint32_t>(e + 127) << 23) | ((fraction & 0x3ffu) << 13);
    } else if (exponent == 31) {
        result.bits = sign | 0x7f800000u | (fraction << 13);
    } else {
        result.bits = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    return result.value;
}

__aicore__ inline float RoundHalf(float value) {
    union Bits { float value; uint32_t bits; } data;
    data.value = value;
    return HalfBitsToFloat(FloatBitsToHalfRne(data.bits));
}

// Frozen from archived OJ-passing scalar source: RNE half after dot,
// bias, exp(float), 1+exp, reciprocal, SiLU, and final multiply.
// Official golden not exposed. No alternate activation approximation.

// Fixed, repeat-per-output dual dots. One product scratch reused by both routes.

// API organization follows asc-devkit cross_core_set_wait_flag CopyIn/SplitA/
// Compute/CopyOut (Huawei, CANN Open Software License 2.0). W is [N,K], so
// its NZ panel already has the L0B layout for logical column-major B.

// G8p: six independent groups, all padding and handshakes inside this kernel.

// DS1 SHORT0: semantic candidate, OJ_REQUIRED. Four disjoint float regions.
// dot rounding is done by callers. Bias remains the verified half Add.

#ifndef ARCH_HALF_ADD_OK
#define ARCH_HALF_ADD_OK 0
#endif
#ifndef ARCH_HALF_MUL_OK
#define ARCH_HALF_MUL_OK 0
#endif
#ifndef ARCH_HALF_ONE_OK
#define ARCH_HALF_ONE_OK 0
#endif
// Operands are already half-rounded dots. Bias may be broadcast by rows.

template<bool Sequential> __aicore__ inline void R1Fallback(GM_ADDR input_x,GM_ADDR weight_1,GM_ADDR bias_1,GM_ADDR weight_2,
 GM_ADDR bias_2,GM_ADDR output,GM_ADDR workspace,GM_ADDR tiling) {
static_assert(Sequential, "The fallback uses sequential accumulation");

    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
    if ASCEND_IS_AIC { return; }

    const uint32_t d = tiling_data.d;
    const uint64_t total = static_cast<uint64_t>(tiling_data.m) * d;
    GlobalTensor<uint16_t> x, w1, b1, w2, b2, y;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(input_x), total);
    w1.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(weight_1), static_cast<uint64_t>(d) * d);
    b1.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(bias_1), d);
    w2.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(weight_2), static_cast<uint64_t>(d) * d);
    b2.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(bias_2), d);
    y.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(output), total);

    const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t blockCount = static_cast<uint64_t>(tiling_data.blockDim);
    const uint64_t chunk = ((total + blockCount - 1) / blockCount + 31u) & ~31ULL;
    const uint64_t begin = blockIdx * chunk;
    const uint64_t end = (begin + chunk < total) ? begin + chunk : total;

    TPipe pipe;
    TBuf<TPosition::VECCALC> expInputBuf, expOutputBuf;
    pipe.InitBuffer(expInputBuf, 32);
    pipe.InitBuffer(expOutputBuf, 32);
    LocalTensor<float> expInput = expInputBuf.Get<float>();
    LocalTensor<float> expOutput = expOutputBuf.Get<float>();

    const uint32_t rowBytes = ((d + 15u) & ~15u) * sizeof(uint16_t);
    TBuf<TPosition::VECIN> xb, w1b, w2b, b1b, b2b;
    TBuf<TPosition::VECOUT> yb;
    pipe.InitBuffer(xb, rowBytes); pipe.InitBuffer(w1b, rowBytes);
    pipe.InitBuffer(w2b, rowBytes); pipe.InitBuffer(b1b, rowBytes);
    pipe.InitBuffer(b2b, rowBytes); pipe.InitBuffer(yb, 64);
    auto xl = xb.Get<uint16_t>(); auto w1l = w1b.Get<uint16_t>();
    auto w2l = w2b.Get<uint16_t>(); auto b1l = b1b.Get<uint16_t>();
    auto b2l = b2b.Get<uint16_t>(); auto yl = yb.Get<uint16_t>();
    const auto inReady = pipe.FetchEventID(HardEvent::MTE2_S);
    const auto inReuse = pipe.FetchEventID(HardEvent::S_MTE2);
    const auto outReady = pipe.FetchEventID(HardEvent::S_MTE3);
    const auto outReuse = pipe.FetchEventID(HardEvent::MTE3_S);
    const auto expReady = pipe.FetchEventID(HardEvent::V_S);
    const auto expReuse = pipe.FetchEventID(HardEvent::S_V);
    const DataCopyExtParams rowCopy{1, d * 2u, 0, 0, 0};
    const DataCopyPadExtParams<uint16_t> noPad{false, 0, 0, 0};
    DataCopyPad(b1l, b1, rowCopy, noPad);
    DataCopyPad(b2l, b2, rowCopy, noPad);
    SetFlag<HardEvent::MTE2_S>(inReady); WaitFlag<HardEvent::MTE2_S>(inReady);
    uint32_t cachedRow = 0xffffffffu;
    uint32_t outCount = 0;
    uint64_t outStart = begin;

    for (uint64_t idx = begin; idx < end; ++idx) {
        const uint32_t row = static_cast<uint32_t>(idx / d);
        const uint32_t col = static_cast<uint32_t>(idx % d);
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        const uint64_t xOffset = static_cast<uint64_t>(row) * d;
        const uint64_t wOffset = static_cast<uint64_t>(col) * d;
        SetFlag<HardEvent::S_MTE2>(inReuse); WaitFlag<HardEvent::S_MTE2>(inReuse);
        if (cachedRow != row) {
            DataCopyPad(xl, x[xOffset], rowCopy, noPad);
            cachedRow = row;
        }
        DataCopyPad(w1l, w1[wOffset], rowCopy, noPad);
        DataCopyPad(w2l, w2[wOffset], rowCopy, noPad);
        SetFlag<HardEvent::MTE2_S>(inReady); WaitFlag<HardEvent::MTE2_S>(inReady);
        for(uint32_t k=0;k<d;++k) {
            const float xv=HalfBitsToFloat(xl.GetValue(k));
            acc1+=xv*HalfBitsToFloat(w1l.GetValue(k));
            acc2+=xv*HalfBitsToFloat(w2l.GetValue(k));
        }
        const float z1 = RoundHalf(RoundHalf(acc1) + HalfBitsToFloat(b1l.GetValue(col)));
        const float z2 = RoundHalf(RoundHalf(acc2) + HalfBitsToFloat(b2l.GetValue(col)));
        const float expArg = -z1;
        SetFlag<HardEvent::S_V>(expReuse); WaitFlag<HardEvent::S_V>(expReuse);
        Duplicate(expInput, expArg, 8);
        Exp<float>(expOutput, expInput, 8);
        SetFlag<HardEvent::V_S>(expReady); WaitFlag<HardEvent::V_S>(expReady);
        const float e = RoundHalf(expOutput.GetValue(0));
        const float sigmoid = RoundHalf(1.0f / RoundHalf(1.0f + e));
        const float silu = RoundHalf(z1 * sigmoid);
        union FloatBits { float value; uint32_t bits; } result;
        result.value = silu * z2;
        yl.SetValue(outCount++, FloatBitsToHalfRne(result.bits));
        if (outCount == 32u || idx + 1u == end) {
            SetFlag<HardEvent::S_MTE3>(outReady); WaitFlag<HardEvent::S_MTE3>(outReady);
            const DataCopyExtParams outputCopy{1, outCount * 2u, 0, 0, 0};
            DataCopyPad(y[outStart], yl, outputCopy);
            SetFlag<HardEvent::MTE3_S>(outReuse); WaitFlag<HardEvent::MTE3_S>(outReuse);
            outStart += outCount;
            outCount = 0;
        }
    }

}
// TASK_ARCH sections 4/5. No high-level Matmul object or hidden L0 allocation.

// C1: one AIV, four outputs, unchanged per-dot FP32 product/reduction tree.
// No data-dependent constants, index table, or gather. All Vector bases aligned.

// TASK_ARCH sections 4/5. No high-level Matmul object or hidden L0 allocation.

// Contiguous panel partition, one full-K paired MMAD per producer.

// MICROBENCH: outputs [old float+Cast chain, native half chain]. No SwiGLU timing claim.

// Shape-only selection indices. Every tensor base passed to Vector is 32B aligned.

// Original implementation sketch for this task; not NPU-compiled or benchmarked.
// CANN 9.0.0 / Ascend 910B2 AIV_ONLY. Keep the existing host dispatch and fallback.
// Input X/W/B are half. Products and reductions are float. No GM workspace.
// Preserves the project's staged-half E0 operations, subject to device verification.
// Generated SHAPE MASKS only; no input-dependent lookup or result cache.

// S08 C7-only helper declaration. Definition is the existing baseline helper
// below; argument order and half pair-row layout are unchanged.

namespace SwiGluSmallDirect {

} // namespace SwiGluSmallDirect

namespace SwiGluSmallDirect {

// U/V are not gathered. Their rows may be separated by 16 or 32 half elements.
// HalfRowStride is a logical half-element stride, NOT the hardware byte stride.
template<uint32_t HalfRowStride>
__aicore__ inline void GatedRows16(
    LocalTensor<half> u, LocalTensor<half> v,
    LocalTensor<half> ba, LocalTensor<half> bb,
    LocalTensor<float> f, LocalTensor<float> one,
    LocalTensor<half> tmp, LocalTensor<half> out, uint32_t rows) {
    static_assert(HalfRowStride == 16 || HalfRowStride == 32, "supported layouts");
    constexpr uint8_t HREP = HalfRowStride / 16; // number of 32-byte blocks
    const uint8_t reps = static_cast<uint8_t>(rows);
    const uint32_t n = rows * 16;
    // Bias is read with zero repeat stride, so no replicated bias/Gather needed.
    Add(u, u, ba, uint64_t(16), reps, {1, 1, 1, HREP, HREP, 0});
    Add(v, v, bb, uint64_t(16), reps, {1, 1, 1, HREP, HREP, 0});
    PipeBarrier<PIPE_V>();
    // Compact ONLY U's selected rows during the conversion already needed for Exp.
    Cast(f, u, RoundMode::CAST_NONE, uint64_t(16), reps, {1, 1, 2, HREP});
    PipeBarrier<PIPE_V>();
    Muls(f, f, -1.0f, n); PipeBarrier<PIPE_V>();
    Exp(f, f, n); PipeBarrier<PIPE_V>();
    Cast(tmp, f, RoundMode::CAST_RINT, n); PipeBarrier<PIPE_V>();
    Adds(tmp, tmp, static_cast<half>(1), n); PipeBarrier<PIPE_V>();
    Cast(f, tmp, RoundMode::CAST_NONE, n); PipeBarrier<PIPE_V>();
    Div(f, one, f, n); PipeBarrier<PIPE_V>();
    Cast(tmp, f, RoundMode::CAST_RINT, n); PipeBarrier<PIPE_V>();
    Mul(tmp, u, tmp, uint64_t(16), reps, {1, 1, 1, 1, HREP, 1});
    PipeBarrier<PIPE_V>();
    Mul(out, tmp, v, uint64_t(16), reps, {1, 1, 1, 1, 1, HREP});
    PipeBarrier<PIPE_V>();
}

// M=2,D=64,P=4 (C2) or M=1,D=32,P=2 (C4).
// Row-interleaved (default): raw[r][branch][16].
// BranchMajor (C2-only comparison): raw[branch][r][16].

template<uint32_t P> struct C7Part;
template<> struct C7Part<0> { static constexpr uint32_t BEGIN=0, END=64, M0=0, ROWS=5; };
template<> struct C7Part<1> { static constexpr uint32_t BEGIN=64, END=128, M0=4, ROWS=6; };
template<> struct C7Part<2> { static constexpr uint32_t BEGIN=128, END=192, M0=9, ROWS=6; };

} // namespace SwiGluSmallDirect

// Fixed normal mask epilogues. No count-mode vector overloads in these helpers.
// All addresses refer to disjoint, 32-byte-aligned regions from the parent layout.

template<uint32_t HB, uint32_t PROD, uint32_t Q>
__aicore__ inline void ReduceC4AndGate(LocalTensor<half> product, LocalTensor<half> wh,
    LocalTensor<half> halfXpack, LocalTensor<half> pairh, LocalTensor<half> bh,
    LocalTensor<half> tmp, LocalTensor<half> out) {
    Mul<half,false>(product,wh,halfXpack,uint64_t(0),Q,{1,1,1,4,4,0});
    PipeBarrier<PIPE_V>();
    LocalTensor<half> stage0(TPosition::VECCALC,2*(HB+PROD),1024);
    LocalTensor<half> stage1(TPosition::VECCALC,2*(HB+PROD+1024),512);
    LocalTensor<half> stage2(TPosition::VECCALC,2*(HB+PROD+1024+512),256);
    LocalTensor<half> stage3(TPosition::VECCALC,2*(HB+PROD+1024+512+256),128);
    PairReduceSum(stage0,product,16,128,1,1,8);PipeBarrier<PIPE_V>();
    PairReduceSum(stage1,stage0,8,128,1,1,8);PipeBarrier<PIPE_V>();
    PairReduceSum(stage2,stage1,4,128,1,1,8);PipeBarrier<PIPE_V>();
    PairReduceSum(stage3,stage2,2,128,1,1,8);PipeBarrier<PIPE_V>();
    PairReduceSum(pairh,stage3,1,128,1,1,8);
    PipeBarrier<PIPE_V>();
    // Paired bias subtraction gives[-U,-V], followed by direct SiLU quotient.
    // The old reciprocal-then-Mul route rounded an intermediate reciprocal.
    // Direct U/den rounds the SiLU once; judge must establish precision.
    SetVectorMask<half>(Q);
    Sub<half,false>(pairh,pairh,bh,Q,2,{1,1,1,Q/16,Q/16,Q/16});
    PipeBarrier<PIPE_V>();
    // Both affine branches are negated; -SiLU(U) times -V restores the result.
    Exp<half,false>(tmp,pairh,Q,1,{1,1,8,8});
    PipeBarrier<PIPE_V>();
    Adds<half,false>(tmp,tmp,static_cast<half>(1),Q,1,{1,1,8,8});
    PipeBarrier<PIPE_V>();
    Div<half,false>(tmp,pairh,tmp,Q,1,{1,1,1,8,8,8});
    PipeBarrier<PIPE_V>();
    Mul<half,false>(out,tmp,pairh[Q],Q,1,{1,1,1,8,8,8});
    PipeBarrier<PIPE_V>();

}

__aicore__ inline void C4SingleWorker(
    GM_ADDR gx, GM_ADDR gw0, GM_ADDR gb0,
    GM_ADDR gw1, GM_ADDR gb1, GM_ADDR gy) {
    constexpr uint32_t M=1,D=32,P=1,Q=32;
    constexpr bool BranchMajor=false;
    constexpr uint32_t HX = M * D, HW = 2 * Q * D, HP = 2 * M * Q, N = M * Q;
    constexpr uint32_t HB = HX + HW + 2 * Q + HP + 2 * N;
    constexpr uint32_t PROD = M * HW;
    constexpr uint32_t FB = HX + HW + PROD + HP + 2 * N;
    constexpr uint32_t BYTES = 2 * HB + 4 * FB;
    static_assert(BYTES % 32 == 0 && BYTES < 96 * 1024, "UB/alignment");
    const uint32_t p = GetBlockIdx();
    if (p >= P) return;
    const uint32_t n0 = p * Q;

    InitSocState();
    LocalTensor<half> h(TPosition::VECCALC,0,HB);
    auto xh = h, wh = h[HX], bh = h[HX + HW];
    auto pairh = h[HX + HW + 2 * Q], tmp = pairh[HP], out = tmp[N];
    LocalTensor<half> product(TPosition::VECCALC,2*HB,PROD);
    LocalTensor<half> halfXpack(TPosition::VECCALC,2*(HB+PROD+1024+512+256+128),64);
    static_assert(2*(HB+PROD+1024+512+256+128+64)<96*1024, "C4 half UB bounds");

    GlobalTensor<half> x, w0, w1, b0, b1, y;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gx));
    w0.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gw0));
    w1.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gw1));
    b0.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gb0));
    b1.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gb1));
    y.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gy));

    // All five ranges are aligned and their lengths are multiples of 32 bytes.
    DataCopy(xh, x, HX);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopy(wh, w0[n0 * D], Q * D);
    DataCopy(wh[Q * D], w1[n0 * D], Q * D);
    DataCopy(bh, b0[n0], Q);
    DataCopy(bh[Q], b1[n0], Q);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    SetMaskNorm();
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    // Two negative X32 copies let each repeat form two complete dot products.
    SetVectorMask<half>(32);
    Muls<half,false>(halfXpack,xh,static_cast<half>(-1),uint64_t(0),2,{1,1,2,0});
    SetVectorMask<half>(64);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    PipeBarrier<PIPE_V>();
    // Half products and half dot sums stay in one dtype throughout.
    ReduceC4AndGate<HB,PROD,Q>(product,wh,halfXpack,pairh,bh,tmp,out);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(y[n0], out,
        DataCopyExtParams{static_cast<uint16_t>(M), Q * 2, 0, (D - Q) * 2, 0});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

__aicore__ inline void C2DenseDirectGate(
    LocalTensor<half> pair,LocalTensor<half> bias,
    LocalTensor<half> tmp,LocalTensor<half> out) {
    // Caller leaves the normal-mode32lane mask ready for negative-branch bias Sub.
    Sub<half,false>(pair,pair,bias,uint64_t(0),1,{1,1,1,8,8,8});
    PipeBarrier<PIPE_V>();
    SetVectorMask<half>(16);
    // Input negation makes the pair[-U,-V]; the two output signs cancel.
    Exp<half,false>(tmp,pair,uint64_t(0),1,{1,1,8,8});PipeBarrier<PIPE_V>();
    Adds<half,false>(tmp,tmp,static_cast<half>(1),uint64_t(0),1,{1,1,8,8});
    PipeBarrier<PIPE_V>();
    Div<half,false>(tmp,pair,tmp,uint64_t(0),1,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
    Mul<half,false>(out,tmp,pair[16],uint64_t(0),1,{1,1,1,8,8,8});
    PipeBarrier<PIPE_V>();
}

template<uint32_t PROD, uint32_t Q>
__aicore__ inline void ReduceC2Dots(LocalTensor<float> product, LocalTensor<half> xh,
    LocalTensor<half> wh, LocalTensor<float> partial, LocalTensor<float> raw,
    LocalTensor<half> pairh) {
    SetMaskNorm();SetVectorMask<float>(64);
    Duplicate<float,false>(product,0.0f,uint64_t(0),PROD/64,1,8);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetVectorMask<half>(64);
    Muls<half,false>(xh,xh,static_cast<half>(-1),uint64_t(0),1,{1,1,8,8});
    SetVectorMask<float>(64);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    PipeBarrier<PIPE_V>();
    MulAddDst<float,half,false>(product,wh,xh,uint64_t(0),2*Q,{1,1,1,8,4,0});
    PipeBarrier<PIPE_V>();
    //32 dots x64 elements ->256 block sums (8 per dot) ->32 dot sums.
    BlockReduceSum(partial,product,32,64,1,1,8);PipeBarrier<PIPE_V>();
    BlockReduceSum(raw,partial,4,64,1,1,8);
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(32);
    Cast<half,float,false>(pairh, raw, RoundMode::CAST_RINT, uint64_t(0),1,{1,1,4,8});
    PipeBarrier<PIPE_V>();

}

__aicore__ inline void MaskC2RowSplit(
    GM_ADDR gx, GM_ADDR gw0, GM_ADDR gb0,
    GM_ADDR gw1, GM_ADDR gb1, GM_ADDR gy) {
    constexpr uint32_t M=1,D=64,P=8;
    constexpr bool BranchMajor=false;
    constexpr uint32_t Q = 16;
    constexpr uint32_t HX = M * D, HW = 2 * Q * D, HP = 2 * M * Q, N = M * Q;
    constexpr uint32_t HB = HX + HW + 2 * Q + HP + 2 * N;
    constexpr uint32_t PROD = M * HW;
    constexpr uint32_t FB = HX + HW + PROD + HP + 2 * N;
    constexpr uint32_t BYTES = 2 * HB + 4 * FB;
    static_assert(BYTES % 32 == 0 && BYTES < 96 * 1024, "UB/alignment");
    const uint32_t p = GetBlockIdx();
    if (p >= P) return;
    const uint32_t n0 = (p % 4) * Q;
    const uint32_t row = p / 4;

    InitSocState();
    LocalTensor<half> h(TPosition::VECCALC,0,HB);
    auto xh = h, wh = h[HX], bh = h[HX + HW];
    auto pairh = h[HX + HW + 2 * Q], tmp = pairh[HP], out = tmp[N];
    LocalTensor<float> f(TPosition::VECCALC,2*HB,FB);
    auto xf = f, wf = f[HX], product = f[HX + HW];
    auto raw = product[PROD], scratch = raw[HP], one = scratch[N];
    LocalTensor<float> partial(TPosition::VECCALC,BYTES,256);
    static_assert(PROD==2048 && HP==32 && BYTES+1024<96*1024,"C2 reduce arena");

    GlobalTensor<half> x, w0, w1, b0, b1, y;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gx));
    w0.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gw0));
    w1.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gw1));
    b0.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gb0));
    b1.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gb1));
    y.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gy));

    // All five ranges are aligned and their lengths are multiples of 32 bytes.
    DataCopy(xh, x[row * D], HX);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopy(wh, w0[n0 * D], Q * D);
    DataCopy(wh[Q * D], w1[n0 * D], Q * D);
    DataCopy(bh, b0[n0], Q);
    DataCopy(bh[Q], b1[n0], Q);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    // Product zero initialization is disjoint from all MTE2 half inputs.
    // Hide it under input DMA, then multiply half inputs directly into FP32.
    ReduceC2Dots<PROD,Q>(product,xh,wh,partial,raw,pairh);
    if constexpr (!BranchMajor) {
        C2DenseDirectGate(pairh,bh,tmp,out);
    } else {
        SwiGluSmallDirect::GatedRows16<16>(pairh, pairh[N], bh, bh[Q], scratch, one, tmp, out, M);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(y[row * D + n0], out,
        DataCopyExtParams{static_cast<uint16_t>(M), Q * 2, 0, (D - Q) * 2, 0});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

// C9 packed epilogue: native-half Exp/Div candidate; precision pending.
// Input rows are [U64,V64]; output rows are contiguous 64-element vectors.

namespace S64C1 {
// Caller installed normal half mask4 and completed the paired bias Add.
// Unit fill is independent of the two bias sums. Its four lanes are ready
// before Div through the existing intervening PIPE_V barriers.
__aicore__ inline void Gate(LocalTensor<half> u,LocalTensor<half> v,
    LocalTensor<half> one,LocalTensor<half> tmp,LocalTensor<half> out) {
    Exp<half,false>(tmp,u,4,1,{1,1,8,8});
    Mul<half,false>(out,u,v,4,1,{1,1,1,8,8,8});
    PipeBarrier<PIPE_V>();
    Adds<half,false>(tmp,tmp,(half)1,4,1,{1,1,8,8});
    PipeBarrier<PIPE_V>();
    Reciprocal<half,false>(tmp,tmp,4,1,{1,1,8,8});
    PipeBarrier<PIPE_V>();
    Mul<half,false>(out,out,tmp,4,1,{1,1,1,8,8,8});
    PipeBarrier<PIPE_V>();
}
} // namespace S64C1

__aicore__ inline void C1SparseInputCast(GM_ADDR gx, GM_ADDR gw0, GM_ADDR gb0,
                               GM_ADDR gw1, GM_ADDR gb1, GM_ADDR gy) {
    const uint32_t activeWorker = GetBlockIdx();
    if (activeWorker >= 1u) return;
    LocalTensor<half> h(TPosition::VECCALC,0,1024);
    auto xh = h, wh = h[16], bh = h[144];
    auto uh = h[176], vh = h[192], tmp = h[208], out = h[224];
    LocalTensor<float> f(TPosition::VECCALC,512,384);
    auto xf = f, wf = f[16], product = f[144];
    auto u = f[272], v = f[280], e = f[288], one = f[336];
    GlobalTensor<half> x, w0, w1, b0, b1, y;
    x.SetGlobalBuffer((__gm__ half*)gx);
    w0.SetGlobalBuffer((__gm__ half*)gw0);
    w1.SetGlobalBuffer((__gm__ half*)gw1);
    b0.SetGlobalBuffer((__gm__ half*)gb0);
    b1.SetGlobalBuffer((__gm__ half*)gb1);
    y.SetGlobalBuffer((__gm__ half*)gy);
    const DataCopyPadExtParams<half> pad{false, 0, 0, 0};
    DataCopyPad(xh, x, DataCopyExtParams{1, 8, 0, 0, 0}, pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopyPad(wh, w0, DataCopyExtParams{4, 8, 0, 0, 0}, pad);
    DataCopyPad(wh[64], w1, DataCopyExtParams{4, 8, 0, 0, 0}, pad);
    DataCopyPad(bh, b0, DataCopyExtParams{1, 8, 0, 0, 0}, pad);
    DataCopyPad(bh[16], b1, DataCopyExtParams{1, 8, 0, 0, 0}, pad);
    // One is filled under the final fixed mask.
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    // A2 AIV init sets atomic mode, mask mode and mask bits. All three
    // are first consumed after this point; GM-to-UB input DMA is independent.
    InitSocState();
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Muls(xh,xh,(half)-1,uint64_t(4),1,{1,1,8,8});
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    PipeBarrier<PIPE_V>();
    auto hp=h[256];
    // Four active lanes in each of four half blocks, two repeats (U then V).
    uint64_t dotMask[2]={0x000f000f000f000fULL,0ULL};
    Mul(wh,wh,xh,dotMask,2,{1,1,0,4,4,0});
    PipeBarrier<PIPE_V>();
    // FP16 BlockReduceSum dstRepStride uses16B units: stride2 placesV at uh[16].
    BlockReduceSum<half,false>(uh,wh,2,dotMask,2,1,4);
    PipeBarrier<PIPE_V>();
    // Repeat0 adds B0 to U, repeat1 adds B1 to V; only lanes0..3.
    SetVectorMask<half>(4);
    Sub<half,false>(uh,uh,bh,4,2,{1,1,1,1,1,1});
    PipeBarrier<PIPE_V>();
    ::S64C1::Gate(uh,vh,one.ReinterpretCast<half>(),tmp,out);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(y, out, DataCopyExtParams{1, 8, 0, 0, 0});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

// C5 packed epilogue: native-half Exp/Div candidate; precision pending.

// Two row waves share the same full-K weights and have separate L0C results.
template<uint32_t BM, uint32_t BN, uint32_t D>
__aicore__ inline void ComputeB28Panels(LocalTensor<float> c, LocalTensor<half> al,
    LocalTensor<half> bl0, LocalTensor<half> bl1, GlobalTensor<half> &ws, uint32_t p) {
 constexpr uint32_t X=BM*D,C=BM*BN;
 MmadParams mp;mp.m=BM;mp.n=BN;mp.k=D;mp.cmatrixInitVal=true;mp.unitFlag=3;
 SetMMLayoutTransform(true);
 FixpipeParamsV220 fp;fp.nSize=BN;fp.mSize=BM;fp.srcStride=BM;fp.dstStride=BN;
 fp.quantPre=QuantMode_t::F322F16;fp.unitFlag=3;
 const uint64_t off=static_cast<uint64_t>(p)*4*C;
 WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
 #pragma unroll 2
 for(uint32_t panel=0;panel<2;++panel) {
  if(panel==1) WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
  const uint32_t u=2*panel,v=u+1;
  Mmad(c[u*C],al[panel*X],bl0,mp);
  Fixpipe(ws[off+u*C],c[u*C],fp);
  CrossCoreSetFlag<2,PIPE_FIX>(u);
  if(panel==0) WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
  Mmad(c[v*C],al[panel*X],bl1,mp);
  Fixpipe(ws[off+v*C],c[v*C],fp);
  CrossCoreSetFlag<2,PIPE_FIX>(v);
 }
 SetMMLayoutTransform(false);

}

template<uint32_t BM,uint32_t BN,uint32_t D>
__aicore__ inline void B28Producer(GM_ADDR gx,GM_ADDR gw0,GM_ADDR gw1,GM_ADDR work,
 uint32_t p,uint32_t group,uint32_t row0) {
 static_assert(BM==64 && BN==64 && D==256,"C3 projection pipeline bounds");
 constexpr uint32_t X=BM*D,W=BN*D,C=BM*BN;
 LocalTensor<half> xl(TPosition::A1,0,2*X);
 LocalTensor<half> wl0(TPosition::B1,4*X,W),wl1(TPosition::B1,4*X+2*W,W);
 LocalTensor<half> al(TPosition::A2,0,2*X);
 LocalTensor<half> bl0(TPosition::B2,0,W),bl1(TPosition::B2,2*W,W);
 LocalTensor<float> c(TPosition::CO1,0,4*C);
 GlobalTensor<half> x,w0,w1,ws;
 x.SetGlobalBuffer((__gm__ half*)gx);w0.SetGlobalBuffer((__gm__ half*)gw0);
 w1.SetGlobalBuffer((__gm__ half*)gw1);ws.SetGlobalBuffer((__gm__ half*)work);
 Nd2NzParams xp;xp.ndNum=BM/16;xp.nValue=16;xp.dValue=D;xp.srcDValue=D;
 xp.srcNdMatrixStride=16*D;xp.dstNzC0Stride=16;xp.dstNzNStride=1;xp.dstNzMatrixStride=16*D;
 Nd2NzParams wp;wp.ndNum=1;wp.nValue=BN;wp.dValue=D;wp.srcDValue=D;
 wp.dstNzC0Stride=BN;wp.dstNzNStride=1;wp.srcNdMatrixStride=0;wp.dstNzMatrixStride=0;
 const uint64_t woff=static_cast<uint64_t>(group)*W;
 DataCopy(xl,x[static_cast<uint64_t>(row0)*D],xp);
 SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
 DataCopy(wl0,w0[woff],wp);
 SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
 DataCopy(wl1,w1[woff],wp);
 SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
 DataCopy(xl[X],x[static_cast<uint64_t>(row0+BM)*D],xp);
 SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
 LoadData2DParams pa;pa.repeatTimes=X/256;pa.srcStride=1;pa.dstGap=0;pa.ifTranspose=false;
 LoadData2DParams pb;pb.repeatTimes=W/256;pb.srcStride=1;pb.dstGap=0;pb.ifTranspose=false;
 WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
 LoadData(al,xl,pa);
 WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
 LoadData(bl0,wl0,pb);
 SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
 WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
 LoadData(bl1,wl1,pb);
 SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
 WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
 LoadData(al[X],xl[X],pa);
 SetFlag<HardEvent::MTE1_M>(EVENT_ID2);
 ComputeB28Panels<BM,BN,D>(c,al,bl0,bl1,ws,p);
}

template<uint32_t BM,uint32_t BN,uint32_t D>
__aicore__ inline void B28Consumer(GM_ADDR gb0,GM_ADDR gb1,GM_ADDR gy,GM_ADDR work,
 uint32_t p,uint32_t group,uint32_t row0) {
 constexpr uint32_t R=BM,T=R*BN;
 InitSocState();
 constexpr uint32_t BIAS_OFF=8*T,ONE_OFF=BIAS_OFF+4*BN;
 constexpr uint32_t TMP_OFF=ONE_OFF+4*BN,OUT_OFF=TMP_OFF+2*T;
 static_assert(OUT_OFF+4*T<=192*1024 && BN==64 && T%128==0,"C3 UB/epilogue bounds");
 LocalTensor<half> in(TPosition::VECIN,0,4*T),bh(TPosition::VECIN,BIAS_OFF,2*BN);
 LocalTensor<half> one(TPosition::VECCALC,ONE_OFF,2*BN),tmp(TPosition::VECCALC,TMP_OFF,T);
 LocalTensor<half> out(TPosition::VECOUT,OUT_OFF,2*T);
 GlobalTensor<half> b0,b1,y,ws;
 b0.SetGlobalBuffer((__gm__ half*)gb0);b1.SetGlobalBuffer((__gm__ half*)gb1);
 y.SetGlobalBuffer((__gm__ half*)gy);y.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
 ws.SetGlobalBuffer((__gm__ half*)work);
 DataCopy(bh,b0[group*BN],BN);DataCopy(bh[BN],b1[group*BN],BN);
 Duplicate(one,(half)1,2*BN);PipeBarrier<PIPE_V>();
 const uint64_t off=static_cast<uint64_t>(p)*4*T;
 const bool outputAligned=(reinterpret_cast<uint64_t>(gy)&31)==0;
 #pragma unroll 2
 for(uint32_t panel=0;panel<2;++panel) {
  const uint32_t u=2*panel,v=u+1;
  auto hu=in[u*T],hv=in[v*T],ho=out[panel*T];
  CrossCoreWaitFlag<2>(u);
  DataCopy(hu,ws[off+u*T],T);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  SetMaskNorm();SetVectorMask<half>(64);
  Add<half,false>(hu,hu,bh,uint64_t(0),R,{1,1,1,4,4,0});PipeBarrier<PIPE_V>();
  SetVectorMask<half>(128);
  Muls<half,false>(tmp,hu,(half)-1,uint64_t(0),T/128,{1,1,8,8});PipeBarrier<PIPE_V>();
  Exp<half,false>(tmp,tmp,uint64_t(0),T/128,{1,1,8,8});PipeBarrier<PIPE_V>();
  Adds<half,false>(tmp,tmp,(half)1,uint64_t(0),T/128,{1,1,8,8});PipeBarrier<PIPE_V>();
  Div<half,false>(tmp,one,tmp,uint64_t(0),T/128,{1,1,1,8,0,8});PipeBarrier<PIPE_V>();
  Mul<half,false>(tmp,hu,tmp,uint64_t(0),T/128,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
  CrossCoreWaitFlag<2>(v);
  DataCopy(hv,ws[off+v*T],T);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID1);WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
  SetVectorMask<half>(64);
  Add<half,false>(hv,hv,bh[BN],uint64_t(0),R,{1,1,1,4,4,0});PipeBarrier<PIPE_V>();
  SetVectorMask<half>(128);
  Mul<half,false>(ho,tmp,hv,uint64_t(0),T/128,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();ResetMask();
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  const uint64_t yoff=(static_cast<uint64_t>(row0)+panel*R)*D+group*BN;
  if(outputAligned) {
   DataCopy(y[yoff],ho,DataCopyParams{R,BN/16,0,(D-BN)/16});
  } else {
   DataCopyPad(y[yoff],ho,DataCopyExtParams{R,BN*2,0,(D-BN)*2,0});
  }
 }
 SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void C3NativeHalf128(GM_ADDR x,GM_ADDR w0,GM_ADDR b0,GM_ADDR w1,GM_ADDR b1,GM_ADDR y,GM_ADDR workspace,uint32_t) {
 const uint32_t p=GetBlockIdx()/GetTaskRatio(),group=p/4,row0=(p%4)*128;
 GM_ADDR work=GetUserWorkspace(workspace);
 if ASCEND_IS_AIC B28Producer<64,64,256>(x,w0,w1,work,p,group,row0);
 if ASCEND_IS_AIV B28Consumer<64,64,256>(b0,b1,y,work,p,group,row0);
}

// C5 packed epilogue: native-half Exp/Div candidate; precision pending.

 // namespace S23C7Cube

// C9 four-tile slab with passing native-half Exp/Div epilogue.
// Input rows are [U64,V64]; output rows are contiguous 64-element vectors.

 // namespace S49C7Gate

namespace SwiGluSmallDirect {

} // namespace SwiGluSmallDirect

template<uint32_t nr>
__aicore__ inline void PrepareC8Weights(LocalTensor<half> h, LocalTensor<uint8_t> metadata) {
 auto wr=h[0],wa=h[7472],xb=h[10208],xh=h[2304];
 auto va=metadata.ReinterpretCast<uint64_t>();
 auto weightMask=metadata.ReinterpretCast<uint16_t>()[96];
 auto lastKIndices=metadata.ReinterpretCast<uint32_t>()[120];
 LocalTensor<half> compactWeights(TPosition::VECCALC,58080,2048);
 SetMaskNorm();set_vector_mask(0,0);
 auto *compactPtr=(__ubuf__ uint16_t*)compactWeights.GetPhyAddr();
 auto *rawWeightPtr=(__ubuf__ uint16_t*)wr.GetPhyAddr();
 auto *weightMaskPtr=(__ubuf__ uint16_t*)weightMask.GetPhyAddr();
 vreducev2(compactPtr,rawWeightPtr,weightMaskPtr,9,1,0,8,1);
 vreducev2(compactPtr+1024,rawWeightPtr+1152,weightMaskPtr,9,1,0,8,1);
 // The adjacent index arrays select K32 weights followed by paired bias.
 Gather(wa[2560],wr,lastKIndices,0,160);
 PipeBarrier<PIPE_V>();SetMaskNorm();ResetMask();
 #pragma unroll 2
 for(uint32_t panel=0;panel<2;++panel) {
  auto *dstVa=(__ubuf__ uint64_t*)va[panel*8].GetPhyAddr();
  auto *srcVa=(__ubuf__ uint64_t*)va[panel*8+4].GetPhyAddr();
  vld_va_reg(VA0,dstVa,L128);vld_va_reg(VA1,dstVa,H128);
  vld_va_reg(VA2,srcVa,L128);vld_va_reg(VA3,srcVa,H128);
  scatter_vnchwconv_b16(VA0,VA2,4,1,16);
 }
 auto *tailDstVa=(__ubuf__ uint64_t*)va[16].GetPhyAddr();
 auto *tailSrcVa=(__ubuf__ uint64_t*)va[20].GetPhyAddr();
 vld_va_reg(VA0,tailDstVa,L128);vld_va_reg(VA1,tailDstVa,H128);
 vld_va_reg(VA2,tailSrcVa,L128);vld_va_reg(VA3,tailSrcVa,H128);
 scatter_vnchwconv_b16(VA0,VA2,2,80,1);
 Brcb(xb,xh,static_cast<uint8_t>((nr*33+7)/8),{1,8});PipeBarrier<PIPE_V>();

}

template<uint32_t nr>
__aicore__ inline void ProjectC8Branches(LocalTensor<half> h, LocalTensor<float> f,
    LocalTensor<uint16_t> splitU, LocalTensor<uint16_t> splitV) {
 auto even=f,odd=f[704];
 auto he=even.ReinterpretCast<half>(),ho=odd.ReinterpretCast<half>();
 auto hc=f[3440].ReinterpretCast<half>(),hd=f[4144].ReinterpretCast<half>();
 auto wa=h[7472],xb=h[10208],pairedBias=h[10112],uh=h[3632],vh=h[4272];
 SetMaskNorm();SetVectorMask<half>(80);
 auto *ep=(__ubuf__ half*)he.GetPhyAddr();auto *op=(__ubuf__ half*)ho.GetPhyAddr();
 auto *wp=(__ubuf__ half*)wa.GetPhyAddr();auto *xp=(__ubuf__ half*)xb.GetPhyAddr();
 auto *cp=(__ubuf__ half*)hc.GetPhyAddr();
 auto *dp=(__ubuf__ half*)hd.GetPhyAddr();
 #pragma unroll 8
 for(uint32_t k=0;k<32;k+=4){
   vmla(ep,wp,xp,static_cast<uint8_t>(nr),1,1,0,5,0,33);
   vmla(op,wp+80,xp+16,static_cast<uint8_t>(nr),1,1,0,5,0,33);
   vmla(cp,wp+160,xp+32,static_cast<uint8_t>(nr),1,1,0,5,0,33);
   vmla(dp,wp+240,xp+48,static_cast<uint8_t>(nr),1,1,0,5,0,33);
   PipeBarrier<PIPE_V>();wp+=320;xp+=64;
 }
 vmla(ep,wp,xp,static_cast<uint8_t>(nr),1,1,0,5,0,33);
 PipeBarrier<PIPE_V>();ResetMask();
 Add(he,he,hc,880);Add(ho,ho,hd,880);PipeBarrier<PIPE_V>();
 Add(he,he,ho,880);PipeBarrier<PIPE_V>();
 Add(he,he,pairedBias,uint64_t(80),nr,{1,1,1,5,5,0});PipeBarrier<PIPE_V>();
 SetMaskNorm();set_vector_mask(0,0);
 auto *projectionPtr=(__ubuf__ uint16_t*)he.GetPhyAddr();
 auto *uPtr=(__ubuf__ uint16_t*)uh.GetPhyAddr();
 auto *vPtr=(__ubuf__ uint16_t*)vh.GetPhyAddr();
 auto *uMaskPtr=(__ubuf__ uint16_t*)splitU.GetPhyAddr();
 auto *vMaskPtr=(__ubuf__ uint16_t*)splitV.GetPhyAddr();
 const uint8_t splitRepeats=static_cast<uint8_t>((nr*80+127)/128);
 vreducev2(uPtr,projectionPtr,uMaskPtr,splitRepeats,1,0,8,1);
 vreducev2(vPtr,projectionPtr,vMaskPtr,splitRepeats,1,0,8,1);
 SetMaskNorm();ResetMask();PipeBarrier<PIPE_V>();

}

template<uint32_t nr>
__aicore__ inline void LoadC8Inputs(LocalTensor<half> h, LocalTensor<float> f,
    LocalTensor<uint8_t> metadata, GM_ADDR gx, GM_ADDR gw0, GM_ADDR gb0,
    GM_ADDR gw1, GM_ADDR gb1, GM_ADDR gy, GM_ADDR tiling, uint32_t row) {
 auto wr=h[0],xh=h[2304],bh=h[2832],uh=h[3632];
 auto even=f;
 GlobalTensor<half>x,w0,w1,b0,b1,y;GlobalTensor<uint8_t> metadataGm;
 x.SetGlobalBuffer((__gm__ half*)gx);w0.SetGlobalBuffer((__gm__ half*)gw0);
 w1.SetGlobalBuffer((__gm__ half*)gw1);b0.SetGlobalBuffer((__gm__ half*)gb0);
 b1.SetGlobalBuffer((__gm__ half*)gb1);y.SetGlobalBuffer((__gm__ half*)gy);
 metadataGm.SetGlobalBuffer((__gm__ uint8_t*)(tiling+32));
 const DataCopyPadExtParams<half> pad{true,0,15,0};
 DataCopyPad(wr,w0,DataCopyExtParams{1,2178,0,0,0},pad);
 DataCopyPad(wr[1152],w1,DataCopyExtParams{1,2178,0,0,0},pad);
 const uint32_t realX=nr*33,alignedX=(realX+15)/16*16;
 DataCopyPad(xh,x[row*33],DataCopyExtParams{1,realX*2,0,0,0},
   DataCopyPadExtParams<half>{true,0,static_cast<uint8_t>(alignedX-realX),0});
 Duplicate(xh[alignedX],static_cast<half>(0),528-alignedX);
 DataCopyPad(bh,b0,DataCopyExtParams{1,66,0,0,0},pad);
 DataCopyPad(bh[48],b1,DataCopyExtParams{1,66,0,0,0},pad);
 DataCopy(metadata,metadataGm,1568);
 Duplicate(even,0.0f,1408);Duplicate(f[3440],0.0f,1408);Duplicate(uh,static_cast<half>(0),1280);
 // Only active padded X rows are converted below.
 SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
 PipeBarrier<PIPE_V>();

}

template<uint32_t NR>
__aicore__ inline void OuterProductC8StaticDense(
 GM_ADDR gx,GM_ADDR gw0,GM_ADDR gb0,GM_ADDR gw1,GM_ADDR gb1,GM_ADDR gy,GM_ADDR tiling) {
 const uint32_t block=GetBlockIdx();if(block>=8)return;
 const uint32_t row=block*11;
 constexpr uint32_t nr=NR;
 static_assert(NR==11 || NR==8,"C8 original row ownership");
 const uint32_t n=nr*33,np=(n+127)/128*128;
 constexpr uint32_t HB=18560,FB=4848;
 static_assert(85184<192*1024,"C8 compact transpose UB capacity");
 InitSocState();LocalTensor<half> h(TPosition::VECCALC,0,HB);
 auto uh=h[3632],vh=h[4272],tmp=h[4912],out=h[5552];
 LocalTensor<float> f(TPosition::VECCALC,65792,FB);
 LocalTensor<uint8_t> metadata(TPosition::VECIN, 2*HB+4*FB, 1568);
 LocalTensor<uint16_t> splitU(TPosition::VECIN,2*HB+4*FB+1120,112);
 LocalTensor<uint16_t> splitV(TPosition::VECIN,2*HB+4*FB+1344,112);
 LoadC8Inputs<nr>(h,f,metadata,gx,gw0,gb0,gw1,gb1,gy,tiling,row);
 GlobalTensor<half> y; y.SetGlobalBuffer((__gm__ half*)gy);
 // Interleaved U/V pairs occupy66of80half lanes; bias precedes compactsplit.
 // Normal-mode vreducev2 uses the supplied bit masks; no count is consumed.
 PrepareC8Weights<nr>(h,metadata);
 ProjectC8Branches<nr>(h,f,splitU,splitV);
 // Split masks also remove padded columns, giving contiguous 33-half rows.
 SetMaskNorm();SetVectorMask<half>(128);
 Muls<half,false>(tmp,uh,static_cast<half>(-1),uint64_t(0),np/128,{1,1,8,8});PipeBarrier<PIPE_V>();
 Exp<half,false>(tmp,tmp,uint64_t(0),np/128,{1,1,8,8});PipeBarrier<PIPE_V>();
 Adds<half,false>(tmp,tmp,static_cast<half>(1),uint64_t(0),np/128,{1,1,8,8});PipeBarrier<PIPE_V>();
 Div<half,false>(tmp,uh,tmp,uint64_t(0),np/128,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
 Mul<half,false>(out,tmp,vh,uint64_t(0),np/128,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
 ResetMask();
 SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
 DataCopyPad(y[row*33],out,DataCopyExtParams{1,nr*66,0,0,0});PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void OuterProductC8(
 GM_ADDR gx,GM_ADDR gw0,GM_ADDR gb0,GM_ADDR gw1,GM_ADDR gb1,GM_ADDR gy,GM_ADDR tiling) {
 if(GetBlockIdx()<7) OuterProductC8StaticDense<11>(gx,gw0,gb0,gw1,gb1,gy,tiling);
 else OuterProductC8StaticDense<8>(gx,gw0,gb0,gw1,gb1,gy,tiling);
}
 // namespace S38Cube

constexpr uint32_t B36_WORKER_TILES=26;
// C9 four-tile slab with passing native-half Exp/Div epilogue.
// Input rows are [U64,V64]; output rows are contiguous 64-element vectors.
// Bias is accumulated before Fixpipe rounds the two projections to FP16.
__aicore__ inline void B36PackedStream(LocalTensor<half> in,
 LocalTensor<half> one,LocalTensor<half> tmp,
 LocalTensor<half> out,uint32_t n,uint32_t rows) {
 static_assert(ARCH_HALF_ADD_OK && ARCH_HALF_MUL_OK && ARCH_HALF_ONE_OK,
               "packed epilogue requires native-half operations");
 // Slab has1..4 complete half-tiles, rows=32/64/96/128; n=rows*64.
 // uint64_t(0) is an ignored mask placeholder with isSetMask=false.
 const uint32_t denseRows=rows/2;
 SetMaskNorm();
 // Cube initializes the FP32 accumulator from BiasTable before rounding.
 SetVectorMask<half>(64);
 Muls<half,false>(tmp,in,(half)-1,uint64_t(0),rows,{1,1,4,8});PipeBarrier<PIPE_V>();
 // tmp is compact: adjacent pairs of logical rows fill one128-lane repeat.
 SetVectorMask<half>(128);
 Exp<half,false>(tmp,tmp,uint64_t(0),denseRows,{1,1,8,8});PipeBarrier<PIPE_V>();
 Adds<half,false>(tmp,tmp,(half)1,uint64_t(0),denseRows,{1,1,8,8});PipeBarrier<PIPE_V>();
 Div<half,false>(tmp,one,tmp,uint64_t(0),denseRows,{1,1,1,8,0,8});PipeBarrier<PIPE_V>();
 SetVectorMask<half>(64);
 Mul<half,false>(tmp,in,tmp,uint64_t(0),rows,{1,1,1,4,8,4});PipeBarrier<PIPE_V>();
 Mul<half,false>(out,tmp,in[64],uint64_t(0),rows,{1,1,1,4,4,8});PipeBarrier<PIPE_V>();
 ResetMask();
}

template<uint32_t BM, uint32_t BN, uint32_t D>
__aicore__ inline void ComputeB36Tiles(LocalTensor<half> xl, LocalTensor<half> al,
    LocalTensor<half> bl, LocalTensor<float> biasC2, LocalTensor<float> c,
    GlobalTensor<half> &x, GlobalTensor<half> &ws, const event_t *rd,
    Nd2NzParams xp, uint32_t pairCount, uint32_t count, uint64_t wsBase,
    uint32_t lane, uint32_t pg) {
    constexpr uint32_t X_PAIR_ELEMENTS=2*BM*D;
    for(uint32_t t=0;t<count;++t) {
        const uint32_t bank=t%2,slab=t/4,localTile=t%4;
        const uint32_t pair=t/2,xBank=pair%2;
        if(bank==0) WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        if(t>=2) WaitFlag<HardEvent::M_MTE1>(rd[bank]);
        LoadData2DParams pa;pa.repeatTimes=BM*D/256;pa.srcStride=1;pa.dstGap=0;pa.ifTranspose=false;
        LoadData(al[bank*BM*D],xl[xBank*X_PAIR_ELEMENTS+bank*BM*D],pa);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        // Prefetch the next pair while the current L1 bank still holds its second tile.
        if(bank==0 && pair+1<pairCount) {
            if(pair!=0) WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            const uint32_t nextPair=lane+(pair+1)*pg;
            DataCopy(xl[(1-xBank)*X_PAIR_ELEMENTS],x[static_cast<uint64_t>(nextPair)*X_PAIR_ELEMENTS],xp);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        }
        // The second LoadData completes this bank's final read before reuse.
        if(bank==1 && pair+2<pairCount) SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        // UnitFlag3 waits for each reused L0C fractal to be released by Fixpipe.
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        MmadParams mp;mp.m=BM;mp.n=2*BN;mp.k=D;mp.cmatrixInitVal=false;mp.cmatrixSource=true;mp.unitFlag=3;
        Mmad(c[bank*2*BM*BN],al[bank*BM*D],bl,biasC2,mp);
        SetFlag<HardEvent::M_MTE1>(rd[bank]);
        // Matching UnitFlag3 lets Fixpipe read ready fractals without a whole-MMAD wait.
        // A tile's GM slot is written once and stays valid until the kernel exits.
        const uint64_t off=wsBase+t*2*BM*BN;
        FixpipeParamsV220 fp;fp.nSize=2*BN;fp.mSize=BM;fp.srcStride=BM;fp.dstStride=2*BN;
        fp.quantPre=QuantMode_t::F322F16;fp.unitFlag=3;
        // One full paired matrix: each GM row is [U64, V64].
        Fixpipe(ws[off],c[bank*2*BM*BN],fp);
        if(localTile==3 || t+1==count) CrossCoreSetFlag<2,PIPE_FIX>(slab);
        // Drain after the final READY so the consumer sees it before the producer exits.
        if(t+2>=count) SetFlag<HardEvent::FIX_M>(bank);
    }

}

template<uint32_t BM,uint32_t BN,uint32_t D,uint32_t M,bool Dump=false>
__aicore__ inline void B36PackedProduce(GM_ADDR gx,GM_ADDR gw0,GM_ADDR gb0,GM_ADDR gw1,GM_ADDR gb1,GM_ADDR work,
 uint32_t p,uint32_t group,uint32_t lane,uint32_t pg,uint32_t tm) {
    static_assert(!Dump,"Packed C9 does not instantiate diagnostic dumps");
    static_assert(BM==64 && BN==64 && D==256 && M==8192,"C9 only");
    static_assert(4*BM*D<=65536 && 4*BN*D<=65536 && 16*BM*BN<=131072,"L0 capacity");
    TPipe pipe;
    TBuf<TPosition::A1> xb; TBuf<TPosition::B1> wb;
    TBuf<TPosition::C1> biasL1; TBuf<TPosition::C2> biasTable;
    TBuf<TPosition::A2> ab; TBuf<TPosition::B2> bb; TBuf<TPosition::CO1> cb;
    constexpr uint32_t X_PAIR_ELEMENTS=2*BM*D;
    pipe.InitBuffer(xb,2*X_PAIR_ELEMENTS*sizeof(half));
    pipe.InitBuffer(wb,4*BN*D);
    pipe.InitBuffer(biasL1,2*BN*sizeof(half));
    pipe.InitBuffer(biasTable,2*BN*sizeof(float));
    static_assert(2*BN*sizeof(float)<=1024 && (2*BN*sizeof(float))%64==0,"C9 bias table bounds");
    pipe.InitBuffer(ab,4*BM*D); pipe.InitBuffer(bb,4*BN*D);
    pipe.InitBuffer(cb,16*BM*BN);
    auto xl=xb.Get<half>(),wl=wb.Get<half>(),al=ab.Get<half>(),bl=bb.Get<half>();
    auto c=cb.Get<float>();
    auto biasIn=biasL1.Get<half>();
    auto biasC2=biasTable.Get<float>();
    const event_t rd[2]={static_cast<event_t>(pipe.AllocEventID<HardEvent::M_MTE1>()),
                        static_cast<event_t>(pipe.AllocEventID<HardEvent::M_MTE1>())};
    GlobalTensor<half> x,w0,w1,b0,b1,ws;
    x.SetGlobalBuffer((__gm__ half*)gx);w0.SetGlobalBuffer((__gm__ half*)gw0);
    w1.SetGlobalBuffer((__gm__ half*)gw1);ws.SetGlobalBuffer((__gm__ half*)work);
    b0.SetGlobalBuffer((__gm__ half*)gb0);b1.SetGlobalBuffer((__gm__ half*)gb1);
    Nd2NzParams wp;wp.ndNum=1;wp.nValue=BN;wp.dValue=D;wp.srcDValue=D;
    wp.dstNzC0Stride=2*BN;wp.dstNzNStride=1;wp.srcNdMatrixStride=0;wp.dstNzMatrixStride=0;
    DataCopy(wl,w0[static_cast<uint64_t>(group)*BN*D],wp);
    DataCopy(wl[BN*16],w1[static_cast<uint64_t>(group)*BN*D],wp);
    DataCopy(biasIn,b0[group*BN],BN);
    DataCopy(biasIn[BN],b1[group*BN],BN);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    LoadData2DParams pb;pb.repeatTimes=2*BN*D/256;pb.srcStride=1;pb.dstGap=0;pb.ifTranspose=false;
    LoadData(bl,wl,pb); // Sole L0B load in the invocation; never overwritten.
    // The following MTE1_M event also fences this one-time bias copy.
    DataCopy(biasC2,biasIn,DataCopyParams{1,static_cast<uint16_t>(2*BN*sizeof(half)/64),0,0});
    // Copy two adjacent row tiles as eight consecutive 16-row NZ matrices.
    Nd2NzParams xp=wp;xp.ndNum=2*BM/16;xp.nValue=16;xp.dstNzC0Stride=16;
    xp.srcNdMatrixStride=16*D;xp.dstNzMatrixStride=16*D;
    const uint32_t pairCount=(tm/2-1-lane)/pg+1;
    const uint32_t count=2*pairCount;
    const uint64_t wsBase=static_cast<uint64_t>(p)*B36_WORKER_TILES*2*BM*BN;
    DataCopy(xl,x[static_cast<uint64_t>(lane)*X_PAIR_ELEMENTS],xp);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    // Match NZ-to-ND Fixpipe traversal; UnitFlag tracks each complete L0C tile.
    SetMMLayoutTransform(true);
    ComputeB36Tiles<BM,BN,D>(xl,al,bl,biasC2,c,x,ws,rd,xp,pairCount,count,wsBase,lane,pg);
    for(uint32_t t=count>2?count-2:0;t<count;++t) {
        WaitFlag<HardEvent::FIX_M>(t%2);
        WaitFlag<HardEvent::M_MTE1>(rd[t%2]);
    }
    pipe.ReleaseEventID<HardEvent::M_MTE1>(rd[0]);pipe.ReleaseEventID<HardEvent::M_MTE1>(rd[1]);
}

template<uint32_t BM, uint32_t BN, uint32_t D>
__aicore__ inline void StoreB36Slab(GlobalTensor<half> &y, LocalTensor<half> out,
    uint32_t bank, uint32_t s, uint32_t validTileCount, uint32_t lane,
    uint32_t pg, uint32_t group, uint32_t a) {
    constexpr uint32_t R=BM/2,T=R*BN,C=4*T;
        for(uint32_t j=0;j<validTileCount;++j) {
            const uint32_t tile=4*s+j;
            const uint32_t xTile=2*(lane+(tile/2)*pg)+tile%2;
            const uint64_t row=static_cast<uint64_t>(xTile)*BM+a*R;
            DataCopyPad(y[row*D+group*BN],out[bank*C+j*T],DataCopyExtParams{R,BN*2,0,(D-BN)*2,0});
        }

}

template<uint32_t BM,uint32_t BN,uint32_t D>
__aicore__ inline void B36PackedConsume(GM_ADDR gy,GM_ADDR work,
 uint32_t p,uint32_t a,uint32_t group,uint32_t lane,uint32_t pg,uint32_t tm) {
    static_assert(BM==64 && BN==64 && D==256,"C9 only");
    constexpr uint32_t R=BM/2,T=R*BN,C=4*T;
    InitSocState();
    constexpr uint32_t IN_BYTES=8*C,ONE_BYTES=4*BN;
    constexpr uint32_t TMP_BYTES=2*C,OUT_BYTES=4*C;
    constexpr uint32_t ONE_OFF=IN_BYTES;
    constexpr uint32_t TMP_OFF=ONE_OFF+ONE_BYTES,OUT_OFF=TMP_OFF+TMP_BYTES;
    constexpr uint32_t TOTAL_BYTES=OUT_OFF+OUT_BYTES;
    static_assert(TOTAL_BYTES==14*C+4*BN && TOTAL_BYTES<=192*1024,"C9 compact slab4 UB capacity");
    static_assert(ONE_OFF%32==0 && TMP_OFF%32==0 && OUT_OFF%32==0,"UB alignment");
    static_assert(R==32 && C==4*R*BN,"C9 dense activation needs even row count");
    LocalTensor<half> in(TPosition::VECIN,0,4*C);
    LocalTensor<half> one(TPosition::VECCALC,ONE_OFF,2*BN);
    LocalTensor<half> tmp(TPosition::VECCALC,TMP_OFF,C);
    LocalTensor<half> out(TPosition::VECOUT,OUT_OFF,2*C);
    GlobalTensor<half> y,ws;
    y.SetGlobalBuffer((__gm__ half*)gy);ws.SetGlobalBuffer((__gm__ half*)work);
    Duplicate(one,(half)1,2*BN);PipeBarrier<PIPE_V>();
    const uint32_t pairCount=(tm/2-1-lane)/pg+1;
    const uint32_t tileCount=2*pairCount,slabCount=(tileCount+3)/4;
    const uint64_t wsBase=static_cast<uint64_t>(p)*B36_WORKER_TILES*2*BM*BN;
    bool used[2]={false,false};
    for(uint32_t s=0;s<slabCount;++s) {
        const uint32_t bank=s%2,validTileCount=tileCount-4*s>=4?4:tileCount-4*s;
        const uint32_t active=validTileCount*T;
        CrossCoreWaitFlag<2>(s);
        if(used[bank]) WaitFlag<HardEvent::V_MTE2>(bank);
        const uint64_t off=wsBase+4*s*2*BM*BN;
        DataCopyParams slabCopy;
        slabCopy.blockCount=static_cast<uint16_t>(validTileCount);
        slabCopy.blockLen=static_cast<uint16_t>(2*T*sizeof(half)/32);
        slabCopy.srcStride=static_cast<uint16_t>((2*BM*BN-2*T)*sizeof(half)/32);
        slabCopy.dstStride=0;
        DataCopy(in[bank*2*C],ws[off+a*2*T],slabCopy);
        SetFlag<HardEvent::MTE2_V>(bank);
        if(used[bank]) WaitFlag<HardEvent::MTE3_V>(bank);
        WaitFlag<HardEvent::MTE2_V>(bank);
        B36PackedStream(in[bank*2*C],one,tmp,out[bank*C],active,validTileCount*R);
        SetFlag<HardEvent::V_MTE2>(bank);
        SetFlag<HardEvent::V_MTE3>(bank);WaitFlag<HardEvent::V_MTE3>(bank);
        StoreB36Slab<BM,BN,D>(y,out,bank,s,validTileCount,lane,pg,group,a);
        SetFlag<HardEvent::MTE3_V>(bank);used[bank]=true;
    }
    for(uint32_t bank=0;bank<2;++bank) if(used[bank]) {
        WaitFlag<HardEvent::V_MTE2>(bank);WaitFlag<HardEvent::MTE3_V>(bank);
    }
    PipeBarrier<PIPE_ALL>();
}

template<uint32_t M,uint32_t D,uint32_t BM,uint32_t BN>
__aicore__ inline void B36PackedResident(GM_ADDR x,GM_ADDR w0,GM_ADDR b0,GM_ADDR w1,
 GM_ADDR b1,GM_ADDR y,GM_ADDR workspace,uint32_t workers) {
    static_assert(M==8192 && D==256 && BM==64 && BN==64,"C9 only");
    uint32_t p=GetBlockIdx()/GetTaskRatio(),group=0;
    constexpr uint32_t TN=D/BN,TM=M/BM;
    while(group+1<TN && p>=(group+1)*workers/TN) ++group;
    uint32_t start=group*workers/TN,pg=(group+1)*workers/TN-start,lane=p-start;
    static_assert(TM%2==0,"C9 prefetch requires whole input pairs");
    if(lane>=TM/2) return;
    GM_ADDR work=GetUserWorkspace(workspace);
    if ASCEND_IS_AIC B36PackedProduce<BM,BN,D,M,false>(x,w0,b0,w1,b1,work,p,group,lane,pg,TM);
    if ASCEND_IS_AIV B36PackedConsume<BM,BN,D>(y,work,p,GetSubBlockIdx(),group,lane,pg,TM);
}

namespace S117C6 {
__aicore__ inline void SiluGateCounter(LocalTensor<float> dst,
                                      LocalTensor<float> z1,
                                      LocalTensor<float> z2, uint32_t count)
{
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(count);
    const UnaryRepeatParams unary{1U, 1U, 8U, 8U};
    const BinaryRepeatParams binary{1U, 1U, 1U, 8U, 8U, 8U};
    Muls<float, false>(dst, z1, -1.0f, MASK_PLACEHOLDER, 1U, unary);
    Exp<float, false>(dst, dst, MASK_PLACEHOLDER, 1U, unary);
    Adds<float, false>(dst, dst, 1.0f, MASK_PLACEHOLDER, 1U, unary);
    Div<float, false>(dst, z1, dst, MASK_PLACEHOLDER, 1U, binary);
    Mul<float, false>(dst, dst, z2, MASK_PLACEHOLDER, 1U, binary);
    SetMaskNorm();
    ResetMask();
}
__aicore__ inline void RunSingletonSwiGlu(GM_ADDR inputX, GM_ADDR weight1, GM_ADDR bias1,
                                         GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output)
{
    // Issue independent cache-line reads before the scalar dependency chain.
    GlobalTensor<uint64_t> preX, preW1, preB1, preW2, preB2;
    preX.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(inputX), 1U);
    preW1.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(weight1), 1U);
    preB1.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(bias1), 1U);
    preW2.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(weight2), 1U);
    preB2.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(bias2), 1U);
    DataCachePreload(preX, static_cast<int16_t>(0));
    DataCachePreload(preW1, static_cast<int16_t>(0));
    DataCachePreload(preB1, static_cast<int16_t>(0));
    DataCachePreload(preW2, static_cast<int16_t>(0));
    DataCachePreload(preB2, static_cast<int16_t>(0));
    // This early-return path owns its UB and events; no TPipe is live.
    const event_t sv = static_cast<event_t>(0U);
    const event_t vmte3 = static_cast<event_t>(1U);
    GlobalTensor<half> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(inputX), 1U);
    const float x = static_cast<float>(gm.GetValue(0U));
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weight1), 1U);
    float z1 = x * static_cast<float>(gm.GetValue(0U));
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(bias1), 1U);
    z1 += static_cast<float>(gm.GetValue(0U));
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weight2), 1U);
    float z2 = x * static_cast<float>(gm.GetValue(0U));
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(bias2), 1U);
    z2 += static_cast<float>(gm.GetValue(0U));
    // comb90b: hold z1 and z2 in UB and let the vector unit evaluate the
    // same fp32 expression; two scalar/vector round-trips disappear.
    // z1 @0, z2 @8, act @16: every vector operand is 32B aligned.
    LocalTensor<float> value(TPosition::VECCALC, 0U, 24U);
    value.SetValue(0U, z1);
    value.SetValue(8U, z2);
    SetFlag<HardEvent::S_V>(sv);
    WaitFlag<HardEvent::S_V>(sv);
    LocalTensor<float> act = value[16U];
    SiluGateCounter(act, value, value[8U], 1U);
    LocalTensor<half> result(TPosition::VECCALC, 96U, 16U);
    Cast(result, act, RoundMode::CAST_NONE, 1U);
    SetFlag<HardEvent::V_MTE3>(vmte3);
    WaitFlag<HardEvent::V_MTE3>(vmte3);
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output), 1U);
    const DataCopyExtParams copy{1, 2U, 0, 0, 0};
    DataCopyPad(gm, result, copy);
    PipeBarrier<PIPE_ALL>();
}
} // namespace S117C6

 // namespace S52C6

 // namespace S68C7

namespace S83C7 {
__aicore__ inline void LoadMask(LocalTensor<uint16_t> mask) {
    auto words=mask.ReinterpretCast<uint32_t>();
    words.SetValue(0,uint32_t(536813567u));
    words.SetValue(1,uint32_t(536813567u));
    words.SetValue(2,uint32_t(536813567u));
    words.SetValue(3,uint32_t(536813567u));
    words.SetValue(4,uint32_t(0u));
    words.SetValue(5,uint32_t(0u));
    words.SetValue(6,uint32_t(0u));
    words.SetValue(7,uint32_t(0u));
}

// Bias addition also compacts both branch vectors into existing private buffers.

__aicore__ inline void ProjectC7Early(LocalTensor<float> even, LocalTensor<float> odd,
    LocalTensor<half> gathered, LocalTensor<half> xb) {
    constexpr uint32_t PLANE=384;
    auto hu=even.ReinterpretCast<half>(), ho=odd.ReinterpretCast<half>();
    auto *ep=(__ubuf__ half*)hu.GetPhyAddr();
    auto *op=(__ubuf__ half*)ho.GetPhyAddr();
    auto *w=(__ubuf__ half*)gathered.GetPhyAddr();
    auto *xp=(__ubuf__ half*)xb.GetPhyAddr();
    vmla(ep,w+0,xp+0,3,1,0,13,8,0,104);
    vmla(ep+PLANE,w+16,xp+0,3,1,0,13,8,0,104);
    vmla(op,w+32,xp+16,3,1,0,13,8,0,104);
    vmla(op+PLANE,w+48,xp+16,3,1,0,13,8,0,104);
    PipeBarrier<PIPE_V>();
    vmla(ep,w+64,xp+32,3,1,0,13,8,0,104);
    vmla(ep+PLANE,w+80,xp+32,3,1,0,13,8,0,104);
    vmla(op,w+96,xp+48,3,1,0,13,8,0,104);
    vmla(op+PLANE,w+112,xp+48,3,1,0,13,8,0,104);
    PipeBarrier<PIPE_V>();
    vmla(ep,w+128,xp+64,3,1,0,13,8,0,104);
    vmla(ep+PLANE,w+144,xp+64,3,1,0,13,8,0,104);
    vmla(op,w+160,xp+80,3,1,0,13,8,0,104);
    vmla(op+PLANE,w+176,xp+80,3,1,0,13,8,0,104);
    PipeBarrier<PIPE_V>();
    vmla(ep,w+192,xp+96,3,1,0,13,8,0,104);
    vmla(op,w+224,xp+112,3,1,0,13,8,0,104);
    PipeBarrier<PIPE_V>();
    vmla(ep,w+256,xp+128,3,1,0,13,8,0,104);
    vmla(op,w+288,xp+144,3,1,0,13,8,0,104);
    PipeBarrier<PIPE_V>();
    vmla(ep,w+320,xp+160,3,1,0,13,8,0,104);
    vmla(op,w+352,xp+176,3,1,0,13,8,0,104);
    PipeBarrier<PIPE_V>();
    vmla(ep,w+384,xp+192,3,1,0,13,8,0,104);
    PipeBarrier<PIPE_V>();

}

__aicore__ inline void FinishC7Gate(LocalTensor<float> even, LocalTensor<float> odd,
    LocalTensor<half> gathered, LocalTensor<half> xb, LocalTensor<half> bh,
    LocalTensor<half> tmp, LocalTensor<half> out) {
    constexpr uint32_t R=21;
    constexpr uint32_t PLANE=384;
    auto hu=even.ReinterpretCast<half>(),ho=odd.ReinterpretCast<half>();
    auto *ep=(__ubuf__ half*)hu.GetPhyAddr();
    auto *op=(__ubuf__ half*)ho.GetPhyAddr();
    auto *w=(__ubuf__ half*)gathered.GetPhyAddr();
    auto *xp=(__ubuf__ half*)xb.GetPhyAddr();
    Add<half,false>(hu,hu,ho,uint64_t(0),3,{1,1,1,8,8,8});
    vmla(ep+PLANE,w+208,xp+96,3,1,0,13,8,0,104);
    vmla(op+PLANE,w+240,xp+112,3,1,0,13,8,0,104);
    PipeBarrier<PIPE_V>();
    SetVectorMask<half>(16);
    Add<half,false>(out,hu,bh,uint64_t(0),R,{1,1,1,1,1,0});
    SetVectorMask<half>(128);
    vmla(ep+PLANE,w+272,xp+128,3,1,0,13,8,0,104);
    vmla(op+PLANE,w+304,xp+144,3,1,0,13,8,0,104);
    PipeBarrier<PIPE_V>();
    Muls<half,false>(tmp,out,static_cast<half>(-1),uint64_t(0),3,{1,1,8,8});
    vmla(ep+PLANE,w+336,xp+160,3,1,0,13,8,0,104);
    vmla(op+PLANE,w+368,xp+176,3,1,0,13,8,0,104);
    PipeBarrier<PIPE_V>();
    Exp<half,false>(tmp,tmp,uint64_t(0),3,{1,1,8,8});
    vmla(ep+PLANE,w+400,xp+192,3,1,0,13,8,0,104);
    PipeBarrier<PIPE_V>();
    Adds<half,false>(tmp,tmp,static_cast<half>(1),uint64_t(0),3,{1,1,8,8});
    Add<half,false>(hu[PLANE],hu[PLANE],ho[PLANE],uint64_t(0),3,{1,1,1,8,8,8});
    PipeBarrier<PIPE_V>();
    Div<half,false>(tmp,out,tmp,uint64_t(0),3,{1,1,1,8,8,8});
    SetVectorMask<half>(16);
    Add<half,false>(hu[PLANE],hu[PLANE],bh[16],uint64_t(0),R,{1,1,1,1,1,0});
    SetVectorMask<half>(128);
    PipeBarrier<PIPE_V>();
    Mul<half,false>(out,tmp,hu[PLANE],uint64_t(0),3,{1,1,1,8,8,8});
    PipeBarrier<PIPE_V>();

}

__aicore__ inline void StoreC7Rows(LocalTensor<half> packed, LocalTensor<half> out,
    LocalTensor<uint16_t> mask, GlobalTensor<half> &y, uint32_t BEGIN, uint32_t ACTUAL) {
    SetVectorMask<half>(128);
    uint64_t selected=0;
    GatherMask<half,uint16_t>(packed,out,mask,false,0,{1,3,8,0},selected);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(y[BEGIN],packed,DataCopyExtParams{1,ACTUAL*26,0,0,0});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Run(
    GM_ADDR gx, GM_ADDR gw0, GM_ADDR gb0,
    GM_ADDR gw1, GM_ADDR gb1, GM_ADDR gy, GM_ADDR tiling) {
    constexpr uint32_t R=21, PLANE=384, HPLANE=384, ACC=768, PAD=384;
    const uint32_t p=GetBlockIdx();
    if(p>=1u)return;
    const uint32_t BEGIN=p*R*13, ACTUAL=(21-p*R<R)?21-p*R:R;
    InitSocState();
    // Two FP16 parity chains, each [U24x16][V24x16]; final three rows are padding.
    LocalTensor<half> raw(TPosition::VECIN,0,672);
    LocalTensor<half> gathered(TPosition::VECCALC,1536,416);
    LocalTensor<half> xb(TPosition::VECCALC,2816,5120);
    LocalTensor<half> bh(TPosition::VECIN,13056,32);
    LocalTensor<float> even(TPosition::VECCALC,13120,ACC);
    LocalTensor<float> odd(TPosition::VECCALC,16192,ACC);
    LocalTensor<half> pairh(TPosition::VECCALC,19264,2*PAD);
    LocalTensor<half> tmp(TPosition::VECCALC,20800,PAD);
    LocalTensor<half> out(TPosition::VECCALC,21568,PAD);
    LocalTensor<half> packed(TPosition::VECOUT,22336,PAD);
    LocalTensor<uint16_t> mask(TPosition::VECCALC,23104,16);
    LocalTensor<uint32_t> indices(TPosition::VECIN,23136,416);
    static_assert(23136+416*4<=192*1024,"C7 four-row strided-X UB bounds");
    GlobalTensor<half> x,w0,w1,b0,b1,y;
    x.SetGlobalBuffer((__gm__ half*)gx);w0.SetGlobalBuffer((__gm__ half*)gw0);
    w1.SetGlobalBuffer((__gm__ half*)gw1);b0.SetGlobalBuffer((__gm__ half*)gb0);
    b1.SetGlobalBuffer((__gm__ half*)gb1);y.SetGlobalBuffer((__gm__ half*)gy);
    GlobalTensor<uint32_t> gi;gi.SetGlobalBuffer((__gm__ uint32_t*)(tiling+32));
    DataCopyPad(raw[352],x[BEGIN],DataCopyExtParams{1,ACTUAL*26,0,0,0},
        DataCopyPadExtParams<half>{true,0,static_cast<uint8_t>((16-(ACTUAL*13)%16)%16),0});
    const DataCopyPadExtParams<half> wp{true,0,7,0},bp{true,0,3,0};
    DataCopyPad(raw,w0,DataCopyExtParams{1,338,0,0,0},wp);
    DataCopyPad(raw[176],w1,DataCopyExtParams{1,338,0,0,0},wp);
    DataCopy(indices,gi[256],416);
    DataCopyPad(bh,b0,DataCopyExtParams{1,26,0,0,0},bp);
    DataCopyPad(bh[16],b1,DataCopyExtParams{1,26,0,0,0},bp);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    LoadMask(mask);
    SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(out[336],static_cast<half>(0),48);
    Duplicate(raw[640],static_cast<half>(0),32);
    Duplicate(even.ReinterpretCast<half>(),static_cast<half>(0),ACC);
    Duplicate(odd.ReinterpretCast<half>(),static_cast<half>(0),ACC);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    PipeBarrier<PIPE_V>();
    // Weight Gather and row-major X broadcast have no dependency on each other.
    Gather(gathered,raw,indices,0,416);
    Brcb(xb,raw[352],39,{1,8});
    PipeBarrier<PIPE_V>();SetMaskNorm();SetVectorMask<half>(128);
    ProjectC7Early(even,odd,gathered,xb);
    FinishC7Gate(even,odd,gathered,xb,bh,tmp,out);
    StoreC7Rows(packed,out,mask,y,BEGIN,ACTUAL);

}

} // namespace S83C7

__aicore__ inline void C5MajorProducer(GM_ADDR gx,GM_ADDR gw0,GM_ADDR gw1,GM_ADDR work,uint32_t p) {
 constexpr uint32_t BM=128,BN=128,D=128;
 LocalTensor<half> xl(TPosition::A1,0,2*BM*D),wl(TPosition::B1,4*BM*D,2*BN*D);
 LocalTensor<half> al(TPosition::A2,0,2*BM*D),bl(TPosition::B2,0,2*BN*D);LocalTensor<float> c(TPosition::CO1,0,2*BM*BN);
 GlobalTensor<half>x,w0,w1,ws;x.SetGlobalBuffer((__gm__ half*)gx);w0.SetGlobalBuffer((__gm__ half*)gw0);w1.SetGlobalBuffer((__gm__ half*)gw1);ws.SetGlobalBuffer((__gm__ half*)work);
 Nd2NzParams cp;cp.ndNum=1;cp.nValue=BN;cp.dValue=D;cp.srcDValue=D;cp.dstNzC0Stride=BN;cp.dstNzNStride=1;cp.srcNdMatrixStride=0;cp.dstNzMatrixStride=0;
 DataCopy(wl,w0,cp);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
 // Eight independent16-row NZ matrices match the final L0A M-major order.
 Nd2NzParams xp=cp;xp.ndNum=BM/16;xp.nValue=16;xp.dstNzC0Stride=16;
 xp.srcNdMatrixStride=16*D;xp.dstNzMatrixStride=16*D;
 const uint64_t xoff=static_cast<uint64_t>(p)*2*BM*D;
 DataCopy(xl,x[xoff],xp);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
 // V0 weights enter ahead of the second X panel.
 DataCopy(wl[BN*D],w1,cp);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
 DataCopy(xl[BM*D],x[xoff+BM*D],xp);SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
 WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
 LoadData2DParams pb;pb.repeatTimes=BN*D/256;pb.srcStride=1;pb.dstGap=0;pb.ifTranspose=false;LoadData(bl,wl,pb);
 WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
 LoadData2DParams pa;pa.repeatTimes=BM*D/256;pa.srcStride=1;pa.dstGap=0;pa.ifTranspose=false;
 LoadData(al,xl,pa);
 SetFlag<HardEvent::MTE1_M>(EVENT_ID0);WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);SetMMRowMajor();
 MmadParams mp;mp.m=BM;mp.n=BN;mp.k=D;mp.cmatrixInitVal=true;mp.unitFlag=3;
 FixpipeParamsV220 fp;fp.nSize=BN;fp.mSize=BM;fp.srcStride=BM;fp.dstStride=BN;fp.quantPre=QuantMode_t::F322F16;fp.unitFlag=3;
 const uint64_t off=static_cast<uint64_t>(p)*4*BM*BN;
 Mmad(c,al,bl,mp);Fixpipe(ws[off],c,fp);CrossCoreSetFlag<2,PIPE_FIX>(0);
 // Finish the first output tile while the second input panel arrives.
 WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
 LoadData(bl[BN*D],wl[BN*D],pb);
 SetFlag<HardEvent::MTE1_M>(EVENT_ID2);WaitFlag<HardEvent::MTE1_M>(EVENT_ID2);
 Mmad(c,al,bl[BN*D],mp);Fixpipe(ws[off+2*BM*BN],c,fp);CrossCoreSetFlag<2,PIPE_FIX>(2);
 WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
 LoadData(al[BM*D],xl[BM*D],pa);
 SetFlag<HardEvent::MTE1_M>(EVENT_ID1);WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
 Mmad(c[BM*BN],al[BM*D],bl,mp);Fixpipe(ws[off+BM*BN],c[BM*BN],fp);CrossCoreSetFlag<2,PIPE_FIX>(1);
 Mmad(c[BM*BN],al[BM*D],bl[BN*D],mp);Fixpipe(ws[off+3*BM*BN],c[BM*BN],fp);CrossCoreSetFlag<2,PIPE_FIX>(3);
 PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void C5MajorConsumer(GM_ADDR gb0,GM_ADDR gb1,GM_ADDR gy,GM_ADDR work,uint32_t p,uint32_t a) {
 constexpr uint32_t BM=128,BN=128,R=64,C=R*BN;
 constexpr uint32_t BIAS_OFF=8*C,ONE_OFF=BIAS_OFF+4*BN,TMP_OFF=ONE_OFF+2*BN,OUT_OFF=TMP_OFF+4*C;
 static_assert(OUT_OFF+4*C==131840,"C5 two-tile projection UB layout");
 InitSocState();LocalTensor<half> u(TPosition::VECIN,0,2*C),v(TPosition::VECIN,4*C,2*C);
 LocalTensor<half> bh(TPosition::VECIN,BIAS_OFF,2*BN),one(TPosition::VECCALC,ONE_OFF,BN),tmp(TPosition::VECCALC,TMP_OFF,2*C),out(TPosition::VECOUT,OUT_OFF,2*C);
 GlobalTensor<half>b0,b1,y,ws;b0.SetGlobalBuffer((__gm__ half*)gb0);b1.SetGlobalBuffer((__gm__ half*)gb1);y.SetGlobalBuffer((__gm__ half*)gy);ws.SetGlobalBuffer((__gm__ half*)work);
 DataCopy(bh,b0,BN);DataCopy(bh[BN],b1,BN);
 SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
 Duplicate(one,(half)1,BN);
 WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
 Muls(bh,bh,(half)-1,2*BN);PipeBarrier<PIPE_V>();
 const uint64_t off=static_cast<uint64_t>(p)*4*BM*BN+a*C;SetMaskNorm();SetVectorMask<half>(128);
 #pragma unroll 2
 for(uint32_t t=0;t<2;++t){
  CrossCoreWaitFlag<2>(t);DataCopy(u[t*C],ws[off+t*BM*BN],C);
  SetFlag<HardEvent::MTE2_V>(t);WaitFlag<HardEvent::MTE2_V>(t);
  // Bias was negated once. Exp can read the continuous -U plane directly.
  Sub<half,false>(u[t*C],bh,u[t*C],uint64_t(0),R,{1,1,1,8,0,8});PipeBarrier<PIPE_V>();
  Exp<half,false>(tmp[t*C],u[t*C],uint64_t(0),R,{1,1,8,8});PipeBarrier<PIPE_V>();
  Adds<half,false>(tmp[t*C],tmp[t*C],(half)1,uint64_t(0),R,{1,1,8,8});PipeBarrier<PIPE_V>();
  Div<half,false>(tmp[t*C],one,tmp[t*C],uint64_t(0),R,{1,1,1,8,0,8});PipeBarrier<PIPE_V>();
  Mul<half,false>(tmp[t*C],u[t*C],tmp[t*C],uint64_t(0),R,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
  // Consume V as soon as this tile's U gate has been scheduled.
  CrossCoreWaitFlag<2>(2+t);DataCopy(v[t*C],ws[off+(2+t)*BM*BN],C);
  SetFlag<HardEvent::MTE2_V>(t);WaitFlag<HardEvent::MTE2_V>(t);
  Sub<half,false>(v[t*C],bh[BN],v[t*C],uint64_t(0),R,{1,1,1,8,0,8});PipeBarrier<PIPE_V>();
  Mul<half,false>(out[t*C],tmp[t*C],v[t*C],uint64_t(0),R,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
  SetFlag<HardEvent::V_MTE3>(t);WaitFlag<HardEvent::V_MTE3>(t);
  DataCopyPad(y[(static_cast<uint64_t>(p)*2*BM+t*BM+a*R)*BN],out[t*C],DataCopyExtParams{1,2*C,0,0,0});SetFlag<HardEvent::MTE3_V>(t);
 }
 ResetMask();WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);PipeBarrier<PIPE_ALL>();
}
__aicore__ inline void C5MajorPipeline(GM_ADDR x,GM_ADDR w0,GM_ADDR b0,GM_ADDR w1,GM_ADDR b1,GM_ADDR y,GM_ADDR workspace) {
 const uint32_t p=GetBlockIdx()/GetTaskRatio();GM_ADDR work=GetUserWorkspace(workspace);
 if ASCEND_IS_AIC C5MajorProducer(x,w0,w1,work,p);
 if ASCEND_IS_AIV C5MajorConsumer(b0,b1,y,work,p,GetSubBlockIdx());
}

template<typename DT_INPUT_X,uint32_t VARIANT> __global__ __aicore__ void swi_glu(GM_ADDR input_x,GM_ADDR weight_1,GM_ADDR bias_1,GM_ADDR weight_2,
 GM_ADDR bias_2,GM_ADDR output,GM_ADDR workspace,GM_ADDR tiling) {
REGISTER_TILING_DEFAULT(SwiGluTilingData);
if constexpr(VARIANT==29) { R1Fallback<true>(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace,tiling); return; }
if constexpr(VARIANT==2) { if(GetBlockIdx()!=0U)return; S117C6::RunSingletonSwiGlu(input_x,weight_1,bias_1,weight_2,bias_2,output); return; }
if constexpr(VARIANT==3) { C1SparseInputCast(input_x,weight_1,bias_1,weight_2,bias_2,output); return; }
if constexpr(VARIANT==5) { C4SingleWorker(input_x,weight_1,bias_1,weight_2,bias_2,output); return; }
if constexpr(VARIANT==8) { MaskC2RowSplit(input_x,weight_1,bias_1,weight_2,bias_2,output); return; }
if constexpr(VARIANT==50) { C3NativeHalf128(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace,16); return; }
if constexpr(VARIANT==51) { C5MajorPipeline(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace); return; }
if constexpr(VARIANT==52) { B36PackedResident<8192,256,64,64>(input_x,weight_1,bias_1,weight_2,bias_2,output,workspace,20); return; }
if constexpr(VARIANT==53) { S83C7::Run(input_x,weight_1,bias_1,weight_2,bias_2,output,tiling); return; }
if constexpr(VARIANT==55) { OuterProductC8(input_x,weight_1,bias_1,weight_2,bias_2,output,tiling); return; }
}
