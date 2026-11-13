/*
 * Copyright (c) 2026 jnu_operators-selling.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Apache License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWI_GLU_TILING_DATA_H
#define SWI_GLU_TILING_DATA_H
#include <cstdint>

constexpr uint32_t SWIGLU_GATHER_CAPACITY = 256;

struct SwiGluTilingData {
    uint32_t rows;
    uint32_t dim;
    uint32_t tileRows;
    uint32_t tileCols;
    uint32_t cores;
};
using SwiGluVectorTilingData = SwiGluTilingData;
static_assert(sizeof(SwiGluTilingData) == 20);

#endif  // SWI_GLU_TILING_DATA_H

// Host compilation sees only the data layout; device-only vector helpers follow.
#if defined(SWIGLU_DEVICE_HELPERS) && !defined(SWIGLU_DEVICE_HELPERS_DEFINED)
#define SWIGLU_DEVICE_HELPERS_DEFINED
#include "kernel_operator.h"

template <typename T>
__aicore__ inline void BindSwiGluTensors(AscendC::GlobalTensor<T> &x,
    AscendC::GlobalTensor<T> &w1, AscendC::GlobalTensor<T> &w2,
    AscendC::GlobalTensor<T> &b1, AscendC::GlobalTensor<T> &b2,
    AscendC::GlobalTensor<T> &y, GM_ADDR input, GM_ADDR weight1,
    GM_ADDR weight2, GM_ADDR bias1, GM_ADDR bias2, GM_ADDR output) {
    x.SetGlobalBuffer((__gm__ T *)input);
    w1.SetGlobalBuffer((__gm__ T *)weight1);
    w2.SetGlobalBuffer((__gm__ T *)weight2);
    b1.SetGlobalBuffer((__gm__ T *)bias1);
    b2.SetGlobalBuffer((__gm__ T *)bias2);
    y.SetGlobalBuffer((__gm__ T *)output);
}

template <typename T>
__aicore__ inline void CopyTensorPair(AscendC::LocalTensor<T> dst, uint32_t dstStep,
    AscendC::GlobalTensor<T> first, AscendC::GlobalTensor<T> second,
    uint32_t elements, AscendC::DataCopyPadExtParams<T> pad) {
    using namespace AscendC;
    const uint64_t firstAddress = reinterpret_cast<uint64_t>(first.GetPhyAddr());
    const uint64_t secondAddress = reinterpret_cast<uint64_t>(second.GetPhyAddr());
    const uint32_t bytes = elements * sizeof(T);
    const uint32_t rounded = (bytes + 31) / 32 * 32;
    const uint32_t dstBytes = dstStep * sizeof(T);
    const uint64_t distance = secondAddress - firstAddress;
    // A two-block DMA is valid only when both source and destination strides fit.
    if (secondAddress >= firstAddress && distance >= bytes &&
        distance - bytes <= 0xffffffffULL && dstBytes >= rounded &&
        (dstBytes - rounded) % 32 == 0) {
        DataCopyExtParams copy{2, bytes, static_cast<uint32_t>(distance - bytes),
                (dstBytes - rounded) / 32, 0};
        DataCopyPad(dst, first, copy, pad);
    } else {
        DataCopyExtParams copy{1, bytes, 0, 0, 0};
        DataCopyPad(dst, first, copy, pad);
        DataCopyPad(dst[dstStep], second, copy, pad);
    }
}

// Read in increasing GM address order and preserve the logical projection order.
template <typename T>
__aicore__ inline bool CopyOrderedTensorPair(AscendC::LocalTensor<T> dst, uint32_t dstStep,
    AscendC::GlobalTensor<T> first, AscendC::GlobalTensor<T> second,
    uint32_t elements, AscendC::DataCopyPadExtParams<T> pad) {
    using namespace AscendC;
    const uint64_t firstAddress = reinterpret_cast<uint64_t>(first.GetPhyAddr());
    const uint64_t secondAddress = reinterpret_cast<uint64_t>(second.GetPhyAddr());
    const bool reverse = secondAddress < firstAddress;
    const uint64_t distance = reverse ? firstAddress - secondAddress : secondAddress - firstAddress;
    const uint32_t bytes = elements * sizeof(T);
    const uint32_t rounded = (bytes + 31) / 32 * 32;
    const uint32_t dstBytes = dstStep * sizeof(T);
    if (distance >= bytes && distance - bytes <= 0xffffffffULL &&
        dstBytes >= rounded && (dstBytes - rounded) % 32 == 0) {
        DataCopyExtParams copy{2, bytes, static_cast<uint32_t>(distance - bytes),
                (dstBytes - rounded) / 32, 0};
        if (reverse) DataCopyPad(dst, second, copy, pad);
        else DataCopyPad(dst, first, copy, pad);
        return reverse;
    }
    DataCopyExtParams copy{1, bytes, 0, 0, 0};
    DataCopyPad(dst, first, copy, pad);
    DataCopyPad(dst[dstStep], second, copy, pad);
    return false;
}

__aicore__ inline void MakeGatherOffsets(AscendC::LocalTensor<uint32_t> offsets,
    uint32_t dim, uint32_t pitch, uint32_t rows) {
    using namespace AscendC;
    auto indices = offsets.ReinterpretCast<int32_t>();
    for (uint32_t i = 0; i < 8 && i < pitch; ++i) indices.SetValue(i, 2 * i);
    SetFlag<HardEvent::S_V>(5);
    WaitFlag<HardEvent::S_V>(5);
    for (uint32_t n = 8; n < pitch; n *= 2) {
        Adds(indices[n], indices, static_cast<int32_t>(n * 2),
            pitch - n < n ? pitch - n : n);
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t n = 1; n < rows; n *= 2) {
        Adds(indices[n * pitch], indices, static_cast<int32_t>(n * dim * 2),
            (rows - n < n ? rows - n : n) * pitch);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void HalfGateSwiGlu(AscendC::LocalTensor<T> out,
    AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b, uint32_t count) {
    if (count <= 128) {
        AscendC::SetVectorMask<T>(count);
    } else {
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(count);
    }
    AscendC::UnaryRepeatParams unary{1, 1, 8, 8};
    AscendC::BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    AscendC::Muls<T, false>(out, a, static_cast<T>(-1.0f), AscendC::MASK_PLACEHOLDER, 1, unary);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp<T, false>(out, out, AscendC::MASK_PLACEHOLDER, 1, unary);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<T, false>(out, out, static_cast<T>(1.0f), AscendC::MASK_PLACEHOLDER, 1, unary);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div<T, false>(out, a, out, AscendC::MASK_PLACEHOLDER, 1, binary);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<T, false>(out, out, b, AscendC::MASK_PLACEHOLDER, 1, binary);
    if (count > 128) AscendC::SetMaskNorm();
}

// Every row contains two consecutive 128-element projections.
template <typename T>
__aicore__ inline void HalfGateInterleaved128(AscendC::LocalTensor<T> out,
    AscendC::LocalTensor<T> pair, uint8_t rows) {
    using namespace AscendC;
    SetVectorMask<T>(128);
    UnaryRepeatParams fromPair{1, 1, 8, 16};
    UnaryRepeatParams self{1, 1, 8, 8};
    BinaryRepeatParams divide{1, 1, 1, 8, 16, 8};
    BinaryRepeatParams multiply{1, 1, 1, 8, 8, 16};
    Muls<T, false>(out, pair, static_cast<T>(-1.0f), MASK_PLACEHOLDER, rows, fromPair);
    PipeBarrier<PIPE_V>();
    Exp<T, false>(out, out, MASK_PLACEHOLDER, rows, self);
    PipeBarrier<PIPE_V>();
    Adds<T, false>(out, out, static_cast<T>(1.0f), MASK_PLACEHOLDER, rows, self);
    PipeBarrier<PIPE_V>();
    Div<T, false>(out, pair, out, MASK_PLACEHOLDER, rows, divide);
    PipeBarrier<PIPE_V>();
    Mul<T, false>(out, out, pair[128], MASK_PLACEHOLDER, rows, multiply);
}

// The current counter mask is shared by independent gather instructions.
template <typename T>
__aicore__ inline void GatherMaskCurrentMode(AscendC::LocalTensor<T> dst,
    AscendC::LocalTensor<T> src,AscendC::LocalTensor<uint16_t> pattern) {
    AscendC::GatherMask(dst.template ReinterpretCast<uint16_t>(),
        src.template ReinterpretCast<uint16_t>(),pattern,static_cast<uint8_t>(0),
        AscendC::GatherMaskParams{1,1,8,0});
}

// Both long-K vector layouts use the same ordered 64-element partial reductions.
template<uint32_t ROWS, uint32_t COLS, typename State>
__aicore__ inline void ComputeLongKPartials(State &s) {
    using namespace AscendC;
    BinaryRepeatParams mulParams{1, 1, 1, static_cast<uint8_t>(s.pitch / 8),
            static_cast<uint8_t>(s.pitch / 8), 0};
    for (uint32_t rr = 0; rr < s.nr; ++rr) {
        for (uint32_t c = 0; c < s.dim; c += 64) {
            const uint32_t width = s.dim - c < 64 ? s.dim - c : 64;
            Mul(s.products[rr * s.weightCount + c], s.weights[c], s.inputs[rr * s.pitch + c],
                width, 2 * COLS, mulParams);
        }
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t c = 0; c < s.dim; c += 64) {
        const uint32_t width = s.dim - c < 64 ? s.dim - c : 64;
        WholeReduceSum(s.sums[c / 64 * ROWS * 2 * COLS], s.products[c], width,
            s.nr * 2 * COLS, 1, 1, s.pitch / 8);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, uint32_t DIM, bool CURRENT_MASK = false>
class TinyVectorSwiGlu {
public:
        __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight1, GM_ADDR bias1,
        GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output, GM_ADDR tiling) {
        BindSwiGluTensors(x, w1, w2, b1, b2, y, input, weight1, weight2, bias1, bias2, output);
    }

    __aicore__ inline void Process() {
        using namespace AscendC;
        static_assert(DIM == 4);
        const uint32_t row = CURRENT_MASK ? 0 : GetBlockIdx();
        LocalTensor<T> packed(TPosition::VECCALC, 0, 48);
        LocalTensor<T> bias(TPosition::VECCALC, 192, 32);
        LocalTensor<T> pair(TPosition::VECCALC, 288, 32);
        LocalTensor<T> out(TPosition::VECCALC, 384, 16);
        LocalTensor<float> weights(TPosition::VECCALC, 512, 48);
        auto input = weights[32];
        LocalTensor<float> repeated(TPosition::VECCALC, 1344, 32);
        LocalTensor<float> products(TPosition::VECCALC, 1632, 32);
        LocalTensor<float> partial(TPosition::VECCALC, 1824, 32);
        LocalTensor<uint32_t> offsets(TPosition::VECCALC, 2048, 32);
        LocalTensor<float> sums(TPosition::VECCALC, 2304, 64);
        DataCopyPadExtParams<T> noPad{false, 0, 0, 0};
        const bool reverseWeights = CopyOrderedTensorPair(packed, 16, w1, w2, 16, noPad);
        const bool reverseBias = reverseWeights;
        DataCopyExtParams vectorCopy{1, DIM * sizeof(T), 0, 0, 0};
        DataCopyPad(bias[reverseWeights ? 16 : 0], b1, vectorCopy, noPad);
        DataCopyPad(bias[reverseWeights ? 0 : 16], b2, vectorCopy, noPad);
        DataCopyPadExtParams<T> inputPad{true, 0, 12, 0};
        DataCopyPad(packed[32], x[row * DIM], vectorCopy, inputPad);

        ReduceTinyProducts(weights, input, repeated, products, partial, offsets, sums, packed);
        Cast(pair, sums, RoundMode::CAST_RINT, DIM, 2, UnaryRepeatParams{1, 1, 1, 4});
        PipeBarrier<PIPE_V>();
        if (reverseWeights == reverseBias) {
            Add(pair, pair, bias, DIM, 2, BinaryRepeatParams{1, 1, 1, 1, 1, 1});
        } else {
            Add(pair, pair, bias[16], DIM);
            Add(pair[16], pair[16], bias, DIM);
        }
        PipeBarrier<PIPE_V>();
        HalfGateSwiGlu(out, pair[reverseWeights ? 16 : 0], pair[reverseWeights ? 0 : 16], DIM);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(y[row * DIM], out, vectorCopy);
    }

private:
        __aicore__ inline void ReduceTinyProducts(AscendC::LocalTensor<float> weights,
        AscendC::LocalTensor<float> input, AscendC::LocalTensor<float> repeated,
        AscendC::LocalTensor<float> products, AscendC::LocalTensor<float> partial,
        AscendC::LocalTensor<uint32_t> offsets, AscendC::LocalTensor<float> sums,
        AscendC::LocalTensor<T> packed) {
        using namespace AscendC;
        auto pattern = offsets.ReinterpretCast<int32_t>();
        Duplicate(pattern, static_cast<int32_t>(0x0f0f0f0f), 2);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        UnaryRepeatParams widen{1, 1, 8, 4};
        Cast(weights, packed, RoundMode::CAST_NONE, 48, 1, widen);
        PipeBarrier<PIPE_V>();
        if constexpr (CURRENT_MASK) {
            // Duplicate's count-mode initialization restored normal mode;
            // normal-mode vreducev2 uses its pattern for the complete repeat.
            GatherMask(repeated,input,offsets.template ReinterpretCast<float>(),
                static_cast<uint8_t>(0),GatherMaskParams{0,1,0,0});
        } else {
            uint64_t gathered = 0;
            GatherMask(repeated, input, offsets, false, 0,
                GatherMaskParams{0, 1, 0, 0}, gathered);
        }
        PipeBarrier<PIPE_V>();
        Mul(products, weights, repeated, 32, 1, BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        PairReduceSum(partial, products, 1, 32, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        PairReduceSum(sums, partial, 2, 8, 1, 1, 1);
        PipeBarrier<PIPE_V>();
    }
    AscendC::GlobalTensor<T> x, w1, w2, b1, b2, y;
};

// One scalar output needs no row scheduling or large vector tile buffers.
template <typename T, bool COMPACT_FMA = false>
class ScalarVectorSwiGlu {
public:
        __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight1, GM_ADDR bias1,
        GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output) {
        BindSwiGluTensors(x, w1, w2, b1, b2, y, input, weight1, weight2, bias1, bias2, output);
    }
    __aicore__ inline void Process() {
        using namespace AscendC;
        LocalTensor<T> coeff(TPosition::VECCALC, 0, 80);
        LocalTensor<T> input(TPosition::VECCALC, 192, 16);
        LocalTensor<T> pair(TPosition::VECCALC, 256, 32);
        LocalTensor<T> out(TPosition::VECCALC, 352, 16);
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyExtParams one{1, sizeof(T), 0, 0, 0};
        const uint64_t xa = reinterpret_cast<uint64_t>(x.GetPhyAddr());
        const uint64_t wa = reinterpret_cast<uint64_t>(w1.GetPhyAddr());
        const uint64_t ba = reinterpret_cast<uint64_t>(b1.GetPhyAddr());
        const uint64_t wb = reinterpret_cast<uint64_t>(w2.GetPhyAddr());
        const uint64_t bb = reinterpret_cast<uint64_t>(b2.GetPhyAddr());
        const uint64_t delta = wa - xa;
        const uint64_t mismatch = (ba - wa - delta) | (wb - ba - delta) | (bb - wb - delta);
        const bool forward = wa >= xa;
        const uint64_t step = forward ? delta : xa - wa;
        auto first = pair;
        auto second = pair[16];
        SetMaskNorm();
        SetVectorMask<T>(1);
        EvaluateScalarProjection(coeff, input, pair, first, second, forward, mismatch, step);
        UnaryRepeatParams unary{1, 1, 8, 8};
        BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        Muls<T, false>(out, first, static_cast<T>(-1.0f), MASK_PLACEHOLDER, 1, unary);
        PipeBarrier<PIPE_V>();
        Exp<T, false>(out, out, MASK_PLACEHOLDER, 1, unary);
        PipeBarrier<PIPE_V>();
        Adds<T, false>(out, out, static_cast<T>(1.0f), MASK_PLACEHOLDER, 1, unary);
        PipeBarrier<PIPE_V>();
        Div<T, false>(out, first, out, MASK_PLACEHOLDER, 1, binary);
        PipeBarrier<PIPE_V>();
        Mul<T, false>(out, out, second, MASK_PLACEHOLDER, 1, binary);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(y, out, one);
    }
private:
        __aicore__ inline void EvaluateScalarProjection(AscendC::LocalTensor<T> coeff,
        AscendC::LocalTensor<T> input, AscendC::LocalTensor<T> pair,
        AscendC::LocalTensor<T> &first, AscendC::LocalTensor<T> &second,
        bool forward, uint64_t mismatch, uint64_t step) {
        using namespace AscendC;
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyExtParams one{1, sizeof(T), 0, 0, 0};
        if (mismatch == 0 && step >= sizeof(T) && step - sizeof(T) <= 0xffffffffULL) {
            DataCopyExtParams all{5, sizeof(T), static_cast<uint32_t>(step - sizeof(T)), 0, 0};
            if (forward) DataCopyPad(coeff, x, all, pad);
            else DataCopyPad(coeff, b2, all, pad);
            if constexpr (COMPACT_FMA) {
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                MulAddDst<T,T,false>(coeff[forward?32:0],coeff[16],coeff[forward?0:64],
                    MASK_PLACEHOLDER,2,BinaryRepeatParams{1,1,1,2,2,0});
            } else {
                SetFlag<HardEvent::MTE2_S>(0);
                WaitFlag<HardEvent::MTE2_S>(0);
                const T scalarX = coeff.GetValue(forward ? 0 : 64);
                Axpy<T, T, false>(coeff[forward ? 32 : 0], coeff[16], scalarX, MASK_PLACEHOLDER, 2,
                    UnaryRepeatParams{1, 1, 2, 2});
            }
            PipeBarrier<PIPE_V>();
            first = coeff[32];
            second = coeff[forward ? 64 : 0];
        } else {
            DataCopyPad(input, x, one, pad);
            DataCopyPad(coeff, w1, one, pad);
            DataCopyPad(coeff[16], w2, one, pad);
            DataCopyPad(pair, b1, one, pad);
            DataCopyPad(pair[16], b2, one, pad);
            if constexpr (COMPACT_FMA) {
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                MulAddDst<T,T,false>(pair,coeff,input,MASK_PLACEHOLDER,2,
                    BinaryRepeatParams{1,1,1,1,1,0});
            } else {
                SetFlag<HardEvent::MTE2_S>(0);
                WaitFlag<HardEvent::MTE2_S>(0);
                const T scalarX = input.GetValue(0);
                Axpy<T, T, false>(pair, coeff, scalarX, MASK_PLACEHOLDER, 2,
                    UnaryRepeatParams{1, 1, 1, 1});
            }
            PipeBarrier<PIPE_V>();
        }
    }
    AscendC::GlobalTensor<T> x, w1, w2, b1, b2, y;
};

template <typename T, bool PADDED, uint32_t STATIC_DIM = 0>
class PairVectorSwiGlu {
public:
        __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight1, GM_ADDR bias1,
        GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output,
        const SwiGluVectorTilingData &data) {
        t = data;
        BindSwiGluTensors(x, w1, w2, b1, b2, y, input, weight1, weight2, bias1, bias2, output);
    }

    __aicore__ inline void Process() {
        using namespace AscendC;
        State s(t);
        for (uint32_t tile = GetBlockIdx(); tile < s.tiles; tile += t.cores) {
            if (!s.first) PipeBarrier<PIPE_ALL>();
            s.first = false;
            s.row = tile / s.nTiles * ROWS;
            s.col = tile % s.nTiles * COLS;
            s.nr = t.rows - s.row < ROWS ? t.rows - s.row : ROWS;
            s.cols = s.dim - s.col < COLS ? s.dim - s.col : COLS;
            s.reload = s.cachedCol != s.col;
            LoadTile(s);
            ComputeTile(s);
            StoreTile(s);
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
        static constexpr uint32_t COLS = 16, ROWS = 2;
    struct State {
        const SwiGluVectorTilingData &t;
        const uint32_t dim = STATIC_DIM == 0 ? t.dim : STATIC_DIM;
        const uint32_t pitch = (dim + 15) / 16 * 16;
        const uint32_t parts = (dim + 63) / 64;
        const uint32_t nTiles = (dim + COLS - 1) / COLS;
        const uint32_t tiles = ((t.rows + 1) / 2) * nTiles;
        const uint32_t weightCount = 2 * COLS * pitch;
        AscendC::LocalTensor<T> input{AscendC::TPosition::VECCALC, 0, (2 * COLS + ROWS) * pitch};
        AscendC::LocalTensor<T> bias{AscendC::TPosition::VECCALC, 18432, 2 * COLS};
        AscendC::LocalTensor<float> weights{AscendC::TPosition::VECCALC, 32768, weightCount};
        AscendC::LocalTensor<float> products{AscendC::TPosition::VECCALC, 65536, ROWS * weightCount};
        AscendC::LocalTensor<float> inputs{AscendC::TPosition::VECCALC, 131072, ROWS * pitch};
        AscendC::LocalTensor<float> sums{AscendC::TPosition::VECCALC, 135168, 4 * ROWS * 2 * COLS};
        AscendC::LocalTensor<T> a{AscendC::TPosition::VECCALC, 139264, ROWS * COLS};
        AscendC::LocalTensor<T> b{AscendC::TPosition::VECCALC, 139328, ROWS * COLS};
        AscendC::LocalTensor<T> out{AscendC::TPosition::VECCALC, 139392, ROWS * COLS};
        uint32_t cachedCol = UINT32_MAX;
        bool first = true;
        uint32_t row = 0, col = 0, nr = 0, cols = 0;
        bool reload = false;
        __aicore__ explicit State(const SwiGluVectorTilingData &data) : t(data) {}
    };

    __aicore__ inline void LoadTile(State &s) {
        using namespace AscendC;
        if (s.reload) {
            if constexpr (PADDED) {
                DataCopyExtParams matrixCopy{static_cast<uint16_t>(s.cols), s.dim * 2, 0, 0, 0};
                DataCopyPadExtParams<T> matrixPad{true, 0,
                        static_cast<uint8_t>(s.pitch - s.dim), 0};
                DataCopyPad(s.input, w1[static_cast<uint64_t>(s.col) * s.dim], matrixCopy, matrixPad);
                DataCopyPad(s.input[COLS * s.pitch], w2[static_cast<uint64_t>(s.col) * s.dim],
                    matrixCopy, matrixPad);
            } else {
                DataCopy(s.input, w1[static_cast<uint64_t>(s.col) * s.dim], s.cols * s.dim);
                DataCopy(s.input[COLS * s.pitch], w2[static_cast<uint64_t>(s.col) * s.dim],
                    s.cols * s.dim);
            }
            DataCopyExtParams biasCopy{1, s.cols * 2, 0, 0, 0};
            DataCopyPadExtParams<T> biasPad{true, 0, static_cast<uint8_t>(COLS - s.cols), 0};
            DataCopyPad(s.bias, b1[s.col], biasCopy, biasPad);
            DataCopyPad(s.bias[COLS], b2[s.col], biasCopy, biasPad);
        }
        if constexpr (PADDED) {
            DataCopyExtParams xCopy{static_cast<uint16_t>(s.nr), s.dim * 2, 0, 0, 0};
            DataCopyPadExtParams<T> xPad{true, 0, static_cast<uint8_t>(s.pitch - s.dim), 0};
            DataCopyPad(s.input[s.weightCount], x[static_cast<uint64_t>(s.row) * s.dim], xCopy, xPad);
        } else {
            DataCopy(s.input[s.weightCount], x[static_cast<uint64_t>(s.row) * s.dim], s.nr * s.dim);
        }
    }

    __aicore__ inline void ComputeTile(State &s) {
        using namespace AscendC;
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        if (s.reload) {
            Cast(s.weights, s.input, RoundMode::CAST_NONE, s.weightCount);
            s.cachedCol = s.col;
        }
        Cast(s.inputs, s.input[s.weightCount], RoundMode::CAST_NONE, s.nr * s.pitch);
        PipeBarrier<PIPE_V>();
        ComputeLongKPartials<ROWS, COLS>(s);
        for (uint32_t part = 1; part < s.parts; ++part) {
            Add(s.sums, s.sums, s.sums[part * ROWS * 2 * COLS], s.nr * 2 * COLS);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreTile(State &s) {
        using namespace AscendC;
        UnaryRepeatParams castParams{1, 1, COLS / 16, 2 * COLS / 8};
        Cast(s.a, s.sums, RoundMode::CAST_RINT, COLS, s.nr, castParams);
        Cast(s.b, s.sums[COLS], RoundMode::CAST_RINT, COLS, s.nr, castParams);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams biasParams{1, 1, 1, COLS / 16, COLS / 16, 0};
        Add(s.a, s.a, s.bias, COLS, s.nr, biasParams);
        Add(s.b, s.b, s.bias[COLS], COLS, s.nr, biasParams);
        PipeBarrier<PIPE_V>();
        HalfGateSwiGlu(s.out, s.a, s.b, s.nr * COLS);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyExtParams outCopy{static_cast<uint16_t>(s.nr), s.cols * 2, 0,
                (s.dim - s.cols) * 2, 0};
        DataCopyPad(y[static_cast<uint64_t>(s.row) * s.dim + s.col], s.out, outCopy);
    }


    SwiGluVectorTilingData t;
    AscendC::GlobalTensor<T> x, w1, w2, b1, b2, y;
};

template<typename T>
__aicore__ inline void PrepareSevenRowOffsets(AscendC::LocalTensor<uint32_t> offsets) {
    using namespace AscendC;
    constexpr uint32_t D=13,P=16;
    // Two uint32 byte offsets per scalar uint64 store initialize one full
    // sixteen-lane row, removing its dependent vector Adds and barrier.
    auto wideOffsets=offsets.template ReinterpretCast<uint64_t>();
    for(uint32_t i=0;i<8;++i)
    wideOffsets.SetValue(i,static_cast<uint64_t>(4*i) |
        (static_cast<uint64_t>(4*i+2)<<32));
    SetFlag<HardEvent::S_V>(5);WaitFlag<HardEvent::S_V>(5);
    auto indices=offsets.template ReinterpretCast<int32_t>();
    for(uint32_t n=1;n<D;n*=2) {
        Adds(indices[n*P],indices,static_cast<int32_t>(n*D*2),
            (D-n<n?D-n:n)*P);
        PipeBarrier<PIPE_V>();
    }
}

template<typename T>
__aicore__ inline void ReduceSevenRows(AscendC::LocalTensor<T> input,
    AscendC::LocalTensor<float> weights, AscendC::LocalTensor<float> expanded,
    AscendC::LocalTensor<float> products, AscendC::LocalTensor<float> blockSums,
    AscendC::LocalTensor<float> sums) {
    using namespace AscendC;
    constexpr uint32_t R=7,P=16,WC=512;
    Cast(weights,input,RoundMode::CAST_NONE,WC);
    // Four row-strided casts replace seven per-row broadcast descriptions.
    for(uint32_t part=0;part<4;++part)
    Cast(expanded[part*P],input[WC],RoundMode::CAST_NONE,P,R,UnaryRepeatParams{1,1,8,1});
    PipeBarrier<PIPE_V>();
    for(uint32_t rr=0;rr<R;++rr)
    Mul(products[rr*WC],weights,expanded[rr*64],64,8,BinaryRepeatParams{1,1,1,8,8,0});
    PipeBarrier<PIPE_V>();
    uint64_t valid[2]={0x1fff1fff1fff1fffULL,0};
    BlockReduceSum(blockSums,products,R*8,valid,1,1,8);
    PipeBarrier<PIPE_V>();PairReduceSum(sums,blockSums,R,64,1,1,8);
    PipeBarrier<PIPE_V>();
}

#endif  // SWIGLU_DEVICE_HELPERS
