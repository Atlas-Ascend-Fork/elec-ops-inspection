/*
 * Copyright (c) 2026 jnu_operators-selling.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Apache License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
// This kernel uses direct Cube instructions and explicit cross-core flags.
// The compiler's KFC message-pool initialization is unnecessary here.
#ifdef __MIX_CORE_MACRO__
#undef __MIX_CORE_MACRO__
#endif
#define SWIGLU_DEVICE_HELPERS
#include "swi_glu_tiling.h"
template <typename T, uint32_t MODE, bool ONE_TILE = false, uint32_t STATIC_DIM = 0, bool SMALL_FMA = false>
class VectorSwiGlu {
public:
        __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight1, GM_ADDR bias1,
        GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output,
        const SwiGluVectorTilingData &data, GM_ADDR tiling) {
        t = data;
        BindSwiGluTensors(x, w1, w2, b1, b2, y, input, weight1, weight2, bias1, bias2, output);
    }
    __aicore__ inline void Process() {
        if constexpr (MODE == 1) {
            if (ProcessScalar()) return;
        } else {
            ProcessProjection();
        }
        if constexpr (!ONE_TILE) AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
        static constexpr bool REVERSE_WEIGHTS = false, REVERSE_BIAS = false;
    struct State {
        const SwiGluVectorTilingData &t;
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        const uint32_t dim = STATIC_DIM == 0 ? t.dim : STATIC_DIM;
        const uint32_t capacity = STATIC_DIM == 32 ? 32 : (STATIC_DIM == 0 ? t.tileCols : 16);
        const uint32_t pitch = (dim + 15) / 16 * 16;
        const uint32_t tileColumns = capacity;
        const uint32_t nTiles = (dim + tileColumns - 1) / tileColumns;
        AscendC::LocalTensor<T> input{AscendC::TPosition::VECCALC, 0, (2 * capacity + 1) * pitch + 2 * capacity};
        AscendC::LocalTensor<float> wf{AscendC::TPosition::VECCALC, 36864, (2 * capacity + 1) * pitch + 8 * capacity};
        AscendC::LocalTensor<float> sums = wf[(2 * capacity + 1) * pitch];
        AscendC::LocalTensor<T> a{AscendC::TPosition::VECCALC, 114688, 2 * capacity};
        AscendC::LocalTensor<T> b = a[capacity];
        AscendC::LocalTensor<T> out{AscendC::TPosition::VECCALC, 114944, capacity};
        const uint32_t gatherRows = dim < capacity ? dim : capacity;
        const bool gatherWeights = dim > 3 && pitch != dim &&
            gatherRows * pitch <= SWIGLU_GATHER_CAPACITY;
        AscendC::LocalTensor<T> raw{AscendC::TPosition::VECCALC, 131072, 2 * capacity * dim};
        AscendC::LocalTensor<uint32_t> offsets{AscendC::TPosition::VECCALC, 163840, capacity * pitch};
        bool first = true;
        bool firstTile = false, reverseWeights = false, reverseBias = false;
        uint32_t row = 0, col = 0, cols = 0;
        uint64_t weightOffset = 0;
        __aicore__ explicit State(const SwiGluVectorTilingData &data) : t(data) {}
    };

    __aicore__ inline bool ProcessScalar() {
        using namespace AscendC;
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        AscendC::LocalTensor<T> input(AscendC::TPosition::VECCALC, 0, 16384);
        AscendC::LocalTensor<T> a(AscendC::TPosition::VECCALC, 32800, 16384);
        AscendC::LocalTensor<T> b(AscendC::TPosition::VECCALC, 65600, 16384);
        AscendC::LocalTensor<T> out(AscendC::TPosition::VECCALC, 98400, 16384);
        AscendC::LocalTensor<T> coeff(AscendC::TPosition::VECCALC, 131200, 80);
        AscendC::DataCopyExtParams one{1, 2, 0, 0, 0};
        AscendC::DataCopyPad(coeff, w1, one, pad);
        AscendC::DataCopyPad(coeff[16], w2, one, pad);
        AscendC::DataCopyPad(coeff[32], b1, one, pad);
        AscendC::DataCopyPad(coeff[48], b2, one, pad);
        if (t.rows == 1) { ProcessSingleScalar(input, a, out, coeff, one, pad); return true; }
        ProcessScalarRows(input, a, b, out, coeff, pad);
        return false;
    }

    __aicore__ inline void ProcessSingleScalar(AscendC::LocalTensor<T> input,
        AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> out, AscendC::LocalTensor<T> coeff,
        AscendC::DataCopyExtParams one, AscendC::DataCopyPadExtParams<T> pad) {
        using namespace AscendC;
        AscendC::DataCopyPad(input, x, one, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::SetVectorMask<T>(1);
        AscendC::BinaryRepeatParams binary{1, 1, 1, 1, 1, 0};
        AscendC::Mul<T, false>(a, coeff, input, AscendC::MASK_PLACEHOLDER, 2, binary);
        AscendC::PipeBarrier<PIPE_V>();
        binary.src1RepStride = 1;
        if (REVERSE_WEIGHTS == REVERSE_BIAS) {
            AscendC::Add<T, false>(a, a, coeff[32], AscendC::MASK_PLACEHOLDER, 2, binary);
        } else {
            AscendC::Add(a, a, coeff[48], 1);
            AscendC::Add(a[16], a[16], coeff[32], 1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        HalfGateSwiGlu(out, a[REVERSE_WEIGHTS ? 16 : 0], a[REVERSE_WEIGHTS ? 0 : 16], 1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyPad(y, out, one);
    }

    __aicore__ inline void ProcessScalarRows(AscendC::LocalTensor<T> input,
        AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b, AscendC::LocalTensor<T> out,
        AscendC::LocalTensor<T> coeff, AscendC::DataCopyPadExtParams<T> pad) {
        using namespace AscendC;
        T ww1 = 0, ww2 = 0, bb1 = 0, bb2 = 0;
        bool first = true;
        for (uint32_t start = AscendC::GetBlockIdx() * t.tileRows; start < t.rows; start += t.cores * t.tileRows) {
            if (!first) AscendC::PipeBarrier<PIPE_ALL>();
            const uint32_t count = t.rows - start < t.tileRows ? t.rows - start : t.tileRows;
            AscendC::DataCopyPad(input, x[start], AscendC::DataCopyExtParams{1, count * 2, 0, 0, 0}, pad);
            if (first) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
                ww1 = coeff.GetValue(REVERSE_WEIGHTS ? 16 : 0);
                ww2 = coeff.GetValue(REVERSE_WEIGHTS ? 0 : 16);
                bb1 = coeff.GetValue(REVERSE_BIAS ? 48 : 32);
                bb2 = coeff.GetValue(REVERSE_BIAS ? 32 : 48);
                first = false;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::Muls(a, input, ww1, count);
            AscendC::Muls(b, input, ww2, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(a, a, bb1, count);
            AscendC::Adds(b, b, bb2, count);
            AscendC::PipeBarrier<PIPE_V>();
            HalfGateSwiGlu(out, a, b, count);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopyPad(y[start], out, AscendC::DataCopyExtParams{1, count * 2, 0, 0, 0});
        }
    }

    __aicore__ inline void ProcessProjection() {
        using namespace AscendC;
        State s(t);
        for (uint32_t tile = AscendC::GetBlockIdx(); ONE_TILE || tile < t.rows * s.nTiles; tile += t.cores) {
            s.firstTile = s.first;
            if (!s.first) AscendC::PipeBarrier<PIPE_ALL>();
            s.first = false;
            s.row = tile / s.nTiles; s.col = tile % s.nTiles * s.tileColumns;
            s.cols = STATIC_DIM != 0 && STATIC_DIM % s.tileColumns == 0 ?
                s.tileColumns : (s.dim - s.col < s.tileColumns ? s.dim - s.col : s.tileColumns);
            s.weightOffset = static_cast<uint64_t>(s.col) * s.dim;
            LoadWeights(s);
            AscendC::DataCopyPadExtParams<T> matrixPad{true, 0, static_cast<uint8_t>(s.pitch - s.dim), 0};
            AscendC::DataCopyPad(s.input[2 * s.cols * s.pitch], x[static_cast<uint64_t>(s.row) * s.dim],
                AscendC::DataCopyExtParams{1, s.dim * 2, 0, 0, 0}, matrixPad);
            LoadBias(s);
            GatherWeights(s);
            MultiplyWeights(s);
            ReduceWeights(s);
            StoreProjection(s);
            if constexpr (ONE_TILE) break;
        }
    }

    __aicore__ inline void LoadWeights(State &s) {
        using namespace AscendC;
        AscendC::DataCopyExtParams matrixCopy{static_cast<uint16_t>(s.cols), s.dim * 2, 0, 0, 0};
        if (s.pitch == s.dim) {
            matrixCopy.blockCount = 1;
            matrixCopy.blockLen = s.cols * s.dim * 2;
        }
        AscendC::DataCopyPadExtParams<T> matrixPad{true, 0, static_cast<uint8_t>(s.pitch - s.dim), 0};
        s.reverseWeights = false;
        if (s.gatherWeights) {
            s.reverseWeights = CopyOrderedTensorPair(s.raw, s.capacity * s.dim, w1[s.weightOffset], w2[s.weightOffset],
                s.cols * s.dim, s.pad);
        } else if (s.pitch == s.dim) {
            if constexpr (STATIC_DIM == 32 || STATIC_DIM == 64) {
                AscendC::DataCopyPad(s.input, w1[s.weightOffset], matrixCopy, matrixPad);
                AscendC::DataCopyPad(s.input[s.cols * s.pitch], w2[s.weightOffset], matrixCopy, matrixPad);
            } else {
                s.reverseWeights = CopyOrderedTensorPair(s.input, s.cols * s.pitch, w1[s.weightOffset], w2[s.weightOffset],
                    s.cols * s.dim, matrixPad);
            }
        } else {
            AscendC::DataCopyPad(s.input, w1[s.weightOffset], matrixCopy, matrixPad);
            AscendC::DataCopyPad(s.input[s.cols * s.pitch], w2[s.weightOffset], matrixCopy, matrixPad);
        }
    }

    __aicore__ inline void LoadBias(State &s) {
        using namespace AscendC;
        AscendC::DataCopyExtParams biasCopy{1, s.cols * 2, 0, 0, 0};
        s.reverseBias = false;
        if constexpr (STATIC_DIM == 64) {
            AscendC::DataCopyPad(s.input[(2 * s.capacity + 1) * s.pitch], b1[s.col], biasCopy, s.pad);
            AscendC::DataCopyPad(s.input[(2 * s.capacity + 1) * s.pitch + s.capacity], b2[s.col], biasCopy, s.pad);
        } else if constexpr (STATIC_DIM == 32) {
            s.reverseBias = s.reverseWeights;
            AscendC::DataCopyPad(s.input[(2 * s.capacity + 1) * s.pitch + (s.reverseWeights ? s.capacity : 0)],
                b1[s.col], biasCopy, s.pad);
            AscendC::DataCopyPad(s.input[(2 * s.capacity + 1) * s.pitch + (s.reverseWeights ? 0 : s.capacity)],
                b2[s.col], biasCopy, s.pad);
        } else {
            s.reverseBias = CopyOrderedTensorPair(s.input[(2 * s.capacity + 1) * s.pitch], s.capacity,
                b1[s.col], b2[s.col], s.cols, s.pad);
        }
    }

    __aicore__ inline void GatherWeights(State &s) {
        using namespace AscendC;
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        if constexpr (SMALL_FMA) {
            // Keep the FMA accumulator initialization before the GM wait.
            AscendC::Duplicate(s.wf, 0.0f, 2 * s.capacity * s.pitch);
        }
        if (s.firstTile && s.gatherWeights) MakeGatherOffsets(s.offsets, s.dim, s.pitch, s.gatherRows);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        if (s.gatherWeights) {
            AscendC::Gather(s.input, s.raw, s.offsets, 0, s.cols * s.pitch);
            AscendC::Gather(s.input[s.cols * s.pitch], s.raw[s.capacity * s.dim], s.offsets, 0, s.cols * s.pitch);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void MultiplyWeights(State &s) {
        using namespace AscendC;
        if constexpr (SMALL_FMA) MultiplyFma(s);
        else MultiplySeparate(s);
    }

    __aicore__ inline void ReduceWeights(State &s) {
        using namespace AscendC;
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (STATIC_DIM == 32) {
            AscendC::LocalTensor<float> partial(AscendC::TPosition::VECCALC, 65536, 4 * s.capacity);
            AscendC::BlockReduceSum(s.sums, s.wf, s.cols, 64, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::PairReduceSum(partial, s.sums, 2 * s.cols / 16, 64, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::PairReduceSum(s.sums, partial, s.cols / 16, 64, 1, 1, 8);
        } else if constexpr (STATIC_DIM == 64) {
            AscendC::LocalTensor<float> blockSums(AscendC::TPosition::VECCALC, 65536, 256);
            AscendC::LocalTensor<float> partial(AscendC::TPosition::VECCALC, 69632, 128);
            AscendC::BlockReduceSum(blockSums, s.wf, 32, 64, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum(s.sums, blockSums, 4, 64, 1, 1, 8);
        } else {
            for (uint32_t c = 0; c < s.dim; c += 64) {
                const uint32_t width = s.dim - c < 64 ? s.dim - c : 64;
                if (s.cols == s.capacity) {
                    AscendC::WholeReduceSum(s.sums[c / 64 * 2 * s.capacity], s.wf[c], width, 2 * s.capacity, 1, 1, s.pitch / 8);
                } else {
                    AscendC::WholeReduceSum(s.sums[c / 64 * 2 * s.capacity], s.wf[c], width, s.cols, 1, 1, s.pitch / 8);
                    AscendC::WholeReduceSum(s.sums[c / 64 * 2 * s.capacity + s.capacity], s.wf[s.cols * s.pitch + c], width,
                        s.cols, 1, 1, s.pitch / 8);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t c = 64; c < s.dim; c += 64) {
            AscendC::Add(s.sums, s.sums, s.sums[c / 64 * 2 * s.capacity], 2 * s.capacity);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreProjection(State &s) {
        using namespace AscendC;
        AscendC::Cast(s.a, s.sums, AscendC::RoundMode::CAST_RINT, 2 * s.capacity);
        AscendC::PipeBarrier<PIPE_V>();
        if (s.reverseWeights == s.reverseBias) {
            AscendC::Add(s.a, s.a, s.input[(2 * s.capacity + 1) * s.pitch], 2 * s.capacity);
        } else {
            AscendC::Add(s.a, s.a, s.input[(2 * s.capacity + 1) * s.pitch + s.capacity], s.capacity);
            AscendC::Add(s.b, s.b, s.input[(2 * s.capacity + 1) * s.pitch], s.capacity);
        }
        AscendC::PipeBarrier<PIPE_V>();
        HalfGateSwiGlu(s.out, s.reverseWeights ? s.b : s.a, s.reverseWeights ? s.a : s.b, s.cols);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyPad(y[static_cast<uint64_t>(s.row) * s.dim + s.col], s.out,
            AscendC::DataCopyExtParams{1, s.cols * 2, 0, 0, 0});
    }

    __aicore__ inline void MultiplyFma(State &s) {
        using namespace AscendC;
        if constexpr(STATIC_DIM==32) {
            // Each repeat owns one32-element projection column;
            // input remains in its original single32-Half row.
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst<float,T>(s.wf,s.input,s.input[2*s.cols*s.pitch],32,2*s.cols,
                AscendC::BinaryRepeatParams{1,1,1,4,2,0});
        } else {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst<float,T>(s.wf,s.input,s.input[2*s.cols*s.pitch],64,2*s.cols,AscendC::BinaryRepeatParams{1,1,1,8,4,0});
        }
    }

    __aicore__ inline void MultiplySeparate(State &s) {
        using namespace AscendC;
        auto xf = s.wf[2 * s.cols * s.pitch];
        if constexpr (STATIC_DIM == 32) {
            AscendC::Cast(s.wf, s.input, AscendC::RoundMode::CAST_NONE, 2 * s.cols * s.pitch);
            AscendC::Cast(xf, s.input[2 * s.cols * s.pitch], AscendC::RoundMode::CAST_NONE, 32, 2,
                AscendC::UnaryRepeatParams{1, 1, 4, 0});
        } else {
            AscendC::Cast(s.wf, s.input, AscendC::RoundMode::CAST_NONE, (2 * s.cols + 1) * s.pitch);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BinaryRepeatParams mulParams{1, 1, 1,
                static_cast<uint8_t>(s.pitch / 8), static_cast<uint8_t>(s.pitch / 8), 0};
        if constexpr (STATIC_DIM == 32) {
            AscendC::Mul(s.wf, s.wf, xf, 64, s.cols, AscendC::BinaryRepeatParams{1, 1, 1, 8, 8, 0});
        } else {
            for (uint32_t c = 0; c < s.dim; c += 64) {
                const uint32_t width = s.dim - c < 64 ? s.dim - c : 64;
                AscendC::Mul(s.wf[c], s.wf[c], xf[c], width, 2 * s.cols, mulParams);
            }
        }
    }


    SwiGluVectorTilingData t;
    AscendC::GlobalTensor<T> x, w1, w2, b1, b2, y;
};

template <typename T, uint32_t ROWS, bool PADDED = false, bool LONG_K = false,
    uint32_t STATIC_DIM = 0, bool ONE_TILE = false, bool FULL_ROWS = false,
    uint32_t COL_TILE = 16, uint32_t LAST_ROWS = 0, bool FAST33 = false, uint32_t PADDED_BLOCKS = 0>
class BatchedVectorSwiGlu {
    static_assert(ROWS == 1 || ROWS == 2 || ROWS == 3 || ROWS == 4 || ROWS == 7 || ROWS == 8 || ROWS == 16 || ROWS == 32);
    static_assert(!LONG_K || ROWS == 4);
public:
        __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight1, GM_ADDR bias1,
        GM_ADDR weight2, GM_ADDR bias2, GM_ADDR output,
        const SwiGluVectorTilingData &data, GM_ADDR tiling) {
        t = data;
        BindSwiGluTensors(x, w1, w2, b1, b2, y, input, weight1, weight2, bias1, bias2, output);
    }
    __aicore__ inline void Process() {
        using namespace AscendC;
        if constexpr (FAST33 && ROWS == 8) {
            if (AscendC::GetBlockIdx() >= AscendC::GetBlockNum() - PADDED_BLOCKS) return;
        }
        State s(t);
        InitializeOutputMask(s);
        for (uint32_t tile = AscendC::GetBlockIdx(); ONE_TILE || tile < s.tiles; tile += t.cores) {
            s.firstTile = s.first;
            if (!s.first) AscendC::PipeBarrier<PIPE_ALL>();
            s.first = false;
            s.row = tile / s.nTiles * ROWS;
            s.col = tile % s.nTiles * COLS;
            s.nr = CurrentTileRows(s);
            s.cols = PADDED && s.dim - s.col < COLS ? s.dim - s.col : COLS;
            s.reload = s.cachedCol != s.col;
            LoadTile(s);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            if constexpr(FAST33 && ROWS==8) {
                if constexpr(FAST33 && ROWS==8) {
                    s.packMask64.SetValue(0,0xffff0001ffffffffULL);
                    s.packMask64.SetValue(1,0xffffffff0001ffffULL);
                    s.packMask64.SetValue(2,0x0001ffffffff0001ULL);
                    s.packMask64.SetValue(3,0xffff0001ffffffffULL);
                    s.packMask64.SetValue(4,0xffffffff0001ffffULL);
                    s.packMask64.SetValue(5,0x0001ffffffff0001ULL);
                }
                AscendC::SetFlag<AscendC::HardEvent::S_V>(6);
            }
            PrepareGather(s);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            GatherTile(s);
            ConvertTile(s);
            if constexpr (LONG_K) ComputeLongK(s);
            else { ComputeProducts(s); ReduceProducts(s); }
            ApplyBias(s);
            StoreTile(s);
            if constexpr (ONE_TILE) break;
        }
        if constexpr (!ONE_TILE) AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
        static constexpr uint32_t COLS = COL_TILE;
    static constexpr uint32_t OUT_PITCH = (COLS + 15) / 16 * 16;
    static constexpr bool WIDE = COLS > 16;
    static constexpr bool SPLIT33 = STATIC_DIM == 33 && WIDE;
    struct State {
        const SwiGluVectorTilingData &t;
        const uint32_t dim = STATIC_DIM == 0 ? t.dim : STATIC_DIM;
        const uint32_t pitch = SPLIT33 ? 32 : (PADDED ? (dim + 15) / 16 * 16 : dim);
        const uint32_t nTiles = PADDED ? (dim + COLS - 1) / COLS : dim / COLS;
        const uint32_t tiles = ((t.rows + ROWS - 1) / ROWS) * nTiles;
        const uint32_t weightCount = 2 * COLS * pitch;
        AscendC::LocalTensor<T> input{AscendC::TPosition::VECCALC, 0, (2 * COLS + ROWS) * pitch};
        AscendC::LocalTensor<T> bias{AscendC::TPosition::VECCALC, LONG_K ? 12288 : 8192, 2 * OUT_PITCH};
        AscendC::LocalTensor<float> weights{AscendC::TPosition::VECCALC, 16384, weightCount};
        AscendC::LocalTensor<float> inputs{AscendC::TPosition::VECCALC, LONG_K ? 104512 : (WIDE ? 36864 : 24640), ROWS * pitch};
        AscendC::LocalTensor<float> products{AscendC::TPosition::VECCALC, WIDE ? 49152 : 32768, ROWS * weightCount};
        AscendC::LocalTensor<float> sums{AscendC::TPosition::VECCALC, FAST33 && ROWS==8 ? 132096 : 98304, ROWS * 2 * COLS * (LONG_K ? 2 : 1)};
        AscendC::LocalTensor<T> a{AscendC::TPosition::VECCALC, FAST33 && ROWS==8 ? 135168 : 114688, ROWS * OUT_PITCH};
        AscendC::LocalTensor<T> b{AscendC::TPosition::VECCALC, (FAST33 && ROWS==8 ? 135168 : 114688) + ROWS * OUT_PITCH * sizeof(T), ROWS * OUT_PITCH};
        AscendC::LocalTensor<T> out{AscendC::TPosition::VECCALC, (FAST33 && ROWS==8 ? 135232 : 114752) + 2 * ROWS * OUT_PITCH * sizeof(T), ROWS * OUT_PITCH};
        const uint32_t rawPlane = (COLS * dim + 15) / 16 * 16;
        AscendC::LocalTensor<T> raw{AscendC::TPosition::VECCALC, 131072, 2 * rawPlane + (FAST33 ? (ROWS*33+15)/16*16 : ROWS*pitch)};
        AscendC::LocalTensor<uint32_t> offsets{AscendC::TPosition::VECCALC, 163840, COLS * pitch};
        AscendC::LocalTensor<uint64_t> packMask64{AscendC::TPosition::VECCALC,179200,FAST33 && ROWS==8 ? 6 : 4};
        AscendC::LocalTensor<uint32_t> tailIndices{AscendC::TPosition::VECCALC,184320,COLS};
        AscendC::LocalTensor<uint64_t> mainMask64{AscendC::TPosition::VECCALC,163840,24};
        AscendC::LocalTensor<uint64_t> tailMask64{AscendC::TPosition::VECCALC,164096,24};
        AscendC::LocalTensor<T> tailHalf{AscendC::TPosition::VECCALC,FAST33 && ROWS==8 ? 41728 : 94208,2*OUT_PITCH+16};
        AscendC::LocalTensor<float> tailWeights{AscendC::TPosition::VECCALC,FAST33 && ROWS==8 ? 40960 : 86016,2*COLS};
        AscendC::LocalTensor<float> tailProducts{AscendC::TPosition::VECCALC,81920,ROWS*2*COLS};
        AscendC::LocalTensor<float> tailX{AscendC::TPosition::VECCALC,FAST33 && ROWS==8 ? 41280 : 91136,8};
        AscendC::LocalTensor<float> tailBroadcast{AscendC::TPosition::VECCALC,FAST33 && ROWS==8 ? 41472 : 90112,64};
        uint32_t cachedCol = UINT32_MAX;
        bool reverseWeights = false, reverseBias = false;
        bool first = true;
        bool firstTile = false, reload = false;
        uint32_t row = 0, col = 0, nr = 0, cols = 0;
        __aicore__ explicit State(const SwiGluVectorTilingData &data) : t(data) {}
    };

    __aicore__ inline uint32_t CurrentTileRows(State &s) {
        using namespace AscendC;
        return LAST_ROWS != 0 ?
            ((FAST33 && ROWS==8 ? AscendC::GetBlockIdx()+1==AscendC::GetBlockNum()-PADDED_BLOCKS : AscendC::GetBlockIdx()+1==AscendC::GetBlockNum()) ? LAST_ROWS : ROWS) :
            (FULL_ROWS ? ROWS : (t.rows - s.row < ROWS ? t.rows - s.row : ROWS));
    }

    __aicore__ inline void SetMaskPair(State &s, uint32_t index, uint64_t main, uint64_t tail) {
        if constexpr (!FAST33) s.mainMask64.SetValue(index, main);
        s.tailMask64.SetValue(index, tail);
    }

    __aicore__ inline void InitializeSplitMasks(State &s) {
        using namespace AscendC;
        SetMaskPair(s, 0, 0xfffffffeffffffffULL, 0x0000000100000000ULL);
        SetMaskPair(s, 1, 0xfffffffbfffffffdULL, 0x0000000400000002ULL);
        SetMaskPair(s, 2, 0xffffffeffffffff7ULL, 0x0000001000000008ULL);
        SetMaskPair(s, 3, 0xffffffbfffffffdfULL, 0x0000004000000020ULL);
        SetMaskPair(s, 4, 0xfffffeffffffff7fULL, 0x0000010000000080ULL);
        SetMaskPair(s, 5, 0xfffffbfffffffdffULL, 0x0000040000000200ULL);
        SetMaskPair(s, 6, 0xffffeffffffff7ffULL, 0x0000100000000800ULL);
        SetMaskPair(s, 7, 0xffffbfffffffdfffULL, 0x0000400000002000ULL);
        SetMaskPair(s, 8, 0xfffeffffffff7fffULL, 0x0001000000008000ULL);
        SetMaskPair(s, 9, 0xfffbfffffffdffffULL, 0x0004000000020000ULL);
        SetMaskPair(s, 10, 0xffeffffffff7ffffULL, 0x0010000000080000ULL);
        SetMaskPair(s, 11, 0xffbfffffffdfffffULL, 0x0040000000200000ULL);
        SetMaskPair(s, 12, 0xfeffffffff7fffffULL, 0x0100000000800000ULL);
        SetMaskPair(s, 13, 0xfbfffffffdffffffULL, 0x0400000002000000ULL);
        SetMaskPair(s, 14, 0xeffffffff7ffffffULL, 0x1000000008000000ULL);
        SetMaskPair(s, 15, 0xbfffffffdfffffffULL, 0x4000000020000000ULL);
        SetMaskPair(s, 16, 0xffffffff7fffffffULL, 0x0000000080000000ULL);
        SetMaskPair(s, 17, 0xfffffffdfffffffeULL, 0x0000000200000001ULL);
        if constexpr (!FAST33) {
            SetMaskPair(s, 18, 0xfffffff7fffffffbULL, 0x0000000800000004ULL);
            SetMaskPair(s, 19, 0xffffffdfffffffefULL, 0x0000002000000010ULL);
            SetMaskPair(s, 20, 0xffffff7fffffffbfULL, 0x0000008000000040ULL);
        }
    }

    __aicore__ inline void InitializeOutputMask(State &s) {
        using namespace AscendC;
        if constexpr (!(FAST33 && ROWS==8) && STATIC_DIM == 33 && (ROWS == 2 || ROWS == 3 || (FAST33 && ROWS == 8)) && WIDE && ONE_TILE) {
            if constexpr(FAST33 && ROWS==8) {
                s.packMask64.SetValue(0,0xffff0001ffffffffULL);
                s.packMask64.SetValue(1,0xffffffff0001ffffULL);
                s.packMask64.SetValue(2,0x0001ffffffff0001ULL);
                s.packMask64.SetValue(3,0xffff0001ffffffffULL);
                s.packMask64.SetValue(4,0xffffffff0001ffffULL);
                s.packMask64.SetValue(5,0x0001ffffffff0001ULL);
            } else {
                // Select 33 valid values from each padded 48-element row.
                s.packMask64.SetValue(0,0xffff0001ffffffffULL);
                s.packMask64.SetValue(1,ROWS == 2 ? 0x000000000001ffffULL : 0xffffffff0001ffffULL);
                if constexpr (ROWS == 3) {
                    s.packMask64.SetValue(2,1ULL);
                    s.packMask64.SetValue(3,0ULL);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(6);
        }
    }

    __aicore__ inline void LoadTile(State &s) {
        using namespace AscendC;
        if constexpr (PADDED) {
            AscendC::DataCopyPadExtParams<T> noPad{false, 0, 0, 0};
            if (s.reload) {
                s.reverseWeights = CopyOrderedTensorPair(s.raw, s.rawPlane, w1[static_cast<uint64_t>(s.col) * s.dim],
                    w2[static_cast<uint64_t>(s.col) * s.dim], s.cols * s.dim, noPad);
                AscendC::DataCopyPadExtParams<T> biasPad{true, 0, static_cast<uint8_t>(OUT_PITCH - s.cols), 0};
                s.reverseBias = CopyOrderedTensorPair(s.bias, OUT_PITCH, b1[s.col], b2[s.col], s.cols, biasPad);
            }
            if constexpr (STATIC_DIM == 13 && !((ROWS == 3 || ROWS == 7) && FULL_ROWS)) {
                AscendC::DataCopyExtParams xCopy{static_cast<uint16_t>(s.nr), s.dim * 2, 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> xPad{true, 0,
                        static_cast<uint8_t>(s.pitch - s.dim), 0};
                AscendC::DataCopyPad(s.input[s.weightCount], x[static_cast<uint64_t>(s.row) * s.dim],
                    xCopy, xPad);
            } else {
                AscendC::DataCopyPad(s.raw[2 * s.rawPlane], x[static_cast<uint64_t>(s.row) * s.dim],
                    AscendC::DataCopyExtParams{1, s.nr * s.dim * 2, 0, 0, 0}, noPad);
            }
        } else {
            if (s.reload) {
                AscendC::DataCopy(s.input, w1[static_cast<uint64_t>(s.col) * s.dim], COLS * s.dim);
                AscendC::DataCopy(s.input[COLS * s.dim], w2[static_cast<uint64_t>(s.col) * s.dim], COLS * s.dim);
                AscendC::DataCopy(s.bias, b1[s.col], COLS);
                AscendC::DataCopy(s.bias[COLS], b2[s.col], COLS);
            }
            AscendC::DataCopy(s.input[s.weightCount], x[static_cast<uint64_t>(s.row) * s.dim], s.nr * s.dim);
        }
    }

    __aicore__ inline void PrepareGather(State &s) {
        using namespace AscendC;
        if constexpr (PADDED) {
            if (s.firstTile) {
                const uint32_t weightRows = s.dim < COLS ? s.dim : COLS;
                const uint32_t gatherRows = weightRows > ROWS ? weightRows : ROWS;
                if constexpr (SPLIT33) {
                    // Main K=32 and final K=1 are complementary selections
                    // from contiguous rows of length33. No index-vector construction.
                    InitializeSplitMasks(s);
                    AscendC::SetFlag<AscendC::HardEvent::S_V>(7); AscendC::WaitFlag<AscendC::HardEvent::S_V>(7);
                    if constexpr(FAST33) {
                        AscendC::Not(s.mainMask64.template ReinterpretCast<uint16_t>(),
                            s.tailMask64.template ReinterpretCast<uint16_t>(),72);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                } else {
                    MakeGatherOffsets(s.offsets, s.dim, s.pitch, gatherRows);
                }
            }
        }
    }

    __aicore__ inline void GatherTile(State &s) {
        using namespace AscendC;
        if constexpr (PADDED) {
            if constexpr (SPLIT33) {
                auto mainMask=s.mainMask64.template ReinterpretCast<uint16_t>();
                auto tailMask=s.tailMask64.template ReinterpretCast<uint16_t>();
                AscendC::SetMaskCount();
                if(s.reload) {
                    AscendC::SetVectorMask<uint16_t,AscendC::MaskMode::COUNTER>(s.cols*33);
                    GatherMaskCurrentMode(s.input,s.raw,mainMask);
                    GatherMaskCurrentMode(s.input[COLS*s.pitch],s.raw[s.rawPlane],mainMask);
                    GatherMaskCurrentMode(s.tailHalf,s.raw,tailMask);
                    GatherMaskCurrentMode(s.tailHalf[OUT_PITCH],s.raw[s.rawPlane],tailMask);
                }
                AscendC::SetVectorMask<uint16_t,AscendC::MaskMode::COUNTER>(s.nr*33);
                GatherMaskCurrentMode(s.input[s.weightCount],s.raw[2*s.rawPlane],mainMask);
                GatherMaskCurrentMode(s.tailHalf[2*OUT_PITCH],s.raw[2*s.rawPlane],tailMask);
                AscendC::SetMaskNorm();
            } else {
                if (s.reload) {
                    AscendC::Gather(s.input, s.raw, s.offsets, 0, s.cols * s.pitch);
                    AscendC::Gather(s.input[COLS * s.pitch], s.raw[s.rawPlane], s.offsets, 0, s.cols * s.pitch);
                }
                if constexpr (STATIC_DIM != 13 || ((ROWS == 3 || ROWS == 7) && FULL_ROWS)) {
                    AscendC::Gather(s.input[s.weightCount], s.raw[2 * s.rawPlane], s.offsets, 0, s.nr * s.pitch);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ConvertTile(State &s) {
        using namespace AscendC;
        if (s.reload) {
            AscendC::Cast(s.weights, s.input, AscendC::RoundMode::CAST_NONE, s.weightCount);
            s.cachedCol = s.col;
        }
        ConvertInputRows(s);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (SPLIT33) {
            AscendC::Brcb(s.tailBroadcast,s.tailX,1,AscendC::BrcbRepeatParams{1,8});
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ApplyBias(State &s) {
        using namespace AscendC;
        AscendC::UnaryRepeatParams castParams{1, 1, OUT_PITCH / 16, 2 * COLS / 8};
        AscendC::Cast(s.a, s.sums, AscendC::RoundMode::CAST_RINT, COLS, s.nr, castParams);
        AscendC::Cast(s.b, s.sums[COLS], AscendC::RoundMode::CAST_RINT, COLS, s.nr, castParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BinaryRepeatParams biasParams{1, 1, 1, OUT_PITCH / 16, OUT_PITCH / 16, 0};
        const bool swapBias = s.reverseWeights != s.reverseBias;
        AscendC::Add(s.a, s.a, s.bias[swapBias ? OUT_PITCH : 0], COLS, s.nr, biasParams);
        AscendC::Add(s.b, s.b, s.bias[swapBias ? 0 : OUT_PITCH], COLS, s.nr, biasParams);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreTile(State &s) {
        using namespace AscendC;
        AscendC::LocalTensor<uint16_t> packMask(AscendC::TPosition::VECCALC, 163840, 16);
        if constexpr (STATIC_DIM == 13 && (ROWS == 3 || ROWS == 7) && FULL_ROWS) {
            AscendC::Duplicate(packMask, static_cast<uint16_t>(0x1fff), 16);
        }
        if constexpr (FAST33) StoreFast33(s);
        else StoreGeneral(s, packMask);
    }

    __aicore__ inline void ComputeLongK(State &s) {
        using namespace AscendC;
        ComputeLongKPartials<ROWS, COLS>(s);
        // Match the original vector path's two 64-element partial sums.
        AscendC::Add(s.sums, s.sums, s.sums[ROWS * 2 * COLS], s.nr * 2 * COLS);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeProducts(State &s) {
        using namespace AscendC;
        AscendC::BinaryRepeatParams mulParams{1, 1, 1, static_cast<uint8_t>(s.pitch / 8),
                static_cast<uint8_t>(s.pitch / 8), 0};
        if constexpr (STATIC_DIM == 13) {
            AscendC::LocalTensor<float> expanded(AscendC::TPosition::VECCALC, 28672, ROWS * 64);
            for (uint32_t rr = 0; rr < s.nr; ++rr) {
                AscendC::Mul(s.products[rr * s.weightCount], s.weights, expanded[rr * 64], 64, 8,
                    AscendC::BinaryRepeatParams{1, 1, 1, 8, 8, 0});
            }
        } else if constexpr (SPLIT33) {
            AscendC::LocalTensor<float> expanded(AscendC::TPosition::VECCALC,FAST33 && ROWS==8 ? 36864 : 102400,ROWS*64);
            for(uint32_t rr=0;rr<s.nr;++rr) {
                AscendC::Mul(s.products[rr*s.weightCount],s.weights,expanded[rr*64],64,COLS,
                    AscendC::BinaryRepeatParams{1,1,1,8,8,0});
                if constexpr(!FAST33) {
                    AscendC::Mul(s.tailProducts[rr*2*COLS],s.tailWeights,s.tailBroadcast[rr*8],COLS,2,
                        AscendC::BinaryRepeatParams{1,1,0,COLS/8,COLS/8,0});
                }
            }
        } else {
            for (uint32_t rr = 0; rr < s.nr; ++rr) {
                AscendC::Mul(s.products[rr * s.weightCount], s.weights, s.inputs[rr * s.pitch], s.dim, 2 * COLS, mulParams);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceProducts(State &s) {
        using namespace AscendC;
        if constexpr (STATIC_DIM == 13) {
            AscendC::LocalTensor<float> blockSums(AscendC::TPosition::VECCALC, 65536, ROWS * 64);
            // Ignore the three padded K lanes in each 16-element row
            // directly in the reduction, avoiding a clearing pass.
            uint64_t validMask[2] = {0x1fff1fff1fff1fffULL, 0};
            AscendC::BlockReduceSum(blockSums, s.products, s.nr * 8, validMask, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::PairReduceSum(s.sums, blockSums, s.nr, 64, 1, 1, 8);
        } else if constexpr (SPLIT33) {
            ReduceSplit33(s);
        } else if constexpr (ROWS == 4) {
            AscendC::WholeReduceSum(s.sums, s.products, s.dim, s.nr * 2 * COLS, 1, 1, s.pitch / 8);
        } else {
            // The instruction accepts at most 255 repeats.
            for (uint32_t rr = 0; rr < s.nr; rr += (WIDE ? 3 : 4)) {
                const uint32_t count = s.nr - rr < (WIDE ? 3 : 4) ? s.nr - rr : (WIDE ? 3 : 4);
                AscendC::WholeReduceSum(s.sums[rr * 2 * COLS], s.products[rr * s.weightCount], s.dim,
                    count * 2 * COLS, 1, 1, s.pitch / 8);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceSplit33(State &s) {
        using namespace AscendC;
        // Each 32-product dot yields four adjacent 8-lane sums.
        AscendC::LocalTensor<float> blockPartial(AscendC::TPosition::VECCALC,FAST33 && ROWS==8 ? 135168 : 131072,ROWS*8*COLS);
        AscendC::LocalTensor<float> pairPartial(AscendC::TPosition::VECCALC,FAST33 && ROWS==8 ? 147456 : 139264,ROWS*4*COLS);
        if constexpr(FAST33 && ROWS==8) {
            // At most6rows*40=240 repeats per instruction.
            const uint32_t first=s.nr>6?6:s.nr;
            AscendC::BlockReduceSum(blockPartial,s.products,first*COLS,64,1,1,8);
            if(s.nr>6) AscendC::BlockReduceSum(blockPartial[6*8*COLS],
                s.products[6*s.weightCount],(s.nr-6)*COLS,64,1,1,8);
        } else {
            AscendC::BlockReduceSum(blockPartial,s.products,s.nr*COLS,64,1,1,8);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PairReduceSum(pairPartial,blockPartial,s.nr*COLS/8,64,1,1,8);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t pairCount=s.nr*4*COLS;
        AscendC::PairReduceSum(s.sums,pairPartial,pairCount/64,64,1,1,8);
        if(pairCount%64) {
            AscendC::PairReduceSum(s.sums[pairCount/64*32],pairPartial[pairCount/64*64],1,pairCount%64,1,1,8);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AccumulateTail(s);
    }

    __aicore__ inline void AccumulateTail(State &s) {
        using namespace AscendC;
        if constexpr(FAST33) {
            // Each tail product is exactly representable in FP32:
            // its operands are half values converted to float.
            if constexpr(ROWS==8) {
                // Each Brcb row has eight copies of the same X33.
                // Advance output rows, reuse all80 tail weights.
                const AscendC::BinaryRepeatParams rowMac{1,1,0,2*COLS/8,0,1};
                AscendC::MulAddDst<float,float>(s.sums,s.tailWeights,s.tailBroadcast,
                    64,s.nr,rowMac);
                AscendC::MulAddDst<float,float>(s.sums[64],s.tailWeights[64],s.tailBroadcast,
                    2*COLS-64,s.nr,rowMac);
            } else {
                for(uint32_t rr=0;rr<s.nr;++rr) {
                    AscendC::MulAddDst<float,float>(s.sums[rr*2*COLS],s.tailWeights,
                        s.tailBroadcast[rr*8],COLS,2,
                        AscendC::BinaryRepeatParams{1,1,0,COLS/8,COLS/8,0});
                }
            }
        } else {
            AscendC::Add(s.sums,s.sums,s.tailProducts,s.nr*2*COLS);
        }
    }

    __aicore__ inline void ConvertInputRows(State &s) {
        using namespace AscendC;
        if constexpr (STATIC_DIM == 13) {
            AscendC::LocalTensor<float> expanded(AscendC::TPosition::VECCALC, 28672, ROWS * 64);
            for (uint32_t rr = 0; rr < s.nr; ++rr) {
                AscendC::Cast(expanded[rr * 64], s.input[s.weightCount + rr * s.pitch], AscendC::RoundMode::CAST_NONE,
                    16, 4, AscendC::UnaryRepeatParams{1, 1, 2, 0});
            }
        } else if constexpr (SPLIT33) {
            AscendC::LocalTensor<float> expanded(AscendC::TPosition::VECCALC,FAST33 && ROWS==8 ? 36864 : 102400,ROWS*64);
            if(s.reload) {
                AscendC::Cast(s.tailWeights,s.tailHalf,AscendC::RoundMode::CAST_NONE,COLS);
                AscendC::Cast(s.tailWeights[COLS],s.tailHalf[OUT_PITCH],AscendC::RoundMode::CAST_NONE,COLS);
            }
            AscendC::Cast(s.tailX,s.tailHalf[2*OUT_PITCH],AscendC::RoundMode::CAST_NONE,s.nr);
            if constexpr(FAST33 && ROWS==8) {
                // Two row-strided casts produce [Xrow,Xrow] for each row.
                AscendC::Cast(expanded,s.input[s.weightCount],AscendC::RoundMode::CAST_NONE,
                    32,s.nr,AscendC::UnaryRepeatParams{1,1,8,2});
                AscendC::Cast(expanded[32],s.input[s.weightCount],AscendC::RoundMode::CAST_NONE,
                    32,s.nr,AscendC::UnaryRepeatParams{1,1,8,2});
            } else {
                for(uint32_t rr=0;rr<s.nr;++rr) {
                    AscendC::Cast(expanded[rr*64],s.input[s.weightCount+rr*s.pitch],AscendC::RoundMode::CAST_NONE,
                        32,2,AscendC::UnaryRepeatParams{1,1,4,0});
                }
            }
        } else {
            AscendC::Cast(s.inputs, s.input[s.weightCount], AscendC::RoundMode::CAST_NONE, s.nr * s.pitch);
        }
    }

    __aicore__ inline void StoreFast33(State &s) {
        using namespace AscendC;
        AscendC::LocalTensor<T> packedA(AscendC::TPosition::VECCALC,ROWS==8 ? 153600 : 131072,ROWS*48);
        AscendC::LocalTensor<T> packedB(AscendC::TPosition::VECCALC,ROWS==8 ? 154400 : 131392,ROWS*48);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(6);
        auto pattern=s.packMask64.template ReinterpretCast<uint16_t>();
        AscendC::SetMaskCount();AscendC::SetVectorMask<uint16_t,AscendC::MaskMode::COUNTER>(s.nr*48);
        GatherMaskCurrentMode(packedA,s.a,pattern);
        GatherMaskCurrentMode(packedB,s.b,pattern);
        AscendC::SetMaskNorm();AscendC::PipeBarrier<PIPE_V>();
        HalfGateSwiGlu(s.out,s.reverseWeights?packedB:packedA,
            s.reverseWeights?packedA:packedB,s.nr*33);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyPad(y[static_cast<uint64_t>(s.row)*33],s.out,
            AscendC::DataCopyExtParams{1,s.nr*66,0,0,0});
    }

    __aicore__ inline void StoreGeneral(State &s, AscendC::LocalTensor<uint16_t> packMask) {
        using namespace AscendC;
        HalfGateSwiGlu(s.out, s.reverseWeights ? s.b : s.a, s.reverseWeights ? s.a : s.b, s.nr * OUT_PITCH);
        if constexpr (STATIC_DIM == 13 && (ROWS == 3 || ROWS == 7) && FULL_ROWS) {
            // Compress padded rows into one contiguous output copy.
            AscendC::LocalTensor<T> packed(AscendC::TPosition::VECCALC, 131072, ROWS*16);
            AscendC::PipeBarrier<PIPE_V>();
            uint64_t selected=0;
            AscendC::GatherMask(packed,s.out,packMask,true,ROWS*16,AscendC::GatherMaskParams{1,1,8,0},selected);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopyPad(y[static_cast<uint64_t>(s.row)*13],packed,AscendC::DataCopyExtParams{1,ROWS*26,0,0,0});
        } else if constexpr (STATIC_DIM == 33 && (ROWS == 2 || ROWS == 3) && WIDE && ONE_TILE) {
            AscendC::LocalTensor<T> packed(AscendC::TPosition::VECCALC,131072,ROWS*48);
            auto packMask=s.packMask64.template ReinterpretCast<uint16_t>();
            AscendC::PipeBarrier<PIPE_V>(); AscendC::WaitFlag<AscendC::HardEvent::S_V>(6);
            uint64_t selected=0;
            AscendC::GatherMask(packed,s.out,packMask,true,s.nr*48,AscendC::GatherMaskParams{1,1,8,0},selected);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopyPad(y[static_cast<uint64_t>(s.row)*33],packed,AscendC::DataCopyExtParams{1,s.nr*66,0,0,0});
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopyExtParams outCopy{static_cast<uint16_t>(s.nr), s.cols * 2, 0, (s.dim - s.cols) * 2, 0};
            AscendC::DataCopyPad(y[static_cast<uint64_t>(s.row) * s.dim + s.col], s.out, outCopy);
        }
    }


    SwiGluVectorTilingData t;
    AscendC::GlobalTensor<T> x, w1, w2, b1, b2, y;
};

// Two rows share each converted weight tile for the remaining vector-small
// dimensions. Products are reduced in 64-element chunks and accumulated in
// the same order as VectorSwiGlu.

template <typename T, uint32_t MODE, bool ONE_TILE = false, uint32_t STATIC_DIM = 0>
using VectorSmallFmaSwiGlu = VectorSwiGlu<T, MODE, ONE_TILE, STATIC_DIM, true>;

template <typename T, uint32_t DIM>
using TinyCurrentMaskSwiGlu = TinyVectorSwiGlu<T, DIM, true>;

template <typename T>
using ScalarVectorFmaCompactSwiGlu = ScalarVectorSwiGlu<T, true>;



template <typename T, bool PACKED_N = false>
__aicore__ inline void CopyProjectionChunk(AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b,
    const AscendC::GlobalTensor<T> &x, const AscendC::GlobalTensor<T> &w1, const AscendC::GlobalTensor<T> &w2,
    uint64_t xOffset, uint64_t wOffset, uint32_t chunk, uint32_t nk, uint32_t dim,
    uint32_t rows, uint32_t cols, uint32_t ma, uint32_t na, bool copyWeights) {
    AscendC::Nd2NzParams aCopy{};
    aCopy.ndNum = 1;
    aCopy.nValue = rows;
    aCopy.dValue = nk;
    aCopy.srcDValue = dim;
    aCopy.dstNzC0Stride = ma;
    aCopy.dstNzNStride = 1;
    AscendC::DataCopy(a, x[xOffset + chunk], aCopy);
    if (copyWeights) {
        AscendC::Nd2NzParams weightCopyParams = aCopy;
        weightCopyParams.nValue = cols;
        weightCopyParams.dstNzC0Stride = PACKED_N ? (2 * cols + 15) / 16 * 16 : 2 * na;
        AscendC::DataCopy(b, w1[wOffset + chunk], weightCopyParams);
        AscendC::DataCopy(b[(PACKED_N ? cols : na) * 16], w2[wOffset + chunk], weightCopyParams);
    }
}

struct ProjectionSchedule {
    uint32_t waveRows, fullWaves, remainingRows, tailTileRows, fullTiles, tiles;
};

template<bool BALANCED_TAIL, uint32_t STATIC_TAIL_ROWS>
__aicore__ inline ProjectionSchedule MakeProjectionSchedule(uint32_t totalRows,
    uint32_t cores, uint32_t nTiles, uint32_t tileRows) {
    ProjectionSchedule schedule;
    schedule.waveRows = cores / nTiles * tileRows;
    schedule.fullWaves = BALANCED_TAIL ? totalRows / schedule.waveRows : 0;
    schedule.remainingRows = BALANCED_TAIL ? totalRows - schedule.fullWaves * schedule.waveRows : 0;
    schedule.tailTileRows = STATIC_TAIL_ROWS != 0 ? STATIC_TAIL_ROWS :
        (BALANCED_TAIL && schedule.remainingRows != 0 ?
        ((schedule.remainingRows + cores / nTiles * 16 - 1) / (cores / nTiles * 16)) * 16 : tileRows);
    schedule.fullTiles = schedule.fullWaves * cores;
    schedule.tiles = BALANCED_TAIL ? schedule.fullTiles +
        ((schedule.remainingRows + schedule.tailTileRows - 1) / schedule.tailTileRows) * nTiles :
        ((totalRows + tileRows - 1) / tileRows) * nTiles;
    return schedule;
}

struct ProjectionTileGeometry {
    bool tail;
    uint32_t row, col, activeTileRows, rows, cols;
};

template<bool BALANCED_TAIL, bool FULL_ROW_TILE, uint32_t STATIC_TAIL_ROWS, bool COMPLETE_TAIL>
__aicore__ inline ProjectionTileGeometry MakeProjectionTile(uint32_t tile, uint32_t core,
    uint32_t tileRows, uint32_t tileCols, uint32_t nTiles, uint32_t totalRows,
    uint32_t dim, const ProjectionSchedule &schedule) {
    ProjectionTileGeometry geometry;
    geometry.tail = BALANCED_TAIL && tile >= schedule.fullTiles;
    geometry.row = geometry.tail ? schedule.fullWaves * schedule.waveRows + (tile - schedule.fullTiles) / nTiles * schedule.tailTileRows :
        tile / nTiles * tileRows;
    geometry.col = (FULL_ROW_TILE ? core % nTiles : tile % nTiles) * tileCols;
    geometry.activeTileRows = geometry.tail ? schedule.tailTileRows : tileRows;
    geometry.rows = COMPLETE_TAIL ? geometry.activeTileRows :
        (FULL_ROW_TILE && (!BALANCED_TAIL || (STATIC_TAIL_ROWS != 0 && !geometry.tail)) ? tileRows :
        (totalRows - geometry.row < geometry.activeTileRows ? totalRows - geometry.row : geometry.activeTileRows));
    geometry.cols = FULL_ROW_TILE ? tileCols : (dim - geometry.col < tileCols ? dim - geometry.col : tileCols);
    return geometry;
}

template <typename T, bool PIPELINED = false, bool PREFETCH = false,
    bool SINGLE_TILE = false, uint32_t PIPE_K_STEP = 64, uint32_t STATIC_DIM = 0, uint32_t STATIC_ROWS = 0,
    bool BALANCED_TAIL = false, uint32_t STATIC_TAIL_ROWS = 0, bool COMPLETE_TAIL = false, bool LAZY_TILING = false>
class ProjectionComputer {
public:
        __aicore__ ProjectionComputer(GM_ADDR ix, GM_ADDR iw1, GM_ADDR iw2, GM_ADDR ws,
        const SwiGluVectorTilingData &data, GM_ADDR td)
        : input(ix), weight1(iw1), weight2(iw2), workspace(ws), t(data), tiling(td) {}
    __aicore__ inline void Run() {
        tileRows = STATIC_ROWS == 0 ? t.tileRows : STATIC_ROWS;
        dim = STATIC_DIM == 0 ? t.dim : STATIC_DIM;
        tileCols = STATIC_DIM == 0 ? t.tileCols : (STATIC_DIM < 128 ? (STATIC_DIM + 15) / 16 * 16 : 128);
        InitializeBuffers();
        PrimePipeline();
        InitializeSchedule();
        for (tile = core; SINGLE_TILE || tile < schedule.tiles; tile += t.cores, ++iteration) {
            PrepareTile();
            PrefetchTile();
            for (chunk = 0; chunk < dim; chunk += L1_K_STEP) {
                PrepareChunk();
                for (k = 0; k < l1k; k += K_STEP) {
                    ka = l1k - k < K_STEP ? l1k - k : K_STEP;
                    slot = PIPELINED ? (k / K_STEP) % 2 : 0;
                    AcquireL0();
                    a = aL0[slot * tileRows * K_STEP];
                    b = bL0[slot * 2 * tileCols * K_STEP];
                    LoadA();
                    MultiplyChunk();
                    PrefetchNextRow();
                }
                PrefetchNextChunk();
            }
            StoreTile();
            if constexpr (SINGLE_TILE) break;
        }
        FinishPipeline();
    }
private:
        ProjectionTileGeometry geometry;
    ProjectionSchedule schedule;
    GM_ADDR input, weight1, weight2, workspace, tiling;
    const SwiGluVectorTilingData &t;
    static constexpr bool UNIT_FLOW = !SINGLE_TILE;
    static constexpr bool DOUBLE_GM = !SINGLE_TILE && (STATIC_DIM == 128 || STATIC_DIM == 256);
    static constexpr bool FULL_ROW_TILE = STATIC_DIM == 256 && STATIC_ROWS == 128;
    static constexpr bool ROW_PREFETCH = FULL_ROW_TILE;
    static constexpr bool PACKED_N = (STATIC_DIM == 33 && STATIC_ROWS == 16) || FULL_ROW_TILE;
    static constexpr uint32_t K_STEP = PIPELINED ? PIPE_K_STEP : 128;
    static constexpr uint32_t BUFFERS = PIPELINED ? 2 : 1;
    static constexpr uint32_t L1_K_STEP = PREFETCH ? 256 : 512;
    static constexpr uint32_t L1_BUFFERS = PREFETCH ? 2 : 1;
    uint32_t tileRows, dim, tileCols, span, core, nTiles, cachedCol, totalRows, iteration, tile, ma, na, projectionCols, rowSlot, chunk, nk, l1k, l1slot, k, ka, slot;
    uint64_t xOffset, wOffset;

    bool pending[2] = {false, false};
    AscendC::GlobalTensor<T> x, w1, w2, first, second;
    AscendC::LocalTensor<T> aL1;
    AscendC::LocalTensor<T> bL1;
    AscendC::LocalTensor<T> aL0;
    AscendC::LocalTensor<T> bL0;
    AscendC::LocalTensor<float> cL0;
    AscendC::LocalTensor<T> currentA;
    AscendC::LocalTensor<T> currentB;
    AscendC::LocalTensor<T> a;
    AscendC::LocalTensor<T> b;
    __aicore__ inline void LoadA() {
        if constexpr (SINGLE_TILE && STATIC_DIM == 256) {
            AscendC::LoadData2dParams loadA{};
            loadA.repeatTimes = ka / 16;
            loadA.srcStride = ma / 16;
            for (uint32_t m = 0; m < ma / 16; ++m) {
                AscendC::LoadData(a[m * ka * 16], currentA[k * ma + m * 256], loadA);
            }
        } else {
            AscendC::LoadData3DParamsV2<T> loadA{};
            loadA.l1H = 1;
            loadA.l1W = ma;
            loadA.channelSize = l1k;
            loadA.kExtension = ka;
            loadA.mExtension = ma;
            loadA.kStartPt = k;
            loadA.strideW = 1;
            loadA.strideH = 1;
            loadA.filterW = 1;
            loadA.filterH = 1;
            loadA.dilationFilterW = 1;
            loadA.dilationFilterH = 1;
            AscendC::LoadData(a, currentA, loadA);
        }
    }

    __aicore__ inline void MultiplyChunk() {
        AscendC::LoadData2dParams load{};
        load.repeatTimes = projectionCols * ka / 256;
        load.srcStride = 1;
        AscendC::LoadData(b, currentB[k * projectionCols], load);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slot);
        AscendC::MmadParams mad{};
        mad.m = ma;
        mad.n = projectionCols;
        mad.k = ka;
        mad.cmatrixInitVal = chunk == 0 && k == 0;
        if constexpr (UNIT_FLOW) mad.unitFlag =
            chunk + nk >= dim && k + ka >= l1k ? 3 : 2;
        AscendC::Mmad(cL0, a, b, mad);
        if constexpr (PIPELINED) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3 + slot);
        }
    }

    __aicore__ inline void PrefetchNextRow() {
        if constexpr (ROW_PREFETCH) {
            if (chunk == 0 && k == 0) {
                const uint32_t nextTile = tile + t.cores;
                if (nextTile < schedule.tiles) {
                    const bool nextTail = BALANCED_TAIL && nextTile >= schedule.fullTiles;
                    const uint32_t nextRow = nextTail ? schedule.fullWaves * schedule.waveRows +
                        (nextTile - schedule.fullTiles) / nTiles * schedule.tailTileRows : nextTile / nTiles * tileRows;
                    const uint32_t nextTileRows = nextTail ? schedule.tailTileRows : tileRows;
                    const uint32_t nextRows = COMPLETE_TAIL ? nextTileRows : (totalRows - nextRow < nextTileRows ? totalRows - nextRow : nextTileRows);
                    const uint32_t nextSlot = rowSlot ^ 1;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(nextSlot);
                    CopyProjectionChunk<T, PACKED_N>(aL1[nextSlot * tileRows * dim],
                        bL1, x, w1, w2, static_cast<uint64_t>(nextRow) * dim, wOffset,
                        0, dim, dim, nextRows, tileCols, STATIC_TAIL_ROWS != 0 ? nextTileRows : (nextRows + 15) / 16 * 16, tileCols, false);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(nextSlot);
                }
            }
        }
    }

    __aicore__ inline void AcquireL0() {
        if constexpr (PIPELINED) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3 + slot);
        } else {
            if constexpr (!SINGLE_TILE) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
            } else if (k != 0) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
            }
        }
    }

    __aicore__ inline void PrefetchNextChunk() {
        if constexpr (PREFETCH) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1slot);
            const uint32_t nextChunk = chunk + 2 * L1_K_STEP;
            if (nextChunk < dim) {
                const uint32_t nextK = dim - nextChunk < L1_K_STEP ?
                    dim - nextChunk : L1_K_STEP;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1slot);
                CopyProjectionChunk<T, PACKED_N>(currentA, currentB, x, w1, w2, xOffset, wOffset,
                    nextChunk, nextK, dim, geometry.rows, geometry.cols, ma, na, true);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1slot);
            }
        }
    }

    __aicore__ inline void PrepareTile() {
        geometry = MakeProjectionTile<BALANCED_TAIL, FULL_ROW_TILE, STATIC_TAIL_ROWS, COMPLETE_TAIL>(
            tile, core, tileRows, tileCols, nTiles, totalRows, dim, schedule);
        ma = STATIC_TAIL_ROWS != 0 ? geometry.activeTileRows :
            (FULL_ROW_TILE && !BALANCED_TAIL ? tileRows : (STATIC_ROWS == 16 ? 16 : (geometry.rows + 15) / 16 * 16));
        na = (geometry.cols + 15) / 16 * 16;
        projectionCols = PACKED_N ? (2 * geometry.cols + 15) / 16 * 16 : 2 * na;
        xOffset = static_cast<uint64_t>(geometry.row) * dim;
        wOffset = static_cast<uint64_t>(geometry.col) * dim;
        rowSlot = ROW_PREFETCH ? (iteration & 1U) : 0;
    }

    __aicore__ inline void PrefetchTile() {
        if constexpr (ROW_PREFETCH) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(rowSlot);
        }
        if constexpr (PREFETCH) {
            for (slot = 0; slot < 2; ++slot) {
                chunk = slot * L1_K_STEP;
                nk = dim - chunk < L1_K_STEP ? dim - chunk : L1_K_STEP;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
                CopyProjectionChunk<T, PACKED_N>(aL1[slot * tileRows * L1_K_STEP],
                    bL1[slot * 2 * tileCols * L1_K_STEP], x, w1, w2,
                    xOffset, wOffset, chunk, nk, dim, geometry.rows, geometry.cols, ma, na, true);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
            }
        }
    }

    __aicore__ inline void PrepareChunk() {
        nk = dim - chunk < L1_K_STEP ? dim - chunk : L1_K_STEP;
        l1k = (nk + 15) / 16 * 16;
        l1slot = PREFETCH ? (chunk / L1_K_STEP) % 2 : 0;
        currentA = aL1[ROW_PREFETCH ? rowSlot * tileRows * dim : l1slot * tileRows * L1_K_STEP];
        currentB = bL1[l1slot * 2 * tileCols * L1_K_STEP];
        if constexpr (!PREFETCH && !ROW_PREFETCH) {
            if constexpr (!SINGLE_TILE) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            }
            // Reuse complete resident weights across row tiles.
            CopyProjectionChunk<T, PACKED_N>(currentA, currentB, x, w1, w2, xOffset, wOffset,
                chunk, nk, dim, geometry.rows, geometry.cols, ma, na,
                dim > L1_K_STEP || cachedCol != geometry.col);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        }
        if constexpr (!ROW_PREFETCH) AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1slot);
    }

    __aicore__ inline void StoreTile() {
        if constexpr (ROW_PREFETCH) AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(rowSlot);
        cachedCol = geometry.col;
        if constexpr (!UNIT_FLOW) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
        }
        // Only the GM result buffer needs the vector reader's release.
        // Compute the next tile while the previous tile is consumed.
        const uint32_t outputSlot = DOUBLE_GM ? (iteration & 1U) : 0;
        if (pending[outputSlot]) AscendC::CrossCoreWaitFlag(9 + 2 * outputSlot);
        AscendC::FixpipeParamsV220 fix{};
        fix.nSize = PACKED_N ? projectionCols : na;
        fix.mSize = ma;
        fix.srcStride = ma;
        fix.dstStride = PACKED_N ? projectionCols : tileCols;
        fix.ndNum = PACKED_N ? 1 : 2;
        fix.srcNdStride = na * ma / 256;
        fix.dstNdStride = span;
        fix.quantPre = AscendC::QuantMode_t::F322F16;
        fix.unitFlag = UNIT_FLOW ? 3 : 0;
        AscendC::Fixpipe(first[outputSlot * 2 * span], cL0, fix);
        if constexpr (!SINGLE_TILE && !UNIT_FLOW) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
        }
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(8 + 2 * outputSlot);
        pending[outputSlot] = true;
    }

    __aicore__ inline void InitializeBuffers() {
        AscendC::SetAtomicNone();
        if constexpr (UNIT_FLOW) AscendC::SetMMRowMajor();
        AscendC::SetLoadDataBoundary(0);
        AscendC::SetLoadDataPaddingValue(static_cast<uint64_t>(0));
        aL1 = AscendC::LocalTensor<T>(AscendC::TPosition::A1, 0, L1_BUFFERS * tileRows * L1_K_STEP);
        bL1 = AscendC::LocalTensor<T>(AscendC::TPosition::A1, 131072, L1_BUFFERS * 2 * tileCols * L1_K_STEP);
        aL0 = AscendC::LocalTensor<T>(AscendC::TPosition::A2, 0, BUFFERS * tileRows * K_STEP);
        bL0 = AscendC::LocalTensor<T>(AscendC::TPosition::B2, 0, BUFFERS * 2 * tileCols * K_STEP);
        cL0 = AscendC::LocalTensor<float>(AscendC::TPosition::CO1, 0, 2 * tileRows * tileCols);
        x.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input));
        w1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight1));
        w2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight2));
        span = tileRows * tileCols;
        core = AscendC::GetBlockIdx();
        first.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + core * (DOUBLE_GM ? 4 : 2) * span);
        second.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + (core * (DOUBLE_GM ? 4 : 2) + 1) * span);
        nTiles = (dim + tileCols - 1) / tileCols;
    }

    __aicore__ inline void PrimePipeline() {
        if constexpr (PIPELINED) {
            // Each event denotes a free L0 slot. Prime both before the first
            // load; Mmad releases its slot for the next alternating load.
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(4);
        }
        if constexpr (PREFETCH || ROW_PREFETCH) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        }
        cachedCol = UINT32_MAX;
        if constexpr (ROW_PREFETCH) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            CopyProjectionChunk<T, PACKED_N>(aL1, bL1, x, w1, w2,
                static_cast<uint64_t>(core / nTiles * tileRows) * dim,
                static_cast<uint64_t>(core % nTiles * tileCols) * dim,
                0, dim, dim, tileRows, tileCols, tileRows, tileCols, true);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        }
    }

    __aicore__ inline void InitializeSchedule() {
        // The large full-row class has at least four complete waves, so
        // its first X/weight prefetch does not depend on the total row count.
        // Read scheduling rows after issuing that asynchronous MTE2 work.
        totalRows = LAZY_TILING ?
 *reinterpret_cast<volatile __gm__ uint32_t *>(tiling) : t.rows;
            schedule = MakeProjectionSchedule<BALANCED_TAIL, STATIC_TAIL_ROWS>(totalRows, t.cores, nTiles, tileRows);
        iteration = 0;
    }

    __aicore__ inline void FinishPipeline() {
        if constexpr (!SINGLE_TILE) {
            if (pending[0]) AscendC::CrossCoreWaitFlag(9);
            if constexpr (DOUBLE_GM) {
                if (pending[1]) AscendC::CrossCoreWaitFlag(11);
            }
        }
        if constexpr (PIPELINED) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(4);
        }
        if constexpr (PREFETCH || ROW_PREFETCH) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        }
        if constexpr (!SINGLE_TILE) AscendC::PipeBarrier<PIPE_ALL>();
        if constexpr (UNIT_FLOW) AscendC::SetMMColumnMajor();
    }

};

template <typename T, bool PIPELINED = false, bool PREFETCH = false,
    bool SINGLE_TILE = false, uint32_t PIPE_K_STEP = 64, uint32_t STATIC_DIM = 0, uint32_t STATIC_ROWS = 0,
    bool BALANCED_TAIL = false, uint32_t STATIC_TAIL_ROWS = 0, bool COMPLETE_TAIL = false, bool LAZY_TILING = false>
__aicore__ inline void ComputeProjections(GM_ADDR input, GM_ADDR weight1,
    GM_ADDR weight2, GM_ADDR workspace, const SwiGluVectorTilingData &t, GM_ADDR tiling = nullptr) {
    if ASCEND_IS_AIC {
        ProjectionComputer<T, PIPELINED, PREFETCH, SINGLE_TILE, PIPE_K_STEP, STATIC_DIM, STATIC_ROWS,
            BALANCED_TAIL, STATIC_TAIL_ROWS, COMPLETE_TAIL, LAZY_TILING> op(input, weight1, weight2, workspace, t, tiling);
        op.Run();
    }
}

// GM handles shared by the complete-K cube schedules.
template<typename T>
struct ProjectionIo {
    GM_ADDR input, weight1, weight2, workspace;
    AscendC::GlobalTensor<T> x, w1, w2, result;
    uint32_t core;

    __aicore__ ProjectionIo(GM_ADDR ix, GM_ADDR iw1, GM_ADDR iw2, GM_ADDR ws)
        : input(ix), weight1(iw1), weight2(iw2), workspace(ws) {}
};

template <typename T, uint32_t D, uint32_t N, bool FULL_ROW_SINGLE_VECTOR = false>
class ComputeFullKTileTask {
public:
        __aicore__ ComputeFullKTileTask(GM_ADDR inputArg, GM_ADDR weight1Arg, GM_ADDR weight2Arg, GM_ADDR workspaceArg, const SwiGluVectorTilingData &tArg) : io(inputArg, weight1Arg, weight2Arg, workspaceArg), t(tArg) {}
    __aicore__ inline void Run() {

        io.core = AscendC::GetBlockIdx();
        row = io.core / (D / N) * t.tileRows;
        col = io.core % (D / N) * N;
        rows = FULL_ROW_SINGLE_VECTOR ? t.tileRows : (t.rows - row < t.tileRows ? t.rows - row : t.tileRows);
        ma = (rows + 15) / 16 * 16;
        span = t.tileRows * N;

        io.x.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(io.input));
        io.w1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(io.weight1));
        io.w2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(io.weight2));
        io.result.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(io.workspace) + io.core * 2 * span);
        aL1 = AscendC::LocalTensor<T>(AscendC::TPosition::A1, 0, t.tileRows * D);
        bL1 = AscendC::LocalTensor<T>(AscendC::TPosition::A1, 131072, 2 * N * D);
        aL0 = AscendC::LocalTensor<T>(AscendC::TPosition::A2, 0, t.tileRows * D);
        bL0 = AscendC::LocalTensor<T>(AscendC::TPosition::B2, 0, N * D);
        cL0 = AscendC::LocalTensor<float>(AscendC::TPosition::CO1, 0, t.tileRows * N);
        AscendC::SetAtomicNone();
        AscendC::SetMMRowMajor();
        AscendC::SetLoadDataBoundary(0);
        AscendC::SetLoadDataPaddingValue(static_cast<uint64_t>(0));
        LoadInputs();
        for (projection = 0; projection < 2; ++projection) {
            if (projection != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(1);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
            }
            ComputeProjection();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SetMMColumnMajor();
    }
private:
    ProjectionIo<T> io;

    const SwiGluVectorTilingData &t;
    uint32_t row;
    uint32_t col;
    uint32_t rows;
    uint32_t ma;
    uint32_t span;
    AscendC::LocalTensor<T> aL1;
    AscendC::LocalTensor<T> bL1;
    AscendC::LocalTensor<T> aL0;
    AscendC::LocalTensor<T> bL0;
    AscendC::LocalTensor<float> cL0;
    AscendC::Nd2NzParams copy;
    AscendC::LoadData3DParamsV2<T> loadA;
    uint32_t projection;
    AscendC::LoadData2dParams loadB;
    AscendC::MmadParams mad;
    AscendC::FixpipeParamsV220 fix;
    __aicore__ inline void LoadInputs() {
        copy = AscendC::Nd2NzParams{};
        copy.ndNum = 1;
        copy.nValue = rows;
        copy.dValue = D;
        copy.srcDValue = D;
        copy.dstNzC0Stride = ma;
        copy.dstNzNStride = 1;
        AscendC::DataCopy(aL1, io.x[static_cast<uint64_t>(row) * D], copy);
        copy.nValue = N;
        copy.dstNzC0Stride = N;
        AscendC::DataCopy(bL1, io.w1[col * D], copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::DataCopy(bL1[N * D], io.w2[col * D], copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        loadA = AscendC::LoadData3DParamsV2<T>{};
        loadA.l1H = 1;
        loadA.l1W = ma;
        loadA.channelSize = D;
        loadA.kExtension = D;
        loadA.mExtension = ma;
        loadA.strideW = loadA.strideH = 1;
        loadA.filterW = loadA.filterH = 1;
        loadA.dilationFilterW = loadA.dilationFilterH = 1;
        AscendC::LoadData(aL0, aL1, loadA);
    }

    __aicore__ inline void ComputeProjection() {
        loadB = AscendC::LoadData2dParams{};
        loadB.repeatTimes = N * D / 256;
        loadB.srcStride = 1;
        AscendC::LoadData(bL0, bL1[projection * N * D], loadB);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
        mad = AscendC::MmadParams{};
        mad.m = ma;
        mad.n = N;
        mad.k = D;
        mad.unitFlag = 3;
        mad.cmatrixInitVal = true;
        AscendC::Mmad(cL0, aL0, bL0, mad);
        fix = AscendC::FixpipeParamsV220{};
        fix.nSize = N;
        fix.mSize = ma;
        fix.srcStride = ma;
        fix.dstStride = N;
        fix.ndNum = 1;
        fix.quantPre = AscendC::QuantMode_t::F322F16;
        fix.unitFlag = 3;
        AscendC::Fixpipe(io.result[projection * span], cL0, fix);
        if(projection==0) AscendC::CrossCoreSetFlag<2,PIPE_FIX>(8);
        else AscendC::CrossCoreSetFlag<2,PIPE_FIX>(10);
    }

};

template <typename T, uint32_t D, uint32_t N, bool FULL_ROW_SINGLE_VECTOR = false>
__aicore__ inline void ComputeFullKTile(GM_ADDR input, GM_ADDR weight1,
    GM_ADDR weight2, GM_ADDR workspace, const SwiGluVectorTilingData &t) {
    if ASCEND_IS_AIC{
        ComputeFullKTileTask<T, D, N, FULL_ROW_SINGLE_VECTOR> op(input, weight1, weight2, workspace, t);
        op.Run();
    }
}

template <typename T, bool FULL_ROW_SINGLE_VECTOR = false>
__aicore__ inline void ComputeSplitFullK(GM_ADDR input, GM_ADDR weight1,
    GM_ADDR weight2, GM_ADDR workspace, const SwiGluVectorTilingData &t) {
    ComputeFullKTile<T, 256, 128, FULL_ROW_SINGLE_VECTOR>(input, weight1, weight2, workspace, t);
}

template <typename T, uint32_t D, uint32_t N, bool FULL_ROW_SINGLE_VECTOR = false, bool ROW_WAVE = false>
class GateFullKTileTask {
public:
        __aicore__ GateFullKTileTask(GM_ADDR bias1Arg, GM_ADDR bias2Arg, GM_ADDR outputArg, GM_ADDR workspaceArg, const SwiGluVectorTilingData &tArg) : bias1(bias1Arg), bias2(bias2Arg), output(outputArg), workspace(workspaceArg), t(tArg) {}
    __aicore__ inline void Run() {

        core =FULL_ROW_SINGLE_VECTOR?AscendC::GetBlockIdx():AscendC::GetBlockIdx()/(ROW_WAVE ? 2 : AscendC::GetTaskRation());
        row =core/(D/N)*t.tileRows;
        col =core%(D/N)*N;
        rows =FULL_ROW_SINGLE_VECTOR?t.tileRows:(t.rows-row<t.tileRows?t.rows-row:t.tileRows);
        halfRows =FULL_ROW_SINGLE_VECTOR?rows:(rows+(ROW_WAVE ? 2 : AscendC::GetTaskRation())-1)/(ROW_WAVE ? 2 : AscendC::GetTaskRation());
        begin =FULL_ROW_SINGLE_VECTOR?0:AscendC::GetSubBlockIdx()*halfRows;
        nr =FULL_ROW_SINGLE_VECTOR?rows:(begin>=rows?0:(rows-begin<halfRows?rows-begin:halfRows));
        count =nr*N;
        span =t.tileRows*N;

        first.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*2*span);
        second.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*2*span+span);
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        y.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        a = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC,0,16384);
        b = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC,32768,16384);
        out = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC,65536,16384);
        bias = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC,98304,256);
        noPad = AscendC::DataCopyPadExtParams<T>{false,0,0,0};
        broadcast = AscendC::BinaryRepeatParams{1,1,1,8,8,0};
        unary = AscendC::UnaryRepeatParams{1,1,8,8};
        binary = AscendC::BinaryRepeatParams{1,1,1,8,8,8};
        AscendC::SetMaskNorm();
        if(nr) CopyTensorPair(bias,N,b1[col],b2[col],N,noPad);
        AscendC::CrossCoreWaitFlag(8);
        GateFirst();
        AscendC::CrossCoreWaitFlag(10);
        GateSecond();

    }
private:
        GM_ADDR bias1;
    GM_ADDR bias2;
    GM_ADDR output;
    GM_ADDR workspace;
    const SwiGluVectorTilingData &t;
    AscendC::GlobalTensor<T> first;
    AscendC::GlobalTensor<T> second;
    AscendC::GlobalTensor<T> b1;
    AscendC::GlobalTensor<T> b2;
    AscendC::GlobalTensor<T> y;
    uint32_t core;
    uint32_t row;
    uint32_t col;
    uint32_t rows;
    uint32_t halfRows;
    uint32_t begin;
    uint32_t nr;
    uint32_t count;
    uint32_t span;
    AscendC::LocalTensor<T> a;
    AscendC::LocalTensor<T> b;
    AscendC::LocalTensor<T> out;
    AscendC::LocalTensor<T> bias;
    AscendC::DataCopyPadExtParams<T> noPad;
    AscendC::BinaryRepeatParams broadcast;
    AscendC::UnaryRepeatParams unary;
    AscendC::BinaryRepeatParams binary;
    __aicore__ inline void GateFirst() {
        if(nr) {
            AscendC::DataCopyPad(a,first[begin*N],AscendC::DataCopyExtParams{1,count*2,0,0,0},noPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::Add(a,a,bias,N,nr,broadcast);
            AscendC::PipeBarrier<PIPE_V>();
            if(count>128) {
                AscendC::SetMaskCount(); AscendC::SetVectorMask<T,AscendC::MaskMode::COUNTER>(count);
            } else AscendC::SetVectorMask<T>(count);
            AscendC::Muls<T,false>(out,a,static_cast<T>(-1),AscendC::MASK_PLACEHOLDER,1,unary);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp<T,false>(out,out,AscendC::MASK_PLACEHOLDER,1,unary);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds<T,false>(out,out,static_cast<T>(1),AscendC::MASK_PLACEHOLDER,1,unary);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div<T,false>(out,a,out,AscendC::MASK_PLACEHOLDER,1,binary);
            // B is independent of this division. Its later Add/PIPE_V fence
            // joins both operations before Mul consumes their results.
            if(count>128) AscendC::SetMaskNorm();
        }
    }

    __aicore__ inline void GateSecond() {
        if(nr) {
            AscendC::DataCopyPad(b,second[begin*N],AscendC::DataCopyExtParams{1,count*2,0,0,0},noPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
            AscendC::Add(b,b,bias[N],N,nr,broadcast);
            AscendC::PipeBarrier<PIPE_V>();
            if(count>128) {
                AscendC::SetMaskCount(); AscendC::SetVectorMask<T,AscendC::MaskMode::COUNTER>(count);
            } else AscendC::SetVectorMask<T>(count);
            AscendC::Mul<T,false>(out,out,b,AscendC::MASK_PLACEHOLDER,1,binary);
            if(count>128) AscendC::SetMaskNorm();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            if constexpr (ROW_WAVE && D==N) AscendC::DataCopyPad(y[static_cast<uint64_t>(row+begin)*D],out,AscendC::DataCopyExtParams{1,count*2,0,0,0});
            else AscendC::DataCopyPad(y[static_cast<uint64_t>(row+begin)*D+col],out,
                AscendC::DataCopyExtParams{static_cast<uint16_t>(nr),N*2,0,(D-N)*2,0});
        }    }

};

template <typename T, uint32_t D, uint32_t N, bool FULL_ROW_SINGLE_VECTOR = false, bool ROW_WAVE = false>
__aicore__ inline void GateFullKTile(GM_ADDR bias1,GM_ADDR bias2,GM_ADDR output,
    GM_ADDR workspace,const SwiGluVectorTilingData &t) {
    if ASCEND_IS_AIV{
        GateFullKTileTask<T, D, N, FULL_ROW_SINGLE_VECTOR, ROW_WAVE> op(bias1, bias2, output, workspace, t);
        op.Run();
    }
}

template <typename T, bool FULL_ROW_SINGLE_VECTOR = false>
__aicore__ inline void GateSplitFullK(GM_ADDR bias1, GM_ADDR bias2, GM_ADDR output,
    GM_ADDR workspace, const SwiGluVectorTilingData &t) {
    GateFullKTile<T, 256, 128, FULL_ROW_SINGLE_VECTOR>(bias1, bias2, output, workspace, t);
}

template <typename T, uint32_t D = 256, uint32_t N = 128>
__aicore__ inline void ComputeSingleRowWave(GM_ADDR input, GM_ADDR weight1,
    GM_ADDR weight2, GM_ADDR workspace, const SwiGluVectorTilingData &t) {
    ComputeFullKTile<T, D, N>(input, weight1, weight2, workspace, t);
}

template <typename T, uint32_t D = 256, uint32_t N = 128>
__aicore__ inline void GateSingleRowWave(GM_ADDR bias1, GM_ADDR bias2, GM_ADDR output,
    GM_ADDR workspace, const SwiGluVectorTilingData &t) {
    GateFullKTile<T, D, N, false, true>(bias1, bias2, output, workspace, t);
}

template <typename T>
class ComputeDualC0TwoWaveTask {
public:
        __aicore__ ComputeDualC0TwoWaveTask(GM_ADDR inputArg, GM_ADDR weight1Arg, GM_ADDR weight2Arg, GM_ADDR workspaceArg, const SwiGluVectorTilingData &tArg) : io(inputArg, weight1Arg, weight2Arg, workspaceArg), t(tArg) {}
    __aicore__ inline void Run() {

        io.core =AscendC::GetBlockIdx();
        remaining =t.rows-t.cores*M;
        tailRows =(remaining+t.cores*16-1)/(t.cores*16)*16;
        tailStart =t.cores*M+io.core*tailRows;
        tailNr =tailStart<t.rows?(t.rows-tailStart<tailRows?t.rows-tailStart:tailRows):0;
        hasTail =tailNr!=0;

        io.x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(io.input));
        io.w1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(io.weight1));
        io.w2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(io.weight2));
        io.result.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(io.workspace)+io.core*4*SPAN);
        aL1 = AscendC::LocalTensor<T>(AscendC::TPosition::A1,0,2*SPAN);
        bL1 = AscendC::LocalTensor<T>(AscendC::TPosition::A1,131072,2*D*D);
        aL0 = AscendC::LocalTensor<T>(AscendC::TPosition::A2,0,2*SPAN);
        bL0 = AscendC::LocalTensor<T>(AscendC::TPosition::B2,0,2*D*D);
        cL0 = AscendC::LocalTensor<float>(AscendC::TPosition::CO1,0,2*SPAN);
        LoadWaveInputs();
        for(projection =0;projection<2;++projection) AscendC::SetFlag<AscendC::HardEvent::FIX_M>(6+projection);
        for(wave =0;wave<(hasTail?2u:1u);++wave) {
            rows =wave?tailNr:M;
            ma =(rows+15)/16*16;
            LoadWaveA();
            for(projection =0;projection<2;++projection) {
                ComputeWaveProjection();
            }
        }
        for(projection =0;projection<2;++projection) AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(6+projection);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
private:
    ProjectionIo<T> io;
        static constexpr uint32_t D=128,M=128,SPAN=M*D;
    const SwiGluVectorTilingData &t;
    uint32_t remaining;
    uint32_t tailRows;
    uint32_t tailStart;
    uint32_t tailNr;
    bool hasTail;
    AscendC::LocalTensor<T> aL1;
    AscendC::LocalTensor<T> bL1;
    AscendC::LocalTensor<T> aL0;
    AscendC::LocalTensor<T> bL0;
    AscendC::LocalTensor<float> cL0;
    AscendC::Nd2NzParams copy;
    AscendC::LoadData2dParams loadB;
    uint32_t projection;
    uint32_t wave;
    uint32_t rows;
    uint32_t ma;
    AscendC::LoadData3DParamsV2<T> loadA;
    AscendC::MmadParams mad;
    AscendC::FixpipeParamsV220 fix;
    __aicore__ inline void LoadWaveInputs() {
        AscendC::SetAtomicNone(); AscendC::SetMMColumnMajor();
        AscendC::SetLoadDataBoundary(0); AscendC::SetLoadDataPaddingValue(static_cast<uint64_t>(0));
        copy = AscendC::Nd2NzParams{};
        copy.ndNum=1; copy.nValue=M; copy.dValue=D; copy.srcDValue=D;
        copy.dstNzC0Stride=M; copy.dstNzNStride=1;
        AscendC::DataCopy(aL1,io.x[static_cast<uint64_t>(io.core*M)*D],copy);
        copy.nValue=D; copy.dstNzC0Stride=D;
        AscendC::DataCopy(bL1,io.w1,copy); AscendC::DataCopy(bL1[D*D],io.w2,copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        if(hasTail) {
            copy.nValue=tailNr; copy.dstNzC0Stride=(tailNr+15)/16*16;
            AscendC::DataCopy(aL1[SPAN],io.x[static_cast<uint64_t>(tailStart)*D],copy);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        loadB = AscendC::LoadData2dParams{};
        loadB.repeatTimes=2*D*D/256; loadB.srcStride=1;
        AscendC::LoadData(bL0,bL1,loadB);
    }

    __aicore__ inline void LoadWaveA() {
        if(wave) AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(1);
        loadA = AscendC::LoadData3DParamsV2<T>{};
        loadA.l1H=1; loadA.l1W=ma; loadA.channelSize=D;
        loadA.kExtension=D; loadA.mExtension=ma;
        loadA.strideW=loadA.strideH=1;
        loadA.filterW=loadA.filterH=1;
        loadA.dilationFilterW=loadA.dilationFilterH=1;
        AscendC::LoadData(aL0[wave*SPAN],aL1[wave*SPAN],loadA);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    }

    __aicore__ inline void ComputeWaveProjection() {
        // Each C0 projection is reused only by the second wave.
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(6+projection);
        mad = AscendC::MmadParams{};
        mad.m=ma; mad.n=D; mad.k=D; mad.cmatrixInitVal=true;
        AscendC::Mmad(cL0[projection*SPAN],aL0[wave*SPAN],bL0[projection*D*D],mad);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(6+projection); AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(6+projection);
        fix = AscendC::FixpipeParamsV220{};
        fix.nSize=D; fix.mSize=ma; fix.srcStride=ma; fix.dstStride=D;
        fix.ndNum=1; fix.quantPre=AscendC::QuantMode_t::F322F16;
        AscendC::Fixpipe(io.result[(2*wave+projection)*SPAN],cL0[projection*SPAN],fix);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(6+projection);
        if(!wave) {
            if(!projection) AscendC::CrossCoreSetFlag<2,PIPE_FIX>(8);
            else AscendC::CrossCoreSetFlag<2,PIPE_FIX>(10);
        } else {
            if(!projection) AscendC::CrossCoreSetFlag<2,PIPE_FIX>(12);
            else AscendC::CrossCoreSetFlag<2,PIPE_FIX>(14);
        }
    }

};

template <typename T>
__aicore__ inline void ComputeDualC0TwoWave(GM_ADDR input,GM_ADDR weight1,GM_ADDR weight2,
    GM_ADDR workspace,const SwiGluVectorTilingData &t) {
    if ASCEND_IS_AIC{
        ComputeDualC0TwoWaveTask<T> op(input, weight1, weight2, workspace, t);
        op.Run();
    }
}

template <typename T>
class GateDualC0TwoWaveTask {
public:
        __aicore__ GateDualC0TwoWaveTask(GM_ADDR bias1Arg, GM_ADDR bias2Arg, GM_ADDR outputArg, GM_ADDR workspaceArg, const SwiGluVectorTilingData &tArg) : bias1(bias1Arg), bias2(bias2Arg), output(outputArg), workspace(workspaceArg), t(tArg) {}
    __aicore__ inline void Run() {

        core =AscendC::GetBlockIdx()/AscendC::GetTaskRation();
        remaining =t.rows-t.cores*M;
        tailRows =(remaining+t.cores*16-1)/(t.cores*16)*16;

        source.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*4*SPAN);
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        y.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        bias = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC,98304,256);
        pad = AscendC::DataCopyPadExtParams<T>{false,0,0,0};
        CopyTensorPair(bias,D,b1,b2,D,pad);
        broadcast = AscendC::BinaryRepeatParams{1,1,1,8,8,0};
        unary = AscendC::UnaryRepeatParams{1,1,8,8};
        binary = AscendC::BinaryRepeatParams{1,1,1,8,8,8};
        AscendC::SetMaskNorm();
        tailStart =t.cores*M+core*tailRows;
        hasTail =tailStart<t.rows;
        for(wave =0;wave<(hasTail?2u:1u);++wave) {
            row =wave?tailStart:core*M;
            limit =wave?tailRows:M;
            rows =t.rows-row<limit?t.rows-row:limit;
            halfRows =(rows+AscendC::GetTaskRation()-1)/AscendC::GetTaskRation();
            begin =AscendC::GetSubBlockIdx()*halfRows;
            nr =begin>=rows?0:(rows-begin<halfRows?rows-begin:halfRows);
            count =nr*D;
            a = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC,wave*16384,8192);
            b = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC,32768+wave*16384,8192);
            out = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC,65536+wave*16384,8192);
            if(!wave) AscendC::CrossCoreWaitFlag(8); else AscendC::CrossCoreWaitFlag(12);
            GateFirst();
            if(!wave) AscendC::CrossCoreWaitFlag(10); else AscendC::CrossCoreWaitFlag(14);
            GateSecond();

        }
    }
private:
        static constexpr uint32_t D=128,M=128,SPAN=M*D;
    GM_ADDR bias1;
    GM_ADDR bias2;
    GM_ADDR output;
    GM_ADDR workspace;
    const SwiGluVectorTilingData &t;
    AscendC::GlobalTensor<T> source;
    AscendC::GlobalTensor<T> b1;
    AscendC::GlobalTensor<T> b2;
    AscendC::GlobalTensor<T> y;
    uint32_t core;
    uint32_t remaining;
    uint32_t tailRows;
    AscendC::LocalTensor<T> bias;
    AscendC::DataCopyPadExtParams<T> pad;
    AscendC::BinaryRepeatParams broadcast;
    AscendC::UnaryRepeatParams unary;
    AscendC::BinaryRepeatParams binary;
    uint32_t tailStart;
    bool hasTail;
    uint32_t wave;
    uint32_t row;
    uint32_t limit;
    uint32_t rows;
    uint32_t halfRows;
    uint32_t begin;
    uint32_t nr;
    uint32_t count;
    AscendC::LocalTensor<T> a;
    AscendC::LocalTensor<T> b;
    AscendC::LocalTensor<T> out;
    __aicore__ inline void GateFirst() {
        if(nr) {
            AscendC::DataCopyPad(a,source[2*wave*SPAN+begin*D],AscendC::DataCopyExtParams{1,count*2,0,0,0},pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::Add(a,a,bias,D,nr,broadcast); AscendC::PipeBarrier<PIPE_V>();
            if(count>128) {AscendC::SetMaskCount(); AscendC::SetVectorMask<T,AscendC::MaskMode::COUNTER>(count);}
            else AscendC::SetVectorMask<T>(count);
            AscendC::Muls<T,false>(out,a,static_cast<T>(-1),AscendC::MASK_PLACEHOLDER,1,unary); AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp<T,false>(out,out,AscendC::MASK_PLACEHOLDER,1,unary); AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds<T,false>(out,out,static_cast<T>(1),AscendC::MASK_PLACEHOLDER,1,unary); AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div<T,false>(out,a,out,AscendC::MASK_PLACEHOLDER,1,binary); AscendC::PipeBarrier<PIPE_V>();
            if(count>128) AscendC::SetMaskNorm();
        }
    }

    __aicore__ inline void GateSecond() {
        if(nr) {
            AscendC::DataCopyPad(b,source[(2*wave+1)*SPAN+begin*D],AscendC::DataCopyExtParams{1,count*2,0,0,0},pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
            AscendC::Add(b,b,bias[D],D,nr,broadcast); AscendC::PipeBarrier<PIPE_V>();
            if(count>128) {AscendC::SetMaskCount(); AscendC::SetVectorMask<T,AscendC::MaskMode::COUNTER>(count);}
            else AscendC::SetVectorMask<T>(count);
            AscendC::Mul<T,false>(out,out,b,AscendC::MASK_PLACEHOLDER,1,binary);
            if(count>128) AscendC::SetMaskNorm();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::DataCopyPad(y[static_cast<uint64_t>(row+begin)*D],out,AscendC::DataCopyExtParams{1,count*2,0,0,0});
        }    }

};

template <typename T>
__aicore__ inline void GateDualC0TwoWave(GM_ADDR bias1,GM_ADDR bias2,GM_ADDR output,
    GM_ADDR workspace,const SwiGluVectorTilingData &t) {
    if ASCEND_IS_AIV{
        GateDualC0TwoWaveTask<T> op(bias1, bias2, output, workspace, t);
        op.Run();
    }
}

template <typename T, bool SINGLE_TILE = false, uint32_t STATIC_DIM = 0, uint32_t STATIC_ROWS = 0,
    bool BALANCED_TAIL = false, uint32_t STATIC_TAIL_ROWS = 0, bool COMPLETE_TAIL = false, bool LAZY_TILING = false>
class GateProjections {
    static constexpr bool DOUBLE_GM = !SINGLE_TILE && (STATIC_DIM == 128 || STATIC_DIM == 256);
    static constexpr bool FULL_ROW_TILE = STATIC_DIM == 256 && STATIC_ROWS == 128;
public:
        __aicore__ inline void Init(GM_ADDR bias1, GM_ADDR bias2, GM_ADDR output,
        GM_ADDR workspace, const SwiGluVectorTilingData &data, GM_ADDR tiling = nullptr) {
        t = data;
        tilingRows = tiling;
        const uint32_t tileRows = STATIC_ROWS == 0 ? t.tileRows : STATIC_ROWS;
        const uint32_t tileCols = STATIC_DIM == 0 ? t.tileCols : (STATIC_DIM < 128 ? (STATIC_DIM + 15) / 16 * 16 : 128);
        const uint32_t span = tileRows * tileCols;
        core = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
        sub = AscendC::GetSubBlockIdx();
        first.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + core * (DOUBLE_GM ? 4 : 2) * span);
        const uint32_t secondOffset = FULL_ROW_TILE ? tileCols : (STATIC_DIM == 33 && STATIC_ROWS == 16 ? STATIC_DIM : span);
        second.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace) + core * (DOUBLE_GM ? 4 : 2) * span + secondOffset);
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
        if constexpr (STATIC_DIM == 128 || STATIC_DIM == 256) {
            y.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        stepRows = STATIC_DIM == 256 && SINGLE_TILE ? 128 : 64;
        if constexpr (LAZY_TILING) {
            // A full-row core retains one column stripe for all waves.
            AscendC::LocalTensor<T> bias(AscendC::TPosition::VECCALC, 98304, 2 * tileCols);
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            const uint32_t preloadCol = core % 2 * tileCols;
            CopyTensorPair(bias, tileCols, b1[preloadCol], b2[preloadCol], tileCols, pad);
        }
    }
    __aicore__ inline void Process() {
        InitializeSchedule();
        for (tile = core; SINGLE_TILE || tile < schedule.tiles; tile += t.cores, ++iteration) {
            PrepareTile();
            AcquireTile();
            ProcessRows();
            ReleaseTile();
            if constexpr (SINGLE_TILE) break;
        }
    }

private:
        ProjectionTileGeometry geometry;
    ProjectionSchedule schedule;
    uint32_t tileRows, dim, tileCols, sourcePitch, nTiles, totalRows, cachedCol, iteration, tile, halfRows, begin, end, inputSlot, inputOffset, rr, nr, count;

    AscendC::LocalTensor<T> bias;
    AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    __aicore__ inline void InitializeSchedule() {
        // A previous kernel can leave counter-mask mode active on an AIV.
        // The row-wise bias adds below use normal repeat masks.
        AscendC::SetMaskNorm();
        tileRows = STATIC_ROWS == 0 ? t.tileRows : STATIC_ROWS;
        dim = STATIC_DIM == 0 ? t.dim : STATIC_DIM;
        tileCols = STATIC_DIM == 0 ? t.tileCols : (STATIC_DIM < 128 ? (STATIC_DIM + 15) / 16 * 16 : 128);
        sourcePitch = FULL_ROW_TILE ? 2 * tileCols : (STATIC_DIM == 33 && STATIC_ROWS == 16 ? 80 : tileCols);
        nTiles = (dim + tileCols - 1) / tileCols;
        totalRows = LAZY_TILING ?
 *reinterpret_cast<volatile __gm__ uint32_t *>(tilingRows) : t.rows;
            schedule = MakeProjectionSchedule<BALANCED_TAIL, STATIC_TAIL_ROWS>(totalRows, t.cores, nTiles, tileRows);
        cachedCol = LAZY_TILING ? core % nTiles * tileCols : UINT32_MAX;
        iteration = 0;
    }

    __aicore__ inline void PrepareTile() {
        geometry = MakeProjectionTile<BALANCED_TAIL, FULL_ROW_TILE, STATIC_TAIL_ROWS, COMPLETE_TAIL>(
            tile, core, tileRows, tileCols, nTiles, totalRows, dim, schedule);
        halfRows = COMPLETE_TAIL ? geometry.activeTileRows / AscendC::GetTaskRation() :
            (FULL_ROW_TILE && (!BALANCED_TAIL || (STATIC_TAIL_ROWS != 0 && !geometry.tail)) ? 128 / AscendC::GetTaskRation() : (geometry.rows + AscendC::GetTaskRation() - 1) / AscendC::GetTaskRation());
        begin = sub * halfRows;
        end = COMPLETE_TAIL ? begin + halfRows : (begin + halfRows < geometry.rows ? begin + halfRows : geometry.rows);
    }

    __aicore__ inline void AcquireTile() {
        bias = AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC, 98304, 2 * tileCols);
        if (cachedCol != geometry.col) {
            CopyTensorPair(bias, tileCols, b1[geometry.col], b2[geometry.col], geometry.cols, pad);
            cachedCol = geometry.col;
        }
        inputSlot = DOUBLE_GM ? (iteration & 1U) : 0;
        inputOffset = inputSlot * 2 * tileRows * tileCols;
        AscendC::CrossCoreWaitFlag(8 + 2 * inputSlot);
    }

    __aicore__ inline void ReleaseTile() {
        if constexpr (!SINGLE_TILE) {
            // The next input may arrive while the current output is stored.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        }
        if constexpr (!SINGLE_TILE && !COMPLETE_TAIL) {
            if (begin >= end) AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(9 + 2 * inputSlot);
        }
    }

    __aicore__ inline void ProcessRows() {
        for (rr = begin; (FULL_ROW_TILE && COMPLETE_TAIL) || rr < end; rr += stepRows) {
            nr = COMPLETE_TAIL ? geometry.activeTileRows / AscendC::GetTaskRation() :
                (FULL_ROW_TILE && (!BALANCED_TAIL || (STATIC_TAIL_ROWS != 0 && !geometry.tail)) ? 128 / AscendC::GetTaskRation() : (end - rr < stepRows ? end - rr : stepRows));
            count = nr * tileCols;
            if (rr != begin) AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::LocalTensor<T> pair(AscendC::TPosition::VECCALC, 0, 2 * stepRows * tileCols);
            auto a = pair;
            auto b = pair[FULL_ROW_TILE ? tileCols : stepRows * tileCols];
            LoadRows(pair, a, b);
            // One chunk belongs to each vector subcore. Once both
            // reads finish, the cube may reuse GM while activation runs.
            if constexpr (!SINGLE_TILE) AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(9 + 2 * inputSlot);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            a = pair;
            b = pair[FULL_ROW_TILE ? tileCols : stepRows * tileCols];
            ActivateRows(pair, a, b);
            if constexpr (FULL_ROW_TILE) break;
        }
    }

    __aicore__ inline void LoadRows(AscendC::LocalTensor<T> pair, AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b) {
        if constexpr (FULL_ROW_TILE) {
            AscendC::DataCopyExtParams flat{1, 2 * count * 2, 0, 0, 0};
            AscendC::DataCopyPad(pair, first[inputOffset + rr * sourcePitch], flat, pad);
        } else {
            AscendC::DataCopyExtParams inputCopy{static_cast<uint16_t>(nr), geometry.cols * 2,
                    (sourcePitch - geometry.cols) * 2,
                    (tileCols - ((geometry.cols + 15) / 16) * 16) / 16, 0};
            if (geometry.cols == tileCols) {
                inputCopy.blockCount = 1;
                inputCopy.blockLen = count * 2;
            }
            AscendC::DataCopyPad(a, first[inputOffset + rr * sourcePitch], inputCopy, pad);
            AscendC::DataCopyPad(b, second[inputOffset + rr * sourcePitch], inputCopy, pad);
        }
    }

    __aicore__ inline void ActivateRows(AscendC::LocalTensor<T> pair, AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b) {
        for (uint32_t c = 0; c < geometry.cols; c += 128) {
            uint32_t width = geometry.cols - c < 128 ? geometry.cols - c : 128;
            AscendC::BinaryRepeatParams params{1, 1, 1,
                    static_cast<uint8_t>((FULL_ROW_TILE ? 2 * tileCols : tileCols) / 16),
                    static_cast<uint8_t>((FULL_ROW_TILE ? 2 * tileCols : tileCols) / 16), 0};
            AscendC::Add(a[c], a[c], bias[c], width, nr, params);
            AscendC::Add(b[c], b[c], bias[tileCols + c], width, nr, params);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<T> out(AscendC::TPosition::VECCALC, 65536, stepRows * tileCols);
        if constexpr (FULL_ROW_TILE) HalfGateInterleaved128(out, pair, nr);
        else HalfGateSwiGlu(out, a, b, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopyExtParams outCopy{static_cast<uint16_t>(nr), geometry.cols * 2,
                (tileCols - ((geometry.cols + 15) / 16) * 16) / 16,
                (dim - geometry.cols) * 2, 0};
        if (geometry.cols == tileCols && geometry.cols == dim) {
            outCopy.blockCount = 1;
            outCopy.blockLen = count * 2;
        }
        AscendC::DataCopyPad(y[static_cast<uint64_t>(geometry.row + rr) * dim + geometry.col], out, outCopy);
    }


    SwiGluVectorTilingData t;
    GM_ADDR tilingRows;
    uint32_t core, sub, stepRows;
    AscendC::GlobalTensor<T> first, second, b1, b2, y;
};

// Complete full-K row tiles: two aligned waves use disjoint A0/B0/C0 slots.
template <typename T, uint32_t SLOT, uint16_t FLAG, bool REUSE, bool RELEASE>
__aicore__ inline void EmitProjectionTile(AscendC::LocalTensor<float> c,
    AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b,
    AscendC::GlobalTensor<T> result, uint32_t rows, uint32_t offset) {
    using namespace AscendC;
    if ASCEND_IS_AIC {
        if constexpr (REUSE) AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(SLOT);
        AscendC::MmadParams mad{};
        mad.m=rows;mad.n=128;mad.k=128;
        mad.unitFlag=3;mad.cmatrixInitVal=true;
        AscendC::Mmad(c,a,b,mad);
        // Paired unitFlag3 provides MMAD-to-Fixpipe data readiness per512B block.
        AscendC::FixpipeParamsV220 fix{};
        fix.nSize=128;fix.mSize=rows;fix.srcStride=rows;fix.dstStride=128;
        fix.ndNum=1;fix.quantPre=AscendC::QuantMode_t::F322F16;fix.unitFlag=3;
        AscendC::Fixpipe(result[offset],c,fix);
        AscendC::CrossCoreSetFlag<2,PIPE_FIX>(FLAG);
        if constexpr (RELEASE) AscendC::SetFlag<AscendC::HardEvent::FIX_M>(SLOT);
    }
}

template <typename T>
class ComputeProjectionWavesTask {
public:
        __aicore__ ComputeProjectionWavesTask(GM_ADDR inputArg, GM_ADDR weight1Arg, GM_ADDR weight2Arg, GM_ADDR workspaceArg) : io(inputArg, weight1Arg, weight2Arg, workspaceArg) {}
    __aicore__ inline void Run() {

        static_assert(2*ASLOT*sizeof(T)<=65536);
        static_assert(2*D*D*sizeof(T)<=65536);
        static_assert(2*CSLOT*sizeof(float)<=131072);
        io.core =AscendC::GetBlockIdx(); row =io.core*STEP;
        // Tiling selects only completeSTEP-row tiles; both waves are full.

        io.x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(io.input));
        io.w1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(io.weight1));
        io.w2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(io.weight2));
        io.result.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(io.workspace)+io.core*2*SPAN);
        aL1 = AscendC::LocalTensor<T>(AscendC::TPosition::A1,0,2*ASLOT);
        bL1 = AscendC::LocalTensor<T>(AscendC::TPosition::A1,131072,2*D*D);
        aL0 = AscendC::LocalTensor<T>(AscendC::TPosition::A2,0,2*ASLOT);
        bL0 = AscendC::LocalTensor<T>(AscendC::TPosition::B2,0,2*D*D);
        cL0 = AscendC::LocalTensor<float>(AscendC::TPosition::CO1,0,2*CSLOT);
        LoadWaveInputs();
        LoadWaveA();
        EmitProjectionTile<T,0,8,false,true>(cL0,aL0,bL0,io.result,r0,0);
        // The upper B0 half is independent of the first MMAD's live lower half.
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(1);
        AscendC::LoadData(bL0[D*D],bL1[D*D],loadB);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(1);
        EmitProjectionTile<T,1,10,false,true>(cL0[CSLOT],aL0,bL0[D*D],io.result,r0,SPAN);
        // The second A0 slot is likewise disjoint from the first row wave.
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(2);
        loadA.l1W=r1;loadA.mExtension=r1;
        AscendC::LoadData(aL0[ASLOT],aL1[ASLOT],loadA);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(2);
        EmitProjectionTile<T,0,12,true,false>(cL0,aL0[ASLOT],bL0,io.result,r1,r0*D);
        EmitProjectionTile<T,1,14,true,false>(cL0[CSLOT],aL0[ASLOT],bL0[D*D],io.result,r1,SPAN+r0*D);
        AscendC::PipeBarrier<PIPE_ALL>();AscendC::SetMMColumnMajor();
    }
private:
    ProjectionIo<T> io;
        static constexpr uint32_t D=128,STEP=256,ASLOT=128*D,CSLOT=128*D,SPAN=STEP*D;
    static constexpr uint32_t r0=128,r1=128;
    uint32_t row;
    AscendC::LocalTensor<T> aL1;
    AscendC::LocalTensor<T> bL1;
    AscendC::LocalTensor<T> aL0;
    AscendC::LocalTensor<T> bL0;
    AscendC::LocalTensor<float> cL0;
    AscendC::Nd2NzParams copy;
    AscendC::LoadData3DParamsV2<T> loadA;
    AscendC::LoadData2dParams loadB;
    __aicore__ inline void LoadWaveInputs() {
        AscendC::SetAtomicNone();AscendC::SetMMRowMajor();AscendC::SetLoadDataBoundary(0);
        AscendC::SetLoadDataPaddingValue(static_cast<uint64_t>(0));
        copy = AscendC::Nd2NzParams{};
        copy.ndNum=1;copy.nValue=r0;copy.dValue=D;copy.srcDValue=D;
        copy.dstNzC0Stride=r0;copy.dstNzNStride=1;
        AscendC::DataCopy(aL1,io.x[static_cast<uint64_t>(row)*D],copy);
        copy.nValue=D;copy.dstNzC0Stride=D;
        AscendC::DataCopy(bL1,io.w1,copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::DataCopy(bL1[D*D],io.w2,copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1);
        copy.nValue=r1;copy.dstNzC0Stride=r1;
        AscendC::DataCopy(aL1[ASLOT],io.x[static_cast<uint64_t>(row+r0)*D],copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
    }

    __aicore__ inline void LoadWaveA() {
        loadA = AscendC::LoadData3DParamsV2<T>{};
        loadA.l1H=1;loadA.l1W=r0;loadA.channelSize=D;
        loadA.kExtension=D;loadA.mExtension=r0;
        loadA.strideW=loadA.strideH=1;loadA.filterW=loadA.filterH=1;
        loadA.dilationFilterW=loadA.dilationFilterH=1;
        AscendC::LoadData(aL0,aL1,loadA);
        loadB = AscendC::LoadData2dParams{};
        loadB.repeatTimes=D*D/256;loadB.srcStride=1;
        AscendC::LoadData(bL0,bL1,loadB);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    }

};

template <typename T>
__aicore__ inline void ComputeProjectionWaves(GM_ADDR input,GM_ADDR weight1,
    GM_ADDR weight2,GM_ADDR workspace) {
    if ASCEND_IS_AIC{
        ComputeProjectionWavesTask<T> op(input, weight1, weight2, workspace);
        op.Run();
    }
}

template <typename T, uint32_t WAVE>
__aicore__ inline void GateProjectionWave(AscendC::GlobalTensor<T> first,
    AscendC::GlobalTensor<T> second,AscendC::GlobalTensor<T> y,
    AscendC::LocalTensor<T> bias,uint32_t row,uint32_t begin,uint32_t nr) {
    using namespace AscendC;
    constexpr uint32_t D=128;
    const uint32_t count=nr*D;
    AscendC::LocalTensor<T> a(AscendC::TPosition::VECCALC,WAVE*16384,8192);
    AscendC::LocalTensor<T> b(AscendC::TPosition::VECCALC,32768+WAVE*16384,8192);
    AscendC::LocalTensor<T> out(AscendC::TPosition::VECCALC,65536+WAVE*16384,8192);
    AscendC::DataCopyPadExtParams<T> pad{false,0,0,0};
    AscendC::BinaryRepeatParams broadcast{1,1,1,8,8,0},binary{1,1,1,8,8,8};
    AscendC::UnaryRepeatParams unary{1,1,8,8};
    AscendC::CrossCoreWaitFlag(8+4*WAVE);
    AscendC::DataCopyPad(a,first[begin*D],AscendC::DataCopyExtParams{1,count*2,0,0,0},pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::Add(a,a,bias,D,nr,broadcast);AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskCount();AscendC::SetVectorMask<T,AscendC::MaskMode::COUNTER>(count);
    AscendC::Muls<T,false>(out,a,static_cast<T>(-1),AscendC::MASK_PLACEHOLDER,1,unary);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp<T,false>(out,out,AscendC::MASK_PLACEHOLDER,1,unary);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<T,false>(out,out,static_cast<T>(1),AscendC::MASK_PLACEHOLDER,1,unary);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div<T,false>(out,a,out,AscendC::MASK_PLACEHOLDER,1,binary);
    // B addition is independent; its fence joins with the preceding division.
    AscendC::SetMaskNorm();
    AscendC::CrossCoreWaitFlag(10+4*WAVE);
    AscendC::DataCopyPad(b,second[begin*D],AscendC::DataCopyExtParams{1,count*2,0,0,0},pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
    AscendC::Add(b,b,bias[D],D,nr,broadcast);AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetMaskCount();AscendC::SetVectorMask<T,AscendC::MaskMode::COUNTER>(count);
    AscendC::Mul<T,false>(out,out,b,AscendC::MASK_PLACEHOLDER,1,binary);AscendC::SetMaskNorm();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopyPad(y[static_cast<uint64_t>(row+begin)*D],out,AscendC::DataCopyExtParams{1,count*2,0,0,0});
}

template <typename T>
__aicore__ inline void GateProjectionWaves(GM_ADDR bias1,GM_ADDR bias2,
    GM_ADDR output,GM_ADDR workspace) {
    using namespace AscendC;
    if ASCEND_IS_AIV {
        constexpr uint32_t D=128,STEP=256,SPAN=STEP*D;
        const uint32_t core=AscendC::GetBlockIdx()/2,row=core*STEP,sub=AscendC::GetSubBlockIdx();
        // Tiling selects only completeSTEP-row tiles; both waves are full.
        constexpr uint32_t r0=128,r1=128;
        AscendC::GlobalTensor<T> first,second,b1,b2,y;
        first.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*2*SPAN);
        second.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)+core*2*SPAN+SPAN);
        b1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias1));
        b2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias2));
        y.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        y.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        AscendC::LocalTensor<T> bias(AscendC::TPosition::VECCALC,98304,256);
        AscendC::DataCopyPadExtParams<T> noPad{false,0,0,0};
        AscendC::SetMaskNorm();CopyTensorPair(bias,D,b1,b2,D,noPad);
        // Each AV consumes flags8/10/12/14 exactly once. The two output slots
        // are disjoint, so first-wave MTE3 may overlap the second-wave gate.
        GateProjectionWave<T,0>(first,second,y,bias,row,sub*(r0/2),r0/2);
        GateProjectionWave<T,1>(first,second,y,bias,row,r0+sub*(r1/2),r1/2);
    }
}



template <typename T>
class ComputeSplit128EarlyFixTask {
public:
        __aicore__ ComputeSplit128EarlyFixTask(GM_ADDR inputArg, GM_ADDR weight1Arg, GM_ADDR weight2Arg, GM_ADDR workspaceArg) : io(inputArg, weight1Arg, weight2Arg, workspaceArg) {}
    __aicore__ inline void Run() {

        io.core =AscendC::GetBlockIdx(); row =io.core/2*M; col =io.core%2*N;

        io.x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(io.input));
        io.w1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(io.weight1));
        io.w2.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(io.weight2));
        io.result.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(io.workspace)+io.core*2*SPAN);
        a1 = AscendC::LocalTensor<T>(AscendC::TPosition::A1,0,M*D);
        b1 = AscendC::LocalTensor<T>(AscendC::TPosition::A1,131072,2*N*D);
        a0 = AscendC::LocalTensor<T>(AscendC::TPosition::A2,0,M*D);
        b0 = AscendC::LocalTensor<T>(AscendC::TPosition::B2,0,2*N*K);
        c0 = AscendC::LocalTensor<float>(AscendC::TPosition::CO1,0,2*SPAN);
        LoadInputs();
        ComputeProjection();
        // The release was set immediately after P1,K0, before P1,K1.
        // Loading P2,K0 therefore overlaps the independent slot1 MMAD.
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::LoadData(b0,b1[N*D],lb);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
        mad.unitFlag=2;mad.cmatrixInitVal=true;AscendC::Mmad(c0[SPAN],a0,b0,mad);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        // Distinct C0 storage lets P1 Fixpipe overlap P2 computation.
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::LoadData(b0[N*K],b1[N*D+N*K],lb);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(1);
        mad.unitFlag=3;mad.cmatrixInitVal=false;AscendC::Mmad(c0[SPAN],a0[M*K],b0[N*K],mad);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::Fixpipe(io.result[SPAN],c0[SPAN],fix);AscendC::CrossCoreSetFlag<2,PIPE_FIX>(10);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        AscendC::PipeBarrier<PIPE_ALL>(); AscendC::SetMMColumnMajor();
    }
private:
    ProjectionIo<T> io;
        static constexpr uint32_t M=64,N=128,D=256,K=128,SPAN=M*N;
    uint32_t row;
    uint32_t col;
    AscendC::LocalTensor<T> a1;
    AscendC::LocalTensor<T> b1;
    AscendC::LocalTensor<T> a0;
    AscendC::LocalTensor<T> b0;
    AscendC::LocalTensor<float> c0;
    AscendC::Nd2NzParams copy;
    AscendC::LoadData3DParamsV2<T> la;
    AscendC::LoadData2dParams lb;
    AscendC::MmadParams mad;
    AscendC::FixpipeParamsV220 fix;
    __aicore__ inline void LoadInputs() {
        AscendC::SetAtomicNone(); AscendC::SetMMRowMajor();
        AscendC::SetLoadDataBoundary(0); AscendC::SetLoadDataPaddingValue(static_cast<uint64_t>(0));
        copy = AscendC::Nd2NzParams{};
        copy.ndNum=1;copy.nValue=M;copy.dValue=D;copy.srcDValue=D;
        copy.dstNzC0Stride=M;copy.dstNzNStride=1;
        AscendC::DataCopy(a1,io.x[row*D],copy);
        copy.nValue=N;copy.dstNzC0Stride=N;
        AscendC::DataCopy(b1,io.w1[col*D],copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        AscendC::DataCopy(b1[N*D],io.w2[col*D],copy);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
        la = AscendC::LoadData3DParamsV2<T>{};
        la.l1H=1;la.l1W=M;la.channelSize=K;la.kExtension=K;la.mExtension=M;
        la.strideW=la.strideH=1;la.filterW=la.filterH=1;
        la.dilationFilterW=la.dilationFilterH=1;
        AscendC::LoadData(a0,a1,la);
        lb = AscendC::LoadData2dParams{};lb.repeatTimes=N*K/256;lb.srcStride=1;
        mad = AscendC::MmadParams{};mad.m=M;mad.n=N;mad.k=K;
        fix = AscendC::FixpipeParamsV220{};
        fix.nSize=N;fix.mSize=M;fix.srcStride=M;fix.dstStride=N;
        fix.ndNum=1;fix.quantPre=AscendC::QuantMode_t::F322F16;fix.unitFlag=3;
    }

    __aicore__ inline void ComputeProjection() {
        // P1,K0 uses B0slot0; slot1 can load while this MMAD executes.
        AscendC::LoadData(b0,b1,lb);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
        mad.unitFlag=2;mad.cmatrixInitVal=true;AscendC::Mmad(c0,a0,b0,mad);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        AscendC::LoadData(a0[M*K],a1[M*K],la);
        AscendC::LoadData(b0[N*K],b1[N*K],lb);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1);AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(1);
        mad.unitFlag=3;mad.cmatrixInitVal=false;AscendC::Mmad(c0,a0[M*K],b0[N*K],mad);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        // Submit P1 writeback before waiting for the independent P2 weights.
        // Paired UnitFlag3 gates Fixpipe on the final P1 MMAD blocks.
        AscendC::Fixpipe(io.result,c0,fix);AscendC::CrossCoreSetFlag<2,PIPE_FIX>(8);
    }

};

template <typename T>
__aicore__ inline void ComputeSplit128EarlyFix(GM_ADDR input,GM_ADDR weight1,
    GM_ADDR weight2,GM_ADDR workspace) {
    if ASCEND_IS_AIC{
        ComputeSplit128EarlyFixTask<T> op(input, weight1, weight2, workspace);
        op.Run();
    }
}


#define SWIGLU_DEVICE_DISPATCH
#include "tiling_key_swi_glu.h"

template <typename DT_INPUT_X, uint32_t CALC_MODE, uint32_t DIM_TAG, uint32_t TAIL_CODE, uint32_t PAD_BLOCKS, uint32_t ROW_WAVES>
__global__ __aicore__ void swi_glu(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
    GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    SwiGluDispatch<DT_INPUT_X, CALC_MODE, DIM_TAG, TAIL_CODE, PAD_BLOCKS, ROW_WAVES> op(
        input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    op.Run();
}

// HardSwish small-shape method: wrapper emits this call after including the
// user translation unit. Only these workspace-free vector tiling keys skip
// system-workspace initialization. Cube/all other keys retain the SDK call.
#if !ASCENDC_CPU_DEBUG && !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON)
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 33281UL || TILING_KEY_VAR == 6657UL || TILING_KEY_VAR == 39425UL || TILING_KEY_VAR == 14593UL || TILING_KEY_VAR == 49665UL || TILING_KEY_VAR == 51713UL || TILING_KEY_VAR == 53761UL || TILING_KEY_VAR == 55809UL || TILING_KEY_VAR == 20737UL || ((TILING_KEY_VAR & 0xffffUL) == 3585UL) || ((TILING_KEY_VAR & 0xffffUL) == 50689UL))
namespace AscendC {
    __aicore__ inline void SwiGluIgnoreUnusedSysWorkspace(GM_ADDR) {}
}
#define SetSysWorkspaceForce SwiGluIgnoreUnusedSysWorkspace
#endif
#endif
