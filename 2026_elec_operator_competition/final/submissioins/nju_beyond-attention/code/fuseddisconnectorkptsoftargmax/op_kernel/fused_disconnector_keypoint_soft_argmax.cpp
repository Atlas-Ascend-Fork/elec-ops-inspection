/*
 * Copyright (c) 2026 Beyond Attention. All rights reserved.
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

#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "adv_api/index/arithprogression.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "tiling_key_fused_disconnector_keypoint_soft_argmax.h"
using namespace AscendC;
using namespace matmul;

struct FusedDisconnectorKeypointSoftArgmaxVectorPrefix {
    uint32_t batch, height, width, rows;
    float tau;
    uint32_t stride;
};
static_assert(sizeof(FusedDisconnectorKeypointSoftArgmaxVectorPrefix) == 24, "Vector prefix size");
static_assert(offsetof(FusedDisconnectorKeypointSoftArgmaxVectorPrefix, batch) ==
                  offsetof(FusedDisconnectorKeypointSoftArgmaxTilingData, batch),
              "batch prefix offset");
static_assert(offsetof(FusedDisconnectorKeypointSoftArgmaxVectorPrefix, height) ==
                  offsetof(FusedDisconnectorKeypointSoftArgmaxTilingData, height),
              "height prefix offset");
static_assert(offsetof(FusedDisconnectorKeypointSoftArgmaxVectorPrefix, width) ==
                  offsetof(FusedDisconnectorKeypointSoftArgmaxTilingData, width),
              "width prefix offset");
static_assert(offsetof(FusedDisconnectorKeypointSoftArgmaxVectorPrefix, rows) ==
                  offsetof(FusedDisconnectorKeypointSoftArgmaxTilingData, rows),
              "rows prefix offset");
static_assert(offsetof(FusedDisconnectorKeypointSoftArgmaxVectorPrefix, tau) ==
                  offsetof(FusedDisconnectorKeypointSoftArgmaxTilingData, tau),
              "tau prefix offset");
static_assert(offsetof(FusedDisconnectorKeypointSoftArgmaxVectorPrefix, stride) ==
                  offsetof(FusedDisconnectorKeypointSoftArgmaxTilingData, stride),
              "stride prefix offset");
static_assert(IsSameType<decltype(FusedDisconnectorKeypointSoftArgmaxVectorPrefix::batch),
                         decltype(FusedDisconnectorKeypointSoftArgmaxTilingData::batch)>::value,
              "batch prefix type");
static_assert(IsSameType<decltype(FusedDisconnectorKeypointSoftArgmaxVectorPrefix::height),
                         decltype(FusedDisconnectorKeypointSoftArgmaxTilingData::height)>::value,
              "height prefix type");
static_assert(IsSameType<decltype(FusedDisconnectorKeypointSoftArgmaxVectorPrefix::width),
                         decltype(FusedDisconnectorKeypointSoftArgmaxTilingData::width)>::value,
              "width prefix type");
static_assert(IsSameType<decltype(FusedDisconnectorKeypointSoftArgmaxVectorPrefix::rows),
                         decltype(FusedDisconnectorKeypointSoftArgmaxTilingData::rows)>::value,
              "rows prefix type");
static_assert(IsSameType<decltype(FusedDisconnectorKeypointSoftArgmaxVectorPrefix::tau),
                         decltype(FusedDisconnectorKeypointSoftArgmaxTilingData::tau)>::value,
              "tau prefix type");
static_assert(IsSameType<decltype(FusedDisconnectorKeypointSoftArgmaxVectorPrefix::stride),
                         decltype(FusedDisconnectorKeypointSoftArgmaxTilingData::stride)>::value,
              "stride prefix type");

#if defined(__DAV_VEC__) || defined(SPLIT_CORE_VEC)
__aicore__ inline void RunOwnerH48(TPipe &pipe, GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR points,
                                   GM_ADDR scores, uint32_t batch, float tau) {
    constexpr uint32_t height = 48, width = 64, n = height * width;
    constexpr uint32_t stripeCount = (height + 2) * width;
    static_assert(3 * stripeCount <= 4 * n, "Conv planes must fit the decode work arena");
    static_assert(n % 64 == 0 && n / 64 <= 255);
    TBuf<TPosition::VECCALC> heatmapBuf, workBuf, weightBuf, biasBuf, scalarBuf;
    TQue<TPosition::VECOUT, 1> out;
    pipe.InitBuffer(heatmapBuf, n * 4);
    pipe.InitBuffer(workBuf, 4 * n * 4);
    pipe.InitBuffer(weightBuf, 1152 * 4);
    pipe.InitBuffer(biasBuf, 32);
    pipe.InitBuffer(scalarBuf, 32);
    pipe.InitBuffer(out, 1, 64);
    auto x = heatmapBuf.Get<float>();
    auto work = workBuf.Get<float>();
    auto w = weightBuf.Get<float>();
    auto biasValues = biasBuf.Get<float>();
    auto scalar = scalarBuf.Get<float>();
    // These four disjoint decode views become live only after the convolution drain.
    auto coord = work;
    auto row = work[n];
    auto product = work[2 * n];
    auto scratch = work[3 * n];
    GlobalTensor<float> fGm, wGm, biasGm, dst, score;
    fGm.SetGlobalBuffer((__gm__ float *)feat, batch * 32 * n);
    wGm.SetGlobalBuffer((__gm__ float *)weight, 1152);
    biasGm.SetGlobalBuffer((__gm__ float *)bias, 4);
    dst.SetGlobalBuffer((__gm__ float *)points, batch * 4 * 2);
    score.SetGlobalBuffer((__gm__ float *)scores, batch * 4);
    DataCopy(w, wGm, 1152);
    DataCopyPad(biasValues, biasGm, DataCopyExtParams{1, 16, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0.f});
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    for (uint32_t map = GetBlockIdx(); map < batch * 4; map += GetBlockNum()) {
        const uint32_t b = map / 4, k = map % 4;
        Duplicate(x, 0.f, n);
        // Full H rows populate all valid/padded columns for every channel;
        // inactive top/bottom rows remain zero until the end of this owner.
        Duplicate(work, 0.f, 3 * stripeCount);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t c = 0; c < 32; ++c) {
            if (c != 0)
                WaitFlag<HardEvent::V_MTE2>(0);
            const uint32_t src = (b * 32 + c) * n;
            DataCopyPad(work[width], fGm[src], DataCopyExtParams{uint16_t(height), (width - 1) * 4, 4, 0, 0},
                        DataCopyPadExtParams<float>{true, 1, 0, 0.f});
            DataCopyPad(work[stripeCount + width], fGm[src], DataCopyExtParams{uint16_t(height), width * 4, 0, 0, 0},
                        DataCopyPadExtParams<float>{true, 0, 0, 0.f});
            DataCopyPad(work[2 * stripeCount + width], fGm[src + 1],
                        DataCopyExtParams{uint16_t(height), (width - 1) * 4, 4, 0, 0},
                        DataCopyPadExtParams<float>{true, 0, 1, 0.f});
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            for (uint32_t ky = 0; ky < 3; ++ky) {
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    auto input = work[kx * stripeCount + ky * width];
                    const uint32_t tap = c * 9 + ky * 3 + kx;
                    Axpy<float, float, true>(x, input, w.GetValue(k * 288 + tap), uint64_t(64), uint8_t(n / 64),
                                             UnaryRepeatParams{1, 1, 8, 8});
                    PipeBarrier<PIPE_V>();
                }
            }
            SetFlag<HardEvent::V_MTE2>(0);
        }
        // Consume channel31 release before any Vector reuse of the work arena.
        WaitFlag<HardEvent::V_MTE2>(0);
        PipeBarrier<PIPE_ALL>();
        // Broadcast a single row/column pattern; avoid building and dividing n indices.
        Arange(coord, 0.0f, 1.0f, width);
        Arange(scratch, 0.0f, 1.0f, height);
        PipeBarrier<PIPE_V>();
        UnaryRepeatParams xParams(1, 1, width / 8, 0);
        Adds(coord[width], coord, 0.0f, uint64_t(64), height - 1, xParams);
        if (width == 96) {
            Adds(coord[width + 64], coord[64], 0.0f, uint64_t(32), height - 1, xParams);
        }
        BrcbRepeatParams yParams;
        yParams.dstBlkStride = width / 8;
        yParams.dstRepStride = width;
        for (uint32_t block = 0; block < width / 8; ++block) {
            Brcb(row[block * 8], scratch, height / 8, yParams);
        }
        PipeBarrier<PIPE_V>();
        ReduceMax(scalar, x, scratch, n, false);
        PipeBarrier<PIPE_ALL>();
        float maximum = scalar.GetValue(0);
        Duplicate(product, tau > 1.0e30f ? tau * 0.5f : tau, n);
        PipeBarrier<PIPE_V>();
        if (tau > 1.0e30f) {
            // Exact half scaling preserves nearby values and bounds the subtraction.
            Muls(x, x, 0.5f, n);
            PipeBarrier<PIPE_V>();
            Adds(x, x, -(maximum * 0.5f), n);
            PipeBarrier<PIPE_V>();
            Div(x, x, product, n);
        } else {
            // Subtract first so small positive tau never overflows the maximum.
            Adds(x, x, -maximum, n);
            PipeBarrier<PIPE_V>();
            Div(x, x, product, n);
        }
        PipeBarrier<PIPE_V>();
        Exp(x, x, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, x, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float denominator = scalar.GetValue(0);
        Mul(product, x, row, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, product, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float y = scalar.GetValue(0) / denominator;
        Mul(product, x, coord, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, product, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float xcoord = scalar.GetValue(0) / denominator;
        auto result = out.AllocTensor<float>();
        result.SetValue(0, (xcoord + 0.5f) * 4.0f);
        result.SetValue(1, (y + 0.5f) * 4.0f);
        result.SetValue(8, maximum + biasValues.GetValue(map % 4));
        out.EnQue(result);
        result = out.DeQue<float>();
        DataCopyPad(dst[map * 2], result, DataCopyExtParams{1, 8, 0, 0, 0});
        DataCopyPad(score[map], result[8], DataCopyExtParams{1, 4, 0, 0, 0});
        out.FreeTensor(result);
        // The original output queue owns its MTE3 lifetime. Drain before the next owner.
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void RunConvolution(TPipe &pipe, GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR output,
                                      uint32_t batch, uint32_t height, uint32_t width, uint32_t rows) {
    const uint32_t pitch = width + 8, count = rows * width;
    const uint32_t haloCount = (rows + 2) * pitch;
    TBuf<TPosition::VECCALC> haloBuf, inputBuf, outputBuf, weightBuf, biasBuf, offsetBuf;
    pipe.InitBuffer(haloBuf, haloCount * 4);
    pipe.InitBuffer(inputBuf, count * 4);
    pipe.InitBuffer(outputBuf, 4 * count * 4);
    pipe.InitBuffer(weightBuf, 1152 * 4);
    pipe.InitBuffer(biasBuf, 32);
    pipe.InitBuffer(offsetBuf, count * 4);
    auto halo = haloBuf.Get<float>();
    auto in = inputBuf.Get<float>();
    auto out = outputBuf.Get<float>();
    auto w = weightBuf.Get<float>();
    auto bs = biasBuf.Get<float>();
    auto offsets = offsetBuf.Get<uint32_t>();
    GlobalTensor<float> fGm, wGm, bGm, oGm;
    fGm.SetGlobalBuffer((__gm__ float *)feat);
    wGm.SetGlobalBuffer((__gm__ float *)weight);
    bGm.SetGlobalBuffer((__gm__ float *)bias);
    oGm.SetGlobalBuffer((__gm__ float *)output);
    auto signedOffsets = offsetBuf.Get<int32_t>();
    for (uint32_t row = 0; row < rows; ++row)
        Arange(signedOffsets[row * width], int32_t(row * pitch * 4), int32_t(4), int32_t(width));
    pipe_barrier(PIPE_V);
    DataCopy(w, wGm, 1152);
    DataCopyPad(bs, bGm, DataCopyExtParams{1, 16, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0.f});
    pipe_barrier(PIPE_ALL);
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    const uint32_t tiles = height / rows;
    for (uint32_t tile = GetBlockIdx(); tile < batch * tiles; tile += GetBlockNum()) {
        const uint32_t b = tile / tiles;
        const int y0 = (tile % tiles) * rows;
        const int first = y0 == 0 ? 0 : y0 - 1;
        const uint32_t top = y0 == 0 ? 1 : 0;
        const uint32_t bottom = y0 + rows == height ? 1 : 0;
        const uint32_t valid = rows + 2 - top - bottom;
        for (int k = 0; k < 4; ++k) {
            Duplicate(out[k * count], 0.f, count);
        }
        for (int c = 0; c < 32; ++c) {
            Duplicate(halo, 0.f, haloCount);
            pipe_barrier(PIPE_ALL);
            const uint32_t src = ((b * 32 + c) * height + first) * width;
            DataCopyPad(halo[top * pitch], fGm[src], DataCopyExtParams{uint16_t(valid), width * 4, 0, 0, 0},
                        DataCopyPadExtParams<float>{true, 1, 7, 0.f});
            pipe_barrier(PIPE_ALL);
            for (uint32_t ky = 0; ky < 3; ++ky) {
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    Gather(in, halo, offsets, (ky * pitch + kx) * 4, count);
                    pipe_barrier(PIPE_V);
                    const uint32_t tap = c * 9 + ky * 3 + kx;
                    for (int k = 0; k < 4; ++k)
                        Axpy(out[k * count], in, w.GetValue(k * 288 + tap), count);
                    pipe_barrier(PIPE_V);
                }
            }
        }
        pipe_barrier(PIPE_ALL);
        for (int k = 0; k < 4; ++k)
            DataCopy(oGm[(b * 4 + k) * height * width + y0 * width], out[k * count], count);
        pipe_barrier(PIPE_ALL);
    }
}

template <uint32_t Rows>
__aicore__ inline void CopyStripes(const LocalTensor<float> &stripes, const GlobalTensor<float> &fGm, uint32_t src,
                                   uint32_t top, uint32_t bottom) {
    constexpr uint32_t width = 96, rows = Rows;
    constexpr uint32_t stripeCount = (rows + 2) * width;
    // 95 floats plus one explicit zero pad form exactly one 384B UB row.
    if (top || bottom) {
        DataCopyPad(stripes[top * width], fGm[src], DataCopyExtParams{uint16_t(rows + 1), (width - 1) * 4, 4, 0, 0},
                    DataCopyPadExtParams<float>{true, 1, 0, 0.f});
        DataCopyPad(stripes[stripeCount + top * width], fGm[src],
                    DataCopyExtParams{uint16_t(rows + 1), width * 4, 0, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 0, 0.f});
        DataCopyPad(stripes[2 * stripeCount + top * width], fGm[src + 1],
                    DataCopyExtParams{uint16_t(rows + 1), (width - 1) * 4, 4, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 1, 0.f});
    } else {
        DataCopyPad(stripes, fGm[src], DataCopyExtParams{uint16_t(rows + 2), (width - 1) * 4, 4, 0, 0},
                    DataCopyPadExtParams<float>{true, 1, 0, 0.f});
        DataCopyPad(stripes[stripeCount], fGm[src], DataCopyExtParams{uint16_t(rows + 2), width * 4, 0, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 0, 0.f});
        DataCopyPad(stripes[2 * stripeCount], fGm[src + 1],
                    DataCopyExtParams{uint16_t(rows + 2), (width - 1) * 4, 4, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 1, 0.f});
    }
}

template <uint32_t Batch>
__aicore__ inline void RunStripeOut2B1(TPipe &pipe, GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR output) {
    static_assert(Batch == 1);
    constexpr uint32_t batch = Batch, height = 72, width = 96;
    constexpr uint32_t outCount = 2, groups = 4 / outCount;
    static_assert(outCount == 2);
    constexpr uint32_t rows = 4;
    static_assert(height % rows == 0);
    constexpr uint32_t count = rows * width;
    constexpr uint32_t stripeCount = (rows + 2) * width;
    TQue<TPosition::VECIN, 2> stripeQueue;
    TBuf<TPosition::VECCALC> outputBuf, weightBuf, biasBuf;
    pipe.InitBuffer(stripeQueue, 2, 3 * stripeCount * 4);
    pipe.InitBuffer(outputBuf, outCount * count * 4);
    pipe.InitBuffer(weightBuf, 1152 * 4);
    pipe.InitBuffer(biasBuf, 32);
    auto out = outputBuf.Get<float>();
    auto w = weightBuf.Get<float>();
    auto bs = biasBuf.Get<float>();
    GlobalTensor<float> fGm, wGm, bGm, oGm;
    fGm.SetGlobalBuffer((__gm__ float *)feat);
    wGm.SetGlobalBuffer((__gm__ float *)weight);
    bGm.SetGlobalBuffer((__gm__ float *)bias);
    oGm.SetGlobalBuffer((__gm__ float *)output);
    DataCopy(w, wGm, 1152);
    DataCopyPad(bs, bGm, DataCopyExtParams{1, 16, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0.f});
    pipe_barrier(PIPE_ALL);
    constexpr uint32_t tiles = height / rows;
    for (uint32_t tile = GetBlockIdx(); tile < batch * groups * tiles; tile += GetBlockNum()) {
        const uint32_t b = tile / (groups * tiles);
        const uint32_t mapBase = ((tile / tiles) % groups) * outCount;
        const int y0 = (tile % tiles) * rows;
        const int first = y0 == 0 ? 0 : y0 - 1;
        const uint32_t top = y0 == 0 ? 1 : 0;
        const uint32_t bottom = y0 + rows == height ? 1 : 0;
        for (uint32_t k = 0; k < outCount; ++k) {
            Duplicate(out[k * count], 0.f, count);
        }
        auto firstStripes = stripeQueue.AllocTensor<float>();
        Duplicate(firstStripes, 0.f, 3 * stripeCount);
        pipe_barrier(PIPE_ALL);
        CopyStripes<rows>(firstStripes, fGm, (b * 32 * height + first) * width, top, bottom);
        stripeQueue.EnQue(firstStripes);
        for (int c = 0; c < 32; ++c) {
            if (c + 1 < 32) {
                auto nextStripes = stripeQueue.AllocTensor<float>();
                if (c == 0) {
                    Duplicate(nextStripes, 0.f, 3 * stripeCount);
                    pipe_barrier(PIPE_ALL);
                }
                CopyStripes<rows>(nextStripes, fGm, ((b * 32 + c + 1) * height + first) * width, top, bottom);
                stripeQueue.EnQue(nextStripes);
            }
            auto stripes = stripeQueue.DeQue<float>();
            for (uint32_t ky = 0; ky < 3; ++ky) {
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    auto in = stripes[kx * stripeCount + ky * width];
                    const uint32_t tap = c * 9 + ky * 3 + kx;
                    for (uint32_t k = 0; k < outCount; ++k)
                        Axpy(out[k * count], in, w.GetValue((mapBase + k) * 288 + tap), count);
                    pipe_barrier(PIPE_V);
                }
            }
            stripeQueue.FreeTensor(stripes);
        }
        pipe_barrier(PIPE_ALL);
        for (uint32_t k = 0; k < outCount; ++k)
            DataCopy(oGm[(b * 4 + mapBase + k) * height * width + y0 * width], out[k * count], count);
        pipe_barrier(PIPE_ALL);
    }
}

template <uint32_t Batch>
__aicore__ inline void RunStripePrefetch72(TPipe &pipe, GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR output) {
    static_assert(Batch >= 1 && Batch <= 4);
    constexpr uint32_t batch = Batch, height = 72, width = 96;
    constexpr uint32_t rows = KEYPOINT_ROWS(batch, height, width);
    constexpr uint32_t count = rows * width;
    constexpr uint32_t stripeCount = (rows + 2) * width;
    TQue<TPosition::VECIN, 2> stripeQueue;
    TBuf<TPosition::VECCALC> outputBuf, weightBuf, biasBuf;
    pipe.InitBuffer(stripeQueue, 2, 3 * stripeCount * 4);
    pipe.InitBuffer(outputBuf, 4 * count * 4);
    pipe.InitBuffer(weightBuf, 1152 * 4);
    pipe.InitBuffer(biasBuf, 32);
    auto out = outputBuf.Get<float>();
    auto w = weightBuf.Get<float>();
    auto bs = biasBuf.Get<float>();
    GlobalTensor<float> fGm, wGm, bGm, oGm;
    fGm.SetGlobalBuffer((__gm__ float *)feat);
    wGm.SetGlobalBuffer((__gm__ float *)weight);
    bGm.SetGlobalBuffer((__gm__ float *)bias);
    oGm.SetGlobalBuffer((__gm__ float *)output);
    DataCopy(w, wGm, 1152);
    DataCopyPad(bs, bGm, DataCopyExtParams{1, 16, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0.f});
    pipe_barrier(PIPE_ALL);
    constexpr uint32_t tiles = height / rows;
    for (uint32_t tile = GetBlockIdx(); tile < batch * tiles; tile += GetBlockNum()) {
        const uint32_t b = tile / tiles;
        const int y0 = (tile % tiles) * rows;
        const int first = y0 == 0 ? 0 : y0 - 1;
        const uint32_t top = y0 == 0 ? 1 : 0;
        const uint32_t bottom = y0 + rows == height ? 1 : 0;
        for (int k = 0; k < 4; ++k) {
            Duplicate(out[k * count], 0.f, count);
        }
        auto firstStripes = stripeQueue.AllocTensor<float>();
        Duplicate(firstStripes, 0.f, 3 * stripeCount);
        pipe_barrier(PIPE_ALL);
        CopyStripes<rows>(firstStripes, fGm, (b * 32 * height + first) * width, top, bottom);
        stripeQueue.EnQue(firstStripes);
        for (int c = 0; c < 32; ++c) {
            if (c + 1 < 32) {
                auto nextStripes = stripeQueue.AllocTensor<float>();
                if (c == 0) {
                    Duplicate(nextStripes, 0.f, 3 * stripeCount);
                    pipe_barrier(PIPE_ALL);
                }
                CopyStripes<rows>(nextStripes, fGm, ((b * 32 + c + 1) * height + first) * width, top, bottom);
                stripeQueue.EnQue(nextStripes);
            }
            auto stripes = stripeQueue.DeQue<float>();
            for (uint32_t ky = 0; ky < 3; ++ky) {
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    auto in = stripes[kx * stripeCount + ky * width];
                    const uint32_t tap = c * 9 + ky * 3 + kx;
                    for (int k = 0; k < 4; ++k)
                        Axpy(out[k * count], in, w.GetValue(k * 288 + tap), count);
                    pipe_barrier(PIPE_V);
                }
            }
            stripeQueue.FreeTensor(stripes);
        }
        pipe_barrier(PIPE_ALL);
        for (int k = 0; k < 4; ++k)
            DataCopy(oGm[(b * 4 + k) * height * width + y0 * width], out[k * count], count);
        pipe_barrier(PIPE_ALL);
    }
}

__aicore__ inline void RunPair2B3(TPipe &pipe, GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR output) {
    static_assert(QUE_MAX_EVENT >= 4, "Four manual slot events are required");
    constexpr uint32_t batch = 3, height = 64, width = 64;
    const uint32_t tiles = 13u;
    const uint32_t maxRows = (height + tiles - 1) / tiles;
    const uint32_t maxCount = maxRows * width;
    const uint32_t maxStripeCount = (maxRows + 2) * width;
    TBuf<TPosition::VECIN> stripeBuf, stripeAltBuf, stripe2Buf, stripe3Buf;
    TBuf<TPosition::VECCALC> outputBuf, weightBuf, biasBuf;
    pipe.InitBuffer(stripeBuf, 3 * maxStripeCount * 4);
    pipe.InitBuffer(stripeAltBuf, 3 * maxStripeCount * 4);
    pipe.InitBuffer(stripe2Buf, 3 * maxStripeCount * 4);
    pipe.InitBuffer(stripe3Buf, 3 * maxStripeCount * 4);
    pipe.InitBuffer(outputBuf, 8 * maxCount * 4);
    pipe.InitBuffer(weightBuf, 1152 * 4);
    pipe.InitBuffer(biasBuf, 32);
    auto stripes0 = stripeBuf.Get<float>();
    auto stripes1 = stripeAltBuf.Get<float>();
    auto stripes2 = stripe2Buf.Get<float>();
    auto stripes3 = stripe3Buf.Get<float>();
    auto out = outputBuf.Get<float>();
    auto odd = out[4 * maxCount];
    auto w = weightBuf.Get<float>();
    auto bs = biasBuf.Get<float>();
    GlobalTensor<float> fGm, wGm, bGm, oGm;
    fGm.SetGlobalBuffer((__gm__ float *)feat);
    wGm.SetGlobalBuffer((__gm__ float *)weight);
    bGm.SetGlobalBuffer((__gm__ float *)bias);
    oGm.SetGlobalBuffer((__gm__ float *)output);
    DataCopy(w, wGm, 1152);
    DataCopyPad(bs, bGm, DataCopyExtParams{1, 16, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0.f});
    pipe_barrier(PIPE_ALL);
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    for (uint32_t tile = GetBlockIdx(); tile < batch * tiles; tile += GetBlockNum()) {
        const uint32_t b = tile / tiles;
        const uint32_t tileInBatch = tile % tiles;
        const int y0 = tileInBatch * height / tiles;
        const uint32_t rows = (tileInBatch + 1) * height / tiles - y0;
        const uint32_t count = rows * width, stripeCount = (rows + 2) * width;
        const int first = y0 == 0 ? 0 : y0 - 1;
        const uint32_t top = y0 == 0 ? 1 : 0;
        const uint32_t bottom = y0 + rows == height ? 1 : 0;
        const uint32_t valid = rows + 2 - top - bottom;
        Duplicate(out, 0.f, 4 * count);
        Duplicate(odd, 0.f, 4 * count);
        Duplicate(stripes0, 0.f, 3 * stripeCount);
        Duplicate(stripes1, 0.f, 3 * stripeCount);
        Duplicate(stripes2, 0.f, 3 * stripeCount);
        Duplicate(stripes3, 0.f, 3 * stripeCount);
        pipe_barrier(PIPE_ALL);
        for (uint32_t initial = 0; initial < 2; ++initial) {
            auto initialStripes = initial == 0 ? stripes0 : stripes1;
            const uint32_t src = ((b * 32 + initial) * height + first) * width;
            DataCopyPad(initialStripes[top * width], fGm[src],
                        DataCopyExtParams{uint16_t(valid), (width - 1) * 4, 4, 0, 0},
                        DataCopyPadExtParams<float>{true, 1, 0, 0.f});
            DataCopyPad(initialStripes[stripeCount + top * width], fGm[src],
                        DataCopyExtParams{uint16_t(valid), width * 4, 0, 0, 0},
                        DataCopyPadExtParams<float>{true, 0, 0, 0.f});
            DataCopyPad(initialStripes[2 * stripeCount + top * width], fGm[src + 1],
                        DataCopyExtParams{uint16_t(valid), (width - 1) * 4, 4, 0, 0},
                        DataCopyPadExtParams<float>{true, 0, 1, 0.f});
            SetFlag<HardEvent::MTE2_V>(initial);
        }
        for (uint32_t pair = 0; pair < 16; ++pair) {
            const uint32_t c0 = pair * 2, c1 = c0 + 1;
            const uint32_t slot = (pair & 1u) * 2;
            WaitFlag<HardEvent::MTE2_V>(slot);
            WaitFlag<HardEvent::MTE2_V>(slot + 1);
            if (pair + 1 < 16) {
                const uint32_t nextSlot = slot ^ 2u;
                for (uint32_t lane = 0; lane < 2; ++lane) {
                    // The first alternate pair has never been read; subsequent reuse consumes its release.
                    if (pair != 0)
                        WaitFlag<HardEvent::V_MTE2>(nextSlot + lane);
                    auto nextEven = nextSlot == 0 ? stripes0 : stripes2;
                    auto nextOdd = nextSlot == 0 ? stripes1 : stripes3;
                    auto nextStripes = lane == 0 ? nextEven : nextOdd;
                    const uint32_t src = ((b * 32 + c0 + 2 + lane) * height + first) * width;
                    DataCopyPad(nextStripes[top * width], fGm[src],
                                DataCopyExtParams{uint16_t(valid), (width - 1) * 4, 4, 0, 0},
                                DataCopyPadExtParams<float>{true, 1, 0, 0.f});
                    DataCopyPad(nextStripes[stripeCount + top * width], fGm[src],
                                DataCopyExtParams{uint16_t(valid), width * 4, 0, 0, 0},
                                DataCopyPadExtParams<float>{true, 0, 0, 0.f});
                    DataCopyPad(nextStripes[2 * stripeCount + top * width], fGm[src + 1],
                                DataCopyExtParams{uint16_t(valid), (width - 1) * 4, 4, 0, 0},
                                DataCopyPadExtParams<float>{true, 0, 1, 0.f});
                    SetFlag<HardEvent::MTE2_V>(nextSlot + lane);
                }
            }
            auto evenStripes = slot == 0 ? stripes0 : stripes2;
            auto oddStripes = slot == 0 ? stripes1 : stripes3;
            for (uint32_t ky = 0; ky < 3; ++ky) {
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    auto evenInput = evenStripes[kx * stripeCount + ky * width];
                    auto oddInput = oddStripes[kx * stripeCount + ky * width];
                    const uint32_t tap0 = c0 * 9 + ky * 3 + kx;
                    const uint32_t tap1 = c1 * 9 + ky * 3 + kx;
                    for (int k = 0; k < 4; ++k)
                        Axpy<float, float, true>(out[k * count], evenInput, w.GetValue(k * 288 + tap0), uint64_t(64),
                                                 uint8_t(count / 64), UnaryRepeatParams{1, 1, 8, 8});
                    for (int k = 0; k < 4; ++k)
                        Axpy<float, float, true>(odd[k * count], oddInput, w.GetValue(k * 288 + tap1), uint64_t(64),
                                                 uint8_t(count / 64), UnaryRepeatParams{1, 1, 8, 8});
                    // Eight destination regions are disjoint; finish them before the next dependent tap.
                    pipe_barrier(PIPE_V);
                }
            }
            SetFlag<HardEvent::V_MTE2>(slot);
            SetFlag<HardEvent::V_MTE2>(slot + 1);
        }
        WaitFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(1);
        WaitFlag<HardEvent::V_MTE2>(2);
        WaitFlag<HardEvent::V_MTE2>(3);
        Add(out, out, odd, 4 * count);
        pipe_barrier(PIPE_ALL);
        for (int k = 0; k < 4; ++k)
            DataCopy(oGm[(b * 4 + k) * height * width + y0 * width], out[k * count], count);
        pipe_barrier(PIPE_ALL);
    }
}

template <uint32_t Batch>
__aicore__ inline void RunStripeDouble64(TPipe &pipe, GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR output) {
    static_assert(Batch >= 1 && Batch <= 4);
    constexpr uint32_t batch = Batch, height = 64, width = 64;
    constexpr uint32_t tiles = Batch == 3 ? 13u : Batch == 4 ? 10u : height / KEYPOINT_ROWS(batch, height, width);
    const uint32_t maxRows = (height + tiles - 1) / tiles;
    const uint32_t maxCount = maxRows * width;
    const uint32_t maxStripeCount = (maxRows + 2) * width;
    TBuf<TPosition::VECIN> stripeBuf, stripeAltBuf;
    TBuf<TPosition::VECCALC> outputBuf, weightBuf, biasBuf;
    pipe.InitBuffer(stripeBuf, 3 * maxStripeCount * 4);
    pipe.InitBuffer(stripeAltBuf, 3 * maxStripeCount * 4);
    pipe.InitBuffer(outputBuf, 4 * maxCount * 4);
    pipe.InitBuffer(weightBuf, 1152 * 4);
    pipe.InitBuffer(biasBuf, 32);
    auto stripes0 = stripeBuf.Get<float>();
    auto stripes1 = stripeAltBuf.Get<float>();
    auto out = outputBuf.Get<float>();
    auto w = weightBuf.Get<float>();
    auto bs = biasBuf.Get<float>();
    GlobalTensor<float> fGm, wGm, bGm, oGm;
    fGm.SetGlobalBuffer((__gm__ float *)feat);
    wGm.SetGlobalBuffer((__gm__ float *)weight);
    bGm.SetGlobalBuffer((__gm__ float *)bias);
    oGm.SetGlobalBuffer((__gm__ float *)output);
    DataCopy(w, wGm, 1152);
    DataCopyPad(bs, bGm, DataCopyExtParams{1, 16, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0.f});
    pipe_barrier(PIPE_ALL);
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    for (uint32_t tile = GetBlockIdx(); tile < batch * tiles; tile += GetBlockNum()) {
        const uint32_t b = tile / tiles;
        const uint32_t tileInBatch = tile % tiles;
        const int y0 = tileInBatch * height / tiles;
        const uint32_t rows = (tileInBatch + 1) * height / tiles - y0;
        const uint32_t count = rows * width, stripeCount = (rows + 2) * width;
        const int first = y0 == 0 ? 0 : y0 - 1;
        const uint32_t top = y0 == 0 ? 1 : 0;
        const uint32_t bottom = y0 + rows == height ? 1 : 0;
        const uint32_t valid = rows + 2 - top - bottom;
        for (int k = 0; k < 4; ++k) {
            Duplicate(out[k * count], 0.f, count);
        }
        Duplicate(stripes0, 0.f, 3 * stripeCount);
        Duplicate(stripes1, 0.f, 3 * stripeCount);
        pipe_barrier(PIPE_ALL);
        const uint32_t src = (b * 32 * height + first) * width;
        DataCopyPad(stripes0[top * width], fGm[src], DataCopyExtParams{uint16_t(valid), (width - 1) * 4, 4, 0, 0},
                    DataCopyPadExtParams<float>{true, 1, 0, 0.f});
        DataCopyPad(stripes0[stripeCount + top * width], fGm[src],
                    DataCopyExtParams{uint16_t(valid), width * 4, 0, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 0, 0.f});
        DataCopyPad(stripes0[2 * stripeCount + top * width], fGm[src + 1],
                    DataCopyExtParams{uint16_t(valid), (width - 1) * 4, 4, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 1, 0.f});
        SetFlag<HardEvent::MTE2_V>(0);
        for (int c = 0; c < 32; ++c) {
            const uint32_t slot = c & 1;
            WaitFlag<HardEvent::MTE2_V>(slot);
            if (c + 1 < 32) {
                const uint32_t nextSlot = slot ^ 1u;
                if (c != 0)
                    WaitFlag<HardEvent::V_MTE2>(nextSlot);
                auto nextStripes = nextSlot == 0 ? stripes0 : stripes1;
                const uint32_t src = ((b * 32 + c + 1) * height + first) * width;
                DataCopyPad(nextStripes[top * width], fGm[src],
                            DataCopyExtParams{uint16_t(valid), (width - 1) * 4, 4, 0, 0},
                            DataCopyPadExtParams<float>{true, 1, 0, 0.f});
                DataCopyPad(nextStripes[stripeCount + top * width], fGm[src],
                            DataCopyExtParams{uint16_t(valid), width * 4, 0, 0, 0},
                            DataCopyPadExtParams<float>{true, 0, 0, 0.f});
                DataCopyPad(nextStripes[2 * stripeCount + top * width], fGm[src + 1],
                            DataCopyExtParams{uint16_t(valid), (width - 1) * 4, 4, 0, 0},
                            DataCopyPadExtParams<float>{true, 0, 1, 0.f});
                SetFlag<HardEvent::MTE2_V>(nextSlot);
            }
            auto stripes = slot == 0 ? stripes0 : stripes1;
            for (uint32_t ky = 0; ky < 3; ++ky) {
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    auto in = stripes[kx * stripeCount + ky * width];
                    const uint32_t tap = c * 9 + ky * 3 + kx;
                    for (int k = 0; k < 4; ++k)
                        Axpy<float, float, true>(out[k * count], in, w.GetValue(k * 288 + tap), uint64_t(64),
                                                 uint8_t(count / 64), UnaryRepeatParams{1, 1, 8, 8});
                    pipe_barrier(PIPE_V);
                }
            }
            SetFlag<HardEvent::V_MTE2>(slot);
        }
        WaitFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(1);
        pipe_barrier(PIPE_ALL);
        for (int k = 0; k < 4; ++k)
            DataCopy(oGm[(b * 4 + k) * height * width + y0 * width], out[k * count], count);
        pipe_barrier(PIPE_ALL);
    }
}

template <uint32_t Rows>
__aicore__ inline void CopyStripes48(const LocalTensor<float> &stripes, const GlobalTensor<float> &fGm, uint32_t src,
                                     uint32_t top, uint32_t bottom) {
    constexpr uint32_t width = 64, rows = Rows;
    constexpr uint32_t stripeCount = (rows + 2) * width;
    // 63 floats plus one explicit zero pad form exactly one 256B UB row.
    if (top || bottom) {
        DataCopyPad(stripes[top * width], fGm[src], DataCopyExtParams{uint16_t(rows + 1), (width - 1) * 4, 4, 0, 0},
                    DataCopyPadExtParams<float>{true, 1, 0, 0.f});
        DataCopyPad(stripes[stripeCount + top * width], fGm[src],
                    DataCopyExtParams{uint16_t(rows + 1), width * 4, 0, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 0, 0.f});
        DataCopyPad(stripes[2 * stripeCount + top * width], fGm[src + 1],
                    DataCopyExtParams{uint16_t(rows + 1), (width - 1) * 4, 4, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 1, 0.f});
    } else {
        DataCopyPad(stripes, fGm[src], DataCopyExtParams{uint16_t(rows + 2), (width - 1) * 4, 4, 0, 0},
                    DataCopyPadExtParams<float>{true, 1, 0, 0.f});
        DataCopyPad(stripes[stripeCount], fGm[src], DataCopyExtParams{uint16_t(rows + 2), width * 4, 0, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 0, 0.f});
        DataCopyPad(stripes[2 * stripeCount], fGm[src + 1],
                    DataCopyExtParams{uint16_t(rows + 2), (width - 1) * 4, 4, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 1, 0.f});
    }
}

template <uint32_t Batch>
__aicore__ inline void RunStripePrefetch48(TPipe &pipe, GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR output) {
    static_assert(Batch >= 1 && Batch <= 4);
    constexpr uint32_t batch = Batch, height = 48, width = 64;
    constexpr uint32_t rows = KEYPOINT_ROWS(batch, height, width);
    constexpr uint32_t count = rows * width;
    constexpr uint32_t stripeCount = (rows + 2) * width;
    TQue<TPosition::VECIN, 2> stripeQueue;
    TBuf<TPosition::VECCALC> outputBuf, weightBuf, biasBuf;
    pipe.InitBuffer(stripeQueue, 2, 3 * stripeCount * 4);
    pipe.InitBuffer(outputBuf, 4 * count * 4);
    pipe.InitBuffer(weightBuf, 1152 * 4);
    pipe.InitBuffer(biasBuf, 32);
    auto out = outputBuf.Get<float>();
    auto w = weightBuf.Get<float>();
    auto bs = biasBuf.Get<float>();
    GlobalTensor<float> fGm, wGm, bGm, oGm;
    fGm.SetGlobalBuffer((__gm__ float *)feat);
    wGm.SetGlobalBuffer((__gm__ float *)weight);
    bGm.SetGlobalBuffer((__gm__ float *)bias);
    oGm.SetGlobalBuffer((__gm__ float *)output);
    DataCopy(w, wGm, 1152);
    DataCopyPad(bs, bGm, DataCopyExtParams{1, 16, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0.f});
    pipe_barrier(PIPE_ALL);
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    constexpr uint32_t tiles = height / rows;
    for (uint32_t tile = GetBlockIdx(); tile < batch * tiles; tile += GetBlockNum()) {
        const uint32_t b = tile / tiles;
        const int y0 = (tile % tiles) * rows;
        const int first = y0 == 0 ? 0 : y0 - 1;
        const uint32_t top = y0 == 0 ? 1 : 0;
        const uint32_t bottom = y0 + rows == height ? 1 : 0;
        for (int k = 0; k < 4; ++k) {
            Duplicate(out[k * count], 0.f, count);
        }
        auto firstStripes = stripeQueue.AllocTensor<float>();
        Duplicate(firstStripes, 0.f, 3 * stripeCount);
        pipe_barrier(PIPE_ALL);
        CopyStripes48<rows>(firstStripes, fGm, (b * 32 * height + first) * width, top, bottom);
        stripeQueue.EnQue(firstStripes);
        for (int c = 0; c < 32; ++c) {
            if (c + 1 < 32) {
                auto nextStripes = stripeQueue.AllocTensor<float>();
                if (c == 0) {
                    Duplicate(nextStripes, 0.f, 3 * stripeCount);
                    pipe_barrier(PIPE_ALL);
                }
                CopyStripes48<rows>(nextStripes, fGm, ((b * 32 + c + 1) * height + first) * width, top, bottom);
                stripeQueue.EnQue(nextStripes);
            }
            auto stripes = stripeQueue.DeQue<float>();
            for (uint32_t ky = 0; ky < 3; ++ky) {
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    auto in = stripes[kx * stripeCount + ky * width];
                    const uint32_t tap = c * 9 + ky * 3 + kx;
                    for (int k = 0; k < 4; ++k)
                        Axpy(out[k * count], in, w.GetValue(k * 288 + tap), count);
                    pipe_barrier(PIPE_V);
                }
            }
            stripeQueue.FreeTensor(stripes);
        }
        pipe_barrier(PIPE_ALL);
        for (int k = 0; k < 4; ++k)
            DataCopy(oGm[(b * 4 + k) * height * width + y0 * width], out[k * count], count);
        pipe_barrier(PIPE_ALL);
    }
}

__aicore__ inline void RunDecode(TPipe &pipe, GM_ADDR input, GM_ADDR bias, GM_ADDR points, GM_ADDR scores,
                                 uint32_t maps, uint32_t height, uint32_t width, float tau) {
    TQue<TPosition::VECIN, 1> in;
    TQue<TPosition::VECOUT, 1> out;
    TBuf<TPosition::VECCALC> coordBuf, rowBuf, productBuf, scratchBuf, scalarBuf, biasBuf;
    uint32_t n = height * width;
    pipe.InitBuffer(in, 1, n * 4);
    pipe.InitBuffer(out, 1, 64);
    pipe.InitBuffer(coordBuf, n * 4);
    pipe.InitBuffer(rowBuf, n * 4);
    pipe.InitBuffer(productBuf, n * 4);
    pipe.InitBuffer(scratchBuf, n * 4);
    pipe.InitBuffer(scalarBuf, 32);
    pipe.InitBuffer(biasBuf, 32);
    auto coord = coordBuf.Get<float>();
    auto row = rowBuf.Get<float>();
    auto product = productBuf.Get<float>();
    auto scratch = scratchBuf.Get<float>();
    auto scalar = scalarBuf.Get<float>();
    auto biasValues = biasBuf.Get<float>();
    // Broadcast a single row/column pattern; avoid building and dividing n indices.
    Arange(coord, 0.0f, 1.0f, width);
    Arange(scratch, 0.0f, 1.0f, height);
    PipeBarrier<PIPE_V>();
    UnaryRepeatParams xParams(1, 1, width / 8, 0);
    Adds(coord[width], coord, 0.0f, uint64_t(64), height - 1, xParams);
    if (width == 96) {
        Adds(coord[width + 64], coord[64], 0.0f, uint64_t(32), height - 1, xParams);
    }
    BrcbRepeatParams yParams;
    yParams.dstBlkStride = width / 8;
    yParams.dstRepStride = width;
    for (uint32_t block = 0; block < width / 8; ++block) {
        Brcb(row[block * 8], scratch, height / 8, yParams);
    }
    PipeBarrier<PIPE_V>();
    GlobalTensor<float> src, biasGm, dst, score;
    src.SetGlobalBuffer((__gm__ float *)input, maps * n);
    biasGm.SetGlobalBuffer((__gm__ float *)bias, 4);
    dst.SetGlobalBuffer((__gm__ float *)points, maps * 2);
    score.SetGlobalBuffer((__gm__ float *)scores, maps);
    DataCopyPad(biasValues, biasGm, DataCopyExtParams{1, 16, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0.f});
    PipeBarrier<PIPE_ALL>();
    for (uint32_t map = GetBlockIdx(); map < maps; map += GetBlockNum()) {
        auto x = in.AllocTensor<float>();
        DataCopy(x, src[map * n], n);
        in.EnQue(x);
        x = in.DeQue<float>();
        ReduceMax(scalar, x, scratch, n, false);
        PipeBarrier<PIPE_ALL>();
        float maximum = scalar.GetValue(0);
        Duplicate(product, tau > 1.0e30f ? tau * 0.5f : tau, n);
        PipeBarrier<PIPE_V>();
        if (tau > 1.0e30f) {
            // Exact half scaling preserves nearby values and bounds the subtraction.
            Muls(x, x, 0.5f, n);
            PipeBarrier<PIPE_V>();
            Adds(x, x, -(maximum * 0.5f), n);
            PipeBarrier<PIPE_V>();
            Div(x, x, product, n);
        } else {
            // Subtract first so small positive tau never overflows the maximum.
            Adds(x, x, -maximum, n);
            PipeBarrier<PIPE_V>();
            Div(x, x, product, n);
        }
        PipeBarrier<PIPE_V>();
        Exp(x, x, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, x, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float denominator = scalar.GetValue(0);
        Mul(product, x, row, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, product, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float y = scalar.GetValue(0) / denominator;
        Mul(product, x, coord, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, product, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float xcoord = scalar.GetValue(0) / denominator;
        auto result = out.AllocTensor<float>();
        result.SetValue(0, (xcoord + 0.5f) * 4.0f);
        result.SetValue(1, (y + 0.5f) * 4.0f);
        result.SetValue(8, maximum + biasValues.GetValue(map % 4));
        out.EnQue(result);
        result = out.DeQue<float>();
        DataCopyPad(dst[map * 2], result, DataCopyExtParams{1, 8, 0, 0, 0});
        DataCopyPad(score[map], result[8], DataCopyExtParams{1, 4, 0, 0, 0});
        out.FreeTensor(result);
        in.FreeTensor(x);
    }
}

__aicore__ inline void RunProjectionPack(TPipe &pipe, GM_ADDR weight, GM_ADDR packed) {
    TBuf<TPosition::VECCALC> weightBuf, packedBuf, indexBuf;
    pipe.InitBuffer(weightBuf, 1152 * 4);
    pipe.InitBuffer(packedBuf, kProjectionWeightFloats * 4);
    pipe.InitBuffer(indexBuf, 128 * 4);
    auto w = weightBuf.Get<float>();
    auto a = packedBuf.Get<float>();
    auto indices = indexBuf.Get<uint32_t>();
    auto signedIndex = indexBuf.Get<int32_t>();
    GlobalTensor<float> wGm, aGm;
    wGm.SetGlobalBuffer((__gm__ float *)weight);
    aGm.SetGlobalBuffer((__gm__ float *)packed);
    DataCopy(w, wGm, 1152);
    Duplicate(a, 0.f, kProjectionWeightFloats);
    for (uint32_t o = 0; o < 4; ++o)
        Arange(signedIndex[o * 32], int32_t(o * 288 * 4), int32_t(9 * 4), int32_t(32));
    pipe_barrier(PIPE_ALL);
    for (uint32_t t = 0; t < 9; ++t)
        Gather(a[t * 128], w, indices, uint32_t(t * 4), uint32_t(128));
    pipe_barrier(PIPE_ALL);
    DataCopy(aGm, a, kProjectionWeightFloats);
    pipe_barrier(PIPE_ALL);
}

__aicore__ inline void RunProjectionShiftDecode(TPipe &pipe, GM_ADDR packed, GM_ADDR bias, GM_ADDR points,
                                                GM_ADDR scores, uint32_t maps, uint32_t height, uint32_t width,
                                                float tau, uint32_t worker, uint32_t workerCount) {
    TQue<TPosition::VECIN, 1> in;
    TQue<TPosition::VECOUT, 1> out;
    TBuf<TPosition::VECCALC> coordBuf, rowBuf, productBuf, scratchBuf, scalarBuf, biasBuf;
    uint32_t n = height * width;
    pipe.InitBuffer(in, 1, n * 4);
    pipe.InitBuffer(out, 1, 64);
    pipe.InitBuffer(coordBuf, n * 4);
    pipe.InitBuffer(rowBuf, n * 4);
    pipe.InitBuffer(productBuf, n * 4);
    pipe.InitBuffer(scratchBuf, n * 4);
    pipe.InitBuffer(scalarBuf, 32);
    pipe.InitBuffer(biasBuf, 32);
    auto coord = coordBuf.Get<float>();
    auto row = rowBuf.Get<float>();
    auto product = productBuf.Get<float>();
    auto scratch = scratchBuf.Get<float>();
    auto scalar = scalarBuf.Get<float>();
    auto biasLocal = biasBuf.Get<float>();
    // Broadcast a single row/column pattern; avoid building and dividing n indices.
    Arange(coord, 0.0f, 1.0f, width);
    Arange(scratch, 0.0f, 1.0f, height);
    PipeBarrier<PIPE_V>();
    UnaryRepeatParams xParams(1, 1, width / 8, 0);
    Adds(coord[width], coord, 0.0f, uint64_t(64), height - 1, xParams);
    if (width == 96) {
        Adds(coord[width + 64], coord[64], 0.0f, uint64_t(32), height - 1, xParams);
    }
    BrcbRepeatParams yParams;
    yParams.dstBlkStride = width / 8;
    yParams.dstRepStride = width;
    for (uint32_t block = 0; block < width / 8; ++block) {
        Brcb(row[block * 8], scratch, height / 8, yParams);
    }
    PipeBarrier<PIPE_V>();
    GlobalTensor<float> src, dst, score, biasGm;
    src.SetGlobalBuffer((__gm__ float *)packed + kProjectionWeightFloats, (maps / 4) * kProjectionRows * n);
    dst.SetGlobalBuffer((__gm__ float *)points, maps * 2);
    score.SetGlobalBuffer((__gm__ float *)scores, maps);
    biasGm.SetGlobalBuffer((__gm__ float *)bias, 4);
    DataCopyPad(biasLocal, biasGm, DataCopyExtParams{1, 16, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0.f});
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    for (uint32_t map = worker; map < maps; map += workerCount) {
        auto x = in.AllocTensor<float>();
        Duplicate(x, 0.f, n);
        const uint32_t b = map / 4, o = map % 4;
        const int y0 = 0;
        const uint32_t rows = height;
        for (int ky = 0; ky < 3; ++ky) {
            const int dy = ky - 1;
            const int r0 = y0 + dy < 0 ? -(y0 + dy) : 0;
            const int r1 = y0 + int(rows) + dy > int(height) ? int(height) - y0 - dy : int(rows);
            for (int kx = 0; kx < 3; ++kx) {
                const int dx = kx - 1;
                Duplicate(product, 0.f, n);
                PipeBarrier<PIPE_ALL>();
                const uint32_t cut = dx == 0 ? 0 : 1;
                const uint32_t t = ky * 3 + kx;
                const uint32_t source =
                    (b * kProjectionRows + t * 4 + o) * n + (y0 + r0 + dy) * width + (dx > 0 ? 1 : 0);
                DataCopyPad(product[r0 * width], src[source],
                            DataCopyExtParams{uint16_t(r1 - r0), (width - cut) * 4, cut * 4, 0, 0},
                            DataCopyPadExtParams<float>{true, uint8_t(dx < 0), uint8_t(dx > 0), 0.f});
                PipeBarrier<PIPE_ALL>();
                Add(x, x, product, n);
                PipeBarrier<PIPE_V>();
            }
        }
        PipeBarrier<PIPE_ALL>();
        in.EnQue(x);
        x = in.DeQue<float>();
        ReduceMax(scalar, x, scratch, n, false);
        PipeBarrier<PIPE_ALL>();
        float maximum = scalar.GetValue(0);
        if (tau == 0.5f || tau == 1.0f || tau == 2.0f) {
            Adds(x, x, -maximum, n);
            PipeBarrier<PIPE_V>();
            Muls(x, x, tau == 0.5f ? 2.0f : tau == 1.0f ? 1.0f : 0.5f, n);
        } else {
            Duplicate(product, tau > 1.0e30f ? tau * 0.5f : tau, n);
            PipeBarrier<PIPE_V>();
            if (tau > 1.0e30f) {
                // Exact half scaling preserves nearby values and bounds the subtraction.
                Muls(x, x, 0.5f, n);
                PipeBarrier<PIPE_V>();
                Adds(x, x, -(maximum * 0.5f), n);
                PipeBarrier<PIPE_V>();
                Div(x, x, product, n);
            } else {
                // Subtract first so small positive tau never overflows the maximum.
                Adds(x, x, -maximum, n);
                PipeBarrier<PIPE_V>();
                Div(x, x, product, n);
            }
        }
        PipeBarrier<PIPE_V>();
        Exp(x, x, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, x, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float denominator = scalar.GetValue(0);
        Mul(product, x, row, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, product, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float y = scalar.GetValue(0) / denominator;
        Mul(product, x, coord, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, product, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float xcoord = scalar.GetValue(0) / denominator;
        auto result = out.AllocTensor<float>();
        result.SetValue(0, (xcoord + 0.5f) * 4.0f);
        result.SetValue(1, (y + 0.5f) * 4.0f);
        result.SetValue(8, maximum + biasLocal.GetValue(map % 4));
        out.EnQue(result);
        result = out.DeQue<float>();
        DataCopyPad(dst[map * 2], result, DataCopyExtParams{1, 8, 0, 0, 0});
        DataCopyPad(score[map], result[8], DataCopyExtParams{1, 4, 0, 0, 0});
        out.FreeTensor(result);
        in.FreeTensor(x);
    }
}

__aicore__ inline void RunA2Pack(TPipe &pipe, GM_ADDR feat, GM_ADDR weight, GM_ADDR packed, GM_ADDR nzWeights,
                                 uint32_t batch, uint32_t worker, uint32_t workers) {
    TBuf<TPosition::VECCALC> rowInBuf, rowOutBuf, rowIndexBuf, weightInBuf, weightOutBuf, weightIndexBuf;
    pipe.InitBuffer(rowInBuf, 2048);
    pipe.InitBuffer(rowOutBuf, 2048);
    pipe.InitBuffer(rowIndexBuf, 2048);
    pipe.InitBuffer(weightInBuf, 4608);
    pipe.InitBuffer(weightOutBuf, 18432);
    pipe.InitBuffer(weightIndexBuf, 128);
    auto rowIn = rowInBuf.Get<float>();
    auto rowOut = rowOutBuf.Get<float>();
    auto rowIndex = rowIndexBuf.Get<int32_t>();
    // Expand the eight channel offsets once; each stage appends disjoint column groups.
    Arange(rowIndex, int32_t(0), int32_t(256), int32_t(8));
    for (uint32_t span = 8; span < 512; span *= 2) {
        PipeBarrier<PIPE_V>();
        Adds(rowIndex[span], rowIndex, int32_t((span / 8) * 4), int32_t(span));
    }
    PipeBarrier<PIPE_V>();
    GlobalTensor<float> src, dst, weights, nz;
    src.SetGlobalBuffer((__gm__ float *)feat);
    dst.SetGlobalBuffer((__gm__ float *)packed);
    weights.SetGlobalBuffer((__gm__ float *)weight);
    nz.SetGlobalBuffer((__gm__ float *)nzWeights);
    for (uint32_t job = worker; job < batch * 4 * 50; job += workers) {
        const uint32_t y = job % 50, c1 = (job / 50) % 4, b = job / (4 * 50);
        if (y == 0 || y == 49) {
            Duplicate(rowOut, 0.f, 512);
            PipeBarrier<PIPE_ALL>();
        } else {
            // Eight channel rows: source gap is (H*W-W)*4 bytes, destination is contiguous.
            DataCopyPad(rowIn, src[((b * 32 + c1 * 8) * 48 + (y - 1)) * 64], DataCopyExtParams{8, 256, 12032, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0.f});
            PipeBarrier<PIPE_ALL>();
            Gather(rowOut, rowIn, rowIndexBuf.Get<uint32_t>(), 0, 512);
            PipeBarrier<PIPE_ALL>();
        }
        DataCopy(dst[job * 512], rowOut, 512);
        PipeBarrier<PIPE_ALL>();
    }
    if (worker == 0) {
        auto original = weightInBuf.Get<float>();
        auto target = weightOutBuf.Get<float>();
        auto indices = weightIndexBuf.Get<int32_t>();
        DataCopy(original, weights, 1152);
        Duplicate(target, 0.f, 4608);
        PipeBarrier<PIPE_ALL>();
        // Reuse four head groups; update only after the previous Gather has read them.
        Arange(indices, int32_t(0), int32_t(36), int32_t(8));
        PipeBarrier<PIPE_V>();
        Adds(indices[8], indices, int32_t(1152), int32_t(8));
        PipeBarrier<PIPE_V>();
        Adds(indices[16], indices, int32_t(2304), int32_t(16));
        for (uint32_t k1 = 0; k1 < 36; ++k1) {
            PipeBarrier<PIPE_V>();
            Gather(target[k1 * 128], original, weightIndexBuf.Get<uint32_t>(), 0, 32);
            PipeBarrier<PIPE_V>();
            if (k1 + 1 < 36)
                Adds(indices, indices, int32_t(k1 % 9 == 8 ? 256 : 4), int32_t(32));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(nz, target, 4608);
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunA2Pack64B1(TPipe &pipe, GM_ADDR feat, GM_ADDR weight, GM_ADDR packed, GM_ADDR nzWeights,
                                     uint32_t batch, uint32_t worker, uint32_t workers) {
    TBuf<TPosition::VECCALC> rowInBuf, rowOutBuf, rowIndexBuf, weightInBuf, weightOutBuf, weightIndexBuf;
    pipe.InitBuffer(rowInBuf, 2048);
    pipe.InitBuffer(rowOutBuf, 2048);
    pipe.InitBuffer(rowIndexBuf, 2048);
    pipe.InitBuffer(weightInBuf, 4608);
    pipe.InitBuffer(weightOutBuf, 18432);
    pipe.InitBuffer(weightIndexBuf, 128);
    auto rowIn = rowInBuf.Get<float>();
    auto rowOut = rowOutBuf.Get<float>();
    auto rowIndex = rowIndexBuf.Get<int32_t>();
    // Expand the eight channel offsets once; each stage appends disjoint column groups.
    Arange(rowIndex, int32_t(0), int32_t(256), int32_t(8));
    for (uint32_t span = 8; span < 512; span *= 2) {
        PipeBarrier<PIPE_V>();
        Adds(rowIndex[span], rowIndex, int32_t((span / 8) * 4), int32_t(span));
    }
    PipeBarrier<PIPE_V>();
    GlobalTensor<float> src, dst, weights, nz;
    src.SetGlobalBuffer((__gm__ float *)feat);
    dst.SetGlobalBuffer((__gm__ float *)packed);
    weights.SetGlobalBuffer((__gm__ float *)weight);
    nz.SetGlobalBuffer((__gm__ float *)nzWeights);
    for (uint32_t job = worker; job < batch * 4 * 66; job += workers) {
        const uint32_t y = job % 66, c1 = (job / 66) % 4, b = job / (4 * 66);
        if (y == 0 || y == 65) {
            Duplicate(rowOut, 0.f, 512);
            PipeBarrier<PIPE_ALL>();
        } else {
            // Eight channel rows: source gap is (H*W-W)*4 bytes, destination is contiguous.
            DataCopyPad(rowIn, src[((b * 32 + c1 * 8) * 64 + (y - 1)) * 64], DataCopyExtParams{8, 256, 16128, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0.f});
            PipeBarrier<PIPE_ALL>();
            Gather(rowOut, rowIn, rowIndexBuf.Get<uint32_t>(), 0, 512);
            PipeBarrier<PIPE_ALL>();
        }
        DataCopy(dst[job * 512], rowOut, 512);
        PipeBarrier<PIPE_ALL>();
    }
    if (worker == 0) {
        auto original = weightInBuf.Get<float>();
        auto target = weightOutBuf.Get<float>();
        auto indices = weightIndexBuf.Get<int32_t>();
        DataCopy(original, weights, 1152);
        Duplicate(target, 0.f, 4608);
        PipeBarrier<PIPE_ALL>();
        // Reuse four head groups; update only after the previous Gather has read them.
        Arange(indices, int32_t(0), int32_t(36), int32_t(8));
        PipeBarrier<PIPE_V>();
        Adds(indices[8], indices, int32_t(1152), int32_t(8));
        PipeBarrier<PIPE_V>();
        Adds(indices[16], indices, int32_t(2304), int32_t(16));
        for (uint32_t k1 = 0; k1 < 36; ++k1) {
            PipeBarrier<PIPE_V>();
            Gather(target[k1 * 128], original, weightIndexBuf.Get<uint32_t>(), 0, 32);
            PipeBarrier<PIPE_V>();
            if (k1 + 1 < 36)
                Adds(indices, indices, int32_t(k1 % 9 == 8 ? 256 : 4), int32_t(32));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(nz, target, 4608);
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void RunA2Decode(TPipe &pipe, GM_ADDR input, GM_ADDR bias, GM_ADDR points, GM_ADDR scores,
                                   uint32_t maps, uint32_t height, uint32_t width, float tau, uint32_t worker,
                                   uint32_t workers) {
    TQue<TPosition::VECIN, 1> in;
    TQue<TPosition::VECOUT, 1> out;
    TBuf<TPosition::VECCALC> coordBuf, rowBuf, productBuf, scratchBuf, scalarBuf, biasBuf, xBuf;
    uint32_t n = height * width;
    pipe.InitBuffer(in, 1, n * 16);
    pipe.InitBuffer(xBuf, n * 4);
    pipe.InitBuffer(out, 1, 64);
    pipe.InitBuffer(coordBuf, n * 4);
    pipe.InitBuffer(rowBuf, n * 4);
    pipe.InitBuffer(productBuf, n * 4);
    pipe.InitBuffer(scratchBuf, n * 4);
    pipe.InitBuffer(scalarBuf, 32);
    pipe.InitBuffer(biasBuf, 32);
    auto coord = coordBuf.Get<float>();
    auto row = rowBuf.Get<float>();
    auto product = productBuf.Get<float>();
    auto scratch = scratchBuf.Get<float>();
    auto scalar = scalarBuf.Get<float>();
    auto biasValues = biasBuf.Get<float>();
    // Broadcast a single row/column pattern; avoid building and dividing n indices.
    Arange(coord, 0.0f, 1.0f, width);
    Arange(scratch, 0.0f, 1.0f, height);
    PipeBarrier<PIPE_V>();
    UnaryRepeatParams xParams(1, 1, width / 8, 0);
    Adds(coord[width], coord, 0.0f, uint64_t(64), height - 1, xParams);
    if (width == 96) {
        Adds(coord[width + 64], coord[64], 0.0f, uint64_t(32), height - 1, xParams);
    }
    BrcbRepeatParams yParams;
    yParams.dstBlkStride = width / 8;
    yParams.dstRepStride = width;
    for (uint32_t block = 0; block < width / 8; ++block) {
        Brcb(row[block * 8], scratch, height / 8, yParams);
    }
    PipeBarrier<PIPE_V>();
    GlobalTensor<float> src, biasGm, dst, score;
    src.SetGlobalBuffer((__gm__ float *)input, maps * n);
    biasGm.SetGlobalBuffer((__gm__ float *)bias, 4);
    dst.SetGlobalBuffer((__gm__ float *)points, maps * 2);
    score.SetGlobalBuffer((__gm__ float *)scores, maps);
    DataCopyPad(biasValues, biasGm, DataCopyExtParams{1, 16, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0.f});
    PipeBarrier<PIPE_ALL>();
    for (uint32_t map = worker; map < maps; map += workers) {
        auto raw = in.AllocTensor<float>();
        DataCopy(raw, src[(map / 4) * n * 4], n * 4);
        in.EnQue(raw);
        raw = in.DeQue<float>();
        auto x = xBuf.Get<float>();
        // Fixed four-lane compaction avoids producing and reading an index tensor.
        uint64_t reservedCount = 0;
        const uint8_t pattern = uint8_t(3 + map % 4);
        for (uint32_t offset = 0; offset < n; offset += 2048) {
            const uint32_t take = n - offset < 2048 ? n - offset : 2048;
            GatherMask(x[offset], raw[offset * 4], pattern, false, uint32_t(0),
                       GatherMaskParams{1, uint16_t(take / 16), 8, 0}, reservedCount);
        }
        PipeBarrier<PIPE_V>();
        ReduceMax(scalar, x, scratch, n, false);
        PipeBarrier<PIPE_ALL>();
        float maximum = scalar.GetValue(0);
        Duplicate(product, tau > 1.0e30f ? tau * 0.5f : tau, n);
        PipeBarrier<PIPE_V>();
        if (tau > 1.0e30f) {
            // Exact half scaling preserves nearby values and bounds the subtraction.
            Muls(x, x, 0.5f, n);
            PipeBarrier<PIPE_V>();
            Adds(x, x, -(maximum * 0.5f), n);
            PipeBarrier<PIPE_V>();
            Div(x, x, product, n);
        } else {
            // Subtract first so small positive tau never overflows the maximum.
            Adds(x, x, -maximum, n);
            PipeBarrier<PIPE_V>();
            Div(x, x, product, n);
        }
        PipeBarrier<PIPE_V>();
        Exp(x, x, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, x, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float denominator = scalar.GetValue(0);
        Mul(product, x, row, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, product, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float y = scalar.GetValue(0) / denominator;
        Mul(product, x, coord, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, product, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float xcoord = scalar.GetValue(0) / denominator;
        auto result = out.AllocTensor<float>();
        result.SetValue(0, (xcoord + 0.5f) * 4.0f);
        result.SetValue(1, (y + 0.5f) * 4.0f);
        result.SetValue(8, maximum + biasValues.GetValue(map % 4));
        out.EnQue(result);
        result = out.DeQue<float>();
        DataCopyPad(dst[map * 2], result, DataCopyExtParams{1, 8, 0, 0, 0});
        DataCopyPad(score[map], result[8], DataCopyExtParams{1, 4, 0, 0, 0});
        out.FreeTensor(result);
        in.FreeTensor(raw);
    }
}

#endif

#if defined(__DAV_CUBE__) || defined(SPLIT_CORE_CUBE)
__aicore__ inline void RunProjectionCube(TPipe &pipe, GM_ADDR feat, GM_ADDR packed, const uint32_t *cubeWords,
                                         uint32_t batch, uint32_t count) {
    TCubeTiling t;
    auto src = cubeWords;
    auto dst = (uint32_t *)&t;
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / 4; ++i)
        dst[i] = src[i];
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float>, MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>>
        mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &t);
    GlobalTensor<float> a, f, p;
    a.SetGlobalBuffer((__gm__ float *)packed);
    f.SetGlobalBuffer((__gm__ float *)feat);
    p.SetGlobalBuffer((__gm__ float *)packed + kProjectionWeightFloats);
    mm.SetHF32(false);
    uint32_t nt = count / kCubeTileN;
    for (uint32_t job = GetBlockIdx(); job < batch * nt; job += GetBlockNum()) {
        uint32_t b = job / nt, n = (job % nt) * kCubeTileN;
        mm.SetOrgShape(kProjectionRows, count, kProjectionChannels);
        mm.SetSingleShape(kProjectionRows, kCubeTileN, kProjectionChannels);
        mm.SetTensorA(a);
        mm.SetTensorB(f[b * kProjectionChannels * count + n]);
        mm.IterateAll(p[b * kProjectionRows * count + n]);
        mm.End();
    }
}

__aicore__ inline void RunA2Cube(TPipe &pipe, GM_ADDR packed, GM_ADDR weight, GM_ADDR output, uint32_t batch,
                                 uint32_t worker, uint32_t workers) {
    TBuf<TPosition::A1> l1Buffer;
    TBuf<TPosition::A2> aBuffer;
    TBuf<TPosition::B2> bBuffer;
    TBuf<TPosition::CO1> cBuffer;
    pipe.InitBuffer(l1Buffer, 55296);
    pipe.InitBuffer(aBuffer, 36864);
    pipe.InitBuffer(bBuffer, 18432);
    pipe.InitBuffer(cBuffer, 2048);
    const auto mToMte1Event = pipe.AllocEventID<HardEvent::M_MTE1>();
    auto l1 = l1Buffer.Get<float>();
    auto a = aBuffer.Get<float>();
    auto b = bBuffer.Get<float>();
    auto c = cBuffer.Get<float>();
    GlobalTensor<float> fGm, wGm, outGm;
    fGm.SetGlobalBuffer((__gm__ float *)packed);
    wGm.SetGlobalBuffer((__gm__ float *)weight);
    outGm.SetGlobalBuffer((__gm__ float *)output);
    SetHF32Mode(HF32Mode::DISABLE);
    DataCopy(l1[9216], wGm, 4608);
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    LoadData2DParams bLoad;
    bLoad.startIndex = 0;
    bLoad.repeatTimes = 36;
    bLoad.srcStride = 1;
    bLoad.dstGap = 0;
    bLoad.sid = 0;
    bLoad.ifTranspose = false;
    bLoad.addrMode = 0;
    LoadData(b, l1[9216], bLoad);
    for (uint32_t job = worker; job < batch * 96; job += workers) {
        const uint32_t image = job / 96, y = (job % 96) / 2, x = (job % 2) * 32;
        // Four C1 slabs: each3*64*8 floats, stride is full50-row packed C1 frame.
        DataCopy(l1, fGm[(image * 4 * 50 + y) * 64 * 8], DataCopyParams{4, 192, 3008, 0});
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadData3DParamsV2<float> load;
        for (uint32_t i = 0; i < 4; ++i)
            load.padList[i] = 1;
        load.l1H = 3;
        load.l1W = 64;
        load.channelSize = 32;
        load.kExtension = 288;
        load.mExtension = 32;
        load.kStartPt = 0;
        load.mStartPt = 64 + x;
        load.strideW = 1;
        load.strideH = 1;
        load.filterW = 3;
        load.filterH = 3;
        load.dilationFilterW = 1;
        load.dilationFilterH = 1;
        load.enTranspose = false;
        load.enSmallK = false;
        load.padValue = 0.f;
        load.filterSizeW = false;
        load.filterSizeH = false;
        load.fMatrixCtrl = false;
        LoadData(a, l1, load);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams mm;
        mm.m = 32;
        mm.n = 16;
        mm.k = 288;
        mm.isBias = false;
        mm.cmatrixSource = false;
        mm.cmatrixInitVal = true;
        Mmad(c, a, b, mm);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 fix;
        fix.nSize = 4;
        fix.mSize = 32;
        fix.srcStride = 32;
        fix.dstStride = 4;
        fix.quantPre = QuantMode_t::NoQuant;
        fix.deqScalar = 0;
        fix.ndNum = 1;
        fix.srcNdStride = 0;
        fix.dstNdStride = 0;
        fix.reluEn = false;
        fix.unitFlag = 0;
        fix.isChannelSplit = false;
        Fixpipe<float, float, CFG_ROW_MAJOR>(outGm[(image * 3072 + y * 64 + x) * 4], c, fix);
        SetFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::M_MTE1>(mToMte1Event);
        WaitFlag<HardEvent::M_MTE1>(mToMte1Event);
        SetFlag<HardEvent::MTE1_MTE2>(0);
        WaitFlag<HardEvent::MTE1_MTE2>(0);
    }
    PipeBarrier<PIPE_ALL>();
    pipe.ReleaseEventID<HardEvent::M_MTE1>(mToMte1Event);
}

__aicore__ inline void RunA2Cube64B1(TPipe &pipe, GM_ADDR packed, GM_ADDR weight, GM_ADDR output, uint32_t batch,
                                     uint32_t worker, uint32_t workers) {
    TBuf<TPosition::A1> l1Buffer;
    TBuf<TPosition::A2> aBuffer;
    TBuf<TPosition::B2> bBuffer;
    TBuf<TPosition::CO1> cBuffer;
    pipe.InitBuffer(l1Buffer, 55296);
    pipe.InitBuffer(aBuffer, 36864);
    pipe.InitBuffer(bBuffer, 18432);
    pipe.InitBuffer(cBuffer, 2048);
    const auto mToMte1Event = pipe.AllocEventID<HardEvent::M_MTE1>();
    auto l1 = l1Buffer.Get<float>();
    auto a = aBuffer.Get<float>();
    auto b = bBuffer.Get<float>();
    auto c = cBuffer.Get<float>();
    GlobalTensor<float> fGm, wGm, outGm;
    fGm.SetGlobalBuffer((__gm__ float *)packed);
    wGm.SetGlobalBuffer((__gm__ float *)weight);
    outGm.SetGlobalBuffer((__gm__ float *)output);
    SetHF32Mode(HF32Mode::DISABLE);
    DataCopy(l1[9216], wGm, 4608);
    SetFlag<HardEvent::MTE2_MTE1>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(0);
    LoadData2DParams bLoad;
    bLoad.startIndex = 0;
    bLoad.repeatTimes = 36;
    bLoad.srcStride = 1;
    bLoad.dstGap = 0;
    bLoad.sid = 0;
    bLoad.ifTranspose = false;
    bLoad.addrMode = 0;
    LoadData(b, l1[9216], bLoad);
    for (uint32_t job = worker; job < batch * 128; job += workers) {
        const uint32_t image = job / 128, y = (job % 128) / 2, x = (job % 2) * 32;
        // Four C1 slabs: each3*64*8 floats, stride is full66-row packed C1 frame.
        DataCopy(l1, fGm[(image * 4 * 66 + y) * 64 * 8], DataCopyParams{4, 192, 4032, 0});
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadData3DParamsV2<float> load;
        for (uint32_t i = 0; i < 4; ++i)
            load.padList[i] = 1;
        load.l1H = 3;
        load.l1W = 64;
        load.channelSize = 32;
        load.kExtension = 288;
        load.mExtension = 32;
        load.kStartPt = 0;
        load.mStartPt = 64 + x;
        load.strideW = 1;
        load.strideH = 1;
        load.filterW = 3;
        load.filterH = 3;
        load.dilationFilterW = 1;
        load.dilationFilterH = 1;
        load.enTranspose = false;
        load.enSmallK = false;
        load.padValue = 0.f;
        load.filterSizeW = false;
        load.filterSizeH = false;
        load.fMatrixCtrl = false;
        LoadData(a, l1, load);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams mm;
        mm.m = 32;
        mm.n = 16;
        mm.k = 288;
        mm.isBias = false;
        mm.cmatrixSource = false;
        mm.cmatrixInitVal = true;
        Mmad(c, a, b, mm);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        FixpipeParamsV220 fix;
        fix.nSize = 4;
        fix.mSize = 32;
        fix.srcStride = 32;
        fix.dstStride = 4;
        fix.quantPre = QuantMode_t::NoQuant;
        fix.deqScalar = 0;
        fix.ndNum = 1;
        fix.srcNdStride = 0;
        fix.dstNdStride = 0;
        fix.reluEn = false;
        fix.unitFlag = 0;
        fix.isChannelSplit = false;
        Fixpipe<float, float, CFG_ROW_MAJOR>(outGm[(image * 4096 + y * 64 + x) * 4], c, fix);
        SetFlag<HardEvent::FIX_M>(0);
        WaitFlag<HardEvent::FIX_M>(0);
        SetFlag<HardEvent::M_MTE1>(mToMte1Event);
        WaitFlag<HardEvent::M_MTE1>(mToMte1Event);
        SetFlag<HardEvent::MTE1_MTE2>(0);
        WaitFlag<HardEvent::MTE1_MTE2>(0);
    }
    PipeBarrier<PIPE_ALL>();
    pipe.ReleaseEventID<HardEvent::M_MTE1>(mToMte1Event);
}

#endif

// H64-only complete FP32 owner path. Host must validate B1..4, finite tau>0, stride4,
// launch pure AIV with min(40, actualAIV, B*4), and retain the generated ABI.
// Explicit UB: n + 4*n + 1152 floats, two 32B buffers, one 64B output queue = 86656B.
#if defined(__DAV_VEC__) || defined(SPLIT_CORE_VEC)
__aicore__ inline void RunOwnerH64(TPipe &pipe, GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR points,
                                   GM_ADDR scores, uint32_t batch, float tau) {
    constexpr uint32_t height = 64, width = 64, n = height * width;
    constexpr uint32_t stripeCount = (height + 2) * width;
    static_assert(3 * stripeCount <= 4 * n, "Conv planes must fit the decode work arena");
    static_assert(n % 64 == 0 && n / 64 <= 255);
    TBuf<TPosition::VECCALC> heatmapBuf, workBuf, stripeAltBuf, weightBuf, biasBuf, scalarBuf;
    TQue<TPosition::VECOUT, 1> out;
    pipe.InitBuffer(heatmapBuf, n * 4);
    pipe.InitBuffer(workBuf, 4 * n * 4);
    pipe.InitBuffer(stripeAltBuf, 3 * stripeCount * 4);
    pipe.InitBuffer(weightBuf, 1152 * 4);
    pipe.InitBuffer(biasBuf, 32);
    pipe.InitBuffer(scalarBuf, 32);
    pipe.InitBuffer(out, 1, 64);
    auto x = heatmapBuf.Get<float>();
    auto work = workBuf.Get<float>();
    auto alternate = stripeAltBuf.Get<float>();
    auto w = weightBuf.Get<float>();
    auto biasValues = biasBuf.Get<float>();
    auto scalar = scalarBuf.Get<float>();
    // These four disjoint decode views become live only after the convolution drain.
    auto coord = work;
    auto row = work[n];
    auto product = work[2 * n];
    auto scratch = work[3 * n];
    GlobalTensor<float> fGm, wGm, biasGm, dst, score;
    fGm.SetGlobalBuffer((__gm__ float *)feat, batch * 32 * n);
    wGm.SetGlobalBuffer((__gm__ float *)weight, 1152);
    biasGm.SetGlobalBuffer((__gm__ float *)bias, 4);
    dst.SetGlobalBuffer((__gm__ float *)points, batch * 4 * 2);
    score.SetGlobalBuffer((__gm__ float *)scores, batch * 4);
    DataCopy(w, wGm, 1152);
    DataCopyPad(biasValues, biasGm, DataCopyExtParams{1, 16, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0.f});
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
    for (uint32_t map = GetBlockIdx(); map < batch * 4; map += GetBlockNum()) {
        const uint32_t b = map / 4, k = map % 4;
        Duplicate(x, 0.f, n);
        // Full H rows populate all valid/padded columns for every channel;
        // inactive top/bottom rows remain zero until the end of this owner.
        Duplicate(work, 0.f, 3 * stripeCount);
        Duplicate(alternate, 0.f, 3 * stripeCount);
        PipeBarrier<PIPE_ALL>();
        const uint32_t initialSrc = b * 32 * n;
        DataCopyPad(work[width], fGm[initialSrc], DataCopyExtParams{uint16_t(height), (width - 1) * 4, 4, 0, 0},
                    DataCopyPadExtParams<float>{true, 1, 0, 0.f});
        DataCopyPad(work[stripeCount + width], fGm[initialSrc], DataCopyExtParams{uint16_t(height), width * 4, 0, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 0, 0.f});
        DataCopyPad(work[2 * stripeCount + width], fGm[initialSrc + 1],
                    DataCopyExtParams{uint16_t(height), (width - 1) * 4, 4, 0, 0},
                    DataCopyPadExtParams<float>{true, 0, 1, 0.f});
        SetFlag<HardEvent::MTE2_V>(0);
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t slot = c & 1u;
            WaitFlag<HardEvent::MTE2_V>(slot);
            if (c + 1 < 32) {
                const uint32_t nextSlot = slot ^ 1u;
                if (c != 0)
                    WaitFlag<HardEvent::V_MTE2>(nextSlot);
                auto nextStripes = nextSlot == 0 ? work : alternate;
                const uint32_t src = (b * 32 + c + 1) * n;
                DataCopyPad(nextStripes[width], fGm[src], DataCopyExtParams{uint16_t(height), (width - 1) * 4, 4, 0, 0},
                            DataCopyPadExtParams<float>{true, 1, 0, 0.f});
                DataCopyPad(nextStripes[stripeCount + width], fGm[src],
                            DataCopyExtParams{uint16_t(height), width * 4, 0, 0, 0},
                            DataCopyPadExtParams<float>{true, 0, 0, 0.f});
                DataCopyPad(nextStripes[2 * stripeCount + width], fGm[src + 1],
                            DataCopyExtParams{uint16_t(height), (width - 1) * 4, 4, 0, 0},
                            DataCopyPadExtParams<float>{true, 0, 1, 0.f});
                SetFlag<HardEvent::MTE2_V>(nextSlot);
            }
            auto stripes = slot == 0 ? work : alternate;
            for (uint32_t ky = 0; ky < 3; ++ky) {
                for (uint32_t kx = 0; kx < 3; ++kx) {
                    auto input = stripes[kx * stripeCount + ky * width];
                    const uint32_t tap = c * 9 + ky * 3 + kx;
                    Axpy<float, float, true>(x, input, w.GetValue(k * 288 + tap), uint64_t(64), uint8_t(n / 64),
                                             UnaryRepeatParams{1, 1, 8, 8});
                    PipeBarrier<PIPE_V>();
                }
            }
            SetFlag<HardEvent::V_MTE2>(slot);
        }
        WaitFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(1);
        PipeBarrier<PIPE_ALL>();
        // Broadcast a single row/column pattern; avoid building and dividing n indices.
        Arange(coord, 0.0f, 1.0f, width);
        Arange(scratch, 0.0f, 1.0f, height);
        PipeBarrier<PIPE_V>();
        UnaryRepeatParams xParams(1, 1, width / 8, 0);
        Adds(coord[width], coord, 0.0f, uint64_t(64), height - 1, xParams);
        if (width == 96) {
            Adds(coord[width + 64], coord[64], 0.0f, uint64_t(32), height - 1, xParams);
        }
        BrcbRepeatParams yParams;
        yParams.dstBlkStride = width / 8;
        yParams.dstRepStride = width;
        for (uint32_t block = 0; block < width / 8; ++block) {
            Brcb(row[block * 8], scratch, height / 8, yParams);
        }
        PipeBarrier<PIPE_V>();
        ReduceMax(scalar, x, scratch, n, false);
        PipeBarrier<PIPE_ALL>();
        float maximum = scalar.GetValue(0);
        Duplicate(product, tau > 1.0e30f ? tau * 0.5f : tau, n);
        PipeBarrier<PIPE_V>();
        if (tau > 1.0e30f) {
            // Exact half scaling preserves nearby values and bounds the subtraction.
            Muls(x, x, 0.5f, n);
            PipeBarrier<PIPE_V>();
            Adds(x, x, -(maximum * 0.5f), n);
            PipeBarrier<PIPE_V>();
            Div(x, x, product, n);
        } else {
            // Subtract first so small positive tau never overflows the maximum.
            Adds(x, x, -maximum, n);
            PipeBarrier<PIPE_V>();
            Div(x, x, product, n);
        }
        PipeBarrier<PIPE_V>();
        Exp(x, x, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, x, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float denominator = scalar.GetValue(0);
        Mul(product, x, row, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, product, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float y = scalar.GetValue(0) / denominator;
        Mul(product, x, coord, n);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, product, scratch, n);
        PipeBarrier<PIPE_ALL>();
        float xcoord = scalar.GetValue(0) / denominator;
        auto result = out.AllocTensor<float>();
        result.SetValue(0, (xcoord + 0.5f) * 4.0f);
        result.SetValue(1, (y + 0.5f) * 4.0f);
        result.SetValue(8, maximum + biasValues.GetValue(map % 4));
        out.EnQue(result);
        result = out.DeQue<float>();
        DataCopyPad(dst[map * 2], result, DataCopyExtParams{1, 8, 0, 0, 0});
        DataCopyPad(score[map], result[8], DataCopyExtParams{1, 4, 0, 0, 0});
        out.FreeTensor(result);
        // The original output queue owns its MTE3 lifetime. Drain before the next owner.
        PipeBarrier<PIPE_ALL>();
    }
}
#endif

template <typename DT_FEAT, uint32_t ROUTE>
__global__ __aicore__ void fused_disconnector_keypoint_soft_argmax(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias,
                                                                   GM_ADDR keypoints, GM_ADDR scores,
                                                                   GM_ADDR user_workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FusedDisconnectorKeypointSoftArgmaxTilingData);
    using DeviceTilingData = typename Conditional<(ROUTE == 1), FusedDisconnectorKeypointSoftArgmaxTilingData,
                                                  FusedDisconnectorKeypointSoftArgmaxVectorPrefix>::type;
    GET_TILING_DATA_WITH_STRUCT(DeviceTilingData, t, tiling);
    // Generated wrapper already supplies usrWorkspace; keep the system prefix intact.
    TPipe pipe;
    if constexpr (ROUTE == 3) {
#if defined(__DAV_VEC__) || defined(SPLIT_CORE_VEC)
        if (t.height == 48)
            RunOwnerH48(pipe, feat, weight, bias, keypoints, scores, t.batch, t.tau);
        else
            RunOwnerH64(pipe, feat, weight, bias, keypoints, scores, t.batch, t.tau);
#endif
    } else if constexpr (ROUTE == 1) {
#if defined(__DAV_VEC__) || defined(SPLIT_CORE_VEC)
        if (GetBlockIdx() == 0)
            RunProjectionPack(pipe, weight, user_workspace);
#endif
        AscendC::SyncAll<false>();
#if defined(__DAV_CUBE__) || defined(SPLIT_CORE_CUBE)
        RunProjectionCube(pipe, feat, user_workspace, t.cube_words, t.batch, t.height * t.width);
#endif
        AscendC::SyncAll<false>();
#if defined(__DAV_VEC__) || defined(SPLIT_CORE_VEC)
        pipe.Reset();
        if (GetBlockIdx() < t.batch * 4)
            RunProjectionShiftDecode(pipe, user_workspace, bias, keypoints, scores, t.batch * 4, t.height, t.width,
                                     t.tau, GetBlockIdx(), 2 * t.groups);
#endif
    } else if constexpr (ROUTE == 2) {
        const uint32_t groups = GetBlockNum();
        GM_ADDR nzWeights = user_workspace + uint64_t(t.batch) * 32 * (t.height + 2) * t.width * 4;
        GM_ADDR heatmap = nzWeights + 18432;
#if defined(__DAV_VEC__) || defined(SPLIT_CORE_VEC)
        if (t.height == 48)
            RunA2Pack(pipe, feat, weight, user_workspace, nzWeights, t.batch, GetBlockIdx(), 2 * groups);
        else
            RunA2Pack64B1(pipe, feat, weight, user_workspace, nzWeights, t.batch, GetBlockIdx(), 2 * groups);
#endif
        AscendC::SyncAll<false>();
#if defined(__DAV_CUBE__) || defined(SPLIT_CORE_CUBE)
        if (t.height == 48)
            RunA2Cube(pipe, user_workspace, nzWeights, heatmap, t.batch, GetBlockIdx(), groups);
        else
            RunA2Cube64B1(pipe, user_workspace, nzWeights, heatmap, t.batch, GetBlockIdx(), groups);
#endif
        AscendC::SyncAll<false>();
#if defined(__DAV_VEC__) || defined(SPLIT_CORE_VEC)
        pipe.Reset();
        RunA2Decode(pipe, heatmap, bias, keypoints, scores, t.batch * 4, t.height, t.width, t.tau, GetBlockIdx(),
                    2 * groups);
#endif
    } else {
#if defined(__DAV_VEC__) || defined(SPLIT_CORE_VEC)
        GM_ADDR heatmap = user_workspace;
        if (t.height == 72 && t.width == 96) {
            switch (t.batch) {
            case 1:
                RunStripeOut2B1<1>(pipe, feat, weight, bias, heatmap);
                break;
            case 2:
                RunStripePrefetch72<2>(pipe, feat, weight, bias, heatmap);
                break;
            case 3:
                RunStripePrefetch72<3>(pipe, feat, weight, bias, heatmap);
                break;
            case 4:
                RunStripePrefetch72<4>(pipe, feat, weight, bias, heatmap);
                break;
            }
        } else if (t.height == 64 && t.width == 64) {
            switch (t.batch) {
            case 1:
                RunStripeDouble64<1>(pipe, feat, weight, bias, heatmap);
                break;
            case 2:
                RunStripeDouble64<2>(pipe, feat, weight, bias, heatmap);
                break;
            case 3:
                RunPair2B3(pipe, feat, weight, bias, heatmap);
                break;
            case 4:
                RunStripeDouble64<4>(pipe, feat, weight, bias, heatmap);
                break;
            }
        } else if (t.height == 48 && t.width == 64) {
            switch (t.batch) {
            case 1:
                RunStripePrefetch48<1>(pipe, feat, weight, bias, heatmap);
                break;
            case 2:
                RunStripePrefetch48<2>(pipe, feat, weight, bias, heatmap);
                break;
            case 3:
                RunStripePrefetch48<3>(pipe, feat, weight, bias, heatmap);
                break;
            case 4:
                RunStripePrefetch48<4>(pipe, feat, weight, bias, heatmap);
                break;
            }
        } else {
            RunConvolution(pipe, feat, weight, bias, heatmap, t.batch, t.height, t.width, t.rows);
        }
        // Every launched AIV participates, including blocks with no decode map.
        AscendC::SyncAll<true>();
        pipe.Reset();
        RunDecode(pipe, heatmap, bias, keypoints, scores, t.batch * 4, t.height, t.width, t.tau);
#endif
    }
}
