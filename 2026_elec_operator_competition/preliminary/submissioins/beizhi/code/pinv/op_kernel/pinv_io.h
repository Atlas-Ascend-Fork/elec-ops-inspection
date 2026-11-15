/*
 * Copyright (c) 2026 Beizhi contributors.
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

template <typename T>
__aicore__ inline void PinvWriteOutput(GlobalTensor<T> output, LocalTensor<T> raw, LocalTensor<float> src, uint32_t at,
                                       uint32_t len)
{
    PipeBarrier<PIPE_ALL>();
    if constexpr (sizeof(T) == 4)
        DataCopyPad(output[at], src, {1, len * 4, 0, 0, 0});
    else {
        Cast(raw, src, RoundMode::CAST_RINT, len);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(output[at], raw, {1, len * 2, 0, 0, 0});
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename Unused = void>
__aicore__ inline void PinvTinySum(LocalTensor<float> mem, LocalTensor<float> dst, LocalTensor<float> src, uint32_t len)
{
    auto first = mem[30000], second = mem[30256];
    uint32_t full = len / 64, tail = len % 64;
    if (full)
        WholeReduceSum(first, src, 64, full, 1, 1, 8);
    if (tail)
        WholeReduceSum(first[full], src[full * 64], tail, 1, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    uint32_t count = full + (tail ? 1 : 0);
    if (count <= 64)
        WholeReduceSum(dst, first, count, 1, 1, 1, 8);
    else {
        full = count / 64;
        tail = count % 64;
        WholeReduceSum(second, first, 64, full, 1, 1, 8);
        if (tail)
            WholeReduceSum(second[full], first[full * 64], tail, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dst, second, full + (tail ? 1 : 0), 1, 1, 1, 8);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline bool PinvTinyEliminate(uint32_t n, uint32_t span, uint32_t width, LocalTensor<float> mat,
                                         LocalTensor<float> u, LocalTensor<float> v, LocalTensor<float> h,
                                         LocalTensor<float> tmp, LocalTensor<float> red, float anorm,
                                         LocalTensor<uint32_t> offsets)
{
    for (uint32_t j = 0; j < n; j++) {
        Gather(u, mat, offsets, j * 4, n);
        PipeBarrier<PIPE_V>();
        Abs(tmp, u, n);
        PipeBarrier<PIPE_V>();
        uint64_t mask[2] = {((~0ULL) << j) & (n == 64 ? ~0ULL : ((1ULL << n) - 1)), 0};
        WholeReduceMax(red, tmp, mask, 1, 1, 1, 8);
        LocalSync<HardEvent::V_S>();
        float mx = red.GetValue(0);
        uint32_t pi = red.template ReinterpretCast<uint32_t>().GetValue(1);
        if (mx * mx < anorm * 1e-10f)
            return false;
        float inv = 1.0f / u.GetValue(pi);
        Muls(v, mat[pi * width], -inv, span);
        Brcb(h, u, (n + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        for (uint32_t c = j / 8 * 8; c < span; c += 64) {
            uint32_t count = span - c < 64 ? span - c : 64;
            MulAddDst(mat[c], v[c], h, count, n, {1, 1, 0, (uint8_t)(width / 8), 0, 1});
        }
        PipeBarrier<PIPE_V>();
        if (pi != j) {
            Adds(mat[pi * width], mat[j * width], 0.0f, width);
            PipeBarrier<PIPE_V>();
        }
        Muls(mat[j * width], v, -1.0f, span);
        PipeBarrier<PIPE_V>();
    }
    return true;
}

template <typename T>
__aicore__ inline void PinvLoadTiny(GlobalTensor<T> input, LocalTensor<T> raw, uint32_t base, uint32_t n, uint32_t np,
                                    uint32_t width, LocalTensor<float> mat)
{
    if constexpr (sizeof(T) == 4)
        DataCopyPad(mat, input[base], {(uint16_t)n, n * 4, 0, (width - np) / 8, 0}, {true, 0, (uint8_t)(np - n), 0});
    else
        for (uint32_t i = 0; i < n; i++) {
            DataCopyPad(raw, input[base + i * n], {1, n * 2, 0, 0, 0}, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Cast(mat[i * width], raw, RoundMode::CAST_NONE, n);
        }
}

template <typename T>
__aicore__ inline void PinvReadInput(GlobalTensor<T> input, LocalTensor<T> raw, LocalTensor<uint32_t> offsets,
                                     LocalTensor<uint32_t> bfOffsets, LocalTensor<float> converted,
                                     LinalgPinvTilingData t, uint32_t base, uint32_t col, LocalTensor<float> dst)
{
    Duplicate(dst, 0.0f, t.qp);
    PipeBarrier<PIPE_ALL>();
    if (t.m < t.n) {
        if constexpr (sizeof(T) == 4)
            DataCopyPad(dst, input[base + col * t.q], {1, t.q * 4, 0, 0, 0}, {true, 0, (uint8_t)(t.qp - t.q), 0});
        else {
            DataCopyPad(raw, input[base + col * t.q], {1, t.q * 2, 0, 0, 0}, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Cast(dst, raw, RoundMode::CAST_NONE, t.q);
        }
    } else {
        if constexpr (sizeof(T) == 4) {
            DataCopyPad(raw, input[base + col], {(uint16_t)t.q, 4, (t.p - 1) * 4, 0, 0}, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Gather(dst, raw.template ReinterpretCast<float>(), offsets, 0, t.q);
        } else
            for (uint32_t start = 0; start < t.q; start += 32) {
                uint32_t count = t.q - start < 32 ? t.q - start : 32;
                DataCopyPad(raw, input[base + col + start * t.p], {(uint16_t)count, 2, (t.p - 1) * 2, 0, 0},
                            {false, 0, 0, 0});
                PipeBarrier<PIPE_ALL>();
                Cast(converted, raw, RoundMode::CAST_NONE, count * 16);
                PipeBarrier<PIPE_ALL>();
                Gather(dst[start], converted, bfOffsets, 0, count);
                PipeBarrier<PIPE_ALL>();
            }
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void PinvStoreTiny(GlobalTensor<T> output, LocalTensor<T> raw, LocalTensor<float> mat, uint32_t base,
                                     uint32_t n, uint32_t np, uint32_t width)
{
    if constexpr (sizeof(T) == 4) {
        LocalSync<HardEvent::V_MTE3>();
        DataCopyPad(output[base], mat[np], {(uint16_t)n, n * 4, (width - np) / 8, 0, 0});
        LocalSync<HardEvent::MTE3_MTE2>();
    } else
        for (uint32_t i = 0; i < n; i++)
            PinvWriteOutput(output, raw, mat[i * width + np], base + i * n, n);
}
