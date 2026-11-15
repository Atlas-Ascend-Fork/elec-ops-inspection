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

template <class T, uint32_t N, bool StaticShape> __aicore__ inline void PinvInverseState<T, N, StaticShape>::Fence()
{
    PipeBarrier<PIPE_ALL>();
}

template <class T, uint32_t N, bool StaticShape> __aicore__ inline void PinvInverseState<T, N, StaticShape>::Sync()
{
    Fence();
    if (t.group * t.groups > 1)
        SyncAll();
    Fence();
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::Store(GlobalTensor<float> dst, float a, float b, float c)
{
    red.SetValue(0, a);
    red.SetValue(1, b);
    red.SetValue(2, c);
    LocalSync<HardEvent::S_MTE3>();
    DataCopy(dst, red, 8);
    LocalSync<HardEvent::MTE3_S>();
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::Load(GlobalTensor<float> src)
{
    DataCopy(red, src, 8);
    LocalSync<HardEvent::MTE2_S>();
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::Sum(LocalTensor<float> dst, LocalTensor<float> src)
{
    uint32_t full = t.pp / 64, tail = t.pp % 64;
    auto sums = mat[41536];
    if (full)
        WholeReduceSum(sums, src, 64, full, 1, 1, 8);
    if (tail)
        WholeReduceSum(sums[full], src[full * 64], tail, 1, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    WholeReduceSum(dst, sums, full + (tail ? 1 : 0), 1, 1, 1, 8);
    PipeBarrier<PIPE_V>();
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline float PinvInverseState<T, N, StaticShape>::Dot(LocalTensor<float> x)
{
    Mul(tmp, x, x, t.pp);
    PipeBarrier<PIPE_V>();
    Sum(red, tmp);
    LocalSync<HardEvent::V_S>();
    return red.GetValue(0);
}

template <class T, uint32_t N, bool StaticShape> __aicore__ inline uint32_t PinvInverseState<T, N, StaticShape>::Pivot()
{
    if constexpr (N == 256) {
        WholeReduceMax<float, false>(mat[40000], tmp, 64, 4, 1, 1, 8);
        LocalSync<HardEvent::V_S>();
        float maximum, indexBits;
        GetReduceMaxMinCount(maximum, indexBits);

        union IndexBits {
            float value;
            uint32_t bits;
        };

        IndexBits selected;
        selected.value = indexBits;
        return selected.bits;
    }
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, t.p);
    WholeReduceMax<float, false>(mat[40000], tmp, 64, 1, 1, 1, 8);
    LocalSync<HardEvent::V_S>();
    float maximum, indexBits;
    GetReduceMaxMinCount(maximum, indexBits);
    SetMaskNorm();
    ResetMask();

    union IndexBits {
        float value;
        uint32_t bits;
    };

    IndexBits selected;
    selected.value = indexBits;
    return selected.bits;
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline float PinvInverseState<T, N, StaticShape>::NormRoot(float z)
{
    red.SetValue(0, z > 0 ? z : 0);
    LocalSync<HardEvent::S_V>();
    Sqrt(red, red, 1);
    LocalSync<HardEvent::V_S>();
    return red.GetValue(0);
}

template <class T, uint32_t N, bool StaticShape>
template <typename ExtractType59_0, typename ExtractType59_1, typename ExtractType59_2, typename ExtractType59_3>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::PowerRowBlock59(ExtractType59_0 &row,
                                                                            ExtractType59_1 &transpose,
                                                                            ExtractType59_2 &dst, ExtractType59_3 &base)
{
    if constexpr (sizeof(T) == 4) {
        if (!transpose)
            DataCopyPad(dst, input[base + row * t.p], {1, t.p * 4, 0, 0, 0}, {true, 0, (uint8_t)(t.pp - t.p), 0});
        else {
            DataCopyPad(rm, input[base + row], {(uint16_t)t.p, 4, (t.p - 1) * 4, 0, 0}, {false, 0, 0, 0});
            Fence();
            Gather(dst, rm, idx, 0, t.p);
        }
        Fence();
    } else {
        auto packed = rm.template ReinterpretCast<T>();
        auto cv = mat[26000];
        if (!transpose) {
            DataCopyPad(packed, input[base + row * t.p], {1, t.p * 2, 0, 0, 0}, {false, 0, 0, 0});
            Fence();
            Cast(dst, packed, RoundMode::CAST_NONE, t.p);
        } else {
            DataCopyPad(packed, input[base + row], {(uint16_t)t.p, 2, (t.p - 1) * 2, 0, 0}, {false, 0, 0, 0});
            Fence();
            Cast(cv, packed, RoundMode::CAST_NONE, t.p * 16);
            Fence();
            Gather(dst, cv, perm, 0, t.p);
        }
        Fence();
    }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::PowerRow(uint32_t batch, uint32_t row, bool transpose,
                                                                     bool inverse, LocalTensor<float> dst)
{
    Duplicate(dst, 0.0f, t.pp);
    Fence();
    if (inverse) {
        if (transpose) {
            DataCopy(dst, iall[batch * t.pp * t.pp + row * t.pp], t.pp);
            Fence();
        } else {
            DataCopyPad(rm, iall[batch * t.pp * t.pp + row], {(uint16_t)t.p, 4, (t.pp - 1) * 4, 0, 0},
                        {false, 0, 0, 0});
            Fence();
            Gather(dst, rm, idx, 0, t.p);
            Fence();
        }
    } else {
        uint32_t base = batch * t.p * t.p;
        PowerRowBlock59(row, transpose, dst, base);
    }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::Update(uint32_t c0, uint32_t c1, uint32_t row)
{
    if (c0 < c1) {
        Gather(tmp, mat[c0 * ld], ofs, row * 4, c1 - c0);
        PipeBarrier<PIPE_V>();
        if constexpr (N == 256) {
            auto opposite = mat[38912];
            Brcb(broad, tmp, (c1 - c0 + 7) / 8, {1, 8});
            Brcb(opposite, tmp, (c1 - c0 + 7) / 8, {1, 8});
            PipeBarrier<PIPE_V>();
            MulAddDst(mat[c0 * ld], piv, broad, 64, c1 - c0, {1, 1, 0, 33, 0, 1});
            MulAddDst(mat[c0 * ld + 64], piv[64], opposite, 64, c1 - c0, {1, 1, 0, 33, 0, 1});
            MulAddDst(mat[c0 * ld + 128], piv[128], broad, 64, c1 - c0, {1, 1, 0, 33, 0, 1});
            MulAddDst(mat[c0 * ld + 192], piv[192], opposite, 64, c1 - c0, {1, 1, 0, 33, 0, 1});
        } else {
            UpdateGeneric(c0, c1);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline float PinvInverseState<T, N, StaticShape>::PowerMultiply(LocalTensor<float> cache,
                                                                           LocalTensor<float> x, LocalTensor<float> y,
                                                                           LocalTensor<float> vals, uint32_t firstRow,
                                                                           uint32_t ac, uint32_t offset)
{
    for (uint32_t r = 0; r < ac; r++) {
        Mul(tmp, cache[r * t.pp], x, t.pp);
        PipeBarrier<PIPE_V>();
        Sum(red, tmp);
        LocalSync<HardEvent::V_S>();
        vals.SetValue(r, red.GetValue(0));
    }
    if (ac) {
        LocalSync<HardEvent::S_MTE3>();
        DataCopyPad(exchange[offset + firstRow], vals, {1, ac * 4, 0, 0, 0});
        LocalSync<HardEvent::MTE3_S>();
    }
    Sync();
    DataCopy(y, exchange[offset], t.pp);
    Fence();
    if (t.pp > t.p)
        Duplicate(y[t.p], 0.0f, t.pp - t.p);
    Fence();
    float estimate = Dot(y);
    float ny = NormRoot(estimate);
    Muls(y, y, ny > 0 ? 1.0f / ny : 0.0f, t.pp);
    Fence();
    return estimate;
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ __attribute__((noinline)) float PinvInverseState<T, N, StaticShape>::PowerNorm(uint32_t batch, bool inverse)
{
    if constexpr (StaticShape && N > 0) {
        t.p = N;
        t.pp = (N + 7) / 8 * 8;
    }
    auto x = mat[38000], y = mat[38512], vals = mat[39024];
    uint32_t rows = (t.p + t.group * 8 - 1) / (t.group * 8) * 8, firstRow = part * rows,
             ac = firstRow < t.p ? (t.p - firstRow < rows ? t.p - firstRow : rows) : 0;
    auto cache = mat, cacheT = mat[rows * t.pp];
    Duplicate(x, 0.0f, t.pp);
    Fence();
    for (uint32_t j = 0; j < t.p; j++) {
        x.SetValue(j, (float)((int32_t)((j * 17 + 13) % 31) - 15));
        idx.SetValue(j, j * 32);
        perm.SetValue(j, j * 64);
    }
    LocalSync<HardEvent::S_V>();
    for (uint32_t r = 0; r < ac; r++) {
        PowerRow(batch, firstRow + r, false, inverse, cache[r * t.pp]);
        PowerRow(batch, firstRow + r, true, inverse, cacheT[r * t.pp]);
    }
    float nx = NormRoot(Dot(x));
    Muls(x, x, nx > 0 ? 1.0f / nx : 0.0f, t.pp);
    Fence();
    float estimate = 0;
    for (uint32_t it = 0; it < 4; it++) {
        estimate = PowerMultiply(cache, x, y, vals, firstRow, ac, 0);
        PowerMultiply(cacheT, y, x, vals, firstRow, ac, t.pp);
    }
    return estimate;
}

template <class T, uint32_t N, bool StaticShape>
template <typename ExtractType80_0, typename ExtractType80_1, typename ExtractType80_2, typename ExtractType80_3,
          typename ExtractType80_4, typename ExtractType80_5, typename ExtractType80_6, typename ExtractType80_7,
          typename ExtractType80_8, typename ExtractType80_9>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::SharedInverseBlock1(
    ExtractType80_0 &chunk, ExtractType80_1 &savedA, ExtractType80_2 &acc, ExtractType80_3 &acc1, ExtractType80_4 &acc2,
    ExtractType80_5 &acc3, ExtractType80_6 &coeff, ExtractType80_7 &cache, ExtractType80_8 &br, ExtractType80_9 &ac)
{
    for (uint32_t cc = 0; cc < t.p; cc += chunk) {
        uint32_t kc = t.p - cc < chunk ? t.p - cc : chunk;
        DataCopy(cache, savedA[cc * t.pp], kc * t.pp);
        Brcb(br, coeff[cc * 8], kc, {1, 8});
        LocalSync<HardEvent::MTE2_V>();
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < kc; j += 4) {
            for (uint32_t r = 0; r < t.pp; r += 64) {
                uint32_t mask = t.pp - r < 64 ? t.pp - r : 64;
                MulAddDst(acc[r], cache[j * t.pp + r], br[j * 64], mask, ac, {1, 1, 0, (uint8_t)(t.pp / 8), 0, 1});
            }
            if (j + 1 < kc)
                for (uint32_t r = 0; r < t.pp; r += 64) {
                    uint32_t mask = t.pp - r < 64 ? t.pp - r : 64;
                    MulAddDst(acc1[r], cache[(j + 1) * t.pp + r], br[(j + 1) * 64], mask, ac,
                              {1, 1, 0, (uint8_t)(t.pp / 8), 0, 1});
                }
            if (j + 2 < kc)
                for (uint32_t r = 0; r < t.pp; r += 64) {
                    uint32_t mask = t.pp - r < 64 ? t.pp - r : 64;
                    MulAddDst(acc2[r], cache[(j + 2) * t.pp + r], br[(j + 2) * 64], mask, ac,
                              {1, 1, 0, (uint8_t)(t.pp / 8), 0, 1});
                }
            if (j + 3 < kc)
                for (uint32_t r = 0; r < t.pp; r += 64) {
                    uint32_t mask = t.pp - r < 64 ? t.pp - r : 64;
                    MulAddDst(acc3[r], cache[(j + 3) * t.pp + r], br[(j + 3) * 64], mask, ac,
                              {1, 1, 0, (uint8_t)(t.pp / 8), 0, 1});
                }
            PipeBarrier<PIPE_V>();
        }
        LocalSync<HardEvent::V_MTE2>();
    }
}

template <class T, uint32_t N, bool StaticShape>
template <typename ExtractType84_0, typename ExtractType84_1, typename ExtractType84_2, typename ExtractType84_3,
          typename ExtractType84_4, typename ExtractType84_5, typename ExtractType84_6, typename ExtractType84_7>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::SharedInverseBlock2(
    ExtractType84_0 &acc, ExtractType84_1 &cache, ExtractType84_2 &br, ExtractType84_3 &ac, ExtractType84_4 &acc1,
    ExtractType84_5 &acc2, ExtractType84_6 &acc3, ExtractType84_7 &kc)
{
    for (uint32_t k = 0; k < kc; k += 4) {
        for (uint32_t r = 0; r < t.pp; r += 64) {
            uint32_t mask = t.pp - r < 64 ? t.pp - r : 64;
            MulAddDst(acc[r], cache[k * t.pp + r], br[k * 64], mask, ac, {1, 1, 0, (uint8_t)(t.pp / 8), 0, 1});
        }
        if (k + 1 < kc)
            for (uint32_t r = 0; r < t.pp; r += 64) {
                uint32_t mask = t.pp - r < 64 ? t.pp - r : 64;
                MulAddDst(acc1[r], cache[(k + 1) * t.pp + r], br[(k + 1) * 64], mask, ac,
                          {1, 1, 0, (uint8_t)(t.pp / 8), 0, 1});
            }
        if (k + 2 < kc)
            for (uint32_t r = 0; r < t.pp; r += 64) {
                uint32_t mask = t.pp - r < 64 ? t.pp - r : 64;
                MulAddDst(acc2[r], cache[(k + 2) * t.pp + r], br[(k + 2) * 64], mask, ac,
                          {1, 1, 0, (uint8_t)(t.pp / 8), 0, 1});
            }
        if (k + 3 < kc)
            for (uint32_t r = 0; r < t.pp; r += 64) {
                uint32_t mask = t.pp - r < 64 ? t.pp - r : 64;
                MulAddDst(acc3[r], cache[(k + 3) * t.pp + r], br[(k + 3) * 64], mask, ac,
                          {1, 1, 0, (uint8_t)(t.pp / 8), 0, 1});
            }
        PipeBarrier<PIPE_V>();
    }
}

template <class T, uint32_t N, bool StaticShape>
template <typename ExtractType75_0, typename ExtractType75_1, typename ExtractType75_2, typename ExtractType75_3,
          typename ExtractType75_4, typename ExtractType75_5, typename ExtractType75_6, typename ExtractType75_7>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::SharedInverseBlock3(
    ExtractType75_0 &acc, ExtractType75_1 &cache, ExtractType75_2 &br, ExtractType75_3 &ac, ExtractType75_4 &acc1,
    ExtractType75_5 &acc2, ExtractType75_6 &acc3, ExtractType75_7 &kc)
{
    if (t.pp <= 256) {
        SharedInverseBlock2(acc, cache, br, ac, acc1, acc2, acc3, kc);
    } else {
        for (uint32_t k = 0; k < kc; k += 2) {
            for (uint32_t r = 0; r < t.pp; r += 64) {
                uint32_t mask = t.pp - r < 64 ? t.pp - r : 64;
                MulAddDst(acc[r], cache[k * t.pp + r], br[k * 64], mask, ac, {1, 1, 0, (uint8_t)(t.pp / 8), 0, 1});
            }
            if (k + 1 < kc)
                for (uint32_t r = 0; r < t.pp; r += 64) {
                    uint32_t mask = t.pp - r < 64 ? t.pp - r : 64;
                    MulAddDst(acc2[r], cache[(k + 1) * t.pp + r], br[(k + 1) * 64], mask, ac,
                              {1, 1, 0, (uint8_t)(t.pp / 8), 0, 1});
                }
            PipeBarrier<PIPE_V>();
        }
    }
}

template <class T, uint32_t N, bool StaticShape>
template <typename ExtractType71_0, typename ExtractType71_1, typename ExtractType71_2, typename ExtractType71_3,
          typename ExtractType71_4, typename ExtractType71_5, typename ExtractType71_6, typename ExtractType71_7,
          typename ExtractType71_8, typename ExtractType71_9>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::SharedInverseBlock4(
    ExtractType71_0 &inv, ExtractType71_1 &chunk, ExtractType71_2 &acc, ExtractType71_3 &cf, ExtractType71_4 &cache,
    ExtractType71_5 &br, ExtractType71_6 &ac, ExtractType71_7 &acc1, ExtractType71_8 &acc2, ExtractType71_9 &acc3)
{
    for (uint32_t cc = 0; cc < t.p; cc += chunk) {
        uint32_t kc = t.p - cc < chunk ? t.p - cc : chunk;
        DataCopy(cache, inv[cc * t.pp], kc * t.pp);
        Brcb(br, cf[cc * 8], kc, {1, 8});
        LocalSync<HardEvent::MTE2_V>();
        PipeBarrier<PIPE_V>();
        SharedInverseBlock3(acc, cache, br, ac, acc1, acc2, acc3, kc);
        LocalSync<HardEvent::V_MTE2>();
    }
}

template <class T, uint32_t N, bool StaticShape>
template <typename ExtractType68_0, typename ExtractType68_1, typename ExtractType68_2, typename ExtractType68_3,
          typename ExtractType68_4, typename ExtractType68_5, typename ExtractType68_6, typename ExtractType68_7,
          typename ExtractType68_8, typename ExtractType68_9>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::SharedInverseBlock5(
    ExtractType68_0 &batch, ExtractType68_1 &residual, ExtractType68_2 &corrected, ExtractType68_3 &inv,
    ExtractType68_4 &chunk, ExtractType68_5 &acc, ExtractType68_6 &cf, ExtractType68_7 &cache, ExtractType68_8 &br,
    ExtractType68_9 &outputIndices)
{
    for (uint32_t c0 = part * 8; c0 < t.p; c0 += t.group * 8) {
        uint32_t ac = t.p - c0 < 8 ? t.p - c0 : 8;
        DataCopy(acc, inv[c0 * t.pp], ac * t.pp);
        DataCopy(cf, residual[c0 * t.pp], 8 * t.pp);
        auto acc1 = mat[2048], acc2 = mat[4096], acc3 = mat[6144];
        if (t.pp <= 256)
            Duplicate(acc1, 0.0f, 6144);
        else
            Duplicate(acc2, 0.0f, ac * t.pp);
        Fence();
        if (t.pp > t.p) {
            Duplicate(cf[t.p * 8], 0.0f, (t.pp - t.p) * 8);
            PipeBarrier<PIPE_V>();
        }
        SharedInverseBlock4(inv, chunk, acc, cf, cache, br, ac, acc1, acc2, acc3);
        if (t.pp <= 256) {
            Add(acc, acc, acc1, ac * t.pp);
            Add(acc2, acc2, acc3, ac * t.pp);
            PipeBarrier<PIPE_V>();
        }
        Add(acc, acc, acc2, ac * t.pp);
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(T) == 4) {
            Gather(rm, acc, outputIndices, 0, t.pp * 8);
            LocalSync<HardEvent::V_MTE3>();
            DataCopyPad(output[batch * t.p * t.p + c0], rm, {(uint16_t)t.p, ac * 4, 0, (t.p - ac) * 4, 0});
            Fence();
        } else {
            DataCopy(corrected[c0 * t.pp], acc, ac * t.pp);
            Fence();
        }
    }
}

template <class T, uint32_t N, bool StaticShape>
template <typename ExtractType92_0, typename ExtractType92_1>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::SharedInverseBlock6(ExtractType92_0 &br,
                                                                                ExtractType92_1 &outputIndices)
{
    if constexpr (sizeof(T) == 4 && N == 0) {
        auto columnIndices = br.template ReinterpretCast<uint32_t>();
        BuildOutputIndices(outputIndices);
    }
}
