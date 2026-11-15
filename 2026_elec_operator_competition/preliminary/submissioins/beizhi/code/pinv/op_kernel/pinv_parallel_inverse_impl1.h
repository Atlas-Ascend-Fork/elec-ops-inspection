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

template <class T, uint32_t N>
template <typename ExtractType78_0, typename ExtractType78_1, typename ExtractType78_2, typename ExtractType78_3,
          typename ExtractType78_4, typename ExtractType78_5, typename ExtractType78_6>
__aicore__ inline void ParallelInverse<T, N>::CubeRefineBlock78(ExtractType78_0 &batch, ExtractType78_1 &residual,
                                                                ExtractType78_2 &savedA, ExtractType78_3 &corr,
                                                                ExtractType78_4 &inv, ExtractType78_5 &row0,
                                                                ExtractType78_6 &rows)
{
    constexpr uint32_t P = 256;
    if (rows > 0) {
        cube->SetTensorA(inv[row0 * P]);
        cube->SetTensorB(savedA);
        cube->SetTail(rows, P, P);
        cube->IterateAll(residual[row0 * P]);
        cube->End();
        DataCopy(this->mat, residual[row0 * P], rows * P);
        LocalSync<HardEvent::MTE2_V>();
        Muls(this->mat, this->mat, -1.0f, rows * P);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < rows; r++) {
            uint32_t c = row0 + r;
            uint64_t mask[2] = {1ULL << (c % 8), 0};
            Adds(this->mat[r * P + c / 8 * 8], this->mat[r * P + c / 8 * 8], 1.0f, mask, 1, {1, 1, 8, 8});
        }
        LocalSync<HardEvent::V_MTE3>();
        DataCopy(residual[row0 * P], this->mat, rows * P);
        this->Fence();
        DataCopy(this->mat[16384], inv[row0 * P], rows * P);
        cube->SetTensorA(residual[row0 * P]);
        cube->SetTensorB(inv);
        cube->SetTail(rows, P, P);
        cube->IterateAll(corr[row0 * P]);
        cube->End();
        DataCopy(this->mat, corr[row0 * P], rows * P);
        LocalSync<HardEvent::MTE2_V>();
        Add(this->mat, this->mat, this->mat[16384], rows * P);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < rows; r += 16) {
            FastTranspose16x256(this->rm, this->mat[r * P]);
            LocalSync<HardEvent::V_MTE3>();
            DataCopyPad(this->output[batch * P * P + row0 + r], this->rm, {P, 64, 0, (P - 16) * 4, 0});
            if (r + 16 < rows)
                LocalSync<HardEvent::MTE3_V>();
            else
                this->Fence();
        }
    }
}

template <class T, uint32_t N>
template <typename ExtractType85_0, typename ExtractType85_1, typename ExtractType85_2>
__aicore__ inline void ParallelInverse<T, N>::CubeRefineBlock83Block85(ExtractType85_0 &aic, ExtractType85_1 &row0,
                                                                       ExtractType85_2 &rows)
{
    if (this->t.group >= 12) {
        if (aic < 4) {
            row0 = aic * 48;
            rows = 48;
        } else if (aic < 6) {
            row0 = 192 + (aic - 4) * 32;
            rows = 32;
        }
    } else if (this->t.group >= 10) {
        if (aic < 3) {
            row0 = aic * 64;
            rows = 64;
        } else if (aic < 5) {
            row0 = 192 + (aic - 3) * 32;
            rows = 32;
        }
    } else {
        if (aic < 4) {
            row0 = aic * 64;
            rows = 64;
        }
    }
}

template <class T, uint32_t N>
template <typename ExtractType83_0, typename ExtractType83_1, typename ExtractType83_2>
__aicore__ inline void ParallelInverse<T, N>::CubeRefineBlock83(ExtractType83_0 &aic, ExtractType83_1 &row0,
                                                                ExtractType83_2 &rows)
{
    if ((this->part & 1) == 0) {
        if (this->t.group >= 16) {
            if (aic < 8) {
                row0 = aic * 32;
                rows = 32;
            }
        } else
            CubeRefineBlock83Block85(aic, row0, rows);
    }
}

template <class T, uint32_t N> __aicore__ inline void ParallelInverse<T, N>::CubeRefine(uint32_t batch)
{
    constexpr uint32_t P = 256;
    auto residual = this->state[(this->t.group * this->t.groups + 1) * 8 + batch * P * P];
    auto savedA = this->state[(this->t.group * this->t.groups + 1) * 8 + this->t.batch * P * P + batch * P * P];
    auto corr = this->state[(this->t.group * this->t.groups + 1) * 8 + 2 * this->t.batch * P * P + batch * P * P];
    auto inv = this->iall[batch * P * P];
    uint32_t aic = this->part >> 1, row0 = P, rows = 0;
    CubeRefineBlock83(aic, row0, rows);
    CubeRefineBlock78(batch, residual, savedA, corr, inv, row0, rows);
    if (batch + this->t.groups < this->t.batch)
        this->Sync();
    this->outputReady = true;
}

template <class T, uint32_t N> __aicore__ inline void ParallelInverse<T, N>::CubeCorrection(uint32_t batch)
{
    constexpr uint32_t P = 256;
    auto residual = this->state[(this->t.group * this->t.groups + 1) * 8 + batch * P * P],
         corrected = this->state[(this->t.group * this->t.groups + 1) * 8 + this->t.batch * P * P + batch * P * P],
         inv = this->iall[batch * P * P];
    uint32_t row0 = this->part * 32;
    if (row0 < P) {
        cube->SetTensorA(residual[row0 * P]);
        cube->SetTensorB(inv);
        cube->SetTail(32, P, P);
        cube->IterateAll(corrected[row0 * P]);
        cube->End();
    }
    this->Sync();
    for (uint32_t c = this->part * 16; c < P; c += this->t.group * 16) {
        DataCopy(this->mat, corrected[c * P], 16 * P);
        DataCopy(this->mat[8192], inv[c * P], 16 * P);
        LocalSync<HardEvent::MTE2_V>();
        Add(this->mat, this->mat, this->mat[8192], 16 * P);
        PipeBarrier<PIPE_V>();
        FastTranspose16x256(this->rm, this->mat);
        LocalSync<HardEvent::V_MTE3>();
        DataCopyPad(this->output[batch * P * P + c], this->rm, {P, 64, 0, (P - 16) * 4, 0});
        this->Fence();
    }
    this->Sync();
    this->outputReady = true;
}

template <class T, uint32_t N>
template <typename ExtractType81_0, typename ExtractType81_1, typename ExtractType81_2, typename ExtractType81_3,
          typename ExtractType81_4, typename ExtractType81_5, typename ExtractType81_6, typename ExtractType81_7,
          typename ExtractType81_8, typename ExtractType81_9, typename ExtractType81_10, typename ExtractType81_11,
          typename ExtractType81_12>
__aicore__ inline void ParallelInverse<T, N>::RefineInverseBlock66Block67Block72Block81(
    ExtractType81_0 &batch, ExtractType81_1 &residual, ExtractType81_2 &wp, ExtractType81_3 &row, ExtractType81_4 &out,
    ExtractType81_5 &products, ExtractType81_6 &sums, ExtractType81_7 &segments, ExtractType81_8 &batchDots,
    ExtractType81_9 &aCache, ExtractType81_10 &cacheA, ExtractType81_11 &cc, ExtractType81_12 &kc)
{
    for (uint32_t r = this->part; r < this->t.p; r += this->t.group) {
        this->LoadRefinementRow(batch, r, wp, row, aCache, cacheA);
        for (uint32_t c = 0; c < kc; c += batchDots) {
            uint32_t ac = kc - c < batchDots ? kc - c : batchDots;
            for (uint32_t q = 0; q < wp; q += 64)
                Mul(products[c * wp + q], this->mat[c * wp + q], row[q], 64, ac,
                    {1, 1, 1, (uint8_t)(wp / 8), (uint8_t)(wp / 8), 0});
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t c = 0; c < kc; c += batchDots) {
            uint32_t ac = kc - c < batchDots ? kc - c : batchDots;
            WholeReduceSum(sums[c * 8], products[c * wp], 64, ac * segments, 8 / segments, 1, 8);
        }
        PipeBarrier<PIPE_V>();
        uint32_t blocks = (kc + 7) / 8;
        BlockReduceSum(out, sums, blocks, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Muls(out, out, -1.0f, blocks * 8);
        if (r >= cc && r < cc + kc) {
            LocalSync<HardEvent::V_S>();
            out.SetValue(r - cc, out.GetValue(r - cc) + 1.0f);
            LocalSync<HardEvent::S_MTE3>();
        } else
            LocalSync<HardEvent::V_MTE3>();
        DataCopyPad(residual[cc * this->t.pp + r * 8], out, {(uint16_t)blocks, 32, 0, (this->t.pp - 1) * 32, 0});
        LocalSync<HardEvent::MTE3_V>();
    }
}

template <class T, uint32_t N>
template <typename ExtractType72_0, typename ExtractType72_1, typename ExtractType72_2, typename ExtractType72_3,
          typename ExtractType72_4, typename ExtractType72_5, typename ExtractType72_6, typename ExtractType72_7,
          typename ExtractType72_8, typename ExtractType72_9, typename ExtractType72_10, typename ExtractType72_11,
          typename ExtractType72_12>
__aicore__ inline void ParallelInverse<T, N>::RefineInverseBlock66Block67Block72(
    ExtractType72_0 &batch, ExtractType72_1 &residual, ExtractType72_2 &inv, ExtractType72_3 &wp, ExtractType72_4 &row,
    ExtractType72_5 &out, ExtractType72_6 &chunk, ExtractType72_7 &products, ExtractType72_8 &sums,
    ExtractType72_9 &segments, ExtractType72_10 &batchDots, ExtractType72_11 &aCache, ExtractType72_12 &cacheA)
{
    for (uint32_t cc = 0; cc < this->t.p; cc += chunk) {
        uint32_t kc = this->t.p - cc < chunk ? this->t.p - cc : chunk;
        if (kc < chunk)
            Duplicate(sums, 0.0f, 512);
        this->LoadRefinementColumns(wp, inv, cc, kc);
        RefineInverseBlock66Block67Block72Block81(batch, residual, wp, row, out, products, sums, segments, batchDots,
                                                  aCache, cacheA, cc, kc);
    }
}

template <class T, uint32_t N>
template <typename ExtractType82_0, typename ExtractType82_1, typename ExtractType82_2, typename ExtractType82_3,
          typename ExtractType82_4, typename ExtractType82_5, typename ExtractType82_6, typename ExtractType82_7,
          typename ExtractType82_8, typename ExtractType82_9, typename ExtractType82_10, typename ExtractType82_11,
          typename ExtractType82_12, typename ExtractType82_13>
__aicore__ inline void ParallelInverse<T, N>::RefineInverseBlock66Block67Block73Block82(
    ExtractType82_0 &batch, ExtractType82_1 &residual, ExtractType82_2 &wp, ExtractType82_3 &row, ExtractType82_4 &out,
    ExtractType82_5 &products, ExtractType82_6 &sums, ExtractType82_7 &values, ExtractType82_8 &segments,
    ExtractType82_9 &batchDots, ExtractType82_10 &aCache, ExtractType82_11 &cacheA, ExtractType82_12 &cc,
    ExtractType82_13 &kc)
{
    for (uint32_t r = this->part; r < this->t.p; r += this->t.group) {
        this->LoadRefinementRow(batch, r, wp, row, aCache, cacheA);
        for (uint32_t c = 0; c < kc; c += batchDots) {
            uint32_t ac = kc - c < batchDots ? kc - c : batchDots;
            for (uint32_t q = 0; q < wp; q += 64)
                Mul(products[q], this->mat[c * wp + q], row[q], 64, ac,
                    {1, 1, 1, (uint8_t)(wp / 8), (uint8_t)(wp / 8), 0});
            PipeBarrier<PIPE_V>();
            WholeReduceSum(sums, products, 64, ac * segments, 8 / segments, 1, 8);
            PipeBarrier<PIPE_V>();
            BlockReduceSum(values, sums, batchDots / 8, 64, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            Muls(out[c], values, -1.0f, ac);
            PipeBarrier<PIPE_V>();
        }
        if (r >= cc && r < cc + kc) {
            LocalSync<HardEvent::V_S>();
            out.SetValue(r - cc, out.GetValue(r - cc) + 1.0f);
            LocalSync<HardEvent::S_MTE3>();
        } else
            LocalSync<HardEvent::V_MTE3>();
        DataCopyPad(residual[cc * this->t.pp + r * 8], out,
                    {(uint16_t)((kc + 7) / 8), 32, 0, (this->t.pp - 1) * 32, 0});
        LocalSync<HardEvent::MTE3_V>();
    }
}

template <class T, uint32_t N>
template <typename ExtractType73_0, typename ExtractType73_1, typename ExtractType73_2, typename ExtractType73_3,
          typename ExtractType73_4, typename ExtractType73_5, typename ExtractType73_6, typename ExtractType73_7,
          typename ExtractType73_8, typename ExtractType73_9, typename ExtractType73_10, typename ExtractType73_11,
          typename ExtractType73_12, typename ExtractType73_13>
__aicore__ inline void ParallelInverse<T, N>::RefineInverseBlock66Block67Block73(
    ExtractType73_0 &batch, ExtractType73_1 &residual, ExtractType73_2 &inv, ExtractType73_3 &wp, ExtractType73_4 &row,
    ExtractType73_5 &out, ExtractType73_6 &chunk, ExtractType73_7 &products, ExtractType73_8 &sums,
    ExtractType73_9 &values, ExtractType73_10 &segments, ExtractType73_11 &batchDots, ExtractType73_12 &aCache,
    ExtractType73_13 &cacheA)
{
    for (uint32_t cc = 0; cc < this->t.p; cc += chunk) {
        uint32_t kc = this->t.p - cc < chunk ? this->t.p - cc : chunk;
        if (kc < chunk) {
            Duplicate(sums, 0.0f, batchDots * 8);
            Duplicate(out, 0.0f, chunk);
        }
        this->LoadRefinementColumns(wp, inv, cc, kc);
        RefineInverseBlock66Block67Block73Block82(batch, residual, wp, row, out, products, sums, values, segments,
                                                  batchDots, aCache, cacheA, cc, kc);
    }
}

template <class T, uint32_t N>
template <typename ExtractType67_0, typename ExtractType67_1, typename ExtractType67_2, typename ExtractType67_3,
          typename ExtractType67_4, typename ExtractType67_5, typename ExtractType67_6>
__aicore__ inline void
ParallelInverse<T, N>::RefineInverseBlock66Block67(ExtractType67_0 &batch, ExtractType67_1 &residual,
                                                   ExtractType67_2 &inv, ExtractType67_3 &wp, ExtractType67_4 &row,
                                                   ExtractType67_5 &out, ExtractType67_6 &chunk)
{
    if constexpr (N == 199) {
        auto row = this->mat[38000], out = this->mat[38512], products = this->mat[24576], sums = this->mat[32768],
             values = this->mat[33152];
        uint32_t chunk = 16384 / wp;
        if (chunk > 64)
            chunk = 64;
        uint32_t segments = wp / 64, batchDots = wp <= 256 ? 32 : 16;
        Duplicate(sums, 0.0f, batchDots * 8);
        Duplicate(out, 0.0f, chunk);
        auto aCache = this->mat[16384];
        uint32_t owned = (this->t.p + this->t.group - 1 - this->part) / this->t.group;
        bool cacheA = owned * wp <= 8192;
        this->SharedPhase1(batch, wp, aCache, owned, cacheA);
        RefineInverseBlock66Block67Block73(batch, residual, inv, wp, row, out, chunk, products, sums, values, segments,
                                           batchDots, aCache, cacheA);
        this->Sync();
    } else {
        auto row = this->mat[41472], out = this->mat[44096], products = this->mat[24576], sums = this->mat[40960];
        uint32_t chunk = 16384 / wp;
        if (chunk > 64)
            chunk = 64;
        uint32_t segments = wp / 64, batchDots = wp <= 256 ? 32 : 16;
        Duplicate(sums, 0.0f, 512);
        auto aCache = this->mat[16384];
        uint32_t owned = (this->t.p + this->t.group - 1 - this->part) / this->t.group;
        bool cacheA = owned * wp <= 8192;
        this->SharedPhase1(batch, wp, aCache, owned, cacheA);
        RefineInverseBlock66Block67Block72(batch, residual, inv, wp, row, out, chunk, products, sums, segments,
                                           batchDots, aCache, cacheA);
        this->Sync();
    }
}

template <class T, uint32_t N>
template <typename ExtractType69_0, typename ExtractType69_1, typename ExtractType69_2, typename ExtractType69_3,
          typename ExtractType69_4, typename ExtractType69_5, typename ExtractType69_6, typename ExtractType69_7,
          typename ExtractType69_8, typename ExtractType69_9, typename ExtractType69_10, typename ExtractType69_11>
__aicore__ inline void ParallelInverse<T, N>::RefineInverseBlock66Block69(
    ExtractType69_0 &residual, ExtractType69_1 &inv, ExtractType69_2 &chunk, ExtractType69_3 &savedA,
    ExtractType69_4 &acc, ExtractType69_5 &acc1, ExtractType69_6 &acc2, ExtractType69_7 &acc3, ExtractType69_8 &coeff,
    ExtractType69_9 &cache, ExtractType69_10 &br, ExtractType69_11 &outputIndices)
{
    for (uint32_t c0 = this->part * 8; c0 < this->t.p; c0 += this->t.group * 8) {
        uint32_t ac = this->t.p - c0 < 8 ? this->t.p - c0 : 8;
        this->BuildResidualBlock(inv, c0, chunk, savedA, acc, acc1, acc2, acc3, coeff, cache, br, outputIndices);
        if constexpr (N == 256) {
            if (cube) {
                LocalSync<HardEvent::S_MTE3>();
                DataCopy(residual[c0 * this->t.pp], acc, this->t.pp * 8);
                this->Fence();
            } else {
                LocalSync<HardEvent::S_V>();
                Gather(this->rm, acc, outputIndices, 0, this->t.pp * 8);
                LocalSync<HardEvent::V_MTE3>();
                DataCopy(residual[c0 * this->t.pp], this->rm, this->t.pp * 8);
                this->Fence();
            }
        } else {
            LocalSync<HardEvent::S_V>();
            Gather(this->rm, acc, outputIndices, 0, this->t.pp * 8);
            LocalSync<HardEvent::V_MTE3>();
            DataCopy(residual[c0 * this->t.pp], this->rm, this->t.pp * 8);
            this->Fence();
        }
    }
}

template <class T, uint32_t N>
template <typename ExtractType66_0, typename ExtractType66_1, typename ExtractType66_2, typename ExtractType66_3,
          typename ExtractType66_4, typename ExtractType66_5, typename ExtractType66_6, typename ExtractType66_7>
__aicore__ inline bool ParallelInverse<T, N>::RefineInverseBlock66(ExtractType66_0 &batch, ExtractType66_1 &residual,
                                                                   ExtractType66_2 &corrected, ExtractType66_3 &inv,
                                                                   ExtractType66_4 &wp, ExtractType66_5 &row,
                                                                   ExtractType66_6 &out, ExtractType66_7 &chunk)
{
    if constexpr (sizeof(T) == 4 && N > 0) {
        auto savedA = corrected;
        auto acc = this->mat, acc1 = this->mat[2048], acc2 = this->mat[4096], acc3 = this->mat[6144],
             coeff = this->mat[8192], cache = this->mat[12288], br = this->mat[30000];
        auto outputIndices = this->mat[40000].template ReinterpretCast<uint32_t>();
        auto columnIndices = br.template ReinterpretCast<uint32_t>();
        this->BuildOutputIndices(outputIndices);
        RefineInverseBlock66Block69(residual, inv, chunk, savedA, acc, acc1, acc2, acc3, coeff, cache, br,
                                    outputIndices);
        this->Sync();
        if constexpr (N == 256) {
            if (cube) {
                CubeCorrection(batch);
                return true;
            }
        }
    } else {
        RefineInverseBlock66Block67(batch, residual, inv, wp, row, out, chunk);
    }
    return false;
}

template <class T, uint32_t N> __aicore__ inline void ParallelInverse<T, N>::RefineInverseBlock95()
{
    if constexpr (!(sizeof(T) == 4 && N > 0)) {
        for (uint32_t j = 0; j < this->t.p; j++) {
            this->idx.SetValue(j, j * 32);
            this->perm.SetValue(j, j * 64);
        }
        LocalSync<HardEvent::S_V>();
    }
}

template <class T, uint32_t N> __aicore__ inline void ParallelInverse<T, N>::RefineInverse(uint32_t batch)
{
    if constexpr (N == 256 && sizeof(T) == 4) {
        if (cube && cubePrecision) {
            CubeRefine(batch);
            return;
        }
    }
    auto residual = this->state[(this->t.group * this->t.groups + 1) * 8 + batch * this->t.pp * this->t.pp];
    auto corrected = this->state[(this->t.group * this->t.groups + 1) * 8 + this->t.batch * this->t.pp * this->t.pp +
                                 batch * this->t.pp * this->t.pp];
    auto inv = this->iall[batch * this->t.pp * this->t.pp];
    RefineInverseBlock95();
    uint32_t wp = 64;
    while (wp < this->t.pp)
        wp *= 2;
    auto row = this->mat[38000], out = this->mat[38512];
    uint32_t chunk = 16384 / wp;
    if (chunk > 64)
        chunk = 64;
    if (RefineInverseBlock66(batch, residual, corrected, inv, wp, row, out, chunk))
        return;
    auto acc = this->mat, cf = this->mat[8192], cache = this->mat[12288], br = this->mat[30000],
         vals = this->mat[34560];
    auto ix = this->mat.template ReinterpretCast<uint32_t>()[35584];
    auto outputIndices = this->mat.template ReinterpretCast<uint32_t>()[40000];
    this->SharedInverseBlock6(br, outputIndices);
    this->SharedInverseBlock5(batch, residual, corrected, inv, chunk, acc, cf, cache, br, outputIndices);
    this->Sync();
    this->FinishRefinement(corrected, inv, out);
}

template <class T, uint32_t N> __aicore__ inline TPipe *ParallelInverse<T, N>::Pipe()
{
    return &pipe;
}

template <class T, uint32_t N> __aicore__ inline void ParallelInverse<T, N>::SetCube(FP32Cube *mm)
{
    cube = mm;
}
