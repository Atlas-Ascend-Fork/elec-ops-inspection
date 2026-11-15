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

template <class T, uint32_t N, bool FIXP>
__aicore__ inline void PinvInverse199<T, N, FIXP>::Update(uint32_t c0, uint32_t c1, uint32_t row)
{
    if (c0 < c1) {
        Gather(this->tmp, this->mat[c0 * this->ld], this->ofs, row * 4, c1 - c0);
        PipeBarrier<PIPE_V>();
        if (c0 == 0 && c1 <= 56 && this->panelCols <= 56) {
            auto low = this->mat[13376], high = this->mat[33408];
            Brcb(low, this->tmp, (c1 + 7) / 8, {10, 80});
            Brcb(high, this->tmp, (c1 + 7) / 8, {10, 80});
            PipeBarrier<PIPE_V>();
            MulAddDst(this->mat, this->piv, low, 64, c1, {1, 1, 0, 26, 0, 10});
            MulAddDst(this->mat[64], this->piv[64], high, 64, c1, {1, 1, 0, 26, 0, 10});
            MulAddDst(this->mat[128], this->piv[128], low, 64, c1, {1, 1, 0, 26, 0, 10});
            MulAddDst(this->mat[192], this->piv[192], high, 8, c1, {1, 1, 0, 26, 0, 10});
        } else {
            this->UpdateGeneric(c0, c1);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType33_0, typename ExtractType33_1, typename ExtractType33_2, typename ExtractType33_3,
          typename ExtractType33_4, typename ExtractType33_5, typename ExtractType33_6, typename ExtractType33_7,
          typename ExtractType33_8, typename ExtractType33_9, typename ExtractType33_10, typename ExtractType33_11>
__aicore__ inline void PinvInverse199<T, N, FIXP>::RefineInverseBlock31Block33(
    ExtractType33_0 &residual, ExtractType33_1 &inv, ExtractType33_2 &chunk, ExtractType33_3 &savedA,
    ExtractType33_4 &acc, ExtractType33_5 &acc1, ExtractType33_6 &acc2, ExtractType33_7 &acc3, ExtractType33_8 &coeff,
    ExtractType33_9 &cache, ExtractType33_10 &br, ExtractType33_11 &outputIndices)
{
    for (uint32_t c0 = this->part * 8; c0 < this->t.p; c0 += this->t.group * 8) {
        uint32_t ac = this->t.p - c0 < 8 ? this->t.p - c0 : 8;
        this->BuildResidualBlock(inv, c0, chunk, savedA, acc, acc1, acc2, acc3, coeff, cache, br, outputIndices);
        LocalSync<HardEvent::S_V>();
        Gather(this->rm, acc, outputIndices, 0, this->t.pp * 8);
        LocalSync<HardEvent::V_MTE3>();
        DataCopy(residual[c0 * this->t.pp], this->rm, this->t.pp * 8);
        this->Fence();
    }
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType31_0, typename ExtractType31_1, typename ExtractType31_2, typename ExtractType31_3>
__aicore__ inline void PinvInverse199<T, N, FIXP>::RefineInverseBlock31(ExtractType31_0 &residual,
                                                                        ExtractType31_1 &corrected,
                                                                        ExtractType31_2 &inv, ExtractType31_3 &chunk)
{
    if constexpr (sizeof(T) == 4 && N > 0) {
        auto savedA = corrected;
        auto acc = this->mat, acc1 = this->mat[2048], acc2 = this->mat[4096], acc3 = this->mat[6144],
             coeff = this->mat[8192], cache = this->mat[N == 256 ? 12352 : 12288], br = this->mat[30000];
        auto outputIndices = this->mat[40000].template ReinterpretCast<uint32_t>();
        auto columnIndices = br.template ReinterpretCast<uint32_t>();
        this->BuildOutputIndices(outputIndices);
        RefineInverseBlock31Block33(residual, inv, chunk, savedA, acc, acc1, acc2, acc3, coeff, cache, br,
                                    outputIndices);
        this->Sync();
    }
}

template <class T, uint32_t N, bool FIXP>
__aicore__ inline void PinvInverse199<T, N, FIXP>::RefineInverse(uint32_t batch)
{
    auto residual = this->state[(this->t.group * this->t.groups + 1) * 8 + batch * this->t.pp * this->t.pp];
    auto corrected = this->state[(this->t.group * this->t.groups + 1) * 8 + this->t.batch * this->t.pp * this->t.pp +
                                 batch * this->t.pp * this->t.pp];
    auto inv = this->iall[batch * this->t.pp * this->t.pp];
    if constexpr (!(sizeof(T) == 4 && N > 0)) {
        for (uint32_t j = 0; j < this->t.p; j++) {
            this->idx.SetValue(j, j * 32);
            this->perm.SetValue(j, j * 64);
        }
        LocalSync<HardEvent::S_V>();
    }
    uint32_t wp = 64;
    while (wp < this->t.pp)
        wp *= 2;
    auto row = this->mat[38000], out = this->mat[38512];
    uint32_t chunk = 16384 / wp;
    if (chunk > 64)
        chunk = 64;
    RefineInverseBlock31(residual, corrected, inv, chunk);
    auto acc = this->mat, cf = this->mat[8192], cache = this->mat[N == 256 ? 12352 : 12288], br = this->mat[30000],
         vals = this->mat[34560];
    auto ix = this->mat.template ReinterpretCast<uint32_t>()[35584];
    auto outputIndices = this->mat.template ReinterpretCast<uint32_t>()[40000];
    this->SharedInverseBlock6(br, outputIndices);
    this->SharedInverseBlock5(batch, residual, corrected, inv, chunk, acc, cf, cache, br, outputIndices);
    this->Sync();
    this->FinishRefinement(corrected, inv, out);
}

template <class T, uint32_t N, bool FIXP> __aicore__ inline void PinvInverse199<T, N, FIXP>::InitBlock50()
{
    if constexpr (N == 199 && sizeof(T) == 4) {
        uint32_t singleCols = (this->t.pp + this->t.group * 8 - 1) / (this->t.group * 8) * 8;
        if (singleCols < 16)
            singleCols = 16;
        if (singleCols <= this->t.cols && singleCols * this->ld <= 19000) {
            this->panelCols = singleCols;
            this->cols = singleCols;
            this->first = this->part * this->cols;
        }
    }
}

template <class T, uint32_t N, bool FIXP>
__aicore__ inline void PinvInverse199<T, N, FIXP>::Init(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws, LinalgPinvTilingData tt)
{
    this->outputReady = false;
    this->InitShape(tt);
    InitBlock50();
    this->InitGlobalViews(ai, ao, ws);
    InitSocState();
    this->mat = LocalTensor<float>(TPosition::VECCALC, 0, 48640);
    this->InitLocalViews();
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType45_0, typename ExtractType45_1, typename ExtractType45_2, typename ExtractType45_3,
          typename ExtractType45_4, typename ExtractType45_5, typename ExtractType45_6, typename ExtractType45_7>
__aicore__ inline void PinvInverse199<T, N, FIXP>::RunBlock29Block34Block43Block45(
    ExtractType45_0 &ac, ExtractType45_1 &pc, ExtractType45_2 &stop, ExtractType45_3 &j, ExtractType45_4 &lCur,
    ExtractType45_5 &hCur, ExtractType45_6 &lNxt, ExtractType45_7 &hNxt)
{
    if (j + 1 < stop) {
        auto nextPiv = this->panel[(j + 1 - pc) * (this->t.pp + 8)];
        uint32_t nextRow = (uint32_t)(int32_t)nextPiv.GetValue(this->t.pp);
        this->perm.SetValue(j + 1, nextRow);
        this->invs.SetValue(j + 1, nextPiv.GetValue(this->t.pp + 1));
        if (this->first > pc)
            this->idx.SetValue(nextRow, uint32_t(0));
        uint32_t q0 = nextRow & 192, q1 = (q0 + 64) & 192, q2 = (q0 + 128) & 192, q3 = (q0 + 192) & 192;
        MulAddDst(this->mat[q0], this->piv[q0], (q0 & 64) ? hCur : lCur, q0 == 192 ? 8 : 64, ac, {1, 1, 0, 26, 0, 10});
        PipeBarrier<PIPE_V>();
        Gather(this->tmp, this->mat, this->ofs, nextRow * 4, ac);
        MulAddDst(this->mat[q1], this->piv[q1], (q1 & 64) ? hCur : lCur, q1 == 192 ? 8 : 64, ac, {1, 1, 0, 26, 0, 10});
        PipeBarrier<PIPE_V>();
        Brcb(lNxt, this->tmp, (ac + 7) / 8, {10, 80});
        Brcb(hNxt, this->tmp, (ac + 7) / 8, {10, 80});
        MulAddDst(this->mat[q2], this->piv[q2], (q2 & 64) ? hCur : lCur, q2 == 192 ? 8 : 64, ac, {1, 1, 0, 26, 0, 10});
        MulAddDst(this->mat[q3], this->piv[q3], (q3 & 64) ? hCur : lCur, q3 == 192 ? 8 : 64, ac, {1, 1, 0, 26, 0, 10});
        PipeBarrier<PIPE_V>();
    } else {
        MulAddDst(this->mat, this->piv, lCur, 64, ac, {1, 1, 0, 26, 0, 10});
        MulAddDst(this->mat[64], this->piv[64], hCur, 64, ac, {1, 1, 0, 26, 0, 10});
        MulAddDst(this->mat[128], this->piv[128], lCur, 64, ac, {1, 1, 0, 26, 0, 10});
        MulAddDst(this->mat[192], this->piv[192], hCur, 8, ac, {1, 1, 0, 26, 0, 10});
        PipeBarrier<PIPE_V>();
    }
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType43_0, typename ExtractType43_1, typename ExtractType43_2, typename ExtractType43_3,
          typename ExtractType43_4, typename ExtractType43_5, typename ExtractType43_6>
__aicore__ inline void
PinvInverse199<T, N, FIXP>::RunBlock29Block34Block43(ExtractType43_0 &ac, ExtractType43_1 &pc, ExtractType43_2 &stop,
                                                     ExtractType43_3 &low0, ExtractType43_4 &high0,
                                                     ExtractType43_5 &low1, ExtractType43_6 &high1)
{
    for (uint32_t j = pc; j < stop; j++) {
        this->piv = this->panel[(j - pc) * (this->t.pp + 8)];
        auto lCur = ((j - pc) & 1) ? low1 : low0, hCur = ((j - pc) & 1) ? high1 : high0;
        auto lNxt = ((j - pc) & 1) ? low0 : low1, hNxt = ((j - pc) & 1) ? high0 : high1;
        RunBlock29Block34Block43Block45(ac, pc, stop, j, lCur, hCur, lNxt, hNxt);
    }
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType34_0, typename ExtractType34_1, typename ExtractType34_2, typename ExtractType34_3>
__aicore__ inline void PinvInverse199<T, N, FIXP>::RunBlock29Block34(ExtractType34_0 &ac, ExtractType34_1 &pc,
                                                                     ExtractType34_2 &slot, ExtractType34_3 &stop)
{
    if (this->first != pc && ac > 0) {
        LocalSync<HardEvent::V_MTE2>();
        DataCopy(this->panel, this->exchange[slot], (stop - pc) * (this->t.pp + 8));
        this->Fence();
        auto low0 = this->mat[13376], high0 = this->mat[33408], low1 = this->mat[15424], high1 = this->mat[35456];
        this->piv = this->panel;
        uint32_t row = (uint32_t)(int32_t)this->piv.GetValue(this->t.pp);
        this->perm.SetValue(pc, row);
        this->invs.SetValue(pc, this->piv.GetValue(this->t.pp + 1));
        if (this->first > pc)
            this->idx.SetValue(row, uint32_t(0));
        Gather(this->tmp, this->mat, this->ofs, row * 4, ac);
        PipeBarrier<PIPE_V>();
        Brcb(low0, this->tmp, (ac + 7) / 8, {10, 80});
        Brcb(high0, this->tmp, (ac + 7) / 8, {10, 80});
        PipeBarrier<PIPE_V>();
        RunBlock29Block34Block43(ac, pc, stop, low0, high0, low1, high1);
    }
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType41_0, typename ExtractType41_1, typename ExtractType41_2, typename ExtractType41_3>
__aicore__ inline void PinvInverse199<T, N, FIXP>::RunBlock29Block37Block41(ExtractType41_0 &ac, ExtractType41_1 &bad,
                                                                            ExtractType41_2 &pc, ExtractType41_3 &stop)
{
    for (uint32_t j = pc; j < stop; j++) {
        auto col = this->mat[(j - this->first) * this->ld];
        And(this->tmp.template ReinterpretCast<uint16_t>(), col.template ReinterpretCast<uint16_t>(),
            this->idx.template ReinterpretCast<uint16_t>(), this->t.pp * 2);
        PipeBarrier<PIPE_V>();
        uint32_t row = this->Pivot();
        uint32_t c0 = j - this->first + 1;
        auto lowC = this->mat[13376 + (c0 & 7) * 80], highC = this->mat[33408 + (c0 & 7) * 80];
        if (c0 < ac) {
            Gather(this->tmp, this->mat[c0 * this->ld], this->ofs, row * 4, ac - c0);
            PipeBarrier<PIPE_V>();
            Brcb(lowC, this->tmp, (ac - c0 + 7) / 8, {10, 80});
            Brcb(highC, this->tmp, (ac - c0 + 7) / 8, {10, 80});
        }
        float inv = this->PreparePivot(col, j, pc, row, bad);
        this->perm.SetValue(j, row);
        this->invs.SetValue(j, inv);
        LocalSync<HardEvent::S_V>();
        Muls(this->piv, col, -inv, this->t.pp);
        PipeBarrier<PIPE_V>();
        if (c0 < ac) {
            MulAddDst(this->mat[c0 * this->ld], this->piv, lowC, 64, ac - c0, {1, 1, 0, 26, 0, 10});
            MulAddDst(this->mat[c0 * this->ld + 64], this->piv[64], highC, 64, ac - c0, {1, 1, 0, 26, 0, 10});
            MulAddDst(this->mat[c0 * this->ld + 128], this->piv[128], lowC, 64, ac - c0, {1, 1, 0, 26, 0, 10});
            MulAddDst(this->mat[c0 * this->ld + 192], this->piv[192], highC, 8, ac - c0, {1, 1, 0, 26, 0, 10});
            PipeBarrier<PIPE_V>();
        }
    }
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType37_0, typename ExtractType37_1, typename ExtractType37_2, typename ExtractType37_3,
          typename ExtractType37_4>
__aicore__ inline void PinvInverse199<T, N, FIXP>::RunBlock29Block37(ExtractType37_0 &ac, ExtractType37_1 &bad,
                                                                     ExtractType37_2 &pc, ExtractType37_3 &slot,
                                                                     ExtractType37_4 &stop)
{
    if (this->first == pc) {
        RunBlock29Block37Block41(ac, bad, pc, stop);
        LocalSync<HardEvent::V_MTE3>();
        DataCopy(this->exchange[slot], this->panel, (stop - pc) * (this->t.pp + 8));
        this->Fence();
    }
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType29_0, typename ExtractType29_1>
__aicore__ inline void PinvInverse199<T, N, FIXP>::RunBlock29(ExtractType29_0 &ac, ExtractType29_1 &bad)
{
    for (uint32_t pc = 0; pc < this->t.p; pc += this->panelCols) {
        uint32_t slot = ((pc / this->panelCols) & 1) * this->panelCols * (this->t.pp + 8),
                 stop = pc + this->panelCols < this->t.p ? pc + this->panelCols : this->t.p;
        RunBlock29Block37(ac, bad, pc, slot, stop);
        this->Sync();
        if (this->first == pc) {
            uint32_t cnt = stop - pc;
            this->RestoreOwnerRows(pc, cnt);
            for (uint32_t m = 1; m < cnt; m++) {
                this->piv = this->panel[m * this->ld];
                this->Update(0, m, this->perm.GetValue(pc + m));
            }
        }
        RunBlock29Block34(ac, pc, slot, stop);
        this->piv = this->mat[40000];
    }
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType40_0, typename ExtractType40_1, typename ExtractType40_2, typename ExtractType40_3,
          typename ExtractType40_4, typename ExtractType40_5>
__aicore__ inline void PinvInverse199<T, N, FIXP>::RunBlock32Block40(ExtractType40_0 &base, ExtractType40_1 &ac,
                                                                     ExtractType40_2 &invall, ExtractType40_3 &loaded,
                                                                     ExtractType40_4 &packed,
                                                                     ExtractType40_5 &rowIndices)
{
    if constexpr (FIXP && N == 199) {
        DataCopy(loaded, invall, 40000);
        for (uint32_t c = 0; c < 8; c++)
            rowIndices.SetValue(c, c * 200 * 4);
        LocalSync<HardEvent::S_V>();
        for (uint32_t step = 8; step < 200; step *= 2) {
            Adds(rowIndices[step].template ReinterpretCast<int32_t>(), rowIndices.template ReinterpretCast<int32_t>(),
                 int32_t(step * 200 * 4), step);
            PipeBarrier<PIPE_V>();
        }
        LocalSync<HardEvent::MTE2_V>();
        for (uint32_t r = this->part * 8; r < 199; r += this->t.group * 8) {
            uint32_t ac = 199 - r < 8 ? 199 - r : 8;
            for (uint32_t k = 0; k < ac; k++)
                Gather(packed[k * 200], loaded, rowIndices, (r + k) * 4, 199);
            LocalSync<HardEvent::V_MTE3>();
            DataCopyPad(this->output[base + r * 199], packed, {(uint16_t)ac, 199 * 4, 0, 0, 0});
            LocalSync<HardEvent::MTE3_V>();
        }
    } else {
        DataCopy(loaded, invall, this->t.p * this->t.pp);
        for (uint32_t c = 0; c < 8; c++)
            rowIndices.SetValue(c, c * this->t.pp * 4);
        LocalSync<HardEvent::S_V>();
        for (uint32_t step = 8; step < this->t.pp; step *= 2) {
            Adds(rowIndices[step].template ReinterpretCast<int32_t>(), rowIndices.template ReinterpretCast<int32_t>(),
                 int32_t(step * this->t.pp * 4), step);
            PipeBarrier<PIPE_V>();
        }
        LocalSync<HardEvent::MTE2_V>();
        for (uint32_t r = this->part * 8; r < this->t.p; r += this->t.group * 8) {
            uint32_t ac = this->t.p - r < 8 ? this->t.p - r : 8;
            for (uint32_t k = 0; k < ac; k++)
                Gather(packed[k * this->t.pp], loaded, rowIndices, (r + k) * 4, this->t.p);
            LocalSync<HardEvent::V_MTE3>();
            DataCopyPad(this->output[base + r * this->t.p], packed, {(uint16_t)ac, this->t.p * 4, 0, 0, 0});
            LocalSync<HardEvent::MTE3_V>();
        }
    }
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType32_0, typename ExtractType32_1, typename ExtractType32_2, typename ExtractType32_3,
          typename ExtractType32_4, typename ExtractType32_5>
__aicore__ inline bool PinvInverse199<T, N, FIXP>::RunBlock32(ExtractType32_0 &failed, ExtractType32_1 &base,
                                                              ExtractType32_2 &ac, ExtractType32_3 &invall,
                                                              ExtractType32_4 &unbounded, ExtractType32_5 &sensitive)
{
    if (!failed && !this->outputReady) {
        if constexpr (sizeof(T) == 4) {
            auto loaded = this->mat, packed = this->mat[40000];
            auto rowIndices = this->mat[42048].template ReinterpretCast<uint32_t>();
            RunBlock32Block40(base, ac, invall, loaded, packed, rowIndices);
        } else {
            this->WriteBf16Inverse(base, invall);
        }
        if (this->t.batch <= this->t.groups && unbounded == 0 && !sensitive)
            return true;
    }
    return false;
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType39_0, typename ExtractType39_1, typename ExtractType39_2>
__aicore__ inline void PinvInverse199<T, N, FIXP>::RunBlock39(ExtractType39_0 &ac, ExtractType39_1 &invall,
                                                              ExtractType39_2 &isq)
{
    if (ac > 0) {
        LocalSync<HardEvent::S_V>();
        ShiftLeft(this->idx, this->perm, uint32_t(2), (int32_t)this->t.p);
        Duplicate(this->rm, 0.0f, ac * this->t.pp);
        for (uint32_t r = this->t.p; r < this->t.pp; r++)
            this->invs.SetValue(r, 0.0f);
        LocalSync<HardEvent::S_V>();
        PipeBarrier<PIPE_V>();
        for (uint32_t c = 0; c < ac; c++)
            Gather(this->rm[c * this->t.pp], this->mat[c * this->ld], this->idx, 0, this->t.p);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < this->t.pp; r += 64) {
            uint32_t mask = this->t.pp - r < 64 ? this->t.pp - r : 64;
            Mul(this->rm[r], this->rm[r], this->invs[r], mask, ac,
                {1, 1, 1, (uint8_t)(this->t.pp / 8), (uint8_t)(this->t.pp / 8), 0});
        }
        LocalSync<HardEvent::V_MTE3>();
        for (uint32_t c = 0; c < ac; c++)
            DataCopy(invall[this->perm.GetValue(this->first + c) * this->t.pp], this->rm[c * this->t.pp], this->t.pp);
        Duplicate(this->mat, 0.0f, 1024);
        PipeBarrier<PIPE_V>();
        for (uint32_t c = 0; c < ac; c += 4) {
            MulAddDst(this->mat, this->rm[c * this->t.pp], this->rm[c * this->t.pp], this->t.pp);
            if (c + 1 < ac)
                MulAddDst(this->mat[256], this->rm[(c + 1) * this->t.pp], this->rm[(c + 1) * this->t.pp], this->t.pp);
            if (c + 2 < ac)
                MulAddDst(this->mat[512], this->rm[(c + 2) * this->t.pp], this->rm[(c + 2) * this->t.pp], this->t.pp);
            if (c + 3 < ac)
                MulAddDst(this->mat[768], this->rm[(c + 3) * this->t.pp], this->rm[(c + 3) * this->t.pp], this->t.pp);
            PipeBarrier<PIPE_V>();
        }
        Add(this->mat, this->mat, this->mat[256], 512);
        PipeBarrier<PIPE_V>();
        Add(this->broad, this->mat, this->mat[512], this->t.pp);
        PipeBarrier<PIPE_V>();
        LocalSync<HardEvent::MTE3_S>();
        this->Sum(this->red, this->broad);
        LocalSync<HardEvent::V_S>();
        isq = this->red.GetValue(0);
    }
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType47_0, typename ExtractType47_1, typename ExtractType47_2, typename ExtractType47_3,
          typename ExtractType47_4>
__aicore__ inline void PinvInverse199<T, N, FIXP>::RunBlock47(ExtractType47_0 &needs, ExtractType47_1 &unbounded,
                                                              ExtractType47_2 &accept, ExtractType47_3 &myOk,
                                                              ExtractType47_4 &sensitive)
{
    for (uint32_t g = 0; g < this->t.groups; g++) {
        uint32_t idx8 = g * 8;
        float aa = this->red.GetValue(idx8), ii = this->red.GetValue(idx8 + 1), bb = this->red.GetValue(idx8 + 2);
        if (!(ii * 1.4210854715202004e-14f < 6.4e-9f))
            sensitive = true;
        this->AssessGroup(aa, ii, bb, g, needs, unbounded, accept, myOk);
    }
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType52_0, typename ExtractType52_1, typename ExtractType52_2, typename ExtractType52_3,
          typename ExtractType52_4>
__aicore__ inline void PinvInverse199<T, N, FIXP>::RunBlock52(ExtractType52_0 &batch, ExtractType52_1 &needs,
                                                              ExtractType52_2 &unbounded, ExtractType52_3 &accept,
                                                              ExtractType52_4 &myOk)
{
    if (needs > 0) {
        this->CheckSpectralBound(batch, unbounded, myOk, accept);
    }
}
