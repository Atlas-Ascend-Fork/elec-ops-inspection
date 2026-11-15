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

template <class T, uint32_t N> __aicore__ inline void ParallelInverse<T, N>::RefineOnly()
{
    for (uint32_t batch = this->groupid; batch < this->t.batch; batch += this->t.groups)
        CubeRefine(batch);
}

template <class T, uint32_t N>
__aicore__ inline void ParallelInverse<T, N>::Init(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws, LinalgPinvTilingData tt)
{
    this->outputReady = false;
    this->InitShape(tt);
    this->SharedInverseBlock8();
    this->InitGlobalViews(ai, ao, ws);
    pipe.InitBuffer(buf, 194560);
    this->mat = buf.Get<float>();
    this->InitLocalViews();
}

template <class T, uint32_t N>
template <typename ExtractType79_0, typename ExtractType79_1, typename ExtractType79_2, typename ExtractType79_3>
__aicore__ inline void ParallelInverse<T, N>::RunBlock70Block79(ExtractType79_0 &bad, ExtractType79_1 &pc,
                                                                ExtractType79_2 &slot, ExtractType79_3 &stop)
{
    if (this->first == pc) {
        for (uint32_t j = pc; j < stop; j++) {
            auto col = this->mat[(j - this->first) * this->ld];
            And(this->tmp.template ReinterpretCast<uint16_t>(), col.template ReinterpretCast<uint16_t>(),
                this->idx.template ReinterpretCast<uint16_t>(), this->t.pp * 2);
            PipeBarrier<PIPE_V>();
            uint32_t row = this->Pivot();
            float inv = this->PreparePivot(col, j, pc, row, bad);
            LocalSync<HardEvent::S_V>();
            Muls(this->piv, col, -inv, this->t.pp);
            this->Update(j - this->first + 1, this->cols, row);
        }
        if (this->first + this->cols > this->t.pp && this->first < this->t.pp + this->t.p) {
            Gather(this->tmp, this->panel, this->ofs, this->t.pp * 4, stop - pc);
            PipeBarrier<PIPE_V>();
            Cast(this->perm[pc].template ReinterpretCast<int32_t>(), this->tmp, RoundMode::CAST_RINT, stop - pc);
            Gather(this->invs[pc], this->panel, this->ofs, (this->t.pp + 1) * 4, stop - pc);
            PipeBarrier<PIPE_V>();
        }
        LocalSync<HardEvent::V_MTE3>();
        DataCopy(this->exchange[slot], this->panel, (stop - pc) * (this->t.pp + 8));
        this->Fence();
    }
}

template <class T, uint32_t N>
template <typename ExtractType70_0, typename ExtractType70_1>
__aicore__ inline void ParallelInverse<T, N>::RunBlock70(ExtractType70_0 &bad, ExtractType70_1 &c1)
{
    for (uint32_t pc = 0; pc < this->t.p; pc += this->panelCols) {
        uint32_t slot = ((pc / this->panelCols) & 1) * this->panelCols * (this->t.pp + 8),
                 stop = pc + this->panelCols < this->t.p ? pc + this->panelCols : this->t.p;
        RunBlock70Block79(bad, pc, slot, stop);
        this->Sync();
        if (this->first > pc && c1 > 0) {
            LocalSync<HardEvent::V_MTE2>();
            DataCopy(this->panel, this->exchange[slot], (stop - pc) * (this->t.pp + 8));
            this->Fence();
            if (this->first + this->cols > this->t.pp && this->first < this->t.pp + this->t.p) {
                Gather(this->tmp, this->panel, this->ofs, this->t.pp * 4, stop - pc);
                PipeBarrier<PIPE_V>();
                Cast(this->perm[pc].template ReinterpretCast<int32_t>(), this->tmp, RoundMode::CAST_RINT, stop - pc);
                Gather(this->invs[pc], this->panel, this->ofs, (this->t.pp + 1) * 4, stop - pc);
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t j = pc; j < stop; j++) {
                this->piv = this->panel[(j - pc) * (this->t.pp + 8)];
                uint32_t row = (uint32_t)(int32_t)this->piv.GetValue(this->t.pp);
                if (this->first < this->t.pp)
                    this->idx.SetValue(row, uint32_t(0));
                this->Update(0, c1, row);
            }
        }
        this->piv = this->mat[40000];
    }
}

template <class T, uint32_t N>
template <typename ExtractType76_0>
__aicore__ inline void ParallelInverse<T, N>::RunBlock74Block76(ExtractType76_0 &invall)
{
    if constexpr (sizeof(T) == 4) {
        uint32_t begin = this->first < this->t.pp ? this->t.pp - this->first : 0;
        uint32_t end =
            this->first + this->cols > this->t.pp + this->t.p ? this->t.pp + this->t.p - this->first : this->cols;
        uint32_t count = end - begin;
        if (this->t.pp > this->t.p) {
            Duplicate(this->rm, 0.0f, count * this->t.pp);
            for (uint32_t r = this->t.p; r < this->t.pp; r++)
                this->invs.SetValue(r, 0.0f);
            LocalSync<HardEvent::S_V>();
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t c = 0; c < count; c++)
            Gather(this->rm[c * this->t.pp], this->mat[(begin + c) * this->ld], this->idx, 0, this->t.p);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < this->t.pp; r += 64) {
            uint32_t mask = this->t.pp - r < 64 ? this->t.pp - r : 64;
            Mul(this->rm[r], this->rm[r], this->invs[r], mask, count,
                {1, 1, 1, (uint8_t)(this->t.pp / 8), (uint8_t)(this->t.pp / 8), 0});
        }
        LocalSync<HardEvent::V_MTE3>();
        DataCopy(invall[(this->first + begin - this->t.pp) * this->t.pp], this->rm, count * this->t.pp);
        for (uint32_t c = 0; c < count; c++) {
            MulAddDst(this->broad, this->rm[c * this->t.pp], this->rm[c * this->t.pp], this->t.pp);
            PipeBarrier<PIPE_V>();
        }
        LocalSync<HardEvent::MTE3_S>();
    } else {
        Duplicate(this->piv, 0.0f, this->t.pp);
        PipeBarrier<PIPE_V>();
        for (uint32_t c = 0; c < this->cols; c++)
            if (this->first + c >= this->t.pp && this->first + c < this->t.pp + this->t.p) {
                Gather(this->piv, this->mat[c * this->ld], this->idx, 0, this->t.p);
                PipeBarrier<PIPE_V>();
                Mul(this->piv, this->piv, this->invs, this->t.p);
                PipeBarrier<PIPE_V>();
                MulAddDst(this->broad, this->piv, this->piv, this->t.pp);
                LocalSync<HardEvent::V_MTE3>();
                DataCopy(invall[(this->first + c - this->t.pp) * this->t.pp], this->piv, this->t.pp);
                LocalSync<HardEvent::MTE3_V>();
            }
    }
}

template <class T, uint32_t N>
template <typename ExtractType74_0, typename ExtractType74_1>
__aicore__ inline void ParallelInverse<T, N>::RunBlock74(ExtractType74_0 &invall, ExtractType74_1 &isq)
{
    if (this->first + this->cols > this->t.pp && this->first < this->t.pp + this->t.p) {
        LocalSync<HardEvent::S_V>();
        ShiftLeft(this->idx, this->perm, uint32_t(2), (int32_t)this->t.p);
        Duplicate(this->broad, 0.0f, this->t.pp);
        RunBlock74Block76(invall);
        this->Sum(this->red, this->broad);
        LocalSync<HardEvent::V_S>();
        isq = this->red.GetValue(0);
    }
}

template <class T, uint32_t N>
template <typename ExtractType93_0, typename ExtractType93_1, typename ExtractType93_2, typename ExtractType93_3,
          typename ExtractType93_4>
__aicore__ inline void ParallelInverse<T, N>::RunBlock93(ExtractType93_0 &batch, ExtractType93_1 &needs,
                                                         ExtractType93_2 &unbounded, ExtractType93_3 &accept,
                                                         ExtractType93_4 &myOk)
{
    if (needs > 0) {
        this->CheckSpectralBound(batch, unbounded, myOk, accept);
        RefineInverse(batch);
    }
}

template <class T, uint32_t N>
template <typename ExtractType94_0, typename ExtractType94_1>
__aicore__ inline void ParallelInverse<T, N>::RunBlock94(ExtractType94_0 &batch, ExtractType94_1 &ac)
{
    if constexpr (sizeof(T) == 4 && N > 0) {
        if (ac) {
            auto savedA = this->state[(this->t.group * this->t.groups + 1) * 8 +
                                      this->t.batch * this->t.pp * this->t.pp + batch * this->t.pp * this->t.pp];
            LocalSync<HardEvent::V_MTE3>();
            DataCopyPad(savedA[this->first * this->t.pp], this->mat, {(uint16_t)ac, this->t.pp * 4, 1, 0, 0});
            LocalSync<HardEvent::MTE3_V>();
        }
    }
}

template <class T, uint32_t N>
__aicore__ inline float ParallelInverse<T, N>::LoadBatchInput(uint32_t &batch, uint32_t &base, uint32_t &ac)
{
    this->LoadBatchColumns(base, ac);
    RunBlock94(batch, ac);
    Duplicate(this->tmp, 0.0f, this->t.pp);
    PipeBarrier<PIPE_V>();
    for (uint32_t c = 0; c < ac; c++) {
        MulAddDst(this->tmp, this->mat[c * this->ld], this->mat[c * this->ld], this->t.pp);
        PipeBarrier<PIPE_V>();
    }
    this->Sum(this->red, this->tmp);
    for (uint32_t c = 0; c < this->cols; c++)
        if (this->first + c >= this->t.pp && this->first + c < 2 * this->t.pp)
            Duplicate(this->mat[c * this->ld], 0.0f, this->t.pp);
    Duplicate(this->idx, uint32_t(0x7fffffff), this->t.pp);
    LocalSync<HardEvent::V_S>();
    return this->red.GetValue(0);
}

template <class T, uint32_t N>
template <typename ExtractType97_0>
__aicore__ inline bool ParallelInverse<T, N>::RunBlock97(ExtractType97_0 &failed)
{
    for (uint32_t batch = this->groupid; batch < this->t.batch; batch += this->t.groups) {
        this->outputReady = false;
        cubePrecision = true;
        uint32_t base = batch * this->t.p * this->t.p,
                 ac = this->first < this->t.p
                          ? (this->t.p - this->first < this->cols ? this->t.p - this->first : this->cols)
                          : 0;
        float asq = LoadBatchInput(batch, base, ac);
        for (uint32_t c = 0; c < this->cols; c++)
            if (this->first + c >= this->t.pp && this->first + c < this->t.pp + this->t.p)
                this->mat.SetValue(c * this->ld + this->first + c - this->t.pp, 1.0f);
        LocalSync<HardEvent::S_V>();
        bool bad = false;
        uint32_t c1 = this->first < 2 * this->t.pp
                          ? (2 * this->t.pp - this->first < this->cols ? 2 * this->t.pp - this->first : this->cols)
                          : 0;
        RunBlock70(bad, c1);
        auto invall = this->iall[batch * this->t.pp * this->t.pp];
        float isq = 0;
        RunBlock74(invall, isq);
        auto assessment = this->AssessBatch(asq, isq, bad);
        RunBlock93(batch, assessment.needs, assessment.unbounded, assessment.accept, assessment.myOk);
        if (!assessment.accept)
            failed = true;
        if (this->t.batch <= this->t.groups && assessment.unbounded == 0 && !assessment.sensitive && this->outputReady)
            return true;
        this->SharedInverseBlock9(failed, base, ac, invall);
        this->Sync();
    }
    return false;
}

template <class T, uint32_t N> __aicore__ inline bool ParallelInverse<T, N>::Run()
{
    bool failed = false;
    if (RunBlock97(failed))
        return true;
    return this->FinalResult(failed);
}
