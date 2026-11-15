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

template <class T, uint32_t N = 0> class PinvDeferred256 : private PinvInverseState<T, N, true> {
    bool needsRefinement = false;

    __aicore__ inline void RefineInverse(uint32_t batch)
    {
        needsRefinement = true;
        this->outputReady = true;
    }

public:
    __aicore__ inline bool NeedsRefine()
    {
        return needsRefinement;
    }

    __aicore__ inline void InitBlock16()
    {
        if constexpr (N == 256 && sizeof(T) == 4) {
            uint32_t singleCols = (this->t.pp + this->t.group * 8 - 1) / (this->t.group * 8) * 8;
            if (singleCols <= this->t.cols && singleCols * this->ld <= 19000) {
                this->panelCols = singleCols;
                this->cols = singleCols;
                this->first = this->part * this->cols;
            }
        } else
            this->SharedInverseBlock8();
    }

    __aicore__ inline void Init(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws, LinalgPinvTilingData tt)
    {
        this->outputReady = false;
        this->InitShape(tt);
        InitBlock16();
        this->InitGlobalViews(ai, ao, ws);
        InitSocState();
        this->mat = LocalTensor<float>(TPosition::VECCALC, 0, 48640);
        this->InitLocalViews();
    }

    template <typename ExtractType15_0, typename ExtractType15_1, typename ExtractType15_2, typename ExtractType15_3,
              typename ExtractType15_4, typename ExtractType15_5, typename ExtractType15_6>
    __aicore__ inline void RunBlock10Block11Block15(ExtractType15_0 &ac, ExtractType15_1 &pc, ExtractType15_2 &stop,
                                                    ExtractType15_3 &br0, ExtractType15_4 &op0, ExtractType15_5 &br1,
                                                    ExtractType15_6 &op1)
    {
        for (uint32_t j = pc; j < stop; j++) {
            this->piv = this->panel[(j - pc) * (this->t.pp + 8)];
            auto bCur = ((j - pc) & 1) ? br1 : br0, oCur = ((j - pc) & 1) ? op1 : op0;
            auto bNxt = ((j - pc) & 1) ? br0 : br1, oNxt = ((j - pc) & 1) ? op0 : op1;
            if (j + 1 < stop) {
                auto nextPiv = this->panel[(j + 1 - pc) * (this->t.pp + 8)];
                uint32_t nextRow = (uint32_t)(int32_t)nextPiv.GetValue(this->t.pp);
                this->perm.SetValue(j + 1, nextRow);
                this->invs.SetValue(j + 1, nextPiv.GetValue(this->t.pp + 1));
                if (this->first > pc)
                    this->idx.SetValue(nextRow, uint32_t(0));
                uint32_t q0 = nextRow & 192, q1 = (q0 + 64) & 192, q2 = (q0 + 128) & 192, q3 = (q0 + 192) & 192;
                MulAddDst(this->mat[q0], this->piv[q0], (q0 & 64) ? oCur : bCur, 64, ac, {1, 1, 0, 33, 0, 1});
                PipeBarrier<PIPE_V>();
                Gather(this->tmp, this->mat, this->ofs, nextRow * 4, ac);
                MulAddDst(this->mat[q1], this->piv[q1], (q1 & 64) ? oCur : bCur, 64, ac, {1, 1, 0, 33, 0, 1});
                PipeBarrier<PIPE_V>();
                Brcb(bNxt, this->tmp, (ac + 7) / 8, {1, 8});
                Brcb(oNxt, this->tmp, (ac + 7) / 8, {1, 8});
                MulAddDst(this->mat[q2], this->piv[q2], (q2 & 64) ? oCur : bCur, 64, ac, {1, 1, 0, 33, 0, 1});
                MulAddDst(this->mat[q3], this->piv[q3], (q3 & 64) ? oCur : bCur, 64, ac, {1, 1, 0, 33, 0, 1});
                PipeBarrier<PIPE_V>();
            } else {
                MulAddDst(this->mat, this->piv, bCur, 64, ac, {1, 1, 0, 33, 0, 1});
                MulAddDst(this->mat[64], this->piv[64], oCur, 64, ac, {1, 1, 0, 33, 0, 1});
                MulAddDst(this->mat[128], this->piv[128], bCur, 64, ac, {1, 1, 0, 33, 0, 1});
                MulAddDst(this->mat[192], this->piv[192], oCur, 64, ac, {1, 1, 0, 33, 0, 1});
                PipeBarrier<PIPE_V>();
            }
        }
    }

    template <typename ExtractType11_0, typename ExtractType11_1, typename ExtractType11_2, typename ExtractType11_3>
    __aicore__ inline void RunBlock10Block11(ExtractType11_0 &ac, ExtractType11_1 &pc, ExtractType11_2 &slot,
                                             ExtractType11_3 &stop)
    {
        if (this->first != pc && ac > 0) {
            LocalSync<HardEvent::V_MTE2>();
            DataCopy(this->panel, this->exchange[slot], (stop - pc) * (this->t.pp + 8));
            this->Fence();
            if constexpr (N == 256) {
                ReceiverFixed256(pc, stop - pc, this->first, ac);
            } else {
                auto br0 = this->broad, op0 = this->mat[38912], br1 = this->mat[38400], op1 = this->mat[39424];
                this->piv = this->panel;
                uint32_t row = (uint32_t)(int32_t)this->piv.GetValue(this->t.pp);
                this->perm.SetValue(pc, row);
                this->invs.SetValue(pc, this->piv.GetValue(this->t.pp + 1));
                if (this->first > pc)
                    this->idx.SetValue(row, uint32_t(0));
                Gather(this->tmp, this->mat, this->ofs, row * 4, ac);
                PipeBarrier<PIPE_V>();
                Brcb(br0, this->tmp, (ac + 7) / 8, {1, 8});
                Brcb(op0, this->tmp, (ac + 7) / 8, {1, 8});
                PipeBarrier<PIPE_V>();
                RunBlock10Block11Block15(ac, pc, stop, br0, op0, br1, op1);
            }
        }
    }

    template <typename ExtractType14_0, typename ExtractType14_1, typename ExtractType14_2, typename ExtractType14_3,
              typename ExtractType14_4>
    __aicore__ inline void RunBlock10Block12Block14(ExtractType14_0 &ac, ExtractType14_1 &bad, ExtractType14_2 &pc,
                                                    ExtractType14_3 &stop, ExtractType14_4 &opposite)
    {
        for (uint32_t j = pc; j < stop; j++) {
            auto col = this->mat[(j - this->first) * this->ld];
            And(this->tmp.template ReinterpretCast<uint16_t>(), col.template ReinterpretCast<uint16_t>(),
                this->idx.template ReinterpretCast<uint16_t>(), this->t.pp * 2);
            PipeBarrier<PIPE_V>();
            uint32_t row = this->Pivot();
            uint32_t c0 = j - this->first + 1;
            if (c0 < ac) {
                Gather(this->tmp, this->mat[c0 * this->ld], this->ofs, row * 4, ac - c0);
                PipeBarrier<PIPE_V>();
                Brcb(this->broad, this->tmp, (ac - c0 + 7) / 8, {1, 8});
                Brcb(opposite, this->tmp, (ac - c0 + 7) / 8, {1, 8});
            }
            float inv = this->PreparePivot(col, j, pc, row, bad);
            this->perm.SetValue(j, row);
            this->invs.SetValue(j, inv);
            LocalSync<HardEvent::S_V>();
            Muls(this->piv, col, -inv, this->t.pp);
            PipeBarrier<PIPE_V>();
            if (c0 < ac) {
                MulAddDst(this->mat[c0 * this->ld], this->piv, this->broad, 64, ac - c0, {1, 1, 0, 33, 0, 1});
                MulAddDst(this->mat[c0 * this->ld + 64], this->piv[64], opposite, 64, ac - c0, {1, 1, 0, 33, 0, 1});
                MulAddDst(this->mat[c0 * this->ld + 128], this->piv[128], this->broad, 64, ac - c0,
                          {1, 1, 0, 33, 0, 1});
                MulAddDst(this->mat[c0 * this->ld + 192], this->piv[192], opposite, 64, ac - c0, {1, 1, 0, 33, 0, 1});
                PipeBarrier<PIPE_V>();
            }
        }
    }

    template <typename ExtractType12_0, typename ExtractType12_1, typename ExtractType12_2, typename ExtractType12_3,
              typename ExtractType12_4>
    __aicore__ inline void RunBlock10Block12(ExtractType12_0 &ac, ExtractType12_1 &bad, ExtractType12_2 &pc,
                                             ExtractType12_3 &slot, ExtractType12_4 &stop)
    {
        if (this->first == pc) {
            if constexpr (N == 256) {
                if (FactorOwnerFixed256(pc, stop - pc))
                    bad = true;
            } else {
                auto opposite = this->mat[38912];
                RunBlock10Block12Block14(ac, bad, pc, stop, opposite);
            }
            LocalSync<HardEvent::V_MTE3>();
            DataCopy(this->exchange[slot], this->panel, (stop - pc) * (this->t.pp + 8));
            this->Fence();
        }
    }

    template <typename ExtractType10_0, typename ExtractType10_1>
    __aicore__ inline void RunBlock10(ExtractType10_0 &ac, ExtractType10_1 &bad)
    {
        for (uint32_t pc = 0; pc < this->t.p; pc += this->panelCols) {
            uint32_t slot = ((pc / this->panelCols) & 1) * this->panelCols * (this->t.pp + 8),
                     stop = pc + this->panelCols < this->t.p ? pc + this->panelCols : this->t.p;
            RunBlock10Block12(ac, bad, pc, slot, stop);
            this->Sync();
            if (this->first == pc) {
                uint32_t cnt = stop - pc;
                this->RestoreOwnerRows(pc, cnt);
                for (uint32_t m = 1; m < cnt; m++) {
                    this->piv = this->panel[m * this->ld];
                    this->Update(0, m, this->perm.GetValue(pc + m));
                }
            }
            RunBlock10Block11(ac, pc, slot, stop);
            this->piv = this->mat[40000];
        }
    }

    template <typename ExtractType17_0, typename ExtractType17_1, typename ExtractType17_2>
    __aicore__ inline void RunBlock17(ExtractType17_0 &ac, ExtractType17_1 &invall, ExtractType17_2 &isq)
    {
        if (ac > 0) {
            LocalSync<HardEvent::S_V>();
            ShiftLeft(this->idx, this->perm, uint32_t(2), (int32_t)this->t.p);
            PipeBarrier<PIPE_V>();
            for (uint32_t c = 0; c < ac; c++)
                Gather(this->rm[c * this->t.pp], this->mat[c * this->ld], this->idx, 0, this->t.p);
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0; r < this->t.pp; r += 64)
                Mul(this->rm[r], this->rm[r], this->invs[r], 64, ac,
                    {1, 1, 1, (uint8_t)(this->t.pp / 8), (uint8_t)(this->t.pp / 8), 0});
            LocalSync<HardEvent::V_MTE3>();
            for (uint32_t c = 0; c < ac; c++)
                DataCopy(invall[this->perm.GetValue(this->first + c) * this->t.pp], this->rm[c * this->t.pp],
                         this->t.pp);
            Mul(this->mat, this->rm, this->rm, ac * this->t.pp);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(this->tmp, this->mat, 64, ac * 4, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            if (ac * 4 > 64) {
                Add(this->tmp, this->tmp, this->tmp[64], ac * 4 - 64);
                PipeBarrier<PIPE_V>();
            }
            WholeReduceSum(this->red, this->tmp, ac * 4 > 64 ? 64 : ac * 4, 1, 1, 1, 8);
            LocalSync<HardEvent::MTE3_S>();
            LocalSync<HardEvent::V_S>();
            isq = this->red.GetValue(0);
        }
    }

    template <typename ExtractType23_0, typename ExtractType23_1, typename ExtractType23_2, typename ExtractType23_3,
              typename ExtractType23_4>
    __aicore__ inline void RunBlock23(ExtractType23_0 &batch, ExtractType23_1 &needs, ExtractType23_2 &unbounded,
                                      ExtractType23_3 &accept, ExtractType23_4 &myOk)
    {
        if (needs > 0) {
            this->CheckSpectralBound(batch, unbounded, myOk, accept);
            RefineInverse(batch);
        }
    }

    template <typename ExtractType24_0, typename ExtractType24_1>
    __aicore__ inline void RunBlock24(ExtractType24_0 &batch, ExtractType24_1 &ac)
    {
        if constexpr (sizeof(T) == 4 && N > 0) {
            if (ac) {
                auto savedA = this->state[(this->t.group * this->t.groups + 1) * 8 +
                                          this->t.batch * this->t.pp * this->t.pp + batch * this->t.pp * this->t.pp];
                LocalSync<HardEvent::V_MTE3>();
                DataCopyPad(savedA[this->first * this->t.pp], this->mat, {(uint16_t)ac, this->t.pp * 4, 1, 0, 0});
            }
        }
    }

    __aicore__ inline float LoadBatchInput(uint32_t &batch, uint32_t &base, uint32_t &ac)
    {
        this->LoadBatchColumns(base, ac);
        RunBlock24(batch, ac);
        Duplicate(this->panel, 0.0f, 1024);
        PipeBarrier<PIPE_V>();
        this->SharedInverseBlock15(ac);
        Add(this->panel, this->panel, this->panel[256], 512);
        PipeBarrier<PIPE_V>();
        Add(this->tmp, this->panel, this->panel[512], 256);
        PipeBarrier<PIPE_V>();
        this->Sum(this->red, this->tmp);
        Duplicate(this->idx, uint32_t(0x7fffffff), this->t.pp);
        LocalSync<HardEvent::V_S>();
        return this->red.GetValue(0);
    }

    template <typename ExtractType25_0> __aicore__ inline bool RunBlock25(ExtractType25_0 &failed)
    {
        for (uint32_t batch = this->groupid; batch < this->t.batch; batch += this->t.groups) {
            this->outputReady = false;
            uint32_t base = batch * this->t.p * this->t.p,
                     ac = this->first < this->t.p
                              ? (this->t.p - this->first < this->cols ? this->t.p - this->first : this->cols)
                              : 0;
            float asq = LoadBatchInput(batch, base, ac);
            if constexpr (sizeof(T) == 4 && N > 0) {
                if (ac)
                    LocalSync<HardEvent::MTE3_V>();
            }
            LocalSync<HardEvent::S_V>();
            bool bad = false;
            RunBlock10(ac, bad);
            auto invall = this->iall[batch * this->t.pp * this->t.pp];
            float isq = 0;
            RunBlock17(ac, invall, isq);
            auto assessment = this->AssessBatch(asq, isq, bad);
            RunBlock23(batch, assessment.needs, assessment.unbounded, assessment.accept, assessment.myOk);
            if (!assessment.accept)
                failed = true;
            if (this->t.batch <= this->t.groups && assessment.unbounded == 0 && !assessment.sensitive &&
                this->outputReady)
                return true;
            this->SharedInverseBlock9(failed, base, ac, invall);
            this->Sync();
        }
        return false;
    }

    __aicore__ inline bool Run()
    {
        bool failed = false;
        if (RunBlock25(failed))
            return true;
        return this->FinalResult(failed);
    }
};
