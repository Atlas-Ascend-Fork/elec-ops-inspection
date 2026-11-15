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
__aicore__ inline float PinvInverse199<T, N, FIXP>::LoadBatchInput(uint32_t &batch, uint32_t &base, uint32_t &ac)
{
    this->LoadBatchColumns(base, ac);
    Duplicate(this->panel, 0.0f, 1024);
    PipeBarrier<PIPE_V>();
    this->SharedInverseBlock15(ac);
    Add(this->panel, this->panel, this->panel[256], 512);
    PipeBarrier<PIPE_V>();
    Add(this->tmp, this->panel, this->panel[512], this->t.pp);
    PipeBarrier<PIPE_V>();
    this->Sum(this->red, this->tmp);
    Duplicate(this->idx, uint32_t(0x7fffffff), this->t.pp);
    LocalSync<HardEvent::V_S>();
    return this->red.GetValue(0);
}

template <class T, uint32_t N, bool FIXP>
template <typename ExtractType53_0>
__aicore__ inline bool PinvInverse199<T, N, FIXP>::RunBlock53(ExtractType53_0 &failed)
{
    for (uint32_t batch = this->groupid; batch < this->t.batch; batch += this->t.groups) {
        this->outputReady = false;
        uint32_t base = batch * this->t.p * this->t.p,
                 ac = this->first < this->t.p
                          ? (this->t.p - this->first < this->cols ? this->t.p - this->first : this->cols)
                          : 0;
        float asq = LoadBatchInput(batch, base, ac);
        LocalSync<HardEvent::S_V>();
        bool bad = false;
        RunBlock29(ac, bad);
        auto invall = this->iall[batch * this->t.pp * this->t.pp];
        float isq = 0;
        RunBlock39(ac, invall, isq);
        this->Store(this->state[(this->part * this->t.groups + this->groupid) * 8], asq, isq, bad ? 1.0f : 0.0f);
        this->Sync();
        DataCopy(this->red, this->state, this->t.group * this->t.groups * 8);
        LocalSync<HardEvent::MTE2_V>();
        this->SharedInverseBlock12();
        LocalSync<HardEvent::V_S>();
        float needs = 0, unbounded = 0;
        bool accept = false, myOk = false, sensitive = false;
        RunBlock47(needs, unbounded, accept, myOk, sensitive);
        if (sensitive) {
            accept = false;
            needs = 0;
        }
        RunBlock52(batch, needs, unbounded, accept, myOk);
        if (!accept)
            failed = true;
        if (RunBlock32(failed, base, ac, invall, unbounded, sensitive))
            return true;
        this->Sync();
    }
    return false;
}

template <class T, uint32_t N, bool FIXP> __aicore__ inline bool PinvInverse199<T, N, FIXP>::Run()
{
    bool failed = false;
    if (RunBlock53(failed))
        return true;
    return this->FinalResult(failed);
}
