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

template <class T> __aicore__ inline void PinvJacobi<T>::FormGram(GM_ADDR gram)
{
    GlobalTensor<float> gg;
    gg.SetGlobalBuffer((__gm__ float *)gram);
    auto cache = mem[24576];
    uint32_t chunk = 16384 / t.qp;
    if (chunk > 64)
        chunk = 64;
    chunk = chunk / 8 * 8;
    for (batch = groupid; batch < t.batch; batch += t.groups) {
        base = batch * t.m * t.n;
        work = all[batch * t.p * (t.qp + t.pp)];
        inputCached = false;
        for (uint32_t c = part; c < t.p; c += t.group) {
            ReadInput(c, u);
            Write(work[c * t.qp], u, t.qp);
        }
        Sync();
        inputCached = true;
        for (uint32_t cc = 0; cc < t.p; cc += chunk) {
            uint32_t kc = t.p - cc < chunk ? t.p - cc : chunk;
            DataCopy(cache, work[cc * t.qp], kc * t.qp);
            Fence();
            for (uint32_t row = part; row < t.p; row += t.group) {
                ReadInput(row, u);
                for (uint32_t c = 0; c < kc; c++)
                    out.SetValue(c, Dot(u, cache[c * t.qp], t.qp));
                LocalSync<HardEvent::S_MTE3>();
                DataCopyPad(gg[(batch * t.p + row) * t.p + cc], out, {1, kc * 4, 0, 0, 0});
                Fence();
            }
        }
        Sync();
    }
}

template <class T> __aicore__ inline void PinvJacobi<T>::GramOutput(GM_ADDR inverse)
{
    GlobalTensor<float> ig;
    ig.SetGlobalBuffer((__gm__ float *)inverse);
    inputCached = true;
    for (batch = groupid; batch < t.batch; batch += t.groups) {
        base = batch * t.m * t.n;
        work = all[batch * t.p * (t.qp + t.pp)];
        vg = work[t.p * t.qp];
        y = yall[batch * t.p * t.qp];
        for (uint32_t row = part; row < t.p; row += t.group) {
            DataCopyPad(vj, ig[(batch * t.p + row) * t.p], {1, t.p * 4, 0, 0, 0}, {true, 0, (uint8_t)(t.pp - t.p), 0});
            Fence();
            Duplicate(out, 0.0f, t.qp);
            Fence();
            for (uint32_t col = 0; col < t.p; col++) {
                ReadInput(col, u);
                Axpy(out, u, vj.GetValue(col), t.qp);
                Fence();
            }
            Write(y[row * t.qp], out, t.qp);
        }
        Sync();
        Refine();
        Sync();
    }
}

template <class T>
template <typename ExtractType64_0>
__aicore__ inline void PinvJacobi<T>::RunBlock60Block64(ExtractType64_0 &blocks)
{
    if (part == 0) {
        float largest = 0;
        for (uint32_t col = 0; col < t.p; col++) {
            float z = Load(ns[col * 8]);
            if (z > largest)
                largest = z;
        }
        Store(state[blocks * 8 + groupid * 8], largest);
    }
}

template <class T>
template <typename ExtractType60_0, typename ExtractType60_1, typename ExtractType60_2>
__aicore__ inline void PinvJacobi<T>::RunBlock60(ExtractType60_0 &blocks, ExtractType60_1 &circle,
                                                 ExtractType60_2 &pairs)
{
    for (uint32_t sweep = 0; sweep < 20; sweep++) {
        float mx = Load(state[blocks * 8 + groupid * 8]), rt = t.rtol * t.rtol;
        float tiny = 0.01f * (t.atol * t.atol > mx * rt ? t.atol * t.atol : mx * rt);
        bool changed = false;
        for (uint32_t round = 0; round < circle; round++) {
            for (uint32_t pair = part; pair < pairs; pair += t.group) {
                uint32_t j = (round + pair) % circle, k = pair == 0 ? circle : (round + circle - pair) % circle;
                if (j < t.p && k < t.p && Pair(j, k, tiny))
                    changed = true;
            }
            Sync();
        }
        Store(state[GetBlockIdx() * 8], changed ? 1.0f : 0.0f);
        Sync();
        RunBlock60Block64(blocks);
        if (GetBlockIdx() == 0) {
            float any = 0;
            for (uint32_t i = 0; i < blocks; i++)
                any += Load(state[i * 8]);
            Store(state[(blocks + t.groups) * 8], any);
        }
        Sync();
        if (Load(state[(blocks + t.groups) * 8]) == 0)
            break;
    }
}

template <class T> __aicore__ inline void PinvJacobi<T>::Run()
{
    auto normsAll = ns;
    uint32_t blocks = t.group * t.groups;
    for (batch = groupid; batch < t.batch; batch += t.groups) {
        base = batch * t.m * t.n;
        work = all[batch * t.p * (t.qp + t.pp)];
        vg = work[t.p * t.qp];
        ns = normsAll[batch * t.p * 8];
        y = yall[batch * t.p * t.qp];
        for (uint32_t col = part; col < t.p; col += t.group) {
            ReadInput(col, u);
            float norm = Dot(u, u, t.qp);
            Write(work[col * t.qp], u, t.qp);
            Duplicate(v, 0.0f, t.pp);
            Fence();
            v.SetValue(col, 1.0f);
            Fence();
            Write(vg[col * t.pp], v, t.pp);
            Store(ns[col * 8], norm);
        }
        Sync();
        if (part == 0) {
            float mx = 0;
            for (uint32_t col = 0; col < t.p; col++) {
                float z = Load(ns[col * 8]);
                if (z > mx)
                    mx = z;
            }
            Store(state[blocks * 8 + groupid * 8], mx);
        }
        Sync();
        uint32_t circle = (t.p + 1) / 2 * 2 - 1, pairs = (t.p + 1) / 2;
        RunBlock60(blocks, circle, pairs);
        Rebuild();
        float mx = Load(state[blocks * 8 + groupid * 8]);
        float threshold2 = t.atol * t.atol > mx * t.rtol * t.rtol ? t.atol * t.atol : mx * t.rtol * t.rtol;
        for (uint32_t row = part; row < t.p; row += t.group)
            OutputRow(row, threshold2);
        Sync();
        Refine();
        Sync();
    }
}
