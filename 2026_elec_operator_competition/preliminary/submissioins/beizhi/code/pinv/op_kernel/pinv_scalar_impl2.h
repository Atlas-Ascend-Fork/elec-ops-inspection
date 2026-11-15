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

template <class T> __aicore__ inline void PinvScalar<T>::AssembleBlock101()
{
    if (t.m >= t.n) {
        for (uint32_t j = 0; j < t.p; j++) {
            Duplicate(out, 0.0f, t.qp);
            Fence();
            for (uint32_t k = 0; k < t.p; k++) {
                float factor = norms.GetValue(k);
                if (factor == 0)
                    continue;
                ReadCol(true, k, v);
                factor *= v.GetValue(j);
                ReadCol(false, k, a);
                Axpy(out, a, factor, t.qp);
                Fence();
            }
            WriteOutput(out, base + j * t.q, t.q);
        }
    } else {
        for (uint32_t j = 0; j < t.q; j++) {
            Duplicate(out, 0.0f, t.pp);
            Fence();
            for (uint32_t k = 0; k < t.p; k++) {
                float factor = norms.GetValue(k);
                if (factor == 0)
                    continue;
                ReadCol(false, k, a);
                factor *= a.GetValue(j);
                ReadCol(true, k, v);
                Axpy(out, v, factor, t.pp);
                Fence();
            }
            WriteOutput(out, base + j * t.p, t.p);
        }
    }
}

template <class T> __aicore__ inline void PinvScalar<T>::Assemble()
{
    float largest = 0;
    for (uint32_t j = 0; j < t.p; j++) {
        float z = Fabs(d.GetValue(j));
        if (z > largest)
            largest = z;
    }
    float threshold = t.atol > largest * t.rtol ? t.atol : largest * t.rtol;
    for (uint32_t j = 0; j < t.p; j++) {
        ReadCol(false, j, a);
        float sq = Dot(a, a, t.qp);
        norms.SetValue(j, (Fabs(d.GetValue(j)) > threshold && sq > 0) ? 1.0f / sq : 0.0f);
    }
    Fence();
    AssembleBlock101();
}

template <class T>
__aicore__ inline void PinvScalar<T>::Init(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws, LinalgPinvTilingData tt)
{
    t = tt;
    input.SetGlobalBuffer((__gm__ T *)ai);
    output.SetGlobalBuffer((__gm__ T *)ao);
    work.SetGlobalBuffer((__gm__ float *)ws);
    pipe.InitBuffer(buf, 194560);
    mem = buf.Get<float>();
    wc = mem;
    vc = mem[t.p * t.qp];
    u = mem[38400];
    v = mem[38912];
    h = mem[39424];
    hv = mem[39936];
    a = mem[40448];
    b = mem[40960];
    tmp = mem[41472];
    tmp2 = mem[41984];
    red = mem[42496];
    d = mem[43008];
    e = mem[43520];
    norms = mem[44032];
    out = mem[44544];
    raw = mem.template ReinterpretCast<T>()[32768 * 4 / sizeof(T)];
    offsets = mem.template ReinterpretCast<uint32_t>()[45568];
    bfOffsets = mem.template ReinterpretCast<uint32_t>()[46080];
}

template <class T> __aicore__ inline void PinvScalar<T>::Run(bool tryFast = true, uint32_t start = ~0U)
{
    auto allwork = work;
    for (batch = start == ~0U ? GetBlockIdx() : start; batch < t.batch; batch += GetBlockNum()) {
        base = batch * t.m * t.n;
        work = allwork[batch * t.p * (t.qp + t.pp)];
        vg = work[t.p * t.qp];
        bool fast = false;
        if (tryFast && t.m == t.n)
            fast = FastInverse<>();
        if (fast)
            continue;
        for (uint32_t i = 0; i < t.q; i++)
            offsets.SetValue(i, i * 32);
        for (uint32_t i = 0; i < 32; i++)
            bfOffsets.SetValue(i, i * 64);
        Fence();
        for (uint32_t j = 0; j < t.p; j++) {
            ReadInput(u, j);
            WriteCol(false, j, u);
            Duplicate(v, 0.0f, t.pp);
            Fence();
            v.SetValue(j, 1.0f);
            Fence();
            WriteCol(true, j, v);
        }
        Bidiagonal();
        DiagonalQR();
        RebuildAndPolish();
        Assemble();
    }
}
