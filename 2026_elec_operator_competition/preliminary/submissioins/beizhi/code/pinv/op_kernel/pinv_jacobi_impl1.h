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

template <class T> __aicore__ inline void PinvJacobi<T>::Fence()
{
    PipeBarrier<PIPE_ALL>();
}

template <class T> __aicore__ inline void PinvJacobi<T>::Sync()
{
    Fence();
    if (t.group * t.groups > 1)
        SyncAll();
    Fence();
}

template <class T> __aicore__ inline float PinvJacobi<T>::AbsScalar(float z)
{
    return z < 0 ? -z : z;
}

template <class T> __aicore__ inline float PinvJacobi<T>::Root(float z)
{
    red.SetValue(0, z > 0 ? z : 0);
    LocalSync<HardEvent::S_V>();
    Sqrt(red, red, 1);
    LocalSync<HardEvent::V_S>();
    return red.GetValue(0);
}

template <class T>
__aicore__ inline float PinvJacobi<T>::Dot(LocalTensor<float> x, LocalTensor<float> y, uint32_t count)
{
    Mul(tmp, x, y, count);
    PipeBarrier<PIPE_V>();
    ReduceSum(red, tmp, tmp2, count);
    LocalSync<HardEvent::V_S>();
    return red.GetValue(0);
}

template <class T> __aicore__ inline void PinvJacobi<T>::Store(GlobalTensor<float> dst, float value)
{
    red.SetValue(0, value);
    LocalSync<HardEvent::S_MTE3>();
    DataCopyPad(dst, red, {1, 4, 0, 0, 0});
    LocalSync<HardEvent::MTE3_S>();
}

template <class T> __aicore__ inline float PinvJacobi<T>::Load(GlobalTensor<float> src)
{
    DataCopy(red, src, 8);
    LocalSync<HardEvent::MTE2_S>();
    return red.GetValue(0);
}

template <class T> __aicore__ inline void PinvJacobi<T>::ReadInput(uint32_t col, LocalTensor<float> dst)
{
    if (inputCached) {
        DataCopy(dst, all[batch * t.p * (t.qp + t.pp) + col * t.qp], t.qp);
        Fence();
        return;
    }
    PinvReadInput(input, raw, offsets, bfOffsets, converted, t, base, col, dst);
}

template <class T>
__aicore__ inline void PinvJacobi<T>::Read(GlobalTensor<float> src, LocalTensor<float> dst, uint32_t len)
{
    DataCopy(dst, src, len);
    Fence();
}

template <class T>
__aicore__ inline void PinvJacobi<T>::Write(GlobalTensor<float> dst, LocalTensor<float> src, uint32_t len)
{
    Fence();
    DataCopy(dst, src, len);
    Fence();
}

template <class T> __aicore__ inline bool PinvJacobi<T>::Pair(uint32_t j, uint32_t k, float tiny, bool precise = false)
{
    Read(work[j * t.qp], u, t.qp);
    Read(work[k * t.qp], v, t.qp);
    float aa = Dot(u, u, t.qp), bb = Dot(v, v, t.qp), ab = precise ? PreciseDot(u, v, t.qp) : Dot(u, v, t.qp);
    if (aa <= 0 || bb <= 0 || aa + bb < tiny)
        return false;
    if (AbsScalar(ab) <= 2e-7f * Root(aa) * Root(bb))
        return false;
    float tau = (bb - aa) / (2 * ab), tt;
    if (AbsScalar(tau) > 1e8f)
        tt = 0.5f / tau;
    else
        tt = (tau < 0 ? -1.0f : 1.0f) / (AbsScalar(tau) + Root(1 + tau * tau));
    float c = 1.0f / Root(1 + tt * tt), s = tt * c;
    Muls(a, u, c, t.qp);
    Muls(tmp, v, s, t.qp);
    PipeBarrier<PIPE_V>();
    Sub(a, a, tmp, t.qp);
    Muls(b, v, c, t.qp);
    Muls(tmp, u, s, t.qp);
    PipeBarrier<PIPE_V>();
    Add(b, b, tmp, t.qp);
    Fence();
    float na = Dot(a, a, t.qp), nb = Dot(b, b, t.qp);
    Write(work[j * t.qp], a, t.qp);
    Write(work[k * t.qp], b, t.qp);
    Read(vg[j * t.pp], vj, t.pp);
    Read(vg[k * t.pp], vk, t.pp);
    Muls(cj, vj, c, t.pp);
    Muls(tmp, vk, s, t.pp);
    PipeBarrier<PIPE_V>();
    Sub(cj, cj, tmp, t.pp);
    Muls(ck, vk, c, t.pp);
    Muls(tmp, vj, s, t.pp);
    PipeBarrier<PIPE_V>();
    Add(ck, ck, tmp, t.pp);
    Fence();
    Write(vg[j * t.pp], cj, t.pp);
    Write(vg[k * t.pp], ck, t.pp);
    Store(ns[j * 8], na);
    Store(ns[k * 8], nb);
    return true;
}

template <class T>
template <typename ExtractType63_0, typename ExtractType63_1, typename ExtractType63_2, typename ExtractType63_3,
          typename ExtractType63_4>
__aicore__ inline void PinvJacobi<T>::PreciseDotBlock63(ExtractType63_0 &sums, ExtractType63_1 &zz,
                                                        ExtractType63_2 &errors, ExtractType63_3 &low,
                                                        ExtractType63_4 &count)
{
    for (uint32_t h = count / 2; h >= 8; h /= 2) {
        Add(sums, tmp, tmp[h], h);
        PipeBarrier<PIPE_V>();
        Sub(zz, sums, tmp, h);
        PipeBarrier<PIPE_V>();
        Sub(errors, tmp[h], zz, h);
        PipeBarrier<PIPE_V>();
        Sub(zz, sums, zz, h);
        PipeBarrier<PIPE_V>();
        Sub(zz, tmp, zz, h);
        PipeBarrier<PIPE_V>();
        Add(errors, errors, zz, h);
        PipeBarrier<PIPE_V>();
        Add(low, low, low[h], h);
        PipeBarrier<PIPE_V>();
        Add(low, low, errors, h);
        Adds(tmp, sums, 0.0f, h);
        PipeBarrier<PIPE_V>();
    }
}

template <class T>
template <typename ExtractType65_0, typename ExtractType65_1, typename ExtractType65_2>
__aicore__ inline void PinvJacobi<T>::PreciseDotBlock65(ExtractType65_0 &low, ExtractType65_1 &sum,
                                                        ExtractType65_2 &err)
{
    for (uint32_t i = 0; i < 8; i++) {
        volatile float a = tmp.GetValue(i), ns = sum + a, z = ns - sum, q = ns - z, e0 = sum - q, e1 = a - z;
        err = err + e0;
        err = err + e1;
        err = err + low.GetValue(i);
        sum = ns;
    }
}

template <class T>
__aicore__ inline float PinvJacobi<T>::PreciseDot(LocalTensor<float> av, LocalTensor<float> xv, uint32_t len)
{
    auto sums = mem[18944], zz = mem[19968], errors = mem[19456], low = mem[17920];
    auto ah = mem[15360], al = mem[15872], bh = mem[16384], bl = mem[16896], pp = mem[18432];
    uint32_t count = 1;
    while (count < len)
        count *= 2;
    Duplicate(tmp, 0.0f, count);
    Duplicate(low, 0.0f, count);
    PipeBarrier<PIPE_V>();
    Mul(tmp, av, xv, len);
    PipeBarrier<PIPE_V>();
    Muls(ah, av, 4097.0f, len);
    Muls(bh, xv, 4097.0f, len);
    PipeBarrier<PIPE_V>();
    Sub(al, ah, av, len);
    Sub(bl, bh, xv, len);
    PipeBarrier<PIPE_V>();
    Sub(ah, ah, al, len);
    Sub(bh, bh, bl, len);
    PipeBarrier<PIPE_V>();
    Sub(al, av, ah, len);
    Sub(bl, xv, bh, len);
    PipeBarrier<PIPE_V>();
    Mul(low, ah, bh, len);
    PipeBarrier<PIPE_V>();
    Sub(low, low, tmp, len);
    PipeBarrier<PIPE_V>();
    Mul(pp, ah, bl, len);
    PipeBarrier<PIPE_V>();
    Add(low, low, pp, len);
    PipeBarrier<PIPE_V>();
    Mul(pp, al, bh, len);
    PipeBarrier<PIPE_V>();
    Add(low, low, pp, len);
    PipeBarrier<PIPE_V>();
    Mul(pp, al, bl, len);
    PipeBarrier<PIPE_V>();
    Add(low, low, pp, len);
    PipeBarrier<PIPE_V>();
    PreciseDotBlock63(sums, zz, errors, low, count);
    LocalSync<HardEvent::V_S>();
    volatile float sum = 0.0f, err = 0.0f;
    PreciseDotBlock65(low, sum, err);
    return sum + err;
}

template <class T>
template <typename ExtractType62_0>
__aicore__ inline void PinvJacobi<T>::RebuildBlock62(ExtractType62_0 &compensation)
{
    for (uint32_t col = part; col < t.p; col += t.group) {
        Read(vg[col * t.pp], vj, t.pp);
        Duplicate(out, 0.0f, t.qp);
        Duplicate(compensation, 0.0f, t.qp);
        Fence();
        for (uint32_t k = 0; k < t.p; k++) {
            ReadInput(k, u);
            Muls(a, u, vj.GetValue(k), t.qp);
            PipeBarrier<PIPE_V>();
            Sub(a, a, compensation, t.qp);
            PipeBarrier<PIPE_V>();
            Add(b, out, a, t.qp);
            PipeBarrier<PIPE_V>();
            Sub(compensation, b, out, t.qp);
            PipeBarrier<PIPE_V>();
            Sub(compensation, compensation, a, t.qp);
            Adds(out, b, 0.0f, t.qp);
            PipeBarrier<PIPE_V>();
        }
        float norm = Dot(out, out, t.qp);
        Write(work[col * t.qp], out, t.qp);
        Store(ns[col * 8], norm);
    }
}

template <class T> __aicore__ inline void PinvJacobi<T>::Rebuild()
{
    auto compensation = mem[7168];
    RebuildBlock62(compensation);
    Sync();
    uint32_t circle = (t.p + 1) / 2 * 2 - 1, pairs = (t.p + 1) / 2;
    float mx = Load(state[t.group * t.groups * 8 + groupid * 8]),
          tiny = 0.01f * (t.atol * t.atol > mx * t.rtol * t.rtol ? t.atol * t.atol : mx * t.rtol * t.rtol);
    for (uint32_t round = 0; round < circle; round++) {
        for (uint32_t pair = part; pair < pairs; pair += t.group) {
            uint32_t j = (round + pair) % circle, k = pair == 0 ? circle : (round + circle - pair) % circle;
            if (j < t.p && k < t.p)
                Pair(j, k, tiny, true);
        }
        Sync();
    }
    if (part == 0) {
        float largest = 0;
        for (uint32_t col = 0; col < t.p; col++) {
            float z = Load(ns[col * 8]);
            if (z > largest)
                largest = z;
        }
        Store(state[t.group * t.groups * 8 + groupid * 8], largest);
    }
    Sync();
}

template <class T> __aicore__ inline void PinvJacobi<T>::OutputRow(uint32_t row, float threshold2)
{
    Duplicate(out, 0.0f, t.qp);
    Fence();
    for (uint32_t k = 0; k < t.p; k++) {
        float norm = Load(ns[k * 8]);
        if (norm == 0.0f || !(norm > threshold2))
            continue;
        Read(work[k * t.qp], u, t.qp);
        Read(vg[k * t.pp], v, t.pp);
        float factor = v.GetValue(row) / norm;
        Axpy(out, u, factor, t.qp);
        Fence();
    }
    Write(y[row * t.qp], out, t.qp);
}

template <class T>
template <typename ExtractType61_0, typename ExtractType61_1>
__aicore__ inline void PinvJacobi<T>::RefineBlock61(ExtractType61_0 &rr, ExtractType61_1 &correction)
{
    for (uint32_t row = part; row < t.p; row += t.group) {
        Read(vg[row * t.pp], rr, t.pp);
        Duplicate(out, 0.0f, t.qp);
        Duplicate(correction, 0.0f, t.qp);
        Fence();
        for (uint32_t col = 0; col < t.p; col++) {
            Read(y[col * t.qp], u, t.qp);
            Muls(a, u, rr.GetValue(col), t.qp);
            PipeBarrier<PIPE_V>();
            Sub(a, a, correction, t.qp);
            PipeBarrier<PIPE_V>();
            Add(b, out, a, t.qp);
            PipeBarrier<PIPE_V>();
            Sub(correction, b, out, t.qp);
            PipeBarrier<PIPE_V>();
            Sub(correction, correction, a, t.qp);
            Adds(out, b, 0.0f, t.qp);
            PipeBarrier<PIPE_V>();
        }
        Read(y[row * t.qp], u, t.qp);
        Add(out, out, u, t.qp);
        Fence();
        Write(work[row * t.qp], out, t.qp);
    }
}

template <class T> __aicore__ inline void PinvJacobi<T>::Refine()
{
    auto rr = mem[6656], correction = mem[7168];
    for (uint32_t row = part; row < t.p; row += t.group) {
        Read(y[row * t.qp], u, t.qp);
        Duplicate(rr, 0.0f, t.pp);
        Fence();
        for (uint32_t col = 0; col < t.p; col++) {
            ReadInput(col, v);
            float z = (col == row ? 1.0f : 0.0f) - PreciseDot(u, v, t.qp);
            rr.SetValue(col, z);
        }
        Fence();
        Write(vg[row * t.pp], rr, t.pp);
    }
    Sync();
    RefineBlock61(rr, correction);
    Sync();
    uint32_t rows = t.m >= t.n ? t.p : t.q;
    for (uint32_t row = part; row < rows; row += t.group) {
        uint32_t count = t.m >= t.n ? t.q : t.p;
        if (t.m >= t.n)
            Read(work[row * t.qp], out, t.qp);
        else {
            auto packed = raw.template ReinterpretCast<float>();
            DataCopyPad(packed, work[row], {(uint16_t)t.p, 4, (t.qp - 1) * 4, 0, 0}, {false, 0, 0, 0});
            Fence();
            Gather(out, packed, offsets, 0, t.p);
            Fence();
        }
        uint32_t at = base + row * count;
        if constexpr (sizeof(T) == 4)
            DataCopyPad(output[at], out, {1, count * 4, 0, 0, 0});
        else {
            Cast(raw, out, RoundMode::CAST_RINT, count);
            Fence();
            DataCopyPad(output[at], raw, {1, count * 2, 0, 0, 0});
        }
        Fence();
    }
}

template <class T>
__aicore__ inline void PinvJacobi<T>::Init(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws, LinalgPinvTilingData tt)
{
    t = tt;
    inputCached = false;
    part = GetBlockIdx() % t.group;
    groupid = GetBlockIdx() / t.group;
    input.SetGlobalBuffer((__gm__ T *)ai);
    output.SetGlobalBuffer((__gm__ T *)ao);
    all.SetGlobalBuffer((__gm__ float *)ws);
    yall = all[t.batch * t.p * (t.qp + t.pp)];
    ns = yall[t.batch * t.p * t.qp];
    state = ns[t.batch * t.p * 8];
    pipe.InitBuffer(buf, 194560);
    mem = buf.Get<float>();
    u = mem;
    v = mem[512];
    a = mem[1024];
    b = mem[1536];
    tmp = mem[2048];
    tmp2 = mem[2560];
    red = mem[3072];
    vj = mem[3584];
    vk = mem[4096];
    cj = mem[4608];
    ck = mem[5120];
    out = mem[5632];
    raw = mem.template ReinterpretCast<T>()[8192 * 4 / sizeof(T)];
    converted = mem[12288];
    offsets = mem.template ReinterpretCast<uint32_t>()[14336];
    bfOffsets = mem.template ReinterpretCast<uint32_t>()[14848];
    for (uint32_t i = 0; i < t.q; i++)
        offsets.SetValue(i, i * 32);
    for (uint32_t i = 0; i < 32; i++)
        bfOffsets.SetValue(i, i * 64);
    Fence();
}
