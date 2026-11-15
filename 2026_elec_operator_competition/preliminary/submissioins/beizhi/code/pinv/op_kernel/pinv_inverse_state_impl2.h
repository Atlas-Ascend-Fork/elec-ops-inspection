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

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::SharedInverseBlock7()
{
    if (t.batch > 1 && t.group >= 3) {
        uint32_t leftCores = t.group * 2 / 3, rightCores = t.group - leftCores;
        uint32_t leftCols = (t.pp + leftCores * 8 - 1) / (leftCores * 8) * 8,
                 rightCols = (t.pp + rightCores * 8 - 1) / (rightCores * 8) * 8;
        if (leftCols * (t.pp + 8) <= 19000 && rightCols * (t.pp + 8) <= 19000 && leftCols <= t.cols) {
            panelCols = leftCols;
            if (part < leftCores) {
                cols = leftCols;
                first = part * cols;
                if (first >= t.pp)
                    first = 2 * t.pp;
            } else {
                cols = rightCols;
                first = t.pp + (part - leftCores) * cols;
            }
        }
    }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::SharedInverseBlock8()
{
    if constexpr (N > 0) {
        SharedInverseBlock7();
    }
}

template <class T, uint32_t N, bool StaticShape>
template <typename ExtractType77_0, typename ExtractType77_1, typename ExtractType77_2, typename ExtractType77_3>
__aicore__ inline void
PinvInverseState<T, N, StaticShape>::SharedInverseBlock9(ExtractType77_0 &failed, ExtractType77_1 &base,
                                                         ExtractType77_2 &ac, ExtractType77_3 &invall)
{
    if (!failed && !outputReady) {
        if constexpr (sizeof(T) == 4) {
            auto outputIndices = mat[8000].template ReinterpretCast<uint32_t>();
            auto columnIndices = mat[12096].template ReinterpretCast<uint32_t>();
            BuildOutputIndices(outputIndices);
            for (uint32_t c = part * 8; c < t.p; c += t.group * 8) {
                uint32_t ac = t.p - c < 8 ? t.p - c : 8;
                DataCopy(mat, invall[c * t.pp], ac * t.pp);
                LocalSync<HardEvent::MTE2_V>();
                Gather(rm, mat, outputIndices, 0, t.pp * 8);
                LocalSync<HardEvent::V_MTE3>();
                DataCopyPad(output[base + c], rm, {(uint16_t)t.p, ac * 4, 0, (t.p - ac) * 4, 0});
                Fence();
            }
        } else {
            WriteBf16Inverse(base, invall);
        }
    }
}

template <class T, uint32_t N, bool StaticShape>
template <typename ExtractType89_0, typename ExtractType89_1, typename ExtractType89_2, typename ExtractType89_3,
          typename ExtractType89_4>
__aicore__ inline void
PinvInverseState<T, N, StaticShape>::SharedInverseBlock10(ExtractType89_0 &needs, ExtractType89_1 &unbounded,
                                                          ExtractType89_2 &accept, ExtractType89_3 &myOk,
                                                          ExtractType89_4 &sensitive)
{
    for (uint32_t g = 0; g < t.groups; g++) {
        uint32_t idx8 = g * 8;
        float aa = red.GetValue(idx8), ii = red.GetValue(idx8 + 1), bb = red.GetValue(idx8 + 2);
        if constexpr (N == 256) {
            if (!(ii * 1.4210854715202004e-14f < 6.4e-9f))
                sensitive = true;
        }
        AssessGroup(aa, ii, bb, g, needs, unbounded, accept, myOk);
    }
}

template <class T, uint32_t N, bool StaticShape>
template <typename ExtractType91_0, typename ExtractType91_1>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::SharedInverseBlock11(ExtractType91_0 &base,
                                                                                 ExtractType91_1 &ac)
{
    if constexpr (sizeof(T) == 4) {
        if (ac)
            DataCopyPad(rm, input[base + first],
                        {(uint16_t)t.p, ac * 4, (t.p - ac) * 4, (cols - (ac + 7) / 8 * 8) / 8, 0}, {false, 0, 0, 0});
    } else
        for (uint32_t i = 0; i < t.p; i++)
            if (ac) {
                DataCopyPad(raw, input[base + i * t.p + first], {1, ac * 2, 0, 0, 0}, {false, 0, 0, 0});
                Fence();
                Cast(rm[i * cols], raw, RoundMode::CAST_NONE, ac);
                Fence();
            }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::SharedInverseBlock12()
{
    for (uint32_t s = 32; s > 0; s >>= 1) {
        if (t.group > s) {
            uint32_t rem = t.group - s < s ? t.group - s : s;
            Add(red, red, red[s * t.groups * 8], rem * t.groups * 8);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::SharedInverseBlock13()
{
    for (uint32_t step = 8; step < t.pp; step *= 2) {
        Adds(idx[step].template ReinterpretCast<int32_t>(), idx.template ReinterpretCast<int32_t>(),
             int32_t(step * cols * 4), step);
        PipeBarrier<PIPE_V>();
    }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::SharedInverseBlock14()
{
    for (uint32_t step = 8; step < cols; step *= 2) {
        Adds(ofs[step].template ReinterpretCast<int32_t>(), ofs.template ReinterpretCast<int32_t>(),
             int32_t(step * ld * 4), step);
        PipeBarrier<PIPE_V>();
    }
}

template <class T, uint32_t N, bool StaticShape>
template <typename ExtractType51_0>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::SharedInverseBlock15(ExtractType51_0 &ac)
{
    for (uint32_t c = 0; c < ac; c += 4) {
        MulAddDst(panel, mat[c * ld], mat[c * ld], t.pp);
        if (c + 1 < ac)
            MulAddDst(panel[256], mat[(c + 1) * ld], mat[(c + 1) * ld], t.pp);
        if (c + 2 < ac)
            MulAddDst(panel[512], mat[(c + 2) * ld], mat[(c + 2) * ld], t.pp);
        if (c + 3 < ac)
            MulAddDst(panel[768], mat[(c + 3) * ld], mat[(c + 3) * ld], t.pp);
        PipeBarrier<PIPE_V>();
    }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::InitShape(LinalgPinvTilingData tt)
{
    t = tt;
    if constexpr (StaticShape && N > 0) {
        t.m = N;
        t.n = N;
        t.p = N;
        t.q = N;
        t.pp = (N + 7) / 8 * 8;
        t.qp = t.pp;
    }
    ld = t.pp + 8;
    part = GetBlockIdx() % t.group;
    groupid = GetBlockIdx() / t.group;
    cols = t.cols;
    panelCols = t.cols;
    first = part * cols;
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::InitGlobalViews(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws)
{
    input.SetGlobalBuffer((__gm__ T *)ai);
    output.SetGlobalBuffer((__gm__ T *)ao);
    exchange.SetGlobalBuffer((__gm__ float *)ws + groupid * 2 * t.cols * (t.pp + 8));
    iall.SetGlobalBuffer((__gm__ float *)ws + t.groups * 2 * t.cols * (t.pp + 8));
    state = iall[t.batch * t.pp * t.pp];
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::InitLocalViews()
{
    rm = mat[20000];
    panel = rm;
    piv = mat[40000];
    tmp = mat[41024];
    idx = mat.template ReinterpretCast<uint32_t>()[42048];
    perm = mat.template ReinterpretCast<uint32_t>()[43072];
    invs = mat[43584];
    broad = mat[44096];
    raw = mat.template ReinterpretCast<T>()[45056 * 4 / sizeof(T)];
    ofs = raw.template ReinterpretCast<uint32_t>();
    red = mat[47872];
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline bool PinvInverseState<T, N, StaticShape>::FinalResult(bool failed)
{
    Store(state[GetBlockIdx() * 8], failed ? 1.0f : 0.0f, 0, 0);
    Sync();
    DataCopy(red, state, t.group * t.groups * 8);
    LocalSync<HardEvent::MTE2_S>();
    float any = 0;
    for (uint32_t k = 0; k < t.group * t.groups; k++) {
        any += red.GetValue(k * 8);
    }
    Sync();
    return any == 0;
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::BuildOutputIndices(LocalTensor<uint32_t> outputIndices)
{
    uint32_t count = t.pp * 8;
    for (uint32_t c = 0; c < 8; c++)
        outputIndices.SetValue(c, c * t.pp * 4);
    LocalSync<HardEvent::S_V>();
    for (uint32_t step = 8; step < count; step *= 2) {
        Adds(outputIndices[step].template ReinterpretCast<int32_t>(), outputIndices.template ReinterpretCast<int32_t>(),
             int32_t(step / 8 * 4), step);
        PipeBarrier<PIPE_V>();
    }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::BuildResidualBlock(
    GlobalTensor<float> inv, uint32_t c0, uint32_t chunk, GlobalTensor<float> savedA, LocalTensor<float> acc,
    LocalTensor<float> acc1, LocalTensor<float> acc2, LocalTensor<float> acc3, LocalTensor<float> coeff,
    LocalTensor<float> cache, LocalTensor<float> br, LocalTensor<uint32_t> outputIndices)
{
    uint32_t ac = t.p - c0 < 8 ? t.p - c0 : 8;
    DataCopy(acc, inv[c0 * t.pp], ac * t.pp);
    LocalSync<HardEvent::MTE2_V>();
    Gather(coeff, acc, outputIndices, 0, t.pp * 8);
    PipeBarrier<PIPE_V>();
    Duplicate(acc, 0.0f, 8192);
    PipeBarrier<PIPE_V>();
    SharedInverseBlock1(chunk, savedA, acc, acc1, acc2, acc3, coeff, cache, br, ac);
    Add(acc, acc, acc1, ac * t.pp);
    Add(acc2, acc2, acc3, ac * t.pp);
    PipeBarrier<PIPE_V>();
    Add(acc, acc, acc2, ac * t.pp);
    PipeBarrier<PIPE_V>();
    Muls(acc, acc, -1.0f, ac * t.pp);
    LocalSync<HardEvent::V_S>();
    for (uint32_t c = 0; c < ac; c++)
        acc.SetValue(c * t.pp + c0 + c, acc.GetValue(c * t.pp + c0 + c) + 1.0f);
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::WriteBf16Inverse(uint32_t base, GlobalTensor<float> invall)
{
    for (uint32_t c = part; c < t.p; c += t.group) {
        DataCopy(tmp, invall[c * t.pp], t.pp);
        Fence();
        Cast(raw, tmp, RoundMode::CAST_RINT, t.p);
        Fence();
        auto bits = raw.template ReinterpretCast<uint16_t>(), packed = rm.template ReinterpretCast<uint16_t>();
        for (uint32_t i = 0; i < t.p; i++)
            packed.SetValue(i * 16, bits.GetValue(i));
        LocalSync<HardEvent::S_MTE3>();
        DataCopyPad(output[base + c], rm.template ReinterpretCast<T>(), {(uint16_t)t.p, 2, 0, (t.p - 1) * 2, 0});
        Fence();
    }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::AssessGroup(float aa, float ii, float bb, uint32_t g,
                                                                        float &needs, float &unbounded, bool &accept,
                                                                        bool &myOk)
{
    float cut2 = t.atol * t.atol > aa * t.rtol * t.rtol ? t.atol * t.atol : aa * t.rtol * t.rtol;
    bool gAccept = bb == 0 && cut2 * ii < 0.25f, gBounded = bb == 0 && cut2 * ii < 25.0f;
    if (!gAccept)
        needs += 1.0f;
    if (!gBounded)
        unbounded += 1.0f;
    if (g == groupid) {
        accept = gAccept || gBounded;
        myOk = (bb == 0);
    }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::UpdateGeneric(uint32_t c0, uint32_t c1)
{
    Brcb(broad, tmp, (c1 - c0 + 7) / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    for (uint32_t r = 0; r < t.pp; r += 64) {
        uint32_t count = t.pp - r < 64 ? t.pp - r : 64;
        MulAddDst(mat[c0 * ld + r], piv[r], broad, count, c1 - c0, {1, 1, 0, (uint8_t)(ld / 8), 0, 1});
    }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::FinishRefinement(GlobalTensor<float> corrected,
                                                                             GlobalTensor<float> inv,
                                                                             LocalTensor<float> out)
{
    if constexpr (sizeof(T) == 4) {
        outputReady = true;
        return;
    }
    for (uint32_t c = part; c < t.p; c += t.group) {
        DataCopy(out, corrected[c * t.pp], t.pp);
        Fence();
        DataCopy(inv[c * t.pp], out, t.pp);
        Fence();
    }
    Sync();
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::CheckSpectralBound(uint32_t batch, float unbounded,
                                                                               bool myOk, bool &accept)
{
    if (unbounded > 0) {
        float ap = PowerNorm(batch, false), ip = PowerNorm(batch, true);
        float spectralCut =
            t.atol * t.atol > ap * t.rtol * t.rtol * 4.0f ? t.atol * t.atol : ap * t.rtol * t.rtol * 4.0f;
        if (myOk && spectralCut * ip * 4.0f < 25.0f)
            accept = true;
    }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::LoadRefinementColumns(uint32_t wp, GlobalTensor<float> inv,
                                                                                  uint32_t cc, uint32_t kc)
{
    if (wp == t.pp) {
        LocalSync<HardEvent::V_MTE2>();
        DataCopy(mat, inv[cc * t.pp], kc * t.pp);
        Fence();
    } else {
        Duplicate(mat, 0.0f, kc * wp);
        LocalSync<HardEvent::V_MTE2>();
        DataCopyPad(mat, inv[cc * t.pp], {(uint16_t)kc, t.pp * 4, 0, (wp - t.pp) / 8, 0}, {false, 0, 0, 0});
        Fence();
    }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::LoadRefinementRow(uint32_t batch, uint32_t r, uint32_t wp,
                                                                              LocalTensor<float> &row,
                                                                              LocalTensor<float> aCache, bool cacheA)
{
    if (cacheA)
        row = aCache[(r - part) / t.group * wp];
    else {
        Duplicate(row, 0.0f, wp);
        Fence();
        PowerRow(batch, r, false, false, row);
    }
}

template <class T, uint32_t N, bool StaticShape>
template <typename ExtractType88_0, typename ExtractType88_1, typename ExtractType88_2, typename ExtractType88_3,
          typename ExtractType88_4>
__aicore__ inline void
PinvInverseState<T, N, StaticShape>::SharedPhase1(ExtractType88_0 &batch, ExtractType88_1 &wp, ExtractType88_2 &aCache,
                                                  ExtractType88_3 &owned, ExtractType88_4 &cacheA)
{
    if (cacheA) {
        if constexpr (sizeof(T) == 4) {
            if (wp > t.pp) {
                Duplicate(aCache, 0.0f, owned * wp);
                LocalSync<HardEvent::V_MTE2>();
            }
            if (owned)
                DataCopyPad(aCache, input[batch * t.p * t.p + part * t.p],
                            {(uint16_t)owned, t.p * 4, (t.group - 1) * t.p * 4, (wp - t.pp) / 8, 0},
                            {true, 0, (uint8_t)(t.pp - t.p), 0});
            Fence();
        } else {
            Duplicate(aCache, 0.0f, owned * wp);
            Fence();
            for (uint32_t k = 0; k < owned; k++)
                PowerRow(batch, part + k * t.group, false, false, aCache[k * wp]);
        }
    }
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline float PinvInverseState<T, N, StaticShape>::PreparePivot(LocalTensor<float> col, uint32_t j,
                                                                          uint32_t pc, uint32_t row, bool &bad)
{
    float value = col.GetValue(row);
    if (!(value > 1e-20f || value < -1e-20f)) {
        bad = true;
        value = 1.0f;
    }
    float inv = 1.0f / value;
    piv = panel[(j - pc) * (t.pp + 8)];
    col.SetValue(row, 0.0f);
    piv.SetValue(t.pp, (float)(int32_t)row);
    piv.SetValue(t.pp + 1, inv);
    idx.SetValue(row, uint32_t(0));
    return inv;
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::RestoreOwnerRows(uint32_t pc, uint32_t cnt)
{
    Adds(mat, panel, 0.0f, cnt * ld);
    LocalSync<HardEvent::V_S>();
    for (uint32_t m = 0; m < cnt; m++) {
        uint32_t r = perm.GetValue(pc + m);
        mat.SetValue(m * ld + r, 1.0f);
    }
    LocalSync<HardEvent::S_V>();
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline void PinvInverseState<T, N, StaticShape>::LoadBatchColumns(uint32_t base, uint32_t ac)
{
    Duplicate(rm, 0.0f, cols * t.pp);
    Fence();
    SharedInverseBlock11(base, ac);
    for (uint32_t i = 0; i < 8; i++)
        idx.SetValue(i, i * cols * 4);
    for (uint32_t c = 0; c < 8; c++)
        ofs.SetValue(c, c * ld * 4);
    LocalSync<HardEvent::S_V>();
    SharedInverseBlock13();
    SharedInverseBlock14();
    Fence();
    for (uint32_t c = 0; c < ac; c++)
        Gather(mat[c * ld], rm, idx, c * 4, t.pp);
}

template <class T, uint32_t N, bool StaticShape>
__aicore__ inline PinvBatchAssessment PinvInverseState<T, N, StaticShape>::AssessBatch(float asq, float isq, bool bad)
{
    Store(state[(part * t.groups + groupid) * 8], asq, isq, bad ? 1.0f : 0.0f);
    Sync();
    DataCopy(red, state, t.group * t.groups * 8);
    LocalSync<HardEvent::MTE2_V>();
    SharedInverseBlock12();
    LocalSync<HardEvent::V_S>();
    PinvBatchAssessment assessment;

    SharedInverseBlock10(assessment.needs, assessment.unbounded, assessment.accept, assessment.myOk,
                         assessment.sensitive);
    if (assessment.sensitive) {
        assessment.accept = false;
        assessment.needs = 0;
    }
    return assessment;
}
