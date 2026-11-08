/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 黎何.
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

#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "linalg_lstsq_tiling.h"
#include "tiling_key_linalg_lstsq.h"
using namespace AscendC;

template <HardEvent E> __aicore__ inline void Fence()
{
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}

__aicore__ inline __attribute__((always_inline)) float ReadPivot(LocalTensor<float> col, uint32_t& row)
{
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    uint32_t maxBits = 0;

    GetReduceRepeatMaxMinSpr<float>(maxBits, row);
    float d = col.GetValue(row);
    col.SetValue(row, 0.0f);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    return d;
}

template <uint32_t N, uint32_t K, uint32_t PANEL, bool Wide> class BlockLu {
    static constexpr uint32_t P = Wide ? 8U : 10U;
    static constexpr uint32_t C = N / P;
    static constexpr uint32_t CP = (C + 7U) / 8U * 8U;
    static constexpr uint32_t MP = Wide ? 264U : N;
    static constexpr uint32_t META = (PANEL + 7) / 8 * 8;
    static constexpr uint32_t LU_BASE = 32768;
    static constexpr uint32_t WS = LU_BASE + N * MP;
    static_assert(N % P == 0U && C % PANEL == 0U, "panel must divide each core column partition");
    TPipe pipe;
    TBuf<TPosition::VECCALC> matBuf;
    TBuf<TPosition::VECCALC> origBuf;
    TBuf<TPosition::VECCALC> rawBuf;
    TBuf<TPosition::VECCALC> auxBuf;
    TBuf<TPosition::VECCALC> panelBuf;
    LocalTensor<float> mat;
    LocalTensor<float> orig;
    LocalTensor<float> raw;
    LocalTensor<float> aux;
    LocalTensor<float> panel;
    LocalTensor<float> v;
    LocalTensor<float> active;
    LocalTensor<float> tmp;
    LocalTensor<float> red;
    LocalTensor<float> meta;
    LocalTensor<float> rhs;
    LocalTensor<float> bOrig;
    LocalTensor<float> sol;
    LocalTensor<float> corr;
    LocalTensor<float> sums;
    LocalTensor<float> errs;
    LocalTensor<float> prod;
    LocalTensor<float> prodErr;
    LocalTensor<float> ah;
    LocalTensor<float> al;
    LocalTensor<float> wh;
    LocalTensor<float> wl;
    LocalTensor<float> scratch;
    LocalTensor<float> allSol;
    LocalTensor<float> factors;
    LocalTensor<uint32_t> indices;
    LocalTensor<uint32_t> piv;
    LocalTensor<uint32_t> cols;
    GlobalTensor<float> a;
    GlobalTensor<float> b;
    GlobalTensor<float> x;
    GlobalTensor<float> res;
    GlobalTensor<float> w;
    GlobalTensor<float> lu;
    uint32_t lane;
    uint32_t batch;

    __aicore__ inline void Barrier()
    {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }

    template <class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void LoadColumnsTile0(A0& shift, A1& ri, A2& ci, A3& denseRows)
    {

        Muls(ci, ri, static_cast<int32_t>(MP * 4), Wide ? 40U : 24U);
        Muls(denseRows, ri, static_cast<int32_t>((Wide ? C + 8 : CP) * 4), N);
        if (lane < K)
            Muls(ri, ri, static_cast<int32_t>(K * 4), N);
        Duplicate(active.template ReinterpretCast<int32_t>(), 2147483647, N);
        if constexpr (!Wide) {
            Duplicate(active[N].template ReinterpretCast<int32_t>(), 0, 256 - N);
        }
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::MTE2_V>();
        for (uint32_t c = 0; c < C; c++)
            Gather(mat[c * MP], orig, denseRows.template ReinterpretCast<uint32_t>(), (c + shift) * 4, N);
        if (lane < K)
            Gather(rhs, raw, indices, lane * 4, N);
        PipeBarrier<PIPE_V>();
        if constexpr (Wide) {
            DataCopy(orig, mat, DataCopyParams{C, N / 8, (MP - N) / 8, 0});
        } else {
            DataCopy(orig, mat, C * N);
        }
        if (lane < K)
            Adds(bOrig, rhs, 0.0f, N);
        Brcb(raw, cols.template ReinterpretCast<float>(), Wide ? 5U : 3U, BrcbRepeatParams{1, 8});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadColumns()
    {
        const uint32_t shift = Wide ? 0U : (lane & 1U) * 4U;
        if constexpr (Wide) {
            DataCopy(orig, a[batch * N * N + lane * C], DataCopyParams{N, C / 8, (N - C) / 8, 1});
        } else {
            DataCopy(orig, a[batch * N * N + lane * C - shift], DataCopyParams{N, CP / 8, (N - CP) / 8, 0});
        }
        if (lane < K)
            DataCopy(raw, b[batch * N * K], N * K);
        if constexpr (Wide) {
            Duplicate(indices.ReinterpretCast<int32_t>(), 0, N + 8);
        }
        auto ri = indices.ReinterpretCast<int32_t>();
        auto ci = cols.ReinterpretCast<int32_t>();
#pragma unroll
        for (uint32_t i = 0; i < 16; i++)
            ri.SetValue(i, static_cast<int32_t>(i));
        Fence<HardEvent::S_V>();
        Adds(ri[16], ri, 16, 16);
        Adds(ri[32], ri, 32, 16);
        Adds(ri[48], ri, 48, 16);
        PipeBarrier<PIPE_V>();
        Adds(ri[64], ri, 64, 64);
        Adds(ri[128], ri, 128, 64);
        Adds(ri[192], ri, 192, Wide ? 64U : 8U);
        PipeBarrier<PIPE_V>();
        auto denseRows = red.ReinterpretCast<int32_t>();
        LoadColumnsTile0(shift, ri, ci, denseRows);
    }

    __aicore__ inline void UpdatePanel(LocalTensor<float> base,
                                       uint32_t count,
                                       const uint32_t* r,
                                       uint32_t j0,
                                       LocalTensor<float> inverse,
                                       bool needActive)
    {
        for (uint32_t q = 0; q < PANEL; q++) {
            auto mult = panel[q * MP];
            Gather(factors[64],
                   base,
                   raw.ReinterpretCast<uint32_t>(),
                   r[q] * 4,
                   64,
                   static_cast<uint8_t>((count + 7U) >> 3U),
                   8);
            PipeBarrier<PIPE_V>();
            if (q == 0U)
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
#pragma unroll
            for (uint32_t off = 0; off < N; off += 64) {
                const uint32_t mask = Wide ? 64U : ((off + 64 <= N) ? 64U : N - off);
                MulAddDst(base[off],
                          mult[off],
                          factors[64],
                          mask,
                          static_cast<uint8_t>(count),
                          BinaryRepeatParams{1, 1, 0, MP / 8, 0, 1});
            }
            PipeBarrier<PIPE_V>();
            if (lane < K) {
                piv.SetValue(j0 + q, r[q]);
                v.SetValue(j0 + q, inverse.GetValue(q));
            }
            if (needActive)
                active.SetValue(r[q], 0.0f);
        }
    }

    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
    __aicore__ inline __attribute__((always_inline)) void
    FactorTile1(A0& rows, A1& inverse, A2& r, A3& j0, A4& slot, A5& begin, A6& count, A7& needActive)
    {

        DataCopy(rows.template ReinterpretCast<float>(), w[slot], 2 * META);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        if (count > 0U) {
            DataCopy(panel, lu[j0 * MP], PANEL * MP);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
#pragma unroll
        for (uint32_t q = 0; q < PANEL; q++)
            r[q] = rows.GetValue(q);
        if (count > 0U) {
            UpdatePanel(mat[begin * MP], count, r, j0, inverse, needActive);
        } else {
            if (lane < K) {
#pragma unroll
                for (uint32_t q = 0; q < PANEL; q++) {
                    piv.SetValue(j0 + q, r[q]);
                    v.SetValue(j0 + q, inverse.GetValue(q));
                }
            }
            if (needActive) {
#pragma unroll
                for (uint32_t q = 0; q < PANEL; q++)
                    active.SetValue(r[q], 0.0f);
            }
        }
    }

    __aicore__ inline __attribute__((always_inline)) void
    UpdateLocalColumns(uint32_t localStart, uint32_t q, LocalTensor<float> mult, uint32_t rem)
    {
#pragma unroll
        for (uint32_t off = 0; off < N; off += 64) {
            const uint32_t mask = Wide ? 64U : ((off + 64 <= N) ? 64U : N - off);
            MulAddDst(mat[(localStart + q + 1) * MP + off],
                      mult[off],
                      factors[64],
                      mask,
                      static_cast<uint8_t>(rem),
                      BinaryRepeatParams{1, 1, 0, MP / 8, 0, 1});
        }
        PipeBarrier<PIPE_V>();
    }

    template <class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void FactorTile2(A0& localStart, A1& q, A2& mult, A3& rem)
    {

        UpdateLocalColumns(localStart, q, mult, rem);
        And(tmp.template ReinterpretCast<uint16_t>(),
            mat[(localStart + q + 1) * MP].template ReinterpretCast<uint16_t>(),
            active.template ReinterpretCast<uint16_t>(),
            512);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(meta, tmp, 64, 4, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
    }

    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
    __aicore__ inline __attribute__((always_inline)) void
    FactorTile3(A0& rows, A1& j0, A2& localStart, A3& slot, A4& q, A5& col, A6& mult, A7& rem, A8& row, A9& inv)
    {

        if (lane < K) {
            piv.SetValue(j0 + q, row);
            v.SetValue(j0 + q, inv);
        }
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(mult, col, -inv, 64, 4, UnaryRepeatParams{1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        if (q + 1U < PANEL) {
            FactorTile2(localStart, q, mult, rem);
        } else {
            Fence<HardEvent::S_MTE3>();
            DataCopy(w[slot], rows.template ReinterpretCast<float>(), 2 * META);
            Fence<HardEvent::V_MTE3>();
            DataCopy(lu[j0 * MP], panel, PANEL * MP);
            if (rem) {
                UpdateLocalColumns(localStart, q, mult, rem);
            }
        }
    }

    template <class A0, class A1, class A2, class A3, class A4, class A5>
    __aicore__ inline __attribute__((always_inline)) void
    FactorTile4(A0& rows, A1& inverse, A2& r, A3& j0, A4& localStart, A5& slot)
    {

        PipeBarrier<PIPE_V>();
        WholeReduceMax(meta, tmp, 64, 4, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t q = 0; q < PANEL; q++) {
            auto col = mat[(localStart + q) * MP];
            auto mult = panel[q * MP];
            uint32_t rem = C + (lane < K ? 1U : 0U) - (localStart + q + 1U);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            uint32_t maxBits = 0;
            uint32_t row = 0;
            GetReduceRepeatMaxMinSpr<float>(maxBits, row);
            float d = col.GetValue(row);
            col.SetValue(row, 0.0f);
            active.SetValue(row, 0.0f);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            if (rem)
                Gather(factors[64],
                       mat[(localStart + q + 1) * MP],
                       raw.template ReinterpretCast<uint32_t>(),
                       row * 4,
                       64,
                       static_cast<uint8_t>((rem + 7U) >> 3U),
                       8);
            ASSERT(d != 0.0f);
            float inv = 1.0f / d;
            r[q] = row;
            rows.SetValue(q, row);
            inverse.SetValue(q, inv);
            FactorTile3(rows, j0, localStart, slot, q, col, mult, rem, row, inv);
        }
    }

    __aicore__ inline void Factor()
    {
        auto rows = panel[PANEL * MP + (Wide ? 0U : 64U)].ReinterpretCast<uint32_t>();
        auto inverse = panel[PANEL * MP + (Wide ? 0U : 64U) + META];
        uint32_t r[PANEL];
        for (uint32_t j0 = 0; j0 < N; j0 += PANEL) {
            uint32_t owner = j0 / C;
            uint32_t localStart = j0 % C;
            uint32_t slot = ((j0 / PANEL) & 1U) * (2 * META);
            if (lane == owner) {
                if (j0 > 0U)
                    Fence<HardEvent::S_V>();
                And(tmp.ReinterpretCast<uint16_t>(),
                    mat[localStart * MP].ReinterpretCast<uint16_t>(),
                    active.ReinterpretCast<uint16_t>(),
                    512);
                FactorTile4(rows, inverse, r, j0, localStart, slot);
            }
            Barrier();
            uint32_t end = j0 + PANEL;
            uint32_t begin = end <= lane * C ? 0U : (end >= lane * C + C ? C : end - lane * C);
            const uint32_t count = C - begin + (lane < K ? 1U : 0U);
            const bool needActive = (lane > owner);
            if (lane != owner && (count > 0U || needActive)) {
                FactorTile1(rows, inverse, r, j0, slot, begin, count, needActive);
            }
        }
        if (lane < K) {
            Fence<HardEvent::S_V>();
            Muls(indices.ReinterpretCast<int32_t>(), piv.ReinterpretCast<int32_t>(), 4, N);
            Duplicate(indices[N].ReinterpretCast<int32_t>(), 0, 8);
            PipeBarrier<PIPE_V>();
            Gather(tmp, rhs, indices, 0, N);
            PipeBarrier<PIPE_V>();
            Mul(sol, tmp, v, N);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Correct()
    {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Fence<HardEvent::V_MTE2>();
        DataCopy(mat, lu[C * MP], C * MP);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        for (uint32_t step = 0; step < P; step++) {
            uint32_t j0 = step * C;
            event_t ev = (step & 1U) ? EVENT_ID1 : EVENT_ID0;
            auto curBuf = (step & 1U) ? mat : orig;
            if (step > 0U)
                WaitFlag<HardEvent::MTE2_V>(ev);
            for (uint32_t q = 0; q < C; q++) {
                Gather(factors, rhs, raw[(j0 + q) * 8].ReinterpretCast<uint32_t>(), 0, 8, 1, 8);
                PipeBarrier<PIPE_V>();
                MulAddDst(rhs,
                          curBuf[q * MP],
                          factors,
                          64,
                          Wide ? N / 64 : (N + 63) / 64,
                          BinaryRepeatParams{1, 1, 0, 8, 8, 0});
                PipeBarrier<PIPE_V>();
            }
            if (step + 2U < P) {
                SetFlag<HardEvent::V_MTE2>(ev);
                WaitFlag<HardEvent::V_MTE2>(ev);
                DataCopy(curBuf, lu[(step + 2U) * C * MP], C * MP);
                SetFlag<HardEvent::MTE2_V>(ev);
            }
        }
        Gather(tmp, rhs, indices, 0, N);
        PipeBarrier<PIPE_V>();
        Mul(corr, tmp, v, N);
        PipeBarrier<PIPE_V>();
    }

    template <class A0, class A1, class A2, class A3, class A4, class A5>
    __aicore__ inline __attribute__((always_inline)) void
    ResidualWideTile5(A0& eA, A1& pA, A2& vhA, A3& c, A4& sIn, A5& sOut)
    {

#pragma unroll
        for (uint32_t off = 0; off < N; off += 64) {
            Mul(pA[off], orig[c * N + off], raw[c * 64], 64, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            Mul(pA[K * N + off],
                orig[(c + 1U) * N + off],
                raw[(c + 1U) * 64],
                64,
                K,
                BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            Mul(vhA[2 * K * N + off],
                orig[(c + 2U) * N + off],
                raw[(c + 2U) * 64],
                64,
                K,
                BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            Mul(vhA[3 * K * N + off],
                orig[(c + 3U) * N + off],
                raw[(c + 3U) * 64],
                64,
                K,
                BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
        }
        PipeBarrier<PIPE_V>();
        Add(sOut, sIn, pA, 2 * K * N);
        Add(sOut[2 * K * N], sIn[2 * K * N], vhA[2 * K * N], 2 * K * N);
        PipeBarrier<PIPE_V>();
        Sub(vhA, sOut, sIn, 2 * K * N);
        PipeBarrier<PIPE_V>();
        Sub(pA, pA, vhA, 2 * K * N);
        Sub(vhA, sOut[2 * K * N], sIn[2 * K * N], 2 * K * N);
        PipeBarrier<PIPE_V>();
        Add(eA, eA, pA, 2 * K * N);
        Sub(vhA[2 * K * N], vhA[2 * K * N], vhA, 2 * K * N);
        PipeBarrier<PIPE_V>();
    }

    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6>
    __aicore__ inline __attribute__((always_inline)) void
    ResidualWideTile6(A0& sA, A1& sB, A2& eA, A3& pA, A4& vhA, A5& coeffHi, A6& coeffLo)
    {

        Brcb(raw[2048], coeffHi, 2, BrcbRepeatParams{1, 8});
        PipeBarrier<PIPE_V>();
        Brcb(raw[2176], coeffLo, 2, BrcbRepeatParams{1, 8});
#pragma unroll
        for (uint32_t off = 0; off < N; off += 64) {
            Mul(sB[off], orig[off], raw, 64, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            Mul(sB[K * N + off], orig[N + off], raw[64], 64, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            Mul(sB[2 * K * N + off], orig[2 * N + off], raw[128], 64, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            Mul(sB[3 * K * N + off], orig[3 * N + off], raw[192], 64, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            Mul(eA[off], ah[off], raw[2048], 64, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            Mul(eA[K * N + off], wh[off], raw[2112], 64, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
        }
        Duplicate(eA[2 * K * N], 0.0f, K * N);
        PipeBarrier<PIPE_V>();
        Sub(eA, eA, sB, 2 * K * N);
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t off = 0; off < N; off += 64) {
            MulAddDst(eA[off], ah[off], raw[2176], 64, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            MulAddDst(eA[K * N + off], wh[off], raw[2240], 64, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
        }
        PipeBarrier<PIPE_V>();
#pragma unroll
        for (uint32_t off = 0; off < N; off += 64) {
            MulAddDst(eA[off], al[off], raw, 64, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            MulAddDst(eA[K * N + off], scratch[off], raw[64], 64, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t q = 1U; q < C / 4U; q++) {
            uint32_t c = q * 4U;
            auto sIn = (q & 1U) ? sB : sA;
            auto sOut = (q & 1U) ? sA : sB;
            ResidualWideTile5(eA, pA, vhA, c, sIn, sOut);
            Add(eA, eA, vhA[2 * K * N], 2 * K * N);
            PipeBarrier<PIPE_V>();
        }
    }

    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
    __aicore__ inline __attribute__((always_inline)) void
    ResidualWideTile7(A0& sA, A1& sB, A2& eA, A3& pA, A4& vhA, A5& coeff, A6& coeffHi, A7& coeffLo)
    {

        for (uint32_t c = 0; c < C; c++) {
#pragma unroll
            for (uint32_t k = 0; k < K; k++)
                coeff.SetValue(c * 8 + k, -allSol.GetValue(k * N + lane * C + c));
        }
        Fence<HardEvent::S_V>();
        And(coeffHi.template ReinterpretCast<uint16_t>(),
            coeff.template ReinterpretCast<uint16_t>(),
            wl.template ReinterpretCast<uint16_t>(),
            32);
        Brcb(raw, coeff, C, BrcbRepeatParams{1, 8});
        Sub(al, orig, ah, N);
        Sub(scratch, orig[N], wh, N);
        PipeBarrier<PIPE_V>();
        Sub(coeffLo, coeff, coeffHi, 16);
        ResidualWideTile6(sA, sB, eA, pA, vhA, coeffHi, coeffLo);
        Add(sB, sA, sA[2 * K * N], 2 * K * N);
        PipeBarrier<PIPE_V>();
        Sub(vhA, sB, sA, 2 * K * N);
        PipeBarrier<PIPE_V>();
        Sub(pA, sA[2 * K * N], vhA, 2 * K * N);
        PipeBarrier<PIPE_V>();
        Add(eA, eA, pA, 2 * K * N);
        Add(sA, sB, sB[K * N], K * N);
        PipeBarrier<PIPE_V>();
        Add(eA, eA, eA[K * N], K * N);
        Sub(vhA, sA, sB, K * N);
        PipeBarrier<PIPE_V>();
        Sub(pA, sB[K * N], vhA, K * N);
        Adds(sums, sA, 0.0f, K * N);
        PipeBarrier<PIPE_V>();
        Add(errs, eA, pA, K * N);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>();
        DataCopy(w[2048 + lane * 2 * K * N], sums, 2 * K * N);
        Barrier();
    }

    template <class A0, class A1>
    __aicore__ inline __attribute__((always_inline)) void ResidualWideTile8(A0& wArr, A1& aArr)
    {

        Sub(wArr, sums[N], sums, P * N);
        PipeBarrier<PIPE_V>();
        Sub(aArr, sums[N], wArr, P * N);
        Sub(mat, mat, wArr, P * N);
        PipeBarrier<PIPE_V>();
        Sub(aArr, sums, aArr, P * N);
        Add(mat[P * N], mat[P * N], mat, P * N);
        PipeBarrier<PIPE_V>();
        Add(mat[P * N], mat[P * N], aArr, P * N);
        PipeBarrier<PIPE_V>();
        Add(mat[P * N], mat[P * N], mat[(P + 4) * N], 4 * N);
        PipeBarrier<PIPE_V>();
        Add(mat[P * N], mat[P * N], mat[(P + 2) * N], 2 * N);
        PipeBarrier<PIPE_V>();
        Add(mat[P * N], mat[P * N], mat[(P + 1) * N], N);
        PipeBarrier<PIPE_V>();
        Add(rhs, sums[P * N], mat[P * N], N);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ResidualWide()
    {
        if (lane < K) {
            Fence<HardEvent::V_MTE3>();
            DataCopy(w[1024 + lane * N], sol, N);
        }
        Duplicate(wl.ReinterpretCast<int32_t>(), -4096, N);
        PipeBarrier<PIPE_V>();
        And(ah.ReinterpretCast<uint16_t>(), orig.ReinterpretCast<uint16_t>(), wl.ReinterpretCast<uint16_t>(), 2 * N);
        And(wh.ReinterpretCast<uint16_t>(), orig[N].ReinterpretCast<uint16_t>(), wl.ReinterpretCast<uint16_t>(), 2 * N);
        Barrier();
        DataCopy(allSol, w[1024], K * N);
        Fence<HardEvent::MTE2_S>();
        auto sA = mat;
        auto sB = mat[4 * K * N];
        auto eA = mat[8 * K * N];
        auto pA = panel;
        auto vhA = sums;
        auto coeff = factors;
        auto coeffHi = factors[256];
        auto coeffLo = factors[320];
        ResidualWideTile7(sA, sB, eA, pA, vhA, coeff, coeffHi, coeffLo);
        if (lane < K) {
            Fence<HardEvent::V_MTE2>();
            DataCopy(mat, w[2048 + lane * N], DataCopyParams{P, N / 8, (2 * K - 1) * N / 8, 0});
            DataCopy(mat[P * N], w[2048 + K * N + lane * N], DataCopyParams{P, N / 8, (2 * K - 1) * N / 8, 0});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            DataCopy(orig, lu, C * MP);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Adds(sums, bOrig, 0.0f, N);
            Brcb(raw, indices.ReinterpretCast<float>(), N / 8 + 1, BrcbRepeatParams{1, 8});
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
#pragma unroll
            for (uint32_t p = 0; p < P; p++) {
                Add(sums[(p + 1) * N], sums[p * N], mat[p * N], N);
                PipeBarrier<PIPE_V>();
            }
            auto wArr = mat[2 * P * N];
            auto aArr = mat[3 * P * N];
            ResidualWideTile8(wArr, aArr);
        }
    }

    template <class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void ResidualNarrowTile9(A0& s0, A1& s1, A2& s2, A3& s3)
    {

#pragma unroll
        for (uint32_t off = 0; off < N; off += 64) {
            uint32_t mask = (off + 64 <= N) ? 64 : N - off;
            Mul(s0[off], orig[off], raw, mask, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            Mul(s1[off], orig[N + off], raw[64], mask, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            Mul(s2[off], orig[2 * N + off], raw[128], mask, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            Mul(s3[off], orig[3 * N + off], raw[192], mask, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t c = 4U; c < C; c += 4U) {
#pragma unroll
            for (uint32_t off = 0; off < N; off += 64) {
                uint32_t mask = (off + 64 <= N) ? 64 : N - off;
                MulAddDst(s0[off], orig[c * N + off], raw[c * 64], mask, K, BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
                MulAddDst(s1[off],
                          orig[(c + 1U) * N + off],
                          raw[(c + 1U) * 64],
                          mask,
                          K,
                          BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
                MulAddDst(s2[off],
                          orig[(c + 2U) * N + off],
                          raw[(c + 2U) * 64],
                          mask,
                          K,
                          BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
                MulAddDst(s3[off],
                          orig[(c + 3U) * N + off],
                          raw[(c + 3U) * 64],
                          mask,
                          K,
                          BinaryRepeatParams{1, 1, 0, N / 8, 0, 1});
            }
            PipeBarrier<PIPE_V>();
        }
    }

    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6>
    __aicore__ inline __attribute__((always_inline)) void
    ResidualNarrowTile10(A0& s0, A1& s1, A2& s2, A3& s3, A4& u0, A5& u1, A6& coeff)
    {

        Fence<HardEvent::S_V>();
        Brcb(raw, coeff, C, BrcbRepeatParams{1, 8});
        PipeBarrier<PIPE_V>();
        ResidualNarrowTile9(s0, s1, s2, s3);
        Add(u0, s0, s1, K * N);
        Add(u1, s2, s3, K * N);
        PipeBarrier<PIPE_V>();
        Add(sums, u0, u1, K * N);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>();
        DataCopy(w[2048 + lane * K * N], sums, K * N);
        Barrier();
        if (lane < K) {
            Fence<HardEvent::V_MTE2>();
            DataCopy(mat, w[2048 + lane * N], DataCopyParams{P, N / 8, (K - 1) * N / 8, 0});
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            DataCopy(orig, lu, C * MP);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            Brcb(raw, indices.template ReinterpretCast<float>(), N / 8 + 1, BrcbRepeatParams{1, 8});
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Add(mat, mat, mat[8 * N], 2 * N);
            PipeBarrier<PIPE_V>();
            Add(mat, mat, mat[4 * N], 4 * N);
            PipeBarrier<PIPE_V>();
            Add(mat, mat, mat[2 * N], 2 * N);
            PipeBarrier<PIPE_V>();
            Add(mat, mat, mat[N], N);
            PipeBarrier<PIPE_V>();
            Add(rhs, bOrig, mat, N);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ResidualNarrow()
    {
        if (lane < K) {
            Fence<HardEvent::V_MTE3>();
            DataCopy(w[1024 + lane * N], sol, N);
        }
        Duplicate(errs, 0.0f, K * N);
        Barrier();
        DataCopy(allSol, w[1024], K * N);
        Fence<HardEvent::MTE2_S>();
        auto s0 = mat;
        auto s1 = mat[K * N];
        auto s2 = mat[2 * K * N];
        auto s3 = mat[3 * K * N];
        auto u0 = mat[4 * K * N];
        auto u1 = mat[5 * K * N];
        auto coeff = prod;
        for (uint32_t c = 0; c < C; c++) {
#pragma unroll
            for (uint32_t k = 0; k < K; k++)
                coeff.SetValue(c * 8 + k, -allSol.GetValue(k * N + lane * C + c));
        }
        ResidualNarrowTile10(s0, s1, s2, s3, u0, u1, coeff);
    }

    __aicore__ inline void Residual()
    {
        if constexpr (Wide) {
            ResidualWide();
        } else {
            ResidualNarrow();
        }
    }

public:
    template <class A0, class A1, class A2, class A3, class A4>
    __aicore__ inline __attribute__((always_inline)) void RunTile11(A0& aa, A1& bb, A2& xx, A3& rr, A4& ww)
    {

        lane = GetBlockIdx() % P;
        batch = GetBlockIdx() / P;
        a.SetGlobalBuffer((__gm__ float*)aa);
        b.SetGlobalBuffer((__gm__ float*)bb);
        x.SetGlobalBuffer((__gm__ float*)xx);
        res.SetGlobalBuffer((__gm__ float*)rr);
        w.SetGlobalBuffer((__gm__ float*)ww + batch * WS);
        lu.SetGlobalBuffer((__gm__ float*)ww + batch * WS + LU_BASE);
        pipe.InitBuffer(matBuf, Wide ? (C + 1) * MP * 4 : 4 * P * N * 4);
        pipe.InitBuffer(origBuf, N * (Wide ? C + 8 : CP) * 4);
        pipe.InitBuffer(rawBuf, 4096 * 4);
        pipe.InitBuffer(auxBuf, (Wide ? 7936U : 7808U) * 4);
        pipe.InitBuffer(panelBuf, (PANEL * MP + (Wide ? 0U : 64U) + 2 * META) * 4);
        mat = matBuf.template Get<float>();
        orig = origBuf.template Get<float>();
        raw = rawBuf.template Get<float>();
        aux = auxBuf.template Get<float>();
        panel = panelBuf.template Get<float>();
        v = aux;
        active = aux[Wide ? N : 2 * N];
        tmp = aux[Wide ? 2 * N : 656U];
        red = aux[Wide ? 3 * N : N];
        meta = aux[Wide ? 4 * N : 912U];
        rhs = aux[5 * N];
        bOrig = aux[6 * N];
        sol = aux[7 * N];
        corr = aux[8 * N];
        indices = aux[9 * N].template ReinterpretCast<uint32_t>();
        piv = aux[10 * N].template ReinterpretCast<uint32_t>();
        cols = aux[11 * N].template ReinterpretCast<uint32_t>();
        sums = aux[12 * N];
        errs = aux[(12 + K) * N];
        prod = aux[(12 + 2 * K) * N];
    }

    __aicore__ inline void Run(GM_ADDR aa, GM_ADDR bb, GM_ADDR xx, GM_ADDR rr, GM_ADDR ww)
    {
        RunTile11(aa, bb, xx, rr, ww);
        prodErr = aux[(13 + 2 * K) * N];
        ah = aux[(14 + 2 * K) * N];
        al = aux[(15 + 2 * K) * N];
        wh = aux[(16 + 2 * K) * N];
        wl = aux[(17 + 2 * K) * N];
        scratch = aux[(18 + 2 * K) * N];
        allSol = aux[(19 + 2 * K) * N];
        factors = aux[(19 + 3 * K) * N];
        rhs = mat[C * MP];
        LoadColumns();
        Factor();
        rhs = aux[5 * N];
        Residual();
        if (lane < K) {
            Correct();
            Add(sol, sol, corr, N);
            PipeBarrier<PIPE_V>();
            Brcb(raw, sol, N / 8, BrcbRepeatParams{1, 8});
            if (rr != nullptr)
                Duplicate(meta, 0.0f, 8);
            PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(x[batch * N * K + lane], raw, DataCopyExtParams{N, 4, 0, (K - 1) * 4, 0});
            if (rr != nullptr)
                DataCopyPad(res[batch * K + lane], meta, DataCopyExtParams{1, 4, 0, 0, 0});
        }
    }
};

template <uint32_t N, uint32_t K, uint32_t PANEL> using BlockLu256 = BlockLu<N, K, PANEL, true>;
template <uint32_t N, uint32_t K, uint32_t PANEL> using BlockLu200 = BlockLu<N, K, PANEL, false>;

template <uint32_t N, uint32_t MP, uint32_t K> class SmallLu {
    static constexpr uint32_t S = 72U;
    TPipe pipe;
    TBuf<TPosition::VECCALC> matBuf;
    TBuf<TPosition::VECCALC> permBuf;
    TBuf<TPosition::VECCALC> rawBuf;
    TBuf<TPosition::VECCALC> auxBuf;
    LocalTensor<float> mat;
    LocalTensor<float> perm;
    LocalTensor<float> raw;
    LocalTensor<float> v0;
    LocalTensor<float> tmp;
    LocalTensor<float> meta;
    LocalTensor<float> factors;
    LocalTensor<float> diagInv;
    LocalTensor<uint32_t> rows;
    LocalTensor<uint32_t> rowsA;
    LocalTensor<uint32_t> columns;
    LocalTensor<uint32_t> rowsB;
    LocalTensor<uint32_t> piv;
    LocalTensor<uint32_t> bcastCols;
    GlobalTensor<float> a;
    GlobalTensor<float> b;
    GlobalTensor<float> x;
    GlobalTensor<float> res;

public:
    __aicore__ inline __attribute__((always_inline)) void RunTile12()
    {

        if constexpr (N < 64U) {
            Duplicate(rowsB[48].template ReinterpretCast<int32_t>(), 0, 16);
            Duplicate(rowsA[48].template ReinterpretCast<int32_t>(), 0, 16);
            Duplicate(piv[96].template ReinterpretCast<int32_t>(), 0, 32);
        }
        Adds(rowsB[16].template ReinterpretCast<int32_t>(), rowsB.template ReinterpretCast<int32_t>(), 128, 16);
        Adds(rowsB[32].template ReinterpretCast<int32_t>(), rowsB.template ReinterpretCast<int32_t>(), 256, 16);
        if constexpr (N == 64U)
            Adds(rowsB[48].template ReinterpretCast<int32_t>(), rowsB.template ReinterpretCast<int32_t>(), 384, 16);
        else
            Adds(rowsB[48].template ReinterpretCast<int32_t>(),
                 rowsB.template ReinterpretCast<int32_t>(),
                 384,
                 N - 48U);
        if constexpr (N < 64U) {
            Adds(rowsA[16].template ReinterpretCast<int32_t>(),
                 rowsA.template ReinterpretCast<int32_t>(),
                 static_cast<int32_t>(16U * N * 4U),
                 16);
            Adds(rowsA[32].template ReinterpretCast<int32_t>(),
                 rowsA.template ReinterpretCast<int32_t>(),
                 static_cast<int32_t>(32U * N * 4U),
                 16);
            Adds(rowsA[48].template ReinterpretCast<int32_t>(),
                 rowsA.template ReinterpretCast<int32_t>(),
                 static_cast<int32_t>(48U * N * 4U),
                 N - 48U);
        }
        Adds(columns[16].template ReinterpretCast<int32_t>(),
             columns.template ReinterpretCast<int32_t>(),
             static_cast<int32_t>(16U * S * 4U),
             16);
        Adds(columns[32].template ReinterpretCast<int32_t>(),
             columns.template ReinterpretCast<int32_t>(),
             static_cast<int32_t>(32U * S * 4U),
             16);
    }

    template <class A0, class A1, class A2, class A3, class A4>
    __aicore__ inline __attribute__((always_inline)) void
    RunTile13(A0& rr, A1& batch, A2& shift, A3& srcRows, A4& srcRows8)
    {

#pragma unroll
        for (uint32_t g = 0; g < 8; g++)
            Adds(srcRows8[g * 64], srcRows.template ReinterpretCast<int32_t>(), static_cast<int32_t>(g * 4U), 64);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::MTE2_V>();
        if (rr != nullptr && batch == 0U) {
            Fence<HardEvent::V_MTE3>();
            if constexpr (N == 64U)
                DataCopy(res, meta[8], 8);
            else
                DataCopyPad(res, meta[8], DataCopyExtParams{1, 24, 0, 0, 0});
        }
#pragma unroll
        for (uint32_t c = 0; c < N / 8U * 8U; c += 8U)
            Gather(mat[c * S], perm, srcRows8.template ReinterpretCast<uint32_t>(), c * 4 + shift, 64, 8, 9);
        if constexpr (N % 8U != 0U)
            Gather(mat[(N / 8U * 8U) * S],
                   perm,
                   srcRows8.template ReinterpretCast<uint32_t>(),
                   (N / 8U * 8U) * 4 + shift,
                   64,
                   N % 8U,
                   9);
#pragma unroll
        for (uint32_t k = 0; k < K; k++)
            Gather(mat[(N + k) * S], raw, rowsB, k * 4 + shift, 64, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    template <class A0, class A1> __aicore__ inline __attribute__((always_inline)) void RunTile14(A0& rr, A1& batch)
    {

#pragma unroll
        for (uint32_t i = 0; i < 16; i++) {
            rowsB.SetValue(i, (i < N) ? i * 8U : 0U);
            if constexpr (N < 64U)
                rowsA.SetValue(i, (i < N) ? i * N * 4U : 0U);
            columns.SetValue(i, i * S * 4U);
        }
        Fence<HardEvent::S_V>();
        if (rr != nullptr && batch == 0U)
            Duplicate(meta[8], 0.0f, 8);
        RunTile12();
        Adds(columns[48].template ReinterpretCast<int32_t>(),
             columns.template ReinterpretCast<int32_t>(),
             static_cast<int32_t>(48U * S * 4U),
             16);
        Adds(columns[64].template ReinterpretCast<int32_t>(),
             columns.template ReinterpretCast<int32_t>(),
             static_cast<int32_t>(64U * S * 4U),
             8);
        PipeBarrier<PIPE_V>();
        Brcb(bcastCols.template ReinterpretCast<float>(),
             columns.template ReinterpretCast<float>(),
             9,
             BrcbRepeatParams{1, 8});
    }

    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6>
    __aicore__ inline __attribute__((always_inline)) void
    RunTile15(A0& remaining, A1& j, A2& col, A3& row, A4& base, A5& count, A6& inv)
    {

        remaining &= ~(uint64_t(1) << row);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(v0, col, -inv, 64, 1, UnaryRepeatParams{1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        MulAddDst(base, v0, factors[128], 64, static_cast<uint8_t>(count), BinaryRepeatParams{1, 1, 0, S / 8, 0, 1});
        PipeBarrier<PIPE_V>();
        if (j + 1U < N) {
            uint64_t nextMask[2] = {remaining, 0};
            Abs(tmp, base, 64, 1, UnaryRepeatParams{1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
            WholeReduceMax(meta, tmp, nextMask, 1, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
        }
    }

    template <class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void
    RunTile16(A0& piv64, A1& diagInv64, A2& remaining, A3& initMask)
    {

        PipeBarrier<PIPE_V>();
        WholeReduceMax(meta, tmp, initMask, 1, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t j = 0; j < N; j++) {
            auto col = mat[j * S];
            uint32_t row = 0;
            float d = ReadPivot(col, row);
            auto base = mat[(j + 1) * S];
            uint32_t count = N + K - j - 1;
            Gather(factors[128], base, bcastCols, row * 4, 64, static_cast<uint8_t>((count + 7U) >> 3U), 8);
            ASSERT(d != 0.0f);
            float inv = 1.0f / d;
            RunTile15(remaining, j, col, row, base, count, inv);
            uint64_t invBits = __builtin_bit_cast(uint32_t, inv);
            uint64_t r4 = row * 4U;
            diagInv64.SetValue(j, (invBits << 32) | invBits);
            piv64.SetValue(j, ((r4 + S * 4U) << 32) | r4);
        }
        Fence<HardEvent::S_V>();
        Gather(raw, mat[N * S], piv, 0, 64, 2, 8);
        PipeBarrier<PIPE_V>();
        Mul(raw, raw, diagInv, 128);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>();
        if constexpr (N == 64U)
            DataCopy(x, raw, 128);
        else
            DataCopyPad(x, raw, DataCopyExtParams{1, 2 * N * 4, 0, 0, 0});
    }

    __aicore__ inline void Run(GM_ADDR aa, GM_ADDR bb, GM_ADDR xx, GM_ADDR rr)
    {
        uint32_t batch = GetBlockIdx();
        uint32_t shift = (N == 64U) ? 0U : (batch & 1U) * 16U;
        a.SetGlobalBuffer((__gm__ float*)aa + ((batch * N * N) & ~7U));
        b.SetGlobalBuffer((__gm__ float*)bb + ((batch * N * K) & ~7U));
        x.SetGlobalBuffer((__gm__ float*)xx + batch * N * K);
        res.SetGlobalBuffer((__gm__ float*)rr);
        pipe.InitBuffer(matBuf, (N + K + 8U) * S * 4);
        pipe.InitBuffer(permBuf, (N + K) * S * 4);
        pipe.InitBuffer(rawBuf, MP * 8 * K * 4);
        pipe.InitBuffer(auxBuf, 2304 * 4);
        mat = matBuf.Get<float>();
        perm = permBuf.Get<float>();
        raw = rawBuf.Get<float>();
        auto aux = auxBuf.Get<float>();
        if constexpr (N == 64U)
            DataCopy(perm, a, DataCopyParams{64, 8, 0, 1});
        else
            DataCopy(perm, a, (N * N + 11U) / 8U * 8U);
        DataCopy(raw, b, (N * K + 11U) / 8U * 8U);
        v0 = aux;
        tmp = aux[128];
        meta = aux[192];
        rows = aux[256].ReinterpretCast<uint32_t>();
        rowsA = aux[320].ReinterpretCast<uint32_t>();
        columns = aux[384].ReinterpretCast<uint32_t>();
        rowsB = aux[464].ReinterpretCast<uint32_t>();
        factors = aux[592];
        bcastCols = aux[1424].ReinterpretCast<uint32_t>();
        piv = aux[2000].ReinterpretCast<uint32_t>();
        diagInv = aux[2128];
        auto piv64 = piv.ReinterpretCast<uint64_t>();
        auto diagInv64 = diagInv.ReinterpretCast<uint64_t>();
        RunTile14(rr, batch);
        auto srcRows = (N == 64U) ? columns : rowsA;
        auto srcRows8 = factors.ReinterpretCast<int32_t>();
        RunTile13(rr, batch, shift, srcRows, srcRows8);
        uint64_t remaining = ~uint64_t(0);
        if constexpr (N < 64)
            remaining = (uint64_t(1) << N) - 1;
        uint64_t initMask[2] = {remaining, 0};
        Abs(tmp, mat, 64, 1, UnaryRepeatParams{1, 1, 8, 8});
        RunTile16(piv64, diagInv64, remaining, initMask);
    }
};

template <uint32_t N, uint32_t MP, uint32_t K> class RowGj {
    static constexpr uint32_t S = 16U;
    TPipe pipe;
    TBuf<TPosition::VECCALC> matBuf;
    TBuf<TPosition::VECCALC> auxBuf;

public:
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
    __aicore__ inline __attribute__((always_inline)) void
    RunTile17(A0& rr, A1& batch, A2& res, A3& zeroBuf, A4& cols, A5& piv, A6& broadcast, A7& transIdx)
    {

#pragma unroll
        for (uint32_t i = 0; i < 16; i++) {
            transIdx.SetValue(i, (i < N) ? i * N * 4U : 0U);
            cols.SetValue(i, i * 64U);
        }
        if constexpr (N < 16U)
            piv.SetValue(15, 0U);
        Fence<HardEvent::S_V>();
        if (rr != nullptr && batch == 0U)
            Duplicate(zeroBuf, 0.0f, 8);
        Brcb(broadcast, cols.template ReinterpretCast<float>(), 2, BrcbRepeatParams{1, 8});
        Adds(transIdx[16].template ReinterpretCast<int32_t>(), transIdx.template ReinterpretCast<int32_t>(), 4, 16);
        Adds(transIdx[32].template ReinterpretCast<int32_t>(), transIdx.template ReinterpretCast<int32_t>(), 8, 16);
        Adds(transIdx[48].template ReinterpretCast<int32_t>(), transIdx.template ReinterpretCast<int32_t>(), 12, 16);
        PipeBarrier<PIPE_V>();
        Adds(transIdx[64].template ReinterpretCast<int32_t>(), transIdx.template ReinterpretCast<int32_t>(), 16, 64);
        Adds(transIdx[128].template ReinterpretCast<int32_t>(), transIdx.template ReinterpretCast<int32_t>(), 32, 64);
        if constexpr (N == 16U)
            Adds(transIdx[192].template ReinterpretCast<int32_t>(),
                 transIdx.template ReinterpretCast<int32_t>(),
                 48,
                 64);
        else
            Adds(transIdx[192].template ReinterpretCast<int32_t>(),
                 transIdx.template ReinterpretCast<int32_t>(),
                 48,
                 48);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::MTE2_V>();
        if (rr != nullptr && batch == 0U) {
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(res, zeroBuf, DataCopyExtParams{1, (N == 16U) ? 16U : 12U, 0, 0, 0});
        }
    }

    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
    __aicore__ inline __attribute__((always_inline)) void
    RunTile18(A0& tmp, A1& meta, A2& mult0, A3& factors, A4& remaining, A5& j, A6& col, A7& base, A8& count, A9& inv)
    {

        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(mult0, col, -inv, 16, 1, UnaryRepeatParams{1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        MulAddDst(base, mult0, factors, 16, static_cast<uint8_t>(count), BinaryRepeatParams{1, 1, 0, 2, 0, 1});
        PipeBarrier<PIPE_V>();
        if (j + 1U < N) {
            uint64_t nextMask[2] = {remaining, 0};
            Abs(tmp, base, 16, 1, UnaryRepeatParams{1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
            WholeReduceMax(meta, tmp, nextMask, 1, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
        }
    }

    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
    __aicore__ inline __attribute__((always_inline)) void RunTile19(A0& mat,
                                                                    A1& tmp,
                                                                    A2& meta,
                                                                    A3& mult0,
                                                                    A4& piv,
                                                                    A5& inverse,
                                                                    A6& broadcast,
                                                                    A7& factors,
                                                                    A8& out,
                                                                    A9& remaining)
    {

        SetFlag<HardEvent::V_S>(EVENT_ID0);
        for (uint32_t j = 0; j < N; j++) {
            auto col = mat[j * S];
            uint32_t row = 0;
            float d = ReadPivot(col, row);
            auto base = mat[(j + 1U) * S];
            uint32_t count = N + K - j - 1U;
            Gather(factors,
                   base,
                   broadcast.template ReinterpretCast<uint32_t>(),
                   row * 4,
                   64,
                   static_cast<uint8_t>((count + 7U) >> 3U),
                   8);
            ASSERT(d != 0.0f);
            float inv = 1.0f / d;
            remaining &= ~(1ULL << row);
            RunTile18(tmp, meta, mult0, factors, remaining, j, col, base, count, inv);
            piv.SetValue(j, row * 4);
            inverse.SetValue(j, inv);
        }
        Fence<HardEvent::S_V>();
        Gather(out, mat[N * S], piv, 0, 16, 1, 8);
        PipeBarrier<PIPE_V>();
        Mul(out, out, inverse, 16, 1, BinaryRepeatParams{1, 1, 1, 8, 8, 8});
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE3>();
    }

    template <class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void RunTile20(A0& a, A1& b, A2& mat, A3& raw)
    {

        if constexpr (N == 16U) {
            DataCopy(raw, a, 256);
            DataCopy(mat[16 * S], b, 16);
        } else {
            DataCopy(raw, a, 232);
            DataCopyPad(mat[N * S],
                        b,
                        DataCopyExtParams{1, N * 4, 0, 0, 0},
                        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(16 - N), 0});
        }
    }

    __aicore__ inline void Run(GM_ADDR aa, GM_ADDR bb, GM_ADDR xx, GM_ADDR rr)
    {
        uint32_t batch = GetBlockIdx();
        uint32_t shiftA = ((batch * N * N) & 7U) * 4U;
        GlobalTensor<float> a;
        GlobalTensor<float> b;
        GlobalTensor<float> x;
        GlobalTensor<float> res;
        a.SetGlobalBuffer((__gm__ float*)aa + ((batch * N * N) & ~7U));
        b.SetGlobalBuffer((__gm__ float*)bb + batch * N * K);
        x.SetGlobalBuffer((__gm__ float*)xx + batch * N * K);
        res.SetGlobalBuffer((__gm__ float*)rr);
        pipe.InitBuffer(matBuf, (MP + 8U) * S * 4);
        pipe.InitBuffer(auxBuf, 2048 * 4);
        auto mat = matBuf.Get<float>();
        auto aux = auxBuf.Get<float>();
        auto raw = aux;
        auto zeroBuf = aux[256];
        auto tmp = aux[320];
        auto meta = aux[384];
        auto mult0 = aux[448];
        auto cols = aux[512].ReinterpretCast<uint32_t>();
        auto piv = aux[576].ReinterpretCast<uint32_t>();
        auto inverse = aux[640];
        auto broadcast = aux[704];
        auto factors = aux[896];
        auto out = aux[1088];
        auto transIdx = aux[1152].ReinterpretCast<uint32_t>();
        RunTile20(a, b, mat, raw);
        RunTile17(rr, batch, res, zeroBuf, cols, piv, broadcast, transIdx);
        if constexpr (N == 16U)
            Gather(mat, raw, transIdx, shiftA, 64, 4, 8);
        else
            Gather(mat, raw, transIdx, shiftA, N * 16);
        PipeBarrier<PIPE_V>();
        uint64_t remaining = (N == 16U) ? 0xFFFFULL : ((1ULL << N) - 1ULL);
        uint64_t initMask[2] = {remaining, 0};
        Abs(tmp, mat, 16, 1, UnaryRepeatParams{1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        WholeReduceMax(meta, tmp, initMask, 1, 1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
        RunTile19(mat, tmp, meta, mult0, piv, inverse, broadcast, factors, out, remaining);
        if constexpr (N == 16U)
            DataCopy(x, out, 16);
        else
            DataCopyPad(x, out, DataCopyExtParams{1, N * 4, 0, 0, 0});
    }
};

template <typename DT_A, bool PAR> class Solver {
    TPipe pipe;
    TBuf<TPosition::VECCALC> matrixBuf;
    TBuf<TPosition::VECCALC> rawBuf;
    TBuf<TPosition::VECCALC> vBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> reduceBuf;
    TBuf<TPosition::VECCALC> indexBuf;
    TBuf<TPosition::VECCALC> scalarBuf;
    TBuf<TPosition::VECCALC> extraBuf;
    TBuf<TPosition::VECCALC> rhsScratchBuf;
    LocalTensor<float> rhsScratch;
    LocalTensor<float> matrix;
    LocalTensor<float> raw;
    LocalTensor<float> v;
    LocalTensor<float> tmp;
    LocalTensor<float> red;
    LocalTensor<float> scalar;
    LocalTensor<float> rh;
    LocalTensor<float> rl;
    LocalTensor<float> ah;
    LocalTensor<float> al;
    LocalTensor<float> ph;
    LocalTensor<float> pl;
    LocalTensor<float> bh;
    LocalTensor<float> bl;
    LocalTensor<float> w;
    LocalTensor<float> g;
    LocalTensor<uint32_t> indices;
    GlobalTensor<DT_A> a;
    GlobalTensor<DT_A> b;
    GlobalTensor<DT_A> x;
    GlobalTensor<DT_A> res;
    GlobalTensor<float> work;
    LinalgLstsqTilingData t;
    uint32_t lane;
    uint32_t group;
    uint32_t total;
    uint32_t localCols;
    uint32_t splitCol;
    bool wantResidual;

    __aicore__ inline float Dot(LocalTensor<float> q, LocalTensor<float> z)
    {
        Mul(tmp, q, z, t.mp);
        PipeBarrier<PIPE_V>();
        ReduceSum(scalar, tmp, red, t.mp);
        Fence<HardEvent::V_S>();
        return scalar.GetValue(0);
    }

    __aicore__ inline void ReadColumn(LocalTensor<float> dst, GlobalTensor<DT_A> src, uint32_t width, uint32_t col)
    {
        Duplicate(dst, 0.0f, t.mp);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_MTE2>();
        if constexpr (sizeof(DT_A) == 4) {
            DataCopyPad(raw,
                        src[col],
                        DataCopyExtParams{static_cast<uint16_t>(t.m), 4, (width - 1) * 4, 0, 0},
                        DataCopyPadExtParams<float>{true, 0, 7, 0.0f});
            Fence<HardEvent::MTE2_V>();
            Gather(dst, raw, indices, 0, t.m);
            PipeBarrier<PIPE_V>();
        } else {
            DataCopyPad(raw.ReinterpretCast<DT_A>(),
                        src[col],
                        DataCopyExtParams{static_cast<uint16_t>(t.m), 2, (width - 1) * 2, 0, 0},
                        DataCopyPadExtParams<DT_A>{false, 0, 0, 0});
            Fence<HardEvent::MTE2_V>();
            Duplicate(red, 0.0f, t.mp);
            PipeBarrier<PIPE_V>();
            Gather(red.ReinterpretCast<half>(), raw.ReinterpretCast<half>(), indices, 0, t.m);
            PipeBarrier<PIPE_V>();
            Cast(dst, red.ReinterpretCast<DT_A>(), RoundMode::CAST_NONE, t.m);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Split(LocalTensor<float> z, LocalTensor<float> hi, LocalTensor<float> lo)
    {
        Muls(w, z, 4097.0f, t.mp);
        PipeBarrier<PIPE_V>();
        Sub(hi, w, z, t.mp);
        PipeBarrier<PIPE_V>();
        Sub(hi, w, hi, t.mp);
        PipeBarrier<PIPE_V>();
        Sub(lo, z, hi, t.mp);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProductScalar(LocalTensor<float> z, float a)
    {
        Split(z, ah, al);
        float c = 4097.0f * a;
        float hi = c - (c - a);
        float lo = a - hi;
        Muls(ph, z, a, t.mp);
        Muls(pl, ah, hi, t.mp);
        PipeBarrier<PIPE_V>();
        Sub(pl, pl, ph, t.mp);
        PipeBarrier<PIPE_V>();
        Muls(w, ah, lo, t.mp);
        PipeBarrier<PIPE_V>();
        Add(pl, pl, w, t.mp);
        PipeBarrier<PIPE_V>();
        Muls(w, al, hi, t.mp);
        PipeBarrier<PIPE_V>();
        Add(pl, pl, w, t.mp);
        PipeBarrier<PIPE_V>();
        Muls(w, al, lo, t.mp);
        PipeBarrier<PIPE_V>();
        Add(pl, pl, w, t.mp);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProductVec(LocalTensor<float> a0, LocalTensor<float> b0)
    {
        Split(a0, ah, al);
        Split(b0, bh, bl);
        Mul(ph, a0, b0, t.mp);
        Mul(pl, ah, bh, t.mp);
        PipeBarrier<PIPE_V>();
        Sub(pl, pl, ph, t.mp);
        PipeBarrier<PIPE_V>();
        Mul(w, ah, bl, t.mp);
        PipeBarrier<PIPE_V>();
        Add(pl, pl, w, t.mp);
        PipeBarrier<PIPE_V>();
        Mul(w, al, bh, t.mp);
        PipeBarrier<PIPE_V>();
        Add(pl, pl, w, t.mp);
        PipeBarrier<PIPE_V>();
        Mul(w, al, bl, t.mp);
        PipeBarrier<PIPE_V>();
        Add(pl, pl, w, t.mp);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateResidual()
    {
        Add(ah, rh, ph, t.mp);
        PipeBarrier<PIPE_V>();
        Sub(al, ah, rh, t.mp);
        PipeBarrier<PIPE_V>();
        Sub(bh, ah, al, t.mp);
        PipeBarrier<PIPE_V>();
        Sub(bh, rh, bh, t.mp);
        Sub(tmp, ph, al, t.mp);
        PipeBarrier<PIPE_V>();
        Add(bh, bh, tmp, t.mp);
        PipeBarrier<PIPE_V>();
        Add(bh, bh, pl, t.mp);
        PipeBarrier<PIPE_V>();
        Add(rl, rl, bh, t.mp);
        Adds(rh, ah, 0.0f, t.mp);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float AccurateSum()
    {
        for (uint32_t count = t.mp; count > 8; count /= 2) {
            uint32_t h = count / 2;
            Add(ah, ph, ph[h], h);
            PipeBarrier<PIPE_V>();
            Sub(al, ah, ph, h);
            PipeBarrier<PIPE_V>();
            Sub(bh, ah, al, h);
            PipeBarrier<PIPE_V>();
            Sub(bh, ph, bh, h);
            Sub(tmp, ph[h], al, h);
            PipeBarrier<PIPE_V>();
            Add(bh, bh, tmp, h);
            Add(pl, pl, pl[h], h);
            PipeBarrier<PIPE_V>();
            Add(pl, pl, bh, h);
            Adds(ph, ah, 0.0f, h);
            PipeBarrier<PIPE_V>();
        }
        Fence<HardEvent::V_S>();
        float hi = 0.0f;
        float lo = 0.0f;
        for (uint32_t i = 0; i < 8; i++) {
            float z = ph.GetValue(i);
            float ss = hi + z;
            float vv = ss - hi;
            lo += ((hi - (ss - vv)) + (z - vv)) + pl.GetValue(i);
            hi = ss;
        }
        return hi + lo;
    }

    __aicore__ inline void LoadCol(LocalTensor<float> dst, uint32_t c)
    {
        if constexpr (PAR) {
            if (c < splitCol) {
                Fence<HardEvent::V_MTE2>();
                DataCopy(dst, work[c * t.mp], t.mp);
                Fence<HardEvent::MTE2_V>();
                Fence<HardEvent::MTE2_S>();
            } else {
                Adds(dst, matrix[(c - splitCol) * t.mp], 0.0f, t.mp);
                PipeBarrier<PIPE_V>();
                Fence<HardEvent::V_S>();
            }
        } else {
            Adds(dst, matrix[c * t.mp], 0.0f, t.mp);
            PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();
        }
    }

    __aicore__ inline void StoreCol(uint32_t c, LocalTensor<float> src)
    {
        if constexpr (PAR) {
            if (c < splitCol) {
                Fence<HardEvent::V_MTE3>();
                DataCopy(work[c * t.mp], src, t.mp);
                Fence<HardEvent::MTE3_V>();
                Fence<HardEvent::MTE3_S>();
            } else {
                Adds(matrix[(c - splitCol) * t.mp], src, 0.0f, t.mp);
                PipeBarrier<PIPE_V>();
                Fence<HardEvent::V_S>();
            }
        } else {
            Adds(matrix[c * t.mp], src, 0.0f, t.mp);
            PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();
        }
    }

    __aicore__ inline void LoadR(uint32_t j)
    {
        LoadCol(v, j);
    }

    __aicore__ inline LocalTensor<float> LoadRhs(uint32_t c)
    {
        if constexpr (PAR) {
            if (c < splitCol) {
                LoadCol(rhsScratch, c);
                return rhsScratch;
            }
        }
        return matrix[(c - splitCol) * t.mp];
    }

    __aicore__ inline void StoreRhs(uint32_t c)
    {
        if constexpr (PAR) {
            if (c < splitCol) {
                StoreCol(c, rhsScratch);
            }
        }
    }

    __aicore__ inline void ComputeResidual(uint32_t batch, uint32_t rhs, LocalTensor<float> sol)
    {
        Duplicate(rh, 0.0f, t.mp);
        Duplicate(rl, 0.0f, t.mp);
        PipeBarrier<PIPE_V>();
        ReadColumn(rh, b[static_cast<uint64_t>(batch) * t.m * t.k], t.k, rhs);
        for (uint32_t j = 0; j < t.n; j++) {
            ReadColumn(v, a[static_cast<uint64_t>(batch) * t.m * t.n], t.n, j);
            Fence<HardEvent::V_S>();
            float val = sol.GetValue(j);
            ProductScalar(v, -val);
            AccumulateResidual();
        }
    }

    __aicore__ inline void StoreResidual(uint32_t batch, uint32_t rhs, float val)
    {
        Duplicate(scalar, val, 8);
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(DT_A) == 4) {
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(res[static_cast<uint64_t>(batch) * t.k + rhs], scalar, DataCopyExtParams{1, 4, 0, 0, 0});
        } else {
            Cast(red.ReinterpretCast<DT_A>(), scalar, RoundMode::CAST_RINT, 8);
            PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_MTE3>();
            DataCopyPad(res[static_cast<uint64_t>(batch) * t.k + rhs],
                        red.ReinterpretCast<DT_A>(),
                        DataCopyExtParams{1, 2, 0, 0, 0});
        }
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::MTE3_S>();
    }

    __aicore__ inline void NormalizeResidual()
    {
        Add(ah, rh, rl, t.mp);
        PipeBarrier<PIPE_V>();
        Sub(al, ah, rh, t.mp);
        PipeBarrier<PIPE_V>();
        Sub(bh, ah, al, t.mp);
        PipeBarrier<PIPE_V>();
        Sub(bh, rh, bh, t.mp);
        Sub(tmp, rl, al, t.mp);
        PipeBarrier<PIPE_V>();
        Add(rl, bh, tmp, t.mp);
        Adds(rh, ah, 0.0f, t.mp);
        PipeBarrier<PIPE_V>();
    }

    template <class A0, class A1, class A2>
    __aicore__ inline __attribute__((always_inline)) void RefineTile21(A0& batch, A1& rhs, A2& sol)
    {

        ComputeResidual(batch, rhs, sol);
        NormalizeResidual();
        Duplicate(g, 0.0f, t.mp);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < t.n; j++) {
            ReadColumn(v, a[static_cast<uint64_t>(batch) * t.m * t.n], t.n, j);
            ProductVec(v, rh);
            Mul(tmp, v, rl, t.mp);
            PipeBarrier<PIPE_V>();
            Add(pl, pl, tmp, t.mp);
            PipeBarrier<PIPE_V>();
            float grad = AccurateSum();
            g.SetValue(j, grad);
            Fence<HardEvent::S_V>();
        }
        for (uint32_t j = 0; j < t.n; j++) {
            LoadR(j);
            float diag = v.GetValue(j);
            float sum = 0.0f;
            if (j) {
                Mul(tmp, v, g, j);
                PipeBarrier<PIPE_V>();
                ReduceSum(scalar, tmp, red, j);
                Fence<HardEvent::V_S>();
                sum = scalar.GetValue(0);
            }
            ASSERT(diag != 0.0f);
            float val = (g.GetValue(j) - sum) / diag;
            g.SetValue(j, val);
            Fence<HardEvent::S_V>();
        }
    }

    __aicore__ inline void Refine(uint32_t batch, uint32_t rhs, LocalTensor<float> sol)
    {
        for (uint32_t iteration = 0; iteration < 1; iteration++) {
            RefineTile21(batch, rhs, sol);
            for (int j = static_cast<int>(t.n) - 1; j >= 0; j--) {
                LoadR(j);
                Fence<HardEvent::V_S>();
                float val = g.GetValue(j) / v.GetValue(j);
                if (j) {
                    Muls(tmp, v, -val, j);
                    PipeBarrier<PIPE_V>();
                    Add(g, g, tmp, j);
                    Fence<HardEvent::V_S>();
                }
                g.SetValue(j, val);
                Fence<HardEvent::S_V>();
            }
            if (t.m > t.n && wantResidual) {
                for (uint32_t j = 0; j < t.n; j++) {
                    ReadColumn(v, a[static_cast<uint64_t>(batch) * t.m * t.n], t.n, j);
                    Fence<HardEvent::V_S>();
                    ProductScalar(v, -g.GetValue(j));
                    AccumulateResidual();
                }
            }
            Add(sol, sol, g, t.n);
            PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();
        }
        if (t.m > t.n && wantResidual) {
            NormalizeResidual();
            ProductVec(rh, rh);
            float value = AccurateSum();
            StoreResidual(batch, rhs, value);
        } else if (t.m == t.n && wantResidual) {
            StoreResidual(batch, rhs, 0.0f);
        }
    }

public:
    template <class A0, class A1, class A2, class A3, class A4, class A5>
    __aicore__ inline __attribute__((always_inline)) void RunTile22(A0& aa, A1& bb, A2& xx, A3& rr, A4& ww, A5& td)
    {

        t = td;
        wantResidual = rr != nullptr && t.m > t.n;
        lane = 0;
        group = GetBlockIdx();
        total = t.n + t.k;
        localCols = PAR ? t.localCols : total;
        splitCol = PAR ? (total - localCols) : 0U;
        a.SetGlobalBuffer((__gm__ DT_A*)aa);
        b.SetGlobalBuffer((__gm__ DT_A*)bb);
        x.SetGlobalBuffer((__gm__ DT_A*)xx);
        res.SetGlobalBuffer((__gm__ DT_A*)rr);
        if constexpr (PAR)
            work.SetGlobalBuffer((__gm__ float*)ww + group * (total + 1U) * t.mp);
        pipe.InitBuffer(matrixBuf, localCols * t.mp * 4);
        pipe.InitBuffer(rawBuf, t.mp * 32);
        pipe.InitBuffer(vBuf, t.mp * 4);
        pipe.InitBuffer(tmpBuf, t.mp * 4);
        pipe.InitBuffer(reduceBuf, t.mp * 4);
        pipe.InitBuffer(indexBuf, t.mp * 4);
        pipe.InitBuffer(scalarBuf, 32);
        pipe.InitBuffer(extraBuf, t.mp * 4 * 10);
        if constexpr (PAR) {
            if (t.k > localCols) {
                // This column must survive all scratch-buffer uses in Refine.
                pipe.InitBuffer(rhsScratchBuf, t.mp * sizeof(float));
                rhsScratch = rhsScratchBuf.template Get<float>();
            }
        }
        matrix = matrixBuf.template Get<float>();
        raw = rawBuf.template Get<float>();
        v = vBuf.template Get<float>();
        tmp = tmpBuf.template Get<float>();
        red = reduceBuf.template Get<float>();
        indices = indexBuf.template Get<uint32_t>();
    }

    template <class A0> __aicore__ inline __attribute__((always_inline)) void RunTile23(A0& batch)
    {

        Fence<HardEvent::V_S>();
        if (t.n == 1 && wantResidual)
            for (uint32_t c = t.n; c < total; c++) {
                if (t.m > t.n) {
                    LoadCol(v, c);
                    Duplicate(v, 0.0f, t.n);
                    PipeBarrier<PIPE_V>();
                    float val = Dot(v, v);
                    StoreResidual(batch, c - t.n, val);
                } else {
                    StoreResidual(batch, c - t.n, 0.0f);
                }
            }
        for (int j = static_cast<int>(t.n) - 1; j >= 0; j--) {
            LoadCol(v, static_cast<uint32_t>(j));
            float diag = v.GetValue(j);
            for (uint32_t c = t.n; c < total; c++) {
                auto z = LoadRhs(c);
                Fence<HardEvent::V_S>();
                ASSERT(diag != 0.0f);
                float value = z.GetValue(j) / diag;
                if (j) {
                    Axpy(z, v, -value, j);
                    PipeBarrier<PIPE_V>();
                    Fence<HardEvent::V_S>();
                }
                z.SetValue(j, value);
                Fence<HardEvent::S_V>();
                StoreRhs(c);
            }
        }
    }

    template <class A0, class A1> __aicore__ inline __attribute__((always_inline)) void RunTile24(A0& batch, A1& c)
    {

        uint32_t rhs = c - t.n;
        LocalTensor<float> sol = LoadRhs(c);
        if constexpr (sizeof(DT_A) == 4) {
            Fence<HardEvent::V_S>();
            auto rawBits = raw.template ReinterpretCast<uint32_t>();
            auto solBits = sol.template ReinterpretCast<uint32_t>();
            for (uint32_t i = 0; i < t.n; i++)
                rawBits.SetValue(i * 8, solBits.GetValue(i));
            Fence<HardEvent::S_MTE3>();
            DataCopyPad(x[static_cast<uint64_t>(batch) * t.n * t.k + rhs],
                        raw,
                        DataCopyExtParams{static_cast<uint16_t>(t.n), 4, 0, (t.k - 1) * 4, 0});
        } else {
            Cast(red.template ReinterpretCast<DT_A>(), sol, RoundMode::CAST_RINT, t.n);
            PipeBarrier<PIPE_V>();
            Fence<HardEvent::V_S>();
            auto rb = raw.template ReinterpretCast<uint16_t>();
            auto sb = red.template ReinterpretCast<uint16_t>();
            for (uint32_t i = 0; i < t.n; i++)
                rb.SetValue(i * 16, sb.GetValue(i));
            Fence<HardEvent::S_MTE3>();
            DataCopyPad(x[static_cast<uint64_t>(batch) * t.n * t.k + rhs],
                        raw.template ReinterpretCast<DT_A>(),
                        DataCopyExtParams{static_cast<uint16_t>(t.n), 2, 0, (t.k - 1) * 2, 0});
        }
        Fence<HardEvent::MTE3_V>();
        Fence<HardEvent::MTE3_S>();
    }

    __aicore__ inline __attribute__((always_inline)) void
    ComputeHouseholderScale(float tail2, float lead, float& alpha, float& tau)
    {
        Duplicate(scalar, tail2 + lead * lead, 8);
        PipeBarrier<PIPE_V>();
        Sqrt(scalar, scalar, 8);
        Fence<HardEvent::V_S>();
        float norm = scalar.GetValue(0);
        alpha = lead >= 0.0f ? -norm : norm;
        ASSERT(alpha != 0.0f);
        tau = (alpha - lead) / alpha;
        Muls(v, v, 1.0f / (lead - alpha), t.mp);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::V_S>();
    }

    template <class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void RunTile25(A0& j, A1& lead, A2& alpha, A3& tau)
    {

        LoadCol(rh, j);
        Adds(v, rh, 0.0f, t.mp);
        PipeBarrier<PIPE_V>();
        Duplicate(v, 0.0f, j + 1);
        PipeBarrier<PIPE_V>();
        float tail2 = Dot(v, v);
        lead = rh.GetValue(j);
        alpha = lead;
        if (tail2 > 0.0f) {
            ComputeHouseholderScale(tail2, lead, alpha, tau);
        }
        v.SetValue(j, 1.0f);
        rh.SetValue(j, alpha);
        Fence<HardEvent::S_V>();
        StoreCol(j, rh);
    }

    template <class A0, class A1, class A2, class A3>
    __aicore__ inline __attribute__((always_inline)) void RunTile26(A0& j, A1& lead, A2& alpha, A3& tau)
    {

        if (PAR && j < splitCol) {
            RunTile25(j, lead, alpha, tau);
        } else {
            auto col = matrix[(j - splitCol) * t.mp];
            Adds(v, col, 0.0f, t.mp);
            PipeBarrier<PIPE_V>();
            Duplicate(v, 0.0f, j + 1);
            PipeBarrier<PIPE_V>();
            float tail2 = Dot(v, v);
            lead = col.GetValue(j);
            alpha = lead;
            if (tail2 > 0.0f) {
                ComputeHouseholderScale(tail2, lead, alpha, tau);
            }
            v.SetValue(j, 1.0f);
            col.SetValue(j, alpha);
            Fence<HardEvent::S_V>();
        }
    }

    template <class A0> __aicore__ inline __attribute__((always_inline)) void RunTile27(A0& batch)
    {

        Duplicate(matrix, 0.0f, localCols * t.mp);
        PipeBarrier<PIPE_V>();
        for (uint32_t c = 0; c < total; c++) {
            LocalTensor<float> dst = (PAR && c < splitCol) ? v : matrix[(c - splitCol) * t.mp];
            if (c < t.n)
                ReadColumn(dst, a[static_cast<uint64_t>(batch) * t.m * t.n], t.n, c);
            else
                ReadColumn(dst, b[static_cast<uint64_t>(batch) * t.m * t.k], t.k, c - t.n);
            if constexpr (PAR) {
                if (c < splitCol)
                    StoreCol(c, v);
            }
        }
        for (uint32_t j = 0; j < t.n; j++) {
            float lead;
            float alpha;
            float tau = 0.0f;
            RunTile26(j, lead, alpha, tau);
            for (uint32_t c = j + 1; c < total; c++) {
                if (PAR && c < splitCol) {
                    LoadCol(rh, c);
                    float dot = Dot(v, rh);
                    Axpy(rh, v, -tau * dot, t.mp);
                    PipeBarrier<PIPE_V>();
                    StoreCol(c, rh);
                } else {
                    auto z = matrix[(c - splitCol) * t.mp];
                    float dot = Dot(v, z);
                    Axpy(z, v, -tau * dot, t.mp);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
    }

    __aicore__ inline void Run(GM_ADDR aa, GM_ADDR bb, GM_ADDR xx, GM_ADDR rr, GM_ADDR ww, const LinalgLstsqTilingData& td)
    {
        RunTile22(aa, bb, xx, rr, ww, td);
        scalar = scalarBuf.Get<float>();
        auto ex = extraBuf.Get<float>();
        rh = ex;
        rl = ex[t.mp];
        ah = ex[2 * t.mp];
        al = ex[3 * t.mp];
        ph = ex[4 * t.mp];
        pl = ex[5 * t.mp];
        bh = ex[6 * t.mp];
        bl = ex[7 * t.mp];
        w = ex[8 * t.mp];
        g = ex[9 * t.mp];
        for (uint32_t i = 0; i < t.mp; i++)
            indices.SetValue(i, i * 32);
        Fence<HardEvent::S_V>();
        for (uint32_t round = 0; round < t.rounds; round++) {
            const uint64_t batchIndex = static_cast<uint64_t>(round) * t.groups + group;
            if (batchIndex >= t.batches)
                break;
            const uint32_t batch = static_cast<uint32_t>(batchIndex);
            RunTile27(batch);
            RunTile23(batch);
            if (t.n > 1) {
                Duplicate(v, 0.0f, t.mp);
                PipeBarrier<PIPE_V>();
                for (uint32_t c = t.n; c < total; c++) {
                    LocalTensor<float> sol = LoadRhs(c);
                    Refine(batch, c - t.n, sol);
                    StoreRhs(c);
                }
            }
            for (uint32_t c = t.n; c < total; c++) {
                RunTile24(batch, c);
            }
        }
    }
};

template <typename DT_A, uint32_t CASE_ID>
__global__ __aicore__ void
linalg_lstsq(GM_ADDR A, GM_ADDR B, GM_ADDR solution, GM_ADDR residuals, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(768, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(1536, KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(LinalgLstsqTilingData);
    if constexpr (sizeof(DT_A) == 4 && CASE_ID == 1U) {
        RowGj<16U, 16U, 1U> op;
        // All specialized paths solve square systems with an empty residuals output.
        op.Run(A, B, solution, nullptr);
    } else if constexpr (sizeof(DT_A) == 4 && CASE_ID == 2U) {
        SmallLu<64U, 64U, 2U> op;
        op.Run(A, B, solution, nullptr);
    } else if constexpr (sizeof(DT_A) == 4 && CASE_ID == 3U) {
        BlockLu256<256U, 3U, 8U> op;
        op.Run(A, B, solution, nullptr, GetUserWorkspace(workspace));
    } else if constexpr (sizeof(DT_A) == 4 && CASE_ID == 4U) {
        RowGj<15U, 16U, 1U> op;
        op.Run(A, B, solution, nullptr);
    } else if constexpr (sizeof(DT_A) == 4 && CASE_ID == 5U) {
        SmallLu<50U, 64U, 2U> op;
        op.Run(A, B, solution, nullptr);
    } else if constexpr (sizeof(DT_A) == 4 && CASE_ID == 6U) {
        BlockLu200<200U, 4U, 10U> op;
        op.Run(A, B, solution, nullptr, GetUserWorkspace(workspace));
    } else {
        GET_TILING_DATA_WITH_STRUCT(LinalgLstsqTilingData, t, tiling);
        if constexpr (CASE_ID == 7U) {
            Solver<DT_A, true> op;
            op.Run(A, B, solution, residuals, GetUserWorkspace(workspace), t);
        } else {
            Solver<DT_A, false> op;
            op.Run(A, B, solution, residuals, nullptr, t);
        }
    }
}
