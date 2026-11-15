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

template <class T, uint32_t N = 0, bool FIXP = false> class PinvInverse199 : private PinvInverseState<T, N, FIXP> {
    __aicore__ inline void Update(uint32_t c0, uint32_t c1, uint32_t row);

    template <typename ExtractType33_0, typename ExtractType33_1, typename ExtractType33_2, typename ExtractType33_3,
              typename ExtractType33_4, typename ExtractType33_5, typename ExtractType33_6, typename ExtractType33_7,
              typename ExtractType33_8, typename ExtractType33_9, typename ExtractType33_10, typename ExtractType33_11>
    __aicore__ inline void
    RefineInverseBlock31Block33(ExtractType33_0 &residual, ExtractType33_1 &inv, ExtractType33_2 &chunk,
                                ExtractType33_3 &savedA, ExtractType33_4 &acc, ExtractType33_5 &acc1,
                                ExtractType33_6 &acc2, ExtractType33_7 &acc3, ExtractType33_8 &coeff,
                                ExtractType33_9 &cache, ExtractType33_10 &br, ExtractType33_11 &outputIndices);
    template <typename ExtractType31_0, typename ExtractType31_1, typename ExtractType31_2, typename ExtractType31_3>
    __aicore__ inline void RefineInverseBlock31(ExtractType31_0 &residual, ExtractType31_1 &corrected,
                                                ExtractType31_2 &inv, ExtractType31_3 &chunk);

    __aicore__ inline void RefineInverse(uint32_t batch);

public:
    __aicore__ inline void InitBlock50();
    __aicore__ inline void Init(GM_ADDR ai, GM_ADDR ao, GM_ADDR ws, LinalgPinvTilingData tt);
    template <typename ExtractType45_0, typename ExtractType45_1, typename ExtractType45_2, typename ExtractType45_3,
              typename ExtractType45_4, typename ExtractType45_5, typename ExtractType45_6, typename ExtractType45_7>
    __aicore__ inline void RunBlock29Block34Block43Block45(ExtractType45_0 &ac, ExtractType45_1 &pc,
                                                           ExtractType45_2 &stop, ExtractType45_3 &j,
                                                           ExtractType45_4 &lCur, ExtractType45_5 &hCur,
                                                           ExtractType45_6 &lNxt, ExtractType45_7 &hNxt);
    template <typename ExtractType43_0, typename ExtractType43_1, typename ExtractType43_2, typename ExtractType43_3,
              typename ExtractType43_4, typename ExtractType43_5, typename ExtractType43_6>
    __aicore__ inline void RunBlock29Block34Block43(ExtractType43_0 &ac, ExtractType43_1 &pc, ExtractType43_2 &stop,
                                                    ExtractType43_3 &low0, ExtractType43_4 &high0,
                                                    ExtractType43_5 &low1, ExtractType43_6 &high1);
    template <typename ExtractType34_0, typename ExtractType34_1, typename ExtractType34_2, typename ExtractType34_3>
    __aicore__ inline void RunBlock29Block34(ExtractType34_0 &ac, ExtractType34_1 &pc, ExtractType34_2 &slot,
                                             ExtractType34_3 &stop);
    template <typename ExtractType41_0, typename ExtractType41_1, typename ExtractType41_2, typename ExtractType41_3>
    __aicore__ inline void RunBlock29Block37Block41(ExtractType41_0 &ac, ExtractType41_1 &bad, ExtractType41_2 &pc,
                                                    ExtractType41_3 &stop);
    template <typename ExtractType37_0, typename ExtractType37_1, typename ExtractType37_2, typename ExtractType37_3,
              typename ExtractType37_4>
    __aicore__ inline void RunBlock29Block37(ExtractType37_0 &ac, ExtractType37_1 &bad, ExtractType37_2 &pc,
                                             ExtractType37_3 &slot, ExtractType37_4 &stop);
    template <typename ExtractType29_0, typename ExtractType29_1>
    __aicore__ inline void RunBlock29(ExtractType29_0 &ac, ExtractType29_1 &bad);
    template <typename ExtractType40_0, typename ExtractType40_1, typename ExtractType40_2, typename ExtractType40_3,
              typename ExtractType40_4, typename ExtractType40_5>
    __aicore__ inline void RunBlock32Block40(ExtractType40_0 &base, ExtractType40_1 &ac, ExtractType40_2 &invall,
                                             ExtractType40_3 &loaded, ExtractType40_4 &packed,
                                             ExtractType40_5 &rowIndices);
    template <typename ExtractType32_0, typename ExtractType32_1, typename ExtractType32_2, typename ExtractType32_3,
              typename ExtractType32_4, typename ExtractType32_5>
    __aicore__ inline bool RunBlock32(ExtractType32_0 &failed, ExtractType32_1 &base, ExtractType32_2 &ac,
                                      ExtractType32_3 &invall, ExtractType32_4 &unbounded, ExtractType32_5 &sensitive);
    template <typename ExtractType39_0, typename ExtractType39_1, typename ExtractType39_2>
    __aicore__ inline void RunBlock39(ExtractType39_0 &ac, ExtractType39_1 &invall, ExtractType39_2 &isq);
    template <typename ExtractType47_0, typename ExtractType47_1, typename ExtractType47_2, typename ExtractType47_3,
              typename ExtractType47_4>
    __aicore__ inline void RunBlock47(ExtractType47_0 &needs, ExtractType47_1 &unbounded, ExtractType47_2 &accept,
                                      ExtractType47_3 &myOk, ExtractType47_4 &sensitive);

    template <typename ExtractType52_0, typename ExtractType52_1, typename ExtractType52_2, typename ExtractType52_3,
              typename ExtractType52_4>
    __aicore__ inline void RunBlock52(ExtractType52_0 &batch, ExtractType52_1 &needs, ExtractType52_2 &unbounded,
                                      ExtractType52_3 &accept, ExtractType52_4 &myOk);

    __aicore__ inline float LoadBatchInput(uint32_t &batch, uint32_t &base, uint32_t &ac);
    template <typename ExtractType53_0> __aicore__ inline bool RunBlock53(ExtractType53_0 &failed);
    __aicore__ inline bool Run();
};

template <typename ExtractType55_0, typename ExtractType55_1, typename ExtractType55_2, typename ExtractType55_3,
          typename ExtractType55_4>
__aicore__ inline void FactorOwnerFixed256Block36Block55(ExtractType55_0 &count, ExtractType55_1 &m,
                                                         ExtractType55_2 &tmp, ExtractType55_3 &row,
                                                         ExtractType55_4 &c0)
{
    if (c0 < count) {
        set_vector_mask((uint64_t)0, ((uint64_t)1 << (count - c0)) - 1);
        vgather((__ubuf__ uint32_t *)tmp, (__ubuf__ uint32_t *)(m + 45056), (c0 * 264 + row) * 4, 8, 1);
        PipeBarrier<PIPE_V>();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        vbrcb((__ubuf__ uint32_t *)(m + 44096), (__ubuf__ uint32_t *)tmp, 1, 8, (count - c0 + 7) / 8);
        vbrcb((__ubuf__ uint32_t *)(m + 38912), (__ubuf__ uint32_t *)tmp, 1, 8, (count - c0 + 7) / 8);
    }
}

template <typename ExtractType58_0, typename ExtractType58_1, typename ExtractType58_2, typename ExtractType58_3>
__aicore__ inline void FactorOwnerFixed256Block36Block58(ExtractType58_0 &count, ExtractType58_1 &m,
                                                         ExtractType58_2 &c0, ExtractType58_3 &piv)
{
    if (c0 < count) {
        vmla(m + c0 * 264, piv, m + 44096, count - c0, 1, 1, 0, 33, 0, 1);
        vmla(m + c0 * 264 + 64, piv + 64, m + 38912, count - c0, 1, 1, 0, 33, 0, 1);
        vmla(m + c0 * 264 + 128, piv + 128, m + 44096, count - c0, 1, 1, 0, 33, 0, 1);
        vmla(m + c0 * 264 + 192, piv + 192, m + 38912, count - c0, 1, 1, 0, 33, 0, 1);
        PipeBarrier<PIPE_V>();
    }
}

template <typename ExtractType36_0, typename ExtractType36_1, typename ExtractType36_2, typename ExtractType36_3,
          typename ExtractType36_4, typename ExtractType36_5, typename ExtractType36_6, typename ExtractType36_7,
          typename ExtractType36_8>
__aicore__ inline void FactorOwnerFixed256Block36(ExtractType36_0 &pc, ExtractType36_1 &count, ExtractType36_2 &m,
                                                  ExtractType36_3 &panel, ExtractType36_4 &tmp, ExtractType36_5 &used,
                                                  ExtractType36_6 &perm, ExtractType36_7 &invs, ExtractType36_8 &bad)
{
    for (uint32_t j = 0; j < count; j++) {
        __ubuf__ float *col = m + j * 264;
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        vand((__ubuf__ uint16_t *)tmp, (__ubuf__ uint16_t *)col, (__ubuf__ uint16_t *)used, 4, 1, 1, 1, 8, 8, 8);
        PipeBarrier<PIPE_V>();
        vcmax(m + 40000, tmp, 4, 1, 1, 8, Order_t::VALUE_INDEX);
        LocalSync<HardEvent::V_S>();
        float maximum, indexBits;
        GetReduceMaxMinCount(maximum, indexBits);

        union Bits {
            float value;
            uint32_t bits;
        };

        Bits selected;
        selected.value = indexBits;
        uint32_t row = selected.bits;
        uint32_t c0 = j + 1;
        FactorOwnerFixed256Block36Block55(count, m, tmp, row, c0);
        float value = col[row];
        if (!(value > 1e-20f || value < -1e-20f)) {
            bad = true;
            value = 1.0f;
        }
        float inv = 1.0f / value;
        __ubuf__ float *piv = panel + j * 264;
        col[row] = 0.0f;
        piv[256] = float(int32_t(row));
        piv[257] = inv;
        used[row] = 0;
        perm[pc + j] = row;
        invs[pc + j] = inv;
        LocalSync<HardEvent::S_V>();
        set_vector_mask((uint64_t)0, (uint64_t)-1);
        vmuls(piv, col, -inv, 4, 1, 1, 8, 8);
        PipeBarrier<PIPE_V>();
        FactorOwnerFixed256Block36Block58(count, m, c0, piv);
    }
}

static __aicore__ __attribute__((noinline)) bool FactorOwnerFixed256(uint32_t pc, uint32_t count)
{
    if ASCEND_IS_AIV {
        auto mem = LocalTensor<float>(TPosition::VECCALC, 0, 48640);
        __ubuf__ float *m = (__ubuf__ float *)0;
        __ubuf__ float *panel = m + 20000;
        __ubuf__ float *tmp = m + 41024;
        __ubuf__ uint32_t *used = (__ubuf__ uint32_t *)(m + 42048);
        __ubuf__ uint32_t *perm = (__ubuf__ uint32_t *)(m + 43072);
        __ubuf__ float *invs = m + 43584;
        bool bad = false;
        FactorOwnerFixed256Block36(pc, count, m, panel, tmp, used, perm, invs, bad);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        return bad;
    }
    return false;
}

template <typename ExtractType42_0, typename ExtractType42_1, typename ExtractType42_2, typename ExtractType42_3,
          typename ExtractType42_4, typename ExtractType42_5, typename ExtractType42_6, typename ExtractType42_7,
          typename ExtractType42_8>
__aicore__ inline void ReceiverFixed256Block42(ExtractType42_0 &pc, ExtractType42_1 &count, ExtractType42_2 &first,
                                               ExtractType42_3 &ac, ExtractType42_4 &m, ExtractType42_5 &panel,
                                               ExtractType42_6 &used, ExtractType42_7 &perm, ExtractType42_8 &invs)
{
    for (uint32_t j = 0; j < count; j++) {
        __ubuf__ float *piv = panel + j * 264;
        __ubuf__ float *bc = m + ((j & 1) ? 38400 : 44096);
        __ubuf__ float *oc = m + ((j & 1) ? 39424 : 38912);
        __ubuf__ float *bn = m + ((j & 1) ? 44096 : 38400);
        __ubuf__ float *on = m + ((j & 1) ? 38912 : 39424);
        set_vector_mask((uint64_t)0, (uint64_t)-1);
        if (j + 1 < count) {
            uint32_t nextRow = uint32_t(int32_t(piv[264 + 256]));
            perm[pc + j + 1] = nextRow;
            invs[pc + j + 1] = piv[264 + 257];
            if (first > pc)
                used[nextRow] = 0;
            uint32_t q0 = nextRow & 192, q1 = (q0 + 64) & 192, q2 = (q0 + 128) & 192, q3 = (q0 + 192) & 192;
            vmla(m + q0, piv + q0, (q0 & 64) ? oc : bc, ac, 1, 1, 0, 33, 0, 1);
            PipeBarrier<PIPE_V>();
            set_vector_mask((uint64_t)0, ((uint64_t)1 << ac) - 1);
            vgather((__ubuf__ uint32_t *)(m + 41024), (__ubuf__ uint32_t *)(m + 45056), nextRow * 4, 8, 1);
            set_vector_mask((uint64_t)0, (uint64_t)-1);
            vmla(m + q1, piv + q1, (q1 & 64) ? oc : bc, ac, 1, 1, 0, 33, 0, 1);
            PipeBarrier<PIPE_V>();
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
            vbrcb((__ubuf__ uint32_t *)bn, (__ubuf__ uint32_t *)(m + 41024), 1, 8, (ac + 7) / 8);
            vbrcb((__ubuf__ uint32_t *)on, (__ubuf__ uint32_t *)(m + 41024), 1, 8, (ac + 7) / 8);
            set_vector_mask((uint64_t)0, (uint64_t)-1);
            vmla(m + q2, piv + q2, (q2 & 64) ? oc : bc, ac, 1, 1, 0, 33, 0, 1);
            vmla(m + q3, piv + q3, (q3 & 64) ? oc : bc, ac, 1, 1, 0, 33, 0, 1);
            PipeBarrier<PIPE_V>();
        } else {
            vmla(m, piv, bc, ac, 1, 1, 0, 33, 0, 1);
            vmla(m + 64, piv + 64, oc, ac, 1, 1, 0, 33, 0, 1);
            vmla(m + 128, piv + 128, bc, ac, 1, 1, 0, 33, 0, 1);
            vmla(m + 192, piv + 192, oc, ac, 1, 1, 0, 33, 0, 1);
            PipeBarrier<PIPE_V>();
        }
    }
}

static __aicore__ __attribute__((noinline)) void ReceiverFixed256(uint32_t pc, uint32_t count, uint32_t first,
                                                                  uint32_t ac)
{
    if ASCEND_IS_AIV {
        auto mem = LocalTensor<float>(TPosition::VECCALC, 0, 48640);
        __ubuf__ float *m = (__ubuf__ float *)0;
        __ubuf__ float *panel = m + 20000;
        __ubuf__ uint32_t *used = (__ubuf__ uint32_t *)(m + 42048);
        __ubuf__ uint32_t *perm = (__ubuf__ uint32_t *)(m + 43072);
        __ubuf__ float *invs = m + 43584;
        uint32_t row = uint32_t(int32_t(panel[256]));
        perm[pc] = row;
        invs[pc] = panel[257];
        if (first > pc)
            used[row] = 0;
        set_vector_mask((uint64_t)0, ((uint64_t)1 << ac) - 1);
        vgather((__ubuf__ uint32_t *)(m + 41024), (__ubuf__ uint32_t *)(m + 45056), row * 4, 8, 1);
        PipeBarrier<PIPE_V>();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        vbrcb((__ubuf__ uint32_t *)(m + 44096), (__ubuf__ uint32_t *)(m + 41024), 1, 8, (ac + 7) / 8);
        vbrcb((__ubuf__ uint32_t *)(m + 38912), (__ubuf__ uint32_t *)(m + 41024), 1, 8, (ac + 7) / 8);
        PipeBarrier<PIPE_V>();
        ReceiverFixed256Block42(pc, count, first, ac, m, panel, used, perm, invs);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
}

#include "pinv_inverse199_impl1.h"

#include "pinv_inverse199_impl2.h"
