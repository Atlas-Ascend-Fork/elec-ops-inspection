/*
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
#ifndef LINALG_EIGH_BASE_S2_H
#define LINALG_EIGH_BASE_S2_H
#include "linalg_eigh_base_s1.h"
namespace AscendC {
template <typename DT_A, u32 SOLVE_MODE = 0u, bool FAST200 = false, bool K96_PASS = false>
class KernelEighBaseS2 : public KernelEighBaseS1<DT_A, SOLVE_MODE, FAST200, K96_PASS> {
public:
    using Base = KernelEighBaseS1<DT_A, SOLVE_MODE, FAST200, K96_PASS>;
    using Base::gm_A, Base::ub_mat, Base::ub_vecs, Base::ub_refl, Base::ub_row, Base::ub_v, Base::ub_w;
    using Base::ub_prod, Base::ub_tmp, Base::ub_red, Base::ub_diag, Base::ub_gather, Base::ub_mask_ge;
    using Base::ub_mask_gt, Base::ComputePow2Scale, Base::AccumulateStripSums, Base::SubTwoBrcbStrips;
    using Base::BacktransformPairShortAligned, Base::BacktransformPairShortUnaligned, Base::ReducePairShortAndBrcb;
    using Base::ReducePairBlockAndOrth, Base::OrthPairVectorInProd, Base::n, Base::stride, Base::uplo;
    using Base::zone_offset, Base::matrix_scale;
    __aicore__ inline KernelEighBaseS2() = default;
    __aicore__ inline void BacktransformPairFast200(LTf matrix_sub, LTf v_k, LTf v_km1, LTf scratch0, LTf scratch1,
    u32 act_rows, u32 k) {
        u32 c_first = k & ~7u, len = 200u - c_first, dual_rows = 2u * act_rows, raw_sc = (len + 7u) & ~7u,
        sc_stride = (raw_sc >= 64u) ? raw_sc : 64u; constexpr u8 rep_stride = 25u; u8 sc_rep_stride =
        u8(sc_stride / 8u), brcb_dual_rep = u8(((dual_rows + 7u) & ~7u) / 8u); LTf brcb0 = ub_gather, brcb1 =
        ub_gather[act_rows * 8u];
        for (u32 c = c_first; c < 200u; c += 64u) {
            u32 width = (200u - c < 64u) ? (200u - c) : 64u, sc = c - c_first; Mul(scratch0[sc], matrix_sub[c],
            v_k[c], width, act_rows, BRP{1, 1, 1, sc_rep_stride, rep_stride, 0}); Mul(scratch1[sc], matrix_sub[c],
            v_km1[c], width, act_rows, BRP{1, 1, 1, sc_rep_stride, rep_stride, 0});
        }
        Mul(ub_prod[c_first], v_k[c_first], v_km1[c_first], len); PipeV();
        if (len <= 64u) {
            ReducePairShortAndBrcb(scratch0, brcb0, c_first, len, dual_rows, sc_rep_stride, brcb_dual_rep);
            BacktransformPairShortAligned(matrix_sub, v_k, v_km1, scratch1, brcb0, brcb1, act_rows, c_first, len,
            rep_stride); return;
        }
        for (u32 c = c_first; c < 200u; c += 64u) {
            u32 width = (200u - c < 64u) ? (200u - c) : 64u; LTf sc = scratch0[c - c_first]; BlockReduceSum(sc, sc,
            i32(dual_rows), i32(width), sc_rep_stride, 1, sc_rep_stride);
        }
        ReducePairBlockAndOrth(v_k, v_km1, scratch0, scratch1, brcb0, c_first, len, dual_rows, sc_rep_stride,
        brcb_dual_rep);
        for (u32 c = c_first; c < 200u; c += 64u) {
            u32 width = (200u - c < 64u) ? (200u - c) : 64u; MulAddDst(matrix_sub[c], scratch1[c - c_first], brcb1,
            width, act_rows, BRP{1, 1, 0, rep_stride, 0, 1});
        }
        PipeV();
        for (u32 c = c_first; c < 200u; c += 64u) {
            u32 width = (200u - c < 64u) ? (200u - c) : 64u; MulAddDst(matrix_sub[c], ub_prod[c], brcb0, width,
            act_rows, BRP{1, 1, 0, rep_stride, 0, 1});
        }
        PipeV();
    }
    __aicore__ inline void ApplyMaskedPairUpdatesM3(LTf matrix_sub, LTf scratch1, LTf brcb0, LTf brcb1, u32 act_rows,
    u32 c_first, u8 rep_stride) {
        SetMaskNorm(); u32 last_bt_mask = 0u;
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u;
            if (width != last_bt_mask) {
                SetVectorMask<float, MaskMode::NORMAL>(width); last_bt_mask = width;
            }
            MulAddDst<float, float, false>(matrix_sub[c], scratch1[c - c_first], brcb1, MASK_PLACEHOLDER, act_rows,
            BRP{1, 1, 0, rep_stride, 0, 1});
        }
        PipeV();
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u;
            if (width != last_bt_mask) {
                SetVectorMask<float, MaskMode::NORMAL>(width); last_bt_mask = width;
            }
            MulAddDst<float, float, false>(matrix_sub[c], ub_prod[c], brcb0, MASK_PLACEHOLDER, act_rows, BRP{1, 1, 0,
            rep_stride, 0, 1});
        }
        SetVectorMask<float, MaskMode::NORMAL>(64u); PipeV();
    }
    __aicore__ inline void FinishPairM3Long(LTf matrix_sub, LTf v_k, LTf v_km1, LTf scratch0, LTf scratch1,
    LTf brcb0, LTf brcb1, u32 act_rows, u32 c_first, u32 len, u32 dual_rows, u8 rep_stride, u8 sc_rep_stride,
    u8 brcb_dual_rep) {
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u; LTf sc = scratch0[c - c_first]; BlockReduceSum(sc, sc,
            i32(dual_rows), i32(width), sc_rep_stride, 1, sc_rep_stride);
        }
        if ((stride & 7u) == 0u) {
            ReducePairBlockAndOrth(v_k, v_km1, scratch0, scratch1, brcb0, c_first, len, dual_rows, sc_rep_stride,
            brcb_dual_rep); ApplyMaskedPairUpdatesM3(matrix_sub, scratch1, brcb0, brcb1, act_rows, c_first,
            rep_stride); return;
        }
        ReduceSum(ub_red, ub_prod[c_first], ub_tmp, len); PipeV(); WholeReduceSum(ub_row, scratch0, sc_rep_stride,
        dual_rows, 1, 8, sc_rep_stride); Brcb(ub_tmp, ub_red, 1u, {1, 8}); PipeV(); OrthPairVectorInProd(v_k, v_km1,
        brcb0, c_first, len, brcb_dual_rep); SubTwoBrcbStrips(matrix_sub, v_km1, ub_prod, scratch1, scratch0, brcb1,
        brcb0, act_rows, c_first, rep_stride, sc_rep_stride);
    }
    __aicore__ inline void BacktransformPairM3(LTf matrix_sub, LTf v_k, LTf v_km1, LTf scratch0, LTf scratch1,
    u32 act_rows, u32 k) {
        u32 c_first = k & ~7u, len = n - c_first, dual_rows = 2u * act_rows, raw_sc = (len + 7u) & ~7u, sc_stride =
        (raw_sc >= 64u) ? raw_sc : ((stride < 64u) ? stride : 64u); u8 rep_stride = u8(stride / 8u), sc_rep_stride =
        u8(sc_stride / 8u), brcb_dual_rep = u8(((dual_rows + 7u) & ~7u) / 8u); LTf brcb0 = ub_gather, brcb1 =
        ub_gather[act_rows * 8u]; SetMaskNorm(); u32 last_mask = 0u;
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u;
            if (width != last_mask) {
                SetVectorMask<float, MaskMode::NORMAL>(width); last_mask = width;
            }
            Mul<float, false>(scratch0[c - c_first], matrix_sub[c], v_k[c], MASK_PLACEHOLDER, act_rows, BRP{1, 1, 1,
            sc_rep_stride, rep_stride, 0}); Mul<float, false>(scratch1[c - c_first], matrix_sub[c], v_km1[c],
            MASK_PLACEHOLDER, act_rows, BRP{1, 1, 1, sc_rep_stride, rep_stride, 0});
        }
        SetVectorMask<float, MaskMode::NORMAL>(64u); Mul(ub_prod[c_first], v_k[c_first], v_km1[c_first], len);
        PipeV();
        if (len <= 64u) {
            ReducePairShortAndBrcb(scratch0, brcb0, c_first, len, dual_rows, sc_rep_stride, brcb_dual_rep);
            ((stride & 7u) == 0u) ? (void)BacktransformPairShortAligned(matrix_sub, v_k, v_km1, scratch1, brcb0,
            brcb1, act_rows, c_first, len, rep_stride) : (void)BacktransformPairShortUnaligned(matrix_sub, v_k,
            v_km1, scratch0, scratch1, brcb0, brcb1, act_rows, c_first, len, sc_stride, rep_stride, sc_rep_stride);
            return;
        }
        FinishPairM3Long(matrix_sub, v_k, v_km1, scratch0, scratch1, brcb0, brcb1, act_rows, c_first, len, dual_rows,
        rep_stride, sc_rep_stride, brcb_dual_rep);
    }
    __aicore__ inline void FinishPairM2Strips(LTf matrix_sub, LTf v_km1, LTf scratch0, LTf scratch1, LTf brcb0,
    LTf brcb1, u32 act_rows, u32 c_first, u32 strip_idx, u32 sc_stride, u8 rep_stride, u8 sc_rep_stride) {
        if (strip_idx < 3u) {
            SubTwoBrcbStrips(matrix_sub, v_km1, ub_prod, scratch1, scratch0, brcb1, brcb0, act_rows, c_first,
            rep_stride, sc_rep_stride); return;
        }
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u, sc = c - c_first; Mul(scratch1[sc], v_km1[c], brcb1, width,
            act_rows, BRP{1, 1, 0, sc_rep_stride, 0, 1}); Mul(scratch0[sc], ub_prod[c], brcb0, width, act_rows,
            BRP{1, 1, 0, sc_rep_stride, 0, 1});
        }
        PipeV(); Add(scratch0, scratch0, scratch1, act_rows * sc_stride); PipeV();
        if (c_first == 0u && n == stride && sc_stride == stride) {
            Sub(matrix_sub, matrix_sub, scratch0, act_rows * stride);
        } else {
            for (u32 c = c_first; c < n; c += 64u) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; Sub(matrix_sub[c], matrix_sub[c], scratch0[c - c_first],
                width, act_rows, BRP{1, 1, 1, rep_stride, rep_stride, sc_rep_stride});
            }
        }
        PipeV();
    }
    __aicore__ inline void BacktransformPairM2(LTf matrix_sub, LTf v_k, LTf v_km1, LTf scratch0, LTf scratch1,
    u32 act_rows, u32 k) {
        u32 c_first = k & ~7u, len = n - c_first, dual_rows = 2u * act_rows, aligned_dual = (dual_rows + 7u) & ~7u,
        raw_sc = (len + 7u) & ~7u, sc_stride = (raw_sc >= 64u) ? raw_sc : ((stride < 64u) ? stride : 64u);
        u8 rep_stride = u8(stride / 8u), sc_rep_stride = u8(sc_stride / 8u), brcb_dual_rep = u8(aligned_dual / 8u);
        LTf brcb0 = ub_gather, brcb1 = ub_gather[act_rows * 8u];
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u, sc = c - c_first; Mul(scratch0[sc], matrix_sub[c], v_k[c],
            width, act_rows, BRP{1, 1, 1, sc_rep_stride, rep_stride, 0}); Mul(scratch1[sc], matrix_sub[c], v_km1[c],
            width, act_rows, BRP{1, 1, 1, sc_rep_stride, rep_stride, 0});
        }
        Mul(ub_prod[c_first], v_k[c_first], v_km1[c_first], len); PipeV(); u32 strip_idx = 0u;
        for (u32 c = c_first; c < n; c += 64u, ++strip_idx) {
            u32 width = (n - c < 64u) ? (n - c) : 64u; LTf dst_r = (strip_idx == 0u) ? ub_row :
            ub_gather[(strip_idx - 1u) * aligned_dual]; WholeReduceSum(dst_r, scratch0[c - c_first], width,
            dual_rows, 1, 1, sc_rep_stride);
        }
        (len <= 64u) ? (void)WholeReduceSum(ub_red, ub_prod[c_first], len, 1, 1, 1, 8) : (void)ReduceSum(ub_red,
        ub_prod[c_first], ub_tmp, len); PipeV(); Brcb(ub_tmp, ub_red, 1u, {1, 8});
        if (strip_idx <= 1u) {
            Brcb(brcb0, ub_row, brcb_dual_rep, {1, 8}); ((SOLVE_MODE == 2u) && ((stride & 15u) == 0u)) ?
            (void)BacktransformPairShortAligned(matrix_sub, v_k, v_km1, scratch1, brcb0, brcb1, act_rows, c_first,
            len, rep_stride) : (void)BacktransformPairShortUnaligned(matrix_sub, v_k, v_km1, scratch0, scratch1,
            brcb0, brcb1, act_rows, c_first, len, sc_stride, rep_stride, sc_rep_stride); return;
        }
        AccumulateStripSums(ub_row, ub_gather, strip_idx, aligned_dual, dual_rows); OrthPairVectorInProd(v_k, v_km1,
        brcb0, c_first, len, brcb_dual_rep); FinishPairM2Strips(matrix_sub, v_km1, scratch0, scratch1, brcb0, brcb1,
        act_rows, c_first, strip_idx, sc_stride, rep_stride, sc_rep_stride);
    }
    __aicore__ inline void BacktransformPairUnitRowsSub(LTf matrix_sub, LTf v_k, LTf v_km1, LTf scratch0,
    LTf scratch1, u32 act_rows, u32 k) {
        if constexpr (FAST200) {
            BacktransformPairFast200(matrix_sub, v_k, v_km1, scratch0, scratch1, act_rows, k);
        } else if constexpr (SOLVE_MODE == 3u) {
            BacktransformPairM3(matrix_sub, v_k, v_km1, scratch0, scratch1, act_rows, k);
        } else {
            BacktransformPairM2(matrix_sub, v_k, v_km1, scratch0, scratch1, act_rows, k);
        }
    }
    __aicore__ inline void BacktransformStepLe64(LTf matrix_sub, LTf v_k, float neg_tau_k, LTf scratch, u32 act_rows,
    u32 c_first, u32 aligned_cnt, u8 rep_stride) {
        u32 width = n - c_first; Mul(scratch, matrix_sub[c_first], v_k[c_first], width, act_rows, BRP{1, 1, 1,
        rep_stride, rep_stride, 0}); Muls(ub_tmp, v_k[c_first], neg_tau_k, width); PipeV(); WholeReduceSum(ub_row,
        scratch, width, act_rows, 1, 1, rep_stride); PipeV(); Brcb(ub_gather, ub_row, u8(aligned_cnt / 8u), {1, 8});
        PipeV();
        if ((stride & 15u) == 0u) {
            MulAddDst(matrix_sub[c_first], ub_tmp, ub_gather, width, act_rows, BRP{1, 1, 0, rep_stride, 0, 1});
        } else {
            Mul(scratch, ub_tmp, ub_gather, width, act_rows, BRP{1, 1, 0, rep_stride, 0, 1}); PipeV();
            Add(matrix_sub[c_first], matrix_sub[c_first], scratch, width, act_rows, BRP{1, 1, 1, rep_stride,
            rep_stride, rep_stride});
        }
        PipeV();
    }
    __aicore__ inline void ApplyBacktransformStepStrips(LTf matrix_sub, LTf scratch, u32 act_rows, u32 c_first,
    u8 rep_stride) {
        if ((stride & 7u) == 0u) {
            for (u32 c = c_first; c < n; c += 64u) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; MulAddDst(matrix_sub[c], ub_tmp[c - c_first], ub_gather,
                width, act_rows, BRP{1, 1, 0, rep_stride, 0, 1});
            }
            PipeV();
        } else {
            for (u32 c = c_first; c < n; c += 64u) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; Mul(scratch[c], ub_tmp[c - c_first], ub_gather, width,
                act_rows, BRP{1, 1, 0, rep_stride, 0, 1});
            }
            PipeV();
            for (u32 c = c_first; c < n; c += 64u) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; Add(matrix_sub[c], matrix_sub[c], scratch[c], width,
                act_rows, BRP{1, 1, 1, rep_stride, rep_stride, rep_stride});
            }
            PipeV();
        }
    }
    __aicore__ inline void BacktransformStepRowsSub(LTf matrix_sub, LTf v_k, float neg_tau_k, LTf scratch,
    u32 act_rows, u32 c_first) {
        if (act_rows == 0u || c_first >= n) {
            return;
        }
        u32 aligned_cnt = (act_rows + 7u) & ~7u; u8 rep_stride = u8(stride / 8u);
        if (n <= 64u) {
            BacktransformStepLe64(matrix_sub, v_k, neg_tau_k, scratch, act_rows, c_first, aligned_cnt, rep_stride);
            return;
        }
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u; Mul(scratch[c], matrix_sub[c], v_k[c], width, act_rows, BRP{1,
            1, 1, rep_stride, rep_stride, 0});
        }
        Muls(ub_tmp, v_k[c_first], neg_tau_k, n - c_first); PipeV(); u32 strip_idx = 0u;
        for (u32 c = c_first; c < n; c += 64u, ++strip_idx) {
            u32 width = (n - c < 64u) ? (n - c) : 64u; LTf dst_r = (strip_idx == 0u) ? ub_row :
            ub_gather[(strip_idx - 1u) * aligned_cnt]; WholeReduceSum(dst_r, scratch[c], width, act_rows, 1, 1,
            rep_stride);
        }
        PipeV(); AccumulateStripSums(ub_row, ub_gather, strip_idx, aligned_cnt, act_rows); Brcb(ub_gather, ub_row,
        u8(aligned_cnt / 8u), {1, 8}); PipeV(); ApplyBacktransformStepStrips(matrix_sub, scratch, act_rows, c_first,
        rep_stride);
    }
    __aicore__ inline void BacktransformPairRowsSub(LTf matrix_sub, LTf v_k, LTf v_km1, float tau_k, float tau_km1,
    LTf scratch0, LTf scratch1, u32 act_rows, u32 k) {
        u32 c_first = k & ~7u, aligned_cnt = (act_rows + 7u) & ~7u, raw_sc = ((n - c_first) + 7u) & ~7u, sc_stride =
        (raw_sc >= 64u) ? raw_sc : ((stride < 64u) ? stride : 64u); u8 rep_stride = u8(stride / 8u), sc_rep_stride =
        u8(sc_stride / 8u), brcb_repeats = u8(aligned_cnt / 8u); LTf sums0 = ub_row, sums1 = ub_row[aligned_cnt];
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u, sc = c - c_first; Mul(scratch0[sc], matrix_sub[c], v_k[c],
            width, act_rows, BRP{1, 1, 1, sc_rep_stride, rep_stride, 0}); Mul(scratch1[sc], matrix_sub[c], v_km1[c],
            width, act_rows, BRP{1, 1, 1, sc_rep_stride, rep_stride, 0});
        }
        Mul(ub_prod[c_first], v_k[c_first], v_km1[c_first], n - c_first); PipeV(); u32 strip_idx = 0u;
        for (u32 c = c_first; c < n; c += 64u, ++strip_idx) {
            u32 width = (n - c < 64u) ? (n - c) : 64u, sc = c - c_first; LTf d0 = (strip_idx == 0u) ? sums0 :
            ub_gather[(strip_idx - 1u) * aligned_cnt], d1 = (strip_idx == 0u) ? sums1 : ub_gather[(strip_idx +
            2u) * aligned_cnt]; WholeReduceSum(d0, scratch0[sc], width, act_rows, 1, 1, sc_rep_stride);
            WholeReduceSum(d1, scratch1[sc], width, act_rows, 1, 1, sc_rep_stride);
        }
        ReduceSum(ub_red, ub_prod[c_first], ub_tmp, n - c_first); PipeV(); Muls(ub_red, ub_red, -tau_km1 * tau_k,
        8u); AccumulateStripSums(sums0, ub_gather, strip_idx, aligned_cnt, act_rows); AccumulateStripSums(sums1,
        ub_gather[3u * aligned_cnt], strip_idx, aligned_cnt, act_rows); Muls(ub_tmp, sums0, tau_k, act_rows);
        Muls(ub_tmp[aligned_cnt], sums1, tau_km1, act_rows); Brcb(ub_tmp[2u * aligned_cnt], ub_red, 1u, {1, 8});
        PipeV(); LTf brcb0 = ub_gather, brcb1 = ub_gather[aligned_cnt * 8u]; Brcb(brcb0, ub_tmp, brcb_repeats, {1,
        8}); Mul(ub_prod, sums0, ub_tmp[2u * aligned_cnt], act_rows, 1u, BRP{1, 1, 0, 8, 8, 0}); PipeV();
        Add(ub_tmp[aligned_cnt], ub_tmp[aligned_cnt], ub_prod, act_rows); Brcb(brcb1, ub_tmp[aligned_cnt],
        brcb_repeats, {1, 8}); PipeV(); SubTwoBrcbStrips(matrix_sub, v_k, v_km1, scratch0, scratch0, brcb1, brcb0,
        act_rows, c_first, rep_stride, sc_rep_stride);
    }
    __aicore__ inline void Rank2UpdateRowsSub(LTf matrix_sub, LTf v_full, LTf w_full, LTf v_sub, LTf w_sub,
    LTf scratch, u32 act_rows, u32 c_first, u32 row_stride = 0u, u32 col_off = 0u) {
        if (act_rows == 0u || c_first >= n) {
            return;
        }
        u32 r_stride = (row_stride != 0u) ? row_stride : stride, raw_sc = ((n - c_first) + 7u) & ~7u, sc_stride =
        (raw_sc >= 64u) ? raw_sc : ((r_stride < 64u) ? r_stride : 64u), aligned_rows = (act_rows + 7u) & ~7u;
        u8 rep_stride = u8(r_stride / 8u), sc_rep_stride = u8(sc_stride / 8u), brcb_repeats = u8(aligned_rows / 8u);
        LTf brcb_v = ub_gather, brcb_w = ub_gather[aligned_rows * 8u]; Brcb(brcb_v, v_sub, brcb_repeats, {1, 8});
        Brcb(brcb_w, w_sub, brcb_repeats, {1, 8}); PipeV();
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u; Mul(scratch[c - c_first], w_full[c], brcb_v, width, act_rows,
            BRP{1, 1, 0, sc_rep_stride, 0, 1});
        }
        PipeV();
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u, mc = c - col_off, sc = c - c_first; Sub(matrix_sub[mc],
            matrix_sub[mc], scratch[sc], width, act_rows, BRP{1, 1, 1, rep_stride, rep_stride, sc_rep_stride});
            Mul(scratch[sc], v_full[c], brcb_w, width, act_rows, BRP{1, 1, 0, sc_rep_stride, 0, 1});
        }
        PipeV();
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u; Sub(matrix_sub[c - col_off], matrix_sub[c - col_off],
            scratch[c - c_first], width, act_rows, BRP{1, 1, 1, rep_stride, rep_stride, sc_rep_stride});
        }
        PipeV();
    }
    __aicore__ inline LTf GetStaircaseReflector(u32 k) const {
        if (n == 256u && k >= 96u) {
            const u32 st_off[7] = {15360u, 17920u, 20224u, 22272u, 24064u, 27136u, 29184u}; const u32 st_row[7] =
            {96u, 112u, 128u, 144u, 160u, 192u, 224u}; u32 idx = (k < 160u) ? ((k - 96u) >> 4u) : (4u + ((k -
            160u) >> 5u)), r0 = st_row[idx]; return ub_mat[st_off[idx] + (k - r0) * (256u - r0) - r0];
        }
        u32 z = k >> 5u, z_row = z << 5u, z_off = (n == 256u) ? ((k >= 24u) ? (zone_offset[z] - 6144u) : 29184u) :
        zone_offset[z]; return ub_mat[z_off + (k - z_row) * (stride - z_row) - z_row];
    }
    __aicore__ inline LTf GetReflector200(u32 k) const {
        const u32 z200_off[11] = {0u, 1600u, 4672u, 7488u, 10048u, 12352u, 14400u, 16192u, 16960u, 19072u, 21120u};
        const u32 z200_row[11] = {0u, 8u, 24u, 40u, 56u, 72u, 88u, 104u, 112u, 136u, 168u}; u32 z = (k < 8u) ? 0u :
        ((k < 104u) ? (1u + ((k - 8u) >> 4u)) : ((k < 112u) ? 7u : ((k < 136u) ? 8u : ((k < 168u) ? 9u : 10u)))),
        z_row = z200_row[z]; return ub_mat[z200_off[z] + (k - z_row) * (200u - z_row) - z_row];
    }
    __aicore__ inline void SymmetrizeFlatUpper(LTu col_idx_buf, u32 n_end8) {
        Sub(ub_tmp, ub_mask_ge, ub_mask_ge, 8u, 8u, BRP{1, 0, 1, 1, 0, 1});
        for (u32 i = 0u; i < n; ++i) {
            u32 c_split = i & ~7u;
            if (c_split > 0u) {
                Gather(ub_mat[i * stride], ub_mat, col_idx_buf, i * 4u, c_split);
            }
            if (c_split < n_end8) {
                Gather(ub_refl[i * 8u], ub_mat[c_split * stride], col_idx_buf, i * 4u, 8u);
            }
        }
        PipeV(); u32 num_g = n_end8 >> 3u; u8 m_rep = u8(stride / 8u);
        for (u32 g = 0u; g < num_g; ++g) {
            u32 c_s = g << 3u; LTf m_blk = ub_mat[c_s * stride + c_s]; Mul(ub_refl[g * 64u], ub_refl[g * 64u],
            ub_tmp, 64u); Mul(m_blk, m_blk, ub_mask_ge, 8u, 8u, BRP{1, 1, 1, m_rep, m_rep, 1});
        }
        PipeV();
        for (u32 g = 0u; g < num_g; ++g) {
            u32 c_s = g << 3u; LTf m_blk = ub_mat[c_s * stride + c_s]; Add(m_blk, m_blk, ub_refl[g * 64u], 8u, 8u,
            BRP{1, 1, 1, m_rep, m_rep, 1});
        }
        if (n_end8 < n) {
            PipeAll();
            for (u32 i = n_end8; i < n; ++i) {
                for (u32 j = n_end8; j < i; ++j) {
                    ub_mat.SetValue(i * stride + j, ub_mat.GetValue(j * stride + i));
                }
            }
        }
    }
    __aicore__ inline void SymmetrizeFlatLower(LTu col_idx_buf, u32 n_end8) {
        Sub(ub_tmp, ub_mask_ge, ub_mask_gt, 8u, 8u, BRP{1, 0, 1, 1, 0, 1});
        for (u32 i = 0u; i < n; ++i) {
            u32 c_split = i & ~7u, c_next8 = c_split + 8u;
            if (c_split < n_end8) {
                Gather(ub_refl[i * 8u], ub_mat[c_split * stride], col_idx_buf, i * 4u, 8u);
            }
            if (c_next8 < n_end8) {
                Gather(ub_mat[i * stride + c_next8], ub_mat[c_next8 * stride], col_idx_buf, i * 4u, n_end8 - c_next8);
            }
        }
        PipeV(); u32 num_g = n_end8 >> 3u; u8 m_rep = u8(stride / 8u);
        for (u32 g = 0u; g < num_g; ++g) {
            u32 c_s = g << 3u; LTf m_blk = ub_mat[c_s * stride + c_s]; Mul(ub_refl[g * 64u], ub_refl[g * 64u],
            ub_mask_gt, 64u); Mul(m_blk, m_blk, ub_tmp, 8u, 8u, BRP{1, 1, 1, m_rep, m_rep, 1});
        }
        PipeV();
        for (u32 g = 0u; g < num_g; ++g) {
            u32 c_s = g << 3u; LTf m_blk = ub_mat[c_s * stride + c_s]; Add(m_blk, m_blk, ub_refl[g * 64u], 8u, 8u,
            BRP{1, 1, 1, m_rep, m_rep, 1});
        }
        if (n_end8 < n) {
            PipeAll();
            for (u32 i = 0u; i < n; ++i) {
                u32 j_start = (i >= n_end8) ? (i + 1u) : n_end8;
                for (u32 j = j_start; j < n; ++j) {
                    ub_mat.SetValue(i * stride + j, ub_mat.GetValue(j * stride + i));
                }
            }
        }
    }
    __aicore__ inline void LoadAndSymmetrizeFlat(u64 batch_base) {
        DataCopyPad(ub_mat, gm_A[batch_base], DCEP{u16(n), n * 4u, 0, 0, 0}, DCPEP{true, 0, u8(stride - n), 0.0f});
        LTu col_idx_buf = ub_diag.template ReinterpretCast<u32>();
        for (u32 j = 0u; j < stride; ++j) {
            col_idx_buf.SetValue(j, j * stride * 4u);
        }
        PipeAll(); u32 n_end8 = n & ~7u; (uplo != 0u) ? (void)SymmetrizeFlatUpper(col_idx_buf, n_end8) :
        (void)SymmetrizeFlatLower(col_idx_buf, n_end8);
        if ((n & 7u) != 0u) {
            this->template fence<HardEvent::S_V>();
        }
        PipeV();
    }
    __aicore__ inline void ReduceMaxChunked(LTf sc_max, u32 total_m, u32 chunk_len) {
        u32 chunk_init = (total_m < chunk_len) ? total_m : chunk_len; Abs(sc_max, ub_mat, chunk_init); u32 prev_len =
        0u;
        if (total_m > chunk_len) {
            prev_len = (total_m - chunk_len < chunk_len) ? (total_m - chunk_len) : chunk_len; Abs(sc_max[chunk_len],
            ub_mat[chunk_len], prev_len);
        }
        PipeV(); u32 buf_sel = 0u;
        for (u32 off = 2u * chunk_len; off < total_m; off += chunk_len, buf_sel ^= 1u) {
            u32 c_len = (total_m - off < chunk_len) ? (total_m - off) : chunk_len; Max(sc_max, sc_max,
            sc_max[(buf_sel == 0u) ? chunk_len : (2u * chunk_len)], prev_len); Abs(sc_max[(buf_sel == 0u) ?
            (2u * chunk_len) : chunk_len], ub_mat[off], c_len); PipeV(); prev_len = c_len;
        }
        if (prev_len > 0u) {
            Max(sc_max, sc_max, sc_max[(buf_sel == 0u) ? chunk_len : (2u * chunk_len)], prev_len); PipeV();
        }
        ReduceMax(ub_red, sc_max, sc_max[chunk_len], chunk_init);
    }
    __aicore__ inline void LoadAndSymmetrizeMedium(u64 b) {
        LoadAndSymmetrizeFlat(b * n * n); Abs(ub_vecs, ub_mat, n * stride); PipeV(); ReduceMax(ub_red, ub_vecs,
        ub_gather, n * stride); Duplicate(ub_v, 0.0f, stride); Duplicate(ub_w, 0.0f, stride);
        if (n < stride) {
            Duplicate(ub_prod, 0.0f, stride);
        }
        this->template fence<HardEvent::V_S>(); float inv_scale = 1.0f; ComputePow2Scale(ub_red.GetValue(0),
        matrix_scale, inv_scale); Muls(ub_mat, ub_mat, inv_scale, n * stride); PipeV();
    }
};
} // namespace AscendC
#endif
