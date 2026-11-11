/*
 * Copyright (c) 2026 gcw_OTnEiQgj.
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
#include "lstsq_p8_lu.h"
#include "lstsq_residual_utils.h"

namespace elec::lstsq_fast {

ELEC_FN void p8_core7_solve_x0(
    uint32_t my_col_start, uint32_t my_w, uint32_t w_p, uint32_t pad_n, const LargeUB& ub,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::GlobalTensor<float>& inv_diag_gm,
    const AscendC::GlobalTensor<float>& x0_gm,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    const AscendC::UnaryRepeatParams m_to_y_rep{1, 1, 1, 5};
    AscendC::Adds(ub.y[0], ub.M[my_w], 0.0f, 8u, 128u, m_to_y_rep);
    AscendC::Adds(ub.y[128u * 8u], ub.M[128u * 40u + my_w], 0.0f, 8u, static_cast<uint8_t>(pad_n - 128u), m_to_y_rep);
    AscendC::Duplicate(ub.lu_stage_pad, 0.0f, 64u);
    AscendC::Duplicate(ub.coeff_pad, 0.0f, 64u);
    fence<AscendC::HardEvent::S_V>();
    AscendC::PipeBarrier<PIPE_V>();

    fence<AscendC::HardEvent::V_MTE2>();
    fence<AscendC::HardEvent::S_MTE2>();
    AscendC::DataCopy(ub.lu_buf0, lu_gm[(32u + 5u * w_p) * pad_n], w_p * pad_n);
    AscendC::DataCopy(ub.inv_diag_vec, inv_diag_gm, my_col_start);
    back_sub_panel_p8(my_col_start, my_w, pad_n, ub.lu_panel7, ub, y_repeat8);
    for (int32_t p = 6; p >= 0; --p) {
        const uint32_t p_col_start = (p == 0) ? 0u : (32u + static_cast<uint32_t>(p - 1) * w_p);
        const uint32_t p_w = (p == 0) ? 32u : w_p;
        auto cur_buf = ((p & 1) == 0) ? ub.lu_buf0 : ub.lu_buf1;
        auto next_buf = ((p & 1) == 0) ? ub.lu_buf1 : ub.lu_buf0;
        fence<AscendC::HardEvent::MTE2_V>();
        if (p >= 1) {
            if (p <= 5) {
                fence<AscendC::HardEvent::V_MTE2>();
            }
            const uint32_t prev_col_start = (p - 1 == 0) ? 0u : (32u + static_cast<uint32_t>(p - 2) * w_p);
            const uint32_t prev_w = (p - 1 == 0) ? 32u : w_p;
            AscendC::DataCopy(next_buf, lu_gm[prev_col_start * pad_n], prev_w * pad_n);
        }
        back_sub_panel_p8(p_col_start, p_w, pad_n, cur_buf, ub, y_repeat8);
    }
    AscendC::Brcb(ub.lu_stage, ub.inv_diag_vec, static_cast<uint8_t>(pad_n >> 3), {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(ub.x0, ub.y, ub.lu_stage, pad_n * 8u);
    fence<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopy(x0_gm, ub.x0, pad_n * 8u);
}

ELEC_FN void p8_init_residual_buffers(
    uint32_t sub_id, uint32_t k, uint32_t pad_n, const LargeUB& ub,
    const AscendC::LocalTensor<int32_t>& split_mask_i32,
    const AscendC::LocalTensor<uint16_t>& split_mask_u16) {
    AscendC::Duplicate(split_mask_i32, static_cast<int32_t>(0xfffff000u), pad_n);
    if (sub_id < 7u) {
        AscendC::Duplicate(ub.s_hi, 0.0f, 2048u);
    } else {
        AscendC::Duplicate(ub.s_hi[1024], 0.0f, 1024u);
    }
    AscendC::Duplicate(ub.s_lo, 0.0f, k * pad_n);
    residual_prepare_col0(pad_n, ub.A_orig, ub.col_offsets_u32, split_mask_u16, ub.col_vec, ub.abs_col, ub.pivot_row);
}

template <bool SMALL_TOTAL_CONST>
ELEC_FN void p8_phase4a_accum(
    uint32_t p4_col_start, uint32_t p4_w, uint32_t k, uint32_t pad_n, const LargeUB& ub,
    const AscendC::LocalTensor<uint16_t>& split_mask_u16) {
    auto col_vec1 = ub.y_col_offsets.template ReinterpretCast<float>();
    for (uint32_t rhs_base = 0; rhs_base < k; rhs_base += 4u) {
        const uint32_t cur_k = (k - rhs_base < 4u) ? (k - rhs_base) : 4u, len = cur_k * pad_n;
        auto s_hi_chunk = ub.s_hi[rhs_base * pad_n], s_lo_chunk = ub.s_lo[rhs_base * pad_n];
        auto s_lo_acc2 = ub.s_hi[1024], p_hi_alt_4 = ub.s_lo[1056];
        if (!SMALL_TOTAL_CONST && rhs_base > 0u) {
            residual_prepare_col0(pad_n, ub.A_orig, ub.col_offsets_u32, split_mask_u16, ub.col_vec, ub.abs_col, ub.pivot_row);
        }
        if (SMALL_TOTAL_CONST || k <= 4u) {
            for (uint32_t step = 0; step < p4_w; step += 2u) {
                const uint32_t col0 = p4_col_start + step;
                residual_step_k4_3wave(
                    step + 0u, p4_w, cur_k, pad_n, len, &ub.x0_ptr[(col0 + 0u) * 8u + rhs_base],
                    ub.A_orig, ub.col_offsets_u32, split_mask_u16, ub.col_vec, col_vec1,
                    s_hi_chunk, ub.t_sum_4, s_lo_chunk, s_lo_acc2, ub.lu_stage[0], p_hi_alt_4, ub.lu_stage[1056],
                    ub.abs_col, ub.row_tmp, ub.pivot_row, ub.scratch, ub.y[0], ub.y[1056]);
                residual_step_k4_3wave(
                    step + 1u, p4_w, cur_k, pad_n, len, &ub.x0_ptr[(col0 + 1u) * 8u + rhs_base],
                    ub.A_orig, ub.col_offsets_u32, split_mask_u16, col_vec1, ub.col_vec,
                    ub.t_sum_4, s_hi_chunk, s_lo_chunk, s_lo_acc2, p_hi_alt_4, ub.lu_stage[0], ub.lu_stage[1056],
                    ub.row_tmp, ub.abs_col, ub.scratch, ub.pivot_row, ub.y[0], ub.y[1056]);
            }
            AscendC::Sub(ub.y[1056], s_hi_chunk, ub.y[0], len);
            AscendC::Sub(p_hi_alt_4, p_hi_alt_4, ub.y[0], len);
            AscendC::Add(s_lo_chunk, s_lo_chunk, s_lo_acc2, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(ub.y[1056], ub.t_sum_4, ub.y[1056], len);
            AscendC::Add(s_lo_chunk, s_lo_chunk, p_hi_alt_4, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(ub.s_hi[len], s_lo_chunk, ub.y[1056], len);
            AscendC::PipeBarrier<PIPE_V>();
        } else if constexpr (!SMALL_TOTAL_CONST) {
            residual_accum_general(
                p4_col_start, p4_w, rhs_base, cur_k, pad_n, len, ub.x0_ptr,
                ub.A_orig, ub.col_offsets_u32, split_mask_u16, ub.col_vec,
                s_hi_chunk, ub.t_sum_4, s_lo_chunk, ub.lu_stage[0], ub.lu_stage[1056],
                ub.abs_col, ub.pivot_row, ub.y[0], ub.y[1056]);
        }
    }
}

template <bool SMALL_TOTAL_CONST>
ELEC_FN void p8_store_partial_residual(
    uint32_t dst_sub, uint32_t total_len,
    const AscendC::GlobalTensor<float>& r_gm,
    const AscendC::LocalTensor<float>& s_hi,
    const AscendC::LocalTensor<float>& s_lo) {
    fence<AscendC::HardEvent::V_MTE3>();
    if (SMALL_TOTAL_CONST || total_len <= 1024u) {
        AscendC::DataCopy(r_gm[dst_sub * 2048u], s_hi, 2u * total_len);
    } else {
        AscendC::DataCopy(r_gm[dst_sub * 4096u], s_hi, total_len);
        AscendC::DataCopy(r_gm[dst_sub * 4096u + 2048u], s_lo, total_len);
    }
    fence<AscendC::HardEvent::MTE3_V>();
}

ELEC_FN void p8_refine_solve_w24(
    uint32_t n, uint32_t pad_n, uint32_t w_p, uint32_t my_col_start, uint32_t my_w,
    const uint16_t* piv, const LargeUB& ub, const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    fence<AscendC::HardEvent::V_MTE2>();
    AscendC::DataCopy(ub.A_orig, lu_gm[(32u + w_p) * pad_n], w_p * pad_n);
    fwd_sub_panel(0u, 32u, n, pad_n, piv, ub.lu_buf0, ub.y, ub.row_tmp, ub.coefficients, ub.lu_stage, y_repeat8);
    fence<AscendC::HardEvent::V_MTE2>();
    AscendC::DataCopy(ub.lu_buf0, lu_gm[(32u + 2u * w_p) * pad_n], w_p * pad_n);
    fwd_sub_panel(32u, w_p, n, pad_n, piv, ub.lu_buf1, ub.y, ub.row_tmp, ub.coefficients, ub.lu_stage, y_repeat8);
    for (uint32_t p = 1u; p <= 3u; ++p) {
        const auto& cur_buf = (p == 1u) ? ub.A_orig : ((p == 2u) ? ub.lu_buf0 : ub.lu_buf1);
        const auto& load_buf = (p == 1u) ? ub.lu_buf1 : ((p == 2u) ? ub.A_orig : ub.lu_buf0);
        fence<AscendC::HardEvent::MTE2_V>();
        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(load_buf, lu_gm[(32u + (p + 2u) * w_p) * pad_n], w_p * pad_n);
        fwd_sub_panel(32u + p * w_p, w_p, n, pad_n, piv, cur_buf, ub.y, ub.row_tmp, ub.coefficients, ub.lu_stage, y_repeat8);
    }
    fence<AscendC::HardEvent::MTE2_V>();
    fwd_sub_panel(32u + 4u * w_p, w_p, n, pad_n, piv, ub.A_orig, ub.y, ub.row_tmp, ub.coefficients, ub.lu_stage, y_repeat8);
    fence<AscendC::HardEvent::MTE2_V>();
    fwd_sub_panel(32u + 5u * w_p, w_p, n, pad_n, piv, ub.lu_buf0, ub.y, ub.row_tmp, ub.coefficients, ub.lu_stage, y_repeat8);
    fwd_sub_panel(my_col_start, my_w, n, pad_n, piv, ub.lu_panel7, ub.y, ub.row_tmp, ub.coefficients, ub.lu_stage, y_repeat8);

    back_sub_panel_p8(my_col_start, my_w, pad_n, ub.lu_panel7, ub, y_repeat8);
    fence<AscendC::HardEvent::V_MTE2>();
    AscendC::DataCopy(ub.lu_panel7, lu_gm[(32u + 2u * w_p) * pad_n], w_p * pad_n);
    back_sub_panel_p8(32u + 5u * w_p, w_p, pad_n, ub.lu_buf0, ub, y_repeat8);
    fence<AscendC::HardEvent::V_MTE2>();
    AscendC::DataCopy(ub.lu_buf0, lu_gm[(32u + w_p) * pad_n], w_p * pad_n);
    back_sub_panel_p8(32u + 4u * w_p, w_p, pad_n, ub.A_orig, ub, y_repeat8);
    fence<AscendC::HardEvent::V_MTE2>();
    AscendC::DataCopy(ub.A_orig, lu_gm[32u * pad_n], w_p * pad_n);
    back_sub_panel_p8(32u + 3u * w_p, w_p, pad_n, ub.lu_buf1, ub, y_repeat8);
    fence<AscendC::HardEvent::MTE2_V>();
    fence<AscendC::HardEvent::V_MTE2>();
    AscendC::DataCopy(ub.lu_buf1, lu_gm[0], 32u * pad_n);
    back_sub_panel_p8(32u + 2u * w_p, w_p, pad_n, ub.lu_panel7, ub, y_repeat8);
    back_sub_panel_p8(32u + w_p, w_p, pad_n, ub.lu_buf0, ub, y_repeat8);
    back_sub_panel_p8(32u, w_p, pad_n, ub.A_orig, ub, y_repeat8);
    fence<AscendC::HardEvent::MTE2_V>();
    back_sub_panel_p8(0u, 32u, pad_n, ub.lu_buf1, ub, y_repeat8);
}

ELEC_FN void p8_refine_solve_w32(
    uint32_t n, uint32_t pad_n, uint32_t w_p, uint32_t my_col_start, uint32_t my_w,
    const uint16_t* piv, const LargeUB& ub, const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    fwd_sub_panel(0u, 32u, n, pad_n, piv, ub.lu_buf0, ub.y, ub.row_tmp, ub.coefficients, ub.lu_stage, y_repeat8);
    fence<AscendC::HardEvent::V_MTE2>();
    AscendC::DataCopy(ub.lu_buf0, lu_gm[(32u + w_p) * pad_n], w_p * pad_n);
    fwd_sub_panel(32u, w_p, n, pad_n, piv, ub.lu_buf1, ub.y, ub.row_tmp, ub.coefficients, ub.lu_stage, y_repeat8);
    for (uint32_t p = 2u; p < 7u; ++p) {
        const uint32_t p_col_start = 32u + (p - 1u) * w_p;
        auto cur_buf = ((p & 1u) == 0u) ? ub.lu_buf0 : ub.lu_buf1;
        auto next_buf = ((p & 1u) == 0u) ? ub.lu_buf1 : ub.lu_buf0;
        fence<AscendC::HardEvent::MTE2_V>();
        if (p + 1u < 7u) {
            fence<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopy(next_buf, lu_gm[(p_col_start + w_p) * pad_n], w_p * pad_n);
        }
        fwd_sub_panel(p_col_start, w_p, n, pad_n, piv, cur_buf, ub.y, ub.row_tmp, ub.coefficients, ub.lu_stage, y_repeat8);
    }
    fwd_sub_panel(my_col_start, my_w, n, pad_n, piv, ub.lu_panel7, ub.y, ub.row_tmp, ub.coefficients, ub.lu_stage, y_repeat8);

    back_sub_panel_p8(my_col_start, my_w, pad_n, ub.lu_panel7, ub, y_repeat8);
    back_sub_panel_p8(32u + 5u * w_p, w_p, pad_n, ub.lu_buf0, ub, y_repeat8);
    for (int32_t p = 5; p >= 0; --p) {
        const uint32_t p_col_start = (p == 0) ? 0u : (32u + static_cast<uint32_t>(p - 1) * w_p);
        const uint32_t p_w = (p == 0) ? 32u : w_p;
        auto cur_buf = ((p & 1) == 0) ? ub.lu_buf0 : ub.lu_buf1;
        auto next_buf = ((p & 1) == 0) ? ub.lu_buf1 : ub.lu_buf0;
        if (p <= 4) {
            fence<AscendC::HardEvent::MTE2_V>();
        }
        if (p >= 1) {
            fence<AscendC::HardEvent::V_MTE2>();
            const uint32_t prev_col_start = (p - 1 == 0) ? 0u : (32u + static_cast<uint32_t>(p - 2) * w_p);
            const uint32_t prev_w = (p - 1 == 0) ? 32u : w_p;
            AscendC::DataCopy(next_buf, lu_gm[prev_col_start * pad_n], prev_w * pad_n);
        }
        back_sub_panel_p8(p_col_start, p_w, pad_n, cur_buf, ub, y_repeat8);
    }
}

template <uint32_t PANEL_MODE, bool SMALL_TOTAL_CONST>
ELEC_FN void p8_core7_finish_refinement(
    uint32_t n, uint32_t k, uint32_t pad_n, uint32_t w_p, uint32_t my_col_start, uint32_t my_w, uint32_t total_len,
    const uint16_t* piv, const LargeUB& ub,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::GlobalTensor<float>& r_gm,
    const AscendC::LocalTensor<float>& s_lo_red,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    two_sum_combine_pair_from_gm<SMALL_TOTAL_CONST>(4u, 5u, total_len, r_gm, ub.s_hi, s_lo_red, ub.t_sum_4, ub.lu_stage, ub.A_orig, ub.y);
    AscendC::Add(ub.s_hi, ub.s_hi, s_lo_red, total_len);
    AscendC::Duplicate(ub.y, 0.0f, pad_n * 8u);
    AscendC::PipeBarrier<PIPE_V>();
    if (SMALL_TOTAL_CONST || k <= 5u) {
        AscendC::Brcb(ub.A_orig, ub.s_hi, static_cast<uint8_t>(total_len >> 3), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t t = 0; t < k; ++t) {
            uint64_t lane_mask[2] = {0x0101010101010101ULL << t, 0ULL};
            AscendC::Adds(ub.y, ub.A_orig[t * pad_n * 8u], 0.0f, lane_mask, static_cast<uint8_t>(pad_n >> 3), {1, 1, 8, 8});
        }
        AscendC::PipeBarrier<PIPE_V>();
    } else if constexpr (!SMALL_TOTAL_CONST) {
        const AscendC::BinaryRepeatParams lane_rep{1, 1, 0, 8, 8, 0};
        for (uint32_t t = 0; t < k; ++t) {
            AscendC::Brcb(ub.coefficients, ub.s_hi[t * pad_n], static_cast<uint8_t>(pad_n >> 3), {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(ub.y, ub.coefficients, ub.unit_masks[t * 8u], 64u, static_cast<uint8_t>(pad_n >> 3), lane_rep);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
    if constexpr (PANEL_MODE != 2u) {
        p8_refine_solve_w24(n, pad_n, w_p, my_col_start, my_w, piv, ub, lu_gm, y_repeat8);
    } else {
        p8_refine_solve_w32(n, pad_n, w_p, my_col_start, my_w, piv, ub, lu_gm, y_repeat8);
    }
    AscendC::Brcb(ub.lu_stage, ub.inv_diag_vec, static_cast<uint8_t>(pad_n >> 3), {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(ub.y, ub.y, ub.lu_stage, pad_n * 8u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(ub.x0, ub.x0, ub.y, pad_n * 8u);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
ELEC_FN void p8_reload_a_orig(
    uint32_t sub_id, uint32_t batch, uint32_t n, uint32_t my_w, uint32_t p4_col_start, uint32_t p4_w, bool rebal_p03,
    const AscendC::DataCopyParams& a_copy,
    const AscendC::GlobalTensor<T>& input_a,
    const LargeUB& ub) {
    if constexpr (sizeof(T) == 4) {
        if (rebal_p03 && sub_id < 4u) {
            fence<AscendC::HardEvent::V_MTE2>();
            const AscendC::DataCopyExtParams a_p4_load{
                static_cast<uint16_t>(n), static_cast<uint32_t>(p4_w * sizeof(float)),
                static_cast<uint32_t>((n - p4_w) * sizeof(float)),
                static_cast<uint32_t>((40u - ((p4_w + 7u) & ~7u)) >> 3), 0};
            const AscendC::DataCopyPadExtParams<float> no_pad{false, 0, 0, 0};
            AscendC::DataCopyPad(ub.A_orig, input_a[batch * n * n + p4_col_start], a_p4_load, no_pad);
            fence<AscendC::HardEvent::MTE2_V>();
        } else if (sub_id == 0u) {
            AscendC::DataCopy(ub.A_orig, input_a[batch * n * n], a_copy);
            fence<AscendC::HardEvent::MTE2_V>();
        }
    } else if (sub_id == 0u) {
        load_panel_a<T>(n, batch, 0u, my_w, my_w, 40u, input_a, ub.A_orig, ub.lu_stage);
    }
}

template <typename T, uint32_t PANEL_MODE, bool SMALL_TOTAL_CONST>
ELEC_FN void p8_reduce_and_solve(
    uint32_t sub_id, uint32_t batch, uint32_t n, uint32_t k, uint32_t pad_n, uint32_t w_p,
    uint32_t my_col_start, uint32_t my_w, const uint16_t* piv, const LargeUB& ub,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::GlobalTensor<float>& r_gm,
    const AscendC::GlobalTensor<T>& output_x,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    const uint32_t total_len = k * pad_n;
    auto s_lo_red = (SMALL_TOTAL_CONST || total_len <= 1024u) ? ub.s_hi[total_len] : ub.s_lo;
    if (sub_id < 4u || sub_id == 6u) {
        p8_store_partial_residual<SMALL_TOTAL_CONST>(sub_id, total_len, r_gm, ub.s_hi, ub.s_lo);
    }
    AscendC::SyncAll<true>();
    if (sub_id == 4u || sub_id == 5u) {
        const uint32_t s0 = (sub_id == 4u) ? 0u : 2u;
        two_sum_combine_pair_from_gm<SMALL_TOTAL_CONST>(s0, s0 + 1u, total_len, r_gm, ub.s_hi, s_lo_red, ub.t_sum_4, ub.lu_stage, ub.A_orig, ub.y);
        p8_store_partial_residual<SMALL_TOTAL_CONST>(sub_id, total_len, r_gm, ub.s_hi, ub.s_lo);
    } else if (sub_id == 7u) {
        two_sum_combine_from_gm<SMALL_TOTAL_CONST>(6u, total_len, r_gm, ub.s_hi, s_lo_red, ub.t_sum_4, ub.lu_stage, ub.y);
        AscendC::Duplicate(ub.lu_stage_pad, 0.0f, 64u);
        AscendC::Duplicate(ub.coeff_pad, 0.0f, 64u);
    }
    AscendC::SyncAll<true>();
    if (sub_id == 7u) {
        p8_core7_finish_refinement<PANEL_MODE, SMALL_TOTAL_CONST>(
            n, k, pad_n, w_p, my_col_start, my_w, total_len, piv, ub, lu_gm, r_gm, s_lo_red, y_repeat8);
        store_square_solution<T>(batch, n, k, output_x, ub.x0, ub.x_out, ub.lu_stage, ub.lu_buf1.template ReinterpretCast<int32_t>());
    }
}

template <typename T, uint32_t PANEL_MODE, bool SMALL_TOTAL_CONST>
ELEC_FN void p8_post_wavefront_and_accum(
    const __gm__ Config* c_ptr, uint32_t sub_id, uint32_t batch, uint32_t n, uint32_t k, uint32_t pad_n, uint32_t w_p,
    uint32_t my_col_start, uint32_t my_w, uint32_t row_w, uint16_t* piv, const LargeUB& ub,
    GM_ADDR residuals, const AscendC::DataCopyParams& a_copy, const P8GlobalTensors<T>& gm,
    const AscendC::BinaryRepeatParams& m_p8_repeat, const AscendC::BinaryRepeatParams& y_repeat8) {
    const bool rebal_p03 = (sizeof(T) == 4 && w_p == 24u);
    const uint32_t p4_col_start = (rebal_p03 && sub_id < 4u) ? (sub_id * 26u) : my_col_start;
    const uint32_t p4_w = (rebal_p03 && sub_id < 4u) ? 26u : my_w;
    p8_reload_a_orig<T>(sub_id, batch, n, my_w, p4_col_start, p4_w, rebal_p03, a_copy, gm.input_a, ub);
    if (sub_id == 0u && batch == 0u && residuals != nullptr && (c_ptr->flags & LSTSQ_SQUARE_RESIDUAL)) {
        fence<AscendC::HardEvent::V_MTE3>();
        const uint32_t r_batches = (PANEL_MODE != 0u) ? (AscendC::GetBlockNum() >> 3) : static_cast<uint32_t>(c_ptr->batch);
        const AscendC::DataCopyExtParams r_copy{1, static_cast<uint32_t>(r_batches * k * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(gm.output_r, ub.unit_masks.template ReinterpretCast<T>(), r_copy);
    }
    if (sub_id == 7u) {
        factor_chunk_p8(my_col_start, my_w, 7u, n, pad_n, my_col_start, row_w, piv, ub,
                        gm.lu_gm, gm.piv_gm, gm.inv_diag_gm, ub.lu_panel7, m_p8_repeat);
        p8_core7_solve_x0(my_col_start, my_w, w_p, pad_n, ub, gm.lu_gm, gm.inv_diag_gm, gm.x0_gm, y_repeat8);
    }
    auto split_mask_i32 = ub.lu_col.template ReinterpretCast<int32_t>();
    auto split_mask_u16 = split_mask_i32.template ReinterpretCast<uint16_t>();
    if (PANEL_MODE != 2u || sub_id < 7u || w_p != 32u) {
        p8_init_residual_buffers(sub_id, k, pad_n, ub, split_mask_i32, split_mask_u16);
    }
    if (sub_id == 7u) {
        fence<AscendC::HardEvent::V_S>();
        fence<AscendC::HardEvent::MTE3_V>();
    }
    AscendC::SyncAll<true>();
    if (sub_id < 7u) {
        fence<AscendC::HardEvent::V_MTE2>();
        fence<AscendC::HardEvent::S_MTE2>();
        AscendC::DataCopy(ub.x0[p4_col_start * 8u], gm.x0_gm[p4_col_start * 8u], p4_w * 8u);
        fence<AscendC::HardEvent::MTE2_S>();
    } else if (PANEL_MODE == 2u && w_p == 32u) {
        p8_init_residual_buffers(sub_id, k, pad_n, ub, split_mask_i32, split_mask_u16);
    }
    p8_phase4a_accum<SMALL_TOTAL_CONST>(p4_col_start, p4_w, k, pad_n, ub, split_mask_u16);
}

template <typename T, uint32_t PANEL_MODE = 0u>
__aicore__ __attribute__((noinline)) int solve_square_large_p8(
    const __gm__ Config* c_ptr, GM_ADDR A, GM_ADDR B, GM_ADDR solution, GM_ADDR residuals,
    GM_ADDR ws_base, uint64_t per_core) {
    constexpr uint32_t W_P_CONST = (PANEL_MODE == 2u) ? 32u : 24u;
    constexpr uint32_t N_UNIFORM = 32u + 7u * W_P_CONST;
    constexpr uint32_t K_REG = (PANEL_MODE == 1u) ? 4u : 3u;
    const uint32_t n = (PANEL_MODE != 0u) ? N_UNIFORM : static_cast<uint32_t>(c_ptr->n);
    const uint32_t k = (PANEL_MODE != 0u) ? K_REG : static_cast<uint32_t>(c_ptr->k);
    const uint32_t pad_n = (PANEL_MODE != 0u) ? N_UNIFORM : ((n + 7u) & ~7u);
    const uint32_t block_idx = AscendC::GetBlockIdx();
    const uint32_t batch = block_idx >> 3;
    const uint32_t sub_id = block_idx & 7u;
    if constexpr (PANEL_MODE == 0u) {
        if (batch >= c_ptr->batch) return SUCCESS;
    }
    const uint32_t w_p = (PANEL_MODE != 0u) ? W_P_CONST : ((n >= 232u) ? 32u : 24u);
    const uint32_t my_col_start = (sub_id == 0u) ? 0u : (32u + (sub_id - 1u) * w_p);
    const uint32_t my_w = (sub_id == 0u) ? 32u : ((PANEL_MODE != 0u || sub_id < 7u) ? w_p : (n - my_col_start));
    const uint32_t row_w = (sub_id < 7u) ? my_w : (my_w + 8u);
    constexpr bool SMALL_TOTAL_CONST = (PANEL_MODE != 0u);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_buf;
    LargeUB ub;
    init_large_ub(ub, pipe, work_buf);
    P8GlobalTensors<T> gm;
    init_p8_gm_tensors(gm, batch, per_core, A, B, solution, residuals, ws_base);

    const AscendC::DataCopyParams a_copy{
        static_cast<uint16_t>(n), static_cast<uint16_t>(my_w >> 3),
        static_cast<uint16_t>((n - my_w) >> 3), static_cast<uint16_t>((40u - my_w) >> 3)};
    const AscendC::BinaryRepeatParams m_p8_repeat{1, 1, 0, 5, 0, 1};
    const AscendC::BinaryRepeatParams y_repeat8{1, 0, 1, 8, 0, 8};
    uint16_t piv[256];
    const uint64_t rhs_batch_stride = SMALL_TOTAL_CONST ? static_cast<uint64_t>(n * k) : c_ptr->rhs_batch_stride;

    p8_prepare_and_wavefront<T, SMALL_TOTAL_CONST>(
        sub_id, batch, n, k, pad_n, w_p, my_col_start, my_w, row_w, static_cast<uint64_t>(batch) * rhs_batch_stride, piv, ub,
        a_copy, gm.input_a, gm.input_b, gm.lu_gm, gm.piv_gm, gm.inv_diag_gm, m_p8_repeat);
    p8_post_wavefront_and_accum<T, PANEL_MODE, SMALL_TOTAL_CONST>(
        c_ptr, sub_id, batch, n, k, pad_n, w_p, my_col_start, my_w, row_w, piv, ub, residuals,
        a_copy, gm, m_p8_repeat, y_repeat8);
    p8_reduce_and_solve<T, PANEL_MODE, SMALL_TOTAL_CONST>(
        sub_id, batch, n, k, pad_n, w_p, my_col_start, my_w, piv, ub, gm.lu_gm, gm.r_gm, gm.output_x, y_repeat8);
    return SUCCESS;
}

}  // namespace elec::lstsq_fast
