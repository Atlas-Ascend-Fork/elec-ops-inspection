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
#include "lstsq_residual_utils.h"

namespace elec::lstsq_fast {

ELEC_FN int large_factor_find_pivot(
    uint32_t j, uint32_t local_col, uint32_t n, uint32_t pad_n, uint32_t& pivot,
    const LargeUB& ub) {
    AscendC::Gather(ub.col_vec, ub.M[local_col], ub.col_offsets_u32, uint32_t(0), n);
    AscendC::PipeBarrier<PIPE_V>();
    const uint32_t j0 = j & ~7u;
    reduce_max_abs_col_tail(j, j0, n, pad_n, ub);
    fence<AscendC::HardEvent::V_S>();
    const uint64_t red_val = ub.reduced_u64_ptr[0];
    const uint32_t max_bits = static_cast<uint32_t>(red_val);
    pivot = j0 + static_cast<uint32_t>(red_val >> 32);
    if (max_bits == 0u) return SINGULAR;
    if (max_bits >= 0x7f800000u || pivot >= n) return NUMERIC_ERROR;
    return SUCCESS;
}

ELEC_FN void large_swap_and_scale_col(
    uint32_t j, uint32_t pivot, uint32_t slot, uint32_t n, uint32_t pad_n, uint32_t m_stride_lu,
    bool has_rem_cols, uint32_t c0, uint32_t rem, const LargeUB& ub) {
    const uint32_t j0 = j & ~7u;
    auto lu_col_slot = ub.lu_stage[slot * pad_n];
    if (has_rem_cols) {
        AscendC::Adds(ub.pivot_row, ub.M[pivot * m_stride_lu + c0], 0.0f, rem);
    }
    if (pivot != j) {
        if (has_rem_cols) {
            AscendC::Adds(ub.row_tmp, ub.M[j * m_stride_lu + c0], 0.0f, rem);
        }
        AscendC::Adds(ub.scratch[0], ub.y[pivot * 8u], 0.0f, 8u);
        AscendC::Adds(ub.scratch[8], ub.y[j * 8u], 0.0f, 8u);
    }
    const float diag = ub.col_vec_ptr[pivot];
    const float inv_diag = (diag != 0.0f) ? (1.0f / diag) : 0.0f;
    if (pivot != j) {
        ub.col_vec_ptr[pivot] = ub.col_vec_ptr[j];
    }
    ub.col_vec_ptr[j] = 0.0f;
    ub.abs_col_ptr[j] = -1.0f;
    ub.inv_diag_ptr[j] = inv_diag;
    fence<AscendC::HardEvent::S_V>();
    AscendC::Muls(lu_col_slot, ub.col_vec, -inv_diag, pad_n);
    AscendC::PipeBarrier<PIPE_V>();

    if (j + 1u < n) {
        AscendC::Brcb(ub.coefficients[j0 * 8u], lu_col_slot[j0], static_cast<uint8_t>((pad_n - j0) >> 3), {1, 8});
    }
    if (pivot != j) {
        if (has_rem_cols) {
            AscendC::Adds(ub.M[pivot * m_stride_lu + c0], ub.row_tmp, 0.0f, rem);
            AscendC::Adds(ub.M[j * m_stride_lu + c0], ub.pivot_row, 0.0f, rem);
        }
        AscendC::Adds(ub.y[j * 8u], ub.scratch[0], 0.0f, 8u);
        AscendC::Adds(ub.y[pivot * 8u], ub.scratch[8], 0.0f, 8u);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

ELEC_FN int large_factor_panel_cols(
    uint32_t col_base, uint32_t panel_w, uint32_t panel_w_pad, uint32_t n, uint32_t pad_n, uint32_t m_stride_lu,
    uint16_t* piv, const LargeUB& ub,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::BinaryRepeatParams& m_lu_repeat,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    for (uint32_t step = 0; step < panel_w; ++step) {
        const uint32_t j = col_base + step;
        uint32_t pivot = j;
        const int status = large_factor_find_pivot(j, step, n, pad_n, pivot, ub);
        if (status != SUCCESS) return status;
        piv[j] = static_cast<uint16_t>(pivot);

        const uint32_t slot = step & 7u;
        const bool has_rem_cols = (step + 1u < panel_w);
        const uint32_t c0 = (step + 1u) & ~7u;
        const uint32_t rem = has_rem_cols ? (panel_w_pad - c0) : 0u;
        large_swap_and_scale_col(j, pivot, slot, n, pad_n, m_stride_lu, has_rem_cols, c0, rem, ub);

        if (j + 1u < n) {
            const uint32_t sub_rows = n - 1u - j;
            for (uint32_t offset = 0; offset < rem; offset += 64u) {
                const uint64_t lanes = rem - offset < 64u ? rem - offset : 64u;
                AscendC::MulAddDst(ub.M[(j + 1u) * m_stride_lu + c0 + offset], ub.pivot_row[offset],
                                   ub.coefficients[(j + 1u) * 8u], lanes, static_cast<uint8_t>(sub_rows), m_lu_repeat);
            }
            AscendC::MulAddDst(ub.y[(j + 1u) * 8u], ub.y[j * 8u], ub.coefficients[(j + 1u) * 8u],
                               64u, static_cast<uint8_t>((sub_rows + 7u) >> 3), y_repeat8);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (slot == 7u || step + 1u == panel_w) {
            fence<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(lu_gm[(j - slot) * pad_n], ub.lu_stage, (slot + 1u) * pad_n);
            fence<AscendC::HardEvent::MTE3_V>();
        }
    }
    fence<AscendC::HardEvent::MTE3_MTE2>();
    return SUCCESS;
}

ELEC_FN void large_apply_panel0_to_panel1(
    uint32_t n, uint32_t pad_n, uint32_t w1_lu_pad, uint32_t m_stride_lu,
    const uint16_t* piv, const LargeUB& ub,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::BinaryRepeatParams& m_lu_repeat) {
    for (uint32_t chunk = 0; chunk < 128u; chunk += 8u) {
        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(ub.lu_stage, lu_gm[chunk * pad_n], 8u * pad_n);
        fence<AscendC::HardEvent::MTE2_V>();
        for (uint32_t s = 0; s < 8u; ++s) {
            const uint32_t j = chunk + s;
            const uint32_t j0 = j & ~7u;
            const uint32_t pivot = piv[j];
            AscendC::Adds(ub.pivot_row, ub.M[pivot * m_stride_lu], 0.0f, w1_lu_pad);
            if (pivot != j) {
                AscendC::Adds(ub.row_tmp, ub.M[j * m_stride_lu], 0.0f, w1_lu_pad);
            }
            AscendC::Brcb(ub.coefficients[j0 * 8u], ub.lu_stage[s * pad_n + j0], static_cast<uint8_t>((pad_n - j0) >> 3), {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            if (pivot != j) {
                AscendC::Adds(ub.M[pivot * m_stride_lu], ub.row_tmp, 0.0f, w1_lu_pad);
                AscendC::Adds(ub.M[j * m_stride_lu], ub.pivot_row, 0.0f, w1_lu_pad);
                AscendC::PipeBarrier<PIPE_V>();
            }
            const uint32_t sub_rows = n - 1u - j;
            for (uint32_t offset = 0; offset < w1_lu_pad; offset += 64u) {
                const uint64_t lanes = w1_lu_pad - offset < 64u ? w1_lu_pad - offset : 64u;
                AscendC::MulAddDst(ub.M[(j + 1u) * m_stride_lu + offset], ub.pivot_row[offset],
                                   ub.coefficients[(j + 1u) * 8u], lanes, static_cast<uint8_t>(sub_rows), m_lu_repeat);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }
}

ELEC_FN void large_accum_panel_residual(
    uint32_t col_start, uint32_t width, uint32_t k, uint32_t pad_n, const LargeUB& ub,
    const AscendC::LocalTensor<uint16_t>& split_mask_u16) {
    for (uint32_t rhs_base = 0; rhs_base < k; rhs_base += 4u) {
        const uint32_t cur_k = (k - rhs_base < 4u) ? (k - rhs_base) : 4u, len = cur_k * pad_n;
        auto s_hi_chunk = ub.s_hi[rhs_base * pad_n], s_lo_chunk = ub.s_lo[rhs_base * pad_n];
        auto d1_tmp_4 = ub.s_hi[1024], p_err_alt_4 = ub.s_lo[1024];
        residual_prepare_col0(pad_n, ub.M, ub.col_offsets_u32, split_mask_u16, ub.col_vec, ub.abs_col, ub.pivot_row);
        if (k <= 4u) {
            for (uint32_t step = 0; step < width; ++step) {
                const bool odd = (step & 1u) != 0u;
                residual_step_k4(
                    step, width, cur_k, pad_n, len, &ub.x0_ptr[(col_start + step) * 8u + rhs_base],
                    ub.M, ub.col_offsets_u32, split_mask_u16, ub.col_vec,
                    odd ? ub.t_sum_4 : s_hi_chunk, odd ? s_hi_chunk : ub.t_sum_4, s_lo_chunk, ub.lu_stage[0],
                    odd ? p_err_alt_4 : ub.lu_stage[1024], odd ? ub.lu_stage[1024] : p_err_alt_4,
                    odd ? ub.row_tmp : ub.abs_col, odd ? ub.abs_col : ub.row_tmp,
                    odd ? ub.scratch : ub.pivot_row, odd ? ub.pivot_row : ub.scratch, ub.y[0], ub.y[1024], d1_tmp_4);
            }
            const bool last_odd = ((width - 1u) & 1u) != 0u;
            AscendC::Sub(ub.y[1024], last_odd ? ub.t_sum_4 : s_hi_chunk, ub.y[1024], len);
            AscendC::Add(d1_tmp_4, d1_tmp_4, last_odd ? p_err_alt_4 : ub.lu_stage[1024], len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(s_lo_chunk, s_lo_chunk, d1_tmp_4, len);
            if ((width & 1u) != 0u) AscendC::Adds(s_hi_chunk, ub.t_sum_4, 0.0f, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(s_lo_chunk, s_lo_chunk, ub.y[1024], len);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            residual_accum_general(
                col_start, width, rhs_base, cur_k, pad_n, len, ub.x0_ptr,
                ub.M, ub.col_offsets_u32, split_mask_u16, ub.col_vec,
                s_hi_chunk, ub.t_sum_4, s_lo_chunk, ub.lu_stage[0], ub.lu_stage[1024],
                ub.abs_col, ub.pivot_row, ub.y[0], ub.y[1024]);
        }
    }
}

template <typename T>
ELEC_FN void large_compute_residual_rhs(
    uint64_t batch, uint64_t b_offset, uint32_t n, uint32_t k, uint32_t pad_n,
    uint32_t w0, uint32_t w0_pad, uint32_t w1, uint32_t w1_pad, uint32_t m_stride_ref,
    const LargeUB& ub,
    const AscendC::GlobalTensor<T>& input_a,
    const AscendC::GlobalTensor<T>& input_b,
    const AscendC::LocalTensor<uint16_t>& split_mask_u16) {
    const AscendC::BinaryRepeatParams lane_rep{1, 1, 0, 8, 8, 0};
    const uint32_t num_panels = w1 > 0u ? 2u : 1u;
    fence<AscendC::HardEvent::V_MTE2>();
    load_rhs_b<T>(n, pad_n, k, b_offset, input_b, ub.y, ub.lu_stage, ub.coefficients);
    for (uint32_t t = 0; t < k; ++t) {
        AscendC::Gather(ub.s_hi[t * pad_n], ub.y[t], ub.y_col_offsets_u32, uint32_t(0), n);
    }
    AscendC::Duplicate(ub.s_lo, 0.0f, k * pad_n);
    AscendC::PipeBarrier<PIPE_V>();
    fence<AscendC::HardEvent::V_S>();
    for (uint32_t panel = 0; panel < num_panels; ++panel) {
        const uint32_t col_start = panel * 128u;
        const uint32_t width = panel == 0u ? w0 : w1;
        const uint32_t width_pad = panel == 0u ? w0_pad : w1_pad;
        fence<AscendC::HardEvent::V_MTE2>();
        load_panel_a<T>(n, batch, col_start, width, width_pad, m_stride_ref, input_a, ub.M, ub.lu_stage);
        large_accum_panel_residual(col_start, width, k, pad_n, ub, split_mask_u16);
    }
    AscendC::Add(ub.s_hi, ub.s_hi, ub.s_lo, k * pad_n);
    AscendC::Duplicate(ub.y, 0.0f, (pad_n + 8u) * 8u);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t t = 0; t < k; ++t) {
        AscendC::Brcb(ub.coefficients, ub.s_hi[t * pad_n], static_cast<uint8_t>(pad_n >> 3), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst(ub.y, ub.coefficients, ub.unit_masks[t * 8u], 64u, static_cast<uint8_t>(pad_n >> 3), lane_rep);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Duplicate(ub.lu_stage[pad_n * 8u], 0.0f, 64u);
    AscendC::Duplicate(ub.coefficients[pad_n * 8u], 0.0f, 64u);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
ELEC_FN void large_refine_batch(
    uint64_t batch, uint64_t b_offset, uint32_t n, uint32_t k, uint32_t pad_n,
    uint32_t w0, uint32_t w0_pad, uint32_t w1, uint32_t w1_pad, uint32_t m_stride_ref,
    const uint16_t* piv, const LargeUB& ub,
    const AscendC::GlobalTensor<T>& input_a,
    const AscendC::GlobalTensor<T>& input_b,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::LocalTensor<uint16_t>& split_mask_u16,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    const uint32_t max_refine_iters = (n > 200u) ? 2u : 1u;
    for (uint32_t refine_iter = 0; refine_iter < max_refine_iters; ++refine_iter) {
        large_compute_residual_rhs<T>(
            batch, b_offset, n, k, pad_n, w0, w0_pad, w1, w1_pad, m_stride_ref, ub, input_a, input_b, split_mask_u16);

        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(ub.M, lu_gm[0], w0 * pad_n);
        fence<AscendC::HardEvent::MTE2_V>();
        fwd_sub_panel(0u, w0, n, pad_n, piv, ub.M, ub.y, ub.row_tmp, ub.coefficients, ub.lu_stage, y_repeat8);
        if (w1 > 0u) {
            fence<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopy(ub.M, lu_gm[128u * pad_n], w1 * pad_n);
            fence<AscendC::HardEvent::MTE2_V>();
            fwd_sub_panel(128u, w1, n, pad_n, piv, ub.M, ub.y, ub.row_tmp, ub.coefficients, ub.lu_stage, y_repeat8);
            back_sub_panel(128u, w1, pad_n, ub.M, ub.y, ub.coefficients, ub.lu_stage, y_repeat8);
            fence<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopy(ub.M, lu_gm[0], w0 * pad_n);
            fence<AscendC::HardEvent::MTE2_V>();
        }
        back_sub_panel(0u, w0, pad_n, ub.M, ub.y, ub.coefficients, ub.lu_stage, y_repeat8);
        AscendC::Brcb(ub.lu_stage, ub.inv_diag_vec, static_cast<uint8_t>(pad_n >> 3), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(ub.y, ub.y, ub.lu_stage, pad_n * 8u);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(ub.x0, ub.x0, ub.y, pad_n * 8u);
        AscendC::PipeBarrier<PIPE_V>();
        if (refine_iter + 1u < max_refine_iters) {
            AscendC::Abs(ub.lu_stage, ub.y, pad_n * 8u);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax(ub.reduced, ub.lu_stage, ub.scratch, pad_n * 8u, false);
            fence<AscendC::HardEvent::V_S>();
            if (!(ub.reduced.GetValue(0) > 3.5e-3f)) break;
        }
    }
}

template <typename T>
ELEC_FN int large_factor_and_backsub(
    uint64_t batch, uint64_t b_offset, uint32_t n, uint32_t k, uint32_t pad_n,
    uint32_t w0_lu, uint32_t w0_lu_pad, uint32_t w1_lu, uint32_t w1_lu_pad, uint32_t m_stride_lu,
    uint32_t w0, uint32_t w1, uint16_t* piv, const LargeUB& ub,
    const AscendC::GlobalTensor<T>& input_a,
    const AscendC::GlobalTensor<T>& input_b,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::BinaryRepeatParams& m_lu_repeat,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    load_rhs_b<T>(n, pad_n, k, b_offset, input_b, ub.y, ub.lu_stage, ub.coefficients);
    AscendC::Duplicate(ub.abs_col, -1.0f, pad_n);
    AscendC::Duplicate(ub.coefficients[pad_n * 8u], 0.0f, 64u);
    AscendC::PipeBarrier<PIPE_V>();

    load_panel_a<T>(n, batch, 0u, w0_lu, w0_lu_pad, m_stride_lu, input_a, ub.M, ub.lu_stage);
    int status = large_factor_panel_cols(0u, w0_lu, w0_lu_pad, n, pad_n, m_stride_lu, piv, ub, lu_gm, m_lu_repeat, y_repeat8);
    if (status != SUCCESS) return status;

    if (w1_lu > 0u) {
        load_panel_a<T>(n, batch, 128u, w1_lu, w1_lu_pad, m_stride_lu, input_a, ub.M, ub.lu_stage);
        large_apply_panel0_to_panel1(n, pad_n, w1_lu_pad, m_stride_lu, piv, ub, lu_gm, m_lu_repeat);
        status = large_factor_panel_cols(128u, w1_lu, w1_lu_pad, n, pad_n, m_stride_lu, piv, ub, lu_gm, m_lu_repeat, y_repeat8);
        if (status != SUCCESS) return status;
    }

    if (w1 > 0u) {
        AscendC::DataCopy(ub.M, lu_gm[128u * pad_n], w1 * pad_n);
        fence<AscendC::HardEvent::MTE2_V>();
        back_sub_panel(128u, w1, pad_n, ub.M, ub.y, ub.coefficients, ub.lu_stage, y_repeat8);
        fence<AscendC::HardEvent::V_MTE2>();
    }
    AscendC::DataCopy(ub.M, lu_gm[0], w0 * pad_n);
    fence<AscendC::HardEvent::MTE2_V>();
    back_sub_panel(0u, w0, pad_n, ub.M, ub.y, ub.coefficients, ub.lu_stage, y_repeat8);
    AscendC::Brcb(ub.lu_stage, ub.inv_diag_vec, static_cast<uint8_t>(pad_n >> 3), {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(ub.x0, ub.y, ub.lu_stage, pad_n * 8u);
    AscendC::PipeBarrier<PIPE_V>();
    return SUCCESS;
}

template <typename T>
ELEC_FN int large_solve_single_batch(
    uint64_t batch, uint64_t b_offset, uint32_t n, uint32_t k, uint32_t pad_n,
    uint32_t w0_lu, uint32_t w0_lu_pad, uint32_t w1_lu, uint32_t w1_lu_pad, uint32_t m_stride_lu,
    uint32_t w0, uint32_t w0_pad, uint32_t w1, uint32_t w1_pad, uint32_t m_stride_ref,
    uint16_t* piv, const LargeUB& ub,
    const AscendC::GlobalTensor<T>& input_a,
    const AscendC::GlobalTensor<T>& input_b,
    const AscendC::GlobalTensor<T>& output_x,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::BinaryRepeatParams& m_lu_repeat,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    if (batch != AscendC::GetBlockIdx() && m_stride_lu != m_stride_ref) {
        AscendC::CreateVecIndex(ub.col_offsets, int32_t(0), pad_n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(ub.col_offsets, ub.col_offsets, static_cast<int32_t>(m_stride_lu * sizeof(float)), pad_n);
        AscendC::PipeBarrier<PIPE_V>();
    }
    const int status = large_factor_and_backsub<T>(
        batch, b_offset, n, k, pad_n, w0_lu, w0_lu_pad, w1_lu, w1_lu_pad, m_stride_lu,
        w0, w1, piv, ub, input_a, input_b, lu_gm, m_lu_repeat, y_repeat8);
    if (status != SUCCESS) return status;

    if (m_stride_lu != m_stride_ref) {
        AscendC::CreateVecIndex(ub.col_offsets, int32_t(0), pad_n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(ub.col_offsets, ub.col_offsets, static_cast<int32_t>(m_stride_ref * sizeof(float)), pad_n);
        AscendC::PipeBarrier<PIPE_V>();
    }
    auto split_mask_i32 = ub.lu_col.template ReinterpretCast<int32_t>();
    AscendC::Duplicate(split_mask_i32, static_cast<int32_t>(0xfffff000u), pad_n);
    AscendC::PipeBarrier<PIPE_V>();
    large_refine_batch<T>(
        batch, b_offset, n, k, pad_n, w0, w0_pad, w1, w1_pad, m_stride_ref, piv, ub,
        input_a, input_b, lu_gm, split_mask_i32.template ReinterpretCast<uint16_t>(), y_repeat8);
    store_square_solution<T>(batch, n, k, output_x, ub.x0, ub.x_out, ub.lu_stage, ub.M.template ReinterpretCast<int32_t>());
    return SUCCESS;
}

template <typename T>
ELEC_FN void large_init_offsets_and_residuals(
    const Config& c, uint32_t k, uint32_t pad_n, uint32_t m_stride_lu, const LargeUB& ub,
    GM_ADDR residuals, const AscendC::GlobalTensor<T>& output_r) {
    AscendC::CreateVecIndex(ub.y_col_offsets, int32_t(0), pad_n);
    AscendC::Duplicate(ub.lu_stage[pad_n * 8u], 0.0f, 64u);
    AscendC::Duplicate(ub.coefficients[pad_n * 8u], 0.0f, 64u);
    AscendC::Duplicate(ub.unit_masks, 0.0f, 64u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(ub.col_offsets, ub.y_col_offsets, static_cast<int32_t>(m_stride_lu * sizeof(float)), pad_n);
    AscendC::Muls(ub.y_col_offsets, ub.y_col_offsets, static_cast<int32_t>(8u * sizeof(float)), pad_n);
    AscendC::PipeBarrier<PIPE_V>();
    fence<AscendC::HardEvent::V_S>();
    for (uint32_t t = 0; t < 8u; ++t) {
        ub.unit_masks.SetValue(t * 8u + t, 1.0f);
    }
    fence<AscendC::HardEvent::S_V>();

    if ((c.flags & LSTSQ_SQUARE_RESIDUAL) && residuals != nullptr && AscendC::GetBlockIdx() == 0) {
        AscendC::Duplicate(ub.lu_stage, 0.0f, 256u);
        fence<AscendC::HardEvent::V_MTE3>();
        const AscendC::DataCopyExtParams r_copy{1, static_cast<uint32_t>(c.batch * k * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(output_r, ub.lu_stage.template ReinterpretCast<T>(), r_copy);
        fence<AscendC::HardEvent::MTE3_V>();
    }
}

template <typename T>
__aicore__ __attribute__((noinline)) int solve_square_large(
    const Config& c, GM_ADDR A, GM_ADDR B, GM_ADDR solution, GM_ADDR residuals, GM_ADDR user_workspace) {
    const uint32_t n = static_cast<uint32_t>(c.n);
    const uint32_t k = static_cast<uint32_t>(c.k);
    const uint32_t pad_n = (n + 7u) & ~7u;
    const uint32_t w0_lu = (n <= 200u) ? n : 128u;
    const uint32_t w0_lu_pad = (w0_lu + 7u) & ~7u;
    const uint32_t w1_lu = (n > w0_lu) ? (n - w0_lu) : 0u;
    const uint32_t w1_lu_pad = (w1_lu + 7u) & ~7u;
    const uint32_t m_stride_lu = ((w0_lu_pad >> 3) | 1u) << 3;
    const uint32_t w0 = (n < 128u) ? n : 128u;
    const uint32_t w0_pad = (w0 + 7u) & ~7u;
    const uint32_t w1 = (n > 128u) ? (n - 128u) : 0u;
    const uint32_t w1_pad = (w1 + 7u) & ~7u;
    constexpr uint32_t m_stride_ref = 136u;

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_buf;
    LargeUB ub;
    init_large_ub(ub, pipe, work_buf);

    AscendC::GlobalTensor<T> input_a, input_b, output_x, output_r;
    AscendC::GlobalTensor<float> lu_gm;
    input_a.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(A));
    input_b.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(B));
    output_x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(solution));
    if (residuals != nullptr) {
        output_r.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(residuals));
    }
    lu_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(user_workspace));

    large_init_offsets_and_residuals<T>(c, k, pad_n, m_stride_lu, ub, residuals, output_r);

    const AscendC::BinaryRepeatParams m_lu_repeat{1, 1, 0, static_cast<uint8_t>(m_stride_lu >> 3), 0, 1};
    const AscendC::BinaryRepeatParams y_repeat8{1, 0, 1, 8, 0, 8};
    uint16_t piv[256];
    for (uint64_t batch = AscendC::GetBlockIdx(); batch < c.batch; batch += AscendC::GetBlockNum()) {
        const int status = large_solve_single_batch<T>(
            batch, batch * c.rhs_batch_stride, n, k, pad_n,
            w0_lu, w0_lu_pad, w1_lu, w1_lu_pad, m_stride_lu,
            w0, w0_pad, w1, w1_pad, m_stride_ref,
            piv, ub, input_a, input_b, output_x, lu_gm, m_lu_repeat, y_repeat8);
        if (status != SUCCESS) return status;
    }
    return SUCCESS;
}

}  // namespace elec::lstsq_fast
