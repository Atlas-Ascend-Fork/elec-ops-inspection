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
#include "lstsq_common.h"

namespace elec::lstsq_fast {

template <typename T>
struct P8GlobalTensors {
    AscendC::GlobalTensor<T> input_a, input_b, output_x, output_r;
    AscendC::GlobalTensor<float> lu_gm, inv_diag_gm, x0_gm, r_gm;
    AscendC::GlobalTensor<int32_t> piv_gm;
};

template <typename T>
ELEC_FN void init_p8_gm_tensors(
    P8GlobalTensors<T>& gm, uint32_t batch, uint64_t per_core,
    GM_ADDR A, GM_ADDR B, GM_ADDR solution, GM_ADDR residuals, GM_ADDR ws_base) {
    gm.input_a.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(A));
    gm.input_b.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(B));
    gm.output_x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(solution));
    if (residuals != nullptr) {
        gm.output_r.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(residuals));
    }
    __gm__ float* batch_ws_f32 = reinterpret_cast<__gm__ float*>(ws_base + 16384u + static_cast<uint64_t>(batch) * (7u * per_core));
    gm.lu_gm.SetGlobalBuffer(batch_ws_f32);
    gm.piv_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(batch_ws_f32 + 65536u));
    gm.inv_diag_gm.SetGlobalBuffer(batch_ws_f32 + 65536u + 256u);
    gm.x0_gm.SetGlobalBuffer(batch_ws_f32 + 66048u);
    gm.r_gm.SetGlobalBuffer(batch_ws_f32 + 68096u);
}

ELEC_FN void apply_chunk_p8_step(
    uint32_t s, uint32_t col_start, uint32_t chunk_w, uint32_t sub_id, uint32_t n, uint32_t pad_n, uint32_t row_w,
    uint16_t* piv, const LargeUB& ub, const AscendC::BinaryRepeatParams& m_p8_repeat) {
    const uint32_t j = col_start + s;
    uint32_t pivot = static_cast<uint32_t>(ub.piv_stage_ptr[32u + s]);
    if (pivot >= n) pivot = j;
    if (sub_id == 7u) piv[j] = static_cast<uint16_t>(pivot);
    const bool use_buf1 = (s & 1u) != 0u;
    if (pivot != j) {
        AscendC::Adds(ub.pivot_row[0], ub.M[pivot * 40u], 0.0f, row_w);
        AscendC::Adds(ub.M[pivot * 40u], ub.M[j * 40u], 0.0f, row_w);
    }
    if (s + 1u < chunk_w && j + 2u < n) {
        const uint32_t next_j0 = (j + 1u) & ~7u;
        const auto& dst_buf = use_buf1 ? ub.coefficients : ub.lu_stage;
        AscendC::Brcb(dst_buf[next_j0 * 8u], ub.lu_panel7[(s + 1u) * pad_n + next_j0],
                      static_cast<uint8_t>((pad_n - next_j0) >> 3), {1, 8});
    }
    if (pivot != j) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(ub.M[j * 40u], ub.pivot_row[0], 0.0f, row_w);
    }
    if (j + 1u < n) {
        const uint32_t sub_rows = n - 1u - j;
        auto src_m = (pivot != j) ? ub.pivot_row[0] : ub.M[j * 40u];
        const auto& cur_buf = use_buf1 ? ub.lu_stage : ub.coefficients;
        AscendC::MulAddDst(ub.M[(j + 1u) * 40u], src_m, cur_buf[(j + 1u) * 8u],
                           row_w, static_cast<uint8_t>(sub_rows), m_p8_repeat);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

ELEC_FN void apply_chunk_p8(
    uint32_t col_start, uint32_t chunk_w, uint32_t sub_id, uint32_t n, uint32_t pad_n, uint32_t row_w,
    uint16_t* piv, const LargeUB& ub,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::GlobalTensor<int32_t>& piv_gm,
    const AscendC::BinaryRepeatParams& m_p8_repeat) {
    fence<AscendC::HardEvent::V_MTE2>();
    fence<AscendC::HardEvent::S_MTE2>();
    AscendC::DataCopy(ub.lu_panel7, lu_gm[col_start * pad_n], chunk_w * pad_n);
    AscendC::DataCopy(ub.piv_stage[32], piv_gm[col_start], chunk_w);
    fence<AscendC::HardEvent::MTE2_V>();
    fence<AscendC::HardEvent::MTE2_S>();
    if (col_start + 1u < n) {
        AscendC::Brcb(ub.coefficients[col_start * 8u], ub.lu_panel7[col_start],
                      static_cast<uint8_t>((pad_n - col_start) >> 3), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }
    for (uint32_t s = 0; s < chunk_w; ++s) {
        apply_chunk_p8_step(s, col_start, chunk_w, sub_id, n, pad_n, row_w, piv, ub, m_p8_repeat);
    }
}

ELEC_FN int32_t factor_p8_reduce_pivot(
    uint32_t j, uint32_t j0, uint32_t n, uint32_t pad_n, uint32_t local_col, bool has_rem_cols, uint32_t& pivot,
    const LargeUB& ub) {
    const uint32_t s8 = j & 7u;
    const uint32_t rem_a = n - j0;
    const uint32_t chunk0 = rem_a < 64u ? rem_a : 64u;
    const uint64_t c_mask = (chunk0 == 64u) ? ~0ULL : ((1ULL << chunk0) - 1ULL);
    AscendC::PipeBarrier<PIPE_V>();
    uint64_t mask[2] = {c_mask & (~0ULL << s8), 0ULL};
    AscendC::Abs(ub.abs_col[j0], ub.col_vec[j0], mask, 1u, {1, 1, 8, 8});
    if (rem_a > 64u) {
        AscendC::Abs(ub.abs_col[j0 + 64u], ub.col_vec[j0 + 64u], rem_a - 64u);
    }
    AscendC::PipeBarrier<PIPE_V>();
    const uint32_t reps = (n - j0 + 63u) >> 6;
    AscendC::WholeReduceMax<float>(
        ub.reduced, ub.abs_col[j0], 64, static_cast<int32_t>(reps), 1, 1, 8,
        AscendC::ReduceOrder::ORDER_VALUE_INDEX);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    if (j0 > 0u) {
        AscendC::Gather(ub.col_vec[0], ub.M[local_col], ub.col_offsets_u32, uint32_t(0), j0);
    }
    if (has_rem_cols) {
        AscendC::Brcb(ub.coefficients[j0 * 8u], ub.col_vec[j0],
                      static_cast<uint8_t>((pad_n - j0) >> 3), {1, 8});
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate(ub.abs_col[j0], -1.0f, 8u);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);

    const uint64_t best_rv = ub.reduced_u64_ptr[0];
    int32_t best_bits = static_cast<int32_t>(best_rv);
    pivot = j0 + static_cast<uint32_t>(best_rv >> 32);
    for (uint32_t r = 1u; r < reps; ++r) {
        const uint64_t rv = ub.reduced_u64_ptr[r];
        const int32_t bits = static_cast<int32_t>(rv);
        if (bits > best_bits) {
            best_bits = bits;
            pivot = j0 + (r << 6) + static_cast<uint32_t>(rv >> 32);
        }
    }
    if (pivot >= n) pivot = j;
    return best_bits;
}

ELEC_FN float factor_p8_find_pivot(
    uint32_t j, uint32_t n, uint32_t pad_n, uint32_t local_col, bool has_rem_cols, uint32_t& pivot,
    const LargeUB& ub) {
    const uint32_t j0 = j & ~7u;
    AscendC::Gather(ub.col_vec[j0], ub.M[j0 * 40u + local_col], ub.col_offsets_u32, uint32_t(0), n - j0);
    pivot = j;
    if (j + 1u < n) {
        const int32_t best_bits = factor_p8_reduce_pivot(j, j0, n, pad_n, local_col, has_rem_cols, pivot, ub);
        const float diag = ub.col_vec_ptr[pivot];
        const float col_j = ub.col_vec_ptr[j];
        const float best_abs = bits_float(static_cast<uint32_t>(best_bits));
        const float inv_abs = (best_bits > 0 && best_abs != 0.0f) ? (1.0f / best_abs) : 0.0f;
        if (pivot != j) {
            ub.col_vec_ptr[pivot] = col_j;
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        }
        const uint32_t sign_bit = (best_bits > 0) ? (float_bits(diag) & 0x80000000u) : 0u;
        return bits_float(float_bits(inv_abs) | sign_bit);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    if (j0 > 0u) {
        AscendC::Gather(ub.col_vec[0], ub.M[local_col], ub.col_offsets_u32, uint32_t(0), j0);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    const float diag = ub.col_vec_ptr[j];
    return (diag != 0.0f) ? (1.0f / diag) : 0.0f;
}

ELEC_FN void factor_p8_store_meta(
    uint32_t s, uint32_t j, uint32_t pivot, float inv_diag, uint32_t sub_id, bool is_last_s, bool has_rem_cols,
    uint32_t col_start, uint32_t chunk_w, uint32_t my_col_start, uint32_t local_col, uint32_t row_w,
    uint16_t* piv, const LargeUB& ub,
    const AscendC::GlobalTensor<int32_t>& piv_gm,
    const AscendC::GlobalTensor<float>& inv_diag_gm) {
    ub.piv_stage_ptr[s] = static_cast<int32_t>(pivot);
    ub.inv_diag_stage_ptr[local_col] = inv_diag;
    if (sub_id == 7u) {
        piv[j] = static_cast<uint16_t>(pivot);
        ub.inv_diag_ptr[j] = inv_diag;
    } else if (is_last_s) {
        fence<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopy(piv_gm[col_start], ub.piv_stage, (chunk_w + 7u) & ~7u);
        if (!has_rem_cols) {
            AscendC::DataCopy(inv_diag_gm[my_col_start], ub.inv_diag_stage, row_w);
        }
    }
}

ELEC_FN void factor_p8_update_rem_cols(
    uint32_t j, uint32_t pivot, float inv_diag, uint32_t sub_id, bool is_last_s,
    uint32_t col_start, uint32_t chunk_w, uint32_t n, uint32_t pad_n, uint32_t c0, uint32_t rem,
    const LargeUB& ub,
    const AscendC::LocalTensor<float>& lu_col_slot,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::BinaryRepeatParams& m_p8_repeat) {
    AscendC::Muls(ub.scratch, ub.M[pivot * 40u + c0], -inv_diag, rem);
    if (pivot != j) {
        AscendC::Adds(ub.pivot_row, ub.M[pivot * 40u + c0], 0.0f, rem);
        AscendC::Adds(ub.M[pivot * 40u + c0], ub.M[j * 40u + c0], 0.0f, rem);
        AscendC::Adds(ub.coefficients[pivot * 8u], ub.coefficients[j * 8u], 0.0f, 8u);
    }
    if (is_last_s) {
        if (pivot != j) {
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        }
        AscendC::Muls(lu_col_slot, ub.col_vec, -inv_diag, pad_n);
    }
    AscendC::PipeBarrier<PIPE_V>();
    if (pivot != j) {
        AscendC::Adds(ub.M[j * 40u + c0], ub.pivot_row, 0.0f, rem);
    }
    if (is_last_s) {
        if (sub_id != 7u) {
            fence<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(lu_gm[(col_start + chunk_w - 1u) * pad_n], lu_col_slot, pad_n);
        }
    } else {
        if (pivot != j) {
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        }
        AscendC::Muls(lu_col_slot, ub.col_vec, -inv_diag, pad_n);
    }
    const uint32_t sub_rows = n - 1u - j;
    if (rem == 8u) {
        const AscendC::BinaryRepeatParams m_p8_rep8{5, 0, 1, 40, 0, 8};
        AscendC::MulAddDst(ub.M[(j + 1u) * 40u + c0], ub.scratch, ub.coefficients[(j + 1u) * 8u],
                           64u, static_cast<uint8_t>((sub_rows + 7u) >> 3), m_p8_rep8);
    } else {
        AscendC::MulAddDst(ub.M[(j + 1u) * 40u + c0], ub.scratch, ub.coefficients[(j + 1u) * 8u],
                           rem, static_cast<uint8_t>(sub_rows), m_p8_repeat);
    }
}

ELEC_FN void factor_chunk_p8(
    uint32_t col_start, uint32_t chunk_w, uint32_t sub_id, uint32_t n, uint32_t pad_n, uint32_t my_col_start, uint32_t row_w,
    uint16_t* piv, const LargeUB& ub, const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::GlobalTensor<int32_t>& piv_gm, const AscendC::GlobalTensor<float>& inv_diag_gm,
    const AscendC::LocalTensor<float>& lu_chunk, const AscendC::BinaryRepeatParams& m_p8_repeat) {
    const uint32_t local_col0 = col_start - my_col_start;
    for (uint32_t s = 0; s < chunk_w; ++s) {
        if (sub_id != 7u && s + 1u == chunk_w && chunk_w > 1u) {
            fence<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopy(lu_gm[col_start * pad_n], lu_chunk[0], (chunk_w - 1u) * pad_n);
        }
        const uint32_t j = col_start + s, local_col = local_col0 + s, c0 = (local_col + 1u) & ~7u;
        const bool has_rem_cols = (local_col + 1u < row_w), is_last_s = (s + 1u == chunk_w);
        const uint32_t rem = has_rem_cols ? (row_w - c0) : 0u;
        auto lu_col_slot = lu_chunk[s * pad_n];
        uint32_t pivot = j;
        const float inv_diag = factor_p8_find_pivot(j, n, pad_n, local_col, has_rem_cols, pivot, ub);
        if (has_rem_cols && j + 1u < n) {
            factor_p8_update_rem_cols(j, pivot, inv_diag, sub_id, is_last_s, col_start, chunk_w, n, pad_n, c0, rem,
                                     ub, lu_col_slot, lu_gm, m_p8_repeat);
            factor_p8_store_meta(s, j, pivot, inv_diag, sub_id, is_last_s, has_rem_cols,
                                 col_start, chunk_w, my_col_start, local_col, row_w, piv, ub, piv_gm, inv_diag_gm);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            if (j + 1u < n && pivot != j) AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            AscendC::Muls(lu_col_slot, ub.col_vec, -inv_diag, pad_n);
            factor_p8_store_meta(s, j, pivot, inv_diag, sub_id, is_last_s, has_rem_cols,
                                 col_start, chunk_w, my_col_start, local_col, row_w, piv, ub, piv_gm, inv_diag_gm);
            AscendC::PipeBarrier<PIPE_V>();
            if (sub_id != 7u && is_last_s) {
                fence<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopy(lu_gm[(col_start + chunk_w - 1u) * pad_n], lu_col_slot, pad_n);
            }
        }
    }
    if (sub_id != 7u) {
        fence<AscendC::HardEvent::MTE3_V>();
        fence<AscendC::HardEvent::MTE3_S>();
    }
}

template <typename T>
ELEC_FN void p8_wavefront_init_c0(
    uint32_t sub_id, uint32_t n, uint32_t k, uint32_t pad_n, uint32_t w_p, uint32_t my_w,
    uint64_t b_offset, const LargeUB& ub, const AscendC::GlobalTensor<T>& input_b,
    const AscendC::UnaryRepeatParams& y_to_m_rep) {
    if (sizeof(T) != 4 || w_p == 32u || sub_id >= 4u) {
        AscendC::Adds(ub.A_orig, ub.M, 0.0f, n * 40u);
    }
    if (sub_id == 7u) {
        load_rhs_b<T>(n, pad_n, k, b_offset, input_b, ub.y, ub.lu_stage, ub.coefficients);
        for (uint32_t t = 0; t < k; ++t) {
            AscendC::Gather(ub.s_hi[t * pad_n], ub.y[t], ub.y_col_offsets_u32, uint32_t(0), n);
        }
        AscendC::Adds(ub.M[my_w], ub.y[0], 0.0f, 8u, 128u, y_to_m_rep);
        AscendC::Adds(ub.M[128u * 40u + my_w], ub.y[128u * 8u], 0.0f, 8u, static_cast<uint8_t>(pad_n - 128u), y_to_m_rep);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
ELEC_FN void p8_wavefront_phase1(
    uint32_t sub_id, uint32_t n, uint32_t k, uint32_t pad_n, uint32_t w_p, uint32_t my_col_start, uint32_t my_w, uint32_t row_w,
    uint64_t b_offset, uint16_t* piv, const LargeUB& ub,
    const AscendC::GlobalTensor<T>& input_b,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::GlobalTensor<int32_t>& piv_gm,
    const AscendC::GlobalTensor<float>& inv_diag_gm,
    const AscendC::BinaryRepeatParams& m_p8_repeat) {
    const AscendC::UnaryRepeatParams y_to_m_rep{1, 1, 5, 1};
    const uint32_t c7 = (32u + 6u * w_p) >> 3;
    for (uint32_t c_idx = 0; c_idx <= c7; ++c_idx) {
        if (c_idx > 0u) {
            const uint32_t prev = c_idx - 1u;
            const uint32_t prev_owner = (w_p == 32u) ? (prev >> 2) : ((prev < 4u) ? 0u : (1u + (prev - 4u) / 3u));
            if (sub_id > prev_owner) {
                apply_chunk_p8(prev * 8u, 8u, sub_id, n, pad_n, row_w, piv, ub, lu_gm, piv_gm, m_p8_repeat);
            }
        }
        if (c_idx < c7) {
            const uint32_t owner = (w_p == 32u) ? (c_idx >> 2) : ((c_idx < 4u) ? 0u : (1u + (c_idx - 4u) / 3u));
            if (sub_id == owner) {
                factor_chunk_p8(c_idx * 8u, 8u, sub_id, n, pad_n, my_col_start, row_w, piv, ub,
                                lu_gm, piv_gm, inv_diag_gm, ub.lu_stage_p1, m_p8_repeat);
            } else if (c_idx == 0u) {
                p8_wavefront_init_c0<T>(sub_id, n, k, pad_n, w_p, my_w, b_offset, ub, input_b, y_to_m_rep);
            }
            AscendC::SyncAll<true>();
        }
    }
}

template <typename T, bool SMALL_TOTAL_CONST>
ELEC_FN void p8_prepare_and_wavefront(
    uint32_t sub_id, uint32_t batch, uint32_t n, uint32_t k, uint32_t pad_n, uint32_t w_p,
    uint32_t my_col_start, uint32_t my_w, uint32_t row_w, uint64_t b_offset, uint16_t* piv, const LargeUB& ub,
    const AscendC::DataCopyParams& a_copy,
    const AscendC::GlobalTensor<T>& input_a,
    const AscendC::GlobalTensor<T>& input_b,
    const AscendC::GlobalTensor<float>& lu_gm,
    const AscendC::GlobalTensor<int32_t>& piv_gm,
    const AscendC::GlobalTensor<float>& inv_diag_gm,
    const AscendC::BinaryRepeatParams& m_p8_repeat) {
    if constexpr (sizeof(T) == 4) {
        AscendC::DataCopy(ub.M, input_a[batch * n * n + my_col_start], a_copy);
    }
    AscendC::CreateVecIndex(ub.y_col_offsets, int32_t(0), pad_n);
    AscendC::Duplicate(ub.abs_col, -1.0f, 320u);
    AscendC::Duplicate(ub.unit_masks, 0.0f, 64u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(ub.col_offsets, ub.y_col_offsets, static_cast<int32_t>(40u * sizeof(float)), pad_n);
    AscendC::Muls(ub.y_col_offsets, ub.y_col_offsets, static_cast<int32_t>(8u * sizeof(float)), pad_n);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (!SMALL_TOTAL_CONST) {
        if (sub_id == 7u && k > 5u) {
            fence<AscendC::HardEvent::V_S>();
            for (uint32_t t = 0; t < 8u; ++t) {
                ub.unit_masks.SetValue(t * 8u + t, 1.0f);
            }
            fence<AscendC::HardEvent::S_V>();
        }
    }
    if constexpr (sizeof(T) == 4) {
        fence<AscendC::HardEvent::MTE2_V>();
    } else {
        load_panel_a<T>(n, batch, my_col_start, my_w, my_w, 40u, input_a, ub.M, ub.lu_stage);
    }
    p8_wavefront_phase1<T>(
        sub_id, n, k, pad_n, w_p, my_col_start, my_w, row_w, b_offset, piv, ub,
        input_b, lu_gm, piv_gm, inv_diag_gm, m_p8_repeat);
}

}  // namespace elec::lstsq_fast
