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

ELEC_FN void residual_prepare_col0(
    uint32_t pad_n, const AscendC::LocalTensor<float>& A_panel,
    const AscendC::LocalTensor<uint32_t>& col_offsets_u32,
    const AscendC::LocalTensor<uint16_t>& split_mask_u16,
    const AscendC::LocalTensor<float>& col_vec,
    const AscendC::LocalTensor<float>& a_hi,
    const AscendC::LocalTensor<float>& a_lo) {
    AscendC::Gather(col_vec, A_panel[0], col_offsets_u32, uint32_t(0), pad_n);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::And(a_hi.template ReinterpretCast<uint16_t>(),
                 col_vec.template ReinterpretCast<uint16_t>(),
                 split_mask_u16, pad_n * 2u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(a_lo, col_vec, a_hi, pad_n);
    AscendC::PipeBarrier<PIPE_V>();
}

ELEC_FN void split_x_and_mul_col(
    uint32_t cur_k, uint32_t pad_n, __ubuf__ const float* x0_col_ptr, float* x_val, float* x_lo_val,
    const AscendC::LocalTensor<float>& col_vec, const AscendC::LocalTensor<float>& a_hi,
    const AscendC::LocalTensor<float>& p_hi_4, const AscendC::LocalTensor<float>& p_err_4) {
    for (uint32_t t = 0; t < cur_k; ++t) {
        const float x = -x0_col_ptr[t];
        const float x_hi = bits_float(float_bits(x) & 0xfffff000u);
        x_val[t] = x;
        x_lo_val[t] = x - x_hi;
        AscendC::Muls(p_hi_4[t * pad_n], col_vec, x, pad_n);
        AscendC::Muls(p_err_4[t * pad_n], a_hi, x_hi, pad_n);
    }
}

ELEC_FN void residual_step_k4(
    uint32_t step, uint32_t width, uint32_t cur_k, uint32_t pad_n, uint32_t len, __ubuf__ const float* x0_col_ptr,
    const AscendC::LocalTensor<float>& M, const AscendC::LocalTensor<uint32_t>& col_offsets_u32,
    const AscendC::LocalTensor<uint16_t>& split_mask_u16, const AscendC::LocalTensor<float>& col_vec,
    const AscendC::LocalTensor<float>& src_hi, const AscendC::LocalTensor<float>& dst_hi,
    const AscendC::LocalTensor<float>& s_lo_chunk, const AscendC::LocalTensor<float>& p_hi_4,
    const AscendC::LocalTensor<float>& p_err_cur, const AscendC::LocalTensor<float>& p_err_prev,
    const AscendC::LocalTensor<float>& a_hi_cur, const AscendC::LocalTensor<float>& a_hi_next,
    const AscendC::LocalTensor<float>& a_lo_cur, const AscendC::LocalTensor<float>& a_lo_next,
    const AscendC::LocalTensor<float>& v_tmp_4, const AscendC::LocalTensor<float>& u_tmp_4,
    const AscendC::LocalTensor<float>& d1_tmp_4) {
    float x_val[4], x_lo_val[4];
    if (step > 0u) {
        AscendC::Sub(u_tmp_4, dst_hi, u_tmp_4, len);
        AscendC::Add(d1_tmp_4, d1_tmp_4, p_err_prev, len);
    }
    split_x_and_mul_col(cur_k, pad_n, x0_col_ptr, x_val, x_lo_val, col_vec, a_hi_cur, p_hi_4, p_err_cur);
    AscendC::PipeBarrier<PIPE_V>();
    if (step > 0u) { AscendC::Add(s_lo_chunk, s_lo_chunk, d1_tmp_4, len); }
    AscendC::Sub(p_err_cur, p_err_cur, p_hi_4, len);
    AscendC::Add(dst_hi, src_hi, p_hi_4, len);
    if (step + 1u < width) { AscendC::Gather(col_vec, M[step + 1u], col_offsets_u32, uint32_t(0), pad_n); }
    AscendC::PipeBarrier<PIPE_V>();
    if (step > 0u) { AscendC::Add(s_lo_chunk, s_lo_chunk, u_tmp_4, len); }
    for (uint32_t t = 0; t < cur_k; ++t) { AscendC::Axpy(p_err_cur[t * pad_n], a_hi_cur, x_lo_val[t], pad_n); }
    AscendC::Sub(v_tmp_4, dst_hi, src_hi, len);
    if (step + 1u < width) {
        AscendC::And(a_hi_next.template ReinterpretCast<uint16_t>(),
                     col_vec.template ReinterpretCast<uint16_t>(), split_mask_u16, pad_n * 2u);
    }
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t t = 0; t < cur_k; ++t) { AscendC::Axpy(p_err_cur[t * pad_n], a_lo_cur, x_val[t], pad_n); }
    AscendC::Sub(u_tmp_4, dst_hi, v_tmp_4, len);
    AscendC::Sub(d1_tmp_4, p_hi_4, v_tmp_4, len);
    if (step + 1u < width) { AscendC::Sub(a_lo_next, col_vec, a_hi_next, pad_n); }
    AscendC::PipeBarrier<PIPE_V>();
}

ELEC_FN void residual_step_k4_3wave(
    uint32_t step, uint32_t width, uint32_t cur_k, uint32_t pad_n, uint32_t len, __ubuf__ const float* x0_col_ptr,
    const AscendC::LocalTensor<float>& M, const AscendC::LocalTensor<uint32_t>& col_offsets_u32,
    const AscendC::LocalTensor<uint16_t>& split_mask_u16, const AscendC::LocalTensor<float>& col_vec_cur,
    const AscendC::LocalTensor<float>& col_vec_next, const AscendC::LocalTensor<float>& src_hi,
    const AscendC::LocalTensor<float>& dst_hi, const AscendC::LocalTensor<float>& s_lo_chunk,
    const AscendC::LocalTensor<float>& s_lo_acc2, const AscendC::LocalTensor<float>& p_hi_cur,
    const AscendC::LocalTensor<float>& p_hi_prev, const AscendC::LocalTensor<float>& p_err_4,
    const AscendC::LocalTensor<float>& a_hi_cur, const AscendC::LocalTensor<float>& a_hi_next,
    const AscendC::LocalTensor<float>& a_lo_cur, const AscendC::LocalTensor<float>& a_lo_next,
    const AscendC::LocalTensor<float>& v_tmp_4, const AscendC::LocalTensor<float>& u_tmp_4) {
    float x_val[4];
    if (step > 0u) {
        AscendC::Sub(u_tmp_4, src_hi, v_tmp_4, len);
        AscendC::Sub(p_hi_prev, p_hi_prev, v_tmp_4, len);
    }
    for (uint32_t t = 0; t < cur_k; ++t) {
        const float x = -x0_col_ptr[t];
        const float x_hi = bits_float(float_bits(x) & 0xfffff000u);
        x_val[t] = x;
        AscendC::Muls(p_hi_cur[t * pad_n], col_vec_cur, x, pad_n);
        AscendC::Muls(p_err_4[t * pad_n], a_hi_cur, x_hi, pad_n);
        AscendC::Axpy(s_lo_chunk[t * pad_n], a_hi_cur, x - x_hi, pad_n);
    }
    if (step + 1u < width) { AscendC::Gather(col_vec_next, M[step + 1u], col_offsets_u32, uint32_t(0), pad_n); }
    AscendC::PipeBarrier<PIPE_V>();
    if (step > 0u) {
        AscendC::Sub(u_tmp_4, dst_hi, u_tmp_4, len);
        AscendC::Add(s_lo_acc2, s_lo_acc2, p_hi_prev, len);
    }
    AscendC::Sub(p_err_4, p_err_4, p_hi_cur, len);
    for (uint32_t t = 0; t < cur_k; ++t) { AscendC::Axpy(s_lo_chunk[t * pad_n], a_lo_cur, x_val[t], pad_n); }
    AscendC::Add(dst_hi, src_hi, p_hi_cur, len);
    if (step + 1u < width) {
        AscendC::And(a_hi_next.template ReinterpretCast<uint16_t>(),
                     col_vec_next.template ReinterpretCast<uint16_t>(), split_mask_u16, pad_n * 2u);
    }
    AscendC::PipeBarrier<PIPE_V>();
    if (step > 0u) { AscendC::Add(s_lo_acc2, s_lo_acc2, u_tmp_4, len); }
    AscendC::Add(s_lo_chunk, s_lo_chunk, p_err_4, len);
    AscendC::Sub(v_tmp_4, dst_hi, src_hi, len);
    if (step + 1u < width) { AscendC::Sub(a_lo_next, col_vec_next, a_hi_next, pad_n); }
    AscendC::PipeBarrier<PIPE_V>();
}

ELEC_FN void residual_step_general(
    uint32_t step, uint32_t width, uint32_t cur_k, uint32_t pad_n, uint32_t len, __ubuf__ const float* x0_col_ptr,
    const AscendC::LocalTensor<float>& A_panel, const AscendC::LocalTensor<uint32_t>& col_offsets_u32,
    const AscendC::LocalTensor<uint16_t>& split_mask_u16, const AscendC::LocalTensor<float>& col_vec,
    const AscendC::LocalTensor<float>& src_hi, const AscendC::LocalTensor<float>& dst_hi,
    const AscendC::LocalTensor<float>& s_lo_chunk, const AscendC::LocalTensor<float>& p_hi_4,
    const AscendC::LocalTensor<float>& p_err_4, const AscendC::LocalTensor<float>& a_hi,
    const AscendC::LocalTensor<float>& a_lo, const AscendC::LocalTensor<float>& v_tmp_4,
    const AscendC::LocalTensor<float>& u_tmp_4) {
    float x_val[4], x_lo_val[4];
    if (step > 0u) { AscendC::Add(s_lo_chunk, s_lo_chunk, v_tmp_4, len); }
    split_x_and_mul_col(cur_k, pad_n, x0_col_ptr, x_val, x_lo_val, col_vec, a_hi, p_hi_4, p_err_4);
    AscendC::PipeBarrier<PIPE_V>();
    if (step > 0u) { AscendC::Add(s_lo_chunk, s_lo_chunk, u_tmp_4, len); }
    AscendC::Sub(p_err_4, p_err_4, p_hi_4, len);
    AscendC::Add(dst_hi, src_hi, p_hi_4, len);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t t = 0; t < cur_k; ++t) { AscendC::Axpy(p_err_4[t * pad_n], a_hi, x_lo_val[t], pad_n); }
    AscendC::Sub(v_tmp_4, dst_hi, src_hi, len);
    if (step + 1u < width) { AscendC::Gather(col_vec, A_panel[step + 1u], col_offsets_u32, uint32_t(0), pad_n); }
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t t = 0; t < cur_k; ++t) { AscendC::Axpy(p_err_4[t * pad_n], a_lo, x_val[t], pad_n); }
    AscendC::Sub(u_tmp_4, dst_hi, v_tmp_4, len);
    AscendC::Sub(p_hi_4, p_hi_4, v_tmp_4, len);
    if (step + 1u < width) {
        AscendC::And(a_hi.template ReinterpretCast<uint16_t>(),
                     col_vec.template ReinterpretCast<uint16_t>(), split_mask_u16, pad_n * 2u);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(v_tmp_4, p_err_4, p_hi_4, len);
    AscendC::Sub(u_tmp_4, src_hi, u_tmp_4, len);
    if (step + 1u < width) { AscendC::Sub(a_lo, col_vec, a_hi, pad_n); }
    AscendC::PipeBarrier<PIPE_V>();
}

ELEC_FN void residual_accum_general(
    uint32_t col_start, uint32_t width, uint32_t rhs_base, uint32_t cur_k, uint32_t pad_n, uint32_t len,
    __ubuf__ const float* x0_ptr, const AscendC::LocalTensor<float>& A_panel,
    const AscendC::LocalTensor<uint32_t>& col_offsets_u32, const AscendC::LocalTensor<uint16_t>& split_mask_u16,
    const AscendC::LocalTensor<float>& col_vec, const AscendC::LocalTensor<float>& s_hi_chunk,
    const AscendC::LocalTensor<float>& t_sum_4, const AscendC::LocalTensor<float>& s_lo_chunk,
    const AscendC::LocalTensor<float>& p_hi_4, const AscendC::LocalTensor<float>& p_err_4,
    const AscendC::LocalTensor<float>& a_hi, const AscendC::LocalTensor<float>& a_lo,
    const AscendC::LocalTensor<float>& v_tmp_4, const AscendC::LocalTensor<float>& u_tmp_4) {
    for (uint32_t step = 0; step < width; ++step) {
        const uint32_t col = col_start + step;
        const bool odd = (step & 1u) != 0u;
        const auto& src_hi = odd ? t_sum_4 : s_hi_chunk;
        const auto& dst_hi = odd ? s_hi_chunk : t_sum_4;
        residual_step_general(
            step, width, cur_k, pad_n, len, &x0_ptr[col * 8u + rhs_base],
            A_panel, col_offsets_u32, split_mask_u16, col_vec,
            src_hi, dst_hi, s_lo_chunk, p_hi_4, p_err_4, a_hi, a_lo, v_tmp_4, u_tmp_4);
    }
    AscendC::Add(s_lo_chunk, s_lo_chunk, v_tmp_4, len);
    if ((width & 1u) != 0u) {
        AscendC::Adds(s_hi_chunk, t_sum_4, 0.0f, len);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(s_lo_chunk, s_lo_chunk, u_tmp_4, len);
    AscendC::PipeBarrier<PIPE_V>();
}

ELEC_FN void two_sum_combine_buf(
    uint32_t offset, uint32_t chunk_len, const AscendC::LocalTensor<float>& other_hi,
    const AscendC::LocalTensor<float>& other_lo, const AscendC::LocalTensor<float>& s_hi,
    const AscendC::LocalTensor<float>& s_lo, const AscendC::LocalTensor<float>& t_sum_4,
    const AscendC::LocalTensor<float>& d_tmp, const AscendC::LocalTensor<float>& u_tmp) {
    AscendC::Add(s_lo[offset], s_lo[offset], other_lo, chunk_len);
    AscendC::Add(t_sum_4, s_hi[offset], other_hi, chunk_len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(d_tmp, t_sum_4, s_hi[offset], chunk_len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(u_tmp, t_sum_4, d_tmp, chunk_len);
    AscendC::Sub(other_hi, other_hi, d_tmp, chunk_len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(u_tmp, s_hi[offset], u_tmp, chunk_len);
    AscendC::Add(s_lo[offset], s_lo[offset], other_hi, chunk_len);
    AscendC::Adds(s_hi[offset], t_sum_4, 0.0f, chunk_len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(s_lo[offset], s_lo[offset], u_tmp, chunk_len);
    AscendC::PipeBarrier<PIPE_V>();
}

template <bool SMALL_CONST = false>
ELEC_FN void two_sum_combine_from_gm(
    uint32_t src_sub, uint32_t total_len, const AscendC::GlobalTensor<float>& r_gm,
    const AscendC::LocalTensor<float>& s_hi, const AscendC::LocalTensor<float>& s_lo,
    const AscendC::LocalTensor<float>& t_sum_4, const AscendC::LocalTensor<float>& lu_stage,
    const AscendC::LocalTensor<float>& y) {
    if (SMALL_CONST || total_len <= 1024u) {
        fence<AscendC::HardEvent::V_MTE2>();
        AscendC::DataCopy(lu_stage[0], r_gm[src_sub * 2048u], 2u * total_len);
        fence<AscendC::HardEvent::MTE2_V>();
        two_sum_combine_buf(0u, total_len, lu_stage[0], lu_stage[total_len], s_hi, s_lo, t_sum_4, y[0], y[1056]);
        return;
    }
    if constexpr (!SMALL_CONST) {
        for (uint32_t offset = 0; offset < total_len; offset += 1024u) {
            const uint32_t chunk_len = (total_len - offset < 1024u) ? (total_len - offset) : 1024u;
            fence<AscendC::HardEvent::V_MTE2>();
            AscendC::DataCopy(lu_stage[0], r_gm[src_sub * 4096u + offset], chunk_len);
            AscendC::DataCopy(lu_stage[1056], r_gm[src_sub * 4096u + 2048u + offset], chunk_len);
            fence<AscendC::HardEvent::MTE2_V>();
            two_sum_combine_buf(offset, chunk_len, lu_stage[0], lu_stage[1056], s_hi, s_lo, t_sum_4, y[0], y[1056]);
        }
    }
}

ELEC_FN void two_sum_combine_pair_small(
    uint32_t src_sub0, uint32_t src_sub1, uint32_t total_len, const AscendC::GlobalTensor<float>& r_gm,
    const AscendC::LocalTensor<float>& s_hi, const AscendC::LocalTensor<float>& s_lo,
    const AscendC::LocalTensor<float>& t_sum_4, const AscendC::LocalTensor<float>& lu_stage,
    const AscendC::LocalTensor<float>& alt_stage, const AscendC::LocalTensor<float>& y) {
    auto other_hi0 = lu_stage[0], other_lo0 = lu_stage[total_len];
    auto other_hi1 = alt_stage[0], other_lo1 = alt_stage[total_len];
    auto t2 = alt_stage[2048], d2 = alt_stage[3072], u2 = alt_stage[4096];
    auto d_tmp = y[0], u_tmp = y[1056];
    fence<AscendC::HardEvent::V_MTE2>();
    AscendC::DataCopy(other_hi0, r_gm[src_sub0 * 2048u], 2u * total_len);
    fence<AscendC::HardEvent::MTE2_V>();
    AscendC::DataCopy(other_hi1, r_gm[src_sub1 * 2048u], 2u * total_len);
    AscendC::Add(s_lo, s_lo, other_lo0, total_len);
    AscendC::Add(t_sum_4, s_hi, other_hi0, total_len);
    AscendC::PipeBarrier<PIPE_V>();
    fence<AscendC::HardEvent::MTE2_V>();
    AscendC::Sub(d_tmp, t_sum_4, s_hi, total_len);
    AscendC::Add(t2, t_sum_4, other_hi1, total_len);
    AscendC::Add(s_lo, s_lo, other_lo1, total_len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(u_tmp, t_sum_4, d_tmp, total_len);
    AscendC::Sub(other_hi0, other_hi0, d_tmp, total_len);
    AscendC::Sub(d2, t2, t_sum_4, total_len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(u_tmp, s_hi, u_tmp, total_len);
    AscendC::Add(s_lo, s_lo, other_hi0, total_len);
    AscendC::Sub(u2, t2, d2, total_len);
    AscendC::Sub(other_hi1, other_hi1, d2, total_len);
    AscendC::Adds(s_hi, t2, 0.0f, total_len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(s_lo, s_lo, u_tmp, total_len);
    AscendC::Sub(u2, t_sum_4, u2, total_len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(u2, u2, other_hi1, total_len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(s_lo, s_lo, u2, total_len);
    AscendC::PipeBarrier<PIPE_V>();
}

template <bool SMALL_CONST = false>
ELEC_FN void two_sum_combine_pair_from_gm(
    uint32_t src_sub0, uint32_t src_sub1, uint32_t total_len, const AscendC::GlobalTensor<float>& r_gm,
    const AscendC::LocalTensor<float>& s_hi, const AscendC::LocalTensor<float>& s_lo,
    const AscendC::LocalTensor<float>& t_sum_4, const AscendC::LocalTensor<float>& lu_stage,
    const AscendC::LocalTensor<float>& alt_stage, const AscendC::LocalTensor<float>& y) {
    if (SMALL_CONST || total_len <= 1024u) {
        two_sum_combine_pair_small(src_sub0, src_sub1, total_len, r_gm, s_hi, s_lo, t_sum_4, lu_stage, alt_stage, y);
        return;
    }
    if constexpr (!SMALL_CONST) {
        two_sum_combine_from_gm<false>(src_sub0, total_len, r_gm, s_hi, s_lo, t_sum_4, lu_stage, y);
        two_sum_combine_from_gm<false>(src_sub1, total_len, r_gm, s_hi, s_lo, t_sum_4, alt_stage, y);
    }
}

}  // namespace elec::lstsq_fast
