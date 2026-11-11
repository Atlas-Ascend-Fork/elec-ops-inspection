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
#include "lstsq_small_fast.h"

namespace elec::lstsq_fast {

template <typename T>
ELEC_FN void small_prepare_batch_bf16(
    uint64_t batch, uint32_t n, uint32_t k, uint32_t pad, uint32_t pad_a, uint64_t b_offset,
    const AscendC::GlobalTensor<T>& input_a,
    const AscendC::GlobalTensor<T>& input_b,
    const SmallUB<T>& ub) {
    const AscendC::DataCopyPadExtParams<T> no_pad{false, 0, 0, 0};
    const uint32_t raw_pad = (n + 15u) & ~15u;
    const AscendC::DataCopyExtParams a_load{static_cast<uint16_t>(n), static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(ub.raw, input_a[batch * n * n], a_load, no_pad);
    fence<AscendC::HardEvent::MTE2_V>();
    for (uint32_t r = 0; r < n; ++r) {
        AscendC::Cast(ub.M[r * pad], ub.raw[r * raw_pad], AscendC::RoundMode::CAST_NONE, pad_a);
    }
    fence<AscendC::HardEvent::V_MTE2>();
    const uint32_t b_elems = n * k;
    const AscendC::DataCopyExtParams b_load{1, static_cast<uint32_t>(b_elems * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(ub.raw, input_b[b_offset], b_load, no_pad);
    fence<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(ub.x_out, ub.raw, AscendC::RoundMode::CAST_NONE, (b_elems + 7u) & ~7u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(ub.x_out, ub.x_out, -1.0f, (b_elems + 7u) & ~7u);
    fence<AscendC::HardEvent::V_S>();
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t t = 0; t < k; ++t) {
            ub.M.SetValue(i * pad + pad_a + t, ub.x_out.GetValue(i * k + t));
        }
    }
    fence<AscendC::HardEvent::S_V>();
    AscendC::Gather(ub.col_vec0, ub.M[0], ub.col_offsets_u32, uint32_t(0), n);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
ELEC_FN void small_prepare_batch_gen(
    uint64_t batch, uint32_t block_idx, uint32_t n, uint32_t k, uint32_t pad, uint32_t pad_a, uint32_t pad_b, uint64_t b_offset,
    const AscendC::GlobalTensor<T>& input_a,
    const AscendC::GlobalTensor<T>& input_b,
    const SmallUB<T>& ub) {
    if constexpr (sizeof(T) == 4) {
        if (batch != block_idx) {
            small_load_batch_f32<T, 0u>(batch, n, k, pad, pad_a, pad_b, b_offset, input_a, input_b, ub);
        }
        fence<AscendC::HardEvent::MTE2_V>();
        AscendC::Gather(ub.col_vec0, ub.M[0], ub.col_offsets_u32, uint32_t(0), n);
        if (k == 1u) {
            AscendC::Brcb(ub.coefficients, ub.col_vec1, static_cast<uint8_t>(pad_a >> 3), {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ub.M[pad_a], ub.coefficients, -1.0f, 8u, static_cast<uint8_t>(n), {1, 1, static_cast<uint8_t>(pad >> 3), 1});
        } else if (k == 2u) {
            AscendC::Brcb(ub.coefficients, ub.x_out, static_cast<uint8_t>((2u * n + 7u) >> 3), {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ub.M[pad_a], ub.coefficients[0], -1.0f, 8u, static_cast<uint8_t>(n), {1, 1, static_cast<uint8_t>(pad >> 3), 2});
            uint64_t mask_lane1[2] = {0xfeULL, 0ULL};
            AscendC::Muls(ub.M[pad_a], ub.coefficients[8], -1.0f, mask_lane1, static_cast<uint8_t>(n), {1, 1, static_cast<uint8_t>(pad >> 3), 2});
        } else {
            AscendC::Muls(ub.M[pad_a], ub.M[pad_a], -1.0f, 8u, static_cast<uint8_t>(n),
                          {1, 1, static_cast<uint8_t>(pad >> 3), static_cast<uint8_t>(pad >> 3)});
            AscendC::PipeBarrier<PIPE_V>();
        }
    } else {
        small_prepare_batch_bf16<T>(batch, n, k, pad, pad_a, b_offset, input_a, input_b, ub);
    }
}

template <typename T>
ELEC_FN int small_elim_step_gen(uint32_t j, uint32_t n, uint32_t k, uint32_t pad, uint32_t pad_a, uint32_t rhs_col,
    uint32_t row_valid_pad, uint64_t& avail_mask, const SmallUB<T>& ub, const AscendC::BinaryRepeatParams& repeat) {
    const uint32_t c0 = (j + 1u < n) ? ((j + 1u) & ~7u) : (rhs_col & ~7u);
    const uint32_t rem = row_valid_pad - c0;
    uint64_t mask[2] = {avail_mask, 0ULL};
    AscendC::Abs(ub.abs_col, ub.col_vec0, mask, 1u, {1, 1, 8, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceMax<float>(ub.reduced, ub.abs_col, static_cast<int32_t>(n), 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_VALUE_INDEX);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    AscendC::Brcb(ub.coefficients, ub.col_vec0, static_cast<uint8_t>(pad_a >> 3), {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate(ub.abs_col, -1.0f, pad_a);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    const uint64_t red_val = ub.reduced_u64_ptr[0];
    const uint32_t max_bits = static_cast<uint32_t>(red_val);
    const uint32_t pivot = static_cast<uint32_t>(red_val >> 32);
    if (max_bits == 0u) { return SINGULAR; }
    if (max_bits >= 0x7f800000u || pivot >= n) { return NUMERIC_ERROR; }
    const float max_abs = bits_float(max_bits);
    const float neg_inv_abs = (max_abs != 0.0f) ? (-1.0f / max_abs) : 0.0f;
    AscendC::Duplicate(ub.coefficients[pivot * 8u], 0.0f, 8u);
    avail_mask &= ~(1ULL << pivot);
    const float neg_inv_diag = bits_float(float_bits(neg_inv_abs) ^ (ub.col_vec_u32_ptr[pivot] & 0x80000000u));
    AscendC::Muls(ub.M[pivot * pad + c0], ub.M[pivot * pad + c0], neg_inv_diag, rem);
    AscendC::PipeBarrier<PIPE_V>();

    const int32_t row_byte_off = static_cast<int32_t>((pivot * pad + (rhs_col & 7u)) * sizeof(float));
    for (uint32_t t = 0; t < k; ++t) { ub.rhs_pack_offsets_ptr[j * k + t] = row_byte_off + static_cast<int32_t>(t * sizeof(float)); }
    if (rem <= 64u) {
        AscendC::MulAddDst(ub.M[c0], ub.M[pivot * pad + c0], ub.coefficients, rem, static_cast<uint8_t>(n), repeat);
        AscendC::PipeBarrier<PIPE_V>();
        if (j + 1u < n) {
            AscendC::Gather(ub.col_vec0, ub.M[j + 1u], ub.col_offsets_u32, uint32_t(0), n);
            AscendC::PipeBarrier<PIPE_V>();
        }
    } else {
        AscendC::MulAddDst(ub.M[c0], ub.M[pivot * pad + c0], ub.coefficients, 64u, static_cast<uint8_t>(n), repeat);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(ub.col_vec0, ub.M[j + 1u], ub.col_offsets_u32, uint32_t(0), n);
        AscendC::MulAddDst(ub.M[c0 + 64u], ub.M[pivot * pad + c0 + 64u], ub.coefficients, rem - 64u, static_cast<uint8_t>(n), repeat);
        AscendC::PipeBarrier<PIPE_V>();
    }
    return SUCCESS;
}

template <typename T>
ELEC_FN void small_store_batch_gen(
    uint64_t batch, uint64_t batches, uint32_t block_num, uint32_t n, uint32_t k, uint32_t rhs_col, bool need_zero_r,
    const AscendC::GlobalTensor<T>& output_x,
    const AscendC::GlobalTensor<T>& output_r,
    const SmallUB<T>& ub) {
    fence<AscendC::HardEvent::S_V>();
    AscendC::Gather(ub.x_out, ub.M[rhs_col & ~7u], ub.rhs_pack_offsets_u32, uint32_t(0), n * k);
    if constexpr (sizeof(T) == 4) {
        fence<AscendC::HardEvent::V_MTE3>();
        if (((n * k) & 7u) == 0u) {
            AscendC::DataCopy(output_x[batch * n * k], ub.x_out, n * k);
        } else {
            const AscendC::DataCopyExtParams x_store{1, static_cast<uint32_t>(n * k * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(output_x[batch * n * k], ub.x_out, x_store);
        }
    } else {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(ub.raw, ub.x_out, AscendC::RoundMode::CAST_RINT, (n * k + 7u) & ~7u);
        fence<AscendC::HardEvent::V_MTE3>();
        const AscendC::DataCopyExtParams x_store{1, static_cast<uint32_t>(n * k * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(output_x[batch * n * k], ub.raw, x_store);
    }
    if (batch + block_num < batches) {
        fence<AscendC::HardEvent::MTE3_V>();
    } else if (need_zero_r) {
        const AscendC::DataCopyExtParams r_copy{1, static_cast<uint32_t>(batches * k * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(output_r, ub.zero_buf.template ReinterpretCast<T>(), r_copy);
    }
}

template <typename T, uint32_t REGIME, bool ZERO_R, bool ALIGNED_N>
ELEC_FN int small_run_fast_regime(
    uint32_t block_idx, uint32_t block_num, uint32_t n, uint32_t pad, uint32_t pad_a,
    uint32_t rhs_col, uint32_t row_valid_pad, uint64_t n_mask,
    const AscendC::GlobalTensor<T>& output_x,
    const AscendC::GlobalTensor<T>& output_r,
    const SmallUB<T>& ub,
    const AscendC::BinaryRepeatParams& repeat,
    const AscendC::BinaryRepeatParams& repeat_rhs8) {
    constexpr uint32_t K_REG = (REGIME <= 2u) ? 1u : 2u;
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::Gather(ub.col_vec0, ub.M[0], ub.col_offsets_u32, uint32_t(0), n);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
    small_inject_rhs_fast<T, REGIME, ZERO_R>(block_idx, block_num, n, pad, pad_a, output_r, ub);
    uint64_t avail_mask = n_mask;
    for (uint32_t j = 0; j + 1u < n; ++j) {
        small_elim_step_fast<T, REGIME, ZERO_R>(
            j, block_idx, block_num, n, pad, pad_a, rhs_col, row_valid_pad, avail_mask, output_r, ub, repeat, repeat_rhs8);
    }
    small_elim_last_col_fast<T, K_REG>(n, pad, pad_a, rhs_col, row_valid_pad, avail_mask, ub, repeat_rhs8);
    AscendC::Gather(ub.x_out, ub.M[rhs_col & ~7u], ub.rhs_pack_offsets_u32, uint32_t(0), n * K_REG);
    fence<AscendC::HardEvent::V_MTE3>();
    if constexpr (ALIGNED_N) {
        AscendC::DataCopy(output_x[static_cast<uint64_t>(block_idx) * n * K_REG], ub.x_out, n * K_REG);
    } else {
        const AscendC::DataCopyExtParams x_store{1, static_cast<uint32_t>(n * K_REG * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(output_x[static_cast<uint64_t>(block_idx) * n * K_REG], ub.x_out, x_store);
    }
    return SUCCESS;
}

template <typename T, uint32_t REGIME, bool ZERO_R>
ELEC_FN void small_init_offsets_and_zero_r(
    uint32_t block_idx, uint32_t pad, uint32_t pad_a, bool need_zero_r, GM_ADDR residuals,
    AscendC::GlobalTensor<T>& output_r, const SmallUB<T>& ub) {
    if (need_zero_r) {
        output_r.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(residuals));
        AscendC::Duplicate(ub.zero_buf, 0.0f, (REGIME != 0u) ? 8u : 64u);
    }
    AscendC::CreateVecIndex(ub.col_offsets, int32_t(0), pad_a);
    AscendC::Duplicate(ub.abs_col, -1.0f, (REGIME != 0u) ? pad_a : 64u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(ub.col_offsets, ub.col_offsets, static_cast<int32_t>(pad * sizeof(float)), pad_a);
    if constexpr (REGIME != 0u && ZERO_R) {
        if (block_idx == 0u) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <uint32_t REGIME>
struct SmallRegimeTraits {
    static constexpr uint32_t K_REG = (REGIME <= 2u) ? 1u : 2u;
    static constexpr bool PACK_TAIL = (REGIME == 2u || REGIME == 3u);
    static constexpr bool ALIGNED_N = (REGIME == 1u || REGIME == 4u);
    static constexpr uint32_t PAD_A_REG = (REGIME <= 2u) ? 16u : ((REGIME == 3u) ? 56u : 64u);
    static constexpr uint32_t ROW_VALID_PAD_REG = (REGIME == 1u) ? 24u : ((REGIME == 2u) ? 16u : ((REGIME == 3u) ? 56u : 72u));
    static constexpr uint32_t PAD_REG = (REGIME <= 2u) ? 24u : ((REGIME == 3u) ? 56u : 72u);
};

struct SmallParams {
    uint64_t batches;
    uint64_t rhs_batch_stride;
    uint64_t n_mask;
    uint32_t n, k, pad_a, rhs_col, row_valid_pad, pad, pad_b;
    bool need_zero_r;
};

template <uint32_t REGIME, bool ZERO_R>
ELEC_FN SmallParams make_small_params(
    const __gm__ Config* c_ptr, uint32_t block_idx, uint32_t block_num, GM_ADDR residuals) {
    using Tr = SmallRegimeTraits<REGIME>;
    SmallParams p;
    p.batches = (REGIME != 0u) ? static_cast<uint64_t>(block_num) : c_ptr->batch;
    p.n = Tr::ALIGNED_N ? Tr::PAD_A_REG : static_cast<uint32_t>(c_ptr->n);
    p.k = (REGIME != 0u) ? Tr::K_REG : static_cast<uint32_t>(c_ptr->k);
    p.rhs_batch_stride = (REGIME != 0u) ? static_cast<uint64_t>(p.n * Tr::K_REG) : c_ptr->rhs_batch_stride;
    p.pad_a = (REGIME != 0u) ? Tr::PAD_A_REG : ((p.n + 7u) & ~7u);
    p.rhs_col = Tr::PACK_TAIL ? p.n : p.pad_a;
    p.row_valid_pad = (REGIME != 0u) ? Tr::ROW_VALID_PAD_REG : (Tr::PACK_TAIL ? p.pad_a : (p.pad_a + 8u));
    p.pad = (REGIME != 0u) ? Tr::PAD_REG : (((p.row_valid_pad >> 3) | 1u) << 3);
    p.pad_b = (REGIME != 0u) ? (Tr::PAD_REG - Tr::PAD_A_REG) : (p.pad - p.pad_a);
    p.n_mask = (p.n == 64u) ? ~0ULL : ((1ULL << p.n) - 1ULL);
    p.need_zero_r = (REGIME != 0u) ? (ZERO_R && block_idx == 0u)
        : (((c_ptr->flags & LSTSQ_SQUARE_RESIDUAL) != 0) && residuals != nullptr && block_idx == 0u);
    return p;
}

template <typename T, uint32_t REGIME = 0u, bool ZERO_R = false>
ELEC_FN int solve_square_small(
    const __gm__ Config* c_ptr, GM_ADDR A, GM_ADDR B, GM_ADDR solution, GM_ADDR residuals) {
    const uint32_t block_idx = AscendC::GetBlockIdx();
    const uint32_t block_num = AscendC::GetBlockNum();
    const SmallParams p = make_small_params<REGIME, ZERO_R>(c_ptr, block_idx, block_num, residuals);
    if constexpr (REGIME == 0u) {
        if (block_idx >= p.batches) return SUCCESS;
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work_buf;
    SmallUB<T> ub;
    init_small_ub(ub, pipe, work_buf);

    AscendC::GlobalTensor<T> input_a, input_b, output_x, output_r;
    input_a.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(A));
    input_b.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(B));
    output_x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(solution));
    if constexpr (sizeof(T) == 4) {
        small_load_batch_f32<T, REGIME>(block_idx, p.n, p.k, p.pad, p.pad_a, p.pad_b, block_idx * p.rhs_batch_stride, input_a, input_b, ub);
    }
    small_init_offsets_and_zero_r<T, REGIME, ZERO_R>(block_idx, p.pad, p.pad_a, p.need_zero_r, residuals, output_r, ub);

    const AscendC::BinaryRepeatParams repeat{1, 1, 0, static_cast<uint8_t>(p.pad >> 3), 0, 1};
    const AscendC::BinaryRepeatParams repeat_rhs8{static_cast<uint8_t>(p.pad >> 3), 0, 1, static_cast<uint8_t>((p.pad >> 3) * 8u), 0, 8};

    if constexpr (REGIME != 0u && sizeof(T) == 4) {
        constexpr bool ALIGNED_N = (REGIME == 1u || REGIME == 4u);
        return small_run_fast_regime<T, REGIME, ZERO_R, ALIGNED_N>(
            block_idx, block_num, p.n, p.pad, p.pad_a, p.rhs_col, p.row_valid_pad, p.n_mask, output_x, output_r, ub, repeat, repeat_rhs8);
    } else {
        for (uint64_t batch = block_idx; batch < p.batches; batch += block_num) {
            small_prepare_batch_gen<T>(batch, block_idx, p.n, p.k, p.pad, p.pad_a, p.pad_b, batch * p.rhs_batch_stride, input_a, input_b, ub);
            uint64_t avail_mask = p.n_mask;
            for (uint32_t j = 0; j < p.n; ++j) {
                const int status = small_elim_step_gen<T>(j, p.n, p.k, p.pad, p.pad_a, p.rhs_col, p.row_valid_pad, avail_mask, ub, repeat);
                if (status != SUCCESS) return status;
            }
            small_store_batch_gen<T>(batch, p.batches, block_num, p.n, p.k, p.rhs_col, p.need_zero_r, output_x, output_r, ub);
        }
        return SUCCESS;
    }
}

}  // namespace elec::lstsq_fast
