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
struct SmallUB {
    AscendC::LocalTensor<float> M, coefficients, col_vec0, abs_col, reduced, col_vec1, zero_buf, x_out;
    AscendC::LocalTensor<int32_t> col_offsets, rhs_pack_offsets;
    AscendC::LocalTensor<uint32_t> col_offsets_u32, rhs_pack_offsets_u32;
    AscendC::LocalTensor<T> raw;
    volatile __ubuf__ const uint32_t* col_vec_u32_ptr;
    __ubuf__ int32_t* rhs_pack_offsets_ptr;
    volatile __ubuf__ const uint64_t* reduced_u64_ptr;
};

template <typename T>
ELEC_FN void init_small_ub(
    SmallUB<T>& ub, AscendC::TPipe& pipe, AscendC::TBuf<AscendC::TPosition::VECCALC>& work_buf) {
    pipe.InitBuffer(work_buf, ((sizeof(T) == 4) ? 7744u : 9792u) * sizeof(float));
    auto base = work_buf.Get<float>();
    ub.M = base[0];
    ub.coefficients = base[4672];
    ub.col_vec0 = base[5760];
    ub.abs_col = base[5856];
    ub.reduced = base[6048];
    ub.col_vec1 = base[6304];
    ub.zero_buf = base[6400];
    ub.x_out = base[6560];
    auto offsets = base[7168].template ReinterpretCast<int32_t>();
    ub.col_offsets = offsets[0];
    ub.rhs_pack_offsets = offsets[64];
    ub.col_offsets_u32 = ub.col_offsets.template ReinterpretCast<uint32_t>();
    ub.rhs_pack_offsets_u32 = ub.rhs_pack_offsets.template ReinterpretCast<uint32_t>();
    ub.raw = base[7744].template ReinterpretCast<T>();
    ub.col_vec_u32_ptr = reinterpret_cast<volatile __ubuf__ const uint32_t*>(ub.col_vec0.GetPhyAddr());
    ub.rhs_pack_offsets_ptr = reinterpret_cast<__ubuf__ int32_t*>(ub.rhs_pack_offsets.GetPhyAddr());
    ub.reduced_u64_ptr = reinterpret_cast<volatile __ubuf__ const uint64_t*>(ub.reduced.GetPhyAddr());
}

template <typename T, uint32_t REGIME>
ELEC_FN void small_load_batch_f32(
    uint64_t batch, uint32_t n, uint32_t k, uint32_t pad, uint32_t pad_a, uint32_t pad_b, uint64_t b_offset,
    const AscendC::GlobalTensor<T>& input_a,
    const AscendC::GlobalTensor<T>& input_b,
    const SmallUB<T>& ub) {
    constexpr uint32_t K_REG = (REGIME <= 2u) ? 1u : 2u;
    constexpr bool ALIGNED_N = (REGIME == 1u || REGIME == 4u);
    const AscendC::DataCopyPadExtParams<T> no_pad{false, 0, 0, 0};
    if ((REGIME != 0u && ALIGNED_N) || (REGIME == 0u && (n & 7u) == 0u)) {
        const AscendC::DataCopyParams a_copy{
            static_cast<uint16_t>(n), static_cast<uint16_t>(n >> 3), 0, static_cast<uint16_t>(pad_b >> 3)};
        AscendC::DataCopy(ub.M, input_a[batch * n * n], a_copy);
    } else {
        const AscendC::DataCopyExtParams a_load{
            static_cast<uint16_t>(n), static_cast<uint32_t>(n * sizeof(float)), 0, static_cast<uint32_t>(pad_b >> 3), 0};
        AscendC::DataCopyPad(ub.M, input_a[batch * n * n], a_load, no_pad);
    }
    if constexpr (REGIME != 0u) {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        if constexpr (REGIME == 1u) {
            AscendC::DataCopy(ub.col_vec1, input_b[b_offset], n);
        } else if constexpr (ALIGNED_N) {
            AscendC::DataCopy(ub.x_out, input_b[b_offset], K_REG * n);
        } else {
            const AscendC::DataCopyExtParams b_load2{1, static_cast<uint32_t>(K_REG * n * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(ub.x_out, input_b[b_offset], b_load2, no_pad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
    } else if (k <= 2u) {
        const auto& dst = (k == 1u) ? ub.col_vec1 : ub.x_out;
        const uint32_t elems = k * n;
        if ((n & 7u) == 0u) {
            AscendC::DataCopy(dst, input_b[b_offset], elems);
        } else {
            const AscendC::DataCopyExtParams b_load{1, static_cast<uint32_t>(elems * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(dst, input_b[b_offset], b_load, no_pad);
        }
    } else {
        const AscendC::DataCopyExtParams b_load{
            static_cast<uint16_t>(n), static_cast<uint32_t>(k * sizeof(float)), 0, static_cast<uint32_t>((pad - 8u) >> 3), 0};
        AscendC::DataCopyPad(ub.M[pad_a], input_b[b_offset], b_load, no_pad);
    }
}

template <typename T, uint32_t K_REG>
ELEC_FN void small_write_zero_r_fast(
    uint32_t block_idx, uint32_t block_num,
    const AscendC::GlobalTensor<T>& output_r,
    const AscendC::LocalTensor<float>& zero_buf) {
    if (block_idx != 0u) return;
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    const uint32_t r_elems = block_num * K_REG;
    if ((r_elems & 7u) == 0u) {
        AscendC::DataCopy(output_r, zero_buf.template ReinterpretCast<T>(), r_elems);
    } else {
        const AscendC::DataCopyExtParams r_copy{1, static_cast<uint32_t>(r_elems * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(output_r, zero_buf.template ReinterpretCast<T>(), r_copy);
    }
}

template <typename T, uint32_t REGIME, bool ZERO_R>
ELEC_FN void small_inject_rhs_fast(
    uint32_t block_idx, uint32_t block_num, uint32_t n, uint32_t pad, uint32_t pad_a,
    const AscendC::GlobalTensor<T>& output_r,
    const SmallUB<T>& ub) {
    constexpr uint32_t K_REG = (REGIME <= 2u) ? 1u : 2u;
    if constexpr (REGIME == 1u) {
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }
    AscendC::Brcb(ub.coefficients, ub.x_out, static_cast<uint8_t>((K_REG * n + 7u) >> 3), {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (REGIME == 2u) {
        const uint64_t lane_mask = 0x0101010101010101ULL << (n & 7u);
        uint64_t mask_lane[2] = {lane_mask, 0ULL};
        AscendC::Muls(ub.M[pad_a - 8u], ub.coefficients, -1.0f, mask_lane, static_cast<uint8_t>(pad_a >> 3),
                      {static_cast<uint16_t>(pad >> 3), 1, static_cast<uint8_t>((pad >> 3) * 8u), 8});
    } else if constexpr (REGIME == 3u) {
        const uint64_t lane0 = 0x0101010101010101ULL << (n & 7u);
        uint64_t mask_lane0[2] = {lane0, 0ULL};
        uint64_t mask_lane1[2] = {lane0 << 1u, 0ULL};
        AscendC::Muls(ub.M[pad_a - 8u], ub.coefficients[0], -1.0f, mask_lane0, static_cast<uint8_t>(pad_a >> 3),
                      {static_cast<uint16_t>(pad >> 3), 2, static_cast<uint8_t>((pad >> 3) * 8u), 16});
        AscendC::Muls(ub.M[pad_a - 8u], ub.coefficients[8], -1.0f, mask_lane1, static_cast<uint8_t>(pad_a >> 3),
                      {static_cast<uint16_t>(pad >> 3), 2, static_cast<uint8_t>((pad >> 3) * 8u), 16});
    } else {
        AscendC::Muls(ub.M[pad_a], ub.coefficients[0], -1.0f, 64u, static_cast<uint8_t>(pad_a >> 3),
                      {static_cast<uint16_t>(pad >> 3), 2, static_cast<uint8_t>((pad >> 3) * 8u), 16});
        uint64_t mask_lane1[2] = {0xfefefefefefefefeULL, 0ULL};
        AscendC::Muls(ub.M[pad_a], ub.coefficients[8], -1.0f, mask_lane1, static_cast<uint8_t>(pad_a >> 3),
                      {static_cast<uint16_t>(pad >> 3), 2, static_cast<uint8_t>((pad >> 3) * 8u), 16});
    }
    if constexpr (ZERO_R) {
        small_write_zero_r_fast<T, K_REG>(block_idx, block_num, output_r, ub.zero_buf);
    }
}

ELEC_FN void small_mul_add_rem32(
    uint32_t c0, uint32_t pivot, uint32_t pad, uint32_t pad_a, uint32_t rem,
    const AscendC::LocalTensor<float>& M,
    const AscendC::LocalTensor<float>& coefficients,
    const AscendC::BinaryRepeatParams& repeat_rhs8) {
    const uint8_t rep8 = static_cast<uint8_t>(pad_a >> 3);
    AscendC::MulAddDst(M[c0], M[pivot * pad + c0], coefficients, 64u, rep8, repeat_rhs8);
    if (rem >= 16u) {
        AscendC::MulAddDst(M[c0 + 8u], M[pivot * pad + c0 + 8u], coefficients, 64u, rep8, repeat_rhs8);
    }
    if (rem >= 24u) {
        AscendC::MulAddDst(M[c0 + 16u], M[pivot * pad + c0 + 16u], coefficients, 64u, rep8, repeat_rhs8);
    }
    if (rem >= 32u) {
        AscendC::MulAddDst(M[c0 + 24u], M[pivot * pad + c0 + 24u], coefficients, 64u, rep8, repeat_rhs8);
    }
}

template <typename T, uint32_t REGIME>
ELEC_FN void small_elim_update_fast(
    uint32_t j, uint32_t n, uint32_t pad, uint32_t pad_a, uint32_t c0, uint32_t rem, uint32_t pivot,
    const SmallUB<T>& ub,
    const AscendC::BinaryRepeatParams& repeat,
    const AscendC::BinaryRepeatParams& repeat_rhs8) {
    if constexpr (REGIME == 4u) {
        if (rem <= 32u) {
            small_mul_add_rem32(c0, pivot, pad, pad_a, rem, ub.M, ub.coefficients, repeat_rhs8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(ub.col_vec0, ub.M[j + 1u], ub.col_offsets_u32, uint32_t(0), n);
        } else if (rem <= 64u) {
            AscendC::MulAddDst(ub.M[c0], ub.M[pivot * pad + c0], ub.coefficients, rem, static_cast<uint8_t>(n), repeat);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(ub.col_vec0, ub.M[j + 1u], ub.col_offsets_u32, uint32_t(0), n);
        } else {
            AscendC::MulAddDst(ub.M[0], ub.M[pivot * pad], ub.coefficients, 64u, static_cast<uint8_t>(n), repeat);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(ub.col_vec0, ub.M[j + 1u], ub.col_offsets_u32, uint32_t(0), n);
            AscendC::MulAddDst(ub.M[64], ub.M[pivot * pad + 64], ub.coefficients, 64u, static_cast<uint8_t>(pad_a >> 3), repeat_rhs8);
        }
    } else if constexpr (REGIME == 3u) {
        if (rem <= 32u) {
            small_mul_add_rem32(c0, pivot, pad, pad_a, rem, ub.M, ub.coefficients, repeat_rhs8);
        } else {
            AscendC::MulAddDst(ub.M[c0], ub.M[pivot * pad + c0], ub.coefficients, rem, static_cast<uint8_t>(n), repeat);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(ub.col_vec0, ub.M[j + 1u], ub.col_offsets_u32, uint32_t(0), n);
    } else {
        if (rem == 8u) {
            AscendC::MulAddDst(ub.M[c0], ub.M[pivot * pad + c0], ub.coefficients, 64u, static_cast<uint8_t>(pad_a >> 3), repeat_rhs8);
        } else {
            AscendC::MulAddDst(ub.M[c0], ub.M[pivot * pad + c0], ub.coefficients, rem, static_cast<uint8_t>(n), repeat);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(ub.col_vec0, ub.M[j + 1u], ub.col_offsets_u32, uint32_t(0), n);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, uint32_t REGIME, bool ZERO_R>
ELEC_FN void small_find_pivot_fast(
    uint32_t j, uint32_t block_idx, uint32_t block_num, uint32_t n, uint32_t pad, uint32_t pad_a, uint64_t avail_mask,
    const AscendC::GlobalTensor<T>& output_r,
    const SmallUB<T>& ub) {
    constexpr uint32_t K_REG = (REGIME <= 2u) ? 1u : 2u;
    uint64_t mask[2] = {avail_mask, 0ULL};
    AscendC::Abs(ub.abs_col, ub.col_vec0, mask, 1u, {1, 1, 8, 8});
    if constexpr (REGIME == 1u) {
        if (j == 0u) {
            AscendC::Brcb(ub.coefficients, ub.col_vec1, static_cast<uint8_t>(pad_a >> 3), {1, 8});
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceMax<float>(ub.reduced, ub.abs_col, static_cast<int32_t>(n), 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_VALUE_INDEX);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    if constexpr (REGIME == 1u) {
        if (j == 0u) {
            AscendC::Muls(ub.M[pad_a], ub.coefficients, -1.0f, 64u, static_cast<uint8_t>(pad_a >> 3),
                          {static_cast<uint16_t>(pad >> 3), 1, static_cast<uint8_t>((pad >> 3) * 8u), 8});
            AscendC::Duplicate(ub.abs_col, -1.0f, pad_a);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(ub.coefficients, ub.col_vec0, static_cast<uint8_t>(pad_a >> 3), {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::Brcb(ub.coefficients, ub.col_vec0, static_cast<uint8_t>(pad_a >> 3), {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(ub.abs_col, -1.0f, pad_a);
        }
        if constexpr (ZERO_R) {
            if (j == 0u) {
                small_write_zero_r_fast<T, K_REG>(block_idx, block_num, output_r, ub.zero_buf);
            }
        }
    } else {
        AscendC::Brcb(ub.coefficients, ub.col_vec0, static_cast<uint8_t>(pad_a >> 3), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(ub.abs_col, -1.0f, pad_a);
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
}

template <typename T, uint32_t REGIME, bool ZERO_R>
ELEC_FN void small_elim_step_fast(
    uint32_t j, uint32_t block_idx, uint32_t block_num, uint32_t n, uint32_t pad, uint32_t pad_a,
    uint32_t rhs_col, uint32_t row_valid_pad, uint64_t& avail_mask,
    const AscendC::GlobalTensor<T>& output_r,
    const SmallUB<T>& ub,
    const AscendC::BinaryRepeatParams& repeat,
    const AscendC::BinaryRepeatParams& repeat_rhs8) {
    constexpr uint32_t K_REG = (REGIME <= 2u) ? 1u : 2u;
    const uint32_t c0 = (j + 1u) & ~7u;
    const uint32_t rem = row_valid_pad - c0;
    small_find_pivot_fast<T, REGIME, ZERO_R>(j, block_idx, block_num, n, pad, pad_a, avail_mask, output_r, ub);
    const uint64_t red_val = ub.reduced_u64_ptr[0];
    const uint32_t max_bits = static_cast<uint32_t>(red_val);
    const uint32_t pivot = static_cast<uint32_t>(red_val >> 32);
    const float max_abs = bits_float(max_bits);
    const float neg_inv_abs = (max_abs != 0.0f) ? (-1.0f / max_abs) : 0.0f;
    AscendC::Duplicate(ub.coefficients[pivot * 8u], 0.0f, 8u);
    avail_mask &= ~(1ULL << pivot);
    const uint32_t sign_bit = ub.col_vec_u32_ptr[pivot] & 0x80000000u;
    const int32_t row_byte_off = static_cast<int32_t>((pivot * pad + (rhs_col & 7u)) * sizeof(float));
    if constexpr (K_REG == 1u) {
        ub.rhs_pack_offsets_ptr[j] = row_byte_off;
    } else {
        ub.rhs_pack_offsets_ptr[2u * j] = row_byte_off;
        ub.rhs_pack_offsets_ptr[2u * j + 1u] = row_byte_off + 4;
    }
    const float neg_inv_diag = bits_float(float_bits(neg_inv_abs) ^ sign_bit);
    AscendC::Muls(ub.M[pivot * pad + c0], ub.M[pivot * pad + c0], neg_inv_diag, rem);
    AscendC::PipeBarrier<PIPE_V>();
    small_elim_update_fast<T, REGIME>(j, n, pad, pad_a, c0, rem, pivot, ub, repeat, repeat_rhs8);
}

template <typename T, uint32_t K_REG>
ELEC_FN void small_elim_last_col_fast(
    uint32_t n, uint32_t pad, uint32_t pad_a, uint32_t rhs_col, uint32_t row_valid_pad, uint64_t avail_mask,
    const SmallUB<T>& ub,
    const AscendC::BinaryRepeatParams& repeat_rhs8) {
    AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
    const uint32_t j = n - 1u;
    const uint32_t c0 = rhs_col & ~7u;
    const uint32_t rem = row_valid_pad - c0;
    const uint32_t pivot = static_cast<uint32_t>(AscendC::ScalarGetSFFValue<1>(avail_mask));
    AscendC::Brcb(ub.coefficients, ub.col_vec0, static_cast<uint8_t>(pad_a >> 3), {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Duplicate(ub.coefficients[pivot * 8u], 0.0f, 8u);
    const int32_t row_byte_off = static_cast<int32_t>((pivot * pad + (rhs_col & 7u)) * sizeof(float));
    if constexpr (K_REG == 1u) {
        ub.rhs_pack_offsets_ptr[j] = row_byte_off;
    } else {
        ub.rhs_pack_offsets_ptr[2u * j + 0u] = row_byte_off;
        ub.rhs_pack_offsets_ptr[2u * j + 1u] = row_byte_off + static_cast<int32_t>(sizeof(float));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
    const float diag = bits_float(ub.col_vec_u32_ptr[pivot]);
    const float neg_inv_diag = (diag != 0.0f) ? (-1.0f / diag) : 0.0f;
    AscendC::Muls(ub.M[pivot * pad + c0], ub.M[pivot * pad + c0], neg_inv_diag, rem);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::MulAddDst(ub.M[c0], ub.M[pivot * pad + c0], ub.coefficients,
                       64u, static_cast<uint8_t>(pad_a >> 3), repeat_rhs8);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    AscendC::PipeBarrier<PIPE_V>();
}

}  // namespace elec::lstsq_fast
