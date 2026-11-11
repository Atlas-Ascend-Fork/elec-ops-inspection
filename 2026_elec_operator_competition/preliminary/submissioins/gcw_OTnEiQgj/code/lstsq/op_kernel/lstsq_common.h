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
#include "linalg_lstsq_tiling.h"
#ifdef ELEC_CPU
#include <cmath>
#define ELEC_FN inline
#else
#include "kernel_operator.h"
#define ELEC_FN __aicore__ inline
#endif

namespace elec {
ELEC_FN uint32_t float_bits(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;
    return v.u;
}

ELEC_FN float bits_float(uint32_t x) {
    union { float f; uint32_t u; } v;
    v.u = x;
    return v.f;
}

ELEC_FN bool is_finite(float x) {
    return (float_bits(x) & 0x7f800000u) != 0x7f800000u;
}

ELEC_FN float bf16_to_float(uint16_t h) {
    return bits_float(uint32_t(h) << 16);
}

ELEC_FN uint16_t float_to_bf16(float x) {
    uint32_t u = float_bits(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}

class Tensor {
public:
    ELEC_FN Tensor() = default;
    ELEC_FN Tensor(GM_ADDR p, uint32_t type = F32) : ptr_(p), type_(type) {
        if (p) {
            f_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p));
            h_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(p));
        }
    }
    ELEC_FN bool valid() const { return ptr_ != nullptr; }
    ELEC_FN Tensor slice(uint64_t offset) const {
        Tensor t = *this;
        t.offset_ += offset;
        return t;
    }
    ELEC_FN float get(uint64_t i) const {
        i += offset_;
        if (type_ == F32) return f_.GetValue(i);
        return bf16_to_float(h_.GetValue(i));
    }
    ELEC_FN void set(uint64_t i, float x) {
        i += offset_;
        if (type_ == F32) {
            f_.SetValue(i, x);
        } else {
            h_.SetValue(i, float_to_bf16(x));
        }
    }
    ELEC_FN void flush() {
        AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                          AscendC::DcciDst::CACHELINE_OUT>(f_);
    }
private:
    GM_ADDR ptr_ = nullptr;
    AscendC::GlobalTensor<float> f_;
    AscendC::GlobalTensor<uint16_t> h_;
    uint32_t type_ = F32;
    uint64_t offset_ = 0;
};

class Math {
public:
    ELEC_FN Math() {
        pipe_.InitBuffer(buffer_, 32);
        data_ = buffer_.Get<float>();
        for (uint32_t i = 0; i < 8; ++i) {
            data_.SetValue(i, 0.0f);
        }
    }
    ELEC_FN float sqrt(float x) {
        data_.SetValue(0, x);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        AscendC::Sqrt(data_, data_, 8);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        return data_.GetValue(0);
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
    AscendC::LocalTensor<float> data_;
};

ELEC_FN void bind_template_tensors(
    const TemplateTilingData& t, GM_ADDR* inputs, GM_ADDR* outputs, Tensor* in, Tensor* out) {
    for (uint32_t i = 0; i < 2; ++i) in[i] = Tensor(inputs[i], t.input_types[i]);
    for (uint32_t i = 0; i < 2; ++i) out[i] = Tensor(outputs[i], t.output_types[i]);
}

ELEC_FN void check_template_status(int status) {
    if (status != SUCCESS) AscendC::Trap();
}
}  // namespace elec

namespace elec::lstsq_fast {

template <AscendC::HardEvent Event>
ELEC_FN void fence() {
    AscendC::SetFlag<Event>(0);
    AscendC::WaitFlag<Event>(0);
}

struct LargeUB {
    AscendC::LocalTensor<float> M, lu_buf0, lu_buf1, lu_stage_p1, lu_panel7, A_orig;
    AscendC::LocalTensor<float> x0, s_hi, x_out, t_sum_4, lu_stage_pad, lu_stage;
    AscendC::LocalTensor<float> y_pad, y, coeff_pad, coefficients, s_lo;
    AscendC::LocalTensor<float> col_vec, lu_col, inv_diag_stage, abs_col, pivot_row;
    AscendC::LocalTensor<float> row_tmp, inv_diag_vec, scratch, reduced, unit_masks;
    AscendC::LocalTensor<int32_t> piv_stage, col_offsets, y_col_offsets;
    AscendC::LocalTensor<uint32_t> col_offsets_u32, y_col_offsets_u32;
    volatile __ubuf__ float* col_vec_ptr;
    volatile __ubuf__ float* abs_col_ptr;
    volatile __ubuf__ float* inv_diag_ptr;
    volatile __ubuf__ int32_t* piv_stage_ptr;
    volatile __ubuf__ float* inv_diag_stage_ptr;
    __ubuf__ const float* x0_ptr;
    volatile __ubuf__ const uint64_t* reduced_u64_ptr;
};

ELEC_FN void init_large_ub(
    LargeUB& ub, AscendC::TPipe& pipe, AscendC::TBuf<AscendC::TPosition::VECCALC>& work_buf) {
    pipe.InitBuffer(work_buf, 48896u * sizeof(float));
    auto base = work_buf.Get<float>();
    ub.M = base[0];
    ub.lu_buf0 = base[0];
    ub.lu_buf1 = base[8192];
    ub.lu_stage_p1 = base[10560];
    ub.lu_panel7 = base[16384];
    ub.A_orig = base[24576];
    ub.x0 = base[34816];
    ub.s_hi = base[36928];
    ub.x_out = base[36928];
    ub.t_sum_4 = base[38976];
    ub.lu_stage_pad = base[39936];
    ub.lu_stage = base[40000];
    ub.y_pad = base[42048];
    ub.y = base[42112];
    ub.coeff_pad = base[44160];
    ub.coefficients = base[44224];
    ub.s_lo = base[44224];
    auto vec = base[46336];
    ub.col_vec = vec[0];
    ub.lu_col = vec[256];
    ub.piv_stage = vec[256].template ReinterpretCast<int32_t>();
    ub.inv_diag_stage = vec[320];
    ub.abs_col = vec[512];
    ub.pivot_row = vec[832];
    ub.row_tmp = vec[1088];
    ub.inv_diag_vec = vec[1344];
    ub.scratch = vec[1600];
    ub.reduced = vec[1856];
    ub.unit_masks = vec[1920];
    ub.col_offsets = vec[2048].template ReinterpretCast<int32_t>();
    ub.y_col_offsets = vec[2304].template ReinterpretCast<int32_t>();
    ub.col_offsets_u32 = ub.col_offsets.template ReinterpretCast<uint32_t>();
    ub.y_col_offsets_u32 = ub.y_col_offsets.template ReinterpretCast<uint32_t>();
    ub.col_vec_ptr = reinterpret_cast<volatile __ubuf__ float*>(ub.col_vec.GetPhyAddr());
    ub.abs_col_ptr = reinterpret_cast<volatile __ubuf__ float*>(ub.abs_col.GetPhyAddr());
    ub.inv_diag_ptr = reinterpret_cast<volatile __ubuf__ float*>(ub.inv_diag_vec.GetPhyAddr());
    ub.piv_stage_ptr = reinterpret_cast<volatile __ubuf__ int32_t*>(ub.piv_stage.GetPhyAddr());
    ub.inv_diag_stage_ptr = reinterpret_cast<volatile __ubuf__ float*>(ub.inv_diag_stage.GetPhyAddr());
    ub.x0_ptr = reinterpret_cast<__ubuf__ const float*>(ub.x0.GetPhyAddr());
    ub.reduced_u64_ptr = reinterpret_cast<volatile __ubuf__ const uint64_t*>(ub.reduced.GetPhyAddr());
}

ELEC_FN void reduce_max_abs_col_tail(
    uint32_t j, uint32_t j0, uint32_t n, uint32_t pad_n, const LargeUB& ub) {
    const uint32_t rem_a = pad_n - j0;
    const uint32_t chunk0 = rem_a < 64u ? rem_a : 64u;
    const uint64_t c_mask = (chunk0 == 64u) ? ~0ULL : ((1ULL << chunk0) - 1ULL);
    uint64_t mask[2] = {c_mask & (~0ULL << (j - j0)), 0ULL};
    AscendC::Abs(ub.abs_col[j0], ub.col_vec[j0], mask, 1u, {1, 1, 8, 8});
    if (rem_a > 64u) {
        AscendC::Abs(ub.abs_col[j0 + 64u], ub.col_vec[j0 + 64u], rem_a - 64u);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax(ub.reduced, ub.abs_col[j0], ub.scratch, n - j0, true);
}

template <typename T>
ELEC_FN void load_panel_a(
    uint32_t n, uint64_t batch, uint32_t col_start, uint32_t width, uint32_t width_pad, uint32_t m_stride,
    const AscendC::GlobalTensor<T>& input_a,
    const AscendC::LocalTensor<float>& M,
    const AscendC::LocalTensor<float>& lu_stage) {
    const AscendC::DataCopyPadExtParams<T> no_pad{false, 0, 0, 0};
    if constexpr (sizeof(T) == 4) {
        if ((n & 7u) == 0u && (width & 7u) == 0u) {
            const AscendC::DataCopyParams a_copy{
                static_cast<uint16_t>(n), static_cast<uint16_t>(width >> 3),
                static_cast<uint16_t>((n - width) >> 3), static_cast<uint16_t>((m_stride - width) >> 3)};
            AscendC::DataCopy(M, input_a[batch * n * n + col_start], a_copy);
        } else {
            const AscendC::DataCopyExtParams a_load{
                static_cast<uint16_t>(n), static_cast<uint32_t>(width * sizeof(float)),
                static_cast<uint32_t>((n - width) * sizeof(float)), static_cast<uint32_t>((m_stride - width_pad) >> 3), 0};
            AscendC::DataCopyPad(M, input_a[batch * n * n + col_start], a_load, no_pad);
        }
        fence<AscendC::HardEvent::MTE2_V>();
    } else {
        auto raw_panel = lu_stage.template ReinterpretCast<T>();
        const uint32_t raw_w_pad = (width + 15u) & ~15u;
        for (uint32_t r0 = 0; r0 < n; r0 += 8u) {
            const uint32_t rows = n - r0 < 8u ? n - r0 : 8u;
            const AscendC::DataCopyExtParams a_load{
                static_cast<uint16_t>(rows), static_cast<uint32_t>(width * sizeof(T)),
                static_cast<uint32_t>((n - width) * sizeof(T)), 0, 0};
            AscendC::DataCopyPad(raw_panel, input_a[(batch * n + r0) * n + col_start], a_load, no_pad);
            fence<AscendC::HardEvent::MTE2_V>();
            for (uint32_t r = 0; r < rows; ++r) {
                AscendC::Cast(M[(r0 + r) * m_stride], raw_panel[r * raw_w_pad], AscendC::RoundMode::CAST_NONE, width_pad);
            }
            fence<AscendC::HardEvent::V_MTE2>();
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T>
ELEC_FN void load_rhs_b(
    uint32_t n, uint32_t pad_n, uint32_t k, uint64_t b_offset,
    const AscendC::GlobalTensor<T>& input_b,
    const AscendC::LocalTensor<float>& y,
    const AscendC::LocalTensor<float>& lu_stage,
    const AscendC::LocalTensor<float>& tmp_cast) {
    const AscendC::DataCopyPadExtParams<T> no_pad{false, 0, 0, 0};
    AscendC::Duplicate(y, 0.0f, (pad_n + 8u) * 8u);
    fence<AscendC::HardEvent::V_MTE2>();
    if constexpr (sizeof(T) == 4) {
        const AscendC::DataCopyExtParams b_load{
            static_cast<uint16_t>(n), static_cast<uint32_t>(k * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(y, input_b[b_offset], b_load, no_pad);
        fence<AscendC::HardEvent::MTE2_V>();
    } else {
        auto raw_b = lu_stage.template ReinterpretCast<T>();
        const AscendC::DataCopyExtParams b_load{1, static_cast<uint32_t>(n * k * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(raw_b, input_b[b_offset], b_load, no_pad);
        fence<AscendC::HardEvent::MTE2_V>();
        AscendC::Cast(tmp_cast, raw_b, AscendC::RoundMode::CAST_NONE, (n * k + 7u) & ~7u);
        fence<AscendC::HardEvent::V_S>();
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t t = 0; t < k; ++t) {
                y.SetValue(i * 8u + t, tmp_cast.GetValue(i * k + t));
            }
        }
        fence<AscendC::HardEvent::S_V>();
    }
}

ELEC_FN bool init_back_sub_panel(
    uint32_t col_start, uint32_t width, uint32_t pad_n, int32_t& first_s, int32_t& last_s,
    const AscendC::LocalTensor<float>& lu_buf,
    const AscendC::LocalTensor<float>& buf0) {
    first_s = static_cast<int32_t>(width) - 1;
    last_s = (col_start == 0u) ? 1 : 0;
    if (first_s < last_s) return false;
    const uint32_t first_uj = col_start + static_cast<uint32_t>(first_s);
    const uint32_t first_j_pad = (first_uj + 7u) & ~7u;
    AscendC::Brcb(buf0, lu_buf[static_cast<uint32_t>(first_s) * pad_n],
                  static_cast<uint8_t>(first_j_pad >> 3), {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    return true;
}

ELEC_FN void prefetch_back_sub_brcb(
    int32_t s, int32_t last_s, uint32_t uj, uint32_t pad_n, bool use_buf1,
    const AscendC::LocalTensor<float>& lu_buf,
    const AscendC::LocalTensor<float>& buf0,
    const AscendC::LocalTensor<float>& buf1) {
    if (s <= last_s) return;
    const uint32_t next_j_pad = ((uj - 1u) + 7u) & ~7u;
    const auto& dst = use_buf1 ? buf0 : buf1;
    AscendC::Brcb(dst, lu_buf[static_cast<uint32_t>(s - 1) * pad_n],
                  static_cast<uint8_t>(next_j_pad >> 3), {1, 8});
}

ELEC_FN void back_sub_panel(
    uint32_t col_start, uint32_t width, uint32_t pad_n,
    const AscendC::LocalTensor<float>& lu_buf,
    const AscendC::LocalTensor<float>& y,
    const AscendC::LocalTensor<float>& buf0,
    const AscendC::LocalTensor<float>& buf1,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    int32_t first_s = 0, last_s = 0;
    if (!init_back_sub_panel(col_start, width, pad_n, first_s, last_s, lu_buf, buf0)) return;
    for (int32_t s = first_s; s >= last_s; --s) {
        const uint32_t uj = col_start + static_cast<uint32_t>(s);
        const bool use_buf1 = ((first_s - s) & 1) != 0;
        prefetch_back_sub_brcb(s, last_s, uj, pad_n, use_buf1, lu_buf, buf0, buf1);
        const auto& cur_buf = use_buf1 ? buf1 : buf0;
        if (uj >= 8u) {
            AscendC::MulAddDst(y[0], y[uj * 8u], cur_buf[0],
                               64u, static_cast<uint8_t>(uj >> 3), y_repeat8);
        }
        if ((uj & 7u) != 0u) {
            const uint32_t tail_row = uj & ~7u;
            AscendC::MulAddDst(y[tail_row * 8u], y[uj * 8u], cur_buf[tail_row * 8u],
                               static_cast<uint64_t>((uj & 7u) * 8u), 1u, y_repeat8);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

ELEC_FN void back_sub_panel_p8(
    uint32_t col_start, uint32_t width, uint32_t pad_n,
    const AscendC::LocalTensor<float>& lu_buf,
    const LargeUB& ub,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    int32_t first_s = 0, last_s = 0;
    if (!init_back_sub_panel(col_start, width, pad_n, first_s, last_s, lu_buf, ub.coefficients)) return;
    for (int32_t s = first_s; s >= last_s; --s) {
        const uint32_t uj = col_start + static_cast<uint32_t>(s);
        const bool use_buf1 = ((first_s - s) & 1) != 0;
        prefetch_back_sub_brcb(s, last_s, uj, pad_n, use_buf1, lu_buf, ub.coefficients, ub.lu_stage);
        const uint32_t start8 = (((uj - 1u) & 7u) + 1u) * 8u;
        const uint8_t reps = static_cast<uint8_t>((uj + 7u) >> 3);
        const auto& cur_pad = use_buf1 ? ub.lu_stage_pad : ub.coeff_pad;
        AscendC::MulAddDst(ub.y_pad[start8], ub.y[uj * 8u], cur_pad[start8], 64u, reps, y_repeat8);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

ELEC_FN void fwd_sub_panel(
    uint32_t col_start, uint32_t width, uint32_t n, uint32_t pad_n,
    const uint16_t* piv,
    const AscendC::LocalTensor<float>& lu_buf,
    const AscendC::LocalTensor<float>& y,
    const AscendC::LocalTensor<float>& row_tmp,
    const AscendC::LocalTensor<float>& buf0,
    const AscendC::LocalTensor<float>& buf1,
    const AscendC::BinaryRepeatParams& y_repeat8) {
    if (width == 0u) return;
    const uint32_t first_j0 = col_start & ~7u;
    if (col_start + 1u < n) {
        AscendC::Brcb(buf0[first_j0 * 8u], lu_buf[first_j0],
                      static_cast<uint8_t>((pad_n - first_j0) >> 3), {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }
    for (uint32_t s = 0; s < width; ++s) {
        const uint32_t j = col_start + s;
        const uint32_t pivot = piv[j];
        const bool use_buf1 = (s & 1u) != 0u;
        if (pivot != j) {
            AscendC::Adds(row_tmp[0], y[pivot * 8u], 0.0f, 8u);
            AscendC::Adds(y[pivot * 8u], y[j * 8u], 0.0f, 8u);
        }
        if (s + 1u < width && j + 2u < n) {
            const uint32_t next_j0 = (j + 1u) & ~7u;
            const auto& next_buf = use_buf1 ? buf0 : buf1;
            AscendC::Brcb(next_buf[next_j0 * 8u], lu_buf[(s + 1u) * pad_n + next_j0],
                          static_cast<uint8_t>((pad_n - next_j0) >> 3), {1, 8});
        }
        if (pivot != j) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(y[j * 8u], row_tmp[0], 0.0f, 8u);
        }
        if (j + 1u < n) {
            const uint32_t sub_rows = n - 1u - j;
            auto src_y = (pivot != j) ? row_tmp[0] : y[j * 8u];
            const auto& cur_buf = use_buf1 ? buf1 : buf0;
            AscendC::MulAddDst(y[(j + 1u) * 8u], src_y, cur_buf[(j + 1u) * 8u],
                               64u, static_cast<uint8_t>((sub_rows + 7u) >> 3), y_repeat8);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T>
ELEC_FN void store_square_solution(
    uint64_t batch, uint32_t n, uint32_t k,
    const AscendC::GlobalTensor<T>& output_x,
    const AscendC::LocalTensor<float>& x0,
    const AscendC::LocalTensor<float>& x_out,
    const AscendC::LocalTensor<float>& lu_stage,
    const AscendC::LocalTensor<int32_t>& pack_offsets) {
    if constexpr (sizeof(T) == 4) {
        fence<AscendC::HardEvent::V_MTE3>();
        const AscendC::DataCopyExtParams x_store{
            static_cast<uint16_t>(n), static_cast<uint32_t>(k * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(output_x[batch * n * k], x0, x_store);
        fence<AscendC::HardEvent::MTE3_V>();
    } else {
        fence<AscendC::HardEvent::V_S>();
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t t = 0; t < k; ++t) {
                pack_offsets.SetValue(i * k + t, static_cast<int32_t>((i * 8u + t) * sizeof(float)));
            }
        }
        fence<AscendC::HardEvent::S_V>();
        AscendC::Gather(x_out, x0, pack_offsets.template ReinterpretCast<uint32_t>(), uint32_t(0), n * k);
        AscendC::PipeBarrier<PIPE_V>();
        auto raw_x = lu_stage.template ReinterpretCast<T>();
        AscendC::Cast(raw_x, x_out, AscendC::RoundMode::CAST_RINT, (n * k + 7u) & ~7u);
        fence<AscendC::HardEvent::V_MTE3>();
        const AscendC::DataCopyExtParams x_store{1, static_cast<uint32_t>(n * k * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(output_x[batch * n * k], raw_x, x_store);
        fence<AscendC::HardEvent::MTE3_V>();
    }
}

}  // namespace elec::lstsq_fast
