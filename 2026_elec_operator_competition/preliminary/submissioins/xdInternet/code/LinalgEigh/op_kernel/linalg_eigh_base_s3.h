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
#ifndef LINALG_EIGH_BASE_S3_H
#define LINALG_EIGH_BASE_S3_H
#include "linalg_eigh_base_s2.h"
namespace AscendC {
template <typename DT_A, u32 SOLVE_MODE = 0u, bool FAST200 = false, bool K96_PASS = false>
class KernelEighBaseS3 : public KernelEighBaseS2<DT_A, SOLVE_MODE, FAST200, K96_PASS> {
public:
    using Base = KernelEighBaseS2<DT_A, SOLVE_MODE, FAST200, K96_PASS>;
    using Base::is_active, Base::matrix_scale, Base::gm_A, Base::gm_ws, Base::gm_ckpt, Base::ub_mat, Base::ub_vecs;
    using Base::ub_refl, Base::ub_row, Base::ub_v, Base::ub_w, Base::ub_cur_row, Base::ub_prod, Base::ub_tmp;
    using Base::ub_red, Base::ub_diag, Base::ub_off, Base::ub_comm, Base::ub_gather, Base::ub_mask_ge;
    using Base::ub_mask_gt, Base::ub_mask_eq, Base::ScalarRsqrt, Base::ComputePow2Scale;
    using Base::ConsumeLookaheadReady, Base::VecDotAxpyMaskBrcbSub, Base::VecDotAxpyNegWMaskBrcbSub;
    using Base::LoadAndSymmetrizeFlat, Base::ReduceMaxChunked, Base::n, Base::stride, Base::uplo, Base::group_lanes;
    using Base::lane_id, Base::row_start, Base::row_count, Base::num_zones, Base::zone_offset;
    using Base::ws_scale_offset;
    __aicore__ inline KernelEighBaseS3() = default;
    __aicore__ inline void SymmetrizeCoopRow(u32 r, u32 col_stride, LTu col_idx_buf) {
        u32 i = row_start + r; LTf row_i = ub_mat[r * stride];
        if (uplo != 0u) {
            u32 c_end8 = i & ~7u;
            if (c_end8 > 0u) {
                Gather(row_i, ub_vecs, col_idx_buf, r * 4u, c_end8);
            }
            for (u32 j = c_end8; j < i; ++j) {
                row_i.SetValue(j, ub_vecs.GetValue(j * col_stride + r));
            }
            return;
        }
        u32 n_end8 = n & ~7u, c_next8 = (i + 8u) & ~7u;
        if (c_next8 < n_end8) {
            Gather(row_i[c_next8], ub_vecs[c_next8 * col_stride], col_idx_buf, r * 4u, n_end8 - c_next8);
        }
        u32 j_head_lim = (c_next8 < n) ? c_next8 : n;
        for (u32 j = i + 1u; j < j_head_lim; ++j) {
            row_i.SetValue(j, ub_vecs.GetValue(j * col_stride + r));
        }
        u32 j_tail_start = (n_end8 > j_head_lim) ? n_end8 : j_head_lim;
        for (u32 j = j_tail_start; j < n; ++j) {
            row_i.SetValue(j, ub_vecs.GetValue(j * col_stride + r));
        }
    }
    __aicore__ inline float LoadAndSymmetrizeCoopActive(u64 batch_base) {
        u32 col_stride = (row_count + 7u) & ~7u; u8 col_pad = u8(col_stride - row_count); DataCopyPad(ub_mat,
        gm_A[batch_base + u64(row_start) * n], DCEP{u16(row_count), n * 4u, 0, 0, 0}, DCPEP{true, 0, u8(stride - n),
        0.0f}); DataCopyPad(ub_vecs, gm_A[batch_base + row_start], DCEP{u16(n), row_count * 4u, (n - row_count) * 4u,
        0, 0}, DCPEP{col_pad != 0u, 0, col_pad, 0.0f}); PipeAll(); LTu col_idx_buf =
        ub_diag.template ReinterpretCast<u32>();
        for (u32 j = 0u; j < stride; ++j) {
            col_idx_buf.SetValue(j, j * col_stride * 4u);
        }
        PipeAll();
        for (u32 r = 0u; r < row_count; ++r) {
            SymmetrizeCoopRow(r, col_stride, col_idx_buf);
        }
        PipeAll(); PipeAll(); Abs(ub_vecs, ub_mat, row_count * stride); PipeV(); ReduceMax(ub_red, ub_vecs,
        ub_gather, row_count * stride); PipeAll();
        if (lane_id == 0u) {
            DataCopy(gm_ws[ws_scale_offset + 8u * group_lanes], ub_mat, stride);
        }
        return ub_red.GetValue(0);
    }
    __aicore__ inline void LoadAndSymmetrizeCoop(u64 b) {
        float local_max = 0.0f; u64 batch_base = b * n * n; local_max = (is_active && row_count > 0u) ?
        (LoadAndSymmetrizeCoopActive(batch_base)) : local_max; ub_comm.SetValue(0, local_max); PipeAll();
        DataCopy(gm_ws[ws_scale_offset + lane_id * 8u], ub_comm, 8u); PipeAll(); SyncAll<true>();
        if (is_active) {
            DataCopy(ub_gather, gm_ws[ws_scale_offset], 8u * group_lanes); DataCopy(ub_cur_row,
            gm_ws[ws_scale_offset + 8u * group_lanes], stride); PipeAll(); float max_val = 0.0f;
            for (u32 l = 0u; l < group_lanes; ++l) {
                float s_l = ub_gather.GetValue(l * 8u); max_val = (s_l > max_val) ? (s_l) : max_val;
            }
            float inv_scale = 1.0f; ComputePow2Scale(max_val, matrix_scale, inv_scale);
            if (row_count > 0u) {
                Muls(ub_mat, ub_mat, inv_scale, row_count * stride);
            }
            Muls(ub_cur_row, ub_cur_row, inv_scale, stride); Duplicate(ub_v, 0.0f, 2u * stride); Duplicate(ub_diag,
            0.0f, 3u * stride); PipeAll();
        }
    }
    __aicore__ inline void LoadK96Checkpoint(u64 b, float *work, u8 *tau) {
        const u64 base = b * 65536ull;
        for (u32 off = 0u; off < 25600u; off += 4096u) {
            const u32 count = (25600u - off < 4096u) ? (25600u - off) : 4096u; DataCopy(ub_mat[15360u + off],
            gm_ckpt[base + off], count);
        }
        DataCopy(ub_mat, gm_ckpt[base + 25600u + 24u * 256u], 2048u); DataCopyPad(ub_mat[2048u], gm_ckpt[base +
        25600u + 32u * 256u + 32u], DCEP{32u, 896u, 128u, 0u, 0u}, DCPEP{false, 0u, 0u, 0.0f});
        DataCopyPad(ub_mat[9216u], gm_ckpt[base + 25600u + 64u * 256u + 64u], DCEP{32u, 768u, 256u, 0u, 0u},
        DCPEP{false, 0u, 0u, 0.0f}); DataCopy(ub_gather, gm_ckpt[base + 50176u], 296u);
        this->template fence<HardEvent::MTE2_S>();
        for (u32 k = 0u; k < 96u; ++k) {
            work[k] = ub_gather.GetValue(k); work[256u + k] = ub_gather.GetValue(96u + k); tau[k] =
            u8(i32(ub_gather.GetValue(192u + k)));
        }
        matrix_scale = ub_gather.GetValue(288u);
    }
    __aicore__ inline void FillZoneLowerScalarTail(u32 z, u32 z_cnt, u32 col_stride, u32 n_z, u32 stride_z,
    u32 n_z_end8) {
        PipeAll();
        for (u32 ir = 0u; ir < z_cnt; ++ir) {
            LTf row_z = ub_mat[zone_offset[z] + ir * stride_z]; u32 c_split = ir & ~7u;
            if (c_split >= n_z_end8) {
                for (u32 jc = c_split; jc <= ir && jc < n_z; ++jc) {
                    row_z.SetValue(jc, ub_refl.GetValue(ir * col_stride + jc));
                }
                for (u32 jc = ir + 1u; jc < n_z; ++jc) {
                    row_z.SetValue(jc, ub_refl.GetValue(jc * col_stride + ir));
                }
            } else {
                for (u32 jc = n_z_end8; jc < n_z; ++jc) {
                    row_z.SetValue(jc, ub_refl.GetValue(jc * col_stride + ir));
                }
            }
            for (u32 jc = n_z; jc < stride_z; ++jc) {
                row_z.SetValue(jc, 0.0f);
            }
        }
        PipeAll();
    }
    __aicore__ inline void SymmetrizeZoneLowerDiag(u32 z, u32 z_cnt, u32 col_stride, u32 n_z, u32 stride_z,
    u32 n_z_end8) {
        u32 full_g = (z_cnt & ~7u) >> 3u; u8 z_rep = u8(stride_z / 8u), c_rep = u8(col_stride / 8u);
        for (u32 g = 0u; g < full_g; ++g) {
            u32 c_s = g << 3u;
            if (c_s < n_z_end8) {
                LTf rz_blk = ub_mat[zone_offset[z] + c_s * stride_z + c_s]; Mul(rz_blk, rz_blk, ub_mask_gt, 8u, 8u,
                BRP{1, 1, 1, z_rep, z_rep, 1}); Mul(ub_w[g * 64u], ub_refl[c_s * col_stride + c_s], ub_tmp, 8u, 8u,
                BRP{1, 1, 1, 1, c_rep, 1});
            }
        }
        for (u32 ir = full_g << 3u; ir < z_cnt; ++ir) {
            u32 c_split = ir & ~7u;
            if (c_split < n_z_end8) {
                LTf rz_row = ub_mat[zone_offset[z] + ir * stride_z + c_split]; Mul(rz_row, rz_row,
                ub_mask_gt[(ir & 7u) * 8u], 8u); Mul(ub_w[ir * 8u], ub_refl[ir * col_stride + c_split],
                ub_tmp[(ir & 7u) * 8u], 8u);
            }
        }
        if constexpr (!FAST200) {
            if (z + 1u < num_zones && n_z == stride_z && n_z_end8 == n_z) {
                this->template fence<HardEvent::V_MTE2>();
            }
        }
        PipeV();
        for (u32 g = 0u; g < full_g; ++g) {
            u32 c_s = g << 3u;
            if (c_s < n_z_end8) {
                LTf rz_blk = ub_mat[zone_offset[z] + c_s * stride_z + c_s]; Add(rz_blk, rz_blk, ub_w[g * 64u], 8u,
                8u, BRP{1, 1, 1, z_rep, z_rep, 1});
            }
        }
        for (u32 ir = full_g << 3u; ir < z_cnt; ++ir) {
            u32 c_split = ir & ~7u;
            if (c_split < n_z_end8) {
                LTf rz_row = ub_mat[zone_offset[z] + ir * stride_z + c_split]; Add(rz_row, rz_row, ub_w[ir * 8u], 8u);
            }
        }
    }
    __aicore__ inline void LoadZonesLower(u64 batch_base, LTu col_idx_buf) {
        for (u32 j = 0u; j < (n & ~7u); ++j) {
            col_idx_buf.SetValue(j, j * 32u * 4u);
        }
        Sub(ub_tmp, ub_mask_ge, ub_mask_gt, 8u, 8u, BRP{1, 0, 1, 1, 0, 1}); PipeAll();
        for (u32 z = 0u; z < num_zones; ++z) {
            u32 z_row = z << 5u, z_cnt = (z_row + 32u < n) ? 32u : (n - z_row), col_stride = (z_cnt + 7u) & ~7u;
            u8 col_pad = u8(col_stride - z_cnt); u32 n_z = n - z_row, stride_z = stride - z_row, n_z_end8 =
            n_z & ~7u; DataCopyPad(ub_refl, gm_A[batch_base + u64(z_row) * n + z_row], DCEP{u16(n_z), z_cnt * 4u,
            (n - z_cnt) * 4u, 0, 0}, DCPEP{col_pad != 0u, 0, col_pad, 0.0f});
            if (col_stride != 32u) {
                for (u32 j = 0u; j < n_z_end8; ++j) {
                    col_idx_buf.SetValue(j, j * col_stride * 4u);
                }
                PipeAll();
            } else {
                this->template fence<HardEvent::MTE2_V>();
            }
            for (u32 ir = 0u; ir < z_cnt; ++ir) {
                LTf row_z = ub_mat[zone_offset[z] + ir * stride_z]; u32 c_split = ir & ~7u;
                if (c_split > 0u) {
                    Adds(row_z, ub_refl[ir * col_stride], 0.0f, c_split);
                }
                if (c_split < n_z_end8) {
                    Gather(row_z[c_split], ub_refl[c_split * col_stride], col_idx_buf, ir * 4u, n_z_end8 - c_split);
                }
            }
            PipeV(); SymmetrizeZoneLowerDiag(z, z_cnt, col_stride, n_z, stride_z, n_z_end8);
            if (n_z != stride_z || n_z_end8 != n_z) {
                FillZoneLowerScalarTail(z, z_cnt, col_stride, n_z, stride_z, n_z_end8);
            } else if constexpr (!FAST200) {
                PipeV();
            } else {
                PipeAll();
            }
        }
    }
    __aicore__ inline void SymmetrizeZoneDiagBlocks(u32 z, bool is_add) {
        u32 z_row = z << 5u, z_cnt = (z_row + 32u < n) ? 32u : (n - z_row), stride_z = stride - z_row, full_g =
        (z_cnt & ~7u) >> 3u, z_end8 = z_cnt & ~7u; u8 z_rep = u8(stride_z / 8u);
        for (u32 g = 0u; g < full_g; ++g) {
            u32 c_s = g << 3u; LTf rz_blk = ub_mat[zone_offset[z] + c_s * stride_z + c_s], rf_blk =
            ub_refl[z * 256u + g * 64u];
            if (!is_add) {
                Mul(rf_blk, rf_blk, ub_tmp, 64u); Mul(rz_blk, rz_blk, ub_mask_ge, 8u, 8u, BRP{1, 1, 1, z_rep, z_rep,
                1});
            } else {
                Add(rz_blk, rz_blk, rf_blk, 8u, 8u, BRP{1, 1, 1, z_rep, z_rep, 1});
            }
        }
        if (is_add && z_end8 < z_cnt) {
            PipeAll();
            for (u32 ir = z_end8; ir < z_cnt; ++ir) {
                LTf row_z = ub_mat[zone_offset[z] + ir * stride_z];
                for (u32 jc = z_end8; jc < ir; ++jc) {
                    row_z.SetValue(jc, ub_mat.GetValue(zone_offset[z] + jc * stride_z + ir));
                }
            }
        }
    }
    __aicore__ inline void SymmetrizeZonesUpperDiag() {
        for (u32 z = 0u; z < num_zones; ++z) {
            SymmetrizeZoneDiagBlocks(z, false);
        }
        PipeV();
        for (u32 z = 0u; z < num_zones; ++z) {
            SymmetrizeZoneDiagBlocks(z, true);
        }
        PipeV();
    }
    __aicore__ inline void LoadZonesUpper(u64 batch_base, LTu col_idx_buf) {
        Sub(ub_tmp, ub_mask_ge, ub_mask_ge, 8u, 8u, BRP{1, 0, 1, 1, 0, 1});
        for (u32 z = 0u; z < num_zones; ++z) {
            u32 z_row = z << 5u, z_cnt = (z_row + 32u < n) ? 32u : (n - z_row), n_z = n - z_row, stride_z = stride -
            z_row; DataCopyPad(ub_mat[zone_offset[z]], gm_A[batch_base + u64(z_row) * n + z_row], DCEP{u16(z_cnt),
            n_z * 4u, z_row * 4u, 0, 0}, DCPEP{true, 0, u8(stride_z - n_z), 0.0f});
            for (u32 j = 0u; j < 32u; ++j) {
                col_idx_buf.SetValue(z * 32u + j, j * stride_z * 4u);
            }
        }
        if constexpr (!FAST200) {
            this->template fence<HardEvent::S_V>(); this->template fence<HardEvent::MTE2_V>();
        } else {
            PipeAll();
        }
        for (u32 z = 0u; z < num_zones; ++z) {
            u32 z_row = z << 5u, z_cnt = (z_row + 32u < n) ? 32u : (n - z_row), stride_z = stride - z_row, z_end8 =
            z_cnt & ~7u; LTu c_idx_z = col_idx_buf[z * 32u];
            for (u32 ir = 0u; ir < z_cnt; ++ir) {
                LTf row_z = ub_mat[zone_offset[z] + ir * stride_z]; u32 c_split = ir & ~7u;
                if (c_split > 0u) {
                    Gather(row_z, ub_mat[zone_offset[z]], c_idx_z, ir * 4u, c_split);
                }
                if (c_split < z_end8) {
                    Gather(ub_refl[z * 256u + ir * 8u], ub_mat[zone_offset[z] + c_split * stride_z], c_idx_z,
                    ir * 4u, 8u);
                }
            }
        }
        PipeV(); SymmetrizeZonesUpperDiag();
    }
    __aicore__ inline void LoadAndSymmetrizeLarge(u64 b) {
        u64 batch_base = b * n * n;
        if (n <= 200u) {
            LoadAndSymmetrizeFlat(batch_base); u32 total_m = n * stride; ReduceMaxChunked(ub_w, total_m, 2880u);
            Duplicate(ub_v, 0.0f, stride); Duplicate((n == 200u) ? ub_row : ub_w, 0.0f, stride);
            if (n < stride) {
                Duplicate(ub_prod, 0.0f, stride);
            }
            this->template fence<HardEvent::V_S>(); float max_val = ub_red.GetValue(0), inv_scale = 1.0f;
            ComputePow2Scale(max_val, matrix_scale, inv_scale); Muls(ub_mat, ub_mat, inv_scale, total_m); PipeV();
            return;
        }
        u32 last_z = num_zones - 1u; Duplicate(ub_mat[zone_offset[last_z]], 0.0f, 32u * (stride - (last_z << 5u)));
        LTu col_idx_buf = ub_off.template ReinterpretCast<u32>(); (uplo == 0u) ? (void)LoadZonesLower(batch_base,
        col_idx_buf) : (void)LoadZonesUpper(batch_base, col_idx_buf); u32 zone_total = zone_offset[num_zones];
        ReduceMaxChunked(ub_diag, zone_total, 3840u); Duplicate(ub_v, 0.0f, stride); Duplicate(ub_w, 0.0f, stride);
        if (n < stride) {
            Duplicate(ub_prod, 0.0f, stride);
        }
        this->template fence<HardEvent::V_S>(); float max_val = ub_red.GetValue(0), inv_scale = 1.0f;
        ComputePow2Scale(max_val, matrix_scale, inv_scale);
        for (u32 off = 0u; off < zone_total; off += 8192u) {
            u32 c_len = (zone_total - off < 8192u) ? (zone_total - off) : 8192u; Muls(ub_mat[off], ub_mat[off],
            inv_scale, c_len);
        }
        PipeV();
    }
    __aicore__ inline i32 IssueLookaheadNormRed(LTf row_next, LTf sc_prod, LTf sc_tmp, u32 c_off_next, u32 rem_next,
    u32 len_next) {
        Mul(sc_prod, row_next[c_off_next], row_next[c_off_next], len_next); PipeV();
        if (rem_next != 0u) {
            Mul(sc_prod, sc_prod, ub_mask_ge[rem_next * 8u], 8u); PipeV();
        }
        (len_next <= 64u) ? (void)WholeReduceSum(ub_red, sc_prod, len_next, 1, 1, 1, 8) : (void)ReduceSum(ub_red,
        sc_prod, sc_tmp, len_next); const i32 ev = i32(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(ev); return ev;
    }
    template <u32 N, u32 PHASE = 0u>
    __aicore__ inline void Rank2UpdateFlatMulAdd(LTf mat_t, LTf negative_w, LTf v, LTf brcb_v, LTf brcb_negative_w,
    u32 row_base, u32 c_first, u32 rows) {
        const u32 stride_t = N - row_base; const u8 t_rep = u8(stride_t / 8u); SetMaskNorm(); u32 last_mask = 0u;
        if constexpr (PHASE != 2u) {
            for (u32 c = c_first; c < N; c += 64u) {
                u32 width = (N - c < 64u) ? (N - c) : 64u;
                if (width != last_mask) {
                    SetVectorMask<float, MaskMode::NORMAL>(width); last_mask = width;
                }
                MulAddDst<float, float, false>(mat_t[c - row_base], negative_w[c], brcb_v, MASK_PLACEHOLDER, rows,
                BRP{1, 1, 0, t_rep, 0, 1});
            }
        }
        if constexpr (PHASE == 0u) {
            PipeV();
        }
        if constexpr (PHASE != 1u) {
            for (u32 c = c_first; c < N; c += 64u) {
                u32 width = (N - c < 64u) ? (N - c) : 64u;
                if (width != last_mask) {
                    SetVectorMask<float, MaskMode::NORMAL>(width); last_mask = width;
                }
                MulAddDst<float, float, false>(mat_t[c - row_base], v[c], brcb_negative_w, MASK_PLACEHOLDER, rows,
                BRP{1, 1, 0, t_rep, 0, 1});
            }
        }
        SetMaskCount();
    }
    __aicore__ inline void TridiagMediumSmallMulAdd(u32 k, u32 c_first, u32 r0, u32 rem, u32 len_k, u8 brcb_repeats,
    u8 rep_st, LTf brcb_v, LTf brcb_w, LTf mat_p3, LTf bv_p3, LTf bw_p3, bool &lookahead_ready, i32 &lookahead_ev) {
        u32 act_p3 = n - (k + 1u); VecDotAxpyNegWMaskBrcbSub(ub_w, ub_v, c_first, rem, 0u, brcb_v, r0, brcb_repeats);
        Brcb(brcb_w, ub_w[r0], brcb_repeats, {1, 8}); u32 first_rows = (act_p3 > 8u && k + 2u < n) ? 8u : act_p3;
        MulAddDst(mat_p3[c_first], ub_w[c_first], bv_p3, len_k, first_rows, BRP{1, 1, 0, rep_st, 0, 1}); PipeV();
        MulAddDst(mat_p3[c_first], ub_v[c_first], bw_p3, len_k, first_rows, BRP{1, 1, 0, rep_st, 0, 1}); PipeV();
        if (act_p3 > 8u && k + 2u < n) {
            u32 c_n = (k + 2u) & ~7u; lookahead_ev = IssueLookaheadNormRed(mat_p3, ub_prod[c_n], ub_tmp, c_n, (k +
            2u) & 7u, n - c_n); lookahead_ready = true; u32 rem_p3 = act_p3 - 8u; LTf mat_p3_1 = mat_p3[8u * stride];
            MulAddDst(mat_p3_1[c_first], ub_w[c_first], bv_p3[64], len_k, rem_p3, BRP{1, 1, 0, rep_st, 0, 1});
            PipeV(); MulAddDst(mat_p3_1[c_first], ub_v[c_first], bw_p3[64], len_k, rem_p3, BRP{1, 1, 0, rep_st, 0,
            1}); PipeV();
        }
    }
    __aicore__ inline void TridiagMediumSmallSub(u32 k, u32 c_first, u32 r0, u32 rem, u8 brcb_repeats, u8 rep_st,
    LTf brcb_v, LTf brcb_w, LTf mat_p3, LTf bv_p3, LTf bw_p3, bool &lookahead_ready, i32 &lookahead_ev) {
        u32 act_p3 = n - (k + 1u); VecDotAxpyMaskBrcbSub(ub_w, ub_v, c_first, rem, 0u, brcb_v, r0, brcb_repeats);
        Brcb(brcb_w, ub_w[r0], brcb_repeats, {1, 8});
        if (act_p3 > 8u && k + 2u < n) {
            Mul(ub_vecs, ub_w, bv_p3, stride, 8u, BRP{1, 1, 0, rep_st, 0, 1}); Mul(ub_vecs[8u * stride], ub_v, bw_p3,
            stride, 8u, BRP{1, 1, 0, rep_st, 0, 1}); PipeV(); Sub(mat_p3, mat_p3, ub_vecs, 8u * stride); PipeV();
            Sub(mat_p3, mat_p3, ub_vecs[8u * stride], 8u * stride); PipeV(); u32 c_n = (k + 2u) & ~7u; lookahead_ev =
            IssueLookaheadNormRed(mat_p3, ub_prod[c_n], ub_tmp, c_n, (k + 2u) & 7u, n - c_n); lookahead_ready = true;
            u32 rem_p3 = act_p3 - 8u; LTf mat_p3_1 = mat_p3[8u * stride]; Mul(ub_vecs, ub_w, bv_p3[64], stride,
            rem_p3, BRP{1, 1, 0, rep_st, 0, 1}); PipeV(); Sub(mat_p3_1, mat_p3_1, ub_vecs, rem_p3 * stride);
            Mul(ub_vecs, ub_v, bw_p3[64], stride, rem_p3, BRP{1, 1, 0, rep_st, 0, 1}); PipeV(); Sub(mat_p3_1,
            mat_p3_1, ub_vecs, rem_p3 * stride); PipeV();
        } else {
            Mul(ub_vecs, ub_w, bv_p3, stride, act_p3, BRP{1, 1, 0, rep_st, 0, 1}); PipeV(); Sub(mat_p3, mat_p3,
            ub_vecs, act_p3 * stride); Mul(ub_vecs, ub_v, bw_p3, stride, act_p3, BRP{1, 1, 0, rep_st, 0, 1});
            PipeV(); Sub(mat_p3, mat_p3, ub_vecs, act_p3 * stride); PipeV();
        }
    }
    __aicore__ inline void TridiagMediumSmallUpdate(u32 k, u32 c_first, u32 rem, u32 len_k, u8 rep_st,
    bool &lookahead_ready, i32 &lookahead_ev) {
        u32 r0 = c_first, act_rows = n - r0; Mul(ub_vecs, ub_mat[c_first * stride + c_first], ub_v[c_first], len_k,
        act_rows, BRP{1, 1, 1, rep_st, rep_st, 0}); Muls(ub_prod[c_first], ub_v[c_first], ((stride & 15u) == 0u) ?
        0.5f : -0.5f, len_k);
        if (rem == 0u && c_first >= 8u) {
            Duplicate(ub_refl[k * stride + c_first - 8u], 0.0f, 8u);
        }
        Adds(ub_refl[k * stride + c_first], ub_v[c_first], 0.0f, stride - c_first); PipeV();
        WholeReduceSum(ub_w[c_first], ub_vecs, len_k, act_rows, 1, 1, rep_st); PipeV(); u32 aligned_rows =
        (act_rows + 7u) & ~7u; u8 brcb_repeats = u8(aligned_rows / 8u); LTf brcb_v = ub_gather, brcb_w =
        ub_gather[aligned_rows * 8u], mat_p3 = ub_mat[(k + 1u) * stride], bv_p3 = brcb_v[rem * 8u], bw_p3 =
        brcb_w[rem * 8u]; ((stride & 15u) == 0u) ? (void)TridiagMediumSmallMulAdd(k, c_first, r0, rem, len_k,
        brcb_repeats, rep_st, brcb_v, brcb_w, mat_p3, bv_p3, bw_p3, lookahead_ready, lookahead_ev) :
        (void)TridiagMediumSmallSub(k, c_first, r0, rem, brcb_repeats, rep_st, brcb_v, brcb_w, mat_p3, bv_p3, bw_p3,
        lookahead_ready, lookahead_ev);
    }
    __aicore__ inline void EvalRowTailNormMedium(LTf row_k, u32 c_first, u32 rem, u32 len_k, bool &lookahead_ready,
    i32 &lookahead_ev) {
        if (ConsumeLookaheadReady(row_k, c_first, c_first, rem, lookahead_ready, lookahead_ev)) {
            return;
        }
        if (rem == 0u) {
            Mul(ub_prod[c_first], row_k[c_first], row_k[c_first], len_k); PipeV();
        } else {
            Mul(ub_v[c_first], row_k[c_first], ub_mask_ge[rem * 8u], 8u);
            if (len_k > 8u) {
                Mul(ub_prod[c_first + 8u], row_k[c_first + 8u], row_k[c_first + 8u], len_k - 8u);
            }
            PipeV(); Mul(ub_prod[c_first], ub_v[c_first], ub_v[c_first], (len_k < 8u) ? len_k : 8u); PipeV();
        }
        (len_k <= 64u) ? (void)WholeReduceSum(ub_red, ub_prod[c_first], len_k, 1, 1, 1, 8) : (void)ReduceSum(ub_red,
        ub_prod[c_first], ub_tmp, len_k); this->template fence<HardEvent::V_S>();
    }
    __aicore__ inline void BuildHouseholderMediumV(LTf row_k, u32 k, u32 c_first, u32 rem, u32 len_k, float sq_sum,
    float norm, float *work_off) {
        float x0 = row_k.GetValue(k + 1u) + 0.0f, beta = (x0 >= 0.0f) ? -norm : norm; work_off[k] = beta;
        float inv_s = ScalarRsqrt(sq_sum - beta * x0);
        if (rem == 0u) {
            Muls(ub_v[c_first], row_k[c_first], inv_s, stride - c_first);
        } else {
            Muls(ub_v[c_first], ub_v[c_first], inv_s, (len_k < 8u) ? len_k : 8u);
            if (stride > c_first + 8u) {
                Muls(ub_v[c_first + 8u], row_k[c_first + 8u], inv_s, stride - c_first - 8u);
            }
        }
        Muls(ub_tmp, ub_mask_eq[rem * 8u], -beta * inv_s, 8u); PipeV(); Add(ub_v[c_first], ub_v[c_first], ub_tmp,
        8u); PipeV();
    }
};
} // namespace AscendC
#endif
