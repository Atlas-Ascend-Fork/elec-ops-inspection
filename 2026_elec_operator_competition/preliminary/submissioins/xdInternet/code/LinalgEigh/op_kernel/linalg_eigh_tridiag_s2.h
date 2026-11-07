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
#ifndef LINALG_EIGH_TRIDIAG_S2_H
#define LINALG_EIGH_TRIDIAG_S2_H
#include "linalg_eigh_tridiag_s1.h"
namespace AscendC {
template <typename DT_A, u32 SOLVE_MODE = 0u, bool FAST200 = false, bool K96_PASS = false>
class KernelEighTridiagS2 : public KernelEighTridiagS1<DT_A, SOLVE_MODE, FAST200, K96_PASS> {
public:
    using Base = KernelEighTridiagS1<DT_A, SOLVE_MODE, FAST200, K96_PASS>;
    using Base::ScalarRsqrt, Base::ReduceRows64ByBlock2Stage, Base::VecDotAxpyNegWMaskBrcbSub;
    using Base::IssueLookaheadNormRed, Base::CompactZone256Tail, Base::Tridiag256TailLt144, Base::n, Base::stride;
    using Base::num_zones, Base::zone_offset, Base::gm_eigvecs, Base::ub_mat, Base::ub_v, Base::ub_w, Base::ub_prod;
    using Base::ub_red, Base::ub_gather, Base::ub_mask_eq;
    __aicore__ inline KernelEighTridiagS2() = default;
    __aicore__ inline void Tridiag256TailGe144P1(u32 c_first, u32 row_base, u32 stride_t, LTf ub_tail, LTf tail_sc,
    LTf w_tail) {
        u8 t_rep = u8(stride_t / 8u); u32 cur_cols = 256u - c_first; u8 sc_rep_p1 = (row_base == 224u) ? 4u : 8u;
        u32 width0_p1 = (cur_cols < 64u) ? cur_cols : 64u; LTf mat_p1 = ub_tail[(c_first - row_base) * (stride_t +
        1u)]; Mul(tail_sc, mat_p1, ub_v[c_first], width0_p1, cur_cols, BRP{1, 1, 1, sc_rep_p1, t_rep, 0});
        if (cur_cols > 64u) {
            PipeV(); MulAddDst(tail_sc, mat_p1[64], ub_v[c_first + 64u], cur_cols - 64u, cur_cols, BRP{1, 1, 1,
            sc_rep_p1, t_rep, 0});
        }
        PipeV(); (width0_p1 == 64u) ? (void)ReduceRows64ByBlock2Stage(w_tail[c_first], tail_sc, cur_cols) :
        (void)WholeReduceSum(w_tail[c_first], tail_sc, width0_p1, cur_cols, 1, 1, sc_rep_p1); Muls(ub_prod[c_first],
        ub_v[c_first], 0.5f, cur_cols); PipeV();
    }
    __aicore__ inline void Tridiag256TailGe144(u32 k, LTf row_k, u32 c_first, u32 c_off_k, u32 rem, u32 row_base,
    u32 stride_t, LTf ub_tail, LTf tail_sc, LTf w_tail, bool &lookahead_ready, i32 &lookahead_ev) {
        Tridiag256TailGe144P1(c_first, row_base, stride_t, ub_tail, tail_sc, w_tail); u32 cur_cols = 256u - c_first;
        LTf brcb_v = ub_mat[44400], brcb_w = ub_mat[45680]; u8 b_rep = u8(cur_cols / 8u);
        VecDotAxpyNegWMaskBrcbSub(w_tail, ub_v, c_first, rem, row_base, brcb_v, c_first, b_rep); u32 r0 = k + 1u,
        total_p3_rows = 256u - r0, max_r0 = (c_first < 224u) ? 1u : 32u, first_rows = (total_p3_rows < max_r0) ?
        total_p3_rows : max_r0; Brcb(brcb_w, w_tail[c_first], b_rep, {1, 8}); LTf mat_t = ub_tail[(r0 -
        row_base) * stride_t];
        if (first_rows < total_p3_rows) {
            this->template Rank2UpdateFlatMulAdd<256u, 1u>(mat_t, w_tail, ub_v, brcb_v[rem * 8u], brcb_w[rem * 8u],
            row_base, c_first, total_p3_rows); PipeV(); this->template Rank2UpdateFlatMulAdd<256u, 2u>(mat_t, w_tail,
            ub_v, brcb_v[rem * 8u], brcb_w[rem * 8u], row_base, c_first, first_rows);
        } else {
            this->template Rank2UpdateFlatMulAdd<256u>(mat_t, w_tail, ub_v, brcb_v[rem * 8u], brcb_w[rem * 8u],
            row_base, c_first, first_rows);
        }
        if (first_rows >= total_p3_rows) {
            if (rem == 0u && c_off_k >= 8u) {
                Duplicate(row_k[c_off_k - 8u], 0.0f, 8u);
            }
            Adds(row_k[c_off_k], ub_v[c_first], 0.0f, cur_cols);
        }
        PipeV();
        if (first_rows < total_p3_rows) {
            if (k + 3u < 256u) {
                u32 c_n = (k + 2u) & ~7u; lookahead_ev = IssueLookaheadNormRed(mat_t, tail_sc, tail_sc[256], c_n -
                row_base, (k + 2u) & 7u, 256u - c_n); lookahead_ready = true;
            }
            u32 r1 = r0 + first_rows; LTf mat_t1 = ub_tail[(r1 - row_base) * stride_t];
            this->template Rank2UpdateFlatMulAdd<256u, 2u>(mat_t1, w_tail, ub_v, brcb_v[(r1 - c_first) * 8u],
            brcb_w[(r1 - c_first) * 8u], row_base, c_first, 256u - r1);
            if (rem == 0u && c_off_k >= 8u) {
                Duplicate(row_k[c_off_k - 8u], 0.0f, 8u);
            }
            Adds(row_k[c_off_k], ub_v[c_first], 0.0f, cur_cols); PipeV();
        }
    }
    __aicore__ inline void GetTail256ZoneLayout(u32 c_first, u32 &row_base, u32 &tail_off) const {
        const u32 bases[8] = {80u, 96u, 112u, 128u, 144u, 160u, 192u, 224u}; const u32 offs[8] = {12288u, 15360u,
        17920u, 20224u, 22272u, 24064u, 27136u, 29184u}; u32 idx = (c_first < 160u) ? ((c_first - 80u) >> 4u) : (5u +
        ((c_first - 160u) >> 5u)); row_base = bases[idx]; tail_off = offs[idx];
    }
    __aicore__ inline void Tridiag256TailStep(u32 k, LTf row_k, u32 z_k_row, u32 c_first, u32 c_off_k, u32 rem,
    float sq_sum, float norm, float *work_off, u8 *tau_mask, bool &lookahead_ready, i32 &lookahead_ev) {
        u8 tau_k = 0u; bool v_norm_done = false;
        if (norm != 0.0f && k != 79u && k != 95u) {
            float x0 = row_k.GetValue(k + 1u - z_k_row) + 0.0f, beta = (x0 >= 0.0f) ? -norm : norm; work_off[k] =
            beta; tau_k = 1u; float inv_s = ScalarRsqrt(sq_sum - beta * x0);
            if (rem == 0u) {
                Muls(ub_v[c_first], row_k[c_off_k], inv_s, 256u - c_first);
            } else {
                Muls(ub_v[c_first], ub_v[c_first], inv_s, 8u);
                if (256u > c_first + 8u) {
                    Muls(ub_v[c_first + 8u], row_k[c_off_k + 8u], inv_s, 256u - c_first - 8u);
                }
            }
            Muls(ub_red[16], ub_mask_eq[rem * 8u], -beta * inv_s, 8u);
        }
        CompactZone256Tail(k, row_k, z_k_row, c_first, c_off_k, sq_sum, norm, work_off, tau_k, v_norm_done);
        if (norm != 0.0f) {
            if (!v_norm_done) {
                PipeV(); Add(ub_v[c_first], ub_v[c_first], ub_red[16], 8u); PipeV();
            }
            u32 row_base = 0u, tail_off = 0u; GetTail256ZoneLayout(c_first, row_base, tail_off); u32 stride_t =
            256u - row_base; LTf ub_tail = ub_mat[tail_off], tail_sc = ub_tail[stride_t * stride_t], w_tail =
            ub_gather[768]; (c_first < 144u) ? (void)Tridiag256TailLt144(k, row_k, c_first, c_first - z_k_row, rem,
            row_base, stride_t, ub_tail, tail_sc, w_tail, lookahead_ready, lookahead_ev) :
            (void)Tridiag256TailGe144(k, row_k, c_first, c_first - z_k_row, rem, row_base, stride_t, ub_tail,
            tail_sc, w_tail, lookahead_ready, lookahead_ev);
        }
        tau_mask[k] = tau_k;
    }
    __aicore__ inline void SpillAndShift256(u32 k, u64 priv_gm_256, u32 &z_shift, LTf &ub_tridiag_sc, LTf &ub_vacc,
    u32 &p1_scratch_cap, u32 &p3_scratch_cap) {
        if (k == 15u) {
            PipeAll(); DataCopyPad(gm_eigvecs[priv_gm_256], ub_mat, DCEP{256u, 64u, 0, 960u, 0}); PipeAll();
            for (u32 s_off = 0u; s_off < 32768u; s_off += 4096u) {
                Adds(ub_mat[s_off], ub_mat[s_off + 4096u], 0.0f, 4096u); PipeV();
            }
            Duplicate(ub_mat[32768], 0.0f, 256u); PipeV(); z_shift = 4096u; ub_w = ub_mat[32768]; ub_tridiag_sc =
            ub_mat[33024]; ub_vacc = ub_tridiag_sc; p1_scratch_cap = 15360u; p3_scratch_cap = 12032u;
        } else if (k == 23u) {
            PipeAll(); DataCopyPad(gm_eigvecs[priv_gm_256 + 16u], ub_mat, DCEP{256u, 32u, 0, 992u, 0}); PipeAll();
            for (u32 s_off = 0u; s_off < 30720u; s_off += 2048u) {
                Adds(ub_mat[s_off], ub_mat[s_off + 2048u], 0.0f, 2048u); PipeV();
            }
            Duplicate(ub_mat[30720], 0.0f, 256u); PipeV(); z_shift = 6144u; ub_w = ub_mat[30720]; ub_tridiag_sc =
            ub_mat[30976]; ub_vacc = ub_tridiag_sc; p1_scratch_cap = 17408u; p3_scratch_cap = 14080u;
        }
    }
    __aicore__ inline void Tridiag256ZoneP1Strip(u32 zb0, u32 zb1, u32 b_r0, u32 s_idx, u32 c_first, u32 z_shift,
    LTf ub_tridiag_sc, u32 &last_p1_mask) {
        u32 c_off_s = s_idx << 6u;
        for (u32 z = zb0; z < zb1; ++z) {
            u32 z_row = z << 5u, r_start = (c_first > z_row) ? c_first : z_row, c = r_start + c_off_s;
            if (c >= 256u) {
                continue;
            }
            u32 act_rows = (z_row + 32u) - r_start, width = (256u - c < 64u) ? (256u - c) : 64u, stride_z = 256u -
            z_row; u8 rep_stride = u8(stride_z >> 3u), sc_rep = (z == 7u) ? 4u : 8u; LTf mat_z =
            ub_mat[zone_offset[z] + (r_start - z_row) * stride_z - z_shift], sc_z = ub_tridiag_sc[(r_start -
            b_r0) * 64u];
            if (width != last_p1_mask) {
                SetVectorMask<float, MaskMode::NORMAL>(width); last_p1_mask = width;
            }
            if (s_idx == 0u) {
                Mul<float, false>(sc_z, mat_z[c - z_row], ub_v[c], MASK_PLACEHOLDER, act_rows, BRP{1, 1, 1, sc_rep,
                rep_stride, 0});
            } else {
                MulAddDst<float, float, false>(sc_z, mat_z[c - z_row], ub_v[c], MASK_PLACEHOLDER, act_rows, BRP{1, 1,
                1, sc_rep, rep_stride, 0});
            }
        }
        PipeV();
    }
    __aicore__ inline void Tridiag256ZoneP1(u32 k, u32 c_first, u32 z_start, u32 z_shift, LTf ub_tridiag_sc,
    bool has_p2, u32 aligned_p2_rows, LTf brcb_v_p2, bool &p2_brcb_issued) {
        u32 zb0 = z_start;
        while (zb0 < 8u) {
            u32 zb1 = (k < 15u && zb0 == 0u) ? 5u : 8u, b_r0 = (c_first > (zb0 << 5u)) ? c_first : (zb0 << 5u),
            max_strips = (256u - b_r0 + 63u) >> 6u, last_p1_mask = 0u; SetMaskNorm();
            for (u32 s_idx = 0u; s_idx < max_strips; ++s_idx) {
                Tridiag256ZoneP1Strip(zb0, zb1, b_r0, s_idx, c_first, z_shift, ub_tridiag_sc, last_p1_mask);
            }
            SetVectorMask<float, MaskMode::NORMAL>(64u); SetMaskCount(); u32 rows64 = ((zb1 <= 7u) ? (zb1 << 5u) :
            224u) - b_r0;
            if (rows64 > 0u) {
                ReduceRows64ByBlock2Stage(ub_w[b_r0], ub_tridiag_sc, rows64);
            }
            if (zb1 > 7u) {
                WholeReduceSum(ub_w[224], ub_tridiag_sc[rows64 * 64u], 32u, 32u, 1, 1, 4);
            }
            if (zb1 > 7u && has_p2) {
                Brcb(brcb_v_p2, ub_v[c_first], u8(aligned_p2_rows / 8u), {1, 8}); p2_brcb_issued = true;
            }
            PipeV(); zb0 = zb1;
        }
    }
    __aicore__ inline u32 GetSub256ZoneScStride(u32 z_row, u32 r_start) const {
        u32 raw_sc = ((n - r_start) + 7u) & ~7u;
        if (n == stride && r_start == z_row) {
            return stride - z_row;
        }
        return (raw_sc >= 64u) ? raw_sc : 64u;
    }
    __aicore__ inline void TridiagSub256ZoneP1Batch(u32 zb0, u32 zb1, u32 c_first, u32 z_shift, LTf ub_tridiag_sc) {
        u32 sc_off = 0u;
        for (u32 z = zb0; z < zb1; ++z) {
            u32 z_row = z << 5u, stride_z = stride - z_row, r_start = (c_first > z_row) ? c_first : z_row, r_end =
            (z_row + 32u < n) ? (z_row + 32u) : n, act_rows = r_end - r_start, sc_stride =
            GetSub256ZoneScStride(z_row, r_start); u8 rep_stride = u8(stride_z / 8u), sc_rep_stride =
            u8(sc_stride / 8u); LTf mat_z = ub_mat[zone_offset[z] + (r_start - z_row) * stride_z - z_shift], sc_z =
            ub_tridiag_sc[sc_off];
            for (u32 c = r_start; c < n; c += 64u) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; Mul(sc_z[c - r_start], mat_z[c - z_row], ub_v[c], width,
                act_rows, BRP{1, 1, 1, sc_rep_stride, rep_stride, 0});
            }
            sc_off += act_rows * sc_stride;
        }
        PipeV(); sc_off = 0u;
        for (u32 z = zb0; z < zb1; ++z) {
            u32 z_row = z << 5u, r_start = (c_first > z_row) ? c_first : z_row, r_end = (z_row + 32u < n) ? (z_row +
            32u) : n, act_rows = r_end - r_start, sc_stride = GetSub256ZoneScStride(z_row, r_start);
            u8 sc_rep_stride = u8(sc_stride / 8u); LTf sc_z = ub_tridiag_sc[sc_off]; u32 strip_idx = 0u;
            for (u32 c = r_start; c < n; c += 64u, ++strip_idx) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; LTf dst_s = (strip_idx == 0u) ? ub_w[r_start] :
                ub_gather[(strip_idx - 1u) * stride + r_start]; WholeReduceSum(dst_s, sc_z[c - r_start], width,
                act_rows, 1, 1, sc_rep_stride);
            }
            sc_off += act_rows * sc_stride;
        }
        PipeV();
    }
    __aicore__ inline void ReduceSub256ZoneP1Strips(u32 c_first) {
        u32 s1_end = ((n - 64u + 31u) & ~31u) < n ? ((n - 64u + 31u) & ~31u) : n;
        if (s1_end <= c_first) {
            return;
        }
        u32 s1_len = s1_end - c_first; LTf s1_buf = ub_gather, s2_buf = ub_gather[stride], s3_buf =
        ub_gather[2u * stride];
        if (n > c_first + 192u) {
            u32 s2_end = ((n - 128u + 31u) & ~31u) < n ? ((n - 128u + 31u) & ~31u) : n, s3_end = ((n - 192u +
            31u) & ~31u) < n ? ((n - 192u + 31u) & ~31u) : n; Add(ub_w[c_first], ub_w[c_first], s1_buf[c_first],
            s1_len); Add(s2_buf[c_first], s2_buf[c_first], s3_buf[c_first], s3_end - c_first); PipeV();
            Add(ub_w[c_first], ub_w[c_first], s2_buf[c_first], s2_end - c_first);
        } else if (n > c_first + 128u) {
            u32 s2_end = ((n - 128u + 31u) & ~31u) < n ? ((n - 128u + 31u) & ~31u) : n; Add(ub_w[c_first],
            ub_w[c_first], s1_buf[c_first], s1_len); PipeV(); Add(ub_w[c_first], ub_w[c_first], s2_buf[c_first],
            s2_end - c_first);
        } else {
            Add(ub_w[c_first], ub_w[c_first], s1_buf[c_first], s1_len);
        }
    }
    __aicore__ inline void TridiagSub256ZoneP1(u32 c_first, u32 z_start, u32 z_shift, u32 p1_scratch_cap,
    LTf ub_tridiag_sc, LTf ub_vacc, bool has_p2, u32 aligned_p2_rows, u32 first_act_rows, u32 vacc_rows,
    u32 vacc_stride, LTf brcb_v_p2, bool &p2_brcb_issued) {
        for (u32 zb0 = z_start; zb0 < num_zones; ) {
            u32 zb1 = zb0, used_sc = 0u;
            while (zb1 < num_zones) {
                u32 z_row = zb1 << 5u, r_start = (c_first > z_row) ? c_first : z_row, r_end = (z_row + 32u < n) ?
                (z_row + 32u) : n, act_rows = r_end - r_start, sc_stride = GetSub256ZoneScStride(z_row, r_start);
                if (zb1 > zb0 && used_sc + act_rows * sc_stride > p1_scratch_cap) {
                    break;
                }
                used_sc += act_rows * sc_stride; ++zb1;
            }
            TridiagSub256ZoneP1Batch(zb0, zb1, c_first, z_shift, ub_tridiag_sc); zb0 = zb1;
        }
        if (n <= c_first + 64u) {
            return;
        }
        ReduceSub256ZoneP1Strips(c_first);
        if (has_p2) {
            Brcb(brcb_v_p2, ub_v[c_first], u8(aligned_p2_rows / 8u), {1, 8});
            if (first_act_rows + 1u < vacc_rows) {
                Duplicate(ub_vacc[(first_act_rows + 1u) * vacc_stride], 0.0f, (vacc_rows - first_act_rows -
                1u) * vacc_stride);
            }
            p2_brcb_issued = true;
        }
        PipeV();
    }
    __aicore__ inline void MulOrAddZoneP2Strip(LTf dst, LTf m_src, LTf b_src, u32 z_next, u32 col_base, u32 z_row,
    u32 act_rows, u8 dst_rep, u8 src_rep, bool first_pair) {
        for (u32 c = z_next; c < 256u; c += 64u) {
            u32 w = (256u - c < 64u) ? (256u - c) : 64u; (first_pair) ? (void)Mul(dst[c - col_base], m_src[c -
            z_row], b_src, w, act_rows, BRP{1, 1, 0, dst_rep, src_rep, 1}) : (void)MulAddDst(dst[c - col_base],
            m_src[c - z_row], b_src, w, act_rows, BRP{1, 1, 0, dst_rep, src_rep, 1});
        }
    }
    __aicore__ inline void Tridiag256ZoneP2Lt15(u32 k, u32 c_first, u32 aligned_p2_rows, u32 first_act_rows,
    LTf ub_tridiag_sc, LTf brcb_v_p2, bool p2_brcb_issued) {
        u8 v0_rep = 28u; LTf ub_vacc0 = ub_tridiag_sc, ub_vacc1 = ub_tridiag_sc[7168];
        if (!p2_brcb_issued) {
            Brcb(brcb_v_p2, ub_v[c_first], u8(aligned_p2_rows / 8u), {1, 8}); PipeV();
        }
        if (first_act_rows + 1u < 32u) {
            Duplicate(ub_vacc0[(first_act_rows + 1u) * 224u], 0.0f, (32u - first_act_rows - 1u) * 224u);
        }
        LTf mat_z0 = ub_mat[(k + 1u) * 256u], brcb_z0 = brcb_v_p2[((k + 1u) - c_first) * 8u];
        for (u32 c = 32u; c < 256u; c += 64u) {
            u32 width = (256u - c < 64u) ? (256u - c) : 64u; Mul(ub_vacc0[c - 32u], mat_z0[c], brcb_z0, width,
            first_act_rows, BRP{1, 1, 0, v0_rep, 32, 1});
        }
        if (first_act_rows < 32u) {
            Adds(ub_vacc0[first_act_rows * 224u], ub_w[32], 0.0f, 224u);
        }
        Mul(ub_vacc1, ub_mat[zone_offset[6] + 32u], brcb_v_p2[(192u - c_first) * 8u], 32u, 32u, BRP{1, 1, 0, 4, 8,
        1}); PipeV();
        for (u32 z = 1u; z < 6u; ++z) {
            u32 z_row = z << 5u, z_next = z_row + 32u, stride_z = 256u - z_row; u8 rep_z = u8(stride_z >> 3u);
            LTf mat_z = ub_mat[zone_offset[z]], brcb_z = brcb_v_p2[(z_row - c_first) * 8u];
            for (u32 c = z_next; c < 256u; c += 64u) {
                u32 width = (256u - c < 64u) ? (256u - c) : 64u; MulAddDst(ub_vacc0[c - 32u], mat_z[c - z_row],
                brcb_z, width, 32u, BRP{1, 1, 0, v0_rep, rep_z, 1});
            }
            u32 f_len = (16u >> (z - 1u)) * 32u; Add(ub_vacc1, ub_vacc1, ub_vacc1[f_len], f_len); PipeV();
        }
        SetMaskNorm(); SetVectorMask<float, MaskMode::NORMAL>(64u);
        for (u32 half = 16u; half >= 2u; half >>= 1u) {
            Add<float, false>(ub_vacc0, ub_vacc0, ub_vacc0[half * 224u], MASK_PLACEHOLDER, u8((half * 224u) >> 6u),
            BRP{1, 1, 1, 8, 8, 8}); PipeV();
        }
        SetMaskCount(); Add(ub_w[32], ub_vacc0, ub_vacc0[224u], 192u); Add(ub_vacc0[192], ub_vacc0[192],
        ub_vacc0[224u + 192u], 32u); Muls(ub_prod[c_first], ub_v[c_first], 0.5f, 256u - c_first); PipeV();
        Add(ub_w[224], ub_vacc0[192], ub_vacc1, 32u); PipeV();
    }
    __aicore__ inline void MaskedZoneP2RowOp(LTf dst, LTf mat_z, LTf brcb_z, u32 c_start, u32 dst_col_base,
    u32 src_col_base, u32 act_rows, u8 dst_rep, u8 src_rep, bool is_mul_add) {
        SetMaskNorm(); u32 last_mask = 0u;
        for (u32 c = c_start; c < 256u; c += 64u) {
            u32 width = (256u - c < 64u) ? (256u - c) : 64u;
            if (width != last_mask) {
                SetVectorMask<float, MaskMode::NORMAL>(width); last_mask = width;
            }
            if (is_mul_add) {
                MulAddDst<float, float, false>(dst[c - dst_col_base], mat_z[c - src_col_base], brcb_z,
                MASK_PLACEHOLDER, act_rows, BRP{1, 1, 0, dst_rep, src_rep, 1});
            } else {
                Mul<float, false>(dst[c - dst_col_base], mat_z[c - src_col_base], brcb_z, MASK_PLACEHOLDER, act_rows,
                BRP{1, 1, 0, dst_rep, src_rep, 1});
            }
        }
        SetVectorMask<float, MaskMode::NORMAL>(64u); SetMaskCount();
    }
    __aicore__ inline void FoldDualVacc256Ge15(u32 c_first, u32 first_z_next, u32 first_act_rows, u32 v0_stride,
    u32 v1_stride, bool v1_folded16, LTf ub_vacc0, LTf ub_vacc1) {
        SetMaskNorm(); SetVectorMask<float, MaskMode::NORMAL>(64u);
        for (u32 half = 16u; half >= 2u; half >>= 1u) {
            Add<float, false>(ub_vacc0, ub_vacc0, ub_vacc0[half * v0_stride], MASK_PLACEHOLDER,
            u8((half * v0_stride) >> 6u), BRP{1, 1, 1, 8, 8, 8});
            if (!v1_folded16) {
                Add<float, false>(ub_vacc1, ub_vacc1, ub_vacc1[half * v1_stride], MASK_PLACEHOLDER,
                u8((half * v1_stride) >> 6u), BRP{1, 1, 1, 8, 8, 8});
            } else if (half >= 4u) {
                u32 h1 = half >> 1u; Add<float, false>(ub_vacc1, ub_vacc1, ub_vacc1[h1 * v1_stride],
                MASK_PLACEHOLDER, u8((h1 * v1_stride) >> 6u), BRP{1, 1, 1, 8, 8, 8});
            } else {
                SetMaskCount(); Add(ub_vacc1, ub_vacc1, ub_vacc1[v1_stride], v1_stride); SetMaskNorm();
                SetVectorMask<float, MaskMode::NORMAL>(64u);
            }
            PipeV();
        }
        SetMaskCount();
        if (first_act_rows < 32u) {
            Add(ub_w[first_z_next], ub_vacc0, ub_vacc0[v0_stride], 32u); Add(ub_vacc0[32], ub_vacc0[32],
            ub_vacc0[v0_stride + 32u], v1_stride);
            if (!v1_folded16) {
                Add(ub_vacc1, ub_vacc1, ub_vacc1[v1_stride], v1_stride);
            }
            Muls(ub_prod[c_first], ub_v[c_first], 0.5f, 256u - c_first); PipeV(); Add(ub_w[first_z_next + 32u],
            ub_vacc0[32], ub_vacc1, v1_stride); PipeV();
        } else {
            Add(ub_vacc0, ub_vacc0, ub_vacc0[v0_stride], v0_stride);
            if (!v1_folded16) {
                Add(ub_vacc1, ub_vacc1, ub_vacc1[v1_stride], v1_stride);
            }
            PipeV(); Add(ub_w[first_z_next], ub_w[first_z_next], ub_vacc0, 32u); Add(ub_vacc0[32], ub_vacc0[32],
            ub_vacc1, v1_stride); Muls(ub_prod[c_first], ub_v[c_first], 0.5f, 256u - c_first); PipeV();
            Add(ub_w[first_z_next + 32u], ub_w[first_z_next + 32u], ub_vacc0[32], v1_stride); PipeV();
        }
    }
    __aicore__ inline void Tridiag256ZoneP2Ge15(u32 k, u32 c_first, u32 p23_z_start, u32 first_z_next, u32 z_shift,
    u32 aligned_p2_rows, u32 first_act_rows, u32 vacc_stride, u8 vacc_rep_stride, LTf ub_tridiag_sc, LTf ub_vacc,
    LTf brcb_v_p2, bool p2_brcb_issued) {
        (void)ub_vacc; u32 v0_stride = vacc_stride, v1_stride = vacc_stride - 32u; u8 v0_rep = vacc_rep_stride,
        v1_rep = u8(v1_stride >> 3u); LTf ub_vacc0 = ub_tridiag_sc, ub_vacc1 = ub_tridiag_sc[32u * v0_stride];
        if (!p2_brcb_issued) {
            Brcb(brcb_v_p2, ub_v[c_first], u8(aligned_p2_rows / 8u), {1, 8}); PipeV();
        }
        if (first_act_rows + 1u < 32u) {
            Duplicate(ub_vacc0[(first_act_rows + 1u) * v0_stride], 0.0f, (32u - first_act_rows - 1u) * v0_stride);
        }
        u32 z0_row = p23_z_start << 5u, stride_z0 = 256u - z0_row; LTf mat_z0 = ub_mat[zone_offset[p23_z_start] +
        ((k + 1u) - z0_row) * stride_z0 - z_shift]; MaskedZoneP2RowOp(ub_vacc0, mat_z0, brcb_v_p2[((k + 1u) -
        c_first) * 8u], first_z_next, first_z_next, z0_row, first_act_rows, v0_rep, u8(stride_z0 >> 3u), false);
        if (first_act_rows < 32u) {
            Adds(ub_vacc0[first_act_rows * v0_stride], ub_w[first_z_next], 0.0f, v0_stride);
        }
        u32 z1 = p23_z_start + 1u, z1_row = z1 << 5u, z1_next = z1_row + 32u, stride_z1 = 256u - z1_row;
        MaskedZoneP2RowOp(ub_vacc1, ub_mat[zone_offset[z1] - z_shift], brcb_v_p2[(z1_row - c_first) * 8u], z1_next,
        z1_next, z1_row, 32u, v1_rep, u8(stride_z1 >> 3u), false); PipeV(); bool v1_folded16 = false;
        for (u32 zp = p23_z_start + 2u; zp < 7u; zp += 2u) {
            u32 ze_row = zp << 5u, stride_ze = 256u - ze_row; MaskedZoneP2RowOp(ub_vacc0, ub_mat[zone_offset[zp] -
            z_shift], brcb_v_p2[(ze_row - c_first) * 8u], ze_row + 32u, first_z_next, ze_row, 32u, v0_rep,
            u8(stride_ze >> 3u), true);
            if (zp + 1u < 7u) {
                u32 zo = zp + 1u, zo_row = zo << 5u, stride_zo = 256u - zo_row; MaskedZoneP2RowOp(ub_vacc1,
                ub_mat[zone_offset[zo] - z_shift], brcb_v_p2[(zo_row - c_first) * 8u], zo_row + 32u, first_z_next +
                32u, zo_row, 32u, v1_rep, u8(stride_zo >> 3u), true);
            } else {
                Add(ub_vacc1, ub_vacc1, ub_vacc1[16u * v1_stride], 16u * v1_stride); v1_folded16 = true;
            }
            PipeV();
        }
        FoldDualVacc256Ge15(c_first, first_z_next, first_act_rows, v0_stride, v1_stride, v1_folded16, ub_vacc0,
        ub_vacc1);
    }
    __aicore__ inline u32 GetSub256ZoneP2ScV(u32 z, u32 p23_z_start, u32 z_next, u32 vacc_stride) const {
        if (z == p23_z_start) {
            return vacc_stride;
        }
        u32 raw_vsc = ((n - z_next) + 7u) & ~7u; return (n == stride) ? raw_vsc : ((raw_vsc >= 64u) ? raw_vsc : 64u);
    }
    __aicore__ inline void FoldSub256ZoneP2Gt16Row(LTf sc_z, u32 z_next, u32 sc_v, u32 fold_cnt) {
        if (sc_v == n - z_next) {
            Add(sc_z, sc_z, sc_z[16u * sc_v], fold_cnt * sc_v); return;
        }
        u8 sc_rep_stride = u8(sc_v / 8u);
        for (u32 c = z_next; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u, sc = c - z_next; Add(sc_z[sc], sc_z[sc], sc_z[16u * sc_v +
            sc], width, fold_cnt, BRP{1, 1, 1, sc_rep_stride, sc_rep_stride, sc_rep_stride});
        }
    }
};
} // namespace AscendC
#endif
