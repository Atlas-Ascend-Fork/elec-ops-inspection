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
#ifndef LINALG_EIGH_TRIDIAG_S1_H
#define LINALG_EIGH_TRIDIAG_S1_H
#include "linalg_eigh_base_s5.h"
namespace AscendC {
template <typename DT_A, u32 SOLVE_MODE = 0u, bool FAST200 = false, bool K96_PASS = false>
class KernelEighTridiagS1 : public KernelEighBase<DT_A, SOLVE_MODE, FAST200, K96_PASS> {
public:
    using Base = KernelEighBase<DT_A, SOLVE_MODE, FAST200, K96_PASS>;
    using Base::n, Base::stride, Base::rows_per_lane, Base::zone_offset, Base::ub_mat, Base::ub_refl, Base::ub_row;
    using Base::ub_v, Base::ub_w, Base::ub_prod, Base::ub_tmp, Base::ub_red, Base::ub_gather, Base::ub_mask_ge;
    using Base::ub_mask_eq, Base::ScalarRsqrt, Base::ScalarSqrt, Base::ConsumeLookaheadReady, Base::VecDotSub;
    using Base::ReduceRows64ByBlock2Stage, Base::VecDotAxpyMaskBrcbSub, Base::VecDotAxpyNegWMaskBrcbSub;
    using Base::AccumulateStripSums, Base::SubTwoBrcbStrips, Base::IssueLookaheadNormRed;
    __aicore__ inline KernelEighTridiagS1() = default;
    __aicore__ inline void ExpandTail176AtK79() {
        LTf flat176 = ub_mat[12288u], bounce = ub_mat[43264u];
        for (u32 c = 0u; c < 176u; c += 64u) {
            u32 width = (176u - c < 64u) ? (176u - c) : 64u; Adds(bounce[c], ub_mat[12304u + c], 0.0f, width, 16u,
            UnaryRepeatParams{1, 1, 22, 24});
        }
        PipeV(); Adds(flat176, bounce, 0.0f, 2816u); PipeV();
        for (i32 zi = 7; zi >= 3; --zi) {
            u32 z = u32(zi), z_row = z << 5u, stride_z = 256u - z_row; Adds(bounce, ub_mat[zone_offset[z] - 6144u],
            0.0f, 32u * stride_z); PipeV(); u32 dst_off = (z_row - 80u) * 176u + (z_row - 80u);
            for (u32 c = 0u; c < stride_z; c += 64u) {
                u32 width = (stride_z - c < 64u) ? (stride_z - c) : 64u; Adds(flat176[dst_off + c], bounce[c], 0.0f,
                width, 32u, UnaryRepeatParams{1, 1, 22, u8(stride_z >> 3u)});
            }
            PipeV();
        }
        LTf goff_f = ub_mat[48384u]; LTu goff = goff_f.ReinterpretCast<u32>();
        for (u32 c = 0u; c < 144u; ++c) {
            goff.SetValue(c, c * 704u);
        }
        PipeAll();
        for (u32 r = 16u; r < 176u; ++r) {
            u32 sym_cols = (((r + 80u) >> 5u) << 5u) - 80u; Gather(flat176[r * 176u], flat176, goff, r * 4u,
            sym_cols);
        }
        PipeAll();
    }
    __aicore__ inline void RepackTail160AndR80AtK95() {
        LTf flat176 = ub_mat[12288u], flat160 = ub_mat[15360u], bounce = ub_mat[43264u];
        for (u32 r0 = 0u; r0 < 160u; r0 += 32u) {
            LTf src = flat176[(r0 + 16u) * 176u + 16u]; Adds(bounce, src, 0.0f, 64u, 32u, UnaryRepeatParams{1, 1, 20,
            22}); Adds(bounce[64u], src[64u], 0.0f, 64u, 32u, UnaryRepeatParams{1, 1, 20, 22}); Adds(bounce[128u],
            src[128u], 0.0f, 32u, 32u, UnaryRepeatParams{1, 1, 20, 22}); PipeV(); Adds(flat160[r0 * 160u], bounce,
            0.0f, 5120u); PipeV();
        }
        Adds(bounce, flat176, 0.0f, 2816u); PipeV();
        for (u32 r = 0u; r < 16u; ++r) {
            Duplicate(ub_mat[12288u + r * 192u], 0.0f, 16u);
        }
        for (u32 c = 0u; c < 176u; c += 64u) {
            u32 width = (176u - c < 64u) ? (176u - c) : 64u; Adds(ub_mat[12304u + c], bounce[c], 0.0f, width, 16u,
            UnaryRepeatParams{1, 1, 24, 22});
        }
        PipeV();
    }
    __aicore__ inline u32 GetZone200Index(u32 idx) const {
        if (idx < 8u) {
            return 0u;
        }
        if (idx < 104u) {
            return 1u + ((idx - 8u) >> 4u);
        }
        return (idx < 112u) ? 7u : ((idx < 136u) ? 8u : ((idx < 168u) ? 9u : 10u));
    }
    __aicore__ inline void CompactZone200(u32 next_z, u32 c_first, u32 stride_zk, float norm, bool &v_norm_done) {
        const u32 z200_off[11] = {0u, 1600u, 4672u, 7488u, 10048u, 12352u, 14400u, 16192u, 16960u, 19072u, 21120u};
        u32 new_stride = 200u - c_first, col_shift = stride_zk - new_stride, z_base = z200_off[next_z]; u8 dst_rep =
        u8(new_stride / 8u), src_rep = u8(stride_zk / 8u);
        if (new_stride <= 128u) {
            LTf bounce = ub_mat[(new_stride == 128u) ? 30784u : 34000u], src_b = ub_mat[z_base + col_shift];
            for (u32 c = 0u; c < new_stride; c += 64u) {
                u32 w = (new_stride - c < 64u) ? (new_stride - c) : 64u; Adds(bounce[c], src_b[c], 0.0f, w,
                u8(new_stride), UnaryRepeatParams{1, 1, dst_rep, src_rep});
            }
            PipeV();
            if (norm != 0.0f) {
                Add(ub_v[c_first], ub_v[c_first], ub_red[16], 8u); v_norm_done = true;
            }
            Adds(ub_mat[z_base], bounce, 0.0f, new_stride * new_stride); PipeV(); return;
        }
        LTf bounce = ub_mat[z_base + new_stride * stride_zk]; u32 b_step = (new_stride == 192u) ? 32u :
        ((new_stride == 176u) ? 44u : ((new_stride == 160u) ? 56u : 72u));
        for (u32 r0 = 0u; r0 < new_stride; r0 += b_step) {
            u32 b_rows = (new_stride - r0 < b_step) ? (new_stride - r0) : b_step; LTf src_b = ub_mat[z_base +
            r0 * stride_zk + col_shift];
            for (u32 c = 0u; c < new_stride; c += 64u) {
                u32 w = (new_stride - c < 64u) ? (new_stride - c) : 64u; Adds(bounce[c], src_b[c], 0.0f, w,
                u8(b_rows), UnaryRepeatParams{1, 1, dst_rep, src_rep});
            }
            PipeV();
            if (r0 == 0u && norm != 0.0f) {
                Add(ub_v[c_first], ub_v[c_first], ub_red[16], 8u); v_norm_done = true;
            }
            Adds(ub_mat[z_base + r0 * new_stride], bounce, 0.0f, b_rows * new_stride); PipeV();
        }
    }
    __aicore__ inline void Tridiag200MatVecP1(u32 z_act, u32 row_base, u32 stride_t, u32 c_first, LTf ub_tail,
    LTf tail_sc, LTf w_200) {
        u8 t_rep = u8(stride_t / 8u); u32 act_rows_p1 = 200u - c_first; u8 sc_rep_p1 = (z_act == 10u) ? 4u : 8u;
        u32 p1_tile = (z_act == 0u) ? 104u : ((z_act == 1u) ? 128u : 176u);
        for (u32 r_off = 0u; r_off < act_rows_p1; r_off += p1_tile) {
            u32 cur_rows = (act_rows_p1 - r_off < p1_tile) ? (act_rows_p1 - r_off) : p1_tile; LTf mat_t =
            ub_tail[(c_first + r_off - row_base) * stride_t + (c_first - row_base)]; u32 width0 =
            (act_rows_p1 < 64u) ? act_rows_p1 : 64u; Mul(tail_sc, mat_t, ub_v[c_first], width0, cur_rows, BRP{1, 1,
            1, sc_rep_p1, t_rep, 0});
            for (u32 c = c_first + 64u; c < 200u; c += 64u) {
                u32 width = (200u - c < 64u) ? (200u - c) : 64u; PipeV(); MulAddDst(tail_sc, mat_t[c - c_first],
                ub_v[c], width, cur_rows, BRP{1, 1, 1, sc_rep_p1, t_rep, 0});
            }
            PipeV(); (width0 == 64u) ? (void)ReduceRows64ByBlock2Stage(w_200[c_first + r_off], tail_sc, cur_rows) :
            (void)WholeReduceSum(w_200[c_first + r_off], tail_sc, width0, cur_rows, 1, 1, sc_rep_p1);
            if (r_off + p1_tile >= act_rows_p1) {
                Muls(ub_prod[c_first], ub_v[c_first], 0.5f, act_rows_p1);
            }
            PipeV();
        }
    }
    __aicore__ inline void Tridiag200UpdateTail(u32 k, LTf row_k, u32 c_first, u32 c_off_k, u32 rem, LTf w_200,
    bool &lookahead_ready, i32 &lookahead_ev) {
        const u32 z200_off[11] = {0u, 1600u, 4672u, 7488u, 10048u, 12352u, 14400u, 16192u, 16960u, 19072u, 21120u};
        const u32 z200_row[11] = {0u, 8u, 24u, 40u, 56u, 72u, 88u, 104u, 112u, 136u, 168u}; u32 z_act =
        GetZone200Index(c_first), row_base = z200_row[z_act], stride_t = 200u - row_base; LTf ub_tail =
        ub_mat[z200_off[z_act]], tail_sc = ub_tail[stride_t * stride_t]; Tridiag200MatVecP1(z_act, row_base,
        stride_t, c_first, ub_tail, tail_sc, w_200); LTf brcb_v = ub_mat[43800], brcb_w = ub_mat[45400];
        u8 init_b_rep = u8((200u - c_first + 7u) >> 3u); VecDotAxpyNegWMaskBrcbSub(w_200, ub_v, c_first, rem,
        row_base, brcb_v, c_first, init_b_rep); Brcb(brcb_w, w_200[c_first], init_b_rep, {1, 8}); u32 total_p3_rows =
        200u - (k + 1u), max_r0 = (z_act <= 9u) ? 1u : 32u, first_rows = (total_p3_rows < max_r0) ? total_p3_rows :
        max_r0; LTf mat_t0 = ub_tail[(k + 1u - row_base) * stride_t];
        if (FAST200 && z_act <= 9u) {
            this->template Rank2UpdateFlatMulAdd<200u, 1u>(mat_t0, w_200, ub_v, brcb_v[rem * 8u], brcb_w[rem * 8u],
            row_base, c_first, total_p3_rows); PipeV(); this->template Rank2UpdateFlatMulAdd<200u, 2u>(mat_t0, w_200,
            ub_v, brcb_v[rem * 8u], brcb_w[rem * 8u], row_base, c_first, first_rows);
        } else {
            this->template Rank2UpdateFlatMulAdd<200u>(mat_t0, w_200, ub_v, brcb_v[rem * 8u], brcb_w[rem * 8u],
            row_base, c_first, first_rows);
        }
        if (first_rows >= total_p3_rows) {
            if (rem == 0u && c_off_k >= 8u) {
                Duplicate(row_k[c_off_k - 8u], 0.0f, 8u);
            }
            Adds(row_k[c_off_k], ub_v[c_first], 0.0f, 200u - c_first);
        }
        PipeV();
        if (first_rows < total_p3_rows) {
            if (k + 3u < 200u) {
                u32 c_n = (k + 2u) & ~7u; lookahead_ev = IssueLookaheadNormRed(mat_t0, tail_sc, tail_sc[256], c_n -
                row_base, (k + 2u) & 7u, 200u - c_n); lookahead_ready = true;
            }
            u32 r1 = k + 1u + first_rows; LTf mat_t1 = ub_tail[(r1 - row_base) * stride_t];
            this->template Rank2UpdateFlatMulAdd<200u, FAST200 ? 2u : 0u>(mat_t1, w_200, ub_v, brcb_v[(r1 -
            c_first) * 8u], brcb_w[(r1 - c_first) * 8u], row_base, c_first, 200u - r1);
            if (rem == 0u && c_off_k >= 8u) {
                Duplicate(row_k[c_off_k - 8u], 0.0f, 8u);
            }
            Adds(row_k[c_off_k], ub_v[c_first], 0.0f, 200u - c_first); PipeV();
        }
    }
    __aicore__ inline void EvalRowTailNorm200(LTf row_k, u32 c_first, u32 c_off_k, u32 rem, u32 len_k,
    bool &lookahead_ready, i32 lookahead_ev) {
        if (ConsumeLookaheadReady(row_k, c_first, c_off_k, rem, lookahead_ready, lookahead_ev)) {
            return;
        }
        if (rem == 0u) {
            Adds(ub_v[c_first], row_k[c_off_k], 0.0f, len_k); Mul(ub_prod[c_first], row_k[c_off_k], row_k[c_off_k],
            len_k); PipeV();
        } else {
            Mul(ub_v[c_first], row_k[c_off_k], ub_mask_ge[rem * 8u], 8u);
            if (200u > c_first + 8u) {
                Adds(ub_v[c_first + 8u], row_k[c_off_k + 8u], 0.0f, 200u - c_first - 8u); Mul(ub_prod[c_first + 8u],
                row_k[c_off_k + 8u], row_k[c_off_k + 8u], len_k - 8u);
            }
            PipeV(); Mul(ub_prod[c_first], ub_v[c_first], ub_v[c_first], 8u); PipeV();
        }
        (len_k <= 64u) ? (void)WholeReduceSum(ub_red, ub_prod[c_first], len_k, 1, 1, 1, 8) : (void)ReduceSum(ub_red,
        ub_prod[c_first], ub_tmp, len_k); this->template fence<HardEvent::V_S>();
    }
    __aicore__ inline void BuildHouseholder200V(LTf row_k, u32 k, u32 z_k_row, u32 c_first, u32 c_off_k, u32 rem,
    float sq_sum, float norm, float *work_off) {
        float x0 = row_k.GetValue(k + 1u - z_k_row) + 0.0f, beta = (x0 >= 0.0f) ? -norm : norm; work_off[k] = beta;
        float inv_s = ScalarRsqrt(sq_sum - beta * x0);
        if (rem == 0u) {
            Muls(ub_v[c_first], row_k[c_off_k], inv_s, 200u - c_first);
        } else {
            Muls(ub_v[c_first], ub_v[c_first], inv_s, 8u);
            if (200u > c_first + 8u) {
                Muls(ub_v[c_first + 8u], row_k[c_off_k + 8u], inv_s, 200u - c_first - 8u);
            }
        }
        Muls(ub_red[16], ub_mask_eq[rem * 8u], -beta * inv_s, 8u);
    }
    __aicore__ inline void TridiagonalizeLarge200(float *work, u8 *tau_mask) {
        float *work_diag = work; float *work_off = work + 256; const u32 z200_off[11] = {0u, 1600u, 4672u, 7488u,
        10048u, 12352u, 14400u, 16192u, 16960u, 19072u, 21120u}; const u32 z200_row[11] = {0u, 8u, 24u, 40u, 56u,
        72u, 88u, 104u, 112u, 136u, 168u}; LTf w_200 = ub_row; Duplicate(w_200, 0.0f, 200u); PipeV();
        bool lookahead_ready = false; i32 lookahead_ev = 0;
        for (u32 k = 0u; k < 200u; ++k) {
            u32 z_k = GetZone200Index(k), z_k_row = z200_row[z_k], stride_zk = 200u - z_k_row; LTf row_k =
            ub_mat[z200_off[z_k] + (k - z_k_row) * stride_zk];
            if (k + 2u >= 200u) {
                if (lookahead_ready) {
                    WaitFlag<HardEvent::V_S>(lookahead_ev); lookahead_ready = false;
                }
                this->template fence<HardEvent::V_S>(); work_diag[k] = row_k.GetValue(k - z_k_row); work_off[k] =
                (k + 1u < 200u) ? row_k.GetValue(k + 1u - z_k_row) : 0.0f; tau_mask[k] = 0u; continue;
            }
            u32 c_first = (k + 1u) & ~7u, c_off_k = c_first - z_k_row, rem = (k + 1u) & 7u; EvalRowTailNorm200(row_k,
            c_first, c_first - z_k_row, rem, 200u - c_first, lookahead_ready, lookahead_ev); float sq_sum =
            ub_red.GetValue(0), norm = ScalarSqrt(sq_sum); work[k] = row_k.GetValue(k - z_k_row); work[256u + k] =
            0.0f; tau_mask[k] = (norm != 0.0f) ? 1u : 0u; bool v_norm_done = false;
            if (norm != 0.0f) {
                BuildHouseholder200V(row_k, k, z_k_row, c_first, c_first - z_k_row, rem, sq_sum, norm, work + 256);
            }
            if ((k < 104u && ((k & 15u) == 7u)) || k == 111u || k == 135u || k == 167u) {
                CompactZone200(z_k + 1u, c_first, stride_zk, norm, v_norm_done);
            }
            if (norm != 0.0f) {
                if (!v_norm_done) {
                    PipeV(); Add(ub_v[c_first], ub_v[c_first], ub_red[16], 8u); PipeV();
                }
                Tridiag200UpdateTail(k, row_k, c_first, c_first - z_k_row, rem, w_200, lookahead_ready, lookahead_ev);
            }
        }
    }
    __aicore__ inline void TridiagSub200MatVecTiles(u32 r0, u32 c_first, u32 act_rows, u32 tile_rows, u8 rep_stride,
    u8 sc_rep_stride) {
        for (u32 r_off = 0u; r_off < act_rows; r_off += tile_rows) {
            u32 cur_rows = (act_rows - r_off < tile_rows) ? (act_rows - r_off) : tile_rows; LTf mat_tile =
            ub_mat[(r0 + r_off) * stride];
            for (u32 c = c_first; c < n; c += 64u) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; Mul(ub_refl[c - c_first], mat_tile[c], ub_v[c], width,
                cur_rows, BRP{1, 1, 1, sc_rep_stride, rep_stride, 0});
            }
            PipeV(); u32 s_idx = 0u;
            for (u32 c = c_first; c < n; c += 64u, ++s_idx) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; LTf dst_s = (s_idx == 0u) ? ub_w[r0 + r_off] :
                ub_gather[(s_idx - 1u) * stride + r_off]; WholeReduceSum(dst_s, ub_refl[c - c_first], width,
                cur_rows, 1, 1, sc_rep_stride);
            }
            PipeV();
        }
        u32 n_strips = (n - c_first + 63u) >> 6u; AccumulateStripSums(ub_w[r0], ub_gather, n_strips, stride,
        act_rows);
    }
    __aicore__ inline void TridiagSub200UpdateStep(LTf row_k, u32 c_first, u32 rem, u32 scratch_cap, u8 rep_stride) {
        u32 r0 = c_first, act_rows = n - r0, raw_sc = ((n - c_first) + 7u) & ~7u, sc_stride = (raw_sc >= 64u) ?
        raw_sc : 64u; u8 sc_rep_stride = u8(sc_stride / 8u); u32 tile_rows = (scratch_cap / sc_stride) & ~7u;
        tile_rows = (tile_rows > 64u) ? (64u) : tile_rows; TridiagSub200MatVecTiles(r0, c_first, act_rows, tile_rows,
        rep_stride, sc_rep_stride); LTf brcb_v_base = ub_gather, brcb_w_base = ub_gather[512]; u32 init_b =
        (act_rows < 64u) ? act_rows : 64u; u8 init_rep = u8(((init_b + 7u) & ~7u) / 8u); u32 brcb_start = 0u,
        brcb_end = init_b; VecDotAxpyMaskBrcbSub(ub_w, ub_v, c_first, rem, stride, brcb_v_base, r0, init_rep);
        for (u32 r_off = 0u; r_off < act_rows; r_off += tile_rows) {
            u32 cur_rows = (act_rows - r_off < tile_rows) ? (act_rows - r_off) : tile_rows;
            if (r_off == 0u) {
                Brcb(brcb_w_base, ub_w[r0], init_rep, {1, 8});
            } else if (r_off + cur_rows > brcb_end) {
                brcb_start = r_off; u32 b_rows = (act_rows - r_off < 64u) ? (act_rows - r_off) : 64u; u8 b_rep =
                u8(((b_rows + 7u) & ~7u) / 8u); Brcb(brcb_v_base, ub_v[r0 + r_off], b_rep, {1, 8}); Brcb(brcb_w_base,
                ub_w[r0 + r_off], b_rep, {1, 8}); PipeV(); brcb_end = r_off + b_rows;
            }
            LTf brcb_v = brcb_v_base[(r_off - brcb_start) * 8u], brcb_w = brcb_w_base[(r_off - brcb_start) * 8u],
            mat_tile = ub_mat[(r0 + r_off) * stride]; SubTwoBrcbStrips(mat_tile, ub_w, ub_v, ub_refl, ub_refl,
            brcb_w, brcb_v, cur_rows, c_first, rep_stride, sc_rep_stride);
        }
        if (rem == 0u && c_first >= 8u) {
            Duplicate(row_k[c_first - 8u], 0.0f, 8u);
        }
        Adds(row_k[c_first], ub_v[c_first], 0.0f, stride - c_first); PipeV();
    }
    __aicore__ inline void TridiagonalizeLargeSub200(float *work, u8 *tau_mask) {
        float *work_diag = work; float *work_off = work + 256; u32 refl_rows = (49152u - 1280u) / stride - n -
        rows_per_lane - 8u, scratch_cap = (refl_rows + rows_per_lane) * stride; u8 rep_stride = u8(stride / 8u);
        for (u32 k = 0u; k < n; ++k) {
            LTf row_k = ub_mat[k * stride];
            if (k + 2u >= n) {
                PipeAll(); work_diag[k] = row_k.GetValue(k); work_off[k] = (k + 1u < n) ? row_k.GetValue(k + 1u) :
                0.0f; tau_mask[k] = 0u; continue;
            }
            u32 c_first = (k + 1u) & ~7u, rem = (k + 1u) & 7u; Mul(ub_v[c_first], row_k[c_first],
            ub_mask_ge[rem * 8u], 8u);
            if (stride > c_first + 8u) {
                Adds(ub_v[c_first + 8u], row_k[c_first + 8u], 0.0f, stride - c_first - 8u);
            }
            float sq_sum = VecDotSub(ub_v, ub_v, c_first), norm = ScalarSqrt(sq_sum); work_diag[k] =
            row_k.GetValue(k); work_off[k] = 0.0f; u8 tau_k = 0u;
            if (norm != 0.0f) {
                float x0 = ub_v.GetValue(k + 1u), beta = (x0 >= 0.0f) ? -norm : norm; work_off[k] = beta; tau_k = 1u;
                float inv_s = ScalarRsqrt(sq_sum - beta * x0); Muls(ub_v[c_first], ub_v[c_first], inv_s, n -
                c_first); Muls(ub_red[16], ub_mask_eq[rem * 8u], -beta * inv_s, 8u); PipeV(); Add(ub_v[c_first],
                ub_v[c_first], ub_red[16], 8u); PipeV(); Muls(ub_prod[c_first], ub_v[c_first], -0.5f, n - c_first);
                TridiagSub200UpdateStep(row_k, c_first, rem, scratch_cap, rep_stride);
            }
            tau_mask[k] = tau_k;
        }
    }
    __aicore__ inline void CompactZone256Chunked(u32 bounce_off, u32 dst_base, u32 total_rows, u32 chunk_rows,
    u32 new_stride, u32 old_stride, u32 c_first, float norm, bool &v_norm_done) {
        LTf bounce = ub_mat[bounce_off]; u8 dst_rep = u8(new_stride / 8u), src_rep = u8(old_stride / 8u);
        for (u32 b_row = 0u; b_row < total_rows; b_row += chunk_rows) {
            u32 cur_r = (total_rows - b_row < chunk_rows) ? (total_rows - b_row) : chunk_rows; LTf src_b =
            ub_mat[dst_base + b_row * old_stride + 16u]; Adds(bounce, src_b, 0.0f, 64u, u8(cur_r),
            UnaryRepeatParams{1, 1, dst_rep, src_rep}); Adds(bounce[64], src_b[64], 0.0f, (new_stride > 128u) ? 64u :
            (new_stride - 64u), u8(cur_r), UnaryRepeatParams{1, 1, dst_rep, src_rep});
            if (new_stride > 128u) {
                Adds(bounce[128], src_b[128], 0.0f, new_stride - 128u, u8(cur_r), UnaryRepeatParams{1, 1, dst_rep,
                src_rep});
            }
            PipeV();
            if (b_row == 0u && norm != 0.0f) {
                Add(ub_v[c_first], ub_v[c_first], ub_red[16], 8u); v_norm_done = true;
            }
            Adds(ub_mat[dst_base + b_row * new_stride], bounce, 0.0f, cur_r * new_stride); PipeV();
        }
    }
    __aicore__ inline void CompactZone256AtK79Or95(u32 k, LTf row_k, u32 z_k_row, u32 c_first, u32 c_off_k,
    float sq_sum, float norm, float *work_off, u8 &tau_k, bool &v_norm_done) {
        u32 cur_len = (k == 79u) ? 176u : 160u;
        if (norm != 0.0f) {
            float x0 = row_k.GetValue(k + 1u - z_k_row) + 0.0f, beta = (x0 >= 0.0f) ? -norm : norm; work_off[k] =
            beta; tau_k = 1u; float inv_s = ScalarRsqrt(sq_sum - beta * x0); Muls(ub_v[c_first], row_k[c_off_k],
            inv_s, cur_len); Muls(ub_red[16], ub_mask_eq[0], -beta * inv_s, 8u); PipeV(); Add(ub_v[c_first],
            ub_v[c_first], ub_red[16], 8u); PipeV(); v_norm_done = true; Duplicate(row_k[c_off_k - 8u], 0.0f, 8u);
            Adds(row_k[c_off_k], ub_v[c_first], 0.0f, cur_len); PipeV();
        }
        (k == 79u) ? (void)ExpandTail176AtK79() : (void)RepackTail160AndR80AtK95();
    }
    __aicore__ inline void CompactZone256Tail(u32 k, LTf row_k, u32 z_k_row, u32 c_first, u32 c_off_k, float sq_sum,
    float norm, float *work_off, u8 &tau_k, bool &v_norm_done) {
        if (k == 79u || k == 95u) {
            CompactZone256AtK79Or95(k, row_k, z_k_row, c_first, c_off_k, sq_sum, norm, work_off, tau_k, v_norm_done);
        } else if (k == 111u) {
            CompactZone256Chunked(40960u, 17920u, 144u, 48u, 144u, 160u, c_first, norm, v_norm_done);
        } else if (k == 127u) {
            CompactZone256Chunked(38656u, 20224u, 128u, 64u, 128u, 144u, c_first, norm, v_norm_done);
        } else if (k == 143u) {
            CompactZone256Chunked(36608u, 22272u, 112u, 56u, 112u, 128u, c_first, norm, v_norm_done);
        } else if (k == 159u) {
            LTf bounce = ub_mat[34816], src_b = ub_mat[24064u + 16u]; Adds(bounce, src_b, 0.0f, 64u, 96u,
            UnaryRepeatParams{1, 1, 12, 14}); Adds(bounce[64], src_b[64], 0.0f, 32u, 96u, UnaryRepeatParams{1, 1, 12,
            14}); PipeV();
            if (norm != 0.0f) {
                Add(ub_v[c_first], ub_v[c_first], ub_red[16], 8u); v_norm_done = true;
            }
            Adds(ub_mat[24064], bounce, 0.0f, 4608u); Adds(ub_mat[24064u + 4608u], bounce[4608], 0.0f, 4608u);
            PipeV();
        } else if (k == 191u || k == 223u) {
            u32 src_off = (k == 191u) ? (24064u + 32u * 96u + 32u) : (27136u + 32u * 64u + 32u), dst_off =
            (k == 191u) ? 27136u : 29184u, dim = (k == 191u) ? 64u : 32u; Adds(ub_mat[34816], ub_mat[src_off], 0.0f,
            dim, u8(dim), UnaryRepeatParams{1, 1, u8(dim / 8u), u8((dim + 32u) / 8u)}); PipeV();
            if (norm != 0.0f) {
                Add(ub_v[c_first], ub_v[c_first], ub_red[16], 8u); v_norm_done = true;
            }
            Adds(ub_mat[dst_off], ub_mat[34816], 0.0f, dim * dim); PipeV();
        }
    }
    __aicore__ inline void Tridiag256TailLt144P1(u32 c_first, u32 row_base, u32 stride_t, LTf ub_tail, LTf tail_sc,
    LTf w_tail) {
        u8 t_rep = u8(stride_t / 8u); u32 cur_cols = 256u - c_first, p1_tile = (c_first < 96u) ? 80u :
        ((c_first < 112u) ? 96u : ((c_first < 128u) ? 144u : 128u));
        for (u32 r0 = c_first; r0 < 256u; r0 += p1_tile) {
            u32 act_r = (256u - r0 < p1_tile) ? (256u - r0) : p1_tile; LTf mat_t = ub_tail[(r0 -
            row_base) * stride_t + (c_first - row_base)]; Mul(tail_sc, mat_t, ub_v[c_first], 64u, act_r, BRP{1, 1, 1,
            8, t_rep, 0}); PipeV(); MulAddDst(tail_sc, mat_t[64], ub_v[c_first + 64u], (cur_cols > 128u) ? 64u :
            (cur_cols - 64u), act_r, BRP{1, 1, 1, 8, t_rep, 0});
            if (cur_cols > 128u) {
                PipeV(); MulAddDst(tail_sc, mat_t[128], ub_v[c_first + 128u], cur_cols - 128u, act_r, BRP{1, 1, 1, 8,
                t_rep, 0});
            }
            PipeV(); ReduceRows64ByBlock2Stage(w_tail[r0], tail_sc, act_r);
            if (r0 + p1_tile >= 256u) {
                Muls(ub_prod[c_first], ub_v[c_first], 0.5f, cur_cols);
            }
            PipeV();
        }
    }
    __aicore__ inline void Tridiag256TailLt144(u32 k, LTf row_k, u32 c_first, u32 c_off_k, u32 rem, u32 row_base,
    u32 stride_t, LTf ub_tail, LTf tail_sc, LTf w_tail, bool &lookahead_ready, i32 &lookahead_ev) {
        Tridiag256TailLt144P1(c_first, row_base, stride_t, ub_tail, tail_sc, w_tail); u32 cur_cols = 256u - c_first;
        LTf brcb_v = (c_first < 96u) ? tail_sc[512u] : ub_mat[44400], brcb_w = (c_first < 96u) ? tail_sc[1920u] :
        ub_mat[45680]; u8 b_rep = u8((cur_cols + 7u) >> 3u); VecDotAxpyNegWMaskBrcbSub(w_tail, ub_v, c_first, rem,
        row_base, brcb_v, c_first, b_rep); Brcb(brcb_w, w_tail[c_first], b_rep, {1, 8}); u32 total_p3_rows = 256u -
        (k + 1u), first_rows = (total_p3_rows < 1u) ? total_p3_rows : 1u; LTf mat_t0 = ub_tail[(k + 1u -
        row_base) * stride_t]; this->template Rank2UpdateFlatMulAdd<256u, 1u>(mat_t0, w_tail, ub_v, brcb_v[rem * 8u],
        brcb_w[rem * 8u], row_base, c_first, total_p3_rows); PipeV(); this->template Rank2UpdateFlatMulAdd<256u,
        2u>(mat_t0, w_tail, ub_v, brcb_v[rem * 8u], brcb_w[rem * 8u], row_base, c_first, first_rows); PipeV();
        if (first_rows < total_p3_rows) {
            if (k + 3u < 256u) {
                u32 c_n = (k + 2u) & ~7u; lookahead_ev = IssueLookaheadNormRed(mat_t0, tail_sc, tail_sc[256], c_n -
                row_base, (k + 2u) & 7u, 256u - c_n); lookahead_ready = true;
            }
            u32 r1 = k + 1u + first_rows; LTf mat_t1 = ub_tail[(r1 - row_base) * stride_t];
            this->template Rank2UpdateFlatMulAdd<256u, 2u>(mat_t1, w_tail, ub_v, brcb_v[(r1 - c_first) * 8u],
            brcb_w[(r1 - c_first) * 8u], row_base, c_first, 256u - r1);
            if (k != 79u && k != 95u) {
                if (rem == 0u && c_off_k >= 8u) {
                    Duplicate(row_k[c_off_k - 8u], 0.0f, 8u);
                }
                Adds(row_k[c_off_k], ub_v[c_first], 0.0f, cur_cols);
            }
            PipeV();
        }
    }
};
} // namespace AscendC
#endif
