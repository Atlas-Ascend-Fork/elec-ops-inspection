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
#ifndef LINALG_EIGH_BASE_S4_H
#define LINALG_EIGH_BASE_S4_H
#include "linalg_eigh_base_s3.h"
namespace AscendC {
template <typename DT_A, u32 SOLVE_MODE = 0u, bool FAST200 = false, bool K96_PASS = false>
class KernelEighBaseS4 : public KernelEighBaseS3<DT_A, SOLVE_MODE, FAST200, K96_PASS> {
public:
    using Base = KernelEighBaseS3<DT_A, SOLVE_MODE, FAST200, K96_PASS>;
    using Base::gm_ws, Base::gm_ckpt, Base::ub_mat, Base::ub_vecs, Base::ub_refl, Base::ub_row, Base::ub_v;
    using Base::ub_w, Base::ub_cur_row, Base::ub_prod, Base::ub_tmp, Base::ub_red, Base::ub_diag, Base::ub_off;
    using Base::ub_tau, Base::ub_comm, Base::ub_gather, Base::ub_mask_ge, Base::ub_mask_eq, Base::ScalarSqrt;
    using Base::VecDotSub, Base::VecDotAxpyMaskBrcbSub, Base::VecDotAxpySub, Base::MatVecRowsSub;
    using Base::BacktransformPairUnitRowsSub, Base::BacktransformStepRowsSub, Base::BacktransformPairRowsSub;
    using Base::Rank2UpdateRowsSub, Base::GetStaircaseReflector, Base::GetReflector200;
    using Base::TridiagMediumSmallUpdate, Base::EvalRowTailNormMedium, Base::BuildHouseholderMediumV, Base::n;
    using Base::stride, Base::group_lanes, Base::rows_per_lane, Base::lane_id, Base::vec_row_start, Base::row_start;
    using Base::row_count, Base::vecs_offset, Base::zone_offset, Base::ws_refl_offset, Base::ws_w_offset;
    using Base::ws_tail_offset, Base::is_active, Base::gm_eigvecs;
    __aicore__ inline KernelEighBaseS4() = default;
    __aicore__ inline void TridiagonalizeMedium(float *work, u8 *tau_mask) {
        if (n > 120u) {
            return;
        }
        float *work_diag = work; float *work_off = work + 256; u8 rep_st = u8(stride / 8u); bool lookahead_ready =
        false; i32 lookahead_ev = 0;
        for (u32 k = 0u; k < n; ++k) {
            LTf row_k = ub_mat[k * stride];
            if (k + 2u >= n) {
                if (lookahead_ready) {
                    WaitFlag<HardEvent::V_S>(lookahead_ev); lookahead_ready = false;
                }
                this->template fence<HardEvent::V_S>(); work_diag[k] = row_k.GetValue(k); work_off[k] = (k +
                1u < n) ? row_k.GetValue(k + 1u) : 0.0f; tau_mask[k] = 0u; continue;
            }
            u32 c_first = (k + 1u) & ~7u, rem = (k + 1u) & 7u, len_k = n - c_first; EvalRowTailNormMedium(row_k,
            c_first, rem, len_k, lookahead_ready, lookahead_ev); float sq_sum = ub_red.GetValue(0), norm =
            ScalarSqrt(sq_sum); work_diag[k] = row_k.GetValue(k); work_off[k] = 0.0f; tau_mask[k] = (norm != 0.0f) ?
            1u : 0u;
            if (norm == 0.0f) {
                continue;
            }
            BuildHouseholderMediumV(row_k, k, c_first, rem, len_k, sq_sum, norm, work_off);
            if (stride <= 64u) {
                TridiagMediumSmallUpdate(k, c_first, rem, len_k, rep_st, lookahead_ready, lookahead_ev);
            } else {
                u32 act_rows = n - c_first; Muls(ub_prod[c_first], ub_v[c_first], -0.5f, len_k);
                if (rem == 0u && c_first >= 8u) {
                    Duplicate(ub_refl[k * stride + c_first - 8u], 0.0f, 8u);
                }
                Adds(ub_refl[k * stride + c_first], ub_v[c_first], 0.0f, stride - c_first);
                MatVecRowsSub(ub_mat[c_first * stride], ub_v, ub_w[c_first], ub_vecs, act_rows, c_first);
                VecDotAxpyMaskBrcbSub(ub_w, ub_v, c_first, rem, stride, ub_gather, c_first, 0u);
                Rank2UpdateRowsSub(ub_mat[c_first * stride], ub_v, ub_w, ub_v[c_first], ub_w[c_first], ub_vecs,
                act_rows, c_first);
            }
        }
    }
    __aicore__ inline void ApplyUnitReflectorPairOrStep(LTf vecs_dst, LTf v_k, LTf v_km1, LTf scratch_base,
    u32 sc_stride, u32 rows, u32 uk, u8 tau_k, u8 tau_km1) {
        if (tau_k != 0u && tau_km1 != 0u) {
            BacktransformPairUnitRowsSub(vecs_dst, v_k, v_km1, scratch_base, scratch_base[rows * sc_stride], rows,
            uk);
        } else {
            if (tau_k != 0u) {
                BacktransformStepRowsSub(vecs_dst, v_k, -1.0f, scratch_base, rows, (uk + 1u) & ~7u);
            }
            if (tau_km1 != 0u) {
                BacktransformStepRowsSub(vecs_dst, v_km1, -1.0f, scratch_base, rows, uk & ~7u);
            }
        }
    }
    __aicore__ inline void BacktransformMedium(const u8 *tau_mask) {
        if (row_count == 0u || n < 3u) {
            return;
        }
        PipeV();
        if (n <= 120u) {
            bool can_pair = (2u * row_count <= rows_per_lane) && (row_count <= 64u); LTf scratch0 = ub_mat; i32 k =
            i32(n) - 3;
            for (; can_pair && k >= 1; k -= 2) {
                u32 uk = u32(k), raw_sc = ((n - (uk & ~7u)) + 7u) & ~7u, sc_stride = (raw_sc >= 64u) ? raw_sc :
                ((stride < 64u) ? stride : 64u); ApplyUnitReflectorPairOrStep(ub_vecs, ub_refl[uk * stride],
                ub_refl[(uk - 1u) * stride], scratch0, sc_stride, row_count, uk, tau_mask[uk], tau_mask[uk - 1u]);
            }
            for (; k >= 0; --k) {
                u32 uk = u32(k);
                if (tau_mask[uk] != 0u) {
                    BacktransformStepRowsSub(ub_vecs, ub_refl[uk * stride], -1.0f, scratch0, row_count, (uk +
                    1u) & ~7u);
                }
            }
        }
        PipeAll();
    }
    __aicore__ inline void IssueReloadRefl0To23(u64 b, u64 priv_gm_256, i32 &bt_mte2_ev) {
        this->template fence<HardEvent::V_MTE2>();
        if constexpr (K96_PASS) {
            DataCopy(ub_mat[29184], gm_ckpt[b * 65536ull + 25600u], 6144u);
        } else {
            DataCopyPad(ub_mat[29184], gm_eigvecs[priv_gm_256], DCEP{256u, 64u, 960u, 0, 0}, DCPEP{false, 0, 0,
            0.0f}); DataCopyPad(ub_mat[33280], gm_eigvecs[priv_gm_256 + 16u], DCEP{256u, 32u, 992u, 0, 0},
            DCPEP{false, 0, 0, 0.0f});
        }
        bt_mte2_ev = i32(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V)); SetFlag<HardEvent::MTE2_V>(bt_mte2_ev);
    }
    __aicore__ inline void BacktransformLarge256(u64 b, const u8 *tau_mask) {
        u64 priv_gm_256 = b * 65536ull + vec_row_start; i32 bt_mte2_ev = 0, k = 253;
        for (; k >= 1; k -= 2) {
            u32 uk = u32(k);
            if (uk == 95u) {
                IssueReloadRefl0To23(b, priv_gm_256, bt_mte2_ev);
            }
            if (uk == 23u) {
                WaitFlag<HardEvent::MTE2_V>(bt_mte2_ev);
            }
            u32 scratch_off = (uk >= 224u) ? 30208u : ((uk >= 192u) ? 29184u : ((uk >= 160u) ? 27136u :
            ((uk >= 144u) ? 24064u : ((uk >= 128u) ? 22272u : ((uk >= 112u) ? 20224u : ((uk >= 96u) ? 17920u :
            ((uk >= 24u) ? 15360u : 6144u))))))), raw_sc = ((256u - (uk & ~7u)) + 7u) & ~7u, sc_stride =
            (raw_sc >= 64u) ? raw_sc : 64u; ApplyUnitReflectorPairOrStep(ub_vecs, GetStaircaseReflector(uk),
            GetStaircaseReflector(uk - 1u), ub_mat[scratch_off], sc_stride, row_count, uk, tau_mask[uk],
            tau_mask[uk - 1u]);
        }
        for (; k >= 0; --k) {
            u32 uk = u32(k);
            if (tau_mask[uk] != 0u) {
                BacktransformStepRowsSub(ub_vecs, GetStaircaseReflector(uk), -1.0f, ub_mat[6144], row_count, (uk +
                1u) & ~7u);
            }
        }
    }
    __aicore__ inline void BacktransformLargeZonePairStep(u32 uk, u8 tau_k, u8 tau_km1) {
        u32 z_k = uk >> 5u, z_k_row = z_k << 5u, scratch_off = zone_offset[z_k] + (uk + 1u - z_k_row) * (stride -
        z_k_row), avail_scratch = vecs_offset - scratch_off; LTf scratch_base = ub_mat[scratch_off]; u32 raw_sc =
        ((n - (uk & ~7u)) + 7u) & ~7u, sc_stride = (raw_sc >= 64u) ? raw_sc : 64u; LTf v_k =
        GetStaircaseReflector(uk), v_km1 = GetStaircaseReflector(uk - 1u); u32 max_pair_rows =
        avail_scratch / (2u * sc_stride), bt_tile_rows = (max_pair_rows >= row_count) ? row_count :
        (max_pair_rows & ~7u);
        if (tau_k != 0u && tau_km1 != 0u && bt_tile_rows >= 8u) {
            for (u32 r_off = 0u; r_off < row_count; r_off += bt_tile_rows) {
                u32 cur_rows = (row_count - r_off < bt_tile_rows) ? (row_count - r_off) : bt_tile_rows;
                BacktransformPairUnitRowsSub(ub_vecs[r_off * stride], v_k, v_km1, scratch_base,
                scratch_base[cur_rows * sc_stride], cur_rows, uk);
            }
            return;
        }
        u32 max_single_rows = avail_scratch / stride, single_tile = (max_single_rows >= row_count) ? row_count :
        (max_single_rows & ~7u);
        for (u32 r_off = 0u; r_off < row_count; r_off += single_tile) {
            u32 cur_rows = (row_count - r_off < single_tile) ? (row_count - r_off) : single_tile;
            ApplyUnitReflectorPairOrStep(ub_vecs[r_off * stride], v_k, v_km1, scratch_base, sc_stride, cur_rows, uk,
            tau_k, 0u);
        }
        for (u32 r_off = 0u; r_off < row_count; r_off += single_tile) {
            u32 cur_rows = (row_count - r_off < single_tile) ? (row_count - r_off) : single_tile;
            ApplyUnitReflectorPairOrStep(ub_vecs[r_off * stride], v_k, v_km1, scratch_base, sc_stride, cur_rows, uk,
            0u, tau_km1);
        }
    }
    __aicore__ inline void BacktransformLargeZone(const u8 *tau_mask) {
        i32 k = i32(n) - 3;
        for (; k >= 1; k -= 2) {
            u32 uk = u32(k); BacktransformLargeZonePairStep(uk, tau_mask[uk], tau_mask[uk - 1u]);
        }
        for (; k >= 0; --k) {
            u32 uk = u32(k);
            if (tau_mask[uk] == 0u) {
                continue;
            }
            u32 z_k = uk >> 5u, z_k_row = z_k << 5u, scratch_off = zone_offset[z_k] + (uk + 1u - z_k_row) * (stride -
            z_k_row), max_single_rows = (vecs_offset - scratch_off) / stride, single_tile =
            (max_single_rows >= row_count) ? row_count : (max_single_rows & ~7u); LTf v_k = GetStaircaseReflector(uk);
            for (u32 r_off = 0u; r_off < row_count; r_off += single_tile) {
                u32 cur_rows = (row_count - r_off < single_tile) ? (row_count - r_off) : single_tile;
                BacktransformStepRowsSub(ub_vecs[r_off * stride], v_k, -1.0f, ub_mat[scratch_off], cur_rows, (uk +
                1u) & ~7u);
            }
        }
    }
    __aicore__ inline void BacktransformLarge(u64 b, const u8 *tau_mask) {
        if (row_count == 0u || n < 3u) {
            return;
        }
        PipeV();
        if (n <= 200u) {
            i32 k = i32(n) - 3;
            for (; k >= 1; k -= 2) {
                u32 uk = u32(k), scratch_off = (n == 200u) ? 22144u : ((uk + 1u) * stride); LTf v_k = (n == 200u) ?
                GetReflector200(uk) : ub_mat[uk * stride], v_km1 = (n == 200u) ? GetReflector200(uk - 1u) :
                ub_mat[(uk - 1u) * stride]; u32 raw_sc = ((n - (uk & ~7u)) + 7u) & ~7u, sc_stride = (raw_sc >= 64u) ?
                raw_sc : 64u; ApplyUnitReflectorPairOrStep(ub_vecs, v_k, v_km1, ub_mat[scratch_off], sc_stride,
                row_count, uk, tau_mask[uk], tau_mask[uk - 1u]);
            }
            for (; k >= 0; --k) {
                u32 uk = u32(k);
                if (tau_mask[uk] != 0u) {
                    LTf v_k = (n == 200u) ? GetReflector200(uk) : ub_mat[uk * stride]; u32 scratch_off =
                    (n == 200u) ? 22144u : ((uk + 1u) * stride); BacktransformStepRowsSub(ub_vecs, v_k, -1.0f,
                    ub_mat[scratch_off], row_count, (uk + 1u) & ~7u);
                }
            }
        } else if constexpr (!FAST200) {
            (n == 256u) ? (void)BacktransformLarge256(b, tau_mask) : (void)BacktransformLargeZone(tau_mask);
        }
    }
    __aicore__ inline void SelectCoopTailConfig(u32 &k_switch, u32 &tail_rows, u32 &tail_stride, bool &expanded_tail,
    u32 &tail_scratch_cap) const {
        u32 plane_cap = rows_per_lane * stride, tail_dim = (plane_cap >= 144u * 144u) ? 144u :
        ((plane_cap >= 136u * 136u) ? 136u : ((plane_cap >= 128u * 128u) ? 128u : ((plane_cap >= 96u * 96u) ? 96u :
        0u))); k_switch = (n > tail_dim && tail_dim >= 64u) ? ((n - tail_dim) & ~7u) : (n - 2u); tail_rows = n -
        k_switch; tail_stride = (tail_rows + 7u) & ~7u;
        if (k_switch < 8u || k_switch + 3u >= n || tail_rows * tail_stride > plane_cap) {
            k_switch = n - 2u;
        }
        expanded_tail = false; tail_scratch_cap = plane_cap;
        if (n == 256u) {
            const u32 arena_cap = 2u * plane_cap;
            for (u32 target = 192u; target >= 96u; target -= 8u) {
                u32 cand_k = (n - target) & ~7u, cand_rows = n - cand_k, cand_stride = (cand_rows + 7u) & ~7u,
                mat_need = cand_rows * cand_stride;
                if (cand_k >= 8u && cand_k + 3u < n && mat_need + 24u * cand_stride <= arena_cap) {
                    k_switch = cand_k; tail_rows = cand_rows; tail_stride = cand_stride; tail_scratch_cap =
                    arena_cap - mat_need; expanded_tail = true; break;
                }
            }
        }
    }
    __aicore__ inline void BuildHouseholderVecCoop(u32 k, u32 c_first, u32 rem, float norm, float &tau_k) {
        if (norm == 0.0f) {
            return;
        }
        float x0 = ub_v.GetValue(k + 1u); ub_v.SetValue(k + 1u, 0.0f); float beta = (x0 >= 0.0f) ? -norm : norm;
        ub_off.SetValue(k, beta); tau_k = (beta - x0) / beta; Muls(ub_v[c_first], ub_v[c_first], 1.0f / (x0 - beta),
        n - c_first); PipeV(); Add(ub_v[c_first], ub_v[c_first], ub_mask_eq[rem * 8u], 8u);
    }
    __aicore__ inline void CopyCoopPreSyncOutputs(u32 k, u32 k_switch, u32 tail_stride, u64 w_bank, u32 c_first,
    float tau_k, u32 r0, u32 act_rows) {
        if ((k % group_lanes) == lane_id && tau_k != 0.0f) {
            DataCopy(gm_ws[ws_refl_offset + u64(k) * stride + c_first], ub_v[c_first], stride - c_first);
        }
        if (tau_k != 0.0f && act_rows > 0u) {
            DataCopy(gm_ws[w_bank + row_start + r0], ub_row[r0], (act_rows + 7u) & ~7u);
        }
        if (k + 1u >= row_start && k + 1u < row_start + row_count) {
            DataCopy(gm_ws[w_bank + stride], ub_mat[(k + 1u - row_start) * stride], stride);
        }
        if (k == k_switch && row_start + row_count > k_switch) {
            u32 r_ts = (row_start >= k_switch) ? 0u : (k_switch - row_start), g_ts = row_start + r_ts;
            DataCopyPad(gm_ws[ws_tail_offset + u64(g_ts - k_switch) * tail_stride], ub_mat[r_ts * stride + k_switch],
            DCEP{u16(row_count - r_ts), tail_stride * 4u, (stride - tail_stride) / 8u, 0, 0});
        }
        if (k + 3u == n && (n - 1u) >= row_start && (n - 1u) < row_start + row_count) {
            ub_comm.SetValue(0, ub_mat.GetValue((n - 1u - row_start) * stride + (n - 1u))); PipeAll();
            DataCopy(gm_ws[w_bank + 2u * stride], ub_comm, 8u);
        }
    }
    __aicore__ inline void TridiagCoopPreSyncStep(u32 k, u32 k_switch, u32 tail_stride, u64 w_bank, u32 c_first,
    u32 rem, float &tau_k, u32 &r0, u32 &act_rows) {
        Mul(ub_v[c_first], ub_cur_row[c_first], ub_mask_ge[rem * 8u], 8u);
        if (stride > c_first + 8u) {
            Adds(ub_v[c_first + 8u], ub_cur_row[c_first + 8u], 0.0f, stride - c_first - 8u);
        }
        float norm = ScalarSqrt(VecDotSub(ub_v, ub_v, c_first)); ub_diag.SetValue(k, ub_cur_row.GetValue(k));
        BuildHouseholderVecCoop(k, c_first, rem, norm, tau_k); ub_tau.SetValue(k, tau_k); r0 = (row_start +
        row_count <= k + 1u) ? row_count : ((row_start < k + 1u) ? ((k + 1u - row_start) & ~7u) : 0u); act_rows =
        (r0 < row_count) ? (row_count - r0) : 0u;
        if (tau_k != 0.0f && act_rows > 0u) {
            PipeV(); MatVecRowsSub(ub_mat[r0 * stride], ub_v, ub_row[r0], ub_vecs, act_rows, c_first);
            Muls(ub_row[r0], ub_row[r0], tau_k, act_rows);
            if (row_start + r0 <= k) {
                PipeV(); Mul(ub_row[r0], ub_row[r0], ub_mask_ge[rem * 8u], 8u);
            }
        }
        PipeAll(); CopyCoopPreSyncOutputs(k, k_switch, tail_stride, w_bank, c_first, tau_k, r0, act_rows);
    }
    __aicore__ inline void UpdateCoopCurRowAndActiveRows(u32 c_first, u32 rem, u32 r0, u32 act_rows) {
        u32 len = n - c_first; Mul(ub_red, ub_w[c_first], ub_mask_eq[rem * 8u], 8u); Sub(ub_cur_row[c_first],
        ub_cur_row[c_first], ub_w[c_first], len); PipeV(); WholeReduceSum(ub_red, ub_red, 8u, 1, 1, 1, 1); PipeV();
        Brcb(ub_tmp, ub_red, 1u, {1, 8}); PipeV(); BRP mul_rp{1, 1, 0, 8, 8, 0};
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u; Mul(ub_prod[c], ub_v[c], ub_tmp, width, 1u, mul_rp);
        }
        PipeV(); Sub(ub_cur_row[c_first], ub_cur_row[c_first], ub_prod[c_first], len); (act_rows > 0u) ?
        (void)Rank2UpdateRowsSub(ub_mat[r0 * stride], ub_v, ub_w, ub_v[row_start + r0], ub_w[row_start + r0],
        ub_vecs, act_rows, c_first) : (void)PipeV();
    }
    __aicore__ inline void TridiagCoopPostSyncStep(u32 k, u32 k_switch, u32 tail_rows, u32 tail_stride,
    bool expanded_tail, u32 tail_scratch_cap, LTf tail_scratch, u64 w_bank, u32 c_first, u32 rem, float tau_k,
    u32 r0, u32 act_rows) {
        u32 load_len = (2u * stride - c_first) + ((k + 3u == n) ? 8u : 0u); DataCopy(ub_w[c_first], gm_ws[w_bank +
        c_first], load_len);
        if (k == k_switch) {
            DataCopy(ub_mat, gm_ws[ws_tail_offset], tail_rows * tail_stride);
        }
        PipeAll();
        if (tau_k != 0.0f) {
            VecDotAxpySub(ub_w, ub_v, -0.5f * tau_k, c_first);
            if (k == k_switch) {
                u32 raw_sc = ((n - c_first) + 7u) & ~7u, sc_stride = (raw_sc >= 64u) ? raw_sc : 64u, tile_r =
                expanded_tail ? ((tail_scratch_cap / sc_stride) & ~7u) : tail_rows; tile_r = (expanded_tail &&
                tile_r > 48u) ? (48u) : tile_r;
                for (u32 r_off = 0u; r_off < tail_rows; r_off += tile_r) {
                    u32 cur_r = (tail_rows - r_off < tile_r) ? (tail_rows - r_off) : tile_r;
                    Rank2UpdateRowsSub(ub_mat[r_off * tail_stride], ub_v, ub_w, ub_v[k_switch + r_off],
                    ub_w[k_switch + r_off], tail_scratch, cur_r, c_first, tail_stride, k_switch);
                }
            } else {
                UpdateCoopCurRowAndActiveRows(c_first, rem, r0, act_rows);
            }
        }
        if (k != k_switch && k + 3u == n) {
            PipeAll(); ub_diag.SetValue(n - 2u, ub_cur_row.GetValue(n - 2u)); ub_off.SetValue(n - 2u,
            ub_cur_row.GetValue(n - 1u)); ub_diag.SetValue(n - 1u, ub_w.GetValue(2u * stride));
        }
    }
    __aicore__ inline void TridiagCoopLocalTail(u32 k_switch, u32 tail_stride, bool expanded_tail,
    u32 tail_scratch_cap, LTf tail_scratch) {
        for (u32 k = k_switch + 1u; k + 2u < n; ++k) {
            u32 tk = k - k_switch, c_first = (k + 1u) & ~7u, tc_first = c_first - k_switch, rem = (k + 1u) & 7u;
            LTf row_tk = ub_mat[tk * tail_stride]; Mul(ub_v[c_first], row_tk[tc_first], ub_mask_ge[rem * 8u], 8u);
            if (tail_stride > tc_first + 8u) {
                Adds(ub_v[c_first + 8u], row_tk[tc_first + 8u], 0.0f, tail_stride - tc_first - 8u);
            }
            float norm = ScalarSqrt(VecDotSub(ub_v, ub_v, c_first)), tau_k = 0.0f; ub_diag.SetValue(k,
            row_tk.GetValue(tk));
            if (norm != 0.0f) {
                BuildHouseholderVecCoop(k, c_first, rem, norm, tau_k); PipeV();
                if ((k % group_lanes) == lane_id) {
                    DataCopy(gm_ws[ws_refl_offset + u64(k) * stride + c_first], ub_v[c_first], stride - c_first);
                }
                u32 act_rows = n - c_first, raw_sc = (act_rows + 7u) & ~7u, sc_stride = (raw_sc >= 64u) ? raw_sc :
                64u, tile_r = expanded_tail ? ((tail_scratch_cap / sc_stride) & ~7u) : act_rows; tile_r =
                (expanded_tail && tile_r > 48u) ? (48u) : tile_r;
                for (u32 r_off = 0u; r_off < act_rows; r_off += tile_r) {
                    u32 cur_r = (act_rows - r_off < tile_r) ? (act_rows - r_off) : tile_r;
                    MatVecRowsSub(ub_mat[(tc_first + r_off) * tail_stride], ub_v, ub_w[c_first + r_off],
                    tail_scratch, cur_r, c_first, tail_stride, k_switch);
                }
                Muls(ub_w[c_first], ub_w[c_first], tau_k, act_rows); VecDotAxpySub(ub_w, ub_v, -0.5f * tau_k,
                c_first);
                for (u32 r_off = 0u; r_off < act_rows; r_off += tile_r) {
                    u32 cur_r = (act_rows - r_off < tile_r) ? (act_rows - r_off) : tile_r;
                    Rank2UpdateRowsSub(ub_mat[(tc_first + r_off) * tail_stride], ub_v, ub_w, ub_v[c_first + r_off],
                    ub_w[c_first + r_off], tail_scratch, cur_r, c_first, tail_stride, k_switch);
                }
            }
            ub_tau.SetValue(k, tau_k);
        }
        PipeAll(); u32 tk_nm2 = (n - 2u) - k_switch, tk_nm1 = (n - 1u) - k_switch; ub_diag.SetValue(n - 2u,
        ub_mat.GetValue(tk_nm2 * tail_stride + tk_nm2)); ub_off.SetValue(n - 2u,
        ub_mat.GetValue(tk_nm2 * tail_stride + tk_nm1)); ub_diag.SetValue(n - 1u,
        ub_mat.GetValue(tk_nm1 * tail_stride + tk_nm1));
    }
    __aicore__ inline void TridiagCoopSmallN2() {
        if (!is_active) {
            return;
        }
        ub_diag.SetValue(0, ub_cur_row.GetValue(0));
        if (n == 2u) {
            ub_off.SetValue(0, ub_cur_row.GetValue(1));
            if (1u >= row_start && 1u < row_start + row_count) {
                ub_diag.SetValue(1, ub_mat.GetValue((1u - row_start) * stride + 1u));
            }
        }
    }
    __aicore__ inline void TridiagonalizeCoop() {
        if (n <= 2u) {
            TridiagCoopSmallN2(); return;
        }
        u32 k_switch = 0u, tail_rows = 0u, tail_stride = 0u, tail_scratch_cap = 0u; bool expanded_tail = false;
        SelectCoopTailConfig(k_switch, tail_rows, tail_stride, expanded_tail, tail_scratch_cap); LTf tail_scratch =
        expanded_tail ? ub_mat[tail_rows * tail_stride] : ub_vecs; u32 k_coop_end = (k_switch < n - 2u) ? (k_switch +
        1u) : (n - 2u);
        for (u32 k = 0u; k < k_coop_end; ++k) {
            u64 w_bank = ws_w_offset + u64(k & 1u) * 3u * stride; u32 c_first = (k + 1u) & ~7u, rem = (k + 1u) & 7u,
            r0 = 0u, act_rows = 0u; float tau_k = 0.0f;
            if (is_active) {
                TridiagCoopPreSyncStep(k, k_switch, tail_stride, w_bank, c_first, rem, tau_k, r0, act_rows);
            }
            PipeAll(); SyncAll<true>();
            if (is_active) {
                TridiagCoopPostSyncStep(k, k_switch, tail_rows, tail_stride, expanded_tail, tail_scratch_cap,
                tail_scratch, w_bank, c_first, rem, tau_k, r0, act_rows);
            }
        }
        if (k_switch < n - 2u) {
            if (is_active) {
                TridiagCoopLocalTail(k_switch, tail_stride, expanded_tail, tail_scratch_cap, tail_scratch);
            }
            PipeAll(); SyncAll<true>();
        } else if (is_active) {
            PipeAll(); ub_diag.SetValue(n - 2u, ub_cur_row.GetValue(n - 2u)); ub_off.SetValue(n - 2u,
            ub_cur_row.GetValue(n - 1u)); float last_diag = ub_w.GetValue(2u * stride);
            if (ub_tau.GetValue(n - 3u) != 0.0f) {
                last_diag -= 2.0f * ub_v.GetValue(n - 1u) * ub_w.GetValue(n - 1u);
            }
            ub_diag.SetValue(n - 1u, last_diag);
        }
        PipeAll();
    }
    __aicore__ inline void BacktransformCoopTileStep(i32 k_start, i32 k_end, bool can_pair, const float *work,
    LTf ub_refl_tile, LTf scratch0, LTf scratch1) {
        i32 kk = k_end;
        for (; can_pair && kk - 1 >= k_start; kk -= 2) {
            u32 k = u32(kk); float tau_k = work[k], tau_km1 = work[k - 1u]; LTf v_k = ub_refl_tile[u32(kk -
            k_start) * stride], v_km1 = ub_refl_tile[u32(kk - 1 - k_start) * stride];
            if (tau_k != 0.0f && tau_km1 != 0.0f) {
                BacktransformPairRowsSub(ub_vecs, v_k, v_km1, tau_k, tau_km1, scratch0, scratch1, row_count, k);
            } else if (tau_k != 0.0f) {
                BacktransformStepRowsSub(ub_vecs, v_k, -tau_k, scratch0, row_count, (k + 1u) & ~7u);
            } else if (tau_km1 != 0.0f) {
                BacktransformStepRowsSub(ub_vecs, v_km1, -tau_k * 0.0f - tau_km1, scratch0, row_count, k & ~7u);
            }
        }
        for (; kk >= k_start; --kk) {
            u32 k = u32(kk); float tau_k = work[k];
            if (tau_k != 0.0f) {
                LTf v_k = ub_refl_tile[u32(kk - k_start) * stride]; BacktransformStepRowsSub(ub_vecs, v_k, -tau_k,
                scratch0, row_count, (k + 1u) & ~7u);
            }
        }
    }
};
} // namespace AscendC
#endif
