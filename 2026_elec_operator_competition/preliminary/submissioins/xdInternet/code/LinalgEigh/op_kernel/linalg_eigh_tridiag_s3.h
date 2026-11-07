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
#ifndef LINALG_EIGH_TRIDIAG_S3_H
#define LINALG_EIGH_TRIDIAG_S3_H
#include "linalg_eigh_tridiag_s2.h"
namespace AscendC {
template <typename DT_A, u32 SOLVE_MODE = 0u, bool FAST200 = false, bool K96_PASS = false>
class KernelEighTridiagS3 : public KernelEighTridiagS2<DT_A, SOLVE_MODE, FAST200, K96_PASS> {
public:
    using Base = KernelEighTridiagS2<DT_A, SOLVE_MODE, FAST200, K96_PASS>;
    using Base::vec_row_start, Base::num_zones, Base::zone_offset, Base::ub_mat, Base::ub_v, Base::ub_w;
    using Base::ub_prod, Base::ub_tmp, Base::ub_red, Base::ub_diag, Base::ub_gather, Base::ub_mask_ge;
    using Base::ub_mask_eq, Base::ScalarRsqrt, Base::ScalarSqrt, Base::ConsumeLookaheadReady;
    using Base::VecDotAxpyMaskBrcbSub, Base::VecDotAxpyNegWMaskBrcbSub, Base::IssueLookaheadNormRed;
    using Base::GetTail256ZoneLayout, Base::Tridiag256TailStep, Base::SpillAndShift256, Base::Tridiag256ZoneP1;
    using Base::TridiagSub256ZoneP1, Base::Tridiag256ZoneP2Lt15, Base::MaskedZoneP2RowOp;
    using Base::Tridiag256ZoneP2Ge15, Base::GetSub256ZoneP2ScV, Base::FoldSub256ZoneP2Gt16Row, Base::n;
    using Base::stride, Base::rows_per_lane;
    __aicore__ inline KernelEighTridiagS3() = default;
    __aicore__ inline void TridiagSub256ZoneP2Mul(u32 zb0, u32 zb1, u32 k, u32 c_first, u32 p23_z_start,
    u32 first_z_next, u32 z_shift, u32 first_act_rows, u32 vacc_rows, u32 vacc_stride, LTf ub_vacc, LTf v_scratch,
    LTf brcb_v_p2, bool &any_gt16) {
        u32 sc_off = 0u;
        for (u32 z = zb0; z < zb1; ++z) {
            u32 z_row = z << 5u, z_next = z_row + 32u, stride_z = stride - z_row, r_start = (k + 1u > z_row) ? (k +
            1u) : z_row, act_rows = z_next - r_start, sc_v = GetSub256ZoneP2ScV(z, p23_z_start, z_next, vacc_stride);
            u8 rep_stride = u8(stride_z / 8u), sc_rep_stride = u8(sc_v / 8u); LTf mat_z = ub_mat[zone_offset[z] +
            (r_start - z_row) * stride_z - z_shift], sc_z = (z == p23_z_start) ? ub_vacc : v_scratch[sc_off],
            brcb_z = brcb_v_p2[(r_start - c_first) * 8u];
            for (u32 c = z_next; c < n; c += 64u) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; Mul(sc_z[c - z_next], mat_z[c - z_row], brcb_z, width,
                act_rows, BRP{1, 1, 0, sc_rep_stride, rep_stride, 1});
            }
            any_gt16 = (act_rows > 16u) ? (true) : any_gt16;
            if (z == p23_z_start) {
                if (first_act_rows < vacc_rows) {
                    Adds(ub_vacc[first_act_rows * vacc_stride], ub_w[first_z_next], 0.0f, n - first_z_next);
                }
                sc_off = (act_rows > vacc_rows) ? ((act_rows - vacc_rows) * vacc_stride) : 0u;
            } else {
                sc_off += act_rows * sc_v;
            }
        }
        PipeV();
    }
    __aicore__ inline void TridiagSub256ZoneP2Accum(u32 zb0, u32 zb1, u32 k, u32 p23_z_start, u32 first_z_next,
    u32 first_act_rows, u32 vacc_rows, u32 vacc_stride, u8 vacc_rep_stride, LTf ub_vacc, LTf v_scratch) {
        u32 add_z0 = (zb0 == p23_z_start) ? (zb0 + 1u) : zb0;
        if (add_z0 >= zb1) {
            return;
        }
        u32 sc_off = (zb0 == p23_z_start && first_act_rows > vacc_rows) ? ((first_act_rows -
        vacc_rows) * vacc_stride) : 0u;
        for (u32 z = add_z0; z < zb1; ++z) {
            u32 z_row = z << 5u, z_next = z_row + 32u, r_start = (k + 1u > z_row) ? (k + 1u) : z_row, act_rows =
            z_next - r_start, rem_r = (act_rows > vacc_rows) ? vacc_rows : act_rows, sc_v = GetSub256ZoneP2ScV(z,
            p23_z_start + 99u, z_next, vacc_stride); u8 sc_rep_stride = u8(sc_v / 8u); LTf sc_z = v_scratch[sc_off];
            for (u32 c = z_next; c < n; c += 64u) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; Add(ub_vacc[c - first_z_next], ub_vacc[c - first_z_next],
                sc_z[c - z_next], width, rem_r, BRP{1, 1, 1, vacc_rep_stride, vacc_rep_stride, sc_rep_stride});
            }
            sc_off += act_rows * sc_v;
        }
        PipeV();
    }
    __aicore__ inline void TridiagSub256ZoneP2Batch(u32 zb0, u32 zb1, u32 k, u32 c_first, u32 p23_z_start,
    u32 first_z_next, u32 z_shift, u32 first_act_rows, u32 vacc_rows, u32 vacc_stride, u8 vacc_rep_stride,
    LTf ub_vacc, LTf v_scratch, LTf brcb_v_p2) {
        bool any_gt16 = false; TridiagSub256ZoneP2Mul(zb0, zb1, k, c_first, p23_z_start, first_z_next, z_shift,
        first_act_rows, vacc_rows, vacc_stride, ub_vacc, v_scratch, brcb_v_p2, any_gt16);
        if (vacc_rows <= 16u && any_gt16) {
            u32 sc_off = 0u;
            for (u32 z = zb0; z < zb1; ++z) {
                u32 z_row = z << 5u, z_next = z_row + 32u, r_start = (k + 1u > z_row) ? (k + 1u) : z_row, act_rows =
                z_next - r_start, sc_v = GetSub256ZoneP2ScV(z, p23_z_start, z_next, vacc_stride); LTf sc_z =
                (z == p23_z_start) ? ub_vacc : v_scratch[sc_off];
                if (act_rows > 16u) {
                    FoldSub256ZoneP2Gt16Row(sc_z, z_next, sc_v, act_rows - 16u);
                }
                if (z == p23_z_start && n == stride && first_act_rows >= vacc_rows && first_act_rows < 32u) {
                    Add(ub_vacc[15u * vacc_stride], ub_vacc[15u * vacc_stride], ub_w[first_z_next], n - first_z_next);
                }
                sc_off = (z == p23_z_start) ? ((act_rows > vacc_rows) ? ((act_rows - vacc_rows) * vacc_stride) :
                0u) : (sc_off + act_rows * sc_v);
            }
            PipeV();
        }
        TridiagSub256ZoneP2Accum(zb0, zb1, k, p23_z_start, first_z_next, first_act_rows, vacc_rows, vacc_stride,
        vacc_rep_stride, ub_vacc, v_scratch);
    }
    __aicore__ inline void TridiagSub256ZoneP2Fold(u32 c_first, u32 first_z_next, u32 first_act_rows, u32 vacc_rows,
    u32 vacc_stride, u8 vacc_rep_stride, LTf ub_vacc) {
        if (n == stride) {
            if (vacc_rows > 16u) {
                Add(ub_vacc, ub_vacc, ub_vacc[16u * vacc_stride], (vacc_rows - 16u) * vacc_stride); PipeV();
            }
            if (vacc_rows > 8u) {
                u32 r16 = (vacc_rows > 16u) ? 16u : vacc_rows; Add(ub_vacc, ub_vacc, ub_vacc[8u * vacc_stride],
                (r16 - 8u) * vacc_stride); PipeV();
            }
            Add(ub_vacc, ub_vacc, ub_vacc[4u * vacc_stride], 4u * vacc_stride); PipeV(); Add(ub_vacc, ub_vacc,
            ub_vacc[2u * vacc_stride], 2u * vacc_stride); PipeV();
            if (first_act_rows < 32u) {
                Add(ub_w[first_z_next], ub_vacc, ub_vacc[vacc_stride], n - first_z_next); Muls(ub_prod[c_first],
                ub_v[c_first], (n == 256u) ? 0.5f : -0.5f, n - c_first); PipeV();
            } else {
                Add(ub_vacc, ub_vacc, ub_vacc[vacc_stride], n - first_z_next); PipeV(); Add(ub_w[first_z_next],
                ub_w[first_z_next], ub_vacc, n - first_z_next); Muls(ub_prod[c_first], ub_v[c_first], (n == 256u) ?
                0.5f : -0.5f, n - c_first); PipeV();
            }
            return;
        }
        for (u32 half = 16u; half > 0u; half >>= 1u) {
            if (vacc_rows <= half) {
                continue;
            }
            u32 cur_r = (vacc_rows < 2u * half) ? (vacc_rows - half) : half;
            for (u32 c = first_z_next; c < n; c += 64u) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; Add(ub_vacc[c - first_z_next], ub_vacc[c - first_z_next],
                ub_vacc[half * vacc_stride + (c - first_z_next)], width, cur_r, BRP{1, 1, 1, vacc_rep_stride,
                vacc_rep_stride, vacc_rep_stride});
            }
            PipeV();
        }
        (first_act_rows >= vacc_rows) ? (void)Add(ub_w[first_z_next], ub_w[first_z_next], ub_vacc, n -
        first_z_next) : (void)Adds(ub_w[first_z_next], ub_vacc, 0.0f, n - first_z_next); Muls(ub_prod[c_first],
        ub_v[c_first], (n == 256u) ? 0.5f : -0.5f, n - c_first); PipeV();
    }
    __aicore__ inline void TridiagSub256ZoneP2(u32 k, u32 c_first, u32 p23_z_start, u32 first_z_next, u32 z_shift,
    u32 aligned_p2_rows, u32 first_act_rows, u32 vacc_rows, u32 vacc_stride, u8 vacc_rep_stride, u32 vacc_floats,
    u32 p2_total_cap, LTf ub_tridiag_sc, LTf ub_vacc, LTf brcb_v_p2, bool p2_brcb_issued) {
        if (n == 256u && first_act_rows + 1u < vacc_rows) {
            Duplicate(ub_vacc[(first_act_rows + 1u) * vacc_stride], 0.0f, (vacc_rows - first_act_rows -
            1u) * vacc_stride); PipeV();
        }
        LTf v_scratch = ub_tridiag_sc[vacc_floats]; u32 v_scratch_cap = p2_total_cap - vacc_floats;
        if (!p2_brcb_issued) {
            Brcb(brcb_v_p2, ub_v[c_first], u8(aligned_p2_rows / 8u), {1, 8});
            if (first_act_rows + 1u < vacc_rows) {
                Duplicate(ub_vacc[(first_act_rows + 1u) * vacc_stride], 0.0f, (vacc_rows - first_act_rows -
                1u) * vacc_stride);
            }
            PipeV();
        }
        for (u32 zb0 = p23_z_start; zb0 + 1u < num_zones; ) {
            u32 zb1 = zb0, used_sc = 0u, cur_cap = (zb0 == p23_z_start) ? p2_total_cap : v_scratch_cap;
            while (zb1 + 1u < num_zones) {
                u32 z_row = zb1 << 5u, z_next = z_row + 32u, r_start = (k + 1u > z_row) ? (k + 1u) : z_row,
                act_rows = z_next - r_start, sc_v = GetSub256ZoneP2ScV(zb1, p23_z_start, z_next, vacc_stride),
                need_sc = (zb1 == p23_z_start && act_rows < vacc_rows) ? vacc_floats : (act_rows * sc_v);
                if (zb1 > zb0 && used_sc + need_sc > cur_cap) {
                    break;
                }
                used_sc += need_sc; ++zb1;
            }
            TridiagSub256ZoneP2Batch(zb0, zb1, k, c_first, p23_z_start, first_z_next, z_shift, first_act_rows,
            vacc_rows, vacc_stride, vacc_rep_stride, ub_vacc, v_scratch, brcb_v_p2); zb0 = zb1;
        }
        TridiagSub256ZoneP2Fold(c_first, first_z_next, first_act_rows, vacc_rows, vacc_stride, vacc_rep_stride,
        ub_vacc);
    }
    __aicore__ inline void Tridiag256ZoneP3Pass1(u32 k, u32 c_first, u32 p23_z_start, u32 z_shift, LTf brcb_v) {
        SetMaskNorm(); u32 last_mask = 0u;
        for (u32 z = p23_z_start; z < num_zones; ++z) {
            u32 z_row = z << 5u, r_start = (z == p23_z_start) ? (k + 1u) : z_row, act_rows = z_row + 32u - r_start,
            stride_z = 256u - z_row; u8 z_rep = u8(stride_z >> 3u); LTf mat_z = ub_mat[zone_offset[z] + (r_start -
            z_row) * stride_z - z_shift], bv_z = brcb_v[(r_start - c_first) * 8u]; u32 c0 = (c_first > z_row) ?
            c_first : z_row;
            for (u32 c = c0; c < 256u; c += 64u) {
                u32 width = (256u - c < 64u) ? (256u - c) : 64u;
                if (width != last_mask) {
                    SetVectorMask<float, MaskMode::NORMAL>(width); last_mask = width;
                }
                MulAddDst<float, float, false>(mat_z[c - z_row], ub_w[c], bv_z, MASK_PLACEHOLDER, act_rows, BRP{1, 1,
                0, z_rep, 0, 1});
            }
        }
        SetVectorMask<float, MaskMode::NORMAL>(64u); SetMaskCount(); PipeV();
    }
    __aicore__ inline void Tridiag256ZoneP3(u32 k, LTf row_k, u32 c_first, u32 c_off_k, u32 rem, u32 p23_z_start,
    u32 z_shift, LTf ub_tridiag_sc, bool &lookahead_ready, i32 &lookahead_ev) {
        LTf brcb_v = ub_tridiag_sc[512], brcb_w = ub_tridiag_sc[2560]; u8 init_b_rep = u8((256u - c_first) >> 3u);
        VecDotAxpyNegWMaskBrcbSub(ub_w, ub_v, c_first, rem, 0u, brcb_v, c_first, init_b_rep); Brcb(brcb_w,
        ub_w[c_first], init_b_rep, {1, 8}); Tridiag256ZoneP3Pass1(k, c_first, p23_z_start, z_shift, brcb_v);
        u32 z0_row = p23_z_start << 5u, r0_start = k + 1u, act_rows0 = (z0_row + 32u) - r0_start, stride_z0 = 256u -
        z0_row; LTf mat_z0 = ub_mat[zone_offset[p23_z_start] + (r0_start - z0_row) * stride_z0 - z_shift];
        MaskedZoneP2RowOp(mat_z0, ub_v, brcb_w[(r0_start - c_first) * 8u], c_first, z0_row, 0u, act_rows0,
        u8(stride_z0 >> 3u), 0u, true); PipeV();
        if (p23_z_start + 1u < num_zones && k + 3u < 256u) {
            u32 c_n = (k + 2u) & ~7u; lookahead_ev = IssueLookaheadNormRed(mat_z0, ub_tridiag_sc, ub_tridiag_sc[256],
            c_n - z0_row, (k + 2u) & 7u, 256u - c_n); lookahead_ready = true;
        }
        if (p23_z_start + 1u < num_zones) {
            for (u32 z = p23_z_start + 1u; z < num_zones; ++z) {
                u32 z_row = z << 5u, stride_z = 256u - z_row; LTf mat_z = ub_mat[zone_offset[z] - z_shift];
                MaskedZoneP2RowOp(mat_z, ub_v, brcb_w[(z_row - c_first) * 8u], z_row, z_row, 0u, 32u,
                u8(stride_z >> 3u), 0u, true);
            }
        }
        if (k != 15u && k != 23u) {
            if (rem == 0u && c_off_k >= 8u) {
                Duplicate(row_k[c_off_k - 8u], 0.0f, 8u);
            }
            Adds(row_k[c_off_k], ub_v[c_first], 0.0f, 256u - c_first);
        }
        PipeV();
    }
    __aicore__ inline void TridiagSub256ZoneP3MulW(u32 zb0, u32 zb1, u32 k, u32 c_first, u32 brcb_start,
    LTf ub_tridiag_sc, LTf brcb_v, LTf brcb_w) {
        u32 sc_off = 0u;
        for (u32 z = zb0; z < zb1; ++z) {
            u32 z_row = z << 5u, r_start = (k + 1u > z_row) ? (k + 1u) : z_row, r_end = (z_row + 32u < n) ? (z_row +
            32u) : n, act_rows = r_end - r_start, c_start = (c_first > z_row) ? c_first : z_row, raw_sc = ((n -
            c_start) + 7u) & ~7u, sc_stride = (raw_sc >= 64u) ? raw_sc : 64u; u8 sc_rep_stride = u8(sc_stride / 8u);
            LTf sc_z = ub_tridiag_sc[sc_off], bv_z = brcb_v[(r_start - brcb_start) * 8u];
            for (u32 c = c_start; c < n; c += 64u) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; Mul(sc_z[c - c_start], ub_w[c], bv_z, width, act_rows,
                BRP{1, 1, 0, sc_rep_stride, 0, 1});
            }
            sc_off += act_rows * sc_stride;
        }
        PipeV();
    }
    __aicore__ inline void StepZoneP3SubPass(u32 zb0, u32 zb1, u32 k, u32 c_first, u32 brcb_start, LTf ub_tridiag_sc,
    LTf brcb_w, bool do_mul_v) {
        u32 sc_off = 0u;
        for (u32 z = zb0; z < zb1; ++z) {
            u32 z_row = z << 5u, stride_z = stride - z_row, r_start = (k + 1u > z_row) ? (k + 1u) : z_row, r_end =
            (z_row + 32u < n) ? (z_row + 32u) : n, act_rows = r_end - r_start, c_start = (c_first > z_row) ?
            c_first : z_row, raw_sc = ((n - c_start) + 7u) & ~7u, sc_stride = (raw_sc >= 64u) ? raw_sc : 64u;
            u8 rep_stride = u8(stride_z / 8u), sc_rep_stride = u8(sc_stride / 8u); LTf mat_z =
            ub_mat[zone_offset[z] + (r_start - z_row) * stride_z], sc_z = ub_tridiag_sc[sc_off], bw_z =
            brcb_w[(r_start - brcb_start) * 8u]; BRP sub_rp{1, 1, 1, rep_stride, rep_stride, sc_rep_stride};
            BRP mul_rp{1, 1, 0, sc_rep_stride, 0, 1};
            for (u32 c = c_start; c < n; c += 64u) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; Sub(mat_z[c - z_row], mat_z[c - z_row], sc_z[c - c_start],
                width, act_rows, sub_rp);
                if (do_mul_v) {
                    Mul(sc_z[c - c_start], ub_v[c], bw_z, width, act_rows, mul_rp);
                }
            }
            sc_off += act_rows * sc_stride;
        }
    }
    __aicore__ inline void TridiagSub256ZoneP3Batch(u32 zb0, u32 zb1, u32 k, u32 c_first, u32 brcb_start,
    LTf ub_tridiag_sc, LTf brcb_v, LTf brcb_w) {
        TridiagSub256ZoneP3MulW(zb0, zb1, k, c_first, brcb_start, ub_tridiag_sc, brcb_v, brcb_w);
        StepZoneP3SubPass(zb0, zb1, k, c_first, brcb_start, ub_tridiag_sc, brcb_w, true); PipeV();
        StepZoneP3SubPass(zb0, zb1, k, c_first, brcb_start, ub_tridiag_sc, brcb_w, false); PipeV();
    }
    __aicore__ inline void FindSub256ZoneP3BatchEnd(u32 zb0, u32 k, u32 c_first, u32 p23_z_start, u32 p3_scratch_cap,
    u32 &zb1, u32 &used_rows) const {
        zb1 = zb0; u32 used_sc = 0u; used_rows = 0u;
        while (zb1 < num_zones) {
            u32 z_row = zb1 << 5u, r_start = (k + 1u > z_row) ? (k + 1u) : z_row, r_end = (z_row + 32u < n) ?
            (z_row + 32u) : n, act_rows = r_end - r_start, c_start = (c_first > z_row) ? c_first : z_row, raw_sc =
            ((n - c_start) + 7u) & ~7u, sc_stride = (raw_sc >= 64u) ? raw_sc : 64u, span_r = r_end -
            ((zb0 == p23_z_start) ? c_first : (zb0 << 5u));
            if (zb1 > zb0 && (used_sc + act_rows * sc_stride > p3_scratch_cap || span_r > 64u)) {
                break;
            }
            used_sc += act_rows * sc_stride; used_rows += act_rows; ++zb1;
        }
    }
    __aicore__ inline void TridiagSub256ZoneP3(u32 k, LTf row_k, u32 c_first, u32 c_off_k, u32 rem, u32 p23_z_start,
    u32 p3_scratch_cap, LTf ub_tridiag_sc) {
        LTf brcb_v = ub_gather[256], brcb_w = ub_prod; u32 brcb_start = c_first, init_b_rows = (n - c_first < 64u) ?
        (n - c_first) : 64u; u8 init_b_rep = u8(((init_b_rows + 7u) & ~7u) / 8u); u32 brcb_end = brcb_start +
        init_b_rows; VecDotAxpyMaskBrcbSub(ub_w, ub_v, c_first, rem, stride, brcb_v, brcb_start, init_b_rep);
        for (u32 zb0 = p23_z_start; zb0 < num_zones; ) {
            u32 zb1 = zb0, used_rows = 0u; FindSub256ZoneP3BatchEnd(zb0, k, c_first, p23_z_start, p3_scratch_cap,
            zb1, used_rows); u32 batch_r0 = (k + 1u > (zb0 << 5u)) ? (k + 1u) : (zb0 << 5u);
            if (zb0 == p23_z_start) {
                Brcb(brcb_w, ub_w[brcb_start], init_b_rep, {1, 8});
            } else if (batch_r0 + used_rows > brcb_end) {
                brcb_start = batch_r0 & ~7u; u32 rem_total = n - brcb_start, b_rows = (rem_total < 64u) ? rem_total :
                64u; u8 brcb_rep = u8(((b_rows + 7u) & ~7u) / 8u); Brcb(brcb_v, ub_v[brcb_start], brcb_rep, {1, 8});
                Brcb(brcb_w, ub_w[brcb_start], brcb_rep, {1, 8}); PipeV(); brcb_end = brcb_start + b_rows;
            }
            TridiagSub256ZoneP3Batch(zb0, zb1, k, c_first, brcb_start, ub_tridiag_sc, brcb_v, brcb_w); zb0 = zb1;
        }
        if (n != stride) {
            if (rem == 0u && c_off_k >= 8u) {
                Duplicate(row_k[c_off_k - 8u], 0.0f, 8u);
            }
            Adds(row_k[c_off_k], ub_v[c_first], 0.0f, stride - c_first); PipeV();
        }
    }
    __aicore__ inline void EvalRowTailNormZone(LTf row_k, u32 c_first, u32 c_off_k, u32 rem, u32 len_k,
    bool &lookahead_ready, i32 lookahead_ev) {
        if (ConsumeLookaheadReady(row_k, c_first, c_off_k, rem, lookahead_ready, lookahead_ev)) {
            return;
        }
        if (rem == 0u) {
            Adds(ub_v[c_first], row_k[c_off_k], 0.0f, stride - c_first);
            if (n == 256u) {
                Mul(ub_prod[c_first], row_k[c_off_k], row_k[c_off_k], len_k);
            }
            PipeV();
        } else {
            Mul(ub_v[c_first], row_k[c_off_k], ub_mask_ge[rem * 8u], 8u);
            if (stride > c_first + 8u) {
                Adds(ub_v[c_first + 8u], row_k[c_off_k + 8u], 0.0f, stride - c_first - 8u);
                if (n == 256u) {
                    Mul(ub_prod[c_first + 8u], row_k[c_off_k + 8u], row_k[c_off_k + 8u], len_k - 8u);
                }
            }
            PipeV();
            if (n == 256u) {
                Mul(ub_prod[c_first], ub_v[c_first], ub_v[c_first], 8u); PipeV();
            }
        }
        if (n != 256u) {
            Mul(ub_prod[c_first], ub_v[c_first], ub_v[c_first], len_k); PipeV();
        }
        (n == 256u && len_k <= 64u) ? (void)WholeReduceSum(ub_red, ub_prod[c_first], len_k, 1, 1, 1, 8) :
        (void)ReduceSum(ub_red, ub_prod[c_first], ub_tmp, len_k); this->template fence<HardEvent::V_S>();
    }
    __aicore__ inline void TridiagZoneGeneralUpdate(u32 k, LTf row_k, u32 c_first, u32 c_off_k, u32 rem, u32 z_shift,
    u32 p1_scratch_cap, u32 p3_scratch_cap, LTf ub_tridiag_sc, LTf ub_vacc, bool &lookahead_ready,
    i32 &lookahead_ev) {
        u32 z_start = c_first >> 5u, p23_z_start = (k + 1u) >> 5u, first_z_next = (p23_z_start + 1u) << 5u;
        bool has_p2 = (first_z_next < n); u32 v_p2_rows = has_p2 ? (((num_zones - 1u) << 5u) - c_first) : 0u,
        aligned_p2_rows = (v_p2_rows + 7u) & ~7u, brcb_p2_floats = aligned_p2_rows * 8u, p2_total_cap = ((19u +
        rows_per_lane) * stride + z_shift) - brcb_p2_floats; LTf brcb_v_p2 = ub_tridiag_sc[p2_total_cap];
        u32 raw_vacc_st = has_p2 ? (stride - first_z_next) : 64u, vacc_stride = (n == stride) ? raw_vacc_st :
        ((raw_vacc_st >= 64u) ? raw_vacc_st : 64u); u8 vacc_rep_stride = u8(vacc_stride / 8u); u32 base_vr =
        (p23_z_start >= 1u && n == 256u) ? 32u : 16u, vacc_rows = (v_p2_rows < base_vr) ? aligned_p2_rows : base_vr,
        vacc_floats = vacc_rows * vacc_stride, first_act_rows = has_p2 ? (first_z_next - (k + 1u)) : 0u;
        bool p2_brcb_issued = false; (n == 256u) ? (void)Tridiag256ZoneP1(k, c_first, z_start, z_shift,
        ub_tridiag_sc, has_p2, aligned_p2_rows, brcb_v_p2, p2_brcb_issued) : (void)TridiagSub256ZoneP1(c_first,
        z_start, z_shift, p1_scratch_cap, ub_tridiag_sc, ub_vacc, has_p2, aligned_p2_rows, first_act_rows, vacc_rows,
        vacc_stride, brcb_v_p2, p2_brcb_issued);
        if (has_p2) {
            if (n == 256u && k < 15u) {
                Tridiag256ZoneP2Lt15(k, c_first, aligned_p2_rows, first_act_rows, ub_tridiag_sc, brcb_v_p2,
                p2_brcb_issued);
            } else if (n == 256u && k >= 15u && p23_z_start + 2u <= 7u) {
                Tridiag256ZoneP2Ge15(k, c_first, p23_z_start, first_z_next, z_shift, aligned_p2_rows, first_act_rows,
                vacc_stride, vacc_rep_stride, ub_tridiag_sc, ub_vacc, brcb_v_p2, p2_brcb_issued);
            } else {
                TridiagSub256ZoneP2(k, c_first, p23_z_start, first_z_next, z_shift, aligned_p2_rows, first_act_rows,
                vacc_rows, vacc_stride, vacc_rep_stride, vacc_floats, p2_total_cap, ub_tridiag_sc, ub_vacc,
                brcb_v_p2, p2_brcb_issued);
            }
        } else {
            Muls(ub_prod[c_first], ub_v[c_first], (n == 256u) ? 0.5f : -0.5f, n - c_first); PipeV();
        }
        (n == 256u) ? (void)Tridiag256ZoneP3(k, row_k, c_first, c_off_k, rem, p23_z_start, z_shift, ub_tridiag_sc,
        lookahead_ready, lookahead_ev) : (void)TridiagSub256ZoneP3(k, row_k, c_first, c_off_k, rem, p23_z_start,
        p3_scratch_cap, ub_tridiag_sc);
    }
    __aicore__ inline void ComputeZoneRowOffset(u32 k, u32 z_shift, u32 &z_k_row, u32 &row_k_off) const {
        if (n == 256u && k >= 80u) {
            u32 tail_off = 0u; GetTail256ZoneLayout(k, z_k_row, tail_off); row_k_off = tail_off + (k -
            z_k_row) * (256u - z_k_row);
        } else {
            u32 z_k = k >> 5u; z_k_row = z_k << 5u; row_k_off = zone_offset[z_k] + (k - z_k_row) * (stride -
            z_k_row) - z_shift;
        }
    }
    __aicore__ inline void BuildHouseholderZoneV(LTf row_k, u32 k, u32 z_k_row, u32 c_first, u32 c_off_k, u32 rem,
    float sq_sum, float norm, float *work_off) {
        float x0 = row_k.GetValue(k + 1u - z_k_row) + 0.0f, beta = (x0 >= 0.0f) ? -norm : norm; work_off[k] = beta;
        float inv_s = ScalarRsqrt(sq_sum - beta * x0);
        if (n == 256u && rem != 0u) {
            Muls(ub_v[c_first], ub_v[c_first], inv_s, 8u);
            if (256u > c_first + 8u) {
                Muls(ub_v[c_first + 8u], row_k[c_off_k + 8u], inv_s, 256u - c_first - 8u);
            }
        } else {
            Muls(ub_v[c_first], (n == 256u) ? row_k[c_off_k] : ub_v[c_first], inv_s, n - c_first);
        }
        Muls(ub_tmp, ub_mask_eq[rem * 8u], -beta * inv_s, 8u); PipeV(); Add(ub_v[c_first], ub_v[c_first], ub_tmp,
        8u); PipeV();
        if (n == 256u && (k == 15u || k == 23u)) {
            Duplicate(row_k[c_off_k - 8u], 0.0f, 8u); Adds(row_k[c_off_k], ub_v[c_first], 0.0f, 256u - c_first);
        }
    }
    __aicore__ inline void TridiagonalizeLargeZoneLoop(u64 b, float *work, u8 *tau_mask) {
        u64 priv_gm_256 = b * 65536ull + vec_row_start; LTf orig_ub_w = ub_w, ub_tridiag_sc = ub_diag;
        u32 p1_scratch_cap = (15u + rows_per_lane) * stride + ((n == 256u) ? 768u : 0u), p3_scratch_cap = (13u +
        rows_per_lane) * stride; LTf ub_vacc = ub_tridiag_sc; u32 z_shift = 0u; bool lookahead_ready = false;
        i32 lookahead_ev = 0;
        for (u32 k = (K96_PASS ? 96u : 0u); k < n; ++k) {
            u32 z_k_row = 0u, row_k_off = 0u; ComputeZoneRowOffset(k, z_shift, z_k_row, row_k_off); LTf row_k =
            ub_mat[row_k_off];
            if (k + 2u >= n) {
                if (lookahead_ready) {
                    WaitFlag<HardEvent::V_S>(lookahead_ev); lookahead_ready = false;
                }
                this->template fence<HardEvent::V_S>(); work[k] = row_k.GetValue(k - z_k_row); work[256u + k] = (k +
                1u < n) ? row_k.GetValue(k + 1u - z_k_row) : 0.0f; tau_mask[k] = 0u; continue;
            }
            u32 c_first = (k + 1u) & ~7u, rem = (k + 1u) & 7u; EvalRowTailNormZone(row_k, c_first, c_first - z_k_row,
            rem, n - c_first, lookahead_ready, lookahead_ev); float sq_sum = ub_red.GetValue(0), norm =
            ScalarSqrt(sq_sum); work[k] = row_k.GetValue(k - z_k_row); work[256u + k] = 0.0f;
            if (n == 256u && k >= 79u) {
                Tridiag256TailStep(k, row_k, z_k_row, c_first, c_first - z_k_row, rem, sq_sum, norm, work + 256,
                tau_mask, lookahead_ready, lookahead_ev); continue;
            }
            tau_mask[k] = (norm != 0.0f) ? 1u : 0u;
            if (norm != 0.0f) {
                BuildHouseholderZoneV(row_k, k, z_k_row, c_first, c_first - z_k_row, rem, sq_sum, norm, work + 256);
            }
            if (n == 256u && (k == 15u || k == 23u)) {
                SpillAndShift256(k, priv_gm_256, z_shift, ub_tridiag_sc, ub_vacc, p1_scratch_cap, p3_scratch_cap);
            }
            if (norm != 0.0f) {
                TridiagZoneGeneralUpdate(k, row_k, c_first, c_first - z_k_row, rem, z_shift, p1_scratch_cap,
                p3_scratch_cap, ub_tridiag_sc, ub_vacc, lookahead_ready, lookahead_ev);
            }
        }
        ub_w = orig_ub_w;
    }
};
} // namespace AscendC
#endif
