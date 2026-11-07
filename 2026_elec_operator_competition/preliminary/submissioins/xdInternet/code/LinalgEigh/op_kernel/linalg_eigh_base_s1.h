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
#ifndef LINALG_EIGH_BASE_S1_H
#define LINALG_EIGH_BASE_S1_H
#include "kernel_operator.h"
#include "linalg_eigh_tiling.h"
namespace AscendC {
using u32 = uint32_t; using i32 = int32_t; using u8 = uint8_t; using u16 = uint16_t; using u64 = uint64_t;
using LTf = LocalTensor<float>; using LTi = LocalTensor<int32_t>; using LTu = LocalTensor<uint32_t>;
using GTf = GlobalTensor<float>; using BRP = BinaryRepeatParams; using DCEP = DataCopyExtParams;
using DCP = DataCopyParams; using DCPEP = DataCopyPadExtParams<float>;
static __aicore__ inline void PipeV() {
    PipeBarrier<PIPE_V>();
}
static __aicore__ inline void PipeAll() {
    PipeBarrier<PIPE_ALL>();
}
static __aicore__ inline float u2f(u32 x) {
    return static_cast<float>(static_cast<i32>(x));
}
struct SturmState4 {
    float m0, m1, m2, m3, pp0, pp1, pp2, pp3, p0, p1, p2, p3; u32 s0, s1, s2, s3, c0, c1, c2, c3;
};
struct PolyState4 {
    float m0, m1, m2, m3, pp0, pp1, pp2, pp3, p0, p1, p2, p3; i32 exp0, exp1, exp2, exp3;
};
struct SturmWorkPtrs {
    float *dd, *ee, *de_pairs; float *q_lo, *q_hi, *iso_lo, *iso_hi; u32 *q_lc, *q_hc, *q_depth, *act_q,
    *act_depth; float *q_plo, *q_phi, *iso_plo, *iso_phi; i32 *q_elo, *q_ehi, *iso_elo, *iso_ehi;
    int8_t *iso_stag; float left_nb_lo, left_nb_hi, left_nb_plo, left_nb_phi; i32 left_nb_elo, left_nb_ehi;
    float right_nb_lo, right_nb_hi, right_nb_plo, right_nb_phi; i32 right_nb_elo, right_nb_ehi;
};
template <typename DT_A, u32 SOLVE_MODE = 0u, bool FAST200 = false, bool K96_PASS = false>
class KernelEighBaseS1 {
public:
    __aicore__ inline KernelEighBaseS1() = default;
    u32 n, stride, uplo, group_lanes, num_groups, rows_per_lane, vec_rows_per_lane, lane_id, group_id, my_blocks,
    coop_row_start, coop_row_count, vec_row_start, vec_row_count, row_start, row_count, solve_mode, num_zones,
    vecs_offset, zone_offset[9]; u64 batch_size, ws_refl_offset, ws_w_offset, ws_scale_offset, ws_tail_offset;
    bool is_active; float matrix_scale; TBuf<TPosition::VECCALC> ub_buf; GTf gm_A; GTf gm_eigvals; GTf gm_eigvecs;
    GTf gm_ws; GTf gm_ckpt; LTf ub_mat, ub_vecs, ub_refl, ub_row, ub_v, ub_w, ub_cur_row, ub_prod, ub_tmp, ub_red,
    ub_diag, ub_off, ub_tau, ub_eig, ub_comm, ub_gather, ub_mask_ge, ub_mask_gt, ub_mask_eq;
    template <HardEvent event>
    __aicore__ inline void fence() const {
        const i32 id = i32(GetTPipePtr()->FetchEventID(event)); SetFlag<event>(id); WaitFlag<event>(id);
    }
    __aicore__ inline float ScalarAbs(float x) const {
        if constexpr (K96_PASS || SOLVE_MODE == 3u) {
            return __builtin_fabsf(x);
        } else {
            return (x < 0.0f) ? -x : x;
        }
    }
    __aicore__ inline float ScalarRsqrtFast(float x) const {
        if (x <= 0.0f) {
            return 0.0f;
        }
        union {
            float f; u32 u;
        } bits;
        bits.f = x; bits.u = 0x5f375a86u - (bits.u >> 1u); float y = bits.f, xhalf = 0.5f * x; y = y * (1.5f -
        xhalf * y * y); y = y * (1.5f - xhalf * y * y); return y;
    }
    __aicore__ inline float ScalarRsqrt(float x) const {
        float y = ScalarRsqrtFast(x); return y * (1.5f - (0.5f * x) * y * y);
    }
    __aicore__ inline float ScalarSqrt(float x) const {
        return x * ScalarRsqrt(x);
    }
    __aicore__ inline void ComputePow2Scale(float max_val, float &scale_out, float &inv_scale_out) const {
        union {
            float f; u32 u;
        } sbits;
        sbits.f = max_val; u32 exp_bits = sbits.u & 0x7f800000u;
        if (exp_bits == 0u || exp_bits == 0x7f800000u) {
            scale_out = 1.0f; inv_scale_out = 1.0f;
        } else {
            sbits.u = exp_bits; scale_out = sbits.f; sbits.u = 0x7f000000u - exp_bits; inv_scale_out = sbits.f;
        }
    }
    __aicore__ inline float ClampPivot(float x) const {
        return (ScalarAbs(x) < 1e-15f) ? ((x < 0.0f) ? -1e-15f : 1e-15f) : x;
    }
    static __aicore__ inline float MaxF(float a, float b) {
        return (b > a) ? b : a;
    }
    static __aicore__ inline float MinF(float a, float b) {
        return (b < a) ? b : a;
    }
    static __aicore__ inline float ClampF(float x, float lo, float hi) {
        return (x < lo) ? lo : ((x > hi) ? hi : x);
    }
    __aicore__ inline bool ConsumeLookaheadReady(LTf row_k, u32 c_first, u32 c_off_k, u32 rem, bool &lookahead_ready,
    i32 lookahead_ev) {
        if (!lookahead_ready) {
            return false;
        }
        if (rem != 0u) {
            Mul(ub_v[c_first], row_k[c_off_k], ub_mask_ge[rem * 8u], 8u); PipeV();
        }
        WaitFlag<HardEvent::V_S>(lookahead_ev); lookahead_ready = false; return true;
    }
    __aicore__ inline bool ComputeSecantWeight(float plo, i32 elo, float phi, i32 ehi, float &t_out) const {
        if (elo <= -10000 || ehi <= -10000) {
            return false;
        }
        i32 ediff = elo - ehi;
        if (ediff < -1 || ediff > 1) {
            return false;
        }
        float fl = (ediff == -1) ? (plo * 5.4210109e-20f) : plo, fh = (ediff == 1) ? (phi * 5.4210109e-20f) : phi,
        denom = fl + fh;
        if (!(denom > 1e-35f)) {
            return false;
        }
        t_out = fl / denom; return true;
    }
    __aicore__ inline void InitVecRangeBalanced() {
        if (lane_id >= group_lanes) {
            vec_row_start = n; vec_row_count = 0u; return;
        }
        u32 base_vecs = n / group_lanes, extra_vecs = n % group_lanes, extra_start = (group_lanes -
        extra_vecs) >> 1u, extra_end = extra_start + extra_vecs, extra_before = 0u;
        if (lane_id > extra_start) {
            extra_before = (lane_id < extra_end) ? (lane_id - extra_start) : extra_vecs;
        }
        u32 has_extra = (lane_id >= extra_start && lane_id < extra_end) ? 1u : 0u; vec_row_start =
        lane_id * base_vecs + extra_before; vec_row_count = (vec_row_start < n) ? (base_vecs + has_extra) : 0u;
    }
    __aicore__ inline void InitVecRange(u32 blk_idx) {
        lane_id = (num_groups == 1u) ? blk_idx : (blk_idx % group_lanes); group_id = (num_groups == 1u) ? 0u :
        (blk_idx / group_lanes);
        if constexpr (SOLVE_MODE == 1u) {
            vec_row_start = lane_id * vec_rows_per_lane; u32 rem = (vec_row_start < n) ? (n - vec_row_start) : 0u;
            vec_row_count = (rem < vec_rows_per_lane) ? rem : vec_rows_per_lane;
        } else if (n == 256u && group_lanes == 10u) {
            if (lane_id == 0u) {
                vec_row_start = 0u; vec_row_count = 24u;
            } else if (lane_id < 9u) {
                vec_row_start = 24u + (lane_id - 1u) * 26u; vec_row_count = 26u;
            } else if (lane_id == 9u) {
                vec_row_start = 232u; vec_row_count = 24u;
            } else {
                vec_row_start = n; vec_row_count = 0u;
            }
        } else {
            InitVecRangeBalanced();
        }
    }
    __aicore__ inline void InitCoopRange() {
        if (solve_mode == 0u) {
            u32 num_blocks = (n + 7u) / 8u, base_blocks = num_blocks / group_lanes, extra_blocks =
            num_blocks % group_lanes, start_block = lane_id * base_blocks + ((lane_id < extra_blocks) ? lane_id :
            extra_blocks); my_blocks = base_blocks + ((lane_id < extra_blocks) ? 1u : 0u); coop_row_start =
            start_block * 8u; u32 coop_end = coop_row_start + my_blocks * 8u; coop_end = (coop_end > n) ? (n) :
            coop_end; coop_row_count = (coop_row_start < n) ? (coop_end - coop_row_start) : 0u;
        } else {
            my_blocks = 0u; coop_row_start = 0u; coop_row_count = n;
        }
        row_start = vec_row_start; row_count = vec_row_count;
    }
    __aicore__ inline void InitMaskTensors() {
        for (u32 rem = 0u; rem < 8u; ++rem) {
            for (u32 c = 0u; c < 8u; ++c) {
                ub_mask_ge.SetValue(rem * 8u + c, (c >= rem) ? 1.0f : 0.0f);
                if (!K96_PASS || n <= 200u) {
                    ub_mask_gt.SetValue(rem * 8u + c, (c > rem) ? 1.0f : 0.0f);
                }
                ub_mask_eq.SetValue(rem * 8u + c, (c == rem) ? 1.0f : 0.0f);
            }
        }
    }
    __aicore__ inline void InitScratchPlane(TPipe *pipe, u32 plane_rows, u32 gather_idx, u32 eig_idx, u32 comm_idx) {
        pipe->InitBuffer(ub_buf, (plane_rows * stride + 29u * stride + 192u) * sizeof(float)); LTf base =
        ub_buf.Get<float>(); ub_mat = base; ub_vecs = base[rows_per_lane * stride]; ub_refl =
        base[2u * rows_per_lane * stride]; LTf scratch_base = base[plane_rows * stride]; ub_row = scratch_base;
        ub_v = scratch_base[stride]; ub_w = scratch_base[2u * stride]; ub_cur_row = scratch_base[3u * stride];
        ub_prod = scratch_base[5u * stride]; ub_tmp = scratch_base[6u * stride]; ub_red = scratch_base[7u * stride];
        ub_diag = scratch_base[8u * stride]; ub_off = scratch_base[9u * stride]; ub_tau = scratch_base[10u * stride];
        ub_gather = scratch_base[gather_idx * stride]; ub_eig = scratch_base[eig_idx * stride]; ub_comm =
        scratch_base[comm_idx * stride]; LTf mask_base = scratch_base[29u * stride]; ub_mask_ge = mask_base;
        ub_mask_gt = mask_base[64]; ub_mask_eq = mask_base[128]; InitMaskTensors();
    }
    __aicore__ inline void InitBuffersLargeSub200(TPipe *pipe) {
        u32 refl_rows = (49152u - 1280u) / stride - n - rows_per_lane - 8u; vecs_offset = (n + 4u +
        refl_rows) * stride; u32 total_floats = (n == 200u) ? 49152u : ((vecs_offset + (rows_per_lane +
        4u) * stride) + 1280u); pipe->InitBuffer(ub_buf, total_floats * sizeof(float)); LTf base =
        ub_buf.Get<float>(); ub_mat = base; ub_w = base[n * stride]; ub_diag = base[(n + 1u) * stride]; ub_off =
        base[(n + 2u) * stride]; ub_tau = base[(n + 3u) * stride]; ub_refl = base[(n + 4u) * stride]; ub_vecs =
        base[vecs_offset]; LTf s1d = base[(n == 200u) ? 47072u : (vecs_offset + rows_per_lane * stride)]; ub_prod =
        s1d; ub_tmp = s1d[stride]; ub_gather = s1d[2u * stride]; ub_row = ub_gather[1024]; ub_v = ub_row[stride];
        ub_eig = ub_v; ub_red = ub_v[stride]; LTf mask_base = ub_red[64]; ub_mask_ge = mask_base; ub_mask_gt =
        mask_base[64]; ub_mask_eq = mask_base[128];
    }
    __aicore__ inline void InitBuffersLarge256(TPipe *pipe, LTf sharedBase) {
        num_zones = (n + 31u) >> 5u; u32 cur_z_off = 0u;
        for (u32 z = 0u; z < num_zones; ++z) {
            zone_offset[z] = cur_z_off; cur_z_off += 32u * (stride - (z << 5u));
        }
        zone_offset[num_zones] = cur_z_off; vecs_offset = cur_z_off + 14u * stride; LTf base;
        if constexpr (K96_PASS) {
            base = sharedBase;
        } else {
            pipe->InitBuffer(ub_buf, 49152u * sizeof(float)); base = ub_buf.Get<float>();
        }
        ub_mat = base; ub_w = base[cur_z_off]; ub_diag = base[cur_z_off + stride]; ub_off = base[cur_z_off +
        2u * stride]; ub_tau = base[cur_z_off + 3u * stride]; ub_refl = base[cur_z_off + 4u * stride]; ub_vecs =
        base[vecs_offset]; LTf s1d = base[vecs_offset + rows_per_lane * stride]; ub_prod = s1d; ub_tmp = s1d[stride];
        ub_gather = s1d[2u * stride]; ub_row = ub_gather[512]; ub_v = ub_gather[1024]; ub_eig = ub_v; ub_red =
        ub_v[stride]; LTf mask_base = base[48960]; ub_mask_gt = mask_base; ub_mask_ge = mask_base[64]; ub_mask_eq =
        K96_PASS ? mask_base : mask_base[128];
    }
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR eigenvalues, GM_ADDR eigenvectors, GM_ADDR workspace,
    const LinalgEighTilingData &t, TPipe *pipe, LTf sharedBase = LTf()) {
        n = t.n; stride = t.stride; uplo = t.uplo; batch_size = t.batch_size; num_groups = t.num_groups;
        group_lanes = t.group_lanes; rows_per_lane = t.rows_per_lane; vec_rows_per_lane = t.vec_rows_per_lane;
        solve_mode = t.solve_mode; InitVecRange(GetBlockIdx()); InitCoopRange();
        gm_A.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A));
        gm_eigvals.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eigenvalues));
        gm_eigvecs.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eigenvectors));
        if constexpr (K96_PASS) {
            gm_ckpt.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace), 4u * 65536u);
        }
        if constexpr (SOLVE_MODE == 1u) {
            pipe->InitBuffer(ub_buf, (2u * rows_per_lane * stride + 18u * stride) * sizeof(float)); LTf base =
            ub_buf.Get<float>(); ub_mat = base; ub_vecs = base[rows_per_lane * stride]; LTf scratch_base =
            base[2u * rows_per_lane * stride]; ub_row = scratch_base; ub_gather = scratch_base[2u * stride];
        } else if constexpr (SOLVE_MODE == 2u) {
            InitScratchPlane(pipe, 3u * rows_per_lane, 13u, 11u, 12u);
        } else if constexpr (SOLVE_MODE == 3u) {
            (n <= 200u) ? (void)InitBuffersLargeSub200(pipe) : (void)InitBuffersLarge256(pipe, sharedBase);
            InitMaskTensors();
        } else {
            gm_ws.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace)); u64 group_ws_span =
            2u * u64(n) * stride + 7u * stride + 8u * group_lanes + 64u; ws_refl_offset =
            u64(group_id) * group_ws_span; ws_w_offset = ws_refl_offset + u64(n) * stride; ws_scale_offset =
            ws_w_offset + 6u * stride; ws_tail_offset = ws_scale_offset + 8u * group_lanes + stride + 64u;
            InitScratchPlane(pipe, 2u * rows_per_lane, 11u, 27u, 28u);
        }
    }
    __aicore__ inline float VecDotSub(LTf x, LTf y, u32 c_first) {
        u32 len = n - c_first; PipeV(); Mul(ub_prod[c_first], x[c_first], y[c_first], len); PipeV(); (len <= 64u) ?
        (void)WholeReduceSum(ub_red, ub_prod[c_first], len, 1, 1, 1, 8) : (void)ReduceSum(ub_red, ub_prod[c_first],
        ub_tmp, len); PipeAll(); return ub_red.GetValue(0);
    }
    __aicore__ inline void ReduceRows64ByBlock2Stage(LTf dst, LTf sc, u32 rows) {
        BlockReduceSum(sc, sc, i32(rows), 64, 1, 1, 8); PipeV(); u32 full8 = rows >> 3u, rem8 = rows & 7u;
        if (full8 > 0u) {
            BlockReduceSum(dst, sc, i32(full8), 64, 1, 1, 8);
        }
        if (rem8 > 0u) {
            BlockReduceSum(dst[full8 << 3u], sc[full8 << 6u], 1, i32(rem8 << 3u), 1, 1, 8);
        }
    }
    template <bool NEG_W>
    __aicore__ inline void VecDotAxpyMaskBrcbImpl(LTf w, LTf v, u32 c_first, u32 rem, u32 zero_base, LTf brcb_v,
    u32 brcb_start, u8 brcb_rep) {
        u32 len = n - c_first; LTf brcb_dst = (stride < 64u) ? ub_cur_row : ub_tmp; Mul(ub_prod[c_first],
        ub_prod[c_first], w[c_first], len);
        if (brcb_rep > 0u) {
            Brcb(brcb_v, v[brcb_start], brcb_rep, {1, 8});
        }
        if constexpr (!NEG_W) {
            if (rem == 0u && c_first > zero_base) {
                Duplicate(v[c_first - 8u], 0.0f, 8u); Duplicate(w[c_first - 8u], 0.0f, 8u);
            }
        }
        PipeV();
        if (rem != 0u) {
            Mul(w[c_first], w[c_first], ub_mask_ge[rem * 8u], 8u);
        }
        (len <= 64u) ? (void)WholeReduceSum(ub_red, ub_prod[c_first], len, 1, 1, 1, 8) : (void)ReduceSum(ub_red,
        ub_prod[c_first], ub_tmp, len); PipeV();
        if constexpr (NEG_W) {
            Muls(w[c_first], w[c_first], -1.0f, len);
        }
        Brcb(brcb_dst, ub_red, 1u, {1, 8}); PipeV(); u32 full_s = len >> 6u, tail_s = len & 63u;
        if (full_s > 0u) {
            MulAddDst(w[c_first], v[c_first], brcb_dst, 64u, u8(full_s), BRP{1, 1, 0, 8, 8, 0});
        }
        if (tail_s > 0u) {
            MulAddDst(w[c_first + (full_s << 6u)], v[c_first + (full_s << 6u)], brcb_dst, tail_s, 1u, BRP{1, 1, 0, 8,
            8, 0});
        }
        PipeV();
    }
    __aicore__ inline void VecDotAxpyMaskBrcbSub(LTf w, LTf v, u32 c_first, u32 rem, u32 zero_base, LTf brcb_v,
    u32 brcb_start, u8 brcb_rep) {
        VecDotAxpyMaskBrcbImpl<false>(w, v, c_first, rem, zero_base, brcb_v, brcb_start, brcb_rep);
    }
    __aicore__ inline void VecDotAxpyNegWMaskBrcbSub(LTf w, LTf v, u32 c_first, u32 rem, u32 zero_base, LTf brcb_v,
    u32 brcb_start, u8 brcb_rep) {
        VecDotAxpyMaskBrcbImpl<true>(w, v, c_first, rem, zero_base, brcb_v, brcb_start, brcb_rep);
    }
    __aicore__ inline void VecDotAxpySub(LTf w, LTf v, float scale, u32 c_first) {
        u32 len = n - c_first; PipeV(); Mul(ub_prod[c_first], v[c_first], w[c_first], len); PipeV(); (len <= 64u) ?
        (void)WholeReduceSum(ub_red, ub_prod[c_first], len, 1, 1, 1, 8) : (void)ReduceSum(ub_red, ub_prod[c_first],
        ub_tmp, len); PipeV(); Muls(ub_red, ub_red, scale, 8u); PipeV(); LTf brcb_dst = (stride < 64u) ? ub_gather :
        ub_tmp; Brcb(brcb_dst, ub_red, 1u, {1, 8}); PipeV();
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u; Mul(ub_prod[c], v[c], brcb_dst, width, 1u, BRP{1, 1, 0, 8, 8,
            0});
        }
        PipeV(); Add(w[c_first], w[c_first], ub_prod[c_first], len); PipeV();
    }
    __aicore__ inline void AccumulateStripSums(LTf dst, LTf gather_base, u32 strip_idx, u32 aligned_cnt,
    u32 act_rows) {
        if (strip_idx == 4u) {
            Add(dst, dst, gather_base, act_rows); Add(gather_base[aligned_cnt], gather_base[aligned_cnt],
            gather_base[2u * aligned_cnt], act_rows); PipeV(); Add(dst, dst, gather_base[aligned_cnt], act_rows);
            PipeV();
        } else {
            for (u32 s = 1u; s < strip_idx; ++s) {
                Add(dst, dst, gather_base[(s - 1u) * aligned_cnt], act_rows); PipeV();
            }
        }
    }
    __aicore__ inline void MatVecRowsSub(LTf matrix_sub, LTf x, LTf sums_sub, LTf scratch, u32 act_rows, u32 c_first,
    u32 row_stride = 0u, u32 col_off = 0u) {
        if (act_rows == 0u || c_first >= n) {
            return;
        }
        u32 r_stride = (row_stride != 0u) ? row_stride : stride, raw_sc = ((n - c_first) + 7u) & ~7u, sc_stride =
        (raw_sc >= 64u) ? raw_sc : ((r_stride < 64u) ? r_stride : 64u), aligned_cnt = (act_rows + 7u) & ~7u;
        u8 rep_stride = u8(r_stride / 8u), sc_rep_stride = u8(sc_stride / 8u);
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u; Mul(scratch[c - c_first], matrix_sub[c - col_off], x[c],
            width, act_rows, BRP{1, 1, 1, sc_rep_stride, rep_stride, 0});
        }
        PipeV(); u32 strip_idx = 0u;
        for (u32 c = c_first; c < n; c += 64u, ++strip_idx) {
            u32 width = (n - c < 64u) ? (n - c) : 64u; LTf out_s = (strip_idx == 0u) ? sums_sub :
            ub_gather[(strip_idx - 1u) * aligned_cnt]; WholeReduceSum(out_s, scratch[c - c_first], width, act_rows,
            1, 1, sc_rep_stride);
        }
        PipeV(); AccumulateStripSums(sums_sub, ub_gather, strip_idx, aligned_cnt, act_rows);
    }
    __aicore__ inline void SubTwoBrcbStrips(LTf matrix_sub, LTf v_km1, LTf prod_v, LTf scratch0, LTf scratch1,
    LTf brcb0, LTf brcb1, u32 act_rows, u32 c_first, u8 rep_stride, u8 sc_rep_stride) {
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u; Mul(scratch1[c - c_first], v_km1[c], brcb1, width, act_rows,
            BRP{1, 1, 0, sc_rep_stride, 0, 1});
        }
        PipeV();
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u, sc = c - c_first; Sub(matrix_sub[c], matrix_sub[c],
            scratch1[sc], width, act_rows, BRP{1, 1, 1, rep_stride, rep_stride, sc_rep_stride}); Mul(scratch0[sc],
            prod_v[c], brcb0, width, act_rows, BRP{1, 1, 0, sc_rep_stride, 0, 1});
        }
        PipeV();
        for (u32 c = c_first; c < n; c += 64u) {
            u32 width = (n - c < 64u) ? (n - c) : 64u; Sub(matrix_sub[c], matrix_sub[c], scratch0[c - c_first],
            width, act_rows, BRP{1, 1, 1, rep_stride, rep_stride, sc_rep_stride});
        }
        PipeV();
    }
    __aicore__ inline void BacktransformPairShortAligned(LTf matrix_sub, LTf v_k, LTf v_km1, LTf scratch1, LTf brcb0,
    LTf brcb1, u32 act_rows, u32 c_first, u32 len, u8 rep_stride) {
        Muls(scratch1, v_km1[c_first], -1.0f, len); Muls(ub_prod[c_first], v_k[c_first], -1.0f, len); PipeV();
        MulAddDst(ub_prod[c_first], v_km1[c_first], ub_tmp, len, 1u, BRP{1, 1, 0, 8, 8, 0});
        MulAddDst(matrix_sub[c_first], scratch1, brcb1, len, act_rows, BRP{1, 1, 0, rep_stride, 0, 1}); PipeV();
        MulAddDst(matrix_sub[c_first], ub_prod[c_first], brcb0, len, act_rows, BRP{1, 1, 0, rep_stride, 0, 1});
        PipeV();
    }
    __aicore__ inline void BacktransformPairShortUnaligned(LTf matrix_sub, LTf v_k, LTf v_km1, LTf scratch0,
    LTf scratch1, LTf brcb0, LTf brcb1, u32 act_rows, u32 c_first, u32 len, u32 sc_stride, u8 rep_stride,
    u8 sc_rep_stride) {
        PipeV(); Mul(ub_prod[c_first], v_km1[c_first], ub_tmp, len, 1u, BRP{1, 1, 0, 8, 8, 0}); Mul(scratch1[0],
        v_km1[c_first], brcb1, len, act_rows, BRP{1, 1, 0, sc_rep_stride, 0, 1}); PipeV(); Sub(ub_prod[c_first],
        v_k[c_first], ub_prod[c_first], len); (c_first == 0u && n == stride && sc_stride == stride) ?
        (void)Sub(matrix_sub, matrix_sub, scratch1, act_rows * stride) : (void)Sub(matrix_sub[c_first],
        matrix_sub[c_first], scratch1[0], len, act_rows, BRP{1, 1, 1, rep_stride, rep_stride, sc_rep_stride});
        PipeV(); Mul(scratch0[0], ub_prod[c_first], brcb0, len, act_rows, BRP{1, 1, 0, sc_rep_stride, 0, 1});
        PipeV(); (c_first == 0u && n == stride && sc_stride == stride) ? (void)Sub(matrix_sub, matrix_sub, scratch0,
        act_rows * stride) : (void)Sub(matrix_sub[c_first], matrix_sub[c_first], scratch0[0], len, act_rows, BRP{1,
        1, 1, rep_stride, rep_stride, sc_rep_stride}); PipeV();
    }
    __aicore__ inline void ReducePairShortAndBrcb(LTf scratch0, LTf brcb0, u32 c_first, u32 len, u32 dual_rows,
    u8 sc_rep_stride, u8 brcb_dual_rep) {
        WholeReduceSum(ub_row, scratch0, len, dual_rows, 1, 1, sc_rep_stride); WholeReduceSum(ub_red,
        ub_prod[c_first], len, 1, 1, 1, 8); PipeV(); Brcb(ub_tmp, ub_red, 1u, {1, 8}); Brcb(brcb0, ub_row,
        brcb_dual_rep, {1, 8});
    }
    __aicore__ inline void ReducePairBlockAndOrth(LTf v_k, LTf v_km1, LTf scratch0, LTf scratch1, LTf brcb0,
    u32 c_first, u32 len, u32 dual_rows, u8 sc_rep_stride, u8 brcb_dual_rep) {
        u32 full_s = len >> 6u, tail_s = len & 63u; BlockReduceSum(ub_tmp, ub_prod[c_first], i32(full_s), 64, 1, 1,
        8);
        if (tail_s > 0u) {
            WholeReduceSum(ub_tmp[full_s * 8u], ub_prod[c_first + (full_s << 6u)], i32(tail_s), 1, 1, 1, 8);
        }
        PipeV(); WholeReduceSum(ub_row, scratch0, sc_rep_stride, dual_rows, 1, 8, sc_rep_stride);
        WholeReduceSum(ub_red, ub_tmp, i32(full_s * 8u + ((tail_s > 0u) ? 1u : 0u)), 1, 1, 1, 8);
        Muls(ub_prod[c_first], v_k[c_first], -1.0f, len); PipeV(); Brcb(ub_tmp, ub_red, 1u, {1, 8}); Brcb(brcb0,
        ub_row, brcb_dual_rep, {1, 8}); Muls(scratch1, v_km1[c_first], -1.0f, len); PipeV();
        if (full_s > 0u) {
            MulAddDst(ub_prod[c_first], v_km1[c_first], ub_tmp, 64u, u8(full_s), BRP{1, 1, 0, 8, 8, 0});
        }
        if (tail_s > 0u) {
            MulAddDst(ub_prod[c_first + (full_s << 6u)], v_km1[c_first + (full_s << 6u)], ub_tmp, tail_s, 1u, BRP{1,
            1, 0, 8, 8, 0});
        }
    }
    __aicore__ inline void OrthPairVectorInProd(LTf v_k, LTf v_km1, LTf brcb0, u32 c_first, u32 len,
    u8 brcb_dual_rep) {
        u32 full_s = len >> 6u, tail_s = len & 63u; Brcb(brcb0, ub_row, brcb_dual_rep, {1, 8});
        if (full_s > 0u) {
            Mul(ub_prod[c_first], v_km1[c_first], ub_tmp, 64u, u8(full_s), BRP{1, 1, 0, 8, 8, 0});
        }
        if (tail_s > 0u) {
            Mul(ub_prod[c_first + (full_s << 6u)], v_km1[c_first + (full_s << 6u)], ub_tmp, tail_s, 1u, BRP{1, 1, 0,
            8, 8, 0});
        }
        PipeV(); Sub(ub_prod[c_first], v_k[c_first], ub_prod[c_first], len);
    }
};
} // namespace AscendC
#endif
