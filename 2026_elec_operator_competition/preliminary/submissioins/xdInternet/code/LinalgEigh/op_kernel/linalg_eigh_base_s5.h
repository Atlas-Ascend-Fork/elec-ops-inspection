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
#ifndef LINALG_EIGH_BASE_S5_H
#define LINALG_EIGH_BASE_S5_H
#include "linalg_eigh_base_s4.h"
namespace AscendC {
template <typename DT_A, u32 SOLVE_MODE = 0u, bool FAST200 = false, bool K96_PASS = false>
class KernelEighBase : public KernelEighBaseS4<DT_A, SOLVE_MODE, FAST200, K96_PASS> {
public:
    using Base = KernelEighBaseS4<DT_A, SOLVE_MODE, FAST200, K96_PASS>;
    using Base::row_start, Base::row_count, Base::ws_refl_offset, Base::gm_ws, Base::ub_mat, Base::ub_diag;
    using Base::ub_off, Base::ub_tau, Base::ub_eig, Base::ub_gather, Base::ScalarAbs, Base::ScalarSqrt;
    using Base::ClampPivot, Base::BacktransformCoopTileStep, Base::n, Base::stride, Base::rows_per_lane;
    __aicore__ inline KernelEighBase() = default;
    __aicore__ inline void BacktransformCoop(float *work) {
        if (row_count == 0u || n < 3u) {
            return;
        }
        for (u32 k = 0u; k + 2u < n; ++k) {
            work[k] = ub_tau.GetValue(k);
        }
        bool can_pair = (2u * row_count + 16u <= rows_per_lane) && (row_count <= 64u); u32 scratch_rows = can_pair ?
        (2u * row_count) : row_count; i32 avail_tile = (rows_per_lane > scratch_rows) ? i32(rows_per_lane -
        scratch_rows) : 0, tile_step = (avail_tile >= 64) ? 64 : ((avail_tile >= 48) ? 48 : ((avail_tile >= 28) ?
        28 : ((avail_tile >= 16) ? 16 : 12))); LTf ub_refl_tile = (avail_tile >= 16) ?
        ub_mat[scratch_rows * stride] : ub_gather[4u * stride], scratch0 = ub_mat, scratch1 =
        ub_mat[row_count * stride];
        for (i32 k_end = i32(n) - 3; k_end >= 0; k_end -= tile_step) {
            i32 k_start = (k_end >= tile_step - 1) ? (k_end - (tile_step - 1)) : 0; u32 blk_cnt = u32(k_end -
            k_start + 1); PipeAll(); DataCopy(ub_refl_tile, gm_ws[ws_refl_offset + u64(k_start) * stride],
            blk_cnt * stride); PipeAll(); BacktransformCoopTileStep(k_start, k_end, can_pair, work, ub_refl_tile,
            scratch0, scratch1);
        }
    }
    __aicore__ inline void FactorizeShiftedTridiag1Way(const float *dd, const float *ee, float lambda, float *piv,
    float *mult, float *u1, u8 *swap_out) const {
        float cur_p = dd[0] - lambda, cur_u = ee[0];
        for (u32 i = 0u; i + 1u < n; ++i) {
            float m_i = ee[i], nd = dd[i + 1u] - lambda, ne = ee[i + 1u]; bool sw =
            (ScalarAbs(cur_p) < ScalarAbs(m_i)); float den = sw ? m_i : ClampPivot(cur_p), inv_p = 1.0f / den, f =
            (sw ? cur_p : m_i) * inv_p; piv[i] = inv_p; mult[i] = f; u1[i] = sw ? nd : cur_u; swap_out[i] = sw ? 1u :
            0u; cur_p = (sw ? cur_u : nd) - f * (sw ? nd : cur_u); cur_u = sw ? (-f * ne) : ne;
        }
        piv[n - 1u] = 1.0f / ClampPivot(cur_p);
    }
    __aicore__ inline float SolveShiftedTridiag1Way(float *qv, float in_scale, const float *ee, const float *piv,
    const float *mult, const float *u1, const u8 *swap_in) const {
        float cq = qv[0] * in_scale;
        for (u32 i = 0u; i + 1u < n; ++i) {
            u8 sw = swap_in[i]; float nq = qv[i + 1u] * in_scale, qi = (sw != 0u) ? nq : cq, base_q = (sw != 0u) ?
            cq : nq; qv[i] = qi; cq = base_q - mult[i] * qi;
        }
        float q_n1 = cq * piv[n - 1u]; qv[n - 1u] = q_n1; float nsq = q_n1 * q_n1, q_n2 = 0.0f;
        if (n > 1u) {
            u32 nm2 = n - 2u; q_n2 = q_n1; q_n1 = (qv[nm2] - u1[nm2] * q_n2) * piv[nm2]; qv[nm2] = q_n1;
            nsq += q_n1 * q_n1;
        }
        for (i32 i = i32(n) - 3; i >= 0; --i) {
            u32 ui = u32(i); float u2_val = (swap_in[ui] != 0u) ? ee[ui + 1u] : 0.0f, q_cur = (qv[ui] -
            u1[ui] * q_n1 - u2_val * q_n2) * piv[ui]; qv[ui] = q_cur; q_n2 = q_n1; q_n1 = q_cur; nsq += q_cur * q_cur;
        }
        return nsq;
    }
    __aicore__ inline void SturmStep2(SturmState4 &st, float d_a, float e2_a, float d_b, float e2_b) const {
        float eb0 = e2_b * st.p0; st.pp0 = (d_a - st.m0) * st.p0 - e2_a * st.pp0; st.p0 = (d_b - st.m0) * st.pp0 -
        eb0; u32 sa0 = (st.pp0 <= 0.0f) ? 1u : 0u; st.c0 += (sa0 ^ st.s0); u32 sb0 = (st.p0 <= 0.0f) ? 1u : 0u;
        st.c0 += (sb0 ^ sa0); st.s0 = sb0; float eb1 = e2_b * st.p1; st.pp1 = (d_a - st.m1) * st.p1 - e2_a * st.pp1;
        st.p1 = (d_b - st.m1) * st.pp1 - eb1; u32 sa1 = (st.pp1 <= 0.0f) ? 1u : 0u; st.c1 += (sa1 ^ st.s1); u32 sb1 =
        (st.p1 <= 0.0f) ? 1u : 0u; st.c1 += (sb1 ^ sa1); st.s1 = sb1; float eb2 = e2_b * st.p2; st.pp2 = (d_a -
        st.m2) * st.p2 - e2_a * st.pp2; st.p2 = (d_b - st.m2) * st.pp2 - eb2; u32 sa2 = (st.pp2 <= 0.0f) ? 1u : 0u;
        st.c2 += (sa2 ^ st.s2); u32 sb2 = (st.p2 <= 0.0f) ? 1u : 0u; st.c2 += (sb2 ^ sa2); st.s2 = sb2; float eb3 =
        e2_b * st.p3; st.pp3 = (d_a - st.m3) * st.p3 - e2_a * st.pp3; st.p3 = (d_b - st.m3) * st.pp3 - eb3; u32 sa3 =
        (st.pp3 <= 0.0f) ? 1u : 0u; st.c3 += (sa3 ^ st.s3); u32 sb3 = (st.p3 <= 0.0f) ? 1u : 0u;
        st.c3 += (sb3 ^ sa3); st.s3 = sb3;
    }
    __aicore__ inline void PolySeqStep2(PolyState4 &st, float d_a, float e2_a, float d_b, float e2_b) const {
        float eb0 = e2_b * st.p0; st.pp0 = (d_a - st.m0) * st.p0 - e2_a * st.pp0; st.p0 = (d_b - st.m0) * st.pp0 -
        eb0; float eb1 = e2_b * st.p1; st.pp1 = (d_a - st.m1) * st.p1 - e2_a * st.pp1; st.p1 = (d_b -
        st.m1) * st.pp1 - eb1; float eb2 = e2_b * st.p2; st.pp2 = (d_a - st.m2) * st.p2 - e2_a * st.pp2; st.p2 =
        (d_b - st.m2) * st.pp2 - eb2; float eb3 = e2_b * st.p3; st.pp3 = (d_a - st.m3) * st.p3 - e2_a * st.pp3;
        st.p3 = (d_b - st.m3) * st.pp3 - eb3;
    }
    __aicore__ inline void RescaleSturmLane(float &p, float &pp, u32 s, u32 &c) const {
        float ap = ScalarAbs(p);
        if (ap > 1e12f) {
            p *= 5.4210109e-20f; pp *= 5.4210109e-20f; c += (1u << 16);
        } else if (ap < 1e-12f) {
            p = (p == 0.0f) ? ((s != 0u) ? -1e-15f : 1e-15f) : (p * 1.8446744e19f); pp *= 1.8446744e19f;
            c -= (1u << 16);
        }
    }
    __aicore__ inline void RescaleSturm4(SturmState4 &st) const {
        float ap0 = ScalarAbs(st.p0), ap1 = ScalarAbs(st.p1), ap2 = ScalarAbs(st.p2), ap3 = ScalarAbs(st.p3), mx =
        __builtin_fmaxf(__builtin_fmaxf(ap0, ap1), __builtin_fmaxf(ap2, ap3)), mn =
        __builtin_fminf(__builtin_fminf(ap0, ap1), __builtin_fminf(ap2, ap3));
        if (__builtin_expect((mx > 1e12f) | (mn < 1e-12f), 0)) {
            RescaleSturmLane(st.p0, st.pp0, st.s0, st.c0); RescaleSturmLane(st.p1, st.pp1, st.s1, st.c1);
            RescaleSturmLane(st.p2, st.pp2, st.s2, st.c2); RescaleSturmLane(st.p3, st.pp3, st.s3, st.c3);
        }
    }
    __aicore__ inline void RescalePolyLane(float &p, float &pp, i32 &exp_v) const {
        float ap = ScalarAbs(p);
        if (ap > 1e12f) {
            p *= 5.4210109e-20f; pp *= 5.4210109e-20f; ++exp_v;
        } else if (ap < 1e-12f) {
            p = (p == 0.0f) ? ((pp <= 0.0f) ? 1e-15f : -1e-15f) : (p * 1.8446744e19f); pp *= 1.8446744e19f; --exp_v;
        }
    }
    __aicore__ inline void RescalePoly4(PolyState4 &st) const {
        float ap0 = ScalarAbs(st.p0), ap1 = ScalarAbs(st.p1), ap2 = ScalarAbs(st.p2), ap3 = ScalarAbs(st.p3), mx =
        __builtin_fmaxf(__builtin_fmaxf(ap0, ap1), __builtin_fmaxf(ap2, ap3)), mn =
        __builtin_fminf(__builtin_fminf(ap0, ap1), __builtin_fminf(ap2, ap3));
        if (__builtin_expect((mx > 1e12f) | (mn < 1e-12f), 0)) {
            RescalePolyLane(st.p0, st.pp0, st.exp0); RescalePolyLane(st.p1, st.pp1, st.exp1); RescalePolyLane(st.p2,
            st.pp2, st.exp2); RescalePolyLane(st.p3, st.pp3, st.exp3);
        }
    }
    __aicore__ inline void InitP0Lane(float d0, float m, float &p, float &pp) const {
        pp = 1.0f; p = d0 - m; p = (ScalarAbs(p) < 1e-20f) ? (-1e-20f) : p;
    }
    template <typename StateT>
    __aicore__ inline void InitP0State4(StateT &st, float d0, const float *b_mid) const {
        st.m0 = b_mid[0]; st.m1 = b_mid[1]; st.m2 = b_mid[2]; st.m3 = b_mid[3]; InitP0Lane(d0, st.m0, st.p0, st.pp0);
        InitP0Lane(d0, st.m1, st.p1, st.pp1); InitP0Lane(d0, st.m2, st.p2, st.pp2); InitP0Lane(d0, st.m3, st.p3,
        st.pp3);
    }
    __aicore__ inline void StepInitSturmLane(float d_1, float e2_1, float m, u32 exp_bias, float &p, float &pp,
    u32 &s, u32 &c, bool do_step1) const {
        s = (p < 0.0f) ? 1u : 0u; c = exp_bias | s;
        if (do_step1) {
            float pn = (d_1 - m) * p - e2_1 * pp; pp = p; p = pn; u32 sn = (pn <= 0.0f) ? 1u : 0u; c += (sn ^ s); s =
            sn;
        }
    }
    __aicore__ inline void InitSturmState4(SturmState4 &st, float d0, const float *de_pairs, const float *b_mid,
    u32 &i0, bool force_even = false, u32 exp_bias = (256u << 16)) const {
        InitP0State4(st, d0, b_mid); bool do_step1 = force_even || ((n & 1u) == 0u); float d_1 = do_step1 ?
        de_pairs[0] : 0.0f, e2_1 = do_step1 ? de_pairs[1] : 0.0f; StepInitSturmLane(d_1, e2_1, st.m0, exp_bias,
        st.p0, st.pp0, st.s0, st.c0, do_step1); StepInitSturmLane(d_1, e2_1, st.m1, exp_bias, st.p1, st.pp1, st.s1,
        st.c1, do_step1); StepInitSturmLane(d_1, e2_1, st.m2, exp_bias, st.p2, st.pp2, st.s2, st.c2, do_step1);
        StepInitSturmLane(d_1, e2_1, st.m3, exp_bias, st.p3, st.pp3, st.s3, st.c3, do_step1); i0 = do_step1 ? 2u : 1u;
    }
    __aicore__ inline void InitPolyState4(PolyState4 &st, float d0, const float *de_pairs, const float *b_mid,
    u32 &i0, bool force_even = false) const {
        InitP0State4(st, d0, b_mid); st.exp0 = 0; st.exp1 = 0; st.exp2 = 0; st.exp3 = 0; i0 = 1u;
        if (force_even || ((n & 1u) == 0u)) {
            float d_1 = de_pairs[0], e2_1 = de_pairs[1], pn0 = (d_1 - st.m0) * st.p0 - e2_1 * st.pp0, pn1 = (d_1 -
            st.m1) * st.p1 - e2_1 * st.pp1, pn2 = (d_1 - st.m2) * st.p2 - e2_1 * st.pp2, pn3 = (d_1 -
            st.m3) * st.p3 - e2_1 * st.pp3; st.pp0 = st.p0; st.p0 = pn0; st.pp1 = st.p1; st.p1 = pn1; st.pp2 = st.p2;
            st.p2 = pn2; st.pp3 = st.p3; st.p3 = pn3; i0 = 2u;
        }
    }
    __aicore__ inline void WriteSturmOut4(const SturmState4 &st, u32 *b_neg, float *b_pval, i32 *b_pexp) const {
        b_neg[0] = st.c0 & 0xFFFFu; b_neg[1] = st.c1 & 0xFFFFu; b_neg[2] = st.c2 & 0xFFFFu; b_neg[3] =
        st.c3 & 0xFFFFu; b_pval[0] = ScalarAbs(st.p0); b_pval[1] = ScalarAbs(st.p1); b_pval[2] = ScalarAbs(st.p2);
        b_pval[3] = ScalarAbs(st.p3);
        if (b_pexp != nullptr) {
            b_pexp[0] = i32(st.c0 >> 16) - 256; b_pexp[1] = i32(st.c1 >> 16) - 256; b_pexp[2] = i32(st.c2 >> 16) -
            256; b_pexp[3] = i32(st.c3 >> 16) - 256;
        }
    }
    __aicore__ inline void WritePolyOut4(const PolyState4 &st, u32 *b_sign_neg, float *b_pval, i32 *b_pexp) const {
        b_sign_neg[0] = (st.p0 <= 0.0f) ? 1u : 0u; b_sign_neg[1] = (st.p1 <= 0.0f) ? 1u : 0u; b_sign_neg[2] =
        (st.p2 <= 0.0f) ? 1u : 0u; b_sign_neg[3] = (st.p3 <= 0.0f) ? 1u : 0u; b_pval[0] = ScalarAbs(st.p0);
        b_pval[1] = ScalarAbs(st.p1); b_pval[2] = ScalarAbs(st.p2); b_pval[3] = ScalarAbs(st.p3); b_pexp[0] =
        st.exp0; b_pexp[1] = st.exp1; b_pexp[2] = st.exp2; b_pexp[3] = st.exp3;
    }
    __aicore__ inline void EvalSturmSmall4(const float *d, const float *e2, const float *b_mid, u32 *b_neg,
    float *b_pval) const {
        SturmState4 st; u32 i = 1u; float de0[2] = {d[1], e2[0]}; InitSturmState4(st, d[0], de0, b_mid, i, false, 0u);
        for (; i + 3u < n; i += 4u) {
            SturmStep2(st, d[i], e2[i - 1u], d[i + 1u], e2[i]); SturmStep2(st, d[i + 2u], e2[i + 1u], d[i + 3u],
            e2[i + 2u]);
        }
        for (; i < n; i += 2u) {
            SturmStep2(st, d[i], e2[i - 1u], d[i + 1u], e2[i]);
        }
        WriteSturmOut4(st, b_neg, b_pval, nullptr);
    }
    __aicore__ __attribute__((noinline)) void EvalSturm4Fast(float d0, const float *de_pairs, const float *b_mid,
    u32 *b_neg, float *b_pval, i32 *b_pexp) const {
        SturmState4 st; u32 i0 = 1u; InitSturmState4(st, d0, de_pairs, b_mid, i0); const float *de = de_pairs + (i0 -
        1u) * 2u;
        for (; i0 + 12u <= n; i0 += 12u, de += 24u) {
            for (u32 s = 0u; s < 24u; s += 4u) {
                SturmStep2(st, de[s], de[s + 1u], de[s + 2u], de[s + 3u]);
            }
            RescaleSturm4(st);
        }
        if (i0 < n) {
            for (; i0 + 1u < n; i0 += 2u, de += 4u) {
                SturmStep2(st, de[0], de[1], de[2], de[3]);
            }
            if constexpr (!FAST200 && SOLVE_MODE != 3u) {
                RescaleSturm4(st);
            }
        }
        WriteSturmOut4(st, b_neg, b_pval, b_pexp);
    }
    template <bool TAIL_RESCALE>
    __aicore__ inline void EvalPoly4FastCore(float d0, const float *de_pairs, const float *b_mid, u32 *b_sign_neg,
    float *b_pval, i32 *b_pexp) const {
        PolyState4 st; u32 i0 = 1u; InitPolyState4(st, d0, de_pairs, b_mid, i0); const float *de = de_pairs + (i0 -
        1u) * 2u;
        for (; i0 + 12u <= n; i0 += 12u, de += 24u) {
            for (u32 s = 0u; s < 24u; s += 4u) {
                PolySeqStep2(st, de[s], de[s + 1u], de[s + 2u], de[s + 3u]);
            }
            RescalePoly4(st);
        }
        if (i0 < n) {
            for (; i0 + 1u < n; i0 += 2u, de += 4u) {
                PolySeqStep2(st, de[0], de[1], de[2], de[3]);
            }
            if constexpr (TAIL_RESCALE && !FAST200 && SOLVE_MODE != 3u) {
                RescalePoly4(st);
            }
        }
        WritePolyOut4(st, b_sign_neg, b_pval, b_pexp);
    }
    __aicore__ __attribute__((noinline)) void EvalPoly4Fast(float d0, const float *de_pairs, const float *b_mid,
    u32 *b_sign_neg, float *b_pval, i32 *b_pexp) const {
        EvalPoly4FastCore<true>(d0, de_pairs, b_mid, b_sign_neg, b_pval, b_pexp);
    }
    __aicore__ __attribute__((noinline)) void EvalPoly4SameScaleAs8Fast(float d0, const float *de_pairs,
    const float *b_mid, u32 *b_sign_neg, float *b_pval, i32 *b_pexp) const {
        EvalPoly4FastCore<false>(d0, de_pairs, b_mid, b_sign_neg, b_pval, b_pexp);
    }
    __aicore__ __attribute__((noinline)) void EvalSturm4Fast256(float d0, const float *de_pairs, const float *b_mid,
    u32 *b_neg, float *b_pval, i32 *b_pexp) const {
        SturmState4 st; u32 i0 = 1u; InitSturmState4(st, d0, de_pairs, b_mid, i0, true); const float *de = de_pairs +
        2u;
        for (i0 = 2u; i0 < 254u; i0 += 12u, de += 24u) {
            for (u32 s = 0u; s < 24u; s += 4u) {
                SturmStep2(st, de[s], de[s + 1u], de[s + 2u], de[s + 3u]);
            }
            RescaleSturm4(st);
        }
        SturmStep2(st, de[0], de[1], de[2], de[3]); WriteSturmOut4(st, b_neg, b_pval, b_pexp);
    }
    __aicore__ __attribute__((noinline)) void EvalPoly4Sequential256(float d0, const float *de_pairs,
    const float *b_mid, u32 *b_sign_neg, float *b_pval, i32 *b_pexp) const {
        PolyState4 st; u32 i0 = 1u; InitPolyState4(st, d0, de_pairs, b_mid, i0, true); const float *de = de_pairs +
        2u;
        for (i0 = 2u; i0 < 254u; i0 += 12u, de += 24u) {
            for (u32 s = 0u; s < 24u; s += 4u) {
                PolySeqStep2(st, de[s], de[s + 1u], de[s + 2u], de[s + 3u]);
            }
            RescalePoly4(st);
        }
        PolySeqStep2(st, de[0], de[1], de[2], de[3]); WritePolyOut4(st, b_sign_neg, b_pval, b_pexp);
    }
    __aicore__ inline void EvalSturm4(float d0, const float *de_pairs, const float *b_mid, u32 *b_neg, float *b_pval,
    i32 *b_pexp) const {
        if constexpr ((FAST200 && !K96_PASS) || SOLVE_MODE == 3u) {
            EvalSturm4Fast(d0, de_pairs, b_mid, b_neg, b_pval, b_pexp); return;
        }
        SturmState4 st; u32 i0 = 1u; InitSturmState4(st, d0, de_pairs, b_mid, i0);
        for (; i0 < n; i0 += 12u) {
            u32 i_lim = (i0 + 12u < n) ? (i0 + 12u) : n; const float *de = de_pairs + (i0 - 1u) * 2u;
            for (u32 i = i0; i + 1u < i_lim; i += 2u, de += 4u) {
                SturmStep2(st, de[0], de[1], de[2], de[3]);
            }
            RescaleSturm4(st);
        }
        WriteSturmOut4(st, b_neg, b_pval, b_pexp);
    }
    __aicore__ inline void EvalPoly4(float d0, const float *de_pairs, const float *b_mid, u32 *b_sign_neg,
    float *b_pval, i32 *b_pexp) const {
        if constexpr ((FAST200 && !K96_PASS) || SOLVE_MODE == 3u) {
            EvalPoly4Fast(d0, de_pairs, b_mid, b_sign_neg, b_pval, b_pexp); return;
        }
        PolyState4 st; u32 i0 = 1u; InitPolyState4(st, d0, de_pairs, b_mid, i0);
        for (; i0 < n; i0 += 12u) {
            u32 i_lim = (i0 + 12u < n) ? (i0 + 12u) : n; const float *de = de_pairs + (i0 - 1u) * 2u;
            for (u32 i = i0; i + 1u < i_lim; i += 2u, de += 4u) {
                PolySeqStep2(st, de[0], de[1], de[2], de[3]);
            }
            RescalePoly4(st);
        }
        WritePolyOut4(st, b_sign_neg, b_pval, b_pexp);
    }
    __aicore__ inline void EvalPoly4SameScaleAs8(float d0, const float *de_pairs, const float *b_mid,
    u32 *b_sign_neg, float *b_pval, i32 *b_pexp) const {
        if constexpr ((FAST200 && !K96_PASS) || SOLVE_MODE == 3u) {
            EvalPoly4SameScaleAs8Fast(d0, de_pairs, b_mid, b_sign_neg, b_pval, b_pexp); return;
        }
        EvalPoly4(d0, de_pairs, b_mid, b_sign_neg, b_pval, b_pexp);
    }
    __aicore__ inline void SortAscendingInPlace(float *arr, u32 len) const {
        for (u32 i = 1u; i < len; ++i) {
            float val = arr[i]; u32 j = i;
            while (j > 0u && arr[j - 1u] > val) {
                arr[j] = arr[j - 1u]; --j;
            }
            arr[j] = val;
        }
    }
    __aicore__ inline bool PWKStepSweep(float *dd, float *ee, u32 l, u32 m, bool check_bounds) const {
        float delta = 0.5f * (dd[l + 1u] - dd[l]), rad = ScalarSqrt(delta * delta + ee[l]), denom = delta +
        ((delta >= 0.0f) ? rad : -rad), sigma = (denom != 0.0f) ? (dd[l] - ee[l] / denom) : (dd[l] -
        ScalarSqrt(ee[l])), c = 1.0f, s = 0.0f, gamma = dd[m] - sigma, pp = gamma * gamma;
        for (i32 ii = i32(m) - 1; ii >= i32(l); --ii) {
            u32 i = u32(ii); float bb = ee[i], rr = pp + bb;
            if (check_bounds && (!(rr > 0.0f) || rr > 3e38f)) {
                return false;
            }
            if (i + 1u != m) {
                ee[i + 1u] = s * rr;
            }
            float old_c = c, old_gamma = gamma, alpha = dd[i];
            if (check_bounds && pp > 1e-28f && rr < 1e28f) {
                float inv_rp = 1.0f / (rr * pp), inv_rr = inv_rp * pp, inv_c = (inv_rp * rr) * rr; c = pp * inv_rr;
                s = bb * inv_rr; gamma = c * (alpha - sigma) - s * old_gamma; dd[i + 1u] = old_gamma + (alpha -
                gamma); pp = (gamma * gamma) * inv_c;
            } else {
                float inv_rr = 1.0f / rr; c = pp * inv_rr; s = bb * inv_rr; gamma = c * (alpha - sigma) -
                s * old_gamma; dd[i + 1u] = old_gamma + (alpha - gamma); pp = (c != 0.0f) ? ((gamma * gamma) / c) :
                (old_c * bb);
            }
        }
        ee[l] = s * pp; dd[l] = sigma + gamma; return true;
    }
    __aicore__ inline bool RunPWKLoop(float *dd, float *ee, bool check_bounds) const {
        u32 total_iters = 0u; const float eps_sq = 1.4210855e-14f;
        for (u32 l = 0u; l < n; ) {
            u32 m = l;
            for (; m + 1u < n; ++m) {
                if (ee[m] <= eps_sq * ScalarAbs(dd[m] * dd[m + 1u]) + 1e-38f) {
                    break;
                }
            }
            ee[m] = (m + 1u < n) ? (0.0f) : ee[m];
            if (m == l) {
                ++l; continue;
            }
            if (m == l + 1u) {
                float mid = 0.5f * (dd[l] + dd[l + 1u]), hdiff = 0.5f * (dd[l] - dd[l + 1u]), rad =
                ScalarSqrt(hdiff * hdiff + ee[l]); dd[l] = mid - rad; dd[l + 1u] = mid + rad; ee[l] = 0.0f; l += 2u;
                continue;
            }
            if (++total_iters > 30u * n) {
                if (check_bounds) {
                    return false;
                }
                break;
            }
            if (!PWKStepSweep(dd, ee, l, m, check_bounds)) {
                return false;
            }
        }
        SortAscendingInPlace(dd, n); return true;
    }
    __aicore__ inline bool SolveTridiagPWK(float *work) {
        float *dd = (SOLVE_MODE == 2u) ? (work + 1024) : work; float *ee = (SOLVE_MODE == 2u) ? (work + 1536) :
        (work + 512);
        for (u32 i = 0u; i < n; ++i) {
            dd[i] = (SOLVE_MODE == 2u) ? work[i] : ub_diag.GetValue(i); float e_val = (SOLVE_MODE == 2u) ?
            work[256u + i] : ub_off.GetValue(i); ee[i] = e_val * e_val;
        }
        if (!RunPWKLoop(dd, ee, true)) {
            return false;
        }
        for (u32 i = 0u; i < n; ++i) {
            if (!(dd[i] >= -3e38f && dd[i] <= 3e38f)) {
                return false;
            }
        }
        for (u32 r = 0u; r < row_count; ++r) {
            ub_eig.SetValue(row_start + r, dd[row_start + r]);
        }
        return true;
    }
};
} // namespace AscendC
#endif
