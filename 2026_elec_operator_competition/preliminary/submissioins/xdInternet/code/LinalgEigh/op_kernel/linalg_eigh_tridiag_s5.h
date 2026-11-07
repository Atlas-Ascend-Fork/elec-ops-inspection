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
#ifndef LINALG_EIGH_TRIDIAG_S5_H
#define LINALG_EIGH_TRIDIAG_S5_H
#include "linalg_eigh_tridiag_s4.h"
namespace AscendC {
template <typename DT_A, u32 SOLVE_MODE = 0u, bool FAST200 = false, bool K96_PASS = false>
class KernelEighTridiag : public KernelEighTridiagS4<DT_A, SOLVE_MODE, FAST200, K96_PASS> {
public:
    using Base = KernelEighTridiagS4<DT_A, SOLVE_MODE, FAST200, K96_PASS>;
    using Base::EvalPoly4Sequential256, Base::EvalPoly4, Base::EvalPoly4SameScaleAs8, Base::SolveTridiagPWK;
    using Base::BindSturmWorkPtrs, Base::RefineNeighborBounds, Base::TryComputeIQIWeight;
    using Base::DampStagnatedSecantWeight, Base::ComputeRowSeparationGap, Base::ComputeAdaptiveRowTol;
    using Base::FindUpdatedPolyBracket, Base::InitNeighborBounds, Base::SolveTridiagSturm, Base::n, Base::stride;
    using Base::group_lanes, Base::row_start, Base::row_count, Base::ub_mat, Base::ub_vecs, Base::ub_diag;
    using Base::ub_off, Base::ub_eig, Base::ScalarAbs, Base::ScalarRsqrt, Base::ClampPivot, Base::MaxF;
    using Base::ClampF, Base::ComputeSecantWeight;
    __aicore__ inline KernelEighTridiag() = default;
    __aicore__ inline void SelectLargePoly1Point(SturmWorkPtrs &wp, u32 r, u32 lane_pos, u32 row_end, float lo,
    float hi, float width, float t_sec, const float *iso_clo, const float *iso_chi, const float *hist_x,
    const float *hist_p, const i32 *hist_e, float max_gap_tol, float *b8_mid) const {
        int8_t stg = wp.iso_stag[r]; float gp = ComputeRowSeparationGap(r, row_end, iso_clo, iso_chi, wp.left_nb_hi,
        wp.right_nb_lo);
        if (width > 0.55f * gp || stg >= 3 || stg <= -3) {
            t_sec = 0.5f; wp.iso_stag[r] = 0;
        } else {
            float iqi_t = t_sec; bool can_iqi = (FAST200 || n == 256u) && hist_e[r] != -30000 && (hist_x[r] < lo ||
            hist_x[r] > hi);
            if (can_iqi && TryComputeIQIWeight(lo, width, wp.iso_elo[r], wp.iso_ehi[r], hist_e[r], wp.iso_plo[r],
                wp.iso_phi[r], hist_p[r], hist_x[r], 0.015f, 0.985f, iqi_t)) {
                t_sec = iqi_t;
            } else {
                DampStagnatedSecantWeight(r, stg, width, max_gap_tol, 6.0f, t_sec);
            }
        }
        b8_mid[lane_pos] = lo + t_sec * width;
    }
    __aicore__ inline void SelectLargePolyTestPoints(SturmWorkPtrs &wp, u32 r, u32 kp, u32 lane_pos, u32 row_end,
    const float *iso_clo, const float *iso_chi, const float *iso_tsec, const float *hist_x, const float *hist_p,
    const i32 *hist_e, float max_gap_tol, float *b8_mid) const {
        float lo = wp.iso_lo[r], hi = wp.iso_hi[r], width = hi - lo, t_sec = iso_tsec[r]; bool has_sec =
        (t_sec >= 0.0f);
        if (kp == 1u && has_sec) {
            SelectLargePoly1Point(wp, r, lane_pos, row_end, lo, hi, width, t_sec, iso_clo, iso_chi, hist_x, hist_p,
            hist_e, max_gap_tol, b8_mid);
        } else if (kp == 2u && has_sec) {
            float tc = ClampF(t_sec, 0.12f, 0.88f); b8_mid[lane_pos] = lo + (tc - 0.08f) * width; b8_mid[lane_pos +
            1u] = lo + (tc + 0.08f) * width;
        } else if ((FAST200 || n == 256u) && kp == 4u && has_sec && wp.iso_stag[r] >= -1 && wp.iso_stag[r] <= 1) {
            float tc = ClampF(t_sec, 0.19f, 0.81f); b8_mid[lane_pos] = lo + (tc - 0.15f) * width; b8_mid[lane_pos +
            1u] = lo + (tc - 0.03f) * width; b8_mid[lane_pos + 2u] = lo + (tc + 0.03f) * width; b8_mid[lane_pos +
            3u] = lo + (tc + 0.15f) * width;
        } else {
            float step = width * ((kp == 4u) ? 0.2f : ((kp == 2u) ? (1.0f / 3.0f) : 0.5f));
            for (u32 k = 0u; k < kp; ++k) {
                b8_mid[lane_pos + k] = lo + step * u2f(k + 1u);
            }
        }
    }
    __aicore__ inline void RecordLargeIQIHistory(u32 r, u32 kp, u32 pos, u32 hit_k, float lo, float hi, float plo,
    float phi, i32 elo, i32 ehi, float new_lo, float new_hi, const float *b8_mid, const float *b8_pval,
    const i32 *b8_pexp, float *hist_x, float *hist_p, i32 *hist_e) const {
        if (!(FAST200 || n == 256u)) {
            return;
        }
        if (hit_k >= 2u || hit_k + 1u < kp) {
            u32 hk = (hit_k >= 2u) ? (hit_k - 2u) : (hit_k + 1u); hist_x[r] = b8_mid[pos + hk]; hist_p[r] =
            b8_pval[pos + hk]; hist_e[r] = b8_pexp[pos + hk];
        } else if (new_lo > lo || new_hi < hi) {
            bool use_lo = (new_lo > lo); hist_x[r] = use_lo ? lo : hi; hist_p[r] = use_lo ? plo : phi; hist_e[r] =
            use_lo ? elo : ehi;
        }
    }
    __aicore__ inline bool CheckLargeActiveConverged(const SturmWorkPtrs &wp, u32 r, bool has_sec, i32 hist_er,
    float new_lo, float new_hi, float next_mid, float t_sec, float width, float cur_tol) const {
        float eff_mult = ((FAST200 || n == 256u) && hist_er != -30000) ? 1.35f : 1.85f, eff_width = has_sec ?
        (eff_mult * t_sec * (1.0f - t_sec) * width) : width; int8_t stag_lim = (FAST200 || n == 256u) ? 2 : 1,
        next_stag = wp.iso_stag[r]; bool stag_ok = (next_stag >= -stag_lim && next_stag <= stag_lim &&
        width <= 7.0f * cur_tol && eff_width <= cur_tol + 1e-30f); return (wp.act_depth[r] >= 32u ||
        next_mid == new_lo || next_mid == new_hi || stag_ok || width <= cur_tol + 1e-30f);
    }
    __aicore__ inline void UpdateLargeActiveRow(SturmWorkPtrs &wp, u32 r, u32 kp, u32 pos, u32 row_end,
    float tol_single, float max_elem, float max_gap_tol, float *iso_clo, float *iso_chi, float *iso_tsec,
    float *hist_x, float *hist_p, i32 *hist_e, const float *b8_mid, const float *b8_pval, const i32 *b8_pexp,
    const u32 *b8_neg, u32 &a_tail, u32 &a_cnt) {
        float lo = wp.iso_lo[r], hi = wp.iso_hi[r], plo = wp.iso_plo[r], phi = wp.iso_phi[r]; i32 elo =
        wp.iso_elo[r], ehi = wp.iso_ehi[r]; u32 hit_k = kp; FindUpdatedPolyBracket(wp, r, kp, pos, b8_mid, b8_pval,
        b8_pexp, b8_neg, hit_k); float new_lo = wp.iso_lo[r], new_hi = wp.iso_hi[r], new_plo = wp.iso_plo[r],
        new_phi = wp.iso_phi[r]; i32 new_elo = wp.iso_elo[r], new_ehi = wp.iso_ehi[r]; RecordLargeIQIHistory(r, kp,
        pos, hit_k, lo, hi, plo, phi, elo, ehi, new_lo, new_hi, b8_mid, b8_pval, b8_pexp, hist_x, hist_p, hist_e);
        float next_mid = 0.5f * new_lo + 0.5f * new_hi, t_sec = 0.5f, width = new_hi - new_lo; bool has_sec =
        ComputeSecantWeight(new_plo, new_elo, new_phi, new_ehi, t_sec); float est = new_lo + t_sec * width;
        iso_tsec[r] = has_sec ? t_sec : -1.0f; iso_clo[r] = has_sec ? (0.5f * (new_lo + est)) : new_lo; iso_chi[r] =
        has_sec ? (0.5f * (est + new_hi)) : new_hi; float gap = ComputeRowSeparationGap(r, row_end, iso_clo, iso_chi,
        wp.left_nb_hi, wp.right_nb_lo), cur_tol = ComputeAdaptiveRowTol(tol_single, gap, next_mid, max_elem,
        FAST200 ? 0.0060f : 0.0068f, max_gap_tol);
        if (CheckLargeActiveConverged(wp, r, has_sec, hist_e[r], new_lo, new_hi, next_mid, t_sec, width, cur_tol)) {
            if (has_sec) {
                bool can_iqi = (FAST200 || n == 256u) && hist_e[r] != -30000 && (hist_x[r] < new_lo ||
                hist_x[r] > new_hi);
                if (can_iqi) {
                    TryComputeIQIWeight(new_lo, width, new_elo, new_ehi, hist_e[r], new_plo, new_phi, hist_p[r],
                    hist_x[r], 0.005f, 0.995f, t_sec);
                }
                next_mid = new_lo + t_sec * width;
            }
            ub_eig.SetValue(row_start + r, next_mid);
        } else {
            wp.act_q[a_tail] = r; a_tail = (a_tail + 1u) & 63u; ++a_cnt;
        }
    }
    __aicore__ inline void InitLargeActiveQueue(SturmWorkPtrs &wp, u32 row_end, float tol_single, float max_elem,
    float max_gap_tol, float *iso_clo, float *iso_chi, float *iso_tsec, i32 *hist_e, u32 &a_tail, u32 &a_cnt) {
        for (u32 r = 0u; r < row_count; ++r) {
            float lo = wp.iso_lo[r], hi = wp.iso_hi[r], t_s = 0.5f;
            if (ComputeSecantWeight(wp.iso_plo[r], wp.iso_elo[r], wp.iso_phi[r], wp.iso_ehi[r], t_s)) {
                iso_tsec[r] = t_s; float est = lo + t_s * (hi - lo); iso_clo[r] = 0.5f * (lo + est); iso_chi[r] =
                0.5f * (est + hi);
            } else {
                iso_tsec[r] = -1.0f; iso_clo[r] = lo; iso_chi[r] = hi;
            }
            hist_e[r] = (FAST200 || n == 256u) ? (-30000) : hist_e[r];
        }
        for (u32 r = 0u; r < row_count; ++r) {
            wp.act_depth[r] = 0u; float lo = wp.iso_lo[r], hi = wp.iso_hi[r], mid = 0.5f * lo + 0.5f * hi, gap =
            ComputeRowSeparationGap(r, row_end, iso_clo, iso_chi, wp.left_nb_hi, wp.right_nb_lo), cur_tol =
            ComputeAdaptiveRowTol(tol_single, gap, mid, max_elem, FAST200 ? 0.0060f : 0.0068f, max_gap_tol), t_sec =
            iso_tsec[r], width = hi - lo; bool has_sec = (t_sec >= 0.0f); float eff_width = has_sec ?
            (1.85f * t_sec * (1.0f - t_sec) * width) : width;
            if (mid == lo || mid == hi || ((FAST200 || n == 256u) && width <= 7.0f * cur_tol &&
                eff_width <= cur_tol + 1e-30f) || width <= cur_tol + 1e-30f) {
                ub_eig.SetValue(row_start + r, has_sec ? (lo + t_sec * width) : mid);
            } else {
                wp.act_q[a_tail] = r; a_tail = (a_tail + 1u) & 63u; ++a_cnt;
            }
        }
    }
    __aicore__ inline void EvalLargePolyBatch(const SturmWorkPtrs &wp, bool use_poly8, const float *b8_mid,
    u32 *b8_neg, float *b8_pval, i32 *b8_pexp) const {
        if constexpr (SOLVE_MODE == 3u && !FAST200 && !K96_PASS) {
            if (n == 256u) {
                EvalPoly4Sequential256(wp.dd[0], wp.de_pairs, b8_mid, b8_neg, b8_pval, b8_pexp);
                if (use_poly8) {
                    EvalPoly4Sequential256(wp.dd[0], wp.de_pairs, b8_mid + 4u, b8_neg + 4u, b8_pval + 4u, b8_pexp +
                    4u);
                }
                return;
            }
        }
        if (use_poly8) {
            EvalPoly4SameScaleAs8(wp.dd[0], wp.de_pairs, b8_mid, b8_neg, b8_pval, b8_pexp);
            EvalPoly4SameScaleAs8(wp.dd[0], wp.de_pairs, b8_mid + 4u, b8_neg + 4u, b8_pval + 4u, b8_pexp + 4u);
        } else {
            EvalPoly4(wp.dd[0], wp.de_pairs, b8_mid, b8_neg, b8_pval, b8_pexp);
        }
    }
    __aicore__ __attribute__((noinline)) void SolveTridiagSturmLarge(float *work) {
        SturmWorkPtrs wp; BindSturmWorkPtrs(work, wp); float gersh_min = 0.0f, gersh_max = 0.0f, max_elem = 0.0f;
        if (this->template InitSturmGershgorin<true>(work, wp, gersh_min, gersh_max, max_elem)) {
            return;
        }
        u32 row_end = 0u, ext_start = 0u, ext_end = 0u; InitNeighborBounds(wp, gersh_min, gersh_max, row_end,
        ext_start, ext_end); float tol_single = (FAST200 ? 1.5e-6f : ((n > 200u) ? 8.0e-7f : 1.0e-5f)) * max_elem,
        max_gap_tol = 1.6e-4f * max_elem; this->template RunSturmBisectionQueue<true>(wp, gersh_min, gersh_max,
        1.0e-6f * max_elem, row_end, ext_start, ext_end); RefineNeighborBounds(wp, row_end); float *iso_clo =
        wp.q_lo, *iso_chi = wp.q_hi, *iso_tsec = reinterpret_cast<float *>(wp.q_hc); float *hist_x = wp.q_phi,
        *hist_p = wp.q_plo; i32 *hist_e = wp.q_elo; u32 a_head = 0u, a_tail = 0u, a_cnt = 0u;
        InitLargeActiveQueue(wp, row_end, tol_single, max_elem, max_gap_tol, iso_clo, iso_chi, iso_tsec, hist_e,
        a_tail, a_cnt); float b8_mid[8], b8_pval[8]; i32 b8_pexp[8]; u32 b8_kp[8], b8_pos[8], b8_neg[8], r_batch[8];
        while (a_cnt > 0u) {
            bool use_poly8 = (n != 256u && a_cnt > 4u); u32 max_b = use_poly8 ? 8u : 4u, bcnt = (a_cnt < max_b) ?
            a_cnt : max_b, lane_pos = 0u;
            for (u32 b = 0u; b < bcnt; ++b) {
                u32 r = wp.act_q[a_head]; a_head = (a_head + 1u) & 63u; --a_cnt; r_batch[b] = r; u32 kp = use_poly8 ?
                ((b < 8u - bcnt) ? 2u : 1u) : ((bcnt == 1u) ? 4u : ((bcnt == 2u) ? 2u : ((bcnt == 3u && b == 0u) ?
                2u : 1u))); b8_kp[b] = kp; b8_pos[b] = lane_pos; SelectLargePolyTestPoints(wp, r, kp, lane_pos,
                row_end, iso_clo, iso_chi, iso_tsec, hist_x, hist_p, hist_e, max_gap_tol, b8_mid); lane_pos += kp;
            }
            EvalLargePolyBatch(wp, use_poly8, b8_mid, b8_neg, b8_pval, b8_pexp);
            for (u32 b = 0u; b < bcnt; ++b) {
                UpdateLargeActiveRow(wp, r_batch[b], b8_kp[b], b8_pos[b], row_end, tol_single, max_elem, max_gap_tol,
                iso_clo, iso_chi, iso_tsec, hist_x, hist_p, hist_e, b8_mid, b8_pval, b8_pexp, b8_neg, a_tail, a_cnt);
            }
        }
    }
    __aicore__ inline void ComputeEigenvalues(float *work) {
        if (group_lanes >= 4u && n >= 40u && row_count <= 64u) {
            SolveTridiagSturm(work); return;
        }
        if (SolveTridiagPWK(work)) {
            return;
        }
        SolveTridiagSturm(work);
    }
    __aicore__ inline u32 CountDiagRank(const float *dd, u32 i) const {
        u32 rank = 0u; float di = dd[i];
        for (u32 k = 0u; k < n; ++k) {
            if (dd[k] < di || (dd[k] == di && k < i)) {
                ++rank;
            }
        }
        return rank;
    }
    __aicore__ inline void SetDiagIdentityVecs(const float *dd) {
        for (u32 r = 0u; r < row_count; ++r) {
            u32 col_idx = row_start + r; LTf vec_r = ub_vecs[r * stride];
            for (u32 i = 0u; i < n; ++i) {
                if (CountDiagRank(dd, i) == col_idx) {
                    vec_r.SetValue(i, 1.0f); break;
                }
            }
        }
    }
    template <bool IS_LARGE>
    __aicore__ inline bool CheckDiagAndSetIdentityVecs(const float *work, float *dd, float *ee, float &max_elem) {
        bool is_diag = true; max_elem = 0.0f;
        if constexpr (IS_LARGE || SOLVE_MODE == 2u) {
            float saved_max = work[1023]; is_diag = (saved_max < 0.0f); max_elem = is_diag ? 0.0f : saved_max;
        } else {
            bool already_loaded = (group_lanes >= 4u && n >= 40u && row_count <= 64u);
            for (u32 i = 0u; i < n; ++i) {
                float d_i = already_loaded ? dd[i] : ub_diag.GetValue(i), e_i = already_loaded ? ee[i] :
                ub_off.GetValue(i); dd[i] = d_i; ee[i] = e_i; max_elem = MaxF(MaxF(max_elem, ScalarAbs(d_i)),
                ScalarAbs(e_i)); is_diag = is_diag && !(i + 1u < n && e_i != 0.0f);
            }
        }
        if (row_count > 0u && (is_diag || n < stride)) {
            Duplicate(ub_vecs, 0.0f, row_count * stride); PipeAll();
        }
        if (!is_diag) {
            return false;
        }
        if constexpr (!IS_LARGE && SOLVE_MODE != 2u) {
            for (u32 i = 0u; i < n; ++i) {
                dd[i] = ub_diag.GetValue(i);
            }
        }
        SetDiagIdentityVecs(dd); PipeAll(); return true;
    }
    __aicore__ inline void PrecomputeOffDiagInvRat(const float *ee, float *ee_rat, float *inv_ee,
    bool &all_e_nz) const {
        all_e_nz = true; u32 i_ee = 0u;
        for (; i_ee + 4u < n; i_ee += 4u) {
            float e0 = ee[i_ee], e1 = ee[i_ee + 1u], e2 = ee[i_ee + 2u], e3 = ee[i_ee + 3u], e4 = ee[i_ee + 4u];
            if (ScalarAbs(e0) >= 1e-7f && ScalarAbs(e1) >= 1e-7f && ScalarAbs(e2) >= 1e-7f &&
                ScalarAbs(e3) >= 1e-7f) {
                float p01 = e0 * e1, p23 = e2 * e3, inv_all = 1.0f / (p01 * p23), inv01 = inv_all * p23, inv23 =
                inv_all * p01, iv0 = inv01 * e1, iv1 = inv01 * e0, iv2 = inv23 * e3, iv3 = inv23 * e2; inv_ee[i_ee] =
                iv0; inv_ee[i_ee + 1u] = iv1; inv_ee[i_ee + 2u] = iv2; inv_ee[i_ee + 3u] = iv3; ee_rat[i_ee] =
                e1 * iv0; ee_rat[i_ee + 1u] = e2 * iv1; ee_rat[i_ee + 2u] = e3 * iv2; ee_rat[i_ee + 3u] = e4 * iv3;
            } else {
                all_e_nz = false;
                for (u32 k = 0u; k < 4u; ++k) {
                    float ek = ee[i_ee + k], ivk = (ek != 0.0f) ? (1.0f / ek) : 0.0f; inv_ee[i_ee + k] = ivk;
                    ee_rat[i_ee + k] = ee[i_ee + k + 1u] * ivk;
                }
            }
        }
        for (; i_ee + 1u < n; ++i_ee) {
            float ei = ee[i_ee]; all_e_nz = (ScalarAbs(ei) < 1e-7f) ? (false) : all_e_nz; float inv_e =
            (ei != 0.0f) ? (1.0f / ei) : 0.0f; inv_ee[i_ee] = inv_e; ee_rat[i_ee] = ee[i_ee + 1u] * inv_e;
        }
    }
    __aicore__ inline void ComputeLast4Reciprocals(float cur_p0, float cur_p1, float cur_p2, float cur_p3,
    float *s_last, bool batched_div) const {
        float lden0 = ClampPivot(cur_p0), lden1 = ClampPivot(cur_p1), lden2 = ClampPivot(cur_p2), lden3 =
        ClampPivot(cur_p3);
        if (batched_div && ScalarAbs(lden0) >= 1e-7f && ScalarAbs(lden1) >= 1e-7f && ScalarAbs(lden2) >= 1e-7f &&
            ScalarAbs(lden3) >= 1e-7f) {
            float lp01 = lden0 * lden1, lp23 = lden2 * lden3, linv_all = 1.0f / (lp01 * lp23), linv01 =
            linv_all * lp23, linv23 = linv_all * lp01; s_last[0] = linv01 * lden1; s_last[1] = linv01 * lden0;
            s_last[2] = linv23 * lden3; s_last[3] = linv23 * lden2;
        } else {
            s_last[0] = (lden0 != 0.0f) ? (1.0f / lden0) : 1e15f; s_last[1] = (lden1 != 0.0f) ? (1.0f / lden1) :
            1e15f; s_last[2] = (lden2 != 0.0f) ? (1.0f / lden2) : 1e15f; s_last[3] = (lden3 != 0.0f) ?
            (1.0f / lden3) : 1e15f;
        }
    }
    template <u32 N_DIM>
    __aicore__ inline void ApplyPrev4DeferredScales(u32 r, float *v_scale) {
        if (r < 4u) {
            return;
        }
        for (u32 ri = r - 4u; ri < r; ++ri) {
            if (v_scale[ri] != 0.0f) {
                Muls(ub_vecs[ri * N_DIM], ub_vecs[ri * N_DIM], v_scale[ri], N_DIM); v_scale[ri] = 0.0f;
            }
        }
    }
    template <u32 N_DIM>
    __aicore__ inline void StartEvaluateFinalPacket(u32 r, float *v_scale, u32 bcnt = 4u) {
        constexpr u32 d_off = (N_DIM == 200u) ? 22144u : 30208u; constexpr u32 e_off = d_off + N_DIM, idx_off =
        e_off + N_DIM, sq_off = idx_off + N_DIM; constexpr u32 off_off = sq_off + 4u * N_DIM, nxt_off = off_off +
        4u * N_DIM; constexpr u32 trm_off = nxt_off + 4u * N_DIM, red_off = trm_off + 4u * N_DIM;
        constexpr u8 rep_st = u8(N_DIM / 8u); LTf d_copy = ub_mat[d_off], e_copy = ub_mat[e_off]; LTu shift_idx =
        ub_mat[idx_off].template ReinterpretCast<u32>(); LTf sq = ub_mat[sq_off], off = ub_mat[off_off], next =
        ub_mat[nxt_off], terms = ub_mat[trm_off], red = ub_mat[red_off]; u32 tot = bcnt * N_DIM; u8 rep = u8(bcnt);
        this->template fence<HardEvent::S_V>(); Mul(sq, ub_vecs[r * N_DIM], ub_vecs[r * N_DIM], tot);
        for (u32 j = 0u; j < bcnt; ++j) {
            Gather(next[j * N_DIM], ub_vecs[(r + j) * N_DIM], shift_idx, 0u, N_DIM);
        }
        PipeV(); Mul(off, next, ub_vecs[r * N_DIM], tot);
        for (u32 j = 0u; j < bcnt; ++j) {
            Mul(terms[j * N_DIM], sq[j * N_DIM], d_copy, N_DIM);
        }
        PipeV();
        for (u32 c = 0u; c < N_DIM; c += 64u) {
            u32 w = (N_DIM - c < 64u) ? (N_DIM - c) : 64u; MulAddDst(terms[c], off[c], e_copy[c], w, rep, BRP{1, 1,
            1, rep_st, rep_st, 0});
        }
        PipeV();
        for (u32 s = 0u; s < (N_DIM >> 6u); ++s) {
            WholeReduceSum(red[s * 8u], sq[s * 64u], 64u, rep, 1u, 1u, rep_st); WholeReduceSum(red[32u + s * 8u],
            terms[s * 64u], 64u, rep, 1u, 1u, rep_st);
        }
        if constexpr (N_DIM == 200u) {
            WholeReduceSum(red[24u], sq[192u], 8u, rep, 1u, 1u, rep_st); WholeReduceSum(red[56u], terms[192u], 8u,
            rep, 1u, 1u, rep_st);
        }
        PipeV(); Add(red, red, red[8u], bcnt); Add(red[16u], red[16u], red[24u], bcnt); Add(red[32u], red[32u],
        red[40u], bcnt); Add(red[48u], red[48u], red[56u], bcnt); PipeV(); Add(red, red, red[16u], bcnt);
        Add(red[32u], red[32u], red[48u], bcnt); ApplyPrev4DeferredScales<N_DIM>(r, v_scale);
    }
    __aicore__ inline void FinishEvaluateFinalPacket200(u32 r, u32 bcnt, u32 c0, u32 c1, u32 c2, u32 c3,
    float *v_scale) {
        this->template FinishEvaluateFinalPacket<200u>(r, bcnt, c0, c1, c2, c3, v_scale);
    }
    __aicore__ inline void FinishEvaluateFinalPacket256(u32 r, u32 c0, u32 c1, u32 c2, u32 c3, float *v_scale,
    u32 bcnt = 4u) {
        this->template FinishEvaluateFinalPacket<256u>(r, bcnt, c0, c1, c2, c3, v_scale);
    }
    template <u32 N_DIM>
    __aicore__ inline void FinishEvaluateFinalPacket(u32 r, u32 bcnt, u32 c0, u32 c1, u32 c2, u32 c3,
    float *v_scale) {
        constexpr u32 red_off = (N_DIM == 200u) ? 25944u : 35072u; LTf red = ub_mat[red_off];
        this->template fence<HardEvent::V_S>(); u32 cols[4] = {c0, c1, c2, c3};
        for (u32 j = 0u; j < bcnt; ++j) {
            float norm = red.GetValue(j), v_sc = ScalarRsqrt(norm); v_scale[r + j] = v_sc;
            if (v_sc != 0.0f && norm != 0.0f) {
                ub_eig.SetValue(cols[j], red.GetValue(32u + j) / norm);
            }
        }
    }
};
} // namespace AscendC
#endif
