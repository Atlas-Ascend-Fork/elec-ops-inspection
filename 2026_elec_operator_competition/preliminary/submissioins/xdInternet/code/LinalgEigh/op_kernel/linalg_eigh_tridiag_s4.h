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
#ifndef LINALG_EIGH_TRIDIAG_S4_H
#define LINALG_EIGH_TRIDIAG_S4_H
#include "linalg_eigh_tridiag_s3.h"
namespace AscendC {
template <typename DT_A, u32 SOLVE_MODE = 0u, bool FAST200 = false, bool K96_PASS = false>
class KernelEighTridiagS4 : public KernelEighTridiagS3<DT_A, SOLVE_MODE, FAST200, K96_PASS> {
public:
    using Base = KernelEighTridiagS3<DT_A, SOLVE_MODE, FAST200, K96_PASS>;
    using Base::ub_off, Base::ub_eig, Base::ScalarAbs, Base::MaxF, Base::MinF, Base::ClampF;
    using Base::ComputeSecantWeight, Base::EvalSturm4Fast256, Base::EvalSturm4, Base::EvalPoly4;
    using Base::SortAscendingInPlace, Base::TridiagonalizeLarge200, Base::TridiagonalizeLargeSub200;
    using Base::TridiagonalizeLargeZoneLoop, Base::n, Base::row_start, Base::row_count, Base::ub_diag;
    __aicore__ inline KernelEighTridiagS4() = default;
    __aicore__ __attribute__((noinline)) void TridiagonalizeLarge(u64 b, float *work, u8 *tau_mask) {
        if (n <= 200u) {
            (n == 200u) ? (void)TridiagonalizeLarge200(work, tau_mask) : (void)TridiagonalizeLargeSub200(work,
            tau_mask); return;
        }
        if constexpr (!FAST200) {
            TridiagonalizeLargeZoneLoop(b, work, tau_mask);
        }
    }
    __aicore__ inline void BindSturmWorkPtrs(float *work, SturmWorkPtrs &wp) const {
        u32 w_cap = (n <= 256u) ? 256u : 512u; wp.dd = work; wp.ee = work + w_cap; wp.de_pairs = work + 2u * w_cap;
        wp.q_lo = work + 4u * w_cap; wp.q_hi = wp.q_lo + 64; wp.iso_lo = wp.q_hi + 64; wp.iso_hi = wp.iso_lo + 64;
        u32 *u_base = reinterpret_cast<u32 *>(wp.iso_hi + 64); wp.q_lc = u_base; wp.q_hc = wp.q_lc + 64; wp.q_depth =
        wp.q_hc + 64; wp.act_q = wp.q_depth + 64; wp.act_depth = wp.act_q + 64; wp.q_plo =
        reinterpret_cast<float *>(wp.act_depth + 64); wp.q_phi = wp.q_plo + 64; wp.iso_plo = wp.q_phi + 64;
        wp.iso_phi = wp.iso_plo + 64; wp.q_elo = reinterpret_cast<i32 *>(wp.iso_phi + 64); wp.q_ehi = wp.q_elo + 64;
        wp.iso_elo = wp.q_ehi + 64; wp.iso_ehi = wp.iso_elo + 64; wp.iso_stag =
        reinterpret_cast<int8_t *>(wp.iso_ehi + 64);
        for (u32 r = 0u; r < row_count; ++r) {
            wp.iso_plo[r] = 0.0f; wp.iso_phi[r] = 0.0f; wp.iso_elo[r] = -30000; wp.iso_ehi[r] = -30000;
            wp.iso_stag[r] = 0;
        }
    }
    __aicore__ inline void FinishDiagSturmSort(float *work, SturmWorkPtrs &wp) {
        float *sd = wp.de_pairs;
        for (u32 i = 0u; i < n; ++i) {
            sd[i] = wp.dd[i];
        }
        SortAscendingInPlace(sd, n);
        for (u32 r = 0u; r < row_count; ++r) {
            ub_eig.SetValue(row_start + r, sd[row_start + r]);
        }
        work[1023] = -1.0f;
    }
    template <bool LARGE_MODE>
    __aicore__ inline bool InitSturmGershgorin(float *work, SturmWorkPtrs &wp, float &gersh_min, float &gersh_max,
    float &max_elem) {
        gersh_min = 0.0f; gersh_max = 0.0f; max_elem = 0.0f; bool is_diag = true; float prev_abs_e = 0.0f;
        for (u32 i = 0u; i < n; ++i) {
            float d_i = (LARGE_MODE || SOLVE_MODE == 2u) ? wp.dd[i] : ub_diag.GetValue(i), e_i = (LARGE_MODE ||
            SOLVE_MODE == 2u) ? wp.ee[i] : ub_off.GetValue(i);
            if constexpr (!LARGE_MODE && SOLVE_MODE != 2u) {
                wp.dd[i] = d_i; wp.ee[i] = e_i;
            }
            if (i > 0u) {
                wp.de_pairs[(i - 1u) * 2u] = d_i;
            }
            float abs_e = 0.0f;
            if (i + 1u < n) {
                wp.de_pairs[i * 2u + 1u] = e_i * e_i; is_diag = is_diag && (e_i == 0.0f); abs_e = ScalarAbs(e_i);
            }
            float rad = prev_abs_e + abs_e; prev_abs_e = abs_e; float lo_i = d_i - rad, hi_i = d_i + rad; gersh_min =
            (i == 0u) ? lo_i : MinF(gersh_min, lo_i); gersh_max = (i == 0u) ? hi_i : MaxF(gersh_max, hi_i);
            max_elem = MaxF(MaxF(max_elem, ScalarAbs(d_i)), abs_e);
        }
        if (is_diag) {
            FinishDiagSturmSort(work, wp); return true;
        }
        work[1023] = max_elem;
        if (row_count == 0u) {
            return true;
        }
        float pad = 1e-5f * (ScalarAbs(gersh_min) + ScalarAbs(gersh_max)) + 1e-6f; gersh_min -= pad;
        gersh_max += pad; return false;
    }
    __aicore__ inline void RecordIsolatedInterval(SturmWorkPtrs &wp, u32 row_end, u32 lc, float lo, float hi,
    float plo, float phi, i32 elo, i32 ehi) {
        if (lc >= row_start && lc < row_end) {
            u32 r = lc - row_start; wp.iso_lo[r] = lo; wp.iso_hi[r] = hi; wp.iso_plo[r] = plo; wp.iso_phi[r] = phi;
            wp.iso_elo[r] = elo; wp.iso_ehi[r] = ehi;
        } else if (row_start > 0u && lc == row_start - 1u) {
            wp.left_nb_lo = lo; wp.left_nb_hi = hi; wp.left_nb_plo = plo; wp.left_nb_phi = phi; wp.left_nb_elo = elo;
            wp.left_nb_ehi = ehi;
        } else if (row_end < n && lc == row_end) {
            wp.right_nb_lo = lo; wp.right_nb_hi = hi; wp.right_nb_plo = plo; wp.right_nb_phi = phi; wp.right_nb_elo =
            elo; wp.right_nb_ehi = ehi;
        }
    }
    __aicore__ inline u32 PopBisectionBatch4(SturmWorkPtrs &wp, u32 &q_head, u32 &q_count, u32 row_end,
    float tol_cluster, float *b_lo, float *b_hi, float *b_plo, float *b_phi, i32 *b_elo, i32 *b_ehi, u32 *b_lc,
    u32 *b_hc, u32 *b_depth) {
        u32 bcnt = 0u;
        while (q_count > 0u && bcnt < 4u) {
            float lo = wp.q_lo[q_head], hi = wp.q_hi[q_head], plo = wp.q_plo[q_head], phi = wp.q_phi[q_head];
            i32 elo = wp.q_elo[q_head], ehi = wp.q_ehi[q_head]; u32 lc = wp.q_lc[q_head], hc = wp.q_hc[q_head],
            depth = wp.q_depth[q_head]; q_head = (q_head + 1u) & 63u; --q_count;
            if (hc - lc == 1u) {
                RecordIsolatedInterval(wp, row_end, lc, lo, hi, plo, phi, elo, ehi); continue;
            }
            float mid = 0.5f * lo + 0.5f * hi;
            if (depth >= 32u || mid == lo || mid == hi || (hi - lo) <= tol_cluster + 1e-30f) {
                u32 s_idx = (lc > row_start) ? lc : row_start, e_idx = (hc < row_end) ? hc : row_end;
                for (u32 j = s_idx; j < e_idx; ++j) {
                    wp.iso_lo[j - row_start] = mid; wp.iso_hi[j - row_start] = mid;
                }
                if (row_start > 0u && lc <= row_start - 1u && hc > row_start - 1u) {
                    wp.left_nb_lo = mid; wp.left_nb_hi = mid;
                }
                if (row_end < n && lc <= row_end && hc > row_end) {
                    wp.right_nb_lo = mid; wp.right_nb_hi = mid;
                }
                continue;
            }
            b_lo[bcnt] = lo; b_hi[bcnt] = hi; b_plo[bcnt] = plo; b_phi[bcnt] = phi; b_elo[bcnt] = elo; b_ehi[bcnt] =
            ehi; b_lc[bcnt] = lc; b_hc[bcnt] = hc; b_depth[bcnt] = depth; ++bcnt;
        }
        return bcnt;
    }
    __aicore__ inline void RecordOrEnqueueSubInterval(SturmWorkPtrs &wp, u32 &q_tail, u32 &q_count, u32 row_end,
    u32 ext_start, u32 ext_end, u32 next_depth, u32 prev_c, u32 cur_c, float prev_v, float cur_v, float prev_p,
    float cur_p, i32 prev_e, i32 cur_e) {
        if (!(prev_c < cur_c && cur_c > ext_start && prev_c < ext_end)) {
            return;
        }
        if (cur_c - prev_c == 1u) {
            RecordIsolatedInterval(wp, row_end, prev_c, prev_v, cur_v, prev_p, cur_p, prev_e, cur_e); return;
        }
        wp.q_lo[q_tail] = prev_v; wp.q_hi[q_tail] = cur_v; wp.q_plo[q_tail] = prev_p; wp.q_phi[q_tail] = cur_p;
        wp.q_elo[q_tail] = prev_e; wp.q_ehi[q_tail] = cur_e; wp.q_lc[q_tail] = prev_c; wp.q_hc[q_tail] = cur_c;
        wp.q_depth[q_tail] = next_depth; q_tail = (q_tail + 1u) & 63u; ++q_count;
    }
    __aicore__ inline void DispatchBisectionResults4(SturmWorkPtrs &wp, u32 bcnt, u32 &q_tail, u32 &q_count,
    u32 row_end, u32 ext_start, u32 ext_end, const float *b_lo, const float *b_hi, const float *b_plo,
    const float *b_phi, const i32 *b_elo, const i32 *b_ehi, const u32 *b_lc, const u32 *b_hc, const u32 *b_depth,
    const u32 *b_kp, const u32 *b_pos, const float *b_mid, const float *b_pval, const i32 *b_pexp, const u32 *b_neg) {
        for (u32 b = 0u; b < bcnt; ++b) {
            u32 lc = b_lc[b], hc = b_hc[b], kp = b_kp[b], pos = b_pos[b], next_depth = b_depth[b] + ((kp >= 3u) ?
            2u : 1u), prev_c = lc; float prev_v = b_lo[b], prev_p = b_plo[b]; i32 prev_e = b_elo[b];
            for (u32 k = 0u; k <= kp; ++k) {
                float cur_v = (k < kp) ? b_mid[pos + k] : b_hi[b], cur_p = (k < kp) ? b_pval[pos + k] : b_phi[b];
                i32 cur_e = (k < kp) ? b_pexp[pos + k] : b_ehi[b]; u32 cur_c = (k < kp) ? b_neg[pos + k] : hc;
                cur_c = (cur_c < prev_c) ? (prev_c) : cur_c; cur_c = (cur_c > hc) ? (hc) : cur_c;
                RecordOrEnqueueSubInterval(wp, q_tail, q_count, row_end, ext_start, ext_end, next_depth, prev_c,
                cur_c, prev_v, cur_v, prev_p, cur_p, prev_e, cur_e); prev_v = cur_v; prev_p = cur_p; prev_e = cur_e;
                prev_c = cur_c;
            }
        }
    }
    static __aicore__ inline u32 SelectBest3BatchIdx(u32 ext_start, u32 ext_end, const u32 *b_lc, const u32 *b_hc) {
        u32 c0 = ((b_hc[0] < ext_end) ? b_hc[0] : ext_end) - ((b_lc[0] > ext_start) ? b_lc[0] : ext_start), c1 =
        ((b_hc[1] < ext_end) ? b_hc[1] : ext_end) - ((b_lc[1] > ext_start) ? b_lc[1] : ext_start), c2 =
        ((b_hc[2] < ext_end) ? b_hc[2] : ext_end) - ((b_lc[2] > ext_start) ? b_lc[2] : ext_start); return (c1 > c0) ?
        ((c2 > c1) ? 2u : 1u) : ((c2 > c0) ? 2u : 0u);
    }
    template <bool LARGE_MODE>
    __aicore__ inline void PrepareBisectionBatchMidpoints(u32 bcnt, u32 ext_start, u32 ext_end, const float *b_lo,
    const float *b_hi, const u32 *b_lc, const u32 *b_hc, u32 *b_kp, u32 *b_pos, float *b_mid) const {
        u32 best_b3 = (!LARGE_MODE && bcnt == 3u) ? SelectBest3BatchIdx(ext_start, ext_end, b_lc, b_hc) : 0u,
        lane_pos = 0u;
        for (u32 b = 0u; b < bcnt; ++b) {
            u32 kp = (bcnt == 1u) ? 4u : ((bcnt == 2u) ? 2u : ((bcnt == 3u && b == best_b3) ? 2u : 1u)); b_kp[b] =
            kp; b_pos[b] = lane_pos; float lo = b_lo[b], w_b = b_hi[b] - lo;
            if constexpr (LARGE_MODE) {
                u32 lc_b = b_lc[b], hc_b = b_hc[b], sp_b = hc_b - lc_b, wl = (lc_b > ext_start) ? lc_b : ext_start,
                wh = (hc_b < ext_end) ? hc_b : ext_end;
                if ((FAST200 || n == 256u) && sp_b >= 6u && (wh - wl) < sp_b) {
                    float inv_sp = 1.0f / u2f(sp_b), f0 = u2f(wl - lc_b) * inv_sp, f1 = u2f(wh - lc_b) * inv_sp, a =
                    0.6f * f0, b_end = (1.0f - 0.6f) + 0.6f * f1; lo = lo + a * w_b; w_b = (b_end - a) * w_b;
                }
            }
            float step = w_b * ((kp == 4u) ? 0.2f : ((kp == 2u) ? (1.0f / 3.0f) : 0.5f));
            for (u32 k = 0u; k < kp; ++k) {
                b_mid[lane_pos + k] = lo + step * u2f(k + 1u);
            }
            lane_pos += kp;
        }
    }
    template <bool LARGE_MODE>
    __aicore__ inline void RunSturmBisectionQueue(SturmWorkPtrs &wp, float gersh_min, float gersh_max,
    float tol_cluster, u32 row_end, u32 ext_start, u32 ext_end) {
        u32 q_head = 0u, q_tail = 1u, q_count = 1u; wp.q_lo[0] = gersh_min; wp.q_hi[0] = gersh_max; wp.q_plo[0] =
        0.0f; wp.q_phi[0] = 0.0f; wp.q_elo[0] = -30000; wp.q_ehi[0] = -30000; wp.q_lc[0] = 0u; wp.q_hc[0] = n;
        wp.q_depth[0] = 0u; float b_lo[4], b_hi[4], b_mid[4], b_plo[4], b_phi[4], b_pval[4]; i32 b_elo[4], b_ehi[4],
        b_pexp[4]; u32 b_lc[4], b_hc[4], b_depth[4], b_kp[4], b_pos[4], b_neg[4];
        while (q_count > 0u) {
            u32 bcnt = PopBisectionBatch4(wp, q_head, q_count, row_end, tol_cluster, b_lo, b_hi, b_plo, b_phi, b_elo,
            b_ehi, b_lc, b_hc, b_depth);
            if (bcnt == 0u) {
                break;
            }
            PrepareBisectionBatchMidpoints<LARGE_MODE>(bcnt, ext_start, ext_end, b_lo, b_hi, b_lc, b_hc, b_kp, b_pos,
            b_mid);
            if constexpr (LARGE_MODE && SOLVE_MODE == 3u && !FAST200) {
                (n == 256u) ? (void)EvalSturm4Fast256(wp.dd[0], wp.de_pairs, b_mid, b_neg, b_pval, b_pexp) :
                (void)EvalSturm4(wp.dd[0], wp.de_pairs, b_mid, b_neg, b_pval, b_pexp);
            } else {
                EvalSturm4(wp.dd[0], wp.de_pairs, b_mid, b_neg, b_pval, b_pexp);
            }
            DispatchBisectionResults4(wp, bcnt, q_tail, q_count, row_end, ext_start, ext_end, b_lo, b_hi, b_plo,
            b_phi, b_elo, b_ehi, b_lc, b_hc, b_depth, b_kp, b_pos, b_mid, b_pval, b_pexp, b_neg);
        }
    }
    __aicore__ inline void RefineNeighborBounds(SturmWorkPtrs &wp, u32 row_end) {
        if (row_start > 0u) {
            float eig_l = wp.left_nb_hi, t_nb = 0.5f;
            if (ComputeSecantWeight(wp.left_nb_plo, wp.left_nb_elo, wp.left_nb_phi, wp.left_nb_ehi, t_nb)) {
                eig_l = wp.left_nb_lo + t_nb * (wp.left_nb_hi - wp.left_nb_lo); wp.left_nb_hi = 0.5f * (eig_l +
                wp.left_nb_hi);
            }
            ub_eig.SetValue(row_start - 1u, eig_l);
        }
        if (row_end < n) {
            float eig_r = wp.right_nb_lo, t_nb = 0.5f;
            if (ComputeSecantWeight(wp.right_nb_plo, wp.right_nb_elo, wp.right_nb_phi, wp.right_nb_ehi, t_nb)) {
                eig_r = wp.right_nb_lo + t_nb * (wp.right_nb_hi - wp.right_nb_lo); wp.right_nb_lo =
                0.5f * (wp.right_nb_lo + eig_r);
            }
            ub_eig.SetValue(row_end, eig_r);
        }
    }
    __aicore__ inline bool TryComputeIQIWeight(float lo, float width, i32 el, i32 eh, i32 ez, float fl, float fh,
    float fz, float hx, float t_min, float t_max, float &iqi_out) const {
        i32 em = (el > eh) ? el : eh; em = (ez > em) ? (ez) : em;
        if (em - el > 1 || em - eh > 1 || em - ez > 1) {
            return false;
        }
        fl = (el < em) ? (fl * 5.4210108624275222e-20f) : fl; fh = (eh < em) ? (fh * 5.4210108624275222e-20f) : fh;
        fz = (ez < em) ? (fz * 5.4210108624275222e-20f) : fz; float fm = MaxF(MaxF(fl, fh), fz);
        if (!(fm > 0.0f && width > 0.0f)) {
            return false;
        }
        fl /= fm; fh /= fm; fz /= fm; fz = (hx < lo) ? (-fz) : fz; float tz = (hx - lo) / width, d0 = fz - fh, d1 =
        fz + fl, d_lh = fl + fh;
        if (!(ScalarAbs(d0) > 1.0e-6f && ScalarAbs(d1) > 1.0e-6f && d_lh > 1.0e-6f && ScalarAbs(tz) < 8.0f)) {
            return false;
        }
        float it = (fl / d_lh) * (fz / d0) - tz * (fl / d1) * (fh / d0);
        if (it >= t_min && it <= t_max) {
            iqi_out = it; return true;
        }
        return false;
    }
    __aicore__ inline void DampStagnatedSecantWeight(u32 r, int8_t stg, float width, float max_gap_tol,
    float edge_mult, float &t_sec) const {
        if (stg == 2) {
            t_sec = MaxF((t_sec < 0.5f) ? (0.45f * t_sec) : (t_sec - 0.20f), 0.015f); return;
        }
        if (stg == -2) {
            t_sec = MinF((t_sec > 0.5f) ? (1.0f - 0.45f * (1.0f - t_sec)) : (t_sec + 0.20f), 0.985f); return;
        }
        if (stg == 1) {
            t_sec = (t_sec > 0.92f) ? (1.0f - 2.2f * (1.0f - t_sec) - 0.012f) : ((t_sec < 0.25f) ? (0.65f * t_sec) :
            (t_sec - 0.045f));
        } else if (stg == -1) {
            t_sec = (t_sec < 0.08f) ? (2.2f * t_sec + 0.012f) : ((t_sec > 0.75f) ? (1.0f - 0.65f * (1.0f - t_sec)) :
            (t_sec + 0.045f));
        }
        bool is_edge = (row_start + r == 0u || row_start + r + 1u == n) && (width > edge_mult * max_gap_tol); t_sec =
        is_edge ? 0.5f : ClampF(t_sec, 0.015f, 0.985f);
    }
    __aicore__ inline float ComputeRowSeparationGap(u32 r, u32 row_end, const float *iso_clo, const float *iso_chi,
    float left_nb_hi, float right_nb_lo) const {
        float l_gap = (r > 0u) ? (iso_clo[r] - iso_chi[r - 1u]) : ((row_start > 0u) ? (iso_clo[r] - left_nb_hi) :
        1.0e30f), r_gap = ((r + 1u) < row_count) ? (iso_clo[r + 1u] - iso_chi[r]) : ((row_end < n) ? (right_nb_lo -
        iso_chi[r]) : 1.0e30f); return (l_gap < r_gap) ? l_gap : r_gap;
    }
    __aicore__ inline float ComputeAdaptiveRowTol(float base_tol, float gap, float mid, float max_elem,
    float ptol_scale, float max_gap_tol) const {
        float cur_tol = base_tol;
        if (gap >= 0.0005f * max_elem && (ScalarAbs(mid) >= 0.015f * max_elem || gap >= 0.0015f * max_elem)) {
            cur_tol = MaxF(cur_tol, MinF(ptol_scale * gap, max_gap_tol));
        }
        return cur_tol;
    }
    __aicore__ inline void SelectMediumPolyTestPoints(SturmWorkPtrs &wp, u32 r, u32 kp, u32 lane_pos, u32 row_end,
    const float *iso_clo, const float *iso_chi, float max_gap_tol, float *b_mid) const {
        float lo = wp.iso_lo[r], width = wp.iso_hi[r] - lo, t_sec = 0.5f; bool has_sec =
        ComputeSecantWeight(wp.iso_plo[r], wp.iso_elo[r], wp.iso_phi[r], wp.iso_ehi[r], t_sec);
        if (kp == 1u && has_sec) {
            int8_t stg = wp.iso_stag[r]; float gp = ComputeRowSeparationGap(r, row_end, iso_clo, iso_chi,
            wp.left_nb_hi, wp.right_nb_lo);
            if (width > 0.55f * gp || stg >= 3 || stg <= -3) {
                t_sec = 0.5f; wp.iso_stag[r] = 0;
            } else {
                DampStagnatedSecantWeight(r, stg, width, max_gap_tol, 8.0f, t_sec);
            }
            b_mid[lane_pos] = lo + t_sec * width;
        } else if (kp == 2u && has_sec) {
            float tc = ClampF(t_sec, 0.12f, 0.88f); b_mid[lane_pos] = lo + (tc - 0.08f) * width; b_mid[lane_pos +
            1u] = lo + (tc + 0.08f) * width;
        } else {
            float step = width * ((kp == 4u) ? 0.2f : ((kp == 2u) ? (1.0f / 3.0f) : 0.5f));
            for (u32 k = 0u; k < kp; ++k) {
                b_mid[lane_pos + k] = lo + step * u2f(k + 1u);
            }
        }
    }
    __aicore__ inline void FindUpdatedPolyBracket(SturmWorkPtrs &wp, u32 r, u32 kp, u32 pos, const float *b_mid,
    const float *b_pval, const i32 *b_pexp, const u32 *b_neg, u32 &hit_k) {
        float lo = wp.iso_lo[r], hi = wp.iso_hi[r], plo = wp.iso_plo[r], phi = wp.iso_phi[r]; i32 elo =
        wp.iso_elo[r], ehi = wp.iso_ehi[r]; u32 target = row_start + r + 1u; float new_lo = b_mid[pos + kp - 1u],
        new_plo = b_pval[pos + kp - 1u], new_hi = hi, new_phi = phi; i32 new_elo = b_pexp[pos + kp - 1u], new_ehi =
        ehi; int8_t next_stag = (wp.iso_stag[r] < 0) ? (wp.iso_stag[r] - 1) : -1; hit_k = kp;
        for (u32 k = 0u; k < kp; ++k) {
            if (b_neg[pos + k] == (target & 1u)) {
                hit_k = k; new_lo = (k == 0u) ? lo : b_mid[pos + k - 1u]; new_plo = (k == 0u) ? plo : b_pval[pos +
                k - 1u]; new_elo = (k == 0u) ? elo : b_pexp[pos + k - 1u]; next_stag = (k == 0u) ?
                ((wp.iso_stag[r] > 0) ? (wp.iso_stag[r] + 1) : 1) : 0; new_hi = b_mid[pos + k]; new_phi =
                b_pval[pos + k]; new_ehi = b_pexp[pos + k]; break;
            }
        }
        wp.iso_stag[r] = next_stag; wp.iso_lo[r] = new_lo; wp.iso_hi[r] = new_hi; wp.iso_plo[r] = new_plo;
        wp.iso_phi[r] = new_phi; wp.iso_elo[r] = new_elo; wp.iso_ehi[r] = new_ehi; wp.act_depth[r] += (kp >= 7u) ?
        3u : ((kp >= 3u) ? 2u : 1u);
    }
    __aicore__ inline void UpdateMediumActiveRow(SturmWorkPtrs &wp, u32 r, u32 kp, u32 pos, u32 row_end,
    float tol_single, float max_elem, float max_gap_tol, float *iso_clo, float *iso_chi, const float *b_mid,
    const float *b_pval, const i32 *b_pexp, const u32 *b_neg, u32 &a_tail, u32 &a_cnt) {
        u32 hit_k = kp; FindUpdatedPolyBracket(wp, r, kp, pos, b_mid, b_pval, b_pexp, b_neg, hit_k); float new_lo =
        wp.iso_lo[r], new_hi = wp.iso_hi[r], next_mid = 0.5f * new_lo + 0.5f * new_hi, t_sec = 0.5f; bool has_sec =
        ComputeSecantWeight(wp.iso_plo[r], wp.iso_elo[r], wp.iso_phi[r], wp.iso_ehi[r], t_sec); float width =
        new_hi - new_lo, est = new_lo + t_sec * width; iso_clo[r] = has_sec ? (0.5f * (new_lo + est)) : new_lo;
        iso_chi[r] = has_sec ? (0.5f * (est + new_hi)) : new_hi; float gap = ComputeRowSeparationGap(r, row_end,
        iso_clo, iso_chi, wp.left_nb_hi, wp.right_nb_lo), cur_tol = ComputeAdaptiveRowTol(tol_single, gap, next_mid,
        max_elem, 0.030f, max_gap_tol), eff_width = has_sec ? (2.2f * t_sec * (1.0f - t_sec) * width) : width,
        w_lim = ((ScalarAbs(next_mid) >= 0.015f * max_elem) ? 8.0f : 4.0f) * cur_tol; int8_t next_stag =
        wp.iso_stag[r];
        if (wp.act_depth[r] >= 32u || next_mid == new_lo || next_mid == new_hi || (next_stag >= -1 &&
            next_stag <= 1 && width <= w_lim && eff_width <= cur_tol + 1e-30f) || width <= cur_tol + 1e-30f) {
            ub_eig.SetValue(row_start + r, has_sec ? est : next_mid);
        } else {
            wp.act_q[a_tail] = r; a_tail = (a_tail + 1u) & 63u; ++a_cnt;
        }
    }
    __aicore__ inline void InitNeighborBounds(SturmWorkPtrs &wp, float gersh_min, float gersh_max, u32 &row_end,
    u32 &ext_start, u32 &ext_end) const {
        row_end = row_start + row_count; ext_start = (row_start > 0u) ? (row_start - 1u) : 0u; ext_end =
        (row_end < n) ? (row_end + 1u) : n; wp.left_nb_lo = gersh_min; wp.left_nb_hi = gersh_min; wp.left_nb_plo =
        0.0f; wp.left_nb_phi = 0.0f; wp.left_nb_elo = -30000; wp.left_nb_ehi = -30000; wp.right_nb_lo = gersh_max;
        wp.right_nb_hi = gersh_max; wp.right_nb_plo = 0.0f; wp.right_nb_phi = 0.0f; wp.right_nb_elo = -30000;
        wp.right_nb_ehi = -30000;
    }
    __aicore__ inline void InitMediumActiveQueue(SturmWorkPtrs &wp, u32 row_end, float tol_single, float max_elem,
    float max_gap_tol, float *iso_clo, float *iso_chi, u32 &a_tail, u32 &a_cnt) {
        for (u32 r = 0u; r < row_count; ++r) {
            float lo = wp.iso_lo[r], hi = wp.iso_hi[r], t_s = 0.5f;
            if (ComputeSecantWeight(wp.iso_plo[r], wp.iso_elo[r], wp.iso_phi[r], wp.iso_ehi[r], t_s)) {
                float est = lo + t_s * (hi - lo); iso_clo[r] = 0.5f * (lo + est); iso_chi[r] = 0.5f * (est + hi);
            } else {
                iso_clo[r] = lo; iso_chi[r] = hi;
            }
        }
        for (u32 r = 0u; r < row_count; ++r) {
            wp.act_depth[r] = 0u; float lo = wp.iso_lo[r], hi = wp.iso_hi[r], mid = 0.5f * lo + 0.5f * hi, gap =
            ComputeRowSeparationGap(r, row_end, iso_clo, iso_chi, wp.left_nb_hi, wp.right_nb_lo), cur_tol =
            ComputeAdaptiveRowTol(tol_single, gap, mid, max_elem, 0.030f, max_gap_tol);
            if (mid == lo || mid == hi || (hi - lo) <= cur_tol + 1e-30f) {
                float t_sec = 0.5f;
                if (ComputeSecantWeight(wp.iso_plo[r], wp.iso_elo[r], wp.iso_phi[r], wp.iso_ehi[r], t_sec)) {
                    mid = lo + t_sec * (hi - lo);
                }
                ub_eig.SetValue(row_start + r, mid);
            } else {
                wp.act_q[a_tail] = r; a_tail = (a_tail + 1u) & 63u; ++a_cnt;
            }
        }
    }
    __aicore__ inline void SolveTridiagSturm(float *work) {
        SturmWorkPtrs wp; BindSturmWorkPtrs(work, wp); float gersh_min = 0.0f, gersh_max = 0.0f, max_elem = 0.0f;
        if (this->template InitSturmGershgorin<false>(work, wp, gersh_min, gersh_max, max_elem)) {
            return;
        }
        u32 row_end = 0u, ext_start = 0u, ext_end = 0u; InitNeighborBounds(wp, gersh_min, gersh_max, row_end,
        ext_start, ext_end); float tol_single = 1.0e-5f * max_elem, max_gap_tol = 4.5e-4f * max_elem;
        this->template RunSturmBisectionQueue<false>(wp, gersh_min, gersh_max, 1.0e-6f * max_elem, row_end,
        ext_start, ext_end); RefineNeighborBounds(wp, row_end); float iso_clo[64], iso_chi[64], b_mid[8], b_pval[8];
        i32 b_pexp[8]; u32 a_head = 0u, a_tail = 0u, a_cnt = 0u, b_kp[8], b_pos[8], b_neg[8], r_batch[8];
        InitMediumActiveQueue(wp, row_end, tol_single, max_elem, max_gap_tol, iso_clo, iso_chi, a_tail, a_cnt);
        while (a_cnt > 0u) {
            bool use_8way = (a_cnt >= 5u); u32 bcnt = use_8way ? ((a_cnt < 8u) ? a_cnt : 8u) : ((a_cnt < 4u) ?
            a_cnt : 4u), lane_pos = 0u;
            for (u32 b = 0u; b < bcnt; ++b) {
                u32 r = wp.act_q[a_head]; a_head = (a_head + 1u) & 63u; --a_cnt; r_batch[b] = r; u32 kp = use_8way ?
                ((b < 8u - bcnt) ? 2u : 1u) : ((bcnt == 1u) ? 4u : ((bcnt == 2u) ? 2u : ((bcnt == 3u && b == 0u) ?
                2u : 1u))); b_kp[b] = kp; b_pos[b] = lane_pos; SelectMediumPolyTestPoints(wp, r, kp, lane_pos,
                row_end, iso_clo, iso_chi, max_gap_tol, b_mid); lane_pos += kp;
            }
            EvalPoly4(wp.dd[0], wp.de_pairs, b_mid, b_neg, b_pval, b_pexp);
            if (use_8way) {
                EvalPoly4(wp.dd[0], wp.de_pairs, b_mid + 4u, b_neg + 4u, b_pval + 4u, b_pexp + 4u);
            }
            for (u32 b = 0u; b < bcnt; ++b) {
                UpdateMediumActiveRow(wp, r_batch[b], b_kp[b], b_pos[b], row_end, tol_single, max_elem, max_gap_tol,
                iso_clo, iso_chi, b_mid, b_pval, b_pexp, b_neg, a_tail, a_cnt);
            }
        }
    }
};
} // namespace AscendC
#endif
