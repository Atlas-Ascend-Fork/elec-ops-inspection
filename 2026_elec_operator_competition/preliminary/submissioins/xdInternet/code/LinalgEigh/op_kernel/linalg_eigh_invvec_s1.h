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
#ifndef LINALG_EIGH_INVVEC_S1_H
#define LINALG_EIGH_INVVEC_S1_H
#include "linalg_eigh_tridiag_s5.h"
namespace AscendC {
template <typename DT_A, u32 SOLVE_MODE = 0u, bool FAST200 = false, bool K96_PASS = false>
class KernelEighInvVecS1 : public KernelEighTridiag<DT_A, SOLVE_MODE, FAST200, K96_PASS> {
public:
    using Base = KernelEighTridiag<DT_A, SOLVE_MODE, FAST200, K96_PASS>;
    using Base::ScalarRsqrt, Base::ClampPivot, Base::MaxF, Base::MinF, Base::ComputeLast4Reciprocals, Base::n;
    using Base::stride, Base::ub_vecs, Base::ub_eig, Base::ScalarAbs, Base::ScalarRsqrtFast;
    __aicore__ inline KernelEighInvVecS1() = default;
    __aicore__ inline void ComputeBatch4Gaps(u32 c0, u32 bcnt, float max_elem, float &lam0, float &lam1, float &lam2,
    float &lam3, float &g0_sq, float &g1_sq, float &g2_sq, float &g3_sq, float &min_g, float &gr3) const {
        u32 c1 = (bcnt > 1u) ? (c0 + 1u) : c0, c2 = (bcnt > 2u) ? (c0 + 2u) : c1, c3 = (bcnt > 3u) ? (c0 + 3u) : c2;
        lam0 = ub_eig.GetValue(c0); lam1 = ub_eig.GetValue(c1); lam2 = ub_eig.GetValue(c2); lam3 =
        ub_eig.GetValue(c3); float gl0 = (c0 > 0u) ? (lam0 - ub_eig.GetValue(c0 - 1u)) : max_elem; gr3 = (c0 +
        bcnt < n) ? (ub_eig.GetValue(c0 + bcnt) - lam3) : max_elem; float g01 = (bcnt > 1u) ? (lam1 - lam0) :
        max_elem, g12 = (bcnt > 2u) ? (lam2 - lam1) : max_elem, g23 = (bcnt > 3u) ? (lam3 - lam2) : gr3, g0 =
        MaxF(MinF(gl0, g01), 0.0f), g1 = MaxF(MinF(g01, g12), 0.0f), g2 = MaxF(MinF(g12, g23), 0.0f), g3 =
        MaxF((bcnt > 3u) ? MinF(g23, gr3) : max_elem, 0.0f); g0_sq = g0 * g0; g1_sq = g1 * g1; g2_sq = g2 * g2;
        g3_sq = g3 * g3; min_g = MinF(MinF(g0, g1), MinF(g2, g3));
    }
    __aicore__ inline void ComputeBatch4InvPivots(u8 sm, bool use_batched, float inv_m, float cur_p0, float cur_p1,
    float cur_p2, float cur_p3, float &inv0, float &inv1, float &inv2, float &inv3) const {
        bool sw0 = (sm & 1u) != 0u, sw1 = (sm & 2u) != 0u, sw2 = (sm & 4u) != 0u, sw3 = (sm & 8u) != 0u;
        if (sm == 15u) {
            inv0 = inv_m; inv1 = inv_m; inv2 = inv_m; inv3 = inv_m;
        } else if (use_batched) {
            float p0 = sw0 ? 1.0f : cur_p0, p1 = sw1 ? 1.0f : cur_p1, p2 = sw2 ? 1.0f : cur_p2, p3 = sw3 ? 1.0f :
            cur_p3, p01 = p0 * p1, p23 = p2 * p3, inv_all = 1.0f / (p01 * p23), inv01 = inv_all * p23, inv23 =
            inv_all * p01; inv0 = sw0 ? inv_m : (inv01 * p1); inv1 = sw1 ? inv_m : (inv01 * p0); inv2 = sw2 ? inv_m :
            (inv23 * p3); inv3 = sw3 ? inv_m : (inv23 * p2);
        } else {
            inv0 = sw0 ? inv_m : (1.0f / ClampPivot(cur_p0)); inv1 = sw1 ? inv_m : (1.0f / ClampPivot(cur_p1));
            inv2 = sw2 ? inv_m : (1.0f / ClampPivot(cur_p2)); inv3 = sw3 ? inv_m : (1.0f / ClampPivot(cur_p3));
        }
    }
    static __aicore__ inline void StepElimLane(bool sw, float m_i, float e_next, float nd, float inv, float &cur_p,
    float &cur_u, float &f, float &u_out) {
        f = (sw ? cur_p : m_i) * inv; float u1 = sw ? nd : cur_u; u_out = u1 * inv; cur_p = (sw ? cur_u : nd) -
        f * u1; cur_u = sw ? (-f * e_next) : e_next;
    }
    __aicore__ inline void StepBatch4ElimAndU(u8 sm, float m_i, float e_next, float nd0, float nd1, float nd2,
    float nd3, float inv0, float inv1, float inv2, float inv3, float &cur_p0, float &cur_p1, float &cur_p2,
    float &cur_p3, float &cur_u0, float &cur_u1, float &cur_u2, float &cur_u3, float &f0, float &f1, float &f2,
    float &f3, float *u_out) const {
        StepElimLane((sm & 1u) != 0u, m_i, e_next, nd0, inv0, cur_p0, cur_u0, f0, u_out[0]);
        StepElimLane((sm & 2u) != 0u, m_i, e_next, nd1, inv1, cur_p1, cur_u1, f1, u_out[1]);
        StepElimLane((sm & 4u) != 0u, m_i, e_next, nd2, inv2, cur_p2, cur_u2, f2, u_out[2]);
        StepElimLane((sm & 8u) != 0u, m_i, e_next, nd3, inv3, cur_p3, cur_u3, f3, u_out[3]);
    }
    template <bool USE_SIGN_RHS>
    static __aicore__ inline void StepRhsLane(bool sw, u32 sm_step, float f, float inv, u32 &sm_i, float &nq,
    float &cq, float &y_out) {
        if constexpr (USE_SIGN_RHS) {
            nq = (f * cq > 0.0f) ? -1.375f : 1.375f;
        } else if (sm_step != 0u) {
            sm_i = (sm_i + sm_step >= 23u) ? (sm_i + sm_step - 23u) : (sm_i + sm_step); nq = 1.0f +
            0.03125f * u2f(sm_i);
        } else {
            nq += (nq >= 1.1875f) ? -0.1875f : 0.53125f;
        }
        float qi = sw ? nq : cq; y_out = qi * inv; cq = (sw ? cq : nq) - f * qi;
    }
    template <bool USE_SIGN_RHS>
    __aicore__ inline void StepBatch4RhsUpdate(u8 sm, u32 sm_step, float f0, float f1, float f2, float f3,
    float inv0, float inv1, float inv2, float inv3, u32 &sm0, u32 &sm1, u32 &sm2, u32 &sm3, float &nq0, float &nq1,
    float &nq2, float &nq3, float &cq0, float &cq1, float &cq2, float &cq3, float *y_out) const {
        this->template StepRhsLane<USE_SIGN_RHS>((sm & 1u) != 0u, sm_step, f0, inv0, sm0, nq0, cq0, y_out[0]);
        this->template StepRhsLane<USE_SIGN_RHS>((sm & 2u) != 0u, sm_step, f1, inv1, sm1, nq1, cq1, y_out[1]);
        this->template StepRhsLane<USE_SIGN_RHS>((sm & 4u) != 0u, sm_step, f2, inv2, sm2, nq2, cq2, y_out[2]);
        this->template StepRhsLane<USE_SIGN_RHS>((sm & 8u) != 0u, sm_step, f3, inv3, sm3, nq3, cq3, y_out[3]);
    }
    template <u32 kU, u32 kY>
    __aicore__ inline void StepBacksolve4(const float *s, u8 sm, float u2_s, float &q0_n1, float &q1_n1,
    float &q2_n1, float &q3_n1, float &q0_n2, float &q1_n2, float &q2_n2, float &q3_n2) const {
        float c0_v = (s[kY + 0u] - ((sm & 1u) ? u2_s : 0.0f) * q0_n2) - s[kU + 0u] * q0_n1, c1_v = (s[kY + 1u] -
        ((sm & 2u) ? u2_s : 0.0f) * q1_n2) - s[kU + 1u] * q1_n1, c2_v = (s[kY + 2u] - ((sm & 4u) ? u2_s :
        0.0f) * q2_n2) - s[kU + 2u] * q2_n1, c3_v = (s[kY + 3u] - ((sm & 8u) ? u2_s : 0.0f) * q3_n2) - s[kU +
        3u] * q3_n1; q0_n2 = q0_n1; q0_n1 = c0_v; q1_n2 = q1_n1; q1_n1 = c1_v; q2_n2 = q2_n1; q2_n1 = c2_v; q3_n2 =
        q3_n1; q3_n1 = c3_v;
    }
    __aicore__ inline void FactorizeAndSolveFast1Pass4(u32 r, u32 n_dim, const float *dd, const float *ee,
    const float *inv_ee, const float *ee_rat, float lam0, float lam1, float lam2, float lam3, float *aos,
    u8 *swap_mask) {
        float cq0 = 1.375f, cq1 = 1.375f, cq2 = 1.375f, cq3 = 1.375f, nq0 = 1.375f, nq1 = 1.375f, nq2 = 1.375f, nq3 =
        1.375f, cur_p0 = dd[0] - lam0, cur_p1 = dd[0] - lam1, cur_p2 = dd[0] - lam2, cur_p3 = dd[0] - lam3, cur_u0 =
        ee[0], cur_u1 = ee[0], cur_u2 = ee[0], cur_u3 = ee[0]; u32 dm0 = 0u, dm1 = 0u, dm2 = 0u, dm3 = 0u; float *s =
        aos;
        for (u32 i = 0u; i + 1u < n_dim; ++i, s += 8u) {
            float m_i = ee[i], abs_m = ScalarAbs(m_i), d_next = dd[i + 1u], e_next = ee[i + 1u]; u8 sm =
            u8(((ScalarAbs(cur_p0) < abs_m) ? 1u : 0u) | ((ScalarAbs(cur_p1) < abs_m) ? 2u :
            0u) | ((ScalarAbs(cur_p2) < abs_m) ? 4u : 0u) | ((ScalarAbs(cur_p3) < abs_m) ? 8u : 0u)); swap_mask[i] =
            sm; float inv0, inv1, inv2, inv3, f0, f1, f2, f3; ComputeBatch4InvPivots(sm, true, inv_ee[i], cur_p0,
            cur_p1, cur_p2, cur_p3, inv0, inv1, inv2, inv3); StepBatch4ElimAndU(sm, m_i, e_next, d_next - lam0,
            d_next - lam1, d_next - lam2, d_next - lam3, inv0, inv1, inv2, inv3, cur_p0, cur_p1, cur_p2, cur_p3,
            cur_u0, cur_u1, cur_u2, cur_u3, f0, f1, f2, f3, s); StepBatch4RhsUpdate<true>(sm, 0u, f0, f1, f2, f3,
            inv0, inv1, inv2, inv3, dm0, dm1, dm2, dm3, nq0, nq1, nq2, nq3, cq0, cq1, cq2, cq3, s + 4u);
        }
        float sl[4]; ComputeLast4Reciprocals(cur_p0, cur_p1, cur_p2, cur_p3, sl, true); LTf vr0 = ub_vecs[r * n_dim],
        vr1 = ub_vecs[(r + 1u) * n_dim], vr2 = ub_vecs[(r + 2u) * n_dim], vr3 = ub_vecs[(r + 3u) * n_dim];
        float q0_n1 = cq0 * sl[0], q1_n1 = cq1 * sl[1], q2_n1 = cq2 * sl[2], q3_n1 = cq3 * sl[3];
        vr0.SetValue(n_dim - 1u, q0_n1); vr1.SetValue(n_dim - 1u, q1_n1); vr2.SetValue(n_dim - 1u, q2_n1);
        vr3.SetValue(n_dim - 1u, q3_n1); const float *s_nm2 = aos + (n_dim - 2u) * 8u; float q0_n2 = q0_n1, q1_n2 =
        q1_n1, q2_n2 = q2_n1, q3_n2 = q3_n1; q0_n1 = s_nm2[4] - s_nm2[0] * q0_n2; q1_n1 = s_nm2[5] -
        s_nm2[1] * q1_n2; q2_n1 = s_nm2[6] - s_nm2[2] * q2_n2; q3_n1 = s_nm2[7] - s_nm2[3] * q3_n2;
        vr0.SetValue(n_dim - 2u, q0_n1); vr1.SetValue(n_dim - 2u, q1_n1); vr2.SetValue(n_dim - 2u, q2_n1);
        vr3.SetValue(n_dim - 2u, q3_n1); const float *sb = aos + (n_dim - 3u) * 8u;
        for (i32 i = i32(n_dim) - 3; i >= 0; --i, sb -= 8u) {
            u32 ui = u32(i); this->template StepBacksolve4<0u, 4u>(sb, swap_mask[ui], ee_rat[ui], q0_n1, q1_n1,
            q2_n1, q3_n1, q0_n2, q1_n2, q2_n2, q3_n2); vr0.SetValue(ui, q0_n1); vr1.SetValue(ui, q1_n1);
            vr2.SetValue(ui, q2_n1); vr3.SetValue(ui, q3_n1);
        }
    }
    template <bool SPLIT_SF, bool COMPACT_PIV>
    __aicore__ inline void StoreBatch4FactorPivots(u32 i, u8 sm, float inv0, float inv1, float inv2, float inv3,
    float f0, float f1, float f2, float f3, float *aos, float *s) const {
        if constexpr (SPLIT_SF || COMPACT_PIV) {
            float *dst = SPLIT_SF ? (aos + 2048u + i * 4u) : s; dst[0] = (sm & 1u) ? f0 : inv0; dst[1] = (sm & 2u) ?
            f1 : inv1; dst[2] = (sm & 4u) ? f2 : inv2; dst[3] = (sm & 8u) ? f3 : inv3;
        } else {
            s[0] = inv0; s[1] = inv1; s[2] = inv2; s[3] = inv3; s[4] = f0; s[5] = f1; s[6] = f2; s[7] = f3;
        }
    }
    template <u32 kA, u32 kU, u32 kY, bool USE_SIGN_RHS, bool SPLIT_SF, bool COMPACT_PIV>
    __aicore__ inline void FactorizeBatch4General(const float *dd, const float *ee, const float *inv_ee,
    bool all_e_nz, u32 c0, float lam0, float lam1, float lam2, float lam3, float *aos, u8 *swap_mask, float &cq0,
    float &cq1, float &cq2, float &cq3, bool batched_last_div, u32 sm_step = 0u) const {
        u32 sm0 = (c0 * 13u) % 23u, sm1 = (sm0 + 13u >= 23u) ? (sm0 - 10u) : (sm0 + 13u), sm2 = (sm1 + 13u >= 23u) ?
        (sm1 - 10u) : (sm1 + 13u), sm3 = (sm2 + 13u >= 23u) ? (sm2 - 10u) : (sm2 + 13u); float nq0 = USE_SIGN_RHS ?
        1.375f : (1.0f + 0.03125f * u2f(sm0)), nq1 = USE_SIGN_RHS ? 1.375f : (1.0f + 0.03125f * u2f(sm1)), nq2 =
        USE_SIGN_RHS ? 1.375f : (1.0f + 0.03125f * u2f(sm2)), nq3 = USE_SIGN_RHS ? 1.375f : (1.0f +
        0.03125f * u2f(sm3)); cq0 = nq0; cq1 = nq1; cq2 = nq2; cq3 = nq3; float cur_p0 = dd[0] - lam0, cur_p1 =
        dd[0] - lam1, cur_p2 = dd[0] - lam2, cur_p3 = dd[0] - lam3, cur_u0 = ee[0], cur_u1 = ee[0], cur_u2 = ee[0],
        cur_u3 = ee[0];
        for (u32 i = 0u; i + 1u < n; ++i) {
            float m_i = ee[i], abs_m = ScalarAbs(m_i), d_next = dd[i + 1u], e_next = ee[i + 1u]; u8 sm =
            u8(((ScalarAbs(cur_p0) < abs_m) ? 1u : 0u) | ((ScalarAbs(cur_p1) < abs_m) ? 2u :
            0u) | ((ScalarAbs(cur_p2) < abs_m) ? 4u : 0u) | ((ScalarAbs(cur_p3) < abs_m) ? 8u : 0u)); swap_mask[i] =
            sm; float inv0, inv1, inv2, inv3, f0, f1, f2, f3; bool use_batched = (sm_step == 0u) && (all_e_nz ||
            abs_m >= 1e-7f); ComputeBatch4InvPivots(sm, use_batched, inv_ee[i], cur_p0, cur_p1, cur_p2, cur_p3, inv0,
            inv1, inv2, inv3); float *s = aos + i * kA; StepBatch4ElimAndU(sm, m_i, e_next, d_next - lam0, d_next -
            lam1, d_next - lam2, d_next - lam3, inv0, inv1, inv2, inv3, cur_p0, cur_p1, cur_p2, cur_p3, cur_u0,
            cur_u1, cur_u2, cur_u3, f0, f1, f2, f3, s + kU); StoreBatch4FactorPivots<SPLIT_SF, COMPACT_PIV>(i, sm,
            inv0, inv1, inv2, inv3, f0, f1, f2, f3, aos, s); StepBatch4RhsUpdate<USE_SIGN_RHS>(sm, sm_step, f0, f1,
            f2, f3, inv0, inv1, inv2, inv3, sm0, sm1, sm2, sm3, nq0, nq1, nq2, nq3, cq0, cq1, cq2, cq3, s + kY);
        }
        ComputeLast4Reciprocals(cur_p0, cur_p1, cur_p2, cur_p3, aos + (n - 1u) * kA, batched_last_div);
    }
    template <bool COMPACT_MODE>
    __aicore__ inline void StepForwardElimLane(bool sw, float in_s, float next_y, float sv, float f_or_m,
    float inv_m, float &cq, float &y_out) const {
        float nq = next_y * in_s, qi = sw ? nq : cq, b = sw ? cq : nq;
        if constexpr (COMPACT_MODE) {
            y_out = qi * (sw ? inv_m : sv); cq = b - (sw ? sv : (f_or_m * sv)) * qi;
        } else {
            y_out = qi * sv; cq = b - f_or_m * qi;
        }
    }
    template <u32 kA, u32 kY, bool SPLIT_SF, bool COMPACT_PIV>
    __aicore__ inline void ForwardElimBatch4(const float *ee, const float *inv_ee, const u8 *swap_mask, float *aos,
    float in0, float in1, float in2, float in3, float &cq0, float &cq1, float &cq2, float &cq3) const {
        constexpr bool kCompact = SPLIT_SF || COMPACT_PIV; cq0 = aos[kY + 0u] * in0; cq1 = aos[kY + 1u] * in1; cq2 =
        aos[kY + 2u] * in2; cq3 = aos[kY + 3u] * in3;
        for (u32 i = 0u; i + 1u < n; ++i) {
            u8 sm = swap_mask[i]; float *s = aos + i * kA; const float *sf = SPLIT_SF ? (aos + 2048u + i * 4u) : s;
            float m_i = kCompact ? ee[i] : s[4], inv_m = kCompact ? inv_ee[i] : 0.0f;
            this->template StepForwardElimLane<kCompact>((sm & 1u) != 0u, in0, s[kA + kY + 0u], sf[0], kCompact ?
            m_i : s[4], inv_m, cq0, s[kY + 0u]); this->template StepForwardElimLane<kCompact>((sm & 2u) != 0u, in1,
            s[kA + kY + 1u], sf[1], kCompact ? m_i : s[5], inv_m, cq1, s[kY + 1u]);
            this->template StepForwardElimLane<kCompact>((sm & 4u) != 0u, in2, s[kA + kY + 2u], sf[2], kCompact ?
            m_i : s[6], inv_m, cq2, s[kY + 2u]); this->template StepForwardElimLane<kCompact>((sm & 8u) != 0u, in3,
            s[kA + kY + 3u], sf[3], kCompact ? m_i : s[7], inv_m, cq3, s[kY + 3u]);
        }
    }
    static __aicore__ inline void AccumulateRayleigh4(float d_i, float two_e, float q0_n1, float q1_n1, float q2_n1,
    float q3_n1, float q0_n2, float q1_n2, float q2_n2, float q3_n2, float &rq0, float &rq1, float &rq2, float &rq3,
    float &nsq0, float &nsq1, float &nsq2, float &nsq3) {
        rq0 += q0_n1 * (d_i * q0_n1 + two_e * q0_n2); rq1 += q1_n1 * (d_i * q1_n1 + two_e * q1_n2);
        rq2 += q2_n1 * (d_i * q2_n1 + two_e * q2_n2); rq3 += q3_n1 * (d_i * q3_n1 + two_e * q3_n2);
        nsq0 += q0_n1 * q0_n1; nsq1 += q1_n1 * q1_n1; nsq2 += q2_n1 * q2_n1; nsq3 += q3_n1 * q3_n1;
    }
    template <u32 kA, u32 kU, u32 kY>
    __aicore__ inline void BacksolveNoGsBatch4(u32 r, u32 bcnt, u32 c0, const float *dd, const float *ee,
    const float *ee_rat, const u8 *swap_mask, const float *aos, float q0_n1, float q1_n1, float q2_n1, float q3_n1,
    float nsq0, float nsq1, float nsq2, float nsq3, float *v_scale) {
        u32 c1 = (bcnt > 1u) ? (c0 + 1u) : c0, c2 = (bcnt > 2u) ? (c0 + 2u) : c1, c3 = (bcnt > 3u) ? (c0 + 3u) : c2;
        LTf vr0 = ub_vecs[r * stride], vr1 = ub_vecs[(r + 1u) * stride], vr2 = ub_vecs[(r + 2u) * stride], vr3 =
        ub_vecs[(r + ((bcnt > 3u) ? 3u : 2u)) * stride]; vr0.SetValue(n - 1u, q0_n1); vr1.SetValue(n - 1u, q1_n1);
        vr3.SetValue(n - 1u, q3_n1); vr2.SetValue(n - 1u, q2_n1); float d_last = dd[n - 1u], rq0 = d_last * nsq0,
        rq1 = d_last * nsq1, rq2 = d_last * nsq2, rq3 = d_last * nsq3, q0_n2 = 0.0f, q1_n2 = 0.0f, q2_n2 = 0.0f,
        q3_n2 = 0.0f;
        if (n > 1u) {
            this->template StepBacksolve4<kU, kY>(aos + (n - 2u) * kA, 0u, 0.0f, q0_n1, q1_n1, q2_n1, q3_n1, q0_n2,
            q1_n2, q2_n2, q3_n2); vr0.SetValue(n - 2u, q0_n1); vr1.SetValue(n - 2u, q1_n1); vr3.SetValue(n - 2u,
            q3_n1); vr2.SetValue(n - 2u, q2_n1); AccumulateRayleigh4(dd[n - 2u], 2.0f * ee[n - 2u], q0_n1, q1_n1,
            q2_n1, q3_n1, q0_n2, q1_n2, q2_n2, q3_n2, rq0, rq1, rq2, rq3, nsq0, nsq1, nsq2, nsq3);
        }
        for (i32 i = i32(n) - 3; i >= 0; --i) {
            u32 ui = u32(i); this->template StepBacksolve4<kU, kY>(aos + ui * kA, swap_mask[ui], ee_rat[ui], q0_n1,
            q1_n1, q2_n1, q3_n1, q0_n2, q1_n2, q2_n2, q3_n2); vr0.SetValue(ui, q0_n1); vr1.SetValue(ui, q1_n1);
            vr3.SetValue(ui, q3_n1); vr2.SetValue(ui, q2_n1); AccumulateRayleigh4(dd[ui], 2.0f * ee[ui], q0_n1,
            q1_n1, q2_n1, q3_n1, q0_n2, q1_n2, q2_n2, q3_n2, rq0, rq1, rq2, rq3, nsq0, nsq1, nsq2, nsq3);
        }
        if (nsq0 != 0.0f) {
            ub_eig.SetValue(c0, rq0 / nsq0);
        }
        if (nsq1 != 0.0f) {
            ub_eig.SetValue(c1, rq1 / nsq1);
        }
        if (nsq2 != 0.0f) {
            ub_eig.SetValue(c2, rq2 / nsq2);
        }
        v_scale[r + 0u] = ScalarRsqrt(nsq0); v_scale[r + 1u] = ScalarRsqrt(nsq1); v_scale[r + 2u] = ScalarRsqrt(nsq2);
        if (bcnt > 3u) {
            if (nsq3 != 0.0f) {
                ub_eig.SetValue(c3, rq3 / nsq3);
            }
            v_scale[r + 3u] = ScalarRsqrt(nsq3);
        }
    }
    template <u32 kA, u32 kU, u32 kY>
    __aicore__ inline void BacksolveStoreAosBatch4(const float *ee_rat, const u8 *swap_mask, float *aos, float q0_n1,
    float q1_n1, float q2_n1, float q3_n1, float &nsq0, float &nsq1, float &nsq2, float &nsq3) const {
        float *s_last = aos + (n - 1u) * kA; s_last[kY + 0u] = q0_n1; s_last[kY + 1u] = q1_n1; s_last[kY + 2u] =
        q2_n1; s_last[kY + 3u] = q3_n1; float q0_n2 = 0.0f, q1_n2 = 0.0f, q2_n2 = 0.0f, q3_n2 = 0.0f;
        if (n > 1u) {
            float *s_nm2 = aos + (n - 2u) * kA; this->template StepBacksolve4<kU, kY>(s_nm2, 0u, 0.0f, q0_n1, q1_n1,
            q2_n1, q3_n1, q0_n2, q1_n2, q2_n2, q3_n2); s_nm2[kY + 0u] = q0_n1; s_nm2[kY + 1u] = q1_n1; s_nm2[kY +
            2u] = q2_n1; s_nm2[kY + 3u] = q3_n1; nsq0 += q0_n1 * q0_n1; nsq1 += q1_n1 * q1_n1; nsq2 += q2_n1 * q2_n1;
            nsq3 += q3_n1 * q3_n1;
        }
        for (i32 i = i32(n) - 3; i >= 0; --i) {
            u32 ui = u32(i); float *s = aos + ui * kA; this->template StepBacksolve4<kU, kY>(s, swap_mask[ui],
            ee_rat[ui], q0_n1, q1_n1, q2_n1, q3_n1, q0_n2, q1_n2, q2_n2, q3_n2); s[kY + 0u] = q0_n1; s[kY + 1u] =
            q1_n1; s[kY + 2u] = q2_n1; s[kY + 3u] = q3_n1; nsq0 += q0_n1 * q0_n1; nsq1 += q1_n1 * q1_n1;
            nsq2 += q2_n1 * q2_n1; nsq3 += q3_n1 * q3_n1;
        }
    }
    template <u32 kA>
    __aicore__ inline void OrthogonalizeAosColAgainstPrev(float *aos, u32 col_off, float &in_col, u32 prev_col_off,
    float in_prev, bool use_ub_prev, LTf prev_v) const {
        float dot = 0.0f;
        for (u32 i = 0u; i < n; ++i) {
            float ref_v = use_ub_prev ? prev_v.GetValue(i) : (aos[i * kA + prev_col_off] * in_prev);
            dot += (aos[i * kA + col_off] * in_col) * ref_v;
        }
        float nsq = 0.0f;
        for (u32 i = 0u; i < n; ++i) {
            float ref_v = use_ub_prev ? prev_v.GetValue(i) : (aos[i * kA + prev_col_off] * in_prev), v = aos[i * kA +
            col_off] * in_col - dot * ref_v; aos[i * kA + col_off] = v; nsq += v * v;
        }
        in_col = ScalarRsqrt(nsq);
    }
    template <u32 kA, u32 kY>
    __aicore__ inline void OrthogonalizeAndStoreBatch4(u32 r, u32 bcnt, u32 c0, float max_elem, float lam0,
    float lam1, float lam2, float lam3, const float *dd, const float *ee, float *aos, float nsq0, float nsq1,
    float nsq2, float nsq3) {
        u32 c1 = (bcnt > 1u) ? (c0 + 1u) : c0, c2 = (bcnt > 2u) ? (c0 + 2u) : c1, c3 = (bcnt > 3u) ? (c0 + 3u) : c2;
        float in0 = ScalarRsqrt(nsq0), in1 = ScalarRsqrt(nsq1), in2 = ScalarRsqrt(nsq2), in3 = ScalarRsqrt(nsq3);
        if (r > 0u && ScalarAbs(lam0 - ub_eig.GetValue(c0 - 1u)) < 0.0001f * max_elem) {
            this->template OrthogonalizeAosColAgainstPrev<kA>(aos, kY + 0u, in0, 0u, 1.0f, true, ub_vecs[(r -
            1u) * stride]);
        }
        if (bcnt > 1u && ScalarAbs(lam1 - lam0) < 0.0001f * max_elem) {
            this->template OrthogonalizeAosColAgainstPrev<kA>(aos, kY + 1u, in1, kY + 0u, in0, false, ub_vecs);
        }
        if (bcnt > 2u && ScalarAbs(lam2 - lam1) < 0.0001f * max_elem) {
            this->template OrthogonalizeAosColAgainstPrev<kA>(aos, kY + 2u, in2, kY + 1u, in1, false, ub_vecs);
        }
        if (bcnt > 3u && ScalarAbs(lam3 - lam2) < 0.0001f * max_elem) {
            this->template OrthogonalizeAosColAgainstPrev<kA>(aos, kY + 3u, in3, kY + 2u, in2, false, ub_vecs);
        }
        float rq0 = 0.0f, rq1 = 0.0f, rq2 = 0.0f, rq3 = 0.0f, cur0 = aos[kY + 0u] * in0, cur1 = aos[kY + 1u] * in1,
        cur2 = aos[kY + 2u] * in2, cur3 = aos[kY + 3u] * in3; LTf vr0 = ub_vecs[r * stride], vr1 = ub_vecs[(r +
        1u) * stride], vr2 = ub_vecs[(r + 2u) * stride], vr3 = ub_vecs[(r + ((bcnt > 3u) ? 3u : 2u)) * stride];
        vr0.SetValue(0, cur0); vr1.SetValue(0, cur1); vr3.SetValue(0, cur3); vr2.SetValue(0, cur2);
        for (u32 i = 0u; i + 1u < n; ++i) {
            const float *sn = aos + (i + 1u) * kA; float nxt0 = sn[kY + 0u] * in0, nxt1 = sn[kY + 1u] * in1, nxt2 =
            sn[kY + 2u] * in2, nxt3 = sn[kY + 3u] * in3; vr0.SetValue(i + 1u, nxt0); vr1.SetValue(i + 1u, nxt1);
            vr3.SetValue(i + 1u, nxt3); vr2.SetValue(i + 1u, nxt2); float d_i = dd[i], two_e = 2.0f * ee[i];
            rq0 += cur0 * (d_i * cur0 + two_e * nxt0); rq1 += cur1 * (d_i * cur1 + two_e * nxt1);
            rq2 += cur2 * (d_i * cur2 + two_e * nxt2); rq3 += cur3 * (d_i * cur3 + two_e * nxt3); cur0 = nxt0; cur1 =
            nxt1; cur2 = nxt2; cur3 = nxt3;
        }
        float d_last = dd[n - 1u]; ub_eig.SetValue(c0, rq0 + d_last * cur0 * cur0); ub_eig.SetValue(c1, rq1 +
        d_last * cur1 * cur1); ub_eig.SetValue(c2, rq2 + d_last * cur2 * cur2);
        if (bcnt > 3u) {
            ub_eig.SetValue(c3, rq3 + d_last * cur3 * cur3);
        }
    }
    template <u32 kA, u32 kU, u32 kY>
    __aicore__ inline void BacksolveFast200PktBatch4(u32 r, const float *ee_rat, const u8 *swap_mask,
    const float *aos, float q0_n1, float q1_n1, float q2_n1, float q3_n1, float *v_scale) {
        LTf vr0 = ub_vecs[r * 200u], vr1 = ub_vecs[(r + 1u) * 200u], vr2 = ub_vecs[(r + 2u) * 200u], vr3 =
        ub_vecs[(r + 3u) * 200u]; vr0.SetValue(199u, q0_n1); vr1.SetValue(199u, q1_n1); vr3.SetValue(199u, q3_n1);
        vr2.SetValue(199u, q2_n1); float q0_n2 = 0.0f, q1_n2 = 0.0f, q2_n2 = 0.0f, q3_n2 = 0.0f;
        this->template StepBacksolve4<kU, kY>(aos + 198u * kA, 0u, 0.0f, q0_n1, q1_n1, q2_n1, q3_n1, q0_n2, q1_n2,
        q2_n2, q3_n2); vr0.SetValue(198u, q0_n1); vr1.SetValue(198u, q1_n1); vr3.SetValue(198u, q3_n1);
        vr2.SetValue(198u, q2_n1);
        for (i32 i = 197; i >= 0; --i) {
            u32 ui = u32(i); this->template StepBacksolve4<kU, kY>(aos + ui * kA, swap_mask[ui], ee_rat[ui], q0_n1,
            q1_n1, q2_n1, q3_n1, q0_n2, q1_n2, q2_n2, q3_n2); vr0.SetValue(ui, q0_n1); vr1.SetValue(ui, q1_n1);
            vr3.SetValue(ui, q3_n1); vr2.SetValue(ui, q2_n1);
        }
        this->template StartEvaluateFinalPacket<200u>(r, v_scale);
    }
    template <bool IS_LARGE>
    __aicore__ inline bool CheckEarlyStopBatch4(u32 bcnt, float nsq0, float nsq1, float nsq2, float nsq3,
    float g0_sq, float g1_sq, float g2_sq, float g3_sq, float s0_0, float s0_1, float s0_2, float s0_3) const {
        float s1_0 = nsq0 * g0_sq, s1_1 = (bcnt > 1u) ? (nsq1 * g1_sq) : 1.0e30f, s1_2 = (bcnt > 2u) ?
        (nsq2 * g2_sq) : 1.0e30f, s1_3 = (bcnt > 3u) ? (nsq3 * g3_sq) : 1.0e30f, min_s1 = MinF(MinF(s1_0, s1_1),
        MinF(s1_2, s1_3)), p0 = s0_0 * s1_0, p1 = (!IS_LARGE || bcnt > 1u) ? (s0_1 * s1_1) : 1.0e30f, p2 =
        (!IS_LARGE || bcnt > 2u) ? (s0_2 * s1_2) : 1.0e30f, p3 = (!IS_LARGE || bcnt > 3u) ? (s0_3 * s1_3) : 1.0e30f,
        min_p = MinF(MinF(p0, p1), MinF(p2, p3)); return IS_LARGE ? ((min_s1 >= 4.0e4f && min_p >= 1.0e9f) ||
        (min_s1 >= 8.0e5f && min_p >= 3.5e8f)) : ((min_s1 >= 4.0e4f && min_p >= 6.0e8f) || (min_s1 >= 2.5e6f &&
        min_p >= 5.0e7f));
    }
    template <bool IS_LARGE>
    __aicore__ inline bool UpdateIterAndCheckFinalBatch4(u32 iter, u32 &max_iter, bool no_gs, bool can_early_stop,
    u32 bcnt, float nsq0, float nsq1, float nsq2, float nsq3, float g0_sq, float g1_sq, float g2_sq, float g3_sq,
    float &s0_0, float &s0_1, float &s0_2, float &s0_3) const {
        if (iter == 0u) {
            s0_0 = nsq0 * g0_sq; s0_1 = (bcnt > 1u) ? (nsq1 * g1_sq) : 1.0e30f; s0_2 = (bcnt > 2u) ? (nsq2 * g2_sq) :
            1.0e30f; s0_3 = (bcnt > 3u) ? (nsq3 * g3_sq) : 1.0e30f; max_iter = (no_gs && MinF(MinF(s0_0, s0_1),
            MinF(s0_2, s0_3)) >= 250.0f) ? (2u) : max_iter;
        }
        if (iter + 1u >= max_iter) {
            return true;
        }
        return (iter == 1u && can_early_stop) && CheckEarlyStopBatch4<IS_LARGE>(bcnt, nsq0, nsq1, nsq2, nsq3, g0_sq,
        g1_sq, g2_sq, g3_sq, s0_0, s0_1, s0_2, s0_3);
    }
    template <u32 kA, u32 kU, u32 kY, bool SPLIT_SF, bool COMPACT_PIV, bool IS_LARGE>
    __aicore__ inline bool RunBatch4IterLoop(u32 r, u32 bcnt, u32 c0, float max_elem, float lam0, float lam1,
    float lam2, float lam3, float g0_sq, float g1_sq, float g2_sq, float g3_sq, float min_g, float gr3,
    const float *dd, const float *ee, const float *inv_ee, const float *ee_rat, const u8 *swap_mask, float *aos,
    float cq0, float cq1, float cq2, float cq3, float *v_scale, bool allow_fast200_pkt) {
        const float *s_last = aos + (n - 1u) * kA; bool can_early_stop = (min_g >= (IS_LARGE ? 0.00008f :
        0.00025f) * max_elem), no_gs = (min_g >= 0.0001f * max_elem) && (gr3 >= 0.0001f * max_elem); float s0_0 =
        0.0f, s0_1 = 0.0f, s0_2 = 0.0f, s0_3 = 0.0f; u32 max_iter = 3u; float in0 = 1.0f, in1 = 1.0f, in2 = 1.0f,
        in3 = 1.0f;
        for (u32 iter = 0u; iter < max_iter; ++iter) {
            if (iter > 0u) {
                this->template ForwardElimBatch4<kA, kY, SPLIT_SF, COMPACT_PIV>(ee, inv_ee, swap_mask, aos, in0, in1,
                in2, in3, cq0, cq1, cq2, cq3);
            }
            float q0_n1 = cq0 * s_last[0], q1_n1 = cq1 * s_last[1], q2_n1 = cq2 * s_last[2], q3_n1 = cq3 * s_last[3],
            nsq0 = q0_n1 * q0_n1, nsq1 = q1_n1 * q1_n1, nsq2 = q2_n1 * q2_n1, nsq3 = q3_n1 * q3_n1;
            if (allow_fast200_pkt && n == 200u && iter + 1u >= max_iter && no_gs && bcnt >= 3u) {
                this->template BacksolveFast200PktBatch4<kA, kU, kY>(r, ee_rat, swap_mask, aos, q0_n1, q1_n1, q2_n1,
                q3_n1, v_scale); return true;
            }
            if (iter + 1u >= max_iter && no_gs) {
                this->template BacksolveNoGsBatch4<kA, kU, kY>(r, bcnt, c0, dd, ee, ee_rat, swap_mask, aos, q0_n1,
                q1_n1, q2_n1, q3_n1, nsq0, nsq1, nsq2, nsq3, v_scale); break;
            }
            this->template BacksolveStoreAosBatch4<kA, kU, kY>(ee_rat, swap_mask, aos, q0_n1, q1_n1, q2_n1, q3_n1,
            nsq0, nsq1, nsq2, nsq3);
            if (!UpdateIterAndCheckFinalBatch4<IS_LARGE>(iter, max_iter, no_gs, can_early_stop, bcnt, nsq0, nsq1,
                nsq2, nsq3, g0_sq, g1_sq, g2_sq, g3_sq, s0_0, s0_1, s0_2, s0_3)) {
                in0 = ScalarRsqrtFast(nsq0); in1 = ScalarRsqrtFast(nsq1); in2 = ScalarRsqrtFast(nsq2); in3 =
                ScalarRsqrtFast(nsq3);
            } else {
                this->template OrthogonalizeAndStoreBatch4<kA, kY>(r, bcnt, c0, max_elem, lam0, lam1, lam2, lam3, dd,
                ee, aos, nsq0, nsq1, nsq2, nsq3); break;
            }
        }
        return false;
    }
};
} // namespace AscendC
#endif
