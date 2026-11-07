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
#include "kernel_operator.h"
#include "linalg_eigh_tiling.h"
#include "tiling_key_linalg_eigh.h"
#include "linalg_eigh_invvec_s1.h"
using namespace AscendC;
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
namespace AscendC {
template <typename DT_A, u32 SOLVE_MODE = 0u, bool FAST200 = false, bool K96_PASS = false>
class KernelLinalgEigh : public KernelEighInvVecS1<DT_A, SOLVE_MODE, FAST200, K96_PASS> {
public:
    using Base = KernelEighInvVecS1<DT_A, SOLVE_MODE, FAST200, K96_PASS>;
    using Base::n, Base::stride, Base::uplo, Base::num_groups, Base::lane_id, Base::group_id, Base::coop_row_start;
    using Base::coop_row_count, Base::vec_row_start, Base::vec_row_count, Base::row_start, Base::row_count;
    using Base::batch_size, Base::is_active, Base::matrix_scale, Base::gm_A, Base::gm_eigvals, Base::gm_eigvecs;
    using Base::ub_mat, Base::ub_vecs, Base::ub_row, Base::ub_tmp, Base::ub_diag, Base::ub_eig, Base::ub_gather;
    using Base::ScalarAbs, Base::ScalarRsqrtFast, Base::ScalarRsqrt, Base::ScalarSqrt, Base::ComputePow2Scale;
    using Base::ClampPivot, Base::MaxF, Base::MinF, Base::LoadAndSymmetrizeMedium, Base::LoadAndSymmetrizeCoop;
    using Base::LoadK96Checkpoint, Base::LoadAndSymmetrizeLarge, Base::TridiagonalizeMedium;
    using Base::BacktransformMedium, Base::BacktransformLarge, Base::TridiagonalizeCoop, Base::BacktransformCoop;
    using Base::FactorizeShiftedTridiag1Way, Base::SolveShiftedTridiag1Way, Base::EvalSturmSmall4, Base::RunPWKLoop;
    using Base::TridiagonalizeLarge, Base::SolveTridiagSturmLarge, Base::ComputeEigenvalues, Base::CountDiagRank;
    using Base::PrecomputeOffDiagInvRat, Base::ComputeBatch4Gaps, Base::StepElimLane;
    using Base::FactorizeAndSolveFast1Pass4;
    __aicore__ inline KernelLinalgEigh() = default;
    __aicore__ inline void BacksolveNoGsBatch1(u32 r, u32 c0, const float *dd, const float *ee, const float *ee_rat,
    const u8 *swap_mask, const float *aos, float q0_n1, float nsq0, float *v_scale) {
        LTf vr0 = ub_vecs[r * stride]; vr0.SetValue(n - 1u, q0_n1); float rq0 = dd[n - 1u] * nsq0, q0_n2 = 0.0f;
        if (n > 1u) {
            const float *s_nm2 = aos + (n - 2u) * 4u; q0_n2 = q0_n1; q0_n1 = s_nm2[3] - s_nm2[2] * q0_n2;
            vr0.SetValue(n - 2u, q0_n1); nsq0 += q0_n1 * q0_n1; rq0 += q0_n1 * (dd[n - 2u] * q0_n1 + (2.0f * ee[n -
            2u]) * q0_n2);
        }
        for (i32 i = i32(n) - 3; i >= 0; --i) {
            u32 ui = u32(i); const float *s = aos + ui * 4u; float c0_v = (s[3] - ((swap_mask[ui] != 0u) ?
            ee_rat[ui] : 0.0f) * q0_n2) - s[2] * q0_n1; vr0.SetValue(ui, c0_v); rq0 += c0_v * (dd[ui] * c0_v +
            (2.0f * ee[ui]) * q0_n1); q0_n2 = q0_n1; q0_n1 = c0_v; nsq0 += c0_v * c0_v;
        }
        if (nsq0 != 0.0f) {
            ub_eig.SetValue(c0, rq0 / nsq0);
        }
        v_scale[r] = ScalarRsqrt(nsq0);
    }
    __aicore__ inline void FactorizeBatch1(u32 c0, float lam0, const float *dd, const float *ee, const float *inv_ee,
    float *aos, u8 *swap_mask, float &cq0, float &inv_last) const {
        u32 sm0 = (c0 * 13u) % 23u; float nq0 = 1.0f + 0.03125f * u2f(sm0); cq0 = nq0; float cur_p0 = dd[0] - lam0,
        cur_u0 = ee[0];
        for (u32 i = 0u; i + 1u < n; ++i) {
            float m_i = ee[i], abs_m = ScalarAbs(m_i), nd0 = dd[i + 1u] - lam0, e_next = ee[i + 1u]; bool sw0 =
            (ScalarAbs(cur_p0) < abs_m); swap_mask[i] = sw0 ? 1u : 0u; float inv0 = sw0 ? inv_ee[i] :
            (1.0f / ClampPivot(cur_p0)), f0 = 0.0f; float *s = aos + i * 4u; StepElimLane(sw0, m_i, e_next, nd0,
            inv0, cur_p0, cur_u0, f0, s[2]); s[0] = inv0; s[1] = f0; this->template StepRhsLane<false>(sw0, 0u, f0,
            inv0, sm0, nq0, cq0, s[3]);
        }
        inv_last = 1.0f / ClampPivot(cur_p0);
    }
    __aicore__ inline void OrthogonalizeAndStoreBatch1(u32 r, u32 c0, float max_elem, float lam0, const float *dd,
    const float *ee, float *aos, float nsq0) {
        LTf vr0 = ub_vecs[r * stride]; float in0 = ScalarRsqrt(nsq0);
        if (r > 0u && ScalarAbs(lam0 - ub_eig.GetValue(c0 - 1u)) < 0.0001f * max_elem) {
            this->template OrthogonalizeAosColAgainstPrev<4u>(aos, 3u, in0, 0u, 1.0f, true, ub_vecs[(r -
            1u) * stride]);
        }
        float rq0 = 0.0f, cur0 = aos[3] * in0; vr0.SetValue(0, cur0);
        for (u32 i = 0u; i + 1u < n; ++i) {
            float nxt0 = aos[(i + 1u) * 4u + 3u] * in0; vr0.SetValue(i + 1u, nxt0); rq0 += dd[i] * cur0 * cur0 +
            (2.0f * ee[i]) * cur0 * nxt0; cur0 = nxt0;
        }
        ub_eig.SetValue(c0, rq0 + dd[n - 1u] * cur0 * cur0);
    }
    __aicore__ inline void SolveInvVecsBatch1(u32 r, u32 c0, float max_elem, const float *dd, const float *ee,
    const float *inv_ee, const float *ee_rat, float *aos, u8 *swap_mask, float *v_scale) {
        float lam0 = ub_eig.GetValue(c0), cq0 = 0.0f, inv_last = 0.0f; FactorizeBatch1(c0, lam0, dd, ee, inv_ee, aos,
        swap_mask, cq0, inv_last); float gl0 = (c0 > 0u) ? (lam0 - ub_eig.GetValue(c0 - 1u)) : max_elem, gr0 = (c0 +
        1u < n) ? (ub_eig.GetValue(c0 + 1u) - lam0) : max_elem, g0 = MaxF(MinF(gl0, gr0), 0.0f), g0_sq = g0 * g0,
        s0_0 = 0.0f, in0 = 1.0f; bool can_early_stop = (g0 >= 0.00025f * max_elem), no_gs1 =
        (g0 >= 0.0001f * max_elem) && (gr0 >= 0.0001f * max_elem); u32 max_iter = 3u;
        for (u32 iter = 0u; iter < max_iter; ++iter) {
            if (iter > 0u) {
                cq0 = aos[3] * in0;
                for (u32 i = 0u; i + 1u < n; ++i) {
                    float *s = aos + i * 4u; this->template StepForwardElimLane<false>(swap_mask[i] != 0u, in0, s[7],
                    s[0], s[1], 0.0f, cq0, s[3]);
                }
            }
            float q0_n1 = cq0 * inv_last, nsq0 = q0_n1 * q0_n1, q0_n2 = 0.0f;
            if (iter + 1u >= max_iter && no_gs1) {
                BacksolveNoGsBatch1(r, c0, dd, ee, ee_rat, swap_mask, aos, q0_n1, nsq0, v_scale); break;
            }
            aos[(n - 1u) * 4u + 3u] = q0_n1;
            if (n > 1u) {
                float *s_nm2 = aos + (n - 2u) * 4u; q0_n2 = q0_n1; q0_n1 = s_nm2[3] - s_nm2[2] * q0_n2; s_nm2[3] =
                q0_n1; nsq0 += q0_n1 * q0_n1;
            }
            for (i32 i = i32(n) - 3; i >= 0; --i) {
                u32 ui = u32(i); float *s = aos + ui * 4u, c0_v = (s[3] - ((swap_mask[ui] != 0u) ? ee_rat[ui] :
                0.0f) * q0_n2) - s[2] * q0_n1; s[3] = c0_v; q0_n2 = q0_n1; q0_n1 = c0_v; nsq0 += c0_v * c0_v;
            }
            float d1 = 1.0e30f, d2 = 1.0e30f, d3 = 1.0e30f;
            if (!this->template UpdateIterAndCheckFinalBatch4<false>(iter, max_iter, no_gs1, can_early_stop, 1u,
                nsq0, 1.0e30f,
                1.0e30f, 1.0e30f, g0_sq, 1.0f, 1.0f, 1.0f, s0_0, d1, d2, d3)) {
                in0 = ScalarRsqrtFast(nsq0);
            } else {
                OrthogonalizeAndStoreBatch1(r, c0, max_elem, lam0, dd, ee, aos, nsq0); break;
            }
        }
    }
    __aicore__ inline void FactorizeAndSolveFast1Pass2_256(u32 r, u32 c0, u32 c1, float lam0, float lam1,
    const float *dd, const float *ee, const float *inv_ee, const float *ee_rat, float *aos, u8 *swap_mask,
    float *v_scale, bool &pend_post, u32 pend_r, u32 pend_c0, u32 pend_c1, u32 pend_c2, u32 pend_c3) {
        float cq0 = 1.375f, cq1 = 1.375f, nq0 = 1.375f, nq1 = 1.375f, cur_p0 = dd[0] - lam0, cur_u0 = ee[0], cur_p1 =
        dd[0] - lam1, cur_u1 = ee[0]; u32 dm0 = 0u, dm1 = 0u; float *s = aos;
        for (u32 i = 0u; i < 255u; ++i, s += 4u) {
            float m_i = ee[i], abs_m = ScalarAbs(m_i), d_next = dd[i + 1u], e_next = ee[i + 1u]; bool sw0 =
            (ScalarAbs(cur_p0) < abs_m), sw1 = (ScalarAbs(cur_p1) < abs_m); u8 sm = u8((sw0 ? 1u : 0u) | (sw1 ? 2u :
            0u)); swap_mask[i] = sm; float inv_m = inv_ee[i], p0 = sw0 ? 1.0f : cur_p0, p1 = sw1 ? 1.0f : cur_p1,
            inv01 = (sm == 3u) ? 0.0f : (1.0f / (p0 * p1)), inv0 = sw0 ? inv_m : (inv01 * p1), inv1 = sw1 ? inv_m :
            (inv01 * p0), f0 = 0.0f, f1 = 0.0f; StepElimLane(sw0, m_i, e_next, d_next - lam0, inv0, cur_p0, cur_u0,
            f0, s[0]); StepElimLane(sw1, m_i, e_next, d_next - lam1, inv1, cur_p1, cur_u1, f1, s[1]);
            this->template StepRhsLane<true>(sw0, 0u, f0, inv0, dm0, nq0, cq0, s[2]);
            this->template StepRhsLane<true>(sw1, 0u, f1, inv1, dm1, nq1, cq1, s[3]);
        }
        float sl0 = 1.0f / ClampPivot(cur_p0), sl1 = 1.0f / ClampPivot(cur_p1);
        if (pend_post) {
            this->template FinishEvaluateFinalPacket<256u>(pend_r, 4u, pend_c0, pend_c1, pend_c2, pend_c3, v_scale);
            pend_post = false;
        }
        LTf vr0_2 = ub_vecs[r * 256u], vr1_2 = ub_vecs[(r + 1u) * 256u]; float q0_n1 = cq0 * sl0, q1_n1 = cq1 * sl1;
        vr0_2.SetValue(255u, q0_n1); vr1_2.SetValue(255u, q1_n1); const float *s_nm2 = aos + 254u * 4u; float q0_n2 =
        q0_n1, q1_n2 = q1_n1; q0_n1 = s_nm2[2] - s_nm2[0] * q0_n2; q1_n1 = s_nm2[3] - s_nm2[1] * q1_n2;
        vr0_2.SetValue(254u, q0_n1); vr1_2.SetValue(254u, q1_n1); const float *sb = aos + 253u * 4u;
        for (i32 i = 253; i >= 0; --i, sb -= 4u) {
            u32 ui = u32(i); u8 sm = swap_mask[ui]; float u2_s = ee_rat[ui], c0_v = (sb[2] - ((sm & 1u) ? u2_s :
            0.0f) * q0_n2) - sb[0] * q0_n1, c1_v = (sb[3] - ((sm & 2u) ? u2_s : 0.0f) * q1_n2) - sb[1] * q1_n1;
            vr0_2.SetValue(ui, c0_v); vr1_2.SetValue(ui, c1_v); q0_n2 = q0_n1; q0_n1 = c0_v; q1_n2 = q1_n1; q1_n1 =
            c1_v;
        }
        this->template StartEvaluateFinalPacket<256u>(r, v_scale, 2u);
        this->template FinishEvaluateFinalPacket<256u>(r, 2u, c0, c1, c1, c1, v_scale);
    }
    template <bool USE_256_FAST, bool IS_SMALL>
    __aicore__ inline void ComputeBatch2InvPivots(u8 sm, bool all_e_nz, float abs_m, float inv_m, float cur_p0,
    float cur_p1, float &inv0, float &inv1) const {
        bool sw0 = (sm & 1u) != 0u, sw1 = (sm & 2u) != 0u;
        if (!IS_SMALL && sm == 3u) {
            inv0 = inv_m; inv1 = inv_m;
        } else if (!IS_SMALL && ((USE_256_FAST && all_e_nz) || abs_m >= 1e-7f)) {
            float p0 = sw0 ? 1.0f : cur_p0, p1 = sw1 ? 1.0f : cur_p1, inv01 = 1.0f / (p0 * p1); inv0 = sw0 ? inv_m :
            (inv01 * p1); inv1 = sw1 ? inv_m : (inv01 * p0);
        } else {
            inv0 = sw0 ? inv_m : (1.0f / ClampPivot(cur_p0)); inv1 = sw1 ? inv_m : (1.0f / ClampPivot(cur_p1));
        }
    }
    template <bool USE_256_FAST, bool IS_SMALL>
    __aicore__ inline void FactorizeBatch2(u32 c0, float lam0, float lam1, bool all_e_nz, const float *dd,
    const float *ee, const float *inv_ee, float *aos, u8 *swap_mask, float &cq0, float &cq1) const {
        u32 sm0 = (c0 * 13u) % 23u, sm1 = (sm0 + 13u >= 23u) ? (sm0 - 10u) : (sm0 + 13u); float nq0 = USE_256_FAST ?
        1.375f : (1.0f + 0.03125f * u2f(sm0)), nq1 = USE_256_FAST ? 1.375f : (1.0f + 0.03125f * u2f(sm1)); cq0 = nq0;
        cq1 = nq1; float cur_p0 = dd[0] - lam0, cur_u0 = ee[0], cur_p1 = dd[0] - lam1, cur_u1 = ee[0];
        constexpr u32 kStep = IS_SMALL ? 7u : 0u;
        for (u32 i = 0u; i + 1u < n; ++i) {
            float m_i = ee[i], abs_m = ScalarAbs(m_i), d_next = dd[i + 1u], e_next = ee[i + 1u]; bool sw0 =
            (ScalarAbs(cur_p0) < abs_m), sw1 = (ScalarAbs(cur_p1) < abs_m); u8 sm = u8((sw0 ? 1u : 0u) | (sw1 ? 2u :
            0u)); swap_mask[i] = sm; float inv0 = 0.0f, inv1 = 0.0f, f0 = 0.0f, f1 = 0.0f;
            ComputeBatch2InvPivots<USE_256_FAST, IS_SMALL>(sm, all_e_nz, abs_m, inv_ee[i], cur_p0, cur_p1, inv0,
            inv1); float *s = aos + i * 8u; StepElimLane(sw0, m_i, e_next, d_next - lam0, inv0, cur_p0, cur_u0, f0,
            s[4]); StepElimLane(sw1, m_i, e_next, d_next - lam1, inv1, cur_p1, cur_u1, f1, s[5]); s[0] = inv0; s[1] =
            inv1; s[2] = f0; s[3] = f1; this->template StepRhsLane<USE_256_FAST>(sw0, kStep, f0, inv0, sm0, nq0, cq0,
            s[6]); this->template StepRhsLane<USE_256_FAST>(sw1, kStep, f1, inv1, sm1, nq1, cq1, s[7]);
        }
        float *s_last = aos + (n - 1u) * 8u; s_last[0] = 1.0f / ClampPivot(cur_p0); s_last[1] =
        1.0f / ClampPivot(cur_p1);
    }
    __aicore__ inline void ForwardElimBatch2(const u8 *swap_mask, float *aos, float in0, float in1, float &cq0,
    float &cq1) const {
        cq0 = aos[6] * in0; cq1 = aos[7] * in1;
        for (u32 i = 0u; i + 1u < n; ++i) {
            u8 sm = swap_mask[i]; float *s = aos + i * 8u; this->template StepForwardElimLane<false>((sm & 1u) != 0u,
            in0, s[14], s[0], s[2], 0.0f, cq0, s[6]); this->template StepForwardElimLane<false>((sm & 2u) != 0u, in1,
            s[15], s[1], s[3], 0.0f, cq1, s[7]);
        }
    }
    __aicore__ inline void BacksolveStoreAosBatch2(const float *ee_rat, const u8 *swap_mask, float *aos, float q0_n1,
    float q1_n1, float &nsq0, float &nsq1) const {
        float *s_last = aos + (n - 1u) * 8u; s_last[6] = q0_n1; s_last[7] = q1_n1; float q0_n2 = 0.0f, q1_n2 = 0.0f;
        if (n > 1u) {
            float *s_nm2 = aos + (n - 2u) * 8u; q0_n2 = q0_n1; q1_n2 = q1_n1; q0_n1 = s_nm2[6] - s_nm2[4] * q0_n2;
            q1_n1 = s_nm2[7] - s_nm2[5] * q1_n2; s_nm2[6] = q0_n1; s_nm2[7] = q1_n1; nsq0 += q0_n1 * q0_n1;
            nsq1 += q1_n1 * q1_n1;
        }
        for (i32 i = i32(n) - 3; i >= 0; --i) {
            u32 ui = u32(i); u8 sm = swap_mask[ui]; float u2_s = ee_rat[ui]; float *s = aos + ui * 8u, c0_v = (s[6] -
            ((sm & 1u) ? u2_s : 0.0f) * q0_n2) - s[4] * q0_n1, c1_v = (s[7] - ((sm & 2u) ? u2_s : 0.0f) * q1_n2) -
            s[5] * q1_n1; s[6] = c0_v; s[7] = c1_v; q0_n2 = q0_n1; q0_n1 = c0_v; q1_n2 = q1_n1; q1_n1 = c1_v;
            nsq0 += c0_v * c0_v; nsq1 += c1_v * c1_v;
        }
    }
    __aicore__ inline void BacksolveNoGsBatch2(u32 r, u32 bcnt, bool store_aos_too, const float *dd, const float *ee,
    const float *ee_rat, const u8 *swap_mask, float *aos, float q0_n1, float q1_n1, float &nsq0, float &nsq1,
    float &rq0, float &rq1) {
        float *s_last = aos + (n - 1u) * 8u; LTf vr0 = ub_vecs[r * stride], vr1 = ub_vecs[(r + ((bcnt > 1u) ? 1u :
        0u)) * stride];
        if (store_aos_too) {
            s_last[6] = q0_n1; s_last[7] = q1_n1;
        }
        vr1.SetValue(n - 1u, q1_n1); vr0.SetValue(n - 1u, q0_n1); float d_last = dd[n - 1u], q0_n2 = 0.0f, q1_n2 =
        0.0f; rq0 = d_last * nsq0; rq1 = d_last * nsq1;
        if (n > 1u) {
            float *s_nm2 = aos + (n - 2u) * 8u; q0_n2 = q0_n1; q1_n2 = q1_n1; q0_n1 = s_nm2[6] - s_nm2[4] * q0_n2;
            q1_n1 = s_nm2[7] - s_nm2[5] * q1_n2;
            if (store_aos_too) {
                s_nm2[6] = q0_n1; s_nm2[7] = q1_n1;
            }
            vr1.SetValue(n - 2u, q1_n1); vr0.SetValue(n - 2u, q0_n1); nsq0 += q0_n1 * q0_n1; nsq1 += q1_n1 * q1_n1;
            rq0 += q0_n1 * (dd[n - 2u] * q0_n1 + 2.0f * ee[n - 2u] * q0_n2); rq1 += q1_n1 * (dd[n - 2u] * q1_n1 +
            2.0f * ee[n - 2u] * q1_n2);
        }
        for (i32 i = i32(n) - 3; i >= 0; --i) {
            u32 ui = u32(i); u8 sm = swap_mask[ui]; float u2_s = ee_rat[ui], *s = aos + ui * 8u, c0_v = (s[6] -
            ((sm & 1u) ? u2_s : 0.0f) * q0_n2) - s[4] * q0_n1, c1_v = (s[7] - ((sm & 2u) ? u2_s : 0.0f) * q1_n2) -
            s[5] * q1_n1;
            if (store_aos_too) {
                s[6] = c0_v; s[7] = c1_v;
            }
            vr1.SetValue(ui, c1_v); vr0.SetValue(ui, c0_v); rq0 += c0_v * (dd[ui] * c0_v + 2.0f * ee[ui] * q0_n1);
            rq1 += c1_v * (dd[ui] * c1_v + 2.0f * ee[ui] * q1_n1); q0_n2 = q0_n1; q0_n1 = c0_v; q1_n2 = q1_n1;
            q1_n1 = c1_v; nsq0 += c0_v * c0_v; nsq1 += c1_v * c1_v;
        }
    }
    __aicore__ inline void OrthogonalizeAndStoreBatch2(u32 r, u32 bcnt, u32 c0, u32 c1, float max_elem, float lam0,
    float lam1, const float *dd, const float *ee, float *aos, float nsq0, float nsq1) {
        LTf vr0 = ub_vecs[r * stride], vr1 = ub_vecs[(r + ((bcnt > 1u) ? 1u : 0u)) * stride]; float in0 =
        ScalarRsqrt(nsq0), in1 = ScalarRsqrt(nsq1);
        if (r > 0u && ScalarAbs(lam0 - ub_eig.GetValue(c0 - 1u)) < 0.0001f * max_elem) {
            this->template OrthogonalizeAosColAgainstPrev<8u>(aos, 6u, in0, 0u, 1.0f, true, ub_vecs[(r -
            1u) * stride]);
        }
        if (bcnt > 1u && ScalarAbs(lam1 - lam0) < 0.0001f * max_elem) {
            this->template OrthogonalizeAosColAgainstPrev<8u>(aos, 7u, in1, 6u, in0, false, ub_vecs);
        }
        float rq0 = 0.0f, rq1 = 0.0f, cur0 = aos[6] * in0, cur1 = aos[7] * in1; vr1.SetValue(0, cur1);
        vr0.SetValue(0, cur0);
        for (u32 i = 0u; i + 1u < n; ++i) {
            const float *sn = aos + (i + 1u) * 8u; float nxt0 = sn[6] * in0, nxt1 = sn[7] * in1; vr1.SetValue(i + 1u,
            nxt1); vr0.SetValue(i + 1u, nxt0); float d_i = dd[i], two_e = 2.0f * ee[i]; rq0 += d_i * cur0 * cur0 +
            two_e * cur0 * nxt0; rq1 += d_i * cur1 * cur1 + two_e * cur1 * nxt1; cur0 = nxt0; cur1 = nxt1;
        }
        float d_last = dd[n - 1u]; ub_eig.SetValue(c0, rq0 + d_last * cur0 * cur0);
        if (bcnt > 1u) {
            ub_eig.SetValue(c1, rq1 + d_last * cur1 * cur1);
        }
    }
    __aicore__ inline bool StepFastRem2OrNoGsBatch2(u32 r, u32 bcnt, u32 c0, u32 c1, bool try_fast_rem2,
    const float *dd, const float *ee, const float *ee_rat, const u8 *swap_mask, float *aos, float q0_n1, float q1_n1,
    float g0_sq, float g1_sq, float &s0_0, float &s0_1, u32 &max_iter, float &in0, float &in1, float *v_scale) {
        float nsq0 = q0_n1 * q0_n1, nsq1 = q1_n1 * q1_n1, rq0 = 0.0f, rq1 = 0.0f; BacksolveNoGsBatch2(r, bcnt,
        try_fast_rem2, dd, ee, ee_rat, swap_mask, aos, q0_n1, q1_n1, nsq0, nsq1, rq0, rq1); s0_0 = nsq0 * g0_sq;
        s0_1 = (bcnt > 1u) ? (nsq1 * g1_sq) : 1.0e30f; float min_s0_2 = MinF(s0_0, s0_1);
        if (!try_fast_rem2 || min_s0_2 >= 2.0e7f) {
            if (nsq0 != 0.0f) {
                ub_eig.SetValue(c0, rq0 / nsq0);
            }
            v_scale[r + 0u] = ScalarRsqrt(nsq0);
            if (bcnt > 1u) {
                if (nsq1 != 0.0f) {
                    ub_eig.SetValue(c1, rq1 / nsq1);
                }
                v_scale[r + 1u] = ScalarRsqrt(nsq1);
            }
            return true;
        }
        max_iter = (min_s0_2 >= 250.0f) ? 2u : max_iter; in0 = ScalarRsqrtFast(nsq0); in1 = ScalarRsqrtFast(nsq1);
        return false;
    }
    __aicore__ inline void ComputeBatch2Gaps(u32 c0, u32 c1, u32 bcnt, float max_elem, float &lam0, float &lam1,
    float &g0_sq, float &g1_sq, float &min_g2, bool &no_gs2) const {
        lam0 = ub_eig.GetValue(c0); lam1 = ub_eig.GetValue(c1); float gl0 = (c0 > 0u) ? (lam0 - ub_eig.GetValue(c0 -
        1u)) : max_elem, g01 = (bcnt > 1u) ? (lam1 - lam0) : max_elem, gr1 = (c0 + bcnt < n) ? (ub_eig.GetValue(c0 +
        bcnt) - lam1) : max_elem, g0 = MaxF(MinF(gl0, g01), 0.0f), g1 = MaxF((bcnt > 1u) ? MinF(g01, gr1) : g0,
        0.0f); g0_sq = g0 * g0; g1_sq = g1 * g1; min_g2 = MinF(g0, g1); no_gs2 = (min_g2 >= 0.0001f * max_elem) &&
        (gr1 >= 0.0001f * max_elem);
    }
    template <bool IS_LARGE, bool USE_256_FAST>
    __aicore__ inline void SolveInvVecsBatch2(u32 r, u32 bcnt, u32 c0, float max_elem, bool all_e_nz,
    const float *dd, const float *ee, const float *inv_ee, const float *ee_rat, float *aos, u8 *swap_mask,
    float *v_scale, bool &pend_post, u32 pend_r, u32 pend_c0, u32 pend_c1, u32 pend_c2, u32 pend_c3) {
        if constexpr (!USE_256_FAST) {
            if (bcnt == 1u) {
                SolveInvVecsBatch1(r, c0, max_elem, dd, ee, inv_ee, ee_rat, aos, swap_mask, v_scale); return;
            }
        }
        u32 c1 = (bcnt > 1u) ? (c0 + 1u) : c0; float lam0 = 0.0f, lam1 = 0.0f, g0_sq = 0.0f, g1_sq = 0.0f, min_g2 =
        0.0f; bool no_gs2 = false; ComputeBatch2Gaps(c0, c1, bcnt, max_elem, lam0, lam1, g0_sq, g1_sq, min_g2,
        no_gs2);
        if constexpr (USE_256_FAST) {
            if (n == 256u && bcnt == 2u && no_gs2 && all_e_nz) {
                FactorizeAndSolveFast1Pass2_256(r, c0, c1, lam0, lam1, dd, ee, inv_ee, ee_rat, aos, swap_mask,
                v_scale, pend_post, pend_r, pend_c0, pend_c1, pend_c2, pend_c3); return;
            }
        }
        float cq0 = 0.0f, cq1 = 0.0f; FactorizeBatch2<USE_256_FAST, false>(c0, lam0, lam1, all_e_nz, dd, ee, inv_ee,
        aos, swap_mask, cq0, cq1);
        if constexpr (USE_256_FAST) {
            if (pend_post) {
                this->template FinishEvaluateFinalPacket<256u>(pend_r, 4u, pend_c0, pend_c1, pend_c2, pend_c3,
                v_scale); pend_post = false;
            }
        }
        RunBatch2IterLoop<IS_LARGE, USE_256_FAST>(r, bcnt, c0, c1, max_elem, lam0, lam1, g0_sq, g1_sq, min_g2,
        no_gs2, dd, ee, ee_rat, swap_mask, aos, cq0, cq1, v_scale);
    }
    template <bool IS_LARGE, bool USE_256_FAST>
    __aicore__ inline void RunBatch2IterLoop(u32 r, u32 bcnt, u32 c0, u32 c1, float max_elem, float lam0, float lam1,
    float g0_sq, float g1_sq, float min_g2, bool no_gs2, const float *dd, const float *ee, const float *ee_rat,
    const u8 *swap_mask, float *aos, float cq0, float cq1, float *v_scale) {
        float *s_last = aos + (n - 1u) * 8u, s0_0 = 0.0f, s0_1 = 0.0f, in0 = 1.0f, in1 = 1.0f; bool can_early_stop =
        (min_g2 >= (IS_LARGE ? 0.00008f : 0.00025f) * max_elem); u32 max_iter = 3u;
        for (u32 iter = 0u; iter < max_iter; ++iter) {
            if (iter > 0u) {
                ForwardElimBatch2(swap_mask, aos, in0, in1, cq0, cq1);
            }
            float q0_n1 = cq0 * s_last[0], q1_n1 = cq1 * s_last[1], nsq0 = q0_n1 * q0_n1, nsq1 = q1_n1 * q1_n1;
            bool try_fast_rem2 = USE_256_FAST && (iter == 0u && no_gs2);
            if ((iter + 1u >= max_iter && no_gs2) || try_fast_rem2) {
                if (StepFastRem2OrNoGsBatch2(r, bcnt, c0, c1, try_fast_rem2, dd, ee, ee_rat, swap_mask, aos, q0_n1,
                    q1_n1, g0_sq, g1_sq, s0_0, s0_1, max_iter, in0, in1, v_scale)) {
                    break;
                }
                continue;
            }
            BacksolveStoreAosBatch2(ee_rat, swap_mask, aos, q0_n1, q1_n1, nsq0, nsq1); float d2 = 1.0e30f, d3 =
            1.0e30f;
            if (!this->template UpdateIterAndCheckFinalBatch4<IS_LARGE>(iter, max_iter, no_gs2, can_early_stop, bcnt,
                nsq0, nsq1,
                1.0e30f, 1.0e30f, g0_sq, g1_sq, 1.0f, 1.0f, s0_0, s0_1, d2, d3)) {
                in0 = ScalarRsqrtFast(nsq0); in1 = ScalarRsqrtFast(nsq1);
            } else {
                OrthogonalizeAndStoreBatch2(r, bcnt, c0, c1, max_elem, lam0, lam1, dd, ee, aos, nsq0, nsq1); break;
            }
        }
    }
    __aicore__ inline void ApplyDeferredVecScales(const float *v_scale) {
        this->template fence<HardEvent::S_V>();
        for (u32 ri = 0u; ri < row_count; ++ri) {
            if (v_scale[ri] != 0.0f) {
                Muls(ub_vecs[ri * stride], ub_vecs[ri * stride], v_scale[ri], n);
            }
        }
    }
    __aicore__ inline void StoreInvVec1WayFinal(u32 r, u32 col_idx, float max_elem, float lambda, float norm_sq,
    const float *dd, const float *ee, float *qv) {
        LTf vec_r = ub_vecs[r * stride]; float inv_norm = ScalarRsqrt(norm_sq);
        if (r > 0u && ScalarAbs(lambda - ub_eig.GetValue(col_idx - 1u)) < 0.0001f * max_elem) {
            this->template OrthogonalizeAosColAgainstPrev<1u>(qv, 0u, inv_norm, 0u, 1.0f, true, ub_vecs[(r -
            1u) * stride]);
        }
        float rq = 0.0f, cur = qv[0] * inv_norm;
        for (u32 i = 0u; i + 1u < n; ++i) {
            float nxt = qv[i + 1u] * inv_norm; rq += dd[i] * cur * cur + 2.0f * ee[i] * cur * nxt; vec_r.SetValue(i,
            cur); cur = nxt;
        }
        vec_r.SetValue(n - 1u, cur); ub_eig.SetValue(col_idx, rq + dd[n - 1u] * cur * cur);
    }
    __aicore__ inline void SolveInvVecs1WayTail(u32 r, float max_elem, float wide_gap_thr, const float *dd,
    const float *ee, float *work, u8 *swap_mask) {
        float *pv = work + 1024, *mult = work + 1536, *u1 = work + 2048, *qv = work + 2560;
        for (; r < row_count; ++r) {
            u32 col_idx = row_start + r; float lambda = ub_eig.GetValue(col_idx);
            for (u32 i = 0u; i < n; ++i) {
                qv[i] = 1.0f + 0.03125f * static_cast<float>(i32((i * 17u + col_idx * 13u) % 23u));
            }
            FactorizeShiftedTridiag1Way(dd, ee, lambda, pv, mult, u1, swap_mask); bool left_ok = (col_idx == 0u) ||
            ((lambda - ub_eig.GetValue(col_idx - 1u)) >= wide_gap_thr), right_ok = (col_idx + 1u >= n) ||
            ((ub_eig.GetValue(col_idx + 1u) - lambda) >= wide_gap_thr); u32 max_iter = (left_ok && right_ok) ? 2u :
            3u; float in_scale = 1.0f;
            for (u32 iter = 0u; iter < max_iter; ++iter) {
                float norm_sq = SolveShiftedTridiag1Way(qv, in_scale, ee, pv, mult, u1, swap_mask);
                if (iter + 1u < max_iter) {
                    in_scale = ScalarRsqrtFast(norm_sq);
                } else {
                    StoreInvVec1WayFinal(r, col_idx, max_elem, lambda, norm_sq, dd, ee, qv);
                }
            }
        }
    }
    __aicore__ inline void ComputeInverseVectors(float *work, u8 *swap_mask) {
        float *dd = work, *ee = work + ((n <= 256u) ? 256u : 512u), max_elem = 0.0f;
        if (this->template CheckDiagAndSetIdentityVecs<false>(work, dd, ee, max_elem)) {
            return;
        }
        float v_scale[64];
        for (u32 ri = 0u; ri < row_count; ++ri) {
            v_scale[ri] = 0.0f;
        }
        u32 r = 0u;
        if (n <= 256u) {
            float *ee_rat = work + 512, *inv_ee = work + 768, *aos = work + 1024; bool all_e_nz = true, dummy_pend =
            false; PrecomputeOffDiagInvRat(ee, ee_rat, inv_ee, all_e_nz); constexpr u32 kA = (SOLVE_MODE == 2u) ?
            16u : 12u; constexpr u32 kU = (SOLVE_MODE == 2u) ? 8u : 4u; constexpr u32 kY = (SOLVE_MODE == 2u) ? 12u :
            8u; constexpr bool kCompact = (SOLVE_MODE != 2u);
            for (; r + 2u < row_count; r += 4u) {
                u32 bcnt = ((row_count - r) < 4u) ? (row_count - r) : 4u, c0 = row_start + r; float lam0, lam1, lam2,
                lam3, g0_sq, g1_sq, g2_sq, g3_sq, min_g, gr3, cq0, cq1, cq2, cq3; ComputeBatch4Gaps(c0, bcnt,
                max_elem, lam0, lam1, lam2, lam3, g0_sq, g1_sq, g2_sq, g3_sq, min_g, gr3);
                this->template FactorizeBatch4General<kA, kU, kY, false, false, kCompact>(dd, ee, inv_ee, false, c0,
                lam0, lam1, lam2, lam3, aos, swap_mask, cq0, cq1, cq2, cq3, false);
                this->template RunBatch4IterLoop<kA, kU, kY, false, kCompact, false>(r, bcnt, c0, max_elem, lam0,
                lam1, lam2, lam3, g0_sq, g1_sq, g2_sq, g3_sq, min_g, gr3, dd, ee, inv_ee, ee_rat, swap_mask, aos,
                cq0, cq1, cq2, cq3, v_scale, false);
            }
            for (; r < row_count; r += 2u) {
                u32 bcnt = ((row_count - r) < 2u) ? (row_count - r) : 2u; SolveInvVecsBatch2<false, false>(r, bcnt,
                row_start + r, max_elem, false, dd, ee, inv_ee, ee_rat, aos, swap_mask, v_scale, dummy_pend, 0u, 0u,
                0u, 0u, 0u);
            }
        }
        SolveInvVecs1WayTail(r, max_elem, 0.036f * max_elem, dd, ee, work, swap_mask);
        ApplyDeferredVecScales(v_scale);
    }
    __aicore__ __attribute__((noinline)) void ComputeInverseVectorsLarge200(float *work, u8 *swap_mask) {
        float *dd = work, *ee = work + ((n <= 256u) ? 256u : 512u), max_elem = 0.0f;
        if (this->template CheckDiagAndSetIdentityVecs<true>(work, dd, ee, max_elem)) {
            return;
        }
        float v_scale[32];
        for (u32 ri = 0u; ri < row_count; ++ri) {
            v_scale[ri] = 0.0f;
        }
        u32 r = 0u;
        if (n <= 256u) {
            float *ee_rat = work + 480, *inv_ee = work + 688, *aos = work + 896; bool all_e_nz = true, dummy_pend =
            false; PrecomputeOffDiagInvRat(ee, ee_rat, inv_ee, all_e_nz);
            for (; r + 2u < row_count; r += 4u) {
                u32 bcnt = (row_count - r < 4u) ? (row_count - r) : 4u, c0 = row_start + r; float lam0, lam1, lam2,
                lam3, g0_sq, g1_sq, g2_sq, g3_sq, min_g, gr3, cq0, cq1, cq2, cq3; ComputeBatch4Gaps(c0, bcnt,
                max_elem, lam0, lam1, lam2, lam3, g0_sq, g1_sq, g2_sq, g3_sq, min_g, gr3);
                this->template FactorizeBatch4General<16u, 8u, 12u, false, false, false>(dd, ee, inv_ee, false, c0,
                lam0, lam1, lam2, lam3, aos, swap_mask, cq0, cq1, cq2, cq3, true);
                this->template RunBatch4IterLoop<16u, 8u, 12u, false, false, true>(r, bcnt, c0, max_elem, lam0, lam1,
                lam2, lam3, g0_sq, g1_sq, g2_sq, g3_sq, min_g, gr3, dd, ee, inv_ee, ee_rat, swap_mask, aos, cq0, cq1,
                cq2, cq3, v_scale, false);
            }
            for (; r < row_count; r += 2u) {
                u32 bcnt = ((row_count - r) < 2u) ? (row_count - r) : 2u; SolveInvVecsBatch2<true, false>(r, bcnt,
                row_start + r, max_elem, false, dd, ee, inv_ee, ee_rat, aos, swap_mask, v_scale, dummy_pend, 0u, 0u,
                0u, 0u, 0u);
            }
        }
        ApplyDeferredVecScales(v_scale);
    }
    template <u32 N_DIM>
    __aicore__ inline void InitFinalPacketBuffers(const float *dd, const float *ee) {
        constexpr u32 d_off = (N_DIM == 200u) ? 22144u : 30208u; LTf d_copy_init = ub_mat[d_off], e_copy_init =
        ub_mat[d_off + N_DIM]; LTu shift_idx_init = ub_mat[d_off + 2u * N_DIM].template ReinterpretCast<u32>();
        for (u32 i = 0u; i + 1u < N_DIM; ++i) {
            d_copy_init.SetValue(i, dd[i]); e_copy_init.SetValue(i, 2.0f * ee[i]); shift_idx_init.SetValue(i, (i +
            1u) * 4u);
        }
        d_copy_init.SetValue(N_DIM - 1u, dd[N_DIM - 1u]); e_copy_init.SetValue(N_DIM - 1u, 0.0f);
        shift_idx_init.SetValue(N_DIM - 1u, 0u);
    }
    __aicore__ __attribute__((noinline)) void ComputeInverseVectorsLarge200Fast(float *work, u8 *swap_mask) {
        float *dd = work, *ee = work + ((n <= 256u) ? 256u : 512u), max_elem = 0.0f;
        if (this->template CheckDiagAndSetIdentityVecs<true>(work, dd, ee, max_elem)) {
            return;
        }
        InitFinalPacketBuffers<200u>(dd, ee); bool pend_post = false; u32 pend_r = 0u, pend_bcnt = 0u, pend_c0 = 0u,
        pend_c1 = 0u, pend_c2 = 0u, pend_c3 = 0u, r = 0u; float v_scale[32];
        for (u32 ri = 0u; ri < row_count; ++ri) {
            v_scale[ri] = 0.0f;
        }
        if (n <= 256u) {
            float *ee_rat = work + 480, *inv_ee = work + 688, *aos = work + 896; bool all_e_nz = true;
            PrecomputeOffDiagInvRat(ee, ee_rat, inv_ee, all_e_nz);
            for (; r + 2u < row_count; r += 4u) {
                u32 bcnt = (row_count - r < 4u) ? (row_count - r) : 4u, c0 = row_start + r, c1 = (bcnt > 1u) ? (c0 +
                1u) : c0, c2 = (bcnt > 2u) ? (c0 + 2u) : c1, c3 = (bcnt > 3u) ? (c0 + 3u) : c2; float lam0, lam1,
                lam2, lam3, g0_sq, g1_sq, g2_sq, g3_sq, min_g, gr3, cq0, cq1, cq2, cq3; ComputeBatch4Gaps(c0, bcnt,
                max_elem, lam0, lam1, lam2, lam3, g0_sq, g1_sq, g2_sq, g3_sq, min_g, gr3);
                if (pend_post) {
                    this->template FinishEvaluateFinalPacket<200u>(pend_r, pend_bcnt, pend_c0, pend_c1, pend_c2,
                    pend_c3, v_scale); pend_post = false;
                }
                if (n == 200u && bcnt >= 3u && min_g >= 0.00025f * max_elem && gr3 >= 0.00025f * max_elem &&
                    all_e_nz) {
                    FactorizeAndSolveFast1Pass4(r, 200u, dd, ee, inv_ee, ee_rat, lam0, lam1, lam2, lam3, aos,
                    swap_mask); this->template StartEvaluateFinalPacket<200u>(r, v_scale); pend_post = true;
                } else {
                    this->template FactorizeBatch4General<16u, 8u, 12u, false, false, false>(dd, ee, inv_ee, false,
                    c0, lam0, lam1, lam2, lam3, aos, swap_mask, cq0, cq1, cq2, cq3, true); pend_post =
                    this->template RunBatch4IterLoop<16u, 8u, 12u, false, false, true>(r, bcnt, c0, max_elem, lam0,
                    lam1, lam2, lam3, g0_sq, g1_sq, g2_sq, g3_sq, min_g, gr3, dd, ee, inv_ee, ee_rat, swap_mask, aos,
                    cq0, cq1, cq2, cq3, v_scale, true);
                }
                if (pend_post) {
                    pend_r = r; pend_bcnt = bcnt; pend_c0 = c0; pend_c1 = c1; pend_c2 = c2; pend_c3 = c3;
                }
            }
        }
        if (pend_post) {
            this->template FinishEvaluateFinalPacket<200u>(pend_r, pend_bcnt, pend_c0, pend_c1, pend_c2, pend_c3,
            v_scale);
        }
        ApplyDeferredVecScales(v_scale);
    }
    __aicore__ __attribute__((noinline)) void ComputeInverseVectorsLarge256(float *work, u8 *swap_mask) {
        float *dd = work, *ee = work + ((n <= 256u) ? 256u : 512u), max_elem = 0.0f;
        if (this->template CheckDiagAndSetIdentityVecs<true>(work, dd, ee, max_elem)) {
            return;
        }
        InitFinalPacketBuffers<256u>(dd, ee); bool pend_post = false; u32 pend_r = 0u, pend_c0 = 0u, pend_c1 = 0u,
        pend_c2 = 0u, pend_c3 = 0u, r = 0u; float v_scale[32];
        for (u32 ri = 0u; ri < row_count; ++ri) {
            v_scale[ri] = 0.0f;
        }
        if (n <= 256u) {
            float *ee_rat = work + 512, *inv_ee = work + 768, *aos = work + 1024; bool all_e_nz = true;
            for (PrecomputeOffDiagInvRat(ee, ee_rat, inv_ee, all_e_nz); r + 2u < row_count; r += 4u) {
                u32 bcnt = (row_count - r < 4u) ? (row_count - r) : 4u, c0 = row_start + r; float lam0, lam1, lam2,
                lam3, g0_sq, g1_sq, g2_sq, g3_sq, min_g, gr3, cq0, cq1, cq2, cq3; ComputeBatch4Gaps(c0, bcnt,
                max_elem, lam0, lam1, lam2, lam3, g0_sq, g1_sq, g2_sq, g3_sq, min_g, gr3);
                if (pend_post) {
                    this->template FinishEvaluateFinalPacket<256u>(pend_r, 4u, pend_c0, pend_c1, pend_c2, pend_c3,
                    v_scale); pend_post = false;
                }
                if (n == 256u && bcnt == 4u && min_g >= 0.0001f * max_elem && gr3 >= 0.0001f * max_elem && all_e_nz) {
                    FactorizeAndSolveFast1Pass4(r, 256u, dd, ee, inv_ee, ee_rat, lam0, lam1, lam2, lam3, aos,
                    swap_mask); this->template StartEvaluateFinalPacket<256u>(r, v_scale); pend_post = true; pend_r =
                    r; pend_c0 = c0; pend_c1 = c0 + 1u; pend_c2 = c0 + 2u; pend_c3 = c0 + 3u;
                } else {
                    this->template FactorizeBatch4General<8u, 0u, 4u, true, true, false>(dd, ee, inv_ee, all_e_nz,
                    c0, lam0, lam1, lam2, lam3, aos, swap_mask, cq0, cq1, cq2, cq3, true);
                    this->template RunBatch4IterLoop<8u, 0u, 4u, true, false, true>(r, bcnt, c0, max_elem, lam0,
                    lam1, lam2, lam3, g0_sq, g1_sq, g2_sq, g3_sq, min_g, gr3, dd, ee, inv_ee, ee_rat, swap_mask, aos,
                    cq0, cq1, cq2, cq3, v_scale, false);
                }
            }
            for (; r < row_count; r += 2u) {
                u32 bcnt = ((row_count - r) < 2u) ? (row_count - r) : 2u; SolveInvVecsBatch2<true, true>(r, bcnt,
                row_start + r, max_elem, all_e_nz, dd, ee, inv_ee, ee_rat, aos, swap_mask, v_scale, pend_post,
                pend_r, pend_c0, pend_c1, pend_c2, pend_c3);
            }
        }
        if (pend_post) {
            this->template FinishEvaluateFinalPacket<256u>(pend_r, 4u, pend_c0, pend_c1, pend_c2, pend_c3, v_scale);
        }
        ApplyDeferredVecScales(v_scale);
    }
    __aicore__ inline void ComputeInverseVectorsLarge(float *work, u8 *swap_mask) {
        if constexpr (FAST200 && !K96_PASS) {
            ComputeInverseVectorsLarge200Fast(work, swap_mask); return;
        }
        (n <= 200u) ? (void)ComputeInverseVectorsLarge200(work, swap_mask) :
        (void)ComputeInverseVectorsLarge256(work, swap_mask);
    }
    __aicore__ inline void SymmetrizeAndScaleSmallInput(float *a_loc, bool &pre_scaled) {
        for (u32 i = 0u; i < n; ++i) {
            for (u32 j = i; j < n; ++j) {
                u32 src_idx = (uplo != 0u) ? (i * n + j) : (j * n + i); a_loc[i * 32u + j] = ub_mat.GetValue(src_idx);
            }
        }
        float p_a0 = a_loc[0], p_a1 = a_loc[1], p_a33 = a_loc[33], probe_sq = p_a0 * p_a0 + p_a1 * p_a1 +
        p_a33 * p_a33; pre_scaled = false; matrix_scale = 1.0f;
        if (probe_sq >= 0.0625f && probe_sq <= 16.0f) {
            return;
        }
        float max_val = 0.0f;
        for (u32 i = 0u; i < n; ++i) {
            for (u32 j = i; j < n; ++j) {
                max_val = MaxF(max_val, ScalarAbs(a_loc[i * 32u + j]));
            }
        }
        if (max_val > 0.0f && (max_val < 0.25f || max_val > 4.0f)) {
            float inv_scale = 1.0f; ComputePow2Scale(max_val, matrix_scale, inv_scale);
            for (u32 i = 0u; i < n; ++i) {
                for (u32 j = i; j < n; ++j) {
                    a_loc[i * 32u + j] *= inv_scale;
                }
            }
            pre_scaled = true;
        }
    }
    __aicore__ inline void LoadAndTridiagSmallScalar(u64 b, float *a_loc, float *dd, float *ee, float *tt, float *ww,
    float *orig_d, float *orig_e, float *ee_sq, bool &is_diag) {
        u32 total_elems = n * n; u8 rpad = u8(((total_elems + 7u) & ~7u) - total_elems); DataCopyPad(ub_mat,
        gm_A[b * total_elems], DCEP{1, total_elems * 4u, 0, 0, 0}, DCPEP{true, 0, rpad, 0.0f});
        this->template fence<HardEvent::MTE2_S>(); bool pre_scaled = false; SymmetrizeAndScaleSmallInput(a_loc,
        pre_scaled); TridiagonalizeSmallScalarLoop(a_loc, dd, ee, tt, ww);
        if (!pre_scaled) {
            float max_val = 0.0f;
            for (u32 i = 0u; i < n; ++i) {
                max_val = MaxF(max_val, ScalarAbs(dd[i])); max_val = (i + 1u < n) ? (MaxF(max_val,
                ScalarAbs(ee[i]))) : max_val;
            }
            if (max_val > 0.0f && (max_val < 0.125f || max_val > 16.0f)) {
                float inv_scale = 1.0f; ComputePow2Scale(max_val, matrix_scale, inv_scale);
                for (u32 i = 0u; i < n; ++i) {
                    dd[i] *= inv_scale; ee[i] *= inv_scale;
                }
            }
        }
        is_diag = true;
        for (u32 i = 0u; i < n; ++i) {
            orig_d[i] = dd[i]; orig_e[i] = ee[i]; ee_sq[i] = ee[i] * ee[i]; is_diag = is_diag && !(i + 1u < n &&
            ee[i] != 0.0f);
        }
    }
    __aicore__ inline void SymMatVec2WaySmall(u32 k, const float *a_k, const float *a_loc, float *ww,
    float &vw_dot) const {
        u32 i_mv = k + 2u;
        for (; i_mv + 1u < n; i_mv += 2u) {
            const float *a_r0 = a_loc + i_mv * 32u, *a_r1 = a_r0 + 32u; float vi0 = a_k[i_mv], vi1 = a_k[i_mv + 1u],
            a01 = a_r0[i_mv + 1u], c0 = ww[i_mv] + a_r0[i_mv] * vi0 + a01 * vi1, c1 = ww[i_mv + 1u] + a01 * vi0 +
            a_r1[i_mv + 1u] * vi1;
            for (u32 j = i_mv + 2u; j < n; ++j) {
                float vj = a_k[j], v0j = a_r0[j], v1j = a_r1[j]; c0 += v0j * vj; c1 += v1j * vj; ww[j] += v0j * vi0 +
                v1j * vi1;
            }
            ww[i_mv] = c0; ww[i_mv + 1u] = c1; vw_dot += vi0 * c0 + vi1 * c1;
        }
        for (; i_mv < n; ++i_mv) {
            const float *a_row = a_loc + i_mv * 32u; float vi = a_k[i_mv], acc = ww[i_mv] + a_row[i_mv] * vi;
            for (u32 j = i_mv + 1u; j < n; ++j) {
                acc += a_row[j] * a_k[j]; ww[j] += a_row[j] * vi;
            }
            ww[i_mv] = acc; vw_dot += vi * acc;
        }
    }
    __aicore__ inline void Rank2UpdateSmallScalar(u32 k, float v0, float vw_dot, const float *a_k, float *a_loc,
    float *ww, float &next_sq_sum) const {
        float alpha = -0.5f * vw_dot, w1 = ww[k + 1u] + alpha * v0, nsq = 0.0f; ww[k + 1u] = w1; float *a_row1 =
        a_loc + (k + 1u) * 32u; a_row1[k + 1u] -= 2.0f * v0 * w1;
        for (u32 j = k + 2u; j < n; ++j) {
            float vj = a_k[j], wj = ww[j] + alpha * vj; ww[j] = wj; float val = a_row1[j] - (v0 * wj + w1 * vj);
            a_row1[j] = val; nsq += val * val;
        }
        next_sq_sum = nsq;
        for (u32 i_r2 = k + 2u; i_r2 < n; ++i_r2) {
            float *a_row = a_loc + i_r2 * 32u, vi = a_k[i_r2], wi = ww[i_r2];
            for (u32 j = i_r2; j < n; ++j) {
                a_row[j] -= (vi * ww[j] + wi * a_k[j]);
            }
        }
    }
    __aicore__ inline void TridiagonalizeSmallScalarLoop(float *a_loc, float *dd, float *ee, float *tt,
    float *ww) const {
        float next_sq_sum = -1.0f;
        for (u32 k = 0u; k < n; ++k) {
            float *a_k = a_loc + k * 32u; dd[k] = a_k[k]; ee[k] = 0.0f; tt[k] = 0.0f;
            if (k + 2u >= n) {
                ee[k] = (k + 1u < n) ? (a_k[k + 1u]) : ee[k]; continue;
            }
            float sq_sum = next_sq_sum;
            if (sq_sum < 0.0f) {
                sq_sum = 0.0f;
                for (u32 j = k + 1u; j < n; ++j) {
                    sq_sum += a_k[j] * a_k[j];
                }
            }
            float norm = ScalarSqrt(sq_sum);
            if (norm == 0.0f) {
                continue;
            }
            float x0 = a_k[k + 1u], beta = (x0 >= 0.0f) ? -norm : norm; ee[k] = beta; tt[k] = 1.0f; float inv_s =
            ScalarRsqrt(sq_sum - beta * x0), v0 = (x0 - beta) * inv_s; a_k[k + 1u] = v0; const float *a_row0 =
            a_loc + (k + 1u) * 32u; float acc0 = a_row0[k + 1u] * v0;
            for (u32 j = k + 2u; j < n; ++j) {
                float vj = a_k[j] * inv_s, a_ij = a_row0[j]; a_k[j] = vj; acc0 += a_ij * vj; ww[j] = a_ij * v0;
            }
            ww[k + 1u] = acc0; float vw_dot = v0 * acc0; SymMatVec2WaySmall(k, a_k, a_loc, ww, vw_dot);
            Rank2UpdateSmallScalar(k, v0, vw_dot, a_k, a_loc, ww, next_sq_sum);
        }
    }
    __aicore__ inline void NarrowSmallSturmBracket4(u32 target, const float *b_mid, const u32 *b_neg,
    const float *b_pval, float &lo, float &hi, float &plo, float &phi, u32 &lc, u32 &hc, u32 k_start = 0u) const {
        for (u32 k = k_start; k < 4u; ++k) {
            if (b_neg[k] < target) {
                lo = b_mid[k]; plo = b_pval[k]; lc = b_neg[k];
            } else {
                hi = b_mid[k]; phi = b_pval[k]; hc = b_neg[k]; break;
            }
        }
    }
    static __aicore__ inline float InterpSmallSturmRoot(float lo, float hi, float plo, float phi, u32 lc, u32 hc) {
        return (hc - lc == 1u && plo >= 0.0f && phi >= 0.0f && (plo + phi) > 1e-30f) ? (lo + (plo / (plo +
        phi)) * (hi - lo)) : (0.5f * (lo + hi));
    }
    __aicore__ inline float RefineSmallSturmRoot4(u32 target, u32 max_it, u32 min_it, const float *orig_d,
    const float *ee_sq, float lo, float hi, float plo, float phi, u32 lc, u32 hc) const {
        float b_mid[4], b_pval[4]; u32 b_neg[4];
        for (u32 iter = 0u; iter < max_it && !(iter >= min_it && (hc - lc <= 1u)); ++iter) {
            float sp = (hi - lo) * 0.2f; b_mid[0] = lo + sp; b_mid[1] = lo + 2.0f * sp; b_mid[2] = lo + 3.0f * sp;
            b_mid[3] = lo + 4.0f * sp; EvalSturmSmall4(orig_d, ee_sq, b_mid, b_neg, b_pval);
            NarrowSmallSturmBracket4(target, b_mid, b_neg, b_pval, lo, hi, plo, phi, lc, hc);
        }
        return InterpSmallSturmRoot(lo, hi, plo, phi, lc, hc);
    }
    __aicore__ inline bool TrySeparatePrevSmallSturm(u32 t0, const float *orig_d, const float *ee_sq, float &lo0,
    float &hi0, float &plo0, float &phi0, u32 &lc0, u32 &hc0, float &lo_p, float &hi_p_min, float &plo_p,
    float &phi_p_min, u32 &lc_p, u32 &hc_p_min, bool &prev_separated) const {
        float b_mid[4], b_pval[4], s0 = (hi0 - lo0) * 0.25f; u32 b_neg[4]; b_mid[0] = lo0 - 0.0015f; b_mid[1] = lo0 +
        s0; b_mid[2] = lo0 + 2.0f * s0; b_mid[3] = lo0 + 3.0f * s0; EvalSturmSmall4(orig_d, ee_sq, b_mid, b_neg,
        b_pval);
        if (b_neg[0] >= row_start) {
            prev_separated = true; hi_p_min = b_mid[0]; phi_p_min = b_pval[0]; hc_p_min = b_neg[0];
        } else {
            lo_p = b_mid[0]; plo_p = b_pval[0]; lc_p = b_neg[0];
        }
        NarrowSmallSturmBracket4(t0, b_mid, b_neg, b_pval, lo0, hi0, plo0, phi0, lc0, hc0, 1u); return prev_separated;
    }
    __aicore__ inline void UpdateSmallSturmIterBounds(u32 iter, u32 t0, u32 t1, float g_max, bool check_prev,
    const float *b_mid, const u32 *b_neg, const float *b_pval, float &lo_n_min, float &lo1, float &hi1, float &plo1,
    float &phi1, u32 &lc1, u32 &hc1, float &lo_p, float &hi_p_min, float &plo_p, float &phi_p_min, u32 &lc_p,
    u32 &hc_p_min) const {
        for (u32 k = 0u; k < 4u; ++k) {
            lo_n_min = (b_neg[k] > t0) ? MinF(lo_n_min, b_mid[k]) : lo_n_min;
            if (iter == 0u && row_count > 1u) {
                if (b_neg[k] < t1) {
                    lo1 = b_mid[k]; plo1 = b_pval[k]; lc1 = b_neg[k];
                } else if (hi1 == g_max) {
                    hi1 = b_mid[k]; phi1 = b_pval[k]; hc1 = b_neg[k];
                }
            }
            if (check_prev && b_neg[k] < row_start && b_mid[k] > lo_p) {
                lo_p = b_mid[k]; plo_p = b_pval[k]; lc_p = b_neg[k];
            } else if (check_prev && b_neg[k] >= row_start && b_mid[k] < hi_p_min) {
                hi_p_min = b_mid[k]; phi_p_min = b_pval[k]; hc_p_min = b_neg[k];
            }
        }
    }
    __aicore__ inline void InitSmallGershgorinBounds(const float *orig_d, const float *orig_e, float &g_min,
    float &g_max) const {
        g_min = 0.0f; g_max = 0.0f;
        for (u32 i = 0u; i < n; ++i) {
            float rad = ((i > 0u) ? ScalarAbs(orig_e[i - 1u]) : 0.0f) + ((i + 1u < n) ? ScalarAbs(orig_e[i]) : 0.0f);
            g_min = (i == 0u) ? (orig_d[i] - rad) : MinF(g_min, orig_d[i] - rad); g_max = (i == 0u) ? (orig_d[i] +
            rad) : MaxF(g_max, orig_d[i] + rad);
        }
        float pad = 1e-5f * (ScalarAbs(g_min) + ScalarAbs(g_max)) + 1e-6f; g_min -= pad; g_max += pad;
    }
    __aicore__ inline void SolveSmallEigenvaluesSturm2(float *dd, float *orig_d, const float *orig_e,
    const float *ee_sq) const {
        float g_min = 0.0f, g_max = 0.0f; InitSmallGershgorinBounds(orig_d, orig_e, g_min, g_max); u32 t0 =
        row_start + 1u, t1 = (row_count > 1u) ? (row_start + 2u) : t0; float lo0 = g_min, hi0 = g_max, plo0 = -1.0f,
        phi0 = -1.0f, lo1 = g_min, hi1 = g_max, plo1 = -1.0f, phi1 = -1.0f, lo_p = g_min, hi_p_min = g_max, plo_p =
        -1.0f, phi_p_min = -1.0f, lo_n_min = g_max; u32 lc0 = 0u, hc0 = n, lc1 = 0u, hc1 = n, lc_p = 0u, hc_p_min =
        n; bool check_prev = (row_start > 0u), prev_separated = !check_prev; float b_mid[4], b_pval[4]; u32 b_neg[4];
        const u32 min_sturm = (n == 16u) ? 4u : 5u, max_sturm = (n == 16u) ? 5u : 6u;
        for (u32 iter = 0u; iter < 8u; ++iter) {
            bool iso_done = (iter >= min_sturm) && (hc0 - lc0 <= 1u); prev_separated = prev_separated || (iso_done &&
            check_prev && (lo0 - hi_p_min >= 0.0015f));
            if (iso_done && iter == min_sturm && !prev_separated && (hi_p_min - lo_p > 0.0015f)) {
                if (TrySeparatePrevSmallSturm(t0, orig_d, ee_sq, lo0, hi0, plo0, phi0, lc0, hc0, lo_p, hi_p_min,
                    plo_p, phi_p_min, lc_p, hc_p_min, prev_separated)) {
                    break;
                }
                continue;
            }
            if (iso_done && iter >= max_sturm) {
                break;
            }
            float s0 = (hi0 - lo0) * 0.2f; b_mid[0] = lo0 + s0; b_mid[1] = lo0 + 2.0f * s0; b_mid[2] = lo0 +
            3.0f * s0; b_mid[3] = lo0 + 4.0f * s0; EvalSturmSmall4(orig_d, ee_sq, b_mid, b_neg, b_pval);
            UpdateSmallSturmIterBounds(iter, t0, t1, g_max, check_prev, b_mid, b_neg, b_pval, lo_n_min, lo1, hi1,
            plo1, phi1, lc1, hc1, lo_p, hi_p_min, plo_p, phi_p_min, lc_p, hc_p_min); NarrowSmallSturmBracket4(t0,
            b_mid, b_neg, b_pval, lo0, hi0, plo0, phi0, lc0, hc0);
        }
        FinalizeSmallSturmRoots(t1, check_prev, prev_separated, lo_n_min, orig_d, ee_sq, lo0, hi0, plo0, phi0, lc0,
        hc0, lo1, hi1, plo1, phi1, lc1, hc1, lo_p, hi_p_min, plo_p, phi_p_min, lc_p, hc_p_min, dd);
    }
    __aicore__ inline void FinalizeSmallSturmRoots(u32 t1, bool check_prev, bool prev_separated, float lo_n_min,
    float *orig_d, const float *ee_sq, float lo0, float hi0, float plo0, float phi0, u32 lc0, u32 hc0, float lo1,
    float hi1, float plo1, float phi1, u32 lc1, u32 hc1, float lo_p, float hi_p_min, float plo_p, float phi_p_min,
    u32 lc_p, u32 hc_p_min, float *dd) const {
        float eig0 = InterpSmallSturmRoot(lo0, hi0, plo0, phi0, lc0, hc0); dd[row_start] = eig0; orig_d[31] =
        ((prev_separated || (eig0 - hi_p_min) >= 0.015f) && (row_start + 1u >= n || (lo_n_min - eig0) >= 0.015f)) ?
        1.0f : 0.0f;
        if (check_prev) {
            dd[row_start - 1u] = (!prev_separated && (eig0 - hi_p_min) < 0.0015f) ? RefineSmallSturmRoot4(row_start,
            5u, 4u, orig_d, ee_sq, lo_p, hi_p_min, plo_p, phi_p_min, lc_p, hc_p_min) : -1.0e30f;
        }
        if (row_count > 1u) {
            dd[row_start + 1u] = RefineSmallSturmRoot4(t1, 9u, 5u, orig_d, ee_sq, lo1, hi1, plo1, phi1, lc1, hc1);
        }
    }
    __aicore__ inline void FactorizeSmall1Way(u32 c0, float lam0, const float *orig_d, const float *orig_e,
    float *vv, float *aos, u8 *swap_mask, float &cq0, float &inv_last) const {
        u32 sm0 = (c0 * 13u) % 23u; float nq0 = 1.0f + 0.03125f * u2f(sm0); cq0 = nq0; float cur_p0 = orig_d[0] -
        lam0, cur_u0 = orig_e[0];
        for (u32 i = 0u; i + 1u < n; ++i) {
            float m_i = orig_e[i], abs_m = ScalarAbs(m_i), nd0 = orig_d[i + 1u] - lam0, e_next = orig_e[i + 1u];
            bool sw0 = (ScalarAbs(cur_p0) < abs_m); swap_mask[i] = sw0 ? 1u : 0u; float inv0 = 1.0f / (sw0 ? m_i :
            ClampPivot(cur_p0)), f0 = 0.0f; float *s = aos + i * 4u; StepElimLane(sw0, m_i, e_next, nd0, inv0,
            cur_p0, cur_u0, f0, s[2]); s[0] = inv0; s[1] = f0; s[3] = sw0 ? (e_next * inv0) : 0.0f;
            this->template StepRhsLane<false>(sw0, 7u, f0, inv0, sm0, nq0, cq0, vv[i]);
        }
        inv_last = 1.0f / ClampPivot(cur_p0);
    }
    __aicore__ inline void ApplySmallReflector1Way(u32 uk, const float *a_loc, const float *tt, float *vv) const {
        if (tt[uk] == 0.0f) {
            return;
        }
        const float *hk = a_loc + uk * 32u; float d0 = 0.0f;
        for (u32 i = uk + 1u; i < n; ++i) {
            d0 += hk[i] * vv[i];
        }
        for (u32 i = uk + 1u; i < n; ++i) {
            vv[i] -= d0 * hk[i];
        }
    }
    __aicore__ inline void BacktransformAndStoreSmall1Way(u32 r0, u32 c0, u32 tile_stride, float in0,
    const float *a_loc, const float *tt, float *vv) {
        i32 k_bt = i32(n) - 3;
        for (; k_bt >= 1; k_bt -= 2) {
            u32 uk = u32(k_bt);
            if (tt[uk] != 0.0f && tt[uk - 1u] != 0.0f) {
                const float *hk1 = a_loc + uk * 32u, *hk0 = a_loc + (uk - 1u) * 32u; float d1 = 0.0f, d0 = 0.0f, ov =
                0.0f;
                for (u32 i = uk + 1u; i < n; ++i) {
                    d1 += hk1[i] * vv[i]; d0 += hk0[i] * vv[i]; ov += hk0[i] * hk1[i];
                }
                float h0_uk = hk0[uk]; d0 = (d0 + h0_uk * vv[uk]) - d1 * ov; vv[uk] -= d0 * h0_uk;
                for (u32 i = uk + 1u; i < n; ++i) {
                    vv[i] -= d1 * hk1[i] + d0 * hk0[i];
                }
            } else {
                ApplySmallReflector1Way(uk, a_loc, tt, vv); ApplySmallReflector1Way(uk - 1u, a_loc, tt, vv);
            }
        }
        for (; k_bt >= 0; --k_bt) {
            ApplySmallReflector1Way(u32(k_bt), a_loc, tt, vv);
        }
        u32 mi0 = 0u; float ma0 = ScalarAbs(vv[0]);
        for (u32 i = 1u; i < n; ++i) {
            if (ScalarAbs(vv[i]) > ma0) {
                ma0 = ScalarAbs(vv[i]); mi0 = i;
            }
        }
        float sc0 = (vv[mi0] < 0.0f) ? -in0 : in0; LTf dst = (row_count == n) ? ub_vecs : ub_gather; u32 dst_st =
        (row_count == n) ? n : tile_stride, dst_col = (row_count == n) ? c0 : r0;
        for (u32 i = 0u; i < n; ++i) {
            dst.SetValue(i * dst_st + dst_col, vv[i] * sc0);
        }
    }
    __aicore__ inline void SolveSmall1WayFast(u32 r0, u32 c0, u32 tile_stride, const float *a_loc, const float *dd,
    const float *orig_d, const float *orig_e, const float *tt, float *vv, float *aos, u8 *swap_mask) {
        float cq0 = 0.0f, inv_last = 0.0f, in0 = 1.0f, s0_small = 0.0f; FactorizeSmall1Way(c0, dd[c0], orig_d,
        orig_e, vv, aos, swap_mask, cq0, inv_last);
        for (u32 iter = 0u; iter < 3u; ++iter) {
            if (iter > 0u) {
                cq0 = vv[0] * in0;
                for (u32 i = 0u; i + 1u < n; ++i) {
                    const float *s = aos + i * 4u; this->template StepForwardElimLane<false>(swap_mask[i] != 0u, in0,
                    vv[i + 1u], s[0], s[1], 0.0f, cq0, vv[i]);
                }
            }
            float q0_n1 = cq0 * inv_last, nsq0 = q0_n1 * q0_n1, q0_n2 = 0.0f; vv[n - 1u] = q0_n1;
            if (n > 1u) {
                const float *s_nm2 = aos + (n - 2u) * 4u; q0_n2 = q0_n1; q0_n1 = vv[n - 2u] - s_nm2[2] * q0_n2;
                vv[n - 2u] = q0_n1; nsq0 += q0_n1 * q0_n1;
            }
            for (i32 i = i32(n) - 3; i >= 0; --i) {
                u32 ui = u32(i); const float *s = aos + ui * 4u; float c0_v = (vv[ui] - s[3] * q0_n2) - s[2] * q0_n1;
                vv[ui] = c0_v; q0_n2 = q0_n1; q0_n1 = c0_v; nsq0 += c0_v * c0_v;
            }
            s0_small = (iter == 0u) ? (nsq0) : s0_small; float prod_s = s0_small * nsq0; bool cont_16 =
            (orig_d[31] == 0.0f && (nsq0 < 1.0e8f || prod_s < 1.0e15f)) || (s0_small < 1.0e5f && prod_s < 1.0e10f),
            cont_gt16 = (orig_d[31] == 0.0f) || (s0_small < 5.0e6f && prod_s < 1.0e13f);
            if (iter == 0u || (iter == 1u && ((n == 16u) ? cont_16 : cont_gt16))) {
                in0 = ScalarRsqrtFast(nsq0);
            } else {
                in0 = ScalarRsqrt(nsq0); float rq0 = orig_d[n - 1u] * vv[n - 1u] * vv[n - 1u];
                for (u32 i = 0u; i + 1u < n; ++i) {
                    rq0 += orig_d[i] * vv[i] * vv[i] + (2.0f * orig_e[i]) * vv[i] * vv[i + 1u];
                }
                if (nsq0 != 0.0f) {
                    ub_row.SetValue(r0, (rq0 / nsq0) * matrix_scale);
                }
                break;
            }
        }
        BacktransformAndStoreSmall1Way(r0, c0, tile_stride, in0, a_loc, tt, vv);
    }
    __aicore__ inline void SolveSmallPrevRef1Way(u32 c_prev, const float *dd, const float *orig_d,
    const float *orig_e, const float *ee_sq, float *vv, float *sp_aos, float &inp) const {
        float cqp = 0.0f, inv_lp = 0.0f; u8 sw_p[32]; FactorizeSmall1Way(c_prev, dd[c_prev], orig_d, orig_e, vv,
        sp_aos, sw_p, cqp, inv_lp); inp = 1.0f;
        for (u32 it_p = 0u; it_p < 3u; ++it_p) {
            if (it_p > 0u) {
                cqp = vv[0] * inp;
                for (u32 i = 0u; i + 1u < n; ++i) {
                    const float *sp = sp_aos + i * 4u; this->template StepForwardElimLane<false>(sw_p[i] != 0u, inp,
                    vv[i + 1u], sp[0], sp[1], 0.0f, cqp, vv[i]);
                }
            }
            float qp_n1 = cqp * inv_lp, nsqp = qp_n1 * qp_n1, qp_n2 = 0.0f; vv[n - 1u] = qp_n1;
            if (n > 1u) {
                const float *sp_nm2 = sp_aos + (n - 2u) * 4u; qp_n2 = qp_n1; qp_n1 = vv[n - 2u] - sp_nm2[2] * qp_n2;
                vv[n - 2u] = qp_n1; nsqp += qp_n1 * qp_n1;
            }
            for (i32 i = i32(n) - 3; i >= 0; --i) {
                u32 ui = u32(i); const float *sp = sp_aos + ui * 4u; float cp_v = (vv[ui] - ((sw_p[ui] != 0u) ?
                ee_sq[ui] : 0.0f) * qp_n2) - sp[2] * qp_n1; vv[ui] = cp_v; qp_n2 = qp_n1; qp_n1 = cp_v;
                nsqp += cp_v * cp_v;
            }
            inp = (it_p + 1u < 3u) ? ScalarRsqrtFast(nsqp) : ScalarRsqrt(nsqp);
        }
    }
    __aicore__ inline void BacktransformAndStoreSmall2Way(u32 r0, u32 bcnt, u32 c0_eff, u32 c1, bool need_orth_prev,
    u32 tile_stride, const float *a_loc, const float *tt, float *aos) {
        for (i32 k = i32(n) - 3; k >= 0; --k) {
            u32 uk = u32(k);
            if (tt[uk] != 0.0f) {
                const float *hk = a_loc + uk * 32u; float d0 = 0.0f, d1 = 0.0f;
                for (u32 i = uk + 1u; i < n; ++i) {
                    d0 += hk[i] * aos[i * 8u + 6u]; d1 += hk[i] * aos[i * 8u + 7u];
                }
                for (u32 i = uk + 1u; i < n; ++i) {
                    aos[i * 8u + 6u] -= d0 * hk[i]; aos[i * 8u + 7u] -= d1 * hk[i];
                }
            }
        }
        LTf dst = (row_count == n) ? ub_vecs : ub_gather; u32 dst_st = (row_count == n) ? n : tile_stride;
        for (u32 b = 0u; b < bcnt; ++b) {
            u32 v_col = (bcnt == 1u && need_orth_prev) ? 7u : (6u + b), dst_col = (row_count == n) ? ((bcnt == 1u) ?
            c1 : (c0_eff + b)) : (r0 + b), mi = 0u; float nsq = 0.0f, ma = ScalarAbs(aos[v_col]);
            for (u32 i = 0u; i < n; ++i) {
                float v = aos[i * 8u + v_col]; nsq += v * v;
                if (ScalarAbs(v) > ma) {
                    ma = ScalarAbs(v); mi = i;
                }
            }
            float sc = ((aos[mi * 8u + v_col] < 0.0f) ? -1.0f : 1.0f) * ScalarRsqrt(nsq);
            for (u32 i = 0u; i < n; ++i) {
                dst.SetValue(i * dst_st + dst_col, aos[i * 8u + v_col] * sc);
            }
        }
    }
    __aicore__ inline void OrthogonalizeSmall2WayCol0(u32 c_prev, const float *dd, const float *orig_d,
    const float *orig_e, const float *ee_sq, float *vv, float *aos, float &in0) const {
        float inp = 1.0f, dot0 = 0.0f, nsq0 = 0.0f; SolveSmallPrevRef1Way(c_prev, dd, orig_d, orig_e, ee_sq, vv,
        aos + 256u, inp);
        for (u32 i = 0u; i < n; ++i) {
            dot0 += (aos[i * 8u + 6u] * in0) * (vv[i] * inp);
        }
        for (u32 i = 0u; i < n; ++i) {
            float v = aos[i * 8u + 6u] * in0 - dot0 * (vv[i] * inp); aos[i * 8u + 6u] = v; nsq0 += v * v;
        }
        in0 = ScalarRsqrt(nsq0);
    }
    __aicore__ inline void SolveSmall2Way(u32 r0, u32 bcnt, u32 c0_eff, u32 c1, bool need_orth_prev, u32 tile_stride,
    const float *a_loc, const float *dd, const float *orig_d, const float *orig_e, const float *tt, const float *ww,
    const float *ee_sq, float *vv, float *aos, u8 *swap_mask) {
        float lam0 = dd[c0_eff], lam1 = dd[c1], cq0 = 0.0f, cq1 = 0.0f, in0 = 1.0f, in1 = 1.0f;
        FactorizeBatch2<false, true>(c0_eff, lam0, lam1, false, orig_d, orig_e, ww, aos, swap_mask, cq0, cq1);
        float *s_last = aos + (n - 1u) * 8u;
        for (u32 iter = 0u; iter < 3u; ++iter) {
            if (iter > 0u) {
                ForwardElimBatch2(swap_mask, aos, in0, in1, cq0, cq1);
            }
            float q0_n1 = cq0 * s_last[0], q1_n1 = cq1 * s_last[1], nsq0 = q0_n1 * q0_n1, nsq1 = q1_n1 * q1_n1;
            BacksolveStoreAosBatch2(ee_sq, swap_mask, aos, q0_n1, q1_n1, nsq0, nsq1);
            if (iter + 1u < 3u) {
                in0 = ScalarRsqrtFast(nsq0); in1 = ScalarRsqrtFast(nsq1);
            } else {
                in0 = ScalarRsqrt(nsq0); in1 = ScalarRsqrt(nsq1);
                if (bcnt == 2u && c0_eff > 0u && ScalarAbs(lam0 - dd[c0_eff - 1u]) < 0.008f) {
                    OrthogonalizeSmall2WayCol0(c0_eff - 1u, dd, orig_d, orig_e, ee_sq, vv, aos, in0);
                }
                if ((bcnt > 1u || need_orth_prev) && ScalarAbs(lam1 - lam0) < 0.008f) {
                    this->template OrthogonalizeAosColAgainstPrev<8u>(aos, 7u, in1, 6u, in0, false, ub_vecs);
                }
                float rq0 = 0.0f, rq1 = 0.0f, cur0 = aos[6] * in0, cur1 = aos[7] * in1; aos[6] = cur0; aos[7] = cur1;
                for (u32 i = 0u; i + 1u < n; ++i) {
                    float *sn = aos + (i + 1u) * 8u, nxt0 = sn[6] * in0, nxt1 = sn[7] * in1, d_i = orig_d[i], two_e =
                    2.0f * orig_e[i]; sn[6] = nxt0; sn[7] = nxt1; rq0 += d_i * cur0 * cur0 + two_e * cur0 * nxt0;
                    rq1 += d_i * cur1 * cur1 + two_e * cur1 * nxt1; cur0 = nxt0; cur1 = nxt1;
                }
                float d_last = orig_d[n - 1u]; rq0 += d_last * cur0 * cur0; rq1 += d_last * cur1 * cur1;
                ub_row.SetValue(r0, (need_orth_prev ? rq1 : rq0) * matrix_scale);
                if (bcnt > 1u) {
                    ub_row.SetValue(r0 + 1u, rq1 * matrix_scale);
                }
            }
        }
        BacktransformAndStoreSmall2Way(r0, bcnt, c0_eff, c1, need_orth_prev, tile_stride, a_loc, tt, aos);
    }
    __aicore__ inline void BacktransformAndStoreSmall4Way(u32 r0, u32 bcnt, u32 c0, u32 tile_stride,
    const float *a_loc, const float *tt, float *aos) {
        for (i32 k = i32(n) - 3; k >= 0; --k) {
            u32 uk = u32(k);
            if (tt[uk] != 0.0f) {
                const float *hk = a_loc + uk * 32u; float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                for (u32 i = uk + 1u; i < n; ++i) {
                    const float *s = aos + i * 16u; d0 += hk[i] * s[12]; d1 += hk[i] * s[13]; d2 += hk[i] * s[14];
                    d3 += hk[i] * s[15];
                }
                for (u32 i = uk + 1u; i < n; ++i) {
                    float *s = aos + i * 16u; s[12] -= d0 * hk[i]; s[13] -= d1 * hk[i]; s[14] -= d2 * hk[i];
                    s[15] -= d3 * hk[i];
                }
            }
        }
        LTf dst = (row_count == n) ? ub_vecs : ub_gather; u32 dst_st = (row_count == n) ? n : tile_stride, dst_base =
        (row_count == n) ? c0 : r0;
        for (u32 b = 0u; b < bcnt; ++b) {
            u32 mi = 0u; float ma = ScalarAbs(aos[12u + b]);
            for (u32 i = 1u; i < n; ++i) {
                float av = ScalarAbs(aos[i * 16u + 12u + b]);
                if (av > ma) {
                    ma = av; mi = i;
                }
            }
            float sc = (aos[mi * 16u + 12u + b] < 0.0f) ? -1.0f : 1.0f;
            for (u32 i = 0u; i < n; ++i) {
                dst.SetValue(i * dst_st + dst_base + b, aos[i * 16u + 12u + b] * sc);
            }
        }
    }
    __aicore__ inline void SolveSmall4Way(u32 r0, u32 bcnt, u32 c0, u32 tile_stride, const float *a_loc,
    const float *dd, const float *orig_d, const float *orig_e, const float *tt, const float *ww, const float *ee_sq,
    float *aos, u8 *swap_mask) {
        u32 c1 = (bcnt > 1u) ? (c0 + 1u) : c0, c2 = (bcnt > 2u) ? (c0 + 2u) : c1, c3 = (bcnt > 3u) ? (c0 + 3u) : c2;
        float lam0 = dd[c0], lam1 = dd[c1], lam2 = dd[c2], lam3 = dd[c3], cq0, cq1, cq2, cq3;
        this->template FactorizeBatch4General<16u, 8u, 12u, false, false, false>(orig_d, orig_e, ww, false, c0, lam0,
        lam1, lam2, lam3, aos, swap_mask, cq0, cq1, cq2, cq3, false, 17u); const float *s_last = aos + (n -
        1u) * 16u; float in0 = 1.0f, in1 = 1.0f, in2 = 1.0f, in3 = 1.0f;
        for (u32 iter = 0u; iter < 2u; ++iter) {
            if (iter > 0u) {
                this->template ForwardElimBatch4<16u, 12u, false, false>(orig_e, ww, swap_mask, aos, in0, in1, in2,
                in3, cq0, cq1, cq2, cq3);
            }
            float q0_n1 = cq0 * s_last[0], q1_n1 = cq1 * s_last[1], q2_n1 = cq2 * s_last[2], q3_n1 = cq3 * s_last[3],
            nsq0 = q0_n1 * q0_n1, nsq1 = q1_n1 * q1_n1, nsq2 = q2_n1 * q2_n1, nsq3 = q3_n1 * q3_n1;
            this->template BacksolveStoreAosBatch4<16u, 8u, 12u>(ee_sq, swap_mask, aos, q0_n1, q1_n1, q2_n1, q3_n1,
            nsq0, nsq1, nsq2, nsq3); in0 = (iter < 1u) ? ScalarRsqrtFast(nsq0) : ScalarRsqrt(nsq0); in1 =
            (iter < 1u) ? ScalarRsqrtFast(nsq1) : ScalarRsqrt(nsq1); in2 = (iter < 1u) ? ScalarRsqrtFast(nsq2) :
            ScalarRsqrt(nsq2); in3 = (iter < 1u) ? ScalarRsqrtFast(nsq3) : ScalarRsqrt(nsq3);
        }
        float rq0 = 0.0f, rq1 = 0.0f, rq2 = 0.0f, rq3 = 0.0f, cur0 = aos[12] * in0, cur1 = aos[13] * in1, cur2 =
        aos[14] * in2, cur3 = aos[15] * in3; aos[12] = cur0; aos[13] = cur1; aos[14] = cur2; aos[15] = cur3;
        for (u32 i = 0u; i + 1u < n; ++i) {
            float *sn = aos + (i + 1u) * 16u, nxt0 = sn[12] * in0, nxt1 = sn[13] * in1, nxt2 = sn[14] * in2, nxt3 =
            sn[15] * in3; sn[12] = nxt0; sn[13] = nxt1; sn[14] = nxt2; sn[15] = nxt3; float d_i = orig_d[i], two_e =
            2.0f * orig_e[i]; rq0 += d_i * cur0 * cur0 + two_e * cur0 * nxt0; rq1 += d_i * cur1 * cur1 +
            two_e * cur1 * nxt1; rq2 += d_i * cur2 * cur2 + two_e * cur2 * nxt2; rq3 += d_i * cur3 * cur3 +
            two_e * cur3 * nxt3; cur0 = nxt0; cur1 = nxt1; cur2 = nxt2; cur3 = nxt3;
        }
        float dl = orig_d[n - 1u]; u32 r_idx[4] = {r0, r0 + 1u, r0 + 2u, r0 + 3u}; float r_eigs[4] = {rq0 +
        dl * cur0 * cur0, rq1 + dl * cur1 * cur1, rq2 + dl * cur2 * cur2, rq3 + dl * cur3 * cur3};
        for (u32 bi = 0u; bi < bcnt; ++bi) {
            ub_row.SetValue(r_idx[bi], r_eigs[bi] * matrix_scale);
        }
        BacktransformAndStoreSmall4Way(r0, bcnt, c0, tile_stride, a_loc, tt, aos);
    }
    __aicore__ inline void StoreSmallDiagBatch(u32 r0, u32 bcnt, u32 tile_stride, const float *dd,
    const float *orig_d) {
        LTf dst = (row_count == n) ? ub_vecs : ub_gather; u32 dst_st = (row_count == n) ? n : tile_stride;
        for (u32 bi = 0u; bi < bcnt; ++bi) {
            u32 r = r0 + bi, col_idx = row_start + r, dst_col = (row_count == n) ? col_idx : r; ub_row.SetValue(r,
            dd[col_idx] * matrix_scale);
            for (u32 i = 0u; i < n; ++i) {
                dst.SetValue(i * dst_st + dst_col, (CountDiagRank(orig_d, i) == col_idx) ? 1.0f : 0.0f);
            }
        }
    }
    __aicore__ inline void PrecomputeSmallOffDiagRat(bool is_diag, float *dd, float *orig_d, const float *orig_e,
    float *ww, float *ee_sq) const {
        (!is_diag && n <= 18u && row_count <= 2u) ? (void)SolveSmallEigenvaluesSturm2(dd, orig_d, orig_e, ee_sq) :
        (void)RunPWKLoop(dd, ee_sq, false);
        if (row_count <= 1u || is_diag) {
            return;
        }
        for (u32 i = 0u; i + 1u < n; ++i) {
            float inv_e = (orig_e[i] != 0.0f) ? (1.0f / orig_e[i]) : 0.0f; ww[i] = inv_e; ee_sq[i] = orig_e[i +
            1u] * inv_e;
        }
    }
    __aicore__ inline void SolveSmallBatch(u64 b, float *work, u8 *swap_mask) {
        row_start = vec_row_start; row_count = vec_row_count;
        if (row_count == 0u) {
            return;
        }
        u32 a_cap = (n <= 16u) ? 512u : 1024u; float *a_loc = work, *dd = work + a_cap, *ee = dd + 32, *tt = ee + 32,
        *vv = tt + 32; float *ww = vv + 32, *orig_d = ww + 32, *orig_e = orig_d + 32, *ee_sq = orig_e + 32, *aos =
        work + a_cap + 256u; bool is_diag = true; LoadAndTridiagSmallScalar(b, a_loc, dd, ee, tt, ww, orig_d, orig_e,
        ee_sq, is_diag); PrecomputeSmallOffDiagRat(is_diag, dd, orig_d, orig_e, ww, ee_sq); u32 tile_stride =
        (row_count + 7u) & ~7u;
        for (u32 r0 = 0u; r0 < row_count; r0 += 4u) {
            u32 bcnt = ((row_count - r0) < 4u) ? (row_count - r0) : 4u, c0 = row_start + r0;
            if (is_diag) {
                StoreSmallDiagBatch(r0, bcnt, tile_stride, dd, orig_d); continue;
            }
            bool need_orth_prev = (bcnt == 1u && c0 > 0u && ScalarAbs(dd[c0] - dd[c0 - 1u]) < 0.0015f);
            if (need_orth_prev) {
                for (u32 i = 0u; i + 1u < n; ++i) {
                    ee_sq[i] = (orig_e[i] != 0.0f) ? (orig_e[i + 1u] / orig_e[i]) : 0.0f;
                }
            }
            if (bcnt == 1u && !need_orth_prev) {
                SolveSmall1WayFast(r0, c0, tile_stride, a_loc, dd, orig_d, orig_e, tt, vv, aos, swap_mask);
            } else if (bcnt <= 2u) {
                u32 c0_eff = need_orth_prev ? (c0 - 1u) : c0, c1 = need_orth_prev ? c0 : ((bcnt > 1u) ? (c0 + 1u) :
                c0); SolveSmall2Way(r0, bcnt, c0_eff, c1, need_orth_prev, tile_stride, a_loc, dd, orig_d, orig_e, tt,
                ww, ee_sq, vv, aos, swap_mask);
            } else {
                SolveSmall4Way(r0, bcnt, c0, tile_stride, a_loc, dd, orig_d, orig_e, tt, ww, ee_sq, aos, swap_mask);
            }
        }
        CopySmallBatchOutputs(b);
    }
    __aicore__ inline void CopySmallBatchOutputs(u64 b) {
        u32 total_elems = n * n; this->template fence<HardEvent::S_MTE3>(); (row_count == n) ?
        (void)DataCopyPad(gm_eigvecs[b * total_elems], ub_vecs, DCEP{1, total_elems * 4u, 0, 0, 0}) :
        (void)DataCopyPad(gm_eigvecs[b * total_elems + row_start], ub_gather, DCEP{u16(n), row_count * 4u, 0, (n -
        row_count) * 4u, 0}); DataCopyPad(gm_eigvals[b * n + row_start], ub_row, DCEP{1, row_count * 4u, 0, 0, 0});
        this->template fence<HardEvent::MTE3_S>();
    }
    __aicore__ inline void FindFastStripScanRange(u32 r, LTf strip_max_base, u32 max_strip_cnt, u32 strip_step,
    u32 aligned_cnt, u32 &scan_start, u32 &scan_end) const {
        u32 best_s = 0u; float best_m = strip_max_base.GetValue(r);
        for (u32 s = 1u; s < max_strip_cnt; ++s) {
            float m_s = strip_max_base.GetValue(s * aligned_cnt + r);
            if (m_s > best_m) {
                best_m = m_s; best_s = s;
            }
        }
        scan_start = best_s * strip_step; scan_end = (scan_start + strip_step < n) ? (scan_start + strip_step) : n;
    }
    template <bool SCALE_NORM>
    __aicore__ inline void CanonicalizeAndScaleRowVecs(bool fast_strip_max, LTf strip_max_base, u32 max_strip_cnt,
    u32 strip_step, u32 aligned_cnt) {
        for (u32 r = 0u; r < row_count; ++r) {
            LTf sq_r = ub_mat[r * stride], vec_r = ub_vecs[r * stride]; u32 scan_start = 0u, scan_end = n;
            if (fast_strip_max) {
                FindFastStripScanRange(r, strip_max_base, max_strip_cnt, strip_step, aligned_cnt, scan_start,
                scan_end);
            }
            u32 max_idx = scan_start; float max_sq = sq_r.GetValue(scan_start);
            for (u32 i = scan_start + 1u; i < scan_end; ++i) {
                float v_sq = sq_r.GetValue(i);
                if (v_sq > max_sq) {
                    max_sq = v_sq; max_idx = i;
                }
            }
            if constexpr (!SCALE_NORM) {
                if (vec_r.GetValue(max_idx) < 0.0f) {
                    Muls(vec_r, vec_r, -1.0f, n);
                }
            } else {
                float sign_scale = ((vec_r.GetValue(max_idx) < 0.0f) ? -1.0f :
                1.0f) * ScalarRsqrt(ub_tmp.GetValue(r)); Muls(vec_r, vec_r, sign_scale, n);
            }
        }
    }
    __aicore__ inline void ComputeOutputNormsAndStripMax(u32 aligned_cnt, u8 rep_stride, bool fast_strip_max,
    u32 strip_step, LTf strip_max_base, u32 &max_strip_cnt) {
        max_strip_cnt = 0u;
        if constexpr (SOLVE_MODE != 2u) {
            u32 strip_idx = 0u;
            for (u32 c = 0u; c < n; c += 64u, ++strip_idx) {
                u32 width = (n - c < 64u) ? (n - c) : 64u; LTf dst = (strip_idx == 0u) ? ub_tmp :
                ub_gather[(strip_idx - 1u) * aligned_cnt]; WholeReduceSum(dst, ub_mat[c], width, row_count, 1, 1,
                rep_stride);
            }
            if (fast_strip_max) {
                for (u32 c = 0u; c < n; c += strip_step, ++max_strip_cnt) {
                    u32 width = (n - c < strip_step) ? (n - c) : strip_step;
                    WholeReduceMax(strip_max_base[max_strip_cnt * aligned_cnt], ub_mat[c], width, row_count, 1, 1,
                    rep_stride, ReduceOrder::ORDER_ONLY_VALUE);
                }
            }
            PipeV();
            for (u32 s = 1u; s < strip_idx; ++s) {
                Add(ub_tmp, ub_tmp, ub_gather[(s - 1u) * aligned_cnt], row_count);
                if (s + 1u < strip_idx) {
                    PipeV();
                }
            }
        } else if (fast_strip_max) {
            for (u32 c = 0u; c < n; c += strip_step, ++max_strip_cnt) {
                u32 width = (n - c < strip_step) ? (n - c) : strip_step;
                WholeReduceMax(strip_max_base[max_strip_cnt * aligned_cnt], ub_mat[c], width, row_count, 1, 1,
                rep_stride, ReduceOrder::ORDER_ONLY_VALUE);
            }
        }
    }
    __aicore__ inline void WriteOutputs(u64 b) {
        if (row_count == 0u) {
            return;
        }
        u32 aligned_cnt = (row_count + 7u) & ~7u, tile_stride = aligned_cnt; u8 rep_stride = u8(stride / 8u);
        Mul(ub_mat, ub_vecs, ub_vecs, row_count * stride); PipeV(); bool fast_strip_max = (n >= 40u && n <= 256u &&
        aligned_cnt <= 64u); LTf strip_max_base = (SOLVE_MODE == 2u) ? ub_gather : ub_gather[3u * aligned_cnt];
        u32 max_strip_cnt = 0u, strip_step = (SOLVE_MODE == 2u && n <= 64u) ? 8u : ((n <= 64u) ? 16u : 32u);
        ComputeOutputNormsAndStripMax(aligned_cnt, rep_stride, fast_strip_max, strip_step, strip_max_base,
        max_strip_cnt); LTu col_idx_buf = ub_diag.template ReinterpretCast<u32>();
        for (u32 c = 0u; c < tile_stride; ++c) {
            col_idx_buf.SetValue(c, c * stride * 4u);
        }
        for (u32 r = 0u; r < row_count; ++r) {
            ub_row.SetValue(r, ub_eig.GetValue(row_start + r) * matrix_scale);
        }
        PipeAll(); DataCopyPad(gm_eigvals[b * n + row_start], ub_row, DCEP{1, row_count * 4u, 0, 0, 0});
        CanonicalizeAndScaleRowVecs<SOLVE_MODE != 2u>(fast_strip_max, strip_max_base, max_strip_cnt, strip_step,
        aligned_cnt); PipeV();
        for (u32 i = 0u; i < n; ++i) {
            Gather(ub_mat[i * tile_stride], ub_vecs, col_idx_buf, i * 4u, row_count);
        }
        PipeAll();
        if (row_count == n && tile_stride == n) {
            DataCopyPad(gm_eigvecs[b * n * n], ub_mat, DCEP{1, n * n * 4u, 0, 0, 0});
        } else if ((n & 63u) != 0u) {
            u32 rot = (lane_id * 8u) % n, tail_rows = n - rot; DataCopyPad(gm_eigvecs[b * n * n + rot * n +
            row_start], ub_mat[rot * tile_stride], DCEP{u16(tail_rows), row_count * 4u, 0, (n - row_count) * 4u, 0});
            if (rot > 0u) {
                DataCopyPad(gm_eigvecs[b * n * n + row_start], ub_mat, DCEP{u16(rot), row_count * 4u, 0, (n -
                row_count) * 4u, 0});
            }
        } else {
            DataCopyPad(gm_eigvecs[b * n * n + row_start], ub_mat, DCEP{u16(n), row_count * 4u, 0, (n -
            row_count) * 4u, 0});
        }
        PipeAll();
    }
    __aicore__ inline void WriteOutputsLarge(u64 b) {
        if (row_count == 0u) {
            return;
        }
        u32 aligned_cnt = (row_count + 7u) & ~7u, tile_stride = aligned_cnt; u8 rep_stride = u8(stride / 8u);
        Mul(ub_mat, ub_vecs, ub_vecs, row_count * stride); PipeV(); bool fast_strip_max = (n >= 40u && n <= 256u &&
        aligned_cnt <= 64u); u32 max_strip_cnt = 0u, strip_step = 16u;
        if (fast_strip_max) {
            for (u32 c = 0u; c < n; c += strip_step, ++max_strip_cnt) {
                u32 width = (n - c < strip_step) ? (n - c) : strip_step;
                WholeReduceMax(ub_gather[max_strip_cnt * aligned_cnt], ub_mat[c], width, row_count, 1, 1, rep_stride,
                ReduceOrder::ORDER_ONLY_VALUE);
            }
        }
        LTu col_idx_buf = ub_diag.template ReinterpretCast<u32>();
        for (u32 c = 0u; c < tile_stride; ++c) {
            col_idx_buf.SetValue(c, c * stride * 4u);
        }
        for (u32 r = 0u; r < row_count; ++r) {
            ub_row.SetValue(r, ub_eig.GetValue(row_start + r) * matrix_scale);
        }
        PipeAll(); CanonicalizeAndScaleRowVecs<false>(fast_strip_max, ub_gather, max_strip_cnt, strip_step,
        aligned_cnt); PipeV();
        for (u32 i = 0u; i < n; ++i) {
            Gather(ub_mat[i * tile_stride], ub_vecs, col_idx_buf, i * 4u, row_count);
        }
        PipeAll(); DataCopyPad(gm_eigvals[b * n + row_start], ub_row, DCEP{1, row_count * 4u, 0, 0, 0});
        DataCopyPad(gm_eigvecs[b * n * n + row_start], ub_mat, DCEP{u16(n), row_count * 4u, 0, (n - row_count) * 4u,
        0});
    }
    __aicore__ __attribute__((noinline)) void ProcessMediumM2() {
        float work[4096]; u8 swap_mask[512];
        for (u64 b = group_id; b < batch_size; b += num_groups) {
            row_start = 0u; row_count = n; u8 *tau_mask = swap_mask + 256u; LoadAndSymmetrizeMedium(b);
            TridiagonalizeMedium(work, tau_mask); row_start = vec_row_start; row_count = vec_row_count;
            ComputeEigenvalues(work); ComputeInverseVectors(work, swap_mask); BacktransformMedium(tau_mask);
            WriteOutputs(b);
        }
    }
    __aicore__ __attribute__((noinline)) void ProcessLargeM2() {
        float work[4096]; u8 swap_mask[512];
        for (u64 b = group_id; b < batch_size; b += num_groups) {
            row_start = 0u; row_count = n; u8 *tau_mask = swap_mask + 256u;
            if constexpr (K96_PASS) {
                LoadK96Checkpoint(b, work, tau_mask); this->template fence<HardEvent::S_V>();
            } else {
                LoadAndSymmetrizeLarge(b);
            }
            TridiagonalizeLarge(b, work, tau_mask); row_start = vec_row_start; row_count = vec_row_count;
            SolveTridiagSturmLarge(work); ComputeInverseVectorsLarge(work, swap_mask); BacktransformLarge(b,
            tau_mask); WriteOutputsLarge(b);
        }
    }
    __aicore__ inline void Process() {
        if constexpr (SOLVE_MODE == 1u) {
            float work_small[1792]; u8 swap_small[32];
            for (u64 b = group_id; b < batch_size; b += num_groups) {
                SolveSmallBatch(b, work_small, swap_small);
            }
        } else if constexpr (SOLVE_MODE == 2u) {
            ProcessMediumM2();
        } else if constexpr (SOLVE_MODE == 3u) {
            ProcessLargeM2();
        } else {
            float work[10 * 512]; u8 swap_mask[512];
            for (u64 wave = 0u; wave * num_groups < batch_size; ++wave) {
                u64 b = wave * num_groups + group_id; is_active = (b < batch_size); row_start = coop_row_start;
                row_count = coop_row_count; LoadAndSymmetrizeCoop(b); TridiagonalizeCoop();
                if (is_active) {
                    row_start = vec_row_start; row_count = vec_row_count; ComputeEigenvalues(work);
                    ComputeInverseVectors(work, swap_mask); BacktransformCoop(work); WriteOutputs(b);
                }
                if ((wave + 1u) * num_groups < batch_size) {
                    PipeAll(); SyncAll<true>();
                }
            }
        }
    }
};
template <typename DT_A, u32 SOLVE_MODE = 0u, bool FAST200 = false>
using KernelLinalgEighK96 = KernelLinalgEigh<DT_A, SOLVE_MODE, FAST200, true>;
class Stage96 {
public:
    static constexpr u32 STEPS = 96u;
    template <HardEvent E>
    static __aicore__ inline void StageFence() {
        const i32 id = i32(GetTPipePtr()->FetchEventID(E)); SetFlag<E>(id); WaitFlag<E>(id);
    }
    static __aicore__ inline u32 Owner(u32 r) {
        return (r >> 3u) & 7u;
    }
    static __aicore__ inline u32 LocalRow(u32 r) {
        return (r >> 6u) * 8u + (r & 7u);
    }
    static __aicore__ inline float RsqrtParent(float x) {
        if (x <= 0.0f) {
            return 0.0f;
        }
        union {
            float f; u32 u;
        } b;
        b.f = x; b.u = 0x5f375a86u - (b.u >> 1u); float y = b.f, h = 0.5f * x; y = y * (1.5f - h * y * y); y =
        y * (1.5f - h * y * y); y = y * (1.5f - h * y * y); return y;
    }
    static __aicore__ inline void StageScale(float mx, float &sc_out, float &inv_out) {
        union {
            float f; u32 u;
        } b;
        b.f = mx; u32 exp = b.u & 0x7f800000u;
        if (exp == 0u || exp == 0x7f800000u) {
            sc_out = 1.0f; inv_out = 1.0f;
        } else {
            b.u = exp; sc_out = b.f; b.u = 0x7f000000u - exp; inv_out = b.f;
        }
    }
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR w, GM_ADDR q, GM_ADDR ws, const LinalgEighTilingData &t, LTf ub) {
        n = t.n; batch = u32(t.batch_size); uplo = t.uplo; cid = GetBlockIdx(); group = cid / 10u; lane = cid % 10u;
        active = (lane < 8u); in.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a));
        outw.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(w));
        outq.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(q));
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws), 268288u); mat = ub; v = ub[8192u]; qpart =
        ub[8448u]; qfull = ub[8704u]; next = ub[8960u]; sc = ub[9216u]; prod = ub[13312u]; tmp = ub[13568u]; red =
        ub[13824u]; bv = ub[14080u]; bw = ub[14592u]; result = ub[15104u]; mask_ge = ub[15616u]; mask_eq =
        ub[15680u]; LTf idx_tensor = ub[15744u]; indices = idx_tensor.ReinterpretCast<u32>(); vown = ub[15808u];
        wown = ub[15872u]; hand = ub[15936u]; bv_next = ub[19840u]; bw_next = ub[19904u];
        for (u32 rem = 0u; rem < 8u; ++rem) {
            for (u32 i = 0u; i < 8u; ++i) {
                mask_ge.SetValue(rem * 8u + i, (i >= rem) ? 1.0f : 0.0f); mask_eq.SetValue(rem * 8u + i, (i == rem) ?
                1.0f : 0.0f);
            }
        }
        if (active) {
            for (u32 i = 0u; i < 32u; ++i) {
                const u32 gr = (i >> 3u) * 64u + lane * 8u + (i & 7u); indices.SetValue(i, gr * 4u);
            }
            LTu ix = hand.ReinterpretCast<u32>();
            for (u32 i = 0u; i < 256u; ++i) {
                ix.SetValue(i, i * 32u);
            }
        }
        StageFence<HardEvent::S_V>();
    }
    __aicore__ inline void LoadSymmetrizeBlock(u32 base, u32 r0, LTu ix) {
        DataCopy(mat[LocalRow(r0) * 256u], in[base + r0 * 256u], 2048u); const u32 start = (uplo == 0u) ? (r0 + 8u) :
        0u; const u32 cnt = (uplo == 0u) ? (256u - (r0 + 8u)) : r0;
        if (cnt > 0u) {
            DataCopy(sc, in[base + start * 256u + r0], DCP{u16(cnt), 1u, 31u, 0u});
        }
        StageFence<HardEvent::MTE2_S>();
        for (u32 i = 0u; i < 8u; ++i) {
            for (u32 j = i + 1u; j < 8u; ++j) {
                const u32 ri = LocalRow(r0 + i) * 256u + r0 + j; const u32 rj = LocalRow(r0 + j) * 256u + r0 + i;
                mat.SetValue((uplo == 0u) ? ri : rj, mat.GetValue((uplo == 0u) ? rj : ri));
            }
        }
        StageFence<HardEvent::S_V>();
        if (cnt > 0u) {
            for (u32 j = 0u; j < 8u; ++j) {
                Gather(mat[LocalRow(r0 + j) * 256u + start], sc, ix, j * 4u, cnt);
            }
        }
        StageFence<HardEvent::V_MTE2>();
    }
    __aicore__ inline void LoadAndScaleActive(u32 base) {
        Duplicate(result, 0.0f, 512u); LTu ix = hand.ReinterpretCast<u32>();
        for (u32 r0 = lane * 8u; r0 < 256u; r0 += 64u) {
            LoadSymmetrizeBlock(base, r0, ix);
        }
        float local_max = 0.0f;
        for (u32 off = 0u; off < 8192u; off += 3840u) {
            const u32 count = (8192u - off < 3840u) ? (8192u - off) : 3840u; Abs(sc, mat[off], count); PipeV();
            ReduceMax(red, sc, hand, count); StageFence<HardEvent::V_S>(); const float m = red.GetValue(0);
            local_max = (m > local_max) ? (m) : local_max;
        }
        Duplicate(tmp, 0.0f, 8u); StageFence<HardEvent::V_S>(); tmp.SetValue(0, local_max);
        StageFence<HardEvent::S_MTE3>(); StageFence<HardEvent::V_MTE3>(); DataCopy(gm[4096u + group * 512u +
        lane * 8u], tmp, 8u);
        if (lane == 0u) {
            DataCopy(gm[4096u + group * 512u + 256u], mat, 256u);
        }
        StageFence<HardEvent::MTE3_S>();
    }
    __aicore__ inline void FinishScaleActive() {
        DataCopy(tmp, gm[4096u + group * 512u], 64u); DataCopy(next, gm[4096u + group * 512u + 256u], 256u);
        StageFence<HardEvent::MTE2_S>(); float mx = 0.0f;
        for (u32 i = 0u; i < 8u; ++i) {
            float m = tmp.GetValue(i * 8u); mx = (m > mx) ? (m) : mx;
        }
        float inv = 1.0f; StageScale(mx, scale, inv); Muls(mat, mat, inv, 8192u); Muls(next, next, inv, 256u);
        PipeV();
    }
    __aicore__ inline void Step96HouseholderBuild(u32 k, u32 c0, u32 rem, u32 len, bool &la_ready) {
        Mul(v[c0], next[c0], mask_ge[rem * 8u], 8u);
        if (len > 8u) {
            Adds(v[c0 + 8u], next[c0 + 8u], 0.0f, len - 8u);
        }
        PipeV();
        if (!la_ready) {
            Mul(prod[c0], v[c0], v[c0], len); PipeV(); (len <= 64u) ? (void)WholeReduceSum(red, prod[c0], len, 1, 1,
            1, 8) : (void)ReduceSum(red, prod[c0], tmp, len);
        }
        la_ready = false; StageFence<HardEvent::V_S>(); const float sq = red.GetValue(0), norm =
        sq * RsqrtParent(sq), diag = next.GetValue(k); float beta = 0.0f; u32 tau = 0u;
        if (norm != 0.0f) {
            const float x0 = next.GetValue(k + 1u) + 0.0f; beta = (x0 >= 0.0f) ? -norm : norm; const float inv =
            RsqrtParent(sq - beta * x0); Muls(v[c0], v[c0], inv, len); Muls(red[16], mask_eq[rem * 8u], -beta * inv,
            8u); PipeV(); Add(v[c0], v[c0], red[16], 8u); tau = 1u;
        } else {
            Duplicate(v[c0], 0.0f, len);
        }
        result.SetValue(k, diag); result.SetValue(96u + k, beta); result.SetValue(192u + k, u2f(tau)); PipeV();
    }
    __aicore__ inline void Step96Phase1(u32 k, u32 c0, u32 rem, u32 len, bool &la_ready, u32 &blocks,
    u32 &first_local, u32 &rows) {
        Step96HouseholderBuild(k, c0, rem, len, la_ready); blocks = ((n / 8u) + 7u - lane) / 8u; u32 first_block =
        (k + 1u) >> 3u; first_block += (lane + 8u - (first_block & 7u)) & 7u; first_local = (first_block >> 3u) * 8u;
        first_local = (first_local > blocks * 8u) ? (blocks * 8u) : first_local; rows = blocks * 8u - first_local;
        if (first_local > 0u) {
            Duplicate(qpart, 0.0f, first_local);
        }
        if (rows > 0u) {
            const u32 first = (len < 64u) ? len : 64u; LTf m = mat[first_local * n + c0]; BRP rp{1, 1, 1, 8,
            u8(n / 8u), 0}; Mul(sc, m, v[c0], first, u8(rows), rp);
            for (u32 c = c0 + 64u; c < n; c += 64u) {
                PipeV(); const u32 width = (n - c < 64u) ? (n - c) : 64u; MulAddDst(sc, m[c - c0], v[c], width,
                u8(rows), rp);
            }
            PipeV(); WholeReduceSum(qpart[first_local], sc, first, u8(rows), 1, 1, 8);
        }
        const u32 bank = group * 4u * 256u + (k & 1u) * 2u * 256u, v_lane = (Owner(k + 1u) + 1u) & 7u;
        if (lane == v_lane && c0 > 0u) {
            Duplicate(v, 0.0f, c0);
        }
        if (rows > 0u) {
            Gather(vown, v, indices[first_local], 0u, rows);
        }
        Brcb(bv_next, v[c0], 1u, {1, 8}); PipeV(); StageFence<HardEvent::V_MTE3>(); DataCopy(gm[bank + lane * 8u],
        qpart, DCP{u16(blocks), 1, 0, 7});
        if (Owner(k + 1u) == lane) {
            DataCopy(gm[bank + 256u], mat[LocalRow(k + 1u) * n], n);
        }
        if (lane == v_lane) {
            DataCopy(gm[6144u + group * 65536u + 25600u + k * n], v, n);
        }
        Muls(prod[c0], v[c0], -0.5f, len);
        if (rows > 0u) {
            Brcb(bv, vown, u8(rows / 8u), {1, 8});
        }
        PipeV(); StageFence<HardEvent::MTE3_S>();
    }
    __aicore__ inline void Step96Lookahead(u32 k, bool &la_ready) {
        const u32 c0_n = (k + 2u) & ~7u, rem_n = (k + 2u) & 7u, len_n = n - c0_n;
        if (rem_n == 0u) {
            Mul(prod[c0_n], next[c0_n], next[c0_n], len_n);
        } else {
            Mul(prod[c0_n], next[c0_n], mask_ge[rem_n * 8u], 8u);
            if (len_n > 8u) {
                Mul(prod[c0_n + 8u], next[c0_n + 8u], next[c0_n + 8u], len_n - 8u);
            }
        }
        PipeV();
        if (rem_n != 0u) {
            Mul(prod[c0_n], prod[c0_n], prod[c0_n], (len_n < 8u) ? len_n : 8u); PipeV();
        }
        (len_n <= 64u) ? (void)WholeReduceSum(red, prod[c0_n], len_n, 1, 1, 1, 8) : (void)ReduceSum(red, prod[c0_n],
        tmp, len_n); la_ready = true;
    }
    __aicore__ inline void Step96Rank2Update(u32 k, u32 c0, u32 rem, u32 first_local, u32 rows, bool &la_ready) {
        Brcb(bw_next, qfull[c0], 1u, {1, 8});
        if (rows > 0u) {
            Gather(wown, qfull, indices[first_local], 0u, rows);
        }
        PipeV();
        if (rows > 0u) {
            Brcb(bw, wown, u8(rows / 8u), {1, 8});
        }
        BRP rp_n{1, 1, 0, u8(n / 8u), 0, 1};
        for (u32 c = c0; c < n; c += 64u) {
            const u32 width = (n - c < 64u) ? (n - c) : 64u; MulAddDst(next[c], qfull[c], bv_next[rem * 8u], width,
            1u, rp_n);
        }
        PipeV();
        for (u32 c = c0; c < n; c += 64u) {
            const u32 width = (n - c < 64u) ? (n - c) : 64u; MulAddDst(next[c], v[c], bw_next[rem * 8u], width, 1u,
            rp_n);
            if (rows > 0u) {
                MulAddDst(mat[first_local * n + c], qfull[c], bv, width, u8(rows), rp_n);
            }
        }
        PipeV();
        if (rows > 0u) {
            for (u32 c = c0; c < n; c += 64u) {
                const u32 width = (n - c < 64u) ? (n - c) : 64u; MulAddDst(mat[first_local * n + c], v[c], bw, width,
                u8(rows), rp_n);
            }
        }
        if (k + 1u < STEPS) {
            Step96Lookahead(k, la_ready);
        } else if (rows > 0u) {
            PipeV();
        }
    }
    __aicore__ inline void Step96Phase2(u32 k, u32 c0, u32 rem, u32 len, u32 first_local, u32 rows, bool &la_ready) {
        const u32 bank = group * 4u * 256u + (k & 1u) * 2u * 256u; DataCopy(qfull, gm[bank], n); DataCopy(next,
        gm[bank + 256u], n); StageFence<HardEvent::MTE2_S>();
        if (rem != 0u) {
            Mul(qfull[c0], qfull[c0], mask_ge[rem * 8u], 8u); PipeV();
        }
        Mul(prod[c0], prod[c0], qfull[c0], len); PipeV(); (len <= 64u) ? (void)WholeReduceSum(red, prod[c0], len, 1,
        1, 1, 8) : (void)ReduceSum(red, prod[c0], tmp, len); PipeV(); Muls(qfull[c0], qfull[c0], -1.0f, len);
        Brcb(tmp, red, 1u, {1, 8}); PipeV(); const u32 full = len >> 6u, tail = len & 63u;
        if (full > 0u) {
            Mul(prod[c0], v[c0], tmp, 64u, u8(full), BRP{1, 1, 0, 8, 8, 0});
        }
        if (tail > 0u) {
            Mul(prod[c0 + full * 64u], v[c0 + full * 64u], tmp, tail, 1u, BRP{1, 1, 0, 8, 8, 0});
        }
        PipeV(); Sub(qfull[c0], qfull[c0], prod[c0], len); PipeV(); Step96Rank2Update(k, c0, rem, first_local, rows,
        la_ready);
    }
    __aicore__ inline void StoreStage96Checkpoint() {
        StageFence<HardEvent::V_MTE3>(); StageFence<HardEvent::S_MTE3>();
        for (u32 r = lane * 8u; r < 256u; r += 64u) {
            if (r >= 96u) {
                DataCopy(gm[6144u + group * 65536u + (r - 96u) * 160u], mat[LocalRow(r) * 256u + 96u], DCP{8u, 20u,
                12u, 0u});
            }
        }
        if (lane == 0u) {
            DataCopy(gm[6144u + group * 65536u + 50176u], result, 288u); Duplicate(tmp, 0.0f, 8u);
            StageFence<HardEvent::V_S>(); tmp.SetValue(0, scale); StageFence<HardEvent::S_MTE3>();
            DataCopy(gm[6144u + group * 65536u + 50464u], tmp, 8u);
        }
        StageFence<HardEvent::MTE3_S>();
    }
    __aicore__ inline void Run() {
        const u32 base = group * 65536u;
        if (active) {
            LoadAndScaleActive(base);
        }
        SyncAll<true>();
        if (active) {
            FinishScaleActive();
        }
        bool la_ready = false;
        for (u32 k = 0u; k < STEPS; ++k) {
            const u32 c0 = (k + 1u) & ~7u, rem = (k + 1u) & 7u, len = n - c0; u32 blocks = 0u, first_local = 0u,
            rows = 0u;
            if (active) {
                Step96Phase1(k, c0, rem, len, la_ready, blocks, first_local, rows);
            }
            SyncAll<true>();
            if (active) {
                Step96Phase2(k, c0, rem, len, first_local, rows, la_ready);
            }
        }
        if (active) {
            StoreStage96Checkpoint();
        }
        SyncAll<true>();
    }
private:
    GTf in, outw, outq, gm; LTf mat, v, qpart, qfull, next, sc, prod, tmp, red, bv, bw, result, mask_ge, mask_eq,
    vown, wown, hand, bv_next, bw_next; LTu indices; u32 n, batch, uplo, cid, group, lane; bool active; float scale;
};
} // namespace AscendC
template <typename DT_A, uint32_t SOLVE_MODE = 0u>
__global__ __aicore__ void linalg_eigh(GM_ADDR A, GM_ADDR eigenvalues, GM_ADDR eigenvectors, GM_ADDR workspace,
GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LinalgEighTilingData); GET_TILING_DATA_WITH_STRUCT(LinalgEighTilingData, tiling_data,
    tiling); TPipe pipe;
    if constexpr (SOLVE_MODE == 5u) {
        TBuf<TPosition::VECCALC> shared; pipe.InitBuffer(shared, 49088u * sizeof(float)); LocalTensor<float> ub =
        shared.Get<float>(); __gm__ float *ws = reinterpret_cast<__gm__ float *>(GetUserWorkspace(workspace)) +
        tiling_data.wsOffsetFloats; AscendC::Stage96 stage; stage.Init(A, eigenvalues, eigenvectors,
        reinterpret_cast<GM_ADDR>(ws), tiling_data, ub); stage.Run(); LinalgEighTilingData parent = tiling_data;
        parent.solve_mode = 3u; AscendC::KernelLinalgEighK96<DT_A, 3u> op; op.Init(A, eigenvalues, eigenvectors,
        reinterpret_cast<GM_ADDR>(ws + 6144u), parent, &pipe, ub); op.Process();
    } else {
        AscendC::KernelLinalgEigh<DT_A, (SOLVE_MODE == 4u ? 3u : SOLVE_MODE), (SOLVE_MODE == 4u)> op;
        if constexpr (SOLVE_MODE == 0u) {
            op.Init(A, eigenvalues, eigenvectors, GetUserWorkspace(workspace), tiling_data, &pipe);
        } else {
            op.Init(A, eigenvalues, eigenvectors, workspace, tiling_data, &pipe);
        }
        op.Process();
    }
}
