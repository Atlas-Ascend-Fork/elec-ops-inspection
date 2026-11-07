/*
 * Copyright (c) 2026 lasth0pe.
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

  template <typename TVal>
  __aicore__ __attribute__((always_inline)) inline void RotateTwoScalars(
    TVal& a, TVal& b, float c_g, float s_g
  ) {
    float x1 = a, x2 = b;
    a = c_g * x1 - s_g * x2;
    b = s_g * x1 + c_g * x2;
  }

  __aicore__ __attribute__((always_inline)) inline void RotateTwoVecRowsAxpy(
    const AscendC::LocalTensor<float>& lt_rcp,
    const AscendC::LocalTensor<float>& lt_rk,
    const AscendC::LocalTensor<float>& lt_tmp,
    float c_g,
    float s_g,
    uint64_t msk_p0,
    const AscendC::UnaryRepeatParams& u_def64
  ) {
    AscendC::Muls<float, true>(lt_tmp[0],   lt_rcp, c_g, msk_p0, 1, u_def64);
    AscendC::Muls<float, false>(lt_tmp[64], lt_rk,  c_g, msk_p0, 1, u_def64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Axpy<float, float, false>(lt_tmp[0],  lt_rk,  -s_g, msk_p0, 1, u_def64);
    AscendC::Axpy<float, float, false>(lt_tmp[64], lt_rcp,  s_g, msk_p0, 1, u_def64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<float, false>(lt_rcp, lt_tmp[0],  0.0f, msk_p0, 1, u_def64);
    AscendC::Adds<float, false>(lt_rk,  lt_tmp[64], 0.0f, msk_p0, 1, u_def64);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline void RotateTwoVecRowsMuls(
    const AscendC::LocalTensor<float>& r_cp,
    const AscendC::LocalTensor<float>& r_k,
    const AscendC::LocalTensor<float>& lt_tmp,
    float c_g,
    float s_g,
    uint64_t rot_m,
    const AscendC::UnaryRepeatParams& u_rot,
    const AscendC::BinaryRepeatParams& b_rot
  ) {
    AscendC::Muls<float, true>(lt_tmp[0],    r_cp, c_g, rot_m, 1, u_rot);
    AscendC::Muls<float, false>(lt_tmp[64],  r_k,  s_g, rot_m, 1, u_rot);
    AscendC::Muls<float, false>(lt_tmp[128], r_cp, s_g, rot_m, 1, u_rot);
    AscendC::Muls<float, false>(lt_tmp[192], r_k,  c_g, rot_m, 1, u_rot);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float, false>(r_cp, lt_tmp[0],   lt_tmp[64],  rot_m, 1, b_rot);
    AscendC::Add<float, false>(r_k,  lt_tmp[128], lt_tmp[192], rot_m, 1, b_rot);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();
  }

  template <typename OrdPtr, typename DPtr, typename TBPtr, typename THPtr>
  __aicore__ __attribute__((always_inline)) inline float BuildMergePermAndZ(
    int32_t s,
    int32_t m1,
    int32_t L,
    bool neg,
    float ar,
    bool zero_zk,
    bool set_hs,
    OrdPtr orig_ord,
    DPtr d,
    TBPtr top,
    TBPtr bot,
    THPtr ths,
    THPtr bhs,
    float* dl_l,
    float* zk_l
  ) {
    float scale = ar;
    for (int32_t k = 0; k < L; ++k) {
      int32_t pk = orig_ord[s + k];
      float dk_v  = d[s + pk];
      float top_v = top[s + pk];
      float bot_v = bot[s + pk];
      dl_l[k] = dk_v;
      zk_l[k] = zero_zk ? 0.0f : ((pk < m1) ? (neg ? -bot_v : bot_v) : top_v);
      if (set_hs) {
        ths[s + pk] = (pk < m1) ? top_v : 0.0f;
        bhs[s + pk] = (pk < m1) ? 0.0f  : bot_v;
      }
      float ad_v = MyAbs(dk_v);
      if (ad_v > scale) {
        scale = ad_v;
      }
    }
    return (scale > 1e-30f) ? scale : 1e-30f;
  }

  template <typename FPtr1, typename FPtr2>
  __aicore__ __attribute__((always_inline)) inline float LoadSecularVectorAndNear(
    int32_t s_my,
    int32_t K_my,
    int32_t K_pad,
    int32_t j_base,
    int32_t j_end,
    FPtr1 ad,
    FPtr1 az,
    FPtr2 d_b_ub,
    FPtr2 z2_b_ub,
    float* near_d,
    float* near_z2
  ) {
    float sum_z2 = 0.0f;
    for (int32_t j = 0; j < K_my; ++j) {
      float zv  = az[s_my + j];
      float z2v = zv * zv;
      d_b_ub[j]  = ad[s_my + j];
      z2_b_ub[j] = z2v;
      sum_z2 += z2v;
    }
    for (int32_t j = K_my; j < K_pad; ++j) {
      d_b_ub[j]  = 1e10f;
      z2_b_ub[j] = 0.0f;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (int32_t jj = 0; jj < j_end - j_base; ++jj) {
      int32_t g_idx = s_my + j_base + jj;
      float zv = az[g_idx];
      near_d[jj]  = ad[g_idx];
      near_z2[jj] = zv * zv;
    }
    return sum_z2;
  }

  __aicore__ __attribute__((always_inline)) inline void PrepareSecularMidpointProbe(
    uint32_t b0,
    uint32_t r_cnt,
    int32_t K_my,
    int32_t K_pad,
    int32_t j_base,
    float dup_delta_val,
    const float* near_d,
    int32_t* done_l,
    float* gap_l,
    const AscendC::LocalTensor<float>& lt_d_b,
    const AscendC::LocalTensor<float>& lt_z2_b,
    const AscendC::LocalTensor<float>& lt_delta,
    const AscendC::LocalTensor<float>& lt_z2m,
    const AscendC::LocalTensor<float>& lt_den
  ) {
    for (uint32_t lane = 0; lane < r_cnt; ++lane) {
      done_l[lane] = 0;
      int32_t ri = static_cast<int32_t>(b0 + lane);
      uint32_t l_off = lane * static_cast<uint32_t>(K_pad);
      if (ri < K_my - 1) {
        int32_t rloc = ri - j_base;
        float gap = near_d[rloc + 1] - near_d[rloc];
        gap_l[lane] = gap;
        AscendC::Adds(lt_delta[l_off], lt_d_b, -near_d[rloc], K_pad);
        AscendC::Adds(lt_z2m[l_off],   lt_z2_b, 0.0f, K_pad);
      } else {
        gap_l[lane] = 0.0f;
        AscendC::Duplicate(lt_delta[l_off], dup_delta_val, K_pad);
        AscendC::Duplicate(lt_z2m[l_off],   0.0f, K_pad);
      }
    }
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t lane = 0; lane < r_cnt; ++lane) {
      int32_t ri = static_cast<int32_t>(b0 + lane);
      uint32_t l_off = lane * static_cast<uint32_t>(K_pad);
      if (ri < K_my - 1) {
        AscendC::Adds(lt_den[l_off], lt_delta[l_off], -0.5f * gap_l[lane], K_pad);
      } else {
        AscendC::Duplicate(lt_den[l_off], 1.0f, K_pad);
      }
    }
    AscendC::PipeBarrier<PIPE_V>();
  }

  template <typename StPtr>
  __aicore__ __attribute__((always_inline)) inline void FinalizeSecularInitBrackets(
    uint32_t b0,
    uint32_t r_cnt,
    int32_t K_my,
    int32_t K_pad,
    int32_t j_base,
    float rho_my,
    float sum_z2,
    float dup_done_delta,
    StPtr st_ub,
    const float* near_d,
    int32_t* orig_l,
    int32_t* done_l,
    float* gap_l,
    float* d_orig_l,
    float* low_l,
    float* high_l,
    float* tau_l,
    const AscendC::LocalTensor<float>& lt_d_b,
    const AscendC::LocalTensor<float>& lt_z2_b,
    const AscendC::LocalTensor<float>& lt_delta,
    const AscendC::LocalTensor<float>& lt_z2m
  ) {
    for (uint32_t lane = 0; lane < r_cnt; ++lane) {
      int32_t ri = static_cast<int32_t>(b0 + lane);
      uint32_t l_off = lane * static_cast<uint32_t>(K_pad);
      if (ri < K_my - 1) {
        float fv = 1.0f + rho_my * st_ub[lane];
        float gap = gap_l[lane];
        float mid_g = 0.5f * gap;
        if (fv >= 0.0f) {
          orig_l[lane] = ri;
          low_l[lane]  = 0.0f;
          high_l[lane] = mid_g;
          tau_l[lane]  = 0.25f * gap;
        } else {
          orig_l[lane] = ri + 1;
          low_l[lane]  = -mid_g;
          high_l[lane] = 0.0f;
          tau_l[lane]  = -0.25f * gap;
        }
      } else {
        orig_l[lane] = K_my - 1;
        float bnd = rho_my * sum_z2;
        gap_l[lane]  = bnd;
        low_l[lane]  = 0.0f;
        high_l[lane] = bnd * 1.001f + 1e-30f;
        tau_l[lane]  = 0.5f * bnd;
      }
      int32_t oloc = orig_l[lane] - j_base;
      d_orig_l[lane] = near_d[oloc];
      AscendC::Adds(lt_delta[l_off], lt_d_b, -d_orig_l[lane], K_pad);
      AscendC::Adds(lt_z2m[l_off],   lt_z2_b, 0.0f, K_pad);
      if (tau_l[lane] <= low_l[lane] || tau_l[lane] >= high_l[lane]) {
        done_l[lane] = 1;
        AscendC::Duplicate(lt_delta[l_off], dup_done_delta, K_pad);
        AscendC::Duplicate(lt_z2m[l_off],   0.0f, K_pad);
      }
    }
  }

  template <typename Z2mPtr>
  __aicore__ __attribute__((always_inline)) inline void MaskSecularNearPoles(
    uint32_t b0,
    uint32_t r_cnt,
    int32_t K_my,
    int32_t K_pad,
    const int32_t* done_l,
    Z2mPtr z2m_p
  ) {
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    for (uint32_t lane = 0; lane < r_cnt; ++lane) {
      if (done_l[lane]) continue;
      int32_t ri = static_cast<int32_t>(b0 + lane);
      uint32_t l_off = lane * static_cast<uint32_t>(K_pad);
      uint32_t idx0 = static_cast<uint32_t>((ri < K_my - 1) ? ri : (K_my - 2));
      z2m_p[l_off + idx0]      = 0.0f;
      z2m_p[l_off + idx0 + 1u] = 0.0f;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void EvalSecularSums8(
    uint32_t r_cnt,
    int32_t K_pad,
    int32_t kpad_blk_i,
    const int32_t* done_l,
    const float* tau_l,
    const AscendC::LocalTensor<float>& lt_delta,
    const AscendC::LocalTensor<float>& lt_z2m,
    const AscendC::LocalTensor<float>& lt_den,
    const AscendC::LocalTensor<float>& lt_w_v,
    const AscendC::LocalTensor<float>& lt_absw,
    const AscendC::LocalTensor<float>& lt_dc_st,
    const AscendC::UnaryRepeatParams& u_kp,
    const AscendC::BinaryRepeatParams& b_kp
  ) {
    AscendC::Adds<float, true>(lt_den[0], lt_delta[0], done_l[0] ? -1.0f : -tau_l[0], static_cast<uint64_t>(K_pad), 1, u_kp);
    for (uint32_t lane = 1u; lane < r_cnt; ++lane) {
      uint32_t l_off = lane * static_cast<uint32_t>(K_pad);
      AscendC::Adds<float, false>(lt_den[l_off], lt_delta[l_off], done_l[lane] ? -1.0f : -tau_l[lane], static_cast<uint64_t>(K_pad), 1, u_kp);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div<float, false>(lt_w_v, lt_z2m, lt_den, static_cast<uint64_t>(K_pad), static_cast<uint8_t>(r_cnt), b_kp);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div<float, false>(lt_den,  lt_w_v, lt_den, static_cast<uint64_t>(K_pad), static_cast<uint8_t>(r_cnt), b_kp);
    AscendC::Abs<float, false>(lt_absw, lt_w_v, static_cast<uint64_t>(K_pad), static_cast<uint8_t>(r_cnt), u_kp);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, false>(lt_dc_st[0],  lt_w_v,  K_pad, static_cast<int32_t>(r_cnt), 1, 1, kpad_blk_i);
    AscendC::WholeReduceSum<float, false>(lt_dc_st[8],  lt_den,  K_pad, static_cast<int32_t>(r_cnt), 1, 1, kpad_blk_i);
    AscendC::WholeReduceSum<float, false>(lt_dc_st[16], lt_absw, K_pad, static_cast<int32_t>(r_cnt), 1, 1, kpad_blk_i);
    AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline float SolveInteriorQuadEta(
    float di_f, float dip1_f, float fv, float ac, float bc, float cc, float eta_default
  ) {
    float C0 = (di_f * dip1_f) * fv;
    float B0 = ac * (di_f + dip1_f) + (bc + cc);
    float disc = B0 * B0 - 4.0f * ac * C0;
    if (disc >= 0.0f && IsFiniteF32(disc)) {
      float sq = MySqrt(disc);
      float denom_q = (B0 >= 0.0f) ? (B0 + sq) : (B0 - sq);
      if (denom_q != 0.0f) {
        float cand_eta = (2.0f * C0) / denom_q;
        if (IsFiniteF32(cand_eta) && cand_eta > di_f && cand_eta < dip1_f) {
          return cand_eta;
        }
      }
    }
    return eta_default;
  }

  __aicore__ __attribute__((always_inline)) inline float SolveExteriorQuadEta(
    float d1_e, float d2_e, float t_cur, float fv, float dfv, float ac, float bc, float cc, float eta_default
  ) {
    if (d1_e < d2_e && d2_e < 0.0f && ac > 0.0f) {
      float C0 = (d1_e * d2_e) * fv;
      float B0 = ac * (d1_e + d2_e) + (bc + cc);
      float disc = B0 * B0 - 4.0f * ac * C0;
      if (disc >= 0.0f && IsFiniteF32(disc)) {
        float sq = MySqrt(disc);
        float cand_eta = (B0 >= 0.0f) ? ((B0 + sq) / (2.0f * ac)) : ((2.0f * C0) / (B0 - sq));
        if (IsFiniteF32(cand_eta) && cand_eta > d2_e) {
          return cand_eta;
        }
      }
    }
    float denom_1p = fv + t_cur * dfv;
    if (denom_1p > 0.0f) {
      float cand_1p = (-t_cur * fv) / denom_1p;
      if (IsFiniteF32(cand_1p) && cand_1p > -t_cur) {
        return cand_1p;
      }
    }
    return eta_default;
  }

  __aicore__ __attribute__((always_inline)) inline void ApplySecularTauStep(
    float t_cur, float eta, float fv, float dfv, float tol_step,
    float low_v, float high_v, float& tau_out, int32_t& done_out
  ) {
    float nt = t_cur + eta;
    if (!(IsFiniteF32(nt) && nt > low_v && nt < high_v)) {
      nt = (dfv != 0.0f) ? (t_cur - fv / dfv) : low_v;
      if (!(IsFiniteF32(nt) && nt > low_v && nt < high_v)) {
        nt = 0.5f * (low_v + high_v);
      }
    }
    float brk_sum = MyAbs(low_v) + MyAbs(high_v);
    if (MyAbs(nt - t_cur) <= tol_step * MyAbs(t_cur) || MyAbs(high_v - low_v) <= tol_step * brk_sum || nt <= low_v || nt >= high_v) {
      if (nt > low_v && nt < high_v) {
        tau_out = nt;
      }
      done_out = 1;
    } else {
      tau_out = nt;
    }
  }

  template <typename StPtr>
  __aicore__ __attribute__((always_inline)) inline void UpdateSecularLanesSplit(
    uint32_t b0,
    uint32_t r_cnt,
    int32_t K_my,
    int32_t j_base,
    float rho_my,
    uint32_t st_stride,
    StPtr st_ub,
    const float* near_d,
    const float* near_z2,
    const int32_t* orig_l,
    const float* gap_l,
    int32_t* done_l,
    float* low_l,
    float* high_l,
    float* tau_l
  ) {
    for (uint32_t lane = 0; lane < r_cnt; ++lane) {
      if (done_l[lane]) continue;
      int32_t ri = static_cast<int32_t>(b0 + lane);
      float t_cur     = tau_l[lane];
      float w_far     = st_ub[lane];
      float dp_far    = (st_ub[st_stride + lane] < 0.0f) ? 0.0f : st_ub[st_stride + lane];
      float w_abs_far = st_ub[2u * st_stride + lane];
      float f_far     = 1.0f + rho_my * w_far;
      bool is_int     = (ri < K_my - 1);
      int32_t r_idx   = is_int ? (ri - j_base) : (K_my - 2 - j_base);
      float gap       = is_int ? gap_l[lane] : (near_d[r_idx + 1] - near_d[r_idx]);
      float d0_f      = -gap - t_cur, d1_f = -t_cur;
      if (is_int && orig_l[lane] == ri) { d0_f = -t_cur; d1_f = gap - t_cur; }
      float z2_0      = near_z2[r_idx], z2_1 = near_z2[r_idx + 1];
      float w0_n      = z2_0 / d0_f,    w1_n = z2_1 / d1_f;
      float fv        = f_far + (rho_my * w0_n + rho_my * w1_n);
      float dfv       = rho_my * (dp_far + (w0_n / d0_f + w1_n / d1_f));
      float w_abs     = w_abs_far + MyAbs(w0_n) + MyAbs(w1_n);
      if (MyAbs(fv) <= 9.0e-7f * (1.0f + rho_my * w_abs)) {
        done_l[lane] = 1;
        continue;
      }
      if (fv < 0.0f && t_cur > low_l[lane])  low_l[lane]  = t_cur;
      if (fv >= 0.0f && t_cur < high_l[lane]) high_l[lane] = t_cur;
      float eta = -fv / dfv;
      if (is_int) {
        if (d0_f != 0.0f && d1_f != 0.0f) {
          float dp_h = 0.5f * dp_far;
          float bc = rho_my * z2_0 + (rho_my * dp_h * d0_f) * d0_f;
          float cc = rho_my * z2_1 + (rho_my * dp_h * d1_f) * d1_f;
          float ac = f_far - rho_my * dp_h * (d0_f + d1_f);
          eta = SolveInteriorQuadEta(d0_f, d1_f, fv, ac, bc, cc, eta);
        }
      } else {
        float bc = rho_my * z2_0 + (rho_my * dp_far * d0_f) * d0_f;
        float cc = rho_my * z2_1;
        float ac = f_far - rho_my * dp_far * d0_f;
        eta = SolveExteriorQuadEta(d0_f, d1_f, t_cur, fv, dfv, ac, bc, cc, eta);
      }
      ApplySecularTauStep(t_cur, eta, fv, dfv, 1.2e-7f, low_l[lane], high_l[lane], tau_l[lane], done_l[lane]);
    }
  }
