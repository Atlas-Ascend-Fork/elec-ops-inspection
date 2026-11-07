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

  __aicore__ __attribute__((always_inline)) inline void EvalGuEisenstatProd64(
    int32_t Km,
    int32_t K_pad,
    uint32_t tot_64,
    uint8_t kp_rep64,
    const AscendC::LocalTensor<float>& lt_ad,
    const AscendC::LocalTensor<float>& lt_oad_b,
    const AscendC::LocalTensor<float>& lt_tau_b,
    const AscendC::LocalTensor<float>& lt_ad_b,
    const AscendC::LocalTensor<float>& lt_num_tmp,
    const AscendC::LocalTensor<float>& lt_num_out,
    const AscendC::LocalTensor<float>& lt_den_out,
    const AscendC::LocalTensor<uint16_t>& lt_num_u16,
    const AscendC::LocalTensor<uint16_t>& lt_den_u16,
    const AscendC::LocalTensor<uint16_t>& lt_dmsk_u16,
    const AscendC::LocalTensor<uint16_t>& lt_eye64_u16,
    const AscendC::BinaryRepeatParams& b_bc64_row0,
    const AscendC::BinaryRepeatParams& b_def64
  ) {
    AscendC::Sub<float, true>(lt_num_tmp, lt_ad, lt_oad_b, 64ULL, kp_rep64, b_bc64_row0);
    AscendC::Sub<float, false>(lt_den_out, lt_ad, lt_ad_b,  64ULL, kp_rep64, b_bc64_row0);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float, false>(lt_num_out, lt_num_tmp, lt_tau_b, 64ULL, kp_rep64, b_def64);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();

    int32_t tot_u16 = static_cast<int32_t>(tot_64 * 2u);
    AscendC::And(lt_num_u16, lt_num_u16, lt_dmsk_u16, tot_u16);
    AscendC::And(lt_den_u16, lt_den_u16, lt_dmsk_u16, tot_u16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Or(lt_num_u16, lt_num_u16, lt_eye64_u16, tot_u16);
    AscendC::Or(lt_den_u16, lt_den_u16, lt_eye64_u16, tot_u16);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Div(lt_den_out, lt_num_out, lt_den_out, static_cast<int32_t>(tot_64));
    AscendC::PipeBarrier<PIPE_V>();
    if (K_pad > Km) {
      AscendC::Duplicate(lt_den_out[static_cast<uint32_t>(Km) * 64u], 1.0f, (K_pad - Km) * 64);
      AscendC::PipeBarrier<PIPE_V>();
    }
    for (int32_t h = K_pad >> 1; h >= 1; h >>= 1) {
      AscendC::Mul(lt_den_out, lt_den_out, lt_den_out[static_cast<uint32_t>(h) * 64u], h * 64);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline float FinalizeGuEisenstatElem(
    float num_v, float rho_m, float ak_v, float old_az
  ) {
    const float SCALE_UP = 18446744073709551616.0f;
    const float SCALE_DN = 5.42101086242752217e-20f;
    float ph = (rho_m != 0.0f) ? ((-num_v / rho_m) * ak_v) : 0.0f;
    int32_t exp_acc = 0;
    if (ph > 0.0f && ph < SCALE_DN) {
      ph *= SCALE_UP;
      exp_acc -= 64;
    } else if (ph > SCALE_UP) {
      ph *= SCALE_DN;
      exp_acc += 64;
    }
    float sgn_zk = (old_az < 0.0f) ? -1.0f : 1.0f;
    return (ph > 0.0f) ? (sgn_zk * LdexpF32(MySqrt(ph), exp_acc >> 1)) : 0.0f;
  }

  template <typename WsPtr, typename ZPtr>
  __aicore__ __attribute__((always_inline)) inline void EvalSecularColsAndBoundaries64(
    int32_t Km,
    int32_t K_pad,
    uint32_t tot_64,
    uint8_t kp_rep64,
    bool pad_inv_nrm,
    WsPtr dc_ws_ub,
    ZPtr nrm_ub,
    const AscendC::LocalTensor<float>& lt_dc_ws,
    const AscendC::LocalTensor<float>& lt_b0,
    const AscendC::LocalTensor<float>& lt_b1,
    const AscendC::LocalTensor<float>& lt_col,
    const AscendC::LocalTensor<float>& lt_nrm,
    const AscendC::LocalTensor<uint32_t>& lt_off_bc64,
    const AscendC::BinaryRepeatParams& b_bc64_s1,
    const AscendC::BinaryRepeatParams& b_def64
  ) {
    AscendC::Gather(lt_b0, lt_dc_ws[0],  lt_off_bc64, 0u, tot_64);
    AscendC::Gather(lt_b1, lt_dc_ws[64], lt_off_bc64, 0u, tot_64);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Sub<float, true>(lt_col, lt_b0, lt_dc_ws[256], 64ULL, kp_rep64, b_bc64_s1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float, false>(lt_col, lt_col, lt_dc_ws[320], 64ULL, kp_rep64, b_bc64_s1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div<float, false>(lt_col, lt_b1, lt_col, 64ULL, kp_rep64, b_def64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(lt_nrm, lt_col, lt_col, 64ULL, kp_rep64, b_def64);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();

    for (int32_t h = K_pad >> 1; h >= 1; h >>= 1) {
      AscendC::Add(lt_nrm, lt_nrm, lt_nrm[static_cast<uint32_t>(h) * 64u], h * 64);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

    for (int32_t col = 0; col < Km; ++col) {
      dc_ws_ub[384 + col] = MyRsqrt(nrm_ub[col]);
    }
    if (pad_inv_nrm) {
      for (int32_t col = Km; col < 64; ++col) {
        dc_ws_ub[384 + col] = 0.0f;
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

    AscendC::Mul<float, true>(lt_col, lt_col, lt_dc_ws[384], 64ULL, kp_rep64, b_bc64_s1);
    AscendC::Gather(lt_b0, lt_dc_ws[128], lt_off_bc64, 0u, tot_64);
    AscendC::Gather(lt_b1, lt_dc_ws[192], lt_off_bc64, 0u, tot_64);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Mul<float, false>(lt_b0, lt_b0, lt_col, 64ULL, kp_rep64, b_def64);
    AscendC::Mul<float, false>(lt_b1, lt_b1, lt_col, 64ULL, kp_rep64, b_def64);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();

    for (int32_t h = K_pad >> 1; h >= 1; h >>= 1) {
      AscendC::Add(lt_b0, lt_b0, lt_b0[static_cast<uint32_t>(h) * 64u], h * 64);
      AscendC::Add(lt_b1, lt_b1, lt_b1[static_cast<uint32_t>(h) * 64u], h * 64);
      AscendC::PipeBarrier<PIPE_V>();
    }
  }

  template <typename WsPtr, typename ZPtr>
  __aicore__ __attribute__((always_inline)) inline void EvalSecularBlock8Cols(
    int32_t Km,
    int32_t K_pad,
    uint8_t k_rep,
    uint32_t b_cnt,
    bool copy_src_cols,
    const int32_t* acol_src,
    WsPtr dc_ws_ub,
    ZPtr v2_vp,
    const AscendC::LocalTensor<float>& lt_dc_ws,
    const AscendC::LocalTensor<float>& lt_ad_v,
    const AscendC::LocalTensor<float>& lt_az_v,
    const AscendC::LocalTensor<float>& lt_col,
    const AscendC::LocalTensor<float>& lt_v2,
    const AscendC::LocalTensor<float>& lt_s,
    const AscendC::LocalTensor<float>& lt_t64,
    const AscendC::LocalTensor<float>& lt_ztmp2,
    const AscendC::LocalTensor<float>& lt_z_src,
    const AscendC::LocalTensor<float>& lt_w_out,
    const AscendC::LocalTensor<uint32_t>& lt_off_rep8_4k,
    const AscendC::UnaryRepeatParams& u_def64,
    const AscendC::BinaryRepeatParams& b_bc8_s1,
    const AscendC::BinaryRepeatParams& b_def64
  ) {
    AscendC::Sub<float, true>(lt_s, lt_ad_v, lt_dc_ws[256], 64ULL, k_rep, b_bc8_s1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub<float, false>(lt_s, lt_s, lt_dc_ws[264], 64ULL, k_rep, b_bc8_s1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div<float, false>(lt_col, lt_az_v, lt_s, 64ULL, k_rep, b_def64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(lt_v2, lt_col, lt_col, 64ULL, k_rep, b_def64);
    AscendC::PipeBarrier<PIPE_V>();

    for (int32_t h = K_pad >> 1; h >= 8; h >>= 1) {
      AscendC::Add<float, false>(lt_v2, lt_v2, lt_v2[static_cast<uint32_t>(h * 8)], 64ULL, static_cast<uint8_t>(h >> 3), b_def64);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Add<float, true>(lt_v2, lt_v2, lt_v2[32], 32ULL, 1, b_def64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float, true>(lt_v2, lt_v2, lt_v2[16], 16ULL, 1, b_def64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float, true>(lt_v2, lt_v2, lt_v2[8],  8ULL,  1, b_def64);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    if (copy_src_cols) {
      AscendC::Adds<float, true>(lt_t64, lt_t64, 0.0f, 64ULL, 1, u_def64);
      AscendC::PipeBarrier<PIPE_V>();
      for (int32_t k = 0; k < Km; ++k) {
        AscendC::Adds<float, false>(lt_ztmp2[static_cast<uint32_t>(k) * 64u], lt_z_src[static_cast<uint32_t>(acol_src[k]) * 64u], 0.0f, 64ULL, 1, u_def64);
      }
    }
    AscendC::ResetMask();
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

    for (uint32_t lane = 0; lane < 8u; ++lane) {
      dc_ws_ub[272 + lane] = (lane < b_cnt) ? MyRsqrt(v2_vp[lane]) : 0.0f;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

    AscendC::Mul<float, true>(lt_s, lt_col, lt_dc_ws[272], 64ULL, k_rep, b_bc8_s1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(lt_w_out, lt_s[0], lt_off_rep8_4k, 0u, static_cast<uint32_t>(Km) * 64u);
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline void AccumRank1Cols4x(
    int32_t Km,
    uint64_t vec_mask,
    uint8_t bc_u8,
    const AscendC::LocalTensor<float>& lt_dst,
    const AscendC::LocalTensor<float>& lt_src,
    const AscendC::LocalTensor<float>& lt_w,
    const AscendC::LocalTensor<float>& lt_acc1,
    const AscendC::LocalTensor<float>& lt_acc2,
    const AscendC::LocalTensor<float>& lt_acc3,
    const AscendC::BinaryRepeatParams& rk_params,
    const AscendC::BinaryRepeatParams& add_params
  ) {
    int32_t k = 0;
    if (Km >= 4) {
      AscendC::Mul<float, true>(lt_dst,   lt_src[0],    lt_w[0],    vec_mask, bc_u8, rk_params);
      AscendC::Mul<float, false>(lt_acc1, lt_src[64u],  lt_w[64u],  vec_mask, bc_u8, rk_params);
      AscendC::Mul<float, false>(lt_acc2, lt_src[128u], lt_w[128u], vec_mask, bc_u8, rk_params);
      AscendC::Mul<float, false>(lt_acc3, lt_src[192u], lt_w[192u], vec_mask, bc_u8, rk_params);
      AscendC::PipeBarrier<PIPE_V>();
      k = 4;
      for (; k + 3 < Km; k += 4) {
        uint32_t ko = static_cast<uint32_t>(k) * 64u;
        AscendC::MulAddDst<float, float, false>(lt_dst,  lt_src[ko],        lt_w[ko],        vec_mask, bc_u8, rk_params);
        AscendC::MulAddDst<float, float, false>(lt_acc1, lt_src[ko + 64u],  lt_w[ko + 64u],  vec_mask, bc_u8, rk_params);
        AscendC::MulAddDst<float, float, false>(lt_acc2, lt_src[ko + 128u], lt_w[ko + 128u], vec_mask, bc_u8, rk_params);
        AscendC::MulAddDst<float, float, false>(lt_acc3, lt_src[ko + 192u], lt_w[ko + 192u], vec_mask, bc_u8, rk_params);
        AscendC::PipeBarrier<PIPE_V>();
      }
    } else {
      AscendC::Mul<float, true>(lt_dst, lt_src[0], lt_w[0], vec_mask, bc_u8, rk_params);
      AscendC::PipeBarrier<PIPE_V>();
      k = 1;
    }
    for (; k < Km; ++k) {
      uint32_t ko = static_cast<uint32_t>(k) * 64u;
      AscendC::MulAddDst<float, float, false>(lt_dst, lt_src[ko], lt_w[ko], vec_mask, bc_u8, rk_params);
      AscendC::PipeBarrier<PIPE_V>();
    }
    if (Km >= 4) {
      AscendC::Add<float, false>(lt_acc1, lt_acc1, lt_acc2, vec_mask, bc_u8, add_params);
      AscendC::Add<float, false>(lt_dst,  lt_dst,  lt_acc3, vec_mask, bc_u8, add_params);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add<float, false>(lt_dst,  lt_dst,  lt_acc1, vec_mask, bc_u8, add_params);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::ResetMask();
  }

  template <typename OutPtr>
  __aicore__ __attribute__((always_inline)) inline void ComputeQuadReflectorDots64(
    uint32_t n_quads,
    const uint32_t* act_uis,
    const AscendC::LocalTensor<float>& lt_u_base,
    const AscendC::LocalTensor<float>& lt_prod,
    const AscendC::LocalTensor<float>& lt_red_out,
    OutPtr red_out_ub,
    float* u_dots
  ) {
    uint32_t n_dp = n_quads * 6u;
    AscendC::BinaryRepeatParams b64_dot(1, 1, 1, 8, 8, 8);
    for (uint32_t qd = 0u; qd < n_quads; ++qd) {
      uint32_t b6  = qd * 6u;
      uint32_t ui0 = act_uis[qd * 4u + 0u];
      uint32_t ui1 = act_uis[qd * 4u + 1u];
      uint32_t ui2 = act_uis[qd * 4u + 2u];
      uint32_t ui3 = act_uis[qd * 4u + 3u];
      AscendC::Mul<float, true>(lt_prod[(b6 + 0u) * 64u],  lt_u_base[ui0 * 64u], lt_u_base[ui1 * 64u], 64ULL, 1, b64_dot);
      AscendC::Mul<float, false>(lt_prod[(b6 + 1u) * 64u], lt_u_base[ui0 * 64u], lt_u_base[ui2 * 64u], 64ULL, 1, b64_dot);
      AscendC::Mul<float, false>(lt_prod[(b6 + 2u) * 64u], lt_u_base[ui0 * 64u], lt_u_base[ui3 * 64u], 64ULL, 1, b64_dot);
      AscendC::Mul<float, false>(lt_prod[(b6 + 3u) * 64u], lt_u_base[ui1 * 64u], lt_u_base[ui2 * 64u], 64ULL, 1, b64_dot);
      AscendC::Mul<float, false>(lt_prod[(b6 + 4u) * 64u], lt_u_base[ui1 * 64u], lt_u_base[ui3 * 64u], 64ULL, 1, b64_dot);
      AscendC::Mul<float, false>(lt_prod[(b6 + 5u) * 64u], lt_u_base[ui2 * 64u], lt_u_base[ui3 * 64u], 64ULL, 1, b64_dot);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, false>(lt_red_out, lt_prod[0], 64, static_cast<int32_t>(n_dp), 1, 1, 8);
    AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    for (uint32_t idx = 0u; idx < n_dp; ++idx) {
      u_dots[idx] = red_out_ub[idx];
    }
  }

  __aicore__ __attribute__((always_inline)) inline void SolveQuadReflectorTriangular(
    uint64_t msk_len,
    uint32_t st_step,
    uint32_t bc_step,
    uint32_t bc_cnt,
    float ih0, float ih1, float ih2, float ih3,
    float u01, float u02, float u03, float u12, float u13, float u23,
    const AscendC::LocalTensor<float>& lt_p,
    const AscendC::LocalTensor<float>& lt_bc,
    const AscendC::LocalTensor<uint32_t>& lt_off_bc,
    const AscendC::UnaryRepeatParams& u_param
  ) {
    AscendC::Muls<float, true>(lt_p[0],            lt_p[0],            -ih0, msk_len, 1, u_param);
    AscendC::Muls<float, false>(lt_p[st_step],     lt_p[st_step],     -ih1, msk_len, 1, u_param);
    AscendC::Muls<float, false>(lt_p[2u * st_step], lt_p[2u * st_step], -ih2, msk_len, 1, u_param);
    AscendC::Muls<float, false>(lt_p[3u * st_step], lt_p[3u * st_step], -ih3, msk_len, 1, u_param);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Axpy<float, float, false>(lt_p[st_step],      lt_p[0], -ih1 * u01, msk_len, 1, u_param);
    AscendC::Axpy<float, float, false>(lt_p[2u * st_step], lt_p[0], -ih2 * u02, msk_len, 1, u_param);
    AscendC::Axpy<float, float, false>(lt_p[3u * st_step], lt_p[0], -ih3 * u03, msk_len, 1, u_param);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Axpy<float, float, false>(lt_p[2u * st_step], lt_p[st_step], -ih2 * u12, msk_len, 1, u_param);
    AscendC::Axpy<float, float, false>(lt_p[3u * st_step], lt_p[st_step], -ih3 * u13, msk_len, 1, u_param);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Axpy<float, float, false>(lt_p[3u * st_step], lt_p[2u * st_step], -ih3 * u23, msk_len, 1, u_param);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Gather(lt_bc[0],            lt_p[0],            lt_off_bc, 0u, bc_cnt);
    AscendC::Gather(lt_bc[bc_step],      lt_p[st_step],      lt_off_bc, 0u, bc_cnt);
    AscendC::Gather(lt_bc[2u * bc_step], lt_p[2u * st_step], lt_off_bc, 0u, bc_cnt);
    AscendC::Gather(lt_bc[3u * bc_step], lt_p[3u * st_step], lt_off_bc, 0u, bc_cnt);
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline bool ScanColSignTieBreak(
    __ubuf__ float* col_ptr
  ) {
    float best_signed = col_ptr[0];
    float best_abs = MyAbs(best_signed);
    for (uint32_t r = 1u; r < n_; ++r) {
      float v = col_ptr[r];
      float av = MyAbs(v);
      if (av > best_abs) {
        best_abs = av;
        best_signed = v;
      }
    }
    return (best_signed < 0.0f);
  }
