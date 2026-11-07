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

  __aicore__ __attribute__((always_inline)) inline void DeflateLargeSingleMerge(
    uint32_t m, int32_t step, bool final_level, uint32_t col_start, uint32_t col_end, uint32_t stride,
    const __ubuf__ int32_t* bounds, __ubuf__ float* d, const __ubuf__ float* e, const __ubuf__ float* top, const __ubuf__ float* bot,
    __ubuf__ float* ths, __ubuf__ float* bhs, __ubuf__ float* ad, __ubuf__ float* az, __ubuf__ float* dd,
    __ubuf__ int32_t* orig_ord, __ubuf__ int32_t* acol_dc, __ubuf__ int32_t* K_arr, __ubuf__ int32_t* D_arr, __ubuf__ int32_t* triv_arr, __ubuf__ float* rho_arr,
    __ubuf__ float* mat_ub
  ) {
    int32_t b_idx = static_cast<int32_t>(m) * 2 * step, s = bounds[b_idx], mid = bounds[b_idx + step], end = bounds[b_idx + 2 * step], L = end - s, m1 = mid - s;
    float r_raw = e[mid - 1];
    bool neg = (r_raw < 0.0f);
    float ar = neg ? -r_raw : r_raw;
    rho_arr[m] = ar;
    if (ar == 0.0f) { triv_arr[m] = 1; K_arr[m] = 0; D_arr[m] = L; return; }
    triv_arr[m] = 0;
    MergeTwoSortedF32(m1, L, &orig_ord[s], &d[s]);

    uint32_t r_act_lo = (static_cast<uint32_t>(s) > col_start) ? static_cast<uint32_t>(s) : col_start;
    uint32_t r_act_hi = (static_cast<uint32_t>(s + L) < col_end) ? static_cast<uint32_t>(s + L) : col_end;
    uint32_t act_c0   = (r_act_lo < r_act_hi) ? (r_act_lo - col_start) : 0u;
    uint32_t act_cols = (r_act_lo < r_act_hi) ? (r_act_hi - r_act_lo) : 0u;

    float dl_p0[256], zk_p0[256];
    float ref_sc = BuildMergePermAndZ(s, m1, L, neg, ar, false, !final_level, orig_ord, d, top, bot, ths, bhs, dl_p0, zk_p0);
    float tol_z = 8.5e-7f * ref_sc, tol_d = 1.8e-6f * ref_sc;
    int32_t K = 0, D = 0, last_cp = 0;
    float last_ad = 0.0f, last_az = 0.0f;

    for (int32_t k = 0; k < L; ++k) {
      int32_t pk = orig_ord[s + k];
      float dk = dl_p0[k], zk = zk_p0[k];
      if (MyAbs(zk) <= tol_z) {
        dd[s + D] = dk; acol_dc[s + L - 1 - D] = pk; D++;
      } else if (K > 0 && MyAbs(dk - last_ad) <= tol_d) {
        float zp = last_az, r_hyp = MySqrt(zp * zp + zk * zk), c_g = zk / r_hyp, s_g = zp / r_hyp;
        int32_t cp = last_cp;
        if (!final_level) {
          RotateTwoScalars(ths[s + cp], ths[s + pk], c_g, s_g);
          RotateTwoScalars(bhs[s + cp], bhs[s + pk], c_g, s_g);
        }
        for (uint32_t c = 0; c < act_cols; ++c) {
          __ubuf__ float* zr = mat_ub + (act_c0 + c) * stride + static_cast<uint32_t>(s);
          RotateTwoScalars(zr[cp], zr[pk], c_g, s_g);
        }
        dd[s + D] = last_ad; acol_dc[s + L - 1 - D] = cp; D++;
        ad[s + K - 1] = last_ad = dk; az[s + K - 1] = last_az = r_hyp; acol_dc[s + K - 1] = last_cp = pk;
      } else { ad[s + K] = last_ad = dk; az[s + K] = last_az = zk; acol_dc[s + K] = last_cp = pk; K++; }
    }
    K_arr[m] = K; D_arr[m] = D;
  }

  __aicore__ __attribute__((always_inline)) inline void EvalLargeSecularSumsFold(
    uint32_t r_cnt, int32_t K_pad, int32_t act_cnt, int32_t kpad_blk_i, const int32_t* done_l, const float* tau_l,
    const AscendC::LocalTensor<float>& lt_delta, const AscendC::LocalTensor<float>& lt_z2m, const AscendC::LocalTensor<float>& lt_den,
    const AscendC::LocalTensor<float>& lt_w_v, const AscendC::LocalTensor<float>& lt_absw, const AscendC::LocalTensor<float>& lt_stage,
    const AscendC::UnaryRepeatParams& u_def64_s3, const AscendC::BinaryRepeatParams& b_def64_s3,
    const AscendC::UnaryRepeatParams& u_def16_s3, const AscendC::BinaryRepeatParams& b_def16_s3, const AscendC::BinaryRepeatParams& fold_b
  ) {
    if (K_pad >= 64) {
      uint8_t rep_kp = static_cast<uint8_t>(K_pad >> 6), rep_ac = static_cast<uint8_t>(act_cnt >> 6), rc_u8 = static_cast<uint8_t>(r_cnt);
      AscendC::Adds<float, true>(lt_den[0], lt_delta[0], done_l[0] ? -1.0f : -tau_l[0], 64ULL, rep_kp, u_def64_s3);
      for (uint32_t lane = 1u; lane < r_cnt; ++lane) {
        AscendC::Adds<float, false>(lt_den[lane * static_cast<uint32_t>(K_pad)], lt_delta[lane * static_cast<uint32_t>(K_pad)], done_l[lane] ? -1.0f : -tau_l[lane], 64ULL, rep_kp, u_def64_s3);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Div<float, false>(lt_w_v, lt_z2m, lt_den, 64ULL, rep_ac, b_def64_s3);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Div<float, false>(lt_den, lt_w_v, lt_den, 64ULL, rep_ac, b_def64_s3);
      AscendC::Abs<float, false>(lt_absw, lt_w_v, 64ULL, rep_ac, u_def64_s3);
      AscendC::PipeBarrier<PIPE_V>();
      for (int32_t h = K_pad >> 1; h >= 64; h >>= 1) {
        uint32_t n_b64 = static_cast<uint32_t>(h >> 6);
        for (uint32_t b = 0; b < n_b64; ++b) {
          uint32_t bo = b * 64u, ho = static_cast<uint32_t>(h) + bo;
          AscendC::Add<float, false>(lt_w_v[bo],  lt_w_v[bo],  lt_w_v[ho],  64ULL, rc_u8, fold_b);
          AscendC::Add<float, false>(lt_den[bo],  lt_den[bo],  lt_den[ho],  64ULL, rc_u8, fold_b);
          AscendC::Add<float, false>(lt_absw[bo], lt_absw[bo], lt_absw[ho], 64ULL, rc_u8, fold_b);
        }
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::WholeReduceSum<float, false>(lt_stage[0],  lt_w_v,  64, static_cast<int32_t>(r_cnt), 1, 1, kpad_blk_i);
      AscendC::WholeReduceSum<float, false>(lt_stage[16], lt_den,  64, static_cast<int32_t>(r_cnt), 1, 1, kpad_blk_i);
      AscendC::WholeReduceSum<float, false>(lt_stage[32], lt_absw, 64, static_cast<int32_t>(r_cnt), 1, 1, kpad_blk_i);
    } else {
      uint8_t rep_kp = static_cast<uint8_t>(K_pad >> 4), rep_ac = static_cast<uint8_t>(act_cnt >> 4);
      AscendC::Adds<float, true>(lt_den[0], lt_delta[0], done_l[0] ? -1.0f : -tau_l[0], 16ULL, rep_kp, u_def16_s3);
      for (uint32_t lane = 1u; lane < r_cnt; ++lane) {
        AscendC::Adds<float, false>(lt_den[lane * static_cast<uint32_t>(K_pad)], lt_delta[lane * static_cast<uint32_t>(K_pad)], done_l[lane] ? -1.0f : -tau_l[lane], 16ULL, rep_kp, u_def16_s3);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Div<float, false>(lt_w_v, lt_z2m, lt_den, 16ULL, rep_ac, b_def16_s3);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Div<float, false>(lt_den, lt_w_v, lt_den, 16ULL, rep_ac, b_def16_s3);
      AscendC::Abs<float, false>(lt_absw, lt_w_v, 16ULL, rep_ac, u_def16_s3);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::WholeReduceSum<float, true>(lt_stage[0],  lt_w_v,  K_pad, static_cast<int32_t>(r_cnt), 1, 1, kpad_blk_i);
      AscendC::WholeReduceSum<float, false>(lt_stage[16], lt_den,  K_pad, static_cast<int32_t>(r_cnt), 1, 1, kpad_blk_i);
      AscendC::WholeReduceSum<float, false>(lt_stage[32], lt_absw, K_pad, static_cast<int32_t>(r_cnt), 1, 1, kpad_blk_i);
    }
    AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void SolveLargeRootsBatch(
    int32_t s_my, int32_t K_my, float rho_my, uint32_t i0, uint32_t i1, const __ubuf__ float* ad, const __ubuf__ float* az,
    __ubuf__ int32_t* orig_ord, __ubuf__ float* tau_a, __ubuf__ float* wide_scr, __ubuf__ float* stage_ub, __ubuf__ float* z_ub,
    const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::UnaryRepeatParams& u_def64_s3, const AscendC::BinaryRepeatParams& b_def64_s3,
    const AscendC::UnaryRepeatParams& u_def16_s3, const AscendC::BinaryRepeatParams& b_def16_s3
  ) {
    if (K_my == 1) { orig_ord[s_my] = 0; tau_a[s_my] = rho_my * (az[s_my] * az[s_my]); return; }
    int32_t K_pad = 16;
    while (K_pad < K_my) K_pad <<= 1;
    uint32_t B_p1 = (K_pad <= 256) ? 16u : 8u;
    uint8_t kpad_blk_u8 = static_cast<uint8_t>(K_pad >> 3);
    int32_t kpad_blk_i  = K_pad >> 3;
    AscendC::BinaryRepeatParams fold_b(1, 1, 1, kpad_blk_u8, kpad_blk_u8, kpad_blk_u8);
    AscendC::LocalTensor<float> lt_d_b = lt_wscr[0], lt_z2_b = lt_wscr[512], lt_absw = lt_wscr[1024];
    AscendC::LocalTensor<float> lt_delta = lt_z[0], lt_z2m = lt_z[4096], lt_den = lt_z[8192], lt_w_v = lt_z[12288];

    for (uint32_t b0 = i0; b0 < i1; b0 += B_p1) {
      uint32_t r_cnt = (b0 + B_p1 <= i1) ? B_p1 : (i1 - b0);
      int32_t act_cnt = static_cast<int32_t>(r_cnt) * K_pad;
      int32_t orig_l[16], done_l[16];
      float gap_l[16], d_orig_l[16], low_l[16], high_l[16], tau_l[16], near_d[24], near_z2[24];
      int32_t j_base = (static_cast<int32_t>(b0) - 4 > 0) ? (static_cast<int32_t>(b0) - 4) : 0;
      int32_t j_end  = (static_cast<int32_t>(b0 + B_p1) + 4 < K_my) ? (static_cast<int32_t>(b0 + B_p1) + 4) : K_my;
      float sum_z2 = LoadSecularVectorAndNear(s_my, K_my, K_pad, j_base, j_end, ad, az, wide_scr, wide_scr + 512, near_d, near_z2);
      PrepareSecularMidpointProbe(b0, r_cnt, K_my, K_pad, j_base, 1.0f, near_d, done_l, gap_l, lt_d_b, lt_z2_b, lt_delta, lt_z2m, lt_den);

      AscendC::Div(lt_w_v, lt_z2m, lt_den, act_cnt);
      AscendC::PipeBarrier<PIPE_V>();
      if (K_pad >= 64) {
        for (int32_t h = K_pad >> 1; h >= 64; h >>= 1) {
          uint32_t n_b64 = static_cast<uint32_t>(h >> 6);
          for (uint32_t b = 0; b < n_b64; ++b) {
            uint32_t bo = b * 64u, ho = static_cast<uint32_t>(h) + bo;
            AscendC::Add<float, true>(lt_w_v[bo], lt_w_v[bo], lt_w_v[ho], 64ULL, static_cast<uint8_t>(r_cnt), fold_b);
          }
          AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::WholeReduceSum<float, false>(lt_stage[0], lt_w_v, 64, static_cast<int32_t>(r_cnt), 1, 1, kpad_blk_i);
      } else {
        AscendC::WholeReduceSum<float, true>(lt_stage[0], lt_w_v, K_pad, static_cast<int32_t>(r_cnt), 1, 1, kpad_blk_i);
      }
      AscendC::ResetMask();
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

      FinalizeSecularInitBrackets(b0, r_cnt, K_my, K_pad, j_base, rho_my, sum_z2, 1.0f, stage_ub, near_d, orig_l, done_l, gap_l, d_orig_l, low_l, high_l, tau_l, lt_d_b, lt_z2_b, lt_delta, lt_z2m);
      MaskSecularNearPoles(b0, r_cnt, K_my, K_pad, done_l, z_ub + 4096);
      for (int32_t it = 0; it < 26; ++it) {
        bool all_done = true;
        for (uint32_t lane = 0; lane < r_cnt; ++lane) if (!done_l[lane]) all_done = false;
        if (all_done) break;
        EvalLargeSecularSumsFold(r_cnt, K_pad, act_cnt, kpad_blk_i, done_l, tau_l, lt_delta, lt_z2m, lt_den, lt_w_v, lt_absw, lt_stage, u_def64_s3, b_def64_s3, u_def16_s3, b_def16_s3, fold_b);
        UpdateSecularLanesSplit(b0, r_cnt, K_my, j_base, rho_my, 16u, stage_ub, near_d, near_z2, orig_l, gap_l, done_l, low_l, high_l, tau_l);
      }
      for (uint32_t lane = 0; lane < r_cnt; ++lane) { orig_ord[s_my + b0 + lane] = orig_l[lane]; tau_a[s_my + b0 + lane] = tau_l[lane]; }
    }
  }

  __aicore__ __attribute__((always_inline)) inline void SolveLargeRootsAndSync(
    uint32_t p, uint32_t P, int32_t step, uint32_t num_merges, int32_t s_my, int32_t K_my, float rho_my, uint32_t i0, uint32_t i1,
    const __ubuf__ int32_t* bounds, const __ubuf__ int32_t* K_arr, const __ubuf__ float* ad, const __ubuf__ float* az,
    __ubuf__ int32_t* orig_ord, __ubuf__ float* tau_a, __ubuf__ float* wide_scr, __ubuf__ float* stage_ub, __ubuf__ float* z_ub,
    const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::GlobalTensor<float>& gt_roots, const AscendC::UnaryRepeatParams& u_def64_s3, const AscendC::BinaryRepeatParams& b_def64_s3,
    const AscendC::UnaryRepeatParams& u_def16_s3, const AscendC::BinaryRepeatParams& b_def16_s3
  ) {
    if (K_my > 0 && i0 < i1) {
      SolveLargeRootsBatch(s_my, K_my, rho_my, i0, i1, ad, az, orig_ord, tau_a, wide_scr, stage_ub, z_ub, lt_wscr, lt_stage, lt_z, u_def64_s3, b_def64_s3, u_def16_s3, b_def16_s3);
      AscendC::Duplicate(lt_stage, 0.0f, 64);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      __ubuf__ int32_t* st_orig = reinterpret_cast<__ubuf__ int32_t*>(stage_ub);
      for (uint32_t ci = 0; ci < i1 - i0; ++ci) { st_orig[ci] = orig_ord[s_my + i0 + ci]; stage_ub[32 + ci] = tau_a[s_my + i0 + ci]; }
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    } else {
      AscendC::Duplicate(lt_stage, 0.0f, 64);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    }
    AscendC::DataCopy(gt_roots[p * 64u], lt_stage, 64);
    SafeSyncAll();

    AscendC::DataCopy(lt_z, gt_roots, static_cast<int32_t>(P * 64u));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    for (uint32_t m = 0; m < num_merges; ++m) {
      int32_t Km = K_arr[m];
      if (Km <= 0) continue;
      int32_t sm = bounds[static_cast<int32_t>(m) * 2 * step];
      uint32_t plo_m = (m * P) / num_merges, phi_m = ((m + 1u) * P) / num_merges, sub_P_m = phi_m - plo_m;
      uint32_t u_cap_m = (static_cast<uint32_t>(Km) + sub_P_m - 1u) / sub_P_m;
      for (uint32_t sp = 0; sp < sub_P_m; ++sp) {
        uint32_t base_off = (plo_m + sp) * 64u;
        __ubuf__ int32_t* kp_orig = reinterpret_cast<__ubuf__ int32_t*>(z_ub + base_off);
        for (uint32_t ci = 0; ci < u_cap_m; ++ci) {
          uint32_t gi = sp * u_cap_m + ci;
          if (gi < static_cast<uint32_t>(Km)) { orig_ord[sm + gi] = kp_orig[ci]; tau_a[sm + gi] = z_ub[base_off + 32u + ci]; }
        }
      }
    }
  }

  __aicore__ __attribute__((always_inline)) inline void InitLargeGuEisenstatTables(
    int32_t s_my, int32_t K_my, int32_t cnt_ge16, int32_t B_ge, int32_t cnt_ge_vec,
    const __ubuf__ int32_t* orig_ord, const __ubuf__ float* ad, const __ubuf__ float* tau_a,
    __ubuf__ float* wide_scr, __ubuf__ float* stage_ub,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_stage,
    const AscendC::LocalTensor<float>& lt_oad, const AscendC::LocalTensor<float>& lt_tau, const AscendC::LocalTensor<float>& lt_dad,
    const AscendC::LocalTensor<uint32_t>& lt_u32
  ) {
    __ubuf__ uint32_t* wscr_u32 = reinterpret_cast<__ubuf__ uint32_t*>(wide_scr);
    AscendC::LocalTensor<uint32_t> lt_off_ge = lt_u32[14 * 512];
    AscendC::LocalTensor<int32_t> lt_off_ge_i = ub_all.ReinterpretCast<int32_t>()[14 * 512];
    for (int32_t j = 0; j < K_my; ++j) {
      int32_t oj = orig_ord[s_my + j];
      stage_ub[j] = ad[s_my + oj]; stage_ub[512 + j] = tau_a[s_my + j]; stage_ub[1024 + j] = ad[s_my + j];
    }
    for (int32_t j = K_my; j < cnt_ge16; ++j) { stage_ub[j] = 0.0f; stage_ub[512 + j] = -1.0f; stage_ub[1024 + j] = 1e10f; }
    for (int32_t idx = 0; idx < 64; ++idx) wscr_u32[idx] = static_cast<uint32_t>((idx / B_ge) * 4);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (int32_t rep = 64; rep < cnt_ge_vec; rep <<= 1) {
      int32_t copy_n = (rep <= cnt_ge_vec - rep) ? rep : (cnt_ge_vec - rep);
      AscendC::Adds(lt_off_ge_i[rep], lt_off_ge_i[0], (rep / B_ge) * 4, copy_n);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Gather(lt_oad, lt_stage[0], lt_off_ge, 0u, static_cast<uint32_t>(cnt_ge_vec));
    AscendC::Gather(lt_tau, lt_stage[512], lt_off_ge, 0u, static_cast<uint32_t>(cnt_ge_vec));
    AscendC::Gather(lt_dad, lt_stage[1024], lt_off_ge, 0u, static_cast<uint32_t>(cnt_ge_vec));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void EvalLargeGuEisenstatBatch(
    int32_t s_my, int32_t K_my, float rho_my, uint32_t ge_i0, uint32_t ge_i1, const __ubuf__ int32_t* orig_ord, const __ubuf__ float* ad, const __ubuf__ float* az, const __ubuf__ float* tau_a,
    float* zhat_out, __ubuf__ float* wide_scr, __ubuf__ float* stage_ub, __ubuf__ float* z_ub,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_u32
  ) {
    int32_t cnt_ge16 = (K_my + 15) & ~15, K_ge_pow2 = 16;
    while (K_ge_pow2 < cnt_ge16) K_ge_pow2 <<= 1;
    int32_t B_ge = (4096 / K_ge_pow2 > 16) ? 16 : (4096 / K_ge_pow2), tot_ge = K_ge_pow2 * B_ge, cnt_ge_vec = cnt_ge16 * B_ge;
    AscendC::LocalTensor<float> lt_oad = lt_z[0], lt_tau = lt_z[4096], lt_dad = lt_z[8192], lt_ak = lt_z[12288], lt_num = lt_wscr[0];
    __ubuf__ float* dad_p = z_ub + 8192, *ak_p = z_ub + 12288, *num_p = wide_scr;
    InitLargeGuEisenstatTables(s_my, K_my, cnt_ge16, B_ge, cnt_ge_vec, orig_ord, ad, tau_a, wide_scr, stage_ub, ub_all, lt_stage, lt_oad, lt_tau, lt_dad, lt_u32);

    uint32_t n_mine = ge_i1 - ge_i0;
    for (uint32_t cb = 0; cb < n_mine; cb += static_cast<uint32_t>(B_ge)) {
      uint32_t b_cnt = (cb + static_cast<uint32_t>(B_ge) <= n_mine) ? static_cast<uint32_t>(B_ge) : (n_mine - cb);
      for (int32_t lane = 0; lane < B_ge; ++lane) ak_p[lane] = ad[s_my + static_cast<int32_t>(ge_i0 + cb) + ((lane < static_cast<int32_t>(b_cnt)) ? lane : 0)];
      for (uint32_t lane = 0; lane < b_cnt; ++lane) {
        int32_t k = static_cast<int32_t>(ge_i0 + cb + lane);
        dad_p[k * B_ge + static_cast<int32_t>(lane)] = ad[s_my + k] - 1.0f;
      }
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      for (int32_t rep = B_ge; rep < tot_ge; rep <<= 1) { AscendC::Adds(lt_ak[rep], lt_ak[0], 0.0f, rep); AscendC::PipeBarrier<PIPE_V>(); }
      AscendC::Sub(lt_num, lt_ak, lt_oad, cnt_ge_vec); AscendC::Sub(lt_ak, lt_ak, lt_dad, cnt_ge_vec); AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(lt_num, lt_num, lt_tau, cnt_ge_vec); AscendC::PipeBarrier<PIPE_V>();
      AscendC::Div(lt_ak, lt_num, lt_ak, cnt_ge_vec);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

      float num_k[16];
      for (uint32_t lane = 0; lane < b_cnt; ++lane) {
        int32_t k = static_cast<int32_t>(ge_i0 + cb + lane), idx_diag = k * B_ge + static_cast<int32_t>(lane);
        dad_p[idx_diag] = ad[s_my + k]; num_k[lane] = num_p[idx_diag]; ak_p[idx_diag] = 1.0f;
      }
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      int32_t kmy_bge = K_my * B_ge;
      if (tot_ge > kmy_bge) { AscendC::Duplicate(lt_ak[kmy_bge], 1.0f, tot_ge - kmy_bge); AscendC::PipeBarrier<PIPE_V>(); }
      for (int32_t h = K_ge_pow2 >> 1; h >= 4; h >>= 1) { AscendC::Mul(lt_ak, lt_ak, lt_ak[h * B_ge], h * B_ge); AscendC::PipeBarrier<PIPE_V>(); }
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

      const float SCALE_UP = 18446744073709551616.0f, SCALE_DN = 5.42101086242752217e-20f;
      for (uint32_t lane = 0; lane < b_cnt; ++lane) {
        int32_t k = static_cast<int32_t>(ge_i0 + cb + lane), exp_acc = 0;
        float ph = -num_k[lane] / rho_my;
        for (int32_t j = 0; j < 4; ++j) {
          ph *= ak_p[j * B_ge + static_cast<int32_t>(lane)];
          if (ph > SCALE_UP) { ph *= SCALE_DN; exp_acc += 64; }
          else if (ph < SCALE_DN) { ph *= SCALE_UP; exp_acc -= 64; }
        }
        float sgn_zk = (az[s_my + k] < 0.0f) ? -1.0f : 1.0f;
        if (ph > 0.0f) zhat_out[cb + lane] = sgn_zk * LdexpF32(MySqrt(ph), exp_acc >> 1);
      }
    }
  }

  __aicore__ __attribute__((always_inline)) inline void EvalLargeGuEisenstatAndSync(
    uint32_t p, uint32_t my_plo, uint32_t sub_P, uint32_t u_cap_my, int32_t s_my, int32_t K_my, float rho_my, uint32_t i0, uint32_t i1,
    const __ubuf__ int32_t* orig_ord, const __ubuf__ float* ad, __ubuf__ float* az, const __ubuf__ float* tau_a,
    __ubuf__ float* wide_scr, __ubuf__ float* stage_ub, __ubuf__ float* z_ub,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<float>& lt_stage,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_u32, const AscendC::GlobalTensor<float>& gt_zhat
  ) {
    float zhat_out[64];
    for (uint32_t ci = 0; ci < 64u; ++ci) zhat_out[ci] = 0.0f;
    if (K_my > 0 && i0 < i1) {
      if (K_my == 1) {
        float ph = -((ad[s_my] - ad[s_my + orig_ord[s_my]]) - tau_a[s_my]) / rho_my;
        if (ph > 0.0f) zhat_out[0] = ((az[s_my] < 0.0f) ? -1.0f : 1.0f) * MySqrt(ph);
      } else {
        EvalLargeGuEisenstatBatch(s_my, K_my, rho_my, i0, i1, orig_ord, ad, az, tau_a, zhat_out, wide_scr, stage_ub, z_ub, ub_all, lt_wscr, lt_stage, lt_z, lt_u32);
      }
    }
    AscendC::Duplicate(lt_stage, 0.0f, 32);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    for (uint32_t ci = 0; ci < i1 - i0; ++ci) stage_ub[ci] = zhat_out[ci];
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopy(gt_zhat[p * 32u], lt_stage, 32);
    SafeSyncAll();

    if (K_my > 0) {
      AscendC::DataCopy(lt_z, gt_zhat[my_plo * 32u], static_cast<int32_t>(sub_P * 32u));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      for (uint32_t sp = 0; sp < sub_P; ++sp) {
        uint32_t base_off = sp * 32u;
        for (uint32_t ci = 0; ci < u_cap_my; ++ci) {
          uint32_t gk = sp * u_cap_my + ci;
          if (gk < static_cast<uint32_t>(K_my)) az[s_my + gk] = z_ub[base_off + ci];
        }
      }
    }
    SafeSyncAll();
  }

  __aicore__ __attribute__((always_inline)) inline void InitLargeP3PanelsInUB(
    bool final_level, int32_t s_my, int32_t K_my, int32_t K16, int32_t K_pow2, int32_t B_p3, int32_t tot_p3,
    const __ubuf__ float* ad, const __ubuf__ float* az, const __ubuf__ float* ths, const __ubuf__ float* bhs, const __ubuf__ int32_t* acol_dc,
    __ubuf__ uint32_t* off_pfull, __ubuf__ float* wide_scr, __ubuf__ float* z_ub,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<float>& lt_ad_v,
    const AscendC::LocalTensor<float>& lt_az_v, const AscendC::LocalTensor<float>& lt_thsa, const AscendC::LocalTensor<float>& lt_bhsa,
    const AscendC::LocalTensor<uint32_t>& lt_u32
  ) {
    for (int32_t k = 0; k < K16; ++k) off_pfull[k] = static_cast<uint32_t>(k * B_p3 * 4);
    __ubuf__ uint32_t* off_p3_u32 = reinterpret_cast<__ubuf__ uint32_t*>(z_ub + 8192);
    AscendC::LocalTensor<uint32_t> lt_off_p3 = lt_u32[32256 + 8192];
    AscendC::LocalTensor<int32_t> lt_off_p3_i = ub_all.ReinterpretCast<int32_t>()[32256 + 8192];
    for (int32_t k = 0; k < K_my; ++k) {
      wide_scr[k] = ad[s_my + k]; wide_scr[512 + k] = az[s_my + k];
      if (!final_level) { int32_t ck = acol_dc[s_my + k]; wide_scr[1024 + k] = ths[s_my + ck]; wide_scr[1536 + k] = bhs[s_my + ck]; }
    }
    for (int32_t k = K_my; k < K_pow2; ++k) {
      wide_scr[k] = 1e20f; wide_scr[512 + k] = 0.0f;
      if (!final_level) { wide_scr[1024 + k] = 0.0f; wide_scr[1536 + k] = 0.0f; }
    }
    for (int32_t idx = 0; idx < 64; ++idx) off_p3_u32[idx] = static_cast<uint32_t>((idx / B_p3) * 4);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (int32_t rep = 64; rep < tot_p3; rep <<= 1) {
      AscendC::Adds(lt_off_p3_i[rep], lt_off_p3_i[0], (rep / B_p3) * 4, rep);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Gather(lt_ad_v, lt_wscr[0], lt_off_p3, 0u, static_cast<uint32_t>(tot_p3));
    AscendC::Gather(lt_az_v, lt_wscr[512], lt_off_p3, 0u, static_cast<uint32_t>(tot_p3));
    if (!final_level) {
      AscendC::Gather(lt_thsa, lt_wscr[1024], lt_off_p3, 0u, static_cast<uint32_t>(tot_p3));
      AscendC::Gather(lt_bhsa, lt_wscr[1536], lt_off_p3, 0u, static_cast<uint32_t>(tot_p3));
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void EvalLargeP3BatchCols(
    bool final_level, int32_t s_my, int32_t K_my, int32_t K16, int32_t K_pow2, int32_t B_p3, int32_t tot_p3, int32_t cnt_p3,
    uint32_t i0, uint32_t cb, uint32_t b_cnt, uint32_t ut_stride,
    const __ubuf__ int32_t* orig_ord, const __ubuf__ float* ad, const __ubuf__ float* tau_a, float* bnd_out,
    __ubuf__ float* col_vp, __ubuf__ float* v2_vp, __ubuf__ float* s_vp, __ubuf__ float* b2_vp,
    const AscendC::LocalTensor<float>& lt_ad_v, const AscendC::LocalTensor<float>& lt_az_v, const AscendC::LocalTensor<float>& lt_thsa,
    const AscendC::LocalTensor<float>& lt_bhsa, const AscendC::LocalTensor<float>& lt_col, const AscendC::LocalTensor<float>& lt_v2,
    const AscendC::LocalTensor<float>& lt_s, const AscendC::LocalTensor<float>& lt_b2, const AscendC::LocalTensor<float>& lt_u_rows,
    const AscendC::LocalTensor<int32_t>& lt_pfull_i32, const AscendC::LocalTensor<int32_t>& lt_tr_p3_i, const AscendC::LocalTensor<uint32_t>& lt_tr_p3,
    const AscendC::GlobalTensor<float>& gt_ut_m
  ) {
    for (int32_t lane = 0; lane < B_p3; ++lane) {
      uint32_t gi = i0 + cb + ((lane < static_cast<int32_t>(b_cnt)) ? static_cast<uint32_t>(lane) : 0u);
      col_vp[lane] = ad[s_my + orig_ord[s_my + gi]]; v2_vp[lane] = tau_a[s_my + gi];
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (int32_t rep = B_p3; rep < tot_p3; rep <<= 1) {
      AscendC::Adds(lt_col[rep], lt_col[0], 0.0f, rep); AscendC::Adds(lt_v2[rep], lt_v2[0], 0.0f, rep);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Sub(lt_s, lt_ad_v, lt_col, cnt_p3); AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(lt_s, lt_s, lt_v2, cnt_p3); AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(lt_col, lt_az_v, lt_s, cnt_p3); AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(lt_v2, lt_col, lt_col, cnt_p3); AscendC::PipeBarrier<PIPE_V>();

    if (tot_p3 > cnt_p3) { AscendC::Duplicate(lt_v2[cnt_p3], 0.0f, tot_p3 - cnt_p3); AscendC::PipeBarrier<PIPE_V>(); }
    for (int32_t h = K_pow2 >> 1; h >= 1; h >>= 1) { AscendC::Add(lt_v2, lt_v2, lt_v2[h * B_p3], h * B_p3); AscendC::PipeBarrier<PIPE_V>(); }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

    for (int32_t lane = 0; lane < B_p3; ++lane) s_vp[lane] = (lane < static_cast<int32_t>(b_cnt)) ? MyRsqrt(v2_vp[lane]) : 1.0f;
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (int32_t rep = B_p3; rep < tot_p3; rep <<= 1) { AscendC::Adds(lt_s[rep], lt_s[0], 0.0f, rep); AscendC::PipeBarrier<PIPE_V>(); }
    AscendC::Mul(lt_s, lt_col, lt_s, cnt_p3); AscendC::PipeBarrier<PIPE_V>();

    if (!final_level) {
      AscendC::Mul(lt_col, lt_thsa, lt_s, cnt_p3); AscendC::Mul(lt_b2, lt_bhsa, lt_s, cnt_p3); AscendC::PipeBarrier<PIPE_V>();
      if (tot_p3 > cnt_p3) { AscendC::Duplicate(lt_col[cnt_p3], 0.0f, tot_p3 - cnt_p3); AscendC::Duplicate(lt_b2[cnt_p3], 0.0f, tot_p3 - cnt_p3); AscendC::PipeBarrier<PIPE_V>(); }
      for (int32_t h = K_pow2 >> 1; h >= 1; h >>= 1) { AscendC::Add(lt_col, lt_col, lt_col[h * B_p3], h * B_p3); AscendC::Add(lt_b2, lt_b2, lt_b2[h * B_p3], h * B_p3); AscendC::PipeBarrier<PIPE_V>(); }
    }
    for (uint32_t lane = 0; lane < b_cnt; ++lane) AscendC::Adds(lt_tr_p3_i[lane * static_cast<uint32_t>(K16)], lt_pfull_i32, static_cast<int32_t>(lane * 4u), K16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(lt_u_rows, lt_s, lt_tr_p3, 0u, b_cnt * static_cast<uint32_t>(K16));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    if (!final_level) { AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0); }

    for (int32_t kb_p = 0; kb_p < K_my; kb_p += 16) {
      uint32_t w_kb = (K_my - kb_p <= 8) ? 8u : 16u;
      uint32_t panel_off = static_cast<uint32_t>(s_my) * ut_stride + static_cast<uint32_t>(kb_p) * static_cast<uint32_t>(K_my) + (i0 + cb) * w_kb;
      AscendC::DataCopyParams cp_p3_out(static_cast<uint16_t>(b_cnt), static_cast<uint16_t>(w_kb / 8u), static_cast<uint16_t>((static_cast<uint32_t>(K16) - w_kb) / 8u), 0);
      AscendC::DataCopy(gt_ut_m[panel_off], lt_u_rows[static_cast<uint32_t>(kb_p)], cp_p3_out);
    }
    if (!final_level) {
      for (uint32_t lane = 0; lane < b_cnt; ++lane) { bnd_out[cb + lane] = col_vp[lane]; bnd_out[32 + cb + lane] = b2_vp[lane]; }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void EvalLargeP3AndSyncSummaries(
    uint32_t p, uint32_t P, int32_t step, bool final_level, uint32_t num_merges, int32_t s_my, int32_t K_my, uint32_t i0, uint32_t i1, uint32_t ut_stride,
    const __ubuf__ int32_t* bounds, const __ubuf__ int32_t* K_arr, const __ubuf__ int32_t* D_arr, const __ubuf__ int32_t* triv_arr,
    const __ubuf__ float* ad, const __ubuf__ float* az, const __ubuf__ float* ths, const __ubuf__ float* bhs, const __ubuf__ float* tau_a,
    const __ubuf__ int32_t* acol_dc, __ubuf__ int32_t* orig_ord, __ubuf__ float* dd, __ubuf__ float* d, __ubuf__ float* top, __ubuf__ float* bot,
    __ubuf__ uint32_t* off_pfull, __ubuf__ float* wide_scr, __ubuf__ float* stage_ub, __ubuf__ float* z_ub,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<float>& lt_stage,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_u32,
    const AscendC::GlobalTensor<float>& gt_ut_m, const AscendC::GlobalTensor<float>& gt_bnd
  ) {
    float bnd_out[64];
    if (!final_level) for (int32_t idx = 0; idx < 64; ++idx) bnd_out[idx] = 0.0f;
    if (K_my > 0 && i0 < i1) {
      int32_t K16 = (K_my + 15) & ~15, K_pow2 = 16;
      while (K_pow2 < K16) K_pow2 <<= 1;
      int32_t B_p3 = static_cast<int32_t>(final_level ? 4096u : 2048u) / K_pow2;
      if (B_p3 > 16) B_p3 = 16;
      int32_t tot_p3 = K_pow2 * B_p3, cnt_p3 = K16 * B_p3;
      uint32_t off_az = final_level ? 4096u : 2048u, off_v2 = final_level ? 12288u : 10240u, tr_u32_off = final_level ? 40448u : 7168u;
      AscendC::LocalTensor<float> lt_ad_v = lt_z[0], lt_az_v = lt_z[off_az], lt_thsa = lt_z[4096], lt_bhsa = lt_z[6144];
      AscendC::LocalTensor<float> lt_col = lt_z[8192], lt_v2 = lt_z[off_v2], lt_s = final_level ? lt_wscr[0] : lt_z[12288], lt_b2 = lt_z[14336];
      __ubuf__ float* col_vp = z_ub + 8192, *v2_vp = z_ub + off_v2, *s_vp = final_level ? wide_scr : (z_ub + 12288), *b2_vp = z_ub + 14336;
      InitLargeP3PanelsInUB(final_level, s_my, K_my, K16, K_pow2, B_p3, tot_p3, ad, az, ths, bhs, acol_dc, off_pfull, wide_scr, z_ub, ub_all, lt_wscr, lt_ad_v, lt_az_v, lt_thsa, lt_bhsa, lt_u32);
      uint32_t n_mine = i1 - i0;
      AscendC::LocalTensor<int32_t> lt_pfull_i32 = ub_all.ReinterpretCast<int32_t>()[13312], lt_tr_p3_i = ub_all.ReinterpretCast<int32_t>()[tr_u32_off];
      AscendC::LocalTensor<float> lt_u_rows = final_level ? lt_z[12288] : lt_wscr[2048];
      for (uint32_t cb = 0; cb < n_mine; cb += static_cast<uint32_t>(B_p3)) {
        uint32_t b_cnt = (cb + static_cast<uint32_t>(B_p3) <= n_mine) ? static_cast<uint32_t>(B_p3) : (n_mine - cb);
        EvalLargeP3BatchCols(final_level, s_my, K_my, K16, K_pow2, B_p3, tot_p3, cnt_p3, i0, cb, b_cnt, ut_stride, orig_ord, ad, tau_a, bnd_out, col_vp, v2_vp, s_vp, b2_vp, lt_ad_v, lt_az_v, lt_thsa, lt_bhsa, lt_col, lt_v2, lt_s, lt_b2, lt_u_rows, lt_pfull_i32, lt_tr_p3_i, lt_u32[tr_u32_off], gt_ut_m);
      }
    }
    if (!final_level) {
      for (int32_t idx = 0; idx < 64; ++idx) stage_ub[idx] = bnd_out[idx];
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
      AscendC::DataCopy(gt_bnd[p * 64u], lt_stage, 64);
    }
    SafeSyncAll();
    SyncAndMergeLargeP3Summaries(P, step, final_level, num_merges, bounds, K_arr, D_arr, triv_arr, ad, ths, bhs, tau_a, acol_dc, orig_ord, dd, d, top, bot, z_ub, lt_z, gt_bnd);
  }
