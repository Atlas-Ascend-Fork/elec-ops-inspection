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

  __aicore__ __attribute__((always_inline)) inline void SolveMedRootsBlock8(
    int32_t s_my, int32_t K_my, float rho_my, uint32_t i0, uint32_t i1, uint32_t r_cnt,
    const float* ad, const float* az, int32_t* orig_ord, float* tau_a, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_st_ub,
    const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_st
  ) {
    if (K_my == 1) {
      orig_ord[s_my] = 0;
      tau_a[s_my]    = rho_my * (az[s_my] * az[s_my]);
      return;
    }
    int32_t K_pad = 16;
    while (K_pad < K_my) K_pad <<= 1;
    int32_t kpad_blk_i = K_pad >> 3, act_cnt = static_cast<int32_t>(r_cnt) * K_pad;
    int32_t orig_l[8], done_l[8];
    float gap_l[8], d_orig_l[8], low_l[8], high_l[8], tau_l[8], near_d[16], near_z2[16];
    int32_t j_base = (static_cast<int32_t>(i0) - 4 > 0) ? (static_cast<int32_t>(i0) - 4) : 0;
    int32_t j_end  = (static_cast<int32_t>(i1) + 4 < K_my) ? (static_cast<int32_t>(i1) + 4) : K_my;
    float sum_z2 = LoadSecularVectorAndNear(s_my, K_my, K_pad, j_base, j_end, ad, az, dc_ws_ub, dc_ws_ub + 64, near_d, near_z2);
    PrepareSecularMidpointProbe(i0, r_cnt, K_my, K_pad, j_base, 2.0f, near_d, done_l, gap_l, lt_dc_ws[0], lt_dc_ws[64], lt_dc_z[0], lt_dc_z[512], lt_dc_z[1024]);
    AscendC::Div(lt_dc_z[1536], lt_dc_z[512], lt_dc_z[1024], act_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, true>(lt_dc_st[0], lt_dc_z[1536], K_pad, static_cast<int32_t>(r_cnt), 1, 1, kpad_blk_i);
    AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    FinalizeSecularInitBrackets(
      i0, r_cnt, K_my, K_pad, j_base, rho_my, sum_z2, 1.0f, dc_st_ub, near_d,
      orig_l, done_l, gap_l, d_orig_l, low_l, high_l, tau_l, lt_dc_ws[0], lt_dc_ws[64], lt_dc_z[0], lt_dc_z[512]
    );
    AscendC::PipeBarrier<PIPE_V>();
    uint8_t kp_u8 = static_cast<uint8_t>(kpad_blk_i);
    AscendC::UnaryRepeatParams u_kp(1, 1, kp_u8, kp_u8);
    AscendC::BinaryRepeatParams b_kp(1, 1, 1, kp_u8, kp_u8, kp_u8);
    for (int32_t it = 0; it < 22; ++it) {
      bool all_done = true;
      for (uint32_t lane = 0; lane < r_cnt; ++lane) if (!done_l[lane]) all_done = false;
      if (all_done) break;
      EvalSecularSums8(r_cnt, K_pad, kpad_blk_i, done_l, tau_l, lt_dc_z[0], lt_dc_z[512], lt_dc_z[1024], lt_dc_z[1536], lt_dc_z[2048], lt_dc_st, u_kp, b_kp);
      UpdateMedSecularLanes(i0, r_cnt, K_my, j_base, rho_my, dc_st_ub, near_d, near_z2, d_orig_l, done_l, low_l, high_l, tau_l);
    }
    for (uint32_t lane = 0; lane < r_cnt; ++lane) {
      orig_ord[s_my + i0 + lane] = orig_l[lane];
      tau_a[s_my + i0 + lane]    = tau_l[lane];
    }
  }

  __aicore__ __attribute__((always_inline)) inline void SolveMedRootsAndSync(
    uint32_t p, int32_t step, bool final_level, uint32_t num_merges, uint32_t sub_P, uint32_t my_m, uint32_t sub_p,
    int32_t s1_tot_len, const int32_t* s1_triv, const int32_t* bounds, const int32_t* K_arr, const float* rho_arr,
    const float* ad, const float* az, int32_t* orig_ord, float* tau_a, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_st_ub,
    const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<float>& lt_dc_ws,
    const AscendC::LocalTensor<float>& lt_dc_st, const AscendC::GlobalTensor<float>& gt_ws_leaf,
    const AscendC::GlobalTensor<float>& gt_ws_roots_s1, const AscendC::GlobalTensor<float>& gt_ws_roots_s2
  ) {
    int32_t s_my = bounds[static_cast<int32_t>(my_m) * 2 * step], K_my = K_arr[my_m];
    float rho_my = rho_arr[my_m];
    uint32_t i0 = sub_p * 8u;
    uint32_t i1 = (K_my > 0 && i0 < static_cast<uint32_t>(K_my))
      ? ((i0 + 8u < static_cast<uint32_t>(K_my)) ? (i0 + 8u) : static_cast<uint32_t>(K_my)) : i0;
    uint32_t r_cnt = i1 - i0;

    if (K_my > 0 && r_cnt > 0u) {
      SolveMedRootsBlock8(s_my, K_my, rho_my, i0, i1, r_cnt, ad, az, orig_ord, tau_a, dc_ws_ub, dc_st_ub, lt_dc_z, lt_dc_ws, lt_dc_st);
      AscendC::Duplicate<float, true>(lt_dc_st[0], 0.0f, 32ULL, 1, 1, 8);
      AscendC::ResetMask();
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      __ubuf__ int32_t* st_orig = reinterpret_cast<__ubuf__ int32_t*>(dc_st_ub);
      for (uint32_t ci = 0; ci < r_cnt; ++ci) {
        st_orig[ci]        = orig_ord[s_my + i0 + ci];
        dc_st_ub[16u + ci] = tau_a[s_my + i0 + ci];
      }
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    } else {
      AscendC::Duplicate<float, true>(lt_dc_st[0], 0.0f, 32ULL, 1, 1, 8);
      AscendC::ResetMask();
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    }
    AscendC::DataCopy(final_level ? gt_ws_roots_s2[p * 32u] : gt_ws_roots_s1[p * 32u], lt_dc_st[0], 32);
    SafeSyncAll();

    AscendC::DataCopy(lt_dc_ws[0], final_level ? gt_ws_roots_s2[0] : gt_ws_roots_s1[0], 256);
    if (final_level && (!s1_triv[0] || !s1_triv[1])) AscendC::DataCopy(lt_ztmp[0], gt_ws_leaf[0], s1_tot_len);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    for (uint32_t m_idx = 0; m_idx < num_merges; ++m_idx) {
      int32_t Km = K_arr[m_idx];
      if (Km <= 0) continue;
      int32_t sm = bounds[static_cast<int32_t>(m_idx) * 2 * step];
      for (uint32_t gi = 0u; gi < static_cast<uint32_t>(Km); ++gi) {
        uint32_t sp = gi >> 3u, ci = gi & 7u, base_off = (m_idx * sub_P + sp) * 32u;
        orig_ord[sm + gi] = reinterpret_cast<__ubuf__ int32_t*>(dc_ws_ub + base_off)[ci];
        tau_a[sm + gi]    = dc_ws_ub[base_off + 16u + ci];
      }
    }
  }

  __aicore__ __attribute__((always_inline)) inline void EvalStep1GuEisenstat32(
    int32_t sm, int32_t Km, float rho_m, const int32_t* acol_dc, const int32_t* orig_ord, const float* ad, float* az, const float* tau_a, const float* ths, const float* bhs, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_z_ub, const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2, const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<uint32_t>& lt_off_bc64, const AscendC::LocalTensor<uint16_t>& lt_u16, uint32_t dcz_u16_base, const AscendC::LocalTensor<uint16_t>& lt_dmsk_u16, const AscendC::LocalTensor<uint16_t>& lt_eye64_u16, const AscendC::BinaryRepeatParams& b_bc64_row0, const AscendC::BinaryRepeatParams& b_def64
  ) {
    int32_t K_pad = 8;
    while (K_pad < Km) K_pad <<= 1;
    uint32_t tot_64 = static_cast<uint32_t>(K_pad) * 64u;
    uint8_t kp_rep64 = static_cast<uint8_t>(K_pad);
    AscendC::Duplicate<float, true>(lt_dc_ws[0], 2e10f, 64ULL, 1, 1, 8);
    AscendC::Duplicate<float, false>(lt_dc_ws[64], 0.0f, 64ULL, 5, 1, 8);
    AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    float num_k[32];
    for (int32_t k = 0; k < Km; ++k) {
      int32_t ck = acol_dc[sm + k], ok = orig_ord[sm + k];
      float ad_k  = ad[sm + k], ad_ok = ad[sm + ok], tau_k = tau_a[sm + k];
      num_k[k]          = (ad_k - ad_ok) - tau_k;
      dc_ws_ub[0   + k] = ad_k;
      dc_ws_ub[128 + k] = ths[sm + ck];
      dc_ws_ub[192 + k] = bhs[sm + ck];
      dc_ws_ub[256 + k] = ad_ok;
      dc_ws_ub[320 + k] = tau_k;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

    AscendC::Gather(lt_ztmp[0],    lt_dc_ws[256], lt_off_bc64, 0u, tot_64);
    AscendC::Gather(lt_ztmp[2048], lt_dc_ws[320], lt_off_bc64, 0u, tot_64);
    AscendC::Gather(lt_ztmp2[0],   lt_dc_ws[0],   lt_off_bc64, 0u, tot_64);
    AscendC::PipeBarrier<PIPE_V>();

    EvalGuEisenstatProd64(
      Km, K_pad, tot_64, kp_rep64, lt_dc_ws[0], lt_ztmp[0], lt_ztmp[2048], lt_ztmp2[0], lt_dc_z[0], lt_dc_z[0], lt_dc_z[4096],
      lt_u16[dcz_u16_base], lt_u16[dcz_u16_base + 8192u], lt_dmsk_u16, lt_eye64_u16, b_bc64_row0, b_def64
    );
    __ubuf__ float* ak_p = dc_z_ub + 4096;
    for (int32_t k = 0; k < Km; ++k) {
      float az_k = FinalizeGuEisenstatElem(num_k[k], rho_m, ak_p[k], az[sm + k]);
      az[sm + k]       = az_k;
      dc_ws_ub[64 + k] = az_k;
    }
  }

  __aicore__ __attribute__((always_inline)) inline void EvalMedStep1SecularCols(
    uint32_t m_idx, uint32_t my_m, uint32_t b0_my, uint32_t c_off, int32_t Km, float* top_sec, float* bot_sec, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_z_ub,
    const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2, const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<float>& lt_dc_st,
    const AscendC::LocalTensor<uint32_t>& lt_off_bc64, const AscendC::LocalTensor<uint32_t>& lt_off_rep8_4k, const AscendC::BinaryRepeatParams& b_bc64_s1, const AscendC::BinaryRepeatParams& b_def64, const AscendC::BinaryRepeatParams& b_s1_32, const AscendC::BinaryRepeatParams& rk_s1_32
  ) {
    int32_t K_pad = 8;
    while (K_pad < Km) K_pad <<= 1;
    uint32_t tot_64  = static_cast<uint32_t>(K_pad) * 64u;
    uint8_t kp_rep64 = static_cast<uint8_t>(K_pad);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

    EvalSecularColsAndBoundaries64(
      Km, K_pad, tot_64, kp_rep64, false, dc_ws_ub, dc_z_ub + 6144,
      lt_dc_ws, lt_dc_z[0], lt_dc_z[2048], lt_dc_z[4096], lt_dc_z[6144], lt_off_bc64, b_bc64_s1, b_def64
    );
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    for (int32_t col = 0; col < Km; ++col) {
      top_sec[col] = dc_z_ub[0 + col];
      bot_sec[col] = dc_z_ub[2048 + col];
    }
    if (m_idx == my_m && b0_my < static_cast<uint32_t>(Km)) {
      uint32_t b_cnt = (b0_my + 8u <= static_cast<uint32_t>(Km)) ? 8u : (static_cast<uint32_t>(Km) - b0_my);
      AscendC::UnaryRepeatParams pack_u8(1, 1, 1, 8);
      AscendC::Adds<float, true>(lt_dc_st[0], lt_dc_z[4096u + b0_my], 0.0f, 8ULL, static_cast<uint8_t>(Km), pack_u8);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Gather(lt_dc_z[6144], lt_dc_st[0], lt_off_rep8_4k, 0u, static_cast<uint32_t>(Km) * 64u);
      AscendC::PipeBarrier<PIPE_V>();
      AccumRank1Cols4x(
        Km, 32ULL, static_cast<uint8_t>(b_cnt), lt_ztmp[b0_my * 64u + c_off], lt_ztmp2[c_off], lt_dc_z[6144],
        lt_dc_st[256], lt_dc_st[768], lt_dc_st[1280], rk_s1_32, b_s1_32
      );
    }
  }

  __aicore__ __attribute__((always_inline)) inline void GatherMedStep1ActiveCols(
    int32_t sm, int32_t Lm, int32_t Km, int32_t Dm, uint32_t b0_my, uint32_t c_off, const int32_t* acol_dc,
    const AscendC::LocalTensor<float>& lt_t64, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2, const AscendC::UnaryRepeatParams& u_s1_32
  ) {
    AscendC::Adds<float, true>(lt_t64, lt_t64, 0.0f, 32ULL, 1, u_s1_32);
    AscendC::PipeBarrier<PIPE_V>();
    if (b0_my < static_cast<uint32_t>(Km)) {
      for (int32_t k = 0; k < Km; ++k) AscendC::Adds<float, false>(lt_ztmp2[static_cast<uint32_t>(k) * 64u + c_off], lt_z[static_cast<uint32_t>(sm + acol_dc[sm + k]) * 64u + c_off], 0.0f, 32ULL, 1, u_s1_32);
    }
    for (int32_t di = 0; di < Dm; ++di) {
      uint32_t col_idx = static_cast<uint32_t>(Km + di);
      if (col_idx >= b0_my && col_idx < b0_my + 8u) {
        AscendC::Adds<float, false>(lt_ztmp[col_idx * 64u + c_off], lt_z[static_cast<uint32_t>(sm + acol_dc[sm + Lm - 1 - di]) * 64u + c_off], 0.0f, 32ULL, 1, u_s1_32);
      }
    }
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline void FinalizeMedStep1MergeOrder(
    uint32_t m_idx, uint32_t s1_m1_off, int32_t sm, int32_t Lm, int32_t Km, int32_t Dm,
    const float* ad, const float* tau_a, const float* top_sec, const float* bot_sec, const float* ths, const float* bhs,
    const int32_t* acol_dc, int32_t* orig_ord, float* dd, float* d, float* top, float* bot, uint32_t* s1_src
  ) {
    for (int32_t di = Dm - 1; di >= 0; --di) dd[sm + Km + di] = dd[sm + di];
    for (int32_t i = 0; i < Km; ++i) dd[sm + i] = ad[sm + orig_ord[sm + i]] + tau_a[sm + i];
    MergeTwoSortedLocal(Km, Lm, &orig_ord[sm], &dd[sm]);
    uint32_t m_base_s1 = (m_idx == 0u) ? 0u : s1_m1_off;
    for (int32_t col_out = 0; col_out < Lm; ++col_out) {
      int32_t oc = orig_ord[sm + col_out], ck = (oc < Km) ? 0 : acol_dc[sm + Lm - 1 - (oc - Km)];
      d[sm + col_out] = dd[sm + oc];
      top[sm + col_out] = (oc < Km) ? top_sec[oc] : ths[sm + ck];
      bot[sm + col_out] = (oc < Km) ? bot_sec[oc] : bhs[sm + ck];
      s1_src[sm + col_out] = m_base_s1 + static_cast<uint32_t>(oc) * 32u;
    }
  }

  __aicore__ __attribute__((always_inline)) inline void UpdateMedStep1SingleMerge(
    uint32_t m_idx, int32_t step, uint32_t my_m, uint32_t sub_p, uint32_t s1_m1_off, const int32_t* bounds, const int32_t* K_arr, const int32_t* D_arr, const int32_t* triv_arr, const float* rho_arr, int32_t* s1_triv, uint32_t* s1_src, float* d, float* top, float* bot, const float* ths, const float* bhs, const float* ad, float* az, float* dd, const float* tau_a, int32_t* orig_ord, const int32_t* acol_dc, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_z_ub,
    const AscendC::LocalTensor<float>& lt_t64, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2, const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<float>& lt_dc_st, const AscendC::LocalTensor<uint32_t>& lt_off_bc64, const AscendC::LocalTensor<uint32_t>& lt_off_rep8_4k, const AscendC::LocalTensor<uint16_t>& lt_u16, const AscendC::LocalTensor<uint16_t>& lt_dmsk_u16, const AscendC::LocalTensor<uint16_t>& lt_eye64_u16, const AscendC::GlobalTensor<float>& gt_ws_leaf,
    const AscendC::UnaryRepeatParams& u_s1_32, const AscendC::BinaryRepeatParams& b_bc64_s1, const AscendC::BinaryRepeatParams& b_bc64_row0, const AscendC::BinaryRepeatParams& b_def64, const AscendC::BinaryRepeatParams& b_s1_32, const AscendC::BinaryRepeatParams& rk_s1_32
  ) {
    int32_t b_idx = static_cast<int32_t>(m_idx) * 2 * step, sm = bounds[b_idx], Lm = bounds[b_idx + 2 * step] - sm, Km = K_arr[m_idx], Dm = D_arr[m_idx];
    float rho_m = rho_arr[m_idx];
    uint32_t c_off = static_cast<uint32_t>(sm);
    if (Km == 0) {
      s1_triv[m_idx] = 1;
      if (!triv_arr[m_idx]) {
        AscendC::Adds<float, true>(lt_ztmp[c_off], lt_z[static_cast<uint32_t>(sm) * 64u + c_off], 0.0f, 32ULL, static_cast<uint8_t>(Lm), u_s1_32);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t col_out = 0; col_out < Lm; ++col_out) {
          int32_t ck = acol_dc[sm + Lm - 1 - col_out];
          d[sm + col_out] = dd[sm + col_out]; top[sm + col_out] = ths[sm + ck]; bot[sm + col_out] = bhs[sm + ck];
          AscendC::Adds<float, false>(lt_z[static_cast<uint32_t>(sm + col_out) * 64u + c_off], lt_ztmp[static_cast<uint32_t>(ck) * 64u + c_off], 0.0f, 32ULL, 1, u_s1_32);
        }
        AscendC::ResetMask();
        AscendC::PipeBarrier<PIPE_V>();
      }
      return;
    }
    s1_triv[m_idx] = 0;
    uint32_t b0_my = sub_p * 8u;
    if (Km == 1) {
      EIGH_PAD256() EIGH_PAD64() EIGH_PAD16() EIGH_PAD16() EIGH_PAD16() EIGH_PAD4() EIGH_PAD4()
      float ph = -((ad[sm] - ad[sm + orig_ord[sm]]) - tau_a[sm]) / rho_m;
      az[sm] = (ph > 0.0f) ? (((az[sm] < 0.0f) ? -1.0f : 1.0f) * MySqrt(ph)) : 0.0f;
    } else {
      EvalStep1GuEisenstat32(sm, Km, rho_m, acol_dc, orig_ord, ad, az, tau_a, ths, bhs, dc_ws_ub, dc_z_ub, lt_ztmp, lt_ztmp2, lt_dc_ws, lt_dc_z, lt_off_bc64, lt_u16, 24576u * 2u, lt_dmsk_u16, lt_eye64_u16, b_bc64_row0, b_def64);
    }
    if (m_idx == my_m && b0_my < static_cast<uint32_t>(Lm)) GatherMedStep1ActiveCols(sm, Lm, Km, Dm, b0_my, c_off, acol_dc, lt_t64, lt_z, lt_ztmp, lt_ztmp2, u_s1_32);
    float top_sec[32], bot_sec[32];
    if (Km == 1) {
      int32_t ck0 = acol_dc[sm];
      top_sec[0] = ths[sm + ck0]; bot_sec[0] = bhs[sm + ck0];
      if (m_idx == my_m && b0_my == 0u) {
        AscendC::Adds<float, true>(lt_ztmp[c_off], lt_ztmp2[c_off], 0.0f, 32ULL, 1, u_s1_32);
        AscendC::ResetMask(); AscendC::PipeBarrier<PIPE_V>();
      }
    } else {
      EvalMedStep1SecularCols(m_idx, my_m, b0_my, c_off, Km, top_sec, bot_sec, dc_ws_ub, dc_z_ub, lt_ztmp, lt_ztmp2, lt_dc_ws, lt_dc_z, lt_dc_st, lt_off_bc64, lt_off_rep8_4k, b_bc64_s1, b_def64, b_s1_32, rk_s1_32);
    }
    if (m_idx == my_m && b0_my < static_cast<uint32_t>(Lm)) {
      uint32_t my_s1_off = ((my_m == 0u) ? 0u : s1_m1_off) + sub_p * 256u;
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::DataCopy(gt_ws_leaf[my_s1_off], lt_ztmp[b0_my * 64u + c_off], AscendC::DataCopyParams(8, 4, 4, 0));
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
    FinalizeMedStep1MergeOrder(m_idx, s1_m1_off, sm, Lm, Km, Dm, ad, tau_a, top_sec, bot_sec, ths, bhs, acol_dc, orig_ord, dd, d, top, bot, s1_src);
  }

  __aicore__ __attribute__((always_inline)) inline void EvalStep2GuEisenstat64(
    int32_t Km, float rho_m, uint32_t oad_off, uint32_t tau_off, uint32_t zero_reps, const int32_t* orig_ord, const float* ad, const float* az, const float* tau_a, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_z_ub, const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2, const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<uint32_t>& lt_off_bc64, const AscendC::LocalTensor<uint16_t>& lt_u16, uint32_t dcz_u16_base, const AscendC::LocalTensor<uint16_t>& lt_dmsk_u16, const AscendC::LocalTensor<uint16_t>& lt_eye64_u16, const AscendC::BinaryRepeatParams& b_bc64_row0, const AscendC::BinaryRepeatParams& b_def64
  ) {
    int32_t K_pad = 8;
    while (K_pad < Km) K_pad <<= 1;
    uint8_t kp_rep64 = static_cast<uint8_t>(K_pad), zr_u8 = static_cast<uint8_t>(zero_reps);
    uint32_t tot_64 = static_cast<uint32_t>(K_pad) * 64u;
    AscendC::Duplicate<float, true>(lt_dc_ws[0], 2e10f, 64ULL, 1, 1, 8);
    AscendC::Duplicate<float, false>(lt_dc_ws[64], 0.0f, 64ULL, zr_u8, 1, 8);
    AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    float num_k[64];
    for (int32_t k = 0; k < Km; ++k) {
      int32_t ok = orig_ord[k];
      float ad_k = ad[k], ad_ok = ad[ok], tau_k = tau_a[k];
      num_k[k]                = (ad_k - ad_ok) - tau_k;
      dc_ws_ub[0       + k]   = ad_k;
      dc_ws_ub[oad_off + k]   = ad_ok;
      dc_ws_ub[tau_off + k]   = tau_k;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

    AscendC::Gather(lt_ztmp[0],  lt_dc_ws[oad_off], lt_off_bc64, 0u, tot_64);
    AscendC::Gather(lt_ztmp2[0], lt_dc_ws[tau_off], lt_off_bc64, 0u, tot_64);
    AscendC::Gather(lt_dc_z[0],  lt_dc_ws[0],       lt_off_bc64, 0u, tot_64);
    AscendC::PipeBarrier<PIPE_V>();

    EvalGuEisenstatProd64(
      Km, K_pad, tot_64, kp_rep64, lt_dc_ws[0], lt_ztmp[0], lt_ztmp2[0], lt_dc_z[0], lt_ztmp[0], lt_dc_z[0], lt_dc_z[4096],
      lt_u16[dcz_u16_base], lt_u16[dcz_u16_base + 8192u], lt_dmsk_u16, lt_eye64_u16, b_bc64_row0, b_def64
    );
    __ubuf__ float* ak_p = dc_z_ub + 4096;
    for (int32_t k = 0; k < Km; ++k) {
      dc_ws_ub[64 + k] = FinalizeGuEisenstatElem(num_k[k], rho_m, ak_p[k], az[k]);
    }
  }

  __aicore__ __attribute__((always_inline)) inline void EvalMedStep2SecularCols(
    uint32_t col_start, uint32_t my_cols, uint8_t mc_u8, int32_t Km, float rho_m, const int32_t* final_oc, const int32_t* acol_dc, const int32_t* orig_ord, const float* ad, const float* az, const float* tau_a, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_z_ub,
    const AscendC::LocalTensor<float>& lt_t64, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2, const AscendC::LocalTensor<float>& lt_z_final, const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_z,
    const AscendC::LocalTensor<uint32_t>& lt_off_bc64, const AscendC::LocalTensor<uint32_t>& lt_off_rep8, const AscendC::LocalTensor<uint32_t>& lt_off_rep8_4k, const AscendC::LocalTensor<uint16_t>& lt_u16, const AscendC::LocalTensor<uint16_t>& lt_dmsk_u16, const AscendC::LocalTensor<uint16_t>& lt_eye64_u16,
    const AscendC::UnaryRepeatParams& u_def64, const AscendC::BinaryRepeatParams& b_bc8_s1, const AscendC::BinaryRepeatParams& b_bc64_row0, const AscendC::BinaryRepeatParams& b_def64, const AscendC::BinaryRepeatParams& rk_mul64
  ) {
    int32_t n_i32 = static_cast<int32_t>(n_);
    uint32_t sec_cnt = 0u, sec_c[8];
    int32_t sec_oc[8];
    for (uint32_t c = 0; c < my_cols; ++c) {
      int32_t oc = final_oc[col_start + c];
      if (oc < Km) { sec_oc[sec_cnt] = oc; sec_c[sec_cnt] = c; sec_cnt++; }
    }
    if (sec_cnt > 0u) {
      EvalStep2GuEisenstat64(Km, rho_m, 64u, 128u, 2u, orig_ord, ad, az, tau_a, dc_ws_ub, dc_z_ub, lt_ztmp, lt_ztmp2, lt_dc_ws, lt_dc_z, lt_off_bc64, lt_u16, 24576u * 2u, lt_dmsk_u16, lt_eye64_u16, b_bc64_row0, b_def64);
      int32_t K_pad = 8;
      while (K_pad < Km) K_pad <<= 1;
      uint32_t tot_ge = static_cast<uint32_t>(K_pad) * 8u;
      uint8_t k_rep   = static_cast<uint8_t>(K_pad >> 3);
      for (uint32_t lane = 0; lane < 8u; ++lane) {
        int32_t gi = sec_oc[(lane < sec_cnt) ? lane : 0u];
        dc_ws_ub[256 + lane] = ad[orig_ord[gi]]; dc_ws_ub[264 + lane] = tau_a[gi];
      }
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::Gather(lt_dc_z[0],   lt_dc_ws[0],  lt_off_rep8, 0u, tot_ge);
      AscendC::Gather(lt_dc_z[512], lt_dc_ws[64], lt_off_rep8, 0u, tot_ge);
      AscendC::PipeBarrier<PIPE_V>();
      EvalSecularBlock8Cols(Km, K_pad, k_rep, sec_cnt, true, acol_dc, dc_ws_ub, dc_z_ub + 2560, lt_dc_ws, lt_dc_z[0], lt_dc_z[512], lt_dc_z[2048], lt_dc_z[2560], lt_dc_z[3072], lt_t64, lt_ztmp2, lt_z, lt_dc_z[4096], lt_off_rep8_4k, u_def64, b_bc8_s1, b_def64);
      AccumRank1Cols4x(Km, 64ULL, static_cast<uint8_t>(sec_cnt), lt_ztmp[0], lt_ztmp2, lt_dc_z[4096], lt_dc_z[8192], lt_dc_z[8704], lt_dc_z[9216], rk_mul64, b_def64);
    }
    AscendC::Adds<float, true>(lt_t64, lt_t64, 0.0f, 64ULL, 1, u_def64);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t c = 0; c < my_cols; ++c) {
      int32_t oc = final_oc[col_start + c];
      if (oc >= Km) AscendC::Adds<float, false>(lt_z_final[c * 64u], lt_z[static_cast<uint32_t>(acol_dc[n_i32 - 1 - (oc - Km)]) * 64u], 0.0f, 64ULL, 1, u_def64);
    }
    for (uint32_t lane = 0; lane < sec_cnt; ++lane) AscendC::Adds<float, false>(lt_z_final[sec_c[lane] * 64u], lt_ztmp[lane * 64u], 0.0f, 64ULL, 1, u_def64);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<float, true>(lt_z[0], lt_z_final[0], 0.0f, 64ULL, mc_u8, u_def64);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline void UnpackAndRotateMedStep2(
    const int32_t* bounds, const int32_t* s1_triv, const uint32_t* s1_src, int32_t n_giv, const int32_t* giv_cp, const int32_t* giv_pk, const float* giv_cg, const float* giv_sg,
    const AscendC::LocalTensor<float>& lt_t64, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_dc_st,
    const AscendC::UnaryRepeatParams& u_def64, const AscendC::UnaryRepeatParams& u_s1_32
  ) {
    if (!s1_triv[0] || !s1_triv[1]) {
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::Adds<float, true>(lt_t64, lt_t64, 0.0f, 32ULL, 1, u_s1_32);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t m_s1 = 0; m_s1 < 2u; ++m_s1) {
        if (s1_triv[m_s1]) continue;
        int32_t sm_s1 = bounds[m_s1 * 2], em_s1 = bounds[m_s1 * 2 + 2];
        uint32_t coff_s1 = static_cast<uint32_t>(sm_s1);
        for (int32_t col_s1 = sm_s1; col_s1 < em_s1; ++col_s1) {
          AscendC::Adds<float, false>(lt_z[static_cast<uint32_t>(col_s1) * 64u + coff_s1], lt_ztmp[s1_src[col_s1]], 0.0f, 32ULL, 1, u_s1_32);
        }
      }
      AscendC::ResetMask();
      AscendC::PipeBarrier<PIPE_V>();
    }
    for (int32_t g = 0; g < n_giv; ++g) {
      RotateTwoVecRowsAxpy(lt_z[static_cast<uint32_t>(giv_cp[g]) * 64u], lt_z[static_cast<uint32_t>(giv_pk[g]) * 64u], lt_dc_st, giv_cg[g], giv_sg[g], 64ULL, u_def64);
    }
  }

  __aicore__ __attribute__((always_inline)) inline void MergeAndSortMedStep2Eigenvals(
    int32_t n_i32, int32_t Km, int32_t Dm, int32_t triv0, float scale_pow2,
    float* d, const float* ad, float* dd, const float* tau_a, const int32_t* orig_ord, const int32_t* acol_dc, int32_t* final_oc
  ) {
    if (triv0) {
      for (int32_t c = 0; c < n_i32; ++c) final_oc[c] = c;
    } else if (Km == 0) {
      for (int32_t c = 0; c < n_i32; ++c) { d[c] = dd[c]; final_oc[c] = acol_dc[n_i32 - 1 - c]; }
    } else {
      for (int32_t di = Dm - 1; di >= 0; --di) dd[Km + di] = dd[di];
      for (int32_t i = 0; i < Km; ++i) dd[i] = ad[orig_ord[i]] + tau_a[i];
      MergeTwoSortedLocal(Km, n_i32, final_oc, dd);
      for (int32_t c = 0; c < n_i32; ++c) d[c] = dd[final_oc[c]];
    }
    for (uint32_t i = 0; i < n_; ++i) d[i] *= scale_pow2;
    SortEigenPairsInplace(n_, d, final_oc);
  }
