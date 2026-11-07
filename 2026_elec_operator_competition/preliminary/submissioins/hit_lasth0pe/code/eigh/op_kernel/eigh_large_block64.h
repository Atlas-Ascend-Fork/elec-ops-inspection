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

  __aicore__ __attribute__((always_inline)) inline void InitAndUnpackBlock64Leaves(
    int32_t L_b, int32_t mid_b, uint32_t my_blk, uint32_t leaf_gm_off, int32_t* lb,
    float* d_loc, float* top_loc, float* bot_loc, __ubuf__ float* dc_st_ub,
    __ubuf__ uint32_t* dmsk_u32, __ubuf__ int32_t* bc64_i32, __ubuf__ uint32_t* rep8_u32, __ubuf__ uint32_t* eye64_u,
    const AscendC::LocalTensor<float>& lt_dc_st, const AscendC::LocalTensor<float>& lt_z_t,
    const AscendC::LocalTensor<int32_t>& lt_dmsk_i32, const AscendC::LocalTensor<int32_t>& lt_bc64_i32,
    const AscendC::LocalTensor<int32_t>& lt_rep8_i32, const AscendC::LocalTensor<int32_t>& lt_eye64_i32,
    const AscendC::GlobalTensor<float>& gt_sym
  ) {
    AscendC::DataCopy(lt_dc_st[0], gt_sym[leaf_gm_off + my_blk * 1280u], 1280);
    AscendC::Duplicate(lt_z_t, 0.0f, 4096);
    AscendC::Duplicate(lt_dmsk_i32, -1, 4096);
    AscendC::Duplicate(lt_eye64_i32, 0, 4096);
    for (uint32_t idx = 0; idx < 64u; ++idx) {
      bc64_i32[idx] = 0;
      rep8_u32[idx] = (idx >> 3u) * 4u;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t rep = 1; rep < 64; rep <<= 1) {
      AscendC::Adds(lt_bc64_i32[rep * 64], lt_bc64_i32[0], rep * 4, rep * 64);
      AscendC::PipeBarrier<PIPE_V>();
    }
    for (int32_t rep = 64; rep < 4096; rep <<= 1) {
      AscendC::Adds(lt_rep8_i32[rep], lt_rep8_i32[0], (rep >> 3) * 4, rep);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    for (uint32_t k = 0; k < 64u; ++k) {
      dmsk_u32[k * 65u] = 0u;
      eye64_u[k * 65u]  = 0x3f800000u;
    }
    lb[0] = 0; lb[1] = mid_b - 16; lb[2] = mid_b; lb[3] = (L_b > 56) ? 48 : (mid_b + 8); lb[4] = L_b;

    AscendC::UnaryRepeatParams u_leaf16(1, 1, 8, 2);
    for (int32_t lf = 0; lf < 4; ++lf) {
      int32_t s_l = lb[lf], m_l = lb[lf + 1] - s_l;
      uint32_t sl_u = static_cast<uint32_t>(s_l);
      __ubuf__ float* l_pack = dc_st_ub + static_cast<uint32_t>(lf) * 320u;
      for (int32_t col = 0; col < m_l; ++col) {
        d_loc[s_l + col] = l_pack[0 + col]; top_loc[s_l + col] = l_pack[16 + col]; bot_loc[s_l + col] = l_pack[32 + col];
      }
      AscendC::Adds<float, true>(lt_z_t[sl_u * 64u + sl_u], lt_dc_st[static_cast<uint32_t>(lf) * 320u + 64u], 0.0f, static_cast<uint64_t>(m_l), static_cast<uint8_t>(m_l), u_leaf16);
    }
    AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline void DeflateBlock64SingleMerge(
    uint32_t m_idx, int32_t step, bool is_lv2, const int32_t* lb, const float* e_cuts,
    float* d_loc, const float* top_loc, const float* bot_loc, float* ths_loc, float* bhs_loc, float* ad_loc, float* az_loc, float* dd_loc,
    int32_t* orig_loc, int32_t* acol_loc, int32_t* K_arr, int32_t* D_arr, int32_t* triv_arr, float* rho_arr,
    const AscendC::LocalTensor<float>& lt_z_t, const AscendC::LocalTensor<float>& lt_ztmp2,
    const AscendC::UnaryRepeatParams& u_def64, const AscendC::UnaryRepeatParams& u_s1_32,
    const AscendC::BinaryRepeatParams& b_def64, const AscendC::BinaryRepeatParams& b_s1_32
  ) {
    int32_t b_idx = static_cast<int32_t>(m_idx) * 2 * step;
    int32_t s = lb[b_idx], mid = lb[b_idx + step], end = lb[b_idx + 2 * step], L = end - s, m1 = mid - s;
    float r_raw = is_lv2 ? e_cuts[1] : e_cuts[m_idx * 2u];
    bool neg = (r_raw < 0.0f);
    float ar = neg ? -r_raw : r_raw;
    rho_arr[m_idx] = ar;
    triv_arr[m_idx] = 0;

    MergeTwoSortedLocal(m1, L, &orig_loc[s], &d_loc[s]);
    float dl_p0[64], zk_p0[64];
    float ref_sc = BuildMergePermAndZ(s, m1, L, neg, ar, ar == 0.0f, true, orig_loc, d_loc, top_loc, bot_loc, ths_loc, bhs_loc, dl_p0, zk_p0);
    uint32_t c_off = is_lv2 ? 0u : static_cast<uint32_t>(s);
    uint64_t rot_m = is_lv2 ? 64ULL : 32ULL;
    const AscendC::UnaryRepeatParams& u_rot  = is_lv2 ? u_def64 : u_s1_32;
    const AscendC::BinaryRepeatParams& b_rot = is_lv2 ? b_def64 : b_s1_32;
    float tol_z = 8.5e-7f * ref_sc, tol_d = 1.8e-6f * ref_sc;
    int32_t K = 0, D = 0, last_cp = 0;
    float last_ad = 0.0f, last_az = 0.0f;

    for (int32_t k = 0; k < L; ++k) {
      int32_t pk = orig_loc[s + k];
      float dk = dl_p0[k], zk = zk_p0[k];
      if (MyAbs(zk) <= tol_z) {
        dd_loc[s + D] = dk; acol_loc[s + L - 1 - D] = pk; D++;
      } else if (K > 0 && MyAbs(dk - last_ad) <= tol_d) {
        float zp = last_az, r_hyp = MySqrt(zp * zp + zk * zk);
        int32_t cp = last_cp;
        float c_g = (r_hyp != 0.0f) ? (zk / r_hyp) : 1.0f, s_g = (r_hyp != 0.0f) ? (zp / r_hyp) : 0.0f;
        RotateTwoScalars(ths_loc[s + cp], ths_loc[s + pk], c_g, s_g);
        RotateTwoScalars(bhs_loc[s + cp], bhs_loc[s + pk], c_g, s_g);
        RotateTwoVecRowsMuls(lt_z_t[static_cast<uint32_t>(s + cp) * 64u + c_off], lt_z_t[static_cast<uint32_t>(s + pk) * 64u + c_off], lt_ztmp2, c_g, s_g, rot_m, u_rot, b_rot);
        dd_loc[s + D] = last_ad; acol_loc[s + L - 1 - D] = cp; D++;
        ad_loc[s + K - 1] = dk; az_loc[s + K - 1] = r_hyp; acol_loc[s + K - 1] = pk;
        last_ad = dk; last_az = r_hyp; last_cp = pk;
      } else {
        ad_loc[s + K] = dk; az_loc[s + K] = zk; acol_loc[s + K] = pk;
        last_ad = dk; last_az = zk; last_cp = pk; K++;
      }
    }
    K_arr[m_idx] = K; D_arr[m_idx] = D;
  }

  __aicore__ __attribute__((always_inline)) inline void SolveBlock64RootsBatch8(
    uint32_t b0, uint32_t r_cnt, int32_t s_my, int32_t K_my, int32_t K_pad, int32_t kpad_blk_i, float rho_my, float sum_z2,
    const float* ad_loc, const float* az_loc, int32_t* orig_loc, float* tau_loc, __ubuf__ float* dc_st_ub, __ubuf__ float* dc_z_ub,
    const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_z,
    const AscendC::LocalTensor<float>& lt_dc_st, const AscendC::LocalTensor<float>& lt_absw,
    const AscendC::UnaryRepeatParams& u_kp, const AscendC::BinaryRepeatParams& b_kp
  ) {
    int32_t act_cnt = static_cast<int32_t>(r_cnt) * K_pad;
    int32_t orig_l[8], done_l[8];
    float gap_l[8], d_orig_l[8], low_l[8], high_l[8], tau_l[8], near_d[24], near_z2[24];
    int32_t j_base = (static_cast<int32_t>(b0) - 4 > 0) ? (static_cast<int32_t>(b0) - 4) : 0;
    int32_t j_end  = (static_cast<int32_t>(b0 + r_cnt) + 4 < K_my) ? (static_cast<int32_t>(b0 + r_cnt) + 4) : K_my;
    for (int32_t jj = 0; jj < j_end - j_base; ++jj) {
      int32_t g_idx = s_my + j_base + jj;
      float zv = az_loc[g_idx];
      near_d[jj] = ad_loc[g_idx]; near_z2[jj] = zv * zv;
    }
    PrepareSecularMidpointProbe(b0, r_cnt, K_my, K_pad, j_base, 2.0f, near_d, done_l, gap_l, lt_dc_ws[0], lt_dc_ws[64], lt_dc_z[0], lt_dc_z[512], lt_dc_z[1024]);
    AscendC::Div(lt_dc_z[1536], lt_dc_z[512], lt_dc_z[1024], act_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, true>(lt_dc_st[0], lt_dc_z[1536], K_pad, static_cast<int32_t>(r_cnt), 1, 1, kpad_blk_i);
    AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    FinalizeSecularInitBrackets(
      b0, r_cnt, K_my, K_pad, j_base, rho_my, sum_z2, 2.0f, dc_st_ub, near_d,
      orig_l, done_l, gap_l, d_orig_l, low_l, high_l, tau_l, lt_dc_ws[0], lt_dc_ws[64], lt_dc_z[0], lt_dc_z[512]
    );
    MaskSecularNearPoles(b0, r_cnt, K_my, K_pad, done_l, dc_z_ub + 512);
    for (int32_t it = 0; it < 26; ++it) {
      bool all_done = true;
      for (uint32_t lane = 0; lane < r_cnt; ++lane) if (!done_l[lane]) all_done = false;
      if (all_done) break;
      EvalSecularSums8(r_cnt, K_pad, kpad_blk_i, done_l, tau_l, lt_dc_z[0], lt_dc_z[512], lt_dc_z[1024], lt_dc_z[1536], lt_absw, lt_dc_st, u_kp, b_kp);
      UpdateSecularLanesSplit(b0, r_cnt, K_my, j_base, rho_my, 8u, dc_st_ub, near_d, near_z2, orig_l, gap_l, done_l, low_l, high_l, tau_l);
    }
    for (uint32_t lane = 0; lane < r_cnt; ++lane) {
      orig_loc[s_my + b0 + lane] = orig_l[lane];
      tau_loc[s_my + b0 + lane]  = tau_l[lane];
    }
  }

  __aicore__ __attribute__((always_inline)) inline void SolveBlock64RootsAndSync(
    int32_t step, bool is_lv2, uint32_t num_merges, uint32_t p, uint32_t plo_b, uint32_t bp, uint32_t subP_b, const int32_t* lb, const int32_t* K_arr, const float* rho_arr, const float* ad_loc, const float* az_loc, int32_t* orig_loc, float* tau_loc, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_st_ub, __ubuf__ float* dc_z_ub, const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_st, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<float>& lt_absw, const AscendC::GlobalTensor<float>& gt_sym
  ) {
    AscendC::Duplicate<float, true>(lt_dc_st[1920], 0.0f, 64ULL, 1, 1, 8); AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    __ubuf__ int32_t* st_pack_i = reinterpret_cast<__ubuf__ int32_t*>(dc_st_ub + 1920);
    __ubuf__ float* st_pack_f   = dc_st_ub + 1920;

    for (uint32_t m_idx = 0; m_idx < num_merges; ++m_idx) {
      int32_t s_my = lb[static_cast<int32_t>(m_idx) * 2 * step], K_my = K_arr[m_idx];
      float rho_my = rho_arr[m_idx];
      if (K_my <= 0) continue;
      uint32_t u_cap = (static_cast<uint32_t>(K_my) + subP_b - 1u) / subP_b, i0 = bp * u_cap;
      uint32_t i1 = i0 + u_cap;
      if (i1 > static_cast<uint32_t>(K_my)) i1 = static_cast<uint32_t>(K_my);
      if (i0 >= i1) continue;
      if (K_my == 1) {
        orig_loc[s_my] = 0; tau_loc[s_my] = rho_my * (az_loc[s_my] * az_loc[s_my]);
      } else {
        int32_t K_pad = 16;
        while (K_pad < K_my) K_pad <<= 1;
        int32_t kpad_blk_i = K_pad >> 3;
        uint8_t kp_u8 = static_cast<uint8_t>(kpad_blk_i);
        AscendC::UnaryRepeatParams u_kp(1, 1, kp_u8, kp_u8);
        AscendC::BinaryRepeatParams b_kp(1, 1, 1, kp_u8, kp_u8, kp_u8);
        float sum_z2 = LoadSecularVectorAndNear(s_my, K_my, K_pad, 0, 0, ad_loc, az_loc, dc_ws_ub, dc_ws_ub + 64, nullptr, nullptr);
        for (uint32_t b0 = i0; b0 < i1; b0 += 8u) SolveBlock64RootsBatch8(b0, (b0 + 8u <= i1) ? 8u : (i1 - b0), s_my, K_my, K_pad, kpad_blk_i, rho_my, sum_z2, ad_loc, az_loc, orig_loc, tau_loc, dc_st_ub, dc_z_ub, lt_dc_ws, lt_dc_z, lt_dc_st, lt_absw, u_kp, b_kp);
      }
      uint32_t m_slot = is_lv2 ? 0u : (m_idx * 16u);
      for (uint32_t ci = 0; ci < i1 - i0; ++ci) { st_pack_i[m_slot + ci] = orig_loc[s_my + i0 + ci]; st_pack_f[32u + m_slot + ci] = tau_loc[s_my + i0 + ci]; }
    }
    uint32_t roots_gm_base = is_lv2 ? 3072u : 0u;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopy(gt_sym[roots_gm_base + p * 64u], lt_dc_st[1920], 64);
    SafeSyncAll();

    AscendC::DataCopy(lt_dc_ws[0], gt_sym[roots_gm_base + plo_b * 64u], static_cast<int32_t>(subP_b * 64u));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    for (uint32_t m_idx = 0; m_idx < num_merges; ++m_idx) {
      int32_t Km = K_arr[m_idx];
      if (Km <= 0) continue;
      int32_t sm = lb[static_cast<int32_t>(m_idx) * 2 * step];
      uint32_t u_cap = (static_cast<uint32_t>(Km) + subP_b - 1u) / subP_b, m_slot = is_lv2 ? 0u : (m_idx * 16u);
      for (uint32_t gi = 0u; gi < static_cast<uint32_t>(Km); ++gi) {
        uint32_t sp = gi / u_cap, ci = gi % u_cap, base_off = sp * 64u;
        orig_loc[sm + gi] = reinterpret_cast<__ubuf__ int32_t*>(dc_ws_ub + base_off)[m_slot + ci];
        tau_loc[sm + gi]  = dc_ws_ub[base_off + 32u + m_slot + ci];
      }
    }
  }

  __aicore__ __attribute__((always_inline)) inline void EvalBlock64Step1SecularCols(
    int32_t Km, uint32_t c_off, float* top_sec, float* bot_sec, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_z_ub,
    const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2,
    const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<float>& lt_dc_st,
    const AscendC::LocalTensor<uint32_t>& lt_off_bc64, const AscendC::LocalTensor<uint32_t>& lt_off_rep8,
    const AscendC::BinaryRepeatParams& b_bc64_s1, const AscendC::BinaryRepeatParams& b_def64,
    const AscendC::BinaryRepeatParams& b_s1_32, const AscendC::BinaryRepeatParams& rk_s1_32
  ) {
    int32_t K_pad = 8;
    while (K_pad < Km) K_pad <<= 1;
    uint32_t tot_64  = static_cast<uint32_t>(K_pad) * 64u;
    uint8_t kp_rep64 = static_cast<uint8_t>(K_pad);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    EvalSecularColsAndBoundaries64(Km, K_pad, tot_64, kp_rep64, false, dc_ws_ub, dc_z_ub + 6144, lt_dc_ws, lt_dc_z[0], lt_dc_z[2048], lt_dc_z[4096], lt_dc_z[6144], lt_off_bc64, b_bc64_s1, b_def64);
    for (uint32_t b0 = 0u; b0 < static_cast<uint32_t>(Km); b0 += 8u) {
      uint32_t b_cnt = (b0 + 8u <= static_cast<uint32_t>(Km)) ? 8u : (static_cast<uint32_t>(Km) - b0);
      AscendC::UnaryRepeatParams pack_u8(1, 1, 1, 8);
      AscendC::Adds<float, true>(lt_dc_st[0], lt_dc_z[4096u + b0], 0.0f, 8ULL, static_cast<uint8_t>(Km), pack_u8);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Gather(lt_dc_z[6144], lt_dc_st[0], lt_off_rep8, 0u, static_cast<uint32_t>(Km) * 64u);
      AscendC::PipeBarrier<PIPE_V>();
      AccumRank1Cols4x(Km, 32ULL, static_cast<uint8_t>(b_cnt), lt_ztmp[b0 * 64u + c_off], lt_ztmp2[c_off], lt_dc_z[6144], lt_dc_st[256], lt_dc_st[768], lt_dc_st[1280], rk_s1_32, b_s1_32);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    for (int32_t col = 0; col < Km; ++col) {
      top_sec[col] = dc_z_ub[0 + col];
      bot_sec[col] = dc_z_ub[2048 + col];
    }
  }

  __aicore__ __attribute__((always_inline)) inline void UpdateBlock64Step1SingleMerge(
    uint32_t m_idx, const int32_t* lb, const int32_t* K_arr, const int32_t* D_arr, const int32_t* triv_arr, const float* rho_arr, float* d_loc, float* top_loc, float* bot_loc, const float* ths_loc, const float* bhs_loc, const float* ad_loc, float* az_loc, float* dd_loc, const float* tau_loc, int32_t* orig_loc, const int32_t* acol_loc, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_z_ub, const AscendC::LocalTensor<float>& lt_t64, const AscendC::LocalTensor<float>& lt_z_t, const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2, const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<float>& lt_dc_st, const AscendC::LocalTensor<uint32_t>& lt_off_bc64, const AscendC::LocalTensor<uint32_t>& lt_off_rep8, const AscendC::LocalTensor<uint16_t>& lt_u16, const AscendC::LocalTensor<uint16_t>& lt_dmsk_u16, const AscendC::LocalTensor<uint16_t>& lt_eye64_u16, const AscendC::UnaryRepeatParams& u_s1_32, const AscendC::BinaryRepeatParams& b_bc64_s1, const AscendC::BinaryRepeatParams& b_bc64_row0, const AscendC::BinaryRepeatParams& b_def64, const AscendC::BinaryRepeatParams& b_s1_32, const AscendC::BinaryRepeatParams& rk_s1_32
  ) {
    int32_t b_idx = static_cast<int32_t>(m_idx) * 2, sm = lb[b_idx], Lm = lb[b_idx + 2] - sm, Km = K_arr[m_idx], Dm = D_arr[m_idx];
    float rho_m = rho_arr[m_idx];
    uint32_t c_off = static_cast<uint32_t>(sm);
    if (Km == 0) {
      if (!triv_arr[m_idx]) {
        AscendC::Adds<float, true>(lt_ztmp[c_off], lt_z_t[static_cast<uint32_t>(sm) * 64u + c_off], 0.0f, 32ULL, static_cast<uint8_t>(Lm), u_s1_32);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t col_out = 0; col_out < Lm; ++col_out) {
          int32_t ck = acol_loc[sm + Lm - 1 - col_out];
          d_loc[sm + col_out] = dd_loc[sm + col_out]; top_loc[sm + col_out] = ths_loc[sm + ck]; bot_loc[sm + col_out] = bhs_loc[sm + ck];
          AscendC::Adds<float, false>(lt_z_t[static_cast<uint32_t>(sm + col_out) * 64u + c_off], lt_ztmp[static_cast<uint32_t>(ck) * 64u + c_off], 0.0f, 32ULL, 1, u_s1_32);
        }
        AscendC::ResetMask(); AscendC::PipeBarrier<PIPE_V>();
      }
      return;
    }
    if (Km == 1) {
      float ph = -((ad_loc[sm] - ad_loc[sm + orig_loc[sm]]) - tau_loc[sm]) / rho_m;
      az_loc[sm] = (ph > 0.0f) ? (((az_loc[sm] < 0.0f) ? -1.0f : 1.0f) * MySqrt(ph)) : 0.0f;
    } else {
      EvalStep1GuEisenstat32(sm, Km, rho_m, acol_loc, orig_loc, ad_loc, az_loc, tau_loc, ths_loc, bhs_loc, dc_ws_ub, dc_z_ub, lt_ztmp, lt_ztmp2, lt_dc_ws, lt_dc_z, lt_off_bc64, lt_u16, 36352u * 2u, lt_dmsk_u16, lt_eye64_u16, b_bc64_row0, b_def64);
    }
    AscendC::Adds<float, true>(lt_t64, lt_t64, 0.0f, 32ULL, 1, u_s1_32); AscendC::PipeBarrier<PIPE_V>();
    for (int32_t k = 0; k < Km; ++k) AscendC::Adds<float, false>(lt_ztmp2[static_cast<uint32_t>(k) * 64u + c_off], lt_z_t[static_cast<uint32_t>(sm + acol_loc[sm + k]) * 64u + c_off], 0.0f, 32ULL, 1, u_s1_32);
    for (int32_t di = 0; di < Dm; ++di) AscendC::Adds<float, false>(lt_ztmp[static_cast<uint32_t>(Km + di) * 64u + c_off], lt_z_t[static_cast<uint32_t>(sm + acol_loc[sm + Lm - 1 - di]) * 64u + c_off], 0.0f, 32ULL, 1, u_s1_32);
    AscendC::ResetMask(); AscendC::PipeBarrier<PIPE_V>();

    float top_sec[32], bot_sec[32];
    if (Km == 1) {
      int32_t ck0 = acol_loc[sm];
      top_sec[0] = ths_loc[sm + ck0]; bot_sec[0] = bhs_loc[sm + ck0];
      AscendC::Adds<float, true>(lt_ztmp[c_off], lt_ztmp2[c_off], 0.0f, 32ULL, 1, u_s1_32);
      AscendC::ResetMask(); AscendC::PipeBarrier<PIPE_V>();
    } else {
      EvalBlock64Step1SecularCols(Km, c_off, top_sec, bot_sec, dc_ws_ub, dc_z_ub, lt_ztmp, lt_ztmp2, lt_dc_ws, lt_dc_z, lt_dc_st, lt_off_bc64, lt_off_rep8, b_bc64_s1, b_def64, b_s1_32, rk_s1_32);
    }
    for (int32_t di = Dm - 1; di >= 0; --di) dd_loc[sm + Km + di] = dd_loc[sm + di];
    for (int32_t i = 0; i < Km; ++i) dd_loc[sm + i] = ad_loc[sm + orig_loc[sm + i]] + tau_loc[sm + i];
    MergeTwoSortedLocal(Km, Lm, &orig_loc[sm], &dd_loc[sm]);
    AscendC::Adds<float, true>(lt_t64, lt_t64, 0.0f, 32ULL, 1, u_s1_32); AscendC::PipeBarrier<PIPE_V>();
    for (int32_t col_out = 0; col_out < Lm; ++col_out) {
      int32_t oc = orig_loc[sm + col_out];
      d_loc[sm + col_out] = dd_loc[sm + oc];
      if (oc < Km) { top_loc[sm + col_out] = top_sec[oc]; bot_loc[sm + col_out] = bot_sec[oc]; }
      else { int32_t ck = acol_loc[sm + Lm - 1 - (oc - Km)]; top_loc[sm + col_out] = ths_loc[sm + ck]; bot_loc[sm + col_out] = bhs_loc[sm + ck]; }
      AscendC::Adds<float, false>(lt_z_t[static_cast<uint32_t>(sm + col_out) * 64u + c_off], lt_ztmp[static_cast<uint32_t>(oc) * 64u + c_off], 0.0f, 32ULL, 1, u_s1_32);
    }
    AscendC::ResetMask(); AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline void EvalBlock64Step2SecularCols(
    int32_t L_b, int32_t Km, float rho_m, uint32_t bp, uint32_t sec_cnt,
    const int32_t* sec_oc, const int32_t* final_oc, const int32_t* acol_loc, const int32_t* orig_loc,
    const float* ad_loc, const float* az_loc, const float* tau_loc, const float* ths_loc, const float* bhs_loc, float* top_loc, float* bot_loc,
    __ubuf__ float* base_ptr, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_z_ub,
    const AscendC::LocalTensor<float>& lt_t64, const AscendC::LocalTensor<float>& lt_z_t, const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2,
    const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<float>& lt_dc_st,
    const AscendC::LocalTensor<uint32_t>& lt_off_bc64, const AscendC::LocalTensor<uint32_t>& lt_off_rep8,
    const AscendC::LocalTensor<uint16_t>& lt_u16, const AscendC::LocalTensor<uint16_t>& lt_dmsk_u16, const AscendC::LocalTensor<uint16_t>& lt_eye64_u16,
    const AscendC::UnaryRepeatParams& u_def64, const AscendC::BinaryRepeatParams& b_bc8_s1, const AscendC::BinaryRepeatParams& b_bc64_s1,
    const AscendC::BinaryRepeatParams& b_bc64_row0, const AscendC::BinaryRepeatParams& b_def64, const AscendC::BinaryRepeatParams& rk_mul64
  ) {
    if (bp != 0u && sec_cnt == 0u) return;
    EvalStep2GuEisenstat64(Km, rho_m, 256u, 320u, 5u, orig_loc, ad_loc, az_loc, tau_loc, dc_ws_ub, dc_z_ub, lt_ztmp, lt_ztmp2, lt_dc_ws, lt_dc_z, lt_off_bc64, lt_u16, 36352u * 2u, lt_dmsk_u16, lt_eye64_u16, b_bc64_row0, b_def64);
    int32_t K_pad = 8;
    while (K_pad < Km) K_pad <<= 1;
    uint32_t tot_64  = static_cast<uint32_t>(K_pad) * 64u;
    uint8_t kp_rep64 = static_cast<uint8_t>(K_pad);

    if (bp == 0u) {
      for (int32_t k = 0; k < Km; ++k) {
        int32_t ck = acol_loc[k];
        dc_ws_ub[128 + k] = ths_loc[ck]; dc_ws_ub[192 + k] = bhs_loc[ck];
      }
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      EvalSecularColsAndBoundaries64(Km, K_pad, tot_64, kp_rep64, true, dc_ws_ub, dc_z_ub + 4096, lt_dc_ws, lt_ztmp[0], lt_ztmp2[0], lt_dc_z[0], lt_dc_z[4096], lt_off_bc64, b_bc64_s1, b_def64);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      __ubuf__ float* th_sec_p = base_ptr + 32256;
      __ubuf__ float* bh_sec_p = base_ptr + 15872 + 4096;
      for (int32_t c = 0; c < L_b; ++c) {
        int32_t oc = final_oc[c], ck = (oc < Km) ? 0 : acol_loc[L_b - 1 - (oc - Km)];
        top_loc[c] = (oc < Km) ? th_sec_p[oc] : ths_loc[ck];
        bot_loc[c] = (oc < Km) ? bh_sec_p[oc] : bhs_loc[ck];
      }
    }
    if (sec_cnt > 0u) {
      uint32_t tot_ge = static_cast<uint32_t>(K_pad) * 8u;
      uint8_t k_rep   = static_cast<uint8_t>(K_pad >> 3);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::Gather(lt_dc_z[0],   lt_dc_ws[0],  lt_off_rep8, 0u, tot_ge);
      AscendC::Gather(lt_dc_z[512], lt_dc_ws[64], lt_off_rep8, 0u, tot_ge);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t sb = 0u; sb < sec_cnt; sb += 8u) {
        uint32_t b_cnt = (sb + 8u <= sec_cnt) ? 8u : (sec_cnt - sb);
        for (uint32_t lane = 0; lane < 8u; ++lane) {
          int32_t gi = sec_oc[sb + ((lane < b_cnt) ? lane : 0u)];
          dc_ws_ub[256 + lane] = ad_loc[orig_loc[gi]]; dc_ws_ub[264 + lane] = tau_loc[gi];
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        EvalSecularBlock8Cols(Km, K_pad, k_rep, b_cnt, sb == 0u, acol_loc, dc_ws_ub, dc_z_ub + 1536, lt_dc_ws, lt_dc_z[0], lt_dc_z[512], lt_dc_z[1024], lt_dc_z[1536], lt_dc_z[2048], lt_t64, lt_ztmp2, lt_z_t, lt_dc_z[4096], lt_off_rep8, u_def64, b_bc8_s1, b_def64);
        AccumRank1Cols4x(Km, 64ULL, static_cast<uint8_t>(b_cnt), lt_ztmp[sb * 64u], lt_ztmp2, lt_dc_z[4096], lt_dc_st[256], lt_dc_st[768], lt_dc_st[1280], rk_mul64, b_def64);
      }
    }
  }

  __aicore__ __attribute__((always_inline)) inline void WriteBackBlock64Summaries(
    int32_t L_b, uint32_t my_blk, uint32_t bp, const float* d_loc, const float* top_loc, const float* bot_loc,
    __ubuf__ float* dc_ws_ub, const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::GlobalTensor<float>& gt_sym
  ) {
    if (bp != 0u) return;
    for (int32_t c = 0; c < L_b; ++c) {
      dc_ws_ub[0 + c] = d_loc[c]; dc_ws_ub[64 + c] = top_loc[c]; dc_ws_ub[128 + c] = bot_loc[c];
    }
    for (int32_t c = L_b; c < 64; ++c) {
      dc_ws_ub[0 + c] = 0.0f; dc_ws_ub[64 + c] = 0.0f; dc_ws_ub[128 + c] = 0.0f;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopy(gt_sym[6144u + my_blk * 192u], lt_dc_ws[0], 192);
  }

  __aicore__ __attribute__((always_inline)) inline void WriteBackBlock64FinalCols(
    int32_t L_b, int32_t Km, uint32_t my_blk, uint32_t blk_slot, uint32_t c_b0, uint32_t my_bcols, uint32_t sec_cnt, const uint32_t* sec_c,
    const int32_t* final_oc, const int32_t* acol_loc, const AscendC::LocalTensor<float>& lt_t64, const AscendC::LocalTensor<float>& lt_z_t,
    const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::GlobalTensor<float>& gt_sym, const AscendC::UnaryRepeatParams& u_def64
  ) {
    if (my_bcols == 0u) return;
    AscendC::Adds<float, true>(lt_t64, lt_t64, 0.0f, 64ULL, 1, u_def64);
    AscendC::PipeBarrier<PIPE_V>();
    if (Km <= 1) {
      for (uint32_t c = 0; c < my_bcols; ++c) AscendC::Adds<float, false>(lt_dc_z[c * 64u], lt_z_t[static_cast<uint32_t>(final_oc[c_b0 + c]) * 64u], 0.0f, 64ULL, 1, u_def64);
    } else {
      for (uint32_t c = 0; c < my_bcols; ++c) {
        int32_t oc = final_oc[c_b0 + c];
        if (oc >= Km) AscendC::Adds<float, false>(lt_dc_z[c * 64u], lt_z_t[static_cast<uint32_t>(acol_loc[L_b - 1 - (oc - Km)]) * 64u], 0.0f, 64ULL, 1, u_def64);
      }
      for (uint32_t si = 0; si < sec_cnt; ++si) AscendC::Adds<float, false>(lt_dc_z[sec_c[si] * 64u], lt_ztmp[si * 64u], 0.0f, 64ULL, 1, u_def64);
    }
    AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::DataCopy(gt_sym[8192u + my_blk * blk_slot + c_b0 * 64u], lt_dc_z[0], static_cast<int32_t>(my_bcols * 64u));
  }

  __aicore__ __attribute__((always_inline)) inline void UpdateBlock64Step2Final(
    int32_t L_b, uint32_t my_blk, uint32_t bp, uint32_t subP_b, uint32_t blk_slot, const int32_t* K_arr, const int32_t* D_arr, const int32_t* triv_arr, const float* rho_arr, float* d_loc, float* top_loc, float* bot_loc, const float* ths_loc, const float* bhs_loc, const float* ad_loc, const float* az_loc, float* dd_loc, const float* tau_loc, const int32_t* orig_loc, const int32_t* acol_loc, int32_t* final_oc, __ubuf__ float* base_ptr, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_z_ub,
    const AscendC::LocalTensor<float>& lt_t64, const AscendC::LocalTensor<float>& lt_z_t, const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2, const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<float>& lt_dc_st, const AscendC::LocalTensor<uint32_t>& lt_off_bc64, const AscendC::LocalTensor<uint32_t>& lt_off_rep8, const AscendC::LocalTensor<uint16_t>& lt_u16, const AscendC::LocalTensor<uint16_t>& lt_dmsk_u16, const AscendC::LocalTensor<uint16_t>& lt_eye64_u16, const AscendC::GlobalTensor<float>& gt_sym,
    const AscendC::UnaryRepeatParams& u_def64, const AscendC::BinaryRepeatParams& b_bc8_s1, const AscendC::BinaryRepeatParams& b_bc64_s1, const AscendC::BinaryRepeatParams& b_bc64_row0, const AscendC::BinaryRepeatParams& b_def64, const AscendC::BinaryRepeatParams& rk_mul64
  ) {
    int32_t Km = K_arr[0], Dm = D_arr[0], sec_oc[32];
    float rho_m = rho_arr[0];
    uint32_t c_b0 = (bp * static_cast<uint32_t>(L_b)) / subP_b, c_b1 = ((bp + 1u) * static_cast<uint32_t>(L_b)) / subP_b, my_bcols = c_b1 - c_b0, sec_cnt = 0u, sec_c[32];
    if (triv_arr[0]) {
      EIGH_PAD256() EIGH_PAD64() EIGH_PAD16() EIGH_PAD16() EIGH_PAD16() EIGH_PAD4() EIGH_PAD4()
      for (int32_t c = 0; c < L_b; ++c) final_oc[c] = c;
    } else if (Km == 0) {
      for (int32_t c = 0; c < L_b; ++c) {
        int32_t ck = acol_loc[L_b - 1 - c];
        d_loc[c] = dd_loc[c]; top_loc[c] = ths_loc[ck]; bot_loc[c] = bhs_loc[ck]; final_oc[c] = ck;
      }
    } else {
      for (int32_t di = Dm - 1; di >= 0; --di) dd_loc[Km + di] = dd_loc[di];
      for (int32_t i = 0; i < Km; ++i) dd_loc[i] = ad_loc[orig_loc[i]] + tau_loc[i];
      MergeTwoSortedLocal(Km, L_b, final_oc, dd_loc);
      for (int32_t c = 0; c < L_b; ++c) d_loc[c] = dd_loc[final_oc[c]];
    }
    if (Km == 1) {
      int32_t ck0 = acol_loc[0];
      for (int32_t c = 0; c < L_b; ++c) {
        int32_t oc = final_oc[c], ck = (oc < 1) ? ck0 : acol_loc[L_b - 1 - (oc - 1)];
        top_loc[c] = ths_loc[ck]; bot_loc[c] = bhs_loc[ck]; final_oc[c] = ck;
      }
    } else if (Km > 1) {
      for (uint32_t c = 0; c < my_bcols; ++c) {
        int32_t oc = final_oc[c_b0 + c];
        if (oc < Km) { sec_oc[sec_cnt] = oc; sec_c[sec_cnt] = c; sec_cnt++; }
      }
      EvalBlock64Step2SecularCols(L_b, Km, rho_m, bp, sec_cnt, sec_oc, final_oc, acol_loc, orig_loc, ad_loc, az_loc, tau_loc, ths_loc, bhs_loc, top_loc, bot_loc, base_ptr, dc_ws_ub, dc_z_ub, lt_t64, lt_z_t, lt_ztmp, lt_ztmp2, lt_dc_ws, lt_dc_z, lt_dc_st, lt_off_bc64, lt_off_rep8, lt_u16, lt_dmsk_u16, lt_eye64_u16, u_def64, b_bc8_s1, b_bc64_s1, b_bc64_row0, b_def64, rk_mul64);
    }
    WriteBackBlock64FinalCols(L_b, Km, my_blk, blk_slot, c_b0, my_bcols, sec_cnt, sec_c, final_oc, acol_loc, lt_t64, lt_z_t, lt_ztmp, lt_dc_z, gt_sym, u_def64);
    WriteBackBlock64Summaries(L_b, my_blk, bp, d_loc, top_loc, bot_loc, dc_ws_ub, lt_dc_ws, gt_sym);
  }

  __aicore__ __attribute__((noinline)) void MergeBlock64InUB(
    int32_t L_b, int32_t mid_b, float e_c0, float e_c1, float e_c2,
    uint32_t p, uint32_t my_blk, uint32_t plo_b, uint32_t bp, uint32_t subP_b, uint32_t blk_slot, uint32_t leaf_gm_off,
    __gm__ float* cur_vec, uint32_t sym_cap, const AscendC::LocalTensor<float>& ub_all
  ) {
    __ubuf__ float* base_ptr = reinterpret_cast<__ubuf__ float*>(ub_all.GetPhyAddr());
    __ubuf__ float* dc_ws_ub = base_ptr + 9216, *dc_st_ub = base_ptr + 13824, *dc_z_ub = base_ptr + 36352;
    __ubuf__ uint32_t* u32_base = reinterpret_cast<__ubuf__ uint32_t*>(base_ptr);

    float d_loc[64], top_loc[64], bot_loc[64], ad_loc[64], az_loc[64], dd_loc[64], ths_loc[64], bhs_loc[64], tau_loc[64], e_cuts[3] = {e_c0, e_c1, e_c2};
    int32_t orig_loc[64], acol_loc[64], final_oc[64], lb[5];

    AscendC::LocalTensor<float> lt_dc_ws = ub_all[9216], lt_dc_st = ub_all[13824], lt_t64 = lt_dc_st[1984];
    AscendC::LocalTensor<float> lt_z_t = ub_all[15872], lt_ztmp2 = ub_all[15872 + 4096], lt_ztmp = ub_all[32256], lt_dc_z = ub_all[36352], lt_absw = lt_dc_z[2048];
    AscendC::LocalTensor<int32_t> lt_i32 = ub_all.ReinterpretCast<int32_t>();
    AscendC::LocalTensor<uint32_t> lt_u32 = ub_all.ReinterpretCast<uint32_t>(), lt_off_bc64 = lt_u32[5120], lt_off_rep8 = lt_u32[9728];
    AscendC::LocalTensor<uint16_t> lt_u16 = ub_all.ReinterpretCast<uint16_t>(), lt_dmsk_u16 = lt_u16[1024u * 2u], lt_eye64_u16 = lt_u16[44544u * 2u];

    AscendC::GlobalTensor<float> gt_sym;
    gt_sym.SetGlobalBuffer(cur_vec, sym_cap);

    AscendC::UnaryRepeatParams u_def64(1, 1, 8, 8), u_s1_32(1, 1, 8, 8);
    AscendC::BinaryRepeatParams b_def64(1, 1, 1, 8, 8, 8), rk_mul64(1, 1, 0, 8, 0, 1), b_bc8_s1(1, 1, 0, 8, 8, 0);
    AscendC::BinaryRepeatParams b_bc64_s1(1, 1, 1, 8, 8, 0), b_bc64_row0(1, 1, 1, 8, 0, 8), b_s1_32(1, 1, 1, 8, 8, 8), rk_s1_32(1, 1, 0, 8, 0, 1);

    InitAndUnpackBlock64Leaves(
      L_b, mid_b, my_blk, leaf_gm_off, lb, d_loc, top_loc, bot_loc, dc_st_ub,
      u32_base + 1024, reinterpret_cast<__ubuf__ int32_t*>(base_ptr + 5120), u32_base + 9728, u32_base + 44544,
      lt_dc_st, lt_z_t, lt_i32[1024], lt_i32[5120], lt_i32[9728], lt_i32[44544], gt_sym
    );

    for (int32_t step = 1; step <= 2; step *= 2) {
      bool is_lv2 = (step == 2);
      uint32_t num_merges = is_lv2 ? 1u : 2u;
      int32_t K_arr[2] = {0, 0}, D_arr[2] = {0, 0}, triv_arr[2] = {0, 0};
      float rho_arr[2] = {0.0f, 0.0f};
      for (uint32_t m_idx = 0; m_idx < num_merges; ++m_idx) {
        DeflateBlock64SingleMerge(m_idx, step, is_lv2, lb, e_cuts, d_loc, top_loc, bot_loc, ths_loc, bhs_loc, ad_loc, az_loc, dd_loc, orig_loc, acol_loc, K_arr, D_arr, triv_arr, rho_arr, lt_z_t, lt_ztmp2, u_def64, u_s1_32, b_def64, b_s1_32);
      }
      SolveBlock64RootsAndSync(step, is_lv2, num_merges, p, plo_b, bp, subP_b, lb, K_arr, rho_arr, ad_loc, az_loc, orig_loc, tau_loc, dc_ws_ub, dc_st_ub, dc_z_ub, lt_dc_ws, lt_dc_st, lt_dc_z, lt_absw, gt_sym);
      if (!is_lv2) {
        for (uint32_t m_idx = 0; m_idx < 2u; ++m_idx) {
          UpdateBlock64Step1SingleMerge(
            m_idx, lb, K_arr, D_arr, triv_arr, rho_arr, d_loc, top_loc, bot_loc, ths_loc, bhs_loc, ad_loc, az_loc, dd_loc, tau_loc,
            orig_loc, acol_loc, dc_ws_ub, dc_z_ub, lt_t64, lt_z_t, lt_ztmp, lt_ztmp2, lt_dc_ws, lt_dc_z, lt_dc_st,
            lt_off_bc64, lt_off_rep8, lt_u16, lt_dmsk_u16, lt_eye64_u16, u_s1_32, b_bc64_s1, b_bc64_row0, b_def64, b_s1_32, rk_s1_32
          );
        }
      } else {
        UpdateBlock64Step2Final(
          L_b, my_blk, bp, subP_b, blk_slot, K_arr, D_arr, triv_arr, rho_arr, d_loc, top_loc, bot_loc, ths_loc, bhs_loc,
          ad_loc, az_loc, dd_loc, tau_loc, orig_loc, acol_loc, final_oc, base_ptr, dc_ws_ub, dc_z_ub,
          lt_t64, lt_z_t, lt_ztmp, lt_ztmp2, lt_dc_ws, lt_dc_z, lt_dc_st, lt_off_bc64, lt_off_rep8,
          lt_u16, lt_dmsk_u16, lt_eye64_u16, gt_sym, u_def64, b_bc8_s1, b_bc64_s1, b_bc64_row0, b_def64, rk_mul64
        );
      }
    }
  }
