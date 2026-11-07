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

  __aicore__ __attribute__((always_inline)) inline void SyncAndMergeLargeP3Summaries(
    uint32_t P, int32_t step, bool final_level, uint32_t num_merges,
    const __ubuf__ int32_t* bounds, const __ubuf__ int32_t* K_arr, const __ubuf__ int32_t* D_arr, const __ubuf__ int32_t* triv_arr,
    const __ubuf__ float* ad, const __ubuf__ float* ths, const __ubuf__ float* bhs, const __ubuf__ float* tau_a,
    const __ubuf__ int32_t* acol_dc, __ubuf__ int32_t* orig_ord, __ubuf__ float* dd, __ubuf__ float* d, __ubuf__ float* top, __ubuf__ float* bot,
    __ubuf__ float* z_ub, const AscendC::LocalTensor<float>& lt_z, const AscendC::GlobalTensor<float>& gt_bnd
  ) {
    if (!final_level) { AscendC::DataCopy(lt_z, gt_bnd, static_cast<int32_t>(P * 64u)); AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0); }
    bool bnd_synced = final_level;
    for (uint32_t m = 0; m < num_merges; ++m) {
      int32_t b_idx = static_cast<int32_t>(m) * 2 * step, sm = bounds[b_idx], Lm = bounds[b_idx + 2 * step] - sm, Km = K_arr[m], Dm = D_arr[m];
      if (Km > 0) {
        uint32_t plo_m = (m * P) / num_merges, sub_P_m = ((m + 1u) * P) / num_merges - plo_m, u_cap_m = (static_cast<uint32_t>(Km) + sub_P_m - 1u) / sub_P_m;
        for (int32_t di = Dm - 1; di >= 0; --di) dd[sm + Km + di] = dd[sm + di];
        for (int32_t i = 0; i < Km; ++i) dd[sm + i] = ad[sm + orig_ord[sm + i]] + tau_a[sm + i];
        MergeTwoSortedF32(Km, Lm, &orig_ord[sm], &dd[sm]);
        if (final_level) {
          for (int32_t c = 0; c < Lm; ++c) d[sm + c] = dd[sm + orig_ord[sm + c]];
          continue;
        }
        if (!bnd_synced) { AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0); bnd_synced = true; }
        for (int32_t c = 0; c < Lm; ++c) {
          int32_t oc = orig_ord[sm + c];
          d[sm + c] = dd[sm + oc];
          if (oc < Km) {
            uint32_t base_off = (plo_m + static_cast<uint32_t>(oc) / u_cap_m) * 64u, ci = static_cast<uint32_t>(oc) % u_cap_m;
            top[sm + c] = z_ub[base_off + ci]; bot[sm + c] = z_ub[base_off + 32u + ci];
          } else {
            int32_t dc = acol_dc[sm + Lm - 1 - (oc - Km)];
            top[sm + c] = ths[sm + dc]; bot[sm + c] = bhs[sm + dc];
          }
        }
      } else if (!triv_arr[m]) {
        for (int32_t c = 0; c < Lm; ++c) {
          d[sm + c] = dd[sm + c];
          if (!final_level) { int32_t dc = acol_dc[sm + Lm - 1 - c]; top[sm + c] = ths[sm + dc]; bot[sm + c] = bhs[sm + dc]; }
        }
      }
    }
    if (!bnd_synced) AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void InitLargeP4BroadcastTables(
    uint32_t act_cols_0, uint32_t act_cols_1, __ubuf__ uint32_t* off_rep8_p4, __ubuf__ uint32_t* off_rep8_p4_hi,
    const AscendC::LocalTensor<int32_t>& lt_off_rep8_p4_i, const AscendC::LocalTensor<int32_t>& lt_off_rep8_p4_hi_i
  ) {
    int32_t mc8_0 = static_cast<int32_t>(act_cols_0 * 8u);
    for (uint32_t c = 0; c < act_cols_0; ++c) {
      for (uint32_t ln = 0; ln < 8u; ++ln) off_rep8_p4[c * 8u + ln] = c * 64u;
    }
    if (act_cols_1 > 0u) {
      for (uint32_t c = 0; c < act_cols_1; ++c) {
        for (uint32_t ln = 0; ln < 8u; ++ln) off_rep8_p4_hi[c * 8u + ln] = c * 64u;
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (int32_t dk = 1; dk < 8; ++dk) AscendC::Adds(lt_off_rep8_p4_i[dk * mc8_0], lt_off_rep8_p4_i[0], dk * 4, mc8_0);
    if (act_cols_1 > 0u) {
      int32_t mc8_1 = static_cast<int32_t>(act_cols_1 * 8u);
      for (int32_t dk = 1; dk < 8; ++dk) AscendC::Adds(lt_off_rep8_p4_hi_i[dk * mc8_1], lt_off_rep8_p4_hi_i[0], dk * 4, mc8_1);
    }
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline void GatherLargeP4ActivePanel(
    int32_t sm, int32_t kb, int32_t cols_k, uint32_t act_c0, uint32_t act_cols, uint32_t act_cols_0, uint32_t act_cols_1, uint32_t stride,
    const __ubuf__ int32_t* acol_dc, __ubuf__ float* stage_ub,
    const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_wscr,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_u32,
    const AscendC::LocalTensor<uint32_t>& lt_off_rep8_p4, const AscendC::LocalTensor<uint32_t>& lt_off_rep8_p4_hi
  ) {
    __ubuf__ uint32_t* off_k16 = reinterpret_cast<__ubuf__ uint32_t*>(stage_ub + 512);
    AscendC::LocalTensor<uint32_t> lt_off_k16 = lt_u32[13824 + 512];
    for (int32_t dk = 0; dk < cols_k; ++dk) off_k16[dk] = static_cast<uint32_t>(sm + acol_dc[sm + kb + dk]) * 4u;
    for (int32_t dk = cols_k; dk < 16; ++dk) off_k16[dk] = 0u;
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (uint32_t c = 0; c < act_cols; ++c) AscendC::Gather(lt_stage[c * 16u], lt_mat[(act_c0 + c) * stride], lt_off_k16, 0u, 16u);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t h1_0 = (cols_k <= 8) ? (static_cast<uint32_t>(cols_k) * act_cols_0) : (8u * act_cols_0);
    if (h1_0 > 0u) AscendC::Gather(lt_wscr[0], lt_stage[0], lt_off_rep8_p4, 0u, h1_0 * 8u);
    if (cols_k > 8) AscendC::Gather(lt_wscr[1024], lt_stage[8], lt_off_rep8_p4, 0u, static_cast<uint32_t>(cols_k - 8) * act_cols_0 * 8u);
    if (act_cols_1 > 0u) {
      uint32_t h1_1 = (cols_k <= 8) ? (static_cast<uint32_t>(cols_k) * act_cols_1) : (8u * act_cols_1);
      if (h1_1 > 0u) AscendC::Gather(lt_z[10240], lt_stage[256], lt_off_rep8_p4_hi, 0u, h1_1 * 8u);
      if (cols_k > 8) AscendC::Gather(lt_z[11264], lt_stage[264], lt_off_rep8_p4_hi, 0u, static_cast<uint32_t>(cols_k - 8) * act_cols_1 * 8u);
    }
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline void AccumLargeP4SubTile16(
    int32_t cols_k, uint32_t sub_cols, uint8_t rz_blk_u8,
    const AscendC::LocalTensor<float>& lt_z_ib, const AscendC::LocalTensor<float>& lt_w_sub,
    const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_z, const AscendC::BinaryRepeatParams& p4_mla
  ) {
    uint8_t mc_u8 = static_cast<uint8_t>(sub_cols);
    AscendC::BinaryRepeatParams p4_mla_loc(1, 1, 0, 8, 0, 1), p4_fold_z(1, 1, 1, rz_blk_u8, rz_blk_u8, 8), p4_fold_loc(1, 1, 1, 8, 8, 8);
    if (cols_k >= 4) {
      AscendC::MulAddDst<float, float, true>(lt_z_ib, lt_stage[1024u], lt_w_sub[0], 64ULL, mc_u8, p4_mla);
      AscendC::Mul<float, false>(lt_z[13312u], lt_stage[1024u + 64u], lt_w_sub[sub_cols * 8u], 64ULL, mc_u8, p4_mla_loc);
      AscendC::Mul<float, false>(lt_z[13312u + sub_cols * 64u], lt_stage[1024u + 128u], lt_w_sub[sub_cols * 16u], 64ULL, mc_u8, p4_mla_loc);
      AscendC::Mul<float, false>(lt_z[13312u + sub_cols * 128u], lt_stage[1024u + 192u], lt_w_sub[sub_cols * 24u], 64ULL, mc_u8, p4_mla_loc);
      AscendC::PipeBarrier<PIPE_V>();
      int32_t dk = 4;
      for (; dk + 3 < cols_k; dk += 4) {
        uint32_t vb0 = (dk < 8) ? (static_cast<uint32_t>(dk) * sub_cols * 8u) : (1024u + static_cast<uint32_t>(dk - 8) * sub_cols * 8u);
        uint32_t vb1 = ((dk + 1) < 8) ? (static_cast<uint32_t>(dk + 1) * sub_cols * 8u) : (1024u + static_cast<uint32_t>(dk + 1 - 8) * sub_cols * 8u);
        uint32_t vb2 = ((dk + 2) < 8) ? (static_cast<uint32_t>(dk + 2) * sub_cols * 8u) : (1024u + static_cast<uint32_t>(dk + 2 - 8) * sub_cols * 8u);
        uint32_t vb3 = ((dk + 3) < 8) ? (static_cast<uint32_t>(dk + 3) * sub_cols * 8u) : (1024u + static_cast<uint32_t>(dk + 3 - 8) * sub_cols * 8u);
        AscendC::MulAddDst<float, float, false>(lt_z_ib, lt_stage[1024u + static_cast<uint32_t>(dk) * 64u], lt_w_sub[vb0], 64ULL, mc_u8, p4_mla);
        AscendC::MulAddDst<float, float, false>(lt_z[13312u], lt_stage[1024u + static_cast<uint32_t>(dk + 1) * 64u], lt_w_sub[vb1], 64ULL, mc_u8, p4_mla_loc);
        AscendC::MulAddDst<float, float, false>(lt_z[13312u + sub_cols * 64u], lt_stage[1024u + static_cast<uint32_t>(dk + 2) * 64u], lt_w_sub[vb2], 64ULL, mc_u8, p4_mla_loc);
        AscendC::MulAddDst<float, float, false>(lt_z[13312u + sub_cols * 128u], lt_stage[1024u + static_cast<uint32_t>(dk + 3) * 64u], lt_w_sub[vb3], 64ULL, mc_u8, p4_mla_loc);
        AscendC::PipeBarrier<PIPE_V>();
      }
      for (; dk < cols_k; ++dk) {
        uint32_t vb_off = (dk < 8) ? (static_cast<uint32_t>(dk) * sub_cols * 8u) : (1024u + static_cast<uint32_t>(dk - 8) * sub_cols * 8u);
        AscendC::MulAddDst<float, float, false>(lt_z_ib, lt_stage[1024u + static_cast<uint32_t>(dk) * 64u], lt_w_sub[vb_off], 64ULL, mc_u8, p4_mla);
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::Add<float, false>(lt_z[13312u], lt_z[13312u], lt_z[13312u + sub_cols * 64u], 64ULL, mc_u8, p4_fold_loc);
      AscendC::Add<float, false>(lt_z_ib, lt_z_ib, lt_z[13312u + sub_cols * 128u], 64ULL, mc_u8, p4_fold_z);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add<float, false>(lt_z_ib, lt_z_ib, lt_z[13312u], 64ULL, mc_u8, p4_fold_z);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::ResetMask();
    } else {
      AscendC::MulAddDst<float, float, true>(lt_z_ib, lt_stage[1024u], lt_w_sub[0], 64ULL, mc_u8, p4_mla);
      AscendC::PipeBarrier<PIPE_V>();
      for (int32_t dk = 1; dk < cols_k; ++dk) {
        uint32_t vb_off = (dk < 8) ? (static_cast<uint32_t>(dk) * sub_cols * 8u) : (1024u + static_cast<uint32_t>(dk - 8) * sub_cols * 8u);
        AscendC::MulAddDst<float, float, false>(lt_z_ib, lt_stage[1024u + static_cast<uint32_t>(dk) * 64u], lt_w_sub[vb_off], 64ULL, mc_u8, p4_mla);
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::ResetMask();
    }
  }

  __aicore__ __attribute__((always_inline)) inline void EvalLargeP4Panel16(
    int32_t Km, int32_t cols_k, int32_t n_ib, uint32_t panel_base, uint32_t act_cols_0, uint32_t act_cols_1, uint32_t num_sub, uint32_t row_z_str, uint8_t rz_blk_u8,
    const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<uint32_t>& lt_off_tr64x16, const AscendC::GlobalTensor<float>& gt_ut_m, const AscendC::BinaryRepeatParams& p4_mla
  ) {
    for (int32_t t = 0; t < n_ib; ++t) {
      int32_t ib = t * 64, rows_r = (ib + 64 <= Km) ? 64 : (Km - ib);
      event_t cur_ev = (t & 1) ? EVENT_ID1 : EVENT_ID0;
      AscendC::LocalTensor<float> lt_dma = lt_wscr[2048u + static_cast<uint32_t>(t & 1) * 1024u];
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(cur_ev);
      if (rows_r < 64) { AscendC::Duplicate(lt_dma[static_cast<uint32_t>(rows_r) * 16u], 0.0f, (64 - rows_r) * 16); AscendC::PipeBarrier<PIPE_V>(); }
      AscendC::Gather(lt_stage[1024], lt_dma, lt_off_tr64x16, 0u, 1024u);
      if (t + 2 < n_ib) {
        int32_t ib_n = (t + 2) * 64, rows_n = (ib_n + 64 <= Km) ? 64 : (Km - ib_n);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(cur_ev); AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(cur_ev);
        AscendC::DataCopy(lt_dma, gt_ut_m[panel_base + static_cast<uint32_t>(ib_n) * 16u], rows_n * 16);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(cur_ev);
      }
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t sub = 0u; sub < num_sub; ++sub) {
        AccumLargeP4SubTile16(cols_k, (sub == 0u) ? act_cols_0 : act_cols_1, rz_blk_u8, lt_z[(sub * 16u) * row_z_str + static_cast<uint32_t>(ib)], (sub == 0u) ? lt_wscr[0] : lt_z[10240], lt_stage, lt_z, p4_mla);
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    if (n_ib >= 2) { AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1); AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1); }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void EvalLargeP4Gemm(
    int32_t sm, int32_t Km, uint32_t act_c0, uint32_t act_cols, uint32_t act_cols_0, uint32_t act_cols_1, uint32_t stride, uint32_t ut_stride, uint32_t row_z_str,
    const __ubuf__ int32_t* acol_dc, __ubuf__ float* stage_ub,
    const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_wscr,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_u32, const AscendC::LocalTensor<uint32_t>& lt_off_tr64x16,
    const AscendC::LocalTensor<uint32_t>& lt_off_rep8_p4, const AscendC::LocalTensor<uint32_t>& lt_off_rep8_p4_hi, const AscendC::GlobalTensor<float>& gt_ut_m
  ) {
    AscendC::Duplicate(lt_z, 0.0f, static_cast<int32_t>(act_cols * row_z_str));
    AscendC::PipeBarrier<PIPE_V>();
    uint8_t rz_blk_u8 = static_cast<uint8_t>(row_z_str >> 3u);
    AscendC::BinaryRepeatParams p4_mla(1, 1, 0, rz_blk_u8, 0, 1);
    uint32_t num_sub = (act_cols_1 > 0u) ? 2u : 1u, ut_row_base = static_cast<uint32_t>(sm);

    for (int32_t kb = 0; kb < Km; kb += 16) {
      int32_t cols_k = (kb + 16 <= Km) ? 16 : (Km - kb), n_ib = (Km + 63) >> 6;
      uint32_t w_kb = (cols_k <= 8) ? 8u : 16u, panel_base = ut_row_base * ut_stride + static_cast<uint32_t>(kb) * static_cast<uint32_t>(Km);
      if (w_kb == 16u) {
        AscendC::DataCopy(lt_wscr[2048], gt_ut_m[panel_base], ((64 <= Km) ? 64 : Km) * 16);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (n_ib >= 2) {
          AscendC::DataCopy(lt_wscr[3072], gt_ut_m[panel_base + 1024u], ((128 <= Km) ? 64 : (Km - 64)) * 16);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        }
      }
      GatherLargeP4ActivePanel(sm, kb, cols_k, act_c0, act_cols, act_cols_0, act_cols_1, stride, acol_dc, stage_ub, lt_stage, lt_mat, lt_wscr, lt_z, lt_u32, lt_off_rep8_p4, lt_off_rep8_p4_hi);
      if (w_kb == 16u) {
        EvalLargeP4Panel16(Km, cols_k, n_ib, panel_base, act_cols_0, act_cols_1, num_sub, row_z_str, rz_blk_u8, lt_stage, lt_wscr, lt_z, lt_off_tr64x16, gt_ut_m, p4_mla);
      } else {
        AscendC::DataCopyParams cp_p4_8(64, 1, 0, 1);
        for (int32_t ib = 0; ib < Km; ib += 64) {
          cp_p4_8.blockCount = static_cast<uint16_t>((ib + 64 <= Km) ? 64 : (Km - ib));
          AscendC::Duplicate(lt_wscr[2048], 0.0f, 1024);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
          AscendC::DataCopy(lt_wscr[2048], gt_ut_m[panel_base + static_cast<uint32_t>(ib) * 8u], cp_p4_8);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
          AscendC::Gather(lt_stage[1024], lt_wscr[2048], lt_off_tr64x16, 0u, 1024u);
          AscendC::PipeBarrier<PIPE_V>();
          for (uint32_t sub = 0u; sub < num_sub; ++sub) {
            AccumLargeP4SubTile16(cols_k, (sub == 0u) ? act_cols_0 : act_cols_1, rz_blk_u8, lt_z[(sub * 16u) * row_z_str + static_cast<uint32_t>(ib)], (sub == 0u) ? lt_wscr[0] : lt_z[10240], lt_stage, lt_z, p4_mla);
          }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      }
    }
  }

  __aicore__ __attribute__((always_inline)) inline void ScatterLargeP4UnalignedCols(
    int32_t sm, int32_t Lm, int32_t Km, uint32_t sm_al, uint32_t sm_rem, uint32_t Lm_u, uint32_t act_c0, uint32_t act_cols, uint32_t stride, uint32_t stride_K64, uint32_t Dm_al, uint32_t row_z_str,
    const __ubuf__ int32_t* orig_ord, __ubuf__ uint32_t* off_pfull,
    const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_off_pfull
  ) {
    uint32_t em_al = (static_cast<uint32_t>(sm + Lm) + 7u) & ~7u, span_al = em_al - sm_al, h_off = stride_K64 + Dm_al;
    for (uint32_t idx = 0; idx < sm_rem; ++idx) off_pfull[idx] = (h_off + idx) * 4u;
    for (uint32_t idx = 0; idx < Lm_u; ++idx) {
      int32_t oc = orig_ord[sm + static_cast<int32_t>(idx)];
      off_pfull[sm_rem + idx] = ((oc < Km) ? static_cast<uint32_t>(oc) : (stride_K64 + static_cast<uint32_t>(oc - Km))) * 4u;
    }
    for (uint32_t idx = sm_rem + Lm_u; idx < span_al; ++idx) off_pfull[idx] = (h_off + 8u + (sm_al + idx - (em_al - 8u))) * 4u;
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (uint32_t c = 0; c < act_cols; ++c) {
      AscendC::Adds(lt_z[c * row_z_str + h_off],      lt_mat[(act_c0 + c) * stride + sm_al],      0.0f, 8);
      AscendC::Adds(lt_z[c * row_z_str + h_off + 8u], lt_mat[(act_c0 + c) * stride + em_al - 8u], 0.0f, 8);
    }
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t c = 0; c < act_cols; ++c) AscendC::Gather(lt_mat[(act_c0 + c) * stride + sm_al], lt_z[c * row_z_str], lt_off_pfull, 0u, span_al);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void ScatterLargeP4MergedCols(
    int32_t sm, int32_t Lm, int32_t Km, int32_t Dm, uint32_t act_c0, uint32_t act_cols, uint32_t stride, uint32_t stride_K64, uint32_t Dm_al, bool unaligned_m, bool vec_pack, uint32_t row_z_str,
    const __ubuf__ int32_t* orig_ord, const __ubuf__ int32_t* acol_dc, __ubuf__ uint32_t* off_pfull, __ubuf__ float* wide_scr, __ubuf__ float* mat_ub, __ubuf__ float* z_ub,
    const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_off_pfull
  ) {
    if (vec_pack) {
      uint32_t sm_al = static_cast<uint32_t>(sm) & ~7u, sm_rem = static_cast<uint32_t>(sm) - sm_al, Lm_u = static_cast<uint32_t>(Lm);
      if (Dm > 0) {
        for (int32_t di = 0; di < Dm; ++di) off_pfull[di] = (sm_rem + static_cast<uint32_t>(acol_dc[sm + Lm - 1 - di])) * 4u;
        for (uint32_t di = static_cast<uint32_t>(Dm); di < Dm_al; ++di) off_pfull[di] = sm_rem * 4u;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        for (uint32_t c = 0; c < act_cols; ++c) AscendC::Gather(lt_z[c * row_z_str + stride_K64], lt_mat[(act_c0 + c) * stride + sm_al], lt_off_pfull, 0u, Dm_al);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      }
      if (!unaligned_m) {
        for (int32_t col_out = 0; col_out < Lm; ++col_out) {
          int32_t oc = orig_ord[sm + col_out];
          off_pfull[col_out] = ((oc < Km) ? static_cast<uint32_t>(oc) : (stride_K64 + static_cast<uint32_t>(oc - Km))) * 4u;
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        for (uint32_t c = 0; c < act_cols; ++c) AscendC::Gather(lt_mat[(act_c0 + c) * stride + static_cast<uint32_t>(sm)], lt_z[c * row_z_str], lt_off_pfull, 0u, Lm_u);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      } else {
        ScatterLargeP4UnalignedCols(sm, Lm, Km, sm_al, sm_rem, Lm_u, act_c0, act_cols, stride, stride_K64, Dm_al, row_z_str, orig_ord, off_pfull, lt_mat, lt_z, lt_off_pfull);
      }
    } else {
      for (uint32_t c = 0; c < act_cols; ++c) {
        __ubuf__ float* row_m = mat_ub + (act_c0 + c) * stride + static_cast<uint32_t>(sm), *row_a = z_ub + c * row_z_str;
        for (int32_t di = 0; di < Dm; ++di) wide_scr[di] = row_m[acol_dc[sm + Lm - 1 - di]];
        for (int32_t col_out = 0; col_out < Lm; ++col_out) {
          int32_t oc = orig_ord[sm + col_out];
          row_m[col_out] = (oc < Km) ? row_a[oc] : wide_scr[oc - Km];
        }
      }
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    }
  }

  __aicore__ __attribute__((always_inline)) inline void ScatterLargeP4PureDeflatedCols(
    int32_t sm, int32_t Lm, uint32_t act_c0, uint32_t act_cols, uint32_t stride,
    const __ubuf__ int32_t* acol_dc, __ubuf__ uint32_t* off_pfull, __ubuf__ float* wide_scr, __ubuf__ float* mat_ub,
    const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_off_pfull
  ) {
    uint32_t sm_al = static_cast<uint32_t>(sm) & ~7u, em_al = (static_cast<uint32_t>(sm + Lm) + 7u) & ~7u, span_al = em_al - sm_al, sm_rem = static_cast<uint32_t>(sm) - sm_al, Lm_u = static_cast<uint32_t>(Lm);
    if (act_cols * span_al <= 16384u) {
      for (uint32_t idx = 0; idx < sm_rem; ++idx) off_pfull[idx] = idx * 4u;
      for (uint32_t idx = 0; idx < Lm_u; ++idx) off_pfull[sm_rem + idx] = (sm_rem + static_cast<uint32_t>(acol_dc[sm + Lm - 1 - static_cast<int32_t>(idx)])) * 4u;
      for (uint32_t idx = sm_rem + Lm_u; idx < span_al; ++idx) off_pfull[idx] = idx * 4u;
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      for (uint32_t c = 0; c < act_cols; ++c) AscendC::Adds(lt_z[c * span_al], lt_mat[(act_c0 + c) * stride + sm_al], 0.0f, static_cast<int32_t>(span_al));
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t c = 0; c < act_cols; ++c) AscendC::Gather(lt_mat[(act_c0 + c) * stride + sm_al], lt_z[c * span_al], lt_off_pfull, 0u, span_al);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    } else {
      for (uint32_t c = 0; c < act_cols; ++c) {
        __ubuf__ float* row_m = mat_ub + (act_c0 + c) * stride + static_cast<uint32_t>(sm);
        for (int32_t di = 0; di < Lm; ++di) wide_scr[di] = row_m[acol_dc[sm + Lm - 1 - di]];
        for (int32_t di = 0; di < Lm; ++di) row_m[di] = wide_scr[di];
      }
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    }
  }

  __aicore__ __attribute__((always_inline)) inline void UpdateLargeP4AllMerges(
    int32_t step, uint32_t num_merges, uint32_t col_start, uint32_t col_end, uint32_t my_cols, uint32_t stride, uint32_t ut_stride,
    const __ubuf__ int32_t* bounds, const __ubuf__ int32_t* K_arr, const __ubuf__ int32_t* D_arr, const __ubuf__ int32_t* triv_arr,
    const __ubuf__ int32_t* orig_ord, const __ubuf__ int32_t* acol_dc, __ubuf__ uint32_t* off_pfull, __ubuf__ float* wide_scr, __ubuf__ float* stage_ub, __ubuf__ float* mat_ub, __ubuf__ float* z_ub,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat,
    const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_u32,
    const AscendC::LocalTensor<uint32_t>& lt_off_tr64x16, const AscendC::LocalTensor<uint32_t>& lt_off_pfull, const AscendC::GlobalTensor<float>& gt_ut_m
  ) {
    if (my_cols == 0u) return;
    __ubuf__ uint32_t* off_rep8_p4 = reinterpret_cast<__ubuf__ uint32_t*>(wide_scr + 4096), *off_rep8_p4_hi = reinterpret_cast<__ubuf__ uint32_t*>(z_ub + 12288);
    AscendC::LocalTensor<uint32_t> lt_off_rep8_p4 = lt_u32[14 * 512 + 4096], lt_off_rep8_p4_hi = lt_u32[32256 + 12288];
    AscendC::LocalTensor<int32_t> lt_off_rep8_p4_i = ub_all.ReinterpretCast<int32_t>()[14 * 512 + 4096], lt_off_rep8_p4_hi_i = ub_all.ReinterpretCast<int32_t>()[32256 + 12288];

    for (uint32_t m = 0; m < num_merges; ++m) {
      int32_t b_idx = static_cast<int32_t>(m) * 2 * step, sm = bounds[b_idx], Lm = bounds[b_idx + 2 * step] - sm, Km = K_arr[m], Dm = D_arr[m];
      uint32_t r_act_lo = (static_cast<uint32_t>(sm) > col_start) ? static_cast<uint32_t>(sm) : col_start;
      uint32_t r_act_hi = (static_cast<uint32_t>(sm + Lm) < col_end) ? static_cast<uint32_t>(sm + Lm) : col_end;
      if (r_act_lo >= r_act_hi) continue;
      uint32_t act_c0 = r_act_lo - col_start, act_cols = r_act_hi - r_act_lo;

      if (Km > 0) {
        uint32_t act_cols_0 = (act_cols <= 16u) ? act_cols : 16u, act_cols_1 = (act_cols > 16u) ? (act_cols - 16u) : 0u;
        InitLargeP4BroadcastTables(act_cols_0, act_cols_1, off_rep8_p4, off_rep8_p4_hi, lt_off_rep8_p4_i, lt_off_rep8_p4_hi_i);
        uint32_t stride_K64 = (static_cast<uint32_t>(Km) + 63u) & ~63u, Dm_al = (static_cast<uint32_t>(Dm) + 7u) & ~7u;
        bool unaligned_m = (((sm & 7) != 0) || ((Lm & 7) != 0));
        uint32_t pad_sl = unaligned_m ? 16u : 0u;
        bool vec_pack = (act_cols * (stride_K64 + Dm_al + pad_sl) <= 10240u);
        uint32_t row_z_str = vec_pack ? (stride_K64 + Dm_al + pad_sl) : stride_K64;
        EvalLargeP4Gemm(sm, Km, act_c0, act_cols, act_cols_0, act_cols_1, stride, ut_stride, row_z_str, acol_dc, stage_ub, lt_stage, lt_mat, lt_wscr, lt_z, lt_u32, lt_off_tr64x16, lt_off_rep8_p4, lt_off_rep8_p4_hi, gt_ut_m);
        ScatterLargeP4MergedCols(sm, Lm, Km, Dm, act_c0, act_cols, stride, stride_K64, Dm_al, unaligned_m, vec_pack, row_z_str, orig_ord, acol_dc, off_pfull, wide_scr, mat_ub, z_ub, lt_mat, lt_z, lt_off_pfull);
      } else if (!triv_arr[m]) {
        ScatterLargeP4PureDeflatedCols(sm, Lm, act_c0, act_cols, stride, acol_dc, off_pfull, wide_scr, mat_ub, lt_mat, lt_z, lt_off_pfull);
      }
    }
  }

  __aicore__ __attribute__((always_inline)) inline void SortAndTransposeLargeStedcToQT(
    uint32_t col_start, uint32_t col_end, uint32_t my_cols, uint32_t stride, float scale_pow2,
    __ubuf__ float* d, __ubuf__ uint32_t* off_pfull, __ubuf__ uint32_t* off_bc16_s4, __ubuf__ float* mat_ub,
    const AscendC::LocalTensor<float>& lt_d, const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<int32_t>& lt_off_pf_i, const AscendC::LocalTensor<uint32_t>& lt_off_pfull, const AscendC::GlobalTensor<float>& gt_sym
  ) {
    AscendC::Muls(lt_d, lt_d, scale_pow2, static_cast<int32_t>(stride));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

    bool already_sorted = true;
    for (uint32_t i = 0; i + 1u < n_; ++i) if (d[i + 1u] < d[i]) { already_sorted = false; break; }
    if (!already_sorted) {
      SortEigenPairsInplace(n_, d, static_cast<uint32_t*>(nullptr), mat_ub, my_cols, stride);
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    } else if (my_cols > 0u) {
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    }
    if (my_cols > 0u) {
      AscendC::DataCopyParams cp_z2gm(static_cast<uint16_t>(my_cols), static_cast<uint16_t>(n_ / 8u), static_cast<uint16_t>((stride - n_) / 8u), 0);
      AscendC::DataCopy(gt_sym[col_start * n_], lt_mat, cp_z2gm);
    }
    SafeSyncAll();

    if (my_cols > 0u) {
      uint32_t cs_al = col_start & ~7u, ce_al = (col_end + 7u) & ~7u, w_al = ce_al - cs_al, cs_rem = col_start - cs_al;
      AscendC::DataCopyParams cp_gm2z(static_cast<uint16_t>(n_), static_cast<uint16_t>(w_al / 8u), static_cast<uint16_t>((n_ - w_al) / 8u), 0);
      AscendC::DataCopy(lt_z, gt_sym[cs_al], cp_gm2z);
      AscendC::Duplicate(lt_mat, 0.0f, static_cast<int32_t>(my_cols * stride));
      for (uint32_t r = 0; r < 8u; ++r) off_pfull[r] = (r * w_al + cs_rem) * 4u;
      for (uint32_t idx = 0; idx < 128u; ++idx) off_bc16_s4[idx] = (idx >> 3u) * 4u;
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      int32_t step4 = static_cast<int32_t>(w_al * 4u);
      for (int32_t rep = 8; rep < static_cast<int32_t>(n_); rep <<= 1) {
        AscendC::Adds(lt_off_pf_i[rep], lt_off_pf_i[0], rep * step4, rep);
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t c = 0; c < my_cols; ++c) {
        if (c > 0u) { AscendC::Adds(lt_off_pf_i, lt_off_pf_i, 4, static_cast<int32_t>(n_)); AscendC::PipeBarrier<PIPE_V>(); }
        AscendC::Gather(lt_mat[c * stride], lt_z, lt_off_pfull, 0u, n_);
      }
      AscendC::PipeBarrier<PIPE_V>();
    }
    SafeSyncAll();
  }

  __aicore__ __attribute__((always_inline)) inline void ApplyQuadReflectorsLargeTail64(
    uint32_t my_cols, uint32_t stride, uint32_t n_q64, const uint32_t* t64_uis, const __ubuf__ float* inv_h_keep, const float* udot64,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_off_bc16_s4,
    const AscendC::UnaryRepeatParams& s4_u16, const AscendC::BinaryRepeatParams& s4_mul_in8,
    const AscendC::BinaryRepeatParams& s4_mla_up, const AscendC::BinaryRepeatParams& s4_mul_up8, const AscendC::BinaryRepeatParams& s4_add_mat_from8
  ) {
    uint32_t ia64 = 0u;
    for (uint32_t qi = 0u; qi < n_q64; ++qi, ia64 += 4u) {
      uint32_t ui0 = t64_uis[ia64 + 0u], ui1 = t64_uis[ia64 + 1u], ui2 = t64_uis[ia64 + 2u], ui3 = t64_uis[ia64 + 3u], b6 = qi * 6u;
      AscendC::LocalTensor<float> u0 = lt_z[8192u + ui0 * 64u], u1 = lt_z[8192u + ui1 * 64u], u2 = lt_z[8192u + ui2 * 64u], u3 = lt_z[8192u + ui3 * 64u];

      for (uint32_t c_base = 0; c_base < my_cols; c_base += 16u) {
        uint32_t c_lim = (c_base + 16u < my_cols) ? (c_base + 16u) : my_cols;
        int32_t c_cnt = static_cast<int32_t>(c_lim - c_base);
        uint8_t c_cnt_u8 = static_cast<uint8_t>(c_cnt);

        AscendC::Mul<float, true>(lt_z[0],       lt_mat[c_base * stride], u0, 64ULL, c_cnt_u8, s4_mul_in8);
        AscendC::Mul<float, false>(ub_all[512],  lt_mat[c_base * stride], u1, 64ULL, c_cnt_u8, s4_mul_in8);
        AscendC::Mul<float, false>(lt_z[1024],   lt_mat[c_base * stride], u2, 64ULL, c_cnt_u8, s4_mul_in8);
        AscendC::Mul<float, false>(ub_all[1536], lt_mat[c_base * stride], u3, 64ULL, c_cnt_u8, s4_mul_in8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float, false>(lt_stage[0],  lt_z[0],      64, c_cnt, 1, 1, 8);
        AscendC::WholeReduceSum<float, false>(lt_stage[16], ub_all[512],  64, c_cnt, 1, 1, 8);
        AscendC::WholeReduceSum<float, false>(lt_stage[32], lt_z[1024],   64, c_cnt, 1, 1, 8);
        AscendC::WholeReduceSum<float, false>(lt_stage[48], ub_all[1536], 64, c_cnt, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();

        SolveQuadReflectorTriangular(
          16ULL, 16u, 128u, static_cast<uint32_t>(c_cnt) * 8u, inv_h_keep[ui0], inv_h_keep[ui1], inv_h_keep[ui2], inv_h_keep[ui3],
          udot64[b6 + 0u], udot64[b6 + 1u], udot64[b6 + 2u], udot64[b6 + 3u], udot64[b6 + 4u], udot64[b6 + 5u],
          lt_stage[0], lt_stage[512], lt_off_bc16_s4, s4_u16
        );
        AscendC::MulAddDst<float, float, true>(lt_mat[c_base * stride], u0, lt_stage[512], 64ULL, c_cnt_u8, s4_mla_up);
        AscendC::Mul<float, false>(lt_z[0], u1, lt_stage[640], 64ULL, c_cnt_u8, s4_mul_up8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst<float, float, false>(lt_mat[c_base * stride], u2, lt_stage[768], 64ULL, c_cnt_u8, s4_mla_up);
        AscendC::MulAddDst<float, float, false>(lt_z[0], u3, lt_stage[896], 64ULL, c_cnt_u8, s4_mul_up8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(lt_mat[c_base * stride], lt_mat[c_base * stride], lt_z[0], 64ULL, c_cnt_u8, s4_add_mat_from8);
        AscendC::ResetMask();
        AscendC::PipeBarrier<PIPE_V>();
      }
    }
  }
