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

  __aicore__ __attribute__((always_inline)) inline void UpdateMedStep2Final(
    uint32_t col_start, uint32_t my_cols, uint8_t mc_u8, float scale_pow2, const int32_t* bounds, const int32_t* K_arr, const int32_t* D_arr, const int32_t* triv_arr, const float* rho_arr,
    const int32_t* s1_triv, const uint32_t* s1_src, int32_t n_giv, const int32_t* giv_cp, const int32_t* giv_pk, const float* giv_cg, const float* giv_sg,
    float* d, const float* ad, const float* az, float* dd, const float* tau_a, const int32_t* orig_ord, const int32_t* acol_dc, int32_t* final_oc, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_z_ub,
    const AscendC::LocalTensor<float>& lt_t64, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2, const AscendC::LocalTensor<float>& lt_z_final, const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<float>& lt_dc_st,
    const AscendC::LocalTensor<uint32_t>& lt_off_bc64, const AscendC::LocalTensor<uint32_t>& lt_off_rep8, const AscendC::LocalTensor<uint32_t>& lt_off_rep8_4k, const AscendC::LocalTensor<uint16_t>& lt_u16, const AscendC::LocalTensor<uint16_t>& lt_dmsk_u16, const AscendC::LocalTensor<uint16_t>& lt_eye64_u16,
    const AscendC::UnaryRepeatParams& u_def64, const AscendC::UnaryRepeatParams& u_s1_32, const AscendC::BinaryRepeatParams& b_bc8_s1, const AscendC::BinaryRepeatParams& b_bc64_row0, const AscendC::BinaryRepeatParams& b_def64, const AscendC::BinaryRepeatParams& rk_mul64
  ) {
    UnpackAndRotateMedStep2(bounds, s1_triv, s1_src, n_giv, giv_cp, giv_pk, giv_cg, giv_sg, lt_t64, lt_z, lt_ztmp, lt_dc_st, u_def64, u_s1_32);
    int32_t n_i32 = static_cast<int32_t>(n_), Km = K_arr[0], Dm = D_arr[0];
    float rho_m = rho_arr[0];
    MergeAndSortMedStep2Eigenvals(n_i32, Km, Dm, triv_arr[0], scale_pow2, d, ad, dd, tau_a, orig_ord, acol_dc, final_oc);
    if (my_cols == 0u) return;
    if (triv_arr[0] || Km <= 1) {
      if (Km == 1 && !triv_arr[0]) { EIGH_PAD256() EIGH_PAD64() EIGH_PAD16() EIGH_PAD16() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4() }
      AscendC::Adds<float, true>(lt_t64, lt_t64, 0.0f, 64ULL, 1, u_def64);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t c = 0; c < my_cols; ++c) {
        int32_t oc = final_oc[col_start + c];
        int32_t ck = (triv_arr[0] || Km == 0) ? oc : ((oc < 1) ? acol_dc[0] : acol_dc[n_i32 - 1 - (oc - 1)]);
        AscendC::Adds<float, false>(lt_z_final[c * 64u], lt_z[static_cast<uint32_t>(ck) * 64u], 0.0f, 64ULL, 1, u_def64);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Adds<float, false>(lt_z[0], lt_z_final[0], 0.0f, 64ULL, mc_u8, u_def64);
      AscendC::ResetMask();
      AscendC::PipeBarrier<PIPE_V>();
    } else {
      EvalMedStep2SecularCols(col_start, my_cols, mc_u8, Km, rho_m, final_oc, acol_dc, orig_ord, ad, az, tau_a, dc_ws_ub, dc_z_ub, lt_t64, lt_z, lt_ztmp, lt_ztmp2, lt_z_final, lt_dc_ws, lt_dc_z, lt_off_bc64, lt_off_rep8, lt_off_rep8_4k, lt_u16, lt_dmsk_u16, lt_eye64_u16, u_def64, b_bc8_s1, b_bc64_row0, b_def64, rk_mul64);
    }
  }

  __aicore__ __attribute__((always_inline)) inline void ApplyQuadReflectorsMed64(
    uint32_t n_quads, uint32_t my_cols, uint8_t mc_u8, int32_t mc_i32,
    const uint32_t* act_uis, const float* inv_h_loc, const float* u_dots,
    const AscendC::LocalTensor<float>& lt_p0, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2,
    const AscendC::LocalTensor<float>& lt_hsave, const AscendC::LocalTensor<float>& lt_dc_z,
    const AscendC::LocalTensor<uint32_t>& lt_off_rep8, const AscendC::UnaryRepeatParams& u_def64,
    const AscendC::BinaryRepeatParams& b_def64, const AscendC::BinaryRepeatParams& mv_mul64, const AscendC::BinaryRepeatParams& rk_mul64
  ) {
    uint32_t idx_a = 0u;
    for (uint32_t qd = 0u; qd < n_quads; ++qd, idx_a += 4u) {
      uint32_t ui0 = act_uis[idx_a + 0u], ui1 = act_uis[idx_a + 1u], ui2 = act_uis[idx_a + 2u], ui3 = act_uis[idx_a + 3u], b6 = qd * 6u;
      AscendC::LocalTensor<float> lt_u0 = lt_hsave[ui0 * 64u], lt_u1 = lt_hsave[ui1 * 64u];
      AscendC::LocalTensor<float> lt_u2 = lt_hsave[ui2 * 64u], lt_u3 = lt_hsave[ui3 * 64u];

      AscendC::Mul<float, true>(lt_ztmp[0],     lt_z[0], lt_u0, 64ULL, mc_u8, mv_mul64);
      AscendC::Mul<float, false>(lt_ztmp[512],  lt_z[0], lt_u1, 64ULL, mc_u8, mv_mul64);
      AscendC::Mul<float, false>(lt_ztmp[1024], lt_z[0], lt_u2, 64ULL, mc_u8, mv_mul64);
      AscendC::Mul<float, false>(lt_ztmp[1536], lt_z[0], lt_u3, 64ULL, mc_u8, mv_mul64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::WholeReduceSum<float, false>(lt_p0[0],  lt_ztmp[0],    64, mc_i32, 1, 1, 8);
      AscendC::WholeReduceSum<float, false>(lt_p0[8],  lt_ztmp[512],  64, mc_i32, 1, 1, 8);
      AscendC::WholeReduceSum<float, false>(lt_p0[16], lt_ztmp[1024], 64, mc_i32, 1, 1, 8);
      AscendC::WholeReduceSum<float, false>(lt_p0[24], lt_ztmp[1536], 64, mc_i32, 1, 1, 8);
      AscendC::PipeBarrier<PIPE_V>();

      SolveQuadReflectorTriangular(
        8ULL, 8u, 64u, my_cols * 8u, inv_h_loc[ui0], inv_h_loc[ui1], inv_h_loc[ui2], inv_h_loc[ui3],
        u_dots[b6 + 0u], u_dots[b6 + 1u], u_dots[b6 + 2u], u_dots[b6 + 3u], u_dots[b6 + 4u], u_dots[b6 + 5u],
        lt_p0, lt_ztmp2, lt_off_rep8, u_def64
      );
      if constexpr (n_ == 64u) {
        AscendC::MulAddDst<float, float, true>(lt_z[0],  lt_u0, lt_ztmp2[0],   64ULL, mc_u8, rk_mul64);
        AscendC::Mul<float, false>(lt_dc_z[0],           lt_u1, lt_ztmp2[64],  64ULL, mc_u8, rk_mul64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst<float, float, false>(lt_z[0],    lt_u2, lt_ztmp2[128], 64ULL, mc_u8, rk_mul64);
        AscendC::MulAddDst<float, float, false>(lt_dc_z[0], lt_u3, lt_ztmp2[192], 64ULL, mc_u8, rk_mul64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(lt_z[0], lt_z[0], lt_dc_z[0], 64ULL, mc_u8, b_def64);
      } else {
        AscendC::MulAddDst<float, float, true>(lt_z[0], lt_u0, lt_ztmp2[0],   64ULL, mc_u8, rk_mul64);
        AscendC::Mul<float, false>(lt_dc_z[0],          lt_u1, lt_ztmp2[64],  64ULL, mc_u8, rk_mul64);
        AscendC::Mul<float, false>(lt_dc_z[512],        lt_u2, lt_ztmp2[128], 64ULL, mc_u8, rk_mul64);
        AscendC::Mul<float, false>(lt_dc_z[1024],       lt_u3, lt_ztmp2[192], 64ULL, mc_u8, rk_mul64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(lt_dc_z[0], lt_dc_z[0], lt_dc_z[512],  64ULL, mc_u8, b_def64);
        AscendC::Add<float, false>(lt_z[0],    lt_z[0],    lt_dc_z[1024], 64ULL, mc_u8, b_def64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float, false>(lt_z[0], lt_z[0], lt_dc_z[0], 64ULL, mc_u8, b_def64);
      }
      AscendC::ResetMask();
      AscendC::PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ __attribute__((always_inline)) inline void BacktransformMedInUB(
    uint32_t my_cols, uint8_t mc_u8, int32_t mc_i32, const float* inv_h_loc, __ubuf__ float* dc_ws_ub,
    const AscendC::LocalTensor<float>& lt_p0, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2,
    const AscendC::LocalTensor<float>& lt_hsave, const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_z,
    const AscendC::LocalTensor<uint32_t>& lt_off_rep8, const AscendC::UnaryRepeatParams& u_def64,
    const AscendC::BinaryRepeatParams& b_def64, const AscendC::BinaryRepeatParams& mv_mul64, const AscendC::BinaryRepeatParams& rk_mul64
  ) {
    uint32_t act_uis[64], act_n = 0u;
    for (uint32_t ui = 2u; ui < n_; ++ui) {
      if (inv_h_loc[ui] > 0.0f) act_uis[act_n++] = ui;
    }
    uint32_t n_quads = act_n >> 2u;
    float u_dots[96];
    if (n_quads > 0u) {
      AscendC::Duplicate<float, true>(lt_dc_ws[0], 0.0f, 64ULL, 2, 1, 8);
      ComputeQuadReflectorDots64(n_quads, act_uis, lt_hsave, lt_dc_z, lt_dc_ws[0], dc_ws_ub, u_dots);
    }
    AscendC::Duplicate<float, true>(lt_p0[0], 0.0f, 32ULL, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    ApplyQuadReflectorsMed64(
      n_quads, my_cols, mc_u8, mc_i32, act_uis, inv_h_loc, u_dots,
      lt_p0, lt_z, lt_ztmp, lt_ztmp2, lt_hsave, lt_dc_z, lt_off_rep8, u_def64, b_def64, mv_mul64, rk_mul64
    );
    for (uint32_t idx_a = n_quads * 4u; idx_a < act_n; ++idx_a) {
      uint32_t ui = act_uis[idx_a];
      AscendC::LocalTensor<float> lt_ui = lt_hsave[ui * 64u];
      AscendC::Mul<float, true>(lt_ztmp[0], lt_z[0], lt_ui, 64ULL, mc_u8, mv_mul64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::WholeReduceSum<float, false>(lt_p0[0], lt_ztmp[0], 64, mc_i32, 1, 1, 8);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls<float, true>(lt_p0[0], lt_p0[0], -inv_h_loc[ui], 8ULL, 1, u_def64);
      AscendC::ResetMask();
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Gather(lt_ztmp2[0], lt_p0[0], lt_off_rep8, 0u, my_cols * 8u);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::MulAddDst<float, float, true>(lt_z[0], lt_ui, lt_ztmp2[0], 64ULL, mc_u8, rk_mul64);
      AscendC::ResetMask();
      AscendC::PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ __attribute__((always_inline)) inline void FixSignMedCols(
    uint32_t my_cols, int32_t mc_i32, __ubuf__ float* z_ub, __ubuf__ float* rb_st_ub,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_ztmp2, const AscendC::LocalTensor<float>& lt_dc_z,
    const AscendC::UnaryRepeatParams& u_def64
  ) {
    if constexpr (n_ == 64u) {
      AscendC::WholeReduceMax<float, true>(lt_ztmp2[0],  lt_z[0], 64, mc_i32, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
      AscendC::WholeReduceMin<float, false>(lt_ztmp2[8], lt_z[0], 64, mc_i32, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      #pragma clang loop unroll(disable)
      for (uint32_t c = 0; c < my_cols; ++c) {
        float mx = rb_st_ub[c], neg_mn = -rb_st_ub[8u + c];
        bool neg_col = false;
        if (neg_mn > mx) {
          neg_col = true;
        } else if (mx > neg_mn) {
          neg_col = false;
        } else {
          EIGH_PAD64() EIGH_PAD64() EIGH_PAD16() EIGH_PAD16() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4()
          neg_col = ScanColSignTieBreak(z_ub + c * 64u);
        }
        if (neg_col) AscendC::Muls<float, false>(lt_z[c * 64u], lt_z[c * 64u], -1.0f, 64ULL, 1, u_def64);
      }
      AscendC::ResetMask();
      AscendC::PipeBarrier<PIPE_V>();
    } else {
      for (uint32_t c = 0; c < my_cols; ++c) {
        AscendC::ReduceMax(lt_ztmp2[c * 16u],      lt_z[c * 64u], lt_dc_z[c * 128u],       static_cast<int32_t>(n_));
        AscendC::ReduceMin(lt_ztmp2[c * 16u + 8u], lt_z[c * 64u], lt_dc_z[c * 128u + 64u], static_cast<int32_t>(n_));
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      for (uint32_t c = 0; c < my_cols; ++c) {
        float mx = rb_st_ub[c * 16u], neg_mn = -rb_st_ub[c * 16u + 8u];
        bool neg_col = false;
        if (neg_mn > mx) {
          neg_col = true;
        } else if (mx > neg_mn) {
          neg_col = false;
        } else {
          EIGH_PAD16() EIGH_PAD4()
          neg_col = ScanColSignTieBreak(z_ub + c * 64u);
        }
        if (neg_col) AscendC::Muls(lt_z[c * 64u], lt_z[c * 64u], -1.0f, 64);
      }
      AscendC::PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ __attribute__((always_inline)) inline void WriteBackMedAligned(
    uint32_t p, uint32_t col_start, uint32_t my_cols, const float* d, __ubuf__ float* val_ub, __ubuf__ uint32_t* st_u32_s4,
    const AscendC::LocalTensor<float>& lt_val, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_ztmp,
    const AscendC::LocalTensor<int32_t>& lt_i32, const AscendC::LocalTensor<uint32_t>& lt_st_u32_s4,
    const AscendC::GlobalTensor<float>& gt_val, const AscendC::GlobalTensor<float>& gt_vec
  ) {
    if (my_cols > 0u) {
      AscendC::LocalTensor<int32_t> lt_st_i32_s4 = lt_i32[46080];
      for (uint32_t c = 0; c < 8u; ++c) st_u32_s4[c] = c * 256u;
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      for (int32_t rep = 8; rep < 512; rep <<= 1) {
        AscendC::Adds(lt_st_i32_s4[rep], lt_st_i32_s4[0], (rep >> 3) * 4, rep);
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::Gather(lt_ztmp, lt_z, lt_st_u32_s4, 0u, n_ * 8u);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    }
    if (p == 0u) {
      for (uint32_t i = 0; i < n_; ++i) val_ub[i] = d[i];
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
      AscendC::DataCopyExtParams cv(1, static_cast<uint32_t>(n_ * sizeof(float)), 0u, 0u, 0u);
      AscendC::DataCopyPad(gt_val, lt_val, cv);
    }
    SafeSyncAll();
    if (my_cols > 0u) {
      AscendC::DataCopyParams cp_q_med(static_cast<uint16_t>(n_), 1, 0, static_cast<uint16_t>((n_ - 8u) / 8u));
      AscendC::DataCopy(gt_vec[col_start], lt_ztmp, cp_q_med);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  __aicore__ __attribute__((always_inline)) inline void WriteBackMedUnaligned(
    uint32_t p, uint32_t col_start, uint32_t my_cols, uint32_t row_blk_gap,
    const float* d, __gm__ float* cur_vec, __ubuf__ float* val_ub, __ubuf__ int32_t* off_tr_i32,
    const AscendC::LocalTensor<float>& lt_val, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_hsave,
    const AscendC::LocalTensor<int32_t>& lt_tr64_i32, const AscendC::LocalTensor<uint32_t>& lt_off_tr64,
    const AscendC::GlobalTensor<float>& gt_val, const AscendC::GlobalTensor<float>& gt_vec,
    const AscendC::DataCopyExtParams& copy_in, const AscendC::DataCopyPadExtParams<float>& pad_in
  ) {
    if (my_cols > 0u) {
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::GlobalTensor<float> gt_qt_slice;
      gt_qt_slice.SetGlobalBuffer(cur_vec + static_cast<uint64_t>(col_start) * n_, my_cols * n_);
      AscendC::DataCopyExtParams cq_qt(static_cast<uint16_t>(my_cols), static_cast<uint32_t>(n_ * sizeof(float)), row_blk_gap, 0u, 0u);
      AscendC::DataCopyPad(gt_qt_slice, lt_z, cq_qt);
    }
    if (p == 0u) {
      for (uint32_t i = 0; i < n_; ++i) val_ub[i] = d[i];
      for (uint32_t j = 0; j < 64u; ++j) off_tr_i32[j] = static_cast<int32_t>(j * 256u);
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
      AscendC::DataCopyExtParams cv(1, static_cast<uint32_t>(n_ * sizeof(float)), 0u, 0u, 0u);
      AscendC::DataCopyPad(gt_val, lt_val, cv);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      for (int32_t rep = 1; rep < 64; rep <<= 1) {
        AscendC::Adds(lt_tr64_i32[rep * 64], lt_tr64_i32[0], rep * 4, rep * 64);
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    }
    SafeSyncAll();
    if (p == 0u) {
      AscendC::DataCopyPad(lt_hsave, gt_vec, copy_in, pad_in);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      AscendC::Gather(lt_z, lt_hsave, lt_off_tr64, 0u, n_ * 64u);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      AscendC::DataCopyExtParams cq(static_cast<uint16_t>(n_), static_cast<uint32_t>(n_ * sizeof(float)), row_blk_gap, 0u, 0u);
      AscendC::DataCopyPad(gt_vec, lt_z, cq);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  __aicore__ __attribute__((always_inline)) inline void RunMedDivideAndConquer(
    uint32_t p, uint32_t sym_cap, uint32_t col_start, uint32_t my_cols, uint8_t mc_u8, float scale_pow2, float* d, float* e, __ubuf__ float* z_ub, __ubuf__ float* ztmp_ub, __ubuf__ float* dc_z_ub, __ubuf__ float* dc_ws_ub, __ubuf__ float* dc_st_ub,
    const AscendC::LocalTensor<float>& lt_t64, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2, const AscendC::LocalTensor<float>& lt_z_final, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<float>& lt_dc_ws, const AscendC::LocalTensor<float>& lt_dc_st,
    const AscendC::LocalTensor<int32_t>& lt_bc64_i32, const AscendC::LocalTensor<uint32_t>& lt_off_bc64, const AscendC::LocalTensor<uint32_t>& lt_off_rep8, const AscendC::LocalTensor<uint32_t>& lt_off_rep8_4k, const AscendC::LocalTensor<uint16_t>& lt_u16, const AscendC::LocalTensor<uint16_t>& lt_ztmp_u16, const AscendC::LocalTensor<uint16_t>& lt_dmsk_u16,
    const AscendC::GlobalTensor<float>& gt_ws_leaf, const AscendC::GlobalTensor<float>& gt_ws_roots_s1, const AscendC::GlobalTensor<float>& gt_ws_roots_s2, const AscendC::UnaryRepeatParams& u_def64, const AscendC::BinaryRepeatParams& b_def64, const AscendC::BinaryRepeatParams& rk_mul64
  ) {
    AscendC::LocalTensor<float> lt_eye64 = lt_dc_z[8192];
    AscendC::LocalTensor<uint16_t> lt_eye64_u16 = lt_u16[32768u * 2u];
    AscendC::Duplicate<float, true>(lt_eye64, 1.0f, 64ULL, 64, 1, 8);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Not(lt_ztmp_u16, lt_dmsk_u16, 8192);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::And(lt_eye64_u16, lt_eye64_u16, lt_ztmp_u16, 8192);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    d[0] = z_ub[0]; d[1] = z_ub[65u]; e[0] = z_ub[64u]; e[n_ - 1u] = 0.0f;

    float top[64], bot[64], ths[64], bhs[64], ad[64], az[64], dd[64], tau_a[64], rho_arr[2], giv_cg[64], giv_sg[64];
    int32_t orig_ord[64], acol_dc[64], final_oc[64], bounds[5], K_arr[2], D_arr[2], triv_arr[2], s1_triv[2] = {1, 1}, giv_cp[64], giv_pk[64], n_giv = 0;
    uint32_t s1_src[64];

    DispatchMedLeavesToGM(p, bounds, d, e, dc_z_ub, dc_st_ub, lt_z, lt_dc_z, lt_bc64_i32, gt_ws_leaf, u_def64);
    UnpackMedLeavesFromGM(sym_cap, bounds, d, top, bot, z_ub, ztmp_ub, lt_z, lt_ztmp, gt_ws_leaf);

    AscendC::BinaryRepeatParams b_bc8_s1(1, 1, 0, 8, 8, 0), b_bc64_s1(1, 1, 1, 8, 8, 0), b_bc64_row0(1, 1, 1, 8, 0, 8);
    AscendC::UnaryRepeatParams u_s1_32(1, 1, 8, 8);
    AscendC::BinaryRepeatParams b_s1_32(1, 1, 1, 8, 8, 8), rk_s1_32(1, 1, 0, 8, 0, 1);
    constexpr uint32_t s1_m1_off = (n_ == 64u) ? 1024u : 768u;
    constexpr int32_t s1_tot_len = (n_ == 64u) ? 2048 : 1792;

    for (int32_t step = 1; step < 4; step *= 2) {
      bool final_level = (step * 2 >= 4);
      uint32_t num_merges = static_cast<uint32_t>(4 / (2 * step)), sub_P = 8u / num_merges, my_m = p / sub_P, sub_p = p % sub_P;
      if (final_level) n_giv = 0;
      for (uint32_t m_idx = 0; m_idx < num_merges; ++m_idx) DeflateMedSingleMerge(m_idx, step, final_level, bounds, d, e, top, bot, ths, bhs, ad, az, dd, orig_ord, acol_dc, K_arr, D_arr, triv_arr, rho_arr, n_giv, giv_cp, giv_pk, giv_cg, giv_sg, lt_z, lt_dc_st, u_def64);
      SolveMedRootsAndSync(p, step, final_level, num_merges, sub_P, my_m, sub_p, s1_tot_len, s1_triv, bounds, K_arr, rho_arr, ad, az, orig_ord, tau_a, dc_ws_ub, dc_st_ub, lt_ztmp, lt_dc_z, lt_dc_ws, lt_dc_st, gt_ws_leaf, gt_ws_roots_s1, gt_ws_roots_s2);
      if (!final_level) {
        for (uint32_t m_idx = 0; m_idx < num_merges; ++m_idx) UpdateMedStep1SingleMerge(m_idx, step, my_m, sub_p, s1_m1_off, bounds, K_arr, D_arr, triv_arr, rho_arr, s1_triv, s1_src, d, top, bot, ths, bhs, ad, az, dd, tau_a, orig_ord, acol_dc, dc_ws_ub, dc_z_ub, lt_t64, lt_z, lt_ztmp, lt_ztmp2, lt_dc_ws, lt_dc_z, lt_dc_st, lt_off_bc64, lt_off_rep8_4k, lt_u16, lt_dmsk_u16, lt_eye64_u16, gt_ws_leaf, u_s1_32, b_bc64_s1, b_bc64_row0, b_def64, b_s1_32, rk_s1_32);
      } else {
        UpdateMedStep2Final(col_start, my_cols, mc_u8, scale_pow2, bounds, K_arr, D_arr, triv_arr, rho_arr, s1_triv, s1_src, n_giv, giv_cp, giv_pk, giv_cg, giv_sg, d, ad, az, dd, tau_a, orig_ord, acol_dc, final_oc, dc_ws_ub, dc_z_ub, lt_t64, lt_z, lt_ztmp, lt_ztmp2, lt_z_final, lt_dc_ws, lt_dc_z, lt_dc_st, lt_off_bc64, lt_off_rep8, lt_off_rep8_4k, lt_u16, lt_dmsk_u16, lt_eye64_u16, u_def64, u_s1_32, b_bc8_s1, b_bc64_row0, b_def64, rk_mul64);
      }
    }
  }

  __aicore__ __attribute__((noinline)) void SolveMediumCoop8(
    __gm__ float* cur_A, __gm__ float* cur_val, __gm__ float* cur_vec, __gm__ float* group_ws,
    const AscendC::LocalTensor<float>& ub_all, uint32_t p
  ) {
    (void)group_ws;
    float d[64], e[64], inv_h_loc[64];
    __ubuf__ float* base_ptr = reinterpret_cast<__ubuf__ float*>(ub_all.GetPhyAddr());
    __ubuf__ uint32_t* u32_base = reinterpret_cast<__ubuf__ uint32_t*>(base_ptr);
    AscendC::LocalTensor<int32_t> lt_i32  = ub_all.ReinterpretCast<int32_t>();
    AscendC::LocalTensor<uint32_t> lt_u32 = ub_all.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint16_t> lt_u16 = ub_all.ReinterpretCast<uint16_t>();

    uint32_t sym_cap = n_ * n_;
    AscendC::GlobalTensor<float> gt_A, gt_val, gt_vec, gt_ws_leaf, gt_ws_roots_s1, gt_ws_roots_s2;
    gt_A.SetGlobalBuffer(cur_A, sym_cap); gt_val.SetGlobalBuffer(cur_val, n_); gt_vec.SetGlobalBuffer(cur_vec, sym_cap);
    gt_ws_leaf.SetGlobalBuffer(cur_vec, 2048u);
    gt_ws_roots_s1.SetGlobalBuffer(cur_vec + sym_cap - 512u, 256u);
    gt_ws_roots_s2.SetGlobalBuffer(cur_vec + sym_cap - 256u, 256u);

    uint32_t row_al = (n_ + 7u) & ~7u, row_blk_gap = (64u - row_al) / 8u, col_start = p * 8u;
    uint32_t col_end = (col_start + 8u < n_) ? (col_start + 8u) : n_;
    uint32_t my_cols = (col_start < col_end) ? (col_end - col_start) : 0u;
    uint8_t right_pad = static_cast<uint8_t>(row_al - n_), mc_u8 = static_cast<uint8_t>(my_cols);
    int32_t mc_i32 = static_cast<int32_t>(my_cols);

    AscendC::UnaryRepeatParams u_def64(1, 1, 8, 8);
    AscendC::BinaryRepeatParams b_def64(1, 1, 1, 8, 8, 8), mv_mul64(1, 1, 1, 8, 8, 0), rk_mul64(1, 1, 0, 8, 0, 1);

    InitMedMasksVec(row_al, ub_all[4096], ub_all[16384], lt_i32[32768], lt_i32[36864], lt_u32[3072], u32_base + 2560, u32_base + 48128, reinterpret_cast<__ubuf__ int32_t*>(base_ptr + 20480));
    AscendC::DataCopyExtParams copy_in(static_cast<uint16_t>(n_), static_cast<uint32_t>(n_ * sizeof(float)), 0u, row_blk_gap, 0u);
    AscendC::DataCopyPadExtParams<float> pad_in(right_pad > 0, 0, right_pad, 0.0f);
    AscendC::DataCopyPad(ub_all[4096], gt_A, copy_in, pad_in);
    InitMedOffsetsAndMasksScalar(row_al, lt_i32[2560], lt_i32[48128], lt_i32[20480], u32_base + 32768, u32_base + 36864);
    float anorm = SymmetrizeMedMatrix(ub_all[4096], ub_all[8192], ub_all[12288], ub_all[784], ub_all[1024], ub_all[32768], lt_u32[20480], lt_u16[4096u * 2u], lt_u16[8192u * 2u], lt_u16[32768u * 2u], lt_u16[36864u * 2u], base_ptr + 784, u_def64, b_def64);
    if (anorm == 0.0f) {
      WriteMedZeroAnorm(p, sym_cap, row_blk_gap, ub_all[0], ub_all[4096], base_ptr + 4096, gt_val, gt_vec);
      return;
    }
    float scale_pow2, inv_scale_pow2;
    GetPow2Scale(anorm, scale_pow2, inv_scale_pow2);
    TridiagMedInUB(inv_scale_pow2, d, e, inv_h_loc, base_ptr + 4096, base_ptr + 16384, base_ptr + 32768, base_ptr + 712, ub_all[4096], ub_all[16384], ub_all[32768], ub_all[8192], ub_all[12288], ub_all[128], ub_all[192], ub_all[712], ub_all[776], lt_i32, lt_i32[20480], lt_u32[3072], lt_u32[48128], u_def64, b_def64, mv_mul64, rk_mul64);
    RunMedDivideAndConquer(p, sym_cap, col_start, my_cols, mc_u8, scale_pow2, d, e, base_ptr + 4096, base_ptr + 8192, base_ptr + 24576, base_ptr + 40960, base_ptr + 46080, ub_all[192], ub_all[4096], ub_all[8192], ub_all[12288], ub_all[36864], ub_all[24576], ub_all[40960], ub_all[46080], lt_i32[41984], lt_u32[41984], lt_u32[48128], lt_u32[20480], lt_u16, lt_u16[8192u * 2u], lt_u16[36864u * 2u], gt_ws_leaf, gt_ws_roots_s1, gt_ws_roots_s2, u_def64, b_def64, rk_mul64);
    if (my_cols > 0u) {
      BacktransformMedInUB(my_cols, mc_u8, mc_i32, inv_h_loc, base_ptr + 40960, ub_all[128], ub_all[4096], ub_all[8192], ub_all[12288], ub_all[16384], ub_all[40960], ub_all[24576], lt_u32[48128], u_def64, b_def64, mv_mul64, rk_mul64);
      FixSignMedCols(my_cols, mc_i32, base_ptr + 4096, base_ptr + 12288, ub_all[4096], ub_all[12288], ub_all[24576], u_def64);
    }
    if ((n_ & 7u) == 0u) {
      WriteBackMedAligned(p, col_start, my_cols, d, base_ptr, u32_base + 46080, ub_all[0], ub_all[4096], ub_all[8192], lt_i32, lt_u32[46080], gt_val, gt_vec);
    } else {
      WriteBackMedUnaligned(p, col_start, my_cols, row_blk_gap, d, cur_vec, base_ptr, reinterpret_cast<__ubuf__ int32_t*>(base_ptr + 20480), ub_all[0], ub_all[4096], ub_all[16384], lt_i32[20480], lt_u32[20480], gt_val, gt_vec, copy_in, pad_in);
    }
  }
