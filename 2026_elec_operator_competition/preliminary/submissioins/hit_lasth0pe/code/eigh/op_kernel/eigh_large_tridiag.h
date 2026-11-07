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

  __aicore__ __attribute__((always_inline)) inline void InitLargeSymTables(
    uint32_t p, uint32_t P, uint32_t stride, uint32_t p_stride,
    __ubuf__ uint32_t* off_tr64x16, __ubuf__ uint32_t* off_pfull, __ubuf__ uint32_t* off_bc16_u, __ubuf__ uint32_t* off_rblk_u, __ubuf__ uint32_t* off_tr16,
    const AscendC::LocalTensor<int32_t>& lt_tr64x16_i, const AscendC::LocalTensor<float>& lt_z
  ) {
    for (uint32_t di = 0; di < 64u; ++di) off_tr64x16[di] = di * 64u;
    for (uint32_t r = 0; r < 512u; ++r) off_pfull[r] = ((r % P) * p_stride + (r / P)) * 4u;
    for (uint32_t idx = 0; idx < 128u; ++idx) off_bc16_u[idx] = (idx >> 3u) * 4u;
    for (uint32_t q = 0; q < 32u; ++q) {
      uint32_t rr = q * P + p, b_off = ((rr < stride) ? rr : 0u) * 4u;
      for (uint32_t ln = 0; ln < 8u; ++ln) off_rblk_u[q * 8u + ln] = b_off;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (uint32_t dk = 1u; dk < 16u; ++dk) AscendC::Adds(lt_tr64x16_i[dk * 64u], lt_tr64x16_i[0], static_cast<int32_t>(dk * 4u), 64);

    for (uint32_t ii = 0; ii < 16u; ++ii) {
      for (uint32_t jj = 0; jj < 16u; ++jj) off_tr16[ii * 16u + jj] = (jj * 16u + ii) * 4u;
    }
    AscendC::Duplicate(lt_z[256], 0.0f, 256);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline void SymmetrizeLargeDiagTile(
    uint32_t r0, uint32_t c0, uint32_t r_cnt, uint32_t c_cnt, uint32_t gm_stride, uint32_t& diag_slot, float& local_anorm,
    __ubuf__ float* stage_ub, const AscendC::LocalTensor<float>& lt_stage, const AscendC::GlobalTensor<float>& gt_A, const AscendC::GlobalTensor<float>& gt_sym
  ) {
    uint32_t d_off = 512u + (diag_slot & 3u) * 256u;
    ++diag_slot;
    AscendC::LocalTensor<float> lt_diag_a = lt_stage[d_off];
    __ubuf__ float* diag_a = stage_ub + d_off;
    uint32_t in_r = (uplo_ == 0u) ? r_cnt : c_cnt, in_c = (uplo_ == 0u) ? c_cnt : r_cnt;
    uint32_t a_off = (uplo_ == 0u) ? (r0 * n_ + c0) : (c0 * n_ + r0);
    uint8_t rpad = static_cast<uint8_t>(16u - in_c);
    AscendC::DataCopyExtParams cp_in(static_cast<uint16_t>(in_r), static_cast<uint32_t>(in_c * sizeof(float)), static_cast<uint32_t>((n_ - in_c) * sizeof(float)), 0u, 0u);
    AscendC::DataCopyPadExtParams<float> pd_in(rpad > 0, 0, rpad, 0.0f);
    AscendC::DataCopyPad(lt_diag_a, gt_A[a_off], cp_in, pd_in);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    for (uint32_t ii = 0; ii < r_cnt; ++ii) {
      for (uint32_t jj = 0; jj <= ii; ++jj) {
        float v = (uplo_ == 0u) ? diag_a[ii * 16u + jj] : diag_a[jj * 16u + ii];
        diag_a[ii * 16u + jj] = v;
        diag_a[jj * 16u + ii] = v;
        float av = MyAbs(v);
        if (av > local_anorm) local_anorm = av;
      }
      for (uint32_t jj = r_cnt; jj < 16u; ++jj) diag_a[ii * 16u + jj] = 0.0f;
    }
    for (uint32_t ii = r_cnt; ii < 16u; ++ii) {
      for (uint32_t jj = 0; jj < 16u; ++jj) diag_a[ii * 16u + jj] = 0.0f;
    }
    AscendC::DataCopyParams cp_diag(static_cast<uint16_t>(r_cnt), static_cast<uint16_t>(c_cnt / 8u), static_cast<uint16_t>((16u - c_cnt) / 8u), static_cast<uint16_t>((gm_stride - c_cnt) / 8u));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopy(gt_sym[r0 * gm_stride + c0], lt_diag_a, cp_diag);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void SymmetrizeLargeOffDiagTile(
    uint32_t r0, uint32_t c0, uint32_t r_cnt, uint32_t c_cnt, uint32_t gm_stride,
    const AscendC::LocalTensor<float>& lt_tile_a, const AscendC::LocalTensor<float>& lt_tile_t, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<uint32_t>& lt_off_tr16, const AscendC::GlobalTensor<float>& gt_A, const AscendC::GlobalTensor<float>& gt_sym
  ) {
    uint32_t in_r = (uplo_ == 0u) ? r_cnt : c_cnt, in_c = (uplo_ == 0u) ? c_cnt : r_cnt;
    uint32_t a_off = (uplo_ == 0u) ? (r0 * n_ + c0) : (c0 * n_ + r0);
    uint8_t rpad = static_cast<uint8_t>(16u - in_c);
    AscendC::DataCopyExtParams cp_in(static_cast<uint16_t>(in_r), static_cast<uint32_t>(in_c * sizeof(float)), static_cast<uint32_t>((n_ - in_c) * sizeof(float)), 0u, 0u);
    AscendC::DataCopyPadExtParams<float> pd_in(rpad > 0, 0, rpad, 0.0f);
    const AscendC::LocalTensor<float>& lt_in = (uplo_ == 0u) ? lt_tile_a : lt_tile_t;
    const AscendC::LocalTensor<float>& lt_out = (uplo_ == 0u) ? lt_tile_t : lt_tile_a;
    AscendC::DataCopyPad(lt_in, gt_A[a_off], cp_in, pd_in);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    if (in_r < 16u) {
      AscendC::Duplicate(lt_in[in_r * 16u], 0.0f, static_cast<int32_t>((16u - in_r) * 16u));
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Gather(lt_out, lt_in, lt_off_tr16, 0u, 256u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Abs(lt_z[512], lt_tile_a, 256);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(lt_z[256], lt_z[256], lt_z[512], 256);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

    AscendC::DataCopyParams cp_lo(static_cast<uint16_t>(r_cnt), static_cast<uint16_t>(c_cnt / 8u), static_cast<uint16_t>((16u - c_cnt) / 8u), static_cast<uint16_t>((gm_stride - c_cnt) / 8u));
    AscendC::DataCopyParams cp_up(static_cast<uint16_t>(c_cnt), static_cast<uint16_t>(r_cnt / 8u), static_cast<uint16_t>((16u - r_cnt) / 8u), static_cast<uint16_t>((gm_stride - r_cnt) / 8u));
    AscendC::DataCopy(gt_sym[r0 * gm_stride + c0], lt_tile_a, cp_lo);
    AscendC::DataCopy(gt_sym[c0 * gm_stride + r0], lt_tile_t, cp_up);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void SymmetrizeAndLoadLargePanel(
    uint32_t p, uint32_t P, uint32_t stride, uint32_t max_rows, uint32_t gm_stride, float& anorm, float& scale_pow2, float& inv_scale_pow2,
    __ubuf__ float* stage_ub, __ubuf__ float* meta_ub, __ubuf__ float* red_ub, __ubuf__ float* z_ub,
    const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_meta, const AscendC::LocalTensor<float>& lt_red, const AscendC::LocalTensor<float>& lt_work, const AscendC::LocalTensor<uint32_t>& lt_off_tr16, const AscendC::GlobalTensor<float>& gt_A, const AscendC::GlobalTensor<float>& gt_sym, const AscendC::GlobalTensor<float>& gt_anorm
  ) {
    uint32_t num_blk = stride / 16u, total_pairs = (num_blk * (num_blk + 1u)) / 2u, diag_slot = 0u, bi = 0u, bj = 0u;
    float local_anorm = 0.0f;
    for (uint32_t pair_idx = 0; pair_idx < total_pairs; ++pair_idx) {
      uint32_t r0 = bi * 16u, c0 = bj * 16u;
      if (pair_idx % P == p && r0 < n_) {
        uint32_t r_cnt = (r0 + 16u <= n_) ? 16u : (n_ - r0), c_cnt = (c0 + 16u <= n_) ? 16u : (n_ - c0);
        if (bi == bj) SymmetrizeLargeDiagTile(r0, c0, r_cnt, c_cnt, gm_stride, diag_slot, local_anorm, stage_ub, lt_stage, gt_A, gt_sym);
        else SymmetrizeLargeOffDiagTile(r0, c0, r_cnt, c_cnt, gm_stride, lt_stage[0], lt_stage[256], lt_z, lt_off_tr16, gt_A, gt_sym);
      }
      if (++bj > bi) { ++bi; bj = 0u; }
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax(lt_red, lt_z[256], lt_work, 256);
    if (gm_stride < stride) { AscendC::Duplicate(lt_stage[512], 0.0f, static_cast<int32_t>(stride)); AscendC::Duplicate(lt_mat, 0.0f, static_cast<int32_t>(max_rows * stride)); }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    if (red_ub[0] > local_anorm) local_anorm = red_ub[0];
    meta_ub[0] = local_anorm;
    for (uint32_t k = 1u; k < 8u; ++k) meta_ub[k] = 0.0f;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopy(gt_anorm[p * 8u], lt_meta, 8);
    SafeSyncAll();

    AscendC::DataCopy(lt_z, gt_anorm, static_cast<int32_t>(P * 8u));
    AscendC::DataCopy(lt_stage[512], gt_sym[(n_ - 1u) * gm_stride], static_cast<int32_t>(gm_stride));
    uint32_t my_rows = 0u;
    for (uint32_t q = 0; q < max_rows; ++q) {
      uint32_t r = q * P + p;
      if (r < n_) { AscendC::DataCopy(lt_mat[q * stride], gt_sym[r * gm_stride], static_cast<int32_t>(gm_stride)); my_rows++; }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    SafeSyncAll();

    for (uint32_t k = 0; k < P; ++k) if (z_ub[k * 8u] > anorm) anorm = z_ub[k * 8u];
    GetPow2Scale(anorm, scale_pow2, inv_scale_pow2);
    if (anorm > 0.0f) {
      AscendC::Muls(lt_stage[512], lt_stage[512], inv_scale_pow2, static_cast<int32_t>(stride));
      if (my_rows > 0u) AscendC::Muls(lt_mat, lt_mat, inv_scale_pow2, static_cast<int32_t>(my_rows * stride));
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    }
  }

  __aicore__ __attribute__((always_inline)) inline void PrefetchLargeRow256Norm(
    uint32_t vc16_next_u, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_wscr,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_red
  ) {
    int32_t vc16_next = static_cast<int32_t>(vc16_next_u);
    AscendC::Mul(lt_stage[1536], lt_stage[512], lt_wscr[3584], vc16_next);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(lt_z[0], lt_stage[1536], lt_stage[1536], vc16_next);
    AscendC::PipeBarrier<PIPE_V>();
    if (vc16_next_u > 64u) {
      for (uint32_t h = 256u; h >= 64u; h >>= 1u) {
        if (vc16_next_u > h) {
          AscendC::Add(lt_z[0], lt_z[0], lt_z[h], static_cast<int32_t>(vc16_next_u - h));
          AscendC::PipeBarrier<PIPE_V>();
        }
      }
    }
    AscendC::WholeReduceSum<float, true>(lt_red, lt_z[0], (vc16_next_u > 64u) ? 64 : vc16_next, 1, 1, 1, 8);
    AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void EvalLargeCoopStepLocalMV(
    uint32_t p, uint32_t P, uint32_t ui, int32_t l, uint32_t vc16, int32_t vc16_i32, uint32_t num_q, uint32_t stride, int32_t p_str_i, uint32_t ping_pp, float inv_h, uint32_t& hvec_used,
    const AscendC::LocalTensor<float>& lt_u, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z, const AscendC::GlobalTensor<float>& gt_meta
  ) {
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    if constexpr (n_ == 256u) {
      AscendC::Adds(lt_u, lt_stage[1536], 0.0f, vc16_i32);
      if (stride > vc16) AscendC::Duplicate(lt_u[vc16], 0.0f, static_cast<int32_t>(stride - vc16));
      AscendC::PipeBarrier<PIPE_V>();
    }
    if (P > 0u && p == (ui % P)) {
      uint32_t al_u = (ui + 7u) & ~7u;
      AscendC::Adds(lt_mat[10240u + hvec_used], lt_u, 0.0f, static_cast<int32_t>(al_u));
      hvec_used += al_u;
    }
    if (num_q > 0u) {
      uint32_t n_rep64 = ((static_cast<uint32_t>(l + 1) + 63u) & ~63u) >> 6u;
      AscendC::BinaryRepeatParams mul_s1(1, 1, 1, 8, static_cast<uint8_t>(stride >> 3u), 0);
      AscendC::Duplicate(lt_stage[0], 0.0f, p_str_i);
      AscendC::Mul<float, true>(lt_z[0], lt_mat[0], lt_u[0], 64ULL, static_cast<uint8_t>(num_q), mul_s1);
      for (uint32_t b = 1u; b < n_rep64; ++b) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst<float, float, false>(lt_z[0], lt_mat[b * 64u], lt_u[b * 64u], 64ULL, static_cast<uint8_t>(num_q), mul_s1);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::WholeReduceSum<float, false>(lt_stage[0], lt_z[0], 64, static_cast<int32_t>(num_q), 1, 1, 8);
      AscendC::ResetMask();
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Muls(lt_stage[0], lt_stage[0], inv_h, p_str_i);
    } else {
      AscendC::Duplicate(lt_stage[0], 0.0f, p_str_i);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::DataCopy(gt_meta[ping_pp + p * static_cast<uint32_t>(p_str_i)], lt_stage[0], p_str_i);
  }

  __aicore__ __attribute__((always_inline)) inline void UpdateLargeCoopStepRank2(
    int32_t l, uint32_t ul, uint32_t vc16, int32_t vc16_i32, int32_t vec_cnt, uint32_t num_q, uint32_t stride, uint32_t pp_slot, uint32_t ping_pp, float inv_h, __ubuf__ float* u_ub, __ubuf__ float* p_full, __ubuf__ float* red_ub,
    const AscendC::LocalTensor<float>& lt_u, const AscendC::LocalTensor<float>& lt_w, const AscendC::LocalTensor<float>& lt_pfull, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_red, const AscendC::LocalTensor<float>& lt_work, const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<uint32_t>& lt_off_pfull, const AscendC::LocalTensor<uint32_t>& lt_off_rblk, const AscendC::GlobalTensor<float>& gt_meta
  ) {
    AscendC::DataCopy(lt_z, gt_meta[ping_pp], static_cast<int32_t>(pp_slot) + vc16_i32);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Adds(lt_stage[512], lt_z[pp_slot], 0.0f, vc16_i32);
    AscendC::Gather(lt_pfull, lt_z, lt_off_pfull, 0u, vc16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(lt_z[0], lt_pfull, lt_u, vc16_i32);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (n_ == 256u) {
      int32_t wred_cnt = vc16_i32;
      if (vc16 > 64u) {
        for (uint32_t h = 256u; h >= 64u; h >>= 1u) if (vc16 > h) { AscendC::Add(lt_z[0], lt_z[0], lt_z[h], static_cast<int32_t>(vc16 - h)); AscendC::PipeBarrier<PIPE_V>(); }
        wred_cnt = 64;
      }
      AscendC::WholeReduceSum<float, true>(lt_red, lt_z[0], wred_cnt, 1, 1, 1, 8);
      AscendC::ResetMask();
    } else {
      AscendC::ReduceSum(lt_red, lt_z[0], lt_work, vc16_i32);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

    uint32_t cnt64_u = (static_cast<uint32_t>(l + 1) + 63u) & ~63u, n_rep64_u = cnt64_u >> 6u;
    float hh = 0.5f * inv_h * red_ub[0], neg_ul = -u_ub[l], neg_wl = -(p_full[l] - hh * u_ub[l]);
    AscendC::Adds(lt_w, lt_pfull, 0.0f, vc16_i32);
    if (cnt64_u > vc16) AscendC::Duplicate(lt_w[vc16], 0.0f, static_cast<int32_t>(cnt64_u - vc16));
    AscendC::PipeBarrier<PIPE_V>(); AscendC::Axpy(lt_w, lt_u, -hh, vc16_i32); AscendC::PipeBarrier<PIPE_V>();
    if (num_q > 0u) {
      AscendC::Gather(lt_stage[1024], lt_u, lt_off_rblk, 0u, num_q * 8u); AscendC::Gather(lt_stage[1280], lt_w, lt_off_rblk, 0u, num_q * 8u);
      AscendC::PipeBarrier<PIPE_V>(); AscendC::Muls(lt_stage[1024], lt_stage[1024], -1.0f, 512);
    }
    AscendC::Axpy(lt_stage[512], lt_w, neg_ul, vec_cnt); AscendC::PipeBarrier<PIPE_V>(); AscendC::Axpy(lt_stage[512], lt_u, neg_wl, vec_cnt);
    if constexpr (n_ == 256u) {
      if (ul >= 128u) { AscendC::PipeBarrier<PIPE_V>(); PrefetchLargeRow256Norm((ul + 15u) & ~15u, lt_stage, lt_wscr, lt_z, lt_red); }
      else AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    } else {
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
    if (num_q > 0u) {
      AscendC::BinaryRepeatParams rk_mla(1, 1, 0, static_cast<uint8_t>(stride >> 3u), 0, 1);
      uint8_t nq_u8 = static_cast<uint8_t>(num_q);
      AscendC::MulAddDst<float, float, true>(lt_mat[0], lt_w[0], lt_stage[1024], 64ULL, nq_u8, rk_mla);
      for (uint32_t b = 1u; b < n_rep64_u; ++b) AscendC::MulAddDst<float, float, false>(lt_mat[b * 64u], lt_w[b * 64u], lt_stage[1024], 64ULL, nq_u8, rk_mla);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t b = 0; b < n_rep64_u; ++b) AscendC::MulAddDst<float, float, false>(lt_mat[b * 64u], lt_u[b * 64u], lt_stage[1280], 64ULL, nq_u8, rk_mla);
      AscendC::ResetMask(); AscendC::PipeBarrier<PIPE_V>();
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    }
    if constexpr (n_ == 256u) { if (ul < 128u) AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0); }
    else AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void EvalLargeCoopReflector(
    int32_t l, uint32_t ul, uint32_t vc16, int32_t vc16_i32, uint32_t stride, float rl_val, float& inv_h, float& ei,
    __ubuf__ float* row_cur, __ubuf__ float* u_ub, __ubuf__ float* red_ub, __ubuf__ float* wide_scr, __ubuf__ float* stage_ub,
    const AscendC::LocalTensor<float>& lt_u, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_red, const AscendC::LocalTensor<float>& lt_work
  ) {
    if constexpr (n_ != 256u) {
      for (uint32_t k = static_cast<uint32_t>(l); k < vc16; ++k) row_cur[k] = 0.0f;
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::Mul(lt_z[0], lt_stage[512], lt_stage[512], vc16_i32);
      AscendC::Adds(lt_u, lt_stage[512], 0.0f, vc16_i32);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::ReduceSum(lt_red, lt_z[0], lt_work, vc16_i32);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    } else if (ul > 0u) {
      wide_scr[3584u + ul - 1u] = 0.0f;
    }
    float nrm2_sub = red_ub[0];
    if (nrm2_sub > 1e-30f) {
      float nrm2 = nrm2_sub + rl_val * rl_val, g = nrm2 * MyRsqrt(nrm2);
      if (rl_val >= 0.0f) g = -g;
      ei = g; inv_h = 1.0f / (nrm2 - rl_val * g);
      if constexpr (n_ == 256u) stage_ub[1536u + ul] = rl_val - g;
      else { u_ub[l] = rl_val - g; if (stride > vc16) AscendC::Duplicate(lt_u[vc16], 0.0f, static_cast<int32_t>(stride - vc16)); }
    } else {
      ei = rl_val;
    }
  }

  __aicore__ __attribute__((always_inline)) inline void TridiagLargeCoopAbove128(
    uint32_t p, uint32_t P, uint32_t stride, uint32_t p_stride, uint32_t pp_slot, uint32_t meta_slot, uint32_t& hvec_used,
    __ubuf__ float* d, __ubuf__ float* e, __ubuf__ float* u_ub, __ubuf__ float* p_full, __ubuf__ float* inv_h_save, __ubuf__ float* red_ub, __ubuf__ float* wide_scr, __ubuf__ float* stage_ub,
    const AscendC::LocalTensor<float>& lt_u, const AscendC::LocalTensor<float>& lt_w, const AscendC::LocalTensor<float>& lt_pfull, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_red, const AscendC::LocalTensor<float>& lt_work, const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<uint32_t>& lt_off_pfull, const AscendC::LocalTensor<uint32_t>& lt_off_rblk, const AscendC::GlobalTensor<float>& gt_meta
  ) {
    __ubuf__ float* row_cur = stage_ub + 512;
    int32_t p_str_i = static_cast<int32_t>(p_stride);
    if constexpr (n_ == 256u) {
      for (uint32_t k = 0; k < 256u; ++k) wide_scr[3584u + k] = (k + 2u < n_) ? 1.0f : 0.0f;
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      int32_t vc16_init_i = static_cast<int32_t>(((n_ - 1u) + 15u) & ~15u);
      AscendC::Mul(lt_stage[1536], lt_stage[512], lt_wscr[3584], vc16_init_i); AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(lt_z[0], lt_stage[1536], lt_stage[1536], vc16_init_i); AscendC::PipeBarrier<PIPE_V>();
      AscendC::ReduceSum(lt_red, lt_z[0], lt_work, vc16_init_i); AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }

    for (int32_t i = static_cast<int32_t>(n_) - 1; i >= 128; --i) {
      int32_t l = i - 1;
      uint32_t ui = static_cast<uint32_t>(i), ul = static_cast<uint32_t>(l), vc16 = (static_cast<uint32_t>(l + 1) + 15u) & ~15u;
      int32_t vc16_i32 = static_cast<int32_t>(vc16), vec_cnt = static_cast<int32_t>((static_cast<uint32_t>(l + 1) + 7u) & ~7u);
      uint32_t ping_pp = (ui & 1u) * meta_slot, ping_row = ping_pp + pp_slot, num_q = (ul >= p) ? ((ul - p) / P + 1u) : 0u;
      if constexpr (n_ == 256u) AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      if (p == (ul % P)) AscendC::DataCopy(gt_meta[ping_row], lt_mat[(ul / P) * stride], vc16_i32);

      d[ui] = row_cur[ui];
      float inv_h = 0.0f, ei = 0.0f;
      EvalLargeCoopReflector(l, ul, vc16, vc16_i32, stride, row_cur[l], inv_h, ei, row_cur, u_ub, red_ub, wide_scr, stage_ub, lt_u, lt_stage, lt_z, lt_red, lt_work);
      inv_h_save[ui] = inv_h; e[ui] = ei;
      if (inv_h > 0.0f) EvalLargeCoopStepLocalMV(p, P, ui, l, vc16, vc16_i32, num_q, stride, p_str_i, ping_pp, inv_h, hvec_used, lt_u, lt_stage, lt_mat, lt_z, gt_meta);

      uint16_t fid = static_cast<uint16_t>(ui & 3u);
      AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(fid); AscendC::CrossCoreWaitFlag(fid);

      if (inv_h > 0.0f) {
        UpdateLargeCoopStepRank2(l, ul, vc16, vc16_i32, vec_cnt, num_q, stride, pp_slot, ping_pp, inv_h, u_ub, p_full, red_ub, lt_u, lt_w, lt_pfull, lt_stage, lt_mat, lt_z, lt_red, lt_work, lt_wscr, lt_off_pfull, lt_off_rblk, gt_meta);
      } else {
        AscendC::DataCopy(lt_stage[512], gt_meta[ping_row], vc16_i32);
        if constexpr (n_ == 256u) {
          if (ul >= 128u) { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0); PrefetchLargeRow256Norm((ul + 15u) & ~15u, lt_stage, lt_wscr, lt_z, lt_red); }
          else { AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0); }
        } else {
          AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        }
      }
    }
  }
