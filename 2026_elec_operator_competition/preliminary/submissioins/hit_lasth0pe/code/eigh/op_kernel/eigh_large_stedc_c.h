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

  __aicore__ __attribute__((always_inline)) inline void BacktransformLargeTail64(
    uint32_t p, uint32_t P, uint32_t my_cols, uint32_t stride, int32_t str_blk_s4, float anorm, uint32_t& hvec_used,
    const __ubuf__ float* inv_h_keep, __ubuf__ float* stage_ub,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_off_bc16_s4, const AscendC::GlobalTensor<float>& gt_sym,
    const AscendC::UnaryRepeatParams& s4_u16, const AscendC::BinaryRepeatParams& s4_mul_in, const AscendC::BinaryRepeatParams& s4_mul_in8,
    const AscendC::BinaryRepeatParams& s4_mla_up, const AscendC::BinaryRepeatParams& s4_mul_up8, const AscendC::BinaryRepeatParams& s4_add_mat_from8
  ) {
    bool any_tail64 = false;
    if (anorm > 0.0f) {
      for (int32_t i = 2; i <= 63; ++i) {
        uint32_t ui = static_cast<uint32_t>(i);
        if (inv_h_keep[ui] > 0.0f) {
          any_tail64 = true;
          if (p == (ui % P)) { hvec_used -= 64u; AscendC::DataCopy(gt_sym[ui * 64u], lt_mat[10240u + hvec_used], 64); }
        }
      }
    }
    SafeSyncAll();
    if (any_tail64 && my_cols > 0u) {
      AscendC::DataCopy(lt_z[8192], gt_sym[0], 4096);
      uint32_t t64_uis[64], t64_n = 0u;
      for (uint32_t ui = 2u; ui <= 63u; ++ui) if (inv_h_keep[ui] > 0.0f) t64_uis[t64_n++] = ui;
      uint32_t n_q64 = t64_n >> 2u;
      float udot64[96];
      if (n_q64 > 0u) AscendC::Duplicate<float, true>(lt_stage[512], 0.0f, 64ULL, 2, 1, 8);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      if (n_q64 > 0u) ComputeQuadReflectorDots64(n_q64, t64_uis, lt_z[8192], lt_z[0], lt_stage[512], stage_ub + 512, udot64);
      AscendC::Duplicate<float, true>(lt_stage[0], 0.0f, 64ULL, 1, 1, 8);
      AscendC::ResetMask(); AscendC::PipeBarrier<PIPE_V>();

      ApplyQuadReflectorsLargeTail64(my_cols, stride, n_q64, t64_uis, inv_h_keep, udot64, ub_all, lt_stage, lt_mat, lt_z, lt_off_bc16_s4, s4_u16, s4_mul_in8, s4_mla_up, s4_mul_up8, s4_add_mat_from8);
      for (uint32_t ia64 = n_q64 * 4u; ia64 < t64_n; ++ia64) {
        uint32_t ui = t64_uis[ia64];
        float inv_h = inv_h_keep[ui];
        AscendC::LocalTensor<float> lt_ui = lt_z[8192u + ui * 64u];
        for (uint32_t c_base = 0; c_base < my_cols; c_base += 16u) {
          uint32_t c_lim = (c_base + 16u < my_cols) ? (c_base + 16u) : my_cols;
          int32_t c_cnt = static_cast<int32_t>(c_lim - c_base);
          uint8_t c_cnt_u8 = static_cast<uint8_t>(c_cnt);
          AscendC::Mul<float, true>(lt_z[0], lt_mat[c_base * stride], lt_ui, 64ULL, c_cnt_u8, s4_mul_in);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::WholeReduceSum<float, false>(lt_stage[0], lt_z[0], 64, c_cnt, 1, 1, str_blk_s4);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Muls<float, true>(lt_stage[0], lt_stage[0], -inv_h, 16ULL, 1, s4_u16);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Gather(lt_stage[256], lt_stage[0], lt_off_bc16_s4, 0u, static_cast<uint32_t>(c_cnt) * 8u);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::MulAddDst<float, float, true>(lt_mat[c_base * stride], lt_ui, lt_stage[256], 64ULL, c_cnt_u8, s4_mla_up);
          AscendC::ResetMask(); AscendC::PipeBarrier<PIPE_V>();
        }
      }
    }
    SafeSyncAll();
  }

  __aicore__ __attribute__((always_inline)) inline void EvalQuadReflectorDotAbove64(
    uint32_t qi, uint32_t stride, uint32_t udot_gm_off, const __ubuf__ uint32_t* act_uis_all,
    const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::GlobalTensor<float>& gt_sym, const AscendC::BinaryRepeatParams& b64_dot
  ) {
    uint32_t b_i = qi * 4u, nr0 = ((act_uis_all[b_i + 0u] + 63u) & ~63u) >> 6u, nr1 = ((act_uis_all[b_i + 1u] + 63u) & ~63u) >> 6u, nr2 = ((act_uis_all[b_i + 2u] + 63u) & ~63u) >> 6u;
    AscendC::DataCopy(lt_z[8192], gt_sym[b_i * stride], static_cast<int32_t>(4u * stride));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    AscendC::LocalTensor<float> u0 = lt_z[8192u + 0u * stride], u1 = lt_z[8192u + 1u * stride], u2 = lt_z[8192u + 2u * stride], u3 = lt_z[8192u + 3u * stride];
    AscendC::Mul<float, true>(lt_z[0],    u0[0], u1[0], 64ULL, 1, b64_dot);
    AscendC::Mul<float, false>(lt_z[64],  u0[0], u2[0], 64ULL, 1, b64_dot);
    AscendC::Mul<float, false>(lt_z[128], u0[0], u3[0], 64ULL, 1, b64_dot);
    AscendC::Mul<float, false>(lt_z[192], u1[0], u2[0], 64ULL, 1, b64_dot);
    AscendC::Mul<float, false>(lt_z[256], u1[0], u3[0], 64ULL, 1, b64_dot);
    AscendC::Mul<float, false>(lt_z[320], u2[0], u3[0], 64ULL, 1, b64_dot);
    for (uint32_t b = 1u; b < nr2; ++b) {
      uint32_t bo = b * 64u;
      AscendC::PipeBarrier<PIPE_V>();
      if (b < nr0) {
        AscendC::MulAddDst<float, float, false>(lt_z[0],   u0[bo], u1[bo], 64ULL, 1, b64_dot);
        AscendC::MulAddDst<float, float, false>(lt_z[64],  u0[bo], u2[bo], 64ULL, 1, b64_dot);
        AscendC::MulAddDst<float, float, false>(lt_z[128], u0[bo], u3[bo], 64ULL, 1, b64_dot);
      }
      if (b < nr1) {
        AscendC::MulAddDst<float, float, false>(lt_z[192], u1[bo], u2[bo], 64ULL, 1, b64_dot);
        AscendC::MulAddDst<float, float, false>(lt_z[256], u1[bo], u3[bo], 64ULL, 1, b64_dot);
      }
      AscendC::MulAddDst<float, float, false>(lt_z[320], u2[bo], u3[bo], 64ULL, 1, b64_dot);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, false>(lt_stage[600], lt_z[0], 64, 6, 1, 1, 8);
    AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::DataCopy(gt_sym[udot_gm_off + qi * 8u], lt_stage[600], 8);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline uint32_t UnpackAndDotLargeReflectorsAbove64(
    uint32_t p, uint32_t P, uint32_t stride, float anorm, uint32_t& hvec_used,
    const __ubuf__ float* inv_h_keep, __ubuf__ uint32_t* act_uis_all,
    const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::GlobalTensor<float>& gt_sym
  ) {
    uint32_t act_total = 0u;
    if (anorm > 0.0f) {
      for (uint32_t ui = 64u; ui < n_; ++ui) {
        if (inv_h_keep[ui] > 0.0f) {
          uint32_t idx = act_total++;
          act_uis_all[idx] = ui;
          if (p == (ui % P)) {
            uint32_t al_u = (ui + 7u) & ~7u, cnt64_u = (ui + 63u) & ~63u;
            hvec_used -= al_u;
            AscendC::Duplicate(lt_stage[512], 0.0f, static_cast<int32_t>(cnt64_u));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(lt_stage[512], lt_mat[10240u + hvec_used], 0.0f, static_cast<int32_t>(al_u));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(gt_sym[idx * stride], lt_stage[512], static_cast<int32_t>(cnt64_u));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
          }
        }
      }
    }
    SafeSyncAll();

    uint32_t n_quads_all = act_total >> 2u, udot_gm_off = act_total * stride;
    if (n_quads_all > 0u) {
      AscendC::Duplicate(lt_stage[600], 0.0f, 8);
      AscendC::BinaryRepeatParams b64_dot(1, 1, 1, 8, 8, 8);
      for (uint32_t qi = p; qi < n_quads_all; qi += P) {
        EvalQuadReflectorDotAbove64(qi, stride, udot_gm_off, act_uis_all, lt_stage, lt_z, gt_sym, b64_dot);
      }
    }
    SafeSyncAll();
    return act_total;
  }

  __aicore__ __attribute__((always_inline)) inline void ApplyQuadReflectorChunkAbove64(
    uint32_t my_cols, uint32_t stride, uint32_t base_a, uint32_t lq, uint32_t loc_i, uint32_t z_ping,
    const __ubuf__ uint32_t* act_uis_all, const __ubuf__ float* inv_h_keep, const __ubuf__ float* wide_scr,
    const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<uint32_t>& lt_off_bc16_s4, const AscendC::UnaryRepeatParams& s4_u16,
    const AscendC::BinaryRepeatParams& s4_mul_in8, const AscendC::BinaryRepeatParams& s4_mla_up,
    const AscendC::BinaryRepeatParams& s4_mul_up8, const AscendC::BinaryRepeatParams& s4_add_mat_from8
  ) {
    uint32_t qi = (base_a >> 2u) + lq, ui0 = act_uis_all[base_a + loc_i + 0u], ui1 = act_uis_all[base_a + loc_i + 1u], ui2 = act_uis_all[base_a + loc_i + 2u], ui3 = act_uis_all[base_a + loc_i + 3u];
    uint32_t nr0 = ((ui0 + 63u) & ~63u) >> 6u, nr1 = ((ui1 + 63u) & ~63u) >> 6u, nr2 = ((ui2 + 63u) & ~63u) >> 6u, nr3 = ((ui3 + 63u) & ~63u) >> 6u, u_off = 1536u + qi * 8u;
    AscendC::LocalTensor<float> lt_u0 = lt_z[z_ping + (loc_i + 0u) * stride], lt_u1 = lt_z[z_ping + (loc_i + 1u) * stride], lt_u2 = lt_z[z_ping + (loc_i + 2u) * stride], lt_u3 = lt_z[z_ping + (loc_i + 3u) * stride];

    for (uint32_t c_base = 0; c_base < my_cols; c_base += 16u) {
      uint32_t c_lim = (c_base + 16u < my_cols) ? (c_base + 16u) : my_cols;
      int32_t c_cnt = static_cast<int32_t>(c_lim - c_base);
      uint8_t c_cnt_u8 = static_cast<uint8_t>(c_cnt);

      AscendC::Mul<float, true>(lt_z[0],     lt_mat[c_base * stride], lt_u0[0], 64ULL, c_cnt_u8, s4_mul_in8);
      AscendC::Mul<float, false>(lt_z[1024], lt_mat[c_base * stride], lt_u1[0], 64ULL, c_cnt_u8, s4_mul_in8);
      AscendC::Mul<float, false>(lt_z[2048], lt_mat[c_base * stride], lt_u2[0], 64ULL, c_cnt_u8, s4_mul_in8);
      AscendC::Mul<float, false>(lt_z[3072], lt_mat[c_base * stride], lt_u3[0], 64ULL, c_cnt_u8, s4_mul_in8);
      for (uint32_t b = 1u; b < nr3; ++b) {
        AscendC::PipeBarrier<PIPE_V>();
        if (b < nr0) AscendC::MulAddDst<float, float, false>(lt_z[0],    lt_mat[c_base * stride + b * 64u], lt_u0[b * 64u], 64ULL, c_cnt_u8, s4_mul_in8);
        if (b < nr1) AscendC::MulAddDst<float, float, false>(lt_z[1024], lt_mat[c_base * stride + b * 64u], lt_u1[b * 64u], 64ULL, c_cnt_u8, s4_mul_in8);
        if (b < nr2) AscendC::MulAddDst<float, float, false>(lt_z[2048], lt_mat[c_base * stride + b * 64u], lt_u2[b * 64u], 64ULL, c_cnt_u8, s4_mul_in8);
        AscendC::MulAddDst<float, float, false>(lt_z[3072], lt_mat[c_base * stride + b * 64u], lt_u3[b * 64u], 64ULL, c_cnt_u8, s4_mul_in8);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::WholeReduceSum<float, false>(lt_stage[0],  lt_z[0],    64, c_cnt, 1, 1, 8);
      AscendC::WholeReduceSum<float, false>(lt_stage[16], lt_z[1024], 64, c_cnt, 1, 1, 8);
      AscendC::WholeReduceSum<float, false>(lt_stage[32], lt_z[2048], 64, c_cnt, 1, 1, 8);
      AscendC::WholeReduceSum<float, false>(lt_stage[48], lt_z[3072], 64, c_cnt, 1, 1, 8);
      AscendC::PipeBarrier<PIPE_V>();

      SolveQuadReflectorTriangular(
        16ULL, 16u, 128u, static_cast<uint32_t>(c_cnt) * 8u, inv_h_keep[ui0], inv_h_keep[ui1], inv_h_keep[ui2], inv_h_keep[ui3],
        wide_scr[u_off + 0u], wide_scr[u_off + 1u], wide_scr[u_off + 2u], wide_scr[u_off + 3u], wide_scr[u_off + 4u], wide_scr[u_off + 5u],
        lt_stage[0], lt_stage[512], lt_off_bc16_s4, s4_u16
      );
      AscendC::MulAddDst<float, float, true>(lt_mat[c_base * stride], lt_u0[0], lt_stage[512], 64ULL, c_cnt_u8, s4_mla_up);
      for (uint32_t b = 1u; b < nr0; ++b) AscendC::MulAddDst<float, float, false>(lt_mat[c_base * stride + b * 64u], lt_u0[b * 64u], lt_stage[512], 64ULL, c_cnt_u8, s4_mla_up);
      for (uint32_t b = 0u; b < nr3; ++b) AscendC::Mul<float, false>(lt_z[b * 1024u], lt_u3[b * 64u], lt_stage[896], 64ULL, c_cnt_u8, s4_mul_up8);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t b = 0u; b < nr2; ++b) AscendC::MulAddDst<float, float, false>(lt_mat[c_base * stride + b * 64u], lt_u2[b * 64u], lt_stage[768], 64ULL, c_cnt_u8, s4_mla_up);
      for (uint32_t b = 0u; b < nr1; ++b) AscendC::MulAddDst<float, float, false>(lt_z[b * 1024u], lt_u1[b * 64u], lt_stage[640], 64ULL, c_cnt_u8, s4_mul_up8);
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t b = 0u; b < nr3; ++b) AscendC::Add<float, false>(lt_mat[c_base * stride + b * 64u], lt_mat[c_base * stride + b * 64u], lt_z[b * 1024u], 64ULL, c_cnt_u8, s4_add_mat_from8);
      AscendC::ResetMask(); AscendC::PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ __attribute__((always_inline)) inline void ApplySingleReflectorsAbove64(
    uint32_t my_cols, uint32_t stride, uint32_t base_a, uint32_t loc_i, uint32_t cur_a, uint32_t z_ping,
    const __ubuf__ uint32_t* act_uis_all, const __ubuf__ float* inv_h_keep,
    const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<uint32_t>& lt_off_bc16_s4, const AscendC::UnaryRepeatParams& s4_u16,
    const AscendC::BinaryRepeatParams& s4_mul_in8, const AscendC::BinaryRepeatParams& s4_mla_up
  ) {
    for (; loc_i < cur_a; ++loc_i) {
      uint32_t ui = act_uis_all[base_a + loc_i], n_rep64_s4 = ((ui + 63u) & ~63u) >> 6u;
      float inv_h = inv_h_keep[ui];
      AscendC::LocalTensor<float> lt_ui = lt_z[z_ping + loc_i * stride];
      for (uint32_t c_base = 0; c_base < my_cols; c_base += 16u) {
        uint32_t c_lim = (c_base + 16u < my_cols) ? (c_base + 16u) : my_cols;
        int32_t c_cnt = static_cast<int32_t>(c_lim - c_base);
        uint8_t c_cnt_u8 = static_cast<uint8_t>(c_cnt);
        AscendC::Mul<float, true>(lt_z[0], lt_mat[c_base * stride], lt_ui[0], 64ULL, c_cnt_u8, s4_mul_in8);
        for (uint32_t b = 1u; b < n_rep64_s4; ++b) { AscendC::PipeBarrier<PIPE_V>(); AscendC::MulAddDst<float, float, false>(lt_z[0], lt_mat[c_base * stride + b * 64u], lt_ui[b * 64u], 64ULL, c_cnt_u8, s4_mul_in8); }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float, false>(lt_stage[0], lt_z[0], 64, c_cnt, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls<float, true>(lt_stage[0], lt_stage[0], -inv_h, 16ULL, 1, s4_u16);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(lt_stage[256], lt_stage[0], lt_off_bc16_s4, 0u, static_cast<uint32_t>(c_cnt) * 8u);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst<float, float, true>(lt_mat[c_base * stride], lt_ui[0], lt_stage[256], 64ULL, c_cnt_u8, s4_mla_up);
        for (uint32_t b = 1u; b < n_rep64_s4; ++b) AscendC::MulAddDst<float, float, false>(lt_mat[c_base * stride + b * 64u], lt_ui[b * 64u], lt_stage[256], 64ULL, c_cnt_u8, s4_mla_up);
        AscendC::ResetMask(); AscendC::PipeBarrier<PIPE_V>();
      }
    }
  }

  __aicore__ __attribute__((always_inline)) inline void BacktransformLargeAbove64(
    uint32_t my_cols, uint32_t stride, uint32_t act_total, const __ubuf__ uint32_t* act_uis_all, const __ubuf__ float* inv_h_keep, __ubuf__ float* wide_scr,
    const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_off_bc16_s4, const AscendC::GlobalTensor<float>& gt_sym,
    const AscendC::UnaryRepeatParams& s4_u16, const AscendC::BinaryRepeatParams& s4_mul_in8,
    const AscendC::BinaryRepeatParams& s4_mla_up, const AscendC::BinaryRepeatParams& s4_mul_up8, const AscendC::BinaryRepeatParams& s4_add_mat_from8
  ) {
    if (act_total == 0u || my_cols == 0u) return;
    uint32_t n_quads_all = act_total >> 2u, udot_gm_off = act_total * stride, n_chunks = (act_total + 7u) >> 3u;
    if (n_quads_all > 0u) {
      AscendC::DataCopy(lt_wscr[1536], gt_sym[udot_gm_off], static_cast<int32_t>(n_quads_all * 8u));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    }
    AscendC::DataCopy(lt_z[8192], gt_sym[0], static_cast<int32_t>(((8u <= act_total) ? 8u : act_total) * stride));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    if (n_chunks >= 2u) {
      AscendC::DataCopy(lt_z[12288], gt_sym[8u * stride], static_cast<int32_t>(((16u <= act_total) ? 8u : (act_total - 8u)) * stride));
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    }
    AscendC::Duplicate<float, true>(lt_stage[0], 0.0f, 64ULL, 1, 1, 8);
    if (n_quads_all > 0u) AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    for (uint32_t chk = 0u; chk < n_chunks; ++chk) {
      uint32_t base_a = chk * 8u, cur_a = (act_total - base_a >= 8u) ? 8u : (act_total - base_a), z_ping = (chk & 1u) ? 12288u : 8192u, local_q = cur_a >> 2u, loc_i = 0u;
      event_t cur_ev = (chk & 1u) ? EVENT_ID1 : EVENT_ID0;
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(cur_ev);
      for (uint32_t lq = 0u; lq < local_q; ++lq, loc_i += 4u) {
        ApplyQuadReflectorChunkAbove64(my_cols, stride, base_a, lq, loc_i, z_ping, act_uis_all, inv_h_keep, wide_scr, lt_stage, lt_mat, lt_z, lt_off_bc16_s4, s4_u16, s4_mul_in8, s4_mla_up, s4_mul_up8, s4_add_mat_from8);
      }
      ApplySingleReflectorsAbove64(my_cols, stride, base_a, loc_i, cur_a, z_ping, act_uis_all, inv_h_keep, lt_stage, lt_mat, lt_z, lt_off_bc16_s4, s4_u16, s4_mul_in8, s4_mla_up);
      if (chk + 2u < n_chunks) {
        uint32_t next_a = (chk + 2u) * 8u, cnt_n = (next_a + 8u <= act_total) ? 8u : (act_total - next_a);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(cur_ev); AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(cur_ev);
        AscendC::DataCopy(lt_z[z_ping], gt_sym[next_a * stride], static_cast<int32_t>(cnt_n * stride));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(cur_ev);
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    if (n_chunks >= 2u) { AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1); AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1); }
  }

  __aicore__ __attribute__((always_inline)) inline void TransposeAndWriteBackLargeVec(
    uint32_t p, uint32_t P, bool write_out, __gm__ float* cur_vec, __ubuf__ uint32_t* off_pfull,
    const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<int32_t>& lt_off_pf_i, const AscendC::LocalTensor<uint32_t>& lt_off_pfull,
    const AscendC::GlobalTensor<float>& gt_sym
  ) {
    uint32_t tot_b8_out = n_ / 8u, r_lo_out = ((p * tot_b8_out) / P) * 8u, r_hi_out = (((p + 1u) * tot_b8_out) / P) * 8u, r_cnt_out = r_hi_out - r_lo_out;
    if (write_out && r_cnt_out > 0u) {
      AscendC::DataCopyParams cp_qt_in(static_cast<uint16_t>(n_), static_cast<uint16_t>(r_cnt_out / 8u), static_cast<uint16_t>((n_ - r_cnt_out) / 8u), 0);
      AscendC::DataCopy(lt_mat, gt_sym[r_lo_out], cp_qt_in);
      for (uint32_t k = 0; k < n_; ++k) off_pfull[k] = (k * r_cnt_out) * 4u;
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      for (uint32_t dr = 0; dr < r_cnt_out; ++dr) {
        if (dr > 0u) { AscendC::Adds(lt_off_pf_i, lt_off_pf_i, 4, static_cast<int32_t>(n_)); AscendC::PipeBarrier<PIPE_V>(); }
        AscendC::Gather(lt_z[dr * n_], lt_mat, lt_off_pfull, 0u, n_);
      }
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    }
    SafeSyncAll();
    if (write_out && r_cnt_out > 0u) {
      AscendC::GlobalTensor<float> gt_vec_out;
      gt_vec_out.SetGlobalBuffer(cur_vec, n_ * n_);
      AscendC::DataCopy(gt_vec_out[r_lo_out * n_], lt_z, static_cast<int32_t>(r_cnt_out * n_));
    }
  }

  __aicore__ __attribute__((always_inline)) inline void FixSignAndWriteBackLarge(
    uint32_t p, uint32_t P, uint32_t col_start, uint32_t my_cols, uint32_t stride, bool write_out,
    __gm__ float* cur_val, __gm__ float* cur_vec, __ubuf__ uint32_t* off_pfull, __ubuf__ float* stage_ub, __ubuf__ float* mat_ub,
    const AscendC::LocalTensor<float>& lt_d, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<int32_t>& lt_off_pf_i, const AscendC::LocalTensor<uint32_t>& lt_off_pfull,
    const AscendC::GlobalTensor<float>& gt_sym
  ) {
    if (write_out && p == 0u) {
      AscendC::GlobalTensor<float> gt_val;
      gt_val.SetGlobalBuffer(cur_val, n_);
      AscendC::DataCopyExtParams cv(1, static_cast<uint32_t>(n_ * sizeof(float)), 0u, 0u, 0u);
      AscendC::DataCopyPad(gt_val, lt_d, cv);
    }
    SafeSyncAll();

    if (my_cols > 0u) {
      uint8_t mc_rb_u8 = static_cast<uint8_t>(my_cols);
      int32_t mc_rb_i32 = static_cast<int32_t>(my_cols);
      AscendC::BinaryRepeatParams b_rb_step1(1, 1, 1, 16, 32, 32), b_rb_step2(1, 1, 1, 16, 16, 16);
      AscendC::Max<float, true>(lt_z[0],      lt_mat[0],  lt_mat[128], 64ULL, mc_rb_u8, b_rb_step1);
      AscendC::Min<float, false>(lt_z[8192u], lt_mat[0],  lt_mat[128], 64ULL, mc_rb_u8, b_rb_step1);
      AscendC::Max<float, false>(lt_z[64u],   lt_mat[64], lt_mat[192], 64ULL, mc_rb_u8, b_rb_step1);
      AscendC::Min<float, false>(lt_z[8256u], lt_mat[64], lt_mat[192], 64ULL, mc_rb_u8, b_rb_step1); AscendC::PipeBarrier<PIPE_V>();
      AscendC::Max<float, false>(lt_z[0],     lt_z[0],     lt_z[64u],   64ULL, mc_rb_u8, b_rb_step2);
      AscendC::Min<float, false>(lt_z[8192u], lt_z[8192u], lt_z[8256u], 64ULL, mc_rb_u8, b_rb_step2); AscendC::PipeBarrier<PIPE_V>();
      AscendC::WholeReduceMax<float, false>(lt_stage[0],  lt_z[0],     64, mc_rb_i32, 1, 1, 16, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
      AscendC::WholeReduceMin<float, false>(lt_stage[32], lt_z[8192u], 64, mc_rb_i32, 1, 1, 16, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
      if constexpr (n_ != 256u) AscendC::ResetMask();
      AscendC::PipeBarrier<PIPE_V>(); AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

      for (uint32_t c = 0; c < my_cols; ++c) {
        float mx = stage_ub[c], neg_mn = -stage_ub[32u + c];
        bool neg_col = false;
        if (neg_mn > mx) neg_col = true;
        else if (mx > neg_mn) neg_col = false;
        else {
          if constexpr (n_ == 256u) {
            EIGH_PAD256() EIGH_PAD256() EIGH_PAD256() EIGH_PAD256() EIGH_PAD256() EIGH_PAD256() EIGH_PAD64() EIGH_PAD64() EIGH_PAD64() EIGH_PAD64() EIGH_PAD64() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4()
          } else {
            EIGH_PAD256() EIGH_PAD256() EIGH_PAD256() EIGH_PAD256() EIGH_PAD256() EIGH_PAD64() EIGH_PAD64() EIGH_PAD64() EIGH_PAD16() EIGH_PAD16() EIGH_PAD16() EIGH_PAD16() EIGH_PAD16() EIGH_PAD16() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4()
          }
          neg_col = ScanColSignTieBreak(mat_ub + c * stride);
        }
        if (neg_col) AscendC::Muls(lt_mat[c * stride], lt_mat[c * stride], -1.0f, static_cast<int32_t>(n_));
      }
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      if (write_out) {
        AscendC::DataCopyParams cp_qt2gm(static_cast<uint16_t>(my_cols), static_cast<uint16_t>(n_ / 8u), static_cast<uint16_t>((stride - n_) / 8u), 0);
        AscendC::DataCopy(gt_sym[col_start * n_], lt_mat, cp_qt2gm);
      }
    }
    SafeSyncAll();
    TransposeAndWriteBackLargeVec(p, P, write_out, cur_vec, off_pfull, lt_mat, lt_z, lt_off_pf_i, lt_off_pfull, gt_sym);
  }

  __aicore__ __attribute__((always_inline)) inline void RunLargeStage4Backtransform(
    uint32_t p, uint32_t P, uint32_t my_cols, uint32_t stride, float anorm, uint32_t& hvec_used,
    __ubuf__ float* wide_scr, __ubuf__ float* stage_ub, __ubuf__ float* mat_ub,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<float>& lt_stage,
    const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_u32,
    const AscendC::GlobalTensor<float>& gt_sym
  ) {
    int32_t str_blk_s4 = static_cast<int32_t>(stride >> 3u);
    uint8_t str_blk_s4_u8 = static_cast<uint8_t>(str_blk_s4);
    AscendC::BinaryRepeatParams s4_mul_in(1, 1, 1, str_blk_s4_u8, str_blk_s4_u8, 0), s4_mul_in8(1, 1, 1, 8, str_blk_s4_u8, 0), s4_mla_up(1, 1, 0, str_blk_s4_u8, 0, 1), s4_mul_up8(1, 1, 0, 8, 0, 1), s4_add_mat_from8(1, 1, 1, str_blk_s4_u8, str_blk_s4_u8, 8);
    AscendC::UnaryRepeatParams s4_u16(1, 1, 8, 8);
    AscendC::LocalTensor<uint32_t> lt_off_bc16_s4 = lt_u32[7168 + 4608];
    __ubuf__ float* inv_h_keep = mat_ub + 9728;
    __ubuf__ uint32_t* act_uis_all = reinterpret_cast<__ubuf__ uint32_t*>(wide_scr + 2560);

    BacktransformLargeTail64(p, P, my_cols, stride, str_blk_s4, anorm, hvec_used, inv_h_keep, stage_ub, ub_all, lt_stage, lt_mat, lt_z, lt_off_bc16_s4, gt_sym, s4_u16, s4_mul_in, s4_mul_in8, s4_mla_up, s4_mul_up8, s4_add_mat_from8);
    uint32_t act_total = UnpackAndDotLargeReflectorsAbove64(p, P, stride, anorm, hvec_used, inv_h_keep, act_uis_all, lt_stage, lt_mat, lt_z, gt_sym);
    BacktransformLargeAbove64(my_cols, stride, act_total, act_uis_all, inv_h_keep, wide_scr, lt_wscr, lt_stage, lt_mat, lt_z, lt_off_bc16_s4, gt_sym, s4_u16, s4_mul_in8, s4_mla_up, s4_mul_up8, s4_add_mat_from8);
  }

  __aicore__ __attribute__((always_inline)) inline void RunLargeStage3Merges(
    uint32_t p, uint32_t P, uint32_t col_start, uint32_t col_end, uint32_t my_cols, uint32_t stride,
    const __ubuf__ int32_t* bounds, __ubuf__ int32_t* K_arr, __ubuf__ int32_t* D_arr, __ubuf__ int32_t* triv_arr, __ubuf__ float* rho_arr,
    __ubuf__ float* d, const __ubuf__ float* e, __ubuf__ float* top, __ubuf__ float* bot, __ubuf__ float* ths, __ubuf__ float* bhs,
    __ubuf__ float* ad, __ubuf__ float* az, __ubuf__ float* dd, __ubuf__ float* tau_a, __ubuf__ int32_t* orig_ord, __ubuf__ int32_t* acol_dc,
    __ubuf__ uint32_t* off_pfull, __ubuf__ float* wide_scr, __ubuf__ float* stage_ub, __ubuf__ float* mat_ub, __ubuf__ float* z_ub,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<float>& lt_stage,
    const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<uint32_t>& lt_u32,
    const AscendC::LocalTensor<uint32_t>& lt_off_tr64x16, const AscendC::LocalTensor<uint32_t>& lt_off_pfull,
    const AscendC::GlobalTensor<float>& gt_ut_m, const AscendC::GlobalTensor<float>& gt_roots, const AscendC::GlobalTensor<float>& gt_zhat, const AscendC::GlobalTensor<float>& gt_bnd
  ) {
    constexpr int32_t M = EighHostPlan<CASE_KEY>::value.M;
    AscendC::UnaryRepeatParams u_def64_s3(1, 1, 8, 8), u_def16_s3(1, 1, 2, 2);
    AscendC::BinaryRepeatParams b_def64_s3(1, 1, 1, 8, 8, 8), b_def16_s3(1, 1, 1, 2, 2, 2);
    for (int32_t step = 1; step < M; step *= 2) {
      bool final_level = (step * 2 >= M);
      uint32_t num_merges = static_cast<uint32_t>(M / (2 * step)), my_m = 0u, my_plo = 0u, sub_p = 0u, sub_P = P;
      for (uint32_t m_chk = 0; m_chk < num_merges; ++m_chk) {
        uint32_t plo_c = (m_chk * P) / num_merges, phi_c = ((m_chk + 1u) * P) / num_merges;
        if (p >= plo_c && p < phi_c) { my_m = m_chk; my_plo = plo_c; sub_p = p - plo_c; sub_P = phi_c - plo_c; break; }
      }
      for (uint32_t m = 0; m < num_merges; ++m) {
        DeflateLargeSingleMerge(m, step, final_level, col_start, col_end, stride, bounds, d, e, top, bot, ths, bhs, ad, az, dd, orig_ord, acol_dc, K_arr, D_arr, triv_arr, rho_arr, mat_ub);
      }
      int32_t s_my = bounds[static_cast<int32_t>(my_m) * 2 * step], K_my = K_arr[my_m];
      float rho_my = rho_arr[my_m];
      uint32_t u_cap_my = (K_my > 0) ? ((static_cast<uint32_t>(K_my) + sub_P - 1u) / sub_P) : 1u, i0 = sub_p * u_cap_my;
      uint32_t i1 = (K_my > 0 && i0 < static_cast<uint32_t>(K_my)) ? ((i0 + u_cap_my < static_cast<uint32_t>(K_my)) ? (i0 + u_cap_my) : static_cast<uint32_t>(K_my)) : i0;
      uint32_t ut_stride = final_level ? n_ : (((n_ >> 1u) + 15u) & ~7u);

      SolveLargeRootsAndSync(p, P, step, num_merges, s_my, K_my, rho_my, i0, i1, bounds, K_arr, ad, az, orig_ord, tau_a, wide_scr, stage_ub, z_ub, lt_wscr, lt_stage, lt_z, gt_roots, u_def64_s3, b_def64_s3, u_def16_s3, b_def16_s3);
      EvalLargeGuEisenstatAndSync(p, my_plo, sub_P, u_cap_my, s_my, K_my, rho_my, i0, i1, orig_ord, ad, az, tau_a, wide_scr, stage_ub, z_ub, ub_all, lt_wscr, lt_stage, lt_z, lt_u32, gt_zhat);
      EvalLargeP3AndSyncSummaries(p, P, step, final_level, num_merges, s_my, K_my, i0, i1, ut_stride, bounds, K_arr, D_arr, triv_arr, ad, az, ths, bhs, tau_a, acol_dc, orig_ord, dd, d, top, bot, off_pfull, wide_scr, stage_ub, z_ub, ub_all, lt_wscr, lt_stage, lt_z, lt_u32, gt_ut_m, gt_bnd);
      UpdateLargeP4AllMerges(step, num_merges, col_start, col_end, my_cols, stride, ut_stride, bounds, K_arr, D_arr, triv_arr, orig_ord, acol_dc, off_pfull, wide_scr, stage_ub, mat_ub, z_ub, ub_all, lt_stage, lt_mat, lt_wscr, lt_z, lt_u32, lt_off_tr64x16, lt_off_pfull, gt_ut_m);
      SafeSyncAll();
    }
  }

  __aicore__ __attribute__((noinline)) void SolveLargeCoopStedc(
    __gm__ float* cur_A, __gm__ float* cur_val, __gm__ float* cur_vec, __gm__ float* group_ws,
    const AscendC::LocalTensor<float>& ub_all, uint32_t p, uint32_t P, bool write_out
  ) {
    (void)group_ws;
    const __gm__ EighCorePlan &cp = Plan().cores[AscendC::GetBlockIdx()];
    constexpr uint32_t stride = EighHostPlan<CASE_KEY>::value.stride, max_rows = EighHostPlan<CASE_KEY>::value.max_rows;
    constexpr uint32_t gm_stride = EighCase<CASE_KEY>::N, sym_cap = EighCase<CASE_KEY>::N * EighCase<CASE_KEY>::N;
    constexpr uint32_t p_stride = EighHostPlan<CASE_KEY>::value.p_stride, pp_slot = EighHostPlan<CASE_KEY>::value.pp_slot;
    constexpr uint32_t meta_slot = EighHostPlan<CASE_KEY>::value.meta_slot, meta_cap = EighHostPlan<CASE_KEY>::value.meta_cap;
    uint32_t col_start = cp.col_start, col_end = cp.col_end, my_cols = cp.my_cols, my_blk = cp.my_blk, plo_b = cp.plo_b, bp_team = cp.bp_team, subP_b = cp.subP_b;

    __ubuf__ float* base_ptr = reinterpret_cast<__ubuf__ float*>(ub_all.GetPhyAddr());
    __ubuf__ float* d = base_ptr, *e = base_ptr + 512, *u_ub = base_ptr + 1024, *w_ub = base_ptr + 1536, *p_full = base_ptr + 2048;
    __ubuf__ float* inv_h_save = base_ptr + 2560, *meta_ub = base_ptr + 3072, *ths = base_ptr + 3584, *bhs = base_ptr + 4096, *tau_a = base_ptr + 4608;
    __ubuf__ int32_t* orig_ord = reinterpret_cast<__ubuf__ int32_t*>(base_ptr + 5120), *acol_dc = reinterpret_cast<__ubuf__ int32_t*>(base_ptr + 5632);
    __ubuf__ float* red_ub = base_ptr + 6144, *wide_scr = base_ptr + 7168, *stage_ub = base_ptr + 13824, *mat_ub = base_ptr + 15872, *z_ub = base_ptr + 32256;
    __ubuf__ uint32_t* off_tr64x16 = reinterpret_cast<__ubuf__ uint32_t*>(base_ptr + 12288), *off_pfull = reinterpret_cast<__ubuf__ uint32_t*>(base_ptr + 13312);

    AscendC::LocalTensor<float> lt_d = ub_all[0], lt_e = ub_all[512], lt_u = ub_all[1024], lt_w = ub_all[1536], lt_pfull = ub_all[2048], lt_meta = ub_all[3072];
    AscendC::LocalTensor<float> lt_red = ub_all[6144], lt_work = ub_all[6656], lt_wscr = ub_all[7168], lt_stage = ub_all[13824], lt_mat = ub_all[15872], lt_z = ub_all[32256];
    AscendC::LocalTensor<uint32_t> lt_u32 = ub_all.ReinterpretCast<uint32_t>(), lt_off_tr64x16 = lt_u32[12288], lt_off_pfull = lt_u32[13312];
    AscendC::LocalTensor<int32_t> lt_tr64x16_i = ub_all.ReinterpretCast<int32_t>()[12288], lt_off_pf_i = ub_all.ReinterpretCast<int32_t>()[13312];

    AscendC::GlobalTensor<float> gt_A, gt_sym, gt_meta, gt_anorm, gt_ut_m, gt_roots, gt_zhat, gt_bnd;
    gt_A.SetGlobalBuffer(cur_A, sym_cap); gt_sym.SetGlobalBuffer(cur_vec, sym_cap); gt_meta.SetGlobalBuffer(cur_vec, meta_cap); gt_anorm.SetGlobalBuffer(cur_val, n_);
    gt_ut_m.SetGlobalBuffer(cur_vec, sym_cap); gt_roots.SetGlobalBuffer(cur_vec, 3072u); gt_zhat.SetGlobalBuffer(cur_vec + 3072, 2048u); gt_bnd.SetGlobalBuffer(cur_vec + sym_cap - 3072, 3072u);

    float scale_pow2 = 1.0f, inv_scale_pow2 = 1.0f, anorm = 0.0f;
    uint32_t hvec_used = 0u;
    InitLargeSymTables(p, P, stride, p_stride, off_tr64x16, off_pfull, reinterpret_cast<__ubuf__ uint32_t*>(wide_scr + 4608), reinterpret_cast<__ubuf__ uint32_t*>(wide_scr + 4864), reinterpret_cast<__ubuf__ uint32_t*>(z_ub), lt_tr64x16_i, lt_z);
    SymmetrizeAndLoadLargePanel(p, P, stride, max_rows, gm_stride, anorm, scale_pow2, inv_scale_pow2, stage_ub, meta_ub, red_ub, z_ub, lt_stage, lt_mat, lt_z, lt_meta, lt_red, lt_work, lt_u32[32256], gt_A, gt_sym, gt_anorm);

    inv_h_save[0] = 0.0f;
    if (anorm == 0.0f) {
      EIGH_PAD256() EIGH_PAD64() EIGH_PAD16() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4() EIGH_PAD16() EIGH_PAD4() EIGH_PAD4()
      int32_t str_i32 = static_cast<int32_t>(stride);
      AscendC::Duplicate(lt_d, 0.0f, str_i32); AscendC::Duplicate(lt_e, 0.0f, str_i32); AscendC::Duplicate(ub_all[2560], 0.0f, str_i32); AscendC::Duplicate(lt_mat[9728], 0.0f, 512);
      AscendC::PipeBarrier<PIPE_V>();
      for (int32_t i = static_cast<int32_t>(n_) - 1; i >= 128; --i) { uint16_t fid = static_cast<uint16_t>(static_cast<uint32_t>(i) & 3u); AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(fid); AscendC::CrossCoreWaitFlag(fid); }
      SafeSyncAll(); SafeSyncAll();
    } else {
      TridiagLargeCoopAbove128(p, P, stride, p_stride, pp_slot, meta_slot, hvec_used, d, e, u_ub, p_full, inv_h_save, red_ub, wide_scr, stage_ub, lt_u, lt_w, lt_pfull, lt_stage, lt_mat, lt_z, lt_red, lt_work, lt_wscr, lt_off_pfull, lt_u32[7168 + 4864], gt_meta);
      TridiagLargeTail128And64(p, P, stride, max_rows, hvec_used, d, e, u_ub, inv_h_save, red_ub, wide_scr, z_ub, ub_all, lt_u, lt_pfull, lt_stage, lt_mat, lt_z, lt_red, lt_wscr, lt_u32, gt_sym);
    }

    __ubuf__ int32_t* bounds = reinterpret_cast<__ubuf__ int32_t*>(red_ub + 256), *K_arr = reinterpret_cast<__ubuf__ int32_t*>(red_ub + 320), *D_arr = reinterpret_cast<__ubuf__ int32_t*>(red_ub + 336), *triv_arr = reinterpret_cast<__ubuf__ int32_t*>(red_ub + 352);
    SolveLargeLeavesAndBlocks64(p, P, my_blk, plo_b, bp_team, subP_b, sym_cap, cur_vec, bounds, d, e, stage_ub, ub_all, lt_stage);
    UnpackLargeBlock64ToCols(col_start, col_end, my_cols, stride, bounds, d, u_ub, w_ub, off_tr64x16, off_pfull, z_ub, ub_all, lt_mat, lt_z, lt_tr64x16_i, lt_off_pfull, gt_sym);
    RunLargeStage3Merges(p, P, col_start, col_end, my_cols, stride, bounds, K_arr, D_arr, triv_arr, red_ub + 368, d, e, u_ub, w_ub, ths, bhs, p_full, inv_h_save, meta_ub, tau_a, orig_ord, acol_dc, off_pfull, wide_scr, stage_ub, mat_ub, z_ub, ub_all, lt_wscr, lt_stage, lt_mat, lt_z, lt_u32, lt_off_tr64x16, lt_off_pfull, gt_ut_m, gt_roots, gt_zhat, gt_bnd);
    SortAndTransposeLargeStedcToQT(col_start, col_end, my_cols, stride, scale_pow2, d, off_pfull, reinterpret_cast<__ubuf__ uint32_t*>(wide_scr + 4608), mat_ub, lt_d, lt_mat, lt_z, lt_off_pf_i, lt_off_pfull, gt_sym);
    RunLargeStage4Backtransform(p, P, my_cols, stride, anorm, hvec_used, wide_scr, stage_ub, mat_ub, ub_all, lt_wscr, lt_stage, lt_mat, lt_z, lt_u32, gt_sym);
    FixSignAndWriteBackLarge(p, P, col_start, my_cols, stride, write_out, cur_val, cur_vec, off_pfull, stage_ub, mat_ub, lt_d, lt_stage, lt_mat, lt_z, lt_off_pf_i, lt_off_pfull, gt_sym);
  }
