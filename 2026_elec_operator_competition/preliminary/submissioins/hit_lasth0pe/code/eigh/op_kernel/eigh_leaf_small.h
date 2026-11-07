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

  __aicore__ __attribute__((always_inline)) inline void InitLeafStage(
    int32_t m, float* d, float* e,
    const __ubuf__ float* d_in, const __ubuf__ float* e_in,
    const AscendC::LocalTensor<float>& lt_stage,
    const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_t1,
    __ubuf__ float* z_p,
    const AscendC::UnaryRepeatParams& leaf_u_rep1
  ) {
    for (int32_t i = 0; i < m; ++i) {
      d[i] = d_in[i];
    }
    for (int32_t i = 0; i < m - 1; ++i) {
      e[i] = e_in[i];
    }
    e[m - 1] = 0.0f;
    AscendC::Duplicate(lt_stage, 0.0f, 320);
    AscendC::Duplicate(lt_z, 0.0f, 256);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    for (int32_t i = 0; i < m; ++i) {
      z_p[i * 17] = 1.0f;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    if constexpr (n_ == 64u) {
      AscendC::SetVectorMask<float>(0ULL, 0xFFFFULL);
    } else {
      AscendC::Adds<float, true>(lt_t1, lt_t1, 0.0f, 16ULL, 1, leaf_u_rep1);
      AscendC::PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ __attribute__((always_inline)) inline float ComputeWilkinsonShift(
    float dl, float dl1, float el, float dm
  ) {
    float g = (dl1 - dl) / (2.0f * el);
    float ag = MyAbs(g);
    float r;
    if (ag > 1e4f) {
      r = ag;
    } else if (ag <= 1.0f) {
      float tg = 1.0f + g * g;
      r = tg * RsqrtUnit12(tg);
    } else {
      float ig = 1.0f / ag;
      float tg = 1.0f + ig * ig;
      r = ag * (tg * RsqrtUnit12(tg));
    }
    float sv = (g >= 0.0f) ? (g + r) : (g - r);
    return (dm - dl) + (el / sv);
  }

  template <bool PAD_N64>
  __aicore__ __attribute__((always_inline)) inline bool StepTqlPlaneRot(
    float f, float g, float dip1, float p,
    float& ei1_out, float& em_out, float& di1_out, float& s, float& c
  ) {
    float r2 = f * f + g * g;
    if (r2 > 1e-24f && r2 < 1e24f) {
      float inv_r = MyRsqrt(r2);
      ei1_out = r2 * inv_r;
      s = f * inv_r;
      c = g * inv_r;
      return false;
    }
    float af = MyAbs(f), ag_i = MyAbs(g);
    if (af == 0.0f && ag_i == 0.0f) {
      if constexpr (PAD_N64 && n_ == 64u) {
        EIGH_PAD16() EIGH_PAD16() EIGH_PAD16()
      }
      di1_out = dip1 - p;
      em_out = 0.0f;
      return true;
    }
    bool f_ge_g = (af >= ag_i);
    float num_g = f_ge_g ? g : f, den_g = f_ge_g ? f : g;
    float q_g = num_g / den_g;
    float t = 1.0f + q_g * q_g;
    float inv_r = RsqrtUnit12(t), q_inv = q_g * inv_r;
    ei1_out = den_g * (t * inv_r);
    s = f_ge_g ? inv_r : q_inv;
    c = f_ge_g ? q_inv : inv_r;
    return false;
  }

  template <bool PAD_N64>
  __aicore__ __attribute__((always_inline)) inline bool RunTqlSweep16(
    int32_t l, int32_t mm, float* d, float* e, float& g,
    const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_t1,
    const AscendC::LocalTensor<float>& lt_t2,
    const AscendC::UnaryRepeatParams& u_rep2,
    const AscendC::BinaryRepeatParams& b_rep1
  ) {
    float s = 1.0f, c = 1.0f, p = 0.0f, dip1 = d[mm];
    for (int32_t i = mm - 1; i >= l; --i) {
      float ei = e[i], di = d[i], f = s * ei, b = c * ei;
      if (StepTqlPlaneRot<PAD_N64>(f, g, dip1, p, e[i + 1], e[mm], d[i + 1], s, c)) {
        return true;
      }
      uint32_t row_off = static_cast<uint32_t>(i) * 16u;
      AscendC::Muls<float, false>(lt_t1, lt_z[row_off], s, 16ULL, 2, u_rep2);
      AscendC::Muls<float, false>(lt_t2, lt_z[row_off], c, 16ULL, 2, u_rep2);
      g = dip1 - p;
      float r_rot = (di - g) * s + 2.0f * c * b;
      p = s * r_rot;
      d[i + 1] = g + p;
      g = c * r_rot - b;
      dip1 = di;
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub<float, false>(lt_z[row_off],       lt_t2,     lt_t1[16], 16ULL, 1, b_rep1);
      AscendC::Add<float, false>(lt_z[row_off + 16u], lt_t2[16], lt_t1,     16ULL, 1, b_rep1);
      AscendC::PipeBarrier<PIPE_V>();
    }
    d[l] = dip1 - p;
    e[l] = g;
    e[mm] = 0.0f;
    return false;
  }

  __aicore__ __attribute__((always_inline)) inline int32_t FindTqlSplit16(
    int32_t l, int32_t m_lim, float tol_rel, const float* d, float* e
  ) {
    int32_t mm = l;
    while (mm + 1 < m_lim) {
      float dd = MyAbs(d[mm]) + MyAbs(d[mm + 1]);
      float ae = MyAbs(e[mm]);
      bool split = (ae <= tol_rel * dd) ? true : (ae <= 1e-24f);
      if (split) {
        e[mm] = 0.0f;
        break;
      }
      mm++;
    }
    return mm;
  }

  template <bool PAD_N64>
  __aicore__ __attribute__((always_inline)) inline void RunTqlLoop16(
    int32_t m_lim, int32_t max_iter, float tol_rel, float* d, float* e,
    const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_t1,
    const AscendC::LocalTensor<float>& lt_t2,
    const AscendC::UnaryRepeatParams& u_rep2,
    const AscendC::BinaryRepeatParams& b_rep1
  ) {
    for (int32_t l = 0; l < m_lim; ++l) {
      int32_t iter = 0;
      while (true) {
        int32_t mm = FindTqlSplit16(l, m_lim, tol_rel, d, e);
        if (mm == l || ++iter > max_iter) {
          break;
        }
        float g = ComputeWilkinsonShift(d[l], d[l + 1], e[l], d[mm]);
        RunTqlSweep16<PAD_N64>(l, mm, d, e, g, lt_z, lt_t1, lt_t2, u_rep2, b_rep1);
      }
    }
  }

  __aicore__ __attribute__((always_inline)) inline void PackLeafStageOutput(
    int32_t m, int32_t* idx, const float* d, const __ubuf__ float* z_p,
    const AscendC::LocalTensor<float>& lt_stage,
    const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_t1,
    __ubuf__ float* stage_ub,
    const AscendC::UnaryRepeatParams& leaf_u_rep1
  ) {
    if constexpr (n_ != 64u) {
      AscendC::ResetMask();
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    SortIdxF32Local(m, idx, d);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    if constexpr (n_ == 64u) {
      if (m <= 0) {
        EIGH_PAD16() EIGH_PAD4() EIGH_PAD4()
      }
    } else {
      AscendC::Adds<float, true>(lt_t1, lt_t1, 0.0f, 16ULL, 1, leaf_u_rep1);
      AscendC::PipeBarrier<PIPE_V>();
    }
    for (int32_t col = 0; col < m; ++col) {
      int32_t s_idx = idx[col];
      stage_ub[0 + col]  = d[s_idx];
      stage_ub[16 + col] = z_p[s_idx * 16 + 0];
      stage_ub[32 + col] = z_p[s_idx * 16 + (m - 1)];
      AscendC::Adds<float, false>(
        lt_stage[64u + static_cast<uint32_t>(col) * 16u],
        lt_z[static_cast<uint32_t>(s_idx) * 16u], 0.0f, 16ULL, 1, leaf_u_rep1
      );
    }
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
  }

  __aicore__ __attribute__((noinline)) void SolveLeafPipeV16(
    int32_t m,
    const __ubuf__ float* d_in,
    const __ubuf__ float* e_in,
    const AscendC::LocalTensor<float>& lt_stage,
    __ubuf__ float* stage_ub
  ) {
    float d[16], e[16];
    int32_t idx[16];
    AscendC::LocalTensor<float> lt_t1 = lt_stage[320];
    AscendC::LocalTensor<float> lt_t2 = lt_stage[352];
    AscendC::LocalTensor<float> lt_z  = lt_stage[512];
    __ubuf__ float* z_p = stage_ub + 512;
    AscendC::UnaryRepeatParams leaf_u_rep1(1, 1, 8, 8);
    AscendC::UnaryRepeatParams leaf_u_rep2(1, 1, 2, 2);
    AscendC::BinaryRepeatParams leaf_b_rep1(1, 1, 1, 8, 8, 8);

    InitLeafStage(m, d, e, d_in, e_in, lt_stage, lt_z, lt_t1, z_p, leaf_u_rep1);
    RunTqlLoop16<true>(m, 40, 2.0e-6f, d, e, lt_z, lt_t1, lt_t2, leaf_u_rep2, leaf_b_rep1);
    PackLeafStageOutput(m, idx, d, z_p, lt_stage, lt_z, lt_t1, stage_ub, leaf_u_rep1);
  }

  __aicore__ __attribute__((always_inline)) inline void SolveSmallN1(
    const AscendC::GlobalTensor<float>& gt_A,
    const AscendC::GlobalTensor<float>& gt_val,
    const AscendC::GlobalTensor<float>& gt_vec,
    const AscendC::LocalTensor<float>& lt_val,
    const AscendC::LocalTensor<float>& lt_z,
    __ubuf__ float* z_ub
  ) {
    AscendC::DataCopyExtParams c1(1, sizeof(float), 0u, 0u, 0u);
    AscendC::DataCopyPadExtParams<float> p1(true, 0, 7, 0.0f);
    AscendC::DataCopyPad(lt_val, gt_A, c1, p1);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    z_ub[0] = 1.0f;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyPad(gt_val, lt_val, c1);
    AscendC::DataCopyPad(gt_vec, lt_z, c1);
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  __aicore__ __attribute__((always_inline)) inline void InitSmallMasksVec(
    uint32_t row_al,
    const AscendC::LocalTensor<float>& lt_umat,
    const AscendC::LocalTensor<int32_t>& lt_msk_i32,
    const AscendC::LocalTensor<int32_t>& lt_dmsk_i32,
    const AscendC::LocalTensor<float>& lt_mask_row,
    const AscendC::LocalTensor<float>& lt_unit_vec,
    const AscendC::LocalTensor<int32_t>& lt_rep8_i32
  ) {
    AscendC::Duplicate<float, true>(lt_umat, 0.0f, 64ULL, 4, 1, 8);
    AscendC::Duplicate<int32_t, false>(lt_msk_i32, 0, 64ULL, 4, 1, 8);
    AscendC::Duplicate<int32_t, false>(lt_dmsk_i32, -1, 64ULL, 4, 1, 8);
    AscendC::Duplicate<float, false>(lt_mask_row, 0.0f, 64ULL, 4, 1, 8);
    AscendC::Duplicate<float, false>(lt_unit_vec, 0.0f, 64ULL, 4, 1, 8);
    AscendC::Duplicate<int32_t, true>(lt_rep8_i32, 0, 8ULL, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    if (uplo_ == 0u) {
      uint8_t n_lo8 = static_cast<uint8_t>((n_ < 8u) ? n_ : 8u);
      AscendC::Duplicate<int32_t, true>(lt_msk_i32[0], -1, 8ULL, n_lo8, 1, 2);
      if (n_ > 8u) {
        AscendC::Duplicate<int32_t, true>(lt_msk_i32[8u * 16u], -1, 16ULL, static_cast<uint8_t>(n_ - 8u), 1, 2);
      }
    } else {
      uint8_t n_hi0 = static_cast<uint8_t>((n_ < 8u) ? n_ : 8u);
      AscendC::Duplicate<int32_t, true>(lt_msk_i32[0], -1, static_cast<uint64_t>(row_al), n_hi0, 1, 2);
      if (n_ > 8u) {
        AscendC::Duplicate<int32_t, true>(lt_msk_i32[8u * 16u + 8u], -1, 8ULL, static_cast<uint8_t>(n_ - 8u), 1, 2);
      }
    }
    if (n_ > 1u) {
      uint8_t r_cnt8 = static_cast<uint8_t>(((n_ - 1u) < 8u) ? (n_ - 1u) : 8u);
      AscendC::Duplicate<float, true>(lt_mask_row[16], 1.0f, 8ULL, r_cnt8, 1, 2);
      if (n_ > 9u) {
        AscendC::Duplicate<float, true>(lt_mask_row[9u * 16u], 1.0f, 16ULL, static_cast<uint8_t>(n_ - 9u), 1, 2);
      }
    }
    AscendC::ResetMask();
  }

  __aicore__ __attribute__((always_inline)) inline void InitSmallMasksScalar(
    uint32_t row_al,
    __ubuf__ int32_t* off_tr_i32,
    __ubuf__ uint32_t* dmsk_u32,
    __ubuf__ uint32_t* msk_u32,
    __ubuf__ float* mask_row_ub,
    __ubuf__ float* unit_vec_ub
  ) {
    unit_vec_ub[0] = 1.0f;
    #pragma unroll
    for (uint32_t i = 0; i < 16u; ++i) {
      off_tr_i32[i] = static_cast<int32_t>(i * 64u);
      dmsk_u32[i * 17u] = 0u;
    }
    for (uint32_t i = 1u; i < n_; ++i) {
      uint32_t al = (i + 7u) & ~7u;
      for (uint32_t j = i; j < al; ++j) {
        mask_row_ub[i * 16u + j] = 0.0f;
      }
      unit_vec_ub[i * 17u] = 1.0f;
    }
    if (uplo_ == 0u) {
      for (uint32_t i = 0; i < n_; ++i) {
        uint32_t al = (i + 1u + 7u) & ~7u;
        for (uint32_t j = i + 1u; j < al; ++j) {
          msk_u32[i * 16u + j] = 0u;
        }
      }
    } else {
      for (uint32_t i = 0; i < n_; ++i) {
        uint32_t j_al = i & ~7u;
        for (uint32_t j = j_al; j < i; ++j) {
          msk_u32[i * 16u + j] = 0u;
        }
        for (uint32_t j = n_; j < row_al; ++j) {
          msk_u32[i * 16u + j] = 0u;
        }
      }
    }
  }

  __aicore__ __attribute__((always_inline)) inline float SymmetrizeSmallMatrix(
    const AscendC::LocalTensor<int32_t>& lt_tr16_i32,
    const AscendC::LocalTensor<int32_t>& lt_rep8_i32,
    const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_ztmp,
    const AscendC::LocalTensor<float>& lt_ztmp2,
    const AscendC::LocalTensor<float>& lt_red,
    const AscendC::LocalTensor<float>& lt_work,
    const AscendC::LocalTensor<uint32_t>& lt_off_tr16,
    const AscendC::LocalTensor<uint16_t>& lt_z_u16,
    const AscendC::LocalTensor<uint16_t>& lt_ztmp_u16,
    const AscendC::LocalTensor<uint16_t>& lt_msk_u16,
    const AscendC::LocalTensor<uint16_t>& lt_dmsk_u16,
    const __ubuf__ float* red_ub
  ) {
    #pragma unroll
    for (int32_t rep = 1; rep < 16; rep <<= 1) {
      int32_t r8 = rep * 8;
      AscendC::Adds(lt_tr16_i32[rep * 16], lt_tr16_i32[0], rep * 4, rep * 16);
      AscendC::Adds(lt_rep8_i32[r8],       lt_rep8_i32[0], rep * 4, r8);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    if (n_ < 16u) {
      AscendC::Duplicate(lt_z[n_ * 16u], 0.0f, static_cast<int32_t>((16u - n_) * 16u));
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::And(lt_z_u16, lt_z_u16, lt_msk_u16, 512);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(lt_ztmp, lt_z, lt_off_tr16, 0u, 256u);
    AscendC::Abs(lt_ztmp2, lt_z, 256);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::And(lt_ztmp_u16, lt_ztmp_u16, lt_dmsk_u16, 512);
    AscendC::ReduceMax(lt_red, lt_ztmp2, lt_work, 256);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(lt_z, lt_z, lt_ztmp, 256);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    return red_ub[0];
  }

  __aicore__ __attribute__((always_inline)) inline void WriteZeroValAndEyeVec(
    uint32_t diag_step,
    int32_t val_cap,
    int32_t vec_cap,
    uint32_t row_blk_gap,
    const AscendC::GlobalTensor<float>& gt_val,
    const AscendC::GlobalTensor<float>& gt_vec,
    const AscendC::LocalTensor<float>& lt_val,
    const AscendC::LocalTensor<float>& lt_z,
    __ubuf__ float* z_ub
  ) {
    AscendC::Duplicate(lt_val, 0.0f, val_cap);
    AscendC::Duplicate(lt_z, 0.0f, vec_cap);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    for (uint32_t i = 0; i < n_; ++i) {
      z_ub[i * diag_step] = 1.0f;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyExtParams cv(1, static_cast<uint32_t>(n_ * sizeof(float)), 0u, 0u, 0u);
    AscendC::DataCopyExtParams cq(static_cast<uint16_t>(n_), static_cast<uint32_t>(n_ * sizeof(float)), row_blk_gap, 0u, 0u);
    AscendC::DataCopyPad(gt_val, lt_val, cv);
    AscendC::DataCopyPad(gt_vec, lt_z, cq);
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  __aicore__ __attribute__((always_inline)) inline void WriteSmallZeroAnorm(
    uint32_t row_blk_gap,
    const AscendC::GlobalTensor<float>& gt_val,
    const AscendC::GlobalTensor<float>& gt_vec,
    const AscendC::LocalTensor<float>& lt_val,
    const AscendC::LocalTensor<float>& lt_z,
    __ubuf__ float* z_ub
  ) {
    AscendC::PipeBarrier<PIPE_V>();
    WriteZeroValAndEyeVec(17u, 16, 256, row_blk_gap, gt_val, gt_vec, lt_val, lt_z, z_ub);
  }

  __aicore__ __attribute__((always_inline)) inline float ComputeHouseholderScalar(
    float nrm2, float f_val, float& g_val
  ) {
    float nrm = nrm2 * MyRsqrt(nrm2);
    g_val = (f_val >= 0.0f) ? -nrm : nrm;
    return nrm2 - f_val * g_val;
  }

  template <typename UPtr, typename FPtr>
  __aicore__ __attribute__((always_inline)) inline float ComputeHouseholderScalar(
    uint32_t ui, uint32_t l, uint32_t row_i, float nrm2,
    __ubuf__ float* z_ub, UPtr umat_ub, FPtr d, FPtr e, FPtr inv_h_loc
  ) {
    float f_val = umat_ub[row_i + l], g_val = 0.0f;
    d[ui] = z_ub[row_i + ui];
    float h_val = ComputeHouseholderScalar(nrm2, f_val, g_val);
    if (h_val > 1e-28f) {
      float inv_h = 1.0f / h_val;
      inv_h_loc[ui] = inv_h;
      e[l] = g_val;
      umat_ub[row_i + l] = f_val - g_val;
      return inv_h;
    }
    inv_h_loc[ui] = 0.0f;
    e[l] = f_val;
    return 0.0f;
  }
