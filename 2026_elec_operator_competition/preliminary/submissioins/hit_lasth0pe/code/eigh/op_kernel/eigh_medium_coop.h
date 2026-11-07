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

  __aicore__ __attribute__((always_inline)) inline void InitMedMasksVec(
    uint32_t row_al, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_hsave,
    const AscendC::LocalTensor<int32_t>& lt_msk_i32, const AscendC::LocalTensor<int32_t>& lt_dmsk_i32,
    const AscendC::LocalTensor<uint32_t>& lt_off_zero64, __ubuf__ uint32_t* off_elem, __ubuf__ uint32_t* off_rep8_u32, __ubuf__ int32_t* off_tr_i32
  ) {
    AscendC::Duplicate<float, true>(lt_z, 0.0f, 64ULL, 64, 1, 8);
    AscendC::Duplicate<float, false>(lt_hsave, 0.0f, 64ULL, 64, 1, 8);
    AscendC::Duplicate<int32_t, false>(lt_msk_i32, 0, 64ULL, 64, 1, 8);
    AscendC::Duplicate<int32_t, false>(lt_dmsk_i32, -1, 64ULL, 64, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t r0 = 0; r0 < n_; r0 += 8u) {
      uint32_t r_cnt = (r0 + 8u <= n_) ? 8u : (n_ - r0);
      uint32_t off   = (uplo_ == 0u) ? (r0 * 64u) : (r0 * 64u + r0);
      uint64_t msk   = (uplo_ == 0u) ? static_cast<uint64_t>(r0 + 8u) : static_cast<uint64_t>(row_al - r0);
      AscendC::Duplicate<int32_t, true>(lt_msk_i32[off], -1, msk, static_cast<uint8_t>(r_cnt), 1, 8);
    }
    AscendC::ResetMask();
    for (uint32_t k = 0; k < 8u; ++k) {
      off_elem[k]     = 0u;
      off_rep8_u32[k] = 0u;
    }
    AscendC::Duplicate<uint32_t, true>(lt_off_zero64, 0u, 64ULL, 1, 1, 8);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < 64u; ++j) off_tr_i32[j] = static_cast<int32_t>(j * 256u);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void InitMedOffsetsAndMasksScalar(
    uint32_t row_al, const AscendC::LocalTensor<int32_t>& lt_off_elem_i, const AscendC::LocalTensor<int32_t>& lt_rep8_i32,
    const AscendC::LocalTensor<int32_t>& lt_tr64_i32, __ubuf__ uint32_t* msk_u32, __ubuf__ uint32_t* dmsk_u32
  ) {
    for (int32_t rep = 8; rep < 512; rep <<= 1) {
      int32_t delta = (rep >> 3) * 4;
      AscendC::Adds(lt_off_elem_i[rep], lt_off_elem_i[0], delta, rep);
      AscendC::Adds(lt_rep8_i32[rep],   lt_rep8_i32[0],   delta, rep);
      AscendC::PipeBarrier<PIPE_V>();
    }
    for (int32_t rep = 1; rep < 64; rep <<= 1) {
      AscendC::Adds(lt_tr64_i32[rep * 64], lt_tr64_i32[0], rep * 4, rep * 64);
      AscendC::PipeBarrier<PIPE_V>();
    }
    for (uint32_t i = 0; i < 64u; ++i) dmsk_u32[i * 65u] = 0u;
    for (uint32_t i = 0; i < n_; ++i) {
      if (uplo_ == 0u) {
        uint32_t al = (i + 1u + 7u) & ~7u;
        for (uint32_t j = i + 1u; j < al; ++j) msk_u32[i * 64u + j] = 0u;
      } else {
        uint32_t j_al = i & ~7u;
        for (uint32_t j = j_al; j < i; ++j) msk_u32[i * 64u + j] = 0u;
        for (uint32_t j = n_; j < row_al; ++j) msk_u32[i * 64u + j] = 0u;
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline float SymmetrizeMedMatrix(
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2,
    const AscendC::LocalTensor<float>& lt_red, const AscendC::LocalTensor<float>& lt_work, const AscendC::LocalTensor<float>& lt_mask_row,
    const AscendC::LocalTensor<uint32_t>& lt_off_tr64, const AscendC::LocalTensor<uint16_t>& lt_z_u16, const AscendC::LocalTensor<uint16_t>& lt_ztmp_u16,
    const AscendC::LocalTensor<uint16_t>& lt_msk_u16, const AscendC::LocalTensor<uint16_t>& lt_dmsk_u16, __ubuf__ float* red_ub,
    const AscendC::UnaryRepeatParams& u_def64, const AscendC::BinaryRepeatParams& b_def64
  ) {
    AscendC::And(lt_z_u16, lt_z_u16, lt_msk_u16, 8192);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(lt_ztmp, lt_z, lt_off_tr64, 0u, 4096u);
    AscendC::Abs<float, true>(lt_ztmp2, lt_z, 64ULL, 64, u_def64);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::And(lt_ztmp_u16, lt_ztmp_u16, lt_dmsk_u16, 8192);
    AscendC::ReduceMax(lt_red, lt_ztmp2, lt_work, 4096);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float, true>(lt_z, lt_z, lt_ztmp, 64ULL, 64, b_def64);

    AscendC::Duplicate<float, false>(lt_mask_row, 0.0f, 64ULL, 64, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t r0 = 1u; r0 < n_; r0 += 8u) {
      uint32_t al = r0 + 7u;
      uint32_t r_cnt = (r0 + 8u <= n_) ? 8u : (n_ - r0);
      AscendC::Duplicate<float, true>(lt_mask_row[r0 * 64u], 1.0f, static_cast<uint64_t>(al), static_cast<uint8_t>(r_cnt), 1, 8);
    }
    AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    return red_ub[0];
  }

  __aicore__ __attribute__((always_inline)) inline void WriteMedZeroAnorm(
    uint32_t p, uint32_t sym_cap, uint32_t row_blk_gap,
    const AscendC::LocalTensor<float>& lt_val, const AscendC::LocalTensor<float>& lt_z, __ubuf__ float* z_ub,
    const AscendC::GlobalTensor<float>& gt_val, const AscendC::GlobalTensor<float>& gt_vec
  ) {
    if constexpr (n_ == 64u) {
      EIGH_PAD64() EIGH_PAD64() EIGH_PAD16() EIGH_PAD16() EIGH_PAD16() EIGH_PAD16() EIGH_PAD16() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4()
    } else {
      EIGH_PAD64() EIGH_PAD16() EIGH_PAD16() EIGH_PAD16() EIGH_PAD4() EIGH_PAD4()
    }
    uint32_t sync_cnt = ((n_ & 7u) == 0u ? 4u : 5u) + ((sym_cap < 1792u) ? 1u : 0u);
    for (uint32_t sc = 0; sc < sync_cnt; ++sc) SafeSyncAll();
    if (p == 0u) WriteZeroValAndEyeVec(65u, 64, 4096, row_blk_gap, gt_val, gt_vec, lt_val, lt_z, z_ub);
  }

  template <bool RESET_AND_SIGNAL>
  __aicore__ __attribute__((always_inline)) inline void EvalRow64Norm2(
    uint32_t row_off, const AscendC::LocalTensor<float>& lt_umat, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_mask_row, const AscendC::LocalTensor<float>& lt_t64, const AscendC::LocalTensor<float>& lt_nrm8,
    const AscendC::BinaryRepeatParams& b_def64
  ) {
    AscendC::Mul<float, RESET_AND_SIGNAL>(lt_umat[row_off], lt_z[row_off], lt_mask_row[row_off], 64ULL, 1, b_def64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(lt_t64, lt_umat[row_off], lt_umat[row_off], 64ULL, 1, b_def64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, false>(lt_nrm8, lt_t64, 64, 1, 1, 1, 8);
    if constexpr (RESET_AND_SIGNAL) {
      AscendC::ResetMask();
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    }
  }

  template <bool CLEAR_ZTMP2, bool PREFETCH_NEXT>
  __aicore__ __attribute__((always_inline)) inline void ApplyHouseholder64Step(
    uint32_t l, float inv_h, const AscendC::LocalTensor<float>& lt_ui, const AscendC::LocalTensor<float>& lt_umat,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_mask_row, const AscendC::LocalTensor<float>& lt_ztmp,
    const AscendC::LocalTensor<float>& lt_ztmp2, const AscendC::LocalTensor<float>& lt_p0, const AscendC::LocalTensor<float>& lt_t64,
    const AscendC::LocalTensor<float>& lt_nrm8, const AscendC::LocalTensor<float>& lt_fdot8, const AscendC::LocalTensor<uint32_t>& lt_off_zero64,
    const AscendC::LocalTensor<uint32_t>& lt_off_rep8, const AscendC::UnaryRepeatParams& u_def64, const AscendC::BinaryRepeatParams& b_def64,
    const AscendC::BinaryRepeatParams& mv_mul64, const AscendC::BinaryRepeatParams& rk_mul64
  ) {
    uint8_t l1_u8  = static_cast<uint8_t>(l + 1u);
    int32_t l1_i32 = static_cast<int32_t>(l + 1u);
    float half_inv_h = -0.5f * inv_h;
    uint32_t rep8_cnt = (l + 1u) * 8u;
    if constexpr (CLEAR_ZTMP2) {
      AscendC::Duplicate<float, true>(lt_ztmp2[0], 0.0f, 64ULL, 1, 1, 8);
      AscendC::Mul<float, false>(lt_ztmp[0], lt_z[0], lt_ui, 64ULL, l1_u8, mv_mul64);
    } else {
      AscendC::Mul<float, true>(lt_ztmp[0], lt_z[0], lt_ui, 64ULL, l1_u8, mv_mul64);
    }
    AscendC::Muls<float, false>(lt_ztmp2[128], lt_ui, -inv_h, 64ULL, 1, u_def64);
    AscendC::Muls<float, false>(lt_ztmp2[192], lt_ui, half_inv_h, 64ULL, 1, u_def64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, false>(lt_ztmp2[0], lt_ztmp[0], 64, l1_i32, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Muls<float, false>(lt_p0, lt_ztmp2[0], -inv_h, 64ULL, 1, u_def64);
    AscendC::Mul<float, false>(lt_t64, lt_ztmp2[0], lt_ztmp2[128], 64ULL, 1, b_def64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, false>(lt_fdot8, lt_t64, 64, 1, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Gather(lt_t64, lt_fdot8, lt_off_zero64, 0u, 64u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::MulAddDst<float, float, false>(lt_p0, lt_ztmp2[192], lt_t64, 64ULL, 1, b_def64);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Gather(lt_ztmp2[0],   lt_p0, lt_off_rep8, 0u, rep8_cnt);
    AscendC::Gather(lt_ztmp2[512], lt_ui, lt_off_rep8, 0u, rep8_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::MulAddDst<float, float, true>(lt_z[0], lt_ui, lt_ztmp2[0], 64ULL, l1_u8, rk_mul64);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (PREFETCH_NEXT) {
      if (l >= 2u) {
        uint32_t row_l = l * 64u;
        uint8_t l_u8 = static_cast<uint8_t>(l);
        AscendC::MulAddDst<float, float, false>(lt_z[row_l], lt_p0, lt_ztmp2[512u + l * 8u], 64ULL, 1, rk_mul64);
        AscendC::PipeBarrier<PIPE_V>();
        EvalRow64Norm2<false>(row_l, lt_umat, lt_z, lt_mask_row, lt_t64, lt_nrm8, b_def64);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::MulAddDst<float, float, false>(lt_z[0], lt_p0, lt_ztmp2[512], 64ULL, l_u8, rk_mul64);
        AscendC::ResetMask();
        AscendC::PipeBarrier<PIPE_V>();
        return;
      }
    }
    AscendC::MulAddDst<float, float, false>(lt_z[0], lt_p0, lt_ztmp2[512], 64ULL, l1_u8, rk_mul64);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline void TridiagMedInUB(
    float inv_scale_pow2, float* d, float* e, float* inv_h_loc,
    __ubuf__ float* z_ub, __ubuf__ float* hsave_ub, __ubuf__ float* mask_row_ub, __ubuf__ float* nrm8_ub,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_hsave, const AscendC::LocalTensor<float>& lt_mask_row,
    const AscendC::LocalTensor<float>& lt_ztmp, const AscendC::LocalTensor<float>& lt_ztmp2, const AscendC::LocalTensor<float>& lt_p0,
    const AscendC::LocalTensor<float>& lt_t64, const AscendC::LocalTensor<float>& lt_nrm8, const AscendC::LocalTensor<float>& lt_fdot8,
    const AscendC::LocalTensor<int32_t>& lt_i32, const AscendC::LocalTensor<int32_t>& lt_tr64_i32,
    const AscendC::LocalTensor<uint32_t>& lt_off_zero64, const AscendC::LocalTensor<uint32_t>& lt_off_rep8,
    const AscendC::UnaryRepeatParams& u_def64, const AscendC::BinaryRepeatParams& b_def64,
    const AscendC::BinaryRepeatParams& mv_mul64, const AscendC::BinaryRepeatParams& rk_mul64
  ) {
    AscendC::Adds<int32_t, true>(lt_tr64_i32[0], lt_i32[48128], 0, 64ULL, 8, u_def64);
    AscendC::PipeBarrier<PIPE_V>();
    for (int32_t rep = 1; rep < 8; ++rep) {
      AscendC::Adds<int32_t, false>(lt_tr64_i32[rep * 512], lt_tr64_i32[0], rep * 256, 64ULL, 8, u_def64);
    }
    AscendC::ResetMask();
    for (uint32_t i = 1u; i < n_; ++i) {
      uint32_t al = (i + 7u) & ~7u;
      for (uint32_t j = i; j < al; ++j) mask_row_ub[i * 64u + j] = 0.0f;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::Muls<float, true>(lt_z, lt_z, inv_scale_pow2, 64ULL, 64, u_def64);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();

    inv_h_loc[0] = 0.0f;
    inv_h_loc[1] = 0.0f;
    if constexpr (n_ == 64u) EvalRow64Norm2<true>((n_ - 1u) * 64u, lt_hsave, lt_z, lt_mask_row, lt_t64, lt_nrm8, b_def64);
    for (int32_t i = static_cast<int32_t>(n_) - 1; i >= 2; --i) {
      uint32_t ui = static_cast<uint32_t>(i), l = ui - 1u, row_i = ui * 64u;
      if constexpr (n_ != 64u) EvalRow64Norm2<true>(row_i, lt_hsave, lt_z, lt_mask_row, lt_t64, lt_nrm8, b_def64);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      float nrm2 = nrm8_ub[0], f_val = hsave_ub[row_i + l], g_val = 0.0f;
      d[ui] = z_ub[row_i + ui];
      float h_val = ComputeHouseholderScalar(nrm2, f_val, g_val);
      if (h_val > 1e-28f) {
        float inv_h = 1.0f / h_val;
        inv_h_loc[ui] = inv_h;
        e[l] = g_val;
        hsave_ub[row_i + l] = f_val - g_val;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        ApplyHouseholder64Step<false, (n_ == 64u)>(
          l, inv_h, lt_hsave[row_i], lt_hsave, lt_z, lt_mask_row, lt_ztmp, lt_ztmp2,
          lt_p0, lt_t64, lt_nrm8, lt_fdot8, lt_off_zero64, lt_off_rep8, u_def64, b_def64, mv_mul64, rk_mul64
        );
      } else {
        inv_h_loc[ui] = 0.0f;
        e[l] = f_val;
        if constexpr (n_ == 64u) {
          if (l >= 2u) EvalRow64Norm2<true>(l * 64u, lt_hsave, lt_z, lt_mask_row, lt_t64, lt_nrm8, b_def64);
        }
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void DispatchMedLeavesToGM(
    uint32_t p, int32_t* bounds, float* d, const float* e, __ubuf__ float* dc_z_ub, __ubuf__ float* dc_st_ub,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_dc_z, const AscendC::LocalTensor<int32_t>& lt_bc64_i32,
    const AscendC::GlobalTensor<float>& gt_ws_leaf, const AscendC::UnaryRepeatParams& u_def64
  ) {
    int32_t n_i32 = static_cast<int32_t>(n_), mid_8 = (n_i32 > 56) ? 32 : 24;
    bounds[0] = 0;
    bounds[1] = (n_ == 50u) ? 12 : (mid_8 - 16);
    bounds[2] = mid_8;
    bounds[3] = (n_ == 50u) ? 37 : ((n_i32 > 56) ? 48 : (mid_8 + 16));
    bounds[4] = n_i32;

    for (int32_t b = 1; b < 4; ++b) {
      int32_t cut = bounds[b];
      float ar = MyAbs(e[cut - 1]);
      d[cut - 1] -= ar;
      d[cut]     -= ar;
    }
    if (p < 4u) {
      int32_t s = bounds[p], m = bounds[p + 1] - s;
      for (int32_t t = 0; t < m; ++t) {
        dc_st_ub[0 + t]  = d[s + t];
        dc_st_ub[16 + t] = (t + 1 < m) ? e[s + t] : 0.0f;
      }
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      SolveLeafPipeV16(m, dc_st_ub + 0, dc_st_ub + 16, lt_dc_z, dc_z_ub);
      AscendC::DataCopy(gt_ws_leaf[p * 320u], lt_dc_z, 320);
    }
    AscendC::Duplicate<float, true>(lt_z, 0.0f, 64ULL, 64, 1, 8);
    AscendC::Duplicate<int32_t, false>(lt_bc64_i32[0], 0, 64ULL, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    #pragma unroll
    for (int32_t rep = 1; rep < 64; rep <<= 1) {
      AscendC::Adds<int32_t, false>(lt_bc64_i32[rep * 64], lt_bc64_i32[0], rep * 4, 64ULL, static_cast<uint8_t>(rep), u_def64);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::ResetMask();
    SafeSyncAll();
  }

  __aicore__ __attribute__((always_inline)) inline void UnpackMedLeavesFromGM(
    uint32_t sym_cap, const int32_t* bounds, float* d, float* top, float* bot,
    __ubuf__ float* z_ub, __ubuf__ float* ztmp_ub, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_ztmp,
    const AscendC::GlobalTensor<float>& gt_ws_leaf
  ) {
    AscendC::DataCopy(lt_ztmp[0], gt_ws_leaf[0], 1280);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::UnaryRepeatParams u_leaf16(1, 1, 8, 2);
    uint32_t sb0 = static_cast<uint32_t>(bounds[0]), sb1 = static_cast<uint32_t>(bounds[1]);
    uint32_t sb2 = static_cast<uint32_t>(bounds[2]), sb3 = static_cast<uint32_t>(bounds[3]);
    AscendC::Adds<float, true>(lt_z[sb0 * 64u + sb0], lt_ztmp[64], 0.0f, 16ULL, static_cast<uint8_t>(bounds[1] - bounds[0]), u_leaf16);
    if constexpr (n_ != 50u) AscendC::Adds<float, false>(lt_z[sb1 * 64u + sb1], lt_ztmp[320u + 64u], 0.0f, 16ULL, static_cast<uint8_t>(bounds[2] - bounds[1]), u_leaf16);
    AscendC::Adds<float, false>(lt_z[sb2 * 64u + sb2], lt_ztmp[640u + 64u], 0.0f, 16ULL, static_cast<uint8_t>(bounds[3] - bounds[2]), u_leaf16);
    if constexpr (n_ != 50u) AscendC::Adds<float, false>(lt_z[sb3 * 64u + sb3], lt_ztmp[960u + 64u], 0.0f, 16ULL, static_cast<uint8_t>(bounds[4] - bounds[3]), u_leaf16);
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    if constexpr (n_ == 50u) {
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      for (uint32_t r = 0; r < 12u; ++r) {
        for (uint32_t c = 0; c < 12u; ++c) z_ub[(12u + r) * 64u + 12u + c] = ztmp_ub[320u + 64u + r * 16u + c];
      }
      for (uint32_t r = 0; r < 13u; ++r) {
        for (uint32_t c = 0; c < 13u; ++c) z_ub[(37u + r) * 64u + 37u + c] = ztmp_ub[960u + 64u + r * 16u + c];
      }
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    }
    if (sym_cap < 1792u) {
      EIGH_PAD256() EIGH_PAD256()
      SafeSyncAll();
    }
    for (int32_t b = 0; b < 4; ++b) {
      int32_t s = bounds[b], m = bounds[b + 1] - s;
      __ubuf__ float* l_pack = ztmp_ub + static_cast<uint32_t>(b) * 320u;
      for (int32_t t = 0; t < m; ++t) {
        d[s + t]   = l_pack[0 + t];
        top[s + t] = l_pack[16 + t];
        bot[s + t] = l_pack[32 + t];
      }
    }
  }

  __aicore__ __attribute__((always_inline)) inline void DeflateMedSingleMerge(
    uint32_t m_idx, int32_t step, bool final_level, const int32_t* bounds,
    float* d, const float* e, const float* top, const float* bot, float* ths, float* bhs, float* ad, float* az, float* dd,
    int32_t* orig_ord, int32_t* acol_dc, int32_t* K_arr, int32_t* D_arr, int32_t* triv_arr, float* rho_arr,
    int32_t& n_giv, int32_t* giv_cp, int32_t* giv_pk, float* giv_cg, float* giv_sg,
    const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_dc_st, const AscendC::UnaryRepeatParams& u_def64
  ) {
    int32_t b_idx = static_cast<int32_t>(m_idx) * 2 * step;
    int32_t s = bounds[b_idx], mid = bounds[b_idx + step], end = bounds[b_idx + 2 * step];
    int32_t L = end - s, m1 = mid - s;
    float r_raw = e[mid - 1];
    bool neg = (r_raw < 0.0f);
    float ar = neg ? -r_raw : r_raw;
    rho_arr[m_idx] = ar;
    if (ar == 0.0f) {
      triv_arr[m_idx] = 1; K_arr[m_idx] = 0; D_arr[m_idx] = L;
      return;
    }
    triv_arr[m_idx] = 0;
    MergeTwoSortedLocal(m1, L, &orig_ord[s], &d[s]);

    float dl_l[64], zk_l[64];
    uint64_t msk_p0 = final_level ? 64ULL : 32ULL;
    uint32_t c_off_p0 = final_level ? 0u : static_cast<uint32_t>(s);
    float ref_sc = BuildMergePermAndZ(s, m1, L, neg, ar, false, !final_level, orig_ord, d, top, bot, ths, bhs, dl_l, zk_l);
    float tol_z = 8.5e-7f * ref_sc, tol_d = 1.8e-6f * ref_sc;
    int32_t K = 0, D = 0, last_cp = 0;
    float last_ad = 0.0f, last_az = 0.0f;

    for (int32_t k = 0; k < L; ++k) {
      int32_t pk = orig_ord[s + k];
      float dk = dl_l[k], zk = zk_l[k];
      if (MyAbs(zk) <= tol_z) {
        dd[s + D] = dk; acol_dc[s + L - 1 - D] = pk; D++;
      } else if (K > 0 && MyAbs(dk - last_ad) <= tol_d) {
        float zp = last_az, r_hyp = MySqrt(zp * zp + zk * zk);
        int32_t cp = last_cp;
        float c_g = (r_hyp != 0.0f) ? (zk / r_hyp) : 1.0f, s_g = (r_hyp != 0.0f) ? (zp / r_hyp) : 0.0f;
        if (!final_level) {
          RotateTwoScalars(ths[s + cp], ths[s + pk], c_g, s_g);
          RotateTwoScalars(bhs[s + cp], bhs[s + pk], c_g, s_g);
          RotateTwoVecRowsAxpy(lt_z[static_cast<uint32_t>(s + cp) * 64u + c_off_p0], lt_z[static_cast<uint32_t>(s + pk) * 64u + c_off_p0], lt_dc_st, c_g, s_g, msk_p0, u_def64);
        } else {
          giv_cp[n_giv] = cp; giv_pk[n_giv] = pk; giv_cg[n_giv] = c_g; giv_sg[n_giv] = s_g; n_giv++;
        }
        dd[s + D] = last_ad; acol_dc[s + L - 1 - D] = cp; D++;
        ad[s + K - 1] = dk; az[s + K - 1] = r_hyp; acol_dc[s + K - 1] = pk;
        last_ad = dk; last_az = r_hyp; last_cp = pk;
      } else {
        ad[s + K] = dk; az[s + K] = zk; acol_dc[s + K] = pk;
        last_ad = dk; last_az = zk; last_cp = pk; K++;
      }
    }
    K_arr[m_idx] = K; D_arr[m_idx] = D;
  }

  __aicore__ __attribute__((always_inline)) inline void UpdateMedSecularLanes(
    uint32_t i0, uint32_t r_cnt, int32_t K_my, int32_t j_base, float rho_my, __ubuf__ float* dc_st_ub,
    const float* near_d, const float* near_z2, const float* d_orig_l, int32_t* done_l, float* low_l, float* high_l, float* tau_l
  ) {
    for (uint32_t lane = 0; lane < r_cnt; ++lane) {
      if (done_l[lane]) continue;
      int32_t ri = static_cast<int32_t>(i0 + lane);
      float t_cur = tau_l[lane], d_orig = d_orig_l[lane];
      float w_tot = dc_st_ub[lane], dp_tot = dc_st_ub[8u + lane], w_abs = dc_st_ub[16u + lane];
      float fv = 1.0f + rho_my * w_tot, dfv = rho_my * dp_tot;
      if (MyAbs(fv) <= 4.0e-7f * (1.0f + rho_my * w_abs)) {
        done_l[lane] = 1;
        continue;
      }
      if (fv < 0.0f && t_cur > low_l[lane]) low_l[lane] = t_cur;
      if (fv >= 0.0f && t_cur < high_l[lane]) high_l[lane] = t_cur;
      float eta = -fv / dfv;
      if (ri < K_my - 1) {
        float di_f = (near_d[ri - j_base] - d_orig) - t_cur, dip1_f = (near_d[ri + 1 - j_base] - d_orig) - t_cur;
        if (di_f != 0.0f && dip1_f != 0.0f) {
          float dp1_near = near_z2[ri - j_base] / (di_f * di_f), dp2_near = near_z2[ri + 1 - j_base] / (dip1_f * dip1_f);
          float dp_far = dp_tot - (dp1_near + dp2_near);
          if (dp_far < 0.0f) dp_far = 0.0f;
          float dp_h = 0.5f * dp_far;
          float bc = rho_my * (dp1_near + dp_h) * (di_f * di_f), cc = rho_my * (dp2_near + dp_h) * (dip1_f * dip1_f);
          float ac = fv - (bc / di_f) - (cc / dip1_f);
          eta = SolveInteriorQuadEta(di_f, dip1_f, fv, ac, bc, cc, eta);
        }
      } else {
        float d1_e = (near_d[K_my - 2 - j_base] - d_orig) - t_cur, d2_e = (near_d[K_my - 1 - j_base] - d_orig) - t_cur;
        float dw_last = (d2_e != 0.0f) ? (near_z2[K_my - 1 - j_base] / (d2_e * d2_e)) : 0.0f;
        float bc = rho_my * (dp_tot - dw_last) * (d1_e * d1_e), cc = rho_my * dw_last * (d2_e * d2_e);
        float ac = fv - (bc / d1_e) - (cc / d2_e);
        eta = SolveExteriorQuadEta(d1_e, d2_e, t_cur, fv, dfv, ac, bc, cc, eta);
      }
      ApplySecularTauStep(t_cur, eta, fv, dfv, 3.0e-7f, low_l[lane], high_l[lane], tau_l[lane], done_l[lane]);
    }
  }
