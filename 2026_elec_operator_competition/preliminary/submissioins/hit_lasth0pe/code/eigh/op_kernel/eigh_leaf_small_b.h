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

  __aicore__ __attribute__((always_inline)) inline void ApplySmallHouseholderStep(
    uint32_t l, uint32_t row_i, float inv_h,
    const AscendC::LocalTensor<float>& lt_ui,
    const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_ztmp,
    const AscendC::LocalTensor<float>& lt_ztmp2,
    const AscendC::LocalTensor<float>& lt_work,
    const AscendC::LocalTensor<float>& lt_p0,
    const AscendC::LocalTensor<float>& lt_t16,
    const AscendC::LocalTensor<float>& lt_fdot8,
    const AscendC::LocalTensor<float>& lt_mask_row,
    const AscendC::LocalTensor<uint32_t>& lt_off_rep8,
    const AscendC::UnaryRepeatParams& u_def16,
    const AscendC::BinaryRepeatParams& b_def16,
    const AscendC::BinaryRepeatParams& mv_mul16,
    const AscendC::BinaryRepeatParams& rk_mul16
  ) {
    uint8_t l1_u8  = static_cast<uint8_t>(l + 1u);
    int32_t l1_i32 = static_cast<int32_t>(l + 1u);
    uint32_t rep8_cnt = (l + 1u) * 8u;
    AscendC::Mul<float, false>(lt_ztmp[0], lt_z[0], lt_ui, 16ULL, l1_u8, mv_mul16);
    AscendC::Muls<float, false>(lt_work[row_i], lt_ui, -inv_h, 16ULL, 1, u_def16);
    AscendC::Muls<float, false>(lt_work[256], lt_ui, -0.5f * inv_h, 16ULL, 1, u_def16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, false>(lt_ztmp2[0], lt_ztmp[0], 16, l1_i32, 1, 1, 2);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float, false>(lt_p0, lt_ztmp2[0], -inv_h, 16ULL, 1, u_def16);
    AscendC::Mul<float, false>(lt_t16, lt_ztmp2[0], lt_work[row_i], 16ULL, 1, b_def16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, false>(lt_fdot8, lt_t16, 16, 1, 1, 1, 2);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::LocalTensor<uint32_t> lt_off_zero16 = lt_mask_row.ReinterpretCast<uint32_t>();
    AscendC::Gather(lt_t16, lt_fdot8, lt_off_zero16, 0u, 16u);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::MulAddDst<float, float, false>(lt_p0, lt_work[256], lt_t16, 16ULL, 1, b_def16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(lt_ztmp2[0],   lt_p0, lt_off_rep8, 0u, rep8_cnt);
    AscendC::Gather(lt_ztmp2[128], lt_ui, lt_off_rep8, 0u, rep8_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::MulAddDst<float, float, true>(lt_z[0], lt_ui, lt_ztmp2[0],   16ULL, l1_u8, rk_mul16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::MulAddDst<float, float, false>(lt_z[0], lt_p0, lt_ztmp2[128], 16ULL, l1_u8, rk_mul16);
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline void AccumSmallQ16(
    const float* inv_h_loc,
    const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_ztmp,
    const AscendC::LocalTensor<float>& lt_ztmp2,
    const AscendC::LocalTensor<float>& lt_umat,
    const AscendC::LocalTensor<float>& lt_unit_vec,
    const AscendC::LocalTensor<float>& lt_work,
    const AscendC::LocalTensor<float>& lt_p0,
    const AscendC::LocalTensor<float>& lt_red,
    const AscendC::LocalTensor<uint32_t>& lt_off_rep8,
    const AscendC::UnaryRepeatParams& u_def16,
    const AscendC::BinaryRepeatParams& mv_mul16,
    const AscendC::BinaryRepeatParams& rk_mul16
  ) {
    AscendC::Adds<float, true>(lt_red, lt_z[0], 0.0f, 32ULL, 1, u_def16);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<float, true>(lt_z, lt_unit_vec, 0.0f, 64ULL, 4, u_def16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<float>(0ULL, 0xFFFFULL);
    for (int32_t i = 2; i < static_cast<int32_t>(n_); ++i) {
      if (inv_h_loc[i] > 0.0f) {
        uint8_t l1_u8 = static_cast<uint8_t>(i);
        uint32_t row_i = static_cast<uint32_t>(i) * 16u;
        AscendC::Mul<float, false>(lt_ztmp[0], lt_z[0], lt_work[row_i], 16ULL, l1_u8, mv_mul16);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float, false>(lt_p0, lt_ztmp[0], 16, i, 1, 1, 2);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(lt_ztmp2[0], lt_p0, lt_off_rep8, 0u, static_cast<uint32_t>(i) * 8u);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst<float, float, true>(lt_z[0], lt_umat[row_i], lt_ztmp2[0], 16ULL, l1_u8, rk_mul16);
        AscendC::PipeBarrier<PIPE_V>();
      }
    }
  }

  __aicore__ __attribute__((always_inline)) inline void TridiagAndAccumSmall(
    float* d, float* e, float* inv_h_loc,
    __ubuf__ float* nrm8_ub, __ubuf__ float* umat_ub, __ubuf__ float* z_ub,
    const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_ztmp,
    const AscendC::LocalTensor<float>& lt_ztmp2,
    const AscendC::LocalTensor<float>& lt_umat,
    const AscendC::LocalTensor<float>& lt_mask_row,
    const AscendC::LocalTensor<float>& lt_unit_vec,
    const AscendC::LocalTensor<float>& lt_work,
    const AscendC::LocalTensor<float>& lt_p0,
    const AscendC::LocalTensor<float>& lt_t16,
    const AscendC::LocalTensor<float>& lt_nrm8,
    const AscendC::LocalTensor<float>& lt_fdot8,
    const AscendC::LocalTensor<float>& lt_red,
    const AscendC::LocalTensor<uint32_t>& lt_off_rep8,
    const AscendC::UnaryRepeatParams& u_def16,
    const AscendC::BinaryRepeatParams& b_def16,
    const AscendC::BinaryRepeatParams& mv_mul16,
    const AscendC::BinaryRepeatParams& rk_mul16
  ) {
    inv_h_loc[0] = 0.0f;
    inv_h_loc[1] = 0.0f;
    AscendC::SetVectorMask<float>(0ULL, 0xFFFFULL);
    for (int32_t i = static_cast<int32_t>(n_) - 1; i >= 2; --i) {
      uint32_t ui = static_cast<uint32_t>(i), l = ui - 1u, row_i = ui * 16u;
      AscendC::LocalTensor<float> lt_ui = lt_umat[row_i];
      AscendC::Mul<float, false>(lt_ui, lt_z[row_i], lt_mask_row[row_i], 16ULL, 1, b_def16);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul<float, false>(lt_t16, lt_ui, lt_ui, 16ULL, 1, b_def16);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::WholeReduceSum<float, false>(lt_nrm8, lt_t16, 16, 1, 1, 1, 2);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      float inv_h = ComputeHouseholderScalar(ui, l, row_i, nrm8_ub[0], z_ub, umat_ub, d, e, inv_h_loc);
      if (inv_h > 0.0f) {
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        ApplySmallHouseholderStep(
          l, row_i, inv_h, lt_ui, lt_z, lt_ztmp, lt_ztmp2, lt_work,
          lt_p0, lt_t16, lt_fdot8, lt_mask_row, lt_off_rep8, u_def16, b_def16, mv_mul16, rk_mul16
        );
      } else {
        if constexpr (n_ == 16u) {
          EIGH_PAD64() EIGH_PAD16() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4()
        } else {
          EIGH_PAD64() EIGH_PAD64() EIGH_PAD64() EIGH_PAD16() EIGH_PAD4() EIGH_PAD4() EIGH_PAD4()
        }
      }
    }
    AccumSmallQ16(
      inv_h_loc, lt_z, lt_ztmp, lt_ztmp2, lt_umat, lt_unit_vec, lt_work,
      lt_p0, lt_red, lt_off_rep8, u_def16, mv_mul16, rk_mul16
    );
  }

  __aicore__ __attribute__((always_inline)) inline void ReduceSmallColExtrema(
    const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_ztmp,
    const AscendC::LocalTensor<float>& lt_umat,
    const AscendC::LocalTensor<uint32_t>& lt_off_tr16,
    const AscendC::BinaryRepeatParams& b_def16
  ) {
    if constexpr (n_ == 16u) {
      AscendC::Gather(lt_ztmp, lt_z, lt_off_tr16, 0u, 256u);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Max<float, true>(lt_umat[0],    lt_ztmp[0],   lt_ztmp[128], 64ULL, 2, b_def16);
      AscendC::Min<float, false>(lt_umat[128], lt_ztmp[0],   lt_ztmp[128], 64ULL, 2, b_def16);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Max<float, false>(lt_umat[0],   lt_umat[0],   lt_umat[64],  64ULL, 1, b_def16);
      AscendC::Min<float, false>(lt_umat[128], lt_umat[128], lt_umat[192], 64ULL, 1, b_def16);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Max<float, true>(lt_umat[0],    lt_umat[0],   lt_umat[32],  32ULL, 1, b_def16);
      AscendC::Min<float, false>(lt_umat[128], lt_umat[128], lt_umat[160], 32ULL, 1, b_def16);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Max<float, true>(lt_umat[0],    lt_umat[0],   lt_umat[16],  16ULL, 1, b_def16);
      AscendC::Min<float, false>(lt_umat[128], lt_umat[128], lt_umat[144], 16ULL, 1, b_def16);
      AscendC::PipeBarrier<PIPE_V>();
    } else {
      AscendC::ResetMask();
      AscendC::Gather(lt_ztmp, lt_z, lt_off_tr16, 0u, 256u);
      AscendC::PipeBarrier<PIPE_V>();
      if (n_ < 16u) {
        AscendC::Duplicate(lt_ztmp[n_ * 16u], 0.0f, static_cast<int32_t>((16u - n_) * 16u));
        AscendC::PipeBarrier<PIPE_V>();
      }
      AscendC::Max(lt_umat[0],   lt_ztmp[0], lt_ztmp[128], 128);
      AscendC::Min(lt_umat[128], lt_ztmp[0], lt_ztmp[128], 128);
      AscendC::PipeBarrier<PIPE_V>();
      for (int32_t h = 64; h >= 16; h >>= 1) {
        AscendC::Max(lt_umat[0],   lt_umat[0],   lt_umat[h],       h);
        AscendC::Min(lt_umat[128], lt_umat[128], lt_umat[128 + h], h);
        AscendC::PipeBarrier<PIPE_V>();
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  template <typename DPtr, typename TIdx>
  __aicore__ __attribute__((always_inline)) inline void SortEigenPairsInplace(
    uint32_t len, DPtr d, TIdx* perm,
    __ubuf__ float* mat_ub = nullptr, uint32_t my_cols = 0u, uint32_t stride = 0u
  ) {
    for (uint32_t i = 0; i + 1u < len; ++i) {
      uint32_t k_min = i;
      float p_min = d[i];
      for (uint32_t j = i + 1u; j < len; ++j) {
        if (d[j] < p_min) {
          k_min = j;
          p_min = d[j];
        }
      }
      if (k_min != i) {
        d[k_min] = d[i];
        d[i] = p_min;
        if (perm != nullptr) {
          TIdx tp = perm[i];
          perm[i] = perm[k_min];
          perm[k_min] = tp;
        } else {
          for (uint32_t c = 0; c < my_cols; ++c) {
            float tmp = mat_ub[c * stride + i];
            mat_ub[c * stride + i] = mat_ub[c * stride + k_min];
            mat_ub[c * stride + k_min] = tmp;
          }
        }
      }
    }
  }

  __aicore__ __attribute__((always_inline)) inline float ResolveSmallColSignTie(
    uint32_t i, const __ubuf__ float* z_ub
  ) {
    if constexpr (n_ == 16u) {
      EIGH_PAD16() EIGH_PAD16() EIGH_PAD4()
    } else {
      EIGH_PAD16() EIGH_PAD16() EIGH_PAD16() EIGH_PAD4()
    }
    uint32_t roff = i * 16u, max_k = 0u;
    float max_abs = MyAbs(z_ub[roff]);
    for (uint32_t r = 1u; r < n_; ++r) {
      float av = MyAbs(z_ub[roff + r]);
      if (av > max_abs) {
        max_abs = av;
        max_k = r;
      }
    }
    return (z_ub[roff + max_k] < 0.0f) ? -1.0f : 1.0f;
  }

  __aicore__ __attribute__((always_inline)) inline void FinalizeSmallOutput(
    float scale_pow2, uint32_t row_blk_gap,
    float* d, uint32_t* perm, float* sgn_loc,
    const __ubuf__ float* umat_ub, const __ubuf__ float* z_ub, __ubuf__ float* val_ub,
    const AscendC::GlobalTensor<float>& gt_val,
    const AscendC::GlobalTensor<float>& gt_vec,
    const AscendC::LocalTensor<float>& lt_val,
    const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_ztmp,
    const AscendC::LocalTensor<uint32_t>& lt_off_tr16,
    const AscendC::UnaryRepeatParams& u_def16
  ) {
    for (uint32_t i = 0; i < n_; ++i) {
      d[i] *= scale_pow2;
      perm[i] = i;
      float mx = umat_ub[i], mn = umat_ub[128u + i];
      if (mx > -mn) {
        sgn_loc[i] = 1.0f;
      } else if (-mn > mx) {
        sgn_loc[i] = -1.0f;
      } else {
        sgn_loc[i] = ResolveSmallColSignTie(i, z_ub);
      }
    }
    SortEigenPairsInplace(n_, d, perm);
    if constexpr (n_ != 16u) {
      AscendC::SetVectorMask<float>(0ULL, 0xFFFFULL);
    }
    for (uint32_t col = 0; col < n_; ++col) {
      val_ub[col] = d[col];
      uint32_t sr = perm[col];
      AscendC::Muls<float, false>(lt_ztmp[col * 16u], lt_z[sr * 16u], sgn_loc[sr], 16ULL, 1, u_def16);
    }
    if constexpr (n_ != 16u) {
      AscendC::ResetMask();
      if (n_ < 16u) {
        AscendC::Duplicate(lt_ztmp[n_ * 16u], 0.0f, static_cast<int32_t>((16u - n_) * 16u));
      }
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(lt_z, lt_ztmp, lt_off_tr16, 0u, 256u);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::DataCopyExtParams cv(1, static_cast<uint32_t>(n_ * sizeof(float)), 0u, 0u, 0u);
    AscendC::DataCopyExtParams cq(static_cast<uint16_t>(n_), static_cast<uint32_t>(n_ * sizeof(float)), row_blk_gap, 0u, 0u);
    AscendC::DataCopyPad(gt_val, lt_val, cv);
    AscendC::DataCopyPad(gt_vec, lt_z, cq);
    AscendC::PipeBarrier<PIPE_ALL>();
  }

  __aicore__ __attribute__((always_inline)) inline float LoadAndPrepareSmall(
    uint32_t row_al, uint8_t right_pad, uint32_t row_blk_gap,
    const AscendC::GlobalTensor<float>& gt_A,
    const AscendC::LocalTensor<float>& ub_all,
    __ubuf__ float* base_ptr
  ) {
    __ubuf__ uint32_t* u32_base = reinterpret_cast<__ubuf__ uint32_t*>(base_ptr);
    AscendC::LocalTensor<int32_t> lt_i32 = ub_all.ReinterpretCast<int32_t>();
    AscendC::LocalTensor<uint32_t> lt_u32 = ub_all.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint16_t> lt_u16 = ub_all.ReinterpretCast<uint16_t>();
    AscendC::DataCopyExtParams copy_in(static_cast<uint16_t>(n_), static_cast<uint32_t>(n_ * sizeof(float)), 0u, row_blk_gap, 0u);
    AscendC::DataCopyPadExtParams<float> pad_in(right_pad > 0, 0, right_pad, 0.0f);
    if constexpr (n_ == 16u) {
      AscendC::DataCopyPad(ub_all[1024], gt_A, copy_in, pad_in);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }
    InitSmallMasksVec(row_al, ub_all[1792], lt_i32[2304], lt_i32[2560], ub_all[2816], ub_all[3072], lt_i32[832]);
    if constexpr (n_ != 16u) {
      AscendC::DataCopyPad(ub_all[1024], gt_A, copy_in, pad_in);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    InitSmallMasksScalar(row_al, reinterpret_cast<__ubuf__ int32_t*>(base_ptr + 2048), u32_base + 2560, u32_base + 2304, base_ptr + 2816, base_ptr + 3072);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    return SymmetrizeSmallMatrix(
      lt_i32[2048], lt_i32[832], ub_all[1024], ub_all[1280], ub_all[1536], ub_all[256], ub_all[320],
      lt_u32[2048], lt_u16[2048], lt_u16[2560], lt_u16[4608], lt_u16[5120], base_ptr + 256
    );
  }

  __aicore__ __attribute__((noinline)) void SolveSmallL1(
    __gm__ float* cur_A,
    __gm__ float* cur_val,
    __gm__ float* cur_vec,
    const AscendC::LocalTensor<float>& ub_all
  ) {
    float d[16], e[16], inv_h_loc[16], sgn_loc[16];
    uint32_t perm[16];
    __ubuf__ float* base_ptr = reinterpret_cast<__ubuf__ float*>(ub_all.GetPhyAddr());
    AscendC::LocalTensor<uint32_t> lt_u32 = ub_all.ReinterpretCast<uint32_t>();
    AscendC::GlobalTensor<float> gt_A, gt_val, gt_vec;
    gt_A.SetGlobalBuffer(cur_A, n_ * n_);
    gt_val.SetGlobalBuffer(cur_val, n_);
    gt_vec.SetGlobalBuffer(cur_vec, n_ * n_);
    if (n_ == 1u) {
      SolveSmallN1(gt_A, gt_val, gt_vec, ub_all[0], ub_all[1024], base_ptr + 1024);
      return;
    }
    uint32_t row_al = (n_ + 7u) & ~7u;
    uint8_t right_pad = static_cast<uint8_t>(row_al - n_);
    uint32_t row_blk_gap = (16u - row_al) / 8u;
    float anorm = LoadAndPrepareSmall(row_al, right_pad, row_blk_gap, gt_A, ub_all, base_ptr);
    if (anorm == 0.0f) {
      WriteSmallZeroAnorm(row_blk_gap, gt_val, gt_vec, ub_all[0], ub_all[1024], base_ptr + 1024);
      return;
    }
    float scale_pow2, inv_scale_pow2;
    GetPow2Scale(anorm, scale_pow2, inv_scale_pow2);
    AscendC::UnaryRepeatParams u_def16(1, 1, 8, 8), tql_u_rep2(1, 1, 2, 2);
    AscendC::BinaryRepeatParams b_def16(1, 1, 1, 8, 8, 8), mv_mul16(1, 1, 1, 2, 2, 0), rk_mul16(1, 1, 0, 2, 0, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<float, true>(ub_all[1024], ub_all[1024], inv_scale_pow2, 64ULL, 4, u_def16);
    AscendC::PipeBarrier<PIPE_V>();
    TridiagAndAccumSmall(
      d, e, inv_h_loc, base_ptr + 184, base_ptr + 1792, base_ptr + 1024,
      ub_all[1024], ub_all[1280], ub_all[1536], ub_all[1792], ub_all[2816], ub_all[3072],
      ub_all[320], ub_all[32], ub_all[48], ub_all[184], ub_all[248], ub_all[256], lt_u32[832],
      u_def16, b_def16, mv_mul16, rk_mul16
    );
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    d[0] = base_ptr[256];
    d[1] = base_ptr[256 + 17];
    e[0] = base_ptr[256 + 16];
    e[n_ - 1u] = 0.0f;
    RunTqlLoop16<false>(static_cast<int32_t>(n_), 45, 2.5e-6f, d, e, ub_all[1024], ub_all[112], ub_all[144], tql_u_rep2, b_def16);
    ReduceSmallColExtrema(ub_all[1024], ub_all[1280], ub_all[1792], lt_u32[2048], b_def16);
    FinalizeSmallOutput(
      scale_pow2, row_blk_gap, d, perm, sgn_loc, base_ptr + 1792, base_ptr + 1024, base_ptr,
      gt_val, gt_vec, ub_all[0], ub_all[1024], ub_all[1280], lt_u32[2048], u_def16
    );
  }
