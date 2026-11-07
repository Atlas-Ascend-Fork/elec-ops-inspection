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

  __aicore__ __attribute__((always_inline)) inline void InitLargeTail64InUB(
    __ubuf__ float* mask_row64_ub, __ubuf__ uint32_t* off_elem64,
    const AscendC::LocalTensor<float>& lt_umat64, const AscendC::LocalTensor<float>& lt_mask_row64,
    const AscendC::LocalTensor<int32_t>& lt_off_elem64_i
  ) {
    AscendC::Duplicate<float, true>(lt_umat64, 0.0f, 64ULL, 64, 1, 8);
    AscendC::Duplicate<float, false>(lt_mask_row64, 0.0f, 64ULL, 64, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t r0 = 1u; r0 < 64u; r0 += 8u) {
      uint32_t al = r0 + 7u, r_cnt = (r0 + 8u <= 64u) ? 8u : (64u - r0);
      AscendC::Duplicate<float, true>(lt_mask_row64[r0 * 64u], 1.0f, static_cast<uint64_t>(al), static_cast<uint8_t>(r_cnt), 1, 8);
    }
    AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

    for (uint32_t idx = 0; idx < 64u; ++idx) off_elem64[idx] = (idx >> 3u) * 4u;
    for (uint32_t r = 1u; r < 64u; ++r) {
      uint32_t al = (r + 7u) & ~7u;
      for (uint32_t j = r; j < al; ++j) mask_row64_ub[r * 64u + j] = 0.0f;
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (int32_t rep = 64; rep < 512; rep <<= 1) {
      AscendC::Adds(lt_off_elem64_i[rep], lt_off_elem64_i[0], (rep >> 3) * 4, rep);
      AscendC::PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ __attribute__((noinline)) void RunLargeTail64InUB(const AscendC::LocalTensor<float>& ub_all) {
    __ubuf__ float* base_ptr   = reinterpret_cast<__ubuf__ float*>(ub_all.GetPhyAddr());
    __ubuf__ float* d          = base_ptr + 0 * 512;
    __ubuf__ float* e          = base_ptr + 1 * 512;
    __ubuf__ float* inv_h_save = base_ptr + 5 * 512;
    __ubuf__ float* stage_ub   = base_ptr + 13824;
    __ubuf__ float* mat_ub     = base_ptr + 15872;
    __ubuf__ float* z_ub       = base_ptr + 32256;
    __ubuf__ float* umat64_ub  = base_ptr + 3584;

    AscendC::LocalTensor<float> lt_stage = ub_all[13824], lt_mat = ub_all[15872], lt_z = ub_all[32256], lt_umat64 = ub_all[3584];
    AscendC::LocalTensor<float> lt_mask_row64 = lt_mat[0], lt_ztmp64 = lt_z[4096], lt_ztmp2_64 = lt_z[8192];
    AscendC::LocalTensor<float> lt_p0_64 = lt_stage[64], lt_t64 = lt_stage[128], lt_nrm8_64 = lt_stage[392], lt_fdot8_64 = lt_stage[448];
    AscendC::LocalTensor<uint32_t> lt_off_elem64 = ub_all.ReinterpretCast<uint32_t>()[15872 + 9216], lt_off_zero64 = lt_mask_row64.ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<int32_t> lt_off_elem64_i = ub_all.ReinterpretCast<int32_t>()[15872 + 9216];

    AscendC::UnaryRepeatParams u64_t1(1, 1, 8, 8);
    AscendC::BinaryRepeatParams b64_t1(1, 1, 1, 8, 8, 8), mv64_t1(1, 1, 1, 8, 8, 0), rk64_t1(1, 1, 0, 8, 0, 1);

    InitLargeTail64InUB(mat_ub, reinterpret_cast<__ubuf__ uint32_t*>(mat_ub + 9216), lt_umat64, lt_mask_row64, lt_off_elem64_i);
    inv_h_save[0] = 0.0f; inv_h_save[1] = 0.0f;
    EvalRow64Norm2<true>(63u * 64u, lt_umat64, lt_z, lt_mask_row64, lt_t64, lt_nrm8_64, b64_t1);

    for (int32_t i = 63; i >= 2; --i) {
      uint32_t ui = static_cast<uint32_t>(i), l = ui - 1u, row_i = ui * 64u;
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      float nrm2 = stage_ub[392], f_val = umat64_ub[row_i + l], g_val = 0.0f;
      d[ui] = z_ub[row_i + ui];
      float h_val = ComputeHouseholderScalar(nrm2, f_val, g_val);
      if (h_val > 1e-28f) {
        float inv_h = 1.0f / h_val;
        inv_h_save[ui] = inv_h; e[l] = g_val; umat64_ub[row_i + l] = f_val - g_val;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        ApplyHouseholder64Step<true, true>(l, inv_h, lt_umat64[row_i], lt_umat64, lt_z, lt_mask_row64, lt_ztmp64, lt_ztmp2_64, lt_p0_64, lt_t64, lt_nrm8_64, lt_fdot8_64, lt_off_zero64, lt_off_elem64, u64_t1, b64_t1, mv64_t1, rk64_t1);
      } else {
        inv_h_save[ui] = 0.0f; e[l] = f_val;
        if (l >= 2u) EvalRow64Norm2<true>(l * 64u, lt_umat64, lt_z, lt_mask_row64, lt_t64, lt_nrm8_64, b64_t1);
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    d[0] = z_ub[0]; d[1] = z_ub[65u]; e[0] = z_ub[64u];
  }

  __aicore__ __attribute__((always_inline)) inline void EvalRow128Norm2(
    uint32_t row_off, bool reset_mask, const AscendC::LocalTensor<float>& lt_u, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_red,
    const AscendC::BinaryRepeatParams& b128_t1
  ) {
    if (reset_mask) AscendC::Mul<float, true>(lt_u, lt_z[row_off], lt_wscr[3072], 64ULL, 2, b128_t1);
    else AscendC::Mul<float, false>(lt_u, lt_z[row_off], lt_wscr[3072], 64ULL, 2, b128_t1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<float, false>(lt_stage[0], lt_u, lt_u, 64ULL, 2, b128_t1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, false>(lt_red, lt_stage[0], 64, 2, 1, 1, 8);
    if (reset_mask) AscendC::ResetMask();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void ApplyHouseholder128Step(
    uint32_t p, uint32_t P, int32_t ti128, uint32_t ui, uint32_t ul, uint8_t ui_u8, float inv_h, uint32_t& hvec_used,
    const AscendC::LocalTensor<float>& lt_u, const AscendC::LocalTensor<float>& lt_pfull, const AscendC::LocalTensor<float>& lt_stage,
    const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_red,
    const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<uint32_t>& lt_off_zero128, const AscendC::LocalTensor<uint32_t>& lt_off_elem128,
    const AscendC::UnaryRepeatParams& u128_t1, const AscendC::BinaryRepeatParams& b128_t1, const AscendC::BinaryRepeatParams& mv128_t1, const AscendC::BinaryRepeatParams& rk128_t1
  ) {
    float half_inv_h = -0.5f * inv_h;
    if (P > 0u && p == (ui % P)) {
      uint32_t al_u = (ui + 7u) & ~7u;
      AscendC::Adds(lt_mat[10240u + hvec_used], lt_u, 0.0f, static_cast<int32_t>(al_u)); AscendC::PipeBarrier<PIPE_V>();
      hvec_used += al_u;
    }

    AscendC::Duplicate<float, true>(lt_stage[0], 0.0f, 64ULL, 2, 1, 8);
    AscendC::Muls<float, false>(lt_stage[128], lt_u, -inv_h, 64ULL, 2, u128_t1);
    AscendC::Muls<float, false>(lt_stage[256], lt_u, half_inv_h, 64ULL, 2, u128_t1);
    AscendC::Mul<float, false>(lt_mat[0], lt_z[0], lt_u[0], 64ULL, ui_u8, mv128_t1); AscendC::PipeBarrier<PIPE_V>();
    AscendC::MulAddDst<float, float, false>(lt_mat[0], lt_z[64], lt_u[64], 64ULL, ui_u8, mv128_t1); AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, false>(lt_stage[0], lt_mat[0], 64, ti128, 1, 1, 8); AscendC::PipeBarrier<PIPE_V>();

    AscendC::Muls<float, false>(lt_pfull, lt_stage[0], -inv_h, 64ULL, 2, u128_t1);
    AscendC::Mul<float, false>(lt_stage[0], lt_stage[0], lt_stage[128], 64ULL, 2, b128_t1); AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float, false>(lt_stage[0], lt_stage[0], lt_stage[64], 64ULL, 1, b128_t1); AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float, false>(lt_red, lt_stage[0], 64, 1, 1, 1, 8); AscendC::PipeBarrier<PIPE_V>();
    AscendC::Gather(lt_stage[0], lt_red, lt_off_zero128, 0u, 128u); AscendC::PipeBarrier<PIPE_V>();
    AscendC::MulAddDst<float, float, true>(lt_pfull, lt_stage[256], lt_stage[0], 64ULL, 2, b128_t1); AscendC::PipeBarrier<PIPE_V>();

    uint32_t rep8_cnt = ui * 8u;
    AscendC::Gather(lt_wscr[0],    lt_pfull, lt_off_elem128, 0u, rep8_cnt);
    AscendC::Gather(lt_wscr[1024], lt_u,     lt_off_elem128, 0u, rep8_cnt);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::MulAddDst<float, float, true>(lt_z[0],   lt_u[0],  lt_wscr[0], 64ULL, ui_u8, rk128_t1);
    AscendC::MulAddDst<float, float, false>(lt_z[64], lt_u[64], lt_wscr[0], 64ULL, ui_u8, rk128_t1);
    AscendC::PipeBarrier<PIPE_V>();
    if (ul >= 64u) {
      uint32_t row_l = ul * 128u;
      uint8_t ul_u8 = static_cast<uint8_t>(ul);
      AscendC::MulAddDst<float, float, false>(lt_z[row_l],       lt_pfull[0],  lt_wscr[1024u + ul * 8u], 64ULL, 1, rk128_t1);
      AscendC::MulAddDst<float, float, false>(lt_z[row_l + 64u], lt_pfull[64], lt_wscr[1024u + ul * 8u], 64ULL, 1, rk128_t1);
      AscendC::PipeBarrier<PIPE_V>();
      EvalRow128Norm2(row_l, false, lt_u, lt_z, lt_wscr, lt_stage, lt_red, b128_t1);
      AscendC::MulAddDst<float, float, false>(lt_z[0],  lt_pfull[0],  lt_wscr[1024], 64ULL, ul_u8, rk128_t1);
      AscendC::MulAddDst<float, float, false>(lt_z[64], lt_pfull[64], lt_wscr[1024], 64ULL, ul_u8, rk128_t1);
    } else {
      AscendC::MulAddDst<float, float, false>(lt_z[0],  lt_pfull[0],  lt_wscr[1024], 64ULL, ui_u8, rk128_t1);
      AscendC::MulAddDst<float, float, false>(lt_z[64], lt_pfull[64], lt_wscr[1024], 64ULL, ui_u8, rk128_t1);
    }
    AscendC::ResetMask();
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline void InitLargeTail128InUB(
    uint32_t p, uint32_t P, uint32_t stride, uint32_t max_rows, __ubuf__ float* wide_scr,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_u, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z, const AscendC::GlobalTensor<float>& gt_sym
  ) {
    SafeSyncAll();
    for (uint32_t q = 0; q < max_rows; ++q) {
      uint32_t r = q * P + p;
      if (r < 127u) AscendC::DataCopy(gt_sym[r * 128u], lt_mat[q * stride], 128);
    }
    SafeSyncAll();

    AscendC::DataCopy(lt_z[0], gt_sym[0], 127 * 128);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Adds(lt_z[127u * 128u], lt_stage[512], 0.0f, 128);
    AscendC::PipeBarrier<PIPE_V>();

    __ubuf__ uint32_t* off_elem128 = reinterpret_cast<__ubuf__ uint32_t*>(wide_scr + 2048);
    AscendC::LocalTensor<int32_t> lt_off_elem128_i = ub_all.ReinterpretCast<int32_t>()[14 * 512 + 2048];
    for (uint32_t idx = 0; idx < 64u; ++idx) off_elem128[idx] = (idx >> 3u) * 4u;
    for (uint32_t k = 0; k < 128u; ++k) wide_scr[3072u + k] = (k < 126u) ? 1.0f : 0.0f;
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (int32_t rep = 64; rep < 1024; rep <<= 1) {
      AscendC::Adds(lt_off_elem128_i[rep], lt_off_elem128_i[0], (rep >> 3) * 4, rep);
      AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Duplicate(lt_u[128], 0.0f, 384);
    AscendC::Duplicate(lt_stage[384], 0.0f, 128);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::PipeBarrier<PIPE_V>();
  }

  __aicore__ __attribute__((always_inline)) inline void TridiagLargeTail128And64(
    uint32_t p, uint32_t P, uint32_t stride, uint32_t max_rows, uint32_t& hvec_used,
    __ubuf__ float* d, __ubuf__ float* e, __ubuf__ float* u_ub, __ubuf__ float* inv_h_save, __ubuf__ float* red_ub, __ubuf__ float* wide_scr, __ubuf__ float* z_ub,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_u, const AscendC::LocalTensor<float>& lt_pfull, const AscendC::LocalTensor<float>& lt_stage, const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z, const AscendC::LocalTensor<float>& lt_red, const AscendC::LocalTensor<float>& lt_wscr, const AscendC::LocalTensor<uint32_t>& lt_u32, const AscendC::GlobalTensor<float>& gt_sym
  ) {
    InitLargeTail128InUB(p, P, stride, max_rows, wide_scr, ub_all, lt_u, lt_stage, lt_mat, lt_z, gt_sym);
    AscendC::LocalTensor<uint32_t> lt_off_elem128 = lt_u32[14 * 512 + 2048], lt_off_zero128 = lt_stage[384].ReinterpretCast<uint32_t>();
    AscendC::UnaryRepeatParams u128_t1(1, 1, 8, 8);
    AscendC::BinaryRepeatParams b128_t1(1, 1, 1, 8, 8, 8), mv128_t1(1, 1, 1, 8, 16, 0), rk128_t1(1, 1, 0, 16, 0, 1);
    EvalRow128Norm2(127u * 128u, true, lt_u, lt_z, lt_wscr, lt_stage, lt_red, b128_t1);

    for (int32_t ti128 = 127; ti128 >= 64; --ti128) {
      uint32_t ui = static_cast<uint32_t>(ti128), ul = ui - 1u, row_i = ui * 128u;
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      d[ui] = z_ub[row_i + ui];
      float rl_val = z_ub[row_i + ul], nrm2_sub = red_ub[0] + red_ub[1], inv_h = 0.0f, ei = 0.0f;
      if (ul > 0u) wide_scr[3072u + ul - 1u] = 0.0f;
      if (nrm2_sub > 1e-30f) {
        float nrm2 = nrm2_sub + rl_val * rl_val, g = nrm2 * MyRsqrt(nrm2);
        if (rl_val >= 0.0f) g = -g;
        ei = g; inv_h = 1.0f / (nrm2 - rl_val * g); u_ub[ul] = rl_val - g;
      } else {
        ei = rl_val;
      }
      inv_h_save[ui] = inv_h; e[ui] = ei;
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      if (inv_h > 0.0f) {
        ApplyHouseholder128Step(p, P, ti128, ui, ul, static_cast<uint8_t>(ui), inv_h, hvec_used, lt_u, lt_pfull, lt_stage, lt_mat, lt_z, lt_red, lt_wscr, lt_off_zero128, lt_off_elem128, u128_t1, b128_t1, mv128_t1, rk128_t1);
      } else if (ul >= 64u) {
        EvalRow128Norm2(ul * 128u, true, lt_u, lt_z, lt_wscr, lt_stage, lt_red, b128_t1);
      }
    }

    AscendC::Adds<float, true>(lt_mat[0], lt_z[0], 0.0f, 64ULL, 64, AscendC::UnaryRepeatParams(1, 1, 8, 16));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<float, false>(lt_z[0], lt_mat[0], 0.0f, 64ULL, 64, u128_t1);
    AscendC::ResetMask(); AscendC::PipeBarrier<PIPE_V>();

    RunLargeTail64InUB(ub_all);
    for (uint32_t k = 64u; k < n_; ++k) e[k - 1u] = e[k];
    e[n_ - 1u] = 0.0f;
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::Adds(lt_mat[9728], ub_all[5 * 512], 0.0f, 512);
    for (int32_t ti = 63; ti >= 2; --ti) {
      uint32_t uti = static_cast<uint32_t>(ti);
      if (inv_h_save[uti] > 0.0f && p == (uti % P)) { AscendC::Adds(lt_mat[10240u + hvec_used], ub_all[3584u + uti * 64u], 0.0f, 64); hvec_used += 64u; }
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ __attribute__((always_inline)) inline void SolveLargeLeavesAndBlocks64(
    uint32_t p, uint32_t P, uint32_t my_blk, uint32_t plo_b, uint32_t bp_team, uint32_t subP_b, uint32_t sym_cap,
    __gm__ float* cur_vec, __ubuf__ int32_t* bounds, __ubuf__ float* d, __ubuf__ float* e, __ubuf__ float* stage_ub,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_stage
  ) {
    constexpr int32_t M = EighHostPlan<CASE_KEY>::value.M, M_leaf = EighHostPlan<CASE_KEY>::value.M_leaf;
    constexpr uint32_t M_leaf_u = static_cast<uint32_t>(M_leaf), leaf_gm_off = EighHostPlan<CASE_KEY>::value.leaf_gm_off, blk_slot = EighHostPlan<CASE_KEY>::value.blk_slot;
    int32_t leaf_bounds[33];
    #pragma unroll
    for (int32_t lf = 0; lf <= M_leaf; ++lf) leaf_bounds[lf] = EighHostPlan<CASE_KEY>::value.leaf_bounds[lf];
    #pragma unroll
    for (int32_t b = 0; b <= M; ++b) bounds[b] = EighHostPlan<CASE_KEY>::value.bounds[b];

    for (int32_t lf = 1; lf < M_leaf; ++lf) {
      int32_t cut = leaf_bounds[lf];
      float ar = MyAbs(e[cut - 1]);
      d[cut - 1] -= ar; d[cut] -= ar;
    }

    AscendC::GlobalTensor<float> gt_leaf_out;
    gt_leaf_out.SetGlobalBuffer(cur_vec + leaf_gm_off, M_leaf_u * 320u);
    constexpr uint32_t leaf_core_c6[16] = {9u, 0u, 9u, 1u, 0u, 2u, 1u, 3u, 2u, 4u, 3u, 5u, 6u, 7u, 4u, 8u};
    #pragma unroll
    for (int32_t lf = 0; lf < M_leaf; ++lf) {
      uint32_t target_p = (CASE_KEY == 6u) ? leaf_core_c6[lf] : (static_cast<uint32_t>(lf) % P);
      if (target_p == p) {
        int32_t s = leaf_bounds[lf], m = leaf_bounds[lf + 1] - s;
        SolveLeafPipeV16(m, &d[s], &e[s], lt_stage, stage_ub);
        AscendC::DataCopy(gt_leaf_out[static_cast<uint32_t>(lf) * 320u], lt_stage, 320);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
      }
    }
    SafeSyncAll();

    int32_t sb_my = bounds[my_blk], Lb_my = bounds[my_blk + 1] - sb_my, mid_my = (Lb_my > 48) ? 32 : 24;
    float ec0 = e[leaf_bounds[4 * static_cast<int32_t>(my_blk) + 1] - 1];
    float ec1 = e[leaf_bounds[4 * static_cast<int32_t>(my_blk) + 2] - 1];
    float ec2 = e[leaf_bounds[4 * static_cast<int32_t>(my_blk) + 3] - 1];
    MergeBlock64InUB(Lb_my, mid_my, ec0, ec1, ec2, p, my_blk, plo_b, bp_team, subP_b, blk_slot, leaf_gm_off, cur_vec, sym_cap, ub_all);
    SafeSyncAll();
  }

  __aicore__ __attribute__((always_inline)) inline void UnpackLargeBlock64ToCols(
    uint32_t col_start, uint32_t col_end, uint32_t my_cols, uint32_t stride,
    __ubuf__ int32_t* bounds, __ubuf__ float* d, __ubuf__ float* top, __ubuf__ float* bot,
    __ubuf__ uint32_t* off_tr64x16, __ubuf__ uint32_t* off_pfull, __ubuf__ float* z_ub,
    const AscendC::LocalTensor<float>& ub_all, const AscendC::LocalTensor<float>& lt_mat, const AscendC::LocalTensor<float>& lt_z,
    const AscendC::LocalTensor<int32_t>& lt_tr64x16_i, const AscendC::LocalTensor<uint32_t>& lt_off_pfull, const AscendC::GlobalTensor<float>& gt_sym
  ) {
    constexpr int32_t M = EighHostPlan<CASE_KEY>::value.M;
    constexpr uint32_t blk_slot = EighHostPlan<CASE_KEY>::value.blk_slot;
    AscendC::DataCopy(lt_z[0], gt_sym[6144], M * 192);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    #pragma unroll
    for (int32_t b = 0; b <= M; ++b) bounds[b] = EighHostPlan<CASE_KEY>::value.bounds[b];
    for (uint32_t di = 0; di < 64u; ++di) off_tr64x16[di] = di * 64u;
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    for (uint32_t dk = 1u; dk < 16u; ++dk) AscendC::Adds(lt_tr64x16_i[dk * 64u], lt_tr64x16_i[0], static_cast<int32_t>(dk * 4u), 64);
    if (my_cols > 0u) AscendC::Duplicate(lt_mat, 0.0f, static_cast<int32_t>(my_cols * stride));
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    for (int32_t b = 0; b < M; ++b) {
      int32_t s_b = bounds[b], L_b = bounds[b + 1] - s_b;
      __ubuf__ float* b_meta = z_ub + static_cast<uint32_t>(b) * 192u;
      for (int32_t t = 0; t < L_b; ++t) { d[s_b + t] = b_meta[t]; top[s_b + t] = b_meta[64 + t]; bot[s_b + t] = b_meta[128 + t]; }
    }
    if (my_cols > 0u) {
      AscendC::LocalTensor<int32_t> lt_off_pf_s3 = ub_all.ReinterpretCast<int32_t>()[13312];
      for (int32_t b = 0; b < M; ++b) {
        uint32_t s_bu = static_cast<uint32_t>(bounds[b]), L_bu = static_cast<uint32_t>(bounds[b + 1]) - s_bu;
        uint32_t r_act_lo = (s_bu > col_start) ? s_bu : col_start, r_act_hi = (s_bu + L_bu < col_end) ? (s_bu + L_bu) : col_end;
        if (r_act_lo >= r_act_hi) continue;
        uint32_t act_c0 = r_act_lo - col_start, act_cols = r_act_hi - r_act_lo, r_loc0 = r_act_lo - s_bu;
        AscendC::DataCopy(lt_z[4096], gt_sym[8192u + static_cast<uint32_t>(b) * blk_slot], static_cast<int32_t>(L_bu * 64u));
        for (uint32_t col = 0; col < L_bu; ++col) off_pfull[col] = (col * 64u + r_loc0) * 4u;
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t c = 0; c < act_cols; ++c) {
          if (c > 0u) { AscendC::Adds(lt_off_pf_s3, lt_off_pf_s3, 4, static_cast<int32_t>(L_bu)); AscendC::PipeBarrier<PIPE_V>(); }
          AscendC::Gather(lt_mat[(act_c0 + c) * stride + s_bu], lt_z[4096], lt_off_pfull, 0u, L_bu);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      }
    }
    SafeSyncAll();
  }
