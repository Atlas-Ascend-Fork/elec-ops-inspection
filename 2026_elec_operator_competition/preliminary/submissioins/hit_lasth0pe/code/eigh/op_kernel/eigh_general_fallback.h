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

  struct EighFallbackCtx {
    uint32_t n;
    uint32_t uplo;
    __gm__ float* cur_A;
    __gm__ float* cur_val;
    __gm__ float* cur_vec;
    __ubuf__ float* d;
    __ubuf__ float* e;
    __ubuf__ float* u;
  };

  __aicore__ inline float FallbackSymmetrizeRow(const EighFallbackCtx& ctx, uint32_t row_i) {
    float row_max = 0.0f;
    uint32_t n = ctx.n;
    for (uint32_t j = 0u; j <= row_i; ++j) {
      uint32_t src_idx = (ctx.uplo == 0u) ? (row_i * n + j) : (j * n + row_i);
      float val = ctx.cur_A[src_idx];
      ctx.cur_vec[row_i * n + j] = val;
      ctx.cur_vec[j * n + row_i] = val;
      float aval = MyAbs(val);
      row_max = (aval > row_max) ? aval : row_max;
    }
    return row_max;
  }

  __aicore__ inline float FallbackSymmetrizeAndNorm(const EighFallbackCtx& ctx) {
    float anorm = 0.0f;
    for (uint32_t i = 0u; i < ctx.n; ++i) {
      float rmax = FallbackSymmetrizeRow(ctx, i);
      if (rmax > anorm) {
        anorm = rmax;
      }
    }
    return anorm;
  }

  __aicore__ inline void FallbackWriteZeroIdentity(const EighFallbackCtx& ctx) {
    uint32_t n = ctx.n;
    for (uint32_t i = 0u; i < n; ++i) {
      ctx.cur_val[i] = 0.0f;
      for (uint32_t j = 0u; j < n; ++j) {
        ctx.cur_vec[i * n + j] = (i == j) ? 1.0f : 0.0f;
      }
    }
  }

  __aicore__ inline float FallbackSymMatVecStep(const EighFallbackCtx& ctx, int32_t l, float inv_h) {
    uint32_t n = ctx.n;
    for (int32_t j = 0; j <= l; ++j) {
      ctx.e[j] = 0.0f;
    }
    for (int32_t j = 0; j <= l; ++j) {
      float uj = ctx.u[j];
      float g_acc = 0.0f;
      uint32_t row_off = static_cast<uint32_t>(j) * n;
      for (int32_t k = 0; k < j; ++k) {
        float ajk = ctx.cur_vec[row_off + static_cast<uint32_t>(k)];
        g_acc += ajk * ctx.u[k];
        ctx.e[k] += ajk * uj;
      }
      ctx.e[j] += ctx.cur_vec[row_off + static_cast<uint32_t>(j)] * uj + g_acc;
    }
    float f_sum = 0.0f;
    for (int32_t j = 0; j <= l; ++j) {
      ctx.e[j] *= inv_h;
      f_sum += ctx.e[j] * ctx.u[j];
    }
    return f_sum;
  }

  __aicore__ inline void FallbackRank2UpdateStep(const EighFallbackCtx& ctx, int32_t l, float hh) {
    uint32_t n = ctx.n;
    for (int32_t j = 0; j <= l; ++j) {
      float f_val = ctx.u[j];
      float g_val = ctx.e[j] - hh * f_val;
      ctx.e[j] = g_val;
      uint32_t row_off = static_cast<uint32_t>(j) * n;
      for (int32_t k = 0; k <= j; ++k) {
        ctx.cur_vec[row_off + static_cast<uint32_t>(k)] -= (f_val * ctx.e[k] + g_val * ctx.u[k]);
      }
    }
  }

  __aicore__ inline void FallbackApplyReflectorToRow(
    const EighFallbackCtx& ctx, int32_t j, int32_t l, float inv_h
  ) {
    uint32_t row_off = static_cast<uint32_t>(j) * ctx.n;
    float dot = 0.0f;
    for (int32_t k = 0; k <= l; ++k) {
      dot += ctx.cur_vec[row_off + static_cast<uint32_t>(k)] * ctx.u[k];
    }
    float factor = dot * inv_h;
    for (int32_t k = 0; k <= l; ++k) {
      ctx.cur_vec[row_off + static_cast<uint32_t>(k)] -= factor * ctx.u[k];
    }
  }

  __aicore__ inline void FallbackAccumulateQStep(const EighFallbackCtx& ctx, int32_t i, float h_i) {
    uint32_t n = ctx.n;
    uint32_t ui = static_cast<uint32_t>(i);
    int32_t l = i - 1;
    if (h_i != 0.0f) {
      float inv_h = 1.0f / h_i;
      for (int32_t k = 0; k <= l; ++k) {
        ctx.u[k] = ctx.cur_vec[ui * n + static_cast<uint32_t>(k)];
      }
      for (int32_t j = 0; j <= l; ++j) {
        FallbackApplyReflectorToRow(ctx, j, l, inv_h);
      }
    }
    ctx.cur_vec[ui * n + ui] = 1.0f;
    for (int32_t j = 0; j <= l; ++j) {
      uint32_t uj = static_cast<uint32_t>(j);
      ctx.cur_vec[uj * n + ui] = 0.0f;
      ctx.cur_vec[ui * n + uj] = 0.0f;
    }
  }

  __aicore__ inline float FallbackEliminateRow(const EighFallbackCtx& ctx, int32_t i) {
    uint32_t n = ctx.n;
    uint32_t ui = static_cast<uint32_t>(i);
    int32_t l = i - 1;
    float nrm2_sub = 0.0f;
    for (int32_t k = 0; k < l; ++k) {
      float v = ctx.cur_vec[ui * n + static_cast<uint32_t>(k)];
      nrm2_sub += v * v;
    }
    float f = ctx.cur_vec[ui * n + static_cast<uint32_t>(l)];
    if (l <= 0 || nrm2_sub <= 1e-30f) {
      ctx.e[i] = f;
      return 0.0f;
    }
    float nrm2 = nrm2_sub + f * f;
    float g = (f >= 0.0f) ? -(nrm2 * MyRsqrt(nrm2)) : (nrm2 * MyRsqrt(nrm2));
    ctx.e[i] = g;
    float h = nrm2 - f * g;
    ctx.cur_vec[ui * n + static_cast<uint32_t>(l)] = f - g;
    for (int32_t k = 0; k <= l; ++k) {
      ctx.u[k] = ctx.cur_vec[ui * n + static_cast<uint32_t>(k)];
    }
    float f_sum = FallbackSymMatVecStep(ctx, l, (h != 0.0f) ? (1.0f / h) : 0.0f);
    FallbackRank2UpdateStep(ctx, l, (h != 0.0f) ? (f_sum / (h + h)) : 0.0f);
    return h;
  }

  __aicore__ inline void FallbackTridiagonalize(const EighFallbackCtx& ctx) {
    uint32_t n = ctx.n;
    for (int32_t i = static_cast<int32_t>(n) - 1; i > 0; --i) {
      ctx.d[i] = FallbackEliminateRow(ctx, i);
    }
    ctx.d[0] = ctx.cur_vec[0];
    ctx.cur_vec[0] = 1.0f;
    ctx.e[0] = 0.0f;
    for (int32_t i = 1; i < static_cast<int32_t>(n); ++i) {
      float h_i = ctx.d[i];
      uint32_t ui = static_cast<uint32_t>(i);
      ctx.d[i] = ctx.cur_vec[ui * n + ui];
      FallbackAccumulateQStep(ctx, i, h_i);
    }
    for (uint32_t i = 1u; i < n; ++i) {
      ctx.e[i - 1u] = ctx.e[i];
    }
    ctx.e[n - 1u] = 0.0f;
  }

  __aicore__ inline uint32_t FallbackFindSplitIndex(const EighFallbackCtx& ctx, uint32_t l) {
    uint32_t m = l;
    while (m + 1u < ctx.n) {
      float dd = MyAbs(ctx.d[m]) + MyAbs(ctx.d[m + 1u]);
      float em = MyAbs(ctx.e[m]);
      if (em <= 2.5e-7f * dd || em <= 1e-24f) {
        ctx.e[m] = 0.0f;
        break;
      }
      m++;
    }
    return m;
  }

  __aicore__ inline void FallbackRotateVecPair(
    const EighFallbackCtx& ctx, uint32_t row_i, float c, float s
  ) {
    uint32_t n = ctx.n;
    uint32_t base0 = row_i * n;
    uint32_t base1 = (row_i + 1u) * n;
    for (uint32_t k = 0u; k < n; ++k) {
      float fz = ctx.cur_vec[base1 + k];
      float zik = ctx.cur_vec[base0 + k];
      ctx.cur_vec[base1 + k] = s * zik + c * fz;
      ctx.cur_vec[base0 + k] = c * zik - s * fz;
    }
  }

  __aicore__ inline void FallbackRunQlSweep(
    const EighFallbackCtx& ctx, uint32_t l, uint32_t m, float g_init
  ) {
    float g = g_init;
    float s = 1.0f;
    float c = 1.0f;
    float p = 0.0f;
    for (int32_t i = static_cast<int32_t>(m) - 1; i >= static_cast<int32_t>(l); --i) {
      float f = s * ctx.e[i];
      float b = c * ctx.e[i];
      float af = MyAbs(f);
      float ag = MyAbs(g);
      if (af == 0.0f && ag == 0.0f) {
        ctx.d[i + 1] -= p;
        ctx.e[m] = 0.0f;
        return;
      }
      bool f_ge_g = (af >= ag);
      float big = f_ge_g ? f : g;
      float sml = f_ge_g ? g : f;
      float ratio = (big != 0.0f) ? (sml / big) : 0.0f;
      float hyp2 = 1.0f + ratio * ratio;
      float inv_r = MyRsqrt(hyp2);
      float prod_r = ratio * inv_r;
      ctx.e[i + 1] = big * (hyp2 * inv_r);
      s = f_ge_g ? inv_r : prod_r;
      c = f_ge_g ? prod_r : inv_r;
      g = ctx.d[i + 1] - p;
      float r_rot = (ctx.d[i] - g) * s + 2.0f * c * b;
      p = s * r_rot;
      ctx.d[i + 1] = g + p;
      g = c * r_rot - b;
      FallbackRotateVecPair(ctx, static_cast<uint32_t>(i), c, s);
    }
    ctx.d[l] -= p;
    ctx.e[l] = g;
    ctx.e[m] = 0.0f;
  }

  __aicore__ inline void FallbackSolveTridiagQl(const EighFallbackCtx& ctx) {
    for (uint32_t l = 0u; l < ctx.n; ++l) {
      uint32_t iter = 0u;
      while (iter < 80u) {
        uint32_t m = FallbackFindSplitIndex(ctx, l);
        if (m == l) {
          break;
        }
        ++iter;
        float g_init = ComputeWilkinsonShift(ctx.d[l], ctx.d[l + 1u], ctx.e[l], ctx.d[m]);
        FallbackRunQlSweep(ctx, l, m, g_init);
      }
    }
  }

  __aicore__ inline void FallbackSwapVecRows(const EighFallbackCtx& ctx, uint32_t r0, uint32_t r1) {
    uint32_t n = ctx.n;
    for (uint32_t j = 0u; j < n; ++j) {
      float tmp = ctx.cur_vec[r0 * n + j];
      ctx.cur_vec[r0 * n + j] = ctx.cur_vec[r1 * n + j];
      ctx.cur_vec[r1 * n + j] = tmp;
    }
  }

  __aicore__ inline void FallbackSortEigenPairs(const EighFallbackCtx& ctx) {
    uint32_t n = ctx.n;
    for (uint32_t i = 0u; i + 1u < n; ++i) {
      uint32_t k = i;
      float p_min = ctx.d[i];
      for (uint32_t j = i + 1u; j < n; ++j) {
        if (ctx.d[j] < p_min) {
          k = j;
          p_min = ctx.d[j];
        }
      }
      if (k != i) {
        ctx.d[k] = ctx.d[i];
        ctx.d[i] = p_min;
        FallbackSwapVecRows(ctx, i, k);
      }
    }
  }

  __aicore__ inline void FallbackNormalizeVecRow(const EighFallbackCtx& ctx, uint32_t row_i) {
    uint32_t n = ctx.n;
    uint32_t base = row_i * n;
    uint32_t max_k = 0u;
    float max_abs = MyAbs(ctx.cur_vec[base]);
    for (uint32_t j = 1u; j < n; ++j) {
      float a = MyAbs(ctx.cur_vec[base + j]);
      if (a > max_abs) {
        max_abs = a;
        max_k = j;
      }
    }
    if (ctx.cur_vec[base + max_k] < 0.0f) {
      for (uint32_t j = 0u; j < n; ++j) {
        ctx.cur_vec[base + j] = -ctx.cur_vec[base + j];
      }
    }
  }

  __aicore__ inline void FallbackFinalizeOutput(const EighFallbackCtx& ctx, float scale_pow2) {
    uint32_t n = ctx.n;
    for (uint32_t i = 0u; i < n; ++i) {
      ctx.d[i] *= scale_pow2;
    }
    FallbackSortEigenPairs(ctx);
    for (uint32_t i = 0u; i < n; ++i) {
      FallbackNormalizeVecRow(ctx, i);
      ctx.cur_val[i] = ctx.d[i];
    }
    for (uint32_t i = 0u; i < n; ++i) {
      for (uint32_t j = i + 1u; j < n; ++j) {
        float tmp = ctx.cur_vec[i * n + j];
        ctx.cur_vec[i * n + j] = ctx.cur_vec[j * n + i];
        ctx.cur_vec[j * n + i] = tmp;
      }
    }
  }

  __aicore__ __attribute__((noinline)) void SolveGeneralBatch(
    uint32_t n, uint32_t uplo, __gm__ float* cur_A, __gm__ float* cur_val, __gm__ float* cur_vec
  ) {
    auto ub_all = ub_buf_.Get<float>();
    __ubuf__ float* base_ptr = reinterpret_cast<__ubuf__ float*>(ub_all.GetPhyAddr());
    EighFallbackCtx ctx{n, uplo, cur_A, cur_val, cur_vec, base_ptr, base_ptr + 512, base_ptr + 1024};
    float anorm = FallbackSymmetrizeAndNorm(ctx);
    if (anorm == 0.0f) {
      FallbackWriteZeroIdentity(ctx);
      return;
    }
    float scale_pow2 = 1.0f;
    float inv_scale_pow2 = 1.0f;
    GetPow2Scale(anorm, scale_pow2, inv_scale_pow2);
    for (uint32_t i = 0u; i < n * n; ++i) {
      cur_vec[i] *= inv_scale_pow2;
    }
    FallbackTridiagonalize(ctx);
    FallbackSolveTridiagQl(ctx);
    FallbackFinalizeOutput(ctx, scale_pow2);
  }

  __aicore__ __attribute__((noinline)) void SolveGeneralFallback(
    __gm__ float* gm_A, __gm__ float* gm_val, __gm__ float* gm_vec
  ) {
    uint32_t block_id = AscendC::GetBlockIdx();
    uint32_t num_blocks = Plan().used_cores;
    if (num_blocks == 0u) {
      num_blocks = 1u;
    }
    if (block_id >= num_blocks) {
      return;
    }
    uint32_t total_batch = Plan().total_batch;
    uint32_t n = Plan().n;
    uint32_t uplo = Plan().uplo;
    uint32_t b_start = (num_blocks > 0u) ? ((block_id * total_batch) / num_blocks) : 0u;
    uint32_t b_end = (num_blocks > 0u) ? (((block_id + 1u) * total_batch) / num_blocks) : total_batch;
    for (uint32_t b = b_start; b < b_end; ++b) {
      uint64_t mat_off = static_cast<uint64_t>(b) * n * n;
      uint64_t val_off = static_cast<uint64_t>(b) * n;
      SolveGeneralBatch(n, uplo, gm_A + mat_off, gm_val + val_off, gm_vec + mat_off);
    }
    AscendC::GlobalTensor<float> gm_out;
    gm_out.SetGlobalBuffer(gm_val);
    AscendC::DataCacheCleanAndInvalid<
      float, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(gm_out);
  }
