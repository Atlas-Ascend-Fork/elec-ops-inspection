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

// Kernel侧核函数实现
#pragma clang fp reassociate(off) contract(off)

#include "kernel_operator.h"

#include "linalg_eigh_tiling.h"
#include "tiling_key_linalg_eigh.h"
#include "eigh_case_traits.h"
#include "../op_host/eigh_host_plan.h"

__aicore__ inline float MyAbs(float x) {
  return (x < 0.0f) ? -x : x;
}

__aicore__ inline float MyRsqrt(float x) {
  if (x <= 0.0f) {
    return 0.0f;
  }
  uint32_t i = __builtin_bit_cast(uint32_t, x);
  i = 0x5f375a86u - (i >> 1);
  float y = __builtin_bit_cast(float, i);
  float hx = 0.5f * x;
  y = y * (1.5f - hx * (y * y));
  y = y * (1.5f - hx * (y * y));
  y = y * (1.5f - hx * (y * y));
  return y;
}

__aicore__ inline float RsqrtUnit12(float t) {
  float y = 1.573691f + t * (-0.722251f + 0.144970f * t);
  float ht = 0.5f * t;
  y = y * (1.5f - ht * (y * y));
  y = y * (1.5f - ht * (y * y));
  return y;
}

__aicore__ inline float MySqrt(float x) {
  return (x <= 0.0f) ? 0.0f : (x * MyRsqrt(x));
}

__aicore__ inline float LdexpF32(float x, int32_t exp_shift) {
  if (x == 0.0f || exp_shift == 0) {
    return x;
  }
  while (exp_shift >= 64) {
    x *= 18446744073709551616.0f;
    exp_shift -= 64;
  }
  while (exp_shift <= -64) {
    x *= 5.42101086242752217e-20f;
    exp_shift += 64;
  }
  if (exp_shift != 0) {
    uint32_t ui = static_cast<uint32_t>((127 + exp_shift) << 23);
    x *= __builtin_bit_cast(float, ui);
  }
  return x;
}

__aicore__ inline bool IsFiniteF32(float x) {
  uint32_t ui = __builtin_bit_cast(uint32_t, x);
  return (ui & 0x7f800000u) != 0x7f800000u;
}

__aicore__ inline void GetPow2Scale(float anorm, float &scale_pow2, float &inv_scale_pow2) {
  if (anorm <= 0.0f) {
    scale_pow2 = 1.0f;
    inv_scale_pow2 = 1.0f;
    return;
  }
  uint32_t exp_bits = __builtin_bit_cast(uint32_t, anorm) & 0x7f800000u;
  const uint32_t min_exp = 27u << 23;
  const uint32_t max_exp = 227u << 23;
  if (exp_bits < min_exp) {
    exp_bits = min_exp;
  }
  if (exp_bits > max_exp) {
    exp_bits = max_exp;
  }
  uint32_t inv_exp_bits = (254u << 23) - exp_bits;
  scale_pow2 = __builtin_bit_cast(float, exp_bits);
  inv_scale_pow2 = __builtin_bit_cast(float, inv_exp_bits);
}

__aicore__ inline void SortIdxF32Local(int32_t len, int32_t* idx, const float* val) {
  for (int32_t i = 0; i < len; ++i) {
    idx[i] = i;
  }
  int32_t gap = 1;
  while (gap * 3 + 1 < len) {
    gap = gap * 3 + 1;
  }
  while (gap > 0) {
    for (int32_t i = gap; i < len; ++i) {
      int32_t v = idx[i];
      float vv = val[v];
      int32_t j = i;
      while (j >= gap) {
        int32_t u = idx[j - gap];
        float uv = val[u];
        bool less = (vv < uv) || (vv == uv && v < u);
        if (!less) {
          break;
        }
        idx[j] = u;
        j -= gap;
      }
      idx[j] = v;
    }
    gap /= 3;
  }
}

__aicore__ __attribute__((always_inline)) inline void MergeTwoSortedLocal(
  int32_t m1,
  int32_t L,
  int32_t* idx_out,
  const float* d_in
) {
  int32_t i = 0;
  int32_t j = m1;
  int32_t k = 0;
  while (i < m1 && j < L) {
    if (d_in[i] <= d_in[j]) {
      idx_out[k++] = i++;
    } else {
      idx_out[k++] = j++;
    }
  }
  while (i < m1) {
    idx_out[k++] = i++;
  }
  while (j < L) {
    idx_out[k++] = j++;
  }
}

__aicore__ inline void MergeTwoSortedF32(
  int32_t m1,
  int32_t L,
  __ubuf__ int32_t* idx_ub,
  const __ubuf__ float* d_ub
) {
  float dl[256];
  int32_t out[256];
  for (int32_t i = 0; i < L; ++i) {
    dl[i] = d_ub[i];
  }
  MergeTwoSortedLocal(m1, L, out, dl);
  for (int32_t t = 0; t < L; ++t) {
    idx_ub[t] = out[t];
  }
}

__aicore__ inline void SafeSyncAll() {
  AscendC::PipeBarrier<PIPE_ALL>();
  AscendC::SyncAll<true>();
  AscendC::PipeBarrier<PIPE_ALL>();
}

template <class DT_A,uint32_t CASE_KEY>
class KernelLinalgEigh {
public:
  __aicore__ inline KernelLinalgEigh() {}

  __aicore__ inline const __gm__ LinalgEighTilingData& Plan() const {
    return *reinterpret_cast<__gm__ LinalgEighTilingData*>(gm_ws_);
  }
  __aicore__ inline void Init(GM_ADDR A,GM_ADDR eigenvalues,GM_ADDR eigenvectors,GM_ADDR tiling) {
    gm_A_=reinterpret_cast<__gm__ float*>(A);
    gm_val_=reinterpret_cast<__gm__ float*>(eigenvalues);
    gm_vec_=reinterpret_cast<__gm__ float*>(eigenvectors);
    gm_ws_=reinterpret_cast<__gm__ float*>(tiling);
    pipe_.InitBuffer(ub_buf_,EighHostPlan<CASE_KEY>::value.ub_floats*sizeof(float));
  }
  __aicore__ inline void Process() {
    auto ub_all=ub_buf_.Get<float>();
    if constexpr (EighCase<CASE_KEY>::Solver==0u) {
      SolveSmallL1(gm_A_,gm_val_,gm_vec_,ub_all);
    } else if constexpr (EighCase<CASE_KEY>::Solver==2u) {
      const __gm__ EighCorePlan &cp=Plan().cores[AscendC::GetBlockIdx()];
      SolveMediumCoop8(gm_A_+cp.a_offset,gm_val_+cp.val_offset,gm_vec_+cp.a_offset,nullptr,ub_all,cp.p);
    } else if constexpr (EighCase<CASE_KEY>::Solver==3u) {
      const __gm__ EighCorePlan &cp=Plan().cores[AscendC::GetBlockIdx()];
      SolveLargeCoopStedc(gm_A_+cp.a_offset,gm_val_+cp.val_offset,gm_vec_+cp.a_offset,nullptr,ub_all,cp.p,EighCase<CASE_KEY>::P,true);
    }
  }

private:
  #define EIGH_PAD4()   AscendC::PipeBarrier<PIPE_S>();
  #define EIGH_PAD16()  EIGH_PAD4()  EIGH_PAD4()  EIGH_PAD4()  EIGH_PAD4()
  #define EIGH_PAD64()  EIGH_PAD16() EIGH_PAD16() EIGH_PAD16() EIGH_PAD16()
  #define EIGH_PAD256() EIGH_PAD64() EIGH_PAD64() EIGH_PAD64() EIGH_PAD64()

  #include "eigh_leaf_small.h"
  #include "eigh_leaf_small_b.h"
  #include "eigh_secular_common.h"
  #include "eigh_secular_common_b.h"
  #include "eigh_medium_coop.h"
  #include "eigh_medium_coop_b.h"
  #include "eigh_medium_coop_c.h"
  #include "eigh_large_block64.h"
  #include "eigh_large_tridiag.h"
  #include "eigh_large_tridiag_b.h"
  #include "eigh_large_stedc.h"
  #include "eigh_large_stedc_b.h"
  #include "eigh_large_stedc_c.h"

private:
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> ub_buf_;

  static constexpr uint32_t n_=EighCase<CASE_KEY>::N;
  static constexpr uint32_t total_batch_=EighCase<CASE_KEY>::B;
  static constexpr uint32_t uplo_=EighCase<CASE_KEY>::U;

  __gm__ float* gm_A_{nullptr};
  __gm__ float* gm_val_{nullptr};
  __gm__ float* gm_vec_{nullptr};
  __gm__ float* gm_ws_{nullptr};
};

template<typename DT_A,uint32_t CASE_KEY>
__global__ __aicore__ void linalg_eigh(GM_ADDR A,GM_ADDR eigenvalues,GM_ADDR eigenvectors,GM_ADDR workspace,GM_ADDR tiling) {
  if ASCEND_IS_AIC { return; }
  REGISTER_TILING_DEFAULT(LinalgEighTilingData);
  KernelLinalgEigh<DT_A,CASE_KEY> op;
  op.Init(A,eigenvalues,eigenvectors,tiling);
  op.Process();
}
