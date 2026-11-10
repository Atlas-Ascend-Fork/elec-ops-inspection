/*
 * Copyright (c) 2026 aodebiao.
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
// Tiling结构体定义的头文件
#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// Dual-projection SwiGLU:
//   a = input_x[M, K] @ weight_1 + bias_1
//   b = input_x[M, K] @ weight_2 + bias_2
//   output = SiLU(a) * b
//
// Weight storage can be either [K, N] (transpose flag = 0) or
// [N, K] (transpose flag = 1). Host tiling detects the layout.
static constexpr uint32_t SWIGLU_OUT_TILE = 16U;

struct SwiGluTilingData {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t weight1Transposed;
    uint32_t weight2Transposed;
    uint32_t blockDim;
    uint32_t useCube;
    uint32_t tileM;
    uint32_t tileN;
    AscendC::tiling::TCubeTiling cube;
};
