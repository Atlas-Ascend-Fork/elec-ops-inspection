/*
 * Copyright (c) 2026 Beizhi contributors.
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

#pragma once

#include "kernel_operator.h"
#include <cmath>
#include "lib/matmul_intf.h"
#include "linalg_pinv_tiling.h"
#include "tiling_key_linalg_pinv.h"
using namespace AscendC;
using FP32Cube =
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float>, MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>>;

struct PinvBatchAssessment {
    float needs = 0;
    float unbounded = 0;
    bool accept = false;
    bool myOk = false;
    bool sensitive = false;
};

template <HardEvent H> __aicore__ inline void LocalSync()
{
    SetFlag<H>(EVENT_ID0);
    WaitFlag<H>(EVENT_ID0);
}

__aicore__ inline void FastTranspose8x256(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float> scratch)
{
    LocalTensor<float> destinations[16];
    LocalTensor<float> sources[16];
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        destinations[2 * i] = dst[i * 8];
        destinations[2 * i + 1] = scratch[i * 8];
        sources[i] = src[i * 256];
        sources[i + 8] = src[i * 256];
    }
    TransDataTo5HD(destinations, sources, {false, false, 32, 8, 1});
}

__aicore__ inline void FastTranspose16x256(LocalTensor<float> dst, LocalTensor<float> src)
{
    LocalTensor<float> destinations[16];
    LocalTensor<float> sources[16];
#pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        destinations[2 * i] = dst[i * 16];
        destinations[2 * i + 1] = dst[i * 16 + 8];
        sources[i] = src[i * 256];
        sources[i + 8] = src[(i + 8) * 256];
    }
    TransDataTo5HD(destinations, sources, {false, false, 32, 16, 1});
}
