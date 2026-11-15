/*
 * Copyright (c) 2026 dhltat.
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
#include "c_api/asc_simd.h"
#include "spline_conv_tiling.h"
#include "tiling_key_spline_conv.h"
using namespace AscendC;

// CI/CO template arguments count input/output channels. Specializations are
// complete alignment/capacity families selected by metadata on the host.
// Every active route computes each output within this physical kernel call.

template <HardEvent E> __aicore__ inline void Fence() {
    SetFlag<E>(0);
    WaitFlag<E>(0);
}
// A direct output owner assigns one 32-byte-aligned UB slab and completes
// each DMA before returning. Buffer counts below are element counts.
template <class U>
__aicore__ inline LocalTensor<U> DirectUbAlloc(uint32_t &offsetBytes, uint32_t elementCount) {
    LocalTensor<U> tensor(TPosition::VECCALC, offsetBytes, elementCount);
    const uint32_t allocationBytes =
        (elementCount * sizeof(U) + kUbAlignmentBytes - 1) / kUbAlignmentBytes * kUbAlignmentBytes;
    offsetBytes += allocationBytes;
    return tensor;
}
