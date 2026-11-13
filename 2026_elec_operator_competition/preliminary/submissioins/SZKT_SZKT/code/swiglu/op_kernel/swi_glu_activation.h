/*
 * Copyright (c) 2026 Beijing Kuntai Information Technology Co., Ltd.
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

namespace swiglu {
using namespace AscendC;

// 输入为负Gate，输出的负号由负Up在最终乘法中抵消。
__aicore__ inline void NegativeSilu(LocalTensor<half> gate, LocalTensor<half> input,
                                   uint32_t count) {
    const UnaryRepeatParams unary;
    const BinaryRepeatParams binary;
    SetMaskCount();
    SetVectorMask<half>(0, count);
    Exp<half, false>(gate, input, MASK_PLACEHOLDER, 1, unary);
    PipeBarrier<PIPE_V>();
    Adds<half, false>(gate, gate, half(1), MASK_PLACEHOLDER, 1, unary);
    PipeBarrier<PIPE_V>();
    Div<half, false>(gate, input, gate, MASK_PLACEHOLDER, 1, binary);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

} // namespace swiglu
