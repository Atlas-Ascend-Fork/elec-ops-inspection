/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 黎何.
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

namespace FusedKeypointBranches {
template <uint64_t KP_PAIR_BITS>
__aicore__ inline void InitCase5UniformKeypoints(AscendC::LocalTensor<float>& outKpLocal)
{
    AscendC::LocalTensor<uint64_t> kpU64 = outKpLocal.ReinterpretCast<uint64_t>();
    kpU64.SetValue(0, KP_PAIR_BITS);
    kpU64.SetValue(1, KP_PAIR_BITS);
    kpU64.SetValue(2, KP_PAIR_BITS);
    kpU64.SetValue(3, KP_PAIR_BITS);
    kpU64.SetValue(4, KP_PAIR_BITS);
    kpU64.SetValue(5, KP_PAIR_BITS);
    kpU64.SetValue(6, KP_PAIR_BITS);
    kpU64.SetValue(7, KP_PAIR_BITS);
    kpU64.SetValue(8, KP_PAIR_BITS);
    kpU64.SetValue(9, KP_PAIR_BITS);
    kpU64.SetValue(10, KP_PAIR_BITS);
    kpU64.SetValue(11, KP_PAIR_BITS);
    kpU64.SetValue(12, KP_PAIR_BITS);
    kpU64.SetValue(13, KP_PAIR_BITS);
    kpU64.SetValue(14, KP_PAIR_BITS);
    kpU64.SetValue(15, KP_PAIR_BITS);
}

__aicore__ inline void ReduceCase5WeightMagnitude(AscendC::LocalTensor<float>& absLocal,
    AscendC::LocalTensor<float>& wLocal,
    AscendC::LocalTensor<float>& reduceTmpLocal,
    AscendC::LocalTensor<float>& scalarLocal)
{
    AscendC::Abs<float>(absLocal, wLocal, 1152);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float>(absLocal[0], absLocal[0], absLocal[512], 512);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float>(absLocal[0], absLocal[0], absLocal[1024], 128);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float>(absLocal[0], absLocal[0], absLocal[256], 256);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float>(absLocal[0], absLocal[0], absLocal[128], 128);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add<float>(absLocal[0], absLocal[0], absLocal[64], 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax<float>(scalarLocal, absLocal, reduceTmpLocal, 64, false);
}

__aicore__ inline void InitCase5UniformScores(AscendC::LocalTensor<float>& outScLocal,
    AscendC::LocalTensor<float>& bLocal)
{
    AscendC::LocalTensor<uint64_t> scU64 = outScLocal.ReinterpretCast<uint64_t>();
    AscendC::LocalTensor<uint64_t> bU64 = bLocal.ReinterpretCast<uint64_t>();
    uint64_t b01 = bU64.GetValue(0);
    uint64_t b23 = bU64.GetValue(1);
    scU64.SetValue(0, b01);
    scU64.SetValue(1, b23);
    scU64.SetValue(2, b01);
    scU64.SetValue(3, b23);
    scU64.SetValue(4, b01);
    scU64.SetValue(5, b23);
    scU64.SetValue(6, b01);
    scU64.SetValue(7, b23);
}

template <uint32_t H_TPL, uint32_t W_TPL>
__aicore__ inline void FoldCase5Rows(AscendC::LocalTensor<float>& xLocal,
    AscendC::LocalTensor<float>& shiftLocal)
{
    uint32_t remH = H_TPL;
    uint32_t half0 = remH >> 1;
    uint32_t odd0 = remH - (half0 << 1);
    uint32_t stride0 = half0 + odd0;
    uint32_t srcOff0 = stride0 * W_TPL;
    int32_t halfCount0 = static_cast<int32_t>(half0 * W_TPL);
    AscendC::Muls<float>(
        xLocal[0],
        shiftLocal[srcOff0],
        static_cast<float>(static_cast<int32_t>(stride0)),
        halfCount0);
    AscendC::Add<float>(shiftLocal[0], shiftLocal[0], shiftLocal[srcOff0], halfCount0);
    AscendC::PipeBarrier<PIPE_V>();
    remH = stride0;
    while (remH > 1) {
        uint32_t half = remH >> 1;
        uint32_t odd = remH - (half << 1);
        uint32_t stride = half + odd;
        uint32_t srcOff = stride * W_TPL;
        int32_t halfCount = static_cast<int32_t>(half * W_TPL);
        AscendC::Axpy<float, float>(
            xLocal[0],
            shiftLocal[srcOff],
            static_cast<float>(static_cast<int32_t>(stride)),
            halfCount);
        AscendC::Add<float>(shiftLocal[0], shiftLocal[0], shiftLocal[srcOff], halfCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(xLocal[0], xLocal[0], xLocal[srcOff], halfCount);
        AscendC::PipeBarrier<PIPE_V>();
        remH = stride;
    }
}
}
