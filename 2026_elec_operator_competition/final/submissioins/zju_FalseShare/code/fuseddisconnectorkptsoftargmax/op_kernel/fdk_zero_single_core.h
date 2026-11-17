/*
 * Copyright (c) 2026 FalseShare Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Apache License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FDK_ZERO_SINGLE_CORE_H
#define FDK_ZERO_SINGLE_CORE_H

#include "kernel_operator.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"

// The caller selects one AIV for B4/H48/W64/tau1 from metadata alone. All
// tensor-dependent work stays here. False requires a sequential 16-plane
// general fallback; true emits both complete outputs with two contiguous DMAs.
__aicore__ inline bool FdkTrySingleCoreZeroWeights(
    GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores,
    const FusedDisconnectorKeypointSoftArgmaxTilingData &t) {
    using namespace AscendC;
    TPipe pipe;
    TBuf<QuePosition::VECCALC> weightBuf, scratchBuf, reducedBuf, biasBuf, keypointsBuf, scoresBuf;
    pipe.InitBuffer(weightBuf, 4 * 32 * 9 * sizeof(float));
    pipe.InitBuffer(scratchBuf, 4 * 32 * 9 * sizeof(float));
    pipe.InitBuffer(reducedBuf, 32);
    pipe.InitBuffer(biasBuf, 32);
    pipe.InitBuffer(keypointsBuf, 32 * sizeof(float));
    pipe.InitBuffer(scoresBuf, 16 * sizeof(float));
    auto weights = weightBuf.Get<float>();
    auto scratch = scratchBuf.Get<float>();
    auto reduced = reducedBuf.Get<float>();
    auto biases = biasBuf.Get<float>();
    auto points = keypointsBuf.Get<float>();
    auto values = scoresBuf.Get<float>();
    GlobalTensor<float> weightGm, biasGm, keypointsGm, scoresGm;
    weightGm.SetGlobalBuffer((__gm__ float *)weight);
    biasGm.SetGlobalBuffer((__gm__ float *)bias);
    keypointsGm.SetGlobalBuffer((__gm__ float *)keypoints);
    scoresGm.SetGlobalBuffer((__gm__ float *)scores);
    DataCopy(weights, weightGm, 4 * 32 * 9);
    DataCopyPad(biases, biasGm, DataCopyExtParams{1, 16, 0, 0, 0},
                DataCopyPadExtParams<float>{false, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
    Abs(weights, weights, 4 * 32 * 9);
    PipeBarrier<PIPE_V>();
    ReduceMax(reduced, weights, scratch, 4 * 32 * 9, false);
    PipeBarrier<PIPE_ALL>();
    if (reduced.GetValue(0) != 0.0f) return false;

    const float x = static_cast<float>(static_cast<int32_t>(t.width * t.stride)) * 0.5f;
    const float y = static_cast<float>(static_cast<int32_t>(t.height * t.stride)) * 0.5f;
    uint64_t xMask[2] = {0x55555555ULL, 0};
    uint64_t yMask[2] = {0xAAAAAAAAULL, 0};
    Duplicate(points, x, xMask, 1, 1, 8);
    Duplicate(points, y, yMask, 1, 1, 8);
    for (uint32_t k = 0; k < 4; ++k) {
        uint64_t mask[2] = {0x1111ULL << k, 0};
        Duplicate(values, biases.GetValue(k), mask, 1, 1, 8);
    }
    PipeBarrier<PIPE_ALL>();
    DataCopy(keypointsGm, points, 32);
    DataCopy(scoresGm, values, 16);
    return true;
}

#endif
