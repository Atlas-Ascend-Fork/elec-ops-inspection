/*
 * Copyright (c) 2026 FalseShare Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Apache License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "kernel_operator.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "tiling_key_fused_disconnector_keypoint_soft_argmax.h"
#include "fdk_zero_single_core.h"
#include "fdk_vector_stats.h"

using namespace AscendC;

template <typename DT_FEAT>
class FdkFullPlaneKernel {
public:
    __aicore__ inline void Init(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias,
                               GM_ADDR keypoints, GM_ADDR scores,
                               const FusedDisconnectorKeypointSoftArgmaxTilingData &t,
                               uint32_t planeOverride = 0xFFFFFFFFU) {
        h = t.height; w = t.width; n = h * w;
        tau = t.tau; stride = t.stride;
        plane = planeOverride == 0xFFFFFFFFU ? GetBlockIdx() : planeOverride;
        featGm.SetGlobalBuffer((__gm__ float *)feat);
        weightGm.SetGlobalBuffer((__gm__ float *)weight);
        biasGm.SetGlobalBuffer((__gm__ float *)bias);
        keypointsGm.SetGlobalBuffer((__gm__ float *)keypoints);
        scoresGm.SetGlobalBuffer((__gm__ float *)scores);
        pipe.InitBuffer(inputBuf, 3 * (n + 2 * w) * sizeof(float));
        pipe.InitBuffer(accBuf, n * sizeof(float));
        pipe.InitBuffer(indexBuf, 2 * n * sizeof(float));
        pipe.InitBuffer(reduceBuf, 32);
        pipe.InitBuffer(weightBuf, 4 * 32 * 9 * sizeof(float));
        pipe.InitBuffer(biasBuf, 32);
    }

    __aicore__ inline void Process() {
        auto weights = weightBuf.Get<float>();
        auto biases = biasBuf.Get<float>();
        DataCopy(weights, weightGm, 4 * 32 * 9);
        fdk::LoadBiases(biases, biasGm);
        if (StoreZeroWeightResult()) return;
        Convolve();
        fdk::VectorStatistics statistics(accBuf.Get<float>(), inputBuf.Get<float>(), indexBuf.Get<float>(),
                                         inputBuf.Get<float>()[n + 2 * w], reduceBuf.Get<float>(), w, h, 0);
        statistics.PrepareSoftmax<true>(tau);
        const float x = fdk::RestoreCoordinate(statistics.SumX<false>(), statistics.GetDenominator(), stride);
        const float y = fdk::RestoreCoordinate(statistics.SumY(), statistics.GetDenominator(), stride);
        fdk::StoreVectorResult(reduceBuf.Get<float>(), keypointsGm, scoresGm, plane, x, y, statistics.GetScore());
    }

private:
    __aicore__ inline bool StoreZeroWeightResult() {
        auto tmp = inputBuf.Get<float>();
        auto scratch = inputBuf.Get<float>()[n + 2 * w];
        auto reduced = reduceBuf.Get<float>();
        // A zero convolution kernel yields a constant heatmap and an exact centre coordinate.
        Abs(tmp, weightBuf.Get<float>()[(plane % 4) * 32 * 9], 32 * 9);
        PipeBarrier<PIPE_V>();
        ReduceMax(reduced, tmp, scratch, 32 * 9, false);
        PipeBarrier<PIPE_ALL>();
        if (reduced.GetValue(0) != 0.0f) return false;
        const float x = static_cast<float>(static_cast<int32_t>(w * stride)) * 0.5f;
        const float y = static_cast<float>(static_cast<int32_t>(h * stride)) * 0.5f;
        fdk::StoreVectorResult(reduced, keypointsGm, scoresGm, plane, x, y, biasBuf.Get<float>().GetValue(plane % 4));
        return true;
    }

    __aicore__ inline void InitializeShiftIndices() {
        auto leftIndex = indexBuf.Get<int32_t>();
        auto rightIndex = leftIndex[n];
        CreateVecIndex(leftIndex, static_cast<int32_t>(w - 1), n);
        CreateVecIndex(rightIndex, static_cast<int32_t>(w + 1), n);
        PipeBarrier<PIPE_V>();
        Muls(leftIndex, leftIndex, static_cast<int32_t>(4), n);
        Muls(rightIndex, rightIndex, static_cast<int32_t>(4), n);
        Duplicate(inputBuf.Get<float>(), 0.0f, 3 * (n + 2 * w));
        PipeBarrier<PIPE_ALL>();
        for (uint32_t y = 0; y < h; ++y) {
            leftIndex.SetValue(y * w, 0);
            rightIndex.SetValue(y * w + w - 1, 0);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void AccumulateChannel(uint32_t channel) {
        auto weights = weightBuf.Get<float>();
        auto acc = accBuf.Get<float>();
        auto input = inputBuf.Get<float>();
        const uint32_t padded = n + 2 * w;
        for (uint32_t dy = 0; dy < 3; ++dy) {
            for (uint32_t dx = 0; dx < 3; ++dx) {
                const float coefficient = weights.GetValue(((plane % 4) * 32 + channel) * 9 + dy * 3 + dx);
                Axpy(acc, input[dx * padded + dy * w], coefficient, n);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void Convolve() {
        auto input = inputBuf.Get<float>();
        auto acc = accBuf.Get<float>();
        Duplicate(acc, biasBuf.Get<float>().GetValue(plane % 4), n);
        PipeBarrier<PIPE_ALL>();
        // A contiguous channel DMA plus two UB gathers creates all horizontal shifts.
        const uint32_t padded = n + 2 * w;
        InitializeShiftIndices();
        auto leftIndex = indexBuf.Get<int32_t>();
        auto rightIndex = leftIndex[n];
        for (uint32_t c = 0; c < 32; ++c) {
            DataCopy(input[padded + w], featGm[((plane / 4) * 32 + c) * n], n);
            PipeBarrier<PIPE_ALL>();
            Gather(input[w], input[padded], leftIndex.ReinterpretCast<uint32_t>(), 0, n);
            Gather(input[2 * padded + w], input[padded], rightIndex.ReinterpretCast<uint32_t>(), 0, n);
            PipeBarrier<PIPE_V>();
            AccumulateChannel(c);
            PipeBarrier<PIPE_ALL>();
        }
    }

    TPipe pipe;
    TBuf<QuePosition::VECCALC> inputBuf, accBuf, indexBuf;
    TBuf<QuePosition::VECCALC> reduceBuf, weightBuf, biasBuf;
    GlobalTensor<float> featGm, weightGm, biasGm, keypointsGm, scoresGm;
    uint32_t h, w, n, plane, stride;
    float tau;
};
