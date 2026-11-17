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
#include "fdk_vector_stats.h"

using namespace AscendC;

template <typename DT_FEAT>
class FdkHiddenC10AivKernel {
public:
    __aicore__ inline void Init(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias,
                               GM_ADDR keypoints, GM_ADDR scores, GM_ADDR workspace, GM_ADDR tilingAddr,
                               const FusedDisconnectorKeypointSoftArgmaxTilingData &t) {
        // This class is dispatched only for c10's 36-core geometry.
        h = 72; w = 96; fullN = 6912; parts = 9;
        block = GetBlockIdx(); plane = block / 9; part = block % 9; blocks = 36;
        tileRows = 8; rowStart = part * 8; rows = 8; n = 768;
        maxTileRows = 8; inputN = 960;
        tau = 1.0f; stride = t.stride;
        featGm.SetGlobalBuffer((__gm__ float *)feat);
        weightGm.SetGlobalBuffer((__gm__ float *)weight);
        biasGm.SetGlobalBuffer((__gm__ float *)bias);
        keypointsGm.SetGlobalBuffer((__gm__ float *)keypoints);
        scoresGm.SetGlobalBuffer((__gm__ float *)scores);
        // The wrapper's workspace argument is offset by RESERVED_WORKSPACE.
        // No high-level system-workspace consumer is present: use the public API
        // to recover the actual allocation base and stay within blocks*32 bytes.
        GM_ADDR scratchBase = GetSysWorkSpacePtr();
        statsGm.SetGlobalBuffer((__gm__ float *)scratchBase);
        pipe.InitBuffer(inputBuf, 32 * inputN * sizeof(float));
        pipe.InitBuffer(packedBuf, 3 * (maxTileRows * w + 8) * sizeof(float));
        pipe.InitBuffer(statsBuf, 16 * 8 * sizeof(float));
        pipe.InitBuffer(expBuf, 64);
        pipe.InitBuffer(accBuf, n * sizeof(float));
        pipe.InitBuffer(coordBuf, n * sizeof(float));
        pipe.InitBuffer(reduceBuf, 32);
        pipe.InitBuffer(weightBuf, 32 * 9 * sizeof(float));
        pipe.InitBuffer(biasBuf, 32);
    }

    __aicore__ inline void Process() {
        auto weights = weightBuf.Get<float>();
        auto biases = biasBuf.Get<float>();
        DataCopy(weights, weightGm[(plane % 4) * 32 * 9], 288);
        fdk::LoadBiases(biases, biasGm);
        ConvolveTiles(weights);
        fdk::VectorStatistics statistics(accBuf.Get<float>(), packedBuf.Get<float>(), coordBuf.Get<float>(),
                                         inputBuf.Get<float>(), reduceBuf.Get<float>(), w, rows, rowStart);
        statistics.ProcessPartial<false, true>(biases.GetValue(plane % 4), tau, statsGm, block);
        if (part != 0) return;
        statistics.MergeAndStore(statsGm, statsBuf.Get<float>(), expBuf.Get<float>(), parts, plane,
                                  stride, keypointsGm, scoresGm);
    }

private:
    __aicore__ inline void ClearHaloRow(const LocalTensor<float>& input, uint32_t offset) {
        Duplicate(input[offset], 0.0f, uint64_t(64), uint8_t(32), 1, static_cast<uint16_t>(inputN / 8));
        Duplicate(input[offset + 64], 0.0f, uint64_t(32), uint8_t(32), 1, static_cast<uint16_t>(inputN / 8));
    }

    __aicore__ inline void InitializeTile(const LocalTensor<float>& input, const LocalTensor<float>& packed,
                                         uint32_t first, uint32_t activeRows, uint32_t bankStride) {
        Duplicate(packed, 0.0f, 3 * bankStride);
        if (first == 0) ClearHaloRow(input, 0);
        if (first + activeRows == h) ClearHaloRow(input, (activeRows + 1) * w);
        PipeBarrier<PIPE_ALL>();
        const fdk::RowCopy copy = fdk::GetRowCopy(first, activeRows, h, w);
        DataCopy(input[copy.destination], featGm[(plane / 4) * 32 * fullN + copy.source],
                 DataCopyParams{32, static_cast<uint16_t>(copy.elements / 8),
                                static_cast<uint16_t>((fullN - copy.elements) / 8),
                                static_cast<uint16_t>((inputN - copy.elements) / 8)});
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProjectChannels(const LocalTensor<float>& input, const LocalTensor<float>& packed,
                                          const LocalTensor<float>& weights, uint32_t tileN, uint32_t bankStride) {
        SetMaskNorm();
        SetVectorMask<float>(64);
        for (uint32_t c = 0; c < 32; ++c) {
            for (uint32_t dx = 0; dx < 3; ++dx) {
                for (uint32_t dy = 0; dy < 3; ++dy) {
                    const float coefficient = weights.GetValue(c * 9 + dy * 3 + dx);
                    Axpy<float, float, false>(packed[dx * bankStride + 8], input[c * inputN + dy * w],
                              coefficient, uint64_t(64), static_cast<uint8_t>(tileN / 64),
                              UnaryRepeatParams{1, 1, 8, 8});
                    PipeBarrier<PIPE_V>();
                }
            }
        }
    }

    __aicore__ inline void SetShiftIndices(const LocalTensor<int32_t>& indices, uint32_t tileN,
                                          uint32_t activeRows, bool right) {
        CreateVecIndex(indices, static_cast<int32_t>(right ? 9 : 7), tileN);
        PipeBarrier<PIPE_V>();
        Muls(indices, indices, static_cast<int32_t>(4), tileN);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t y = 0; y < activeRows; ++y) indices.SetValue(y * w + (right ? w - 1 : 0), 0);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void MergeColumns(const LocalTensor<float>& acc, const LocalTensor<float>& input,
                                        const LocalTensor<float>& packed, uint32_t activeRows, uint32_t tileN,
                                        uint32_t bankStride) {
        auto indices = coordBuf.Get<int32_t>();
        SetShiftIndices(indices, tileN, activeRows, false);
        Gather(acc, packed, indices.ReinterpretCast<uint32_t>(), 0, tileN);
        PipeBarrier<PIPE_V>();
        Add(acc, acc, packed[bankStride + 8], tileN);
        PipeBarrier<PIPE_V>();
        SetShiftIndices(indices, tileN, activeRows, true);
        Gather(input, packed[2 * bankStride], indices.ReinterpretCast<uint32_t>(), 0, tileN);
        PipeBarrier<PIPE_V>();
        Add(acc, acc, input, tileN);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ConvolveTiles(const LocalTensor<float>& weights) {
        auto input = inputBuf.Get<float>();
        auto packed = packedBuf.Get<float>();
        auto acc = accBuf.Get<float>();
        for (uint32_t tileOffset = 0; tileOffset < rows; tileOffset += maxTileRows) {
            const uint32_t activeRows = (maxTileRows < rows - tileOffset ? maxTileRows : rows - tileOffset);
            const uint32_t tileN = activeRows * w;
            const uint32_t bankStride = tileN + 8;
            InitializeTile(input, packed, rowStart + tileOffset, activeRows, bankStride);
            ProjectChannels(input, packed, weights, tileN, bankStride);
            MergeColumns(acc[tileOffset * w], input, packed, activeRows, tileN, bankStride);
        }
    }

    TPipe pipe;
    TBuf<QuePosition::VECCALC> inputBuf, packedBuf, accBuf, coordBuf;
    TBuf<QuePosition::VECCALC> reduceBuf, weightBuf, biasBuf, statsBuf, expBuf;
    GlobalTensor<float> featGm, weightGm, biasGm, keypointsGm, scoresGm, statsGm;
    uint32_t h, w, n, plane, stride, fullN, parts, block, blocks, part, tileRows, rowStart, rows, inputN, maxTileRows;
    float tau;
};

