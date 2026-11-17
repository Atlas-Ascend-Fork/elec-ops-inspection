/*
 * Copyright (c) 2026 FalseShare Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Apache License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
namespace matmul {
__aicore__ inline void clearWorkspace(GM_ADDR) {
 AscendC::SetAtomicNone();
 if ASCEND_IS_AIC {
  AscendC::SetMaskNorm();AscendC::SetLoadDataBoundary(uint64_t(0));AscendC::SetLoadDataPaddingValue(uint64_t(0));
 }
}
}
#include "fdk_fullplane.h"
#include "fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "tiling_key_fused_disconnector_keypoint_soft_argmax.h"

#include "fdk_onchip_join.h"
#include "fdk_vertical_projection.h"
#include "fdk_hidden_c10_aiv.h"
#include "fdk_vector_stats.h"

using namespace AscendC;

template <typename DT_FEAT>
class FdkSpatialKernel {
public:
    __aicore__ inline void Init(GM_ADDR feat, GM_ADDR weight, GM_ADDR bias,
                               GM_ADDR keypoints, GM_ADDR scores, GM_ADDR workspace, GM_ADDR tilingAddr,
                               const FusedDisconnectorKeypointSoftArgmaxTilingData &t) {
        h = t.height; w = t.width; fullN = h * w; parts = t.parts;
        block = GetBlockIdx(); plane = block / parts; part = block % parts; blocks = t.batch * 4 * parts;
        tileRows = (h + parts - 1) / parts; rowStart = part * tileRows;
        rows = (tileRows < h - rowStart ? tileRows : h - rowStart); n = rows * w;
        maxTileRows = w <= 64 ? 24 : 12;
        if (maxTileRows > tileRows) maxTileRows = tileRows;
        inputN = ((maxTileRows + 2) * w + 16 + 15) / 16 * 16;
        tau = t.tau; stride = t.stride;
        featGm.SetGlobalBuffer((__gm__ float *)feat);
        weightGm.SetGlobalBuffer((__gm__ float *)weight);
        biasGm.SetGlobalBuffer((__gm__ float *)bias);
        weightMapGm.SetGlobalBuffer((__gm__ uint32_t *)tilingAddr + 8);
        keypointsGm.SetGlobalBuffer((__gm__ float *)keypoints);
        scoresGm.SetGlobalBuffer((__gm__ float *)scores);
        // The wrapper's workspace argument is offset by RESERVED_WORKSPACE.
        // No high-level system-workspace consumer is present: use the public API
        // to recover the actual allocation base and stay within blocks*32 bytes.
        GM_ADDR scratchBase = GetSysWorkSpacePtr();
        statsGm.SetGlobalBuffer((__gm__ float *)scratchBase);
        pipe.InitBuffer(inputBuf, 8 * inputN * sizeof(float));
        pipe.InitBuffer(packedBuf, 8 * inputN * sizeof(float));
        pipe.InitBuffer(laneBuf, 8 * maxTileRows * w * sizeof(float));
        pipe.InitBuffer(edgeBuf, (maxTileRows + 2) * 8 * sizeof(float));
        pipe.InitBuffer(packedWeightBuf, 4 * 16 * 8 * sizeof(float));
        pipe.InitBuffer(statsBuf, 16 * 8 * sizeof(float));
        pipe.InitBuffer(expBuf, 64);
        pipe.InitBuffer(accBuf, n * sizeof(float));
        pipe.InitBuffer(coordBuf, n * sizeof(float));
        pipe.InitBuffer(reduceBuf, 32);
        pipe.InitBuffer(weightBuf, 32 * 9 * sizeof(float));
        pipe.InitBuffer(weightMapBuf, 512 * sizeof(uint32_t));
        pipe.InitBuffer(biasBuf, 32);
    }

    __aicore__ inline void Process() {
        auto weights = weightBuf.Get<float>();
        auto biases = biasBuf.Get<float>();
        auto weightMap = weightMapBuf.Get<uint32_t>();
        // Keep the map DMA between the weight and bias transfers.
        DataCopy(weights, weightGm[(plane % 4) * 32 * 9], 288);
        DataCopy(weightMap, weightMapGm, 512);
        fdk::LoadBiases(biases, biasGm);
        Gather(packedWeightBuf.Get<float>(), weights, weightMap, 0, 512);
        ConvolveTiles();
        fdk::VectorStatistics statistics(accBuf.Get<float>(), packedBuf.Get<float>(), coordBuf.Get<float>(),
                                         inputBuf.Get<float>(), reduceBuf.Get<float>(), w, rows, rowStart);
        statistics.ProcessPartial<true, false>(biases.GetValue(plane % 4), tau, statsGm, block);
        if (part != 0) return;
        statistics.MergeAndStore(statsGm, statsBuf.Get<float>(), expBuf.Get<float>(), parts, plane,
                                  stride, keypointsGm, scoresGm);
    }

private:
    __aicore__ inline void LoadGroup(const LocalTensor<float>& input, const fdk::RowCopy& copy, uint32_t group) {
        DataCopy(input[8 + copy.destination], featGm[((plane / 4) * 32 + group * 8) * fullN + copy.source],
                 DataCopyParams{8, static_cast<uint16_t>(copy.elements / 8),
                                static_cast<uint16_t>((fullN - copy.elements) / 8),
                                static_cast<uint16_t>((inputN - copy.elements) / 8)});
    }

    __aicore__ inline void AccumulateColumn(const LocalTensor<float>& lanes, const LocalTensor<float>& packed,
                                            uint32_t group, uint32_t dx, uint32_t tileN) {
        auto packedWeights = packedWeightBuf.Get<float>();
        for (uint32_t dy = 0; dy < 3; ++dy) {
            MulAddDst(lanes, packed[(7 + dx + dy * w) * 8], packedWeights[group * 128 + (dy * 3 + dx) * 8],
                      uint64_t(64), static_cast<uint8_t>(tileN / 8), BinaryRepeatParams{1, 1, 0, 8, 8, 0});
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareLeftEdge(const LocalTensor<float>& packed, uint32_t activeRows) {
        // Save the right edge, zero it for left-neighbour reads, then restore it in PrepareRightEdge.
        auto edge = edgeBuf.Get<float>();
        PipeBarrier<PIPE_ALL>();
        DataCopy(edge, packed[(8 + w - 1) * 8],
                 DataCopyParams{static_cast<uint16_t>(activeRows + 2), 1, static_cast<uint16_t>(w - 1), 0});
        PipeBarrier<PIPE_ALL>();
        Duplicate(packed[(8 + w - 1) * 8], 0.0f, uint64_t(8), static_cast<uint8_t>(activeRows + 2),
                  1, static_cast<uint16_t>(w));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrepareRightEdge(const LocalTensor<float>& packed, uint32_t activeRows) {
        PipeBarrier<PIPE_ALL>();
        DataCopy(packed[(8 + w - 1) * 8], edgeBuf.Get<float>(),
                 DataCopyParams{static_cast<uint16_t>(activeRows + 2), 1, 0, static_cast<uint16_t>(w - 1)});
        PipeBarrier<PIPE_ALL>();
        Duplicate(packed[8 * 8], 0.0f, uint64_t(8), static_cast<uint8_t>(activeRows + 2), 1, static_cast<uint16_t>(w));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ConvolveTiles() {
        auto input = inputBuf.Get<float>();
        auto packed = packedBuf.Get<float>();
        auto lanes = laneBuf.Get<float>();
        auto acc = accBuf.Get<float>();
        for (uint32_t tileOffset = 0; tileOffset < rows; tileOffset += maxTileRows) {
            const uint32_t activeRows = (maxTileRows < rows - tileOffset ? maxTileRows : rows - tileOffset);
            const uint32_t tileN = activeRows * w;
            const fdk::RowCopy copy = fdk::GetRowCopy(rowStart + tileOffset, activeRows, h, w);
            Duplicate(lanes, 0.0f, tileN * 8);
            // Pack8 leaves this raw buffer's halo zeros intact for every channel group.
            Duplicate(input, 0.0f, 8 * inputN);
            PipeBarrier<PIPE_ALL>();
            LoadGroup(input, copy, 0);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t group = 0; group < 4; ++group) {
                Pack8(packed, input, inputN);
                PipeBarrier<PIPE_V>();
                AccumulateColumn(lanes, packed, group, 1, tileN);
                PrepareLeftEdge(packed, activeRows);
                AccumulateColumn(lanes, packed, group, 0, tileN);
                PrepareRightEdge(packed, activeRows);
                if (group < 3) {
                    // Prefetch after Pack8 releases input; the final PIPE_ALL orders the next Pack8.
                    LoadGroup(input, copy, group + 1);
                }
                AccumulateColumn(lanes, packed, group, 2, tileN);
                PipeBarrier<PIPE_ALL>();
            }
            BlockReduceSum(acc[tileOffset * w], lanes, tileN / 8, 64, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }
    }

    // Float32 8-channel by L-spatial transpose, derived from the SDK's
    // TransposeBase8M16 recipe. Every output pixel occupies one 32-byte block.
    __aicore__ inline void Pack8(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t len) {
        uint64_t srcList[16], dstList[16];
        for (uint32_t i = 0; i < 16; ++i) {
            srcList[i] = (uint64_t)src[(i % 8) * len + (i / 8) * 8].GetPhyAddr();
            dstList[i] = (uint64_t)dst[(i % 2 * 8 + i / 2) * 8].GetPhyAddr();
        }
        TransDataTo5HD<float>(dstList, srcList,
            TransDataTo5HDParams{false, false, static_cast<uint8_t>(len / 16), static_cast<uint16_t>(len == 16 ? 0 : 16), static_cast<uint16_t>(len == 16 ? 0 : 2)});
    }
    TPipe pipe;
    TBuf<QuePosition::VECCALC> inputBuf, packedBuf, laneBuf, edgeBuf, packedWeightBuf, accBuf, coordBuf;
    TBuf<QuePosition::VECCALC> reduceBuf, weightBuf, biasBuf, statsBuf, expBuf, weightMapBuf;
    GlobalTensor<uint32_t> weightMapGm;
    GlobalTensor<float> featGm, weightGm, biasGm, keypointsGm, scoresGm, statsGm;
    uint32_t h, w, n, plane, stride, fullN, parts, block, blocks, part, tileRows, rowStart, rows, inputN, maxTileRows;
    float tau;
};

template <typename DT_FEAT, uint32_t DISPATCH>
__global__ __aicore__ void fused_disconnector_keypoint_soft_argmax(
    GM_ADDR feat, GM_ADDR weight, GM_ADDR bias, GM_ADDR keypoints, GM_ADDR scores,
    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FdkWeightMapTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedDisconnectorKeypointSoftArgmaxTilingData, tilingData, tiling);
    if constexpr (DISPATCH == 0) {
        if (FdkTrySingleCoreZeroWeights(weight,bias,keypoints,scores,tilingData)) return;
        for (uint32_t plane=0; plane<tilingData.batch*4; ++plane) {
            FdkFullPlaneKernel<DT_FEAT> op;
            op.Init(feat,weight,bias,keypoints,scores,tilingData,plane);
            op.Process();
        }
    } else if constexpr(DISPATCH == 3) {
        FdkVerticalProjection op;
        op.Init(feat,weight,bias,keypoints,scores,tilingData,tiling);
        op.Process();
    } else if constexpr(DISPATCH == 2) {
        uint32_t invalid;
        {FdkOnchipJoin op;op.Init(feat,weight,bias,keypoints,scores,tilingData,tiling);invalid=op.Process();}
        if ASCEND_IS_AIV {
            for(uint32_t k=0;k<2;++k) if(invalid&(1U<<k)) {
                uint32_t plane=(GetBlockIdx()/GetTaskRation()/tilingData.parts)*4+GetSubBlockIdx()*2+k;
                FdkFullPlaneKernel<DT_FEAT> op;op.Init(feat,weight,bias,keypoints,scores,tilingData,plane);op.Process();
            }
        }
    } else {
        if (tilingData.batch == 1 && tilingData.height == 72 && tilingData.width == 96
            && tilingData.tau == 1.0f && tilingData.stride == 4) {
            FdkHiddenC10AivKernel<DT_FEAT> op;
            op.Init(feat,weight,bias,keypoints,scores,workspace,tiling,tilingData);
            op.Process();
        } else {
            FdkSpatialKernel<DT_FEAT> op;
            op.Init(feat,weight,bias,keypoints,scores,workspace,tiling,tilingData);
            op.Process();
        }
    }
}
