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
#include "spline_support_math.h"

template <class T> __aicore__ inline float LocalPseudo(LocalTensor<T> coords, uint32_t i) {
    if constexpr (sizeof(T) == 4)
        return SplineCGet(coords, i);
    else {
        union {
            uint32_t u;
            float f;
        } v;
        v.u = uint32_t(SplineCGet(coords.template ReinterpretCast<uint16_t>(),i)) << 16;
        return v.f;
    }
}
template <class T, uint32_t CI>
__aicore__ inline void StageFeature(GlobalTensor<T> x, uint32_t source, LocalTensor<float> feature,
                                    LocalTensor<T> raw) {
    if constexpr (sizeof(T) == 4)
        SplineCCopy(feature, x[uint64_t(source) * CI], CI);
    else if constexpr (CI % 16 == 0)
        SplineCCopy(raw, x[uint64_t(source) * CI], CI);
    else
        SplineCCopyPad(raw, x[uint64_t(source) * CI], {1, uint32_t(CI * 2), 0, 0, 0}, {false, 0, 0, 0});
}
template <class T>
__aicore__ inline void CopyControls(uint32_t e, LocalTensor<int64_t> indices, LocalTensor<T> coords,
                                    GlobalTensor<int64_t> edge, GlobalTensor<T> attr) {
    if (e % 2 == 0)
        SplineCCopy(indices, edge, 2 * e);
    else
        SplineCCopyPad(indices, edge, {1, uint32_t(2 * e * 8), 0, 0, 0}, {false, 0, 0, 0});
    if ((2 * e * sizeof(T)) % 32 == 0)
        SplineCCopy(coords, attr, 2 * e);
    else
        SplineCCopyPad(coords, attr, {1, uint32_t(2 * e * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
}
template <class T>
__aicore__ inline void FastStore(GlobalTensor<T> output, uint64_t start, LocalTensor<float> values,
                                 LocalTensor<T> raw, uint32_t count) {
    if constexpr (sizeof(T) == 4) {
        Fence<HardEvent::V_MTE3>();
        SplineCCopy(output[start], values, count);
    } else {
        SplineCCast(raw, values, RoundMode::CAST_RINT, count);
        Fence<HardEvent::V_MTE3>();
        if (count % 16 == 0)
            SplineCCopy(output[start], raw, count);
        else
            SplineCCopyPad(output[start], raw, {1, uint32_t(count * 2), 0, 0, 0});
    }
    Fence<HardEvent::MTE3_V>();
}
// Aligned message and zero-complement owners store disjoint row intervals.
// Small runtime single-edge kernels stage real controls; larger existing
// single-edge families retain their original scalar-control path.
template <class T, uint32_t CO>
__aicore__ inline void StoreSingleEdgeZeroRows(GlobalTensor<T> outputGm, LocalTensor<T> zeroTile,
                                               uint32_t target, uint32_t rows) {
    Fence<HardEvent::V_MTE3>();
    if (target != 0)
        SplineCCopyPad(outputGm, zeroTile, {1, uint32_t(target * CO * sizeof(T)), 0, 0, 0});
    if (target + 1 < rows)
        SplineCCopyPad(outputGm[uint64_t(target + 1) * CO], zeroTile,
            {1, uint32_t((rows - target - 1) * CO * sizeof(T)), 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
}

template <class T, uint32_t CO, uint32_t FixedNodeCount = 0, bool StagedControls = false>
__aicore__ inline void ZeroSingleEdgeComplement(GM_ADDR edgeAddress, GM_ADDR outputAddress, GM_ADDR tilingAddress) {
    if constexpr (FixedNodeCount == 1)
        return;
    GlobalTensor<uint64_t> metadataGm;
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<T> outputGm;
    metadataGm.SetGlobalBuffer((__gm__ uint64_t *)tilingAddress);
    edgesGm.SetGlobalBuffer((__gm__ int64_t *)edgeAddress);
    outputGm.SetGlobalBuffer((__gm__ T *)outputAddress);
    if constexpr (StagedControls) {
        // Host selects this owner only for the complete small-output family:
        // N*CO<=1024. Real controls occupy disjoint 32-byte UB blocks.
        LocalTensor<int64_t> endpointControl(TPosition::VECCALC, 0, 2);
        LocalTensor<uint64_t> nodeControl(TPosition::VECCALC, 32, 1);
        LocalTensor<T> zeroTile(TPosition::VECCALC, 64, 1024);
        SplineCCopyPad(endpointControl, edgesGm, {1, 16, 0, 0, 0}, {false, 0, 0, 0});
        if constexpr (FixedNodeCount == 0)
            SplineCCopyPad(nodeControl, metadataGm, {1, 8, 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        constexpr uint32_t zeroElements = FixedNodeCount ? FixedNodeCount * CO : 1024;
        if constexpr (sizeof(T) == 4)
            SplineCDuplicate(zeroTile, 0.0f, zeroElements);
        else
            SplineCDuplicate(zeroTile.template ReinterpretCast<uint16_t>(), uint16_t(0), zeroElements);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        const uint32_t rows = FixedNodeCount ? FixedNodeCount : SplineCGet(nodeControl, 0) & kNodeCountMask;
        const uint32_t target = uint32_t(SplineCGet(endpointControl, 1));
        StoreSingleEdgeZeroRows<T, CO>(outputGm, zeroTile, target, rows);
        return;
    }
    const uint32_t nodeCount = FixedNodeCount ? FixedNodeCount : metadataGm.GetValue(0) & kNodeCountMask;
    if (nodeCount <= 1)
        return;
    LocalTensor<T> zeroValues(TPosition::VECCALC, 0, kWholeOutputCapacityElements);
    if constexpr (sizeof(T) == 4)
        SplineCDuplicate(zeroValues, 0.0f, nodeCount * CO);
    else
        SplineCDuplicate(zeroValues.template ReinterpretCast<uint16_t>(), uint16_t(0), nodeCount * CO);
    // Endpoints are immutable scalar controls. A direct GM read avoids a
    // separate control DMA; this owner never consumes another core's data.
    const uint32_t targetRowIndex = uint32_t(edgesGm.GetValue(1));
    StoreSingleEdgeZeroRows<T, CO>(outputGm, zeroValues, targetRowIndex, nodeCount);
}
