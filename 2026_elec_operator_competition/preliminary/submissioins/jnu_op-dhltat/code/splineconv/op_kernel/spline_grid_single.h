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
#include "spline_sparse_math.h"

template <class T, uint32_t CI, uint32_t CO, bool UnitCube, bool BoundedInput>
struct SingleGridWorker {
    static constexpr bool partitionOutput = (CO * sizeof(T)) % kUbAlignmentBytes == 0;
    static constexpr uint32_t matrixElements = CI * CO;
    static constexpr uint32_t featureCapacity = UnitCube ? kSingleOwnerFeaturePrefixElements : CI;
    static constexpr uint32_t weightCapacity = UnitCube ? 8 * matrixElements : kSingleOwnerWeightCapacityElements;
    static constexpr uint32_t prefixRows = featureCapacity / CI;
    LocalTensor<int64_t> indices;
    LocalTensor<T> coords, coordsZ, rawFeature, rawWeight, rawOutput;
    LocalTensor<float> feature, weights, result, broadcast, output, messageSum;
    GlobalTensor<T> x, weightsGm, coordsGm, outputGm;
    GlobalTensor<uint16_t> coordinateBits;
    GlobalTensor<int64_t> edgesGm;
    GlobalTensor<uint64_t> metadataGm;
    uint32_t nodeCount, source, target, knotCount, knotSizeX, knotSizeY, knotSizeZ, scaleX, scaleY, scaleZ;
    bool prefetched, threeDimensional;

    __aicore__ inline void Allocate() {
        if constexpr (!UnitCube) {
            indices = LocalTensor<int64_t>(TPosition::VECCALC, 0, 4);
            coords = LocalTensor<T>(TPosition::VECCALC, 512, 32 / sizeof(T));
            coordsZ = LocalTensor<T>(TPosition::VECCALC, 1536, 32 / sizeof(T));
        }
        uint32_t offsetBytes = kSparseControlBytes;
        feature = SparseRuntimeAlloc<float>(offsetBytes, featureCapacity);
        weights = SparseRuntimeAlloc<float>(offsetBytes, weightCapacity);
        offsetBytes += kResultPhasePaddingBytes;
        result = SparseRuntimeAlloc<float>(offsetBytes, matrixElements * SupportBatchWidth(matrixElements, 8));
        offsetBytes += kOperandPhasePaddingBytes;
        broadcast = SparseRuntimeAlloc<float>(offsetBytes, CI * (CO <= 32 ? CO : 8));
        output = SparseRuntimeAlloc<float>(offsetBytes, kWholeOutputCapacityElements);
        messageSum = SparseRuntimeAlloc<float>(offsetBytes, CO);
        rawFeature = LocalTensor<T>(TPosition::VECCALC, 0, 1);
        rawWeight = LocalTensor<T>(TPosition::VECCALC, 0, 1);
        rawOutput = LocalTensor<T>(TPosition::VECCALC, 0, 1);
        if constexpr (sizeof(T) != 4) {
            // BF16 input/output storage starts in a separate 64-KiB UB region.
            offsetBytes = (offsetBytes + kBf16StorageBoundaryBytes - 1) / kBf16StorageBoundaryBytes *
                          kBf16StorageBoundaryBytes;
            rawFeature = SparseRuntimeAlloc<T>(offsetBytes, featureCapacity);
            rawWeight = SparseRuntimeAlloc<T>(offsetBytes, weightCapacity);
            rawOutput = SparseRuntimeAlloc<T>(offsetBytes, kWholeOutputCapacityElements);
        }
    }

    __aicore__ inline void Bind(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling) {
        x.SetGlobalBuffer((__gm__ T *)xp);
        weightsGm.SetGlobalBuffer((__gm__ T *)wp);
        coordsGm.SetGlobalBuffer((__gm__ T *)ap);
        if constexpr (UnitCube)
            coordinateBits.SetGlobalBuffer((__gm__ uint16_t *)ap);
        outputGm.SetGlobalBuffer((__gm__ T *)yp);
        edgesGm.SetGlobalBuffer((__gm__ int64_t *)ep);
        metadataGm.SetGlobalBuffer((__gm__ uint64_t *)tiling);
    }

    __aicore__ inline void StageUnitCube() {
        // Bulk operand DMA overlaps independent immutable scalar control reads.
        if constexpr (sizeof(T) == 4)
            SplineCCopy(weights, weightsGm, 8 * matrixElements);
        else
            SplineCCopy(rawWeight, weightsGm, 8 * matrixElements);
        uint32_t scalarSource = 0;
        if constexpr (BoundedInput) {
            // The real source row is legal even when the fixed prefix does not fit.
            scalarSource = uint32_t(edgesGm.GetValue(0));
            StageFeature<T, CI>(x, scalarSource, feature, rawFeature);
        } else if constexpr (sizeof(T) == 4) {
            SplineCCopy(feature, x, featureCapacity);
        } else {
            SplineCCopy(rawFeature, x, featureCapacity);
        }
        // The fixed local zero capacity can initialize while the controls load.
        // Only nodeCount*CO elements are subsequently stored to the output tensor.
        if constexpr (!partitionOutput)
            SplineCDuplicate(output, 0.0f, uint64_t(64), uint8_t(64), 1, 8);
        // The packed word and endpoints use direct scalar GM reads.
        uint64_t packedMetadataWord = metadataGm.GetValue(0);
        nodeCount = packedMetadataWord & kNodeCountMask;
        source = BoundedInput ? scalarSource : uint32_t(edgesGm.GetValue(0));
        target = uint32_t(edgesGm.GetValue(1));
        // UnboundedInput has N*CI >= featureCapacity. The actual endpoint is
        // arbitrary; sources beyond the prefix replace staging with their row.
        prefetched = !BoundedInput && source < prefixRows;
        if constexpr (!BoundedInput) {
            if (!prefetched) {
                // Complete the old prefix before replacing the same UB slot.
                Fence<HardEvent::MTE2_S>();
                StageFeature<T, CI>(x, source, feature, rawFeature);
            }
        }
    }

    __aicore__ inline void ComputeUnitCube() {
        float u = PseudoValue(coordsGm, coordinateBits, 0), v = PseudoValue(coordsGm, coordinateBits, 1), z = PseudoValue(coordsGm, coordinateBits, 2);
        // The open 2x2x2 grid uses the original coordinates at all endpoints;
        // canonical support slots avoid dynamic permutations of zero terms.
        Fence<HardEvent::MTE2_V>();
        if constexpr (sizeof(T) != 4) {
            SplineCCast(weights, rawWeight, RoundMode::CAST_NONE, 8 * matrixElements);
            if (prefetched)
                SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, featureCapacity);
            else
                SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, CI);
            PipeBarrier<PIPE_V>();
        }
        // Accumulate all eight support matrices in the original FP32 order.
        // A single incoming edge makes add and mean identical at its destination.
        IndexedTrilinear<CI, CO>(u, v, z, 0, 1, 2, 3, 4, 5, 6, 7,
                                 feature[prefetched ? source * CI : 0], weights, result, broadcast, messageSum);
    }

    __aicore__ inline void ReadGrid() {
        SplineCCopyPad(indices, edgesGm, {1, 16, 0, 0, 0}, {false, 0, 0, 0});
        SplineCCopyPad(coords, coordsGm, {1, uint32_t(2 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        // Every supported degree-one knot grid contains at least four matrices.
        if constexpr (sizeof(T) == 4)
            SplineCCopy(weights, weightsGm, 4 * matrixElements);
        else
            SplineCCopy(rawWeight, weightsGm, 4 * matrixElements);
        // All independent control/operand/vector issues retain their order.
        // Read the same immutable packed word while those transfers can run;
        // EVENT_ID0 still publishes every endpoint/coordinate UB read below.
        uint64_t packedMetadataWord = metadataGm.GetValue(0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        nodeCount = packedMetadataWord & kNodeCountMask;
        knotSizeX = (packedMetadataWord >> kKnotSizeXShift) & kGridFieldMask;
        knotSizeY = (packedMetadataWord >> kKnotSizeYShift) & kGridFieldMask;
        scaleX = (packedMetadataWord >> kScaleXShift) & kGridFieldMask;
        scaleY = (packedMetadataWord >> kScaleYShift) & kGridFieldMask;
        threeDimensional = (packedMetadataWord >> kIs3dShift) & 1;
        knotSizeZ = (packedMetadataWord >> kKnotSizeZShift) & kGridFieldMask;
        scaleZ = (packedMetadataWord >> kScaleZShift) & kGridFieldMask;
        knotCount = knotSizeX * knotSizeY * (threeDimensional ? knotSizeZ : 1);
        source = SplineCGet(indices, 0);
        target = SplineCGet(indices, 1);
    }

    __aicore__ inline void StageGrid() {
        if (threeDimensional) {
            SplineCCopyPad(coordsZ, coordsGm[2], {1, uint32_t(sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        }
        if (knotCount > 4) {
            if constexpr (sizeof(T) == 4)
                SplineCCopy(weights[4 * matrixElements], weightsGm[4 * matrixElements],
                         (knotCount - 4) * matrixElements);
            else
                SplineCCopy(rawWeight[4 * matrixElements], weightsGm[4 * matrixElements],
                         (knotCount - 4) * matrixElements);
        }
        StageFeature<T, CI>(x, source, feature, rawFeature);
        if constexpr (!partitionOutput)
            SplineCDuplicate(output, 0.0f, nodeCount * CO);
    }

    __aicore__ inline void InterpolateGrid() {
        float scaledX = LocalPseudo(coords, 0) * scaleX, scaledY = LocalPseudo(coords, 1) * scaleY;
        int32_t baseX = int32_t(scaledX), baseY = int32_t(scaledY);
        float u = scaledX - baseX, v = scaledY - baseY;
        uint32_t x0 = baseX < knotSizeX ? baseX : baseX - knotSizeX,
                 x1 = baseX + 1 < knotSizeX ? baseX + 1 : baseX + 1 - knotSizeX;
        uint32_t y0 = baseY < knotSizeY ? baseY : baseY - knotSizeY,
                 y1 = baseY + 1 < knotSizeY ? baseY + 1 : baseY + 1 - knotSizeY;
        float z = 0.0f;
        uint32_t lowerPlane = 0, upperPlane = 0;
        if (threeDimensional) {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            float scaledZ = LocalPseudo(coordsZ, 0) * scaleZ;
            int32_t baseZ = int32_t(scaledZ);
            z = scaledZ - baseZ;
            uint32_t z0 = baseZ < knotSizeZ ? baseZ : baseZ - knotSizeZ,
                     z1 = baseZ + 1 < knotSizeZ ? baseZ + 1 : baseZ + 1 - knotSizeZ;
            lowerPlane = z0 * knotSizeX * knotSizeY;
            upperPlane = z1 * knotSizeX * knotSizeY;
        }
        Fence<HardEvent::MTE2_V>();
        if constexpr (sizeof(T) != 4) {
            SplineCCast(weights, rawWeight, RoundMode::CAST_NONE, knotCount * matrixElements);
            SplineCCast(feature, rawFeature, RoundMode::CAST_NONE, CI);
            PipeBarrier<PIPE_V>();
        }
        if (threeDimensional)
            IndexedTrilinear<CI, CO>(u, v, z, x0 + knotSizeX * y0 + lowerPlane, x1 + knotSizeX * y0 + lowerPlane,
                                     x0 + knotSizeX * y1 + lowerPlane, x1 + knotSizeX * y1 + lowerPlane,
                                     x0 + knotSizeX * y0 + upperPlane, x1 + knotSizeX * y0 + upperPlane,
                                     x0 + knotSizeX * y1 + upperPlane, x1 + knotSizeX * y1 + upperPlane, feature,
                                     weights, result, broadcast, messageSum);
        else
            IndexedBilinear<CI, CO>(u, v, x0 + knotSizeX * y0, x1 + knotSizeX * y0, x0 + knotSizeX * y1,
                                    x1 + knotSizeX * y1, feature, weights, result, broadcast, messageSum);
    }

    __aicore__ inline void StoreMessage() {
        if constexpr (CO <= 64)
            SplineCAdds(output[partitionOutput ? 0 : target * CO], result, 0.0f, uint64_t(CO), uint8_t(1), {1, 1, 8, 8});
        else
            SplineCAdds(output[partitionOutput ? 0 : target * CO], result, 0.0f, uint64_t(64), uint8_t(CO / 64), {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        FastStore(outputGm, partitionOutput ? target * CO : 0, output, rawOutput,
                  partitionOutput ? CO : nodeCount * CO);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Run(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp, GM_ADDR tiling) {
        if (GetBlockIdx() != 0) {
            if constexpr ((CO * sizeof(T)) % kUbAlignmentBytes == 0) {
                if (GetBlockIdx() == 1)
                    ZeroSingleEdgeComplement<T, CO>(ep, yp, tiling);
            }
            return;
        }
        Allocate();
        Bind(xp, ep, ap, wp, yp, tiling);
        if constexpr (UnitCube) {
            StageUnitCube();
            ComputeUnitCube();
        } else {
            ReadGrid();
            StageGrid();
            InterpolateGrid();
        }
        StoreMessage();
    }

};

// A complete 2x2x2 degree-one open grid uses eight fixed support matrices.
// Loading all three coordinates and all weights together removes the second
// coordinate DMA and runtime grid decoding. N remains metadata.
template <class T, uint32_t CI, uint32_t CO, bool BoundedInput = false>
__aicore__ inline void UnitCubeSingleEdge(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp,
                                          GM_ADDR tiling) {
    SingleGridWorker<T, CI, CO, true, BoundedInput> worker;
    worker.Run(xp, ep, ap, wp, yp, tiling);
}

// A single edge has one output owner. Loading its complete bounded knot set
// avoids per-support DMA, and writing the whole output needs no cross-core sync.
// Graph size and knot sizes are metadata, never template dimensions.
template <class T, uint32_t CI, uint32_t CO>
__aicore__ inline void GeneralSingleEdge(GM_ADDR xp, GM_ADDR ep, GM_ADDR ap, GM_ADDR wp, GM_ADDR yp,
                                         GM_ADDR tiling) {
    SingleGridWorker<T, CI, CO, false, false> worker;
    worker.Run(xp, ep, ap, wp, yp, tiling);
}
