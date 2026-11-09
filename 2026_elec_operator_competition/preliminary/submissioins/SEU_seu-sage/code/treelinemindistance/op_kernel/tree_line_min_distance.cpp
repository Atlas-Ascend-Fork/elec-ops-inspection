// Copyright 2026 Electrical Engineering SIG - CANN Community
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernel_operator.h"
#include "tiling_key_tree_line_min_distance.h"
#include "tree_line_min_distance_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kMaxSegments = kTreeLineMaxSegments;
// The host selects the tiny dispatch only when the total point count times the
// batch segment count stays within eight per block and at least eight blocks
// are available, so the whole workload never exceeds 64 point-segment pairs.
constexpr uint32_t kTinyPointBound = 64;
constexpr uint32_t kPointTile = 64;
constexpr uint32_t kFewPointTile = 512;
constexpr uint32_t kFewSegmentThreshold = 8;
constexpr uint32_t kSegmentTile = 64;
constexpr uint32_t kMaxSegmentChunks = kMaxSegments / kSegmentTile;
constexpr uint32_t kFloatElementsPerBlock = 8;
constexpr uint32_t kMatrixElements = kPointTile * kSegmentTile;
constexpr uint32_t kPointBroadcastElements = kPointTile * kFloatElementsPerBlock;
constexpr uint32_t kChunkResultStride = 2 * kPointTile;
constexpr uint32_t kGroupedReductionMinChunks = 8;
constexpr uint32_t kChunkRowElements = kMaxSegmentChunks;
constexpr uint32_t kMatrixChunkResultStride = kChunkResultStride + kFloatElementsPerBlock;
constexpr uint32_t kWorkspaceAlignment = 32;
constexpr uint32_t kGeometryPlanes = 7;
constexpr uint8_t kGeometryPlaneBlocks =
    static_cast<uint8_t>(kMaxSegments * sizeof(float) / kWorkspaceAlignment);
constexpr uint32_t kMaxWorkspaceBytes = 184 * 1024;
constexpr uint32_t kPointPipelineBuffers = 2;
constexpr int32_t kPointPipelineEventBase = 1;
constexpr uint32_t kOutputPipelineBuffers = 2;
constexpr int32_t kOutputPipelineEventBase = 3;
constexpr uint32_t kIndexSeedElements = 8;
constexpr uint32_t kScalarPointBroadcastPointLimit = 2;
constexpr uint32_t kStartXPlane = 0;
constexpr uint32_t kStartYPlane = kMaxSegments;
constexpr uint32_t kStartZPlane = 2 * kMaxSegments;
constexpr uint32_t kDirectionXPlane = 3 * kMaxSegments;
constexpr uint32_t kDirectionYPlane = 4 * kMaxSegments;
constexpr uint32_t kDirectionZPlane = 5 * kMaxSegments;
constexpr uint32_t kScalePlane = 6 * kMaxSegments;
// Matrix intermediates rotate between the three physical UB bank regions.
constexpr uint32_t kMatrixPlane0 = 0;
constexpr uint32_t kMatrixPlane1 = 4160;
constexpr uint32_t kMatrixPlane2 = 16384;
constexpr uint32_t kMatrixPlane3 = 20544;
constexpr uint32_t kMatrixPlane4 = 32768;
constexpr uint32_t kMatrixPlane5 = 36928;
constexpr uint32_t kDistancePlane = kMatrixPlane2;
constexpr uint32_t kFewXPlane = 0;
constexpr uint32_t kFewYPlane = kFewPointTile;
constexpr uint32_t kFewZPlane = 2 * kFewPointTile;
constexpr uint32_t kFewProjectionPlane = 3 * kFewPointTile;
constexpr uint32_t kFewTemporaryPlane = 4 * kFewPointTile;
constexpr float kFloatMax = 3.402823466e+38F;
constexpr float kFloatMin = 1.175494351e-38F;

constexpr uint32_t AlignBytes(uint32_t bytes)
{
    return (bytes + kWorkspaceAlignment - 1) / kWorkspaceAlignment * kWorkspaceAlignment;
}

template <HardEvent event>
__aicore__ inline void Synchronize()
{
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

template <typename T, int TINY_MODE>
class TreeLineKernel {
    static constexpr uint32_t kSegmentInputElements = 6 * kMaxSegments;
    static constexpr uint32_t kGatherStagingElements =
        sizeof(T) == sizeof(half) ? 6 * kMaxSegments : kPointTile;
    static constexpr uint32_t kPointInputElements = 3 * kPointTile;
    static constexpr uint32_t kGeometryElements = 7 * kMaxSegments;
    static constexpr uint32_t kWorkPlaneElements = kMatrixPlane5 + kMatrixElements;
    static constexpr uint32_t kPointCoordinateElements = 3 * kPointTile;
    static constexpr uint32_t kPointBroadcastRegionElements = 3 * kPointBroadcastElements;
    static constexpr uint32_t kChunkResultElements = kMaxSegmentChunks * kMatrixChunkResultStride;
    static constexpr uint32_t kPointResultElements = kPointTile;
    static constexpr uint32_t kDistanceOutputElements = kPointTile;
    static constexpr uint32_t kSegmentOffsetElements = kMaxSegments;
    static constexpr uint32_t kPointOffsetElements = kPointTile;
    static constexpr uint32_t kPointResultOffsetElements = kPointTile;
    static constexpr uint32_t kChunkGatherElements = kPointTile * kChunkRowElements;

    static constexpr uint32_t kSegmentInputBytes = kSegmentInputElements * sizeof(T);
    static constexpr uint32_t kGatherStagingBytes = kGatherStagingElements * sizeof(T);
    static constexpr uint32_t kPointInputBytes = kPointInputElements * sizeof(T);
    static constexpr uint32_t kGeometryBytes = kGeometryElements * sizeof(float);
    static constexpr uint32_t kWorkPlaneBytes = kWorkPlaneElements * sizeof(float);
    static constexpr uint32_t kPointCoordinateBytes = kPointCoordinateElements * sizeof(float);
    static constexpr uint32_t kPointBroadcastBytes = kPointBroadcastRegionElements * sizeof(float);
    static constexpr uint32_t kChunkResultBytes = kChunkResultElements * sizeof(float);
    static constexpr uint32_t kPointResultBytes = kPointResultElements * sizeof(float);
    static constexpr uint32_t kDistanceOutputBytes = kDistanceOutputElements * sizeof(T);
    static constexpr uint32_t kSegmentOffsetBytes = kSegmentOffsetElements * sizeof(int32_t);
    static constexpr uint32_t kPointOffsetBytes = kPointOffsetElements * sizeof(int32_t);
    static constexpr uint32_t kPointResultOffsetBytes = kPointResultOffsetElements * sizeof(int32_t);

    static constexpr uint32_t kSegmentInputOffset = 33024;
    static constexpr uint32_t kGatherStagingOffset =
        AlignBytes(kSegmentInputOffset + kSegmentInputBytes);
    static constexpr uint32_t kPointInputOffset =
        AlignBytes(kGatherStagingOffset + kGatherStagingBytes);
    static constexpr uint32_t kPointRawBytes = AlignBytes(kPointInputBytes);
    static constexpr uint32_t kPointRawCount = kPointRawBytes / sizeof(T);
    static constexpr uint32_t kPointPipelineBytes =
        kPointPipelineBuffers * kPointRawBytes;
    static constexpr uint32_t kGeometryOffset = 98560;
    static constexpr uint32_t kWorkPlaneOffset = 0;
    static constexpr uint32_t kPointCoordinateOffset =
        AlignBytes(kWorkPlaneOffset + kWorkPlaneBytes);
    static constexpr uint32_t kPointBroadcastOffset =
        AlignBytes(kPointCoordinateOffset + kPointCoordinateBytes);
    static constexpr uint32_t kChunkResultOffset =
        AlignBytes(kPointBroadcastOffset + kPointBroadcastBytes);
    static constexpr uint32_t kCandidateValuesOffset =
        AlignBytes(kChunkResultOffset + kChunkResultBytes);
    static constexpr uint32_t kCandidateIndicesOffset =
        AlignBytes(kCandidateValuesOffset + kPointResultBytes);
    static constexpr uint32_t kBestValuesOffset =
        AlignBytes(kCandidateIndicesOffset + kPointResultBytes);
    static constexpr uint32_t kBestIndicesOffset =
        AlignBytes(kBestValuesOffset + kOutputPipelineBuffers * kPointResultBytes);
    static constexpr uint32_t kDistanceOutputOffset =
        AlignBytes(kBestIndicesOffset + kOutputPipelineBuffers * kPointResultBytes);
    static constexpr uint32_t kSegmentOffsetsOffset =
        AlignBytes(
            kDistanceOutputOffset + kOutputPipelineBuffers * kDistanceOutputBytes);
    static constexpr uint32_t kPointOffsetsOffset =
        AlignBytes(kSegmentOffsetsOffset + kSegmentOffsetBytes);
    static constexpr uint32_t kPointResultOffsetsOffset =
        AlignBytes(kPointOffsetsOffset + kPointOffsetBytes);
    static constexpr uint32_t kWorkspaceBytes =
        AlignBytes(kPointResultOffsetsOffset + kPointResultOffsetBytes);
    static_assert(kWorkspaceBytes <= kMaxWorkspaceBytes);
    static_assert(kChunkGatherElements * sizeof(int32_t) <= kSegmentInputBytes);
    static_assert(kPointInputOffset + kPointPipelineBytes <= kGeometryOffset);
    static_assert(kGeometryOffset + kGeometryBytes <= 131072);
    static_assert(kMatrixPlane1 * sizeof(float) + kMatrixElements * sizeof(float) <= kSegmentInputOffset);
    static_assert(kMatrixPlane3 * sizeof(float) + kMatrixElements * sizeof(float) <= kGeometryOffset);

    static constexpr uint32_t kFewPointInputElements = 3 * kFewPointTile;
    static constexpr uint32_t kFewPointCoordinateElements = 3 * kFewPointTile;
    static constexpr uint32_t kFewWorkPlaneElements = 5 * kFewPointTile;
    static constexpr uint32_t kFewPointResultElements = kFewPointTile;
    static constexpr uint32_t kFewDistanceOutputElements = kFewPointTile;
    static constexpr uint32_t kFewPointInputBytes = kFewPointInputElements * sizeof(T);
    static constexpr uint32_t kFewPointCoordinateBytes = kFewPointCoordinateElements * sizeof(float);
    static constexpr uint32_t kFewWorkPlaneBytes = kFewWorkPlaneElements * sizeof(float);
    static constexpr uint32_t kFewPointResultBytes = kFewPointResultElements * sizeof(float);
    static constexpr uint32_t kFewDistanceOutputBytes = kFewDistanceOutputElements * sizeof(T);

    // The few-segment path has exclusive use of these large-path regions.
    static constexpr uint32_t kFewPointInputOffset = kGeometryOffset;
    static constexpr uint32_t kFewPointCoordinateOffset =
        AlignBytes(kFewPointInputOffset + kFewPointInputBytes);
    static constexpr uint32_t kFewCandidateIndicesOffset = kChunkResultOffset;
    static constexpr uint32_t kFewBestValuesOffset =
        AlignBytes(kFewCandidateIndicesOffset + kFewPointResultBytes);
    static constexpr uint32_t kFewBestIndicesOffset =
        AlignBytes(kFewBestValuesOffset + kFewPointResultBytes);
    static constexpr uint32_t kFewDistanceOutputOffset = kPointBroadcastOffset;
    static constexpr uint32_t kFewPointOffsetsOffset = kSegmentOffsetsOffset;

    static_assert(kFewPointCoordinateOffset + kFewPointCoordinateBytes <=
        kGeometryOffset + kGeometryBytes);
    static_assert(kFewWorkPlaneBytes <= kWorkPlaneBytes);
    static_assert(kFewBestIndicesOffset + kFewPointResultBytes <=
        kChunkResultOffset + kChunkResultBytes);
    static_assert(kFewDistanceOutputBytes <= kPointBroadcastBytes);
    static_assert(kFewPointTile * sizeof(int32_t) <= kSegmentOffsetBytes);

    __aicore__ inline LocalTensor<T> SegmentInput()
    {
        return LocalTensor<T>(
            TPosition::VECCALC, kSegmentInputOffset, kSegmentInputElements);
    }

    __aicore__ inline LocalTensor<T> GatherStaging()
    {
        return LocalTensor<T>(
            TPosition::VECCALC, kGatherStagingOffset, kGatherStagingElements);
    }

    __aicore__ inline LocalTensor<T> PointPipelineBuffer(uint32_t index)
    {
        return LocalTensor<T>(TPosition::VECCALC,
            kPointInputOffset + index * kPointRawBytes, kPointRawCount);
    }

    __aicore__ inline LocalTensor<float> Geometry()
    {
        return LocalTensor<float>(
            TPosition::VECCALC, kGeometryOffset, kGeometryElements);
    }

    __aicore__ inline LocalTensor<float> WorkPlanes()
    {
        return LocalTensor<float>(
            TPosition::VECCALC, kWorkPlaneOffset, kWorkPlaneElements);
    }

    __aicore__ inline LocalTensor<float> PointCoordinates()
    {
        return LocalTensor<float>(
            TPosition::VECCALC, kPointCoordinateOffset, kPointCoordinateElements);
    }

    __aicore__ inline LocalTensor<float> PointBroadcast()
    {
        return LocalTensor<float>(
            TPosition::VECCALC, kPointBroadcastOffset, kPointBroadcastRegionElements);
    }

    __aicore__ inline LocalTensor<float> ChunkResults()
    {
        return LocalTensor<float>(
            TPosition::VECCALC, kChunkResultOffset, kChunkResultElements);
    }

    __aicore__ inline LocalTensor<float> CandidateValues()
    {
        return LocalTensor<float>(
            TPosition::VECCALC, kCandidateValuesOffset, kPointResultElements);
    }

    __aicore__ inline LocalTensor<float> CandidateIndices()
    {
        return LocalTensor<float>(
            TPosition::VECCALC, kCandidateIndicesOffset, kPointResultElements);
    }

    __aicore__ inline LocalTensor<float> BestValues()
    {
        return PipelineBestValues(0);
    }

    __aicore__ inline LocalTensor<float> BestIndices()
    {
        return PipelineBestIndices(0);
    }

    __aicore__ inline LocalTensor<float> PipelineBestValues(uint32_t buffer)
    {
        return LocalTensor<float>(
            TPosition::VECCALC,
            kBestValuesOffset + buffer * kPointResultBytes, kPointResultElements);
    }

    __aicore__ inline LocalTensor<float> PipelineBestIndices(uint32_t buffer)
    {
        return LocalTensor<float>(
            TPosition::VECCALC,
            kBestIndicesOffset + buffer * kPointResultBytes, kPointResultElements);
    }

    __aicore__ inline LocalTensor<T> DistanceOutput()
    {
        return PipelineDistanceOutput(0);
    }

    __aicore__ inline LocalTensor<T> PipelineDistanceOutput(uint32_t buffer)
    {
        return LocalTensor<T>(
            TPosition::VECCALC,
            kDistanceOutputOffset + buffer * kDistanceOutputBytes,
            kDistanceOutputElements);
    }

    __aicore__ inline LocalTensor<int32_t> SegmentOffsets()
    {
        return LocalTensor<int32_t>(
            TPosition::VECCALC, kSegmentOffsetsOffset, kSegmentOffsetElements);
    }

    __aicore__ inline LocalTensor<int32_t> PointOffsets()
    {
        return LocalTensor<int32_t>(
            TPosition::VECCALC, kPointOffsetsOffset, kPointOffsetElements);
    }

    __aicore__ inline LocalTensor<int32_t> PointResultOffsets()
    {
        return LocalTensor<int32_t>(
            TPosition::VECCALC, kPointResultOffsetsOffset, kPointResultOffsetElements);
    }

    __aicore__ inline LocalTensor<T> FewPointInput()
    {
        return LocalTensor<T>(TPosition::VECCALC, kFewPointInputOffset, kFewPointInputElements);
    }

    __aicore__ inline LocalTensor<float> FewPointCoordinates()
    {
        return LocalTensor<float>(TPosition::VECCALC, kFewPointCoordinateOffset, kFewPointCoordinateElements);
    }

    __aicore__ inline LocalTensor<float> FewWorkPlanes()
    {
        return LocalTensor<float>(TPosition::VECCALC, kWorkPlaneOffset, kFewWorkPlaneElements);
    }

    __aicore__ inline LocalTensor<float> FewCandidateIndices()
    {
        return LocalTensor<float>(TPosition::VECCALC, kFewCandidateIndicesOffset, kFewPointResultElements);
    }

    __aicore__ inline LocalTensor<float> FewBestValues()
    {
        return LocalTensor<float>(TPosition::VECCALC, kFewBestValuesOffset, kFewPointResultElements);
    }

    __aicore__ inline LocalTensor<float> FewBestIndices()
    {
        return LocalTensor<float>(TPosition::VECCALC, kFewBestIndicesOffset, kFewPointResultElements);
    }

    __aicore__ inline LocalTensor<T> FewDistanceOutput()
    {
        return LocalTensor<T>(TPosition::VECCALC, kFewDistanceOutputOffset, kFewDistanceOutputElements);
    }

    __aicore__ inline LocalTensor<int32_t> FewPointOffsets()
    {
        return LocalTensor<int32_t>(TPosition::VECCALC, kFewPointOffsetsOffset, kFewPointTile);
    }

    __aicore__ inline LocalTensor<int32_t> ChunkGatherOffsets()
    {
        return LocalTensor<int32_t>(TPosition::VECCALC,
            kSegmentInputOffset, kChunkGatherElements);
    }

public:
    __aicore__ inline void Init(GM_ADDR points, GM_ADDR segments, GM_ADDR distances,
        GM_ADDR indices, const TreeLineMinDistanceTilingData& data)
    {
        data_.batches = data.batches;
        data_.pointsPerBatch = data.pointsPerBatch;
        data_.segmentsPerBatch = data.segmentsPerBatch;
        data_.totalPoints = data.totalPoints;
        data_.blockDim = data.blockDim;
        points_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points));
        segments_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(segments));
        distances_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(distances));
        indices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
        usePointVectors_ = false;
        useScalarGeometry_ = false;
        useScalarPointBroadcast_ = false;
        firstSegmentIssued_ = false;

        if constexpr (TINY_MODE == 0) {
            data_.coreDivisionMultiplier = data.coreDivisionMultiplier;
            data_.coreDivisionShift = data.coreDivisionShift;
            data_.batchDivisionMultiplier = data.batchDivisionMultiplier;
            data_.batchDivisionShift = data.batchDivisionShift;
            PrepareRegularPath();
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (TINY_MODE == 1) {
            // Minimal shapes finish in fewer cycles than a block dispatch costs,
            // so one block resolves every point and writes the two outputs
            // itself; parallel tiny blocks would share the output cache line.
            if (GetBlockIdx() == 0) {
                ProcessTinyStores(data_.totalPoints);
            }
            return;
        }

        const uint32_t core = GetBlockIdx();
        const uint32_t begin = CoreBoundary(core);
        const uint32_t end = CoreBoundary(core + 1);
        if (useScalarGeometry_) {
            ProcessScalarRange(begin, end);
            return;
        }
        ProcessPointRange(begin, end);
    }

private:
    __aicore__ inline void PrepareRegularPath()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t corePoints = CoreBoundary(core + 1) - CoreBoundary(core);
        usePointVectors_ = data_.segmentsPerBatch <= kPointMajorSegmentLimit &&
            uint64_t(data_.segmentsPerBatch) * kPointMajorLaneTarget <= data_.totalPoints;
        useScalarGeometry_ =
            uint64_t(corePoints) * data_.segmentsPerBatch <= kFewSegmentThreshold;
        if (useScalarGeometry_) {
            return;
        }
        if (usePointVectors_) {
            InitializePointVectorPath(corePoints);
            return;
        }
        useScalarPointBroadcast_ =
            corePoints <= kScalarPointBroadcastPointLimit &&
            data_.segmentsPerBatch > kSegmentTile;
        InitializeMatrixPipeline(core);
    }

    __aicore__ inline void InitializePointVectorPath(uint32_t corePoints)
    {
        const uint32_t pointLaneCount =
            corePoints < kFewPointTile ? corePoints : kFewPointTile;
        auto offsets = FewPointOffsets();
        SeedIndexSequence(offsets, pointLaneCount);
        Synchronize<HardEvent::S_V>();
        DoubleSequence(offsets, pointLaneCount);
        Muls(offsets, offsets, static_cast<int32_t>(3 * sizeof(T)), pointLaneCount);
    }

    __aicore__ inline void InitializeMatrixPipeline(uint32_t core)
    {
        if (!useScalarPointBroadcast_) {
            SetFlag<HardEvent::MTE3_V>(kOutputPipelineEventBase);
            SetFlag<HardEvent::MTE3_V>(kOutputPipelineEventBase + 1);
        }
        const uint32_t begin = CoreBoundary(core);
        const uint32_t end = CoreBoundary(core + 1);
        if (begin < end) {
            IssueSegmentCopy(BatchIndex(begin));
            firstSegmentIssued_ = true;
        }
        InitializeGatherOffsets();
        if (!useScalarPointBroadcast_) {
            Duplicate(PointCoordinates(), 0.0F, 3 * kPointTile);
            SetFlag<HardEvent::V_MTE2>(kPointPipelineEventBase);
            SetFlag<HardEvent::V_MTE2>(kPointPipelineEventBase + 1);
        }
    }

    __aicore__ inline void ProcessPointRange(uint32_t begin, uint32_t end)
    {
        uint32_t loadedBatch = data_.batches;
        bool pipelineStart = true;
        uint32_t tileBuffer = kPointPipelineBuffers - 1;
        for (uint32_t offset = begin; offset < end;) {
            const uint32_t batch = BatchIndex(offset);
            if (!usePointVectors_ && batch != loadedBatch) {
                PrepareBatchSegments(batch, offset, end);
                loadedBatch = batch;
            }
            const uint32_t pointTile =
                usePointVectors_ ? kFewPointTile : kPointTile;
            const uint32_t count = PointCount(offset, batch, end, pointTile);
            const uint32_t prefetchCount = usePointVectors_
                ? 0U : PrefetchCount(offset, count, end, pointTile);
            tileBuffer = (tileBuffer + 1) % kPointPipelineBuffers;
            ProcessPoints(offset, batch, count, tileBuffer, prefetchCount,
                pipelineStart);
            pipelineStart = false;
            offset += count;
        }
        if (!usePointVectors_ && !useScalarPointBroadcast_) {
            DrainPointPipeline();
            WaitFlag<HardEvent::MTE3_V>(kOutputPipelineEventBase);
            WaitFlag<HardEvent::MTE3_V>(kOutputPipelineEventBase + 1);
        }
    }

    __aicore__ inline void PrepareBatchSegments(
        uint32_t batch, uint32_t offset, uint32_t end)
    {
        LoadSegments(batch, firstSegmentIssued_);
        firstSegmentIssued_ = false;
        if (CeilDiv(data_.segmentsPerBatch, kSegmentTile) >=
                kGroupedReductionMinChunks &&
            end - offset > 4 && BatchEnd(batch) - offset > 4) {
            // Geometry now owns every segment value; raw input holds gather indices.
            InitializeChunkGatherOffsets();
        }
    }

    __aicore__ inline uint32_t PointCount(
        uint32_t offset, uint32_t batch, uint32_t end, uint32_t pointTile)
    {
        const uint32_t rangeRemaining = end - offset;
        const uint32_t batchRemaining = BatchEnd(batch) - offset;
        const uint32_t count = rangeRemaining < pointTile
            ? rangeRemaining : pointTile;
        return count < batchRemaining ? count : batchRemaining;
    }

    static __aicore__ inline uint32_t PrefetchCount(
        uint32_t offset, uint32_t count, uint32_t end, uint32_t pointTile)
    {
        const uint32_t available = end - offset - count;
        return available == 0
            ? 0U : (available < pointTile ? available : pointTile);
    }

    static __aicore__ inline uint32_t DivideWithMetadata(
        uint32_t value, uint32_t multiplier, uint32_t shift)
    {
        const uint64_t highProduct = (uint64_t(value) * multiplier) >> 32;
        return static_cast<uint32_t>((highProduct + value) >> shift);
    }

    __aicore__ inline uint32_t CoreBoundary(uint32_t core)
    {
        const uint64_t scaledTotal = uint64_t(core) * data_.totalPoints;
        if (scaledTotal <= 0xFFFFFFFFULL) {
            return DivideWithMetadata(static_cast<uint32_t>(scaledTotal),
                data_.coreDivisionMultiplier, data_.coreDivisionShift);
        }
        return data_.blockDim == 0
            ? 0U : static_cast<uint32_t>(scaledTotal / data_.blockDim);
    }

    __aicore__ inline uint32_t BatchIndex(uint32_t point)
    {
        return DivideWithMetadata(point,
            data_.batchDivisionMultiplier, data_.batchDivisionShift);
    }

    __aicore__ inline uint32_t BatchEnd(uint32_t batch)
    {
        return static_cast<uint32_t>(
            (uint64_t(batch) + 1) * data_.pointsPerBatch);
    }

    __aicore__ inline uint64_t SegmentBase(uint32_t batch)
    {
        return uint64_t(batch) * data_.segmentsPerBatch * 6;
    }

    static __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t alignment)
    {
        return (value + alignment - 1) / alignment;
    }

    static __aicore__ inline void SeedIndexSequence(
        LocalTensor<int32_t> sequence, uint32_t count)
    {
        const uint32_t seedCount =
            count < kIndexSeedElements ? count : kIndexSeedElements;
        for (uint32_t index = 0; index < seedCount; ++index) {
            sequence.SetValue(index, static_cast<int32_t>(index));
        }
    }

    static __aicore__ inline void DoubleSequence(
        LocalTensor<int32_t> sequence, uint32_t count)
    {
        const uint32_t seedCount =
            count < kIndexSeedElements ? count : kIndexSeedElements;
        for (uint32_t filled = seedCount; filled < count;) {
            const uint32_t remaining = count - filled;
            const uint32_t appendCount = remaining < filled ? remaining : filled;
            Adds(sequence[filled], sequence,
                static_cast<int32_t>(filled), appendCount);
            PipeBarrier<PIPE_V>();
            filled += appendCount;
        }
    }

    __aicore__ inline void InitializeGatherOffsets()
    {
        auto segmentIndex = SegmentOffsets();
        auto pointIndex = PointOffsets();
        auto pointResultOffsets = PointResultOffsets();
        const uint32_t pointLaneCount =
            data_.pointsPerBatch < kPointTile
                ? data_.pointsPerBatch
                : kPointTile;
        const uint32_t chunkCount =
            CeilDiv(data_.segmentsPerBatch, kSegmentTile);
        const uint32_t pointResultMinimum =
            pointLaneCount < chunkCount ? chunkCount : pointLaneCount;
        const uint32_t pointResultLaneCount = chunkCount >= kGroupedReductionMinChunks
            ? CeilDiv(pointResultMinimum, kFloatElementsPerBlock) * kFloatElementsPerBlock
            : pointResultMinimum;
        const uint32_t segmentIndexCount = data_.segmentsPerBatch;
        const uint32_t indexCount = segmentIndexCount > pointResultLaneCount
            ? segmentIndexCount
            : pointResultLaneCount;
        SeedIndexSequence(segmentIndex, indexCount);
        Synchronize<HardEvent::S_V>();
        DoubleSequence(segmentIndex, indexCount);
        if (!useScalarPointBroadcast_) {
            Muls(pointIndex, segmentIndex, static_cast<int32_t>(3 * sizeof(T)),
                pointLaneCount);
        }
        Muls(pointResultOffsets, segmentIndex,
            static_cast<int32_t>(kChunkResultStride / kPointTile * sizeof(float)),
            pointResultLaneCount);
        Muls(segmentIndex, segmentIndex,
            static_cast<int32_t>(6 * sizeof(T)), segmentIndexCount);
    }

    __aicore__ inline void InitializeChunkGatherOffsets()
    {
        const uint32_t points = data_.pointsPerBatch < kPointTile
            ? data_.pointsPerBatch : kPointTile;
        const uint32_t chunks = CeilDiv(data_.segmentsPerBatch, kSegmentTile);
        const uint32_t rowElements =
            CeilDiv(chunks, kFloatElementsPerBlock) * kFloatElementsPerBlock;
        auto offsets = ChunkGatherOffsets();
        auto pointOffsets = PointResultOffsets();
        auto chunkOffsets = CandidateValues().template ReinterpretCast<int32_t>();
        Duplicate(chunkOffsets, int32_t(0), kChunkRowElements);
        Muls(chunkOffsets, pointOffsets, int32_t(kMatrixChunkResultStride / 2), chunks);
        const uint8_t repeats = static_cast<uint8_t>(CeilDiv(points, kFloatElementsPerBlock));
        const uint8_t rowBlocks = static_cast<uint8_t>(rowElements / kFloatElementsPerBlock);
        Brcb(offsets, pointOffsets, repeats, BrcbRepeatParams{rowBlocks,
            static_cast<uint8_t>(kFloatElementsPerBlock * rowBlocks)});
        if (rowBlocks == 2) {
            Brcb(offsets[kFloatElementsPerBlock], pointOffsets, repeats,
                BrcbRepeatParams{rowBlocks,
                    static_cast<uint8_t>(kFloatElementsPerBlock * rowBlocks)});
        }
        Add(offsets, offsets, chunkOffsets, uint64_t(rowElements),
            static_cast<uint8_t>(points),
            BinaryRepeatParams{1, 1, 1, rowBlocks, rowBlocks, 0});
    }

    __aicore__ inline void IssueSegmentCopy(uint32_t batch)
    {
        const uint32_t segmentCount = data_.segmentsPerBatch;
        DataCopyExtParams copy{1, segmentCount * 6 * uint32_t(sizeof(T)), 0, 0, 0};
        DataCopyPad(SegmentInput(), segments_[SegmentBase(batch)], copy,
            DataCopyPadExtParams<T>{false, 0, 0, 0});
    }

    __aicore__ inline void LoadSegments(uint32_t batch, bool copyIssued)
    {
        const uint32_t segmentCount = data_.segmentsPerBatch;
        auto raw = SegmentInput();
        if (!copyIssued) {
            Synchronize<HardEvent::V_MTE2>();
            IssueSegmentCopy(batch);
        }
        Synchronize<HardEvent::MTE2_V>();
        LoadSegmentGeometry(raw, segmentCount);
        PrepareSegmentGeometry(segmentCount);
        PadSegmentGeometry(segmentCount);
    }

    __aicore__ inline void LoadSegmentGeometry(
        LocalTensor<T> raw, uint32_t segmentCount)
    {
        auto geometry = Geometry();
        auto offsets = SegmentOffsets().template ReinterpretCast<uint32_t>();
        if constexpr (sizeof(T) == sizeof(float)) {
            Gather(geometry[kStartXPlane], raw, offsets, 0, segmentCount);
            Gather(geometry[kStartYPlane], raw, offsets, sizeof(T), segmentCount);
            Gather(geometry[kStartZPlane], raw, offsets, 2 * sizeof(T), segmentCount);
            Gather(geometry[kDirectionXPlane], raw, offsets, 3 * sizeof(T), segmentCount);
            Gather(geometry[kDirectionYPlane], raw, offsets, 4 * sizeof(T), segmentCount);
            Gather(geometry[kDirectionZPlane], raw, offsets, 5 * sizeof(T), segmentCount);
        } else {
            auto planar = GatherStaging();
            Gather(planar[0], raw, offsets, 0, segmentCount);
            Gather(planar[kMaxSegments], raw, offsets, sizeof(T), segmentCount);
            Gather(planar[2 * kMaxSegments], raw, offsets, 2 * sizeof(T), segmentCount);
            Gather(planar[3 * kMaxSegments], raw, offsets, 3 * sizeof(T), segmentCount);
            Gather(planar[4 * kMaxSegments], raw, offsets, 4 * sizeof(T), segmentCount);
            Gather(planar[5 * kMaxSegments], raw, offsets, 5 * sizeof(T), segmentCount);
            Cast(geometry[kStartXPlane], planar[0], RoundMode::CAST_NONE, segmentCount);
            Cast(geometry[kStartYPlane], planar[kMaxSegments], RoundMode::CAST_NONE, segmentCount);
            Cast(geometry[kStartZPlane], planar[2 * kMaxSegments], RoundMode::CAST_NONE, segmentCount);
            Cast(geometry[kDirectionXPlane], planar[3 * kMaxSegments], RoundMode::CAST_NONE, segmentCount);
            Cast(geometry[kDirectionYPlane], planar[4 * kMaxSegments], RoundMode::CAST_NONE, segmentCount);
            Cast(geometry[kDirectionZPlane], planar[5 * kMaxSegments], RoundMode::CAST_NONE, segmentCount);
        }
    }

    __aicore__ inline void PrepareSegmentGeometry(uint32_t segmentCount)
    {
        auto geometry = Geometry();
        auto directionX = geometry[kDirectionXPlane];
        auto directionY = geometry[kDirectionYPlane];
        auto directionZ = geometry[kDirectionZPlane];
        auto scale = geometry[kScalePlane];
        auto temporary = WorkPlanes();
        Sub(directionX, directionX, geometry[kStartXPlane], segmentCount);
        Sub(directionY, directionY, geometry[kStartYPlane], segmentCount);
        Sub(directionZ, directionZ, geometry[kStartZPlane], segmentCount);
        Mul(scale, directionX, directionX, segmentCount);
        Mul(temporary, directionY, directionY, segmentCount);
        Add(scale, scale, temporary, segmentCount);
        Mul(temporary, directionZ, directionZ, segmentCount);
        Add(scale, scale, temporary, segmentCount);

        // A degenerate segment has an exactly zero direction, so any finite
        // scale keeps its projection weight at zero. Flooring the squared
        // length at the smallest normal keeps the reciprocal finite and maps
        // that case to the distance from the segment start without a mask.
        Maxs(scale, scale, kFloatMin, segmentCount);
        Duplicate(temporary, -1.0F, segmentCount);
        Div(scale, temporary, scale, segmentCount);
    }

    __aicore__ inline void PadSegmentGeometry(uint32_t segmentCount)
    {
        const uint32_t tail = segmentCount % kSegmentTile;
        if (tail != 0) {
            const uint32_t chunkStart = segmentCount - tail;
            uint64_t paddingMask[2] = {~uint64_t(0) << tail, 0};
            Duplicate(Geometry()[chunkStart], 0.0F, paddingMask,
                static_cast<uint8_t>(kGeometryPlanes), 1, kGeometryPlaneBlocks);
            ResetMask();
        }
    }

    __aicore__ inline void LoadPointTile(
        uint32_t offset, uint32_t count, uint32_t buffer)
    {
        const int32_t event = kPointPipelineEventBase + static_cast<int32_t>(buffer);
        WaitFlag<HardEvent::V_MTE2>(event);
        auto raw = PointPipelineBuffer(buffer);
        DataCopyExtParams copy{1, count * 3 * uint32_t(sizeof(T)), 0, 0, 0};
        DataCopyPad(raw, points_[uint64_t(offset) * 3], copy,
            DataCopyPadExtParams<T>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void DrainPointPipeline()
    {
        WaitFlag<HardEvent::V_MTE2>(kPointPipelineEventBase);
        WaitFlag<HardEvent::V_MTE2>(kPointPipelineEventBase + 1);
    }

    __aicore__ inline void LoadScalarPointBroadcast(uint32_t offset, uint32_t count)
    {
        auto broadcast = PointBroadcast();
        for (uint32_t point = 0; point < count; ++point) {
            const uint64_t source = uint64_t(offset + point) * 3;
            const uint32_t lane = point * kFloatElementsPerBlock;
            Duplicate(broadcast[lane],
                static_cast<float>(points_.GetValue(source)),
                kFloatElementsPerBlock);
            Duplicate(broadcast[kPointBroadcastElements + lane],
                static_cast<float>(points_.GetValue(source + 1)),
                kFloatElementsPerBlock);
            Duplicate(broadcast[2 * kPointBroadcastElements + lane],
                static_cast<float>(points_.GetValue(source + 2)),
                kFloatElementsPerBlock);
        }
    }

    __aicore__ inline void ProcessPoints(uint32_t offset, uint32_t batch,
        uint32_t count, uint32_t tileBuffer, uint32_t prefetchCount,
        bool pipelineStart)
    {
        if (usePointVectors_) {
            ProcessFewPoints(offset, batch, count);
            return;
        }
        if (useScalarPointBroadcast_) {
            ProcessScalarPointTile(offset, count);
            return;
        }
        ProcessMatrixPoints(offset, count, tileBuffer, prefetchCount,
            pipelineStart);
    }

    __aicore__ inline void ProcessScalarPointTile(uint32_t offset, uint32_t count)
    {
        LoadScalarPointBroadcast(offset, count);
        ComputeWideSegmentResults(count);
        MergeWideSegmentResults(count, BestValues(), BestIndices());
        WriteOutput(offset, count, BestValues(),
            BestIndices().template ReinterpretCast<int32_t>(), DistanceOutput());
    }

    __aicore__ inline void ProcessMatrixPoints(uint32_t offset, uint32_t count,
        uint32_t tileBuffer, uint32_t prefetchCount, bool pipelineStart)
    {
        LoadPointCoordinates(offset, count, tileBuffer, prefetchCount,
            pipelineStart);
        BuildPointBroadcast(count);
        if (count <= 4 && data_.segmentsPerBatch > kSegmentTile) {
            ProcessWidePointTile(offset, count, tileBuffer);
            return;
        }
        ProcessChunkedPointTile(offset, count, tileBuffer);
    }

    __aicore__ inline void LoadPointCoordinates(uint32_t offset, uint32_t count,
        uint32_t tileBuffer, uint32_t prefetchCount, bool pipelineStart)
    {
        const int32_t event = kPointPipelineEventBase + static_cast<int32_t>(tileBuffer);
        if (pipelineStart) {
            LoadPointTile(offset, count, tileBuffer);
        }
        WaitFlag<HardEvent::MTE2_V>(event);
        auto raw = PointPipelineBuffer(tileBuffer);

        auto coordinates = PointCoordinates();
        auto offsets = PointOffsets().template ReinterpretCast<uint32_t>();
        if constexpr (sizeof(T) == sizeof(float)) {
            Gather(coordinates[0], raw, offsets, 0, count);
            Gather(coordinates[kPointTile], raw, offsets, sizeof(T), count);
            Gather(coordinates[2 * kPointTile], raw, offsets, 2 * sizeof(T), count);
        } else {
            auto planar = GatherStaging();
            Duplicate(planar, static_cast<half>(0.0F), 3 * kPointTile);
            Gather(planar[0], raw, offsets, 0, count);
            Gather(planar[kPointTile], raw, offsets, sizeof(T), count);
            Gather(planar[2 * kPointTile], raw, offsets, 2 * sizeof(T), count);
        }
        SetFlag<HardEvent::V_MTE2>(event);
        if (prefetchCount != 0) {
            LoadPointTile(offset + count, prefetchCount,
                (tileBuffer + 1) % kPointPipelineBuffers);
        }
        if constexpr (sizeof(T) != sizeof(float)) {
            auto planar = GatherStaging();
            Cast(coordinates, planar, RoundMode::CAST_NONE, 3 * kPointTile);
        }
    }

    __aicore__ inline void BuildPointBroadcast(uint32_t count)
    {
        auto broadcast = PointBroadcast();
        auto coordinates = PointCoordinates();
        const uint8_t pointRepeats = static_cast<uint8_t>((count + 7) / 8);
        Brcb(broadcast[0], coordinates[0], pointRepeats, BrcbRepeatParams{1, 8});
        Brcb(broadcast[kPointBroadcastElements], coordinates[kPointTile],
            pointRepeats, BrcbRepeatParams{1, 8});
        Brcb(broadcast[2 * kPointBroadcastElements], coordinates[2 * kPointTile],
            pointRepeats, BrcbRepeatParams{1, 8});
    }

    __aicore__ inline void ProcessWidePointTile(
        uint32_t offset, uint32_t count, uint32_t tileBuffer)
    {
        ComputeWideSegmentResults(count);
        WaitForOutputBuffer(tileBuffer);
        auto bestValues = PipelineBestValues(tileBuffer);
        auto bestIndices = PipelineBestIndices(tileBuffer);
        MergeWideSegmentResults(count, bestValues, bestIndices);
        WritePipelineOutput(offset, count, bestValues,
            bestIndices.template ReinterpretCast<int32_t>(),
            PipelineDistanceOutput(tileBuffer), tileBuffer);
    }

    __aicore__ inline void ProcessChunkedPointTile(
        uint32_t offset, uint32_t count, uint32_t tileBuffer)
    {
        auto distance = WorkPlanes()[kDistancePlane];
        const uint32_t segmentCount = data_.segmentsPerBatch;
        const uint32_t chunkCount = CeilDiv(segmentCount, kSegmentTile);
        const bool groupedReduction = chunkCount >= kGroupedReductionMinChunks;
        auto chunkResults = ChunkResults();
        const uint32_t chunkStride = groupedReduction ? kMatrixChunkResultStride : kChunkResultStride;
        for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
            const uint32_t remaining = segmentCount - chunk * kSegmentTile;
            const uint32_t validSegments =
                remaining < kSegmentTile ? remaining : kSegmentTile;
            ComputeSegmentChunk(chunk * kSegmentTile, count);
            WholeReduceMin<float>(
                chunkResults[chunk * chunkStride],
                distance,
                validSegments,
                count,
                1,
                1,
                8,
                ReduceOrder::ORDER_VALUE_INDEX);
        }

        WaitForOutputBuffer(tileBuffer);
        auto bestValues = PipelineBestValues(tileBuffer);
        auto bestIndices = PipelineBestIndices(tileBuffer);
        if (groupedReduction) {
            MergeGroupedChunkResults(chunkCount, count, bestValues, bestIndices);
        } else {
            MergeChunkResults(chunkCount, count, bestValues, bestIndices);
        }
        WritePipelineOutput(offset, count, bestValues,
            bestIndices.template ReinterpretCast<int32_t>(),
            PipelineDistanceOutput(tileBuffer), tileBuffer);
    }

    __aicore__ inline void ProcessScalarRange(uint32_t begin, uint32_t end)
    {
        if (begin == end) {
            return;
        }
        Synchronize<HardEvent::V_S>();
        auto minimum = BestValues();
        auto index = BestIndices().template ReinterpretCast<int32_t>();
        ComputeScalarOutputs(begin, end, minimum, index);
        Synchronize<HardEvent::S_V>();
        WriteOutput(begin, end - begin, minimum, index, DistanceOutput());
    }

    struct ScalarNearest {
        float squaredDistance;
        int32_t segmentIndex;
    };

    struct ScalarSegmentGeometry {
        float startX;
        float startY;
        float startZ;
        float directionX;
        float directionY;
        float directionZ;
        float lengthSquared;
    };

    __aicore__ inline void ComputeScalarOutputs(uint32_t begin, uint32_t end,
        LocalTensor<float> minimum, LocalTensor<int32_t> index)
    {
        for (uint32_t offset = begin; offset < end;) {
            const uint32_t batch = BatchIndex(offset);
            const uint32_t batchEnd = BatchEnd(batch);
            const uint32_t rangeEnd = end < batchEnd ? end : batchEnd;
            const uint64_t segmentBase = SegmentBase(batch);
            for (; offset < rangeEnd; ++offset) {
                const ScalarNearest nearest = FindNearestSegment(segmentBase,
                    static_cast<float>(points_.GetValue(uint64_t(offset) * 3)),
                    static_cast<float>(points_.GetValue(uint64_t(offset) * 3 + 1)),
                    static_cast<float>(points_.GetValue(uint64_t(offset) * 3 + 2)));
                const uint32_t localPoint = offset - begin;
                minimum.SetValue(localPoint, nearest.squaredDistance);
                index.SetValue(localPoint, nearest.segmentIndex);
            }
        }
    }

    __aicore__ inline ScalarNearest FindNearestSegment(uint64_t segmentBase,
        float pointX, float pointY, float pointZ)
    {
        float best = kFloatMax;
        int32_t nearest = 0;
        for (uint32_t segment = 0; segment < data_.segmentsPerBatch; ++segment) {
            const ScalarSegmentGeometry geometry =
                ReadScalarSegmentGeometry(segmentBase, segment);
            const float denominator =
                geometry.lengthSquared +
                (geometry.lengthSquared == 0.0F ? 1.0F : 0.0F);
            const float inverseLength = 1.0F / denominator;
            float x = pointX - geometry.startX;
            float y = pointY - geometry.startY;
            float z = pointZ - geometry.startZ;
            float projection =
                ((x * geometry.directionX + y * geometry.directionY) +
                    z * geometry.directionZ) *
                inverseLength;
            projection = projection < 0.0F ? 0.0F : projection;
            projection = projection > 1.0F ? 1.0F : projection;
            x -= projection * geometry.directionX;
            y -= projection * geometry.directionY;
            z -= projection * geometry.directionZ;
            const float distanceSquared = (x * x + y * y) + z * z;
            if (distanceSquared < best) {
                best = distanceSquared;
                nearest = static_cast<int32_t>(segment);
            }
        }
        return {best, nearest};
    }

    __aicore__ inline void ProcessTinyStores(uint32_t total)
    {
        LocalTensor<float> squaredDistances(
            TPosition::VECCALC, 0, kTinyPointBound);
        LocalTensor<int32_t> nearestIndices(
            TPosition::VECCALC, 1024, kTinyPointBound);
        ComputeScalarOutputs(0, total, squaredDistances, nearestIndices);

        // The nearest indices are independent of the distance transform. Start
        // their transfer while the vector pipe transforms squared distances.
        Synchronize<HardEvent::S_MTE3>();
        DataCopyPad(indices_, nearestIndices,
            DataCopyExtParams{1, uint32_t(total * sizeof(int32_t)), 0, 0, 0});
        Synchronize<HardEvent::S_V>();
        Sqrt(squaredDistances, squaredDistances, total);
        if constexpr (sizeof(T) == sizeof(float)) {
            Synchronize<HardEvent::V_MTE3>();
            DataCopyPad(distances_, squaredDistances,
                DataCopyExtParams{1, uint32_t(total * sizeof(T)), 0, 0, 0});
        } else {
            LocalTensor<T> output(
                TPosition::VECCALC, 2048, kTinyPointBound);
            Cast(output, squaredDistances, RoundMode::CAST_RINT, total);
            Synchronize<HardEvent::V_MTE3>();
            DataCopyPad(distances_, output,
                DataCopyExtParams{1, uint32_t(total * sizeof(T)), 0, 0, 0});
        }
    }

    __aicore__ inline void ProcessFewPoints(uint32_t offset, uint32_t batch,
        uint32_t count)
    {
        Synchronize<HardEvent::V_MTE2>();
        auto raw = FewPointInput();
        DataCopyExtParams copy{1, count * 3 * uint32_t(sizeof(T)), 0, 0, 0};
        DataCopyPad(raw, points_[uint64_t(offset) * 3], copy,
            DataCopyPadExtParams<T>{false, 0, 0, 0});
        Synchronize<HardEvent::MTE2_V>();

        auto coordinates = FewPointCoordinates();
        auto offsets = FewPointOffsets().template ReinterpretCast<uint32_t>();
        if constexpr (sizeof(T) == sizeof(float)) {
            Gather(coordinates[0], raw, offsets, 0, count);
            Gather(coordinates[kFewPointTile], raw, offsets, sizeof(T), count);
            Gather(coordinates[2 * kFewPointTile], raw, offsets,
                2 * sizeof(T), count);
        } else {
            auto planar = GatherStaging();
            Duplicate(planar, static_cast<half>(0.0F), 3 * kFewPointTile);
            Gather(planar[0], raw, offsets, 0, count);
            Gather(planar[kFewPointTile], raw, offsets, sizeof(T), count);
            Gather(planar[2 * kFewPointTile], raw, offsets,
                2 * sizeof(T), count);
            Cast(coordinates, planar, RoundMode::CAST_NONE,
                3 * kFewPointTile);
        }

        const uint64_t segmentBase = SegmentBase(batch);
        if (data_.segmentsPerBatch == 1) {
            // A batch holding one segment has nothing to compare: the running
            // minimum is that segment's distance and the index is zero, so the
            // compare/select ladder would only rewrite the same lane values.
            auto index = FewBestIndices().template ReinterpretCast<int32_t>();
            auto distance = ComputeSegmentDistance(segmentBase, 0, count);
            Duplicate(index, 0, count);
            WriteOutput(offset, count, distance, index, FewDistanceOutput());
            return;
        }
        ProcessFewSegments(segmentBase, count);
        WriteOutput(offset, count, FewBestValues(),
            FewBestIndices().template ReinterpretCast<int32_t>(),
            FewDistanceOutput());
    }

    __aicore__ inline LocalTensor<float> ComputeSegmentDistance(
        uint64_t segmentBase, uint32_t segment, uint32_t pointCount)
    {
        auto coordinates = FewPointCoordinates();
        auto work = FewWorkPlanes();
        auto x = work[kFewXPlane];
        auto y = work[kFewYPlane];
        auto z = work[kFewZPlane];
        auto projection = work[kFewProjectionPlane];
        auto temporary = work[kFewTemporaryPlane];
        const ScalarSegmentGeometry geometry =
            ReadScalarSegmentGeometry(segmentBase, segment);
        const float denominator = geometry.lengthSquared +
            (geometry.lengthSquared == 0.0F ? 1.0F : 0.0F);
        const float scale = 1.0F / denominator;

        Adds(x, coordinates[0], -geometry.startX, pointCount);
        Adds(y, coordinates[kFewPointTile], -geometry.startY, pointCount);
        Adds(z, coordinates[2 * kFewPointTile], -geometry.startZ, pointCount);

        Muls(projection, x, geometry.directionX, pointCount);
        Muls(temporary, y, geometry.directionY, pointCount);
        Add(projection, projection, temporary, pointCount);
        Muls(temporary, z, geometry.directionZ, pointCount);
        Add(projection, projection, temporary, pointCount);
        Muls(projection, projection, scale, pointCount);
        Maxs(projection, projection, 0.0F, pointCount);
        Mins(projection, projection, 1.0F, pointCount);

        Muls(temporary, projection, geometry.directionX, pointCount);
        Sub(x, x, temporary, pointCount);
        Muls(temporary, projection, geometry.directionY, pointCount);
        Sub(y, y, temporary, pointCount);
        Muls(temporary, projection, geometry.directionZ, pointCount);
        Sub(z, z, temporary, pointCount);

        Mul(projection, x, x, pointCount);
        Mul(temporary, y, y, pointCount);
        Add(projection, projection, temporary, pointCount);
        Mul(temporary, z, z, pointCount);
        Add(projection, projection, temporary, pointCount);
        return projection;
    }

    __aicore__ inline ScalarSegmentGeometry ReadScalarSegmentGeometry(
        uint64_t segmentBase, uint32_t segment)
    {
        const uint64_t base = segmentBase + uint64_t(segment) * 6;
        const float startX = static_cast<float>(segments_.GetValue(base));
        const float startY = static_cast<float>(segments_.GetValue(base + 1));
        const float startZ = static_cast<float>(segments_.GetValue(base + 2));
        const float directionX =
            static_cast<float>(segments_.GetValue(base + 3)) - startX;
        const float directionY =
            static_cast<float>(segments_.GetValue(base + 4)) - startY;
        const float directionZ =
            static_cast<float>(segments_.GetValue(base + 5)) - startZ;
        const float lengthSquared =
            directionX * directionX + directionY * directionY +
            directionZ * directionZ;
        return {startX, startY, startZ, directionX, directionY, directionZ,
            lengthSquared};
    }

    __aicore__ inline void ProcessFewSegments(uint64_t segmentBase,
        uint32_t pointCount)
    {
        auto bestValues = FewBestValues();
        auto bestIndexData =
            FewBestIndices().template ReinterpretCast<int32_t>();
        auto candidateIndexData =
            FewCandidateIndices().template ReinterpretCast<int32_t>();
        auto bestIndexBits = FewBestIndices();
        auto candidateIndexBits = FewCandidateIndices();
        const BinaryRepeatParams rowWise{1, 1, 1, 8, 8, 8};

        Duplicate(bestValues, kFloatMax, pointCount);
        Duplicate(bestIndexData, 0, pointCount);
        for (uint32_t segment = 0; segment < data_.segmentsPerBatch; ++segment) {
            auto projection =
                ComputeSegmentDistance(segmentBase, segment, pointCount);
            Duplicate(candidateIndexData, static_cast<int32_t>(segment), pointCount);
            for (uint32_t pointOffset = 0; pointOffset < pointCount;
                 pointOffset += kPointTile) {
                const uint32_t remaining = pointCount - pointOffset;
                const uint32_t maskLength =
                    remaining < kPointTile
                        ? remaining
                        : kPointTile;
                Compare(projection[pointOffset], bestValues[pointOffset],
                    CMPMODE::LT, static_cast<uint64_t>(maskLength), rowWise);
                PipeBarrier<PIPE_V>();
                SetMaskNorm();
                SetVectorMask<float, MaskMode::NORMAL>(maskLength);
                Select<float, SELMODE::VSEL_CMPMASK_SPR>(
                    bestValues[pointOffset], projection[pointOffset],
                    bestValues[pointOffset], 1, rowWise);
                Select<float, SELMODE::VSEL_CMPMASK_SPR>(
                    bestIndexBits[pointOffset], candidateIndexBits[pointOffset],
                    bestIndexBits[pointOffset], 1, rowWise);
                ResetMask();
            }
        }
    }

    __aicore__ inline void ComputeSegmentChunk(uint32_t segmentOffset,
        uint32_t pointCount)
    {
        SetMaskNorm();
        SetVectorMask<float, MaskMode::NORMAL>(kSegmentTile);
        ComputeDistances(segmentOffset, 0, MASK_PLACEHOLDER, pointCount,
            BinaryRepeatParams{1, 1, 0, 8, 0, 1},
            BinaryRepeatParams{1, 1, 1, 8, 8, 0});
        ResetMask();
    }

    __aicore__ inline void ComputeWideSegmentResults(uint32_t pointCount)
    {
        const uint32_t segments = data_.segmentsPerBatch;
        const uint32_t fullChunks = segments / kSegmentTile;
        const uint32_t tail = segments % kSegmentTile;
        const uint32_t chunkCount = CeilDiv(segments, kSegmentTile);
        auto work = WorkPlanes();
        auto results = ChunkResults();
        auto chunkOffsets = CandidateIndices().template ReinterpretCast<int32_t>();
        auto pointOffsets = PointResultOffsets();
        Muls(chunkOffsets, pointOffsets, int32_t(4), chunkCount);
        auto gatherOffsets = chunkOffsets.template ReinterpretCast<uint32_t>();
        for (uint32_t point = 0; point < pointCount; ++point) {
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(segments);
            ComputeDistances(0, point * kFloatElementsPerBlock, MASK_PLACEHOLDER, 1,
                BinaryRepeatParams{1, 1, 0, 8, 8, 0},
                BinaryRepeatParams{1, 1, 1, 8, 8, 8});
            SetMaskNorm();
            ResetMask();

            auto chunks = results[(point + 1) * kChunkResultStride];
            WholeReduceMin<float>(chunks, work[kDistancePlane], kSegmentTile,
                fullChunks, 4, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
            if (tail != 0) {
                WholeReduceMin<float>(chunks[fullChunks * kFloatElementsPerBlock],
                    work[kDistancePlane + fullChunks * kSegmentTile], tail,
                    1, 4, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
            }
            Gather(CandidateValues(), chunks, gatherOffsets, 0, chunkCount);
            WholeReduceMin<float>(results[point * kFloatElementsPerBlock],
                CandidateValues(), chunkCount, 1, 1, 1, 8,
                ReduceOrder::ORDER_VALUE_INDEX);
        }
    }

    __aicore__ inline void MergeWideSegmentResults(uint32_t pointCount,
        LocalTensor<float> bestValues, LocalTensor<float> bestIndices)
    {
        auto results = ChunkResults();
        auto chunkOffsets = CandidateIndices().template ReinterpretCast<int32_t>();
        auto pointOffsets = PointResultOffsets();
        Muls(chunkOffsets, pointOffsets, int32_t(4), pointCount);
        auto gatherOffsets = chunkOffsets.template ReinterpretCast<uint32_t>();
        Gather(bestValues, results, gatherOffsets, 0, pointCount);
        Gather(bestIndices, results, gatherOffsets, sizeof(float), pointCount);
        auto bestIndex = bestIndices.template ReinterpretCast<int32_t>();
        auto chunkBases = CandidateValues().template ReinterpretCast<int32_t>();
        Muls(chunkOffsets, bestIndex, int32_t(kWorkspaceAlignment), pointCount);
        Muls(chunkBases, pointOffsets, int32_t(kChunkResultStride / 2), pointCount);
        Adds(chunkBases, chunkBases, int32_t(kChunkResultStride * sizeof(float)), pointCount);
        Add(chunkOffsets, chunkOffsets, chunkBases, pointCount);
        Gather(CandidateValues(), results, gatherOffsets, sizeof(float), pointCount);
        auto localIndex = CandidateValues().template ReinterpretCast<int32_t>();
        Muls(bestIndex, bestIndex, int32_t(kSegmentTile), pointCount);
        Add(bestIndex, bestIndex, localIndex, pointCount);
    }

    __aicore__ inline void ComputeDistances(uint32_t segmentOffset, uint32_t pointOffset,
        uint64_t mask, uint8_t repeatCount, const BinaryRepeatParams& geometryAndPoint,
        const BinaryRepeatParams& matrixAndGeometry)
    {
        auto geometry = Geometry();
        auto broadcast = PointBroadcast();
        auto work = WorkPlanes();
        auto plane0 = work[kMatrixPlane0];
        auto plane1 = work[kMatrixPlane1];
        auto plane2 = work[kMatrixPlane2];
        auto plane3 = work[kMatrixPlane3];
        auto plane4 = work[kMatrixPlane4];
        auto plane5 = work[kMatrixPlane5];
        const BinaryRepeatParams matrixAndMatrix{1, 1, 1, 8, 8, 8};
        const UnaryRepeatParams matrixOnly{1, 1, 8, 8};

        Sub<float, false>(plane0, geometry[kStartXPlane][segmentOffset], broadcast[pointOffset],
            mask, repeatCount, geometryAndPoint);
        Sub<float, false>(plane1, geometry[kStartYPlane][segmentOffset],
            broadcast[kPointBroadcastElements + pointOffset], mask, repeatCount,
            geometryAndPoint);
        Sub<float, false>(plane2, geometry[kStartZPlane][segmentOffset],
            broadcast[2 * kPointBroadcastElements + pointOffset], mask, repeatCount,
            geometryAndPoint);

        Mul<float, false>(plane4, plane0, geometry[kDirectionXPlane][segmentOffset],
            mask, repeatCount, matrixAndGeometry);
        Mul<float, false>(plane5, plane1, geometry[kDirectionYPlane][segmentOffset],
            mask, repeatCount, matrixAndGeometry);
        Add<float, false>(plane3, plane4, plane5, mask, repeatCount, matrixAndMatrix);
        Mul<float, false>(plane4, plane2, geometry[kDirectionZPlane][segmentOffset],
            mask, repeatCount, matrixAndGeometry);
        Add<float, false>(plane5, plane3, plane4, mask, repeatCount, matrixAndMatrix);
        Mul<float, false>(plane4, plane5, geometry[kScalePlane][segmentOffset],
            mask, repeatCount, matrixAndGeometry);
        Maxs<float, false>(plane3, plane4, 0.0F, mask, repeatCount, matrixOnly);
        Mins<float, false>(plane4, plane3, 1.0F, mask, repeatCount, matrixOnly);

        Mul<float, false>(plane5, plane4, geometry[kDirectionXPlane][segmentOffset],
            mask, repeatCount, matrixAndGeometry);
        Add<float, false>(plane3, plane0, plane5, mask, repeatCount, matrixAndMatrix);
        Mul<float, false>(plane0, plane4, geometry[kDirectionYPlane][segmentOffset],
            mask, repeatCount, matrixAndGeometry);
        Add<float, false>(plane5, plane1, plane0, mask, repeatCount, matrixAndMatrix);
        Mul<float, false>(plane1, plane4, geometry[kDirectionZPlane][segmentOffset],
            mask, repeatCount, matrixAndGeometry);
        Add<float, false>(plane0, plane2, plane1, mask, repeatCount, matrixAndMatrix);

        // Two fused multiply-adds replace three multiplies and two adds; the sum lives in
        // plane2 because its UB bank class differs from both squared residuals.
        Mul<float, false>(plane2, plane0, plane0, mask, repeatCount, matrixAndMatrix);
        MulAddDst<float, float, false>(plane2, plane3, plane3, mask, repeatCount,
            matrixAndMatrix);
        MulAddDst<float, float, false>(plane2, plane5, plane5, mask, repeatCount,
            matrixAndMatrix);
    }

    __aicore__ inline void MergeChunkResults(uint32_t chunkCount, uint32_t pointCount,
        LocalTensor<float> bestValues, LocalTensor<float> bestIndices)
    {
        auto chunkResults = ChunkResults();
        auto offsets = PointResultOffsets().template ReinterpretCast<uint32_t>();
        auto candidateValues = CandidateValues();
        auto candidateIndices = CandidateIndices();
        auto candidateIndexBits = candidateIndices.template ReinterpretCast<int32_t>();
        const BinaryRepeatParams rowWise{1, 1, 1, 8, 8, 8};

        Gather(bestValues, chunkResults, offsets, 0, pointCount);
        Gather(bestIndices, chunkResults, offsets, sizeof(float), pointCount);
        for (uint32_t chunk = 1; chunk < chunkCount; ++chunk) {
            const uint32_t chunkResultOffset =
                chunk * kChunkResultStride * sizeof(float);
            Gather(candidateValues, chunkResults, offsets,
                chunkResultOffset, pointCount);
            Gather(candidateIndices, chunkResults, offsets,
                chunkResultOffset + sizeof(float), pointCount);
            Adds(candidateIndexBits, candidateIndexBits,
                static_cast<int32_t>(chunk * kSegmentTile), pointCount);
            Compare(candidateValues, bestValues, CMPMODE::LT,
                uint64_t{pointCount}, rowWise);
            PipeBarrier<PIPE_V>();
            SetVectorMask<float, MaskMode::NORMAL>(pointCount);
            Select<float, SELMODE::VSEL_CMPMASK_SPR>(
                bestValues, candidateValues, bestValues, 1, rowWise);
            Select<float, SELMODE::VSEL_CMPMASK_SPR>(
                bestIndices, candidateIndices, bestIndices, 1, rowWise);
        }
        ResetMask();
    }

    __aicore__ inline void MergeGroupedChunkResults(uint32_t chunkCount,
        uint32_t pointCount, LocalTensor<float> bestValues,
        LocalTensor<float> bestIndices)
    {
        auto chunkResults = ChunkResults();
        auto pointOffsets = PointResultOffsets();
        auto pointGather = pointOffsets.template ReinterpretCast<uint32_t>();
        const uint32_t rowElements =
            CeilDiv(chunkCount, kFloatElementsPerBlock) * kFloatElementsPerBlock;
        auto pointRows = WorkPlanes()[kMatrixPlane0];
        auto winners = WorkPlanes()[kMatrixPlane1];
        Gather(pointRows, chunkResults,
            ChunkGatherOffsets().template ReinterpretCast<uint32_t>(),
            0, pointCount * rowElements);
        WholeReduceMin<float>(winners, pointRows, chunkCount, pointCount,
            1, 1, rowElements / kFloatElementsPerBlock,
            ReduceOrder::ORDER_VALUE_INDEX);
        Gather(bestValues, winners, pointGather, 0, pointCount);
        Gather(bestIndices, winners, pointGather, sizeof(float), pointCount);

        auto winningChunks = bestIndices.template ReinterpretCast<int32_t>();
        auto sourceOffsets = CandidateIndices().template ReinterpretCast<int32_t>();
        auto chunkBases = CandidateValues().template ReinterpretCast<int32_t>();
        Muls(sourceOffsets, winningChunks, int32_t(kMatrixChunkResultStride * sizeof(float)),
            pointCount);
        Add(sourceOffsets, sourceOffsets, pointOffsets, pointCount);
        Muls(chunkBases, winningChunks, int32_t(kSegmentTile), pointCount);
        Gather(bestIndices, chunkResults, sourceOffsets.template ReinterpretCast<uint32_t>(),
            sizeof(float), pointCount);
        Add(winningChunks, winningChunks, chunkBases, pointCount);
    }

    __aicore__ inline void WaitForOutputBuffer(uint32_t buffer)
    {
        WaitFlag<HardEvent::MTE3_V>(
            kOutputPipelineEventBase + static_cast<int32_t>(buffer));
    }

    __aicore__ inline void PrepareOutput(uint32_t count,
        LocalTensor<float> minimum, LocalTensor<T> output)
    {
        Sqrt(minimum, minimum, count);
        if constexpr (sizeof(T) != sizeof(float)) {
            Cast(output, minimum, RoundMode::CAST_RINT, count);
        }
    }

    __aicore__ inline void CopyOutput(uint32_t offset, uint32_t count,
        LocalTensor<float> minimum, LocalTensor<int32_t> index,
        LocalTensor<T> output)
    {
        if constexpr (sizeof(T) == sizeof(float)) {
            DataCopyPad(distances_[offset], minimum,
                DataCopyExtParams{1, count * uint32_t(sizeof(float)), 0, 0, 0});
        } else {
            DataCopyPad(distances_[offset], output,
                DataCopyExtParams{1, count * uint32_t(sizeof(T)), 0, 0, 0});
        }
        DataCopyPad(indices_[offset], index,
            DataCopyExtParams{1, count * uint32_t(sizeof(int32_t)), 0, 0, 0});
    }

    __aicore__ inline void WriteOutput(uint32_t offset, uint32_t count,
        LocalTensor<float> minimum, LocalTensor<int32_t> index,
        LocalTensor<T> output)
    {
        PrepareOutput(count, minimum, output);
        Synchronize<HardEvent::V_MTE3>();
        CopyOutput(offset, count, minimum, index, output);
        Synchronize<HardEvent::MTE3_V>();
    }

    __aicore__ inline void WritePipelineOutput(uint32_t offset, uint32_t count,
        LocalTensor<float> minimum, LocalTensor<int32_t> index,
        LocalTensor<T> output, uint32_t buffer)
    {
        const int32_t event = kOutputPipelineEventBase + static_cast<int32_t>(buffer);
        PrepareOutput(count, minimum, output);
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
        CopyOutput(offset, count, minimum, index, output);
        SetFlag<HardEvent::MTE3_V>(event);
    }

    GlobalTensor<T> points_;
    GlobalTensor<T> segments_;
    GlobalTensor<T> distances_;
    GlobalTensor<int32_t> indices_;
    TreeLineMinDistanceTilingData data_;
    bool usePointVectors_;
    bool useScalarGeometry_;
    bool useScalarPointBroadcast_;
    bool firstSegmentIssued_;
};
}

template <typename DT_TREE_POINTS, int TINY_MODE>
__aicore__ inline void RunTreeLineKernel(GM_ADDR tree_points, GM_ADDR line_segments,
    GM_ADDR min_distance, GM_ADDR nearest_segment_idx,
    const TreeLineMinDistanceTilingData& data)
{
    InitSocState();
    TreeLineKernel<DT_TREE_POINTS, TINY_MODE> kernel;
    kernel.Init(tree_points, line_segments, min_distance, nearest_segment_idx, data);
    kernel.Process();
}

template <typename DT_TREE_POINTS, int TINY_MODE>
__global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points, GM_ADDR line_segments,
    GM_ADDR min_distance, GM_ADDR nearest_segment_idx, GM_ADDR, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    GET_TILING_DATA_PTR_WITH_STRUCT(TreeLineMinDistanceTilingData, dataGlobal, tiling);
    if constexpr (TINY_MODE == 1) {
        // A tiny dispatch reads seven tiling scalars, which costs less than
        // staging the whole structure and draining a copy barrier first.
        TreeLineMinDistanceTilingData tilingScalars;
        tilingScalars.batches = dataGlobal->batches;
        tilingScalars.pointsPerBatch = dataGlobal->pointsPerBatch;
        tilingScalars.segmentsPerBatch = dataGlobal->segmentsPerBatch;
        tilingScalars.totalPoints = dataGlobal->totalPoints;
        tilingScalars.blockDim = dataGlobal->blockDim;
        RunTreeLineKernel<DT_TREE_POINTS, TINY_MODE>(tree_points, line_segments,
            min_distance, nearest_segment_idx, tilingScalars);
        return;
    } else {
        COPY_TILING_WITH_STRUCT(TreeLineMinDistanceTilingData, dataGlobal, dataLocal);
        RunTreeLineKernel<DT_TREE_POINTS, TINY_MODE>(tree_points, line_segments,
            min_distance, nearest_segment_idx, *dataLocal);
    }
}
