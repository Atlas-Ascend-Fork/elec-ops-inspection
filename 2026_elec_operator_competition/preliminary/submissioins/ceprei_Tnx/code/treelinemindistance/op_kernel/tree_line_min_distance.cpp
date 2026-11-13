/*
 * Copyright (c) 2026 ceprei_Tnx.
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
// Set to 0 to fall back to the plain mul-then-add form of the segment-axis
// point kernel. Both forms are mathematically identical; the fused one is
// cheaper on the UB operand path.
#ifndef TREE_LINE_USE_MULADD
#define TREE_LINE_USE_MULADD 1
#endif

#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"

#include <type_traits>

using namespace AscendC;

namespace {
// Keep one prepared segment tile resident while processing a large point
// group. This removes repeated GM copy, Gather and segment preprocessing
// without changing the floating-point expression for any point.
constexpr uint32_t POINTS_PER_GROUP = 256U;
constexpr uint32_t SINGLE_RESULT_WIDTH = 8U;
constexpr uint32_t POINT_VECTOR_TILE = TREE_LINE_SEGMENTS_PER_TILE;
// In the point-vector path, raw xyz points are dead immediately after column
// extraction, so the same segmentBuf_ allocation can hold the small segment
// set without consuming another UB buffer.
constexpr uint32_t POINT_VECTOR_SEGMENTS_PER_CHUNK = TREE_LINE_SEGMENTS_PER_TILE;
constexpr uint32_t POINT_VECTOR_SEGMENT_LIMIT = 64U;
constexpr uint32_t SCALAR_POINT_LIMIT = 8U;
constexpr uint32_t SCALAR_SEGMENT_LIMIT = 32U;
constexpr uint32_t PLANE_SIZE = TREE_LINE_SEGMENTS_PER_TILE;
// Keep the empirically faster 256-byte separation used by the 17/17 baseline.
constexpr uint32_t PLANE_STRIDE = PLANE_SIZE + 64U;
// Seven resident segment planes, three private planes per in-flight point,
// and two shared vector temporaries. The shared planes are safe because all
// uses are ordered on PIPE_V; only the squared-distance plane survives into
// the following reduction.
constexpr uint32_t PLANE_COUNT = 15U;
// Two floats (value/index) per 64-element repeat. A 2048-element tile has
// 32 repeats, so 64 floats cover the full tile and every tail configuration.
constexpr uint32_t REDUCE_TMP_FLOATS = 64U;
constexpr uint32_t REDUCE_OUT_FLOATS = 8U;
constexpr uint32_t REDUCE_CONTEXT_FLOATS = REDUCE_TMP_FLOATS + REDUCE_OUT_FLOATS;
constexpr uint32_t REDUCE_CONTEXT_BASE = PLANE_COUNT * PLANE_STRIDE;
constexpr uint32_t WORKSPACE_FLOATS = REDUCE_CONTEXT_BASE + 2U * REDUCE_CONTEXT_FLOATS;
constexpr uint32_t PARTIAL_SLOT_FLOATS = 8U;
constexpr float FLOAT_MAX_VALUE = 3.402823466e+38F;
} // namespace

template <typename T> class KernelTreeLineMinDistance {
  public:
    __aicore__ inline void Init(GM_ADDR treePoints, GM_ADDR lineSegments, GM_ADDR minDistance,
                                GM_ADDR nearestSegmentIdx, GM_ADDR workspace,
                                const TreeLineMinDistanceTilingData &tiling, TPipe *pipe) {
        pipe_ = pipe;
        pointCount_ = tiling.pointCount;
        batchCount_ = tiling.batchCount;
        pointsPerBatch_ = tiling.pointsPerBatch;
        segmentsPerBatch_ = tiling.segmentsPerBatch;
        pointsPerCore_ = tiling.pointsPerCore;
        extraCoreCount_ = tiling.extraCoreCount;
        segmentTileSize_ = tiling.segmentTileSize;
        coresPerPoint_ = tiling.coresPerPoint;

        treePointsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(treePoints),
                                      static_cast<uint64_t>(pointCount_) * 3U);
        lineSegmentsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(lineSegments),
                                        static_cast<uint64_t>(batchCount_) * segmentsPerBatch_ * 6U);
        minDistanceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(minDistance), pointCount_);
        nearestSegmentIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(nearestSegmentIdx), pointCount_);
        if (coresPerPoint_ > 1U) {
            workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace),
                                         static_cast<uint64_t>(pointCount_) * coresPerPoint_ * PARTIAL_SLOT_FLOATS);
        }

        // The same physical buffers serve three mutually-exclusive paths.
        // 256 xyz tuples also exceed the storage needed by the mutually
        // exclusive small-M path's 64 six-value segments.
        pipe_->InitBuffer(pointBuf_, POINTS_PER_GROUP * 3U * sizeof(T));
        pipe_->InitBuffer(segmentBuf_, TREE_LINE_SEGMENTS_PER_TILE * 6U * sizeof(T));
        // All columns share row-start offsets; Gather's byte base address
        // selects the requested column. This saves five full index planes.
        pipe_->InitBuffer(gatherIndexBuf_, TREE_LINE_SEGMENTS_PER_TILE * sizeof(uint32_t));
        // Column Gather needs this staging tensor only for FP16. FP32 gathers
        // directly into its destination planes; the other uses (mask/output)
        // are smaller than this allocation.
        pipe_->InitBuffer(gatherValueBuf_, TREE_LINE_SEGMENTS_PER_TILE * sizeof(half));
        pipe_->InitBuffer(workspaceBuf_, WORKSPACE_FLOATS * sizeof(float));
        pipe_->InitBuffer(resultValueBuf_, POINTS_PER_GROUP * sizeof(float));
        pipe_->InitBuffer(resultIndexBuf_, POINTS_PER_GROUP * sizeof(int32_t));
    }

    __aicore__ inline void Process() {
        if (coresPerPoint_ > 1U) {
            ProcessGroupReduce();
            return;
        }

        const uint32_t coreIndex = GetBlockIdx();
        const uint32_t leadingExtra = coreIndex < extraCoreCount_ ? coreIndex : extraCoreCount_;
        const uint32_t corePointCount = pointsPerCore_ + (coreIndex < extraCoreCount_ ? 1U : 0U);
        const uint32_t pointBegin = coreIndex * pointsPerCore_ + leadingExtra;
        if (pointBegin >= pointCount_) {
            return;
        }
        const uint32_t pointEnd = pointBegin + corePointCount;

        const bool useScalar = pointsPerBatch_ <= SCALAR_POINT_LIMIT && segmentsPerBatch_ <= SCALAR_SEGMENT_LIMIT;
        // Two layouts are available and they trade different costs.
        //
        // Segment axis (segments in the vector lanes): one full-width pass over
        // the segment lanes per point, but every point additionally pays a
        // whole-axis WholeReduceMin plus a V/S handshake, and only three scalar
        // reads (the point coordinates).
        //
        // Point axis (points in the vector lanes): every multiplier becomes a
        // scalar, so each term folds into an FMA, and the running minimum and
        // its index are maintained incrementally with Compare/Select over the
        // point lanes - there is no whole-axis reduction and no per-point
        // handshake at all. The price is one scalar read of six segment
        // components per segment, amortised over corePointCount points.
        //
        // The point axis therefore wins once that amortisation covers the extra
        // reads, i.e. roughly corePointCount > 2 * segmentsPerBatch. Small M
        // keeps the existing proven threshold; the second clause only routes
        // shapes whose points-per-core dwarf the segment count.
        const bool vectorizePoints =
            !useScalar && pointsPerBatch_ >= 64U && corePointCount >= 16U &&
            (segmentsPerBatch_ <= POINT_VECTOR_SEGMENT_LIMIT || corePointCount / 2U > segmentsPerBatch_);
        // The index count follows the number of lanes the consuming Gather
        // reads, which differs per layout: points for the point axis, segments
        // for the segment axis.
        if (vectorizePoints) {
            const uint32_t pointBlock = corePointCount < POINT_VECTOR_TILE ? corePointCount : POINT_VECTOR_TILE;
            BuildGatherIndices(3U, GatherIndexCount(pointBlock));
        } else if (!useScalar) {
            const uint32_t segmentBlock =
                segmentsPerBatch_ < TREE_LINE_SEGMENTS_PER_TILE ? segmentsPerBatch_ : TREE_LINE_SEGMENTS_PER_TILE;
            BuildGatherIndices(6U, GatherIndexCount(segmentBlock));
        }

        uint32_t cursor = pointBegin;
        while (cursor < pointEnd) {
            const uint32_t batchIndex = cursor / pointsPerBatch_;
            const uint32_t nextBatch = (batchIndex + 1U) * pointsPerBatch_;
            const uint32_t batchEnd = nextBatch < pointEnd ? nextBatch : pointEnd;
            const uint64_t batchSegmentBase = static_cast<uint64_t>(batchIndex) * segmentsPerBatch_;
            if (useScalar) {
                for (uint32_t pointIndex = cursor; pointIndex < batchEnd; ++pointIndex) {
                    ProcessScalarPoint(pointIndex, batchSegmentBase);
                }
            } else if (vectorizePoints) {
                ProcessPointVectorRange(cursor, batchEnd, batchSegmentBase);
            } else {
                ProcessSegmentVectorRange(cursor, batchEnd, batchSegmentBase);
            }
            cursor = batchEnd;
        }
    }

  private:
    __aicore__ inline LocalTensor<float> WorkAt(uint32_t floatOffset) {
        return workspaceBuf_.Get<float>()[floatOffset];
    }

    __aicore__ inline LocalTensor<float> Plane(uint32_t plane) {
        return WorkAt(plane * PLANE_STRIDE);
    }

    __aicore__ inline LocalTensor<float> ScratchPlane(uint32_t context, uint32_t plane) {
        return Plane(7U + context * 3U + plane);
    }

    __aicore__ inline LocalTensor<float> SharedWork() {
        return Plane(13U);
    }

    __aicore__ inline LocalTensor<float> SharedTemporary() {
        return Plane(14U);
    }

    __aicore__ inline LocalTensor<float> ReduceTmp(uint32_t context) {
        return WorkAt(REDUCE_CONTEXT_BASE + context * REDUCE_CONTEXT_FLOATS);
    }

    __aicore__ inline LocalTensor<float> ReduceOut(uint32_t context) {
        return WorkAt(REDUCE_CONTEXT_BASE + context * REDUCE_CONTEXT_FLOATS + REDUCE_TMP_FLOATS);
    }

    // Gather only ever reads indices[0 .. count-1], so the plane only has to
    // cover the largest count ExtractColumns will be handed, rounded up to one
    // vector repeat. Arange builds the sequence element by element, so its cost
    // scales with the count - and the full tile width used to be materialised
    // even for a batch holding a handful of segments. That was a prologue whose
    // price did not depend on the shape at all, paid once by every launch.
    __aicore__ inline uint32_t GatherIndexCount(uint32_t elementCount) {
        const uint32_t rounded = (elementCount + 63U) / 64U * 64U;
        return rounded < TREE_LINE_SEGMENTS_PER_TILE ? rounded : TREE_LINE_SEGMENTS_PER_TILE;
    }

    __aicore__ inline void BuildGatherIndices(uint32_t rowWidth, uint32_t indexCount) {
        LocalTensor<uint32_t> indices = gatherIndexBuf_.Get<uint32_t>();
        LocalTensor<int32_t> signedIndices = indices.ReinterpretCast<int32_t>();
        const uint32_t elementBytes = static_cast<uint32_t>(sizeof(T));
        const uint32_t rowBytes = rowWidth * elementBytes;
        Arange<int32_t>(signedIndices, 0, static_cast<int32_t>(rowBytes), static_cast<int32_t>(indexCount));
        PipeBarrier<PIPE_V>();
    }

    // Issue a GM transfer without waiting for it. Two independent transfers can
    // then be in flight at the same time, so their GM latencies overlap and the
    // caller pays one barrier instead of one per copy.
    __aicore__ inline void IssuePoints(const LocalTensor<T> &destination, uint32_t pointOffset, uint32_t count) {
        DataCopyExtParams copy{1U, static_cast<uint32_t>(count * 3U * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> pad{true, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(destination, treePointsGm_[static_cast<uint64_t>(pointOffset) * 3U], copy, pad);
    }

    __aicore__ inline void IssueSegments(const LocalTensor<T> &destination, uint64_t segmentOffset, uint32_t count) {
        DataCopyExtParams copy{1U, static_cast<uint32_t>(count * 6U * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> pad{true, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(destination, lineSegmentsGm_[segmentOffset * 6U], copy, pad);
    }

    __aicore__ inline void CopyPoints(const LocalTensor<T> &destination, uint32_t pointOffset, uint32_t count) {
        IssuePoints(destination, pointOffset, count);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopySegments(const LocalTensor<T> &destination, uint64_t segmentOffset, uint32_t count) {
        IssueSegments(destination, segmentOffset, count);
        PipeBarrier<PIPE_ALL>();
    }

    // Extract the requested interleaved columns. Every column is acquired
    // back-to-back instead of one at a time. The FP32 form has no ordering
    // constraint at all; the FP16 form gives each column its own private
    // staging slice, which removes the per-column Gather -> barrier -> Cast
    // serialisation that the single shared scratch tensor used to force. That
    // chain is pure exposed latency and this routine runs once per point group,
    // so it is paid in full by the small shapes where the group holds a single
    // point.
    //
    // Two layout facts make the wider form legal. Widening a half into a float
    // cannot be done in place, because the four-byte store of element i lands
    // on the not-yet-read two-byte source of element 2*i, so staging and
    // destination are always disjoint regions. And the staging planes are
    // scratch planes that PrepareSegmentTile only fills after this routine has
    // returned, so no live tile data is disturbed.
    __aicore__ inline void ExtractColumns(const LocalTensor<T> &source, uint32_t columnCount, uint32_t count) {
        constexpr uint32_t PLANE_STAGE_BASE = 8U;
        LocalTensor<uint32_t> indices = gatherIndexBuf_.Get<uint32_t>();
        for (uint32_t column = 0; column < columnCount; ++column) {
            const uint32_t columnBase = column * static_cast<uint32_t>(sizeof(T));
            if constexpr (std::is_same_v<T, float>) {
                Gather(Plane(column), source, indices, columnBase, count);
            } else {
                Gather(Plane(PLANE_STAGE_BASE + column).template ReinterpretCast<half>(), source, indices, columnBase,
                       count);
            }
        }
        if constexpr (!std::is_same_v<T, float>) {
            for (uint32_t column = 0; column < columnCount; ++column) {
                Cast(Plane(column), Plane(PLANE_STAGE_BASE + column).template ReinterpretCast<half>(),
                     RoundMode::CAST_NONE, count);
            }
        }
    }

    // -------------------- tiny scalar path --------------------
    __aicore__ inline void ProcessScalarPoint(uint32_t pointIndex, uint64_t batchSegmentBase) {
        LocalTensor<T> pointLocal = pointBuf_.Get<T>();
        LocalTensor<T> segmentLocal = segmentBuf_.Get<T>();
        // The point tuple and the whole segment set are independent transfers:
        // issue both, then wait once so the two GM latencies overlap instead of
        // being serialised by a barrier after each copy.
        IssuePoints(pointLocal, pointIndex, 1U);
        IssueSegments(segmentLocal, batchSegmentBase, segmentsPerBatch_);
        PipeBarrier<PIPE_ALL>();

        const float px = static_cast<float>(pointLocal.GetValue(0));
        const float py = static_cast<float>(pointLocal.GetValue(1));
        const float pz = static_cast<float>(pointLocal.GetValue(2));
        float bestSquaredDistance = FLOAT_MAX_VALUE;
        int32_t bestSegmentIndex = 0;
        for (uint32_t segmentIndex = 0; segmentIndex < segmentsPerBatch_; ++segmentIndex) {
            const uint32_t base = segmentIndex * 6U;
            const float ax = static_cast<float>(segmentLocal.GetValue(base));
            const float ay = static_cast<float>(segmentLocal.GetValue(base + 1U));
            const float az = static_cast<float>(segmentLocal.GetValue(base + 2U));
            const float bx = static_cast<float>(segmentLocal.GetValue(base + 3U));
            const float by = static_cast<float>(segmentLocal.GetValue(base + 4U));
            const float bz = static_cast<float>(segmentLocal.GetValue(base + 5U));
            const float vx = bx - ax;
            const float vy = by - ay;
            const float vz = bz - az;
            const float wx = px - ax;
            const float wy = py - ay;
            const float wz = pz - az;
            const float lengthSquared = vx * vx + vy * vy + vz * vz;
            float projection = 0.0F;
            if (lengthSquared > 0.0F) {
                projection = (wx * vx + wy * vy + wz * vz) / lengthSquared;
                projection = projection < 0.0F ? 0.0F : projection;
                projection = projection > 1.0F ? 1.0F : projection;
            }
            const float dx = wx - projection * vx;
            const float dy = wy - projection * vy;
            const float dz = wz - projection * vz;
            const float squaredDistance = dx * dx + dy * dy + dz * dz;
            if (squaredDistance < bestSquaredDistance) {
                bestSquaredDistance = squaredDistance;
                bestSegmentIndex = static_cast<int32_t>(segmentIndex);
            }
        }
        WriteSingle(pointIndex, bestSquaredDistance, bestSegmentIndex);
    }

    __aicore__ inline void WriteSingle(uint32_t pointIndex, float squaredDistance, int32_t segmentIndex) {
        LocalTensor<float> resultValue = resultValueBuf_.Get<float>();
        LocalTensor<int32_t> resultIndex = resultIndexBuf_.Get<int32_t>();
        LocalTensor<T> converted = gatherValueBuf_.Get<T>();
        Duplicate(resultValue, squaredDistance, SINGLE_RESULT_WIDTH);
        resultIndex.SetValue(0, segmentIndex);
        Sqrt(resultValue, resultValue, SINGLE_RESULT_WIDTH);
        if constexpr (std::is_same_v<T, half>) {
            Cast(converted, resultValue, RoundMode::CAST_ROUND, SINGLE_RESULT_WIDTH);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams valueCopy{1U, static_cast<uint32_t>(sizeof(T)), 0U, 0U, 0U};
        if constexpr (std::is_same_v<T, half>) {
            DataCopyPad(minDistanceGm_[pointIndex], converted, valueCopy);
        } else {
            DataCopyPad(minDistanceGm_[pointIndex], resultValue, valueCopy);
        }
        DataCopyExtParams indexCopy{1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(nearestSegmentIdxGm_[pointIndex], resultIndex, indexCopy);
        PipeBarrier<PIPE_ALL>();
    }

    // -------------------- small M / large N: vectorize points --------------------
    __aicore__ inline void ComputePointVectorDistance(float ax, float ay, float az, float vx, float vy, float vz,
                                                      float inverseLengthSquared, uint32_t count) {
        LocalTensor<float> wx = Plane(6U);
        LocalTensor<float> wy = Plane(7U);
        LocalTensor<float> wz = Plane(8U);
        LocalTensor<float> temporary = Plane(9U);
        LocalTensor<float> projection = Plane(5U);
        Adds(wx, Plane(0U), -ax, count);
        Adds(wy, Plane(1U), -ay, count);
        Adds(wz, Plane(2U), -az, count);
        Muls(projection, wx, vx, count);
        Muls(temporary, wy, vy, count);
        Add(projection, projection, temporary, count);
        Muls(temporary, wz, vz, count);
        Add(projection, projection, temporary, count);
        Muls(projection, projection, inverseLengthSquared, count);
        Maxs(projection, projection, 0.0F, count);
        Mins(projection, projection, 1.0F, count);
        Muls(temporary, projection, vx, count);
        Sub(wx, wx, temporary, count);
        Mul(wx, wx, wx, count);
        Muls(temporary, projection, vy, count);
        Sub(wy, wy, temporary, count);
        Mul(wy, wy, wy, count);
        Add(wx, wx, wy, count);
        Muls(temporary, projection, vz, count);
        Sub(wz, wz, temporary, count);
        Mul(wz, wz, wz, count);
        Add(wx, wx, wz, count);
    }

    __aicore__ inline void ProcessPointVectorRange(uint32_t pointBegin, uint32_t pointEnd, uint64_t batchSegmentBase) {
        LocalTensor<T> segmentLocal = segmentBuf_.Get<T>();
        LocalTensor<T> rawPoints = segmentBuf_.Get<T>();

        for (uint32_t pointOffset = pointBegin; pointOffset < pointEnd; pointOffset += POINT_VECTOR_TILE) {
            const uint32_t remaining = pointEnd - pointOffset;
            const uint32_t pointCount = remaining < POINT_VECTOR_TILE ? remaining : POINT_VECTOR_TILE;
            const uint32_t alignedCount = (pointCount + 63U) / 64U * 64U;
            CopyPoints(rawPoints, pointOffset, pointCount);
            ExtractColumns(rawPoints, 3U, pointCount);

            LocalTensor<float> bestDistance = Plane(3U);
            LocalTensor<float> bestIndex = Plane(4U);
            LocalTensor<uint8_t> compareMask = gatherValueBuf_.Get<uint8_t>();
            Duplicate(bestDistance, FLOAT_MAX_VALUE, alignedCount);
            Duplicate(bestIndex, 0.0F, alignedCount);
            PipeBarrier<PIPE_V>();

            float segmentIndexFloat = 0.0F;
            for (uint32_t segmentOffset = 0; segmentOffset < segmentsPerBatch_;
                 segmentOffset += POINT_VECTOR_SEGMENTS_PER_CHUNK) {
                const uint32_t remainingSegments = segmentsPerBatch_ - segmentOffset;
                const uint32_t chunkCount = remainingSegments < POINT_VECTOR_SEGMENTS_PER_CHUNK
                                                ? remainingSegments
                                                : POINT_VECTOR_SEGMENTS_PER_CHUNK;
                CopySegments(segmentLocal, batchSegmentBase + segmentOffset, chunkCount);

                for (uint32_t localSegment = 0; localSegment < chunkCount; ++localSegment) {
                    const uint32_t base = localSegment * 6U;
                    const float ax = static_cast<float>(segmentLocal.GetValue(base));
                    const float ay = static_cast<float>(segmentLocal.GetValue(base + 1U));
                    const float az = static_cast<float>(segmentLocal.GetValue(base + 2U));
                    const float vx = static_cast<float>(segmentLocal.GetValue(base + 3U)) - ax;
                    const float vy = static_cast<float>(segmentLocal.GetValue(base + 4U)) - ay;
                    const float vz = static_cast<float>(segmentLocal.GetValue(base + 5U)) - az;
                    const float lengthSquared = vx * vx + vy * vy + vz * vz;
                    const float inverseLengthSquared = lengthSquared > 0.0F ? 1.0F / lengthSquared : 0.0F;
                    ComputePointVectorDistance(ax, ay, az, vx, vy, vz, inverseLengthSquared, alignedCount);
                    Compare(compareMask, Plane(6U), bestDistance, CMPMODE::GE, alignedCount);
                    Select(bestDistance, compareMask, bestDistance, Plane(6U), SELMODE::VSEL_TENSOR_TENSOR_MODE,
                           alignedCount);
                    Select(bestIndex, compareMask, bestIndex, segmentIndexFloat, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                           alignedCount);
                    segmentIndexFloat += 1.0F;
                }
            }
            PipeBarrier<PIPE_V>();
            WritePointVectorResults(pointOffset, pointCount, alignedCount, bestDistance, bestIndex);
        }
    }

    __aicore__ inline void WritePointVectorResults(uint32_t pointOffset, uint32_t pointCount, uint32_t alignedCount,
                                                   const LocalTensor<float> &bestDistance,
                                                   const LocalTensor<float> &bestIndex) {
        LocalTensor<int32_t> indexOutput = Plane(0U).template ReinterpretCast<int32_t>();
        Sqrt(bestDistance, bestDistance, alignedCount);
        Cast(indexOutput, bestIndex, RoundMode::CAST_ROUND, alignedCount);
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<half> distanceOutput = Plane(1U).template ReinterpretCast<half>();
            Cast(distanceOutput, bestDistance, RoundMode::CAST_ROUND, alignedCount);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams valueCopy{1U, static_cast<uint32_t>(pointCount * sizeof(half)), 0U, 0U, 0U};
            DataCopyPad(minDistanceGm_[pointOffset], distanceOutput, valueCopy);
        } else {
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams valueCopy{1U, static_cast<uint32_t>(pointCount * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(minDistanceGm_[pointOffset], bestDistance, valueCopy);
        }
        DataCopyExtParams indexCopy{1U, static_cast<uint32_t>(pointCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(nearestSegmentIdxGm_[pointOffset], indexOutput, indexCopy);
        PipeBarrier<PIPE_ALL>();
    }

    // -------------------- large M: segment tile resident across a point group --------------------
    __aicore__ inline void PrepareSegmentTile(const LocalTensor<T> &rawSegments, uint32_t count) {
        ExtractColumns(rawSegments, 6U, count);
        // Use V=A-B together with W=A-P. Both are the negation of the
        // conventional vectors, preserving projection and residual exactly.
        Sub(Plane(3U), Plane(0U), Plane(3U), count);
        Sub(Plane(4U), Plane(1U), Plane(4U), count);
        Sub(Plane(5U), Plane(2U), Plane(5U), count);
        // |V|^2 is accumulated into Plane(7U) so that the reciprocal below can
        // be written with destination == first source, the same aliasing form
        // already used by the Sub calls above.
        Mul(Plane(7U), Plane(3U), Plane(3U), count);
        Mul(Plane(6U), Plane(4U), Plane(4U), count);
        Add(Plane(7U), Plane(7U), Plane(6U), count);
        Mul(Plane(6U), Plane(5U), Plane(5U), count);
        Add(Plane(7U), Plane(7U), Plane(6U), count);
        Maxs(Plane(7U), Plane(7U), 1.0e-30F, count);
        // Hoist the per-point division out of the point loop: one reciprocal per
        // segment tile, then one vector multiply per point instead of a
        // full-width vector divide. The 1e-30 floor keeps the reciprocal finite
        // (<= 1e30), and a zero-length segment still yields projection 0
        // because W . V is exactly zero in that case.
        Duplicate(Plane(6U), 1.0F, count);
        Div(Plane(6U), Plane(6U), Plane(7U), count);
    }

    __aicore__ inline void ComputePreparedSegmentDistance(float px, float py, float pz, uint32_t count,
                                                          uint32_t context) {
        LocalTensor<float> wx = ScratchPlane(context, 0U);
        LocalTensor<float> wy = ScratchPlane(context, 1U);
        LocalTensor<float> projection = ScratchPlane(context, 2U);
        // wz and temporary do not need to survive after the point's vector
        // instruction sequence. Sharing them between the two scalar-overlap
        // contexts saves two full planes without adding vector work.
        LocalTensor<float> wz = SharedWork();
        LocalTensor<float> temporary = SharedTemporary();
        Adds(wx, Plane(0U), -px, count);
        Adds(wy, Plane(1U), -py, count);
        Adds(wz, Plane(2U), -pz, count);
#if TREE_LINE_USE_MULADD
        // Each accumulator step folds into the multiply that feeds it. A vector
        // instruction here costs about 1.3 core cycles per vector operand it
        // touches, so dst += src0 * src1 (four accesses) beats a separate mul
        // plus add (six) even though it is the same arithmetic. This takes the
        // point kernel from 22 instructions and 61 operand accesses per pair to
        // 18 and 53, a 13 percent cut, which is what the segment axis needs.
        Mul(projection, wx, Plane(3U), count);
        MulAddDst(projection, wy, Plane(4U), count);
        MulAddDst(projection, wz, Plane(5U), count);
        // Plane(6U) carries 1 / |V|^2, prepared once for the whole tile.
        Mul(projection, projection, Plane(6U), count);
        Maxs(projection, projection, 0.0F, count);
        Mins(projection, projection, 1.0F, count);
        Mul(temporary, projection, Plane(3U), count);
        Sub(wx, wx, temporary, count);
        Mul(wx, wx, wx, count);
        Mul(temporary, projection, Plane(4U), count);
        Sub(wy, wy, temporary, count);
        MulAddDst(wx, wy, wy, count);
        Mul(temporary, projection, Plane(5U), count);
        Sub(wz, wz, temporary, count);
        MulAddDst(wx, wz, wz, count);
#else
        Mul(projection, wx, Plane(3U), count);
        Mul(temporary, wy, Plane(4U), count);
        Add(projection, projection, temporary, count);
        Mul(temporary, wz, Plane(5U), count);
        Add(projection, projection, temporary, count);
        // Plane(6U) carries 1 / |V|^2, prepared once for the whole tile.
        Mul(projection, projection, Plane(6U), count);
        Maxs(projection, projection, 0.0F, count);
        Mins(projection, projection, 1.0F, count);
        Mul(temporary, projection, Plane(3U), count);
        Sub(wx, wx, temporary, count);
        Mul(wx, wx, wx, count);
        Mul(temporary, projection, Plane(4U), count);
        Sub(wy, wy, temporary, count);
        Mul(wy, wy, wy, count);
        Add(wx, wx, wy, count);
        Mul(temporary, projection, Plane(5U), count);
        Sub(wz, wz, temporary, count);
        Mul(wz, wz, wz, count);
        Add(wx, wx, wz, count);
#endif
    }

    __aicore__ inline void IssueReduce(uint32_t count, uint32_t fullRepeats, uint32_t tailCount,
                                       uint32_t reduceGroupCount, uint64_t finalValueMask[2], uint32_t context) {
        LocalTensor<float> reduceTmp = ReduceTmp(context);
        LocalTensor<float> reduceOut = ReduceOut(context);
        LocalTensor<float> squaredDistance = ScratchPlane(context, 0U);
        if (reduceGroupCount == 1U) {
            WholeReduceMin<float>(reduceOut, squaredDistance, static_cast<int32_t>(count), 1, 1, 1, 8);
        } else {
            if (fullRepeats != 0U) {
                WholeReduceMin<float>(reduceTmp, squaredDistance, 64, fullRepeats, 1, 1, 8);
            }
            if (tailCount != 0U) {
                WholeReduceMin<float>(reduceTmp[fullRepeats * 2U], squaredDistance[fullRepeats * 64U],
                                      static_cast<int32_t>(tailCount), 1, 1, 1, 8);
            }
            PipeBarrier<PIPE_V>();
            WholeReduceMin<float>(reduceOut, reduceTmp, finalValueMask, 1, 1, 1, 8);
        }
    }

    template <uint32_t CONTEXT> __aicore__ inline void SignalReduceReady() {
        if constexpr (CONTEXT == 0U) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);
        } else {
            SetFlag<HardEvent::V_S>(EVENT_ID1);
        }
    }

    template <uint32_t CONTEXT>
    __aicore__ inline void WaitAndUpdate(uint32_t reduceGroupCount, uint32_t segmentOffset, uint32_t resultSlot,
                                         const LocalTensor<float> &bestValues,
                                         const LocalTensor<int32_t> &bestIndices) {
        if constexpr (CONTEXT == 0U) {
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
        } else {
            WaitFlag<HardEvent::V_S>(EVENT_ID1);
        }
        LocalTensor<float> reduceTmp = ReduceTmp(CONTEXT);
        LocalTensor<float> reduceOut = ReduceOut(CONTEXT);
        const float tileMinimum = reduceOut.GetValue(0);
        float indexBits = reduceOut.GetValue(1);
        uint32_t tileIndex = *reinterpret_cast<uint32_t *>(&indexBits);
        if (reduceGroupCount != 1U) {
            const uint32_t pairPosition = tileIndex;
            float localIndexBits = reduceTmp.GetValue(pairPosition + 1U);
            const uint32_t localIndex = *reinterpret_cast<uint32_t *>(&localIndexBits);
            tileIndex = (pairPosition >> 1U) * 64U + localIndex;
        }
        if (tileMinimum < bestValues.GetValue(resultSlot)) {
            bestValues.SetValue(resultSlot, tileMinimum);
            bestIndices.SetValue(resultSlot, static_cast<int32_t>(segmentOffset + tileIndex));
        }
    }

    template <uint32_t CONTEXT>
    __aicore__ inline void IssuePointCompute(const LocalTensor<T> &pointLocal, uint32_t pointSlot, uint32_t tileCount) {
        const uint32_t base = pointSlot * 3U;
        const float px = static_cast<float>(pointLocal.GetValue(base));
        const float py = static_cast<float>(pointLocal.GetValue(base + 1U));
        const float pz = static_cast<float>(pointLocal.GetValue(base + 2U));
        ComputePreparedSegmentDistance(px, py, pz, tileCount, CONTEXT);
    }

    template <uint32_t CONTEXT>
    __aicore__ inline void FinishPointReduce(uint32_t tileCount, uint32_t fullRepeats, uint32_t tailCount,
                                             uint32_t reduceGroupCount, uint64_t finalValueMask[2]) {
        IssueReduce(tileCount, fullRepeats, tailCount, reduceGroupCount, finalValueMask, CONTEXT);
        SignalReduceReady<CONTEXT>();
    }

    __aicore__ inline void ProcessPreparedTilePoints(const LocalTensor<T> &pointLocal, uint32_t groupCount,
                                                     uint32_t tileCount, uint32_t fullRepeats, uint32_t tailCount,
                                                     uint32_t reduceGroupCount, uint64_t finalValueMask[2],
                                                     uint32_t segmentOffset, const LocalTensor<float> &bestValues,
                                                     const LocalTensor<int32_t> &bestIndices) {
        IssuePointCompute<0U>(pointLocal, 0U, tileCount);
        FinishPointReduce<0U>(tileCount, fullRepeats, tailCount, reduceGroupCount, finalValueMask);
        uint32_t pointSlot = 1U;
        for (; pointSlot + 1U < groupCount; pointSlot += 2U) {
            IssuePointCompute<1U>(pointLocal, pointSlot, tileCount);
            WaitAndUpdate<0U>(reduceGroupCount, segmentOffset, pointSlot - 1U, bestValues, bestIndices);
            FinishPointReduce<1U>(tileCount, fullRepeats, tailCount, reduceGroupCount, finalValueMask);
            IssuePointCompute<0U>(pointLocal, pointSlot + 1U, tileCount);
            WaitAndUpdate<1U>(reduceGroupCount, segmentOffset, pointSlot, bestValues, bestIndices);
            FinishPointReduce<0U>(tileCount, fullRepeats, tailCount, reduceGroupCount, finalValueMask);
        }
        if (pointSlot < groupCount) {
            IssuePointCompute<1U>(pointLocal, pointSlot, tileCount);
            WaitAndUpdate<0U>(reduceGroupCount, segmentOffset, pointSlot - 1U, bestValues, bestIndices);
            FinishPointReduce<1U>(tileCount, fullRepeats, tailCount, reduceGroupCount, finalValueMask);
            WaitAndUpdate<1U>(reduceGroupCount, segmentOffset, pointSlot, bestValues, bestIndices);
        } else {
            WaitAndUpdate<0U>(reduceGroupCount, segmentOffset, pointSlot - 1U, bestValues, bestIndices);
        }
    }

    __aicore__ inline void ProcessGroupTileRange(const LocalTensor<T> &pointLocal, const LocalTensor<float> &bestValues,
                                                 const LocalTensor<int32_t> &bestIndices, uint32_t shardBegin,
                                                 uint32_t shardEnd, uint64_t batchSegmentBase) {
        LocalTensor<T> rawSegments = segmentBuf_.Get<T>();
        for (uint32_t segmentOffset = shardBegin; segmentOffset < shardEnd; segmentOffset += segmentTileSize_) {
            const uint32_t remainingSegments = shardEnd - segmentOffset;
            const uint32_t tileCount = remainingSegments < segmentTileSize_ ? remainingSegments : segmentTileSize_;
            CopySegments(rawSegments, batchSegmentBase + segmentOffset, tileCount);
            PrepareSegmentTile(rawSegments, tileCount);
            const uint32_t fullRepeats = tileCount / 64U;
            const uint32_t tailCount = tileCount % 64U;
            const uint32_t reduceGroupCount = fullRepeats + (tailCount != 0U ? 1U : 0U);
            const uint32_t maskBitCount = reduceGroupCount * 2U;
            uint64_t finalValueMask[2] = {maskBitCount >= 64U ? 0x5555555555555555ULL
                                                              : ((static_cast<uint64_t>(1U) << maskBitCount) - 1U) &
                                                                    0x5555555555555555ULL,
                                          0U};
            ProcessPreparedTilePoints(pointLocal, 1U, tileCount, fullRepeats, tailCount, reduceGroupCount,
                                      finalValueMask, segmentOffset, bestValues, bestIndices);
        }
    }

    __aicore__ inline void MergeGroupPartials(uint32_t coreIndex, uint32_t pointIndex, uint32_t lane,
                                              const LocalTensor<float> &bestValues,
                                              const LocalTensor<int32_t> &bestIndices) {
        LocalTensor<float> partial = pointBuf_.Get<float>();
        LocalTensor<int32_t> partialBits = partial.ReinterpretCast<int32_t>();
        Duplicate(partial, 0.0F, PARTIAL_SLOT_FLOATS);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        partial.SetValue(0U, bestValues.GetValue(0U));
        partialBits.SetValue(1U, bestIndices.GetValue(0U));
        PipeBarrier<PIPE_ALL>();
        DataCopy(workspaceGm_[coreIndex * PARTIAL_SLOT_FLOATS], partial, PARTIAL_SLOT_FLOATS);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        if (lane != 0U) {
            return;
        }
        LocalTensor<float> allPartials = segmentBuf_.Get<float>();
        LocalTensor<int32_t> allPartialBits = allPartials.ReinterpretCast<int32_t>();
        DataCopy(allPartials, workspaceGm_[coreIndex * PARTIAL_SLOT_FLOATS], coresPerPoint_ * PARTIAL_SLOT_FLOATS);
        PipeBarrier<PIPE_ALL>();
        float bestSquaredDistance = FLOAT_MAX_VALUE;
        int32_t bestSegmentIndex = 0;
        for (uint32_t mergeLane = 0; mergeLane < coresPerPoint_; ++mergeLane) {
            const uint32_t slot = mergeLane * PARTIAL_SLOT_FLOATS;
            const float candidate = allPartials.GetValue(slot);
            if (candidate < bestSquaredDistance) {
                bestSquaredDistance = candidate;
                bestSegmentIndex = allPartialBits.GetValue(slot + 1U);
            }
        }
        WriteSingle(pointIndex, bestSquaredDistance, bestSegmentIndex);
    }

    // Long-R / small-A path. Several vector cores cooperate on one point,
    // write cache-line-isolated partial minima, then the first core of each
    // point group performs the tiny deterministic merge.
    __aicore__ inline void ProcessGroupReduce() {
        const uint32_t coreIndex = GetBlockIdx();
        const uint32_t pointIndex = coreIndex / coresPerPoint_;
        const uint32_t lane = coreIndex - pointIndex * coresPerPoint_;
        const uint32_t segmentsPerLane = segmentsPerBatch_ / coresPerPoint_;
        const uint32_t extraSegments = segmentsPerBatch_ % coresPerPoint_;
        const uint32_t leadingExtra = lane < extraSegments ? lane : extraSegments;
        const uint32_t shardBegin = lane * segmentsPerLane + leadingExtra;
        const uint32_t shardCount = segmentsPerLane + (lane < extraSegments ? 1U : 0U);
        const uint32_t shardEnd = shardBegin + shardCount;
        const uint32_t batchIndex = pointIndex / pointsPerBatch_;
        const uint64_t batchSegmentBase = static_cast<uint64_t>(batchIndex) * segmentsPerBatch_;

        BuildGatherIndices(6U, GatherIndexCount(shardCount < segmentTileSize_ ? shardCount : segmentTileSize_));
        LocalTensor<T> pointLocal = pointBuf_.Get<T>();
        LocalTensor<float> bestValues = resultValueBuf_.Get<float>();
        LocalTensor<int32_t> bestIndices = resultIndexBuf_.Get<int32_t>();
        CopyPoints(pointLocal, pointIndex, 1U);
        Duplicate(bestValues, FLOAT_MAX_VALUE, 64U);
        Duplicate(bestIndices, static_cast<int32_t>(0), 64U);
        PipeBarrier<PIPE_ALL>();

        ProcessGroupTileRange(pointLocal, bestValues, bestIndices, shardBegin, shardEnd, batchSegmentBase);
        MergeGroupPartials(coreIndex, pointIndex, lane, bestValues, bestIndices);
    }

    __aicore__ inline void ProcessSegmentVectorRange(uint32_t pointBegin, uint32_t pointEnd,
                                                     uint64_t batchSegmentBase) {
        LocalTensor<T> pointLocal = pointBuf_.Get<T>();
        LocalTensor<T> rawSegments = segmentBuf_.Get<T>();
        LocalTensor<float> bestValues = resultValueBuf_.Get<float>();
        LocalTensor<int32_t> bestIndices = resultIndexBuf_.Get<int32_t>();

        for (uint32_t groupBegin = pointBegin; groupBegin < pointEnd; groupBegin += POINTS_PER_GROUP) {
            const uint32_t remainingPoints = pointEnd - groupBegin;
            const uint32_t groupCount = remainingPoints < POINTS_PER_GROUP ? remainingPoints : POINTS_PER_GROUP;
            const uint32_t alignedGroupCount = (groupCount + 63U) / 64U * 64U;
            const uint32_t firstTileCount = segmentsPerBatch_ < segmentTileSize_ ? segmentsPerBatch_ : segmentTileSize_;
            // The point block and the first segment tile are independent
            // transfers, so both are issued before a single wait: their GM
            // latencies overlap instead of being serialised. The remaining
            // tiles still copy-then-wait because each one reuses the same
            // segment buffer as the tile being computed.
            IssuePoints(pointLocal, groupBegin, groupCount);
            IssueSegments(rawSegments, batchSegmentBase, firstTileCount);
            Duplicate(bestValues, FLOAT_MAX_VALUE, alignedGroupCount);
            Duplicate(bestIndices, static_cast<int32_t>(0), alignedGroupCount);
            PipeBarrier<PIPE_ALL>();

            for (uint32_t segmentOffset = 0; segmentOffset < segmentsPerBatch_; segmentOffset += segmentTileSize_) {
                const uint32_t remainingSegments = segmentsPerBatch_ - segmentOffset;
                const uint32_t tileCount = remainingSegments < segmentTileSize_ ? remainingSegments : segmentTileSize_;
                if (segmentOffset != 0U) {
                    CopySegments(rawSegments, batchSegmentBase + segmentOffset, tileCount);
                }
                PrepareSegmentTile(rawSegments, tileCount);
                const uint32_t fullRepeats = tileCount / 64U;
                const uint32_t tailCount = tileCount % 64U;
                const uint32_t reduceGroupCount = fullRepeats + (tailCount != 0U ? 1U : 0U);
                const uint32_t maskBitCount = reduceGroupCount * 2U;
                uint64_t finalValueMask[2] = {maskBitCount >= 64U ? 0x5555555555555555ULL
                                                                  : ((static_cast<uint64_t>(1U) << maskBitCount) - 1U) &
                                                                        0x5555555555555555ULL,
                                              0U};
                ProcessPreparedTilePoints(pointLocal, groupCount, tileCount, fullRepeats, tailCount, reduceGroupCount,
                                          finalValueMask, segmentOffset, bestValues, bestIndices);
            }
            WriteGroupResults(groupBegin, groupCount, alignedGroupCount, bestValues, bestIndices);
        }
    }

    __aicore__ inline void WriteGroupResults(uint32_t pointOffset, uint32_t pointCount, uint32_t alignedPointCount,
                                             const LocalTensor<float> &bestValues,
                                             const LocalTensor<int32_t> &bestIndices) {
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Sqrt(bestValues, bestValues, alignedPointCount);
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<half> converted = gatherValueBuf_.Get<half>();
            Cast(converted, bestValues, RoundMode::CAST_ROUND, alignedPointCount);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams valueCopy{1U, static_cast<uint32_t>(pointCount * sizeof(half)), 0U, 0U, 0U};
            DataCopyPad(minDistanceGm_[pointOffset], converted, valueCopy);
        } else {
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams valueCopy{1U, static_cast<uint32_t>(pointCount * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(minDistanceGm_[pointOffset], bestValues, valueCopy);
        }
        DataCopyExtParams indexCopy{1U, static_cast<uint32_t>(pointCount * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(nearestSegmentIdxGm_[pointOffset], bestIndices, indexCopy);
        PipeBarrier<PIPE_ALL>();
    }

    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> pointBuf_;
    TBuf<TPosition::VECCALC> segmentBuf_;
    TBuf<TPosition::VECCALC> gatherIndexBuf_;
    TBuf<TPosition::VECCALC> gatherValueBuf_;
    TBuf<TPosition::VECCALC> workspaceBuf_;
    TBuf<TPosition::VECCALC> resultValueBuf_;
    TBuf<TPosition::VECCALC> resultIndexBuf_;
    GlobalTensor<T> treePointsGm_;
    GlobalTensor<T> lineSegmentsGm_;
    GlobalTensor<T> minDistanceGm_;
    GlobalTensor<int32_t> nearestSegmentIdxGm_;
    GlobalTensor<float> workspaceGm_;
    uint32_t pointCount_ = 0U;
    uint32_t batchCount_ = 0U;
    uint32_t pointsPerBatch_ = 0U;
    uint32_t segmentsPerBatch_ = 0U;
    uint32_t pointsPerCore_ = 0U;
    uint32_t extraCoreCount_ = 0U;
    uint32_t segmentTileSize_ = 0U;
    uint32_t coresPerPoint_ = 1U;
};

template <typename DT_TREE_POINTS>
__global__ __aicore__ void tree_line_min_distance(GM_ADDR treePoints, GM_ADDR lineSegments, GM_ADDR minDistance,
                                                  GM_ADDR nearestSegmentIdx, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, tilingData, tiling);
    TPipe pipe;
    KernelTreeLineMinDistance<DT_TREE_POINTS> op;
    op.Init(treePoints, lineSegments, minDistance, nearestSegmentIdx, workspace, tilingData, &pipe);
    op.Process();
}
