/*
 * Copyright (c) 2026 zjyzjy2002.
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

#include "tree_line_spatial.h"
#include "tree_line_point_tiles.h"
#include "tree_line_tiled.h"
#include "tree_line_packed.h"
#include "tree_line_vector.h"

namespace OptimizedLayout {
// Only the existing single-point algorithms call this helper. One
// producer writes both current output elements and waits for GM completion.
template <typename T>
__aicore__ inline void StoreSinglePointResult(GM_ADDR distanceAddress, GM_ADDR indexAddress,
                      float squaredDistance, int32_t segmentIndex) {
  T distanceValue = T(sqrt(squaredDistance));
  if constexpr (IsSameType<T, float>::value) {
    uint32_t distanceBits = __builtin_bit_cast(uint32_t, distanceValue);
    WriteGmByPassDCache(reinterpret_cast<__gm__ uint32_t *>(distanceAddress), distanceBits);
  } else {
    uint16_t distanceBits = __builtin_bit_cast(uint16_t, distanceValue);
    WriteGmByPassDCache(reinterpret_cast<__gm__ uint16_t *>(distanceAddress), distanceBits);
  }
  WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t *>(indexAddress), segmentIndex);
  DataSyncBarrier<MemDsbT::DDR>();
}

template <uint32_t W>
struct SingleQueryPlanes {
  LocalTensor<float> raw, ax, ay, az, dx, dy, dz, len, x, y, z, dot, tmp, red;
  LocalTensor<uint32_t> off;
  __aicore__ explicit SingleQueryPlanes(LocalTensor<float> mem)
      : raw(mem), ax(mem[7*W]), ay(mem[8*W]), az(mem[9*W]), dx(mem[10*W]), dy(mem[11*W]),
        dz(mem[12*W]), len(mem[13*W]), x(mem[14*W]), y(mem[15*W]), z(mem[16*W]),
        dot(mem[17*W]), tmp(mem[18*W]), red(mem[28*W]), off(mem[6*W].template ReinterpretCast<uint32_t>()) {}
  __aicore__ inline void LoadSegmentPlanes() {
    Gather(ax, raw, off, 0, W);
    Gather(ay, raw, off, 4, W);
    Gather(az, raw, off, 8, W);
    Gather(dx, raw, off, 12, W);
    Gather(dy, raw, off, 16, W);
    Gather(dz, raw, off, 20, W);
    Sub(dx, dx, ax, W);
    Sub(dy, dy, ay, W);
    Sub(dz, dz, az, W);
    Mul(len, dx, dx, W);
    Mul(tmp, dy, dy, W);
    Add(len, len, tmp, W);
    Mul(tmp, dz, dz, W);
    Add(len, len, tmp, W);
    Mins(tmp.template ReinterpretCast<int32_t>(), len.template ReinterpretCast<int32_t>(), int32_t(1), W);
    Muls(tmp.template ReinterpretCast<int32_t>(), tmp.template ReinterpretCast<int32_t>(),
      int32_t(-0x3f800000), W);
    Adds(tmp.template ReinterpretCast<int32_t>(), tmp.template ReinterpretCast<int32_t>(),
      int32_t(0x3f800000), W);
    Or(len.template ReinterpretCast<uint16_t>(), len.template ReinterpretCast<uint16_t>(),
     tmp.template ReinterpretCast<uint16_t>(), W * 2);
  }
  __aicore__ inline void ProjectPoint(float px, float py, float pz) {
    Adds(x, ax, -px, W);
    Adds(y, ay, -py, W);
    Adds(z, az, -pz, W);
    Mul(dot, x, dx, W);
    Mul(tmp, y, dy, W);
    Add(dot, dot, tmp, W);
    Mul(tmp, z, dz, W);
    Add(dot, dot, tmp, W);
    Div(dot, dot, len, W);
    Maxs(dot, dot, -1.0f, W);
    Mins(dot, dot, 0.0f, W);
    Mul(tmp, dot, dx, W);
    Sub(x, x, tmp, W);
    Mul(tmp, dot, dy, W);
    Sub(y, y, tmp, W);
    Mul(tmp, dot, dz, W);
    Sub(z, z, tmp, W);
    Mul(tmp, x, x, W);
    MulAddDst(tmp, y, y, W);
    MulAddDst(tmp, z, z, W);
  }
  __aicore__ inline void ReduceNearest(uint32_t m, float &value, int32_t &index) {
    if constexpr (W <= 64) {
      WholeReduceMin(red, tmp, int32_t(m), 1, 1, 1, 8);
      Fence<HardEvent::V_S>();
      value = red.GetValue(0);
      index = red.template ReinterpretCast<int32_t>().GetValue(1);
    } else {
      WholeReduceMin(red, tmp, int32_t(64), W / 64, 1, 1, 8);
      uint32_t tail = m % 64;
      if (tail)
        WholeReduceMin(red[(m / 64) * 2], tmp[(m / 64) * 64], int32_t(tail), 1, 1, 1, 8);
      uint64_t bits[2] = {0, 0};
      for (uint32_t c = 0; c < (m + 63) / 64; ++c)
        bits[0] |= uint64_t(1) << (c * 2);
      WholeReduceMin(red[128], red, bits, 1, 1, 1, 8);
      Fence<HardEvent::V_S>();
      value = red.GetValue(128);
      uint32_t at = red.template ReinterpretCast<uint32_t>().GetValue(129);
      index = int32_t((at / 2) * 64 + red.template ReinterpretCast<uint32_t>().GetValue(at + 1));
    }
  }
};
template <typename T, uint32_t W>
__aicore__ inline void SinglePointTreeLine(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR oi, uint32_t m) {
  GlobalTensor<T> pg, sg;
  pg.SetGlobalBuffer((__gm__ T *)p);
  sg.SetGlobalBuffer((__gm__ T *)s);
  LocalTensor<float> mem(TPosition::VECCALC, 0, 32 * W + 512);
  SingleQueryPlanes<W> planes(mem);
  auto raw = planes.raw;
  auto off = planes.off;
  auto pointValues = mem[32 * W].template ReinterpretCast<T>();
  DataCopyPad(pointValues, pg, {1, uint32_t(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  SeedDeviceOffsets(off, W, 24);
  Duplicate(raw, 0.0f, 6 * W);
  Fence<HardEvent::V_MTE2>();
  if constexpr (IsSameType<T, float>::value) {
    DataCopyPad(raw, sg, {1, m * 24, 0, 0, 0}, {false, 0, 0, 0});
  } else {
    auto h = mem[24 * W].template ReinterpretCast<half>();
    DataCopyPad(h, sg, {1, m * 12, 0, 0, 0}, {false, 0, 0, 0});
  }
  Fence<HardEvent::S_V>();
  ExpandDeviceOffsetSeeds(off, W, 24);
  Fence<HardEvent::MTE2_S>();
  float px = float(pointValues.GetValue(0)), py = float(pointValues.GetValue(1)),
     pz = float(pointValues.GetValue(2));
  Fence<HardEvent::MTE2_V>();
  if constexpr (IsSameType<T, half>::value) {
    Cast(raw, mem[24 * W].template ReinterpretCast<half>(), RoundMode::CAST_NONE, m * 6);
    // The old offset generator's V->S fence also completed this cast.
    // Fresh-table expansion now runs before it, so retain the local RAW barrier.
    PipeBarrier<PIPE_V>();
  }
  planes.LoadSegmentPlanes();
  planes.ProjectPoint(px, py, pz);
  int32_t index;
  float value;
  planes.ReduceNearest(m, value, index);
  StoreSinglePointResult<T>(o, oi, value, index);
}

// Scalar workloads consume a small tiling descriptor. Read its current
// fields directly, leaving UB free for the input DMA; this changes metadata
// transport only and never moves input computation outside the kernel.
__aicore__ inline TreeLineMinDistanceTilingData LoadScalarTilingMetadata(GM_ADDR tilingAddress) {
  GlobalTensor<uint32_t> metadataWords;
  metadataWords.SetGlobalBuffer((__gm__ uint32_t *)tilingAddress);
  TreeLineMinDistanceTilingData metadata;
  metadata.batchCount = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, batchCount) / sizeof(uint32_t));
  metadata.pointsPerBatch = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, pointsPerBatch) / sizeof(uint32_t));
  metadata.segmentCount = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, segmentCount) / sizeof(uint32_t));
  metadata.workerCount = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, workerCount) / sizeof(uint32_t));
  metadata.algorithm = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, algorithm) / sizeof(uint32_t));
  metadata.workersPerBatch = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, workersPerBatch) / sizeof(uint32_t));
  metadata.unifiedBufferBytes = metadataWords.GetValue(
    offsetof(TreeLineMinDistanceTilingData, unifiedBufferBytes) / sizeof(uint32_t));
  return metadata;
}
// Read the metadata-only 51-bit descriptor used exclusively by layout templates.
__aicore__ inline TreeLineMinDistanceTilingData LoadLayoutTemplateMetadata(GM_ADDR address) {
  uint64_t descriptor;
  if ((reinterpret_cast<uint64_t>(address) & 7) == 0) {
    GlobalTensor<uint64_t> words;
    words.SetGlobalBuffer((__gm__ uint64_t *)address);
    descriptor = words.GetValue(0);
  } else {
    GlobalTensor<uint32_t> words;
    words.SetGlobalBuffer((__gm__ uint32_t *)address);
    descriptor = uint64_t(words.GetValue(0)) | (uint64_t(words.GetValue(1)) << 32);
  }
  TreeLineMinDistanceTilingData metadata;
  metadata.batchCount = uint32_t(descriptor & 7) + 1;
  metadata.pointsPerBatch = uint32_t((descriptor >> kLayoutDescriptorPointShiftBits) & 16383) + 1;
  metadata.segmentCount = uint32_t((descriptor >> kLayoutDescriptorSegmentShiftBits) & 1023) + 1;
  metadata.algorithm = 2;
  metadata.workerCount = uint32_t((descriptor >> kLayoutDescriptorWorkerShiftBits) & 63) + 1;
  metadata.workersPerBatch = 0; // Tiled ownership uses total points and workerCount.
  metadata.unifiedBufferBytes = uint32_t((descriptor >> kLayoutDescriptorBufferShiftBits) & 262143);
  return metadata;
}

// FP32 distance and int32 index have equal element widths. A two-block
// DMA can write both current results when the address gap fits dstStride.
// Pack in increasing GM address order; all other addresses use the original
// two requests. FP16 keeps its original unequal-width writeback.
__aicore__ inline bool TryStorePairedFloatResults(GM_ADDR distanceAddress, GM_ADDR indexAddress,
                        uint32_t outputUbByteOffset,
                        float squaredDistance, int32_t segmentIndex) {
  uint64_t distanceByteAddress = reinterpret_cast<uint64_t>(distanceAddress);
  uint64_t indexByteAddress = reinterpret_cast<uint64_t>(indexAddress);
  bool distanceFirst = distanceByteAddress < indexByteAddress;
  uint64_t gapBytes = distanceFirst ? indexByteAddress - distanceByteAddress
                  : distanceByteAddress - indexByteAddress;
  if (gapBytes < sizeof(uint32_t) || gapBytes - sizeof(uint32_t) > uint64_t(0xffffffffU))
    return false;
  float distanceValue = float(sqrt(squaredDistance));
  uint32_t distanceBits = __builtin_bit_cast(uint32_t, distanceValue);
  uint32_t indexBits = __builtin_bit_cast(uint32_t, segmentIndex);
  LocalTensor<uint32_t> packedResults(TPosition::VECCALC, outputUbByteOffset, 16);
  packedResults.SetValue(0, distanceFirst ? distanceBits : indexBits);
  packedResults.SetValue(8, distanceFirst ? indexBits : distanceBits);
  GlobalTensor<uint32_t> resultAddress;
  resultAddress.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(
    distanceFirst ? distanceAddress : indexAddress));
  Fence<HardEvent::S_MTE3>();
  DataCopyPad(resultAddress, packedResults,
    {2, sizeof(uint32_t), 0, uint32_t(gapBytes - sizeof(uint32_t)), 0});
  Fence<HardEvent::MTE3_S>();
  return true;
}

// Complete current small M1 batch; no state survives this invocation.
// Use the same direct one-segment projection as FixedPoint<T,1>. Segment
// direction/length are shared across points within this one invocation only.
template <typename T, uint32_t PointCount>
__aicore__ inline void CoalescedSingleSegment(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR indices) {
  static_assert(PointCount >= 2 && PointCount <= kCoalescedSingleSegmentMaximumPoints);
  if (GetBlockIdx() != 0)
    return;
  constexpr uint32_t pointBufferBytes = (PointCount * 3 * sizeof(T) + 31) / 32 * 32;
  constexpr uint32_t segmentBufferBytes = 32;
  constexpr uint32_t outputOffsetBytes = pointBufferBytes + segmentBufferBytes;
  GlobalTensor<T> points, segments, distances;
  GlobalTensor<int32_t> nearest;
  points.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p));
  segments.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(s));
  distances.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(o));
  nearest.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
  LocalTensor<T> pointValues(TPosition::VECCALC, 0, pointBufferBytes / sizeof(T));
  LocalTensor<T> segmentValues(TPosition::VECCALC, pointBufferBytes, segmentBufferBytes / sizeof(T));
  LocalTensor<T> output(TPosition::VECCALC, outputOffsetBytes, 32 / sizeof(T));
  LocalTensor<int32_t> indexValues(TPosition::VECCALC, outputOffsetBytes + 32, 8);
  DataCopyPad(pointValues, points, {1, PointCount * 3 * sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
  DataCopyPad(segmentValues, segments, {1, 6 * sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
  Fence<HardEvent::MTE2_S>();
  float a = float(segmentValues.GetValue(0));
  float b = float(segmentValues.GetValue(1));
  float c = float(segmentValues.GetValue(2));
  float u = float(segmentValues.GetValue(3)) - a;
  float v = float(segmentValues.GetValue(4)) - b;
  float w = float(segmentValues.GetValue(5)) - c;
  float squaredLength = (u * u + v * v) + w * w;
#pragma unroll
  for (uint32_t point = 0; point < PointCount; ++point) {
    float px = float(pointValues.GetValue(point * 3));
    float py = float(pointValues.GetValue(point * 3 + 1));
    float pz = float(pointValues.GetValue(point * 3 + 2));
    float projection = 0.0f;
    if (squaredLength > 0.0f)
      projection = (((px - a) * u + (py - b) * v) + (pz - c) * w) / squaredLength;
    projection = projection < 0.0f ? 0.0f : (projection > 1.0f ? 1.0f : projection);
    float ex = px - (a + projection * u);
    float ey = py - (b + projection * v);
    float ez = pz - (c + projection * w);
    float squaredDistance = (ex * ex + ey * ey) + ez * ez;
    output.SetValue(point, T(sqrt(squaredDistance)));
    // Exactly one valid candidate exists, so its index is always zero.
    indexValues.SetValue(point, int32_t(0));
  }
  Fence<HardEvent::S_MTE3>();
  DataCopyPad(distances, output, {1, PointCount * sizeof(T), 0, 0, 0});
  DataCopyPad(nearest, indexValues, {1, PointCount * sizeof(int32_t), 0, 0, 0});
  Fence<HardEvent::MTE3_S>();
}

// Each productive block owns one point; the launch preserves the minimum eight cores.
// All coordinates are loaded and every one of the M segments is evaluated.
template <typename T, uint32_t SegmentCount>
__aicore__ inline void FindNearestStaticSegment(LocalTensor<T> pointValues, LocalTensor<T> segmentValues,
                                                 float &best, int32_t &arg) {
  float px = float(pointValues.GetValue(0)), py = float(pointValues.GetValue(1)),
     pz = float(pointValues.GetValue(2));
#pragma unroll
  for (uint32_t segment = 0; segment < SegmentCount; ++segment) {
    uint32_t coordinate = segment * 6;
    float value = PointSegmentSquaredDistance(px, py, pz,
      float(segmentValues.GetValue(coordinate)), float(segmentValues.GetValue(coordinate + 1)),
      float(segmentValues.GetValue(coordinate + 2)), float(segmentValues.GetValue(coordinate + 3)),
      float(segmentValues.GetValue(coordinate + 4)), float(segmentValues.GetValue(coordinate + 5)));
    if (segment == 0 || value < best) {
      best = value;
      arg = int32_t(segment);
    }
  }
}
template <typename T, uint32_t PointsPerBatch, uint32_t SegmentCount, bool SingleBatch = false>
__aicore__ inline void StaticScalarDomain(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR indices,
                     GM_ADDR tiling) {
  // Eight launched cores preserve the existing minimum group policy. Small
  // point rows may need padding; one current metadata word bounds ownership.
  // N>=8 never pads because every selected worker owns exactly one point.
  if constexpr (SingleBatch && PointsPerBatch != 0) {
    if (GetBlockIdx() >= PointsPerBatch)
      return;
  } else if constexpr (!SingleBatch && PointsPerBatch < 8) {
    GlobalTensor<uint32_t> schedulingWords;
    schedulingWords.SetGlobalBuffer((__gm__ uint32_t *)tiling);
    uint32_t workerCount = schedulingWords.GetValue(
      offsetof(TreeLineMinDistanceTilingData, workerCount) / sizeof(uint32_t));
    if (GetBlockIdx() >= workerCount)
      return;
  }
  static_assert(SegmentCount >= 1 && SegmentCount <= 16);
  static_assert((PointsPerBatch >= 1 && PointsPerBatch <= 16) ||
                (SingleBatch && PointsPerBatch == 0 && SegmentCount == 2));
  static_assert(PointsPerBatch * SegmentCount <= kMaximumScalarPairCount);
  uint32_t pointIndex = GetBlockIdx(), batchIndex = 0;
  if constexpr (!SingleBatch)
    batchIndex = pointIndex / PointsPerBatch;
  GlobalTensor<T> points, segments, distances;
  GlobalTensor<int32_t> nearest;
  points.SetGlobalBuffer((__gm__ T *)p);
  segments.SetGlobalBuffer((__gm__ T *)s);
  distances.SetGlobalBuffer((__gm__ T *)o);
  nearest.SetGlobalBuffer((__gm__ int32_t *)indices);
  constexpr uint32_t segmentBytes = (SegmentCount * 6 * sizeof(T) + 31) / 32 * 32;
  constexpr uint32_t outputOffsetBytes = 32 + segmentBytes;
  LocalTensor<T> pointValues(TPosition::VECCALC, 0, 32 / sizeof(T));
  LocalTensor<T> segmentValues(TPosition::VECCALC, 32, segmentBytes / sizeof(T));
  DataCopyPad(pointValues, points[pointIndex * 3], {1, 3 * sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
  DataCopyPad(segmentValues, segments[batchIndex * SegmentCount * 6],
        {1, SegmentCount * 6 * sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
  Fence<HardEvent::MTE2_S>();
  float best = 3.402823466e38f;
  int32_t arg = 0;
  FindNearestStaticSegment<T, SegmentCount>(pointValues, segmentValues, best, arg);
  if constexpr (IsSameType<T, float>::value) {
    if (TryStorePairedFloatResults(o + pointIndex * sizeof(float),
                   indices + pointIndex * sizeof(int32_t),
                   outputOffsetBytes, best, arg))
      return;
  }
  LocalTensor<T> output(TPosition::VECCALC, outputOffsetBytes, 32 / sizeof(T));
  LocalTensor<int32_t> indexValues(TPosition::VECCALC, outputOffsetBytes + 32, 8);
  output.SetValue(0, T(sqrt(best)));
  indexValues.SetValue(0, arg);
  Fence<HardEvent::S_MTE3>();
  DataCopyPad(distances[pointIndex], output, {1, sizeof(T), 0, 0, 0});
  DataCopyPad(nearest[pointIndex], indexValues, {1, 4, 0, 0, 0});
  Fence<HardEvent::MTE3_S>();
}

template <typename T, uint32_t Algorithm>
__aicore__ inline void ExecuteScalarOrPackedLayout(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR indices,
                   GM_ADDR tiling) {
  if constexpr (Algorithm >= 768) {
    StaticScalarDomain<T, Algorithm - 768 + 2, 1, true>(p, s, o, indices, tiling);
  } else if constexpr (Algorithm == 608) {
    StaticScalarDomain<T, 0, 2, true>(p, s, o, indices, tiling);
  } else if constexpr (Algorithm >= 512) {
    constexpr uint32_t shape = Algorithm - 512;
    StaticScalarDomain<T, (shape >> 4) + 2, (shape & 15) + 2, true>(p, s, o, indices, tiling);
  } else if constexpr (Algorithm >= kStaticSingleSegmentAlgorithmBase) {
    constexpr uint32_t pointsPerBatch = Algorithm - kStaticSingleSegmentAlgorithmBase + 1;
    StaticScalarDomain<T, pointsPerBatch, 1>(p, s, o, indices, tiling);
  } else if constexpr (Algorithm >= kStaticScalarAlgorithmBase) {
    constexpr uint32_t shape = Algorithm - kStaticScalarAlgorithmBase;
    constexpr uint32_t pointsPerBatch = (shape >> kStaticScalarPointShift) + 1;
    constexpr uint32_t segmentCount = (shape & kStaticScalarSegmentMask) + 2;
    StaticScalarDomain<T, pointsPerBatch, segmentCount>(p, s, o, indices, tiling);
  } else {
    auto metadata = LoadLayoutTemplateMetadata(tiling);
    if (GetBlockIdx() >= metadata.workerCount)
      return;
    if constexpr (Algorithm == 57) {
      metadata.segmentCount = 1;
      metadata.algorithm = 3;
      KernelTreeLineMinDistance<T, 3> op;
      op.Init(p, s, o, indices, metadata);
      op.Process();
      PipeBarrier<PIPE_ALL>();
    } else if constexpr (Algorithm <= 51) {
      constexpr uint32_t segmentPitchElements = (Algorithm - 35) * 64;
      TiledTreeLine<T, segmentPitchElements> op;
      op.Run(p, s, o, indices, metadata);
    } else {
      constexpr uint32_t segmentLanes = Algorithm == 55 ? 2 :
        (Algorithm == 56 ? 4 : (Algorithm == 54 ? 8 : (Algorithm == 52 ? 16 : 32)));
      metadata.algorithm = Algorithm - 20;
      PackedRowsTreeLine<T, segmentLanes, true> op;
      op.Run(p, s, o, indices, metadata);
    }
  }
}
template <typename T, uint32_t Algorithm>
__aicore__ inline void ExecuteNewLayout(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR indices,
                   GM_ADDR workspace, GM_ADDR tiling) {
  if constexpr (Algorithm >= kCoalescedSingleSegmentAlgorithmBase &&
         Algorithm < kCoalescedSingleSegmentAlgorithmBase + kCoalescedSingleSegmentMaximumPoints - 1) {
    CoalescedSingleSegment<T, Algorithm - kCoalescedSingleSegmentAlgorithmBase + 2>(p, s, o, indices);
  } else if constexpr (Algorithm >= kDenseScratchAlgorithmBase &&
         Algorithm < kDenseScratchAlgorithmBase + kDenseScratchPitchCount) {
    auto metadata = LoadLayoutTemplateMetadata(tiling);
    if (GetBlockIdx() >= metadata.workerCount)
      return;
    constexpr uint32_t pitchElements = 256 + (Algorithm - kDenseScratchAlgorithmBase) * 64;
    TiledTreeLine<T, pitchElements, true> op;
    op.Run(p, s, o, indices, metadata);
  } else if constexpr (Algorithm >= kWideOutputAlgorithmBase &&
         Algorithm < kWideOutputAlgorithmBase + kWideOutputWidthCount) {
    // This route has exactly one logical output. Read only its current
    // segment count; the appended tags keep the original full descriptor.
    if (GetBlockIdx() != 0)
      return;
    GlobalTensor<uint32_t> metadataWords;
    metadataWords.SetGlobalBuffer((__gm__ uint32_t *)tiling);
    uint32_t segmentCount = metadataWords.GetValue(
      offsetof(TreeLineMinDistanceTilingData, segmentCount) / sizeof(uint32_t));
    SinglePointTreeLine<T, (16U << (Algorithm - kWideOutputAlgorithmBase))>(
      p, s, o, indices, segmentCount);
    PipeBarrier<PIPE_ALL>();
  } else if constexpr (Algorithm >= kOptimizedGenericAlgorithmBase && Algorithm < 512) {
    constexpr uint32_t originalAlgorithm = Algorithm - kOptimizedGenericAlgorithmBase;
    auto metadata = LoadLayoutTemplateMetadata(tiling);
    if (GetBlockIdx() >= metadata.workerCount)
      return;
    metadata.algorithm = originalAlgorithm;
    KernelTreeLineMinDistance<T, originalAlgorithm> op;
    op.Init(p, s, o, indices, metadata);
    op.Process();
    PipeBarrier<PIPE_ALL>();
  } else {
    ExecuteScalarOrPackedLayout<T, Algorithm>(p, s, o, indices, tiling);
  }
}

}
template <typename DT_TREE_POINTS, uint32_t ALGORITHM>
__global__ __aicore__ void tree_line_min_distance(GM_ADDR tree_points, GM_ADDR line_segments,
  GM_ADDR min_distance, GM_ADDR nearest_segment_idx, GM_ADDR workspace, GM_ADDR tiling) {
  REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
  if constexpr (ALGORITHM <= 36) {

  if constexpr (ALGORITHM >= 5 && ALGORITHM <= 12) {
    FixedPoint<DT_TREE_POINTS, ALGORITHM - 4>(tree_points, line_segments, min_distance,
                         nearest_segment_idx);
    return;
  } else {
    GET_TILING_DATA_WITH_STRUCT(TreeLineMinDistanceTilingData, t, tiling);
    if (GetBlockIdx() >= t.workerCount)
      return;
    if constexpr (ALGORITHM == 1) {
      if (t.segmentCount == 1 && t.batchCount * t.pointsPerBatch >= 8)
        CachedSingleSegmentTiny<DT_TREE_POINTS>(tree_points, line_segments, min_distance,
                            nearest_segment_idx, t);
      else
        ScalarDmaTiny<DT_TREE_POINTS>(tree_points, line_segments, min_distance, nearest_segment_idx,
                       t);
      PipeBarrier<PIPE_ALL>();
      return;
    } else if constexpr (ALGORITHM == 4) {
      SpatialTreeLine<DT_TREE_POINTS> op;
      op.Run(tree_points, line_segments, min_distance, nearest_segment_idx,
         GetUserWorkspace(workspace), t);
      return;
    } else {
      static_assert(ALGORITHM == 3, "Unregistered legacy TreeLineMinDistance tiling key");
      ShortSegmentPointTiles<DT_TREE_POINTS> op;
      op.Init(tree_points, line_segments, min_distance, nearest_segment_idx, t);
      op.Process();
      PipeBarrier<PIPE_ALL>();
    }
  }

  } else {
    OptimizedLayout::ExecuteNewLayout<DT_TREE_POINTS, ALGORITHM>(tree_points, line_segments,
      min_distance, nearest_segment_idx, workspace, tiling);
  }
}
