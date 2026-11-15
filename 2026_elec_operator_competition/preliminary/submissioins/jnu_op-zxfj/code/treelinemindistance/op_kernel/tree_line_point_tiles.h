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

#pragma once
#include "tree_line_device_common.h"

// Vectorized point tiles for short segment tables.
template <typename T> class ShortSegmentPointTiles : public TreeLinePointKernel<T> {
  uint32_t bufferBase = 0;
  using TreeLinePointKernel<T>::addresses;
  using TreeLinePointKernel<T>::t;
  LocalTensor<float> raw, ax, ay, az, x, y, z, dot, tmp, dist;
  LocalTensor<uint32_t> gather;
  __aicore__ inline void SingleSegmentDistance(uint32_t count, float u, float v, float q, float l,
                                             LocalTensor<float> next, LocalTensor<uint8_t> mask) {
    Muls(dot, ax, u, count);
    Axpy(dot, ay, v, count);
    Axpy(dot, az, q, count);
    NormalizeProjection(count, u, v, q, l, dot, raw, tmp, dist, next, mask);
    FinishSingleSegment(count, u, v, q, ax, ay, az, dot, dist, mask);
  }
  __aicore__ inline void CandidateDistance(uint32_t count, float a, float b, float c,
                                       float u, float v, float q, float l, LocalTensor<float> next) {
    Muls(dot, ax, u, count);
    Muls(tmp, ay, v, count);
    Add(dot, dot, tmp, count);
    Muls(tmp, az, q, count);
    Add(dot, dot, tmp, count);
    Duplicate(next, u, count);
    Mul(next, next, next, count);
    Duplicate(tmp, v, count);
    Mul(tmp, tmp, tmp, count);
    Add(next, next, tmp, count);
    Duplicate(tmp, q, count);
    Mul(tmp, tmp, tmp, count);
    Add(next, next, tmp, count);
    if (l == 0.0f)
      Duplicate(next, 1.0f, count);
    Div(tmp, dot, next, count);
    Muls(dist, tmp, -1.0f, count);
    MulAddDst(dot, dist, next, count);
    Div(dot, dot, next, count);
    Add(dot, tmp, dot, count);
    Maxs(dot, dot, 0.0f, count);
    Mins(dot, dot, 1.0f, count);
    Muls(ax, dot, u, count);
    Muls(ay, dot, v, count);
    Muls(az, dot, q, count);
    Adds(ax, ax, a, count);
    Adds(ay, ay, b, count);
    Adds(az, az, c, count);
    Sub(ax, x, ax, count);
    Sub(ay, y, ay, count);
    Sub(az, z, az, count);
    Mul(ax, ax, ax, count);
    Mul(ay, ay, ay, count);
    Mul(az, az, az, count);
    Add(dist, ax, ay, count);
    Add(dist, dist, az, count);
  }
  __aicore__ inline void LoadPoints(uint32_t i, uint32_t w, uint32_t live) {
    Duplicate(raw, 0.0f, w * 3);
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(raw, addresses.points[i * 3], {1, live * 12, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
    } else {
      auto hr = raw.ReinterpretCast<half>()[w * 6];
      DataCopyPad(hr, addresses.points[i * 3], {1, live * 6, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
      Cast(raw, hr, RoundMode::CAST_NONE, live * 3);
    }
    Gather(x, raw, gather, 0, live);
    Gather(y, raw, gather, 4, live);
    Gather(z, raw, gather, 8, live);
  }
  __aicore__ inline void WritePointTile(uint32_t i, uint32_t count, uint32_t live, LocalTensor<float> best,
                                      LocalTensor<float> arg, LocalTensor<float> next) {
    Sqrt(best, best, count);
    if (t.segmentCount == 1)
      Duplicate(next.template ReinterpretCast<int32_t>(), int32_t(0), count);
    else
      Cast(next.template ReinterpretCast<int32_t>(), arg, RoundMode::CAST_RINT, count);
    StorePointMinima(addresses.output, addresses.indicesOut, i, live, count, best, next);
  }
  __aicore__ inline void LoadBatchSegments(uint32_t batch, LocalTensor<float> segmentLocal) {
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(segmentLocal, addresses.segments[batch * t.segmentCount * 6],
            {1, t.segmentCount * 24, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_S>();
    } else {
      auto halfSegments = segmentLocal[64].ReinterpretCast<half>();
      DataCopyPad(halfSegments, addresses.segments[batch * t.segmentCount * 6],
            {1, t.segmentCount * 12, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
      Cast(segmentLocal, halfSegments, RoundMode::CAST_NONE, t.segmentCount * 6);
      Fence<HardEvent::V_S>();
    }
  }
  __aicore__ inline void SegmentCoordinates(uint32_t batch, uint32_t j, LocalTensor<float> segmentLocal,
                                        float &a, float &b, float &c, float &u, float &v, float &q) {
    if (t.segmentCount == 1) {
      uint32_t k = batch * 6;
      a = float(addresses.segments.GetValue(k));
      b = float(addresses.segments.GetValue(k + 1));
      c = float(addresses.segments.GetValue(k + 2));
      u = float(addresses.segments.GetValue(k + 3)) - a;
      v = float(addresses.segments.GetValue(k + 4)) - b;
      q = float(addresses.segments.GetValue(k + 5)) - c;
    } else {
      uint32_t k = j * 6;
      a = segmentLocal.GetValue(k);
      b = segmentLocal.GetValue(k + 1);
      c = segmentLocal.GetValue(k + 2);
      u = segmentLocal.GetValue(k + 3) - a;
      v = segmentLocal.GetValue(k + 4) - b;
      q = segmentLocal.GetValue(k + 5) - c;
    }
  }
  __aicore__ inline void BindPointCoordinates(LocalTensor<float> mem, uint32_t w) {
    raw = mem;
    x = mem[6 * w];
    y = mem[7 * w];
    z = mem[8 * w];
    ax = mem[9 * w];
    ay = mem[10 * w];
    az = mem[11 * w];
    dot = mem[12 * w];
    tmp = mem[13 * w];
    dist = mem[14 * w];
  }
  __aicore__ inline void AccumulateCandidates(uint32_t batch, uint32_t count, LocalTensor<float> segmentLocal, LocalTensor<float> &best, LocalTensor<float> arg, LocalTensor<float> next, LocalTensor<uint8_t> mask) {
    if (t.segmentCount != 1) {
      Duplicate(best, 3.402823466e38f, count);
      Duplicate(arg, 0.0f, count);
    }
    for (uint32_t j = 0; j < t.segmentCount; ++j) {
      float a, b, c, u, v, q;
      SegmentCoordinates(batch, j, segmentLocal, a, b, c, u, v, q);
      float l = (u * u + v * v) + q * q;
      Adds(ax, x, -a, count);
      Adds(ay, y, -b, count);
      Adds(az, z, -c, count);
      if (t.segmentCount == 1) {
        SingleSegmentDistance(count, u, v, q, l, next, mask);
      } else {
        CandidateDistance(count, a, b, c, u, v, q, l, next);
      }
      if (t.segmentCount == 1)
        best = dist;
      else {
        Compare(mask, dist, best, CMPMODE::LT, count);
        Duplicate(next, float(int32_t(j)), count);
        Select(best, mask, dist, best, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        Select(arg, mask, next, arg, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
      }
    }
  }
  __aicore__ inline void PointVector(uint32_t start, uint32_t end) {
    uint32_t w = (end - start + 63) / 64 * 64;
    if (w > 2048)
      w = 2048;
    LocalTensor<float> mem(TPosition::VECCALC, bufferBase,
               (t.unifiedBufferBytes - 8192 - bufferBase) / 4);
    BindPointCoordinates(mem, w);
    auto best = mem[15 * w];
    auto arg = mem[16 * w];
    auto next = mem[17 * w];
    gather = mem[18 * w].template ReinterpretCast<uint32_t>();
    auto mask = mem[19 * w].template ReinterpretCast<uint8_t>();
    auto segmentLocal = mem[19 * w + 256];
    DeviceOffsets(gather, w, 12);
    uint32_t oldBatch = 0xffffffff;
    for (uint32_t i = start; i < end;) {
      uint32_t batch = i / t.pointsPerBatch, count = end - i;
      if (count > w)
        count = w;
      // All points in a batch share the same short segment table.
      if (t.segmentCount != 1 && batch != oldBatch) {
        LoadBatchSegments(batch, segmentLocal);
        oldBatch = batch;
      }
      uint32_t remain = (batch + 1) * t.pointsPerBatch - i;
      if (count > remain)
        count = remain;
      uint32_t live = count;
      count = (count + 63) / 64 * 64;
      LoadPoints(i, w, live);
      AccumulateCandidates(batch, count, segmentLocal, best, arg, next, mask);
      WritePointTile(i, count, live, best, arg, next);
      i += live;
    }
  }
public:
  __aicore__ inline void Process() {
    const auto range = AssignedPoints(t);
    PointVector(range.begin, range.end);
  }
};

template <typename T>
__aicore__ inline void FindCachedMinimum(LocalTensor<T> pointValues, LocalTensor<T> segmentValues,
    LocalTensor<T> distanceValues, LocalTensor<int32_t> indexValues, uint32_t localPoint, uint32_t segmentCount) {
  float px = float(pointValues.GetValue(localPoint * 3));
  float py = float(pointValues.GetValue(localPoint * 3 + 1));
  float pz = float(pointValues.GetValue(localPoint * 3 + 2));
  float best = 3.402823466e38f;
  int32_t arg = 0;
  for (uint32_t segment = 0; segment < segmentCount; ++segment) {
    uint32_t coordinate = segment * 6;
    float a = float(segmentValues.GetValue(coordinate));
    float b = float(segmentValues.GetValue(coordinate + 1));
    float c = float(segmentValues.GetValue(coordinate + 2));
    float endX = float(segmentValues.GetValue(coordinate + 3));
    float endY = float(segmentValues.GetValue(coordinate + 4));
    float endZ = float(segmentValues.GetValue(coordinate + 5));
    float squaredDistance = PointSegmentSquaredDistance(px, py, pz, a, b, c, endX, endY, endZ);
    if (segment == 0 || squaredDistance < best) {
      best = squaredDistance;
      arg = int32_t(segment);
    }
  }
  distanceValues.SetValue(localPoint, T(sqrt(best)));
  indexValues.SetValue(localPoint, arg);
}
template <typename T>
__aicore__ inline void CachedSingleSegmentTiny(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR idx,
                       const TreeLineMinDistanceTilingData &t) {
  constexpr uint32_t maximumSlicePoints = 32, maximumScalarSegments = 32;
  constexpr uint32_t elementBytes = sizeof(T), indexBytes = sizeof(int32_t);
  constexpr uint32_t pointBufferBytes = maximumSlicePoints * 3 * sizeof(float);
  constexpr uint32_t segmentBufferBytes = maximumScalarSegments * 6 * sizeof(float);
  constexpr uint32_t distanceBufferBytes = maximumSlicePoints * sizeof(float);
  GlobalTensor<T> points, segments, output;
  GlobalTensor<int32_t> indicesOut;
  points.SetGlobalBuffer((__gm__ T *)p);
  segments.SetGlobalBuffer((__gm__ T *)s);
  output.SetGlobalBuffer((__gm__ T *)o);
  indicesOut.SetGlobalBuffer((__gm__ int32_t *)idx);
  uint32_t totalPoints = t.batchCount * t.pointsPerBatch;
  uint32_t begin = totalPoints * GetBlockIdx() / t.workerCount;
  uint32_t end = totalPoints * (GetBlockIdx() + 1) / t.workerCount;
  uint32_t slicePoints = end - begin;
  LocalTensor<T> pointValues(TPosition::VECCALC, 0, maximumSlicePoints * 3);
  LocalTensor<T> segmentValues(TPosition::VECCALC, pointBufferBytes, maximumScalarSegments * 6);
  LocalTensor<T> distanceValues(TPosition::VECCALC, pointBufferBytes + segmentBufferBytes,
                 maximumSlicePoints);
  LocalTensor<int32_t> indexValues(TPosition::VECCALC,
                  pointBufferBytes + segmentBufferBytes + distanceBufferBytes,
                  maximumSlicePoints);
  DataCopyPad(pointValues, points[begin * 3], {1, slicePoints * 3 * elementBytes, 0, 0, 0},
        {false, 0, 0, 0});
  uint32_t loadedBatch = uint32_t(-1);
  for (uint32_t point = begin; point < end; ++point) {
    uint32_t batch = point / t.pointsPerBatch, localPoint = point - begin;
    if (batch != loadedBatch) {
      // Only this batch reuses the table. The fence also protects a later
      // reload when a nonaligned worker slice crosses a batch boundary.
      Fence<HardEvent::S_MTE2>();
      DataCopyPad(segmentValues, segments[batch * t.segmentCount * 6],
            {1, t.segmentCount * 6 * elementBytes, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_S>();
      loadedBatch = batch;
    }
    FindCachedMinimum(pointValues, segmentValues, distanceValues, indexValues, localPoint, t.segmentCount);
  }
  Fence<HardEvent::S_MTE3>();
  DataCopyPad(output[begin], distanceValues, {1, slicePoints * elementBytes, 0, 0, 0});
  DataCopyPad(indicesOut[begin], indexValues, {1, slicePoints * indexBytes, 0, 0, 0});
  Fence<HardEvent::MTE3_S>();
}
template <typename T>
__aicore__ inline void ScalarDmaTiny(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR idx,
                  const TreeLineMinDistanceTilingData &t) {
  uint32_t i = GetBlockIdx(), batch = i / t.pointsPerBatch;
  GlobalTensor<T> points, segments, output;
  GlobalTensor<int32_t> indicesOut;
  points.SetGlobalBuffer((__gm__ T *)p);
  segments.SetGlobalBuffer((__gm__ T *)s);
  output.SetGlobalBuffer((__gm__ T *)o);
  indicesOut.SetGlobalBuffer((__gm__ int32_t *)idx);
  // Two independent DMA requests stage the whole bounded scalar workload.
  // Scalar reads then use UB rather than serial GM cache-line misses.
  constexpr uint32_t segmentBufferBytes = kMaximumScalarPairCount * 6 * sizeof(T);
  constexpr uint32_t outputBaseBytes = 32 + segmentBufferBytes;
  LocalTensor<T> pointValues(TPosition::VECCALC, 0, 32 / sizeof(T));
  LocalTensor<T> segmentValues(TPosition::VECCALC, 32, kMaximumScalarPairCount * 6);
  DataCopyPad(pointValues, points[i * 3], {1, uint32_t(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  DataCopyPad(segmentValues, segments[batch * t.segmentCount * 6],
        {1, t.segmentCount * uint32_t(6 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  Fence<HardEvent::MTE2_S>();
  float px = float(pointValues.GetValue(0)), py = float(pointValues.GetValue(1)),
     pz = float(pointValues.GetValue(2));
  float best = 3.402823466e38f;
  int32_t arg = 0;
  for (uint32_t j = 0; j < t.segmentCount; ++j) {
    uint32_t k = j * 6;
    float a = float(segmentValues.GetValue(k)), b = float(segmentValues.GetValue(k + 1)),
       c = float(segmentValues.GetValue(k + 2));
    float endX = float(segmentValues.GetValue(k + 3)), endY = float(segmentValues.GetValue(k + 4)),
       endZ = float(segmentValues.GetValue(k + 5));
    float d = PointSegmentSquaredDistance(px, py, pz, a, b, c, endX, endY, endZ);
    if (j == 0 || d < best) {
      best = d;
      arg = int32_t(j);
    }
  }
  LocalTensor<T> dist(TPosition::VECCALC, outputBaseBytes, 32 / sizeof(T));
  LocalTensor<int32_t> ind(TPosition::VECCALC, outputBaseBytes + 32, 8);
  dist.SetValue(0, T(sqrt(best)));
  ind.SetValue(0, arg);
  Fence<HardEvent::S_MTE3>();
  DataCopyPad(output[i], dist, {1, sizeof(T), 0, 0, 0});
  DataCopyPad(indicesOut[i], ind, {1, 4, 0, 0, 0});
  Fence<HardEvent::MTE3_S>();
}
template <typename T, uint32_t M>
__aicore__ inline void RankFixedSegments(LocalTensor<T> coordinates, uint32_t segmentBaseElements,
    float px, float py, float pz, float &best, int32_t &arg) {
#pragma unroll
  for (uint32_t j = 0; j < M; ++j) {
    float a = float(coordinates.GetValue(segmentBaseElements + j * 6)),
       b = float(coordinates.GetValue(segmentBaseElements + j * 6 + 1)),
       c = float(coordinates.GetValue(segmentBaseElements + j * 6 + 2));
    float d;
    if constexpr (M == 1) {
      float u = float(coordinates.GetValue(segmentBaseElements + j * 6 + 3)) - a,
         v = float(coordinates.GetValue(segmentBaseElements + j * 6 + 4)) - b,
         w = float(coordinates.GetValue(segmentBaseElements + j * 6 + 5)) - c;
      float l = (u * u + v * v) + w * w, q = 0.0f;
      if (l > 0.0f)
        q = (((px - a) * u + (py - b) * v) + (pz - c) * w) / l;
      q = q < 0.0f ? 0.0f : (q > 1.0f ? 1.0f : q);
      float ex = px - (a + q * u), ey = py - (b + q * v), ez = pz - (c + q * w),
         squaredDistance = (ex * ex + ey * ey) + ez * ez;
      d = squaredDistance;
    } else {
      float endX = float(coordinates.GetValue(segmentBaseElements + j * 6 + 3)),
         endY = float(coordinates.GetValue(segmentBaseElements + j * 6 + 4)),
         endZ = float(coordinates.GetValue(segmentBaseElements + j * 6 + 5));
      constexpr bool clipEndpoints = M == 2 || IsSameType<T, float>::value;
      d = PointSegmentSquaredDistance<clipEndpoints>(px, py, pz, a, b, c, endX, endY, endZ);
    }
    if (j == 0 || d < best) {
      best = d;
      arg = int32_t(j);
    }
  }
}
template <typename T, uint32_t M>
__aicore__ inline void FixedPoint(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR idx) {
  if (GetBlockIdx() != 0)
    return;
  GlobalTensor<T> points, segments, output;
  GlobalTensor<int32_t> indicesOut;
  points.SetGlobalBuffer((__gm__ T *)p);
  segments.SetGlobalBuffer((__gm__ T *)s);
  output.SetGlobalBuffer((__gm__ T *)o);
  indicesOut.SetGlobalBuffer((__gm__ int32_t *)idx);
  constexpr uint32_t segmentBaseElements = 32 / sizeof(T);
  static_assert(segmentBaseElements + M * 6 <= 64);
  LocalTensor<T> coordinates(TPosition::VECCALC, 0, 64);
  DataCopyPad(coordinates, points, {1, 3 * sizeof(T), 0, 0, 0}, {false, 0, 0, 0});
  DataCopyPad(coordinates[segmentBaseElements], segments, {1, M * 6 * sizeof(T), 0, 0, 0},
        {false, 0, 0, 0});
  Fence<HardEvent::MTE2_S>();
  float px = float(coordinates.GetValue(0)), py = float(coordinates.GetValue(1)),
     pz = float(coordinates.GetValue(2));
  float best = 3.402823466e38f;
  int32_t arg = 0;
  RankFixedSegments<T, M>(coordinates, segmentBaseElements, px, py, pz, best, arg);
  if constexpr (M <= 2 && IsSameType<T, float>::value) {
    // FixedPoint has exactly one producer. The bypass stores and DDR fence
    // are the same completion protocol as the existing wide single query.
    float distanceValue = float(sqrt(best));
    WriteGmByPassDCache(reinterpret_cast<__gm__ uint32_t *>(o),
                       __builtin_bit_cast(uint32_t, distanceValue));
    WriteGmByPassDCache(reinterpret_cast<__gm__ int32_t *>(idx), arg);
    DataSyncBarrier<MemDsbT::DDR>();
    return;
  }
  coordinates.SetValue(0, T(sqrt(best)));
  LocalTensor<int32_t> indexLocal(TPosition::VECCALC, 256, 8);
  indexLocal.SetValue(0, arg);
  Fence<HardEvent::S_MTE3>();
  DataCopyPad(output, coordinates, {1, sizeof(T), 0, 0, 0});
  DataCopyPad(indicesOut, indexLocal, {1, 4, 0, 0, 0});
  Fence<HardEvent::MTE3_S>();
}
