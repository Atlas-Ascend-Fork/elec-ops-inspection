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
#include "kernel_operator.h"
#include "tiling_key_tree_line_min_distance.h"
#include "tree_line_min_distance_tiling.h"
using namespace AscendC;
template <HardEvent E> __aicore__ inline void Fence() {
  SetFlag<E>(0);
  WaitFlag<E>(0);
}
// Build device-local byte-offset sequences by doubling an aligned prefix.
__aicore__ inline void DeviceOffsets(LocalTensor<uint32_t> dst, uint32_t count, uint32_t stride) {
  Fence<HardEvent::V_S>();
  uint32_t seed = count < 8 ? count : 8;
  for (uint32_t i = 0; i < seed; ++i)
    dst.SetValue(i, i * stride);
  Fence<HardEvent::S_V>();
  for (uint32_t have = 8; have < count; have *= 2) {
    uint32_t take = count - have;
    if (take > have)
      take = have;
    Adds(dst[have].ReinterpretCast<int32_t>(), dst.ReinterpretCast<int32_t>(), int32_t(have * stride),
      take);
    PipeBarrier<PIPE_V>();
  }
}

// These two helpers initialize disjoint fresh scratch tables in one S->V
// handoff. They may not replace DeviceOffsets when a previous phase still
// reads or writes the same table. Expansion can overlap an unrelated input DMA.
__aicore__ inline void SeedDeviceOffsets(LocalTensor<uint32_t> destination, uint32_t elementCount,
                    uint32_t byteStride) {
  uint32_t seedElementCount = elementCount < 8 ? elementCount : 8;
  for (uint32_t element = 0; element < seedElementCount; ++element)
    destination.SetValue(element, element * byteStride);
}
__aicore__ inline void ExpandDeviceOffsetSeeds(LocalTensor<uint32_t> destination, uint32_t elementCount,
                       uint32_t byteStride) {
  for (uint32_t prefixElements = 8; prefixElements < elementCount; prefixElements *= 2) {
    uint32_t appendedElements = elementCount - prefixElements;
    if (appendedElements > prefixElements)
      appendedElements = prefixElements;
    Adds(destination[prefixElements].ReinterpretCast<int32_t>(), destination.ReinterpretCast<int32_t>(),
      int32_t(prefixElements * byteStride), appendedElements);
    PipeBarrier<PIPE_V>();
  }
}
// An axis-parallel finite segment projects by clamping that coordinate.
// This also covers a point segment, without a divide or a rounded reconstructed endpoint.
__aicore__ inline float AxisIntervalResidual(float point, float begin, float end) {
  float low = begin < end ? begin : end, high = begin > end ? begin : end;
  return point < low ? point - low : (point > high ? point - high : 0.0f);
}
template <bool ClipEndpoints = false>
__aicore__ inline float PointSegmentSquaredDistance(float px, float py, float pz, float a, float b, float c,
                          float endX, float endY, float endZ) {
  float u = endX - a, v = endY - b, w = endZ - c;
  float ex = px - a, ey = py - b, ez = pz - c;
  if (v == 0.0f && w == 0.0f) {
    ex = AxisIntervalResidual(px, a, endX);
  } else if (u == 0.0f && w == 0.0f) {
    ey = AxisIntervalResidual(py, b, endY);
  } else if (u == 0.0f && v == 0.0f) {
    ez = AxisIntervalResidual(pz, c, endZ);
  } else {
    float squaredLength = (u * u + v * v) + w * w;
    if constexpr (ClipEndpoints) {
      float projection = (ex * u + ey * v) + ez * w;
      if (squaredLength > 0.0f && projection > 0.0f) {
        if (projection >= squaredLength) {
          ex = px - endX;
          ey = py - endY;
          ez = pz - endZ;
        } else {
          float quotient = projection / squaredLength;
          ex = px - (a + quotient * u);
          ey = py - (b + quotient * v);
          ez = pz - (c + quotient * w);
        }
      }
    } else {
      float quotient = 0.0f;
      if (squaredLength > 0.0f)
        quotient = ((ex * u + ey * v) + ez * w) / squaredLength;
      quotient = quotient < 0.0f ? 0.0f : (quotient > 1.0f ? 1.0f : quotient);
      ex = px - (a + quotient * u);
      ey = py - (b + quotient * v);
      ez = pz - (c + quotient * w);
    }
  }
  return (ex * ex + ey * ey) + ez * ez;
}

__aicore__ inline void NormalizeProjection(
    uint32_t count, float u, float v,
    float q, float l, LocalTensor<float> dot,
    LocalTensor<float> raw, LocalTensor<float> tmp, LocalTensor<float> dist,
    LocalTensor<float> next, LocalTensor<uint8_t> mask) {
  // Segment invariants use the same vector arithmetic as the dot product.
  Duplicate(next, u, 64);
  Mul(next, next, next, 64);
  Duplicate(tmp, v, 64);
  Axpy(next, tmp, v, 64);
  Duplicate(tmp, q, 64);
  Axpy(next, tmp, q, 64);
  if (l == 0.0f)
    Duplicate(next, 1.0f, 64);
  Duplicate(tmp, 1.0f, 64);
  Div(tmp, tmp, next, 64);
  Duplicate(raw[64], 1.0f, 64);
  Muls(dist, tmp, -1.0f, 64);
  MulAddDst(raw[64], dist, next, 64);
  MulAddDst(tmp, tmp, raw[64], 64);
  Copy(raw, next, uint64_t(64), uint8_t(count / 64), {1, 1, 8, 0});
  Copy(next, tmp, uint64_t(64), uint8_t(count / 64), {1, 1, 8, 0});
  Compare(mask, dot, raw, CMPMODE::LT, count);
  Mul(dot, dot, next, count);
}

__aicore__ inline void FinishSingleSegment(
    uint32_t count, float u, float v,
    float q, LocalTensor<float> ax, LocalTensor<float> ay,
    LocalTensor<float> az, LocalTensor<float> dot, LocalTensor<float> dist,
    LocalTensor<uint8_t> mask) {
  Select(dot, mask, dot, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
  Maxs(dot, dot, 0.0f, count);
  Mins(dot, dot, 1.0f, count);
  Axpy(ax, dot, -u, count);
  Axpy(ay, dot, -v, count);
  Axpy(az, dot, -q, count);
  Mul(dist, ax, ax, count);
  MulAddDst(dist, ay, ay, count);
  MulAddDst(dist, az, az, count);
}

template <typename T>
struct TreeLineAddresses {
  GlobalTensor<T> points, segments, output;
  GlobalTensor<int32_t> indicesOut;
  __aicore__ inline void Bind(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR indices) {
    points.SetGlobalBuffer((__gm__ T *)p);
    segments.SetGlobalBuffer((__gm__ T *)s);
    output.SetGlobalBuffer((__gm__ T *)o);
    indicesOut.SetGlobalBuffer((__gm__ int32_t *)indices);
  }
};

template <typename T>
class TreeLinePointKernel {
 protected:
  TreeLineAddresses<T> addresses;
  TreeLineMinDistanceTilingData t;

 public:
  __aicore__ inline void Init(GM_ADDR points, GM_ADDR segments, GM_ADDR distances, GM_ADDR indices,
                             const TreeLineMinDistanceTilingData &tiling) {
    t = tiling;
    addresses.Bind(points, segments, distances, indices);
  }
};

struct TreeLinePointRange {
  uint32_t begin, end;
};

__aicore__ inline TreeLinePointRange AssignedPoints(const TreeLineMinDistanceTilingData &metadata) {
  uint32_t total = metadata.batchCount * metadata.pointsPerBatch;
  return {total * GetBlockIdx() / metadata.workerCount, total * (GetBlockIdx() + 1) / metadata.workerCount};
}

template <typename T>
__aicore__ inline void StorePointMinima(GlobalTensor<T> output, GlobalTensor<int32_t> indicesOut, uint32_t i,
    uint32_t live, uint32_t count, LocalTensor<float> best, LocalTensor<float> next) {
  if constexpr (IsSameType<T, float>::value) {
    Fence<HardEvent::V_MTE3>();
    DataCopyPad(output[i], best, {1, live * 4, 0, 0, 0});
  } else {
    auto halfout = best.ReinterpretCast<half>();
    Cast(halfout, best, RoundMode::CAST_RINT, count);
    Fence<HardEvent::V_MTE3>();
    DataCopyPad(output[i], halfout, {1, live * 2, 0, 0, 0});
  }
  DataCopyPad(indicesOut[i], next.template ReinterpretCast<int32_t>(), {1, live * 4, 0, 0, 0});
  Fence<HardEvent::MTE3_V>();
}
