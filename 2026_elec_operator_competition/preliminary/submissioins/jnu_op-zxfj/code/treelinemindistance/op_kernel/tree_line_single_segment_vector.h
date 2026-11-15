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
#include "tree_line_vector.h"

namespace OptimizedLayout {
template <typename T, uint32_t ALGORITHM>
__aicore__ inline void KernelTreeLineMinDistance<T, ALGORITHM>::ReadSegment(uint32_t batch, bool cacheInverse, Segment &s, LocalTensor<float> next, LocalTensor<float> segmentLocal) {
    uint32_t k = batch * 6;
    s.a = float(addresses.segments.GetValue(k));
    s.b = float(addresses.segments.GetValue(k + 1));
    s.c = float(addresses.segments.GetValue(k + 2));
    s.u = float(addresses.segments.GetValue(k + 3)) - s.a;
    s.v = float(addresses.segments.GetValue(k + 4)) - s.b;
    s.w = float(addresses.segments.GetValue(k + 5)) - s.c;
    if (!cacheInverse) return;
    float squaredLength = (s.u * s.u + s.v * s.v) + s.w * s.w;
    // Preserve the original FP32 vector FMA and reciprocal correction.
    // This prefix overlaps the first point DMA; scratch is disjoint from raw input.
    Duplicate(next, s.u, 64);
    Mul(next, next, next, 64);
    Duplicate(tmp, s.v, 64);
    Axpy(next, tmp, s.v, 64);
    Duplicate(tmp, s.w, 64);
    Axpy(next, tmp, s.w, 64);
    if (squaredLength == 0.0f)
      Duplicate(next, 1.0f, 64);
    Duplicate(tmp, 1.0f, 64);
    Div(tmp, tmp, next, 64);
    Duplicate(segmentLocal, 1.0f, 64);
    Muls(dist, tmp, -1.0f, 64);
    MulAddDst(segmentLocal, dist, next, 64);
    MulAddDst(tmp, tmp, segmentLocal, 64);
    Fence<HardEvent::V_S>();
    s.squaredLength = next.GetValue(0);
    s.inverseLength = tmp.GetValue(0);
  }

template <typename T, uint32_t ALGORITHM>
__aicore__ inline void KernelTreeLineMinDistance<T, ALGORITHM>::PrefetchSinglePoints(uint32_t start, uint32_t firstPointCount, uint32_t w) {
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value)
      DataCopyPad(raw, addresses.points[start * 3], {1, firstPointCount * 12, 0, 0, 0}, {false, 0, 0, 0});
    else
      DataCopyPad(raw.ReinterpretCast<half>()[w * 6], addresses.points[start * 3],
            {1, firstPointCount * 6, 0, 0, 0}, {false, 0, 0, 0});
  }

template <typename T, uint32_t ALGORITHM>
__aicore__ inline void KernelTreeLineMinDistance<T, ALGORITHM>::GatherSinglePoints(uint32_t w, uint32_t live) {
    Fence<HardEvent::MTE2_V>();
    if constexpr (IsSameType<T, half>::value)
      Cast(raw, raw.ReinterpretCast<half>()[w * 6], RoundMode::CAST_NONE, live * 3);
    Gather(x, raw, gather, 0, live);
    Gather(y, raw, gather, 4, live);
    Gather(z, raw, gather, 8, live);
  }

template <typename T, uint32_t ALGORITHM>
__aicore__ inline void KernelTreeLineMinDistance<T, ALGORITHM>::ProjectSingleSegment(const Segment &s, bool cacheInverse, uint32_t count, LocalTensor<float> next, LocalTensor<uint8_t> mask) {
    float u = s.u, v = s.v, q = s.w;
    float l = (u * u + v * v) + q * q;
    Adds(ax, x, -s.a, count);
    Adds(ay, y, -s.b, count);
    Adds(az, z, -s.c, count);
    Muls(dot, ax, u, count);
    Axpy(dot, ay, v, count);
    Axpy(dot, az, q, count);
    if (cacheInverse) {
      CompareScalar(mask, dot, s.squaredLength, CMPMODE::LT, count);
      Muls(dot, dot, s.inverseLength, count);
    } else {
      NormalizeProjection(count, u, v, q, l, dot, raw, tmp, dist, next, mask);
    }
    FinishSingleSegment(count, u, v, q, ax, ay, az, dot, dist, mask);
  }

template <typename T, uint32_t ALGORITHM>
__aicore__ inline void KernelTreeLineMinDistance<T, ALGORITHM>::WriteSingleDistances(uint32_t i, uint32_t live, uint32_t count, LocalTensor<float> next) {
    auto best = dist;
    Sqrt(best, best, count);
    Duplicate(next.template ReinterpretCast<int32_t>(), int32_t(0), count);
    StorePointMinima(addresses.output, addresses.indicesOut, i, live, count, best, next);
  }

template <typename T, uint32_t ALGORITHM>
__aicore__ inline void KernelTreeLineMinDistance<T, ALGORITHM>::BindSinglePointPlanes(LocalTensor<float> mem, uint32_t planePitchElements) {
    raw = mem;
    x = mem[6 * planePitchElements];
    y = mem[7 * planePitchElements];
    z = mem[8 * planePitchElements];
    ax = mem[9 * planePitchElements];
    ay = mem[10 * planePitchElements];
    az = mem[11 * planePitchElements];
    dot = mem[12 * planePitchElements];
    tmp = mem[13 * planePitchElements];
    dist = mem[14 * planePitchElements];
  }

template <typename T, uint32_t ALGORITHM>
__aicore__ inline void KernelTreeLineMinDistance<T, ALGORITHM>::PointVector(uint32_t start, uint32_t end) {
    uint32_t w = (end - start + 63) / 64 * 64;
    if (w > 2048)
      w = 2048;
    // A64-float gap separates the origins of simultaneously read planes.
    // Scalar invariant caching avoids the old full-width length broadcasts.
    bool usesPaddedPointPlanes = IsSameType<T, float>::value &&
      w >= 1024 && (w % 128) == 0;
    uint32_t planePitchElements = w + (usesPaddedPointPlanes ? 64 : 0);
    LocalTensor<float> mem(TPosition::VECCALC, bufferBase,
               (t.unifiedBufferBytes - 8192 - bufferBase) / 4);
    BindSinglePointPlanes(mem, planePitchElements);
    auto next = mem[17 * planePitchElements];
    gather = mem[18 * planePitchElements].template ReinterpretCast<uint32_t>();
    auto mask = mem[19 * planePitchElements].template ReinterpretCast<uint8_t>();
    auto segmentLocal = mem[19 * planePitchElements + 256];
    bool usesScalarM1Invariants = IsSameType<T, float>::value && w >= 512;
    if (usesPaddedPointPlanes) {
      // The single-segment route has no argmin state in these two planes.
      // Its norm accumulator reads residuals from different bank origins.
      tmp = mem[14 * planePitchElements];
      dist = mem[16 * planePitchElements];
    }
    uint32_t firstPointCount = end - start;
    if (firstPointCount > w)
      firstPointCount = w;
    uint32_t firstBatchRemaining = t.pointsPerBatch - start % t.pointsPerBatch;
    if (firstPointCount > firstBatchRemaining)
      firstPointCount = firstBatchRemaining;
    // The first point DMA is independent of offset generation and the
    // segment table. Every live RGB coordinate is written by current
    // DMA/Cast before Gather, so no redundant raw reset is needed.
    // All input transfers and arithmetic stay inside this timed kernel.
    PrefetchSinglePoints(start, firstPointCount, w);
    DeviceOffsets(gather, w, 12);
    uint32_t oldBatch = ~0U;
    Segment segment{};
    for (uint32_t i = start; i < end;) {
      const uint32_t batch = i / t.pointsPerBatch;
      const uint32_t remain = (batch + 1) * t.pointsPerBatch - i;
      uint32_t live = end - i;
      if (live > w) live = w;
      if (live > remain) live = remain;
      const uint32_t count = (live + 63) / 64 * 64;
      if (usesScalarM1Invariants && oldBatch != batch) {
        ReadSegment(batch, true, segment, next, segmentLocal);
        oldBatch = batch;
      }
      if (i != start) PrefetchSinglePoints(i, live, w);
      GatherSinglePoints(w, live);
      if (!usesScalarM1Invariants) ReadSegment(batch, false, segment, next, segmentLocal);
      ProjectSingleSegment(segment, usesScalarM1Invariants, count, next, mask);
      WriteSingleDistances(i, live, count, next);
      i += live;
    }
  }
}
