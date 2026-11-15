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

namespace OptimizedLayout {
// Reuse each device-loaded segment across a tile of query points.
template <typename T, uint32_t StaticSegmentPitchElements = 0, bool DenseScratch = false> class TiledTreeLine {
  static_assert(StaticSegmentPitchElements == 0 ||
         (StaticSegmentPitchElements >= 128 && StaticSegmentPitchElements <= 1024 &&
         StaticSegmentPitchElements % 64 == 0));
  static constexpr uint32_t residualScratchElements = 4096;
  GlobalTensor<T> pg, sg, og;
  GlobalTensor<int32_t> ig;
  LocalTensor<float> ax, ay, az, dx, dy, dz, inv, raw, x, y, z, dot, tmp, group, broadcast, c[8], red,
    best, arg, pointInput, residualProductScratch, residualY, residualSquared;
  // arg uses a float view only for bit-preserving Select; its values are int32.
  LocalTensor<uint32_t> off, pointOff;
  uint32_t mp, m, n, live;
  bool usesResidualGeometry;
  __aicore__ inline void Load(LocalTensor<float> dst, GlobalTensor<T> src, uint32_t count,
                LocalTensor<float> scratch) {
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(dst, src, {1, count * 4, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
    } else {
      auto h = scratch.ReinterpretCast<half>();
      DataCopyPad(h, src, {1, count * 2, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
      Cast(dst, h, RoundMode::CAST_NONE, count);
    }
  }
  __aicore__ inline void PrefetchPointTile(uint32_t pointBegin, uint32_t pointCount) {
    DataCopyPad(pointInput.ReinterpretCast<T>(), pg[pointBegin * 3],
          {1, pointCount * uint32_t(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  }
  __aicore__ inline void StartInitialInputPrefetch(uint32_t batch, uint32_t pointBegin,
                          uint32_t pointCount) {
    Duplicate(raw, 0.0f, 6 * mp);
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value)
      DataCopyPad(raw, sg[batch * m * 6], {1, m * 24, 0, 0, 0}, {false, 0, 0, 0});
    else
      DataCopyPad(raw[6 * mp].ReinterpretCast<half>(), sg[batch * m * 6],
            {1, m * 12, 0, 0, 0}, {false, 0, 0, 0});
    PrefetchPointTile(pointBegin, pointCount);
  }
  __aicore__ inline void FinishInitialInputPrefetch() {
    Fence<HardEvent::MTE2_V>();
    if constexpr (IsSameType<T, half>::value)
      Cast(raw, raw[6 * mp].ReinterpretCast<half>(), RoundMode::CAST_NONE, m * 6);
  }
  __aicore__ inline void Segments(uint32_t batch, bool hasPrefetchedInputs = false) {
    if (!hasPrefetchedInputs) {
      Duplicate(raw, 0.0f, 6 * mp);
      Load(raw, sg[batch * m * 6], m * 6, raw[6 * mp]);
    }
    Gather(ax, raw, off, 0, mp);
    Gather(ay, raw, off, 4, mp);
    Gather(az, raw, off, 8, mp);
    Gather(dx, raw, off, 12, mp);
    Gather(dy, raw, off, 16, mp);
    Gather(dz, raw, off, 20, mp);
    Sub(dx, dx, ax, mp);
    Sub(dy, dy, ay, mp);
    Sub(dz, dz, az, mp);
    Mul(inv, dx, dx, mp);
    Mul(tmp, dy, dy, mp);
    Add(inv, inv, tmp, mp);
    Mul(tmp, dz, dz, mp);
    Add(inv, inv, tmp, mp);
    auto mask = red.ReinterpretCast<uint8_t>();
    CompareScalar(mask, inv, 0.0f, CMPMODE::GT, mp);
    Select(inv, mask, inv, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mp);
    // Zero lengths have a safe denominator; their zero direction makes t irrelevant.
    Duplicate(tmp, 1.0f, mp);
    Div(inv, tmp, inv, mp);
  }
  __aicore__ inline void Mul64(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b) {
    Mul(dst, a, b, uint64_t(64), live, {1, 1, 1, 8, 8, 0});
  }
  __aicore__ inline void ResidualSquaredDistances(uint32_t repeatCount) {
    if constexpr (DenseScratch && (StaticSegmentPitchElements == 0 || StaticSegmentPitchElements >= 256)) {
      // A dense plane has 8192 data elements. Short point slices occupy
      // at most its lower half, leaving the upper half for products.
      auto scratch = x[residualScratchElements];
      Mul(dot, x, c[3], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Mul(scratch, y, c[4], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Add(dot, dot, scratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(scratch, z, c[5], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Add(dot, dot, scratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(dot, dot, c[6], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Maxs(dot, dot, -1.0f, uint64_t(64), repeatCount, {1, 1, 8, 8});
      Mins(dot, dot, 0.0f, uint64_t(64), repeatCount, {1, 1, 8, 8});
      Mul(residualProductScratch, dot, c[3], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Sub(x, x, residualProductScratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(residualProductScratch, dot, c[4], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Sub(residualY, y, residualProductScratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(residualProductScratch, dot, c[5], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
      Sub(z, z, residualProductScratch, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      Mul(residualSquared, x, x, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      MulAddDst(residualSquared, residualY, residualY, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      MulAddDst(residualSquared, z, z, uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 8});
      return;
    }
    // A dense plane has 8192 data elements. Short point slices occupy
    // at most its lower half, leaving the upper half for products.
    auto scratch = x[residualScratchElements];
    uint32_t laneCount = repeatCount * 64;
    Mul(dot, x, c[3], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Mul(scratch, y, c[4], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Add(dot, dot, scratch, laneCount);
    Mul(scratch, z, c[5], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Add(dot, dot, scratch, laneCount);
    Mul(dot, dot, c[6], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Maxs(dot, dot, -1.0f, laneCount);
    Mins(dot, dot, 0.0f, laneCount);
    Mul(scratch, dot, c[3], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Sub(x, x, scratch, laneCount);
    Mul(scratch, dot, c[4], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Sub(y, y, scratch, laneCount);
    Mul(scratch, dot, c[5], uint64_t(64), repeatCount, {1, 1, 1, 8, 8, 0});
    Sub(z, z, scratch, laneCount);
    Mul(x, x, x, laneCount);
    MulAddDst(x, y, y, laneCount);
    MulAddDst(x, z, z, laneCount);
  }
  __aicore__ inline void MergeChunkMinima(uint32_t at) {
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(live * 2);
    GatherMask(x, red, uint8_t(1), {1, 1, 8, 1});
    GatherMask(y, red, uint8_t(2), {1, 1, 8, 1});
    SetMaskNorm();
    ResetMask();
    Adds(y.ReinterpretCast<int32_t>(), y.ReinterpretCast<int32_t>(), int32_t(at), live);
    // Keep comparison masks at a full vector width, including short point tails.
    auto mask = red[256].ReinterpretCast<uint8_t>();
    Compare(mask, x, best, CMPMODE::LT, 128);
    Min(best, best, x, 128);
    Select(arg, mask, y, arg, SELMODE::VSEL_TENSOR_TENSOR_MODE, 128);
  }
  __aicore__ inline void Chunk(uint32_t at, uint32_t count) {
    LocalTensor<float> source[7] = {ax, ay, az, dx, dy, dz, inv};
    for (uint32_t i = 0; i < 7; ++i)
      c[i] = source[i][at];

    Sub(x, c[0], broadcast, uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    Sub(y, c[1], broadcast[1024], uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    Sub(z, c[2], broadcast[2048], uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    if (usesResidualGeometry) {
      ResidualSquaredDistances(live);
    } else {
    Mul64(dot, x, c[3]);
    Mul64(x, y, c[4]);
    Add(dot, dot, x, live * 64);
    Mul64(x, z, c[5]);
    Add(dot, dot, x, live * 64);
    Mul64(dot, dot, c[6]);
    Maxs(dot, dot, -1.0f, live * 64);
    Mins(dot, dot, 0.0f, live * 64);
    Mul64(x, dot, c[3]);
    Mul64(y, dot, c[4]);
    Mul64(z, dot, c[5]);
    Sub(x, c[0], x, uint64_t(64), live, {1, 1, 1, 8, 0, 8});
    Sub(y, c[1], y, uint64_t(64), live, {1, 1, 1, 8, 0, 8});
    Sub(z, c[2], z, uint64_t(64), live, {1, 1, 1, 8, 0, 8});

    Sub(x, broadcast, x, uint64_t(64), live, {1, 0, 1, 8, 1, 8});
    Sub(y, broadcast[1024], y, uint64_t(64), live, {1, 0, 1, 8, 1, 8});
    Sub(z, broadcast[2048], z, uint64_t(64), live, {1, 0, 1, 8, 1, 8});
    Mul(x, x, x, live * 64);
    Mul(y, y, y, live * 64);
    Add(x, x, y, live * 64);
    Mul(y, z, z, live * 64);
    Add(x, x, y, live * 64);
    }
    if constexpr (DenseScratch && (StaticSegmentPitchElements == 0 || StaticSegmentPitchElements >= 256))
      WholeReduceMin(red, residualSquared, int32_t(count), live, 1, 1, 8);
    else
      WholeReduceMin(red, x, int32_t(count), live, 1, 1, 8);
    MergeChunkMinima(at);
  }
  template <uint32_t W> __aicore__ inline void LoadTailCoefficients(uint32_t at) {
    LocalTensor<float> source[7] = {ax, ay, az, dx, dy, dz, inv};
    for (uint32_t a = 0; a < 7; ++a) {
      c[a] = x[30720 + a * 64];
      Copy(c[a], source[a][at], uint64_t(W), uint8_t(64 / W), {1, 1, W / 8, 0});
    }
  }
  template <uint32_t W> __aicore__ inline void TailPacked(uint32_t at, uint32_t count) {
    LoadTailCoefficients<W>(at);
    uint32_t reps = (live + 64 / W - 1) / (64 / W), lanes = reps * 64;
    LocalTensor<float> bx = broadcast, by = broadcast[1024], bz = broadcast[2048];
    if constexpr (W > 8) {
      bx = x[4096];
      by = y[4096];
      bz = z[4096];
      uint32_t rows = reps * (64 / W);
      Copy(bx, broadcast, uint64_t(W), uint8_t(rows), {1, 0, W / 8, 1});
      Copy(by, broadcast[1024], uint64_t(W), uint8_t(rows), {1, 0, W / 8, 1});
      Copy(bz, broadcast[2048], uint64_t(W), uint8_t(rows), {1, 0, W / 8, 1});
    }
    Sub(x, c[0], bx, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(y, c[1], by, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(z, c[2], bz, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    if (usesResidualGeometry) {
      // Wider packed tails put bx in the same upper half. Its last
      // read must finish before that storage becomes product scratch.
      PipeBarrier<PIPE_V>();
      ResidualSquaredDistances(reps);
    } else {
    Mul(dot, x, c[3], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Mul(x, y, c[4], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Add(dot, dot, x, lanes);
    Mul(x, z, c[5], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Add(dot, dot, x, lanes);
    Mul(dot, dot, c[6], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Maxs(dot, dot, -1.0f, lanes);
    Mins(dot, dot, 0.0f, lanes);
    Mul(x, dot, c[3], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Mul(y, dot, c[4], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Mul(z, dot, c[5], uint64_t(64), reps, {1, 1, 1, 8, 8, 0});
    Sub(x, c[0], x, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(y, c[1], y, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(z, c[2], z, uint64_t(64), reps, {1, 1, 1, 8, 0, 8});
    Sub(x, bx, x, lanes);
    Sub(y, by, y, lanes);
    Sub(z, bz, z, lanes);
    Mul(x, x, x, lanes);
    Mul(y, y, y, lanes);
    Add(x, x, y, lanes);
    Mul(y, z, z, lanes);
    Add(x, x, y, lanes);
    }
    if constexpr (DenseScratch && (StaticSegmentPitchElements == 0 || StaticSegmentPitchElements >= 256))
      WholeReduceMin(red, residualSquared, int32_t(count), live, 1, 1, W / 8);
    else
      WholeReduceMin(red, x, int32_t(count), live, 1, 1, W / 8);
    MergeChunkMinima(at);
  }
  __aicore__ inline void UpdateScalarTailMinimum(uint32_t j, uint32_t lanes) {
    auto mask = red[256].ReinterpretCast<uint8_t>();
    Compare(mask, x, best, CMPMODE::LT, lanes);
    Duplicate(y.ReinterpretCast<int32_t>(), int32_t(j), lanes);
    Min(best, best, x, lanes);
    Select(arg, mask, y, arg, SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
  }
  __aicore__ inline void Tail(uint32_t at, uint32_t count) {
    uint32_t lanes = (live + 63) / 64 * 64;
    auto savedTmp = tmp;
    tmp = x[32512];
    Fence<HardEvent::V_S>();
    for (uint32_t j = at; j < at + count; ++j) {
      float a = ax.GetValue(j), b = ay.GetValue(j), c0 = az.GetValue(j), u = dx.GetValue(j),
         v = dy.GetValue(j), w = dz.GetValue(j), r = inv.GetValue(j);
      Adds(x, group, -a, lanes);
      Adds(y, group[128], -b, lanes);
      Adds(z, group[256], -c0, lanes);
      Muls(dot, x, u, lanes);
      Muls(tmp, y, v, lanes);
      Add(dot, dot, tmp, lanes);
      Muls(tmp, z, w, lanes);
      Add(dot, dot, tmp, lanes);
      Muls(dot, dot, r, lanes);
      Maxs(dot, dot, 0.0f, lanes);
      Mins(dot, dot, 1.0f, lanes);
      if (usesResidualGeometry) {
        // Match the residual formula and sum order of full chunks,
        // including ties between a full chunk and this scalar tail.
        Muls(tmp, dot, u, lanes);
        Sub(x, x, tmp, lanes);
        Muls(tmp, dot, v, lanes);
        Sub(y, y, tmp, lanes);
        Muls(tmp, dot, w, lanes);
        Sub(z, z, tmp, lanes);
        Mul(x, x, x, lanes);
        MulAddDst(x, y, y, lanes);
        MulAddDst(x, z, z, lanes);
      } else {
      Muls(x, dot, u, lanes);
      Muls(y, dot, v, lanes);
      Muls(z, dot, w, lanes);
      Adds(x, x, a, lanes);
      Adds(y, y, b, lanes);
      Adds(z, z, c0, lanes);
      Sub(x, group, x, lanes);
      Sub(y, group[128], y, lanes);
      Sub(z, group[256], z, lanes);
      Mul(x, x, x, lanes);
      Mul(tmp, y, y, lanes);
      Add(x, x, tmp, lanes);
      Mul(tmp, z, z, lanes);
      Add(x, x, tmp, lanes);
      }
      UpdateScalarTailMinimum(j, lanes);
    }
    tmp = savedTmp;
  }

  __aicore__ inline void BindTiledPlanes(const TreeLineMinDistanceTilingData &t) {
    m = t.segmentCount;
    n = t.pointsPerBatch;
    // Template pitch changes addresses only; actual m still bounds all loads and reductions.
    mp = StaticSegmentPitchElements ? StaticSegmentPitchElements : (m + 63) / 64 * 64;
    LocalTensor<float> mem(TPosition::VECCALC, 0, (t.unifiedBufferBytes - 8192) / 4);
    ax = mem;
    ay = mem[mp];
    az = mem[2 * mp];
    dx = mem[3 * mp];
    dy = mem[4 * mp];
    dz = mem[5 * mp];
    inv = mem[6 * mp];
    off = mem[7 * mp].ReinterpretCast<uint32_t>();
    // Separate the four dense planes by 256 bytes to vary their UB bank origins.
    raw = mem[8 * mp];
    x = raw;
    y = x[8256];
    z = x[16512];
    dot = x[24768];
    tmp = dot;
    group = x[33024];
    red = x[33408];
    best = x[33920];
    arg = x[34048];
    pointOff = x[34176].ReinterpretCast<uint32_t>();
    broadcast = x[34432];
    // The staging tile lies beyond all four geometry planes and output
    // state. Host reserves these 384 FP32 elements in the UB budget.
    pointInput = x[37504];
  }

  __aicore__ inline void PrefetchFirstTile(uint32_t start, uint32_t end) {
    uint32_t firstPointCount = end - start;
    if (firstPointCount > 128)
      firstPointCount = 128;
    uint32_t firstBatch = start / n;
    if (firstPointCount > (firstBatch + 1) * n - start)
      firstPointCount = (firstBatch + 1) * n - start;
    StartInitialInputPrefetch(firstBatch, start, firstPointCount);
    // These two scratch tables are disjoint and fresh. Share their
    // scalar-to-vector handoff and retain segment offsets across batches.
    SeedDeviceOffsets(pointOff, 128, 12);
    SeedDeviceOffsets(off, mp, 24);
    Fence<HardEvent::S_V>();
    ExpandDeviceOffsetSeeds(pointOff, 128, 12);
    ExpandDeviceOffsetSeeds(off, mp, 24);
  }

  __aicore__ inline void ChooseResidualScratch() {
    if constexpr (DenseScratch && (StaticSegmentPitchElements == 0 || StaticSegmentPitchElements >= 256)) {
      // Select once for this complete point tile; full and packed
      // chunks share the same physical squared-norm destination.
      residualProductScratch = x[residualScratchElements];
      residualY = y;
      residualSquared = x;
      if (usesResidualGeometry && live >= 32 && mp >= 256) {
        // Projection uses the lower dot half. Packed tails can
        // overwrite only dead anchors below direction staging.
        residualProductScratch = dot[residualScratchElements];
        residualY = x[residualScratchElements];
        residualSquared = y;
      }
    }
  }

  __aicore__ inline void GatherNextTile(uint32_t at, uint32_t start, uint32_t end) {
    Duplicate(group, 0.0f, 384);
    if (at != start)
      Fence<HardEvent::MTE2_V>();
    auto pointCoordinates = pointInput;
    if constexpr (IsSameType<T, half>::value) {
      Cast(raw, pointInput.ReinterpretCast<half>(), RoundMode::CAST_NONE, live * 3);
      pointCoordinates = raw;
    }
    Gather(group, pointCoordinates, pointOff, 0, live);
    Gather(group[128], pointCoordinates, pointOff, 4, live);
    Gather(group[256], pointCoordinates, pointOff, 8, live);
    uint32_t nextPoint = at + live;
    if (nextPoint < end) {
      uint32_t nextPointCount = end - nextPoint;
      if (nextPointCount > 128)
        nextPointCount = 128;
      uint32_t nextBatchEnd = (nextPoint / n + 1) * n;
      if (nextPointCount > nextBatchEnd - nextPoint)
        nextPointCount = nextBatchEnd - nextPoint;
      Fence<HardEvent::V_MTE2>();
      PrefetchPointTile(nextPoint, nextPointCount);
    }
  }

  __aicore__ inline void SearchSegments() {
    Brcb(broadcast, group, uint8_t((live + 7) / 8), {1, 8});
    Brcb(broadcast[1024], group[128], uint8_t((live + 7) / 8), {1, 8});
    Brcb(broadcast[2048], group[256], uint8_t((live + 7) / 8), {1, 8});
    Duplicate(best, 3.402823466e38f, 128);
    Duplicate(arg.ReinterpretCast<int32_t>(), int32_t(0), 128);
    for (uint32_t j = 0; j < m; j += 64) {
      uint32_t count = m - j;
      if (count > 64)
        count = 64;
      if (count <= 3 && live >= 32)
        Tail(j, count);
      else if (count <= 32 && live >= 8) {
        // A short point slice packs a small segment tail into
        // full vectors, avoiding a 64-segment row for one tail.
        if (count <= 8)
          TailPacked<8>(j, count);
        else if (count <= 16)
          TailPacked<16>(j, count);
        else
          TailPacked<32>(j, count);
      } else
        Chunk(j, count);
    }
  }

  __aicore__ inline void StoreBestTile(uint32_t at) {
    Sqrt(best, best, live);
    if constexpr (IsSameType<T, float>::value) {
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(og[at], best, {1, live * 4, 0, 0, 0});
    } else {
      auto h = best.ReinterpretCast<half>();
      Cast(h, best, RoundMode::CAST_RINT, live);
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(og[at], h, {1, live * 2, 0, 0, 0});
    }
    DataCopyPad(ig[at], arg.ReinterpretCast<int32_t>(), {1, live * 4, 0, 0, 0});
    Fence<HardEvent::MTE3_V>();
  }

 public:
  __aicore__ inline void Run(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR oi,
               const TreeLineMinDistanceTilingData &t) {
    pg.SetGlobalBuffer((__gm__ T *)p);
    sg.SetGlobalBuffer((__gm__ T *)s);
    og.SetGlobalBuffer((__gm__ T *)o);
    ig.SetGlobalBuffer((__gm__ int32_t *)oi);
    BindTiledPlanes(t);
    uint32_t total = t.batchCount * n, start = total * GetBlockIdx() / t.workerCount,
        end = total * (GetBlockIdx() + 1) / t.workerCount, oldBatch = 0xffffffff;
    usesResidualGeometry = uint64_t(end - start) * 64 <= residualScratchElements;
    PrefetchFirstTile(start, end);

    for (uint32_t at = start; at < end;) {
      uint32_t batch = at / n;
      if (batch != oldBatch) {
        if (at == start)
          FinishInitialInputPrefetch();
        Segments(batch, at == start);
        oldBatch = batch;
      }
      live = end - at;
      if (live > 128)
        live = 128;
      if (live > (batch + 1) * n - at)
        live = (batch + 1) * n - at;
      ChooseResidualScratch();
      GatherNextTile(at, start, end);
      SearchSegments();
      StoreBestTile(at);
      at += live;
    }
  }
};
} // namespace OptimizedLayout
