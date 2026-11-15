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

// Device-side spatial ordering and conservative segment-box lower bounds.
template <typename T> class SpatialTreeLine {
  // The first 64 state elements hold the ends of these cells. Geometry uses
  // 64 point rows; a 32-point target balances cell bounds and padded chunks.
  static constexpr uint32_t maximumSpatialCellCount = 64;
  static constexpr uint32_t minimumSpatialCellPointCount = 32;
  static constexpr uint32_t coefficientPlaneCount = 7, coefficientLanesPerPlane = 64;
  static constexpr uint32_t coefficientOffsetElements = coefficientPlaneCount * coefficientLanesPerPlane;
  static constexpr uint32_t coefficientScratchElements = coefficientOffsetElements * 2;
  static constexpr uint32_t queryPairStateElements = 4 * 4160 + 2688;
  // Reuse the available UB budget for segment invariants.
  static constexpr uint32_t segmentStatePlaneCount = 22;
  LocalTensor<float> absoluteSegmentDirection[3];
  bool usePackedCoefficientGather;
  TPipe pipe;
  TBuf<TPosition::VECCALC> storage;
  GlobalTensor<T> pg, sg, og;
  GlobalTensor<float> coordG;
  GlobalTensor<int32_t> ig, orderG;
  LocalTensor<float> out, order, pool, raw, lo[3], hi[3], inv, recip, sorted, sortTmp, key, seq, red,
    group, perpendicularInverse[3];
  LocalTensor<float> x, y, z, dot, tmp, dist, c[8], best, broadcast;
  LocalTensor<int32_t> outIdx;
  LocalTensor<float> bestArg;
  uint32_t m, n, start, batch, mp, k, live, cellShift, cellCount, rankCount;
  LocalTensor<int32_t> cellEnd;
  float scaleBound, firstScore;
  bool keepPoints, outputPending;
  uint32_t pointPitch;
  LocalTensor<float> localPoints;
  LocalTensor<int32_t> localOrder;
  LocalTensor<uint32_t> pointOffsets, coefficientSourceByteOffsets, coefficientPlaneByteOffsets;
  __aicore__ inline uint32_t Spread(uint32_t v) {
    v = (v | (v << 16)) & 0x030000ffU;
    v = (v | (v << 8)) & 0x0300f00fU;
    v = (v | (v << 4)) & 0x030c30c3U;
    v = (v | (v << 2)) & 0x09249249U;
    return v;
  }
  __aicore__ inline void Load(LocalTensor<float> dst, GlobalTensor<T> src, uint32_t count,
                LocalTensor<float> halfScratch) {
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(dst, src, {1, count * 4, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
    } else {
      auto h = halfScratch.ReinterpretCast<half>();
      DataCopyPad(h, src, {1, count * 2, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
      Cast(dst, h, RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
    }
  }
  __aicore__ inline void SpreadMortonAxis(LocalTensor<uint32_t> bits, LocalTensor<uint32_t> shifted, LocalTensor<uint32_t> bitsMask);
  __aicore__ inline void BuildMortonKeys(LocalTensor<float> keys, LocalTensor<float> idxf, LocalTensor<float> sortedPoints, LocalTensor<float> scratch, LocalTensor<float> praw);

  __aicore__ inline void FindCellEnds(LocalTensor<float> keys);

  __aicore__ inline void PointSort();
  __aicore__ inline void BindSegmentPlanes() {
    for (int a = 0; a < 3; ++a) {
      lo[a] = pool[a * mp];
      hi[a] = pool[(a + 3) * mp];
    }
    inv = pool[6 * mp];
    raw = pool[7 * mp];
    // A, D and the reciprocal occupy seven adjacent source planes.
    // The compacted index list and bound keys stay disjoint from them.
    recip = pool[13 * mp];
    sorted = pool[14 * mp];
    key = pool[15 * mp];
    seq = pool[16 * mp];
    perpendicularInverse[0] = inv;
    perpendicularInverse[1] = pool[17 * mp];
    perpendicularInverse[2] = pool[18 * mp];
    for (uint32_t axis = 0; axis < 3; ++axis)
      absoluteSegmentDirection[axis] = pool[(19 + axis) * mp];
    sortTmp = pool[segmentStatePlaneCount * mp];
  }

  __aicore__ inline void LoadSegmentGeometry(LocalTensor<float> work) {
    auto input = work, endpoint = work[9 * mp], square = work[10 * mp];
    Duplicate(input, 0.0f, 6 * mp);
    PipeBarrier<PIPE_V>();
    Load(input, sg[batch * m * 6], m * 6, work[6 * mp]);
    ReduceMin(red, input, square, m * 6, false);
    PipeBarrier<PIPE_V>();
    ReduceMax(red[8], input, square, m * 6, false);
    Fence<HardEvent::V_S>();
    float av = red.GetValue(0), bv = red.GetValue(8);
    scaleBound = av * av > bv * bv ? av * av : bv * bv;
    Fence<HardEvent::S_V>();
    // Indices remain integer bits through compaction, selection and workspace.
    // Float views below serve bit-preserving data-movement APIs only.
    DeviceOffsets(seq.ReinterpretCast<uint32_t>(), mp, 1);
    Muls(key.ReinterpretCast<int32_t>(), seq.ReinterpretCast<int32_t>(), int32_t(24), mp);
    auto offsets = key.ReinterpretCast<uint32_t>();
    for (uint32_t axis = 0; axis < 3; ++axis) {
      Gather(raw[axis * mp], input, offsets, axis * 4, mp);
      Gather(endpoint, input, offsets, (axis + 3) * 4, mp);
      Min(lo[axis], raw[axis * mp], endpoint, mp);
      Max(hi[axis], raw[axis * mp], endpoint, mp);
      Sub(raw[(axis + 3) * mp], endpoint, raw[axis * mp], mp);
      Mul(square, raw[(axis + 3) * mp], raw[(axis + 3) * mp], mp);
      if (axis == 0)
        Adds(inv, square, 0.0f, mp);
      else
        Add(inv, inv, square, mp);
    }
    auto mask = red.ReinterpretCast<uint8_t>();
    CompareScalar(mask, inv, 0.0f, CMPMODE::GT, mp);
    Select(inv, mask, inv, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mp);
    Duplicate(recip, 1.0f, mp);
    Div(recip, recip, inv, mp);
    // The three cross-product axes are perpendicular to the segment.
    // A floor only weakens the distance bound and avoids division by zero
    // for a parallel or point segment; it never changes exact geometry.
    if (m >= kSatMinimumSegmentCount) {
      for (uint32_t axis = 0; axis < 3; ++axis)
        Abs(absoluteSegmentDirection[axis], raw[(axis + 3) * mp], mp);
      for (uint32_t axis = 0; axis < 3; ++axis) {
        uint32_t other = (axis + 1) % 3, last = (axis + 2) % 3;
        Mul(endpoint, raw[(other + 3) * mp], raw[(other + 3) * mp], mp);
        Mul(square, raw[(last + 3) * mp], raw[(last + 3) * mp], mp);
        Add(endpoint, endpoint, square, mp);
        Maxs(endpoint, endpoint, 1.0e-20f, mp);
        Duplicate(square, 1.0f, mp);
        Div(perpendicularInverse[axis], square, endpoint, mp);
      }
    }
  }

  __aicore__ inline void Segments() {
    BindSegmentPlanes();
    auto work = pool[segmentStatePlaneCount * mp + 64];
    x = work;
    y = work[4160];
    z = work[8320];
    dot = work[12480];
    // Rank uses a short temporary slice in the dot plane. Chunk reuses x
    // after its first product and dot after the projection has been consumed.
    tmp = dot[mp];
    dist = dot;
    for (int axis = 0; axis < 8; ++axis)
      c[axis] = work[16640 + axis * 64];
    red = work[17152];
    group = work[17408];
    best = work[17600];
    bestArg = work[17664];
    broadcast = work[17792];
    LoadSegmentGeometry(work);
    pointOffsets = work[17728].ReinterpretCast<uint32_t>();
    if (keepPoints)
      DeviceOffsets(pointOffsets, 64, 4);
    // A short candidate chunk overwrites only its valid coefficient lanes.
    // Define every inactive lane once so later full-width geometry and
    // scratch reuse never consume uninitialized coefficient memory.
    Duplicate(c[0], 0.0f, 7 * 64);
    if (usePackedCoefficientGather) {
      coefficientSourceByteOffsets = work[queryPairStateElements].ReinterpretCast<uint32_t>();
      coefficientPlaneByteOffsets =
        work[queryPairStateElements + coefficientOffsetElements].ReinterpretCast<uint32_t>();
      // Define inactive offsets too; the masked gather only consumes live
      // columns, and later chunks can safely retain their previous offsets.
      Duplicate(coefficientSourceByteOffsets, uint32_t(0), coefficientOffsetElements);
      for (uint32_t plane = 0; plane < coefficientPlaneCount; ++plane)
        Duplicate(coefficientPlaneByteOffsets[plane * coefficientLanesPerPlane],
             plane * mp * uint32_t(sizeof(float)), coefficientLanesPerPlane);
    }
  }
  __aicore__ inline void ReadPointBounds(uint32_t at, float *mn, float *mx);

  __aicore__ inline void RankSeparatingAxes(const float *mn, const float *mx);

  __aicore__ inline void RankProjection(const float *mn, const float *mx);

  __aicore__ inline void Rank(uint32_t at);
  __aicore__ inline uint32_t Compress(float threshold, bool second);
  __aicore__ inline void BroadcastMul(LocalTensor<float> dst, LocalTensor<float> a,
                    LocalTensor<float> b) {
    Mul(dst, a, b, uint64_t(64), live, {1, 1, 1, 8, 8, 0});
  }
  __aicore__ inline void UpdateClosestRows(bool initializeBest) {
    // Every live row is overwritten below. The first 64 scratch
    // lanes were all defined by geometry (live>=1); inactive row values
    // never enter the live-row bound reduction or the output DMA.
    // ORDER_VALUE_INDEX stores tightly packed (float, uint32) pairs.
    // Fixed patterns 1/2 select their even/odd elements without indexed
    // loads or a reserved-count read; both outputs contain live rows.
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(live * 2);
    GatherMask(x, red, uint8_t(1), {1, 1, 8, 1});
    GatherMask(y, red, uint8_t(2), {1, 1, 8, 1});
    SetMaskNorm();
    ResetMask();
    ShiftLeft(y.ReinterpretCast<uint32_t>(), y.ReinterpretCast<uint32_t>(), uint32_t(2), 64);
    Gather(z, c[7], y.ReinterpretCast<uint32_t>(), 0, live);
    if (initializeBest) {
      // The seed/fallback path always supplies a nonempty list. Its first chunk
      // defines both best arrays before refinement can read them.
      // Geometry defines all first 64 scratch lanes even for live=1;
      // inactive point rows never enter the live output or bound reduction.
      Adds(best.ReinterpretCast<int32_t>(), x.ReinterpretCast<int32_t>(), int32_t(0), 64);
      Adds(bestArg.ReinterpretCast<int32_t>(), z.ReinterpretCast<int32_t>(), int32_t(0), 64);
    } else {
      auto lt = red[128].ReinterpretCast<uint8_t>(), eq = red[160].ReinterpretCast<uint8_t>();
      Compare(lt, x, best, CMPMODE::LT, 64);
      Compare(eq, x, best, CMPMODE::EQ, 64);
      Min(y.ReinterpretCast<int32_t>(), z.ReinterpretCast<int32_t>(),
        bestArg.ReinterpretCast<int32_t>(), 64);
      Select(bestArg, lt, z, bestArg, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
      Min(best, best, x, 64);
      Select(bestArg, eq, y, bestArg, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64);
    }
    tmp = dot[mp];
  }
  __aicore__ inline void Chunk(uint32_t pos, uint32_t count, bool initializeBest = false) {
    tmp = x;
    // The compacted list stays read-only until both geometry passes finish.
    // Alias this chunk instead of copying and converting its 64 indices.
    c[7] = sorted[pos];
    Muls(tmp.ReinterpretCast<int32_t>(), c[7].ReinterpretCast<int32_t>(), 4, count);
    if (usePackedCoefficientGather) {
      // Each row adds one source plane's byte origin to the same indices.
      Add(coefficientSourceByteOffsets.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(),
        coefficientPlaneByteOffsets.ReinterpretCast<int32_t>(), uint64_t(count),
        uint8_t(coefficientPlaneCount), {1, 1, 1, 8, 0, 8});
      Gather(c[0], raw, coefficientSourceByteOffsets, 0, uint64_t(count),
         uint8_t(coefficientPlaneCount), uint16_t(coefficientLanesPerPlane / 8));
      ResetMask();
    } else {
      // Preserve on-chip points when the extra offset tables would
      // otherwise displace them. The original gathers need no scratch.
      auto offsets = tmp.ReinterpretCast<uint32_t>();
      for (uint32_t axis = 0; axis < 6; ++axis)
        Gather(c[axis], raw[axis * mp], offsets, 0, count);
      Gather(c[6], recip, offsets, 0, count);
    }

    Sub(x, c[0], broadcast, uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    Sub(y, c[1], broadcast[512], uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    Sub(z, c[2], broadcast[1024], uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    BroadcastMul(dot, x, c[3]);
    BroadcastMul(tmp, y, c[4]);
    uint32_t q = live * 64;
    Add(dot, dot, tmp, q);
    BroadcastMul(tmp, z, c[5]);
    Add(dot, dot, tmp, q);
    BroadcastMul(dot, dot, c[6]);
    Maxs(dot, dot, -1.0f, q);
    Mins(dot, dot, 0.0f, q);
    BroadcastMul(x, dot, c[4]);
    Sub(y, y, x, q);
    BroadcastMul(x, dot, c[5]);
    Sub(z, z, x, q);
    BroadcastMul(x, dot, c[3]);
    Sub(dot, c[0], broadcast, uint64_t(64), live, {1, 1, 0, 8, 0, 1});
    Sub(x, dot, x, q);
    Mul(dist, x, x, q);
    if constexpr (IsSameType<T, half>::value) {
      // Inputs were converted to Float; accumulation retains Float range.
      MulAddDst(dist, y, y, q);
      MulAddDst(dist, z, z, q);
    } else {
      Mul(x, y, y, q);
      Add(dist, dist, x, q);
      Mul(x, z, z, q);
      Add(dist, dist, x, q);
    }
    WholeReduceMin(red, dist, int32_t(count), live, 1, 1, 8);
    UpdateClosestRows(initializeBest);
  }
  template <uint32_t segmentLanes>
  __aicore__ inline void GatherCandidateCoefficients(uint32_t pos, uint32_t count) {
    // Replicate a short index row before the single coefficient gather.
    // Every padded index is a legal zero index; min masks exclude padded
    // candidates. Integer Copy preserves the byte offsets exactly.
    tmp = x;
    c[7] = sorted[pos];
    Duplicate(tmp.ReinterpretCast<int32_t>(), int32_t(0), 64);
    Muls(tmp.ReinterpretCast<int32_t>(), c[7].ReinterpretCast<int32_t>(), int32_t(4), count);
    Copy(tmp[segmentLanes].ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(),
      uint64_t(segmentLanes), uint8_t(64 / segmentLanes - 1), {1, 1, segmentLanes / 8, 0});
    Add(coefficientSourceByteOffsets.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(),
      coefficientPlaneByteOffsets.ReinterpretCast<int32_t>(), uint64_t(64),
      uint8_t(coefficientPlaneCount), {1, 1, 1, 8, 0, 8});
    Gather(c[0], raw, coefficientSourceByteOffsets, 0, coefficientOffsetElements);
  }
  __aicore__ inline void AccumulateCandidateSquares(uint32_t activeLanes) {
    Mul(dist, x, x, activeLanes);
    if constexpr (IsSameType<T, half>::value) {
      // Inputs were converted to Float; accumulation retains Float range.
      MulAddDst(dist, y, y, activeLanes);
      MulAddDst(dist, z, z, activeLanes);
    } else {
      Mul(x, y, y, activeLanes);
      Add(dist, dist, x, activeLanes);
      Mul(x, z, z, activeLanes);
      Add(dist, dist, x, activeLanes);
    }
  }
  template <uint32_t segmentLanes>
  __aicore__ inline void PackedCandidateChunk(uint32_t pos, uint32_t count, bool initializeBest) {
    static_assert(segmentLanes == 8 || segmentLanes == 16 || segmentLanes == 32);
    GatherCandidateCoefficients<segmentLanes>(pos, count);
    constexpr uint32_t pointRowsPerVector = 64 / segmentLanes;
    uint32_t vectorRepeats = (live + pointRowsPerVector - 1) / pointRowsPerVector;
    uint32_t activeLanes = vectorRepeats * 64;
    auto bx = broadcast, by = broadcast[512], bz = broadcast[1024];
    if constexpr (segmentLanes > 8) {
      // Geometry occupies at most 2048 floats in each 4160-float plane.
      // The upper half stages expanded point rows until all residuals finish.
      bx = x[2048];
      by = y[2048];
      bz = z[2048];
      uint32_t paddedPointRows = vectorRepeats * pointRowsPerVector;
      Copy(bx, broadcast, uint64_t(segmentLanes), uint8_t(paddedPointRows),
        {1, 0, segmentLanes / 8, 1});
      Copy(by, broadcast[512], uint64_t(segmentLanes), uint8_t(paddedPointRows),
        {1, 0, segmentLanes / 8, 1});
      Copy(bz, broadcast[1024], uint64_t(segmentLanes), uint8_t(paddedPointRows),
        {1, 0, segmentLanes / 8, 1});
    }
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams sharedCoefficient{1, 1, 1, 8, 8, 0};
    Sub(x, c[0], bx, uint64_t(64), vectorRepeats, {1, 1, 1, 8, 0, 8});
    Sub(y, c[1], by, uint64_t(64), vectorRepeats, {1, 1, 1, 8, 0, 8});
    Sub(z, c[2], bz, uint64_t(64), vectorRepeats, {1, 1, 1, 8, 0, 8});
    Mul(dot, x, c[3], uint64_t(64), vectorRepeats, sharedCoefficient);
    Mul(x, y, c[4], uint64_t(64), vectorRepeats, sharedCoefficient);
    Add(dot, dot, x, activeLanes);
    Mul(x, z, c[5], uint64_t(64), vectorRepeats, sharedCoefficient);
    Add(dot, dot, x, activeLanes);
    Mul(dot, dot, c[6], uint64_t(64), vectorRepeats, sharedCoefficient);
    Maxs(dot, dot, -1.0f, activeLanes);
    Mins(dot, dot, 0.0f, activeLanes);
    Mul(x, dot, c[4], uint64_t(64), vectorRepeats, sharedCoefficient);
    Sub(y, y, x, activeLanes);
    Mul(x, dot, c[5], uint64_t(64), vectorRepeats, sharedCoefficient);
    Sub(z, z, x, activeLanes);
    Mul(x, dot, c[3], uint64_t(64), vectorRepeats, sharedCoefficient);
    Sub(dot, c[0], bx, uint64_t(64), vectorRepeats, {1, 1, 1, 8, 0, 8});
    Sub(x, dot, x, activeLanes);
    AccumulateCandidateSquares(activeLanes);
    WholeReduceMin(red, dist, int32_t(count), live, 1, 1, segmentLanes / 8);
    UpdateClosestRows(initializeBest);
  }
  __aicore__ inline void ComputeCandidateChunk(uint32_t pos, uint32_t count,
                        bool initializeBest = false) {
    // The capacity fallback retains the original full-width geometry.
    if (!usePackedCoefficientGather || count > 32)
      Chunk(pos, count, initializeBest);
    else if (count <= 8)
      PackedCandidateChunk<8>(pos, count, initializeBest);
    else if (count <= 16)
      PackedCandidateChunk<16>(pos, count, initializeBest);
    else
      PackedCandidateChunk<32>(pos, count, initializeBest);
  }
  __aicore__ inline void Restore() {
    PipeBarrier<PIPE_ALL>();
    Fence<HardEvent::V_S>();
    Fence<HardEvent::MTE3_S>();
    auto keys = pool, rank = pool[k], sortOut = pool[k * 2], scratch = pool[k * 4];
    if (keepPoints) {
      Adds(keys.ReinterpretCast<int32_t>(), localOrder, int32_t(0), n);
      PipeBarrier<PIPE_V>();
    } else {
      Fence<HardEvent::V_MTE2>();
      DataCopyPad(keys.ReinterpretCast<int32_t>(), orderG[start], {1, n * 4, 0, 0, 0},
            {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
    }
    Cast(keys, keys.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, n);
    Muls(keys, keys, -1.0f, n);
    uint32_t aligned = (n + 7) / 8 * 8;
    if (aligned < k)
      Duplicate(keys[aligned], -3.402823466e38f, k - aligned);
    Fence<HardEvent::V_S>();
    for (uint32_t i = n; i < aligned; ++i)
      keys.SetValue(i, -3.402823466e38f);
    Fence<HardEvent::S_V>();
    DeviceOffsets(rank.ReinterpretCast<uint32_t>(), k, 1);
    Sort<float, true>(sortOut, keys, rank.ReinterpretCast<uint32_t>(), scratch, k / 32);
    Extract(keys, rank.ReinterpretCast<uint32_t>(), sortOut, k / 32);
    Muls(rank.ReinterpretCast<int32_t>(), rank.ReinterpretCast<int32_t>(), 4, n);
    Fence<HardEvent::V_MTE2>();
    DataCopyPad(sortOut, coordG[start * 3], {1, n * 4, 0, 0, 0}, {false, 0, 0, 0});
    DataCopyPad(sortOut[k], coordG[start * 3 + n], {1, n * 4, 0, 0, 0}, {false, 0, 0, 0});
    Fence<HardEvent::MTE2_V>();
    Gather(out, sortOut, rank.ReinterpretCast<uint32_t>(), 0, n);
    Gather(outIdx.ReinterpretCast<float>(), sortOut[k], rank.ReinterpretCast<uint32_t>(), 0, n);
    // Workspace stores raw int32 bits; Gather already produced the output type.
  }

  __aicore__ inline void BindSpatialWorkspace(LocalTensor<float> mem, uint32_t pitch, const TreeLineMinDistanceTilingData &t) {
    out = mem;
    outIdx = mem[pitch].ReinterpretCast<int32_t>();
    order = mem[pitch * 2];
    pool = mem[pitch * 2 + k];
    pointPitch = pitch;
    uint32_t stateSize = 3 * pitch + k, userSize = (t.unifiedBufferBytes - 8192) / 4;
    uint32_t initialSize = 2 * pitch + 7 * k + 3 * n,
        querySize = 64 + segmentStatePlaneCount * mp + 64 + queryPairStateElements;
    keepPoints = stateSize < userSize && initialSize <= userSize - stateSize &&
          querySize <= userSize - stateSize;
    usePackedCoefficientGather =
      querySize + coefficientScratchElements <= userSize &&
      (!keepPoints || querySize + coefficientScratchElements <= userSize - stateSize);
    if (keepPoints) {
      localPoints = mem[userSize - stateSize];
      localOrder = localPoints[3 * pitch].ReinterpretCast<int32_t>();
    }
    cellShift = 30;
    // Each additional Morton prefix bit bisects a cell. Growing by one bit
    // avoids an eightfold jump in bound/reduction setup work while targeting
    // at least thirty-two points per cell and at most sixty-four cells.
    for (uint32_t spatialCellCount = 1; spatialCellCount < maximumSpatialCellCount &&
                      spatialCellCount * 2 <= n / minimumSpatialCellPointCount;
      spatialCellCount *= 2)
      --cellShift;
  }

  __aicore__ inline void RefineCandidates() {
    firstScore = 0;
    uint32_t initial = Compress(firstScore, false);
    if (!initial) {
      ReduceMax(red, key, red[64], mp, false);
      Fence<HardEvent::V_S>();
      firstScore = red.GetValue(0);
      Fence<HardEvent::S_V>();
      initial = Compress(firstScore, false);
    }
    if (!initial) {
      // Exceptional bound arithmetic still evaluates every segment;
      // best state is always defined by an actual nonempty chunk.
      DeviceOffsets(sorted.ReinterpretCast<uint32_t>(), m, 1);
      initial = m;
    }
    for (uint32_t pos = 0; pos < initial; pos += 64) {
      uint32_t count = initial - pos;
      if (count > 64)
        count = 64;
      ComputeCandidateChunk(pos, count, pos == 0);
    }
    WholeReduceMax(red, best, int32_t(live), 1, 1, 1, 8);
    Fence<HardEvent::V_S>();
    float bound = red.GetValue(0);
    Fence<HardEvent::S_V>();
    uint32_t remaining = Compress(bound + 0.000032f * (scaleBound + bound + 1.0f), true);
    for (uint32_t pos = 0; pos < remaining; pos += 64) {
      uint32_t count = remaining - pos;
      if (count > 64)
        count = 64;
      ComputeCandidateChunk(pos, count);
    }
  }

  __aicore__ inline void WriteSpatialResult(LocalTensor<float> mem, uint32_t pitch) {
    if (outputPending) {
      WaitFlag<HardEvent::MTE3_V>(0);
      outputPending = false;
    }
    pool = mem[pitch * 2];
    Restore();
    PipeBarrier<PIPE_V>();
    Sqrt(out, out, n);
    if constexpr (IsSameType<T, float>::value) {
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(og[start], out, {1, n * 4, 0, 0, 0});
    } else {
      auto h = out.ReinterpretCast<half>();
      Cast(h, out, RoundMode::CAST_RINT, n);
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(og[start], h, {1, n * 2, 0, 0, 0});
    }
    DataCopyPad(ig[start], outIdx, {1, n * 4, 0, 0, 0});
    Fence<HardEvent::MTE3_S>();
  }

  __aicore__ inline void SearchSpatialCells() {
    uint32_t cell = 0;
    for (uint32_t at = 0; at < n;) {
      uint32_t limit = uint32_t(cellEnd.GetValue(cell));
      live = limit - at;
      if (live > 64)
        live = 64;
      Rank(at);
      RefineCandidates();
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(coordG[start * 3 + at], best, {1, live * 4, 0, 0, 0});
      DataCopyPad(coordG[start * 3 + n + at], bestArg, {1, live * 4, 0, 0, 0});
      SetFlag<HardEvent::MTE3_V>(0);
      outputPending = true;
      at += live;
      if (at == limit)
        ++cell;
    }
  }
 public:
  __aicore__ inline void Run(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR idx, GM_ADDR workspace,
               const TreeLineMinDistanceTilingData &t) {
    outputPending = false;
    coordG.SetGlobalBuffer((__gm__ float *)workspace);
    orderG.SetGlobalBuffer((__gm__ int32_t *)workspace + t.batchCount * t.pointsPerBatch * 3);
    pg.SetGlobalBuffer((__gm__ T *)p);
    sg.SetGlobalBuffer((__gm__ T *)s);
    og.SetGlobalBuffer((__gm__ T *)o);
    ig.SetGlobalBuffer((__gm__ int32_t *)idx);
    batch = GetBlockIdx() / t.workersPerBatch;
    uint32_t worker = GetBlockIdx() % t.workersPerBatch;
    uint32_t begin = t.pointsPerBatch * worker / t.workersPerBatch,
        end = t.pointsPerBatch * (worker + 1) / t.workersPerBatch;
    n = end - begin;
    start = batch * t.pointsPerBatch + begin;
    m = t.segmentCount;
    k = 32;
    while (k < n)
      k *= 2;
    mp = 32;
    while (mp < m)
      mp *= 2;
    rankCount = (m + 63) / 64 * 64;
    pipe.InitBuffer(storage, t.unifiedBufferBytes - 8192);
    auto mem = storage.Get<float>();
    uint32_t pitch = (n + 15) / 16 * 16;
    BindSpatialWorkspace(mem, pitch, t);
    PointSort();
    PipeBarrier<PIPE_ALL>();
    Fence<HardEvent::V_S>();
    Fence<HardEvent::MTE3_S>();
    cellEnd = mem.ReinterpretCast<int32_t>();
    pool = mem[64];
    Segments();
    Fence<HardEvent::V_S>();
    SearchSpatialCells();
    WriteSpatialResult(mem, pitch);
  }
};

#include "tree_line_spatial_index.h"
