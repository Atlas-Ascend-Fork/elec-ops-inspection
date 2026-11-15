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
// Segment-major triples let three short queries share one SIMD repeat.
// Every real pair is computed. Fixed slot masks keep reductions independent.
struct InterleavedQueryTile {
  LocalTensor<float> raw, ax, ay, az, dx, dy, dz, reciprocal, x, y, z, projection, temporary, reduction, pointInput;
  LocalTensor<uint32_t> pointOffsets, slotIndices, segmentOffsets;
  uint64_t slot0[2], slot1[2], slot2[2];
  static constexpr uint64_t firstSlotMask = 0x9249249249249249ULL;
  __aicore__ explicit InterleavedQueryTile(LocalTensor<float> memory)
      : raw(memory),
        ax(memory[192]),
        ay(memory[256]),
        az(memory[320]),
        dx(memory[384]),
        dy(memory[448]),
        dz(memory[512]),
        reciprocal(memory[576]),
        x(memory[640]),
        y(memory[704]),
        z(memory[768]),
        projection(memory[832]),
        temporary(memory[896]),
        reduction(memory[960]),
        pointInput(memory[1024]),
        pointOffsets(memory[1088].ReinterpretCast<uint32_t>()),
        slotIndices(memory[1152].ReinterpretCast<uint32_t>()),
        segmentOffsets(memory[1216].ReinterpretCast<uint32_t>()),
        slot0{firstSlotMask, 0}, slot1{firstSlotMask << 1, 0}, slot2{firstSlotMask << 2, 0} {}
  __aicore__ inline void Load(GlobalTensor<float> points, GlobalTensor<float> segments, uint32_t pointCount, uint32_t segmentCount) {
    Duplicate(raw, 0.0f, 192);
    Duplicate(pointInput, 0.0f, 16);
    Fence<HardEvent::V_MTE2>();
    DataCopyPad(raw, segments, {1, segmentCount * 24, 0, 0, 0}, {false, 0, 0, 0});
    // Initialize all three point slots, including rows outside a short tail.
    uint8_t pointPaddingElements = uint8_t((8 - pointCount * 3 % 8) % 8);
    DataCopyPad(pointInput, points, {1, pointCount * 12, 0, 0, 0},
                {true, 0, pointPaddingElements, 0.0f});
    Duplicate(slotIndices, uint32_t(0), slot0, 1, 1, 8);
    Duplicate(slotIndices, uint32_t(1), slot1, 1, 1, 8);
    Duplicate(slotIndices, uint32_t(2), slot2, 1, 1, 8);
    Muls(pointOffsets.ReinterpretCast<int32_t>(), slotIndices.ReinterpretCast<int32_t>(),
         int32_t(12), 64);
    Muls(slotIndices.ReinterpretCast<int32_t>(), slotIndices.ReinterpretCast<int32_t>(),
         int32_t(8), 64);
    DeviceOffsets(segmentOffsets, 64, 8);
    // lane = 3*segment + slot, hence byteOffset = 8*(lane-slot).
    Sub(segmentOffsets.ReinterpretCast<int32_t>(), segmentOffsets.ReinterpretCast<int32_t>(),
        slotIndices.ReinterpretCast<int32_t>(), 64);
    Fence<HardEvent::MTE2_V>();
  }
  __aicore__ inline void BuildGeometry() {
    Gather(ax, raw, segmentOffsets, 0, 64);
    Gather(ay, raw, segmentOffsets, 4, 64);
    Gather(az, raw, segmentOffsets, 8, 64);
    Gather(dx, raw, segmentOffsets, 12, 64);
    Gather(dy, raw, segmentOffsets, 16, 64);
    Gather(dz, raw, segmentOffsets, 20, 64);
    Sub(dx, dx, ax, 64);
    Sub(dy, dy, ay, 64);
    Sub(dz, dz, az, 64);
    Mul(reciprocal, dx, dx, 64);
    Mul(temporary, dy, dy, 64);
    Add(reciprocal, reciprocal, temporary, 64);
    Mul(temporary, dz, dz, 64);
    Add(reciprocal, reciprocal, temporary, 64);
    auto positiveLength = reduction.ReinterpretCast<uint8_t>();
    CompareScalar(positiveLength, reciprocal, 0.0f, CMPMODE::GT, 64);
    Select(reciprocal, positiveLength, reciprocal, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 64);
    Duplicate(temporary, 1.0f, 64);
    Div(reciprocal, temporary, reciprocal, 64);
  }
  __aicore__ inline void ProjectPoints() {
    Gather(x, pointInput, pointOffsets, 0, 64);
    Gather(y, pointInput, pointOffsets, 4, 64);
    Gather(z, pointInput, pointOffsets, 8, 64);
    Sub(x, ax, x, 64);
    Sub(y, ay, y, 64);
    Sub(z, az, z, 64);
    Mul(projection, x, dx, 64);
    MulAddDst(projection, y, dy, 64);
    MulAddDst(projection, z, dz, 64);
    Mul(projection, projection, reciprocal, 64);
    Maxs(projection, projection, -1.0f, 64);
    Mins(projection, projection, 0.0f, 64);
    Mul(temporary, projection, dx, 64);
    Sub(x, x, temporary, 64);
    Mul(temporary, projection, dy, 64);
    Sub(y, y, temporary, 64);
    Mul(temporary, projection, dz, 64);
    Sub(z, z, temporary, 64);
    Mul(temporary, x, x, 64);
    MulAddDst(temporary, y, y, 64);
    MulAddDst(temporary, z, z, 64);
  }
  __aicore__ inline void StoreMinima(GlobalTensor<float> distances, GlobalTensor<int32_t> indices, uint32_t pointCount, uint32_t segmentCount) {
    uint64_t validPairMask = (uint64_t(1) << (segmentCount * 3)) - 1;
    slot0[0] &= validPairMask;
    slot1[0] &= validPairMask;
    slot2[0] &= validPairMask;
    WholeReduceMin(reduction, temporary, slot0, 1, 1, 1, 8);
    WholeReduceMin(reduction[2], temporary, slot1, 1, 1, 1, 8);
    WholeReduceMin(reduction[4], temporary, slot2, 1, 1, 1, 8);
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(pointCount * 2);
    GatherMask(x, reduction, uint8_t(1), {1, 1, 8, 1});
    GatherMask(y, reduction, uint8_t(2), {1, 1, 8, 1});
    SetMaskNorm();
    ResetMask();
    // All selected indices are lanes0..63; floor(43*lane/128) is exactly lane/3.
    Muls(y.ReinterpretCast<int32_t>(), y.ReinterpretCast<int32_t>(), int32_t(43), pointCount);
    ShiftRight(y.ReinterpretCast<uint32_t>(), y.ReinterpretCast<uint32_t>(), uint32_t(7), pointCount);
    Sqrt(x, x, pointCount);
    Fence<HardEvent::V_MTE3>();
    DataCopyPad(distances, x, {1, pointCount * 4, 0, 0, 0});
    DataCopyPad(indices, y.ReinterpretCast<int32_t>(), {1, pointCount * 4, 0, 0, 0});
    Fence<HardEvent::MTE3_V>();
  }
};
__aicore__ inline void ComputeThreeInterleavedFloatQueries(
    GM_ADDR pointAddress, GM_ADDR segmentAddress, GM_ADDR distanceAddress,
    GM_ADDR indexAddress, uint32_t pointCount, uint32_t segmentCount) {
  LocalTensor<float> memory(TPosition::VECCALC, 0, 2048);
  InterleavedQueryTile tile(memory);
  GlobalTensor<float> points, segments, distances;
  GlobalTensor<int32_t> indices;
  points.SetGlobalBuffer((__gm__ float *)pointAddress);
  segments.SetGlobalBuffer((__gm__ float *)segmentAddress);
  distances.SetGlobalBuffer((__gm__ float *)distanceAddress);
  indices.SetGlobalBuffer((__gm__ int32_t *)indexAddress);
  tile.Load(points, segments, pointCount, segmentCount);
  tile.BuildGeometry();
  tile.ProjectPoints();
  tile.StoreMinima(distances, indices, pointCount, segmentCount);
}

// Pack several point rows into a full 64-float SIMD iteration.
template <typename T, uint32_t segmentLaneCount, bool FusedDotProduct = false> class PackedRowsTreeLine {
  // A vector repeat contains 64 FP32 lanes. Each point row must fit uint8 repeat fields;
  // these capacities also keep the six dense planes within the available UB budget.
  static constexpr uint32_t pointRowCapacity =
                 segmentLaneCount == 2 ? 224 : (segmentLaneCount <= 16 ? 240 : 216),
               pairCapacity = pointRowCapacity * segmentLaneCount,
               planePitchElements = pairCapacity + 64,
               pointRowsPerVector = 64 / segmentLaneCount;
  // Sub-block rows retain compensated quotient reconstruction. Rows of
  // eight or more segments use the same reciprocal/residual arithmetic as
  // the original general vector path, with all intermediates in FP32.
  static constexpr bool usesCompensatedDivision = segmentLaneCount < 8;
  GlobalTensor<T> pg, sg, og;
  GlobalTensor<int32_t> ig;
  LocalTensor<float> mem, ax, ay, az, dx, dy, dz, inv, raw, x, y, z, dot, tmp, brc, group, seed, red,
    segmentSquaredLengths, pointInput;
  LocalTensor<uint32_t> segOff, pointOff, pointPackingMask;
  uint32_t segmentCount, pointsPerBatch, livePointCount, vectorRepeatCount, activeLaneCount;
  __aicore__ inline void Load(LocalTensor<float> dst, GlobalTensor<T> src, uint32_t items,
                LocalTensor<float> scratch) {
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(dst, src, {1, items * 4, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
    } else {
      auto h = scratch.ReinterpretCast<half>();
      DataCopyPad(h, src, {1, items * 2, 0, 0, 0}, {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
      Cast(dst, h, RoundMode::CAST_NONE, items);
    }
  }
  __aicore__ inline void Product(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b) {
    Mul(dst, a, b, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 8, 0});
  }
  __aicore__ inline void PointBroadcast(uint32_t axis) {
    if constexpr (segmentLaneCount < 8) {
      uint32_t paddedPointRows = vectorRepeatCount * pointRowsPerVector;
      Brcb(seed, group[axis * pointRowCapacity], uint8_t(paddedPointRows / 8), {1, 8});
      // Brcb creates eight copies; retain W of each. The packed row
      // count is known, so keep counter semantics without reading its SPR.
      SetMaskCount();
      SetVectorMask<float, MaskMode::COUNTER>(int32_t(paddedPointRows * 8));
      GatherMask(brc, seed, pointPackingMask.ReinterpretCast<float>(), uint8_t(0), {1, 1, 8, 1});
      SetMaskNorm();
      ResetMask();
    } else {
      Brcb(seed, group[axis * pointRowCapacity], uint8_t((livePointCount + 7) / 8), {1, 8});
      Copy(brc, seed, uint64_t(segmentLaneCount),
        uint8_t((livePointCount + pointRowsPerVector - 1) / pointRowsPerVector *
            pointRowsPerVector),
        {1, 0, segmentLaneCount / 8, 1});
    }
  }
  // A dedicated point staging tile fits after the live offset table. Once
  // all three coordinate gathers complete, the next DMA may replace it while
  // geometry and output DMA use disjoint planes.
  __aicore__ inline void PrefetchPointTile(uint32_t pointBegin, uint32_t pointCount) {
    DataCopyPad(pointInput.ReinterpretCast<T>(), pg[pointBegin * 3],
          {1, pointCount * uint32_t(3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  }
  __aicore__ inline void StartInitialInputPrefetch(uint32_t batch, uint32_t pointBegin,
                          uint32_t pointCount) {
    Duplicate(raw, 0.0f, 256);
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(raw, sg[batch * segmentCount * 6],
            {1, segmentCount * 24, 0, 0, 0}, {false, 0, 0, 0});
    } else {
      DataCopyPad(brc.ReinterpretCast<half>(), sg[batch * segmentCount * 6],
            {1, segmentCount * 12, 0, 0, 0}, {false, 0, 0, 0});
    }
    PrefetchPointTile(pointBegin, pointCount);
  }
  __aicore__ inline void FinishInitialInputPrefetch() {
    Fence<HardEvent::MTE2_V>();
    if constexpr (IsSameType<T, half>::value)
      Cast(raw, brc.ReinterpretCast<half>(), RoundMode::CAST_NONE, segmentCount * 6);
  }
  __aicore__ inline void Segments(uint32_t batch, bool hasPrefetchedInputs = false) {
    if (!hasPrefetchedInputs) {
      Duplicate(raw, 0.0f, 256);
      Load(raw, sg[batch * segmentCount * 6], segmentCount * 6, brc);
    }
    LocalTensor<float> cols[6] = {ax, ay, az, dx, dy, dz};
    for (uint32_t a = 0; a < 6; ++a) {
      if constexpr (segmentLaneCount < 8) {
        Gather(cols[a], raw, segOff, a * 4, 64);
      } else {
        Gather(cols[a], raw, segOff, a * 4, segmentLaneCount);
        if constexpr (pointRowsPerVector > 1)
          Copy(cols[a][segmentLaneCount], cols[a], uint64_t(segmentLaneCount),
            uint8_t(pointRowsPerVector - 1), {1, 1, segmentLaneCount / 8, 0});
      }
    }
    Sub(dx, dx, ax, 64);
    Sub(dy, dy, ay, 64);
    Sub(dz, dz, az, 64);
    Mul(inv, dx, dx, 64);
    Mul(tmp, dy, dy, 64);
    Add(inv, inv, tmp, 64);
    Mul(tmp, dz, dz, 64);
    Add(inv, inv, tmp, 64);
    auto mask = red.ReinterpretCast<uint8_t>();
    CompareScalar(mask, inv, 0.0f, CMPMODE::GT, 64);
    Select(inv, mask, inv, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 64);
    if constexpr (usesCompensatedDivision)
      Adds(segmentSquaredLengths, inv, 0.0f, 64);
    if constexpr (segmentLaneCount >= 8) {
      // Preserve the existing W>=8 setup; W<8 consumes L directly.
      Duplicate(tmp, 1.0f, 64);
      Div(inv, tmp, inv, 64);
    }
  }
  __aicore__ inline void ReducePointRows() {
        auto reductionInput = tmp;
        if constexpr (usesCompensatedDivision) {
          // Preserve the sub-block row reduction layout. Other rows
          // consume the products directly and avoid one dense copy.
          Adds(x, tmp, 0.0f, activeLaneCount);
          reductionInput = x;
        }
        if constexpr (segmentLaneCount < 8) {
          // Reduce independent sub-block point rows without requiring a 16B/8B repeat stride.
          constexpr uint32_t pointRowsPerBlock = 8 / segmentLaneCount;
    #pragma unroll
          for (uint32_t slot = 0; slot < pointRowsPerBlock; ++slot) {
            uint64_t laneMask[2] = {((uint64_t(1) << segmentCount) - 1) << (slot * segmentLaneCount),
                        0};
            WholeReduceMin(red[2 * slot], reductionInput, laneMask,
                   uint8_t((livePointCount + pointRowsPerBlock - 1) / pointRowsPerBlock),
                   pointRowsPerBlock, 1, 1);
          }
        } else {
          WholeReduceMin(red, reductionInput, int32_t(segmentCount), livePointCount, 1, 1,
                 segmentLaneCount / 8);
        }
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(livePointCount * 2);
        GatherMask(x, red, uint8_t(1), {1, 1, 8, 1});
        GatherMask(y, red, uint8_t(2), {1, 1, 8, 1});
        SetMaskNorm();
        ResetMask();
        if constexpr (segmentLaneCount < 8) {
          constexpr uint32_t indexShiftBits = segmentLaneCount == 2 ? 31 : 30;
          ShiftLeft(y.ReinterpretCast<uint32_t>(), y.ReinterpretCast<uint32_t>(), indexShiftBits,
               livePointCount);
          ShiftRight(y.ReinterpretCast<uint32_t>(), y.ReinterpretCast<uint32_t>(), indexShiftBits,
               livePointCount);
        }
        Sqrt(x, x, livePointCount);
  }

  __aicore__ inline void Compute() {
    LocalTensor<float> delta[3] = {x, y, z}, starts[3] = {ax, ay, az}, directions[3] = {dx, dy, dz};
    for (uint32_t a = 0; a < 3; ++a) {
      PointBroadcast(a);
      Sub(delta[a], starts[a], brc, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 0, 8});
    }
    Product(dot, x, dx);
    if constexpr (FusedDotProduct) {
      MulAddDst(dot, y, dy, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 8, 0});
      MulAddDst(dot, z, dz, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 8, 0});
    } else {
      Product(tmp, y, dy);
      Add(dot, dot, tmp, activeLaneCount);
      Product(tmp, z, dz);
      Add(dot, dot, tmp, activeLaneCount);
    }
    if constexpr (usesCompensatedDivision) {
      Div(tmp, dot, segmentSquaredLengths, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 8, 0});
      Muls(x, tmp, -1.0f, activeLaneCount);
      MulAddDst(dot, x, segmentSquaredLengths, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 8, 0});
      Div(dot, dot, segmentSquaredLengths, uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 8, 0});
      Add(dot, tmp, dot, activeLaneCount);
    } else {
      Product(dot, dot, inv);
    }
    Maxs(dot, dot, -1.0f, activeLaneCount);
    Mins(dot, dot, 0.0f, activeLaneCount);
    if constexpr (usesCompensatedDivision) {
      for (uint32_t a = 0; a < 3; ++a) {
        Product(delta[a], dot, directions[a]);
        Sub(delta[a], starts[a], delta[a], uint64_t(64), vectorRepeatCount, {1, 1, 1, 8, 0, 8});
        PointBroadcast(a);
        Sub(delta[a], brc, delta[a], activeLaneCount);
      }
      Mul(tmp, x, x, activeLaneCount);
      Mul(y, y, y, activeLaneCount);
      Add(tmp, tmp, y, activeLaneCount);
      Mul(z, z, z, activeLaneCount);
      Add(tmp, tmp, z, activeLaneCount);
    } else {
      for (uint32_t a = 0; a < 3; ++a) {
        Product(tmp, dot, directions[a]);
        Sub(delta[a], delta[a], tmp, activeLaneCount);
      }
      Mul(tmp, x, x, activeLaneCount);
      MulAddDst(tmp, y, y, activeLaneCount);
      MulAddDst(tmp, z, z, activeLaneCount);
    }
    ReducePointRows();
  }

  __aicore__ inline void BindPackedPlanes(const TreeLineMinDistanceTilingData &t) {
    segmentCount = t.segmentCount;
    pointsPerBatch = t.pointsPerBatch;
    mem = LocalTensor<float>(TPosition::VECCALC, 0, (t.unifiedBufferBytes - 8192) / 4);
    ax = mem;
    ay = mem[64];
    az = mem[128];
    dx = mem[192];
    dy = mem[256];
    dz = mem[320];
    inv = mem[384];
    segOff = mem[448].ReinterpretCast<uint32_t>();
    raw = mem[512];
    segmentSquaredLengths = mem[832];
    x = mem[1024];
    y = x[planePitchElements];
    z = x[2 * planePitchElements];
    dot = x[3 * planePitchElements];
    tmp = x[4 * planePitchElements];
    brc = x[5 * planePitchElements];
    group = x[6 * planePitchElements];
    seed = group[3 * pointRowCapacity];
    red = seed[8 * pointRowCapacity];
    pointOff = red[512].ReinterpretCast<uint32_t>();
    pointInput = pointOff[segmentLaneCount < 8 ? 2 * pointRowCapacity + 64
                         : pointRowCapacity].ReinterpretCast<float>();
  }

  __aicore__ inline void PrefetchPackedInputs(uint32_t begin, uint32_t end) {
        uint32_t firstPointCount = end - begin;
        if (firstPointCount > pointRowCapacity)
          firstPointCount = pointRowCapacity;
        uint32_t firstBatch = begin / pointsPerBatch;
        uint32_t firstBatchRemaining = (firstBatch + 1) * pointsPerBatch - begin;
        if (firstPointCount > firstBatchRemaining)
          firstPointCount = firstBatchRemaining;
        StartInitialInputPrefetch(firstBatch, begin, firstPointCount);
        if constexpr (segmentLaneCount < 8) {
          pointPackingMask = red[512 + 2 * pointRowCapacity].ReinterpretCast<uint32_t>();
          constexpr uint32_t packingByte = (1U << segmentLaneCount) - 1U;
          Duplicate(pointPackingMask, packingByte * uint32_t(0x01010101), 64);
          Fence<HardEvent::V_S>();
    #pragma unroll
          for (uint32_t i = 0; i < 8; ++i)
            segOff.SetValue(i, (i % segmentLaneCount) * 24);
          Fence<HardEvent::S_V>();
    #pragma unroll
          for (uint32_t prefixElements = 8; prefixElements < 64; prefixElements *= 2) {
            Adds(segOff[prefixElements].ReinterpretCast<int32_t>(), segOff.ReinterpretCast<int32_t>(),
              int32_t(0), prefixElements);
            PipeBarrier<PIPE_V>();
          }
        }

        uint32_t offsetPointCount = end - begin;
        if (offsetPointCount > pointRowCapacity)
          offsetPointCount = pointRowCapacity;
        // Only live point rows index these tables; a batch boundary can shorten
        // a tile but cannot enlarge it beyond this worker's entire point slice.
        if constexpr (segmentLaneCount >= 8) {
          SeedDeviceOffsets(segOff, segmentLaneCount, 24);
          SeedDeviceOffsets(pointOff, offsetPointCount, 12);
          Fence<HardEvent::S_V>();
          ExpandDeviceOffsetSeeds(segOff, segmentLaneCount, 24);
          ExpandDeviceOffsetSeeds(pointOff, offsetPointCount, 12);
        } else {
          DeviceOffsets(pointOff, offsetPointCount, 12);
        }
        if constexpr (segmentLaneCount >= 16)
          Duplicate(group, 0.0f, 3 * pointRowCapacity);
  }

  __aicore__ inline void GatherNextTile(uint32_t at, uint32_t begin, uint32_t end, uint32_t batch) {
    livePointCount = end - at;
    if (livePointCount > pointRowCapacity)
      livePointCount = pointRowCapacity;
    if (livePointCount > (batch + 1) * pointsPerBatch - at)
      livePointCount = (batch + 1) * pointsPerBatch - at;
    vectorRepeatCount = (livePointCount + pointRowsPerVector - 1) / pointRowsPerVector;
    activeLaneCount = vectorRepeatCount * 64;
    // Wide rows reduce independently. Define padding once and
    // overwrite every live coordinate on each tile; padded rows never
    // enter output reduction. Narrow packing retains its full reset.
    if constexpr (segmentLaneCount < 16)
      Duplicate(group, 0.0f, 3 * pointRowCapacity);
    if (at != begin)
      Fence<HardEvent::MTE2_V>();
    auto pointCoordinates = pointInput;
    if constexpr (IsSameType<T, half>::value) {
      Cast(x, pointInput.ReinterpretCast<half>(), RoundMode::CAST_NONE, livePointCount * 3);
      pointCoordinates = x;
    }
    Gather(group, pointCoordinates, pointOff, 0, livePointCount);
    Gather(group[pointRowCapacity], pointCoordinates, pointOff, 4, livePointCount);
    Gather(group[2 * pointRowCapacity], pointCoordinates, pointOff, 8, livePointCount);
    uint32_t nextPoint = at + livePointCount;
    if (nextPoint < end) {
      uint32_t nextPointCount = end - nextPoint;
      if (nextPointCount > pointRowCapacity)
        nextPointCount = pointRowCapacity;
      uint32_t nextBatchEnd = (nextPoint / pointsPerBatch + 1) * pointsPerBatch;
      if (nextPointCount > nextBatchEnd - nextPoint)
        nextPointCount = nextBatchEnd - nextPoint;
      Fence<HardEvent::V_MTE2>();
      PrefetchPointTile(nextPoint, nextPointCount);
    }
  }

  __aicore__ inline void StoreDistances(uint32_t at) {
    if constexpr (IsSameType<T, float>::value) {
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(og[at], x, {1, livePointCount * 4, 0, 0, 0});
    } else {
      auto h = x.ReinterpretCast<half>();
      Cast(h, x, RoundMode::CAST_RINT, livePointCount);
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(og[at], h, {1, livePointCount * 2, 0, 0, 0});
    }
    DataCopyPad(ig[at], y.ReinterpretCast<int32_t>(), {1, livePointCount * 4, 0, 0, 0});
    Fence<HardEvent::MTE3_V>();
  }

 public:
  __aicore__ inline void Run(GM_ADDR p, GM_ADDR s, GM_ADDR o, GM_ADDR oi,
               const TreeLineMinDistanceTilingData &t) {
    if constexpr (segmentLaneCount == 32 && IsSameType<T, float>::value) {
      uint32_t totalPoints = t.batchCount * t.pointsPerBatch;
      uint32_t pointBegin = totalPoints * GetBlockIdx() / t.workerCount;
      uint32_t pointEnd = totalPoints * (GetBlockIdx() + 1) / t.workerCount;
      if (pointEnd > pointBegin && pointEnd - pointBegin <= 3 && t.segmentCount * 3 <= 64 &&
          pointBegin / t.pointsPerBatch == (pointEnd - 1) / t.pointsPerBatch) {
        uint32_t batchIndex = pointBegin / t.pointsPerBatch;
        ComputeThreeInterleavedFloatQueries(
          p + uint64_t(pointBegin) * 12, s + uint64_t(batchIndex) * t.segmentCount * 24,
          o + uint64_t(pointBegin) * 4, oi + uint64_t(pointBegin) * 4,
          pointEnd - pointBegin, t.segmentCount);
        return;
      }
    }
    pg.SetGlobalBuffer((__gm__ T *)p);
    sg.SetGlobalBuffer((__gm__ T *)s);
    og.SetGlobalBuffer((__gm__ T *)o);
    ig.SetGlobalBuffer((__gm__ int32_t *)oi);
    BindPackedPlanes(t);
    uint32_t total = t.batchCount * pointsPerBatch, begin = total * GetBlockIdx() / t.workerCount,
        end = total * (GetBlockIdx() + 1) / t.workerCount, oldBatch = 0xffffffff;
    PrefetchPackedInputs(begin, end);
    for (uint32_t at = begin; at < end;) {
      uint32_t batch = at / pointsPerBatch;
      if (batch != oldBatch) {
        if (at == begin)
          FinishInitialInputPrefetch();
        Segments(batch, at == begin);
        oldBatch = batch;
      }
      GatherNextTile(at, begin, end, batch);
      Compute();
      StoreDistances(at);
      at += livePointCount;
    }
  }
};
} // namespace OptimizedLayout
