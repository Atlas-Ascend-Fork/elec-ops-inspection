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
template <typename T, uint32_t ALGORITHM> class KernelTreeLineMinDistance : public TreeLinePointKernel<T> {
  static constexpr uint32_t STATIC_WIDTH =
    ALGORITHM >= 24 ? 8U * (ALGORITHM - 23)
            : (ALGORITHM >= 13 && ALGORITHM <= 16 ? (128U << (ALGORITHM - 13)) : 0);
  uint32_t bufferBase = 0;
  LocalTensor<float> outLocal;
  uint32_t outStart;
  LocalTensor<int32_t> idxLocal;
  using TreeLinePointKernel<T>::addresses;
  using TreeLinePointKernel<T>::t;
  uint32_t width, rows, pairs;
  LocalTensor<float> raw, ax, ay, az, dx, dy, dz, len, x, y, z, dot, tmp, dist, red;
  LocalTensor<uint32_t> gather;
  __aicore__ inline void SeedOffsetTable(LocalTensor<uint32_t> destination,
                       uint32_t elementCount, uint32_t byteStride) {
    if constexpr (STATIC_WIDTH == 64 && IsSameType<T, half>::value) {
      uint32_t seedElements = elementCount < 8 ? elementCount : 8;
      auto pairedSeeds = destination.ReinterpretCast<uint64_t>();
      // Offset tables are32-byte aligned. Little-endian paired stores keep
      // the existing byte offsets; all table contents are generated here.
      uint64_t packedPair = uint64_t(byteStride) << 32;
      uint64_t packedStep = (uint64_t(byteStride * 2) << 32) | uint64_t(byteStride * 2);
      for (uint32_t pair = 0; pair < seedElements / 2; ++pair) {
        pairedSeeds.SetValue(pair, packedPair);
        packedPair += packedStep;
      }
      if (seedElements & 1)
        destination.SetValue(seedElements - 1, (seedElements - 1) * byteStride);
    } else {
      SeedDeviceOffsets(destination, elementCount, byteStride);
    }
  }
  __aicore__ inline void Store(uint32_t i, float d, int32_t arg) {
    outLocal.SetValue(i - outStart, d);
    idxLocal.SetValue(i - outStart, arg);
  }
  __aicore__ inline void Flush(uint32_t count) {
    Fence<HardEvent::S_V>();
    Sqrt(outLocal, outLocal, count);
    if constexpr (IsSameType<T, float>::value) {
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(addresses.output[outStart], outLocal, {1, count * 4, 0, 0, 0});
    } else {
      auto halfout = outLocal.ReinterpretCast<half>();
      Cast(halfout, outLocal, RoundMode::CAST_RINT, count);
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(addresses.output[outStart], halfout, {1, count * 2, 0, 0, 0});
    }
    DataCopyPad(addresses.indicesOut[outStart], idxLocal, {1, count * 4, 0, 0, 0});
    Fence<HardEvent::MTE3_S>();
  }
  __aicore__ inline void PrefetchSegmentInputs(uint32_t batch) {
    if constexpr (STATIC_WIDTH == 64 && IsSameType<T, half>::value) {
      // A full segment tile is overwritten by this invocation's DMA/Cast
      // before Gather. Partial tiles still initialize every padded lane.
      if (t.segmentCount < width)
        Duplicate(raw, 0.0f, width * 6);
    } else {
      Duplicate(raw, 0.0f, width * 6);
    }
    Fence<HardEvent::V_MTE2>();
    if constexpr (IsSameType<T, float>::value) {
      DataCopyPad(raw, addresses.segments[batch * t.segmentCount * 6], {1, t.segmentCount * 24, 0, 0, 0},
            {false, 0, 0, 0});
    } else {
      auto halfraw = raw.ReinterpretCast<half>()[width * 12];
      DataCopyPad(halfraw, addresses.segments[batch * t.segmentCount * 6], {1, t.segmentCount * 12, 0, 0, 0},
            {false, 0, 0, 0});
    }
  }
  __aicore__ inline void PrefetchPointInputs(uint32_t pointBegin, uint32_t pointCount) {
    if constexpr (IsSameType<T, float>::value)
      DataCopyPad(red[512], addresses.points[pointBegin * 3], {1, pointCount * 12, 0, 0, 0},
            {false, 0, 0, 0});
    else
      DataCopyPad(red[768].ReinterpretCast<half>(), addresses.points[pointBegin * 3],
            {1, pointCount * 6, 0, 0, 0}, {false, 0, 0, 0});
  }
  __aicore__ inline void LoadSegments(uint32_t batch, bool inputsAlreadyStaged = false) {
    if (!inputsAlreadyStaged)
      PrefetchSegmentInputs(batch);
    Fence<HardEvent::MTE2_V>();
    if constexpr (IsSameType<T, half>::value) {
      auto halfraw = raw.ReinterpretCast<half>()[width * 12];
      Cast(raw, halfraw, RoundMode::CAST_NONE, t.segmentCount * 6);
    }
    Gather(ax, raw, gather, 0, width);
    Gather(ay, raw, gather, 4, width);
    Gather(az, raw, gather, 8, width);
    Gather(dx, raw, gather, 12, width);
    Gather(dy, raw, gather, 16, width);
    Gather(dz, raw, gather, 20, width);
    Sub(dx, dx, ax, width);
    Sub(dy, dy, ay, width);
    Sub(dz, dz, az, width);
    Mul(len, dx, dx, width);
    Mul(tmp, dy, dy, width);
    Mul(dot, dz, dz, width);
    Add(len, len, tmp, width);
    Add(len, len, dot, width);
    if constexpr (STATIC_WIDTH == 64 && IsSameType<T, half>::value) {
      // Only an exactly zero squared length receives a safe denominator.
      // The mask uses reduction scratch, disjoint from staged input data.
      auto nonzeroLengthMask = red.ReinterpretCast<uint8_t>();
      CompareScalar(nonzeroLengthMask, len, 0.0f, CMPMODE::NE, width);
      Select(len, nonzeroLengthMask, len, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, width);
    } else {
    // Repair exactly zero nonnegative squared lengths without a vector select.
    Mins(tmp.ReinterpretCast<int32_t>(), len.ReinterpretCast<int32_t>(), int32_t(1), width);
    Muls(tmp.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(-0x3f800000), width);
    Adds(tmp.ReinterpretCast<int32_t>(), tmp.ReinterpretCast<int32_t>(), int32_t(0x3f800000), width);
    Or(len.ReinterpretCast<uint16_t>(), len.ReinterpretCast<uint16_t>(),
     tmp.ReinterpretCast<uint16_t>(), width * 2);
    }
    // A batch shares segment lengths across every point tile.
    Duplicate(tmp, 1.0f, width);
    Div(len, tmp, len, width);
  }

  struct Segment {
    float a, b, c, u, v, w;
    float squaredLength, inverseLength;
  };
  __aicore__ inline void ReadSegment(uint32_t batch, bool cacheInverse, Segment &s, LocalTensor<float> next, LocalTensor<float> segmentLocal);
  __aicore__ inline void PrefetchSinglePoints(uint32_t start, uint32_t firstPointCount, uint32_t w);
  __aicore__ inline void GatherSinglePoints(uint32_t w, uint32_t live);
  __aicore__ inline void ProjectSingleSegment(const Segment &s, bool cacheInverse, uint32_t count, LocalTensor<float> next, LocalTensor<uint8_t> mask);
  __aicore__ inline void WriteSingleDistances(uint32_t i, uint32_t live, uint32_t count, LocalTensor<float> next);
  __aicore__ inline void BindSinglePointPlanes(LocalTensor<float> mem, uint32_t planePitchElements);
  __aicore__ inline void PointVector(uint32_t start, uint32_t end);
  __aicore__ inline void Product(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b,
                 uint32_t live) {
    uint32_t ew = STATIC_WIDTH ? STATIC_WIDTH : width;
    if (ew <= 64 || live >= ew / 64) {
#pragma unroll
      for (uint32_t c = 0; c < ew; c += 64) {
        uint32_t mask = ew < 64 ? ew : 64;
        Mul(dst[c], a[c], b[c], uint64_t(mask), live,
          {1, 1, 1, uint8_t(ew / 8), uint8_t(ew / 8), 0});
      }
    } else {
      for (uint32_t r = 0; r < live; ++r)
        Mul(dst[r * ew], a[r * ew], b, ew);
    }
  }
  __aicore__ inline void Delta(LocalTensor<float> dst, LocalTensor<float> axis, uint32_t i, uint32_t live,
                uint32_t coordinate) {
    for (uint32_t r = 0; r < live; ++r) {
      float p = float(addresses.points.GetValue((i + r) * 3 + coordinate));
      Adds(dst[r * width], axis, -p, width);
    }
  }
  __aicore__ inline void ChooseVectorTile(uint32_t start, uint32_t end) {
    if constexpr (ALGORITHM >= 13)
      width = STATIC_WIDTH;
    else {
      width = (t.segmentCount + 7) / 8 * 8;
      if (width > 64) {
        width = 128;
        while (width < t.segmentCount)
          width *= 2;
      }
    }
    uint32_t outBytes = ((end - start + 15) / 16 * 16) * 8;
    uint32_t available = t.unifiedBufferBytes - outBytes - width * 32 - 14592;
    rows = available / (width * 20);
    if (rows > 64)
      rows = 64;
    if (rows < 1)
      rows = 1;
    if (rows > end - start)
      rows = end - start;
    if (rows > t.pointsPerBatch)
      rows = t.pointsPerBatch;
    if (width < 64)
      rows = (rows + 7) / 8 * 8;
    pairs = rows * width;
  }

  __aicore__ inline void BindVectorPlanes() {
    uint32_t planePitch = pairs + 64, work = 5 * planePitch;
    if (work < 9 * width)
      work = 9 * width;
    LocalTensor<float> mem(TPosition::VECCALC, bufferBase,
               (t.unifiedBufferBytes - 8192 - bufferBase) / 4);
    ax = mem;
    ay = mem[width];
    az = mem[width * 2];
    dx = mem[width * 3];
    dy = mem[width * 4];
    dz = mem[width * 5];
    len = mem[width * 6];
    gather = mem[width * 7].template ReinterpretCast<uint32_t>();
    auto v = mem[width * 8];
    raw = v;
    x = v;
    y = v[planePitch];
    z = v[planePitch * 2];
    dot = v[planePitch * 3];
    tmp = v[planePitch * 4];
    if constexpr (STATIC_WIDTH == 64 && IsSameType<T, half>::value) {
      // Projection is dead after all three residuals have been formed.
      // Its plane is distinct from each residual source during the norm.
      dist = dot;
    } else {
      dist = tmp;
    }
    red = v[work];
  }

  __aicore__ inline void BuildVectorOffsets(bool usesPointCoordinateGather) {
    // These fresh offset tables are disjoint from each other and from
    // both DMA destinations. Seed them with one scalar-to-vector handoff.
    if constexpr (STATIC_WIDTH == 64 && IsSameType<T, half>::value) {
      // All three address tables are linear in the physical lane. Seed
      // one sequence here, then derive the current point/output/segment
      // byte strides on device. No table survives the invocation.
      SeedOffsetTable(gather, width, 1);
      Fence<HardEvent::S_V>();
      ExpandDeviceOffsetSeeds(gather, width, 1);
      if (usesPointCoordinateGather)
        Muls(red[704].ReinterpretCast<int32_t>(), gather.ReinterpretCast<int32_t>(),
             int32_t(12), 64);
      if (rows >= 8)
        Muls(red[1024].ReinterpretCast<int32_t>(), gather.ReinterpretCast<int32_t>(),
             int32_t(8), 64);
      // Preserve both consumers of the lane sequence before replacing it.
      PipeBarrier<PIPE_V>();
      Muls(gather.ReinterpretCast<int32_t>(), gather.ReinterpretCast<int32_t>(),
           int32_t(24), width);
      PipeBarrier<PIPE_V>();
    } else {
      SeedOffsetTable(gather, width, 24);
      if (usesPointCoordinateGather)
        SeedOffsetTable(red[704].ReinterpretCast<uint32_t>(), 64, 12);
      if (rows >= 8)
        SeedOffsetTable(red[1024].ReinterpretCast<uint32_t>(), 64, 8);
      if (rows >= 8 && width > 64) {
        uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2;
        SeedOffsetTable(red[1088].ReinterpretCast<uint32_t>(), 64, chunks * spacing * 4);
      }
      Fence<HardEvent::S_V>();
      ExpandDeviceOffsetSeeds(gather, width, 24);
      if (usesPointCoordinateGather)
        ExpandDeviceOffsetSeeds(red[704].ReinterpretCast<uint32_t>(), 64, 12);
      if (rows >= 8)
        ExpandDeviceOffsetSeeds(red[1024].ReinterpretCast<uint32_t>(), 64, 8);
      if (rows >= 8 && width > 64) {
        uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2;
        ExpandDeviceOffsetSeeds(red[1088].ReinterpretCast<uint32_t>(), 64, chunks * spacing * 4);
      }
    }
  }

  __aicore__ inline void GatherPointTile(uint32_t i, uint32_t start, uint32_t live, bool hasStagedPointTile) {
    if (!(i == start && hasStagedPointTile)) {
      Fence<HardEvent::V_MTE2>();
      PrefetchPointInputs(i, live);
      Fence<HardEvent::MTE2_V>();
    }
    if constexpr (IsSameType<T, half>::value) {
      auto h = red[768].ReinterpretCast<half>();
      Cast(red[512], h, RoundMode::CAST_NONE, live * 3);
    }
    if constexpr (STATIC_WIDTH == 64 && IsSameType<T, half>::value) {
      // An aligned tile overwrites every coordinate Brcb will read.
      // Retain complete initialization for the final partial group.
      if (live % 8 != 0)
        Duplicate(red[800], 0.0f, 192);
    } else {
      Duplicate(red[800], 0.0f, 192);
    }
    for (uint32_t axis = 0; axis < 3; ++axis)
      Gather(red[800 + axis * 64], red[512], red[704].ReinterpretCast<uint32_t>(), axis * 4,
         live);
  }

  __aicore__ inline void BroadcastPointDeltas(uint32_t live) {
    uint32_t block = (live + 7) / 8 * 64;
    if (width < 24) {
      LocalTensor<float> target[3] = {x, y, z}, source[3] = {ax, ay, az};
      for (uint32_t axis = 0; axis < 3; ++axis) {
        Brcb(tmp, red[800 + axis * 64], uint8_t((live + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();
        Sub(target[axis], source[axis], tmp, uint64_t(width), live,
          {1, 1, 0, uint8_t(width / 8), 0, 1});
        PipeBarrier<PIPE_V>();
      }
    } else {
      for (uint32_t axis = 0; axis < 3; ++axis)
        Brcb(tmp[axis * block], red[800 + axis * 64], uint8_t((live + 7) / 8), {1, 8});
      for (uint32_t c = 0; c < width; c += 64) {
        uint32_t mask = width < 64 ? width : 64;
        Sub(x[c], ax[c], tmp, uint64_t(mask), live, {1, 1, 0, uint8_t(width / 8), 0, 1});
        Sub(y[c], ay[c], tmp[block], uint64_t(mask), live,
          {1, 1, 0, uint8_t(width / 8), 0, 1});
        Sub(z[c], az[c], tmp[block * 2], uint64_t(mask), live,
          {1, 1, 0, uint8_t(width / 8), 0, 1});
      }
    }
  }

  __aicore__ inline void SquaredDistances(uint32_t live, uint32_t count) {
    Product(dot, x, dx, live);
    Product(tmp, y, dy, live);
    Add(dot, dot, tmp, count);
    Product(tmp, z, dz, live);
    Add(dot, dot, tmp, count);
    Product(dot, dot, len, live);
    Maxs(dot, dot, -1.0f, count);
    Mins(dot, dot, 0.0f, count);
    Product(tmp, dot, dx, live);
    Sub(x, x, tmp, count);
    Product(tmp, dot, dy, live);
    Sub(y, y, tmp, count);
    Product(tmp, dot, dz, live);
    Sub(z, z, tmp, count);
    Mul(dist, x, x, count);
    MulAddDst(dist, y, y, count);
    MulAddDst(dist, z, z, count);
  }

  __aicore__ inline void ReduceDistances(uint32_t live) {
    if (width <= 64) {
      WholeReduceMin(red, dist, int32_t(t.segmentCount), live, 1, 1, width / 8);
    } else {
      uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2, pitch = chunks * spacing;
      auto partial = red[128];
      WholeReduceMin(partial, dist, 64, live * chunks, spacing / 2, 1, 8);
      uint32_t tail = t.segmentCount % 64;
      if (tail) {
        WholeReduceMin(partial[(t.segmentCount / 64) * spacing],
               dist[(t.segmentCount / 64) * 64], int32_t(tail), live, pitch / 2, 1,
               width / 8);
      }
      uint64_t mask[2] = {0, 0};
      for (uint32_t c = 0; c < (t.segmentCount + 63) / 64; ++c)
        mask[0] |= uint64_t(1) << (c * spacing);
      WholeReduceMin(red, partial, mask, live, 1, 1, pitch / 8);
    }
  }

  __aicore__ inline void StoreRowMinima(uint32_t i, uint32_t live) {
    uint32_t offset = i - outStart;
    if (rows >= 8 && (offset & 7) == 0) {
      Gather(outLocal[offset], red, red[1024].ReinterpretCast<uint32_t>(), 0, live);
      auto dst = idxLocal[offset];
      Gather(dst.ReinterpretCast<float>(), red, red[1024].ReinterpretCast<uint32_t>(), 4, live);
      if (width > 64) {
        uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2;
        auto addr = red[1216].ReinterpretCast<int32_t>();
        Muls(addr, dst, int32_t(4), live);
        Add(addr, addr, red[1088].ReinterpretCast<int32_t>(), live);
        Adds(addr, addr, int32_t(128 * 4 + 4), live);
        Gather(red[1152], red, addr.ReinterpretCast<uint32_t>(), 0, live);
        ShiftRight(dst, dst, int32_t(spacing == 4 ? 2 : 1), live);
        ShiftLeft(dst, dst, int32_t(6), live);
        Add(dst, dst, red[1152].ReinterpretCast<int32_t>(), live);
      }
    } else {
      Fence<HardEvent::V_S>();
      auto ir = red.template ReinterpretCast<int32_t>();
      for (uint32_t r = 0; r < live; ++r) {
        int32_t arg = ir.GetValue(r * 2 + 1);
        if (width > 64) {
          uint32_t chunks = width / 64, spacing = chunks == 2 ? 4 : 2,
              pitch = chunks * spacing;
          arg = (arg / spacing) * 64 + ir.GetValue(128 + r * pitch + arg + 1);
        }
        Store(i + r, red.GetValue(r * 2), arg);
      }
      Fence<HardEvent::S_V>();
    }
  }

  __aicore__ inline void Vector(uint32_t start, uint32_t end) {
    ChooseVectorTile(start, end);
    BindVectorPlanes();
    // Rounded allocation rows can exceed the actual point slice. Only
    // initialize point-gather offsets if a tile can consume them.
    bool usesPointCoordinateGather = rows >= 8 && end - start >= 8 && t.pointsPerBatch >= 8;
    uint32_t firstBatch = start / t.pointsPerBatch;
    uint32_t firstPointCount = end - start;
    if (firstPointCount > rows)
      firstPointCount = rows;
    uint32_t firstBatchRemaining = (firstBatch + 1) * t.pointsPerBatch - start;
    if (firstPointCount > firstBatchRemaining)
      firstPointCount = firstBatchRemaining;
    PrefetchSegmentInputs(firstBatch);
    bool hasStagedPointTile = firstPointCount >= 8;
    if (hasStagedPointTile)
      PrefetchPointInputs(start, firstPointCount);
    BuildVectorOffsets(usesPointCoordinateGather);
    uint32_t oldBatch = 0xffffffff;
    for (uint32_t i = start; i < end;) {
      uint32_t batch = i / t.pointsPerBatch;
      if (batch != oldBatch) {
        LoadSegments(batch, i == start);
        oldBatch = batch;
      }
      uint32_t live = end - i;
      if (live > rows)
        live = rows;
      uint32_t batchEnd = (batch + 1) * t.pointsPerBatch;
      if (live > batchEnd - i)
        live = batchEnd - i;
      uint32_t count = live * width;
      if (live >= 8) {
        GatherPointTile(i, start, live, hasStagedPointTile);
        BroadcastPointDeltas(live);
      } else {
        for (uint32_t r = 0; r < live; ++r) {
          float px = float(addresses.points.GetValue((i + r) * 3)),
             py = float(addresses.points.GetValue((i + r) * 3 + 1)),
             pz = float(addresses.points.GetValue((i + r) * 3 + 2));
          Adds(x[r * width], ax, -px, width);
          Adds(y[r * width], ay, -py, width);
          Adds(z[r * width], az, -pz, width);
        }
      }

      SquaredDistances(live, count);
      ReduceDistances(live);
      StoreRowMinima(i, live);
      i += live;
    }
  }

 public:
  __aicore__ inline void Process() {
    auto range = AssignedPoints(t);
    uint32_t start = range.begin, end = range.end;
    if constexpr (ALGORITHM == 3) {
      PointVector(start, end);
      return;
    }
    uint32_t pitch = (end - start + 15) / 16 * 16;
    outStart = start;
    outLocal = LocalTensor<float>(TPosition::VECCALC, 0, pitch);
    idxLocal = LocalTensor<int32_t>(TPosition::VECCALC, pitch * 4, pitch);
    bufferBase = pitch * 8;
    Vector(start, end);
    Flush(end - start);
  }
};
} // namespace OptimizedLayout

#include "tree_line_single_segment_vector.h"
