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
#include "tree_line_spatial.h"

template <typename T>
__aicore__ inline void SpatialTreeLine<T>::SpreadMortonAxis(LocalTensor<uint32_t> bits, LocalTensor<uint32_t> shifted, LocalTensor<uint32_t> bitsMask) {
    const uint32_t shifts[4] = {16, 8, 4, 2},
           masks[4] = {0x030000ffU, 0x0300f00fU, 0x030c30c3U, 0x09249249U};
    for (uint32_t phase = 0; phase < 4; ++phase) {
      ShiftLeft(shifted, bits, shifts[phase], n);
      Duplicate(bitsMask, masks[phase], n);
      PipeBarrier<PIPE_V>();
      Or(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(),
       shifted.ReinterpretCast<uint16_t>(), n * 2);
      PipeBarrier<PIPE_V>();
      And(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(),
        bitsMask.ReinterpretCast<uint16_t>(), n * 2);
      PipeBarrier<PIPE_V>();
    }
  }

template <typename T>
__aicore__ inline void SpatialTreeLine<T>::BuildMortonKeys(LocalTensor<float> keys, LocalTensor<float> idxf, LocalTensor<float> sortedPoints, LocalTensor<float> scratch, LocalTensor<float> praw) {
    Load(praw, pg[start * 3], n * 3, scratch);
    ReduceMin(keys, praw, scratch, n * 3, false);
    PipeBarrier<PIPE_V>();
    ReduceMax(keys[8], praw, scratch, n * 3, false);
    Fence<HardEvent::V_S>();
    float low = keys.GetValue(0), high = keys.GetValue(8),
       factor = high > low ? 1023.0f / (high - low) : 0.0f;
    Fence<HardEvent::S_V>();
    DeviceOffsets(idxf.ReinterpretCast<uint32_t>(), n, 12);
    auto code = keys.ReinterpretCast<uint32_t>(), bits = sortedPoints[k].ReinterpretCast<uint32_t>();
    auto shifted = scratch.ReinterpretCast<uint32_t>(),
      bitsMask = scratch[k].ReinterpretCast<uint32_t>();
    Duplicate(code, uint32_t(0), k);
    PipeBarrier<PIPE_V>();
    for (uint32_t axis = 0; axis < 3; ++axis) {
      Gather(sortedPoints, praw, idxf.ReinterpretCast<uint32_t>(), axis * 4, n);
      PipeBarrier<PIPE_V>();
      Adds(sortedPoints, sortedPoints, -low, n);
      PipeBarrier<PIPE_V>();
      Muls(sortedPoints, sortedPoints, factor, n);
      PipeBarrier<PIPE_V>();
      Maxs(sortedPoints, sortedPoints, 0.0f, n);
      PipeBarrier<PIPE_V>();
      Mins(sortedPoints, sortedPoints, 1023.0f, n);
      PipeBarrier<PIPE_V>();
      Cast(bits.ReinterpretCast<int32_t>(), sortedPoints, RoundMode::CAST_FLOOR, n);
      PipeBarrier<PIPE_V>();
      SpreadMortonAxis(bits, shifted, bitsMask);
      if (axis) {
        ShiftLeft(bits, bits, axis, n);
        PipeBarrier<PIPE_V>();
      }
      Or(code.ReinterpretCast<uint16_t>(), code.ReinterpretCast<uint16_t>(),
       bits.ReinterpretCast<uint16_t>(), n * 2);
      PipeBarrier<PIPE_V>();
    }
    Adds(code.ReinterpretCast<int32_t>(), code.ReinterpretCast<int32_t>(), int32_t(0x00800000), n);
    PipeBarrier<PIPE_V>();
  }

template <typename T>
__aicore__ inline void SpatialTreeLine<T>::FindCellEnds(LocalTensor<float> keys) {
    cellCount = 0;
    auto bitsKey = keys.ReinterpretCast<uint32_t>();
    auto endTmp = out.ReinterpretCast<int32_t>();
    for (uint32_t at = 0; at < n;) {
      uint32_t region = (bitsKey.GetValue(at) - 0x00800000U) >> cellShift, low = at + 1, high = n;
      while (low < high) {
        uint32_t mid = (low + high) / 2;
        if (((bitsKey.GetValue(mid) - 0x00800000U) >> cellShift) == region)
          low = mid + 1;
        else
          high = mid;
      }
      endTmp.SetValue(cellCount++, int32_t(low));
      at = low;
    }
    for (uint32_t i = cellCount; i < (cellCount + 7) / 8 * 8; ++i)
      endTmp.SetValue(i, int32_t(n));
    Fence<HardEvent::S_V>();
  }

template <typename T>
__aicore__ inline void SpatialTreeLine<T>::PointSort() {
    auto keys = pool, idxf = pool[k], sortedPoints = pool[k * 2], scratch = pool[k * 4],
      praw = pool[k * 6];
    BuildMortonKeys(keys, idxf, sortedPoints, scratch, praw);
    uint32_t aligned = (n + 7) / 8 * 8;
    if (aligned < k)
      Duplicate(keys[aligned], -3.402823466e38f, k - aligned);
    Fence<HardEvent::V_S>();
    for (uint32_t i = n; i < aligned; ++i)
      keys.SetValue(i, -3.402823466e38f);
    Fence<HardEvent::S_V>();
    DeviceOffsets(idxf.ReinterpretCast<uint32_t>(), k, 1);
    Sort<float, true>(sortedPoints, keys, idxf.ReinterpretCast<uint32_t>(), scratch, k / 32);
    PipeBarrier<PIPE_V>();
    Extract(keys, order.ReinterpretCast<uint32_t>(), sortedPoints, k / 32);
    PipeBarrier<PIPE_V>();
    if (keepPoints) {
      DataCopy(localOrder, order.ReinterpretCast<int32_t>(), k);
      PipeBarrier<PIPE_V>();
    } else {
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(orderG[start], order.ReinterpretCast<int32_t>(), {1, n * 4, 0, 0, 0});
      Fence<HardEvent::MTE3_V>();
    }
    Muls(idxf.ReinterpretCast<int32_t>(), order.ReinterpretCast<int32_t>(), 12, n);
    PipeBarrier<PIPE_V>();
    for (uint32_t a = 0; a < 3; ++a) {
      Gather(scratch, praw, idxf.ReinterpretCast<uint32_t>(), a * 4, n);
      if (keepPoints) {
        Adds(localPoints[a * pointPitch].ReinterpretCast<int32_t>(),
          scratch.ReinterpretCast<int32_t>(), int32_t(0), n);
        PipeBarrier<PIPE_V>();
      } else {
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(coordG[start * 3 + a * n], scratch, {1, n * 4, 0, 0, 0});
        Fence<HardEvent::MTE3_V>();
      }
    }
    Fence<HardEvent::V_S>();
    FindCellEnds(keys);
  }

template <typename T>
__aicore__ inline void SpatialTreeLine<T>::ReadPointBounds(uint32_t at, float *mn, float *mx) {
    Duplicate(group, 0.0f, 192);
    if (keepPoints) {
      for (uint32_t a = 0; a < 3; ++a)
        Gather(group[a * 64], localPoints[a * pointPitch], pointOffsets, at * 4, live);
    } else {
      Fence<HardEvent::V_MTE2>();
      for (uint32_t a = 0; a < 3; ++a)
        DataCopyPad(group[a * 64], coordG[start * 3 + a * n + at], {1, live * 4, 0, 0, 0},
              {false, 0, 0, 0});
      Fence<HardEvent::MTE2_V>();
    }
    for (uint32_t a = 0; a < 3; ++a) {
      WholeReduceMin(red[a * 16], group[a * 64], int32_t(live), 1, 1, 1, 8);
      WholeReduceMax(red[a * 16 + 8], group[a * 64], int32_t(live), 1, 1, 1, 8);
    }
    Fence<HardEvent::V_S>();
    for (uint32_t a = 0; a < 3; ++a) {
      mn[a] = red.GetValue(a * 16);
      mx[a] = red.GetValue(a * 16 + 8);
    }
    Fence<HardEvent::S_V>();
    if (outputPending) {
      WaitFlag<HardEvent::MTE3_V>(0);
      outputPending = false;
    }
    Brcb(broadcast, group, uint8_t((live + 7) / 8), {1, 8});
    Brcb(broadcast[512], group[64], uint8_t((live + 7) / 8), {1, 8});
    Brcb(broadcast[1024], group[128], uint8_t((live + 7) / 8), {1, 8});
  }

template <typename T>
__aicore__ inline void SpatialTreeLine<T>::RankSeparatingAxes(const float *mn, const float *mx) {
    // Segment-box SAT axes: coordinate axes above, then D cross each
    // coordinate axis. Every plane gap divided by its normal length is a
    // lower bound on every point-to-segment distance in this point box.
    LocalTensor<float> delta[3] = {x, y, z};
    for (uint32_t axis = 0; axis < 3; ++axis)
      Adds(delta[axis], raw[axis * mp], -(mn[axis] + mx[axis]) * 0.5f, rankCount);
    auto support = x[mp], absoluteDirection = x[mp * 2];
    for (uint32_t axis = 0; axis < 3; ++axis) {
      uint32_t other = (axis + 1) % 3, last = (axis + 2) % 3;
      Mul(dot, delta[other], raw[(last + 3) * mp], rankCount);
      Mul(tmp, delta[last], raw[(other + 3) * mp], rankCount);
      Sub(dot, dot, tmp, rankCount);
      Abs(dot, dot, rankCount);
      Muls(support, absoluteSegmentDirection[last], (mx[other] - mn[other]) * 0.5f,
           rankCount);
      Axpy(support, absoluteSegmentDirection[other], (mx[last] - mn[last]) * 0.5f,
           rankCount);
      Sub(dot, dot, support, rankCount);
      Maxs(dot, dot, 0.0f, rankCount);
      Mul(dot, dot, dot, rankCount);
      Mul(dot, dot, perpendicularInverse[axis], rankCount);
      Max(key, key, dot, rankCount);
    }
  }

template <typename T>
__aicore__ inline void SpatialTreeLine<T>::RankProjection(const float *mn, const float *mx) {
    // Project the point box onto a separating plane of each segment.
    auto norm = x[mp], support = x[mp * 2], aScratch = x[mp * 3];
    LocalTensor<float> delta[3] = {x, y, z};
    for (uint32_t axis = 0; axis < 3; ++axis) {
      Adds(delta[axis], raw[axis * mp], -(mn[axis] + mx[axis]) * 0.5f, rankCount);
      if (axis == 0)
        Mul(dot, delta[axis], raw[(axis + 3) * mp], rankCount);
      else
        MulAddDst(dot, delta[axis], raw[(axis + 3) * mp], rankCount);
    }
    Mul(dot, dot, recip, rankCount);
    Maxs(dot, dot, -1.0f, rankCount);
    Mins(dot, dot, 0.0f, rankCount);
    Muls(dot, dot, -1.0f, rankCount);
    // Only conservative candidate bounds use fused arithmetic; exact Q remains unchanged.
    for (uint32_t axis = 0; axis < 3; ++axis) {
      MulAddDst(delta[axis], raw[(axis + 3) * mp], dot, rankCount);
      if (axis == 0)
        Mul(norm, delta[axis], delta[axis], rankCount);
      else
        MulAddDst(norm, delta[axis], delta[axis], rankCount);
      Abs(tmp, delta[axis], rankCount);
      if (axis == 0)
        Muls(support, tmp, (mx[axis] - mn[axis]) * 0.5f, rankCount);
      else
        Axpy(support, tmp, (mx[axis] - mn[axis]) * 0.5f, rankCount);
    }
    Sub(dot, norm, support, rankCount);
    Maxs(dot, dot, 0.0f, rankCount);
    Mul(dot, dot, dot, rankCount);
    auto nonzero = red[128].ReinterpretCast<uint8_t>();
    CompareScalar(nonzero, norm, 0.0f, CMPMODE::GT, rankCount);
    Select(tmp, nonzero, norm, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, rankCount);
    Div(dot, dot, tmp, rankCount);
    Max(key, key, dot, rankCount);
  }

template <typename T>
__aicore__ inline void SpatialTreeLine<T>::Rank(uint32_t at) {
    float mn[3], mx[3];
    ReadPointBounds(at, mn, mx);
    for (uint32_t a = 0; a < 3; ++a) {
      Adds(x, lo[a], -mx[a], rankCount);
      Adds(y, hi[a], -mn[a], rankCount);
      Maxs(x, x, 0.0f, rankCount);
      Mins(y, y, 0.0f, rankCount);
      Add(x, x, y, rankCount);
      if (a == 0)
        Mul(key, x, x, rankCount);
      else
        MulAddDst(key, x, x, rankCount);
    }
    if (m >= kSatMinimumSegmentCount) {
      RankSeparatingAxes(mn, mx);
    } else {
      RankProjection(mn, mx);
    }
    Muls(key, key, -1.0f, rankCount);
    uint32_t aligned = (m + 7) / 8 * 8;
    if (aligned < mp)
      Duplicate(key[aligned], -3.402823466e38f, mp - aligned);
    Fence<HardEvent::V_S>();
    for (uint32_t i = m; i < aligned; ++i)
      key.SetValue(i, -3.402823466e38f);
    Fence<HardEvent::S_V>();
  }

template <typename T>
__aicore__ inline uint32_t SpatialTreeLine<T>::Compress(float threshold, bool second) {
    auto maskA = red[128].ReinterpretCast<uint8_t>(), maskB = red[192].ReinterpretCast<uint8_t>();
    if (second) {
      CompareScalar(maskA, key, -threshold, CMPMODE::GE, mp);
      CompareScalar(maskB, key, firstScore, CMPMODE::LT, mp);
      And(maskA.ReinterpretCast<uint16_t>(), maskA.ReinterpretCast<uint16_t>(),
        maskB.ReinterpretCast<uint16_t>(), mp / 16);
    } else {
      CompareScalar(maskA, key, threshold, CMPMODE::EQ, mp);
    }
    uint64_t count = 0;
    GatherMask(sorted, seq, maskA.ReinterpretCast<uint32_t>(), true, mp, {1, 1, 8, 1}, count);
    return uint32_t(count);
  }
