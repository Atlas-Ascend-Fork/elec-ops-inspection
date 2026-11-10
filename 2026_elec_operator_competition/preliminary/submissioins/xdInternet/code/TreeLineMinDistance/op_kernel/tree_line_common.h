/*
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

#ifndef TREE_LINE_COMMON_H
#define TREE_LINE_COMMON_H

#include "kernel_operator.h"
#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"

using namespace AscendC;

namespace TreeLineCommon {

__aicore__ inline uint32_t MinU(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint32_t Align64(uint32_t x)
{
    return (x + 63U) & ~63U;
}

__aicore__ inline uint32_t Align128(uint32_t x)
{
    return (x + 127U) & ~127U;
}

__aicore__ inline uint32_t SafeDivU(uint32_t num, uint32_t den)
{
    return (den > 0U) ? (num / den) : 0U;
}

__aicore__ inline uint32_t SafeModU(uint32_t num, uint32_t den)
{
    return (den > 0U) ? (num % den) : 0U;
}

__aicore__ inline void VBarrier()
{
    PipeBarrier<PIPE_V>();
}

template <HardEvent E>
__aicore__ inline void SyncStatic()
{
    constexpr event_t event = static_cast<event_t>(0);
    SetFlag<E>(event);
    WaitFlag<E>(event);
}

template <HardEvent E>
__aicore__ inline void SyncPipe(TPipe* pipe)
{
    const event_t event = static_cast<event_t>(pipe->FetchEventID(E));
    SetFlag<E>(event);
    WaitFlag<E>(event);
}

template <typename T>
__aicore__ inline void BindGlobalBuffersAndMask(
    GlobalTensor<T>& pointsGm, GlobalTensor<T>& segmentsGm,
    GlobalTensor<T>& distanceGm, GlobalTensor<int32_t>& indexGm,
    GM_ADDR points, GM_ADDR segments, GM_ADDR distances, GM_ADDR indices,
    uint32_t totalPts, uint32_t totalSegs)
{
    pointsGm.SetGlobalBuffer((__gm__ T*)points, totalPts * 3U);
    segmentsGm.SetGlobalBuffer((__gm__ T*)segments, totalSegs * 6U);
    distanceGm.SetGlobalBuffer((__gm__ T*)distances, totalPts);
    indexGm.SetGlobalBuffer((__gm__ int32_t*)indices, totalPts);
    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void CopyPointTileToQueue(
    TQue<QuePosition::VECIN, 2>& pointQueue,
    const GlobalTensor<T>& pointsGm, uint32_t pos, uint32_t count)
{
    auto local = pointQueue.AllocTensor<T>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(3U * count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(local, pointsGm[3U * pos], cp, pad);
    pointQueue.EnQue(local);
}

template <typename T>
__aicore__ inline void StoreQueuedOutputs(
    TQue<QuePosition::VECOUT, 1>& distanceQueue,
    TQue<QuePosition::VECOUT, 1>& indexQueue,
    const GlobalTensor<T>& distanceGm, const GlobalTensor<int32_t>& indexGm,
    LocalTensor<T> d, LocalTensor<int32_t> idx, uint32_t pos, uint32_t count)
{
    distanceQueue.EnQue(d);
    indexQueue.EnQue(idx);
    d = distanceQueue.DeQue<T>();
    idx = indexQueue.DeQue<int32_t>();
    DataCopyExtParams dc{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyExtParams ic{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(distanceGm[pos], d, dc);
    DataCopyPad(indexGm[pos], idx, ic);
    distanceQueue.FreeTensor(d);
    indexQueue.FreeTensor(idx);
}

__aicore__ inline void ExpandStaticIndex64(LocalTensor<int32_t> dst)
{
    #pragma unroll
    for (uint32_t i = 0U; i < 8U; ++i) {
        dst.SetValue(i, static_cast<int32_t>(i));
    }
    SyncStatic<HardEvent::S_V>();
    Adds(dst[8U], dst, static_cast<int32_t>(8), 8U);
    VBarrier();
    Adds(dst[16U], dst, static_cast<int32_t>(16), 16U);
    VBarrier();
    Adds(dst[32U], dst, static_cast<int32_t>(32), 32U);
    VBarrier();
}

__aicore__ inline void GatherSegmentAoS(
    LocalTensor<float> ax, LocalTensor<float> ay, LocalTensor<float> az,
    LocalTensor<float> dx, LocalTensor<float> dy, LocalTensor<float> dz,
    LocalTensor<float> aos, LocalTensor<uint32_t> off, uint32_t count)
{
    Gather(ax, aos, off, 0U, count);
    Gather(ay, aos, off, 4U, count);
    Gather(az, aos, off, 8U, count);
    Gather(dx, aos, off, 12U, count);
    Gather(dy, aos, off, 16U, count);
    Gather(dz, aos, off, 20U, count);
}

__aicore__ inline void GatherPointAoS(
    LocalTensor<float> px, LocalTensor<float> py, LocalTensor<float> pz,
    LocalTensor<float> ptFloat, LocalTensor<uint32_t> off12, uint32_t count)
{
    Gather(px, ptFloat, off12, 0U, count);
    Gather(py, ptFloat, off12, 4U, count);
    Gather(pz, ptFloat, off12, 8U, count);
}

__aicore__ inline void BroadcastPointTriplet(
    LocalTensor<float> pxBcast, LocalTensor<float> pyBcast, LocalTensor<float> pzBcast,
    LocalTensor<float> px, LocalTensor<float> py, LocalTensor<float> pz,
    uint8_t brcbRepeats)
{
    const BrcbRepeatParams brcbParams{1, 8};
    Brcb(pxBcast, px, brcbRepeats, brcbParams);
    Brcb(pyBcast, py, brcbRepeats, brcbParams);
    Brcb(pzBcast, pz, brcbRepeats, brcbParams);
}

__aicore__ inline void SumSquares3(
    LocalTensor<float> dst, LocalTensor<float> x,
    LocalTensor<float> y, LocalTensor<float> z, uint8_t repeats)
{
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    Mul<float, false>(dst, x, x, MASK_PLACEHOLDER, repeats, binary);
    VBarrier();
    MulAddDst<float, float, false>(dst, y, y, MASK_PLACEHOLDER, repeats, binary);
    VBarrier();
    MulAddDst<float, float, false>(dst, z, z, MASK_PLACEHOLDER, repeats, binary);
    VBarrier();
}

__aicore__ inline void ComputeSegmentNegMu(
    LocalTensor<float> ax, LocalTensor<float> ay, LocalTensor<float> az,
    LocalTensor<float> dx, LocalTensor<float> dy, LocalTensor<float> dz,
    LocalTensor<float> negMux, LocalTensor<float> negMuy, LocalTensor<float> negMuz,
    LocalTensor<float> invLen, LocalTensor<float> negOne, uint32_t mp)
{
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const UnaryRepeatParams unary{1, 1, 8, 8};
    const uint8_t repeats = static_cast<uint8_t>(mp >> 6U);
    ResetMask();
    Sub<float, false>(dx, dx, ax, MASK_PLACEHOLDER, repeats, binary);
    Sub<float, false>(dy, dy, ay, MASK_PLACEHOLDER, repeats, binary);
    Sub<float, false>(dz, dz, az, MASK_PLACEHOLDER, repeats, binary);
    VBarrier();
    Mul<float, false>(invLen, dx, dx, MASK_PLACEHOLDER, repeats, binary);
    Mul<float, false>(negMux, dy, dy, MASK_PLACEHOLDER, repeats, binary);
    Mul<float, false>(negMuy, dz, dz, MASK_PLACEHOLDER, repeats, binary);
    VBarrier();
    Add<float, false>(invLen, invLen, negMux, MASK_PLACEHOLDER, repeats, binary);
    VBarrier();
    Add<float, false>(invLen, invLen, negMuy, MASK_PLACEHOLDER, repeats, binary);
    VBarrier();
    Maxs<float, false>(invLen, invLen, 1.0e-30f, MASK_PLACEHOLDER, repeats, unary);
    Duplicate(negOne, -1.0f, mp);
    VBarrier();
    Div(invLen, negOne, invLen, mp);
    VBarrier();
    ResetMask();
    Mul<float, false>(negMuy, dy, invLen, MASK_PLACEHOLDER, repeats, binary);
    Mul<float, false>(negMuz, dz, invLen, MASK_PLACEHOLDER, repeats, binary);
    Mul<float, false>(negMux, dx, invLen, MASK_PLACEHOLDER, repeats, binary);
    VBarrier();
}

__aicore__ inline void ReadSegment6(
    LocalTensor<float> values, uint32_t off,
    float& ax, float& ay, float& az, float& dx, float& dy, float& dz)
{
    ax = values.GetValue(off);
    ay = values.GetValue(off + 1U);
    az = values.GetValue(off + 2U);
    dx = values.GetValue(off + 3U) - ax;
    dy = values.GetValue(off + 4U) - ay;
    dz = values.GetValue(off + 5U) - az;
}

__aicore__ inline void EvalSingleSegmentCore(
    LocalTensor<float> apx, LocalTensor<float> apy, LocalTensor<float> apz,
    LocalTensor<float> px, LocalTensor<float> tz, LocalTensor<float> aosSeg,
    uint8_t repeats)
{
    float ax0 = 0.0f, ay0 = 0.0f, az0 = 0.0f, dx0 = 0.0f, dy0 = 0.0f, dz0 = 0.0f;
    ReadSegment6(aosSeg, 0U, ax0, ay0, az0, dx0, dy0, dz0);
    const float len = (dx0 * dx0 + dy0 * dy0) + dz0 * dz0;
    const float inv = (len > 0.0f) ? (1.0f / len) : 0.0f;
    const UnaryRepeatParams unary{1, 1, 8, 8};
    VBarrier();
    ResetMask();
    Adds<float, false>(apx, apx, -ax0, MASK_PLACEHOLDER, repeats, unary);
    Adds<float, false>(apy, apy, -ay0, MASK_PLACEHOLDER, repeats, unary);
    Adds<float, false>(apz, apz, -az0, MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    Muls<float, false>(px, apx, dx0 * inv, MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    Axpy<float, float, false>(px, apy, dy0 * inv, MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    Axpy<float, float, false>(px, apz, dz0 * inv, MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    Maxs<float, false>(tz, px, 0.0f, MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    Mins<float, false>(px, tz, 1.0f, MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    Axpy<float, float, false>(apx, px, -dx0, MASK_PLACEHOLDER, repeats, unary);
    Axpy<float, float, false>(apy, px, -dy0, MASK_PLACEHOLDER, repeats, unary);
    Axpy<float, float, false>(apz, px, -dz0, MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    SumSquares3(tz, apx, apy, apz, repeats);
}

template <typename T>
__aicore__ inline void FinishSqrtAndCast(
    LocalTensor<T> outDistT, LocalTensor<float> outDist32,
    LocalTensor<float> srcSq, uint32_t count)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        Sqrt(outDistT.template ReinterpretCast<float>(), srcSq, count);
    } else {
        Sqrt(outDist32, srcSq, count);
        VBarrier();
        Cast(outDistT, outDist32, RoundMode::CAST_RINT, count);
    }
}

__aicore__ inline void EvalBlockStepProjectAndResidual(
    LocalTensor<float> x0, LocalTensor<float> y0, LocalTensor<float> z0,
    LocalTensor<float> t0, LocalTensor<float> t1,
    LocalTensor<float> u0, LocalTensor<float> v0, LocalTensor<float> pX,
    LocalTensor<float> dx, LocalTensor<float> dy, LocalTensor<float> dz,
    LocalTensor<float> negMux, LocalTensor<float> negMuy, LocalTensor<float> negMuz,
    uint32_t segOff, uint8_t stepRepeats)
{
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const BinaryRepeatParams segBinary{1, 1, 1, 8, 8, 0};
    const UnaryRepeatParams unary{1, 1, 8, 8};
    Mul<float, false>(t1, x0, negMux[segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
    Mul<float, false>(u0, y0, negMuy[segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
    Mul<float, false>(v0, z0, negMuz[segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
    VBarrier();
    Add<float, false>(t0, t1, u0, MASK_PLACEHOLDER, stepRepeats, binary);
    VBarrier();
    Add<float, false>(t0, t0, v0, MASK_PLACEHOLDER, stepRepeats, binary);
    VBarrier();
    Maxs<float, false>(t1, t0, 0.0f, MASK_PLACEHOLDER, stepRepeats, unary);
    VBarrier();
    Mins<float, false>(t0, t1, 1.0f, MASK_PLACEHOLDER, stepRepeats, unary);
    VBarrier();
    Mul<float, false>(pX, t0, dx[segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
    Mul<float, false>(u0, t0, dy[segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
    Mul<float, false>(v0, t0, dz[segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
    VBarrier();
    Add<float, false>(pX, x0, pX, MASK_PLACEHOLDER, stepRepeats, binary);
    Add<float, false>(y0, y0, u0, MASK_PLACEHOLDER, stepRepeats, binary);
    Add<float, false>(v0, z0, v0, MASK_PLACEHOLDER, stepRepeats, binary);
    VBarrier();
}

__aicore__ inline void SubBcastPointBlocks(
    LocalTensor<float> x0, LocalTensor<float> y0, LocalTensor<float> z0,
    LocalTensor<float> ax, LocalTensor<float> ay, LocalTensor<float> az,
    LocalTensor<float> pxBcast, LocalTensor<float> pyBcast, LocalTensor<float> pzBcast,
    uint32_t bStart, uint32_t numBlk, uint32_t blockFloats, uint8_t stepRepeats)
{
    const BinaryRepeatParams bcastBinary{1, 1, 0, 8, 0, 1};
    for (uint32_t k = 0U; k < numBlk; ++k) {
        const uint32_t off = k * blockFloats;
        const uint32_t segOff = (bStart + k) * 64U;
        Sub<float, false>(x0[off], ax[segOff], pxBcast, MASK_PLACEHOLDER, stepRepeats, bcastBinary);
        Sub<float, false>(y0[off], ay[segOff], pyBcast, MASK_PLACEHOLDER, stepRepeats, bcastBinary);
        Sub<float, false>(z0[off], az[segOff], pzBcast, MASK_PLACEHOLDER, stepRepeats, bcastBinary);
    }
    VBarrier();
}

__aicore__ inline void EvalSplitHeavySubAndDot(
    LocalTensor<float> w,
    LocalTensor<float> ax, LocalTensor<float> ay, LocalTensor<float> az,
    LocalTensor<float> negMux, LocalTensor<float> negMuy, LocalTensor<float> negMuz,
    LocalTensor<float> pxBcast, LocalTensor<float> pyBcast, LocalTensor<float> pzBcast,
    uint32_t bStart, uint32_t numBlk, uint32_t cPad)
{
    const uint32_t blockFloats = cPad * 64U;
    const uint8_t stepRepeats = static_cast<uint8_t>(cPad);
    const uint8_t totalRepeats = static_cast<uint8_t>(numBlk * cPad);
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const BinaryRepeatParams segBinary{1, 1, 1, 8, 8, 0};
    const UnaryRepeatParams unary{1, 1, 8, 8};
    SubBcastPointBlocks(
        w[16448U], w[0U], w[20544U], ax, ay, az,
        pxBcast, pyBcast, pzBcast, bStart, numBlk, blockFloats, stepRepeats);
    for (uint32_t k = 0U; k < numBlk; ++k) {
        const uint32_t off = k * blockFloats;
        const uint32_t segOff = (bStart + k) * 64U;
        Mul<float, false>(w[4096U + off], w[16448U + off], negMux[segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
        Mul<float, false>(w[28736U + off], w[0U + off], negMuy[segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
        Mul<float, false>(w[8192U + off], w[20544U + off], negMuz[segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
    }
    VBarrier();
    Add<float, false>(w[24640U], w[4096U], w[28736U], MASK_PLACEHOLDER, totalRepeats, binary);
    VBarrier();
    Add<float, false>(w[24640U], w[24640U], w[8192U], MASK_PLACEHOLDER, totalRepeats, binary);
    VBarrier();
    Maxs<float, false>(w[4096U], w[24640U], 0.0f, MASK_PLACEHOLDER, totalRepeats, unary);
    VBarrier();
    Mins<float, false>(w[24640U], w[4096U], 1.0f, MASK_PLACEHOLDER, totalRepeats, unary);
    VBarrier();
}

__aicore__ inline void EvalSplitHeavyResidualAndReduce(
    LocalTensor<float> w, LocalTensor<float> pairs,
    LocalTensor<float> dx, LocalTensor<float> dy, LocalTensor<float> dz,
    uint32_t bStart, uint32_t numBlk, uint32_t count, uint32_t cPad, int32_t dstStride)
{
    const uint32_t blockFloats = cPad * 64U;
    const uint8_t stepRepeats = static_cast<uint8_t>(cPad);
    const uint8_t totalRepeats = static_cast<uint8_t>(numBlk * cPad);
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const BinaryRepeatParams segBinary{1, 1, 1, 8, 8, 0};
    for (uint32_t k = 0U; k < numBlk; ++k) {
        const uint32_t off = k * blockFloats;
        const uint32_t segOff = (bStart + k) * 64U;
        Mul<float, false>(w[12288U + off], w[24640U + off], dx[segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
        Mul<float, false>(w[28736U + off], w[24640U + off], dy[segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
        Mul<float, false>(w[8192U + off], w[24640U + off], dz[segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
    }
    VBarrier();
    Add<float, false>(w[12288U], w[16448U], w[12288U], MASK_PLACEHOLDER, totalRepeats, binary);
    Add<float, false>(w[0U], w[0U], w[28736U], MASK_PLACEHOLDER, totalRepeats, binary);
    Add<float, false>(w[8192U], w[20544U], w[8192U], MASK_PLACEHOLDER, totalRepeats, binary);
    VBarrier();
    SumSquares3(w[24640U], w[12288U], w[0U], w[8192U], totalRepeats);
    for (uint32_t k = 0U; k < numBlk; ++k) {
        WholeReduceMin<float>(
            pairs[2U * (bStart + k)], w[24640U + k * blockFloats],
            64, static_cast<int32_t>(count),
            dstStride, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
    }
}

template <typename T, bool ScaleOff32By4>
__aicore__ inline void DecodeHierarchicalMin(
    LocalTensor<T> outDistT, LocalTensor<int32_t> outIdx,
    LocalTensor<float> outDist32, LocalTensor<float> r, LocalTensor<float> pairs,
    LocalTensor<int32_t> idxByteOff, LocalTensor<int32_t> localIdxVec,
    LocalTensor<uint32_t> off32, uint32_t numBlocks, uint32_t count, int32_t srcStride)
{
    auto off32Int = off32.ReinterpretCast<int32_t>();
    uint64_t valueMask[2] = {
        0x5555555555555555ULL & ((1ULL << (2U * numBlocks)) - 1ULL), 0ULL};
    WholeReduceMin<float>(
        r, pairs, valueMask, static_cast<int32_t>(count),
        4, 1, srcStride, ReduceOrder::ORDER_VALUE_INDEX);
    VBarrier();
    Gather(outDist32, r, off32, 0U, count);
    Gather(outIdx, r.ReinterpretCast<int32_t>(), off32, 4U, count);
    if constexpr (ScaleOff32By4) {
        Muls(localIdxVec, off32Int, static_cast<int32_t>(4), count);
    }
    VBarrier();
    if constexpr (sizeof(T) == sizeof(float)) {
        Sqrt(outDistT.template ReinterpretCast<float>(), outDist32, count);
    } else {
        Sqrt(outDist32, outDist32, count);
    }
    Muls(idxByteOff, outIdx, static_cast<int32_t>(4), count);
    Muls(outIdx, outIdx, static_cast<int32_t>(32), count);
    VBarrier();
    Add(idxByteOff, idxByteOff, ScaleOff32By4 ? localIdxVec : off32Int, count);
    VBarrier();
    Gather(localIdxVec, pairs.ReinterpretCast<int32_t>(),
        idxByteOff.ReinterpretCast<uint32_t>(), 4U, count);
    VBarrier();
    Add(outIdx, outIdx, localIdxVec, count);
    if constexpr (sizeof(T) != sizeof(float)) {
        Cast(outDistT, outDist32, RoundMode::CAST_RINT, count);
    }
}

}  // namespace TreeLineCommon

#endif  // TREE_LINE_COMMON_H
