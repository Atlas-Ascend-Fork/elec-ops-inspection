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

#ifndef TREE_LINE_GENERAL_COMPUTE_H
#define TREE_LINE_GENERAL_COMPUTE_H

#include "tree_line_general_modes.h"

template <typename T, uint32_t MODE>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::EvalQuadSegmentsMode0(
    LocalTensor<float> px, LocalTensor<float> py, LocalTensor<float> pz,
    LocalTensor<float> best, LocalTensor<float> bestIdx,
    uint32_t j, uint32_t k, uint8_t repeats)
{
    using namespace TreeLineCommon;
    auto w = scratch_.Get<float>();
    auto s = segmentSoa_.Get<float>();
    const uint32_t maskBytes = ((width_ + 255U) / 256U) * 32U;
    auto maskBase = mask_.Get<uint8_t>();
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const UnaryRepeatParams unary{1, 1, 8, 8};
    LocalTensor<float> apx[4], apy[4], apz[4], t[4];
    float dx[4], dy[4], dz[4], mux[4], muy[4], muz[4];
    #pragma unroll
    for (uint32_t r = 0U; r < 4U; ++r) {
        const uint32_t base = (4U * r) * width_ + r * 128U;
        apx[r] = w[base];
        apy[r] = w[base + width_];
        apz[r] = w[base + 2U * width_];
        t[r] = w[base + 3U * width_ + 64U];
        const uint32_t idx = j + r;
        dz[r] = s.GetValue(3U * soaStride_ + idx);
        dx[r] = s.GetValue(4U * soaStride_ + idx);
        dy[r] = s.GetValue(5U * soaStride_ + idx);
        muz[r] = s.GetValue(6U * soaStride_ + idx);
        mux[r] = s.GetValue(7U * soaStride_ + idx);
        muy[r] = s.GetValue(8U * soaStride_ + idx);
        Adds<float, false>(apx[r], px, -s.GetValue(idx), MASK_PLACEHOLDER, repeats, unary);
        Adds<float, false>(apy[r], py, -s.GetValue(soaStride_ + idx), MASK_PLACEHOLDER, repeats, unary);
        Adds<float, false>(apz[r], pz, -s.GetValue(2U * soaStride_ + idx), MASK_PLACEHOLDER, repeats, unary);
    }
    VBarrier();
    FinishQuadSegmentsMode0(apx, apy, apz, t, dx, dy, dz, mux, muy, muz,
        maskBase, maskBytes, best, bestIdx, j, k, repeats, binary, unary);
}

template <typename T, uint32_t MODE>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::FinishQuadSegmentsMode0(
    LocalTensor<float> apx[4], LocalTensor<float> apy[4],
    LocalTensor<float> apz[4], LocalTensor<float> t[4],
    const float dx[4], const float dy[4], const float dz[4],
    const float mux[4], const float muy[4], const float muz[4],
    LocalTensor<uint8_t> maskBase, uint32_t maskBytes,
    LocalTensor<float> best, LocalTensor<float> bestIdx,
    uint32_t j, uint32_t k, uint8_t repeats,
    const BinaryRepeatParams& binary, const UnaryRepeatParams& unary)
{
    using namespace TreeLineCommon;
    #pragma unroll
    for (uint32_t r = 0U; r < 4U; ++r) {
        Muls<float, false>(t[r], apx[r], mux[r], MASK_PLACEHOLDER, repeats, unary);
    }
    VBarrier();
    #pragma unroll
    for (uint32_t r = 0U; r < 4U; ++r) {
        Axpy<float, float, false>(t[r], apy[r], muy[r], MASK_PLACEHOLDER, repeats, unary);
    }
    VBarrier();
    #pragma unroll
    for (uint32_t r = 0U; r < 4U; ++r) {
        Axpy<float, float, false>(t[r], apz[r], muz[r], MASK_PLACEHOLDER, repeats, unary);
    }
    VBarrier();
    #pragma unroll
    for (uint32_t r = 0U; r < 4U; ++r) {
        Maxs<float, false>(t[r], t[r], 0.0f, MASK_PLACEHOLDER, repeats, unary);
    }
    VBarrier();
    #pragma unroll
    for (uint32_t r = 0U; r < 4U; ++r) {
        Mins<float, false>(t[r], t[r], 1.0f, MASK_PLACEHOLDER, repeats, unary);
    }
    VBarrier();
    #pragma unroll
    for (uint32_t r = 0U; r < 4U; ++r) {
        Axpy<float, float, false>(apx[r], t[r], -dx[r], MASK_PLACEHOLDER, repeats, unary);
        Axpy<float, float, false>(apy[r], t[r], -dy[r], MASK_PLACEHOLDER, repeats, unary);
        Axpy<float, float, false>(apz[r], t[r], -dz[r], MASK_PLACEHOLDER, repeats, unary);
    }
    VBarrier();
    ReduceQuadSegmentsMode0(apx, apy, apz, t, maskBase, maskBytes, best, bestIdx, j, k, repeats, binary);
}

template <typename T, uint32_t MODE>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::ReduceQuadSegmentsMode0(
    LocalTensor<float> apx[4], LocalTensor<float> apy[4],
    LocalTensor<float> apz[4], LocalTensor<float> t[4],
    LocalTensor<uint8_t> maskBase, uint32_t maskBytes,
    LocalTensor<float> best, LocalTensor<float> bestIdx,
    uint32_t j, uint32_t k, uint8_t repeats, const BinaryRepeatParams& binary)
{
    using namespace TreeLineCommon;
    #pragma unroll
    for (uint32_t r = 0U; r < 4U; ++r) {
        Mul<float, false>(t[r], apx[r], apx[r], MASK_PLACEHOLDER, repeats, binary);
    }
    VBarrier();
    #pragma unroll
    for (uint32_t r = 0U; r < 4U; ++r) {
        MulAddDst<float, float, false>(t[r], apy[r], apy[r], MASK_PLACEHOLDER, repeats, binary);
    }
    VBarrier();
    #pragma unroll
    for (uint32_t r = 0U; r < 4U; ++r) {
        MulAddDst<float, float, false>(t[r], apz[r], apz[r], MASK_PLACEHOLDER, repeats, binary);
    }
    VBarrier();
    #pragma unroll
    for (uint32_t r = 0U; r < 4U; ++r) {
        auto maskR = maskBase[r * maskBytes];
        Compare(maskR, t[r], best, CMPMODE::GE, k);
        VBarrier();
        Select(bestIdx, maskR, bestIdx,
            static_cast<float>(static_cast<int32_t>(j + r)),
            SELMODE::VSEL_TENSOR_SCALAR_MODE, k);
        Min(best, best, t[r], k);
        VBarrier();
    }
    ResetMask();
}

template <typename T, uint32_t MODE>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::EvalSingleSegmentTailMode0(
    LocalTensor<float> px, LocalTensor<float> py, LocalTensor<float> pz,
    LocalTensor<float> best, LocalTensor<float> bestIdx,
    uint32_t j, uint32_t k, uint8_t repeats)
{
    using namespace TreeLineCommon;
    auto w = scratch_.Get<float>();
    auto s = segmentSoa_.Get<float>();
    auto apx0 = w[0U];
    auto apy0 = w[width_];
    auto apz0 = w[2U * width_];
    auto t0 = w[3U * width_ + 64U];
    const UnaryRepeatParams unary{1, 1, 8, 8};
    Adds<float, false>(apx0, px, -s.GetValue(j), MASK_PLACEHOLDER, repeats, unary);
    Adds<float, false>(apy0, py, -s.GetValue(soaStride_ + j), MASK_PLACEHOLDER, repeats, unary);
    Adds<float, false>(apz0, pz, -s.GetValue(2U * soaStride_ + j), MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    Muls<float, false>(t0, apx0, s.GetValue(7U * soaStride_ + j), MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    Axpy<float, float, false>(t0, apy0, s.GetValue(8U * soaStride_ + j), MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    Axpy<float, float, false>(t0, apz0, s.GetValue(6U * soaStride_ + j), MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    Maxs<float, false>(t0, t0, 0.0f, MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    Mins<float, false>(t0, t0, 1.0f, MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    Axpy<float, float, false>(apx0, t0, -s.GetValue(4U * soaStride_ + j), MASK_PLACEHOLDER, repeats, unary);
    Axpy<float, float, false>(apy0, t0, -s.GetValue(5U * soaStride_ + j), MASK_PLACEHOLDER, repeats, unary);
    Axpy<float, float, false>(apz0, t0, -s.GetValue(3U * soaStride_ + j), MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    SumSquares3(t0, apx0, apy0, apz0, repeats);
    if (m_ > 1U) {
        auto mask0 = mask_.Get<uint8_t>();
        Compare(mask0, t0, best, CMPMODE::GE, k);
        VBarrier();
        Select(bestIdx, mask0, bestIdx,
            static_cast<float>(static_cast<int32_t>(j)),
            SELMODE::VSEL_TENSOR_SCALAR_MODE, k);
        Min(best, best, t0, k);
        VBarrier();
        ResetMask();
    }
}

template <typename T, uint32_t MODE>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::ComputePoints(
    LocalTensor<float> aos, uint32_t pos, uint32_t count)
{
    using namespace TreeLineCommon;
    const uint32_t k = Align64(count);
    const uint8_t repeats = static_cast<uint8_t>(k >> 6U);
    auto p = pointSoa_.Get<float>();
    auto px = p[64U];
    auto py = p[width_ + 64U];
    auto pz = p[2U * width_ + 64U];
    GatherPointAoS(px, py, pz, aos, offsets_.Get<uint32_t>(), count);
    auto best = results_.Get<float>();
    auto bestIdx = best[tile_];
    if (m_ > 1U) {
        Duplicate(best, 3.402823466e+38F, k);
        Duplicate(bestIdx, 0.0f, k);
    }
    VBarrier();
    ResetMask();
    uint32_t j = 0U;
    for (; j + 4U <= m_; j += 4U) {
        EvalQuadSegmentsMode0(px, py, pz, best, bestIdx, j, k, repeats);
    }
    for (; j < m_; ++j) {
        EvalSingleSegmentTailMode0(px, py, pz, best, bestIdx, j, k, repeats);
    }
    if (m_ == 1U) {
        best = scratch_.Get<float>()[3U * width_ + 64U];
    }
    FinishComputePointsOutput(best, bestIdx, pos, count, k);
}

template <typename T, uint32_t MODE>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::FinishComputePointsOutput(
    LocalTensor<float> best, LocalTensor<float> bestIdx,
    uint32_t pos, uint32_t count, uint32_t k)
{
    using namespace TreeLineCommon;
    auto d = distanceQueue_.AllocTensor<T>();
    auto idx = indexQueue_.AllocTensor<int32_t>();
    if constexpr (sizeof(T) == sizeof(float)) {
        Sqrt(d.template ReinterpretCast<float>(), best, k);
        if (m_ == 1U) {
            Duplicate(idx, static_cast<int32_t>(0), k);
        } else {
            Cast(idx, bestIdx, RoundMode::CAST_RINT, k);
        }
    } else {
        Sqrt(best, best, k);
        if (m_ == 1U) {
            Duplicate(idx, static_cast<int32_t>(0), k);
        } else {
            Cast(idx, bestIdx, RoundMode::CAST_RINT, k);
        }
        VBarrier();
        Cast(d, best, RoundMode::CAST_RINT, k);
    }
    StoreQueuedOutputs(distanceQueue_, indexQueue_, distanceGm_, indexGm_, d, idx, pos, count);
}

template <typename T, uint32_t MODE>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::StepSingleBlockMode1(
    LocalTensor<float> w, LocalTensor<float> s, LocalTensor<float> bcast,
    LocalTensor<float> pairs, uint32_t b, uint32_t numBlocks,
    uint32_t count, uint8_t stepRepeats)
{
    using namespace TreeLineCommon;
    const uint32_t segOff = b * 64U;
    const uint32_t highOff = 3U * soaStride_ + 64U;
    const uint32_t muxOff = 6U * soaStride_ + 128U;
    EvalBlockStepProjectAndResidual(
        w[16448U], w[0U], w[20544U], w[24640U], w[4096U], w[28736U], w[8192U], w[12288U],
        s[highOff + soaStride_], s[highOff + 2U * soaStride_], s[highOff],
        s[muxOff + soaStride_], s[muxOff + 2U * soaStride_], s[muxOff], segOff, stepRepeats);
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const BinaryRepeatParams bcastBinary{1, 1, 0, 8, 0, 1};
    const uint32_t nextSegOff = (b + 1U) * 64U;
    auto t0 = w[24640U];
    Mul<float, false>(t0, w[12288U], w[12288U], MASK_PLACEHOLDER, stepRepeats, binary);
    if (b + 1U < numBlocks) {
        Sub<float, false>(w[16448U], s[nextSegOff], bcast[192U], MASK_PLACEHOLDER, stepRepeats, bcastBinary);
    }
    VBarrier();
    MulAddDst<float, float, false>(t0, w[0U], w[0U], MASK_PLACEHOLDER, stepRepeats, binary);
    if (b + 1U < numBlocks) {
        Sub<float, false>(w[20544U], s[2U * soaStride_ + nextSegOff], bcast[1216U], MASK_PLACEHOLDER, stepRepeats, bcastBinary);
    }
    VBarrier();
    MulAddDst<float, float, false>(t0, w[8192U], w[8192U], MASK_PLACEHOLDER, stepRepeats, binary);
    if (b + 1U < numBlocks) {
        Sub<float, false>(w[0U], s[soaStride_ + nextSegOff], bcast[704U], MASK_PLACEHOLDER, stepRepeats, bcastBinary);
    }
    VBarrier();
    WholeReduceMin<float>(
        pairs[2U * b], t0, 64, static_cast<int32_t>(count),
        16, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
    ResetMask();
}

template <typename T, uint32_t MODE>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::ComputeSegmentsBlocks1(
    uint32_t count, uint32_t cPad)
{
    using namespace TreeLineCommon;
    auto s = segmentSoa_.Get<float>();
    auto bcast = pointSoa_.Get<float>();
    auto w = scratch_.Get<float>();
    auto pairs = reduceWork_.Get<float>();
    const uint32_t numBlocks = mp_ >> 6U;
    const uint8_t stepRepeats = static_cast<uint8_t>(cPad);
    ResetMask();
    SubBcastPointBlocks(
        w[16448U], w[0U], w[20544U], s[0U], s[soaStride_], s[2U * soaStride_],
        bcast[192U], bcast[704U], bcast[1216U], 0U, 1U, 0U, stepRepeats);
    for (uint32_t b = 0U; b < numBlocks; ++b) {
        StepSingleBlockMode1(w, s, bcast, pairs, b, numBlocks, count, stepRepeats);
    }
}

template <typename T, uint32_t MODE>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::EvalCompactBlockGroup(
    uint32_t bStart, uint32_t numBlk, uint32_t count, uint32_t cPad)
{
    using namespace TreeLineCommon;
    auto s = segmentSoa_.Get<float>();
    auto bcast = pointSoa_.Get<float>();
    auto w = scratch_.Get<float>();
    const uint32_t highBase = 3U * soaStride_ + 64U;
    const uint32_t muxBase = 6U * soaStride_ + 128U;
    const uint32_t rawGrp = groupStride_ - 64U;
    auto x0 = w[0U * rawGrp];
    auto y0 = w[1U * rawGrp];
    auto z0 = w[2U * rawGrp];
    auto t0 = w[3U * rawGrp + 64U];
    if (m_ < 64U) {
        ResetMask();
    }
    const uint32_t blockFloats = cPad * 64U;
    const uint8_t stepRepeats = static_cast<uint8_t>(cPad);
    SubBcastPointBlocks(
        x0, y0, z0, s[0U], s[soaStride_], s[2U * soaStride_],
        bcast[192U], bcast[704U], bcast[1216U], bStart, numBlk, blockFloats, stepRepeats);
    FinishCompactBlockGroup(x0, y0, z0, t0, s, highBase, muxBase,
        bStart, numBlk, count, blockFloats, stepRepeats, static_cast<uint8_t>(numBlk * cPad));
}

template <typename T, uint32_t MODE>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::FinishCompactBlockGroup(
    LocalTensor<float> x0, LocalTensor<float> y0, LocalTensor<float> z0, LocalTensor<float> t0,
    LocalTensor<float> s, uint32_t highBase, uint32_t muxBase,
    uint32_t bStart, uint32_t numBlk, uint32_t count, uint32_t blockFloats,
    uint8_t stepRepeats, uint8_t totalRepeats)
{
    using namespace TreeLineCommon;
    const BinaryRepeatParams segBinary{1, 1, 1, 8, 8, 0};
    const UnaryRepeatParams unary{1, 1, 8, 8};
    for (uint32_t k = 0U; k < numBlk; ++k) {
        Mul<float, false>(t0[k * blockFloats], x0[k * blockFloats],
            s[muxBase + soaStride_ + (bStart + k) * 64U], MASK_PLACEHOLDER, stepRepeats, segBinary);
    }
    VBarrier();
    for (uint32_t k = 0U; k < numBlk; ++k) {
        MulAddDst<float, float, false>(t0[k * blockFloats], y0[k * blockFloats],
            s[muxBase + 2U * soaStride_ + (bStart + k) * 64U], MASK_PLACEHOLDER, stepRepeats, segBinary);
    }
    VBarrier();
    for (uint32_t k = 0U; k < numBlk; ++k) {
        MulAddDst<float, float, false>(t0[k * blockFloats], z0[k * blockFloats],
            s[muxBase + (bStart + k) * 64U], MASK_PLACEHOLDER, stepRepeats, segBinary);
    }
    VBarrier();
    Maxs<float, false>(t0, t0, 0.0f, MASK_PLACEHOLDER, totalRepeats, unary);
    VBarrier();
    Mins<float, false>(t0, t0, 1.0f, MASK_PLACEHOLDER, totalRepeats, unary);
    VBarrier();
    for (uint32_t k = 0U; k < numBlk; ++k) {
        const uint32_t off = k * blockFloats;
        const uint32_t segOff = (bStart + k) * 64U;
        MulAddDst<float, float, false>(x0[off], t0[off], s[highBase + soaStride_ + segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
        MulAddDst<float, float, false>(y0[off], t0[off], s[highBase + 2U * soaStride_ + segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
        MulAddDst<float, float, false>(z0[off], t0[off], s[highBase + segOff], MASK_PLACEHOLDER, stepRepeats, segBinary);
    }
    VBarrier();
    SumSquares3(t0, x0, y0, z0, totalRepeats);
    WholeReduceMin<float>(
        results_.Get<float>(), t0, static_cast<int32_t>(m_),
        static_cast<int32_t>(count), 4, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
}

template <typename T, uint32_t MODE>
template <bool SplitHeavy>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::SegmentBlockGroup(
    uint32_t bStart, uint32_t numBlk, uint32_t count, uint32_t cPad)
{
    using namespace TreeLineCommon;
    if constexpr (SplitHeavy) {
        auto s = segmentSoa_.Get<float>();
        const uint32_t highBase = 3U * soaStride_ + 64U;
        const uint32_t muxBase = 6U * soaStride_ + 128U;
        auto bcast = pointSoa_.Get<float>();
        auto w = scratch_.Get<float>();
        constexpr int32_t dstStride = (MODE == 10U) ? 4 : 16;
        EvalSplitHeavySubAndDot(
            w, s[0U], s[soaStride_], s[2U * soaStride_],
            s[muxBase + soaStride_], s[muxBase + 2U * soaStride_], s[muxBase],
            bcast[192U], bcast[704U], bcast[1216U], bStart, numBlk, cPad);
        EvalSplitHeavyResidualAndReduce(
            w, reduceWork_.Get<float>(),
            s[highBase + soaStride_], s[highBase + 2U * soaStride_], s[highBase],
            bStart, numBlk, count, cPad, dstStride);
    } else {
        EvalCompactBlockGroup(bStart, numBlk, count, cPad);
    }
}

template <typename T, uint32_t MODE>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::ComputeSegments(
    LocalTensor<float> p, uint32_t pos, uint32_t count)
{
    using namespace TreeLineCommon;
    ResetMask();
    auto off32 = offsets_.Get<uint32_t>();
    auto bcast = pointSoa_.Get<float>();
    GatherPointAoS(bcast[0U], bcast[64U], bcast[128U], p, off32[tile_], count);
    VBarrier();
    const uint32_t cPad = (MODE == 10U) ? corePad_ :
        ((m_ <= 64U) ? count : ((end_ - begin_) > 64U ? ((count + 1U) & ~1U) : corePad_));
    BroadcastPointTriplet(
        bcast[192U], bcast[704U], bcast[1216U],
        bcast[0U], bcast[64U], bcast[128U], static_cast<uint8_t>((cPad + 7U) >> 3U));
    VBarrier();
    DispatchSegmentBlocks(count, cPad);
    VBarrier();
    ReduceAndStoreSegments(off32, pos, count);
}

template <typename T, uint32_t MODE>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::DispatchSegmentBlocks(
    uint32_t count, uint32_t cPad)
{
    using namespace TreeLineCommon;
    if constexpr (MODE == 10U) {
        SegmentBlockGroup<true>(0U, mp_ >> 6U, count, cPad);
    } else if (m_ <= 64U) {
        SegmentBlockGroup<false>(0U, 1U, count, count);
    } else if (cPad > 32U) {
        ComputeSegmentsBlocks1(count, cPad);
    } else {
        const uint32_t numBlocks = mp_ >> 6U;
        const uint32_t maxBlk = MinU(16U, SafeDivU(64U, cPad));
        const uint32_t stepBlk = (maxBlk > 0U) ? maxBlk : 1U;
        for (uint32_t b = 0U; b < numBlocks; b += stepBlk) {
            const uint32_t curBlk = MinU(stepBlk, numBlocks - b);
            SegmentBlockGroup<true>(b, curBlk, count, cPad);
        }
    }
}

template <typename T, uint32_t MODE>
__aicore__ inline void KernelTreeLineMinDistance<T, MODE>::ReduceAndStoreSegments(
    LocalTensor<uint32_t> off32, uint32_t pos, uint32_t count)
{
    using namespace TreeLineCommon;
    auto w = scratch_.Get<float>();
    auto d = distanceQueue_.AllocTensor<T>();
    auto idx = indexQueue_.AllocTensor<int32_t>();
    if (m_ <= 64U) {
        auto r = results_.Get<float>();
        Gather(w, r, off32, 0U, count);
        Gather(idx, r.ReinterpretCast<int32_t>(), off32, 4U, count);
        VBarrier();
        FinishSqrtAndCast<T>(d, w, w, count);
    } else {
        constexpr int32_t srcStride = (MODE == 10U) ? 1 : 4;
        DecodeHierarchicalMin<T, (MODE != 10U)>(
            d, idx, w, w[2112U], reduceWork_.Get<float>(),
            w[64U].ReinterpretCast<int32_t>(), w[128U].ReinterpretCast<int32_t>(),
            off32, mp_ >> 6U, count, srcStride);
    }
    StoreQueuedOutputs(distanceQueue_, indexQueue_, distanceGm_, indexGm_, d, idx, pos, count);
}

#endif  // TREE_LINE_GENERAL_COMPUTE_H
