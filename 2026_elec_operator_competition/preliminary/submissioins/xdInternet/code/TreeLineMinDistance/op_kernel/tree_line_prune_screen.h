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

#ifndef TREE_LINE_PRUNE_SCREEN_H
#define TREE_LINE_PRUNE_SCREEN_H

#include "tree_line_prune.h"

template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::ScaleFp16SegmentsAndSelectEff()
{
    using namespace TreeLineCommon;
    auto w = scratch_.Get<float>();
    auto s32 = segmentSoa32_.Get<float>();
    auto regMask = w[11520U].ReinterpretCast<uint8_t>();
    if (m_ < mp128_) {
        Duplicate(w[0U], 96.0f, mp128_);
        Duplicate(w[1024U], 96.0f, mp128_);
        Duplicate(w[2048U], 96.0f, mp128_);
        Duplicate(w[3072U], 0.0f, mp128_);
        Duplicate(w[4096U], 0.0f, mp128_);
        Duplicate(w[5120U], 0.0f, mp128_);
        VBarrier();
    }
    Muls(w[0U], s32[0U * kSoaStride32], coordScale_, m_);
    Muls(w[1024U], s32[1U * kSoaStride32], coordScale_, m_);
    Muls(w[2048U], s32[2U * kSoaStride32], coordScale_, m_);
    Muls(w[3072U], s32[5U * kSoaStride32 + 64U], coordScale_, m_);
    Muls(w[4096U], s32[6U * kSoaStride32 + 64U], coordScale_, m_);
    Muls(w[5120U], s32[4U * kSoaStride32 + 64U], coordScale_, m_);
    VBarrier();
    const uint8_t rep128f = static_cast<uint8_t>(mp128_ >> 6U);
    ResetMask();
    SumSquares3(w[9344U], w[3072U], w[4096U], w[5120U], rep128f);
    Compares(regMask, w[9344U], 0.00006103515625f, CMPMODE::GT, mp128_);
    Duplicate(w[10432U], -1.0f, mp128_);
    VBarrier();
    Select(w[6208U], regMask, w[3072U], 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mp128_);
    Select(w[7232U], regMask, w[4096U], 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mp128_);
    Select(w[8256U], regMask, w[5120U], 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mp128_);
    Select(w[9344U], regMask, w[9344U], 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mp128_);
    VBarrier();
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::FinalizeAndCastFp16Segments()
{
    using namespace TreeLineCommon;
    auto w = scratch_.Get<float>();
    auto regMask = w[11520U].ReinterpretCast<uint8_t>();
    const uint8_t rep128f = static_cast<uint8_t>(mp128_ >> 6U);
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const UnaryRepeatParams unary{1, 1, 8, 8};
    Div(w[9344U], w[10432U], w[9344U], mp128_);
    ResetMask();
    Sub<float, false>(w[3072U], w[3072U], w[6208U], MASK_PLACEHOLDER, rep128f, binary);
    Sub<float, false>(w[4096U], w[4096U], w[7232U], MASK_PLACEHOLDER, rep128f, binary);
    Sub<float, false>(w[5120U], w[5120U], w[8256U], MASK_PLACEHOLDER, rep128f, binary);
    VBarrier();
    Axpy<float, float, false>(w[0U], w[3072U], 0.5f, MASK_PLACEHOLDER, rep128f, unary);
    Axpy<float, float, false>(w[1024U], w[4096U], 0.5f, MASK_PLACEHOLDER, rep128f, unary);
    Axpy<float, float, false>(w[2048U], w[5120U], 0.5f, MASK_PLACEHOLDER, rep128f, unary);
    Select(w[9344U], regMask, w[9344U], 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mp128_);
    VBarrier();
    ResetMask();
    Mul<float, false>(w[3072U], w[6208U], w[9344U], MASK_PLACEHOLDER, rep128f, binary);
    Mul<float, false>(w[4096U], w[7232U], w[9344U], MASK_PLACEHOLDER, rep128f, binary);
    Mul<float, false>(w[5120U], w[8256U], w[9344U], MASK_PLACEHOLDER, rep128f, binary);
    VBarrier();
    auto s16 = segmentSoa16_.Get<half>();
    Cast(s16[0U * kSoaStride16], w[0U], RoundMode::CAST_RINT, mp128_);
    Cast(s16[7U * kSoaStride16 + 128U], w[1024U], RoundMode::CAST_RINT, mp128_);
    Cast(s16[1U * kSoaStride16], w[2048U], RoundMode::CAST_RINT, mp128_);
    Cast(s16[2U * kSoaStride16], w[3072U], RoundMode::CAST_RINT, mp128_);
    Cast(s16[8U * kSoaStride16 + 128U], w[4096U], RoundMode::CAST_RINT, mp128_);
    Cast(s16[3U * kSoaStride16], w[5120U], RoundMode::CAST_RINT, mp128_);
    Cast(s16[4U * kSoaStride16], w[6208U], RoundMode::CAST_RINT, mp128_);
    Cast(s16[5U * kSoaStride16], w[7232U], RoundMode::CAST_RINT, mp128_);
    Cast(s16[6U * kSoaStride16], w[8256U], RoundMode::CAST_RINT, mp128_);
    VBarrier();
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::LoadSegments(
    uint32_t batch, uint32_t firstPos, uint32_t firstCount)
{
    using namespace TreeLineCommon;
    const uint32_t segFloats = 6U * m_;
    const uint32_t segPad64 = Align64(segFloats);
    CopyAndUnpackSegments32(batch, firstPos, firstCount, segFloats, segPad64);
    FoldMaxAbsAndLength32(segPad64);
    uint32_t redBlks = 0U;
    event_t maxVsEvent = static_cast<event_t>(0);
    ReduceMaxAbsAndComputeNegMu32(segPad64, redBlks, maxVsEvent);
    UpdateCoordScaleFromMax(redBlks, maxVsEvent);
    ScaleFp16SegmentsAndSelectEff();
    FinalizeAndCastFp16Segments();
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::StepFallbackBlock(
    LocalTensor<float> w, LocalTensor<float> s32, LocalTensor<float> pairs,
    uint32_t b, uint32_t count, uint8_t stepRep)
{
    using namespace TreeLineCommon;
    constexpr uint32_t stride32 = 4160U;
    const uint32_t segOff = b * 64U;
    EvalBlockStepProjectAndResidual(
        w[0U * stride32], w[1U * stride32], w[2U * stride32],
        w[4U * stride32], w[5U * stride32], w[4U * stride32], w[3U * stride32], w[5U * stride32],
        s32[5U * kSoaStride32 + 64U], s32[6U * kSoaStride32 + 64U], s32[4U * kSoaStride32 + 64U],
        s32[8U * kSoaStride32 + 64U], s32[3U * kSoaStride32], s32[7U * kSoaStride32 + 64U],
        segOff, stepRep);
    SumSquares3(w[3U * stride32], w[5U * stride32], w[1U * stride32], w[3U * stride32], stepRep);
    WholeReduceMin<float>(
        pairs[2U * b], w[3U * stride32], 64, static_cast<int32_t>(count),
        16, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
    ResetMask();
    if (b + 1U < numBlocks_) {
        const uint32_t nextSegOff = (b + 1U) * 64U;
        const BinaryRepeatParams bcastBinary{1, 1, 0, 8, 0, 1};
        Sub<float, false>(w[0U * stride32], s32[0U * kSoaStride32 + nextSegOff], w[28032U], MASK_PLACEHOLDER, stepRep, bcastBinary);
        Sub<float, false>(w[1U * stride32], s32[1U * kSoaStride32 + nextSegOff], w[28544U], MASK_PLACEHOLDER, stepRep, bcastBinary);
        Sub<float, false>(w[2U * stride32], s32[2U * kSoaStride32 + nextSegOff], w[29056U], MASK_PLACEHOLDER, stepRep, bcastBinary);
    }
    VBarrier();
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::FallbackFullFP32(uint32_t pos, uint32_t count)
{
    using namespace TreeLineCommon;
    auto meta = metaBuf_.Get<float>();
    auto off32 = metaBuf_.Get<uint32_t>()[64U];
    auto w = scratch_.Get<float>();
    auto pairs = meta[1344U];
    BroadcastPointTriplet(
        w[28032U], w[28544U], w[29056U],
        meta[1152U], meta[1216U], meta[1280U], static_cast<uint8_t>((count + 7U) >> 3U));
    VBarrier();
    auto s32 = segmentSoa32_.Get<float>();
    needRebroadcast_ = true;
    constexpr uint32_t stride32 = 4160U;
    const uint8_t stepRep = static_cast<uint8_t>(count);
    const BinaryRepeatParams bcastBinary{1, 1, 0, 8, 0, 1};
    ResetMask();
    Sub<float, false>(w[0U * stride32], s32[0U * kSoaStride32], w[28032U], MASK_PLACEHOLDER, stepRep, bcastBinary);
    Sub<float, false>(w[1U * stride32], s32[1U * kSoaStride32], w[28544U], MASK_PLACEHOLDER, stepRep, bcastBinary);
    Sub<float, false>(w[2U * stride32], s32[2U * kSoaStride32], w[29056U], MASK_PLACEHOLDER, stepRep, bcastBinary);
    VBarrier();
    for (uint32_t b = 0U; b < numBlocks_; ++b) {
        StepFallbackBlock(w, s32, pairs, b, count, stepRep);
    }
    auto d = distanceQueue_.AllocTensor<T>();
    auto idx = indexQueue_.AllocTensor<int32_t>();
    DecodeHierarchicalMin<T, true>(
        d, idx, meta[3392U], w[2048U], pairs,
        w[0U].ReinterpretCast<int32_t>(), w[64U].ReinterpretCast<int32_t>(),
        off32, numBlocks_, count, 4);
    Duplicate(w[kDecodeFloat], 0.0f, 192U);
    VBarrier();
    StoreQueuedOutputs(distanceQueue_, indexQueue_, distanceGm_, indexGm_, d, idx, pos, count);
}
template <typename T>
__aicore__ inline LocalTensor<half> KernelTreeLinePrune<T>::FineMin(uint32_t chunk, bool)
{
    return scratch_.Get<half>()[kFineBaseHalf + chunk * 512U];
}
template <typename T>
__aicore__ inline LocalTensor<half> KernelTreeLinePrune<T>::LookaheadFine(uint32_t chunk)
{
    return scratch_.Get<half>()[kFineBaseHalf + chunk * 512U];
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::BroadcastPoints16(
    LocalTensor<half> p16, uint32_t count)
{
    auto meta = metaBuf_.Get<float>();
    const uint8_t brcbRep = static_cast<uint8_t>((count + 7U) >> 3U);
    const BrcbRepeatParams params{1, 8};
    Brcb(meta[1408U].ReinterpretCast<half>(), p16[0U], brcbRep, params);
    Brcb(meta[1920U].ReinterpretCast<half>(), p16[64U], brcbRep, params);
    Brcb(meta[2432U].ReinterpretCast<half>(), p16[128U], brcbRep, params);
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::PreparePoints(
    LocalTensor<float> p, uint32_t count,
    LocalTensor<float> xyz, LocalTensor<float> range, LocalTensor<half> p16)
{
    using namespace TreeLineCommon;
    auto w = scratch_.Get<float>();
    auto off12 = metaBuf_.Get<uint32_t>()[0U];
    if (count < 64U) {
        Duplicate(xyz, 0.0f, 192U);
        VBarrier();
    }
    GatherPointAoS(xyz[0U], xyz[64U], xyz[128U], p, off12, count);
    VBarrier();
    Muls(w[6208U], xyz, coordScale_, 192U);
    VBarrier();
    (void)range;
    Cast(p16, w[6208U], RoundMode::CAST_RINT, 192U);
    VBarrier();
    BroadcastPoints16(p16, count);
    VBarrier();
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::RestoreLookaheadPoints(uint32_t count)
{
    using namespace TreeLineCommon;
    if (nextPrefixPrepared_) {
        return;
    }
    auto w = scratch_.Get<float>();
    auto wi = w.ReinterpretCast<int32_t>();
    auto mi = metaBuf_.Get<int32_t>();
    Adds(mi[1152U], wi[kRecordFloat], static_cast<int32_t>(0), 192U);
    if (needRebroadcast_) {
        BroadcastPoints16(w[kRecordFloat + 224U].ReinterpretCast<half>(), count);
        needRebroadcast_ = false;
        VBarrier();
    }
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::PrepareLookahead(
    LocalTensor<half> p16, uint32_t count)
{
    needRebroadcast_ = false;
    BroadcastPoints16(p16, count);
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::PrepareFirstChunkPrefix(
    uint32_t startChunk, uint8_t stepRep16)
{
    using namespace TreeLineCommon;
    auto meta = metaBuf_.Get<float>();
    auto runningMin16 = meta[2944U].ReinterpretCast<half>();
    auto s16 = segmentSoa16_.Get<half>();
    auto sc16 = scratch_.Get<half>();
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const BinaryRepeatParams segBinary{1, 1, 1, 8, 8, 0};
    const BinaryRepeatParams bcastBinary{1, 1, 0, 8, 0, 1};
    const uint32_t firstShift = ((startChunk & 1U) ^ 1U) << 7U;
    const uint32_t firstSeg = startChunk * 128U;
    ResetMask();
    if (!nextPrefixPrepared_) {
        Sub<half, false>(sc16[0U * kMatrixStride16 + firstShift], s16[0U * kSoaStride16 + firstSeg],
            meta[1408U].ReinterpretCast<half>(), MASK_PLACEHOLDER, stepRep16, bcastBinary);
        Sub<half, false>(sc16[1U * kMatrixStride16 + firstShift], s16[7U * kSoaStride16 + 128U + firstSeg],
            meta[1920U].ReinterpretCast<half>(), MASK_PLACEHOLDER, stepRep16, bcastBinary);
        Sub<half, false>(sc16[2U * kMatrixStride16 + firstShift], s16[1U * kSoaStride16 + firstSeg],
            meta[2432U].ReinterpretCast<half>(), MASK_PLACEHOLDER, stepRep16, bcastBinary);
        VBarrier();
        Mul<half, false>(sc16[4U * kMatrixStride16 + firstShift], sc16[1U * kMatrixStride16 + firstShift],
            s16[8U * kSoaStride16 + 128U + firstSeg], MASK_PLACEHOLDER, stepRep16, segBinary);
        Mul<half, false>(sc16[5U * kMatrixStride16 + firstShift], sc16[0U * kMatrixStride16 + firstShift],
            s16[2U * kSoaStride16 + firstSeg], MASK_PLACEHOLDER, stepRep16, segBinary);
        Mul<half, false>(sc16[3U * kMatrixStride16 + firstShift], sc16[2U * kMatrixStride16 + firstShift],
            s16[3U * kSoaStride16 + firstSeg], MASK_PLACEHOLDER, stepRep16, segBinary);
        if (startChunk == 2U) {
            Min<half, false>(runningMin16, FineMin(0U, false), FineMin(1U, false), MASK_PLACEHOLDER, 4, binary);
        }
        VBarrier();
    } else {
        nextPrefixPrepared_ = false;
        if (startChunk == 2U) {
            Min<half, false>(runningMin16, FineMin(0U, false), FineMin(1U, false), MASK_PLACEHOLDER, 4, binary);
            VBarrier();
        }
    }
    Add<half, false>(sc16[4U * kMatrixStride16 + firstShift],
        sc16[5U * kMatrixStride16 + firstShift], sc16[4U * kMatrixStride16 + firstShift],
        MASK_PLACEHOLDER, stepRep16, binary);
    VBarrier();
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::UpdateRunningMinBeforeChunk(
    uint32_t c, uint32_t startChunk, uint8_t stepRep16, bool prefetched)
{
    using namespace TreeLineCommon;
    auto runningMin16 = metaBuf_.Get<float>()[2944U].ReinterpretCast<half>();
    auto s16 = segmentSoa16_.Get<half>();
    auto sc16 = scratch_.Get<half>();
    const uint32_t segOff16 = c * 128U;
    const uint32_t cShift = ((c & 1U) ^ 1U) << 7U;
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const BinaryRepeatParams segBinary{1, 1, 1, 8, 8, 0};
    if (c > startChunk) {
        Mul<half, false>(sc16[3U * kMatrixStride16 + cShift], sc16[2U * kMatrixStride16 + cShift],
            s16[3U * kSoaStride16 + segOff16], MASK_PLACEHOLDER, stepRep16, segBinary);
        if (c == 2U) {
            Min<half, false>(runningMin16, FineMin(0U, prefetched), FineMin(1U, prefetched), MASK_PLACEHOLDER, 4, binary);
        } else if (c > 2U) {
            Min<half, false>(runningMin16, runningMin16, FineMin(c - 1U, prefetched), MASK_PLACEHOLDER, 4, binary);
        }
        VBarrier();
    } else if (c > 2U) {
        Min<half, false>(runningMin16, runningMin16, FineMin(c - 1U, prefetched), MASK_PLACEHOLDER, 4, binary);
        VBarrier();
    }
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::EvalChunkBody16(
    uint32_t c, uint32_t stopChunk, uint32_t count,
    uint8_t stepRep16, bool prefetched, bool lookaheadOutput)
{
    using namespace TreeLineCommon;
    auto meta = metaBuf_.Get<float>();
    auto s16 = segmentSoa16_.Get<half>();
    auto sc16 = scratch_.Get<half>();
    const uint32_t segOff16 = c * 128U;
    const uint32_t cShift = ((c & 1U) ^ 1U) << 7U;
    auto x0_16 = sc16[0U * kMatrixStride16 + cShift];
    auto y0_16 = sc16[1U * kMatrixStride16 + cShift];
    auto z0_16 = sc16[2U * kMatrixStride16 + cShift];
    auto t0_16 = sc16[3U * kMatrixStride16 + cShift];
    auto u0_16 = sc16[4U * kMatrixStride16 + cShift];
    auto u1_16 = sc16[5U * kMatrixStride16 + cShift];
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const BinaryRepeatParams segBinary{1, 1, 1, 8, 8, 0};
    const UnaryRepeatParams unary{1, 1, 8, 8};
    Add<half, false>(u0_16, t0_16, u0_16, MASK_PLACEHOLDER, stepRep16, binary);
    VBarrier();
    Maxs<half, false>(t0_16, u0_16, zero16_, MASK_PLACEHOLDER, stepRep16, unary);
    VBarrier();
    Mins<half, false>(u0_16, t0_16, one16_, MASK_PLACEHOLDER, stepRep16, unary);
    VBarrier();
    Mul<half, false>(u1_16, u0_16, s16[4U * kSoaStride16 + segOff16], MASK_PLACEHOLDER, stepRep16, segBinary);
    Mul<half, false>(t0_16, u0_16, s16[6U * kSoaStride16 + segOff16], MASK_PLACEHOLDER, stepRep16, segBinary);
    Mul<half, false>(u0_16, u0_16, s16[5U * kSoaStride16 + segOff16], MASK_PLACEHOLDER, stepRep16, segBinary);
    VBarrier();
    Add<half, false>(x0_16, x0_16, u1_16, MASK_PLACEHOLDER, stepRep16, binary);
    Add<half, false>(u0_16, y0_16, u0_16, MASK_PLACEHOLDER, stepRep16, binary);
    Add<half, false>(z0_16, z0_16, t0_16, MASK_PLACEHOLDER, stepRep16, binary);
    VBarrier();
    FinishChunkBody16(c, stopChunk, count, stepRep16, prefetched, lookaheadOutput,
        x0_16, u0_16, z0_16, t0_16, meta, s16, sc16);
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::FinishChunkBody16(
    uint32_t c, uint32_t stopChunk, uint32_t count,
    uint8_t stepRep16, bool prefetched, bool lookaheadOutput,
    LocalTensor<half> x0_16, LocalTensor<half> u0_16,
    LocalTensor<half> z0_16, LocalTensor<half> t0_16,
    LocalTensor<float> meta, LocalTensor<half> s16, LocalTensor<half> sc16)
{
    using namespace TreeLineCommon;
    const uint32_t nextOff16 = (c + 1U) * 128U;
    const uint32_t nextShift = (c & 1U) << 7U;
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const BinaryRepeatParams bcastBinary{1, 1, 0, 8, 0, 1};
    Mul<half, false>(t0_16, x0_16, x0_16, MASK_PLACEHOLDER, stepRep16, binary);
    if (c + 1U < stopChunk) {
        Sub<half, false>(sc16[1U * kMatrixStride16 + nextShift], s16[7U * kSoaStride16 + 128U + nextOff16],
            meta[1920U].ReinterpretCast<half>(), MASK_PLACEHOLDER, stepRep16, bcastBinary);
    }
    VBarrier();
    MulAddDst<half, half, false>(t0_16, u0_16, u0_16, MASK_PLACEHOLDER, stepRep16, binary);
    if (c + 1U < stopChunk) {
        Sub<half, false>(sc16[0U * kMatrixStride16 + nextShift], s16[0U * kSoaStride16 + nextOff16],
            meta[1408U].ReinterpretCast<half>(), MASK_PLACEHOLDER, stepRep16, bcastBinary);
    }
    VBarrier();
    ReduceChunkAndPrefetchNext16(c, stopChunk, count, stepRep16, prefetched,
        lookaheadOutput, nextOff16, nextShift, z0_16, t0_16, meta, s16, sc16);
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::ReduceChunkAndPrefetchNext16(
    uint32_t c, uint32_t stopChunk, uint32_t count,
    uint8_t stepRep16, bool prefetched, bool lookaheadOutput,
    uint32_t nextOff16, uint32_t nextShift,
    LocalTensor<half> z0_16, LocalTensor<half> t0_16,
    LocalTensor<float> meta, LocalTensor<half> s16, LocalTensor<half> sc16)
{
    using namespace TreeLineCommon;
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const BinaryRepeatParams segBinary{1, 1, 1, 8, 8, 0};
    const BinaryRepeatParams bcastBinary{1, 1, 0, 8, 0, 1};
    auto fineMin16 = lookaheadOutput ? LookaheadFine(c) : FineMin(c, prefetched);
    MulAddDst<half, half, false>(t0_16, z0_16, z0_16, MASK_PLACEHOLDER, stepRep16, binary);
    if (c + 1U < stopChunk) {
        Mul<half, false>(sc16[4U * kMatrixStride16 + nextShift], sc16[1U * kMatrixStride16 + nextShift],
            s16[8U * kSoaStride16 + 128U + nextOff16], MASK_PLACEHOLDER, stepRep16, segBinary);
        Mul<half, false>(sc16[5U * kMatrixStride16 + nextShift], sc16[0U * kMatrixStride16 + nextShift],
            s16[2U * kSoaStride16 + nextOff16], MASK_PLACEHOLDER, stepRep16, segBinary);
    }
    if (count < 64U) {
        Duplicate(fineMin16, one16_, 512U);
    }
    VBarrier();
    BlockReduceMin<half>(fineMin16, t0_16, static_cast<int32_t>(count), 128, 1, 1, 8);
    ResetMask();
    if (c + 1U < stopChunk) {
        Sub<half, false>(sc16[2U * kMatrixStride16 + nextShift], s16[1U * kSoaStride16 + nextOff16],
            meta[2432U].ReinterpretCast<half>(), MASK_PLACEHOLDER, stepRep16, bcastBinary);
        Add<half, false>(sc16[4U * kMatrixStride16 + nextShift],
            sc16[5U * kMatrixStride16 + nextShift], sc16[4U * kMatrixStride16 + nextShift],
            MASK_PLACEHOLDER, stepRep16, binary);
    }
    VBarrier();
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::ScreenChunks(
    uint32_t startChunk, uint32_t stopChunk, uint32_t count,
    bool prefetched, bool lookaheadOutput)
{
    const uint8_t stepRep16 = static_cast<uint8_t>(count);
    PrepareFirstChunkPrefix(startChunk, stepRep16);
    for (uint32_t c = startChunk; c < stopChunk; ++c) {
        UpdateRunningMinBeforeChunk(c, startChunk, stepRep16, prefetched);
        EvalChunkBody16(c, stopChunk, count, stepRep16, prefetched, lookaheadOutput);
    }
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::BuildSurvivalMask()
{
    using namespace TreeLineCommon;
    auto meta = metaBuf_.Get<float>();
    auto metaInt = metaBuf_.Get<int32_t>();
    auto w = scratch_.Get<float>();
    auto runningMin16 = meta[2944U].ReinterpretCast<half>();
    auto minAcross32 = meta[3712U];
    auto thresh32 = w[6848U];
    auto surviveBits = w[kMaskFloat].ReinterpretCast<uint8_t>();
    auto thresh64x16 = w[7232U].ReinterpretCast<half>();
    auto pointGroupMin32 = w[4672U];
    Cast(pointGroupMin32, runningMin16, RoundMode::CAST_NONE, 512U);
    Duplicate(thresh32, 0.0028f, 64U);
    if (numChunks128_ < 8U) {
        Duplicate(w[kMaskFloat], 0.0f, 128U);
    }
    ResetMask();
    Adds<int32_t, false>(w[kDecodeFloat + 192U].ReinterpretCast<int32_t>(), metaInt[640U],
        static_cast<int32_t>(0), MASK_PLACEHOLDER, 4U, UnaryRepeatParams{1, 1, 8, 0});
    VBarrier();
    BlockReduceMin<float>(minAcross32, pointGroupMin32, 8, 64, 1, 1, 8);
    ResetMask();
    VBarrier();
    ResetMask();
    Axpy<float, float, false>(thresh32, minAcross32, 1.024f, MASK_PLACEHOLDER, 1, UnaryRepeatParams{1, 1, 8, 8});
    VBarrier();
    Brcb(w[4608U], thresh32, 8U, BrcbRepeatParams{1, 8});
    VBarrier();
    Cast(w[5120U].ReinterpretCast<half>(), w[4608U], RoundMode::CAST_RINT, 512U);
    VBarrier();
    ResetMask();
    const uint8_t nChunks = static_cast<uint8_t>(numChunks128_);
    const UnaryRepeatParams copyParams{1, 1, 32, 0};
    Adds<float, false>(w[7232U + 0U],   w[5120U + 0U],   0.0f, MASK_PLACEHOLDER, nChunks, copyParams);
    Adds<float, false>(w[7232U + 64U],  w[5120U + 64U],  0.0f, MASK_PLACEHOLDER, nChunks, copyParams);
    Adds<float, false>(w[7232U + 128U], w[5120U + 128U], 0.0f, MASK_PLACEHOLDER, nChunks, copyParams);
    Adds<float, false>(w[7232U + 192U], w[5120U + 192U], 0.0f, MASK_PLACEHOLDER, nChunks, copyParams);
    VBarrier();
    Compare(surviveBits, scratch_.Get<half>()[kFineBaseHalf], thresh64x16, CMPMODE::LE, numChunks128_ * 512U);
    VBarrier();
}
template <typename T>
__aicore__ inline void KernelTreeLinePrune<T>::ComputeTileScreened(
    LocalTensor<float> p, uint32_t pos, uint32_t count, bool prefetched)
{
    using namespace TreeLineCommon;
    (void)pos;
    auto meta = metaBuf_.Get<float>();
    auto w = scratch_.Get<float>();
    auto runningMin16 = meta[2944U].ReinterpretCast<half>();
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    if (prefetched) {
        RestoreLookaheadPoints(count);
    } else {
        PreparePoints(p, count, meta[1152U], meta[1344U], w[384U].ReinterpretCast<half>());
    }
    ScreenChunks(0U, numChunks128_, count, false, false);
    if (numChunks128_ == 2U) {
        Min<half, false>(runningMin16, FineMin(0U, prefetched), FineMin(1U, prefetched), MASK_PLACEHOLDER, 4, binary);
        VBarrier();
    } else if (numChunks128_ > 2U) {
        Min<half, false>(runningMin16, runningMin16, FineMin(numChunks128_ - 1U, false), MASK_PLACEHOLDER, 4, binary);
        VBarrier();
    }
    BuildSurvivalMask();
}
#endif  // TREE_LINE_PRUNE_SCREEN_H
