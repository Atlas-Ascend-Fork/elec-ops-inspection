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

#ifndef TREE_LINE_PRUNE_VERIFY_H
#define TREE_LINE_PRUNE_VERIFY_H

#include "tree_line_prune.h"

template <typename T> __aicore__ inline void KernelTreeLinePrune<T>::StartPrimaryAndLookaheadGather(
    LocalTensor<uint8_t> surviveBits, bool primaryPreBroadcasted,
    bool hasLookahead, LocalTensor<T> nextRaw, uint32_t nextCount) {
    using namespace TreeLineCommon;
    auto w = scratch_.Get<float>();
    auto meta = metaBuf_.Get<float>();
    auto off12 = metaBuf_.Get<uint32_t>()[0U];
    auto rawHalf = w[18176U].ReinterpretCast<half>();
    auto codes = w[18432U];
    auto xyz = w[kRecordFloat];
    auto x = w[0U];
    auto y = w[1536U];
    auto z = w[3072U];
    LocalTensor<float> nextP;
    if (hasLookahead) {
        if constexpr (sizeof(T) == sizeof(float)) {
            nextP = nextRaw.template ReinterpretCast<float>();
        } else {
            nextP = pointCast_.Get<float>();
            Cast(nextP, nextRaw, RoundMode::CAST_NONE, 3U * nextCount);
        }
    }
    Cast(rawHalf, surviveBits, RoundMode::CAST_NONE, 512U);
    if (!primaryPreBroadcasted) {
        Brcb(x, meta[1152U], 8U, BrcbRepeatParams{2, 16});
        Brcb(y, meta[1216U], 8U, BrcbRepeatParams{2, 16});
        Brcb(z, meta[1280U], 8U, BrcbRepeatParams{2, 16});
    }
    VBarrier();
    Cast(codes, rawHalf, RoundMode::CAST_NONE, 512U);
    Compares(w[20736U].ReinterpretCast<uint8_t>(), rawHalf, zero16_, CMPMODE::GT, 512U);
    if (hasLookahead) {
        GatherPointAoS(xyz[0U], xyz[64U], xyz[128U], nextP, off12, nextCount);
    }
    if (!primaryPreBroadcasted) {
        ResetMask();
        const UnaryRepeatParams dupParams{2, 2, 16, 16};
        Adds<float, false>(x[8U], x, 0.0f, MASK_PLACEHOLDER, 8U, dupParams);
        Adds<float, false>(y[8U], y, 0.0f, MASK_PLACEHOLDER, 8U, dupParams);
        Adds<float, false>(z[8U], z, 0.0f, MASK_PLACEHOLDER, 8U, dupParams);
    }
    VBarrier();
}
template <typename T> __aicore__ inline void KernelTreeLinePrune<T>::ScaleAndFoldCodes(
    bool hasLookahead, uint32_t nextCount) {
    using namespace TreeLineCommon;
    auto w = scratch_.Get<float>();
    auto codes = w[18432U];
    auto fold = w[19008U];
    auto maxCode = w[19328U];
    auto xyz = w[kRecordFloat];
    auto p16 = w[kRecordFloat + 224U].ReinterpretCast<half>();
    const BinaryRepeatParams pairFold{1, 1, 1, 8, 16, 16};
    float factor = 256.0f;
    #pragma unroll
    for (uint32_t c = 1U; c < 8U; ++c) {
        Muls(codes[c * 64U], codes[c * 64U], factor, 64U);
        factor *= 256.0f;
    }
    if (hasLookahead) {
        Muls(w[6208U], xyz, coordScale_, 192U);
    }
    VBarrier();
    ResetMask();
    Max<float, false>(fold, codes, codes[64U], MASK_PLACEHOLDER, 4U, pairFold);
    if (hasLookahead) {
        Cast(p16, w[6208U], RoundMode::CAST_RINT, 192U);
    }
    VBarrier();
    ResetMask();
    Max<float, false>(fold, fold, fold[64U], MASK_PLACEHOLDER, 2U, pairFold);
    if (hasLookahead) {
        PrepareLookahead(p16, nextCount);
    }
    VBarrier();
    Max(maxCode, fold, fold[64U], 64U);
    VBarrier();
}
template <typename T> __aicore__ inline void KernelTreeLinePrune<T>::SplitPrimaryAndPrepare(
    LocalTensor<uint8_t> surviveBits, bool primaryPreBroadcasted,
    bool hasLookahead, LocalTensor<T> nextRaw, uint32_t nextCount) {
    using namespace TreeLineCommon;
    StartPrimaryAndLookaheadGather(
        surviveBits, primaryPreBroadcasted, hasLookahead, nextRaw, nextCount);
    ScaleAndFoldCodes(hasLookahead, nextCount);
    auto w = scratch_.Get<float>();
    auto codes = w[18432U];
    auto maxCode = w[19328U];
    auto exponent = w[19392U].ReinterpretCast<uint32_t>();
    auto power = w[19456U];
    auto clear = w[20160U];
    auto validBits = w[20800U].ReinterpretCast<uint8_t>();
    auto primaryBase = w[kDecodeFloat].ReinterpretCast<int32_t>();
    ShiftRight(exponent, maxCode.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(23), 64U);
    Compares(validBits, maxCode, 0.0f, CMPMODE::GT, 64U);
    VBarrier();
    Muls(power.ReinterpretCast<int32_t>(), exponent.ReinterpretCast<int32_t>(),
        static_cast<int32_t>(0x00800000), 64U);
    Adds(primaryBase, exponent.ReinterpretCast<int32_t>(), static_cast<int32_t>(-127), 64U);
    VBarrier();
    Maxs(primaryBase, primaryBase, static_cast<int32_t>(0), 64U);
    ResetMask();
    Adds<float, false>(clear, power, 0.0f, MASK_PLACEHOLDER, 8U, UnaryRepeatParams{1, 1, 8, 0});
    VBarrier();
    Muls(w[17024U].ReinterpretCast<int32_t>(), primaryBase, static_cast<int32_t>(64), 64U);
    Muls(primaryBase, primaryBase, static_cast<int32_t>(16), 64U);
    Compare(w[20864U].ReinterpretCast<uint8_t>(), codes, clear, CMPMODE::EQ, 512U);
    VBarrier();
}
template <typename T> __aicore__ inline uint32_t KernelTreeLinePrune<T>::AssignDupPassIndex(
    uint32_t point, ExtraDecodeState& st) {
    const uint64_t bit = 1ULL << point;
    if (!(st.seenDup0 & bit)) {
        st.seenDup0 |= bit;
        return 0U;
    }
    if (!(st.seenDup1 & bit)) {
        st.seenDup1 |= bit;
        return 1U;
    }
    if (!(st.seenDup2 & bit)) {
        st.seenDup2 |= bit;
        st.ptDupCnt[point] = 3U;
        return 2U;
    }
    return st.ptDupCnt[point]++;
}
template <typename T> __aicore__ inline void KernelTreeLinePrune<T>::RecordExtraSlot(
    uint32_t point, int32_t gBase, uint32_t extraIdx, ExtraDecodeState& st) {
    if (extraIdx >= kMaxExtra) {
        return;
    }
    auto w = scratch_.Get<float>();
    const uint32_t slot = 64U + extraIdx;
    w[kDecodeFloat].ReinterpretCast<int32_t>().SetValue(slot, gBase);
    w[17024U].ReinterpretCast<int32_t>().SetValue(extraIdx, gBase << 2);
    w[kDecodeFloat + 128U].ReinterpretCast<int32_t>().SetValue(extraIdx, static_cast<int32_t>(point << 2U));
    const uint32_t dp = AssignDupPassIndex(point, st);
    if (dp < kMaxDupSlots) {
        if (dp >= st.numDupPasses) {
            st.numDupPasses = dp + 1U;
        }
        w[kDecodeFloat + 192U].ReinterpretCast<int32_t>().SetValue(
            (dp << 6U) + point, static_cast<int32_t>(slot << 2U));
    } else {
        st.dupOverflow = true;
    }
}
template <typename T> __aicore__ inline void KernelTreeLinePrune<T>::DecodePointExtraMask(
    uint32_t point, uint32_t c, uint32_t gChunkBase,
    uint32_t byteMask, ExtraDecodeState& st) {
    auto groupBase = scratch_.Get<float>()[kDecodeFloat].ReinterpretCast<int32_t>();
    if ((static_cast<uint32_t>(groupBase.GetValue(point)) >> 7U) == c) {
        byteMask ^= 1U << (31U - static_cast<uint32_t>(__builtin_clz(byteMask)));
    }
    while (byteMask != 0U) {
        const uint32_t gSub = 31U - static_cast<uint32_t>(__builtin_clz(byteMask));
        byteMask ^= 1U << gSub;
        const int32_t gBase = static_cast<int32_t>((gChunkBase | gSub) << 4U);
        const uint32_t extraIdx = st.numExtra++;
        RecordExtraSlot(point, gBase, extraIdx, st);
    }
}
template <typename T> __aicore__ inline void KernelTreeLinePrune<T>::DecodeAllExtras(
    LocalTensor<uint8_t> surviveBits, ExtraDecodeState& st) {
    auto w = scratch_.Get<float>();
    auto onlyPrimaryU64 = w[20864U].ReinterpretCast<uint64_t>();
    auto nonZeroU64 = w[20736U].ReinterpretCast<uint64_t>();
    #pragma clang loop unroll(disable)
    for (uint32_t cRev = 8U; cRev != 0U; --cRev) {
        const uint32_t c = cRev - 1U;
        uint64_t extraPts = nonZeroU64.GetValue(c) & ~onlyPrimaryU64.GetValue(c);
        const uint32_t gChunkBase = c << 3U;
        const uint32_t byteBase = c << 6U;
        while (extraPts != 0ULL) {
            const uint32_t point = 63U - static_cast<uint32_t>(__builtin_clzll(extraPts));
            extraPts ^= (1ULL << point);
            const uint32_t byteMask = static_cast<uint32_t>(surviveBits.GetValue(byteBase + point));
            DecodePointExtraMask(point, c, gChunkBase, byteMask, st);
        }
    }
}
template <typename T> __aicore__ inline void KernelTreeLinePrune<T>::PrepareExtraPhaseVectors(
    uint32_t numExtra, uint32_t& groups, uint32_t& firstSlot, uint32_t& padLen) {
    using namespace TreeLineCommon;
    auto w = scratch_.Get<float>();
    auto meta = metaBuf_.Get<float>();
    auto metaInt = metaBuf_.Get<int32_t>();
    auto metaU32 = metaBuf_.Get<uint32_t>();
    auto blockOffsets = w[15616U].ReinterpretCast<int32_t>();
    auto blockBytes = w[17024U].ReinterpretCast<int32_t>();
    auto extraPtByte = w[kDecodeFloat + 128U].ReinterpretCast<uint32_t>();
    groups = numExtra;
    firstSlot = 64U;
    padLen = (numExtra + 7U) & ~7U;
    SyncPipe<HardEvent::S_V>(pipe_);
    auto packPx = w[17152U];
    auto packPy = w[17280U];
    auto packPz = w[17408U];
    Gather(blockOffsets, blockBytes, metaU32[448U], 0U, 2U * padLen);
    Gather(packPx, meta[1152U], extraPtByte, 0U, padLen);
    Gather(packPy, meta[1216U], extraPtByte, 0U, padLen);
    Gather(packPz, meta[1280U], extraPtByte, 0U, padLen);
    VBarrier();
    const uint8_t extraBrcb = static_cast<uint8_t>(padLen >> 3U);
    Add(blockOffsets, blockOffsets, metaInt[960U], 2U * padLen);
    Brcb(w[1024U], packPx, extraBrcb, BrcbRepeatParams{2, 16});
    Brcb(w[1536U + 1024U], packPy, extraBrcb, BrcbRepeatParams{2, 16});
    Brcb(w[3072U + 1024U], packPz, extraBrcb, BrcbRepeatParams{2, 16});
    VBarrier();
    ResetMask();
    const UnaryRepeatParams dupParams{2, 2, 16, 16};
    Adds<float, false>(w[1032U], w[1024U], 0.0f, MASK_PLACEHOLDER, extraBrcb, dupParams);
    Adds<float, false>(w[1536U + 1032U], w[1536U + 1024U], 0.0f, MASK_PLACEHOLDER, extraBrcb, dupParams);
    Adds<float, false>(w[3072U + 1032U], w[3072U + 1024U], 0.0f, MASK_PLACEHOLDER, extraBrcb, dupParams);
    VBarrier();
}
template <typename T> __aicore__ inline void KernelTreeLinePrune<T>::EvalExactPhaseGeometry(
    uint32_t firstSlot, uint32_t groups) {
    using namespace TreeLineCommon;
    auto w = scratch_.Get<float>();
    auto s32 = segmentSoa32_.Get<float>();
    const uint8_t rep = static_cast<uint8_t>((groups + 3U) >> 2U);
    const uint32_t mOff = firstSlot * kGroupLanes;
    auto phaseX = w[0U + mOff];
    auto phaseY = w[1536U + mOff];
    auto phaseZ = w[3072U + mOff];
    auto phaseUDot = w[4608U + mOff];
    auto phaseTempX = w[6208U + mOff];
    auto phaseTempY = w[7744U + mOff];
    auto phaseTempZ = w[9280U + mOff];
    auto phaseTX = w[10816U + mOff];
    auto phaseTY = w[12352U + mOff];
    auto phaseTZ = w[13888U + mOff];
    auto offsets = w[15616U].ReinterpretCast<uint32_t>();
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const UnaryRepeatParams unary{1, 1, 8, 8};
    const GatherRepeatParams gParams{1, 8};
    Gatherb<uint32_t>(phaseTempX.ReinterpretCast<uint32_t>(), s32[0U * kSoaStride32].ReinterpretCast<uint32_t>(), offsets, rep, gParams);
    Gatherb<uint32_t>(phaseTempY.ReinterpretCast<uint32_t>(), s32[1U * kSoaStride32].ReinterpretCast<uint32_t>(), offsets, rep, gParams);
    Gatherb<uint32_t>(phaseTempZ.ReinterpretCast<uint32_t>(), s32[2U * kSoaStride32].ReinterpretCast<uint32_t>(), offsets, rep, gParams);
    VBarrier();
    ResetMask();
    Sub<float, false>(phaseX, phaseTempX, phaseX, MASK_PLACEHOLDER, rep, binary);
    Sub<float, false>(phaseY, phaseTempY, phaseY, MASK_PLACEHOLDER, rep, binary);
    Sub<float, false>(phaseZ, phaseTempZ, phaseZ, MASK_PLACEHOLDER, rep, binary);
    Gatherb<uint32_t>(phaseTZ.ReinterpretCast<uint32_t>(), s32[7U * kSoaStride32 + 64U].ReinterpretCast<uint32_t>(), offsets, rep, gParams);
    Gatherb<uint32_t>(phaseTY.ReinterpretCast<uint32_t>(), s32[3U * kSoaStride32].ReinterpretCast<uint32_t>(), offsets, rep, gParams);
    Gatherb<uint32_t>(phaseTX.ReinterpretCast<uint32_t>(), s32[8U * kSoaStride32 + 64U].ReinterpretCast<uint32_t>(), offsets, rep, gParams);
    VBarrier();
    ResetMask();
    Mul<float, false>(phaseTZ, phaseZ, phaseTZ, MASK_PLACEHOLDER, rep, binary);
    Mul<float, false>(phaseUDot, phaseY, phaseTY, MASK_PLACEHOLDER, rep, binary);
    Mul<float, false>(phaseTX, phaseX, phaseTX, MASK_PLACEHOLDER, rep, binary);
    Gatherb<uint32_t>(phaseTempX.ReinterpretCast<uint32_t>(), s32[5U * kSoaStride32 + 64U].ReinterpretCast<uint32_t>(), offsets, rep, gParams);
    Gatherb<uint32_t>(phaseTempY.ReinterpretCast<uint32_t>(), s32[6U * kSoaStride32 + 64U].ReinterpretCast<uint32_t>(), offsets, rep, gParams);
    Gatherb<uint32_t>(phaseTempZ.ReinterpretCast<uint32_t>(), s32[4U * kSoaStride32 + 64U].ReinterpretCast<uint32_t>(), offsets, rep, gParams);
    VBarrier();
    FinishExactPhaseGeometry(phaseX, phaseY, phaseZ, phaseUDot,
        phaseTempX, phaseTempY, phaseTempZ, phaseTX, phaseTZ, rep, binary, unary);
}
template <typename T> __aicore__ inline void KernelTreeLinePrune<T>::FinishExactPhaseGeometry(
    LocalTensor<float> phaseX, LocalTensor<float> phaseY, LocalTensor<float> phaseZ,
    LocalTensor<float> phaseUDot, LocalTensor<float> phaseTempX,
    LocalTensor<float> phaseTempY, LocalTensor<float> phaseTempZ,
    LocalTensor<float> phaseTX, LocalTensor<float> phaseTZ,
    uint8_t rep, const BinaryRepeatParams& binary, const UnaryRepeatParams& unary) {
    using namespace TreeLineCommon;
    ResetMask();
    Add<float, false>(phaseUDot, phaseTZ, phaseUDot, MASK_PLACEHOLDER, rep, binary);
    VBarrier();
    Add<float, false>(phaseUDot, phaseTX, phaseUDot, MASK_PLACEHOLDER, rep, binary);
    VBarrier();
    Maxs<float, false>(phaseTZ, phaseUDot, 0.0f, MASK_PLACEHOLDER, rep, unary);
    VBarrier();
    Mins<float, false>(phaseUDot, phaseTZ, 1.0f, MASK_PLACEHOLDER, rep, unary);
    VBarrier();
    Mul<float, false>(phaseTempX, phaseUDot, phaseTempX, MASK_PLACEHOLDER, rep, binary);
    Mul<float, false>(phaseTempY, phaseUDot, phaseTempY, MASK_PLACEHOLDER, rep, binary);
    Mul<float, false>(phaseTempZ, phaseUDot, phaseTempZ, MASK_PLACEHOLDER, rep, binary);
    VBarrier();
    Add<float, false>(phaseX, phaseX, phaseTempX, MASK_PLACEHOLDER, rep, binary);
    Add<float, false>(phaseY, phaseY, phaseTempY, MASK_PLACEHOLDER, rep, binary);
    Add<float, false>(phaseZ, phaseZ, phaseTempZ, MASK_PLACEHOLDER, rep, binary);
    VBarrier();
    SumSquares3(phaseTZ, phaseX, phaseY, phaseZ, rep);
}
template <typename T> __aicore__ inline void KernelTreeLinePrune<T>::ReduceCandidatesAndPrefetchNext(
    uint32_t totalPairs, bool nextPrepared) {
    using namespace TreeLineCommon;
    auto w = scratch_.Get<float>();
    auto meta = metaBuf_.Get<float>();
    auto metaInt = metaBuf_.Get<int32_t>();
    auto pairOffsets = metaBuf_.Get<uint32_t>()[704U];
    auto pairs = w[17664U];
    WholeReduceMin<float>(
        pairs, w[13888U], static_cast<int32_t>(kGroupLanes),
        static_cast<int32_t>(totalPairs), 1, 1, 2, ReduceOrder::ORDER_VALUE_INDEX);
    VBarrier();
    Gather(meta[3648U], pairs, pairOffsets, 0U, totalPairs);
    Gather(metaInt[3776U], pairs.ReinterpretCast<int32_t>(), pairOffsets, 4U, totalPairs);
    if (nextPrepared) {
        auto sc16 = scratch_.Get<half>();
        auto s16 = segmentSoa16_.Get<half>();
        const BinaryRepeatParams bcastBinary{1, 1, 0, 8, 0, 1};
        ResetMask();
        Sub<half, false>(sc16[0U * kMatrixStride16 + 128U], s16[0U * kSoaStride16],
            meta[1408U].ReinterpretCast<half>(), MASK_PLACEHOLDER, 64U, bcastBinary);
        Sub<half, false>(sc16[1U * kMatrixStride16 + 128U], s16[7U * kSoaStride16 + 128U],
            meta[1920U].ReinterpretCast<half>(), MASK_PLACEHOLDER, 64U, bcastBinary);
        Sub<half, false>(sc16[2U * kMatrixStride16 + 128U], s16[1U * kSoaStride16],
            meta[2432U].ReinterpretCast<half>(), MASK_PLACEHOLDER, 64U, bcastBinary);
        Adds(metaInt[1152U], w.ReinterpretCast<int32_t>()[kRecordFloat], static_cast<int32_t>(0), 192U);
    }
    VBarrier();
    if (nextPrepared) {
        auto sc16 = scratch_.Get<half>();
        auto s16 = segmentSoa16_.Get<half>();
        const BinaryRepeatParams segBinary{1, 1, 1, 8, 8, 0};
        ResetMask();
        Mul<half, false>(sc16[4U * kMatrixStride16 + 128U], sc16[1U * kMatrixStride16 + 128U],
            s16[8U * kSoaStride16 + 128U], MASK_PLACEHOLDER, 64U, segBinary);
        Mul<half, false>(sc16[5U * kMatrixStride16 + 128U], sc16[0U * kMatrixStride16 + 128U],
            s16[2U * kSoaStride16], MASK_PLACEHOLDER, 64U, segBinary);
        Mul<half, false>(sc16[3U * kMatrixStride16 + 128U], sc16[2U * kMatrixStride16 + 128U],
            s16[3U * kSoaStride16], MASK_PLACEHOLDER, 64U, segBinary);
        nextPrefixPrepared_ = true;
    }
}
template <typename T> __aicore__ inline void KernelTreeLinePrune<T>::StepMergeDupPass(
    LocalTensor<T> d, LocalTensor<float> candidateD2, LocalTensor<float> candidateIdxF,
    LocalTensor<float> candD2, LocalTensor<float> candIdxF, LocalTensor<uint8_t> updateMask,
    LocalTensor<uint32_t> dupCandOffU32, uint32_t dp, uint32_t numDupPasses) {
    using namespace TreeLineCommon;
    Gather(candIdxF, candidateIdxF, dupCandOffU32[dp * 64U], 0U, 64U);
    Select(candidateD2, updateMask, candD2, candidateD2, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64U);
    VBarrier();
    Select(candidateIdxF, updateMask, candIdxF, candidateIdxF, SELMODE::VSEL_TENSOR_TENSOR_MODE, 64U);
    if (dp + 1U < numDupPasses) {
        Gather(candD2, candidateD2, dupCandOffU32[(dp + 1U) * 64U], 0U, 64U);
        VBarrier();
        Compare(updateMask, candD2, candidateD2, CMPMODE::LE, 64U);
        VBarrier();
        return;
    }
    if constexpr (sizeof(T) == sizeof(float)) {
        Sqrt(d.template ReinterpretCast<float>(), candidateD2, 64U);
    } else {
        Sqrt(candidateD2, candidateD2, 64U);
    }
    VBarrier();
}
template <typename T> __aicore__ inline void KernelTreeLinePrune<T>::MergeDuplicatePasses(
    LocalTensor<T> d, LocalTensor<int32_t> idx,
    uint32_t totalPairs, uint32_t numDupPasses) {
    using namespace TreeLineCommon;
    auto w = scratch_.Get<float>();
    auto meta = metaBuf_.Get<float>();
    auto metaInt = metaBuf_.Get<int32_t>();
    auto groupBase = w[kDecodeFloat].ReinterpretCast<int32_t>();
    auto dupCandOffU32 = w[kDecodeFloat + 192U].ReinterpretCast<uint32_t>();
    auto candidateD2 = meta[3648U];
    auto candidateIdx = metaInt[3776U];
    auto candidateIdxF = meta[3904U];
    auto candD2 = meta[128U];
    auto candIdxF = meta[256U];
    auto updateMask = meta[384U].ReinterpretCast<uint8_t>();
    Add(candidateIdx, candidateIdx, groupBase, totalPairs);
    Gather(candD2, candidateD2, dupCandOffU32[0U], 0U, 64U);
    VBarrier();
    Cast(candidateIdxF, candidateIdx, RoundMode::CAST_NONE, totalPairs);
    Compare(updateMask, candD2, candidateD2, CMPMODE::LE, 64U);
    VBarrier();
    for (uint32_t dp = 0U; dp < numDupPasses; ++dp) {
        StepMergeDupPass(d, candidateD2, candidateIdxF, candD2, candIdxF,
            updateMask, dupCandOffU32, dp, numDupPasses);
    }
    Cast(idx, candidateIdxF, RoundMode::CAST_RINT, 64U);
    if constexpr (sizeof(T) != sizeof(float)) {
        Cast(d, candidateD2, RoundMode::CAST_RINT, 64U);
    }
}
template <typename T> __aicore__ inline bool KernelTreeLinePrune<T>::RunExtraPhaseAndCheckFallback(
    uint32_t pos, uint32_t count, LocalTensor<uint8_t> surviveBits,
    event_t v2sEvent, ExtraDecodeState& st) {
    using namespace TreeLineCommon;
    auto w = scratch_.Get<float>();
    WaitFlag<HardEvent::V_S>(v2sEvent);
    const uint64_t seen = w[20800U].ReinterpretCast<uint64_t>().GetValue(0U);
    DecodeAllExtras(surviveBits, st);
    if (st.numExtra > kMaxExtra || st.dupOverflow || seen != ~0ULL) {
        SyncPipe<HardEvent::S_V>(pipe_);
        FallbackFullFP32(pos, count);
        return true;
    }
    if (st.numExtra != 0U) {
        uint32_t groups = 0U;
        uint32_t firstSlot = 0U;
        uint32_t padLen = 0U;
        PrepareExtraPhaseVectors(st.numExtra, groups, firstSlot, padLen);
        EvalExactPhaseGeometry(firstSlot, groups);
    }
    return false;
}
template <typename T> __aicore__ inline bool KernelTreeLinePrune<T>::VerifyTileVector(
    uint32_t pos, uint32_t count, LocalTensor<uint8_t> surviveBits,
    LocalTensor<float> tileMaxPairs, bool primaryPreBroadcasted,
    uint32_t nextCount, LocalTensor<T>& nextRaw) {
    using namespace TreeLineCommon;
    (void)tileMaxPairs;
    bool nextPrepared = false;
    nextPrefixPrepared_ = false;
    if (count != 64U || m_ != 1024U) {
        if (nextCount != 0U) {
            nextRaw = pointQueue_.DeQue<T>();
        }
        FallbackFullFP32(pos, count);
        return false;
    }
    if (nextCount != 0U) {
        nextRaw = pointQueue_.DeQue<T>();
        nextPrepared = (kLookaheadChunks != 0U && nextCount == 64U);
    }
    SplitPrimaryAndPrepare(surviveBits, primaryPreBroadcasted, nextPrepared, nextRaw, nextCount);
    const event_t v2sEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(v2sEvent);
    auto w = scratch_.Get<float>();
    auto blockOffsets = w[15616U].ReinterpretCast<int32_t>();
    Gather(blockOffsets, w[17024U].ReinterpretCast<int32_t>(), metaBuf_.Get<uint32_t>()[448U], 0U, 128U);
    VBarrier();
    Add(blockOffsets, blockOffsets, metaBuf_.Get<int32_t>()[960U], 128U);
    VBarrier();
    EvalExactPhaseGeometry(0U, 64U);
    ExtraDecodeState st{0ULL, 0ULL, 0ULL, {}, 0U, 0U, false};
    if (RunExtraPhaseAndCheckFallback(pos, count, surviveBits, v2sEvent, st)) {
        return nextPrepared;
    }
    const uint32_t totalPairs = 64U + st.numExtra;
    ReduceCandidatesAndPrefetchNext(totalPairs, nextPrepared);
    FinalizeVerifiedOutput(pos, count, totalPairs, st.numDupPasses);
    return nextPrepared;
}
template <typename T> __aicore__ inline void KernelTreeLinePrune<T>::FinalizeVerifiedOutput(
    uint32_t pos, uint32_t count, uint32_t totalPairs, uint32_t numDupPasses) {
    using namespace TreeLineCommon;
    auto d = distanceQueue_.AllocTensor<T>();
    auto idx = indexQueue_.AllocTensor<int32_t>();
    if (numDupPasses != 0U) {
        MergeDuplicatePasses(d, idx, totalPairs, numDupPasses);
    } else {
        auto w = scratch_.Get<float>();
        auto meta = metaBuf_.Get<float>();
        auto metaInt = metaBuf_.Get<int32_t>();
        Add(idx, metaInt[3776U], w[kDecodeFloat].ReinterpretCast<int32_t>(), 64U);
        FinishSqrtAndCast<T>(d, meta[3648U], meta[3648U], 64U);
    }
    StoreQueuedOutputs(distanceQueue_, indexQueue_, distanceGm_, indexGm_, d, idx, pos, count);
}
template <typename T>
using KernelTreeLinePruneLookahead = KernelTreeLinePrune<T>;
#endif  // TREE_LINE_PRUNE_VERIFY_H
