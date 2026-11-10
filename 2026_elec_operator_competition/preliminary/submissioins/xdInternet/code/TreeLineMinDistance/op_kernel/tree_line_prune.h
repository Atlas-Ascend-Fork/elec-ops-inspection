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

#ifndef TREE_LINE_PRUNE_H
#define TREE_LINE_PRUNE_H

#include "tree_line_common.h"

template <typename T>
class KernelTreeLinePrune {
public:
    __aicore__ inline void Init(
        GM_ADDR points, GM_ADDR segments, GM_ADDR distances, GM_ADDR indices,
        const TreeLineMinDistanceTilingData& tiling, TPipe* pipe)
    {
        using namespace TreeLineCommon;
        pipe_ = pipe;
        tile_ = 64U;
        coordScale_ = 1.0f;
        n_ = tiling.points;
        m_ = tiling.segments;
        mp64_ = Align64(m_);
        numBlocks_ = mp64_ >> 6U;
        mp128_ = Align128(m_);
        numChunks128_ = mp128_ >> 7U;
        ComputeCoreRange(tiling.batch, n_, tiling.blockDim, GetBlockIdx(), begin_, end_);
        if (begin_ >= end_) {
            return;
        }
        BindGlobalBuffersAndMask(
            pointsGm_, segmentsGm_, distanceGm_, indexGm_,
            points, segments, distances, indices, tiling.batch * n_, tiling.batch * m_);
        InitPruneBuffers();
        InitMetaTables();
    }

    __aicore__ inline void Process()
    {
        using namespace TreeLineCommon;
        uint32_t curPos = begin_;
        while (curPos < end_) {
            const uint32_t curBatch = SafeDivU(curPos, n_);
            const uint32_t curBatchEnd = MinU(end_, (curBatch + 1U) * n_);
            uint32_t curCount = MinU(tile_, curBatchEnd - curPos);
            LoadSegments(curBatch, curPos, curCount);
            ProcessPruneBatch(curPos, curBatchEnd, curCount);
        }
    }

private:
    static constexpr uint32_t kLookaheadChunks = 1U;
    static constexpr uint32_t kSoaStride32 = 1024U;
    static constexpr uint32_t kSoaStride16 = 1024U;
    static constexpr float kFloatMax = 3.402823466e+38F;
    static constexpr uint32_t kMatrixStride16 = 8320U;
    static constexpr uint32_t kFineBaseHalf = 49920U;
    static constexpr uint32_t kMaskFloat = 27008U;
    static constexpr uint32_t kRecordFloat = 27136U;
    static constexpr uint32_t kDecodeFloat = 27968U;
    static constexpr uint32_t kGroupLanes = 16U;
    static constexpr uint32_t kMaxExtra = 24U;
    static constexpr uint32_t kMaxDupSlots = 4U;

    union BitsF32 {
        float f;
        uint32_t u;
    };

    struct ExtraDecodeState {
        uint64_t seenDup0;
        uint64_t seenDup1;
        uint64_t seenDup2;
        uint8_t ptDupCnt[64];
        uint32_t numExtra;
        uint32_t numDupPasses;
        bool dupOverflow;
    };

    __aicore__ inline bool Finite(float x)
    {
        return x <= kFloatMax && x >= -kFloatMax;
    }

    __aicore__ inline void InitPruneBuffers()
    {
        pipe_->InitBuffer(scratch_, 29568U * sizeof(float));
        pipe_->InitBuffer(segmentSoa32_, 9280U * sizeof(float));
        pipe_->InitBuffer(segmentSoa16_, 9344U * sizeof(half));
        pipe_->InitBuffer(metaBuf_, 4032U * sizeof(float));
        pipe_->InitBuffer(pointQueue_, 2, 3U * 64U * sizeof(T));
        pipe_->InitBuffer(distanceQueue_, 1, 64U * sizeof(T));
        pipe_->InitBuffer(indexQueue_, 1, 64U * sizeof(int32_t));
        if constexpr (sizeof(T) != sizeof(float)) {
            pipe_->InitBuffer(pointCast_, 3U * 64U * sizeof(float));
        }
    }

    __aicore__ inline void ProcessPruneBatch(
        uint32_t& pos, uint32_t batchEnd, uint32_t& count)
    {
        using namespace TreeLineCommon;
        auto raw = pointQueue_.DeQue<T>();
        bool prefetched = false;
        while (pos < batchEnd) {
            const uint32_t nextPos = pos + count;
            const uint32_t nextCnt = MinU(tile_, batchEnd - nextPos);
            if (nextCnt != 0U) {
                CopyPointTileToQueue(pointQueue_, pointsGm_, nextPos, nextCnt);
            }
            LocalTensor<float> p = PrepareTilePoints32(raw, count, prefetched);
            ComputeTileScreened(p, pos, count, prefetched);
            LocalTensor<T> nextRaw;
            auto w = scratch_.Get<float>();
            auto meta = metaBuf_.Get<float>();
            const bool nextPrefetched = VerifyTileVector(
                pos, count, w[kMaskFloat].ReinterpretCast<uint8_t>(),
                meta[1344U], false, nextCnt, nextRaw);
            pointQueue_.FreeTensor(raw);
            if (nextCnt != 0U) {
                raw = nextRaw;
            }
            prefetched = nextPrefetched;
            pos = nextPos;
            count = nextCnt;
        }
    }

    __aicore__ inline LocalTensor<float> PrepareTilePoints32(
        LocalTensor<T> raw, uint32_t count, bool prefetched)
    {
        using namespace TreeLineCommon;
        if constexpr (sizeof(T) == sizeof(float)) {
            (void)count;
            (void)prefetched;
            return raw.template ReinterpretCast<float>();
        } else {
            LocalTensor<float> p = pointCast_.Get<float>();
            if (!prefetched) {
                Cast(p, raw, RoundMode::CAST_NONE, 3U * count);
                VBarrier();
            }
            return p;
        }
    }

    __aicore__ inline bool TryComputeBatchRange(
        uint32_t b, uint32_t n, uint32_t blocks, uint32_t core,
        uint32_t& outBegin, uint32_t& outEnd)
    {
        using namespace TreeLineCommon;
        if (b <= 1U || blocks < b || SafeModU(blocks, b) != 0U) {
            return false;
        }
        const uint32_t cpb = SafeDivU(blocks, b);
        if (cpb == 0U || n < cpb * 64U) {
            return false;
        }
        const uint32_t myBatch = SafeDivU(core, cpb);
        const uint32_t subCore = SafeModU(core, cpb);
        const uint32_t chunks = (n + 63U) >> 6U;
        const uint32_t subBegin = MinU(n, SafeDivU(chunks * subCore, cpb) << 6U);
        const uint32_t subEnd = (subCore + 1U == cpb)
            ? n
            : MinU(n, SafeDivU(chunks * (subCore + 1U), cpb) << 6U);
        outBegin = myBatch * n + subBegin;
        outEnd = myBatch * n + subEnd;
        return true;
    }

    __aicore__ inline void ComputeCoreRange(
        uint32_t b, uint32_t n, uint32_t blocks, uint32_t core,
        uint32_t& outBegin, uint32_t& outEnd)
    {
        using namespace TreeLineCommon;
        const uint32_t total = b * n;
        if (blocks == 0U || core >= blocks) {
            outBegin = total;
            outEnd = total;
            return;
        }
        if (TryComputeBatchRange(b, n, blocks, core, outBegin, outEnd)) {
            return;
        }
        outBegin = SafeDivU(total * core, blocks);
        outEnd = SafeDivU(total * (core + 1U), blocks);
    }

    __aicore__ inline void InitMetaTables()
    {
        using namespace TreeLineCommon;
        auto metaInt = metaBuf_.Get<int32_t>();
        auto off12Int = metaInt[0U];
        auto off32Int = metaInt[64U];
        auto basePtByte = metaInt[640U];
        auto pairOffsets = metaInt[704U];
        auto laneInt16 = metaInt[896U];
        union { uint16_t u; half h; } z16{0x0000U}, o16{0x3C00U};
        zero16_ = z16.h;
        one16_ = o16.h;
        needRebroadcast_ = false;
        nextPrefixPrepared_ = false;
        Duplicate(scratch_.Get<float>()[kDecodeFloat], 0.0f, 192U);
        CreateVecIndex(off12Int, static_cast<int32_t>(0), 64U);
        CreateVecIndex(pairOffsets, static_cast<int32_t>(0), 128U);
        VBarrier();
        Muls(basePtByte, off12Int, static_cast<int32_t>(4), 64U);
        Muls(pairOffsets, pairOffsets, static_cast<int32_t>(8), 128U);
        Muls(laneInt16, off12Int, static_cast<int32_t>(4), 16U);
        Muls(off32Int, off12Int, static_cast<int32_t>(32), 64U);
        Muls(off12Int, off12Int, static_cast<int32_t>(12), 64U);
        VBarrier();
        auto blockDupBytes = metaInt[448U];
        auto blockLaneBytes = metaInt[960U];
        #pragma unroll
        for (uint32_t i = 0U; i < 8U; ++i) {
            blockDupBytes.SetValue(i, static_cast<int32_t>((i >> 1U) << 2U));
            blockLaneBytes.SetValue(i, static_cast<int32_t>((i & 1U) << 5U));
        }
        SyncPipe<HardEvent::S_V>(pipe_);
        #pragma clang loop unroll(disable)
        for (uint32_t filled = 8U; filled < 192U; filled <<= 1U) {
            const uint32_t take = MinU(filled, 192U - filled);
            Adds(blockDupBytes[filled], blockDupBytes, static_cast<int32_t>(filled << 1U), take);
            Adds(blockLaneBytes[filled], blockLaneBytes, static_cast<int32_t>(0), take);
            VBarrier();
        }
    }

    __aicore__ inline void CopyAndUnpackSegments32(
        uint32_t batch, uint32_t firstPos, uint32_t firstCount,
        uint32_t segFloats, uint32_t segPad64)
    {
        using namespace TreeLineCommon;
        auto w = scratch_.Get<float>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(6U * m_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        if constexpr (sizeof(T) == sizeof(float)) {
            DataCopyPad(w[10240U].template ReinterpretCast<T>(), segmentsGm_[batch * m_ * 6U], cp, pad);
        } else {
            DataCopyPad(w[13312U].template ReinterpretCast<T>(), segmentsGm_[batch * m_ * 6U], cp, pad);
        }
        auto s32 = segmentSoa32_.Get<float>();
        auto segOffInt = w[9216U].ReinterpretCast<int32_t>();
        CreateVecIndex(segOffInt, static_cast<int32_t>(0), mp64_);
        if (segFloats < segPad64) {
            Duplicate(w[0U], 0.0f, segPad64);
        }
        if (m_ < mp64_) {
            Duplicate(s32[0U * kSoaStride32], 1.0e18f, mp64_);
            Duplicate(s32[1U * kSoaStride32], 1.0e18f, mp64_);
            Duplicate(s32[2U * kSoaStride32], 1.0e18f, mp64_);
            Duplicate(s32[5U * kSoaStride32 + 64U], 1.0e18f, mp64_);
            Duplicate(s32[6U * kSoaStride32 + 64U], 1.0e18f, mp64_);
            Duplicate(s32[4U * kSoaStride32 + 64U], 1.0e18f, mp64_);
        }
        VBarrier();
        Muls(segOffInt, segOffInt, static_cast<int32_t>(24), mp64_);
        SyncPipe<HardEvent::MTE2_V>(pipe_);
        CopyPointTileToQueue(pointQueue_, pointsGm_, firstPos, firstCount);
        auto aos = w[10240U];
        if constexpr (sizeof(T) != sizeof(float)) {
            Cast(aos, w[13312U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, segFloats);
        }
        VBarrier();
        Abs(w[0U], aos, segFloats);
        GatherSegmentAoS(
            s32[0U * kSoaStride32], s32[1U * kSoaStride32], s32[2U * kSoaStride32],
            s32[5U * kSoaStride32 + 64U], s32[6U * kSoaStride32 + 64U], s32[4U * kSoaStride32 + 64U],
            aos, segOffInt.ReinterpretCast<uint32_t>(), m_);
        VBarrier();
    }

    __aicore__ inline void FoldMaxAbsAndLength32(uint32_t segPad64)
    {
        using namespace TreeLineCommon;
        auto w = scratch_.Get<float>();
        auto s32 = segmentSoa32_.Get<float>();
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        const BinaryRepeatParams pairFold{1, 1, 1, 8, 16, 16};
        auto tmpAbs = w[0U];
        auto invLen32 = w[6400U];
        const uint8_t rep64 = static_cast<uint8_t>(mp64_ >> 6U);
        ResetMask();
        Sub<float, false>(s32[5U * kSoaStride32 + 64U], s32[5U * kSoaStride32 + 64U], s32[0U * kSoaStride32], MASK_PLACEHOLDER, rep64, binary);
        Sub<float, false>(s32[6U * kSoaStride32 + 64U], s32[6U * kSoaStride32 + 64U], s32[1U * kSoaStride32], MASK_PLACEHOLDER, rep64, binary);
        Sub<float, false>(s32[4U * kSoaStride32 + 64U], s32[4U * kSoaStride32 + 64U], s32[2U * kSoaStride32], MASK_PLACEHOLDER, rep64, binary);
        if (segPad64 >= 6144U) {
            Max<float, false>(tmpAbs[0U], tmpAbs[0U], tmpAbs[64U], MASK_PLACEHOLDER, 48, pairFold);
        }
        VBarrier();
        Mul<float, false>(invLen32, s32[5U * kSoaStride32 + 64U], s32[5U * kSoaStride32 + 64U], MASK_PLACEHOLDER, rep64, binary);
        Mul<float, false>(w[4288U], s32[6U * kSoaStride32 + 64U], s32[6U * kSoaStride32 + 64U], MASK_PLACEHOLDER, rep64, binary);
        Mul<float, false>(w[5312U], s32[4U * kSoaStride32 + 64U], s32[4U * kSoaStride32 + 64U], MASK_PLACEHOLDER, rep64, binary);
        if (segPad64 >= 6144U) {
            Max<float, false>(tmpAbs[0U], tmpAbs[0U], tmpAbs[64U], MASK_PLACEHOLDER, 24, pairFold);
        }
        VBarrier();
        Add<float, false>(invLen32, invLen32, w[4288U], MASK_PLACEHOLDER, rep64, binary);
        if (segPad64 >= 6144U) {
            Max<float, false>(tmpAbs[0U], tmpAbs[0U], tmpAbs[64U], MASK_PLACEHOLDER, 12, pairFold);
        }
        VBarrier();
        Add<float, false>(invLen32, invLen32, w[5312U], MASK_PLACEHOLDER, rep64, binary);
        if (segPad64 >= 6144U) {
            Max<float, false>(tmpAbs[0U], tmpAbs[0U], tmpAbs[64U], MASK_PLACEHOLDER, 6, pairFold);
        }
        VBarrier();
    }

    __aicore__ inline void ReduceMaxAbsAndComputeNegMu32(
        uint32_t segPad64, uint32_t& redBlks, event_t& maxVsEvent)
    {
        using namespace TreeLineCommon;
        auto w = scratch_.Get<float>();
        auto s32 = segmentSoa32_.Get<float>();
        auto invLen32 = w[6400U];
        const uint8_t rep64 = static_cast<uint8_t>(mp64_ >> 6U);
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        const UnaryRepeatParams unary{1, 1, 8, 8};
        redBlks = (segPad64 >= 6144U) ? 6U : (segPad64 >> 6U);
        WholeReduceMax<float>(
            w[16512U], w[0U], 64, static_cast<int32_t>(redBlks),
            1, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
        ResetMask();
        maxVsEvent = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(maxVsEvent);
        Maxs<float, false>(invLen32, invLen32, 1.0e-30f, MASK_PLACEHOLDER, rep64, unary);
        Duplicate(w[7488U], -1.0f, mp64_);
        VBarrier();
        Div(invLen32, w[7488U], invLen32, mp64_);
        VBarrier();
        ResetMask();
        Mul<float, false>(s32[8U * kSoaStride32 + 64U], s32[5U * kSoaStride32 + 64U], invLen32, MASK_PLACEHOLDER, rep64, binary);
        Mul<float, false>(s32[3U * kSoaStride32], s32[6U * kSoaStride32 + 64U], invLen32, MASK_PLACEHOLDER, rep64, binary);
        Mul<float, false>(s32[7U * kSoaStride32 + 64U], s32[4U * kSoaStride32 + 64U], invLen32, MASK_PLACEHOLDER, rep64, binary);
        VBarrier();
    }

    __aicore__ inline void UpdateCoordScaleFromMax(uint32_t redBlks, event_t maxVsEvent)
    {
        WaitFlag<HardEvent::V_S>(maxVsEvent);
        auto maxPairs = scratch_.Get<float>()[16512U];
        float maxAbs = 0.0f;
        for (uint32_t k = 0U; k < redBlks; ++k) {
            const float v = maxPairs.GetValue(2U * k);
            if (v > maxAbs) {
                maxAbs = v;
            }
        }
        coordScale_ = 1.0f;
        if (maxAbs > 0.0f && Finite(maxAbs)) {
            BitsF32 mb{maxAbs};
            int32_t exp2 = static_cast<int32_t>((mb.u >> 23U) & 0xFFU) - 127;
            int32_t shift = 2 - exp2;
            if (shift > 120) {
                shift = 120;
            }
            if (shift < -120) {
                shift = -120;
            }
            BitsF32 sc{0.0f};
            sc.u = static_cast<uint32_t>(shift + 127) << 23U;
            coordScale_ = sc.f;
        }
    }

    __aicore__ inline void ScaleFp16SegmentsAndSelectEff();
    __aicore__ inline void FinalizeAndCastFp16Segments();
    __aicore__ inline void LoadSegments(uint32_t batch, uint32_t firstPos, uint32_t firstCount);

    __aicore__ inline void StepFallbackBlock(
        LocalTensor<float> w, LocalTensor<float> s32, LocalTensor<float> pairs,
        uint32_t b, uint32_t count, uint8_t stepRep);
    __aicore__ inline void FallbackFullFP32(uint32_t pos, uint32_t count);

    __aicore__ inline LocalTensor<half> FineMin(uint32_t chunk, bool);
    __aicore__ inline LocalTensor<half> LookaheadFine(uint32_t chunk);
    __aicore__ inline void BroadcastPoints16(LocalTensor<half> p16, uint32_t count);
    __aicore__ inline void PreparePoints(
        LocalTensor<float> p, uint32_t count,
        LocalTensor<float> xyz, LocalTensor<float> range, LocalTensor<half> p16);
    __aicore__ inline void RestoreLookaheadPoints(uint32_t count);
    __aicore__ inline void PrepareLookahead(LocalTensor<half> p16, uint32_t count);
    __aicore__ inline void PrepareFirstChunkPrefix(uint32_t startChunk, uint8_t stepRep16);
    __aicore__ inline void UpdateRunningMinBeforeChunk(
        uint32_t c, uint32_t startChunk, uint8_t stepRep16, bool prefetched);
    __aicore__ inline void EvalChunkBody16(
        uint32_t c, uint32_t stopChunk, uint32_t count,
        uint8_t stepRep16, bool prefetched, bool lookaheadOutput);
    __aicore__ inline void FinishChunkBody16(
        uint32_t c, uint32_t stopChunk, uint32_t count,
        uint8_t stepRep16, bool prefetched, bool lookaheadOutput,
        LocalTensor<half> x0_16, LocalTensor<half> u0_16,
        LocalTensor<half> z0_16, LocalTensor<half> t0_16,
        LocalTensor<float> meta, LocalTensor<half> s16, LocalTensor<half> sc16);
    __aicore__ inline void ReduceChunkAndPrefetchNext16(
        uint32_t c, uint32_t stopChunk, uint32_t count,
        uint8_t stepRep16, bool prefetched, bool lookaheadOutput,
        uint32_t nextOff16, uint32_t nextShift,
        LocalTensor<half> z0_16, LocalTensor<half> t0_16,
        LocalTensor<float> meta, LocalTensor<half> s16, LocalTensor<half> sc16);
    __aicore__ inline void ScreenChunks(
        uint32_t startChunk, uint32_t stopChunk, uint32_t count,
        bool prefetched, bool lookaheadOutput);
    __aicore__ inline void BuildSurvivalMask();
    __aicore__ inline void ComputeTileScreened(
        LocalTensor<float> p, uint32_t pos, uint32_t count, bool prefetched);

    __aicore__ inline void StartPrimaryAndLookaheadGather(
        LocalTensor<uint8_t> surviveBits, bool primaryPreBroadcasted,
        bool hasLookahead, LocalTensor<T> nextRaw, uint32_t nextCount);
    __aicore__ inline void ScaleAndFoldCodes(bool hasLookahead, uint32_t nextCount);
    __aicore__ inline void SplitPrimaryAndPrepare(
        LocalTensor<uint8_t> surviveBits, bool primaryPreBroadcasted,
        bool hasLookahead, LocalTensor<T> nextRaw, uint32_t nextCount);
    __aicore__ inline uint32_t AssignDupPassIndex(uint32_t point, ExtraDecodeState& st);
    __aicore__ inline void RecordExtraSlot(
        uint32_t point, int32_t gBase, uint32_t extraIdx, ExtraDecodeState& st);
    __aicore__ inline void DecodePointExtraMask(
        uint32_t point, uint32_t c, uint32_t gChunkBase,
        uint32_t byteMask, ExtraDecodeState& st);
    __aicore__ inline void DecodeAllExtras(
        LocalTensor<uint8_t> surviveBits, ExtraDecodeState& st);
    __aicore__ inline void PrepareExtraPhaseVectors(
        uint32_t numExtra, uint32_t& groups, uint32_t& firstSlot, uint32_t& padLen);
    __aicore__ inline void EvalExactPhaseGeometry(uint32_t firstSlot, uint32_t groups);
    __aicore__ inline void FinishExactPhaseGeometry(
        LocalTensor<float> phaseX, LocalTensor<float> phaseY, LocalTensor<float> phaseZ,
        LocalTensor<float> phaseUDot, LocalTensor<float> phaseTempX,
        LocalTensor<float> phaseTempY, LocalTensor<float> phaseTempZ,
        LocalTensor<float> phaseTX, LocalTensor<float> phaseTZ,
        uint8_t rep, const BinaryRepeatParams& binary, const UnaryRepeatParams& unary);
    __aicore__ inline void ReduceCandidatesAndPrefetchNext(
        uint32_t totalPairs, bool nextPrepared);
    __aicore__ inline void StepMergeDupPass(
        LocalTensor<T> d, LocalTensor<float> candidateD2, LocalTensor<float> candidateIdxF,
        LocalTensor<float> candD2, LocalTensor<float> candIdxF, LocalTensor<uint8_t> updateMask,
        LocalTensor<uint32_t> dupCandOffU32, uint32_t dp, uint32_t numDupPasses);
    __aicore__ inline void MergeDuplicatePasses(
        LocalTensor<T> d, LocalTensor<int32_t> idx,
        uint32_t totalPairs, uint32_t numDupPasses);
    __aicore__ inline bool RunExtraPhaseAndCheckFallback(
        uint32_t pos, uint32_t count, LocalTensor<uint8_t> surviveBits,
        event_t v2sEvent, ExtraDecodeState& st);
    __aicore__ inline bool VerifyTileVector(
        uint32_t pos, uint32_t count, LocalTensor<uint8_t> surviveBits,
        LocalTensor<float> tileMaxPairs, bool primaryPreBroadcasted,
        uint32_t nextCount, LocalTensor<T>& nextRaw);
    __aicore__ inline void FinalizeVerifiedOutput(
        uint32_t pos, uint32_t count, uint32_t totalPairs, uint32_t numDupPasses);

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 2> pointQueue_;
    TQue<QuePosition::VECOUT, 1> distanceQueue_;
    TQue<QuePosition::VECOUT, 1> indexQueue_;
    TBuf<TPosition::VECCALC> scratch_, segmentSoa32_, segmentSoa16_, metaBuf_, pointCast_;
    GlobalTensor<T> pointsGm_, segmentsGm_, distanceGm_;
    GlobalTensor<int32_t> indexGm_;
    uint32_t n_, m_, mp64_, numBlocks_, mp128_, numChunks128_, tile_, begin_, end_;
    float coordScale_;
    half zero16_, one16_;
    bool needRebroadcast_, nextPrefixPrepared_;
};

#include "tree_line_prune_screen.h"
#include "tree_line_prune_verify.h"

#endif  // TREE_LINE_PRUNE_H
