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

#ifndef TREE_LINE_GENERAL_MODES_H
#define TREE_LINE_GENERAL_MODES_H

#include "tree_line_common.h"

template <typename T, uint32_t MODE>
class KernelTreeLineMinDistance {
public:
    __aicore__ inline void Init(
        GM_ADDR points, GM_ADDR segments, GM_ADDR distances, GM_ADDR indices,
        const TreeLineMinDistanceTilingData& tiling, TPipe* pipe)
    {
        using namespace TreeLineCommon;
        pipe_ = pipe;
        tile_ = tiling.tilePoints;
        n_ = tiling.points;
        m_ = tiling.segments;
        mp_ = Align64(m_);
        const uint32_t total = tiling.batch * n_;
        ComputeCoreRange(tiling, total);
        BindGlobalBuffersAndMask(
            pointsGm_, segmentsGm_, distanceGm_, indexGm_,
            points, segments, distances, indices, total, tiling.batch * m_);
        InitBuffers();
        offsetsReady_ = false;
    }

    __aicore__ inline void Process()
    {
        using namespace TreeLineCommon;
        for (uint32_t pos = begin_; pos < end_;) {
            const uint32_t batch = SafeDivU(pos, n_);
            const uint32_t batchEnd = MinU(end_, (batch + 1U) * n_);
            uint32_t count = MinU(tile_, batchEnd - pos);
            LoadSegments(batch, pos, count);
            ProcessBatchTiles(pos, batchEnd, count);
        }
    }

private:
    __aicore__ inline void ComputeCoreRange(
        const TreeLineMinDistanceTilingData& tiling, uint32_t total)
    {
        using namespace TreeLineCommon;
        const uint32_t blk = tiling.blockDim;
        const uint32_t core = GetBlockIdx();
        if (blk == 0U || core >= blk) {
            begin_ = end_ = total;
            return;
        }
        if (MODE == 0U && tiling.batch == 1U && total >= blk * 32U) {
            const uint32_t chunks = (total + 15U) >> 4U;
            begin_ = MinU(total, SafeDivU(chunks * core, blk) << 4U);
            end_ = (core + 1U == blk)
                       ? total
                       : MinU(total, SafeDivU(chunks * (core + 1U), blk) << 4U);
            return;
        }
        if (MODE == 1U && tiling.batch > 1U && blk >= tiling.batch &&
            SafeModU(blk, tiling.batch) == 0U) {
            const uint32_t cpb = SafeDivU(blk, tiling.batch);
            if (cpb > 0U && n_ >= cpb * 64U) {
                const uint32_t myBatch = SafeDivU(core, cpb);
                const uint32_t subCore = SafeModU(core, cpb);
                const uint32_t chunks = (n_ + 63U) >> 6U;
                const uint32_t subBegin = MinU(n_, SafeDivU(chunks * subCore, cpb) << 6U);
                const uint32_t subEnd = (subCore + 1U == cpb)
                    ? n_
                    : MinU(n_, SafeDivU(chunks * (subCore + 1U), cpb) << 6U);
                begin_ = myBatch * n_ + subBegin;
                end_ = myBatch * n_ + subEnd;
                return;
            }
        }
        begin_ = SafeDivU(total * core, blk);
        end_ = SafeDivU(total * (core + 1U), blk);
    }

    __aicore__ inline void InitBuffers()
    {
        using namespace TreeLineCommon;
        const uint32_t baseWidth = mp_ > tile_ ? mp_ : tile_;
        soaStride_ = (MODE != 0U) ? ((mp_ + 127U) & ~127U) : mp_;
        width_ = (MODE == 0U) ? ((baseWidth + 127U) & ~127U) : baseWidth;
        const uint32_t maxCorePts = MinU(tile_, end_ - begin_);
        corePad_ = (m_ > 64U && (end_ - begin_) <= 64U) ? ((maxCorePts + 3U) & ~3U) : 64U;
        const uint32_t rawGroup = (m_ <= 64U) ? 4096U : 8192U;
        groupStride_ = ((rawGroup + 255U) & ~255U) + 64U;
        const uint32_t scratchFloats = (MODE != 0U)
            ? ((3U * groupStride_ + rawGroup + 255U) & ~255U)
            : (16U * width_ + 512U);
        pipe_->InitBuffer(scratch_, scratchFloats * sizeof(float));
        const uint32_t soaFloats = (MODE != 0U)
            ? ((9U * soaStride_ + 128U + 255U) & ~255U)
            : (9U * soaStride_);
        pipe_->InitBuffer(segmentSoa_, soaFloats * sizeof(float));
        const uint32_t ptSoaFloats = (MODE == 0U) ? (3U * width_ + 64U) : 1728U;
        pipe_->InitBuffer(pointSoa_, ptSoaFloats * sizeof(float));
        InitAuxBuffers(baseWidth);
    }

    __aicore__ inline void InitAuxBuffers(uint32_t baseWidth)
    {
        pipe_->InitBuffer(pointQueue_, 2, 3U * tile_ * sizeof(T));
        pipe_->InitBuffer(distanceQueue_, 1, tile_ * sizeof(T));
        pipe_->InitBuffer(indexQueue_, 1, tile_ * sizeof(int32_t));
        if constexpr (sizeof(T) != sizeof(float) && MODE == 0U) {
            pipe_->InitBuffer(pointCast_, 3U * tile_ * sizeof(float));
        }
        if (MODE == 0U || m_ <= 64U) {
            pipe_->InitBuffer(results_, (MODE ? 8U : 2U) * tile_ * sizeof(float));
        }
        const uint32_t baseOffCount = (MODE != 0U) ? (2U * tile_) : tile_;
        pipe_->InitBuffer(offsets_, baseOffCount * sizeof(uint32_t));
        if constexpr (MODE == 0U) {
            pipe_->InitBuffer(mask_, ((baseWidth + 255U) / 256U) * 128U);
        }
        if constexpr (MODE == 1U || MODE == 10U) {
            if (m_ > 64U) {
                pipe_->InitBuffer(reduceWork_, 2048U * sizeof(float));
            }
        }
    }

    __aicore__ inline void ProcessBatchTiles(
        uint32_t& pos, uint32_t batchEnd, uint32_t& count)
    {
        using namespace TreeLineCommon;
        while (pos < batchEnd) {
            auto raw = pointQueue_.DeQue<T>();
            const uint32_t next = pos + count;
            const uint32_t nextCount = MinU(tile_, batchEnd - next);
            if (nextCount != 0U) {
                CopyPointTileToQueue(pointQueue_, pointsGm_, next, nextCount);
            }
            LocalTensor<float> p;
            if constexpr (sizeof(T) == sizeof(float)) {
                p = raw.template ReinterpretCast<float>();
            } else {
                p = (MODE == 0U) ? pointCast_.Get<float>() : scratch_.Get<float>()[256U];
                Cast(p, raw, RoundMode::CAST_NONE, 3U * count);
                VBarrier();
            }
            if constexpr (MODE == 0U) {
                ComputePoints(p, pos, count);
            } else {
                ComputeSegments(p, pos, count);
            }
            pointQueue_.FreeTensor(raw);
            pos = next;
            count = nextCount;
        }
    }

    __aicore__ inline void EnsureMode0PointOffsets()
    {
        using namespace TreeLineCommon;
        if (!offsetsReady_) {
            offsetsReady_ = true;
            auto pointOffsets = offsets_.Get<int32_t>();
            CreateVecIndex(pointOffsets, static_cast<int32_t>(0), tile_);
            VBarrier();
            Muls(pointOffsets, pointOffsets, static_cast<int32_t>(12), tile_);
            VBarrier();
        }
    }

    __aicore__ inline void LoadSegmentsMode0SmallScalar(
        uint32_t batch, uint32_t firstPos, uint32_t firstCount)
    {
        using namespace TreeLineCommon;
        auto w = scratch_.Get<float>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(6U * m_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        LocalTensor<float> aos = w;
        if constexpr (sizeof(T) == sizeof(float)) {
            DataCopyPad(w.template ReinterpretCast<T>(), segmentsGm_[batch * m_ * 6U], cp, pad);
            EnsureMode0PointOffsets();
            SyncPipe<HardEvent::MTE2_S>(pipe_);
        } else {
            auto raw = w[6U * mp_].template ReinterpretCast<T>();
            DataCopyPad(raw, segmentsGm_[batch * m_ * 6U], cp, pad);
            EnsureMode0PointOffsets();
            SyncPipe<HardEvent::MTE2_V>(pipe_);
            Cast(aos, raw, RoundMode::CAST_NONE, 6U * m_);
            VBarrier();
            SyncPipe<HardEvent::V_S>(pipe_);
        }
        PopulateScalarSegmentSoA(aos);
        CopyPointTileToQueue(pointQueue_, pointsGm_, firstPos, firstCount);
    }

    __aicore__ inline void PopulateScalarSegmentSoA(LocalTensor<float> aos)
    {
        using namespace TreeLineCommon;
        auto s = segmentSoa_.Get<float>();
        for (uint32_t j = 0U; j < m_; ++j) {
            float ax0 = 0.0f, ay0 = 0.0f, az0 = 0.0f, dx0 = 0.0f, dy0 = 0.0f, dz0 = 0.0f;
            ReadSegment6(aos, 6U * j, ax0, ay0, az0, dx0, dy0, dz0);
            const float len = (dx0 * dx0 + dy0 * dy0) + dz0 * dz0;
            const float inv = (len > 0.0f) ? (1.0f / len) : 0.0f;
            s.SetValue(j, ax0);
            s.SetValue(soaStride_ + j, ay0);
            s.SetValue(2U * soaStride_ + j, az0);
            s.SetValue(3U * soaStride_ + j, dz0);
            s.SetValue(4U * soaStride_ + j, dx0);
            s.SetValue(5U * soaStride_ + j, dy0);
            s.SetValue(6U * soaStride_ + j, dz0 * inv);
            s.SetValue(7U * soaStride_ + j, dx0 * inv);
            s.SetValue(8U * soaStride_ + j, dy0 * inv);
        }
    }

    __aicore__ inline void PrepareSegmentVectorOffsets(LocalTensor<int32_t> segOffInt)
    {
        using namespace TreeLineCommon;
        auto s = segmentSoa_.Get<float>();
        const uint32_t highBase = (MODE != 0U) ? (3U * soaStride_ + 64U) : (3U * soaStride_);
        auto pointOffsets = offsets_.Get<int32_t>();
        CreateVecIndex(segOffInt, static_cast<int32_t>(0), mp_);
        if (!offsetsReady_) {
            CreateVecIndex(pointOffsets, static_cast<int32_t>(0), tile_);
        }
        if (m_ < mp_) {
            const float padVal = (MODE != 0U && m_ > 64U) ? 1.0e18f : 0.0f;
            Duplicate(s[0U], padVal, mp_);
            Duplicate(s[soaStride_], padVal, mp_);
            Duplicate(s[2U * soaStride_], padVal, mp_);
            Duplicate(s[highBase + soaStride_], padVal, mp_);
            Duplicate(s[highBase + 2U * soaStride_], padVal, mp_);
            Duplicate(s[highBase], padVal, mp_);
        }
        VBarrier();
        Muls(segOffInt, segOffInt, static_cast<int32_t>(24), mp_);
        if (!offsetsReady_) {
            offsetsReady_ = true;
            if constexpr (MODE == 0U) {
                Muls(pointOffsets, pointOffsets, static_cast<int32_t>(12), tile_);
            } else {
                Muls(pointOffsets[tile_], pointOffsets, static_cast<int32_t>(12), tile_);
                Muls(pointOffsets, pointOffsets, static_cast<int32_t>(32), tile_);
            }
        }
    }

    __aicore__ inline void ComputeSegmentInvLen(LocalTensor<float> invLen, LocalTensor<float> u)
    {
        using namespace TreeLineCommon;
        const uint8_t repeats = static_cast<uint8_t>(mp_ >> 6U);
        if constexpr (MODE == 0U) {
            auto mask = mask_.Get<uint8_t>();
            Compares(mask, invLen, 0.0f, CMPMODE::GT, mp_);
            Duplicate(u, 1.0f, mp_);
            VBarrier();
            Select(invLen, mask, invLen, 1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mp_);
            VBarrier();
            Div(invLen, u, invLen, mp_);
            VBarrier();
            Select(invLen, mask, invLen, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, mp_);
            VBarrier();
        } else {
            Maxs<float, false>(invLen, invLen, 1.0e-30f, MASK_PLACEHOLDER, repeats, UnaryRepeatParams{1, 1, 8, 8});
            Duplicate(u, -1.0f, mp_);
            VBarrier();
            Div(invLen, u, invLen, mp_);
            VBarrier();
        }
    }

    __aicore__ inline void LoadSegments(uint32_t batch, uint32_t firstPos, uint32_t firstCount)
    {
        using namespace TreeLineCommon;
        if constexpr (MODE == 0U) {
            if (m_ <= 16U) {
                LoadSegmentsMode0SmallScalar(batch, firstPos, firstCount);
                return;
            }
        }
        auto w = scratch_.Get<float>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(6U * m_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        if constexpr (sizeof(T) == sizeof(float)) {
            DataCopyPad(w.template ReinterpretCast<T>(), segmentsGm_[batch * m_ * 6U], cp, pad);
        } else {
            DataCopyPad(w[6U * mp_].template ReinterpretCast<T>(), segmentsGm_[batch * m_ * 6U], cp, pad);
        }
        CopyPointTileToQueue(pointQueue_, pointsGm_, firstPos, firstCount);
        auto segOffInt = w[9U * mp_].ReinterpretCast<int32_t>();
        PrepareSegmentVectorOffsets(segOffInt);
        SyncPipe<HardEvent::MTE2_V>(pipe_);
        if constexpr (sizeof(T) != sizeof(float)) {
            Cast(w, w[6U * mp_].template ReinterpretCast<T>(), RoundMode::CAST_NONE, 6U * m_);
        }
        VBarrier();
        SetupVectorSegmentSoA(w, segOffInt.ReinterpretCast<uint32_t>());
    }

    __aicore__ inline void SetupVectorSegmentSoA(LocalTensor<float> w, LocalTensor<uint32_t> off)
    {
        using namespace TreeLineCommon;
        auto s = segmentSoa_.Get<float>();
        const uint32_t highBase = (MODE != 0U) ? (3U * soaStride_ + 64U) : (3U * soaStride_);
        const uint32_t muxBase = (MODE != 0U) ? (6U * soaStride_ + 128U) : (6U * soaStride_);
        auto ax = s[0U];
        auto ay = s[soaStride_];
        auto az = s[2U * soaStride_];
        auto dz = s[highBase];
        auto dx = s[highBase + soaStride_];
        auto dy = s[highBase + 2U * soaStride_];
        auto muz = s[muxBase];
        auto mux = s[muxBase + soaStride_];
        auto muy = s[muxBase + 2U * soaStride_];
        GatherSegmentAoS(ax, ay, az, dx, dy, dz, w, off, m_);
        VBarrier();
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        const uint8_t repeats = static_cast<uint8_t>(mp_ >> 6U);
        ResetMask();
        Sub<float, false>(dx, dx, ax, MASK_PLACEHOLDER, repeats, binary);
        Sub<float, false>(dy, dy, ay, MASK_PLACEHOLDER, repeats, binary);
        Sub<float, false>(dz, dz, az, MASK_PLACEHOLDER, repeats, binary);
        VBarrier();
        Mul<float, false>(w, dx, dx, MASK_PLACEHOLDER, repeats, binary);
        Mul<float, false>(mux, dy, dy, MASK_PLACEHOLDER, repeats, binary);
        Mul<float, false>(muy, dz, dz, MASK_PLACEHOLDER, repeats, binary);
        VBarrier();
        Add<float, false>(w, w, mux, MASK_PLACEHOLDER, repeats, binary);
        VBarrier();
        Add<float, false>(w, w, muy, MASK_PLACEHOLDER, repeats, binary);
        VBarrier();
        ComputeSegmentInvLen(w, w[mp_]);
        ResetMask();
        Mul<float, false>(muy, dy, w, MASK_PLACEHOLDER, repeats, binary);
        Mul<float, false>(muz, dz, w, MASK_PLACEHOLDER, repeats, binary);
        Mul<float, false>(mux, dx, w, MASK_PLACEHOLDER, repeats, binary);
        VBarrier();
        if constexpr (MODE == 0U) {
            SyncPipe<HardEvent::V_S>(pipe_);
        }
    }

    __aicore__ inline void EvalQuadSegmentsMode0(
        LocalTensor<float> px, LocalTensor<float> py, LocalTensor<float> pz,
        LocalTensor<float> best, LocalTensor<float> bestIdx,
        uint32_t j, uint32_t k, uint8_t repeats);

    __aicore__ inline void FinishQuadSegmentsMode0(
        LocalTensor<float> apx[4], LocalTensor<float> apy[4], LocalTensor<float> apz[4], LocalTensor<float> t[4],
        const float dx[4], const float dy[4], const float dz[4], const float mux[4], const float muy[4], const float muz[4],
        LocalTensor<uint8_t> maskBase, uint32_t maskBytes, LocalTensor<float> best, LocalTensor<float> bestIdx,
        uint32_t j, uint32_t k, uint8_t repeats, const BinaryRepeatParams& binary, const UnaryRepeatParams& unary);

    __aicore__ inline void ReduceQuadSegmentsMode0(
        LocalTensor<float> apx[4], LocalTensor<float> apy[4],
        LocalTensor<float> apz[4], LocalTensor<float> t[4],
        LocalTensor<uint8_t> maskBase, uint32_t maskBytes,
        LocalTensor<float> best, LocalTensor<float> bestIdx,
        uint32_t j, uint32_t k, uint8_t repeats, const BinaryRepeatParams& binary);

    __aicore__ inline void EvalSingleSegmentTailMode0(
        LocalTensor<float> px, LocalTensor<float> py, LocalTensor<float> pz,
        LocalTensor<float> best, LocalTensor<float> bestIdx,
        uint32_t j, uint32_t k, uint8_t repeats);

    __aicore__ inline void ComputePoints(LocalTensor<float> aos, uint32_t pos, uint32_t count);

    __aicore__ inline void FinishComputePointsOutput(
        LocalTensor<float> best, LocalTensor<float> bestIdx,
        uint32_t pos, uint32_t count, uint32_t k);

    __aicore__ inline void StepSingleBlockMode1(
        LocalTensor<float> w, LocalTensor<float> s, LocalTensor<float> bcast,
        LocalTensor<float> pairs, uint32_t b, uint32_t numBlocks,
        uint32_t count, uint8_t stepRepeats);

    __aicore__ inline void ComputeSegmentsBlocks1(uint32_t count, uint32_t cPad);

    __aicore__ inline void EvalCompactBlockGroup(
        uint32_t bStart, uint32_t numBlk, uint32_t count, uint32_t cPad);

    __aicore__ inline void FinishCompactBlockGroup(
        LocalTensor<float> x0, LocalTensor<float> y0, LocalTensor<float> z0, LocalTensor<float> t0,
        LocalTensor<float> s, uint32_t highBase, uint32_t muxBase,
        uint32_t bStart, uint32_t numBlk, uint32_t count, uint32_t blockFloats,
        uint8_t stepRepeats, uint8_t totalRepeats);

    template <bool SplitHeavy>
    __aicore__ inline void SegmentBlockGroup(
        uint32_t bStart, uint32_t numBlk, uint32_t count, uint32_t cPad);

    __aicore__ inline void ComputeSegments(LocalTensor<float> p, uint32_t pos, uint32_t count);

    __aicore__ inline void DispatchSegmentBlocks(uint32_t count, uint32_t cPad);

    __aicore__ inline void ReduceAndStoreSegments(
        LocalTensor<uint32_t> off32, uint32_t pos, uint32_t count);

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 2> pointQueue_;
    TQue<QuePosition::VECOUT, 1> distanceQueue_;
    TQue<QuePosition::VECOUT, 1> indexQueue_;
    TBuf<TPosition::VECCALC> segmentSoa_, pointCast_, pointSoa_, scratch_, results_, offsets_, mask_, reduceWork_;
    GlobalTensor<T> pointsGm_, segmentsGm_, distanceGm_;
    GlobalTensor<int32_t> indexGm_;
    uint32_t n_, m_, mp_, corePad_, soaStride_, groupStride_, tile_, width_, begin_, end_;
    bool offsetsReady_;
};

#include "tree_line_general_compute.h"

#endif  // TREE_LINE_GENERAL_MODES_H
