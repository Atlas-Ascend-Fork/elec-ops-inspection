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

#ifndef TREE_LINE_STATIC_BATCH_MODES_H
#define TREE_LINE_STATIC_BATCH_MODES_H

#include "tree_line_mode3.h"

// MODE 6: B=1, N=1, M>64 (Point 10). Static tensor setup + Maxs + MulAddDst.
template <typename T>
class KernelTreeLineSinglePointStatic {
public:
    __aicore__ inline void Init(
        GM_ADDR points, GM_ADDR segments, GM_ADDR distances,
        GM_ADDR indices, const TreeLineMinDistanceTilingData& tiling)
    {
        using namespace TreeLineCommon;
        m_ = tiling.segments;
        mp_ = Align64(m_);
        soaStride_ = (mp_ + 127U) & ~127U;
        BindGlobalBuffersAndMask(
            pointsGm_, segmentsGm_, distanceGm_, indexGm_,
            points, segments, distances, indices, 1U, m_);
    }

    __aicore__ inline void Process()
    {
        const bool isActiveCore = (GetBlockIdx() == 0U);
        const uint32_t activeM = isActiveCore ? m_ : 64U;
        const uint32_t activeMp = isActiveCore ? mp_ : 64U;
        LocalTensor<float> w(TPosition::VECCALC, 0U, 24U * soaStride_ + 1024U);
        LoadAndGatherStatic(w, isActiveCore, activeM, activeMp);
        ComputeDistancesStatic(w, activeMp);
        ReduceAndStoreStatic(w, isActiveCore, activeMp);
    }

private:
    __aicore__ inline void BuildStaticSegOffsets(LocalTensor<int32_t> segOffInt, uint32_t activeMp)
    {
        using namespace TreeLineCommon;
        constexpr uint32_t kSegByteStride = 6U * sizeof(T);
        #pragma unroll
        for (uint32_t i = 0U; i < 8U; ++i) {
            segOffInt.SetValue(i, static_cast<int32_t>(kSegByteStride * i));
        }
        SyncStatic<HardEvent::S_V>();
        #pragma clang loop unroll(disable)
        for (uint32_t filled = 8U; filled < activeMp; filled <<= 1U) {
            const uint32_t take = MinU(filled, activeMp - filled);
            Adds(segOffInt[filled], segOffInt, static_cast<int32_t>(filled * kSegByteStride), take);
            VBarrier();
        }
        Adds(segOffInt[soaStride_], segOffInt, static_cast<int32_t>(sizeof(T)), activeMp);
        Adds(segOffInt[2U * soaStride_], segOffInt, static_cast<int32_t>(2U * sizeof(T)), activeMp);
        VBarrier();
    }

    __aicore__ inline void LoadAndGatherStatic(
        LocalTensor<float> w, bool isActiveCore, uint32_t activeM, uint32_t activeMp)
    {
        using namespace TreeLineCommon;
        using namespace TreeLineMode3Shared;
        if (isActiveCore) {
            CopySinglePointInputsToUb<T>(w, soaStride_, pointsGm_, segmentsGm_, 0U, 0U, m_);
        }
        auto segOffInt = w[22U * soaStride_].ReinterpretCast<int32_t>();
        BuildStaticSegOffsets(segOffInt, activeMp);
        if (isActiveCore) {
            SyncStatic<HardEvent::MTE2_V>();
        }
        GatherToSoAStatic(w, segOffInt.ReinterpretCast<uint32_t>(), isActiveCore, activeM);
    }

    __aicore__ inline void GatherToSoAStatic(
        LocalTensor<float> w, LocalTensor<uint32_t> off, bool isActiveCore, uint32_t activeM)
    {
        using namespace TreeLineCommon;
        auto ax = w[5U * soaStride_ + 128U];
        auto dx = w[8U * soaStride_ + 192U];
        auto aos = w[11U * soaStride_ + 256U];
        auto bcast = w[18U * soaStride_ + 320U];
        auto ptRaw = w[18U * soaStride_ + 512U].template ReinterpretCast<T>();
        auto pFloat = w[18U * soaStride_ + 576U];
        auto segRaw = w[18U * soaStride_ + 768U].template ReinterpretCast<T>();
        const BrcbRepeatParams brcbParams{1, 8};
        if constexpr (sizeof(T) != sizeof(float)) {
            auto aos16 = aos.template ReinterpretCast<T>();
            auto ax16 = aos16[0U * soaStride_ + 128U];
            auto dx16 = aos16[3U * soaStride_ + 256U];
            const uint32_t gCount = isActiveCore ? (3U * m_) : 64U;
            const uint32_t cCount = isActiveCore ? (3U * soaStride_) : 64U;
            Cast(pFloat, ptRaw, RoundMode::CAST_NONE, 8U);
            Gather(ax16, segRaw, off, 0U, gCount);
            Gather(dx16, segRaw, off, 3U * sizeof(T), gCount);
            VBarrier();
            Brcb(bcast, pFloat, 1, brcbParams);
            Cast(ax, ax16, RoundMode::CAST_NONE, cCount);
            Cast(dx, dx16, RoundMode::CAST_NONE, cCount);
            VBarrier();
        } else {
            Brcb(bcast, pFloat, 1, brcbParams);
            Gather(ax, aos, off, 0U, 3U * activeM);
            Gather(dx, aos, off, 3U * sizeof(T), 3U * activeM);
            VBarrier();
        }
    }

    __aicore__ inline void ComputeDistancesStatic(LocalTensor<float> w, uint32_t activeMp)
    {
        using namespace TreeLineCommon;
        using namespace TreeLineMode3Shared;
        const uint32_t s = soaStride_;
        auto bcast = w[18U * s + 320U];
        const uint8_t repeats = static_cast<uint8_t>(activeMp >> 6U);
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        const BinaryRepeatParams bcastBinary{1, 1, 0, 8, 8, 0};
        const UnaryRepeatParams unary{1, 1, 8, 8};
        ResetMask();
        Sub<float, false>(w[8U * s + 192U], w[5U * s + 128U], w[8U * s + 192U], MASK_PLACEHOLDER, static_cast<uint8_t>(3U * repeats), binary);
        Sub<float, false>(w[0U * s], w[5U * s + 128U], bcast, MASK_PLACEHOLDER, repeats, bcastBinary);
        Sub<float, false>(w[1U * s], w[6U * s + 128U], bcast[8U], MASK_PLACEHOLDER, repeats, bcastBinary);
        Sub<float, false>(w[2U * s], w[7U * s + 128U], bcast[16U], MASK_PLACEHOLDER, repeats, bcastBinary);
        VBarrier();
        Mul<float, false>(w[5U * s + 128U], w[8U * s + 192U], w[8U * s + 192U], MASK_PLACEHOLDER, static_cast<uint8_t>(3U * repeats), binary);
        Mul<float, false>(w[3U * s + 64U], w[0U * s], w[8U * s + 192U], MASK_PLACEHOLDER, static_cast<uint8_t>(2U * repeats), binary);
        Mul<float, false>(w[11U * s + 256U], w[2U * s], w[10U * s + 192U], MASK_PLACEHOLDER, repeats, binary);
        VBarrier();
        Add<float, false>(w[5U * s + 128U], w[5U * s + 128U], w[6U * s + 128U], MASK_PLACEHOLDER, repeats, binary);
        Add<float, false>(w[3U * s + 64U], w[3U * s + 64U], w[4U * s + 64U], MASK_PLACEHOLDER, repeats, binary);
        VBarrier();
        Add<float, false>(w[4U * s + 64U], w[5U * s + 128U], w[7U * s + 128U], MASK_PLACEHOLDER, repeats, binary);
        Add<float, false>(w[3U * s + 64U], w[3U * s + 64U], w[11U * s + 256U], MASK_PLACEHOLDER, repeats, binary);
        VBarrier();
        Maxs<float, false>(w[4U * s + 64U], w[4U * s + 64U], 1.0e-30f, MASK_PLACEHOLDER, repeats, unary);
        Maxs<float, false>(w[3U * s + 64U], w[3U * s + 64U], 0.0f, MASK_PLACEHOLDER, repeats, unary);
        VBarrier();
        Div(w[3U * s + 64U], w[3U * s + 64U], w[4U * s + 64U], activeMp);
        VBarrier();
        ResetMask();
        FinishSinglePointClampAndSq(w, s, repeats, false);
    }

    __aicore__ inline void ReduceAndStoreStatic(
        LocalTensor<float> w, bool isActiveCore, uint32_t activeMp)
    {
        using namespace TreeLineCommon;
        auto t0 = w[3U * soaStride_ + 64U];
        auto pairs = w[4U * soaStride_ + 64U];
        auto aos = w[11U * soaStride_ + 256U];
        auto d = w[18U * soaStride_ + 640U].template ReinterpretCast<T>();
        auto idx = w[18U * soaStride_ + 704U].ReinterpretCast<int32_t>();
        const uint32_t numBlocks = activeMp >> 6U;
        WholeReduceMin<float>(
            pairs, t0, 64, static_cast<int32_t>(numBlocks), 1, 1, 8,
            ReduceOrder::ORDER_VALUE_INDEX);
        VBarrier();
        uint64_t valueMask[2] = {0x55555555ULL, 0ULL};
        WholeReduceMin<float>(
            aos, pairs, valueMask, 1, 4, 1, 1, ReduceOrder::ORDER_VALUE_INDEX);
        if (!isActiveCore) {
            return;
        }
        SetFlag<HardEvent::V_S>(static_cast<event_t>(0));
        VBarrier();
        FinishSqrtAndCast<T>(d, aos[192U], aos, 1U);
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
        WaitFlag<HardEvent::V_S>(static_cast<event_t>(0));
        const uint32_t twoBlk = static_cast<uint32_t>(aos.ReinterpretCast<int32_t>().GetValue(1U));
        const int32_t bestIdx = static_cast<int32_t>(twoBlk << 5U) +
            pairs.ReinterpretCast<int32_t>().GetValue(twoBlk + 1U);
        idx.SetValue(0U, bestIdx);
        SetFlag<HardEvent::S_MTE3>(static_cast<event_t>(0));
        DataCopyExtParams dc{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyExtParams ic{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        WaitFlag<HardEvent::S_MTE3>(static_cast<event_t>(0));
        DataCopyPad(indexGm_[0U], idx, ic);
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
        DataCopyPad(distanceGm_[0U], d, dc);
    }

    GlobalTensor<T> pointsGm_, segmentsGm_, distanceGm_;
    GlobalTensor<int32_t> indexGm_;
    uint32_t m_, mp_, soaStride_;
};

// MODE 7: B=1, M=1, 64..512 nominal points/core (Point 9). Static tensor setup + Axpy + MulAddDst.
template <typename T>
class KernelTreeLineSingleSegmentStatic {
public:
    __aicore__ inline void Init(
        GM_ADDR points, GM_ADDR segments, GM_ADDR distances,
        GM_ADDR indices, const TreeLineMinDistanceTilingData& tiling)
    {
        using namespace TreeLineCommon;
        const uint32_t total = tiling.points;
        const uint32_t core = GetBlockIdx();
        const uint32_t blk = tiling.blockDim;
        const uint32_t chunks = (total + 15U) >> 4U;
        begin_ = MinU(total, SafeDivU(chunks * core, blk) << 4U);
        end_ = (core + 1U == blk)
                   ? total
                   : MinU(total, SafeDivU(chunks * (core + 1U), blk) << 4U);
        BindGlobalBuffersAndMask(
            pointsGm_, segmentsGm_, distanceGm_, indexGm_,
            points, segments, distances, indices, total, 1U);
    }

    __aicore__ inline void Process()
    {
        using namespace TreeLineCommon;
        if (begin_ >= end_) {
            return;
        }
        LocalTensor<float> w(TPosition::VECCALC, 0U, 10240U);
        const uint32_t pos = begin_;
        const uint32_t count = end_ - begin_;
        const uint32_t k = Align64(count);
        const uint8_t repeats = static_cast<uint8_t>(k >> 6U);
        LoadSingleSegStaticInputs(w, pos, count, k);
        EvalSingleSegmentCore(w[1600U], w[2112U], w[2624U], w[0U], w[512U], w[5760U], repeats);
        auto outDistT = w[1024U].template ReinterpretCast<T>();
        FinishSqrtAndCast<T>(outDistT, w[3136U], w[512U], k);
        SyncStatic<HardEvent::V_MTE3>();
        DataCopyExtParams dc{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(distanceGm_[pos], outDistT, dc);
    }

private:
    __aicore__ inline void LoadSingleSegStaticInputs(
        LocalTensor<float> w, uint32_t pos, uint32_t count, uint32_t k)
    {
        using namespace TreeLineCommon;
        DataCopyExtParams pcp{1, static_cast<uint32_t>(3U * count * sizeof(T)), 0, 0, 0};
        DataCopyExtParams scp{1, static_cast<uint32_t>(6U * sizeof(T)), 0, 0, 0};
        DataCopyExtParams ic{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        if constexpr (sizeof(T) == sizeof(float)) {
            DataCopyPad(w[5760U].template ReinterpretCast<T>(), segmentsGm_[0U], scp, pad);
            SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
            DataCopyPad(w[6144U].template ReinterpretCast<T>(), pointsGm_[3U * pos], pcp, pad);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
        } else {
            DataCopyPad(w[5824U].template ReinterpretCast<T>(), segmentsGm_[0U], scp, pad);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            DataCopyPad(w[7680U].template ReinterpretCast<T>(), pointsGm_[3U * pos], pcp, pad);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(1));
        }
        auto off12Int = w[3712U].ReinterpretCast<int32_t>();
        auto outIdx = w[4224U].ReinterpretCast<int32_t>();
        Duplicate(outIdx, static_cast<int32_t>(0), k);
        CreateVecIndex(off12Int, static_cast<int32_t>(0), k);
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
        VBarrier();
        Muls(off12Int, off12Int, static_cast<int32_t>(12), k);
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
        DataCopyPad(indexGm_[pos], outIdx, ic);
        WaitAndGatherSingleSeg(w, count);
    }

    __aicore__ inline void WaitAndGatherSingleSeg(LocalTensor<float> w, uint32_t count)
    {
        using namespace TreeLineCommon;
        auto off12 = w[3712U].ReinterpretCast<uint32_t>();
        if constexpr (sizeof(T) == sizeof(float)) {
            WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            GatherPointAoS(w[1600U], w[2112U], w[2624U], w[6144U], off12, count);
        } else {
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            Cast(w[5760U], w[5824U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, 6U);
            SetFlag<HardEvent::V_S>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(1));
            Cast(w[6144U], w[7680U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, 3U * count);
            VBarrier();
            GatherPointAoS(w[1600U], w[2112U], w[2624U], w[6144U], off12, count);
            WaitFlag<HardEvent::V_S>(static_cast<event_t>(0));
        }
    }

    GlobalTensor<T> pointsGm_, segmentsGm_, distanceGm_;
    GlobalTensor<int32_t> indexGm_;
    uint32_t begin_, end_;
};

// MODE 8 & 9: Static-tensor MultiSmall path for Points 4 & 6 (no TPipe!).
template <typename T>
class KernelTreeLineMultiSmallStatic {
public:
    __aicore__ inline void Init(
        GM_ADDR points, GM_ADDR segments, GM_ADDR distances, GM_ADDR indices,
        const TreeLineMinDistanceTilingData& tiling)
    {
        using namespace TreeLineCommon;
        const uint32_t total = tiling.batch * tiling.points;
        m_ = tiling.segments;
        n_ = tiling.points;
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blk = tiling.blockDim;
        if (blk == tiling.batch) {
            begin_ = blockIdx * n_;
            end_ = begin_ + n_;
        } else {
            const uint32_t totalGroups = (total + 15U) / 16U;
            const uint32_t baseGroups = SafeDivU(totalGroups, blk);
            const uint32_t remGroups = SafeModU(totalGroups, blk);
            const uint32_t startGroup = blockIdx * baseGroups + (blockIdx < remGroups ? blockIdx : remGroups);
            const uint32_t myGroups = baseGroups + (blockIdx < remGroups ? 1U : 0U);
            begin_ = MinU(total, startGroup * 16U);
            end_ = MinU(total, begin_ + myGroups * 16U);
        }
        BindGlobalBuffersAndMask(
            pointsGm_, segmentsGm_, distanceGm_, indexGm_,
            points, segments, distances, indices, total, tiling.batch * m_);
    }

    __aicore__ inline void Process()
    {
        using namespace TreeLineCommon;
        using namespace TreeLineMode3Shared;
        if (begin_ >= end_) {
            return;
        }
        LocalTensor<float> w(TPosition::VECCALC, 0U, 22528U);
        const uint32_t count = end_ - begin_;
        const uint32_t pos = begin_;
        CopyMultiSmallInputsToUb<T>(w, pointsGm_, segmentsGm_, pos, SafeDivU(pos, n_), m_, count);
        auto off12Int = w[19136U].ReinterpretCast<int32_t>();
        ExpandStaticIndex64(off12Int);
        Duplicate(w[21248U], -1.0f, 64U);
        if (m_ < 64U) {
            Duplicate(w[16512U], 0.0f, 320U);
            Duplicate(w[16960U], 0.0f, 320U);
        }
        Muls(w[19200U].ReinterpretCast<int32_t>(), off12Int, static_cast<int32_t>(32), 64U);
        Muls(w[19264U].ReinterpretCast<int32_t>(), off12Int, static_cast<int32_t>(24), 64U);
        Muls(off12Int, off12Int, static_cast<int32_t>(12), 64U);
        VBarrier();
        SyncStatic<HardEvent::MTE2_V>();
        PrepareMultiSmallSoA<T>(
            w, w[16512U], w[16576U], w[16640U], w[16960U], w[17024U], w[17088U],
            w[17152U], w[17280U], w[17408U], m_, count);
        EvalMultiSmallSetup(
            w, w[16512U], w[16576U], w[16640U], w[16960U], w[17024U], w[17088U],
            w[16704U], w[16768U], w[16896U], static_cast<uint8_t>(count));
        EvalMultiSmallDistanceAndStore<T>(
            w, w[16960U], w[17024U], w[17088U], w[16704U], w[16768U], w[16896U],
            distanceGm_, indexGm_, pos, m_, count);
    }

private:
    GlobalTensor<T> pointsGm_, segmentsGm_, distanceGm_;
    GlobalTensor<int32_t> indexGm_;
    uint32_t n_, m_, begin_, end_;
};

#endif  // TREE_LINE_STATIC_BATCH_MODES_H
