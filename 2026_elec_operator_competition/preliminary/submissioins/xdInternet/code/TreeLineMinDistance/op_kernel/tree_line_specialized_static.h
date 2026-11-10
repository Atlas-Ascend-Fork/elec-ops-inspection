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

#ifndef TREE_LINE_SPECIALIZED_STATIC_H
#define TREE_LINE_SPECIALIZED_STATIC_H

#include "tree_line_pair32.h"

// MODE 10: Static-UB + constexpr fixedTiling specialization for Point 5 (4, 127, 129)
template <typename T>
class KernelTreeLineSmallMStatic {
public:
    __aicore__ inline void Init(
        GM_ADDR points, GM_ADDR segments, GM_ADDR distances, GM_ADDR indices,
        const TreeLineMinDistanceTilingData& tiling)
    {
        using namespace TreeLineCommon;
        n_ = tiling.points;
        m_ = tiling.segments;
        mp_ = Align64(m_);
        soaStride_ = (mp_ + 127U) & ~127U;
        const uint32_t total = tiling.batch * n_;
        const uint32_t core = GetBlockIdx();
        const uint32_t blk = tiling.blockDim;
        begin_ = SafeDivU(total * core, blk);
        end_ = SafeDivU(total * (core + 1U), blk);
        BindGlobalBuffersAndMask(
            pointsGm_, segmentsGm_, distanceGm_, indexGm_,
            points, segments, distances, indices, total, tiling.batch * m_);
    }

    __aicore__ inline void Process()
    {
        using namespace TreeLineCommon;
        if (begin_ >= end_) {
            return;
        }
        LocalTensor<float> w(TPosition::VECCALC, 0U, 41472U);
        const uint32_t pos = begin_;
        const uint32_t batch = SafeDivU(pos, n_);
        const uint32_t count = end_ - begin_;
        LoadAndGatherInputs(w, pos, batch, count);
        SetupSegmentsAndEval(w, count);
        ReduceAndStore(w, pos, count);
    }

private:
    __aicore__ inline void LoadAndGatherInputs(
        LocalTensor<float> w, uint32_t pos, uint32_t batch, uint32_t count)
    {
        using namespace TreeLineCommon;
        auto off32Int = w[38016U].ReinterpretCast<int32_t>();
        auto off12Int = w[38080U].ReinterpretCast<int32_t>();
        auto segOffInt = w[38144U].ReinterpretCast<int32_t>();
        DataCopyExtParams scp{1, static_cast<uint32_t>(6U * m_ * sizeof(T)), 0, 0, 0};
        DataCopyExtParams pcp{1, static_cast<uint32_t>(3U * count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        if constexpr (sizeof(T) == sizeof(float)) {
            DataCopyPad(w[40640U].template ReinterpretCast<T>(), pointsGm_[3U * pos], pcp, pad);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            DataCopyPad(w[38336U].template ReinterpretCast<T>(), segmentsGm_[batch * m_ * 6U], scp, pad);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(1));
        } else {
            DataCopyPad(w[40832U].template ReinterpretCast<T>(), pointsGm_[3U * pos], pcp, pad);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            DataCopyPad(w[39488U].template ReinterpretCast<T>(), segmentsGm_[batch * m_ * 6U], scp, pad);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(1));
        }
        ExpandStaticIndex64(segOffInt);
        Muls(off12Int, segOffInt, static_cast<int32_t>(12), 64U);
        Muls(off32Int, segOffInt, static_cast<int32_t>(32), 64U);
        Muls(segOffInt, segOffInt, static_cast<int32_t>(24), 64U);
        VBarrier();
        Adds(segOffInt[64U], segOffInt, static_cast<int32_t>(64 * 24), 64U);
        Duplicate(segOffInt[128U], static_cast<int32_t>((m_ - 1U) * 24U), 64U);
        WaitAndUnpackSoA(w, count);
    }

    __aicore__ inline void WaitAndUnpackSoA(LocalTensor<float> w, uint32_t count)
    {
        using namespace TreeLineCommon;
        auto s = w[33024U];
        const uint32_t highBase = 3U * soaStride_ + 64U;
        auto bcast = w[35584U];
        auto off12 = w[38080U].ReinterpretCast<uint32_t>();
        auto off = w[38144U].ReinterpretCast<uint32_t>();
        auto aosSeg = w[38336U];
        auto ptFloat = w[40640U];
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
        if constexpr (sizeof(T) != sizeof(float)) {
            Cast(ptFloat, w[40832U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, 3U * count);
        }
        VBarrier();
        GatherPointAoS(bcast[0U], bcast[64U], bcast[128U], ptFloat, off12, count);
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(1));
        if constexpr (sizeof(T) != sizeof(float)) {
            Cast(aosSeg, w[39488U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, 6U * m_);
        }
        VBarrier();
        GatherSegmentAoS(
            s[0U], s[soaStride_], s[2U * soaStride_],
            s[highBase + soaStride_], s[highBase + 2U * soaStride_], s[highBase],
            aosSeg, off, mp_);
        VBarrier();
        const uint8_t brcbRepeats = static_cast<uint8_t>((count + 7U) >> 3U);
        BroadcastPointTriplet(
            bcast[192U], bcast[704U], bcast[1216U],
            bcast[0U], bcast[64U], bcast[128U], brcbRepeats);
    }

    __aicore__ inline void SetupSegmentsAndEval(LocalTensor<float> w, uint32_t count)
    {
        using namespace TreeLineCommon;
        auto s = w[33024U];
        const uint32_t highBase = 3U * soaStride_ + 64U;
        const uint32_t muxBase = 6U * soaStride_ + 128U;
        auto bcast = w[35584U];
        ComputeSegmentNegMu(
            s[0U], s[soaStride_], s[2U * soaStride_],
            s[highBase + soaStride_], s[highBase + 2U * soaStride_], s[highBase],
            s[muxBase + soaStride_], s[muxBase + 2U * soaStride_], s[muxBase],
            w[37632U], w[37824U], mp_);
        const uint32_t numBlk = mp_ >> 6U;
        EvalSplitHeavySubAndDot(
            w, s[0U], s[soaStride_], s[2U * soaStride_],
            s[muxBase + soaStride_], s[muxBase + 2U * soaStride_], s[muxBase],
            bcast[192U], bcast[704U], bcast[1216U], 0U, numBlk, count);
        EvalSplitHeavyResidualAndReduce(
            w, w[37376U],
            s[highBase + soaStride_], s[highBase + 2U * soaStride_], s[highBase],
            0U, numBlk, count, count, 4);
        VBarrier();
    }

    __aicore__ inline void ReduceAndStore(LocalTensor<float> w, uint32_t pos, uint32_t count)
    {
        using namespace TreeLineCommon;
        auto outDistT = w[41088U].template ReinterpretCast<T>();
        auto outIdx = w[41152U].ReinterpretCast<int32_t>();
        DecodeHierarchicalMin<T, false>(
            outDistT, outIdx, w[41024U], w[41216U], w[37376U],
            w[41344U].ReinterpretCast<int32_t>(), w[41408U].ReinterpretCast<int32_t>(),
            w[38016U].ReinterpretCast<uint32_t>(), mp_ >> 6U, count, 1);
        SyncStatic<HardEvent::V_MTE3>();
        DataCopyExtParams dc{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyExtParams ic{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(distanceGm_[pos], outDistT, dc);
        DataCopyPad(indexGm_[pos], outIdx, ic);
    }

    GlobalTensor<T> pointsGm_, segmentsGm_, distanceGm_;
    GlobalTensor<int32_t> indexGm_;
    uint32_t n_, m_, mp_, soaStride_, begin_, end_;
};

// MODE 12: Static-UB + Zero-Transpose specialization for Point 7 (2, 1003, 513) FP32
class KernelTreeLine513Static {
public:
    __aicore__ inline void Process(
        GM_ADDR points, GM_ADDR segments, GM_ADDR distances, GM_ADDR indices)
    {
        using namespace TreeLineCommon;
        const uint32_t core = GetBlockIdx();
        const uint32_t pos = (2006U * core) / 40U;
        const uint32_t end = (2006U * (core + 1U)) / 40U;
        const uint32_t count = end - pos;
        const uint32_t batch = pos / 1003U;
        BindGlobalBuffersAndMask(
            pointsGm_, segmentsGm_, distanceGm_, indexGm_,
            points, segments, distances, indices, 2006U, 1026U);
        LoadSegments(batch, pos, count);
        ComputeSegments(pos, count);
    }

private:
    static constexpr uint32_t m_ = 513U;
    static constexpr uint32_t mp_ = 576U;
    static constexpr uint32_t tile_ = 64U;
    static constexpr uint32_t corePad_ = 52U;
    static constexpr uint32_t soaStride_ = 640U;

    __aicore__ inline LocalTensor<float> UB()
    {
        return LocalTensor<float>(TPosition::VECCALC, 0U, 44160U);
    }

    __aicore__ inline void Build513Offsets(
        LocalTensor<int32_t> segOffInt, LocalTensor<int32_t> off12Int, LocalTensor<int32_t> off32Int)
    {
        using namespace TreeLineCommon;
        ExpandStaticIndex64(segOffInt);
        Muls(off12Int, segOffInt, static_cast<int32_t>(12), tile_);
        Muls(off32Int, segOffInt, static_cast<int32_t>(32), tile_);
        Muls(segOffInt, segOffInt, static_cast<int32_t>(24), 64U);
        VBarrier();
        Adds(segOffInt[64U], segOffInt, static_cast<int32_t>(64 * 24), 64U);
        VBarrier();
        Adds(segOffInt[128U], segOffInt, static_cast<int32_t>(128 * 24), 128U);
        VBarrier();
        Adds(segOffInt[256U], segOffInt, static_cast<int32_t>(256 * 24), 256U);
        Duplicate(segOffInt[512U], static_cast<int32_t>(512 * 24), 64U);
    }

    __aicore__ inline void LoadSegments(uint32_t batch, uint32_t pos, uint32_t count)
    {
        using namespace TreeLineCommon;
        auto w = UB();
        auto s = w[33024U];
        auto rawPoints = w[40640U];
        auto off32Int = w[41152U].ReinterpretCast<int32_t>();
        auto off12Int = off32Int[64U];
        auto segOffInt = w[9U * mp_].ReinterpretCast<int32_t>();
        DataCopyExtParams scp{1, static_cast<uint32_t>(6U * m_ * sizeof(float)), 0, 0, 0};
        DataCopyExtParams pcp{1, static_cast<uint32_t>(count * 3U * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(rawPoints, pointsGm_[3U * pos], pcp, pad);
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
        DataCopyPad(w, segmentsGm_[batch * m_ * 6U], scp, pad);
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(1));
        Build513Offsets(segOffInt, off12Int, off32Int);
        auto bcast = w[38912U];
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
        VBarrier();
        GatherPointAoS(bcast[0U], bcast[64U], bcast[128U], rawPoints, off12Int.ReinterpretCast<uint32_t>(), count);
        VBarrier();
        BroadcastPointTriplet(bcast[192U], bcast[704U], bcast[1216U], bcast[0U], bcast[64U], bcast[128U], 7U);
        const uint32_t highBase = 3U * soaStride_ + 64U;
        const uint32_t muxBase = 6U * soaStride_ + 128U;
        WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(1));
        VBarrier();
        GatherSegmentAoS(
            s[0U], s[soaStride_], s[2U * soaStride_],
            s[highBase + soaStride_], s[highBase + 2U * soaStride_], s[highBase],
            w, segOffInt.ReinterpretCast<uint32_t>(), mp_);
        VBarrier();
        ComputeSegmentNegMu(
            s[0U], s[soaStride_], s[2U * soaStride_],
            s[highBase + soaStride_], s[highBase + 2U * soaStride_], s[highBase],
            s[muxBase + soaStride_], s[muxBase + 2U * soaStride_], s[muxBase],
            w, w[mp_], mp_);
    }

    __aicore__ inline void StepFullBlock513(
        LocalTensor<float> w, LocalTensor<float> s, LocalTensor<float> bcast,
        uint32_t b, uint32_t count, uint8_t stepRepeats)
    {
        using namespace TreeLineCommon;
        const uint32_t highBase = 3U * soaStride_ + 64U;
        const uint32_t muxBase = 6U * soaStride_ + 128U;
        const uint32_t segOff = b * 64U;
        const uint32_t nextSegOff = (b + 1U) * 64U;
        EvalBlockStepProjectAndResidual(
            w[16448U], w[0U], w[20544U], w[24640U], w[4096U], w[28736U], w[8192U], w[12288U],
            s[highBase + soaStride_], s[highBase + 2U * soaStride_], s[highBase],
            s[muxBase + soaStride_], s[muxBase + 2U * soaStride_], s[muxBase], segOff, stepRepeats);
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        const BinaryRepeatParams bcastBinary{1, 1, 0, 8, 0, 1};
        auto t0 = w[24640U];
        Mul<float, false>(t0, w[12288U], w[12288U], MASK_PLACEHOLDER, stepRepeats, binary);
        if (b + 1U < 8U) {
            Sub<float, false>(w[16448U], s[nextSegOff], bcast[192U], MASK_PLACEHOLDER, stepRepeats, bcastBinary);
        } else {
            Sub<float, false>(w[16448U], s[512U], bcast[0U], MASK_PLACEHOLDER, 1U, binary);
        }
        VBarrier();
        MulAddDst<float, float, false>(t0, w[0U], w[0U], MASK_PLACEHOLDER, stepRepeats, binary);
        if (b + 1U < 8U) {
            Sub<float, false>(w[20544U], s[2U * soaStride_ + nextSegOff], bcast[1216U], MASK_PLACEHOLDER, stepRepeats, bcastBinary);
        } else {
            Sub<float, false>(w[20544U], s[2U * soaStride_ + 512U], bcast[128U], MASK_PLACEHOLDER, 1U, binary);
        }
        VBarrier();
        MulAddDst<float, float, false>(t0, w[8192U], w[8192U], MASK_PLACEHOLDER, stepRepeats, binary);
        if (b + 1U < 8U) {
            Sub<float, false>(w[0U], s[soaStride_ + nextSegOff], bcast[704U], MASK_PLACEHOLDER, stepRepeats, bcastBinary);
        } else {
            Sub<float, false>(w[0U], s[soaStride_ + 512U], bcast[64U], MASK_PLACEHOLDER, 1U, binary);
        }
        VBarrier();
        WholeReduceMin<float>(
            w[42112U + 2U * b], t0, 64, static_cast<int32_t>(count),
            16, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
        ResetMask();
    }

    __aicore__ inline void ComputeTailSegment512(LocalTensor<float> w, LocalTensor<float> s)
    {
        using namespace TreeLineCommon;
        const uint32_t highBase = 3U * soaStride_ + 64U;
        const uint32_t muxBase = 6U * soaStride_ + 128U;
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        const UnaryRepeatParams unary{1, 1, 8, 8};
        Mul<float, false>(w[4096U], w[16448U], s[muxBase + soaStride_ + 512U], MASK_PLACEHOLDER, 1U, binary);
        Mul<float, false>(w[28736U], w[0U], s[muxBase + 2U * soaStride_ + 512U], MASK_PLACEHOLDER, 1U, binary);
        Mul<float, false>(w[8192U], w[20544U], s[muxBase + 512U], MASK_PLACEHOLDER, 1U, binary);
        VBarrier();
        Add<float, false>(w[24640U], w[4096U], w[28736U], MASK_PLACEHOLDER, 1U, binary);
        VBarrier();
        Add<float, false>(w[24640U], w[24640U], w[8192U], MASK_PLACEHOLDER, 1U, binary);
        VBarrier();
        Maxs<float, false>(w[4096U], w[24640U], 0.0f, MASK_PLACEHOLDER, 1U, unary);
        VBarrier();
        Mins<float, false>(w[24640U], w[4096U], 1.0f, MASK_PLACEHOLDER, 1U, unary);
        VBarrier();
        Mul<float, false>(w[12288U], w[24640U], s[highBase + soaStride_ + 512U], MASK_PLACEHOLDER, 1U, binary);
        Mul<float, false>(w[28736U], w[24640U], s[highBase + 2U * soaStride_ + 512U], MASK_PLACEHOLDER, 1U, binary);
        Mul<float, false>(w[4096U], w[24640U], s[highBase + 512U], MASK_PLACEHOLDER, 1U, binary);
        VBarrier();
        Add<float, false>(w[12288U], w[16448U], w[12288U], MASK_PLACEHOLDER, 1U, binary);
        Add<float, false>(w[0U], w[0U], w[28736U], MASK_PLACEHOLDER, 1U, binary);
        Add<float, false>(w[4096U], w[20544U], w[4096U], MASK_PLACEHOLDER, 1U, binary);
        VBarrier();
        Mul<float, false>(w[8192U], w[12288U], w[12288U], MASK_PLACEHOLDER, 1U, binary);
        VBarrier();
        MulAddDst<float, float, false>(w[8192U], w[0U], w[0U], MASK_PLACEHOLDER, 1U, binary);
        VBarrier();
        MulAddDst<float, float, false>(w[8192U], w[4096U], w[4096U], MASK_PLACEHOLDER, 1U, binary);
    }

    __aicore__ inline void ComputeSegmentsBlocks1(uint32_t count, uint32_t cPad)
    {
        using namespace TreeLineCommon;
        auto w = UB();
        auto s = w[33024U];
        auto bcast = w[38912U];
        const uint8_t stepRepeats = static_cast<uint8_t>(cPad);
        const BinaryRepeatParams bcastBinary{1, 1, 0, 8, 0, 1};
        ResetMask();
        Sub<float, false>(w[16448U], s[0U], bcast[192U], MASK_PLACEHOLDER, stepRepeats, bcastBinary);
        Sub<float, false>(w[0U], s[soaStride_], bcast[704U], MASK_PLACEHOLDER, stepRepeats, bcastBinary);
        Sub<float, false>(w[20544U], s[2U * soaStride_], bcast[1216U], MASK_PLACEHOLDER, stepRepeats, bcastBinary);
        VBarrier();
        for (uint32_t b = 0U; b < 8U; ++b) {
            StepFullBlock513(w, s, bcast, b, count, stepRepeats);
        }
        ComputeTailSegment512(w, s);
    }

    __aicore__ inline void ComputeSegments(uint32_t pos, uint32_t count)
    {
        using namespace TreeLineCommon;
        ResetMask();
        auto w = UB();
        auto off32 = w[41152U].ReinterpretCast<uint32_t>();
        ComputeSegmentsBlocks1(count, corePad_);
        VBarrier();
        auto d = w[41024U];
        auto idx = w[41088U].ReinterpretCast<int32_t>();
        auto pairs = w[42112U];
        auto r = w[2112U];
        auto idxByteOff = w[64U].ReinterpretCast<int32_t>();
        auto localIdxVec = w[128U].ReinterpretCast<int32_t>();
        uint64_t valueMask[2] = {0x5555555555555555ULL & ((1ULL << 16U) - 1ULL), 0ULL};
        WholeReduceMin<float>(
            r, pairs, valueMask, static_cast<int32_t>(count),
            4, 1, 4, ReduceOrder::ORDER_VALUE_INDEX);
        VBarrier();
        Gather(w, r, off32, 0U, count);
        Gather(idx, r.ReinterpretCast<int32_t>(), off32, 4U, count);
        Muls(localIdxVec, off32.ReinterpretCast<int32_t>(), static_cast<int32_t>(4), count);
        VBarrier();
        MergeTailSegment512AndStore(w, d, idx, pairs, idxByteOff, localIdxVec, pos, count);
    }

    __aicore__ inline void MergeTailSegment512AndStore(
        LocalTensor<float> w, LocalTensor<float> d, LocalTensor<int32_t> idx,
        LocalTensor<float> pairs, LocalTensor<int32_t> idxByteOff,
        LocalTensor<int32_t> localIdxVec, uint32_t pos, uint32_t count)
    {
        using namespace TreeLineCommon;
        auto t1 = w[4096U];
        auto v0 = w[8192U];
        auto u0 = w[28736U];
        Muls(idxByteOff, idx, static_cast<int32_t>(4), count);
        Muls(idx, idx, static_cast<int32_t>(32), count);
        Compare(u0.ReinterpretCast<uint8_t>(), v0, w, CMPMODE::LT, 64U);
        Duplicate(t1.ReinterpretCast<int32_t>(), static_cast<int32_t>(512), 64U);
        VBarrier();
        Add(idxByteOff, idxByteOff, localIdxVec, count);
        Min(w, w, v0, count);
        VBarrier();
        Gather(localIdxVec, pairs.ReinterpretCast<int32_t>(),
            idxByteOff.ReinterpretCast<uint32_t>(), 4U, count);
        Sqrt(d, w, count);
        VBarrier();
        Add(idx, idx, localIdxVec, count);
        VBarrier();
        Select(idx.ReinterpretCast<float>(), u0.ReinterpretCast<uint8_t>(),
            t1, idx.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, 64U);
        SyncStatic<HardEvent::V_MTE3>();
        DataCopyExtParams dc{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyExtParams ic{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(distanceGm_[pos], d, dc);
        DataCopyPad(indexGm_[pos], idx, ic);
    }

    GlobalTensor<float> pointsGm_, segmentsGm_, distanceGm_;
    GlobalTensor<int32_t> indexGm_;
};

#endif  // TREE_LINE_SPECIALIZED_STATIC_H
