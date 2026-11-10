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

#ifndef TREE_LINE_MODE3_H
#define TREE_LINE_MODE3_H

#include "tree_line_common.h"

namespace TreeLineMode3Shared {

using namespace TreeLineCommon;
template <typename T>
__aicore__ inline void CopyMultiSmallInputsToUb(
    LocalTensor<float> w, const GlobalTensor<T>& pointsGm,
    const GlobalTensor<T>& segmentsGm, uint32_t pos, uint32_t batch, uint32_t m, uint32_t count)
{
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyExtParams pcp{1, static_cast<uint32_t>(3U * count * sizeof(T)), 0, 0, 0};
    DataCopyExtParams scp{1, static_cast<uint32_t>(6U * m * sizeof(T)), 0, 0, 0};
    if constexpr (sizeof(T) == sizeof(float)) {
        DataCopyPad(w[20096U].template ReinterpretCast<T>(), pointsGm[3U * pos], pcp, pad);
        DataCopyPad(w[19328U].template ReinterpretCast<T>(), segmentsGm[batch * m * 6U], scp, pad);
    } else {
        DataCopyPad(w[20288U].template ReinterpretCast<T>(), pointsGm[3U * pos], pcp, pad);
        DataCopyPad(w[19712U].template ReinterpretCast<T>(), segmentsGm[batch * m * 6U], scp, pad);
    }
}
template <typename T>
__aicore__ inline void CopySingleSegmentInputsToUb(
    LocalTensor<float> w, const GlobalTensor<T>& pointsGm,
    const GlobalTensor<T>& segmentsGm, uint32_t pos, uint32_t batch, uint32_t count)
{
    DataCopyExtParams pcp{1, static_cast<uint32_t>(3U * count * sizeof(T)), 0, 0, 0};
    DataCopyExtParams scp{1, static_cast<uint32_t>(6U * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    if constexpr (sizeof(T) == sizeof(float)) {
        DataCopyPad(w[5760U].template ReinterpretCast<T>(), segmentsGm[batch * 6U], scp, pad);
        DataCopyPad(w[6144U].template ReinterpretCast<T>(), pointsGm[3U * pos], pcp, pad);
    } else {
        DataCopyPad(w[5824U].template ReinterpretCast<T>(), segmentsGm[batch * 6U], scp, pad);
        DataCopyPad(w[7680U].template ReinterpretCast<T>(), pointsGm[3U * pos], pcp, pad);
    }
}
template <typename T>
__aicore__ inline void CopySinglePointInputsToUb(
    LocalTensor<float> w, uint32_t soaStride,
    const GlobalTensor<T>& pointsGm, const GlobalTensor<T>& segmentsGm,
    uint32_t pos, uint32_t batch, uint32_t m)
{
    DataCopyExtParams pcp{1, static_cast<uint32_t>(3U * sizeof(T)), 0, 0, 0};
    DataCopyExtParams scp{1, static_cast<uint32_t>(6U * m * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    if constexpr (sizeof(T) == sizeof(float)) {
        DataCopyPad(w[18U * soaStride + 576U].template ReinterpretCast<T>(), pointsGm[3U * pos], pcp, pad);
        DataCopyPad(w[11U * soaStride + 256U].template ReinterpretCast<T>(), segmentsGm[batch * m * 6U], scp, pad);
    } else {
        DataCopyPad(w[18U * soaStride + 512U].template ReinterpretCast<T>(), pointsGm[3U * pos], pcp, pad);
        DataCopyPad(w[18U * soaStride + 768U].template ReinterpretCast<T>(), segmentsGm[batch * m * 6U], scp, pad);
    }
}
template <typename T>
__aicore__ inline void PrepareMultiSmallSoA(
    LocalTensor<float> w, LocalTensor<float> ax, LocalTensor<float> ay, LocalTensor<float> az,
    LocalTensor<float> dx, LocalTensor<float> dy, LocalTensor<float> dz,
    LocalTensor<float> px, LocalTensor<float> py, LocalTensor<float> pz,
    uint32_t m, uint32_t count)
{
    auto off12 = w[19136U].ReinterpretCast<uint32_t>();
    auto segOff = w[19264U].ReinterpretCast<uint32_t>();
    auto aosSeg = w[19328U];
    auto segRaw16 = w[19712U].template ReinterpretCast<T>();
    auto ptFloat = w[20096U];
    auto ptRaw16 = w[20288U].template ReinterpretCast<T>();
    if constexpr (sizeof(T) != sizeof(float)) {
        Cast(ptFloat, ptRaw16, RoundMode::CAST_NONE, 3U * count);
        Cast(aosSeg, segRaw16, RoundMode::CAST_NONE, 6U * m);
        VBarrier();
    }
    GatherSegmentAoS(ax, ay, az, dx, dy, dz, aosSeg, segOff, m);
    GatherPointAoS(px, py, pz, ptFloat, off12, count);
    VBarrier();
    const uint8_t brcbRep = static_cast<uint8_t>((count + 7U) >> 3U);
    BroadcastPointTriplet(w[17600U], w[18112U], w[18624U], px, py, pz, brcbRep);
}
__aicore__ inline void EvalMultiSmallSetup(
    LocalTensor<float> w, LocalTensor<float> ax, LocalTensor<float> ay, LocalTensor<float> az,
    LocalTensor<float> dx, LocalTensor<float> dy, LocalTensor<float> dz,
    LocalTensor<float> negMux, LocalTensor<float> negMuy, LocalTensor<float> negMuz,
    uint8_t stepRep)
{
    auto invLen = w[21184U];
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const BinaryRepeatParams bcastBinary{1, 1, 0, 8, 0, 1};
    const UnaryRepeatParams unary{1, 1, 8, 8};
    ResetMask();
    Sub<float, false>(dx, dx, ax, MASK_PLACEHOLDER, 1, binary);
    Sub<float, false>(dy, dy, ay, MASK_PLACEHOLDER, 1, binary);
    Sub<float, false>(dz, dz, az, MASK_PLACEHOLDER, 1, binary);
    VBarrier();
    Mul<float, false>(invLen, dx, dx, MASK_PLACEHOLDER, 1, binary);
    Mul<float, false>(negMux, dy, dy, MASK_PLACEHOLDER, 1, binary);
    Mul<float, false>(negMuy, dz, dz, MASK_PLACEHOLDER, 1, binary);
    Sub<float, false>(w[8256U], ax, w[17600U], MASK_PLACEHOLDER, stepRep, bcastBinary);
    VBarrier();
    Add<float, false>(invLen, invLen, negMux, MASK_PLACEHOLDER, 1, binary);
    Sub<float, false>(w[0U], ay, w[18112U], MASK_PLACEHOLDER, stepRep, bcastBinary);
    VBarrier();
    Add<float, false>(invLen, invLen, negMuy, MASK_PLACEHOLDER, 1, binary);
    Sub<float, false>(w[10304U], az, w[18624U], MASK_PLACEHOLDER, stepRep, bcastBinary);
    VBarrier();
    Maxs<float, false>(invLen, invLen, 1.0e-30f, MASK_PLACEHOLDER, 1, unary);
    VBarrier();
    Div(invLen, w[21248U], invLen, 64U);
    VBarrier();
    ResetMask();
    Mul<float, false>(negMux, dx, invLen, MASK_PLACEHOLDER, 1, binary);
    Mul<float, false>(negMuy, dy, invLen, MASK_PLACEHOLDER, 1, binary);
    Mul<float, false>(negMuz, dz, invLen, MASK_PLACEHOLDER, 1, binary);
    VBarrier();
}
template <typename T>
__aicore__ inline void EvalMultiSmallDistanceAndStore(
    LocalTensor<float> w, LocalTensor<float> dx, LocalTensor<float> dy, LocalTensor<float> dz,
    LocalTensor<float> negMux, LocalTensor<float> negMuy, LocalTensor<float> negMuz,
    const GlobalTensor<T>& distanceGm, const GlobalTensor<int32_t>& indexGm,
    uint32_t pos, uint32_t m, uint32_t count)
{
    const uint8_t stepRep = static_cast<uint8_t>(count);
    EvalBlockStepProjectAndResidual(
        w[8256U], w[0U], w[10304U], w[12352U], w[2048U], w[14400U], w[4096U], w[6144U],
        dx, dy, dz, negMux, negMuy, negMuz, 0U, stepRep);
    auto t0 = w[12352U];
    SumSquares3(t0, w[6144U], w[0U], w[4096U], stepRep);
    auto res = w[20480U];
    WholeReduceMin<float>(
        res, t0, static_cast<int32_t>(m), static_cast<int32_t>(count),
        4, 1, 8, ReduceOrder::ORDER_VALUE_INDEX);
    VBarrier();
    auto off32 = w[19200U].ReinterpretCast<uint32_t>();
    auto outDist32 = w[20992U];
    auto outDistT = w[21056U].template ReinterpretCast<T>();
    auto outIdx = w[21120U].ReinterpretCast<int32_t>();
    Gather(outIdx, res.ReinterpretCast<int32_t>(), off32, 4U, count);
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    Gather(outDist32, res, off32, 0U, count);
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    DataCopyExtParams ic{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(indexGm[pos], outIdx, ic);
    VBarrier();
    FinishSqrtAndCast<T>(outDistT, outDist32, outDist32, count);
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(1));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(1));
    DataCopyExtParams dc{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(distanceGm[pos], outDistT, dc);
}
__aicore__ inline void FinishSinglePointClampAndSq(
    LocalTensor<float> w, uint32_t soaStride, uint8_t repeats, bool addResidual)
{
    const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
    const UnaryRepeatParams unary{1, 1, 8, 8};
    auto x0 = w[0U * soaStride];
    auto y0 = w[1U * soaStride];
    auto z0 = w[2U * soaStride];
    auto t0 = w[3U * soaStride + 64U];
    auto ax = w[5U * soaStride + 128U];
    auto dx = w[8U * soaStride + 192U];
    auto dy = w[9U * soaStride + 192U];
    auto dz = w[10U * soaStride + 192U];
    Mins<float, false>(ax, t0, 1.0f, MASK_PLACEHOLDER, repeats, unary);
    VBarrier();
    Mul<float, false>(dx, ax, dx, MASK_PLACEHOLDER, repeats, binary);
    Mul<float, false>(dy, ax, dy, MASK_PLACEHOLDER, repeats, binary);
    Mul<float, false>(dz, ax, dz, MASK_PLACEHOLDER, repeats, binary);
    VBarrier();
    if (addResidual) {
        Add<float, false>(x0, x0, dx, MASK_PLACEHOLDER, static_cast<uint8_t>(3U * repeats), binary);
    } else {
        Sub<float, false>(x0, x0, dx, MASK_PLACEHOLDER, static_cast<uint8_t>(3U * repeats), binary);
    }
    VBarrier();
    SumSquares3(t0, x0, y0, z0, repeats);
}
}  // namespace TreeLineMode3Shared

template <typename T>
class KernelTreeLineMode3 {
public:
    __aicore__ inline void Init(
        GM_ADDR points, GM_ADDR segments, GM_ADDR distances, GM_ADDR indices,
        const TreeLineMinDistanceTilingData& tiling, TPipe* pipe)
    {
        using namespace TreeLineCommon;
        pipe_ = pipe;
        n_ = tiling.points;
        m_ = tiling.segments;
        mp_ = Align64(m_);
        const uint32_t total = tiling.batch * n_;
        ComputeRange(tiling, total);
        BindGlobalBuffersAndMask(
            pointsGm_, segmentsGm_, distanceGm_, indexGm_,
            points, segments, distances, indices, total, tiling.batch * m_);
        InitScratchBuffer();
    }
    __aicore__ inline void Process()
    {
        if (begin_ >= end_) {
            return;
        }
        const uint32_t count = end_ - begin_;
        if (count == 1U) {
            ProcessSinglePoint(begin_);
        } else if (m_ == 1U && count > 32U) {
            ProcessSingleSegment();
        } else {
            ProcessMultiSmall();
        }
    }
private:
    __aicore__ inline void ComputeRange(
        const TreeLineMinDistanceTilingData& tiling, uint32_t total)
    {
        using namespace TreeLineCommon;
        const uint32_t core = GetBlockIdx();
        const uint32_t blk = tiling.blockDim;
        if (blk == 0U || core >= blk) {
            begin_ = total;
            end_ = total;
            return;
        }
        if (tiling.batch == 1U && total > blk) {
            const uint32_t chunks = (total + 15U) >> 4U;
            begin_ = MinU(total, SafeDivU(chunks * core, blk) << 4U);
            end_ = (core + 1U == blk) ? total : MinU(total, SafeDivU(chunks * (core + 1U), blk) << 4U);
        } else if (tiling.batch >= 2U && blk <= tiling.batch) {
            begin_ = SafeDivU(tiling.batch * core, blk) * n_;
            end_ = SafeDivU(tiling.batch * (core + 1U), blk) * n_;
        } else {
            begin_ = SafeDivU(total * core, blk);
            end_ = SafeDivU(total * (core + 1U), blk);
        }
    }
    __aicore__ inline void InitScratchBuffer()
    {
        const uint32_t count = end_ - begin_;
        if (count == 1U) {
            soaStride_ = (mp_ + 127U) & ~127U;
            pipe_->InitBuffer(scratch_, (24U * soaStride_ + 1024U) * sizeof(float));
        } else if (m_ == 1U && count > 32U) {
            pipe_->InitBuffer(scratch_, 10240U * sizeof(float));
        } else {
            pipe_->InitBuffer(scratch_, 22528U * sizeof(float));
        }
    }
    __aicore__ inline void LoadSingleSegmentInputs(
        LocalTensor<float> w, uint32_t pos, uint32_t batch, uint32_t count, uint32_t k)
    {
        using namespace TreeLineCommon;
        using namespace TreeLineMode3Shared;
        CopySingleSegmentInputsToUb<T>(w, pointsGm_, segmentsGm_, pos, batch, count);
        auto off12Int = w[3712U].ReinterpretCast<int32_t>();
        CreateVecIndex(off12Int, static_cast<int32_t>(0), k);
        Duplicate(w[4224U].ReinterpretCast<int32_t>(), static_cast<int32_t>(0), k);
        VBarrier();
        Muls(off12Int, off12Int, static_cast<int32_t>(12), k);
        VBarrier();
        SyncPipe<HardEvent::MTE2_V>(pipe_);
        if constexpr (sizeof(T) != sizeof(float)) {
            Cast(w[6144U], w[7680U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, 3U * count);
            Cast(w[5760U], w[5824U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, 6U);
            VBarrier();
        }
        GatherPointAoS(w[1600U], w[2112U], w[2624U], w[6144U], off12Int.ReinterpretCast<uint32_t>(), count);
        if constexpr (sizeof(T) == sizeof(float)) {
            SyncPipe<HardEvent::MTE2_S>(pipe_);
        } else {
            SyncPipe<HardEvent::V_S>(pipe_);
        }
    }
    __aicore__ inline void ProcessSingleSegment()
    {
        using namespace TreeLineCommon;
        auto w = scratch_.Get<float>();
        const uint32_t pos = begin_;
        const uint32_t batch = SafeDivU(pos, n_);
        const uint32_t count = end_ - begin_;
        const uint32_t k = Align64(count);
        const uint8_t repeats = static_cast<uint8_t>(k >> 6U);
        LoadSingleSegmentInputs(w, pos, batch, count, k);
        EvalSingleSegmentCore(w[1600U], w[2112U], w[2624U], w[0U], w[3136U], w[5760U], repeats);
        auto outDistT = w[5248U].template ReinterpretCast<T>();
        FinishSqrtAndCast<T>(outDistT, w[4736U], w[3136U], k);
        SyncPipe<HardEvent::V_MTE3>(pipe_);
        DataCopyExtParams dc{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyExtParams ic{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(distanceGm_[pos], outDistT, dc);
        DataCopyPad(indexGm_[pos], w[4224U].ReinterpretCast<int32_t>(), ic);
    }
    __aicore__ inline void LoadSinglePointInputs(LocalTensor<float> w, uint32_t pos)
    {
        using namespace TreeLineCommon;
        using namespace TreeLineMode3Shared;
        const uint32_t batch = SafeDivU(pos, n_);
        CopySinglePointInputsToUb<T>(w, soaStride_, pointsGm_, segmentsGm_, pos, batch, m_);
        auto segOffInt = w[17U * soaStride_ + 256U].ReinterpretCast<int32_t>();
        auto offZeroInt = w[18U * soaStride_ + 384U].ReinterpretCast<int32_t>();
        CreateVecIndex(segOffInt, static_cast<int32_t>(0), mp_);
        Duplicate(offZeroInt, static_cast<int32_t>(0), 8);
        if (m_ < mp_) {
            const float padVal = (m_ > 64U) ? 1.0e18f : 0.0f;
            for (uint32_t k = 0U; k < 6U; ++k) {
                const uint32_t off = (5U + k) * soaStride_ + ((k < 3U) ? 128U : 192U);
                Duplicate(w[off], padVal, mp_);
            }
        }
        VBarrier();
        if (m_ > 64U) {
            Muls(offZeroInt[64U], segOffInt, static_cast<int32_t>(8), 64);
            Muls(w[18U * soaStride_ + 256U].ReinterpretCast<int32_t>(), segOffInt, static_cast<int32_t>(64), 64);
        }
        Muls(segOffInt, segOffInt, static_cast<int32_t>(24), mp_);
        SyncPipe<HardEvent::MTE2_V>(pipe_);
        if constexpr (sizeof(T) != sizeof(float)) {
            Cast(w[18U * soaStride_ + 576U], w[18U * soaStride_ + 512U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, 8U);
            Cast(w[11U * soaStride_ + 256U], w[18U * soaStride_ + 768U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, 6U * m_);
        }
        VBarrier();
    }
    __aicore__ inline void EvalSinglePointDistances(LocalTensor<float> w)
    {
        using namespace TreeLineCommon;
        using namespace TreeLineMode3Shared;
        const uint32_t s = soaStride_;
        auto bcast = w[18U * s + 320U];
        Brcb(bcast, w[18U * s + 576U], 1, BrcbRepeatParams{1, 8});
        GatherSegmentAoS(
            w[5U * s + 128U], w[6U * s + 128U], w[7U * s + 128U],
            w[8U * s + 192U], w[9U * s + 192U], w[10U * s + 192U],
            w[11U * s + 256U], w[17U * s + 256U].ReinterpretCast<uint32_t>(), m_);
        VBarrier();
        const uint8_t repeats = static_cast<uint8_t>(mp_ >> 6U);
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        const BinaryRepeatParams bcastBinary{1, 1, 0, 8, 8, 0};
        const UnaryRepeatParams unary{1, 1, 8, 8};
        ResetMask();
        Sub<float, false>(w[8U * s + 192U], w[8U * s + 192U], w[5U * s + 128U], MASK_PLACEHOLDER, static_cast<uint8_t>(3U * repeats), binary);
        Sub<float, false>(w[0U * s], w[5U * s + 128U], bcast, MASK_PLACEHOLDER, repeats, bcastBinary);
        Sub<float, false>(w[1U * s], w[6U * s + 128U], bcast[8U], MASK_PLACEHOLDER, repeats, bcastBinary);
        Sub<float, false>(w[2U * s], w[7U * s + 128U], bcast[16U], MASK_PLACEHOLDER, repeats, bcastBinary);
        VBarrier();
        Mul<float, false>(w[4U * s + 64U], w[8U * s + 192U], w[8U * s + 192U], MASK_PLACEHOLDER, repeats, binary);
        Mul<float, false>(w[5U * s + 128U], w[9U * s + 192U], w[9U * s + 192U], MASK_PLACEHOLDER, repeats, binary);
        Mul<float, false>(w[6U * s + 128U], w[10U * s + 192U], w[10U * s + 192U], MASK_PLACEHOLDER, repeats, binary);
        Mul<float, false>(w[3U * s + 64U], w[0U * s], w[8U * s + 192U], MASK_PLACEHOLDER, repeats, binary);
        Mul<float, false>(w[7U * s + 128U], w[1U * s], w[9U * s + 192U], MASK_PLACEHOLDER, repeats, binary);
        VBarrier();
        Add<float, false>(w[4U * s + 64U], w[4U * s + 64U], w[5U * s + 128U], MASK_PLACEHOLDER, repeats, binary);
        Add<float, false>(w[3U * s + 64U], w[3U * s + 64U], w[7U * s + 128U], MASK_PLACEHOLDER, repeats, binary);
        Mul<float, false>(w[7U * s + 128U], w[2U * s], w[10U * s + 192U], MASK_PLACEHOLDER, repeats, binary);
        VBarrier();
        Add<float, false>(w[4U * s + 64U], w[4U * s + 64U], w[6U * s + 128U], MASK_PLACEHOLDER, repeats, binary);
        Add<float, false>(w[3U * s + 64U], w[3U * s + 64U], w[7U * s + 128U], MASK_PLACEHOLDER, repeats, binary);
        VBarrier();
        Maxs<float, false>(w[4U * s + 64U], w[4U * s + 64U], 1.0e-30f, MASK_PLACEHOLDER, repeats, unary);
        Muls<float, false>(w[3U * s + 64U], w[3U * s + 64U], -1.0f, MASK_PLACEHOLDER, repeats, unary);
        VBarrier();
        Div(w[3U * s + 64U], w[3U * s + 64U], w[4U * s + 64U], mp_);
        VBarrier();
        ResetMask();
        Maxs<float, false>(w[3U * s + 64U], w[3U * s + 64U], 0.0f, MASK_PLACEHOLDER, repeats, unary);
        VBarrier();
        FinishSinglePointClampAndSq(w, s, repeats, true);
    }
    __aicore__ inline void ReduceSinglePointLargeM(
        LocalTensor<float> w, LocalTensor<T> d, LocalTensor<int32_t> idx,
        LocalTensor<float> r, LocalTensor<uint32_t> offZero)
    {
        using namespace TreeLineCommon;
        const uint32_t numBlocks = mp_ >> 6U;
        auto t0 = w[3U * soaStride_ + 64U];
        auto pairs = w[4U * soaStride_ + 64U];
        auto aos = w[11U * soaStride_ + 256U];
        auto vals = aos[64U];
        auto idxs = aos[128U].ReinterpretCast<int32_t>();
        auto blkOff = offZero[64U];
        auto blkBase64 = w[18U * soaStride_ + 256U].ReinterpretCast<int32_t>();
        WholeReduceMin<float>(
            pairs, t0, 64, static_cast<int32_t>(numBlocks), 1, 1, 8,
            ReduceOrder::ORDER_VALUE_INDEX);
        VBarrier();
        Gather(vals, pairs, blkOff, 0U, 16U);
        Gather(idxs, pairs.ReinterpretCast<int32_t>(), blkOff, 4U, 16U);
        VBarrier();
        WholeReduceMin<float>(
            r, vals, static_cast<int32_t>(numBlocks), 1, 4, 1, 2,
            ReduceOrder::ORDER_VALUE_INDEX);
        Add(idxs, idxs, blkBase64, 16U);
        VBarrier();
        auto idxByteOff = vals.ReinterpretCast<int32_t>();
        Gather(idxByteOff, r.ReinterpretCast<int32_t>(), offZero, 4U, 1U);
        if constexpr (sizeof(T) == sizeof(float)) {
            Sqrt(d.template ReinterpretCast<float>(), r, 1U);
        } else {
            Sqrt(pairs, r, 1U);
        }
        VBarrier();
        Muls(idxByteOff, idxByteOff, static_cast<int32_t>(4), 1U);
        if constexpr (sizeof(T) != sizeof(float)) {
            Cast(d, pairs, RoundMode::CAST_RINT, 1U);
        }
        VBarrier();
        Gather(idx, idxs, idxByteOff.ReinterpretCast<uint32_t>(), 0U, 1U);
    }
    __aicore__ inline void ProcessSinglePoint(uint32_t pos)
    {
        using namespace TreeLineCommon;
        auto w = scratch_.Get<float>();
        LoadSinglePointInputs(w, pos);
        EvalSinglePointDistances(w);
        auto d = w[18U * soaStride_ + 640U].template ReinterpretCast<T>();
        auto idx = w[18U * soaStride_ + 704U].ReinterpretCast<int32_t>();
        auto offZero = w[18U * soaStride_ + 384U].ReinterpretCast<uint32_t>();
        auto r = w[11U * soaStride_ + 256U];
        if (m_ <= 64U) {
            WholeReduceMin<float>(
                r, w[3U * soaStride_ + 64U], static_cast<int32_t>(m_), 1, 4, 1, 8,
                ReduceOrder::ORDER_VALUE_INDEX);
            VBarrier();
            Gather(idx, r.ReinterpretCast<int32_t>(), offZero, 4U, 1U);
            FinishSqrtAndCast<T>(d, w[4U * soaStride_ + 64U], r, 1U);
        } else {
            ReduceSinglePointLargeM(w, d, idx, r, offZero);
        }
        SyncPipe<HardEvent::V_MTE3>(pipe_);
        DataCopyExtParams dc{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyExtParams ic{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(distanceGm_[pos], d, dc);
        DataCopyPad(indexGm_[pos], idx, ic);
    }
    __aicore__ inline void ProcessMultiSmall()
    {
        using namespace TreeLineCommon;
        using namespace TreeLineMode3Shared;
        auto w = scratch_.Get<float>();
        const uint32_t pos = begin_;
        const uint32_t count = end_ - begin_;
        CopyMultiSmallInputsToUb<T>(w, pointsGm_, segmentsGm_, pos, SafeDivU(pos, n_), m_, count);
        auto off12Int = w[19136U].ReinterpretCast<int32_t>();
        CreateVecIndex(off12Int, static_cast<int32_t>(0), 64U);
        Duplicate(w[21248U], -1.0f, 64U);
        if (m_ < 64U) {
            Duplicate(w[16640U], 0.0f, 320U);
            Duplicate(w[17280U], 0.0f, 320U);
        }
        VBarrier();
        Muls(w[19200U].ReinterpretCast<int32_t>(), off12Int, static_cast<int32_t>(32), 64U);
        Muls(w[19264U].ReinterpretCast<int32_t>(), off12Int, static_cast<int32_t>(24), 64U);
        Muls(off12Int, off12Int, static_cast<int32_t>(12), 64U);
        VBarrier();
        SyncPipe<HardEvent::MTE2_V>(pipe_);
        PrepareMultiSmallSoA<T>(
            w, w[16640U], w[16768U], w[16896U], w[17280U], w[17408U], w[17536U],
            w[16704U], w[16832U], w[16960U], m_, count);
        EvalMultiSmallSetup(
            w, w[16640U], w[16768U], w[16896U], w[17280U], w[17408U], w[17536U],
            w[17024U], w[17088U], w[17152U], static_cast<uint8_t>(count));
        EvalMultiSmallDistanceAndStore<T>(
            w, w[17280U], w[17408U], w[17536U], w[17024U], w[17088U], w[17152U],
            distanceGm_, indexGm_, pos, m_, count);
    }
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> scratch_;
    GlobalTensor<T> pointsGm_, segmentsGm_, distanceGm_;
    GlobalTensor<int32_t> indexGm_;
    uint32_t n_, m_, mp_, soaStride_, begin_, end_;
};
#endif  // TREE_LINE_MODE3_H
