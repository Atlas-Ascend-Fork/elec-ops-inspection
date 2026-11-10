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

#ifndef TREE_LINE_PAIR32_H
#define TREE_LINE_PAIR32_H

#include "tree_line_common.h"

// MODE 5 & MODE 21: Pair32 specialization
template <typename T>
class KernelTreeLinePair32 {
public:
    __aicore__ inline void Init(
        GM_ADDR points, GM_ADDR segments, GM_ADDR distances, GM_ADDR indices,
        const TreeLineMinDistanceTilingData& tiling)
    {
        using namespace TreeLineCommon;
        const uint32_t step = (tiling.tilePoints > 0U && tiling.tilePoints <= 24U) ? tiling.tilePoints : 2U;
        n_ = tiling.points;
        m_ = tiling.segments;
        pos_ = GetBlockIdx() * step;
        count_ = (pos_ < n_) ? ((pos_ + step <= n_) ? step : (n_ - pos_)) : 0U;
        BindGlobalBuffersAndMask(
            pointsGm_, segmentsGm_, distanceGm_, indexGm_,
            points, segments, distances, indices, n_, m_);
    }

    __aicore__ inline void Process()
    {
        if (count_ == 0U) {
            return;
        }
        LocalTensor<float> w(TPosition::VECCALC, 0U, 4096U);
        const uint8_t reps = static_cast<uint8_t>((m_ <= 8U) ? ((count_ + 7U) >> 3U) : ((count_ + 1U) >> 1U));
        const uint32_t lanes = static_cast<uint32_t>(reps) * 64U;
        IssueInputCopies(w);
        if (m_ <= 8U) {
            PrepareSmallM8(w, reps, lanes);
        } else {
            PrepareMediumM32(w, reps, lanes);
        }
        ComputePair32Distances(w, reps, lanes);
        ReduceAndStorePair32(w);
    }

private:
    __aicore__ inline void IssueInputCopies(LocalTensor<float> w)
    {
        const uint32_t pointElems = 3U * count_;
        DataCopyExtParams pc{1, pointElems * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyExtParams sc{1, 6U * m_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        if constexpr (sizeof(T) == sizeof(float)) {
            if (m_ <= 8U) {
                DataCopyPad(w[2816U].template ReinterpretCast<T>(), segmentsGm_, sc, pad);
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
                DataCopyPad(w[3200U].template ReinterpretCast<T>(), pointsGm_[3U * pos_], pc, pad);
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(1));
                return;
            }
        }
        DataCopyPad(w[3008U].template ReinterpretCast<T>(), segmentsGm_, sc, pad);
        DataCopyPad(w[3280U].template ReinterpretCast<T>(), pointsGm_[3U * pos_], pc, pad);
    }

    __aicore__ inline void GatherSegAndPoints(LocalTensor<float> w, uint32_t lanes)
    {
        using namespace TreeLineCommon;
        auto segOff = w[3360U].template ReinterpretCast<uint32_t>();
        auto pointOff = w[3424U].template ReinterpretCast<uint32_t>();
        GatherSegmentAoS(
            w[1280U], w[1536U], w[1792U], w[2048U], w[2304U], w[2560U],
            w[2816U], segOff, 64U);
        GatherPointAoS(w[0U], w[256U], w[512U], w[3200U], pointOff, lanes);
        VBarrier();
    }

    __aicore__ inline void PrepareSmallM8(LocalTensor<float> w, uint8_t reps, uint32_t lanes)
    {
        using namespace TreeLineCommon;
        auto segOffInt = w[3360U].template ReinterpretCast<int32_t>();
        auto pointOffInt = w[3424U].template ReinterpretCast<int32_t>();
        auto outOffInt = w[3680U].template ReinterpretCast<int32_t>();
        auto outIdx = w[4000U].template ReinterpretCast<int32_t>();
        #pragma unroll
        for (uint32_t i = 0U; i < 8U; ++i) {
            segOffInt.SetValue(i, static_cast<int32_t>((i < m_) ? (24U * i) : 0U));
            outOffInt.SetValue(i, static_cast<int32_t>(32U * i));
            outIdx.SetValue(i, static_cast<int32_t>((i < count_) ? (12U * i) : 0U));
            if (reps > 2U) {
                outIdx.SetValue(16U + i, static_cast<int32_t>((16U + i < count_) ? (12U * (16U + i)) : 0U));
            }
        }
        SyncStatic<HardEvent::S_V>();
        if constexpr (sizeof(T) != sizeof(float)) {
            Duplicate(w[3200U], 0.0f, 64U);
            Duplicate(w[2816U], 0.0f, 192U);
        }
        if (reps > 1U) {
            Adds(outOffInt[8U], outOffInt, static_cast<int32_t>(256), 8U);
            Adds(outIdx[8U], outIdx, static_cast<int32_t>(96), 8U);
            if (reps > 2U) {
                Adds(outOffInt[16U], outOffInt, static_cast<int32_t>(512), 8U);
            }
        }
        Adds(segOffInt[8U], segOffInt, static_cast<int32_t>(0), 8U);
        VBarrier();
        Brcb(pointOffInt.template ReinterpretCast<uint32_t>(),
            outIdx.template ReinterpretCast<uint32_t>(), reps, BrcbRepeatParams{1, 8});
        Adds(segOffInt[16U], segOffInt, static_cast<int32_t>(0), 16U);
        VBarrier();
        Adds(segOffInt[32U], segOffInt, static_cast<int32_t>(0), 32U);
        WaitAndGatherSmallM8(w, lanes);
    }

    __aicore__ inline void WaitAndGatherSmallM8(LocalTensor<float> w, uint32_t lanes)
    {
        using namespace TreeLineCommon;
        if constexpr (sizeof(T) == sizeof(float)) {
            auto segOff = w[3360U].template ReinterpretCast<uint32_t>();
            auto pointOff = w[3424U].template ReinterpretCast<uint32_t>();
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
            VBarrier();
            GatherSegmentAoS(
                w[1280U], w[1536U], w[1792U], w[2048U], w[2304U], w[2560U],
                w[2816U], segOff, 64U);
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(1));
            GatherPointAoS(w[0U], w[256U], w[512U], w[3200U], pointOff, lanes);
            VBarrier();
        } else {
            SyncStatic<HardEvent::MTE2_V>();
            Cast(w[3200U], w[3280U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, 3U * count_);
            Cast(w[2816U], w[3008U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, 6U * m_);
            VBarrier();
            GatherSegAndPoints(w, lanes);
        }
    }

    __aicore__ inline void PrepareMediumM32(LocalTensor<float> w, uint8_t reps, uint32_t lanes)
    {
        using namespace TreeLineCommon;
        auto segOffInt = w[3360U].template ReinterpretCast<int32_t>();
        auto pointOffInt = w[3424U].template ReinterpretCast<int32_t>();
        auto outOffInt = w[3680U].template ReinterpretCast<int32_t>();
        #pragma unroll
        for (uint32_t i = 0U; i < 8U; ++i) {
            segOffInt.SetValue(i, static_cast<int32_t>(i));
        }
        SyncStatic<HardEvent::S_V>();
        Duplicate(w[3200U], 0.0f, 16U);
        Duplicate(w[2816U], 0.0f, 192U);
        Adds(segOffInt[8U], segOffInt, static_cast<int32_t>(8), 8U);
        Duplicate(pointOffInt, static_cast<int32_t>(0), 32U);
        Duplicate(pointOffInt[32U], static_cast<int32_t>((count_ > 1U) ? 12 : 0), 32U);
        if (reps > 1U) {
            Duplicate(pointOffInt[64U], static_cast<int32_t>((count_ > 2U) ? 24 : 0), 32U);
            Duplicate(pointOffInt[96U], static_cast<int32_t>((count_ > 3U) ? 36 : 0), 32U);
        }
        VBarrier();
        Adds(segOffInt[16U], segOffInt, static_cast<int32_t>(16), 16U);
        VBarrier();
        Muls(outOffInt, segOffInt, static_cast<int32_t>(32), 8U);
        Muls(segOffInt, segOffInt, static_cast<int32_t>(24), 32U);
        VBarrier();
        Adds(segOffInt[32U], segOffInt, static_cast<int32_t>(0), 32U);
        SyncStatic<HardEvent::MTE2_V>();
        if constexpr (sizeof(T) == sizeof(float)) {
            Adds(w[3200U], w[3280U], 0.0f, 3U * count_);
            Adds(w[2816U], w[3008U], 0.0f, 6U * m_);
        } else {
            Cast(w[3200U], w[3280U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, 3U * count_);
            Cast(w[2816U], w[3008U].template ReinterpretCast<T>(), RoundMode::CAST_NONE, 6U * m_);
        }
        VBarrier();
        GatherSegAndPoints(w, lanes);
    }

    __aicore__ inline void ComputePair32Distances(LocalTensor<float> w, uint8_t reps, uint32_t lanes)
    {
        using namespace TreeLineCommon;
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        const BinaryRepeatParams segFirstBinary{1, 1, 1, 8, 0, 8};
        const BinaryRepeatParams segSecondBinary{1, 1, 1, 8, 8, 0};
        const UnaryRepeatParams unary{1, 1, 8, 8};
        ResetMask();
        Sub<float, false>(w[2048U], w[1280U], w[2048U], MASK_PLACEHOLDER, 1, binary);
        Sub<float, false>(w[2304U], w[1536U], w[2304U], MASK_PLACEHOLDER, 1, binary);
        Sub<float, false>(w[2560U], w[1792U], w[2560U], MASK_PLACEHOLDER, 1, binary);
        Sub<float, false>(w[0U], w[1280U], w[0U], MASK_PLACEHOLDER, reps, segFirstBinary);
        Sub<float, false>(w[256U], w[1536U], w[256U], MASK_PLACEHOLDER, reps, segFirstBinary);
        Sub<float, false>(w[512U], w[1792U], w[512U], MASK_PLACEHOLDER, reps, segFirstBinary);
        VBarrier();
        Mul<float, false>(w[1024U], w[2048U], w[2048U], MASK_PLACEHOLDER, 1, binary);
        Mul<float, false>(w[1280U], w[2304U], w[2304U], MASK_PLACEHOLDER, 1, binary);
        Mul<float, false>(w[1536U], w[2560U], w[2560U], MASK_PLACEHOLDER, 1, binary);
        Mul<float, false>(w[768U], w[0U], w[2048U], MASK_PLACEHOLDER, reps, segSecondBinary);
        Mul<float, false>(w[1792U], w[256U], w[2304U], MASK_PLACEHOLDER, reps, segSecondBinary);
        VBarrier();
        Add<float, false>(w[1024U], w[1024U], w[1280U], MASK_PLACEHOLDER, 1, binary);
        Add<float, false>(w[768U], w[768U], w[1792U], MASK_PLACEHOLDER, reps, binary);
        Mul<float, false>(w[1792U], w[512U], w[2560U], MASK_PLACEHOLDER, reps, segSecondBinary);
        VBarrier();
        Add<float, false>(w[1024U], w[1024U], w[1536U], MASK_PLACEHOLDER, 1, binary);
        Add<float, false>(w[768U], w[768U], w[1792U], MASK_PLACEHOLDER, reps, binary);
        VBarrier();
        Maxs<float, false>(w[1024U], w[1024U], 1.0e-30f, MASK_PLACEHOLDER, 1, unary);
        Maxs<float, false>(w[768U], w[768U], 0.0f, MASK_PLACEHOLDER, reps, unary);
        VBarrier();
        FinishPair32Residuals(w, reps, lanes);
    }

    __aicore__ inline void FinishPair32Residuals(
        LocalTensor<float> w, uint8_t reps, uint32_t lanes)
    {
        using namespace TreeLineCommon;
        auto invLen = w[1024U];
        if (reps > 1U) {
            Adds(invLen[64U], invLen, 0.0f, 64U);
            if (reps > 2U) {
                Adds(invLen[128U], invLen, 0.0f, 64U);
            }
            VBarrier();
        }
        Div(w[768U], w[768U], invLen, lanes);
        VBarrier();
        const BinaryRepeatParams binary{1, 1, 1, 8, 8, 8};
        const BinaryRepeatParams segSecondBinary{1, 1, 1, 8, 8, 0};
        const UnaryRepeatParams unary{1, 1, 8, 8};
        ResetMask();
        Mins<float, false>(w[1280U], w[768U], 1.0f, MASK_PLACEHOLDER, reps, unary);
        VBarrier();
        Mul<float, false>(w[1536U], w[1280U], w[2048U], MASK_PLACEHOLDER, reps, segSecondBinary);
        Mul<float, false>(w[1792U], w[1280U], w[2304U], MASK_PLACEHOLDER, reps, segSecondBinary);
        Mul<float, false>(invLen, w[1280U], w[2560U], MASK_PLACEHOLDER, reps, segSecondBinary);
        VBarrier();
        Sub<float, false>(w[0U], w[0U], w[1536U], MASK_PLACEHOLDER, reps, binary);
        Sub<float, false>(w[256U], w[256U], w[1792U], MASK_PLACEHOLDER, reps, binary);
        Sub<float, false>(w[512U], w[512U], invLen, MASK_PLACEHOLDER, reps, binary);
        VBarrier();
        SumSquares3(w[768U], w[0U], w[256U], w[512U], reps);
    }

    __aicore__ inline void ReduceAndStorePair32(LocalTensor<float> w)
    {
        using namespace TreeLineCommon;
        auto outOff = w[3680U].template ReinterpretCast<uint32_t>();
        auto pairs = w[3744U];
        auto outD = w[3936U];
        auto outIdx = w[4000U].template ReinterpretCast<int32_t>();
        auto outT = w[4064U].template ReinterpretCast<T>();
        WholeReduceMin<float>(
            pairs, w[768U], static_cast<int32_t>(m_), static_cast<int32_t>(count_),
            4, 1, (m_ <= 8U) ? 1 : 4, ReduceOrder::ORDER_VALUE_INDEX);
        VBarrier();
        Gather(outIdx, pairs.template ReinterpretCast<int32_t>(), outOff, 4U, count_);
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
        Gather(outD, pairs, outOff, 0U, count_);
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
        DataCopyExtParams dc{1, count_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyExtParams ic{1, count_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(indexGm_[pos_], outIdx, ic);
        VBarrier();
        Sqrt(outD, outD, count_);
        if constexpr (sizeof(T) == sizeof(float)) {
            outT = outD.template ReinterpretCast<T>();
        } else {
            VBarrier();
            Cast(outT, outD, RoundMode::CAST_RINT, count_);
        }
        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(1));
        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(1));
        DataCopyPad(distanceGm_[pos_], outT, dc);
    }

    GlobalTensor<T> pointsGm_, segmentsGm_, distanceGm_;
    GlobalTensor<int32_t> indexGm_;
    uint32_t n_, m_, pos_, count_;
};

#endif  // TREE_LINE_PAIR32_H
