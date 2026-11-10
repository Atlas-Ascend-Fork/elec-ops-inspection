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

#ifndef TREE_LINE_SMALL_MODES_H
#define TREE_LINE_SMALL_MODES_H

#include "tree_line_common.h"

template <typename T>
class KernelTreeLineSmallCompileShape {
public:
    static constexpr uint32_t kSegOffset = 16U;

    template <HardEvent E>
    __aicore__ inline static void Sync()
    {
        constexpr event_t event = static_cast<event_t>(0);
        SetFlag<E>(event);
        WaitFlag<E>(event);
    }

    __aicore__ inline static float DistanceSquared(
        float px, float py, float pz,
        float ax, float ay, float az,
        float dx, float dy, float dz, float len)
    {
        const float apx = px - ax;
        const float apy = py - ay;
        const float apz = pz - az;
        const float num = (apx * dx + apy * dy) + apz * dz;
        float t = 0.0f;
        if (num > 0.0f && len > 0.0f) {
            t = (num >= len) ? 1.0f : (num / len);
        }
        const float rx = apx - t * dx;
        const float ry = apy - t * dy;
        const float rz = apz - t * dz;
        return (rx * rx + ry * ry) + rz * rz;
    }

    __aicore__ inline static float DistanceSquaredLazyLen(
        float px, float py, float pz,
        float ax, float ay, float az,
        float dx, float dy, float dz)
    {
        const float len = (dx * dx + dy * dy) + dz * dz;
        return DistanceSquared(px, py, pz, ax, ay, az, dx, dy, dz, len);
    }

    __aicore__ inline static float FastSqrtPositive(float x)
    {
        union FloatBits {
            float f;
            uint32_t u;
        } seed{x};
        seed.u = 0x5f375a86U - (seed.u >> 1U);
        float y = seed.f;
        const float halfX = 0.5f * x;
        y = y * (1.5f - (halfX * y) * y);
        y = y * (1.5f - (halfX * y) * y);
        return x * y;
    }

    __aicore__ inline static float FinishDistance(float best)
    {
        if (best <= 0.0f) {
            return 0.0f;
        }
        if (__builtin_expect(best < 1.17549435e-38f, 0)) {
            return FastSqrtPositive(best * 16777216.0f) * 0.000244140625f;
        }
        return FastSqrtPositive(best);
    }

    template <uint32_t COUNT, uint32_t M_SEG, bool MULTI_CORE = false>
    __aicore__ inline void ProcessFixed(
        GM_ADDR points, GM_ADDR segments, GM_ADDR distances, GM_ADDR indices)
    {
        const uint32_t blkIdx = GetBlockIdx();
        if constexpr (sizeof(T) == 4U && !MULTI_CORE) {
            if (blkIdx != 0U) {
                return;
            }
        }
        constexpr bool kSkipWarmupDma = (!MULTI_CORE) && (M_SEG >= 2U);
        constexpr bool kUseScalarGmM1 = (!MULTI_CORE) && (M_SEG == 1U) && (sizeof(T) == 4U);
        const bool isActiveCore = MULTI_CORE || (blkIdx == 0U);
        const uint32_t ptStart = MULTI_CORE ? blkIdx : 0U;
        const uint32_t totalPts = MULTI_CORE ? 19U : COUNT;

        LocalTensor<float> base(TPosition::VECCALC, 0U, 256U);
        auto in = base.template ReinterpretCast<T>();
        CopyFixedInputs<COUNT, M_SEG, kSkipWarmupDma, kUseScalarGmM1>(
            in, points, segments, ptStart, totalPts, isActiveCore);
        LocalTensor<float> values = PrepareFixedValues<M_SEG, kSkipWarmupDma, kUseScalarGmM1>(
            base, in, isActiveCore);

        float px[3] = {0.0f, 0.0f, 0.0f};
        float py[3] = {0.0f, 0.0f, 0.0f};
        float pz[3] = {0.0f, 0.0f, 0.0f};
        float best[3] = {3.402823466e+38F, 3.402823466e+38F, 3.402823466e+38F};
        int32_t idx[3] = {0, 0, 0};
        LoadFixedPoints<COUNT, kUseScalarGmM1>(points, ptStart, values, px, py, pz);
        if constexpr (M_SEG == 1U) {
            EvalFixedSingleSeg<COUNT, kUseScalarGmM1>(segments, values, px, py, pz, best);
        } else {
            EvalFixedMultiSeg<COUNT, M_SEG>(values, px, py, pz, best, idx);
        }
        StoreFixedOutputs<COUNT>(
            base, distances, indices, ptStart, totalPts, isActiveCore, best, idx);
    }

private:
    template <uint32_t COUNT, uint32_t M_SEG>
    __aicore__ inline static void DoFixedDmaCopy(
        LocalTensor<T> in, GM_ADDR points, GM_ADDR segments, uint32_t ptStart, uint32_t totalPts)
    {
        GlobalTensor<T> pointsGm, segmentsGm;
        pointsGm.SetGlobalBuffer((__gm__ T*)points, 3U * totalPts);
        segmentsGm.SetGlobalBuffer((__gm__ T*)segments, 6U * M_SEG);
        DataCopyExtParams pc{1, static_cast<uint32_t>(3U * COUNT * sizeof(T)), 0, 0, 0};
        DataCopyExtParams sc{1, static_cast<uint32_t>(6U * M_SEG * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(in, pointsGm[3U * ptStart], pc, pad);
        DataCopyPad(in[kSegOffset], segmentsGm, sc, pad);
    }

    template <uint32_t COUNT, uint32_t M_SEG, bool kSkipWarmupDma, bool kUseScalarGmM1>
    __aicore__ inline static void CopyFixedInputs(
        LocalTensor<T> in, GM_ADDR points, GM_ADDR segments,
        uint32_t ptStart, uint32_t totalPts, bool isActiveCore)
    {
        if constexpr (kSkipWarmupDma) {
            if (isActiveCore) {
                DoFixedDmaCopy<COUNT, M_SEG>(in, points, segments, ptStart, totalPts);
            }
        } else if constexpr (!kUseScalarGmM1) {
            DoFixedDmaCopy<COUNT, M_SEG>(in, points, segments, ptStart, totalPts);
        } else {
#ifndef __DAV_C220_VEC__
            DoFixedDmaCopy<COUNT, M_SEG>(in, points, segments, ptStart, totalPts);
#else
            (void)in;
            (void)points;
            (void)segments;
            (void)ptStart;
            (void)totalPts;
#endif
        }
        (void)isActiveCore;
    }

    template <uint32_t M_SEG, bool kSkipWarmupDma, bool kUseScalarGmM1>
    __aicore__ inline static LocalTensor<float> PrepareFixedValues(
        LocalTensor<float> base, LocalTensor<T> in, bool isActiveCore)
    {
        if constexpr (sizeof(T) == 2U) {
            if (!kSkipWarmupDma || isActiveCore) {
                Sync<HardEvent::MTE2_V>();
            }
            LocalTensor<float> values = base[96U];
            Cast(values, in, RoundMode::CAST_NONE, kSegOffset + 6U * M_SEG);
            Sync<HardEvent::V_S>();
            return values;
        }
        if constexpr (kSkipWarmupDma) {
            if (isActiveCore) {
                Sync<HardEvent::MTE2_S>();
            }
        } else if constexpr (!kUseScalarGmM1) {
            Sync<HardEvent::MTE2_S>();
        } else {
#ifndef __DAV_C220_VEC__
            Sync<HardEvent::MTE2_S>();
#endif
        }
        (void)in;
        return base;
    }

    template <uint32_t COUNT, bool kUseScalarGmM1>
    __aicore__ inline static void LoadFixedPoints(
        GM_ADDR points, uint32_t ptStart, LocalTensor<float> values,
        float px[3], float py[3], float pz[3])
    {
#ifdef __DAV_C220_VEC__
        __gm__ const float* pGmPtr = reinterpret_cast<__gm__ const float*>(points) + 3U * ptStart;
        #pragma unroll
        for (uint32_t i = 0U; i < COUNT; ++i) {
            px[i] = kUseScalarGmM1 ? pGmPtr[3U * i] : values.GetValue(3U * i);
            py[i] = kUseScalarGmM1 ? pGmPtr[3U * i + 1U] : values.GetValue(3U * i + 1U);
            pz[i] = kUseScalarGmM1 ? pGmPtr[3U * i + 2U] : values.GetValue(3U * i + 2U);
        }
#else
        (void)points;
        (void)ptStart;
        #pragma unroll
        for (uint32_t i = 0U; i < COUNT; ++i) {
            px[i] = values.GetValue(3U * i);
            py[i] = values.GetValue(3U * i + 1U);
            pz[i] = values.GetValue(3U * i + 2U);
        }
#endif
    }

    template <uint32_t COUNT, bool kUseScalarGmM1>
    __aicore__ inline static void EvalFixedSingleSeg(
        GM_ADDR segments, LocalTensor<float> values,
        const float px[3], const float py[3], const float pz[3], float best[3])
    {
#ifdef __DAV_C220_VEC__
        __gm__ const float* sGmPtr = reinterpret_cast<__gm__ const float*>(segments);
        const float ax = kUseScalarGmM1 ? sGmPtr[0U] : values.GetValue(kSegOffset);
        const float ay = kUseScalarGmM1 ? sGmPtr[1U] : values.GetValue(kSegOffset + 1U);
        const float az = kUseScalarGmM1 ? sGmPtr[2U] : values.GetValue(kSegOffset + 2U);
        const float dx = (kUseScalarGmM1 ? sGmPtr[3U] : values.GetValue(kSegOffset + 3U)) - ax;
        const float dy = (kUseScalarGmM1 ? sGmPtr[4U] : values.GetValue(kSegOffset + 4U)) - ay;
        const float dz = (kUseScalarGmM1 ? sGmPtr[5U] : values.GetValue(kSegOffset + 5U)) - az;
#else
        (void)segments;
        float ax = 0.0f, ay = 0.0f, az = 0.0f, dx = 0.0f, dy = 0.0f, dz = 0.0f;
        TreeLineCommon::ReadSegment6(values, kSegOffset, ax, ay, az, dx, dy, dz);
#endif
        if constexpr (COUNT == 1U) {
            best[0] = DistanceSquaredLazyLen(px[0], py[0], pz[0], ax, ay, az, dx, dy, dz);
        } else {
            const float len = (dx * dx + dy * dy) + dz * dz;
            #pragma unroll
            for (uint32_t i = 0U; i < COUNT; ++i) {
                best[i] = DistanceSquared(px[i], py[i], pz[i], ax, ay, az, dx, dy, dz, len);
            }
        }
    }

    template <uint32_t COUNT>
    __aicore__ inline static void UpdateFixedCandidates(
        const float px[3], const float py[3], const float pz[3],
        float ax, float ay, float az, float dx, float dy, float dz,
        uint32_t j, float best[3], int32_t idx[3])
    {
        if constexpr (COUNT == 1U) {
            const float d0 = DistanceSquaredLazyLen(px[0], py[0], pz[0], ax, ay, az, dx, dy, dz);
            if (d0 < best[0]) {
                best[0] = d0;
                idx[0] = static_cast<int32_t>(j);
            }
        } else {
            const float len = (dx * dx + dy * dy) + dz * dz;
            #pragma unroll
            for (uint32_t i = 0U; i < COUNT; ++i) {
                const float d = DistanceSquared(px[i], py[i], pz[i], ax, ay, az, dx, dy, dz, len);
                if (d < best[i]) {
                    best[i] = d;
                    idx[i] = static_cast<int32_t>(j);
                }
            }
        }
    }

    template <uint32_t COUNT, uint32_t M_SEG>
    __aicore__ inline static void EvalFixedMultiSeg(
        LocalTensor<float> values,
        const float px[3], const float py[3], const float pz[3],
        float best[3], int32_t idx[3])
    {
        #pragma unroll
        for (uint32_t j = 0U; j < M_SEG; ++j) {
            float ax = 0.0f, ay = 0.0f, az = 0.0f, dx = 0.0f, dy = 0.0f, dz = 0.0f;
            TreeLineCommon::ReadSegment6(values, kSegOffset + 6U * j, ax, ay, az, dx, dy, dz);
            UpdateFixedCandidates<COUNT>(px, py, pz, ax, ay, az, dx, dy, dz, j, best, idx);
        }
    }

    template <uint32_t COUNT>
    __aicore__ inline static void StoreFixedOutputs(
        LocalTensor<float> base, GM_ADDR distances, GM_ADDR indices,
        uint32_t ptStart, uint32_t totalPts, bool isActiveCore,
        const float best[3], const int32_t idx[3])
    {
        GlobalTensor<T> distancesGm;
        GlobalTensor<int32_t> indicesGm;
        distancesGm.SetGlobalBuffer((__gm__ T*)distances, totalPts);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices, totalPts);
        auto out = base[64U];
        auto outIdx = base[72U].ReinterpretCast<int32_t>();
        constexpr bool useScalarSqrt = (COUNT == 1U) || (sizeof(T) == 4U);
        #pragma unroll
        for (uint32_t i = 0U; i < COUNT; ++i) {
            out.SetValue(i, useScalarSqrt ? FinishDistance(best[i]) : best[i]);
            outIdx.SetValue(i, idx[i]);
        }
        WriteOutFixedTensors<COUNT, useScalarSqrt>(
            base, out, outIdx, distancesGm, indicesGm, ptStart, isActiveCore);
    }

    template <uint32_t COUNT, bool useScalarSqrt>
    __aicore__ inline static void WriteOutFixedTensors(
        LocalTensor<float> base, LocalTensor<float> out, LocalTensor<int32_t> outIdx,
        const GlobalTensor<T>& distancesGm, const GlobalTensor<int32_t>& indicesGm,
        uint32_t ptStart, bool isActiveCore)
    {
        DataCopyExtParams dc{1, static_cast<uint32_t>(COUNT * sizeof(T)), 0, 0, 0};
        DataCopyExtParams ic{1, static_cast<uint32_t>(COUNT * sizeof(int32_t)), 0, 0, 0};
        if constexpr (useScalarSqrt) {
            LocalTensor<T> finalOut;
            if constexpr (sizeof(T) == 2U) {
                Sync<HardEvent::S_V>();
                finalOut = base[80U].template ReinterpretCast<T>();
                Cast(finalOut, out, RoundMode::CAST_RINT, COUNT);
                if (!isActiveCore) {
                    return;
                }
                Sync<HardEvent::V_MTE3>();
            } else {
                finalOut = out.template ReinterpretCast<T>();
                if (!isActiveCore) {
                    return;
                }
                Sync<HardEvent::S_MTE3>();
            }
            DataCopyPad(distancesGm[ptStart], finalOut, dc);
            DataCopyPad(indicesGm[ptStart], outIdx, ic);
        } else {
            Sync<HardEvent::S_V>();
            Sqrt(out, out, COUNT);
            PipeBarrier<PIPE_V>();
            LocalTensor<T> finalOut = base[80U].template ReinterpretCast<T>();
            Cast(finalOut, out, RoundMode::CAST_RINT, COUNT);
            if (!isActiveCore) {
                return;
            }
            SetFlag<HardEvent::S_MTE3>(static_cast<event_t>(0));
            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
            WaitFlag<HardEvent::S_MTE3>(static_cast<event_t>(0));
            DataCopyPad(indicesGm[ptStart], outIdx, ic);
            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
            DataCopyPad(distancesGm[ptStart], finalOut, dc);
        }
    }
};

template <typename T>
class KernelTreeLineSmall {
public:
    __aicore__ inline void Init(
        GM_ADDR points, GM_ADDR segments, GM_ADDR distances,
        GM_ADDR indices, const TreeLineMinDistanceTilingData& tiling)
    {
        using namespace TreeLineCommon;
        n_ = tiling.points;
        m_ = tiling.segments;
        if (tiling.blockDim == tiling.batch) {
            b_ = GetBlockIdx();
            ptStart_ = b_ * n_;
            count_ = n_;
        } else {
            ptStart_ = GetBlockIdx();
            b_ = (tiling.batch == 1U) ? 0U : SafeDivU(ptStart_, n_);
            count_ = 1U;
        }
        points_.SetGlobalBuffer((__gm__ T*)points, tiling.batch * n_ * 3U);
        segments_.SetGlobalBuffer((__gm__ T*)segments, tiling.batch * m_ * 6U);
        distances_.SetGlobalBuffer((__gm__ T*)distances, tiling.batch * n_);
        indices_.SetGlobalBuffer((__gm__ int32_t*)indices, tiling.batch * n_);
    }

    __aicore__ inline void Process()
    {
        using namespace TreeLineCommon;
        constexpr uint32_t kSegOffset = 16U;
        LocalTensor<float> base(TPosition::VECCALC, 0U, 256U);
        auto in = base.template ReinterpretCast<T>();
        auto out = base[64U];
        auto outIdx = base[72U].ReinterpretCast<int32_t>();
        DataCopyExtParams pc{1, static_cast<uint32_t>(3U * count_ * sizeof(T)), 0, 0, 0};
        DataCopyExtParams sc{1, static_cast<uint32_t>(6U * m_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(in, points_[ptStart_ * 3U], pc, pad);
        DataCopyPad(in[kSegOffset], segments_[b_ * m_ * 6U], sc, pad);
        LocalTensor<float> values = PrepareValues(base, in, kSegOffset);
        for (uint32_t i = 0U; i < count_; ++i) {
            EvalSinglePointSmall(values, out, outIdx, i, kSegOffset);
        }
        StoreSmallResults(base, out, outIdx);
    }

private:
    __aicore__ inline LocalTensor<float> PrepareValues(
        LocalTensor<float> base, LocalTensor<T> in, uint32_t kSegOffset)
    {
        using namespace TreeLineCommon;
        if constexpr (sizeof(T) == 2U) {
            SyncStatic<HardEvent::MTE2_V>();
            LocalTensor<float> values = base[96U];
            Cast(values, in, RoundMode::CAST_NONE, kSegOffset + 6U * m_);
            PipeBarrier<PIPE_V>();
            SyncStatic<HardEvent::V_S>();
            return values;
        }
        SyncStatic<HardEvent::MTE2_S>();
        (void)in;
        (void)kSegOffset;
        return base;
    }

    __aicore__ inline void EvalSinglePointSmall(
        LocalTensor<float> values, LocalTensor<float> out,
        LocalTensor<int32_t> outIdx, uint32_t i, uint32_t kSegOffset)
    {
        const float px = values.GetValue(3U * i);
        const float py = values.GetValue(3U * i + 1U);
        const float pz = values.GetValue(3U * i + 2U);
        float best = 3.402823466e+38F;
        int32_t bestIdx = 0;
        for (uint32_t j = 0U; j < m_; ++j) {
            float ax = 0.0f, ay = 0.0f, az = 0.0f, dx = 0.0f, dy = 0.0f, dz = 0.0f;
            TreeLineCommon::ReadSegment6(values, kSegOffset + 6U * j, ax, ay, az, dx, dy, dz);
            const float d2 = KernelTreeLineSmallCompileShape<T>::DistanceSquaredLazyLen(
                px, py, pz, ax, ay, az, dx, dy, dz);
            if (d2 < best) {
                best = d2;
                bestIdx = static_cast<int32_t>(j);
            }
        }
        out.SetValue(i, best);
        outIdx.SetValue(i, bestIdx);
    }

    __aicore__ inline void StoreSmallResults(
        LocalTensor<float> base, LocalTensor<float> out, LocalTensor<int32_t> outIdx)
    {
        using namespace TreeLineCommon;
        SyncStatic<HardEvent::S_V>();
        Sqrt(out, out, count_);
        LocalTensor<T> finalOut;
        if constexpr (sizeof(T) == 2U) {
            PipeBarrier<PIPE_V>();
            finalOut = base[80U].template ReinterpretCast<T>();
            Cast(finalOut, out, RoundMode::CAST_RINT, count_);
        } else {
            finalOut = out.template ReinterpretCast<T>();
        }
        SyncStatic<HardEvent::V_MTE3>();
        DataCopyExtParams dc{1, static_cast<uint32_t>(count_ * sizeof(T)), 0, 0, 0};
        DataCopyExtParams ic{1, static_cast<uint32_t>(count_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(distances_[ptStart_], finalOut, dc);
        DataCopyPad(indices_[ptStart_], outIdx, ic);
    }

    GlobalTensor<T> points_, segments_, distances_;
    GlobalTensor<int32_t> indices_;
    uint32_t n_, m_, b_, ptStart_, count_;
};

#endif  // TREE_LINE_SMALL_MODES_H
