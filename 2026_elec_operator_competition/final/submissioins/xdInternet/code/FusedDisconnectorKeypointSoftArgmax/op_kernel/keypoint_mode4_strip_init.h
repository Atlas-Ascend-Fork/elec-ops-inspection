/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef KEYPOINT_MODE4_STRIP_INIT_H
#define KEYPOINT_MODE4_STRIP_INIT_H

#include "keypoint_common.h"

namespace keypoint_impl {

static constexpr uint32_t kM4InChannels = 32U;
static constexpr uint32_t kM4OutChannels = 4U;
static constexpr uint32_t kM4PlaneWeights = 288U;
static constexpr uint32_t kM4ChanFloats = 4104U;
static constexpr uint32_t kM4ChanBytes = kM4ChanFloats * 4U;

static constexpr uint32_t kM4UCoordOff = 1024U;
static constexpr uint32_t kM4VBaseOff = 2048U;
static constexpr uint32_t kM4BiasOff = 2304U;
static constexpr uint32_t kM4ValidMaskOff = 2560U;
static constexpr uint32_t kM4TailFloorOff = 3584U;
static constexpr uint32_t kM4WeightOff = 4608U;
static constexpr uint32_t kM4Slot0Off = 9216U;
static constexpr uint32_t kM4Slot1Off = 42048U;
static constexpr uint32_t kM4LeftOff = 74944U;
static constexpr uint32_t kM4RightOff = 91392U;
static constexpr uint32_t kM4OffLeftOff = 107776U;
static constexpr uint32_t kM4OffRightOff = 124160U;
static constexpr uint32_t kM4Acc0Off = 140544U;
static constexpr uint32_t kM4Acc1Off = 156992U;
static constexpr uint32_t kM4Acc2Off = 173440U;
static constexpr uint32_t kM4WorkOff = 189824U;
static constexpr uint32_t kM4RowSumAOff = 190848U;
static constexpr uint32_t kM4RowSumBOff = 191104U;
static constexpr uint32_t kM4VStripOff = 191360U;
static constexpr uint32_t kM4ScalarBoxOff = 191616U;
static constexpr uint32_t kM4ExpTinyOff = 191872U;
static constexpr uint32_t kM4AccStatsOff = 191936U;
static constexpr uint32_t kM4LocStatsOff = 192000U;
static constexpr uint32_t kM4OutOff = 192064U;

class Mode4StripBase {
protected:
    KeypointGmTensors gm_;

    static __aicore__ inline uint32_t ComputeMaxInRows(const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        uint32_t maxR = (t.maxStripRows > 0U) ? t.maxStripRows : 1U;
        uint32_t maxIn = t.height;
        if (t.height > 2U * maxR) {
            maxIn = maxR + 2U;
        } else if (t.height > maxR) {
            maxIn = maxR + 1U;
        }
        uint32_t capRows = SafeDivU(4096U, t.wPad);
        if (capRows == 0U) {
            capRows = 1U;
        }
        return (maxIn > capRows) ? capRows : maxIn;
    }

    static __aicore__ inline bool HasPlaneGroupNz(const AscendC::LocalTensor<float>& sBox, uint32_t k)
    {
        for (uint32_t g = 0U; g < 4U; ++g) {
            if (sBox.GetValue(k * 4U + g) != 0.0f) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline uint32_t ClassifyPlanesM4()
    {
        auto wUb = UbTensor<float>(kM4WeightOff, 1152U);
        auto bUb = UbTensor<float>(kM4BiasOff, 8U);
        auto tmpUb = UbTensor<float>(kM4Slot0Off, 2560U);
        auto sBox = UbTensor<float>(kM4ScalarBoxOff, 64U);
        AscendC::DataCopyPad(
            wUb, gm_.weight, AscendC::DataCopyExtParams{1U, 1152U * 4U, 0U, 0U, 0U},
            AscendC::DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
        AscendC::DataCopyPad(
            bUb, gm_.bias, AscendC::DataCopyExtParams{1U, 16U, 0U, 0U, 0U},
            AscendC::DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        ReduceSuperGroups72MaxUb(sBox, tmpUb, wUb, 1152U, 16);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        uint32_t mask = 0U;
        for (uint32_t k = 0U; k < 4U; ++k) {
            if (HasPlaneGroupNz(sBox, k)) {
                mask |= (1U << k);
            }
        }
        return mask;
    }

    static __aicore__ inline void PatchRow0Masks(const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        uint32_t wPad = t.wPad;
        auto offL = UbTensor<uint32_t>(kM4OffLeftOff, wPad);
        auto offR = UbTensor<uint32_t>(kM4OffRightOff, wPad);
        auto maskL32 = UbTensor<uint32_t>(kM4LeftOff, wPad);
        auto maskR32 = UbTensor<uint32_t>(kM4RightOff, wPad);
        offL.SetValue(0U, 0U);
        maskL32.SetValue(0U, 0U);
        uint32_t rightCut = (t.width > 0U) ? (t.width - 1U) : 0U;
        for (uint32_t u = rightCut; u < wPad; ++u) {
            offR.SetValue(u, 0U);
            maskR32.SetValue(u, 0U);
        }
        if ((t.flags & 1U) == 0U) {
            auto validUb = UbTensor<float>(kM4ValidMaskOff, wPad);
            auto floorUb = UbTensor<float>(kM4TailFloorOff, wPad);
            for (uint32_t u = t.width; u < wPad; ++u) {
                offL.SetValue(u, 0U);
                maskL32.SetValue(u, 0U);
                validUb.SetValue(u, 0.0f);
                floorUb.SetValue(u, -3.402823466e+38f);
            }
        }
    }

    static __aicore__ inline void ReplicateRowMasksM4(
        const AscendC::LocalTensor<int32_t>& offLI32,
        const AscendC::LocalTensor<int32_t>& offRI32,
        const AscendC::LocalTensor<int32_t>& maskLI32,
        const AscendC::LocalTensor<int32_t>& maskRI32,
        uint32_t wPad, uint32_t maxInRows, uint32_t totalIn)
    {
        for (uint32_t rStep = 1U; rStep < maxInRows; rStep <<= 1U) {
            uint32_t remR = maxInRows - rStep;
            uint32_t copyR = (remR < rStep) ? remR : rStep;
            uint32_t elemCnt = copyR * wPad;
            uint32_t dstOff = rStep * wPad;
            int32_t byteDelta = static_cast<int32_t>(dstOff * 4U);
            AscendC::Adds(offLI32[dstOff], offLI32, byteDelta, elemCnt);
            AscendC::Adds(offRI32[dstOff], offRI32, byteDelta, elemCnt);
            AscendC::Adds(maskLI32[dstOff], maskLI32, 0, elemCnt);
            AscendC::Adds(maskRI32[dstOff], maskRI32, 0, elemCnt);
            AscendC::PipeBarrier<PIPE_V>();
        }
        uint32_t u16Cnt = totalIn * 2U;
        AscendC::And(
            UbTensor<uint16_t>(kM4OffLeftOff, u16Cnt),
            UbTensor<uint16_t>(kM4OffLeftOff, u16Cnt),
            UbTensor<uint16_t>(kM4LeftOff, u16Cnt), static_cast<int32_t>(u16Cnt));
        AscendC::And(
            UbTensor<uint16_t>(kM4OffRightOff, u16Cnt),
            UbTensor<uint16_t>(kM4OffRightOff, u16Cnt),
            UbTensor<uint16_t>(kM4RightOff, u16Cnt), static_cast<int32_t>(u16Cnt));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ResetMask();
    }

    static __aicore__ inline void InitMasksAndGatherOffsets(
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        uint32_t wPad = t.wPad;
        uint32_t maxInRows = ComputeMaxInRows(t);
        uint32_t totalIn = maxInRows * wPad;
        auto uUb = UbTensor<float>(kM4UCoordOff, wPad);
        auto vBaseUb = UbTensor<float>(kM4VBaseOff, 64U);
        auto offLI32 = UbTensor<int32_t>(kM4OffLeftOff, totalIn);
        auto offRI32 = UbTensor<int32_t>(kM4OffRightOff, totalIn);
        auto maskLI32 = UbTensor<int32_t>(kM4LeftOff, totalIn);
        auto maskRI32 = UbTensor<int32_t>(kM4RightOff, totalIn);

        for (uint32_t u = 0U; u < 8U; ++u) {
            uUb.SetValue(u, ScalarToFloat(u));
            vBaseUb.SetValue(u, ScalarToFloat(u));
            offLI32.SetValue(u, static_cast<int32_t>((7U + u) * 4U));
            offRI32.SetValue(u, static_cast<int32_t>((9U + u) * 4U));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        for (uint32_t step = 8U; step < wPad; step <<= 1U) {
            uint32_t rem = wPad - step;
            uint32_t cnt = (rem < step) ? rem : step;
            int32_t byteStep = static_cast<int32_t>(step * 4U);
            AscendC::Adds(uUb[step], uUb, ScalarToFloat(step), cnt);
            AscendC::Adds(offLI32[step], offLI32, byteStep, cnt);
            AscendC::Adds(offRI32[step], offRI32, byteStep, cnt);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t step = 8U; step < 64U; step <<= 1U) {
            AscendC::Adds(vBaseUb[step], vBaseUb, ScalarToFloat(step), step);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        PatchRow0Masks(t);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
        ReplicateRowMasksM4(offLI32, offRI32, maskLI32, maskRI32, wPad, maxInRows, totalIn);
    }

    __aicore__ inline uint32_t InitBuffersAndMask(
        const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        uint32_t nzMask = ClassifyPlanesM4();
        if (nzMask == 0U) {
            return 0U;
        }
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            uint32_t slotBase = (slot == 0U) ? kM4Slot0Off : kM4Slot1Off;
            AscendC::Duplicate(UbTensor<float>(slotBase, 8U), 0.0f, 8U);
            AscendC::Duplicate(UbTensor<float>(slotBase + kM4ChanBytes, 8U), 0.0f, 8U);
        }
        AscendC::Duplicate(UbTensor<int32_t>(kM4LeftOff, t.wPad), -1, t.wPad);
        AscendC::Duplicate(UbTensor<int32_t>(kM4RightOff, t.wPad), -1, t.wPad);
        AscendC::Duplicate(UbTensor<float>(kM4RowSumAOff, 64U), 0.0f, 64U);
        AscendC::Duplicate(UbTensor<float>(kM4RowSumBOff, 64U), 0.0f, 64U);
        AscendC::Duplicate(UbTensor<float>(kM4WorkOff, 256U), 0.0f, 256U);
        if ((t.flags & 1U) == 0U) {
            AscendC::Duplicate(UbTensor<float>(kM4ValidMaskOff, t.wPad), 1.0f, t.wPad);
            AscendC::Duplicate(UbTensor<float>(kM4TailFloorOff, t.wPad), 0.0f, t.wPad);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        InitMasksAndGatherOffsets(t);
        return nzMask;
    }

    __aicore__ inline void LoadChannelPair(
        uint64_t batchBase, uint32_t pairIdx, uint32_t inStartRow, uint32_t inRows,
        uint32_t slotIdx, const FusedDisconnectorKeypointSoftArgmaxTilingData& t)
    {
        uint32_t slotBase = (slotIdx == 0U) ? kM4Slot0Off : kM4Slot1Off;
        uint32_t elems = inRows * t.wPad;
        bool alignedW = ((t.flags & 1U) != 0U) && gm_.featAligned;
        uint16_t copyBlks = static_cast<uint16_t>(elems >> 3U);

        if (alignedW) {
            uint64_t planeElems = static_cast<uint64_t>(t.height) * t.width;
            uint64_t srcGapBlks = (planeElems - elems) >> 3U;
            uint32_t dstGapBlks = (kM4ChanFloats - elems) >> 3U;
            uint64_t srcOff0 = batchBase + (static_cast<uint64_t>(pairIdx * 2U) * t.height + inStartRow) * t.width;
            if (srcGapBlks <= 65535ULL) {
                auto dst0 = UbTensor<float>(slotBase + 32U, kM4ChanFloats + elems);
                AscendC::DataCopy(
                    dst0, gm_.feat[srcOff0],
                    AscendC::DataCopyParams{
                        2U, copyBlks, static_cast<uint16_t>(srcGapBlks), static_cast<uint16_t>(dstGapBlks)});
                return;
            }
            for (uint32_t cc = 0U; cc < 2U; ++cc) {
                auto dst = UbTensor<float>(slotBase + cc * kM4ChanBytes + 32U, elems);
                AscendC::DataCopy(
                    dst, gm_.feat[srcOff0 + cc * planeElems], AscendC::DataCopyParams{1U, copyBlks, 0U, 0U});
            }
            return;
        }

        uint8_t rightPad = static_cast<uint8_t>(t.wPad - t.width);
        bool needPad = (rightPad > 0U);
        for (uint32_t cc = 0U; cc < 2U; ++cc) {
            uint32_t c = pairIdx * 2U + cc;
            auto dst = UbTensor<float>(slotBase + cc * kM4ChanBytes + 32U, elems);
            uint64_t srcOff = batchBase + (static_cast<uint64_t>(c) * t.height + inStartRow) * t.width;
            AscendC::DataCopyPad(
                dst, gm_.feat[srcOff],
                AscendC::DataCopyExtParams{static_cast<uint16_t>(inRows), t.width * 4U, 0U, 0U, 0U},
                AscendC::DataCopyPadExtParams<float>{needPad, 0U, rightPad, 0.0f});
        }
    }
};

}

#endif
