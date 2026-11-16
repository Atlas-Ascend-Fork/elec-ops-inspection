/*
 * Copyright (c) 2026 gcw_OTnEiQgj.
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

#pragma once

#include "kpt_p0.h"

namespace elec::impl {

KPT_INL void PackVec64(
    float totE, float totU, float totV, float v0Float, uint32_t &outLo, uint32_t &outHi) {
    const float invE = RecipPos(totE);
    const float limE = (totE < 1.0f) ? 1.0f : ((totE > 8191.0f) ? 8191.0f : totE);
    float encX = totU * invE + 1.0f;
    encX = (encX < 1.0f) ? 1.0f : ((encX > 127.0f) ? 127.0f : encX);
    float encY = (totV * invE - v0Float) + 1.0f;
    encY = (encY < 1.0f) ? 1.0f : ((encY > 127.0f) ? 127.0f : encY);

    const uint32_t be = BitsU32(limE) + 0x10U;
    const uint32_t bu = BitsU32(encX) + 0x10U;
    const uint32_t bv = BitsU32(encY) + 0x10U;

    const uint32_t bits0 = ((be >> 5U) - (127U << 18U)) & 0x3FFFFFU;
    const uint32_t bits1 = ((bu >> 5U) - (127U << 18U)) & 0x1FFFFFU;
    const uint32_t bits2 = ((bv >> 5U) - (127U << 18U)) & 0x1FFFFFU;

    const uint64_t pkt = static_cast<uint64_t>(bits0) |
                         (static_cast<uint64_t>(bits1) << 22U) |
                         (static_cast<uint64_t>(bits2) << 43U);
    outLo = static_cast<uint32_t>(pkt);
    outHi = static_cast<uint32_t>(pkt >> 32U);
}

KPT_INL void UnpackVec64(
    uint32_t lo, uint32_t hi, float v0Float, float &outExp, float &outPU, float &outPV) {
    const uint64_t pkt = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32U);
    const uint32_t bits0 = static_cast<uint32_t>(pkt) & 0x3FFFFFU;
    const uint32_t bits1 = static_cast<uint32_t>(pkt >> 22U) & 0x1FFFFFU;
    const uint32_t bits2 = static_cast<uint32_t>(pkt >> 43U) & 0x1FFFFFU;

    const float e = BitsF32((bits0 + (127U << 18U)) << 5U);
    const float normX = BitsF32((bits1 + (127U << 18U)) << 5U) - 1.0f;
    const float normY = (BitsF32((bits2 + (127U << 18U)) << 5U) - 1.0f) + v0Float;

    outExp = e;
    outPU = e * normX;
    outPV = e * normY;
}

KPT_INL void StoreTaskOut(
    const AscendC::GlobalTensor<float> &gmKeypoints,
    const AscendC::GlobalTensor<float> &gmScores,
    const AscendC::LocalTensor<float> &ubOutKp,
    const AscendC::LocalTensor<float> &ubOutScore,
    uint32_t bk, float totE, float totU, float totV,
    float maxVal, float stride, float biasVal) {
    const AscendC::DataCopyExtParams scalar1Params{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyExtParams vec2Params{1U, 2U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const float invSumExp = RecipPos(totE);
    ubOutKp.SetValue(0U, (totU * invSumExp + 0.5f) * stride);
    ubOutKp.SetValue(1U, (totV * invSumExp + 0.5f) * stride);
    ubOutScore.SetValue(0U, maxVal + biasVal);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyPad(gmKeypoints[bk * 2U], ubOutKp, vec2Params);
    AscendC::DataCopyPad(gmScores[bk], ubOutScore, scalar1Params);
}

KPT_INL void StoreSlotInit(
    const AscendC::GlobalTensor<uint32_t> &gmKeypointsU32,
    const AscendC::GlobalTensor<float> &gmScores,
    const AscendC::LocalTensor<uint32_t> &ubOutKpU32,
    const AscendC::LocalTensor<float> &ubOutScore,
    uint32_t bk, float v0_1,
    float valM, float valE, float valU, float valV) {
    const AscendC::DataCopyExtParams scalar1Params{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyExtParams vec2Params{1U, 2U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    uint32_t lo1 = 0U;
    uint32_t hi1 = 0U;
    PackVec64(valE, valU, valV, v0_1, lo1, hi1);
    ubOutKpU32.SetValue(0U, lo1);
    ubOutKpU32.SetValue(1U, hi1);
    ubOutScore.SetValue(0U, valM);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyPad(gmKeypointsU32[bk * 2U], ubOutKpU32, vec2Params);
    AscendC::DataCopyPad(gmScores[bk], ubOutScore, scalar1Params);
}

KPT_INL void JoinStatPair(
    float mA, float eA, float puA, float pvA,
    float mB, float eB, float puB, float pvB,
    float f0, float &outMax, float &outExp, float &outPU, float &outPV) {
    outMax = (mA >= mB) ? mA : mB;
    float a0 = 1.0f;
    float a1 = 1.0f;
    if (mA != mB) {
        const float z = ((mA < outMax) ? (mA - outMax) : (mB - outMax)) * f0;
        const float ev = EvalPolyExp(z);
        if (mA < outMax) {
            a0 = ev;
        } else {
            a1 = ev;
        }
    }
    outExp = eA * a0 + eB * a1;
    outPU = puA * a0 + puB * a1;
    outPV = pvA * a0 + pvB * a1;
}

KPT_INL void StepSlotJoin(
    const AscendC::GlobalTensor<uint32_t> &gmKeypointsU32,
    const AscendC::GlobalTensor<float> &gmScores,
    const AscendC::LocalTensor<uint32_t> &ubOutKpU32,
    const AscendC::LocalTensor<float> &ubOutScore,
    uint32_t bk, float v0_1,
    float valM, float valE, float valU, float valV, float f0) {
    const AscendC::DataCopyExtParams scalar1Params{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyExtParams vec2Params{1U, 2U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> padZeroF{false, 0U, 0U, 0.0f};
    const AscendC::DataCopyPadExtParams<uint32_t> padZeroU{false, 0U, 0U, 0U};
    AscendC::DataCopyPad(ubOutKpU32, gmKeypointsU32[bk * 2U], vec2Params, padZeroU);
    AscendC::DataCopyPad(ubOutScore, gmScores[bk], scalar1Params, padZeroF);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    const float accM = ubOutScore.GetValue(0U);
    float accE = 0.0f, accU = 0.0f, accV = 0.0f;
    UnpackVec64(ubOutKpU32.GetValue(0U), ubOutKpU32.GetValue(1U), v0_1, accE, accU, accV);

    float curMax = 0.0f, joinE = 0.0f, joinU = 0.0f, joinV = 0.0f;
    JoinStatPair(
        accM, accE, accU, accV,
        valM, valE, valU, valV,
        f0, curMax, joinE, joinU, joinV);
    StoreSlotInit(
        gmKeypointsU32, gmScores, ubOutKpU32, ubOutScore, bk, v0_1, curMax, joinE, joinU, joinV);
}

KPT_INL void FinishSlot0(
    const AscendC::GlobalTensor<float> &gmKeypoints,
    const AscendC::GlobalTensor<uint32_t> &gmKeypointsU32,
    const AscendC::GlobalTensor<float> &gmScores,
    const AscendC::LocalTensor<float> &ubOutKp,
    const AscendC::LocalTensor<uint32_t> &ubOutKpU32,
    const AscendC::LocalTensor<float> &ubOutScore,
    uint32_t bk, float v0_1,
    float valM, float valE, float valU, float valV,
    float f0, float stride, float biasVal) {
    const AscendC::DataCopyExtParams scalar1Params{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyExtParams vec2Params{1U, 2U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> padZeroF{false, 0U, 0U, 0.0f};
    const AscendC::DataCopyPadExtParams<uint32_t> padZeroU{false, 0U, 0U, 0U};
    AscendC::DataCopyPad(ubOutKpU32, gmKeypointsU32[bk * 2U], vec2Params, padZeroU);
    AscendC::DataCopyPad(ubOutScore, gmScores[bk], scalar1Params, padZeroF);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

    const float otherM = ubOutScore.GetValue(0U);
    float otherE = 0.0f, otherU = 0.0f, otherV = 0.0f;
    UnpackVec64(ubOutKpU32.GetValue(0U), ubOutKpU32.GetValue(1U), v0_1, otherE, otherU, otherV);

    float gMax = 0.0f, totE = 0.0f, totU = 0.0f, totV = 0.0f;
    JoinStatPair(
        valM, valE, valU, valV,
        otherM, otherE, otherU, otherV,
        f0, gMax, totE, totU, totV);
    StoreTaskOut(gmKeypoints, gmScores, ubOutKp, ubOutScore, bk, totE, totU, totV, gMax, stride, biasVal);
}

template <uint32_t S_CST, bool OPT_R>
KPT_INL void InitSyncGates(uint32_t s) {
    if constexpr (S_CST == 2U) {
        if (s == 0U) {
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(8U);
        }
    } else if constexpr (S_CST == 3U) {
        constexpr uint32_t sFirst = OPT_R ? 2U : 1U;
        if (s != sFirst) {
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(8U);
        }
        if (s == 0U || s == sFirst) {
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(9U);
        }
    } else if constexpr (S_CST == 4U) {
        constexpr uint32_t sFirst = OPT_R ? 3U : 1U;
        constexpr uint32_t sThird = OPT_R ? 1U : 3U;
        if (s != sFirst) {
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(8U);
        }
        if (s != 2U) {
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(9U);
        }
        if (s != sThird) {
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(10U);
        }
    }
}

template <bool OPT_R>
KPT_INL void JoinFixed4(
    const AscendC::GlobalTensor<float> &gmKeypoints,
    const AscendC::GlobalTensor<uint32_t> &gmKeypointsU32,
    const AscendC::GlobalTensor<float> &gmScores,
    const AscendC::LocalTensor<float> &ubOutKp,
    const AscendC::LocalTensor<uint32_t> &ubOutKpU32,
    const AscendC::LocalTensor<float> &ubOutScore,
    uint32_t s, uint32_t bk, float v0_1,
    float valM, float valE, float valU, float valV,
    float f0, float stride, float biasVal) {
    constexpr uint32_t sFirst = OPT_R ? 3U : 1U;
    constexpr uint32_t sThird = OPT_R ? 1U : 3U;
    if (s == sFirst) {
        StoreSlotInit(gmKeypointsU32, gmScores, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV);
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(8U);
        AscendC::CrossCoreWaitFlag(8U);
        AscendC::CrossCoreWaitFlag(9U);
        AscendC::CrossCoreWaitFlag(10U);
    } else if (s == 2U) {
        AscendC::CrossCoreWaitFlag(8U);
        StepSlotJoin(gmKeypointsU32, gmScores, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV, f0);
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(9U);
        AscendC::CrossCoreWaitFlag(9U);
        AscendC::CrossCoreWaitFlag(10U);
    } else if (s == sThird) {
        AscendC::CrossCoreWaitFlag(8U);
        AscendC::CrossCoreWaitFlag(9U);
        StepSlotJoin(gmKeypointsU32, gmScores, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV, f0);
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(10U);
        AscendC::CrossCoreWaitFlag(10U);
    } else {
        AscendC::CrossCoreWaitFlag(8U);
        AscendC::CrossCoreWaitFlag(9U);
        AscendC::CrossCoreWaitFlag(10U);
        FinishSlot0(gmKeypoints, gmKeypointsU32, gmScores, ubOutKp, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV, f0, stride, biasVal);
    }
}

template <uint32_t S_CST, bool OPT_R>
KPT_INL void JoinFixedN(
    const AscendC::GlobalTensor<float> &gmKeypoints,
    const AscendC::GlobalTensor<uint32_t> &gmKeypointsU32,
    const AscendC::GlobalTensor<float> &gmScores,
    const AscendC::LocalTensor<float> &ubOutKp,
    const AscendC::LocalTensor<uint32_t> &ubOutKpU32,
    const AscendC::LocalTensor<float> &ubOutScore,
    uint32_t s, uint32_t bk, float v0_1,
    float valM, float valE, float valU, float valV,
    float f0, float stride, float biasVal) {
    if constexpr (S_CST == 2U) {
        if (s == 1U) {
            StoreSlotInit(gmKeypointsU32, gmScores, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV);
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(8U);
            AscendC::CrossCoreWaitFlag(8U);
        } else {
            AscendC::CrossCoreWaitFlag(8U);
            FinishSlot0(gmKeypoints, gmKeypointsU32, gmScores, ubOutKp, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV, f0, stride, biasVal);
        }
    } else if constexpr (S_CST == 3U) {
        constexpr uint32_t sFirst = OPT_R ? 2U : 1U;
        constexpr uint32_t sSecond = OPT_R ? 1U : 2U;
        if (s == sFirst) {
            StoreSlotInit(gmKeypointsU32, gmScores, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV);
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(8U);
            AscendC::CrossCoreWaitFlag(8U);
            AscendC::CrossCoreWaitFlag(9U);
        } else if (s == sSecond) {
            AscendC::CrossCoreWaitFlag(8U);
            StepSlotJoin(gmKeypointsU32, gmScores, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV, f0);
            AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(9U);
            AscendC::CrossCoreWaitFlag(9U);
        } else {
            AscendC::CrossCoreWaitFlag(8U);
            AscendC::CrossCoreWaitFlag(9U);
            FinishSlot0(gmKeypoints, gmKeypointsU32, gmScores, ubOutKp, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV, f0, stride, biasVal);
        }
    } else if constexpr (S_CST == 4U) {
        JoinFixed4<OPT_R>(
            gmKeypoints, gmKeypointsU32, gmScores, ubOutKp, ubOutKpU32, ubOutScore,
            s, bk, v0_1, valM, valE, valU, valV, f0, stride, biasVal);
    }
}

KPT_INL void JoinChainVar(
    const AscendC::GlobalTensor<float> &gmKeypoints,
    const AscendC::GlobalTensor<uint32_t> &gmKeypointsU32,
    const AscendC::GlobalTensor<float> &gmScores,
    const AscendC::LocalTensor<float> &ubOutKp,
    const AscendC::LocalTensor<uint32_t> &ubOutKpU32,
    const AscendC::LocalTensor<float> &ubOutScore,
    uint32_t effS, uint32_t s, uint32_t bk, float v0_1,
    float valM, float valE, float valU, float valV,
    float f0, float stride, float biasVal) {
    if (s == 1U) {
        StoreSlotInit(gmKeypointsU32, gmScores, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV);
    }
    SyncStepBlk<8U>();
    if (effS >= 3U) {
        if (s == 2U) {
            StepSlotJoin(gmKeypointsU32, gmScores, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV, f0);
        }
        SyncStepBlk<9U>();
    }
    if (effS >= 4U) {
        if (s == 3U) {
            StepSlotJoin(gmKeypointsU32, gmScores, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV, f0);
        }
        SyncStepBlk<10U>();
    }
    if (effS >= 5U) {
        if (s == 4U) {
            StepSlotJoin(gmKeypointsU32, gmScores, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV, f0);
        }
        SyncStepBlk<7U>();
    }
    if (effS >= 6U) {
        if (s == 5U) {
            StepSlotJoin(gmKeypointsU32, gmScores, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV, f0);
        }
        SyncStepBlk<15U>();
    }
    if (s == 0U) {
        FinishSlot0(gmKeypoints, gmKeypointsU32, gmScores, ubOutKp, ubOutKpU32, ubOutScore, bk, v0_1, valM, valE, valU, valV, f0, stride, biasVal);
    }
}

KPT_INL float SyncGlobalPeak(
    const AscendC::GlobalTensor<float> &gmKeypoints,
    const AscendC::GlobalTensor<float> &gmScores,
    const AscendC::LocalTensor<float> &ubOutKp,
    const AscendC::LocalTensor<float> &ubOutScore,
    const AscendC::LocalTensor<float> &ubTmpVec,
    uint32_t s, uint32_t bk, float valM) {
    const AscendC::DataCopyExtParams scalar1Params{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyExtParams vec2Params{1U, 2U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> padZeroF{false, 0U, 0U, 0.0f};
    if (s == 0U) {
        ubOutScore.SetValue(0U, valM);
        ubOutKp.SetValue(0U, 0.0f);
        ubOutKp.SetValue(1U, 0.0f);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(gmScores[bk], ubOutScore, scalar1Params);
        AscendC::DataCopyPad(gmKeypoints[bk * 2U], ubOutKp, vec2Params);
    }
    SyncAllBlk<4U>();
    if (s > 0U) {
        ubOutScore.SetValue(0U, valM);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetAtomicMax<float>();
        AscendC::DataCopyPad(gmScores[bk], ubOutScore, scalar1Params);
        AscendC::DisableDmaAtomic();
    }
    SyncAllBlk<5U>();
    AscendC::DataCopyPad(ubTmpVec[0], gmScores[bk], scalar1Params, padZeroF);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    return ubTmpVec.GetValue(0U);
}

KPT_INL void AccSlotVec(
    const AscendC::GlobalTensor<float> &gmKeypoints,
    const AscendC::GlobalTensor<float> &gmScores,
    const AscendC::LocalTensor<float> &ubOutKp,
    const AscendC::LocalTensor<float> &ubOutScore,
    const AscendC::LocalTensor<float> &ubTmpVec,
    uint32_t s, uint32_t bk, float adjE, float adjU, float adjV,
    float &outOtherU, float &outOtherV) {
    const AscendC::DataCopyExtParams scalar1Params{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyExtParams vec2Params{1U, 2U * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> padZeroF{false, 0U, 0U, 0.0f};
    if (s > 0U) {
        ubOutKp.SetValue(0U, adjU);
        ubOutKp.SetValue(1U, adjV);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopyPad(gmKeypoints[bk * 2U], ubOutKp, vec2Params);
        AscendC::DisableDmaAtomic();
    }
    SyncAllBlk<6U>();

    outOtherU = 0.0f;
    outOtherV = 0.0f;
    if (s == 0U) {
        AscendC::DataCopyPad(ubTmpVec[8], gmKeypoints[bk * 2U], vec2Params, padZeroF);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        ubOutScore.SetValue(0U, adjE);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID1);
        AscendC::DataCopyPad(gmScores[bk], ubOutScore, scalar1Params);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        outOtherU = ubTmpVec.GetValue(8U);
        outOtherV = ubTmpVec.GetValue(9U);
    }
}

KPT_INL void JoinManySlots(
    const AscendC::GlobalTensor<float> &gmKeypoints,
    const AscendC::GlobalTensor<float> &gmScores,
    const AscendC::LocalTensor<float> &ubOutKp,
    const AscendC::LocalTensor<float> &ubOutScore,
    const AscendC::LocalTensor<float> &ubTmpVec,
    uint32_t s, uint32_t bk,
    float valM, float valE, float valU, float valV,
    float f0, float stride, float biasVal) {
    const AscendC::DataCopyExtParams scalar1Params{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const AscendC::DataCopyPadExtParams<float> padZeroF{false, 0U, 0U, 0.0f};
    const float gMax = SyncGlobalPeak(gmKeypoints, gmScores, ubOutKp, ubOutScore, ubTmpVec, s, bk, valM);
    const float alpha = (valM < gMax) ? EvalPolyExp((valM - gMax) * f0) : 1.0f;
    const float adjE = alpha * valE;
    const float adjU = alpha * valU;
    const float adjV = alpha * valV;

    float otherU = 0.0f, otherV = 0.0f;
    AccSlotVec(
        gmKeypoints, gmScores, ubOutKp, ubOutScore, ubTmpVec, s, bk, adjE, adjU, adjV, otherU, otherV);
    SyncAllBlk<7U>();

    if (s > 0U) {
        ubOutScore.SetValue(0U, adjE);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopyPad(gmScores[bk], ubOutScore, scalar1Params);
        AscendC::DisableDmaAtomic();
    }
    SyncAllBlk<8U>();

    if (s == 0U) {
        AscendC::DataCopyPad(ubTmpVec[0], gmScores[bk], scalar1Params, padZeroF);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        StoreTaskOut(
            gmKeypoints, gmScores, ubOutKp, ubOutScore, bk,
            ubTmpVec.GetValue(0U), adjU + otherU, adjV + otherV, gMax, stride, biasVal);
    }
}

template <uint32_t S_CST, bool PRE_SYNC = false, bool OPT_R = false>
KPT_INL void JoinTileOut(
    const AscendC::GlobalTensor<float> &gmKeypoints,
    const AscendC::GlobalTensor<uint32_t> &gmKeypointsU32,
    const AscendC::GlobalTensor<float> &gmScores,
    const AscendC::LocalTensor<float> &ubOutKp,
    const AscendC::LocalTensor<uint32_t> &ubOutKpU32,
    const AscendC::LocalTensor<float> &ubOutScore,
    const AscendC::LocalTensor<float> &ubTmpVec,
    uint32_t S, uint32_t s, uint32_t bk, uint32_t v0, uint32_t baseR, uint32_t remR,
    float valM, float valE, float valU, float valV,
    float f0, float stride, float biasVal) {
    (void)v0;
    const uint32_t effS = (S_CST > 0U) ? S_CST : S;
    if constexpr (S_CST == 0U || S_CST == 1U) {
        if (effS == 1U) {
            StoreTaskOut(
                gmKeypoints, gmScores, ubOutKp, ubOutScore, bk,
                valE, valU, valV, valM, stride, biasVal);
            return;
        }
    }
    const float v0_1 = static_cast<float>(static_cast<int32_t>(baseR + ((1U < remR) ? 1U : remR)));
    if constexpr (PRE_SYNC && S_CST >= 2U && S_CST <= 4U) {
        JoinFixedN<S_CST, OPT_R>(
            gmKeypoints, gmKeypointsU32, gmScores, ubOutKp, ubOutKpU32, ubOutScore,
            s, bk, v0_1, valM, valE, valU, valV, f0, stride, biasVal);
        return;
    }
    if constexpr (S_CST == 0U || (S_CST >= 2U && S_CST <= 6U)) {
        if (effS <= 6U) {
            JoinChainVar(
                gmKeypoints, gmKeypointsU32, gmScores, ubOutKp, ubOutKpU32, ubOutScore,
                effS, s, bk, v0_1, valM, valE, valU, valV, f0, stride, biasVal);
            return;
        }
    }
    if constexpr (S_CST == 0U || S_CST > 6U) {
        JoinManySlots(
            gmKeypoints, gmScores, ubOutKp, ubOutScore, ubTmpVec,
            s, bk, valM, valE, valU, valV, f0, stride, biasVal);
    }
}

}
