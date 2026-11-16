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

KPT_INL void InitBaseSeq(
    const AscendC::LocalTensor<float> &ubSeqF,
    const AscendC::LocalTensor<int32_t> &ubSeqI) {
    #pragma unroll
    for (int32_t i = 0; i < 8; ++i) {
        ubSeqF.SetValue(static_cast<uint32_t>(i), static_cast<float>(i));
        ubSeqI.SetValue(static_cast<uint32_t>(i), i * 4 - 4);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
}

template <uint32_t W>
KPT_INL void FillAxisBase(
    __ubuf__ float *pAxisU,
    __ubuf__ float *pAxisV,
    __ubuf__ int32_t *pOutL,
    __ubuf__ int32_t *pOutR,
    __ubuf__ float *pSeqF,
    __ubuf__ int32_t *pSeqI,
    uint32_t v0) {
    set_vector_mask(0ULL, 0xFFULL);
    #pragma unroll
    for (int32_t g = 0; g < 8; ++g) {
        OpVadds(pAxisU + static_cast<uint32_t>(g * 8), pSeqF, static_cast<float>(g * 8), 1U);
        OpVaddsI(pOutL + static_cast<uint32_t>(g * 8), pSeqI, g * 32, 1U);
        OpVaddsI(pOutR + static_cast<uint32_t>(g * 8), pSeqI, g * 32 + 8, 1U);
    }
    if constexpr (W == 96U) {
        #pragma unroll
        for (int32_t g = 8; g < 12; ++g) {
            OpVadds(pAxisU + static_cast<uint32_t>(g * 8), pSeqF, static_cast<float>(g * 8), 1U);
        }
    }
    set_vector_mask(~0ULL, ~0ULL);
    AscendC::PipeBarrier<PIPE_V>();
    OpVadds(pAxisV, pAxisU, static_cast<float>(static_cast<int32_t>(v0)), 1U);
}

template <bool EXT_40>
KPT_INL void GrowMapFixed(__ubuf__ int32_t *pMap0, __ubuf__ int32_t *pMap1, uint32_t M) {
    #pragma unroll
    for (int32_t g = 1; g < 8; ++g) {
        OpVaddsI(pMap0 + static_cast<uint32_t>(g * 64), pMap0, g * 256, 1U);
        OpVaddsI(pMap1 + static_cast<uint32_t>(g * 64), pMap1, g * 256, 1U);
    }
    AscendC::PipeBarrier<PIPE_V>();

    const uint32_t mReps = M >> 6U;
    if (mReps > 8U) {
        const uint8_t r1 = static_cast<uint8_t>((mReps > 16U) ? 8U : (mReps - 8U));
        OpVaddsI(pMap0 + 512U, pMap0, 2048, r1);
        OpVaddsI(pMap1 + 512U, pMap1, 2048, r1);
    }
    if (mReps > 16U) {
        const uint8_t r2 = static_cast<uint8_t>((mReps > 24U) ? 8U : (mReps - 16U));
        OpVaddsI(pMap0 + 1024U, pMap0, 4096, r2);
        OpVaddsI(pMap1 + 1024U, pMap1, 4096, r2);
    }
    if (mReps > 24U) {
        const uint8_t r3 = static_cast<uint8_t>((mReps > 32U) ? 8U : (mReps - 24U));
        OpVaddsI(pMap0 + 1536U, pMap0, 6144, r3);
        OpVaddsI(pMap1 + 1536U, pMap1, 6144, r3);
    }
    if constexpr (EXT_40) {
        if (mReps > 32U) {
            const uint8_t r4 = static_cast<uint8_t>((mReps > 40U) ? 8U : (mReps - 32U));
            OpVaddsI(pMap0 + 2048U, pMap0, 8192, r4);
            OpVaddsI(pMap1 + 2048U, pMap1, 8192, r4);
        }
        if (mReps > 40U) {
            const uint8_t r5 = static_cast<uint8_t>(mReps - 40U);
            OpVaddsI(pMap0 + 2560U, pMap0, 10240, r5);
            OpVaddsI(pMap1 + 2560U, pMap1, 10240, r5);
        }
    } else if (mReps > 32U) {
        const uint8_t r4 = static_cast<uint8_t>(mReps - 32U);
        OpVaddsI(pMap0 + 2048U, pMap0, 8192, r4);
        OpVaddsI(pMap1 + 2048U, pMap1, 8192, r4);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

KPT_INL void GrowMapVar(__ubuf__ int32_t *pMap0, __ubuf__ int32_t *pMap1, uint32_t M) {
    if (M <= 64U) {
        return;
    }
    #pragma unroll
    for (int32_t g = 1; g < 8; ++g) {
        OpVaddsI(pMap0 + static_cast<uint32_t>(g * 64), pMap0, g * 256, 1U);
        OpVaddsI(pMap1 + static_cast<uint32_t>(g * 64), pMap1, g * 256, 1U);
    }
    AscendC::PipeBarrier<PIPE_V>();
    if (M > 512U) {
        for (uint32_t off = 512U; off < M; off += 512U) {
            const uint32_t chunk = ((M - off) < 512U) ? (M - off) : 512U;
            VecAddsISpan(pMap0 + off, pMap0, static_cast<int32_t>(off << 2U), chunk);
            VecAddsISpan(pMap1 + off, pMap1, static_cast<int32_t>(off << 2U), chunk);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <uint32_t W, bool FAST_M = false, bool EXT_40 = false>
KPT_INL void PrepAxisFull(
    const AscendC::LocalTensor<float> &ubAxisU,
    const AscendC::LocalTensor<float> &ubAxisV,
    const AscendC::LocalTensor<float> &mapLBuf,
    const AscendC::LocalTensor<float> &mapRBuf,
    const AscendC::LocalTensor<float> &ubSeqF,
    const AscendC::LocalTensor<int32_t> &ubSeqI,
    uint32_t v0,
    uint32_t M,
    uint32_t R) {
    __ubuf__ float *pAxisU = reinterpret_cast<__ubuf__ float *>(ubAxisU.GetPhyAddr());
    __ubuf__ float *pAxisV = reinterpret_cast<__ubuf__ float *>(ubAxisV.GetPhyAddr());
    __ubuf__ float *pSeqF = reinterpret_cast<__ubuf__ float *>(ubSeqF.GetPhyAddr());
    __ubuf__ int32_t *pSeqI = reinterpret_cast<__ubuf__ int32_t *>(ubSeqI.GetPhyAddr());
    __ubuf__ int32_t *pMap0 = reinterpret_cast<__ubuf__ int32_t *>(mapLBuf.GetPhyAddr());
    __ubuf__ int32_t *pMap1 = reinterpret_cast<__ubuf__ int32_t *>(mapRBuf.GetPhyAddr());
    __ubuf__ uint32_t *pMap0U = reinterpret_cast<__ubuf__ uint32_t *>(mapLBuf.GetPhyAddr());
    __ubuf__ uint32_t *pMap1U = reinterpret_cast<__ubuf__ uint32_t *>(mapRBuf.GetPhyAddr());

    FillAxisBase<W>(pAxisU, pAxisV, pMap0, pMap1, pSeqF, pSeqI, v0);
    if constexpr (FAST_M) {
        GrowMapFixed<EXT_40>(pMap0, pMap1, M);
    } else {
        GrowMapVar(pMap0, pMap1, M);
    }

    const uint32_t endBytes = M * static_cast<uint32_t>(sizeof(float));
    constexpr uint8_t W_BLK = static_cast<uint8_t>(W >> 3U);
    set_vector_mask(0ULL, 1ULL);
    OpVdupS(pMap0U, endBytes, static_cast<uint8_t>(R), W_BLK);
    set_vector_mask(0ULL, 0x80ULL);
    OpVdupS(pMap1U + (W - 8U), endBytes, static_cast<uint8_t>(R), W_BLK);
    set_vector_mask(~0ULL, ~0ULL);
}

template <uint32_t W, uint32_t LEN_SD>
KPT_INL void PrepAxisCompact(
    const AscendC::LocalTensor<float> &ubAxisU,
    const AscendC::LocalTensor<float> &ubAxisV,
    const AscendC::LocalTensor<int32_t> &ubMapBase,
    const AscendC::LocalTensor<float> &ubSeqF,
    const AscendC::LocalTensor<int32_t> &ubSeqI,
    uint32_t v0) {
    __ubuf__ float *pAxisU = reinterpret_cast<__ubuf__ float *>(ubAxisU.GetPhyAddr());
    __ubuf__ float *pAxisV = reinterpret_cast<__ubuf__ float *>(ubAxisV.GetPhyAddr());
    __ubuf__ float *pSeqF = reinterpret_cast<__ubuf__ float *>(ubSeqF.GetPhyAddr());
    __ubuf__ int32_t *pSeqI = reinterpret_cast<__ubuf__ int32_t *>(ubSeqI.GetPhyAddr());
    __ubuf__ int32_t *pMapL = reinterpret_cast<__ubuf__ int32_t *>(ubMapBase.GetPhyAddr());
    __ubuf__ int32_t *pMapR = pMapL + LEN_SD;

    FillAxisBase<W>(pAxisU, pAxisV, pMapL, pMapR, pSeqF, pSeqI, v0);
    constexpr int32_t N_REP = static_cast<int32_t>(LEN_SD >> 6U);
    #pragma unroll
    for (int32_t g = 1; g < N_REP; ++g) {
        OpVaddsI(pMapL + static_cast<uint32_t>(g * 64), pMapL, g * 256, 1U);
        OpVaddsI(pMapR + static_cast<uint32_t>(g * 64), pMapR, g * 256, 1U);
    }
    AscendC::PipeBarrier<PIPE_V>();

    set_vector_mask(0ULL, 1ULL);
    OpVdupS(reinterpret_cast<__ubuf__ uint32_t *>(pMapL), 0U, 1U, 8U);
    set_vector_mask(~0ULL, ~0ULL);
}

}
