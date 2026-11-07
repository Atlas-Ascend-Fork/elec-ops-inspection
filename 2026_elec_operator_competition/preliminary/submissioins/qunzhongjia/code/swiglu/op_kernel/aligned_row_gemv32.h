/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 黎何.
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

template <typename T> class AlignedRowGemv32 {
public:
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6>
    __aicore__ inline __attribute__((always_inline)) void RunStage2c286fd7(A0& gx, A1& gw1, A2& gw2, A3& gb1,
                                                                           A4& gb2, A5& gy, A6& hw1);
    template <class B0, class B1, class B2, class B3, class B4, class B5>
    __aicore__ inline __attribute__((always_inline)) void RunActive(B0& b1, B1& b2, B2& w1, B3& w2, B4& x,
                                                                    B5& y);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y);
};

template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6>
__aicore__ inline __attribute__((always_inline)) void
AlignedRowGemv32<T>::RunStage2c286fd7(A0& gx, A1& gw1, A2& gw2, A3& gb1, A4& gb2, A5& gy, A6& hw1)
{
    LocalTensor<T> hw2(TPosition::VECIN, 2048, 1024);
    LocalTensor<T> hx(TPosition::VECIN, 4096, 64);
    LocalTensor<T> hb(TPosition::VECIN, 4224, 64);
    LocalTensor<float> pg(TPosition::VECCALC, 4352, 2048);
    LocalTensor<float> pu(TPosition::VECCALC, 12544, 256);
    LocalTensor<float> ag(TPosition::VECCALC, 13568, 64);
    LocalTensor<T> gu(TPosition::VECCALC, 13824, 64);
    LocalTensor<T> sig(TPosition::VECCALC, 13952, 32);
    LocalTensor<T> ones(TPosition::VECCALC, 14016, 32);
    LocalTensor<T> out(TPosition::VECOUT, 14080, 32);
    SetAtomicNone();
    SetMaskNorm();
    constexpr uint64_t mask = 0;
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    DataCopy(hx, gx, 32);
    DataCopy(hx[32], gx, 32);
    DataCopy(hb, gb1, 32);
    DataCopy(hb[32], gb2, 32);
    DataCopy(hw1, gw1, 1024);
    DataCopy(hw2, gw2, 1024);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    Duplicate<float, false>(pg, 0.0f, mask, 32U, 1, 8);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    MulAddDst<float, T, false>(pg, hx, hw1, mask, 32U, BinaryRepeatParams{1, 1, 1, 8, 0, 4});
    BlockReduceSum<float, false>(pu, pg, 32U, mask, 1, 1, 8);
    PairReduceSum<float, false>(pg, pu, 4U, mask, 1, 1, 8);
    PairReduceSum<float, false>(ag, pg, 2U, mask, 1, 1, 8);
    Cast<T, float, false>(gu, ag, RoundMode::CAST_RINT, mask, 1, UnaryRepeatParams{1, 1, 4, 8});
    SetVectorMask<T, MaskMode::NORMAL>(0, ~0ULL);
    Add<T, false>(gu, gu, hb, mask, 1, BinaryRepeatParams{1, 1, 1, 8, 8, 8});
    FastGate<T, 32U, false>(gu, gu[32], sig, out, ones);
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopy(gy, out, 32U);
}

template <typename T>
template <class B0, class B1, class B2, class B3, class B4, class B5>
__aicore__ inline __attribute__((always_inline)) void AlignedRowGemv32<T>::RunActive(B0& b1, B1& b2, B2& w1,
                                                                                     B3& w2, B4& x, B5& y)
{
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
    gx.SetGlobalBuffer((__gm__ T*)x, 32);
    gy.SetGlobalBuffer((__gm__ T*)y, 32);
    gw1.SetGlobalBuffer((__gm__ T*)w1, 1024);
    gw2.SetGlobalBuffer((__gm__ T*)w2, 1024);
    gb1.SetGlobalBuffer((__gm__ T*)b1, 32);
    gb2.SetGlobalBuffer((__gm__ T*)b2, 32);
    LocalTensor<T> hw1(TPosition::VECIN, 0, 1024);
    RunStage2c286fd7(gx, gw1, gw2, gb1, gb2, gy, hw1);
}

template <typename T>
__aicore__ inline void AlignedRowGemv32<T>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
                                                GM_ADDR y)
{
    if (GetBlockIdx() != 0U) {
        return;
    }
    RunActive(b1, b2, w1, w2, x, y);
}
