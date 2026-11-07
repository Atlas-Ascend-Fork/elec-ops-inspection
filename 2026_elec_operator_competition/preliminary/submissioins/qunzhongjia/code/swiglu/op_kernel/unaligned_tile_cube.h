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

namespace optiling {
template <typename T> class UnalignedTileCube {
public:
    template <class A0, class A1, class A2, class A3, class A4, class A5>
    __aicore__ inline __attribute__((always_inline)) void RunStage55ed93fe(A0& core, A1& gb1, A2& gb2, A3& gy,
                                                                           A4& tmp, A5& off);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               GM_ADDR ws);

private:
    DirectPairCube<T, 32U, 0U, true, false, true> cube_;
};
}

namespace optiling {
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5>
__aicore__ inline __attribute__((always_inline)) void
UnalignedTileCube<T>::RunStage55ed93fe(A0& core, A1& gb1, A2& gb2, A3& gy, A4& tmp, A5& off)
{
    const uint32_t sub = GetSubBlockIdx();
    const uint32_t nr = 9U - sub;
    const uint32_t begin = 9U * sub;
    LocalTensor<T> g(TPosition::VECIN, 0, 432);
    LocalTensor<T> u(TPosition::VECIN, 864, 432);
    LocalTensor<T> b1t(TPosition::VECIN, 1728, 48);
    LocalTensor<T> b2t(TPosition::VECIN, 1824, 48);
    LocalTensor<T> sig(TPosition::VECCALC, 1920, 432);
    LocalTensor<T> out(TPosition::VECOUT, 2784, 432);
    SetMaskNorm();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    DataCopyExtParams bc{1, 66, 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, 15, static_cast<T>(0)};
    DataCopyPad(b1t, gb1, bc, pad);
    DataCopyPad(b2t, gb2, bc, pad);
    CrossCoreWaitFlag(0x6);
    DataCopyParams cp{static_cast<uint16_t>(nr), 3, 3, 0};
    DataCopy(g, tmp[off + begin * 96U], cp);
    DataCopy(u, tmp[off + begin * 96U + 48U], cp);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    const BinaryRepeatParams rp{1, 1, 1, 3, 3, 0};
    Add(g, g, b1t, static_cast<uint64_t>(33), static_cast<uint8_t>(nr), rp);
    Add(u, u, b2t, static_cast<uint64_t>(33), static_cast<uint8_t>(nr), rp);
    PipeBarrier<PIPE_V>();
    GateCounter<T, true>(g, u, sig, out, out, nr * 48U);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams oc{static_cast<uint16_t>(nr), 66, 0, 0, 0};
    DataCopyPad(gy[(core * 17U + begin) * 33U], out, oc);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8);
}

template <typename T>
__aicore__ inline void UnalignedTileCube<T>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
                                                 GM_ADDR y, GM_ADDR ws)
{
    uint32_t core = GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= 2U;
    }
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
    GlobalTensor<T> tmp;
    ::SharedPrimitive9aae833572<T>(b1, b2, gb1, gb2, gw1, gw2, gx, gy, w1, w2, x, y);
    tmp.SetGlobalBuffer((__gm__ T*)ws);
    const uint32_t off = core * 32U * 96U;
    SetAtomicNone();
    if ASCEND_IS_AIC {
        cube_.Init(33U, 32U, 48U);
        cube_.Compute(gx[core * 17U * 33U], gw1, gw2, tmp[off], 17U, 33U);
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x6);
        CrossCoreWaitFlag(0x8);
    }
    if ASCEND_IS_AIV {
        RunStage55ed93fe(core, gb1, gb2, gy, tmp, off);
    }
    PipeBarrier<PIPE_ALL>();
}

}
