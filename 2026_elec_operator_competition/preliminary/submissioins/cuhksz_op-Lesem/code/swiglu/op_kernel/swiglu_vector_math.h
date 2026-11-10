/*
 * Copyright (c) 2026 Lesem.
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

namespace AscendC::SwiGluKernel {

template <typename T>
__aicore__ inline void ApplySilu(LocalTensor<T> z, LocalTensor<T> neg, uint32_t count) {
    Muls(neg, z, -1.0f, count);
    PipeBarrier<PIPE_V>();
    Exp(neg, neg, count);
    PipeBarrier<PIPE_V>();
    Adds(neg, neg, 1.0f, count);
    PipeBarrier<PIPE_V>();
    Div(z, z, neg, count);
    PipeBarrier<PIPE_V>();
}

// Both GM scratch slots remain owned by Cube until the AIV finishes reading C2.
template <uint32_t ReadyFlag, uint32_t AckFlag>
__aicore__ inline void FinishCubeProjectionPair(LocalTensor<half> c1, LocalTensor<half> c2,
    LocalTensor<float> z, LocalTensor<half> b1, LocalTensor<half> b2, GlobalTensor<half> scratch,
    uint32_t area, uint32_t slot, uint32_t tileRows, uint32_t tileN, TEventID m2v) {
    const uint32_t count = tileRows * tileN;
    const auto stride = static_cast<uint8_t>(tileN / 16U);
    const BinaryRepeatParams repeat{1, 1, 1, stride, stride, 0};
    CrossCoreWaitFlag<2, PIPE_MTE2>(ReadyFlag + slot * 2U);
    DataCopy(c1, scratch[slot * 2U * area], count);
    SetFlag<HardEvent::MTE2_V>(m2v);
    WaitFlag<HardEvent::MTE2_V>(m2v);
    Add(c1, c1, b1, static_cast<uint64_t>(tileN), static_cast<uint8_t>(tileRows), repeat);
    PipeBarrier<PIPE_V>();
    Cast(z, c1, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    ApplySilu(z, c1.ReinterpretCast<float>(), count);
    Cast(c1, z, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    CrossCoreWaitFlag<2, PIPE_MTE2>(ReadyFlag + 1U + slot * 2U);
    DataCopy(c2, scratch[(slot * 2U + 1U) * area], count);
    CrossCoreSetFlag<2, PIPE_MTE2>(AckFlag + slot);
    SetFlag<HardEvent::MTE2_V>(m2v);
    WaitFlag<HardEvent::MTE2_V>(m2v);
    Add(c2, c2, b2, static_cast<uint64_t>(tileN), static_cast<uint8_t>(tileRows), repeat);
    PipeBarrier<PIPE_V>();
    Mul(c1, c1, c2, count);
}

__aicore__ inline void StoreStridedCubeOutput(GlobalTensor<half> dst, LocalTensor<half> src,
    uint32_t mc, uint32_t nc, uint32_t tileN, uint32_t dim,
    TEventID vm3, TEventID m3v, TEventID m3m2) {
    SetFlag<HardEvent::V_MTE3>(vm3);
    WaitFlag<HardEvent::V_MTE3>(vm3);
    const DataCopyExtParams out{static_cast<uint16_t>(mc), nc * 2U,
        (tileN - nc) * 2U / 32U, (dim - nc) * 2U, 0};
    DataCopyPad(dst, src, out);
    SetFlag<HardEvent::MTE3_V>(m3v);
    WaitFlag<HardEvent::MTE3_V>(m3v);
    SetFlag<HardEvent::MTE3_MTE2>(m3m2);
    WaitFlag<HardEvent::MTE3_MTE2>(m3m2);
}

} // namespace AscendC::SwiGluKernel
