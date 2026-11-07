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
struct CubeSlotSync {
public:
    __aicore__ static inline __attribute__((always_inline)) void WaitReady(uint32_t s);
    __aicore__ static inline __attribute__((always_inline)) void Ready(uint32_t s);
    __aicore__ static inline __attribute__((always_inline)) void WaitConsumed(uint32_t s);
    __aicore__ static inline __attribute__((always_inline)) void Consumed(uint32_t s);
};
}

namespace optiling {
__aicore__ inline __attribute__((always_inline)) void CubeSlotSync::WaitReady(uint32_t s)
{
    if (s == 0) {
        CrossCoreWaitFlag(0x6);
    } else if (s == 1) {
        CrossCoreWaitFlag(0x7);
    } else if (s == 2) {
        CrossCoreWaitFlag(0xB);
    } else {
        CrossCoreWaitFlag(0xE);
    }
}

__aicore__ inline __attribute__((always_inline)) void CubeSlotSync::Ready(uint32_t s)
{
    if (s == 0) {
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x6);
    } else if (s == 1) {
        CrossCoreSetFlag<0x2, PIPE_FIX>(0x7);
    } else if (s == 2) {
        CrossCoreSetFlag<0x2, PIPE_FIX>(0xB);
    } else {
        CrossCoreSetFlag<0x2, PIPE_FIX>(0xE);
    }
}

__aicore__ inline __attribute__((always_inline)) void CubeSlotSync::WaitConsumed(uint32_t s)
{
    if (s == 0) {
        CrossCoreWaitFlag(0x8);
    } else if (s == 1) {
        CrossCoreWaitFlag(0x9);
    } else if (s == 2) {
        CrossCoreWaitFlag(0xC);
    } else {
        CrossCoreWaitFlag(0xF);
    }
}

__aicore__ inline __attribute__((always_inline)) void CubeSlotSync::Consumed(uint32_t s)
{
    if (s == 0) {
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x8);
    } else if (s == 1) {
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9);
    } else if (s == 2) {
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0xC);
    } else {
        CrossCoreSetFlag<0x2, PIPE_MTE3>(0xF);
    }
}

}
