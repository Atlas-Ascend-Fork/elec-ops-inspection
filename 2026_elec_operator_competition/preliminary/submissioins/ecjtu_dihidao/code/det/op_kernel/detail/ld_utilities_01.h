/*
 * Copyright (c) 2026 dihidao.
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
// Private device implementation; included by ld_device.h.
template <uint32_t... Keys> __aicore__ constexpr bool LdetRouteKeyIs(uint32_t selected) {
    constexpr uint32_t candidates[] = {Keys...};
    for (uint32_t key : candidates)
        if (selected == key)
            return true;
    return false;
}

__aicore__ inline uint32_t LuSpan(uint32_t m) {
    return (((m + 3U) / 4U + 15U) / 16U) * 16U;
}

__aicore__ inline void LuFlushFlag(GM_ADDR flag) {
    AscendC::GlobalTensor<int64_t> line;
    line.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(flag), 8U);
    AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        line);
}

__aicore__ inline void LuPublish(GM_ADDR flag, uint32_t value) {
    *reinterpret_cast<volatile __gm__ uint32_t *>(flag) = value;
    LuFlushFlag(flag);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

__aicore__ inline void LuPoll(GM_ADDR flag, uint32_t value) {
    uint32_t spins = 0;
    bool ready = false;
    do {
        LuFlushFlag(flag);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        ready = *reinterpret_cast<volatile __gm__ uint32_t *>(flag) == value;
        if (!ready && ++spins > 8U)
            for (volatile uint32_t delay = 0; delay < 64U; ++delay) {
            }
    } while (!ready);
}

__aicore__ inline void LuControlWait(GM_ADDR f, uint32_t epoch = 1U) {
    auto p = reinterpret_cast<volatile __gm__ uint32_t *>(f);
    bool ready = false;
    do {
        LuFlushFlag(f);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        ready = p[0] == epoch * 32U + 31U && p[2] == 0xC1493B27U;
    } while (!ready);
}

__aicore__ inline void LuPollN256Prefix(GM_ADDR flag, uint32_t value) {
    bool ready = false;
    do {
        LuFlushFlag(flag);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        ready = *reinterpret_cast<volatile __gm__ uint32_t *>(flag) == value;
    } while (!ready);
}

__aicore__ inline void LuClear(GM_ADDR flags, uint32_t count = 18U) {
    for (uint32_t i = 0; i < count; ++i) {
        auto p = reinterpret_cast<volatile __gm__ uint32_t *>(flags + i * 64U);
        for (uint32_t j = 0; j < 16U; ++j)
            p[j] = 0U;
        LuFlushFlag(flags + i * 64U);
    }
    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
}

__aicore__ inline uint64_t LuWideWorkspaceBytes(uint32_t n) {
    return (2ULL * n * 512U + 512U * 32U + 48U + 256U) * 4U;
}

__aicore__ inline uint64_t LuWideGroupBytes(uint32_t n) {
    return LuWideWorkspaceBytes(n) + 65536U + 65536U + 27U * 64U;
}

__aicore__ inline uint32_t LuN256Span(uint32_t rows, uint32_t cores) {
    if (cores == 0U)
        return 0U;
    return (((rows + cores - 1U) / cores + 15U) / 16U) * 16U;
}
