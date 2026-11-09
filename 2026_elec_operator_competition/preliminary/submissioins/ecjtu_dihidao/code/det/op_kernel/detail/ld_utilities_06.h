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
#if (defined(__DAV_C220_VEC__))
template <class DT_A, uint32_t N>
__aicore__ inline void RunUB216Batch(GM_ADDR a, GM_ADDR det, const LinalgDetTilingData &td) {
    const uint32_t cores = GetBlockNum(), id = GetBlockIdx();
    const uint32_t base = td.batchCount / cores, extra = td.batchCount % cores;
    const uint32_t start = id * base + (id < extra ? id : extra), count = base + (id < extra ? 1U : 0U);
    LinalgDetTilingData one{};
    one.batchCount = 1U;
    one.n = N;
    one.stride = N;
    one.matPerCore = 1U;
    one.isLarge = 0U;
    one.groupSize = 1U;
    for (uint32_t m = 0U; m < count; ++m) {
        const uint64_t index = static_cast<uint64_t>(start + m);
        KernelUB216<DT_A, N> op;
        op.template Run<true>(a + index * N * N * 4U, det + index * 4U, one);
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
    }
}
#endif
