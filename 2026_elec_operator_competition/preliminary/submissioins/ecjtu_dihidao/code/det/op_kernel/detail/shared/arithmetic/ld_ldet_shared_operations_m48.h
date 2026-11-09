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
// Included in the LdetSharedOperations class; preserves its member scope and access.
    template <class Capture0> __aicore__ static inline void Shared14PipeBarrierStage3(Capture0 &state) {
        for (uint32_t i = 0; i < 8U; ++i)
            state.tab.SetValue(i, int32_t(i * Capture0::P * 4U));
        state.SV();
        Adds(state.tab[8U], state.tab, int32_t(8U * Capture0::P * 4U), 8);
        PipeBarrier<PIPE_V>();
        Adds(state.tab[16U], state.tab, int32_t(16U * Capture0::P * 4U), 16);
        PipeBarrier<PIPE_V>();
        Adds(state.tab[32U], state.tab, int32_t(32U * Capture0::P * 4U), 32);
        PipeBarrier<PIPE_V>();
        Adds(state.tab[64U], state.tab, int32_t(64U * Capture0::P * 4U), 64);
        PipeBarrier<PIPE_V>();
    }
    template <class Self> __aicore__ static inline void SharedOwnerPipeBarrierStage9(Self &state) {
        Shared14PipeBarrierStage3(state);
        Adds(state.tab[128U], state.tab, int32_t(128U * Self::P * 4U), 128);
        PipeBarrier<PIPE_V>();
        LDET_MASK_SET(state.n);
        for (uint32_t r = 0; r < 8; ++r)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(state.lanes[r * Self::NMAX].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(state.tab.GetPhyAddr()), int32_t(r * 4), 1, 1, 1, 8, 8);
    }
