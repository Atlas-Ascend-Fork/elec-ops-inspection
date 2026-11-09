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
    template <class OwnerState, class Capture0>
    __aicore__ static inline void SharedPipeBarrierStage29(Capture0 &state) {
        for (uint32_t i = 0; i < 8U; ++i)
            state.tab.SetValue(i, int32_t(i * OwnerState::P * 4U));
        state.SV();
        Adds(state.tab[8U], state.tab, int32_t(8U * OwnerState::P * 4U), 8);
        PipeBarrier<PIPE_V>();
        Adds(state.tab[16U], state.tab, int32_t(16U * OwnerState::P * 4U), 16);
        PipeBarrier<PIPE_V>();
        Adds(state.tab[32U], state.tab, int32_t(32U * OwnerState::P * 4U), 32);
        PipeBarrier<PIPE_V>();
        Adds(state.tab[64U], state.tab, int32_t(64U * OwnerState::P * 4U), 64);
        PipeBarrier<PIPE_V>();
    }
    template <class OwnerState> __aicore__ static inline void RunPipeBarrierPhase20SharedPhase7(OwnerState &state) {

        if (state.rank == 3U) {
            state.PrepareTail64Indices();
        } else {
            SharedPipeBarrierStage29<OwnerState>(state);
            Adds(state.tab[128U], state.tab, int32_t(128U * OwnerState::P * 4U), 128);
            PipeBarrier<PIPE_V>();
            LDET_MASK_SET(state.n);
            for (uint32_t r = 0; r < 8; ++r)
                vadds(reinterpret_cast<__ubuf__ int32_t *>(state.lanes[r * OwnerState::NMAX].GetPhyAddr()),
                      reinterpret_cast<__ubuf__ int32_t *>(state.tab.GetPhyAddr()), int32_t(r * 4), 1, 1, 1, 8, 8);
            LDET_MASK_RST();
        }
    }
