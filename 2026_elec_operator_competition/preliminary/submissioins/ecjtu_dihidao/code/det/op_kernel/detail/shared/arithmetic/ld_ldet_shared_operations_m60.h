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
    template <class Self, class Block36Capture3_0>
    __aicore__ static inline void Shared36PipeBarrierStage4(Block36Capture3_0 &state) {
        for (uint32_t i = 0; i < 8U; ++i)
            state.tab.SetValue(i, int32_t(i * Self::P * 4U));
        state.SV();
        Adds(state.tab[8U], state.tab, int32_t(8U * Self::P * 4U), 8);
        PipeBarrier<PIPE_V>();
        Adds(state.tab[16U], state.tab, int32_t(16U * Self::P * 4U), 16);
        PipeBarrier<PIPE_V>();
        Adds(state.tab[32U], state.tab, int32_t(32U * Self::P * 4U), 32);
        PipeBarrier<PIPE_V>();
        Adds(state.tab[64U], state.tab, int32_t(64U * Self::P * 4U), 64);
    }
    template <class Self, bool ClassValue0, bool ClassValue1>
    __aicore__ static inline void SharedOwnerPipeBarrierStage23(Self &state, const bool &valid) {
        if constexpr (Self::PREFIX_STREAM && (!ClassValue0 || ClassValue1)) {
            if (valid && state.rank < 6U)
                LuPublish(state.prefixTicket(state.rank), 0U);
        }
        Shared36PipeBarrierStage4<Self>(state);
    }
