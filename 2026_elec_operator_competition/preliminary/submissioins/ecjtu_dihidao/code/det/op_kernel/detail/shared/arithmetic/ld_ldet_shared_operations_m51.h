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
    template <class Self, bool ClassValue0> __aicore__ static inline void SharedOwnerPipeBarrierStage13(Self &state) {
        if constexpr (ClassValue0) {
            Shared14PipeBarrierStage3(state);
            Adds(state.tab[128U], state.tab, int32_t(128U * Self::P * 4U), Self::NMAX - 128U);
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t i = 0; i < state.n; ++i)
                state.tab.SetValue(i, int32_t(i * Self::P * 4));
            state.SV();
        }
    }
