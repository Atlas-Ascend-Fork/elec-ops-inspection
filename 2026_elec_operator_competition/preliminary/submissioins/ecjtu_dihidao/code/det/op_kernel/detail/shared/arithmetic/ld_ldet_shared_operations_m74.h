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
    template <class Self, class Capture0> __aicore__ static inline void Shared24PipeBarrierStage3(Capture0 &state) {
        Shared36PipeBarrierStage4<Self>(state);
        PipeBarrier<PIPE_V>();
        Adds(state.tab[128U], state.tab, int32_t(128U * Self::P * 4U), 128);
        PipeBarrier<PIPE_V>();
        LDET_MASK_SET(state.n);
    }
    template <class Self> __aicore__ static inline void SharedOwner15PipeBarrierStage1(Self &state) {
        Shared24PipeBarrierStage3<Self>(state);
        for (uint32_t r = 0; r < 8; ++r)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(state.lanes[r * Self::NMAX].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(state.tab.GetPhyAddr()), int32_t(r * 4), 1, 1, 1, 8, 8);
    }
