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
    template <class Self, auto T, auto TS>
    __aicore__ static inline void SharedOwnerPipeBarrierStage15(Self &state, uint32_t &f, uint32_t &h, float &inv,
                                                                __ubuf__ float *&mp, const uint32_t &p,
                                                                bool &pendingFar, __ubuf__ float *&rp) {
        if (p != f)
            state.sign = -state.sign;
        Shared36PipeBarrierStage7<T, TS>(f, h, inv, mp, p, pendingFar, rp, state);
        state.repair(p - f);
        PipeBarrier<PIPE_V>();
    }
