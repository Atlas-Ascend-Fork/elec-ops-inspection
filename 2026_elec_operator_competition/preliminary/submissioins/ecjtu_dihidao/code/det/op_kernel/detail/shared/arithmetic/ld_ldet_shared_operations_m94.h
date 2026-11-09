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
    template <auto TS, auto LAST_COUNT, class State>
    __aicore__ static inline void PrepareTailRowIndices46(State &state) {
        for (uint32_t j = 0; j < 8U; ++j)
            state.tab.SetValue(j, static_cast<int32_t>(j * TS * 4U));
        state.SV();
        Adds(state.tab[8U], state.tab, static_cast<int32_t>(8U * TS * 4U), 8);
        PipeBarrier<PIPE_V>();
        Adds(state.tab[16U], state.tab, static_cast<int32_t>(16U * TS * 4U), 16);
        PipeBarrier<PIPE_V>();
        Adds(state.tab[32U], state.tab, static_cast<int32_t>(32U * TS * 4U), LAST_COUNT);
        PipeBarrier<PIPE_V>();
    }

    template <bool DEFER, class State>
    __aicore__ static inline bool PrepareSignedPivot51(State &state, float maxv, __ubuf__ float *cp, uint32_t k,
                                                       uint32_t p, float &piv, float &inv) {
        if (maxv == 0) {
            state.singular = 1;
            return false;
        }
        const bool normalPivot = maxv >= 1.1754943508222875e-38f && maxv <= 3.4028234663852886e38f;
        piv = cp[p - k];
        if constexpr (DEFER) {
            if (!normalPivot && (piv == 0.0f || ((state.word(piv) >> 23) & 255U) == 255U)) {
                state.acc(piv);
                return false;
            }
        } else {
            state.acc(piv);
            if (state.singular)
                return false;
        }
        inv = -1.0f / piv;
        if (!normalPivot && ((state.word(inv) >> 23) & 255U) == 255U) {
            if constexpr (DEFER)
                state.acc(piv);
            state.singular = 2;
            return false;
        }
        return true;
    }
