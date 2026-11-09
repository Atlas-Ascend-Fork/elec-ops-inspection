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
    template <class OwnerState, bool ClassValue0, bool ClassValue1, bool ClassValue2, bool WAVED>
    __aicore__ static inline void RunDataCopyPadPhase19SharedPhase6(OwnerState &state, uint32_t &rounds,
                                                                    const bool &valid) {

        const uint32_t last = rounds - 1U;
        if (valid && state.rank == (ClassValue0 ? last / 2U : last % OwnerState::G)) {
            if (!state.singular) {
                if constexpr (ClassValue1 || ClassValue2)
                    state.template factor<false>(last * OwnerState::B, last);
                else
                    state.factor(last * OwnerState::B, last);
            }
            state.meta.SetValue(0, state.singular ? 0.0f : state.sign * state.final());
            state.SM();
            DataCopyPad(state.out, state.meta, DataCopyExtParams{1, 4, 0, 0, 0});
            state.MS();
        }
        if constexpr (WAVED)
            SyncAll<true>();
    }
