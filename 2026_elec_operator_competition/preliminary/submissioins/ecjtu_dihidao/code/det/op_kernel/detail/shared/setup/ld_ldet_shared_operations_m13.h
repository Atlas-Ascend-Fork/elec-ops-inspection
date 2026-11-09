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
    template <class OwnerState>
    __aicore__ static inline void RunDataCopyPhase16SharedPhase5(OwnerState &state, uint32_t &q) {

        if (q == 1U) {
            state.recycleState = 1U;
            if (state.rank == 0U) {
                state.owned = 1U;
                if (!state.singular) {
                    DataCopy(state.mat[32U * OwnerState::S], state.relay, {224U, 4U, 0U, (OwnerState::S - 32U) / 8U});
                    state.MV();
                }
            } else if (state.rank == 1U)
                state.owned = 1U;
        }
        if (q == 3U) {
            state.recycleState = 2U;
            if (state.rank == 1U) {
                state.owned = 1U;
                if (!state.singular) {
                    DataCopy(state.mat[96U * OwnerState::S], state.relay, {160U, 4U, 0U, (OwnerState::S - 32U) / 8U});
                    state.MV();
                }
            } else if (state.rank == 2U)
                state.owned = 1U;
        }
    }
