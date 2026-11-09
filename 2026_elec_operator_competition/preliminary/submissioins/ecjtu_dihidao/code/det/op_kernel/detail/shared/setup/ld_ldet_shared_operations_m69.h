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
    template <class Self> __aicore__ static inline void SharedOwnerDataCopyStage32(Self &state, uint32_t &q) {
        if (q == 1U && state.rank == 1U) {
            state.VM();
            DataCopy(state.relay, state.mat[32U * Self::S + 32U], {224U, 4U, (Self::S - 32U) / 8U, 0U});
            state.MS();
        }
        if (q == 3U && state.rank == 2U) {
            state.VM();
            DataCopy(state.relay, state.mat[96U * Self::S + 32U], {160U, 4U, (Self::S - 32U) / 8U, 0U});
            state.MS();
        }
    }
