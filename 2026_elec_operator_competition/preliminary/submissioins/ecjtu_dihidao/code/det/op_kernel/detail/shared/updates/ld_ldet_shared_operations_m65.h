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
    template <class Self, bool ClassValue0, bool ClassValue1>
    __aicore__ static inline void SharedOwnerScalarUpdateStage28(Self &state, const uint32_t &addressMatrix,
                                                                 uint32_t &kb, uint32_t &owner, uint32_t &q,
                                                                 const bool &valid, GM_ADDR &workspace) {
        const uint32_t slotWords = (addressMatrix * 2U + (q & 1U)) * Self::GM_WORDS;
        state.ws.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + slotWords);
        state.wu.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace) + slotWords);
        if (valid && state.rank == owner) {
            if (!state.singular)
                state.factor(kb, q);
            else if constexpr (ClassValue0) {
                if (ClassValue1 || q == 1U || q == 3U)
                    state.publishPrefix16(kb);
            }
            state.publish(kb);
        }
    }
