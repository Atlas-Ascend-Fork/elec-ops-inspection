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
    template <class OwnerState, uint32_t ClassValue0>
    __aicore__ static inline void RunSetGlobalBufferPhase125SharedPhase31(OwnerState &state,
                                                                          const uint32_t &addressMatrix,
                                                                          const bool &valid, GM_ADDR &workspace) {

        for (uint32_t q = 0U; q < ClassValue0 / OwnerState::B; ++q) {
            const uint32_t kb = q * OwnerState::B, slotWords = (addressMatrix * 5U + q) * OwnerState::GM_WORDS;
            state.ws.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + slotWords);
            state.wu.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace) + slotWords);
            if (valid && state.rank == q) {
                if (!state.singular)
                    state.factor(kb, q);
                else
                    state.publishPrefix16(kb);
                state.publish(kb);
            } else if (valid && state.rank > q) {
                state.waitReady(q, 0x6A1F0010U);
                state.template receiveRange<0U, 16U>(kb);
                if (!state.singular)
                    state.template ApplyPanelUpdates<0U, 16U>(kb, q);
                state.waitReady(q, 0x6A1F0020U);
                state.template receiveRange<16U, 32U>(kb);
                if (!state.singular)
                    state.template ApplyPanelUpdates<16U, 32U>(kb, q);
            }
        }
    }
