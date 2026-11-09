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
    template <bool ClassValue0, class Depth22Arg3_3>
    __aicore__ static inline void
    RunfactorPhase12SharedPhase4LuPollDepth22Stage1(const bool &actualPrefixConsumer, uint32_t &kb, uint32_t &q,
                                                    Depth22Arg3_3 &state, const bool &valid) {

        if (valid && actualPrefixConsumer) {
            LuPoll(state.prefixTicket(q), 0x6A1F0010U);
            state.template receiveRange<0U, 16U>(kb);
            if (!state.singular) {
                if constexpr (!ClassValue0) {
                    const uint32_t priorOwned = state.owned;
                    if (q == 1U || q == 3U)
                        state.owned = 1U;
                    state.template ApplyPanelUpdates<0U, 16U>(kb, q);
                    state.owned = priorOwned;
                } else
                    state.template ApplyPanelUpdates<0U, 16U>(kb, q);
            }
        }
    }
    template <class OwnerState, bool ClassValue0>
    __aicore__ static inline void RunfactorPhase12SharedPhase4(OwnerState &state, const bool &actualPrefixConsumer,
                                                               uint32_t &kb, uint32_t &owner, uint32_t &q,
                                                               const bool &valid) {

        if (valid && state.rank == owner) {
            if (!state.singular)
                state.factor(kb, q);
            else if constexpr (OwnerState::PREFIX_STREAM) {
                if (q)
                    state.publishPrefix16(kb);
            }
            state.publish(kb);
        }
        if constexpr (OwnerState::PREFIX_STREAM) {
            RunfactorPhase12SharedPhase4LuPollDepth22Stage1<ClassValue0>(actualPrefixConsumer, kb, q, state, valid);
        }
        SyncAll<true>();
    }
