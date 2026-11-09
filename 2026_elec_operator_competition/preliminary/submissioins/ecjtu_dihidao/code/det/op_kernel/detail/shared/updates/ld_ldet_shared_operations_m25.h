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
    template <bool ClassValue0, class OwnerState, class Capture4>
    __aicore__ static inline void SharedScalarUpdateStage32(uint32_t &kb, uint32_t &owner, const bool &prefixConsumer,
                                                            uint32_t &q, Capture4 &state, const bool &valid) {
        if (valid && state.rank != owner &&
            (ClassValue0 ? state.bandFirst(state.rank) + state.owned - 1U
                         : state.rank + (state.owned - 1U) * OwnerState::G) > q) {
            if constexpr (OwnerState::PREFIX_STREAM) {
                if (prefixConsumer)
                    state.template receiveRange<16U, 32U>(kb);
                else
                    state.receive(kb);
            } else
                state.receive(kb);
        }
        if (valid && !state.singular) {
            if constexpr (OwnerState::PREFIX_STREAM) {
                if (prefixConsumer)
                    state.template ApplyPanelUpdates<16U, 32U>(kb, q);
                else
                    state.ApplyPanelUpdates(kb, q);
            } else
                state.ApplyPanelUpdates(kb, q);
        }
    }
    template <class OwnerState, bool ClassValue0>
    __aicore__ static inline void RunconstexprPhase63SharedPhase17(OwnerState &state, uint32_t &kb, uint32_t &owner,
                                                                   const bool &prefixConsumer, uint32_t &q,
                                                                   const bool &valid) {

        if constexpr (OwnerState::PREFIX_STREAM) {
            if (valid && prefixConsumer) {
                LuPollN256Prefix(state.prefixTicket(q), 0x6A1F0010U);
                state.template receiveRange<0U, 16U>(kb);
                if (!state.singular)
                    state.template ApplyPanelUpdates<0U, 16U>(kb, q);
            }
        }
        SyncAll<true>();
        SharedScalarUpdateStage32<ClassValue0, OwnerState>(kb, owner, prefixConsumer, q, state, valid);
    }
