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
    template <class OwnerState, bool ClassValue0, bool ClassValue1>
    __aicore__ static inline void SharedOwner26ScalarUpdateStage3(OwnerState &ownerState, uint32_t &kb, uint32_t &q,
                                                                  const bool &valid) {
        if constexpr (ClassValue0) {
            if (valid && (ClassValue1 ? ownerState.rank == q + 1U
                                      : ((ownerState.rank == 1U && q == 1U) || (ownerState.rank == 2U && q == 3U)))) {
                LuPoll(ownerState.prefixFlag + (ClassValue1 ? q * 64U : (q == 3U ? 64U : 0U)), 0x6A1F0010U);
                ownerState.template receiveRange<0U, 16U>(kb);
                if (!ownerState.singular)
                    ownerState.template ApplyPanelUpdates<0U, 16U>(kb, q);
            }
        }
    }
