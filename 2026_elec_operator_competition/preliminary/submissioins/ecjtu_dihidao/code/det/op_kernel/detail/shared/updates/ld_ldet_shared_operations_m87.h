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
    template <class OwnerState, bool ClassValue0>
    __aicore__ static inline void SharedOwner26ScalarUpdateStage8(OwnerState &ownerState,
                                                                  const bool &actualPrefixConsumer, uint32_t &kb,
                                                                  uint32_t &owner, uint32_t &q, const bool &valid) {
        if (valid && ownerState.rank != owner &&
            (ClassValue0 ? ownerState.bandFirst(ownerState.rank) + ownerState.owned - 1U
                         : ownerState.rank + (ownerState.owned - 1U) * OwnerState::G) > q) {
            if constexpr (OwnerState::PREFIX_STREAM) {
                if (actualPrefixConsumer)
                    ownerState.template receiveRange<16U, 32U>(kb);
                else
                    ownerState.receive(kb);
            } else
                ownerState.receive(kb);
        }
    }
