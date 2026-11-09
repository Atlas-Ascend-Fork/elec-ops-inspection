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
    __aicore__ static inline void SharedOwner26ScalarUpdateStage5(OwnerState &ownerState, uint32_t &kb, uint32_t &q) {
        if constexpr (ClassValue0) {
            if (ClassValue1 ? ownerState.rank == q + 1U
                            : ((ownerState.rank == 1U && q == 1U) || (ownerState.rank == 2U && q == 3U)))
                ownerState.template ApplyPanelUpdates<16U, 32U>(kb, q);
            else
                ownerState.ApplyPanelUpdates(kb, q);
        } else
            ownerState.ApplyPanelUpdates(kb, q);
    }
