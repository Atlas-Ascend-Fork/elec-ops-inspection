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
    template <class Self, uint32_t FIRST = 0U, uint32_t LAST = Self::B>
    __aicore__ static inline void ApplyPanelUpdatesShared69(Self &state, uint32_t kb, uint32_t q) {
        if (state.rank == Self::G - 1U)
            state.template ApplyPairs<FIRST, LAST, true>(kb, q);
        else
            state.template ApplyPairs<FIRST, LAST>(kb, q);
    }
