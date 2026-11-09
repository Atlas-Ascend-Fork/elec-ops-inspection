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
    template <class Self, bool ClassValue0, bool ClassValue1, bool ClassValue2, uint32_t ClassValue3, bool ClassValue4,
              bool ClassValue5, bool ClassValue6, bool ClassValue7>
    __aicore__ static inline void SharedOwnerScalarUpdateStage27(Self &state) {
        if constexpr (ClassValue6) {
            static_assert(ClassValue3 == 208U && !ClassValue2 && !ClassValue4 && !ClassValue5,
                          "identity padding scope");
            for (uint32_t j = 0; j < state.owned; ++j) {
                const uint32_t first = (state.rank + j * Self::G) * Self::B;
                for (uint32_t r = state.inputDim; r < Self::NMAX; ++r)
                    if (r >= first && r < first + Self::B)
                        state.mat.SetValue(r * Self::S + j * Self::B + r - first, 1.0f);
            }
            state.SV();
        }
        if constexpr ((!ClassValue1 || ClassValue7) && !ClassValue0)
            SyncAll<true>();
    }
