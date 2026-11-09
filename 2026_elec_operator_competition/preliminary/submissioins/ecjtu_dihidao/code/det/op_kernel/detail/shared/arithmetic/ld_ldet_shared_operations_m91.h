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
    template <uint32_t SelectorValue, class SelectorQ>
    __aicore__ static inline auto SelectPanelOwner32(SelectorQ &q)
        -> decltype(SelectorValue == 5U ? (q < 2U ? 0U : (q < 5U ? q - 1U : 0U))
                                        : (q < 2U ? 0U : (q == 2U ? 1U : (q == 3U ? 0U : (q == 4U ? 2U : 1U))))) {
        if (SelectorValue == 5U) {
            if (q < 2U) {
                return 0U;
            }
            if (q < 5U) {
                return q - 1U;
            }
            return 0U;
        }
        if (q < 2U) {
            return 0U;
        }
        if (q == 2U) {
            return 1U;
        }
        if (q == 3U) {
            return 0U;
        }
        if (q == 4U) {
            return 2U;
        }
        return 1U;
    }
