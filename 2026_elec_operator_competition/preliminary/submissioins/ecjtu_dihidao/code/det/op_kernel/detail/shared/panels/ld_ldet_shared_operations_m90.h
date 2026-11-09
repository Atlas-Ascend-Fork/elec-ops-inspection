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
    template <bool SelectorValue, class SelectorQ>
    __aicore__ static inline auto SelectPrefixConsumer32(SelectorQ &q)
        -> decltype(SelectorValue ? (q == 4U ? 0U : (q == 5U ? 4U : q))
                                  : (q == 1U ? 1U : (q == 2U ? 0U : (q == 3U ? 2U : (q == 4U ? 1U : 3U))))) {
        if (SelectorValue) {
            if (q == 4U) {
                return 0U;
            }
            if (q == 5U) {
                return 4U;
            }
            return q;
        }
        if (q == 1U) {
            return 1U;
        }
        if (q == 2U) {
            return 0U;
        }
        if (q == 3U) {
            return 2U;
        }
        if (q == 4U) {
            return 1U;
        }
        return 3U;
    }
