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
// Included in the CyclicRowLUMedium class; preserves its member scope and access.
    __aicore__ inline uint32_t bandFirst(uint32_t r) const {
        return LdetSharedOperations::bandFirstShared41<CyclicRowLUMedium>(*this, r);
    }
    __aicore__ inline uint32_t bandCount(uint32_t r) const {
        if constexpr (SPLIT_OWNERS)
            return r == 4U ? 3U : 1U;
        else if constexpr (BALANCED200) {
            if constexpr (LOCAL_TAIL)
                return r == 2U ? 3U : 2U;
            else
                return r == 0U ? 3U : 2U;
        } else
            return OWNER_BANDS;
    }
    __aicore__ inline uint32_t ownerOf(uint32_t q) const {
        if constexpr (SPLIT_OWNERS)
            return q < 4U ? q : 4U;
        else if constexpr (BALANCED200) {
            if constexpr (LOCAL_TAIL)
                return q < 2U ? 0U : (q < 4U ? 1U : 2U);
            else
                return q < 3U ? 0U : (q < 5U ? 1U : 2U);
        } else
            return q / OWNER_BANDS;
    }
