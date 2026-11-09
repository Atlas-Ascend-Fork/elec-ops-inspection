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
    template <class Self, bool ClassValue0>
    __aicore__ static inline void SharedOwnerVmulsStage29(Self &state, const uint32_t &begin, float &inv, uint32_t &k,
                                                          __ubuf__ float *&mp, uint32_t &p, __ubuf__ float *&rp,
                                                          const uint32_t &width) {
        if constexpr (ClassValue0)
            AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
        else
            LDET_MASK_SET(width);
        vmuls(rp, mp + p * Self::S + begin, inv, 1, 1, 1, 8, 8);
        if (p != k)
            vmuls(mp + p * Self::S + begin, mp + k * Self::S + begin, 1.0f, 1, 1, 1, 8, 8);
        if constexpr (ClassValue0)
            ResetMask();
        else
            LDET_MASK_RST();
    }
