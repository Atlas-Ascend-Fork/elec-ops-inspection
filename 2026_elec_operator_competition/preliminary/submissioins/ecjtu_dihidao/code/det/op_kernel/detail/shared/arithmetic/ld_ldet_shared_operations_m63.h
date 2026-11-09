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
    template <class Self, bool ClassValue0, bool ClassValue1, bool EXPORT>
    __aicore__ static inline void SharedOwnerVmulsStage26(Self &state, uint32_t &f, float &inv, uint32_t &k,
                                                          uint32_t &p, __ubuf__ float *&pp, __ubuf__ float *&rp) {
        if (p != k)
            state.sign = -state.sign;
        if constexpr (EXPORT && !ClassValue0) {
            state.mu.SetValue(8 + f, p);
            state.mu.SetValue(8U + Self::B + f, state.word(inv));
        }
        if constexpr (ClassValue1)
            AscendCUtils::SetMask<float>(static_cast<int32_t>(Self::B));
        else
            LDET_MASK_SET(Self::B);
        vmuls(rp, pp + p * Self::P, inv, 1, 1, 1, 8, 8);
        if (p != k)
            vmuls(pp + p * Self::P, pp + k * Self::P, 1.0f, 1, 1, 1, 8, 8);
    }
