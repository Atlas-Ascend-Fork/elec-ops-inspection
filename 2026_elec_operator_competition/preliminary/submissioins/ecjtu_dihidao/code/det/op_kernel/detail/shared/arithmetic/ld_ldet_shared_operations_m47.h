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
    template <class Self>
    __aicore__ static inline void SharedOwnerVmulsStage8(Self &state, const uint32_t &begin, uint32_t &f, uint32_t &h,
                                                         float &inv, uint32_t &k, uint32_t &kb, __ubuf__ float *&mp,
                                                         uint32_t &p, __ubuf__ float *&rp, const uint32_t &width) {
        Shared36VmulsStage1<Self>(begin, f, h, inv, k, kb, mp, p, rp, state, width);
    }
