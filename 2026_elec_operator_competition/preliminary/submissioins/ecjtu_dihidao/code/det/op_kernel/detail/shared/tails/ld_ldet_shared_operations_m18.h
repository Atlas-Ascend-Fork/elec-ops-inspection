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
    template <class OwnerState, auto T, auto TS>
    __aicore__ static inline void
    Finish64vgatherPhase4SharedPhase10(OwnerState &state, __ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f,
                                       uint32_t &h, __ubuf__ uint32_t *&ip, __ubuf__ float *&mp, bool &prepared) {
        return Finish64Shared21vgatherPhase4<T, TS, OwnerState>(ap, cp, f, h, ip, mp, prepared, state);
    }
