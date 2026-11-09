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
    __aicore__ static inline void SharedOwner26VmlaStage6(OwnerState &ownerState, uint32_t &f, const uint32_t &first,
                                                          uint32_t &h, __ubuf__ float *&mp, __ubuf__ float *&rp) {
        return Shared24VmlaStage21<T, TS, OwnerState>(f, first, h, mp, rp, ownerState);
    }
