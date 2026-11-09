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
    template <class OwnerState>
    __aicore__ static inline void SharedOwner26ScalarUpdateStage4(OwnerState &ownerState, uint32_t &h, uint32_t &k,
                                                                  float &maxv, uint32_t &p, __ubuf__ float *&tp,
                                                                  float &v) {
        if (h > 128U) {
            v = tp[8];
            if (v > maxv) {
                maxv = v;
                p = k + 128U + ownerState.word(tp[9]);
            }
        }
        if (h > 192U) {
            v = tp[10];
            if (v > maxv) {
                maxv = v;
                p = k + 192U + ownerState.word(tp[11]);
            }
        }
    }
