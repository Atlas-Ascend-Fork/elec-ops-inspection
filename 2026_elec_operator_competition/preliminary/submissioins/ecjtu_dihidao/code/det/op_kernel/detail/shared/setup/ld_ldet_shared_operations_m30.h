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
    template <class OwnerState, class Capture0>
    __aicore__ static inline void Shared24ScalarUpdateStage13(Capture0 &state) {
        LDET_MASK_SET(state.n);
        for (uint32_t r = 0; r < 8; ++r)
            vadds(reinterpret_cast<__ubuf__ int32_t *>(state.lanes[r * OwnerState::NMAX].GetPhyAddr()),
                  reinterpret_cast<__ubuf__ int32_t *>(state.tab.GetPhyAddr()), int32_t(r * 4), 1, 1, 1, 8, 8);
        LDET_MASK_RST();
    }
    template <class OwnerState, bool ClassValue0, uint32_t ClassValue1, uint32_t ClassValue2, bool ClassValue3,
              bool ClassValue4, bool ClassValue5, bool ClassValue6, bool WAVED>
    __aicore__ static inline void RunLDET_MASK_SETPhase104SharedPhase24(OwnerState &state, const bool &valid) {

        Shared24ScalarUpdateStage13<OwnerState>(state);
        if constexpr (ClassValue0) {
            static_assert(ClassValue1 == 200U && (ClassValue2 == 5U || ClassValue2 == 6U) && ClassValue6 &&
                              ClassValue5 && !WAVED && !ClassValue3 && !ClassValue4,
                          "Owner-only first input scope");
            if (state.rank == 0U) {
                CopyInputAndWait64<true>(state, valid);
            }
            SyncAll<true>();
            if (state.rank != 0U) {
                CopyInputAndWait64<true>(state, valid);
            }
        } else {
            CopyInputAndWait64<!ClassValue3>(state, valid);
        }
    }
