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
    template <class OwnerState, bool ClassValue0, uint32_t ClassValue1, uint32_t ClassValue2, uint32_t ClassValue3,
              bool ClassValue4>
    __aicore__ static inline void Runstatic_assertPhase108SharedPhase27(OwnerState &state, const bool &valid) {

        static_assert(
            ClassValue0 &&
                (((ClassValue2 == 3U || ClassValue4) && (ClassValue3 == 128U || (ClassValue4 && ClassValue3 == 160U)) &&
                  (ClassValue1 == 192U || ClassValue1 == 200U || OwnerState::NATIVE208)) ||
                 (ClassValue2 == 2U && ((ClassValue1 == 192U && ClassValue3 == 96U) || OwnerState::PAIR200))),
            "local tail route scope");
        if (valid && state.rank == OwnerState::G - 1U)
            state.FinishLocalTail();
        return;
    }
