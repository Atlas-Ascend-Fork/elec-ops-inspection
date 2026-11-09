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
    template <class Self, bool ClassValue0, bool ClassValue1, bool ClassValue2, bool ClassValue3>
    __aicore__ static inline void SharedOwner15DuplicateStage5(Self &state) {
        Duplicate(state.fac, 0.0f, Self::B * Self::NMAX);
        Duplicate(state.col, 0.0f, Self::NMAX);
        if constexpr (Self::GUARDED) {
            Duplicate(state.panel[Self::NMAX * Self::P], 0.0f, Self::GUARD_ROWS * Self::P);
            Duplicate(state.brc, 0.0f, 8 * (Self::NMAX + Self::GUARD_COEFF));
        }
        if constexpr (!ClassValue3) {
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(4);
            WaitFlag<HardEvent::V_MTE2>(4);
        } else {
            static_assert((Self::G == 4U || ClassValue2 || ClassValue1) && Self::S == (ClassValue1 ? 96U : 64U) &&
                              Self::B == 32U && ClassValue0,
                          "validownedcolumnsfullyoverwritten");
        }
    }
