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
    template <class OwnerState, bool ClassValue1, bool ClassValue3, class Depth22Arg4_2>
    __aicore__ static inline void factorDataCopyPhase18SharedPhase11DataCopyDepth22Stage1(uint32_t &kb, uint32_t &q,
                                                                                          Depth22Arg4_2 &state) {

        DataCopy(state.panel[kb * OwnerState::P],
                 state.mat[kb * OwnerState::S + (ClassValue1 ? (ClassValue3 ? q - state.bandFirst(state.rank) : q % 2U)
                                                             : q / OwnerState::G) *
                                                    OwnerState::B],
                 {static_cast<uint16_t>(state.n - kb), OwnerState::B / 8U, (OwnerState::S - OwnerState::B) / 8U,
                  (OwnerState::P - OwnerState::B) / 8U});
    }
    template <class OwnerState, bool ClassValue1, bool ClassValue3, class Depth22Arg5_1>
    __aicore__ static inline void factorDataCopyPhase18SharedPhase11DataCopyDepth22Stage2(uint32_t &q,
                                                                                          Depth22Arg5_1 &state) {

        DataCopy(
            state.panel,
            state.mat[(ClassValue1 ? (ClassValue3 ? q - state.bandFirst(state.rank) : q % 2U) : q / OwnerState::G) *
                      OwnerState::B],
            {static_cast<uint16_t>(state.n), OwnerState::B / 8U, (OwnerState::S - OwnerState::B) / 8U,
             (OwnerState::P - OwnerState::B) / 8U});
    }
    template <class OwnerState, bool ClassValue0, bool ClassValue1, uint32_t ClassValue2, bool ClassValue3,
              bool ClassValue4, bool ClassValue5>
    __aicore__ static inline void factorDataCopyPhase18SharedPhase11(OwnerState &state, uint32_t &kb, uint32_t &q) {

        state.bindPayloadMeta(kb);
        if constexpr (ClassValue5) {
            static_assert(OwnerState::S == OwnerState::P && OwnerState::S == 40U && OwnerState::B == 32U,
                          "singleownedpanelalreadycurrentinMAT");
        } else if constexpr (ClassValue2 == 8U && ClassValue4) {
            static_assert(OwnerState::B == 32U && OwnerState::S == OwnerState::P,
                          "B4 G8 owns exactly one in-place panel");
        } else if constexpr (ClassValue0) {
            factorDataCopyPhase18SharedPhase11DataCopyDepth22Stage1<OwnerState, ClassValue1, ClassValue3>(kb, q, state);
        } else {
            factorDataCopyPhase18SharedPhase11DataCopyDepth22Stage2<OwnerState, ClassValue1, ClassValue3>(q, state);
        }
        PipeBarrier<PIPE_V>();
    }
