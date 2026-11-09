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
    template <class OwnerState, bool ClassValue0, bool ClassValue2, bool ClassValue4, class Depth22Arg6_3>
    __aicore__ static inline void
    RunPipeBarrierPhase106SharedPhase25constexprDepth22Stage1(uint32_t &kb, uint32_t &owner, uint32_t &q,
                                                              Depth22Arg6_3 &state, const bool &valid) {

        if (valid && state.rank != owner && (!ClassValue0 || state.rank + (state.owned - 1U) * OwnerState::G > q)) {
            if constexpr (ClassValue2) {
                if (ClassValue4 ? state.rank == q + 1U
                                : ((state.rank == 1U && q == 1U) || (state.rank == 2U && q == 3U)))
                    state.template receiveRange<16U, 32U>(kb);
                else
                    state.receive(kb);
            } else
                state.receive(kb);
        }
    }
    template <class OwnerState, bool ClassValue0, bool ClassValue1, bool ClassValue2, bool ClassValue3,
              bool ClassValue4>
    __aicore__ static inline void RunPipeBarrierPhase106SharedPhase25(OwnerState &state, uint32_t &kb, uint32_t &owner,
                                                                      uint32_t &q, const bool &valid) {

        if constexpr (ClassValue1) {
            PipeBarrier<PIPE_ALL>();
            CrossCoreSetFlag<1, PIPE_MTE3>(14U);
            CrossCoreWaitFlag<1>(14U);
        } else
            SyncAll<true>();
        if constexpr (ClassValue3) {
            static_assert(OwnerState::G == (OwnerState::NMAX + OwnerState::B - 1U) / OwnerState::B,
                          "receive retirement requires exactly one column band per core");
            if (valid && state.rank > q)
                state.receive(kb);
        } else {
            RunPipeBarrierPhase106SharedPhase25constexprDepth22Stage1<OwnerState, ClassValue0, ClassValue2,
                                                                      ClassValue4>(kb, owner, q, state, valid);
        }
    }
