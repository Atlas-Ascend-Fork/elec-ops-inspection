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
    template <uint32_t ClassValue0, bool ClassValue1, bool ClassValue2, bool ClassValue4, uint32_t ClassValue5,
              class OwnerState, class Capture0>
    __aicore__ static inline void SharedDuplicateStage15(Capture0 &state) {
        state.sign = 1.0f;
        state.prod = 1.0f;
        state.ex = 0;
        state.singular = 0;
        if constexpr (!ClassValue1)
            Duplicate(state.mat, 0.0f, OwnerState::NMAX * OwnerState::S);
        Duplicate(state.fac, 0.0f, OwnerState::B * OwnerState::NMAX);
        Duplicate(state.col, 0.0f, OwnerState::NMAX);
        if constexpr ((ClassValue0 == 192U || ClassValue0 == 200U || ClassValue0 == 208U)) {
            Duplicate(state.panel[OwnerState::NMAX * ClassValue5], 0.0f, OwnerState::GUARD_ROWS * ClassValue5);
            if constexpr (ClassValue2 || (OwnerState::BALANCED200 && ClassValue4))
                Duplicate(state.mat[OwnerState::NMAX * OwnerState::S], 0.0f, OwnerState::GUARD_ROWS * OwnerState::S);
            Duplicate(state.brc, 0.0f, 8 * (OwnerState::NMAX + OwnerState::GUARD_COEFF));
        }
        PipeBarrier<PIPE_V>();
    }
    template <class OwnerState, uint32_t ClassValue0, bool ClassValue1, bool ClassValue2, bool ClassValue3,
              bool ClassValue4, uint32_t ClassValue5, bool ClassValue6, bool ClassValue7, bool WAVED>
    __aicore__ static inline void RunDuplicatePhase103SharedPhase23(OwnerState &state, const uint32_t &addressMatrix,
                                                                    GM_ADDR &input, GM_ADDR &output, const bool &valid,
                                                                    GM_ADDR &workspace) {

        state.in.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input) +
                                 static_cast<uint64_t>(addressMatrix) * state.inputDim * state.inputDim);
        state.ws.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) +
                                 addressMatrix * (2U * OwnerState::GM_WORDS));
        state.wu.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace) +
                                 addressMatrix * (2U * OwnerState::GM_WORDS));
        state.out.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output) + addressMatrix);
        if constexpr (ClassValue6) {
            PreparePrefixTicket56<WAVED, ClassValue7, 2U>(state, workspace, addressMatrix, valid);
        }
        SharedDuplicateStage15<ClassValue0, ClassValue1, ClassValue2, ClassValue4, ClassValue5, OwnerState>(state);
        if constexpr (ClassValue3)
            state.copyInput(valid);
    }
