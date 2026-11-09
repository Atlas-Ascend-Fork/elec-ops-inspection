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
    template <bool ClassValue0, class OwnerState, class Capture0>
    __aicore__ static inline void SharedDataCopyStage43(Capture0 &state, const bool &valid) {
        if (valid)
            for (uint32_t j = 0; j < state.owned; ++j)
                DataCopy(state.mat[j * OwnerState::B],
                         state.in[(ClassValue0 ? state.bandFirst(state.rank) + j : state.rank + j * OwnerState::G) *
                                  OwnerState::B],
                         {static_cast<uint16_t>(state.n), OwnerState::B / 8U,
                          static_cast<uint16_t>((state.n - OwnerState::B) / 8), (OwnerState::S - OwnerState::B) / 8U});
    }
    template <class OwnerState, bool ClassValue0, bool ClassValue1, bool ClassValue2, bool ONE_INPUT_DMA>
    __aicore__ static inline void RunDataCopyPhase70SharedPhase19(OwnerState &state, const bool &firstInputOwner,
                                                                  const bool &splitNextInput, const bool &valid) {

        if (splitNextInput) {
            if (valid)
                DataCopy(state.mat, state.in[state.bandFirst(state.rank) * OwnerState::B],
                         {static_cast<uint16_t>(state.n), 1U, static_cast<uint16_t>((state.n - 8U) / 8U),
                          static_cast<uint16_t>((OwnerState::S - 8U) / 8U)});
        }
        if (firstInputOwner) {
            if constexpr (ONE_INPUT_DMA) {
                static_assert(ClassValue0 && (OwnerState::G == 4U || ClassValue1) && OwnerState::B == 32U &&
                                  (OwnerState::S == 64U || ClassValue2),
                              "single input DMA scope");
                const uint32_t inputWidth = ClassValue1 ? state.owned * OwnerState::B : 2U * OwnerState::B;
                if (valid)
                    DataCopy(state.mat, state.in[state.bandFirst(state.rank) * OwnerState::B],
                             {static_cast<uint16_t>(state.n), static_cast<uint16_t>(inputWidth / 8U),
                              static_cast<uint16_t>((state.n - inputWidth) / 8U),
                              static_cast<uint16_t>((OwnerState::S - inputWidth) / 8U)});
            } else {
                SharedDataCopyStage43<ClassValue0, OwnerState>(state, valid);
            }
        }
        SharedPipeBarrierStage29<OwnerState>(state);
    }
