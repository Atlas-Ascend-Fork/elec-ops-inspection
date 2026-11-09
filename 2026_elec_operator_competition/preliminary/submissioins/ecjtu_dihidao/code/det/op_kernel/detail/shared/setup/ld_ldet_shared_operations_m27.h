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
    template <class OwnerState, bool ClassValue0, bool ClassValue1, bool ONE_INPUT_DMA>
    __aicore__ static inline void RunDataCopyPhase73SharedPhase20(OwnerState &state, const bool &firstInputOwner,
                                                                  const bool &splitNextInput, const bool &valid) {

        LDET_MASK_RST();
        if (firstInputOwner) {
            state.MV();
            state.VS();
        } else {
            state.VS();
        }
        if (state.rank == 0U)
            SyncAll<true>();
        if (!firstInputOwner) {
            if constexpr (ONE_INPUT_DMA) {
                const uint32_t inputWidth = ClassValue1 ? state.owned * OwnerState::B : 2U * OwnerState::B;
                if (valid) {
                    if (splitNextInput)
                        DataCopy(state.mat[8U], state.in[state.bandFirst(state.rank) * OwnerState::B + 8U],
                                 {static_cast<uint16_t>(state.n), 3U, static_cast<uint16_t>((state.n - 24U) / 8U),
                                  static_cast<uint16_t>((OwnerState::S - 24U) / 8U)});
                    else
                        DataCopy(state.mat, state.in[state.bandFirst(state.rank) * OwnerState::B],
                                 {static_cast<uint16_t>(state.n), static_cast<uint16_t>(inputWidth / 8U),
                                  static_cast<uint16_t>((state.n - inputWidth) / 8U),
                                  static_cast<uint16_t>((OwnerState::S - inputWidth) / 8U)});
                }
            } else {
                SharedDataCopyStage43<ClassValue0, OwnerState>(state, valid);
            }
            state.MV();
            state.VS();
        }
    }
