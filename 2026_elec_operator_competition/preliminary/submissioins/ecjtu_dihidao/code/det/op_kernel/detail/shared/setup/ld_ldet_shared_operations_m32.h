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
    template <class OwnerState, bool ClassValue0, bool ClassValue1, bool ClassValue2, bool ClassValue3,
              uint32_t ClassValue4, bool WAVED, class PhaseArg0, class PhaseArg1, class PhaseArg2>
    __aicore__ static inline void RunGetBlockIdxPhase107SharedPhase26(OwnerState &state, PhaseArg0 &matrixIndex,
                                                                      PhaseArg1 &originalDim,
                                                                      PhaseArg2 &physicalGroup) {

        state.inputDim = ClassValue3 ? originalDim : OwnerState::NMAX;
        state.n = OwnerState::NMAX;
        state.rank = (GetBlockIdx() % physicalGroup) / 2U + ((GetBlockIdx() % physicalGroup) & 1U) * 3U;
        state.matrix = WAVED ? matrixIndex : GetBlockIdx() / physicalGroup;
        state.blocks = (state.n + OwnerState::B - 1U) / OwnerState::B;
        state.owned = state.blocks > state.rank ? (state.blocks - 1 - state.rank) / OwnerState::G + 1 : 0;
        if constexpr (ClassValue0) {
            const uint32_t first = state.bandFirst(state.rank), count = state.bandCount(state.rank);
            state.owned = state.blocks > first ? (state.blocks - first < count ? state.blocks - first : count) : 0U;
        }
        state.mat = LocalTensor<float>(
            TPosition::VECCALC, OwnerState::MAT,
            (OwnerState::NMAX +
             ((ClassValue1 || (OwnerState::BALANCED200 && ClassValue2)) ? OwnerState::GUARD_ROWS : 0U)) *
                OwnerState::S);
        state.panel = LocalTensor<float>(TPosition::VECCALC, OwnerState::PANEL,
                                         ClassValue4 * (OwnerState::NMAX + OwnerState::GUARD_ROWS));
        state.fac = LocalTensor<float>(TPosition::VECCALC, OwnerState::FAC, OwnerState::B * OwnerState::NMAX);
        state.brc =
            LocalTensor<float>(TPosition::VECCALC, OwnerState::BRC, 8 * (OwnerState::NMAX + OwnerState::GUARD_COEFF));
        state.col = LocalTensor<float>(TPosition::VECCALC, OwnerState::COL, OwnerState::NMAX);
        state.ab = LocalTensor<float>(TPosition::VECCALC, OwnerState::ABS, OwnerState::NMAX);
        state.row = LocalTensor<float>(TPosition::VECCALC, OwnerState::ROW, OwnerState::S);
        state.prow = LocalTensor<float>(TPosition::VECCALC, OwnerState::PROW, OwnerState::B);
        state.tmp = LocalTensor<float>(TPosition::VECCALC, OwnerState::TMP, OwnerState::TMP_BYTES / 4U);
        state.tab = LocalTensor<int32_t>(TPosition::VECCALC, OwnerState::TAB, OwnerState::NMAX);
        state.lanes = LocalTensor<int32_t>(TPosition::VECCALC, OwnerState::LANE, 8 * OwnerState::NMAX);
        state.meta = LocalTensor<float>(TPosition::VECCALC, OwnerState::META, OwnerState::META_WORDS);
        state.mu = state.meta.template ReinterpretCast<uint32_t>();
    }
