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
    template <class Self, bool ClassValue0, bool ClassValue1, bool ClassValue2, uint32_t ClassValue3>
    __aicore__ static inline void SharedOwnerScalarUpdateStage3(Self &state) {
        state.owned = state.blocks > state.rank ? (state.blocks - 1 - state.rank) / Self::G + 1 : 0;
        if constexpr (ClassValue0) {
            const uint32_t first = state.bandFirst(state.rank), count = state.bandCount(state.rank);
            state.owned = state.blocks > first ? (state.blocks - first < count ? state.blocks - first : count) : 0U;
        }
        state.mat = LocalTensor<float>(
            TPosition::VECCALC, Self::MAT,
            (Self::NMAX + ((ClassValue1 || (Self::BALANCED200 && ClassValue2)) ? Self::GUARD_ROWS : 0U)) * Self::S);
        state.panel =
            LocalTensor<float>(TPosition::VECCALC, Self::PANEL, ClassValue3 * (Self::NMAX + Self::GUARD_ROWS));
        state.fac = LocalTensor<float>(TPosition::VECCALC, Self::FAC, Self::B * Self::NMAX);
        state.brc = LocalTensor<float>(TPosition::VECCALC, Self::BRC, 8 * (Self::NMAX + Self::GUARD_COEFF));
        state.col = LocalTensor<float>(TPosition::VECCALC, Self::COL, Self::NMAX);
        state.ab = LocalTensor<float>(TPosition::VECCALC, Self::ABS, Self::NMAX);
        state.row = LocalTensor<float>(TPosition::VECCALC, Self::ROW, Self::S);
        state.prow = LocalTensor<float>(TPosition::VECCALC, Self::PROW, Self::B);
        state.tmp = LocalTensor<float>(TPosition::VECCALC, Self::TMP, Self::TMP_BYTES / 4U);
        state.tab = LocalTensor<int32_t>(TPosition::VECCALC, Self::TAB, Self::NMAX);
        state.lanes = LocalTensor<int32_t>(TPosition::VECCALC, Self::LANE, 8 * Self::NMAX);
        state.meta = LocalTensor<float>(TPosition::VECCALC, Self::META, Self::META_WORDS);
    }
