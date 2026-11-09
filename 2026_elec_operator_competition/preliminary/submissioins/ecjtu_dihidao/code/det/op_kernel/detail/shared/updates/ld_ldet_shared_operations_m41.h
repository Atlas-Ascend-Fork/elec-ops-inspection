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
    template <class Capture0> __aicore__ static inline void Shared14ScalarUpdateStage1(Capture0 &state) {
        state.fac =
            LocalTensor<float>(TPosition::VECCALC, Capture0::FAC, Capture0::B * Capture0::NMAX + Capture0::META_WORDS);
        state.brc = LocalTensor<float>(TPosition::VECCALC, Capture0::BRC, 8 * (Capture0::NMAX + Capture0::GUARD_COEFF));
        state.col = LocalTensor<float>(TPosition::VECCALC, Capture0::COL, Capture0::NMAX);
        state.ab = LocalTensor<float>(TPosition::VECCALC, Capture0::ABS, Capture0::NMAX);
        state.row = LocalTensor<float>(TPosition::VECCALC, Capture0::ROW, Capture0::S);
        state.prow = LocalTensor<float>(TPosition::VECCALC, Capture0::PROW, Capture0::B);
        state.tmp = LocalTensor<float>(TPosition::VECCALC, Capture0::TMP, 256);
        state.tab = LocalTensor<int32_t>(TPosition::VECCALC, Capture0::TAB, Capture0::NMAX);
        state.lanes = LocalTensor<int32_t>(TPosition::VECCALC, Capture0::LANE, 8 * Capture0::NMAX);
    }
    template <class Self, uint32_t ClassValue0, bool ClassValue1>
    __aicore__ static inline void SharedOwnerScalarUpdateStage2(Self &state) {
        state.mat = LocalTensor<float>(TPosition::VECCALC, Self::MAT, Self::NMAX * Self::S);
        state.panel =
            LocalTensor<float>(TPosition::VECCALC, (ClassValue0 == 8U && ClassValue1) ? Self::MAT : Self::PANEL,
                               Self::P * (Self::NMAX + Self::GUARD_ROWS));
        Shared14ScalarUpdateStage1(state);
        state.meta = LocalTensor<float>(TPosition::VECCALC, Self::META, Self::META_WORDS);
        state.mu = state.meta.template ReinterpretCast<uint32_t>();
    }
