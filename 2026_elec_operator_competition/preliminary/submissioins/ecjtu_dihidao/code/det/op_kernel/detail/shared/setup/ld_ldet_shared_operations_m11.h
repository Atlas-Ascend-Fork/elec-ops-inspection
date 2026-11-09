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
    template <class OwnerState, class Capture0>
    __aicore__ static inline void SharedScalarUpdateStage4(Capture0 &state) {
        state.fac = LocalTensor<float>(TPosition::VECCALC, OwnerState::FAC,
                                       OwnerState::B * OwnerState::NMAX + OwnerState::META_WORDS);
        state.brc =
            LocalTensor<float>(TPosition::VECCALC, OwnerState::BRC, 8 * (OwnerState::NMAX + OwnerState::GUARD_COEFF));
        state.col = LocalTensor<float>(TPosition::VECCALC, OwnerState::COL, OwnerState::NMAX);
        state.ab = LocalTensor<float>(TPosition::VECCALC, OwnerState::ABS, OwnerState::NMAX);
        state.row = LocalTensor<float>(TPosition::VECCALC, OwnerState::ROW, OwnerState::S);
        state.prow = LocalTensor<float>(TPosition::VECCALC, OwnerState::PROW, OwnerState::B);
        state.tmp = LocalTensor<float>(TPosition::VECCALC, OwnerState::TMP, 256);
        state.tab = LocalTensor<int32_t>(TPosition::VECCALC, OwnerState::TAB, OwnerState::NMAX);
        state.lanes = LocalTensor<int32_t>(TPosition::VECCALC, OwnerState::LANE, 8 * OwnerState::NMAX);
        state.meta = LocalTensor<float>(TPosition::VECCALC, OwnerState::META, OwnerState::META_WORDS);
        state.mu = state.meta.template ReinterpretCast<uint32_t>();
    }
    template <class OwnerState, bool ClassValue0, uint32_t ClassValue1, bool ClassValue2, bool ClassValue3,
              bool ClassValue4, bool WAVED, class PhaseArg1>
    __aicore__ static inline void RunGetBlockIdxPhase11SharedPhase3(OwnerState &state, GM_ADDR &input,
                                                                    PhaseArg1 &matrixIndex, GM_ADDR &output,
                                                                    const bool &valid, GM_ADDR &workspace) {

        state.rank = GetBlockIdx() % OwnerState::G;
        state.matrix = WAVED ? matrixIndex : GetBlockIdx() / OwnerState::G;
        state.blocks = state.n / OwnerState::B;
        state.owned = state.blocks > state.rank ? (state.blocks - 1 - state.rank) / OwnerState::G + 1 : 0;
        if constexpr (ClassValue0) {
            const uint32_t first = state.bandFirst(state.rank);
            const uint32_t maxOwned = ClassValue2 ? 3U : 2U;
            state.owned =
                (ClassValue3 && (state.rank == 1U || state.rank == 2U))
                    ? 1U
                    : (state.blocks > first ? (state.blocks - first < maxOwned ? state.blocks - first : maxOwned) : 0U);
        }
        state.mat = LocalTensor<float>(TPosition::VECCALC, OwnerState::MAT, OwnerState::NMAX * OwnerState::S);
        state.panel = LocalTensor<float>(TPosition::VECCALC,
                                         (ClassValue1 == 8U && ClassValue4) ? OwnerState::MAT : OwnerState::PANEL,
                                         OwnerState::P * (OwnerState::NMAX + OwnerState::GUARD_ROWS));
        SharedScalarUpdateStage4<OwnerState>(state);
        const uint32_t addressMatrix = valid ? state.matrix : 0U;
        state.in.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input) + addressMatrix * state.n * state.n);
        state.ws.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) +
                                 addressMatrix * (2U * OwnerState::GM_WORDS));
        state.wu.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace) +
                                 addressMatrix * (2U * OwnerState::GM_WORDS));
        state.out.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output) + addressMatrix);
        state.relay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) +
                                    (GetBlockNum() / OwnerState::G) * (2U * OwnerState::GM_WORDS) +
                                    addressMatrix * (224U * 32U));
        if constexpr (OwnerState::PREFIX_STREAM && !ClassValue3)
            state.prefixBase = workspace +
                               (GetBlockNum() / OwnerState::G) * (2U * OwnerState::GM_WORDS + 224U * 32U) * 4U +
                               addressMatrix * 6U * 64U;
    }
