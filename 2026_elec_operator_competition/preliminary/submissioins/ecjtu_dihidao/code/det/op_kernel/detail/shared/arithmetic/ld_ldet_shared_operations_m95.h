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
    template <class State, class Matrix>
    __aicore__ static inline TailRawViews51 GetTailRawViews51(State &state, Matrix &tm) {
        return {reinterpret_cast<__ubuf__ float *>(tm.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(state.col.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(state.ab.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(state.row.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(state.tmp.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(state.lanes.GetPhyAddr())};
    }

    template <class State, bool VIRTUAL>
    __aicore__ static inline bool SkipInactivePhysicalGroup51(State &state, uint32_t physicalGroup) {
        if constexpr (State::LdetPolicy::DIM == 200U && State::LdetPolicy::GROUP == 3U && State::LdetPolicy::COMPACT &&
                      State::LdetPolicy::LOCAL_TAIL && State::LdetPolicy::LOCAL_CUT == 128U) {
            if (physicalGroup > State::G) {
                uint32_t block;
                if constexpr (VIRTUAL)
                    block = state.ReadPhysicalIndex51();
                else
                    block = GetBlockIdx();
                if (block % physicalGroup >= State::G) {
                    for (uint32_t q = 0U; q < State::LdetPolicy::LOCAL_CUT / State::B; ++q)
                        SyncAll<true>();
                    return true;
                }
            }
        }
        return false;
    }

    template <class State, bool CONTIGUOUS>
    __aicore__ static inline uint32_t FirstOwnedColumn54(State &state, uint32_t q) {
        uint32_t first = q < state.rank ? 0 : (q - state.rank) / State::G + 1;
        if constexpr (CONTIGUOUS) {
            const uint32_t base = state.bandFirst(state.rank);
            first = q < base ? 0U : q - base + 1U;
        }
        return first;
    }

    template <bool WAVED, bool SPLIT, uint32_t BANDS, class OwnerState>
    __aicore__ static inline void PreparePrefixTicket56(OwnerState &state, GM_ADDR workspace, uint32_t addressMatrix,
                                                        bool valid) {
        static_assert(!WAVED, "prefix ticket freshness relies on the original first ready join");
        state.prefixFlag = workspace + ((3U * BANDS * OwnerState::GM_WORDS * 4U + 63U) & ~63U) +
                           addressMatrix * (SPLIT ? (OwnerState::G - 1U) * 64U : 128U);
        if (valid && state.rank == 0U) {
            for (uint32_t i = 0U; i < (SPLIT ? OwnerState::G - 1U : 2U); ++i)
                LuPublish(state.prefixFlag + i * 64U, 0U);
        }
    }

    template <class State> __aicore__ static inline void RecycleFiveWay56(State &state, uint32_t q) {
        if (q == 2U) {
            state.recycleState = 1U;
            if (state.rank == 0U) {
                state.owned = 1U;
                if (!state.singular) {
                    DataCopy(state.mat[64U * State::S], state.relay, {192U, 4U, 0U, (State::S - 32U) / 8U});
                    state.MV();
                }
            } else if (state.rank == 3U)
                state.owned = 1U;
        }
    }

    __aicore__ static inline float ReciprocalWithZero58(float divisor, float numerator) {
        if (divisor == 0.0f)
            return 0.0f;
        return numerator / divisor;
    }

    template <class State>
    __aicore__ static inline bool PrepareTailPivot60(State &state, __ubuf__ float *tp, __ubuf__ float *cp, uint32_t h,
                                                     uint32_t f, uint32_t &p, float &piv, float &inv) {
        const uint32_t ri = state.word(tp[1]);
        if (ri >= h) {
            state.singular = 1;
            return false;
        }
        p = f + ri;
        piv = cp[ri];
        if (piv == 0.0f || ((state.word(piv) >> 23U) & 255U) == 255U) {
            state.acc(piv);
            return false;
        }
        inv = -1.0f / piv;
        if (((state.word(inv) >> 23U) & 255U) == 255U) {
            state.acc(piv);
            state.singular = 2;
            return false;
        }
        return true;
    }
