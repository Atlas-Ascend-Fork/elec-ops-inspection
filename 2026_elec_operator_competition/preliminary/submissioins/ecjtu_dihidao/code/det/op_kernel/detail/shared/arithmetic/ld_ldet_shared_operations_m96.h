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
    template <class State> __aicore__ static inline FactorRawViews60 GetFactorRawViews60(State &state) {
        return {reinterpret_cast<__ubuf__ float *>(state.col.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(state.ab.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(state.panel.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(state.prow.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(state.tmp.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(state.lanes.GetPhyAddr())};
    }

    template <auto T, auto TS, class State>
    __aicore__ static inline void PrepareNextTailColumn60(State &state, uint32_t nf, uint32_t nh, __ubuf__ float *ap,
                                                          __ubuf__ float *cp, __ubuf__ uint32_t *ip,
                                                          __ubuf__ float *mp) {
        LDET_MASK_SET(nh);
        vgather(reinterpret_cast<__ubuf__ uint32_t *>(cp), ip + (nf & 7U) * T,
                reinterpret_cast<uint64_t>(mp) + (nf * TS + (nf & ~7U)) * 4U, 8, 1);
        PipeBarrier<PIPE_V>();
        vabs(ap, cp, 1, 1, 1, 8, 8);
        LDET_MASK_RST();
        LDET_MASK_SET(nh);
        LdetVcmax128(state.tmp, state.ab, nh);
        LDET_MASK_RST();
        SetFlag<HardEvent::V_S>(2);
    }

    template <class State, bool CONTIGUOUS> __aicore__ static inline void CopyOwnedBands60(State &state, bool valid) {
        if (valid)
            for (uint32_t j = 0; j < state.owned; ++j)
                DataCopy(
                    state.mat[j * State::B],
                    state.in[(CONTIGUOUS ? state.bandFirst(state.rank) + j : state.rank + j * State::G) * State::B],
                    {static_cast<uint16_t>(state.n), State::B / 8U, static_cast<uint16_t>((state.n - State::B) / 8),
                     (State::S - State::B) / 8U});
    }

    template <class State>
    __aicore__ static inline float ReadFirstPivotSegments60(State &state, uint32_t h, uint32_t k, float &maxv,
                                                            uint32_t &p, __ubuf__ float *tp) {
        float v = ReadPivotSegment64(state, tp, 0U, k, maxv, p);
        if (h > 64U) {
            v = ReadPivotSegment64(state, tp, 2U, k + 64U, maxv, p);
        }
        return v;
    }

    template <class State, bool PREFIX>
    __aicore__ static inline void PrefixAwarePanelUpdate64(State &state, uint32_t kb, uint32_t q, bool consumer) {
        if constexpr (PREFIX) {
            if (consumer)
                state.template ApplyPanelUpdates<16U, 32U>(kb, q);
            else
                state.ApplyPanelUpdates(kb, q);
        } else
            state.ApplyPanelUpdates(kb, q);
    }
    template <class State> __aicore__ static inline void FinishOrWriteZero64(State &state) {
        if (!state.singular)
            state.Finish64();
        else {
            state.meta.SetValue(0, 0.0f);
            state.SM();
            DataCopyPad(state.out, state.meta, DataCopyExtParams{1, 4, 0, 0, 0});
            state.MS();
        }
    }
    template <class State, bool PREFIX>
    __aicore__ static inline void FactorAndPublish64(State &state, uint32_t kb, uint32_t q, uint32_t owner,
                                                     bool valid) {
        if (valid && state.rank == owner) {
            if (!state.singular)
                state.factor(kb, q);
            else if constexpr (PREFIX)
                state.publishPrefix16(kb);
            state.publish(kb);
        }
    }
    template <class State, bool DEFER, bool EXPORT>
    __aicore__ static inline void FinalizeDeferredPivot64(State &state, uint32_t f, uint32_t p, float piv, float inv) {
        if constexpr (DEFER) {
            state.acc(piv);
            if constexpr (EXPORT) {
                state.mu.SetValue(8 + f, p);
                state.mu.SetValue(8U + State::B + f, state.word(inv));
            }
        }
        PipeBarrier<PIPE_V>();
    }
    template <class State, bool REPAIRED>
    __aicore__ static inline void RepairAndUpdateRows64(State &state, uint32_t p, uint32_t k, uint32_t begin,
                                                        uint32_t h, uint32_t width) {
        if constexpr (!REPAIRED) {
            state.repair(p - k);
            PipeBarrier<PIPE_V>();
        }
        state.updateRows(state.mat[(k + 1) * State::S + begin], state.row, h, State::S, width);
        PipeBarrier<PIPE_V>();
    }
    template <bool COPY, class State> __aicore__ static inline void CopyInputAndWait64(State &state, bool valid) {
        if constexpr (COPY)
            state.copyInput(valid);
        state.MV();
        state.VS();
    }
    template <class State>
    __aicore__ static inline float ReadPivotSegment64(State &state, __ubuf__ float *tp, uint32_t pair, uint32_t base,
                                                      float &maxv, uint32_t &p) {
        float v = tp[pair];
        if (v > maxv) {
            maxv = v;
            p = base + state.word(tp[pair + 1]);
        }
        return v;
    }
