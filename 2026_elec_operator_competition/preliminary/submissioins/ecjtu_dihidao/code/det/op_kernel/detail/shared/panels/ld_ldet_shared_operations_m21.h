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
    template <class OwnerState, bool ClassValue0, bool ClassValue1, bool ClassValue2, bool ClassValue3, bool EXPORT>
    __aicore__ static inline void factorPipeBarrierPhase21SharedPhase13(OwnerState &state, uint32_t &f, uint32_t &h,
                                                                        float &inv, uint32_t &k, uint32_t &kb,
                                                                        uint32_t &p, float &piv, bool &pivotPrepared,
                                                                        bool &prefixPosted) {

        if constexpr (ClassValue2)
            ResetMask();
        else
            LDET_MASK_RST();
        if constexpr (OwnerState::GUARDED || ClassValue3) {
            if (p != k)
                PipeBarrier<PIPE_V>();
        }
        state.repair(p - k);
        PipeBarrier<PIPE_V>();
        uint32_t c0 = (f + 1) & ~7U, w = OwnerState::B - c0;
        if constexpr (ClassValue1 && EXPORT)
            state.patchFac(f, kb, p - k);
        if constexpr (ClassValue3) {
            pivotPrepared = w ? state.template updatePanelAndPrepare<EXPORT>(kb, f, h, c0, w) : false;
        } else {
            if (w)
                state.updatePanelRows8x8(state.panel[(k + 1) * OwnerState::P + c0], state.prow[c0], h, w);
        }
        if constexpr (ClassValue0) {
            state.acc(piv);
            if constexpr (EXPORT) {
                state.mu.SetValue(8 + f, p);
                state.mu.SetValue(8U + OwnerState::B + f, state.word(inv));
            }
        }
        PipeBarrier<PIPE_V>();
        if constexpr (OwnerState::PREFIX_STREAM && EXPORT) {
            if (f == 15U) {
                state.publishPrefix16(kb);
                prefixPosted = true;
            }
        }
    }
