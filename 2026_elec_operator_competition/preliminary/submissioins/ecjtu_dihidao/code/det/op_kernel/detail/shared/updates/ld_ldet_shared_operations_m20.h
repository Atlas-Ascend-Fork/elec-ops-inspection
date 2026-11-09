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
    template <class OwnerState, bool ClassValue0, bool ClassValue1, bool ClassValue2, uint32_t FIRST, uint32_t LAST>
    __aicore__ static inline void ApplyPanelUpdatesvmulsPhase6SharedPhase12(OwnerState &state, const uint32_t &begin,
                                                                            uint32_t &kb, const uint32_t &width) {

        auto mp = reinterpret_cast<__ubuf__ float *>(state.mat.GetPhyAddr());
        auto rp = reinterpret_cast<__ubuf__ float *>(state.row.GetPhyAddr());
        for (uint32_t f = FIRST; f < LAST; ++f) {
            uint32_t k = kb + f, h = state.n - k, p = state.mu.GetValue(8 + f);
            float inv = state.value(state.mu.GetValue(8U + OwnerState::B + f));
            if constexpr (ClassValue1)
                AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
            else
                LDET_MASK_SET(width);
            vmuls(rp, mp + p * OwnerState::S + begin, inv, 1, 1, 1, 8, 8);
            if (p != k)
                vmuls(mp + p * OwnerState::S + begin, mp + k * OwnerState::S + begin, 1.0f, 1, 1, 1, 8, 8);
            if constexpr (ClassValue1)
                ResetMask();
            else
                LDET_MASK_RST();
            if (h > 1)
                vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.brc.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(state.fac[f * state.facPitch(kb)].GetPhyAddr()), 1, 8,
                      static_cast<uint8_t>((h + 7) / 8));
            PipeBarrier<PIPE_V>();
            if constexpr (!ClassValue0) {
                state.repair(p - k);
                PipeBarrier<PIPE_V>();
            }
            state.updateRows(state.mat[(k + 1) * OwnerState::S + begin], state.row, h, OwnerState::S, width);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (!ClassValue2)
            state.VS();
    }
