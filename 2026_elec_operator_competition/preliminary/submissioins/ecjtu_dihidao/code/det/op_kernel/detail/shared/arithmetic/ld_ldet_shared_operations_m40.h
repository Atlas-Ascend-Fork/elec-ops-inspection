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
    template <class Self> __aicore__ static inline void SharedOwnerPipeBarrierStage1(Self &state, uint32_t &kb) {
        auto mp = reinterpret_cast<__ubuf__ float *>(state.mat.GetPhyAddr());
        auto rp = reinterpret_cast<__ubuf__ float *>(state.row.GetPhyAddr());
        auto coeff = reinterpret_cast<__ubuf__ float *>(state.brc[8U].GetPhyAddr());
        for (uint32_t half = 0U; half < 2U; ++half) {
            const uint32_t begin = half == 0U ? 32U : 0U;
            SetMaskNorm();
            for (uint32_t f = 0U; f < Self::B; ++f) {
                const uint32_t k = kb + f, h = state.n - k, p = state.mu.GetValue(8U + f), rows = h - 1U;
                float inv = state.value(state.mu.GetValue(8U + Self::B + f));
                if (f)
                    PipeBarrier<PIPE_V>();
                AscendCUtils::SetMask<float>(32);
                vmuls(rp, mp + p * Self::S + begin, inv, 1, 1, 1, 8, 8);
                if (p != k)
                    vmuls(mp + p * Self::S + begin, mp + k * Self::S + begin, 1.0f, 1, 1, 1, 8, 8);
                vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.brc.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ uint32_t *>(state.fac[f * state.facPitch(kb)].GetPhyAddr()), 1, 8,
                      static_cast<uint8_t>((h + 7U) / 8U));
                PipeBarrier<PIPE_V>();
                AscendCUtils::SetMask<float>(32);
                vmla(mp + (k + 1U) * Self::S + begin, rp, coeff, static_cast<uint8_t>(rows), 1, 1, 0, Self::S / 8U, 0,
                     1);
            }
            PipeBarrier<PIPE_V>();
            ResetMask();
            SetMaskNorm();
            if (half == 0U) {
                state.VM();
                DataCopy(state.relay, state.mat[(kb + 32U) * Self::S + 32U],
                         {static_cast<uint16_t>(state.n - kb - 32U), 4U, (Self::S - 32U) / 8U, 0U});
            }
        }
        state.MS();
    }
