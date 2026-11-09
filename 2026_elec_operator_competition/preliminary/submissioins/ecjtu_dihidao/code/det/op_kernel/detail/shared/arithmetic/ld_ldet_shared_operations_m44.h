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
    template <class Capture9>
    __aicore__ static inline void Shared14VmulsStage2(const uint32_t &begin, uint32_t &f, uint32_t &h, float &inv,
                                                      uint32_t &k, uint32_t &kb, __ubuf__ float *&mp, uint32_t &p,
                                                      __ubuf__ float *&rp, Capture9 &state) {
        vmuls(rp, mp + p * Capture9::S + begin, inv, 1, 1, 1, 8, 8);
        if (p != k)
            vmuls(mp + p * Capture9::S + begin, mp + k * Capture9::S + begin, 1.0f, 1, 1, 1, 8, 8);
        if (h > 1U)
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.brc.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(state.fac[f * state.facPitch(kb)].GetPhyAddr()), 1, 8,
                  static_cast<uint8_t>((h + 7U) / 8U));
        PipeBarrier<PIPE_V>();
        if (h > 1U)
            vmla(mp + (k + 1U) * Capture9::S + begin, rp,
                 reinterpret_cast<__ubuf__ float *>(state.brc[8U].GetPhyAddr()), static_cast<uint8_t>(h - 1U), 1, 1, 0,
                 Capture9::S / 8U, 0, 1);
    }
    template <class Self, bool ClassValue0, uint32_t FIRST, uint32_t LAST>
    __aicore__ static inline void SharedOwnerVmulsStage5(Self &state, const uint32_t &begin, uint32_t &kb,
                                                         const uint32_t &width) {
        auto mp = reinterpret_cast<__ubuf__ float *>(state.mat.GetPhyAddr()),
             rp = reinterpret_cast<__ubuf__ float *>(state.row.GetPhyAddr());
        SetMaskNorm();
        AscendCUtils::SetMask<float>(static_cast<int32_t>(width));
        for (uint32_t f = FIRST; f < LAST; ++f) {
            uint32_t k = kb + f, h = state.n - k, p = state.mu.GetValue(8 + f);
            float inv = state.value(state.mu.GetValue(8U + Self::B + f));
            Shared14VmulsStage2(begin, f, h, inv, k, kb, mp, p, rp, state);
            PipeBarrier<PIPE_V>();
        }
        ResetMask();
        SetMaskNorm();
        if constexpr (!ClassValue0)
            state.VS();
    }
