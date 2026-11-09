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
    template <class OwnerState, bool ClassValue0, auto T, auto TS>
    __aicore__ static inline void
    Finish64GeppvmlaPhase1SharedPhase22(OwnerState &state, __ubuf__ float *&ap, __ubuf__ float *&cp, uint32_t &f,
                                        const uint32_t &first, uint32_t &h, __ubuf__ uint32_t *&ip, __ubuf__ float *&mp,
                                        bool &pendingFar, float &piv, bool &prepared, __ubuf__ float *&rp) {

        if (h > 1U && first < T) {
            SetMaskNorm();
            AscendCUtils::SetMask<float>(64);
            vmla(mp + (f + 1U) * TS + first, rp + first, reinterpret_cast<__ubuf__ float *>(state.brc[8U].GetPhyAddr()),
                 static_cast<uint8_t>((h - 1U + 7U) / 8U), TS / 8U, 0, 1, TS, 0, 8);
            PipeBarrier<PIPE_V>();
            const uint32_t nf = f + 1U, nh = h - 1U;
            PrepareNextTailColumn60<T, TS>(state, nf, nh, ap, cp, ip, mp);
            prepared = true;
            if (first + 8U < T) {
                SetMaskNorm();
                AscendCUtils::SetMask<float>(static_cast<int32_t>(T - first - 8U));
                vmla(mp + (f + 1U) * TS + first + 8U, rp + first + 8U,
                     reinterpret_cast<__ubuf__ float *>(state.brc[8U].GetPhyAddr()), static_cast<uint8_t>(h - 1U), 1, 1,
                     0, TS / 8U, 0, 1);
                ResetMask();
                SetMaskNorm();
                pendingFar = true;
            }
        }
        state.acc(piv);
        if constexpr (ClassValue0) {
            if ((f & 31U) == 31U)
                state.normalizeProd();
        }
    }
