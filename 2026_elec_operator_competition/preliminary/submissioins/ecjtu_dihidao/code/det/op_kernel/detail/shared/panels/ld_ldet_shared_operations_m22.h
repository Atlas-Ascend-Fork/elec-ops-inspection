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
    template <class OwnerState, bool ClassValue0, bool ClassValue1, bool EXPORT>
    __aicore__ static inline void
    factorvgatherPhase22SharedPhase14(OwnerState &state, __ubuf__ float *&ap, uint32_t &be, __ubuf__ float *&cp,
                                      uint32_t &f, uint32_t &h, __ubuf__ uint32_t *&ip, uint32_t &k, uint32_t &kb,
                                      bool &pivotPrepared, __ubuf__ float *&pp) {

        Shared10SetFlagStage6<ClassValue0, ClassValue1, EXPORT, OwnerState>(ap, be, cp, f, h, ip, k, kb, pivotPrepared,
                                                                            pp, state);
        vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.brc.GetPhyAddr()), reinterpret_cast<__ubuf__ uint32_t *>(cp),
              1, 8, static_cast<uint8_t>((h + 7) / 8));
        if constexpr (EXPORT && !ClassValue0)
            DataCopy(state.fac[f * state.facPitch(kb)], state.col, ((h + 7U) / 8U) * 8U);
    }
