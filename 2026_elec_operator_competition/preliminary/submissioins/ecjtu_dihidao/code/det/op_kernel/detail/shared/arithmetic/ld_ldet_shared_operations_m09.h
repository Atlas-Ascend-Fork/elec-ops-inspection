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
    template <class OwnerState, class Capture7>
    __aicore__ static inline void SharedVgatherStage13(__ubuf__ float *&ap, uint32_t &be, __ubuf__ float *&cp,
                                                       uint32_t &f, uint32_t &h, __ubuf__ uint32_t *&ip,
                                                       __ubuf__ float *&pp, Capture7 &state) {
        return Shared10VgatherStage4<OwnerState, uint32_t, uint32_t, Capture7>(ap, be, cp, f, h, ip, pp, state);
    }
    template <bool ClassValue0, bool ClassValue1, bool EXPORT, class OwnerState, class Capture10>
    __aicore__ static inline void Shared10SetFlagStage6(__ubuf__ float *&ap, uint32_t &be, __ubuf__ float *&cp,
                                                        uint32_t &f, uint32_t &h, __ubuf__ uint32_t *&ip, uint32_t &k,
                                                        uint32_t &kb, bool &pivotPrepared, __ubuf__ float *&pp,
                                                        Capture10 &state) {
        if constexpr (ClassValue0 && EXPORT)
            cp = reinterpret_cast<__ubuf__ float *>(state.fac[f * state.facPitch(kb)].GetPhyAddr());
        if constexpr (ClassValue1) {
            if (!pivotPrepared)
                state.template issuePreparedPivot<EXPORT>(k, f);
        } else {
            SharedVgatherStage13<OwnerState>(ap, be, cp, f, h, ip, pp, state);
        }
        if constexpr (OwnerState::GUARDED && !ClassValue1)
            SetFlag<HardEvent::V_S>(2);
    }
    template <class OwnerState, bool ClassValue0, bool ClassValue1, bool EXPORT>
    __aicore__ static inline void factorvgatherPhase2SharedPhase1(OwnerState &state, __ubuf__ float *&ap, uint32_t &be,
                                                                  __ubuf__ float *&cp, uint32_t &f, uint32_t &h,
                                                                  __ubuf__ uint32_t *&ip, uint32_t &k, uint32_t &kb,
                                                                  bool &pivotPrepared, __ubuf__ float *&pp) {

        Shared10SetFlagStage6<ClassValue0, ClassValue1, EXPORT, OwnerState>(ap, be, cp, f, h, ip, k, kb, pivotPrepared,
                                                                            pp, state);
        if (h > 1)
            vbrcb(reinterpret_cast<__ubuf__ uint32_t *>(state.brc.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t *>(cp), 1, 8, static_cast<uint8_t>((h + 7) / 8));
        if constexpr (EXPORT && !ClassValue0)
            DataCopy(state.fac[f * state.facPitch(kb)], state.col, ((h + 7U) / 8U) * 8U);
    }
