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
    template <class Capture2>
    __aicore__ static inline void Shared14ScalarUpdateStage4(GM_ADDR &input, GM_ADDR &output, Capture2 &state,
                                                             const bool &valid, GM_ADDR &workspace) {
        const uint32_t addressMatrix = valid ? state.matrix : 0U;
        state.in.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input) + addressMatrix * state.n * state.n);
        state.ws.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) +
                                 addressMatrix * (2U * Capture2::GM_WORDS));
        state.wu.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace) +
                                 addressMatrix * (2U * Capture2::GM_WORDS));
        state.out.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output) + addressMatrix);
        if constexpr (Capture2::PREFIX_STREAM)
            state.prefixBase =
                workspace + (GetBlockNum() / Capture2::G) * (2U * Capture2::GM_WORDS) * 4U + addressMatrix * 6U * 64U;
    }
    template <class Self, bool ClassValue0, bool ClassValue1>
    __aicore__ static inline void SharedOwnerScalarUpdateStage7(Self &state, GM_ADDR &input, GM_ADDR &output,
                                                                const bool &valid, GM_ADDR &workspace) {
        Shared14ScalarUpdateStage4(input, output, state, valid, workspace);
        if constexpr (Self::PREFIX_STREAM && (!ClassValue0 || ClassValue1)) {
            if (valid && state.rank < 6U)
                LuPublish(state.prefixTicket(state.rank), 0U);
        }
        if (state.rank != 0U)
            SyncAll<true>();
        state.sign = 1.0f;
    }
