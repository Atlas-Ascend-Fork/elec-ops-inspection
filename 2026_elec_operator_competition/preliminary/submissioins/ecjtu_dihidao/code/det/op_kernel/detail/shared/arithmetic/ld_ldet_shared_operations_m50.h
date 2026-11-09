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
    template <class Self, bool ClassValue0, bool ClassValue1, bool ClassValue2, bool EXPORT>
    __aicore__ static inline void SharedOwnerVmulsStage11(Self &state, uint32_t &f, uint32_t &h, float &inv,
                                                          uint32_t &k, uint32_t &kb, uint32_t &p, bool &pivotPrepared,
                                                          __ubuf__ float *&pp, __ubuf__ float *&rp) {
        if constexpr (ClassValue1)
            AscendCUtils::SetMask<float>(static_cast<int32_t>(Self::B));
        else
            LDET_MASK_SET(Self::B);
        vmuls(rp, pp + p * Self::P, inv, 1, 1, 1, 8, 8);
        if (p != k)
            vmuls(pp + p * Self::P, pp + k * Self::P, 1.0f, 1, 1, 1, 8, 8);
        if constexpr (ClassValue1)
            ResetMask();
        else
            LDET_MASK_RST();
        if constexpr (Self::GUARDED || ClassValue2) {
            if (p != k)
                PipeBarrier<PIPE_V>();
        }
        state.repair(p - k);
        PipeBarrier<PIPE_V>();
        uint32_t c0 = (f + 1) & ~7U, w = Self::B - c0;
        if constexpr (ClassValue0 && EXPORT)
            state.patchFac(f, kb, p - k);
        if constexpr (ClassValue2) {
            pivotPrepared = w ? state.template updatePanelAndPrepare<EXPORT>(kb, f, h, c0, w) : false;
        } else {
            if (w)
                state.updatePanelRows8x8(state.panel[(k + 1) * Self::P + c0], state.prow[c0], h, w);
        }
    }
