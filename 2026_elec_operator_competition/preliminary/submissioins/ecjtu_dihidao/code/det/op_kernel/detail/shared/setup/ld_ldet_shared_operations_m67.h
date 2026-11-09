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
    template <class Self, bool ClassValue0, bool ClassValue1>
    __aicore__ static inline void SharedOwnerDuplicateStage30(Self &state, GM_ADDR &input, GM_ADDR &output,
                                                              const bool &valid, GM_ADDR &workspace) {
        Shared14ScalarUpdateStage4(input, output, state, valid, workspace);
        state.sign = 1.0f;
        state.prod = 1.0f;
        state.ex = 0;
        state.singular = 0;
        if constexpr (!ClassValue0 && !ClassValue1)
            Duplicate(state.mat, 0.0f, Self::NMAX * Self::S);
    }
