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
    __aicore__ static inline void ProdAccumPureShared33Stage1(float &prod, int32_t &ex, float piv) {
        if (piv == 0.0f) {
            prod = 0.0f;
            ex = 0;
            return;
        }
        if (piv != piv || piv > 3.0e38f || piv < -3.0e38f) {
            prod = 0.0f;
            ex = 0;
            return;
        }
        float m;
        int32_t e;
        union {
            float f;
            uint32_t u;
        } cv;
        LdetSharedOperations::ProdAccumStatePhase1PureShared1(cv, e, ex, m, piv, prod);
    }
