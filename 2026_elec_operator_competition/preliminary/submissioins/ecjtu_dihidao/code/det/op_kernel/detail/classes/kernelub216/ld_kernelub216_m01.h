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
// Included in the KernelUB216 class; preserves its member scope and access.
    __aicore__ inline void ProdAccum(float &prod, int32_t &ex, float piv) {
        return LdetSharedOperations::ProdAccumPureShared33Stage1(prod, ex, piv);
    }
    __aicore__ inline float ProdFinal(float prod, int32_t ex) {
        return LdetSharedOperations::ProdFinalPureShared33Stage2(prod, ex);
    }
