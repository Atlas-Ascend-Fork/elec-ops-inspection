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
    template <class PhaseArg0, class PhaseArg1, class PhaseArg2, class PhaseArg3, class PhaseArg4, class PhaseArg5>
    __aicore__ static inline void ProdAccumStatePhase1PureShared1(PhaseArg0 &cv, PhaseArg1 &e, PhaseArg2 &ex,
                                                                  PhaseArg3 &m, PhaseArg4 &piv, PhaseArg5 &prod) {

        cv.f = piv;
        uint32_t ef = (cv.u >> 23) & 0xFFu;
        if (ef >= 2u && ef <= 253u) {
            e = static_cast<int32_t>(ef) - 126;
            cv.u = (cv.u & 0x807FFFFFu) | 0x3F000000u;
            m = cv.f;
        } else {
            m = piv;
            e = 0;
            float am = m < 0.0f ? -m : m;
            while (am < 1.0f) {
                m *= ::kTwoPow64;
                e -= 64;
                am *= ::kTwoPow64;
            }
            while (am >= 2.0f) {
                m *= ::kTwoPowNeg64;
                e += 64;
                am *= ::kTwoPowNeg64;
            }
        }
        prod *= m;
        ex += e;
        float ap = prod < 0.0f ? -prod : prod;
        if (ap >= ::kTwoPow60) {
            prod *= ::kTwoPowNeg64;
            ex += 64;
        } else if (ap != 0.0f && ap < ::kTwoPowNeg63) {
            prod *= ::kTwoPow64;
            ex -= 64;
        }
    }
