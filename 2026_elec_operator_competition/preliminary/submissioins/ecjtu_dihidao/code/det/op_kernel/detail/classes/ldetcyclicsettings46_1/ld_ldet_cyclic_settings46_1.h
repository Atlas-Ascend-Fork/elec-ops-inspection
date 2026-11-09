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
// Private device implementation; included by ld_device.h.
#if (defined(__DAV_C220_VEC__))
template <uint32_t Value0, bool Value1 = false, bool Value2 = false, uint32_t Value3 = 32U, bool Value4 = false,
          bool Value5 = false, uint32_t Value6 = 32U, bool Value7 = false, bool Value8 = false, bool Value9 = false,
          bool Value10 = false, bool Value11 = false, bool Value12 = false, bool Value13 = false, bool Value14 = false,
          bool Value15 = false, bool Value16 = false, bool Value17 = false, bool Value18 = false, bool Value19 = false,
          bool Value20 = false>
struct LdetCyclicSettings46_1 {
    #include "../../declarations/ld_common_declarations_07.inc"
    static constexpr bool HETERO5 = Value17;
    static constexpr bool HETERO3 = Value18;
    static constexpr bool EARLY_TAIL_INDEX = Value19;
    static constexpr bool PHASE_TRIM = Value20;
};
#endif
