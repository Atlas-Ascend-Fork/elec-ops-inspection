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
          bool Value5 = false, uint32_t Value6 = 32U, uint32_t Value7 = 208U, bool Value8 = false, bool Value9 = false,
          bool Value10 = false, bool Value11 = false, bool Value12 = false, bool Value13 = false, bool Value14 = false,
          bool Value15 = false, bool Value16 = false, bool Value17 = false, uint32_t Value18 = 128U,
          bool Value19 = false, bool Value20 = false, bool Value21 = false, bool Value22 = false, bool Value23 = false,
          bool Value24 = false>
struct LdetCyclicSettings46_7 {
    static constexpr uint32_t GROUP = Value0;
    static constexpr bool COMPACT = Value1;
    static constexpr bool EXTRA = Value2;
    static constexpr uint32_t PANEL_PITCH = Value3;
    static constexpr bool FAC_REPAIRED = Value4;
    static constexpr bool DEFER_BOOKKEEP = Value5;
    static constexpr uint32_t PANEL_WIDTH = Value6;
    static constexpr uint32_t DIM = Value7;
    static constexpr bool INDEX_SEEDS = Value8;
    static constexpr bool PANEL_ACC = Value9;
    static constexpr bool INPUT_OVERLAP = Value10;
    static constexpr bool EARLY_FAC = Value11;
    static constexpr bool FIXED_MERGE = Value12;
    static constexpr bool FIXED_REDUCE = Value13;
    static constexpr bool RETIRE_RECEIVE = Value14;
    static constexpr bool CONTIGUOUS_BANDS = Value15;
    static constexpr bool PAD_INPUT = Value16;
    static constexpr bool LOCAL_TAIL = Value17;
    static constexpr uint32_t LOCAL_CUT = Value18;
    static constexpr bool INPLACE_PANEL = Value19;
    static constexpr bool PAIR_LOCAL = Value20;
    static constexpr bool WORD_SELECT = Value21;
    static constexpr bool PREFIX_STREAM = Value22;
    static constexpr bool SPLIT_OWNERS = Value23;
    static constexpr bool ASYNC_FRESHNESS = Value24;
};
#endif
