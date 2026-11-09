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
template <auto Value0 = false, auto Value1 = false, auto Value2 = false, auto Value3 = false, auto Value4 = false,
          auto Value5 = false, auto Value6 = false, auto Value7 = false, auto Value8 = false, auto Value9 = false,
          auto Value10 = false, auto Value11 = false, auto Value12 = false, auto Value13 = false, auto Value14 = false,
          auto Value15 = false, auto Value16 = false, auto Value17 = false, auto Value18 = false, auto Value19 = false,
          auto Value20 = false, auto Value21 = false, auto Value22 = false, auto Value23 = false, auto Value24 = false,
          auto Value25 = false, auto Value26 = false, auto Value27 = false, auto Value28 = false, auto Value29 = false>
struct LdetCyclicPolicy {
    static constexpr auto ACTIVE_PANEL_ROWS = Value0;
    static constexpr auto COMPACT = Value1;
    static constexpr auto CONTIGUOUS_BANDS = Value2;
    static constexpr auto DEFER_BOOKKEEP = Value3;
    static constexpr auto DIM = Value4;
    static constexpr auto DIRECT_FAC = Value5;
    static constexpr auto EARLY_FAC = Value6;
    static constexpr auto EARLY_TAIL_INDEX = Value7;
    static constexpr auto EXTRA = Value8;
    static constexpr auto FAC_REPAIRED = Value9;
    static constexpr auto FIXED_MERGE = Value10;
    static constexpr auto FIXED_REDUCE = Value11;
    static constexpr auto GROUP = Value12;
    static constexpr auto HETERO3 = Value13;
    static constexpr auto HETERO5 = Value14;
    static constexpr auto HETERO6 = Value15;
    static constexpr auto INPLACE_PANEL = Value16;
    static constexpr auto LOCAL_CUT = Value17;
    static constexpr auto LOCAL_TAIL = Value18;
    static constexpr auto NORMAL_COPY = Value19;
    static constexpr auto PAD_INPUT = Value20;
    static constexpr auto PANEL_ACC = Value21;
    static constexpr auto PANEL_MASK = Value22;
    static constexpr auto PHASE_TRIM = Value23;
    static constexpr auto PIVOT8_AHEAD = Value24;
    static constexpr auto PREFIX_STREAM = Value25;
    static constexpr auto PRUNE_STAGE_WAITS = Value26;
    static constexpr auto SPLIT_OWNERS = Value27;
    static constexpr auto THIN_H6 = Value28;
    static constexpr auto WORD_SELECT = Value29;
};
#endif
