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
template <uint32_t GROUP, bool COMPACT = false, bool EXTRA = false, uint32_t PANEL_PITCH = 32U,
          bool FAC_REPAIRED = false, bool DEFER_BOOKKEEP = false, uint32_t PANEL_WIDTH = 32U, uint32_t DIM = 208U,
          bool INDEX_SEEDS = false, bool PANEL_ACC = false, bool INPUT_OVERLAP = false, bool EARLY_FAC = false,
          bool FIXED_MERGE = false, bool FIXED_REDUCE = false, bool RETIRE_RECEIVE = false,
          bool CONTIGUOUS_BANDS = false, bool PAD_INPUT = false, bool LOCAL_TAIL = false, uint32_t LOCAL_CUT = 128U,
          bool INPLACE_PANEL = false, bool PAIR_LOCAL = false, bool WORD_SELECT = false, bool PREFIX_STREAM = false,
          bool SPLIT_OWNERS = false>
class CyclicRowLUMedium {
    friend struct LdetSharedOperations;

    static_assert(!PAIR_LOCAL || (GROUP == 2U && COMPACT && CONTIGUOUS_BANDS && LOCAL_TAIL &&
                                  ((DIM == 192U && LOCAL_CUT == 96U) || (DIM == 200U && LOCAL_CUT == 128U))),
                  "pair sync only bounded n192/n200 G2");
    static_assert(!CONTIGUOUS_BANDS || (!EXTRA && !RETIRE_RECEIVE &&
                                        (((GROUP == 2U || GROUP == 3U) && DIM == 192U) ||
                                         ((GROUP == 2U || GROUP == 3U || GROUP == 4U || SPLIT_OWNERS) && DIM == 200U) ||
                                         (GROUP == 3U && DIM == 208U && LOCAL_TAIL))),
                  "contiguous owner scope");
    static_assert(!EXTRA || (COMPACT && GROUP == 4U), "extra cleanup needs compact four-core MAT coverage");
    static constexpr bool NATIVE208 = DIM == 208U && GROUP == 3U && CONTIGUOUS_BANDS && LOCAL_TAIL;
    static constexpr bool MAXV_INVERSE = CONTIGUOUS_BANDS && LOCAL_TAIL && (DIM == 192U || DIM == 200U || NATIVE208);
    static constexpr bool META_AHEAD = CONTIGUOUS_BANDS && LOCAL_TAIL && DIM == 192U && GROUP == 3U;
    static constexpr bool OWNER_NORMAL = DIM == 192U && GROUP == 6U && PANEL_WIDTH == 32U;
    static constexpr bool EARLY_REPAIR =
        PANEL_WIDTH == 32U && ((DIM == 192U && (GROUP == 6U || GROUP == 3U)) ||
                               (DIM == 200U && ((GROUP == 4U && CONTIGUOUS_BANDS) || LOCAL_TAIL)) || NATIVE208);
    static constexpr bool FAR_STATIC =
        PANEL_WIDTH == 32U && ((DIM == 192U && (!CONTIGUOUS_BANDS || GROUP == 3U)) ||
                               (DIM == 200U && ((GROUP == 4U && CONTIGUOUS_BANDS) || LOCAL_TAIL)) || NATIVE208);
    static constexpr bool TRIM_PUBLISH =
        DIM == 192U && !CONTIGUOUS_BANDS && PANEL_WIDTH == 32U && (GROUP == 4U || (GROUP == 3U && PANEL_ACC));
    static constexpr bool PIVOT_AHEAD = CONTIGUOUS_BANDS || DIM == 208U || DIM == 192U || DIM == 200U;
    static constexpr uint32_t OWNER_BANDS = (GROUP == 2U && CONTIGUOUS_BANDS) ? (DIM == 192U ? 3U : 4U) : 2U;
    static constexpr bool TUNED_PANEL =
        PANEL_WIDTH == 32U && (DIM == 192U || (DIM == 200U && (CONTIGUOUS_BANDS || GROUP == 2U)) || NATIVE208);
    static constexpr uint32_t NMAX = DIM, B = PANEL_WIDTH,
                              S = SPLIT_OWNERS
                                      ? 136U
                                      : (GROUP == 2U
                                             ? 136U
                                             : ((GROUP == 3U && (DIM == 200U || DIM == 208U))
                                                    ? (((DIM == 200U || DIM == 208U) && CONTIGUOUS_BANDS && LOCAL_TAIL)
                                                           ? 136U
                                                           : 104U)
                                                    : 64U)),
                              G = GROUP, P = INPLACE_PANEL ? S : PANEL_PITCH;
    static_assert((INPLACE_PANEL && P == S) || (P == 32U || P == 40U) || (GROUP == 2U && B == 16U && P == 16U),
                  "panel pitch must preserve eight-float alignment");
    static constexpr bool BALANCED200 = CONTIGUOUS_BANDS && (DIM == 200U || NATIVE208) && (GROUP == 3U || SPLIT_OWNERS);
    static constexpr bool PAIR200 =
        PAIR_LOCAL && CONTIGUOUS_BANDS && DIM == 200U && GROUP == 2U && LOCAL_TAIL && LOCAL_CUT == 128U;
#include "ld_cyclic_rowlu_medium_m01.h"

    static constexpr uint32_t GUARD_ROWS = (DIM == 192U || DIM == 200U || DIM == 208U) ? 8U : 0U,
                              GUARD_COEFF = (DIM == 192U || DIM == 200U || DIM == 208U) ? 1U : 0U;
    static constexpr uint32_t MAT = 0,
                              PANEL =
                                  MAT +
                                  (NMAX + ((INPLACE_PANEL || (BALANCED200 && LOCAL_TAIL)) ? GUARD_ROWS : 0U)) * S * 4U,
                              FAC = PANEL + PANEL_PITCH * (NMAX + GUARD_ROWS) * 4;
    static_assert(PANEL >= MAT + NMAX * S * sizeof(float), "cyclic matrix overlaps panel");
    static constexpr uint32_t BRC = FAC + B * NMAX * 4, COL = BRC + (NMAX + GUARD_COEFF) * 8 * 4;
    static constexpr uint32_t ABS = COL + NMAX * 4, ROW = ABS + NMAX * 4, PROW = ROW + S * 4;
    static constexpr uint32_t TMP_BYTES = NATIVE208 ? 768U : 1024U;
    static constexpr uint32_t TMP = PROW + B * 4, TAB = TMP + TMP_BYTES, LANE = TAB + NMAX * 4;
    static constexpr uint32_t META = LANE + 8 * NMAX * 4, META_WORDS = 8U + 2U * B, END = META + META_WORDS * 4U;
    static constexpr uint32_t GM_WORDS = B * NMAX + META_WORDS;
    static_assert(!PREFIX_STREAM || ((GROUP == 3U || SPLIT_OWNERS) && DIM == 200U && B == 32U && LOCAL_TAIL &&
                                     LOCAL_CUT == 128U && CONTIGUOUS_BANDS && INPLACE_PANEL && FAC_REPAIRED),
                  "prefix stream bounded to n200 B3 key62");
    static_assert(!SPLIT_OWNERS ||
                      (GROUP == 5U && PREFIX_STREAM && DIM == 200U && B == 32U && LOCAL_TAIL && LOCAL_CUT == 128U),
                  "four32 Owners plus72 tail only");
    #include "../../declarations/ld_common_declarations_11.inc"
#include "ld_cyclic_rowlu_medium_m02.h"

  public:
#include "ld_cyclic_rowlu_medium_m03.h"

  private:
    using LdetPolicy = LdetCyclicPolicy<false, COMPACT, CONTIGUOUS_BANDS, DEFER_BOOKKEEP, DIM, false, EARLY_FAC, false,
                                        EXTRA, FAC_REPAIRED, FIXED_MERGE, FIXED_REDUCE, GROUP, false, false, false,
                                        INPLACE_PANEL, LOCAL_CUT, LOCAL_TAIL, false, PAD_INPUT, PANEL_ACC, false, false,
                                        false, PREFIX_STREAM, false, SPLIT_OWNERS, false, WORD_SELECT>;
};
#endif
