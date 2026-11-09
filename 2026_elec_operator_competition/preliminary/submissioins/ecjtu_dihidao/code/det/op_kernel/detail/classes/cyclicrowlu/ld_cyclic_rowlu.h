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
          bool FAC_REPAIRED = false, bool DEFER_BOOKKEEP = false, uint32_t PANEL_WIDTH = 32U,
          bool PRUNE_STAGE_WAITS = false, bool ACTIVE_PANEL_ROWS = false, bool FINAL_NO_EXPORT = false,
          bool PANEL_ACC = false, bool PIVOT8_AHEAD = false, bool CONTIGUOUS_BANDS = false, bool NORMAL_COPY = false,
          bool DIRECT_FAC = false, bool UB_FINISH64 = false, bool PANEL_MASK = false, bool HETERO5 = false,
          bool HETERO3 = false>
class CyclicRowLU {
    friend struct LdetSharedOperations;

    static_assert(!HETERO3 ||
                      (GROUP == 3U && PANEL_WIDTH == 32U && CONTIGUOUS_BANDS && UB_FINISH64 && PANEL_MASK && !HETERO5),
                  "G3threebands36AIV");
    static_assert(!HETERO5 || (GROUP == 5U && PANEL_WIDTH == 32U && CONTIGUOUS_BANDS && UB_FINISH64 && PANEL_MASK),
                  "G5fivebands40AIV");
    static_assert(!PANEL_MASK || ((GROUP == 4U || HETERO5 || HETERO3) && FAC_REPAIRED && CONTIGUOUS_BANDS),
                  "persistent mask scope");
    static_assert(!UB_FINISH64 || ((GROUP == 4U || HETERO5 || HETERO3) && CONTIGUOUS_BANDS), "tail64 scope");
    static_assert(!DIRECT_FAC || ((GROUP == 4U || HETERO5 || HETERO3) && CONTIGUOUS_BANDS), "directFAC scope");
    static_assert(!NORMAL_COPY || ((GROUP == 4U || HETERO5 || HETERO3) && CONTIGUOUS_BANDS), "normal-copy kept scope");
    static_assert(!CONTIGUOUS_BANDS || ((GROUP == 4U || HETERO5 || HETERO3) && PANEL_WIDTH == 32U),
                  "n256 adjacent owner scope");
    static_assert(!ACTIVE_PANEL_ROWS || ((GROUP == 4U || HETERO5 || HETERO3) && PANEL_WIDTH == 32U),
                  "active panel copy scoped to four-core K32");
    static_assert(!PRUNE_STAGE_WAITS || (GROUP == 8U && !COMPACT), "sync pruning scoped to noncompact G8");
    static_assert(!EXTRA || (COMPACT && (GROUP == 4U || HETERO5 || HETERO3)),
                  "extra cleanup needs compact four-core MAT coverage");
    static constexpr uint32_t NMAX = 256, B = PANEL_WIDTH,
                              S = HETERO3       ? 96U
                                  : GROUP == 2U ? 136U
                                                : ((GROUP == 8U && PRUNE_STAGE_WAITS) ? 40U : 64U),
                              G = GROUP, P = PANEL_PITCH;
    static_assert((P == 32U || P == 40U) || (GROUP == 2U && B == 16U && P == 16U),
                  "panel pitch must preserve eight-float alignment");
    static constexpr bool GUARDED = GROUP == 8U || GROUP == 4U || HETERO5 || HETERO3;
    static_assert(!PIVOT8_AHEAD || B <= 32U, "first retired TMP tile covers all reduction destinations");
    static constexpr uint32_t GUARD_ROWS = GUARDED ? 8U : 0U, GUARD_COEFF = GUARDED ? 1U : 0U;
    static constexpr uint32_t MAT = 0, PANEL = MAT + NMAX * S * 4U, FAC = PANEL + P * (NMAX + GUARD_ROWS) * 4;
    static_assert(PANEL >= MAT + NMAX * S * sizeof(float), "cyclic matrix overlaps panel");
    #include "../../declarations/ld_common_declarations_06.inc"
    LocalTensor<uint32_t> mu;
    GlobalTensor<float> in, ws, out;
    GlobalTensor<uint32_t> wu;
    uint32_t n, rank, matrix, blocks, owned;
    float sign, prod;
    int32_t ex;
    uint32_t singular;
#include "ld_cyclic_rowlu_m01.h"

  public:
#include "ld_cyclic_rowlu_m02.h"

  private:
    using LdetPolicy = LdetCyclicPolicy<false, COMPACT, false, false, false, DIRECT_FAC, false, false, false, false,
                                        false, false, GROUP, false, false, false, false, false, false, NORMAL_COPY,
                                        false, PANEL_ACC, false, false, false, false, false, false, false, false>;
};
#endif
