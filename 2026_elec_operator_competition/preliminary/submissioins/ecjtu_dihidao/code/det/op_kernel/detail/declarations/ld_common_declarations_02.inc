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
// Intentionally included multiple times in class scope.
// Complete declarations only; template names and member order are preserved.
friend struct LdetSharedOperations;

    static constexpr bool BRANCHLESS_SEGMENTS = ClassPlan46::HETERO5 || ClassPlan46::GROUP == 4U;
    static constexpr bool PREFIX_STREAM = ClassPlan46::HETERO5 || ClassPlan46::GROUP == 4U;
    GM_ADDR prefixBase;
    static_assert(!ClassPlan46::HETERO3 ||
                      (ClassPlan46::GROUP == 3U && ClassPlan46::PANEL_WIDTH == 32U && ClassPlan46::CONTIGUOUS_BANDS &&
                       ClassPlan46::UB_FINISH64 && ClassPlan46::PANEL_MASK && !ClassPlan46::HETERO5),
                  "G3threebands36AIV");
    static_assert(!ClassPlan46::HETERO5 ||
                      (ClassPlan46::GROUP == 5U && ClassPlan46::PANEL_WIDTH == 32U && ClassPlan46::CONTIGUOUS_BANDS &&
                       ClassPlan46::UB_FINISH64 && ClassPlan46::PANEL_MASK),
                  "G5fivebands40AIV");
    static_assert(!ClassPlan46::PANEL_MASK ||
                      ((ClassPlan46::GROUP == 4U || ClassPlan46::HETERO5 || ClassPlan46::HETERO3) &&
                       ClassPlan46::FAC_REPAIRED && ClassPlan46::CONTIGUOUS_BANDS),
                  "persistent mask scope");
    static_assert(!ClassPlan46::UB_FINISH64 ||
                      ((ClassPlan46::GROUP == 4U || ClassPlan46::HETERO5 || ClassPlan46::HETERO3) &&
                       ClassPlan46::CONTIGUOUS_BANDS),
                  "tail64 scope");
    static_assert(!ClassPlan46::DIRECT_FAC ||
                      ((ClassPlan46::GROUP == 4U || ClassPlan46::HETERO5 || ClassPlan46::HETERO3) &&
                       ClassPlan46::CONTIGUOUS_BANDS),
                  "directFAC scope");
    static_assert(!ClassPlan46::NORMAL_COPY ||
                      ((ClassPlan46::GROUP == 4U || ClassPlan46::HETERO5 || ClassPlan46::HETERO3) &&
                       ClassPlan46::CONTIGUOUS_BANDS),
                  "normal-copy kept scope");
    static_assert(!ClassPlan46::CONTIGUOUS_BANDS ||
                      ((ClassPlan46::GROUP == 4U || ClassPlan46::HETERO5 || ClassPlan46::HETERO3) &&
                       ClassPlan46::PANEL_WIDTH == 32U),
                  "n256 adjacent owner scope");
    static_assert(!ClassPlan46::ACTIVE_PANEL_ROWS ||
                      ((ClassPlan46::GROUP == 4U || ClassPlan46::HETERO5 || ClassPlan46::HETERO3) &&
                       ClassPlan46::PANEL_WIDTH == 32U),
                  "active panel copy scoped to four-core K32");
    static_assert(!ClassPlan46::PRUNE_STAGE_WAITS || (ClassPlan46::GROUP == 8U && !ClassPlan46::COMPACT),
                  "sync pruning scoped to noncompact G8");
    static_assert(!ClassPlan46::EXTRA || (ClassPlan46::COMPACT &&
                                          (ClassPlan46::GROUP == 4U || ClassPlan46::HETERO5 || ClassPlan46::HETERO3)),
                  "extra cleanup needs compact four-core MAT coverage");
    static constexpr bool FAST_OWNER = ClassPlan46::HETERO5 || ClassPlan46::PHASE_TRIM;
    static constexpr uint32_t NMAX = 256, B = ClassPlan46::PANEL_WIDTH,
                              S = ClassPlan46::HETERO3 ? 96U
                                  : ClassPlan46::GROUP == 2U
                                      ? 136U
                                      : ((ClassPlan46::GROUP == 8U && ClassPlan46::PRUNE_STAGE_WAITS) ? 40U : 64U),
                              G = ClassPlan46::GROUP, P = ClassPlan46::PANEL_PITCH;
    static_assert((P == 32U || P == 40U) || (ClassPlan46::GROUP == 2U && B == 16U && P == 16U),
                  "panel pitch must preserve eight-float alignment");
    static constexpr bool GUARDED =
        ClassPlan46::GROUP == 8U || ClassPlan46::GROUP == 4U || ClassPlan46::HETERO5 || ClassPlan46::HETERO3;
    static_assert(!ClassPlan46::PIVOT8_AHEAD || B <= 32U, "first retired TMP tile covers all reduction destinations");
    static constexpr uint32_t GUARD_ROWS = GUARDED ? 8U : 0U, GUARD_COEFF = GUARDED ? 1U : 0U;
    static constexpr uint32_t MAT = 0, PANEL = MAT + NMAX * S * 4U, FAC = PANEL + P * (NMAX + GUARD_ROWS) * 4;
    static_assert(PANEL >= MAT + NMAX * S * sizeof(float), "cyclic matrix overlaps panel");
    #include "ld_common_declarations_10.inc"
    LocalTensor<uint32_t> mu;
    GlobalTensor<float> in, ws, out, relay;
    GlobalTensor<uint32_t> wu;
    uint32_t n, rank, matrix, blocks, owned, recycleState;
    float sign, prod;
    int32_t ex;
    uint32_t singular;
