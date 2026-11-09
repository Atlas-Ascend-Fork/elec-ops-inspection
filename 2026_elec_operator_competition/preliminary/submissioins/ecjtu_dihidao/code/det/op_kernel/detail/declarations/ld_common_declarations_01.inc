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

    static_assert(!ClassPlan46::PAIR_LOCAL || (ClassPlan46::GROUP == 2U && ClassPlan46::COMPACT &&
                                               ClassPlan46::CONTIGUOUS_BANDS && ClassPlan46::LOCAL_TAIL &&
                                               ((ClassPlan46::DIM == 192U && ClassPlan46::LOCAL_CUT == 96U) ||
                                                (ClassPlan46::DIM == 200U && ClassPlan46::LOCAL_CUT == 128U))),
                  "pair sync only bounded n192/n200 G2");
    static_assert(!ClassPlan46::CONTIGUOUS_BANDS ||
                      (!ClassPlan46::EXTRA && !ClassPlan46::RETIRE_RECEIVE &&
                       (((ClassPlan46::GROUP == 2U || ClassPlan46::GROUP == 3U) && ClassPlan46::DIM == 192U) ||
                        ((ClassPlan46::GROUP == 2U || ClassPlan46::GROUP == 3U || ClassPlan46::GROUP == 4U ||
                          ClassPlan46::SPLIT_OWNERS) &&
                         ClassPlan46::DIM == 200U) ||
                        (ClassPlan46::GROUP == 3U && ClassPlan46::DIM == 208U && ClassPlan46::LOCAL_TAIL))),
                  "contiguous owner scope");
    static_assert(!ClassPlan46::EXTRA || (ClassPlan46::COMPACT && ClassPlan46::GROUP == 4U),
                  "extra cleanup needs compact four-core MAT coverage");
    static constexpr bool NATIVE208 = ClassPlan46::DIM == 208U && ClassPlan46::GROUP == 3U &&
                                      ClassPlan46::CONTIGUOUS_BANDS && ClassPlan46::LOCAL_TAIL;
    static constexpr bool MAXV_INVERSE = ClassPlan46::CONTIGUOUS_BANDS && ClassPlan46::LOCAL_TAIL &&
                                         (ClassPlan46::DIM == 192U || ClassPlan46::DIM == 200U || NATIVE208);
    static constexpr bool META_AHEAD = ClassPlan46::CONTIGUOUS_BANDS && ClassPlan46::LOCAL_TAIL &&
                                       ClassPlan46::DIM == 192U && ClassPlan46::GROUP == 3U;
    static constexpr bool OWNER_NORMAL =
        ClassPlan46::DIM == 192U && ClassPlan46::GROUP == 6U && ClassPlan46::PANEL_WIDTH == 32U;
    static constexpr bool EARLY_REPAIR =
        ClassPlan46::PANEL_WIDTH == 32U &&
        ((ClassPlan46::DIM == 192U && (ClassPlan46::GROUP == 6U || ClassPlan46::GROUP == 3U)) ||
         (ClassPlan46::DIM == 200U &&
          ((ClassPlan46::GROUP == 4U && ClassPlan46::CONTIGUOUS_BANDS) || ClassPlan46::LOCAL_TAIL)) ||
         NATIVE208);
    static constexpr bool FAR_STATIC =
        ClassPlan46::PANEL_WIDTH == 32U &&
        ((ClassPlan46::DIM == 192U && (!ClassPlan46::CONTIGUOUS_BANDS || ClassPlan46::GROUP == 3U)) ||
         (ClassPlan46::DIM == 200U &&
          ((ClassPlan46::GROUP == 4U && ClassPlan46::CONTIGUOUS_BANDS) || ClassPlan46::LOCAL_TAIL)) ||
         NATIVE208);
    static constexpr bool TRIM_PUBLISH =
        ClassPlan46::DIM == 192U && !ClassPlan46::CONTIGUOUS_BANDS && ClassPlan46::PANEL_WIDTH == 32U &&
        (ClassPlan46::GROUP == 4U || (ClassPlan46::GROUP == 3U && ClassPlan46::PANEL_ACC));
    static constexpr bool PIVOT_AHEAD = ClassPlan46::CONTIGUOUS_BANDS || ClassPlan46::DIM == 208U ||
                                        ClassPlan46::DIM == 192U || ClassPlan46::DIM == 200U;
    static constexpr uint32_t OWNER_BANDS =
        (ClassPlan46::GROUP == 2U && ClassPlan46::CONTIGUOUS_BANDS) ? (ClassPlan46::DIM == 192U ? 3U : 4U) : 2U;
    static constexpr bool TUNED_PANEL =
        ClassPlan46::PANEL_WIDTH == 32U &&
        (ClassPlan46::DIM == 192U ||
         (ClassPlan46::DIM == 200U && (ClassPlan46::CONTIGUOUS_BANDS || ClassPlan46::GROUP == 2U)) || NATIVE208);
    static constexpr uint32_t NMAX = ClassPlan46::DIM, B = ClassPlan46::PANEL_WIDTH,
                              S = ClassPlan46::SPLIT_OWNERS
                                      ? 136U
                                      : (ClassPlan46::GROUP == 2U
                                             ? 136U
                                             : ((ClassPlan46::GROUP == 3U &&
                                                 (ClassPlan46::DIM == 200U || ClassPlan46::DIM == 208U))
                                                    ? (((ClassPlan46::DIM == 200U || ClassPlan46::DIM == 208U) &&
                                                        ClassPlan46::CONTIGUOUS_BANDS && ClassPlan46::LOCAL_TAIL)
                                                           ? 136U
                                                           : 104U)
                                                    : 64U)),
                              G = ClassPlan46::GROUP, P = ClassPlan46::INPLACE_PANEL ? S : ClassPlan46::PANEL_PITCH;
    static_assert((ClassPlan46::INPLACE_PANEL && P == S) || (P == 32U || P == 40U) ||
                      (ClassPlan46::GROUP == 2U && B == 16U && P == 16U),
                  "panel pitch must preserve eight-float alignment");
    static constexpr bool BALANCED200 = ClassPlan46::CONTIGUOUS_BANDS && (ClassPlan46::DIM == 200U || NATIVE208) &&
                                        (ClassPlan46::GROUP == 3U || ClassPlan46::SPLIT_OWNERS);
    static constexpr bool PAIR200 = ClassPlan46::PAIR_LOCAL && ClassPlan46::CONTIGUOUS_BANDS &&
                                    ClassPlan46::DIM == 200U && ClassPlan46::GROUP == 2U && ClassPlan46::LOCAL_TAIL &&
                                    ClassPlan46::LOCAL_CUT == 128U;
