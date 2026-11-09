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
static_assert(!ClassPlan46::PANEL_MASK || ((ClassPlan46::GROUP == 4U || ClassPlan46::HETERO6) &&
                                               ClassPlan46::FAC_REPAIRED && ClassPlan46::CONTIGUOUS_BANDS),
                  "persistent mask scope");
    static_assert(!ClassPlan46::UB_FINISH64 ||
                      ((ClassPlan46::GROUP == 4U || ClassPlan46::HETERO6) && ClassPlan46::CONTIGUOUS_BANDS),
                  "tail64 scope");
    static_assert(!ClassPlan46::DIRECT_FAC ||
                      ((ClassPlan46::GROUP == 4U || ClassPlan46::HETERO6) && ClassPlan46::CONTIGUOUS_BANDS),
                  "directFAC scope");
    static_assert(!ClassPlan46::NORMAL_COPY ||
                      ((ClassPlan46::GROUP == 4U || ClassPlan46::HETERO6) && ClassPlan46::CONTIGUOUS_BANDS),
                  "normal-copy kept scope");
    static_assert(!ClassPlan46::CONTIGUOUS_BANDS ||
                      ((ClassPlan46::GROUP == 4U || ClassPlan46::HETERO6) && ClassPlan46::PANEL_WIDTH == 32U),
                  "n256 adjacent owner scope");
    static_assert(!ClassPlan46::ACTIVE_PANEL_ROWS ||
                      ((ClassPlan46::GROUP == 4U || ClassPlan46::HETERO6) && ClassPlan46::PANEL_WIDTH == 32U),
                  "active panel copy scoped to four-core K32");
    static_assert(!ClassPlan46::PRUNE_STAGE_WAITS || (ClassPlan46::GROUP == 8U && !ClassPlan46::COMPACT),
                  "sync pruning scoped to noncompact G8");
    static_assert(!ClassPlan46::EXTRA || (ClassPlan46::COMPACT && (ClassPlan46::GROUP == 4U || ClassPlan46::HETERO6)),
                  "extra cleanup needs compact four-core MAT coverage");
