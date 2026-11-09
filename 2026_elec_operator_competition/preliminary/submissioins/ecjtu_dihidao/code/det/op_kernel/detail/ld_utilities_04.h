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
template <class ClassPlan46, bool PREFIX_STREAM>
using LdetPriorityPolicy54 =
    LdetCyclicPolicy<ClassPlan46::ACTIVE_PANEL_ROWS, ClassPlan46::COMPACT, ClassPlan46::CONTIGUOUS_BANDS,
                     ClassPlan46::DEFER_BOOKKEEP, false, ClassPlan46::DIRECT_FAC, false, ClassPlan46::EARLY_TAIL_INDEX,
                     false, ClassPlan46::FAC_REPAIRED, false, false, ClassPlan46::GROUP, ClassPlan46::HETERO3,
                     ClassPlan46::HETERO5, false, false, false, false, ClassPlan46::NORMAL_COPY, false,
                     ClassPlan46::PANEL_ACC, ClassPlan46::PANEL_MASK, ClassPlan46::PHASE_TRIM,
                     ClassPlan46::PIVOT8_AHEAD, PREFIX_STREAM, ClassPlan46::PRUNE_STAGE_WAITS, false, false, false>;
#endif
