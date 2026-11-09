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
static_assert(!ClassPlan46::PREFIX_STREAM ||
                      ((ClassPlan46::GROUP == 3U || ClassPlan46::SPLIT_OWNERS) && ClassPlan46::DIM == 200U &&
                       B == 32U && ClassPlan46::LOCAL_TAIL &&
                       (ClassPlan46::LOCAL_CUT == 128U ||
                        (ClassPlan46::SPLIT_OWNERS && ClassPlan46::LOCAL_CUT == 160U)) &&
                       ClassPlan46::CONTIGUOUS_BANDS && ClassPlan46::INPLACE_PANEL && ClassPlan46::FAC_REPAIRED),
                  "prefix stream bounded to n200 B3 key62");
    static_assert(!ClassPlan46::SPLIT_OWNERS ||
                      (((ClassPlan46::GROUP == 5U && ClassPlan46::LOCAL_CUT == 128U) ||
                        (ClassPlan46::GROUP == 6U && ClassPlan46::LOCAL_CUT == 160U)) &&
                       ClassPlan46::PREFIX_STREAM && ClassPlan46::DIM == 200U && B == 32U && ClassPlan46::LOCAL_TAIL),
                  "four32 Owners plus72 tail only");
    #include "ld_common_declarations_11.inc"
