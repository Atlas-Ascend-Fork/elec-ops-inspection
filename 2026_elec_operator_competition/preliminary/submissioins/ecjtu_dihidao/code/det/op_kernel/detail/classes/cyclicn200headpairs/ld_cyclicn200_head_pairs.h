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
template <class ClassPlan46> class CyclicN200HeadPairs {
    #include "../../declarations/ld_common_declarations_01.inc"
#include "ld_cyclicn200_head_pairs_m01.h"

    #include "../../declarations/ld_common_declarations_05.inc"
    static constexpr uint32_t GM_WORDS = B * NMAX + 2U * META_WORDS;
    #include "../../declarations/ld_common_declarations_04.inc"
#include "ld_cyclicn200_head_pairs_m02.h"

  public:
#include "ld_cyclicn200_head_pairs_m03.h"

  private:
    using LdetPolicy = LdetCyclicPolicy<
        false, ClassPlan46::COMPACT, ClassPlan46::CONTIGUOUS_BANDS, ClassPlan46::DEFER_BOOKKEEP, ClassPlan46::DIM,
        false, ClassPlan46::EARLY_FAC, false, ClassPlan46::EXTRA, ClassPlan46::FAC_REPAIRED, ClassPlan46::FIXED_MERGE,
        ClassPlan46::FIXED_REDUCE, ClassPlan46::GROUP, false, false, false, ClassPlan46::INPLACE_PANEL,
        ClassPlan46::LOCAL_CUT, ClassPlan46::LOCAL_TAIL, false, ClassPlan46::PAD_INPUT, ClassPlan46::PANEL_ACC, false,
        false, false, ClassPlan46::PREFIX_STREAM, false, ClassPlan46::SPLIT_OWNERS, false, ClassPlan46::WORD_SELECT>;
};
#endif
