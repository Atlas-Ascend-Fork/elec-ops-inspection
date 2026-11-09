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
template <class ClassPlan46> class CyclicN256Seven {
    #include "../../declarations/ld_common_declarations_03.inc"
    LocalTensor<uint32_t> mu;
    GlobalTensor<float> in, ws, out;
    GlobalTensor<uint32_t> wu;
    uint32_t n, rank, matrix, blocks, owned;
    float sign, prod;
    int32_t ex;
    uint32_t singular;
#include "ld_cyclicn256_seven_m01.h"

  public:
#include "ld_cyclicn256_seven_m02.h"

  private:
    using LdetPolicy =
        LdetCyclicPolicy<false, ClassPlan46::COMPACT, ClassPlan46::CONTIGUOUS_BANDS, false, false,
                         ClassPlan46::DIRECT_FAC, false, false, false, ClassPlan46::FAC_REPAIRED, false, false,
                         ClassPlan46::GROUP, false, false, ClassPlan46::HETERO6, false, false, false,
                         ClassPlan46::NORMAL_COPY, false, ClassPlan46::PANEL_ACC, ClassPlan46::PANEL_MASK, false, false,
                         PREFIX_STREAM, ClassPlan46::PRUNE_STAGE_WAITS, false, false, false>;
};
#endif
