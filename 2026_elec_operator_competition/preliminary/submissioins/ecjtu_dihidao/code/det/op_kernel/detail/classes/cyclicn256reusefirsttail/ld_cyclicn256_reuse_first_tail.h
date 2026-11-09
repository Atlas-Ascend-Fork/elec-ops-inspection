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
template <class ClassPlan46> class CyclicN256ReuseFirstTail {
    friend struct LdetSharedOperations;

    static_assert(!ClassPlan46::THIN_H6 ||
                      (ClassPlan46::HETERO6 && ClassPlan46::GROUP == 6U && ClassPlan46::PANEL_PITCH == 104U),
                  "singlepanelinplaceG7");
    static_assert(!ClassPlan46::HETERO6 ||
                      (ClassPlan46::GROUP == 6U && ClassPlan46::PANEL_WIDTH == 32U && ClassPlan46::CONTIGUOUS_BANDS &&
                       ClassPlan46::UB_FINISH64 && ClassPlan46::PANEL_MASK),
                  "G7sixowners_tail64");
    #include "../../declarations/ld_common_declarations_09.inc"
    static constexpr uint32_t NMAX = 256, B = ClassPlan46::PANEL_WIDTH,
                              S = ClassPlan46::THIN_H6
                                      ? 104U
                                      : (ClassPlan46::GROUP == 2U
                                             ? 136U
                                             : ((ClassPlan46::GROUP == 8U && ClassPlan46::PRUNE_STAGE_WAITS) ? 40U
                                                                                                             : 64U)),
                              G = ClassPlan46::GROUP, P = ClassPlan46::PANEL_PITCH;
    static_assert((P == 32U || P == 40U || P == 104U) || (ClassPlan46::GROUP == 2U && B == 16U && P == 16U),
                  "panel pitch must preserve eight-float alignment");
    #include "../../declarations/ld_common_declarations_08.inc"
    LocalTensor<uint32_t> mu;
    GlobalTensor<float> in, ws, out;
    GlobalTensor<uint32_t> wu;
    uint32_t n, rank, matrix, blocks, owned;
    float sign, prod;
    int32_t ex;
    uint32_t singular;
#include "ld_cyclicn256_reuse_first_tail_m01.h"

  public:
#include "ld_cyclicn256_reuse_first_tail_m02.h"

  private:
    using LdetPolicy =
        LdetCyclicPolicy<false, ClassPlan46::COMPACT, false, false, false, ClassPlan46::DIRECT_FAC, false, false, false,
                         false, false, false, ClassPlan46::GROUP, false, false, ClassPlan46::HETERO6, false, false,
                         false, ClassPlan46::NORMAL_COPY, false, ClassPlan46::PANEL_ACC, false, false, false,
                         PREFIX_STREAM, false, false, false, false>;
};
#endif
