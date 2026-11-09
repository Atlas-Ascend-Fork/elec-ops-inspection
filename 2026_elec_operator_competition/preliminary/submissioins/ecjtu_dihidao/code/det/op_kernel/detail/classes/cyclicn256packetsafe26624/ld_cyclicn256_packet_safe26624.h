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
template <class ClassPlan46> class CyclicN256PacketSafe26624 {
    #include "../../declarations/ld_common_declarations_03.inc"
    LocalTensor<uint32_t> mu, prefixReady, fu;
    static_assert(END % 32U == 0U && END + 32U <= 196352U, "prefix ready immutable UB");
    static constexpr uint32_t CONSUMER_COEFF =
        ClassPlan46::THIN_H6 ? ((END + 32U + 511U) / 512U) * 512U + 128U : PANEL + 256U;
    static constexpr uint32_t ACTIVE_FAC = ClassPlan46::THIN_H6 ? FAC : END + 32U;
    static_assert(ClassPlan46::THIN_H6 ? CONSUMER_COEFF >= END + 32U
                                       : CONSUMER_COEFF + (NMAX + GUARD_ROWS) * S * 4U < BRC,
                  "consumer coefficient live regions");
    static_assert(CONSUMER_COEFF + (NMAX + GUARD_ROWS) * S * 4U <= 196352U &&
                      ACTIVE_FAC + (B * NMAX + META_WORDS) * 4U <= 196352U,
                  "coefficient and relocated FAC fit UB");
    static_assert(B == 32U && ClassPlan46::COMPACT && ClassPlan46::HETERO6 && ClassPlan46::DIRECT_FAC,
                  "26624 packet scope");
    static_assert(!ClassPlan46::THIN_H6 || END + 128U <= CONSUMER_COEFF, "immutable tickets fit unchanged owner gap");

    GlobalTensor<float> in, ws, out;
    GlobalTensor<uint32_t> wu;
    uint32_t n, rank, matrix, blocks, owned;
    float sign, prod;
    int32_t ex;
    uint32_t singular;
#include "ld_cyclicn256_packet_safe26624_m01.h"

#include "ld_cyclicn256_packet_safe26624_m02.h"

  public:
#include "ld_cyclicn256_packet_safe26624_m03.h"

  private:
    using LdetPolicy = LdetCyclicPolicy<false, ClassPlan46::COMPACT, false, false, false, false, false, false, false,
                                        false, false, false, ClassPlan46::GROUP, false, false, ClassPlan46::HETERO6,
                                        false, false, false, ClassPlan46::NORMAL_COPY, false, ClassPlan46::PANEL_ACC,
                                        false, false, false, false, false, false, false, false>;
};
#endif
