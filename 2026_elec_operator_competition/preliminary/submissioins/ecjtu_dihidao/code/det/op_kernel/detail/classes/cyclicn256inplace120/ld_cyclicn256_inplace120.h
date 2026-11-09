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
template <uint32_t GROUP, bool EARLY_TAIL_INDEX = false, bool SCALAR_DMA_RECEIVE = false, bool PUBLISH_VM_TRIM = false,
          bool READY_ALL_TRIM = false, bool LOOKAHEAD_Q3 = false, uint32_t FAC_SLOTS = 2U, bool DISJOINT_SIGNAL = false,
          bool NORMAL_INVERSE_GUARD = false, bool PROVEN_PANEL_BOUNDS = false, bool ACTIVE12_B11 = false>
class CyclicN256Inplace120 {
    friend struct LdetSharedOperations;

    static constexpr bool BRANCHLESS_SEGMENTS = FAC_SLOTS == 3U;
    static_assert(GROUP == 3U, "G3 relay scope only");
    static constexpr bool COMPACT = true, EXTRA = true, FAC_REPAIRED = true, DEFER_BOOKKEEP = true,
                          PRUNE_STAGE_WAITS = false, ACTIVE_PANEL_ROWS = true, FINAL_NO_EXPORT = true, PANEL_ACC = true,
                          PIVOT8_AHEAD = true, CONTIGUOUS_BANDS = true, NORMAL_COPY = false, DIRECT_FAC = true,
                          UB_FINISH64 = true, PANEL_MASK = true, HETERO3 = GROUP == 3U, HETERO5 = GROUP == 5U;
    static constexpr uint32_t PANEL_WIDTH = 32U, PANEL_PITCH = 120U;
    static constexpr uint32_t NMAX = 256, B = PANEL_WIDTH, S = 120U, G = GROUP, P = PANEL_PITCH;
    static_assert(P == S && P == 120U, "inplace odd DataBlock pitch");
    static constexpr bool GUARDED = GROUP == 8U || GROUP == 4U || HETERO5 || HETERO3;
    static_assert(!PIVOT8_AHEAD || B <= 32U, "first retired TMP tile covers all reduction destinations");
    static constexpr uint32_t GUARD_ROWS = GUARDED ? 8U : 0U, GUARD_COEFF = GUARDED ? 1U : 0U;
    static constexpr uint32_t MAT = 0, PANEL = 0, FAC = (NMAX + GUARD_ROWS) * S * 4U + 32U * 4U;
    static_assert(FAC >= MAT + (NMAX + GUARD_ROWS) * S * sizeof(float) + 32U * 4U,
                  "FAC after live aliased matrix and guards");
    #include "../../declarations/ld_common_declarations_06.inc"
    LocalTensor<uint32_t> mu, signalBuf;
    GlobalTensor<float> in, ws, out, relay;
    GlobalTensor<uint32_t> wu, tags;
    bool prefactored3;
    uint32_t n, rank, matrix, blocks, owned, matShift;
    uint32_t rebalanced;
    float sign, prod;
    int32_t ex;
    uint32_t singular;
#include "ld_cyclicn256_inplace120_m01.h"

#include "ld_cyclicn256_inplace120_m02.h"

  public:
#include "ld_cyclicn256_inplace120_m03.h"

  private:
    using LdetPolicy = LdetCyclicPolicy<false, COMPACT, false, false, false, DIRECT_FAC, false, false, false, false,
                                        false, false, GROUP, false, false, false, false, false, false, NORMAL_COPY,
                                        false, PANEL_ACC, false, false, false, false, false, false, false, false>;
};
#endif
