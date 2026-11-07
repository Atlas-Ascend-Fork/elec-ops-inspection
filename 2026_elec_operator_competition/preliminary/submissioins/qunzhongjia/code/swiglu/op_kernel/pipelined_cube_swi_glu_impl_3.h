/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 黎何.
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

namespace optiling {
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T, uint32_t D, uint32_t TM, uint32_t Cores, uint32_t Iterations, uint32_t TileN>
template <uint32_t I, uint32_t CurTM, class A0, class A1, class A2, class A3>
__aicore__ inline __attribute__((always_inline)) void
PipelinedCubeSwiGlu<T, D, TM, Cores, Iterations, TileN>::EpilogueStage224a0a0c(A0& row, A1& begin, A2& input,
                                                                               A3& out)
{
    constexpr uint32_t ubSlot = I % InSlots;
    constexpr event_t vEv =
        ubSlot == 0U ? EVENT_ID0 : (ubSlot == 1U ? EVENT_ID1 : (ubSlot == 2U ? EVENT_ID2 : EVENT_ID3));
    constexpr uint32_t CurNR = CurTM / 2U;
    constexpr uint32_t CurCap = CurNR * N;
    SetFlag<HardEvent::V_MTE3>(vEv);
    ::SharedPrefetchTile<I, Iterations, InSlots, Plane, Cap, N, (Groups == 4U) ? 24U : 32U, false, false>(
        input, slots_, core_);
    WaitFlag<HardEvent::V_MTE3>(vEv);
    DataCopyParams op{CurNR, N / 16U, 0, (D - N) / 16U};
    ::SharedPrimitiveac2daf19f8<D, TM, Iterations, N, InSlots, OutSlots, I, CurTM>(y_, col_, begin, op, out,
                                                                                   row);
}

#endif

}
