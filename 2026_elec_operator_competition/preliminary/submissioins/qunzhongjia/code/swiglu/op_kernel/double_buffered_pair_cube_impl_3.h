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
template <typename T, uint32_t TM, uint32_t D, uint32_t N, uint32_t Iterations>
template <uint32_t I, uint32_t CurTM, class A0>
__aicore__ inline __attribute__((always_inline)) void
DoubleBufferedPairCube<T, TM, D, N, Iterations>::ComputeStage584a2d8d(A0& dst)
{
    LocalTensor<T> a1(TPosition::A1, I * Common::A * sizeof(T), CurTM * D);
    LocalTensor<T> b2_0(TPosition::B2, 0U, N * D);
    LocalTensor<T> b2_1(TPosition::B2, N * D * sizeof(T), N * D);
    if constexpr (I == 0U) {
        LocalTensor<T> a2_0(TPosition::A2, 0U, 32U * D);
        LocalTensor<float> c0(TPosition::CO1, 0U, 2U * 32U * N);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        ::SharedPrimitive54529556e1<T, D, N>(a2_0, b2_0, b2_1, c0, dst);
    } else if constexpr (I == 1U) {
        ComputeStageeeb2fd57<I, CurTM>(dst, a1, b2_0, b2_1);
    } else {
        ComputeStage21e20e1c<I, CurTM>(dst, a1, b2_0, b2_1);
    }
}

#endif

}
