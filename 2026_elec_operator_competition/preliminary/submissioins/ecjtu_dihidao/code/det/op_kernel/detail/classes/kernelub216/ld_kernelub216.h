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
template <class DT_A, uint32_t N = 216U> class KernelUB216 {
    static constexpr uint32_t S = 216U;
    static_assert(N >= 209U && N <= S, "UB216 active dimension out of range");
    static constexpr uint32_t Mat = 0U, Col = 186624U, Abs = 187488U, Row = 188352U;
    static constexpr uint32_t Tabs = 189216U, Tmp = 196128U, Res = 196192U, End = 196224U;
    static_assert(S * S * 4U == Col && Res + 32U == End && End <= 196608U, "UB216 layout overflow");
    LocalTensor<float> mat_, colbuf_, absbuf_, rowkbuf_, tmp_, res_;
    LocalTensor<int32_t> tabI_, tabRI_;
    LocalTensor<uint32_t> tabRU_;

#include "ld_kernelub216_m01.h"

  public:
#include "ld_kernelub216_m02.h"

};
#endif
