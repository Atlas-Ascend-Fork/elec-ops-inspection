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
#if (defined(__DAV_C220_CUBE__))
class LuCube {
    using A = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using B = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using C = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    static constexpr MatmulConfig cfg = GetMDLConfig(false, false, 2);
    TPipe pipe;
    TCubeTiling tiling;
    MatmulImpl<A, B, C, C, cfg> mm;
    GlobalTensor<float> L, U, D;

  public:
#include "ld_lu_cube_m01.h"

};
#endif
