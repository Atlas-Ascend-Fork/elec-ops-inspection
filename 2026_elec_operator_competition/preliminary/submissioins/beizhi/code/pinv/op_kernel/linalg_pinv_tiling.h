/*
 * Copyright (c) 2026 Beizhi contributors.
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
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct LinalgPinvTilingData {
    uint32_t m, n, p, q, pp, qp, batch, cached;
    float atol, rtol;
    uint32_t group, groups, cols;
};

struct LinalgPinvCubeTilingData {
    LinalgPinvTilingData base;
    AscendC::tiling::TCubeTiling cube;
};
