/*
 * Copyright (c) 2026 SwiGlu Authors.
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
struct SwiGluTilingData {
    uint32_t m;
    uint32_t d;
    uint32_t blockDim;
};

struct C8OuterTilingData {
    SwiGluTilingData base;
    uint32_t reserved[5];
    uint64_t va_configs[24];
    uint16_t weight_main_mask[144];
    uint32_t weight_last_k_indices[80];
    uint32_t bias_indices[80];
    uint16_t split_mask_u[112];
    uint16_t split_mask_v[112];
};
static_assert(sizeof(C8OuterTilingData)==1600,"C8 compact transpose tiling layout");

struct C7VectorTilingData {
 SwiGluTilingData base;
 uint32_t reserved[5];
 uint32_t weight_indices[256];
 uint32_t outer_indices[416];
};
static_assert(sizeof(C7VectorTilingData)==2720,"C7 weight index tiling layout");
