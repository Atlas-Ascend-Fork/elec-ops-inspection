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
#include <cstdint>
#include <cstddef>
#include "kernel_tiling/kernel_tiling.h"

struct SwiGluDirectTilingData {
    uint32_t path;
    uint32_t reserved0;
    uint64_t m;
    uint64_t d;
    uint32_t dAlign;
    uint32_t tileN;
    uint32_t usedCoreNum;
    uint32_t taskM;
    uint32_t taskNum;
    uint32_t taskN;
};

struct SwiGluTilingData {
    SwiGluDirectTilingData dimensions;
    AscendC::tiling::TCubeTiling cubeTilingData;
#if defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL
    alignas(32) uint32_t gatherOffsetsA[1024];
#endif
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 11265UL || TILING_KEY_VAR == 12033UL)
    alignas(32) uint32_t gatherOffsetsB[2000];
#endif
};

static_assert(offsetof(SwiGluTilingData, dimensions) == 0, "Tiling dimensions must be the initial member");
static_assert(sizeof(SwiGluDirectTilingData) == offsetof(SwiGluTilingData, cubeTilingData),
              "Direct tiling must match prefix");

struct SwiGluTinyTilingData {
    uint32_t path;
    uint32_t reserved0;
    uint64_t m;
    uint64_t d;
};

static_assert(sizeof(SwiGluTinyTilingData) == 24, "Tiny tiling prefix size");
