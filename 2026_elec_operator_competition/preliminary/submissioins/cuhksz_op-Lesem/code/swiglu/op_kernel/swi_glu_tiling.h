/*
 * Copyright (c) 2026 Lesem.
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
struct SwiGluTilingData {
    uint64_t rows;
    uint64_t tasks;
    uint32_t dim;
    uint32_t alignedDim;
    uint32_t tileN;
    uint32_t blocks;
    uint32_t useCube;
    uint32_t tileM;
    AscendC::tiling::TCubeTiling cube;
};

struct SwiGluWorkTiling {
    uint64_t rows; uint64_t tasks;
    uint32_t dim; uint32_t alignedDim; uint32_t tileN; uint32_t blocks;
    uint32_t useCube; uint32_t tileM;
};
static_assert(sizeof(SwiGluWorkTiling)==40);
static_assert(offsetof(SwiGluTilingData,cube)==sizeof(SwiGluWorkTiling));

// Both metadata views share every consumed field, including the idle-worker guard.
static_assert(offsetof(SwiGluTilingData,rows)==offsetof(SwiGluWorkTiling,rows));
static_assert(offsetof(SwiGluTilingData,tasks)==offsetof(SwiGluWorkTiling,tasks));
static_assert(offsetof(SwiGluTilingData,dim)==offsetof(SwiGluWorkTiling,dim));
static_assert(offsetof(SwiGluTilingData,alignedDim)==offsetof(SwiGluWorkTiling,alignedDim));
static_assert(offsetof(SwiGluTilingData,tileN)==offsetof(SwiGluWorkTiling,tileN));
static_assert(offsetof(SwiGluTilingData,blocks)==offsetof(SwiGluWorkTiling,blocks));
static_assert(offsetof(SwiGluTilingData,useCube)==offsetof(SwiGluWorkTiling,useCube));
static_assert(offsetof(SwiGluTilingData,tileM)==offsetof(SwiGluWorkTiling,tileM));
