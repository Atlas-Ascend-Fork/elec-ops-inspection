/*
 * Copyright (c) 2026 Beijing Kuntai Information Technology Co., Ltd.
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

// 数值保持稳定；kernel入口保留字面量供CANN扫描分派宏。
enum SwiGluKey : uint32_t {
    SWIGLU_SCALAR = 0,
    SWIGLU_ROW4 = 1,
    SWIGLU_ROW32 = 2,
    SWIGLU_ROW64 = 3,
    SWIGLU_ROWS13_STATIC = 4,
    SWIGLU_ROW_DYNAMIC = 6,
    SWIGLU_FULL_K = 7,
    SWIGLU_SPLIT_K = 8,
    SWIGLU_ROWS33 = 9,
    SWIGLU_CUBE256 = 10,
    SWIGLU_CUBE128 = 11,
    SWIGLU_RESIDENT_STATIC = 12,
    SWIGLU_ROWS13_DYNAMIC = 13,
};

struct SwiGluTilingData {
    uint32_t rows, dim;
    // Zero groups selects the generic kernel; otherwise every row group
    // contains dim / GATE_RESIDENT_BN column partitions.
    uint32_t residentGroups, residentPanels;
};

constexpr uint32_t GATE_RESIDENT_BM = 64;
constexpr uint32_t GATE_RESIDENT_BN = 128;
constexpr uint32_t GATE_RESIDENT_VM = 32;
// Four READY flags per projection, with one notification per four panels.
constexpr uint32_t GATE_RESIDENT_MAX_PANELS = 16;

constexpr uint32_t TILE_L1_BASEM = 64;
constexpr uint32_t TILE_L1_BASEN = 256;
constexpr uint32_t TILE_L1_BASEK = 256;
constexpr uint32_t TILE_L2_BASEN = 64;
constexpr uint32_t TILE_L2_BASEK = 256;

constexpr uint32_t SMALL13_ROWS = 7;
