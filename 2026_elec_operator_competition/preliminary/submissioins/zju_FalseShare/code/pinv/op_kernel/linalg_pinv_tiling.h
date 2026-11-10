/*
 * Copyright (c) 2026 FalseShare contributors.
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
#define PINV_TINY_MIN_BLOCKS 8
#define PINV_TINY_CAPACITY 16
#define PINV_TINY_MIN_ORDER 9
#include <cstdint>
#define PINV_COLUMN_CAPACITY 64
#define PINV_COLUMN_MAX_SHARDS 8
#define PINV_COLUMN_MAX_WIDTH 8
#define PINV_CUBE_CORES_PER_BATCH 4
#define PINV_ROW_BLOCK_CAPACITY 256
#define PINV_ROW_BLOCK_PANEL 64
#define PINV_ROW_BLOCK_WORKSPACE_FLOATS (5*PINV_ROW_BLOCK_CAPACITY*PINV_ROW_BLOCK_CAPACITY + 5*PINV_ROW_BLOCK_PANEL*PINV_ROW_BLOCK_PANEL + PINV_ROW_BLOCK_CAPACITY*PINV_ROW_BLOCK_PANEL + 512 + 8192)
struct LinalgPinvTilingData {
    uint32_t m, n, batches, columns, rows, stride, resident, parallel;
    float atol, rtol;
};

struct LinalgPinvCubeTilingData {
    LinalgPinvTilingData base;
    // Launch metadata only; the device entry already receives the user pointer.
    uint32_t systemWorkspaceBytes;
};

#define PINV_TINY_TABLES 1
#define PINV_TINY_PACKED_PAYLOAD 1
struct LinalgPinvTinyTilingData {
    LinalgPinvTilingData base;
    uint32_t padding[6];
    uint32_t packed[512];
};
struct LinalgPinvMaxTilingData {
    uint32_t words[(sizeof(LinalgPinvTinyTilingData)>sizeof(LinalgPinvCubeTilingData)?sizeof(LinalgPinvTinyTilingData):sizeof(LinalgPinvCubeTilingData))/4];
};

static_assert(sizeof(LinalgPinvTinyTilingData)==2112);
