/*
 * Copyright (c) 2026 FalseShare Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Apache License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct FusedDisconnectorKeypointSoftArgmaxTilingData {
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t stride;
    float tau;
    uint32_t parts;
    uint32_t tileRows;
    uint32_t blocks;
};

// Geometry-only offsets: 4 groups, 16 tap slots, 8 channel lanes.
struct FdkWeightMapTilingData {
    FusedDisconnectorKeypointSoftArgmaxTilingData base;
    uint32_t weightOffsets[576];
    uint16_t joinCoeffs[768];
};
