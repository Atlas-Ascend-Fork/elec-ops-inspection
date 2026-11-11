/*
 * Copyright (c) 2026 Qianqiuer.
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
#include <cstddef>
#include <cstdint>

struct SplineResidentTilingData {
    uint32_t nodes;
    uint32_t edges;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t mean;
    uint32_t owners;
    uint32_t paddedInput;
    uint32_t ownerShift;
};

struct SplineConvTilingData {
    uint32_t nodes;
    uint32_t edges;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t mean;
    uint32_t owners;
    uint32_t paddedInput;
    uint32_t ownerShift;
    uint32_t dim;
    uint32_t degree;
    uint32_t hasRoot;
    uint32_t hasBias;
    uint32_t nodeTile;
    uint32_t edgeStride;
    uint32_t blocks;
    uint32_t schedule;
    uint32_t kernelSize[3];
    uint32_t open[3];
};
static_assert(offsetof(SplineConvTilingData, dim) == sizeof(SplineResidentTilingData),
              "Resident tiling must be the full record's 32-byte prefix");
