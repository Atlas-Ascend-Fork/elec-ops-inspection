// Copyright 2026 Qianqiuer
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if defined(TREE_LINE_MIN_DISTANCE_DEVICE_BODY)
#ifndef TREE_LINE_MIN_DISTANCE_DEVICE_CLASS_DEFINED
#define TREE_LINE_MIN_DISTANCE_DEVICE_CLASS_DEFINED
#include "tree_batch.h"

#endif
#else
#ifndef TREE_LINE_MIN_DISTANCE_TILING_DATA_DEFINED
#define TREE_LINE_MIN_DISTANCE_TILING_DATA_DEFINED

#include <cstdint>

struct TreeLineMinDistanceTilingData {
    uint32_t batches;
    uint32_t points;
    uint32_t segments;
    uint32_t paddedSegments;
    uint32_t pointTile;
    uint32_t blocks;
    uint32_t schedule;
    uint32_t segmentLog;
};

#endif
#endif
