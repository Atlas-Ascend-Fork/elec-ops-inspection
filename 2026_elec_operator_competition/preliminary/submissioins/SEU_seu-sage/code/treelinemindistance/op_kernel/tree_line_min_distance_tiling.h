// Copyright 2026 Electrical Engineering SIG - CANN Community
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

#pragma once

#include <cstdint>

struct TreeLineMinDistanceTilingData {
    uint32_t batches;
    uint32_t pointsPerBatch;
    uint32_t segmentsPerBatch;
    uint32_t coreDivisionMultiplier;
    uint32_t coreDivisionShift;
    uint32_t batchDivisionMultiplier;
    uint32_t batchDivisionShift;
    uint32_t totalPoints;
    uint32_t blockDim;
};

constexpr uint32_t kTreeLineMaxSegments = 1024;
constexpr uint32_t kTreeLineMaxPoints = 0x7FFFFFFFU;
constexpr uint32_t kPointMajorSegmentLimit = 8;
constexpr uint32_t kPointMajorLaneTarget = 512;

// The point-major path walks one segment at a time and updates the running
// minimum through a compare/select ladder; the segment-major matrix path
// reduces a whole segment tile in one pass. Tiling and kernel both pick
// point-major when segs * kPointMajorLaneTarget <= totalPoints, because the
// ladder only amortizes once the point count fills wide vectors.
