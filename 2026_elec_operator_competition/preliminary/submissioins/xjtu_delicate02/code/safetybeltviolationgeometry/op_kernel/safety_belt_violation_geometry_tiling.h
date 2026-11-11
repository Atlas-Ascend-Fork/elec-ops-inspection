// Copyright 2026 Delicate02
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
// Legal batch capacity and the Float wide-family lower owner bound are shared
// by validation, tiling and the matching static device address calculation.
static constexpr uint32_t kGeometryMaxBatch=4096U;
static constexpr uint32_t kGeometryFloatWideOwners=kGeometryMaxBatch/512U;
static constexpr uint32_t kGeometryFoldMinLaunch=4U;
struct SafetyBeltViolationGeometryTilingData {
 uint32_t batch;
 uint32_t perCore;
 uint32_t hasThreshold;
 uint32_t thresholdIsHalf;
 float defaultThreshold;
 float waistRatio;
};
