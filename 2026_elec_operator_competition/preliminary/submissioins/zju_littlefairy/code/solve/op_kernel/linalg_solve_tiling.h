/*
 * Copyright (c) 2026 LittleFairy.
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
struct LinalgSolveTilingData {
    uint32_t n, k, batches, rows, blocks, broadcastB, teams;
};

struct LinalgSolveCubeTilingData {
    LinalgSolveTilingData base;
    uint32_t cubeWords[128];
    uint32_t halfWords[128];
    uint32_t systemWorkspaceBytes;
};

// CT, inverse, negative H, normalized A; two B64 blocks, RHS buffers, reduction slots, and local-Newton scratch.
constexpr uint32_t kOrthogonalWorkspaceFloats=4*256*256+2*64*64+4*256*32+256+3*64*64+4096;

// Narrow reprojected QR tail and six signed base-128 residual weights.
constexpr uint32_t kNarrowOrthogonalWorkspaceFloats=538368;
