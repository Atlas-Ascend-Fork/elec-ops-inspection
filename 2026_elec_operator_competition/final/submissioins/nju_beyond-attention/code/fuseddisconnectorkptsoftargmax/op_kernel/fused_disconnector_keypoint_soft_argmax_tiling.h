/*
 * Copyright (c) 2026 Beyond Attention. All rights reserved.
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
struct FusedDisconnectorKeypointSoftArgmaxTilingData {
    uint32_t batch, height, width, rows;
    float tau;
    uint32_t stride;
    uint32_t groups;
    int32_t cube_status;
    uint32_t cube_words[50];
};
static_assert(sizeof(FusedDisconnectorKeypointSoftArgmaxTilingData) == 232, "internal tiling ABI");
static_assert(offsetof(FusedDisconnectorKeypointSoftArgmaxTilingData, cube_words) == 32, "TCube words alignment");
constexpr uint32_t kCubeTileN = 192;
constexpr uint32_t kCubeBaseM = 16;
constexpr uint32_t kCubeBaseK = 32;
constexpr uint32_t kProjectionRows = 48;
constexpr uint32_t kProjectionLiveRows = 36;
constexpr uint32_t kProjectionChannels = 32;
constexpr uint32_t kProjectionWeightFloats = kProjectionRows * kProjectionChannels;
constexpr uint32_t kShiftRows = 8;
static_assert((72 * 96) % kCubeTileN == 0);
static_assert(kProjectionRows % kCubeBaseM == 0);

constexpr uint64_t PROJECTION_PACK_UB_BYTES = 11264, PROJECTION_DECODE_UB_BYTES = 138368;
constexpr uint64_t ProjectionBytes(uint32_t b, uint32_t h, uint32_t w) {
    return (uint64_t(kProjectionWeightFloats) + uint64_t(b) * kProjectionRows * h * w) * 4;
}
constexpr uint64_t HeatmapBytes(uint32_t b, uint32_t h, uint32_t w) { return uint64_t(b) * 4 * h * w * 4; }
enum CubeStatus : int32_t {
    CUBE_OK = 0,
    CUBE_NOT_SELECTED = 1,
    CUBE_NO_GROUP = 2,
    CUBE_CAPACITY = 3,
    CUBE_SETUP = 4,
    CUBE_GET_TILING = 5,
    CUBE_TMP = 6,
    CUBE_RECORD = 7
};
constexpr uint64_t A2_L1_BYTES = 55296, A2_L0A_BYTES = 36864, A2_L0B_BYTES = 18432, A2_L0C_BYTES = 2048;
constexpr uint64_t A2_PACK_UB_BYTES = 29312, A2_H48_DECODE_UB_BYTES = 110720, A2_H64_DECODE_UB_BYTES = 147584;
constexpr uint64_t A2FramesBytes(uint32_t b, uint32_t h, uint32_t w) { return uint64_t(b) * 32 * (h + 2) * w * 4; }
constexpr uint64_t A2HeatmapOffset(uint32_t b, uint32_t h, uint32_t w) { return A2FramesBytes(b, h, w) + 18432; }
#define KEYPOINT_ROWS(B, H, W)                                                                                         \
    (((H) == 48 && (W) == 64 && (B) >= 1 && (B) <= 4)   ? ((B) == 1   ? 2u                                             \
                                                           : (B) == 2 ? 3u                                             \
                                                           : (B) == 3 ? 4u                                             \
                                                                      : 6u)                                            \
     : ((H) == 64 && (W) == 64 && (B) >= 1 && (B) <= 4) ? ((B) == 1   ? 2u                                             \
                                                           : (B) == 2 ? 4u                                             \
                                                                      : 8u)                                            \
     : ((H) == 72 && (W) == 96 && (B) >= 1 && (B) <= 4) ? ((B) == 1   ? 2u                                             \
                                                           : (B) == 2 ? 4u                                             \
                                                           : (B) == 3 ? 6u                                             \
                                                                      : 8u)                                            \
                                                        : 8u)
