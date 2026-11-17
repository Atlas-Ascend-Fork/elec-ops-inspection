#pragma once

#include <cstdint>

struct FusedDisconnectorKeypointSoftArgmaxTilingData {
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t stride;
    float tau;
    uint32_t tileHeight;
    uint32_t tileWidth;
    uint32_t tilesPerBatch;
    uint32_t blocks;
};

constexpr uint32_t FUSED_KEYPOINT_CHANNELS = 4;
constexpr uint32_t FUSED_KEYPOINT_RECORD_FLOATS = 8;
constexpr uint32_t FUSED_KEYPOINT_TILE_RECORD_BYTES =
    FUSED_KEYPOINT_CHANNELS * FUSED_KEYPOINT_RECORD_FLOATS * sizeof(float);
