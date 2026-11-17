#pragma once
#include <cstdint>
struct FusedDisconnectorKeypointSoftArgmaxTilingData {
    uint32_t batch, height, width, stride;
    float tau;
};
static_assert(sizeof(FusedDisconnectorKeypointSoftArgmaxTilingData)==20);
