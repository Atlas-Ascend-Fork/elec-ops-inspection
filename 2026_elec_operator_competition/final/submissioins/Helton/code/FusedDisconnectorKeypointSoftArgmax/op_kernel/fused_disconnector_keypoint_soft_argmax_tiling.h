#pragma once

#include <cstdint>

struct FusedDisconnectorKeypointSoftArgmaxTilingData {
    int32_t batch;
    int32_t height;
    int32_t width;
    int32_t rows;
    int32_t blocks;
    float prescale;
    float invTau;
    float stride;
};
