// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct FusedDisconnectorKeypointSoftArgmaxTilingData {
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t tileCount;
    uint32_t coreCount;
    float tau;
    float stride;
};
