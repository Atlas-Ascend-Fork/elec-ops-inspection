#pragma once
#include <cstdint>
// Local launch interface only. Official ABI/tiling are pending publication.
struct alignas(8) FdksaPlan {
    uint32_t batch, height, width, stride;
    float tau;
    uint32_t convCores;
};
static_assert(sizeof(FdksaPlan) == 24);
