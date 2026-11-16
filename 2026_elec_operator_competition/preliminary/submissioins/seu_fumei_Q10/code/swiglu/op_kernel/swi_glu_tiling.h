#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// Static kernels receive dimensions through template selection, not tiling loads.
// Keep one aligned word for framework registration; no kernel reads this payload.
struct alignas(8) SwiGluTilingData {
    uint64_t reserved;
};
static_assert(sizeof(SwiGluTilingData) == 8, "Tiling payload must be 8 bytes");
