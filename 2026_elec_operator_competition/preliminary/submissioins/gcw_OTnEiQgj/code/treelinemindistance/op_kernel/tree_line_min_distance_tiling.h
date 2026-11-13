/*
 * Copyright (c) 2026 gcw_OTnEiQgj.
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

namespace elec {
enum DType : uint32_t { F32 = 0, F16 = 1, BF16 = 2, I32 = 3, I64 = 4 };

struct Config {
    uint64_t batch = 1, m = 1, n = 1, k = 1, edges = 0, dim = 2;
    uint64_t rhs_batch_stride = 0;
    uint32_t dtype = F32, flags = 0, degree = 1;
    uint32_t kernel_size[3] = {2, 2, 2};
    uint32_t open_spline[3] = {1, 1, 1};
    uint32_t max_sweeps = 100;
    float waist_ratio = 1.0f, threshold = 0.0f;
    float atol = 0.0f, rtol = 0.0f, tolerance = 1.0e-7f;
};

struct TemplateTilingData {
    Config config;
    uint64_t scratch_bytes = 0;
    uint32_t input_types[6] = {};
    uint32_t output_types[2] = {};
};

constexpr uint64_t tree_short_point_tile(uint64_t /*element_bytes*/, uint64_t width) {
    return width > 16 ? 4 : 8;
}

constexpr uint64_t tree_short_align(uint64_t bytes) {
    return (bytes + 31) / 32 * 32;
}

inline uint64_t tree_short_ub_bytes(uint64_t element_bytes, uint64_t width, uint64_t tile) {
    const uint64_t pairs = width * tile;
    const uint64_t stage = 6 * width + 3 * tile;
    const uint64_t work_stride = (pairs & 64) ? pairs : (pairs + 64);
    const uint64_t mask_bytes = (pairs / 8 < 32) ? 32 : (pairs / 8);
    return tree_short_align(stage * element_bytes) +
           tree_short_align(stage * sizeof(float)) +
           tree_short_align(12 * work_stride * sizeof(float)) +
           tree_short_align((3 * pairs + tile) * sizeof(int32_t)) +
           160 * sizeof(float) +
           tree_short_align(mask_bytes);
}
}  // namespace elec

using TreeLineMinDistanceTilingData = elec::TemplateTilingData;
