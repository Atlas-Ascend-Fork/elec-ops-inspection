/*
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
// Tiling结构体定义的头文件
#ifndef LINALG_EIGH_TILING_H
#define LINALG_EIGH_TILING_H
#include <cstdint>
struct LinalgEighTilingData {
    uint32_t n, stride, uplo, active_cores;
    uint64_t batch_size;
    uint32_t num_groups, group_lanes, rows_per_lane, vec_rows_per_lane, solve_mode, wsOffsetFloats;
};
#endif
