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

#include "tree_common.h"
#include "tree_pairs.h"
#include "tree_pairs_1024.h"
#include "tree_edges.h"
#include "tree_tiny.h"
#include "tree_short.h"
#include "tree_line_min_distance_tiling.h"
#include "tiling_key_tree_line_min_distance.h"

template <typename DT_TREE_POINTS, uint32_t PATH>
__global__ __aicore__ void tree_line_min_distance(
    GM_ADDR tree_points, GM_ADDR line_segments,
    GM_ADDR min_distance, GM_ADDR nearest_segment_idx,
    GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TreeLineMinDistanceTilingData);
    (void)workspace;
    if constexpr (PATH == 7 || PATH == 9) {
        elec::tree_tiny<DT_TREE_POINTS, PATH == 9>(tiling, tree_points, line_segments, min_distance, nearest_segment_idx);
        return;
    }
    const auto* raw_cfg = reinterpret_cast<const __gm__ uint64_t*>(tiling);
    elec::Config c;
    c.batch = raw_cfg[0];
    c.m = raw_cfg[1];
    c.n = raw_cfg[2];
    c.k = raw_cfg[3];
    if constexpr (PATH == 8) {
        elec::TreePairs1024<DT_TREE_POINTS, 64> implementation;
        implementation.run(c, tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH == 5 || PATH == 6) {
        constexpr uint32_t width = (PATH == 5) ? 8 : 32;
        constexpr uint32_t tile = elec::tree_short_point_tile(sizeof(DT_TREE_POINTS), width);
        elec::TreeShort<DT_TREE_POINTS, width, tile> implementation;
        implementation.run(c, tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH == 4) {
        elec::tree_one_point<DT_TREE_POINTS>(c, tree_points, line_segments, min_distance, nearest_segment_idx);
    } else if constexpr (PATH >= 2) {
        elec::TreePairs<DT_TREE_POINTS, (PATH == 3 || PATH >= 11) ? 64 : 32, (PATH == 10 || PATH == 11 || PATH == 13), (PATH == 12 || PATH == 13)> implementation;
        implementation.run(c, tree_points, line_segments, min_distance, nearest_segment_idx);
    } else {
        elec::tree_one_segment<DT_TREE_POINTS>(c, tree_points, line_segments, min_distance, nearest_segment_idx);
    }
}
