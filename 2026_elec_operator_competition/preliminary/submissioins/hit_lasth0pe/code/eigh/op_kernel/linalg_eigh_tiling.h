/*
 * Copyright (c) 2026 lasth0pe.
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
#pragma once
#include <cstdint>
struct EighCorePlan {
  uint32_t a_offset, val_offset, p;
  uint32_t col_start, col_end, my_cols;
  uint32_t my_blk, plo_b, bp_team, subP_b;
};
struct LinalgEighTilingData {
  uint32_t n,total_batch,uplo,used_cores,cores_per_batch,ub_floats;
  uint32_t stride,max_rows,p_stride,pp_slot,meta_slot,meta_cap;
  int32_t M,M_leaf;
  uint32_t max_Lb,blk_slot,leaf_gm_off;
  int32_t bounds[9],leaf_bounds[33];
  EighCorePlan cores[32];
};
