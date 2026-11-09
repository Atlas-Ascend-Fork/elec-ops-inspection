/*
 * Copyright (c) 2026 dihidao.
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
// Tiling 结构体定义（host 写入，kernel 只读消费，单一事实来源）
#pragma once

#include <cstdint>

struct LinalgDetTilingData {
    uint32_t batchCount;  // batch 总数（输入前导维乘积，A 为 [..., n, n]）
    uint32_t n;           // 方阵阶数，1 <= n <= 512
    uint32_t stride;      // workspace 行距 = 向上 8 元素（32B）对齐的 n
    uint32_t matPerCore;  // 每核连续处理的矩阵个数（尾核吃余量）
    uint32_t isLarge;     // 0 = UB 内原地 LU（n <= 208）；1 = 单核 GM 分块 LU；2 = 多核并行分块 LU
    uint32_t groupSize;   // isLarge=2 时每矩阵协作核数（blockDim = batchCount * groupSize）
};

// Old routes still consume the unchanged 24-byte prefix. Mixed keys20/21/22 use the tail.
struct LinalgDetTilingDataFull : public LinalgDetTilingData {
    uint32_t pad[10];
    uint64_t cubeTiling[64];
    uint32_t block4Offsets[352];
    float block4Signs[64];
};
static_assert(sizeof(LinalgDetTilingData)==24,"old tiling prefix changed");
static_assert(sizeof(LinalgDetTilingDataFull)==2240,"private block4 UB-chain layout");
