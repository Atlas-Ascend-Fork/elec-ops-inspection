/*
 * Copyright (c) 2026 zhiyuan (致远队).
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

// Tiling结构体定义的头文件（W13 压平版）
#pragma once

#include <cstdint>

// ── 三档判据（host 与 kernel 同源；R6 合规形态 = B 的范围连续划分）──────────────
// ── P-D1 AIC 空核探针（默认 0 = W13 提交态不变；置 1 需重编译，kernel/host 同步）──
//   dhltat 先例（jnu_dhltat/batch_to_space）：FORMAT 维携带场景键 + 每组合
//   ASCENDC_TPL_KERNEL_TYPE_SEL(AIC_ONLY/AIV_ONLY)，同源双编译（AIC 实例无 V 代码）。
//   档案警告"KERNEL_TYPE_DECL 维 v52 时代 judge 不可达"——本探针即该轴的再实测。
#ifndef DIAG_FLOOR_AIC_MODE
#define DIAG_FLOOR_AIC_MODE 0
#endif
#define SB_MODE_NORM      0     // 现役 AIV 路径
#define SB_MODE_AIC_EMPTY 1     // AIC 空核探针（AIC_ONLY 编译，入口直接 return）

constexpr uint32_t SB_SMALL_MAX_B = 33u;    // 小点档上界（含）
constexpr uint32_t SB_MID_MAX_B   = 1024u;  // 中点档上界（含）

struct SafetyBeltViolationGeometryTilingData {
    // W3：block_dim / has_threshold_gm 两个死字段已删（kernel 分别用 GetBlockNum()
    //   与 threshold 指针判空）。结构体 16B，整体落在小点那笔 32B tiling DMA 内。
    uint32_t total_b;            // 总 Batch 大小 B                       [字偏移 0]
    float waist_ratio;           // 判定参考线比例 (属性)                  [字偏移 1]
    float threshold;             // 距离容差 (若未传则为 0.0f，占位)       [字偏移 2]
    uint32_t reserved;           // 对齐保留字段                           [字偏移 3]
};
