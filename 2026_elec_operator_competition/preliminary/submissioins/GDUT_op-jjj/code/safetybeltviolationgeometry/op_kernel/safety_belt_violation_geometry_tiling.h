/*
 * Copyright (c) 2026 JJJie999.
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

/**
 * 安全带低挂高用几何判定融合算子 —— Tiling 参数
 *
 * 字段顺序即内存布局，host 侧通过 context->GetTilingData<T>() 直接写入，
 * kernel 侧通过 GET_TILING_DATA_WITH_STRUCT 整体拷入后读取。
 * 请勿随意插入/删除字段（会破坏 host 与 kernel 的二进制一致性）。
 *
 * 说明：输入 dtype 由 ASCENDC_TPL 的 tiling key（DT_PERSON_BOXES）在编译期决定，
 *       因此本结构体不需要 dtype 字段。
 */

namespace sbv {

/**
 * host 与 kernel 共用的常量（单一真源）
 *
 * 刻意放在共享头文件里而不是两边各自定义，避免「同一配置多处真源」的漂移隐患：
 * TILE_ROWS 同时决定 host 侧 tileB 的上界与 kernel 侧 UB 缓冲的容量，
 * 一旦两边取值不一致，host 可能下发超出 kernel 缓冲容量的 tileB 而不报错。
 */
constexpr uint32_t TILE_ROWS       = 512U;   // 单 chunk 行数上限
constexpr uint32_t MAX_BLOCK_COUNT = 4095U;  // DataCopyPad::blockCount 的硬件上限

}  // namespace sbv

struct SafetyBeltViolationGeometryTilingData {
    uint32_t batch;           // 输入 Batch B，题目范围 [1, 4096]         —— kernel: 行上界
    uint32_t hasThreshold;    // 0 = threshold 未传入（按 0.0f），1 = 传入 —— kernel: 是否读该 GM
    uint32_t dtypeFlag;       // 0 = float32，1 = float16                  —— kernel: 运行期分派
    uint32_t thrIsF32;        // threshold 的 dtype：1 = float32，0 = float16 —— kernel: GM 标量读取宽度
    uint32_t outIsF32;        // 输出 dtype：1 = float32，0 = float16 —— kernel: 写回宽度（2B/4B）
    float    waistRatio;      // ATTR waist_ratio，合法范围 (0, 1]         —— kernel: Muls 标量
    float    thresholdValue;  // threshold 未传入时的兜底值，固定 0.0f      —— kernel: Adds 标量
    uint32_t reserved2;       // 预留（当前未消费），维持 32B 布局
};

static_assert(sizeof(SafetyBeltViolationGeometryTilingData) == 32,
              "TilingData 必须是 32 字节紧凑布局（host 与 kernel 共享同一份内存解释）");

// blockCount 字段是 uint16_t 且 DataCopyPad 的上限为 4095；TILE_ROWS 必须落在该范围内，
// 否则 static_cast<uint16_t>(count) 会静默截断。
static_assert(sbv::TILE_ROWS <= sbv::MAX_BLOCK_COUNT,
              "TILE_ROWS 超过 DataCopyPad blockCount 上限（4095），uint16_t 转换会截断");
