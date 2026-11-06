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

struct SafetyBeltViolationGeometryTilingData {
    uint32_t length;        // batch 行数 B（person_boxes [B,4]）
    uint32_t rowsPerCore;   // 每核处理行数（尾核吃余量）
    uint32_t tileRows;      // 单 tile 行数（UB 内循环粒度）
    uint32_t hasThreshold;  // 0=未传入 threshold（按 0.0f 处理，kernel 不得访问其 GM 地址）；1=传入
    float waistRatio;       // 判定参考线比例（attr，合法范围 (0,1]）
    float thresholdValue;   // hasThreshold=0 时使用的缺省值 0.0f
    uint32_t thrIsFp16;     // 1=threshold 张量 dtype 为 fp16（判题环境 fp16 用例实测如此），0=fp32
    uint32_t outIsFp16;     // 1=violation_score 张量 dtype 为 fp16，0=fp32
};

