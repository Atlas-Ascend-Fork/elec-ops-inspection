/*
 * Copyright (c) 2026 李冠杰.
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

// TilingKey模板定义的头文件（R14：新增 BATCH_SPLIT，实例追加在末尾）
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// 四个模板参数：dtype + 编译期布局（USE_PAD）+ 编译期标量专用（SCALAR_ONLY）
//              + 编译期按 batch 分核（BATCH_SPLIT）
// 参数声明顺序在原三项之后追加 BATCH_SPLIT，前三个参数与 R13-A 完全一致。
// BATCH_SPLIT 只对 (FP32, USE_PAD=0, SCALAR_ONLY=0) 一个组合有意义，其余组合不生成。
ASCENDC_TPL_ARGS_DECL(FireCandidateDecodeFusion,
    ASCENDC_TPL_DATATYPE_DECL(DT_RAW_PRED, C_DT_FLOAT, C_DT_FLOAT16),
    ASCENDC_TPL_BOOL_DECL(USE_PAD, 0, 1),
    ASCENDC_TPL_BOOL_DECL(SCALAR_ONLY, 0, 1),
    ASCENDC_TPL_BOOL_DECL(BATCH_SPLIT, 0, 1),
);

// 顺序（前五项与 R13-A 完全相同，BATCH_SPLIT 实例追加在末尾）：
//   1. float / 0 / 0 / 0 : key0    （FP32 flat 常规）
//   2. float / 1 / 0 / 0 : key256  （FP32 pad）
//   3. half  / 0 / 0 / 0 : key1    （FP16 flat）
//   4. half  / 1 / 0 / 0 : key257  （FP16 pad）
//   5. float / 0 / 1 / 0 : key512  （FP32 flat 标量专用）
//   6. float / 0 / 0 / 1 : key1024 （FP32 flat 常规 + 按 batch 分核）
// 只生成 FP32 flat normal 的一个 BATCH_SPLIT 组合，不生成任何非法组合。
// 声明顺序不等于实际产物顺序，必须在安装产物上核验入口偏移/尺寸/重定位。
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_RAW_PRED, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(USE_PAD, 0),
        ASCENDC_TPL_BOOL_SEL(SCALAR_ONLY, 0),
        ASCENDC_TPL_BOOL_SEL(BATCH_SPLIT, 0),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_RAW_PRED, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(USE_PAD, 1),
        ASCENDC_TPL_BOOL_SEL(SCALAR_ONLY, 0),
        ASCENDC_TPL_BOOL_SEL(BATCH_SPLIT, 0),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_RAW_PRED, C_DT_FLOAT16),
        ASCENDC_TPL_BOOL_SEL(USE_PAD, 0),
        ASCENDC_TPL_BOOL_SEL(SCALAR_ONLY, 0),
        ASCENDC_TPL_BOOL_SEL(BATCH_SPLIT, 0),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_RAW_PRED, C_DT_FLOAT16),
        ASCENDC_TPL_BOOL_SEL(USE_PAD, 1),
        ASCENDC_TPL_BOOL_SEL(SCALAR_ONLY, 0),
        ASCENDC_TPL_BOOL_SEL(BATCH_SPLIT, 0),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_RAW_PRED, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(USE_PAD, 0),
        ASCENDC_TPL_BOOL_SEL(SCALAR_ONLY, 1),
        ASCENDC_TPL_BOOL_SEL(BATCH_SPLIT, 0),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_RAW_PRED, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(USE_PAD, 0),
        ASCENDC_TPL_BOOL_SEL(SCALAR_ONLY, 0),
        ASCENDC_TPL_BOOL_SEL(BATCH_SPLIT, 1),
    ),
);
