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

// TilingKey 模板定义（W13 压平版 · TPL 绑定修复，回 W12 已验证 2 维外壳）
//   机制（PM 考古 + /root/sbops_w12_snatch 生成物实证）：opbuild 的扫描器
//   kernel_info_infer.py 抓 @@ASCENDC_TPL_ARGS_DECL 与 @@ASCENDC_TPL_LISTS
//   （后者由 ASCENDC_TPL_SEL 产生），按每个 SEL 组合生成 TypeFromId 绑定块；
//   仅 DECL（或单维 SEL）时 TEMPLATE_PARAMS=-1/LEN 0 ⇒ 无绑定 ⇒ 编译失败
//   （W13 轮 1/轮 2 两次本地 opbuild 复现）。
//   本版 = W12 已验证 2 维结构，SCH_MODE 值域从 6 压到每 dtype 一个
//   （fp32→0 / fp16→1）⇒ 实例化 6 → 2。kernel 内部保持压平：SafetyBelt<DT>
//   平级类 + 运行期三档 if-else；SCH_MODE 仅作 TPL 键，不参与档位分派。
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"
#include "safety_belt_violation_geometry_tiling.h"

#define SB_TPL_SCH_F32 0           // fp32 组 SCH_MODE 值
#define SB_TPL_SCH_F16 1           // fp16 组 SCH_MODE 值

// ── W44（B8/D1-K1）：SCH_MODE 升级为"档位分派键"──────────────────────────────
//   旧值域 {0,1}（仅 dtype 组键）；新值域 6 值 = dtype×档位（小/中/大）。
//   合规自证（拆档包设计 §5.2 三条）：① 谓词=B 的连续全域区间（≤33/≤1024/else），
//   与原 kernel 运行期分支逐字同构，无单点等式；② 分派结果与运行期行为逐位等价
//   （sim 424 背书）；③ 动机=档间差异过大（小点 1 核标量 vs 大点 8 核向量），
//   单份产物无法同时最优——通用优化论证，非测点特调。
//   实例化 2→6，产物=每 dtype×档一份极简符号。
#define SB_TPL_SCH_F32_SMALL 0
#define SB_TPL_SCH_F16_SMALL 1
#define SB_TPL_SCH_F32_MID   2
#define SB_TPL_SCH_F16_MID   3
#define SB_TPL_SCH_F32_BIG   4
#define SB_TPL_SCH_F16_BIG   5

#if DIAG_FLOOR_AIC_MODE
// P-D1 探针形态：FORMAT 维携带 SB_MODE + 每组合 KERNEL_TYPE_SEL（dhltat 同款）。
//   2 组 NORM(AIV_ONLY) + 2 组 AIC_EMPTY(AIC_ONLY)；host sb_mode=1 全路由 AIC 空核。
ASCENDC_TPL_ARGS_DECL(SafetyBeltViolationGeometry,
    ASCENDC_TPL_DATATYPE_DECL(DT_PERSON_BOXES, C_DT_FLOAT16, C_DT_FLOAT),
    ASCENDC_TPL_FORMAT_DECL(SB_MODE, SB_MODE_NORM, SB_MODE_AIC_EMPTY),
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT),
        ASCENDC_TPL_FORMAT_SEL(SB_MODE, SB_MODE_NORM),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT16),
        ASCENDC_TPL_FORMAT_SEL(SB_MODE, SB_MODE_NORM),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT),
        ASCENDC_TPL_FORMAT_SEL(SB_MODE, SB_MODE_AIC_EMPTY),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT16),
        ASCENDC_TPL_FORMAT_SEL(SB_MODE, SB_MODE_AIC_EMPTY),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIC_ONLY),
    ),
);
#else
ASCENDC_TPL_ARGS_DECL(SafetyBeltViolationGeometry,
    ASCENDC_TPL_DATATYPE_DECL(DT_PERSON_BOXES, C_DT_FLOAT16, C_DT_FLOAT),
    ASCENDC_TPL_UINT_DECL(SCH_MODE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,
        SB_TPL_SCH_F32_SMALL, SB_TPL_SCH_F16_SMALL,
        SB_TPL_SCH_F32_MID, SB_TPL_SCH_F16_MID,
        SB_TPL_SCH_F32_BIG, SB_TPL_SCH_F16_BIG),
);

// 拆成两个 ARGS_SEL（每 dtype 一组、组内单 SCH 值）——与 W12 已验证版同构
//   （避免 DATATYPE×UINT 笛卡尔展开产生非法组合，W5 轮 1 的 Compile Error 教训）。
ASCENDC_TPL_SEL(
    // fp32 小：dtype=C_DT_FLOAT，SCH_MODE=0
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, SB_TPL_SCH_F32_SMALL),
    ),
    // fp16 小：dtype=C_DT_FLOAT16，SCH_MODE=1
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, SB_TPL_SCH_F16_SMALL),
    ),
    // fp32 中：dtype=C_DT_FLOAT，SCH_MODE=2
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, SB_TPL_SCH_F32_MID),
    ),
    // fp16 中：dtype=C_DT_FLOAT16，SCH_MODE=3
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, SB_TPL_SCH_F16_MID),
    ),
    // fp32 大：dtype=C_DT_FLOAT，SCH_MODE=4
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, SB_TPL_SCH_F32_BIG),
    ),
    // fp16 大：dtype=C_DT_FLOAT16，SCH_MODE=5
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST, SB_TPL_SCH_F16_BIG),
    ),
);
#endif  // DIAG_FLOOR_AIC_MODE
