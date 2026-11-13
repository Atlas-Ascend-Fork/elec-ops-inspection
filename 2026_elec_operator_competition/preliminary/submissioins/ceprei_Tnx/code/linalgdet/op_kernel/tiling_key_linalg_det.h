/*
 * Copyright (c) 2026 ceprei_Tnx.
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
// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

#if DET_AIV_KEY
// 两组组合：DET_CUBE=0 走纯向量（只启动 Vector 核）；DET_CUBE=1 保留 Cube+Vector 混合。
// ASCENDC_TPL_KERNEL_TYPE_SEL 无需在 ARGS_DECL 里声明，直接写在 ARGS_SEL 内；
// 同一个 ASCENDC_TPL_SEL 下每组的类型必须都显式配置。
ASCENDC_TPL_ARGS_DECL(LinalgDet, ASCENDC_TPL_DATATYPE_DECL(DT_A, C_DT_FLOAT), ASCENDC_TPL_BOOL_DECL(DET_CUBE, 0, 1), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_DATATYPE_SEL(DT_A, C_DT_FLOAT),
                                     ASCENDC_TPL_BOOL_SEL(DET_CUBE, 0), ), );
#else
ASCENDC_TPL_ARGS_DECL(LinalgDet, ASCENDC_TPL_DATATYPE_DECL(DT_A, C_DT_FLOAT), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_A, C_DT_FLOAT), ), );
#endif
