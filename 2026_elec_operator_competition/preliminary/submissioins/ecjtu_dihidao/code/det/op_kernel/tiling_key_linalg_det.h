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
#pragma once
#include "ascendc/host_api/tiling/template_argument.h"

// Shared template keys and execution types for host and device.
ASCENDC_TPL_ARGS_DECL(LinalgDet,
ASCENDC_TPL_DATATYPE_DECL(DT_A, C_DT_FLOAT),
ASCENDC_TPL_UINT_DECL(LD_BATCH, 7, ASCENDC_TPL_UI_RANGE, 4, 0, 36, 38, 102, 37, 37, 103, 108),
);
ASCENDC_TPL_SEL(
ASCENDC_TPL_ARGS_SEL(
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_1),
ASCENDC_TPL_DATATYPE_SEL(DT_A, C_DT_FLOAT),
ASCENDC_TPL_UINT_SEL(LD_BATCH, ASCENDC_TPL_UI_LIST, 64, 65, 85, 105, 106),
),
ASCENDC_TPL_ARGS_SEL(
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
ASCENDC_TPL_DATATYPE_SEL(DT_A, C_DT_FLOAT),
ASCENDC_TPL_UINT_SEL(LD_BATCH, ASCENDC_TPL_UI_LIST, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 25, 32, 33, 34, 42, 43, 44, 45, 47, 48, 49, 50, 62, 107),
),
ASCENDC_TPL_ARGS_SEL(
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
ASCENDC_TPL_DATATYPE_SEL(DT_A, C_DT_FLOAT),
ASCENDC_TPL_UINT_SEL(LD_BATCH, ASCENDC_TPL_UI_LIST, 20, 21, 22, 24, 59, 108),
),
ASCENDC_TPL_ARGS_SEL(
ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
ASCENDC_TPL_DATATYPE_SEL(DT_A, C_DT_FLOAT),
ASCENDC_TPL_UINT_SEL(LD_BATCH, ASCENDC_TPL_UI_LIST, 27, 30, 31, 35, 38, 40, 41, 51, 52, 53, 54, 55, 56, 57, 58, 60, 63, 37, 103, 104),
),
);
