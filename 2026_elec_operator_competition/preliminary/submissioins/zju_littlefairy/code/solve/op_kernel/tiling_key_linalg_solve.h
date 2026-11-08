/*
 * Copyright (c) 2026 LittleFairy.
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
ASCENDC_TPL_ARGS_DECL(LinalgSolve,
    ASCENDC_TPL_DATATYPE_DECL(DT_A, C_DT_FLOAT),
    ASCENDC_TPL_UINT_DECL(KERNEL_PATH, 5, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28),
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_A, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 9, 10, 11, 13, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_A, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 0, 1, 27, 28),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_A, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 2, 3, 5, 6, 7),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
    ),
);
