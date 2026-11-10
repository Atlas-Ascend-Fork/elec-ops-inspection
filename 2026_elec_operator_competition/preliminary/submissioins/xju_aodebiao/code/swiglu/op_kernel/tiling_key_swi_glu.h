/*
 * Copyright (c) 2026 aodebiao.
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
ASCENDC_TPL_ARGS_DECL(SwiGlu,
    ASCENDC_TPL_DATATYPE_DECL(DT_INPUT_X, C_DT_FLOAT16),
    ASCENDC_TPL_KERNEL_TYPE_DECL(KERNEL_MODE, ASCENDC_TPL_AIV_ONLY, ASCENDC_TPL_MIX_AIC_1_1),
    ASCENDC_TPL_BOOL_DECL(USE_CUBE, 0, 1)
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(KERNEL_MODE, ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_BOOL_SEL(USE_CUBE, 0)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_SHARED_KERNEL_TYPE_SEL(KERNEL_MODE, ASCENDC_TPL_MIX_AIC_1_1),
        ASCENDC_TPL_BOOL_SEL(USE_CUBE, 1)
    )
);
