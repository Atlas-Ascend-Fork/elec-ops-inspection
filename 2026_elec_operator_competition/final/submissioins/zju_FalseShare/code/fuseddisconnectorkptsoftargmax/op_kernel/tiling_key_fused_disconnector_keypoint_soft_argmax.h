/*
 * Copyright (c) 2026 FalseShare Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Apache License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "ascendc/host_api/tiling/template_argument.h"
ASCENDC_TPL_ARGS_DECL(FusedDisconnectorKeypointSoftArgmax,
    ASCENDC_TPL_DATATYPE_DECL(DT_FEAT, C_DT_FLOAT),
    ASCENDC_TPL_UINT_DECL(DISPATCH, 2, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3),
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_FEAT, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(DISPATCH, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_FEAT, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(DISPATCH, ASCENDC_TPL_UI_LIST, 2, 3),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
);
