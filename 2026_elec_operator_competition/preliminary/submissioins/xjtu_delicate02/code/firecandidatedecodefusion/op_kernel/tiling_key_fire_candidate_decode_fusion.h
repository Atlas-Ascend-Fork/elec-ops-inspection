/*
 * Copyright (c) 2026 Delicate02. All rights reserved.
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
ASCENDC_TPL_ARGS_DECL(FireCandidateDecodeFusion,
    ASCENDC_TPL_DATATYPE_DECL(DT_RAW_PRED, C_DT_FLOAT, C_DT_FLOAT16),
    ASCENDC_TPL_UINT_DECL(TILE, 11, ASCENDC_TPL_UI_LIST, 32, 64, 128, 512, 528, 640, 896, 1024, 1640, 1664),
    ASCENDC_TPL_UINT_DECL(MODE, 2, ASCENDC_TPL_UI_LIST, 0, 1, 2),
);
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_DATATYPE_SEL(DT_RAW_PRED, C_DT_FLOAT, C_DT_FLOAT16),
    ASCENDC_TPL_UINT_SEL(TILE, ASCENDC_TPL_UI_LIST, 32, 64, 128, 512, 528, 640, 896, 1024, 1640, 1664),
    ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2),
));
