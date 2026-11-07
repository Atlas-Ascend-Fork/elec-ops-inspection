/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 黎何.
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
ASCENDC_TPL_ARGS_DECL(SwiGlu, ASCENDC_TPL_DATATYPE_DECL(DT_INPUT_X, C_DT_FLOAT16),
                      ASCENDC_TPL_UINT_DECL(PATH, 6, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                            51),);
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 11265UL || TILING_KEY_VAR == 12033UL)
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 43, 46, 48, 44, 49, 50, 45, 47, 51,
                                                          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                                          18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                                          33, 34, 35, 36, 37, 38, 39, 40, 41, 42),),);
#else
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
                                     ASCENDC_TPL_UINT_SEL(PATH, ASCENDC_TPL_UI_LIST, 43, 46, 48, 44, 49, 50, 45, 47, 51,
                                                          0, 1),),);
#endif
