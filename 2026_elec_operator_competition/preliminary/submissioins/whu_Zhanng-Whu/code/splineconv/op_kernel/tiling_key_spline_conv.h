/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#define SPLINE_DISPATCH_KEY(dtype, single, sparse, closed0, closed1)                                                   \
    ASCENDC_TPL_ARGS_SEL(                                                                                              \
        ASCENDC_TPL_DATATYPE_SEL(DT_X, dtype), ASCENDC_TPL_UINT_SEL(SINGLE, ASCENDC_TPL_UI_LIST, single),              \
        ASCENDC_TPL_UINT_SEL(SPARSE, ASCENDC_TPL_UI_LIST, sparse), ASCENDC_TPL_BOOL_SEL(CLOSED0, closed0),             \
        ASCENDC_TPL_BOOL_SEL(CLOSED1, closed1), ASCENDC_TPL_UINT_SEL(NODEPAD, ASCENDC_TPL_UI_LIST, 0),)

#define SPLINE_DISPATCH_ORDER                                                                                          \
    SPLINE_DISPATCH_KEY(C_DT_FLOAT, 3, 0, 0, 0), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 4, 0, 0, 0),                          \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 0, 0, 0), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 3, 0, 0),                      \
        SPLINE_DISPATCH_KEY(C_DT_BF16, 0, 0, 0, 0), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 1, 0, 0, 0),                       \
        SPLINE_DISPATCH_KEY(C_DT_BF16, 1, 0, 0, 0), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 2, 0, 0, 0),                       \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 2, 0, 0, 1), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 2, 0, 1, 0),                      \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 2, 0, 1, 1), SPLINE_DISPATCH_KEY(C_DT_BF16, 2, 0, 0, 0),                       \
        SPLINE_DISPATCH_KEY(C_DT_BF16, 2, 0, 0, 1), SPLINE_DISPATCH_KEY(C_DT_BF16, 2, 0, 1, 0),                        \
        SPLINE_DISPATCH_KEY(C_DT_BF16, 2, 0, 1, 1), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 3, 0, 0, 1),                       \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 3, 0, 1, 0), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 3, 0, 1, 1),                      \
        SPLINE_DISPATCH_KEY(C_DT_BF16, 3, 0, 0, 0), SPLINE_DISPATCH_KEY(C_DT_BF16, 3, 0, 0, 1),                        \
        SPLINE_DISPATCH_KEY(C_DT_BF16, 3, 0, 1, 0), SPLINE_DISPATCH_KEY(C_DT_BF16, 3, 0, 1, 1),                        \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 4, 0, 0, 1), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 4, 0, 1, 0),                      \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 4, 0, 1, 1), SPLINE_DISPATCH_KEY(C_DT_BF16, 4, 0, 0, 0),                       \
        SPLINE_DISPATCH_KEY(C_DT_BF16, 4, 0, 0, 1), SPLINE_DISPATCH_KEY(C_DT_BF16, 4, 0, 1, 0),                        \
        SPLINE_DISPATCH_KEY(C_DT_BF16, 4, 0, 1, 1), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 1, 0, 0),                       \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 1, 0, 1), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 1, 1, 0),                      \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 1, 1, 1), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 2, 0, 0),                      \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 2, 0, 1), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 2, 1, 0),                      \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 2, 1, 1), SPLINE_DISPATCH_KEY(C_DT_BF16, 0, 1, 0, 0),                       \
        SPLINE_DISPATCH_KEY(C_DT_BF16, 0, 1, 0, 1), SPLINE_DISPATCH_KEY(C_DT_BF16, 0, 1, 1, 0),                        \
        SPLINE_DISPATCH_KEY(C_DT_BF16, 0, 1, 1, 1), SPLINE_DISPATCH_KEY(C_DT_BF16, 0, 2, 0, 0),                        \
        SPLINE_DISPATCH_KEY(C_DT_BF16, 0, 2, 0, 1), SPLINE_DISPATCH_KEY(C_DT_BF16, 0, 2, 1, 0),                        \
        SPLINE_DISPATCH_KEY(C_DT_BF16, 0, 2, 1, 1), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 3, 0, 1),                       \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 3, 1, 0), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 3, 1, 1),                      \
        ASCENDC_TPL_ARGS_SEL(                                                                                          \
            ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT), ASCENDC_TPL_UINT_SEL(SINGLE, ASCENDC_TPL_UI_LIST, 0),          \
            ASCENDC_TPL_UINT_SEL(SPARSE, ASCENDC_TPL_UI_LIST, 6), ASCENDC_TPL_BOOL_SEL(CLOSED0, 0, 1),                 \
            ASCENDC_TPL_BOOL_SEL(CLOSED1, 0, 1), ASCENDC_TPL_UINT_SEL(NODEPAD, ASCENDC_TPL_UI_RANGE, 1, 0, 15),),     \
        ASCENDC_TPL_ARGS_SEL(                                                                                          \
            ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT), ASCENDC_TPL_UINT_SEL(SINGLE, ASCENDC_TPL_UI_LIST, 0),          \
            ASCENDC_TPL_UINT_SEL(SPARSE, ASCENDC_TPL_UI_LIST, 5), ASCENDC_TPL_BOOL_SEL(CLOSED0, 0, 1),                 \
            ASCENDC_TPL_BOOL_SEL(CLOSED1, 0, 1), ASCENDC_TPL_UINT_SEL(NODEPAD, ASCENDC_TPL_UI_RANGE, 1, 0, 15),),     \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 5, 0, 0, 0), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 5, 0, 0, 1),                      \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 5, 0, 1, 0), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 5, 0, 1, 1),                      \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 8, 0, 0), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 8, 0, 1),                      \
        SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 8, 1, 0), SPLINE_DISPATCH_KEY(C_DT_FLOAT, 0, 8, 1, 1),                      \
        ASCENDC_TPL_ARGS_SEL(                                                                                          \
            ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT), ASCENDC_TPL_UINT_SEL(SINGLE, ASCENDC_TPL_UI_LIST, 6),          \
            ASCENDC_TPL_UINT_SEL(SPARSE, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_BOOL_SEL(CLOSED0, 0, 1),                 \
            ASCENDC_TPL_BOOL_SEL(CLOSED1, 0, 1), ASCENDC_TPL_UINT_SEL(NODEPAD, ASCENDC_TPL_UI_RANGE, 1, 0, 14),)
ASCENDC_TPL_ARGS_DECL(SplineConv, ASCENDC_TPL_DATATYPE_DECL(DT_X, ASCENDC_TPL_INPUT(0), C_DT_FLOAT, C_DT_BF16),
                      ASCENDC_TPL_UINT_DECL(SINGLE, 3, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6),
                      ASCENDC_TPL_UINT_DECL(SPARSE, 4, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 5, 6, 8),
                      ASCENDC_TPL_BOOL_DECL(CLOSED0, 0, 1), ASCENDC_TPL_BOOL_DECL(CLOSED1, 0, 1),
                      ASCENDC_TPL_UINT_DECL(NODEPAD, 5, ASCENDC_TPL_UI_RANGE, 1, 0, 30),);
ASCENDC_TPL_SEL(SPLINE_DISPATCH_ORDER);
