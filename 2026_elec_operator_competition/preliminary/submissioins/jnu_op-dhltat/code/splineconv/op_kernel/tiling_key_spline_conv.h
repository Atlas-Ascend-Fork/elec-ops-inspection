/*
 * Copyright (c) 2026 dhltat.
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
// Complete462 entries:252FP32 and210BF16.
// Historical separate-zero E4 layouts replace six unproven coowned layouts.
// Four static8 keys always dispatch to existing paired entries and remain omitted.
// All32-byte alignment and complete cache-family constraints stay unchanged.
#include "ascendc/host_api/tiling/template_argument.h"
// No CI/CO Cartesian expansion in the generic owned-row family.
// DIM2/3 and DEG1/2/3 stay mathematical variants; channel counts remain runtime metadata.
ASCENDC_TPL_ARGS_DECL(SplineConv,
                      ASCENDC_TPL_UINT_DECL(MODE, 4, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 7, 13, 14, 15),
                      ASCENDC_TPL_UINT_DECL(DIM, 2, ASCENDC_TPL_UI_LIST, 2, 3),
                      ASCENDC_TPL_UINT_DECL(DEG, 2, ASCENDC_TPL_UI_LIST, 1, 2, 3),
                      ASCENDC_TPL_UINT_DECL(CI, 8, ASCENDC_TPL_UI_LIST, 0, 8, 16, 24, 32, 64, 128),
                      ASCENDC_TPL_UINT_DECL(CO, 8, ASCENDC_TPL_UI_LIST, 0, 8, 16, 24, 32, 64, 128),
                      ASCENDC_TPL_UINT_DECL(ES, 4, ASCENDC_TPL_UI_LIST, 0, 1, 2, 4, 8, 3, 5, 6, 7),
                      ASCENDC_TPL_UINT_DECL(NN, 8, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
                      ASCENDC_TPL_DATATYPE_DECL(INPUT_T, C_DT_FLOAT, C_DT_BF16, ASCENDC_TPL_INPUT(0)));
// One row per compiled selector family. Lists in parentheses belong to one axis.
#define SPLINE_KEY_VALUES(...) __VA_ARGS__
#define SPLINE_UINT_AXIS(key, ...) ASCENDC_TPL_UINT_SEL(key, ASCENDC_TPL_UI_LIST, __VA_ARGS__)
#define SPLINE_AIV_KEY(mode, dim, degree, ci, co, edges, nodes, dtype) \
    ASCENDC_TPL_ARGS_SEL( \
        SPLINE_UINT_AXIS(MODE, SPLINE_KEY_VALUES mode), \
        SPLINE_UINT_AXIS(DIM, SPLINE_KEY_VALUES dim), \
        SPLINE_UINT_AXIS(DEG, SPLINE_KEY_VALUES degree), \
        SPLINE_UINT_AXIS(CI, SPLINE_KEY_VALUES ci), \
        SPLINE_UINT_AXIS(CO, SPLINE_KEY_VALUES co), \
        SPLINE_UINT_AXIS(ES, SPLINE_KEY_VALUES edges), \
        SPLINE_UINT_AXIS(NN, SPLINE_KEY_VALUES nodes), \
        ASCENDC_TPL_DATATYPE_SEL(INPUT_T, dtype), \
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY))

ASCENDC_TPL_SEL(
    SPLINE_AIV_KEY((1), (2, 3), (1, 2, 3), (0), (0), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((1), (2, 3), (1, 2, 3), (0), (0), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((2), (2), (1), (8), (8, 16, 24, 32), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((2), (2), (1), (8), (8, 16, 24, 32), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((2), (2), (1), (16), (8, 16), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((2), (2), (1), (16), (8, 16), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((2), (2), (1), (24), (8), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((2), (2), (1), (24), (8), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((2), (2), (1), (32), (8), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((2), (2), (1), (32), (8), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((3), (2), (1), (8), (8, 16, 32, 64, 128), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((3), (2), (1), (8), (8, 16, 32, 64, 128), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((3), (2), (1), (16), (8, 16, 32, 64, 128), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((3), (2), (1), (16), (8, 16, 32, 64, 128), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((3), (2), (1), (32), (8, 16, 32, 64, 128), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((3), (2), (1), (32), (8, 16, 32, 64, 128), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((3), (2), (1), (64), (8, 16, 32, 64), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((3), (2), (1), (64), (8, 16, 32, 64), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((3), (2), (1), (128), (8, 16, 32), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((3), (2), (1), (128), (8, 16, 32), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((5), (2), (1), (8), (8, 16, 32, 64, 128), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((5), (2), (1), (8), (8, 16, 32, 64, 128), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((5), (2), (1), (16), (8, 16, 32, 64), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((5), (2), (1), (16), (8, 16, 32, 64), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((5), (2), (1), (32), (8, 16, 32), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((5), (2), (1), (32), (8, 16, 32), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((5), (2), (1), (64), (8, 16), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((5), (2), (1), (64), (8, 16), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((5), (2), (1), (128), (8), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((5), (2), (1), (128), (8), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((6), (2), (1), (8), (8, 16, 32, 64, 128), (0, 2, 4, 8), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((6), (2), (1), (8), (16, 32, 64, 128), (0, 2, 4, 8), (0), C_DT_BF16),
    SPLINE_AIV_KEY((6), (2), (1), (16), (8, 16, 32, 64, 128), (0, 2, 4, 8), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((6), (2), (1), (16), (16, 32, 64, 128), (0, 2, 4, 8), (0), C_DT_BF16),
    SPLINE_AIV_KEY((6), (2), (1), (32), (8, 16, 32, 64, 128), (0, 2, 4), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((6), (2), (1), (32), (8, 16, 64, 128), (8), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((6), (2), (1), (32), (16, 32, 64, 128), (0, 2, 4, 8), (0), C_DT_BF16),
    SPLINE_AIV_KEY((6), (2), (1), (64), (8, 16, 32, 64), (0, 2, 4), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((6), (2), (1), (64), (8, 64), (8), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((6), (2), (1), (64), (16, 32, 64), (0, 2, 4, 8), (0), C_DT_BF16),
    SPLINE_AIV_KEY((6), (2), (1), (128), (8, 16, 32), (0, 2, 4), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((6), (2), (1), (128), (8, 32), (8), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((6), (2), (1), (128), (16, 32), (0, 2, 4, 8), (0), C_DT_BF16),
    SPLINE_AIV_KEY((8), (3), (1), (8), (8, 16, 32, 64, 128), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((8), (3), (1), (8), (8, 16, 32, 64, 128), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((8), (3), (1), (16), (8, 16, 32, 64), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((8), (3), (1), (16), (8, 16, 32, 64), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((8), (3), (1), (32), (8, 16, 32), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((8), (3), (1), (32), (8, 16, 32), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((8), (3), (1), (64), (8, 16), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((8), (3), (1), (64), (8, 16), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((8), (3), (1), (128), (8), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((8), (3), (1), (128), (8), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((9), (2), (1), (8), (8, 16, 32, 64, 128), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((10), (2), (1), (8), (8, 16, 32, 64, 128), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((9), (2), (1), (16), (8, 16, 32, 64, 128), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((9), (2), (1), (16), (8, 16, 32, 64, 128), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((10), (2), (1), (16), (8, 16, 32, 64, 128), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((10), (2), (1), (16), (16, 32, 64, 128), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((9), (2), (1), (32), (8, 16, 32, 64, 128), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((9), (2), (1), (32), (8, 16, 32, 64, 128), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((10), (2), (1), (32), (8, 16, 32, 64, 128), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((10), (2), (1), (32), (16, 32, 64, 128), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((9), (2), (1), (64), (8, 16, 32, 64), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((9), (2), (1), (64), (8, 16, 32, 64), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((10), (2), (1), (64), (8, 16, 32, 64), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((10), (2), (1), (64), (16, 32, 64), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((9), (2), (1), (128), (8, 16, 32), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((9), (2), (1), (128), (8, 16, 32), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((10), (2), (1), (128), (8, 16, 32), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((10), (2), (1), (128), (16, 32), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((7), (3), (1), (8), (8, 16, 32, 64, 128), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((7), (3), (1), (8), (8, 16, 32, 64, 128), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((7), (3), (1), (16), (8, 16, 32, 64), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((7), (3), (1), (16), (8, 16, 32, 64), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((7), (3), (1), (32), (8, 16, 32), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((7), (3), (1), (32), (8, 16, 32), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((7), (3), (1), (64), (8, 16), (0), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((7), (3), (1), (64), (8, 16), (0), (0), C_DT_BF16),
    SPLINE_AIV_KEY((2), (2), (1), (8), (8, 24), (0), (1, 2, 3, 4, 5, 6, 7, 8), C_DT_BF16),
    SPLINE_AIV_KEY((2), (2), (1), (16), (8), (0), (1, 2, 3, 4, 5, 6, 7, 8), C_DT_BF16),
    SPLINE_AIV_KEY((2), (2), (1), (24), (8), (0), (1, 2, 3, 4, 5, 6, 7, 8), C_DT_BF16),
    SPLINE_AIV_KEY((14), (2), (1), (8), (16, 32), (0), (2, 4, 6, 8, 10, 12, 14, 16), C_DT_FLOAT),
    SPLINE_AIV_KEY((12), (2), (1), (8), (16, 32), (0), (1, 3, 5, 7, 9, 11, 13, 15), C_DT_FLOAT),
    SPLINE_AIV_KEY((14), (2), (1), (16), (16), (0), (2, 4, 6, 8, 10, 12, 14, 16), C_DT_FLOAT),
    SPLINE_AIV_KEY((15), (2), (1), (32), (32), (8), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((15), (2), (1), (64), (16, 32), (8), (0), C_DT_FLOAT),
    SPLINE_AIV_KEY((15), (2), (1), (128), (16), (8), (0), C_DT_FLOAT)
)

#undef SPLINE_AIV_KEY
#undef SPLINE_UINT_AXIS
#undef SPLINE_KEY_VALUES
