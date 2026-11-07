/*
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
#ifndef TILING_KEY_LINALG_EIGH_H
#define TILING_KEY_LINALG_EIGH_H
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(LinalgEigh,
    ASCENDC_TPL_DTYPE_DECL(DT_A, ASCENDC_TPL_DTYPE_FLOAT),
    ASCENDC_TPL_UINT_DECL(SOLVE_MODE, ASCENDC_TPL_4BITS, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(DT_A, ASCENDC_TPL_DTYPE_FLOAT),
        ASCENDC_TPL_UINT_SEL(SOLVE_MODE, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5)
    )
);
#endif
