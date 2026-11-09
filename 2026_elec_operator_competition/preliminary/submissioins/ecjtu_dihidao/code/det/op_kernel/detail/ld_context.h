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
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "../linalg_det_tiling.h"
#include "../tiling_key_linalg_det.h"
#if defined(ASCENDC_TPL_KERNEL) && defined(TILING_KEY_VAR) && ((TILING_KEY_VAR == 8704UL) || (TILING_KEY_VAR == 16896UL)) && defined(__DAV_C220_VEC__) && !defined(ASCENDC_TIME_STAMP_ON) && (!defined(ASCENDC_DUMP) || (ASCENDC_DUMP == 0))
#define LDET_N16_EXPLICIT_SCALAR_ARGUMENTS 1
#endif
#if defined(ASCENDC_TPL_KERNEL) && defined(TILING_KEY_VAR) && ((TILING_KEY_VAR == 26880UL) || (TILING_KEY_VAR == 27136UL) || (TILING_KEY_VAR == 27392UL)) && defined(__DAV_C220_VEC__) && !defined(ASCENDC_TIME_STAMP_ON) && (!defined(ASCENDC_DUMP) || (ASCENDC_DUMP == 0))
#define LDET_RAGGED_EXPLICIT_SCALAR_ARGUMENTS 1
#endif
#include "adv_api/matmul/matmul.h"
#include "adv_api/matmul/matmul_intf.h"
#if defined(LDET_N16_EXPLICIT_SCALAR_ARGUMENTS)
#include "kernel_common.h"
#endif
