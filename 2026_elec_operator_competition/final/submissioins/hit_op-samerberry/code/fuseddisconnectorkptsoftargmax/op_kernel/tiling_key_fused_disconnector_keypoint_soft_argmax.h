/*
 * Copyright (c) 2026 hit_op-samerberry.
 * SPDX-License-Identifier: Apache-2.0
 *
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

#ifndef TILING_KEY_FUSED_DISCONNECTOR_KEYPOINT_SOFT_ARGMAX_H
#define TILING_KEY_FUSED_DISCONNECTOR_KEYPOINT_SOFT_ARGMAX_H

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(FusedDisconnectorKeypointSoftArgmax,
    ASCENDC_TPL_UINT_DECL(MODE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 1, 2)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, 2),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0)
    )
);

#endif  // TILING_KEY_FUSED_DISCONNECTOR_KEYPOINT_SOFT_ARGMAX_H
