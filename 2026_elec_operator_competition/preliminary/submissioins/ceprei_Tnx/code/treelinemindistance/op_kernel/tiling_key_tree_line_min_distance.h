/*
 * Copyright (c) 2026 ceprei_Tnx.
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

// line_segments and min_distance use the same dtype as tree_points. The
// nearest-segment output is always int32, so only the primary input dtype is
// needed in the tiling key.
ASCENDC_TPL_ARGS_DECL(TreeLineMinDistance, ASCENDC_TPL_DATATYPE_DECL(DT_TREE_POINTS, C_DT_FLOAT16, C_DT_FLOAT), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_TREE_POINTS, C_DT_FLOAT16, C_DT_FLOAT), ), );
