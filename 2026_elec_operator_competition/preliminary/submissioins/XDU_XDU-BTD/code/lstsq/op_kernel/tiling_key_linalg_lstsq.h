// Copyright 2026 XDU_XDU-BTD Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#pragma once
#include "ascendc/host_api/tiling/template_argument.h"
ASCENDC_TPL_ARGS_DECL(LinalgLstsq,
ASCENDC_TPL_DATATYPE_DECL(DT_A,C_DT_FLOAT),
ASCENDC_TPL_BOOL_DECL(BIG,0,1),
ASCENDC_TPL_UINT_DECL(SHAPE,4,ASCENDC_TPL_UI_LIST,0,1,2,3,4,5,6,7,8,9,10,11),
);
ASCENDC_TPL_SEL(
ASCENDC_TPL_ARGS_SEL(
ASCENDC_TPL_DATATYPE_SEL(DT_A,C_DT_FLOAT),
ASCENDC_TPL_BOOL_SEL(BIG,0),
ASCENDC_TPL_UINT_SEL(SHAPE,ASCENDC_TPL_UI_LIST,0,5),
),
ASCENDC_TPL_ARGS_SEL(
ASCENDC_TPL_DATATYPE_SEL(DT_A,C_DT_FLOAT),
ASCENDC_TPL_BOOL_SEL(BIG,1),
ASCENDC_TPL_UINT_SEL(SHAPE,ASCENDC_TPL_UI_LIST,0,1,2,3,4,7,8,9),
),
ASCENDC_TPL_ARGS_SEL(
ASCENDC_TPL_DATATYPE_SEL(DT_A,C_DT_FLOAT),
ASCENDC_TPL_BOOL_SEL(BIG,0),
ASCENDC_TPL_UINT_SEL(SHAPE,ASCENDC_TPL_UI_LIST,6,10,11),
),
);
