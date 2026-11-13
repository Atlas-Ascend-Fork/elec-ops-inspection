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
// TilingKey模板定义的头文件
//
// 只有一个模板维度：raw_pred 的数据类型。priors 恒为 float32，输出跟随 raw_pred，
// 所以不存在其它需要特化的分支。
//
// Host 与 Kernel 都直接 include 本文件，模板声明只有一份。DATATYPE_SEL 的枚举
// 顺序按位置与 OpDef 里 DataType({...}) 的列表一一对应，且必须与 tiling key 的
// 取值方式（ASCENDC_TPL_SEL_PARAM 传入 raw_pred 的 ge::DataType 数值）保持一致。
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(FireCandidateDecodeFusion, ASCENDC_TPL_DATATYPE_DECL(DT_RAW_PRED, C_DT_FLOAT, C_DT_FLOAT16), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(DT_RAW_PRED, C_DT_FLOAT, C_DT_FLOAT16), ), );
