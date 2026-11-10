/*
 * Copyright (c) 2026 aodebiao.
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

#include "kernel_operator.h"
#include "linalg_eigh_wide.h"
#include "linalg_eigh_kernel.h"
#include "linalg_eigh_coop.h"
#include "linalg_eigh_pooled.h"

template<typename DT_A,bool SMALL,bool RESIDENT,bool COOPERATIVE,bool POOLED> __global__ __aicore__ void linalg_eigh(GM_ADDR a,GM_ADDR values,
    GM_ADDR vectors,GM_ADDR workspace,GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LinalgEighTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgEighTilingData,t,tiling);
    if constexpr (POOLED) {KernelPooledEigh<DT_A> op;op.Init(a,values,vectors,workspace,t);op.Process();}
    else if constexpr (COOPERATIVE) {KernelCooperativeEigh<DT_A> op;op.Init(a,values,vectors,workspace,t);op.Process();}
    else {KernelLinalgEigh<DT_A,SMALL,RESIDENT> op;op.Init(a,values,vectors,workspace,t);op.Process();}
}
