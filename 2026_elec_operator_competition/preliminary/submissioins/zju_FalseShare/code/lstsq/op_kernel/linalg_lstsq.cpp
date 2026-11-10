/*
 * Copyright (c) 2026 FalseShare contributors.
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

#include "lstsq_narrow.h"

#ifndef LSTSQ_DIRECT_BENCH
 template <typename DT_A,uint32_t KERNEL_PATH>
__global__ __aicore__ __attribute__((aligned(4096))) void linalg_lstsq(GM_ADDR A,GM_ADDR B,GM_ADDR solution,GM_ADDR residuals,GM_ADDR workspace,GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::InitSocState();
    REGISTER_TILING_DEFAULT(LinalgLstsqTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinalgLstsqTilingData,t,tiling);
    if constexpr(KERNEL_PATH==0) {
        if ASCEND_IS_AIC {NarrowOrthogonal::Run<true>(A,B,solution,AscendC::GetUserWorkspace(workspace),t,residuals);}
        if ASCEND_IS_AIV {NarrowOrthogonal::Run<false>(A,B,solution,AscendC::GetUserWorkspace(workspace),t,residuals);}
    }
    else if constexpr(KERNEL_PATH==8) {
        if ASCEND_IS_AIC {NarrowOrthogonal::Run<true,112>(A,B,solution,AscendC::GetUserWorkspace(workspace),t,residuals);}
        if ASCEND_IS_AIV {NarrowOrthogonal::Run<false,112>(A,B,solution,AscendC::GetUserWorkspace(workspace),t,residuals);}
    }
    else if constexpr(KERNEL_PATH==5) {
        if ASCEND_IS_AIC {Orthogonal::Run<true>(A,B,solution,AscendC::GetUserWorkspace(workspace),t,residuals);}
        if ASCEND_IS_AIV {Orthogonal::Run<false>(A,B,solution,AscendC::GetUserWorkspace(workspace),t,residuals);}
    }
    else if constexpr(KERNEL_PATH==1) {ResidentElimination<16> op;op.Run(A,B,solution,residuals,AscendC::GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==2) {ResidentElimination<64> op;op.Run(A,B,solution,residuals,AscendC::GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==6) {ResidentElimination<64,false,false,false,true> op;op.Run(A,B,solution,residuals,AscendC::GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==3) {RefinedSquare<128,32> op;op.Run(A,B,solution,residuals,AscendC::GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==4) {RefinedSquare<256,32> op;op.Run(A,B,solution,residuals,AscendC::GetUserWorkspace(workspace),t);}
    else if constexpr(KERNEL_PATH==7) {Lstsq op;op.Run(A,B,solution,residuals,AscendC::GetUserWorkspace(workspace),t);}
}
#endif
