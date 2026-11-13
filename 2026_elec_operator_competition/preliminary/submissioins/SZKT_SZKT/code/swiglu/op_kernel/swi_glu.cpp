/*
 * Copyright (c) 2026 Beijing Kuntai Information Technology Co., Ltd.
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
#include "swi_glu_tiling.h"
#include "swi_glu_vector.h"
#include "swi_glu_cube.h"
#include "swi_glu_resident.h"
#include "swi_glu_generic.h"

// 本算子只使用基础Mmad；若引入Matmul对象，须恢复库的工作区初始化。
namespace matmul {
__aicore__ inline void clearWorkspace(__gm__ uint8_t *workspace) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
    AscendC::SetAtomicNone();
    if ASCEND_IS_AIC {
        AscendC::SetMaskNorm();
        AscendC::SetLoadDataBoundary((uint64_t)0);
        AscendC::SetLoadDataPaddingValue((uint64_t)0);
    } else {
        AscendC::SetVectorMask<uint64_t, AscendC::MaskMode::NORMAL>((uint64_t)-1, (uint64_t)-1);
        AscendC::SetMaskNorm();
    }
#endif
}
} // namespace matmul

using namespace AscendC;

namespace {
__aicore__ inline void RunCube256(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                  GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                  GM_ADDR workspace) {
    swiglu::KernelCubePipeline<256, 64, 64, 4> op;
    op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace);
    op.Process();
}

__aicore__ inline void RunCube128(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                  GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                  GM_ADDR workspace) {
    swiglu::KernelCubePipeline<128, 128, 128, 2> op;
    op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace);
    op.Process();
}

__aicore__ inline void RunResidentStatic(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                  GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                  GM_ADDR workspace) {
    TPipe pipe;
    constexpr SwiGluTilingData fixedPlan{8192, 256, 10, 13};
    swiglu::KernelGateResident<256, 8192, 10> op;
    op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, fixedPlan, &pipe);
    op.Process();
}

__aicore__ inline void RunFullKPlan(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                  GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                  GM_ADDR workspace, const SwiGluTilingData &tiling_data) {
    TPipe pipe;
    if (tiling_data.residentGroups != 0 && tiling_data.dim == 256) {
        swiglu::KernelGateResident<256> op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling_data, &pipe);
        op.Process();
    } else if (tiling_data.residentGroups != 0 && tiling_data.dim == 128) {
        swiglu::KernelGateResident<128> op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling_data, &pipe);
        op.Process();
    } else {
        swiglu::KernelFullK op;
        op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling_data.dim, tiling_data.rows, &pipe);
        op.Process();
    }
}

__aicore__ inline void RunSplitKPlan(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                  GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                  GM_ADDR workspace, const SwiGluTilingData &tiling_data) {
    TPipe pipe;
    swiglu::KernelSwiGluSplitK op;
    op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace,
            tiling_data.dim, tiling_data.rows, &pipe);
    op.Process();
}

} // namespace

__global__ __aicore__ void swi_glu(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                 GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                 GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if TILING_KEY_IS(0) {
        KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
        swiglu::RunSingleRow<1, 1>(input_x, weight_1, bias_1, weight_2, bias_2, output);
    } else if TILING_KEY_IS(1) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
        swiglu::RunSingleRow<4, 1>(input_x, weight_1, bias_1, weight_2, bias_2, output);
    } else if TILING_KEY_IS(2) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY);
        swiglu::RunSingleRow<32, 1>(input_x, weight_1, bias_1, weight_2, bias_2, output);
    } else if TILING_KEY_IS(3) {
        KERNEL_TASK_TYPE(3, KERNEL_TYPE_AIV_ONLY);
        swiglu::RunSingleRow<64, 2>(input_x, weight_1, bias_1, weight_2, bias_2, output);
    } else if TILING_KEY_IS(4) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_AIV_ONLY);
        swiglu::RunSmall13<21>(input_x, weight_1, bias_1, weight_2, bias_2, output, 0);
    } else if TILING_KEY_IS(13) {
        KERNEL_TASK_TYPE(13, KERNEL_TYPE_AIV_ONLY);
        REGISTER_TILING_DEFAULT(SwiGluTilingData);
        GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, plan, tiling);
        swiglu::RunSmall13<0>(input_x, weight_1, bias_1, weight_2, bias_2, output, plan.rows);
    } else if TILING_KEY_IS(6) {
        KERNEL_TASK_TYPE(6, KERNEL_TYPE_AIV_ONLY);
        REGISTER_TILING_DEFAULT(SwiGluTilingData);
        GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, plan, tiling);
        swiglu::RunDynamicRow(input_x, weight_1, bias_1, weight_2, bias_2, output, plan.dim);
    } else if TILING_KEY_IS(9) {
        KERNEL_TASK_TYPE(9, KERNEL_TYPE_AIV_ONLY);
        swiglu::RunSmall33(input_x, weight_1, bias_1, weight_2, bias_2, output);
    } else if TILING_KEY_IS(10) {
        KERNEL_TASK_TYPE(10, KERNEL_TYPE_MIX_AIC_1_2);
        RunCube256(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace);
    } else if TILING_KEY_IS(11) {
        KERNEL_TASK_TYPE(11, KERNEL_TYPE_MIX_AIC_1_2);
        RunCube128(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace);
    } else if TILING_KEY_IS(12) {
        KERNEL_TASK_TYPE(12, KERNEL_TYPE_MIX_AIC_1_2);
        RunResidentStatic(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace);
    } else if TILING_KEY_IS(7) {
        KERNEL_TASK_TYPE(7, KERNEL_TYPE_MIX_AIC_1_2);
        REGISTER_TILING_DEFAULT(SwiGluTilingData);
        GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
        RunFullKPlan(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling_data);
    } else if TILING_KEY_IS(8) {
        KERNEL_TASK_TYPE(8, KERNEL_TYPE_MIX_AIC_1_2);
        REGISTER_TILING_DEFAULT(SwiGluTilingData);
        GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
        RunSplitKPlan(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling_data);
    }

}
