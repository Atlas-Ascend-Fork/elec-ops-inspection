/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Author: Twilight-Fanyi.
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
#include "linalg_det_tiling.h"
#include "tiling_key_linalg_det.h"
#include "kernel_linalg_det_impl.h"
#include "resident_det.h"

template <typename DT_A, uint32_t LocalLuBand>
__global__ __aicore__ void linalg_det(GM_ADDR a, GM_ADDR det, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(LinalgDetTilingData);
    KERNEL_TASK_TYPE(2304, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(2560, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(2816, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(3072, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(3328, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(3584, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(3840, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(4096, KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(4352, KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(14592, KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA_WITH_STRUCT(LinalgDetTilingData, tilingData, tiling);
    if constexpr (LocalLuBand == 63U) {
        RunResidentDet(a, det, tilingData.batchCount, tilingData.matrixSize);
    } else if constexpr (LocalLuBand == 12U) {
        // The outer key keeps key12's proven scheduler contract; the isolated
        // implementation still compiles its key62-only pipeline branches.
        KernelLinalgDet<62U, true> op;
        op.Init(a, det, workspace, &tilingData);
        op.Process();
    } else {
        KernelLinalgDet<LocalLuBand> op;
        op.Init(a, det, workspace, &tilingData);
        op.Process();
    }
}
