/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 黎何.
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

#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "swi_glu_tiling.h"
#include "tiling_key_swi_glu.h"
#include "swi_glu_common.h"
#include "swi_glu_common_1.h"
#include "swi_glu_common_2.h"
#include "kernel_swi_glu.h"
#include "tiny_vector_swi_glu.h"
#include "batched_tiny_vector_swi_glu.h"
#include "scalar_feature_swi_glu.h"
#include "small_m_vector_swi_glu.h"
#include "single_tiny_swi_glu.h"
#include "tiny_columns_swi_glu.h"
#include "sub_block_gemv4.h"
#include "scalar_channel_swi_glu.h"
#include "aligned_row_gemv32.h"
#include "aligned_col_gemv.h"
#include "batched_row_gemv.h"
#include "packed_multi_block_gemv.h"
#include "packed_multi_block_gemv_impl_1.h"
#include "packed_multi_block_gemv_impl_2.h"
#include "packed_sub_block_gemv.h"
#include "packed_sub_block_gemv_impl_1.h"
#include "packed_sub_block_gemv_impl_2.h"
#include "cube_slot_sync.h"
#include "direct_pair_cube.h"
#include "swi_glu_cube_impl.h"
#include "slim_direct_cube.h"
#include "unaligned_tile_cube.h"
#include "pair_cube_common.h"
#include "double_buffered_pair_cube.h"
#include "double_buffered_pair_cube_impl_1.h"
#include "double_buffered_pair_cube_impl_2.h"
#include "double_buffered_pair_cube_impl_3.h"
#include "pipelined_cube_swi_glu.h"
#include "pipelined_cube_swi_glu_impl_1.h"
#include "pipelined_cube_swi_glu_impl_2.h"
#include "pipelined_cube_swi_glu_impl_3.h"
#include "single_consumer_pair_cube.h"
#include "projection_stream_case3.h"
#include "single_consumer_swi_glu.h"
#include "swi_glu_dispatch.h"
#include "swi_glu_dispatch_impl_1.h"

template <typename DT_INPUT_X, int PATH>
__global__ __aicore__ void swi_glu(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                   GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    SwiGluDispatch<DT_INPUT_X, PATH> dispatch;
    dispatch.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
}
#if defined(TILING_KEY_VAR) && \
    (TILING_KEY_VAR == 11009UL || TILING_KEY_VAR == 11265UL || TILING_KEY_VAR == 11521UL || \
     TILING_KEY_VAR == 11777UL || TILING_KEY_VAR == 12033UL || TILING_KEY_VAR == 12289UL || \
     TILING_KEY_VAR == 12545UL || TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL) && \
    !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON)
namespace AscendC {
__aicore__ inline void NoSysWorkspaceForce(GM_ADDR)
{
}

__aicore__ inline GM_ADDR FastGetUserWorkspace(GM_ADDR workspace)
{
    return workspace + RESERVED_WORKSPACE;
}
}

#define SetSysWorkspaceForce NoSysWorkspaceForce
#define GetUserWorkspace FastGetUserWorkspace
#define cce_kernel cce_kernel, aligned(512), section(".text.hot")
#if TILING_KEY_VAR == 11009UL || TILING_KEY_VAR == 11265UL || TILING_KEY_VAR == 11777UL || \
    TILING_KEY_VAR == 12289UL || TILING_KEY_VAR == 12545UL || TILING_KEY_VAR == 12801UL
#define set_ffts_base_addr(...) ((void)0)
#endif
#endif
