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

#pragma once

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) GM_ADDR
SwiGluDispatch<DT_INPUT_X, PATH>::ResolveWorkspace(GM_ADDR workspace)
{
    if (workspace == nullptr) {
        return nullptr;
    }
    return AscendC::GetUserWorkspace(workspace);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::SpecializedRoute0(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                                    GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE(11009, KERNEL_TYPE_AIV_ONLY);
    SubBlockGemv4<DT_INPUT_X> op;
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::SpecializedRoute1(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                                    GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE(11265, KERNEL_TYPE_AIV_ONLY);
    AlignedColGemv<DT_INPUT_X, 64U, 2U> op;
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::SpecializedRoute2(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                                    GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    optiling::PipelinedCubeSwiGlu<DT_INPUT_X, 256U, 32U, 16U, 4U> op;
#elif (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    optiling::PipelinedCubeSwiGlu<DT_INPUT_X, 256U, 64U, 16U, 3U, 64U> op;
#else
    KERNEL_TASK_TYPE(11521, KERNEL_TYPE_MIX_AIC_1_1);
    optiling::SingleConsumerSwiGlu<DT_INPUT_X, 256U, 48U, 16U, 3U, 64U> op;
#endif
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::SpecializedRoute3(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                                    GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE(11777, KERNEL_TYPE_AIV_ONLY);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    AlignedColGemv<DT_INPUT_X, 32U, 1U> op;
#else
    AlignedRowGemv32<DT_INPUT_X> op;
#endif
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::SpecializedRoute4(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                                    GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    optiling::PipelinedCubeSwiGlu<DT_INPUT_X, 128U, 64U, 16U, 4U, 128U> op;
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::SpecializedRoute5(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                                    GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE(12289, KERNEL_TYPE_AIV_ONLY);
    ScalarChannelSwiGlu<DT_INPUT_X> op;
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::SpecializedRoute6(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                                    GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE(12545, KERNEL_TYPE_AIV_ONLY);
    PackedSubBlockGemv<DT_INPUT_X> op;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, tiling);
#else
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output);
#endif
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::SpecializedRoute7(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                                    GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE(12801, KERNEL_TYPE_AIV_ONLY);
    PackedMultiBlockGemv<DT_INPUT_X> op;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, tiling);
#else
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output);
#endif
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::SpecializedRoute8(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1,
                                                    GM_ADDR weight_2, GM_ADDR bias_2, GM_ADDR output,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    optiling::PipelinedCubeSwiGlu<DT_INPUT_X, 256U, 128U, 16U, 16U> op;
#else
    optiling::PipelinedCubeSwiGlu<DT_INPUT_X, 256U, 128U, 20U, 7U, 128U> op;
#endif
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::Route0(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                         GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (PATH == 43) {
        SpecializedRoute0(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH == 44) {
        SpecializedRoute1(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH == 45) {
        SpecializedRoute2(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH == 46) {
        SpecializedRoute3(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH == 47) {
        SpecializedRoute4(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH == 48) {
        SpecializedRoute5(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH == 49) {
        SpecializedRoute6(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH == 50) {
        SpecializedRoute7(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH == 51) {
        SpecializedRoute8(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    }
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::Route1(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                         GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE(10753, KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    BatchedTinyVectorSwiGlu<DT_INPUT_X, 13U, 7U, true> op;
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, nullptr);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::Route2(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                         GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (PATH == 38) {
        KERNEL_TASK_TYPE(9729, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 39) {
        KERNEL_TASK_TYPE(9985, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 40) {
        KERNEL_TASK_TYPE(10241, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 41) {
        KERNEL_TASK_TYPE(10497, KERNEL_TYPE_AIV_ONLY);
    }
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    if constexpr (PATH == 41) {
        GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
        TinyColumnsSwiGlu<DT_INPUT_X, 0U, false> op;
        op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, &tiling_data);
    } else {
        TinyColumnsSwiGlu<DT_INPUT_X, PATH == 39 ? 0U : 13U, PATH != 40> op;
        op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, nullptr);
    }
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::Route3(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                         GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (PATH == 36) {
        KERNEL_TASK_TYPE(9217, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 37) {
        KERNEL_TASK_TYPE(9473, KERNEL_TYPE_AIV_ONLY);
    }
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
    SmallMVectorSwiGlu<DT_INPUT_X, PATH == 36 ? 512U : 0U, 512U> op;
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, &tiling_data);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::Route4(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                         GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWorkspace = ResolveWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData, tiling_data, tiling);
    constexpr uint32_t fixedD = PATH == 32 ? 256U : (PATH == 33 ? 128U : 0U);
    optiling::SlimDirectCube<DT_INPUT_X, 64U, fixedD, PATH == 35> op;
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, userWorkspace, &tiling_data);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::Route5(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                         GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWorkspace = ResolveWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData, tiling_data, tiling);
    constexpr uint32_t fixedD = (PATH == 26 || PATH == 30) ? 256U : ((PATH == 27 || PATH == 31) ? 128U : 0U);
    optiling::SlimDirectCube<DT_INPUT_X, 128U, fixedD, PATH == 29, true, PATH >= 30> op;
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, userWorkspace, &tiling_data);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::Route6(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                         GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    if constexpr (PATH == 23) {
        KERNEL_TASK_TYPE(5889, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 24) {
        KERNEL_TASK_TYPE(6145, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 25) {
        KERNEL_TASK_TYPE(6401, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 25) {
        GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
        SingleTinySwiGlu<DT_INPUT_X, 0U> op;
        op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, &tiling_data);
    } else {
        SingleTinySwiGlu<DT_INPUT_X, PATH == 23 ? 13U : 1U> op;
        op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, nullptr);
    }
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::Route7(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                         GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (PATH == 21) {
        KERNEL_TASK_TYPE(5377, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 22) {
        KERNEL_TASK_TYPE(5633, KERNEL_TYPE_AIV_ONLY);
    }
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
    BatchedTinyVectorSwiGlu<DT_INPUT_X, PATH == 21 ? 13U : 0U, 4U> op;
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, &tiling_data);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::Route8(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                         GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (PATH == 17) {
        KERNEL_TASK_TYPE(4353, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 18) {
        KERNEL_TASK_TYPE(4609, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 19) {
        KERNEL_TASK_TYPE(4865, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 20) {
        KERNEL_TASK_TYPE(5121, KERNEL_TYPE_AIV_ONLY);
    }
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
    constexpr uint32_t fixedD = PATH == 17 ? 64U : (PATH == 18 ? 128U : (PATH == 19 ? 256U : 0U));
    SmallMVectorSwiGlu<DT_INPUT_X, fixedD> op;
    op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, &tiling_data);
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::Route9(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                         GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (PATH == 14) {
        KERNEL_TASK_TYPE(3585, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 15) {
        KERNEL_TASK_TYPE(3841, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 16) {
        KERNEL_TASK_TYPE(4097, KERNEL_TYPE_AIV_ONLY);
    }
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwiGluTinyTilingData, tiling_data, tiling);
    if constexpr (PATH == 15) {
        ScalarFeatureSwiGlu<DT_INPUT_X> op;
        op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, &tiling_data);
    } else {
        TinyVectorSwiGlu<DT_INPUT_X, PATH == 14 ? 13U : 0U> op;
        op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, &tiling_data);
    }
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::Route10(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                          GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (PATH == 0) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 4) {
        KERNEL_TASK_TYPE(1025, KERNEL_TYPE_AIV_ONLY);
    }
    if constexpr (PATH == 5) {
        KERNEL_TASK_TYPE(1281, KERNEL_TYPE_AIV_ONLY);
    }
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
    constexpr int mode = PATH == 4 ? 1 : (PATH == 5 ? 2 : 0);
    KernelSwiGlu<DT_INPUT_X, mode> op;
    op.Init(input_x, weight_1, bias_1, weight_2, bias_2, output, tiling_data);
    op.Process();
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::Route11(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                          GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWorkspace = ResolveWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(SwiGluTilingData);
    if constexpr (PATH == 2 || PATH == 3 || PATH >= 8) {
        GET_TILING_DATA_WITH_STRUCT(SwiGluDirectTilingData, tiling_data, tiling);
        constexpr uint32_t tileM = (PATH == 3 || PATH == 9 || PATH == 11 || PATH == 13) ? 128U : 32U;
        constexpr uint32_t fixedD =
            (PATH == 8 || PATH == 9) ? 256U : ((PATH == 10 || PATH == 11) ? 128U : 0U);
        optiling::SlimDirectCube<DT_INPUT_X, tileM, fixedD, PATH == 12 || PATH == 13> op;
        op.Run(input_x, weight_1, bias_1, weight_2, bias_2, output, userWorkspace, &tiling_data);
    } else {
        AscendC::TPipe tPipe;
        GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tiling_data, tiling);
        optiling::RunCube<DT_INPUT_X, PATH == 6 || PATH == 7, PATH == 7>(
            input_x, weight_1, bias_1, weight_2, bias_2, output, userWorkspace, &tPipe, &tiling_data);
        tPipe.Destroy();
    }
}

template <typename DT_INPUT_X, int PATH>
__aicore__ inline __attribute__((always_inline)) void
SwiGluDispatch<DT_INPUT_X, PATH>::Run(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2,
                                      GM_ADDR bias_2, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (PATH >= 43) {
        Route0(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH == 42) {
        Route1(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH >= 38) {
        Route2(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH >= 36) {
        Route3(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH >= 32) {
        Route4(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH >= 26) {
        Route5(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH >= 23) {
        Route6(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH == 21 || PATH == 22) {
        Route7(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH >= 17) {
        Route8(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH >= 14) {
        Route9(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else if constexpr (PATH == 0 || PATH == 4 || PATH == 5) {
        Route10(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    } else {
        Route11(input_x, weight_1, bias_1, weight_2, bias_2, output, workspace, tiling);
    }
}
