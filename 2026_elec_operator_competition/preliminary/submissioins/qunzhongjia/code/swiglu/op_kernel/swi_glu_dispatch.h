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

template <typename DT_INPUT_X, int PATH> class SwiGluDispatch {
public:
    __aicore__ inline __attribute__((always_inline)) void
    SpecializedRoute0(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
                      GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void
    SpecializedRoute1(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
                      GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void
    SpecializedRoute2(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
                      GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void
    SpecializedRoute3(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
                      GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void
    SpecializedRoute4(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
                      GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void
    SpecializedRoute5(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
                      GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void
    SpecializedRoute6(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
                      GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void
    SpecializedRoute7(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
                      GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void
    SpecializedRoute8(GM_ADDR input_x, GM_ADDR weight_1, GM_ADDR bias_1, GM_ADDR weight_2, GM_ADDR bias_2,
                      GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void Route0(GM_ADDR input_x, GM_ADDR weight_1,
                                                                 GM_ADDR bias_1, GM_ADDR weight_2,
                                                                 GM_ADDR bias_2, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void Route1(GM_ADDR input_x, GM_ADDR weight_1,
                                                                 GM_ADDR bias_1, GM_ADDR weight_2,
                                                                 GM_ADDR bias_2, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void Route2(GM_ADDR input_x, GM_ADDR weight_1,
                                                                 GM_ADDR bias_1, GM_ADDR weight_2,
                                                                 GM_ADDR bias_2, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void Route3(GM_ADDR input_x, GM_ADDR weight_1,
                                                                 GM_ADDR bias_1, GM_ADDR weight_2,
                                                                 GM_ADDR bias_2, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void Route4(GM_ADDR input_x, GM_ADDR weight_1,
                                                                 GM_ADDR bias_1, GM_ADDR weight_2,
                                                                 GM_ADDR bias_2, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void Route5(GM_ADDR input_x, GM_ADDR weight_1,
                                                                 GM_ADDR bias_1, GM_ADDR weight_2,
                                                                 GM_ADDR bias_2, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void Route6(GM_ADDR input_x, GM_ADDR weight_1,
                                                                 GM_ADDR bias_1, GM_ADDR weight_2,
                                                                 GM_ADDR bias_2, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void Route7(GM_ADDR input_x, GM_ADDR weight_1,
                                                                 GM_ADDR bias_1, GM_ADDR weight_2,
                                                                 GM_ADDR bias_2, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void Route8(GM_ADDR input_x, GM_ADDR weight_1,
                                                                 GM_ADDR bias_1, GM_ADDR weight_2,
                                                                 GM_ADDR bias_2, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void Route9(GM_ADDR input_x, GM_ADDR weight_1,
                                                                 GM_ADDR bias_1, GM_ADDR weight_2,
                                                                 GM_ADDR bias_2, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void Route10(GM_ADDR input_x, GM_ADDR weight_1,
                                                                  GM_ADDR bias_1, GM_ADDR weight_2,
                                                                  GM_ADDR bias_2, GM_ADDR output,
                                                                  GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void Route11(GM_ADDR input_x, GM_ADDR weight_1,
                                                                  GM_ADDR bias_1, GM_ADDR weight_2,
                                                                  GM_ADDR bias_2, GM_ADDR output,
                                                                  GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline __attribute__((always_inline)) void Run(GM_ADDR input_x, GM_ADDR weight_1,
                                                              GM_ADDR bias_1, GM_ADDR weight_2,
                                                              GM_ADDR bias_2, GM_ADDR output,
                                                              GM_ADDR workspace, GM_ADDR tiling);

private:
    __aicore__ inline __attribute__((always_inline)) GM_ADDR ResolveWorkspace(GM_ADDR workspace);
};
