// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// 两个 tiling key 维度：
//   1) SWIGLU_CUBE = 1 -> Cube（Matmul/MIX_AIC_1_2）路径；= 0 -> 纯矢量回退路径
//   2) DT_INPUT_X  -> 输入 dtype（题目只要求 float16）
// 每个 ARGS_SEL 都必须带 ASCENDC_TPL_KERNEL_TYPE_SEL（CANN 强制：要么全带、要么全不带），
// 这样两种路径可以用**各自的 kernel 类型**独立编译：
//   * Cube 路径编成 MIX_AIC_1_2（每 block 1 AIC + 2 AIV，KFC 负责跨核同步）；
//   * 矢量回退路径编成 AIV_ONLY，不再被 MIX 包装（省掉 clearWorkspace、ffts 及
//     "blockDim 只能是 AIC 数" 的约束 —— 矢量路径的并行度按 AIV 数下发）。
ASCENDC_TPL_ARGS_DECL(SwiGlu,
    ASCENDC_TPL_UINT_DECL(SWIGLU_CUBE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_DATATYPE_DECL(DT_INPUT_X, C_DT_FLOAT16),
);

ASCENDC_TPL_SEL(
    // 矢量回退路径：AIV_ONLY
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(SWIGLU_CUBE, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
    ),
    // Cube 路径：MIX_AIC_1_2
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
        ASCENDC_TPL_UINT_SEL(SWIGLU_CUBE, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
    ),
);
