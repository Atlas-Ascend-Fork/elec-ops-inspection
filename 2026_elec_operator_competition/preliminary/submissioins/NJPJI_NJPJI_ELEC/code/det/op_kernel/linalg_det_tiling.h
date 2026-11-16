// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct LinalgDetTilingData {
    uint32_t length;      // 模板占位字段（保留）
    uint32_t numMatrix;   // batch 维元素个数
    uint32_t n;           // 方阵阶数
    uint32_t stride;      // 行对齐后的行跨度（8 的倍数）
    uint32_t tileRows;    // 一次驻留 UB 的行数
    uint32_t blockNum;    // 启动核数
};
