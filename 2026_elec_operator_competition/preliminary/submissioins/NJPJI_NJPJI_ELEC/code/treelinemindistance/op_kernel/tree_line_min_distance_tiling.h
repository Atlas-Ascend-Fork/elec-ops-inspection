// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct TreeLineMinDistanceTilingData {
    uint32_t length;      // 模板占位字段（保留）
    uint32_t batch;       // B
    uint32_t numPoints;   // N
    uint32_t numSeg;      // M
    uint32_t tilePoint;   // 单 tile 树障点数
    uint32_t tileSeg;     // 单次搬入 UB 的线段数
    uint32_t blockNum;    // 启动核数
};
