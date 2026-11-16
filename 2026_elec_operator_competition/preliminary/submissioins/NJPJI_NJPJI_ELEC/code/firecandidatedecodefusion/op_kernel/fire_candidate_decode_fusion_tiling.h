// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct FireCandidateDecodeFusionTilingData {
    uint32_t length;     // 模板占位字段（保留）
    uint32_t batch;      // B
    uint32_t numCand;    // N
    uint32_t tileLen;    // 单 tile 候选数
    uint32_t blockNum;   // 启动核数
};
