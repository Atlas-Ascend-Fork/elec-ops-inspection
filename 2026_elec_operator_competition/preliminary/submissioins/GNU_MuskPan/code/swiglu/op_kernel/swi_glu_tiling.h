// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// SwiGLU 算子的 Tiling 数据结构（host 侧填充、kernel 侧读取）
//
// 算子语义: out[i, :] = swish(x[i, :] * W1^T + b1) * (x[i, :] * W2^T + b2)
//   swish(v) = v * sigmoid(v) = v / (1 + exp(-v))
// 其中 x: (m, d) fp16, W1/W2: (d, d) fp16 (行主序), b1/b2: (d,) fp16,
// out: (m, d) fp16, m = 前导维度之积, d = 最后一维.
//
// 切分: 行按核分（rowsPerCore），行块大小 mb，输出列块大小 nj，
//       K 方向按 kPad 切块并累加（acc1/acc2 为 fp32 累加器）。
struct SwiGluTilingData {
    uint32_t length;       // input_x 元素总个数（保留字段）
    uint32_t m;            // 行数（前导维度之积，>= 1）
    uint32_t d;            // 内维（= 输出维，>= 1）
    uint32_t kPad;         // K 方向分块长度（16 的倍数，<= 256）
    uint32_t nj;           // 输出列分块大小（<= d）
    uint32_t mb;           // 行分块大小（<= m）
    uint32_t numKBlocks;   // K 分块个数 = ceil(d / kPad)
    uint32_t rowsPerCore;  // 每核处理的行数
    uint32_t coreNum;      // 实际启动核数
    // 以下为 UB 各 buffer 的字节数（host 计算，保证 host/kernel 一致）
    uint32_t wBytes;       // 3 * nj * kPad * 4  (w1f32 / w2f32 / p32)
    uint32_t xBytes;       // mb * kPad * 4       (xf32)
    uint32_t accBytes;     // align8(mb * nj) * 4  (acc1 或 acc2 之一)
    uint32_t smallBytes;   // 7 * align8(nj) * 4  (sums/b1f/b2f/z1f/swf/z2f/o32f)
    uint32_t stageBytes;   // max(nj, mb) * kPad * 2 (fp16 暂存队列)
    uint32_t outBytes;     // align32(nj * 2)      (fp16 输出队列)
    uint32_t tmpBytes;     // Pattern ReduceSum 所需临时空间
    // 以下为近零修正（refine）相关参数
    uint32_t refineChunk;  // 重算时分块长度 CH = min(d, 512)
    uint32_t refineBytes;  // 4 * CH * 4  (refXf / refW1f / refW2f / refPf)
    float refineRel;       // 近零相对阈值系数: T = refineRel * maxAbsZ
};
