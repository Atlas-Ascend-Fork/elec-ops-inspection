// Cube(Matmul) 路径的 host 侧 tiling 构建。
//
// 只有定义了 SWIGLU_CUBE_TILING_AVAILABLE 的调用方（算子 host、CPU 调测工程）才会真正包含
// CANN 的 matmul tiling 头并使用 tiling API；未定义时给出恒失败的 stub，因此纯 C++ 环境
// （如 tiling 遍历工具）仍可编译同一份 swi_glu_tiling_search.h。
//
// 已由原型工程在 CANN 9.0.0 + tikicpulib 上验证：
//   * 权重按 (out_dim, in_dim) 行主序存放时，B 必须同时声明为转置：
//     模板 MatmulType<..., true> 与运行时 SetTensorB(gB, true)；
//   * 不使用 Matmul 自带 bias（它在 fp32 加 bias 且只取整一次，与判题 fp16 链语义不同）；
//   * C 的前导维必须是 d —— 本算子输出即 (m, d) 行主序，天然满足。
#pragma once

#include <cstdint>
#include <cstring>

#if defined(SWIGLU_CUBE_TILING_AVAILABLE)
#include "tiling/matrix/matmul_tiling.h"
#include "tiling/matrix/bmm_tiling.h"
#endif

namespace swiglu_cube {

constexpr uint32_t CUBE_TILING_BYTES = 256;                            // TCubeTiling 序列化上限
constexpr uint64_t CUBE_WS_LIMIT = 2ull * 1024ull * 1024ull * 1024ull;  // z2 workspace 上限 2GB
constexpr uint32_t CUBE_EPI_ROWS = 32;                                 // epilogue 行分块
constexpr uint32_t CUBE_EPI_COLS_MAX = 128;                            // epilogue 列分块上限（16 的倍数）
// 列分块上限取 128（= 一个矢量 repeat 的 fp16 lane 数）：kernel 用 Level-0 掩码 Add 一次给整块
// 的所有行加 bias，一个 repeat 恰好处理一行，因此 ccA = align16(cc) 必须不超过 128。若放宽到
// 256，一行会跨两个 repeat，而掩码只能描述单个 repeat 内的 128 个 lane ⇒ 无法用一条指令表达行距。


// epilogue 分块形状：列方向尽量覆盖整个 singleCoreN（列块数 = 「加 bias + 整块链」的执行次数），
// 行方向固定 32（2D DataCopyPad 的 strided 行数保持已被真机 9/9 验证过的规模，不再加大）。
// kernel 侧用 ccA = align16(cc) 作 UB 行距，故 cbCols 必须是 16 的倍数且 >= ccA。
inline void EpiChunkShape(uint32_t scN, uint32_t *cbRows, uint32_t *cbCols)
{
    uint32_t cols = ((scN + 15u) / 16u) * 16u;
    if (cols == 0u) {
        cols = 16u;
    }
    if (cols > CUBE_EPI_COLS_MAX) {
        cols = CUBE_EPI_COLS_MAX;
    }
    *cbRows = CUBE_EPI_ROWS;
    *cbCols = cols;
}

// 构造 Cube 路径所需的 TCubeTiling 并写入 blob。
// 返回 0 表示可用；负值表示前提不满足，调用方回退纯矢量路径。
inline int32_t BuildTiling(uint32_t m, uint32_t d, int32_t dim, uint8_t *blob)
{
#if defined(SWIGLU_CUBE_TILING_AVAILABLE)
    if (dim <= 0 || m == 0u || d == 0u) {
        return -1;
    }
    matmul_tiling::MultiCoreMatmulTiling mm;
    mm.SetDim(dim);
    mm.SetShape(static_cast<int32_t>(m), static_cast<int32_t>(d), static_cast<int32_t>(d));
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT16, false);
    // B = W（(out_dim, in_dim) 行主序）⇒ isTrans = true（必须与 kernel 的 SetTensorB(g,true) 一致）
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT16, true);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_FLOAT16);
    AscendC::tiling::TCubeTiling t;
    if (mm.GetTiling(t) != 0) {
        return -2;
    }
    // 严格校验返回的 tiling：MultiCoreMatmulTiling 对某些形状会返回 rc==0 但未填结构体
    // （CPU 实测 d=8192/d=4096 等 K 很大的形状），因此必须逐字段确认可用。
    if (t.M != static_cast<int32_t>(m) || t.N != static_cast<int32_t>(d) ||
        t.Ka != static_cast<int32_t>(d) || t.Kb != static_cast<int32_t>(d)) {
        return -5;
    }
    if (t.singleCoreM <= 0 || t.singleCoreN <= 0 || t.singleCoreK <= 0 || t.usedCoreNum <= 0 ||
        t.baseM <= 0 || t.baseN <= 0 || t.baseK <= 0) {
        return -3;
    }
    // 注意：**不校验 singleCoreM <= m，也不校验 singleCoreN <= d**。
    // MultiCoreMatmulTiling 对 tiny-M 形状会把 singleCoreM 抬到 baseM（实测 M<=16、N=8192 时
    // scM=16 > M、scN=N/dim，即把切分全放在 N 维上）；对非 16 倍数的 N 也会把 singleCoreN/baseN
    // 向上取整到 16 的倍数（d=177 → scN=192、d=13 → scN=16）。这对我们是安全的：kernel 侧每个
    // tile 都先取 mT = min(scM, M-mOff)、nT = min(scN, N-nOff) 再 SetSingleShape，matmul 绝不会
    // 写到真实 M/N 之外（已用 MM-only 通道逐位验证：m<16 与 scN>d 两类形状的 z1/z2 与 fp16 参考
    // 位一致，说明 fixpipe 没有溢出到相邻行列）。
    // kernel 侧 epilogue 按「列块 cc 为 16 的倍数」布局 UB（DataCopyPad 每行自动补齐到 32B，
    // 因此 UB 行距 = align16(cc)）；singleCoreK 必须等于完整 K，否则需要自己做 K 累加。
    if (t.singleCoreK != static_cast<int32_t>(d)) {
        return -7;
    }
    const uint32_t scM = static_cast<uint32_t>(t.singleCoreM);
    const uint32_t scN = static_cast<uint32_t>(t.singleCoreN);
    // 注意：singleCoreN 不要求是 16 的倍数 —— kernel epilogue 用 align16(cc) 作 UB 行距
    // （见 EpiChunk 的 ccA）。CPU 实测 scN 常为 205/410/171 这类值。
    const uint32_t nTile = (d + scN - 1u) / scN;
    const uint32_t mTile = (m + scM - 1u) / scM;
    if (static_cast<uint32_t>(dim) < nTile * mTile) {
        return -6;  // tile 数超过启动核数，会漏算 ⇒ 回退
    }
    std::memset(blob, 0, CUBE_TILING_BYTES);
    std::memcpy(blob, &t, sizeof(t));
    return 0;
#else
    (void)m;
    (void)d;
    (void)dim;
    (void)blob;
    return -99;
#endif
}

}  // namespace swiglu_cube
