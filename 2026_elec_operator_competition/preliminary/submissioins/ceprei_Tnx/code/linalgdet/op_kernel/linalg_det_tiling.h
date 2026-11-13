/*
 * Copyright (c) 2026 ceprei_Tnx.
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

// 【v28 开关】把"小 shape"那一组 tiling key 的 Kernel 类型显式声明为 AIV_ONLY：
//   官方文档：显式声明 AIV_ONLY 后调度器会跳过 AIC 核的适配检查，减少调度开销；
//             不必要的核类型声明会增加编译产物体积和调度开销。
//   本 op 的 AIC 侧在两条路径里都直接 return（不做事），却仍按 mix 核派发。
//   置 0 即完全回到上一版的单 key 形式（tiling_key_linalg_det.h / host 一起由本宏切换）。
#ifndef DET_AIV_KEY
#define DET_AIV_KEY 1
#endif

// Shared host/device layout switch; keep it outside template-selection metadata.
#ifndef DET_UB_FACTOR_ALIAS
#define DET_UB_FACTOR_ALIAS 1
#endif
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
struct LinalgDetTilingData {
    TCubeTiling cubeTiling;
    uint64_t coreWsElems;
    uint32_t n;
    uint32_t batch;
    uint32_t stride;
    uint32_t tileRows;
    uint32_t matPerCore;
    uint32_t resident;
    uint32_t cubeEnabled;
    uint32_t scratchStride;
};

// 每"次调用"固定开销相关的开关（探针测得 F≈11~16us，其中与形状无关部分 a≈8~10us）。
#ifndef DET_OFFS_INIT
#define DET_OFFS_INIT 1 // Gather 列偏移表：每矩阵重建 -> 每核建一次
#endif
// v40b: the pure-vector key never touches GetUserWorkspace, so declaring 0
// removes the 16MB-class system workspace allocation from the timed path.
// The cube key still declares the full lib workspace.
#ifndef DET_WS_MIN
#define DET_WS_MIN 1
#endif

// 【v35】CompactResident 每个 pivot 省掉一次 Gather（v31 的正确修法）。
// v31 错在：行交换夹在两次 Gather 之间，被换上去那行的系数必须取交换后的值。
// v35：把行交换挪到更新之后 —— 用「更新后的行 k」复制到行 row。
//   行 k 的更新结果 = old_h[k] − (a[0]/pv)·U，恰等于原方案中该行应得的值；
//   行 k 此后不再被任何 Gather/更新读取（下一轮从 k+1 起），是死行。
//   行 k+1..n-1 的 Mul/Sub 操作数与顺序完全不变 → 逐位相同（纯 JS fp32 逐拍比对已验证）。
//   置 0 即完全回到 v28。
#ifndef DET_MERGE_GATHER
#define DET_MERGE_GATHER 1
#endif

// 【v36】把 n<=128 的立方形状改走整块常驻 UB 的向量路径。
// 实测：立方路径在 n<=128 时效率只有 1/26~1/140（面板宽 8 × K=32 的 overhead），
//       而纯向量路径按 W=0.515·n 外推只要 ~49us(n=96)。#6/#12 恰在 64<n<128。
// 只影响 n∈(64,128]：n<=64 的 resident 点与 n>=128 的立方点行为不变。
// 置 64 即完全回到原阈值。
#ifndef DET_CUBE_MIN_N
#define DET_CUBE_MIN_N 128U
#endif

// 【v36】resident 路径把 broadcast_ 缩到 1024 float（CompactResident 的 Brcb 只需前 512），
// 腾出 UB 让 n 到 128 也能整块常驻。置 0 恢复 4*scratchStride。
#ifndef DET_BCAST_MIN
#define DET_BCAST_MIN 1
#endif

// 【v36】NormalizeResidentRows 支持 stride 65..128：先把每行后半段 max 折进前半段，
// 再按 64 lane 归约。max 满足结合律，折叠不引入舍入 → 与原实现精确等价。置 0 则只支持 stride<=64。
#ifndef DET_WIDE_NORM
#define DET_WIDE_NORM 1
#endif

// 【v38】resident 路径扩容：矩阵只保留一个平面（低半平面在 lean resident 下恒为零且从不被读），
// NormalizeResidentRows 改为按 scratch 容量分块处理，scratchStride 由 32*stride 降到 4*stride。
// 效果：n<=192 都能整块常驻 UB，走向量 LU；n>192 仍回退 cube 路径。
// 置 0 完全回到 v36 的 2 平面/整块归一化布局。
#ifndef DET_RESIDENT_WIDE
#define DET_RESIDENT_WIDE 1
#endif

// Diagnostic only: adds n*1000 vector ops to every active core so the reported
// per-case time encodes n (delta / unit). Must stay 0 for real submissions.
#ifndef DET_SHAPE_PROBE
#define DET_SHAPE_PROBE 0
#endif

// 【v40】resident 主更新改为单遍 FMA：把主元行取负后直接 MulAddDst(dst, q, -U)，
// 省掉 product 平面的一次写和一次读，更新指令数近似减半。置 0 回到 Mul+Sub。
#ifndef DET_FMA_UPDATE
#define DET_FMA_UPDATE 1
#endif

// Diagnostic only: skip all LU work and just write the output, to measure the
// launch + tiling + Init floor. Must stay 0 for real submissions.
#ifndef DET_ABLATE_EMPTY
#define DET_ABLATE_EMPTY 0
#endif

// Fast-path experiment: use the diagonal element as the pivot (no column max
// reduction). The determinant identity is unchanged; only numerical pivoting is
// skipped. Must be paired with a fallback before production use.
#ifndef DET_NO_PIVOT_PROBE
#define DET_NO_PIVOT_PROBE 0
#endif

// 【v41】resident 纯向量无标量停顿 LU：对角主元 + 向量 Div 求系数 +
// 向量累乘行列式，主循环没有 V_S/S_V；只在每 16 个 pivot 和结尾做一次标量归一化。
// 若任一 |pivot|<0.05（等价于 cube 路径的 retry 阈值），调用方重新加载并走带主元的路径。
#ifndef DET_NO_PIVOT_VEC
#define DET_NO_PIVOT_VEC 0
#endif

// Diagnostic only: replace the strided column Gather with a contiguous read of
// row k (same element count, wrong values) to measure Gather cost.
#ifndef DET_ABLATE_GATHER
#define DET_ABLATE_GATHER 0
#endif

// 【v44】cube 面板改用对角主元：不再做列最大值搜索，也不做每 pivot 的 GM 行交换；
// 面板内的行仍保持 k..n-1 连续。若对角元 <1e-3，则本次矩阵回退到带主元的 cube 路径。
// 置 0 回到原先的 partial-pivot cube 面板。
// Diagnostic only: skip the trailing FMA update in the resident no-pivot path.
#ifndef DET_ABLATE_UPDATE
#define DET_ABLATE_UPDATE 0
#endif

#ifndef DET_CUBE_NO_PIVOT
#define DET_CUBE_NO_PIVOT 1
#endif

// 【v46】cube 尾部更新改用单次 FP32 乘（K=16，8 个真实 K 槽 + 8 个恒零槽），
// 去掉 4 项补偿的拆分/PackCubeA，K 从 32 降到 16。
// 置 0 回到 4 项补偿（K=32）。
// 【v48】resident 无主元路径跳过整行 2 的幂均衡；每个 pivot 仍单独 NormalizePivot 并累计指数。
// 省掉一遍 Abs/ReduceMax/标量缩放/Brcb/Mul。置 0 恢复行均衡。
// 【v50】AIV_ONLY kernel 只走 resident 路径，编译掉 Blocked<> 重试路径，缩小二进制/减轻 I-cache。
#ifndef DET_AIV_LEAN
#define DET_AIV_LEAN 0
#endif

#ifndef DET_SKIP_EQUIL
#define DET_SKIP_EQUIL 1
#endif

// 【v52】cube 面板宽度 16（两个 8 列 plane），matmul K=64，调用次数减半。
#ifndef DET_CUBE_B16
#define DET_CUBE_B16 0
#endif

#ifndef DET_CUBE_PLAIN
#define DET_CUBE_PLAIN 0
#endif
#define DET_CUBE_K (DET_CUBE_B16 ? 64 : (DET_CUBE_PLAIN ? 16 : 32))
