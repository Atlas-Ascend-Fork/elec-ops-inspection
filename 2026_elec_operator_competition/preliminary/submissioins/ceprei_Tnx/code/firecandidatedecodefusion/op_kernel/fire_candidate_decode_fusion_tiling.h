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
// Tiling结构体定义的头文件（v10 分支：向量侧由「行车道」改为「列平面」）
//
// ── v16 一句话说明 ───────────────────────────────────────────────────────────
// v16 只动 Kernel 小档路径的**索引表生成方式**（改成与列平面同源的向量自加翻倍），
// 结构体字段、常量取值、Host/Kernel 之间的 ABI **全部不变**，因此本文件相对 v15
// 只多了一段说明文字，没有任何语义改动。
// 唯一需要留意的关联量是小档路径的零区尺寸 —— 它不在本文件里，而是 Kernel 的
// PRIOR_ZERO_PAD_BYTES / SIG_ZERO_PAD_BYTES，用的是小档自己的 MAX_ROWS = 48。
//
// 与 4files（v5，上机 100.00）、v6、v7、v8、v9 的差别只在常量的取值与解释：
// **结构体字段完全不变，Host 侧一行未改** —— v8 以来的改动都是纯 Kernel 侧改动。
// v10 连常量都没动：FIRE_MAX_TILE_ROWS 仍是 224、GROUP 仍是 64 ——
// v10 改的是「同一份数据在 UB 里怎么摆」，摆法不改变 tiling 的语义。
// tiling 的语义（按 priors 行数 N 切、tileRows 上限）不变，Host/Kernel 之间的
// ABI 不变，两边可独立重编。
//
// ── v8 相对 v7 唯一的实质改动：把向量指令的「行宽」从 8/16 统一压到 6 ──────
// v7 上机 12/12 Pass（fp32 各点 4.46/20.17/73.56μs，fp16 4.73/42.14/132.66μs），
// 但逐点比 rank1 慢 16%（fp32）到 2.4 倍（fp16）。差距的根因有一条极硬的证据：
//   **fp32 与 fp16 在 v7 里不是同速的**（中档 20.17 vs 42.14 = 2.09x，
//     大档 73.56 vs 132.66 = 1.80x），而 rank1（dihidao）两种 dtype 逐点几乎相同
//     （17.39/17.58、63.26/63.19）。
// 这说明 v7 的时间**随「向量指令覆盖的元素数」变化**，而 fp16 的元素数被浪费了一倍：
//   v7 的 LANES = fp16 ? 16 : 8，于是 count = rows*LANES 在 fp16 下是 rows*16，
//   而所有中间缓冲（sig / acc / p3 / xmul / strideW / strideC）都是 **float** 类型
//   ⇒ fp16 档的向量指令覆盖的字节数是 fp32 的 2 倍（rows*64B vs rows*32B）。
// 更糟的是 fp16 的 16 条车道里只有 6 条是真字段（10 条哑车道）。
//
// 为什么 v7 要用 8/16 而不是 6？追根溯源是 v5 留下的一条约束：UB 落点被定成
// 「一行一个 32B block」。**这条约束在 v7 里已经不存在了** —— 搬运用的是
// DataCopyPad 的「GROUP 行一个大 block、dstStride = 0」写法，UB 落点本来就是
// **紧凑数组**（行距 = 载荷字节数 24B / 12B），32B 只要求 **块的起点**对齐，不要求
// 每一行的起点对齐。而 `rawExp` / `sig` / `acc` 这些中间缓冲**根本不经搬运**，
// 只被向量指令读写，行距取多少完全自由。
// ⇒ v8 把行宽统一成 **6 = RAW_FIELDS**，于是：
//   ① fp16 的向量元素数从 rows*16 降到 rows*6（2.67 倍）；
//   ② fp32 从 rows*8 降到 rows*6（1.33 倍），而且哑车道消失了；
//   ③ 「展开 Gather」变成恒等映射 —— 紧凑 [rows,6] 就是最终行布局，
//      那次 Gather（v7 每个 item 一次、覆盖 rows*8）**整个删掉**。
//
// ── 行宽 6 与 32B 对齐的关系（本分支唯一的对齐风险点）─────────────────────
// 硬约束只有一条（官方 API 文档）：**DataCopyPad 的 UB 端起始地址必须 32B 对齐**
// （blockLen 可以非 32B 对齐）。本分支里被 DataCopyPad 碰到的只有三块：
//   raw 紧凑区、priors 紧凑区、out 紧凑区 —— 它们各自的行距是 24/20/12/10B，
//   但搬运永远是「GROUP 行一个 block、起点 32B 对齐、dstStride = 0」，
//   所以 UB 里得到的是一个**紧凑数组**，行的起点本来就不在 32B 边界上，也不需要。
//   其余缓冲（sig/acc/... 与 6 张索引表）各自**起点**都在 32B 边界上（下面
//   WORKSPACE_BYTES 的每一项都做了 AlignUp32），因此满足向量指令与 Gather 的
//   地址对齐要求；Gather 的 srcOffset 全部按 4 字节（float 元素）对齐。
#pragma once

#include <cstdint>

struct FireCandidateDecodeFusionTilingData {
    uint32_t batchSize;     // B，priors 在 batch 维共享
    uint32_t numCandidates; // N
    uint32_t totalRows;     // B * N
    uint32_t rowsPerCore;   // 单核负责的 priors 行数（N 的一份，核内在其上遍历 B）
    uint32_t tileRows;      // 单个 tile 的 priors 行数上限，不超过 FIRE_MAX_TILE_ROWS
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
};

// ── 一个 tile 的行数上限 ─────────────────────────────────────────────────────
// v8 把两种 dtype 统一到同一个档位：行宽压到 6 之后，fp16 的 float 侧缓冲不再
// 比 fp32 大一倍（两者都是 MAX_ROWS*6*4），UB 预算因此减半，fp16 也能吃下
// v7 只给 fp32 的 224 行（v7 的 fp16 档是 160）。
// Host 与 Kernel 共用这个常量（Host 用它切 tileRows，Kernel 再用自己的 MAX_ROWS
// 夹一次），所以两边不会不同步。
//
// UB 预算（每个 AIV 独享 192KB）：当前大路径以生命周期复用后的 11 个列平面、
// priors/raw 各一张共享行偏移表和一张 5 列压紧表组成；832 行时 fp32/fp16 工作区
// 分别为 187104B/187168B，仍保留 9KB 以上物理余量。
// 大 shape 路径按 910B 的 192KB UB 重新做了生命周期预算。Kernel 将 priors 展布阶段
// 的临时平面与后续 decode 输出平面复用，并且只在 fp16 实例中保留 Cast 中转区；
// priors/raw 的多列 Gather 共用行偏移表后，两种 dtype 在 832 行时仍可为栈与运行时
// 留出 8KB 以上空间。
//
// ── v16.2：832 → 416 的对照已上机，结论是**证伪**，故回退到 832 ────────────────
// 上机 A/B（其余逐字相同，只改这一个常量）：
//   fp32 大档  25.12 → 26.04μs（+0.92）
//   fp16 大档  24.31 → 27.33μs（+3.02）
//   中/小档    7.63~7.84 → 7.62~7.80（在噪声内，符合预期：它们本来就是 1 个 tile）
// 三个推论：
//   ① 把大档拆成「两个 tile、每 tile 412 行」是**净亏**的。
//      ⇒ 「一个 item 无法流水」这个假设不成立 —— 若真是流水不足，拆成 2 个 item 只会变快。
//   ② 增量为正且与行块大小无关，说明成本落在**每个 tile 的固定量**上（展布那一组
//      Gather + 向量指令、以及多出来的一轮搬运事务），而不是落在算力量上。
//   ③ fp16 的增量是 fp32 的 3.3 倍 ⇒ fp16 每个 item 的两次 Cast（6·rows + 5·rows）
//      是**每 item 可见**的成本，而不是被完全掩盖的。
// 因此回到 832：大档保持一个 tile，同时把「每 tile 固定量」压到最低。
constexpr uint32_t FIRE_MAX_TILE_ROWS = 832U;

// 每核 priors 行数的下限（Host 侧只用来兜住 ceil 出来的 0）。
constexpr uint32_t FIRE_MIN_ROWS_PER_CORE = 1U;

// ── 整块搬运的分组粒度：一个 MTE block 装多少行 ──────────────────────────────
// GROUP 必须让「三种行宽 × GROUP」都是 32B 的整数倍，这样 UB 侧的落点才是一个
// **紧凑数组**（行距 = 载荷字节数），而不是「一行一个 32B block」：
//   fp32: raw 6×4=24B, priors 5×4=20B, out 5×4=20B
//   fp16: raw 6×2=12B, priors 5×4=20B, out 5×2=10B
// 取 GROUP=64：24×64=1536B、20×64=1280B、12×64=768B、10×64=640B —— 全是 32B 倍数。
// 实机标定过「小事务」的代价与字节数几乎无关，于是每 64 行只付一次事务。
constexpr uint32_t FIRE_PACK_GROUP = 64U;

// UB 容量预算（Ascend910B1：每个 AIV 独享 192KB）
// v8 的静态布局：两组 raw slot（各含 raw 紧凑区 + priors 紧凑区）、两组 out slot、
// 9 个 float 向量缓冲（sig/p/t/v/u/wtab/strideC/strideW/g，行宽 6）、6 张索引表。
constexpr uint32_t FIRE_UB_BLOCK_BYTES = 32U;
