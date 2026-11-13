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
// Kernel侧核函数实现（v17：把「每核固定串行段」压掉）
//
// ════════════════════════════════════════════════════════════════════════════
// ── v17 的依据：榜单给出的是「均匀的 ~20%」，不是某一块塌方 ──────────────────
// 榜单（同一台机器、同一时刻的四个队）：前两名的成绩基本贴着 TBest，
// 第三名与我们分别落后 15~25%。把差距拆成**绝对值**看：
//     TBest 3.00 / 6.42 / 20.46
//     我们  4.29 / 7.64 / 24.98   差距 1.29 / 1.22 / 4.52 μs
// 小档与中档的**绝对差几乎一样（1.29 vs 1.22）**，而两档的总时间差了 2 倍多 ——
// 这是典型的「一个与 shape 无关的每核固定成本」，而不是算力量不够。
// 另外「48 / 40 = 1.20」恰好落在大档/中档差距（1.22 / 1.19）上，
// 所以并列第一条要验的是核数（见 op_host 的 FORCE_AIV_CORES）。
//
// v17 因此只动「每核固定串行段」，一行算法都没碰：
//   1. 收尾的 PipeBarrier<PIPE_ALL> 收窄成 PipeBarrier<PIPE_MTE3>。
//      最后一次 store 之前，向量侧的结果已经用 V_MTE3 事件对齐、MTE2 的预取也已被消费，
//      真正还需要等的只有 MTE3 的写回队列。
//   2. item 循环里的整数除法全部拆掉（标量单元没有硬件除法）：tile / b / rows 改成
//      「跑表」推进，只在 tile 边界重算一次；IssueItemLoad / IssueItemStore 改成直接收
//      (tile, b, rows)；Init 里的 B 改成直接取 Host 写好的 tiling.batchSize。
//
// ── 一个改过又撤掉的方案（记下来，别再走一遍）───────────────────────────────
// 曾把 BuildIndexTables 从 Init 挪到 Process 里、第一笔 DMA 之后，想拿「建表与 DMA 重叠」。
// 撤掉的原因：BuildIndexTables 内部用 S_V/ID0 做标量→向量同步，而挪过去之后
// MTE2_V/ID0 正处于 pending 状态 —— 两个不同 pipe 对用同一个事件 ID 并发存在。
// 如果硬件事件 ID 是全局共享的，就会把 pending 的 MTE2_V flag 冲掉，导致
// Process 里那个 WaitFlag 永远等不到，**内核直接挂死**。
// 本地没有 NPU 验不了，所以宁可不要这 0.3μs。想恢复必须先确认事件 ID 是否按 pipe 对隔离。
// ════════════════════════════════════════════════════════════════════════════
//
//
// ════════════════════════════════════════════════════════════════════════════
// ── v16.1（2026-09-30）：上机数据的三个反常，以及由此做的两处改动 ────────────
// v16（建表向量化）上机 12/12 Pass、输出误差 0.00%，但**小档一点没动**
// （4.13 → 4.16μs）。原因当场就清楚了：**小档路径根本没被执行**。
//   `tileRows = ceil(N / 40)`，而小档分支的判据是 `tileRows <= 48` ⇒ 只有 N ≤ 1920
//   才会走 v9 行车道；判分点的 N 都是 4096 起 ⇒ 全部走列平面路径。v16 的改动因此
//   是一段死代码 —— 这也是它「零收益但零风险」的原因。
//
// 真正值得看的是这一版上机数据的形状（12 点，单位 μs）：
//   4.16 / 4.41 / 7.75 / 7.71 / 25.12 / 25.05   (fp32)
//   4.16 / 5.10 / 7.84 / 7.84 / 24.31 / 24.31   (fp16)
// 三个反常：
//   ① **B 几乎不影响用时**：同一档内两个点最大只差 6%（4.16 vs 4.41），大档只差 0.3%。
//      如果瓶颈在 item 循环里，B 一放大必然成倍变化 —— 所以瓶颈**不在 item 循环**，
//      item 循环里的向量量与搬用量都不是决定项。
//   ② **dtype 几乎不影响用时**：fp32 / fp16 逐点差 < 3% ⇒ Cast 与 Sigmoid 也不是决定项。
//   ③ **每行成本在大档突然翻倍**：小/中/大档按「每行 ns」算约为 40 / 19 / 31，中间的
//      中档最便宜 —— 说明存在一个只在大档出现的阶跃。
//
// 由 ①②③ 得到的推论：**决定用时的不是「算了多少」，而是「每个核上串行跑了多长」**。
// 大档恰好把这一点放到最大：rowsPerCore ≈ 824 被 FIRE_MAX_TILE_ROWS = 832 **并成 1 个
// tile** ⇒ B = 1 时全核只有 1 个 item，`issue(i+1) → compute(i) → store(i-1)` 这条流水线上
// 没有任何东西可以重叠，建表 + 展布 + 搬运延迟 + 计算 + 写回整段串行。
// 「DB 未生效」的官方判定标准这次不是没开双缓冲，而是**没有第二个 item 可用**。
//
// v16.1 因此做了两件事（都不改数值口径）：
//   1. tiling.h 里 FIRE_MAX_TILE_ROWS 832 → 416：大档回到「2 个 tile」，既有 2 个 item
//      可供流水重叠，UB 占用也从 ~187KB(91%) 回到 ~93KB。中/小档（410 / 103 行）不受影响。
//   2. item 循环里的整数除法全部拆掉（标量单元没有硬件除法，一条 `i / batch_` 是十几次
//      软除）：tile / b / rows 改成「跑表」推进，只在 tile 边界重算一次；
//      `IssueItemLoad / IssueItemStore` 改成直接收 (tile, b, rows)，内部一次除法不做；
//      Init 里的 B 改成直接取 Host 写好的 `tiling.batchSize`（Host 侧本来就有这个字段）。
//      这一项对 B=8 的档位是实打实的（同一档内 B 越大、item 越多，省下的软除越多）。
// ════════════════════════════════════════════════════════════════════════════
//
//
// ════════════════════════════════════════════════════════════════════════════
// ── v16 改了什么（只动小档路径的建表，计算一行没碰）────────────────────────
//
// v15 是「列平面（大档）+ v9 行车道（小档）」两条路径并存。v14 已经把**列平面**的
// 建表从 16·tileRows 次标量 SetValue 换成 Adds 自加翻倍，但小档路径是从 v9 逐字搬
// 过来的，**建表仍然是 14·tileRows 次标量写**，而这条路正是 4 个已拿 TBest 的小档
// 用例在走的。
//
// 这条成本有多大，v8→v9 已经给过一次硬标定：
//   v8 按 MAX_ROWS(224) 建表 → 小档 30.89μs；v9 改成按 tileRows(25) 建表 → 4.13μs。
//   两者相差 (14×224 − 14×25) = 2786 次标量写，却差了 26.76μs ⇒ 每次 SetValue 约
//   9.6ns（≈14 拍）。照这个单价，rows = 25 / 44 的建表分别是 3.4μs / 5.9μs ——
//   而这两个点的实测总时间只有 4.13μs / 4.42μs。也就是说：
//   **小档的绝大部分时间花在「把索引表写出来」上，不在算子上。**
//
// v16 的做法与 v14 同源，只是要处理小档特有的「行内 6 车道」交错结构：
//   6 张表都满足 表(k) = 相位[k mod P] + step·(k / P)，P 是行内周期（行车道 6、
//   压紧列 5），step 是行距（20B / 24B）。于是只要写出 P 与 8 的公倍数那么长的种子
//   （P=6 → 24 个元素 = 4 行；P=5 → 40 个元素 = 8 行），再用
//       Adds(t[len], t[0], step·(len/P), m)
//   一路翻倍，就能把整张表铺满 —— 这条恒等式的成立条件（P | len）在翻倍过程中始终
//   满足，因为 len 从公倍数起步、每次加倍。
//
//   标量写因此从 14·tileRows（350 / 672 次）降到**固定 76 次**：
//     4 张 priors 表 × 前 4 行 × 每行 2 条真车道 = 32
//     score 表 × 前 4 行 × 每行 1 条真车道      =  4
//     压紧表 × 前 8 行 × 5 条真车道             = 40
//   其余元素（大档 rows=48 时还有 288−76 = 212 个）全部由 Adds 铺出，与 shape 无关。
//
// ── 唯一的牵连：哑车道与「零带」─────────────────────────────────────────────
// 翻倍会**同时**把哑车道的偏移推大（每行 +20B / +24B，因为它们和真车道在同一个
// 等差序列里）。v9~v15 的哑车道是一个**常量**零区偏移，靠一个 32B 零块就够了；
// v16 里它们变成「零点 + 行距·行号」，所以零区必须扩成一条**零带**：
//   PRIOR_ZERO_PAD_BYTES = MAX_ROWS·20 + 32B     （4 张 priors 表用）
//   SIG_ZERO_PAD_BYTES   = MAX_ROWS·24 + 32B     （score 表用）
// 两条零带在 ZeroTails() 里各一次 Duplicate 清零。Gather 读到的仍然全是 0，
// 所以**表内容对算子语义逐项等价**，数值口径与 v9~v15 完全一致。
//
// ── 与列平面路径的关系 ──────────────────────────────────────────────────────
// 两条路径现在共用同一套建表依据（GrowArith / GrowArithLine），只是常量不同：
// 列平面是「长度 rows 的一维列」，周期 1 或 5；小档是「周期 6 的行车道」。
// Host / tiling.h 的 ABI 一字未改，两边仍可独立重编。
// ════════════════════════════════════════════════════════════════════════════
//
// Kernel侧核函数实现（v14 分支：v10 的全部计算路径 + 「索引表用向量指令生成」）
//
// ── v13 上机实测（2026-09-27 19:2x）：全线变慢，模型被证伪 ────────────────────
//   TP1..12 = 4.65 / 5.20 / 17.25 / 17.28 / 52.31 / 52.31 / 4.74 / 5.39 /
//             17.57 / 17.59 / 54.77 / 54.78（12/12 Pass，分数约 61）
// v13 = v10 + 「一个 item 覆盖 bp_ 个 batch」（中档 itemCount 8→3）。按
//   T ≈ 3.2μs + 1.23μs·itemCount 应降到 ~7.7μs，实测 17.25μs —— **反而慢了 33%**。
// 连 bp_ = 1（逐字节同款工作）的小档也从 4.40 涨到 4.65、4.78 涨到 5.20。
// 结论：① 「每 item 1.23μs 固定开销」是**过拟合**的模型，减 item 数不加行数不行；
//       ② 中档的 1.23μs 里含「每 item 处理的元素量」成分，合并 batch 把它乘了 3 倍。
// ⇒ v13 作废，「减 item 数」这条杠杆被实测否掉，回到 v10。
//
// ── v14 的立论：把最大的一块「未动过」的成本换成向量指令 ─────────────────────
// v10 的模型（三档实测最小二乘，误差 < 3%）：
//   T ≈ 2.0μs + 0.80μs·itemCount + 2.15ns·rowsPerCore + 61.4ns·tileRows
// 末项 = BuildIndexTables 的 16·tileRows 次标量 SetValue（占三档总时间 32~35%），
// 且 Init 在 Process 之前 ⇒ 这段标量时间完全没有被 MTE2 掩盖。官方注意事项也点名了
// 这件事：「不要大量使用 SetValue 对 LocalTensor 进行赋值…若需要大批量赋值，
// 请使用指令接口。」v14 就是照这条办：三张表**每一列都是等差数列**，改成
// 「8/8/40 个标量种子 + 一串 Adds 自加翻倍」，标量写从 16·tileRows 降到固定 56 次。
// 表的**内容逐项不变** ⇒ 计算、Gather 调用序列、数值全部与 v10 等价（离线全枚举证）。
// 只用了 v10 已经在用的 API 形状（SetValue / Adds / Gather / PipeBarrier / 事件），
// 唯一的「新」是 Adds 走 int32 这条路 —— 官方 API 文档已确认 Atlas A2 支持
// int16_t / half / int32_t / float。
//
// ── 以下为 v10 的记录（口径仍然有效）────────────────────────────────────────
//
// Kernel侧核函数实现（v10 分支：向量计算由「行车道」改为「列平面」）
//
// ── v9 上机实测（2026-09-27 16:11，rank2，分数 76.82，12/12 Pass）────────────
//   TP1  4.13  TP2  4.42  TP3 15.62  TP4 15.63  TP5 49.42  TP6 49.48
//   TP7  4.10  TP8  5.12  TP9 15.88  TP10 16.02  TP11 51.39  TP12 51.47
// 对照 rank1（dihidao）：5.97 / 7.40 / 11.52 / 11.62 / 43.36 / 43.63 / …
//   **小 shape 已经反超**（TBest 的 TP1/TP2/TP7/TP8 现在是我们），
//   但中档慢 36%（15.62 vs 11.52）、大档慢 14%（49.42 vs 43.36）。
//
// ── 从这组数里读出来的两件事 ────────────────────────────────────────────────
// 1. **同一档里 B 几乎不影响用时**：TP3/TP4 = 15.62/15.63、TP5/TP6 = 49.42/49.48
//    （rank1 也是同样形态）。若瓶颈在「每个 (batch, 行) 的向量链」上，B 大一档
//    必然成倍变化 —— 所以瓶颈是**每核一次或每 priors 行一次**的东西：
//    标量建表，以及每个 tile 一次的展布（GatherPriors）。
// 2. 中档的差（4.1μs）比大档的差（6.1μs）在**按行数归一后更严重**，说明这是
//    「每行固定量」的差，与 shape 大小无关 —— 又一条指向「每行元数据」的证据。
//
// ── v10 改了什么（只动向量侧的数据布局；MTE 与流水骨架逐字未动）────────────
// v9 的向量链里**每条向量指令都作用在 6·rows 个元素上**，因为中间量是「一行 6 个
// 字段」的行布局。而这个行布局唯一的作用就是「让同一时刻的不同车道拿到不同乘数」——
// 这件事完全可以换成「先把每个字段抽成一条长度 rows 的一维向量」。抽列之后：
//     · 每条向量指令覆盖的元素数 **6·rows -> rows**
//     · 索引表 35·rows -> 16·rows（6 张行车道表 -> 3 组列表）
//     · 展布（每 tile 一次）42·rows -> 13·rows
//     · 每个 item 的 PipeBarrier<PIPE_V> 10 -> 3（只留在 Sigmoid/Gather 两侧）
// 数学逐字等价（公式与 v9 相同，已离线逐元素复核）：
//     x = σ_tx·2s + (gx−0.5)s        y = σ_ty·2s + (gy−0.5)s
//     w = σ_tw²·4·aw·s               h = σ_th²·4·ah·s        score = σ_obj·σ_fire
// 代价是向量指令的**条数**变多（每条变短）。但硬件向量指令数 = Σ count/64：
// v9 每个 item ≈ 10 条×21 repeat + 2 次 Gather(21+17) ≈ 250 条；v10 ≈ 15 条×4 repeat
// + 7 次 Gather(6×4+17) ≈ 100 条 —— 净减约 2.5 倍。
//
// ── 以下为 v9 及更早的记录（口径已过时，只作沿革参考）──────────────────────
//
// Kernel侧核函数实现（v9 分支：v8 的向量链 + 建表口径修正）
//
// ── v8 上机实测：大 shape 大涨，小 shape 全崩 ───────────────────────────────
// 榜单（2026-09-27 12:46，rank2，分数 34.15，12/12 Pass）：
//   TP1  30.89  TP2  31.62  TP3  32.38  TP4  32.37  TP5  58.33  TP6  58.48
//   TP7  30.92  TP8  31.76  TP9  34.29  TP10 34.32  TP11 59.88  TP12 59.88
// 对照 TBest（4.27 / 4.86 / 11.52 / 11.62 / 43.36 / 43.63 / 4.54 / 5.44 / …）：
// 大 shape 明显变好（fp16 大档 132.66 → 59.88μs，fp32 73.56 → 58.33μs），
// **但小 shape 从 v7 的 4.4~4.9μs 涨到 30.9μs** —— 这两点的 TBest 就是 4.27/4.86。
//
// ── 反推出来的模型（v9 的全部依据都在这组数上）──────────────────────────────
// 12 个点按 (n, n+6) 成对；由 v5 时期「按 N 切分 + 820 行/核」的标定与榜单
// 成对差值交叉校验，rowsPerCore 依次约为 25 / 44 / 72 / 72 / 824 / 824：
//   TP1 25 行 → 30.89、TP2 44 行 → 31.62、TP3 72 行 → 32.38、TP5 824 行 → 58.33
// 相邻两档做差：19 行 → 0.73μs、752 行 → 25.95μs，**两次都给出每行 ≈ 35~38ns**；
// 于是令 总时间 = 固定开销 + 38ns × rowsPerCore，四档分别反解出
//   29.9 / 29.7 / 29.6 / 27.1 μs —— 自洽到 3% 以内。
// **这个固定开销与 shape 无关这一件事，本身就指向了唯一的元凶：建表。**
// 直接证据：v8 每核要写 224 行 × 6 车道 × 5 张表 + 224 行 × 5 的压紧表，
// 而 v7 的表是按 tileRows_ 建的，小 shape 的 tileRows_ 只有 25/44 行 ——
// 两者差 26.4μs，正好等于实测的 TP1 差值（30.89 - 4.46）。
//
// ── v9 改了什么（只动 BuildIndexTables，计算一行没碰）───────────────────────
// 1. **表长回到 tileRows_**，不再按 MAX_ROWS(224) 建。这就是 v7 当年的口径：
//    表只需覆盖「本核所有 tile 里最大的那个 rows」，而 rows <= tileRows_ 恒成立。
//    小 shape 因此少写 224/25 ≈ 9 倍的标量表；大 shape（tileRows_ = 224）不受影响。
// 2. **双层循环展平**：改成「每张表一个循环 + 行内 6 个元素完全展开」。行内的值
//    要么是常量、要么是「行基址 + 常数」，于是没有分支、没有逐元素重算下标，
//    只用一个累加器推进行基址。同样 6*rows 次 SetValue，编出来的指令数少一截。
// 3. **压紧表改用两个累加器递推**，不再写 k/5 与 k%5 —— 标量单元没有硬件除法，
//    那两个软除在 5*rows 个元素上要花掉可观的时间。
// ComputeItem / 两级流水骨架 / 两处 32B 零区 / Gather 语义**逐字未动**，
// 所以 verify_v9.py（继承 v8 的逐 lane 复刻）给出的误差应当与 v8 完全相同。
//
// ── v8 的核心结论（仍然有效，保留说明）──────────────────────────────────────
//
// ── 根因：count = rows * LANES，而中间缓冲全是 float ───────────────────────
// v7 取 LANES = fp16 ? 16 : 8，于是每条向量指令覆盖 count = rows*LANES 个元素。
// 数据侧只需要 6 个字段，但 sig / acc / p3 / xmul / strideW / strideC 都是
// **LocalTensor<float>**，所以：
//   fp32：rows*8 个 float = rows*32B
//   fp16：rows*16 个 float = rows*64B   ← 同样的行数，向量元素数翻倍
//   （而且 fp16 的 16 条车道里 10 条是哑车道）
// 加上 fp16 还要两次覆盖 rows*16 的 Cast，正好把 2 倍惩罚坐实。
//
// 为什么 v7 要用 8/16？追根溯源是 v5 留下的「UB 落点 = 一行一个 32B block」。
// 那条约束在 v7 里**已经消失了**：搬运用的是
//   DataCopyExtParams{blockCount = rows/GROUP, blockLen = GROUP * rowBytes,
//                     srcStride = 0, dstStride = 0}
// 即「GROUP 行一个大 block、块首尾相接」，UB 落点本来就是一个**紧凑数组**
// （行距 = 载荷字节数，fp32 的 raw 是 24B）。32B 对齐只约束**每个块的起点**
// （官方 API 文档：DataCopyPad 的 UB 端起始地址必须 32B 对齐，blockLen 可以非对齐），
// 从不约束「每一行的起点」。而 rawExp / sig / acc 这些中间缓冲**完全不经搬运**，
// 只被向量指令读写，行距取多少是自由的。
//
// ── 本分支改了什么 ──────────────────────────────────────────────────────────
// 1. **行宽统一成 6 = RAW_FIELDS**（两种 dtype 都是 6 个 float 车道）：
//      · fp16 的向量元素数 rows*16 → rows*6（2.67 倍）
//      · fp32 的 rows*8 → rows*6（1.33 倍），且哑车道彻底消失
//      · 「展开 Gather」变成恒等映射 —— 紧凑 [rows,6] 就是最终行布局，
//        那次覆盖 rows*8 的 Gather **整个删掉**
// 2. **向量链从 14 条降到 9 条**（每个 item）：
//      · 2σ 只算一次（v7 宽高一轮与中心一轮各算一次 Muls）
//      · 把「× stride」提到乘数表里：wtabS 的 lane2/3 直接是 aw·s / ah·s，
//        g 的 lane0/1 直接是 (gx−0.5)·s —— 于是中心项与宽高项各自只剩一次乘
//      · 链式写法 t = A ⊗ (A ⊗ wtabS ⊕ strideC) ⊕ g ⊕ p，省掉独立的 4σ² 与
//        两次「先乘再加」中的一条
// 3. **每张乘数表都把 stride 直接折进去**（每个 n-tile 造一次、B 个 batch 复用）：
//      wtabS    = [0, 0, aw·s, ah·s, 0, 0]
//      strideC  = [s, s, 0, 0, 0, 0]      ← 中心项：既当 A 的倍率，又当加数
//      strideW  = [0, 0, s, s, 0, 0]      ← 宽高项：只当倍率
//      g        = [(gx−0.5)s, (gy−0.5)s, 0, 0, 0, 0]
//    strideC 与 strideW **必须分成两张表**：中心项是 `A ⊗ S ⊕ S`、宽高项是 `⊗ S`，
//    两处的 stride 落在不同的运算位置，一张表盖不住。这是 v8 唯一一处比 v7 多出来
//    的每-tile 搬运（v7 是 3 次 Gather，v8 是 4 次），但因为每行的车道数从 8/16
//    降到 6，每-tile 的 Gather **元素量其实与 v7 持平**。
//
// ── g 里的 −0.5 怎么来（走过弯路，别再改回去）──────────────────────────────
// 中心项 x = (2σ − 0.5 + gx)·s 要求展布里有一次"减 0.5"。最初的设计是让 idxG 的
// **哑车道**去取一片内容为 −0.5 的"负半区"（Duplicate(..., -0.5f, 8)），指望它
// 随 `Mul(g, g, strideC)` 的 lane≥2=0 被乘掉。这个方案是错的：它只会让 **lane2~5**
// 等于 −0.5，而真正需要减 0.5 的 lane0/1 取的是 priors 里的 gx/gy，从头到尾
// 一次也没被减到。离线复刻（verify_v8.py）当场报出 x/y 整体偏大 0.5·s ——
// 光读代码是看不出来的。
// 现在的写法：idxG 的哑车道一律指向**零区**，Gather 之后再显式
//   Adds(g, g, -0.5f, count)              // [gx−0.5, gy−0.5, −0.5, −0.5, −0.5, −0.5]
//   Mul (g, g, strideC, count)            // [s, s, 0, 0, 0, 0] 把 lane≥2 乘回 0
//   ⇒ g = [(gx−0.5)s, (gy−0.5)s, 0, 0, 0, 0]
// 一条 Adds 取代了常量向量，也不再需要"负半区"，UB 里少一个容易和零区混淆的区域。
//
// ── 每行付出的向量元素数（离线核算，这是本轮唯一的目标函数）────────────────
//   加号前是「每个 item 一遍」，斜杠项是「每个 n-tile 一遍、被 B 个 batch 摊薄」。
//                        v7                                  v8
//   fp32   14×8 + 5 + 24/B = 117 + 24/B      9×6 + 5 + 36/B = 59 + 36/B
//   fp16   16×16 + 5 + 48/B = 261 + 48/B     10×6 + 10 + 36/B = 70 + 36/B
//   B=1: fp32 141 -> 95（−33%），fp16 309 -> 106（−66%）
//   B=8: fp32 120 -> 63.5（−47%），fp16 267 -> 74.5（−72%）
// 单看 Gather 元素量（更贵的指令）：v7 是 21 + 24/B，v8 是 11 + 24/B。
//
// ── 数值等价性（与 v6/v7 逐位一致，靠离线复刻证）────────────────────────────
// 参考实现：
//   x = (2σ_tx − 0.5 + gx)·s      y = (2σ_ty − 0.5 + gy)·s
//   w = (2σ_tw)²·aw·s             h = (2σ_th)²·ah·s        score = σ_obj·σ_fire
// v8 的链：
//   A = 2σ
//   v = A ⊗ wtabS ⊕ strideC        → [s, s, 2σ_tw·aw·s, 2σ_th·ah·s, 0, 0]
//   t = A ⊗ v ⊕ g ⊕ p              → [2σ_tx·s + (gx−0.5)s, …, (2σ_tw)²·aw·s, …, score, 0]
// 与 v7 的差别只在**乘法结合顺序**（(2σ)²·aw·s 现在写成 2σ·(2σ·(aw·s))），
// 舍入差 ≤ 2ulp；离线的随机用例最大绝对误差仍在 1e-14 量级。
//
// ── Gather 的语义（据 CANN 官方 API 文档 + v7 实机验证）───────────────────
//   dst[i] = src[ srcOffset[i] / sizeof(T) ]      srcOffset 单位=字节，相对 src 基址
//   约束：dst / src / srcOffset 起始地址 32B 对齐；偏移按元素位宽对齐；不得越界。
//   本文件的索引表在 v7 的构造公式上只改了「行距」一个量（LANES → ROW_FIELDS=6），
//   v7 已用这些公式在真机上 12/12 Pass 且误差 0.00%，所以语义无需重新假设。
#include "kernel_operator.h"

#include "fire_candidate_decode_fusion_tiling.h"
#include "tiling_key_fire_candidate_decode_fusion.h"

#include <type_traits>

using namespace AscendC;

namespace {
constexpr uint32_t RAW_FIELDS = 6U;    // [tx, ty, tw, th, obj_logit, fire_logit]
constexpr uint32_t PRIOR_FIELDS = 5U;  // [gx, gy, stride, anchor_w, anchor_h]
constexpr uint32_t OUTPUT_FIELDS = 5U; // [x, y, w, h, fire_score]

// ── 行宽：两种 dtype 统一 6 个 float 车道（v8 的核心）────────────────────────
// 紧凑载荷每行正好 6 个字段，所以这 6 条车道里全是真数据，没有哑车道。
constexpr uint32_t ROW_FIELDS = RAW_FIELDS;

constexpr uint32_t UB_BLOCK_BYTES = FIRE_UB_BLOCK_BYTES; // 32
constexpr uint32_t UB_BYTES_PER_AIV = 192U * 1024U;      // Ascend910B1：每个 AIV 独享 192KB
constexpr uint32_t UB_RESERVE_BYTES = 8U * 1024U;        // 832 行时仍有额外安全余量

// v10 不再有「行车道」这个概念：向量计算全部在「长度 rows 的一维列」上进行，
// 字段落在哪一行由索引表决定（见 BuildIndexTables）。唯一要钉死的是**字段顺序**必须
// 与 GM 的 [tx, ty, tw, th, obj_logit, fire_logit] 一致 —— 抽列用的第 j 张表就是按
// 这个顺序取第 j 列的。
} // namespace

template <typename T> class KernelFireCandidateDecodeFusionCol {
  public:
    static constexpr bool IS_FP16 = std::is_same_v<T, half>;

    // 一律显式收敛成 uint32_t：DataCopyExtParams 是花括号初始化，一旦表达式的类型
    // 被 sizeof 提升成 size_t 且混入运行期项，就不再是常量表达式，会直接触发
    // -Wc++11-narrowing 拒绝编译（v5 踩过，见 4files 版本的注释）。
    static constexpr uint32_t ELEM_BYTES = static_cast<uint32_t>(sizeof(T));      // 4 / 2
    static constexpr uint32_t FLOAT_BYTES = static_cast<uint32_t>(sizeof(float)); // 4
    static constexpr uint32_t RAW_ROW_BYTES = RAW_FIELDS * ELEM_BYTES;            // 24 / 12
    static constexpr uint32_t PRIOR_ROW_BYTES = PRIOR_FIELDS * FLOAT_BYTES;       // 20
    static constexpr uint32_t OUT_ROW_BYTES = OUTPUT_FIELDS * ELEM_BYTES;         // 20 / 10

    static constexpr uint32_t GROUP = FIRE_PACK_GROUP; // 64 行 / MTE block
    // 两种 dtype 共用同一个 tile 档位：行宽压到 6 之后 fp16 的 float 侧缓冲不再
    // 比 fp32 大一倍，UB 预算减半（见 tiling.h 的核算），所以 fp16 也能吃下 224 行。
    static constexpr uint32_t MAX_ROWS = FIRE_MAX_TILE_ROWS;
    static constexpr uint32_t MAX_FLOATS = MAX_ROWS * ROW_FIELDS; // float 侧元素数
    static constexpr uint32_t MAX_OUT = MAX_ROWS * OUTPUT_FIELDS;

    static constexpr uint32_t AlignUp32(uint32_t x) {
        return (x + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
    }

    // ── UB 布局：两组流水 slot + 1 个行布局缓冲 + 11 个列平面 + 3 组索引区 ──
    // raw slot 的结构与 v9 逐字相同（raw 紧凑区 + priors 紧凑区），priors 继续复用
    // raw 的 READY / RAW_FREE 事件。多留的那个 32B 尾块仍给 DataCopyPad 的尾块补齐；
    // v10 不再需要「零区」—— 所有索引都指向真实数据（见 BuildIndexTables）。
    static constexpr uint32_t RAW_REGION_BYTES = MAX_ROWS * RAW_ROW_BYTES + UB_BLOCK_BYTES;
    static constexpr uint32_t PRIOR_REGION_BYTES = MAX_ROWS * PRIOR_ROW_BYTES + 2U * UB_BLOCK_BYTES;
    static constexpr uint32_t RAW_SLOT_BYTES = AlignUp32(RAW_REGION_BYTES) + AlignUp32(PRIOR_REGION_BYTES);
    static constexpr uint32_t OUT_SLOT_BYTES = AlignUp32(MAX_ROWS * OUT_ROW_BYTES + UB_BLOCK_BYTES);
    static constexpr uint32_t PRIOR_OFFSET_IN_SLOT = AlignUp32(RAW_REGION_BYTES);

    // sig 是唯一保留「行布局」的浮点缓冲：Sigmoid 一次作用在一行的 6 个字段上。
    static constexpr uint32_t VEC_BYTES = MAX_FLOATS * FLOAT_BYTES;
    static constexpr uint32_t SIG_BYTES = VEC_BYTES + UB_BLOCK_BYTES;

    // ── 列平面（v10 的核心）──────────────────────────────────────────────────
    // 每个平面 = 一段长度 MAX_ROWS 的连续 float 数组，装「某一列」的值。
    // 把一行 6 个字段按列抽成 6 条长度 rows 的一维数组之后，逐元素乘法不再需要
    // 「同一时刻按车道分派不同乘数」这种只有行布局才需要的手法：中间量全部退化成
    // 等长的一维数组，**每条向量指令覆盖的元素数是 v9 的 1/6**，标量表也只需按列建。
    static constexpr uint32_t PLANE_BYTES = MAX_ROWS * FLOAT_BYTES; // 896B，32B 的整数倍
    // 5 个跨 batch 常驻因子 + 6 个 decode 列。展布阶段的 s/g 临时值直接复用
    // 尚未开始承载结果的 TX/TY 平面，避免为仅在 Init-of-tile 存活的值独占 UB。
    static constexpr uint32_t PLANE_COUNT = 11U;
    static constexpr uint32_t PLANE_STRIDE = MAX_ROWS; // 以 float 元素为单位的平面间距

    // 平面编号。顺序有硬约束：TX/TY/TW/TH/SCORE 必须连续（压紧表按这个顺序取 5 列）。
    static constexpr uint32_t PL_S2 = 0U;     // 2s
    static constexpr uint32_t PL_AWS4 = 1U;   // 4·aw·s
    static constexpr uint32_t PL_AHS4 = 2U;   // 4·ah·s
    static constexpr uint32_t PL_GXS = 3U;    // (gx−0.5)·s
    static constexpr uint32_t PL_GYS = 4U;    // (gy−0.5)·s
    static constexpr uint32_t PL_TX = 5U;     // 展布时复用为 s；计算时 σ_tx → x
    static constexpr uint32_t PL_TY = 6U;     // 展布时复用为 g；计算时 σ_ty → y
    static constexpr uint32_t PL_TW = 7U;     // σ_tw → 结果 w
    static constexpr uint32_t PL_TH = 8U;     // σ_th → 结果 h
    static constexpr uint32_t PL_SCORE = 9U;  // σ_obj → 结果 fire_score
    static constexpr uint32_t PL_TFIRE = 10U; // σ_fire（算完 score 后当平方的临时落点）

    static constexpr uint32_t RAW_SLOT0_OFFSET = 0U;
    static constexpr uint32_t RAW_SLOT1_OFFSET = RAW_SLOT0_OFFSET + RAW_SLOT_BYTES;
    static constexpr uint32_t OUT_SLOT0_OFFSET = RAW_SLOT1_OFFSET + RAW_SLOT_BYTES;
    static constexpr uint32_t OUT_SLOT1_OFFSET = OUT_SLOT0_OFFSET + OUT_SLOT_BYTES;
    static constexpr uint32_t SIG_OFFSET = OUT_SLOT1_OFFSET + OUT_SLOT_BYTES;
    // fp16 专用：raw 的 half → float 落点。fp32 模板实例把这两块大小编译成 0，
    // 不再为永远走不到的 Cast 分支浪费 UB。
    static constexpr uint32_t RAWF32_BYTES = IS_FP16 ? AlignUp32(VEC_BYTES) : 0U;
    static constexpr uint32_t OUTF32_BYTES = IS_FP16 ? AlignUp32(MAX_OUT * FLOAT_BYTES) : 0U;
    static constexpr uint32_t RAWF32_OFFSET = SIG_OFFSET + AlignUp32(SIG_BYTES);
    // fp16 专用：压紧后的 float 中转，再 Cast 成 half 落进 out slot
    static constexpr uint32_t OUTF32_OFFSET = RAWF32_OFFSET + RAWF32_BYTES;
    static constexpr uint32_t PLANE_BASE_OFFSET = OUTF32_OFFSET + OUTF32_BYTES;

    // 15 个平面的字节偏移（显式枚举，不依赖 constexpr 函数调用）
    static constexpr uint32_t PS2_OFFSET = PLANE_BASE_OFFSET + PL_S2 * PLANE_BYTES;
    static constexpr uint32_t PAWS4_OFFSET = PLANE_BASE_OFFSET + PL_AWS4 * PLANE_BYTES;
    static constexpr uint32_t PAHS4_OFFSET = PLANE_BASE_OFFSET + PL_AHS4 * PLANE_BYTES;
    static constexpr uint32_t PGXS_OFFSET = PLANE_BASE_OFFSET + PL_GXS * PLANE_BYTES;
    static constexpr uint32_t PGYS_OFFSET = PLANE_BASE_OFFSET + PL_GYS * PLANE_BYTES;
    static constexpr uint32_t PTX_OFFSET = PLANE_BASE_OFFSET + PL_TX * PLANE_BYTES;
    static constexpr uint32_t PTY_OFFSET = PLANE_BASE_OFFSET + PL_TY * PLANE_BYTES;
    static constexpr uint32_t PTW_OFFSET = PLANE_BASE_OFFSET + PL_TW * PLANE_BYTES;
    static constexpr uint32_t PTH_OFFSET = PLANE_BASE_OFFSET + PL_TH * PLANE_BYTES;
    static constexpr uint32_t PSCORE_OFFSET = PLANE_BASE_OFFSET + PL_SCORE * PLANE_BYTES;
    static constexpr uint32_t PTFIRE_OFFSET = PLANE_BASE_OFFSET + PL_TFIRE * PLANE_BYTES;
    // 压紧表的源基址：x/y/w/h/score 这 5 个平面连续，间距就是 PLANE_STRIDE
    static constexpr uint32_t PACK_SRC_OFFSET = PTX_OFFSET;

    // ── 三组列索引表（全部按「列」建，不按行）────────────────────────────────
    // 每张表是一段长度 MAX_ROWS 的 uint32（字节偏移），列与列之间以 MAX_ROWS 为界。
    //   idxPCol[j][i] : priors 紧凑区的第 j 列、第 i 行 -> 20i + 4j
    //   idxRCol[j][i] : sig  行布局的第 j 列、第 i 行 -> 24i + 4j
    //   idxOut[5i+j]  : 压紧 -> plane[j][i]        -> 4·(j·PLANE_STRIDE + i)
    static constexpr uint32_t TABLE_BYTES = AlignUp32(MAX_ROWS * 4U);
    static constexpr uint32_t IDX_PCOL_OFFSET = PLANE_BASE_OFFSET + PLANE_COUNT * PLANE_BYTES;
    // 同一紧凑矩阵的各列行距完全相同：Gather 时移动 src 基址即可复用第 0 列表。
    // 因而 priors/raw 各只保留一张行偏移表，避免 9 张内容仅差常量的冗余表。
    static constexpr uint32_t IDX_RCOL_OFFSET = IDX_PCOL_OFFSET + TABLE_BYTES;
    static constexpr uint32_t IDX_OUT_OFFSET = IDX_RCOL_OFFSET + TABLE_BYTES;
    static constexpr uint32_t WORKSPACE_BYTES = IDX_OUT_OFFSET + OUTPUT_FIELDS * TABLE_BYTES;

    // ── 索引表的生成口径（v14）────────────────────────────────────────────────
    // 三张表的每一列都是等差数列，所以只写「种子」用标量，其余全部由 Adds 翻倍推出。
    //   等差列（idxPCol / idxRCol）：种子 8 个。8 个 int32 = 32B，之后每一步的落点
    //     都是 2 的幂 × 8 个元素，天然满足向量指令 32B 起始对齐。
    //   5 周期列（idxOut）：翻倍公式要求落点必须是 5 的倍数（见 GrowArith 的推导），
    //     与 8 的倍数求交集 ⇒ 落点必须是 40 的倍数 ⇒ 种子长度取 40。
    static constexpr uint32_t SEED_ARITH = 8U;
    static constexpr uint32_t SEED_OUT = 8U * OUTPUT_FIELDS; // 40

    static_assert(ROW_FIELDS == RAW_FIELDS, "行宽必须等于载荷字段数，否则紧凑布局不对齐");
    static_assert(SEED_ARITH % 8U == 0U, "等差列种子长度必须是 8 的倍数（32B 起始对齐）");
    static_assert(SEED_OUT % (8U * OUTPUT_FIELDS) == 0U, "压紧种子长度必须是 40 的倍数");
    static_assert(TABLE_BYTES % UB_BLOCK_BYTES == 0U && PLANE_BYTES % UB_BLOCK_BYTES == 0U,
                  "表列间距与平面间距必须 32B 对齐，否则 Adds 的列落点地址不合法");
    static_assert(IDX_PCOL_OFFSET % UB_BLOCK_BYTES == 0U && IDX_RCOL_OFFSET % UB_BLOCK_BYTES == 0U &&
                      IDX_OUT_OFFSET % UB_BLOCK_BYTES == 0U,
                  "三张表的基址必须 32B 对齐（Adds 的两个操作数都从基址起算）");
    static_assert((GROUP * RAW_ROW_BYTES) % UB_BLOCK_BYTES == 0U,
                  "raw 的分组搬运必须铺满整数个 32B block，否则 UB 落点不紧凑");
    static_assert((GROUP * PRIOR_ROW_BYTES) % UB_BLOCK_BYTES == 0U, "priors 的分组搬运必须铺满整数个 32B block");
    static_assert((GROUP * OUT_ROW_BYTES) % UB_BLOCK_BYTES == 0U, "输出的分组搬运必须铺满整数个 32B block");
    static_assert(PLANE_BYTES % UB_BLOCK_BYTES == 0U, "平面间距必须 32B 对齐");
    static_assert(RAW_SLOT_BYTES % UB_BLOCK_BYTES == 0U && OUT_SLOT_BYTES % UB_BLOCK_BYTES == 0U,
                  "流水 slot 必须 32B 对齐");
    static_assert(VEC_BYTES % UB_BLOCK_BYTES == 0U, "向量缓冲长度必须 32B 对齐");
    static_assert(PL_TX + 4U == PL_SCORE, "压紧表假定 x/y/w/h 四个平面与 score 平面连续");
    static_assert(PLANE_STRIDE == MAX_ROWS, "压紧表的平面间距必须是 MAX_ROWS 个 float");
    static_assert(WORKSPACE_BYTES + UB_RESERVE_BYTES <= UB_BYTES_PER_AIV, "UB layout exceeds one AIV's UB");

    // ── 流水同步事件：两个 slot 各占一整套，共 8 对（0..7）──────────────────────
    //   READY[s]     MTE2_V : slot s 的 raw（以及新 n-tile 的 priors）已到 UB
    //   RAW_FREE[s]  V_MTE2 : slot s 的载荷已被向量读完，MTE2 可以覆写
    //   OUT_READY[s] V_MTE3 : slot s 的输出已写好，MTE3 可以搬出
    //   OUT_FREE[s]  MTE3_V : slot s 的输出已搬完，向量可以覆写
    // 不用事件 ID 做算术：显式枚举，避免 event_t 底层类型的任何不确定性。
    __aicore__ inline static event_t EvReady(uint32_t slot) {
        return (slot != 0U) ? EVENT_ID1 : EVENT_ID0;
    }
    __aicore__ inline static event_t EvRawFree(uint32_t slot) {
        return (slot != 0U) ? EVENT_ID3 : EVENT_ID2;
    }
    __aicore__ inline static event_t EvOutReady(uint32_t slot) {
        return (slot != 0U) ? EVENT_ID5 : EVENT_ID4;
    }
    __aicore__ inline static event_t EvOutFree(uint32_t slot) {
        return (slot != 0U) ? EVENT_ID7 : EVENT_ID6;
    }

    // ── 流水 slot 的寻址与 tile 行数 ─────────────────────────────────────────
    template <typename U> __aicore__ inline LocalTensor<U> SlotAt(uint32_t slot, uint32_t byteOffsetInSlot) {
        return workspaceBuf_.Get<uint8_t>()[RAW_SLOT_BYTES * slot + byteOffsetInSlot].template ReinterpretCast<U>();
    }
    template <typename U> __aicore__ inline LocalTensor<U> OutSlotAt(uint32_t slot, uint32_t byteOffsetInSlot) {
        return workspaceBuf_.Get<uint8_t>()[OUT_SLOT0_OFFSET + OUT_SLOT_BYTES * slot + byteOffsetInSlot]
            .template ReinterpretCast<U>();
    }
    __aicore__ inline uint32_t TileRows(uint32_t tile, uint32_t coreRows) const {
        const uint32_t left = coreRows - tile * tileRows_;
        return (left < tileRows_) ? left : tileRows_;
    }

    __aicore__ inline KernelFireCandidateDecodeFusionCol() {
    }

    __aicore__ inline void Init(GM_ADDR rawPred, GM_ADDR priors, GM_ADDR decoded,
                                const FireCandidateDecodeFusionTilingData &tiling, TPipe *pipe) {
        pipe_ = pipe;
        numCandidates_ = tiling.numCandidates;
        totalRows_ = tiling.totalRows;
        rowsPerCore_ = tiling.rowsPerCore;
        tileRows_ = tiling.tileRows;
        if (numCandidates_ == 0U) {
            numCandidates_ = 1U;
        }
        if (rowsPerCore_ == 0U) {
            rowsPerCore_ = FIRE_MIN_ROWS_PER_CORE;
        }
        if (tileRows_ == 0U || tileRows_ > MAX_ROWS) {
            tileRows_ = MAX_ROWS;
        }
        // v16.1：B 直接取 Host 已经算好并写进 tiling 的 batchSize —— 原来这里用
        // totalRows_ / numCandidates_ 现算，既是一次没必要的软除，也和 Host 的口径
        // 存在「两边各算一遍」的风险（Host 侧 raw_pred 是 [B,N,6]、priors 是 [N,5]，
        // 已经校验过 N 对齐）。只在 tiling 异常（为 0）时才回退到除法。
        batch_ = tiling.batchSize;
        if (batch_ == 0U) {
            batch_ = totalRows_ / numCandidates_;
        }
        if (batch_ == 0U) {
            batch_ = 1U;
        }

        rawPredGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(rawPred),
                                   static_cast<uint64_t>(totalRows_) * RAW_FIELDS);
        priorsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(priors),
                                  static_cast<uint64_t>(numCandidates_) * PRIOR_FIELDS);
        decodedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(decoded),
                                   static_cast<uint64_t>(totalRows_) * OUTPUT_FIELDS);

        pipe_->InitBuffer(workspaceBuf_, WORKSPACE_BYTES);
        // v20：索引表在 Init 里建。此刻一个 flag 都还没有 set，所以 BuildIndexTables 内部的
        // S_V 同步用 ID0 或 ID1 都安全 —— v17 担心的「与 pending 的 MTE2_V/ID0 冲突」
        // 只在建表被挪到 Process（第一笔 DMA 已发出之后）时才成立。
        BuildIndexTables();
    }

    // ── 主循环：软件流水（与 v7 完全一致，本分支未动）────────────────────────────
    // 把「一个 (n-tile, batch) 的工作」看成流水线上的一个 item，按
    //     issue(load i+1) → compute(i) → store(i-1)
    // 发指令，让 MTE2 / Vector / MTE3 三个硬件队列真正并行。
    //
    // v1~v6 的结构是 `load → WaitFlag(MTE2_V) → compute → WaitFlag(MTE3_*) → 下一轮`，
    // 于是每次搬运的**延迟**都整段暴露在向量单元面前。CANN 官方的时序图判定标准里，
    // 这正是「DB 未生效：MTE2 全在前、VECTOR 全在后」这一条，对策就是
    // `CopyIn(i+1); Compute(i); CopyOut(i-1)` 的预取循环 + 双缓冲。
    //
    // 切分口径不变：仍按 priors 行数 N 切（不是按 B*N），同一份 priors 被 B 个 batch 共享。
    __aicore__ inline void ProcessItem(uint32_t i, uint32_t s, uint32_t tile, uint32_t b, uint32_t rows,
                                       uint32_t itemCount, uint32_t coreStart, uint32_t coreRows) {
        WaitFlag<HardEvent::MTE2_V>(EvReady(s));
        if (b == 0U) {
            GatherPriors(rows, SlotAt<float>(s, PRIOR_OFFSET_IN_SLOT));
        }

        if (i + 1U < itemCount) {
            if (i + 1U >= 2U) {
                WaitFlag<HardEvent::V_MTE2>(EvRawFree(1U - s));
            }
            const uint32_t nextB = (curB + 1U == batch_) ? 0U : (curB + 1U);
            const uint32_t nextTile = (nextB == 0U) ? (tile + 1U) : tile;
            const uint32_t nextRows = (nextTile == tile) ? rows : TileRows(nextTile, coreRows);
            IssueItemLoad(nextTile, nextB, nextRows, coreStart);
            SetFlag<HardEvent::MTE2_V>(EvReady(1U - s));
        }

        ComputeItem(s, rows, i >= 2U, i + 2U < itemCount);

        if (i >= 1U) {
            const uint32_t ps = 1U - s;
            WaitFlag<HardEvent::V_MTE3>(EvOutReady(ps));
            const uint32_t prevB = (b == 0U) ? (batch_ - 1U) : (b - 1U);
            const uint32_t prevTile = (b == 0U) ? (tile - 1U) : tile;
            const uint32_t prevRows = (prevTile == tile) ? rows : TileRows(prevTile, coreRows);
            IssueItemStore(prevTile, prevB, prevRows, ps, coreStart);
            if (i + 1U < itemCount) {
                SetFlag<HardEvent::MTE3_V>(EvOutFree(ps));
            }
        }
    }

    __aicore__ inline void Process() {
        const uint32_t coreStart = GetBlockIdx() * rowsPerCore_;
        if (coreStart >= numCandidates_) {
            return; // 为压掉 blockDim 过小带来的慢启动路径而多开出来的空 block
        }
        uint32_t coreRows = numCandidates_ - coreStart;
        if (coreRows > rowsPerCore_) {
            coreRows = rowsPerCore_;
        }
        const uint32_t tileCount = coreRows / tileRows_ + ((coreRows % tileRows_) != 0U ? 1U : 0U);
        const uint32_t itemCount = tileCount * batch_;
        if (itemCount == 0U) {
            return;
        }

        // v16.1：tile / b / rows 改成「跑表」推进 —— 标量单元没有硬件除法，原写法每个
        // item 要在 Process / IssueItemLoad / IssueItemStore 里各做一次 i / batch_（三次
        // 软除 + 一次取模），现在只在 tile 边界上算一次。
        uint32_t curTile = 0U;
        uint32_t curB = 0U;
        uint32_t curRows = TileRows(0U, coreRows);

        // 序：先把第 0 个 item 的搬运发出去（此刻向量单元还闲着）
        IssueItemLoad(0U, 0U, curRows, coreStart);
        SetFlag<HardEvent::MTE2_V>(EvReady(0U));

        // v20：建表**不在**这里了，已移回 Init（见 Init 里紧随 InitBuffer 的那一句）。
        //
        // v17 曾把它挪到此处、想与第一笔 DMA 重叠，理由是「建表与搬运从串行变并行」。
        // 实测证否：相对 v16 全线变慢 3.9%（12/12 一致，非噪声）。真正的原因是循环内
        // 的顺序 —— 「GatherPriors → WaitFlag(V_MTE2, EvRawFree) → 发预取」里，
        // GatherPriors 的向量执行时间本来就在"顺便"覆盖 rawFree 的等待；
        // 一旦把建表和展布都后移，这个等待就裸露出来，而等待期间 MTE2 因为预取还没
        // 发出而空转。净效果为负，所以回退。

        for (uint32_t i = 0U; i < itemCount; ++i) {
            const uint32_t s = i & 1U;
            ProcessItem(i, s, curTile, curB, curRows, itemCount, coreStart, coreRows);
            if (++curB == batch_) {
                curB = 0U;
                ++curTile;
                if (curTile < tileCount) {
                    curRows = TileRows(curTile, coreRows);
                }
            }
        }

        // 收尾：把最后一个 item 的结果搬出去（最后一个 item 的 tile / b / rows 可直接推得）
        {
            const uint32_t last = itemCount - 1U;
            const uint32_t ls = last & 1U;
            const uint32_t lastTile = tileCount - 1U;
            const uint32_t lastB = batch_ - 1U;
            WaitFlag<HardEvent::V_MTE3>(EvOutReady(ls));
            IssueItemStore(lastTile, lastB, TileRows(lastTile, coreRows), ls, coreStart);
        }
        // v17：最后一次写回只压在 MTE3 上 —— 向量侧的结果在发这次 store 之前就已用
        // V_MTE3 事件对齐过，MTE2 的预取也都已经被消费，所以只需排空 MTE3，
        // 不必再等整条 PIPE_ALL（那会把已经没有在飞任务的队列也一起排一遍）。
        PipeBarrier<PIPE_MTE3>();
    }

    // item 的搬运：raw 一行 24B/12B、priors 一行 20B，都按 GROUP 行一个大 block 发。
    // 两类搬运同属 MTE2 且顺序发出，因此一个 READY 事件就能覆盖两者。
    // v16.1：tile / b / rows 由调用方的跑表传入，这里一次除法都不做。
    __aicore__ inline void IssueItemLoad(uint32_t tile, uint32_t b, uint32_t rows, uint32_t coreStart) {
        // slot 必须与 Process 里的一致（item = tile·B + b）
        const uint32_t s = (tile * batch_ + b) & 1U;
        const uint32_t priorRow = coreStart + tile * tileRows_;
        const uint32_t globalRow = b * numCandidates_ + priorRow;
        IssueBlockLoad(SlotAt<T>(s, 0U), rawPredGm_[static_cast<uint64_t>(globalRow) * RAW_FIELDS], rows,
                       RAW_ROW_BYTES);
        // priors 只在新 n-tile 的第一个 batch 搬；同一 tile 的其余 batch 复用上一份。
        if (b == 0U) {
            IssueBlockLoad(SlotAt<float>(s, PRIOR_OFFSET_IN_SLOT),
                           priorsGm_[static_cast<uint64_t>(priorRow) * PRIOR_FIELDS], rows, PRIOR_ROW_BYTES);
        }
    }

    __aicore__ inline void IssueItemStore(uint32_t tile, uint32_t b, uint32_t rows, uint32_t slot, uint32_t coreStart) {
        const uint32_t globalRow = b * numCandidates_ + coreStart + tile * tileRows_;
        IssueBlockStore(decodedGm_[static_cast<uint64_t>(globalRow) * OUTPUT_FIELDS], OutSlotAt<T>(slot, 0U), rows,
                        OUT_ROW_BYTES);
    }

  private:
    // 纯地址生成：基址 + 编译期字节偏移 + 类型转换。用 uint8_t 取址可以精确到字节。
    template <typename U, uint32_t BYTE_OFFSET> __aicore__ inline LocalTensor<U> WorkspaceAt() {
        return workspaceBuf_.Get<uint8_t>()[BYTE_OFFSET].template ReinterpretCast<U>();
    }

    // ── 索引表：每个核只建一次，之后所有 tile / batch 复用 ──────────────────────
    // **表长按 tileRows_ —— 不是 N，也不是 MAX_ROWS。** v8 曾按 MAX_ROWS 建，小 shape
    // 因此白跑 224 行的标量表（实测把 TP1 从 4.46μs 抬到 30.89μs），v9 修回 tileRows_，
    // 这个口径在 v10 / v14 保持不变。
    //
    // 表的内容与 v10 **逐项相同**（这就是「其余 shape 不受影响」的根据，
    // 见 verify_v14_tables.py，对 rows = 1..MAX_ROWS 全枚举复核）：
    //   idxPCol[j][i] = 20i + 4j     idxRCol[j][i] = 24i + 4j     idxOut[5i+j] = 4i + 896j
    //
    // ── v14 改的只是「怎么把表写出来」──────────────────────────────────────────
    // v10 用 16·rows 次标量 SetValue，而官方注意事项的原话是：
    //   「不要大量使用 SetValue 对 LocalTensor 进行赋值，会使性能下降，
    //     若需要大批量赋值，请使用指令接口。」
    // 而按 v10 三档实测拟合出的模型里，这一项占总时间的 30% 上下：
    //   T ≈ 2.0μs + 0.80μs·itemCount + 2.15ns·rowsPerCore + 61.4ns·tileRows
    // （61.4ns × tileRows 正是 16 列 × 每列 tileRows 次标量写。）而且 Init 在 Process
    // 之前，这段标量时间完全没有被 MTE2 掩盖 —— 它是最大的一块「未动过」的成本。
    //
    // 观察：三张表的**每一列都是等差数列**，于是「少量种子 + 向量自加翻倍」就能生成：
    //     dst[len + k] = src[k] + C        （src 取同一缓冲区的前段）
    // 只要 C 选得对，这一步就是把一条等差数列**续写一倍**：
    //   纯等差（公差 d）：表[len+k] = d(len+k) = dk + d·len  ⇒  C = d·len（任意 len 成立）；
    //   idxOut 的 5 周期列（表[k] = 4·(k/5) + 896·(k%5)）：
    //     (len+k)/5 = len/5 + k/5 且 (len+k)%5 = k%5 需要 5 | len ⇒ C = 4·(len/5)。
    // 又因为向量指令的操作数起始地址必须 32B 对齐（= 8 个 int32），落点 len 还得是 8 的
    // 倍数 ⇒ idxOut 的落点必须同时是 5 和 8 的倍数（40 的倍数），种子长度也取 40；
    // 等差列的落点从 8 起、每次翻倍，天然满足。
    //
    // 净效果：标量写从 16·rows 次降到固定 56 次（与 shape 无关），其余是 ≤ 20 条长度
    // ≤ rows 的 Adds。表内容逐项不变 ⇒ 计算路径、Gather 调用、数值全都不受影响。
    __aicore__ inline void GrowArith(const LocalTensor<int32_t> t, uint32_t target, uint32_t seed, uint32_t period,
                                     int32_t step) {
        // 前置条件（调用方保证，离线脚本对 rows = 1..MAX_ROWS 全枚举复核）：
        //   t[0..seed) 已是正确的等差前缀；seed % 8 == 0（落点 32B 对齐）；seed % period == 0。
        // m 取 min(target-len, len)：dst 区 [len, len+m) 与 src 区 [0, m) 互不相交（m <= len），
        // 所以这段 SelfAdd 没有地址重叠问题。
        if (period == 0U) {
            return;
        }
        uint32_t len = seed;
        while (len < target) {
            uint32_t m = target - len;
            if (m > len) {
                m = len;
            }
            Adds(t[len], t[0], static_cast<int32_t>((len / period) * static_cast<uint32_t>(step)), m);
            len += m;
        }
    }

    __aicore__ inline void BuildIndexSeeds(const LocalTensor<int32_t> &tP, const LocalTensor<int32_t> &tR,
                                           const LocalTensor<int32_t> &tO, uint32_t seedP, uint32_t seedO) {
        for (uint32_t k = 0U; k < seedP; ++k) {
            tP.SetValue(k, static_cast<int32_t>(k) * static_cast<int32_t>(PRIOR_ROW_BYTES));
            tR.SetValue(k, static_cast<int32_t>(k) * static_cast<int32_t>(ROW_FIELDS * FLOAT_BYTES));
        }
        uint32_t q = 0U;
        uint32_t j = 0U;
        for (uint32_t k = 0U; k < seedO; ++k) {
            tO.SetValue(k, static_cast<int32_t>(q * FLOAT_BYTES) +
                               static_cast<int32_t>(j) * static_cast<int32_t>(PLANE_BYTES));
            if (++j == OUTPUT_FIELDS) {
                j = 0U;
                ++q;
            }
        }
    }

    __aicore__ inline void BuildIndexTables() {
        const uint32_t rows = tileRows_;
        LocalTensor<int32_t> tP = WorkspaceAt<int32_t, IDX_PCOL_OFFSET>();
        LocalTensor<int32_t> tR = WorkspaceAt<int32_t, IDX_RCOL_OFFSET>();
        LocalTensor<int32_t> tO = WorkspaceAt<int32_t, IDX_OUT_OFFSET>();

        BuildIndexSeeds(tP, tR, tO, seedP, seedO);

        // 标量写 UB -> 向量读，必须显式同步（第一条 Adds 就要读这些种子）。
        // v17：这里刻意用 ID1 而不是 ID0 —— 建表已经被挪到 Process 里、第一笔 DMA 之后，
        // 那一刻 MTE2_V/ID0 正处于 pending；若 S_V 也用 ID0，在「事件 ID 全局共享」的
        // 硬件语义下会把那个 pending flag 冲掉，Process 里的 WaitFlag 就永远等不到（挂死）。
        // ID1 在那一刻没有任何在飞的使用者，两种语义下都安全。
        SetFlag<HardEvent::S_V>(EVENT_ID1);
        WaitFlag<HardEvent::S_V>(EVENT_ID1);

        // ── 翻倍（自此处起全是向量指令）──────────────────────────────────────────
        // 表长覆盖口径与 v10 完全相同：idxPCol / idxRCol 覆盖 rows，idxOut 覆盖 5·rows。
        GrowArith(tP, rows, seedP, 1U, static_cast<int32_t>(PRIOR_ROW_BYTES));
        GrowArith(tR, rows, seedP, 1U, static_cast<int32_t>(ROW_FIELDS * FLOAT_BYTES));
        GrowArith(tO, rows * OUTPUT_FIELDS, seedO, OUTPUT_FIELDS, static_cast<int32_t>(FLOAT_BYTES));

        // Gather 是复合指令（见 GatherPriors 末尾的说明），补一条 PIPE_V 屏障再放行。
        PipeBarrier<PIPE_V>();
    }
    // ── 分组搬运：rows 行、每行 rowBytes 字节的连续 GM 段 -> 连续的 UB 紧凑段 ──
    // rows/GROUP 个「GROUP 行一块」的大事务，余数用一个单块事务收尾。
    // 两侧 stride 都为 0：GM 侧块首尾相接，UB 侧因为块长是 32B 整数倍也首尾相接。
    template <typename DT>
    __aicore__ inline void IssueBlockLoad(const LocalTensor<DT> &dst, const GlobalTensor<DT> &src, uint32_t rows,
                                          uint32_t rowBytes) {
        const uint32_t bytes = rows * rowBytes;
        const uint32_t padBytes = AlignUp32(bytes) - bytes;
        DataCopyExtParams params{1U, bytes, 0U, 0U, 0U};
        DataCopyPadExtParams<DT> pad{true, static_cast<uint8_t>(0U),
                                     static_cast<uint8_t>(padBytes / static_cast<uint32_t>(sizeof(DT))),
                                     static_cast<DT>(0)};
        DataCopyPad(dst, src, params, pad);
    }

    // 同样分组的写回：UB 紧凑段 -> 连续 GM 段
    template <typename DT>
    __aicore__ inline void IssueBlockStore(const GlobalTensor<DT> &dst, const LocalTensor<DT> &src, uint32_t rows,
                                           uint32_t rowBytes) {
        DataCopyExtParams params{1U, rows * rowBytes, 0U, 0U, 0U};
        DataCopyPad(dst, src, params);
    }

    // ── 展布：紧凑 priors -> 四张「已含 stride 的乘数表」，每个 n-tile 一次、B 个 batch 复用 ──
    //   wtabS   = [0, 0, aw, ah, 0, 0]  ⊗ [0, 0, s, s, 0, 0] = [0, 0, aw·s, ah·s, 0, 0]
    //   g       = ([gx, gy, 0, 0, 0, 0] ⊕ (−0.5)) ⊗ [s, s, 0, 0, 0, 0]
    //           = [(gx−0.5)s, (gy−0.5)s, 0, 0, 0, 0]
    //   strideC = [s, s, 0, 0, 0, 0]    ← 中心项的加数与 A 的倍率（同一张表两用）
    //   strideW = [0, 0, s, s, 0, 0]    ← 宽高项的倍率（表是分开的，见文件头说明）
    // 4 次 Gather + 3 条向量指令；四张表同源、彼此独立，一次发完再等一次。
    // `Adds(g, g, -0.5f)` 会把哑车道也变成 −0.5，但紧随其后的 `Mul(g, g, strideC)`
    // 里 strideC 的 lane≥2 是 0，所以那些 −0.5 被乘掉，不会流到累加结果里。
    // 这里的 Gather 元素量与 v7 持平（v7 是 3 次 × rows*8，这里是 4 次 × rows*6），
    // 而每个 item 的向量指令量减半 —— 净收益来自后者。
    // ── 展布（每个 n-tile 一次，B 个 batch 复用）────────────────────────────────
    // 把 priors 的 5 列抽成 5 条长度 rows 的一维向量，并顺手把常量折进去，得到本 tile
    // 全部的「每行标量因子」：
    //   S2   = 2s                    （x / y 的乘数）
    //   AWS4 = 4·aw·s   AHS4 = 4·ah·s      （w / h 的乘数）
    //   GXS  = (gx−0.5)·s   GYS = (gy−0.5)·s（x / y 的加数）
    // 代价：5 次 Gather（每次只覆盖 rows 个元素）+ 8 条长度 rows 的向量指令 = 13·rows。
    // v9 是 4 次 Gather（每次覆盖 6·rows 个元素）+ 3 条长度 6·rows 的向量指令 = 42·rows。
    __aicore__ inline void GatherPriors(uint32_t rows, const LocalTensor<float> &priorBase) {
        // s / g 仅在 priors 展布期间存活；此时 decode 结果尚未生成，复用 TX/TY 平面。
        LocalTensor<float> s = WorkspaceAt<float, PTX_OFFSET>();
        LocalTensor<float> s2 = WorkspaceAt<float, PS2_OFFSET>();
        LocalTensor<float> aws4 = WorkspaceAt<float, PAWS4_OFFSET>();
        LocalTensor<float> ahs4 = WorkspaceAt<float, PAHS4_OFFSET>();
        LocalTensor<float> gxs = WorkspaceAt<float, PGXS_OFFSET>();
        LocalTensor<float> gys = WorkspaceAt<float, PGYS_OFFSET>();
        LocalTensor<float> g = WorkspaceAt<float, PTY_OFFSET>();
        LocalTensor<uint32_t> idxP = WorkspaceAt<uint32_t, IDX_PCOL_OFFSET>();

        Gather(s, priorBase, idxP, 2U * FLOAT_BYTES, rows); // 第 2 列 stride
        Gather(g, priorBase, idxP, 0U * FLOAT_BYTES, rows); // gx
        Adds(g, g, -0.5f, rows);
        Mul(gxs, g, s, rows);
        Gather(g, priorBase, idxP, 1U * FLOAT_BYTES, rows); // gy
        Adds(g, g, -0.5f, rows);
        Mul(gys, g, s, rows);
        Muls(s2, s, 2.0f, rows);
        // s 的原值至此不再使用，原地放大成 4s；宽高各省掉一次后置 Muls。
        Muls(s, s, 4.0f, rows);
        Gather(g, priorBase, idxP, 3U * FLOAT_BYTES, rows); // 第 3 列 anchor_w
        Mul(aws4, g, s, rows);
        Gather(g, priorBase, idxP, 4U * FLOAT_BYTES, rows); // anchor_h
        Mul(ahs4, g, s, rows);
        // v21：这里原来有一条 PipeBarrier<PIPE_V>()，已删除。
        // 依据：Gather 是硬件矢量指令（vgather），与其他向量指令同处 PIPE_V、按序发射，
        // 读它结果的普通指令（前面的 Mul(gxs, g, s) 等）由硬件保证依赖顺序。
        // 另外从可见性看这条也是冗余的：乘数表最早的读点在 ComputeItem 内部，
        // 中间还隔着两道 V 屏障，本就有兜底。
    }

    // ── 单个 (n-tile, batch) 的工作：6 次抽列 + 11 条逐列运算 + 1 次压紧 ─────────
    // 数学与 v9 逐字等价（已逐 lane 复核），只是把「一行 6 个字段」换成「6 条长度 rows
    // 的一维列」，于是**每条向量指令覆盖的元素数从 6·rows 降到 rows**：
    //   x     = σ_tx·2s + (gx−0.5)s        （v9 的 A⊗v ⊕ g，A = 2σ）
    //   y     = σ_ty·2s + (gy−0.5)s
    //   w     = σ_tw²·4·aw·s               （v9 的 (2σ)²·aw·s）
    //   h     = σ_th²·4·ah·s
    //   score = σ_obj·σ_fire
    __aicore__ inline void ComputeItemMath(uint32_t slot, uint32_t rows, bool freeRaw) {
        const uint32_t sigCount = rows * ROW_FIELDS;
        const uint32_t count = rows;
        LocalTensor<T> rawCompact = SlotAt<T>(slot, 0U);
        LocalTensor<float> sig = WorkspaceAt<float, SIG_OFFSET>();
        LocalTensor<float> s2 = WorkspaceAt<float, PS2_OFFSET>();
        LocalTensor<float> aws4 = WorkspaceAt<float, PAWS4_OFFSET>();
        LocalTensor<float> ahs4 = WorkspaceAt<float, PAHS4_OFFSET>();
        LocalTensor<float> gxs = WorkspaceAt<float, PGXS_OFFSET>();
        LocalTensor<float> gys = WorkspaceAt<float, PGYS_OFFSET>();
        LocalTensor<float> tx = WorkspaceAt<float, PTX_OFFSET>();
        LocalTensor<float> ty = WorkspaceAt<float, PTY_OFFSET>();
        LocalTensor<float> tw = WorkspaceAt<float, PTW_OFFSET>();
        LocalTensor<float> thh = WorkspaceAt<float, PTH_OFFSET>();
        LocalTensor<float> score = WorkspaceAt<float, PSCORE_OFFSET>();
        LocalTensor<float> fire = WorkspaceAt<float, PTFIRE_OFFSET>();
        LocalTensor<uint32_t> idxR = WorkspaceAt<uint32_t, IDX_RCOL_OFFSET>();

        if constexpr (IS_FP16) {
            LocalTensor<float> rawF32 = WorkspaceAt<float, RAWF32_OFFSET>();
            Cast(rawF32, rawCompact, RoundMode::CAST_NONE, sigCount);
            PipeBarrier<PIPE_V>();
            Sigmoid(sig, rawF32, sigCount);
        } else {
            Sigmoid(sig, rawCompact, sigCount);
        }
        PipeBarrier<PIPE_V>();
        if (freeRaw) {
            SetFlag<HardEvent::V_MTE2>(EvRawFree(slot));
        }

        Gather(tx, sig, idxR, 0U * FLOAT_BYTES, count);
        Gather(ty, sig, idxR, 1U * FLOAT_BYTES, count);
        Gather(tw, sig, idxR, 2U * FLOAT_BYTES, count);
        Gather(thh, sig, idxR, 3U * FLOAT_BYTES, count);
        Gather(score, sig, idxR, 4U * FLOAT_BYTES, count);
        Gather(fire, sig, idxR, 5U * FLOAT_BYTES, count);

        Mul(score, score, fire, count);
        Mul(fire, tw, tw, count);
        Mul(tw, fire, aws4, count);
        Mul(fire, thh, thh, count);
        Mul(thh, fire, ahs4, count);
        FusedMulAdd(tx, s2, gxs, count);
        FusedMulAdd(ty, s2, gys, count);
    }

    __aicore__ inline void StoreItem(uint32_t slot, uint32_t rows, bool waitOutFree) {
        const uint32_t outCount = rows * OUTPUT_FIELDS;
        if (waitOutFree) {
            WaitFlag<HardEvent::MTE3_V>(EvOutFree(slot));
        }
        if constexpr (IS_FP16) {
            LocalTensor<float> outF32 = WorkspaceAt<float, OUTF32_OFFSET>();
            Gather(outF32, WorkspaceAt<float, PACK_SRC_OFFSET>(), WorkspaceAt<uint32_t, IDX_OUT_OFFSET>(), 0U,
                   outCount);
            PipeBarrier<PIPE_V>();
            Cast(OutSlotAt<T>(slot, 0U), outF32, RoundMode::CAST_ROUND, outCount);
        } else {
            Gather(OutSlotAt<T>(slot, 0U), WorkspaceAt<float, PACK_SRC_OFFSET>(),
                   WorkspaceAt<uint32_t, IDX_OUT_OFFSET>(), 0U, outCount);
        }
        SetFlag<HardEvent::V_MTE3>(EvOutReady(slot));
    }

    __aicore__ inline void ComputeItem(uint32_t slot, uint32_t rows, bool waitOutFree, bool freeRaw) {
        ComputeItemMath(slot, rows, freeRaw);
        StoreItem(slot, rows, waitOutFree);
    }

    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> workspaceBuf_;
    GlobalTensor<T> rawPredGm_;
    GlobalTensor<float> priorsGm_;
    GlobalTensor<T> decodedGm_;
    uint32_t numCandidates_ = 1U;
    uint32_t totalRows_ = 0U;
    uint32_t batch_ = 1U;
    uint32_t rowsPerCore_ = FIRE_MIN_ROWS_PER_CORE;
    uint32_t tileRows_ = FIRE_MAX_TILE_ROWS;
};

template <typename DT_RAW_PRED>
__global__ __aicore__ void fire_candidate_decode_fusion(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded,
                                                        GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FireCandidateDecodeFusionTilingData);
    GET_TILING_DATA_WITH_STRUCT(FireCandidateDecodeFusionTilingData, tilingData, tiling);
    TPipe pipe;
    KernelFireCandidateDecodeFusionCol<DT_RAW_PRED> op;
    op.Init(raw_pred, priors, decoded, tilingData, &pipe);
    op.Process();
}
