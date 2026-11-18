/*
 * Copyright (c) 2026 zhiyuan (致远队).
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

// Kernel侧核函数（W13 压平版）—— 每 dtype 一个平级类 + 运行期三档分派。
// 行为与 W11 上板组合逐位一致：TILE=256 双缓冲流水（MTE2_V/V_MTE3 定向事件）+
// fp16 SPLIT 升位算术（A）+ S53 倍增表 + 8 核 launch + riders（threshold 标量读/
// 删冗余屏障/删死 TPipe）。休眠开关（W9_DCP/W10_ARITH/TILE128/PROLOGUE_HIDE/
// TILE1024/16核）全部删除，只留现役路径；保留 W12-DIAG 四探针 + 新增 P-B 地板探针。
#include "kernel_operator.h"

#include "safety_belt_violation_geometry_tiling.h"

// W13 根因修复：本头携带 @@ASCENDC_TPL_ARGS_DECL/@LISTS 标记，opbuild 扫描器扫的是
// kernel TU 的预处理输出——缺此 include 则解码为空（TEMPLATE_PARAMS=-1/LEN 0），
// 任何 TPL 头形态都无法绑定（轮 1/轮 2 的真正根因）。
#include "tiling_key_safety_belt_violation_geometry.h"
#include "safety_belt_violation_geometry_tables.h"

// ── 诊断探针（全部默认关；大点档 WA 预期，纪律=不挂死不越界）────────────────────
#ifndef DIAG_EMPTY_CORE
#define DIAG_EMPTY_CORE 0       // 大点档各核立即退出（近似空核地板）
#endif
#ifndef DIAG_FLOOR_NOTILING
#define DIAG_FLOOR_NOTILING 0   // 入口 2 笔 tiling 标量读跳过（与 EMPTY_CORE 差分=tiling 读成本）
#endif
#ifndef DIAG_FLOOR_MINIMAL
#define DIAG_FLOOR_MINIMAL 0    // P-C：入口直返（无 tiling 读/无类实例化/无分派；与 EMPTY_CORE 差分=init 成本）
#endif
#ifndef DIAG_NO_GATHER
#define DIAG_NO_GATHER 0        // 跳过 Gather×3（保留 DMA/表构造/算术/写回）
#endif
#ifndef DIAG_NO_ARITH
#define DIAG_NO_ARITH 0         // 跳过算术段（Cast/算术/Compare/Select 全跳）
#endif
#ifndef DIAG_NO_WRITE
#define DIAG_NO_WRITE 0         // 跳过 MTE3 写回（输出 buffer 不动）
#endif

// ── W15：明日组合包（三项独立开关，默认全关；基线 = W13 树，W14c 待发配置不受影响）──
//   W15_GATHER_MERGE：大点 Gather 3→1——y1/y2/cy 合成一张 3*TILE 绝对偏移表
//     （cy 项 +B_TILE_BYTES 指进同一行缓冲区，boxes/anchors 行区本就连续），
//     单 Gather 调用链（fp16 1×6rep / fp32 1×8+1×4rep）直出 [y1,y2,cy] 相邻槽，
//     槽布局/算术/Cast/Select 零改动；表构造仅 +1 条 Adds（y2=y1+2DT）。
//     军规：offset uint32；单调用 repeat=3*tv/LANE（fp16 6 / fp32 12>8 分两段）≤255；
//     dst 槽 32B 对齐（3*TBL_BYTES=3072 是 32 倍数）。
//   W15_SMALL_CASE：小点（B≤33）单 tile 向量路径——编译期 TILE=128、无 tile 循环、
//     无 colP/双缓冲（单 tile 无 RAW/WAR）、无尾块分支（tv 上取整即现役尾块语义）。
//     形状特化（整档 B≤33 泛化，非 32/33 精确等式）；threshold/wr 仍运行时读。
//     B>33 中/大点走现役路径不变（fallback 兜底）。
//   W15_PROLOGUE：W12 PROLOGUE_HIDE 在 W13 压平树复活——首 tile 两笔行 DMA 提到
//     表构造之前发出，MTE2 飞行遮表构造（~0.05~0.10）；off==0 跳过重发。
#ifndef W15_GATHER_MERGE
#define W15_GATHER_MERGE 1
#endif
#ifndef W15_SMALL_CASE
#define W15_SMALL_CASE 1
#endif
// ── W34：Axpy 融合（通用链 Muls+Add → 硬件 vaxpy 单指令，4→3 条 V）─────────────
//   ⚠ 开关纪律（W26 事故教训）：本宏 define 与引用同 commit 落地。
//   数值语义：vaxpy 硬件单指令 dst+=a*src（若为融合乘加则 lim 单次舍入，与官方
//   参考的双舍入位模式可能漂移）——sim 424 例 HASH-SAME 为上板前置硬门。
#ifndef W34_AXPY_FUSE
#define W34_AXPY_FUSE 1
#endif
#ifndef W15_PROLOGUE
#define W15_PROLOGUE 1
#endif
//   W15_MID_PIPELINE（追加项 0，最高优先级）：中点档（33<B≤1024）改走大点同款
//     GatherVec 双缓冲流水路径——天南星对照：中点地板+0.1 vs 我方地板+0.6~0.8，
//     旧标量直通（fp32 8-Way/fp16 4-Way）是差距来源。256/257 行 ÷ 核 = 单 tile
//     （n<TILE=256，无跨 tile 循环），B 参数天然泛化；尾核（257÷8: 7×32+尾 33 或
//     per 对齐后 5×48+尾 17，任一 host 核数下均走单 tile t 截断路径）；
//     fp16 SPLIT/A 照大点。host 不动（核数由 GetBlockNum 适配）。
#ifndef W15_MID_PIPELINE
#define W15_MID_PIPELINE 1
#endif
// ── W16：中点固定成本攻击（板测 6adcd7d2：MID_PIPELINE 无变化 ⇒ 中点工作层
//   600~800ns 固定成本主导，19~25ns/行 vs 大点 1.9~2.3ns/行）────────────────────
//   考证定案（不改代码）：
//   · V→MTE3 无天然保序——dav_c220 DataCopyPadUB2GMImpl 是裸 copy_ubuf_to_gm_align
//     intrinsic（无 SetFlag/WaitFlag/Barrier），跨 PIPE_V→PIPE_MTE3 必须显式事件，
//     2 对事件不可减（减=WA）。
//   · 入口 2 笔 tiling 读合 1：需 uint32↔float 位重解释（军规#12 灰区）且两读同
//     cacheline 第 2 笔为命中（~20ns），收益/风险比不足，不实施。
//   · DMA 形态：单块连续载荷 DataCopy 与 DataCopyPad(isPad=false) 同硬件路径
//     （S33 口径 0.025μs/笔同载荷），无形态红利。
//   W16_MID_TABLE_DYN：offset 表倍增上界从 TILE 收紧到 ceil(n/LANE)*LANE——
//     中点 n=32~40 时省 2~4 条 Adds（非 GM 路径专用；GM 需满表）。
#ifndef W16_MID_TABLE_DYN
#define W16_MID_TABLE_DYN 1
#endif

// ── W21（二分实验）：表构造前置到 prologue 前 ──
#ifndef W21_TABLE_PRE
#define W21_TABLE_PRE 1
#endif
#ifndef W21_EARLY_CONST
#define W21_EARLY_CONST 1
#endif

// ── W22（隔离重测）：入口两读合一——total_b(4B)+waist(4B) 同 cacheline，一次 8B 读
//   后位拆包，省一次 GM 往返延迟（~100-150ns，全档适用）。W1/P-B 的进一步。
#ifndef W22_TILING_HIDE
#define W22_TILING_HIDE 1
#endif

// ── W24：小点输出 __gm__ 直写（S8 冠军 sohnkee 标量直写形态，授权盲发弹）────────
//   小点输出 ≤33×4B：标量循环直接写 yPtr[i]，跳过 oLocal 中转 + S_MTE3 握手 +
//   DataCopyPad 写出（省 1 笔 MTE3 全链 ~125-150ns）。逐元素写无对齐要求；泛化 ✓。
#ifndef W24_SCALAR_OUT
#define W24_SCALAR_OUT 1
#endif

// ── W25：中点紧凑消费（每核 n=32~40 行时，Gather/Cast/表全按 tv 紧凑执行）──────
//   现状：MID_PIPELINE 下中点走 GatherVec，W15 修复后恒按 3*TILE 满块消费——
//   中点每核只 32~40 行，fp32 白做 12→3 rep、fp16 白做 6→3 rep 的 Gather +
//   Cast 白做 12/24→3/6 rep（合计每 tile 多 ~30% 向量工作量）。
//   修复 = 3*TILE 满块消费仅在 B>TILE 档保留；中点档改按 3*tv 紧凑消费 + 前缀表
//   （表构造本就按 tblBound=ceil(n/LANE)*LANE 生成前缀——W16_DYN 已备）。
//   关键约束：全 32B 对齐（tv 是 LANE 倍数 ⇒ 3*tv·4B/fp16 行块仍 32B 对齐）；
//   Compare/Select count=tv≥64（fp32 64/fp16 128 均过 NOP 线）。
#ifndef W25_COMPACT_MID
#define W25_COMPACT_MID 0   // 三轮修复后 bigp fp32 仍有 180~250 错（表/别名/写出三方耦合超出搭车范畴），按 S8 纪律止损回滚；实现留档
#endif
// ── W23：Scale 式 warmup+prefetch 流水编排（52 冲刺带宽战）──────────────────────
//   参考 Ascend-Kernels-S8 Scale.cpp L500-570（warmup 填满 → 稳态
//   Wait(MTE2_V,i%N)→算→Set(V_MTE3)→Wait→写出→Set(MTE3_MTE2)→立即 prefetch→
//   Set(MTE2_V) → 尾 drain 逆序）。
//   本树差异审计（W23 前置，量化）：
//   · 大点 8 核=2 tile/核。现树 W15_PROLOGUE 只发 tile0 的行 DMA——**tile1 的
//     行 DMA 在循环内 tile0 全部落定后才发**（唯一真暴露段）。W23 把 tile1 行 DMA
//     提到 prologue（双 tile 全填满=Scale warmup），循环内 tile0 处理时 tile1
//     DMA 在飞 ⇒ 省 tile1 DMA 全延迟（~0.15~0.25，含 MTE 排队）。
//   · MTE2_V/V_MTE3 事件按 bufferIndex 分 id（0/1 轮转，Scale 同款）——单 id 下
//     tile0 的 Set(MTE2_V) 会被 tile1 的 Wait 误消费（事件语义必须分 id）。
//   · MTE3→MTE2 反向事件（Scale 的 SetScaleCycleMte3Mte2）：本树双缓冲槽位物理
//     分离（ROW1_BASE≠0），tile1 DMA 与 tile0 算术/写出无槽冲突 ⇒ 不需要。
//   · 中点单 tile（32 行/核）无流水空间，不动。
//   增量建模：大点 0.15~0.25/点（T5/T6/T11/T12），中点/小点 0。
#ifndef W23_PIPELINE
#define W23_PIPELINE 1
#endif
// ── W26：wr/th 运行期值特化（wr=1.0 链塌缩 lim≡y2+th，th=0 省 Adds）────────────
//   ⚠ 事故修复备案（10-26 晚）：本宏自 W26 首发（638da8d）起从未有 #define——
//   预处理按 0 求值，特化链（含 W30-b/b2 两处 seg 特化）四发板测全部未编译！
//   50.06/52.92 的收益实际来自 W20/W22/W23/W24+池动态，与 W26 无关。
//   本行补定义 = W26 语义首次真正生效；上板前 G1/G2 重新背书。
#ifndef W26_WR_SHORTCUT
#define W26_WR_SHORTCUT 1
#endif
// ── W17：奇数 B 罚金攻击（板测 6adce8a9：四对同向 +0.12~0.21）────────────────────
//   机理假设（量纲吻合）：奇数 B 的 DMA 块长非 32B 倍数（fp32 B=33：528/264/132B
//   全非倍；大点尾核 255 行：4080/2040/1020B 非倍）⇒ MTE 尾 burst 处理罚金
//   ~0.05/笔 × 3 笔 ≈ +0.15。输出写出**不动**（补齐=越界写=红线）。
//   W17_EVENIZE = 1（输出正确，探针/正选两用——即 PM 的 DIAG_EVENIZE）：
//   · 小点：行处理偶化到 align8(B)（纯 8-Way 组、无标量尾循环；尾组读 UB 垃圾行
//     不写出）+ 输入 DMA 长度 align32（GM 读越界 ≤31B，池内页安全，风险备案）。
//   · 中点：同式（n→align8 处理、DMA align32；写真实 n 行）。
//   · 大点：tile 行 DMA 长度 align32（fp32 255→4096=B_TILE 恰不越槽；fp16 同）。
//   解读：EVENIZE 后若奇数对回到偶数水平 ⇒ 罚金在输入 DMA/行处理；若仍差 ⇒ 残差
//   在 MTE3 写出尾（不可修，红线）。
#ifndef W17_EVENIZE
#define W17_EVENIZE 0
#endif
// ── W18：写出奇尾残差攻击（W17 板测实锤：残差在 MTE3 写出非 32B 倍数块）──────────
//   W18_WRITE_SPLIT：奇长写出拆两笔——对齐主块（floor32）+ 余数小块，两笔同 MTE3
//   队列保序、GM 区间不相交（并集=原区间，无越界）；数据源同一 UB 槽不同偏移。
//   代价 +1 MTE3 握手 ~25-50ns，收益免主块尾 burst ~50-100ns（W17 板测 T6 未回收
//   +0.13~0.23 口径）。仅对 len≥32 且非 32 倍数的写出生效；len<32 不拆（主块=0）。
//   W18_FIRST_TILE_SPLIT：判负不实施——现树 GM 合一 Gather 按 3*TILE 连续消费 +
//   fp16 宽 Cast 读 [y1|y2|cy] 连续区，首半算术所需集合非连续前缀，半启动需重排
//   表布局（半交错），侵入深、建模 -0.05~0.1 低于风险阈。
#ifndef W18_WRITE_SPLIT
#define W18_WRITE_SPLIT 0   // 板测证伪(6adcefe2): 四奇点零回收+握手净负, 尾burst=每笔固定成本非长度相关, 轴双侧关闭
#endif

// ── W11 定态（压平后不再可关，非用户开关）────────────────────────────────────────
//   细粒度事件（MTE2_V / V_MTE3）+ 行/列双缓冲：流水结构的两半，缺一即 WA
//   （单缓冲下跨 tile RAW/WAR 无屏障覆盖）。
#ifndef W10_FINE_EVENT
#define W10_FINE_EVENT 1
#endif
#ifndef W10_DUAL_BUF
#define W10_DUAL_BUF 1
#endif

// ── 大点档常量（TILE=256，W11 定态；SEGW 封顶后 = TILE，单 tile 单段）────────────
//   LANE  = 256B/sizeof(DT)              -> Gather 满 repeat lane（fp32 64 / fp16 128）
//   ALANE = 256B/sizeof(CT) = 64         -> 算术/Compare 满 repeat lane（CT=float 恒定）
//   SEGW  = TILE = 256                   -> 算术/Compare/Select 段宽（count=256≥64 无 NOP）
//   槽容量随 TILE 派生，自动匹配每核最大行数（军规）；双缓冲第二组槽在 mask 之后。
#if W19_TILE128
constexpr uint32_t SB_TILE = 128u;   // W19 重测档：8 核 4 tile 深流水
#elif W27_TILE128
constexpr uint32_t SB_TILE = 128u;   // W27：中点单核 2 tile 双缓冲流水形态
#else
constexpr uint32_t SB_TILE = 256u;
#endif

// ── W27：中点档单核化（天南星实证：中点=地板+0.05，切分开销是主体）──────────────
//   W27_SINGLE_CORE（host 同名宏同门控）：中点档 block_dim=1，核 0 全量 B 行
//     走现役 GatherVec 流水（W16/W20const/W21/W22/W23 全保留）。
//   W27_TILE128：TILE 256→128，单核 2 tile 双缓冲流水（W23 编排复用，与
//     W27_SINGLE_CORE 组合使用；单独开=8 核×2 tile 形态不推荐）。
//   泛化：任意 33<B≤1024 走单核路径（范围判据，非测点假设）；大点/小点不动。
#ifndef W27_SINGLE_CORE
#define W27_SINGLE_CORE 0
#endif
#ifndef W27_TILE128
#define W27_TILE128 0
#endif

// ── W20-const：编译期常量合一表（盲发裁决：__gm__ const 段真板可用性）──────────
//   表 = f(dtype) 纯编译期常量，布局 [y1(256)|y2(256)|cy(256)] 与 S53 构造逐元素恒等。
//   kernel 一笔 3KB DMA 取表，S53 构造链（8 播种+S_V+倍增）整体删除。
//   sim 失明（__gm__ const 段不加载）已备案——本发即真板裁决。
#ifndef W20_GM_CONST_TABLE
#define W20_GM_CONST_TABLE 1
#endif

// ── W46：中点扁平化（8 核各自跑小点同构扁平路径，替代 GatherVec 框架机械）————
//   依据：小点 aiv=500ns vs 中点 aiv=1018ns（同 32 行/核）——差的 500ns 是框架机械。
#ifndef W46_MID_FLAT
#define W46_MID_FLAT 0
#endif
#ifndef W46_MID_TBL_DMA
#define W46_MID_TBL_DMA 1
#endif

// ── W38：中点紧凑消费（编译期紧凑常量表 × 运行期选表，零构造链）────────────────
//   中点每核 n≤128 行却按 3*TILE=768 满块 Gather/Cast。tv 仅两档（fp32 {64,128}/
//   fp16 {128,256}），预生成四张紧凑表 [y1(tv)|y2(tv)|cy(tv)]，运行期按 tv 选表。
//   W25 失败在"运行期前缀表构造"标量链——本方案纯 DMA 取表，无构造。
#ifndef W38_COMPACT_TBL
#define W38_COMPACT_TBL 1
#endif

#ifndef W30_SMALL_DMA_TBL
#define W30_SMALL_DMA_TBL 0
#endif

// 编译期常量合一表（布局 [y1|y2|cy]，与 S53 构造逐元素恒等）






template <typename DT_, uint32_t SCH_MODE_V, bool SCH_MODE_USE_TPL>
class SafetyBelt {
    static_assert(sizeof(DT_) == 4u || sizeof(DT_) == 2u, "only fp32 / fp16 are supported");
    using DT = DT_;
    using CT = float;                      // 算术位宽恒 fp32（fp16 走 SPLIT 升位，N2 红线）
    static constexpr bool F16 = (sizeof(DT_) == 2u);
    static constexpr uint32_t DT_BYTES = sizeof(DT_);
    static constexpr uint32_t CT_BYTES = sizeof(float);
    static constexpr uint32_t LANE = 256u / DT_BYTES;    // Gather 满 repeat lane
        static constexpr uint32_t TILE = 128u;
        static constexpr uint32_t SEGW = TILE;                 // cap 后 =128（Compare count=128≥64）
        static constexpr uint32_t DT_VEC_BYTES = TILE * DT_BYTES;
        static constexpr uint32_t CT_VEC_BYTES = TILE * CT_BYTES;
        static constexpr uint32_t B_TILE_BYTES = TILE * ROW_B;
        static constexpr uint32_t A_TILE_BYTES = TILE * ROW_A;
        static constexpr uint32_t TBL_BYTES = TILE * 4;
        static constexpr uint32_t OFF_Y1 = B_TILE_BYTES + A_TILE_BYTES;
        static constexpr uint32_t OFF_CY = OFF_Y1 + TBL_BYTES;
        static constexpr uint32_t GATHER0 = OFF_CY + TBL_BYTES;
        static constexpr uint32_t GV_BYTES = F16 ? (3u * DT_VEC_BYTES) : 0u;
        static constexpr uint32_t CT_Y1 = GATHER0 + GV_BYTES;
        static constexpr uint32_t CT_LIM = CT_Y1 + 3u * CT_VEC_BYTES;   // 修复：跳过 cyv 槽（曾与 lim 重叠）
        static constexpr uint32_t CT_ACC = CT_LIM + CT_VEC_BYTES;
        static constexpr uint32_t CNST_BYTES = SEGW * (F16 ? DT_BYTES : CT_BYTES);
        static constexpr uint32_t CNST_ONE = CT_ACC + CT_VEC_BYTES;
        static constexpr uint32_t CNST_ZERO = CNST_ONE + CNST_BYTES;
#if W34_AXPY_FUSE
        // W34：fp32 档 Select 输出独立槽（Axpy 改写 y1v 后 y1v 不再可作输出）；
        //   fp16 档输出本就落 gAllS（half 域），不占用。
        static constexpr uint32_t CT_OUT = CNST_ZERO + CNST_BYTES + 128;
#endif
        static constexpr uint32_t MSK_STRIDE = (SEGW / ALANE) * 32u;
        static_assert(TILE % LANE == 0 && TILE % ALANE == 0, "small tile lane alignment");
    static constexpr uint32_t ALANE = 256u / CT_BYTES;   // 算术满 repeat lane = 64
    static constexpr uint32_t ROW_B = 4u * DT_BYTES;     // person_boxes 行宽
    static constexpr uint32_t ROW_A = 2u * DT_BYTES;     // anchor_points 行宽
        static constexpr uint32_t GV_TILE = SB_TILE;
        static constexpr uint32_t GV_SEGW = GV_TILE;                    // 算术段宽=TILE（单 tile 单段，count=256）
        static constexpr uint32_t GV_DT_VEC_BYTES = GV_TILE * DT_BYTES; // 每根 Gather 目标向量
        static constexpr uint32_t GV_CT_VEC_BYTES = GV_TILE * CT_BYTES; // 每根算术向量
        static constexpr uint32_t GV_MASK_PER_SEG = 32u;
        static constexpr uint32_t GV_MSK_NBLK = (SEGW + ALANE - 1u) / ALANE;
        static constexpr uint32_t GV_MSK_STRIDE = MSK_NBLK * MASK_PER_SEG;
        static_assert(TILE % LANE == 0, "TILE must be a multiple of the DT repeat lane");
        static_assert(TILE % ALANE == 0, "TILE must be a multiple of the CT repeat lane");
        static_assert(SEGW % ALANE == 0, "segment must stay CT-repeat aligned");
        static_assert(LANE * DT_BYTES == 256u, "Gather count must be exactly one DT repeat");
        static_assert(ALANE * CT_BYTES == 256u, "arith count must be exactly one CT repeat");
        static_assert(SEGW * CT_BYTES / 256u <= 255u, "segment repeat beyond one instruction budget");

        AscendC::GlobalTensor<DT> bGm, aGm, oGm;
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(personBoxes));
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(anchorPoints));
        oGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(violationScore));

        const uint32_t core_id = AscendC::GetBlockIdx();
        uint32_t raw = (block_dim == 8u) ? ((total_b + 7u) >> 3)
                                         : (block_dim == 4u) ? ((total_b + 3u) >> 2)
                                         : (block_dim == 16u) ? ((total_b + 15u) >> 4)  // W42: bd=16 移位快路径
                                         : ((total_b + block_dim - 1u) / block_dim);   // W36: 移位快路径族+通用回退保泛化
        constexpr uint32_t PER_ALIGN = 8u;
        uint32_t per = ((raw + 7u) >> 3) << 3;   // W36: align8 移位（除法→移位）
        if (per == 0) per = PER_ALIGN;
        const uint32_t b_start = core_id * per;
        uint32_t b_end = b_start + per;
        if (b_end > total_b_) b_end = total_b;
        if (b_start >= b_end) return;
        const uint32_t n = b_end - b_start;

        // ---- UB 布局：按 TILE 分区，槽距 32B 倍数；双缓冲第二组在 mask 之后 ----
        static constexpr uint32_t GV_B_TILE_BYTES = GV_TILE * ROW_B;
        static constexpr uint32_t GV_A_TILE_BYTES = GV_TILE * ROW_A;
        static constexpr uint32_t GV_TBL_BYTES = GV_TILE * 4;           // 每张 offset 表（uint32）
        static constexpr uint32_t GV_OFF_Y1 = B_TILE_BYTES + A_TILE_BYTES;
#if W15_GATHER_MERGE
        // W15：三段合一表 [y1 | y2(=y1+2DT) | cy(=i*ROW_A+DT+B_TILE_BYTES)]，绝对偏移。
        //   boxes/anchors 行区连续（bTile/aTile 同区），cy 项加行区偏移即可共基址。
        static constexpr uint32_t GV_OFF_Y2 = OFF_Y1 + TBL_BYTES;
        static constexpr uint32_t GV_OFF_CY = OFF_Y2 + TBL_BYTES;
        static constexpr uint32_t GV_GATHER0 = OFF_CY + TBL_BYTES;
#else
        static constexpr uint32_t GV_OFF_CY = OFF_Y1 + TBL_BYTES;    // S53/T1：y2 表删除（srcBaseAddr 复用）
        static constexpr uint32_t GV_GATHER0 = OFF_CY + TBL_BYTES;
#endif
        static constexpr uint32_t GV_GV_BYTES = F16 ? (3u * DT_VEC_BYTES) : 0u;  // SPLIT：Gather 目标独立槽
        static constexpr uint32_t GV_CT_Y1 = GATHER0 + GV_BYTES;     // fp32 时与 GATHER0 同址
        static constexpr uint32_t GV_CT_Y2 = CT_Y1 + CT_VEC_BYTES;
        static constexpr uint32_t GV_CT_CY = CT_Y2 + CT_VEC_BYTES;
        static constexpr uint32_t GV_CT_LIM = CT_CY + CT_VEC_BYTES;
        static constexpr uint32_t GV_CT_ACC = CT_LIM + CT_VEC_BYTES;
        static constexpr uint32_t GV_CNST_BYTES = SEGW * (F16 ? DT_BYTES : CT_BYTES);
        static constexpr uint32_t GV_CNST_ONE = CT_ACC + CT_VEC_BYTES;
        static constexpr uint32_t GV_CNST_ZERO = CNST_ONE + CNST_BYTES;
        static constexpr uint32_t GV_SLOT_MASK = CNST_ZERO + CNST_BYTES + 128;
        static constexpr uint32_t GV_MASK_BYTES = (GV_TILE / GV_SEGW) * MSK_STRIDE;
        static constexpr uint32_t GV_COL_BYTES = 3u * (F16 ? DT_VEC_BYTES : CT_VEC_BYTES);
        static constexpr uint32_t GV_COL1_BASE = SLOT_MASK + MASK_BYTES;
        static constexpr uint32_t GV_ROW_BYTES = B_TILE_BYTES + A_TILE_BYTES;
        static constexpr uint32_t GV_ROW1_BASE = COL1_BASE + COL_BYTES;
        static_assert(ROW1_BASE + ROW_BYTES <= 192u * 1024u, "W11 double buffers exceed UB");

        AscendC::LocalTensor<DT> bTile(AscendC::TPosition::VECCALC, 0, GV_TILE * 4);
        AscendC::LocalTensor<DT> aTile(AscendC::TPosition::VECCALC, B_TILE_BYTES, TILE * 2);
        AscendC::LocalTensor<int32_t> y1Off(AscendC::TPosition::VECCALC, OFF_Y1, TILE);
        AscendC::LocalTensor<int32_t> cyOff(AscendC::TPosition::VECCALC, OFF_CY, TILE);

public:
    __aicore__ inline SafetyBelt() {}

    __aicore__ inline SafetyBelt(GM_ADDR person_boxes, GM_ADDR anchor_points,
                                 GM_ADDR threshold_gm, GM_ADDR violation_score,
                                 uint32_t tb, float wr) {
        personBoxes_ = person_boxes;
        anchorPoints_ = anchor_points;
        threshold_ = threshold_gm;
        violationScore_ = violation_score;
        total_b_ = tb;
        waist_ratio_ = wr;
    }

    // 运行期三档分派（判据与 host SelectSafetyBeltSchMode 同源：B 的范围连续划分）
    // W35 薄入口：Init+Process 合并为唯一启动函数 Start——成员读即局部变量，
    //   GetBlockNum() 单点调用；档位函数参数直传（无壳）。
    // W44（B8/D1-K1）：档位编译期分派——SCH_MODE 6 值域 {0,1}=小 {2,3}=中 {4,5}=大。
    //   每 dtype×档一份极简产物（I-cache 足迹减半以上）；运行期 total_b 只算切分。
    //   兼容回退：SCH_MODE 值域外的旧键走运行期分支（TILING_KEY_IS 语义由 opbuild
    //   绑定保证，此处 constexpr 判档位）。
    template <uint32_t SCH_MODE>
    __aicore__ inline void StartT() {
        GM_ADDR personBoxes = personBoxes_;
        GM_ADDR anchorPoints = anchorPoints_;
        GM_ADDR threshold = threshold_;
        GM_ADDR violationScore = violationScore_;
        const uint32_t total_b = total_b_;
        const float waist_ratio = waist_ratio_;
        const uint32_t block_dim = static_cast<uint32_t>(AscendC::GetBlockNum());
#if W15_SMALL_CASE
        if constexpr (SCH_MODE == SB_TPL_SCH_F32_SMALL || SCH_MODE == SB_TPL_SCH_F16_SMALL) {
            ProcessSmallVec(personBoxes, anchorPoints, threshold, violationScore,
                            total_b, waist_ratio, block_dim);
            return;
        }
#else
        if constexpr (SCH_MODE == SB_TPL_SCH_F32_SMALL || SCH_MODE == SB_TPL_SCH_F16_SMALL) {
            ProcessSmallPoint(personBoxes, anchorPoints, threshold, violationScore,
                              total_b, waist_ratio, block_dim);
            return;
        }
#endif
#if W15_MID_PIPELINE
        if constexpr (SCH_MODE == SB_TPL_SCH_F32_MID || SCH_MODE == SB_TPL_SCH_F16_MID) {
#if W46_MID_FLAT
            ProcessMidFlat(personBoxes, anchorPoints, threshold, violationScore,
                           total_b, waist_ratio, block_dim);
            return;
#else
            ProcessGatherVec(personBoxes, anchorPoints, threshold, violationScore,
                             total_b, waist_ratio, block_dim);
            return;
#endif
        }
        if constexpr (SCH_MODE == SB_TPL_SCH_F32_BIG || SCH_MODE == SB_TPL_SCH_F16_BIG) {
            ProcessGatherVec(personBoxes, anchorPoints, threshold, violationScore,
                             total_b, waist_ratio, block_dim);
            return;
        }
#else
        if constexpr (SCH_MODE == SB_TPL_SCH_F32_MID || SCH_MODE == SB_TPL_SCH_F16_MID) {
            ProcessLargeDirect(personBoxes, anchorPoints, threshold, violationScore,
                               total_b, waist_ratio, block_dim);
            return;
        }
        if constexpr (SCH_MODE == SB_TPL_SCH_F32_BIG || SCH_MODE == SB_TPL_SCH_F16_BIG) {
            ProcessGatherVec(personBoxes, anchorPoints, threshold, violationScore,
                             total_b, waist_ratio, block_dim);
            return;
        }
#endif
    }

    __aicore__ inline void Start() {
        if constexpr (SCH_MODE_USE_TPL) {
            StartT<SCH_MODE_V>();
        } else {
            StartRuntime();
        }
    }

    __aicore__ inline void StartRuntime() {
        GM_ADDR personBoxes = personBoxes_;
        GM_ADDR anchorPoints = anchorPoints_;
        GM_ADDR threshold = threshold_;
        GM_ADDR violationScore = violationScore_;
        const uint32_t total_b = total_b_;
        const float waist_ratio = waist_ratio_;
        const uint32_t block_dim = static_cast<uint32_t>(AscendC::GetBlockNum());
        if (total_b <= SB_SMALL_MAX_B) {
#if W15_SMALL_CASE
            ProcessSmallVec(personBoxes, anchorPoints, threshold, violationScore,
                            total_b, waist_ratio, block_dim);
#else
            ProcessSmallPoint(personBoxes, anchorPoints, threshold, violationScore,
                              total_b, waist_ratio, block_dim);
#endif
        } else if (total_b <= SB_MID_MAX_B) {
#if W15_MID_PIPELINE
#if W46_MID_FLAT
            ProcessMidFlat(personBoxes, anchorPoints, threshold, violationScore,
                           total_b, waist_ratio, block_dim);
#else
            ProcessGatherVec(personBoxes, anchorPoints, threshold, violationScore,
                             total_b, waist_ratio, block_dim);
#endif
#else
            ProcessLargeDirect(personBoxes, anchorPoints, threshold, violationScore,
                               total_b, waist_ratio, block_dim);
#endif
        } else {
            ProcessGatherVec(personBoxes, anchorPoints, threshold, violationScore,
                             total_b, waist_ratio, block_dim);
        }
    }

private:
    // ── 小点：B ≤ 33，core-0-only 直通静态管线（Z2 零壳：tiling/threshold 标量直读）──
    // ---- ProcessSmallPoint 通用计算链（CodeCheck nbnc 拆分；语义逐指令等价）----
    template <typename PB, typename PA, typename PO, typename DT2>
    // 4-Way 与尾循环（CodeCheck nbnc 二级拆分）
    template <typename PB, typename PA, typename PO, typename DT2>
    __aicore__ inline void SmallGenericTail(PB &p_b, PA &p_a, PO &p_o, float wrL, float thL,
                                            DT2 one_val, DT2 zero_val, uint32_t rowsProc) {
        uint32_t i = 0;
for (; i + 4 <= rowsProc; i += 4, p_b += 16, p_a += 8, p_o += 4) {
    float y1_0 = static_cast<float>(p_b[1]);
    float y2_0 = static_cast<float>(p_b[3]);
    float cy_0 = static_cast<float>(p_a[1]);
    float y1_1 = static_cast<float>(p_b[5]);
    float y2_1 = static_cast<float>(p_b[7]);
    float cy_1 = static_cast<float>(p_a[3]);
    float y1_2 = static_cast<float>(p_b[9]);
    float y2_2 = static_cast<float>(p_b[11]);
    float cy_2 = static_cast<float>(p_a[5]);
    float y1_3 = static_cast<float>(p_b[13]);
    float y2_3 = static_cast<float>(p_b[15]);
    float cy_3 = static_cast<float>(p_a[7]);
    float lim_0 = (y1_0 + wrL * (y2_0 - y1_0)) + thL;
    float lim_1 = (y1_1 + wrL * (y2_1 - y1_1)) + thL;
    float lim_2 = (y1_2 + wrL * (y2_2 - y1_2)) + thL;
    float lim_3 = (y1_3 + wrL * (y2_3 - y1_3)) + thL;
    p_o[0] = (cy_0 > lim_0) ? one_val : zero_val;
    p_o[1] = (cy_1 > lim_1) ? one_val : zero_val;
    p_o[2] = (cy_2 > lim_2) ? one_val : zero_val;
    p_o[3] = (cy_3 > lim_3) ? one_val : zero_val;
}
for (; i < rowsProc; ++i, p_b += 4, p_a += 2, p_o += 1) {
    float y1 = static_cast<float>(p_b[1]);
    float y2 = static_cast<float>(p_b[3]);
    float cy = static_cast<float>(p_a[1]);
    float limit = (y1 + wrL * (y2 - y1)) + thL;
    p_o[0] = (cy > limit) ? one_val : zero_val;
}    }


    __aicore__ inline void SmallGenericLoop(PB &p_b, PA &p_a, PO &p_o, float wrL, float thL,
                                            DT2 one_val, DT2 zero_val, uint32_t rowsProc) {
        uint32_t i = 0;
for (; i + 8 <= rowsProc; i += 8, p_b += 32, p_a += 16, p_o += 8) {
    float y1_0 = static_cast<float>(p_b[1]);   float y2_0 = static_cast<float>(p_b[3]);   float cy_0 = static_cast<float>(p_a[1]);
    float y1_1 = static_cast<float>(p_b[5]);   float y2_1 = static_cast<float>(p_b[7]);   float cy_1 = static_cast<float>(p_a[3]);
    float y1_2 = static_cast<float>(p_b[9]);   float y2_2 = static_cast<float>(p_b[11]);  float cy_2 = static_cast<float>(p_a[5]);
    float y1_3 = static_cast<float>(p_b[13]);  float y2_3 = static_cast<float>(p_b[15]);  float cy_3 = static_cast<float>(p_a[7]);
    float y1_4 = static_cast<float>(p_b[17]);  float y2_4 = static_cast<float>(p_b[19]);  float cy_4 = static_cast<float>(p_a[9]);
    float y1_5 = static_cast<float>(p_b[21]);  float y2_5 = static_cast<float>(p_b[23]);  float cy_5 = static_cast<float>(p_a[11]);
    float y1_6 = static_cast<float>(p_b[25]);  float y2_6 = static_cast<float>(p_b[27]);  float cy_6 = static_cast<float>(p_a[13]);
    float y1_7 = static_cast<float>(p_b[29]);  float y2_7 = static_cast<float>(p_b[31]);  float cy_7 = static_cast<float>(p_a[15]);
    float d0 = y2_0 - y1_0, d1 = y2_1 - y1_1, d2 = y2_2 - y1_2, d3 = y2_3 - y1_3;
    float d4 = y2_4 - y1_4, d5 = y2_5 - y1_5, d6 = y2_6 - y1_6, d7 = y2_7 - y1_7;
    // S31/G-1: 官方结合顺序 (y1 + wr*d) + th
    float m0 = d0 * wr, m1 = d1 * wr, m2 = d2 * wr, m3 = d3 * wr;
    float m4 = d4 * wr, m5 = d5 * wr, m6 = d6 * wr, m7 = d7 * wr;
    float t0 = y1_0 + m0, t1 = y1_1 + m1, t2 = y1_2 + m2, t3 = y1_3 + m3;
    float t4 = y1_4 + m4, t5 = y1_5 + m5, t6 = y1_6 + m6, t7 = y1_7 + m7;
    p_o[0] = (cy_0 > t0 + th) ? one_val : zero_val;
    p_o[1] = (cy_1 > t1 + th) ? one_val : zero_val;
    p_o[2] = (cy_2 > t2 + th) ? one_val : zero_val;
    p_o[3] = (cy_3 > t3 + th) ? one_val : zero_val;
    p_o[4] = (cy_4 > t4 + th) ? one_val : zero_val;
    p_o[5] = (cy_5 > t5 + th) ? one_val : zero_val;
    p_o[6] = (cy_6 > t6 + th) ? one_val : zero_val;
    p_o[7] = (cy_7 > t7 + th) ? one_val : zero_val;
}
        SmallGenericTail(p_b, p_a, p_o, wrL, thL, one_val, zero_val, rowsProc);
    }


    // ---- 快捷链（waist==1 或 threshold==0）：返回 true 表示已处理完毕 ----
    template <typename PB, typename PA, typename PO, typename DT2>
    template <typename PB, typename PA, typename PO, typename DT2>
    __aicore__ inline void SmallWrShortcut(PB &p_b, PA &p_a, PO &p_o, float ths,
                                           DT2 one_val, DT2 zero_val, uint32_t rowsProc) {
        uint32_t i = 0;
    const float thL2 = scNoThs ? 0.0f : thL;
    for (; i + 8 <= rowsProc; i += 8, p_b += 32, p_a += 16, p_o += 8) {
        p_o[0] = (p_a[1]  > p_b[3]  + thL2) ? one_val : zero_val;
        p_o[1] = (p_a[3]  > p_b[7]  + thL2) ? one_val : zero_val;
        p_o[2] = (p_a[5]  > p_b[11] + thL2) ? one_val : zero_val;
        p_o[3] = (p_a[7]  > p_b[15] + thL2) ? one_val : zero_val;
        p_o[4] = (p_a[9]  > p_b[19] + thL2) ? one_val : zero_val;
        p_o[5] = (p_a[11] > p_b[23] + thL2) ? one_val : zero_val;
        p_o[6] = (p_a[13] > p_b[27] + thL2) ? one_val : zero_val;
        p_o[7] = (p_a[15] > p_b[31] + thL2) ? one_val : zero_val;
    }
    for (; i < rowsProc; ++i, p_b += 4, p_a += 2, p_o += 1) {
        p_o[0] = (p_a[1] > p_b[3] + thL2) ? one_val : zero_val;
    }    }

    template <typename PB, typename PA, typename PO, typename DT2>
    __aicore__ inline void SmallTh0Shortcut(PB &p_b, PA &p_a, PO &p_o, float wrL,
                                            DT2 one_val, DT2 zero_val, uint32_t rowsProc) {
        uint32_t i = 0;
    for (; i + 8 <= rowsProc; i += 8, p_b += 32, p_a += 16, p_o += 8) {
        float lim0 = static_cast<float>(p_b[1]) + wrL * (static_cast<float>(p_b[3]) - static_cast<float>(p_b[1]));
        float lim1 = static_cast<float>(p_b[5]) + wrL * (static_cast<float>(p_b[7]) - static_cast<float>(p_b[5]));
        float lim2 = static_cast<float>(p_b[9]) + wrL * (static_cast<float>(p_b[11]) - static_cast<float>(p_b[9]));
        float lim3 = static_cast<float>(p_b[13]) + wrL * (static_cast<float>(p_b[15]) - static_cast<float>(p_b[13]));
        float lim4 = static_cast<float>(p_b[17]) + wrL * (static_cast<float>(p_b[19]) - static_cast<float>(p_b[17]));
        float lim5 = static_cast<float>(p_b[21]) + wrL * (static_cast<float>(p_b[23]) - static_cast<float>(p_b[21]));
        float lim6 = static_cast<float>(p_b[25]) + wrL * (static_cast<float>(p_b[27]) - static_cast<float>(p_b[25]));
        float lim7 = static_cast<float>(p_b[29]) + wrL * (static_cast<float>(p_b[31]) - static_cast<float>(p_b[29]));
        p_o[0] = (static_cast<float>(p_a[1])  > lim0) ? one_val : zero_val;
        p_o[1] = (static_cast<float>(p_a[3])  > lim1) ? one_val : zero_val;
        p_o[2] = (static_cast<float>(p_a[5])  > lim2) ? one_val : zero_val;
        p_o[3] = (static_cast<float>(p_a[7])  > lim3) ? one_val : zero_val;
        p_o[4] = (static_cast<float>(p_a[9])  > lim4) ? one_val : zero_val;
        p_o[5] = (static_cast<float>(p_a[11]) > lim5) ? one_val : zero_val;
        p_o[6] = (static_cast<float>(p_a[13]) > lim6) ? one_val : zero_val;
        p_o[7] = (static_cast<float>(p_a[15]) > lim7) ? one_val : zero_val;
    }
    for (; i < rowsProc; ++i, p_b += 4, p_a += 2, p_o += 1) {
        float y1 = static_cast<float>(p_b[1]);
        float y2 = static_cast<float>(p_b[3]);
        float cy = static_cast<float>(p_a[1]);
        p_o[0] = (cy > (y1 + wrL * (y2 - y1))) ? one_val : zero_val;
    }    }

    template <typename PB, typename PA, typename PO, typename DT2>
    __aicore__ inline bool SmallShortcuts(PB &p_b, PA &p_a, PO &p_o, float wrL, float thL,
                                          GM_ADDR threshold_gm, DT2 one_val, DT2 zero_val, uint32_t rowsProc) {
        const bool scWr1s = (wrL == 1.0f);
        const bool scNoThs = (threshold_gm == nullptr);
        const bool scTh0s = (threshold_gm != nullptr) && (thL == 0.0f);
        if (scWr1s) {
            SmallWrShortcut(p_b, p_a, p_o, scNoThs ? 0.0f : thL, one_val, zero_val, rowsProc);
            return true;
        }
        if (scTh0s) {
            SmallTh0Shortcut(p_b, p_a, p_o, wrL, one_val, zero_val, rowsProc);
            return true;
        }
        return false;
    }




    // ---- 小点写出段（CodeCheck nbnc 拆分；三种 W24/W18 形态与原逐字一致）----
    __aicore__ inline void SmallWriteOut(AscendC::GlobalTensor<DT> &oGm,
                                         AscendC::LocalTensor<DT> &oLocal, uint32_t rowsTotal) {
#if !W24_SCALAR_OUT
        // S33/SP3：细粒度事件 S_MTE3（标量写 -> DMA 写出）
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
#endif

#if W24_SCALAR_OUT
        // W24：输出已由标量循环直写 GM，无 MTE3 写出段
#elif W18_WRITE_SPLIT
        {
            const uint32_t wlen = static_cast<uint32_t>(rowsTotal * sizeof(DT));
            if (wlen >= 32u && (wlen & 31u) != 0u) {
                const uint32_t mainB = wlen & ~31u;
                AscendC::DataCopyExtParams om{1, mainB, 0, 0, 0};
                AscendC::DataCopyPad(oGm[0], oLocal, om);
                AscendC::DataCopyExtParams orr{1, wlen - mainB, 0, 0, 0};
                AscendC::DataCopyPad(oGm[mainB / sizeof(DT)], oLocal[mainB / sizeof(DT)], orr);
            } else {
                AscendC::DataCopyExtParams oCp{1, wlen, 0, 0, 0};
                AscendC::DataCopyPad(oGm[0], oLocal, oCp);
            }
        }
#else
        AscendC::DataCopyExtParams oCp{1, static_cast<uint32_t>(rowsTotal * sizeof(DT)), 0, 0, 0};
        AscendC::DataCopyPad(oGm[0], oLocal, oCp);
#endif    }

    // ① 两笔数据 DMA 先发（多路 DMA 并发、共享同一屏障）
    __aicore__ inline void SmallLoadInputs(AscendC::GlobalTensor<DT> &bGm, AscendC::GlobalTensor<DT> &aGm,
                                           AscendC::LocalTensor<DT> &bLocal, AscendC::LocalTensor<DT> &aLocal,
                                           uint32_t b_len, uint32_t a_len) {
        AscendC::DataCopyExtParams bCp{1, b_len, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT> bPad{false, 0, 0, 0};
        AscendC::DataCopyPad(bLocal, bGm[0], bCp, bPad);
        AscendC::DataCopyExtParams aCp{1, a_len, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT> aPad{false, 0, 0, 0};
        AscendC::DataCopyPad(aLocal, aGm[0], aCp, aPad);
    }

    // threshold 裸 __gm__ 标量直读（省一笔 DMA 固定握手）
    __aicore__ inline float SmallReadThreshold(GM_ADDR threshold_gm) {
        float th = SmallReadThreshold(threshold_gm);
        return th;
    }


    __aicore__ inline void ProcessSmallPoint(GM_ADDR personBoxes, GM_ADDR anchorPoints, GM_ADDR threshold_gm, GM_ADDR violationScore, uint32_t total_b, float waist_ratio, uint32_t block_dim) {
        if (AscendC::GetBlockIdx() != 0u) { return; }   // S45-B 空核早退（launch 形态选择）
                AscendC::GlobalTensor<DT> bGm, aGm, oGm;
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(personBoxes));
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(anchorPoints));
        oGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(violationScore));

        constexpr uint32_t ubBoxes = 0;      // B=33 fp32 needs 528B
        constexpr uint32_t ubAnchors = 1024; // B=33 fp32 needs 264B
        constexpr uint32_t ubScore = 3072;
        AscendC::LocalTensor<DT> bLocal(AscendC::TPosition::VECCALC, ubBoxes, 1024);
        AscendC::LocalTensor<DT> aLocal(AscendC::TPosition::VECCALC, ubAnchors, 512);
        AscendC::LocalTensor<DT> oLocal(AscendC::TPosition::VECCALC, ubScore, 128);

        uint32_t b_len = total_b * 4 * sizeof(DT);
        uint32_t a_len = total_b * 2 * sizeof(DT);
        if (b_len < 32) b_len = 32;
        if (a_len < 32) a_len = 32;
        // 输入 DMA 长度对齐 32B（GM 读越界 ≤31B，池内安全；写出仍真实行数）
        b_len = W17_EVENIZE ? ((b_len + 31u) & ~31u) : b_len;
        a_len = W17_EVENIZE ? ((a_len + 31u) & ~31u) : a_len;
        // 行处理偶化：尾组（≤7 行）读 UB 垃圾行，输出不写出 ⇒ 无标量尾循环
        const uint32_t rowsProc = W17_EVENIZE ? ((total_b + 7u) & ~7u) : total_b;

        SmallLoadInputs(bGm, aGm, bLocal, aLocal, b_len, a_len);

        // ② threshold 裸 __gm__ 标量直读（省一笔 DMA 固定握手）
        const float th = SmallReadThreshold(threshold_gm);

        // S33/SP3：细粒度事件 MTE2_S（DMA 落地 -> 标量读）
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);

        const float wr = td_waist;   // 入口已付款的热读，零新增 GM 事务
        const DT one_val = static_cast<DT>(1.0f);
        const DT zero_val = static_cast<DT>(0.0f);

        auto p_b = &bLocal(0);
        auto p_a = &aLocal(0);
#if W24_SCALAR_OUT
        // W24：输出直写 GM（跳过 oLocal + MTE3 写出全程）
        __gm__ DT* p_o = reinterpret_cast<__gm__ DT*>(violationScore);
#else
        auto p_o = &oLocal(0);
#endif

        // 小点 8-Way 展开（先 8-Way 后 4-Way 后尾）
        uint32_t i = 0;
#if W26_WR_SHORTCUT
        if (SmallShortcuts(p_b, p_a, p_o, wr, th, threshold_gm, one_val, zero_val, rowsProc)) {
            return;
        }
#endif
        SmallGenericLoop(p_b, p_a, p_o, wr, th, one_val, zero_val, rowsProc);
        SmallWriteOut(oGm, oLocal, total_b);
    }

    // ── 中点：33 < B ≤ 1024，标量直通（fp32 8-Way / fp16 4-Way）──────────────────
    // ---- ProcessLargeDirect 拆分（CodeCheck nbnc）：装载 / 计算 / 写出 ----
    struct LargeDirectCtx {
        uint32_t n8;
        float th;
    };

    __aicore__ inline void LargeDirectLoad(AscendC::GlobalTensor<DT> &bGm, AscendC::GlobalTensor<DT> &aGm,
                                           AscendC::LocalTensor<DT> &bL, AscendC::LocalTensor<DT> &aL,
                                           uint32_t b_start, uint32_t n, GM_ADDR threshold_gm,
                                           LargeDirectCtx &ctx) {
AscendC::GlobalTensor<DT> bGm, aGm, oGm;
bGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(personBoxes));
aGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(anchorPoints));
oGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(violationScore));

const uint32_t core_id = AscendC::GetBlockIdx();
uint32_t raw = (block_dim == 8u) ? ((total_b + 7u) >> 3)
                                 : (block_dim == 4u) ? ((total_b + 3u) >> 2)
                                 : (block_dim == 16u) ? ((total_b + 15u) >> 4)  // W42: bd=16 移位快路径
                                 : ((total_b + block_dim - 1u) / block_dim);   // W36: 移位快路径族+通用回退保泛化
uint32_t per = ((raw + 7u) >> 3) << 3;   // W36: align8 移位（除法→移位）
if (per == 0) per = 8;
const uint32_t b_start = core_id * per;
uint32_t b_end = b_start + per;
if (b_end > total_b) b_end = total_b;
if (b_start >= b_end) return;
const uint32_t n = b_end - b_start;

// 静态 LocalTensor（免 TPipe）；槽按 8 核最大 per-core 256 行 fp32 布局
AscendC::LocalTensor<DT> bL(AscendC::TPosition::VECCALC, 0, 4096);
AscendC::LocalTensor<DT> aL(AscendC::TPosition::VECCALC, 4096, 2048);
AscendC::LocalTensor<DT> sL(AscendC::TPosition::VECCALC, 6144, 1024);

#if W17_EVENIZE
// 输入 DMA 长度对齐 32B；行处理 n8=align8(n)（尾组垃圾行不写出）
const uint32_t blen32 = (n * 4 * sizeof(DT) + 31u) & ~31u;
const uint32_t alen32 = (n * 2 * sizeof(DT) + 31u) & ~31u;
const uint32_t n8 = (n + 7u) & ~7u;
AscendC::DataCopyExtParams bc{1, blen32, 0, 0, 0};
AscendC::DataCopyPadExtParams<DT> bp{false, 0, 0, 0};
AscendC::DataCopyPad(bL, bGm[b_start * 4], bc, bp);
AscendC::DataCopyExtParams ac{1, alen32, 0, 0, 0};
AscendC::DataCopyPadExtParams<DT> ap{false, 0, 0, 0};
AscendC::DataCopyPad(aL, aGm[b_start * 2], ac, ap);
#else
AscendC::DataCopyExtParams bc{1, static_cast<uint32_t>(n * 4 * sizeof(DT)), 0, 0, 0};
AscendC::DataCopyPadExtParams<DT> bp{false, 0, 0, 0};
AscendC::DataCopyPad(bL, bGm[b_start * 4], bc, bp);
AscendC::DataCopyExtParams ac{1, static_cast<uint32_t>(n * 2 * sizeof(DT)), 0, 0, 0};
AscendC::DataCopyPadExtParams<DT> ap{false, 0, 0, 0};
AscendC::DataCopyPad(aL, aGm[b_start * 2], ac, ap);
const uint32_t n8 = n;
#endif

float th = 0.0f;
if (threshold_gm != nullptr) {
    const __gm__ DT* thp = reinterpret_cast<__gm__ DT*>(threshold_gm);
    th = static_cast<float>(thp[0]);
}
AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 4-Way + 尾循环（从 LargeDirectCompute 拆出）
    __aicore__ inline void LargeDirectComputeTail(DT* pb, DT* pa, DT* po, uint32_t &i,
                                                  uint32_t n8, float th, float wr,
                                                  const DT one_v, const DT zero_v) {
for (; i + 4 <= n8; i += 4, pb += 16, pa += 8, po += 4) {
float a0 = static_cast<float>(pb[1]), b0 = static_cast<float>(pb[3]), c0 = static_cast<float>(pa[1]);
float a1 = static_cast<float>(pb[5]), b1 = static_cast<float>(pb[7]), c1 = static_cast<float>(pa[3]);
float a2 = static_cast<float>(pb[9]), b2 = static_cast<float>(pb[11]), c2 = static_cast<float>(pa[5]);
float a3 = static_cast<float>(pb[13]), b3 = static_cast<float>(pb[15]), c3 = static_cast<float>(pa[7]);
float l0 = (a0 + wr * (b0 - a0)) + th;
float l1 = (a1 + wr * (b1 - a1)) + th;
float l2 = (a2 + wr * (b2 - a2)) + th;
float l3 = (a3 + wr * (b3 - a3)) + th;
po[0] = (c0 > l0) ? one_v : zero_v;
po[1] = (c1 > l1) ? one_v : zero_v;
po[2] = (c2 > l2) ? one_v : zero_v;
po[3] = (c3 > l3) ? one_v : zero_v;
}
for (; i < n8; ++i, pb += 4, pa += 2, po += 1) {
float a = static_cast<float>(pb[1]);
float b = static_cast<float>(pb[3]);
float c = static_cast<float>(pa[1]);
float l = (a + wr * (b - a)) + th;
po[0] = (c > l) ? one_v : zero_v;
}    }


    __aicore__ inline void LargeDirectCompute(AscendC::LocalTensor<DT> &bL, AscendC::LocalTensor<DT> &aL,
                                              AscendC::LocalTensor<DT> &sL, uint32_t n8,
                                              float th, float waist_ratio) {

const DT one_v = static_cast<DT>(1.0f);
const DT zero_v = static_cast<DT>(0.0f);
const float wr = waist_ratio;
auto pb = &bL(0);
auto pa = &aL(0);
auto po = &sL(0);

uint32_t i = 0;
// fp32 8-Way / fp16 4-Way（v43 实测口径）
if constexpr (!F16) {
for (; i + 8 <= n8; i += 8, pb += 32, pa += 16, po += 8) {
    float a0 = static_cast<float>(pb[1]),  b0 = static_cast<float>(pb[3]),  c0 = static_cast<float>(pa[1]);
    float a1 = static_cast<float>(pb[5]),  b1 = static_cast<float>(pb[7]),  c1 = static_cast<float>(pa[3]);
    float a2 = static_cast<float>(pb[9]),  b2 = static_cast<float>(pb[11]), c2 = static_cast<float>(pa[5]);
    float a3 = static_cast<float>(pb[13]), b3 = static_cast<float>(pb[15]), c3 = static_cast<float>(pa[7]);
    float a4 = static_cast<float>(pb[17]), b4 = static_cast<float>(pb[19]), c4 = static_cast<float>(pa[9]);
    float a5 = static_cast<float>(pb[21]), b5 = static_cast<float>(pb[23]), c5 = static_cast<float>(pa[11]);
    float a6 = static_cast<float>(pb[25]), b6 = static_cast<float>(pb[27]), c6 = static_cast<float>(pa[13]);
    float a7 = static_cast<float>(pb[29]), b7 = static_cast<float>(pb[31]), c7 = static_cast<float>(pa[15]);
    float d0 = b0 - a0, d1 = b1 - a1, d2 = b2 - a2, d3 = b3 - a3;
    float d4 = b4 - a4, d5 = b5 - a5, d6 = b6 - a6, d7 = b7 - a7;
    float m0 = d0 * wr, m1 = d1 * wr, m2 = d2 * wr, m3 = d3 * wr;
    float m4 = d4 * wr, m5 = d5 * wr, m6 = d6 * wr, m7 = d7 * wr;
    float t0 = a0 + m0, t1 = a1 + m1, t2 = a2 + m2, t3 = a3 + m3;
    float t4 = a4 + m4, t5 = a5 + m5, t6 = a6 + m6, t7 = a7 + m7;
    po[0] = (c0 > t0 + th) ? one_v : zero_v;
    po[1] = (c1 > t1 + th) ? one_v : zero_v;
    po[2] = (c2 > t2 + th) ? one_v : zero_v;
    po[3] = (c3 > t3 + th) ? one_v : zero_v;
    po[4] = (c4 > t4 + th) ? one_v : zero_v;
    po[5] = (c5 > t5 + th) ? one_v : zero_v;
    po[6] = (c6 > t6 + th) ? one_v : zero_v;
    po[7] = (c7 > t7 + th) ? one_v : zero_v;
}
}
        LargeDirectComputeTail(pb, pa, po, i, n8, th, wr, one_v, zero_v);
for (; i < n8; ++i, pb += 4, pa += 2, po += 1) {
    float a = static_cast<float>(pb[1]);
    float b = static_cast<float>(pb[3]);
    float c = static_cast<float>(pa[1]);
    float l = (a + wr * (b - a)) + th;
    po[0] = (c > l) ? one_v : zero_v;
}

AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LargeDirectStore(AscendC::GlobalTensor<DT> &oGm, AscendC::LocalTensor<DT> &sL,
                                            uint32_t b_start, uint32_t n) {
#if W18_WRITE_SPLIT
{
    const uint32_t wlen = static_cast<uint32_t>(n * sizeof(DT));
    if (wlen >= 32u && (wlen & 31u) != 0u) {
        const uint32_t mainB = wlen & ~31u;
        AscendC::DataCopyExtParams sm{1, mainB, 0, 0, 0};
        AscendC::DataCopyPad(oGm[b_start], sL, sm);
        AscendC::DataCopyExtParams sr{1, wlen - mainB, 0, 0, 0};
        AscendC::DataCopyPad(oGm[b_start + mainB / sizeof(DT)], sL[mainB / sizeof(DT)], sr);
    } else {
        AscendC::DataCopyExtParams sc{1, wlen, 0, 0, 0};
        AscendC::DataCopyPad(oGm[b_start], sL, sc);
    }
}
#else
AscendC::DataCopyExtParams sc{1, static_cast<uint32_t>(n * sizeof(DT)), 0, 0, 0};
AscendC::DataCopyPad(oGm[b_start], sL, sc);
#endif
    }

    __aicore__ inline void ProcessLargeDirect(GM_ADDR personBoxes, GM_ADDR anchorPoints,
                                             GM_ADDR threshold_gm, GM_ADDR violationScore,
                                             uint32_t total_b, float waist_ratio, uint32_t block_dim) {
        const uint32_t per = (total_b + block_dim - 1u) / (block_dim > 0u ? block_dim : 1u);
        const uint32_t b_start = AscendC::GetBlockIdx() * per;
        uint32_t b_end = b_start + per;
        if (b_end > total_b) b_end = total_b;
        if (b_start >= b_end) return;
        const uint32_t n = b_end - b_start;

        AscendC::GlobalTensor<DT> bGm, aGm, oGm;
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(personBoxes));
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(anchorPoints));
        oGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(violationScore));

        AscendC::LocalTensor<DT> bL(AscendC::TPosition::VECCALC, 0, 4096);
        AscendC::LocalTensor<DT> aL(AscendC::TPosition::VECCALC, 4096, 2048);
        AscendC::LocalTensor<DT> sL(AscendC::TPosition::VECCALC, 6144, 1024);

        LargeDirectCtx ctx;
        LargeDirectLoad(bGm, aGm, bL, aL, b_start, n, threshold_gm, ctx);
        LargeDirectCompute(bL, aL, sL, ctx.n8, ctx.th, waist_ratio);
        LargeDirectStore(oGm, sL, b_start, n);
    }


    // ── W15_SMALL_CASE：小点单 tile 向量路径（B≤33 整档泛化，core-0-only）──────────
    //   TILE=128 编译期：无 tile 循环/无 colP/无双缓冲（单 tile 无 RAW/WAR）/无尾块分支
    //   （tv=ceil(n/LANE)*LANE 即现役尾块语义，垃圾 lane 不写出）。阈值/wr 运行时读。
    // 小点向量主循环（从 ProcessSmallVec 拆出；W26/W34 分支语义逐字保留）
    template <typename CT2, typename DT2>
    __aicore__ inline void SmallVecSegLoop(AscendC::LocalTensor<CT2> y1v, AscendC::LocalTensor<CT2> y2v,
                                           AscendC::LocalTensor<CT2> cyv, AscendC::LocalTensor<CT2> lim,
                                           AscendC::LocalTensor<CT2> acc, AscendC::LocalTensor<CT2> oneC,
                                           AscendC::LocalTensor<CT2> zeroC, AscendC::LocalTensor<DT2> gAllS,
                                           AscendC::LocalTensor<uint8_t> maskU8,
                                           float waist_ratio, float th, uint32_t tv) {
for (uint32_t seg = 0; seg < tv; seg += SEGW) {
    AscendC::LocalTensor<uint8_t> ms = maskU8[(seg / SEGW) * MSK_STRIDE];
#if W26_WR_SHORTCUT
    // W30-b2：小点 seg 循环运行期值特化——wr=1.0 时 lim≡y2+th（W26 板测
    //   已验证的塌缩语义，始终保留 +th 含 0.0f，Sub/Muls/Add 三条省略）。
    if (waist_ratio == 1.0f) {
        AscendC::Adds(acc[seg], y2v[seg], static_cast<CT>(th), SEGW);
        AscendC::Compare(ms, cyv[seg], acc[seg], AscendC::CMPMODE::GT, SEGW);
    } else
#endif
    {
#if W34_AXPY_FUSE
        // W34：通用链 4→3 条——Axpy(y1v, d, wr) 单指令完成 Muls+Add
        //   （dst += scalar*src，硬件 vaxpy）。y1v 被改写为 lim，输出 Select
        //   换独立槽 outV（fp32 档原写 y1v，槽布局已扩 OUT_BYTES）。
        AscendC::Sub(lim[seg], y2v[seg], y1v[seg], SEGW);
        AscendC::Axpy(y1v[seg], lim[seg], static_cast<CT>(waist_ratio), SEGW);
        AscendC::Adds(acc[seg], y1v[seg], static_cast<CT>(th), SEGW);
        AscendC::Compare(ms, cyv[seg], acc[seg], AscendC::CMPMODE::GT, SEGW);
#else
        AscendC::Sub(lim[seg], y2v[seg], y1v[seg], SEGW);
        AscendC::Muls(acc[seg], lim[seg], static_cast<CT>(waist_ratio), SEGW);
        AscendC::Add(lim[seg], y1v[seg], acc[seg], SEGW);
        AscendC::Adds(acc[seg], lim[seg], static_cast<CT>(th), SEGW);
        AscendC::Compare(ms, cyv[seg], acc[seg], AscendC::CMPMODE::GT, SEGW);
#endif
    }
    if constexpr (F16) {
        AscendC::Select(gAllS[seg], ms, oneH, zeroH,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, SEGW);
    } else {
#if W34_AXPY_FUSE
        AscendC::Select(outV[seg], ms, oneC, zeroC,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, SEGW);
#else
        AscendC::Select(y1v[seg], ms, oneC, zeroC,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, SEGW);
#endif
    }
}    }


    // 小点向量：DMA + threshold + 表 + 常量 + Gather/cast（从 ProcessSmallVec 拆出）
    template <bool F16v, typename DT2, typename CT2>
    __aicore__ inline float SmallVecPrepare(AscendC::GlobalTensor<DT2> &bGm, AscendC::GlobalTensor<DT2> &aGm,
                                            AscendC::LocalTensor<DT2> &bTile, AscendC::LocalTensor<DT2> &aTile,
                                            AscendC::LocalTensor<DT2> &gAllS,
                                            AscendC::LocalTensor<int32_t> &y1Off, AscendC::LocalTensor<int32_t> &cyOff,
                                            AscendC::LocalTensor<uint32_t> &y1OffU, AscendC::LocalTensor<uint32_t> &cyOffU,
                                            AscendC::LocalTensor<CT2> &y1v, AscendC::LocalTensor<CT2> &lim,
                                            AscendC::LocalTensor<CT2> &oneC, AscendC::LocalTensor<CT2> &zeroC,
                                            AscendC::LocalTensor<DT2> &oneH, AscendC::LocalTensor<DT2> &zeroH,
                                            uint32_t TILE, uint32_t DT_BYTES_v, uint32_t LANE_v,
                                            GM_ADDR threshold_gm, uint32_t n) {
        (void)DT_BYTES_v; (void)LANE_v;
AscendC::DataCopyExtParams bc{1, b_len, 0, 0, 0};
AscendC::DataCopyPadExtParams<DT> bp{false, 0, 0, 0};
AscendC::DataCopyPad(bTile, bGm[0], bc, bp);
AscendC::DataCopyExtParams ac{1, a_len, 0, 0, 0};
AscendC::DataCopyPadExtParams<DT> ap{false, 0, 0, 0};
AscendC::DataCopyPad(aTile, aGm[0], ac, ap);

float th = 0.0f;   // threshold 运行时读（红线）
if (threshold_ != nullptr) {
    const __gm__ DT* thp = reinterpret_cast<__gm__ DT*>(threshold_);
    th = static_cast<float>(thp[0]);
}

// offset 表：与现役同式（绝对字节偏移）
#if W30_SMALL_DMA_TBL
// W30-a：小点表构造链 DMA 化——SB_TBL_SMALL 两段表 [y1|cy]（TILE=128，
//   y2 靠 Gather srcBaseAddr=ROW_B/2 复用 y1 段，与现役同式）。
//   删 SetValue 播种 16 条 + S_V 对 + 倍增 Adds 14 条 ≈ scalar -250ns。
//   DMA 挂 bTile/aTile 同 MTE2 队列保序，尾端 MTE2_V 覆盖。
{
    AscendC::GlobalTensor<int32_t> tblGm;
    tblGm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(F16 ? SB_TBL_SMALL_F16 : SB_TBL_SMALL_F32));
    AscendC::DataCopyExtParams tcp{1, 2u * TILE * sizeof(int32_t), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> tpp{false, 0, 0, 0};
    // sim stub 校验目标声明尺寸：借 2*TILE 全视图（OFF_Y1 起，恰好覆盖
    //   y1Off+cyOff 两张表槽，与中点 W20 all3Off 同款手法）。
    AscendC::LocalTensor<int32_t> all2Off(AscendC::TPosition::VECCALC, OFF_Y1, 2 * TILE);
    AscendC::DataCopyPad(all2Off, tblGm[0], tcp, tpp);
}
#else
GatherVecSeedOffsets(y1Off, cyOff, 0, TILE, ROW_B, ROW_A, DT_BYTES);
#endif
if constexpr (F16) {
    AscendC::Duplicate(oneH, static_cast<DT>(1.0f), SEGW);
    AscendC::Duplicate(zeroH, static_cast<DT>(0.0f), SEGW);
} else {
    AscendC::Duplicate(oneC, static_cast<CT>(1.0f), SEGW);
    AscendC::Duplicate(zeroC, static_cast<CT>(0.0f), SEGW);
}

const uint32_t tv = ((n + LANE - 1) / LANE) * LANE;
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::Gather(gAllS, bTile, y1OffU, static_cast<uint32_t>(0),
                static_cast<uint64_t>(LANE), static_cast<uint8_t>(tv / LANE),
                static_cast<uint16_t>(8));
AscendC::Gather(gAllS[TILE], bTile, y1OffU, static_cast<uint32_t>(ROW_B / 2),
                static_cast<uint64_t>(LANE), static_cast<uint8_t>(tv / LANE),
                static_cast<uint16_t>(8));
AscendC::Gather(gAllS[2 * TILE], aTile, cyOffU, static_cast<uint32_t>(0),
                static_cast<uint64_t>(LANE), static_cast<uint8_t>(tv / LANE),
                static_cast<uint16_t>(8));
if constexpr (F16) {
    constexpr uint32_t WIDE_REP = 3u * TILE / 64u;   // =6
    AscendC::LocalTensor<DT> gAll(AscendC::TPosition::VECCALC, GATHER0, 3 * TILE);
    AscendC::LocalTensor<CT> cAll(AscendC::TPosition::VECCALC, CT_Y1, 3 * TILE);
    uint64_t fullMask[2] = {~0ull, ~0ull};
    AscendC::UnaryRepeatParams upP{1, 1, static_cast<uint8_t>(8), static_cast<uint8_t>(4)};
    AscendC::Cast(cAll, gAll, AscendC::RoundMode::CAST_NONE, fullMask,
                  static_cast<uint8_t>(WIDE_REP), upP);
}
    }


    __aicore__ inline void ProcessSmallVec(GM_ADDR personBoxes, GM_ADDR anchorPoints,
                                         GM_ADDR threshold_gm, GM_ADDR violationScore,
                                         uint32_t total_b, float waist_ratio, uint32_t block_dim) {

        AscendC::GlobalTensor<DT> bGm, aGm, oGm;
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(personBoxes));
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(anchorPoints));
        oGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(violationScore));

        AscendC::LocalTensor<DT> bTile(AscendC::TPosition::VECCALC, 0, TILE * 4);
        AscendC::LocalTensor<DT> aTile(AscendC::TPosition::VECCALC, B_TILE_BYTES, TILE * 2);
        AscendC::LocalTensor<int32_t> y1Off(AscendC::TPosition::VECCALC, OFF_Y1, TILE);
        AscendC::LocalTensor<int32_t> cyOff(AscendC::TPosition::VECCALC, OFF_CY, TILE);
        AscendC::LocalTensor<DT> gAllS(AscendC::TPosition::VECCALC, GATHER0, 3 * TILE);  // [y1|y2|cy] 单张目标
        AscendC::LocalTensor<CT> y1v(AscendC::TPosition::VECCALC, CT_Y1, TILE);
        AscendC::LocalTensor<CT> y2v(AscendC::TPosition::VECCALC, CT_Y1 + CT_VEC_BYTES, TILE);
        AscendC::LocalTensor<CT> cyv(AscendC::TPosition::VECCALC, CT_Y1 + 2u * CT_VEC_BYTES, TILE);
        AscendC::LocalTensor<CT> lim(AscendC::TPosition::VECCALC, CT_LIM, TILE);
        AscendC::LocalTensor<CT> acc(AscendC::TPosition::VECCALC, CT_ACC, TILE);
        AscendC::LocalTensor<CT> oneC(AscendC::TPosition::VECCALC, CNST_ONE, SEGW);
        AscendC::LocalTensor<CT> zeroC(AscendC::TPosition::VECCALC, CNST_ZERO, SEGW);
        AscendC::LocalTensor<DT> oneH(AscendC::TPosition::VECCALC, CNST_ONE, SEGW);
        AscendC::LocalTensor<DT> zeroH(AscendC::TPosition::VECCALC, CNST_ZERO, SEGW);
        AscendC::LocalTensor<uint8_t> maskU8(AscendC::TPosition::VECCALC,
#if W34_AXPY_FUSE
            CT_OUT + CT_VEC_BYTES,   // W34：mask 槽后移，让位给 fp32 outV
#else
            CNST_ZERO + CNST_BYTES + 128,
#endif
            MSK_STRIDE);
#if W34_AXPY_FUSE
        AscendC::LocalTensor<CT> outV(AscendC::TPosition::VECCALC, CT_OUT, TILE);  // W34：fp32 输出独立槽
#endif
        AscendC::LocalTensor<uint32_t> y1OffU = y1Off.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<uint32_t> cyOffU = cyOff.ReinterpretCast<uint32_t>();

        const uint32_t n = total_b;   // 单核全量（core-0-only）
        uint32_t b_len = n * 4 * sizeof(DT);
        uint32_t a_len = n * 2 * sizeof(DT);
        if (b_len < 32) b_len = 32;    // 现役小点同款 32B 合规下限
        if (a_len < 32) a_len = 32;
        const uint32_t tv = ((total_b + LANE - 1) / LANE) * LANE;
        const float th = SmallVecPrepare<F16, DT, CT>(bGm, aGm, bTile, aTile, gAllS,
                                            y1Off, cyOff, y1OffU, cyOffU, y1v, lim,
                                            oneC, zeroC, oneH, zeroH,
                                            TILE, DT_BYTES, LANE, threshold_, n);
        SmallVecSegLoop<CT, DT>(y1v, y2v, cyv, lim, acc, oneC, zeroC,
                                gAllS, maskU8, waist_ratio, th, tv);
        SmallVecWriteOut<F16, DT, CT>(oGm, outV, gAllS, n);
    }

    // ── W46：中点扁平化（8 核各自跑 ProcessSmallVec 同构扁平路径）─────────────────
    //   依据：小点（1 核×32 行，扁平结构）aiv=500ns vs 中点（8 核×32 行，GatherVec 框架）aiv=1018ns
    //   ——同样的每核工作量，框架机械（prologue 双缓冲/W23/evId 轮转/tile 循环/W38 选择器）吃掉 500ns/核。
    //   本函数 = ProcessSmallVec 去掉 core-0 早退 + 按核切行（bGm[b_start*4]/oGm[b_start]），
    //   表用 SB_TBL_SMALL（W30-a 形态 DMA；中点现役本就有一笔表 DMA，等量替换不减反增项）。
    //   覆盖：B∈(33,1024] 8 核 n=8~128 行 ≤ TILE=128 ✓；DataCopyPad 短写沿小点板测先例。
    // 中点主循环（从 ProcessMidFlat 拆出）
    template <typename CT2, typename DT2>
    __aicore__ inline void MidFlatSegLoop(AscendC::LocalTensor<CT2> y1v, AscendC::LocalTensor<CT2> y2v,
                                          AscendC::LocalTensor<CT2> cyv, AscendC::LocalTensor<CT2> lim,
                                          AscendC::LocalTensor<CT2> acc, AscendC::LocalTensor<CT2> oneC,
                                          AscendC::LocalTensor<CT2> zeroC, AscendC::LocalTensor<DT2> gAllS,
                                          AscendC::LocalTensor<uint8_t> maskU8,
                                          float waist_ratio, float th, uint32_t tv) {
for (uint32_t seg = 0; seg < tv; seg += SEGW) {
    AscendC::LocalTensor<uint8_t> ms = maskU8[(seg / SEGW) * MSK_STRIDE];
#if W26_WR_SHORTCUT
    if (waist_ratio == 1.0f) {
        AscendC::Adds(acc[seg], y2v[seg], static_cast<CT>(th), SEGW);
        AscendC::Compare(ms, cyv[seg], acc[seg], AscendC::CMPMODE::GT, SEGW);
    } else
#endif
    {
#if W34_AXPY_FUSE
        AscendC::Sub(lim[seg], y2v[seg], y1v[seg], SEGW);
        AscendC::Axpy(y1v[seg], lim[seg], static_cast<CT>(waist_ratio), SEGW);
        AscendC::Adds(acc[seg], y1v[seg], static_cast<CT>(th), SEGW);
        AscendC::Compare(ms, cyv[seg], acc[seg], AscendC::CMPMODE::GT, SEGW);
#else
        AscendC::Sub(lim[seg], y2v[seg], y1v[seg], SEGW);
        AscendC::Muls(acc[seg], lim[seg], static_cast<CT>(waist_ratio), SEGW);
        AscendC::Add(lim[seg], y1v[seg], acc[seg], SEGW);
        AscendC::Adds(acc[seg], lim[seg], static_cast<CT>(th), SEGW);
        AscendC::Compare(ms, cyv[seg], acc[seg], AscendC::CMPMODE::GT, SEGW);
#endif
    }
    if constexpr (F16) {
        AscendC::Select(gAllS[seg], ms, oneH, zeroH,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, SEGW);
    } else {
#if W34_AXPY_FUSE
        AscendC::Select(outV[seg], ms, oneC, zeroC,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, SEGW);
#else
        AscendC::Select(y1v[seg], ms, oneC, zeroC,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, SEGW);
#endif
    }
}    }


    // 中点：写出（从 ProcessMidFlat 拆出）
    template <bool F16v, typename DT2, typename CT2>
    __aicore__ inline void MidFlatWriteOut(AscendC::GlobalTensor<DT2> &oGm,
                                           AscendC::LocalTensor<CT2> &outV, AscendC::LocalTensor<DT2> &gAllS,
                                           uint32_t b_start, uint32_t n) {
AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
#if W34_AXPY_FUSE
if constexpr (!F16) {
    AscendC::DataCopyExtParams ocp{1, static_cast<uint32_t>(n * DT_BYTES), 0, 0, 0};
    AscendC::DataCopyPad(oGm[b_start], outV, ocp);
} else
#endif
{
    AscendC::DataCopyExtParams ocp{1, static_cast<uint32_t>(n * DT_BYTES), 0, 0, 0};
    AscendC::DataCopyPad(oGm[b_start], gAllS, ocp);
}    }


    // 中点：offset 表构造（从 ProcessMidFlat 拆出）
    template <bool F16v>
    __aicore__ inline void MidFlatBuildOffsets(AscendC::LocalTensor<int32_t> &y1Off, AscendC::LocalTensor<int32_t> &cyOff,
                                               uint32_t TILE, uint32_t ROW_B, uint32_t ROW_A, uint32_t DT_BYTES) {
#if W46_MID_TBL_DMA
{
    AscendC::GlobalTensor<int32_t> tblGm;
    tblGm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(F16 ? SB_TBL_SMALL_F16 : SB_TBL_SMALL_F32));
    AscendC::DataCopyExtParams tcp{1, 2u * TILE * sizeof(int32_t), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> tpp{false, 0, 0, 0};
    AscendC::LocalTensor<int32_t> all2Off(AscendC::TPosition::VECCALC, OFF_Y1, 2 * TILE);
    AscendC::DataCopyPad(all2Off, tblGm[0], tcp, tpp);
}
#else
GatherVecSeedOffsets(y1Off, cyOff, 0, TILE, ROW_B, ROW_A, DT_BYTES);
#endif    }


    // 中点：MTE2_V 等待 + 三列 Gather + fp16 升位（从 ProcessMidFlat 拆出）
    template <bool F16v, typename DT2, typename CT2>
    __aicore__ inline void MidFlatGather(AscendC::LocalTensor<DT2> &bTile, AscendC::LocalTensor<DT2> &aTile,
                                         AscendC::LocalTensor<DT2> &gAllS,
                                         AscendC::LocalTensor<uint32_t> &y1OffU, AscendC::LocalTensor<uint32_t> &cyOffU,
                                         uint32_t TILE, uint32_t LANE, uint32_t ROW_B, uint32_t tv) {
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
AscendC::Gather(gAllS, bTile, y1OffU, static_cast<uint32_t>(0),
                static_cast<uint64_t>(LANE), static_cast<uint8_t>(tv / LANE),
                static_cast<uint16_t>(8));
AscendC::Gather(gAllS[TILE], bTile, y1OffU, static_cast<uint32_t>(ROW_B / 2),
                static_cast<uint64_t>(LANE), static_cast<uint8_t>(tv / LANE),
                static_cast<uint16_t>(8));
AscendC::Gather(gAllS[2 * TILE], aTile, cyOffU, static_cast<uint32_t>(0),
                static_cast<uint64_t>(LANE), static_cast<uint8_t>(tv / LANE),
                static_cast<uint16_t>(8));
if constexpr (F16) {
    constexpr uint32_t WIDE_REP = 3u * TILE / 64u;
    AscendC::LocalTensor<DT> gAll(AscendC::TPosition::VECCALC, GATHER0, 3 * TILE);
    AscendC::LocalTensor<CT> cAll(AscendC::TPosition::VECCALC, CT_Y1, 3 * TILE);
    uint64_t fullMask[2] = {~0ull, ~0ull};
    AscendC::UnaryRepeatParams upP{1, 1, static_cast<uint8_t>(8), static_cast<uint8_t>(4)};
    AscendC::Cast(cAll, gAll, AscendC::RoundMode::CAST_NONE, fullMask,
                  static_cast<uint8_t>(WIDE_REP), upP);
    }
    }


    // 中点：DMA + threshold + 常量播种（从 ProcessMidFlat 拆出）
    template <bool F16v, typename DT2, typename CT2>
    __aicore__ inline float MidFlatLoadSeed(AscendC::GlobalTensor<DT2> &bGm, AscendC::GlobalTensor<DT2> &aGm,
                                            AscendC::LocalTensor<DT2> &bTile, AscendC::LocalTensor<DT2> &aTile,
                                            AscendC::LocalTensor<CT2> &oneC, AscendC::LocalTensor<CT2> &zeroC,
                                            AscendC::LocalTensor<DT2> &oneH, AscendC::LocalTensor<DT2> &zeroH,
                                            uint32_t b_start, uint32_t n, uint32_t SEGW, GM_ADDR threshold_gm) {
uint32_t b_len = n * 4 * sizeof(DT);
uint32_t a_len = n * 2 * sizeof(DT);
if (b_len < 32) b_len = 32;
if (a_len < 32) a_len = 32;
AscendC::DataCopyExtParams bc{1, b_len, 0, 0, 0};
AscendC::DataCopyPadExtParams<DT> bp{false, 0, 0, 0};
AscendC::DataCopyPad(bTile, bGm[b_start * 4], bc, bp);
AscendC::DataCopyExtParams ac{1, a_len, 0, 0, 0};
AscendC::DataCopyPadExtParams<DT> ap{false, 0, 0, 0};
AscendC::DataCopyPad(aTile, aGm[b_start * 2], ac, ap);

// threshold 读（紧随 DMA 发射，停等被飞行遮蔽）
float th = 0.0f;
if (threshold_gm != nullptr) {
    const __gm__ DT* thp = reinterpret_cast<__gm__ DT*>(threshold_gm);
    th = static_cast<float>(thp[0]);
}

// 表：SB_TBL_SMALL 两段 [y1|cy] DMA（W30-a 形态；y2 靠 srcBaseAddr=ROW_B/2 复用）
MidFlatBuildOffsets<F16>(y1Off, cyOff, TILE, ROW_B, ROW_A, DT_BYTES);
if constexpr (F16) {
    AscendC::Duplicate(oneH, static_cast<DT>(1.0f), SEGW);
    AscendC::Duplicate(zeroH, static_cast<DT>(0.0f), SEGW);
} else {
    AscendC::Duplicate(oneC, static_cast<CT>(1.0f), SEGW);
    AscendC::Duplicate(zeroC, static_cast<CT>(0.0f), SEGW);
}

    }


    __aicore__ inline void ProcessMidFlat(GM_ADDR personBoxes, GM_ADDR anchorPoints,
                                         GM_ADDR threshold_gm, GM_ADDR violationScore,
                                         uint32_t total_b, float waist_ratio, uint32_t block_dim) {

        // 按核切行（与 host per=align8(ceil(B/8)) 同式）
        const uint32_t core_id = AscendC::GetBlockIdx();
        uint32_t raw = (block_dim == 8u) ? ((total_b + 7u) >> 3)
                                         : ((total_b + block_dim - 1u) / block_dim);
        uint32_t per = ((raw + 7u) >> 3) << 3;
        if (per == 0) per = 8u;
        const uint32_t b_start = core_id * per;
        uint32_t b_end = b_start + per;
        if (b_end > total_b) b_end = total_b;
        if (b_start >= b_end) { return; }   // 空核（尾批）
        const uint32_t n = b_end - b_start;

        AscendC::GlobalTensor<DT> bGm, aGm, oGm;
        bGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(personBoxes));
        aGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(anchorPoints));
        oGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(violationScore));

        AscendC::LocalTensor<DT> bTile(AscendC::TPosition::VECCALC, 0, TILE * 4);
        AscendC::LocalTensor<DT> aTile(AscendC::TPosition::VECCALC, B_TILE_BYTES, TILE * 2);
        AscendC::LocalTensor<int32_t> y1Off(AscendC::TPosition::VECCALC, OFF_Y1, TILE);
        AscendC::LocalTensor<int32_t> cyOff(AscendC::TPosition::VECCALC, OFF_CY, TILE);
        AscendC::LocalTensor<DT> gAllS(AscendC::TPosition::VECCALC, GATHER0, 3 * TILE);
        AscendC::LocalTensor<CT> y1v(AscendC::TPosition::VECCALC, CT_Y1, TILE);
        AscendC::LocalTensor<CT> y2v(AscendC::TPosition::VECCALC, CT_Y1 + CT_VEC_BYTES, TILE);
        AscendC::LocalTensor<CT> cyv(AscendC::TPosition::VECCALC, CT_Y1 + 2u * CT_VEC_BYTES, TILE);
        AscendC::LocalTensor<CT> lim(AscendC::TPosition::VECCALC, CT_LIM, TILE);
        AscendC::LocalTensor<CT> acc(AscendC::TPosition::VECCALC, CT_ACC, TILE);
        AscendC::LocalTensor<CT> oneC(AscendC::TPosition::VECCALC, CNST_ONE, SEGW);
        AscendC::LocalTensor<CT> zeroC(AscendC::TPosition::VECCALC, CNST_ZERO, SEGW);
        AscendC::LocalTensor<DT> oneH(AscendC::TPosition::VECCALC, CNST_ONE, SEGW);
        AscendC::LocalTensor<DT> zeroH(AscendC::TPosition::VECCALC, CNST_ZERO, SEGW);
        AscendC::LocalTensor<uint8_t> maskU8(AscendC::TPosition::VECCALC,
#if W34_AXPY_FUSE
            CT_OUT + CT_VEC_BYTES,
#else
            CNST_ZERO + CNST_BYTES + 128,
#endif
            MSK_STRIDE);
#if W34_AXPY_FUSE
        AscendC::LocalTensor<CT> outV(AscendC::TPosition::VECCALC, CT_OUT, TILE);
#endif
        AscendC::LocalTensor<uint32_t> y1OffU = y1Off.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<uint32_t> cyOffU = cyOff.ReinterpretCast<uint32_t>();

        // 输入 DMA（各核自己的行区间）
        const float th = MidFlatLoadSeed<F16, DT, CT>(bGm, aGm, bTile, aTile, oneC, zeroC,
                                                      oneH, zeroH, b_start, n, SEGW, threshold_gm);
        const uint32_t tv = ((n + LANE - 1) / LANE) * LANE;
        MidFlatGather<F16, DT, CT>(bTile, aTile, gAllS, y1OffU, cyOffU,
                                   TILE, LANE, ROW_B, tv);
        MidFlatSegLoop<CT, DT>(y1v, y2v, cyv, lim, acc, oneC, zeroC,
                               gAllS, maskU8, waist_ratio, th, tv);
        MidFlatWriteOut<F16, DT, CT>(oGm, outV, gAllS, b_start, n);
    }

    // ── 大点：B > 1024，TILE=256 双缓冲流水 gather-vec（W11 定态）─────────────────
    // GatherVec tile 主循环（从 ProcessGatherVec 拆出；常量块同式重声明）
    // GatherVec：tile 内算术+门控+写回（从 GatherVecTileLoop 拆出）
    template <bool F16v, typename DT2, typename CT2>
    // GatherVec SEGW seg 循环（从 GatherVecTileArith 拆出）
    template <bool F16v, typename DT2, typename CT2>
    __aicore__ inline void GatherVecSegF32(AscendC::LocalTensor<CT2> &y1vB, AscendC::LocalTensor<CT2> &y2vB,
                                           AscendC::LocalTensor<CT2> &cyvB, AscendC::LocalTensor<DT2> &gy1B,
                                           AscendC::LocalTensor<uint8_t> &maskU8,
                                           float waist_ratio, float th, uint32_t MSK_STRIDE,
                                           uint32_t SEGW, uint32_t tv) {
    for (uint32_t seg = 0; seg < tv; seg += SEGW) {
AscendC::LocalTensor<uint8_t> ms = maskU8[(seg / SEGW) * MSK_STRIDE];
#if W26_WR_SHORTCUT
// W30-b：中/大点活跃 seg 循环 wr=1.0 特化——lim≡y2+th（W26 板测
//   塌缩语义，含 th=0.0f），Sub/Muls/Add 三条省略，非 1.0 走通用链。
if (waist_ratio == 1.0f) {
    AscendC::Adds(acc[seg], y2vB[seg], static_cast<CT>(th), SEGW);
    AscendC::Compare(ms, cyvB[seg], acc[seg], AscendC::CMPMODE::GT, SEGW);
} else
#endif
{
    AscendC::Sub(lim[seg], y2vB[seg], y1vB[seg], SEGW);
    AscendC::Muls(acc[seg], lim[seg], static_cast<CT>(waist_ratio), SEGW);
    AscendC::Add(lim[seg], y1vB[seg], acc[seg], SEGW);
    AscendC::Adds(acc[seg], lim[seg], static_cast<CT>(th), SEGW);
    AscendC::Compare(ms, cyvB[seg], acc[seg], AscendC::CMPMODE::GT, SEGW);
}
if constexpr (F16) {
    // S16：fp32 mask 直接驱动 half Select，结果就地落 gy1（省降位 Cast）
    AscendC::Select(gy1B[seg], ms, oneH, zeroH,
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, SEGW);
} else {
    AscendC::Select(y1vB[seg], ms, oneC, zeroC,
            AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, SEGW);
}
    }    }


    __aicore__ inline void GatherVecTileArith(
        AscendC::LocalTensor<DT2> &bTileB, AscendC::LocalTensor<DT2> &aTileB,
        AscendC::LocalTensor<DT2> &gy1B, AscendC::LocalTensor<DT2> &gy2B, AscendC::LocalTensor<DT2> &gycB,
        AscendC::LocalTensor<CT2> &y1vB, AscendC::LocalTensor<CT2> &y2vB, AscendC::LocalTensor<CT2> &cyvB,
        AscendC::LocalTensor<CT2> &oneC, AscendC::LocalTensor<CT2> &zeroC,
        AscendC::LocalTensor<DT2> &oneH, AscendC::LocalTensor<DT2> &zeroH,
        AscendC::LocalTensor<uint8_t> &maskU8,
        AscendC::LocalTensor<uint32_t> &y1OffU, AscendC::LocalTensor<uint32_t> &cyOffU,
        uint32_t colBase, uint32_t colCTBase, uint32_t ROW1_BASE, uint32_t COL1_BASE,
        uint32_t GATHER0, uint32_t CT_Y1, uint32_t TILE, uint32_t LANE,
        uint32_t MSK_STRIDE, uint32_t SEGW, uint32_t DT_VEC_BYTES, uint32_t CT_VEC_BYTES,
        float th, uint32_t &colP, uint32_t n, uint32_t b_start, uint32_t off) {
    for (uint32_t seg = 0; seg < tv; seg += LANE * 8u) {
const uint32_t left = tv - seg;
const uint32_t reps = (left / LANE < 8u) ? (left / LANE) : 8u;
const uint8_t rep = static_cast<uint8_t>(reps);
AscendC::Gather(gy1B[seg], bTileB, y1OffU[seg], static_cast<uint32_t>(0),
                static_cast<uint64_t>(LANE), rep, static_cast<uint16_t>(8));
// S53/T1：y2 = 同一张 y1 表 + 字节基址 ROW_B/2（y1/y2 偏移差恒 2*DT）
AscendC::Gather(gy2B[seg], bTileB, y1OffU[seg], static_cast<uint32_t>(ROW_B / 2),
                static_cast<uint64_t>(LANE), rep, static_cast<uint16_t>(8));
AscendC::Gather(gycB[seg], aTileB, cyOffU[seg], static_cast<uint32_t>(0),
                static_cast<uint64_t>(LANE), rep, static_cast<uint16_t>(8));
    }

#if !DIAG_NO_ARITH
    if constexpr (F16) {
// S16：升位 Cast 收成 1 条宽 Cast（3*TILE/64 repeats），half→fp32 无损
// W25：中点紧凑 = 3*tv/64（fp16 tv=128 → 6 rep vs 24）
#if W25_COMPACT_MID
const uint32_t WIDE_REP = (n <= TILE) ? (3u * tv / 64u) : (3u * TILE / 64u);
#else
#if W38_COMPACT_TBL
// W40：Cast 按 n 分叉——中点 3*tv/64（运行期），大点恢复 constexpr 满块
//   （W38 期大点 +0.3 嫌疑：WIDE_REP 运行期化阻碍指令折叠）。
AscendC::LocalTensor<DT> gAll(AscendC::TPosition::VECCALC, colBase, 3 * TILE);
AscendC::LocalTensor<CT> cAll(AscendC::TPosition::VECCALC, CT_Y1, 3 * TILE);
uint64_t fullMask[2] = {~0ull, ~0ull};
AscendC::UnaryRepeatParams upP{1, 1, static_cast<uint8_t>(8), static_cast<uint8_t>(4)};
if (n <= TILE) {
    AscendC::Cast(cAll, gAll, AscendC::RoundMode::CAST_NONE, fullMask,
                  static_cast<uint8_t>(3u * tv / 64u), upP);
} else {
    constexpr uint32_t WIDE_REP_FULL = 3u * TILE / 64u;
    AscendC::Cast(cAll, gAll, AscendC::RoundMode::CAST_NONE, fullMask,
                  static_cast<uint8_t>(WIDE_REP_FULL), upP);
}
#else
constexpr uint32_t WIDE_REP = 3u * TILE / 64u;
AscendC::LocalTensor<DT> gAll(AscendC::TPosition::VECCALC, colBase, 3 * TILE);
AscendC::LocalTensor<CT> cAll(AscendC::TPosition::VECCALC, CT_Y1, 3 * TILE);
uint64_t fullMask[2] = {~0ull, ~0ull};
AscendC::UnaryRepeatParams upP{1, 1, static_cast<uint8_t>(8), static_cast<uint8_t>(4)};
AscendC::Cast(cAll, gAll, AscendC::RoundMode::CAST_NONE, fullMask,
              static_cast<uint8_t>(WIDE_REP), upP);
#endif
#endif
    }
    // S31/G-1：官方结合顺序 (y1 + wr*d) + th
#if W25_COMPACT_MID
    // W25：紧凑布局下 Cast 输出连续 [y1c(tv)|y2c(tv)|cyc(tv)]，算术段用运行期
    //   视窗替代编译期 CT_Y1/CT_Y2/CT_CY 别名（tv=TILE 时与原别名重合，等价）。
    AscendC::LocalTensor<CT> cAllV(AscendC::TPosition::VECCALC, colCTBase, 3 * TILE);
    const uint32_t tvCT = tv;   // CT 段长（fp16 升位后 CT=float，长度=tv 项）
    AscendC::LocalTensor<CT> y1vC = cAllV;
    AscendC::LocalTensor<CT> y2vC = cAllV[tvCT];
    AscendC::LocalTensor<CT> cyvC = cAllV[2u * tvCT];
    // 段粒度 = LANE（fp32 64 / fp16 128，均 ≥64 无 NOP），循环 tv/LANE 轮；
    // maskU8 按 (seg/LANE)*MSK_PER 排布——W25 用独立游标避免与满块布局混用。
    const uint32_t W25_SEG = LANE;
    // mask 槽容量 = 1 段（TILE/SEGW=1）：每段 Compare 后 Select 立即消费，
    //   各段复用 maskU8[0]（V 队列内顺序保序，无跨段保留需求）。
    for (uint32_t seg = 0; seg < tv; seg += W25_SEG) {
AscendC::LocalTensor<uint8_t> ms = maskU8;
AscendC::Sub(lim[seg], y2vC[seg], y1vC[seg], W25_SEG);
AscendC::Muls(acc[seg], lim[seg], static_cast<CT>(waist_ratio), W25_SEG);
AscendC::Add(lim[seg], y1vC[seg], acc[seg], W25_SEG);
AscendC::Adds(acc[seg], lim[seg], static_cast<CT>(th), W25_SEG);
AscendC::Compare(ms, cyvC[seg], acc[seg], AscendC::CMPMODE::GT, W25_SEG);
if constexpr (F16) {
    AscendC::Select(gy1B[seg], ms, oneH, zeroH,
                    AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, W25_SEG);
} else {
    AscendC::Select(cAllV[seg], ms, oneC, zeroC,
                    AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, W25_SEG);
}
    }
#else
#if W38_COMPACT_TBL
    if (n <= TILE) {
// W38 中点算术段：紧凑 Cast 输出 [y1c(tv)|y2c(tv)|cyc(tv)] 连续区，
//   tv 视窗别名；段宽 LANE。fp32 输出落 cAllV[y1 段] = colCTBase =
//   GATHER0（与写出源 gy1B 同址）。
AscendC::LocalTensor<CT> cAllV(AscendC::TPosition::VECCALC, colCTBase, 3 * TILE);
AscendC::LocalTensor<CT> y1vC = cAllV;
AscendC::LocalTensor<CT> y2vC = cAllV[tv];
AscendC::LocalTensor<CT> cyvC = cAllV[2u * tv];
const uint32_t W38_SEG = LANE;
for (uint32_t seg = 0; seg < tv; seg += W38_SEG) {
    AscendC::LocalTensor<uint8_t> ms = maskU8;
#if W26_WR_SHORTCUT
    // W47：W38 中点段补 wr=1.0 特化——通用链走 Axpy 融合（W34 同式
    //   4→3 条），wr=1.0 走 Adds 直达（Sub/Muls/Add 全省，W26 塌缩语义）。
    if (waist_ratio == 1.0f) {
        AscendC::Adds(acc[seg], y2vC[seg], static_cast<CT>(th), W38_SEG);
        AscendC::Compare(ms, cyvC[seg], acc[seg], AscendC::CMPMODE::GT, W38_SEG);
    } else
#endif
    {
#if W34_AXPY_FUSE
        AscendC::Sub(lim[seg], y2vC[seg], y1vC[seg], W38_SEG);
        AscendC::Axpy(y1vC[seg], lim[seg], static_cast<CT>(waist_ratio), W38_SEG);
        AscendC::Adds(acc[seg], y1vC[seg], static_cast<CT>(th), W38_SEG);
        AscendC::Compare(ms, cyvC[seg], acc[seg], AscendC::CMPMODE::GT, W38_SEG);
#else
        AscendC::Sub(lim[seg], y2vC[seg], y1vC[seg], W38_SEG);
        AscendC::Muls(acc[seg], lim[seg], static_cast<CT>(waist_ratio), W38_SEG);
        AscendC::Add(lim[seg], y1vC[seg], acc[seg], W38_SEG);
        AscendC::Adds(acc[seg], lim[seg], static_cast<CT>(th), W38_SEG);
        AscendC::Compare(ms, cyvC[seg], acc[seg], AscendC::CMPMODE::GT, W38_SEG);
#endif
    }
    if constexpr (F16) {
        AscendC::Select(gy1B[seg], ms, oneH, zeroH,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, W38_SEG);
    } else {
        AscendC::Select(cAllV[seg], ms, oneC, zeroC,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, W38_SEG);
    }
}
    } else
#endif
        GatherVecSegF32<F16, DT, CT>(y1vB, y2vB, cyvB, gy1B, maskU8,
                                     waist_ratio, th, MSK_STRIDE, SEGW, tv);
#endif  // W25_COMPACT_MID
#endif  // !DIAG_NO_ARITH
#if W23_PIPELINE
    // W23：V_MTE3 按 bufferIndex 分 id（Set/Wait 紧邻仍需——本树写出在
    //   Wait 后立即执行，双缓冲下与 tile1 的 Gather 无槽冲突）。
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evId);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evId);
#elif W10_FINE_EVENT
    // 屏障 3: V -> MTE3 定向事件（Select 落地后才允许写出 DMA）
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
#else
    AscendC::PipeBarrier<PIPE_ALL>();    }


    __aicore__ inline void GatherVecTileLoop(
        AscendC::LocalTensor<uint32_t> &allOffU, AscendC::LocalTensor<DT> &gy1,
        AscendC::LocalTensor<CT> &y1v, AscendC::LocalTensor<CT> &lim, AscendC::LocalTensor<CT> &acc,
        AscendC::LocalTensor<CT> &oneC, AscendC::LocalTensor<DT> &oneH, AscendC::LocalTensor<DT> &zeroH,
        AscendC::LocalTensor<uint8_t> &maskU8, AscendC::LocalTensor<uint32_t> &y1OffU,
        AscendC::LocalTensor<uint32_t> &cyOffU,
        float th, uint32_t &colP, uint32_t n, uint32_t b_start) {
    constexpr uint32_t TILE = SB_TILE;
    constexpr uint32_t SEGW = TILE;                    // 算术段宽=TILE（单 tile 单段，count=256）
    constexpr uint32_t DT_VEC_BYTES = TILE * DT_BYTES; // 每根 Gather 目标向量
    constexpr uint32_t CT_VEC_BYTES = TILE * CT_BYTES; // 每根算术向量
    constexpr uint32_t MSK_STRIDE = MSK_NBLK * MASK_PER_SEG;
    constexpr uint32_t B_TILE_BYTES = TILE * ROW_B;
    constexpr uint32_t GATHER0 = OFF_CY + TBL_BYTES;
    constexpr uint32_t GATHER0 = OFF_CY + TBL_BYTES;
    constexpr uint32_t CT_Y1 = GATHER0 + GV_BYTES;     // fp32 时与 GATHER0 同址
    constexpr uint32_t COL_BYTES = 3u * (F16 ? DT_VEC_BYTES : CT_VEC_BYTES);
    constexpr uint32_t ROW1_BASE = COL1_BASE + COL_BYTES;
for (uint32_t off = 0; off < n; off += TILE) {
    const uint32_t t = (n - off < TILE) ? (n - off) : TILE;
    const uint32_t tv = ((t + LANE - 1) / LANE) * LANE;
#if W10_DUAL_BUF
    const uint32_t colBase = colP ? COL1_BASE : GATHER0;
    const uint32_t rowBase = colP ? ROW1_BASE : 0u;
#else
    const uint32_t colBase = GATHER0;
    const uint32_t rowBase = 0u;
#endif
    const uint32_t colCTBase = F16 ? CT_Y1 : colBase;  // fp16 算术槽固定 CT_Y1
    AscendC::LocalTensor<DT> gy1B(AscendC::TPosition::VECCALC, colBase, TILE);
    AscendC::LocalTensor<DT> gy2B(AscendC::TPosition::VECCALC, colBase + DT_VEC_BYTES, TILE);
    AscendC::LocalTensor<DT> gycB(AscendC::TPosition::VECCALC, colBase + 2 * DT_VEC_BYTES, TILE);
    AscendC::LocalTensor<CT> y1vB(AscendC::TPosition::VECCALC, colCTBase, TILE);
    AscendC::LocalTensor<CT> y2vB(AscendC::TPosition::VECCALC, colCTBase + CT_VEC_BYTES, TILE);
    AscendC::LocalTensor<CT> cyvB(AscendC::TPosition::VECCALC, colCTBase + 2 * CT_VEC_BYTES, TILE);
    // W16 修复：GM 下 cy 项经 bTileB 基址读锚区（偏移至 B_TILE+A_TILE 字节），
    //   CPU stub 的 Gather 源窗口 = src0.GetSize()（kernel_operator_vec_gather_
    //   intf_impl.h L65 实证）——声明扩到 6*TILE 元素覆盖整个行区，越窗读零根除。
    AscendC::LocalTensor<DT> bTileB(AscendC::TPosition::VECCALC, rowBase, TILE * 6);
    AscendC::LocalTensor<DT> aTileB(AscendC::TPosition::VECCALC, rowBase + B_TILE_BYTES, TILE * 2);

#if W15_PROLOGUE
    if (off != 0u)   // 首 tile 已由 prologue 发出
#endif
    {
#if W17_EVENIZE
        // W17：行 DMA 长度对齐 32B——align32(t*ROW_B) ≤ B_TILE（两 dtype 的
        // B_TILE 均为 32 倍数，t ≤ TILE）⇒ 恰不越槽；GM 读越界 ≤31B。
        AscendC::DataCopyExtParams bc{1, (t * ROW_B + 31u) & ~31u, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT> bp{false, 0, 0, 0};
        AscendC::DataCopyPad(bTileB, bGm[(b_start + off) * 4], bc, bp);
        AscendC::DataCopyExtParams ac{1, (t * ROW_A + 31u) & ~31u, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT> ap{false, 0, 0, 0};
        AscendC::DataCopyPad(aTileB, aGm[(b_start + off) * 2], ac, ap);
#else
        AscendC::DataCopyExtParams bc{1, static_cast<uint32_t>(t * ROW_B), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT> bp{false, 0, 0, 0};
        AscendC::DataCopyPad(bTileB, bGm[(b_start + off) * 4], bc, bp);
        AscendC::DataCopyExtParams ac{1, static_cast<uint32_t>(t * ROW_A), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT> ap{false, 0, 0, 0};
        AscendC::DataCopyPad(aTileB, aGm[(b_start + off) * 2], ac, ap);
#endif
    }
#if W23_PIPELINE
    // W23：事件 id 按 bufferIndex 轮转（Scale 同款 SetScaleCycleMte2V）——
    //   prologue 已为两 tile 各 Set(id=0/1)，循环内只 Wait；off==1 的行 DMA
    //   已由 prologue 发出（跳过重发）。
    const uint32_t evId = colP;
    // W23 编排：tile0/tile1 的 DMA+Set 已由 prologue 完成（off==0/256 只 Wait）；
    //   >2 tile 形状（B>8192 等）循环内对 tile>=2 走现役发射+Set（同 id 轮转）。
    if (off >= 2u * TILE) {
        AscendC::DataCopyExtParams bc{1, static_cast<uint32_t>(t * ROW_B), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT> bp{false, 0, 0, 0};
        AscendC::DataCopyPad(bTileB, bGm[(b_start + off) * 4], bc, bp);
        AscendC::DataCopyExtParams ac{1, static_cast<uint32_t>(t * ROW_A), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT> ap{false, 0, 0, 0};
        AscendC::DataCopyPad(aTileB, aGm[(b_start + off) * 2], ac, ap);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evId);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evId);
#elif W10_FINE_EVENT
    // 屏障 1（行DMA MTE2 -> Gather V）定向事件；跨 tile 复用由双缓冲消除
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
#else
    AscendC::PipeBarrier<PIPE_ALL>();
#endif

    // Gather：单次调用 reps 个满 repeat 段（fp32 4 / fp16 2），≤255
#if !DIAG_NO_GATHER
#if W15_GATHER_MERGE
    // W15/W16 合一表修复：恒按 3*TILE 满块消费——表布局是 [y1|y2|cy] 三个
    //   TILE 整块，按 3*tv 消费在 tv<TILE 时读不到 cy 段（B=1025 fp32 tv=192
    //   首发失败实证）。满块下三段位置恒对，尾 tile 垃圾 lane 不落地。
    //   fp16 3*256/128=6 rep 单调用；fp32 12 rep 分 8+4 两段（≤8/调用口径）。
    //   W25：中点（n≤TILE 且表已前缀化）按 3*tv 紧凑消费——省 ~70% 向量 rep；
    //   表布局 [y1|y2|cy] 逐段排布，3*tv 消费时 cy 段起于 tv 而非 TILE——
    //   所以紧凑消费的表必须也紧凑生成。表构造处 W25 同步改。
    AscendC::LocalTensor<DT> gAllB(AscendC::TPosition::VECCALC, colBase, 3 * TILE);
#if W25_COMPACT_MID
    if (n <= TILE) {
        // 紧凑：单 tile（中点），表 = [y1(0..tv) | y2(0..tv) | cy(0..tv)] 前缀布局
        //（W25 表构造见下），消费 3*tv，段调用仍按 ≤8rep 切分。
        for (uint32_t seg = 0; seg < 3u * tv; seg += LANE * 8u) {
            const uint32_t left = 3u * tv - seg;
            const uint32_t reps = (left / LANE < 8u) ? (left / LANE) : 8u;
            const uint8_t rep = static_cast<uint8_t>(reps);
            AscendC::Gather(gAllB[seg], bTileB, allOffU[seg], static_cast<uint32_t>(0),
                            static_cast<uint64_t>(LANE), rep, static_cast<uint16_t>(8));
        }
    } else
#endif
    if (n <= TILE) {
#if W38_COMPACT_TBL
        // W38：3*tv 紧凑消费（表已按 tv 选定）——fp32 12→3~6 rep / fp16 6→3 rep
        for (uint32_t seg = 0; seg < 3u * tv; seg += LANE * 8u) {
            const uint32_t left = 3u * tv - seg;
            const uint32_t reps = (left / LANE < 8u) ? (left / LANE) : 8u;
            const uint8_t rep = static_cast<uint8_t>(reps);
            AscendC::Gather(gAllB[seg], bTileB, allOffU[seg], static_cast<uint32_t>(0),
                            static_cast<uint64_t>(LANE), rep, static_cast<uint16_t>(8));
        }
    } else
#endif
    {
        for (uint32_t seg = 0; seg < 3u * TILE; seg += LANE * 8u) {
            const uint32_t left = 3u * TILE - seg;
            const uint32_t reps = (left / LANE < 8u) ? (left / LANE) : 8u;
            const uint8_t rep = static_cast<uint8_t>(reps);
            AscendC::Gather(gAllB[seg], bTileB, allOffU[seg], static_cast<uint32_t>(0),
                            static_cast<uint64_t>(LANE), rep, static_cast<uint16_t>(8));
        }
    }
#else
        GatherVecTileArith<F16, DT, CT>(bTileB, aTileB, gy1B, gy2B, gycB, y1vB, y2vB, cyvB,
            oneC, zeroC, oneH, zeroH, maskU8, y1OffU, cyOffU,
            colBase, colCTBase, ROW1_BASE, COL1_BASE, GATHER0, CT_Y1, TILE, LANE,
            MSK_STRIDE, SEGW, DT_VEC_BYTES, CT_VEC_BYTES, th, colP, n, b_start, off);
#endif

    // 只写真实 t 行；tile 尾部无效 lane 不落地
#if !DIAG_NO_WRITE
#if W18_WRITE_SPLIT
    {
        const uint32_t wlen = static_cast<uint32_t>(t * DT_BYTES);
        if (wlen >= 32u && (wlen & 31u) != 0u) {
            const uint32_t mainB = wlen & ~31u;
            AscendC::DataCopyExtParams om{1, mainB, 0, 0, 0};
            AscendC::DataCopyPad(oGm[b_start + off], gy1B, om);
            AscendC::DataCopyExtParams orr{1, wlen - mainB, 0, 0, 0};
            AscendC::DataCopyPad(oGm[b_start + off + mainB / DT_BYTES],
                                 gy1B[mainB / DT_BYTES], orr);
        } else {
            AscendC::DataCopyExtParams ocp{1, wlen, 0, 0, 0};
            AscendC::DataCopyPad(oGm[b_start + off], gy1B, ocp);
        }
    }
#else
    AscendC::DataCopyExtParams ocp{1, static_cast<uint32_t>(t * DT_BYTES), 0, 0, 0};
    AscendC::DataCopyPad(oGm[b_start + off], gy1B, ocp);
#endif
#endif  // !DIAG_NO_WRITE
#if W10_DUAL_BUF
    colP ^= 1u;
#endif
#endif  // !DIAG_NO_GATHER
#endif  // W23_PIPELINE
    }
    }



    // GatherVec：W21 变体表构造（从 GatherVecBuildTables 拆出）
    template <bool F16v, typename DT2, typename CT2>
    // GatherVec：prologue DMA + 表构造 + threshold 读（从 ProcessGatherVec 拆出）
    template <bool F16v, typename DT2, typename CT2>
    __aicore__ inline float GatherVecPrologue(
        AscendC::GlobalTensor<DT2> &bGm, AscendC::GlobalTensor<DT2> &aGm,
        AscendC::LocalTensor<DT2> &bTile, AscendC::LocalTensor<DT2> &aTile,
        AscendC::LocalTensor<DT2> &gAllS,
        AscendC::LocalTensor<int32_t> &y1Off, AscendC::LocalTensor<int32_t> &cyOff,
        AscendC::LocalTensor<int32_t> &y2OffM, AscendC::LocalTensor<int32_t> &all3Off,
        AscendC::LocalTensor<CT2> &oneC, AscendC::LocalTensor<CT2> &zeroC,
        AscendC::LocalTensor<DT2> &oneH, AscendC::LocalTensor<DT2> &zeroH,
        uint32_t TILE, uint32_t LANE, uint32_t ROW_B, uint32_t ROW_A,
        uint32_t B_TILE_BYTES, uint32_t ROW1_BASE, uint32_t SEGW,
        uint32_t DT_BYTES_v, uint32_t b_start, uint32_t n, GM_ADDR threshold_gm) {
#if W15_PROLOGUE
// W15_PROLOGUE：首 tile 两笔行 DMA 提前发出（colP==0 ⇒ 落 0 号行缓冲组），
//   MTE2 飞行遮住表构造；循环内 off==0 跳过重发，tile 头 MTE2_V 事件仍覆盖
//   prologue DMA（MTE2 队列排空语义）。
{
    const uint32_t t0 = (n < TILE) ? n : TILE;
    AscendC::DataCopyExtParams bc0{1, (W17_EVENIZE ? ((t0 * ROW_B + 31u) & ~31u)
                                                   : static_cast<uint32_t>(t0 * ROW_B)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<DT> bp0{false, 0, 0, 0};
    AscendC::DataCopyPad(bTile, bGm[b_start * 4], bc0, bp0);
    AscendC::DataCopyExtParams ac0{1, (W17_EVENIZE ? ((t0 * ROW_A + 31u) & ~31u)
                                                   : static_cast<uint32_t>(t0 * ROW_A)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<DT> ap0{false, 0, 0, 0};
    AscendC::DataCopyPad(aTile, aGm[b_start * 2], ac0, ap0);
#if W23_PIPELINE
    // W23：prologue 侧 Set(MTE2_V, id=0)——tile0 数据已发，循环内只 Wait。
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
#endif
}
#endif
#if W23_PIPELINE
// W23：tile1 行 DMA 同批发出（Scale warmup——双 tile 全填满，tile1 DMA 在
//   tile0 处理期间飞行，消除 tile 间搬运停顿）。双缓冲槽相位 1（ROW1_BASE）。
{
    const uint32_t t1 = (n > TILE) ? ((n - TILE < TILE) ? (n - TILE) : TILE) : 0;
    if (t1 != 0u) {
        AscendC::LocalTensor<DT> bTile1(AscendC::TPosition::VECCALC, ROW1_BASE, TILE * 6);
        AscendC::LocalTensor<DT> aTile1(AscendC::TPosition::VECCALC, ROW1_BASE + B_TILE_BYTES, TILE * 2);
        AscendC::DataCopyExtParams bc1{1, static_cast<uint32_t>(t1 * ROW_B), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT> bp1{false, 0, 0, 0};
        AscendC::DataCopyPad(bTile1, bGm[(b_start + TILE) * 4], bc1, bp1);
        AscendC::DataCopyExtParams ac1{1, static_cast<uint32_t>(t1 * ROW_A), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT> ap1{false, 0, 0, 0};
        AscendC::DataCopyPad(aTile1, aGm[(b_start + TILE) * 2], ac1, ap1);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
    }
}
#endif
#if W20_GM_CONST_TABLE
// W20-const：一笔 3KB DMA 取编译期常量表（与行 DMA 同 MTE2 队列保序）。
//   ⚠ W36 板测定案(6adf7328/6adf746a)：提首重排两发一致负收益（大点
//   +0.25~0.35，表 DMA 抢占 MTE2 队列头使首 tile 行 DMA 落地变晚）——
//   回退原位（prologue/tile1 行 DMA 之后），保持行 DMA 队头。
{
    AscendC::GlobalTensor<int32_t> tblGm;
#if W38_COMPACT_TBL
    // W38：中点单 tile 按 tv 选紧凑表（DMA 量与 Gather/Cast rep 同步缩减）；
    //   大点（n>TILE）保持满表满块。
    if (n <= TILE) {
        const uint32_t tvSel = ((n + LANE - 1) / LANE) * LANE;
        // fp32 tv∈{64,128,192,256}（B=1025~2048 时 per 核 n=136~255 → tv=192/256）；
        //   tv=256 用满表。fp16 tv∈{128,256}（LANE=128），256 用满表。
        // W38 消重：CMP 表前缀关系（64⊂128⊂192）已验证 → 统一取满表，DMA 按 3*tv 截断（逐位一致）
        const __gm__ int32_t* tbl =
            F16 ? (tvSel <= 128u ? SB_TBL_F16 : SB_TBL_F16)
                : (tvSel <= 192u ? SB_TBL_CMP_F32_192 : SB_TBL_F32);
        tblGm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(tbl));
        AscendC::DataCopyExtParams tcpC{1, static_cast<uint32_t>(3u * tvSel * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> tpp{false, 0, 0, 0};
        AscendC::DataCopyPad(all3Off, tblGm[0], tcpC, tpp);
    } else
#endif
    {
        tblGm.SetGlobalBuffer(const_cast<__gm__ int32_t*>(F16 ? SB_TBL_F16 : SB_TBL_F32));
        AscendC::DataCopyExtParams tcp{1, 3u * TILE * sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> tpp{false, 0, 0, 0};
        AscendC::DataCopyPad(all3Off, tblGm[0], tcp, tpp);
    }
}
#elif W21_TABLE_PRE
GatherVecBuildTablesW21<F16, DT, CT>(y1Off, cyOff, bTile, aTile, oneC, zeroC, oneH, zeroH,
// ---- offset 表：S53/T1 8 元素标量播种 + 倍增铺满 ----
#if W21_TABLE_PRE
// 已前置（W21_TABLE_PRE），原位跳过
#elif W15_GATHER_MERGE
GatherVecSeedOffsets(y1Off, cyOff, static_cast<int32_t>(B_TILE_BYTES), TILE, ROW_B, ROW_A, DT_BYTES);
// y2 段 = y1 表 + 2*DT（y1/y2 偏移差恒 2*DT），一条 Adds 铺满 —— 合一表第三段就位
AscendC::Adds(y2OffM, y1Off, static_cast<int32_t>(2 * DT_BYTES), static_cast<int32_t>(TILE));
#else
for (uint32_t k = 0; k < 8u; ++k) {
    y1Off.SetValue(k, static_cast<int32_t>(k * ROW_B + DT_BYTES));
    cyOff.SetValue(k, static_cast<int32_t>(k * ROW_A + DT_BYTES));
}
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
#if W16_MID_TABLE_DYN
// W16：倍增上界 = 本核所需最大 lane 数（ceil(n/LANE)*LANE，2 的幂块覆盖），
//   中点 n=32~40 省末尾 1~2 次倍增（2~4 条 Adds）；大点 n≥TILE 时与原式相同。
const uint32_t tblBound = ((n + LANE - 1) / LANE) * LANE;
const uint32_t tblEnd = (tblBound > TILE) ? TILE : tblBound;
for (uint32_t nb = 8u; nb < tblEnd; nb <<= 1) {
    AscendC::Adds(y1Off[nb], y1Off[0], static_cast<int32_t>(nb * ROW_B), static_cast<int32_t>(nb));
    AscendC::Adds(cyOff[nb], cyOff[0], static_cast<int32_t>(nb * ROW_A), static_cast<int32_t>(nb));
}
#else
for (uint32_t nb = 8u; nb < TILE; nb <<= 1) {
    AscendC::Adds(y1Off[nb], y1Off[0], static_cast<int32_t>(nb * ROW_B), static_cast<int32_t>(nb));
    AscendC::Adds(cyOff[nb], cyOff[0], static_cast<int32_t>(nb * ROW_A), static_cast<int32_t>(nb));
}
#endif
#endif
if constexpr (F16) {
    AscendC::Duplicate(oneH, static_cast<DT>(1.0f), SEGW);
    AscendC::Duplicate(zeroH, static_cast<DT>(0.0f), SEGW);
} else {
    AscendC::Duplicate(oneC, static_cast<CT>(1.0f), SEGW);
    AscendC::Duplicate(zeroC, static_cast<CT>(0.0f), SEGW);
}

// W30-b：threshold GM 标量读在此执行（三笔 DMA 已发出，读延迟被 MTE2
//   飞行遮蔽）；tile 循环的 wr 特化分支依赖 th 就位。
if (threshold_gm != nullptr) {
    const __gm__ DT* thp = reinterpret_cast<__gm__ DT*>(threshold_gm);
    th = static_cast<float>(thp[0]);
}

#if W10_DUAL_BUF    }

    // 表播种+倍增公共段（消重复代码；各变体语义逐字保留——cy 偏移差用 cyExtra 传参）
    __aicore__ inline void GatherVecSeedOffsets(AscendC::LocalTensor<int32_t> &y1Off, AscendC::LocalTensor<int32_t> &cyOff,
                                                int32_t cyExtra, uint32_t TILE, uint32_t ROW_B, uint32_t ROW_A,
                                                uint32_t DT_BYTES) {
        for (uint32_t k = 0; k < 8u; ++k) {
            y1Off.SetValue(k, static_cast<int32_t>(k * ROW_B + DT_BYTES));
            cyOff.SetValue(k, static_cast<int32_t>(k * ROW_A + DT_BYTES) + cyExtra);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
        for (uint32_t nb = 8u; nb < TILE; nb <<= 1) {
            AscendC::Adds(y1Off[nb], y1Off[0], static_cast<int32_t>(nb * ROW_B), static_cast<int32_t>(nb));
            AscendC::Adds(cyOff[nb], cyOff[0], static_cast<int32_t>(nb * ROW_A), static_cast<int32_t>(nb));
        }
    }


    __aicore__ inline void GatherVecBuildTablesW21(
        AscendC::LocalTensor<int32_t> &y1Off, AscendC::LocalTensor<int32_t> &cyOff,
        AscendC::LocalTensor<DT2> &bTile, AscendC::LocalTensor<DT2> &aTile,
        AscendC::LocalTensor<CT2> &oneC, AscendC::LocalTensor<CT2> &zeroC,
        AscendC::LocalTensor<DT2> &oneH, AscendC::LocalTensor<DT2> &zeroH,
        uint32_t TILE, uint32_t ROW_B, uint32_t ROW_A, uint32_t DT_BYTES_v,
        uint32_t B_TILE_BYTES, uint32_t ROW1_BASE, uint32_t SEGW, uint32_t n) {
// W21 修正版：表构造链在 prologue/tile1 DMA 发射之后执行——标量链与 MTE2 飞行真正并行（原位置在 DMA 前，串行无重叠，W21 首板中点无改善的真因）。
#if W15_GATHER_MERGE
#if W25_COMPACT_MID
if (n <= TILE) {
    // W25 紧凑表：单视图 all3Off 三段 [y1(tv)|y2(tv)|cy(tv)]，段长=tv（运行期）。
    const uint32_t tvC = ((n + LANE - 1) / LANE) * LANE;
    // all3Off 声明 3*TILE 长（槽容量覆盖 tv≤TILE），消费端 3*tv 对齐。
    AscendC::LocalTensor<int32_t> c1 = all3Off;                          // y1 段 [0, tvC)
    AscendC::LocalTensor<int32_t> c2 = all3Off[tvC];                     // y2 段 [tvC, 2tvC)
    AscendC::LocalTensor<int32_t> c3 = all3Off[2u * tvC];                // cy 段 [2tvC, 3tvC)
    for (uint32_t k = 0; k < 8u; ++k) {
        c1.SetValue(k, static_cast<int32_t>(k * ROW_B + DT_BYTES));
        c2.SetValue(k, static_cast<int32_t>(k * ROW_B + 3u * DT_BYTES));
        c3.SetValue(k, static_cast<int32_t>(k * ROW_A + DT_BYTES + static_cast<int32_t>(B_TILE_BYTES)));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    for (uint32_t nb = 8u; nb < tvC; nb <<= 1) {
        AscendC::Adds(c1[nb], c1[0], static_cast<int32_t>(nb * ROW_B), static_cast<int32_t>(nb));
        AscendC::Adds(c2[nb], c2[0], static_cast<int32_t>(nb * ROW_B), static_cast<int32_t>(nb));
        AscendC::Adds(c3[nb], c3[0], static_cast<int32_t>(nb * ROW_A), static_cast<int32_t>(nb));
    }
    // 尾段铺满到 tv（tv 可能 > 8 且非 2 幂对齐余项：倍增到 ≥tv 后停止，剩余项
    // 由最后一次 Adds 覆盖 count=nb 上限——倍增法覆盖 [8, 2*nb) 前缀，tv 为
    // LANE 倍数 ⇒ 恰好被 2 幂链覆盖到 ≥tv；多铺到 2^ceil 无害（c 段余量）。
} else
#endif
{
GatherVecSeedOffsets(y1Off, cyOff, static_cast<int32_t>(B_TILE_BYTES), TILE, ROW_B, ROW_A, DT_BYTES);
AscendC::Adds(y2OffM, y1Off, static_cast<int32_t>(2 * DT_BYTES), static_cast<int32_t>(TILE));
}
#else
for (uint32_t k = 0; k < 8u; ++k) {
    y1Off.SetValue(k, static_cast<int32_t>(k * ROW_B + DT_BYTES));
    cyOff.SetValue(k, static_cast<int32_t>(k * ROW_A + DT_BYTES));
}
AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
#if W16_MID_TABLE_DYN
const uint32_t tblBound = ((n + LANE - 1) / LANE) * LANE;
const uint32_t tblEnd = (tblBound > TILE) ? TILE : tblBound;
for (uint32_t nb = 8u; nb < tblEnd; nb <<= 1) {
    AscendC::Adds(y1Off[nb], y1Off[0], static_cast<int32_t>(nb * ROW_B), static_cast<int32_t>(nb));
    AscendC::Adds(cyOff[nb], cyOff[0], static_cast<int32_t>(nb * ROW_A), static_cast<int32_t>(nb));
}
#else
for (uint32_t nb = 8u; nb < TILE; nb <<= 1) {
    AscendC::Adds(y1Off[nb], y1Off[0], static_cast<int32_t>(nb * ROW_B), static_cast<int32_t>(nb));
    AscendC::Adds(cyOff[nb], cyOff[0], static_cast<int32_t>(nb * ROW_A), static_cast<int32_t>(nb));
}
#endif
#endif
#if W21_EARLY_CONST
#if !W21_EARLY_CONST
if constexpr (F16) {
    AscendC::Duplicate(oneH, static_cast<DT>(1.0f), SEGW);
    AscendC::Duplicate(zeroH, static_cast<DT>(0.0f), SEGW);
} else {
    AscendC::Duplicate(oneC, static_cast<CT>(1.0f), SEGW);
    AscendC::Duplicate(zeroC, static_cast<CT>(0.0f), SEGW);
}
#endif
#endif
#endif  // W21_TABLE_PRE



    }

    __aicore__ inline void ProcessGatherVec(GM_ADDR personBoxes, GM_ADDR anchorPoints,
                                          GM_ADDR threshold_gm, GM_ADDR violationScore,
                                          uint32_t total_b, float waist_ratio, uint32_t block_dim) {
#if DIAG_EMPTY_CORE
        // 诊断探针：大点档近似空核——各核立即退出，测该档 fork/launch 地板。WA 预期。
        return;
#endif

        AscendC::LocalTensor<DT> bTile(AscendC::TPosition::VECCALC, 0, TILE * 4);
        AscendC::LocalTensor<DT> aTile(AscendC::TPosition::VECCALC, B_TILE_BYTES, TILE * 2);
        AscendC::LocalTensor<int32_t> y1Off(AscendC::TPosition::VECCALC, OFF_Y1, TILE);
        AscendC::LocalTensor<int32_t> cyOff(AscendC::TPosition::VECCALC, OFF_CY, TILE);
#if W15_GATHER_MERGE
        AscendC::LocalTensor<int32_t> y2OffM(AscendC::TPosition::VECCALC, OFF_Y2, TILE);
        // 合一表整张视图（3*TILE 长；借 y1Off 声明会在 operator[] 处越界断言）
        AscendC::LocalTensor<int32_t> all3Off(AscendC::TPosition::VECCALC, OFF_Y1, 3 * TILE);
        AscendC::LocalTensor<uint32_t> allOffU = all3Off.ReinterpretCast<uint32_t>();
#endif
        AscendC::LocalTensor<DT> gy1(AscendC::TPosition::VECCALC, GATHER0, TILE);
        AscendC::LocalTensor<DT> gy2(AscendC::TPosition::VECCALC, GATHER0 + DT_VEC_BYTES, TILE);
        AscendC::LocalTensor<DT> gyc(AscendC::TPosition::VECCALC, GATHER0 + 2 * DT_VEC_BYTES, TILE);
        AscendC::LocalTensor<CT> y1v(AscendC::TPosition::VECCALC, CT_Y1, TILE);
        AscendC::LocalTensor<CT> y2v(AscendC::TPosition::VECCALC, CT_Y2, TILE);
        AscendC::LocalTensor<CT> cyv(AscendC::TPosition::VECCALC, CT_CY, TILE);
        AscendC::LocalTensor<CT> lim(AscendC::TPosition::VECCALC, CT_LIM, TILE);
        AscendC::LocalTensor<CT> acc(AscendC::TPosition::VECCALC, CT_ACC, TILE);
        AscendC::LocalTensor<CT> oneC(AscendC::TPosition::VECCALC, CNST_ONE, SEGW);
        AscendC::LocalTensor<CT> zeroC(AscendC::TPosition::VECCALC, CNST_ZERO, SEGW);
        AscendC::LocalTensor<DT> oneH(AscendC::TPosition::VECCALC, CNST_ONE, SEGW);
        AscendC::LocalTensor<DT> zeroH(AscendC::TPosition::VECCALC, CNST_ZERO, SEGW);
        AscendC::LocalTensor<uint8_t> maskU8(AscendC::TPosition::VECCALC, SLOT_MASK, MASK_BYTES);
        AscendC::LocalTensor<uint32_t> y1OffU = y1Off.ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<uint32_t> cyOffU = cyOff.ReinterpretCast<uint32_t>();




        const float th = GatherVecPrologue<F16, DT, CT>(bGm, aGm, bTile, aTile, gAllS,
            y1Off, cyOff, y2OffM, all3Off, oneC, zeroC, oneH, zeroH,
            TILE, LANE, ROW_B, ROW_A, B_TILE_BYTES, ROW1_BASE, SEGW,
            DT_BYTES, b_start, n, threshold_gm);
uint32_t colP = 0;   // 列/行槽双缓冲相位（逐 tile 翻转）
#endif
        GatherVecTileLoop(allOffU, gy1, y1v, lim, acc, oneC, oneH, zeroH,
                                maskU8, y1OffU, cyOffU, th, colP, n, b_start);
    }

private:
    GM_ADDR  personBoxes_{nullptr};
    GM_ADDR  anchorPoints_{nullptr};
    GM_ADDR  threshold_{nullptr};
    GM_ADDR  violationScore_{nullptr};
    uint32_t total_b_{0};
    uint32_t block_dim_{1};
    float    waist_ratio_{1.0f};
};

// ============================================================================
// 入口（压平 · TPL 绑定修复）：外壳回 W12 已验证 2 维 <DT, SCH_MODE>（SCH 值域
//   0/1，见 tiling_key 头），类内保持压平（运行期三档分派）。SCH_MODE 仅作键。
//   W1/P-B 形态保留：入口两笔 tiling 标量读同线发射（W11 定态）。
// ============================================================================
#if DIAG_FLOOR_AIC_MODE
// P-D1：TPL 多出 FORMAT 维（SB_MODE），AIC_ONLY 组按 AIC 核型编译——空核分支外的
//   V 代码经 if constexpr 丢弃，AIC 实例零 V 代码。
template <typename DT_PERSON_BOXES, uint8_t SB_MODE>
#else
template <typename DT_PERSON_BOXES, uint32_t SCH_MODE>
#endif
__global__ __aicore__ void safety_belt_violation_geometry(GM_ADDR person_boxes, GM_ADDR anchor_points,
                                                  GM_ADDR threshold, GM_ADDR violation_score,
                                                  GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SafetyBeltViolationGeometryTilingData);
#if !DIAG_FLOOR_AIC_MODE
    // W41（D1-K2）：显式声明 AIV_ONLY 核型——纯向量算子若被当 MIX 任务下发会连带
    //   拉起用不到的 Cube 核（fdu 案例 readme:27 "避免混合启动空转 Cube 核的头开销"）。
    //   我方从未显式声明过（旧档"真板默认已 AIV-only"是行为推断，非声明等价）。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
#if DIAG_FLOOR_AIC_MODE
    if constexpr (SB_MODE == SB_MODE_AIC_EMPTY) {
return;   // P-D1：AIC 空核地板（本实例按 AIC_ONLY 编译，输出全空 = 预期 WA）
    }
#endif
#if DIAG_FLOOR_MINIMAL
    return;   // P-C：纯框架地板（无 tiling 读/无类实例化/无档位分派）
#endif
#if DIAG_FLOOR_AIC_MODE
    (void)SB_MODE;   // P-D1：FORMAT 维键（0=NORM / 1=AIC_EMPTY）
#else
    (void)SCH_MODE;   // 仅作 TPL 键（0=fp32 组 / 1=fp16 组）
#endif
    uint32_t total_b = 1u;
    float waist = 1.0f;
#if W22_TILING_HIDE
    // W22：两读合一——一次 8B 读后位拆包（float 位模式原样搬运，逐位安全）。
    AscendC::GlobalTensor<uint64_t> tGm64;
    tGm64.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(tiling));
    const uint64_t tboth = tGm64.GetValue(0);
    total_b = static_cast<uint32_t>(tboth & 0xFFFFFFFFu);
    {
uint32_t wbits = static_cast<uint32_t>(tboth >> 32);
__builtin_memcpy(&waist, &wbits, 4);
    }
#elif !DIAG_FLOOR_NOTILING
    // W1/P-B：两笔 GM 标量读同处连续发射（第 2 读延迟被第 1 读遮住）
    AscendC::GlobalTensor<uint32_t> tGmU;
    tGmU.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tiling));
    AscendC::GlobalTensor<float> tGmF;
    tGmF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling));
    total_b = tGmU.GetValue(0);
    waist = tGmF.GetValue(1);
#endif
    (void)workspace;
    SafetyBelt<DT_PERSON_BOXES, SCH_MODE, true> op(person_boxes, anchor_points, threshold,
                           violation_score, total_b, waist);
    op.Start();
}