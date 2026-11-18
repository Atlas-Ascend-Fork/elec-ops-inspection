# safetybeltviolationgeometry

提交团队: 致远队
提交者: Purs


## 算子实现介绍

Host 侧：只做形状读取、属性透传与 launch 分派，零数据运算。按 B 三档调度并写入 tiling，核数按档定死。
Kernel 侧：按 dtype×档位 编译期模板特化（TilingKey 6 变体，无运行期分支税）：
小点（B≤33）：单核单 tile 向量路径，无 tile 循环、无双缓冲，全链最短。
中/大点（B&gt;33）：8 核按行切分，TILE=256 双缓冲流水——行 DMA 提前发射（prologue+warmup），MTE2 飞行遮蔽表构造与 threshold 读。
核心技巧：
Gather 抽列：y1/y2/cy 三列经编译期常量偏移表（GM 表一笔 DMA）+ 向量 Gather 抽取，替代标量循环；
三表合一：三张偏移表合并为单张绝对偏移表，Gather 调用 3→1；
fp16 SPLIT：Gather 按 half 宽度（2× rep）执行，Cast 单条宽指令升位，算术全 fp32；
指令融合：waist_ratio=1.0 走塌缩快捷链（省 3 条 V）；通用链用 Axpy 硬件单指令 4→3 条；
紧凑消费：按本核实际行数 tv 选紧凑表/紧凑 rep，尾核不多算。
