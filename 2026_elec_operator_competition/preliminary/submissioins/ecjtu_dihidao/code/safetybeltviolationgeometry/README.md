# safetybeltviolationgeometry

提交团队: dihidao 的队伍
提交者: 冯柄志


## 算子实现介绍

本源码基于 Ascend C 实现安全带违规几何判定，采用 Host 侧 Tiling 与 Device 侧 Kernel 分工的结构。Host 侧负责按批量大小分配各核处理行数、设置分块参数，并传递腰部比例及可选阈值信息；Kernel 侧批量读取人体框和锚点，通过 GatherMask 提取 y1、y2、cy，使用 FP32 向量指令完成参考高度计算和严格大于比较，最终输出 0/1 违规标记。
实现支持 FP16、FP32 数据，采用静态 UB 地址布局和显式事件同步，减少缓冲管理开销；对齐数据使用 DataCopy 搬运，尾块使用 DataCopyPad 处理，并复用中间缓冲区。整个判定过程在单个 Kernel 内完成，无需将中间结果写回全局内存。
