# fuseddisconnectorkptsoftargmax

提交团队: SZKT
提交者: 顾景超


## 算子实现介绍

FusedDisconnectorKeypointSoftArgmax 使用纯 AIV 完成 FP32 卷积和关键点解码。算子按空间分块并行，每次分组处理 8 个输入通道，通过双缓冲搬运带边界的输入块，供 4 个输出通道共享。卷积生成的热图块留在片上，直接计算局部最大值、稳定指数和坐标加权和。各核只将局部统计量写入紧凑的工作区，经一次跨核同步后完成全图合并，输出 Soft-Argmax 坐标和未缩放热图的最大值，无需将整张热图写回全局内存。
