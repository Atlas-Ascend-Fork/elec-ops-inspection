# qunzhongjia 的队伍

## 参赛信息

- 队伍名称：qunzhongjia 的队伍
- 提交账号：qunzhongjia
- 赛题：隔离开关关键点坐标和置信度分数融合解码算子
- 算子名称：`FusedDisconnectorKeypointSoftArgmax`
- CANNJudge 提交编号：**1347504**
- 提交时间：**2026-11-15 15:14:35（北京时间）**
- 评测状态：**Pass**

## 算子说明

使用 Ascend C 融合 3×3 卷积、全空间 Softmax 和 Soft-Argmax，输出关键点坐标及原始卷积热图最大值。输入包括 FP32 特征、卷积权重和偏置；坐标遵循半像素中心约定，Softmax 使用实际温度参数 tau。

Host 根据输入形状选择模板分支。主要卷积路径将权重保留在 UB，并在 UB 容量允许时预载完整的多通道输入条带；其余形状使用原有搬运流水和通用路径。

源码和构建说明位于 [code/FusedDisconnectorKeypointSoftArgmax](code/FusedDisconnectorKeypointSoftArgmax/README.md)。工程目标为 `ascend910b`，构建步骤见该目录 README。

## 赛时评测记录

以下为提交 1347504 的八个公开测试点耗时，单位为微秒。对应榜单截图分数为 38.80；分数是该截图时的记录，可能随榜单基准变化。

| Case | 状态 | 耗时（µs） |
| --- | --- | ---: |
| 1 | Pass | 20.33 |
| 2 | Pass | 20.47 |
| 3 | Pass | 23.75 |
| 4 | Pass | 25.41 |
| 5 | Pass | 3.95 |
| 6 | Pass | 21.24 |
| 7 | Pass | 25.06 |
| 8 | Pass | 28.05 |

平台整体状态为 Pass；隐藏测试点没有公开逐点明细，本目录不推测隐藏用例内容。

## 源码一致性

本次开源归档保留赛时工程的全部八个文件。四个可提交源码文件与当时上传内容、平台返回的提交源码逐一核对一致；对应 SHA-256 见 [verification.json](verification.json)。本次只增加队伍说明及核对记录，没有改动算子实现或构建文件。

上述 Pass 与耗时来自赛时 OJ 记录。本次归档没有重新进行 CANN 编译或 NPU 测试。
