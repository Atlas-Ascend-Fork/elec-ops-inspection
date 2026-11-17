# Helton

## 参赛信息

- 队伍名称：Helton
- 提交账号：Helton
- 赛题：隔离开关关键点坐标和置信度分数融合解码算子
- 算子名称：`FusedDisconnectorKeypointSoftArgmax`
- CANNJudge 提交编号：**1348245**
- 提交时间：**2026-11-15 15:46:38（北京时间）**
- 评测状态：**Pass**
- 正式回执：[CANNJudge 1348245](https://cannjudge.cn/submission/6af9635e6bd0f9be37df08c6)

赛后题目榜于 2026-11-15 16:03:33（北京时间）的快照采用本次提交，记录为第 8 名、37.58 分。榜单分数和排名是该快照时的记录。

## 算子说明

本算子面向变电设备巡检中的隔离开关状态识别，将解码头输出的特征图在 Ascend NPU 上融合完成 3×3 卷积、全空间 Softmax 期望和 Soft-Argmax，输出铰链、刀尖及两侧静触头的坐标与热图分数，避免将完整概率图写回全局内存再读回。

实现使用 Ascend C 和一次 AIV kernel 发射。Host 根据 batch、空间尺寸和可用 AIV 核数选择整图或分带方案；Device 侧完成卷积、跨带统计合并及坐标还原。坐标采用半像素中心约定，温度参数使用实际输入的 `tau`。

源码和构建说明位于 [code/FusedDisconnectorKeypointSoftArgmax](code/FusedDisconnectorKeypointSoftArgmax/README.md)。工程目标为 `ascend910b`，已在 Ascend 910B3 / CANN 9.0.0 环境验证。

## 接口与支持范围

| 名称 | 类型 | dtype | shape / format | 说明 |
| --- | --- | --- | --- | --- |
| `feat` | 输入 | FP32 | `[B,32,H,W]`，ND | 连续 NCHW 语义的解码头特征 |
| `weight` | 输入 | FP32 | `[4,32,3,3]`，ND | 连续 OIHW 语义的互相关权重 |
| `bias` | 输入 | FP32 | `[4]`，ND | 四个输出通道的偏置 |
| `keypoints` | 输出 | FP32 | `[B,4,2]`，ND | 坐标顺序为 x、y |
| `scores` | 输出 | FP32 | `[B,4]`，ND | 各输出通道卷积热图的最大值 |
| `tau` | 必选属性 | float | 有限且大于 0 | Softmax 温度 |
| `stride` | 必选属性 | int | 仅支持 4 | 坐标还原步长 |

已验证 `(H,W)` 为 `(48,64)`、`(64,64)`、`(72,96)`，`B=1..4`。其他 dtype、layout、batch 或空间尺寸不在本次归档的支持范围内。

## 赛时评测记录

以下为提交 1348245 的八个公开测试点结果；耗时单位为微秒，精度比均为 1。

| Case | 状态 | 耗时（µs） |
| --- | --- | ---: |
| 1 | Pass | 17.57 |
| 2 | Pass | 17.68 |
| 3 | Pass | 24.31 |
| 4 | Pass | 25.89 |
| 5 | Pass | 5.73 |
| 6 | Pass | 20.27 |
| 7 | Pass | 23.49 |
| 8 | Pass | 28.11 |

八个公开点合计 163.054 µs。平台整体状态为 Pass，另外七个测试点仅返回 `Hidden`，本目录不推测其逐点成绩。

## 正确性与源码一致性

- 正式提交 1348245 的整体评测为 Pass，八个公开点均为 Pass。
- 同一正式 ACLNN 源码包在 Ascend 910B3 / CANN 9.0.0 上通过 15/15 个本地合成代理负载和 12/12 个自造正确性用例。这些用例不是官方测试数据，也不用于预测官方隐藏点成绩。
- 本地验证使用 FP64 卷积与期望计算、再舍入 FP32 的 CPU 参考实现，逐项比较标准为 `abs(actual-expected) <= 1e-4 + 1e-4*abs(expected)`。
- 本次开源归档保留正式工程的全部 12 个文件。当前本地源码、赛时 dry-run 记录、CANNJudge 回执及平台返回源码已核对一致；逐文件 SHA-256 见 [verification.json](verification.json)。
- 本次归档只增加队伍说明和核对记录，没有改动算子实现或构建文件。
