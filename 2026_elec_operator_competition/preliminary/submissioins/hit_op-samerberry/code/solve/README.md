# solve

提交团队: 南工大果
提交者: 刘家良


## 算子实现介绍

本算子是针对昇腾 910B4（Ascend C）深度定制的 LinalgSolve（求解线性方程组 $AX=B$，支持 FP32/BF16）高性能实现，通过 src_modular/ 模块化组织并由构建脚本内联组装为标准交付工程，采用编译期模板分派（SHAPE 分支）+ Tiling 参数位域压缩将各内核分支的指令体积严格控制在 AIV 16 KiB ICache 限制内，并按问题规模自适应分派至四类求解器：① TinySolver（极小规模单右端项，AIV_ONLY） 与 ② BatchSolver（中小规模多右端项，AIV_ONLY） 采用单核独立处理单 Batch 的零 Workspace 架构，将增广矩阵 $[A \mid B]$ 整体载入 UB 并利用 TransDataTo5HD 及 GatherMask 转换为列主序排布，结合启发式阈值快速主元判定与 WholeReduceMax 列选主元，直接通过底层 vgather（GatherNoMask）、vbrcb（FastBrcb）广播及 Axpy/MulAddDst 向量外积在片上完成 Gauss-Jordan 消元与行置换恢复；③ CoopGJ（中大规模矩阵多核协同路径） 采用每 Batch 8 核协同的 8 列面板（Panel）循环交错 Gauss-Jordan 消元架构，利用 GM 共享内存标志位配合 GetSystemCycle() 硬件周期退避实现核间无锁流水同步（面板分解 FactorTile 与尾矩阵更新 UpdatePanels 重叠执行），并在消元后协同完成分块残差计算与一步迭代精化（支持根据主元数量级跨度自适应启用基于 Dekker 双字拆分与动态指数缩放的高精度补偿残差计算，或对良态对角占优矩阵动态跳过精化）；④ Solver（通用兜底路径） 则面向任意合法维度、右端项广播及 BF16（自动升精度至 FP32 计算）场景，提供基于跨核全局同步（GlobalBarrier）的多核分块 Gauss-Jordan 消元与自适应残差迭代精化能力。
