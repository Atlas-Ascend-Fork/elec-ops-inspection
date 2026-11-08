# lstsq

提交团队: qunzhongjia的队伍
提交者: 黎何


## 算子实现介绍

基于 Ascend C 实现面向 Ascend 910B 的 LinalgLstsq 最小二乘求解算子（driver=gels），对列满秩且 m≥n 的输入矩阵完成最小二乘解与残差计算。Host 侧根据 batch、矩阵维度和 AI Core 数量制定 Tiling 策略，按 batch 分核，并结合 UB 容量选择片上驻留或 UB 与 Global Workspace 混合分块存储；大规模方阵采用 batch 级与矩阵列分区的两级并行。通用路径采用 Householder QR 分解与上三角回代，方阵路径采用向量化消元和分块 LU，结合主元归约、Gather 重排及迭代精化完成求解。UB 缓冲区与列存储步长按 32 字节对齐，通过 DataCopyPad、向量掩码及核间同步处理非对齐数据与多核协作。
