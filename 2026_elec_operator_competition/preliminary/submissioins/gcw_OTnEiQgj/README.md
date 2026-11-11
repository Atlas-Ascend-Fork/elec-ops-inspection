# gcw_OTnEiQgj

提交团队: gcw_OTnEiQgj

## 算子: lstsq

基于 Ascend C 实现面向 Ascend 910B 的 LinalgLstsq 批量列满秩最小二乘求解算子（支持 FP32/BF16 输入输出与残差平方和计算）。Host 侧根据批次大小 B、矩阵维度 (M, N)、右端项列数 K、UB 32 字节对齐块步长及可用 AIV 核数规划 Workspace 并分派执行路径。

小规模方阵（N≤64）将增广矩阵按 32 字节对齐常驻片上 UB，按 batch 分核并行，采用基于可用行位掩码的原位部分主元 Gauss-Jordan 消元、尾块 8 行合并更新与 Gather 重排输出；中大规模方阵（64<N≤256）采用单核或多核协同波前面板 LU 分解，结合上下三角回代与一步全精度残差迭代精化提升数值精度；超定矩阵（M>N）采用向量化 Householder QR 分解、正交变换及上三角回代完成最小二乘解与各列残差平方和计算。
