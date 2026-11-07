# xdInternet

## 参赛信息

- 队伍名称：尼古
- 身份：学生
- 学校：哈尔滨工业大学（深圳）
- 学校代码缩写：hitsz
- CANNJudge 用户：xdInternet

## 算子

- `LinalgEigh`：实对称矩阵特征分解，源码位于 `code/LinalgEigh/`。
- 代码对应 CANNJudge 提交 #1174206（v326）。
- 本实现使用 Ascend C，针对 CANN 9.0 和昇腾 910B 构建。

开发与优化过程中使用了 Gemini 与 OpenAI Codex 辅助。

## 实现思路

基于 Ascend C 的 Vector 多核并行实现实对称矩阵谱分解：

1. 利用 DataCopyPad 与 Gather 完成上下三角对称化加载及 2 的幂次动态缩放。
2. 利用 WholeReduceSum、两阶段 BlockReduceSum、Brcb 与 MulAddDst 实现向量化 Householder 三对角化及提前归约。
3. 采用四路展开的 Sturm 序列，结合割线法与逆二次插值求解特征值。
4. 采用移位逆迭代、近邻特征向量正交化和双反射子合并的 Householder 逆变换求解特征向量；通过分条带归约处理特征向量符号。

## Tiling 策略

矩阵行跨度 `stride` 按 32 B（8 个 float）对齐。`n <= 32` 使用小矩阵路径；在 AIV 核数满足条件时，`32 < n <= 120` 和 `120 < n <= 256` 分别选择对应的 UB 驻留模式，其余情况使用通用协作路径。Batch 在核组间分配，组内按 `vec_rows_per_lane` 分配特征对。
