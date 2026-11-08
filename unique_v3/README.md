# UniqueV3

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|

## 功能说明

- 算子功能：对输入的一维张量进行全局去重，返回排序后的唯一值序列、唯一值个数，以及可选的反向索引映射（inverse）和每个唯一值的出现次数（counts）。`flag_sorted=0` 时按降序输出，非 0 时按升序输出。

- 计算公式：

  设输入张量为 $x = [x_0, x_1, \ldots, x_{N-1}]$，共 $N$ 个元素。

  1. **output**：将 $x$ 中所有不同的值排序，得到 $output = [u_0, u_1, \ldots, u_{M-1}]$，$M$ 为唯一值个数。`flag_sorted=0` 时满足 $u_0 > u_1 > \ldots > u_{M-1}$；`flag_sorted \ne 0` 时满足 $u_0 < u_1 < \ldots < u_{M-1}$。

  2. **uniqueCnt**：$uniqueCnt = M$。

  3. **inverse**（可选）：对于每个 $i \in [0, N)$，$inverse[i] = j$，满足 $output[j] = x_i$。

  4. **counts**（可选）：对于每个 $j \in [0, M)$，$counts[j] = |\{i \mid x_i = output[j]\}|$，即输入中等于 $output[j]$ 的元素个数，满足 $\sum_{j=0}^{M-1} counts[j] = N$。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>待去重的一维张量。</td>
      <td>FLOAT、INT32、FLOAT16、BFLOAT16、INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>排序后的唯一值序列，前 uniqueCnt 个元素有效，shape 与 input 相同；排序方向由 flag_sorted 控制。</td>
      <td>FLOAT、INT32、FLOAT16、BFLOAT16、INT16（与 input 相同）</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>uniqueCnt</td>
      <td>输出</td>
      <td>唯一值的个数，INT32 Tensor，shape 为 [1]。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inverse</td>
      <td>输出（可选）</td>
      <td>反向索引映射，inverse[i] 表示 input[i] 在 output 中的索引位置，shape 与 input 相同。flag_inverse 非 0 时有效；为 0 时不计算。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counts</td>
      <td>输出（可选）</td>
      <td>每个唯一值的出现次数，counts[j] 表示 output[j] 在 input 中出现的次数，前 uniqueCnt 个元素有效。flag_counts 非 0 时有效；为 0 时不计算。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>flag_sorted</td>
      <td>属性（可选）</td>
      <td>唯一值的排序方向：0 表示降序，非 0 表示升序，默认值为 0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>flag_inverse</td>
      <td>属性（可选）</td>
      <td>是否计算反向索引输出 inverse：0 表示不计算，非 0 表示计算，默认值为 0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>flag_counts</td>
      <td>属性（可选）</td>
      <td>是否计算唯一值计数输出 counts：0 表示不计算，非 0 表示计算，默认值为 0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- input 仅支持 1 维张量，不支持多维输入。
- 不支持空张量（元素个数为 0 的输入）。
- input、output 的数据类型只支持 FLOAT、INT32、FLOAT16、BFLOAT16、INT16，且 output 与 input 数据类型一致；uniqueCnt、inverse、counts 数据类型固定为 INT32。数据格式只支持 ND。
- 所有输入类型在 kernel 内部都会转换为 FLOAT 进行排序。INT32 输入如需保持逐元素精确区分，建议限制在 FP32 可精确表示的整数范围 $[-2^{24}, 2^{24}]$ 内。
- kernel 使用 `3.402823e+38f`（接近 FLOAT 最大有限值）及其负值作为对齐和边界处理的哨兵值，FLOAT 输入应避免使用这两个值。
- 支持动态 shape，不支持动态 rank。
- inverse、counts 在 flag_inverse/flag_counts 为 0 时，对应输出 tensor 仍需传入，但输出内容无意义。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_unique_v3](./examples/test_aclnn_unique_v3.cpp) | 通过aclnnUniqueV3接口方式调用UniqueV3算子。 |
