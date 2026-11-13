# solve

提交团队：ruby0451 的队伍

提交者：ruby0451

## 算子实现

基于 Ascend C 实现 FP32 线性方程组 AX=B 求解，支持多右端项与题目约定的广播。按矩阵和右端项规模选择片上驻留、列分配及多核协作路径，并通过残差计算和迭代修正控制误差。

## 验证版本

本实现源自 CANNJudge 最终提交 #1171359。[原始获奖源码](https://gitcode.com/ruby0451/elec-ops-inspection/tree/archive/ruby0451-preliminary-award/2026_elec_operator_competition/preliminary/submissioins/zju_ruby0451/code/solve) 的提交为 `93129ff6f4a9753e27f6d13dd615db9a84f2ec4f`，七个工程文件逐一核对 SHA-256，与平台保存的源码一致；该版本于 2026-11-06 在 Ascend 910B、CANN 9.0 环境通过官方 12 个测试点。

当前版本补充 Apache-2.0 许可证头，按计算策略拆分头文件，提取重复的消元和数据处理步骤，并将长函数拆为加载、计算、同步及回写阶段。当前版本的编译与社区流水线结果见 [PR #115](https://gitcode.com/cann/elec-ops-inspection/merge_requests/115)。原始竞赛结果仅对应上述归档版本；尚无新版 NPU 精度或性能结论。

## 构建

在已配置 CANN 9.0 开发环境的 Linux 主机上，从本目录执行：

```bash
cmake -S . -B build -DASCEND_COMPUTE_UNIT=ascend910b
cmake --build build -j
```

工程使用 CANN 的 `ASC` CMake 包和 `npu_op_package`。如未自动发现工具链，先加载实际安装位置的 `set_env.sh`。官方测试运行由 CANNJudge 的题目测试框架负责；本目录不包含官方测试数据或独立测试程序。
