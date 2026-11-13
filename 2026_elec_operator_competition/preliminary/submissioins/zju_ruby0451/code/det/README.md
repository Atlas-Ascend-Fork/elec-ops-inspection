# det

提交团队：ruby0451 的队伍

提交者：ruby0451

## 算子实现

基于 Ascend C 实现 FP32 批量矩阵行列式计算。按矩阵规模采用片上驻留的小矩阵消元、列布局的中矩阵消元以及 Cube/Vector 协作的分块计算；Host 负责形状、任务分配与 workspace 规划。

## 验证版本

本实现源自 CANNJudge 最终提交 #1171264。[原始获奖源码](https://gitcode.com/ruby0451/elec-ops-inspection/tree/archive/ruby0451-preliminary-award/2026_elec_operator_competition/preliminary/submissioins/zju_ruby0451/code/det) 的提交为 `93129ff6f4a9753e27f6d13dd615db9a84f2ec4f`，七个工程文件逐一核对 SHA-256，与平台保存的源码一致；该版本于 2026-11-06 在 Ascend 910B、CANN 9.0 环境通过官方 12 个测试点。

当前版本补充 Apache-2.0 许可证头，按计算策略拆分头文件，提取重复的消元和数据处理步骤，并将长函数拆为加载、计算、同步及回写阶段，并增加调度参数的零值检查。当前版本的编译与社区流水线结果见 [PR #115](https://gitcode.com/cann/elec-ops-inspection/merge_requests/115)。原始竞赛结果仅对应上述归档版本；尚无新版 NPU 精度或性能结论。

## 构建

在已配置 CANN 9.0 开发环境的 Linux 主机上，从本目录执行：

```bash
cmake -S . -B build -DASCEND_COMPUTE_UNIT=ascend910b
cmake --build build -j
```

工程使用 CANN 的 `ASC` CMake 包和 `npu_op_package`。如未自动发现工具链，先加载实际安装位置的 `set_env.sh`。官方测试运行由 CANNJudge 的题目测试框架负责；本目录不包含官方测试数据或独立测试程序。
