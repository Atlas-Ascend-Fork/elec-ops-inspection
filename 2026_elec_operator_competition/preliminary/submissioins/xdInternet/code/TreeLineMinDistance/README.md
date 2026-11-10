# TreeLineMinDistance

2026 电力行业算子大赛初赛参赛工程，队伍名称为尼古，学校为哈尔滨工业大学（深圳），CANNJudge 用户为 `xdInternet`。开发与优化过程中使用了 Gemini 和 OpenAI Codex 辅助。

算子对每个批次中的树障点，计算其到该批次全部有限线段的最小欧氏距离，并返回最近线段的索引。输入 `tree_points` 的形状为 `[B, N, 3]`，`line_segments` 的形状为 `[B, M, 2, 3]`，两者均为连续的 float16 或 float32。输出距离为 `[B, N]`、类型与输入相同；输出索引为 `[B, N]` 的 int32。同距时题目要求返回最小线段索引，零长度线段按点处理。

工程使用 Ascend C，Host 根据输入形状和类型选择 Tiling 路径并分配任务，Kernel 计算点到线段的距离并归约最小值和索引。工程中包含针对不同输入规模的路径。

## 代码来源与验证状态

原始核心实现来自 v136，原始文件曾与 CANNJudge 提交 `#1177155` 的源码逐文件核对一致。该原始提交在判题平台的 17 个测试点全部通过（17/17 Pass）；该结果仅覆盖平台测试数据，不等于对全部合法输入的证明。

为满足仓库 PR `#101` 检查要求：
1. 分支提交 `c92ea958` 在四个 C++ 源文件开头添加了 Apache 2.0 许可证注释；
2. 分支提交 `0033df9e`（对应 CI `#209`）完成了 7 项严重除零告警与 73 项函数长度/圈复杂度/嵌套深度/重复代码/源文件长度重构，SCA 检查通过，CodeCheck 由 80 项降至 4 项一般告警（均为「超大头文件[C++]」，有效代码行 `> 500`）；
3. 当前版本在 `0033df9e` 基础上将上述 4 个超大头文件进一步拆分为职责内聚的子头文件，使全部 `.h` 文件的总行数与有效代码行数（`nloc`）均严格低于 `500` 行，同时保持 `#include` 顺序、类模板与内联成员定义可见性、类/函数行为与指令执行顺序完全不变（详见 `PR101_CI_AND_SCA_REPORT.md`）。

**重要说明**：添加 Apache 2.0 注释后的 PR 版本（`c92ea958`）、首次重构版本（`0033df9e`）以及本次二次拆分头文件后的版本**均未在 CANN 工具链下编译，未在 Ascend NPU 设备上运行测试，也未重新提交 CANNJudge 判题**，不能声称重构后版本已重新取得 `#1177155` 的判题成绩或设备实测性能。Gemini 报告完成静态指标、桩编译与 CPU 等价检查，并提供独立验证套件和日志；其中 28,560 组等价检查比较的是测试文件中独立整理的函数副本，不能证明整个算子实现等价。

## 构建

工程依赖 CANN Toolkit 9.0 和 CMake；默认计算单元为 `ascend910b`（Ascend 910B4）。在已配置好 CANN 环境后，可参考：

```bash
source /path/to/cann/set_env.sh
cmake -S . -B build -DCANNJUDGE_COMPILE_JOBS=4
cmake --build build --parallel 4 --target binary package
```

## 文件

- `CMakeLists.txt`：工程入口。
- `PR101_CI_AND_SCA_REPORT.md`：PR #101 CI CodeCheck 修复报告（含 CI #209 剩余 4 项超大头文件二次拆分说明）、SCA 记录说明与可复现验证指南。
- `op_host/`：算子注册和 Tiling（`tree_line_min_distance.cpp`）。
- `op_kernel/`：设备侧实现、模块化头文件与 Tiling 定义（全部 `.h` 文件有效代码行 `<= 473`、总行数 `<= 489`）：
  - `tree_line_min_distance.cpp`：Kernel 顶层入口与编译期/运行期 Mode 分发；
  - `tree_line_common.h`：公共内联向量算子辅助函数、安全整除/取模与规约解码工具；
  - `tree_line_mode3.h`：`TreeLineMode3Shared` 公共工具与 `Mode 3`（`KernelTreeLineMode3<T>`）实现；
  - `tree_line_static_batch_modes.h`：单点/单线段/多批次小规模静态 UB 实现（`Mode 6/7/8/9`）；
  - `tree_line_pair32.h`：`Pair32` 静态特化实现（`Mode 5/21`）；
  - `tree_line_specialized_static.h`：特定规模静态 UB 实现（`Mode 10/12`）；
  - `tree_line_general_modes.h`：通用双缓冲流水线类定义与调度框架（`Mode 0/1`）；
  - `tree_line_general_compute.h`：`KernelTreeLineMinDistance<T, MODE>` 的点块与线段块核心计算及归约类外内联成员实现；
  - `tree_line_prune.h`：FP16 筛选 + FP32 精确验证类定义、元数据表与初始化（`Mode 4/11`）；
  - `tree_line_prune_screen.h`：`KernelTreeLinePrune<T>` 的 FP16 缩放、分块粗筛与全量 FP32 回退类外内联成员实现；
  - `tree_line_prune_verify.h`：`KernelTreeLinePrune<T>` 的主/副候选解码、FP32 精确几何验证与平局归并类外内联成员实现；
  - `tree_line_small_modes.h`：极小规模路径（`Mode 2` 及 `Mode 16..20`）；
  - `tree_line_min_distance_tiling.h`、`tiling_key_tree_line_min_distance.h`：Tiling 结构体与模板参数选择宏。
