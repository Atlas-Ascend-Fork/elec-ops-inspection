# GitCode PR #101 (`c92ea958` $\rightarrow$ `0033df9e` $\rightarrow$ v2) CI CodeCheck 修复与验证报告

- **关联 PR**：[cann/elec-ops-inspection#101](https://gitcode.com/cann/elec-ops-inspection/pull/101)
- **基准提交与首轮 CI**：
  - 基准提交 `c92ea958`（[Actions Run 8b825eb3379e457987598d2283e6b763](https://gitcode.com/cann/elec-ops-inspection/actions/runs/8b825eb3379e457987598d2283e6b763)，[CodeCheck 报告](https://www.openlibing.com/apps/entryCheckDashCode/MR_50492713628c4ad98915b27614d42549/2e0e861c429266cf1e709b123d877ffb?projectId=300033&codeHostingPlatformFlag=gitcode)）：CodeCheck 共 80 项告警（7 项严重除零、73 项一般问题），SCA 2 项待审。
  - 首轮修复提交 `0033df9e`（[CI #209 Actions Run a53e5f2e61c942ad8220687d8e839e50](https://gitcode.com/CANN/elec-ops-inspection/actions/runs/a53e5f2e61c942ad8220687d8e839e50)，[CodeCheck 报告](https://www.openlibing.com/apps/entryCheckDashCode/MR_cfb9f57f82914e369a3f89cb24160c24/774428d4886e09cd1d5d6015fddad97e?projectId=300033&codeHostingPlatformFlag=gitcode)）：**SCA 通过**；CodeCheck 由 80 项降至 **4 项（严重 0 项、一般 4 项）**，全部为「超大头文件[C++]」（`.h` 文件有效代码行 `nloc > 500`）。
- **验证状态重要声明**：原始 `v136` 在 CANNJudge `#1177155` 上取得 17/17 Pass；加注释后的 `c92ea958`、首轮修复提交 `0033df9e` 以及本次二次拆分头文件版本（v2）**均未进行 CANN 工具链编译、未在 Ascend NPU 设备上运行测试、也未提交 CANNJudge 判题**。切勿将本地静态检查、桩编译或 CPU 等价测试当作 CANN 编译或 NPU 判题结果。

---

## 一、CI #209 剩余 4 项「超大头文件[C++]」告警的二次拆分方案

OpenLiBing CodeCheck 对 C++ 头文件（`.h`）实施 `有效代码行 (nloc) <= 500` 的门禁规则。本次在不改变任何类布局、模板参数、函数逻辑或指令执行顺序的前提下，将 CI #209 报告指出的 4 个超限头文件按职责与类外内联成员实现拆分为 9 个头文件，确保**每个 `.h` 文件的有效代码行（`nloc`）和总行数（`total`）双双低于 500 行**：

| CI #209 告警头文件（`0033df9e`） | CI #209 有效代码行 | 二次拆分后的目标头文件（v2） | 拆分后有效代码行 (`nloc`) | 拆分后总行数 (`total`) | 拆分方式与可见性保证 |
| :--- | :---: | :--- | :---: | :---: | :--- |
| `op_kernel/tree_line_mode3_and_static.h` | 797 | `op_kernel/tree_line_mode3.h`<br>`op_kernel/tree_line_static_batch_modes.h` | 469<br>332 | 487<br>367 | 按独立算子 Kernel 类拆分：`tree_line_mode3.h` 包含 `TreeLineMode3Shared` 命名空间与 `KernelTreeLineMode3<T>`（Mode 3）；`tree_line_static_batch_modes.h` `#include "tree_line_mode3.h"` 并定义 `KernelTreeLineSinglePointStatic<T>`（Mode 6）、`KernelTreeLineSingleSegmentStatic<T>`（Mode 7）、`KernelTreeLineMultiSmallStatic<T>`（Mode 8/9）。 |
| `op_kernel/tree_line_specialized_static.h` | 635 | `op_kernel/tree_line_pair32.h`<br>`op_kernel/tree_line_specialized_static.h` | 263<br>376 | 291<br>412 | 按独立算子 Kernel 类拆分：`tree_line_pair32.h` 定义 `KernelTreeLinePair32<T>`（Mode 5/21）；`tree_line_specialized_static.h` `#include "tree_line_pair32.h"` 并定义 `KernelTreeLineSmallMStatic<T>`（Mode 10）与 `KernelTreeLine513Static`（Mode 12）。 |
| `op_kernel/tree_line_general_modes.h` | 756 | `op_kernel/tree_line_general_modes.h`<br>`op_kernel/tree_line_general_compute.h` | 379<br>442 | 424<br>472 | 采用 C++ 标准模板头文件拆分模式：`tree_line_general_modes.h` 定义 `KernelTreeLineMinDistance<T, MODE>` 类结构、初始化、核间切分、UB 分配与主循环，并在文件末尾 `#include "tree_line_general_compute.h"`；`tree_line_general_compute.h` 提供 `template <typename T, uint32_t MODE> __aicore__ inline ... KernelTreeLineMinDistance<T, MODE>::...` 的类外内联定义。任何包含 `tree_line_general_modes.h` 的编译单元均自动获得完整模板定义。 |
| `op_kernel/tree_line_prune.h` | 1277 | `op_kernel/tree_line_prune.h`<br>`op_kernel/tree_line_prune_screen.h`<br>`op_kernel/tree_line_prune_verify.h` | 441<br>473<br>464 | 479<br>489<br>480 | 采用相同的 C++ 类外内联模板成员定义模式：`tree_line_prune.h` 定义 `KernelTreeLinePrune<T>` 类结构、初始化、核间切分与 FP32 线段预处理，并在文件末尾依次 `#include "tree_line_prune_screen.h"` 与 `#include "tree_line_prune_verify.h"`；前一个子头文件实现 FP16 缩放、分块粗筛与 `FallbackFullFP32`，后一个子头文件实现候选解码、FP32 精确几何验证、平局归并与 `KernelTreeLinePruneLookahead` 类型别名。 |

### 全仓文件行数汇总（`check_metrics.py` 实测结果）

| 文件路径 | 总行数 (`total`) | 有效代码行 (`nloc`) | 门禁上限 | 状态 |
| :--- | :---: | :---: | :---: | :---: |
| `op_host/tree_line_min_distance.cpp` | 318 | 267 | 2000 | 通过 |
| `op_kernel/tree_line_min_distance.cpp` | 179 | 155 | 2000 | 通过 |
| `op_kernel/tiling_key_tree_line_min_distance.h` | 29 | 13 | 500 | 通过 |
| `op_kernel/tree_line_min_distance_tiling.h` | 25 | 10 | 500 | 通过 |
| `op_kernel/tree_line_common.h` | 428 | 384 | 500 | 通过 |
| `op_kernel/tree_line_mode3.h` | 487 | 469 | 500 | 通过 |
| `op_kernel/tree_line_static_batch_modes.h` | 367 | 332 | 500 | 通过 |
| `op_kernel/tree_line_pair32.h` | 291 | 263 | 500 | 通过 |
| `op_kernel/tree_line_specialized_static.h` | 412 | 376 | 500 | 通过 |
| `op_kernel/tree_line_general_modes.h` | 424 | 379 | 500 | 通过 |
| `op_kernel/tree_line_general_compute.h` | 472 | 442 | 500 | 通过 |
| `op_kernel/tree_line_prune.h` | 479 | 441 | 500 | 通过 |
| `op_kernel/tree_line_prune_screen.h` | 489 | 473 | 500 | 通过 |
| `op_kernel/tree_line_prune_verify.h` | 480 | 464 | 500 | 通过 |
| `op_kernel/tree_line_small_modes.h` | 466 | 426 | 500 | 通过 |

---

## 二、首轮已解决的 80 项 CodeCheck 问题回顾（保持 0 回归）

### 1. 7 项严重（Critical）除零告警分析与保护
1. `op_host/tree_line_min_distance.cpp:157`（`cores`）：通过 `safeCores = (cores > 0U) ? cores : 1U` 与三元表达式 `(safeCores > 0U) ? ((total + safeCores - 1U) / safeCores) : total` 显式保护。
2. `op_host/tree_line_min_distance.cpp:195`（`blocks`）：在 `ComputeMode1OrMode0Tiling` 中增加 `if (blocks == 0U) { blocks = 1U; }` 及 `(blocks > 0U) ? ((total + blocks - 1U) / blocks) : total`。
3. `op_kernel/tree_line_min_distance.cpp:3784/3785`（`KernelTreeLinePrune::ComputeCoreRange` 的 `blocks`）：入口条件显式写为 `if (blocks == 0U || core >= blocks)`，除法统一通过 `TreeLineCommon::SafeDivU`（内含 `den > 0U` 判断）执行。
4. `op_kernel/tree_line_min_distance.cpp:5236/5237`（`KernelTreeLinePruneLookahead::ComputeCoreRange` 的 `blocks`）：通过 `using KernelTreeLinePruneLookahead = KernelTreeLinePrune<T>;` 复用同一安全实现。
5. `op_kernel/tree_line_min_distance.cpp:6653`（`KernelTreeLineSmallCompileShape::DistanceSquared` 的 `len`）：条件改为 `if (num > 0.0f && len > 0.0f) { t = (num >= len) ? 1.0f : (num / len); }`，零长度线段 `len == 0.0f` 时保持 `t = 0.0f`，距离退化为点到线段端点距离。

### 2. 函数长度、圈复杂度、嵌套深度与重复代码指标
- **超大函数（`nloc > 50`）**：`0` 项（全仓所有函数最大 `nloc = 42`，最大总行数 `45`）。
- **圈复杂度过高（`CCN >= 20`）**：`0` 项（全仓最大 `CCN = 10`）。
- **嵌套深度过深（`depth > 4`）**：`0` 项（全仓最大控制流嵌套深度 `depth = 3`）。
- **重复代码块**：`0` 项（跨文件连续 6 行及以上归一化重复窗口数为 `0`）。

---

## 三、可复现校验套件说明与验证边界

为便于独立复核本地做过的每一项静态检查、桩编译与 CPU 等价测试，同时避免测试脚本或桩头文件混入 PR 分支污染构建或触发新的 CodeCheck 扫描告警，本次提供**两个独立的压缩包**：

1. **`TreeLineMinDistance_v136_PR101_ci_fixed_v2.zip`（可直接提交至 PR #101 的纯净工程源码包）**
   - 仅包含 `CMakeLists.txt`、`README.md`、`PR101_CI_AND_SCA_REPORT.md`、`op_host/`、`op_kernel/`。
2. **`TreeLineMinDistance_v136_PR101_verification_suite.zip`（独立可复现校验脚本、桩头文件与实测日志包）**
   - `check_metrics.py`：扫描目标目录下所有 `.cpp` / `.h` 文件的总行数、有效代码行（`nloc`）、函数行数、圈复杂度（CCN）、控制流嵌套深度、6 行归一化重复窗口与变量除零保护。
   - `mock_ascend/`：用于无 CANN 环境下的宿主 C++17 语法、类型与模板实例化检查的最小 Ascend C / Tiling / Register 桩头文件。
   - `test_compile_all.cpp`：包含 `op_kernel/tree_line_min_distance.cpp` 并显式实例化全部 38 个 Kernel 入口组合（`float`/`half` $\times$ 19 个 `MODE`），同时测试每个 `.h` 头文件的独立包含编译。
   - `test_compile_host.cpp`：包含 `op_host/tree_line_min_distance.cpp` 并执行 Host Tiling 与 InferShape/InferDataType 桩编译测试。
   - `verify_equivalence.cpp`：将原始 `v136` 的 `TilingFunc` / `DistanceSquared` 逻辑与重构后的实现进行 `28,560` 组 `(batch, points, segments, cores, dtype)` Exhaustive Tiling 对比测试，以及零长度线段、端点重合、内部投影、极短线段距离等价性对比测试。
   - `run_verification.sh`：一键运行上述全部检查的 Shell 脚本（用法：`bash run_verification.sh /path/to/TreeLineMinDistance_v136_PR101_ci_fixed_v2`）。
   - `verification_run.log`：在本地实际运行 `run_verification.sh` 生成的完整终端输出日志。

### Gemini 报告的检查与未验证项

以下本地结果及日志由 Gemini 提供。等价测试使用测试文件中独立整理的函数副本，不能证明整个算子实现等价。

| 检查项目 | 状态 | 验证方式与复现命令 |
| :--- | :---: | :--- |
| CodeCheck 静态门禁指标（文件/头文件 `nloc<=500`、函数 `nloc<=45`、`CCN<=10`、`depth<=3`、重复代码=0、除零保护） | **已验证（本地实测通过）** | `python3 check_metrics.py <repo_dir>` |
| 13 个 Kernel 头文件独立 `#include` 编译检查（`g++ -std=c++17 -Wall -Wextra -Werror`） | **已验证（本地实测通过）** | 包含在 `run_verification.sh` 步骤 2 |
| Kernel 全部 38 个模板实例（`float`/`half` $\times$ 19 个 `MODE`，含默认宏与 `-D__DAV_VEC__`）桩编译与空调用检查 | **已验证（本地实测通过）** | 包含在 `run_verification.sh` 步骤 3、4 |
| Host Tiling `28,560` 组参数等价性 + `DistanceSquared` 零长度线段数值等价性测试 | **已验证（本地实测通过）** | 包含在 `run_verification.sh` 步骤 5、6 |
| **CANN 9.0 工具链（`bisheng` / `ccec`）真实编译与算子包生成** | **未验证（本地无 CANN 环境）** | 需在 CANN 环境执行 `cmake -S . -B build && cmake --build build --target binary package` |
| **Ascend 910B4 真机 NPU 运行测试（精度回归与耗时测试）** | **未验证（本地无 NPU 设备）** | 需在 Ascend 910B4 设备上运行算子测试程序 |
| **CANNJudge 平台在线判题（17 个测试点）** | **未验证（历史成绩来自原始 v136 `#1177155`）** | 需打包提交至 CANNJudge 平台验证 |
