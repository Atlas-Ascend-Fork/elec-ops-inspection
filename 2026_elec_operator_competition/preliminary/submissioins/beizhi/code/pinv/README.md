# Pinv：批量矩阵伪逆

本工程是“北枝 的队伍”在 2026 电力行业算子预赛中的最终获奖代码，
面向 Ascend 910B，使用 Ascend C 和 CANN 9.0.0 工程模板。

## 来源与验证

- CANNJudge 最终提交：[1173359](https://cannjudge.cn/public/2026_electronic/pinv/submission/6aed12dda5348dcef55fc401)。
- 提交时间：2026-11-06 07:35:57（北京时间）；官方结果：Pass 12/12。
- 开源前从该提交下载完整工程，七个文件与最终本地版本 `codex_v463_tiny8` 逐一核对一致。
- 初始归档只增加许可证头、格式化 C++ 源码并补充说明；当时四个 C++ 文件的非注释 token 与官方工程一致，三个 CMake 文件内容保持一致。
- 后续 CI 维护补充了下述零分母检查，当前开源版本与比赛归档有这些明确差异；后续静态门禁整改还按计算阶段拆分函数、合并三套求逆路径的相同辅助方法，并把模板实现分入对应头文件。Pass 12/12 仅指上述最终比赛提交，不扩展为维护版本或全部形状、框架 API 的验证声明。

### CI 维护与验证范围

- QR 消去在零范数时停止；QR 移位分母为零时使用不带移位的迭代。
- 小矩阵求逆遇到零主元时拒绝快速路径并沿用原有回退；Jacobi 输出跳过零范数。
- 提取 QR 移位表达式进行 CPU 检查：128 组正常值与原表达式逐位一致，4 个零分母边界检查通过。这些检查不模拟 Ascend 浮点行为或完整矩阵路径。
- 维护版本尚未重新完成 CANN 构建或设备测试：已尝试连接分配的开发环境，SSH 转发目标超时。未发起新的比赛评测。

- CI #312 的主 CodeCheck 已通过：最初 5 项严重除零和后续 85 项重复代码、规模、复杂度及嵌套问题均已消除。该结论不覆盖 CANN 编译、设备正确性或性能。
- Host 维护补齐了原先空的形状与类型推导：输出保留 batch 维并交换矩阵末两维，类型沿用已注册的 FP32/BF16，并检查空指针、非法 rank/类型以及输出类型设置失败。输入输出使用同一规格配置函数；输入输出名称、属性默认值与 Tiling 回调配置保持不变。
- 上述 Host 变更的 CPU API doubles 检查覆盖 17 个形状用例、6 个类型及错误传播用例，注册配置与修改前一致。CPU 检查不替代 CANN 编译或设备测试；Host Tiling 与 Kernel 计算实现未改。
- CI #317 的主 CodeCheck 为 0 问题，唯一失败项为 SCA 的 Host 片段许可证匹配；本次 Host 维护仍需新流水线核定，不能预先声称 SCA 已通过。保留参赛贡献者的真实版权归属与 Apache-2.0 声明。

### Host 模板来源

Host 的推导函数和算子注册声明来自本题 CANNJudge 官方工程模板，未从 `cann-ops/scopy` 移植。
初始归档的该部分与比赛提交前保存的官方模板逐 token 一致（忽略注释及空白）；
官方模板 Host 去 CR 后的 UTF-8 SHA256 为 `8725880a7f7d4e9ce66760dffffd7de85a3ac0d6821afc645a2819eb215ef544`。
SCA 报告对公共注册接口片段提出了许可证匹配问题；模板来源证据用于合规复核，不表示该项已经审核通过。

## 功能与接口

对于输入矩阵 `A`，按奇异值阈值构造 Moore–Penrose 伪逆：

```text
A = U diag(sigma) V^T
threshold = max(atol, sigma_max * rtol)
sigma_plus[i] = 1 / sigma[i]  if sigma[i] > threshold, else 0
A_plus = V diag(sigma_plus) U^T
```

| 项目 | 说明 |
| --- | --- |
| 输入 | ND 布局的 `A`，形状 `[..., m, n]` |
| 输出 | ND 布局的 `A_pinv`，形状 `[..., n, m]` |
| 注册类型 | FP32、BF16；计算中间量使用 FP32 |
| 属性 | 标量 `atol`、`rtol`，比赛固定 `hermitian=False` |
| 默认阈值 | 未指定 `rtol` 时，`atol > 0` 则取 0，否则取 `max(m,n) * FP32 epsilon` |

本实现保留最终竞赛工程的输入输出及属性注册，并补齐 Pinv 形状与类型推导。上表描述源码注册行为；
当前题面输入规格为 FP32、`m,n` 在 1–512、batch 维大小在 1–64，
并要求保留与舍弃奇异值之间具有明确谱间隔。
BF16 注册和接口不代表额外完成了该类型的全范围验证。

## 实现要点

- Host 根据矩阵尺寸、batch、片上缓存容量和 AIV 核数设置 Tiling、工作空间与执行路径。
- 小矩阵采用片上驻留、按 batch 分核和向量化消元；单行或单列输入走专用向量路径。
- 较大方阵采用多核分片消元与数值检查；199/200 阶路径使用专用求逆实现，256 阶路径结合 Cube 矩阵乘和 Vector 计算、残差修正。
- 快速路径保留阈值及数值拒收检查，不满足条件时进入 Jacobi SVD 回退；通用路径处理非方阵和需要奇异值截断的输入。
- 使用 UB 数据复用、对齐补齐、Gather/转置、核间同步和分块搬运减少计算及数据交换开销。

## 目录

```text
pinv/
├── CMakeLists.txt
├── README.md
├── op_host/
│   ├── CMakeLists.txt
│   └── linalg_pinv.cpp
└── op_kernel/
    ├── CMakeLists.txt
    ├── linalg_pinv.cpp       # 分档入口与向量路径
    ├── pinv_*.h              # 标量、小矩阵、分布式求逆及 Jacobi 实现
    ├── linalg_pinv_tiling.h
    └── tiling_key_linalg_pinv.h
```

## 构建

在已安装匹配 CANN/Ascend C 工具链的环境中，先按安装路径加载
`set_env.sh`，然后在本目录执行：

```bash
cmake -S . -B build -DASCEND_COMPUTE_UNIT=ascend910b
cmake --build build --parallel
```

工程使用模板提供的 `find_package(ASC)`、`npu_op_code_gen`、
`npu_op_library` 和 `npu_op_kernel_library`，生成算子接口与动态库。
设备使用与部署应遵守实际环境的分配及安装要求。

## 许可证

Apache License 2.0，见仓库根目录的 `LICENSE`。
