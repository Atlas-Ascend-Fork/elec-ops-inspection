# fuseddisconnectorkptsoftargmax

提交团队: 伍六七-JLU
提交者: wellszhou


## 算子实现介绍

# FusedDisconnectorKeypointSoftArgmax

基于 Ascend C 实现、面向 `ascend910b` 的隔离开关关键点热图预测与 Soft-Argmax 融合算子。

## 1. 算子功能与计算语义

本算子将 $3 \times 3$ 空间卷积（`padding = 1`、`stride = 1`、零填充）、温度缩放 Softmax 以及 2D Soft-Argmax 坐标期望计算融合在单核内完成，输出 4 个关键点的原图坐标与响应峰值。

对于样本 $b \in [0, B)$ 与关键点通道 $k \in \{0, 1, 2, 3\}$：

1. **卷积响应计算**
   $$L_{b,k}(y, x) = \text{bias}[k] + \sum_{c=0}^{31} \sum_{r=0}^{2} \sum_{s=0}^{2} \text{weight}[k, c, r, s] \cdot \text{feat}[b, c, y + r - 1, x + s - 1]$$
   越界位置按 `0.0f` 填充。

2. **峰值提取与温度缩放指数权重**
   $$m_{b,k} = \max_{y, x} L_{b,k}(y, x)$$
   $$w_{b,k}(y, x) = \exp\left(\frac{L_{b,k}(y, x) - m_{b,k}}{\tau}\right), \qquad Z_{b,k} = \sum_{y, x} w_{b,k}(y, x)$$
   其中输出 `scores[b, k]` 即为 $m_{b,k}$。

3. **2D Soft-Argmax 坐标回归**
   $$\hat{x}_{b,k} = \left(\frac{\sum_{y, x} x \cdot w_{b,k}(y, x)}{Z_{b,k}} + 0.5\right) \times \text{stride}, \qquad \hat{y}_{b,k} = \left(\frac{\sum_{y, x} y \cdot w_{b,k}(y, x)}{Z_{b,k}} + 0.5\right) \times \text{stride}$$

## 2. 接口与支持范围

| 参数 | 类型 | 数据类型 | Shape / Format | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `feat` | Input | `DT_FLOAT` | `[B, 32, H, W]` (`FORMAT_ND`) | 输入特征图 |
| `weight` | Input | `DT_FLOAT` | `[4, 32, 3, 3]` (`FORMAT_ND`) | $3 \times 3$ 卷积核权重 |
| `bias` | Input | `DT_FLOAT` | `[4]` (`FORMAT_ND`) | 卷积偏置 |
| `tau` | Attr | `Float` | — | 温度系数，需为有限正数（`tau &gt; 0`） |
| `stride` | Attr | `Int` | — | 坐标映射步长，当前约束为 `4` |
| `keypoints` | Output | `DT_FLOAT` | `[B, 4, 2]` (`FORMAT_ND`) | 关键点 `(x, y)` 坐标 |
| `scores` | Output | `DT_FLOAT` | `[B, 4]` (`FORMAT_ND`) | 卷积响应图空间最大值 |

支持输入规格：
- `B`：`1 &lt;= B &lt;= 4`
- `(H, W)`：`(48, 64)`、`(64, 64)`、`(72, 96)`

## 3. 代码结构

```text
.
├── CMakeLists.txt
├── README.md
├── op_host
│   ├── CMakeLists.txt
│   └── fused_disconnector_keypoint_soft_argmax.cpp
└── op_kernel
    ├── CMakeLists.txt
    ├── fused_disconnector_keypoint_soft_argmax.cpp
    ├── fused_disconnector_keypoint_soft_argmax_tiling.h
    └── tiling_key_fused_disconnector_keypoint_soft_argmax.h
```

## 4. 实现架构

### 4.1 Host 侧 Tiling 与校验

Host 侧 `TilingFunc` 完成以下处理：

1. **规格与类型校验**：校验输入输出形状、数据类型（`DT_FLOAT`）、`batch` 范围、`tau` 与 `stride` 合法性。
2. **核数分配与 Tile 划分**：按每 2 行划分为一个 Tile（`tileCount = height / 2`）。通过 `PlatformAscendC::GetCoreNumAiv()` 获取 AIV 核数，按 `batch` 均分确定每样本的工作核数 `workersPerBatch` 与总核数 `cores = workersPerBatch * batch`（校验 `cores &gt;= batch * 4`），并通过 `SetScheduleMode(1)` 调度纯 AIV 核执行。
3. **UB 容量检查**：校核硬件 UB 字节数同时满足稀疏路径 `sparseUbBytes = (height + 2) * (width + 16) * 4 + 5 * height * width * 4 + 6016` 与稠密路径 `requiredUbBytes`（`width == 64` 为 `163328` 字节，`width == 96` 为 `174848` 字节）。
4. **倒数预计算与 Workspace 配置**：在 Host 端以 `double` 计算 `1.0 / tau`，若 `tau` 及倒数均在 FP32 正规数范围内则下发 `invTau`（否则置 `0.0f` 由核内回退至向量除法）；申请 `batch * tileCount * 64` 字节的全局 Workspace，供稠密路径存储各 Tile 的中间统计量。

### 4.2 Kernel 侧执行路径

核函数 `fused_disconnector_keypoint_soft_argmax` 全程在 Vector 单元（AIV）上完成计算。执行入口依据运行时 `weight` 稀疏特征在两条路径间自动分流：

#### 路径一：`SparseFullImage&lt;Tiling, Height, Width&gt;`

1. **运行时稀疏检测（`Classify`）**：将 `1152` 个权重读入 UB，使用 `CompareScalar` 生成非零掩码并按 18 个 `uint64_t` 扫描。当总非零权重数不超过 `36` 且每个关键点最多只依赖 1 个输入通道时进入该路径；否则所有核一致退回至稠密路径。
2. **单核直算与边界处理**：仅由 `GetBlockIdx() &lt; batch * 4` 的核分别处理 1 个 `(batch, keypoint)` 任务，不使用 Workspace，也不进行核间同步：
   - 若对应关键点权重全零（`channels[keypoint] &lt; 0`），直接写出中心坐标 `(Width * 0.5f * stride, Height * 0.5f * stride)` 与 `bias[keypoint]`。
   - 若仅依赖单个通道，则将整张单通道特征图加载至 UB，构建步幅 `Pitch = Width + 16` 的零填充 Halo（上下各 1 行置零，左右各 `8` 个 `0.0f` 填充），通过 `MakeCoordinates` 生成坐标与字节偏移，仅针对非零卷积核元素调用 `Gather` + `Axpy` 累加。
   - 在 UB 内完成加偏置、`ReduceMax`、`ScaleByTemperature`（`invTau &gt; 0.0f` 时走 `Muls`，否则用 `Duplicate` + `Div`）、`Exp`、`ReduceSum` 及坐标加权期望计算，通过 `DataCopyPad` 写出 `keypoints` 与 `scores`。

#### 路径二：`BatchedDenseStatistics&lt;Tiling, Width, Channels&gt;` + `KeypointMerge`

当权重不满足单通道稀疏条件时，通过两阶段计算完成全图融合：

1. **阶段一：`BatchedDenseStatistics`（Tile 级卷积与局部统计）**
   - **模板参数特化**：`Width == 64` 时实例化 `Channels = 32`；`Width == 96` 时实例化 `Channels = 16`，分两轮累加 32 个输入通道的部分和，以控制 UB 占用并避免跨距超出指令 stride 上限。
   - **条带搬运与边界清零（`LoadStrip` / `ClearStripEdges`）**：按 `InputChannels = 16` 分批搬运当前 Tile（2 行输出，对应 3~4 行输入）。针对 $3\times 3$ 卷积核的左/中/右 3 列偏移发起 3 次多通道 `DataCopyPad` 连续搬运，随后使用带掩码的 `Duplicate` 清除因连续移位拷贝导致的跨行回绕元素及图像上下界外行。
   - **向量化卷积与通道归约（`Convolve`）**：初始化时使用 `Brcb` 将 1152 个权重广播为 32 字节对齐块存入 `weightStore`；遍历 9 个卷积 tap 并以 `VectorWidth = 64` 切分 `Count = 2 * Width`，通过 `Mul` / `MulAddDst`（`repeat = 16`）并发完成 16 通道乘累加，之后在 `partialStore` 上执行树状折半归约得到 4 个关键点的卷积结果。
   - **局部统计量打包（`CalculateStatistics`）**：在 UB 内直接完成加偏置及 4 项 Tile 统计量计算——局部最大值 $m_{\text{tile}}$、局部指数和 $Z_{\text{tile}}$、X 坐标加权和 $X_{\text{tile}}$ 与 Y 坐标加权和 $Y_{\text{tile}}$（利用每 Tile 仅含 2 行、局部行偏移为 `0/1` 的特征，仅对第二行做宽度为 `Width` 的 `ReduceRows` 并叠加 `y * Z_tile`）。最终通过 `DataCopyPad` 将每个 `(batch, tile)` 对应的 4 项统计量打包为 64 字节写入 `workspace`。

2. **阶段二：`KeypointMerge`（全核同步与跨 Tile 合并）**
   - 释放阶段一对象后调用 `AscendC::SyncAll&lt;true&gt;()` 等待全部 AIV 完成写入。
   - 由 `GetBlockIdx() &lt; batch * 4` 的核分别读取对应样本的 `tileCount` 个 64 字节统计包，使用 `Gather` 抽取目标关键点的 $m_{\text{tile}}$、$Z_{\text{tile}}$、$X_{\text{tile}}$、$Y_{\text{tile}}$ 序列。
   - 使用 `WholeReduceMax` 求得全局最大值 $M$（即 `scores[b, k]`），计算重缩放因子 $\alpha_{\text{tile}} = \exp((m_{\text{tile}} - M) / \tau)$，通过 `repeat = 3` 的 `Mul` 与 `WholeReduceSum` 同步汇总得到 $Z_{\text{global}}$、$X_{\text{global}}$、$Y_{\text{global}}$，最终写出映射后的 `keypoints` 与 `scores`。

## 5. 编译方法

安装 CANN Toolkit 9.0（或兼容版本）及 CMake 后执行：

```bash
source /path/to/cann/set_env.sh
cmake -S . -B build -DCANNJUDGE_COMPILE_JOBS=4
cmake --build build --parallel 4 --target binary package
```

编译产物位于 `build` 目录。
