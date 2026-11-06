# rnnt-loss-ascend

**项目简介**

本项目为基于华为 CANN 计算框架开发的 RNN-T Loss 专用算子，并参考 [optimized_transducer](https://arxiv.org/abs/1909.12415) 的内存与计算优化策略，在昇腾 NPU 上联合计算每条样本的 loss 与 logits 梯度。当前仓库提供 ACLNN 两段式接口和 pybind 原始绑定，支持与 `torchaudio.functional.rnnt_loss`（torchaudio v2.8.0）的 float32 结果进行对比；pybind 暂未封装 autograd 与 reduction，因此不是 torchaudio 接口的直接替换。

---

**主要功能**

- **参考结果对比**：仓库测试脚本使用 `torchaudio.functional.rnnt_loss` 生成 CPU loss 与梯度，供 NPU 结果比对。
- **分层接口**：ACLNN 接口接收 `blank`、`clamp` 和 `fused_log_softmax`；当前 pybind 接口仅暴露 `blank`，内部固定 `clamp=-1.0`、`fused_log_softmax=true`。
- **内存优化**：将原始 `(N, T, U, V)` 四维广播张量压缩为 `(Σ(t,u), V)` 二维紧凑表示，消除冗余内存分配。
- **梯度融合**：将 softmax 与梯度计算原地合并，省去中间大张量存储，梯度结果直接原地覆写 logits 输出缓冲区。
- **多核负载均衡**：Host 侧采用最大任务优先 + 最小负载优先策略，将批次样本均衡分配至多个 AI Core，充分利用 NPU 算力。
- **当前递推实现**：Kernel 侧按 t 行顺序计算 α，按 t 行逆序计算 β；每行内部按 u 方向串行递推。片上缓存保留相邻两行，完整 α/β 表写入 workspace，供后续梯度计算使用。
- **当前数据类型**：OpDef 与 Tiling 当前仅接受 float32 logits，loss 与 grad 也以 float32 输出。

---

**应用场景**

| 应用领域 | 典型场景 | 说明 |
|---|---|---|
| 语音识别 | 流式 ASR 模型训练 | 以 RNN-T 为解码器的端到端语音识别，online streaming 场景 |
| 语音翻译 | 端到端语音翻译 | 多语言序列转换任务中的 transducer loss 计算 |
| 大规模预训练 | 工业级语音数据训练 | 万小时级数据下 batch size 受限问题，通过内存优化扩大批量 |
| 模型蒸馏 | 轻量化 RNN-T 模型 | 教师-学生框架中高效计算 soft target 的 transducer loss |

---

**昇腾原生 × 电力行业赋能**

**场景：电力设备巡检 Agent 交互（elec-ops-inspection）**

**业务背景**：
在特高压换流站（如主变压器室、冷却塔区域）的巡检作业现场，常年伴随 80-90 分贝的严重机械低频噪音。巡检工人作业时通常携带各类现场作业工具，在采集数据或者填写表单时，传统的平板触控或按键终端容易误触且效率低下。通过端侧算力部署离线语音 Agent 助理实现“解放双手”是未来智能巡检的发展方向，但这要求模型在强噪下具备极高的鲁棒性，且受限于边缘算力，模型训练与推理的显存调度必须极致优化。

- **算子价值**：将复杂的 Transducer Loss 内存占用降低约 50%，前反向极致融合，使得在同等算力集群下能吞吐更大 Batch Size 的长音频训练数据。
- **对应模型**：Conformer / Emformer 强抗噪流式 ASR 模型、Qwen-Audio 等端侧多模态 Agent 基座大模型。
- **应用落地**：
  1. **强噪环境离线语音指令响应（应用：巡检缺陷无接触一键录入）**：利用本算子高效的内存复用机制，可在国产算力上训练更高精度的 Conformer 工业抗噪模型。一线人员在嘈杂的变压器旁语音交互“记录二号主变油温过高并生成紧急缺陷工单”，端侧 Agent 能够以极低延迟精准转写文本，并联动内部系统接口自动派单，实现全程无接触作业。

---

**价值说明**

**为什么需要 RNN-T Loss 专用融合算子？**

**1. 内存搬运优化**

传统 RNN-T Loss 实现在训练时需维护 `(B, T, U, V)` 量级的多个大张量，以标准 softmax + 梯度分步计算为例，典型 HBM 搬运路径如下：

- 读取 logits → 计算 softmax → 写回概率张量 P（2 次搬运）
- 读取 P → 计算 ∂P/∂h → 写回雅可比张量（2 次搬运）
- 读取 P、α、β → 计算 ∂L/∂h → 写回梯度张量（6 次搬运）
- 读取梯度 → 做 reduction → 写回最终结果（2 次搬运）

总计约 **12 次以上** HBM 搬运，中间结果全部占据 HBM。

当前源码在一次 kernel 调用中执行 log_softmax、β/α 递推和梯度计算，但这不等于只读写 HBM 两次：log_prob 与 grad 共用 GM 缓冲区，α/β 完整表使用额外 workspace；各阶段仍通过 DataCopy/DataCopyPad 搬运数据。不能把理论融合路径的两次搬运当成此实现的实测或静态准确计数。

| 优化点 | 传统方案 | 融合算子 |
|---|---|---|
| HBM 搬运路径 | 上述为分步计算的示意路径 | 当前实现多阶段访问 GM；精确次数需结合实际路径与设备分析 |
| 中间大张量数量 | 3 个（P、∂P/∂h、∂L/∂h） | 0 个（全部原地覆写） |
| logits 内存布局 | 4D `(B,T,U,V)` 广播展开 | 2D `(Σ(t,u), V)` 紧凑排列 |
| Kernel 启动次数 | 多次（softmax、forward、backward 分离） | 1 次（全流程融合） |

**2. 计算优化**

| 优化技术 | 说明 |
|---|---|
| 2D 张量压缩 | 每条样本的 logits 按实际有效 (t,u) 位置拼接为 2D，避免 padding 位置的无效计算 |
| 行递推 α/β | α 沿 t 正序、u 正序递推；β 沿 t 逆序、u 逆序递推。使用相邻两行 UB 缓存并将完整表写入 GM workspace |
| 梯度公式化简 | 将 softmax → P → ∂L/∂h 三步融合为公式 13（见论文），直接计算梯度，省去中间变量 |
| tile 自适应分块 | 根据词表大小 V 和 UB 容量自动确定 tile 面积 M，充分利用片上缓存 |
| double buffer | 数据搬运与计算流水并发，隐藏 HBM 访问延迟 |

核心梯度融合公式如下：

$$\frac{\partial L}{\partial h_{t,u}^k} = P(k|t,u) \cdot \frac{\alpha(t,u)}{P(\mathbf{y}|\mathbf{x})} \cdot \left[\beta(t,u) - \beta(\text{next})\right]$$

其中 $P(k|t,u) = \text{softmax}(h_{t,u})_k$，$\alpha$、$\beta$ 为前向后向概率，临时变量 $P(k|t,u)$ 直接复用梯度输出缓冲区，避免额外分配。

**3. 精度保证**

| 特性 | 说明 |
|---|---|
| loss 输出与独立验证 | 当前由 β(0,0) 生成 loss；ComputeAlpha 内双路 loss 对比代码处于注释块，没有启用阈值报错。精度须通过独立 CPU/reference 回归验证 |
| 数值稳定 | 全程在 log 域进行 α/β 递推（log-sum-exp），避免概率连乘下溢 |
| clamp 保护 | 支持对梯度做 `[-clamp, clamp]` 截断，防止极端梯度破坏训练稳定性 |
| 可复现性 | 相同输入保证相同输出，满足科研可复现要求 |

---

**参数说明**

| 参数名 | 输入/输出 | 描述 | 数据类型 | 形状 |
|---|---|---|---|---|
| logits | 输入 | joiner 输出的未归一化 logits，按样本拼接为 2D | float32 | `(Σ(t_i × u_i), V)` |
| targets | 输入 | 目标序列 token，zero padding 对齐 | int32 | `(B, max_U - 1)` |
| logit_lengths | 输入 | 每条样本 encoder 输出的有效帧长 | int32 | `(B,)` |
| target_lengths | 输入 | 每条样本目标序列的真实长度 | int32 | `(B,)` |
| blank | 属性 | blank 标签 ID，-1 表示使用词表最后一个 class | int64 | — |
| clamp | 属性 | 梯度截断值，-1 表示不启用 | float32 | — |
| fused_log_softmax | 属性 | 是否在 kernel 内执行 log_softmax；若外部已做则设为 False | bool | — |
| loss | 输出 | 每条样本的 RNN-T loss | float32 | `(B,)` |
| grad | 输出 | 对 logits 的梯度，与输入 logits 同形 | float32 | `(Σ(t_i × u_i), V)` |

接口层说明：

- ACLNN 两段式接口接收表中的全部输入与属性。
- pybind 签名为 `rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=0)`，固定使用 `clamp=-1.0` 与 `fused_log_softmax=true`。
- pybind 直接返回 `(loss, grad)`，其中 `loss` 为 `(B,)`；调用方如需 `mean`、`sum` 或 autograd，需要在上层自行封装。

---

**约束说明**

- 当前实现要求 logits 为 float32，targets / logit_lengths / target_lengths 为 int32。
- blank 为负数时映射为 `V - 1`；传入非负值时，调用方应保证其位于 `[0, V - 1]`。
- 每条样本的有效帧数 T 和目标长度 U 满足 `T ≥ 1`，`U ≥ 1`，且 `logit_lengths[i] * (target_lengths[i] + 1)` 之和等于 logits 的第 0 维大小。
- clamp ≤ 0 时视为不启用梯度截断。

---

**架构设计**

```
Python 层（pybind 原始绑定）
    │  optimized_transducer_ascend_ops.rnnt_loss()
    │  直接返回 loss 与 grad
    ▼
aclnn 层（C++ 两段式接口，pybind 绑定）
    │  aclnnOptimizedTransducerGetWorkspaceSize()
    │  aclnnOptimizedTransducer()
    ▼
Host 侧（任务调度）
    │  最大任务优先 + 最小负载优先负载均衡
    │  将 B 条样本分配至多个 AI Core
    ▼
Kernel 侧（AscendC，单核处理单样本）
    │  Init：tile 切分、缓冲区初始化、缓存 targets/blank/clamp
    │  Process（常规 loss+grad 路径）：
    │    1. log_softmax(logits) → log_prob（写入 grad 缓冲区）
    │    2. 按 t/u 逆序行递推 log_β，并输出 loss
    │    3. 常规路径按 t/u 正序行递推 log_α
    │    4. 融合梯度公式计算 ∂L/∂h，原地覆写 grad
    └──────────────────────────────────────────
```

| 层级 | 职责 |
|---|---|
| Python | pybind 调用入口；当前不包含 autograd 与 reduction 封装 |
| aclnn | 两段式 C++ 接口，workspace 计算，pybind 导出 |
| Host | 核间负载均衡（最大任务优先 + 最小负载优先） |
| Kernel | 顺序行递推 β/α，tile 分块，log_prob / grad 缓冲区复用；α/β 完整表另占 workspace |

---

**调用说明**

```python
import torch
import torch_npu
import optimized_transducer_ascend_ops as ops

torch_npu.npu.set_device(0)

# logits 为 2D 紧凑张量，按样本顺序拼接
# 样本 0: T0=4, U0=3, V=8  →  12 行
# 样本 1: T1=6, U1=2, V=8  →  12 行
# logits shape: (24, 8)
logits = torch.randn(24, 8, dtype=torch.float32, device="npu")
targets = torch.tensor([[2, 3], [1, 0]], dtype=torch.int32, device="npu")
logit_lengths = torch.tensor([4, 6], dtype=torch.int32, device="npu")
target_lengths = torch.tensor([2, 1], dtype=torch.int32, device="npu")

loss, grad = ops.rnnt_loss(
    logits,
    targets,
    logit_lengths,
    target_lengths,
    -1,  # 使用词表最后一个 class 作为 blank
)

torch_npu.npu.synchronize()
print("loss:", loss.cpu())       # shape: (2,)
print("grad shape:", grad.shape) # (24, 8)
```

| 调用方式 | 样例入口 | 说明 |
|---|---|---|
| Python 接口 | `optimized_transducer_ascend_ops.rnnt_loss()` | pybind 原始接口，返回 loss 与 grad |
| ACLNN 接口 | [`examples/test_aclnn_optimizd_transducer.cpp`](./examples/test_aclnn_optimizd_transducer.cpp) | 先获取 workspace，再调用 `aclnnOptimizedTransducer` |

---

**算子特性**

| 特性 | 说明 |
|---|---|
| 前反向融合 | 一次 kernel 调用同时输出 loss 和 grad，避免二次遍历 α/β 表 |
| 内存原地复用 | log_prob 与 grad 共享同一 GM 缓冲区，峰值内存降低约 50% |
| 2D 紧凑输入 | 按样本拼接 logits，消除 padding 位置的无效 softmax 与梯度计算 |
| 当前行递推 | 相邻两行 α/β 位于 UB；完整表写入 GM workspace，没有实现反对角线批量并行 |
| 精度验证 | 当前未启用 kernel 内双路 loss 阈值校验，使用独立 CPU/reference 对照验证 |
| 多核负载均衡 | 按 T×U 面积排序，优先分配大任务到空闲核，减少批次尾部等待 |
| 梯度截断保护 | clamp > 0 时对每个梯度分量执行 `grad = clamp(grad, -clamp, clamp)` |

---

**精度测试**

与 torchaudio 2.8.0 CPU 参考实现对比（`reduction="none"`）：

| 数据类型 | batch | T 范围 | U 范围 | V | loss 最大绝对误差 | grad 最大绝对误差 | 是否通过 |
|---|---|---|---|---|---|---|---|
| float32 | 2 | 100–1000 | 100–300 | 50 | < 1e-4 | < 1e-4 | ✓ |
| float32 | 4 | 200–2000 | 50–500 | 128 | < 1e-4 | < 1e-4 | ✓ |

表中记录的 float32 最大绝对误差小于 **1e-4**。测试入口见 [`pybind/test.py`](./pybind/test.py)。

---

**性能数据**

与 torchaudio 2.8.0 CPU 参考实现对比（`reduction="none"`）：

| batch | T | U | V | torchaudio | NPU 融合算子 | 加速比 |
|---|---|---|---|---|---|---|
| 4 | 200 | 50 | 50 | 12.74 | 0.28 | 46.02 |
| 8 | 500 | 100 | 500 | 13.44 | 0.28 | 48.02 |
| 16 | 1000 | 200 | 5000 | 92.27 | 0.48 | 200.56 |
| 32 | 2000 | 500 | 50000 | 35993.41 | 8.22 | 4389.44 |
