<p align="center">
  <img src="https://img.shields.io/badge/Electrical%20Engineering%20SIG-CANN%20Community-orange" />
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" />
  <img src="https://img.shields.io/badge/status-active-brightgreen" />
</p>

> **elec-ops-inspection** 是 CANN 社区 Electrical Engineering SIG（电力行业兴趣小组）旗下的电力装备巡检算子库，
> 覆盖 CV 视觉检测与具身智能两大技术路线，面向输电线路、变电设备、配电设施等电力装备的智能化巡检场景，
> 基于华为昇腾（Ascend）硬件平台进行深度优化。

---


## 项目背景

电力装备巡检是保障电网安全稳定运行的核心运维环节。传统人工巡检面临效率低、成本高、高危场景作业风险大等问题。本仓库聚焦两大技术方向：

**① CV 视觉检测**：面向输电通道隐患识别，包括大型设备作业、鸟巢、绝缘子破损、锈蚀等典型缺陷的高精度检测，基于 Co-DETR 等先进目标检测模型，在 910B 上实现推理性能从**秒（s）级到毫秒（ms）级**的跨越。

**② 具身智能巡检**：面向机器狗、无人机等具身智能形态的"地空协同"移动巡检场景，提供面向大规模空间尺度建模（3DGS）的昇腾专用算子，支撑换流站等大型设施的空间感知与智能巡检。

---

## 核心算子列表

| 算子名称                    | 场景     | 描述               | 状态     |
| ----------------------- | ------ | ---------------- | ------ |
| [`optimized_transducer`](optimized_transducer/README.md) | 语音识别 | 对 `RNN-T` 模型的 `loss` 算子和 `softmax` 算子做了融合，显存优化和性能优化 | ✅ 已发布 |
| [`unique_v3`](unique_v3/README.md) | 数学通用算子 | 对排序算法在 AIV 上做了优化，增加 `inverse`、`counts` 输出 | ✅ 已发布 |

---

## 开发辅助

## 开发辅助

| Skill | 用途 |
|---|---|
| [`electric-inspection-shape-boundary`](.agents/skills/electric-inspection-shape-boundary/SKILL.md) | 从接口、Host Tiling、Kernel 和样例推导 Shape、rank、tile/block、尾块、对齐、packed 输入关系与整数范围边界，生成有公式依据的最小边界用例。 |
| [`electric-inspection-example-reproducibility`](.agents/skills/electric-inspection-example-reproducibility/SKILL.md) | 帮助新增或审查算子示例、README 调用片段和 PR 验证说明，检查运行命令、依赖、合成输入、参考输出、无 NPU 降级和可复现性。 |
| [`electric-inspection-operator-catalog`](.agents/skills/electric-inspection-operator-catalog/SKILL.md) | 维护根 README 算子清单、目录链接、场景分类和发布状态。 |
| [`electric-inspection-golden-reference`](.agents/skills/electric-inspection-golden-reference/SKILL.md) | 为算子建立可信、独立且可复现的 Golden 参考结果，设计测试向量、随机种子、精确/容差比较与不变量，并记录 loss、grad、output、inverse、counts 的验证证据。 |
| [`electric-inspection-tiling-review`](.agents/skills/electric-inspection-tiling-review/SKILL.md) | 审查 Host Tiling、TilingData、blockDim、tiling key、任务分配和 workspace 字节布局，定位未消费字段、重复常量、单位漂移与容量待确认项。 |
| [`electric-inspection-build-install-triage`](.agents/skills/electric-inspection-build-install-triage/SKILL.md) | 按入口、环境、CMake、编译、打包、`.run` 安装、vendor 验收和 Python 扩展阶段定位首个构建安装故障，并给出最小重跑路径。 |
| [`electric-inspection-runtime-triage`](.agents/skills/electric-inspection-runtime-triage/SKILL.md) | 在模块已能加载后，按动态库/符号、描述符、GetWorkspaceSize、workspace、设备流、提交、同步和结果读取阶段定位 CANN/ACLNN 算子运行时故障。 |

---

## 🤝 参与贡献

欢迎所有对电力 AI 和昇腾开发感兴趣的开发者参与共建！贡献方式：

1. 提交 Issue 反馈问题或建议
2. Fork 本仓库并提交 Pull Request
3. 参与 Electrical Engineering SIG 定期研讨会
4. 完善文档与示例

贡献规范请参考 [`CONTRIBUTING.md`](CONTRIBUTING.md)，新增算子请统一放入 [`operators/`](operators/) 目录。
新增或重写算子说明时，可参考 [`docs/operator-readme-template.md`](docs/operator-readme-template.md)。
设计和记录算子测试时，可使用 [`docs/operator-test-plan-template.md`](docs/operator-test-plan-template.md)。
涉及接口、行为或安装兼容性变化时，可使用 [`docs/operator-change-note-template.md`](docs/operator-change-note-template.md)。

仓库提供了面向算子性能证据整理的辅助 Skill：[`inspection-benchmark-evidence`](.agents/skills/inspection-benchmark-evidence/SKILL.md)，用于在提交 PR 前检查巡检算子的 benchmark 环境、shape 矩阵、baseline、统计口径和可复现命令。

仓库提供了面向技能贡献的辅助 Skill：[`electric-inspection-skill-creator`](.agents/skills/electric-inspection-skill-creator/SKILL.md)，用于为电力巡检算子场景创建差异化、可验证、可提 PR 的仓库本地 Skill。

仓库提供了面向算子贡献的辅助 Skill：[`electric-inspection-op-readiness`](.agents/skills/electric-inspection-op-readiness/SKILL.md)，用于在提交 PR 前检查电力巡检场景说明、接口约束、精度证据、性能证据和文档完整性。

仓库提供了本地 Codex 知识库框架：[`electric-inspection-knowledge-base`](.agents/skills/electric-inspection-knowledge-base/SKILL.md) 与 [`.agents/knowledge`](.agents/knowledge/README.md)，用于记录公开、可审查、可渐进加载的项目上下文，不包含个人知识库内容。

---

## 👥 维护团队

**Maintainers**

|姓名|GitCode ID|单位|
|---|---|---|
|梁寿愚|@jason2025|南方电网人工智能研究中心|
|陆璐|@Lulu_scut|华南理工大学|
|陈辰|@xchencehn|杭州天宽科技|
|张玉橙|@Splendid2025|昇腾产品线|
|田野|@tianye525|AI算力基础设施|

**Committers**

|姓名|GitCode ID|单位|
|---|---|---|
|余涛|@yutao_scut|华南理工大学电力学院|
|刘迪|@weixin_34344963|清华大学电机系|
|江豪|@yhyyyl|华南理工大学 / 南方电网联培|
|陈昀|@edconeone|华南理工大学|
|莫程翔|@Andrewmo1|华为公司|
|李博|@gcw_FHrfwZBn|华为公司|

---

## 📄 许可证

本项目基于 [Apache License 2.0]() 开源。
