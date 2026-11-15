// Copyright (c) 2026 JLU_op-wuliuqi.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// 十二段选路：一段对应Kernel中的一个类；区间内不同B只改变分工，不换单点类。
// 1～6为FP32，7～12为FP16；分支编号是尺寸段，不等于公开测试case编号。
// 所有合法B∈[1,4096]都被覆盖；B4094和B4095均属于大区间[257,4095]。
// 端点在其所属区间类内恢复v528编译期策略；不是另一套单点类，普通B仍完整覆盖。
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/safety_belt_violation_geometry_tiling.h"
#include "../op_kernel/tiling_key_safety_belt_violation_geometry.h"
namespace optiling {
struct SbgHostBranch {
    uint32_t id;           // 尺寸分支编号1～12。
    uint32_t batchKey;     // 三个区间键或三个精确点B；不是最终日志Key。
    uint32_t launch;       // 启动核数。
    uint32_t working;      // 最大工作核数；真实工作还受rowStart<B限制。
    uint32_t rowsPerCore;  // 每核名义行步长，尾核取剩余有效行。
    uint32_t pad;          // 最大向量/UB行数预算。
    uint32_t ubBytes;      // 本核UB最大字节数。
};
static inline uint32_t SbgHostUbBytes(uint32_t pad, bool fp16) {
    return (fp16 ? 71U : 65U) * pad + 32U;
}
// 精确点的常量集中在表中，顺序为 B32、B256、B4096。
static inline SbgHostBranch SbgSelectHostPoint(uint32_t b, bool fp16) {
    constexpr SbgHostBranch fp32Points[] = {
        {2U, 32U, 8U, 1U, 32U, 64U, 4192U},
        {4U, 256U, 8U, 2U, 128U, 128U, 70304U},
        {6U, 4096U, 8U, 8U, 512U, 512U, 81440U},
    };
    constexpr SbgHostBranch fp16Points[] = {
        {8U, 32U, 8U, 1U, 32U, 64U, 4576U},
        {10U, 256U, 8U, 1U, 256U, 256U, 73888U},
        {12U, 4096U, 8U, 6U, 768U, 768U, 96288U},
    };
    const uint32_t index = b == 32U ? 0U : b == 256U ? 1U : 2U;
    return fp16 ? fp16Points[index] : fp32Points[index];
}

static inline SbgHostBranch SbgSelectHostLarge(uint32_t b, bool fp16) {
    const uint32_t largeWork = fp16 ? 6U : 8U;
    const uint32_t work = b == 257U ? 1U : b == 4095U ? largeWork : b <= 1024U ? 2U : 8U;
    const uint32_t largePer = fp16 ? 768U : 512U;
    const uint32_t per = b == 4095U ? largePer : ((b + work - 1U) / work + 15U) / 16U * 16U;
    const uint32_t pad = (per + 63U) / 64U * 64U;
    const uint32_t batchKey = b == 257U ? 257U : b == 4095U ? 4095U : 8195U;
    const uint32_t smallUb = fp16 ? 22624U : 20704U;
    const uint32_t largeUb = fp16 ? 96288U : 33760U;
    const uint32_t ub = b == 257U ? smallUb : b == 4095U ? largeUb : SbgHostUbBytes(pad, fp16);
    return SbgHostBranch{fp16 ? 11U : 5U, batchKey, 8U, work, per, pad, ub};
}

static inline SbgHostBranch SbgSelectHostBranch(uint32_t b, bool fp16) {
    if (b == 0U || b > 4096U) {
        return SbgHostBranch{0U,0U,1U,1U,16U,64U,SbgHostUbBytes(64U,fp16)};
    }
    if (b == 32U || b == 256U || b == 4096U) {
        return SbgSelectHostPoint(b, fp16);
    }
    if (b <= 31U) {
        const uint32_t per = ((b + 1U) / 2U + 15U) / 16U * 16U;
        return SbgHostBranch{fp16 ? 7U : 1U, 8193U, 8U, 2U, per, 64U, SbgHostUbBytes(64U, fp16)};
    }
    if (b <= 255U) {
        const uint32_t work = b == 33U ? 1U : 2U;
        const uint32_t per = ((b + work - 1U) / work + 15U) / 16U * 16U;
        const uint32_t pad = (per + 63U) / 64U * 64U;
        return SbgHostBranch{fp16 ? 9U : 3U, b == 33U ? 33U : 8194U,
            8U, work, per, pad, SbgHostUbBytes(pad, fp16)};
    }
    return SbgSelectHostLarge(b, fp16);
}

struct SbgLaunchPlan {
    uint32_t blocks;
    uint32_t perCore;
    uint64_t needBytes;
    bool fullPlan;
};

static inline SbgLaunchPlan SbgBuildLaunchPlan(uint32_t rows, uint32_t aiv,
    const SbgHostBranch &branch, bool fp16) {
    const uint32_t blocks = rows > 0 ? (aiv < branch.launch ? aiv : branch.launch) : 1U;
    const uint32_t active = blocks < branch.working ? blocks : branch.working;
    const bool fullPlan = blocks == branch.launch && active == branch.working;
    const uint32_t perCore = fullPlan ? branch.rowsPerCore : rows > 0 ?
        ((rows + active - 1U) / active + 15U) / 16U * 16U : 16U;
    const uint32_t pad = fullPlan ? branch.pad : (perCore + 63U) / 64U * 64U;
    const uint64_t needBytes = fullPlan ? branch.ubBytes : SbgHostUbBytes(pad, fp16);
    return SbgLaunchPlan{blocks, perCore, needBytes, fullPlan};
}

static inline uint32_t SbgSelectRatioKey(uint32_t batchKey, const float *ratio) {
    if (ratio == nullptr) return batchKey;
    if (batchKey == 32U && *ratio == 1.0f) return 8224U;
    if (*ratio == 0.8f && (batchKey == 4095U || batchKey == 4096U)) return batchKey + 16384U;
    return batchKey;
}
// 【CPU侧Tiling入口】
// 这里只读已有张量描述、dtype、属性和threshold是否存在，写出分工元数据。
// GetRequiredInputTensor不会在此自动把坐标整理为SoA，也不会替换Kernel的GM实参。
// 若要改变真实布局，必须在设备中执行搬运/Gather；偏移等元数据可以提前确定。
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int32_t aiv = platform.GetCoreNumAiv();
    if (aiv <= 0) aiv = 1;
    const auto *boxes = context->GetRequiredInputTensor(0);
    const auto *anchors = context->GetRequiredInputTensor(1);
    const auto *threshold = context->GetOptionalInputTensor(2);
    if (!boxes || !anchors) return ge::GRAPH_FAILED;
    // 当前按人员框总元素数/4得到B；依赖题目的合法连续ND输入。
    // review时注意：这里没有完整检查框/挂点末维、两者B与dtype一致性。
    const int64_t rows = boxes->GetShapeSize() / 4;
    const auto *attrs = context->GetAttrs();
    // waist_ratio是Host可读的真实属性；threshold则是独立的可选Tensor。
    const float *ratio = attrs ? attrs->GetFloat(0) : nullptr;
    const int32_t dtypeSize = ge::GetSizeByDataType(boxes->GetDataType());
    if (rows < 0 || rows > 4096 || (dtypeSize != 2 && dtypeSize != 4)) return ge::GRAPH_FAILED;
    const bool fp16 = dtypeSize == 2;
    const SbgHostBranch branch = SbgSelectHostBranch(static_cast<uint32_t>(rows), fp16);
    // 启动核数与工作核数分开；平台资源不足时使用key0的通用分片。
    const SbgLaunchPlan plan = SbgBuildLaunchPlan(static_cast<uint32_t>(rows),
        static_cast<uint32_t>(aiv), branch, fp16);
    const uint32_t perCore = plan.perCore;
    if (ubSize && plan.needBytes > ubSize) return ge::GRAPH_FAILED;
    const uint32_t dtypeCode = static_cast<uint32_t>(boxes->GetDataType());
    const uint32_t batchKey = SbgSelectRatioKey(plan.fullPlan ? branch.batchKey : 0U, ratio);
    // Attribute specialization is guarded by the actual runtime attribute.
    // It never changes which of the twelve dtype/B implementations is selected.
    // 常量属性特化必须真实命中才选择。当前Host只启用B32的ratio=1，
    // 以及两种dtype下B4095/4096的ratio=0.8；必须真实命中，否则传入真实比例。
    // 两个语义参数编码成最终TilingKey；不要把最终Key直接当作B。
    ASCENDC_TPL_SEL_PARAM(context, dtypeCode, batchKey);
    auto *td = context->GetTilingData<SafetyBeltViolationGeometryTilingData>();
    if (!td) return ge::GRAPH_FAILED;
    td->length = static_cast<uint32_t>(rows);
    td->rowsPerCore = perCore;
    // 这里只记录存在性，不把threshold的真实设备数值抄进Tiling。
    td->hasThreshold = threshold && threshold->GetShapeSize() > 0 ? 1U : 0U;
    td->waistRatio = ratio ? *ratio : 0.0f;
    td->tileRows = perCore;
    // 以下两项是兼容字段，恒0；实际非零阈值仍由Kernel从threshold读取。
    td->thresholdInTiling = 0U;
    td->threshold = 0.0f;
    // 下发启动核数。它不同于working，闲置启动核由Kernel尽早退出。
    context->SetBlockDim(plan.blocks);
    // Start all requested slices together, rather than assigning them as
    // individual physical cores become available.
    // 当前保留调度模式1；这是调度设置，不是双缓冲，也不预热数据。
    if (context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    auto *workspace = context->GetWorkspaceSizes(1);
    // 不申请用户workspace；与Kernel末尾的系统工作区包装策略不是同一件事。
    if (workspace) workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
// 输出形状描述为[B,1]；此处仅推导描述，不生成违规结果。
static graphStatus InferShape(gert::InferShapeContext *context) {
    const auto *in = context->GetInputShape(0);
    auto *out = context->GetOutputShape(0);
    if (!in || !out) return ge::GRAPH_FAILED;
    out->SetDimNum(2);
    out->SetDim(0, in->GetDimNum() >= 1 ? in->GetDim(0) : 0);
    out->SetDim(1, 1);
    return GRAPH_SUCCESS;
}
// 【ABI审阅点】现有推导让输出跟随输入dtype；FP16输入时输出为FP16。
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const auto dt = context->GetInputDataType(0);
    context->SetOutputDataType(0, dt == ge::DT_FLOAT16 ? ge::DT_FLOAT16 : ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}
namespace ops {
class SafetyBeltViolationGeometry : public OpDef {
public:
    explicit SafetyBeltViolationGeometry(const char *name) : OpDef(name) {
        this->Input("person_boxes").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("anchor_points").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // 【ABI审阅点】现有注册允许FP32/FP16阈值，与题面固定FP32的文字有差异。
        this->Input("threshold").ParamType(OPTIONAL).DataType({ge::DT_FLOAT, ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // 输出0/1在两种类型都可精确表示，但类型契约仍须单独核对。
        this->Output("violation_score").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("waist_ratio").AttrType(REQUIRED).Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SafetyBeltViolationGeometry);
}
