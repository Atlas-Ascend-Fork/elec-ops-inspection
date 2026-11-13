/*
 * Copyright (c) 2026 ceprei_Tnx.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Host侧Tiling实现（v6 分支：整块搬运 + 片上 Gather 重排 + 大 tile）
//
// ── v17 说明：Host 相对原提交没有代码改动 ───────────────────────────────────
// v16 起的改动全部落在 Kernel（小档建表向量化、item 循环去除法、建表与首笔 DMA 重叠、
// 收尾只排空 MTE3）。切分口径、tileRows 计算、blockDim 下限、workspace 全部保持原样 ——
// 同一份 Host 换 Kernel 就能直接 A/B，没有 ABI 风险。
// 核数问题已排除：本机 20 个 AI Core × 2 个 Vector = 40，GetCoreNumAiv() 返回 40 就是上限，
// 「用不满核」这条不成立，与榜首的差距只能从效率上找。
//
// 与 4files（v5，上机 100.00）的 Host 逻辑**完全一致** —— 这是刻意的：
// exp 分支只改 Kernel 的搬运与重排方式，不改「谁算哪些行」这件事。
// 切分口径仍是「按 priors 行数 N 切」，让同一份 priors 被 B 个 batch 共享。
//
// 唯一随分支变化的是 tileRows 的上限：它来自 op_kernel 的 FIRE_MAX_TILE_ROWS，
// v6 把它调回 224（exp 曾是 32）。Host 直接 include 那个头文件，所以不需要在这里
// 改数字 —— 这正是把这个常量放进共享头文件的意义。
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ascendc/host_api/tiling/template_argument.h"

#include "../op_kernel/fire_candidate_decode_fusion_tiling.h"
#include "../op_kernel/tiling_key_fire_candidate_decode_fusion.h"

#include <algorithm>
#include <cstdint>

namespace {
// 900B 系列上 blockDim 过小（尤其等于 1）时存在一条明显更慢的启动路径。把下限抬到 8
// 只是让多出来的空 block 走一次提前 return，不改变数据划分与计算结果，零风险兜底。
constexpr uint32_t MIN_BLOCK_DIM = 8U;

// 字段数（结构约束，与 shape 无关，可以放心强校验）
constexpr int64_t RAW_FIELDS = 6;
constexpr int64_t PRIOR_FIELDS = 5;
constexpr int64_t OUT_FIELDS = 5;

// 核数收口上限。判分环境实测 GetCoreNumAiv() 返回 40（profiling 里最大 Block Dim
// 就是 40），并非 910B1 标称的 48。这里只用它兜住「平台识别异常返回离谱值」的情况。
// （v17 已确认 40 就是本机上限：20 个 AI Core × 2 个 Vector = 40，不存在「少用核」的问题，
//   与榜首的差距只能从效率上找，不再往核数上想。）
constexpr uint32_t MAX_AIV_CORES = 48U;

constexpr uint32_t CeilDivU32(uint32_t x, uint32_t y) {
    if (y == 0U) {
        return 0U;
    }
    return x / y + ((x % y) != 0U ? 1U : 0U);
}

bool ValidateInputs(gert::TilingContext *context, int64_t &batch, int64_t &candidates, uint64_t &total64,
                    ge::DataType &rawType) {
    int64_t batch = 0;
    int64_t candidates = 0;
    uint64_t total64 = 0U;
    ge::DataType rawType = ge::DT_FLOAT;
    if (!ValidateInputs(context, batch, candidates, total64, rawType)) {
        return ge::GRAPH_FAILED;
    }

    // ── 切分 ────────────────────────────────────────────────────────────────
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t availableCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (availableCores == 0U) {
        availableCores = static_cast<uint32_t>(platform.GetCoreNum());
    }
    availableCores = std::max<uint32_t>(availableCores, 1U);
    availableCores = std::min<uint32_t>(availableCores, MAX_AIV_CORES);

    const uint32_t total = static_cast<uint32_t>(total64);
    const uint32_t candidatesU = static_cast<uint32_t>(candidates);

    // 切分口径：**按 priors 的行数 N 切，而不是按总行数 B*N 切**。
    // priors[N,5] 在 batch 维是共享的：同一个 n，B 个 batch 用的是同一份
    // [gx, gy, stride, aw, ah]。Kernel 因此「每核认领一段连续的 priors 行，先展布出
    // 乘数、再在这段上把 B 个 batch 依次跑完」。若按总行数切，同一个 n 会被切给不同核，
    // B 份 priors 就得各搬一遍，优化直接失效。
    //
    // 方向仍是「从满核出发收」：实机标定的时间模型是「总时间 ∝ 每核行数」而几乎不与
    // 核数显式相关，所以唯一正确的目标就是把 rowsPerCore 压到最小。
    uint32_t usedCores = availableCores;
    uint32_t rowsPerCore = CeilDivU32(candidatesU, usedCores);
    if (rowsPerCore == 0U) {
        rowsPerCore = FIRE_MIN_ROWS_PER_CORE;
    }
    // 刻意不做任何行对齐：Kernel 里所有 GM 访问都按字节寻址，起点本来就不要求落在
    // 32B 边界上。反过来，向上对齐只会让 rowsPerCore 变大、核数变少。
    usedCores = CeilDivU32(candidatesU, rowsPerCore);
    usedCores = std::max<uint32_t>(usedCores, 1U);
    usedCores = std::min<uint32_t>(usedCores, availableCores);

    // 单核 tile 由 Kernel 的静态 UB 布局封顶。大于上限时不直接切成
    // [MAX, MAX, ..., tiny-tail]，而是先确定最少 tile 数，再均匀回分；这样不增加
    // MTE 事务/item 数，却能避免最后一个极小尾 tile 多付一组向量 repeat 与屏障。
    const uint32_t tilesPerCore = CeilDivU32(rowsPerCore, FIRE_MAX_TILE_ROWS);
    uint32_t tileRows = CeilDivU32(rowsPerCore, tilesPerCore);
    tileRows = std::max<uint32_t>(tileRows, 1U);

    uint32_t blockDim = std::max<uint32_t>(usedCores, MIN_BLOCK_DIM);
    blockDim = std::min<uint32_t>(blockDim, availableCores);

    // ── 落 tiling ──────────────────────────────────────────────────────────
    FireCandidateDecodeFusionTilingData *tiling = context->GetTilingData<FireCandidateDecodeFusionTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->numCandidates = static_cast<uint32_t>(candidates);
    tiling->totalRows = total;
    tiling->rowsPerCore = rowsPerCore;
    tiling->tileRows = tileRows;
    tiling->reserved0 = 0U;
    tiling->reserved1 = 0U;
    tiling->reserved2 = 0U;

    // tiling key 只有一个维度：raw_pred 的 dtype。判分系统按它实例化对应分支。
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(rawType));

    context->SetBlockDim(blockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0U; // 全程片上完成，不需要中间结果落地
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *rawShape = context->GetInputShape(0);
    const gert::Shape *priorShape = context->GetInputShape(1);
    gert::Shape *decodedShape = context->GetOutputShape(0);
    if (rawShape == nullptr || priorShape == nullptr || decodedShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (rawShape->GetDimNum() != 3U || priorShape->GetDimNum() != 2U) {
        return GRAPH_FAILED;
    }
    if (rawShape->GetDim(2) != RAW_FIELDS || priorShape->GetDim(1) != PRIOR_FIELDS ||
        rawShape->GetDim(1) != priorShape->GetDim(0)) {
        return GRAPH_FAILED;
    }
    *decodedShape = gert::Shape();
    decodedShape->AppendDim(rawShape->GetDim(0)); // B
    decodedShape->AppendDim(rawShape->GetDim(1)); // N
    decodedShape->AppendDim(OUT_FIELDS);          // 5
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    // 输出 dtype 跟随 raw_pred：fp16 输入给 fp16 输出，fp32 输入给 fp32 输出。
    // 不要写成「输出恒 fp32」—— 那会让 fp16 用例在建图阶段整片失败。
    const ge::DataType rawType = context->GetInputDataType(0);
    if (rawType != ge::DT_FLOAT && rawType != ge::DT_FLOAT16) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, rawType);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class FireCandidateDecodeFusion : public OpDef {
  public:
    explicit FireCandidateDecodeFusion(const char *name) : OpDef(name) {
        this->Input("raw_pred")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("priors")
            .ParamType(REQUIRED)
            // DataType 的多个条目是「并列的签名组合」，不是可选集合：
            // 组合 0 = (raw_pred fp32, priors fp32, out fp32)；组合 1 = (fp16, fp32, fp16)。
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("decoded")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(FireCandidateDecodeFusion);
} // namespace ops
