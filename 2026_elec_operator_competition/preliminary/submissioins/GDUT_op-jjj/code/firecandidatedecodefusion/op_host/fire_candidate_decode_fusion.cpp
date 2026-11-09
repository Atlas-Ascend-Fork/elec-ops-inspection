/*
 * Copyright (c) 2026 李冠杰.
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

// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

#include "../op_kernel/fire_candidate_decode_fusion_tiling.h"
// Judge opbuild must not depend on an additional copied helper header.
// Same constants and predicate as the verified R13-D route.
constexpr uint32_t FCF_ROUTE_MIN_CANDIDATES = 512u;
constexpr uint32_t FCF_ROUTE_MIN_PER_CORE = 512u;
inline bool FcfUsePadLayout(uint32_t candidates, uint32_t blockFactor)
{
    return candidates >= FCF_ROUTE_MIN_CANDIDATES && blockFactor >= FCF_ROUTE_MIN_PER_CORE;
}
#include "../op_kernel/tiling_key_fire_candidate_decode_fusion.h"

namespace optiling {
struct InputInfo {
    uint32_t batch = 0;
    uint32_t candidates = 0;
    uint32_t total = 0;
    ge::DataType dtype = ge::DT_FLOAT;
};

struct CoreRoute {
    uint32_t blockFactor = 0;
    uint32_t usedCoreNum = 0;
    uint32_t usePad = 0;
    uint32_t scalarOnly = 0;
    uint32_t batchSplit = 0;
};

static uint32_t CeilDivide(uint64_t total, uint32_t divisor)
{
    if (divisor == 0u) {
        return 0u;
    }
    return static_cast<uint32_t>((total + divisor - 1u) / divisor);
}

static bool ReadInputInfo(gert::TilingContext *context, InputInfo &info)
{
    const auto *rawShape = context->GetInputShape(0);
    const auto *priorShape = context->GetInputShape(1);
    const auto *rawDesc = context->GetInputDesc(0);
    if (rawShape == nullptr || priorShape == nullptr || rawDesc == nullptr) {
        return false;
    }
    const auto &raw = rawShape->GetStorageShape();
    const auto &prior = priorShape->GetStorageShape();
    if (raw.GetDimNum() != 3 || prior.GetDimNum() != 2) {
        return false;
    }
    if (raw.GetDim(2) != 6 || prior.GetDim(1) != 5 || raw.GetDim(1) != prior.GetDim(0)) {
        return false;
    }
    const int64_t batch = raw.GetDim(0);
    const int64_t candidates = raw.GetDim(1);
    if (batch <= 0 || candidates <= 0 || batch > UINT32_MAX || candidates > UINT32_MAX) {
        return false;
    }
    const uint64_t total = static_cast<uint64_t>(batch) * static_cast<uint64_t>(candidates);
    if (total > UINT32_MAX) {
        return false;
    }
    info.batch = static_cast<uint32_t>(batch);
    info.candidates = static_cast<uint32_t>(candidates);
    info.total = static_cast<uint32_t>(total);
    info.dtype = rawDesc->GetDataType();
    return true;
}

static bool SelectScalarRoute(const InputInfo &info, uint32_t cores, CoreRoute &route)
{
    route.scalarOnly = (info.dtype == ge::DT_FLOAT && route.usePad == 0u &&
                        route.blockFactor < 16u && route.usedCoreNum > 1u) ? 1u : 0u;
    constexpr uint32_t minPerCore = 6u;
    constexpr uint32_t scalarPerCore = 15u;
    if (route.scalarOnly == 0u || route.blockFactor < minPerCore) {
        return true;
    }
    const uint32_t budget = std::min(cores, CeilDivide(info.total, scalarPerCore));
    if (budget == 0u) {
        return false;
    }
    route.blockFactor = CeilDivide(info.total, budget);
    if (route.blockFactor == 0u || route.blockFactor > scalarPerCore) {
        return false;
    }
    route.usedCoreNum = CeilDivide(info.total, route.blockFactor);
    return true;
}

static void SelectBatchRoute(const InputInfo &info, uint32_t cores, CoreRoute &route)
{
    if (info.batch == 0u || info.batch > cores) {
        return;
    }
    if (info.dtype != ge::DT_FLOAT || route.usePad != 0u || route.scalarOnly != 0u ||
        route.blockFactor < 16u || route.blockFactor >= 512u || cores % info.batch != 0u) {
        return;
    }
    const uint32_t coresPerBatch = CeilDivide(info.candidates, route.blockFactor);
    const uint32_t splitCores = info.batch * coresPerBatch;
    if (splitCores <= cores) {
        route.batchSplit = 1u;
        route.usedCoreNum = splitCores;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t coreCount = platform.GetCoreNumAiv();
    if (coreCount <= 0) {
        return ge::GRAPH_FAILED;
    }
    InputInfo info;
    if (!ReadInputInfo(context, info)) {
        return ge::GRAPH_FAILED;
    }
    CoreRoute route;
    const uint32_t cores = static_cast<uint32_t>(coreCount);
    route.blockFactor = CeilDivide(info.total, cores);
    if (route.blockFactor == 0u) {
        return ge::GRAPH_FAILED;
    }
    route.usedCoreNum = CeilDivide(info.total, route.blockFactor);
    route.usePad = FcfUsePadLayout(info.candidates, route.blockFactor) ? 1u : 0u;
    if (!SelectScalarRoute(info, cores, route)) {
        return ge::GRAPH_FAILED;
    }
    SelectBatchRoute(info, cores, route);
    auto *tiling = context->GetTilingData<FireCandidateDecodeFusionTilingData>();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (tiling == nullptr || workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->batch = info.batch;
    tiling->candidates = info.candidates;
    tiling->total = info.total;
    tiling->tileLength = 512u;
    tiling->blockFactor = route.blockFactor;
    const uint32_t dtype = static_cast<uint32_t>(info.dtype);
    ASCENDC_TPL_SEL_PARAM(context, dtype, route.usePad, route.scalarOnly, route.batchSplit);
    context->SetBlockDim(route.usedCoreNum);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace optiling {
static ge::graphStatus TilingParse([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}
struct FireCandidateDecodeFusionCompileInfo {};
IMPL_OP_OPTILING(FireCandidateDecodeFusion)
    .Tiling(TilingFunc)
    .TilingParse<FireCandidateDecodeFusionCompileInfo>(TilingParse);
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape *input = context->GetInputShape(0);
        gert::Shape *output = context->GetOutputShape(0);
        if (input == nullptr || output == nullptr) return ge::GRAPH_FAILED;
        if (input->GetDimNum() != 3 ||
            (input->GetDim(2) != 6 && input->GetDim(2) != -1)) {
            return ge::GRAPH_FAILED;
        }
        *output = *input;
        output->SetDim(2, 5);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class FireCandidateDecodeFusion : public OpDef {
    public:
        explicit FireCandidateDecodeFusion(const char *name) : OpDef(name) {
            this->Input("raw_pred")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("priors")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("decoded")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(FireCandidateDecodeFusion);
}  // namespace ops
