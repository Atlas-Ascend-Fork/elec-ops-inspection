/*
 * Copyright (c) 2026 aodebiao.
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
// Host side tiling for the FireCandidateDecodeFusion operator.
#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fire_candidate_decode_fusion_tiling.h"
#include "../op_kernel/tiling_key_fire_candidate_decode_fusion.h"

namespace optiling {

static bool CheckShapes(const gert::Shape &raw, const gert::Shape &priors)
{
    if (raw.GetDimNum() != 3U || raw.GetDim(2) != 6 ||
        priors.GetDimNum() != 2U || priors.GetDim(1) != 5) {
        return false;
    }
    return raw.GetDim(1) < 0 || priors.GetDim(0) < 0 || raw.GetDim(1) == priors.GetDim(0);
}

static ge::graphStatus FillShapeTiling(gert::TilingContext *context, FireCandidateDecodeFusionTilingData *tiling)
{
    const auto *raw = context->GetInputShape(0);
    const auto *priors = context->GetInputShape(1);
    if (raw == nullptr || priors == nullptr || tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &inputShape = raw->GetStorageShape();
    if (!CheckShapes(inputShape, priors->GetStorageShape())) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = inputShape.GetDim(0);
    const int64_t count = inputShape.GetDim(1);
    if (batch < 0 || count < 0 || batch > UINT32_MAX || count > UINT32_MAX - 63U ||
        priors->GetStorageShape().GetDim(0) != count) {
        return ge::GRAPH_FAILED;
    }
    tiling->batch = static_cast<uint32_t>(batch);
    tiling->n = static_cast<uint32_t>(count);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillBlockDim(gert::TilingContext *context, FireCandidateDecodeFusionTilingData *tiling,
                                   int32_t coreNumAiv)
{
    const auto *inputDesc = context->GetInputDesc(0);
    if (coreNumAiv < 1 || inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t jobs = static_cast<uint64_t>(tiling->batch) * ((tiling->n + 63U) / 64U);
    if (jobs > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t cores = static_cast<uint32_t>(coreNumAiv);
    tiling->blocks = jobs == 0U ? 1U : ((jobs < cores) ? static_cast<uint32_t>(jobs) : cores);
    const uint32_t bytes = ge::GetSizeByDataType(inputDesc->GetDataType());
    if (((static_cast<uint64_t>(tiling->n) * 5U * bytes) % 32U) != 0U) {
        tiling->blocks = 1U;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto *tiling = context->GetTilingData<FireCandidateDecodeFusionTilingData>();

    if (FillShapeTiling(context, tiling) != ge::GRAPH_SUCCESS ||
        FillBlockDim(context, tiling, platform.GetCoreNumAiv()) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t inputDtype = static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, inputDtype);

    context->SetBlockDim(tiling->blocks);
    context->GetWorkspaceSizes(1)[0] = 0U;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *inputShape = context->GetInputShape(0);
    const auto *priorsShape = context->GetInputShape(1);
    auto *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || priorsShape == nullptr || outputShape == nullptr ||
        !optiling::CheckShapes(*inputShape, *priorsShape)) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    outputShape->SetDim(2, 5);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class FireCandidateDecodeFusion : public OpDef {
public:
    explicit FireCandidateDecodeFusion(const char *name) : OpDef(name)
    {
        InitPorts();
        InitInfer();
        InitAICore();
    }

private:
    void InitPorts()
    {
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
    }

    void InitInfer()
    {
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    }

    void InitAICore()
    {
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(FireCandidateDecodeFusion);

} // namespace ops
