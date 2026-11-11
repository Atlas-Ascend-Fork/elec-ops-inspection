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
#include <cstdint>
#include <cstddef>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/safety_belt_violation_geometry_tiling.h"
#include "../op_kernel/tiling_key_safety_belt_violation_geometry.h"

namespace optiling {

static constexpr uint32_t SAMPLES_PER_TILE = 128U;
static constexpr uint32_t SMALL_SAMPLES_PER_TILE = 64U;
static constexpr uint32_t PERSON_ELEMENTS = 4U;
static constexpr uint32_t ANCHOR_ELEMENTS = 2U;
static constexpr uint32_t MAX_BATCH = 4096U;

struct TilingParams {
    const gert::StorageShape *personBoxes = nullptr;
    const gert::StorageShape *anchorPoints = nullptr;
    const gert::StorageShape *threshold = nullptr;
    ge::DataType dtypePerson = ge::DT_UNDEFINED;
    ge::DataType dtypeAnchor = ge::DT_UNDEFINED;
    ge::DataType dtypeThreshold = ge::DT_UNDEFINED;
    uint32_t batch = 0U;
    uint32_t hasThreshold = 0U;
    uint32_t thresholdIsFp16 = 0U;
    uint32_t outputIsFp16 = 0U;
    float waistRatio = 0.0f;
    bool small = false;
};

static ge::graphStatus CollectInputs(gert::TilingContext *context, TilingParams &params)
{
    params.personBoxes = context->GetInputShape(0);
    params.anchorPoints = context->GetInputShape(1);
    params.threshold = context->GetOptionalInputShape(2);
    const auto *personDesc = context->GetInputDesc(0);
    const auto *anchorDesc = context->GetInputDesc(1);
    const auto *thresholdDesc = context->GetOptionalInputDesc(2);
    if (params.personBoxes == nullptr || params.anchorPoints == nullptr ||
        personDesc == nullptr || anchorDesc == nullptr ||
        (params.threshold != nullptr && thresholdDesc == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    params.dtypePerson = personDesc->GetDataType();
    params.dtypeAnchor = anchorDesc->GetDataType();
    if (thresholdDesc != nullptr) {
        params.dtypeThreshold = thresholdDesc->GetDataType();
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckPersonShape(const TilingParams &params)
{
    const int64_t personLength = params.personBoxes->GetStorageShape().GetShapeSize();
    if (personLength <= 0 || personLength > MAX_BATCH * PERSON_ELEMENTS ||
        (personLength % PERSON_ELEMENTS) != 0) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAnchorShape(const TilingParams &params)
{
    const int64_t anchorLength = params.anchorPoints->GetStorageShape().GetShapeSize();
    if (anchorLength != params.batch * ANCHOR_ELEMENTS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckPersonDtype(const TilingParams &params)
{
    const ge::DataType dtypeAnchor = params.dtypeAnchor;
    if (params.dtypePerson != dtypeAnchor) {
        return ge::GRAPH_FAILED;
    }
    if (params.dtypePerson != ge::DT_FLOAT && params.dtypePerson != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckThreshold(const gert::StorageShape *threshold, TilingParams &params)
{
    if (threshold == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (threshold->GetStorageShape().GetShapeSize() != 1) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dtypeThreshold = params.dtypeThreshold;
    if (dtypeThreshold == ge::DT_FLOAT16) {
        params.thresholdIsFp16 = 1U;
    } else if (dtypeThreshold == ge::DT_FLOAT) {
        params.thresholdIsFp16 = 0U;
    } else {
        return ge::GRAPH_FAILED;
    }
    params.hasThreshold = 1U;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputDtype(gert::TilingContext *context, TilingParams &params)
{
    const auto *outputDesc = context->GetOutputDesc(0);
    if (outputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dtypeOutput = outputDesc->GetDataType();
    if (dtypeOutput == ge::DT_FLOAT16) {
        params.outputIsFp16 = 1U;
    } else if (dtypeOutput == ge::DT_FLOAT) {
        params.outputIsFp16 = 0U;
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWaistRatio(gert::TilingContext *context, TilingParams &params)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const float *waistRatioPtr = attrs->GetFloat(0);
    if (waistRatioPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    params.waistRatio = *waistRatioPtr;
    if (params.waistRatio <= 0.0f || params.waistRatio > 1.0f) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillTilingData(gert::TilingContext *context, const TilingParams &params)
{
    SafetyBeltViolationGeometryTilingData *tiling =
        context->GetTilingData<SafetyBeltViolationGeometryTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->batch = params.batch;
    tiling->hasThreshold = params.hasThreshold;
    tiling->thresholdIsFp16 = params.thresholdIsFp16;
    tiling->outputIsFp16 = params.outputIsFp16;
    tiling->waistRatio = params.waistRatio;
    tiling->thresholdDefault = 0.0f;
    return ge::GRAPH_SUCCESS;
}

static void SetBlockDim(gert::TilingContext *context, uint32_t batch, int32_t numCoresAiv, bool small)
{
    const uint32_t samplesPerTile = small ? SMALL_SAMPLES_PER_TILE : SAMPLES_PER_TILE;
    uint32_t tileCount = (batch + samplesPerTile - 1U) / samplesPerTile;
    tileCount = (tileCount == 0U) ? 1U : tileCount;
    const uint32_t maxCores = static_cast<uint32_t>(numCoresAiv);
    uint32_t blockDim = (tileCount > maxCores) ? maxCores : tileCount;
    blockDim = (blockDim == 0U) ? 1U : blockDim;
    context->SetBlockDim(blockDim);
}

static ge::graphStatus ValidateParams(gert::TilingContext *context, TilingParams &params)
{
    ge::graphStatus status = CollectInputs(context, params);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    status = CheckPersonShape(params);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    const uint32_t personLength = static_cast<uint32_t>(params.personBoxes->GetStorageShape().GetShapeSize());
    params.batch = personLength / PERSON_ELEMENTS;
    if (params.batch == 0U || params.batch > MAX_BATCH) {
        return ge::GRAPH_FAILED;
    }
    status = CheckAnchorShape(params);
    if (status == ge::GRAPH_SUCCESS) {
        status = CheckPersonDtype(params);
    }
    if (status == ge::GRAPH_SUCCESS) {
        status = CheckThreshold(params.threshold, params);
    }
    if (status == ge::GRAPH_SUCCESS) {
        status = CheckOutputDtype(context, params);
    }
    if (status == ge::GRAPH_SUCCESS) {
        status = CheckWaistRatio(context, params);
    }
    return status;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t numCoresAiv = platform.GetCoreNumAiv();
    numCoresAiv = (numCoresAiv <= 0) ? 1 : numCoresAiv;

    TilingParams params;
    ge::graphStatus status = ValidateParams(context, params);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    params.small = params.batch <= static_cast<uint32_t>(numCoresAiv) * SMALL_SAMPLES_PER_TILE;
    const uint32_t DT_PERSON_BOXES = static_cast<uint32_t>(params.dtypePerson);
    const bool HAS_THRESHOLD = params.hasThreshold != 0U;
    const bool THRESHOLD_HALF = params.thresholdIsFp16 != 0U;
    const bool OUTPUT_HALF = params.outputIsFp16 != 0U;
    const bool SMALL = params.small;
    ASCENDC_TPL_SEL_PARAM(context, DT_PERSON_BOXES, HAS_THRESHOLD, THRESHOLD_HALF, OUTPUT_HALF, SMALL);

    status = FillTilingData(context, params);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    SetBlockDim(context, params.batch, numCoresAiv, params.small);

    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0U;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *personShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (personShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    if (personShape->GetDimNum() != 2U || personShape->GetDim(1) != 4) {
        return GRAPH_FAILED;
    }
    *outputShape = *personShape;
    outputShape->SetDim(1, 1);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    ge::DataType outputType = context->GetOutputDataType(0);
    if (outputType == DT_UNDEFINED) {
        outputType = context->GetInputDataType(0);
    }
    if (outputType != DT_FLOAT && outputType != DT_FLOAT16) {
        return GRAPH_FAILED;
    }
    return context->SetOutputDataType(0, outputType);
}

} // namespace ge

namespace ops {

class SafetyBeltViolationGeometry : public OpDef {
public:
    explicit SafetyBeltViolationGeometry(const char *name) : OpDef(name)
    {
        InitInputs();
        InitOutputs();
        InitAttrs();
        InitAICore();
    }

private:
    void InitInputs()
    {
        this->Input("person_boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("anchor_points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("threshold")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
    }

    void InitOutputs()
    {
        this->Output("violation_score")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
    }

    void InitAttrs()
    {
        this->Attr("waist_ratio").AttrType(REQUIRED).Float();
    }

    void InitAICore()
    {
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(SafetyBeltViolationGeometry);

} // namespace ops
