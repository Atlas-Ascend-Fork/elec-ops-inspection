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
// Host side tiling for the TreeLineMinDistance operator.
#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/tree_line_min_distance_tiling.h"
#include "../op_kernel/tiling_key_tree_line_min_distance.h"

namespace optiling {

static bool CheckShapes(const gert::Shape &points, const gert::Shape &lines)
{
    if (points.GetDimNum() != 3U || points.GetDim(2) != 3) {
        return false;
    }
    const bool packed = lines.GetDimNum() == 3U && lines.GetDim(2) == 6;
    const bool endpoints = lines.GetDimNum() == 4U && lines.GetDim(2) == 2 && lines.GetDim(3) == 3;
    if (!packed && !endpoints) {
        return false;
    }
    return points.GetDim(0) < 0 || lines.GetDim(0) < 0 || points.GetDim(0) == lines.GetDim(0);
}

static ge::graphStatus FillTiling(gert::TilingContext *context, TreeLineMinDistanceTilingData *tiling)
{
    const auto *pointDesc = context->GetInputShape(0);
    const auto *lineDesc = context->GetInputShape(1);
    if (pointDesc == nullptr || lineDesc == nullptr || tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &points = pointDesc->GetStorageShape();
    const auto &lines = lineDesc->GetStorageShape();
    if (!CheckShapes(points, lines)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batch = points.GetDim(0), n = points.GetDim(1), m = lines.GetDim(1);
    if (batch < 0 || n < 0 || m < 1 || batch > UINT32_MAX || n > UINT32_MAX || m > UINT32_MAX - 7U ||
        lines.GetDim(0) != batch || static_cast<uint64_t>(batch) * n > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    tiling->batch = static_cast<uint32_t>(batch);
    tiling->n = static_cast<uint32_t>(n);
    tiling->m = static_cast<uint32_t>(m);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckLocalMemory(gert::TilingContext *context, const TreeLineMinDistanceTilingData *tiling,
                                       platform_ascendc::PlatformAscendC &platform)
{
    const auto *desc = context->GetInputDesc(0);
    if (desc == nullptr || (desc->GetDataType() != ge::DT_FLOAT && desc->GetDataType() != ge::DT_FLOAT16)) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t width = ge::GetSizeByDataType(desc->GetDataType());
    const uint64_t aligned = (static_cast<uint64_t>(tiling->m) + 7U) / 8U * 8U;
    const uint64_t buffers = aligned * 68U +
                             (aligned + 255U) / 256U * 32U + 32U + 64U + 16U * width + 64U;
    uint64_t capacity = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, capacity);
    return buffers <= capacity ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

static ge::graphStatus FillBlockDim(TreeLineMinDistanceTilingData *tiling, int32_t coreNumAiv)
{
    if (coreNumAiv < 1) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t jobs = (static_cast<uint64_t>(tiling->batch) * tiling->n + 15U) / 16U;
    const uint32_t cores = static_cast<uint32_t>(coreNumAiv);
    tiling->blocks = jobs == 0U ? 1U : ((jobs < cores) ? static_cast<uint32_t>(jobs) : cores);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto *tiling = context->GetTilingData<TreeLineMinDistanceTilingData>();

    if (FillTiling(context, tiling) != ge::GRAPH_SUCCESS ||
        CheckLocalMemory(context, tiling, platform) != ge::GRAPH_SUCCESS ||
        FillBlockDim(tiling, platform.GetCoreNumAiv()) != ge::GRAPH_SUCCESS) {
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
    const auto *lineShape = context->GetInputShape(1);
    if (inputShape == nullptr || lineShape == nullptr || !optiling::CheckShapes(*inputShape, *lineShape)) {
        return GRAPH_FAILED;
    }
    for (uint32_t i = 0U; i < 2U; ++i) {
        auto *outputShape = context->GetOutputShape(i);
        if (outputShape == nullptr) {
            return GRAPH_FAILED;
        }
        outputShape->SetDimNum(2);
        outputShape->SetDim(0, inputShape->GetDim(0));
        outputShape->SetDim(1, inputShape->GetDim(1));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, DT_INT32);
    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class TreeLineMinDistance : public OpDef {
public:
    explicit TreeLineMinDistance(const char *name) : OpDef(name)
    {
        InitPorts();
        InitInfer();
        InitAICore();
    }

private:
    void AddFloatInput(const char *name)
    {
        this->Input(name)
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    }

    void AddFloatOutput(const char *name)
    {
        this->Output(name)
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    }

    void AddIndexOutput(const char *name)
    {
        this->Output(name)
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
    }

    void InitPorts()
    {
        AddFloatInput("tree_points");
        AddFloatInput("line_segments");
        AddFloatOutput("min_distance");
        AddIndexOutput("nearest_segment_idx");
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

OP_ADD(TreeLineMinDistance);

} // namespace ops
