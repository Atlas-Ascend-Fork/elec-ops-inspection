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
// Host side tiling for the LinalgDet operator.
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/linalg_det_tiling.h"
#include "../op_kernel/tiling_key_linalg_det.h"

namespace optiling {

static bool CheckMatrixShape(const gert::Shape &shape)
{
    const size_t dimNum = shape.GetDimNum();
    if (dimNum < 2U) {
        return false;
    }
    const int64_t n = shape.GetDim(dimNum - 1U);
    if (n < 0 || n > UINT32_MAX - 7U || shape.GetDim(dimNum - 2U) != n) {
        return false;
    }
    uint64_t batch = 1U;
    for (size_t i = 0U; i + 2U < dimNum; ++i) {
        const int64_t dimension = shape.GetDim(i);
        if (dimension < 0 || dimension > UINT32_MAX - 7U) {
            return false;
        }
        batch *= static_cast<uint64_t>(dimension);
        if (batch > UINT32_MAX - 7U) {
            return false;
        }
    }
    return true;
}

static void FillShapeTiling(gert::TilingContext *context, LinalgDetTilingData *tiling)
{
    const auto &inputShape = context->GetInputShape(0)->GetStorageShape();
    const uint32_t dimNum = inputShape.GetDimNum();
    tiling->n = inputShape.GetDim(dimNum - 1U);
    tiling->batch = 1U;
    for (uint32_t i = 0U; i + 2U < dimNum; ++i) {
        tiling->batch *= inputShape.GetDim(i);
    }
    tiling->k = 0U;
    tiling->broadcastB = 0U;
    tiling->p = (tiling->n + 7U) / 8U * 8U;
    tiling->q = (tiling->k + 7U) / 8U * 8U;
}

static void FillBlockDim(LinalgDetTilingData *tiling, int32_t coreNumAiv)
{
    const uint32_t tasks = (tiling->batch + 7U) / 8U;
    const uint32_t cores = static_cast<uint32_t>(coreNumAiv);
    tiling->blocks = (tasks < cores) ? tasks : cores;
    if (tiling->blocks == 0U) {
        tiling->blocks = 1U;
    }
}

static void FillWorkspace(gert::TilingContext *context, const LinalgDetTilingData *tiling,
                          uint64_t libApiSize)
{
    const uint64_t elements = static_cast<uint64_t>(tiling->batch) * tiling->n *
                              (tiling->p + tiling->q);
    context->GetWorkspaceSizes(1)[0] = libApiSize + elements * 4U + 64U;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *shape = context->GetInputShape(0);
    if (shape == nullptr || !CheckMatrixShape(shape->GetStorageShape())) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto *tiling = context->GetTilingData<LinalgDetTilingData>();

    FillShapeTiling(context, tiling);
    uint64_t ubSize = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t ubBytes = sizeof(float) * (4ULL * tiling->p + 8ULL * tiling->n + 8U);
    if (ubBytes > ubSize) {
        return ge::GRAPH_FAILED;
    }
    FillBlockDim(tiling, platform.GetCoreNumAiv());

    const uint32_t DT_A = static_cast<uint32_t>(ge::DT_FLOAT);
    ASCENDC_TPL_SEL_PARAM(context, DT_A);

    context->SetBlockDim(tiling->blocks);
    context->GetRawTilingData()->SetDataSize(sizeof(LinalgDetTilingData));
    FillWorkspace(context, tiling, platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *inputShape = context->GetInputShape(0);
    auto *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr || !optiling::CheckMatrixShape(*inputShape)) {
        return GRAPH_FAILED;
    }
    const uint32_t dimNum = inputShape->GetDimNum();
    outputShape->SetDimNum(dimNum - 2U);
    for (uint32_t i = 0U; i + 2U < dimNum; ++i) {
        outputShape->SetDim(i, inputShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, DT_FLOAT);
    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class LinalgDet : public OpDef {
public:
    explicit LinalgDet(const char *name) : OpDef(name)
    {
        InitPorts();
        InitInfer();
        InitAICore();
    }

private:
    void InitPorts()
    {
        this->Input("A")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("det")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
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

OP_ADD(LinalgDet);

} // namespace ops
