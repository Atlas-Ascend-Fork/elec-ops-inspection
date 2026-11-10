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
// Host side tiling for the LinalgSolve operator.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/linalg_solve_tiling.h"
#include "../op_kernel/tiling_key_linalg_solve.h"
#include "linalg_solve_shape.h"

namespace optiling {

static bool ReadLeft(const gert::RuntimeAttrs *attrs)
{
    const bool *left = attrs == nullptr ? nullptr : attrs->GetBool(0);
    return left == nullptr || *left;
}

static void FillBlockDim(LinalgSolveTilingData *tiling, int32_t coreNumAiv)
{
    const uint32_t tasks = tiling->batch;
    const uint32_t cores = static_cast<uint32_t>(coreNumAiv);
    tiling->blocks = (tasks < cores) ? tasks : cores;
    if (tiling->blocks == 0U || ((static_cast<uint64_t>(tiling->n) * tiling->k) % 8U) != 0U) {
        tiling->blocks = 1U;
    }
}

static void FillWorkspace(gert::TilingContext *context, const LinalgSolveTilingData *tiling,
                          uint64_t libApiSize)
{
    const uint64_t elements = static_cast<uint64_t>(tiling->batch) * tiling->n *
                              (static_cast<uint64_t>(tiling->p) + 2U * tiling->q);
    context->GetWorkspaceSizes(1)[0] = libApiSize + elements * 4U + 64U;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *matrix = context->GetInputShape(0);
    const auto *rhs = context->GetInputShape(1);
    SolveShapePlan plan;
    if (matrix == nullptr || rhs == nullptr ||
        !MakeSolvePlan(&matrix->GetStorageShape(), &rhs->GetStorageShape(), ReadLeft(context->GetAttrs()), plan)) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto *tiling = context->GetTilingData<LinalgSolveTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *tiling = plan.data;
    uint64_t ubSize = 0U;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint64_t ubBytes = sizeof(float) * (5ULL * tiling->p + 3ULL * tiling->q + 8ULL * tiling->n + 8U);
    if (ubBytes > ubSize) {
        return ge::GRAPH_FAILED;
    }
    int32_t cores = platform.GetCoreNumAiv();
    FillBlockDim(tiling, cores > 0 ? cores : 1);

    const uint32_t DT_A = static_cast<uint32_t>(ge::DT_FLOAT);
    ASCENDC_TPL_SEL_PARAM(context, DT_A);

    context->SetBlockDim(tiling->blocks);
    context->GetRawTilingData()->SetDataSize(sizeof(LinalgSolveTilingData));
    FillWorkspace(context, tiling, platform.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *matrixShape = context->GetInputShape(0);
    const auto *rhsShape = context->GetInputShape(1);
    auto *outputShape = context->GetOutputShape(0);
    optiling::SolveShapePlan plan;
    if (outputShape == nullptr || !optiling::MakeSolvePlan(matrixShape, rhsShape,
                                                        optiling::ReadLeft(context->GetAttrs()), plan)) {
        return GRAPH_FAILED;
    }
    const auto &data = plan.data;
    outputShape->SetDimNum(data.batchRank + (plan.vectorRhs ? 1U : 2U));
    for (uint32_t i = 0U; i < data.batchRank; ++i) {
        outputShape->SetDim(i, data.batchDims[i]);
    }
    if (plan.vectorRhs) {
        outputShape->SetDim(data.batchRank, data.n);
    } else {
        outputShape->SetDim(data.batchRank, data.rightSide ? data.k : data.n);
        outputShape->SetDim(data.batchRank + 1U, data.rightSide ? data.n : data.k);
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

class LinalgSolve : public OpDef {
public:
    explicit LinalgSolve(const char *name) : OpDef(name)
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
        this->Input("B")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("X")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Attr("left").AttrType(OPTIONAL).Bool(true);
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

OP_ADD(LinalgSolve);

} // namespace ops
