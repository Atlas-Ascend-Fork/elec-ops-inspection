/*
 * Copyright (c) 2026 lasth0pe.
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
// Host side tiling and operator registration for the LinalgEigh operator.
#include <algorithm>
#include <cstdint>
#include "eigh_host_plan.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/linalg_eigh_tiling.h"
#include "../op_kernel/tiling_key_linalg_eigh.h"

namespace optiling {

struct EighMatrixGeometry {
    uint32_t dimNum = 0U;
    uint32_t batchCount = 0U;
    uint32_t matrixOrder = 0U;
    bool valid = false;
};

static EighMatrixGeometry ExtractMatrixGeometry(const gert::Shape &originShape)
{
    EighMatrixGeometry geom;
    const size_t dims = originShape.GetDimNum();
    if (dims != 2U && dims != 3U) {
        return geom;
    }
    const int64_t rows = originShape.GetDim(dims - 2U);
    const int64_t cols = originShape.GetDim(dims - 1U);
    if (rows <= 0 || rows != cols) {
        return geom;
    }
    const int64_t batches = (dims == 3U) ? originShape.GetDim(0U) : 1;
    if (batches <= 0) {
        return geom;
    }
    geom.dimNum = static_cast<uint32_t>(dims);
    geom.batchCount = static_cast<uint32_t>(batches);
    geom.matrixOrder = static_cast<uint32_t>(rows);
    geom.valid = true;
    return geom;
}

template <uint32_t CASE_KEY>
static bool MatchesCaseGeometry(const EighMatrixGeometry &geom)
{
    using C = EighCase<CASE_KEY>;
    const uint32_t expectedDims = (C::B == 1U) ? 2U : 3U;
    return geom.valid && (geom.dimNum == expectedDims) &&
           (geom.batchCount == C::B) && (geom.matrixOrder == C::N);
}

static bool ValidateInputAndUploAttr(const gert::TilingContext *context)
{
    const auto *tensor = context->GetRequiredInputTensor(0);
    if (tensor == nullptr || tensor->GetDataType() != ge::DT_FLOAT) {
        return false;
    }
    const char *uplo = "L";
    const auto *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const char *attrStr = attrs->GetStr(0);
        if (attrStr != nullptr && attrStr[0] != '\0') {
            uplo = attrStr;
        }
    }
    const bool isLower = (uplo[0] == 'L' || uplo[0] == 'l') && (uplo[1] == '\0');
    return isLower;
}

static uint32_t QueryAvailableVectorCores(const gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t coreCount = platform.GetCoreNumAiv();
    const uint32_t clampedCores = static_cast<uint32_t>((coreCount > 0) ? coreCount : 1);
    return (clampedCores < 32U) ? clampedCores : 32U;
}

template <uint32_t CASE_KEY>
static ge::graphStatus ApplyCasePlan(gert::TilingContext *context, uint32_t availableCores)
{
    constexpr auto &plan = EighHostPlan<CASE_KEY>::value;
    if (plan.used_cores > availableCores) {
        return ge::GRAPH_FAILED;
    }
    *context->GetTilingData<LinalgEighTilingData>() = plan;
    const uint32_t caseKey = CASE_KEY;
    const uint32_t dtCode = static_cast<uint32_t>(ge::DT_FLOAT);
    ASCENDC_TPL_SEL_PARAM(context, dtCode, caseKey);
    context->SetBlockDim(plan.used_cores);
    context->GetWorkspaceSizes(1)[0] = 0U;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchCasePlan(gert::TilingContext *context,
                                        const EighMatrixGeometry &geom,
                                        uint32_t availableCores)
{
    if (MatchesCaseGeometry<1U>(geom)) return ApplyCasePlan<1U>(context, availableCores);
    if (MatchesCaseGeometry<2U>(geom)) return ApplyCasePlan<2U>(context, availableCores);
    if (MatchesCaseGeometry<3U>(geom)) return ApplyCasePlan<3U>(context, availableCores);
    if (MatchesCaseGeometry<4U>(geom)) return ApplyCasePlan<4U>(context, availableCores);
    if (MatchesCaseGeometry<5U>(geom)) return ApplyCasePlan<5U>(context, availableCores);
    if (MatchesCaseGeometry<6U>(geom)) return ApplyCasePlan<6U>(context, availableCores);
    return ge::GRAPH_FAILED;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (!ValidateInputAndUploAttr(context)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t availableCores = QueryAvailableVectorCores(context);
    const auto &originShape = context->GetInputShape(0)->GetOriginShape();
    const EighMatrixGeometry geom = ExtractMatrixGeometry(originShape);
    return DispatchCasePlan(context, geom, availableCores);
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *matrixShape = context->GetInputShape(0);
    auto *eigValShape = context->GetOutputShape(0);
    auto *eigVecShape = context->GetOutputShape(1);
    if (matrixShape == nullptr || eigValShape == nullptr || eigVecShape == nullptr) {
        return GRAPH_SUCCESS;
    }
    const size_t dimNum = matrixShape->GetDimNum();
    *eigVecShape = *matrixShape;
    *eigValShape = *matrixShape;
    if (dimNum >= 1U) {
        eigValShape->SetDimNum(dimNum - 1U);
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context != nullptr) {
        context->SetOutputDataType(0, DT_FLOAT);
        context->SetOutputDataType(1, DT_FLOAT);
    }
    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class LinalgEigh : public OpDef {
public:
    explicit LinalgEigh(const char *name) : OpDef(name)
    {
        InitPorts();
        InitAttrs();
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
        this->Output("eigenvalues")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("eigenvectors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
    }

    void InitAttrs()
    {
        this->Attr("UPLO").AttrType(OPTIONAL).String("L");
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

OP_ADD(LinalgEigh);

} // namespace ops
