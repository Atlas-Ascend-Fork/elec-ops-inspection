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
    uint32_t uplo = 0U;
    bool valid = false;
};

static EighMatrixGeometry ExtractMatrixGeometry(const gert::Shape &originShape, uint32_t uploVal)
{
    EighMatrixGeometry geom;
    const size_t dims = originShape.GetDimNum();
    if (dims < 2U) {
        return geom;
    }
    const int64_t rows = originShape.GetDim(dims - 2U);
    const int64_t cols = originShape.GetDim(dims - 1U);
    if (rows <= 0 || rows > 512 || rows != cols) {
        return geom;
    }
    int64_t batches = 1;
    for (size_t d = 0U; d + 2U < dims; ++d) {
        const int64_t dimVal = originShape.GetDim(d);
        if (dimVal <= 0) {
            return geom;
        }
        batches *= dimVal;
    }
    geom.dimNum = static_cast<uint32_t>(dims);
    geom.batchCount = static_cast<uint32_t>(batches);
    geom.matrixOrder = static_cast<uint32_t>(rows);
    geom.uplo = uploVal;
    geom.valid = true;
    return geom;
}

static bool ValidateInputAndParseUplo(const gert::TilingContext *context, uint32_t &uploOut)
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
    if (uplo[1] != '\0') {
        return false;
    }
    if (uplo[0] == 'L' || uplo[0] == 'l') {
        uploOut = 0U;
        return true;
    }
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        uploOut = 1U;
        return true;
    }
    return false;
}

static uint32_t QueryAvailableVectorCores(const gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t coreCount = platform.GetCoreNumAiv();
    const uint32_t clampedCores = static_cast<uint32_t>((coreCount > 0) ? coreCount : 1);
    return (clampedCores < 32U) ? clampedCores : 32U;
}

static ge::graphStatus CommitTilingPlan(gert::TilingContext *context,
                                        const LinalgEighTilingData &plan,
                                        uint32_t caseKey)
{
    *context->GetTilingData<LinalgEighTilingData>() = plan;
    const uint32_t dtCode = static_cast<uint32_t>(ge::DT_FLOAT);
    ASCENDC_TPL_SEL_PARAM(context, dtCode, caseKey);
    context->SetBlockDim(plan.used_cores);
    context->GetWorkspaceSizes(1)[0] = 0U;
    return ge::GRAPH_SUCCESS;
}

template <uint32_t CASE_KEY>
static bool MatchesPresetCase(const EighMatrixGeometry &geom, uint32_t availableCores)
{
    using C = EighCase<CASE_KEY>;
    constexpr uint32_t reqCores = C::B * C::P;
    return (geom.batchCount == C::B) &&
           (geom.matrixOrder == C::N) && (availableCores >= reqCores);
}

static ge::graphStatus ApplyGeneralPlan(gert::TilingContext *context,
                                        const EighMatrixGeometry &geom,
                                        uint32_t availableCores)
{
    LinalgEighTilingData plan = BuildEighTilingPlan(
        geom.matrixOrder, geom.batchCount, 1U, 4U, geom.uplo);
    const uint32_t maxSafeCores = ((geom.matrixOrder & 15U) == 0U) ? availableCores : 1U;
    const uint32_t targetCores = (geom.batchCount < maxSafeCores) ? geom.batchCount : maxSafeCores;
    plan.used_cores = (targetCores > 0U) ? targetCores : 1U;
    return CommitTilingPlan(context, plan, 0U);
}

template <uint32_t CASE_A, uint32_t CASE_B>
static ge::graphStatus DispatchScaleTier(gert::TilingContext *context,
                                         const EighMatrixGeometry &geom,
                                         uint32_t availableCores)
{
    if (MatchesPresetCase<CASE_A>(geom, availableCores)) {
        return CommitTilingPlan(context, BuildCaseTilingPlan<CASE_A>(geom.uplo), CASE_A);
    }
    if (MatchesPresetCase<CASE_B>(geom, availableCores)) {
        return CommitTilingPlan(context, BuildCaseTilingPlan<CASE_B>(geom.uplo), CASE_B);
    }
    return ApplyGeneralPlan(context, geom, availableCores);
}

static ge::graphStatus DispatchByMatrixScale(gert::TilingContext *context,
                                             const EighMatrixGeometry &geom,
                                             uint32_t availableCores)
{
    if (!geom.valid) {
        return ge::GRAPH_FAILED;
    }
    if (geom.matrixOrder <= 16U) {
        return DispatchScaleTier<1U, 4U>(context, geom, availableCores);
    }
    if (geom.matrixOrder <= 64U) {
        return DispatchScaleTier<2U, 5U>(context, geom, availableCores);
    }
    return DispatchScaleTier<3U, 6U>(context, geom, availableCores);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    uint32_t uploVal = 0U;
    if (!ValidateInputAndParseUplo(context, uploVal)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t availableCores = QueryAvailableVectorCores(context);
    const auto &originShape = context->GetInputShape(0)->GetOriginShape();
    const EighMatrixGeometry geom = ExtractMatrixGeometry(originShape, uploVal);
    return DispatchByMatrixScale(context, geom, availableCores);
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
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("eigenvalues").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("eigenvectors").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
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
