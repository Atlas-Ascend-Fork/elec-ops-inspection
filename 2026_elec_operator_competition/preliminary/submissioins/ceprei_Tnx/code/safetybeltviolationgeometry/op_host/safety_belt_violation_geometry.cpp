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
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/safety_belt_violation_geometry_tiling.h"
#include "../op_kernel/tiling_key_safety_belt_violation_geometry.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>

namespace {
constexpr uint32_t kMaxCores = 8U;

struct InputInfo {
    uint32_t batch;
    ge::DataType dtype;
    ge::DataType anchorType;
    ge::DataType outType;
    bool hasThreshold;
};

uint32_t AlignRows(uint32_t rows)
{
    return ((rows + 63U) / 64U) * 64U;
}

bool GetBatch(const gert::TilingContext *context, InputInfo &info)
{
    const auto *personDesc = context->GetInputDesc(0);
    const auto *anchorDesc = context->GetInputDesc(1);
    const auto *personShape = context->GetInputShape(0);
    const auto *anchorShape = context->GetInputShape(1);
    if (personDesc == nullptr || anchorDesc == nullptr ||
        personShape == nullptr || anchorShape == nullptr) {
        return false;
    }
    const auto &ps = personShape->GetStorageShape();
    const auto &as = anchorShape->GetStorageShape();
    constexpr int64_t kMaxBatch =
        static_cast<int64_t>(kMaxCores) * static_cast<int64_t>(SBG_MAX_LOCAL_ROWS);
    if (ps.GetDimNum() != 2U || as.GetDimNum() != 2U || ps.GetDim(1) != 4 ||
        as.GetDim(1) != 2 || ps.GetDim(0) != as.GetDim(0) ||
        ps.GetDim(0) < 1 || ps.GetDim(0) > kMaxBatch) {
        return false;
    }
    info.batch = static_cast<uint32_t>(ps.GetDim(0));
    info.dtype = personDesc->GetDataType();
    info.anchorType = anchorDesc->GetDataType();
    return true;
}

bool GetTypes(const gert::TilingContext *context, InputInfo &info)
{
    const auto *outDesc = context->GetOutputDesc(0);
    if (outDesc == nullptr) {
        return false;
    }
    info.outType = outDesc->GetDataType();
    const bool legacyIO = info.dtype == ge::DT_FLOAT16 && info.outType == ge::DT_FLOAT16;
    if ((info.dtype != ge::DT_FLOAT && info.dtype != ge::DT_FLOAT16) ||
        info.anchorType != info.dtype) {
        return false;
    }
    if (info.outType != ge::DT_FLOAT && !legacyIO) {
        return false;
    }
    info.hasThreshold = false;
    const auto *thresholdShape = context->GetOptionalInputShape(2);
    if (thresholdShape != nullptr) {
        const auto *thresholdDesc = context->GetOptionalInputDesc(2);
        const auto &ts = thresholdShape->GetStorageShape();
        if (thresholdDesc == nullptr ||
            thresholdDesc->GetDataType() != (legacyIO ? ge::DT_FLOAT16 : ge::DT_FLOAT) ||
            ts.GetDimNum() != 1U || ts.GetShapeSize() != 1) {
            return false;
        }
        info.hasThreshold = true;
    }
    return true;
}

bool GetRatio(const gert::TilingContext *context, float &ratio)
{
    const auto *attrs = context->GetAttrs();
    const float *ratioPtr = attrs == nullptr ? nullptr : attrs->GetFloat(0);
    if (ratioPtr == nullptr || !std::isfinite(*ratioPtr) ||
        *ratioPtr <= 0.0f || *ratioPtr > 1.0f) {
        return false;
    }
    ratio = *ratioPtr;
    return true;
}

bool WriteTiling(gert::TilingContext *context, const InputInfo &info, float ratio)
{
    const uint32_t workers = std::min(
        kMaxCores, (info.batch + SBG_MAX_LOCAL_ROWS - 1U) / SBG_MAX_LOCAL_ROWS);
    if (workers == 0U) {
        return false;
    }
    const uint32_t rowsPerCore = AlignRows((info.batch + workers - 1U) / workers);
    if (rowsPerCore == 0U || rowsPerCore > SBG_MAX_LOCAL_ROWS) {
        return false;
    }
    auto *tiling = context->GetTilingData<SafetyBeltViolationGeometryTilingData>();
    if (tiling == nullptr) {
        return false;
    }
    tiling->slots[0].pack = (info.batch - 1U) | ((rowsPerCore / 64U) << 12U) |
                            (static_cast<uint32_t>(info.hasThreshold) << 16U);
    tiling->slots[0].waistRatio = ratio;
    const uint32_t used = (info.batch + rowsPerCore - 1U) / rowsPerCore;
    ASCENDC_TPL_SEL_PARAM(context,
                          static_cast<uint32_t>(info.dtype),
                          static_cast<uint32_t>(info.outType),
                          0U,
                          0U,
                          static_cast<uint32_t>(info.hasThreshold),
                          0U,
                          0U);
    context->SetBlockDim(std::min(kMaxCores, std::max(used, 1U)));
    context->GetWorkspaceSizes(1)[0] = 0U;
    return true;
}

template <typename Port>
void SetPortLayout(Port &port, std::initializer_list<ge::DataType> types)
{
    port.DataType(types)
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    InputInfo info{};
    float ratio = 0.0f;
    if (!GetBatch(context, info) || !GetTypes(context, info) || !GetRatio(context, ratio) ||
        !WriteTiling(context, info, ratio)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *person = context->GetInputShape(0);
    const auto *anchor = context->GetInputShape(1);
    auto *output = context->GetOutputShape(0);
    if (person == nullptr || anchor == nullptr || output == nullptr ||
        person->GetDimNum() != 2U || anchor->GetDimNum() != 2U ||
        person->GetDim(1) != 4 || anchor->GetDim(1) != 2 ||
        person->GetDim(0) != anchor->GetDim(0)) {
        return GRAPH_FAILED;
    }
    *output = gert::Shape();
    output->AppendDim(person->GetDim(0));
    output->AppendDim(1);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto thresholdType = context->GetOptionalInputDataType(2);
    const bool legacyIO = context->GetInputDataType(0) == ge::DT_FLOAT16 &&
        thresholdType == ge::DT_FLOAT16;
    context->SetOutputDataType(0, legacyIO ? ge::DT_FLOAT16 : ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SafetyBeltViolationGeometry : public OpDef {
public:
    explicit SafetyBeltViolationGeometry(const char *name) : OpDef(name)
    {
        auto &person = this->Input("person_boxes").ParamType(REQUIRED);
        SetPortLayout(person, {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16});
        auto &anchor = this->Input("anchor_points").ParamType(REQUIRED);
        SetPortLayout(anchor, {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16});
        auto &threshold = this->Input("threshold").ParamType(OPTIONAL);
        SetPortLayout(threshold, {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16});
        auto &output = this->Output("violation_score").ParamType(REQUIRED);
        SetPortLayout(output, {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16});
        this->Attr("waist_ratio").AttrType(REQUIRED).Float();
        this->SetInferDataType(ge::InferDataType);
        this->SetInferShape(ge::InferShape);
        auto &aiCore = this->AICore();
        aiCore.SetTiling(optiling::TilingFunc);
        aiCore.AddConfig("ascend910b");
    }
};
OP_ADD(SafetyBeltViolationGeometry);
}  // namespace ops
