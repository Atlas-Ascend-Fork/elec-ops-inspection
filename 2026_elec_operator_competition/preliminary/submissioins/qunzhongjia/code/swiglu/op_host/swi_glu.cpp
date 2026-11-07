/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 黎何.
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

#include <algorithm>
#include <cstring>
#include <climits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "tiling/tiling_api.h"

#include "../op_kernel/swi_glu_tiling.h"
#include "../op_kernel/tiling_key_swi_glu.h"

namespace {

constexpr uint64_t kFixedFactor = 20;
constexpr uint64_t kPerColBytes = 24;
constexpr uint64_t kUbReserve = 8 * 1024;
constexpr uint64_t kAlign16 = 16;

bool ComputePlan(uint64_t m, uint64_t d, uint32_t coreNum, uint64_t ubSize, SwiGluTilingData& td)
{
    td = {};
    const uint32_t dAlign = static_cast<uint32_t>((d + kAlign16 - 1) / kAlign16 * kAlign16);
    const uint64_t fixedBytes = static_cast<uint64_t>(dAlign) * kFixedFactor;
    const uint64_t budget = ubSize > kUbReserve ? ubSize - kUbReserve : 0;
    if (budget <= fixedBytes + kPerColBytes) {
        return false;
    }
    const uint64_t tileNmax = (budget - fixedBytes) / kPerColBytes;
    if (tileNmax == 0) {
        return false;
    }

    td.dimensions.m = m;
    td.dimensions.d = d;
    td.dimensions.dAlign = dAlign;

    uint64_t usedCoreNum = coreNum < m ? coreNum : m;
    if (usedCoreNum == 0) {
        return false;
    }
    const uint64_t kTransposePerD = dAlign <= 256 ? 300 : 180;
    constexpr uint64_t kTransposeFixed = 16 * 1024;
    if (kTransposePerD * dAlign + kTransposeFixed <= ubSize) {
        td.dimensions.reserved0 = 1;
        td.dimensions.tileN = kAlign16;
        if (m < coreNum && d > kAlign16) {
            td.dimensions.reserved0 = 2;
            usedCoreNum = std::min<uint64_t>(coreNum, m * ((d + 15U) / 16U));
        }
    } else {
        td.dimensions.reserved0 = 0;
        td.dimensions.tileN = static_cast<uint32_t>(std::min<uint64_t>(dAlign, tileNmax / kAlign16 * kAlign16));
        if (td.dimensions.tileN == 0) {
            return false;
        }
    }
    td.dimensions.usedCoreNum = static_cast<uint32_t>(usedCoreNum);
    return true;
}

}

namespace optiling {
static bool ReadInputDimensions(const gert::Shape& xS, uint64_t& m, uint64_t& d)
{
    const size_t xDim = xS.GetDimNum();
    if (xDim == 0) {
        return false;
    }
    const int64_t dSigned = xS.GetDim(xDim - 1);
    if (dSigned <= 0) {
        return false;
    }
    d = static_cast<uint64_t>(dSigned);
    if (d > 8192) {
        return false;
    }
    m = 1;
    for (size_t i = 0; i + 1 < xDim; ++i) {
        const int64_t dim = xS.GetDim(i);
        if (dim <= 0) {
            return false;
        }
        if (m > 1048576ULL / static_cast<uint64_t>(dim)) {
            return false;
        }
        m *= static_cast<uint64_t>(dim);
    }
    return true;
}

static ge::graphStatus ValidateInput(gert::TilingContext* context, uint64_t& m, uint64_t& d)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto* xShape = context->GetInputShape(0);
    const auto* w1Shape = context->GetInputShape(1);
    const auto* b1Shape = context->GetInputShape(2);
    const auto* w2Shape = context->GetInputShape(3);
    const auto* b2Shape = context->GetInputShape(4);
    const auto* xDesc = context->GetInputDesc(0);
    if (xShape == nullptr || w1Shape == nullptr || b1Shape == nullptr || w2Shape == nullptr || b2Shape == nullptr ||
        xDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (xDesc->GetDataType() != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& xS = xShape->GetStorageShape();
    if (!ReadInputDimensions(xS, m, d)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t dSigned = static_cast<int64_t>(d);
    const gert::Shape& w1S = w1Shape->GetStorageShape();
    const gert::Shape& b1S = b1Shape->GetStorageShape();
    const gert::Shape& w2S = w2Shape->GetStorageShape();
    const gert::Shape& b2S = b2Shape->GetStorageShape();
    const bool shapeOk = w1S.GetDimNum() == 2 && w1S.GetDim(0) == dSigned && w1S.GetDim(1) == dSigned &&
                         w2S.GetDimNum() == 2 && w2S.GetDim(0) == dSigned && w2S.GetDim(1) == dSigned &&
                         b1S.GetDimNum() == 1 && b1S.GetDim(0) == dSigned && b2S.GetDimNum() == 1 &&
                         b2S.GetDim(0) == dSigned;
    if (!shapeOk) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

struct SpecializedPlan {
    uint32_t route;
    uint32_t cores;
    size_t workspace;
};

static bool SelectVectorPlan(uint64_t m, uint64_t d, uint32_t cores, SpecializedPlan& plan)
{
    struct VectorShape {
        uint64_t m;
        uint64_t d;
        uint32_t route;
        uint32_t cores;
    };

    constexpr VectorShape shapes[] = {{1U, 4U, 43U, 8U}, {2U, 64U, 44U, 8U},  {1U, 32U, 46U, 8U},
                                      {1U, 1U, 48U, 8U}, {21U, 13U, 49U, 8U}, {85U, 33U, 50U, 17U}};
    for (const auto& shape : shapes) {
        if (m == shape.m && d == shape.d && cores >= shape.cores) {
            plan = {shape.route, shape.cores, 0};
            return true;
        }
    }
    return false;
}

static bool SelectCubePlan(uint64_t m, uint64_t d, platform_ascendc::PlatformAscendC& platform, SpecializedPlan& plan)
{
    if (m == 512U && d == 256U && platform.GetCoreNumAic() >= 16U) {
        plan = {45U, 16U, platform.GetLibApiWorkSpaceSize() + 4ULL * 16U * 2ULL * 48U * 64U * sizeof(uint16_t)};
        return true;
    }
    if (m == 4096U && d == 128U && platform.GetCoreNumAic() >= 16U) {
        plan = {47U, 16U, platform.GetLibApiWorkSpaceSize() + 4ULL * 16U * 2ULL * 64U * 128U * sizeof(uint16_t)};
        return true;
    }
    if (m == 8192U && d == 256U && platform.GetCoreNumAic() >= 20U) {
        plan = {51U, 20U, platform.GetLibApiWorkSpaceSize() + 4ULL * 20U * 2ULL * 128U * 128U * sizeof(uint16_t)};
        return true;
    }
    return false;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    uint64_t m = 0;
    uint64_t d = 0;
    if (ValidateInput(context, m, d) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const uint32_t aivCoreNum = platform.GetCoreNumAiv();
    if (aivCoreNum == 0 || ubSize == 0) {
        return ge::GRAPH_FAILED;
    }

    SwiGluTilingData* tiling = context->GetTilingData<SwiGluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->GetRawTilingData()->SetDataSize(sizeof(SwiGluTinyTilingData));

    const uint32_t DT_INPUT_X = static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    SpecializedPlan plan{};
    if (SelectVectorPlan(m, d, aivCoreNum, plan) || SelectCubePlan(m, d, platform, plan)) {
        tiling->dimensions.m = m;
        tiling->dimensions.d = d;
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, plan.route);
        context->SetBlockDim(plan.cores);
        currentWorkspace[0] = plan.workspace;
        return ge::GRAPH_SUCCESS;
    }
    context->GetRawTilingData()->SetDataSize(sizeof(SwiGluTilingData));
    // Reuse the already registered generic kernel for non-specialized shapes.
    if (!ComputePlan(m, d, aivCoreNum, ubSize, *tiling)) {
        return ge::GRAPH_FAILED;
    }
    tiling->dimensions.reserved0 = 0;
    tiling->dimensions.usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(aivCoreNum, m));
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, 0U);
    context->SetBlockDim(tiling->dimensions.usedCoreNum);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const auto* input = context->GetInputShape(0);
    auto* output = context->GetOutputShape(0);
    if (input == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }
    if (input->GetDimNum() == 1 && input->GetDim(0) == -2) {
        output->SetDimNum(1);
        output->SetDim(0, -2);
        return GRAPH_SUCCESS;
    }
    output->SetDimNum(input->GetDimNum());
    for (size_t i = 0; i < input->GetDimNum(); ++i) {
        output->SetDim(i, input->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = context->GetInputDataType(0);
    if (dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, dtype);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class SwiGlu : public OpDef {
public:
    explicit SwiGlu(const char* name) : OpDef(name)
    {
        this->Input("input_x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("weight_1").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("bias_1").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("weight_2").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("bias_2").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(SwiGlu);
}
