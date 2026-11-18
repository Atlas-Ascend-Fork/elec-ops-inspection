/*
 * Copyright (c) 2026 hit_op-samerberry.
 * SPDX-License-Identifier: Apache-2.0
 *
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
#include <cmath>
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "../op_kernel/tiling_key_fused_disconnector_keypoint_soft_argmax.h"

namespace optiling {

namespace {

constexpr int64_t kMaxBatch = 4;
constexpr int64_t kExpectedCin = 32;
constexpr int64_t kExpectedCout = 4;
constexpr int64_t kKernelDim = 3;
constexpr int64_t kExpectedStride = 4;
constexpr uint32_t kPressureCap = 24u;
constexpr uint32_t kMaxTileHeight = 16u;
constexpr uint32_t kSmallStripeElems = 512u;
constexpr uint32_t kTargetStripeElems = 256u;
constexpr uint32_t kMaxChanStride = 2040u;
constexpr uint32_t kMaxUbFloats = 48128u;
constexpr uint32_t kPacketFloats = 24u;
constexpr uint32_t kWorkspacePadFloats = 64u;

struct ProblemSpec {
    uint32_t batch;
    uint32_t cIn;
    uint32_t cOut;
    uint32_t h;
    uint32_t w;
    int32_t stride;
    float tau;
};

struct LaunchPlan {
    uint32_t mode;
    uint32_t stripes;
    uint32_t cores;
};

bool IsSupportedSpatial(int64_t h, int64_t w)
{
    return (h == 48 && w == 64) || (h == 64 && w == 64) || (h == 72 && w == 96);
}

bool FetchFloatInputShapes(const gert::TilingContext *context,
                           gert::Shape &featShape, gert::Shape &weightShape, gert::Shape &biasShape)
{
    const gert::Tensor *tensors[3] = {
        context->GetRequiredInputTensor(0),
        context->GetRequiredInputTensor(1),
        context->GetRequiredInputTensor(2)
    };
    for (const gert::Tensor *tensor : tensors) {
        if (tensor == nullptr || tensor->GetDataType() != ge::DT_FLOAT) {
            return false;
        }
    }
    featShape = tensors[0]->GetStorageShape();
    weightShape = tensors[1]->GetStorageShape();
    biasShape = tensors[2]->GetStorageShape();
    return featShape.GetDimNum() == 4 && weightShape.GetDimNum() == 4 && biasShape.GetDimNum() == 1;
}

bool ValidateFilterDims(const gert::Shape &weightShape, const gert::Shape &biasShape, int64_t cIn64)
{
    if (weightShape.GetDim(0) != kExpectedCout || biasShape.GetDim(0) != kExpectedCout) {
        return false;
    }
    if (weightShape.GetDim(1) != cIn64) {
        return false;
    }
    return weightShape.GetDim(2) == kKernelDim && weightShape.GetDim(3) == kKernelDim;
}

bool ValidateInputTensors(const gert::TilingContext *context, ProblemSpec &spec)
{
    gert::Shape featShape;
    gert::Shape weightShape;
    gert::Shape biasShape;
    if (!FetchFloatInputShapes(context, featShape, weightShape, biasShape)) {
        return false;
    }
    const int64_t batch64 = featShape.GetDim(0);
    const int64_t cIn64 = featShape.GetDim(1);
    const int64_t h64 = featShape.GetDim(2);
    const int64_t w64 = featShape.GetDim(3);
    if (batch64 < 1 || batch64 > kMaxBatch || cIn64 != kExpectedCin) {
        return false;
    }
    if (!IsSupportedSpatial(h64, w64) || !ValidateFilterDims(weightShape, biasShape, cIn64)) {
        return false;
    }
    spec.batch = static_cast<uint32_t>(batch64);
    spec.cIn = static_cast<uint32_t>(cIn64);
    spec.cOut = static_cast<uint32_t>(kExpectedCout);
    spec.h = static_cast<uint32_t>(h64);
    spec.w = static_cast<uint32_t>(w64);
    return true;
}

bool ValidateAttrs(const gert::TilingContext *context, ProblemSpec &spec)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }
    const float *tauPtr = attrs->GetFloat(0);
    const int64_t *stridePtr = attrs->GetInt(1);
    if (tauPtr == nullptr || stridePtr == nullptr) {
        return false;
    }
    const float tau = *tauPtr;
    const int64_t strideVal = *stridePtr;
    if (!(tau > 0.0f) || !std::isfinite(tau) || strideVal != kExpectedStride) {
        return false;
    }
    spec.tau = tau;
    spec.stride = static_cast<int32_t>(strideVal);
    return true;
}

uint32_t SelectCoreBudget(const ProblemSpec &spec, uint32_t aivCores)
{
    uint32_t coreBudget = aivCores;
    const uint32_t pressureCap = std::min(aivCores, kPressureCap);
    if (spec.batch <= pressureCap) {
        const uint32_t cappedStripes = pressureCap / spec.batch;
        const uint32_t cappedHeight = (spec.h + cappedStripes - 1u) / cappedStripes;
        if (cappedHeight <= kMaxTileHeight && cappedHeight * spec.w <= kSmallStripeElems) {
            coreBudget = pressureCap;
        }
    }
    return coreBudget;
}

uint32_t ChooseStripeCount(const ProblemSpec &spec, uint32_t minTileH, uint32_t coreBudget)
{
    const uint32_t minFourRepH = (kTargetStripeElems + spec.w - 1u) / spec.w;
    const uint32_t targetMinH = std::min(spec.h, std::max(minTileH, minFourRepH));
    const uint32_t maxS = std::max(1u, std::min(spec.h / targetMinH, coreBudget / spec.batch));
    const uint32_t maxH = (spec.h + maxS - 1u) / maxS;
    uint32_t stripes = (spec.h + maxH - 1u) / maxH;
    const uint32_t availableS = std::max(1u, std::min(spec.h / minTileH, coreBudget / spec.batch));
    uint32_t packetExtent = 2u;
    while (packetExtent < stripes) {
        packetExtent <<= 1;
    }
    if (availableS <= packetExtent && (spec.h + availableS - 1u) / availableS == maxH) {
        stripes = availableS;
    }
    return stripes;
}

uint32_t ComputeTotalUbFloats(uint32_t maxH, uint32_t w, uint32_t &cStrideOut)
{
    const uint32_t outRep = (maxH * w + 63u) >> 6;
    uint32_t cStride = ((2u * w + (outRep << 6)) + 127u) & ~127u;
    if ((cStride % 384u) == 0u) {
        cStride += 128u;
    }
    uint32_t oStride = ((outRep << 6) + 127u) & ~127u;
    if (outRep == 2u) {
        oStride = 256u;
    } else {
        while ((oStride % 384u) != 128u) {
            oStride += 128u;
        }
    }
    cStrideOut = cStride;
    return 1536u + 24u * cStride + 12u * oStride + 1536u;
}

bool IsLaunchGeometryValid(const ProblemSpec &spec, uint32_t minTileH, uint32_t aivCores)
{
    if ((spec.w % 8u) != 0u || spec.w < 8u || spec.w > 128u) {
        return false;
    }
    return spec.h >= minTileH && spec.batch <= aivCores;
}

bool FitsHardwareLimits(const ProblemSpec &spec, uint32_t stripes, uint32_t aivCores)
{
    const uint32_t maxH = (spec.h + stripes - 1u) / stripes;
    uint32_t cStride = 0u;
    const uint32_t totalUbFloats = ComputeTotalUbFloats(maxH, spec.w, cStride);
    if (maxH > kMaxTileHeight || cStride > kMaxChanStride) {
        return false;
    }
    return totalUbFloats <= kMaxUbFloats && spec.batch * stripes <= aivCores;
}

bool BuildLaunchPlan(const ProblemSpec &spec, uint32_t aivCores, LaunchPlan &plan)
{
    const uint32_t minTileH = std::max(2u, (65u + spec.w - 1u) / spec.w);
    if (!IsLaunchGeometryValid(spec, minTileH, aivCores)) {
        return false;
    }
    const uint32_t coreBudget = SelectCoreBudget(spec, aivCores);
    const uint32_t stripes = ChooseStripeCount(spec, minTileH, coreBudget);
    if (!FitsHardwareLimits(spec, stripes, aivCores)) {
        return false;
    }
    plan.mode = 1u;
    plan.stripes = stripes;
    plan.cores = spec.batch * stripes;
    if (plan.cores <= coreBudget / 2u) {
        plan.mode = 2u;
        plan.cores *= 2u;
    }
    return true;
}

}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (aivCores == 0u) {
        aivCores = 1u;
    }
    ProblemSpec spec{};
    LaunchPlan plan{};
    if (!ValidateInputTensors(context, spec) || !ValidateAttrs(context, spec) ||
        !BuildLaunchPlan(spec, aivCores, plan)) {
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<FusedDisconnectorKeypointSoftArgmaxTilingData>();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (tiling == nullptr || currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ASCENDC_TPL_SEL_PARAM(context, plan.mode);
    tiling->batch = spec.batch;
    tiling->cIn = spec.cIn;
    tiling->cOut = spec.cOut;
    tiling->h = spec.h;
    tiling->w = spec.w;
    tiling->stripes = plan.stripes;
    tiling->cores = plan.cores;
    tiling->stride = spec.stride;
    tiling->tau = spec.tau;
    context->SetBlockDim(plan.cores);
    context->SetScheduleMode(1);
    currentWorkspace[0] = (static_cast<size_t>(plan.cores) * kPacketFloats + kWorkspacePadFloats) * sizeof(float);
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *featShape = context->GetInputShape(0);
    const gert::Shape *weightShape = context->GetInputShape(1);
    gert::Shape *kpShape = context->GetOutputShape(0);
    gert::Shape *scoreShape = context->GetOutputShape(1);
    if (featShape == nullptr || weightShape == nullptr || kpShape == nullptr || scoreShape == nullptr) {
        return GRAPH_FAILED;
    }
    const int64_t b = (featShape->GetDimNum() >= 1) ? featShape->GetDim(0) : -1;
    const int64_t k = (weightShape->GetDimNum() >= 1) ? weightShape->GetDim(0) : 4;
    kpShape->SetDimNum(3);
    kpShape->SetDim(0, b);
    kpShape->SetDim(1, k);
    kpShape->SetDim(2, 2);
    scoreShape->SetDimNum(2);
    scoreShape->SetDim(0, b);
    scoreShape->SetDim(1, k);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class FusedDisconnectorKeypointSoftArgmax : public OpDef {
public:
    explicit FusedDisconnectorKeypointSoftArgmax(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> floatTypes = {ge::DT_FLOAT};
        const std::vector<ge::Format> ndFormats = {ge::FORMAT_ND};
        for (const char *inName : {"feat", "weight", "bias"}) {
            this->Input(inName).ParamType(REQUIRED).DataType(floatTypes).Format(ndFormats);
        }
        for (const char *outName : {"keypoints", "scores"}) {
            this->Output(outName).ParamType(REQUIRED).DataType(floatTypes).Format(ndFormats);
        }
        this->Attr("tau").AttrType(REQUIRED).Float();
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(FusedDisconnectorKeypointSoftArgmax);

}  // namespace ops
