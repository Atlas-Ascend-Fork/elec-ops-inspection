/*
 * Copyright (c) 2026 ceprei_Tnx.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not
 * use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software

 * * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied.
 * See the License for the specific language governing permissions and
 * limitations
 * under the License.
 */
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "../op_kernel/tiling_key_fused_disconnector_keypoint_soft_argmax.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace {
constexpr uint32_t kChannels = 32;
constexpr uint32_t kKeypoints = 4;
constexpr uint32_t kTaps = 9;
constexpr uint32_t kPaddedOutputChannels = 16;
constexpr uint32_t kLogitsSlotsPerGroup = 2;
constexpr uint32_t kRequiredStride = 4;

bool HasShape(const gert::Shape &shape, std::initializer_list<int64_t> dimensions) {
    if (shape.GetDimNum() != dimensions.size()) {
        return false;
    }
    size_t axis = 0;
    for (const int64_t dimension : dimensions) {
        if (shape.GetDim(axis++) != dimension) {
            return false;
        }
    }
    return true;
}

bool ValidSpatialShape(int64_t height, int64_t width) {
    return (width == 64 && (height == 48 || height == 64)) || (width == 96 && height == 72);
}

void SetUserWorkspaceLayout(FusedDisconnectorKeypointSoftArgmaxTilingData &tiling) {
    // The only GM intermediate is one 32-FP32 partial record per output row.
    tiling.packedWeightOffsetBytes = 0;
    tiling.packedFeatOffsetBytes = 0;
    tiling.logitsOffsetBytes = 0;
    tiling.partialOffsetBytes = 0;
    tiling.workspaceBytes = tiling.batch * tiling.height * kKeypointPartialFloats * sizeof(float);
}
} // namespace

namespace {
bool IsSupportedSpatialShape(int64_t height, int64_t width) {
    return (width == 64 && (height == 48 || height == 64)) || (width == 96 && height == 72);
}
} // namespace

namespace optiling {
static bool ReadInputs(gert::TilingContext *context, int64_t &batch, int64_t &height, int64_t &width) {
    const auto *feat = context->GetRequiredInputTensor(0);
    const auto *weight = context->GetRequiredInputTensor(1);
    const auto *bias = context->GetRequiredInputTensor(2);
    if (feat == nullptr || weight == nullptr || bias == nullptr || feat->GetDataType() != ge::DT_FLOAT ||
        weight->GetDataType() != ge::DT_FLOAT || bias->GetDataType() != ge::DT_FLOAT) {
        return false;
    }
    const auto featShape = feat->GetOriginShape();
    if (featShape.GetDimNum() != 4 || featShape.GetDim(1) != kChannels ||
        !HasShape(weight->GetOriginShape(), {kKeypoints, kChannels, 3, 3}) ||
        !HasShape(bias->GetOriginShape(), {kKeypoints})) {
        return false;
    }
    batch = featShape.GetDim(0);
    height = featShape.GetDim(2);
    width = featShape.GetDim(3);
    return batch >= 1 && batch <= 4 && ValidSpatialShape(height, width);
}

static bool ReadAttrs(gert::TilingContext *context, float &tau, int64_t &stride) {
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr || attrs->GetAttrNum() < 2) {
        return false;
    }
    const float *tauPtr = attrs->GetFloat(0);
    const int64_t *stridePtr = attrs->GetInt(1);
    if (tauPtr == nullptr || !std::isfinite(*tauPtr) || *tauPtr <= 0.0f || stridePtr == nullptr ||
        *stridePtr != kRequiredStride) {
        return false;
    }
    tau = *tauPtr;
    stride = *stridePtr;
    return true;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t batch = 0;
    int64_t height = 0;
    int64_t width = 0;
    float tau = 0.0f;
    int64_t stride = 0;
    if (!ReadInputs(context, batch, height, width) || !ReadAttrs(context, tau, stride)) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t vectorBlocks = static_cast<uint32_t>(platform.GetCoreNumAiv());
    // Launch the full platform Vector set. On the target 910B4 this is 40 AIVs.
    if (vectorBlocks == 0) {
        return ge::GRAPH_FAILED;
    }

    auto *tiling = context->GetTilingData<FusedDisconnectorKeypointSoftArgmaxTilingData>();
    size_t *workspace = context->GetWorkspaceSizes(1);
    // Hard SyncAll across the full AIV set requires resident cores.
    // CANN SyncAll recommends batch mode to prevent dual-stream deadlock.
    if (tiling == nullptr || workspace == nullptr || context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    *tiling = {};
    tiling->batch = static_cast<uint32_t>(batch);
    tiling->height = static_cast<uint32_t>(height);
    tiling->width = static_cast<uint32_t>(width);
    tiling->stride = static_cast<uint32_t>(stride);
    tiling->tau = tau;
    tiling->mixBlocks = vectorBlocks;
    tiling->finishBlocks = (tiling->batch + 1) / 2;
    SetUserWorkspaceLayout(*tiling);
    // Basic MMAD only: the state-only wrapper initializer needs no GM KFC region.
    workspace[0] = static_cast<size_t>(tiling->workspaceBytes);

    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(ge::DT_FLOAT));
    context->SetBlockDim(vectorBlocks);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *feat = context->GetInputShape(0);
    const gert::Shape *weight = context->GetInputShape(1);
    const gert::Shape *bias = context->GetInputShape(2);
    gert::Shape *keypoints = context->GetOutputShape(0);
    gert::Shape *scores = context->GetOutputShape(1);
    if (feat == nullptr || weight == nullptr || bias == nullptr || keypoints == nullptr || scores == nullptr ||
        feat->GetDimNum() != 4 || weight->GetDimNum() != 4 || bias->GetDimNum() != 1) {
        return GRAPH_FAILED;
    }
    const int64_t batch = feat->GetDim(0);
    if (batch < 1 || batch > 4 || feat->GetDim(1) != 32 || !IsSupportedSpatialShape(feat->GetDim(2), feat->GetDim(3)) ||
        weight->GetDim(0) != 4 || weight->GetDim(1) != 32 || weight->GetDim(2) != 3 || weight->GetDim(3) != 3 ||
        bias->GetDim(0) != 4) {
        return GRAPH_FAILED;
    }

    *keypoints = gert::Shape();
    keypoints->AppendDim(batch);
    keypoints->AppendDim(4);
    keypoints->AppendDim(2);
    *scores = gert::Shape();
    scores->AppendDim(batch);
    scores->AppendDim(4);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context->GetInputDataType(0) != DT_FLOAT || context->GetInputDataType(1) != DT_FLOAT ||
        context->GetInputDataType(2) != DT_FLOAT) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, DT_FLOAT);
    context->SetOutputDataType(1, DT_FLOAT);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class FusedDisconnectorKeypointSoftArgmax : public OpDef {
  public:
    explicit FusedDisconnectorKeypointSoftArgmax(const char *name) : OpDef(name) {
        this->Input("feat").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("keypoints").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("tau").AttrType(REQUIRED).Float();
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(FusedDisconnectorKeypointSoftArgmax);
} // namespace ops
