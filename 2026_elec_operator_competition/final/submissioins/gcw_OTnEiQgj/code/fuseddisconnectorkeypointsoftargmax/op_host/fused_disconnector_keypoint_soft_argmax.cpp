/*
 * Copyright (c) 2026 gcw_OTnEiQgj.
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

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "../op_kernel/tiling_key_fused_disconnector_keypoint_soft_argmax.h"

namespace optiling {

struct PlanItem {
    uint8_t mode;
    uint8_t parts;
    uint8_t cores;
};

static uint32_t GetCoreLimit(gert::TilingContext *context) {
    static const uint32_t coreLim = [&]() {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        return std::max<uint32_t>(1U, static_cast<uint32_t>(platform.GetCoreNumAiv()));
    }();
    return coreLim;
}

static uint32_t MatchGridTag(uint32_t h, uint32_t w) {
    if (h == 48U && w == 64U) {
        return 0U;
    }
    if (h == 64U && w == 64U) {
        return 1U;
    }
    if (h == 72U && w == 96U) {
        return 2U;
    }
    return 3U;
}

static void SelectVarPlan(
    uint32_t b, uint32_t h, uint32_t tag, uint32_t coreLim,
    uint64_t &mode, uint32_t &parts, uint32_t &cores) {
    const uint32_t items = b * 4U;
    if (items == 0U) {
        return;
    }
    const uint32_t lim = std::min<uint32_t>(coreLim / items, h / 2U);
    if (lim >= 2U) {
        mode = 12ULL + static_cast<uint64_t>(tag);
        parts = 2U;
        cores = items * 2U;
    } else if (lim == 1U && tag == 0U) {
        mode = 12ULL;
        parts = 1U;
        cores = items;
    }
}

static void ResolveExecPlan(
    uint32_t b, uint32_t h, uint32_t tag, uint32_t coreLim,
    uint64_t &mode, uint32_t &parts, uint32_t &cores) {
    static constexpr PlanItem kPlanLut[4][3] = {
        {{0U, 3U, 12U}, {1U, 3U, 12U}, {2U, 4U, 16U}},
        {{3U, 3U, 24U}, {4U, 3U, 24U}, {5U, 3U, 24U}},
        {{6U, 2U, 24U}, {7U, 2U, 24U}, {8U, 2U, 24U}},
        {{15U, 0U, 1U}, {10U, 2U, 32U}, {11U, 2U, 32U}},
    };
    mode = 15ULL;
    parts = 0U;
    cores = std::min<uint32_t>(b * 4U, coreLim);
    if (tag < 3U && b >= 1U && b <= 4U) {
        const PlanItem item = kPlanLut[b - 1U][tag];
        const uint32_t minH = std::max<uint32_t>(4U, static_cast<uint32_t>(item.parts) * 2U);
        if (item.cores <= coreLim && h >= minH) {
            mode = static_cast<uint64_t>(item.mode);
            parts = static_cast<uint32_t>(item.parts);
            cores = static_cast<uint32_t>(item.cores);
            return;
        }
    }
    if (tag >= 3U || b == 0U || h < 4U || (b * 4U) > coreLim) {
        return;
    }
    SelectVarPlan(b, h, tag, coreLim, mode, parts, cores);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::StorageShape *inShape = context->GetInputShape(0);
    const gert::CompileTimeTensorDesc *inDesc = context->GetRequiredInputDesc(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (inShape == nullptr || inDesc == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape = inShape->GetStorageShape();
    if (shape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    const float *pTau = attrs->GetFloat(0);
    const int64_t *pStride = attrs->GetInt(1);
    if (pTau == nullptr || pStride == nullptr || *pTau <= 0.0f) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batch = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t height = static_cast<uint32_t>(shape.GetDim(2));
    const uint32_t width = static_cast<uint32_t>(shape.GetDim(3));
    const uint32_t coreLim = GetCoreLimit(context);
    const uint32_t tag = MatchGridTag(height, width);

    uint64_t mode = 15ULL;
    uint32_t parts = 0U;
    uint32_t cores = 1U;
    ResolveExecPlan(batch, height, tag, coreLim, mode, parts, cores);

    const uint32_t dt = static_cast<uint32_t>(inDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, dt, mode);

    auto *td = context->GetTilingData<FusedDisconnectorKeypointSoftArgmaxTilingData>();
    if (td == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const float tauVal = *pTau;
    td->f0 = (tauVal > 0.0f) ? (1.0f / tauVal) : 1.0f;
    td->stride = static_cast<float>(*pStride);
    td->batch = batch;
    td->height = height;
    td->width = width;
    td->n0 = cores;
    td->n1 = parts;

    context->SetBlockDim(cores);
    size_t *ws = context->GetWorkspaceSizes(1);
    if (ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inShape = context->GetInputShape(0);
    gert::Shape *kpShape = context->GetOutputShape(0);
    gert::Shape *scShape = context->GetOutputShape(1);
    if (inShape == nullptr || kpShape == nullptr || scShape == nullptr) {
        return GRAPH_FAILED;
    }
    const int64_t b = inShape->GetDim(0);
    kpShape->SetDimNum(3);
    kpShape->SetDim(0, b);
    kpShape->SetDim(1, 4);
    kpShape->SetDim(2, 2);

    scShape->SetDimNum(2);
    scShape->SetDim(0, b);
    scShape->SetDim(1, 4);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    const ge::DataType dt = context->GetInputDataType(0);
    context->SetOutputDataType(0, dt);
    context->SetOutputDataType(1, dt);
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class FusedDisconnectorKeypointSoftArgmax : public OpDef {
public:
    explicit FusedDisconnectorKeypointSoftArgmax(const char *name) : OpDef(name) {
        this->Input("feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("keypoints")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("scores")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Attr("tau").AttrType(REQUIRED).Float();
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(FusedDisconnectorKeypointSoftArgmax);

}
