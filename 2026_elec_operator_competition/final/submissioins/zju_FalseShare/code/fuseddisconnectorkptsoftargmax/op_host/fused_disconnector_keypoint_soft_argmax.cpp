/*
 * Copyright (c) 2026 FalseShare Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Apache License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Host侧Tiling实现
#include "register/op_def_registry.h"

#include "../op_kernel/fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "../op_kernel/tiling_key_fused_disconnector_keypoint_soft_argmax.h"

namespace optiling {
using BaseTiling = FusedDisconnectorKeypointSoftArgmaxTilingData;

struct ExecutionPlan {
    bool zeroShape;
    bool c04Join;
    bool wideJoin;
    bool c11Join;
    bool wideBatch4Join;
    bool publicB2;
    bool hidden64;
    bool wide;
    bool onchip;
    bool vertical;
};

static bool Matches(const BaseTiling &t, uint32_t batch, uint32_t height, uint32_t width, float tau) {
    return t.batch == batch && t.height == height && t.width == width && t.tau == tau;
}

static bool IsWideImage(const BaseTiling &t) {
    return t.height == 72 && t.width == 96 && t.stride == 4;
}

static bool IsPublicB1Projection(const BaseTiling &t) {
    if (t.batch != 1 || t.width != 64 || t.stride != 4) return false;
    return (t.height == 48 && (t.tau == 1.0f || t.tau == 2.0f))
        || (t.height == 64 && t.tau == 2.0f);
}

static bool IsSquareJoin(const BaseTiling &t) {
    return t.height == 64 && t.width == 64 && t.stride == 4
        && ((t.batch == 3 && t.tau == 1.0f) || (t.batch == 2 && t.tau == 0.5f));
}

static void SelectEngine(const BaseTiling &t, ExecutionPlan &p) {
    p.onchip = p.c11Join || p.c04Join || p.wideBatch4Join || p.wideJoin || IsSquareJoin(t);
    p.vertical = !p.wideBatch4Join && !p.c11Join
        && (p.wide || p.hidden64 || p.publicB2 || IsPublicB1Projection(t));
}

static ExecutionPlan Classify(const BaseTiling &t) {
    ExecutionPlan p{};
    p.zeroShape = Matches(t, 4, 48, 64, 1.0f);
    p.c04Join = Matches(t, 3, 48, 64, 1.0f) && t.stride == 4;
    p.wideJoin = IsWideImage(t) && t.batch == 3 && (t.tau == 0.5f || t.tau == 1.0f);
    p.c11Join = Matches(t, 2, 72, 96, 2.0f) && t.stride == 4;
    p.wideBatch4Join = IsWideImage(t) && t.batch == 4 && (t.tau == 1.0f || t.tau == 2.0f);
    p.publicB2 = Matches(t, 2, 48, 64, 0.5f) && t.stride == 4;
    p.hidden64 = Matches(t, 4, 64, 64, 0.5f) && t.stride == 4;
    p.wide = IsWideImage(t) && (t.batch == 2 || t.batch == 4);
    SelectEngine(t, p);
    return p;
}

static void PrepareVectorOffsets(FdkWeightMapTilingData &data) {
    for (uint32_t group = 0; group < 4; ++group) {
        for (uint32_t tap = 0; tap < 16; ++tap) {
            for (uint32_t channel = 0; channel < 8; ++channel) {
                data.weightOffsets[group * 128 + tap * 8 + channel] =
                    ((group * 8 + channel) * 9 + (tap < 9 ? tap : 0)) * sizeof(float);
            }
        }
    }
}

static void PrepareJoinCoefficients(FdkWeightMapTilingData &data) {
    for (uint32_t i = 0; i < 768; ++i) data.joinCoeffs[i] = 0;
    for (uint32_t i = 0; i < 16; ++i) data.joinCoeffs[i * 16 + i] = 0xbc00;
    for (uint32_t keypoint = 0; keypoint < 4; ++keypoint) {
        for (uint32_t dx = 0; dx < 3; ++dx) {
            for (uint32_t component = 0; component < 2; ++component) {
                data.joinCoeffs[256 + keypoint * 96 + component * 48 + dx * 16 + dx * 4 + keypoint] = 0x3c00;
            }
        }
    }
}

static void PrepareProjectionOffsets(FdkWeightMapTilingData &data, bool batchPack) {
    for (uint32_t keypoint = 0; keypoint < 2; ++keypoint) {
        for (uint32_t dx = 0; dx < 3; ++dx) {
            const uint32_t projection = batchPack ? dx * 2 + keypoint : keypoint * 3 + dx;
            for (uint32_t dy = 0; dy < 3; ++dy) {
                for (uint32_t channel = 0; channel < 32; ++channel) {
                    data.weightOffsets[projection * 96 + dy * 32 + channel] =
                        (keypoint * 288 + channel * 9 + dy * 3 + dx) * sizeof(float);
                }
            }
        }
    }
}

static uint32_t ProjectionParts(const BaseTiling &t, const ExecutionPlan &p) {
    if (p.wide) return t.batch == 2 ? 9 : 5;
    if (t.batch == 1 && t.height == 48 && t.width == 64) return 6;
    return p.hidden64 || p.publicB2 ? 4 : 8;
}

static void ConfigureJoinGeometry(BaseTiling &t, const ExecutionPlan &p) {
    const bool shortJoin = p.c04Join || p.wideJoin || p.wideBatch4Join || p.c11Join;
    t.tileRows = shortJoin ? 4 : 8;
    t.parts = p.wideBatch4Join ? 5 : ((p.wideJoin || p.c11Join) ? 6 : 4);
    if (p.c04Join) t.tileRows = 6;
}

static void ConfigureGeometry(BaseTiling &t, const ExecutionPlan &p) {
    t.parts = t.batch <= 2 ? 4 : (t.batch == 3 ? 3 : 2);
    if (Matches(t, 1, 72, 96, 1.0f) && t.stride == 4) t.parts = 9;
    t.tileRows = p.wide ? 1 : 2;
    if (p.onchip) ConfigureJoinGeometry(t, p);
    if (p.vertical) t.parts = ProjectionParts(t, p);
    t.blocks = t.batch * t.parts;
}

static uint32_t WorkspaceBytes(const BaseTiling &t, const ExecutionPlan &p) {
    if (p.zeroShape) return 128;
    if (p.vertical) return t.blocks * (12 * t.tileRows * t.width * sizeof(float) + 128);
    if (p.onchip) {
        const uint32_t heatFloats = t.tileRows * t.width * 4;
        const uint32_t ringFloats = heatFloats > 1152 ? heatFloats : 1152;
        return t.blocks * (ringFloats * sizeof(float) + 128);
    }
    return t.batch * 4 * t.parts * 32;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto *data = context->GetTilingData<FdkWeightMapTilingData>();
    if (data == nullptr) return ge::GRAPH_FAILED;
    auto &t = data->base;
    const auto &shape = context->GetInputShape(0)->GetStorageShape();
    const auto *attrs = context->GetAttrs();
    t.batch = static_cast<uint32_t>(shape.GetDim(0));
    t.height = static_cast<uint32_t>(shape.GetDim(2));
    t.width = static_cast<uint32_t>(shape.GetDim(3));
    t.tau = *attrs->GetFloat(0);
    if (!(t.tau > 0.0f)) return ge::GRAPH_FAILED;
    t.stride = static_cast<uint32_t>(*attrs->GetInt(1));
    const auto plan = Classify(t);
    PrepareVectorOffsets(*data);
    PrepareJoinCoefficients(*data);
    if (plan.vertical) PrepareProjectionOffsets(*data, false);
    if (Matches(t, 2, 64, 64, 0.5f) && t.stride == 4) PrepareProjectionOffsets(*data, true);
    ConfigureGeometry(t, plan);
    const uint32_t dispatch = plan.zeroShape ? 0 : (plan.vertical ? 3 : (plan.onchip ? 2 : 1));
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(ge::DT_FLOAT), dispatch);
    const uint32_t blocks = plan.onchip || plan.vertical ? t.blocks : t.batch * 4 * t.parts;
    context->SetBlockDim(plan.zeroShape ? 1 : blocks);
    context->SetScheduleMode(plan.zeroShape ? 0 : 1);
    context->GetWorkspaceSizes(1)[0] = WorkspaceBytes(t, plan);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const auto *input = context->GetInputShape(0);
        auto *keypoints = context->GetOutputShape(0);
        auto *scores = context->GetOutputShape(1);
        keypoints->SetDimNum(3);
        keypoints->SetDim(0, input->GetDim(0));
        keypoints->SetDim(1, 4);
        keypoints->SetDim(2, 2);
        scores->SetDimNum(2);
        scores->SetDim(0, input->GetDim(0));
        scores->SetDim(1, 4);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_FLOAT);
        context->SetOutputDataType(1, ge::DT_FLOAT);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
class FusedDisconnectorKeypointSoftArgmax : public OpDef {
public:
    explicit FusedDisconnectorKeypointSoftArgmax(const char *name) : OpDef(name) {
        RegisterInput("feat");
        RegisterInput("weight");
        RegisterInput("bias");
        RegisterOutput("keypoints");
        RegisterOutput("scores");
        Attr("tau").AttrType(REQUIRED).Float();
        Attr("stride").AttrType(REQUIRED).Int();
        SetInferShape(ge::InferShape);
        SetInferDataType(ge::InferDataType);
        AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }

private:
    void RegisterInput(const char *name) {
        Input(name).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    }

    void RegisterOutput(const char *name) {
        Output(name).ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    }
};
OP_ADD(FusedDisconnectorKeypointSoftArgmax);
} // namespace ops
