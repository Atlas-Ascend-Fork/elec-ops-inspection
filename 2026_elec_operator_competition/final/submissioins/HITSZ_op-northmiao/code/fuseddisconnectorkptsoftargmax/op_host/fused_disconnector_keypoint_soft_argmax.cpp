/*
 * Copyright (c) 2026 HITSZ_op-northmiao.
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

// HYBRID_V45_CLEAN tiling: retain the submitted shape-to-kernel mapping.
#include <algorithm>
#include <cmath>
#include "register/op_def_registry.h"
#include "../op_kernel/fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "../op_kernel/tiling_key_fused_disconnector_keypoint_soft_argmax.h"

namespace optiling {
static bool IsFastShape(const FusedDisconnectorKeypointSoftArgmaxTilingData& t) {
    return t.channels == 32u && ((t.height == 48u && t.width == 64u) ||
        (t.height == 64u && t.width == 64u) || (t.height == 72u && t.width == 96u));
}

static void ReadAttributes(gert::TilingContext *context, FusedDisconnectorKeypointSoftArgmaxTilingData& t) {
    t.tau = 1.0f;
    t.stride = 4.0f;
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) return;
    const float *tau = attrs->GetFloat(0);
    const int64_t *stride = attrs->GetInt(1);
    if (tau != nullptr) t.tau = *tau;
    if (stride != nullptr) t.stride = static_cast<float>(*stride);
}

static uint32_t ConfigureTiles(FusedDisconnectorKeypointSoftArgmaxTilingData& t, size_t& workspace) {
    t.blocks = std::min<uint32_t>(t.batch * 4u, 40u);
    t.tileRows = 1u;
    t.tiles = 0u;
    t.hideCase = 0u;
    workspace = 0u;
    if (!IsFastShape(t)) return 0u;
    if (t.batch <= 3u || (t.batch == 4u && t.height == 72u)) {
        t.tiles = (t.batch == 1u) ? 5u : ((t.batch == 2u) ? 3u : 2u);
        t.tileRows = (t.height + t.tiles - 1u) / t.tiles;
        t.blocks = t.batch * 4u * t.tiles;
        workspace = 4096u;
        return 1u;
    }
    t.tileRows = t.height;
    t.tiles = 1u;
    t.blocks = (t.batch == 4u) ? 4u : t.batch * 4u;
    return 0u;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr) return ge::GRAPH_FAILED;
    const gert::StorageShape *storage = context->GetInputShape(0);
    if (storage == nullptr) return ge::GRAPH_FAILED;
    const gert::Shape& shape = storage->GetStorageShape();
    if (shape.GetDimNum() < 4) return ge::GRAPH_FAILED;
    auto *t = context->GetTilingData<FusedDisconnectorKeypointSoftArgmaxTilingData>();
    if (t == nullptr) return ge::GRAPH_FAILED;
    t->batch = static_cast<uint32_t>(shape.GetDim(0));
    t->channels = static_cast<uint32_t>(shape.GetDim(1));
    t->height = static_cast<uint32_t>(shape.GetDim(2));
    t->width = static_cast<uint32_t>(shape.GetDim(3));
    ReadAttributes(context, *t);
    if (!(t->tau > 0.0f) || !std::isfinite(t->tau)) return ge::GRAPH_FAILED;
    size_t workspace = 0u;
    uint32_t mode = ConfigureTiles(*t, workspace);
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(ge::DT_FLOAT), mode);
    context->SetBlockDim(t->blocks);
    size_t *sizes = context->GetWorkspaceSizes(1);
    if (sizes == nullptr) return ge::GRAPH_FAILED;
    sizes[0] = workspace;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        if (context == nullptr) return GRAPH_FAILED;
        const gert::Shape *feat_shape = context->GetInputShape(0);
        gert::Shape *kpt_shape = context->GetOutputShape(0);
        gert::Shape *score_shape = context->GetOutputShape(1);
        if (feat_shape != nullptr && feat_shape->GetDimNum() >= 1) {
            int64_t B = feat_shape->GetDim(0);
            if (kpt_shape != nullptr) {
                kpt_shape->SetDimNum(3);
                kpt_shape->SetDim(0, B);
                kpt_shape->SetDim(1, 4);
                kpt_shape->SetDim(2, 2);
            }
            if (score_shape != nullptr) {
                score_shape->SetDimNum(2);
                score_shape->SetDim(0, B);
                score_shape->SetDim(1, 4);
            }
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        if (context == nullptr) return ge::GRAPH_FAILED;
        context->SetOutputDataType(0, ge::DT_FLOAT);
        context->SetOutputDataType(1, ge::DT_FLOAT);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    template <typename Parameter>
    static void ConfigureRequiredFloatNd(Parameter& parameter) {
        parameter.ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    }

    class FusedDisconnectorKeypointSoftArgmax : public OpDef {
    public:
        explicit FusedDisconnectorKeypointSoftArgmax(const char *name) : OpDef(name) {
            RegisterInputs();
            RegisterOutputs();
            this->Attr("tau").AttrType(REQUIRED).Float();
            this->Attr("stride").AttrType(REQUIRED).Int();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    private:
        void RegisterInputs() {
            ConfigureRequiredFloatNd(this->Input("feat"));
            ConfigureRequiredFloatNd(this->Input("weight"));
            ConfigureRequiredFloatNd(this->Input("bias"));
        }
        void RegisterOutputs() {
            ConfigureRequiredFloatNd(this->Output("keypoints"));
            ConfigureRequiredFloatNd(this->Output("scores"));
        }
    };
    OP_ADD(FusedDisconnectorKeypointSoftArgmax);
}  // namespace ops
