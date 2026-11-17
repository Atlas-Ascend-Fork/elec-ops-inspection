#include <algorithm>
#include <cmath>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "../op_kernel/tiling_key_fused_disconnector_keypoint_soft_argmax.h"

namespace optiling {
    static bool IsDenseFloatNd(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *desc) {
        return shape != nullptr && desc != nullptr && shape->GetOriginShape() == shape->GetStorageShape() &&
            desc->GetDataType() == ge::DT_FLOAT && desc->GetStorageFormat() == ge::FORMAT_ND;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        for (size_t i = 0; i < 3; ++i) {
            if (!IsDenseFloatNd(context->GetInputShape(i), context->GetInputDesc(i))) {
                return ge::GRAPH_FAILED;
            }
        }
        const auto &feat = context->GetInputShape(0)->GetStorageShape();
        const auto &weight = context->GetInputShape(1)->GetStorageShape();
        const auto &bias = context->GetInputShape(2)->GetStorageShape();
        const auto *attrs = context->GetAttrs();
        if (attrs == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const auto *tau = attrs->GetFloat(0);
        const auto *stride = attrs->GetInt(1);
        if (feat.GetDimNum() != 4 || weight.GetDimNum() != 4 || bias.GetDimNum() != 1 ||
            feat.GetDim(1) != 32 || weight.GetDim(0) != 4 || weight.GetDim(1) != 32 ||
            weight.GetDim(2) != 3 || weight.GetDim(3) != 3 || bias.GetDim(0) != 4 ||
            tau == nullptr || !(*tau > 0.0f) || !std::isfinite(*tau) || stride == nullptr || *stride != 4) {
            return ge::GRAPH_FAILED;
        }

        const auto batch = feat.GetDim(0);
        const auto height = feat.GetDim(2);
        const auto width = feat.GetDim(3);
        if (batch < 1 || batch > 4 ||
            !((height == 48 && width == 64) || (height == 64 && width == 64) ||
              (height == 72 && width == 96))) {
            return ge::GRAPH_FAILED;
        }
        const auto *keypoints = context->GetOutputShape(0);
        const auto *scores = context->GetOutputShape(1);
        if (!IsDenseFloatNd(keypoints, context->GetOutputDesc(0)) ||
            !IsDenseFloatNd(scores, context->GetOutputDesc(1)) ||
            keypoints->GetStorageShape() != gert::Shape({batch, 4, 2}) ||
            scores->GetStorageShape() != gert::Shape({batch, 4})) {
            return ge::GRAPH_FAILED;
        }

        const uint32_t tileHeight = height == 72 ? 6 : 4;
        const uint32_t tileWidth = 32;
        const uint32_t tilesPerBatch = ((height + tileHeight - 1) / tileHeight) *
            ((width + tileWidth - 1) / tileWidth);
        const uint32_t tasks = static_cast<uint32_t>(batch) * tilesPerBatch;
        platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
        const uint32_t available = platform.GetCoreNumAiv();
        uint64_t ubBytes = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
        constexpr uint64_t requiredUbBytes = 112 * 1024;
        if (available < 2 || ubBytes < requiredUbBytes) {
            return ge::GRAPH_FAILED;
        }
        const uint32_t blocks = std::min(tasks, available);
        if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        uint32_t DT_FEAT = static_cast<uint32_t>(ge::DT_FLOAT);
        uint32_t SHAPE_ID = height == 48 ? 1 : (height == 64 ? 2 : 3);
        uint32_t UNIT_TAU = *tau == 1.0f ? 1U : 0U;
        ASCENDC_TPL_SEL_PARAM(context, DT_FEAT, SHAPE_ID, UNIT_TAU);
        auto *tiling = context->GetTilingData<FusedDisconnectorKeypointSoftArgmaxTilingData>();
        tiling->batch = static_cast<uint32_t>(batch);
        tiling->height = static_cast<uint32_t>(height);
        tiling->width = static_cast<uint32_t>(width);
        tiling->stride = static_cast<uint32_t>(*stride);
        tiling->tau = *tau;

        tiling->tileHeight = tileHeight;
        tiling->tileWidth = tileWidth;
        tiling->tilesPerBatch = tilesPerBatch;
        tiling->blocks = blocks;

        context->SetBlockDim(blocks);
        context->GetWorkspaceSizes(1)[0] =
            static_cast<uint64_t>(tasks) * FUSED_KEYPOINT_TILE_RECORD_BYTES + 64;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const auto *feat = context->GetInputShape(0);
        if (feat == nullptr || feat->GetDimNum() != 4) {
            return GRAPH_FAILED;
        }

        const auto batch = feat->GetDim(0);
        auto *keypoints = context->GetOutputShape(0);
        keypoints->SetDimNum(3);
        keypoints->SetDim(0, batch);
        keypoints->SetDim(1, 4);
        keypoints->SetDim(2, 2);
        auto *scores = context->GetOutputShape(1);
        scores->SetDimNum(2);
        scores->SetDim(0, batch);
        scores->SetDim(1, 4);
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, DT_FLOAT);
        context->SetOutputDataType(1, DT_FLOAT);
        return GRAPH_SUCCESS;
    }
}  // namespace ge

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
}  // namespace ops
