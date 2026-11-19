#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "../op_kernel/tiling_key_fused_disconnector_keypoint_soft_argmax.h"
#include <algorithm>
#include <cmath>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *feat = context->GetInputShape(0);
    const auto *weight = context->GetInputShape(1);
    const auto *bias = context->GetInputShape(2);
    const auto *attrs = context->GetAttrs();
    if (!feat || !weight || !bias || !attrs) return ge::GRAPH_FAILED;
    const auto &shape = feat->GetStorageShape();
    const auto &weightShape = weight->GetStorageShape();
    const auto &biasShape = bias->GetStorageShape();
    if (shape.GetDimNum() != 4 || shape.GetDim(1) != 32 ||
        weightShape.GetDimNum() != 4 || weightShape.GetDim(0) != 4 ||
        weightShape.GetDim(1) != 32 || weightShape.GetDim(2) != 3 ||
        weightShape.GetDim(3) != 3 || biasShape.GetDimNum() != 1 ||
        biasShape.GetDim(0) != 4) return ge::GRAPH_FAILED;

    const int64_t batch = shape.GetDim(0);
    const int64_t height = shape.GetDim(2);
    const int64_t width = shape.GetDim(3);
    const auto *tau = attrs->GetAttrPointer<float>(0);
    const auto *stride = attrs->GetAttrPointer<int64_t>(1);
    if (batch < 1 || batch > 4 || !((height == 48 && width == 64) ||
        (height == 64 && width == 64) || (height == 72 && width == 96)) ||
        !tau || !std::isfinite(*tau) || !(*tau > 0) || !stride || *stride != 4)
        return ge::GRAPH_FAILED;
    for (uint32_t input = 0; input < 3; ++input) {
        const auto *desc = context->GetInputDesc(input);
        if (!desc || desc->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t tiles = static_cast<uint32_t>(height / 2);
    const uint32_t batchCapacity = platform.GetCoreNumAiv() / static_cast<uint32_t>(batch);
    if (!batchCapacity) return ge::GRAPH_FAILED;
    const uint32_t jobsPerWorker = (tiles + batchCapacity - 1) / batchCapacity;
    const uint32_t workersPerBatch = (tiles + jobsPerWorker - 1) / jobsPerWorker;
    const uint32_t cores = workersPerBatch * static_cast<uint32_t>(batch);
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint64_t requiredUbBytes = width == 64 ? 188832 : 184224;
    const uint64_t sparseUbBytes = (height + 2) * (width + 16) * 4 + 5 * height * width * 4 + 6016;
    if (ubBytes < sparseUbBytes) return ge::GRAPH_FAILED;
    if (cores < static_cast<uint32_t>(batch * 4) || ubBytes < requiredUbBytes)
        return ge::GRAPH_FAILED;

    auto *data = context->GetTilingData<FusedDisconnectorKeypointSoftArgmaxTilingData>();
    if (!data) return ge::GRAPH_FAILED;
    data->batch = static_cast<uint32_t>(batch);
    data->height = static_cast<uint32_t>(height);
    data->width = static_cast<uint32_t>(width);
    data->tileCount = static_cast<uint32_t>(height / 2);
    data->coreCount = cores;
    data->tau = *tau;
    data->stride = static_cast<float>(*stride);
    uint32_t DT_FEAT = static_cast<uint32_t>(ge::DT_FLOAT);
    ASCENDC_TPL_SEL_PARAM(context, DT_FEAT);
    context->SetBlockDim(cores);
    context->SetScheduleMode(1);
    const size_t statisticsBytes = static_cast<size_t>(cores) * 64;
    context->GetWorkspaceSizes(1)[0] = statisticsBytes;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *input = context->GetInputShape(0);
    auto *keypoints = context->GetOutputShape(0);
    auto *scores = context->GetOutputShape(1);
    if (!input || !keypoints || !scores || input->GetDimNum() != 4)
        return GRAPH_FAILED;
    keypoints->SetDimNum(3);
    keypoints->SetDim(0, input->GetDim(0));
    keypoints->SetDim(1, 4);
    keypoints->SetDim(2, 2);
    scores->SetDimNum(2);
    scores->SetDim(0, input->GetDim(0));
    scores->SetDim(1, 4);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    for (uint32_t input = 0; input < 3; ++input) {
        if (context->GetInputDataType(input) != DT_FLOAT) return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, DT_FLOAT);
    context->SetOutputDataType(1, DT_FLOAT);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class FusedDisconnectorKeypointSoftArgmax : public OpDef {
public:
    explicit FusedDisconnectorKeypointSoftArgmax(const char *name) : OpDef(name)
    {
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
