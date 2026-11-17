#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "../op_kernel/tiling_key_fused_disconnector_keypoint_soft_argmax.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto* feat = context->GetInputShape(0);
    const auto* weight = context->GetInputShape(1);
    const auto* bias = context->GetInputShape(2);
    const auto* attrs = context->GetAttrs();
    const auto* featDesc=context->GetInputDesc(0);
    const auto* weightDesc=context->GetInputDesc(1);
    const auto* biasDesc=context->GetInputDesc(2);
    if (!feat || !weight || !bias || !attrs || !featDesc || !weightDesc || !biasDesc) return ge::GRAPH_FAILED;
    const float* tau = attrs->GetFloat(0);
    const int64_t* stride = attrs->GetInt(1);
    if (!tau || !stride) return ge::GRAPH_FAILED;
    const auto& fs = feat->GetStorageShape();
    const auto& ws = weight->GetStorageShape();
    const auto& bs = bias->GetStorageShape();
    if (fs.GetDimNum() != 4 || ws.GetDimNum() != 4 || bs.GetDimNum() != 1 ||
        fs.GetDim(0) < 1 || fs.GetDim(0) > 4 || fs.GetDim(1) != 32 ||
        ws.GetDim(0) != 4 || ws.GetDim(1) != 32 || ws.GetDim(2) != 3 || ws.GetDim(3) != 3 || bs.GetDim(0) != 4 ||
        featDesc->GetDataType() != ge::DT_FLOAT || weightDesc->GetDataType() != ge::DT_FLOAT || biasDesc->GetDataType() != ge::DT_FLOAT ||
        *stride != 4 || !(*tau > 0)) return ge::GRAPH_FAILED;
    const uint32_t b = static_cast<uint32_t>(fs.GetDim(0));
    const uint32_t h = static_cast<uint32_t>(fs.GetDim(2));
    const uint32_t w = static_cast<uint32_t>(fs.GetDim(3));
    if (!((h == 48 && w == 64) || (h == 64 && w == 64) || (h == 72 && w == 96))) return ge::GRAPH_FAILED;
    auto* data = context->GetTilingData<FusedDisconnectorKeypointSoftArgmaxTilingData>();
    data->batch = b; data->height = h; data->width = w; data->stride = 4; data->tau = *tau;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t availableCores=platform.GetCoreNumAiv();
    if (availableCores==0) return ge::GRAPH_FAILED;
    const uint32_t outputChannels=b*4;
    const uint32_t vectorCores=outputChannels<availableCores?outputChannels:availableCores;
    context->SetBlockDim(vectorCores);
    context->GetWorkspaceSizes(1)[0]=0;
    const uint32_t DT_FEAT = static_cast<uint32_t>(ge::DT_FLOAT);
    ASCENDC_TPL_SEL_PARAM(context, DT_FEAT);
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    const auto* input = context->GetInputShape(0);
    auto* points = context->GetOutputShape(0);
    auto* scores = context->GetOutputShape(1);
    if (!input || !points || !scores || input->GetDimNum() != 4) return GRAPH_FAILED;
    points->SetDimNum(3); points->SetDim(0, input->GetDim(0)); points->SetDim(1, 4); points->SetDim(2, 2);
    scores->SetDimNum(2); scores->SetDim(0, input->GetDim(0)); scores->SetDim(1, 4);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, ge::DT_FLOAT); context->SetOutputDataType(1, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}
namespace ops {
class FusedDisconnectorKeypointSoftArgmax : public OpDef {
public:
    explicit FusedDisconnectorKeypointSoftArgmax(const char* name) : OpDef(name) {
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
}
