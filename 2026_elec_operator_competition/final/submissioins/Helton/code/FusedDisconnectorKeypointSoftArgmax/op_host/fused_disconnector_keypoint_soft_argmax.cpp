// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/launch_plan.h"
#include "../op_kernel/tiling_key_fused_disconnector_keypoint_soft_argmax.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const auto* input = context->GetInputShape(0);
        const auto* attrs = context->GetAttrs();
        if (!input || !attrs || input->GetStorageShape().GetDimNum() != 4)
            return ge::GRAPH_FAILED;
        const auto* tau = attrs->GetFloat(0);
        const auto* stride = attrs->GetInt(1);
        if (!tau || !stride || *stride != 4) return ge::GRAPH_FAILED;
        const auto& shape = input->GetStorageShape();
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto* plan = context->GetTilingData<wave1::Tiling>();
        if (!plan || !wave1::MakePlan(*plan, shape.GetDim(0), shape.GetDim(2), shape.GetDim(3),
                                     *tau, *stride, platform.GetCoreNumAiv()))
            return ge::GRAPH_FAILED;
        const bool split = plan->blocks > plan->batch * 4;
        const uint32_t path = split ? 1 : 0;
        ASCENDC_TPL_SEL_PARAM(context, path);
        context->SetBlockDim(plan->blocks);
        // This op uses basic AIV instructions only. Allocate exactly one aligned
        // record per block; the kernel uses the allocation base set by its stub.
        if (split) context->SetScheduleMode(1);
        context->GetWorkspaceSizes(1)[0] = split ? plan->blocks * 32 : 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const auto* input = context->GetInputShape(0);
        if (!input || input->GetDimNum() != 4) return ge::GRAPH_FAILED;
        auto* keypoints = context->GetOutputShape(0);
        auto* scores = context->GetOutputShape(1);
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
}  // namespace ops
