// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/swi_glu_tiling.h"
#include "../op_kernel/tiling_key_swi_glu.h"
#include "swi_glu_tiling_search.h"  // 切分参数搜索（纯 C++，与 CPU 调测工程共用同一份实现）

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 获取平台信息
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    // 获取算子输入数组信息
    const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensor_bias_1 = context->GetRequiredInputTensor(2);
    if (tensor_input_x == nullptr || tensor_bias_1 == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType dtype_input_x = tensor_input_x->GetDataType();
    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    // 维度推导：d = input_x 最后一维；若与 bias_1 长度不一致且可整除，
    // 则以 bias_1 长度为准（兼容把 (m, d) 展平成 1 维的输入）。
    uint64_t total = static_cast<uint64_t>(tensor_input_x->GetShapeSize());
    uint32_t dLast = 1;
    const gert::StorageShape *xStorageShape = context->GetInputShape(0);
    if (xStorageShape != nullptr) {
        const gert::Shape &shape = xStorageShape->GetStorageShape();
        if (shape.GetDimNum() >= 1) {
            int64_t last = shape.GetDim(shape.GetDimNum() - 1);
            if (last > 0) {
                dLast = static_cast<uint32_t>(last);
            }
        }
    }
    uint32_t dBias = static_cast<uint32_t>(tensor_bias_1->GetShapeSize());
    uint32_t d = dLast;
    if (dBias > 0 && dBias != dLast && (total % dBias) == 0) {
        d = dBias;
    }
    if (d == 0) {
        d = 1;
    }
    uint64_t m64 = (total > 0) ? (total / d) : 1;
    if (m64 == 0) {
        m64 = 1;
    }
    uint32_t m = (m64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(m64);

    // 切分方案 + tiling 结构体填充
    SwiGluTilingData *tiling = context->GetTilingData<SwiGluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t coreAiv = (num_cores_aiv > 0) ? static_cast<uint32_t>(num_cores_aiv) : 1u;
    swiglu_tiling::SwiGluFillTiling(m, d, ub_size, coreAiv, tiling);

    context->SetBlockDim(tiling->coreNum);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        if (xShape == nullptr || yShape == nullptr) {
            return GRAPH_FAILED;
        }
        *yShape = *xShape;  // 输出形状与输入完全一致
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const ge::DataType xDtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, xDtype);
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class SwiGlu : public OpDef {
    public:
        explicit SwiGlu(const char *name) : OpDef(name) {
            this->Input("input_x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Input("weight_1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Input("bias_1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Input("weight_2")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Input("bias_2")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(SwiGlu);
}  // namespace ops
