// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/swi_glu_tiling.h"
#include "../op_kernel/tiling_key_swi_glu.h"
// 定义后 swi_glu_tiling_search.h 才会调用 CANN 的 matmul tiling API 构造 Cube 路径 tiling
#define SWIGLU_CUBE_TILING_AVAILABLE 1
// ===== 性能归因诊断开关（默认关闭）=====
// 取消下一行注释后，Cube 路径只跑两个 Matmul、跳过 epilogue（useCube=2 是 kernel 的 matmul-only 取值），
// 判题会显示 ~100% 输出错误，但「用时」列有效：它把总耗时拆成「Matmul + KFC 侧」与「epilogue 侧」。
// 仅用于诊断，测完必须注释回去（正常版本保持 tiling->useCube ∈ {0,1}）。
// #define SWIGLU_ABLATE_MM 1
#include "swi_glu_tiling_search.h"  // 切分参数搜索（纯 C++，与 CPU 调测工程共用同一份实现）

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 获取平台信息
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    // MIX kernel 的 blockDim 是 AIC 个数（每块 AIC 配 1 个 AIV），cube 路径的并行度上限用 AIC 数
    int32_t num_cores_aic = platform.GetCoreNumAic();
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
    uint32_t coreAic = (num_cores_aic > 0) ? static_cast<uint32_t>(num_cores_aic) : coreAiv;
    swiglu_tiling::SwiGluFillTiling(m, d, ub_size, coreAiv, coreAic, tiling);

#if defined(SWIGLU_ABLATE_MM)
    // 诊断：只跑 Matmul（kernel 里 useCube==2 ⇒ 跳过 CubeEpilogue），用于定位耗时归属。
    if (tiling->useCube == 1u) {
        tiling->useCube = 2u;
    }
#endif

    // tiling key：SWIGLU_CUBE（1 = Cube/MIX_AIC_1_2 路径、0 = 纯矢量 AIV_ONLY 路径）。
    // 必须在 FillTiling 之后设置（useCube 由切分搜索决定）；顺序与 tiling_key_swi_glu.h 的
    // ARGS_SEL 一致，框架据此选择对应的 kernel 类型与函数实例。
    const uint32_t cubeKey = (tiling->useCube != 0u) ? 1u : 0u;
    ASCENDC_TPL_SEL_PARAM(context, cubeKey, DT_INPUT_X);

    context->SetBlockDim(tiling->blockDim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // workspace = 系统（KFC/libapi）部分 + 用户部分。
    //   * 系统部分：MIX/Cube 路径的 KFC 消息队列与 cube 服务端态放在 workspace 开头
    //     （RESERVED_WORKSPACE = 16MB @ 910B，与 GetLibApiWorkSpaceSize() 实测一致：
    //     libtiling_api 的 platform_ascendc.cpp.o 对 socVersion 0x899(=2201) 返回 0x1000000）。
    //     编译管线生成的 kernel 入口包装会先算 `usrWorkspace = workspace + RESERVED_WORKSPACE`
    //     再传给用户 kernel ⇒ kernel 里拿到的 workspace 实参已是用户区，不要再偏移。
    //   * 用户部分：Cube 路径存放 z1/z2（2*m*d*2 字节）；矢量路径不用，但仍按框架约定
    //     申报系统部分，避免运行时侧 workspace 检查失败。
    const size_t sysWorkspaceSize = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    size_t userWorkspaceSize = (tiling->useCube != 0u) ? static_cast<size_t>(tiling->wsBytes) : 0u;
    const size_t kWsAlign = 512u;
    userWorkspaceSize = (userWorkspaceSize + kWsAlign - 1u) / kWsAlign * kWsAlign;
    currentWorkspace[0] = sysWorkspaceSize + userWorkspaceSize;
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
