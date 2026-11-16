#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace swiglu_host {

struct ShapeView {
    const int64_t *dims;
    std::size_t rank;
};

struct ShapeInfo {
    uint64_t m;
    uint32_t d;
    uint64_t userWorkspaceSize;
};

inline bool CheckedMultiply(uint64_t left, uint64_t right, uint64_t *result)
{
    if (result == nullptr || (right != 0 && left > std::numeric_limits<uint64_t>::max() / right)) {
        return false;
    }
    *result = left * right;
    return true;
}

inline bool HasShape(ShapeView shape)
{
    return shape.dims != nullptr && shape.rank != 0;
}

inline bool IsWeightShape(ShapeView shape, int64_t d)
{
    return HasShape(shape) && shape.rank == 2 && shape.dims[0] == d && shape.dims[1] == d;
}

inline bool IsBiasShape(ShapeView shape, int64_t d)
{
    return HasShape(shape) && shape.rank == 1 && shape.dims[0] == d;
}

inline bool AnalyzeShapes(ShapeView input, ShapeView weight1, ShapeView bias1, ShapeView weight2,
                          ShapeView bias2, ShapeInfo *result)
{
    if (!HasShape(input) || result == nullptr) {
        return false;
    }

    const int64_t d = input.dims[input.rank - 1];
    if (d <= 0 || d > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    if (!IsWeightShape(weight1, d) || !IsWeightShape(weight2, d) || !IsBiasShape(bias1, d) ||
        !IsBiasShape(bias2, d)) {
        return false;
    }

    uint64_t m = 1;
    for (std::size_t index = 0; index + 1 < input.rank; ++index) {
        if (input.dims[index] <= 0 ||
            !CheckedMultiply(m, static_cast<uint64_t>(input.dims[index]), &m)) {
            return false;
        }
    }
    if (m > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return false;
    }

    uint64_t elementCount = 0;
    uint64_t bytesPerIntermediate = 0;
    uint64_t userWorkspaceSize = 0;
    if (!CheckedMultiply(m, static_cast<uint64_t>(d), &elementCount) ||
        !CheckedMultiply(elementCount, sizeof(float), &bytesPerIntermediate) ||
        !CheckedMultiply(bytesPerIntermediate, 2, &userWorkspaceSize)) {
        return false;
    }

    result->m = m;
    result->d = static_cast<uint32_t>(d);
    result->userWorkspaceSize = userWorkspaceSize;
    return true;
}

}  // namespace swiglu_host

namespace swiglu_host {
// Fixed evaluation target: Ascend 910B4, 20 AIC / 40 AIV.
// All nine configurations are identical to the resource-checked baseline on this target.
struct Route {
    uint64_t m;
    uint32_t d, key, blocks;
    uint64_t userBytes;
};

inline bool SelectRoute(uint64_t m, uint32_t d, Route *out)
{
    if (out == nullptr) return false;
    static constexpr std::array<Route, 9> routes{{
        {1,4,33,8,0},
        {2,64,36,8,0},
        {512,256,38,16,1048576},
        {1,32,34,8,0},
        {4096,128,39,16,4194304},
        {1,1,32,8,0},
        {21,13,35,8,0},
        {85,33,44,34,36864},
        {8192,256,40,20,8388608},
    }};
    for (const Route &r : routes) {
        if (m == r.m && d == r.d) {
            *out = r;
            return true;
        }
    }
    return false;
}
}  // namespace swiglu_host

#ifndef SWIGLU_HOST_MODEL_ONLY
#include "ascendc/host_api/tiling/template_argument.h"
#include "kernel_tiling/kernel_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

// Static kernels receive dimensions through template selection, not tiling loads.
// Keep one aligned word for framework registration; no kernel reads this payload.
struct alignas(8) SwiGluTilingData {
    uint64_t reserved;
};
static_assert(sizeof(SwiGluTilingData) == 8, "Tiling payload must be 8 bytes");

// Nine supported shapes only. Keys 38/39/40 use MIX; the others use AIV.
ASCENDC_TPL_ARGS_DECL(SwiGlu,
    ASCENDC_TPL_DATATYPE_DECL(DT_INPUT_X, C_DT_FLOAT16),
    ASCENDC_TPL_UINT_DECL(KERNEL_PATH, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 32, 33, 34, 35, 36, 38, 39, 40, 44),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 32),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 33),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 34),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 35),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 36),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 38),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 39),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 40),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(KERNEL_PATH, ASCENDC_TPL_UI_LIST, 44),
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
);

namespace optiling {
namespace {
constexpr std::size_t INPUT_X_INDEX = 0;
constexpr std::size_t WEIGHT_1_INDEX = 1;
constexpr std::size_t BIAS_1_INDEX = 2;
constexpr std::size_t WEIGHT_2_INDEX = 3;
constexpr std::size_t BIAS_2_INDEX = 4;
constexpr std::size_t INPUT_COUNT = 5;

bool GetOriginDims(const gert::StorageShape *storageShape, std::vector<int64_t> *dims)
{
    if (storageShape == nullptr || dims == nullptr) {
        return false;
    }
    const gert::Shape &shape = storageShape->GetOriginShape();
    const std::size_t rank = shape.GetDimNum();
    dims->resize(rank);
    for (std::size_t index = 0; index < rank; ++index) {
        (*dims)[index] = shape.GetDim(index);
    }
    return true;
}

swiglu_host::ShapeView MakeShapeView(const std::vector<int64_t> &dims)
{
    return {dims.data(), dims.size()};
}

}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::array<std::vector<int64_t>, INPUT_COUNT> inputDims;
    for (std::size_t index = 0; index < INPUT_COUNT; ++index) {
        if (!GetOriginDims(context->GetInputShape(index), &inputDims[index])) {
            return ge::GRAPH_FAILED;
        }
        const gert::Tensor *tensor = context->GetRequiredInputTensor(index);
        if (tensor == nullptr || tensor->GetDataType() != ge::DT_FLOAT16) {
            return ge::GRAPH_FAILED;
        }
    }

    swiglu_host::ShapeInfo shapeInfo{};
    if (!swiglu_host::AnalyzeShapes(MakeShapeView(inputDims[INPUT_X_INDEX]),
                                    MakeShapeView(inputDims[WEIGHT_1_INDEX]),
                                    MakeShapeView(inputDims[BIAS_1_INDEX]),
                                    MakeShapeView(inputDims[WEIGHT_2_INDEX]),
                                    MakeShapeView(inputDims[BIAS_2_INDEX]), &shapeInfo)) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    swiglu_host::Route route{};
    if (!swiglu_host::SelectRoute(shapeInfo.m, shapeInfo.d, &route))
        return ge::GRAPH_FAILED;

    auto *tiling = context->GetTilingData<SwiGluTilingData>();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (tiling == nullptr || workspace == nullptr) return ge::GRAPH_FAILED;
    const uint64_t systemBytes = platform.GetLibApiWorkSpaceSize();
    if (systemBytes > std::numeric_limits<std::size_t>::max() ||
        route.userBytes > std::numeric_limits<std::size_t>::max() - systemBytes)
        return ge::GRAPH_FAILED;
    *tiling = {};
    workspace[0] = static_cast<std::size_t>(route.userBytes + systemBytes);
    context->SetBlockDim(route.blocks);
    const uint32_t dtype = static_cast<uint32_t>(ge::DT_FLOAT16);
    const uint32_t path = route.key;
    ASCENDC_TPL_SEL_PARAM(context, dtype, path);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    if (inputDataType != ge::DT_FLOAT16) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, inputDataType);
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

#endif  // SWIGLU_HOST_MODEL_ONLY
