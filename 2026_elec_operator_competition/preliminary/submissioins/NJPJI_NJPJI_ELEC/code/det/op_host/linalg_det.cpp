// Host侧Tiling实现
// 第 6 题：电力系统线性化分析算子 LinalgDet（det）
// A[*][n][n] float32 -> det[*] float32；det = sign(P)·∏U_ii（部分主元 LU）
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/linalg_det_tiling.h"
#include "../op_kernel/tiling_key_linalg_det.h"

namespace {
constexpr int64_t MAT_BUDGET = 176 * 1024;   // 矩阵/行 tile 可用的 UB 预算（字节）

constexpr uint32_t BLOCK_MAX = 40;
}  // namespace

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 输入信息：A[*][n][n]
        const gert::Tensor *tensor_A = context->GetRequiredInputTensor(0);
        ge::DataType dtype_A = tensor_A->GetDataType();
        uint32_t length_A = tensor_A->GetShapeSize();

        const gert::Shape &aShape = tensor_A->GetShape().GetStorageShape();
        const size_t dimNum = aShape.GetDimNum();
        int64_t n = (dimNum >= 1) ? aShape.GetDim(dimNum - 1) : 0;
        int64_t numMatrix = 1;
        for (size_t i = 0; i + 2 < dimNum; ++i) {
            const int64_t d = aShape.GetDim(i);
            numMatrix *= (d > 0 ? d : 1);
        }
        if (dimNum < 2) n = 0;
        if (n < 0) n = 0;
        if (numMatrix < 0) numMatrix = 0;

        int64_t stride = (n + 7) / 8 * 8;   // 行跨度补到 8 元素（32B）对齐
        if (stride < 8 && n > 0) stride = 8;

        // 行 tile：整矩阵能进 UB 就驻留，否则按行 tile 流式
        int64_t tileRows = 1;
        if (stride > 0) {
            const int64_t rowBytes = stride * (int64_t)sizeof(float);
            tileRows = MAT_BUDGET / rowBytes;   // 搬运已按 burst 上限分批，只需受 UB 约束
        }
        if (tileRows < 1) tileRows = 1;
        if (tileRows > n) tileRows = n;
        if (tileRows > 128) tileRows = 128;   // 主元归约缓冲 1024 长，rows*8 不能超
        if (tileRows < 1) tileRows = 1;

        uint32_t blockDim = 1;
        if (numMatrix > 1) {
            blockDim = static_cast<uint32_t>(numMatrix > (int64_t)BLOCK_MAX ? BLOCK_MAX : numMatrix);
            if (num_cores_aiv > 0 && blockDim > static_cast<uint32_t>(num_cores_aiv)) {
                blockDim = static_cast<uint32_t>(num_cores_aiv);
            }
            if (blockDim < 1) blockDim = 1;
        }

        // tiling key：按输入数据类型区分 kernel 模板实例
        uint32_t DT_A = static_cast<uint32_t>(dtype_A);
        ASCENDC_TPL_SEL_PARAM(context, DT_A);

        // 填充 tiling
        LinalgDetTilingData *tiling = context->GetTilingData<LinalgDetTilingData>();
        tiling->length = length_A;
        tiling->numMatrix = static_cast<uint32_t>(numMatrix);
        tiling->n = static_cast<uint32_t>(n);
        tiling->stride = static_cast<uint32_t>(stride);
        tiling->tileRows = static_cast<uint32_t>(tileRows);
        tiling->blockNum = blockDim;

        context->SetBlockDim(blockDim);
        // streaming 路径每核需要 n*stride 个 float 的 workspace
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        const bool needWs = (n > tileRows);
        // 注意：框架实际启动的核数可能大于这里 SetBlockDim 的值，workspace 必须按**最大核数**
        // 预留每核独立的 n*stride 段；否则框架会因空间不足而给 kernel 传空指针
        // （实测：按 2 核报 528KB 时 kernel 收到 nullptr，n>216 的 streaming 路径直接返回 0）。
        // 额外留 16MB 余量：框架除每核 n*stride 段外还有系统保留区，
        // 报得刚好够时会被判定不足而给 kernel 传空指针（实测 n=256 卡在边界）。
        currentWorkspace[0] = needWs
            ? ((size_t)BLOCK_MAX * (size_t)n * (size_t)stride * sizeof(float)
               + (size_t)16 * 1024 * 1024)
            : 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *aShape = context->GetInputShape(0);
        gert::Shape *outShape = context->GetOutputShape(0);
        if (aShape == nullptr || outShape == nullptr) {
            return GRAPH_FAILED;
        }
        const size_t dimNum = aShape->GetDimNum();
        const int64_t outDim = (dimNum >= 2) ? (int64_t)dimNum - 2 : 0;
        outShape->SetDimNum(outDim);
        for (int64_t i = 0; i < outDim; ++i) {
            outShape->SetDim(i, aShape->GetDim((size_t)i));
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_FLOAT);
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class LinalgDet : public OpDef {
    public:
        explicit LinalgDet(const char *name) : OpDef(name) {
            this->Input("A")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Output("det")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(LinalgDet);
}  // namespace ops
