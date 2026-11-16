// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fire_candidate_decode_fusion_tiling.h"
#include "../op_kernel/tiling_key_fire_candidate_decode_fusion.h"

namespace {
constexpr uint32_t TILE_MIN = 64;
constexpr uint32_t TILE_MAX = 1024;
constexpr uint32_t BLOCK_MAX = 64;
}  // namespace

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 输入信息：raw_pred[B,N,6]、priors[N,5]
        const gert::Tensor *tensor_raw_pred = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_priors = context->GetRequiredInputTensor(1);
        ge::DataType dtype_raw_pred = tensor_raw_pred->GetDataType();
        uint32_t length_raw_pred = tensor_raw_pred->GetShapeSize();

        const gert::Shape &predShape = tensor_raw_pred->GetShape().GetStorageShape();
        int64_t batch = (predShape.GetDimNum() >= 1) ? predShape.GetDim(0) : 0;
        int64_t numCand = (predShape.GetDimNum() >= 2) ? predShape.GetDim(1) : 0;
        if (batch < 0) batch = 0;
        if (numCand < 0) numCand = 0;

        // 分核自适应：先定「目标核数」，再反推 tileLen，保证 blockDim 能跑满核
        // （原来 tileLen 下限 TILE_MIN=64 会让小 numCand 只用 8 个核甚至更少）
        uint32_t blockDim = 1;
        uint32_t tileLen = TILE_MAX;
        if (batch > 0 && numCand > 0) {
            int64_t wantBlocks = (num_cores_aiv > 0) ? (int64_t)num_cores_aiv : (int64_t)BLOCK_MAX;
            if (wantBlocks > (int64_t)BLOCK_MAX) wantBlocks = (int64_t)BLOCK_MAX;
            if (wantBlocks < 1) wantBlocks = 1;
            // 每 batch 需要多少个 tile 才能凑够 wantBlocks 个并行块
            int64_t tilesPerBatch = (wantBlocks + batch - 1) / batch;
            if (tilesPerBatch < 1) tilesPerBatch = 1;
            int64_t tl = (numCand + tilesPerBatch - 1) / tilesPerBatch;
            tl = (tl + 7) / 8 * 8;               // 8 元素（32B）对齐
            if (tl < 8) tl = 8;
            if (tl > (int64_t)TILE_MAX) tl = (int64_t)TILE_MAX;
            tileLen = static_cast<uint32_t>(tl);
            const int64_t numNTiles = (numCand + (int64_t)tileLen - 1) / (int64_t)tileLen;
            int64_t totalTiles = batch * numNTiles;
            if (totalTiles > (int64_t)BLOCK_MAX) totalTiles = (int64_t)BLOCK_MAX;
            if (totalTiles < 1) totalTiles = 1;
            blockDim = static_cast<uint32_t>(totalTiles);
            if (num_cores_aiv > 0 && blockDim > static_cast<uint32_t>(num_cores_aiv)) {
                blockDim = static_cast<uint32_t>(num_cores_aiv);
            }
        }

        // tiling key：按 raw_pred 数据类型区分 kernel 模板实例
        uint32_t DT_RAW_PRED = static_cast<uint32_t>(dtype_raw_pred);
        ASCENDC_TPL_SEL_PARAM(context, DT_RAW_PRED);

        // 填充 tiling
        FireCandidateDecodeFusionTilingData *tiling =
            context->GetTilingData<FireCandidateDecodeFusionTilingData>();
        tiling->length = length_raw_pred;
        tiling->batch = static_cast<uint32_t>(batch);
        tiling->numCand = static_cast<uint32_t>(numCand);
        tiling->tileLen = tileLen;
        tiling->blockNum = blockDim;

        context->SetBlockDim(blockDim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *predShape = context->GetInputShape(0);
        gert::Shape *outShape = context->GetOutputShape(0);   // [B,N,5]
        if (predShape == nullptr || outShape == nullptr) {
            return GRAPH_FAILED;
        }
        const int64_t batch = (predShape->GetDimNum() >= 1) ? predShape->GetDim(0) : 0;
        const int64_t numCand = (predShape->GetDimNum() >= 2) ? predShape->GetDim(1) : 0;
        outShape->SetDimNum(3);
        outShape->SetDim(0, batch);
        outShape->SetDim(1, numCand);
        outShape->SetDim(2, 5);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));   // decoded dtype 与 raw_pred 一致
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class FireCandidateDecodeFusion : public OpDef {
    public:
        explicit FireCandidateDecodeFusion(const char *name) : OpDef(name) {
            // 注意：DataType 列表按索引组合成支持组合（不是笛卡尔积）。
            // 题面规定 priors 恒为 float32、decoded 跟随 raw_pred，
            // 因此组合必须是 (fp32,fp32,fp32) 与 (fp16,fp32,fp16)；
            // 之前三个都写 {FLOAT, FLOAT16}，fp16 输入用例会因组合未声明报 aclnn 161002。
            this->Input("raw_pred")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("priors")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("decoded")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(FireCandidateDecodeFusion);
}  // namespace ops
