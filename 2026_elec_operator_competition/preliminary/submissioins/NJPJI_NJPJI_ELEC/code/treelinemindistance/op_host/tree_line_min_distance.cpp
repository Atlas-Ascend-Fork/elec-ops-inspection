// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/tree_line_min_distance_tiling.h"
#include "../op_kernel/tiling_key_tree_line_min_distance.h"

namespace {
constexpr uint32_t TILE_PT_MIN = 64;
constexpr uint32_t TILE_PT_MAX = 1024;
constexpr uint32_t SEG_TILE = 512;
constexpr uint32_t BLOCK_MAX = 40;
}  // namespace

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 输入信息：tree_points[B,N,3]、line_segments[B,M,2,3]
        const gert::Tensor *tensor_tree_points = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_line_segments = context->GetRequiredInputTensor(1);
        ge::DataType dtype_tree_points = tensor_tree_points->GetDataType();
        uint32_t length_tree_points = tensor_tree_points->GetShapeSize();

        const gert::Shape &ptShape = tensor_tree_points->GetShape().GetStorageShape();
        const gert::Shape &segShape = tensor_line_segments->GetShape().GetStorageShape();
        int64_t batch = (ptShape.GetDimNum() >= 1) ? ptShape.GetDim(0) : 0;
        int64_t numPoints = (ptShape.GetDimNum() >= 2) ? ptShape.GetDim(1) : 0;
        int64_t numSeg = (segShape.GetDimNum() >= 2) ? segShape.GetDim(1) : 0;
        if (batch < 0) batch = 0;
        if (numPoints < 0) numPoints = 0;
        if (numSeg < 0) numSeg = 0;

        // 分核自适应：先定目标核数，再反推 tilePoint，保证 blockDim 能跑满核
        // （原来 tilePoint 下限 TILE_PT_MIN=64 会让小 totalPts 只用几个核）
        uint32_t blockDim = 1;
        uint32_t tilePoint = TILE_PT_MAX;
        if (batch > 0 && numPoints > 0) {
            int64_t wantBlocks = (num_cores_aiv > 0) ? (int64_t)num_cores_aiv : (int64_t)BLOCK_MAX;
            if (wantBlocks > (int64_t)BLOCK_MAX) wantBlocks = (int64_t)BLOCK_MAX;
            if (wantBlocks < 1) wantBlocks = 1;
            int64_t tilesPerBatch = (wantBlocks + batch - 1) / batch;
            if (tilesPerBatch < 1) tilesPerBatch = 1;
            int64_t tp = (numPoints + tilesPerBatch - 1) / tilesPerBatch;
            tp = (tp + 7) / 8 * 8;                 // 8 元素（32B）对齐
            if (tp < 8) tp = 8;
            if (tp > (int64_t)TILE_PT_MAX) tp = (int64_t)TILE_PT_MAX;
            tilePoint = static_cast<uint32_t>(tp);
            const int64_t numTiles = (numPoints + (int64_t)tilePoint - 1) / (int64_t)tilePoint;
            int64_t totalTiles = batch * numTiles;
            if (totalTiles > (int64_t)BLOCK_MAX) totalTiles = (int64_t)BLOCK_MAX;
            if (totalTiles < 1) totalTiles = 1;
            blockDim = static_cast<uint32_t>(totalTiles);
            if (num_cores_aiv > 0 && blockDim > static_cast<uint32_t>(num_cores_aiv)) {
                blockDim = static_cast<uint32_t>(num_cores_aiv);
            }
        }

        // tiling key：按输入数据类型区分 kernel 模板实例
        uint32_t DT_TREE_POINTS = static_cast<uint32_t>(dtype_tree_points);
        ASCENDC_TPL_SEL_PARAM(context, DT_TREE_POINTS);

        // 填充 tiling
        TreeLineMinDistanceTilingData *tiling = context->GetTilingData<TreeLineMinDistanceTilingData>();
        tiling->length = length_tree_points;
        tiling->batch = static_cast<uint32_t>(batch);
        tiling->numPoints = static_cast<uint32_t>(numPoints);
        tiling->numSeg = static_cast<uint32_t>(numSeg);
        tiling->tilePoint = tilePoint;
        tiling->tileSeg = SEG_TILE;
        tiling->blockNum = blockDim;

        context->SetBlockDim(blockDim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *ptShape = context->GetInputShape(0);
        gert::Shape *distShape = context->GetOutputShape(0);   // [B,N]
        gert::Shape *idxShape = context->GetOutputShape(1);    // [B,N]
        if (ptShape == nullptr || distShape == nullptr || idxShape == nullptr) {
            return GRAPH_FAILED;
        }
        const int64_t batch = (ptShape->GetDimNum() >= 1) ? ptShape->GetDim(0) : 0;
        const int64_t numPoints = (ptShape->GetDimNum() >= 2) ? ptShape->GetDim(1) : 0;
        distShape->SetDimNum(2);
        distShape->SetDim(0, batch);
        distShape->SetDim(1, numPoints);
        idxShape->SetDimNum(2);
        idxShape->SetDim(0, batch);
        idxShape->SetDim(1, numPoints);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));   // 距离与输入同 dtype
        context->SetOutputDataType(1, ge::DT_INT32);                   // 索引 int32
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class TreeLineMinDistance : public OpDef {
    public:
        explicit TreeLineMinDistance(const char *name) : OpDef(name) {
            this->Input("tree_points")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("line_segments")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("min_distance")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("nearest_segment_idx")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(TreeLineMinDistance);
}  // namespace ops
