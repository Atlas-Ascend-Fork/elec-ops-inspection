
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/spline_conv_tiling.h"
#include "../op_kernel/tiling_key_spline_conv.h"

namespace {
constexpr int64_t UB_BUDGET = 180 * 1024;
constexpr int64_t EDGE_TILE = 512;
constexpr int64_t TILE_NODES_MAX = 64;
constexpr int64_t MAX_C = 128;
}

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);


        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_edge_index = context->GetRequiredInputTensor(1);
        const gert::Tensor *tensor_weight = context->GetRequiredInputTensor(3);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t length_x = tensor_x->GetShapeSize();

        const gert::Shape &xShape = tensor_x->GetShape().GetStorageShape();
        const gert::Shape &eiShape = tensor_edge_index->GetShape().GetStorageShape();
        const gert::Shape &wShape = tensor_weight->GetShape().GetStorageShape();
        int64_t numNodes = (xShape.GetDimNum() >= 1) ? xShape.GetDim(0) : 0;
        int64_t numCin = (xShape.GetDimNum() >= 2) ? xShape.GetDim(1) : 0;
        int64_t numEdges = (eiShape.GetDimNum() >= 2) ? eiShape.GetDim(1) : 0;
        int64_t numCout = (wShape.GetDimNum() >= 3) ? wShape.GetDim(2) : 0;
        if (numNodes < 0) numNodes = 0;
        if (numCin < 0) numCin = 0;
        if (numEdges < 0) numEdges = 0;
        if (numCout < 0) numCout = 0;


        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        int64_t dim = 2;
        int64_t degree = 1;
        int64_t ks[3] = {2, 2, 2};
        bool open[3] = {true, true, true};
        bool aggrMean = true;
        if (attrs != nullptr) {
            const int64_t *attr_dim = attrs->GetInt(0);
            if (attr_dim != nullptr) dim = *attr_dim;
            const gert::TypedContinuousVector<int64_t> *attr_ks = attrs->GetListInt(1);
            if (attr_ks != nullptr) {
                const int64_t *data = attr_ks->GetData();
                for (int64_t i = 0; i < attr_ks->GetSize() && i < 3; ++i) ks[i] = data[i];
            }
            const gert::TypedContinuousVector<int64_t> *attr_open = attrs->GetListInt(2);
            if (attr_open != nullptr) {
                const int64_t *data = attr_open->GetData();
                for (int64_t i = 0; i < attr_open->GetSize() && i < 3; ++i) open[i] = (data[i] != 0);
            }
            const int64_t *attr_degree = attrs->GetInt(3);
            if (attr_degree != nullptr) degree = *attr_degree;
            const char *attr_aggr = attrs->GetStr(4);
            if (attr_aggr != nullptr) aggrMean = (attr_aggr[0] == 'm' || attr_aggr[0] == 'M');
        }
        if (dim < 1) dim = 1;
        if (dim > 3) dim = 3;
        if (degree < 1) degree = 1;
        if (degree > 3) degree = 3;
        for (int64_t i = dim; i < 3; ++i) {
            ks[i] = 1;
            open[i] = false;
        }

        int64_t numBasis = 1;
        int64_t numKernel = 1;
        for (int64_t i = 0; i < dim; ++i) {
            numBasis *= (degree + 1);
            numKernel *= ks[i];
        }


        int64_t wBytes = numCin * numCout * (int64_t)sizeof(float);
        int64_t edgeBytes = EDGE_TILE * (int64_t)(sizeof(int64_t) * 2 + sizeof(float) * 3);
        int64_t avail = UB_BUDGET - (2 * wBytes + edgeBytes + 4096);
        int64_t tileNodes = TILE_NODES_MAX;
        if (numCout > 0 && avail > 0) {
            int64_t cap = avail / (numCout * (int64_t)sizeof(float));
            if (cap < tileNodes) tileNodes = cap;
        }
        if (tileNodes < 1) tileNodes = 1;
        if (tileNodes > MAX_C) tileNodes = MAX_C;
        // 分核自适应：tileNodes 不能大到让 blockDim 跑不满核（原来只看 UB 预算，忽略核数）
        {
            int64_t wantBlocks = (num_cores_aiv > 0) ? (int64_t)num_cores_aiv : 40;
            if (wantBlocks > 64) wantBlocks = 64;
            if (numNodes > 0 && wantBlocks > 0) {
                const int64_t capByNodes = (numNodes + wantBlocks - 1) / wantBlocks;
                if (capByNodes > 0 && tileNodes > capByNodes) tileNodes = capByNodes;
                if (tileNodes < 1) tileNodes = 1;
            }
        }


        uint32_t blockDim = 1;
        if (numNodes > 0) {
            blockDim = static_cast<uint32_t>((numNodes + tileNodes - 1) / tileNodes);
            if (blockDim > 64) blockDim = 64;
            if (blockDim < 1) blockDim = 1;
            if (num_cores_aiv > 0 && blockDim > static_cast<uint32_t>(num_cores_aiv)) {
                blockDim = static_cast<uint32_t>(num_cores_aiv);
            }
        }


        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);


        SplineConvTilingData *tiling = context->GetTilingData<SplineConvTilingData>();
        tiling->length = length_x;
        tiling->numNodes = static_cast<uint32_t>(numNodes);
        tiling->numEdges = static_cast<uint32_t>(numEdges);
        tiling->numCin = static_cast<uint32_t>(numCin);
        tiling->numCout = static_cast<uint32_t>(numCout);
        tiling->dim = static_cast<uint32_t>(dim);
        tiling->numBasis = static_cast<uint32_t>(numBasis);
        tiling->numKernel = static_cast<uint32_t>(numKernel);
        tiling->ks0 = static_cast<uint32_t>(ks[0]);
        tiling->ks1 = static_cast<uint32_t>(ks[1]);
        tiling->ks2 = static_cast<uint32_t>(ks[2]);
        tiling->open0 = open[0] ? 1u : 0u;
        tiling->open1 = open[1] ? 1u : 0u;
        tiling->open2 = open[2] ? 1u : 0u;
        tiling->degree = static_cast<uint32_t>(degree);
        tiling->aggrMean = aggrMean ? 1u : 0u;
        tiling->tileNodes = static_cast<uint32_t>(tileNodes);
        tiling->edgeTile = static_cast<uint32_t>(EDGE_TILE);
        tiling->blockNum = blockDim;

        context->SetBlockDim(blockDim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        const gert::Shape *wShape = context->GetInputShape(3);
        gert::Shape *outShape = context->GetOutputShape(0);
        if (xShape == nullptr || outShape == nullptr) {
            return GRAPH_FAILED;
        }
        const int64_t numNodes = (xShape->GetDimNum() >= 1) ? xShape->GetDim(0) : 0;
        int64_t numCout = 0;
        if (wShape != nullptr && wShape->GetDimNum() >= 3) {
            numCout = wShape->GetDim(2);
        }
        outShape->SetDimNum(2);
        outShape->SetDim(0, numNodes);
        outShape->SetDim(1, numCout);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class SplineConv : public OpDef {
    public:
        explicit SplineConv(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Input("edge_index")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND});
            this->Input("edge_attr")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Input("weight")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Attr("dim").AttrType(OPTIONAL).Int(2);
            this->Attr("kernel_size").AttrType(OPTIONAL).ListInt({2, 2});
            this->Attr("is_open_spline").AttrType(OPTIONAL).ListInt({1, 1});
            this->Attr("degree").AttrType(OPTIONAL).Int(1);
            this->Attr("aggr").AttrType(OPTIONAL).String("mean");
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(SplineConv);
}
