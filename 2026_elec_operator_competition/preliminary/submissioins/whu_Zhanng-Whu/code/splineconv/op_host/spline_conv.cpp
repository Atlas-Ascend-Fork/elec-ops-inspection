/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/spline_conv_tiling.h"
#include "../op_kernel/tiling_key_spline_conv.h"
#include <algorithm>
#include <cstring>

namespace optiling {
static bool SetAggregation(const char *aggr, SplineConvTilingData &t)
{
    t.mean = !aggr || !aggr[0] || std::strcmp(aggr, "mean") == 0;
    return t.mean || std::strcmp(aggr, "add") == 0 || std::strcmp(aggr, "sum") == 0;
}

static bool SetSplineAttributes(gert::TilingContext *context, SplineConvTilingData &t)
{
    const auto *attrs = context->GetAttrs();
    if (!attrs) {
        return false;
    }
    const auto *dim = attrs->GetInt(0);
    const auto *kernelSize = attrs->GetListInt(1);
    const auto *open = attrs->GetListInt(2);
    const auto *degree = attrs->GetInt(3);
    const char *aggr = attrs->GetStr(4);
    if (!dim || !kernelSize || !degree) {
        return false;
    }
    if (*dim < 1 || *dim > 8 || *degree < 0 || *degree > 3 || kernelSize->GetSize() == 0) {
        return false;
    }
    if (!SetAggregation(aggr, t)) {
        return false;
    }
    const bool defaultOpen = !open || open->GetSize() == 0 || (open->GetSize() == 1 && open->GetData()[0] == -1);
    t.dim = *dim;
    t.degree = *degree;
    t.kernels = 1;
    t.slots = 1;
    for (uint32_t d = 0; d < t.dim; ++d) {
        const int64_t size = kernelSize->GetData()[std::min<uint32_t>(d, kernelSize->GetSize() - 1)];
        if (size < 1 || size > 65535 || t.kernels > 65535 / size) {
            return false;
        }
        const bool isOpen = defaultOpen || open->GetData()[std::min<uint32_t>(d, open->GetSize() - 1)] != 0;
        t.ks[d] = size;
        t.scale[d] = size - *degree * isOpen;
        t.kernels *= size;
        t.slots *= *degree + 1;
    }
    return true;
}

static bool SetNodeLayout(const gert::Shape &x, const gert::Shape &weight, SplineConvTilingData &t)
{
    if (x.GetDimNum() != 2 || weight.GetDimNum() != 3) {
        return false;
    }
    const int64_t n = x.GetDim(0);
    const int64_t ci = x.GetDim(1);
    t.weightTrans = weight.GetDim(1) != ci && weight.GetDim(2) == ci;
    const int64_t co = weight.GetDim(t.weightTrans ? 1 : 2);
    if (n < 1 || n > 65536 || ci < 0 || ci > 65535 || co < 1 || co > 65535) {
        return false;
    }
    if ((!t.weightTrans && weight.GetDim(1) != ci) || weight.GetDim(0) < t.kernels) {
        return false;
    }
    t.n = n;
    t.ci = ci;
    t.co = co;
    return true;
}

static bool SetGraphLayout(const gert::Shape &edges, const gert::Shape &pseudo, SplineConvTilingData &t)
{
    if (edges.GetDimNum() != 2 || pseudo.GetDimNum() != 2) {
        return false;
    }
    t.edgeTrans = edges.GetDim(0) != 2 && edges.GetDim(1) == 2;
    const int64_t e = edges.GetDim(t.edgeTrans ? 0 : 1);
    if (e < 0 || e > 262144 || (edges.GetDim(0) != 2 && !t.edgeTrans)) {
        return false;
    }
    const bool pseudoRowMajor = pseudo.GetDim(0) == e && pseudo.GetDim(1) == t.dim;
    t.pseudoTrans = !pseudoRowMajor && pseudo.GetDim(0) == t.dim && pseudo.GetDim(1) == e;
    if (!pseudoRowMajor && !t.pseudoTrans) {
        return false;
    }
    const uint64_t workPerSlot = uint64_t(t.ci) * t.co;
    if (workPerSlot != 0 && uint64_t(e) * t.slots > (uint64_t(1) << 31) / workPerSlot) {
        return false;
    }
    t.e = e;
    return true;
}

static bool SetRequiredInputs(gert::TilingContext *context, SplineConvTilingData &t, ge::DataType &dtype)
{
    const auto *x = context->GetRequiredInputShape(0);
    const auto *edges = context->GetRequiredInputShape(1);
    const auto *pseudo = context->GetRequiredInputShape(2);
    const auto *weight = context->GetRequiredInputShape(3);
    const auto *xDesc = context->GetRequiredInputDesc(0);
    const auto *edgesDesc = context->GetRequiredInputDesc(1);
    const auto *pseudoDesc = context->GetRequiredInputDesc(2);
    const auto *weightDesc = context->GetRequiredInputDesc(3);
    if (!x || !edges || !pseudo || !weight || !xDesc || !edgesDesc || !pseudoDesc || !weightDesc) {
        return false;
    }
    dtype = xDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_BF16) {
        return false;
    }
    if (edgesDesc->GetDataType() != ge::DT_INT64 || pseudoDesc->GetDataType() != dtype ||
        weightDesc->GetDataType() != dtype) {
        return false;
    }
    return SetNodeLayout(x->GetOriginShape(), weight->GetOriginShape(), t) &&
           SetGraphLayout(edges->GetOriginShape(), pseudo->GetOriginShape(), t);
}

static bool SetOptionalInputs(gert::TilingContext *context, SplineConvTilingData &t, ge::DataType dtype)
{
    const auto *root = context->GetOptionalInputShape(4);
    if (root) {
        const auto *desc = context->GetOptionalInputDesc(4);
        const auto &shape = root->GetOriginShape();
        if (!desc || desc->GetDataType() != dtype || shape.GetDimNum() != 2 || shape.GetDim(0) != t.ci ||
            shape.GetDim(1) != t.co) {
            return false;
        }
        t.hasRoot = 1;
    }
    const auto *bias = context->GetOptionalInputShape(5);
    if (bias) {
        const auto *desc = context->GetOptionalInputDesc(5);
        const auto &shape = bias->GetOriginShape();
        if (!desc || desc->GetDataType() != dtype || shape.GetDimNum() != 1 || shape.GetDim(0) != t.co) {
            return false;
        }
        t.hasBias = 1;
    }
    return true;
}

static bool IsLinearTwoDimensional(const SplineConvTilingData &t)
{
    return t.dim == 2 && t.degree == 1 && t.ks[0] == 2 && t.ks[1] == 2;
}

static bool CanSelectSpecializedKernel(const SplineConvTilingData &t)
{
    return !t.edgeTrans && !t.pseudoTrans && !t.weightTrans && !t.hasRoot && !t.hasBias && t.co == 16 &&
           IsLinearTwoDimensional(t);
}

static void SelectBlockCount(SplineConvTilingData &t, uint32_t coreCount)
{
    t.blocks = std::min(coreCount, t.n);
    if (!CanSelectSpecializedKernel(t) || t.e == 0 || t.e >= t.n || t.e > 64) {
        return;
    }
    if (t.ci == 32) {
        const uint32_t proposed = std::min(t.blocks, std::max(8u, t.e));
        if (proposed == 0) {
            return;
        }
        const uint32_t rows = (t.n + proposed - 1) / proposed;
        if (rows <= t.ci && uint64_t(rows) * t.e <= 512) {
            t.blocks = proposed;
        }
    } else if (t.ci == 128) {
        const uint32_t rowBound = (t.n + 127) / 128;
        const uint32_t scanBound = (uint64_t(t.n) * t.e + 511) / 512;
        t.blocks = std::min(t.blocks, std::max({8u, t.e, rowBound, scanBound}));
    }
}

static uint32_t SelectSingleLayout(SplineConvTilingData &t, ge::DataType dtype, uint32_t coreCount)
{
    if (!CanSelectSpecializedKernel(t) || t.ci != 8 || t.e != 1 || t.n > 64 || t.blocks != t.n) {
        return 0;
    }
    if (t.n % (dtype == ge::DT_BF16 ? 4 : 2) != 0 && t.n < coreCount) {
        ++t.blocks;
        return 4;
    }
    return t.n % 16 == 0 ? 3 : 0;
}

static void SelectCachesAndDot(SplineConvTilingData &t, uint64_t cacheElements, uint64_t usedBytes, uint64_t available,
                               uint64_t elementBytes, uint64_t weightBytes)
{
    t.cacheWeights = cacheElements * 4 <= 65536 && usedBytes + cacheElements * weightBytes <= available;
    if (t.cacheWeights) {
        usedBytes += cacheElements * weightBytes;
    }
    const uint64_t graphBytes = uint64_t((2 * t.e + 3) / 4 * 4) * 8 +
                                uint64_t((t.e * t.dim + 15) / 16 * 16) * elementBytes + uint64_t((t.e + 7) / 8 * 8) * 4;
    t.cacheGraph = t.e > 0 && usedBytes + graphBytes <= available;
    usedBytes += t.cacheGraph ? graphBytes : 0;
    const uint64_t dotBytes = uint64_t(t.ciPad) * t.coPad * 8;
    t.vectorDot = t.ci <= 128 && t.coPad <= 128 && usedBytes + dotBytes <= available;
    const uint64_t fusedRows = uint64_t(t.kernels) * t.ciPad;
    const uint64_t fusedBytes = fusedRows * (t.coPad + 8) * 4;
    t.fusedDot =
        t.cacheWeights && t.ci == t.ciPad && t.coPad <= 128 && fusedRows <= 2040 && usedBytes + fusedBytes <= available;
}

static bool SetBufferPlan(SplineConvTilingData &t, ge::DataType dtype, uint64_t ubSize)
{
    constexpr uint64_t reservedBytes = 16384;
    if (ubSize < reservedBytes) {
        return false;
    }
    const uint64_t available = ubSize - reservedBytes;
    const uint32_t alignment = dtype == ge::DT_BF16 ? 16 : 8;
    t.ciPad = (t.ci + alignment - 1) / alignment * alignment;
    t.coPad = (t.co + alignment - 1) / alignment * alignment;
    const uint64_t elementBytes = dtype == ge::DT_BF16 ? 2 : 4;
    const uint64_t weightBytes = elementBytes + (dtype == ge::DT_BF16 ? 4 : 0);
    const uint64_t matrixElements = uint64_t(t.ci) * t.coPad;
    const uint64_t cacheElements = t.kernels * matrixElements;
    const uint64_t headBytes = uint64_t((t.nodesPerBlock + 7) / 8 * 8) * 4;
    uint64_t usedBytes = headBytes + uint64_t(t.kernels) * t.ciPad * 8 + uint64_t(t.ciPad) * 12 +
                         matrixElements * weightBytes + uint64_t(t.ciPad + t.coPad) * (elementBytes + 4);
    if (t.kernels > 64) {
        usedBytes += uint64_t((t.kernels + 7) / 8 * 8) * 4;
    }
    t.streaming = t.edgeTrans || t.pseudoTrans || t.weightTrans || t.ci == 0 || usedBytes > available;
    if (t.streaming) {
        return headBytes + 128 * (24 + 2 * elementBytes) <= available;
    }
    SelectCachesAndDot(t, cacheElements, usedBytes, available, elementBytes, weightBytes);
    const uint64_t smallBytes = uint64_t(t.n) * 8 * weightBytes + uint64_t(t.e) * (16 + 2 * elementBytes) + 4096;
    t.smallScalar = t.e > 0 && t.ci == 8 && t.co == 8 && IsLinearTwoDimensional(t) &&
                    uint64_t(t.nodesPerBlock) * t.e <= 512 && smallBytes <= available;
    return true;
}

static uint32_t SelectSingleMode(SplineConvTilingData &t, ge::DataType dtype, uint32_t layout, uint32_t &pad)
{
    if (t.streaming || !CanSelectSpecializedKernel(t) || t.ci != 8 || t.e != 1 || t.nodesPerBlock > 8) {
        return 0;
    }
    if (dtype == ge::DT_FLOAT && layout != 0) {
        t.blocks /= 2;
        if (layout == 4 && t.n > 1) {
            if (t.n >= 3 && t.n <= 31 && t.n % 2 != 0) {
                pad = (31 - t.n) / 2;
                return 6;
            }
            return 5;
        }
    }
    if (layout != 0) {
        return layout;
    }
    return t.nodesPerBlock == 1 ? 2 : 1;
}

static bool FitsSparseWork(const SplineConvTilingData &t)
{
    return (t.ci == 32 || t.ci == 128) && t.ci == t.ciPad && t.e > 0 && t.e < t.n && t.e <= 64 &&
           t.nodesPerBlock <= t.ci && uint64_t(t.nodesPerBlock) * t.e <= 512;
}

static uint32_t SelectSparseMode(SplineConvTilingData &t, ge::DataType dtype, uint32_t coreCount, uint32_t &pad)
{
    if (t.streaming || !CanSelectSpecializedKernel(t) || !t.cacheGraph || !t.cacheWeights || !t.fusedDot) {
        return 0;
    }
    if (!FitsSparseWork(t)) {
        return 0;
    }
    if (t.ci == 128) {
        // Mode 8 clears at most 32 rows per block while assigning one block to each edge.
        return dtype == ge::DT_FLOAT && t.e == t.blocks && t.nodesPerBlock <= 32 ? 8 : 2;
    }
    if (dtype != ge::DT_FLOAT || t.n > 64 || t.e > t.blocks) {
        return 1;
    }
    const uint32_t groups = (t.n + 15) / 16;
    const uint32_t needed = 4 + groups;
    if (t.e != 4 || coreCount < needed) {
        return 3;
    }
    t.blocks = needed;
    pad = groups * 16 - t.n;
    return groups == 4 && pad != 0 ? 5 : 6;
}

static ge::graphStatus Tile(gert::TilingContext *context, bool full)
{
    auto *t = context->GetTilingData<SplineConvTilingData>();
    if (!t) {
        return ge::GRAPH_FAILED;
    }
    *t = {};
    ge::DataType dtype = ge::DT_FLOAT;
    if (!SetSplineAttributes(context, *t) || !SetRequiredInputs(context, *t, dtype) ||
        (full && !SetOptionalInputs(context, *t, dtype))) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t coreCount = platform.GetCoreNumAiv();
    SelectBlockCount(*t, coreCount);
    const uint32_t layout = SelectSingleLayout(*t, dtype, coreCount);
    if (t->blocks == 0) {
        return ge::GRAPH_FAILED;
    }
    t->nodesPerBlock = (t->n + t->blocks - 1) / t->blocks;
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (!SetBufferPlan(*t, dtype, ubSize)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t nodePad = 0;
    const uint32_t single = SelectSingleMode(*t, dtype, layout, nodePad);
    const uint32_t sparse = SelectSparseMode(*t, dtype, coreCount, nodePad);
    const uint32_t closed0 = (single >= 2 || sparse != 0) && t->scale[0] == 2;
    const uint32_t closed1 = (single >= 2 || sparse != 0) && t->scale[1] == 2;
    const uint32_t inputType = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, inputType, single, sparse, closed0, closed1, nodePad);
    context->SetBlockDim(t->blocks);
    if ((single == 3 || single == 5 || single == 6) && t->n <= 32 && t->blocks > 1) {
        context->SetScheduleMode(1);
    }
    context->GetWorkspaceSizes(1)[0] =
        platform.GetLibApiWorkSpaceSize() + uint64_t(t->blocks) * ((t->e + 15) / 16 * 16) * 4;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    return Tile(context, false);
}

static ge::graphStatus FullTilingFunc(gert::TilingContext *context)
{
    return Tile(context, true);
}
} // namespace optiling
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c)
{
    auto *out = c->GetOutputShape(0);
    const auto *x = c->GetInputShape(0);
    const auto *w = c->GetInputShape(3);
    if (!out || !x || !w || x->GetDimNum() != 2 || w->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }
    bool trans = w->GetDim(1) != x->GetDim(1) && w->GetDim(2) == x->GetDim(1);
    out->SetDimNum(2);
    out->SetDim(0, x->GetDim(0));
    out->SetDim(1, w->GetDim(trans ? 1 : 2));
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *c)
{
    c->SetOutputDataType(0, c->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge
namespace ops {
class SplineConvDefinition : public OpDef {
protected:
    explicit SplineConvDefinition(const char *name, bool full) : OpDef(name)
    {
        Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        Input("edge_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        Input("edge_attr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        if (full) {
            Input("root_weight")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            Input("bias")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        Attr("dim").AttrType(OPTIONAL).Int(2);
        Attr("kernel_size").AttrType(OPTIONAL).ListInt({2, 2});
        Attr("is_open_spline").AttrType(OPTIONAL).ListInt({-1});
        Attr("degree").AttrType(OPTIONAL).Int(1);
        Attr("aggr").AttrType(OPTIONAL).String("mean");
        SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    }
};

class SplineConv : public SplineConvDefinition {
public:
    explicit SplineConv(const char *name) : SplineConvDefinition(name, false)
    {
        AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SplineConv);

class SplineConvFull : public SplineConvDefinition {
public:
    explicit SplineConvFull(const char *name) : SplineConvDefinition(name, true)
    {
        OpAICoreConfig config;
        config.ExtendCfgInfo("opFile.value", "spline_conv").ExtendCfgInfo("opInterface.value", "spline_conv_full");
        AICore().SetTiling(optiling::FullTilingFunc).AddConfig("ascend910b", config);
    }
};
OP_ADD(SplineConvFull);
} // namespace ops
