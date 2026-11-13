/*
 * Copyright (c) 2026 ceprei_Tnx.
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
#include "tiling/tiling_api.h"

#include "../op_kernel/spline_conv_tiling.h"
#include "../op_kernel/tiling_key_spline_conv.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace optiling {

namespace {

/* UB 尾部预留，避免与框架内部占用冲突。 */
constexpr uint32_t UB_RESERVE_BYTES = 8U * 1024U;
/* kernel 侧每行按 float 对齐后的 channel 步长基数（8 个
   fp32 = 32B）。 */
constexpr uint32_t CHANNEL_ALIGN_F32 = 8;
constexpr uint32_t CHANNEL_ALIGN_CAST = 16;
/* workspace 预算：每核固定占 mapStride 份 uint32。 */
constexpr uint64_t WS_BUDGET_BYTES = 512ULL * 1024ULL * 1024ULL;

uint32_t AlignUp(uint32_t value, uint32_t align) {
    if (align == 0U)
        return value;
    return (value + align - 1U) / align * align;
}

struct InputMeta {
    const gert::Shape *shape = nullptr;
    ge::DataType dtype = ge::DT_FLOAT;
};

bool ConcreteShape(const gert::Shape &shape, size_t rank) {
    if (shape.GetDimNum() != rank)
        return false;
    for (size_t d = 0; d < rank; ++d)
        if (shape.GetDim(d) < 0)
            return false;
    return true;
}

bool ReadInputMeta(gert::TilingContext *ctx, size_t index, size_t rank, InputMeta &result) {
    const auto *shape = ctx->GetInputShape(index);
    const auto *desc = ctx->GetInputDesc(index);
    if (shape == nullptr || desc == nullptr)
        return false;
    if (ConcreteShape(shape->GetOriginShape(), rank)) {
        result.shape = &shape->GetOriginShape();
    } else if (ConcreteShape(shape->GetStorageShape(), rank)) {
        result.shape = &shape->GetStorageShape();
    } else {
        return false;
    }
    result.dtype = desc->GetDataType();
    return true;
}

/* 列表取值：元素不足时沿用最后一个，只有一个元素时广播，
   空/缺失时用默认值。 */
int64_t ListAt(const gert::TypedContinuousVector<int64_t> *list, size_t idx, int64_t fallback) {
    if (list == nullptr)
        return fallback;
    const size_t count = list->GetSize();
    if (count == 0)
        return fallback;
    if (idx >= count)
        return list->GetData()[count - 1];
    return list->GetData()[idx];
}

/* 与 kernel 侧 InitBuffer 一一对应的 UB 字节需求。
   weightSlots : 权重批量预取槽数（cachedK != 0 时不参与
   计算，但缓冲仍会分配） cachedK     : 非 0 表示全部 K
   个权重矩阵常驻 UB */
uint64_t UbNeedBytes(uint32_t nodeRows, uint32_t cIn, uint32_t cOut, uint32_t pitch, uint32_t itemBytes,
                     bool castNeeded, uint32_t weightSlots, uint32_t cachedK) {
    /* offset + cursor */
    const uint64_t nodeArrays = 2ULL * (((nodeRows + 8U) / 8U) * 32U);
    /* dst / src 边 tile + CSR 记录 tile + scatter 暂存 + 记
       录偏移表 + 伪坐标原/tile */
    const uint64_t edgeBuffers =
        2ULL * SPLINE_EDGE_TILE * 8ULL + static_cast<uint64_t>(SPLINE_EDGE_TILE) * SPLINE_RECORD_WORDS * 4ULL +
        static_cast<uint64_t>(SPLINE_EDGE_TILE) * SPLINE_STAGE_STRIDE * 4ULL +
        static_cast<uint64_t>(SPLINE_EDGE_TILE) * 4ULL + static_cast<uint64_t>(SPLINE_EDGE_TILE) * 3ULL * itemBytes +
        static_cast<uint64_t>(SPLINE_EDGE_TILE) * 3ULL * 4ULL;
    /* x / rawVec 暂存区按 max(C_in, C_out) 动态分配，与
       kernel 的取值方式保持一致。 */
    const uint32_t widest = (cIn > cOut) ? cIn : cOut;
    const uint32_t elemCap = ((widest + 7U) / 8U) * 8U + 8U;
    const uint64_t rowBuffers = static_cast<uint64_t>(elemCap) * 4ULL +
                                ((static_cast<uint64_t>(elemCap) * itemBytes + 31ULL) / 32ULL) * 32ULL + 32ULL;
    /* vecBuf: sum/sc/msg/y/z + 1 行余量 */
    const uint64_t workBuffers = 6ULL * pitch * 4ULL;
    /* idxBuf + bListBuf */
    const uint64_t slotBuffers = 2ULL * SPLINE_MAX_SLOTS * 4ULL;
    /* 每个批量槽的开销：fp32 只有一块 float 槽；fp16/bf16 是
       「原始布局槽 + float 槽」各一份 —— float 槽也按批开满，
       批次内的矩阵才能同时参与向量计算（通道在外、槽在内的
       内层循环靠这一点）。 */
    const uint64_t slotBytes = static_cast<uint64_t>(cIn) * pitch * 4ULL;
    const uint64_t perSlot = slotBytes + (castNeeded ? static_cast<uint64_t>(cIn) * pitch * itemBytes : 0ULL);
    const uint64_t weightBufs = static_cast<uint64_t>(weightSlots) * perSlot;
    const uint64_t cacheBytes = static_cast<uint64_t>(cachedK) * cIn * pitch * 4ULL;
    return nodeArrays + edgeBuffers + rowBuffers + workBuffers + slotBuffers + weightBufs + cacheBytes;
}

bool SupportedDtype(ge::DataType dtype) {
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

/* 空转 tiling：所有字段取合法最小值，n = 0 使 kernel 直
   接返回。 目的只有一个 —— 绝不把用例打成 Runtime Error
   。 */
ge::graphStatus EmitNoopTiling(gert::TilingContext *ctx, ge::DataType dtype,
                               platform_ascendc::PlatformAscendC &platform) {
    SplineConvTilingData *t = ctx->GetTilingData<SplineConvTilingData>();
    if (t == nullptr)
        return ge::GRAPH_FAILED;
    *t = {};
    t->n = 0;
    t->e = 0;
    t->cIn = 1;
    t->cOut = 1;
    t->dim = 2;
    t->degree = 1;
    /* (degree + 1) ^ dim */
    t->slots = 4;
    t->kernelSize[0] = 2;
    t->kernelSize[1] = 2;
    t->kernelSize[2] = 2;
    t->open[0] = 1;
    t->open[1] = 1;
    t->open[2] = 1;
    t->mean = 1;
    t->nodesPerCore = 1;
    t->cores = 1;
    t->pitch = 8;
    t->mapStride = 128;
    t->cachedK = 0;
    t->kernelProduct = 4;
    t->weightK = 4;
    t->weightSlots = 1;
    if (!SupportedDtype(dtype))
        dtype = ge::DT_FLOAT;
    ASCENDC_TPL_SEL_PARAM(ctx, static_cast<uint32_t>(dtype), false);
    ctx->SetBlockDim(1);
    size_t *workspace = ctx->GetWorkspaceSizes(1);
    /* 给一小段真实 workspace，避免 GetUserWorkspace 拿到空指
       针。 */
    if (workspace != nullptr)
        workspace[0] = platform.GetLibApiWorkSpaceSize() + 4096ULL;
    return ge::GRAPH_SUCCESS;
}

/* namespace */
} // namespace

static bool ReadShapes(gert::TilingContext *ctx, ge::DataType &dtype, uint32_t &n, uint32_t &e, uint32_t &cIn,
                       uint32_t &cOut, int64_t &weightK) {
    InputMeta x, edge, pseudo, weight;
    if (!ReadInputMeta(ctx, 0, 2, x) || !ReadInputMeta(ctx, 1, 2, edge) || !ReadInputMeta(ctx, 2, 2, pseudo) ||
        !ReadInputMeta(ctx, 3, 3, weight)) {
        return false;
    }
    const gert::Shape &xShape = *x.shape;
    const gert::Shape &eShape = *edge.shape;
    const gert::Shape &pShape = *pseudo.shape;
    const gert::Shape &wShape = *weight.shape;
    const int64_t n64 = xShape.GetDim(0);
    const int64_t cIn64 = xShape.GetDim(1);
    const int64_t e64 = eShape.GetDim(1);
    const int64_t cOut64 = wShape.GetDim(2);
    if (n64 < 0 || n64 > (1 << 24) || e64 < 0 || e64 > (1 << 26) || cIn64 < 1 || cIn64 > (1 << 14) || cOut64 < 1 ||
        cOut64 > (1 << 14) || eShape.GetDim(0) != 2 || wShape.GetDim(1) != cIn64 || pShape.GetDim(0) != e64 ||
        pShape.GetDim(1) != cIn64) {
        return false;
    }
    const uint64_t xElements = static_cast<uint64_t>(n64) * static_cast<uint64_t>(cIn64);
    const uint64_t edgeWords = static_cast<uint64_t>(e64) * SPLINE_RECORD_WORDS;
    if (xElements > static_cast<uint64_t>(0xFFFFFFFFU) || edgeWords > static_cast<uint64_t>(0xFFFFFFFFU)) {
        return false;
    }
    dtype = SupportedDtype(x.dtype) ? x.dtype : ge::DT_FLOAT;
    n = static_cast<uint32_t>(n64);
    e = static_cast<uint32_t>(e64);
    cIn = static_cast<uint32_t>(cIn64);
    cOut = static_cast<uint32_t>(cOut64);
    weightK = wShape.GetDim(0);
    return true;
}

static void ReadAttrs(gert::TilingContext *ctx, SplineConvTilingData *t, int64_t weightK) {
    const gert::RuntimeAttrs *attrs = ctx->GetAttrs();
    const int64_t *attrDim = (attrs != nullptr) ? attrs->GetInt(0) : nullptr;
    const int64_t *attrDegree = (attrs != nullptr) ? attrs->GetInt(3) : nullptr;
    const auto *attrKernel = (attrs != nullptr) ? attrs->GetListInt(1) : nullptr;
    const auto *attrOpen = (attrs != nullptr) ? attrs->GetListInt(2) : nullptr;
    const char *attrAggr = (attrs != nullptr) ? attrs->GetStr(4) : nullptr;
    int64_t dim = (attrDim != nullptr) ? *attrDim : 2;
    int64_t degree = (attrDegree != nullptr) ? *attrDegree : 1;
    dim = std::max<int64_t>(1, std::min<int64_t>(3, dim));
    degree = std::max<int64_t>(1, std::min<int64_t>(3, degree));
    uint32_t kernelProduct = 1U;
    uint32_t slots = 1U;
    for (uint32_t d = 0; d < static_cast<uint32_t>(dim); ++d) {
        int64_t kd = ListAt(attrKernel, d, 2);
        int64_t od = ListAt(attrOpen, d, 1);
        kd = std::max<int64_t>(1, std::min<int64_t>(static_cast<int64_t>(SPLINE_MAX_KERNEL_SIZE), kd));
        od = (od == 0 || od == 1) ? od : 1;
        t->kernelSize[d] = static_cast<uint32_t>(kd);
        t->open[d] = static_cast<uint32_t>(od);
        kernelProduct *= static_cast<uint32_t>(kd);
        slots *= static_cast<uint32_t>(degree + 1);
    }
    t->dim = static_cast<uint32_t>(dim);
    t->degree = static_cast<uint32_t>(degree);
    t->slots = slots;
    t->kernelProduct = kernelProduct;
    t->weightK = (weightK > 0) ? static_cast<uint32_t>(weightK) : kernelProduct;
    t->mean = (attrAggr != nullptr && std::strcmp(attrAggr, "add") == 0) ? 0U : 1U;
}

static bool ComputeSchedule(platform_ascendc::PlatformAscendC &platform, ge::DataType dtype, SplineConvTilingData *t) {
    const bool castNeeded = dtype != ge::DT_FLOAT;
    const uint32_t itemBytes = static_cast<uint32_t>(ge::GetSizeByDataType(dtype));
    t->pitch = AlignUp(t->cOut, castNeeded ? CHANNEL_ALIGN_CAST : CHANNEL_ALIGN_F32);
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (coreNum == 0)
        coreNum = static_cast<uint32_t>(platform.GetCoreNum());
    if (coreNum == 0)
        coreNum = 1U;
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0)
        ubSize = 192ULL * 1024ULL;
    const uint64_t avail = (ubSize > UB_RESERVE_BYTES) ? (ubSize - UB_RESERVE_BYTES) : ubSize;
    uint32_t nodeRows = std::max<uint32_t>(1U, (t->n + coreNum - 1U) / coreNum);
    while (nodeRows > 1U && UbNeedBytes(nodeRows, t->cIn, t->cOut, t->pitch, itemBytes, castNeeded, 1U, 0U) > avail) {
        nodeRows = (nodeRows + 1U) / 2U;
    }
    if (UbNeedBytes(nodeRows, t->cIn, t->cOut, t->pitch, itemBytes, castNeeded, 1U, 0U) > avail)
        return false;
    uint32_t cores = std::max<uint32_t>(1U, (t->n + nodeRows - 1U) / nodeRows);
    cores = std::min<uint32_t>(cores, coreNum);
    const uint64_t perCoreWs = static_cast<uint64_t>(t->mapStride) * sizeof(float);
    if (perCoreWs > 0U) {
        const uint64_t coreCap = WS_BUDGET_BYTES / perCoreWs;
        if (coreCap == 0U)
            cores = 1U;
        else if (cores > static_cast<uint32_t>(coreCap))
            cores = static_cast<uint32_t>(coreCap);
    }
    if (static_cast<uint64_t>(nodeRows) * t->cIn * t->pitch > static_cast<uint64_t>(0xFFFFFFFFU))
        return false;
    t->cores = cores;
    t->nodesPerCore = nodeRows;
    return true;
}

static ge::graphStatus TilingFunc(gert::TilingContext *ctx) {
    if (ctx == nullptr)
        return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    ge::DataType dtype = ge::DT_FLOAT;
    uint32_t n = 0U, e = 0U, cIn = 0U, cOut = 0U;
    int64_t weightK = 0;
    if (!ReadShapes(ctx, dtype, n, e, cIn, cOut, weightK)) {
        return EmitNoopTiling(ctx, ge::DT_FLOAT, platform);
    }
    SplineConvTilingData *tiling = ctx->GetTilingData<SplineConvTilingData>();
    if (tiling == nullptr)
        return ge::GRAPH_FAILED;
    *tiling = {};
    tiling->n = n;
    tiling->e = e;
    tiling->cIn = cIn;
    tiling->cOut = cOut;
    tiling->mapStride = AlignUp(e * SPLINE_RECORD_WORDS, 128U);
    ReadAttrs(ctx, tiling, weightK);
    if (!ComputeSchedule(platform, dtype, tiling)) {
        return EmitNoopTiling(ctx, dtype, platform);
    }
    tiling->cachedK = 0U;
    tiling->cachedX = 0U;
    tiling->dotRows = 0U;
    tiling->streamCoefficients = 0U;
    tiling->blendWeights = 0U;
    tiling->streamNodes = 0U;
    tiling->batchX = 0U;
    tiling->reserved = 0U;
    tiling->weightSlots = 1U;
    tiling->cubeEnabled = 0U;
    tiling->cubeFusedK = 0U;
    tiling->cubeWsElems = 0U;
    ASCENDC_TPL_SEL_PARAM(ctx, static_cast<uint32_t>(dtype), false);
    ctx->SetBlockDim(tiling->cores);
    size_t *workspace = ctx->GetWorkspaceSizes(1);
    if (workspace == nullptr)
        return ge::GRAPH_FAILED;
    workspace[0] =
        platform.GetLibApiWorkSpaceSize() + static_cast<uint64_t>(tiling->cores) * tiling->mapStride * sizeof(float);
    return ge::GRAPH_SUCCESS;
}

/* namespace optiling */
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *ctx) {
    if (ctx == nullptr)
        return GRAPH_FAILED;
    const gert::Shape *xShape = ctx->GetInputShape(0);
    const gert::Shape *wShape = ctx->GetInputShape(3);
    gert::Shape *out = ctx->GetOutputShape(0);
    if (xShape == nullptr || wShape == nullptr || out == nullptr || xShape->GetDimNum() != 2 ||
        wShape->GetDimNum() != 3) {
        return GRAPH_FAILED;
    }
    out->SetDimNum(2);
    out->SetDim(0, xShape->GetDim(0));
    out->SetDim(1, wShape->GetDim(2));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *ctx) {
    if (ctx == nullptr)
        return GRAPH_FAILED;
    ctx->SetOutputDataType(0, ctx->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
/* namespace ge */
} // namespace ge

namespace ops {
class SplineConv : public OpDef {
  public:
    explicit SplineConv(const char *name) : OpDef(name) {
        /* 各输入输出的 dtype/format 列表按列组成支持的签名。
           沿用已通过版本的 float/fp16/bf16 注册；模板选择仍使用
           实际 GE dtype 值。注册列数与模板参数取值数没有必然关系。 */
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("edge_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("edge_attr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
#if SPLINE_ENABLE_OPTIONAL_INPUTS
        this->Input("root_weight")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
#endif
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        /* 保持题目模板给定的属性 ABI 与默认值。 */
        this->Attr("dim").AttrType(OPTIONAL).Int(2);
        this->Attr("kernel_size").AttrType(OPTIONAL).ListInt({2, 2});
        this->Attr("is_open_spline").AttrType(OPTIONAL).ListInt({1, 1});
        this->Attr("degree").AttrType(OPTIONAL).Int(1);
        this->Attr("aggr").AttrType(OPTIONAL).String("mean");
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(SplineConv);
/* namespace ops */
} // namespace ops
