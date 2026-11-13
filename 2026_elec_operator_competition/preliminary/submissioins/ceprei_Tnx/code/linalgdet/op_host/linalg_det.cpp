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
#include "../op_kernel/linalg_det_tiling.h"
#include "../op_kernel/tiling_key_linalg_det.h"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {
static bool ReadShape(gert::TilingContext *context, uint32_t &n, uint32_t &batch, uint32_t &stride) {
    auto a = context->GetRequiredInputTensor(0);
    if (a == nullptr || a->GetDataType() != ge::DT_FLOAT)
        return false;
    auto shape = a->GetOriginShape();
    size_t rank = shape.GetDimNum();
    if (rank < 2U)
        return false;
    int64_t n64 = shape.GetDim(rank - 1U);
    if (n64 < 1 || n64 > 512 || shape.GetDim(rank - 2U) != n64)
        return false;
    uint64_t batch64 = 1;
    for (size_t i = 0; i + 2U < rank; ++i) {
        int64_t dim = shape.GetDim(i);
        if (dim < 1 || dim > 256 || batch64 > UINT32_MAX / static_cast<uint64_t>(dim))
            return false;
        batch64 *= static_cast<uint64_t>(dim);
    }
    n = static_cast<uint32_t>(n64);
    batch = static_cast<uint32_t>(batch64);
    stride = (n + 7U) / 8U * 8U;
    return true;
}

static void ComputeVecLayout(uint64_t ub, uint32_t n, uint32_t stride, uint32_t &vecSs, uint64_t &vecFixed,
                             uint32_t &vecRows) {
    const uint32_t an = (n + 7U) / 8U * 8U;
    const uint64_t fixedNoMat = (static_cast<uint64_t>(4U * stride + 16U * n + 4U * an + 256U + 256U + 1024U)) * 4U;
    const uint64_t matrixBytes = static_cast<uint64_t>(n) * stride * 4U;
    constexpr uint64_t reserveWide = 16U * 1024U;
    vecSs = 0U;
    vecFixed = 0U;
    vecRows = 0U;
    if (ub > fixedNoMat + reserveWide + 8U * stride + matrixBytes) {
        uint64_t avail = ub - fixedNoMat - reserveWide - 8U * stride - matrixBytes;
        uint32_t ssCap = static_cast<uint32_t>(avail / (6U * 4U));
        ssCap = ssCap / 8U * 8U;
        uint64_t wanted = static_cast<uint64_t>(32U) * stride;
        uint64_t need = static_cast<uint64_t>(n) * stride / 6U;
        if (wanted < need)
            wanted = need;
        if (wanted < n)
            wanted = n;
        wanted = (wanted + 7U) / 8U * 8U;
        vecSs = (wanted < ssCap) ? static_cast<uint32_t>(wanted) : ssCap;
    }
    if (vecSs >= n && vecSs >= 4U * stride) {
        vecFixed = fixedNoMat + static_cast<uint64_t>(6U * vecSs) * 4U;
        vecRows = (ub > vecFixed + reserveWide + 8U * stride)
                      ? std::min(n, static_cast<uint32_t>((ub - vecFixed - reserveWide - 8U * stride) / (4U * stride)))
                      : 0U;
    }
}

static bool ComputeSchedule(platform_ascendc::PlatformAscendC &platform, uint32_t batch, uint32_t &used,
                            uint32_t &perCore) {
    uint32_t cores = platform.GetCoreNumAiv();
    if (cores == 0U)
        return false;
    used = std::min(cores, batch);
    if (used == 0U)
        return false;
    perCore = (batch - 1U) / used + 1U;
    if (perCore == 0U)
        return false;
    used = (batch - 1U) / perCore + 1U;
    return true;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr)
        return ge::GRAPH_FAILED;
    uint32_t n = 0U, batch = 0U, stride = 0U;
    if (!ReadShape(context, n, batch, stride))
        return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    if (ub == 0)
        return ge::GRAPH_FAILED;
    uint32_t vecSs = 0U, vecRows = 0U;
    uint64_t vecFixed = 0U;
    ComputeVecLayout(ub, n, stride, vecSs, vecFixed, vecRows);
    if (vecRows == 0U || ub <= vecFixed + 16U * 1024U + 8U * stride)
        return ge::GRAPH_FAILED;
    uint32_t used = 0U, perCore = 0U;
    if (!ComputeSchedule(platform, batch, used, perCore))
        return ge::GRAPH_FAILED;
    auto t = context->GetTilingData<LinalgDetTilingData>();
    if (t == nullptr)
        return ge::GRAPH_FAILED;
    t->n = n;
    t->batch = batch;
    t->stride = stride;
    t->tileRows = vecRows;
    t->resident = 0U;
    t->cubeEnabled = 0U;
    t->scratchStride = vecSs;
    t->matPerCore = perCore;
    t->coreWsElems = (static_cast<uint64_t>(4U) * n * stride + (DET_CUBE_B16 ? 64U : 32U) * n +
                      (DET_CUBE_B16 ? 96U : 64U) * stride + 127U) /
                     128U * 128U;
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(ge::DT_FLOAT), 0U);
    context->SetBlockDim(used);
    auto workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr)
        return ge::GRAPH_FAILED;
    workspace[0] = t->coreWsElems * used * sizeof(float);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    if (context == nullptr)
        return GRAPH_FAILED;
    auto a = context->GetInputShape(0);
    auto d = context->GetOutputShape(0);
    if (a == nullptr || d == nullptr || a->GetDimNum() < 2U)
        return GRAPH_FAILED;
    size_t rank = a->GetDimNum();
    d->SetDimNum(rank - 2U);
    for (size_t i = 0; i + 2U < rank; ++i)
        d->SetDim(i, a->GetDim(i));
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context == nullptr)
        return GRAPH_FAILED;
    context->SetOutputDataType(0, DT_FLOAT);
    return GRAPH_SUCCESS;
}
} // namespace ge
namespace ops {
class LinalgDet : public OpDef {
  public:
    explicit LinalgDet(const char *name) : OpDef(name) {
        this->Input("A")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("det")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(LinalgDet);
} // namespace ops
