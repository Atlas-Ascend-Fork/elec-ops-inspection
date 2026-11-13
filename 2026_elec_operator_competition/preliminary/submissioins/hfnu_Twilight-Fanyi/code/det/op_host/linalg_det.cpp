/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Author: Twilight-Fanyi.
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
#include <cstddef>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_det_tiling.h"
#include "../op_kernel/tiling_key_linalg_det.h"

namespace {
constexpr size_t A_INDEX = 0U;
constexpr size_t DET_INDEX = 0U;
constexpr int64_t MAX_N = 512;
constexpr int64_t MAX_BATCH = 256;

bool ProductDims(const gert::Shape &shape, size_t end, int64_t &product)
{
    product = 1;
    for (size_t index = 0; index < end; ++index) {
        const int64_t dim = shape.GetDim(index);
        if (dim < 1 || product > MAX_BATCH / dim)
            return false;
        product *= dim;
    }
    return true;
}
}  // namespace

namespace optiling {
namespace {
struct LaunchConfig {
    uint32_t n;
    uint32_t batches;
    uint32_t availableCores;
    uint32_t matrixCores;
    uint32_t groupedCores;
    bool residentMatrix;
    bool largeSingleMatrix;
    bool groupedBatches;
};

bool ReadInput(gert::TilingContext *context, int64_t &n, int64_t &batches)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr || context->GetInputShape(A_INDEX) == nullptr ||
        context->GetInputDesc(A_INDEX) == nullptr || context->GetWorkspaceSizes(1) == nullptr) {
        return false;
    }
    const gert::Shape &shape = context->GetInputShape(A_INDEX)->GetOriginShape();
    const size_t rank = shape.GetDimNum();
    if (rank < 2U) {
        return false;
    }
    n = shape.GetDim(rank - 1U);
    if (n < 1 || n > MAX_N || shape.GetDim(rank - 2U) != n) {
        return false;
    }
    return ProductDims(shape, rank - 2U, batches) && context->GetInputDesc(A_INDEX)->GetDataType() == ge::DT_FLOAT;
}

LaunchConfig MakeLaunchConfig(uint32_t n, uint32_t batches, uint32_t availableCores)
{
    LaunchConfig config{};
    config.n = n;
    config.batches = batches;
    config.availableCores = availableCores;
    config.residentMatrix = n >= 48U && n <= 128U;
    config.largeSingleMatrix = batches == 1U && n >= 128U && !config.residentMatrix;
    config.groupedBatches =
        batches >= 2U && batches <= availableCores / 2U && n >= 128U && (n & 7U) == 0U && !config.residentMatrix;
    const uint32_t requestedMatrixCores = n <= 160U ? 2U : (n + 39U) / 40U;
    config.matrixCores = requestedMatrixCores < availableCores ? requestedMatrixCores : availableCores;
    // ReadInput bounds the batch product to [1, MAX_BATCH].
    if (batches == 0U) {
        return config;
    }
    const uint32_t groupCapacity = availableCores / batches;
    const uint32_t groupSize = requestedMatrixCores < groupCapacity ? requestedMatrixCores : groupCapacity;
    config.groupedCores = batches * groupSize;
    return config;
}

uint32_t SelectFallbackBand(uint32_t n)
{
    if (n >= 49U && n <= 55U) {
        return 2U;
    }
    if (n >= 57U && n <= 63U) {
        return 3U;
    }
    if (n >= 65U && n <= 71U) {
        return 4U;
    }
    if (n >= 73U && n <= 79U) {
        return 5U;
    }
    if (n >= 81U && n <= 87U) {
        return 6U;
    }
    if (n >= 89U && (n & 7U) != 0U) {
        return 7U;
    }
    if (n >= 128U && (n & 7U) == 0U) {
        return 8U;
    }
    return 0U;
}

uint32_t SelectSmallBand(uint32_t n)
{
    if (n >= 4U && n <= 32U) {
        return n + 14U;
    }
    if ((n >= 33U && n <= 39U) || (n >= 45U && n <= 47U)) {
        return n + 14U;
    }
    if (n >= 33U && n <= 47U) {
        return 1U;
    }
    return SelectFallbackBand(n);
}

uint32_t SelectLocalLuBand(const LaunchConfig &config)
{
    const uint32_t n = config.n;
    const bool alignedLargeSingleMatrix = config.largeSingleMatrix && (n & 7U) == 0U;
    if (config.residentMatrix) {
        return 63U;
    }
    if (config.groupedBatches) {
        return 62U;
    }
    if (alignedLargeSingleMatrix && n >= 448U) {
        return 12U;
    }
    if (alignedLargeSingleMatrix) {
        return 13U;
    }
    if (config.largeSingleMatrix) {
        return 14U;
    }
    return SelectSmallBand(n);
}

bool SetWorkspace(gert::TilingContext *context, const LaunchConfig &config, uint64_t workspaceStride,
                  uint64_t libWorkspace)
{
    if (config.batches == 0U || workspaceStride > std::numeric_limits<size_t>::max() / sizeof(float) / config.batches) {
        return false;
    }
    // Only large single-matrix protocols consume the per-AIV metadata slots.
    const uint64_t metadataElements = config.largeSingleMatrix ? 16ULL * config.matrixCores : 0ULL;
    const uint64_t workspaceElements = workspaceStride * config.batches;
    const uint64_t maxElements = std::numeric_limits<size_t>::max() / sizeof(float);
    if (workspaceElements > maxElements - metadataElements) {
        return false;
    }
    const size_t bytes = static_cast<size_t>((workspaceElements + metadataElements) * sizeof(float));
    if (libWorkspace > std::numeric_limits<size_t>::max() - bytes) {
        return false;
    }
    context->GetWorkspaceSizes(1)[0] = static_cast<size_t>(libWorkspace) + bytes;
    return true;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    int64_t n = 0;
    int64_t batchCount = 1;
    if (!ReadInput(context, n, batchCount)) {
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<::LinalgDetTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t workspaceStride = (static_cast<uint64_t>(n) * n + 15ULL) & ~15ULL;
    if (workspaceStride > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    tiling->batchCount = static_cast<uint32_t>(batchCount);
    tiling->matrixSize = static_cast<uint32_t>(n);
    tiling->workspaceStride = static_cast<uint32_t>(workspaceStride);
    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores == 0U) {
        return ge::GRAPH_FAILED;
    }
    const auto config = MakeLaunchConfig(static_cast<uint32_t>(n), static_cast<uint32_t>(batchCount), availableCores);
    uint32_t DT_A = static_cast<uint32_t>(ge::DT_FLOAT);
    uint32_t LOCAL_LU_BAND = SelectLocalLuBand(config);
    ASCENDC_TPL_SEL_PARAM(context, DT_A, LOCAL_LU_BAND);
    context->SetBlockDim(config.groupedBatches
                             ? config.groupedCores
                             : (config.largeSingleMatrix
                                    ? config.matrixCores
                                    : (config.batches < availableCores ? config.batches : availableCores)));
    if ((config.largeSingleMatrix || config.groupedBatches) && context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return SetWorkspace(context, config, workspaceStride, platform.GetLibApiWorkSpaceSize()) ? ge::GRAPH_SUCCESS
                                                                                             : ge::GRAPH_FAILED;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(A_INDEX) == nullptr ||
        context->GetOutputShape(DET_INDEX) == nullptr)
        return GRAPH_FAILED;
    const gert::Shape &input = *context->GetInputShape(A_INDEX);
    const size_t rank = input.GetDimNum();
    if (rank < 2U)
        return GRAPH_FAILED;
    gert::Shape *output = context->GetOutputShape(DET_INDEX);
    output->SetDimNum(rank - 2U);
    for (size_t index = 0; index + 2U < rank; ++index) output->SetDim(index, input.GetDim(index));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr)
        return GRAPH_FAILED;
    context->SetOutputDataType(DET_INDEX, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class LinalgDet : public OpDef {
public:
    explicit LinalgDet(const char *name) : OpDef(name)
    {
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("det").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(LinalgDet);
}  // namespace ops
