// Copyright 2026 Electrical Engineering SIG - CANN Community
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/tiling_key_tree_line_min_distance.h"
#include "../op_kernel/tree_line_min_distance_tiling.h"

namespace optiling {
static bool MakeDivisionMetadata(uint32_t divisor, uint32_t& multiplier, uint32_t& shift)
{
    if (divisor == 0) {
        return false;
    }
    shift = 0;
    while ((uint64_t{1} << shift) < divisor) {
        ++shift;
    }
    multiplier = static_cast<uint32_t>(
        ((uint64_t{1} << 32) * ((uint64_t{1} << shift) - divisor)) / divisor + 1);
    return true;
}

static bool IsSupportedShape(int64_t batches, int64_t pointsPerBatch,
    int64_t segmentsPerBatch, uint64_t& totalPoints)
{
    if (batches <= 0 || pointsPerBatch <= 0 || segmentsPerBatch <= 0 ||
        segmentsPerBatch > kTreeLineMaxSegments) {
        return false;
    }
    const uint64_t batches64 = static_cast<uint64_t>(batches);
    const uint64_t points64 = static_cast<uint64_t>(pointsPerBatch);
    const uint64_t segments64 = static_cast<uint64_t>(segmentsPerBatch);
    if (batches64 > UINT32_MAX || points64 > UINT32_MAX ||
        segments64 > UINT32_MAX || points64 > kTreeLineMaxPoints / batches64) {
        return false;
    }
    totalPoints = batches64 * points64;
    return totalPoints <= kTreeLineMaxPoints;
}

static bool InitializeTilingData(const gert::Tensor* points,
    const gert::Tensor* segments, TreeLineMinDistanceTilingData* data,
    uint64_t& totalPoints)
{
    if (points == nullptr || segments == nullptr || data == nullptr) {
        return false;
    }
    const auto& pointShape = points->GetStorageShape();
    const auto& segmentShape = segments->GetStorageShape();
    if (pointShape.GetDimNum() != 3 || segmentShape.GetDimNum() != 4 ||
        pointShape.GetDim(2) != 3 ||
        segmentShape.GetDim(2) != 2 || segmentShape.GetDim(3) != 3 ||
        segmentShape.GetDim(0) != pointShape.GetDim(0) ||
        !IsSupportedShape(pointShape.GetDim(0), pointShape.GetDim(1),
            segmentShape.GetDim(1), totalPoints)) {
        return false;
    }
    data->batches = static_cast<uint32_t>(pointShape.GetDim(0));
    data->pointsPerBatch = static_cast<uint32_t>(pointShape.GetDim(1));
    data->segmentsPerBatch = static_cast<uint32_t>(segmentShape.GetDim(1));
    data->totalPoints = static_cast<uint32_t>(totalPoints);
    return true;
}

static uint32_t ComputeBlockDim(
    uint64_t totalPoints, uint32_t segmentsPerBatch, uint32_t deviceBlocks)
{
    constexpr uint32_t kPointVectorPointsPerBlock = 512;
    constexpr uint32_t kMatrixWorkPerBlock = 4096;
    constexpr uint32_t kMinBlocks = 8;

    // Every block repeats the per-core setup and occupies a full dispatch slot,
    // so a block is only added once the blocks before it hold a full tile: 512
    // point-vector points, or 4096 point-segment products on the matrix path.
    // Shapes that cannot fill such a tile still keep eight blocks.
    const bool pointVectorRegime =
        segmentsPerBatch <= kPointMajorSegmentLimit &&
        uint64_t(segmentsPerBatch) * kPointMajorLaneTarget <= totalPoints;
    const uint64_t totalWork = pointVectorRegime
        ? totalPoints
        : totalPoints * segmentsPerBatch;
    const uint64_t workPerBlock = pointVectorRegime
        ? kPointVectorPointsPerBlock
        : kMatrixWorkPerBlock;
    const uint64_t workPacedBlocks = (totalWork + workPerBlock - 1) / workPerBlock;
    const uint64_t requiredBlocks =
        workPacedBlocks > kMinBlocks ? workPacedBlocks : kMinBlocks;
    return requiredBlocks < deviceBlocks
        ? static_cast<uint32_t>(requiredBlocks)
        : deviceBlocks;
}

static bool SetDivisionMetadata(
    TreeLineMinDistanceTilingData* data, uint32_t blockDim)
{
    return MakeDivisionMetadata(blockDim, data->coreDivisionMultiplier,
            data->coreDivisionShift) &&
        MakeDivisionMetadata(data->pointsPerBatch,
            data->batchDivisionMultiplier, data->batchDivisionShift);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* points = context->GetRequiredInputTensor(0);
    const auto* segments = context->GetRequiredInputTensor(1);
    auto* data = context->GetTilingData<TreeLineMinDistanceTilingData>();
    uint64_t totalPoints = 0;
    if (!InitializeTilingData(points, segments, data, totalPoints)) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t deviceBlocks = platform.GetCoreNumAiv();
    const uint32_t blockDim = deviceBlocks == 0
        ? 0U : ComputeBlockDim(
            totalPoints, data->segmentsPerBatch, deviceBlocks);
    if (blockDim == 0 || !SetDivisionMetadata(data, blockDim)) {
        return ge::GRAPH_FAILED;
    }
    data->blockDim = blockDim;
    context->SetBlockDim(blockDim);
    const uint32_t dataType = static_cast<uint32_t>(points->GetDataType());
    const uint64_t maxBlockPoints = (totalPoints + blockDim - 1) / blockDim;
    const uint32_t tinyOnly =
        maxBlockPoints * data->segmentsPerBatch <= 8 ? 1U : 0U;
    ASCENDC_TPL_SEL_PARAM(context, dataType, tinyOnly);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* points = context->GetInputShape(0);
    auto* distances = context->GetOutputShape(0);
    distances->SetDimNum(2);
    distances->SetDim(0, points->GetDim(0));
    distances->SetDim(1, points->GetDim(1));
    *context->GetOutputShape(1) = *distances;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, DT_INT32);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class TreeLineMinDistance : public OpDef {
public:
    explicit TreeLineMinDistance(const char* name) : OpDef(name)
    {
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
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(TreeLineMinDistance);
}
