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

#include "../op_kernel/tree_line_min_distance_tiling.h"
#include "../op_kernel/tiling_key_tree_line_min_distance.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {
constexpr uint32_t GROUP_REDUCE_MIN_SEGMENTS = 4096U;
constexpr uint32_t GROUP_REDUCE_SEGMENTS_PER_CORE = 2048U;

bool IsTreePointShape(const gert::Shape &shape) {
    const size_t dims = shape.GetDimNum();
    return (dims == 2U || dims == 3U) && shape.GetDim(dims - 1U) == 3;
}

bool IsLineSegmentShape(const gert::Shape &shape) {
    const size_t dims = shape.GetDimNum();
    if ((dims == 2U || dims == 3U) && shape.GetDim(dims - 1U) == 6) {
        return true;
    }
    return (dims == 3U || dims == 4U) && shape.GetDim(dims - 2U) == 2 && shape.GetDim(dims - 1U) == 3;
}

bool GetBatchGeometry(const gert::Shape &pointShape, const gert::Shape &segmentShape, uint32_t &batchCount,
                      uint32_t &pointsPerBatch, uint32_t &segmentsPerBatch) {
    if (!IsTreePointShape(pointShape) || !IsLineSegmentShape(segmentShape)) {
        return false;
    }
    const size_t pointDims = pointShape.GetDimNum();
    const size_t segmentDims = segmentShape.GetDimNum();
    const bool pointsBatched = pointDims == 3U;
    const bool segmentsBatched = (segmentShape.GetDim(segmentDims - 1U) == 6) ? segmentDims == 3U : segmentDims == 4U;
    if (pointsBatched != segmentsBatched) {
        return false;
    }

    const int64_t batch64 = pointsBatched ? pointShape.GetDim(0) : 1;
    const int64_t points64 = pointShape.GetDim(pointDims - 2U);
    const int64_t segments64 =
        segmentShape.GetDim(segmentDims - (segmentShape.GetDim(segmentDims - 1U) == 6 ? 2U : 3U));
    if (batch64 < 1 || points64 < 1 || segments64 < 1 || batch64 > std::numeric_limits<uint32_t>::max() ||
        points64 > std::numeric_limits<uint32_t>::max() || segments64 > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    if (segmentsBatched && segmentShape.GetDim(0) != batch64) {
        return false;
    }
    batchCount = static_cast<uint32_t>(batch64);
    pointsPerBatch = static_cast<uint32_t>(points64);
    segmentsPerBatch = static_cast<uint32_t>(segments64);
    return true;
}

ge::graphStatus SetPointOutputShape(const gert::Shape &input, gert::Shape *output) {
    if (output == nullptr || !IsTreePointShape(input)) {
        return ge::GRAPH_FAILED;
    }
    *output = gert::Shape();
    const size_t dims = input.GetDimNum();
    for (size_t i = 0; i + 1U < dims; ++i) {
        output->AppendDim(input.GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

struct Schedule {
    uint32_t usedCores;
    uint32_t pointsPerCore;
    uint32_t extraCoreCount;
    uint32_t coresPerPoint;
};

uint32_t GetAvailableCores(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t availableCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (availableCores == 0U) {
        availableCores = static_cast<uint32_t>(platform.GetCoreNum());
    }
    return std::max<uint32_t>(availableCores, 1U);
}

Schedule ComputeSchedule(uint32_t pointCount, uint32_t segmentsPerBatch, uint32_t availableCores) {
    Schedule schedule{};
    if (pointCount == 0U) {
        schedule.usedCores = 1U;
        schedule.coresPerPoint = 1U;
        return schedule;
    }
    uint32_t coresPerPoint = 1U;
    if (pointCount < availableCores && segmentsPerBatch >= GROUP_REDUCE_MIN_SEGMENTS) {
        const uint32_t coreCapacity = availableCores / pointCount;
        const uint32_t usefulCores = 1U + (segmentsPerBatch - 1U) / GROUP_REDUCE_SEGMENTS_PER_CORE;
        coresPerPoint = std::min<uint32_t>(coreCapacity, usefulCores);
        coresPerPoint = std::max<uint32_t>(coresPerPoint, 1U);
    }
    const uint32_t usedCores =
        coresPerPoint > 1U ? pointCount * coresPerPoint : std::min<uint32_t>(availableCores, pointCount);
    schedule.usedCores = usedCores;
    schedule.pointsPerCore = pointCount / usedCores;
    schedule.extraCoreCount = pointCount % usedCores;
    schedule.coresPerPoint = coresPerPoint;
    return schedule;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *treePoints = context->GetRequiredInputTensor(0);
    const gert::Tensor *lineSegments = context->GetRequiredInputTensor(1);
    if (treePoints == nullptr || lineSegments == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &pointShape = treePoints->GetOriginShape();
    const gert::Shape &segmentShape = lineSegments->GetOriginShape();
    uint32_t batchCount = 0;
    uint32_t pointsPerBatch = 0;
    uint32_t segmentsPerBatch = 0;
    if (!GetBatchGeometry(pointShape, segmentShape, batchCount, pointsPerBatch, segmentsPerBatch)) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = treePoints->GetDataType();
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) || lineSegments->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t pointCount64 = static_cast<uint64_t>(batchCount) * static_cast<uint64_t>(pointsPerBatch);
    const uint64_t segmentCount64 = static_cast<uint64_t>(batchCount) * static_cast<uint64_t>(segmentsPerBatch);
    if (static_cast<uint64_t>(treePoints->GetShapeSize()) != pointCount64 * 3U ||
        static_cast<uint64_t>(lineSegments->GetShapeSize()) != segmentCount64 * 6U) {
        return ge::GRAPH_FAILED;
    }
    if (pointCount64 > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t pointCount = static_cast<uint32_t>(pointCount64);

    const uint32_t availableCores = GetAvailableCores(context);
    const Schedule schedule = ComputeSchedule(pointCount, segmentsPerBatch, availableCores);

    TreeLineMinDistanceTilingData *tiling = context->GetTilingData<TreeLineMinDistanceTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->pointCount = pointCount;
    tiling->batchCount = batchCount;
    tiling->pointsPerBatch = pointsPerBatch;
    tiling->segmentsPerBatch = segmentsPerBatch;
    tiling->pointsPerCore = schedule.pointsPerCore;
    tiling->extraCoreCount = schedule.extraCoreCount;
    tiling->segmentTileSize = TREE_LINE_SEGMENTS_PER_TILE;
    tiling->coresPerPoint = schedule.coresPerPoint;

    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype));
    context->SetBlockDim(schedule.usedCores);
    size_t *workspace = context->GetWorkspaceSizes(1);
    // Group-reduce partials use one isolated 32-byte slot per participating
    // core. The normal point-parallel path does not access workspace.
    workspace[0] = schedule.coresPerPoint > 1U ? static_cast<size_t>(schedule.usedCores) * 32U : 0U;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *pointShape = context->GetInputShape(0);
    const gert::Shape *segmentShape = context->GetInputShape(1);
    gert::Shape *distanceShape = context->GetOutputShape(0);
    gert::Shape *indexShape = context->GetOutputShape(1);
    uint32_t batchCount = 0;
    uint32_t pointsPerBatch = 0;
    uint32_t segmentsPerBatch = 0;
    if (pointShape == nullptr || segmentShape == nullptr ||
        !GetBatchGeometry(*pointShape, *segmentShape, batchCount, pointsPerBatch, segmentsPerBatch) ||
        SetPointOutputShape(*pointShape, distanceShape) != GRAPH_SUCCESS ||
        SetPointOutputShape(*pointShape, indexShape) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class TreeLineMinDistance : public OpDef {
  public:
    explicit TreeLineMinDistance(const char *name) : OpDef(name) {
        this->Input("tree_points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("line_segments")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("min_distance")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("nearest_segment_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(TreeLineMinDistance);
} // namespace ops
