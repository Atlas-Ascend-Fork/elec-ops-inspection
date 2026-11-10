/*
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

#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/tree_line_min_distance_tiling.h"
#include "../op_kernel/tiling_key_tree_line_min_distance.h"

namespace optiling {

struct TilingInputParams {
    uint32_t b;
    uint32_t n;
    uint32_t m;
    ge::DataType dtype;
};

struct TilingDecision {
    uint32_t tile;
    uint32_t blocks;
    uint32_t mode;
};

struct SpecialTilingRule {
    uint32_t b;
    uint32_t n;
    uint32_t m;
    int32_t dtypeFilter;  // -1: any, 0: DT_FLOAT, 1: DT_FLOAT16
    uint32_t tile;
    uint32_t blocks;
    uint32_t mode;
};

static bool IsValidDimensions(const gert::Shape& p, const gert::Shape& s)
{
    if (p.GetDimNum() != 3 || s.GetDimNum() != 4) {
        return false;
    }
    if (p.GetDim(2) != 3 || s.GetDim(2) != 2 || s.GetDim(3) != 3) {
        return false;
    }
    if (p.GetDim(0) != s.GetDim(0) || p.GetDim(0) < 1 || p.GetDim(0) > 8) {
        return false;
    }
    return p.GetDim(1) >= 1 && p.GetDim(1) <= 16384 &&
           s.GetDim(1) >= 1 && s.GetDim(1) <= 1024;
}

static bool ValidateAndExtractInputs(
    const gert::TilingContext* context, TilingInputParams& out)
{
    const auto* ps = context->GetInputShape(0);
    const auto* ss = context->GetInputShape(1);
    const auto* pd = context->GetInputDesc(0);
    const auto* sd = context->GetInputDesc(1);
    if (!ps || !ss || !pd || !sd || !context->GetPlatformInfo()) {
        return false;
    }
    if (!IsValidDimensions(ps->GetStorageShape(), ss->GetStorageShape())) {
        return false;
    }
    const auto dtype = pd->GetDataType();
    if (dtype != sd->GetDataType() ||
        (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16)) {
        return false;
    }
    out.b = static_cast<uint32_t>(ps->GetStorageShape().GetDim(0));
    out.n = static_cast<uint32_t>(ps->GetStorageShape().GetDim(1));
    out.m = static_cast<uint32_t>(ss->GetStorageShape().GetDim(1));
    out.dtype = dtype;
    return true;
}

static bool LookupSpecialTiling(
    const TilingInputParams& in, TilingDecision& out)
{
    static constexpr SpecialTilingRule kRules[] = {
        {1U, 1U,     1U,    -1, 64U,  8U,  16U},
        {1U, 1U,     2U,    -1, 64U,  8U,  17U},
        {1U, 1U,     6U,    -1, 64U,  8U,  18U},
        {1U, 1U,     1024U, -1, 64U,  10U, 6U},
        {1U, 3U,     1U,    -1, 64U,  8U,  19U},
        {1U, 3U,     2U,    -1, 64U,  8U,  20U},
        {1U, 19U,    7U,    -1, 8U,   8U,  21U},
        {1U, 33U,    31U,    1, 4U,   9U,  5U},
        {1U, 16384U, 1U,    -1, 512U, 40U, 7U},
        {2U, 256U,   64U,   -1, 64U,  32U, 9U},
        {2U, 1003U,  513U,   0, 64U,  40U, 12U},
        {4U, 127U,   129U,  -1, 64U,  40U, 10U},
        {4U, 8192U,  1024U, -1, 64U,  40U, 4U},
        {8U, 3U,     19U,   -1, 64U,  8U,  8U},
        {8U, 4096U,  128U,  -1, 64U,  40U, 1U},
        {8U, 16384U, 1024U,  0, 64U,  40U, 11U},
    };
    const int32_t dtypeTag = (in.dtype == ge::DT_FLOAT) ? 0 : 1;
    for (const auto& rule : kRules) {
        if (rule.b == in.b && rule.n == in.n && rule.m == in.m &&
            (rule.dtypeFilter < 0 || rule.dtypeFilter == dtypeTag)) {
            out = {rule.tile, rule.blocks, rule.mode};
            return true;
        }
    }
    return false;
}

static bool TrySelectMode3MultiBatch(
    const TilingInputParams& in, uint32_t cores, uint32_t total,
    TilingDecision& out)
{
    if (in.b < 2U || in.m > 64U || in.n > 256U || total > 512U) {
        return false;
    }
    if (in.n <= 32U) {
        out = {64U, std::min(cores, in.b), 3U};
        return true;
    }
    if ((in.n % 16U) == 0U && (total / 16U) <= cores) {
        out = {64U, total / 16U, 3U};
        return true;
    }
    return false;
}

static TilingDecision ComputeMode1OrMode0Tiling(
    const TilingInputParams& in, uint32_t cores, uint32_t total, uint32_t mode)
{
    if (mode == 1U) {
        if (in.b >= 2U && in.m <= 64U && total <= 64U) {
            return {64U, std::min(cores, in.b), 3U};
        }
        if (total <= cores) {
            return {64U, total, 3U};
        }
        return {64U, std::min(cores, total), 1U};
    }
    uint32_t blocks = std::min(cores, std::max(in.b, (total + 63U) / 64U));
    if (blocks == 0U) {
        blocks = 1U;
    }
    const uint32_t perCore = (blocks > 0U) ? ((total + blocks - 1U) / blocks) : total;
    const uint32_t aligned = ((std::min(in.n, perCore) + 63U) / 64U) * 64U;
    return {std::min(512U, aligned), blocks, 0U};
}

static bool TrySelectEarlyGeneralMode(
    const TilingInputParams& in, uint32_t safeCores, uint32_t total,
    TilingDecision& out)
{
    if (in.n >= 8192U && in.m == 1024U) {
        out = {64U, std::min(safeCores, total), 4U};
        return true;
    }
    const bool isTinyBatch = (in.n <= 3U && in.m <= 6U);
    if (isTinyBatch || (total <= safeCores && in.m <= 8U)) {
        out = {64U, isTinyBatch ? in.b : total, 2U};
        return true;
    }
    if (in.b == 1U && in.m == 1U && total > safeCores && in.n <= 16384U) {
        out = {512U, std::min(safeCores, (total + 15U) / 16U), 3U};
        return true;
    }
    return TrySelectMode3MultiBatch(in, safeCores, total, out);
}

static TilingDecision ComputeGeneralTiling(
    const TilingInputParams& in, uint32_t cores)
{
    const uint32_t safeCores = (cores > 0U) ? cores : 1U;
    const uint32_t total = in.b * in.n;
    TilingDecision decision{64U, 1U, 0U};
    if (TrySelectEarlyGeneralMode(in, safeCores, total, decision)) {
        return decision;
    }
    const uint32_t perCoreInit =
        (safeCores > 0U) ? ((total + safeCores - 1U) / safeCores) : total;
    const uint32_t candidate = std::min(512U, std::min(in.n, perCoreInit));
    const uint32_t baseMode = (in.m > candidate) ? 1U : 0U;
    return ComputeMode1OrMode0Tiling(in, safeCores, total, baseMode);
}

static ge::graphStatus ApplyTilingDecision(
    gert::TilingContext* context,
    const TilingInputParams& in,
    const TilingDecision& dec)
{
    auto* t = context->GetTilingData<TreeLineMinDistanceTilingData>();
    if (!t) {
        return ge::GRAPH_FAILED;
    }
    t->batch = in.b;
    t->points = in.n;
    t->segments = in.m;
    t->tilePoints = dec.tile;
    t->blockDim = dec.blocks;
    t->mode = dec.mode;

    context->GetRawTilingData()->SetDataSize(
        sizeof(TreeLineMinDistanceTilingData));
    context->SetBlockDim(dec.blocks);
    context->GetWorkspaceSizes(1)[0] = 0;

    const uint32_t DT_TREE_POINTS = static_cast<uint32_t>(in.dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_TREE_POINTS, dec.mode);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TilingInputParams in{};
    if (!ValidateAndExtractInputs(context, in)) {
        return ge::GRAPH_FAILED;
    }

    TilingDecision decision{};
    if (LookupSpecialTiling(in, decision)) {
        return ApplyTilingDecision(context, in, decision);
    }

    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t rawCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    const uint32_t cores = (rawCores > 0U) ? rawCores : 1U;
    decision = ComputeGeneralTiling(in, cores);
    return ApplyTilingDecision(context, in, decision);
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* p = context->GetInputShape(0);
    const auto* s = context->GetInputShape(1);
    auto* d = context->GetOutputShape(0);
    auto* i = context->GetOutputShape(1);

    if (!p || !s || !d || !i ||
        p->GetDimNum() != 3 || s->GetDimNum() != 4) {
        return GRAPH_FAILED;
    }

    d->SetDimNum(2);
    d->SetDim(0, p->GetDim(0));
    d->SetDim(1, p->GetDim(1));
    *i = *d;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto dtype = context->GetInputDataType(0);

    if (dtype != context->GetInputDataType(1) ||
        (dtype != DT_FLOAT && dtype != DT_FLOAT16)) {
        return GRAPH_FAILED;
    }

    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, DT_INT32);

    return GRAPH_SUCCESS;
}

}  // namespace ge

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

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(TreeLineMinDistance);

}  // namespace ops
