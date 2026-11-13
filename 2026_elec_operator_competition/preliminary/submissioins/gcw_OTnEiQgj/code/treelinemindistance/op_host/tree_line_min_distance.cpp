/*
 * Copyright (c) 2026 gcw_OTnEiQgj.
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

namespace {
bool ValidateTreeShapes(const gert::Shape& pts, const gert::Shape& segs, elec::Config& c) {
    if (pts.GetDimNum() != 3 || segs.GetDimNum() != 4) return false;
    const int64_t b = pts.GetDim(0);
    const int64_t m = pts.GetDim(1);
    const int64_t n = segs.GetDim(1);
    const bool dims_ok = (pts.GetDim(2) == 3) && (segs.GetDim(0) == b) && (segs.GetDim(2) == 2) && (segs.GetDim(3) == 3);
    const bool range_ok = (b >= 1 && b <= 8) && (m >= 1 && m <= 16384) && (n >= 1 && n <= 1024);
    if (!dims_ok || !range_ok) return false;
    c.batch = static_cast<uint64_t>(b);
    c.m = static_cast<uint64_t>(m);
    c.n = static_cast<uint64_t>(n);
    return true;
}

bool ValidateTreeOutputs(gert::TilingContext* context, ge::DataType dt, const elec::Config& c) {
    const auto* o0 = context->GetOutputShape(0);
    const auto* o1 = context->GetOutputShape(1);
    const auto* od0 = context->GetOutputDesc(0);
    const auto* od1 = context->GetOutputDesc(1);
    if (!o0 || !o1 || !od0 || !od1) return false;
    if (od0->GetDataType() != dt || od1->GetDataType() != ge::DT_INT32) return false;
    const auto& os0 = o0->GetStorageShape();
    const auto& os1 = o1->GetStorageShape();
    const int64_t b = static_cast<int64_t>(c.batch);
    const int64_t m = static_cast<int64_t>(c.m);
    const bool s0_ok = (os0.GetDimNum() == 2) && (os0.GetDim(0) == b) && (os0.GetDim(1) == m);
    const bool s1_ok = (os1.GetDimNum() == 2) && (os1.GetDim(0) == b) && (os1.GetDim(1) == m);
    return s0_ok && s1_ok;
}

bool ValidateAndInitTreeTiling(gert::TilingContext* context, TreeLineMinDistanceTilingData* data, uint32_t& raw_type) {
    const auto* d0 = context->GetRequiredInputDesc(0);
    const auto* d1 = context->GetRequiredInputDesc(1);
    const auto* s0 = context->GetRequiredInputShape(0);
    const auto* s1 = context->GetRequiredInputShape(1);
    if (!d0 || !d1 || !s0 || !s1 || !data) return false;
    const ge::DataType dt = d0->GetDataType();
    if ((dt != ge::DT_FLOAT && dt != ge::DT_FLOAT16) || d1->GetDataType() != dt) return false;
    *data = TreeLineMinDistanceTilingData{};
    if (!ValidateTreeShapes(s0->GetStorageShape(), s1->GetStorageShape(), data->config)) return false;
    if (!ValidateTreeOutputs(context, dt, data->config)) return false;
    const uint32_t dtype = (dt == ge::DT_FLOAT) ? elec::F32 : elec::F16;
    data->config.dtype = dtype;
    data->input_types[0] = dtype;
    data->input_types[1] = dtype;
    data->output_types[0] = dtype;
    data->output_types[1] = elec::I32;
    raw_type = static_cast<uint32_t>(dt);
    return true;
}

uint32_t SelectLargeTreePath(const elec::Config& c, uint64_t cores) {
    const bool tail1 = (c.n > 64) && ((c.n & 63) == 1);
    const bool div_batch = (c.batch >= 1) && (cores >= c.batch) && ((cores % c.batch) == 0);
    if (c.n == 1024 && c.m >= 2048 && (c.m & 63) == 0 && div_batch) return 8;
    if (c.m >= 512 && c.batch > 1 && div_batch) return tail1 ? 13 : 12;
    return tail1 ? 11 : 3;
}

bool SelectTreePath(const elec::Config& c, uint64_t cores, uint64_t ub_bytes, uint32_t& path, uint64_t& tile) {
    const uint64_t points = c.batch * c.m;
    path = 2;
    tile = 32;
    if ((points <= 3 && c.n <= 6) || (c.n == 1 && points <= 32)) {
        path = (c.n <= 2) ? 7 : 9;
        return true;
    }
    if (c.n == 1) {
        path = 1;
        tile = 2048;
        return true;
    }
    if (c.m == 1 && c.n <= 1024) {
        path = 4;
        tile = 1;
        return true;
    }
    if (c.n <= 32 && points <= 64) {
        path = (c.n <= 8) ? 5 : 6;
        const uint64_t width = (c.n <= 8) ? 8 : 32;
        const uint64_t element_bytes = (c.dtype == elec::F32) ? 4 : 2;
        tile = elec::tree_short_point_tile(element_bytes, width);
        return elec::tree_short_ub_bytes(element_bytes, width, tile) <= ub_bytes;
    }
    if (c.m >= 256 && c.n >= 64) {
        path = SelectLargeTreePath(c, cores);
        tile = 64;
    } else if ((c.n > 64) && ((c.n & 63) == 1)) {
        path = 10;
    }
    return tile > 0;
}

bool EncodeTreeSchedule(elec::Config& c, uint32_t path, uint64_t tile, uint64_t cores, uint64_t& groups) {
    if (tile == 0 || c.batch == 0 || cores == 0) return false;
    const bool batch_pinned = (path == 8 || path == 12 || path == 13);
    const uint64_t per_batch = (c.m + tile - 1) / tile;
    if (per_batch == 0) return false;
    if (batch_pinned) {
        const uint64_t cores_per_batch = cores / c.batch;
        if (cores_per_batch == 0) return false;
        const uint64_t wave_cap = cores_per_batch * tile;
        if (wave_cap == 0) return false;
        const uint64_t full_waves = c.m / wave_cap;
        const uint64_t rem_points = c.m - full_waves * wave_cap;
        const uint64_t tail_tile = (rem_points + cores_per_batch - 1) / cores_per_batch;
        const uint32_t inv_d = static_cast<uint32_t>(((0x100000000ULL + cores_per_batch - 1) / cores_per_batch) - 1);
        c.k = (full_waves << 48) | (tail_tile << 40) | (cores_per_batch << 32) | inv_d;
        groups = cores;
    } else {
        const uint32_t inv_d = static_cast<uint32_t>(((0x100000000ULL + per_batch - 1) / per_batch) - 1);
        c.k = (per_batch << 32) | inv_d;
        groups = (path == 7 || path == 9) ? 8 : (c.batch * per_batch);
        if (groups < 8) groups = 8;
    }
    return true;
}

ge::graphStatus TilingFunc(gert::TilingContext* context) {
    auto* data = context->GetTilingData<TreeLineMinDistanceTilingData>();
    uint32_t raw_type = 0;
    if (!ValidateAndInitTreeTiling(context, data, raw_type)) return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint64_t cores = static_cast<uint64_t>(platform.GetCoreNumAiv());
    uint64_t ub_bytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);
    uint32_t path = 2;
    uint64_t tile = 32;
    if (!SelectTreePath(data->config, cores, ub_bytes, path, tile)) return ge::GRAPH_FAILED;
    uint64_t groups = 8;
    if (!EncodeTreeSchedule(data->config, path, tile, cores, groups)) return ge::GRAPH_FAILED;
    ASCENDC_TPL_SEL_PARAM(context, raw_type, path);
    context->SetBlockDim(static_cast<uint32_t>(groups < cores ? groups : cores));
    auto* ws = context->GetWorkspaceSizes(1);
    if (!ws) return ge::GRAPH_FAILED;
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const auto* s0 = context->GetInputShape(0);
    const auto* s1 = context->GetInputShape(1);
    auto* o0 = context->GetOutputShape(0);
    auto* o1 = context->GetOutputShape(1);
    elec::Config c;
    if (!s0 || !s1 || !o0 || !o1 || !ValidateTreeShapes(*s0, *s1, c)) return ge::GRAPH_FAILED;
    o0->SetDimNum(2);
    o0->SetDim(0, static_cast<int64_t>(c.batch));
    o0->SetDim(1, static_cast<int64_t>(c.m));
    o1->SetDimNum(2);
    o1->SetDim(0, static_cast<int64_t>(c.batch));
    o1->SetDim(1, static_cast<int64_t>(c.m));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto t = context->GetInputDataType(0);
    if (context->SetOutputDataType(0, t) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    return context->SetOutputDataType(1, ge::DT_INT32);
}
}  // namespace

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
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore()
            .SetTiling(TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(TreeLineMinDistance);
}  // namespace ops
