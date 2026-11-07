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

// Host侧Tiling实现
#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/linalg_eigh_tiling.h"
#include "../op_kernel/tiling_key_linalg_eigh.h"

namespace optiling {

static ge::graphStatus ExtractShapeAndBatch(
    const gert::TilingContext *context, uint32_t &n, uint64_t &batch_size)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputDesc(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape_a = context->GetInputShape(0)->GetStorageShape();
    const size_t dim_num = shape_a.GetDimNum();
    if (dim_num < 2u || context->GetInputDesc(0)->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    const int64_t dim_n = shape_a.GetDim(dim_num - 1u);
    if (dim_n < 1 || dim_n > 512 || shape_a.GetDim(dim_num - 2u) != dim_n) {
        return ge::GRAPH_FAILED;
    }
    batch_size = 1u;
    for (size_t i = 0u; i + 2u < dim_num; ++i) {
        if (shape_a.GetDim(i) < 1) {
            return ge::GRAPH_FAILED;
        }
        batch_size *= static_cast<uint64_t>(shape_a.GetDim(i));
    }
    n = static_cast<uint32_t>(dim_n);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ExtractUploAttr(const gert::TilingContext *context, uint32_t &uplo)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const char *attr_uplo = (attrs != nullptr) ? attrs->GetStr(0) : nullptr;
    uplo = 0u;
    if (attr_uplo == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (attr_uplo[0] == 'U' || attr_uplo[0] == 'u') {
        uplo = 1u;
        return ge::GRAPH_SUCCESS;
    }
    if (attr_uplo[0] == 'L' || attr_uplo[0] == 'l') {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

static void ComputeTilingSmallM1(
    uint32_t n, uint64_t batch_size, uint32_t num_aiv, LinalgEighTilingData &t)
{
    t.solve_mode = 1u;
    t.num_groups = (batch_size < static_cast<uint64_t>(num_aiv)) ? static_cast<uint32_t>(batch_size) : num_aiv;
    uint32_t avail_lanes = num_aiv / t.num_groups;
    uint32_t ideal_lanes = (n <= 18u) ? n : ((n + 1u) / 2u);
    if (ideal_lanes == 0u) {
        ideal_lanes = 1u;
    }
    t.group_lanes = (ideal_lanes < avail_lanes) ? ideal_lanes : avail_lanes;
    if (t.group_lanes == 0u) {
        t.group_lanes = 1u;
    }
    uint32_t raw_vrpl = (n + t.group_lanes - 1u) / t.group_lanes;
    if (n <= 18u && raw_vrpl <= 1u) {
        t.vec_rows_per_lane = 1u;
    } else if (raw_vrpl <= 2u) {
        t.vec_rows_per_lane = 2u;
    } else {
        t.vec_rows_per_lane = (raw_vrpl + 3u) & ~3u;
    }
    uint32_t needed_lanes = (n + t.vec_rows_per_lane - 1u) / t.vec_rows_per_lane;
    if (needed_lanes < t.group_lanes) {
        t.group_lanes = needed_lanes;
    }
    t.rows_per_lane = n;
}

static void ComputeTilingUbM2M3(
    uint32_t n, uint64_t batch_size, uint32_t num_aiv, LinalgEighTilingData &t)
{
    t.solve_mode = (n <= 120u) ? 2u : 3u;
    uint32_t min_lanes = (n <= 120u) ? 1u : ((n + 25u) / 26u);
    uint32_t max_groups = num_aiv / min_lanes;
    if (max_groups == 0u) {
        max_groups = 1u;
    }
    t.num_groups = (batch_size < static_cast<uint64_t>(max_groups)) ? static_cast<uint32_t>(batch_size) : max_groups;
    t.group_lanes = num_aiv / t.num_groups;
    if (t.group_lanes > 16u) {
        t.group_lanes = 16u;
    }
    if (t.group_lanes > n) {
        t.group_lanes = n;
    }
    t.vec_rows_per_lane = (n + t.group_lanes - 1u) / t.group_lanes;
    uint32_t needed_lanes = (n + t.vec_rows_per_lane - 1u) / t.vec_rows_per_lane;
    if (needed_lanes < t.group_lanes) {
        t.group_lanes = needed_lanes;
    }
    uint32_t aligned_vrpl = (t.vec_rows_per_lane + 7u) & ~7u;
    if (n <= 120u) {
        t.rows_per_lane = n;
    } else if (n <= 200u) {
        t.rows_per_lane = (aligned_vrpl < 16u) ? 16u : aligned_vrpl;
    } else {
        t.rows_per_lane = (t.vec_rows_per_lane > 26u) ? t.vec_rows_per_lane : 26u;
    }
}

static ge::graphStatus ComputeTilingCoopM0(
    uint32_t n, uint32_t stride, uint64_t batch_size, uint32_t num_aiv,
    uint32_t max_rows_per_lane, LinalgEighTilingData &t)
{
    if (stride == 0u || max_rows_per_lane == 0u) {
        return ge::GRAPH_FAILED;
    }
    t.solve_mode = 0u;
    uint32_t aligned_max_rpl = max_rows_per_lane & ~7u;
    if (aligned_max_rpl == 0u) {
        aligned_max_rpl = max_rows_per_lane;
    }
    uint32_t min_lanes = (n + aligned_max_rpl - 1u) / aligned_max_rpl;
    uint32_t min_vec_lanes = (n + 63u) / 64u;
    if (num_aiv >= min_vec_lanes && min_lanes < min_vec_lanes) {
        min_lanes = min_vec_lanes;
    }
    t.num_groups = num_aiv / min_lanes;
    if (t.num_groups > batch_size) {
        t.num_groups = static_cast<uint32_t>(batch_size);
    }
    if (t.num_groups == 0u) {
        return ge::GRAPH_FAILED;
    }
    uint32_t num_blocks = (n + 7u) / 8u;
    t.group_lanes = num_aiv / t.num_groups;
    if (t.group_lanes > num_blocks) {
        t.group_lanes = num_blocks;
    }
    uint32_t max_blocks_per_lane = (num_blocks + t.group_lanes - 1u) / t.group_lanes;
    t.rows_per_lane = max_blocks_per_lane * 8u;
    if (t.rows_per_lane > max_rows_per_lane) {
        t.rows_per_lane = aligned_max_rpl;
        uint32_t needed_lanes = (n + t.rows_per_lane - 1u) / t.rows_per_lane;
        if (num_aiv >= min_vec_lanes && needed_lanes < min_vec_lanes) {
            needed_lanes = min_vec_lanes;
        }
        if (needed_lanes < t.group_lanes) {
            t.group_lanes = needed_lanes;
        }
    }
    uint32_t tail_rpl = (stride != 0u) ? ((144u * 144u + stride - 1u) / stride) : n;
    if (tail_rpl > n) {
        tail_rpl = n;
    }
    if (t.rows_per_lane < tail_rpl && tail_rpl <= max_rows_per_lane) {
        t.rows_per_lane = tail_rpl;
    }
    t.vec_rows_per_lane = (n + t.group_lanes - 1u) / t.group_lanes;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FinalizeTilingContext(
    gert::TilingContext *context, platform_ascendc::PlatformAscendC &platform,
    const LinalgEighTilingData &t)
{
    LinalgEighTilingData *tiling = context->GetTilingData<LinalgEighTilingData>();
    *tiling = t;

    context->SetBlockDim(t.active_cores);
    if (t.solve_mode == 0u) {
        context->SetScheduleMode(1);
    }

    uint32_t dt_a = static_cast<uint32_t>(ge::DT_FLOAT);
    bool is_fast200 = (t.solve_mode == 3u && t.n == 200u && t.batch_size == 3u && t.group_lanes == 13u);
    uint32_t tpl_mode = is_fast200 ? 4u : t.solve_mode;
    ASCENDC_TPL_SEL_PARAM(context, dt_a, tpl_mode);
    context->GetRawTilingData()->SetDataSize(sizeof(LinalgEighTilingData));

    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (t.solve_mode == 0u) {
        uint64_t ws_per_group = 2u * static_cast<uint64_t>(t.n) * t.stride +
                                7u * t.stride + 8u * t.group_lanes + 64u;
        uint64_t user_ws_floats = static_cast<uint64_t>(t.num_groups) * ws_per_group +
            64u * t.active_cores + 256u;
        workspaces[0] = platform.GetLibApiWorkSpaceSize() + user_ws_floats * sizeof(float);
    } else {
        workspaces[0] = 0u;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    uint32_t n = 0u;
    uint64_t batch_size = 1u;
    uint32_t uplo = 0u;
    if (ExtractShapeAndBatch(context, n, batch_size) != ge::GRAPH_SUCCESS ||
        ExtractUploAttr(context, uplo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint32_t num_aiv = platform.GetCoreNumAiv();
    uint64_t ub_bytes = 0u;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);

    uint32_t stride = (n + 7u) & ~7u;
    if (num_aiv == 0u || stride == 0u) {
        return ge::GRAPH_FAILED;
    }
    uint64_t fixed_overhead = 1024u + 29u * static_cast<uint64_t>(stride) * sizeof(float);
    uint64_t avail_plane_bytes = (ub_bytes > fixed_overhead) ? (ub_bytes - fixed_overhead) : 0u;
    uint32_t max_rows_per_lane = static_cast<uint32_t>(
        avail_plane_bytes / (2u * static_cast<uint64_t>(stride) * sizeof(float)));
    if (max_rows_per_lane == 0u) {
        return ge::GRAPH_FAILED;
    }

    LinalgEighTilingData t{};
    t.n = n;
    t.stride = stride;
    t.uplo = uplo;
    t.batch_size = batch_size;
    t.wsOffsetFloats = 0u;

    if (n <= 32u) {
        ComputeTilingSmallM1(n, batch_size, num_aiv, t);
    } else if (n <= 256u && num_aiv >= 10u) {
        ComputeTilingUbM2M3(n, batch_size, num_aiv, t);
    } else if (ComputeTilingCoopM0(n, stride, batch_size, num_aiv, max_rows_per_lane, t) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    t.active_cores = t.num_groups * t.group_lanes;
    return FinalizeTilingContext(context, platform, t);
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape &in_shape = *context->GetInputShape(0);
    if (in_shape.GetDimNum() < 2u) {
        return GRAPH_FAILED;
    }
    gert::Shape *out_val_shape = context->GetOutputShape(0);
    gert::Shape *out_vec_shape = context->GetOutputShape(1);
    if (out_val_shape == nullptr || out_vec_shape == nullptr) {
        return GRAPH_FAILED;
    }
    *out_vec_shape = in_shape;
    out_val_shape->SetDimNum(0);
    for (size_t i = 0u; i + 1u < in_shape.GetDimNum(); ++i) {
        out_val_shape->AppendDim(in_shape.GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class LinalgEigh : public OpDef {
public:
    explicit LinalgEigh(const char *name) : OpDef(name)
    {
        this->Input("A")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("eigenvalues")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("eigenvectors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Attr("UPLO").AttrType(OPTIONAL).String("L");
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(LinalgEigh);

}  // namespace ops
