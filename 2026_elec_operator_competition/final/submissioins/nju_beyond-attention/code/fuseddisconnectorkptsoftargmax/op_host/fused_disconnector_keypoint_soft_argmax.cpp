/*
 * Copyright (c) 2026 Beyond Attention. All rights reserved.
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
#include "../op_kernel/fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "../op_kernel/tiling_key_fused_disconnector_keypoint_soft_argmax.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include "tiling/matmul/matmul_tiling.h"

namespace optiling {
// Reject the Cube route when SDK tiling fields cannot be inspected.
template <int Rank> struct CubePriority : CubePriority<Rank - 1> {};
template <> struct CubePriority<0> {};
#define CUBE_GETTER(NAME, LOWER, UPPER, FIELD)                                                                         \
    template <class T> auto NAME(const T &t, CubePriority<3>)->decltype(t.LOWER()) { return t.LOWER(); }               \
    template <class T> auto NAME(const T &t, CubePriority<2>)->decltype(t.UPPER()) { return t.UPPER(); }               \
    template <class T> auto NAME(const T &t, CubePriority<1>)->decltype(t.FIELD) { return t.FIELD; }                   \
    template <class T> int NAME(const T &, CubePriority<0>) { return -1; }
CUBE_GETTER(CubeBaseM, get_baseM, GetBaseM, baseM)
CUBE_GETTER(CubeBaseN, get_baseN, GetBaseN, baseN)
CUBE_GETTER(CubeBaseK, get_baseK, GetBaseK, baseK)
CUBE_GETTER(CubeDbA, get_dbL0A, GetDbL0A, dbL0A)
CUBE_GETTER(CubeDbB, get_dbL0B, GetDbL0B, dbL0B)
CUBE_GETTER(CubeDbC, get_dbL0C, GetDbL0C, dbL0C)
CUBE_GETTER(CubeShareL1, get_shareL1Size, GetShareL1Size, shareL1Size)
#undef CUBE_GETTER

static int32_t TryProjectionTiling(platform_ascendc::PlatformAscendC &platform,
                                   FusedDisconnectorKeypointSoftArgmaxTilingData &out) {
    using namespace matmul_tiling;
    uint64_t l1 = 0, l0a = 0, l0b = 0, l0c = 0, ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    if (ub < std::max(PROJECTION_PACK_UB_BYTES, PROJECTION_DECODE_UB_BYTES) || l0a < 4096 || l0b < 49152 ||
        l0c < 12288 || l1 < 30720)
        return CUBE_CAPACITY;
    MatmulApiTiling api(platform);
    int32_t setup = 0;
    auto check = [&setup](int32_t status) {
        if (setup == 0 && status != 0)
            setup = status;
    };
    check(api.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT));
    check(api.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT));
    check(api.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT));
    check(api.SetShape(kProjectionRows, kCubeTileN, kProjectionChannels));
    check(api.SetOrgShape(kProjectionRows, out.height * out.width, kProjectionChannels));
    check(api.SetBias(false));
    check(api.SetFixSplit(kCubeBaseM, kCubeTileN, kCubeBaseK));
    check(api.SetBufferSpace(-1, -1, -1));
    if (setup)
        return CUBE_SETUP;
    AscendC::tiling::TCubeTiling data{};
    static_assert(sizeof(data) == 200, "actual TCube record must be200B");
    if (api.GetTiling(data) != 0)
        return CUBE_GET_TILING;
    SysTilingTempBufSize tmp{};
    static_assert(sizeof(tmp) == 12, "actual CANN9 tmp record");
    if (MatmulGetTmpBufSizeV2(data, tmp) != 0 || tmp.ubSize < 0 || tmp.l1Size < 0 || tmp.l0cSize < 0 ||
        uint64_t(tmp.ubSize) > ub || uint64_t(tmp.l1Size) > l1 || uint64_t(tmp.l0cSize) > l0c)
        return CUBE_TMP;
    const auto priority = CubePriority<3>{};
    if (CubeBaseM(data, priority) != 16 || CubeBaseN(data, priority) != 192 || CubeBaseK(data, priority) != 32 ||
        CubeDbA(data, priority) != 2 || CubeDbB(data, priority) != 2 || CubeDbC(data, priority) != 1 ||
        CubeShareL1(data, priority) != 30720)
        return CUBE_RECORD;
    std::memcpy(out.cube_words, &data, sizeof(data));
    return CUBE_OK;
}

// Direct Mmad has explicit buffers: no TCube record and no Matmul temporary allocation.
static int32_t TryA2Resources(platform_ascendc::PlatformAscendC &platform, uint32_t height) {
    uint64_t l1 = 0, l0a = 0, l0b = 0, l0c = 0, ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    return l1 >= A2_L1_BYTES && l0a >= A2_L0A_BYTES && l0b >= A2_L0B_BYTES && l0c >= A2_L0C_BYTES &&
                   ub >= std::max(A2_PACK_UB_BYTES, height == 64 ? A2_H64_DECODE_UB_BYTES : A2_H48_DECODE_UB_BYTES)
               ? CUBE_OK
               : CUBE_CAPACITY;
}

static ge::graphStatus TilingFunc(gert::TilingContext *c) {
    if (!c->GetInputShape(0) || !c->GetInputShape(1) || !c->GetInputShape(2) || !c->GetInputDesc(0) ||
        !c->GetInputDesc(1) || !c->GetInputDesc(2) || !c->GetAttrs() || !c->GetPlatformInfo())
        return ge::GRAPH_FAILED;
    const auto &x = c->GetInputShape(0)->GetStorageShape();
    const auto &w = c->GetInputShape(1)->GetStorageShape();
    const auto &b = c->GetInputShape(2)->GetStorageShape();
    if (x.GetDimNum() != 4 || w.GetDimNum() != 4 || b.GetDimNum() != 1 || x.GetDim(0) < 1 || x.GetDim(0) > 4 ||
        x.GetDim(1) != 32 || w.GetDim(0) != 4 || w.GetDim(1) != 32 || w.GetDim(2) != 3 || w.GetDim(3) != 3 ||
        b.GetDim(0) != 4 ||
        !((x.GetDim(2) == 48 && x.GetDim(3) == 64) || (x.GetDim(2) == 64 && x.GetDim(3) == 64) ||
          (x.GetDim(2) == 72 && x.GetDim(3) == 96)))
        return ge::GRAPH_FAILED;
    for (uint32_t i = 0; i < 3; ++i)
        if (c->GetInputDesc(i)->GetDataType() != ge::DT_FLOAT)
            return ge::GRAPH_FAILED;
    const float *tau = c->GetAttrs()->GetFloat(0);
    const int64_t *stride = c->GetAttrs()->GetInt(1);
    if (!tau || !stride || !std::isfinite(*tau) || *tau <= 0 || *stride != 4)
        return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    const uint32_t available = platform.GetCoreNumAiv();
    if (!available)
        return ge::GRAPH_FAILED;
    auto *t = c->GetTilingData<FusedDisconnectorKeypointSoftArgmaxTilingData>();
    auto *raw = c->GetRawTilingData();
    size_t *workspace = c->GetWorkspaceSizes(1);
    if (!t || !raw || !workspace)
        return ge::GRAPH_FAILED;
    t->batch = x.GetDim(0);
    t->height = x.GetDim(2);
    t->width = x.GetDim(3);
    t->rows = KEYPOINT_ROWS(t->batch, t->height, t->width);
    t->tau = *tau;
    t->stride = static_cast<uint32_t>(*stride);
    uint32_t groups = 0, route = 0;
    int32_t cube_status = CUBE_NOT_SELECTED;
    t->groups = 0;
    t->cube_status = CUBE_NOT_SELECTED;
    std::memset(t->cube_words, 0, sizeof(t->cube_words));
    uint64_t ownerUb = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ownerUb);
    if ((t->height == 64 && t->width == 64 && t->batch <= 2 && ownerUb >= 137344) ||
        (t->height == 48 && t->width == 64 && t->batch <= 3 && available >= t->batch * 4 && ownerUb >= 66176)) {
        route = 3;
    } else if ((*tau == 0.5f || *tau == 1.0f || *tau == 2.0f) &&
               ((t->height == 72 && t->width == 96) || (t->batch == 4 && t->height == 48 && t->width == 64))) {
        groups = std::min({20u, platform.GetCoreNumAic(), std::min(40u, available) / 2, t->batch * 36u});
        cube_status = groups ? TryProjectionTiling(platform, *t) : CUBE_NO_GROUP;
        if (cube_status == CUBE_OK) {
            route = 1;
            t->groups = groups;
        } else
            groups = 0;
    } else if (t->width == 64 && ((t->height == 48 && t->batch <= 3) || (t->height == 64 && t->batch <= 2))) {
        groups = std::min({20u, platform.GetCoreNumAic(), std::min(40u, available) / 2, t->batch * t->height * 2u});
        cube_status = groups ? TryA2Resources(platform, t->height) : CUBE_NO_GROUP;
        if (cube_status == CUBE_OK)
            route = 2;
        else
            groups = 0;
    }
    t->cube_status = cube_status;
    const uint32_t dtype = static_cast<uint32_t>(ge::DT_FLOAT);
    ASCENDC_TPL_SEL_PARAM(c, dtype, route);
    raw->SetDataSize(sizeof(*t));
    const uint32_t tiles = (t->height == 64 && t->width == 64 && t->batch == 3)   ? 13u
                           : (t->height == 64 && t->width == 64 && t->batch == 4) ? 10u
                                                                                  : t->height / t->rows;
    c->SetBlockDim(route == 3 ? std::min({40u, available, t->batch * 4u})
                   : route    ? groups
                              : std::min({40u, available, t->batch * tiles}));
    // SyncAll needs batch scheduling; generated runtime must configure the FFTS context.
    if (route != 3 && c->SetScheduleMode(1) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
    uint64_t userBytes = HeatmapBytes(t->batch, t->height, t->width);
    if (route == 1)
        userBytes = ProjectionBytes(t->batch, t->height, t->width);
    else if (route == 2)
        userBytes += A2HeatmapOffset(t->batch, t->height, t->width);
    workspace[0] =
        route == 3 ? 0 : static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) + static_cast<size_t>(userBytes);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c) {
    const auto *x = c->GetInputShape(0);
    auto *points = c->GetOutputShape(0);
    auto *scores = c->GetOutputShape(1);
    if (!x || !points || !scores || x->GetDimNum() != 4 || x->GetDim(0) < 1 || x->GetDim(0) > 4)
        return GRAPH_FAILED;
    points->SetDimNum(3);
    points->SetDim(0, x->GetDim(0));
    points->SetDim(1, 4);
    points->SetDim(2, 2);
    scores->SetDimNum(2);
    scores->SetDim(0, x->GetDim(0));
    scores->SetDim(1, 4);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *c) {
    c->SetOutputDataType(0, ge::DT_FLOAT);
    c->SetOutputDataType(1, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class FusedDisconnectorKeypointSoftArgmax : public OpDef {
  public:
    explicit FusedDisconnectorKeypointSoftArgmax(const char *name) : OpDef(name) {
        this->Input("feat").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("keypoints").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("scores").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("tau").AttrType(REQUIRED).Float();
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(FusedDisconnectorKeypointSoftArgmax);
} // namespace ops
