/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 黎何.
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
#include <cstring>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_lstsq_tiling.h"
#include "../op_kernel/tiling_key_linalg_lstsq.h"

namespace {
bool ValidateShapes(const gert::Shape& a, const gert::Shape& b, uint32_t& batches)
{
    const size_t rank = a.GetDimNum();
    if (rank < 2 || b.GetDimNum() != rank) {
        return false;
    }
    // Validate signed dimensions before narrowing them to the tiling fields.
    const int64_t m = a.GetDim(rank - 2);
    const int64_t n = a.GetDim(rank - 1);
    const int64_t k = b.GetDim(rank - 1);
    if (m < 1 || m > 1024 || n < 1 || n > 512 || n > m || k < 1 || k > 64 || b.GetDim(rank - 2) != m) {
        return false;
    }
    batches = 1;
    for (size_t i = 0; i + 2 < rank; ++i) {
        const int64_t dim = a.GetDim(i);
        if (batches == 0U || dim < 1 || dim != b.GetDim(i) || dim > std::numeric_limits<uint32_t>::max() / batches) {
            return false;
        }
        batches *= static_cast<uint32_t>(dim);
    }
    return true;
}
}

namespace optiling {
static uint32_t SelectSquareCase(uint32_t m, uint32_t k)
{
    if (m == 16U && k == 1U) {
        return 1U;
    }
    if (m == 64U && k == 2U) {
        return 2U;
    }
    if (m == 256U && k == 3U) {
        return 3U;
    }
    if (m == 15U && k == 1U) {
        return 4U;
    }
    if (m == 50U && k == 2U) {
        return 5U;
    }
    return m == 200U && k == 4U ? 6U : 0U;
}

static bool TrySquareTiling(gert::TilingContext* c,
                            platform_ascendc::PlatformAscendC& platform,
                            uint32_t m,
                            uint32_t k,
                            uint32_t batches,
                            uint32_t cores)
{
    const uint32_t caseId = SelectSquareCase(m, k);
    const uint32_t coresPerBatch = caseId == 3U ? 8U : (caseId == 6U ? 10U : 1U);
    if (caseId == 0U || batches > cores / coresPerBatch) {
        return false;
    }
    ASCENDC_TPL_SEL_PARAM(c, 0U, caseId);
    c->SetBlockDim(batches * coresPerBatch);
    if (coresPerBatch == 1U) {
        c->GetWorkspaceSizes(1)[0] = 0;
        return true;
    }
    c->SetScheduleMode(1);
    const uint32_t matrixSize = caseId == 3U ? 256U * 264U : 200U * 200U;
    c->GetWorkspaceSizes(1)[0] =
        platform.GetLibApiWorkSpaceSize() + static_cast<size_t>(batches) * (32768U + matrixSize) * sizeof(float);
    return true;
}

static ge::graphStatus SetGeneralTiling(gert::TilingContext* c,
                                        platform_ascendc::PlatformAscendC& platform,
                                        uint32_t m,
                                        uint32_t n,
                                        uint32_t k,
                                        uint32_t batches,
                                        uint32_t cores,
                                        uint32_t DT_A)
{
    uint32_t mp = 8;
    while (mp < m)
        mp *= 2;
    uint32_t total = n + k;
    uint32_t PAR = (total * mp > 20480U) ? 1U : 0U;
    uint32_t groupSize = 1;
    uint32_t groups = std::min(batches, cores);
    if (groups == 0U) {
        return ge::GRAPH_FAILED;
    }
    uint32_t localCols = PAR ? std::min(total, std::max(1U, 20480U / mp)) : total;
    auto* t = c->GetTilingData<LinalgLstsqTilingData>();
    *t = {m, n, k, batches, mp, groupSize, groups, localCols, batches / groups + (batches % groups != 0U)};
    uint32_t CASE_ID = PAR ? 7U : 0U;
    ASCENDC_TPL_SEL_PARAM(c, DT_A, CASE_ID);
    c->SetBlockDim(groups);
    c->GetWorkspaceSizes(1)[0] =
        PAR ? platform.GetLibApiWorkSpaceSize() + static_cast<size_t>(groups) * 2U * (total + 1U) * mp * sizeof(float)
            : 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* c)
{
    auto sa = c->GetInputShape(0);
    auto sb = c->GetInputShape(1);
    if (!sa || !sb)
        return ge::GRAPH_FAILED;
    const auto& a = sa->GetStorageShape();
    const auto& b = sb->GetStorageShape();
    uint32_t batches = 1;
    if (!ValidateShapes(a, b, batches)) {
        return ge::GRAPH_FAILED;
    }
    const size_t rank = a.GetDimNum();
    const uint32_t m = static_cast<uint32_t>(a.GetDim(rank - 2));
    const uint32_t n = static_cast<uint32_t>(a.GetDim(rank - 1));
    const uint32_t k = static_cast<uint32_t>(b.GetDim(rank - 1));
    const auto* descA = c->GetInputDesc(0);
    const auto* descB = c->GetInputDesc(1);
    if (descA == nullptr || descB == nullptr || descA->GetDataType() != descB->GetDataType() ||
        (descA->GetDataType() != ge::DT_FLOAT && descA->GetDataType() != ge::DT_BF16)) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    uint32_t cores = std::max(1U, static_cast<uint32_t>(platform.GetCoreNumAiv()));
    uint32_t DT_A = static_cast<uint32_t>(descA->GetDataType());
    if (DT_A == 0U && m == n && batches <= cores && TrySquareTiling(c, platform, m, k, batches, cores)) {
        return ge::GRAPH_SUCCESS;
    }
    return SetGeneralTiling(c, platform, m, n, k, batches, cores, DT_A);
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* c)
{
    auto* a = c->GetInputShape(0);
    auto* b = c->GetInputShape(1);
    uint32_t batches = 1;
    if (!a || !b || !ValidateShapes(*a, *b, batches))
        return GRAPH_FAILED;
    auto rank = a->GetDimNum();
    auto* x = c->GetOutputShape(0);
    if (x == nullptr) {
        return GRAPH_FAILED;
    }
    *x = *a;
    x->SetDim(rank - 2, a->GetDim(rank - 1));
    x->SetDim(rank - 1, b->GetDim(rank - 1));
    auto* r = c->GetOutputShape(1);
    if (r) {
        if (a->GetDim(rank - 2) == a->GetDim(rank - 1)) {
            r->SetDimNum(1);
            r->SetDim(0, 0);
        } else {
            r->SetDimNum(rank - 1);
            for (size_t i = 0; i + 2 < rank; i++)
                r->SetDim(i, a->GetDim(i));
            r->SetDim(rank - 2, b->GetDim(rank - 1));
        }
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* c)
{
    const auto dtype = c->GetInputDataType(0);
    if ((dtype != DT_FLOAT && dtype != DT_BF16) || dtype != c->GetInputDataType(1)) {
        return GRAPH_FAILED;
    }
    if (c->SetOutputDataType(0, dtype) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (c->GetComputeNodeOutputNum() > 1) {
        return c->SetOutputDataType(1, dtype);
    }
    return GRAPH_SUCCESS;
}
}

namespace ops {
class LinalgLstsq : public OpDef {
public:
    explicit LinalgLstsq(const char* name) : OpDef(name)
    {
        this->Input("A")
            .AutoContiguous()
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("B")
            .AutoContiguous()
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("solution")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("residuals")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("driver").AttrType(OPTIONAL).String("gels");
        this->Attr("rcond").AttrType(OPTIONAL).Float(-1.0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(LinalgLstsq);
}
