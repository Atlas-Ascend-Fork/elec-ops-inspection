/*
 * Copyright (c) 2026 hit_op-samerberry.
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
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/linalg_solve_tiling.h"
#include "../op_kernel/tiling_key_linalg_solve.h"

namespace optiling {
    static bool ResolveSolveDims(const gert::Shape &as, const gert::Shape &bs,
                                 const gert::RuntimeAttrs *attrs, uint32_t &n, uint32_t &k) {
        const bool *attrLeft = attrs ? attrs->GetBool(0) : nullptr;
        n = as.GetDim(as.GetDimNum() - 1);
        k = bs.GetDimNum() == 1 ? 1 : bs.GetDim(bs.GetDimNum() - 1);
        if (bs.GetDimNum() == as.GetDimNum() - 1 && bs.GetDim(bs.GetDimNum() - 1) == n &&
            (bs.GetDimNum() < 2 || bs.GetDim(bs.GetDimNum() - 2) != n)) {
            k = 1;
        }
        if (n == 0 || n > 512 || as.GetDim(as.GetDimNum() - 2) != n) {
            return false;
        }
        if (k == 0 || k > 256 || (attrLeft && !*attrLeft)) {
            return false;
        }
        return true;
    }

    static uint32_t SelectShapeRegime(ge::DataType dtypeA, uint32_t n, uint32_t k,
                                      uint32_t batch, uint32_t aivCores) {
        if (dtypeA != ge::DT_FLOAT || n < 8 || batch > aivCores) {
            return 0;
        }
        if (n <= 16 && k == 1) return 1;
        if (n <= 64 && k <= 8) return 2;
        if (n == 200 && k == 4) return 3;
        if (n == 256 && k == 3) return 4;
        return 0;
    }

    static void ConfigureTilingPlan(LinalgSolveTilingData *tiling, uint32_t shape,
                                    uint32_t n, uint32_t k, uint32_t batch,
                                    uint32_t padn, uint32_t aivCores) {
        if (shape == 1 || shape == 2) {
            tiling->parts = 1;
            tiling->stride = padn;
            tiling->groups = std::min(batch, aivCores);
            tiling->cores = tiling->groups;
        } else if (shape == 3 || shape == 4) {
            tiling->parts = 8;
            tiling->stride = ((padn / 8 + 7) / 8) * 8;
            tiling->groups = std::max(1u, std::min(batch, aivCores / tiling->parts));
            tiling->cores = tiling->groups * tiling->parts;
        } else {
            tiling->stride = padn;
            tiling->parts = 1;
            while (padn * tiling->stride > 40960 ||
                   (8 * tiling->stride + 2 * padn > 4096 - 8 * padn &&
                    8 * tiling->stride + 2 * padn > 40960 - padn * tiling->stride)) {
                tiling->parts *= 2;
                tiling->stride = ((n + tiling->parts - 1) / tiling->parts + 7) / 8 * 8;
            }
            tiling->groups = std::min(batch * k, aivCores / tiling->parts);
            tiling->cores = tiling->groups * tiling->parts;
        }
        if (shape != 0) {
            tiling->n = n | (batch << 9) | ((tiling->bBatch == 1 ? 1u : 0u) << 15) |
                        (tiling->cores << 16) | (k << 22) | (tiling->groups << 26);
        }
        tiling->rows = 0;
    }

    static size_t CalcWorkspaceElements(const LinalgSolveTilingData *tiling, uint32_t shape,
                                        uint32_t n, uint32_t k, uint32_t batch, uint32_t padn) {
        if (shape == 1 || shape == 2) {
            return 0;
        }
        if (shape == 3 || shape == 4) {
            return size_t(tiling->groups) * (n * 16 + 4 * padn + 128 + n * padn) +
                   size_t(batch) * k * padn;
        }
        return size_t(tiling->groups) * 2 * (padn + 8) +
               (size_t(tiling->groups) * n + size_t(batch) * k) * padn;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t aivCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        const gert::Tensor *tensorA = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensorB = context->GetRequiredInputTensor(1);
        ge::DataType dtypeA = tensorA->GetDataType();
        uint32_t n = 0;
        uint32_t k = 0;
        if (!ResolveSolveDims(tensorA->GetStorageShape(), tensorB->GetStorageShape(),
                              context->GetAttrs(), n, k)) {
            return ge::GRAPH_FAILED;
        }

        uint32_t batch = tensorA->GetShapeSize() / (n * n);
        uint32_t DT_A = static_cast<uint32_t>(dtypeA);
        uint32_t SHAPE = SelectShapeRegime(dtypeA, n, k, batch, aivCores);
        ASCENDC_TPL_SEL_PARAM(context, DT_A, SHAPE);

        LinalgSolveTilingData *tiling = context->GetTilingData<LinalgSolveTilingData>();
        tiling->n = n;
        tiling->k = k;
        tiling->batch = batch;
        tiling->bBatch = tensorB->GetShapeSize() / (n * k);

        uint32_t padn = (SHAPE == 3 || SHAPE == 4) ? ((n + 15) / 16 * 16) : ((n + 7) / 8 * 8);
        ConfigureTilingPlan(tiling, SHAPE, n, k, batch, padn, aivCores);

        context->SetBlockDim(tiling->cores);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        size_t wsElements = CalcWorkspaceElements(tiling, SHAPE, n, k, batch, padn);
        currentWorkspace[0] = platform.GetLibApiWorkSpaceSize() + wsElements * sizeof(float);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const auto *a = context->GetInputShape(0);
        const auto *b = context->GetInputShape(1);
        auto *x = context->GetOutputShape(0);
        if (!a || !b || !x || a->GetDimNum() < 2 || b->GetDimNum() < 1) return GRAPH_FAILED;
        int64_t n = a->GetDim(a->GetDimNum() - 1);
        bool vec = b->GetDimNum() == 1 ||
            (b->GetDimNum() == a->GetDimNum() - 1 &&
             b->GetDim(b->GetDimNum() - 1) == n &&
             b->GetDim(b->GetDimNum() - 2) != n);
        *x = *a;
        if (vec) {
            x->SetDimNum(a->GetDimNum() - 1);
            x->SetDim(a->GetDimNum() - 2, n);
        } else {
            x->SetDim(a->GetDimNum() - 1, b->GetDim(b->GetDimNum() - 1));
        }
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class LinalgSolve : public OpDef {
    public:
        explicit LinalgSolve(const char *name) : OpDef(name) {
            const std::vector<ge::DataType> solveTypes = {ge::DT_FLOAT, ge::DT_BF16};
            const std::vector<ge::Format> solveFormats = {ge::FORMAT_ND, ge::FORMAT_ND};
            for (const char *portName : {"A", "B"}) {
                this->Input(portName).ParamType(REQUIRED).DataType(solveTypes).Format(solveFormats);
            }
            this->Output("X").ParamType(REQUIRED).DataType(solveTypes).Format(solveFormats);
            this->Attr("left").AttrType(OPTIONAL).Bool(true);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        }
    };
    OP_ADD(LinalgSolve);
}  // namespace ops
