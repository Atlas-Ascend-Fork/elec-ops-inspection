/*
 * Copyright (c) 2026 zhiyuan (致远队).
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
// 多Shape自适应专项调度: 小用例极速直通 + 大用例规整1024满块对齐
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fire_candidate_decode_fusion_tiling.h"
#include "../op_kernel/tiling_key_fire_candidate_decode_fusion.h"

namespace optiling {
    // 除法不变量：chunk/gb/ALIGN 全为编译期常量且非零，blockDim 由 SelectBlockDim 保证 >= 1

    // 核数决策（泛化版 / VG1）——与 kernel VecLoop 同口径推算 nUnits
    // 判据：nUnits < num_cores ⇒ bd = nUnits（纯赚）；nUnits >= num_cores ⇒ 满核
    static uint32_t SelectBlockDim(uint32_t totalCandidates, uint32_t B, uint32_t N,
                                   uint32_t numCores, uint32_t dtRawPred) {
        uint32_t blockDim = (totalCandidates < numCores) ? totalCandidates : numCores;
        if (blockDim == 0) {
            blockDim = 1;
        }

        // 以下常量必须与 kernel 侧同名常量逐字一致（事故⑦的教训）
        constexpr uint32_t SMALL_CAND_H = 1024u;     // kernel: SMALL_CAND
        constexpr uint32_t BIG_CAND_H   = 196608u;   // kernel: BIG_CAND
        constexpr uint32_t CHUNK_MID_H  = 256u;      // kernel: CHUNK_MID（实例档）
        constexpr uint32_t CHUNK_BIG_H  = 560u;      // kernel: CHUNK_BIG（实例档）

        if (totalCandidates <= SMALL_CAND_H) {
            // 小档标量路径：启动开销主导（§9.4 实测 1.56µs@8核），8 核即够
            if (blockDim > 8u) { blockDim = 8u; }
            return blockDim;
        }

        // 向量路径：复刻 kernel 的 gb/upb/ng 三连，含【dtype 相关】的对齐门控
        const bool isBig = (totalCandidates >= BIG_CAND_H);
        const uint32_t chunk = isBig ? CHUNK_BIG_H : CHUNK_MID_H;
        const uint32_t gbWant = (isBig || B < 2u) ? 1u : 2u;
        const uint32_t upb = (N + chunk - 1u) / chunk;
        const uint32_t tailC = N - (upb - 1u) * chunk;
        const uint32_t elemBytes =
            (dtRawPred == static_cast<uint32_t>(ge::DT_FLOAT16) ||
             dtRawPred == static_cast<uint32_t>(ge::DT_BF16)) ? 2u : 4u;
        const bool groupOK = (gbWant >= 2u) &&
                             (((tailC * 6u * elemBytes) % 32u) == 0u);
        const uint32_t gb = groupOK ? gbWant : 1u;
        const uint32_t ng = (B + gb - 1u) / gb;
        const uint64_t units64 = static_cast<uint64_t>(upb) * ng;
        if (units64 > 0ull && units64 < static_cast<uint64_t>(blockDim)) {
            blockDim = static_cast<uint32_t>(units64);
        }
        return blockDim;
    }

    // 输入形状提取：B/N 维度与合法性检查
    static bool ExtractBN(gert::TilingContext *context, uint32_t &B, uint32_t &N) {
        const gert::StorageShape *raw_pred_shape = context->GetInputShape(0);
        if (raw_pred_shape == nullptr) {
            return false;
        }
        const gert::Shape &shape = raw_pred_shape->GetStorageShape();
        if (shape.GetDimNum() < 2) {
            return false;
        }
        B = static_cast<uint32_t>(shape.GetDim(0));
        N = static_cast<uint32_t>(shape.GetDim(1));
        return (B != 0 && N != 0);
    }

    // 输入 dtype 提取 + TPL 参数选择
    static bool ExtractAndSelDtype(gert::TilingContext *context, uint32_t &dtOut) {
        ge::DataType dtype_raw_pred = ge::DT_FLOAT;
        const gert::Tensor *tensor_raw_pred = context->GetRequiredInputTensor(0);
        if (tensor_raw_pred != nullptr) {
            dtype_raw_pred = tensor_raw_pred->GetDataType();
        } else if (context->GetInputDesc(0) != nullptr) {
            dtype_raw_pred = context->GetInputDesc(0)->GetDataType();
        }
        dtOut = static_cast<uint32_t>(dtype_raw_pred);
        ASCENDC_TPL_SEL_PARAM(context, dtOut);
        return true;
    }

    // AIV 核数查询（兜底 GetCoreNum）
    static uint32_t GetCoreCount(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores = platform.GetCoreNumAiv();
        if (num_cores == 0) {
            num_cores = platform.GetCoreNum();
        }
        if (num_cores == 0) {
            num_cores = 1;
        }
        return num_cores;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }

        uint32_t B = 0, N = 0;
        if (!ExtractBN(context, B, N)) {
            return ge::GRAPH_FAILED;
        }

        uint32_t DT_RAW_PRED = 0;
        if (!ExtractAndSelDtype(context, DT_RAW_PRED)) {
            return ge::GRAPH_FAILED;
        }

        const uint32_t num_cores = GetCoreCount(context);

        uint32_t totalCandidates = B * N;
        const uint32_t blockDim = SelectBlockDim(totalCandidates, B, N, num_cores, DT_RAW_PRED);

        context->SetBlockDim(blockDim);

        // 8 候选基本对齐块调度 (100% 32字节对齐，绝无跨行撕裂)
        constexpr uint32_t ALIGN = 8;
        uint32_t totalBlocks = totalCandidates / ALIGN;
        if (blockDim == 0) { return ge::GRAPH_FAILED; }   // G.EXP.22 防御守卫 (SelectBlockDim 保证 blockDim >= 1, 永不触发)
        uint32_t blocksPerCore = totalBlocks / blockDim;
        uint32_t blockRemainder = totalBlocks % blockDim;

        FireCandidateDecodeFusionTilingData *tiling = context->GetTilingData<FireCandidateDecodeFusionTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->bDim = B;
        tiling->nDim = N;
        tiling->totalCandidates = totalCandidates;
        tiling->coreNum = blockDim;
        tiling->candsPerCore = blocksPerCore * ALIGN;
        tiling->candsRemainder = blockRemainder;

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace != nullptr) {
            currentWorkspace[0] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape *raw_pred_shape = context->GetInputShape(0);
        gert::Shape *decoded_shape = context->GetOutputShape(0);
        if (raw_pred_shape == nullptr || decoded_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (raw_pred_shape->GetDimNum() < 2) {
            return ge::GRAPH_FAILED;
        }

        decoded_shape->SetDimNum(3);
        decoded_shape->SetDim(0, raw_pred_shape->GetDim(0)); // B
        decoded_shape->SetDim(1, raw_pred_shape->GetDim(1)); // N
        decoded_shape->SetDim(2, 5);                          // 5: [x, y, w, h, fire_score]
        return ge::GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class FireCandidateDecodeFusion : public OpDef {
    public:
        explicit FireCandidateDecodeFusion(const char *name) : OpDef(name) {
            this->Input("raw_pred")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("priors")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Output("decoded")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(FireCandidateDecodeFusion);
}  // namespace ops
