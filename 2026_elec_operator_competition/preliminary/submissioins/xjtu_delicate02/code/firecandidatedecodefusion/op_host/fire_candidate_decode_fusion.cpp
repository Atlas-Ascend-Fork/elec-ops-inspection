/*
 * Copyright (c) 2026 Delicate02. All rights reserved.
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

#include "securec.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "fire_host_plan.h"
#include "../op_kernel/tiling_key_fire_candidate_decode_fusion.h"

namespace optiling {
static uint32_t SelectBasicMode(uint32_t TILE,uint32_t control,bool halfInput) {
    const bool batchOne=(control>>29)&1u,dense=(control>>28)&1u;
    uint32_t MODE = 0;
    if (TILE <= 128) MODE = (TILE == 128 || dense) ? 1 : 0;
    else if (TILE <= 1024) MODE = batchOne ? (dense ? 2 : 1) : 0;
    else MODE = !halfInput && TILE == 1640 && (control & 65535u) ? 1 : 0;
    return MODE;
}
static uint32_t SelectFireMode(const FireHostPlan& plan,uint32_t batch,uint32_t n,bool halfInput) {
    const uint32_t TILE=plan.tile;
    const uint32_t control = plan.packet.control;
    const bool batchOne = (control >> 29) & 1u;
    const bool dense = (control >> 28) & 1u;
    uint32_t MODE=SelectBasicMode(TILE,control,halfInput);
    // A complete dense tile with one logical owner has no dynamic header fields.
    if (TILE <= 128 && dense && ((control >> 22U) & 63U) == 1U && n == TILE)
        MODE = 2U;

    // A nonaligned output batch stride proves the existing dual-output domain
    // without depending on the runtime output pointer. MODE2 is an existing,
    // otherwise unused large-Float uniform resource.
    const uint32_t owners = (control >> 22U) & 63U;
    const uint64_t totalTiles = ((uint64_t(n) + TILE - 1U) / TILE) * batch;
    if (!halfInput && TILE == 1640U && MODE == 1U && batch > 1U && owners != 0U &&
        totalTiles >= uint64_t(2) * owners && ((uint64_t(n) * 20U) & 31U) != 0U)
        MODE = 2U;

    return MODE;
}
static ge::graphStatus WriteFirePacket(gert::TilingContext* context,const FireHostPlan& plan,uint32_t batch) {
    const uint32_t TILE=plan.tile,control=plan.packet.control;
    constexpr size_t bytes=sizeof(FireCandidateDecodeFusionHeaderOnlyTilingData);
    auto *tiling=context->GetRawTilingData();
    if (tiling==nullptr || tiling->GetCapacity()<bytes) return ge::GRAPH_FAILED;
    auto *data=static_cast<uint8_t*>(tiling->GetData());
    if (data==nullptr) return ge::GRAPH_FAILED;
    if (memset_s(data, bytes, 0, bytes) != EOK) return ge::GRAPH_FAILED;
    if (TILE > 1024) {
        // Large blocked owners do not consume the packet's remainder field.
        // Its six bits carry the bounded batch in one ordinary launch word.
        const uint32_t compact[2] = {
            plan.packet.n,
            (control & ~(63u << 16)) | ((batch - 1u) << 16)};
        if (memcpy_s(data, bytes, compact, sizeof(compact)) != EOK) return ge::GRAPH_FAILED;
    } else {
        // The existing small/mid planner bounds units per owner below 8192:
        // compact <=1024, one-tile <=128, general mid <=1170.
        // Reuse the redundant bits for the bounded batch, preserving all flags.
        const uint32_t packed = (control & 65535u) |
            (((control >> 16) & 63u) << 13) |
            (((control >> 22) & 63u) << 19) |
            ((batch - 1u) << 25) | (control & 0xf0000000u);
        const uint32_t compact[2] = {plan.packet.n, packed};
        if (memcpy_s(data, bytes, compact, sizeof(compact)) != EOK) return ge::GRAPH_FAILED;
    }
    tiling->SetDataSize(bytes);
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const auto *input = context->GetRequiredInputTensor(0);
    if (input == nullptr) return ge::GRAPH_FAILED;
    const auto shape = input->GetShape().GetStorageShape();
    const uint32_t batch = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t n = static_cast<uint32_t>(shape.GetDim(1));
    const uint32_t DT_RAW_PRED = static_cast<uint32_t>(input->GetDataType());
    const bool halfInput = input->GetDataType() == ge::DT_FLOAT16;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAiv();
    if (batch == 0 || batch > 8 || n == 0 || cores == 0) return ge::GRAPH_FAILED;

    const FireHostPlan plan = fire_make_plan(batch, n, halfInput, cores);
    if (plan.launchBlocks == 0 || plan.tile == 0) return ge::GRAPH_FAILED;
    const uint32_t TILE = plan.tile;
    const uint32_t MODE=SelectFireMode(plan,batch,n,halfInput);
    if(WriteFirePacket(context,plan,batch)!=ge::GRAPH_SUCCESS)return ge::GRAPH_FAILED;
    ASCENDC_TPL_SEL_PARAM(context, DT_RAW_PRED, TILE, MODE);
    context->SetBlockDim(plan.launchBlocks);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    auto shape = *context->GetInputShape(0);
    shape.SetDim(2, 5);
    *context->GetOutputShape(0) = shape;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class FireCandidateDecodeFusion : public OpDef {
public:
    explicit FireCandidateDecodeFusion(const char *name) : OpDef(name) {
        this->Input("raw_pred").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("priors").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("decoded").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(FireCandidateDecodeFusion);
}
