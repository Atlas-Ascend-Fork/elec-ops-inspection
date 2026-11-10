/*
 * Copyright (c) 2026 aodebiao.
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
#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_eigh_tiling.h"
#include "../op_kernel/tiling_key_linalg_eigh.h"
namespace optiling {
// Validate the origin shape and derive the matrix order and the batch count.
static ge::graphStatus CheckShape(gert::TilingContext *context, uint32_t &order, uint64_t &batch) {
    const auto *shapeDesc=context->GetInputShape(0);
    if (!shapeDesc) return ge::GRAPH_FAILED;
    const auto &shape=shapeDesc->GetOriginShape();
    const size_t dims=shape.GetDimNum();
    if (dims<2U) return ge::GRAPH_FAILED;
    const int64_t size=shape.GetDim(dims-1);
    if (size<1 || size>512 || shape.GetDim(dims-2)!=size) return ge::GRAPH_FAILED;
    for (size_t d=0;d+2<dims;++d)
        if (shape.GetDim(d)<1 || shape.GetDim(d)>64) return ge::GRAPH_FAILED;
    order=static_cast<uint32_t>(size);
    const uint64_t inner=static_cast<uint64_t>(order)*order;
    if (inner==0U) return ge::GRAPH_FAILED;
    batch=static_cast<uint64_t>(shape.GetShapeSize())/inner;
    if (!batch || batch>UINT32_MAX) return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}
// Validate the dtype and the optional UPLO attribute.
static ge::graphStatus CheckAttribute(gert::TilingContext *context, bool &lower) {
    const auto *inputDesc=context->GetInputDesc(0);
    if (!inputDesc || inputDesc->GetDataType()!=ge::DT_FLOAT) return ge::GRAPH_FAILED;
    const auto *attrs=context->GetAttrs();
    const char *uplo=attrs?attrs->GetStr(0):nullptr;
    lower=!uplo || (uplo[0]=='L' && uplo[1]==0);
    if (!lower && !(uplo[0]=='U' && uplo[1]==0)) return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus CheckInput(gert::TilingContext *context, uint32_t &order, uint64_t &batch, bool &lower) {
    if (CheckShape(context,order,batch)!=ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    return CheckAttribute(context,lower);
}
// One workspace slot per matrix: a core team shares the joining vector.
static void SelectGroupTeam(uint32_t n,uint64_t batch64,uint32_t cores,LinalgEighTilingData *t,uint32_t &blocks) {
    t->groups=blocks;t->team=1U;
    // A forced single block (odd order or tiny batch) must survive the grouping below.
    const uint32_t activeBlocks=blocks;
    if (activeBlocks<=1U) return;
    if (n<=25U) return;
    const uint32_t sectors=(n+7U)/8U;
    uint32_t team=1U;
    if (n<=128U) team=sectors;
    else if (activeBlocks<cores) team=cores/activeBlocks;
    if (team>cores) team=cores;
    if (team<1U) team=1U;
    uint32_t groups=team>0U?cores/team:1U;
    if (groups<1U) groups=1U;
    if (groups>static_cast<uint32_t>(batch64)) groups=static_cast<uint32_t>(batch64);
    t->groups=groups;t->team=team;
    if (t->team>1U) {blocks=t->groups*t->team;t->workspaceStride+=6U*512U+16U;t->resident=0U;}
    else blocks=t->groups;
}
// Pooled variant: a core keeps several matrices when the batch is small.
static void SelectPooledTeam(uint32_t n,uint64_t batch64,uint32_t cores,LinalgEighTilingData *t,uint32_t &blocks) {
    t->groups=blocks;t->team=1U;
    if (blocks<=1U) return;
    if (!(n>25U)) return;
    if (!(batch64<static_cast<uint64_t>(cores) || n%8U)) return;
    t->groups=static_cast<uint32_t>(batch64);
    if (t->groups>cores) t->groups=cores;
    blocks=t->groups*((n+7U)/8U);
    if (blocks>cores) blocks=cores;
    if (blocks>t->groups || n%8U) {t->team=2U;t->workspaceStride+=6U*512U+16U;t->resident=0U;}
}
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    uint32_t n=0U;uint64_t batch64=0U;bool lower=false;
    if (CheckInput(context,n,batch64,lower)!=ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    auto platform=platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t cores=platform.GetCoreNumAiv();if (cores<1) cores=1;
    const uint32_t coreCount=static_cast<uint32_t>(cores);
    uint32_t blocks=batch64<static_cast<uint64_t>(coreCount)?static_cast<uint32_t>(batch64):coreCount;
    if (n%8U) blocks=1U;
    auto *t=context->GetTilingData<LinalgEighTilingData>();if (!t) return ge::GRAPH_FAILED;
    t->batch=static_cast<uint32_t>(batch64);t->n=n;t->pitch=(n+7U)&~7U;
    // The kernel addresses the input with 32-bit offsets, so reject overflowing products here.
    if (static_cast<uint64_t>(t->batch)*n*t->pitch>UINT32_MAX) return ge::GRAPH_FAILED;
    t->lower=lower?1U:0U;t->workspaceStride=2U*n*t->pitch;t->resident=n<=128U?1U:0U;
    const bool POOLED=false;
    if (POOLED) SelectPooledTeam(n,batch64,coreCount,t,blocks);
    else SelectGroupTeam(n,batch64,coreCount,t,blocks);
    context->SetBlockDim(blocks);
    if (t->team>1U) context->SetScheduleMode(1U);
    const uint32_t datatype=static_cast<uint32_t>(ge::DT_FLOAT);
    const bool SMALL=n<=25U,RESIDENT=t->resident!=0U,COOPERATIVE=t->team>1U;
    ASCENDC_TPL_SEL_PARAM(context,datatype,SMALL,RESIDENT,COOPERATIVE,POOLED);
    context->GetWorkspaceSizes(1)[0]=static_cast<size_t>(t->groups)*t->workspaceStride*4U+platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const auto *a=context->GetInputShape(0);if (!a || a->GetDimNum()<2) return GRAPH_FAILED;
    *context->GetOutputShape(1)=*a;
    auto *v=context->GetOutputShape(0);*v=*a;v->SetDimNum(a->GetDimNum()-1);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0,DT_FLOAT);context->SetOutputDataType(1,DT_FLOAT);return GRAPH_SUCCESS;
}
}
namespace ops {
class LinalgEigh : public OpDef {
public:
explicit LinalgEigh(const char *name):OpDef(name) {
    this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Output("eigenvalues").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Output("eigenvectors").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
    this->Attr("UPLO").AttrType(OPTIONAL).String("L");
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
}
};OP_ADD(LinalgEigh);
}
