/*
 * Copyright (c) 2026 FalseShare contributors.
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

#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_pinv_tiling.h"
#include "../op_kernel/tiling_key_linalg_pinv.h"

namespace optiling {
struct PinvPlan {
    uint32_t m, n, batches, cores, cubeCores, dtype, smallWidth, smallShards;
    bool useCube, tinyPath, smallTeam;
};

static bool UseCube(const PinvPlan &p) {
    return p.dtype==ge::DT_FLOAT && p.m>PINV_ROW_BLOCK_PANEL && p.m<=PINV_ROW_BLOCK_CAPACITY
        && p.n==p.m && p.batches>0 && p.batches<=p.cubeCores/PINV_CUBE_CORES_PER_BATCH
        && p.cores==2*p.cubeCores && p.cores<=64;
}

static bool ReadPlan(gert::TilingContext *c, PinvPlan &p) {
    const auto &shape=c->GetInputShape(0)->GetStorageShape();
    const auto rank=shape.GetDimNum();
    if(rank<2)return false;
    const int64_t m=shape.GetDim(rank-2), n=shape.GetDim(rank-1);
    if(m<1 || n<1 || m>512 || n>512)return false;
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    p.m=m; p.n=n; p.batches=shape.GetShapeSize()/(m*n);
    p.cores=platform.GetCoreNumAiv(); p.cubeCores=platform.GetCoreNumAic();
    p.dtype=c->GetInputDesc(0)->GetDataType(); p.useCube=UseCube(p);
    p.tinyPath=p.dtype==ge::DT_FLOAT && m==n && m>=PINV_TINY_MIN_ORDER && m<=PINV_TINY_CAPACITY;
    return true;
}

static void PlanColumns(PinvPlan &p) {
    uint32_t budget=p.batches?p.cores/p.batches:0;
    if(budget>PINV_COLUMN_MAX_SHARDS)budget=PINV_COLUMN_MAX_SHARDS;
    p.smallWidth=budget?(p.n+budget-1)/budget:0;
    p.smallShards=p.smallWidth?(p.n+p.smallWidth-1)/p.smallWidth:0;
    p.smallTeam=p.m==p.n && p.dtype==ge::DT_FLOAT && p.n>PINV_TINY_CAPACITY
        && p.n<=PINV_COLUMN_CAPACITY && p.smallWidth>0 && p.smallWidth<=PINV_COLUMN_MAX_WIDTH;
}

static uint16_t EncodeHalfOffset(uint32_t offset) {
    if(offset==0)return 0;
    union {float value; uint32_t bits;} encoded;
    encoded.value=static_cast<float>(offset);
    return static_cast<uint16_t>((((encoded.bits>>23)-112)<<10)|((encoded.bits>>13)&1023));
}

static void PrepareTinyTables(LinalgPinvTinyTilingData &tiny, uint32_t m) {
    for(auto &padding:tiny.padding)padding=0;
    for(uint32_t row=0;row<16;++row) {
        for(uint32_t col=0;col<8;++col) {
            auto inputOffset=[&](uint32_t c){return (row<m && c<m?row*m+c:m*m)*4;};
            tiny.packed[row*32+col]=(uint32_t)EncodeHalfOffset(inputOffset(2*col))|((uint32_t)EncodeHalfOffset(inputOffset(2*col+1))<<16);
            tiny.packed[row*32+8+col]=(uint32_t)EncodeHalfOffset((2*col*32+row)*4)|((uint32_t)EncodeHalfOffset(((2*col+1)*32+row)*4)<<16);
        }
        for(uint32_t col=0;col<16;++col)tiny.packed[row*32+16+col]=row==col?0x3f800000u:0u;
    }
}

static bool ConfigureBase(gert::TilingContext *c, const PinvPlan &p, LinalgPinvTilingData &t) {
    t.m=p.m; t.n=p.n; t.batches=p.batches;
    t.columns=p.m<p.n?p.m:p.n; t.rows=p.m>p.n?p.m:p.n;
    t.stride=((t.rows+7)/8+(t.columns+7)/8)*8;
    t.parallel=p.smallTeam?2:(t.columns>=64 && !(p.m==p.n && p.m<=64));
    t.resident=t.parallel!=1 && t.columns*t.stride*4<=128*1024;
    const auto *attrs=c->GetAttrs();
    t.atol=*attrs->GetFloat(0); t.rtol=*attrs->GetFloat(1);
    if(t.rtol<0)t.rtol=t.atol>0?0.0f:(float)(p.m>p.n?p.m:p.n)*1.1920928955078125e-7f;
    return !*attrs->GetBool(2);
}

static uint32_t LaunchBlocks(const PinvPlan &p, const LinalgPinvTilingData &t) {
    uint32_t blocks=p.useCube?(p.cubeCores>PINV_CUBE_CORES_PER_BATCH*t.batches?PINV_CUBE_CORES_PER_BATCH*t.batches:p.cubeCores)
        :p.smallTeam?t.batches*p.smallShards:(t.parallel?p.cores:(t.batches<p.cores?t.batches:p.cores));
    if(p.dtype==ge::DT_FLOAT && p.m==p.n && p.m<=16 && blocks<PINV_TINY_MIN_BLOCKS)
        blocks=p.cores<PINV_TINY_MIN_BLOCKS?p.cores:PINV_TINY_MIN_BLOCKS;
    return blocks;
}

static uint64_t UserWorkspace(const PinvPlan &p, const LinalgPinvTilingData &t) {
    uint64_t bytes=p.smallTeam?((uint64_t)t.batches*p.smallShards*(p.smallWidth*((p.n+7)/8*8)+8))*4
        :(t.resident?0:(uint64_t)t.batches*t.columns*(t.stride+(t.parallel?8:0))*4)
        +((t.parallel && p.m==p.n)?(uint64_t)t.batches*p.m*p.n*4:0);
    if(p.useCube) {
        const uint64_t blockBytes=(uint64_t)t.batches*PINV_ROW_BLOCK_WORKSPACE_FLOATS*sizeof(float);
        if(bytes<blockBytes)bytes=blockBytes;
    }
    return bytes;
}

static ge::graphStatus TilingFunc(gert::TilingContext *c) {
    PinvPlan plan;
    if(!ReadPlan(c,plan))return ge::GRAPH_FAILED;
    PlanColumns(plan);
    auto platform=platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    auto *cube=plan.useCube?c->GetTilingData<LinalgPinvCubeTilingData>():nullptr;
    if(plan.useCube && cube==nullptr)return ge::GRAPH_FAILED;
    auto *tiny=plan.tinyPath?c->GetTilingData<LinalgPinvTinyTilingData>():nullptr;
    if(plan.tinyPath && tiny==nullptr)return ge::GRAPH_FAILED;
    auto *t=plan.useCube?&cube->base:plan.tinyPath?&tiny->base:c->GetTilingData<LinalgPinvTilingData>();
    if(plan.tinyPath)PrepareTinyTables(*tiny,plan.m);
    if(t==nullptr || !ConfigureBase(c,plan,*t))return ge::GRAPH_FAILED;
    uint32_t DT_A=plan.dtype,KERNEL_PATH=t->parallel?1:0;
    if(plan.smallTeam)KERNEL_PATH=6;
    if(plan.tinyPath)KERNEL_PATH=5;
    if(plan.useCube) {
        KERNEL_PATH=4;
        cube->systemWorkspaceBytes=platform.GetLibApiWorkSpaceSize();
    }
    ASCENDC_TPL_SEL_PARAM(c,DT_A,KERNEL_PATH);
    c->SetBlockDim(LaunchBlocks(plan,*t));
    if(t->parallel)c->SetScheduleMode(1);
    const uint64_t bytes=UserWorkspace(plan,*t);
    c->GetWorkspaceSizes(1)[0]=bytes?bytes+platform.GetLibApiWorkSpaceSize():0;
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c) {
    auto shape=*c->GetInputShape(0);
    auto rank=shape.GetDimNum();
    if(rank<2) return GRAPH_FAILED;
    auto m=shape.GetDim(rank-2), n=shape.GetDim(rank-1);
    shape.SetDim(rank-2,n); shape.SetDim(rank-1,m);
    *c->GetOutputShape(0)=shape;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *c) {
    c->SetOutputDataType(0,c->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}
namespace ops {
class LinalgPinv:public OpDef {
public:
    explicit LinalgPinv(const char *name):OpDef(name) {
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        this->Output("A_pinv").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_BF16}).Format({ge::FORMAT_ND,ge::FORMAT_ND});
        this->Attr("atol").AttrType(OPTIONAL).Float(0.0f);
        this->Attr("rtol").AttrType(OPTIONAL).Float(-1.0f);
        this->Attr("hermitian").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(LinalgPinv);
}
