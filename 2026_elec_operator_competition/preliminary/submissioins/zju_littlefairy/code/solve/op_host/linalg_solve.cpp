/*
 * Copyright (c) 2026 LittleFairy.
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
#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_solve_tiling.h"
#include "../op_kernel/tiling_key_linalg_solve.h"

namespace {
bool CountBatches(const gert::Shape &a,const gert::Shape &b,size_t ar,bool broadcast,uint64_t &batches) {
    batches=1;
    for(size_t i=0;i+2<ar;++i) {
        const int64_t dim=a.GetDim(i);
        if(dim<1 || dim>64 || batches>UINT32_MAX/(uint64_t)dim)return false;
        if(!broadcast && b.GetDim(i)!=dim)return false;
        batches*=dim;
    }
    return true;
}
// Matching batches, or an unbatched RHS broadcast over A, as in the problem.
bool Shapes(const gert::Shape &a, const gert::Shape &b,
            gert::Shape &x, LinalgSolveTilingData &t) {
    const size_t ar=a.GetDimNum(), br=b.GetDimNum();
    if(ar<2 || br<1)return false;
    const int64_t n=a.GetDim(ar-1);
    if(n<1 || n>512 || a.GetDim(ar-2)!=n)return false;
    bool matchingVector=br+1==ar;
    if(matchingVector)
        for(size_t i=0;i<br;++i)matchingVector &= b.GetDim(i)==a.GetDim(i);
    const bool vector=br==1 || matchingVector;
    const size_t tail=vector?1:2;
    if(br<tail || b.GetDim(br-tail)!=n)return false;
    const int64_t k=vector?1:b.GetDim(br-1);
    if(k<1 || k>256)return false;
    const bool broadcast=br==tail;
    if(!broadcast && br-tail!=ar-2)return false;
    uint64_t batches;
    if(!CountBatches(a,b,ar,broadcast,batches))return false;
    x=a;
    x.SetDimNum(ar-2+tail);
    x.SetDim(ar-2,n);
    if(!vector)x.SetDim(ar-1,k);
    t.n=n;t.k=k;t.batches=batches;t.broadcastB=broadcast;t.rows=8;
    while(t.rows<t.n)t.rows*=2;
    return true;
}
}
namespace optiling {
using Platform = platform_ascendc::PlatformAscendC;
static bool ValidateCall(gert::TilingContext *c, LinalgSolveTilingData &t) {
    gert::Shape expected;
    if(!Shapes(c->GetInputShape(0)->GetStorageShape(),c->GetInputShape(1)->GetStorageShape(),expected,t))return false;
    const auto &actual=c->GetOutputShape(0)->GetStorageShape();
    if(actual.GetDimNum()!=expected.GetDimNum())return false;
    for(size_t i=0;i<actual.GetDimNum();++i)
        if(actual.GetDim(i)!=expected.GetDim(i))return false;
    if(c->GetInputDesc(0)->GetDataType()!=ge::DT_FLOAT || c->GetInputDesc(1)->GetDataType()!=ge::DT_FLOAT)return false;
    const bool *left=c->GetAttrs()->GetBool(0);
    return !left || *left;
}
static void SelectVectorEntry(gert::TilingContext *c,LinalgSolveTilingData *t,Platform &platform,uint32_t path) {
    const uint32_t cores=platform.GetCoreNumAiv();
    uint32_t KERNEL_PATH=path,DT_A=ge::DT_FLOAT;
    t->teams=1;t->blocks=t->batches<cores?t->batches:cores;
    const uint32_t groupedBlocks=(t->blocks+7)/8*8;
    t->blocks=groupedBlocks<cores?groupedBlocks:cores;
    ASCENDC_TPL_SEL_PARAM(c,DT_A,KERNEL_PATH);c->SetBlockDim(t->blocks);
    c->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize();
}
static uint32_t SelectGeneralCubePath(const LinalgSolveTilingData *t,uint32_t cores,uint32_t cubeCores,uint32_t KERNEL_PATH) {
    if(t->k>8 && t->n>64 && t->n<=256 && cubeCores>=4 && cores==2*cubeCores)KERNEL_PATH=4;
    if(t->n>256 && cubeCores>=4 && cores==2*cubeCores)KERNEL_PATH=18;
    if(KERNEL_PATH==4 && t->n<=128)KERNEL_PATH=9;
    if(KERNEL_PATH==4 && t->batches>(cubeCores<16?cubeCores:16)/2)KERNEL_PATH=10;
    return KERNEL_PATH;
}
static uint32_t SelectResidentPath(const LinalgSolveTilingData *t,uint32_t cores,uint32_t cubeCores,uint32_t KERNEL_PATH) {
    // One residual tile (16 rows) per core avoids a second compensated pass.
    // Select these narrower column teams only when every batch remains resident.
    if(t->k<=8 && t->n>128 && t->n<=256) {
        if(t->n<=160 && t->batches<=cores/10)KERNEL_PATH=7;
        else if(t->n<=208 && t->batches<=cores/13)KERNEL_PATH=5;
        else if(t->batches<=cores/16)KERNEL_PATH=6;
        else if(cubeCores>=4 && cores==2*cubeCores && t->batches<=(cubeCores<16?cubeCores:16)/2)KERNEL_PATH=4;
    }
    KERNEL_PATH=SelectGeneralCubePath(t,cores,cubeCores,KERNEL_PATH);
    return KERNEL_PATH;
}
static uint32_t SelectBasePath(const LinalgSolveTilingData *t,uint32_t cores,uint32_t cubeCores) {
    uint32_t KERNEL_PATH=0;
    // Narrow RHS fits in one padded vector block. The larger path distributes
    // columns over a resident team and pads A by an identity block to its capacity.
    if(t->n<=64)KERNEL_PATH=1;
    else if(t->k<=8) {
        if(t->n<=128 && cores>=5)KERNEL_PATH=2;
        else if(t->n<=256 && cores>=9)KERNEL_PATH=3;
    }
    KERNEL_PATH=SelectResidentPath(t,cores,cubeCores,KERNEL_PATH);
    return KERNEL_PATH;
}
static uint32_t SelectFinalCapacityPath(const LinalgSolveTilingData *t,uint32_t cores,uint32_t cubeCores,uint32_t KERNEL_PATH) {
    if(KERNEL_PATH==18 && t->n<=384)KERNEL_PATH=21;
    if(t->n>208 && t->n<=256 && cubeCores>=4 && cores==2*cubeCores && KERNEL_PATH!=20)KERNEL_PATH=22;
    if(t->n>64 && t->n<=128 && t->k<=8 && cubeCores>=4 && cores==2*cubeCores)KERNEL_PATH=23;
    return KERNEL_PATH;
}
static uint32_t SelectOrthogonalPath(const LinalgSolveTilingData *t,uint32_t cores,uint32_t cubeCores,uint32_t KERNEL_PATH) {
    if((KERNEL_PATH==4 && t->n<=208) || (KERNEL_PATH==5 && cubeCores>=4 && cores==2*cubeCores))KERNEL_PATH=19;
    if(t->n>128 && t->n<=208 && t->k<=8 && t->batches>8 && cubeCores>=4 && cores==2*cubeCores)KERNEL_PATH=19;
    if(KERNEL_PATH==4 && t->n==256 && t->k<=8)KERNEL_PATH=20;
    KERNEL_PATH=SelectFinalCapacityPath(t,cores,cubeCores,KERNEL_PATH);
    return KERNEL_PATH;
}
static uint32_t SelectSpecialPath(const LinalgSolveTilingData *t,uint32_t cores,uint32_t cubeCores,uint32_t KERNEL_PATH) {
    if(t->k>=16) {
        if(KERNEL_PATH==4 || KERNEL_PATH==10)KERNEL_PATH=11;
        else if(KERNEL_PATH==9)KERNEL_PATH=13;
    }
    if(KERNEL_PATH==11 && t->k>=64)KERNEL_PATH=15;
    KERNEL_PATH=SelectOrthogonalPath(t,cores,cubeCores,KERNEL_PATH);
    return KERNEL_PATH;
}
static bool SelectOrthogonalEntry(gert::TilingContext *c,LinalgSolveTilingData *t,Platform &platform) {
    const uint32_t cores=platform.GetCoreNumAiv(),cubeCores=platform.GetCoreNumAic();
    uint32_t KERNEL_PATH=0;
    if(t->n>192 && t->n<=208 && t->k<=8 && t->batches<=4 && cubeCores>=16 && cores==2*cubeCores) {
        KERNEL_PATH=t->k<=4?26:25;uint32_t DT_A=ge::DT_FLOAT;
        ASCENDC_TPL_SEL_PARAM(c,DT_A,KERNEL_PATH);
        t->blocks=t->batches*4;t->teams=0;c->SetBlockDim(t->blocks);c->SetScheduleMode(1);
        c->GetWorkspaceSizes(1)[0]=(uint64_t)t->batches*kNarrowOrthogonalWorkspaceFloats*4+platform.GetLibApiWorkSpaceSize();
        return true;
    }
    if(t->n==256 && t->k<=8 && t->batches<=4 && cubeCores>=16 && cores==2*cubeCores) {
        KERNEL_PATH=24;uint32_t DT_A=ge::DT_FLOAT;
        ASCENDC_TPL_SEL_PARAM(c,DT_A,KERNEL_PATH);
        t->blocks=16;t->teams=0;c->SetBlockDim(16);c->SetScheduleMode(1);
        c->GetWorkspaceSizes(1)[0]=(uint64_t)t->batches*kOrthogonalWorkspaceFloats*4+platform.GetLibApiWorkSpaceSize();
        return true;
    }
    return false;
}
struct SolvePlan {
    uint32_t KERNEL_PATH,cubeN,cubeB,cubeLimit,cubeGroups,capacity;
    bool matrixPath,splitPath,cubePath;
};
static bool SelectMatrixPath(uint32_t KERNEL_PATH) {
    return KERNEL_PATH==22 || KERNEL_PATH==21 || KERNEL_PATH==18 || KERNEL_PATH==11 || KERNEL_PATH==13 || KERNEL_PATH==15;
}
static bool SelectSplitPath(uint32_t KERNEL_PATH) {
    return KERNEL_PATH==22 || KERNEL_PATH==21 || KERNEL_PATH==20 || KERNEL_PATH==19 || KERNEL_PATH==15 || KERNEL_PATH==18;
}
static uint32_t SelectCubeN(uint32_t KERNEL_PATH) {
    return KERNEL_PATH==21?384:(KERNEL_PATH==18?512:((KERNEL_PATH==9 || KERNEL_PATH==13 || KERNEL_PATH==23)?128:256));
}
static SolvePlan PlanResources(const LinalgSolveTilingData *t,uint32_t cores,uint32_t cubeCores,uint32_t KERNEL_PATH) {
    const bool matrixPath=SelectMatrixPath(KERNEL_PATH);
    const bool splitPath=SelectSplitPath(KERNEL_PATH);
    const bool cubePath=KERNEL_PATH>=8;
    const uint32_t cubeN=SelectCubeN(KERNEL_PATH);
    const uint32_t cubeB=32;
    const bool normalPath=KERNEL_PATH==18 || KERNEL_PATH==20 || KERNEL_PATH==21 || KERNEL_PATH==22;
    const uint32_t baseCubeLimit=cubeCores<16?cubeCores:16;
    const uint64_t normalTiles=(uint64_t)t->batches*((cubeN+127)/128)*((cubeN+127)/128);
    const bool fewerWaves=normalPath && cubeCores>baseCubeLimit && (normalTiles+baseCubeLimit-1)/baseCubeLimit>(normalTiles+cubeCores-1)/cubeCores;
    const uint32_t cubeLimit=(cubePath && t->batches>8) || fewerWaves?cubeCores:baseCubeLimit;
    const uint32_t cubeGroups=t->batches<cubeLimit/2?t->batches:cubeLimit/2;
    const uint32_t capacity=KERNEL_PATH==2?128:(KERNEL_PATH==5?208:(KERNEL_PATH==7?160:256));
    return {KERNEL_PATH,cubeN,cubeB,cubeLimit,cubeGroups,capacity,matrixPath,splitPath,cubePath};
}
template<size_t Count> static void CopyTiling(uint32_t (&words)[Count],const AscendC::tiling::TCubeTiling &value) {
    static_assert(sizeof(value)<=sizeof(words));
    std::fill_n(words,Count,uint32_t(0));
    const auto *src=reinterpret_cast<const unsigned char*>(&value);
    std::copy_n(src,sizeof(value),reinterpret_cast<unsigned char*>(words));
}
static bool ConfigureSplitTiling(Platform &platform,LinalgSolveCubeTilingData *cube,const SolvePlan &p) {
    const auto *t=&cube->base;
    const uint32_t KERNEL_PATH=p.KERNEL_PATH,cubeN=p.cubeN;
            matmul_tiling::MatmulApiTiling hp(platform);
            const bool int8Residual=KERNEL_PATH==18 || KERNEL_PATH==20 || KERNEL_PATH==21 || KERNEL_PATH==22;
            hp.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,(int8Residual?matmul_tiling::DataType::DT_INT8:matmul_tiling::DataType::DT_FLOAT16),false);
            hp.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,(int8Residual?matmul_tiling::DataType::DT_INT8:matmul_tiling::DataType::DT_FLOAT16),true);
            hp.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,(int8Residual?matmul_tiling::DataType::DT_INT32:matmul_tiling::DataType::DT_FLOAT));
            hp.SetOrgShape(t->k,cubeN,int8Residual?4*cubeN:cubeN);hp.SetShape(128,128,int8Residual?4*cubeN:cubeN);hp.SetFixSplit(128,128,64);hp.EnableBias(false);hp.SetBufferSpace(128*1024,-1,-1);
            AscendC::tiling::TCubeTiling ht{};if(hp.GetTiling(ht)!=0)return false;
            CopyTiling(cube->halfWords,ht);
    return true;
}
static bool ConfigureCubeTiling(Platform &platform,LinalgSolveCubeTilingData *cube,const SolvePlan &p) {
    const uint32_t KERNEL_PATH=p.KERNEL_PATH,cubeN=p.cubeN,cubeB=p.cubeB;
    const bool matrixPath=p.matrixPath;
        matmul_tiling::MatmulApiTiling api(platform);
        api.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_FLOAT,true);
        api.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_FLOAT,false);
        api.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_FLOAT);
        const uint32_t maxK=(matrixPath || KERNEL_PATH==20)?cubeN:cubeB;
        api.SetOrgShape(cubeN,2*cubeN,maxK);api.SetShape(128,128,maxK);api.SetFixSplit(128,128,cubeB);api.EnableBias(false);api.SetBufferSpace(-1,-1,-1);
        AscendC::tiling::TCubeTiling tc{};if(api.GetTiling(tc)!=0)return false;
        CopyTiling(cube->cubeWords,tc);cube->systemWorkspaceBytes=platform.GetLibApiWorkSpaceSize();
    return !p.splitPath || ConfigureSplitTiling(platform,cube,p);
}
static void SetWorkspace(gert::TilingContext *c,LinalgSolveTilingData *t,Platform &platform,const SolvePlan &p) {
    const uint32_t KERNEL_PATH=p.KERNEL_PATH,capacity=p.capacity,cubeN=p.cubeN,cubeB=p.cubeB,cubeGroups=p.cubeGroups;
    const bool cubePath=p.cubePath,splitPath=p.splitPath;
    const uint64_t words=KERNEL_PATH>=2?
        (uint64_t)(t->blocks/t->teams)*(2*(capacity*16+16)+capacity*capacity+2*t->k*256):
        (uint64_t)t->blocks*(2*(t->n+t->k)*t->rows+((t->n*t->k+7)/8*8));
    c->GetWorkspaceSizes(1)[0]=(cubePath?(uint64_t)cubeGroups*(3*cubeN*cubeN+3*cubeN*cubeB+cubeB+3*t->k*cubeN):words)*4+platform.GetLibApiWorkSpaceSize();
    if(splitPath)c->GetWorkspaceSizes(1)[0]+=(uint64_t)cubeGroups*(2*cubeN*cubeN+2*t->k*cubeN+16*t->k*cubeN+2*cubeN)*4;
    if(KERNEL_PATH==18 || KERNEL_PATH==20 || KERNEL_PATH==21 || KERNEL_PATH==22)c->GetWorkspaceSizes(1)[0]+=(uint64_t)cubeGroups*cubeN*cubeN*4;
}
static ge::graphStatus TilingFunc(gert::TilingContext *c) {
    auto *cube=c->GetTilingData<LinalgSolveCubeTilingData>();
    auto *t=&cube->base;
    if(!ValidateCall(c,*t))return ge::GRAPH_FAILED;
    Platform platform(c->GetPlatformInfo());
    const uint32_t cores=platform.GetCoreNumAiv(),cubeCores=platform.GetCoreNumAic();
    if(!cores)return ge::GRAPH_FAILED;
    if(t->n>8 && t->n<=16 && t->k==1) {
        SelectVectorEntry(c,t,platform,27);return ge::GRAPH_SUCCESS;
    }
    if(t->n>32 && t->n<=64 && t->k<=8) {
        SelectVectorEntry(c,t,platform,28);return ge::GRAPH_SUCCESS;
    }
    uint32_t KERNEL_PATH=SelectBasePath(t,cores,cubeCores);
    KERNEL_PATH=SelectSpecialPath(t,cores,cubeCores,KERNEL_PATH);
    if(SelectOrthogonalEntry(c,t,platform))return ge::GRAPH_SUCCESS;
    const auto plan=PlanResources(t,cores,cubeCores,KERNEL_PATH);
    const uint32_t capacity=plan.capacity;
    t->teams=KERNEL_PATH>=5?capacity/16:(KERNEL_PATH>=2?capacity/32+1:1);
    const uint32_t groups=cores/t->teams;
    t->blocks=(t->batches<groups?t->batches:groups)*t->teams;
    uint32_t DT_A=ge::DT_FLOAT;
    if(plan.cubePath) {
        if(!ConfigureCubeTiling(platform,cube,plan))return ge::GRAPH_FAILED;
        c->SetScheduleMode(1);
    }
    ASCENDC_TPL_SEL_PARAM(c,DT_A,KERNEL_PATH);
    c->SetBlockDim(plan.cubePath?plan.cubeLimit:t->blocks);
    SetWorkspace(c,t,platform,plan);
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c) {
    LinalgSolveTilingData t{};
    return Shapes(*c->GetInputShape(0),*c->GetInputShape(1),*c->GetOutputShape(0),t)
        ? GRAPH_SUCCESS:GRAPH_FAILED;
}
static graphStatus InferDataType(gert::InferDataTypeContext *c) {
    c->SetOutputDataType(0,ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}
namespace ops {
class LinalgSolve:public OpDef {
public:
    explicit LinalgSolve(const char *name):OpDef(name) {
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("B").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("X").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("left").AttrType(OPTIONAL).Bool(true);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(LinalgSolve);
}
