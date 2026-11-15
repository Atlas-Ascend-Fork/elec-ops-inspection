/*
 * Copyright (c) 2026 zjyzjy2002.
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
#include "../op_kernel/linalg_eigh_tiling.h"
#include "../op_kernel/tiling_key_linalg_eigh.h"
namespace optiling {
static ge::graphStatus ReadShape(gert::TilingContext *ctx, LinalgEighTilingData *t) {
    const auto *shape = ctx->GetInputShape(0);
    const auto *desc = ctx->GetInputDesc(0);
    if (!shape || !desc || desc->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    const auto &s = shape->GetStorageShape();
    const auto rank = s.GetDimNum();
    if (rank < 2) return ge::GRAPH_FAILED;
    int64_t n = s.GetDim(rank - 1), batch = 1;
    if (n < 1 || n > 512 || s.GetDim(rank - 2) != n) return ge::GRAPH_FAILED;
    for (size_t i = 0; i + 2 < rank; ++i) {
        if (s.GetDim(i) < 1 || s.GetDim(i) > 64 || batch > INT32_MAX / s.GetDim(i)) return ge::GRAPH_FAILED;
        batch *= s.GetDim(i);
    }
    const char *uplo = ctx->GetAttrs() ? ctx->GetAttrs()->GetStr(0) : nullptr;
    if (uplo && std::strcmp(uplo, "L") && std::strcmp(uplo, "U")) return ge::GRAPH_FAILED;
    t->n = n; t->batch = batch; t->stride = (n + 7) / 8 * 8;
    t->upper = uplo && uplo[0] == 'U';
    return ge::GRAPH_SUCCESS;
}
static uint32_t MinimumTeam(const LinalgEighTilingData *t, uint32_t cores, uint64_t ub) {
    const auto n = t->n;
    uint32_t minTeam = 1;
    while (minTeam <= cores) {
        uint32_t rows = (n + minTeam - 1) / minTeam;
        uint64_t dense = std::max<uint64_t>(uint64_t(rows) * t->stride,
            uint64_t(t->stride) * ((rows + 7) / 8 * 8));
        uint64_t wideN = n <= 25 ? n : 25;
        uint64_t wide = std::max<uint64_t>(3 * wideN * wideN + 21 * wideN + 64, 8 * t->stride);
        wide=std::max<uint64_t>(wide,((t->stride+1)/2)*((rows+7)/8*8));
        // Keep inverse iteration within one 64-float SIMD lane group.
        uint64_t color=n>25 && t->stride%64==0?1:0;
        uint64_t resident=3*dense+((dense+1)/2+7)/8*8+22*t->stride+64+(128+128*(t->stride%128==64))*color;
        if(n>25)resident=std::max<uint64_t>(resident,8192);
        uint64_t small=n>=3 && n<=25?15*n*t->stride:0;
        if ((n <= 25 || rows <= 64) && (resident+wide+128*color+small)*4<=ub) break;
        if (minTeam > cores / 2) return 0;
        minTeam *= 2;
    }
    if (minTeam > cores) return 0;
    return minTeam;
}
static ge::graphStatus ChooseTeams(LinalgEighTilingData *t, uint32_t cores, uint32_t minTeam) {
    const auto n = t->n;
    const auto batch = t->batch;
    if (minTeam == 0) return ge::GRAPH_FAILED;
    uint32_t groups = std::min<uint32_t>(batch, cores / minTeam);
    if (groups == 0) return ge::GRAPH_FAILED;
    uint32_t targetRows = n <= 25 ? 1 : 16;
    uint32_t team = std::max(minTeam, std::min<uint32_t>((n + targetRows - 1) / targetRows, cores / groups));
    if (team == 0) return ge::GRAPH_FAILED;
    t->rows = (n + team - 1) / team;
    t->team = (n + t->rows - 1) / t->rows;
    t->groups = std::min<uint32_t>(batch, cores / t->team);
    return ge::GRAPH_SUCCESS;
}
static uint32_t ChooseCube(LinalgEighTilingData *t, uint32_t minTeam) {
    const auto n = t->n;
    const uint32_t padded = (t->rows + 7) / 8 * 8;
    uint32_t USE_CUBE=n>25 && t->stride>=64 && t->stride%64==0 && padded>=16;
    if(USE_CUBE && t->stride>=384)USE_CUBE=2;
    if(USE_CUBE && t->team%2){
        // Keep the existing matrix groups and their shared bisection grid.
        // Round the team down to an even divisor; minTeam remains the UB bound.
        bool found=false;
        for(uint32_t requested=t->team-1;requested>=minTeam && requested>=2;requested-=2){
            uint32_t otherRows=(n+requested-1)/requested,otherTeam=(n+otherRows-1)/otherRows;
            if(otherTeam%2==0){t->rows=otherRows;t->team=otherTeam;found=true;break;}
        }
        if(!found)USE_CUBE=0;
    }
    return USE_CUBE;
}
static void AlignRows(LinalgEighTilingData *t, uint32_t cores, uint32_t minTeam, uint32_t USE_CUBE) {
    const auto n = t->n;
    if(n>25 && n%64==0 && t->rows%8){
        for(uint32_t alignedRows=(t->rows+7)/8*8;alignedRows<=64;alignedRows+=8){
            if(n%alignedRows)continue;
            uint32_t alignedTeam=uint32_t(n)/alignedRows;
            if(alignedTeam<minTeam || alignedTeam*t->groups>cores)continue;
            if(USE_CUBE && alignedTeam%2)continue;
            t->rows=alignedRows;t->team=alignedTeam;break;
        }
    }
}
static ge::graphStatus TilingFunc(gert::TilingContext *ctx) {
    auto *t = ctx->GetTilingData<LinalgEighTilingData>();
    if (!t || ReadShape(ctx, t) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    const auto n = t->n;
    const auto batch = t->batch;
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    uint32_t cores = platform.GetCoreNumAiv();
    if (!cores) return ge::GRAPH_FAILED;
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    // Keep space for compiler spills and library scratch in addition to TBufs.
    if (ub <= 16 * 1024) return ge::GRAPH_FAILED;
    ub -= 16 * 1024;
    uint32_t minTeam = MinimumTeam(t, cores, ub);
    if (!minTeam || ChooseTeams(t, cores, minTeam) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
    uint32_t USE_CUBE = ChooseCube(t, minTeam);
    AlignRows(t, cores, minTeam, USE_CUBE);
    uint32_t padded = (t->rows + 7) / 8 * 8;
    uint32_t physicalTeam = t->team;
    if(USE_CUBE)ctx->SetBlockDim(t->team/2*t->groups);
    else ctx->SetBlockDim(t->team*t->groups);
    uint32_t DT_A = static_cast<uint32_t>(ge::DT_FLOAT);
    ASCENDC_TPL_SEL_PARAM(ctx, DT_A, USE_CUBE);
    uint64_t perMatrix=uint64_t(n<=25?2*n+3:2*n+5)*t->stride;
    if(USE_CUBE){
        perMatrix+=uint64_t(n)*t->stride+uint64_t(physicalTeam)*padded*(t->stride+(USE_CUBE==1?16:64));
        if(USE_CUBE==2)perMatrix+=uint64_t((n-2+63)/64)*6*64*64;
    }
    ctx->GetWorkspaceSizes(1)[0]=uint64_t(batch)*perMatrix*sizeof(float)+platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *ctx) {
    const auto *s = ctx->GetInputShape(0);
    auto *values = ctx->GetOutputShape(0); auto *vectors = ctx->GetOutputShape(1);
    if (!s || !values || !vectors || s->GetDimNum() < 2) return GRAPH_FAILED;
    int64_t n = s->GetDim(s->GetDimNum() - 1);
    if (n < 1 || n > 512 || s->GetDim(s->GetDimNum() - 2) != n) return GRAPH_FAILED;
    const size_t rank = s->GetDimNum();
    *vectors = *s; values->SetDimNum(rank - 1);
    for (size_t i = 0; i + 1 < rank; ++i) values->SetDim(i, s->GetDim(i));
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *ctx) {
    if (ctx->GetInputDataType(0) != DT_FLOAT) return GRAPH_FAILED;
    ctx->SetOutputDataType(0, DT_FLOAT); ctx->SetOutputDataType(1, DT_FLOAT);
    return GRAPH_SUCCESS;
}
}
namespace ops {
class LinalgEigh : public OpDef {
public:
    explicit LinalgEigh(const char *name) : OpDef(name) {
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("eigenvalues").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("eigenvectors").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("UPLO").AttrType(OPTIONAL).String("L");
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};
OP_ADD(LinalgEigh);
}
