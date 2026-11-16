#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/linalg_solve_tiling.h"
#include "../op_kernel/tiling_key_linalg_solve.h"
#include <algorithm>
#include <cstdint>

namespace {
struct ShapePlan {
    uint32_t n = 0, k = 1, rank = 0, batches = 1;
    bool vector = false;
    uint32_t dims[8] = {}, as[8] = {}, bs[8] = {};
};
bool Plan(const gert::Shape &a, const gert::Shape &b, ShapePlan &p) {
    const int ar = a.GetDimNum(), br = b.GetDimNum();
    if (ar < 2 || br < 1 || ar > 10 || br > 10) return false;
    const int64_t n = a.GetDim(ar - 1);
    if (n < 1 || n > 512 || a.GetDim(ar - 2) != n) return false;
    p.n = n;
    // The statement explicitly broadcasts an unbatched (n,k) matrix RHS.
    const bool matrix = br >= 2 && b.GetDim(br - 2) == n;
    p.vector = !matrix;
    const int ab = ar - 2, bb = br - (matrix ? 2 : 1);
    if (!matrix && b.GetDim(br - 1) != n) return false;
    if (matrix) {
        const int64_t k = b.GetDim(br - 1);
        if (k < 1 || k > 256) return false;
        p.k = k;
    }
    p.rank = std::max(ab, bb);
    if (p.rank > 8) return false;
    uint64_t ac = 1, bc = 1, total = 1;
    for (int d = static_cast<int>(p.rank) - 1; d >= 0; --d) {
        const int ai = d - (static_cast<int>(p.rank) - ab);
        const int bi = d - (static_cast<int>(p.rank) - bb);
        const int64_t ad = ai < 0 ? 1 : a.GetDim(ai);
        const int64_t bd = bi < 0 ? 1 : b.GetDim(bi);
        if (ad < 1 || bd < 1 || (ad != bd && ad != 1 && bd != 1)) return false;
        p.dims[d] = std::max(ad, bd);
        p.as[d] = ad == 1 ? 0 : ac;
        p.bs[d] = bd == 1 ? 0 : bc;
        ac *= ad; bc *= bd; total *= p.dims[d];
        if (ac > UINT32_MAX || bc > UINT32_MAX || total > UINT32_MAX) return false;
    }
    p.batches = total;
    return true;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto *a = context->GetInputShape(0);
    const auto *b = context->GetInputShape(1);
    const auto *attrs = context->GetAttrs();
    const bool *left = attrs ? attrs->GetBool(0) : nullptr;
    if ((left && !*left) || !a || !b ||
        context->GetInputDesc(0)->GetDataType() != ge::DT_FLOAT ||
        context->GetInputDesc(1)->GetDataType() != ge::DT_FLOAT) return ge::GRAPH_FAILED;
    ShapePlan p;
    if (!Plan(a->GetStorageShape(), b->GetStorageShape(), p)) return ge::GRAPH_FAILED;
    auto *t = context->GetTilingData<LinalgSolveTilingData>();
    t->n = p.n; t->k = p.k;
    t->nPad = (p.n + 7) / 8 * 8; t->kPad = (p.k + 7) / 8 * 8;
    t->rowPitch = t->nPad + t->kPad;
    t->dotSize = 0;
    if (t->k <= 4 && t->n >= 8) { t->dotSize = 8; while (t->dotSize < t->n) t->dotSize *= 2; }
    t->batches = p.batches;
    t->paired = p.n == 256 && p.k <= 4 && 2 * p.batches <= platform.GetCoreNumAiv();
    t->rhsSplit = (p.n == 50 || p.n == 64) && p.k <= 4 && p.batches * p.k <= platform.GetCoreNumAiv();
    t->blocked = p.n == 200 && p.k <= 4 && p.batches <= platform.GetCoreNumAic();
    if (t->blocked && p.batches * p.k <= platform.GetCoreNumAic()) t->rhsSplit = true;
    if(t->paired && p.k>=2 && 2*p.batches*p.k<=platform.GetCoreNumAiv() && p.batches*p.k<=platform.GetCoreNumAic())t->rhsSplit=true;
    t->blocks = t->paired ? 2 * p.batches * (t->rhsSplit?p.k:1) : t->rhsSplit ? p.batches * p.k : std::min(p.batches, static_cast<uint32_t>(platform.GetCoreNumAiv()));
    t->originalSchur=p.n==256 && p.k==3 && 4*p.batches<=platform.GetCoreNumAic() && 8*p.batches<=platform.GetCoreNumAiv();
    t->pairsPerMatrix=t->originalSchur?4:(t->paired && t->rhsSplit?p.k:1);t->schurOffset=0;
    if(t->originalSchur){t->paired=true;t->rhsSplit=true;t->blocks=8*p.batches;}
    if (t->paired || t->blocked) context->SetScheduleMode(1);
    t->batchRank = p.rank;
    for (int i = 0; i < 8; ++i) {
        t->batchDims[i] = p.dims[i]; t->aStrides[i] = p.as[i]; t->bStrides[i] = p.bs[i];
    }
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const uint64_t storage = uint64_t(t->n) * (t->nPad + 2 * t->kPad) * sizeof(float);
    const uint64_t scratchFloats = std::max(uint64_t(2 * t->rowPitch + 9 * t->nPad + 8 * t->kPad), uint64_t(17) * t->dotSize);
    const uint64_t scratch = (scratchFloats + t->nPad) * sizeof(float);
    t->onChip = storage + scratch + 4096 <= ub;
    t->systemWorkspace = platform.GetLibApiWorkSpaceSize();
    const bool residualSpill = t->dotSize && t->n >= 128 && t->n <= 256;
    context->GetWorkspaceSizes(1)[0] = t->systemWorkspace + ((!t->onChip || residualSpill) ? storage * t->blocks : 0);
    if (t->blocked) context->GetWorkspaceSizes(1)[0] += uint64_t(t->blocks) * 200 * 208 * 4;
    if(t->originalSchur){t->schurOffset=context->GetWorkspaceSizes(1)[0]-t->systemWorkspace;context->GetWorkspaceSizes(1)[0]+=uint64_t(t->batches)*kOS_Total*4;}
    const bool pairCube=t->paired && t->blocks/2<=platform.GetCoreNumAic();
    context->SetBlockDim(pairCube?t->blocks/2:t->blocks);
    uint32_t DT_A = static_cast<uint32_t>(ge::DT_FLOAT);
    uint32_t MODE = t->originalSchur ? 4 : pairCube ? 3 : t->blocked ? 2 : t->paired ? 1 : 0;
    if(!p.vector && p.k==1 && p.rank==1 && p.as[0]==1 && p.bs[0]==1){
        if(p.n==16 && p.batches>=2 && p.batches<=platform.GetCoreNumAiv())MODE=10;
        else if(p.n==15 && p.batches>=2 && p.batches<=platform.GetCoreNumAiv())MODE=6;
    }
    if(!p.vector && p.k==2 && p.rank==1 && p.as[0]==1 && p.bs[0]==1 && t->rhsSplit){
        if(p.n==64 && p.batches>=2 && p.batches<=platform.GetCoreNumAiv()/2){MODE=13;context->SetBlockDim(p.batches);}
        else if(p.n==50 && p.batches>=2){MODE=12;context->SetBlockDim(p.batches);}
    }
    if(!p.vector && p.n==200 && p.k==4 && p.rank==1 && p.as[0]==1 && p.bs[0]==1 && p.batches>=2 && 4*p.batches<=platform.GetCoreNumAiv()){
        MODE=14;t->blocked=true;t->paired=false;t->rhsSplit=true;t->blocks=4*p.batches;
        context->SetBlockDim(t->blocks);
        context->GetWorkspaceSizes(1)[0]=t->systemWorkspace+uint64_t(t->blocks)*200*216*4;
        if(4*p.batches<=platform.GetCoreNumAic()){
            MODE=15;
            context->GetWorkspaceSizes(1)[0]+=uint64_t(t->blocks)*200*208*4;
        }
    }
    ASCENDC_TPL_SEL_PARAM(context, DT_A, MODE);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const auto *a = context->GetInputShape(0);
    const auto *b = context->GetInputShape(1);
    auto *out = context->GetOutputShape(0);
    if (!a || !b || !out) return GRAPH_FAILED;
    ShapePlan p;
    if (!Plan(*a, *b, p)) return GRAPH_FAILED;
    out->SetDimNum(p.rank + (p.vector ? 1 : 2));
    for (uint32_t i = 0; i < p.rank; ++i) out->SetDim(i, p.dims[i]);
    out->SetDim(p.rank, p.n);
    if (!p.vector) out->SetDim(p.rank + 1, p.k);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context->GetInputDataType(0) != DT_FLOAT || context->GetInputDataType(1) != DT_FLOAT)
        return GRAPH_FAILED;
    context->SetOutputDataType(0, DT_FLOAT);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class LinalgSolve : public OpDef {
public:
    explicit LinalgSolve(const char *name) : OpDef(name) {
        this->Input("A").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("B").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("X").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("left").AttrType(OPTIONAL).Bool(true);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(LinalgSolve);
} // namespace ops
