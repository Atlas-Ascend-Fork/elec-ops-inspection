/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>

#include "../op_kernel/fused_disconnector_keypoint_soft_argmax_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/tiling_key_fused_disconnector_keypoint_soft_argmax.h"

namespace optiling {

struct ParsedKeypointShapes {
    uint32_t batch, inChannels, outChannels;
    uint32_t kernelH, kernelW, height, width;
    uint32_t tempMode;
    float invTau, strideF;
};

struct PlatformCacheState {
    uint32_t numAiv;
    bool canSyncAll;
};

struct StripeRule {
    uint8_t widthCond;
    uint8_t mapsPerBlock;
    uint8_t stripes;
    uint8_t chPerDma;
    uint16_t accPitch;
    uint16_t maxPlane;
    uint64_t mode;
};

struct TilingDispatchOut {
    FusedDisconnectorKeypointSoftArgmaxTilingData* td;
    size_t userWsBytes;
    uint64_t modeCode;
};

static inline uint32_t SafeDivU(uint32_t num, uint32_t den)
{
    return (den > 0U) ? (num / den) : 0U;
}

static inline float SafeDivF(float num, float den)
{
    return (den > 0.0f) ? (num / den) : 0.0f;
}

template <typename PlatPtr>
static inline PlatformCacheState ReadPlatformUncached(PlatPtr platInfo)
{
    platform_ascendc::PlatformAscendC platform(platInfo);
    int32_t rawAiv = platform.GetCoreNumAiv();
    uint32_t numAiv = (rawAiv > 0) ? static_cast<uint32_t>(rawAiv) : 1U;
    uint64_t sysWs = platform.GetLibApiWorkSpaceSize();
    bool canSync = (sysWs > 0ULL) && (sysWs != 0xFFFFFFFFULL);
    return PlatformCacheState{numAiv, canSync};
}

static inline PlatformCacheState QueryPlatformFast(gert::TilingContext* context)
{
    static std::atomic<uint32_t> sCachedPlat{0U};
    auto platInfo = (context != nullptr) ? context->GetPlatformInfo() : nullptr;
    if (platInfo == nullptr) {
        return ReadPlatformUncached(nullptr);
    }
    uint32_t cached = sCachedPlat.load(std::memory_order_acquire);
    if (cached != 0U) {
        return PlatformCacheState{cached & 0xFFFFU, (cached & 0x10000U) != 0U};
    }
    PlatformCacheState state = ReadPlatformUncached(platInfo);
    uint32_t packed = (state.numAiv & 0xFFFFU) | (state.canSyncAll ? 0x10000U : 0U) | 0x80000000U;
    sCachedPlat.store(packed, std::memory_order_release);
    return state;
}

static inline void ComputeTempScale(float tauVal, uint32_t& tempMode, float& invTau)
{
    if (tauVal < std::ldexp(1.0f, -90)) {
        tempMode = 1U;
        invTau = SafeDivF(1.0f, std::ldexp(tauVal, 126));
    } else if (tauVal > std::ldexp(1.0f, 90)) {
        tempMode = 2U;
        invTau = SafeDivF(1.0f, std::ldexp(tauVal, -64));
    } else {
        tempMode = 0U;
        invTau = SafeDivF(1.0f, tauVal);
    }
}

static inline bool CheckInputDtypes(const gert::TilingContext* context)
{
    for (size_t idx = 0U; idx < 3U; ++idx) {
        const gert::CompileTimeTensorDesc* desc = context->GetInputDesc(idx);
        if (desc != nullptr && desc->GetDataType() != ge::DT_FLOAT) {
            return false;
        }
    }
    return true;
}

static inline bool ValidateTensorRanks(const gert::Shape& f, const gert::Shape& w, const gert::Shape& b)
{
    if (f.GetDimNum() != 4U || w.GetDimNum() != 4U || b.GetDimNum() != 1U) {
        return false;
    }
    if (f.GetDim(0) <= 0 || f.GetDim(1) <= 0 || f.GetDim(2) <= 0 || f.GetDim(3) <= 0) {
        return false;
    }
    if (w.GetDim(0) <= 0 || w.GetDim(2) <= 0 || w.GetDim(3) <= 0) {
        return false;
    }
    return (w.GetDim(1) == f.GetDim(1)) && (b.GetDim(0) == w.GetDim(0));
}

static inline bool ExtractValidShapes(const gert::TilingContext* context, ParsedKeypointShapes& out)
{
    const gert::StorageShape* featS = context->GetInputShape(0);
    const gert::StorageShape* wS = context->GetInputShape(1);
    const gert::StorageShape* bS = context->GetInputShape(2);
    if (featS == nullptr || wS == nullptr || bS == nullptr) {
        return false;
    }
    const gert::Shape& f = featS->GetStorageShape();
    const gert::Shape& w = wS->GetStorageShape();
    const gert::Shape& b = bS->GetStorageShape();
    if (!ValidateTensorRanks(f, w, b)) {
        return false;
    }
    out = ParsedKeypointShapes{
        static_cast<uint32_t>(f.GetDim(0)), static_cast<uint32_t>(f.GetDim(1)),
        static_cast<uint32_t>(w.GetDim(0)), static_cast<uint32_t>(w.GetDim(2)),
        static_cast<uint32_t>(w.GetDim(3)), static_cast<uint32_t>(f.GetDim(2)),
        static_cast<uint32_t>(f.GetDim(3)), 0U, 0.0f, 0.0f};
    return true;
}

static inline bool ExtractValidAttrs(const gert::RuntimeAttrs* attrs, ParsedKeypointShapes& out)
{
    const float* tauPtr = attrs->GetFloat(0);
    const int64_t* stridePtr = attrs->GetInt(1);
    if (tauPtr == nullptr || stridePtr == nullptr) {
        return false;
    }
    if (*tauPtr <= 0.0f || !std::isfinite(*tauPtr) || *stridePtr <= 0) {
        return false;
    }
    ComputeTempScale(*tauPtr, out.tempMode, out.invTau);
    out.strideF = static_cast<float>(*stridePtr);
    return true;
}

static inline bool ParseShapesAndAttrs(const gert::TilingContext* context, ParsedKeypointShapes& out)
{
    if (context == nullptr || context->GetAttrs() == nullptr) {
        return false;
    }
    if (!CheckInputDtypes(context)) {
        return false;
    }
    if (!ExtractValidShapes(context, out)) {
        return false;
    }
    return ExtractValidAttrs(context->GetAttrs(), out);
}

static inline uint32_t CalcStripeFastUbBytes(
    uint32_t mapsPerBlock, uint32_t accPitch, uint32_t chPerDma, uint32_t maxRows, uint32_t width)
{
    const uint32_t accTotalFloats = (9U * mapsPerBlock * accPitch + 191U) & ~127U;
    const uint32_t viewElems = (maxRows + 2U) * width;
    const uint32_t rem128 = viewElems & 127U;
    const uint32_t dstGapBlks = (rem128 == 0U) ? 16U : ((128U - rem128) >> 3U);
    const uint32_t centerStride = viewElems + dstGapBlks * 8U;
    const uint32_t slotFloats = (chPerDma * centerStride + 64U + 127U) & ~127U;
    const uint32_t inFloats = std::max(2U * slotFloats, 4096U);
    const uint32_t offFloats = 2U * ((maxRows * width + 15U) & ~15U);
    const uint32_t fixedFloats = 256U + 1168U + 160U;
    return 1024U + (accTotalFloats + inFloats + offFloats + fixedFloats) * 4U;
}

static inline bool CanFitStripeCandidate(
    const ParsedKeypointShapes& s, uint32_t numAiv, const StripeRule& rule)
{
    const uint32_t mapsPerBlock = rule.mapsPerBlock;
    const uint32_t stripes = rule.stripes;
    if (mapsPerBlock == 0U || stripes == 0U) {
        return false;
    }
    const uint32_t blocks = SafeDivU(s.batch * 4U, mapsPerBlock) * stripes;
    if (blocks > numAiv || s.height < 2U * stripes || s.height > 128U) {
        return false;
    }
    const uint32_t minRows = SafeDivU(s.height, stripes);
    const uint32_t maxRows = SafeDivU(s.height + stripes - 1U, stripes);
    const uint32_t reqPitch = ((maxRows * s.width + 63U) & ~63U) + 8U;
    if (minRows < 2U || maxRows > 64U || reqPitch > rule.accPitch) {
        return false;
    }
    const uint32_t ubBytes = CalcStripeFastUbBytes(
        mapsPerBlock, rule.accPitch, rule.chPerDma, maxRows, s.width);
    return ubBytes <= 192000U;
}

static inline bool MatchRuleCond(const ParsedKeypointShapes& s, const StripeRule& rule)
{
    const bool wMul64 = ((s.width & 63U) == 0U);
    if (rule.widthCond == 1U && !wMul64) {
        return false;
    }
    if (rule.widthCond == 2U && wMul64) {
        return false;
    }
    if (rule.maxPlane > 0U && (s.height * s.width) > rule.maxPlane) {
        return false;
    }
    return true;
}

static inline bool MatchStripeTable(
    const ParsedKeypointShapes& s, uint32_t numAiv,
    const StripeRule* rules, size_t count, StripeRule& matched)
{
    for (size_t i = 0U; i < count; ++i) {
        if (MatchRuleCond(s, rules[i]) && CanFitStripeCandidate(s, numAiv, rules[i])) {
            matched = rules[i];
            return true;
        }
    }
    return false;
}

static inline bool SelectStripeFastCandidate(
    const ParsedKeypointShapes& s, uint32_t numAiv, StripeRule& matched)
{
    static constexpr StripeRule kBatch1Rules[] = {
        {1U, 1U, 5U, 8U, 1536U, 4096U, 5ULL},
        {0U, 1U, 5U, 8U, 1536U, 4096U, 6ULL},
        {2U, 1U, 9U, 8U, 896U, 0U, 14ULL},
        {1U, 1U, 10U, 4U, 1792U, 0U, 15ULL},
        {1U, 1U, 5U, 8U, 1536U, 0U, 5ULL},
        {0U, 1U, 5U, 8U, 1536U, 0U, 6ULL},
    };
    static constexpr StripeRule kBatch2Rules[] = {
        {1U, 1U, 4U, 8U, 1536U, 0U, 10ULL},
        {0U, 1U, 4U, 4U, 1792U, 0U, 12ULL},
        {1U, 1U, 5U, 1U, 3456U, 0U, 16ULL},
        {0U, 2U, 5U, 4U, 1536U, 0U, 8ULL},
    };
    static constexpr StripeRule kBatch3Rules[] = {
        {1U, 1U, 3U, 8U, 1536U, 0U, 11ULL},
        {0U, 1U, 3U, 2U, 2432U, 0U, 7ULL},
    };
    static constexpr StripeRule kBatch4Rules[] = {
        {1U, 2U, 5U, 8U, 896U, 0U, 9ULL},
        {0U, 2U, 5U, 4U, 1536U, 0U, 8ULL},
    };
    if (s.batch == 1U) {
        return MatchStripeTable(s, numAiv, kBatch1Rules, 6U, matched);
    }
    if (s.batch == 2U) {
        return MatchStripeTable(s, numAiv, kBatch2Rules, 4U, matched);
    }
    if (s.batch == 3U) {
        return MatchStripeTable(s, numAiv, kBatch3Rules, 2U, matched);
    }
    return MatchStripeTable(s, numAiv, kBatch4Rules, 2U, matched);
}

static inline bool CanUsePlaneFast(const ParsedKeypointShapes& s, uint32_t numAiv)
{
    if (s.batch < 1U || s.batch > 10U || (s.batch * 4U) > numAiv) {
        return false;
    }
    if (s.height < 2U || s.height > 64U || (s.width & 7U) != 0U) {
        return false;
    }
    if (s.width < 8U || s.width > 128U) {
        return false;
    }
    return (s.height * s.width) <= 3072U;
}

static inline bool TryFillPlaneFastTiling(
    gert::TilingContext* context, const ParsedKeypointShapes& s,
    const PlatformCacheState& plat, TilingDispatchOut& out)
{
    if (!CanUsePlaneFast(s, plat.numAiv)) {
        return false;
    }
    (void)context->SetScheduleMode(0);
    const uint32_t maps = s.batch * 4U;
    const bool useTileExclusive = (2U * maps <= plat.numAiv);
    out.td->coresPerBatch = 4U;
    out.td->activeCores = useTileExclusive ? (2U * maps) : maps;
    out.td->maxStripRows = s.height;
    out.td->flags = useTileExclusive ? 34U : 32U;
    out.userWsBytes = 0ULL;
    out.modeCode = 13ULL;
    return true;
}

static inline bool CanAttemptStripeFast(const ParsedKeypointShapes& s, const PlatformCacheState& plat)
{
    if (!plat.canSyncAll || plat.numAiv < 20U) {
        return false;
    }
    if (s.batch < 1U || s.batch > 4U) {
        return false;
    }
    return ((s.width & 7U) == 0U) && (s.width >= 8U) && (s.width <= 128U);
}

static inline bool TryFillStripeFastTiling(
    gert::TilingContext* context, const ParsedKeypointShapes& s,
    const PlatformCacheState& plat, TilingDispatchOut& out)
{
    if (CanAttemptStripeFast(s, plat)) {
        StripeRule matched{};
        if (SelectStripeFastCandidate(s, plat.numAiv, matched) &&
            context->SetScheduleMode(1) == ge::GRAPH_SUCCESS) {
            const uint32_t mapsPerBlock = matched.mapsPerBlock;
            const uint32_t stripes = matched.stripes;
            const uint32_t blocks = SafeDivU(s.batch * 4U, mapsPerBlock) * stripes;
            out.td->coresPerBatch = stripes;
            out.td->activeCores = blocks;
            out.td->maxStripRows = SafeDivU(s.height + stripes - 1U, stripes);
            out.td->flags = 16U | mapsPerBlock;
            out.userWsBytes = static_cast<size_t>(blocks) * 32ULL * sizeof(float);
            out.modeCode = matched.mode;
            return true;
        }
    }
    return TryFillPlaneFastTiling(context, s, plat, out);
}

static inline uint32_t ClampActiveCores(uint64_t totalPlanes64, uint32_t numAiv)
{
    uint32_t totalPlanes = (totalPlanes64 > 0xFFFFFFFFULL)
        ? 0xFFFFFFFFU
        : static_cast<uint32_t>(totalPlanes64);
    uint32_t activeCores = std::min(totalPlanes, numAiv);
    return (activeCores == 0U) ? 1U : activeCores;
}

static inline void FillMode4Tiling(
    gert::TilingContext* context, const ParsedKeypointShapes& s,
    uint32_t numAiv, TilingDispatchOut& out)
{
    (void)context->SetScheduleMode(0);
    uint32_t wPad = (s.width + 7U) & ~7U;
    uint32_t alignFlag = ((s.width & 7U) == 0U) ? 1U : 0U;
    uint32_t capInRows = std::min(SafeDivU(4096U, wPad), 64U);
    uint32_t maxR = 1U;
    if (s.height <= capInRows) {
        maxR = s.height;
    } else if (capInRows >= 2U && s.height <= 2U * (capInRows - 1U)) {
        maxR = (s.height + 1U) >> 1U;
    } else if (capInRows > 2U) {
        maxR = capInRows - 2U;
    }
    out.td->coresPerBatch = 4U;
    out.td->activeCores = ClampActiveCores(static_cast<uint64_t>(s.batch) * 4ULL, numAiv);
    out.td->maxStripRows = maxR;
    out.td->flags = alignFlag | 8U;
    out.userWsBytes = 0ULL;
    out.modeCode = 4ULL;
}

static inline void FillMode2Tiling(
    gert::TilingContext* context, const ParsedKeypointShapes& s,
    uint32_t numAiv, TilingDispatchOut& out)
{
    (void)context->SetScheduleMode(0);
    uint64_t totalPlanes64 = static_cast<uint64_t>(s.batch) * s.outChannels;
    out.td->activeCores = ClampActiveCores(totalPlanes64, numAiv);
    out.td->coresPerBatch = 1U;
    out.td->maxStripRows = 1U;
    out.td->flags = 4U;
    out.userWsBytes = 0ULL;
    out.modeCode = 2ULL;
}

static inline void PopulateBaseTilingFields(
    const ParsedKeypointShapes& s, FusedDisconnectorKeypointSoftArgmaxTilingData* td)
{
    td->batch = s.batch;
    td->inChannels = s.inChannels;
    td->outChannels = s.outChannels;
    td->kernelH = s.kernelH;
    td->kernelW = s.kernelW;
    td->height = s.height;
    td->width = s.width;
    td->wPad = (s.width + 7U) & ~7U;
    td->invTau = s.invTau;
    td->strideF = s.strideF;
    td->halfStrideF = 0.5f * s.strideF;
    td->tempMode = s.tempMode;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ParsedKeypointShapes s{};
    if (!ParseShapesAndAttrs(context, s)) {
        return ge::GRAPH_FAILED;
    }
    auto* td = context->GetTilingData<FusedDisconnectorKeypointSoftArgmaxTilingData>();
    if (td == nullptr) {
        return ge::GRAPH_FAILED;
    }
    PopulateBaseTilingFields(s, td);
    PlatformCacheState plat = QueryPlatformFast(context);
    TilingDispatchOut out{td, 0ULL, 2ULL};
    bool isConv3x3 = (s.inChannels == 32U) && (s.outChannels == 4U) &&
                     (s.kernelH == 3U) && (s.kernelW == 3U);
    if (!isConv3x3 || !TryFillStripeFastTiling(context, s, plat, out)) {
        if (isConv3x3 && s.width >= 1U && s.width <= 256U) {
            FillMode4Tiling(context, s, plat.numAiv, out);
        } else {
            FillMode2Tiling(context, s, plat.numAiv, out);
        }
    }
    context->SetBlockDim(td->activeCores);
    uint32_t dt = static_cast<uint32_t>(ge::DT_FLOAT);
    ASCENDC_TPL_SEL_PARAM(context, dt, out.modeCode);
    context->GetRawTilingData()->SetDataSize(sizeof(FusedDisconnectorKeypointSoftArgmaxTilingData));
    size_t* wsSizes = context->GetWorkspaceSizes(1U);
    if (wsSizes != nullptr) {
        wsSizes[0] = out.userWsBytes;
    }
    return ge::GRAPH_SUCCESS;
}

}

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape* feat = context->GetInputShape(0);
    const gert::Shape* weight = context->GetInputShape(1);
    gert::Shape* kpShape = context->GetOutputShape(0);
    gert::Shape* scShape = context->GetOutputShape(1);
    if (feat == nullptr || weight == nullptr || kpShape == nullptr || scShape == nullptr) {
        return GRAPH_FAILED;
    }
    int64_t b = feat->GetDim(0);
    int64_t k = weight->GetDim(0);
    kpShape->SetDimNum(3U);
    kpShape->SetDim(0U, b);
    kpShape->SetDim(1U, k);
    kpShape->SetDim(2U, 2);
    scShape->SetDimNum(2U);
    scShape->SetDim(0U, b);
    scShape->SetDim(1U, k);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    DataType dt = context->GetInputDataType(0);
    if (dt != DT_FLOAT) {
        dt = DT_FLOAT;
    }
    context->SetOutputDataType(0U, dt);
    context->SetOutputDataType(1U, dt);
    return GRAPH_SUCCESS;
}

}

namespace ops {

class FusedDisconnectorKeypointSoftArgmax : public OpDef {
public:
    explicit FusedDisconnectorKeypointSoftArgmax(const char* name) : OpDef(name)
    {
        this->Input("feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("keypoints")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("scores")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("tau").AttrType(REQUIRED).Float();
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(FusedDisconnectorKeypointSoftArgmax);

}
