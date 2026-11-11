// Copyright 2026 Delicate02
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "securec.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/safety_belt_violation_geometry_tiling.h"
#include "../op_kernel/tiling_key_safety_belt_violation_geometry.h"
namespace optiling {
static ge::graphStatus ValidateGeometryShapes(gert::TilingContext *context) {
    const auto *boxes=context->GetInputShape(0);
    const auto *anchors=context->GetInputShape(1);
    if (!boxes || !anchors) return ge::GRAPH_FAILED;
    const auto &b=boxes->GetStorageShape();
    const auto &a=anchors->GetStorageShape();
    if(b.GetDimNum()!=2 || a.GetDimNum()!=2 || b.GetDim(1)!=4 || a.GetDim(1)!=2 || b.GetDim(0)!=a.GetDim(0) || b.GetDim(0)<1 || b.GetDim(0)>kGeometryMaxBatch) return ge::GRAPH_FAILED;
    return ge::GRAPH_SUCCESS;
}

// Each planning stage owns one structural decision and preserves the
// original ordering of resource refinements and launch-budget updates.
class GeometryTilingPlan {
public:
    explicit GeometryTilingPlan(gert::TilingContext *ctx)
        : context(ctx), t(ctx->GetTilingData<SafetyBeltViolationGeometryTilingData>()),
          platform(ctx->GetPlatformInfo()) {}

    ge::graphStatus Run() {
        if (RunPlanningGroup1() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (RunPlanningGroup2() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (RunPlanningGroup3() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        PlanRepeatAddress();
        return Serialize();
    }

private:
    ge::graphStatus RunPlanningGroup1() {
        return PlanTileWidth() == ge::GRAPH_SUCCESS &&
               PlanIdentityOwners() == ge::GRAPH_SUCCESS &&
               PlanLaunchBudget() == ge::GRAPH_SUCCESS &&
               PlanInputSignature() == ge::GRAPH_SUCCESS &&
               PlanInitialResourceClass() == ge::GRAPH_SUCCESS &&
               PlanFoldedFloatOwners() == ge::GRAPH_SUCCESS
            ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
    }

    ge::graphStatus RunPlanningGroup2() {
        return PlanSmallHalfAndFloatTiles() == ge::GRAPH_SUCCESS &&
               PlanFloatAndFoldedCoefficients() == ge::GRAPH_SUCCESS &&
               PlanCompleteCoefficients() == ge::GRAPH_SUCCESS &&
               PlanAddressAndBroadcast() == ge::GRAPH_SUCCESS &&
               PlanWideBroadcast() == ge::GRAPH_SUCCESS &&
               PlanFoldedHalfMetadata() == ge::GRAPH_SUCCESS
            ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
    }

    ge::graphStatus RunPlanningGroup3() {
        return PlanFloatMetadata() == ge::GRAPH_SUCCESS &&
               PlanHalfEntryResources() == ge::GRAPH_SUCCESS &&
               PlanFloatEntryResources() == ge::GRAPH_SUCCESS &&
               PlanFoldedAndGenericEntries() == ge::GRAPH_SUCCESS &&
               PlanShortCoefficients() == ge::GRAPH_SUCCESS &&
               PlanMergedBatchOwners() == ge::GRAPH_SUCCESS
            ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
    }

    ge::graphStatus PlanTileWidth() {
        t->batch=context->GetInputShape(0)->GetStorageShape().GetDim(0);
        t->waistRatio=*context->GetAttrs()->GetFloat(0);
        t->hasThreshold=context->GetOptionalInputShape(2)!=nullptr;
        t->thresholdIsHalf=t->hasThreshold && context->GetInputDesc(2)->GetDataType()==ge::DT_FLOAT16;
        t->defaultThreshold=0.0f;
        tile=256;
        if(t->batch<=512 && (t->batch+63)/64<=platform.GetCoreNumAiv())tile=64;
        wideWorkers=(t->batch+511)/512;
        wideTail=t->batch%512;
        halfInput=context->GetInputDesc(0)->GetDataType()==ge::DT_FLOAT16;
        if((halfInput && wideWorkers>=4 && (wideTail==0 || wideTail>256)) ||
        (!halfInput && wideWorkers>=kGeometryFloatWideOwners && (wideTail==0 || wideTail>256)))tile=512;
        if(!halfInput && t->batch<=64 && platform.GetCoreNumAiv()>=2)tile=32;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanIdentityOwners() {
        floatUnitTiles=!halfInput && t->batch<=64 &&
        context->GetOutputDesc(0)->GetDataType()==ge::DT_FLOAT;
        identityTiling=(halfInput || floatUnitTiles) && t->waistRatio==1.0f && t->batch%32==0 &&
        t->batch/32<=platform.GetCoreNumAiv();
        if(identityTiling)tile=32;
        cores=(t->batch+tile-1)/tile;
        if(cores>platform.GetCoreNumAiv()) cores=platform.GetCoreNumAiv();
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanLaunchBudget() {
        if(cores==0U) return ge::GRAPH_FAILED;
        t->perCore=tile;
        if((t->batch+tile-1)/tile>cores)t->perCore=((t->batch+cores-1)/cores+15)/16*16;
        workers=(t->batch+t->perCore-1)/t->perCore;
        launchCores=workers;
        if(tile==64 || tile==32){
        launchCores=8;
        if(t->batch>64 && t->batch%64==0)launchCores=workers;
        if(launchCores<workers)launchCores=workers;
        if(launchCores>platform.GetCoreNumAiv())launchCores=platform.GetCoreNumAiv();
        }
        if(identityTiling){
        launchCores=workers<8?8:workers;
        if(launchCores>platform.GetCoreNumAiv())launchCores=platform.GetCoreNumAiv();
        }
        context->SetBlockDim(launchCores);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanInputSignature() {
        dtype=static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
        outdtype=static_cast<uint32_t>(context->GetOutputDesc(0)->GetDataType());
        thresholdKind=t->hasThreshold?(t->thresholdIsHalf?2:1):0;
        nativeSignature=dtype==outdtype &&
        ((!halfInput && thresholdKind==1U) || (halfInput && thresholdKind==2U));
        full64=t->batch/64U;
        short64=t->batch%64U;
        runtimeFold=!halfInput && nativeSignature && t->waistRatio!=0.5f &&
        tile==64U && t->perCore==64U && full64>=1U && full64<=7U &&
        short64>=1U && short64<=8U && platform.GetCoreNumAiv()>=8U;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanInitialResourceClass() {
        tileClass=tile==32?23+2*(workers-1)+(t->batch%32!=0):tile==512 && t->perCore==512?(t->batch%512==0?19:20):tile==64?3+2*(workers-1)+(t->batch%64!=0):t->perCore>256?0:(t->batch%256==0?1:2);
        if(tile==64 && t->perCore==64 && t->batch%64==0 && workers>1)tileClass=5;
        if(identityTiling)tileClass=workers<=8?26+workers:35;
        outputBlockRows=halfInput?16U:8U;
        if((!halfInput || dtype==outdtype) && tile==512 && t->perCore==512 &&
        wideTail>512U-outputBlockRows)
        tileClass=36+wideTail-449;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanFoldedFloatOwners() {
        if(tile==64 && t->perCore==64 && workers>1 && t->batch%64>0 && t->batch%64<=8 &&
        (halfInput || t->waistRatio==0.5f || runtimeFold))
        tileClass=99+(workers-2)*8+(t->batch%64-1);
        if(!halfInput && dtype==outdtype && tileClass>=99 && tileClass<=154){
        const uint32_t foldedWorkers=(tileClass-99)/8+1;
        uint32_t foldedLaunch=foldedWorkers<kGeometryFoldMinLaunch?kGeometryFoldMinLaunch:foldedWorkers;
        if(foldedLaunch>platform.GetCoreNumAiv())foldedLaunch=platform.GetCoreNumAiv();
        context->SetBlockDim(foldedLaunch);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanSmallHalfAndFloatTiles() {
        if(halfInput && dtype==outdtype && tile==64 && t->perCore==64 &&
        t->batch>128 && t->batch%128>0 && t->batch%128<=8)
        tileClass=155+(t->batch/128-1)*8+(t->batch%128-1);
        if(halfInput && dtype==outdtype && tile==64 && t->perCore==64 &&
        workers==1 && t->batch<64)
        tileClass=179+t->batch-1;
        if(!halfInput && dtype==outdtype && tile==32 && t->perCore==32 &&
        t->batch<64 && t->batch%32!=0){
        tileClass=179+t->batch-1;
        t->perCore=64;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanFloatAndFoldedCoefficients() {
        if(!halfInput && dtype==outdtype && thresholdKind==1U &&
        ((tileClass>=36U && tileClass<=98U) || tileClass==19U) && t->waistRatio==0.8f)thresholdKind=3;
        if(halfInput && dtype==outdtype && thresholdKind==2U &&
        tileClass>=155U && tileClass<=178U && t->waistRatio==0.5f)thresholdKind=3;
        if(!halfInput && dtype==outdtype && thresholdKind==1U &&
        tileClass>=179U && tileClass<=241U && t->waistRatio==0.8f)thresholdKind=3;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanCompleteCoefficients() {
        if(halfInput && dtype==outdtype && thresholdKind==2U &&
        (tileClass==19U || (tileClass>=36U && tileClass<=98U)) &&
        t->waistRatio==0.8f)thresholdKind=3;
        if(halfInput && dtype==outdtype && thresholdKind==2U && t->waistRatio==0.9f &&
        tileClass>=3U && tileClass<=17U && (tileClass&1U)!=0U)thresholdKind=3U;
        if(!halfInput && dtype==outdtype && thresholdKind==1U && tileClass==5U &&
        t->waistRatio==0.9f)thresholdKind=3U;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanAddressAndBroadcast() {
        addressPlan=!halfInput && dtype==outdtype && thresholdKind==1U &&
        tileClass>=27U && tileClass<=35U?1U:0U;
        broadcastPlan=halfInput && dtype==outdtype && (thresholdKind==2U || thresholdKind==3U) &&
        tileClass>=3U && tileClass<=17U && (tileClass&1U)!=0U?1U:0U;
        if(!halfInput && dtype==outdtype && thresholdKind==3U && tileClass==19U)
        broadcastPlan=1U;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanWideBroadcast() {
        if(halfInput && dtype==outdtype && thresholdKind==3U && tileClass==19U)
        broadcastPlan=1U;
        if(!halfInput && dtype==outdtype && thresholdKind==3U && tileClass>=92U && tileClass<=98U)
        broadcastPlan=1U;
        if(!halfInput && dtype==outdtype && thresholdKind==3U && tileClass>=92U && tileClass<=98U)
        addressPlan=1U;
        if(halfInput && dtype==outdtype && thresholdKind==3U && tileClass>=84U && tileClass<=98U)
        broadcastPlan=1U;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanFoldedHalfMetadata() {
        metadataPayload=context->GetRawTilingData();
        floatMetadataPlan=!halfInput && dtype==outdtype && thresholdKind==1U &&
        tileClass>=3U && tileClass<=17U && (tileClass&1U)!=0U &&
        metadataPayload->GetCapacity()>=32U?1U:0U;
        folded256=halfInput && dtype==outdtype && thresholdKind==3U &&
        tileClass>=155U && tileClass<=178U && t->batch>256U &&
        t->batch%256U>0U && t->batch%256U<=8U;
        if(folded256){tileClass=155U+t->batch%256U-1U;addressPlan=1U;}
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanFloatMetadata() {
        selectScalarPitchPlan=!halfInput && dtype==outdtype && thresholdKind==3U &&
        tileClass>=92U && tileClass<=98U && addressPlan==1U && broadcastPlan==1U;
        if(selectScalarPitchPlan)floatMetadataPlan=3U;
        if(!halfInput && dtype==outdtype && thresholdKind==3U && tileClass==19U && addressPlan==0U && broadcastPlan==1U)
        floatMetadataPlan=2U;
        foldedRawEntry=!halfInput && dtype==outdtype && thresholdKind==1U &&
        tileClass>=99U+(kGeometryFoldMinLaunch-1U)*8U && tileClass<=154U;
        if(foldedRawEntry){
        tileClass=99U+(tileClass-99U)%8U;
        addressPlan=1U;broadcastPlan=1U;floatMetadataPlan=3U;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanHalfEntryResources() {
        splitHalfEntryV2=halfInput && dtype==outdtype && thresholdKind==3U &&
        tileClass>=84U && tileClass<=98U && addressPlan==0U && broadcastPlan==1U;
        if(splitHalfEntryV2){
        tileClass=107U+tileClass-84U;addressPlan=1U;floatMetadataPlan=3U;
        }
        halfFullEarlyEntry=halfInput && dtype==outdtype && thresholdKind==3U &&
        tileClass==5U && addressPlan==0U && broadcastPlan==1U;
        if(halfFullEarlyEntry){tileClass=255U;addressPlan=1U;floatMetadataPlan=3U;}
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanFloatEntryResources() {
        if(!halfInput && dtype==outdtype && thresholdKind==1U && addressPlan==1U &&
        broadcastPlan==1U && floatMetadataPlan==3U && tileClass>=99U && tileClass<=106U)
        tileClass=125U+tileClass-99U;
        floatFullEntryMaskReuse=!halfInput && dtype==outdtype &&
        thresholdKind==3U && tileClass==5U && addressPlan==0U && broadcastPlan==0U;
        if(floatFullEntryMaskReuse){tileClass=250U;addressPlan=1U;broadcastPlan=1U;floatMetadataPlan=3U;}
        if(splitHalfEntryV2)tileClass=136U+tileClass-107U;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanFoldedAndGenericEntries() {
        floatUnitEntry= !halfInput && dtype==outdtype && thresholdKind==1U &&
        tileClass>=27U && tileClass<=28U && addressPlan==1U && t->waistRatio==1.0f;
        if(floatUnitEntry){
        tileClass=248U+tileClass-27U;broadcastPlan=1U;floatMetadataPlan=3U;
        }
        if(foldedRawEntry && tileClass>=125U && tileClass<=132U &&
        t->batch/64U==kGeometryFoldMinLaunch){
        const uint32_t extra=tileClass-125U+1U;
        tileClass=251U+(extra-1U)%4U;
        thresholdKind=extra<=4U?1U:3U;
        if(platform.GetCoreNumAiv()>=8U)context->SetBlockDim(8U);
        }
        if(runtimeFold){
        tileClass=251U+(short64-1U)%4U;
        thresholdKind=short64<=4U?1U:3U;
        addressPlan=1U;broadcastPlan=1U;floatMetadataPlan=3U;
        context->SetBlockDim(8U);
        }
        if(!nativeSignature){
        tileClass=0;addressPlan=0;broadcastPlan=0;floatMetadataPlan=0;
        thresholdKind=t->hasThreshold?(t->thresholdIsHalf?2U:1U):0U;

        const uint32_t genericWorkers=(t->batch+t->perCore-1U)/t->perCore;
        context->SetBlockDim(genericWorkers);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanShortCoefficients() {
        halfShortLiteralBits=halfInput && dtype==outdtype && thresholdKind==2U &&
        tileClass>=211U && tileClass<=241U && addressPlan==0U && broadcastPlan==0U &&
        floatMetadataPlan==0U && t->waistRatio==0.8f;
        floatShortPositiveZeroPlan=!halfInput && nativeSignature && dtype==outdtype &&
        thresholdKind==3U && tileClass>=211U && tileClass<=241U &&
        addressPlan==0U && broadcastPlan==0U && floatMetadataPlan==0U && t->waistRatio==0.8f;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus PlanMergedBatchOwners() {
        halfBatchMerge=halfFullEarlyEntry && t->batch>=128U && t->batch<=512U &&
        t->batch%64U==0U && platform.GetCoreNumAiv()>=8U;
        if(halfBatchMerge)context->SetBlockDim(8U);
        floatBatchMerge=floatFullEntryMaskReuse && t->batch>=128U && t->batch<=512U &&
        t->batch%64U==0U && platform.GetCoreNumAiv()>=8U;
        if(floatBatchMerge){tileClass=62U+t->batch/64U;context->SetBlockDim(8U);}
        return ge::GRAPH_SUCCESS;
    }

    void PlanRepeatAddress() {
        repeatAddressPlan = 0U;
        if (TryRepeatResource1()) {
            return;
        }
        if (TryRepeatResource2()) {
            return;
        }
        if (TryRepeatResource3()) {
            return;
        }
    }

    bool TryRepeatResource1() {
        if (runtimeFold) {
            repeatAddressPlan = 68U+full64;
            return true;
        }
        if (floatBatchMerge) {
            repeatAddressPlan = 66U;
            return true;
        }
        if (halfBatchMerge) {
            repeatAddressPlan = (50U+t->batch/64U);
            return true;
        }
        if (floatShortPositiveZeroPlan) {
            repeatAddressPlan = 51U;
            return true;
        }
        if ((splitHalfEntryV2 && tileClass>=136U && tileClass<=150U)) {
            repeatAddressPlan = 46U;
            return true;
        }
        if ((halfInput && nativeSignature && thresholdKind==2U &&
     tileClass==27U && addressPlan==0U && broadcastPlan==0U && floatMetadataPlan==0U)) {
            repeatAddressPlan = 64U;
            return true;
        }
        return false;
    }

    bool TryRepeatResource2() {
        if ((floatUnitEntry && tileClass==248U && thresholdKind==1U &&
     addressPlan==1U && broadcastPlan==1U && floatMetadataPlan==3U)) {
            repeatAddressPlan = 65U;
            return true;
        }
        if (halfShortLiteralBits) {
            repeatAddressPlan = 60U;
            return true;
        }
        if ((selectScalarPitchPlan && floatMetadataPlan==3U)) {
            repeatAddressPlan = 19U;
            return true;
        }
        if ((halfInput && dtype==outdtype && thresholdKind==3U &&
     tileClass==19U && addressPlan==0U && broadcastPlan==1U && floatMetadataPlan==0U)) {
            repeatAddressPlan = 67U;
            return true;
        }
        if (folded256) {
            repeatAddressPlan = 13U;
            return true;
        }
        return false;
    }

    bool TryRepeatResource3() {
        if ((floatFullEntryMaskReuse && tileClass==250U)) {
            repeatAddressPlan = 45U;
            return true;
        }
        if ((halfFullEarlyEntry && tileClass==255U)) {
            repeatAddressPlan = 49U;
            return true;
        }
        if ((halfInput && dtype==outdtype && thresholdKind==2U && tileClass>=211U && tileClass<=241U &&
        addressPlan==0U && broadcastPlan==0U && floatMetadataPlan==0U)) {
            repeatAddressPlan = 18U;
            return true;
        }
        if ((foldedRawEntry && tileClass>=251U && tileClass<=254U)) {
            repeatAddressPlan = 68U;
            return true;
        }
        return false;
    }

    ge::graphStatus Serialize() {
        ASCENDC_TPL_SEL_PARAM(context,dtype,outdtype,thresholdKind,tileClass,addressPlan,broadcastPlan,floatMetadataPlan,repeatAddressPlan);
        if(floatMetadataPlan!=0U && !selectScalarPitchPlan && !foldedRawEntry && !runtimeFold && !splitHalfEntryV2 && !halfFullEarlyEntry && !floatFullEntryMaskReuse && !floatUnitEntry){
        const float metadataBlock[8]={t->waistRatio,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        if(metadataPayload->GetCapacity()<sizeof(metadataBlock)) return ge::GRAPH_FAILED;
        if(memcpy_s(metadataPayload->GetData(),metadataPayload->GetCapacity(),
        metadataBlock,sizeof(metadataBlock))!=EOK) return ge::GRAPH_FAILED;
        metadataPayload->SetDataSize(sizeof(metadataBlock));
        }
        context->GetWorkspaceSizes(1)[0]=0;
        return ge::GRAPH_SUCCESS;
    }

    gert::TilingContext *context;
    SafetyBeltViolationGeometryTilingData *t;
    platform_ascendc::PlatformAscendC platform;
    decltype(static_cast<gert::TilingContext *>(nullptr)->GetRawTilingData()) metadataPayload = nullptr;
    uint32_t tile = 0U;
    uint32_t wideWorkers = 0U;
    uint32_t wideTail = 0U;
    bool halfInput = false;
    bool floatUnitTiles = false;
    bool identityTiling = false;
    uint32_t cores = 0U;
    uint32_t workers = 0U;
    uint32_t launchCores = 0U;
    uint32_t dtype = 0U;
    uint32_t outdtype = 0U;
    uint32_t thresholdKind = 0U;
    bool nativeSignature = false;
    uint32_t full64 = 0U;
    uint32_t short64 = 0U;
    bool runtimeFold = false;
    uint32_t tileClass = 0U;
    uint32_t outputBlockRows = 0U;
    uint32_t addressPlan = 0U;
    uint32_t broadcastPlan = 0U;
    uint32_t floatMetadataPlan = 0U;
    bool folded256 = false;
    bool selectScalarPitchPlan = false;
    bool foldedRawEntry = false;
    bool splitHalfEntryV2 = false;
    bool halfFullEarlyEntry = false;
    bool floatFullEntryMaskReuse = false;
    bool floatUnitEntry = false;
    bool halfShortLiteralBits = false;
    bool floatShortPositiveZeroPlan = false;
    bool halfBatchMerge = false;
    bool floatBatchMerge = false;
    uint32_t repeatAddressPlan = 0U;
};

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (ValidateGeometryShapes(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    GeometryTilingPlan plan(context);
    return plan.Run();
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c) {
 *c->GetOutputShape(0)=*c->GetInputShape(0); c->GetOutputShape(0)->SetDim(1,1); return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *c) {
 c->SetOutputDataType(0,DT_FLOAT); return GRAPH_SUCCESS;
}
}
namespace ops {
class SafetyBeltViolationGeometry:public OpDef {
public:
 explicit SafetyBeltViolationGeometry(const char *n):OpDef(n) {
 this->Input("person_boxes").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
 this->Input("anchor_points").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
 this->Input("threshold").ParamType(OPTIONAL).DataType({ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT16}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
 this->Output("violation_score").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT16}).Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
 this->Attr("waist_ratio").AttrType(REQUIRED).Float();
 this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
 this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
 }
}; OP_ADD(SafetyBeltViolationGeometry);
}
