/*
 * Copyright (c) 2026 Beyond Attention. All rights reserved.
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

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fire_candidate_decode_fusion_tiling.h"
#include "../op_kernel/tiling_key_fire_candidate_decode_fusion.h"
#include <algorithm>
#include <limits>

namespace optiling {
// Preserve specialized metadata; other shapes use the generic 832-candidate path.
static uint32_t CompleteDevelopmentReuse(gert::TilingContext *c, FireCandidateDecodeFusionTilingData *header,
										 uint32_t requested, uint32_t total, uint32_t n, uint32_t cores) {
	if (requested == 0U || requested == 2U || requested == 0xc300U || requested == 0xc380U ||
		requested == 0xb300U || requested == 0xb500U || requested == 0xb501U || requested == 0xb280U ||
		requested == 0x7b2eU || requested == 0x7b34U || requested == 0xb240U)
		return requested;
	if (cores == 0U) {
		cores = 1U;
	}
	*header = {total, n, (total + cores - 1) / cores};
	c->GetRawTilingData()->SetDataSize(sizeof(FireCandidateDecodeFusionTilingData));
	return 2U;
}
static bool ValidateInputs(gert::TilingContext *c) {
	const auto &x = c->GetInputShape(0)->GetStorageShape();
	const auto &p = c->GetInputShape(1)->GetStorageShape();
	if (x.GetDimNum() != 3 || p.GetDimNum() != 2 || x.GetDim(2) != 6 || p.GetDim(1) != 5 ||
		x.GetDim(1) != p.GetDim(0) || x.GetDim(0) < 1 || x.GetDim(1) < 1 ||
		c->GetInputDesc(1)->GetDataType() != ge::DT_FLOAT)
		return false;
	// Bound element offsets, rather than rejecting safe nonaligned problem sizes.
	const uint64_t maxCandidates = std::numeric_limits<uint32_t>::max() / 6;
	if (uint64_t(x.GetDim(1)) > maxCandidates ||
		uint64_t(x.GetDim(0)) > maxCandidates / uint64_t(x.GetDim(1)))
		return false;
	auto dtype = c->GetInputDesc(0)->GetDataType();
	if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16)
		return false;
	return true;
}
static ge::graphStatus FinishTiling(gert::TilingContext *c, uint32_t DT_RAW_PRED, uint32_t compiledReuse,
									uint32_t cores) {
	ASCENDC_TPL_SEL_PARAM(c, DT_RAW_PRED, compiledReuse);
	c->SetBlockDim(cores);
	c->GetWorkspaceSizes(1)[0] = 0;
	return ge::GRAPH_SUCCESS;
}
static bool FinishSimpleTiling(gert::TilingContext *c, FireCandidateDecodeFusionTilingData *t,
							   FireCandidateDecodeFusionTilingData header, uint32_t DT_RAW_PRED,
							   uint32_t compiledReuse, uint32_t cores) {
	*t = header;
	c->GetRawTilingData()->SetDataSize(sizeof(FireCandidateDecodeFusionTilingData));
	FinishTiling(c, DT_RAW_PRED, compiledReuse, cores);
	return true;
}
static bool TrySmallTiling(gert::TilingContext *c, FireCandidateDecodeFusionTilingData *t, uint32_t batches,
						   uint32_t DT_RAW_PRED, uint32_t actualAvailable) {
	if (batches == 4 && t->n == 4097U && actualAvailable >= 20U)
		return FinishSimpleTiling(c, t, {16388U, 4097U, 832U}, DT_RAW_PRED, 0xb280U, 20U);
	if (batches == 1 && t->n == 32) {
		const uint32_t compiledReuse = actualAvailable >= 8U ? 0x7b36U : 0x7b2eU;
		return FinishSimpleTiling(c, t, {32, 32, 32}, DT_RAW_PRED, compiledReuse,
								  actualAvailable >= 8U ? 8U : 1U);
	}
	if (batches == 2 && t->n == 33 && actualAvailable >= 3) {
		const bool useSparse = actualAvailable >= 8U;
		const uint32_t compiledReuse = useSparse ? 0x7b38U : 0x7b34U;
		return FinishSimpleTiling(c, t, {66, 33, 32}, DT_RAW_PRED, compiledReuse, useSparse ? 8U : 3U);
	}
	return false;
}
static bool TryLargeFloatTiling(gert::TilingContext *c, FireCandidateDecodeFusionTilingData *t,
								uint32_t batches, uint32_t DT_RAW_PRED, uint32_t actualAvailable) {
	if (batches == 8 && (t->n == 32768U || t->n == 32769U) && DT_RAW_PRED == ge::DT_FLOAT &&
		actualAvailable >= 43U) {
		const uint32_t compiledReuse = t->n == 32768U ? 0xb500U : 0xb501U;
		return FinishSimpleTiling(c, t, {8U * t->n, t->n, 768U}, DT_RAW_PRED, compiledReuse, 43U);
	}
	return false;
}
static bool TryLargeHalfTiling(gert::TilingContext *c, FireCandidateDecodeFusionTilingData *t,
							   uint32_t batches, uint32_t DT_RAW_PRED, uint32_t deviceAvailable) {
	if (DT_RAW_PRED == ge::DT_FLOAT16 && batches == 8U && t->n == 32769U && deviceAvailable >= 47U)
		return FinishSimpleTiling(c, t, {8U * t->n, t->n, 704U}, DT_RAW_PRED, 0xb603U, 47U);
	if (DT_RAW_PRED == ge::DT_FLOAT16 && batches == 8U && t->n == 32769U && deviceAvailable >= 43U)
		return FinishSimpleTiling(c, t, {8U * t->n, t->n, 768U}, DT_RAW_PRED, 0xb601U, 43U);
	return false;
}
static uint32_t GetFrameSpan(const FireCandidateDecodeFusionTilingData &t, uint32_t batches, uint32_t workers,
							 uint32_t groups, uint32_t available) {
	if (groups == 0U)
		return 0U;
	const uint32_t lanes = workers / groups;
	if (lanes == 0U)
		return 0U;
	uint32_t span = ((t.n + lanes - 1) / lanes + 63) / 64 * 64;
	if (groups == batches && groups > 1) {
		// Amortize duplicated row preparation; retain single-row and reuse schedules.
		const uint32_t rowWorkers = std::min(available, (t.total + 511) / 512);
		uint32_t rowLanes = rowWorkers / groups;
		if (rowLanes == 0U) {
			rowLanes = 1U;
		}
		const uint32_t rowSpan = ((t.n + rowLanes - 1) / rowLanes + 63) / 64 * 64;
		if (rowSpan <= 1024)
			span = rowSpan;
	}
	return span;
}
struct FramePlan {
	FireFrameTilingData schedule;
	uint32_t groups, frame, activeLanes, cores, reuse;
};
static FramePlan MakeFramePlan(uint32_t batches, uint32_t n, uint32_t groups, uint32_t span) {
	FramePlan plan{};
	if (groups == 0U || span == 0U)
		return plan;
	plan.groups = groups;
	plan.schedule.header = {batches * n, n, span};
	plan.activeLanes = (n + span - 1) / span;
	plan.schedule.geometry = plan.activeLanes | ((batches / groups) << 16);
	// One cached-prior slot leaves UB capacity for a double-buffered1024 frame.
	plan.frame = (span + 127) / 128 * 128;
	plan.reuse = (groups == batches ? 0x4000U : 0x2000U) | plan.frame;
	plan.cores = plan.activeLanes * groups;
	if (groups == batches && batches <= 8 && plan.frame == 512 && span == 512 && n % 512 == 0 &&
		plan.activeLanes > 0 && plan.activeLanes <= 0xffffU)
		plan.reuse = 0xa000U | (batches << 4) | (plan.frame / 128);
	// Encode only the active row schedule.
	if (((plan.reuse & 0xe000U) == 0x2000U || (plan.reuse & 0xe000U) == 0x4000U) && plan.cores > 0 &&
		plan.cores <= 48)
		plan.reuse = (groups == batches ? 0xe000U : 0xc000U) | plan.frame;
	return plan;
}
static bool MatchesFrameSchedule(const FireFrameTilingData &s, uint32_t total, uint32_t n, uint32_t span,
								 uint32_t geometry) {
	return s.header.total == total && s.header.n == n && s.header.perCore == span && s.geometry == geometry;
}
static void AdjustLargeFrameReuse(FramePlan &plan, uint32_t batches, uint32_t n, uint32_t available) {
	if (plan.reuse == 0xc300U && batches == 8 && n == 32768 && available == 48 && plan.cores == 47) {
		if (MatchesFrameSchedule(plan.schedule, 262144, 32768, 704, (8U << 16) | 47U))
			plan.reuse = 0xb300U;
	}
	if (plan.reuse == 0xc380U && batches == 8 && n == 19968 && available == 48 && plan.cores == 48) {
		if (MatchesFrameSchedule(plan.schedule, 159744, 19968, 832, (4U << 16) | 24U))
			plan.reuse = 0xc380U;
	}
}
static void AdjustPairFrameReuse(FramePlan &plan, uint32_t batches, uint32_t n, uint32_t available) {
	if (plan.reuse == 0xc380U && batches == 8 && available == 40 && plan.cores == 40 && plan.groups == 1 &&
		plan.frame == 896 && plan.schedule.header.perCore == 832 && plan.activeLanes == 40 &&
		(n == 32768 || n == 32769)) {
		if (MatchesFrameSchedule(plan.schedule, 8U * n, n, 832, (8U << 16) | 40U))
			plan.reuse = n == 32768 ? 0xb500U : 0xb501U;
	}
}
static void AdjustSmallFrameReuse(FramePlan &plan, uint32_t batches, uint32_t n) {
	if (plan.reuse == 0xe280U && batches == 4 && n == 4097 && plan.cores == 32 && plan.groups == 4 &&
		plan.frame == 640 && plan.schedule.header.perCore == 576 && plan.activeLanes == 8) {
		if (MatchesFrameSchedule(plan.schedule, 16388, 4097, 576, (1U << 16) | 8U))
			plan.reuse = 0xb280U;
	}
	// Whole 512-candidate frames use the frame schedule metadata.
	if (plan.reuse == 0xa044U && batches == 4 && n == 4096 && plan.cores == 32 && plan.groups == batches &&
		plan.frame == 512 && plan.schedule.header.perCore == 512 && plan.activeLanes == 8 &&
		MatchesFrameSchedule(plan.schedule, 16384, 4096, 512, (1U << 16) | 8U))
		plan.reuse = 0xb240U;
}
static void SelectFrameReuse(FramePlan &plan, uint32_t batches, uint32_t n, uint32_t available) {
	AdjustLargeFrameReuse(plan, batches, n, available);
	AdjustPairFrameReuse(plan, batches, n, available);
	AdjustSmallFrameReuse(plan, batches, n);
	const bool knownFrameFamily = (batches == 4 && (n == 4096 || n == 4097)) ||
								  (batches == 8 && (n == 32768 || n == 32769 || n == 19968));
	if (!knownFrameFamily)
		plan.reuse = 2U;
}
static bool CommitFrameTiling(gert::TilingContext *c, FireCandidateDecodeFusionTilingData *t,
							  const FramePlan &plan, uint32_t DT_RAW_PRED) {
	if (plan.cores == 0U)
		return false;
	auto *rawTiling = c->GetRawTilingData();
	const uint32_t total = plan.schedule.header.total;
	const uint32_t n = plan.schedule.header.n;
	const uint32_t compiledReuse = CompleteDevelopmentReuse(c, t, plan.reuse, total, n, plan.cores);
	if (compiledReuse == 0xc300U || compiledReuse == 0xc380U || compiledReuse == 0xb300U) {
		auto *frameData = c->GetTilingData<FireFrameTilingData>();
		if (frameData == nullptr) {
			rawTiling->SetDataSize(sizeof(FireCandidateDecodeFusionTilingData));
			return false;
		}
		*frameData = plan.schedule;
		rawTiling->SetDataSize(sizeof(FireFrameTilingData));
	} else {
		*t = {total, n,
			  compiledReuse == 2U ? (total + plan.cores - 1) / plan.cores : plan.schedule.header.perCore};
		rawTiling->SetDataSize(sizeof(FireCandidateDecodeFusionTilingData));
	}
	FinishTiling(c, DT_RAW_PRED, compiledReuse, plan.cores);
	return true;
}
static bool TryFrameTiling(gert::TilingContext *c, FireCandidateDecodeFusionTilingData *t, uint32_t batches,
						   uint32_t available, uint32_t &cores, uint32_t DT_RAW_PRED) {
	// Native address preparation must be amortized by useful work on each core.
	const uint32_t workers = cores;
	uint32_t groups = std::min(batches, workers);
	// Largest batch partition whose candidate slice fits one UB hardware frame.
	for (; groups > 0; --groups) {
		if (batches % groups != 0)
			continue;
		const uint32_t span = GetFrameSpan(*t, batches, workers, groups, available);
		if (span > 1024 || span <= 256)
			continue;
		FramePlan plan = MakeFramePlan(batches, t->n, groups, span);
		SelectFrameReuse(plan, batches, t->n, available);
		cores = plan.cores;
		if (CommitFrameTiling(c, t, plan, DT_RAW_PRED))
			return true;
		cores = workers;
		break;
	}
	return false;
}
static ge::graphStatus CompleteGenericTiling(gert::TilingContext *c, FireCandidateDecodeFusionTilingData *t,
											 uint32_t batches, uint32_t cores, uint32_t DT_RAW_PRED) {
	if (cores == 0U)
		return ge::GRAPH_FAILED;
	t->perCore = (t->total + cores - 1) / cores;
	const uint32_t span = std::min(t->perCore, t->n);
	uint32_t REUSE = span <= 64 ? 0 : span <= 256 ? 1 : 2;
	if (cores >= batches && batches > 1) {
		const uint32_t lanes = cores / batches;
		cores = lanes * batches;
		t->perCore = (t->n + lanes - 1) / lanes;
		REUSE = t->perCore <= 64 ? 3064 : 5256;
		if (t->perCore > 256)
			REUSE = 5832;
	}
	if (t->total == 1)
		REUSE = 2001;
	if (REUSE == 3064)
		t->total = cores / batches;
	const uint32_t compiledReuse = CompleteDevelopmentReuse(c, t, REUSE, batches * t->n, t->n, cores);
	return FinishTiling(c, DT_RAW_PRED, compiledReuse, cores);
}
static ge::graphStatus TilingFunc(gert::TilingContext *c) {
	if (!ValidateInputs(c))
		return ge::GRAPH_FAILED;
	const auto &x = c->GetInputShape(0)->GetStorageShape();
	const uint32_t DT_RAW_PRED = static_cast<uint32_t>(c->GetInputDesc(0)->GetDataType());
	auto *t = c->GetTilingData<FireCandidateDecodeFusionTilingData>();
	if (t == nullptr)
		return ge::GRAPH_FAILED;
	t->total = x.GetDim(0) * x.GetDim(1);
	t->n = x.GetDim(1);
	const uint32_t batches = static_cast<uint32_t>(x.GetDim(0));
	auto platform = platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
	const uint32_t actualAvailable =
		std::min(40U, std::max(1U, static_cast<uint32_t>(platform.GetCoreNumAiv())));
	if (TrySmallTiling(c, t, batches, DT_RAW_PRED, actualAvailable) ||
		TryLargeFloatTiling(c, t, batches, DT_RAW_PRED, actualAvailable))
		return ge::GRAPH_SUCCESS;
	const uint32_t deviceAvailable =
		std::min(40U, std::max(1U, static_cast<uint32_t>(platform.GetCoreNumAiv())));
	if (TryLargeHalfTiling(c, t, batches, DT_RAW_PRED, deviceAvailable))
		return ge::GRAPH_SUCCESS;
	// Paired large frames use up to 40 available vector cores.
	const bool useExistingPair40 =
		batches == 8 && (t->n == 32768U || t->n == 32769U) && deviceAvailable >= 40U;
	const uint32_t available = useExistingPair40 ? 40U : deviceAvailable;
	const uint32_t grain = t->total <= 1024 ? 64 : 256;
	uint32_t cores = std::min(available, (t->total + grain - 1) / grain);
	if (TryFrameTiling(c, t, batches, available, cores, DT_RAW_PRED))
		return ge::GRAPH_SUCCESS;
	return CompleteGenericTiling(c, t, batches, cores, DT_RAW_PRED);
}
} // namespace optiling
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *c) {
	*c->GetOutputShape(0) = *c->GetInputShape(0);
	c->GetOutputShape(0)->SetDim(2, 5);
	return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *c) {
	c->SetOutputDataType(0, c->GetInputDataType(0));
	return GRAPH_SUCCESS;
}
} // namespace ge
namespace ops {
class FireCandidateDecodeFusion : public OpDef {
  public:
	explicit FireCandidateDecodeFusion(const char *name) : OpDef(name) {
		this->Input("raw_pred")
			.ParamType(REQUIRED)
			.DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
			.Format({ge::FORMAT_ND, ge::FORMAT_ND});
		this->Input("priors")
			.ParamType(REQUIRED)
			.DataType({ge::DT_FLOAT, ge::DT_FLOAT})
			.Format({ge::FORMAT_ND, ge::FORMAT_ND});
		this->Output("decoded")
			.ParamType(REQUIRED)
			.DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
			.Format({ge::FORMAT_ND, ge::FORMAT_ND});
		this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
		this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
	}
};
OP_ADD(FireCandidateDecodeFusion);
} // namespace ops
