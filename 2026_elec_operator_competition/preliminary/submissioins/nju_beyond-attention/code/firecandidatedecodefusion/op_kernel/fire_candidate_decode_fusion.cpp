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

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "fire_candidate_decode_fusion_tiling.h"
#include "tiling_key_fire_candidate_decode_fusion.h"

#include "fire_indices.h"
#include "fire_generic.h"
#include "fire_frame.h"
#include "fire_pairs.h"
#include "fire_small.h"
#include "fire_half_pairs.h"
#include "fire_a2_rows.h"
#include "fire_a2_pairs.h"
#include "fire_dispatch.h"

template <typename DT_RAW_PRED, uint32_t REUSE>
__global__ __aicore__ void fire_candidate_decode_fusion(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded,
														GM_ADDR workspace, GM_ADDR tiling) {
	if (FireDispatchSkipBlock<REUSE>())
		return;
	AscendC::InitSocState();
	REGISTER_TILING_DEFAULT(FireFrameTilingData);
	REGISTER_TILING_FOR_TILINGKEY(
		"(TILING_KEY_VAR == 16128) || (TILING_KEY_VAR == 16129) || (TILING_KEY_VAR == 16384) || "
		"(TILING_KEY_VAR == 16385) || (TILING_KEY_VAR == 18432) || (TILING_KEY_VAR == 18433)",
		FireFrameTilingData);
	if constexpr (REUSE == 0xb240U) {
		FireDispatchWhole<DT_RAW_PRED>(raw_pred, priors, decoded);
	} else if constexpr (REUSE == 0xb500U || REUSE == 0xb501U) {
		if (FireDispatchPairs<DT_RAW_PRED, REUSE>(raw_pred, priors, decoded))
			return;
	} else if constexpr (REUSE == 0xb602U || REUSE == 0xb603U || REUSE == 0xb600U || REUSE == 0xb601U) {
		if (FireDispatchSpans<DT_RAW_PRED, REUSE>(raw_pred, priors, decoded))
			return;
	} else if constexpr (REUSE == 0xb280U) {
		if (FireDispatchRows<DT_RAW_PRED>(raw_pred, priors, decoded))
			return;
	} else if constexpr (REUSE == 0xb300U || (REUSE & 0xc000U) == 0xc000U) {
		if (FireDispatchScheduledFrame<DT_RAW_PRED, REUSE>(raw_pred, priors, decoded, tiling))
			return;
	} else {
		FireDispatchSmallOrGeneric<DT_RAW_PRED, REUSE>(raw_pred, priors, decoded, tiling);
	}
	AscendC::PipeBarrier<PIPE_ALL>();
}

#undef FIRE_PAIR_QUAL11_IMAGE
#undef FIRE_PAIR_HALF_SCORE_IMAGE

#undef FIRE_P8O_INDEX
#undef FIRE_P8O_HALF_IMAGE
#undef FIRE_P8O_TU
#undef FIRE_P8O_TU_I
#undef FIRE_P8O_TU_half
#undef FIRE_P8O_TU_float
