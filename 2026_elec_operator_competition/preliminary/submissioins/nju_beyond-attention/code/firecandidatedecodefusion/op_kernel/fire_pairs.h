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

#pragma once

#include "fire_pair_driver.h"

#define FIRE_TU_KIND_float 1
#define FIRE_TU_KIND_half 2
#define FIRE_TU_KIND_I(t) FIRE_TU_KIND_##t
#define FIRE_TU_KIND(t) FIRE_TU_KIND_I(t)
#if defined(DTYPE_RAW_PRED) && FIRE_TU_KIND(DTYPE_RAW_PRED) == 2
#define FIRE_PAIR_BASE_IMAGE 1
#else
#define FIRE_PAIR_BASE_IMAGE 0
#endif

template <typename T> using FirePairStorage576 = FireFixedPairStorage<T, false>;

template <typename T, uint32_t N, bool SCORE_FUSED = false, bool ROTATE_PAIRS = false,
		  bool QUALIFIED_SCORE_ROTATE = false, bool QUALIFIED_HALF_SCORE = false>
using KernelFirePair576 =
	KernelFireFixedPair<T, N, FirePairMode::Fixed576, SCORE_FUSED, ROTATE_PAIRS, QUALIFIED_SCORE_ROTATE,
						QUALIFIED_HALF_SCORE, false, false, 832U, FIRE_PAIR_BASE_IMAGE>;
#if FIRE_PAIR_BASE_IMAGE
#define FIRE_PAIR_QUAL11_IMAGE 0
#define FIRE_PAIR_HALF_SCORE_IMAGE 1
#else
#define FIRE_PAIR_QUAL11_IMAGE 1
#define FIRE_PAIR_HALF_SCORE_IMAGE 0
#endif
#undef FIRE_PAIR_BASE_IMAGE
#undef FIRE_TU_KIND
#undef FIRE_TU_KIND_I
#undef FIRE_TU_KIND_half
#undef FIRE_TU_KIND_float

// Row metadata packs rowLanes into total; n/perCore retain the row geometry.
// Direct tensors use fixed 64-candidate resource addresses without slot reuse.
