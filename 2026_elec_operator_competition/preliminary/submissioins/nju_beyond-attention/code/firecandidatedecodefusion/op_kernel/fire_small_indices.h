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

#include "fire_index_segments.h"

// Fixed half full32 input, cached-prior and output layout.
alignas(256) static __gm__ const uint32_t FireEven8Half32Index[608] = {
	FIRE_INDEX_CHANNEL32(4416U, 24U, 0U) FIRE_INDEX_CHANNEL32(4420U, 24U, 0U)
		FIRE_INDEX_CHANNEL32(4424U, 24U, 0U) FIRE_INDEX_CHANNEL32(4428U, 24U, 0U)
			FIRE_INDEX_CHANNEL32(4432U, 24U, 0U) FIRE_INDEX_CHANNEL32(384U, 20U, 0U)
				FIRE_INDEX_CHANNEL32(388U, 20U, 0U) FIRE_INDEX_CHANNEL32(396U, 20U, 0U)
					FIRE_INDEX_CHANNEL32(400U, 20U, 0U) FIRE_INDEX_CHANNEL32(392U, 20U, 0U)
						FIRE_INDEX_CHANNEL32(392U, 20U, 0U) FIRE_INDEX_CHANNEL32(392U, 20U, 0U)
							FIRE_INDEX_CHANNEL32(392U, 20U, 0U) FIRE_INDEX_CHANNEL32(4436U, 24U, 0U)
								FIRE_INDEX_OUTPUT32(FIRE_INDEX_HALF)};
static_assert(sizeof(FireEven8Half32Index) == 2432, "exact608u32");

#undef FIRE_INDEX_CHANNEL32
#undef FIRE_INDEX_RAW32
#undef FIRE_INDEX_PRIOR32
#undef FIRE_INDEX_OUTPUT32
#undef FIRE_INDEX_IDENTITY
#undef FIRE_INDEX_HALF
