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

// Immutable device index table avoids per-launch index generation.
alignas(256) static __gm__ const uint32_t FireSmallIndexGM548[512] = {
	FIRE_INDEX_RAW32 FIRE_INDEX_PRIOR32 FIRE_INDEX_OUTPUT32(FIRE_INDEX_IDENTITY)};
static_assert(sizeof(FireSmallIndexGM548) == 2048, "exact512u32");

// Packed input, prior and output index tables are read-only in GM.
#define FIRE_P8O_TU_float 1
#define FIRE_P8O_TU_half 2
#define FIRE_P8O_TU_I(t) FIRE_P8O_TU_##t
#define FIRE_P8O_TU(t) FIRE_P8O_TU_I(t)
#if defined(DTYPE_RAW_PRED) && FIRE_P8O_TU(DTYPE_RAW_PRED) == 2
#define FIRE_P8O_HALF_IMAGE 1
#define FIRE_P8O_INDEX(v) ((v) / 2U)
#else
#define FIRE_P8O_HALF_IMAGE 0
#define FIRE_P8O_INDEX(v) v
#endif
alignas(256) static __gm__ const uint32_t FirePacketIndexGM556[1152] = {
	FIRE_INDEX_RAW32 FIRE_INDEX_PRIOR32 FIRE_INDEX_OUTPUT32(FIRE_P8O_INDEX)
		FIRE_INDEX_RAW32 FIRE_INDEX_CHANNEL32(0U, 20U, 32U) FIRE_INDEX_CHANNEL32(4U, 20U, 32U)
			FIRE_INDEX_CHANNEL32(8U, 20U, 32U) FIRE_INDEX_CHANNEL32(12U, 20U, 32U)
				FIRE_INDEX_CHANNEL32(16U, 20U, 32U) FIRE_INDEX_OUTPUT32(FIRE_P8O_INDEX) 0U,
	24U,
	0U,
	24U,
	0U,
	24U,
	0U,
	24U,
	4U,
	28U,
	4U,
	28U,
	4U,
	28U,
	4U,
	28U,
	8U,
	32U,
	8U,
	32U,
	8U,
	32U,
	8U,
	32U,
	12U,
	36U,
	12U,
	36U,
	12U,
	36U,
	12U,
	36U,
	16U,
	40U,
	16U,
	40U,
	16U,
	40U,
	16U,
	40U,
	20U,
	44U,
	20U,
	44U,
	20U,
	44U,
	20U,
	44U,
	0U,
	20U,
	0U,
	20U,
	0U,
	20U,
	0U,
	20U,
	4U,
	24U,
	4U,
	24U,
	4U,
	24U,
	4U,
	24U,
	8U,
	28U,
	8U,
	28U,
	8U,
	28U,
	8U,
	28U,
	12U,
	32U,
	12U,
	32U,
	12U,
	32U,
	12U,
	32U,
	16U,
	36U,
	16U,
	36U,
	16U,
	36U,
	16U,
	36U,
	0U,
	32U,
	64U,
	96U,
	128U,
	4U,
	36U,
	68U,
	100U,
	132U,
	8U,
	40U,
	72U,
	104U,
	136U,
	12U,
	44U,
	76U,
	108U,
	140U,
	16U,
	48U,
	80U,
	112U,
	144U,
	20U,
	52U,
	84U,
	116U,
	148U,
	24U,
	56U,
	88U,
	120U,
	152U,
	28U,
	60U,
	92U,
	124U,
	156U,
};
static_assert(sizeof(FirePacketIndexGM556) == 4608, "exact1152u32");

alignas(256) static __gm__ const uint32_t FireSmallIndexGM678[544] = {
	FIRE_INDEX_RAW32 FIRE_INDEX_CHANNEL32(0U, 20U, 0U) FIRE_INDEX_CHANNEL32(4U, 20U, 0U)
		FIRE_INDEX_CHANNEL32(8U, 20U, 0U) FIRE_INDEX_CHANNEL32(8U, 20U, 0U) FIRE_INDEX_CHANNEL32(12U, 20U, 0U)
			FIRE_INDEX_CHANNEL32(16U, 20U, 0U) FIRE_INDEX_OUTPUT32(FIRE_INDEX_IDENTITY)};
static_assert(sizeof(FireSmallIndexGM678) == 2176, "exact544u32");

#undef FIRE_INDEX_CHANNEL32
#undef FIRE_INDEX_RAW32
#undef FIRE_INDEX_PRIOR32
#undef FIRE_INDEX_OUTPUT32
#undef FIRE_INDEX_IDENTITY
#undef FIRE_INDEX_HALF
