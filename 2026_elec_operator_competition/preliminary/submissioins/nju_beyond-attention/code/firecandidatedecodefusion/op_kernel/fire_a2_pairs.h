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
#include "fire_pair_tail.h"

template <typename T, uint32_t TILE> using FireA2F5FrameStorage = FirePairFrameStorage<T, TILE>;
template <typename T, uint32_t TILE> using FireA2F6AddFrameStorage = FireA2F5FrameStorage<T, TILE>;
template <typename T, bool VA_GROUP38 = false>
using FireA2F5PairStorage = FireFixedPairStorage<T, VA_GROUP38>;
template <typename T, bool VA_GROUP38 = false>
using FireA2F6SpanPairStorage = FireFixedPairStorage<T, VA_GROUP38>;
template <typename T, bool VA_GROUP38 = false>
using FireA2F6AddPairStorage = FireFixedPairStorage<T, VA_GROUP38>;

template <typename T, uint32_t N, bool SCORE_FUSED = false, bool ROTATE_PAIRS = false,
		  bool QUALIFIED_SCORE_ROTATE = false, bool QUALIFIED_HALF_SCORE = false, bool COORD_DIV2 = false,
		  bool VA_GROUP38 = false, uint32_t ROW = 832U>
using KernelFireA2F5Pair =
	KernelFireFixedPair<T, N, FirePairMode::A2F5, SCORE_FUSED, ROTATE_PAIRS, QUALIFIED_SCORE_ROTATE,
						QUALIFIED_HALF_SCORE, COORD_DIV2, VA_GROUP38, ROW>;

template <typename T, uint32_t N, bool SCORE_FUSED = false, bool ROTATE_PAIRS = false,
		  bool QUALIFIED_SCORE_ROTATE = false, bool QUALIFIED_HALF_SCORE = false, bool COORD_DIV2 = false,
		  bool VA_GROUP38 = false, uint32_t ROW = 832U>
using KernelFireA2F6SpanPair =
	KernelFireFixedPair<T, N, FirePairMode::A2F6Span, SCORE_FUSED, ROTATE_PAIRS, QUALIFIED_SCORE_ROTATE,
						QUALIFIED_HALF_SCORE, COORD_DIV2, VA_GROUP38, ROW>;

template <typename T, uint32_t N, bool SCORE_FUSED = false, bool ROTATE_PAIRS = false,
		  bool QUALIFIED_SCORE_ROTATE = false, bool QUALIFIED_HALF_SCORE = false, bool COORD_DIV2 = false,
		  bool VA_GROUP38 = false>
using KernelFireA2F6AddPair =
	KernelFireFixedPair<T, N, FirePairMode::A2F6Add, SCORE_FUSED, ROTATE_PAIRS, QUALIFIED_SCORE_ROTATE,
						QUALIFIED_HALF_SCORE, COORD_DIV2, VA_GROUP38, 832U>;

template <typename T, uint32_t N>
using KernelFireA2F5Tail320 =
	KernelFireA2TailPair<T, N, 320U, 320U, 832U, 39U, FireA2F5FrameStorage<T, 640U>>;
template <typename T, uint32_t N>
using KernelFireA2F5Tail512 =
	KernelFireA2TailPair<T, N, 512U, 512U, 768U, 42U, FireA2F5FrameStorage<T, 1024U>>;
template <typename T, uint32_t N>
using KernelFireA2F6SpanTail513 =
	KernelFireA2TailPair<T, N, 513U, 576U, 768U, 42U, FireA2F6SpanTailStorage<T>>;
template <typename T, uint32_t N>
using KernelFireA2F6AddTail321 =
	KernelFireA2TailPair<T, N, 321U, 384U, 832U, 39U, FireA2F5FrameStorage<T, 768U>>;
