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

template <uint32_t FRAME, bool ROW, bool WHOLE> struct FireFrameRunSchedule {
	uint32_t block, begin, batchCount, batchFirst, count;

	__aicore__ inline bool Init(const FireFrameTilingData &schedule) {
		const auto &t = schedule.header;
		const uint32_t groupLanes = schedule.geometry & 0xffffU;
		if (groupLanes == 0U)
			return false;
		block = static_cast<uint32_t>(AscendC::GetBlockIdx());
		begin = (block % groupLanes) * t.perCore;
		if (begin >= t.n)
			return false;
		batchCount = ROW ? 1U : schedule.geometry >> 16;
		batchFirst = (block / groupLanes) * batchCount;
		count = WHOLE ? FRAME : (t.n - begin < t.perCore ? t.n - begin : t.perCore);
		return true;
	}
};
