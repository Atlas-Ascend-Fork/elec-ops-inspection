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

__aicore__ inline void FireExpDenominator(AscendC::LocalTensor<float> rf, uint32_t words) {
	AscendC::Muls(rf, rf, -1.0f, words);
	AscendC::Exp(rf, rf, words);
	AscendC::Adds(rf, rf, 1.0f, words);
}
__aicore__ inline void FireReciprocalRaw(AscendC::LocalTensor<float> rf, AscendC::LocalTensor<float> one,
										 uint32_t words) {
	AscendC::Div(rf, one, rf, words);
	AscendC::PipeBarrier<PIPE_V>();
}
__aicore__ inline void FireActivateRaw(AscendC::LocalTensor<float> rf, AscendC::LocalTensor<float> one,
									   uint32_t words) {
	FireExpDenominator(rf, words);
	FireReciprocalRaw(rf, one, words);
}
__aicore__ inline void FireActivateOne8Raw(AscendC::LocalTensor<float> rf, AscendC::LocalTensor<float> one,
										   uint32_t words, bool one8Full) {
	FireExpDenominator(rf, words);
	if (one8Full) {
		const AscendC::BinaryRepeatParams divOne8{1, 0, 1, 8, 0, 8};
		AscendC::Div<float>(rf, one, rf, uint64_t(64), uint8_t(78), divOne8);
	} else
		AscendC::Div(rf, one, rf, words);
	AscendC::PipeBarrier<PIPE_V>();
}
