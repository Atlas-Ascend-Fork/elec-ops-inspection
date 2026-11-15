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
#include "fire_activation.h"

template <typename T, typename Storage>
__aicore__ inline AscendC::LocalTensor<T>
FireLoadFrameBatch(Storage &mem, AscendC::GlobalTensor<T> rawGm, AscendC::LocalTensor<T> firstRaw,
				   uint32_t count, uint32_t pos, uint32_t batch, uint32_t batchFirst) {
	auto r = firstRaw;
	if (batch != batchFirst) {
		r = mem.AllocRaw();
		AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
		AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
		mem.RawReady();
	}
	return r;
}
template <typename T, bool ONE8 = false, typename Storage>
__aicore__ inline AscendC::LocalTensor<float> FirePrepareFrameRaw(Storage &mem, AscendC::LocalTensor<T> r,
																  AscendC::LocalTensor<float> one,
																  uint32_t count, bool one8Full = false) {
	mem.RawWait();
	AscendC::LocalTensor<float> rf;
	if constexpr (sizeof(T) == sizeof(float))
		rf = r.template ReinterpretCast<float>();
	else {
		rf = mem.RawFloat();
		AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
	}
	if constexpr (ONE8)
		FireActivateOne8Raw(rf, one, count * 6, one8Full);
	else
		FireActivateRaw(rf, one, count * 6);
	return rf;
}
template <typename T, typename Storage>
__aicore__ inline void FireStoreDecodedFrame(Storage &mem, AscendC::GlobalTensor<T> outGm,
											 AscendC::LocalTensor<T> out, uint32_t count, uint32_t pos) {
	mem.OutReady();
	mem.OutWait();
	AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
	AscendC::DataCopyPad(outGm[pos * 5], out, op);
	mem.FreeOut();
	mem.FreeRaw();
}
