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
#include "fire_pair_common.h"

template <uint32_t FRAME, uint32_t STRIDE, bool DIRECT_STRIDE>
__aicore__ inline void FireDecodeFrame(AscendC::LocalTensor<float> *f, AscendC::LocalTensor<float> p) {
	AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
	FireDecodeBox<STRIDE>(f);
	if constexpr (DIRECT_STRIDE) {
#pragma unroll
		for (uint32_t k = 0; k < 4; ++k)
			AscendC::Mul(f[k], f[k], f[8], FRAME);
	} else
		AscendC::Mul(f[0], f[0], p, 4 * STRIDE);
	AscendC::Mul(f[4], f[4], f[5], FRAME);
}

template <uint32_t FRAME, uint32_t STRIDE>
__aicore__ inline void FireDecodeWholeFrame(AscendC::LocalTensor<float> work) {
	auto *wPtr = reinterpret_cast<__ubuf__ float *>(work.GetPhyAddr());
	AscendC::SetMaskCount();
	AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(4 * STRIDE);
	asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
	AscendC::PipeBarrier<PIPE_V>();
	AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2 * STRIDE);
	asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
	asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1),
			uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
	AscendC::PipeBarrier<PIPE_V>();
	asc_add(wPtr, wPtr, wPtr + 6 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8),
			uint8_t(8), uint8_t(8));
	asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + 9 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1),
			uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
	AscendC::PipeBarrier<PIPE_V>();
	AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(FRAME);
#pragma unroll
	for (uint32_t k = 0; k < 4; ++k)
		asc_mul(wPtr + k * STRIDE, wPtr + k * STRIDE, wPtr + 8 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1),
				uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
	asc_mul(wPtr + 4 * STRIDE, wPtr + 4 * STRIDE, wPtr + 5 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1),
			uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
	AscendC::SetMaskNorm();
	AscendC::ResetMask();
}
