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

#include "fire_tensor_common.h"

template <uint32_t STRIDE>
__aicore__ inline void FireCopyPriorScale(AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> scale) {
	AscendC::DataCopy(p, scale, STRIDE);
	AscendC::DataCopy(p[STRIDE], scale, STRIDE);
	AscendC::DataCopy(p[2 * STRIDE], scale, STRIDE);
	AscendC::DataCopy(p[3 * STRIDE], scale, STRIDE);
	AscendC::PipeBarrier<PIPE_V>();
}

template <uint32_t STRIDE> __aicore__ inline void FireDecodeBox(AscendC::LocalTensor<float> *f) {
	AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
	AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
	AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
	AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
}

template <uint32_t STRIDE, uint32_t COUNT>
__aicore__ inline void FireScaleBoxes(AscendC::LocalTensor<float> work) {
	auto *wPtr = reinterpret_cast<__ubuf__ float *>(work.GetPhyAddr());
	AscendC::SetMaskCount();
	AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(COUNT);
#pragma unroll
	for (uint32_t k = 0; k < 4; ++k)
		asc_mul(wPtr + k * STRIDE, wPtr + k * STRIDE, wPtr + 8 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1),
				uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
	AscendC::SetMaskNorm();
	AscendC::ResetMask();
}
