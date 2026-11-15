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

#include "fire_storage.h"

template <typename T, bool VA_GROUP38 = false> class FireFixedPairStorage {
	static constexpr uint32_t RAW_BYTES = 1664U * 6U * sizeof(T);
	static constexpr uint32_t PRIOR_BASE = 2U * RAW_BYTES;
	static constexpr uint32_t PRIOR_BYTES = 4U * 1680U * sizeof(float);
	static constexpr uint32_t WORK_BASE = PRIOR_BASE + PRIOR_BYTES + 256U;
	static constexpr uint32_t WORK_BYTES = 74496U;
	static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
	static constexpr uint32_t RF_BASE = ONE_BASE + 32U + (sizeof(T) == sizeof(half) ? 224U : 0U);
	static constexpr uint32_t VA_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? 39936U : 0U);
	static constexpr uint32_t VA_WORDS = VA_GROUP38 ? 152U : 172U;
	static_assert(!VA_GROUP38 || sizeof(T) == sizeof(float), "VA38 only FP32");
	static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) ==
					  (VA_GROUP38 ? 182752U : (sizeof(T) == sizeof(float) ? 182912U : 183136U)),
				  "fixed576 UB high water");
	static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) < 188416U, "retain SDK UB reserve");
	static_assert(WORK_BASE % 512U == 0U &&
					  (ONE_BASE - (sizeof(T) == sizeof(float) ? 0U : RF_BASE)) % 512U == 256U,
				  "fixed phases");
	uint32_t used = 0;

  public:
	__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
		return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, slot * RAW_BYTES, 9984U);
	}
	__aicore__ inline AscendC::LocalTensor<T> OutAt(uint32_t slot) {
		return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, slot * RAW_BYTES, 8320U);
	}
	__aicore__ inline AscendC::LocalTensor<float> Prior() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE, 6720U);
	}
	__aicore__ inline AscendC::LocalTensor<float> Work() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, WORK_BYTES / 4U);
	}
	__aicore__ inline AscendC::LocalTensor<float> Ones() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, 8U);
	}
	__aicore__ inline AscendC::LocalTensor<float> Twos() {
		static_assert(sizeof(T) == sizeof(float), "coordinate numerator2 only FP32");
		constexpr uint32_t VA_END = VA_BASE + VA_WORDS * sizeof(uint64_t);
		constexpr uint32_t TWO_BASE = ONE_BASE + ((VA_END - ONE_BASE + 511U) / 512U) * 512U;
		static_assert(ONE_BASE == 181504U && VA_END == (VA_GROUP38 ? 182752U : 182912U) &&
						  TWO_BASE == 183040U,
					  "qualified FP32 numerator address");
		static_assert(TWO_BASE % 32U == 0U && TWO_BASE % 512U == 256U && TWO_BASE >= VA_END &&
						  TWO_BASE + 32U < 188416U,
					  "separate numerator2 after fixed VA extent");
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, TWO_BASE, 8U);
	}
	__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 9984U);
	}
	__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 8320U);
	}
	__aicore__ inline AscendC::LocalTensor<uint64_t> PackedVA() {
		return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, VA_BASE, VA_WORDS);
	}
	__aicore__ inline AscendC::LocalTensor<T> AllocSlot(uint32_t slot) {
		if (used & (1U << slot)) {
			AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
			AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
		}
		return RawFixedSlot(slot);
	}
	__aicore__ inline void PublishOutputAndRelease(uint32_t slot) {
		AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
		AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
		used |= 1U << slot;
	}
	__aicore__ inline void Drain() {
		for (uint32_t slot = 0; slot < 2; ++slot)
			if (used & (1U << slot)) {
				AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
				AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
			}
	}
};

template <typename T>
class FireA2F6SpanTailStorage : public FireBufferedStorage<T, 1152U, 16U, 172U, false, 8U, 0U> {
	using Base = FireBufferedStorage<T, 1152U, 16U, 172U, false, 8U, 0U>;
	static_assert(sizeof(T) == sizeof(float), "tail513 float storage only");
	static_assert(Base::RAW_BYTES == 27648U && Base::PRIOR_BASE == 55296U && Base::OUT_BASE == 78336U &&
					  Base::WORK_BASE == 124416U && Base::ONE_BASE == 176384U && Base::VA_BASE == 176416U &&
					  Base::VA_BASE + 172U * 8U == 177792U,
				  "qualified tail513 exact fixed layout");
};

template <typename T, uint32_t WORK_WORDS, uint32_t ONE_BASE, uint32_t HALF_WORDS> class FireHalfPairStorage {
	uint32_t used = 0;

  public:
	__aicore__ inline AscendC::LocalTensor<T> Raw(uint32_t slot) {
		return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, slot * 19968U, 9984U);
	}
	__aicore__ inline AscendC::LocalTensor<T> Out(uint32_t slot) {
		return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, 39936U + slot * 16640U, 8320U);
	}
	__aicore__ inline AscendC::LocalTensor<float> Prior() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, 39936U, 4160U);
	}
	__aicore__ inline AscendC::LocalTensor<float> Work() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, 73216U, WORK_WORDS);
	}
	__aicore__ inline AscendC::LocalTensor<float> Ones() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, 8U);
	}
	__aicore__ inline AscendC::LocalTensor<float> Twos() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE + 32U, 8U);
	}
	__aicore__ inline AscendC::LocalTensor<T> HalfRF() {
		return AscendC::LocalTensor<T>(AscendC::TPosition::VECCALC, ONE_BASE + 256U, HALF_WORDS);
	}
	__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE + 256U, 8320U);
	}
	__aicore__ inline AscendC::LocalTensor<uint64_t> VA() {
		return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, 187232U, 132U);
	}
	__aicore__ inline AscendC::LocalTensor<T> AllocRaw(uint32_t slot) {
		if (used & (1U << slot))
			AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
		return Raw(slot);
	}
	__aicore__ inline void ReleaseRaw(uint32_t slot) {
		AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
		used |= 1U << slot;
	}
	__aicore__ inline void AcquireOutput(uint32_t slot) {
		if (used & (1U << (slot + 2U)))
			AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
	}
	__aicore__ inline void PublishOutput(uint32_t slot) {
		AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot);
		used |= 1U << (slot + 2U);
	}
	__aicore__ inline void Drain() {
		for (uint32_t slot = 0; slot < 2; ++slot) {
			if (used & (1U << slot))
				AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
			if (used & (1U << (slot + 2U)))
				AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
		}
	}
};
