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

template <typename T, uint32_t TILE, uint32_t PAD, uint32_t VA_WORDS, bool DOUBLE_PRIOR,
		  uint32_t ONE_WORDS = TILE * 6U, uint32_t INDEX_WORDS = DOUBLE_PRIOR ? TILE * 7U : TILE>
class FireBufferedStorage {
  protected:
	static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
	static constexpr uint32_t STRIDE = TILE + PAD;
	static constexpr uint32_t RAW_BYTES = TILE * 6 * sizeof(T);
	static constexpr uint32_t PRIOR_BYTES = TILE * 5 * sizeof(float);
	static constexpr uint32_t OUT_BYTES = TILE * 5 * sizeof(T);
	static constexpr uint32_t WORK_BYTES = ((STRIDE * 11 * sizeof(float) + 511) / 512) * 512 + 256;
	static constexpr uint32_t ONE_BYTES = ONE_WORDS * sizeof(float) + (sizeof(T) == sizeof(half) ? 256 : 0);
	static constexpr uint32_t RAW_BASE = 0;
	static constexpr uint32_t PRIOR_BASE = RAW_BASE + DEPTH * RAW_BYTES;
	static constexpr uint32_t OUT_BASE = PRIOR_BASE + (DOUBLE_PRIOR ? DEPTH : 1U) * PRIOR_BYTES;
	static constexpr uint32_t WORK_BASE = OUT_BASE + DEPTH * OUT_BYTES;
	static constexpr uint32_t ONE_BASE = WORK_BASE + WORK_BYTES;
	static constexpr uint32_t RF_BASE = ONE_BASE + ONE_BYTES;
	static constexpr uint32_t PACK_BASE =
		RF_BASE + (sizeof(T) == sizeof(half) ? TILE * 6 * sizeof(float) : 0);
	static constexpr uint32_t INDEX_BASE =
		PACK_BASE + (sizeof(T) == sizeof(half) ? TILE * 5 * sizeof(float) + (DOUBLE_PRIOR ? 0U : 256U) : 0);
	static constexpr uint32_t VA_BASE = INDEX_BASE + INDEX_WORDS * sizeof(uint32_t);
	static_assert(VA_BASE + VA_WORDS * sizeof(uint64_t) < 188416, "reserve the SDK temporary UB region");
	static_assert((RAW_BYTES | PRIOR_BYTES | OUT_BYTES | WORK_BYTES | ONE_BYTES | INDEX_BASE) % 32 == 0,
				  "all static addresses and capacities retain 32-byte alignment");
	uint32_t rawCursor = 0, priorCursor = 0, outCursor = 0;
	uint32_t rawSlot = 0, priorSlot = 0, outSlot = 0;
	uint32_t rawUsed = 0, priorUsed = 0, outUsed = 0;

  public:
	__aicore__ inline AscendC::LocalTensor<uint64_t> PackedVA() {
		return AscendC::LocalTensor<uint64_t>(AscendC::TPosition::VECCALC, VA_BASE, VA_WORDS);
	}
	__aicore__ inline AscendC::LocalTensor<float> PriorFixedSlot(uint32_t slot) {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + slot * PRIOR_BYTES,
										   TILE * 5);
	}
	__aicore__ inline AscendC::LocalTensor<T> OutAt(uint32_t slot) {
		return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + slot * OUT_BYTES, TILE * 5);
	}
	__aicore__ inline uint32_t CurrentOutSlot() const { return outSlot; }
	__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
		return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, INDEX_WORDS);
	}
	__aicore__ inline AscendC::LocalTensor<float> Work() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE,
										   WORK_BYTES / sizeof(float));
	}
	__aicore__ inline AscendC::LocalTensor<float> Ones() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, ONE_BYTES / sizeof(float));
	}
	__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, TILE * 6);
	}
	__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, PACK_BASE, TILE * 5);
	}
	// A view only: allocation/cursor/event ownership remains in AllocRaw/FreeRaw.
	__aicore__ inline AscendC::LocalTensor<T> RawFixedSlot(uint32_t slot) {
		return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + slot * RAW_BYTES, TILE * 6);
	}
	__aicore__ inline AscendC::LocalTensor<T> AllocRaw() {
		rawSlot = rawCursor;
		rawCursor = (rawCursor + 1) % DEPTH;
		if (rawUsed & (1U << rawSlot))
			AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
		return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, RAW_BASE + rawSlot * RAW_BYTES, TILE * 6);
	}
	__aicore__ inline AscendC::LocalTensor<float> AllocPrior() {
		priorSlot = priorCursor;
		priorCursor = DOUBLE_PRIOR ? (priorCursor + 1) % DEPTH : 0U;
		if (priorUsed & (1U << priorSlot))
			AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE + priorSlot * PRIOR_BYTES,
										   TILE * 5);
	}
	__aicore__ inline AscendC::LocalTensor<T> AllocOut() {
		outSlot = outCursor;
		outCursor = (outCursor + 1) % DEPTH;
		if (outUsed & (1U << outSlot))
			AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outSlot);
		return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE + outSlot * OUT_BYTES, TILE * 5);
	}
	__aicore__ inline void RawReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
	__aicore__ inline void RawWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(rawSlot); }
	__aicore__ inline void PriorReady() { AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
	__aicore__ inline void PriorWait() { AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2 + priorSlot); }
	__aicore__ inline void OutReady() { AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
	__aicore__ inline void OutWait() { AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outSlot); }
	__aicore__ inline void FreeRaw() {
		AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(rawSlot);
		rawUsed |= 1U << rawSlot;
	}
	__aicore__ inline void FreePrior() {
		AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2 + priorSlot);
		priorUsed |= 1U << priorSlot;
	}
	__aicore__ inline void FreeOut() {
		AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outSlot);
		outUsed |= 1U << outSlot;
	}
	__aicore__ inline void Drain() {
		for (uint32_t slot = 0; slot < DEPTH; ++slot) {
			if (rawUsed & (1U << slot))
				AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
			if (priorUsed & (1U << slot))
				AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2 + slot);
			if (outUsed & (1U << slot))
				AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot);
		}
	}
};

template <typename T, uint32_t TILE> using FireStaticStorage = FireBufferedStorage<T, TILE, 8U, 0U, true>;

template <typename T, uint32_t TILE, uint32_t PAD = 16U>
class FireFrameStorage : public FireBufferedStorage<T, TILE, PAD, 172U, false> {
	static_assert(TILE >= 128 && TILE <= 1024 && TILE % 128 == 0, "native hardware frame geometry");
	static_assert(PAD == 16U || (PAD == 8U && TILE == 512U && sizeof(T) == sizeof(half)),
				  "qualified frame padding");
};

template <typename T, uint32_t TILE> using FirePairFrameStorage = FireFrameStorage<T, TILE>;

template <typename T, uint32_t TILE, bool VA_GROUP28, bool H10>
class FireA2RowsStorage : public FireBufferedStorage<T, TILE, 16U, VA_GROUP28 ? 112U : 172U, false> {
	using Base = FireBufferedStorage<T, TILE, 16U, VA_GROUP28 ? 112U : 172U, false>;
	static_assert(TILE >= 128 && TILE <= 1024 && TILE % 128 == 0, "native hardware frame geometry");
	static_assert(H10 || !VA_GROUP28 ||
					  (sizeof(T) == sizeof(float) && TILE == 896U && Base::VA_BASE == 162560U &&
					   Base::VA_BASE + 112U * sizeof(uint64_t) == 163456U),
				  "qualified FP32 M20 VA28 fixed UB");
	static_assert(!H10 || !VA_GROUP28 ||
					  (sizeof(T) == sizeof(half) && TILE == 896U && Base::VA_BASE == 163072U &&
					   Base::VA_BASE + 112U * sizeof(uint64_t) == 163968U),
				  "qualified P10 VA28 fixed UB");
};
