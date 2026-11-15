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

template <uint32_t PHASE, uint32_t STRIDE, uint32_t Q, bool RAW3 = true>
__aicore__ inline void FireInputAddresses(uint64_t sourceBase, uint64_t workBase, uint64_t *source,
										  uint64_t *target) {
#pragma unroll
	for (uint32_t k = 0; k < 16; ++k) {
		const uint32_t sourceOffset = RAW3 ? 48U * (k % 8U) + 24U * (k / 8U) : 48U * k;
		source[k] = sourceBase + uint64_t(sourceOffset + 8U * PHASE) * sizeof(float);
		const uint32_t field = 8U * PHASE + k / 2U;
		target[k] =
			workBase + uint64_t((field % 6U) * STRIDE + (field / 6U) * Q + (k % 2U) * (RAW3 ? 4U * Q : 8U)) *
						   sizeof(float);
	}
}

template <uint32_t PHASE, uint32_t STRIDE, uint32_t Q>
__aicore__ inline void FireOutputAddresses(uint64_t workBase, uint64_t outputBase, uint64_t *source,
										   uint64_t *target) {
#pragma unroll
	for (uint32_t k = 0; k < 16; ++k) {
		const uint32_t field = 8U * PHASE + k % 8U;
		source[k] =
			workBase + uint64_t((field % 5U) * STRIDE + (field / 5U) * Q + (k / 8U) * 8U) * sizeof(float);
		target[k] = outputBase + uint64_t((5U * (k / 2U + 8U * (k % 2U)) + PHASE) * 8U) * sizeof(float);
	}
}

template <uint32_t PHASE, uint32_t STRIDE, uint32_t Q, uint32_t GAP, bool DUPLICATE = true>
__aicore__ inline void FirePriorAddresses(uint64_t priorBase, uint64_t workBase, uint64_t *source,
										  uint64_t *target) {
#pragma unroll
	for (uint32_t k = 0; k < 16; ++k) {
		source[k] = priorBase + uint64_t(40U * (DUPLICATE ? k % 8U : k) + 8U * PHASE) * sizeof(float);
		const uint32_t field = 8U * PHASE + k / 2U;
		target[k] = workBase +
					uint64_t((6U + field % 5U) * STRIDE + (field / 5U) * Q + (k % 2U) * GAP) * sizeof(float);
	}
}

template <bool VA_GROUP38, uint32_t OFFSET> __aicore__ constexpr uint32_t FireVAOffset() {
	return FireFrameVAOffset<VA_GROUP38, OFFSET, 38U>();
}

template <uint32_t WORDS> __aicore__ inline void FireAdvanceVA(AscendC::LocalTensor<uint64_t> va) {
	auto *entries = reinterpret_cast<__ubuf__ int16_t *>(va.GetPhyAddr());
	AscendC::SetMaskCount();
	AscendC::SetVectorMask<int16_t, AscendC::MaskMode::COUNTER>(WORDS);
#pragma unroll
	for (uint32_t phase = 1U; phase < 5U; ++phase)
		asc_add_scalar(entries + phase * WORDS, entries, int16_t(phase), uint8_t(1), uint16_t(1), uint16_t(1),
					   uint16_t(8), uint16_t(8));
	AscendC::PipeBarrier<PIPE_V>();
	AscendC::SetMaskNorm();
	AscendC::ResetMask();
}

template <typename T, uint32_t STRIDE, uint32_t Q, uint32_t GAP, bool VA_GROUP38 = false>
struct FirePairAddressTable {
	template <uint32_t SLOT, uint32_t PHASE, bool RAW3 = true, typename Storage>
	__aicore__ static inline void Input(Storage &mem) {
		static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < (RAW3 ? 3U : 6U),
					  "valid packed input specialization");
		uint64_t sourceBase;
		if constexpr (sizeof(T) == sizeof(float))
			sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
		else
			sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
		const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
		uint64_t source[16], target[16];
		FireInputAddresses<PHASE, STRIDE, Q, RAW3>(sourceBase, workBase, source, target);
		if constexpr (SLOT == 0)
			FireStorePacked(mem.PackedVA()[FireVAOffset<VA_GROUP38, 4 * PHASE>()], target);
		if constexpr (!VA_GROUP38 || PHASE == 0U)
			FireStorePacked(mem.PackedVA()[FireVAOffset<VA_GROUP38, 24 + SLOT * 24 + 4 * PHASE>()], source);
	}
	template <uint32_t SLOT, uint32_t PHASE, typename Storage>
	__aicore__ static inline void Output(Storage &mem) {
		static_assert(SLOT < (sizeof(T) == sizeof(float) ? 2U : 1U) && PHASE < 5U,
					  "valid packed output specialization");
		const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
		uint64_t outputBase;
		if constexpr (sizeof(T) == sizeof(float))
			outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
		else
			outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
		uint64_t source[16], target[16];
		FireOutputAddresses<PHASE, STRIDE, Q>(workBase, outputBase, source, target);
		if constexpr (SLOT == 0)
			FireStorePacked(mem.PackedVA()[FireVAOffset<VA_GROUP38, 72 + 4 * PHASE>()], source);
		if constexpr (!VA_GROUP38 || PHASE == 0U)
			FireStorePacked(mem.PackedVA()[FireVAOffset<VA_GROUP38, 92 + SLOT * 20 + 4 * PHASE>()], target);
	}
	template <uint32_t PHASE, bool DUPLICATE = true, typename Storage>
	__aicore__ static inline void Prior(Storage &mem, uint64_t priorBase) {
		static_assert(PHASE < 5U, "valid packed prior phase");
		const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
		uint64_t source[16], target[16];
		FirePriorAddresses<PHASE, STRIDE, Q, GAP, DUPLICATE>(priorBase, workBase, source, target);
		FireStorePacked(mem.PackedVA()[FireVAOffset<VA_GROUP38, 132 + 4 * PHASE>()], target);
		if constexpr (!VA_GROUP38 || PHASE == 0U)
			FireStorePacked(mem.PackedVA()[FireVAOffset<VA_GROUP38, 152 + 4 * PHASE>()], source);
	}
	template <typename Storage> __aicore__ static inline void Init(Storage &mem, uint64_t priorBase) {
		Input<0, 0>(mem);
		Input<0, 1>(mem);
		Input<0, 2>(mem);
		Output<0, 0>(mem);
		Output<0, 1>(mem);
		Output<0, 2>(mem);
		Output<0, 3>(mem);
		Output<0, 4>(mem);
		if constexpr (sizeof(T) == sizeof(float)) {
			Input<1, 0>(mem);
			Input<1, 1>(mem);
			Input<1, 2>(mem);
			Output<1, 0>(mem);
			Output<1, 1>(mem);
			Output<1, 2>(mem);
			Output<1, 3>(mem);
			Output<1, 4>(mem);
		}
		Prior<0>(mem, priorBase);
		Prior<1>(mem, priorBase);
		Prior<2>(mem, priorBase);
		Prior<3>(mem, priorBase);
		Prior<4>(mem, priorBase);
		AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
		AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
		if constexpr (VA_GROUP38)
			FireAdvanceVA<80U>(mem.PackedVA());
	}
};

template <uint32_t STRIDE, uint32_t Q, uint32_t GAP> struct FireHalfPairAddressTable {
	template <uint32_t PHASE, typename Storage> __aicore__ static inline void Input(Storage &mem) {
		uint64_t target[16], source[16];
		const uint64_t h = reinterpret_cast<uint64_t>(mem.HalfRF().GetPhyAddr());
#pragma unroll
		for (uint32_t k = 0; k < 16; ++k)
			target[k] = h + (((16U * PHASE + k) % 6U) * STRIDE + ((16U * PHASE + k) / 6U) * Q) * 2U;
		FireStorePacked(mem.VA()[80U + 4U * PHASE], target);
#pragma unroll
		for (uint32_t slot = 0; slot < 2; ++slot) {
			const uint64_t r = reinterpret_cast<uint64_t>(mem.Raw(slot).GetPhyAddr());
#pragma unroll
			for (uint32_t i = 0; i < 16; ++i)
				source[i] = r + (48U * i + 16U * PHASE) * 2U;
			if constexpr (PHASE == 0U)
				FireStorePacked(mem.VA()[4U * slot], source);
		}
	}
	template <uint32_t PHASE, typename Storage> __aicore__ static inline void Output(Storage &mem) {
		uint64_t target[16], source[16];
		const uint64_t w = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
		const uint64_t r = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
#pragma unroll
		for (uint32_t i = 0; i < 16; ++i)
			source[i] = w + (((8U * PHASE + i % 8U) % 5U) * STRIDE + ((8U * PHASE + i % 8U) / 5U) * Q +
							 (i / 8U) * 8U) *
								4U;
#pragma unroll
		for (uint32_t k = 0; k < 16; ++k)
			target[k] = r + (5U * (k / 2U + 8U * (k % 2U)) + PHASE) * 32U;
		FireStorePacked(mem.VA()[92U + 4U * PHASE], source);
		if constexpr (PHASE == 0U)
			FireStorePacked(mem.VA()[8U], target);
	}
	template <uint32_t PHASE, typename Storage> __aicore__ static inline void Prior(Storage &mem) {
		uint64_t target[16], source[16];
		const uint64_t p = reinterpret_cast<uint64_t>(mem.Prior().GetPhyAddr());
		const uint64_t w = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
#pragma unroll
		for (uint32_t i = 0; i < 16; ++i)
			source[i] = p + (40U * (i % 8U) + 8U * PHASE) * 4U;
#pragma unroll
		for (uint32_t k = 0; k < 8; ++k) {
			target[2 * k] = w + ((6U + (8U * PHASE + k) % 5U) * STRIDE + ((8U * PHASE + k) / 5U) * Q) * 4U;
			target[2 * k + 1] = target[2 * k] + GAP * 4U;
		}
		FireStorePacked(mem.VA()[112U + 4U * PHASE], target);
		if constexpr (PHASE == 0U)
			FireStorePacked(mem.VA()[12U], source);
	}
	template <typename Storage> __aicore__ static inline void Init(Storage &mem) {
		Input<0>(mem);
		Input<1>(mem);
		Input<2>(mem);
		Output<0>(mem);
		Output<1>(mem);
		Output<2>(mem);
		Output<3>(mem);
		Output<4>(mem);
		Prior<0>(mem);
		Prior<1>(mem);
		Prior<2>(mem);
		Prior<3>(mem);
		Prior<4>(mem);
		AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
		AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
		FireAdvanceVA<64U>(mem.VA());
	}
};

template <uint32_t REPEAT, bool VA_GROUP38 = false> struct FirePairPhases {
	template <uint32_t SLOT> __aicore__ static inline void Input(AscendC::LocalTensor<uint64_t> va) {
		FirePackedTranspose<REPEAT * 2U, 1, 48>(va[FireVAOffset<VA_GROUP38, 0>()],
												va[FireVAOffset<VA_GROUP38, 24 + SLOT * 24>()]);
		FirePackedTranspose<REPEAT * 2U, 1, 48>(va[FireVAOffset<VA_GROUP38, 4>()],
												va[FireVAOffset<VA_GROUP38, 28 + SLOT * 24>()]);
		FirePackedTranspose<REPEAT * 2U, 1, 48>(va[FireVAOffset<VA_GROUP38, 8>()],
												va[FireVAOffset<VA_GROUP38, 32 + SLOT * 24>()]);
	}
	template <uint32_t SLOT> __aicore__ static inline void Output(AscendC::LocalTensor<uint64_t> va) {
		FirePackedTranspose<REPEAT, 80, 2>(va[FireVAOffset<VA_GROUP38, 92 + SLOT * 20>()],
										   va[FireVAOffset<VA_GROUP38, 72>()]);
		FirePackedTranspose<REPEAT, 80, 2>(va[FireVAOffset<VA_GROUP38, 96 + SLOT * 20>()],
										   va[FireVAOffset<VA_GROUP38, 76>()]);
		FirePackedTranspose<REPEAT, 80, 2>(va[FireVAOffset<VA_GROUP38, 100 + SLOT * 20>()],
										   va[FireVAOffset<VA_GROUP38, 80>()]);
		FirePackedTranspose<REPEAT, 80, 2>(va[FireVAOffset<VA_GROUP38, 104 + SLOT * 20>()],
										   va[FireVAOffset<VA_GROUP38, 84>()]);
		FirePackedTranspose<REPEAT, 80, 2>(va[FireVAOffset<VA_GROUP38, 108 + SLOT * 20>()],
										   va[FireVAOffset<VA_GROUP38, 88>()]);
	}
	__aicore__ static inline void Prior(AscendC::LocalTensor<uint64_t> va) {
		FirePackedTranspose<REPEAT, 1, 40>(va[FireVAOffset<VA_GROUP38, 132>()],
										   va[FireVAOffset<VA_GROUP38, 152>()]);
		FirePackedTranspose<REPEAT, 1, 40>(va[FireVAOffset<VA_GROUP38, 136>()],
										   va[FireVAOffset<VA_GROUP38, 156>()]);
		FirePackedTranspose<REPEAT, 1, 40>(va[FireVAOffset<VA_GROUP38, 140>()],
										   va[FireVAOffset<VA_GROUP38, 160>()]);
		FirePackedTranspose<REPEAT, 1, 40>(va[FireVAOffset<VA_GROUP38, 144>()],
										   va[FireVAOffset<VA_GROUP38, 164>()]);
		FirePackedTranspose<REPEAT, 1, 40>(va[FireVAOffset<VA_GROUP38, 148>()],
										   va[FireVAOffset<VA_GROUP38, 168>()]);
	}
};
