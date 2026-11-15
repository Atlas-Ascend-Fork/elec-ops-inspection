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

template <typename T, uint32_t FRAME, uint32_t STRIDE, bool RAW3, bool VA_GROUP28 = false>
struct FireFrameAddressTable {
	template <uint32_t OFFSET> __aicore__ static constexpr uint32_t Offset() {
		return FireFrameVAOffset<VA_GROUP28, OFFSET, 28U>();
	}
	template <uint32_t SLOT, uint32_t PHASE, typename Storage>
	__aicore__ static inline void Input(Storage &mem) {
		uint64_t sourceBase;
		if constexpr (sizeof(T) == sizeof(float))
			sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
		else
			sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
		const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
		uint64_t source[16], target[16];
		FireInputAddresses<PHASE, STRIDE, FRAME / 8, RAW3>(sourceBase, workBase, source, target);
		if constexpr (SLOT == 0)
			FireStorePacked(mem.PackedVA()[Offset<4 * PHASE>()], target);
		if constexpr (!VA_GROUP28 || PHASE == 0U)
			FireStorePacked(mem.PackedVA()[Offset<24 + SLOT * 24 + 4 * PHASE>()], source);
	}
	template <uint32_t SLOT, uint32_t PHASE, typename Storage>
	__aicore__ static inline void Output(Storage &mem) {
		static_assert(PHASE < 5 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1),
					  "valid padded output specialization");
		const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
		uint64_t outputBase;
		if constexpr (sizeof(T) == sizeof(float))
			outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
		else
			outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
		uint64_t source[16], target[16];
		FireOutputAddresses<PHASE, STRIDE, FRAME / 8>(workBase, outputBase, source, target);
		if constexpr (SLOT == 0)
			FireStorePacked(mem.PackedVA()[Offset<72 + 4 * PHASE>()], source);
		if constexpr (!VA_GROUP28 || PHASE == 0U)
			FireStorePacked(mem.PackedVA()[Offset<92 + SLOT * 20 + 4 * PHASE>()], target);
	}
	template <uint32_t PHASE, typename Storage> __aicore__ static inline void Prior(Storage &mem) {
		static_assert(PHASE < 5, "native prior hardware phase");
		const uint64_t sourceBase = reinterpret_cast<uint64_t>(mem.PriorFixedSlot(0).GetPhyAddr());
		const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
		uint64_t source[16], target[16];
		FirePriorAddresses<PHASE, STRIDE, FRAME / 8, 8U, false>(sourceBase, workBase, source, target);
		FireStorePacked(mem.PackedVA()[Offset<132 + 4 * PHASE>()], target);
		if constexpr (!VA_GROUP28 || PHASE == 0U)
			FireStorePacked(mem.PackedVA()[Offset<152 + 4 * PHASE>()], source);
	}
	template <uint32_t SLOT, uint32_t PHASE, typename Storage>
	__aicore__ static inline void ReadInput(Storage &mem) {
		static_assert(PHASE < 6 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1), "valid packed input phase");
		if constexpr (RAW3)
			FireFramePackedTranspose<FRAME / 64, 1, 48>(mem.PackedVA()[Offset<4 * PHASE>()],
														mem.PackedVA()[Offset<24 + SLOT * 24 + 4 * PHASE>()]);
		else
			FireFramePackedTranspose<FRAME / 128, 2, 96>(
				mem.PackedVA()[Offset<4 * PHASE>()], mem.PackedVA()[Offset<24 + SLOT * 24 + 4 * PHASE>()]);
	}
	template <uint32_t SLOT, uint32_t PHASE, typename Storage>
	__aicore__ static inline void WriteOutput(Storage &mem) {
		static_assert(PHASE < 5 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1), "valid packed output phase");
		FireFramePackedTranspose<FRAME / 128, 80, 2>(mem.PackedVA()[Offset<92 + SLOT * 20 + 4 * PHASE>()],
													 mem.PackedVA()[Offset<72 + 4 * PHASE>()]);
	}
	template <uint32_t PHASE, typename Storage> __aicore__ static inline void ReadPrior(Storage &mem) {
		FireFramePackedTranspose<FRAME / 128, 2, 80>(mem.PackedVA()[Offset<132 + 4 * PHASE>()],
													 mem.PackedVA()[Offset<152 + 4 * PHASE>()]);
	}
};
