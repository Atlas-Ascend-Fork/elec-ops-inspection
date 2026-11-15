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

template <typename T, bool ROW, bool RAW3, bool CUSTOM_TABLE = false> struct FireFramePhases {
	template <bool INIT, uint32_t SLOT, uint32_t PHASE, typename Kernel>
	__aicore__ static inline void Phase(Kernel &kernel) {
		if constexpr (INIT)
			kernel.template InitInputPhase<SLOT, PHASE>();
		else {
			using Addresses = typename Kernel::Addresses;
			Addresses::template ReadInput<SLOT, PHASE>(kernel.mem);
		}
	}
	template <bool INIT, uint32_t SLOT, typename Kernel> __aicore__ static inline void Input(Kernel &kernel) {
		Phase<INIT, SLOT, 0>(kernel);
		Phase<INIT, SLOT, 1>(kernel);
		Phase<INIT, SLOT, 2>(kernel);
		if constexpr (!RAW3) {
			Phase<INIT, SLOT, 3>(kernel);
			Phase<INIT, SLOT, 4>(kernel);
			Phase<INIT, SLOT, 5>(kernel);
		}
	}
	template <bool INIT, uint32_t SLOT, uint32_t PHASE, typename Kernel>
	__aicore__ static inline void OutputPhase(Kernel &kernel) {
		using Addresses = typename Kernel::Addresses;
		if constexpr (INIT) {
			if constexpr (CUSTOM_TABLE)
				kernel.template InitOutputPhase<SLOT, PHASE>();
			else
				Addresses::template Output<SLOT, PHASE>(kernel.mem);
		} else
			Addresses::template WriteOutput<SLOT, PHASE>(kernel.mem);
	}
	template <bool INIT, uint32_t SLOT, typename Kernel>
	__aicore__ static inline void Output(Kernel &kernel) {
		OutputPhase<INIT, SLOT, 0>(kernel);
		OutputPhase<INIT, SLOT, 1>(kernel);
		OutputPhase<INIT, SLOT, 2>(kernel);
		OutputPhase<INIT, SLOT, 3>(kernel);
		OutputPhase<INIT, SLOT, 4>(kernel);
	}
	template <bool INIT, uint32_t PHASE, typename Kernel>
	__aicore__ static inline void PriorPhase(Kernel &kernel) {
		if constexpr (CUSTOM_TABLE) {
			if constexpr (INIT)
				kernel.template InitPriorPhase<PHASE>();
			else
				kernel.template NativePriorPhase<PHASE>();
		} else {
			using Addresses = typename Kernel::Addresses;
			if constexpr (INIT)
				Addresses::template Prior<PHASE>(kernel.mem);
			else
				Addresses::template ReadPrior<PHASE>(kernel.mem);
		}
	}
	template <bool INIT, typename Kernel> __aicore__ static inline void Prior(Kernel &kernel) {
		PriorPhase<INIT, 0>(kernel);
		PriorPhase<INIT, 1>(kernel);
		PriorPhase<INIT, 2>(kernel);
		PriorPhase<INIT, 3>(kernel);
		PriorPhase<INIT, 4>(kernel);
	}
	template <typename Kernel> __aicore__ static inline void Init(Kernel &kernel) {
		Input<true, 0>(kernel);
		if constexpr (sizeof(T) == sizeof(float) && !ROW)
			Input<true, 1>(kernel);
		Output<true, 0>(kernel);
		if constexpr (sizeof(T) == sizeof(float) && !ROW)
			Output<true, 1>(kernel);
		Prior<true>(kernel);
		AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
		AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
	}
	template <typename Kernel, typename Storage>
	__aicore__ static inline void ReadInput(Kernel &kernel, Storage &mem, AscendC::LocalTensor<float> rf) {
		if constexpr (sizeof(T) == sizeof(float) && !ROW) {
			if (reinterpret_cast<uint64_t>(rf.GetPhyAddr()) ==
				reinterpret_cast<uint64_t>(mem.RawFixedSlot(0).GetPhyAddr()))
				Input<false, 0>(kernel);
			else
				Input<false, 1>(kernel);
		} else
			Input<false, 0>(kernel);
	}
	template <typename Kernel, typename Storage>
	__aicore__ static inline void WriteOutput(Kernel &kernel, Storage &mem) {
		if constexpr (sizeof(T) == sizeof(float) && !ROW) {
			if (mem.CurrentOutSlot() == 0)
				Output<false, 0>(kernel);
			else
				Output<false, 1>(kernel);
		} else
			Output<false, 0>(kernel);
	}
};
