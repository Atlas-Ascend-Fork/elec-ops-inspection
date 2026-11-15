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

template <typename T, uint32_t N, uint32_t COUNT, uint32_t PITCH, uint32_t ROW, uint32_t BLOCK,
		  typename Storage>
class KernelFireA2TailPair {
	static_assert(sizeof(T) == sizeof(float) && (N == 32768U || N == 32769U), "qualified FP32 tail pairs");
	static constexpr uint32_t FRAME = PITCH * 2U, STRIDE = FRAME + 16U, Q = FRAME / 8U;
	static constexpr bool PADDED = COUNT != PITCH;
	static_assert((!PADDED && (COUNT == 320U || COUNT == 512U) && N == 32768U) ||
					  (PADDED && (COUNT == 321U || COUNT == 513U) && N == 32769U),
				  "qualified tail count");
	static_assert(FRAME % 128U == 0U && BLOCK * ROW + COUNT == N, "exact tail pair geometry");
	static_assert(!PADDED || (((COUNT * 6U * sizeof(T) + 31U) / 32U) + 47U == PITCH * 6U * sizeof(T) / 32U &&
							  ((COUNT * 5U * sizeof(T) + 31U) / 32U) + 39U == PITCH * 5U * sizeof(T) / 32U),
				  "qualified byte and UB block gap geometry");
	Storage mem;
	AscendC::GlobalTensor<T> rawGm, outGm;
	AscendC::GlobalTensor<float> priorGm;

	__aicore__ inline void InitPackedVA() {
		FirePairAddressTable<T, STRIDE, Q, PITCH / 8U, false>::Init(
			mem, reinterpret_cast<uint64_t>(mem.PriorFixedSlot(0).GetPhyAddr()));
	}
	template <uint32_t SLOT> __aicore__ inline void Input() {
		FirePairPhases<FRAME / 128U, false>::template Input<SLOT>(mem.PackedVA());
	}
	template <uint32_t SLOT> __aicore__ inline void Output() {
		FirePairPhases<FRAME / 128U, false>::template Output<SLOT>(mem.PackedVA());
	}

	__aicore__ inline void PreparePrior(AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> *f) {
		mem.PriorWait();
		FirePairPhases<FRAME / 128U, false>::Prior(mem.PackedVA());
		AscendC::PipeBarrier<PIPE_V>();
		FireCopyPriorScale<STRIDE>(p, f[8]);
	}

	__aicore__ inline void TransformPair(uint32_t iteration, uint32_t slot, uint32_t pos,
										 AscendC::LocalTensor<T> firstRaw, AscendC::LocalTensor<float> one,
										 const AscendC::DataCopyExtParams &rp) {
		auto r = firstRaw;
		if (iteration != 0U) {
			r = mem.AllocRaw();
			if constexpr (PADDED)
				AscendC::DataCopyPad(r, rawGm[pos * 6U], rp,
									 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
			else
				AscendC::DataCopy(r, rawGm[pos * 6U],
								  AscendC::DataCopyParams{2U, uint16_t(COUNT * 24U / 32U),
														  uint16_t((N - COUNT) * 24U / 32U), 0U});
			mem.RawReady();
		}
		mem.RawWait();
		auto rf = r.template ReinterpretCast<float>();
		FireExpDenominator(rf, FRAME * 6U);
		if constexpr (COUNT == 513U)
			AscendC::Div<float>(rf, one, rf, uint64_t(64), uint8_t(108),
								AscendC::BinaryRepeatParams{1, 0, 1, 8, 0, 8});
		else
			AscendC::Div(rf, one, rf, FRAME * 6U);
		AscendC::PipeBarrier<PIPE_V>();
		if (slot == 0U)
			Input<0>();
		else
			Input<1>();
		AscendC::PipeBarrier<PIPE_V>();
	}

	__aicore__ inline void DecodeCoordinates(AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> *f) {
		AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
		FireDecodeBox<STRIDE>(f);
		AscendC::Mul(f[0], f[0], p, 4 * STRIDE);
		AscendC::Mul(f[4], f[4], f[5], FRAME);
	}

	__aicore__ inline void StorePair(uint32_t slot, uint32_t pos) {
		auto out = mem.AllocOut();
		AscendC::PipeBarrier<PIPE_V>();
		if (slot == 0U)
			Output<0>();
		else
			Output<1>();
		AscendC::PipeBarrier<PIPE_V>();
		mem.OutReady();
		mem.OutWait();
		const AscendC::DataCopyExtParams op{2, COUNT * 5U * uint32_t(sizeof(T)), PADDED ? 39U : 0U,
											(N - COUNT) * 5U * uint32_t(sizeof(T)), 0};
		if constexpr (PADDED)
			AscendC::DataCopyPad(outGm[pos * 5U], out, op);
		else
			AscendC::DataCopy(outGm[pos * 5U], out,
							  AscendC::DataCopyParams{2U, uint16_t(COUNT * 20U / 32U), 0U,
													  uint16_t((N - COUNT) * 20U / 32U)});
		mem.FreeOut();
		mem.FreeRaw();
	}

	__aicore__ inline void ProcessPair(uint32_t iteration, uint32_t begin, AscendC::LocalTensor<T> firstRaw,
									   AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> one,
									   AscendC::LocalTensor<float> *f, const AscendC::DataCopyExtParams &rp) {
		const uint32_t slot = iteration & 1U;
		const uint32_t pos = iteration * 2U * N + begin;
		TransformPair(iteration, slot, pos, firstRaw, one, rp);
		DecodeCoordinates(p, f);
		StorePair(slot, pos);
	}

  public:
	__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
		const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
		if (block != BLOCK)
			return;
		constexpr uint32_t begin = BLOCK * ROW;
		rawGm.SetGlobalBuffer((__gm__ T *)raw);
		priorGm.SetGlobalBuffer((__gm__ float *)priors);
		outGm.SetGlobalBuffer((__gm__ T *)decoded);
		const AscendC::DataCopyExtParams rp{2, COUNT * 6U * uint32_t(sizeof(T)),
											(N - COUNT) * 6U * uint32_t(sizeof(T)), PADDED ? 47U : 0U, 0};
		const AscendC::DataCopyExtParams pp{1, COUNT * 5U * uint32_t(sizeof(float)), 0, 0, 0};
		auto firstRaw = mem.AllocRaw();
		auto p = mem.AllocPrior();
		if constexpr (PADDED)
			AscendC::DataCopyPad(firstRaw, rawGm[begin * 6U], rp,
								 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
		else
			AscendC::DataCopy(firstRaw, rawGm[begin * 6U],
							  AscendC::DataCopyParams{2U, uint16_t(COUNT * 24U / 32U),
													  uint16_t((N - COUNT) * 24U / 32U), 0U});
		AscendC::DataCopyPad(p, priorGm[begin * 5U], pp,
							 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
		mem.RawReady();
		mem.PriorReady();
		auto one = mem.Ones();
		AscendC::Duplicate(one, 1.0f, COUNT == 513U ? 8U : FRAME * 6U);
		AscendC::LocalTensor<float> f[11];
		for (uint32_t k = 0; k < 11; ++k)
			f[k] = mem.Work()[k * STRIDE];
		InitPackedVA();
		PreparePrior(p, f);
		for (uint32_t iteration = 0; iteration < 4U; ++iteration) {
			ProcessPair(iteration, begin, firstRaw, p, one, f, rp);
		}
		mem.FreePrior();
		mem.Drain();
	}
};
