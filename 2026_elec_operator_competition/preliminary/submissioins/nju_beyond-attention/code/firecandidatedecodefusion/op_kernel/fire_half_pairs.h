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

#include "fire_half_frame.h"

template <typename T, uint32_t N> struct FireHalfPairLoop {
	template <uint32_t WORDS, typename Kernel>
	__aicore__ static inline void Store(Kernel &kernel, uint32_t slot, uint32_t pos) {
		kernel.Output();
		AscendC::PipeBarrier<PIPE_V>();
		kernel.mem.AcquireOutput(slot);
		asc_float2half_rn(reinterpret_cast<__ubuf__ half *>(kernel.mem.Out(slot).GetPhyAddr()),
						  reinterpret_cast<__ubuf__ float *>(kernel.mem.PackedFloat().GetPhyAddr()),
						  uint32_t(WORDS));
		AscendC::PipeBarrier<PIPE_V>();
		AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
		AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
		kernel.CopyOutput(slot, pos);
		kernel.mem.PublishOutput(slot);
	}
	template <typename Kernel>
	__aicore__ static inline void Process(Kernel &kernel, uint32_t iteration, uint32_t cohort, uint32_t begin,
										  AscendC::LocalTensor<float> *f,
										  const AscendC::DataCopyExtParams &rp) {
		const uint32_t pair = (iteration + cohort) & 3U, slot = iteration & 1U, pos = 2U * pair * N + begin;
		if (iteration != 0) {
			auto r = kernel.mem.AllocRaw(slot);
			AscendC::DataCopyPad(r, kernel.rawGm[pos * 6U], rp,
								 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
			AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
		}
		kernel.TransformPair(slot, f);
		kernel.DecodeCoordinates(f);
		kernel.StorePair(slot, pos);
	}
	template <typename Kernel>
	__aicore__ static inline void RunTail(Kernel &kernel, uint32_t cohort, uint32_t begin,
										  const AscendC::DataCopyExtParams &rp) {
		AscendC::Duplicate(kernel.mem.Twos(), 2.0f, 8U);
		AscendC::LocalTensor<float> f[11];
		for (uint32_t k = 0; k < 11; ++k)
			f[k] = kernel.mem.Work()[k * Kernel::STRIDE];
		kernel.InitVA();
		kernel.PreparePrior();
		for (uint32_t iteration = 0; iteration < 4; ++iteration)
			Process(kernel, iteration, cohort, begin, f, rp);
		kernel.mem.Drain();
	}
};
template <typename T, uint32_t N, bool QUALIFIED_HALF_INPUT_PI8, uint32_t QUALIFIED_N>
class KernelFireHalfInputPi8Pair {
	friend struct FireHalfPairLoop<T, N>;
	static_assert(QUALIFIED_HALF_INPUT_PI8 && sizeof(T) == sizeof(half) && N == QUALIFIED_N,
				  "qualified half input B16 pi8");
	static constexpr uint32_t FRAME = 1664U, STRIDE = 1696U, Q = 208U;
	FireHalfPairStorage<T, 18656U, 147840U, 10176U> mem;
	AscendC::GlobalTensor<T> rawGm, outGm;
	AscendC::GlobalTensor<float> priorGm;
	template <bool B16, uint8_t REPEAT, uint16_t DST, uint16_t SRC>
	__aicore__ inline void Transpose(uint32_t dst, uint32_t src) {
		FireTranspose<B16, REPEAT, DST, SRC>(mem.VA()[dst], mem.VA()[src]);
	}
	__aicore__ inline void InitVA() { FireHalfPairAddressTable<STRIDE, Q, 104U>::Init(mem); }
	template <uint32_t SLOT> __aicore__ inline void Input() {
		Transpose<true, 13, 1, 48>(80, 4 * SLOT);
		Transpose<true, 13, 1, 48>(84, 16 + 4 * SLOT);
		Transpose<true, 13, 1, 48>(88, 32 + 4 * SLOT);
	}
	__aicore__ inline void Output() {
		Transpose<false, 13, 80, 2>(8, 92);
		Transpose<false, 13, 80, 2>(24, 96);
		Transpose<false, 13, 80, 2>(40, 100);
		Transpose<false, 13, 80, 2>(56, 104);
		Transpose<false, 13, 80, 2>(72, 108);
	}

	__aicore__ inline void PreparePrior() {
		AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
		Transpose<false, 13, 1, 40>(112, 12);
		Transpose<false, 13, 1, 40>(116, 28);
		Transpose<false, 13, 1, 40>(120, 44);
		Transpose<false, 13, 1, 40>(124, 60);
		Transpose<false, 13, 1, 40>(128, 76);
		AscendC::PipeBarrier<PIPE_V>();
	}

	__aicore__ inline void TransformPair(uint32_t slot, AscendC::LocalTensor<float> *f) {
		AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
		auto signedRaw = mem.Raw(slot).template ReinterpretCast<int16_t>();
		AscendC::Adds(signedRaw, signedRaw, int16_t(-32768), 9984U);
		AscendC::PipeBarrier<PIPE_V>();
		if (slot == 0)
			Input<0>();
		else
			Input<1>();
		AscendC::PipeBarrier<PIPE_V>();
		mem.ReleaseRaw(slot);
		AscendC::Cast(mem.Work(), mem.HalfRF(), AscendC::RoundMode::CAST_NONE, 10176U);
		AscendC::PipeBarrier<PIPE_V>();
		AscendC::Exp(mem.Work(), mem.Work(), 10176U);
		AscendC::Adds(mem.Work(), mem.Work(), 1.0f, 10176U);
		AscendC::Mul(f[4], f[4], f[5], FRAME);
		const AscendC::BinaryRepeatParams divBinary{1, 0, 1, 8, 0, 8};
		AscendC::Div<float>(mem.Work(), mem.Twos(), mem.Work(), uint64_t(64), uint8_t(106), divBinary);
		AscendC::Reciprocal<float>(f[4], f[4], uint64_t(64), uint8_t(26),
								   AscendC::UnaryRepeatParams{1, 1, 8, 8});
		AscendC::PipeBarrier<PIPE_V>();
	}

	__aicore__ inline void DecodeCoordinates(AscendC::LocalTensor<float> *f) {
		auto *wm = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
		AscendC::SetMaskCount();
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2U * STRIDE);
		asc_add_scalar(wm, wm, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
		asc_mul(wm + 2U * STRIDE, wm + 2U * STRIDE, wm + 2U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1),
				uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
		AscendC::PipeBarrier<PIPE_V>();
		asc_add(wm, wm, wm + 6U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8),
				uint8_t(8), uint8_t(8));
		asc_mul(wm + 2U * STRIDE, wm + 2U * STRIDE, wm + 9U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1),
				uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
		AscendC::PipeBarrier<PIPE_V>();
		FireScaleBoxes<STRIDE, QUALIFIED_N == 32768U ? FRAME : STRIDE>(mem.Work());
		AscendC::PipeBarrier<PIPE_V>();
	}

	__aicore__ inline void CopyOutput(uint32_t slot, uint32_t pos) {
		const AscendC::DataCopyExtParams op{2, 8320U, 0U, (N - 832U) * 10U, 0};
		AscendC::DataCopyPad(outGm[pos * 5U], mem.Out(slot), op);
	}
	__aicore__ inline void StorePair(uint32_t slot, uint32_t pos) {
		FireHalfPairLoop<T, N>::template Store<8320U>(*this, slot, pos);
	}

  public:
	__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
		const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
		if (block >= 39U)
			return;
		const uint32_t begin = block * 832U, cohort = block & 3U;
		rawGm.SetGlobalBuffer((__gm__ T *)raw);
		priorGm.SetGlobalBuffer((__gm__ float *)priors);
		outGm.SetGlobalBuffer((__gm__ T *)decoded);
		const AscendC::DataCopyExtParams rp{2, 9984U, (N - 832U) * 12U, 0U, 0};
		const AscendC::DataCopyExtParams pp{1, 16640U, 0, 0, 0};
		auto firstRaw = mem.AllocRaw(0);
		AscendC::DataCopyPad(firstRaw, rawGm[(2U * cohort * N + begin) * 6U], rp,
							 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
		AscendC::DataCopyPad(mem.Prior(), priorGm[begin * 5U], pp,
							 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
		AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
		AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
		if constexpr (QUALIFIED_N == 32768U)
			AscendC::Duplicate(mem.Ones(), 1.0f, 8U);
		FireHalfPairLoop<T, N>::RunTail(*this, cohort, begin, rp);
	}
};

template <typename T, uint32_t N, bool QUALIFIED_HALF_INPUT_PI8 = false>
using KernelFirePairInputPi8H11 = KernelFireHalfInputPi8Pair<T, N, QUALIFIED_HALF_INPUT_PI8, 32768U>;

template <typename T, uint32_t N, bool QUALIFIED_HALF_INPUT_PI8 = false>
using KernelFirePairInputPi8H12 = KernelFireHalfInputPi8Pair<T, N, QUALIFIED_HALF_INPUT_PI8, 32769U>;

template <typename T, uint32_t N, uint32_t SPAN, uint32_t COUNT, bool RAW_SCORE, uint32_t ROW,
		  uint32_t BLOCKS>
class KernelFireHalfSpanPair {
	friend struct FireHalfPairLoop<T, N>;
	static_assert(!RAW_SCORE || (N == 32769U && SPAN == ROW && COUNT == ROW), "score qualification scope");
	static_assert(sizeof(T) == sizeof(half) && (N == 32768U || N == 32769U) && SPAN % 64U == 0U &&
					  COUNT <= SPAN && COUNT > SPAN - 64U && SPAN <= ROW,
				  "qualified paired span");
	static constexpr uint32_t FRAME = 2U * SPAN, STRIDE = FRAME + 16U, Q = SPAN / 4U;
	FireHalfPairStorage<T, 18496U, 147200U, 10080U> mem;
	AscendC::GlobalTensor<T> rawGm, outGm;
	AscendC::GlobalTensor<float> priorGm;
	template <bool B16, uint8_t REPEAT, uint16_t DST, uint16_t SRC>
	__aicore__ inline void Transpose(uint32_t dst, uint32_t src) {
		FireTranspose<B16, REPEAT, DST, SRC>(mem.VA()[dst], mem.VA()[src]);
	}
	__aicore__ inline void InitVA() { FireHalfPairAddressTable<STRIDE, Q, SPAN / 8U>::Init(mem); }
	template <uint32_t SLOT> __aicore__ inline void Input() {
		Transpose<true, SPAN / 64U, 1, 48>(80, 4 * SLOT);
		Transpose<true, SPAN / 64U, 1, 48>(84, 16 + 4 * SLOT);
		Transpose<true, SPAN / 64U, 1, 48>(88, 32 + 4 * SLOT);
	}
	__aicore__ inline void Output() {
		Transpose<false, SPAN / 64U, 80, 2>(8, 92);
		Transpose<false, SPAN / 64U, 80, 2>(24, 96);
		Transpose<false, SPAN / 64U, 80, 2>(40, 100);
		Transpose<false, SPAN / 64U, 80, 2>(56, 104);
		Transpose<false, SPAN / 64U, 80, 2>(72, 108);
	}

	__aicore__ inline void PreparePrior() {
		AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
		Transpose<false, SPAN / 64U, 1, 40>(112, 12);
		Transpose<false, SPAN / 64U, 1, 40>(116, 28);
		Transpose<false, SPAN / 64U, 1, 40>(120, 44);
		Transpose<false, SPAN / 64U, 1, 40>(124, 60);
		Transpose<false, SPAN / 64U, 1, 40>(128, 76);
		AscendC::PipeBarrier<PIPE_V>();
	}

	__aicore__ inline void TransformPair(uint32_t slot, AscendC::LocalTensor<float> *f) {
		AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
		auto signedRaw = mem.Raw(slot).template ReinterpretCast<int16_t>();
		AscendC::Adds(signedRaw, signedRaw, int16_t(-32768), FRAME * 6U);
		AscendC::PipeBarrier<PIPE_V>();
		if (slot == 0)
			Input<0>();
		else
			Input<1>();
		AscendC::PipeBarrier<PIPE_V>();
		mem.ReleaseRaw(slot);
		AscendC::Cast(mem.Work(), mem.HalfRF(), AscendC::RoundMode::CAST_NONE, 6U * STRIDE);
		AscendC::PipeBarrier<PIPE_V>();
		AscendC::Exp(mem.Work(), mem.Work(), 6U * STRIDE);
		AscendC::Adds(mem.Work(), mem.Work(), 1.0f, 6U * STRIDE);
		AscendC::Mul(f[4], f[4], f[5], FRAME);
		const AscendC::BinaryRepeatParams divBinary{1, 0, 1, 8, 0, 8};
		AscendC::Div<float>(mem.Work(), mem.Twos(), mem.Work(), uint64_t(64), uint8_t(STRIDE / 16U),
							divBinary);
		if constexpr (RAW_SCORE) {
			AscendC::Reciprocal<float>(f[4], f[4], uint64_t(64), uint8_t(FRAME / 64U),
									   AscendC::UnaryRepeatParams{1, 1, 8, 8});
		} else {
			AscendC::Div<float>(f[4], mem.Ones(), f[4], uint64_t(64), uint8_t(FRAME / 64U), divBinary);
		}
		AscendC::PipeBarrier<PIPE_V>();
	}

	__aicore__ inline void DecodeCoordinates(AscendC::LocalTensor<float> *f) {
		FireDecodeBox<STRIDE>(f);
		AscendC::PipeBarrier<PIPE_V>();
		FireScaleBoxes<STRIDE, STRIDE>(mem.Work());
		AscendC::PipeBarrier<PIPE_V>();
	}

	__aicore__ inline void CopyOutput(uint32_t slot, uint32_t pos) {
		const AscendC::DataCopyExtParams op{
			2, COUNT * 10U, (SPAN * 10U - ((COUNT * 10U + 31U) / 32U) * 32U) / 32U, (N - COUNT) * 10U, 0};
		AscendC::DataCopyPad(outGm[pos * 5U], mem.Out(slot), op);
	}
	__aicore__ inline void StorePair(uint32_t slot, uint32_t pos) {
		FireHalfPairLoop<T, N>::template Store<FRAME * 5U>(*this, slot, pos);
	}

  public:
	__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
		const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
		if (block >= BLOCKS)
			return;
		const uint32_t begin = block * ROW, cohort = block & 3U;
		rawGm.SetGlobalBuffer((__gm__ T *)raw);
		priorGm.SetGlobalBuffer((__gm__ float *)priors);
		outGm.SetGlobalBuffer((__gm__ T *)decoded);
		const AscendC::DataCopyExtParams rp{2, COUNT * 12U, (N - COUNT) * 12U,
											(SPAN * 12U - ((COUNT * 12U + 31U) / 32U) * 32U) / 32U, 0};
		const AscendC::DataCopyExtParams pp{1, COUNT * 20U, 0, 0, 0};
		auto firstRaw = mem.AllocRaw(0);
		AscendC::DataCopyPad(firstRaw, rawGm[(2U * cohort * N + begin) * 6U], rp,
							 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
		AscendC::DataCopyPad(mem.Prior(), priorGm[begin * 5U], pp,
							 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
		AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
		AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
		AscendC::Duplicate(mem.Ones(), 1.0f, 8U);
		FireHalfPairLoop<T, N>::RunTail(*this, cohort, begin, rp);
	}
};

template <typename T, uint32_t N, uint32_t SPAN, uint32_t COUNT, bool RAW_SCORE = false>
using KernelFirePairSpan1112 = KernelFireHalfSpanPair<T, N, SPAN, COUNT, RAW_SCORE, 768U, 43U>;

template <typename T, uint32_t N, uint32_t SPAN, uint32_t COUNT, bool RAW_SCORE = false>
using KernelFirePairSpan47 = KernelFireHalfSpanPair<T, N, SPAN, COUNT, RAW_SCORE, 704U, 47U>;
