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

#include "fire_frame_addresses.h"
#include "fire_frame_io.h"
#include "fire_frame_phases.h"
#include "fire_frame_math.h"
#include "fire_storage.h"

#include "fire_tensor_common.h"

template <typename T, uint32_t TILE, bool ROW, bool RAW3 = false, bool DIRECT_STRIDE = false,
		  bool PAIR256 = false, bool VA_PHASE_ADDS = false, bool HALF_ONE8 = false, bool QUAD128 = false,
		  bool F32_QUAD128 = false, uint32_t PAD = 16U>
class KernelFireFrame {
	using Phases = FireFramePhases<T, ROW, RAW3, true>;
	friend struct FireFramePhases<T, ROW, RAW3, true>;
	using Addresses = FireFrameAddressTable<T, TILE, TILE + PAD, RAW3, false>;
	static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
	static constexpr uint32_t FRAME = TILE;
	static constexpr bool ANY_QUAD = QUAD128 || F32_QUAD128;
	static constexpr uint32_t STRIDE = FRAME + PAD;
	static constexpr uint32_t Q = FRAME / 8;
	static constexpr bool RAW3_FRAME_ALLOWED =
		((FRAME == 896 || FRAME == 768) && !ROW) || ((FRAME == 512 || FRAME == 640) && ROW);
	static constexpr bool CENTER_AXPY =
		FRAME == 512 && ROW && RAW3 && DIRECT_STRIDE &&
		(sizeof(T) == sizeof(float) || (PAIR256 && HALF_ONE8 && sizeof(T) == sizeof(half)));
	static constexpr bool SCORE_IN_ONES =
		DIRECT_STRIDE && RAW3 && ROW && FRAME == 512 && sizeof(T) == sizeof(half);
	static constexpr uint32_t SwapXYOffset(uint32_t offset) {
		const uint32_t field = offset / STRIDE;
		const uint32_t lane = offset % STRIDE;
		return CENTER_AXPY
				   ? ((field < 2U ? field + 6U : (field >= 6U && field < 8U ? field - 6U : field)) * STRIDE +
					  lane)
				   : offset;
	}
	FireFrameStorage<T, TILE, PAD> mem;
	AscendC::GlobalTensor<T> rawGm, outGm;
	AscendC::GlobalTensor<float> priorGm;
	// Packed address tables remain read-only after scalar-to-vector synchronization.

	template <uint32_t SLOT, uint32_t PHASE> __aicore__ inline void InitInputPhase() {
		static_assert(PHASE < 6 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1),
					  "valid padded input specialization");
		static_assert(!RAW3 || (RAW3_FRAME_ALLOWED && PHASE < 3), "raw3 verified reuse or whole512 frames");
		uint64_t sourceBase;
		if constexpr (sizeof(T) == sizeof(float))
			sourceBase = reinterpret_cast<uint64_t>(mem.RawFixedSlot(SLOT).GetPhyAddr());
		else
			sourceBase = reinterpret_cast<uint64_t>(mem.RawFloat().GetPhyAddr());
		const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
		const uint64_t oneScoreBase =
			reinterpret_cast<uint64_t>(mem.Ones().GetPhyAddr()) + 64U * sizeof(float);
		uint64_t source[16], target[16];
#pragma unroll
		for (uint32_t lane = 0; lane < 16; ++lane) {
			const uint32_t rawLane = RAW3 ? (lane % 8) * 48 + (lane / 8) * 24 : lane * 48;
			source[lane] = sourceBase + uint64_t(rawLane + 8 * PHASE) * sizeof(float);
			const uint32_t field = (8 * PHASE + lane / 2) % 6;
			const uint32_t row = (8 * PHASE + lane / 2) / 6;
			const uint32_t offset = RAW3 ? (row + (lane % 2) * 4) * Q : row * Q + (lane % 2) * 8;
			target[lane] = RAW3 && SCORE_IN_ONES && field == 5
							   ? oneScoreBase + uint64_t(offset) * sizeof(float)
							   : workBase + uint64_t(SwapXYOffset(field * STRIDE + offset)) * sizeof(float);
		}
		if constexpr (SLOT == 0)
			FireStorePacked(mem.PackedVA()[4 * PHASE], target);
		if constexpr (!VA_PHASE_ADDS || PHASE == 0)
			FireStorePacked(mem.PackedVA()[24 + SLOT * 24 + 4 * PHASE], source);
	}
	template <uint32_t SLOT, uint32_t PHASE> __aicore__ inline void InitOutputPhase() {
		static_assert(PHASE < 5 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1),
					  "valid padded output specialization");
		const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
		uint64_t outputBase;
		if constexpr (sizeof(T) == sizeof(float))
			outputBase = reinterpret_cast<uint64_t>(mem.OutAt(SLOT).GetPhyAddr());
		else
			outputBase = reinterpret_cast<uint64_t>(mem.PackedFloat().GetPhyAddr());
		uint64_t source[16], target[16];
#pragma unroll
		for (uint32_t lane = 0; lane < 16; ++lane) {
			const uint32_t field = (8 * PHASE + lane % 8) % 5;
			const uint32_t mapped = CENTER_AXPY && field < 2 ? field + 6 : field;
			const uint32_t offset = mapped * STRIDE + ((8 * PHASE + lane % 8) / 5) * Q + (lane / 8) * 8;
			source[lane] = workBase + uint64_t(SwapXYOffset(offset)) * sizeof(float);
			target[lane] =
				outputBase + uint64_t((5 * (lane / 2 + (lane % 2) * 8) + PHASE) * 8) * sizeof(float);
		}
		constexpr uint32_t INPUT_WORDS = 72;
		if constexpr (SLOT == 0)
			FireStorePacked(mem.PackedVA()[INPUT_WORDS + 4 * PHASE], source);
		if constexpr (!VA_PHASE_ADDS || PHASE == 0)
			FireStorePacked(mem.PackedVA()[INPUT_WORDS + 20 + SLOT * 20 + 4 * PHASE], target);
	}
	template <uint32_t PHASE> __aicore__ inline void InitPriorPhase() {
		static_assert(PHASE < 5, "native prior hardware phase");
		constexpr uint32_t PRIOR_TARGET = 132;
		constexpr uint32_t PRIOR_SOURCE = 152 + 4 * PHASE;
		const uint64_t sourceBase = reinterpret_cast<uint64_t>(mem.PriorFixedSlot(0).GetPhyAddr());
		const uint64_t workBase = reinterpret_cast<uint64_t>(mem.Work().GetPhyAddr());
		uint64_t source[16], target[16];
#pragma unroll
		for (uint32_t lane = 0; lane < 16; ++lane) {
			const uint32_t priorLane = PAIR256 && !ANY_QUAD ? lane % 8 : lane;
			source[lane] = sourceBase + uint64_t(priorLane * 40 + 8 * PHASE) * sizeof(float);
			const uint32_t field = 6 + (8 * PHASE + lane / 2) % 5;
			const uint32_t step = PAIR256 && !ANY_QUAD ? 32 : 8;
			const uint32_t offset = field * STRIDE + ((8 * PHASE + lane / 2) / 5) * Q + (lane % 2) * step;
			target[lane] = workBase + uint64_t(SwapXYOffset(offset)) * sizeof(float);
		}
		FireStorePacked(mem.PackedVA()[PRIOR_TARGET + 4 * PHASE], target);
		if constexpr (!VA_PHASE_ADDS || PHASE == 0)
			FireStorePacked(mem.PackedVA()[PRIOR_SOURCE], source);
	}
	template <uint32_t PHASE> __aicore__ inline void NativePriorPhase() {
		constexpr uint32_t PRIOR_TARGET = 132;
		constexpr uint32_t PRIOR_SOURCE = 152 + 4 * PHASE;
		if constexpr (ANY_QUAD)
			FireFramePackedTranspose<4, 2, 0>(mem.PackedVA()[PRIOR_TARGET + 4 * PHASE],
											  mem.PackedVA()[PRIOR_SOURCE]);
		else if constexpr (PAIR256)
			FireFramePackedTranspose<4, 1, 40>(mem.PackedVA()[PRIOR_TARGET + 4 * PHASE],
											   mem.PackedVA()[PRIOR_SOURCE]);
		else
			FireFramePackedTranspose<FRAME / 128, 2, 80>(mem.PackedVA()[PRIOR_TARGET + 4 * PHASE],
														 mem.PackedVA()[PRIOR_SOURCE]);
	}
	__aicore__ inline void InitPackedVA() {
		Phases::Init(*this);
		AddAddressPhases();
	}

	template <bool WHOLE> struct RunConstraints {

		static_assert(!WHOLE || ROW, "whole entry requires one batch per core");
		static_assert(!VA_PHASE_ADDS || (WHOLE && ROW && RAW3 && DIRECT_STRIDE && !PAIR256 && FRAME == 512 &&
										 sizeof(T) == sizeof(float)),
					  "fixed float whole phase table");
		static_assert(!F32_QUAD128 || (WHOLE && ROW && RAW3 && DIRECT_STRIDE && !PAIR256 && VA_PHASE_ADDS &&
									   !HALF_ONE8 && !QUAD128 && FRAME == 512 && sizeof(T) == sizeof(float)),
					  "qualified float quad keeps axpy swap va");
		static_assert(!QUAD128 || (PAIR256 && HALF_ONE8), "quad shares complete half pair math");
		static_assert(PAD == 16U || (PAD == 8U && QUAD128 && !F32_QUAD128 && WHOLE && ROW && RAW3 &&
									 DIRECT_STRIDE && PAIR256 && HALF_ONE8 && !VA_PHASE_ADDS &&
									 FRAME == 512U && sizeof(T) == sizeof(half)),
					  "PAD8 only complete half quad128");
		static_assert(!HALF_ONE8 || (WHOLE && ROW && RAW3 && DIRECT_STRIDE && PAIR256 && FRAME == 512 &&
									 sizeof(T) == sizeof(half)),
					  "qualified half pair one broadcast");
		static_assert(!PAIR256 || (WHOLE && ROW && RAW3 && DIRECT_STRIDE && FRAME == 512 &&
								   sizeof(T) == sizeof(half)),
					  "two fixed half batches only");
		static_assert(!DIRECT_STRIDE || (FRAME == 512 && ROW && WHOLE),
					  "direct stride only original whole512");

		static constexpr bool valid = true;
	};

	template <bool WHOLE>
	__aicore__ inline AscendC::LocalTensor<float>
	ActivateBatch(AscendC::LocalTensor<T> r, AscendC::LocalTensor<float> one, uint32_t count) {
		constexpr bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half);
		AscendC::LocalTensor<float> rf;
		if constexpr (rawFirst)
			rf = mem.RawFloat();
		else {
			rf = FirePrepareFrameRaw<T>(mem, r, one, count);
		}

		return rf;
	}

	__aicore__ inline void TransformInput(AscendC::LocalTensor<float> rf) {
		Phases::ReadInput(*this, mem, rf);
	}

	__aicore__ inline void DecodeDirectCoordinates(__ubuf__ float *wPtr) {
		AscendC::SetMaskCount();
		if constexpr (CENTER_AXPY) {
			AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2 * STRIDE);
			asc_mul_scalar(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1),
						   uint16_t(8), uint16_t(8));
		} else {
			AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(4 * STRIDE);
			asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
		}
		AscendC::PipeBarrier<PIPE_V>();
		if constexpr (!CENTER_AXPY)
			AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2 * STRIDE);
		asc_add_scalar(wPtr + (CENTER_AXPY ? 6U : 0U) * STRIDE, wPtr + (CENTER_AXPY ? 6U : 0U) * STRIDE,
					   CENTER_AXPY ? -0.25f : -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8),
					   uint16_t(8));
		asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1),
				uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
		AscendC::PipeBarrier<PIPE_V>();
		if constexpr (CENTER_AXPY)
			asc_axpy(wPtr, wPtr + 6 * STRIDE, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8),
					 uint16_t(8));
		else
			asc_add(wPtr, wPtr, wPtr + 6 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8),
					uint8_t(8), uint8_t(8));
		asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + 9 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1),
				uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
		AscendC::PipeBarrier<PIPE_V>();
	}

	__aicore__ inline void DecodeDirectScale(__ubuf__ float *wPtr) {
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(FRAME);
#pragma unroll
		for (uint32_t k = 0; k < 4; ++k) {
			constexpr uint32_t priorXY = 0;
			const uint32_t field = k < 2 ? k + priorXY : k;
			asc_mul(wPtr + field * STRIDE, wPtr + field * STRIDE, wPtr + 8 * STRIDE, uint8_t(1), uint8_t(1),
					uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
		}
		asc_mul(wPtr + 4 * STRIDE, wPtr + 4 * STRIDE,
				SCORE_IN_ONES ? reinterpret_cast<__ubuf__ float *>(mem.Ones().GetPhyAddr()) + 64U
							  : wPtr + 5 * STRIDE,
				uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
		AscendC::SetMaskNorm();
		AscendC::ResetMask();
	}

	template <bool WHOLE>
	__aicore__ inline void Decode(AscendC::LocalTensor<float> (&f)[11], AscendC::LocalTensor<float> p) {
		if constexpr (WHOLE && DIRECT_STRIDE) {
			auto *wPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
			DecodeDirectCoordinates(wPtr);
			DecodeDirectScale(wPtr);
		} else {
			FireDecodeFrame<FRAME, STRIDE, DIRECT_STRIDE>(f, p);
		}
	}

	__aicore__ inline void TransformOutput() { Phases::WriteOutput(*this, mem); }

	template <bool WHOLE>
	__aicore__ inline void StoreBatch(AscendC::LocalTensor<T> out, uint32_t pos, uint32_t count,
									  const FireCandidateDecodeFusionTilingData &t) {
		const AscendC::DataCopyExtParams op =
			ANY_QUAD  ? AscendC::DataCopyExtParams{4, 128U * 5U * uint32_t(sizeof(T)), 0,
												   (t.n - 128U) * 5U * uint32_t(sizeof(T)), 0}
			: PAIR256 ? AscendC::DataCopyExtParams{2, 256U * 5U * uint32_t(sizeof(T)), 0,
												   (t.n - 256U) * 5U * uint32_t(sizeof(T)), 0}
					  : AscendC::DataCopyExtParams{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
		if constexpr (ANY_QUAD) {
			if constexpr (F32_QUAD128)
				AscendC::DataCopy(outGm[pos * 5], out, AscendC::DataCopyParams{4, 80, 0, 2480});
			else
				AscendC::DataCopy(outGm[pos * 5], out, AscendC::DataCopyParams{4, 40, 0, 1240});
		} else if constexpr (PAIR256) {
			AscendC::DataCopy(outGm[pos * 5], out, AscendC::DataCopyParams{2, 80, 0, 1200});
		} else {
			AscendC::DataCopyPad(outGm[pos * 5], out, op);
		}
	}

	template <bool WHOLE>
	__aicore__ inline void RunBatch(const FireCandidateDecodeFusionTilingData &t, uint32_t block,
									uint32_t batch, uint32_t batchFirst, uint32_t priorPos, uint32_t count,
									AscendC::LocalTensor<T> firstRaw, AscendC::LocalTensor<float> one,
									AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> (&f)[11]) {

		const uint32_t pos = F32_QUAD128 ? priorPos
							 : PAIR256	 ? batchFirst * t.n + priorPos
										 : (WHOLE ? block * FRAME : batch * t.n + priorPos);
		auto r = FireLoadFrameBatch<T>(mem, rawGm, firstRaw, count, pos, batch, batchFirst);
		auto rf = ActivateBatch<WHOLE>(r, one, count);
		TransformInput(rf);
		AscendC::PipeBarrier<PIPE_V>();
		Decode<WHOLE>(f, p);
		auto out = mem.AllocOut();
		AscendC::PipeBarrier<PIPE_V>();
		TransformOutput();
		AscendC::PipeBarrier<PIPE_V>();
		if constexpr (sizeof(T) != sizeof(float))
			AscendC::Cast(out, mem.PackedFloat(), AscendC::RoundMode::CAST_RINT, count * 5);
		mem.OutReady();
		mem.OutWait();
		StoreBatch<WHOLE>(out, pos, count, t);
		mem.FreeOut();
		if constexpr (!(WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half))) {
			mem.FreeRaw();
		}
	}

	template <bool WHOLE>
	__aicore__ inline void LoadFirstFrame(AscendC::LocalTensor<T> &firstRaw, AscendC::LocalTensor<float> &p,
										  const FireCandidateDecodeFusionTilingData &t, uint32_t block,
										  uint32_t batchFirst, uint32_t priorPos, uint32_t count) {
		constexpr bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half);
		firstRaw = mem.AllocRaw();
		p = mem.AllocPrior();
		const AscendC::DataCopyExtParams firstRp =
			ANY_QUAD  ? AscendC::DataCopyExtParams{4, 128U * 6U * uint32_t(sizeof(T)),
												   (t.n - 128U) * 6U * uint32_t(sizeof(T)), 0, 0}
			: PAIR256 ? AscendC::DataCopyExtParams{2, 256U * 6U * uint32_t(sizeof(T)),
												   (t.n - 256U) * 6U * uint32_t(sizeof(T)), 0, 0}
					  : AscendC::DataCopyExtParams{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
		const AscendC::DataCopyExtParams firstPp{1,
												 (ANY_QUAD	? 128U
												  : PAIR256 ? 256U
															: count) *
													 5 * uint32_t(sizeof(float)),
												 0, 0, 0};
		if constexpr (ANY_QUAD) {
			if constexpr (F32_QUAD128)
				AscendC::DataCopy(firstRaw, rawGm[priorPos * 6], AscendC::DataCopyParams{4, 96, 2976, 0});
			else
				AscendC::DataCopy(firstRaw, rawGm[priorPos * 6], AscendC::DataCopyParams{4, 48, 1488, 0});
		} else if constexpr (PAIR256) {
			AscendC::DataCopy(firstRaw, rawGm[(batchFirst * t.n + priorPos) * 6],
							  AscendC::DataCopyParams{2, 96, 1440, 0});
		} else {
			AscendC::DataCopyPad(firstRaw, rawGm[(WHOLE ? block * FRAME : batchFirst * t.n + priorPos) * 6],
								 firstRp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
		}
		if constexpr (rawFirst)
			mem.RawReady();
		AscendC::DataCopyPad(p, priorGm[priorPos * 5], firstPp,
							 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
		if constexpr (!rawFirst)
			mem.RawReady();
		mem.PriorReady();
	}

	template <bool WHOLE>
	__aicore__ inline void EarlyActivation(AscendC::LocalTensor<T> firstRaw, AscendC::LocalTensor<float> one,
										   uint32_t count) {
		constexpr bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half);
		if constexpr (rawFirst) {
			mem.RawWait();
			auto rf = mem.RawFloat();
			AscendC::Cast(rf, firstRaw, AscendC::RoundMode::CAST_NONE, count * 6);
			AscendC::Muls(rf, rf, -1.0f, count * 6);
			AscendC::Exp(rf, rf, count * 6);
			AscendC::Adds(rf, rf, 1.0f, count * 6);
			if constexpr (HALF_ONE8) {
				const AscendC::BinaryRepeatParams divOne8{1, 0, 1, 8, 0, 8};
				AscendC::Div<float>(rf, one, rf, uint64_t(64), uint8_t(48), divOne8);
			} else
				AscendC::Div(rf, one, rf, count * 6);
			AscendC::PipeBarrier<PIPE_V>();
		}
	}

	template <bool WHOLE>
	__aicore__ inline void PreparePrior(AscendC::LocalTensor<float> (&f)[11], AscendC::LocalTensor<float> p) {
		mem.PriorWait();
		Phases::template Prior<false>(*this);
		if constexpr (!(WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half)))
			AscendC::PipeBarrier<PIPE_V>();
		if constexpr (!DIRECT_STRIDE) {
			// The original prior AoS has been consumed; copy stride bits exactly.
			FireCopyPriorScale<STRIDE>(p, f[8]);
		}
	}

	__aicore__ inline uint32_t FirstCandidate(uint32_t block, uint32_t groupLanes, uint32_t perCore) {
		if constexpr (ANY_QUAD)
			return block * 128U;
		if constexpr (PAIR256)
			return (block % 16U) * 256U;
		if (groupLanes == 0U)
			return uint32_t(-1);
		return (block % groupLanes) * perCore;
	}
	__aicore__ inline uint32_t FirstBatch(uint32_t block, uint32_t groupLanes, uint32_t batchCount) {
		if constexpr (ANY_QUAD)
			return 0U;
		if constexpr (PAIR256)
			return (block / 16U) * 2U;
		if (groupLanes == 0U)
			return 0U;
		return (block / groupLanes) * batchCount;
	}

	__aicore__ inline void AddAddressPhases() {
		if constexpr (VA_PHASE_ADDS) {
			auto *entries = reinterpret_cast<__ubuf__ int16_t *>(mem.PackedVA().GetPhyAddr());
			AscendC::SetMaskCount();
			AscendC::SetVectorMask<int16_t, AscendC::MaskMode::COUNTER>(16U);
#pragma unroll
			for (uint32_t phase = 1U; phase < 3U; ++phase)
				asc_add_scalar(entries + (24U + 4U * phase) * 4U, entries + 24U * 4U, int16_t(phase),
							   uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
#pragma unroll
			for (uint32_t phase = 1U; phase < 5U; ++phase)
				asc_add_scalar(entries + (92U + 4U * phase) * 4U, entries + 92U * 4U, int16_t(phase),
							   uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
#pragma unroll
			for (uint32_t phase = 1U; phase < 5U; ++phase)
				asc_add_scalar(entries + (152U + 4U * phase) * 4U, entries + 152U * 4U, int16_t(phase),
							   uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
			AscendC::PipeBarrier<PIPE_V>();
			AscendC::SetMaskNorm();
			AscendC::ResetMask();
		}
	}

  public:
	template <bool WHOLE = false>
	__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
							   const FireFrameTilingData &schedule) {
		static_assert(RunConstraints<WHOLE>::valid, "valid frame configuration");
		constexpr bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half);
		const auto &t = schedule.header;
		rawGm.SetGlobalBuffer((__gm__ T *)raw);
		priorGm.SetGlobalBuffer((__gm__ float *)priors);
		outGm.SetGlobalBuffer((__gm__ T *)decoded);
		const uint32_t groupLanes = schedule.geometry & 0xffffU;
		const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
		const uint32_t begin = FirstCandidate(block, groupLanes, t.perCore);
		if (begin >= t.n)
			return;
		const uint32_t batchCount = ROW ? 1U : schedule.geometry >> 16;
		const uint32_t batchFirst = FirstBatch(block, groupLanes, batchCount);
		const uint32_t priorPos = begin;
		const uint32_t count = WHOLE ? FRAME : (t.n - priorPos < t.perCore ? t.n - priorPos : t.perCore);
		// Overlap first raw/prior DMA with address-table setup.
		AscendC::LocalTensor<T> firstRaw;
		AscendC::LocalTensor<float> p;
		LoadFirstFrame<WHOLE>(firstRaw, p, t, block, batchFirst, priorPos, count);
		auto one = mem.Ones();
		AscendC::Duplicate(one, 1.0f, HALF_ONE8 ? 8U : FRAME * 6);
		AscendC::LocalTensor<float> f[11];
		for (uint32_t k = 0; k < 11; ++k)
			f[k] = mem.Work()[k * STRIDE];
		EarlyActivation<WHOLE>(firstRaw, one, count);
		InitPackedVA();
		{
			PreparePrior<WHOLE>(f, p);
			// Cached f[6..10] are read-only throughout the batch loop.
			for (uint32_t batch = batchFirst; batch < batchFirst + batchCount; ++batch) {
				RunBatch<WHOLE>(t, block, batch, batchFirst, priorPos, count, firstRaw, one, p, f);
			}
			if constexpr (!rawFirst) {
				mem.FreePrior();
			}
		}
		mem.Drain();
	}
};

// Flatten two complete 832-candidate rows into one frame.
