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

#include "fire_frame_schedule.h"
#include "fire_frame_addresses.h"
#include "fire_frame_io.h"
#include "fire_frame_phases.h"
#include "fire_frame_math.h"
#include "fire_pair_common.h"

template <typename T, uint32_t TILE> using FireFrameStorageStock1112 = FirePairFrameStorage<T, TILE>;

template <typename T, uint32_t TILE, bool ROW, bool RAW3 = false, bool DIRECT_STRIDE = false,
		  uint32_t QUALIFIED_TAIL_N = 0U>
class KernelFireFrameStock1112 {
	using Phases = FireFramePhases<T, ROW, RAW3>;
	friend struct FireFramePhases<T, ROW, RAW3>;
	using Addresses = FireFrameAddressTable<T, TILE, TILE + 16U, RAW3, false>;
	static_assert(QUALIFIED_TAIL_N == 0U ||
					  ((QUALIFIED_TAIL_N == 32768U || QUALIFIED_TAIL_N == 32769U) &&
					   sizeof(T) == sizeof(half) && TILE == 384U && !ROW && RAW3 && !DIRECT_STRIDE),
				  "qualified N32769 half partial tail384");
	static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
	static constexpr uint32_t FRAME = TILE;
	static constexpr uint32_t STRIDE = FRAME + 16;
	static constexpr uint32_t Q = FRAME / 8;
	FireFrameStorageStock1112<T, TILE> mem;
	AscendC::GlobalTensor<T> rawGm, outGm;
	AscendC::GlobalTensor<float> priorGm;
	// Packed address tables remain read-only after scalar-to-vector synchronization.

	template <uint32_t SLOT, uint32_t PHASE> __aicore__ inline void InitInputPhase() {
		static_assert(PHASE < 6 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1),
					  "valid padded input specialization");
		static_assert(!RAW3 ||
						  ((((FRAME == 896 || FRAME == 768 ||
							  (FRAME == 384 && (QUALIFIED_TAIL_N == 32768U || QUALIFIED_TAIL_N == 32769U))) &&
							 !ROW) ||
							((FRAME == 512 || FRAME == 640) && ROW)) &&
						   PHASE < 3),
					  "raw3 verified reuse or qualified tail frames");
		Addresses::template Input<SLOT, PHASE>(mem);
	}

	__aicore__ inline void InitPackedVA() { Phases::Init(*this); }

	template <bool WHOLE,
			  bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half)>
	__aicore__ inline void TransformFirstRaw(AscendC::LocalTensor<T> firstRaw,
											 AscendC::LocalTensor<float> one, uint32_t count) {
		if constexpr (rawFirst) {
			mem.RawWait();
			auto rf = mem.RawFloat();
			AscendC::Cast(rf, firstRaw, AscendC::RoundMode::CAST_NONE, count * 6);
			FireActivateRaw(rf, one, count * 6);
		}
	}

	__aicore__ inline void PreparePrior(AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> *f) {
		mem.PriorWait();
		Phases::template Prior<false>(*this);
		AscendC::PipeBarrier<PIPE_V>();
		if constexpr (!DIRECT_STRIDE) {
			// The original prior AoS has been consumed; copy stride bits exactly.
			FireCopyPriorScale<STRIDE>(p, f[8]);
		}
	}

	__aicore__ inline void TransposeInput(AscendC::LocalTensor<float> rf) {
		Phases::ReadInput(*this, mem, rf);
		AscendC::PipeBarrier<PIPE_V>();
	}

	template <bool WHOLE>
	__aicore__ inline void DecodeCoordinates(AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> *f) {
		if constexpr (WHOLE && DIRECT_STRIDE) {
			FireDecodeWholeFrame<FRAME, STRIDE>(mem.Work());
		} else {
			FireDecodeFrame<FRAME, STRIDE, DIRECT_STRIDE>(f, p);
		}
	}

	__aicore__ inline void TransposeOutput() {
		Phases::WriteOutput(*this, mem);
		AscendC::PipeBarrier<PIPE_V>();
	}

	__aicore__ inline void StoreBatch(uint32_t count, uint32_t pos) {
		auto out = mem.AllocOut();
		AscendC::PipeBarrier<PIPE_V>();
		TransposeOutput();
		if constexpr (sizeof(T) != sizeof(float))
			AscendC::Cast(out, mem.PackedFloat(), AscendC::RoundMode::CAST_RINT, count * 5);
		FireStoreDecodedFrame<T>(mem, outGm, out, count, pos);
	}

	template <bool WHOLE,
			  bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half)>
	__aicore__ inline void TransformBatch(uint32_t batch, uint32_t batchFirst, uint32_t count, uint32_t pos,
										  AscendC::LocalTensor<T> firstRaw, AscendC::LocalTensor<float> one) {
		auto r = FireLoadFrameBatch<T>(mem, rawGm, firstRaw, count, pos, batch, batchFirst);
		AscendC::LocalTensor<float> rf;
		if constexpr (rawFirst)
			rf = mem.RawFloat();
		else {
			rf = FirePrepareFrameRaw<T>(mem, r, one, count);
		}
		TransposeInput(rf);
	}

	template <bool WHOLE>
	__aicore__ inline void ProcessBatch(uint32_t batch, uint32_t batchFirst, uint32_t count, uint32_t pos,
										AscendC::LocalTensor<T> firstRaw, AscendC::LocalTensor<float> p,
										AscendC::LocalTensor<float> one, AscendC::LocalTensor<float> *f) {
		TransformBatch<WHOLE>(batch, batchFirst, count, pos, firstRaw, one);
		DecodeCoordinates<WHOLE>(p, f);
		StoreBatch(count, pos);
	}

	template <bool WHOLE,
			  bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half)>
	__aicore__ inline void LoadInitialPrior(AscendC::LocalTensor<float> p, uint32_t priorPos,
											const AscendC::DataCopyExtParams &firstPp) {
		if constexpr (rawFirst)
			mem.RawReady();
		AscendC::DataCopyPad(p, priorGm[priorPos * 5], firstPp,
							 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
		if constexpr (!rawFirst)
			mem.RawReady();
		mem.PriorReady();
	}

  public:
	template <bool WHOLE = false>
	__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
							   const FireFrameTilingData &schedule) {
		static_assert(!WHOLE || ROW, "whole entry requires one batch per core");
		static_assert(QUALIFIED_TAIL_N == 0U || !WHOLE, "qualified tail remains partial");
		static_assert(!DIRECT_STRIDE || (FRAME == 512 && ROW && WHOLE),
					  "direct stride only original whole512");
		const auto &t = schedule.header;
		rawGm.SetGlobalBuffer((__gm__ T *)raw);
		priorGm.SetGlobalBuffer((__gm__ float *)priors);
		outGm.SetGlobalBuffer((__gm__ T *)decoded);
		FireFrameRunSchedule<FRAME, ROW, WHOLE> plan;
		if (!plan.Init(schedule))
			return;
		const uint32_t block = plan.block, priorPos = plan.begin, count = plan.count;
		const uint32_t batchCount = plan.batchCount, batchFirst = plan.batchFirst;
		// Overlap first raw/prior DMA with address-table setup.
		auto firstRaw = mem.AllocRaw();
		auto p = mem.AllocPrior();
		AscendC::DataCopyExtParams firstRp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
		AscendC::DataCopyExtParams firstPp{1, count * 5 * uint32_t(sizeof(float)), 0, 0, 0};
		AscendC::DataCopyPad(firstRaw, rawGm[(WHOLE ? block * FRAME : batchFirst * t.n + priorPos) * 6],
							 firstRp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
		LoadInitialPrior<WHOLE>(p, priorPos, firstPp);
		auto one = mem.Ones();
		AscendC::Duplicate(one, 1.0f, FRAME * 6);
		AscendC::LocalTensor<float> f[11];
		for (uint32_t k = 0; k < 11; ++k)
			f[k] = mem.Work()[k * STRIDE];
		TransformFirstRaw<WHOLE>(firstRaw, one, count);
		InitPackedVA();
		{
			PreparePrior(p, f);
			// Cached f[6..10] are read-only throughout the batch loop.
			for (uint32_t batch = batchFirst; batch < batchFirst + batchCount; ++batch) {
				const uint32_t pos = WHOLE ? block * FRAME : batch * t.n + priorPos;
				ProcessBatch<WHOLE>(batch, batchFirst, count, pos, firstRaw, p, one, f);
			}
			mem.FreePrior();
		}
		mem.Drain();
	}
};
