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
#include "fire_storage.h"

#include "fire_pair_common.h"

template <typename T, uint32_t TILE, bool ROW, bool RAW3 = false, bool DIRECT_STRIDE = false,
		  uint32_t QUALIFIED_ROW_N = 0U, bool VA_GROUP28 = false, bool ROW_CACHED_STRIDE = false,
		  bool H10 = false>
class KernelFireA2Rows {
	using Phases = FireFramePhases<T, ROW, RAW3>;
	friend struct FireFramePhases<T, ROW, RAW3>;
	using Addresses = FireFrameAddressTable<T, TILE, TILE + 16U, RAW3, VA_GROUP28>;
	static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
	static constexpr uint32_t FRAME = TILE;
	static constexpr uint32_t STRIDE = FRAME + 16;
	static constexpr uint32_t Q = FRAME / 8;
	static_assert(H10 || QUALIFIED_ROW_N == 0U ||
					  (QUALIFIED_ROW_N == 4097U && sizeof(T) == sizeof(float) && FRAME == 896U && ROW &&
					   RAW3 && !DIRECT_STRIDE),
				  "qualified FP32 P4 C20 ROW896 N4097");
	static_assert(H10 || !VA_GROUP28 ||
					  (sizeof(T) == sizeof(float) && FRAME == 896U && ROW && RAW3 && !DIRECT_STRIDE &&
					   QUALIFIED_ROW_N == 4097U),
				  "qualified FP32 P4 M20 ROW896 VA28");
	static_assert(H10 || !ROW_CACHED_STRIDE ||
					  (VA_GROUP28 && sizeof(T) == sizeof(float) && FRAME == 896U && STRIDE == 912U && ROW &&
					   RAW3 && !DIRECT_STRIDE && QUALIFIED_ROW_N == 4097U),
				  "qualified FP32 P4 cached stride ROW896");
	static_assert(!H10 || !VA_GROUP28 ||
					  (sizeof(T) == sizeof(half) && FRAME == 896U && ROW && RAW3 && !DIRECT_STRIDE &&
					   QUALIFIED_ROW_N == 4097U),
				  "qualified P10 half C20 N4097 ROW896 raw3 VA28");
	static_assert(!H10 || !ROW_CACHED_STRIDE ||
					  (VA_GROUP28 && sizeof(T) == sizeof(half) && FRAME == 896U && STRIDE == 912U && ROW &&
					   RAW3 && !DIRECT_STRIDE && QUALIFIED_ROW_N == 4097U),
				  "qualified P10 cached stride ROW896");
	static constexpr bool RAW3_REUSE_FRAME =
		((FRAME == 896 || FRAME == 768) && !ROW) || ((FRAME == 512 || FRAME == 640) && ROW);
	static constexpr bool F4_RAW3_ROW =
		QUALIFIED_ROW_N == 4097U && FRAME == 896U && ROW && sizeof(T) == sizeof(float);
	static constexpr bool H10_RAW3_ROW =
		VA_GROUP28 && QUALIFIED_ROW_N == 4097U && FRAME == 896U && ROW && sizeof(T) == sizeof(half);
	FireA2RowsStorage<T, TILE, VA_GROUP28, H10> mem;
	template <uint32_t OFFSET> __aicore__ static constexpr uint32_t VAOffset() {
		return FireFrameVAOffset<VA_GROUP28, OFFSET, 28U>();
	}
	AscendC::GlobalTensor<T> rawGm, outGm;
	AscendC::GlobalTensor<float> priorGm;

	template <uint32_t SLOT, uint32_t PHASE> __aicore__ inline void InitInputPhase() {
		static_assert(PHASE < 6 && SLOT < (sizeof(T) == sizeof(float) ? 2 : 1),
					  "valid padded input specialization");
		static_assert(H10 || !RAW3 || ((RAW3_REUSE_FRAME || F4_RAW3_ROW) && PHASE < 3),
					  "raw3 verified reuse or qualified C20 ROW896");
		static_assert(!H10 || !RAW3 || ((RAW3_REUSE_FRAME || H10_RAW3_ROW) && PHASE < 3),
					  "raw3 original or qualified halfC20 ROW896");
		Addresses::template Input<SLOT, PHASE>(mem);
	}

	__aicore__ inline void InitPackedVA() {
		Phases::Init(*this);
		if constexpr (VA_GROUP28) {
			auto *entries = reinterpret_cast<__ubuf__ int16_t *>(mem.PackedVA().GetPhyAddr());
			AscendC::SetMaskCount();
			AscendC::SetVectorMask<int16_t, AscendC::MaskMode::COUNTER>(48U);
#pragma unroll
			for (uint32_t phase = 1U; phase < 5U; ++phase)
				asc_add_scalar(entries + phase * 48U, entries, int16_t(phase), uint8_t(1), uint16_t(1),
							   uint16_t(1), uint16_t(8), uint16_t(8));
			AscendC::PipeBarrier<PIPE_V>();
			AscendC::SetMaskNorm();
			AscendC::ResetMask();
		}
	}

	template <bool WHOLE> __aicore__ inline void ValidateF4Run() {
		static_assert(QUALIFIED_ROW_N == 0U || !WHOLE, "qualified C20 nonwhole only");
		constexpr bool rawFirstRow = ROW_CACHED_STRIDE && QUALIFIED_ROW_N == 4097U;
		static_assert(!rawFirstRow || (!WHOLE && ROW && RAW3 && !DIRECT_STRIDE && VA_GROUP28 &&
									   FRAME == 896U && STRIDE == 912U && sizeof(T) == sizeof(float)),
					  "qualified F4 C20 raw before VA only");
	}
	template <bool WHOLE> __aicore__ inline void ValidateH10Run() {
		static_assert(!VA_GROUP28 || !WHOLE, "qualified P10 VA28 nonwhole only");
		constexpr bool rawFirstRow = ROW_CACHED_STRIDE && QUALIFIED_ROW_N == 4097U;
		static_assert(!rawFirstRow || (!WHOLE && ROW && RAW3 && !DIRECT_STRIDE && VA_GROUP28 &&
									   FRAME == 896U && STRIDE == 912U && sizeof(T) == sizeof(half)),
					  "qualified H10 C20 Cast raw before VA only");
	}
	template <bool WHOLE> __aicore__ inline void ValidateRun() {
		static_assert(!WHOLE || ROW, "whole entry requires one batch per core");
		static_assert(!DIRECT_STRIDE || (FRAME == 512 && ROW && WHOLE),
					  "direct stride only original whole512");
		if constexpr (H10)
			ValidateH10Run<WHOLE>();
		else
			ValidateF4Run<WHOLE>();
	}
	template <bool WHOLE>
	__aicore__ inline void LoadFirst(AscendC::LocalTensor<T> firstRaw, AscendC::LocalTensor<float> p,
									 uint32_t count, uint32_t block, uint32_t batchFirst, uint32_t n,
									 uint32_t priorPos) {
		constexpr bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half);
		constexpr bool rawFirstRow = ROW_CACHED_STRIDE && QUALIFIED_ROW_N == 4097U;
		AscendC::DataCopyExtParams firstRp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
		AscendC::DataCopyExtParams firstPp{1, count * 5 * uint32_t(sizeof(float)), 0, 0, 0};
		AscendC::DataCopyPad(firstRaw, rawGm[(WHOLE ? block * FRAME : batchFirst * n + priorPos) * 6],
							 firstRp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
		if constexpr (rawFirst || rawFirstRow)
			mem.RawReady();
		AscendC::DataCopyPad(p, priorGm[priorPos * 5], firstPp,
							 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
		if constexpr (!(rawFirst || rawFirstRow))
			mem.RawReady();
		mem.PriorReady();
	}
	template <bool WHOLE>
	__aicore__ inline void PreprocessEarlyRaw(AscendC::LocalTensor<T> firstRaw,
											  AscendC::LocalTensor<float> one, uint32_t count,
											  bool one8Full) {
		constexpr bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half);
		constexpr bool rawFirstRow = ROW_CACHED_STRIDE && QUALIFIED_ROW_N == 4097U;
		if constexpr (rawFirst) {
			mem.RawWait();
			auto rf = mem.RawFloat();
			AscendC::Cast(rf, firstRaw, AscendC::RoundMode::CAST_NONE, count * 6);
			FireActivateRaw(rf, one, count * 6);
		}
		if constexpr (rawFirstRow) {
			mem.RawWait();
			AscendC::LocalTensor<float> rf;
			if constexpr (H10) {
				rf = mem.RawFloat();
				AscendC::Cast(rf, firstRaw, AscendC::RoundMode::CAST_NONE, count * 6);
			} else
				rf = firstRaw.template ReinterpretCast<float>();
			FireActivateOne8Raw(rf, one, count * 6, one8Full);
		}
	}
	__aicore__ inline void PreparePrior(AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> *f) {
		mem.PriorWait();
		Phases::template Prior<false>(*this);
		AscendC::PipeBarrier<PIPE_V>();
		if constexpr (!DIRECT_STRIDE && !ROW_CACHED_STRIDE) {
			FireCopyPriorScale<STRIDE>(p, f[8]);
		}
	}
	template <bool WHOLE>
	__aicore__ inline AscendC::LocalTensor<float>
	PrepareRaw(AscendC::LocalTensor<T> r, AscendC::LocalTensor<float> one, uint32_t count, bool one8Full) {
		constexpr bool rawFirst = WHOLE && DIRECT_STRIDE && RAW3 && FRAME == 512 && sizeof(T) == sizeof(half);
		constexpr bool rawFirstRow = ROW_CACHED_STRIDE && QUALIFIED_ROW_N == 4097U;
		AscendC::LocalTensor<float> rf;
		if constexpr (rawFirstRow && !H10)
			rf = r.template ReinterpretCast<float>();
		else if constexpr (rawFirst || rawFirstRow)
			rf = mem.RawFloat();
		else {
			rf = FirePrepareFrameRaw<T, true>(mem, r, one, count, one8Full);
		}
		return rf;
	}

	__aicore__ inline void TransposeInput(AscendC::LocalTensor<float> rf) {
		Phases::ReadInput(*this, mem, rf);
	}
	template <bool WHOLE> __aicore__ inline void ComputeCachedStride() {
		static_assert(H10 || !WHOLE && ROW && FRAME == 896U && STRIDE == 912U && VA_GROUP28 &&
								 QUALIFIED_ROW_N == 4097U && sizeof(T) == sizeof(float),
					  "qualified FP32 P4 cached stride run");
		static_assert(!H10 || !WHOLE && ROW && FRAME == 896U && STRIDE == 912U && VA_GROUP28 &&
								  QUALIFIED_ROW_N == 4097U && sizeof(T) == sizeof(half),
					  "qualified P10 cached stride run");
		auto *wPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
		AscendC::SetMaskCount();
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(4U * STRIDE);
		asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
		AscendC::PipeBarrier<PIPE_V>();
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2U * STRIDE);
		asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
		if constexpr (H10) {
			asc_mul(wPtr + 2U * STRIDE, wPtr + 2U * STRIDE, wPtr + 2U * STRIDE, uint8_t(1), uint8_t(1),
					uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
			AscendC::PipeBarrier<PIPE_V>();
			asc_add(wPtr, wPtr, wPtr + 6U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1),
					uint8_t(8), uint8_t(8), uint8_t(8));
		} else {
			AscendC::PipeBarrier<PIPE_V>();
			asc_add(wPtr, wPtr, wPtr + 6U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1),
					uint8_t(8), uint8_t(8), uint8_t(8));
			AscendC::PipeBarrier<PIPE_V>();
			asc_mul(wPtr + 2U * STRIDE, wPtr + 2U * STRIDE, wPtr + 2U * STRIDE, uint8_t(1), uint8_t(1),
					uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
			AscendC::PipeBarrier<PIPE_V>();
		}
		asc_mul(wPtr + 2U * STRIDE, wPtr + 2U * STRIDE, wPtr + 9U * STRIDE, uint8_t(1), uint8_t(1),
				uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
		AscendC::PipeBarrier<PIPE_V>();
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(FRAME);
#pragma unroll
		for (uint32_t k = 0; k < 4U; ++k)
			asc_mul(wPtr + k * STRIDE, wPtr + k * STRIDE, wPtr + 8U * STRIDE, uint8_t(1), uint8_t(1),
					uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
		if constexpr (!H10)
			AscendC::PipeBarrier<PIPE_V>();
		asc_mul(wPtr + 4U * STRIDE, wPtr + 4U * STRIDE, wPtr + 5U * STRIDE, uint8_t(1), uint8_t(1),
				uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
		AscendC::SetMaskNorm();
	}
	__aicore__ inline void ComputeWholeStride() { FireDecodeWholeFrame<FRAME, STRIDE>(mem.Work()); }
	__aicore__ inline void ComputeRegular(AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> *f) {
		FireDecodeFrame<FRAME, STRIDE, DIRECT_STRIDE>(f, p);
	}
	template <bool WHOLE>
	__aicore__ inline void Compute(AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> *f) {
		if constexpr (ROW_CACHED_STRIDE)
			ComputeCachedStride<WHOLE>();
		else if constexpr (WHOLE && DIRECT_STRIDE)
			ComputeWholeStride();
		else
			ComputeRegular(p, f);
	}

	__aicore__ inline void TransposeOutput() { Phases::WriteOutput(*this, mem); }
	__aicore__ inline void StoreOutput(uint32_t count, uint32_t pos) {
		auto out = mem.AllocOut();
		AscendC::PipeBarrier<PIPE_V>();
		TransposeOutput();
		AscendC::PipeBarrier<PIPE_V>();
		if constexpr (sizeof(T) != sizeof(float)) {
			if constexpr (H10 && VA_GROUP28) {
				static_assert(sizeof(T) == sizeof(half) && FRAME == 896U && ROW && RAW3 && !DIRECT_STRIDE &&
								  QUALIFIED_ROW_N == 4097U,
							  "qualified C20H10 output RN count");
				asc_float2half_rn(reinterpret_cast<__ubuf__ half *>(out.GetPhyAddr()),
								  reinterpret_cast<__ubuf__ float *>(mem.PackedFloat().GetPhyAddr()),
								  uint32_t(count * 5U));
			} else
				AscendC::Cast(out, mem.PackedFloat(), AscendC::RoundMode::CAST_RINT, count * 5);
		}
		FireStoreDecodedFrame<T>(mem, outGm, out, count, pos);
	}
	template <bool WHOLE>
	__aicore__ inline void ProcessBatch(AscendC::LocalTensor<T> firstRaw, AscendC::LocalTensor<float> one,
										AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> *f,
										uint32_t count, bool one8Full, uint32_t pos, uint32_t batch,
										uint32_t batchFirst) {
		auto r = FireLoadFrameBatch<T>(mem, rawGm, firstRaw, count, pos, batch, batchFirst);
		auto rf = PrepareRaw<WHOLE>(r, one, count, one8Full);
		TransposeInput(rf);
		AscendC::PipeBarrier<PIPE_V>();
		Compute<WHOLE>(p, f);
		StoreOutput(count, pos);
	}

  public:
	template <bool WHOLE = false>
	__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
							   const FireFrameTilingData &schedule) {
		ValidateRun<WHOLE>();
		const auto &t = schedule.header;
		rawGm.SetGlobalBuffer((__gm__ T *)raw);
		priorGm.SetGlobalBuffer((__gm__ float *)priors);
		outGm.SetGlobalBuffer((__gm__ T *)decoded);
		FireFrameRunSchedule<FRAME, ROW, WHOLE> plan;
		if (!plan.Init(schedule))
			return;
		const uint32_t block = plan.block, priorPos = plan.begin, count = plan.count;
		const uint32_t batchCount = plan.batchCount, batchFirst = plan.batchFirst;
		const bool one8Full = ROW_CACHED_STRIDE && QUALIFIED_ROW_N == 4097U && count == 832U;
		auto firstRaw = mem.AllocRaw();
		auto p = mem.AllocPrior();
		LoadFirst<WHOLE>(firstRaw, p, count, block, batchFirst, t.n, priorPos);
		auto one = mem.Ones();
		AscendC::Duplicate(one, 1.0f, one8Full ? 8U : FRAME * 6);
		AscendC::LocalTensor<float> f[11];
		for (uint32_t k = 0; k < 11; ++k)
			f[k] = mem.Work()[k * STRIDE];
		PreprocessEarlyRaw<WHOLE>(firstRaw, one, count, one8Full);
		InitPackedVA();
		{
			PreparePrior(p, f);
			for (uint32_t batch = batchFirst; batch < batchFirst + batchCount; ++batch) {
				const uint32_t pos = WHOLE ? block * FRAME : batch * t.n + priorPos;
				ProcessBatch<WHOLE>(firstRaw, one, p, f, count, one8Full, pos, batch, batchFirst);
			}
			mem.FreePrior();
		}
		mem.Drain();
	}
};

template <typename T, uint32_t TILE, bool VA_GROUP28 = false>
using FireA2RowF4Storage = FireA2RowsStorage<T, TILE, VA_GROUP28, false>;
template <typename T, uint32_t TILE, bool VA_GROUP28 = false>
using FireA2RowH10Storage = FireA2RowsStorage<T, TILE, VA_GROUP28, true>;
template <typename T, uint32_t TILE, bool ROW, bool RAW3 = false, bool DIRECT_STRIDE = false,
		  uint32_t QUALIFIED_ROW_N = 0U, bool VA_GROUP28 = false, bool ROW_CACHED_STRIDE = false>
using KernelFireA2RowF4 = KernelFireA2Rows<T, TILE, ROW, RAW3, DIRECT_STRIDE, QUALIFIED_ROW_N, VA_GROUP28,
										   ROW_CACHED_STRIDE, false>;
template <typename T, uint32_t TILE, bool ROW, bool RAW3 = false, bool DIRECT_STRIDE = false,
		  bool VA_GROUP28 = false, uint32_t QUALIFIED_N = 0U, bool ROW_CACHED_STRIDE = false>
using KernelFireA2RowH10 =
	KernelFireA2Rows<T, TILE, ROW, RAW3, DIRECT_STRIDE, QUALIFIED_N, VA_GROUP28, ROW_CACHED_STRIDE, true>;
