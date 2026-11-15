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

#include "fire_activation.h"
#include "fire_storage.h"

// Work is dead until the first raw/prior Gather; seed and broadcast areas are disjoint.
template <uint32_t TILE>
__aicore__ inline void
MakeFireOutputOffsets(AscendC::LocalTensor<int32_t> dst, AscendC::LocalTensor<int32_t> rawOffsets,
					  AscendC::LocalTensor<int32_t> priorOffsets, AscendC::LocalTensor<int32_t> scratch) {
	constexpr uint32_t STRIDE = TILE + 8;
	auto seed = dst;
	if constexpr (TILE > 256)
		seed = scratch;
	for (int32_t j = 0; j < 8; ++j)
		for (int32_t k = 0; k < 5; ++k)
			seed.SetValue(j * 5 + k, (k * int32_t(STRIDE) + j) * 4);
	constexpr int32_t event = 2;
	AscendC::SetFlag<AscendC::HardEvent::S_V>(event);
	AscendC::WaitFlag<AscendC::HardEvent::S_V>(event);
	if constexpr (TILE > 256) {
		auto blockRamp = scratch[64];
		auto blockBroadcast = scratch[64 + TILE / 8];
		AscendC::Sub<int32_t>(blockRamp, rawOffsets, priorOffsets, TILE / 8);
		AscendC::Muls<int32_t>(blockRamp, blockRamp, int32_t(8), TILE / 8);
		AscendC::Brcb<int32_t>(blockBroadcast, blockRamp, uint8_t(TILE / 64), {1, 8});
		AscendC::Add<int32_t>(dst, seed, blockBroadcast, uint64_t(40), uint8_t(TILE / 8), {1, 1, 0, 5, 0, 1});
	} else {
		for (uint32_t width = 8; width < TILE; width *= 2) {
			const uint32_t count = width < TILE - width ? width : TILE - width;
			AscendC::Adds(dst[width * 5], dst, int32_t(width * 4), count * 5);
			AscendC::PipeBarrier<PIPE_V>();
		}
	}
}

template <typename T, uint32_t TILE, bool ROW = false> class KernelFire {
	static constexpr uint32_t DEPTH = TILE > 256 ? 2 : 1;
	static constexpr uint32_t STRIDE = TILE + 8;
	FireStaticStorage<T, TILE> mem;
	AscendC::GlobalTensor<T> rawGm, outGm;
	AscendC::GlobalTensor<float> priorGm;
	template <typename V, uint32_t COUNT, uint32_t SEED_EVENT>
	__aicore__ inline void MakeOffsets(AscendC::LocalTensor<V> dst, V step) {
		for (int32_t j = 0; j < 8; ++j)
			dst.SetValue(j, V(j) * step);
		constexpr int32_t event = SEED_EVENT;
		AscendC::SetFlag<AscendC::HardEvent::S_V>(event);
		AscendC::WaitFlag<AscendC::HardEvent::S_V>(event);
		for (uint32_t width = 8; width < COUNT; width *= 2) {
			const uint32_t count = width < COUNT - width ? width : COUNT - width;
			AscendC::Adds(dst[width], dst, V(int32_t(width)) * step, count);
			AscendC::PipeBarrier<PIPE_V>();
		}
	}

	__aicore__ inline void DecodeCoordinates(AscendC::LocalTensor<float> (&f)[11], uint32_t count) {
		for (uint32_t k = 0; k < 2; ++k) {
			AscendC::Muls(f[k], f[k], 2.0f, count);
			AscendC::Adds(f[k], f[k], -0.5f, count);
			AscendC::Add(f[k], f[k], f[k + 6], count);
			AscendC::Mul(f[k], f[k], f[8], count);
			AscendC::Muls(f[k + 2], f[k + 2], 2.0f, count);
			AscendC::Mul(f[k + 2], f[k + 2], f[k + 2], count);
			AscendC::Mul(f[k + 2], f[k + 2], f[k + 9], count);
			AscendC::Mul(f[k + 2], f[k + 2], f[8], count);
		}
		AscendC::Mul(f[4], f[4], f[5], count);
	}
	__aicore__ inline void Decode(uint32_t count, uint32_t pos, AscendC::LocalTensor<T> r,
								  AscendC::LocalTensor<float> p, AscendC::LocalTensor<int32_t> rawOffsets,
								  AscendC::LocalTensor<int32_t> priorOffsets,
								  AscendC::LocalTensor<int32_t> outOffsets, AscendC::LocalTensor<float> one) {
		AscendC::LocalTensor<float> f[11];
		for (uint32_t k = 0; k < 11; ++k)
			f[k] = mem.Work()[k * STRIDE];
		AscendC::LocalTensor<float> rf;
		if constexpr (sizeof(T) == sizeof(float))
			rf = r.template ReinterpretCast<float>();
		else {
			rf = mem.RawFloat();
			AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
		}
		FireExpDenominator(rf, count * 6);
		AscendC::Div(rf, one, rf, count * 6);
		for (uint32_t k = 0; k < 6; ++k)
			AscendC::Gather(f[k], rf, rawOffsets.template ReinterpretCast<uint32_t>(), k * 4, count);
		for (uint32_t k = 0; k < 5; ++k)
			AscendC::Gather(f[k + 6], p, priorOffsets.template ReinterpretCast<uint32_t>(), k * 4, count);
		DecodeCoordinates(f, count);
		auto out = mem.AllocOut();
		if constexpr (sizeof(T) == sizeof(float)) {
			AscendC::Gather(out.template ReinterpretCast<float>(), mem.Work(),
							outOffsets.template ReinterpretCast<uint32_t>(), 0, count * 5);
		} else {
			auto packed = mem.PackedFloat();
			AscendC::Gather(packed, mem.Work(), outOffsets.template ReinterpretCast<uint32_t>(), 0,
							count * 5);
			AscendC::Cast(out, packed, AscendC::RoundMode::CAST_RINT, count * 5);
		}
		mem.OutReady();
		mem.OutWait();
		AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
		AscendC::DataCopyPad(outGm[pos * 5], out, op);
	}
	__aicore__ inline void Bounds(const FireCandidateDecodeFusionTilingData &t, uint32_t &begin,
								  uint32_t &end) {
		if constexpr (ROW) {
			const uint32_t batches = t.total / t.n;
			const uint32_t coresPerBatch = static_cast<uint32_t>(AscendC::GetBlockNum()) / batches;
			const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
			const uint32_t batch = block / coresPerBatch;
			const uint32_t lane = block % coresPerBatch;
			begin = batch * t.n + lane * t.perCore;
			const uint32_t rowEnd = (batch + 1) * t.n;
			end = begin + t.perCore < rowEnd ? begin + t.perCore : rowEnd;

		} else {
			begin = AscendC::GetBlockIdx() * t.perCore;
			end = begin + t.perCore < t.total ? begin + t.perCore : t.total;
		}
	}

	__aicore__ inline void LoadRawAndPrior(AscendC::LocalTensor<T> r, AscendC::LocalTensor<float> p,
										   uint32_t pos, uint32_t priorPos, uint32_t count) {
		AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
		AscendC::DataCopyExtParams pp{1, count * 5 * uint32_t(sizeof(float)), 0, 0, 0};
		AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
		AscendC::DataCopyPad(p, priorGm[priorPos * 5], pp,
							 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
		mem.RawReady();
		mem.PriorReady();
	}

  public:
	__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded,
							   const FireCandidateDecodeFusionTilingData &t) {
		rawGm.SetGlobalBuffer((__gm__ T *)raw);
		priorGm.SetGlobalBuffer((__gm__ float *)priors);
		outGm.SetGlobalBuffer((__gm__ T *)decoded);
		uint32_t begin, end;
		Bounds(t, begin, end);
		if (begin >= end)
			return;
		uint32_t firstCount = end - begin < TILE ? end - begin : TILE;
		const uint32_t firstPriorPos = begin % t.n;
		if (firstCount > t.n - firstPriorPos)
			firstCount = t.n - firstPriorPos;
		auto firstR = mem.AllocRaw();
		auto firstP = mem.AllocPrior();
		LoadRawAndPrior(firstR, firstP, begin, firstPriorPos, firstCount);
		auto rawOffsets = mem.Indices();
		auto priorOffsets = rawOffsets[TILE];
		auto outOffsets = rawOffsets[2 * TILE];
		MakeOffsets<int32_t, TILE, 0>(rawOffsets, 4);
		AscendC::Muls<int32_t>(priorOffsets, rawOffsets, int32_t(5), TILE);
		AscendC::Muls<int32_t>(rawOffsets, rawOffsets, int32_t(6), TILE);
		MakeFireOutputOffsets<TILE>(outOffsets, rawOffsets, priorOffsets,
									mem.Work().template ReinterpretCast<int32_t>());
		auto one = mem.Ones();
		AscendC::Duplicate(one, 1.0f, TILE * 6);
		for (uint32_t pos = begin; pos < end;) {
			uint32_t count = end - pos < TILE ? end - pos : TILE;
			uint32_t priorPos = pos % t.n;
			if (count > t.n - priorPos)
				count = t.n - priorPos;
			auto r = firstR;
			auto p = firstP;
			if (pos != begin) {
				r = mem.AllocRaw();
				p = mem.AllocPrior();
				LoadRawAndPrior(r, p, pos, priorPos, count);
			}
			mem.RawWait();
			mem.PriorWait();
			Decode(count, pos, r, p, rawOffsets, priorOffsets, outOffsets, one);
			mem.FreeOut();
			mem.FreeRaw();
			mem.FreePrior();
			pos += count;
		}
		mem.Drain();
	}
};

template <typename T, uint32_t TILE> using KernelFireRow = KernelFire<T, TILE, true>;
