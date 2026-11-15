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

template <typename T>
__aicore__ inline void FireSmallActivate(AscendC::LocalTensor<T> raw, AscendC::LocalTensor<float> rf,
										 __ubuf__ float *onePtr) {
	auto rfPtr = (__ubuf__ float *)rf.GetPhyAddr();
	if constexpr (sizeof(T) != sizeof(float)) {
		asc_half2float(rfPtr, (__ubuf__ half *)raw.GetPhyAddr(), uint8_t(1), uint16_t(1), uint16_t(1),
					   uint16_t(8), uint16_t(4));
		AscendC::PipeBarrier<PIPE_V>();
	}
	asc_mul_scalar(rfPtr, rfPtr, -1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
	AscendC::PipeBarrier<PIPE_V>();
	asc_exp(rfPtr, rfPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
	AscendC::PipeBarrier<PIPE_V>();
	asc_add_scalar(rfPtr, rfPtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
	AscendC::PipeBarrier<PIPE_V>();
	asc_div(rfPtr, onePtr, rfPtr, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8),
			uint8_t(8));
	AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
	AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, bool SINGLE, bool FULL, typename Storage, bool CAPI_PIPE, bool PACKET_GM,
		  bool EARLY_RAW_C1, bool INVENTORY, bool EARLY_ONE>
class KernelFireFlatCompositeCommon {
	static_assert(!PACKET_GM || (CAPI_PIPE && !SINGLE && !FULL), "556 fixed C3 only");
	static_assert(!EARLY_RAW_C1 || (CAPI_PIPE && SINGLE && FULL && sizeof(T) == sizeof(half)),
				  "591 halfC1 only");
	Storage mem;
	AscendC::GlobalTensor<T> rawGm, outGm;
	AscendC::GlobalTensor<float> priorGm;
	AscendC::GlobalTensor<uint32_t> tableGm;
	template <uint32_t STRIDE> __aicore__ inline void PrepareEarlyOnes() {
		if constexpr (EARLY_ONE && CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
			auto earlyOne = mem.Ones();
			AscendC::PipeBarrier<PIPE_V>();
			AscendC::SetMaskCount();
			AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(192);
			asc_duplicate((__ubuf__ float *)earlyOne.GetPhyAddr(), 1.0f, uint8_t(1), uint16_t(1),
						  uint16_t(8));
		}
	}
	template <uint32_t STRIDE>
	__aicore__ inline void BindBuffers(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded, GM_ADDR tiling,
									   uint32_t tableOffset) {
		rawGm.SetGlobalBuffer((__gm__ T *)raw);
		priorGm.SetGlobalBuffer((__gm__ float *)priors);
		outGm.SetGlobalBuffer((__gm__ T *)decoded);
		if constexpr (PACKET_GM) {
			static_assert(STRIDE == 32 || STRIDE == 8, "556 full32 and tail8");
			tableGm.SetGlobalBuffer(
				const_cast<__gm__ uint32_t *>(FirePacketIndexGM556 + (tableOffset - 32) / 4));
		} else if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
			// Read-only constant GM table.
			if constexpr (INVENTORY)
				tableGm.SetGlobalBuffer(const_cast<__gm__ uint32_t *>(FireSmallIndexGM678));
			else
				tableGm.SetGlobalBuffer(const_cast<__gm__ uint32_t *>(FireSmallIndexGM548));
		} else {
			tableGm.SetGlobalBuffer((__gm__ uint32_t *)(tiling + tableOffset));
		}
	}
	template <uint32_t STRIDE>
	__aicore__ inline void LoadInputs(uint32_t pos, uint32_t priorStart, uint32_t priorCount,
									  uint32_t count) {
		auto r = mem.Raw();
		auto p = mem.Prior();
		if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
			// A complete 32-candidate block requires only element-aligned typed GM addresses.
			AscendC::DataCopy(r, rawGm[pos * 6], uint32_t(192));
			if constexpr (EARLY_RAW_C1)
				AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
			AscendC::DataCopy(p, priorGm[priorStart * 5], uint32_t(160));
		} else {
			AscendC::DataCopyExtParams rp{1, count * 6 * uint32_t(sizeof(T)), 0, 0, 0};
			AscendC::DataCopyExtParams pp{1, priorCount * 5 * uint32_t(sizeof(float)), 0, 0, 0};
			AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
			AscendC::DataCopyPad(p, priorGm[priorStart * 5], pp,
								 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
		}
		if constexpr (!EARLY_RAW_C1)
			AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
	}
	template <uint32_t STRIDE> __aicore__ inline void LoadIndices() {
		// Publish the later index DMA separately: activation only requires the first fence.
		if constexpr (CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
			AscendC::DataCopy(mem.IndexInput(), tableGm, uint32_t(INVENTORY ? 544 : 512));
		} else {
			AscendC::DataCopyExtParams ip{1, 16 * STRIDE * uint32_t(sizeof(uint32_t)), 0, 0, 0};
			AscendC::DataCopyPad(mem.IndexInput(), tableGm, ip,
								 AscendC::DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
		}
		AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
	}
	__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
		if constexpr (sizeof(T) == sizeof(float))
			return mem.Raw().template ReinterpretCast<float>();
		else
			return mem.RawFloat();
	}
	template <bool TAIL>
	__aicore__ inline void ActivateCount(AscendC::LocalTensor<float> rf, __ubuf__ float *onePtr) {
		if constexpr (!(EARLY_ONE && SINGLE && FULL)) {
			AscendC::PipeBarrier<PIPE_V>();
			AscendC::SetMaskCount();
			AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(TAIL ? 12 : 192);
			asc_duplicate(onePtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(8));
		}
		AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
		FireSmallActivate(mem.Raw(), rf, onePtr);
	}
	template <uint32_t STRIDE, bool TAIL>
	__aicore__ inline void DecodeCountCoordinates(__ubuf__ float *rfPtr, __ubuf__ float *wPtr,
												  __ubuf__ float *pPtr, __ubuf__ uint32_t *indexPtr) {
		if constexpr (TAIL)
			AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(6 * STRIDE);
		asc_gather(wPtr, rfPtr, indexPtr, uint16_t(8), uint8_t(1));
		AscendC::PipeBarrier<PIPE_V>();
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(4 * STRIDE);
		asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
		AscendC::PipeBarrier<PIPE_V>();
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(INVENTORY && SINGLE && FULL ? 192
																							  : 5 * STRIDE);
		asc_gather(wPtr + 6 * STRIDE, pPtr, indexPtr + 6 * STRIDE, uint16_t(8), uint8_t(1));
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2 * STRIDE);
		asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
		asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1),
				uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
		AscendC::PipeBarrier<PIPE_V>();
		asc_add(wPtr, wPtr, wPtr + 6 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8),
				uint8_t(8), uint8_t(8));
		asc_mul(wPtr + 2 * STRIDE, wPtr + 2 * STRIDE, wPtr + (INVENTORY && SINGLE && FULL ? 10 : 9) * STRIDE,
				uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
		AscendC::PipeBarrier<PIPE_V>();
	}
	template <uint32_t STRIDE> __aicore__ inline void ApplyCountStride(__ubuf__ float *wPtr) {
		if constexpr (INVENTORY && SINGLE && FULL) {
			AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(128);
			asc_mul(wPtr, wPtr, wPtr + 256, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8),
					uint8_t(8), uint8_t(0));
		} else {
			AscendC::SetMaskNorm();
			AscendC::SetVectorMask<float>(STRIDE);
			asc_mul(wPtr, wPtr, wPtr + 8 * STRIDE, uint8_t(4), uint8_t(1), uint8_t(1), uint8_t(1),
					uint8_t(STRIDE / 8), uint8_t(STRIDE / 8), uint8_t(0));
			AscendC::SetMaskCount();
		}
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(STRIDE);
		asc_mul(wPtr + 4 * STRIDE, wPtr + 4 * STRIDE, wPtr + 5 * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1),
				uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
		AscendC::PipeBarrier<PIPE_V>();
	}
	template <uint32_t STRIDE, bool TAIL>
	__aicore__ inline void PackCountOutput(AscendC::LocalTensor<float> rf, __ubuf__ float *wPtr,
										   __ubuf__ uint32_t *indexPtr) {
		auto out = mem.Out();
		constexpr uint32_t offset = (INVENTORY && SINGLE && FULL ? 12 : 11) * STRIDE;
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(TAIL ? 10 : 160);
		if constexpr (sizeof(T) == sizeof(float)) {
			asc_gather((__ubuf__ float *)out.GetPhyAddr(), wPtr, indexPtr + offset, uint16_t(8), uint8_t(1));
		} else {
#if FIRE_P8O_HALF_IMAGE
			if constexpr (!INVENTORY && !EARLY_ONE && PACKET_GM && !TAIL) {
				auto halfPlanar = (__ubuf__ half *)rf.GetPhyAddr();
				asc_float2half_rn(halfPlanar, wPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(4),
								  uint16_t(8));
				AscendC::PipeBarrier<PIPE_V>();
				asc_gather((__ubuf__ half *)out.GetPhyAddr(), halfPlanar, indexPtr + offset, uint16_t(8),
						   uint8_t(1));
			} else {
#endif
				auto packed = mem.PackedFloat();
				auto packedPtr = (__ubuf__ float *)packed.GetPhyAddr();
				asc_gather(packedPtr, wPtr, indexPtr + offset, uint16_t(8), uint8_t(1));
				AscendC::PipeBarrier<PIPE_V>();
				asc_float2half_rn((__ubuf__ half *)out.GetPhyAddr(), packedPtr, uint8_t(1), uint16_t(1),
								  uint16_t(1), uint16_t(4), uint16_t(8));
#if FIRE_P8O_HALF_IMAGE
			}
#endif
		}
		AscendC::SetMaskNorm();
		AscendC::ResetMask();
	}
	template <uint32_t STRIDE, bool COUNT_PIPELINE>
	__aicore__ inline void StoreOutput(uint32_t pos, uint32_t count) {
		auto out = mem.Out();
		AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
		AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
		if constexpr (COUNT_PIPELINE && CAPI_PIPE && SINGLE && FULL && STRIDE == 32) {
			AscendC::DataCopy(outGm[pos * 5], out, uint32_t(160));
		} else {
			AscendC::DataCopyExtParams op{1, count * 5 * uint32_t(sizeof(T)), 0, 0, 0};
			AscendC::DataCopyPad(outGm[pos * 5], out, op);
		}
		AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
		AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
	}
	template <uint32_t STRIDE, bool TAIL>
	__aicore__ inline void RunCountPipeline(uint32_t pos, uint32_t count) {
		auto rf = RawFloat();
		auto rfPtr = (__ubuf__ float *)rf.GetPhyAddr();
		auto onePtr = (__ubuf__ float *)mem.Ones().GetPhyAddr();
		auto wPtr = (__ubuf__ float *)mem.Work().GetPhyAddr();
		auto pPtr = (__ubuf__ float *)mem.Prior().GetPhyAddr();
		auto indexPtr = (__ubuf__ uint32_t *)mem.Indices().GetPhyAddr();
		ActivateCount<TAIL>(rf, onePtr);
		DecodeCountCoordinates<STRIDE, TAIL>(rfPtr, wPtr, pPtr, indexPtr);
		ApplyCountStride<STRIDE>(wPtr);
		PackCountOutput<STRIDE, TAIL>(rf, wPtr, indexPtr);
		StoreOutput<STRIDE, true>(pos, count);
	}
	__aicore__ inline AscendC::LocalTensor<float> ActivateGeneric(uint32_t count) {
		auto r = mem.Raw();
		auto one = mem.Ones();
		AscendC::Duplicate(one, 1.0f, count * 6);
		AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
		auto rf = RawFloat();
		if constexpr (sizeof(T) != sizeof(float))
			AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, count * 6);
		AscendC::Muls(rf, rf, -1.0f, count * 6);
		AscendC::Exp(rf, rf, count * 6);
		AscendC::Adds(rf, rf, 1.0f, count * 6);
		AscendC::Div(rf, one, rf, count * 6);
		return rf;
	}
	template <uint32_t STRIDE>
	__aicore__ inline void DecodeGeneric(AscendC::LocalTensor<float> rf, uint32_t count) {
		auto work = mem.Work();
		auto p = mem.Prior();
		auto indices = mem.Indices().template ReinterpretCast<uint32_t>();
		AscendC::LocalTensor<float> f[11];
#pragma unroll
		for (uint32_t k = 0; k < 11; ++k)
			f[k] = work[k * STRIDE];
		AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
		AscendC::Gather(work, rf, indices, 0, 6 * STRIDE);
		AscendC::Gather(f[6], p, indices[6 * STRIDE], 0, 5 * STRIDE);
		// Compact contiguous planes retain every FP32 node, including four stride uses.
		AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
		AscendC::Adds(f[0], f[0], -0.5f, 2 * STRIDE);
		AscendC::Add(f[0], f[0], f[6], 2 * STRIDE);
		AscendC::Mul(f[2], f[2], f[2], 2 * STRIDE);
		AscendC::Mul(f[2], f[2], f[9], 2 * STRIDE);
		const AscendC::BinaryRepeatParams strideBinary{1, 1, 1, STRIDE / 8, STRIDE / 8, 0};
		AscendC::Mul<float>(f[0], f[0], f[8], uint64_t(STRIDE), uint8_t(4), strideBinary);
		AscendC::Mul(f[4], f[4], f[5], STRIDE);
		AscendC::ResetMask();
		auto out = mem.Out();
		if constexpr (sizeof(T) == sizeof(float)) {
			AscendC::Gather(out.template ReinterpretCast<float>(), work, indices[11 * STRIDE], 0, count * 5);
		} else {
			auto packed = mem.PackedFloat();
			AscendC::Gather(packed, work, indices[11 * STRIDE], 0, count * 5);
			AscendC::Cast(out, packed, AscendC::RoundMode::CAST_RINT, count * 5);
		}
	}
	template <uint32_t STRIDE>
	__aicore__ inline void RunBlock(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded, GM_ADDR tiling,
									uint32_t pos, uint32_t priorStart, uint32_t priorCount, uint32_t count,
									uint32_t tableOffset) {
		PrepareEarlyOnes<STRIDE>();
		BindBuffers<STRIDE>(raw, priors, decoded, tiling, tableOffset);
		LoadInputs<STRIDE>(pos, priorStart, priorCount, count);
		LoadIndices<STRIDE>();
		if constexpr (!INVENTORY && !EARLY_ONE && CAPI_PIPE && PACKET_GM && STRIDE == 8) {
			RunCountPipeline<STRIDE, true>(pos, count);
		} else if constexpr (CAPI_PIPE && STRIDE == 32) {
			RunCountPipeline<STRIDE, false>(pos, count);
		} else {
			auto rf = ActivateGeneric(count);
			DecodeGeneric<STRIDE>(rf, count);
			StoreOutput<STRIDE, false>(pos, count);
		}
	}
	template <uint32_t BLOCK, uint32_t TOTAL, uint32_t N>
	__aicore__ inline void RunPacketBlock(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded, GM_ADDR tiling) {
		constexpr uint32_t POS = BLOCK * 32;
		constexpr uint32_t COUNT = BLOCK < 2 ? 32 : TOTAL - 64;
		constexpr uint32_t FIRST = POS % N;
		constexpr bool WRAPS = FIRST + COUNT > N;
		constexpr uint32_t PRIOR_START = WRAPS ? 0 : FIRST;
		constexpr uint32_t PRIOR_COUNT = WRAPS ? N : COUNT;
		constexpr uint32_t STRIDE = (COUNT + 7) / 8 * 8;
		constexpr uint32_t TABLE = BLOCK == 0 ? 32 : BLOCK == 1 ? 2080 : 4128;
		static_assert(COUNT > 0 && COUNT <= 32 && PRIOR_COUNT * 5 <= 288, "packet staging bounds");
		RunBlock<STRIDE>(raw, priors, decoded, tiling, POS, PRIOR_START, PRIOR_COUNT, COUNT, TABLE);
	}

  public:
	template <uint32_t TOTAL, uint32_t N, bool SPARSE_EVEN = false>
	__aicore__ inline void RunPacket(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded, GM_ADDR tiling) {
		static_assert(TOTAL > 64 && TOTAL <= 72 && TOTAL % N == 0 && TOTAL / N <= 8,
					  "two full blocks and a short tail");
		static_assert(!PACKET_GM || (TOTAL == 66 && N == 33), "556 exact packet payload");
		const uint32_t logicalBlock = static_cast<uint32_t>(AscendC::GetBlockIdx());
		const uint32_t block = SPARSE_EVEN ? logicalBlock / 2U : logicalBlock;
		if (block < 2) {
			constexpr uint32_t FIRST1 = 32 % N;
			constexpr uint32_t START1 = FIRST1 + 32 > N ? 0 : FIRST1;
			constexpr uint32_t COUNT0 = 32 > N ? N : 32;
			constexpr uint32_t COUNT1 = FIRST1 + 32 > N ? N : 32;
			const uint32_t priorStart = block == 0 ? 0 : START1;
			const uint32_t priorCount = block == 0 ? COUNT0 : COUNT1;
			RunBlock<32>(raw, priors, decoded, tiling, block * 32, priorStart, priorCount, 32,
						 32 + block * 2048);
		} else if (block == 2)
			RunPacketBlock<2, TOTAL, N>(raw, priors, decoded, tiling);
	}
	__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded, GM_ADDR tiling,
							   const FireCandidateDecodeFusionTilingData &t) {
		if constexpr (FULL) {
			static_assert(SINGLE, "one complete hardware block");
			RunBlock<32>(raw, priors, decoded, tiling, 0, 0, t.n, 32, 32);
		} else if constexpr (SINGLE) {
			const uint32_t count = t.total;
			if (count <= 8)
				RunBlock<8>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
			else if (count <= 16)
				RunBlock<16>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
			else if (count <= 24)
				RunBlock<24>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
			else
				RunBlock<32>(raw, priors, decoded, tiling, 0, 0, t.n, count, 32);
		} else {
			const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
			const uint32_t pos = block * 32;
			if (pos >= t.total)
				return;
			const uint32_t count = t.total - pos < 32 ? t.total - pos : 32;
			const uint32_t firstPrior = pos % t.n;
			const bool wraps = firstPrior + count > t.n;
			const uint32_t priorStart = wraps ? 0 : firstPrior;
			const uint32_t priorCount = wraps ? t.n : count;
			const uint32_t tableOffset = block < 2 ? 32 + block * 2048 : 4128;
			if (count <= 8)
				RunBlock<8>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
			else if (count <= 16)
				RunBlock<16>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
			else if (count <= 24)
				RunBlock<24>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
			else
				RunBlock<32>(raw, priors, decoded, tiling, pos, priorStart, priorCount, count, tableOffset);
		}
	}
};
