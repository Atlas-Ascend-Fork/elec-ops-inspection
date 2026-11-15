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

template <typename T, uint32_t PRIOR_WORDS> class FireComposite32Storage {
	static_assert(PRIOR_WORDS == 160 || PRIOR_WORDS == 288, "small prior staging layouts");
	static constexpr uint32_t RAW_BYTES = 192 * sizeof(T);
	static constexpr uint32_t PRIOR_BASE = RAW_BYTES;
	static constexpr uint32_t OUT_BASE = PRIOR_BASE + PRIOR_WORDS * sizeof(float);
	static constexpr uint32_t WORK_BASE = OUT_BASE + 160 * sizeof(T);
	static constexpr uint32_t ONE_BASE = WORK_BASE + 2304;
	static constexpr uint32_t RF_BASE = ONE_BASE + 192 * sizeof(float);
	static constexpr uint32_t PACK_BASE = RF_BASE + (sizeof(T) == sizeof(half) ? 192 * sizeof(float) : 0);
	static constexpr uint32_t INDEX_BASE = PACK_BASE + (sizeof(T) == sizeof(half) ? 160 * sizeof(float) : 0);
	static_assert(INDEX_BASE + 512 * sizeof(uint32_t) == (PRIOR_WORDS == 160
															  ? (sizeof(T) == sizeof(float) ? 7168 : 7872)
															  : (sizeof(T) == sizeof(float) ? 7680 : 8384)),
				  "bounded small32 storage");
	static_assert(INDEX_BASE + 512 * sizeof(uint32_t) < 188416, "preserve SDK temporary UB");
	static_assert((RAW_BYTES | PRIOR_BASE | OUT_BASE | WORK_BASE | ONE_BASE | RF_BASE | PACK_BASE |
				   INDEX_BASE) %
						  32 ==
					  0,
				  "static32 UB views retain32B alignment");
	static_assert(sizeof(T) == sizeof(float) ? ONE_BASE % 512 == 256 : (RF_BASE - ONE_BASE) % 512 == 256,
				  "native Div sources retain opposite bank groups");

  public:
	__aicore__ inline AscendC::LocalTensor<T> Raw() {
		return AscendC::LocalTensor<T>(AscendC::TPosition::VECIN, 0, 192);
	}
	__aicore__ inline AscendC::LocalTensor<float> Prior() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, PRIOR_BASE, PRIOR_WORDS);
	}
	__aicore__ inline AscendC::LocalTensor<T> Out() {
		return AscendC::LocalTensor<T>(AscendC::TPosition::VECOUT, OUT_BASE, 160);
	}
	__aicore__ inline AscendC::LocalTensor<float> Work() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, WORK_BASE, 576);
	}
	__aicore__ inline AscendC::LocalTensor<float> Ones() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ONE_BASE, 192);
	}
	__aicore__ inline AscendC::LocalTensor<float> RawFloat() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, RF_BASE, 192);
	}
	__aicore__ inline AscendC::LocalTensor<float> PackedFloat() {
		return AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, PACK_BASE, 160);
	}
	// GM DMA uses the public VECIN contract; Gather sees the same physical index bytes.
	__aicore__ inline AscendC::LocalTensor<uint32_t> IndexInput() {
		return AscendC::LocalTensor<uint32_t>(AscendC::TPosition::VECIN, INDEX_BASE, 512);
	}
	__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
		return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, 512);
	}
};

template <typename T> using FireHostComposite32Storage = FireComposite32Storage<T, 160>;
template <typename T> using FireFlatCompositeStorage = FireComposite32Storage<T, 288>;

#include "fire_small_common.h"

template <typename T, bool SINGLE = false, bool FULL = false, typename Storage = FireFlatCompositeStorage<T>,
		  bool CAPI_PIPE = false, bool PACKET_GM = false, bool EARLY_RAW_C1 = false>
using KernelFireFlatComposite =
	KernelFireFlatCompositeCommon<T, SINGLE, FULL, Storage, CAPI_PIPE, PACKET_GM, EARLY_RAW_C1, false, false>;
template <typename T> class FireHostComposite32Stride64Storage678 : public FireHostComposite32Storage<T> {
	static constexpr uint32_t INDEX_BASE = sizeof(T) == sizeof(float) ? 5120 : 5824;
	static_assert(INDEX_BASE + 544 * sizeof(uint32_t) == (sizeof(T) == sizeof(float) ? 7296 : 8000),
				  "bounded stride64 index storage");
	static_assert(INDEX_BASE % 32 == 0 && INDEX_BASE + 544 * sizeof(uint32_t) < 188416,
				  "retain SDK temporary UB reservation");

  public:
	__aicore__ inline AscendC::LocalTensor<uint32_t> IndexInput() {
		return AscendC::LocalTensor<uint32_t>(AscendC::TPosition::VECIN, INDEX_BASE, 544);
	}
	__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
		return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, 544);
	}
};

template <typename T, bool SINGLE = false, bool FULL = false, typename Storage = FireFlatCompositeStorage<T>,
		  bool CAPI_PIPE = false, bool PACKET_GM = false, bool EARLY_RAW_C1 = false>
using KernelFireFlatCompositeInventory684 =
	KernelFireFlatCompositeCommon<T, SINGLE, FULL, Storage, CAPI_PIPE, PACKET_GM, EARLY_RAW_C1, true, false>;
template <typename T, bool SINGLE = false, bool FULL = false, typename Storage = FireFlatCompositeStorage<T>,
		  bool CAPI_PIPE = false, bool PACKET_GM = false, bool EARLY_RAW_C1 = false>
using KernelFireFlatCompositeEarlyOne684 =
	KernelFireFlatCompositeCommon<T, SINGLE, FULL, Storage, CAPI_PIPE, PACKET_GM, EARLY_RAW_C1, false, true>;

#include "fire_small_indices.h"
template <typename T> class FireEven8Half32Storage : public FireHostComposite32Storage<T> {
	static_assert(sizeof(T) == sizeof(half), "half full32 only");
	static constexpr uint32_t INDEX_BASE = 192 * sizeof(T) + 160 * sizeof(float) + 160 * sizeof(T) + 2304 +
										   192 * sizeof(float) + 192 * sizeof(float) + 160 * sizeof(float);
	static_assert(INDEX_BASE == 5824 && INDEX_BASE % 32 == 0, "same static base");
	static_assert(INDEX_BASE + 608 * sizeof(uint32_t) == 8256, "bounded608 index view");
	static_assert(INDEX_BASE + 608 * sizeof(uint32_t) < 188416, "preserve SDK temporary UB");

  public:
	__aicore__ inline AscendC::LocalTensor<uint32_t> IndexInput() {
		return AscendC::LocalTensor<uint32_t>(AscendC::TPosition::VECIN, INDEX_BASE, 608);
	}
	__aicore__ inline AscendC::LocalTensor<int32_t> Indices() {
		return AscendC::LocalTensor<int32_t>(AscendC::TPosition::VECCALC, INDEX_BASE, 608);
	}
};
template <typename T> class KernelFireEven8Half32 {
	static_assert(sizeof(T) == sizeof(half), "half geometry only");
	FireEven8Half32Storage<T> mem;
	__aicore__ inline void LoadInputs(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded, uint32_t pos,
									  AscendC::GlobalTensor<T> &outGm) {
		AscendC::GlobalTensor<T> rawGm;
		AscendC::GlobalTensor<float> priorGm;
		AscendC::GlobalTensor<uint32_t> tableGm;
		rawGm.SetGlobalBuffer((__gm__ T *)raw);
		priorGm.SetGlobalBuffer((__gm__ float *)priors);
		outGm.SetGlobalBuffer((__gm__ T *)decoded);
		tableGm.SetGlobalBuffer(const_cast<__gm__ uint32_t *>(FireEven8Half32Index));
		auto r = mem.Raw();
		auto p = mem.Prior();
		auto one = mem.Ones();
		auto onePtr = (__ubuf__ float *)one.GetPhyAddr();
		AscendC::SetMaskCount();
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(192);
		asc_duplicate(onePtr, 1.0f, uint8_t(1), uint16_t(1), uint16_t(8));
		AscendC::DataCopyExtParams rp{1, 192 * uint32_t(sizeof(T)), 0, 0, 0};
		AscendC::DataCopyExtParams pp{1, 160 * uint32_t(sizeof(float)), 0, 0, 0};
		AscendC::DataCopyPad(r, rawGm[pos * 6], rp, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
		AscendC::DataCopyPad(p, priorGm[pos * 5], pp, AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
		AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
		AscendC::DataCopy(mem.IndexInput(), tableGm, uint32_t(608));
		AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
		AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
	}
	__aicore__ inline void Activate() {
		auto r = mem.Raw();
		auto rf = sizeof(T) == sizeof(float) ? r.template ReinterpretCast<float>() : mem.RawFloat();
		auto onePtr = (__ubuf__ float *)mem.Ones().GetPhyAddr();
		FireSmallActivate(r, rf, onePtr);
	}

	__aicore__ inline void DecodeGeometry() {
		auto r = mem.Raw();
		auto wPtr = (__ubuf__ float *)mem.Work().GetPhyAddr();
		auto indexPtr = (__ubuf__ uint32_t *)mem.Indices().GetPhyAddr();
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(448);
		asc_gather(wPtr, (__ubuf__ float *)r.GetPhyAddr(), indexPtr, uint16_t(8), uint8_t(1));
		AscendC::PipeBarrier<PIPE_V>();
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(128);
		asc_mul_scalar(wPtr, wPtr, 2.0f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
		AscendC::PipeBarrier<PIPE_V>();
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(64);
		asc_add_scalar(wPtr, wPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8), uint16_t(8));
		asc_mul(wPtr + 64, wPtr + 64, wPtr + 64, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8),
				uint8_t(8), uint8_t(8));
		AscendC::PipeBarrier<PIPE_V>();
		asc_add(wPtr, wPtr, wPtr + 160, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8),
				uint8_t(8), uint8_t(8));
		asc_mul(wPtr + 64, wPtr + 64, wPtr + 224, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8),
				uint8_t(8), uint8_t(8));
		AscendC::PipeBarrier<PIPE_V>();
		AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(160);
		asc_mul(wPtr, wPtr, wPtr + 288, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8),
				uint8_t(8), uint8_t(8));
		AscendC::PipeBarrier<PIPE_V>();
	}
	__aicore__ inline void PackAndStore(AscendC::GlobalTensor<T> outGm, uint32_t pos) {
		auto dst = mem.Out();
		auto wPtr = (__ubuf__ float *)mem.Work().GetPhyAddr();
		auto indexPtr = (__ubuf__ uint32_t *)mem.Indices().GetPhyAddr();
		if constexpr (sizeof(T) == sizeof(float)) {
			asc_gather((__ubuf__ float *)dst.GetPhyAddr(), wPtr, indexPtr + 448, uint16_t(8), uint8_t(1));
		} else {
			auto packed = mem.PackedFloat().template ReinterpretCast<half>();
			auto packedPtr = (__ubuf__ half *)packed.GetPhyAddr();
			asc_float2half_rn(packedPtr, wPtr, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(4),
							  uint16_t(8));
			AscendC::PipeBarrier<PIPE_V>();
			asc_gather((__ubuf__ half *)dst.GetPhyAddr(), packedPtr, indexPtr + 448, uint16_t(8), uint8_t(1));
		}
		AscendC::SetMaskNorm();
		AscendC::ResetMask();
		AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
		AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
		AscendC::DataCopyExtParams op{1, 160 * uint32_t(sizeof(T)), 0, 0, 0};
		AscendC::DataCopyPad(outGm[pos * 5], dst, op);
		AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
		AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
	}

  public:
	__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
		const uint32_t pos = (static_cast<uint32_t>(AscendC::GetBlockIdx()) / 8U) * 32U;
		AscendC::GlobalTensor<T> outGm;
		LoadInputs(raw, priors, decoded, pos, outGm);
		Activate();
		DecodeGeometry();
		PackAndStore(outGm, pos);
	}
};
