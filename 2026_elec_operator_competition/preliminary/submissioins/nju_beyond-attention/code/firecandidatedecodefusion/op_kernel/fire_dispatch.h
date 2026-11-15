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

template <uint32_t REUSE> __aicore__ inline bool FireDispatchSkipBlock() {
	if constexpr (REUSE == 0x7b36U) {
		constexpr uint32_t activeMask = 7U;
		if ((static_cast<uint32_t>(AscendC::GetBlockIdx()) & activeMask) != 0U) {
			return true;
		}
	}
	if constexpr (REUSE == 0x7b38U) {
		const uint32_t logicalBlock = static_cast<uint32_t>(AscendC::GetBlockIdx());
		if ((logicalBlock & 1U) != 0U || logicalBlock >= 6U) {
			AscendC::PipeBarrier<PIPE_ALL>();
			return true;
		}
	}
	return false;
}

template <typename DT_RAW_PRED>
__aicore__ inline void FireDispatchWhole(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded) {
	AscendC::ICachePreLoad(3);
	constexpr uint32_t BATCHES = 4;
	constexpr uint32_t FRAME = 512;
	static_assert(BATCHES >= 1 && BATCHES <= 8 && FRAME == 512, "whole resource geometry");
	const uint32_t cores =
		sizeof(DT_RAW_PRED) == sizeof(float) ? 32U : static_cast<uint32_t>(AscendC::GetBlockNum());
	const uint32_t rowLanes = cores / BATCHES;
	const FireFrameTilingData schedule{{cores * FRAME, rowLanes * FRAME, FRAME}, rowLanes | (1U << 16)};
	KernelFireFrame<DT_RAW_PRED, FRAME, true, true, true, sizeof(DT_RAW_PRED) == sizeof(half),
					sizeof(DT_RAW_PRED) == sizeof(float), sizeof(DT_RAW_PRED) == sizeof(half),
					sizeof(DT_RAW_PRED) == sizeof(half), sizeof(DT_RAW_PRED) == sizeof(float),
					sizeof(DT_RAW_PRED) == sizeof(half) ? 8U : 16U>
		kernel;
	kernel.template Run<true>(raw_pred, priors, decoded, schedule);
}

template <typename DT_RAW_PRED, uint32_t N>
__aicore__ inline void FireDispatchFloatPair43(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded,
											   uint32_t block) {
	if constexpr (N == 32768U) {
		if (block < 42U) {
			KernelFireA2F5Pair<DT_RAW_PRED, N, true, true, true, false, true, true, 768U> kernel;
			kernel.Run(raw_pred, priors, decoded);
		} else if (block == 42U) {
			KernelFireA2F5Tail512<DT_RAW_PRED, N> kernel;
			kernel.Run(raw_pred, priors, decoded);
		}
	} else {
		if (block < 42U) {
			KernelFireA2F6SpanPair<DT_RAW_PRED, N, true, true, true, false, true, true, 768U> kernel;
			kernel.Run(raw_pred, priors, decoded);
		} else if (block == 42U) {
			KernelFireA2F6SpanTail513<DT_RAW_PRED, N> kernel;
			kernel.Run(raw_pred, priors, decoded);
		}
	}
}

template <typename DT_RAW_PRED, uint32_t N>
__aicore__ inline void FireDispatchFloatPair40(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded,
											   uint32_t block) {
	if (block < 39U) {
		if constexpr (N == 32768U) {
			KernelFireA2F5Pair<DT_RAW_PRED, N, true, true, true, false, true, true> kernel;
			kernel.Run(raw_pred, priors, decoded);
		} else {
			KernelFireA2F6AddPair<DT_RAW_PRED, N, true, true, true, false, true, true> kernel;
			kernel.Run(raw_pred, priors, decoded);
		}
	} else if (block == 39U) {
		if constexpr (N == 32768U) {
			KernelFireA2F5Tail320<DT_RAW_PRED, N> kernel;
			kernel.Run(raw_pred, priors, decoded);
		} else {
			KernelFireA2F6AddTail321<DT_RAW_PRED, N> kernel;
			kernel.Run(raw_pred, priors, decoded);
		}
	}
}

template <typename DT_RAW_PRED, uint32_t N>
__aicore__ inline void FireDispatchPairMain(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded) {
#if FIRE_PAIR_QUAL11_IMAGE
	if constexpr (sizeof(DT_RAW_PRED) == sizeof(float) && (N == 32768U || N == 32769U)) {
		KernelFirePair576<DT_RAW_PRED, N, true, true, true, false> kernel;
		kernel.Run(raw_pred, priors, decoded);
	} else
#endif
#if FIRE_PAIR_HALF_SCORE_IMAGE
		if constexpr (sizeof(DT_RAW_PRED) == sizeof(half)) {
		if constexpr (N == 32768U) {
			KernelFirePairInputPi8H11<DT_RAW_PRED, N, true> kernel;
			kernel.Run(raw_pred, priors, decoded);
		} else {
			KernelFirePairInputPi8H12<DT_RAW_PRED, N, true> kernel;
			kernel.Run(raw_pred, priors, decoded);
		}
	} else
#endif
	{
		KernelFirePair576<DT_RAW_PRED, N, (sizeof(DT_RAW_PRED) == sizeof(float) && N == 32768U),
						  (sizeof(DT_RAW_PRED) == sizeof(float) && N == 32769U)>
			kernel;
		kernel.Run(raw_pred, priors, decoded);
	}
}

template <typename DT_RAW_PRED, uint32_t N>
__aicore__ inline void FireDispatchPairTail(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded,
											const FireFrameTilingData &schedule) {
#if FIRE_PAIR_HALF_SCORE_IMAGE
	if constexpr (sizeof(DT_RAW_PRED) == sizeof(half)) {
		KernelFireFrameStock1112<DT_RAW_PRED, 384, false, true, false, N> kernel;
		kernel.Run(raw_pred, priors, decoded, schedule);
	} else
#endif
	{
		KernelFireFrame<DT_RAW_PRED, 896, false, true> kernel;
		kernel.Run(raw_pred, priors, decoded, schedule);
	}
}

template <typename DT_RAW_PRED, uint32_t REUSE>
__aicore__ inline bool FireDispatchPairs(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded) {
	AscendC::ICachePreLoad(3);
	const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
	constexpr uint32_t N = REUSE == 0xb500U ? 32768U : 32769U;
	if constexpr (sizeof(DT_RAW_PRED) == sizeof(float)) {
		const uint32_t launched = static_cast<uint32_t>(AscendC::GetBlockNum());
		if (launched == 43U)
			FireDispatchFloatPair43<DT_RAW_PRED, N>(raw_pred, priors, decoded, block);
		else
			FireDispatchFloatPair40<DT_RAW_PRED, N>(raw_pred, priors, decoded, block);
		// These FP32 paths return before the entry's final PIPE_ALL barrier.
		return true;
	}
	if (block >= 40)
		return true;
	const FireFrameTilingData schedule{{8U * N, N, 832}, (8U << 16) | 40U};
	if (block < 39U)
		FireDispatchPairMain<DT_RAW_PRED, N>(raw_pred, priors, decoded);
	else
		FireDispatchPairTail<DT_RAW_PRED, N>(raw_pred, priors, decoded, schedule);
	return false;
}

template <typename T, uint32_t N, bool SPAN47, uint32_t SPAN, uint32_t COUNT, bool RAW_SCORE = false>
__aicore__ inline void FireDispatchHalfSpanKernel(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded) {
	if constexpr (SPAN47) {
		KernelFirePairSpan47<T, N, SPAN, COUNT, RAW_SCORE> kernel;
		kernel.Run(raw_pred, priors, decoded);
	} else {
		KernelFirePairSpan1112<T, N, SPAN, COUNT, RAW_SCORE> kernel;
		kernel.Run(raw_pred, priors, decoded);
	}
}
template <typename T, uint32_t N, bool SPAN47>
__aicore__ inline void FireDispatchHalfSpan(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded,
											uint32_t block) {
	constexpr uint32_t MAIN_SPAN = SPAN47 ? 704U : 768U;
	if (block < (SPAN47 ? 46U : 42U)) {
		FireDispatchHalfSpanKernel<T, N, SPAN47, MAIN_SPAN, MAIN_SPAN, (N == 32769U)>(raw_pred, priors,
																					  decoded);
	} else if constexpr (N == 32768U) {
		constexpr uint32_t TAIL = SPAN47 ? 384U : 512U;
		FireDispatchHalfSpanKernel<T, N, SPAN47, TAIL, TAIL>(raw_pred, priors, decoded);
	} else {
		constexpr uint32_t TAIL_SPAN = SPAN47 ? 448U : 576U;
		constexpr uint32_t TAIL_COUNT = SPAN47 ? 385U : 513U;
		FireDispatchHalfSpanKernel<T, N, SPAN47, TAIL_SPAN, TAIL_COUNT>(raw_pred, priors, decoded);
	}
}

template <typename DT_RAW_PRED, uint32_t REUSE>
__aicore__ inline bool FireDispatchSpans(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded) {
	AscendC::ICachePreLoad(3);
	const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
	constexpr bool span47 = REUSE == 0xb602U || REUSE == 0xb603U;
	if (block >= (span47 ? 47U : 43U))
		return true;
	constexpr uint32_t N = REUSE == 0xb602U || REUSE == 0xb600U ? 32768U : 32769U;
	if constexpr (sizeof(DT_RAW_PRED) == sizeof(half)) {
		FireDispatchHalfSpan<DT_RAW_PRED, N, span47>(raw_pred, priors, decoded, block);
	} else {
		if constexpr (span47) {
			const FireCandidateDecodeFusionTilingData schedule{8U * N, N, (8U * N + 46U) / 47U};
			KernelFire<DT_RAW_PRED, 832U> kernel;
			kernel.Run(raw_pred, priors, decoded, schedule);
		} else {
			const FireCandidateDecodeFusionTilingData schedule{8U * N, N, (8U * N + 42U) / 43U};
			KernelFire<DT_RAW_PRED, 832U> kernel;
			kernel.Run(raw_pred, priors, decoded, schedule);
		}
	}
	return false;
}

template <typename DT_RAW_PRED>
__aicore__ inline bool FireDispatchRows(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded) {
	AscendC::ICachePreLoad(3);
	const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
	if (block >= 20U)
		return true;
	constexpr FireFrameTilingData schedule{{16388U, 4097U, 832U}, (1U << 16) | 5U};
	if constexpr (sizeof(DT_RAW_PRED) == sizeof(float)) {
		KernelFireA2RowF4<DT_RAW_PRED, 896, true, true, false, 4097U, true, true> kernel;
		kernel.Run(raw_pred, priors, decoded, schedule);
	} else {
		KernelFireA2RowH10<DT_RAW_PRED, 896, true, true, false, true, 4097U, true> kernel;
		kernel.Run(raw_pred, priors, decoded, schedule);
	}

	return false;
}

template <typename DT_RAW_PRED, uint32_t REUSE>
__aicore__ inline bool FireDispatchScheduledFrame(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded,
												  GM_ADDR tiling) {
	AscendC::ICachePreLoad(3);
	const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
	if (block >= 48)
		return true;
	GET_TILING_DATA_PTR_WITH_STRUCT(FireFrameTilingData, schedulePtr, tiling);
	const FireFrameTilingData schedule{
		{(*schedulePtr).header.total, (*schedulePtr).header.n, (*schedulePtr).header.perCore},
		(*schedulePtr).geometry};
	if constexpr (REUSE == 0xb300U) {
		KernelFireFrame<DT_RAW_PRED, 768, false, true> kernel;
		kernel.Run(raw_pred, priors, decoded, schedule);
	} else {
		constexpr uint32_t FRAME = REUSE & 0x1fffU;
		constexpr bool ROW = (REUSE & 0x2000U) != 0;
		KernelFireFrame<DT_RAW_PRED, FRAME, ROW> kernel;
		kernel.Run(raw_pred, priors, decoded, schedule);
	}
	return false;
}

template <typename DT_RAW_PRED, uint32_t REUSE>
__aicore__ inline void FireDispatchSmall(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded, GM_ADDR tiling) {
	AscendC::ICachePreLoad(2);
	if constexpr (REUSE == 0x7b36U) {
		if constexpr (sizeof(DT_RAW_PRED) == sizeof(half)) {
			KernelFireEven8Half32<DT_RAW_PRED> kernel;
			kernel.Run(raw_pred, priors, decoded);
		} else {
			const FireCandidateDecodeFusionTilingData t{32, 32, 32};
			KernelFireFlatCompositeInventory684<DT_RAW_PRED, true, true,
												FireHostComposite32Stride64Storage678<DT_RAW_PRED>, true,
												false, false>
				kernel;
			kernel.Run(raw_pred, priors, decoded, tiling, t);
		}
	} else if constexpr (REUSE == 0x7b2eU) {
		const FireCandidateDecodeFusionTilingData t{32, 32, 32};
		if constexpr (sizeof(DT_RAW_PRED) == sizeof(float)) {
			KernelFireFlatCompositeInventory684<DT_RAW_PRED, true, true,
												FireHostComposite32Stride64Storage678<DT_RAW_PRED>, true,
												false, false>
				kernel;
			kernel.Run(raw_pred, priors, decoded, tiling, t);
		} else {
			KernelFireFlatCompositeEarlyOne684<DT_RAW_PRED, true, true,
											   FireHostComposite32Storage<DT_RAW_PRED>, true, false, true>
				kernel;
			kernel.Run(raw_pred, priors, decoded, tiling, t);
		}
	} else {
		KernelFireFlatComposite<DT_RAW_PRED, false, false, FireFlatCompositeStorage<DT_RAW_PRED>, true, true>
			kernel;
		kernel.template RunPacket<66, 33, REUSE == 0x7b38U>(raw_pred, priors, decoded, tiling);
	}
}

template <typename DT_RAW_PRED, uint32_t REUSE>
__aicore__ inline void FireDispatchGeneric(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded,
										   GM_ADDR tiling) {
	constexpr int64_t PRELOAD_LEN = (REUSE == 0 || REUSE == 1) ? 2 : 3;
	AscendC::ICachePreLoad(PRELOAD_LEN);
	GET_TILING_DATA_WITH_STRUCT(FireCandidateDecodeFusionTilingData, t, tiling);
	if constexpr (REUSE <= 2) {
		constexpr uint32_t TILE = REUSE == 0 ? 64 : REUSE == 1 ? 256 : 832;
		KernelFire<DT_RAW_PRED, TILE> kernel;
		kernel.Run(raw_pred, priors, decoded, t);
	} else {
		KernelFireRow<DT_RAW_PRED, REUSE == 5256 ? 256 : 832> kernel;
		kernel.Run(raw_pred, priors, decoded, t);
	}
}

template <typename DT_RAW_PRED, uint32_t REUSE>
__aicore__ inline void FireDispatchSmallOrGeneric(GM_ADDR raw_pred, GM_ADDR priors, GM_ADDR decoded,
												  GM_ADDR tiling) {
	if constexpr (REUSE == 0x7b36U || REUSE == 0x7b2eU || REUSE == 0x7b34U || REUSE == 0x7b38U) {
		FireDispatchSmall<DT_RAW_PRED, REUSE>(raw_pred, priors, decoded, tiling);
	} else {
		FireDispatchGeneric<DT_RAW_PRED, REUSE>(raw_pred, priors, decoded, tiling);
	}
}
