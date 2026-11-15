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

enum class FirePairMode { Fixed576, A2F5, A2F6Span, A2F6Add };

template <typename T, uint32_t N, FirePairMode MODE, bool SCORE_FUSED, bool ROTATE_PAIRS,
		  bool QUALIFIED_SCORE_ROTATE, bool QUALIFIED_HALF_SCORE, bool COORD_DIV2, bool VA_GROUP38,
		  uint32_t ROW = 832U, bool BASE_IMAGE = false>
class KernelFireFixedPair {
	static constexpr bool FIXED576 = MODE == FirePairMode::Fixed576;
	static constexpr bool F5 = MODE == FirePairMode::A2F5;
	static constexpr bool SIGNED_INPUT = MODE == FirePairMode::A2F6Add && VA_GROUP38;
	static constexpr uint32_t QUALIFIED_N = F5 ? 32768U : 32769U;
	static constexpr uint32_t FRAME = 2U * ROW;
	static constexpr uint32_t WORK_FRAME = ROW == 768U ? 1600U : FRAME;
	static constexpr uint32_t STRIDE = WORK_FRAME + 16U, Q = WORK_FRAME / 8U;
	static_assert(!SCORE_FUSED || sizeof(T) == sizeof(float) || QUALIFIED_HALF_SCORE,
				  "score fusion requires dtype qualification");
	static_assert(!(SCORE_FUSED && ROTATE_PAIRS) || QUALIFIED_SCORE_ROTATE, "636 mechanisms are exclusive");
	static_assert(!QUALIFIED_SCORE_ROTATE || ((!FIXED576 || !BASE_IMAGE) && sizeof(T) == sizeof(float) &&
											  (N == 32768U || N == 32769U) && SCORE_FUSED && ROTATE_PAIRS),
				  "only qualified float pair fusion rotation");
	static_assert(!QUALIFIED_HALF_SCORE ||
					  (FIXED576 && BASE_IMAGE && sizeof(T) == sizeof(half) && SCORE_FUSED && !ROTATE_PAIRS),
				  "only qualified half pair score fusion");
	static_assert(!(QUALIFIED_SCORE_ROTATE && QUALIFIED_HALF_SCORE), "qualified pair modes are separate");
	static_assert(N == 32768U || N == 32769U, "only qualifiedC40 B8 dimensions");
	static_assert(ROW == 832U ||
					  (ROW == 768U && (F5 || MODE == FirePairMode::A2F6Span) && sizeof(T) == sizeof(float) &&
					   N == QUALIFIED_N && SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE &&
					   !QUALIFIED_HALF_SCORE && COORD_DIV2 && VA_GROUP38 && WORK_FRAME == 1600U &&
					   STRIDE == 1616U && Q == 200U && 107008U + 11U * STRIDE * 4U < 181504U),
				  "qualified medium row");
	static_assert(!COORD_DIV2 ||
					  (!FIXED576 && sizeof(T) == sizeof(float) && N == QUALIFIED_N && SCORE_FUSED &&
					   ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && !QUALIFIED_HALF_SCORE),
				  "qualified FP32 coordinate Div2");
	static_assert(FRAME % 128U == 0U && 4U * STRIDE % 64U == 0U && WORK_FRAME % 64U == 0U,
				  "exact coordinate and score Div spans");
	static_assert(!VA_GROUP38 ||
					  (COORD_DIV2 && !FIXED576 && sizeof(T) == sizeof(float) && N == QUALIFIED_N &&
					   SCORE_FUSED && ROTATE_PAIRS && QUALIFIED_SCORE_ROTATE && !QUALIFIED_HALF_SCORE),
				  "qualified FP32 VA38");
	static constexpr bool BASIC2D =
		F5 && sizeof(T) == sizeof(float) && N == 32768U && COORD_DIV2 && VA_GROUP38;
	static constexpr bool FIXED_TOKENS =
		!FIXED576 && sizeof(T) == sizeof(float) && N == QUALIFIED_N && COORD_DIV2 && VA_GROUP38;
	static constexpr bool PRIOR_OVERLAP =
		FIXED_TOKENS && (MODE == FirePairMode::A2F6Add || (F5 && ROW == 832U));
	FireFixedPairStorage<T, VA_GROUP38> mem;
	AscendC::GlobalTensor<T> rawGm, outGm;
	AscendC::GlobalTensor<float> priorGm;

	__aicore__ inline void InitPackedVA() {
		FirePairAddressTable<T, STRIDE, Q, ROW / 8U, VA_GROUP38>::Init(
			mem, reinterpret_cast<uint64_t>(mem.Prior().GetPhyAddr()));
	}
	template <uint32_t SLOT> __aicore__ inline void Input() {
		FirePairPhases<ROW / 64U, VA_GROUP38>::template Input<SLOT>(mem.PackedVA());
	}
	template <uint32_t SLOT> __aicore__ inline void Output() {
		FirePairPhases<ROW / 64U, VA_GROUP38>::template Output<SLOT>(mem.PackedVA());
	}

	__aicore__ inline void FuseScore(AscendC::LocalTensor<float> one, AscendC::LocalTensor<float> *f,
									 const AscendC::BinaryRepeatParams &divBinary) {
		if constexpr (SCORE_FUSED) {
			AscendC::Mul(f[4], f[4], f[5], WORK_FRAME);
			if constexpr (COORD_DIV2) {
				AscendC::Div<float>(mem.Work(), mem.Twos(), mem.Work(), uint64_t(64),
									uint8_t(4U * STRIDE / 64U), divBinary);
				if constexpr (VA_GROUP38) {
					auto *scorePtr = reinterpret_cast<__ubuf__ float *>(f[4].GetPhyAddr());
					auto *onePtr = reinterpret_cast<__ubuf__ float *>(one.GetPhyAddr());
					asc_div(scorePtr, onePtr, scorePtr, uint8_t(WORK_FRAME / 64U), uint8_t(1), uint8_t(0),
							uint8_t(1), uint8_t(8), uint8_t(0), uint8_t(8));
				} else {
					AscendC::Div<float>(f[4], one, f[4], uint64_t(64), uint8_t(F5 ? WORK_FRAME / 64U : 26U),
										divBinary);
				}
			} else {
				AscendC::Div<float>(mem.Work(), one, mem.Work(), uint64_t(64), uint8_t(132), divBinary);
			}
			AscendC::PipeBarrier<PIPE_V>();
		}
	}

	__aicore__ inline void DecodeCoordinates(AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> *f) {
		if constexpr (!COORD_DIV2)
			AscendC::Muls(f[0], f[0], 2.0f, 4 * STRIDE);
		if constexpr (F5 && FIXED_TOKENS && ROW == 832U) {
			auto *mathPtr = reinterpret_cast<__ubuf__ float *>(mem.Work().GetPhyAddr());
			AscendC::SetMaskCount();
			AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2U * STRIDE);
			asc_add_scalar(mathPtr, mathPtr, -0.5f, uint8_t(1), uint16_t(1), uint16_t(1), uint16_t(8),
						   uint16_t(8));
			asc_mul(mathPtr + 2U * STRIDE, mathPtr + 2U * STRIDE, mathPtr + 2U * STRIDE, uint8_t(1),
					uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
			AscendC::PipeBarrier<PIPE_V>();
			asc_add(mathPtr, mathPtr, mathPtr + 6U * STRIDE, uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(1),
					uint8_t(8), uint8_t(8), uint8_t(8));
			asc_mul(mathPtr + 2U * STRIDE, mathPtr + 2U * STRIDE, mathPtr + 9U * STRIDE, uint8_t(1),
					uint8_t(1), uint8_t(1), uint8_t(1), uint8_t(8), uint8_t(8), uint8_t(8));
		} else {
			FireDecodeBox<STRIDE>(f);
		}
		if constexpr (FIXED576 && BASE_IMAGE) {
			AscendC::Mul(f[0], f[0], p, 4 * STRIDE);
		} else if constexpr (sizeof(T) == sizeof(float)) {
			AscendC::PipeBarrier<PIPE_V>();
			FireScaleBoxes<STRIDE, FIXED_TOKENS ? WORK_FRAME : STRIDE>(mem.Work());
		} else {
			AscendC::Mul(f[0], f[0], p, 4 * STRIDE);
		}
		if constexpr (!SCORE_FUSED)
			AscendC::Mul(f[4], f[4], f[5], FRAME);
		AscendC::PipeBarrier<PIPE_V>();
	}

	__aicore__ inline void StorePair(uint32_t iteration, uint32_t slot, uint32_t pos) {
		if constexpr (sizeof(T) == sizeof(float)) {
			if (slot == 0)
				Output<0>();
			else
				Output<1>();
		} else
			Output<0>();
		AscendC::PipeBarrier<PIPE_V>();
		auto out = mem.OutAt(slot);
		if constexpr (sizeof(T) != sizeof(float)) {
			AscendC::Cast(out, mem.PackedFloat(), AscendC::RoundMode::CAST_RINT, 8320U);
			AscendC::PipeBarrier<PIPE_V>();
		}
		AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot);
		AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot);
		const AscendC::DataCopyExtParams op{2, ROW * 5U * uint32_t(sizeof(T)), 0,
											(N - ROW) * 5U * uint32_t(sizeof(T)), 0};
		if constexpr (BASIC2D) {
			AscendC::DataCopy(
				outGm[pos * 5U], out,
				AscendC::DataCopyParams{2U, uint16_t(ROW * 20U / 32U), 0U, uint16_t((N - ROW) * 20U / 32U)});
		} else
			AscendC::DataCopyPad(outGm[pos * 5U], out, op);
		if constexpr (FIXED_TOKENS) {
			if (MODE != FirePairMode::A2F6Add || iteration < 2U) {
				AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
				AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot);
			}
		} else
			mem.PublishOutputAndRelease(slot);
	}

	__aicore__ inline AscendC::LocalTensor<T> LoadPair(uint32_t iteration, uint32_t slot, uint32_t pos,
													   AscendC::LocalTensor<T> firstRaw,
													   const AscendC::DataCopyExtParams &rp) {
		auto r = firstRaw;
		if (iteration != 0) {
			if constexpr (FIXED_TOKENS) {
				if (iteration >= 2U) {
					AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(slot);
					AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
				}
				r = mem.RawFixedSlot(slot);
			} else
				r = mem.AllocSlot(slot);
			if constexpr (BASIC2D) {
				AscendC::DataCopy(r, rawGm[pos * 6U],
								  AscendC::DataCopyParams{2U, uint16_t(ROW * 24U / 32U),
														  uint16_t((N - ROW) * 24U / 32U), 0U});
			} else
				AscendC::DataCopyPad(r, rawGm[pos * 6U], rp,
									 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
			AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(slot);
		}
		return r;
	}

	__aicore__ inline void PrepareOverlappedPrior(uint32_t iteration) {
		if constexpr (PRIOR_OVERLAP) {
			if (iteration == 0U) {
				AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
				FirePairPhases<ROW / 64U, VA_GROUP38>::Prior(mem.PackedVA());
			}
		}
	}

	__aicore__ inline void TransformPair(uint32_t iteration, uint32_t slot, uint32_t pos,
										 AscendC::LocalTensor<T> firstRaw, AscendC::LocalTensor<float> one,
										 AscendC::LocalTensor<float> *f,
										 const AscendC::DataCopyExtParams &rp) {
		auto r = LoadPair(iteration, slot, pos, firstRaw, rp);
		AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot);
		AscendC::LocalTensor<float> rf;
		if constexpr (sizeof(T) == sizeof(float))
			rf = r.template ReinterpretCast<float>();
		else {
			rf = mem.RawFloat();
			AscendC::Cast(rf, r, AscendC::RoundMode::CAST_NONE, FRAME * 6U);
		}
		if constexpr (SIGNED_INPUT) {
			auto signedRaw = rf.template ReinterpretCast<int32_t>();
			AscendC::Adds<int32_t>(signedRaw, signedRaw, int32_t(-2147483647 - 1), 9984U);
		} else {
			AscendC::Muls(rf, rf, -1.0f, FRAME * 6U);
		}
		AscendC::Exp(rf, rf, FRAME * 6U);
		PrepareOverlappedPrior(iteration);
		AscendC::Adds(rf, rf, 1.0f, FRAME * 6U);
		const AscendC::BinaryRepeatParams divBinary{1, 0, 1, 8, 0, 8};
		if constexpr (!SCORE_FUSED)
			AscendC::Div<float>(rf, one, rf, uint64_t(64), uint8_t(FRAME * 6U / 64U), divBinary);
		AscendC::PipeBarrier<PIPE_V>();
		if constexpr (sizeof(T) == sizeof(float)) {
			if (slot == 0)
				Input<0>();
			else
				Input<1>();
		} else
			Input<0>();
		AscendC::PipeBarrier<PIPE_V>();
		FuseScore(one, f, divBinary);
	}

	__aicore__ inline void PreparePrior(AscendC::LocalTensor<float> p, AscendC::LocalTensor<float> *f) {
		if constexpr (!PRIOR_OVERLAP) {
			AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
			FirePairPhases<ROW / 64U, VA_GROUP38>::Prior(mem.PackedVA());
			AscendC::PipeBarrier<PIPE_V>();
		}
		if constexpr (sizeof(T) != sizeof(float) || (FIXED576 && BASE_IMAGE)) {
			FireCopyPriorScale<STRIDE>(p, f[8]);
		}
	}

	__aicore__ inline void ProcessPair(uint32_t iteration, uint32_t cohort, uint32_t begin,
									   AscendC::LocalTensor<T> firstRaw, AscendC::LocalTensor<float> p,
									   AscendC::LocalTensor<float> one, AscendC::LocalTensor<float> *f,
									   const AscendC::DataCopyExtParams &rp) {
		const uint32_t pair = ROTATE_PAIRS ? ((iteration + cohort) & 3U) : iteration;
		const uint32_t slot = iteration & 1U;
		const uint32_t pos = pair * 2U * N + begin;
		TransformPair(iteration, slot, pos, firstRaw, one, f, rp);
		DecodeCoordinates(p, f);
		StorePair(iteration, slot, pos);
	}

	__aicore__ inline void LoadInitialPair(uint32_t begin, uint32_t cohort, AscendC::LocalTensor<T> firstRaw,
										   AscendC::LocalTensor<float> p,
										   const AscendC::DataCopyExtParams &rp,
										   const AscendC::DataCopyExtParams &pp) {
		if constexpr (BASIC2D) {
			AscendC::DataCopy(
				firstRaw, rawGm[(2U * cohort * N + begin) * 6U],
				AscendC::DataCopyParams{2U, uint16_t(ROW * 24U / 32U), uint16_t((N - ROW) * 24U / 32U), 0U});
		} else
			AscendC::DataCopyPad(firstRaw, rawGm[(2U * cohort * N + begin) * 6U], rp,
								 AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
		if constexpr (PRIOR_OVERLAP)
			AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
		if constexpr (!FIXED576 && !F5 && VA_GROUP38)
			AscendC::DataCopy(p, priorGm[begin * 5U],
							  AscendC::DataCopyParams{1U, uint16_t(ROW * 20U / 32U), 0U, 0U});
		else
			AscendC::DataCopyPad(p, priorGm[begin * 5U], pp,
								 AscendC::DataCopyPadExtParams<float>{false, 0, 0, 0});
		if constexpr (!PRIOR_OVERLAP)
			AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
		AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
	}

  public:
	__aicore__ inline void Run(GM_ADDR raw, GM_ADDR priors, GM_ADDR decoded) {
		const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
		if constexpr (ROW == 768U) {
			if (block >= 42U)
				return;
		} else {
			if (block >= 39U)
				return;
		}
		const uint32_t begin = ROW == 768U ? block * 768U : block * 832U;
		rawGm.SetGlobalBuffer((__gm__ T *)raw);
		priorGm.SetGlobalBuffer((__gm__ float *)priors);
		outGm.SetGlobalBuffer((__gm__ T *)decoded);
		const AscendC::DataCopyExtParams rp{2, ROW * 6U * uint32_t(sizeof(T)),
											(N - ROW) * 6U * uint32_t(sizeof(T)), 0, 0};
		const AscendC::DataCopyExtParams pp{1, ROW * 5U * 4U, 0, 0, 0};
		const uint32_t cohort = ROTATE_PAIRS ? (block & 3U) : 0U;
		auto firstRaw = mem.AllocSlot(0);
		auto p = mem.Prior();
		LoadInitialPair(begin, cohort, firstRaw, p, rp, pp);
		auto one = mem.Ones();
		AscendC::Duplicate(one, 1.0f, 8U);
		if constexpr (COORD_DIV2)
			AscendC::Duplicate(mem.Twos(), 2.0f, 8U);
		AscendC::LocalTensor<float> f[11];
		for (uint32_t k = 0; k < 11; ++k)
			f[k] = mem.Work()[k * STRIDE];
		InitPackedVA();
		PreparePrior(p, f);
		for (uint32_t iteration = 0; iteration < 4; ++iteration) {
			ProcessPair(iteration, cohort, begin, firstRaw, p, one, f, rp);
		}
		if constexpr (FIXED_TOKENS && MODE == FirePairMode::A2F6Add) {
			AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
			AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
		} else if constexpr (FIXED_TOKENS) {
			AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
			AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
			AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
			AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
		} else
			mem.Drain();
	}
};
