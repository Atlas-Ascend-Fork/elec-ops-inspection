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

__aicore__ inline void FireStorePacked(AscendC::LocalTensor<uint64_t> dst, const uint64_t *address) {
#pragma unroll
	for (uint32_t q = 0; q < 4; ++q) {
		const uint64_t value =
			((address[4 * q] >> 5) & 0x1fffULL) | (((address[4 * q + 1] >> 5) & 0x1fffULL) << 16) |
			(((address[4 * q + 2] >> 5) & 0x1fffULL) << 32) | (((address[4 * q + 3] >> 5) & 0x1fffULL) << 48);
		dst.SetValue(q, value);
	}
}

template <bool B16, uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void FireTranspose(AscendC::LocalTensor<uint64_t> dst, AscendC::LocalTensor<uint64_t> src) {
	auto *dstPacked = reinterpret_cast<__ubuf__ uint64_t *>(dst.GetPhyAddr());
	auto *srcPacked = reinterpret_cast<__ubuf__ uint64_t *>(src.GetPhyAddr());
	__cce_scalar::vld_va_reg(VA0, dstPacked, L128);
	__cce_scalar::vld_va_reg(VA1, dstPacked, H128);
	__cce_scalar::vld_va_reg(VA2, srcPacked, L128);
	__cce_scalar::vld_va_reg(VA3, srcPacked, H128);
	if constexpr (B16)
		__cce_scalar::scatter_vnchwconv_b16(VA0, VA2, REPEAT, DST_STRIDE, SRC_STRIDE);
	else
		__cce_scalar::scatter_vnchwconv_b32(VA0, VA2, REPEAT, DST_STRIDE, SRC_STRIDE);
}

template <uint8_t REPEAT, uint16_t DST_STRIDE, uint16_t SRC_STRIDE>
__aicore__ inline void FirePackedTranspose(AscendC::LocalTensor<uint64_t> dst,
										   AscendC::LocalTensor<uint64_t> src) {
	FireTranspose<false, REPEAT, DST_STRIDE, SRC_STRIDE>(dst, src);
}

template <uint32_t GROUPS> __aicore__ constexpr bool FireFrameVAGroupActive(uint32_t group) {
	return group < 3U || (group >= 6U && group < 9U) || (group >= 18U && group < 28U) ||
		   (group >= 33U && group < 43U) ||
		   (GROUPS == 38U && ((group >= 12U && group < 15U) || (group >= 28U && group < 33U)));
}

template <bool ENABLED, uint32_t OFFSET, uint32_t GROUPS> __aicore__ constexpr uint32_t FireFrameVAOffset() {
	if constexpr (!ENABLED)
		return OFFSET;
	else {
		static_assert(GROUPS == 28U || GROUPS == 38U, "qualified compressed VA layout");
		static_assert(OFFSET % 4U == 0U && OFFSET < 172U, "compressed VA original group bounds");
		constexpr uint32_t group = OFFSET / 4U;
		static_assert(FireFrameVAGroupActive<GROUPS>(group), "compressed VA original active group");
		constexpr uint32_t pitch = GROUPS == 38U ? 5U : 3U;
		if constexpr (group < 3U)
			return 4U * ((GROUPS == 38U ? 25U : 15U) + group);
		else if constexpr (group < 9U)
			return 4U * (pitch * (group - 6U));
		else if constexpr (GROUPS == 38U && group < 15U)
			return 4U * (1U + 5U * (group - 12U));
		else if constexpr (group < 23U)
			return 4U * ((GROUPS == 38U ? 28U : 18U) + group - 18U);
		else if constexpr (group < 28U)
			return 4U * ((GROUPS == 38U ? 2U : 1U) + pitch * (group - 23U));
		else if constexpr (GROUPS == 38U && group < 33U)
			return 4U * (3U + 5U * (group - 28U));
		else if constexpr (group < 38U)
			return 4U * ((GROUPS == 38U ? 33U : 23U) + group - 33U);
		else
			return 4U * ((GROUPS == 38U ? 4U : 2U) + pitch * (group - 38U));
	}
}

template <uint8_t REPEAT, uint16_t DST, uint16_t SRC>
__aicore__ inline void FireFramePackedTranspose(AscendC::LocalTensor<uint64_t> dst,
												AscendC::LocalTensor<uint64_t> src) {
	FirePackedTranspose<REPEAT, REPEAT == 1 ? 0 : DST, REPEAT == 1 ? 0 : SRC>(dst, src);
}
