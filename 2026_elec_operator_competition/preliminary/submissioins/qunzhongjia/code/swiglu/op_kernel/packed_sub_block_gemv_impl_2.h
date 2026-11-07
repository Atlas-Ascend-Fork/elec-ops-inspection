/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Copyright (c) 2026 黎何.
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
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
#if !((defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::RunSharedStage0(A0& hw1, A1& hw2, A2& sig, A3& out, A4& rawW1, A5& rawW2, A6& rawX,
                                       A7& offsets, A8& offsetsI, A9& bothP)
#endif
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <typename T>
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12>
    __aicore__ inline __attribute__((always_inline)) void PackedSubBlockGemv<T>::RunSharedStage0(
        A0& hw1, A1& hw2, A2& sig, A3& out, A4& rawW1, A5& rawW2, A6& rawX, A7& offsets, A8& offsetsI,
        A9& bothP, A10& bothBlock, A11& acc, A12& bothBias)
#endif
{
    constexpr uint64_t mask = 0;
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
#else
    LocalTensor<float> bothBlock(TPosition::VECCALC, 34048, 192);
    LocalTensor<float> acc(TPosition::VECCALC, AB, 96);
    LocalTensor<T> bothBias(TPosition::VECIN, HB, 32);
#endif
    LocalTensor<T> gu(TPosition::VECCALC, GB, 96);
    LocalTensor<T> bothHalfW(TPosition::VECIN, 0, 512);
    LocalTensor<T> halfExpandedX(TPosition::VECCALC, 10752, 192);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    auto offsets64 = offsets.template ReinterpretCast<uint64_t>();
    RunStage058d2452(offsetsI, bothP, offsets64);
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
#else
    RunStaged70b3486(offsets, offsetsI, bothP);
#endif
    Gather(halfExpandedX, rawX, offsets[256], 0, 192);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Gather(hw1, rawW1, offsets, 0, W);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    Gather(hw2, rawW2, offsets, 0, W);
    SharedStage11510c1ed8(acc, bothBlock, bothHalfW, bothP, gu, halfExpandedX);
    SharedStaged379073a67(bothBias, gu, out, sig);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T>
template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8, class B9,
          class B10, class B11, class B12, class B13>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::SharedStagee9c87e4530(B0& bc, B1& gb1, B2& gb2, B3& gw1, B4& gw2, B5& gx, B6& hb1,
                                             B7& hb2, B8& rawW1, B9& rawW2, B10& rawX, B11& row, B12& wc,
                                             B13& xcFlat)
{
    DataCopyPadExtParams<T> wp{true, 0, 7, static_cast<T>(0)};
    DataCopyPadExtParams<T> bp{true, 0, 3, static_cast<T>(0)};
    DataCopyPadExtParams<T> xp{true, 0, 9, static_cast<T>(0)};
    DataCopyPad(rawX, gx[row * D], xcFlat, xp);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
    DataCopyPad(rawW1, gw1, wc, wp);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopyPad(rawW2, gw2, wc, wp);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    DataCopyPad(hb1, gb1, bc, bp);
    DataCopyPad(hb2, gb2, bc, bp);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    constexpr uint64_t mask = 0;
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T>
template <class A0, class A1, class A2>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::RunStaged70b3486(A0& offsets, A1& offsetsI, A2& bothP)
{
    constexpr uint64_t mask = 0;
    auto offsets64 = offsets.template ReinterpretCast<uint64_t>();
#pragma unroll

    for (uint32_t k = 0; k < 6U; ++k) {
        const uint64_t c4 = k * 4U;
        const uint64_t valA = ((c4 + 2U) << 32) | c4;
        offsets64.SetValue(k, valA);
    }
    offsets64.SetValue(6U, (338ULL << 32) | 24ULL);
    offsets64.SetValue(7U, (338ULL << 32) | 338ULL);
    offsets64.SetValue(224U, 0x1FFFULL);
    offsets64.SetValue(225U, 0ULL);
    SharedStage1a9cd481b9(bothP, offsetsI);
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
}

#endif
