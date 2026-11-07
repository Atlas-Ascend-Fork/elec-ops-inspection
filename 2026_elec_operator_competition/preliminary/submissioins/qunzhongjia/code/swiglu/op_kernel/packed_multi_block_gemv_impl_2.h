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
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18,
          class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26>
__aicore__ inline __attribute__((always_inline)) void
PackedMultiBlockGemv<T>::RunStage2f288430(A0& gb1, A1& gb2, A2& hw1, A3& hw2, A4& bothHalfW,
                                          A5& halfExpandedX, A6& hwCol32, A7& hxCol32, A8& bothBias, A9& fw1,
                                          A10& fw2, A11& bothW, A12& expandedX, A13& bothP, A14& bothBlock,
                                          A15& bothPair, A16& acc, A17& gu, A18& ones, A19& rawW1, A20& rawW2,
                                          A21& rawX, A22& rawB1, A23& rawB2, A24& offsets, A25& bc, A26& wp)
{
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopyPad(rawB1, gb1, bc, wp);
    DataCopyPad(rawB2, gb2, bc, wp);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    constexpr uint64_t mask = 0;
    Duplicate<float, false>(bothP[3456], 0.0f, mask, 10U, 1, 8);
    Duplicate<T, false>(ones, static_cast<T>(1), mask, 1U, 1, 8);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Gather(hw1, rawW1, offsets, 0, 1152);
    Gather(hw2, rawW2, offsets, 0, 1152);
    Gather(halfExpandedX, rawX, offsets[1152], 0, 192);
    Gather(hwCol32, rawW1, offsets[1344], 0, 112);
    Gather(hwCol32[128], rawW2, offsets[1344], 0, 112);
    Gather(hxCol32, rawX, offsets[1456], 0, 112);
    SetVectorMask<float, MaskMode::NORMAL>(0, ~0ULL);
    Cast<float, T, false>(bothW, bothHalfW, RoundMode::CAST_NONE, mask, 36U, UnaryRepeatParams{1, 1, 8, 4});
    Cast<float, T, false>(expandedX, halfExpandedX, RoundMode::CAST_NONE, mask, 3U,
                          UnaryRepeatParams{1, 1, 8, 4});
    const BinaryRepeatParams rp{1, 1, 1, 8, 0, 8};
#pragma unroll

    for (uint32_t i = 0; i < Rows; ++i) {
        Mul<float, false>(bothP[i * 1152U], expandedX[i * 64U], fw1, mask, 18U, rp);
        Mul<float, false>(bothP[4096U + i * 1152U], expandedX[i * 64U], fw2, mask, 18U, rp);
    }
    BlockReduceSum<float, false>(bothBlock, bothP, 118U, mask, 1, 1, 8);
    PairReduceSum<float, false>(bothPair, bothBlock, 15U, mask, 1, 1, 8);
    PairReduceSum<float, false>(acc, bothPair, 8U, mask, 1, 1, 8);
    MulAddDst<float, T, false>(acc, hxCol32, hwCol32, mask, 2U, BinaryRepeatParams{1, 1, 1, 8, 4, 4});
    MulAddDst<float, T, false>(acc[128], hxCol32, hwCol32[128], mask, 2U,
                               BinaryRepeatParams{1, 1, 1, 8, 4, 4});
    Cast<T, float, false>(gu, acc, RoundMode::CAST_RINT, mask, 4U, UnaryRepeatParams{1, 1, 4, 8});
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Gather(bothBias, rawB1, offsets[1568], 0, 112);
    Gather(bothBias[128], rawB2, offsets[1568], 0, 112);
    SetVectorMask<T, MaskMode::NORMAL>(~0ULL, ~0ULL);
    Add<T, false>(gu, gu, bothBias, mask, 2U, BinaryRepeatParams{1, 1, 1, 8, 8, 8});
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18,
          class A19, class A20>
__aicore__ inline __attribute__((always_inline)) void
PackedMultiBlockGemv<T>::RunStagedfff6edb(A0& tiling, A1& blk, A2& row, A3& gx, A4& gw1, A5& gw2, A6& gb1,
                                          A7& gb2, A8& gy, A9& hw1, A10& hw2, A11& bothHalfW,
                                          A12& halfExpandedX, A13& hwCol32, A14& hxCol32, A15& bothBias,
                                          A16& fw1, A17& fw2, A18& bothW, A19& expandedX, A20& bothP)
{
    LocalTensor<float> bothBlock(TPosition::VECCALC, 46848, 960);
    LocalTensor<float> bothPair(TPosition::VECCALC, 50688, 512);
    LocalTensor<float> acc(TPosition::VECCALC, 52736, 256);
    LocalTensor<T> gu(TPosition::VECCALC, 53760, 256);
    LocalTensor<T> sig(TPosition::VECCALC, 54272, 128);
    LocalTensor<T> ones(TPosition::VECCALC, 54528, 128);
    LocalTensor<T> out(TPosition::VECOUT, 54784, 128);
    LocalTensor<T> rawW1(TPosition::VECIN, 55040, 1104);
    LocalTensor<T> rawW2(TPosition::VECIN, 57344, 1104);
    LocalTensor<T> rawX(TPosition::VECIN, 59648, 112);
    LocalTensor<T> rawB1(TPosition::VECIN, 59904, 48);
    LocalTensor<T> rawB2(TPosition::VECIN, 60000, 48);
    LocalTensor<uint32_t> offsets(TPosition::VECIN, 60160, 1792);
    LocalTensor<T> packedOut(TPosition::VECOUT, 67328, 112);
    SetAtomicNone();
    SetMaskNorm();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    GlobalTensor<uint32_t> indexGm;
    indexGm.SetGlobalBuffer((__gm__ uint32_t*)(tiling + offsetof(SwiGluTilingData, gatherOffsetsB)), 1792);
    DataCopyExtParams wc{1, 2178, 0, 0, 0};
    DataCopyExtParams bc{1, 66, 0, 0, 0};
    DataCopyExtParams xcFlat{1, static_cast<uint32_t>((blk < 28U ? 198U : 66U)), 0, 0, 0};
    DataCopyPadExtParams<T> wp{true, 0, 15, static_cast<T>(0)};
    DataCopyPadExtParams<T> xp{true, 0, static_cast<uint8_t>(blk < 28U ? 13U : 15U), static_cast<T>(0)};
    DataCopyPad(rawW1, gw1, wc, wp);
    DataCopyPad(rawW2, gw2, wc, wp);
    DataCopyPad(rawX, gx[row * D], xcFlat, xp);
    DataCopy(offsets, indexGm, 1792);
    RunStage2f288430(gb1, gb2, hw1, hw2, bothHalfW, halfExpandedX, hwCol32, hxCol32, bothBias, fw1, fw2,
                     bothW, expandedX, bothP, bothBlock, bothPair, acc, gu, ones, rawW1, rawW2, rawX, rawB1,
                     rawB2, offsets, bc, wp);
    FastGate<T, 108U, false>(gu, gu[128], sig, out, ones);
    Gather(packedOut, out, offsets[1680], 0, Rows * D);
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopyPad(gy[row * D], packedOut, xcFlat);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T>
template <class B0, class B1>
__aicore__ inline __attribute__((always_inline)) void
PackedMultiBlockGemv<T>::SharedStageff98d898dd(B0& offsetsI, B1& urI)
{
    constexpr uint64_t mask = 0;
    Adds<int32_t, false>(offsetsI[32], offsetsI[0], 66, mask, 1U, urI);
    Adds<int32_t, false>(offsetsI[1024], offsetsI[0], 2112, mask, 1U, urI);
    Adds<int32_t, false>(offsetsI[1504], offsetsI[0], 0, mask, 1U, urI);
    Adds<int32_t, false>(offsetsI[1536], offsetsI[0], 0, mask, 1U, urI);
    Adds<int32_t, false>(offsetsI[1824], offsetsI[0], 0, mask, 1U, urI);
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    Adds<int32_t, false>(offsetsI[64], offsetsI[0], 132, mask, 1U, urI);
    Adds<int32_t, false>(offsetsI[128], offsetsI[0], 264, mask, 2U, urI);
    Adds<int32_t, false>(offsetsI[256], offsetsI[0], 528, mask, 4U, urI);
    Adds<int32_t, false>(offsetsI[512], offsetsI[0], 1056, mask, 8U, urI);
    Adds<int32_t, false>(offsetsI[1568], offsetsI[1504], 66, mask, 1U, urI);
    Adds<int32_t, false>(offsetsI[1632], offsetsI[1504], 132, mask, 2U, urI);
    Adds<int32_t, false>(offsetsI[1760], offsetsI[1504], 264, mask, 1U, urI);
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T>
template <class A0, class A1, class A2>
__aicore__ inline __attribute__((always_inline)) void
PackedMultiBlockGemv<T>::RunStage43538853(A0& offsets, A1& offsetsI, A2& offsets64)
{
    constexpr uint64_t mask = 0;
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    const UnaryRepeatParams urI{1, 1, 8, 8};
    SharedStageff98d898dd(offsetsI, urI);
    // Repeat the33-entry address seed directly in the910B compaction unit.
    // Only addresses are replicated; input tensors and arithmetic are unchanged.
    Duplicate<int32_t, false>(offsetsI[1824], 0, mask, 3U, 1, 8);
    offsets.SetValue(2208U, 64U);
    offsets64.SetValue(1040U, 0x1FFFFFFFFULL);
    offsets64.SetValue(1041U, 0ULL);
    SetFlag<HardEvent::S_V>(EVENT_ID1);
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0xFFFFFFFFULL);
    Adds<int32_t, false>(offsetsI[2176], offsetsI[0], 0, mask, 1U, urI);
    WaitFlag<HardEvent::S_V>(EVENT_ID1);
    SetMaskNorm();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    PipeBarrier<PIPE_V>();
    // Explicit per-repeat bitmap avoids hardware counter mode consuming only33 total items.
    vreducev2((__ubuf__ uint32_t*)offsets[1824].GetPhyAddr(), (__ubuf__ uint32_t*)offsets[2176].GetPhyAddr(),
              (__ubuf__ uint32_t*)offsets[2080].GetPhyAddr(), 5, 1, 0, 0, 0);
    PipeBarrier<PIPE_V>();
    Muls<int32_t, false>(offsetsI[1088], offsetsI[1824], 33, mask, 3U, urI);
    Adds<int32_t, false>(offsetsI[1088], offsetsI[1088], 64, mask, 2U, urI);
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, (1ULL << 37) - 1ULL);
    Adds<int32_t, false>(offsetsI[1216], offsetsI[1216], 64, mask, 1U, urI);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T>
template <class B0, class B1, class B2>
__aicore__ inline __attribute__((always_inline)) void
PackedMultiBlockGemv<T>::SharedStage32aa9531e9(B0& bothP, B1& offsets64, B2& offsetsI)
{
    constexpr uint64_t mask = 0;
#pragma unroll

    for (uint32_t k = 0; k < 16U; ++k) {
        const uint64_t c4 = k * 4U;
        offsets64.SetValue(k, ((c4 + 2U) << 32) | c4);
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate<float, false>(bothP[5280], 0.0f, mask, 6U, 1, 8);
    Duplicate<float, false>(bothP[10912], 0.0f, mask, 6U, 1, 8);
    Duplicate<int32_t, false>(offsetsI[1024], 0, mask, 1U, 1, 8);
    Duplicate<int32_t, false>(offsetsI[1408], 0, mask, 2U, 1, 8);
    Duplicate<int32_t, false>(offsetsI[1992], 0, mask, 1U, 1, 8);
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0x1FFFFFFFFULL);
    Duplicate<int32_t, false>(offsetsI[1280], 64, mask, 1U, 1, 8);
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0x3FFFFFFFEULL);
    Duplicate<int32_t, false>(offsetsI[1312], 130, mask, 1U, 1, 8);
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0x7FFFFFFFCULL);
    Duplicate<int32_t, false>(offsetsI[1344], 196, mask, 1U, 1, 8);
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0xFFFFFFFF8ULL);
    Duplicate<int32_t, false>(offsetsI[1376], 262, mask, 1U, 1, 8);
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0x1FFFFFFFF0ULL);
    Duplicate<int32_t, false>(offsetsI[1408], 328, mask, 1U, 1, 8);
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0xFFFFFFFFULL);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
#if !((defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12>
__aicore__ inline __attribute__((always_inline)) void
PackedMultiBlockGemv<T>::RunSharedStage1(A0& gb1, A1& gb2, A2& hxAll, A3& rawB1, A4& rawB2, A5& bothP,
                                         A6& rawX, A7& offsets, A8& offsetsI, A9& allHalf, A10& allF, A11& bc,
                                         A12& wp)
#endif
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <typename T>
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15>
    __aicore__ inline __attribute__((always_inline)) void PackedMultiBlockGemv<T>::RunSharedStage1(
        A0& gw2, A1& gb1, A2& gb2, A3& hxAll, A4& rawB1, A5& rawB2, A6& bothP, A7& rawW2, A8& rawX,
        A9& offsets, A10& offsetsI, A11& allHalf, A12& allF, A13& wc, A14& bc, A15& wp)
#endif
{
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    DataCopyPad(rawW2, gw2, wc, wp);
#else
#endif
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    DataCopyPad(rawB1, gb1, bc, wp);
    DataCopyPad(rawB2, gb2, bc, wp);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    constexpr uint64_t mask = 0;
    auto offsets64 = offsets.template ReinterpretCast<uint64_t>();
    SharedStage32aa9531e9(bothP, offsets64, offsetsI);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    RunStage43538853(offsets, offsetsI, offsets64);
#else
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    RunStage52d1eeb6(offsetsI, offsets64);
#endif
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    Gather(hxAll, rawX, offsets[1280], 0, 544);
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    Cast<float, T, false>(allF[2560], allHalf[2560], RoundMode::CAST_NONE, mask, 9U,
                          UnaryRepeatParams{1, 1, 8, 4});
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T>
template <class A0, class A1>
__aicore__ inline __attribute__((always_inline)) void PackedMultiBlockGemv<T>::RunStage52d1eeb6(A0& offsetsI,
                                                                                                A1& offsets64)
{
    constexpr uint64_t mask = 0;
    const UnaryRepeatParams urI{1, 1, 8, 8};
    SharedStageff98d898dd(offsetsI, urI);
#pragma unroll

    for (uint32_t k = 0; k < 16U; ++k) {
        const uint64_t c4 = k * 4U;
        const uint64_t valA = ((c4 + 2U) << 32) | c4;
        const uint64_t valB = (c4 << 32) | (k == 0U ? 64ULL : c4 - 2U);
        offsets64.SetValue(945U + k, valA);
        offsets64.SetValue(978U + k, valA);
        offsets64.SetValue(928U + k, valB);
        offsets64.SetValue(961U + k, valB);
    }
    offsets64.SetValue(944U, (64ULL << 32) | 62ULL);
    offsets64.SetValue(977U, (64ULL << 32) | 62ULL);
    offsets64.SetValue(994U, 64ULL);
    offsets64.SetValue(995U, 0ULL);
    SetFlag<HardEvent::S_V>(EVENT_ID1);
    WaitFlag<HardEvent::S_V>(EVENT_ID1);
    Muls<int32_t, false>(offsetsI[1088], offsetsI[1824], 33, mask, 3U, urI);
    Adds<int32_t, false>(offsetsI[1088], offsetsI[1088], 64, mask, 2U, urI);
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, (1ULL << 37) - 1ULL);
    Adds<int32_t, false>(offsetsI[1216], offsetsI[1216], 64, mask, 1U, urI);
}

#endif
