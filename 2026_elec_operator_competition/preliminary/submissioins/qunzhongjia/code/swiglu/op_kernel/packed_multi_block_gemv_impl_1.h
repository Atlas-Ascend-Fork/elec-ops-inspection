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
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18,
          class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27,
          class A28, class A29>
__aicore__ inline __attribute__((always_inline)) void
PackedMultiBlockGemv<T>::RunSharedStage0(A0& hw1, A1& hw2, A2& hxAll, A3& rawB1, A4& rawB2, A5& hb1, A6& hb2,
                                         A7& fw1, A8& fw1Last, A9& fw2, A10& fw2Last, A11& fxLast, A12& fx,
                                         A13& bothP, A14& blkP, A15& pairP, A16& ag, A17& lastG, A18& lastU,
                                         A19& g, A20& u, A21& sig, A22& ones, A23& out, A24& rawW1,
                                         A25& rawW2, A26& rawX, A27& offsets, A28& allHalf, A29& allF)
{
    constexpr uint64_t mask = 0;
    Duplicate<float, false>(bothP[10912], 0.0f, mask, 6U, 1, 8);
    Duplicate<T, false>(ones, static_cast<T>(1), mask, 2U, 1, 8);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Gather(hw1, rawW1, offsets, 0, 1280);
    Gather(hw2, rawW2, offsets, 0, 1280);
    Gather(hxAll, rawX, offsets[1280], 0, 544);
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    Cast<float, T, false>(allF, allHalf, RoundMode::CAST_NONE, mask, 49U, UnaryRepeatParams{1, 1, 8, 4});
    const BinaryRepeatParams rp{1, 1, 1, 8, 0, 8};
    const BinaryRepeatParams br8{1, 1, 1, 8, 8, 8};
#pragma unroll
    for (uint32_t i = 0; i < Rows; ++i) {
        Mul<float, false>(bothP[i * W], fx[i * 64U], fw1, mask, 17U, rp);
        Mul<float, false>(bothP[5632U + i * W], fx[i * 64U], fw2, mask, 17U, rp);
    }
    BlockReduceSum<float, false>(blkP, bothP, 176U, mask, 1, 1, 8);
    PairReduceSum<float, false>(pairP, blkP, 22U, mask, 1, 1, 8);
    PairReduceSum<float, false>(ag, pairP, 11U, mask, 1, 1, 8);
    Mul<float, false>(lastG, fxLast, fw1Last, mask, 3U, br8);
    Mul<float, false>(lastU, fxLast, fw2Last, mask, 3U, br8);
    Add<float, false>(ag, ag, lastG, mask, 6U, br8);
    Cast<T, float, false>(g, ag, RoundMode::CAST_RINT, mask, 6U, UnaryRepeatParams{1, 1, 4, 8});
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Gather(hb1, rawB1, offsets[1824], 0, 176);
    Gather(hb2, rawB2, offsets[1824], 0, 176);
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    Add<T, false>(g, g, hb1, mask, 3U, br8);
    const UnaryRepeatParams ur8{1, 1, 8, 8};
    Muls<T, false>(sig, g, static_cast<T>(-1), mask, 2U, ur8);
    Exp<T, false>(sig, sig, mask, 2U, ur8);
    Adds<T, false>(sig, sig, static_cast<T>(1), mask, 2U, ur8);
    Div<T, false>(sig, ones, sig, mask, 2U, br8);
    Mul<T, false>(out, g, sig, mask, 2U, br8);
    Mul<T, false>(out, out, u, mask, 2U, br8);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18,
          class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26>
__aicore__ inline __attribute__((always_inline)) void PackedMultiBlockGemv<T>::RunSharedStage0(
    A0& hw1, A1& hw2, A2& rawB1, A3& rawB2, A4& hb1, A5& hb2, A6& fw1, A7& fw1Last, A8& fw2, A9& fw2Last,
    A10& fxLast, A11& fx, A12& bothP, A13& blkP, A14& pairP, A15& ag, A16& lastG, A17& lastU, A18& g, A19& u,
    A20& sig, A21& out, A22& rawW1, A23& rawW2, A24& offsets, A25& allHalf, A26& allF)
{
    constexpr uint64_t mask = 0;
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Gather(hw1, rawW1, offsets, 0, 1280);
    Cast<float, T, false>(allF, allHalf, RoundMode::CAST_NONE, mask, 20U, UnaryRepeatParams{1, 1, 8, 4});
    const BinaryRepeatParams rp{1, 1, 1, 8, 0, 8};
    const BinaryRepeatParams br8{1, 1, 1, 8, 8, 8};
#pragma unroll

    for (uint32_t i = 0; i < Rows; ++i) {
        Mul<float, false>(bothP[i * W], fx[i * 64U], fw1, mask, 17U, rp);
    }
    Mul<float, false>(lastG, fxLast, fw1Last, mask, 3U, br8);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    Gather(hw2, rawW2, offsets, 0, 1280);
    Cast<float, T, false>(allF[1280], allHalf[1280], RoundMode::CAST_NONE, mask, 20U,
                          UnaryRepeatParams{1, 1, 8, 4});
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Gather(hb1, rawB1, offsets[1824], 0, 176);
    Gather(hb2, rawB2, offsets[1824], 0, 176);
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
#pragma unroll

    for (uint32_t i = 0; i < Rows; ++i) {
        Mul<float, false>(bothP[5632U + i * W], fx[i * 64U], fw2, mask, 17U, rp);
    }
    BlockReduceSum<float, false>(blkP, bothP, 176U, mask, 1, 1, 8);
    PairReduceSum<float, false>(pairP, blkP, 22U, mask, 1, 1, 8);
    PairReduceSum<float, false>(ag, pairP, 11U, mask, 1, 1, 8);
    Mul<float, false>(lastU, fxLast, fw2Last, mask, 3U, br8);
    Add<float, false>(ag, ag, lastG, mask, 6U, br8);
    Cast<T, float, false>(g, ag, RoundMode::CAST_RINT, mask, 6U, UnaryRepeatParams{1, 1, 4, 8});
    Add<T, false>(g, g, hb1, mask, 3U, br8);
    const UnaryRepeatParams ur8{1, 1, 8, 8};
    Muls<T, false>(sig, g, static_cast<T>(-1), mask, 2U, ur8);
    Exp<T, false>(sig, sig, mask, 2U, ur8);
    Adds<T, false>(sig, sig, static_cast<T>(1), mask, 2U, ur8);
    Div<T, false>(sig, g, sig, mask, 2U, br8);
    Mul<T, false>(out, sig, u, mask, 2U, br8);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T>
template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8, class B9,
          class B10, class B11, class B12, class B13, class B14, class B15, class B16, class B17, class B18,
          class B19, class B20, class B21, class B22, class B23, class B24, class B25, class B26, class B27,
          class B28, class B29, class B30, class B31, class B32, class B33, class B34, class B35, class B36,
          class B37, class B38, class B39>
__aicore__ inline __attribute__((always_inline)) void PackedMultiBlockGemv<T>::LoadAndComputeTile(
    B0& ag, B1& allF, B2& allHalf, B3& bc, B4& blkP, B5& bothP, B6& fw1, B7& fw1Last, B8& fw2, B9& fw2Last,
    B10& fx, B11& fxLast, B12& g, B13& gb1, B14& gb2, B15& gx, B16& gy, B17& hb1, B18& hb2, B19& hw1,
    B20& hw2, B21& hxAll, B22& indexGm, B23& lastG, B24& lastU, B25& offsets, B26& ones, B27& out, B28& pairP,
    B29& rawB1, B30& rawB2, B31& rawW1, B32& rawW2, B33& rawX, B34& row, B35& sig, B36& u, B37& wp,
    B38& xcFlat, B39& xp)
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL))
    template <typename T>
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8,
              class B9, class B10, class B11, class B12, class B13, class B14, class B15, class B16,
              class B17, class B18, class B19, class B20, class B21, class B22, class B23, class B24,
              class B25, class B26, class B27, class B28, class B29, class B30, class B31, class B32,
              class B33, class B34, class B35, class B36, class B37, class B38, class B39, class B40,
              class B41>
    __aicore__ inline __attribute__((always_inline)) void PackedMultiBlockGemv<T>::LoadAndComputeTile(
        B0& ag, B1& allF, B2& allHalf, B3& bc, B4& blkP, B5& bothP, B6& fw1, B7& fw1Last, B8& fw2,
        B9& fw2Last, B10& fx, B11& fxLast, B12& g, B13& gb1, B14& gb2, B15& gw1, B16& gw2, B17& gx, B18& gy,
        B19& hb1, B20& hb2, B21& hw1, B22& hw2, B23& hxAll, B24& lastG, B25& lastU, B26& offsets,
        B27& offsetsI, B28& out, B29& pairP, B30& rawB1, B31& rawB2, B32& rawW1, B33& rawW2, B34& rawX,
        B35& row, B36& sig, B37& u, B38& wc, B39& wp, B40& xcFlat, B41& xp)
#endif
{
    DataCopyPad(rawX, gx[row * D], xcFlat, xp);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    DataCopy(offsets, indexGm, 2000);
#else
    SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
    DataCopyPad(rawW1, gw1, wc, wp);
#endif
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    DataCopyPad(rawB1, gb1, bc, wp);
    DataCopyPad(rawB2, gb2, bc, wp);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    constexpr uint64_t mask = 0;
    Duplicate<float, false>(bothP[5280], 0.0f, mask, 6U, 1, 8);
    RunSharedStage0(hw1, hw2, hxAll, rawB1, rawB2, hb1, hb2, fw1, fw1Last, fw2, fw2Last, fxLast, fx, bothP,
                    blkP, pairP, ag, lastG, lastU, g, u, sig, ones, out, rawW1, rawW2, rawX, offsets, allHalf,
                    allF);
#else
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    RunSharedStage1(gw2, gb1, gb2, hxAll, rawB1, rawB2, bothP, rawW2, rawX, offsets, offsetsI, allHalf, allF,
                    wc, bc, wp);
#else
    DataCopyPad(rawW2, gw2, wc, wp);
    RunSharedStage1(gb1, gb2, hxAll, rawB1, rawB2, bothP, rawX, offsets, offsetsI, allHalf, allF, bc, wp);
#endif
    RunSharedStage0(hw1, hw2, rawB1, rawB2, hb1, hb2, fw1, fw1Last, fw2, fw2Last, fxLast, fx, bothP, blkP,
                    pairP, ag, lastG, lastU, g, u, sig, out, rawW1, rawW2, offsets, allHalf, allF);
#endif
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopyPad(gy[row * D], out, xcFlat);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T>
template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8, class B9,
          class B10, class B11, class B12, class B13, class B14, class B15, class B16, class B17, class B18,
          class B19, class B20, class B21, class B22, class B23, class B24, class B25, class B26, class B27,
          class B28, class B29, class B30, class B31, class B32, class B33, class B34>
__aicore__ inline __attribute__((always_inline)) void PackedMultiBlockGemv<T>::FinishTileLayout(
    B0& ag, B1& blkP, B2& bothP, B3& fw1, B4& fw1Last, B5& fw2, B6& fw2Last, B7& fx, B8& fxLast, B9& g,
    B10& gb1, B11& gb2, B12& gw1, B13& gw2, B14& gx, B15& gy, B16& hb1, B17& hb2, B18& hw1, B19& hw2,
    B20& hxAll, B21& lastG, B22& lastU, B23& ones, B24& out, B25& pairP, B26& rawB1, B27& rawB2, B28& rawW1,
    B29& rawW2, B30& rawX, B31& row, B32& sig, B33& tiling, B34& u)
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL))
    template <typename T>
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8,
              class B9, class B10, class B11, class B12, class B13, class B14, class B15, class B16,
              class B17, class B18, class B19, class B20, class B21, class B22, class B23, class B24,
              class B25, class B26, class B27, class B28, class B29, class B30, class B31, class B32>
    __aicore__ inline __attribute__((always_inline)) void PackedMultiBlockGemv<T>::FinishTileLayout(
        B0& ag, B1& blkP, B2& bothP, B3& fw1, B4& fw1Last, B5& fw2, B6& fw2Last, B7& fx, B8& fxLast, B9& g,
        B10& gb1, B11& gb2, B12& gw1, B13& gw2, B14& gx, B15& gy, B16& hb1, B17& hb2, B18& hw1, B19& hw2,
        B20& hxAll, B21& lastG, B22& lastU, B23& out, B24& pairP, B25& rawB1, B26& rawB2, B27& rawW1,
        B28& rawW2, B29& rawX, B30& row, B31& sig, B32& u)
#endif
{
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    LocalTensor<uint32_t> offsets(TPosition::VECIN, 83968, 2000);
#elif (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    LocalTensor<uint32_t> offsets(TPosition::VECIN, 83968, 2240);
    LocalTensor<int32_t> offsetsI(TPosition::VECIN, 83968, 2240);
#else
    LocalTensor<uint32_t> offsets(TPosition::VECIN, 83968, 2056);
    LocalTensor<int32_t> offsetsI(TPosition::VECIN, 83968, 2056);
#endif
    LocalTensor<T> allHalf(TPosition::VECIN, 0, 3136);
    LocalTensor<float> allF(TPosition::VECCALC, FB, 3136);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    GlobalTensor<uint32_t> indexGm;
    indexGm.SetGlobalBuffer((__gm__ uint32_t*)(tiling + offsetof(SwiGluTilingData, gatherOffsetsB)), 2000);
#else
#endif
    DataCopyExtParams wc{1, 2178, 0, 0, 0};
    DataCopyExtParams bc{1, 66, 0, 0, 0};
    DataCopyExtParams xcFlat{1, 330, 0, 0, 0};
    DataCopyPadExtParams<T> wp{true, 0, 15, static_cast<T>(0)};
    DataCopyPadExtParams<T> xp{true, 0, 11, static_cast<T>(0)};
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    DataCopyPad(rawW1, gw1, wc, wp);
    DataCopyPad(rawW2, gw2, wc, wp);
    LoadAndComputeTile(ag, allF, allHalf, bc, blkP, bothP, fw1, fw1Last, fw2, fw2Last, fx, fxLast, g, gb1,
                       gb2, gx, gy, hb1, hb2, hw1, hw2, hxAll, indexGm, lastG, lastU, offsets, ones, out,
                       pairP, rawB1, rawB2, rawW1, rawW2, rawX, row, sig, u, wp, xcFlat, xp);
#else
    LoadAndComputeTile(ag, allF, allHalf, bc, blkP, bothP, fw1, fw1Last, fw2, fw2Last, fx, fxLast, g, gb1,
                       gb2, gw1, gw2, gx, gy, hb1, hb2, hw1, hw2, hxAll, lastG, lastU, offsets, offsetsI, out,
                       pairP, rawB1, rawB2, rawW1, rawW2, rawX, row, sig, u, wc, wp, xcFlat, xp);
#endif
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18,
          class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27>
__aicore__ inline __attribute__((always_inline)) void PackedMultiBlockGemv<T>::RunSharedStage2(
    A0& tiling, A1& row, A2& gx, A3& gw1, A4& gw2, A5& gb1, A6& gb2, A7& gy, A8& hw1, A9& hw2, A10& hxAll,
    A11& rawB1, A12& rawB2, A13& hb1, A14& hb2, A15& fw1, A16& fw1Last, A17& fw2, A18& fw2Last, A19& fxLast,
    A20& fx, A21& bothP, A22& blkP, A23& pairP, A24& ag, A25& lastG, A26& lastU, A27& g)
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
    template <typename T>
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20, class A21>
    __aicore__ inline __attribute__((always_inline)) void PackedMultiBlockGemv<T>::RunSharedStage2(
        A0& row, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& gy, A7& hw1, A8& hw2, A9& hxAll, A10& rawB1,
        A11& rawB2, A12& hb1, A13& hb2, A14& fw1, A15& fw1Last, A16& fw2, A17& fw2Last, A18& fxLast, A19& fx,
        A20& bothP, A21& blkP)
#endif
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
        template <typename T>
        template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
                  class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
                  class A17, class A18, class A19, class A20>
        __aicore__ inline __attribute__((always_inline)) void PackedMultiBlockGemv<T>::RunSharedStage2(
            A0& row, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& gy, A7& hw1, A8& hw2, A9& hxAll,
            A10& rawB1, A11& rawB2, A12& hb1, A13& hb2, A14& fw1, A15& fw1Last, A16& fw2, A17& fw2Last,
            A18& fxLast, A19& fx, A20& bothP)
#endif
{
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
#else
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    LocalTensor<float> blkP(TPosition::VECCALC, BKB, 1408);
#else
#endif
    LocalTensor<float> pairP(TPosition::VECCALC, PRB, 704);
    LocalTensor<float> ag(TPosition::VECCALC, AB, 384);
    LocalTensor<float> lastG(TPosition::VECCALC, LB, 384);
    LocalTensor<float> lastU(TPosition::VECCALC, LB + 704U, 192);
    LocalTensor<T> g(TPosition::VECCALC, GB, 384);
#endif
    LocalTensor<T> u(TPosition::VECCALC, GB + 352U, 256);
    LocalTensor<T> sig(TPosition::VECCALC, GB + 768U, 256);
    LocalTensor<T> ones(TPosition::VECCALC, GB + 1280U, 256);
    LocalTensor<T> out(TPosition::VECOUT, GB + 1792U, 256);
    SetAtomicNone();
    SetMaskNorm();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    LocalTensor<T> rawW1(TPosition::VECIN, 79104, 1104);
    LocalTensor<T> rawW2(TPosition::VECIN, 81312, 1104);
    LocalTensor<T> rawX(TPosition::VECIN, 83520, 176);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    FinishTileLayout(ag, blkP, bothP, fw1, fw1Last, fw2, fw2Last, fx, fxLast, g, gb1, gb2, gw1, gw2, gx, gy,
                     hb1, hb2, hw1, hw2, hxAll, lastG, lastU, ones, out, pairP, rawB1, rawB2, rawW1, rawW2,
                     rawX, row, sig, tiling, u);
#else
    FinishTileLayout(ag, blkP, bothP, fw1, fw1Last, fw2, fw2Last, fx, fxLast, g, gb1, gb2, gw1, gw2, gx, gy,
                     hb1, hb2, hw1, hw2, hxAll, lastG, lastU, out, pairP, rawB1, rawB2, rawW1, rawW2, rawX,
                     row, sig, u);
#endif
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T>
template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8>
__aicore__ inline __attribute__((always_inline)) void
PackedMultiBlockGemv<T>::RunTile(B0& blk, B1& gb1, B2& gb2, B3& gw1, B4& gw2, B5& gx, B6& gy, B7& row,
                                 B8& tiling)
{
    LocalTensor<T> hw1(TPosition::VECIN, 0, 1152);
    LocalTensor<T> hw2(TPosition::VECIN, 2304, 1152);
    LocalTensor<T> bothHalfW(TPosition::VECIN, 0, 2304);
    LocalTensor<T> halfExpandedX(TPosition::VECIN, 4608, 192);
    LocalTensor<T> hwCol32(TPosition::VECIN, 5120, 256);
    LocalTensor<T> hxCol32(TPosition::VECIN, 5632, 128);
    LocalTensor<T> bothBias(TPosition::VECIN, 5888, 256);
    LocalTensor<float> fw1(TPosition::VECCALC, 6400, 1152);
    LocalTensor<float> fw2(TPosition::VECCALC, 11008, 1152);
    LocalTensor<float> bothW(TPosition::VECCALC, 6400, 2304);
    LocalTensor<float> expandedX(TPosition::VECCALC, 15680, 192);
    LocalTensor<float> bothP(TPosition::VECCALC, 16512, 7552);
    RunStagedfff6edb(tiling, blk, row, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, bothHalfW, halfExpandedX,
                     hwCol32, hxCol32, bothBias, fw1, fw2, bothW, expandedX, bothP);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T>
template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7>
__aicore__ inline __attribute__((always_inline)) void
PackedMultiBlockGemv<T>::RunTile(B0& gb1, B1& gb2, B2& gw1, B3& gw2, B4& gx, B5& gy, B6& row, B7& tiling)
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL))
    template <typename T>
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6>
    __aicore__ inline
    __attribute__((always_inline)) void PackedMultiBlockGemv<T>::RunTile(B0& gb1, B1& gb2, B2& gw1, B3& gw2,
                                                                         B4& gx, B5& gy, B6& row)
#endif
{
    LocalTensor<T> hw1(TPosition::VECIN, 0, 1280);
    LocalTensor<T> hw2(TPosition::VECIN, 2560, 1280);
    LocalTensor<T> hxAll(TPosition::VECIN, 5120, 544);
    LocalTensor<T> rawB1(TPosition::VECIN, 6272, K);
    LocalTensor<T> rawB2(TPosition::VECIN, 6368, K);
    LocalTensor<T> hb1(TPosition::VECIN, 6464, 384);
    LocalTensor<T> hb2(TPosition::VECIN, 6816, 192);
    LocalTensor<float> fw1(TPosition::VECCALC, FB, 1088);
    LocalTensor<float> fw1Last(TPosition::VECCALC, FB + 4352U, 192);
    LocalTensor<float> fw2(TPosition::VECCALC, FB + 5120U, 1088);
    LocalTensor<float> fw2Last(TPosition::VECCALC, FB + 9472U, 192);
    LocalTensor<float> fxLast(TPosition::VECCALC, FB + 10240U, 224);
    LocalTensor<float> fx(TPosition::VECCALC, FB + 11136U, Rows * 64U);
    LocalTensor<float> bothP(TPosition::VECCALC, PB, 11264);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    LocalTensor<float> blkP(TPosition::VECCALC, BKB, 1408);
    LocalTensor<float> pairP(TPosition::VECCALC, PRB, 704);
    LocalTensor<float> ag(TPosition::VECCALC, AB, 384);
    LocalTensor<float> lastG(TPosition::VECCALC, LB, 384);
    LocalTensor<float> lastU(TPosition::VECCALC, LB + 704U, 192);
    LocalTensor<T> g(TPosition::VECCALC, GB, 384);
    RunSharedStage2(tiling, row, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hxAll, rawB1, rawB2, hb1, hb2, fw1,
                    fw1Last, fw2, fw2Last, fxLast, fx, bothP, blkP, pairP, ag, lastG, lastU, g);
#elif (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    RunSharedStage2(row, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hxAll, rawB1, rawB2, hb1, hb2, fw1, fw1Last,
                    fw2, fw2Last, fxLast, fx, bothP);
#else
    LocalTensor<float> blkP(TPosition::VECCALC, BKB, 1408);
    RunSharedStage2(row, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hxAll, rawB1, rawB2, hb1, hb2, fw1, fw1Last,
                    fw2, fw2Last, fxLast, fx, bothP, blkP);
#endif
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T>
__aicore__ inline void PackedMultiBlockGemv<T>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
                                                    GM_ADDR y, GM_ADDR tiling)
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
    template <typename T>
    __aicore__ inline void PackedMultiBlockGemv<T>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
                                                        GM_ADDR b2, GM_ADDR y)
#endif
{
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    const uint32_t blk = GetBlockIdx();
    const uint64_t row = static_cast<uint64_t>(blk) * Rows;
#else
    const uint64_t row = static_cast<uint64_t>(GetBlockIdx()) * Rows;
#endif
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
    ::SharedPrimitive9aae833572<T>(b1, b2, gb1, gb2, gw1, gw2, gx, gy, w1, w2, x, y);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    RunTile(gb1, gb2, gw1, gw2, gx, gy, row, tiling);
#elif (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    RunTile(blk, gb1, gb2, gw1, gw2, gx, gy, row, tiling);
#else
    RunTile(gb1, gb2, gw1, gw2, gx, gy, row);
#endif
}
