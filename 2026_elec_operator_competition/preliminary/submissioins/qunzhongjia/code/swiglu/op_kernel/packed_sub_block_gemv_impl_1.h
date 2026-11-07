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

template <typename T>
template <class B0, class B1, class B2, class B3, class B4, class B5>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::SharedStage11510c1ed8(B0& acc, B1& bothBlock, B2& bothHalfW, B3& bothP, B4& gu,
                                             B5& halfExpandedX)
{
    constexpr uint64_t mask = 0;
    SetVectorMask<float, MaskMode::NORMAL>(0, ~0ULL);
    const BinaryRepeatParams rp{1, 1, 1, 8, 0, 4};
#pragma unroll

    for (uint32_t i = 0; i < Rows; ++i) {
        MulAddDst<float, T, false>(bothP[i * 512U], halfExpandedX[i * 64U], bothHalfW, mask, 8, rp);
    }
    BlockReduceSum<float, false>(bothBlock, bothP, 24, mask, 1, 1, 8);
    PairReduceSum<float, false>(acc, bothBlock, 3, mask, 1, 1, 8);
    Cast<T, float, false>(gu, acc, RoundMode::CAST_RINT, mask, 2U, UnaryRepeatParams{1, 1, 4, 8});
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    SetVectorMask<T, MaskMode::NORMAL>(0, 0xFFFFFFFFULL);
}

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::RunStage082ff1a2(A0& hw1, A1& hw2, A2& gu, A3& sig, A4& ones, A5& out, A6& bothHalfW,
                                        A7& bothBias, A8& halfExpandedX, A9& bothP, A10& bothBlock, A11& acc)
{
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    constexpr uint64_t mask = 0;
    Duplicate<T, false>(ones, static_cast<T>(1), mask, 1U, 1, 8);
    Duplicate<float, false>(bothP, 0.0f, mask, 24U, 1, 8);
    SetVectorMask<T, MaskMode::NORMAL>(0, 0xFFFFFFFFFFFFULL);
    Duplicate<T, false>(hw1[208], static_cast<T>(0), mask, 1U, 1, 8);
    Duplicate<T, false>(hw2[208], static_cast<T>(0), mask, 1U, 1, 8);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetVectorMask<T, MaskMode::NORMAL>(0, 0xFFFFULL);
    Adds<T, false>(halfExpandedX[16], halfExpandedX, static_cast<T>(0), mask, 3U,
                   UnaryRepeatParams{1, 1, 4, 4});
    SetVectorMask<T, MaskMode::NORMAL>(0, 0xFFFFFFFFULL);
    Adds<T, false>(halfExpandedX[32], halfExpandedX, static_cast<T>(0), mask, 3U,
                   UnaryRepeatParams{1, 1, 4, 4});
    SharedStage11510c1ed8(acc, bothBlock, bothHalfW, bothP, gu, halfExpandedX);
    Add<T, false>(gu, gu, bothBias, mask, 3U, BinaryRepeatParams{1, 1, 1, 2, 2, 0});
    SetVectorMask<T, MaskMode::NORMAL>(~0ULL, ~0ULL);
    const UnaryRepeatParams ur8{1, 1, 8, 8};
    const BinaryRepeatParams br8{1, 1, 1, 8, 8, 8};
    Muls<T, false>(sig, gu, static_cast<T>(-1), mask, 1U, ur8);
    Exp<T, false>(sig, sig, mask, 1U, ur8);
    Adds<T, false>(sig, sig, static_cast<T>(1), mask, 1U, ur8);
    Div<T, false>(sig, ones, sig, mask, 1U, br8);
    Mul<T, false>(sig, gu, sig, mask, 1U, br8);
    SetVectorMask<T, MaskMode::NORMAL>(0, 0xFFFFFFFFFFFFULL);
    Mul<T, false>(out, sig, gu[16], mask, 1U, BinaryRepeatParams{1, 2, 2, 8, 8, 8});
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::RunStage7b956e3f(A0& b2, A1& y, A2& row, A3& gx, A4& gw1, A5& gw2, A6& gb1, A7& gb2,
                                        A8& gy)
{
    gb2.SetGlobalBuffer((__gm__ T*)b2);
    gy.SetGlobalBuffer((__gm__ T*)y);
    LocalTensor<T> hw1(TPosition::VECIN, 0, W);
    LocalTensor<T> hw2(TPosition::VECIN, 2U * W, W);
    LocalTensor<T> hb1(TPosition::VECIN, HB, K);
    LocalTensor<T> hb2(TPosition::VECIN, HB + 2U * K, K);
    LocalTensor<T> gu(TPosition::VECCALC, GB, 128);
    LocalTensor<T> sig(TPosition::VECCALC, GB + 256U, 128);
    LocalTensor<T> ones(TPosition::VECCALC, GB + 512U, 128);
    LocalTensor<T> out(TPosition::VECOUT, GB + 768U, 128);
    SetAtomicNone();
    SetMaskNorm();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    LocalTensor<T> bothHalfW(TPosition::VECIN, 0, 512);
    LocalTensor<T> bothBias(TPosition::VECIN, HB, 32);
    LocalTensor<T> halfExpandedX(TPosition::VECIN, 10752, 192);
    LocalTensor<float> bothP(TPosition::VECCALC, 19904, 1536);
    LocalTensor<float> bothBlock(TPosition::VECCALC, 34048, 192);
    LocalTensor<float> acc(TPosition::VECCALC, AB, 96);
    DataCopyExtParams wc{D, D * 2U, 0, 0, 0};
    DataCopyExtParams bc{1, D * 2U, 0, 0, 0};
    DataCopyExtParams xcIn{Rows, D * 2U, 0, 3, 0};
    DataCopyExtParams xcOut{Rows, D * 2U, 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, 3, static_cast<T>(0)};
    DataCopyPad(hw1, gw1, wc, pad);
    DataCopyPad(hw2, gw2, wc, pad);
    DataCopyPad(halfExpandedX, gx[row * D], xcIn, pad);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopyPad(hb1, gb1, bc, pad);
    DataCopyPad(hb2, gb2, bc, pad);
    RunStage082ff1a2(hw1, hw2, gu, sig, ones, out, bothHalfW, bothBias, halfExpandedX, bothP, bothBlock, acc);
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopyPad(gy[row * D], out, xcOut);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T>
template <class B0, class B1, class B2, class B3, class B4, class B5, class B6>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::RunActive(B0& b1, B1& b2, B2& row, B3& w1, B4& w2, B5& x, B6& y)
#endif
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    template <typename T>
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7>
    __aicore__ inline
    __attribute__((always_inline)) void PackedSubBlockGemv<T>::RunActive(B0& b1, B1& b2, B2& row, B3& tiling,
                                                                         B4& w1, B5& w2, B6& x, B7& y)
#endif
{
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    gx.SetGlobalBuffer((__gm__ T*)x);
    gw1.SetGlobalBuffer((__gm__ T*)w1);
    gw2.SetGlobalBuffer((__gm__ T*)w2);
    gb1.SetGlobalBuffer((__gm__ T*)b1);
    RunStage7b956e3f(b2, y, row, gx, gw1, gw2, gb1, gb2, gy);
#else
    ::SharedPrimitive9aae833572<T>(b1, b2, gb1, gb2, gw1, gw2, gx, gy, w1, w2, x, y);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    RunTile(gb1, gb2, gw1, gw2, gx, gy, row, tiling);
#else
    RunTile(gb1, gb2, gw1, gw2, gx, gy, row);
#endif
#endif
}

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T>
__aicore__ inline void PackedSubBlockGemv<T>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2,
                                                  GM_ADDR y)
#endif
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    template <typename T>
    __aicore__ inline void PackedSubBlockGemv<T>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
                                                      GM_ADDR b2, GM_ADDR y, GM_ADDR tiling)
#endif
{
    const uint32_t blk = GetBlockIdx();
    if (blk >= 7U) {
        return;
    }
    const uint64_t row = static_cast<uint64_t>(blk) * Rows;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    RunActive(b1, b2, row, tiling, w1, w2, x, y);
#else
    RunActive(b1, b2, row, w1, w2, x, y);
#endif
}

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::RunStagef8bae1b0(A0& hw1, A1& hw2, A2& sig, A3& ones, A4& out, A5& rawW1, A6& rawW2,
                                        A7& rawX, A8& offsets, A9& bothP, A10& bothBlock)
{
    constexpr uint64_t mask = 0;
    LocalTensor<float> acc(TPosition::VECCALC, AB, 96);
    LocalTensor<T> bothBias(TPosition::VECIN, HB, 32);
    LocalTensor<T> gu(TPosition::VECCALC, GB, 96);
    LocalTensor<T> bothHalfW(TPosition::VECIN, 0, 512);
    LocalTensor<T> halfExpandedX(TPosition::VECCALC, 9984, 192);
    Duplicate<float, false>(bothP, 0.0f, mask, 24U, 1, 8);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Gather(hw1, rawW1, offsets, 0, W);
    Gather(hw2, rawW2, offsets, 0, W);
    Gather(halfExpandedX, rawX, offsets[256], 0, 192);
    SharedStage11510c1ed8(acc, bothBlock, bothHalfW, bothP, gu, halfExpandedX);
    Add<T, false>(gu, gu, bothBias, mask, 3, BinaryRepeatParams{1, 1, 1, 2, 2, 0});
    SetVectorMask<T, MaskMode::NORMAL>(0, 0xFFFFULL);
    const UnaryRepeatParams compactG{1, 1, 1, 2};
    const UnaryRepeatParams contiguous{1, 1, 1, 1};
    const BinaryRepeatParams onesBroadcast{1, 1, 1, 1, 0, 1};
    const BinaryRepeatParams gateProduct{1, 1, 1, 1, 2, 1};
    const BinaryRepeatParams upProduct{1, 1, 1, 1, 1, 2};
    Muls<T, false>(sig, gu, static_cast<T>(-1), mask, 3, compactG);
    Exp<T, false>(sig, sig, mask, 3, contiguous);
    Adds<T, false>(sig, sig, static_cast<T>(1), mask, 3, contiguous);
    Div<T, false>(sig, ones, sig, mask, 3, onesBroadcast);
    Mul<T, false>(out, gu, sig, mask, 3, gateProduct);
    Mul<T, false>(out, out, gu[16], mask, 3, upProduct);
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::RunSharedStage1(A0& tiling, A1& row, A2& gx, A3& gw1, A4& gw2, A5& gb1, A6& gb2,
                                       A7& gy, A8& hw1, A9& hw2, A10& hb1, A11& hb2, A12& sig, A13& ones)
{
    LocalTensor<T> out(TPosition::VECOUT, GB + 8U * Rows * K, Rows * K);
    SetAtomicNone();
    SetMaskNorm();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    LocalTensor<T> rawW1(TPosition::VECIN, 8448, 176);
    LocalTensor<T> rawW2(TPosition::VECIN, 8960, 176);
    LocalTensor<T> rawX(TPosition::VECIN, 9472, 48);
    LocalTensor<uint32_t> offsets(TPosition::VECIN, 13056, 496);
    GlobalTensor<uint32_t> indexGm;
    indexGm.SetGlobalBuffer((__gm__ uint32_t*)(tiling + offsetof(SwiGluTilingData, gatherOffsetsA)), 496);
    DataCopyExtParams wc{1, 338, 0, 0, 0};
    DataCopyExtParams bc{1, 26, 0, 0, 0};
    DataCopyExtParams xcFlat{1, 78, 0, 0, 0};
    DataCopyPadExtParams<T> wp{true, 0, 7, static_cast<T>(0)};
    DataCopyPadExtParams<T> xp{true, 0, 9, static_cast<T>(0)};
    DataCopyPadExtParams<T> bp0{true, 0, 3, static_cast<T>(0)};
    DataCopyPad(rawW1, gw1, wc, wp);
    DataCopyPad(rawW2, gw2, wc, wp);
    DataCopyPad(rawX, gx[row * D], xcFlat, xp);
    DataCopy(offsets, indexGm, 496);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    DataCopyPad(hb1, gb1, bc, bp0);
    DataCopyPad(hb2, gb2, bc, bp0);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    constexpr uint64_t mask = 0;
    Duplicate<T, false>(ones, static_cast<T>(1), mask, 1U, 1, 8);
    LocalTensor<T> packedOut(TPosition::VECOUT, 19392, 48);
    LocalTensor<float> bothP(TPosition::VECCALC, 19840, 1536);
    LocalTensor<float> bothBlock(TPosition::VECCALC, 34176, 192);
    RunStagef8bae1b0(hw1, hw2, sig, ones, out, rawW1, rawW2, rawX, offsets, bothP, bothBlock);
    Gather(packedOut, out, offsets[448], 0, Rows * D);
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopyPad(gy[row * D], packedOut, xcFlat);
}
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
#if !((defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL)))
template <typename T>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::RunSharedStage1(A0& row, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& gy, A7& hw1,
                                       A8& hw2, A9& hb1, A10& hb2, A11& sig)
#endif
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <typename T>
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13>
    __aicore__ inline
    __attribute__((always_inline)) void PackedSubBlockGemv<T>::RunSharedStage1(A0& row, A1& gx, A2& gw1,
                                                                               A3& gw2, A4& gb1, A5& gb2,
                                                                               A6& gy, A7& hw1, A8& hw2,
                                                                               A9& hb1, A10& hb2, A11& sig,
                                                                               A12& out, A13& packedOut)
#endif
{
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
#else
    LocalTensor<T> out(TPosition::VECOUT, GB + 8U * Rows * K, Rows * K);
    LocalTensor<T> packedOut(TPosition::VECOUT, GB + 10U * Rows * K, Rows * K);
#endif
    SetAtomicNone();
    SetMaskNorm();
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    LocalTensor<T> rawW1(TPosition::VECIN, 8448, 176);
    LocalTensor<T> rawW2(TPosition::VECIN, 8960, 176);
    LocalTensor<T> rawX(TPosition::VECIN, 9472, 48);
    LocalTensor<uint32_t> offsets(TPosition::VECIN, 13056, 512);
    LocalTensor<int32_t> offsetsI(TPosition::VECIN, 13056, 512);
    DataCopyExtParams wc{1, 338, 0, 0, 0};
    DataCopyExtParams bc{1, 26, 0, 0, 0};
    DataCopyExtParams xcFlat{1, 78, 0, 0, 0};
    SharedStagee9c87e4530(bc, gb1, gb2, gw1, gw2, gx, hb1, hb2, rawW1, rawW2, rawX, row, wc, xcFlat);
    LocalTensor<float> bothP(TPosition::VECCALC, 19904, 1536);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    LocalTensor<float> bothBlock(TPosition::VECCALC, 34048, 192);
    LocalTensor<float> acc(TPosition::VECCALC, AB, 96);
    LocalTensor<T> bothBias(TPosition::VECIN, HB, 32);
    RunSharedStage0(hw1, hw2, sig, out, rawW1, rawW2, rawX, offsets, offsetsI, bothP, bothBlock, acc,
                    bothBias);
    SetVectorMask<T, MaskMode::NORMAL>(0, (1ULL << (Rows * D)) - 1ULL);
    Gather(packedOut, out, offsets[448], 0, Rows * D);
#else
    RunSharedStage0(hw1, hw2, sig, out, rawW1, rawW2, rawX, offsets, offsetsI, bothP);
    // Pack the13 live half elements of each16-element output row.
    // Counter-mode repeat gathering removes the output index table entirely.
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
    vreducev2((__ubuf__ uint16_t*)packedOut.GetPhyAddr(), (__ubuf__ uint16_t*)out.GetPhyAddr(),
              (__ubuf__ uint16_t*)offsets[448].GetPhyAddr(), Rows, 1, 0, 1, 0);
#endif
    ::SharedSynccc9d9b6d<HardEvent::V_MTE3>();
    DataCopyPad(gy[row * D], packedOut, xcFlat);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL))
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T>
template <class B0, class B1, class B2, class B3, class B4, class B5, class B6>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::RunTile(B0& gb1, B1& gb2, B2& gw1, B3& gw2, B4& gx, B5& gy, B6& row)
#endif
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    template <typename T>
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7>
    __aicore__ inline
    __attribute__((always_inline)) void PackedSubBlockGemv<T>::RunTile(B0& gb1, B1& gb2, B2& gw1, B3& gw2,
                                                                       B4& gx, B5& gy, B6& row, B7& tiling)
#endif
{
    LocalTensor<T> hw1(TPosition::VECIN, 0, W);
    LocalTensor<T> hw2(TPosition::VECIN, 2U * W, W);
    LocalTensor<T> hb1(TPosition::VECIN, HB, K);
    LocalTensor<T> hb2(TPosition::VECIN, HB + 2U * K, K);
    LocalTensor<T> sig(TPosition::VECCALC, GB + 4U * Rows * K, Rows * K);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    LocalTensor<T> ones(TPosition::VECCALC, GB + 6U * Rows * K, Rows * K);
    RunSharedStage1(tiling, row, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hb1, hb2, sig, ones);
#elif (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    LocalTensor<T> out(TPosition::VECOUT, GB + 8U * Rows * K, Rows * K);
    LocalTensor<T> packedOut(TPosition::VECOUT, GB + 10U * Rows * K, Rows * K);
    RunSharedStage1(row, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hb1, hb2, sig, out, packedOut);
#else
    RunSharedStage1(row, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hb1, hb2, sig);
#endif
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T>
template <class B0, class B1>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::SharedStage1a9cd481b9(B0& bothP, B1& offsetsI)
{
    constexpr uint64_t mask = 0;
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate<float, false>(bothP, 0.0f, mask, 24U, 1, 8);
    Duplicate<int32_t, false>(offsetsI[16], 338, mask, 3U, 1, 8);
    Duplicate<int32_t, false>(offsetsI[256], 78, mask, 3U, 1, 8);
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0xFFFFFFFFFFFFULL);
    Duplicate<int32_t, false>(offsetsI[208], 338, mask, 1U, 1, 8);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    const UnaryRepeatParams urI{1, 1, 8, 8};
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0x1FFFULL);
    Adds<int32_t, false>(offsetsI[16], offsetsI[0], 26, mask, 1U, urI);
    Adds<int32_t, false>(offsetsI[192], offsetsI[0], 312, mask, 1U, urI);
    Adds<int32_t, false>(offsetsI[256], offsetsI[0], 0, mask, 1U, urI);
    Adds<int32_t, false>(offsetsI[272], offsetsI[0], 0, mask, 1U, urI);
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0x1FFF1FFFULL);
    Adds<int32_t, false>(offsetsI[32], offsetsI[0], 52, mask, 1U, urI);
    Adds<int32_t, false>(offsetsI[288], offsetsI[256], 0, mask, 1U, urI);
    SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0x1FFF1FFF1FFF1FFFULL);
    Adds<int32_t, false>(offsetsI[64], offsetsI[0], 104, mask, 1U, urI);
    Adds<int32_t, false>(offsetsI[128], offsetsI[0], 208, mask, 1U, urI);
    Adds<int32_t, false>(offsetsI[320], offsetsI[256], 26, mask, 1U, urI);
    Adds<int32_t, false>(offsetsI[384], offsetsI[256], 52, mask, 1U, urI);
}

#endif

#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
template <typename T>
template <class A0, class A1, class A2>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::RunStage058d2452(A0& offsetsI, A1& bothP, A2& offsets64)
{
    constexpr uint64_t mask = 0;
#pragma unroll

    for (uint32_t k = 0; k < 6U; ++k) {
        const uint64_t c4 = k * 4U;
        const uint64_t valA = ((c4 + 2U) << 32) | c4;
        offsets64.SetValue(k, valA);
        offsets64.SetValue(224U + k, valA);
        offsets64.SetValue(231U + k, valA + ((34ULL << 32) | 34ULL));
        offsets64.SetValue(237U + k, valA + ((64ULL << 32) | 64ULL));
    }
    offsets64.SetValue(6U, (338ULL << 32) | 24ULL);
    offsets64.SetValue(7U, (338ULL << 32) | 338ULL);
    offsets64.SetValue(230U, (32ULL << 32) | 24ULL);
    offsets64.SetValue(243U, 88ULL);
    SharedStage1a9cd481b9(bothP, offsetsI);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T>
template <class B0, class B1, class B2, class B3>
__aicore__ inline __attribute__((always_inline)) void
PackedSubBlockGemv<T>::SharedStaged379073a67(B0& bothBias, B1& gu, B2& out, B3& sig)
{
    constexpr uint64_t mask = 0;
    Add<T, false>(gu, gu, bothBias, mask, 3, BinaryRepeatParams{1, 1, 1, 2, 2, 0});
    SetVectorMask<T, MaskMode::NORMAL>(0, 0xFFFFULL);
    const UnaryRepeatParams compactG{1, 1, 1, 2};
    const UnaryRepeatParams contiguous{1, 1, 1, 1};
    const BinaryRepeatParams gateProduct{1, 1, 1, 1, 2, 1};
    const BinaryRepeatParams upProduct{1, 1, 1, 1, 1, 2};
    Muls<T, false>(sig, gu, static_cast<T>(-1), mask, 3, compactG);
    Exp<T, false>(sig, sig, mask, 3, contiguous);
    Adds<T, false>(sig, sig, static_cast<T>(1), mask, 3, contiguous);
    Div<T, false>(sig, gu, sig, mask, 3, gateProduct);
    Mul<T, false>(out, sig, gu[16], mask, 3, upProduct);
}

#endif
