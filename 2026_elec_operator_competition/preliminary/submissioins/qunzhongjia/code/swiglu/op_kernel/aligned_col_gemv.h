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

template <typename T, uint32_t FixedD, uint32_t FixedM> class AlignedColGemv {
public:
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
    __aicore__ inline __attribute__((always_inline)) void
    RunStageff62cac3(A0& d, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& hw1, A7& hw2, A8& hx, A9& hb1,
                     A10& hb2, A11& pg, A12& pu, A13& ag, A14& ones, A15& row, A16& col);
#else
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
    __aicore__ inline __attribute__((always_inline)) void
    RunStage5972703b(A0& d, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& hw1, A7& hw2, A8& hx, A9& hb1,
                     A10& hb2, A11& pg, A12& pu, A13& ag, A14& g, A15& row, A16& col);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20>
    __aicore__ inline __attribute__((always_inline)) void
    RunStage25bf2efe(A0& d, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& gy, A7& hw1, A8& hw2, A9& hx,
                     A10& hb1, A11& hb2, A12& pg, A13& pu, A14& ag, A15& g, A16& sig, A17& ones, A18& out,
                     A19& row, A20& col);
#endif
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8,
              class B9, class B10, class B11, class B12, class B13, class B14, class B15, class B16,
              class B17, class B18, class B19, class B20>
    __aicore__ inline __attribute__((always_inline)) void
    ComputeRow(B0& ag, B1& columns, B2& d, B3& g, B4& gb1, B5& gb2, B6& gw1, B7& gw2, B8& gx, B9& gy,
               B10& hb1, B11& hb2, B12& hw1, B13& hw2, B14& hx, B15& ones, B16& out, B17& pg, B18& pu,
               B19& sig, B20& task);
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage0(A0& x, A1& w1, A2& b1, A3& w2, A4& b2, A5& y, A6& task, A7& d, A8& columns, A9& gx,
                    A10& gw1, A11& gw2, A12& gb1, A13& gb2, A14& gy);
#else
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage0(A0& task, A1& d, A2& columns, A3& gx, A4& gw1, A5& gw2, A6& gb1, A7& gb2, A8& gy, A9& hw1,
                    A10& hw2, A11& hx);
#else
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage0(A0& task, A1& d, A2& columns, A3& gx, A4& gw1, A5& gw2, A6& gb1, A7& gb2, A8& gy, A9& hw1,
                    A10& hw2, A11& hx, A12& hb1, A13& hb2, A14& pg, A15& pu);
#endif
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8>
    __aicore__ inline __attribute__((always_inline)) void RunTile(B0& columns, B1& d, B2& gb1, B3& gb2,
                                                                  B4& gw1, B5& gw2, B6& gx, B7& gy, B8& task);
#endif
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
private:
    template <bool First>
    __aicore__ inline void Dot(const LocalTensor<float>& ag, const LocalTensor<float>& au,
                               const LocalTensor<float>& sg, const LocalTensor<float>& su,
                               const LocalTensor<float>& pg, const LocalTensor<float>& pu,
                               const LocalTensor<float>& fx, const LocalTensor<float>& fw1,
                               const LocalTensor<float>& fw2, uint32_t ka, uint32_t k, uint32_t cols);
#else
#endif
};

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t FixedD, uint32_t FixedM>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16>
__aicore__ inline __attribute__((always_inline)) void
AlignedColGemv<T, FixedD, FixedM>::RunStageff62cac3(A0& d, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2,
                                                    A6& hw1, A7& hw2, A8& hx, A9& hb1, A10& hb2, A11& pg,
                                                    A12& pu, A13& ag, A14& ones, A15& row, A16& col)
{
    constexpr uint64_t mask = 0;
    if constexpr (FixedD == 64U) {
        DataCopy(hw1, gw1[static_cast<uint64_t>(col) * d], 16U * FixedD);
        DataCopy(hw2, gw2[static_cast<uint64_t>(col) * d], 16U * FixedD);
        DataCopy(hx, gx[row * d], FixedD);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        DataCopy(hb1, gb1[col], 16U);
        DataCopy(hb2, gb2[col], 16U);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Duplicate<T, false>(ones, static_cast<T>(1), mask, 1, 1, 8);
        Duplicate<float, false>(pg, 0.0f, mask, 32U, 1, 8);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        MulAddDst<float, T, false>(pg, hx, hw1, mask, 32U, BinaryRepeatParams{1, 1, 1, 8, 0, 4});
        BlockReduceSum<float, false>(pu, pg, 32U, mask, 1, 1, 8);
        BlockReduceSum<float, false>(ag, pu, 4U, mask, 1, 1, 8);
    } else {
        DataCopy(hw1, gw1[static_cast<uint64_t>(col) * d], 16U * FixedD);
        DataCopy(hw2, gw2[static_cast<uint64_t>(col) * d], 16U * FixedD);
        DataCopy(hx, gx[row * d], FixedD);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        DataCopy(hb1, gb1[col], 16U);
        DataCopy(hb2, gb2[col], 16U);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Duplicate<T, false>(ones, static_cast<T>(1), mask, 1, 1, 8);
        Duplicate<float, false>(pg, 0.0f, mask, 16U, 1, 8);
        SetVectorMask<float, MaskMode::NORMAL>(0, 0xFFFFFFFFULL);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        MulAddDst<float, T, false>(pg, hx, hw1, mask, 32U, BinaryRepeatParams{1, 1, 1, 4, 0, 2});
        SetVectorMask<float, MaskMode::NORMAL>(0, ~0ULL);
        BlockReduceSum<float, false>(pu, pg, 16U, mask, 1, 1, 8);
        PairReduceSum<float, false>(pu[128], pu, 2U, mask, 1, 1, 8);
        PairReduceSum<float, false>(ag, pu[128], 1U, mask, 1, 1, 8);
    }
    SetVectorMask<float, MaskMode::NORMAL>(0, 0xFFFFFFFFULL);
}

#endif

template <typename T, uint32_t FixedD, uint32_t FixedM>
template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8, class B9,
          class B10, class B11, class B12, class B13, class B14, class B15, class B16, class B17, class B18,
          class B19, class B20>
__aicore__ inline __attribute__((always_inline)) void AlignedColGemv<T, FixedD, FixedM>::ComputeRow(
    B0& ag, B1& columns, B2& d, B3& g, B4& gb1, B5& gb2, B6& gw1, B7& gw2, B8& gx, B9& gy, B10& hb1, B11& hb2,
    B12& hw1, B13& hw2, B14& hx, B15& ones, B16& out, B17& pg, B18& pu, B19& sig, B20& task)
{
    SetAtomicNone();
    SetMaskNorm();
    constexpr uint64_t mask = 0;
    SetVectorMask<uint64_t, MaskMode::NORMAL>(~0ULL, ~0ULL);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    {
        const uint64_t row = FixedM == 1U ? 0U : task / columns;
        const uint32_t col = (FixedM == 1U ? task : task % columns) * 16U;
        static_assert(FixedD == 32U || FixedD == 64U);
        RunStageff62cac3(d, gx, gw1, gw2, gb1, gb2, hw1, hw2, hx, hb1, hb2, pg, pu, ag, ones, row, col);
        Cast<T, float, false>(g, ag, RoundMode::CAST_RINT, mask, 1, UnaryRepeatParams{1, 1, 4, 8});
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Add<T, false>(g, g, hb1, mask, 1, BinaryRepeatParams{1, 1, 1, 2, 2, 2});
        FastGate<T, 16U>(g, g[16], sig, out, ones);
        ::SharedPrimitive8310a39caa<FixedM>(col, d, gy, out, row);
    }
#elif (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    {
        const uint64_t row = FixedM == 1U ? 0U : task / columns;
        const uint32_t col = (FixedM == 1U ? task : task % columns) * 16U;
        static_assert(FixedD == 32U || FixedD == 64U);
        RunStageff62cac3(d, gx, gw1, gw2, gb1, gb2, hw1, hw2, hx, hb1, hb2, pg, pu, ag, ones, row, col);
        Cast<T, float, false>(g, ag, RoundMode::CAST_RINT, mask, 1, UnaryRepeatParams{1, 1, 4, 8});
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        SetVectorMask<T, MaskMode::NORMAL>(0, 0xFFFFFFFFULL);
        Add<T, false>(g, g, hb1, mask, 1, BinaryRepeatParams{1, 1, 1, 2, 2, 2});
        FastGate<T, 16U>(g, g[16], sig, out, ones);
        ::SharedPrimitive8310a39caa<FixedM>(col, d, gy, out, row);
    }
#else
    {
        const uint64_t row =
            FixedM == 1U ? 0U : (FixedD == 64U && FixedM == 2U ? (task >> 2U) : task / columns);
        const uint32_t col =
            (FixedM == 1U ? task : (FixedD == 64U && FixedM == 2U ? (task & 3U) : task % columns)) * 16U;
        static_assert(FixedD == 32U || FixedD == 64U);
        RunStage25bf2efe(d, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hx, hb1, hb2, pg, pu, ag, g, sig, ones, out,
                         row, col);
    }
#endif
}

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t FixedD, uint32_t FixedM>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15>
__aicore__ inline __attribute__((always_inline)) void
AlignedColGemv<T, FixedD, FixedM>::RunSharedStage0(A0& task, A1& d, A2& columns, A3& gx, A4& gw1, A5& gw2,
                                                   A6& gb1, A7& gb2, A8& gy, A9& hw1, A10& hw2, A11& hx,
                                                   A12& hb1, A13& hb2, A14& pg, A15& pu)
{
    constexpr uint32_t maxK = FixedD;
    constexpr uint32_t wElems = 16U * maxK;
    constexpr uint32_t fBase = 4U * wElems + 2U * maxK + 64U;
    constexpr uint32_t xBase = fBase + 16U * wElems;
    LocalTensor<float> fx(TPosition::VECCALC, xBase, maxK);
    constexpr uint32_t aBase = xBase + 4U * maxK;
    LocalTensor<float> ag(TPosition::VECCALC, aBase, 32);
    LocalTensor<float> au(TPosition::VECCALC, aBase + 64U, 16);
    LocalTensor<float> sg(TPosition::VECCALC, aBase + 128U, 16);
    LocalTensor<float> su(TPosition::VECCALC, aBase + 192U, 16);
    constexpr uint32_t hBase = aBase + 256U;
    LocalTensor<T> g(TPosition::VECCALC, hBase, 32);
    LocalTensor<T> u(TPosition::VECCALC, hBase + 32U, 16);
    LocalTensor<T> sig(TPosition::VECCALC, hBase + 64U, 16);
    LocalTensor<T> ones(TPosition::VECCALC, hBase + 96U, 16);
    LocalTensor<T> out(TPosition::VECOUT, hBase + 128U, 16);
    ComputeRow(ag, columns, d, g, gb1, gb2, gw1, gw2, gx, gy, hb1, hb2, hw1, hw2, hx, ones, out, pg, pu, sig,
               task);
}
#else
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
template <typename T, uint32_t FixedD, uint32_t FixedM>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11>
__aicore__ inline __attribute__((always_inline)) void
AlignedColGemv<T, FixedD, FixedM>::RunSharedStage0(A0& task, A1& d, A2& columns, A3& gx, A4& gw1, A5& gw2,
                                                   A6& gb1, A7& gb2, A8& gy, A9& hw1, A10& hw2, A11& hx)
#endif
#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL))
    template <typename T, uint32_t FixedD, uint32_t FixedM>
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14>
    __aicore__ inline __attribute__((always_inline)) void AlignedColGemv<T, FixedD, FixedM>::RunSharedStage0(
        A0& x, A1& w1, A2& b1, A3& w2, A4& b2, A5& y, A6& task, A7& d, A8& columns, A9& gx, A10& gw1,
        A11& gw2, A12& gb1, A13& gb2, A14& gy)
#endif
{
    constexpr uint32_t maxK = FixedD;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
#else
    constexpr uint32_t xCap = 64U;
#endif
    constexpr uint32_t wElems = 16U * maxK;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    constexpr uint32_t fBase = 4U * wElems + 2U * maxK + 128U;
    LocalTensor<T> hb1(TPosition::VECIN, 4U * wElems + 2U * maxK, 32);
    LocalTensor<T> hb2(TPosition::VECIN, 4U * wElems + 32U + 2U * maxK, 16);
#else
    constexpr uint32_t fBase = 4U * wElems + 2U * xCap + 128U;
    ::SharedPrimitive3eb9fd04b0<T, FixedM>(b1, b2, d, gb1, gb2, gw1, gw2, gx, gy, w1, w2, x, y);
    LocalTensor<T> hw1(TPosition::VECIN, 0, 2U * wElems);
    LocalTensor<T> hw2(TPosition::VECIN, 2U * wElems, wElems);
    LocalTensor<T> hx(TPosition::VECIN, 4U * wElems, xCap);
    LocalTensor<T> hb1(TPosition::VECIN, 4U * wElems + 2U * xCap, 32);
    LocalTensor<T> hb2(TPosition::VECIN, 4U * wElems + 2U * xCap + 32U, 16);
#endif
    LocalTensor<float> pg(TPosition::VECCALC, fBase + 8U * wElems, 2U * wElems);
    LocalTensor<float> pu(TPosition::VECCALC, fBase + 16U * wElems, wElems);
    constexpr uint32_t xBase = fBase + 20U * wElems;
    constexpr uint32_t aBase = xBase + 4U * maxK;
    LocalTensor<float> ag(TPosition::VECCALC, aBase, 32);
    constexpr uint32_t hBase = aBase + 256U;
    LocalTensor<T> g(TPosition::VECCALC, hBase, 32);
    LocalTensor<T> u(TPosition::VECCALC, hBase + 32U, 16);
    LocalTensor<T> sig(TPosition::VECCALC, hBase + 64U, 16);
    LocalTensor<T> ones(TPosition::VECCALC, hBase + 96U, 16);
    LocalTensor<T> out(TPosition::VECOUT, hBase + 128U, 16);
    ComputeRow(ag, columns, d, g, gb1, gb2, gw1, gw2, gx, gy, hb1, hb2, hw1, hw2, hx, ones, out, pg, pu, sig,
               task);
}

#endif

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t FixedD, uint32_t FixedM>
template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8>
__aicore__ inline __attribute__((always_inline)) void
AlignedColGemv<T, FixedD, FixedM>::RunTile(B0& columns, B1& d, B2& gb1, B3& gb2, B4& gw1, B5& gw2, B6& gx,
                                           B7& gy, B8& task)
{
    constexpr uint32_t maxK = FixedD;
    constexpr uint32_t wElems = 16U * maxK;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
#else
    constexpr uint32_t fBase = 4U * wElems + 2U * maxK + 64U;
#endif
    LocalTensor<T> hw1(TPosition::VECIN, 0, 2U * wElems);
    LocalTensor<T> hw2(TPosition::VECIN, 2U * wElems, wElems);
    LocalTensor<T> hx(TPosition::VECIN, 4U * wElems, maxK);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    RunSharedStage0(task, d, columns, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hx);
#else
    LocalTensor<T> hb1(TPosition::VECIN, 4U * wElems + 2U * maxK, 32);
    LocalTensor<T> hb2(TPosition::VECIN, 4U * wElems + 2U * maxK + 32U, 16);
    LocalTensor<float> fw1(TPosition::VECCALC, fBase, 2U * wElems);
    LocalTensor<float> fw2(TPosition::VECCALC, fBase + 4U * wElems, wElems);
    LocalTensor<float> pg(TPosition::VECCALC, fBase + 8U * wElems, 2U * wElems);
    LocalTensor<float> pu(TPosition::VECCALC, fBase + 12U * wElems, wElems);
    constexpr uint32_t xBase = fBase + 16U * wElems;
    RunSharedStage0(task, d, columns, gx, gw1, gw2, gb1, gb2, gy, hw1, hw2, hx, hb1, hb2, pg, pu);
#endif
}

#endif

template <typename T, uint32_t FixedD, uint32_t FixedM>
__aicore__ inline void AlignedColGemv<T, FixedD, FixedM>::Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2,
                                                              GM_ADDR b2, GM_ADDR y)
{
    const uint64_t task = GetBlockIdx();
    if constexpr (FixedM == 1U) {
        if (task >= 2U) {
            return;
        }
    }
    constexpr uint32_t maxK = FixedD;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
#else
    constexpr uint32_t xCap = 64U;
#endif
    constexpr uint32_t wElems = 16U * maxK;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    constexpr uint32_t fBase = 4U * wElems + 2U * maxK + 128U;
#elif (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    constexpr uint32_t fBase = 4U * wElems + 2U * maxK + 64U;
#else
    constexpr uint32_t fBase = 4U * wElems + 2U * xCap + 128U;
#endif
    const uint32_t d = FixedD;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    const uint32_t ka = (d + 15U) / 16U * 16U;
#else
#endif
    const uint32_t columns = (d + 15U) / 16U;
    GlobalTensor<T> gx;
    GlobalTensor<T> gw1;
    GlobalTensor<T> gw2;
    GlobalTensor<T> gb1;
    GlobalTensor<T> gb2;
    GlobalTensor<T> gy;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    ::SharedPrimitive3eb9fd04b0<T, FixedM>(b1, b2, d, gb1, gb2, gw1, gw2, gx, gy, w1, w2, x, y);
    RunTile(columns, d, gb1, gb2, gw1, gw2, gx, gy, task);
#else
    RunSharedStage0(x, w1, b1, w2, b2, y, task, d, columns, gx, gw1, gw2, gb1, gb2, gy);
#endif
}

#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
template <typename T, uint32_t FixedD, uint32_t FixedM>
template <bool First>
__aicore__ inline void
AlignedColGemv<T, FixedD, FixedM>::Dot(const LocalTensor<float>& ag, const LocalTensor<float>& au,
                                       const LocalTensor<float>& sg, const LocalTensor<float>& su,
                                       const LocalTensor<float>& pg, const LocalTensor<float>& pu,
                                       const LocalTensor<float>& fx, const LocalTensor<float>& fw1,
                                       const LocalTensor<float>& fw2, uint32_t ka, uint32_t k, uint32_t cols)
{
    SharedKernel0c144b8d11<First>(ag, au, sg, su, pg, pu, fx, fw1, fw2, ka, k, cols);
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t FixedD, uint32_t FixedM>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16>
__aicore__ inline __attribute__((always_inline)) void
AlignedColGemv<T, FixedD, FixedM>::RunStage5972703b(A0& d, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2,
                                                    A6& hw1, A7& hw2, A8& hx, A9& hb1, A10& hb2, A11& pg,
                                                    A12& pu, A13& ag, A14& g, A15& row, A16& col)
{
    constexpr uint64_t mask = 0;
    DataCopy(hx, gx[row * d], FixedD);
    DataCopy(hx[32], gx[row * d], FixedD);
    ::SharedPrimitivee882d043f2<FixedD>(col, d, gb1, gb2, gw1, gw2, hb1, hb2, hw1, hw2);
    Duplicate<float, false>(pg, 0.0f, mask, 16U, 1, 8);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    MulAddDst<float, T, false>(pg, hx, hw1, mask, 8U, BinaryRepeatParams{1, 1, 1, 8, 0, 4});
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    MulAddDst<float, T, false>(pg[512], hx, hw2, mask, 8U, BinaryRepeatParams{1, 1, 1, 8, 0, 4});
    BlockReduceSum<float, false>(pu, pg, 16U, mask, 1, 1, 8);
    PairReduceSum<float, false>(pg, pu, 2U, mask, 1, 1, 8);
    PairReduceSum<float, false>(ag, pg, 1U, mask, 1, 1, 8);
    SetVectorMask<float, MaskMode::NORMAL>(0, 0xFFFFFFFFULL);
    Cast<T, float, false>(g, ag, RoundMode::CAST_RINT, mask, 1, UnaryRepeatParams{1, 1, 4, 8});
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Add<T, false>(g, g, hb1, mask, 1, BinaryRepeatParams{1, 1, 1, 2, 2, 2});
}

#endif

#if !((defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
      (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL))
template <typename T, uint32_t FixedD, uint32_t FixedM>
template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9,
          class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18,
          class A19, class A20>
__aicore__ inline __attribute__((always_inline)) void AlignedColGemv<T, FixedD, FixedM>::RunStage25bf2efe(
    A0& d, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& gy, A7& hw1, A8& hw2, A9& hx, A10& hb1, A11& hb2,
    A12& pg, A13& pu, A14& ag, A15& g, A16& sig, A17& ones, A18& out, A19& row, A20& col)
{
    constexpr uint64_t mask = 0;
    if constexpr (FixedD == 64U) {
        DataCopy(hx, gx[row * d], FixedD);
        ::SharedPrimitivee882d043f2<FixedD>(col, d, gb1, gb2, gw1, gw2, hb1, hb2, hw1, hw2);
        Duplicate<float, false>(pg, 0.0f, mask, 32U, 1, 8);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        MulAddDst<float, T, false>(pg, hx, hw1, mask, 16U, BinaryRepeatParams{1, 1, 1, 8, 0, 4});
        BlockReduceSum<float, false>(pu, pg, 16U, mask, 1, 1, 8);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        MulAddDst<float, T, false>(pg[1024], hx, hw2, mask, 16U, BinaryRepeatParams{1, 1, 1, 8, 0, 4});
        BlockReduceSum<float, false>(pu[128], pg[1024], 16U, mask, 1, 1, 8);
        BlockReduceSum<float, false>(ag, pu, 4U, mask, 1, 1, 8);
        SetVectorMask<float, MaskMode::NORMAL>(0, 0xFFFFFFFFULL);
        Cast<T, float, false>(g, ag, RoundMode::CAST_RINT, mask, 1, UnaryRepeatParams{1, 1, 4, 8});
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Add<T, false>(g, g, hb1, mask, 1, BinaryRepeatParams{1, 1, 1, 2, 2, 2});
    } else {
        RunStage5972703b(d, gx, gw1, gw2, gb1, gb2, hw1, hw2, hx, hb1, hb2, pg, pu, ag, g, row, col);
    }
    FastGate<T, 16U, false>(g, g[16], sig, out, ones);
    ::SharedPrimitive8310a39caa<FixedM>(col, d, gy, out, row);
}

#endif
