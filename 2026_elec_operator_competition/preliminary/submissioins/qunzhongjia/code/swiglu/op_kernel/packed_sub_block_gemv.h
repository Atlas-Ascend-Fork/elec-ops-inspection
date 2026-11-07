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

template <typename T> class PackedSubBlockGemv {
public:
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL) || \
    (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
#else
    template <class B0, class B1>
    __aicore__ inline __attribute__((always_inline)) void SharedStage1a9cd481b9(B0& bothP, B1& offsetsI);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <class A0, class A1, class A2>
    __aicore__ inline __attribute__((always_inline)) void RunStage058d2452(A0& offsetsI, A1& bothP,
                                                                           A2& offsets64);
#else
    template <class A0, class A1, class A2>
    __aicore__ inline __attribute__((always_inline)) void RunStaged70b3486(A0& offsets, A1& offsetsI,
                                                                           A2& bothP);
#endif
#endif
    template <class B0, class B1, class B2, class B3, class B4, class B5>
    __aicore__ inline __attribute__((always_inline)) void
    SharedStage11510c1ed8(B0& acc, B1& bothBlock, B2& bothHalfW, B3& bothP, B4& gu, B5& halfExpandedX);
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10>
    __aicore__ inline __attribute__((always_inline)) void
    RunStagef8bae1b0(A0& hw1, A1& hw2, A2& sig, A3& ones, A4& out, A5& rawW1, A6& rawW2, A7& rawX,
                     A8& offsets, A9& bothP, A10& bothBlock);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage1(A0& tiling, A1& row, A2& gx, A3& gw1, A4& gw2, A5& gb1, A6& gb2, A7& gy, A8& hw1, A9& hw2,
                    A10& hb1, A11& hb2, A12& sig, A13& ones);
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7>
    __aicore__ inline __attribute__((always_inline)) void RunTile(B0& gb1, B1& gb2, B2& gw1, B3& gw2, B4& gx,
                                                                  B5& gy, B6& row, B7& tiling);
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7>
    __aicore__ inline __attribute__((always_inline)) void RunActive(B0& b1, B1& b2, B2& row, B3& tiling,
                                                                    B4& w1, B5& w2, B6& x, B7& y);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               GM_ADDR tiling);
#else
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11>
    __aicore__ inline __attribute__((always_inline)) void
    RunStage082ff1a2(A0& hw1, A1& hw2, A2& gu, A3& sig, A4& ones, A5& out, A6& bothHalfW, A7& bothBias,
                     A8& halfExpandedX, A9& bothP, A10& bothBlock, A11& acc);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
    __aicore__ inline __attribute__((always_inline)) void
    RunStage7b956e3f(A0& b2, A1& y, A2& row, A3& gx, A4& gw1, A5& gw2, A6& gb1, A7& gb2, A8& gy);
#else
    template <class B0, class B1, class B2, class B3>
    __aicore__ inline __attribute__((always_inline)) void SharedStaged379073a67(B0& bothBias, B1& gu, B2& out,
                                                                                B3& sig);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage0(A0& hw1, A1& hw2, A2& sig, A3& out, A4& rawW1, A5& rawW2, A6& rawX, A7& offsets,
                    A8& offsetsI, A9& bothP, A10& bothBlock, A11& acc, A12& bothBias);
#else
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage0(A0& hw1, A1& hw2, A2& sig, A3& out, A4& rawW1, A5& rawW2, A6& rawX, A7& offsets,
                    A8& offsetsI, A9& bothP);
#endif
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8,
              class B9, class B10, class B11, class B12, class B13>
    __aicore__ inline __attribute__((always_inline)) void
    SharedStagee9c87e4530(B0& bc, B1& gb1, B2& gb2, B3& gw1, B4& gw2, B5& gx, B6& hb1, B7& hb2, B8& rawW1,
                          B9& rawW2, B10& rawX, B11& row, B12& wc, B13& xcFlat);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage1(A0& row, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& gy, A7& hw1, A8& hw2, A9& hb1,
                    A10& hb2, A11& sig, A12& out, A13& packedOut);
#else
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage1(A0& row, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& gy, A7& hw1, A8& hw2, A9& hb1,
                    A10& hb2, A11& sig);
#endif
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6>
    __aicore__ inline __attribute__((always_inline)) void RunTile(B0& gb1, B1& gb2, B2& gw1, B3& gw2, B4& gx,
                                                                  B5& gy, B6& row);
#endif
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6>
    __aicore__ inline __attribute__((always_inline)) void RunActive(B0& b1, B1& b2, B2& row, B3& w1, B4& w2,
                                                                    B5& x, B6& y);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y);
#endif
private:
    static constexpr uint32_t D = 13U;
    static constexpr uint32_t Rows = 3U;
    static constexpr uint32_t K = 16U;
    static constexpr uint32_t W = 256U;
    static constexpr uint32_t HB = 4U * W + 2U * Rows * K;
    static constexpr uint32_t FB = HB + 4U * K;
    static constexpr uint32_t XB = FB + 8U * W;
    static constexpr uint32_t PB = XB + 4U * Rows * K;
    static constexpr uint32_t AB = PB + 8U * W;
    static constexpr uint32_t GB = AB + 8U * Rows * K;
};
