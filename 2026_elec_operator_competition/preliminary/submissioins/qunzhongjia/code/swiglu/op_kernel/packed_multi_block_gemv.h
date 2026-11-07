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

template <typename T> class PackedMultiBlockGemv {
public:
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 11265UL)
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24,
              class A25, class A26, class A27, class A28, class A29>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage0(A0& hw1, A1& hw2, A2& hxAll, A3& rawB1, A4& rawB2, A5& hb1, A6& hb2, A7& fw1, A8& fw1Last,
                    A9& fw2, A10& fw2Last, A11& fxLast, A12& fx, A13& bothP, A14& blkP, A15& pairP, A16& ag,
                    A17& lastG, A18& lastU, A19& g, A20& u, A21& sig, A22& ones, A23& out, A24& rawW1,
                    A25& rawW2, A26& rawX, A27& offsets, A28& allHalf, A29& allF);
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8,
              class B9, class B10, class B11, class B12, class B13, class B14, class B15, class B16,
              class B17, class B18, class B19, class B20, class B21, class B22, class B23, class B24,
              class B25, class B26, class B27, class B28, class B29, class B30, class B31, class B32,
              class B33, class B34, class B35, class B36, class B37, class B38, class B39>
    __aicore__ inline __attribute__((always_inline)) void
    LoadAndComputeTile(B0& ag, B1& allF, B2& allHalf, B3& bc, B4& blkP, B5& bothP, B6& fw1, B7& fw1Last,
                       B8& fw2, B9& fw2Last, B10& fx, B11& fxLast, B12& g, B13& gb1, B14& gb2, B15& gx,
                       B16& gy, B17& hb1, B18& hb2, B19& hw1, B20& hw2, B21& hxAll, B22& indexGm, B23& lastG,
                       B24& lastU, B25& offsets, B26& ones, B27& out, B28& pairP, B29& rawB1, B30& rawB2,
                       B31& rawW1, B32& rawW2, B33& rawX, B34& row, B35& sig, B36& u, B37& wp, B38& xcFlat,
                       B39& xp);
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8,
              class B9, class B10, class B11, class B12, class B13, class B14, class B15, class B16,
              class B17, class B18, class B19, class B20, class B21, class B22, class B23, class B24,
              class B25, class B26, class B27, class B28, class B29, class B30, class B31, class B32,
              class B33, class B34>
    __aicore__ inline __attribute__((always_inline)) void
    FinishTileLayout(B0& ag, B1& blkP, B2& bothP, B3& fw1, B4& fw1Last, B5& fw2, B6& fw2Last, B7& fx,
                     B8& fxLast, B9& g, B10& gb1, B11& gb2, B12& gw1, B13& gw2, B14& gx, B15& gy, B16& hb1,
                     B17& hb2, B18& hw1, B19& hw2, B20& hxAll, B21& lastG, B22& lastU, B23& ones, B24& out,
                     B25& pairP, B26& rawB1, B27& rawB2, B28& rawW1, B29& rawW2, B30& rawX, B31& row,
                     B32& sig, B33& tiling, B34& u);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24,
              class A25, class A26, class A27>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage2(A0& tiling, A1& row, A2& gx, A3& gw1, A4& gw2, A5& gb1, A6& gb2, A7& gy, A8& hw1, A9& hw2,
                    A10& hxAll, A11& rawB1, A12& rawB2, A13& hb1, A14& hb2, A15& fw1, A16& fw1Last, A17& fw2,
                    A18& fw2Last, A19& fxLast, A20& fx, A21& bothP, A22& blkP, A23& pairP, A24& ag,
                    A25& lastG, A26& lastU, A27& g);
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7>
    __aicore__ inline __attribute__((always_inline)) void RunTile(B0& gb1, B1& gb2, B2& gw1, B3& gw2, B4& gx,
                                                                  B5& gy, B6& row, B7& tiling);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               GM_ADDR tiling);
#else
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24,
              class A25, class A26>
    __aicore__ inline __attribute__((always_inline)) void
    RunStage2f288430(A0& gb1, A1& gb2, A2& hw1, A3& hw2, A4& bothHalfW, A5& halfExpandedX, A6& hwCol32,
                     A7& hxCol32, A8& bothBias, A9& fw1, A10& fw2, A11& bothW, A12& expandedX, A13& bothP,
                     A14& bothBlock, A15& bothPair, A16& acc, A17& gu, A18& ones, A19& rawW1, A20& rawW2,
                     A21& rawX, A22& rawB1, A23& rawB2, A24& offsets, A25& bc, A26& wp);
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20>
    __aicore__ inline __attribute__((always_inline)) void
    RunStagedfff6edb(A0& tiling, A1& blk, A2& row, A3& gx, A4& gw1, A5& gw2, A6& gb1, A7& gb2, A8& gy,
                     A9& hw1, A10& hw2, A11& bothHalfW, A12& halfExpandedX, A13& hwCol32, A14& hxCol32,
                     A15& bothBias, A16& fw1, A17& fw2, A18& bothW, A19& expandedX, A20& bothP);
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8>
    __aicore__ inline __attribute__((always_inline)) void RunTile(B0& blk, B1& gb1, B2& gb2, B3& gw1, B4& gw2,
                                                                  B5& gx, B6& gy, B7& row, B8& tiling);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y,
                               GM_ADDR tiling);
#else
    template <class B0, class B1>
    __aicore__ inline __attribute__((always_inline)) void SharedStageff98d898dd(B0& offsetsI, B1& urI);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <class A0, class A1, class A2>
    __aicore__ inline __attribute__((always_inline)) void RunStage43538853(A0& offsets, A1& offsetsI,
                                                                           A2& offsets64);
#else
    template <class A0, class A1>
    __aicore__ inline __attribute__((always_inline)) void RunStage52d1eeb6(A0& offsetsI, A1& offsets64);
#endif
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24,
              class A25, class A26>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage0(A0& hw1, A1& hw2, A2& rawB1, A3& rawB2, A4& hb1, A5& hb2, A6& fw1, A7& fw1Last, A8& fw2,
                    A9& fw2Last, A10& fxLast, A11& fx, A12& bothP, A13& blkP, A14& pairP, A15& ag, A16& lastG,
                    A17& lastU, A18& g, A19& u, A20& sig, A21& out, A22& rawW1, A23& rawW2, A24& offsets,
                    A25& allHalf, A26& allF);
    template <class B0, class B1, class B2>
    __aicore__ inline __attribute__((always_inline)) void SharedStage32aa9531e9(B0& bothP, B1& offsets64,
                                                                                B2& offsetsI);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage1(A0& gw2, A1& gb1, A2& gb2, A3& hxAll, A4& rawB1, A5& rawB2, A6& bothP, A7& rawW2,
                    A8& rawX, A9& offsets, A10& offsetsI, A11& allHalf, A12& allF, A13& wc, A14& bc, A15& wp);
#else
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage1(A0& gb1, A1& gb2, A2& hxAll, A3& rawB1, A4& rawB2, A5& bothP, A6& rawX, A7& offsets,
                    A8& offsetsI, A9& allHalf, A10& allF, A11& bc, A12& wp);
#endif
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8,
              class B9, class B10, class B11, class B12, class B13, class B14, class B15, class B16,
              class B17, class B18, class B19, class B20, class B21, class B22, class B23, class B24,
              class B25, class B26, class B27, class B28, class B29, class B30, class B31, class B32,
              class B33, class B34, class B35, class B36, class B37, class B38, class B39, class B40,
              class B41>
    __aicore__ inline __attribute__((always_inline)) void
    LoadAndComputeTile(B0& ag, B1& allF, B2& allHalf, B3& bc, B4& blkP, B5& bothP, B6& fw1, B7& fw1Last,
                       B8& fw2, B9& fw2Last, B10& fx, B11& fxLast, B12& g, B13& gb1, B14& gb2, B15& gw1,
                       B16& gw2, B17& gx, B18& gy, B19& hb1, B20& hb2, B21& hw1, B22& hw2, B23& hxAll,
                       B24& lastG, B25& lastU, B26& offsets, B27& offsetsI, B28& out, B29& pairP, B30& rawB1,
                       B31& rawB2, B32& rawW1, B33& rawW2, B34& rawX, B35& row, B36& sig, B37& u, B38& wc,
                       B39& wp, B40& xcFlat, B41& xp);
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6, class B7, class B8,
              class B9, class B10, class B11, class B12, class B13, class B14, class B15, class B16,
              class B17, class B18, class B19, class B20, class B21, class B22, class B23, class B24,
              class B25, class B26, class B27, class B28, class B29, class B30, class B31, class B32>
    __aicore__ inline __attribute__((always_inline)) void
    FinishTileLayout(B0& ag, B1& blkP, B2& bothP, B3& fw1, B4& fw1Last, B5& fw2, B6& fw2Last, B7& fx,
                     B8& fxLast, B9& g, B10& gb1, B11& gb2, B12& gw1, B13& gw2, B14& gx, B15& gy, B16& hb1,
                     B17& hb2, B18& hw1, B19& hw2, B20& hxAll, B21& lastG, B22& lastU, B23& out, B24& pairP,
                     B25& rawB1, B26& rawB2, B27& rawW1, B28& rawW2, B29& rawX, B30& row, B31& sig, B32& u);
#if (defined(TILING_KEY_VAR) && (TILING_KEY_VAR == 12801UL || TILING_KEY_VAR == 13057UL))
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage2(A0& row, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& gy, A7& hw1, A8& hw2, A9& hxAll,
                    A10& rawB1, A11& rawB2, A12& hb1, A13& hb2, A14& fw1, A15& fw1Last, A16& fw2,
                    A17& fw2Last, A18& fxLast, A19& fx, A20& bothP);
#else
    template <class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8,
              class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16,
              class A17, class A18, class A19, class A20, class A21>
    __aicore__ inline __attribute__((always_inline)) void
    RunSharedStage2(A0& row, A1& gx, A2& gw1, A3& gw2, A4& gb1, A5& gb2, A6& gy, A7& hw1, A8& hw2, A9& hxAll,
                    A10& rawB1, A11& rawB2, A12& hb1, A13& hb2, A14& fw1, A15& fw1Last, A16& fw2,
                    A17& fw2Last, A18& fxLast, A19& fx, A20& bothP, A21& blkP);
#endif
    template <class B0, class B1, class B2, class B3, class B4, class B5, class B6>
    __aicore__ inline __attribute__((always_inline)) void RunTile(B0& gb1, B1& gb2, B2& gw1, B3& gw2, B4& gx,
                                                                  B5& gy, B6& row);
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR y);
#endif
#endif
private:
    static constexpr uint32_t D = 33U;
#if (defined(TILING_KEY_VAR) && TILING_KEY_VAR == 12033UL)
    static constexpr uint32_t Rows = 3U;
#else
    static constexpr uint32_t Rows = 5U;
    static constexpr uint32_t K = 48U;
    static constexpr uint32_t W = D * 32U;
    static constexpr uint32_t FB = 7424U;
    static constexpr uint32_t PB = 20032U;
    static constexpr uint32_t BKB = PB + 11264U * 4U;
    static constexpr uint32_t PRB = BKB + 1408U * 4U;
    static constexpr uint32_t AB = PRB + 704U * 4U;
    static constexpr uint32_t LB = AB + 384U * 4U;
    static constexpr uint32_t GB = LB + 384U * 4U;
#endif
};
